#!/usr/bin/env python3
"""Training-only P4 full-track spawn scheduling and reset integration."""

from __future__ import annotations

import copy
import functools
import inspect
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch


SEGMENT_LABELS = ("slope", "slope_inv", "stairs", "stairs_inv", "maze")
_TERRAIN_NAMES = {
    "slope": "pyramid_slope",
    "slope_inv": "pyramid_slope_inv",
    "stairs": "pyramid_stairs",
    "stairs_inv": "pyramid_stairs_inv",
    "maze": "open_entry_maze",
}
_STATE_ATTR = "_agent_ppo_p4_full_track_spawn"


@dataclass(frozen=True)
class SpawnAssignments:
    full_start: torch.Tensor
    segment: torch.Tensor
    quartile: torch.Tensor
    safe_point: torch.Tensor
    reason4_retry: torch.Tensor
    reason4_exhausted: torch.Tensor


def _normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    supplied = dict(config or {})
    result = {
        "enabled": bool(supplied.get("enabled", False)),
        "phase_boundaries_s": tuple(
            float(value)
            for value in supplied.get(
                "phase_boundaries_s", supplied.get("phase_boundaries", (7200.0, 21600.0))
            )
        ),
        "full_start_share": tuple(
            float(value) for value in supplied.get("full_start_share", (0.70, 0.60, 0.75))
        ),
        "random_segment_weights": tuple(
            float(value)
            for value in supplied.get(
                "random_segment_weights", (0.15, 0.15, 0.20, 0.20, 0.30)
            )
        ),
        "safe_point_share": float(supplied.get("safe_point_share", 0.70)),
        "quartiles": int(supplied.get("quartiles", 4)),
        "slope_yaw_abs": float(supplied.get("slope_yaw_abs", 0.35)),
        "maze_yaw_abs": float(supplied.get("maze_yaw_abs", 0.50)),
        "max_reason4_same_bucket_retries": int(
            supplied.get("max_reason4_same_bucket_retries", 2)
        ),
        "seed": int(supplied.get("seed", 0)),
        "resume_offset_s": float(supplied.get("resume_offset_s", 0.0)),
    }
    if len(result["phase_boundaries_s"]) + 1 != len(result["full_start_share"]):
        raise ValueError("full_track_spawn full_start_share must have one entry per phase")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in result["phase_boundaries_s"]
    ) or tuple(sorted(result["phase_boundaries_s"])) != result["phase_boundaries_s"]:
        raise ValueError("full_track_spawn phase boundaries must be finite and sorted")
    if any(not 0.0 <= value <= 1.0 for value in result["full_start_share"]):
        raise ValueError("full_track_spawn full_start_share values must be in [0, 1]")
    weights = result["random_segment_weights"]
    if len(weights) != len(SEGMENT_LABELS) or any(value < 0.0 for value in weights):
        raise ValueError("full_track_spawn needs five non-negative segment weights")
    weight_sum = sum(weights)
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("full_track_spawn segment weights must have a positive sum")
    result["random_segment_weights"] = tuple(value / weight_sum for value in weights)
    if not 0.0 <= result["safe_point_share"] <= 1.0:
        raise ValueError("full_track_spawn safe_point_share must be in [0, 1]")
    if result["quartiles"] <= 0:
        raise ValueError("full_track_spawn quartiles must be positive")
    if result["slope_yaw_abs"] < 0.0 or result["maze_yaw_abs"] < 0.0:
        raise ValueError("full_track_spawn yaw limits must be non-negative")
    if result["max_reason4_same_bucket_retries"] < 0:
        raise ValueError("full_track_spawn retry count must be non-negative")
    return result


class P4SpawnQuotaState:
    """Deterministic quota sampler with per-environment reason-4 stickiness."""

    def __init__(
        self,
        num_envs: int,
        config: dict[str, Any] | None,
        *,
        seed: int = 0,
    ) -> None:
        self.config = _normalize_config(config)
        self.num_envs = int(num_envs)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(self.config["seed"] or seed))
        phase_count = len(self.config["full_start_share"])
        self.phase_total = torch.zeros(phase_count, dtype=torch.long)
        self.phase_full = torch.zeros(phase_count, dtype=torch.long)
        self.segment_counts = torch.zeros(phase_count, len(SEGMENT_LABELS), dtype=torch.long)
        self.quartile_counts = torch.zeros(
            phase_count, len(SEGMENT_LABELS), self.config["quartiles"], dtype=torch.long
        )
        self.safe_counts = torch.zeros(phase_count, 2, dtype=torch.long)
        # The wrapper may be installed after the platform's initial reset.  Do
        # not claim that unseen initial episodes were full-track starts.
        self.last_full = torch.zeros(self.num_envs, dtype=torch.bool)
        self.last_segment = torch.full((self.num_envs,), -1, dtype=torch.long)
        self.last_quartile = torch.full((self.num_envs,), -1, dtype=torch.long)
        self.last_safe = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reason4_retries = torch.zeros(self.num_envs, dtype=torch.long)

    def phase_index(self, elapsed_s: float) -> int:
        value = max(0.0, float(elapsed_s))
        return sum(value >= boundary for boundary in self.config["phase_boundaries_s"])

    def _quota_choice(self, counts: torch.Tensor, weights: torch.Tensor) -> int:
        target = weights * float(int(counts.sum()) + 1)
        deficit = target - counts.to(torch.float64)
        maximum = deficit.max()
        candidates = torch.nonzero(
            torch.isclose(deficit, maximum, atol=1.0e-12, rtol=0.0), as_tuple=False
        ).reshape(-1)
        if candidates.numel() == 1:
            return int(candidates.item())
        pick = int(torch.randint(candidates.numel(), (1,), generator=self.generator).item())
        return int(candidates[pick].item())

    def assign(
        self,
        env_ids: torch.Tensor,
        *,
        elapsed_s: float,
        terminal_reason: torch.Tensor | None = None,
    ) -> SpawnAssignments:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu").reshape(-1)
        if ids.numel() == 0:
            empty_long = torch.empty(0, dtype=torch.long)
            empty_bool = torch.empty(0, dtype=torch.bool)
            return SpawnAssignments(
                empty_bool,
                empty_long,
                empty_long,
                empty_bool,
                empty_bool,
                empty_bool,
            )
        if bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise IndexError("P4 spawn env_ids are outside the configured environment count")
        if terminal_reason is None:
            reasons = torch.zeros(ids.numel(), dtype=torch.long)
        else:
            reasons = torch.as_tensor(terminal_reason, device="cpu").reshape(-1).round().long()
            if reasons.numel() != ids.numel():
                raise ValueError("P4 spawn terminal_reason must align with env_ids")

        phase = self.phase_index(elapsed_s)
        full = torch.zeros(ids.numel(), dtype=torch.bool)
        segment = torch.full((ids.numel(),), -1, dtype=torch.long)
        quartile = torch.full_like(segment, -1)
        safe = torch.ones(ids.numel(), dtype=torch.bool)
        retries = torch.zeros(ids.numel(), dtype=torch.bool)
        exhausted = torch.zeros(ids.numel(), dtype=torch.bool)
        full_weights = torch.tensor(
            (1.0 - self.config["full_start_share"][phase], self.config["full_start_share"][phase]),
            dtype=torch.float64,
        )
        segment_weights = torch.tensor(self.config["random_segment_weights"], dtype=torch.float64)
        quartile_weights = torch.full(
            (self.config["quartiles"],), 1.0 / self.config["quartiles"], dtype=torch.float64
        )
        safe_weights = torch.tensor(
            (1.0 - self.config["safe_point_share"], self.config["safe_point_share"]),
            dtype=torch.float64,
        )

        for output_index, env_id_tensor in enumerate(ids):
            env_id = int(env_id_tensor.item())
            reason4 = reasons[output_index].item() == 4
            has_previous_bucket = bool(self.last_full[env_id]) or (
                self.last_segment[env_id].item() >= 0
            )
            sticky = (
                reason4
                and has_previous_bucket
                and self.reason4_retries[env_id].item()
                < self.config["max_reason4_same_bucket_retries"]
            )
            if sticky:
                full[output_index] = self.last_full[env_id]
                segment[output_index] = self.last_segment[env_id]
                quartile[output_index] = self.last_quartile[env_id]
                safe[output_index] = self.last_safe[env_id]
                retries[output_index] = True
                self.reason4_retries[env_id] += 1
                continue

            if reason4 and has_previous_bucket:
                # Retry exhaustion never advances the robot toward the goal.
                # Fall back to the entry of the same segment so completion
                # statistics cannot be inflated by an easier later spawn.
                full[output_index] = self.last_full[env_id]
                if bool(self.last_full[env_id]):
                    segment[output_index] = 0
                else:
                    segment[output_index] = self.last_segment[env_id]
                quartile[output_index] = 0
                safe[output_index] = True
                self.last_quartile[env_id] = 0
                self.last_safe[env_id] = True
                self.reason4_retries[env_id] = 0
                exhausted[output_index] = True
                continue

            self.reason4_retries[env_id] = 0
            category_counts = torch.stack(
                (self.phase_total[phase] - self.phase_full[phase], self.phase_full[phase])
            )
            use_full = self._quota_choice(category_counts, full_weights) == 1
            full[output_index] = use_full
            self.phase_total[phase] += 1
            if use_full:
                self.phase_full[phase] += 1
                self.last_full[env_id] = True
                self.last_segment[env_id] = -1
                self.last_quartile[env_id] = -1
                self.last_safe[env_id] = True
                continue

            segment_index = self._quota_choice(self.segment_counts[phase], segment_weights)
            quartile_index = self._quota_choice(
                self.quartile_counts[phase, segment_index], quartile_weights
            )
            safe_index = self._quota_choice(self.safe_counts[phase], safe_weights)
            segment[output_index] = segment_index
            quartile[output_index] = quartile_index
            safe[output_index] = safe_index == 1
            self.segment_counts[phase, segment_index] += 1
            self.quartile_counts[phase, segment_index, quartile_index] += 1
            self.safe_counts[phase, safe_index] += 1
            self.last_full[env_id] = False
            self.last_segment[env_id] = segment_index
            self.last_quartile[env_id] = quartile_index
            self.last_safe[env_id] = safe[output_index]

        return SpawnAssignments(full, segment, quartile, safe, retries, exhausted)

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator_state": self.generator.get_state().clone(),
            "phase_total": self.phase_total.clone(),
            "phase_full": self.phase_full.clone(),
            "segment_counts": self.segment_counts.clone(),
            "quartile_counts": self.quartile_counts.clone(),
            "safe_counts": self.safe_counts.clone(),
            "last_full": self.last_full.clone(),
            "last_segment": self.last_segment.clone(),
            "last_quartile": self.last_quartile.clone(),
            "last_safe": self.last_safe.clone(),
            "reason4_retries": self.reason4_retries.clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in (
            "phase_total", "phase_full", "segment_counts", "quartile_counts", "safe_counts",
            "last_full", "last_segment", "last_quartile", "last_safe", "reason4_retries",
        ):
            source = torch.as_tensor(state[name], device="cpu")
            target = getattr(self, name)
            if source.shape != target.shape:
                raise ValueError(f"P4 spawn state shape mismatch for {name}")
            target.copy_(source.to(dtype=target.dtype))
        generator_state = torch.as_tensor(state["generator_state"], device="cpu", dtype=torch.uint8)
        self.generator.set_state(generator_state)


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


class P4FullTrackSpawnController:
    """Wrap the public reset term while retaining its pose/velocity writer contract."""

    def __init__(self, env, config: dict[str, Any] | None, *, seed: int = 0) -> None:
        self.env = env
        self.config = _normalize_config(config)
        self.num_envs = int(getattr(env, "num_envs", 0) or self._robot().data.default_root_state.shape[0])
        self.device = self._robot().data.default_root_state.device
        self.quota = P4SpawnQuotaState(self.num_envs, self.config, seed=seed)
        self.started_monotonic = time.monotonic()
        self.installed = False
        self.diagnostic_counts = {
            "reset_batches": 0,
            "full_start_count": 0,
            "segment_start_count": 0,
            "reason4_retry_count": 0,
            "reason4_exhausted_count": 0,
            "reason4_fallback_applied_count": 0,
            "safe_point_count": 0,
            "all_position_requested_count": 0,
            "all_position_applied_count": 0,
            "segment_entry_fallback_count": 0,
            "spawn_validation_failure_count": 0,
            "spawn_write_failure_count": 0,
        }
        self.last_status = "not_installed"
        self._raycast_mesh = None
        self._raycast_status = "not_checked"

    def _robot(self, asset_cfg=None):
        name = getattr(asset_cfg, "name", "robot") if asset_cfg is not None else "robot"
        scene = getattr(self.env, "scene", None)
        try:
            return scene[name]
        except Exception:
            robot = getattr(scene, name, None)
            if robot is None:
                raise RuntimeError("P4 full-track spawn requires the robot scene asset")
            return robot

    def elapsed_s(self) -> float:
        live = getattr(self.env, "_agent_ppo_p4_session_seconds", None)
        if live is not None:
            try:
                value = float(live)
                if math.isfinite(value) and value >= 0.0:
                    return value
            except (TypeError, ValueError):
                pass
        return self.config["resume_offset_s"] + max(0.0, time.monotonic() - self.started_monotonic)

    def _terminal_reasons(self, env_ids: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(env_ids.numel(), dtype=torch.long)
        manager = getattr(self.env, "termination_manager", None)
        getter = getattr(manager, "get_term", None)
        if not callable(getter):
            return result
        try:
            wall = getter("nav_stuck_timeout")
        except Exception:
            return result
        if not torch.is_tensor(wall) or wall.numel() != self.num_envs:
            return result
        wall = wall.to(self.device).reshape(-1).bool()[env_ids]
        result[wall.cpu()] = 4
        return result

    def retry_reason4(
        self, env_ids: torch.Tensor, terminal_reason: torch.Tensor
    ) -> SpawnAssignments:
        """Public sticky-retry API used by the reset wrapper and CPU tests."""
        return self.quota.assign(
            env_ids,
            elapsed_s=self.elapsed_s(),
            terminal_reason=terminal_reason,
        )

    def _terrain_rows(self) -> tuple[dict[int, int], str]:
        terrain = getattr(getattr(self.env, "scene", None), "terrain", None)
        origins = getattr(terrain, "terrain_origins", None)
        if not torch.is_tensor(origins) or origins.ndim != 3 or origins.shape[0] < 5:
            return {}, "terrain_origins_missing"
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        order = getattr(generator, "sub_terrains_order", None)
        if not order:
            order = list(getattr(generator, "sub_terrains", {}) or {})
        if not order:
            return {}, "track_sequence_missing"
        expanded = list(order)
        while len(expanded) < origins.shape[0]:
            expanded.extend(order)
        expanded = expanded[: origins.shape[0]]
        rows = {}
        for segment_index, label in enumerate(SEGMENT_LABELS):
            expected = _TERRAIN_NAMES[label]
            matches = [index for index, name in enumerate(expanded) if str(name) == expected]
            if len(matches) != 1:
                return {}, f"segment_mapping_{label}_{len(matches)}"
            rows[segment_index] = matches[0]
        return rows, "valid"

    def _surface_query(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        callback = getattr(self.env, "_agent_ppo_p4_spawn_surface_query", None)
        if callable(callback):
            try:
                height, valid = callback(xy)
                height = torch.as_tensor(height, device=xy.device, dtype=xy.dtype).reshape(-1)
                valid = torch.as_tensor(valid, device=xy.device).reshape(-1).bool()
                if height.numel() == xy.shape[0] and valid.numel() == xy.shape[0]:
                    return height, valid & torch.isfinite(height)
            except Exception:
                pass
        if self._raycast_status == "not_checked":
            self._raycast_status = "unavailable"
            scene = getattr(self.env, "scene", None)
            sensors = getattr(scene, "sensors", None)
            if sensors is None:
                sensors = getattr(scene, "_sensors", None)
            candidates = []
            if isinstance(sensors, dict):
                candidates.extend(sensors.values())
            elif sensors is not None:
                try:
                    candidates.extend(sensors.values())
                except (AttributeError, TypeError):
                    pass
            for sensor in candidates:
                meshes = getattr(sensor, "meshes", None)
                if isinstance(meshes, dict) and meshes:
                    self._raycast_mesh = next(iter(meshes.values()))
                    self._raycast_status = "available"
                    break
        if self._raycast_mesh is None:
            return torch.zeros(xy.shape[0], device=xy.device, dtype=xy.dtype), torch.zeros(
                xy.shape[0], dtype=torch.bool, device=xy.device
            )
        try:
            from isaaclab.utils.warp.ops import raycast_mesh

            vertical_starts = torch.cat(
                (
                    xy,
                    torch.full(
                        (xy.shape[0], 1),
                        8.0,
                        device=xy.device,
                        dtype=xy.dtype,
                    ),
                ),
                dim=-1,
            )
            vertical_directions = torch.zeros_like(vertical_starts)
            vertical_directions[:, 2] = -1.0
            vertical_hits = raycast_mesh(
                vertical_starts,
                vertical_directions,
                self._raycast_mesh,
                max_dist=16.0,
            )[0]
            vertical_valid = torch.isfinite(vertical_hits).all(dim=-1)
            height = torch.nan_to_num(vertical_hits[:, 2])

            angles = torch.arange(8, device=xy.device, dtype=xy.dtype) * (
                2.0 * math.pi / 8.0
            )
            directions_xy = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
            horizontal_starts = torch.cat(
                (
                    xy.unsqueeze(1).expand(-1, 8, -1),
                    (height + 0.22).reshape(-1, 1, 1).expand(-1, 8, 1),
                ),
                dim=-1,
            )
            horizontal_directions = torch.cat(
                (
                    directions_xy.unsqueeze(0).expand(xy.shape[0], -1, -1),
                    torch.zeros(xy.shape[0], 8, 1, device=xy.device, dtype=xy.dtype),
                ),
                dim=-1,
            )
            horizontal_hits = raycast_mesh(
                horizontal_starts.reshape(-1, 3),
                horizontal_directions.reshape(-1, 3),
                self._raycast_mesh,
                max_dist=0.35,
                return_distance=True,
            )
            distances = horizontal_hits[1]
            if distances is None:
                clearance_valid = torch.zeros_like(vertical_valid)
            else:
                clearance_valid = ~torch.isfinite(distances.reshape(-1, 8)).any(dim=-1)
            return height, vertical_valid & clearance_valid
        except Exception:
            self._raycast_status = "runtime_failed"
            return torch.zeros(xy.shape[0], device=xy.device, dtype=xy.dtype), torch.zeros(
                xy.shape[0], dtype=torch.bool, device=xy.device
            )

    def _safe_map_point(
        self, row: int, col: int, quartile: int, origin: torch.Tensor, size_x: float
    ) -> torch.Tensor | None:
        terrain = getattr(getattr(self.env, "scene", None), "terrain", None)
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        mapping = getattr(generator, "terrain_spawn_positions", None)
        if not isinstance(mapping, dict):
            return None
        values = mapping.get((row, col))
        if not values:
            return None
        points = torch.as_tensor(values, device=self.device, dtype=torch.float32)
        if points.ndim != 2 or points.shape[1] < 2 or not bool(torch.isfinite(points[:, :2]).all()):
            return None
        lower = origin[0] - 0.5 * size_x + quartile * size_x / self.config["quartiles"]
        upper = lower + size_x / self.config["quartiles"]
        bucket = points[(points[:, 0] >= lower) & (points[:, 0] <= upper)]
        if bucket.numel() == 0:
            bucket = points
        pick = int(torch.randint(bucket.shape[0], (1,), generator=self.quota.generator).item())
        return bucket[pick, :2]

    def _apply_segment_spawns(
        self,
        env_ids: torch.Tensor,
        assignments: SpawnAssignments,
        *,
        asset_cfg=None,
    ) -> None:
        if env_ids.numel() == 0:
            return
        terrain = getattr(getattr(self.env, "scene", None), "terrain", None)
        origins = getattr(terrain, "terrain_origins", None)
        types = getattr(terrain, "terrain_types", None)
        levels = getattr(terrain, "terrain_levels", None)
        rows, status = self._terrain_rows()
        valid_runtime = (
            bool(rows)
            and torch.is_tensor(types)
            and types.numel() == self.num_envs
            and torch.is_tensor(levels)
            and levels.numel() == self.num_envs
        )
        if not valid_runtime:
            self.last_status = status if not rows else "terrain_indices_missing"
            self.diagnostic_counts["spawn_validation_failure_count"] += int(env_ids.numel())
            return
        robot = self._robot(asset_cfg)
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        size = getattr(generator, "size", (0.0, 0.0))
        try:
            size_x, size_y = float(size[0]), float(size[1])
        except (IndexError, TypeError, ValueError):
            size_x, size_y = 0.0, 0.0
        if not (math.isfinite(size_x) and size_x > 0.0 and math.isfinite(size_y) and size_y > 0.0):
            self.last_status = "terrain_size_invalid"
            self.diagnostic_counts["spawn_validation_failure_count"] += int(env_ids.numel())
            return

        cols = types[env_ids].long().clamp(0, origins.shape[1] - 1)
        target_rows = torch.tensor(
            [rows[int(value)] for value in assignments.segment.tolist()],
            device=self.device,
            dtype=torch.long,
        )
        target_origins = origins[target_rows, cols]
        spawn_xy = target_origins[:, :2].clone()
        candidate_valid = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
        all_position = ~assignments.safe_point.to(self.device)
        self.diagnostic_counts["all_position_requested_count"] += int(all_position.sum())

        for index in range(env_ids.numel()):
            segment_index = int(assignments.segment[index])
            quartile = int(assignments.quartile[index])
            row = int(target_rows[index])
            col = int(cols[index])
            origin = target_origins[index]
            if bool(assignments.safe_point[index]):
                mapped = self._safe_map_point(row, col, quartile, origin, size_x)
                if mapped is not None:
                    spawn_xy[index] = mapped
                    candidate_valid[index] = True
                    continue
                local_x = (-0.5 + (quartile + 0.5) / self.config["quartiles"]) * size_x
                spawn_xy[index, 0] = origin[0] + local_x
                spawn_xy[index, 1] = origin[1]
            else:
                bucket_low = -0.5 + quartile / self.config["quartiles"]
                bucket_high = -0.5 + (quartile + 1) / self.config["quartiles"]
                random_values = torch.rand(2, generator=self.quota.generator)
                spawn_xy[index, 0] = origin[0] + (
                    bucket_low + (bucket_high - bucket_low) * float(random_values[0])
                ) * size_x * 0.90
                spawn_xy[index, 1] = origin[1] + (float(random_values[1]) - 0.5) * size_y * 0.50
            if SEGMENT_LABELS[segment_index] == "maze" and not bool(assignments.safe_point[index]):
                candidate_valid[index] = False

        surface_z, surface_valid = self._surface_query(spawn_xy)
        # The fallback ray query includes both a vertical terrain hit and an
        # eight-direction body-clearance test, so a validated maze point is as
        # usable as a validated open-terrain point.  If raycasting is absent or
        # malformed, ``surface_valid`` stays false and the code fails closed to
        # the segment entry.
        candidate_valid |= surface_valid
        validator = getattr(self.env, "_agent_ppo_p4_spawn_validator", None)
        if callable(validator):
            try:
                validated = torch.as_tensor(
                    validator(spawn_xy, assignments.segment.to(self.device)),
                    device=self.device,
                ).reshape(-1).bool()
                if validated.numel() == env_ids.numel():
                    candidate_valid &= validated
                    candidate_valid |= all_position & validated & surface_valid
            except Exception:
                candidate_valid.zero_()
        finite_xy = torch.isfinite(spawn_xy).all(dim=-1)
        candidate_valid &= finite_xy
        fallback = ~candidate_valid
        spawn_xy[fallback] = target_origins[fallback, :2]
        self.diagnostic_counts["segment_entry_fallback_count"] += int(fallback.sum())
        self.diagnostic_counts["spawn_validation_failure_count"] += int(fallback.sum())
        self.diagnostic_counts["all_position_applied_count"] += int(
            (all_position & candidate_valid).sum()
        )

        root_state = robot.data.default_root_state[env_ids].clone()
        root_state[:, :2] = spawn_xy.to(root_state)
        root_state[:, 2] += target_origins[:, 2].to(root_state)
        root_state[candidate_valid & surface_valid, 2] = (
            surface_z[candidate_valid & surface_valid].to(root_state)
            + robot.data.default_root_state[env_ids[candidate_valid & surface_valid], 2]
        )
        yaw_abs = torch.tensor(
            [
                self.config["maze_yaw_abs"]
                if SEGMENT_LABELS[int(value)] == "maze"
                else self.config["slope_yaw_abs"]
                for value in assignments.segment.tolist()
            ],
            device=self.device,
            dtype=root_state.dtype,
        )
        yaw_random = 2.0 * torch.rand(env_ids.numel(), generator=self.quota.generator) - 1.0
        yaw = yaw_random.to(self.device, dtype=root_state.dtype) * yaw_abs
        root_state[:, 3:7] = _quat_multiply(
            root_state[:, 3:7], _yaw_quaternion(yaw)
        )
        root_state[:, 7:13] = robot.data.default_root_state[env_ids, 7:13]

        try:
            levels[env_ids] = target_rows.to(levels)
            for owner in (terrain, getattr(self.env, "scene", None)):
                env_origins = getattr(owner, "env_origins", None)
                if torch.is_tensor(env_origins) and env_origins.shape[0] >= self.num_envs:
                    env_origins[env_ids] = target_origins.to(env_origins)
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
            self.last_status = "applied"
        except Exception as exc:
            self.diagnostic_counts["spawn_write_failure_count"] += int(env_ids.numel())
            self.last_status = "physical_writer_failed"
            raise RuntimeError(
                "P4 full-track spawn could not write the validated root state"
            ) from exc

    def _publish(self, assignments: SpawnAssignments) -> None:
        self.diagnostic_counts["reset_batches"] += 1
        self.diagnostic_counts["full_start_count"] += int(assignments.full_start.sum())
        self.diagnostic_counts["segment_start_count"] += int((~assignments.full_start).sum())
        self.diagnostic_counts["reason4_retry_count"] += int(assignments.reason4_retry.sum())
        self.diagnostic_counts["reason4_exhausted_count"] += int(
            assignments.reason4_exhausted.sum()
        )
        self.diagnostic_counts["safe_point_count"] += int(assignments.safe_point.sum())
        diagnostics = self.diagnostics()
        setattr(self.env, "_agent_ppo_p4_spawn_diagnostics", diagnostics)
        extras = getattr(self.env, "extras", None)
        if isinstance(extras, dict):
            extras["p4_full_track_spawn"] = diagnostics

    def diagnostics(self) -> dict[str, Any]:
        result = dict(self.diagnostic_counts)
        result.update(
            {
                "installed": int(self.installed),
                "status": self.last_status,
                "raycast_status": self._raycast_status,
                "all_position_spawn_active": int(
                    self.diagnostic_counts["all_position_applied_count"] > 0
                ),
                "phase_index": self.quota.phase_index(self.elapsed_s()),
                "segment_labels": SEGMENT_LABELS,
                "segment_counts": self.quota.segment_counts.clone(),
                "quartile_counts": self.quota.quartile_counts.clone(),
            }
        )
        return result

    def state_dict(self) -> dict[str, Any]:
        """Return exact quota/RNG/retry state for an optional worker checkpoint."""
        return {
            "version": "p4_full_track_spawn_v1",
            "quota": self.quota.state_dict(),
            "diagnostic_counts": dict(self.diagnostic_counts),
            "elapsed_s": self.elapsed_s(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != "p4_full_track_spawn_v1":
            raise ValueError("unsupported P4 full-track spawn state version")
        self.quota.load_state_dict(state["quota"])
        counts = state.get("diagnostic_counts", {})
        if set(counts) != set(self.diagnostic_counts):
            raise ValueError("P4 full-track spawn diagnostic state is incomplete")
        self.diagnostic_counts.update({name: int(value) for name, value in counts.items()})
        elapsed_s = float(state.get("elapsed_s", 0.0))
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValueError("P4 full-track spawn elapsed state must be finite")
        self.config["resume_offset_s"] = elapsed_s
        self.started_monotonic = time.monotonic()

    def _wrap_reset(self, original: Callable[..., Any]) -> Callable[..., Any]:
        if bool(getattr(original, "_is_p4_full_track_spawn_wrapper", False)):
            return original
        signature = inspect.signature(original)
        controller = self

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any):
            bound = signature.bind_partial(*args, **kwargs)
            env = bound.arguments.get("env", args[0] if args else controller.env)
            env_ids = bound.arguments.get("env_ids", args[1] if len(args) > 1 else None)
            if env is None or bool(getattr(env, "_is_eval", False)):
                return original(*args, **kwargs)
            if env_ids is None:
                env_ids = torch.arange(controller.num_envs, device=controller.device)
            env_ids = torch.as_tensor(env_ids, device=controller.device, dtype=torch.long).reshape(-1)
            if env_ids.numel() == 0:
                return original(*args, **kwargs)
            reasons = controller._terminal_reasons(env_ids)
            result = original(*args, **kwargs)
            assignments = controller.retry_reason4(env_ids.cpu(), reasons)
            full_ids = env_ids[assignments.full_start.to(env_ids.device)]
            if full_ids.numel():
                full_bound = signature.bind_partial(*args, **kwargs)
                full_bound.arguments["env_ids"] = full_ids
                previous_eval = bool(getattr(env, "_is_eval", False))
                env._is_eval = True
                try:
                    original(*full_bound.args, **full_bound.kwargs)
                finally:
                    env._is_eval = previous_eval
            segment_mask = ~assignments.full_start
            if bool(segment_mask.any()):
                controller._apply_segment_spawns(
                    env_ids[segment_mask.to(env_ids.device)],
                    SpawnAssignments(
                        assignments.full_start[segment_mask],
                        assignments.segment[segment_mask],
                        assignments.quartile[segment_mask],
                        assignments.safe_point[segment_mask],
                        assignments.reason4_retry[segment_mask],
                        assignments.reason4_exhausted[segment_mask],
                    ),
                    asset_cfg=bound.arguments.get("asset_cfg"),
                )
            exhausted_full = assignments.reason4_exhausted & assignments.full_start
            exhausted_segment = assignments.reason4_exhausted & ~assignments.full_start
            fallback_applied = int(exhausted_full.sum())
            if controller.last_status == "applied":
                fallback_applied += int(exhausted_segment.sum())
            controller.diagnostic_counts["reason4_fallback_applied_count"] += (
                fallback_applied
            )
            controller._publish(assignments)
            return result

        wrapped._is_p4_full_track_spawn_wrapper = True
        return wrapped

    def install(self) -> bool:
        if not self.config["enabled"] or bool(getattr(self.env, "_is_eval", False)):
            self.last_status = "disabled_or_eval"
            return False
        manager = getattr(self.env, "event_manager", None)
        getter = getattr(manager, "get_term_cfg", None)
        setter = getattr(manager, "set_term_cfg", None)
        if not callable(getter) or not callable(setter):
            self.last_status = "reset_api_unavailable"
            return False
        try:
            cfg = copy.deepcopy(getter("reset_base"))
            original = getattr(cfg, "func", None)
            if not callable(original):
                self.last_status = "reset_function_missing"
                return False
            cfg.func = self._wrap_reset(original)
            setter("reset_base", cfg)
            readback = getter("reset_base")
            self.installed = bool(
                getattr(getattr(readback, "func", None), "_is_p4_full_track_spawn_wrapper", False)
            )
            self.last_status = "installed" if self.installed else "reset_readback_failed"
        except Exception as exc:
            self.last_status = f"install_failed:{type(exc).__name__}"
            self.installed = False
        return self.installed


def install_p4_full_track_spawn(
    env, config: dict[str, Any] | None, *, seed: int = 0
) -> P4FullTrackSpawnController | None:
    normalized = _normalize_config(config)
    if not normalized["enabled"] or bool(getattr(env, "_is_eval", False)):
        return None
    existing = getattr(env, _STATE_ATTR, None)
    if isinstance(existing, P4FullTrackSpawnController):
        return existing
    controller = P4FullTrackSpawnController(env, normalized, seed=seed)
    controller.install()
    setattr(env, _STATE_ATTR, controller)
    return controller


__all__ = [
    "P4FullTrackSpawnController",
    "P4SpawnQuotaState",
    "SEGMENT_LABELS",
    "SpawnAssignments",
    "install_p4_full_track_spawn",
]
