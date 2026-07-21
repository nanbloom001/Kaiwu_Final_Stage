# -*- coding: UTF-8 -*-
"""Training-only hard-segment reset sampling and worker-side diagnostics.

The hook replaces only the configured ``reset_base`` event. It first runs the
platform reset function so difficulty sampling and the normal reset pipeline
remain intact, then moves selected environments to a point immediately before
a configured track segment. Evaluation never installs this hook.
"""

from __future__ import annotations

import functools
import inspect
import math
from typing import Any

import torch


START_NAMES = (
    "full_track",
    "stairs_down",
    "slope_down",
    "maze_entry",
    "stairs_up",
)

SEGMENT_START_NAMES = {
    "pyramid_stairs_inv": "stairs_down",
    "pyramid_slope_inv": "slope_down",
    "open_entry_maze": "maze_entry",
    "pyramid_stairs": "stairs_up",
}

_HARD_START_HOOK_CONFIGURED = False


def is_hard_start_hook_configured() -> bool:
    return _HARD_START_HOOK_CONFIGURED


def _log_initialized_state_once(env, state) -> None:
    if bool(getattr(env, "_hard_start_replay_startup_logged", False)):
        return
    counts = state["start_counts"].detach().cpu().tolist()
    print(
        "[Opt5HardStart] initialized in Isaac worker: "
        f"start_counts={dict(zip(START_NAMES, counts))}"
    )
    env._hard_start_replay_startup_logged = True


def _normalized_weights(values: list[float], expected: int) -> torch.Tensor:
    weights = torch.as_tensor(values, dtype=torch.float32)
    if weights.numel() != expected or torch.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError(
            f"hard_start_replay.hard_weights must contain {expected} non-negative values"
        )
    return weights / weights.sum()


def _resolve_track_sequence(config: dict) -> list[str]:
    sequence = config.get("track_sequence", [])
    if not isinstance(sequence, (list, tuple)) or not sequence:
        raise ValueError("hard_start_replay.track_sequence must be a non-empty list")
    return [str(name) for name in sequence]


def _resolve_hard_segments(config: dict) -> list[str]:
    sequence = _resolve_track_sequence(config)
    requested = config.get(
        "hard_segments",
        [
            "pyramid_stairs_inv",
            "pyramid_slope_inv",
            "open_entry_maze",
            "pyramid_stairs",
        ],
    )
    if not isinstance(requested, (list, tuple)) or not requested:
        raise ValueError("hard_start_replay.hard_segments must be a non-empty list")
    requested = [str(name) for name in requested]
    missing = [name for name in requested if name not in sequence]
    unsupported = [name for name in requested if name not in SEGMENT_START_NAMES]
    aliases = [SEGMENT_START_NAMES[name] for name in requested if name in SEGMENT_START_NAMES]
    if missing:
        raise ValueError(f"hard-start segments are absent from track_sequence: {missing}")
    if unsupported:
        raise ValueError(f"hard-start segments have no diagnostic alias: {unsupported}")
    if len(set(aliases)) != len(aliases):
        raise ValueError("hard_start_replay.hard_segments contains duplicate start types")
    return requested


def _segment_rows(config: dict) -> torch.Tensor:
    sequence = _resolve_track_sequence(config)
    requested = _resolve_hard_segments(config)
    return torch.tensor([sequence.index(name) for name in requested], dtype=torch.long)


def _segment_kind_ids(config: dict) -> torch.Tensor:
    requested = _resolve_hard_segments(config)
    return torch.tensor(
        [START_NAMES.index(SEGMENT_START_NAMES[name]) for name in requested],
        dtype=torch.long,
    )


def _sample_start_kinds(num_samples: int, config: dict, device) -> torch.Tensor:
    """Return stable IDs into ``START_NAMES`` for full and hard starts."""
    full_ratio = float(config.get("full_track_ratio", 0.5))
    if not 0.0 <= full_ratio <= 1.0:
        raise ValueError("hard_start_replay.full_track_ratio must be in [0, 1]")

    hard_segments = _resolve_hard_segments(config)
    kinds = torch.zeros(num_samples, dtype=torch.long, device=device)
    hard_mask = torch.rand(num_samples, device=device) >= full_ratio
    hard_count = int(hard_mask.sum().item())
    if hard_count:
        weights = _normalized_weights(
            list(config.get("hard_weights", [])),
            expected=len(hard_segments),
        ).to(device)
        sampled_segments = torch.multinomial(weights, hard_count, replacement=True)
        kind_ids = _segment_kind_ids(config).to(device)
        kinds[hard_mask] = kind_ids[sampled_segments]
    return kinds


def _target_rows_for_kinds(kinds, config: dict, device) -> torch.Tensor:
    rows = _segment_rows(config).to(device)
    kind_ids = _segment_kind_ids(config).to(device)
    target_rows = torch.full_like(kinds, -1)
    for kind_id, row in zip(kind_ids, rows):
        target_rows[kinds == kind_id] = row
    if torch.any(target_rows < 0):
        unknown = torch.unique(kinds[target_rows < 0]).detach().cpu().tolist()
        raise ValueError(f"No hard-start segment row for kind IDs: {unknown}")
    return target_rows


def _quat_from_roll_pitch_yaw(roll, pitch, yaw):
    """Build WXYZ quaternions without importing Isaac modules at import time."""
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=1,
    )


def _resolve_robot(env, asset_cfg):
    asset_name = getattr(asset_cfg, "name", "robot") if asset_cfg is not None else "robot"
    return env.scene[asset_name]


def _state_for_env(env, num_envs: int, device):
    state = getattr(env, "_hard_start_replay_state", None)
    if state is None or state["start_kind"].numel() != num_envs:
        state = {
            "start_kind": torch.zeros(num_envs, dtype=torch.long, device=device),
            "assigned": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "start_counts": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            "episode_counts": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            "success_counts": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            "bad_counts": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            "base_counts": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            "timeout_counts": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            "early_bad_counts": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            "early_base_counts": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            "early_term_counts": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            "spawn_clearance_sum": torch.zeros(len(START_NAMES), device=device),
            "spawn_clearance_count": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            # Retained for compatibility with the old learner-side collector.
            # The debug stage uses the worker-side tensors above as authority.
            "reason_counts": {},
            "surface_query_failures": 0,
            "summary_bucket": -1,
            "spawn_logs": 0,
        }
        env._hard_start_replay_state = state
    return state


def _set_env_origins(env, terrain, env_ids, origins):
    for owner in (terrain, getattr(env, "scene", None)):
        target = getattr(owner, "env_origins", None)
        if torch.is_tensor(target) and target.shape[0] >= env.num_envs:
            target[env_ids] = origins.to(device=target.device, dtype=target.dtype)


def _candidate_surface_meshes(env) -> list[Any]:
    """Return the worker's terrain Warp meshes without importing simulator modules."""
    meshes = []
    seen = set()

    sensors = getattr(getattr(env, "scene", None), "sensors", {})
    scanner = sensors.get("height_scanner") if hasattr(sensors, "get") else None
    if scanner is not None:
        # Accessing data initializes the ray caster and its shared mesh cache.
        try:
            _ = scanner.data
        except Exception:
            pass
        configured_paths = list(getattr(getattr(scanner, "cfg", None), "mesh_prim_paths", []) or [])
        for owner in (scanner, type(scanner)):
            cache = getattr(owner, "meshes", None)
            if not isinstance(cache, dict):
                continue
            candidates = (
                [cache[path] for path in configured_paths if path in cache]
                if configured_paths
                else list(cache.values())
            )
            for mesh in candidates:
                if id(mesh) not in seen:
                    seen.add(id(mesh))
                    meshes.append(mesh)

    terrain = getattr(getattr(env, "scene", None), "terrain", None)
    terrain_meshes = getattr(terrain, "warp_meshes", None)
    if isinstance(terrain_meshes, dict):
        for mesh in terrain_meshes.values():
            if id(mesh) not in seen:
                seen.add(id(mesh))
                meshes.append(mesh)
    return meshes


def _query_terrain_surface_z(env, spawn_xy, config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Ray-cast vertically at spawn XY and return height plus validity mask."""
    fallback_z = float(config.get("track_ground_z", 0.0))
    surface_z = torch.full(
        (spawn_xy.shape[0],),
        fallback_z,
        device=spawn_xy.device,
        dtype=spawn_xy.dtype,
    )
    valid = torch.zeros(spawn_xy.shape[0], dtype=torch.bool, device=spawn_xy.device)
    meshes = _candidate_surface_meshes(env)
    errors = []
    if meshes:
        try:
            from isaaclab.utils.warp import raycast_mesh

            ray_starts = torch.zeros(
                spawn_xy.shape[0], 3, device=spawn_xy.device, dtype=spawn_xy.dtype
            )
            ray_starts[:, :2] = spawn_xy
            ray_starts[:, 2] = float(config.get("surface_query_height_m", 100.0))
            ray_directions = torch.zeros_like(ray_starts)
            ray_directions[:, 2] = -1.0
            hit_heights = []
            for mesh in meshes:
                try:
                    hits = raycast_mesh(ray_starts, ray_directions, mesh)[0]
                    height = hits[:, 2]
                    mesh_valid = torch.isfinite(height) & (torch.abs(height) < 1.0e4)
                    hit_heights.append(
                        torch.where(mesh_valid, height, torch.full_like(height, -torch.inf))
                    )
                except Exception as exc:
                    errors.append(str(exc))
            if hit_heights:
                highest = torch.stack(hit_heights, dim=0).amax(dim=0)
                valid = torch.isfinite(highest)
                surface_z = torch.where(valid, highest, surface_z)
        except Exception as exc:
            errors.append(str(exc))

    if bool(config.get("require_surface_query", False)) and not bool(valid.all().item()):
        invalid_count = int((~valid).sum().item())
        detail = errors[0] if errors else "no terrain Warp mesh was available"
        raise RuntimeError(
            "hard-start terrain surface query failed for "
            f"{invalid_count}/{spawn_xy.shape[0]} spawns: {detail}"
        )
    return surface_z, valid


def _term_mask(term_mgr, name: str, num_envs: int, device) -> torch.Tensor:
    try:
        value = term_mgr.get_term(name)
    except Exception:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return torch.as_tensor(value, device=device).bool().view(-1)


def _record_previous_outcomes(env, env_ids, state, config: dict) -> None:
    """Capture the episode that is about to be reset, inside the Isaac worker."""
    assigned_mask = state["assigned"][env_ids]
    if not bool(assigned_mask.any().item()):
        return
    term_mgr = getattr(env, "termination_manager", None)
    if term_mgr is None:
        return

    num_envs = env.num_envs
    device = env_ids.device
    terminated = torch.as_tensor(
        getattr(term_mgr, "terminated", torch.zeros(num_envs, device=device)),
        device=device,
    ).bool().view(-1)
    timeouts = torch.as_tensor(
        getattr(term_mgr, "time_outs", torch.zeros(num_envs, device=device)),
        device=device,
    ).bool().view(-1)
    done = terminated | timeouts
    record_ids = env_ids[assigned_mask & done[env_ids]]
    if record_ids.numel() == 0:
        return

    active_terms = set(str(name) for name in getattr(term_mgr, "active_terms", []))
    goal = (
        _term_mask(term_mgr, "goal_reached", num_envs, device)
        if "goal_reached" in active_terms
        else torch.zeros(num_envs, dtype=torch.bool, device=device)
    )
    bad = (
        _term_mask(term_mgr, "bad_orientation", num_envs, device)
        if "bad_orientation" in active_terms
        else torch.zeros(num_envs, dtype=torch.bool, device=device)
    )
    base = (
        _term_mask(term_mgr, "base_contact", num_envs, device)
        if "base_contact" in active_terms
        else torch.zeros(num_envs, dtype=torch.bool, device=device)
    )
    kinds = state["start_kind"][record_ids]
    state["episode_counts"] += torch.bincount(kinds, minlength=len(START_NAMES))
    for mask, key in (
        (goal, "success_counts"),
        (bad, "bad_counts"),
        (base, "base_counts"),
        (timeouts, "timeout_counts"),
    ):
        selected = record_ids[mask[record_ids]]
        if selected.numel():
            state[key] += torch.bincount(
                state["start_kind"][selected], minlength=len(START_NAMES)
            )

    episode_steps = getattr(env, "episode_length_buf", None)
    if torch.is_tensor(episode_steps):
        early_limit = int(config.get("early_failure_steps", 20))
        early_ids = record_ids[episode_steps[record_ids].long() <= early_limit]
        if early_ids.numel():
            early_terminated = early_ids[terminated[early_ids] & ~goal[early_ids]]
            if early_terminated.numel():
                state["early_term_counts"] += torch.bincount(
                    state["start_kind"][early_terminated], minlength=len(START_NAMES)
                )
            for mask, key in ((bad, "early_bad_counts"), (base, "early_base_counts")):
                selected = early_ids[mask[early_ids]]
                if selected.numel():
                    state[key] += torch.bincount(
                        state["start_kind"][selected], minlength=len(START_NAMES)
                    )


def _format_named_counts(values) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in zip(START_NAMES, values.detach().cpu().tolist())
    }


def _log_debug_summary_if_due(state, config: dict) -> None:
    if not bool(config.get("debug_logging", False)):
        return
    interval = max(1, int(config.get("summary_interval_resets", 1000)))
    starts = state["start_counts"]
    total = int(starts.sum().item())
    bucket = total // interval
    if bucket <= int(state["summary_bucket"]):
        return
    state["summary_bucket"] = bucket
    hard_total = max(1, int(starts[1:].sum().item()))
    episodes = state["episode_counts"].float()
    success_rate = state["success_counts"].float() / torch.clamp(episodes, min=1.0)
    hard_mix = {
        START_NAMES[index]: round(float(starts[index].item()) / hard_total, 4)
        for index in range(1, len(START_NAMES))
    }
    print(
        "[Opt5DebugSummary] "
        f"resets={total}, actual_full_track_ratio={starts[0].item() / max(1, total):.4f}, "
        f"actual_hard_start_ratio={starts[1:].sum().item() / max(1, total):.4f}, "
        f"hard_mix={hard_mix}, starts={_format_named_counts(starts)}, "
        f"success_rate={dict(zip(START_NAMES, success_rate.detach().cpu().tolist()))}, "
        f"bad_orientation={_format_named_counts(state['bad_counts'])}, "
        f"base_contact={_format_named_counts(state['base_counts'])}, "
        f"timeout={_format_named_counts(state['timeout_counts'])}, "
        f"early_bad={_format_named_counts(state['early_bad_counts'])}, "
        f"early_base={_format_named_counts(state['early_base_counts'])}, "
        f"early_termination={_format_named_counts(state['early_term_counts'])}, "
        f"surface_query_failures={state['surface_query_failures']}"
    )


def _record_spawn_diagnostics(env, env_ids, kinds, diagnostics, state, config: dict) -> None:
    clearance = diagnostics["spawn_z"] - diagnostics["terrain_surface_z"]
    state["spawn_clearance_sum"].index_add_(0, kinds, clearance)
    state["spawn_clearance_count"] += torch.bincount(
        kinds, minlength=len(START_NAMES)
    )
    invalid = ~diagnostics["surface_valid"]
    state["surface_query_failures"] += int(invalid.sum().item())
    if not bool(config.get("debug_logging", False)):
        return

    limit = int(config.get("spawn_sample_log_limit", 16))
    available = max(0, limit - int(state["spawn_logs"]))
    if available <= 0:
        return
    sample_count = min(available, env_ids.numel())
    sequence = _resolve_track_sequence(config)
    for index in range(sample_count):
        row = int(diagnostics["target_rows"][index].item())
        segment = sequence[row] if 0 <= row < len(sequence) else f"row_{row}"
        print(
            "[Opt5DebugSpawn] "
            f"env_id={int(env_ids[index].item())}, "
            f"terrain_level={int(diagnostics['difficulty_cols'][index].item())}, "
            f"track_id={int(diagnostics['difficulty_cols'][index].item())}, "
            f"segment={segment}, spawn_x={diagnostics['spawn_xy'][index, 0].item():.4f}, "
            f"spawn_y={diagnostics['spawn_xy'][index, 1].item():.4f}, "
            f"terrain_surface_z={diagnostics['terrain_surface_z'][index].item():.4f}, "
            f"spawn_z={diagnostics['spawn_z'][index].item():.4f}, "
            f"clearance={clearance[index].item():.4f}, "
            f"spawn_yaw={diagnostics['yaw'][index].item():.4f}, "
            f"entry_speed={diagnostics['speeds'][index].item():.4f}"
        )
    state["spawn_logs"] += sample_count


def _apply_hard_start_poses(env, env_ids, kinds, config: dict, asset_cfg=None):
    terrain = env.scene.terrain
    terrain_origins = getattr(terrain, "terrain_origins", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if not (torch.is_tensor(terrain_origins) and terrain_origins.ndim == 3):
        raise RuntimeError("Track terrain_origins[row, col] is unavailable")
    if not (torch.is_tensor(terrain_types) and torch.is_tensor(terrain_levels)):
        raise RuntimeError("Track terrain_levels/terrain_types are unavailable")

    device = terrain_origins.device
    env_ids = env_ids.to(device=device, dtype=torch.long)
    kinds = kinds.to(device=device, dtype=torch.long)
    if torch.any(kinds <= 0):
        raise ValueError("_apply_hard_start_poses accepts hard-start kinds only")
    target_rows = _target_rows_for_kinds(kinds, config, device)

    difficulty_cols = terrain_types[env_ids].long().clamp(
        min=0, max=terrain_origins.shape[1] - 1
    )
    target_rows = target_rows.clamp(min=1, max=terrain_origins.shape[0] - 1)
    target_origins = terrain_origins[target_rows, difficulty_cols]
    previous_origins = terrain_origins[target_rows - 1, difficulty_cols]
    direction_xy = target_origins[:, :2] - previous_origins[:, :2]

    segment_length = torch.linalg.norm(direction_xy, dim=1).clamp(min=1.0)
    direction_xy = direction_xy / segment_length.unsqueeze(1)
    lateral_xy = torch.stack((-direction_xy[:, 1], direction_xy[:, 0]), dim=1)
    boundary_xy = 0.5 * (previous_origins[:, :2] + target_origins[:, :2])

    offset_range = config.get("approach_offset_m", [0.5, 1.2])
    offset_low, offset_high = float(offset_range[0]), float(offset_range[1])
    approach_offset = offset_low + (offset_high - offset_low) * torch.rand(
        env_ids.numel(), device=device
    )
    lateral_range = config.get("lateral_offset_m", [0.05, 0.08])
    lateral_min, lateral_max = float(lateral_range[0]), float(lateral_range[1])
    lateral_magnitude = lateral_min + (lateral_max - lateral_min) * torch.rand(
        env_ids.numel(), device=device
    )
    lateral_sign = torch.where(
        torch.rand(env_ids.numel(), device=device) < 0.5,
        -torch.ones(env_ids.numel(), device=device),
        torch.ones(env_ids.numel(), device=device),
    )
    spawn_xy = (
        boundary_xy
        - approach_offset.unsqueeze(1) * direction_xy
        + (lateral_sign * lateral_magnitude).unsqueeze(1) * lateral_xy
    )
    surface_z, surface_valid = _query_terrain_surface_z(env, spawn_xy, config)

    robot = _resolve_robot(env, asset_cfg)
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :2] = spawn_xy.to(root_state.dtype)
    root_state[:, 2] = surface_z.to(root_state.dtype) + robot.data.default_root_state[
        env_ids, 2
    ]

    yaw_center = torch.atan2(direction_xy[:, 1], direction_xy[:, 0])
    yaw_range = config.get("yaw_offset_deg", [3.0, 5.0])
    yaw_min = math.radians(float(yaw_range[0]))
    yaw_max = math.radians(float(yaw_range[1]))
    yaw_magnitude = yaw_min + (yaw_max - yaw_min) * torch.rand_like(yaw_center)
    yaw_sign = torch.where(
        torch.rand_like(yaw_center) < 0.5,
        -torch.ones_like(yaw_center),
        torch.ones_like(yaw_center),
    )
    yaw = yaw_center + yaw_sign * yaw_magnitude
    roll_pitch_max = float(config.get("roll_pitch_offset_max_rad", 0.025))
    roll = (2.0 * torch.rand_like(yaw) - 1.0) * roll_pitch_max
    pitch = (2.0 * torch.rand_like(yaw) - 1.0) * roll_pitch_max
    root_state[:, 3:7] = _quat_from_roll_pitch_yaw(roll, pitch, yaw).to(root_state.dtype)

    sequence = _resolve_track_sequence(config)
    entry_speeds = torch.as_tensor(
        config.get("entry_speeds_mps", []), device=device, dtype=root_state.dtype
    )
    if entry_speeds.numel() != len(sequence):
        raise ValueError(
            "entry_speeds_mps must follow track_sequence and contain "
            f"{len(sequence)} values"
        )
    speed_jitter = float(config.get("entry_speed_jitter_mps", 0.04))
    speeds = entry_speeds[target_rows] + (
        2.0 * torch.rand(env_ids.numel(), device=device, dtype=root_state.dtype) - 1.0
    ) * speed_jitter
    speeds = torch.clamp(speeds, min=0.0)
    root_state[:, 7:9] = direction_xy.to(root_state.dtype) * speeds.unsqueeze(1)
    root_state[:, 9:13] = 0.0

    terrain_levels[env_ids] = target_rows.to(terrain_levels.dtype)
    _set_env_origins(env, terrain, env_ids, target_origins)
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    return {
        "difficulty_cols": difficulty_cols,
        "target_rows": target_rows,
        "spawn_xy": spawn_xy,
        "terrain_surface_z": surface_z,
        "surface_valid": surface_valid,
        "spawn_z": root_state[:, 2],
        "yaw": yaw,
        "speeds": speeds,
    }


def _record_start_assignments(env, env_ids, kinds):
    state = _state_for_env(env, env.num_envs, env_ids.device)
    state["start_kind"][env_ids] = kinds
    state["assigned"][env_ids] = True
    state["start_counts"] += torch.bincount(kinds, minlength=len(START_NAMES))
    return state


def _build_reset_wrapper(original, config: dict):
    if getattr(original, "_is_hard_start_replay_wrapper", False):
        return original

    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        bound = signature.bind_partial(*args, **kwargs)
        env = bound.arguments.get("env", args[0] if args else None)
        env_ids = bound.arguments.get("env_ids", args[1] if len(args) > 1 else None)
        asset_cfg = bound.arguments.get("asset_cfg")
        if env is None or bool(getattr(env, "_is_eval", False)):
            return original(*args, **kwargs)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=env.device)
        else:
            env_ids = env_ids.to(env.device)
        env_ids = env_ids.long().view(-1)
        if env_ids.numel() == 0:
            return original(*args, **kwargs)

        state = _state_for_env(env, env.num_envs, env_ids.device)
        _record_previous_outcomes(env, env_ids, state, config)
        result = original(*args, **kwargs)

        kinds = _sample_start_kinds(env_ids.numel(), config, env_ids.device)
        full_ids = env_ids[kinds == 0]
        if full_ids.numel():
            # Reuse the platform's exact full-track evaluation-start semantics.
            full_bound = signature.bind_partial(*args, **kwargs)
            full_bound.arguments["env_ids"] = full_ids
            had_is_eval = hasattr(env, "_is_eval")
            old_is_eval = getattr(env, "_is_eval", False)
            env._is_eval = True
            try:
                original(*full_bound.args, **full_bound.kwargs)
            finally:
                if had_is_eval:
                    env._is_eval = old_is_eval
                else:
                    delattr(env, "_is_eval")

        hard_mask = kinds > 0
        hard_diagnostics = None
        if hard_mask.any():
            hard_diagnostics = _apply_hard_start_poses(
                env,
                env_ids[hard_mask],
                kinds[hard_mask],
                config,
                asset_cfg=asset_cfg,
            )
        state = _record_start_assignments(env, env_ids, kinds)
        if hard_diagnostics is not None:
            _record_spawn_diagnostics(
                env,
                env_ids[hard_mask],
                kinds[hard_mask],
                hard_diagnostics,
                state,
                config,
            )
        _log_debug_summary_if_due(state, config)
        return result

    wrapped._is_hard_start_replay_wrapper = True
    return wrapped


def install_hard_start_replay_event(env_cfg, config: dict) -> bool:
    """Wrap ``env_cfg.events.reset_base`` before EventManager construction."""
    global _HARD_START_HOOK_CONFIGURED
    if not bool(config.get("enabled", False)):
        return False
    events = getattr(env_cfg, "events", None)
    reset_term = getattr(events, "reset_base", None)
    original = getattr(reset_term, "func", None)
    if original is None:
        raise RuntimeError("Cannot install hard-start replay: reset_base event is missing")
    reset_term.func = _build_reset_wrapper(original, config)
    env_cfg._hard_start_replay_config = dict(config)
    _HARD_START_HOOK_CONFIGURED = True
    return True


def initialize_hard_start_replay(env) -> bool:
    """Initialize the first hard-start batch inside the Isaac worker process."""
    if not _HARD_START_HOOK_CONFIGURED or bool(getattr(env, "_is_eval", False)):
        return False
    state = getattr(env, "_hard_start_replay_state", None)
    if state is not None:
        _log_initialized_state_once(env, state)
        return False

    event_manager = getattr(env, "event_manager", None)
    get_term_cfg = getattr(event_manager, "get_term_cfg", None)
    if not callable(get_term_cfg):
        raise RuntimeError("Cannot initialize hard starts: EventManager is unavailable")
    try:
        reset_term = get_term_cfg("reset_base")
    except Exception as exc:
        raise RuntimeError("Cannot initialize hard starts: reset_base is missing") from exc
    reset_func = getattr(reset_term, "func", None)
    if not getattr(reset_func, "_is_hard_start_replay_wrapper", False):
        raise RuntimeError("Cannot initialize hard starts: reset_base wrapper was not preserved")

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    reset_func(env, env_ids, **dict(getattr(reset_term, "params", {}) or {}))
    state = getattr(env, "_hard_start_replay_state", None)
    if state is None:
        raise RuntimeError("Cannot initialize hard starts: reset state was not created")
    _log_initialized_state_once(env, state)
    return True


def publish_hard_start_metrics(env) -> None:
    """Publish authoritative worker-side metrics through Isaac Lab step extras."""
    state = getattr(env, "_hard_start_replay_state", None)
    extras = getattr(env, "extras", None)
    if state is None or not isinstance(extras, dict):
        return
    starts = state["start_counts"].float()
    total = torch.clamp(starts.sum(), min=1.0)
    episodes = state["episode_counts"].float()
    successes = state["success_counts"].float()
    metrics = {
        "full_track_ratio": (starts[0] / total).item(),
        "hard_start_ratio": (starts[1:].sum() / total).item(),
        "stairs_down_starts": starts[1].item(),
        "slope_down_starts": starts[2].item(),
        "maze_entry_starts": starts[3].item(),
        "stairs_up_starts": starts[4].item(),
        "early_bad_total": state["early_bad_counts"].sum().item(),
        "early_base_total": state["early_base_counts"].sum().item(),
        "early_term_total": state["early_term_counts"].sum().item(),
        "surface_query_fail": float(state["surface_query_failures"]),
    }
    for index, name in enumerate(START_NAMES[1:], start=1):
        metrics[f"{name}_success"] = (
            successes[index] / torch.clamp(episodes[index], min=1.0)
        ).item()
        metrics[f"bad_{name}"] = state["bad_counts"][index].item()
        metrics[f"base_{name}"] = state["base_counts"][index].item()
        metrics[f"timeout_{name}"] = state["timeout_counts"][index].item()
    clearance_count = state["spawn_clearance_count"][1:].sum().float()
    metrics["spawn_clearance"] = (
        state["spawn_clearance_sum"][1:].sum()
        / torch.clamp(clearance_count, min=1.0)
    ).item()
    extras["hard_start_metrics"] = metrics
