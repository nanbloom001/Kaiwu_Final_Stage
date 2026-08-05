#!/usr/bin/env python3
"""Windowed P2 gait diagnostics and parent-baseline protection."""

from __future__ import annotations

import math
import sys

import torch

from agent_ppo.feature import p2_contract


def _warn(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class P2GaitWindowProbe:
    """Compute per-environment, per-leg statistics over a fixed 1.5s window."""

    def __init__(self, env, robot, *, num_envs: int, device):
        self.env = env
        self.robot = robot
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.window_frames = max(
            1,
            int(round(p2_contract.GAIT_WINDOW_SECONDS / p2_contract.CONTROL_DT_S)),
        )
        self.index = 0
        self.count = 0
        self.env_counts = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.valid = False
        self.collision_valid = False
        self.invalid_reason = "unresolved"
        self.collision_invalid_reason = "unresolved"
        self.robot_foot_ids = None
        self.sensor_foot_ids = None
        self.robot_foot_names: tuple[str, ...] = ()
        self.sensor_body_names: tuple[str, ...] = ()
        self.sensor = None
        self.previous_contact = torch.ones(
            self.num_envs, 4, device=self.device, dtype=torch.bool
        )
        self.previous_foot_vz = torch.zeros(self.num_envs, 4, device=self.device)
        self.continuous_stance = torch.zeros(self.num_envs, 4, device=self.device)
        self.stance_slip_distance = torch.zeros(self.num_envs, 4, device=self.device)
        # P3 consumes event-only stance slip. Keep it separate from the shared
        # P2 diagnostic, whose slip field is a contact-frame speed.
        self.last_p3_detail = torch.zeros(self.num_envs, 24, device=self.device)
        shape = (self.window_frames, self.num_envs, 4)
        self.contact = torch.zeros(shape, device=self.device)
        self.swing_duration = torch.zeros(shape, device=self.device)
        self.max_air = torch.zeros(shape, device=self.device)
        self.prolonged = torch.zeros(shape, device=self.device)
        self.contact_event = torch.zeros(shape, device=self.device)
        self.slip_speed = torch.zeros(shape, device=self.device)
        self._resolve()

    def _invalidate(self, reason: str) -> None:
        self.valid = False
        self.collision_valid = False
        self.invalid_reason = str(reason)
        self.collision_invalid_reason = str(reason)
        _warn(
            "[P2GaitProbe] disabled gait_and_collision_rewards "
            f"reason={self.invalid_reason}"
        )

    def disable_collision(self, reason: str) -> None:
        if self.collision_valid:
            _warn(
                "[P2GaitProbe] disabled body_collision_reward "
                f"reason={reason}"
            )
        self.collision_valid = False
        self.collision_invalid_reason = str(reason)

    def _resolve(self) -> None:
        try:
            body_ids, body_names = self.robot.find_bodies(".*_foot")
        except Exception as exc:
            self._invalidate(f"robot_foot_lookup_failed:{type(exc).__name__}")
            return
        body_ids = body_ids.tolist() if torch.is_tensor(body_ids) else list(body_ids)
        body_names = [str(name) for name in body_names]
        ordered_ids = []
        ordered_names = []
        for prefix in ("FL", "FR", "RL", "RR"):
            matches = [
                (index, name)
                for index, name in zip(body_ids, body_names)
                if prefix.lower() in name.lower()
            ]
            if len(matches) != 1:
                self._invalidate(
                    f"robot_foot_name_{prefix}_matches={len(matches)}"
                )
                return
            ordered_ids.append(matches[0][0])
            ordered_names.append(matches[0][1])
        if len(set(ordered_ids)) != 4 or len(set(ordered_names)) != 4:
            self._invalidate("robot_foot_mapping_not_unique")
            return

        sensors = getattr(getattr(self.env, "scene", None), "sensors", {})
        sensor = None
        if hasattr(sensors, "get"):
            sensor = sensors.get("contact_forces")
        if sensor is None:
            try:
                sensor = sensors["contact_forces"]
            except (KeyError, TypeError):
                sensor = None
        if sensor is None:
            self._invalidate("contact_forces_sensor_missing")
            return

        try:
            sensor_body_names = [str(name) for name in sensor.body_names]
        except Exception as exc:
            self._invalidate(f"sensor_body_names_unavailable:{type(exc).__name__}")
            return
        sensor_name_to_ids: dict[str, list[int]] = {}
        for index, name in enumerate(sensor_body_names):
            sensor_name_to_ids.setdefault(name, []).append(index)
        sensor_foot_ids = []
        for name in ordered_names:
            matches = sensor_name_to_ids.get(name, ())
            if len(matches) != 1:
                self._invalidate(
                    f"sensor_foot_name_{name}_matches={len(matches)}"
                )
                return
            sensor_foot_ids.append(matches[0])
        if len(set(sensor_foot_ids)) != 4:
            self._invalidate("sensor_foot_mapping_not_unique")
            return

        data = getattr(sensor, "data", None)
        air = getattr(data, "current_air_time", None)
        if (
            not torch.is_tensor(air)
            or air.ndim != 2
            or air.shape[0] != self.num_envs
            or air.shape[1] != len(sensor_body_names)
        ):
            shape = tuple(air.shape) if torch.is_tensor(air) else None
            self._invalidate(
                "current_air_time_shape_mismatch:"
                f"actual={shape},expected=({self.num_envs},{len(sensor_body_names)})"
            )
            return

        self.robot_foot_ids = torch.tensor(
            ordered_ids, device=self.device, dtype=torch.long
        )
        self.sensor_foot_ids = torch.tensor(
            sensor_foot_ids, device=self.device, dtype=torch.long
        )
        self.robot_foot_names = tuple(ordered_names)
        self.sensor_body_names = tuple(sensor_body_names)
        self.sensor = sensor
        self.valid = True
        self.invalid_reason = ""
        # Body collision needs a full-body contact sensor, not a foot-only view.
        self.collision_valid = len(sensor_body_names) > len(sensor_foot_ids)
        self.collision_invalid_reason = (
            "" if self.collision_valid else "contact_forces_contains_only_feet"
        )
        _warn(
            "[P2GaitProbe] contact_mapping "
            f"sensor=contact_forces robot_foot_names={list(self.robot_foot_names)} "
            f"robot_foot_ids={self.robot_foot_ids.tolist()} "
            f"sensor_foot_ids={self.sensor_foot_ids.tolist()} "
            f"sensor_body_count={len(self.sensor_body_names)} "
            f"current_air_time_shape={tuple(air.shape)} "
            f"gait_valid={self.valid} collision_valid={self.collision_valid}"
        )
        if not self.collision_valid:
            _warn(
                "[P2GaitProbe] disabled body_collision_reward "
                f"reason={self.collision_invalid_reason}"
            )

    def _clear_envs(self, reset_mask: torch.Tensor) -> None:
        if not bool(reset_mask.any()):
            return
        self.contact[:, reset_mask] = 0.0
        self.swing_duration[:, reset_mask] = 0.0
        self.max_air[:, reset_mask] = 0.0
        self.prolonged[:, reset_mask] = 0.0
        self.contact_event[:, reset_mask] = 0.0
        self.slip_speed[:, reset_mask] = 0.0
        self.previous_contact[reset_mask] = True
        self.previous_foot_vz[reset_mask] = 0.0
        self.continuous_stance[reset_mask] = 0.0
        self.stance_slip_distance[reset_mask] = 0.0
        self.last_p3_detail[reset_mask] = 0.0
        self.env_counts[reset_mask] = 0

    def step(self, reset_mask: torch.Tensor) -> torch.Tensor:
        self.last_p3_detail.zero_()
        output = torch.zeros(
            self.num_envs,
            p2_contract.GAIT_DIAGNOSTIC_DIM,
            device=self.device,
        )
        if not self.valid:
            return output
        reset = reset_mask.to(device=self.device).reshape(-1).bool()
        self._clear_envs(reset)
        data = self.sensor.data
        source_air = getattr(data, "current_air_time", None)
        robot_velocity = getattr(self.robot.data, "body_lin_vel_w", None)
        if (
            not torch.is_tensor(source_air)
            or source_air.ndim != 2
            or source_air.shape[0] != self.num_envs
            or source_air.shape[1] != len(self.sensor_body_names)
            or not torch.is_tensor(robot_velocity)
            or robot_velocity.ndim != 3
            or robot_velocity.shape[0] != self.num_envs
            or int(self.robot_foot_ids.max()) >= robot_velocity.shape[1]
        ):
            self._invalidate("runtime_contact_or_robot_tensor_shape_mismatch")
            return output
        air = source_air[:, self.sensor_foot_ids]
        contact = air <= 0.0
        events = contact & ~self.previous_contact
        last_air = getattr(data, "last_air_time", None)
        if not torch.is_tensor(last_air):
            last_air = air
        else:
            if (
                last_air.ndim != 2
                or last_air.shape[0] != self.num_envs
                or last_air.shape[1] != len(self.sensor_body_names)
            ):
                self._invalidate("runtime_last_air_time_shape_mismatch")
                return output
            last_air = last_air[:, self.sensor_foot_ids]
        foot_velocity3 = robot_velocity[:, self.robot_foot_ids, :3]
        foot_velocity = foot_velocity3[:, :, :2].norm(dim=-1)
        impact_speed = torch.clamp(-self.previous_foot_vz, min=0.0) * events.float()
        previous_stance = self.stance_slip_distance
        previous_stance_time = self.continuous_stance
        stance_end = self.previous_contact & ~contact & (previous_stance_time > 0.0)
        completed_slip = previous_stance * stance_end.float()
        self.continuous_stance = torch.where(
            contact, self.continuous_stance + p2_contract.CONTROL_DT_S, 0.0
        )
        self.stance_slip_distance = torch.where(
            contact,
            self.stance_slip_distance + foot_velocity * p2_contract.CONTROL_DT_S,
            0.0,
        )
        body_position = getattr(self.robot.data, "body_pos_w", None)
        root_position = getattr(self.robot.data, "root_pos_w", None)
        root_quat = getattr(self.robot.data, "root_quat_w", None)
        touchdown_y = torch.zeros_like(foot_velocity)
        if (
            torch.is_tensor(body_position)
            and torch.is_tensor(root_position)
            and torch.is_tensor(root_quat)
            and body_position.ndim == 3
            and body_position.shape[0] == self.num_envs
        ):
            foot_delta = body_position[:, self.robot_foot_ids, :3] - root_position[:, None, :3]
            w, x, y, z = root_quat.unbind(-1)
            # root_quat rotates body to world. Dot with its body-y basis column
            # to apply the full inverse rotation, including roll and pitch.
            body_y_world = torch.stack(
                (
                    2.0 * (x * y - w * z),
                    1.0 - 2.0 * (x.square() + z.square()),
                    2.0 * (y * z + w * x),
                ),
                dim=-1,
            )
            touchdown_y = (foot_delta * body_y_world.unsqueeze(1)).sum(dim=-1)
            touchdown_y = touchdown_y * events.float()
        self.last_p3_detail[:, 0:4] = events.float()
        self.last_p3_detail[:, 4:8] = impact_speed
        side = torch.tensor((1.0, -1.0, 1.0, -1.0), device=self.device)
        self.last_p3_detail[:, 8:12] = touchdown_y * side
        self.last_p3_detail[:, 12:16] = self.continuous_stance
        self.last_p3_detail[:, 16:20] = completed_slip
        self.last_p3_detail[:, 20:24] = stance_end.float()
        self.last_p3_detail[reset] = 0.0
        self.previous_foot_vz = torch.where(
            reset.unsqueeze(-1), torch.zeros_like(self.previous_foot_vz), foot_velocity3[:, :, 2]
        )

        slot = self.index
        self.contact[slot] = contact.float()
        self.swing_duration[slot] = last_air * events.float()
        self.max_air[slot] = air
        self.prolonged[slot] = (
            air > p2_contract.GAIT_PROLONGED_AIR_SECONDS
        ).float()
        self.contact_event[slot] = events.float()
        # Shared P2 contract: average planar foot speed over valid contact
        # frames. Per-stance distance is exported through the P3 detail tail.
        self.slip_speed[slot] = foot_velocity * contact.float()
        # A reset row is the first sample of a new episode, not a continuation
        # of the old gait window. Keep its ring slot empty and its denominator at
        # zero so the next frame cannot observe two samples with a count of one.
        if bool(reset.any()):
            self.contact[slot, reset] = 0.0
            self.swing_duration[slot, reset] = 0.0
            self.max_air[slot, reset] = 0.0
            self.prolonged[slot, reset] = 0.0
            self.contact_event[slot, reset] = 0.0
            self.slip_speed[slot, reset] = 0.0
        self.previous_contact = torch.where(
            reset.unsqueeze(-1),
            torch.ones_like(contact),
            contact,
        )
        self.index = (self.index + 1) % self.window_frames
        self.count = min(self.count + 1, self.window_frames)
        self.env_counts = torch.clamp(
            self.env_counts + (~reset).to(self.env_counts.dtype),
            max=self.window_frames,
        )

        denominator = self.env_counts.clamp_min(1).to(torch.float32).unsqueeze(-1)
        duty = self.contact.sum(dim=0) / denominator
        event_count = self.contact_event.sum(dim=0)
        mean_swing = self.swing_duration.sum(dim=0) / event_count.clamp_min(1.0)
        max_air = self.max_air.amax(dim=0)
        prolonged = self.prolonged.sum(dim=0) / denominator
        step_frequency = event_count / (
            denominator * p2_contract.CONTROL_DT_S
        )
        contact_count = self.contact.sum(dim=0)
        slip = self.slip_speed.sum(dim=0) / contact_count.clamp_min(1.0)
        output[:, 0:4] = duty
        output[:, 4:8] = mean_swing
        output[:, 8:12] = max_air
        output[:, 12:16] = prolonged
        output[:, 16:20] = step_frequency
        output[:, 20:24] = slip
        output[:, 24] = (self.env_counts >= self.window_frames).float()
        output[reset] = 0.0
        return output


class P2GaitBaseline:
    """Apply a fixed, versioned P1.5 parent gait non-degradation envelope."""

    VERSION = p2_contract.GAIT_BASELINE_VERSION

    def __init__(self):
        self.samples = 0
        self.finalized = True
        self.thresholds = dict(p2_contract.GAIT_BASELINE_THRESHOLDS)

    @staticmethod
    def _metrics(diagnostic_aux: torch.Tensor) -> tuple[torch.Tensor, ...]:
        duty = diagnostic_aux[:, p2_contract.GAIT_DUTY_SLICE]
        swing = diagnostic_aux[:, p2_contract.GAIT_MEAN_SWING_SLICE]
        prolonged = diagnostic_aux[:, p2_contract.GAIT_PROLONGED_RATIO_SLICE]
        frequency = diagnostic_aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE]
        return (
            duty.amax(dim=-1) - duty.amin(dim=-1),
            swing.amax(dim=-1) - swing.amin(dim=-1),
            prolonged.amax(dim=-1),
            frequency.amax(dim=-1) - frequency.amin(dim=-1),
        )

    def observe(self, diagnostic_aux: torch.Tensor) -> None:
        """Compatibility no-op: live P2 data must never redefine the parent baseline."""
        del diagnostic_aux

    def finalize(self) -> None:
        """Compatibility no-op for callers written against the v1 API."""
        self.finalized = True

    def penalty(
        self,
        diagnostic_aux: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        count = diagnostic_aux.shape[0]
        zeros = torch.zeros(count, device=diagnostic_aux.device)
        if not self.finalized:
            return zeros, {
                "duty_excess": zeros,
                "swing_excess": zeros,
                "prolonged_excess": zeros,
                "frequency_excess": zeros,
            }
        valid = diagnostic_aux[:, p2_contract.GAIT_VALID_INDEX] > 0.5
        duty, swing, prolonged, frequency = self._metrics(diagnostic_aux)
        duty_excess = torch.clamp(
            (duty - self.thresholds["duty_imbalance"]) / 0.25, 0.0, 1.0
        )
        swing_excess = torch.clamp(
            (swing - self.thresholds["swing_imbalance_s"]) / 0.25, 0.0, 1.0
        )
        prolonged_excess = torch.clamp(
            (prolonged - self.thresholds["prolonged_air_ratio"]) / 0.25,
            0.0,
            1.0,
        )
        frequency_excess = torch.clamp(
            (
                frequency
                - self.thresholds["step_frequency_imbalance_hz"]
            )
            / 1.5,
            0.0,
            1.0,
        )
        excess = (
            duty_excess
            + swing_excess
            + prolonged_excess
            + frequency_excess
        ) / 4.0
        penalty = -p2_contract.GAIT_PENALTY_CAP * torch.clamp(excess, 0.0, 1.0)
        penalty = torch.where(valid, penalty, torch.zeros_like(penalty))
        return penalty, {
            "duty_excess": duty_excess,
            "swing_excess": swing_excess,
            "prolonged_excess": prolonged_excess,
            "frequency_excess": frequency_excess,
        }

    def state_dict(self, *, parent_sha256: str | None = None) -> dict[str, object]:
        return {
            "version": self.VERSION,
            "source_parent_model_id": p2_contract.GAIT_BASELINE_PARENT_MODEL_ID,
            "source_parent_label": p2_contract.GAIT_BASELINE_PARENT_LABEL,
            "source_parent_sha256": parent_sha256,
            "source": "fixed_versioned_parent_envelope",
            "window_seconds": p2_contract.GAIT_WINDOW_SECONDS,
            "calibration_seconds": 0.0,
            "generation_config": {
                "kind": "conservative_parent_envelope",
                "empirical_p99_claimed": False,
                "thresholds": dict(p2_contract.GAIT_BASELINE_THRESHOLDS),
            },
            "samples": self.samples,
            "finalized": self.finalized,
            "thresholds": dict(self.thresholds),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("version") != self.VERSION:
            raise ValueError("P2 gait baseline version mismatch")
        thresholds = state.get("thresholds")
        if not isinstance(thresholds, dict):
            raise ValueError("P2 gait baseline thresholds missing")
        loaded = {key: float(value) for key, value in thresholds.items()}
        if loaded != p2_contract.GAIT_BASELINE_THRESHOLDS:
            raise ValueError("P2 fixed gait baseline thresholds mismatch")
        if not all(
            math.isfinite(value) and value >= 0.0 for value in loaded.values()
        ):
            raise ValueError("P2 gait baseline thresholds must be finite and non-negative")
        self.thresholds = loaded
        self.samples = int(state.get("samples", 0))
        if not bool(state.get("finalized", False)):
            raise ValueError("P2 fixed gait baseline must be finalized")
        self.finalized = True
