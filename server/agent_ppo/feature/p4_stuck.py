#!/usr/bin/env python3
"""P4-only confirmed wall-stuck tracker backed by Isaac termination manager."""

from __future__ import annotations

import copy
import math
import sys
import time
from typing import Any

import torch

from agent_ppo.feature import p2_contract, p4_contract


class MotionWallStuckTracker:
    """Detect physical confinement without command-intent or synthetic done flags."""

    LEGACY_SPATIAL_DIAMETER_M = 0.50
    RECOVERY_WALL_ABSENCE_S = 0.50
    RECOVERY_CENTER_DISPLACEMENT_M = 0.50
    ROTATIONAL_RECOVERY_YAW_RAD = math.radians(20.0)
    WALL_CONTACT_EMA_TAU_S = 0.20
    ROTATIONAL_RECOVERY_WALL_EMA_RATIO = 0.70

    def __init__(
        self,
        env,
        *,
        num_envs: int,
        device,
        config: dict[str, Any] | None = None,
        episode_length_s: float = 120.0,
    ):
        self.env = env
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        supplied = dict(config or {})
        spatial_diameter_m = supplied.pop("spatial_diameter_m", None)
        if spatial_diameter_m is not None and "radius_m" in supplied:
            if not math.isclose(
                float(spatial_diameter_m),
                float(supplied["radius_m"]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "P4 stuck reset spatial_diameter_m conflicts with legacy radius_m"
                )
        merged = p4_contract.normalize_stuck_reset_contract(supplied)
        self.enabled = bool(merged["enabled"])
        self.requested_mode = str(merged.get("mode", "shadow"))
        self.schedule_enabled = bool(merged["schedule_enabled"])
        if self.requested_mode not in {"shadow", "active", "disabled"}:
            raise ValueError(
                f"unsupported P4 stuck-reset mode {self.requested_mode!r}"
            )
        self.tight_confirmation_s = float(merged["confirmation_s"])
        self.initial_confirmation_s = float(merged["initial_confirmation_s"])
        self.activation_delay_s = float(merged["activation_delay_s"])
        self.tighten_after_s = float(merged["tighten_after_s"])
        self.resume_offset_s = float(merged["resume_offset_s"])
        self._started_monotonic = time.monotonic()
        initial_elapsed = self.resume_offset_s
        self.mode = self._scheduled_mode(initial_elapsed)
        self.confirmation_s = self._scheduled_confirmation_s(initial_elapsed)
        # ``spatial_diameter_m`` is the unambiguous runtime quantity.  Older
        # checkpoint/config payloads use ``radius_m`` as its numeric alias.
        # Keep the latter readable, but never reinterpret it as a radius.
        self.radius_m = float(merged["radius_m"])
        self.spatial_diameter_m = float(
            self.radius_m if spatial_diameter_m is None else spatial_diameter_m
        )
        if (
            not math.isfinite(self.spatial_diameter_m)
            or self.spatial_diameter_m < 0.0
        ):
            raise ValueError(
                "P4 stuck reset spatial_diameter_m must be finite and non-negative"
            )
        self.min_goal_distance_m = float(merged["min_goal_distance_m"])
        self.body_collision_force_n = float(merged["body_collision_force_n"])
        self.wall_evidence_latch_s = float(merged["wall_evidence_latch_s"])
        self.episode_grace_s = float(merged["episode_grace_s"])
        self.push_grace_s = float(merged["push_grace_s"])
        self.max_true_motion_speed_m_s = float(merged["max_true_motion_speed_m_s"])
        self.episode_length_s = float(episode_length_s)
        self.dt_s = float(getattr(env, "step_dt", p2_contract.CONTROL_DT_S))
        self.dt_valid = abs(self.dt_s - p2_contract.CONTROL_DT_S) <= 1.0e-6
        self.confirmation_steps = max(1, round(self.confirmation_s / self.dt_s))
        self.max_confirmation_steps = max(
            1,
            round(
                max(self.initial_confirmation_s, self.tight_confirmation_s)
                / self.dt_s
            ),
        )
        self.wall_latch_steps = max(1, round(self.wall_evidence_latch_s / self.dt_s))
        self.recovery_wall_absence_steps = max(
            1, round(self.RECOVERY_WALL_ABSENCE_S / self.dt_s)
        )

        self.position_history = torch.zeros(
            self.max_confirmation_steps, self.num_envs, 2, device=self.device
        )
        self.position_history_valid = torch.zeros(
            self.max_confirmation_steps,
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.position_history_index = 0
        self.position_sample_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.window_center = torch.zeros(self.num_envs, 2, device=self.device)
        self.window_diameter = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        self.candidate_center = torch.zeros_like(self.window_center)
        self.candidate_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.candidate_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.eligible_steps = torch.zeros_like(self.candidate_steps)
        self.wall_evidence_age_steps = torch.full(
            (self.num_envs,), self.wall_latch_steps + 1,
            dtype=torch.long,
            device=self.device,
        )
        self.wall_absence_steps = torch.zeros_like(self.candidate_steps)
        self.wall_sequence_steps = torch.zeros_like(self.candidate_steps)
        self.episode_age_steps = torch.zeros_like(self.candidate_steps)
        self.wall_contact_ema = torch.zeros(self.num_envs, device=self.device)
        self.candidate_wall_contact_ema = torch.zeros(
            self.num_envs, device=self.device
        )
        self.candidate_yaw = torch.zeros(self.num_envs, device=self.device)
        self.foot_jam_shadow = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.triggered = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.shadow_event_reported = torch.zeros_like(self.triggered)
        self.term_available = False
        self.term_config_valid = False
        self._termination_config_attempts = 0
        self.last_diagnostics = torch.zeros(self.num_envs, 13, device=self.device)
        self._configure_termination()

    def _elapsed_s(self) -> float:
        return self.resume_offset_s + max(
            0.0, time.monotonic() - self._started_monotonic
        )

    def _scheduled_mode(self, elapsed_s: float) -> str:
        if not self.enabled or self.requested_mode == "disabled":
            return "disabled"
        if self.requested_mode == "shadow":
            return "shadow"
        if not self.schedule_enabled:
            return self.requested_mode
        return "shadow" if elapsed_s < self.activation_delay_s else "active"

    def _scheduled_confirmation_s(self, elapsed_s: float) -> float:
        if not self.schedule_enabled:
            return self.tight_confirmation_s
        return (
            self.initial_confirmation_s
            if elapsed_s < self.tighten_after_s
            else self.tight_confirmation_s
        )

    def _reset_detection_state(self) -> None:
        self.position_history.zero_()
        self.position_history_valid.zero_()
        self.position_history_index = 0
        self.position_sample_count.zero_()
        self.window_center.zero_()
        self.window_diameter.fill_(float("inf"))
        self.candidate_center.zero_()
        self.candidate_active.zero_()
        self.candidate_steps.zero_()
        self.eligible_steps.zero_()
        self.wall_evidence_age_steps.fill_(self.wall_latch_steps + 1)
        self.wall_absence_steps.zero_()
        self.wall_sequence_steps.zero_()
        self.wall_contact_ema.zero_()
        self.candidate_wall_contact_ema.zero_()
        self.candidate_yaw.zero_()
        self.foot_jam_shadow.zero_()
        self.triggered.zero_()
        self.shadow_event_reported.zero_()
        self._write_counter(torch.zeros_like(self.candidate_steps))

    def _refresh_schedule(self) -> None:
        elapsed_s = self._elapsed_s()
        next_mode = self._scheduled_mode(elapsed_s)
        next_confirmation_s = self._scheduled_confirmation_s(elapsed_s)
        next_steps = max(1, round(next_confirmation_s / self.dt_s))
        if next_mode == self.mode and next_steps == self.confirmation_steps:
            return
        previous_mode = self.mode
        previous_confirmation_s = self.confirmation_s
        self.mode = next_mode
        self.confirmation_s = next_confirmation_s
        self.confirmation_steps = next_steps
        self.term_config_valid = False
        self._termination_config_attempts = 0
        self._reset_detection_state()
        self._configure_termination()
        print(
            "[P4StuckResetPhase] "
            f"elapsed_s={elapsed_s:.3f} "
            f"previous_mode={previous_mode} mode={self.mode} "
            f"previous_confirmation_s={previous_confirmation_s:.3f} "
            f"confirmation_s={self.confirmation_s:.3f} "
            f"term_available={int(self.term_available)} "
            f"term_config_valid={int(self.term_config_valid)}",
            file=sys.stderr,
            flush=True,
        )

    def _configure_termination(self) -> None:
        self._termination_config_attempts += 1
        manager = getattr(self.env, "termination_manager", None)
        getter = getattr(manager, "get_term_cfg", None)
        setter = getattr(manager, "set_term_cfg", None)
        active = getattr(manager, "active_terms", None)
        if active is None:
            active = getattr(manager, "_term_names", ())
        self.term_available = bool(
            manager is not None
            and "nav_stuck_timeout" in set(active or ())
            and callable(getter)
            and callable(setter)
        )
        if not self.term_available:
            self._write_counter(torch.zeros_like(self.candidate_steps))
            return
        try:
            cfg = copy.deepcopy(getter("nav_stuck_timeout"))
            if not bool(getattr(cfg, "time_out", False)):
                return
            params = getattr(cfg, "params", None)
            if not isinstance(params, dict):
                return
            params["max_stuck"] = int(self.confirmation_steps)
            setter("nav_stuck_timeout", cfg)
            readback = getter("nav_stuck_timeout")
            readback_params = getattr(readback, "params", None)
            self.term_config_valid = bool(
                self.dt_valid
                and getattr(readback, "time_out", False)
                and isinstance(readback_params, dict)
                and int(readback_params.get("max_stuck", -1)) == self.confirmation_steps
            )
        except Exception:
            self.term_config_valid = False
        if not self.term_config_valid:
            self._write_counter(torch.zeros_like(self.candidate_steps))

    def _write_counter(self, value: torch.Tensor) -> None:
        setattr(self.env, "_nav_motion_stuck", value.to(self.device).float())

    def termination_mask(self, reset: torch.Tensor) -> torch.Tensor:
        """Return only the platform-owned stuck timeout, never a local proxy."""
        reset = reset.to(self.device).reshape(-1).bool()
        if reset.numel() != self.num_envs:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        manager = getattr(self.env, "termination_manager", None)
        getter = getattr(manager, "get_term", None)
        if callable(getter) and self.term_available:
            try:
                value = getter("nav_stuck_timeout")
                if torch.is_tensor(value) and value.numel() == self.num_envs:
                    return value.to(self.device).reshape(-1).bool() & reset
            except Exception:
                pass
        # ``triggered`` only records that the worker supplied a counter to a
        # validated term.  It is not evidence that the manager actually reset
        # this row, and must never manufacture reason 4 after a failed read.
        return torch.zeros_like(reset)

    def _update_position_window(
        self, root_xy: torch.Tensor, valid: torch.Tensor, reset: torch.Tensor
    ) -> None:
        if bool(reset.any()):
            self.position_history_valid[:, reset] = False
            self.position_sample_count[reset] = 0
        index = self.position_history_index
        self.position_history[index] = root_xy
        self.position_history_valid[index] = valid & ~reset
        self.position_history_index = (index + 1) % self.max_confirmation_steps
        if self.confirmation_steps < self.max_confirmation_steps:
            expire_index = (
                self.position_history_index - self.confirmation_steps - 1
            ) % self.max_confirmation_steps
            self.position_history_valid[expire_index] = False
        self.position_sample_count = torch.where(
            valid & ~reset,
            torch.clamp(self.position_sample_count + 1, max=self.confirmation_steps),
            torch.zeros_like(self.position_sample_count),
        )

        history_valid = self.position_history_valid
        expanded_valid = history_valid.unsqueeze(-1)
        sample_count = history_valid.sum(dim=0).clamp_min(1).to(root_xy.dtype)
        self.window_center = torch.where(
            (self.position_sample_count > 0).unsqueeze(-1),
            (self.position_history * expanded_valid).sum(dim=0)
            / sample_count.unsqueeze(-1),
            root_xy,
        )
        positive_inf = torch.full_like(self.position_history, float("inf"))
        negative_inf = torch.full_like(self.position_history, float("-inf"))
        minimum = torch.where(expanded_valid, self.position_history, positive_inf).amin(dim=0)
        maximum = torch.where(expanded_valid, self.position_history, negative_inf).amax(dim=0)
        enclosure = torch.linalg.vector_norm(maximum - minimum, dim=-1)
        self.window_diameter = torch.where(
            self.position_sample_count >= self.confirmation_steps,
            enclosure,
            torch.full_like(enclosure, float("inf")),
        )

    def update(
        self,
        *,
        root_xy: torch.Tensor,
        goal_distance: torch.Tensor,
        collision_force: torch.Tensor,
        mapping_valid: torch.Tensor,
        reset: torch.Tensor,
        terminal_reason: torch.Tensor,
        seconds_since_push: torch.Tensor,
        episode_age_s: torch.Tensor,
        true_velocity3: torch.Tensor,
        motion_intent: torch.Tensor | None = None,
        yaw: torch.Tensor | None = None,
        foot_jam: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._refresh_schedule()
        # Manager assembly can lag observation construction. Retry a bounded
        # number of real frames, but never synthesize a terminal when unavailable.
        if self.enabled and not self.term_config_valid and self._termination_config_attempts < 8:
            self._configure_termination()
        if (
            self.enabled
            and self.mode == "active"
            and self._termination_config_attempts >= 8
            and not (self.term_available and self.term_config_valid)
        ):
            raise RuntimeError(
                "P4 active wall-stuck reset requires a validated nav_stuck_timeout "
                f"termination hook after {self._termination_config_attempts} attempts"
            )

        raw_root = root_xy.to(self.device)
        root_shape_valid = raw_root.shape == (self.num_envs, 2)
        if root_shape_valid:
            root_valid_mask = torch.isfinite(raw_root).all(dim=-1)
            root_xy = torch.nan_to_num(raw_root)
        else:
            root_valid_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            root_xy = torch.zeros(self.num_envs, 2, device=self.device)
        reset = reset.to(self.device).bool().reshape(-1)
        reason = terminal_reason.to(self.device).round().long().reshape(-1)
        previous = self.last_diagnostics.clone()

        raw_true = true_velocity3.to(self.device)
        velocity_shape_valid = raw_true.shape == (self.num_envs, 3)
        if velocity_shape_valid:
            velocity_valid = torch.isfinite(raw_true).all(dim=-1)
            true_speed = torch.linalg.vector_norm(raw_true[:, :2], dim=-1)
            true_motion_low = true_speed < self.max_true_motion_speed_m_s
        else:
            velocity_valid = torch.zeros_like(root_valid_mask)
            true_motion_low = torch.zeros_like(root_valid_mask)

        if torch.is_tensor(yaw) and yaw.numel() == self.num_envs:
            raw_yaw = yaw.to(self.device).reshape(-1)
            yaw_valid = torch.isfinite(raw_yaw)
            current_yaw = torch.nan_to_num(raw_yaw)
        else:
            yaw_valid = torch.zeros_like(root_valid_mask)
            current_yaw = torch.zeros(self.num_envs, device=self.device)

        mapping = mapping_valid.to(self.device).bool().reshape(-1)
        raw_collision = collision_force.to(self.device).reshape(-1)
        collision_valid = raw_collision.numel() == self.num_envs
        if collision_valid:
            collision_valid = torch.isfinite(raw_collision)
            safe_collision = torch.nan_to_num(raw_collision)
        else:
            collision_valid = torch.zeros_like(root_valid_mask)
            safe_collision = torch.zeros(self.num_envs, device=self.device)
        wall_now = (
            (safe_collision >= self.body_collision_force_n) & mapping & collision_valid
        )
        ema_alpha = min(1.0, self.dt_s / self.WALL_CONTACT_EMA_TAU_S)
        self.wall_contact_ema = torch.where(
            mapping & collision_valid,
            self.wall_contact_ema
            + ema_alpha * (safe_collision - self.wall_contact_ema),
            self.wall_contact_ema,
        )
        self.wall_evidence_age_steps = torch.where(
            wall_now,
            torch.zeros_like(self.wall_evidence_age_steps),
            self.wall_evidence_age_steps + 1,
        )
        recent_wall = self.wall_evidence_age_steps <= self.wall_latch_steps
        self.wall_absence_steps = torch.where(
            wall_now,
            torch.zeros_like(self.wall_absence_steps),
            self.wall_absence_steps + 1,
        )
        self.episode_age_steps = torch.round(
            episode_age_s.to(self.device).reshape(-1) / self.dt_s
        ).long().clamp_min(0)
        structural_eligible = (
            self.enabled
            & (self.mode != "disabled")
            & (goal_distance.to(self.device).reshape(-1) > self.min_goal_distance_m)
            & (episode_age_s.to(self.device).reshape(-1) >= self.episode_grace_s)
            & (seconds_since_push.to(self.device).reshape(-1) >= self.push_grace_s)
            & mapping
            & root_valid_mask
            & velocity_valid
        )
        stuck_sample = structural_eligible & true_motion_low
        self.eligible_steps = torch.where(
            stuck_sample,
            self.eligible_steps + 1,
            torch.zeros_like(self.eligible_steps),
        )
        self._update_position_window(root_xy, root_valid_mask, reset)

        candidate_entry = stuck_sample & recent_wall & ~self.candidate_active
        self.candidate_center = torch.where(
            candidate_entry.unsqueeze(-1), self.window_center, self.candidate_center
        )
        self.candidate_yaw = torch.where(
            candidate_entry & yaw_valid, current_yaw, self.candidate_yaw
        )
        self.candidate_wall_contact_ema = torch.where(
            candidate_entry,
            self.wall_contact_ema,
            self.candidate_wall_contact_ema,
        )
        self.candidate_active |= candidate_entry
        self.candidate_wall_contact_ema = torch.where(
            self.candidate_active & wall_now,
            torch.maximum(
                self.candidate_wall_contact_ema,
                self.wall_contact_ema,
            ),
            self.candidate_wall_contact_ema,
        )
        self.candidate_steps = torch.where(
            self.candidate_active,
            self.candidate_steps + 1,
            torch.zeros_like(self.candidate_steps),
        )
        center_displacement = torch.linalg.vector_norm(
            self.window_center - self.candidate_center, dim=-1
        )
        translational_recovered = (
            self.candidate_active
            & structural_eligible
            & (self.wall_absence_steps >= self.recovery_wall_absence_steps)
            & (center_displacement >= self.RECOVERY_CENTER_DISPLACEMENT_M)
        )
        yaw_delta = torch.atan2(
            torch.sin(current_yaw - self.candidate_yaw),
            torch.cos(current_yaw - self.candidate_yaw),
        ).abs()
        rotational_recovered = (
            self.candidate_active
            & structural_eligible
            & yaw_valid
            & (yaw_delta >= self.ROTATIONAL_RECOVERY_YAW_RAD)
            & (self.candidate_wall_contact_ema > 0.0)
            & (
                self.wall_contact_ema
                <= self.candidate_wall_contact_ema
                * self.ROTATIONAL_RECOVERY_WALL_EMA_RATIO
            )
        )
        recovered = translational_recovered | rotational_recovered
        # Invalid/missing evidence may suspend confirmation, but it must never
        # manufacture a recovery edge for the learner-side monitor.
        self.candidate_active &= ~recovered & ~reset
        self.candidate_steps = torch.where(
            self.candidate_active,
            self.candidate_steps,
            torch.zeros_like(self.candidate_steps),
        )

        spatially_confined = self.window_diameter < self.spatial_diameter_m
        confirmed = (
            self.candidate_active
            & recent_wall
            & spatially_confined
            & (self.candidate_steps >= self.confirmation_steps)
            & (self.eligible_steps >= self.confirmation_steps)
            & (self.position_sample_count >= self.confirmation_steps)
        )
        would_reset = confirmed & ~self.shadow_event_reported
        self.shadow_event_reported |= confirmed
        self.shadow_event_reported &= ~recovered & ~reset
        active_trigger = (
            confirmed
            & (self.mode == "active")
            & self.term_available
            & self.term_config_valid
        )
        self.triggered |= active_trigger
        counter = (
            torch.where(confirmed, self.eligible_steps, torch.zeros_like(self.eligible_steps))
            if self.mode == "active" and self.term_config_valid
            else torch.zeros_like(self.eligible_steps)
        )
        self._write_counter(counter)
        self.wall_sequence_steps = torch.where(
            self.candidate_active & recent_wall,
            self.wall_sequence_steps + 1,
            torch.zeros_like(self.wall_sequence_steps),
        )
        wall_reset = reset & (reason == 4)

        # Foot-jam is an optional teacher/shadow signal.  It is deliberately
        # excluded from ``wall_now``, active counters, and reason 4.
        self.foot_jam_shadow = (
            foot_jam.to(self.device).reshape(-1).bool()
            if torch.is_tensor(foot_jam) and foot_jam.numel() == self.num_envs
            else torch.zeros_like(reset)
        )
        setattr(self.env, "_p4_foot_jam_shadow", self.foot_jam_shadow.clone())

        diagnostics = torch.zeros_like(self.last_diagnostics)
        diagnostics[:, 0] = spatially_confined.float()
        diagnostics[:, 1] = recent_wall.float()
        diagnostics[:, 2] = self.candidate_active.float()
        diagnostics[:, 3] = self.candidate_steps.float() * self.dt_s
        diagnostics[:, 4] = would_reset.float()
        diagnostics[:, 5] = 0.0
        diagnostics[:, 6] = mapping.float()
        diagnostics[:, 7] = (
            wall_reset
            & (seconds_since_push.to(self.device).reshape(-1) < self.push_grace_s)
        ).float()
        diagnostics[:, 8] = torch.where(
            would_reset,
            torch.clamp(
                self.episode_length_s - episode_age_s.to(self.device).reshape(-1),
                min=0.0,
            ),
            torch.zeros(self.num_envs, device=self.device),
        )
        diagnostics[:, 9] = self.wall_sequence_steps.float() * self.dt_s
        diagnostics[:, 10] = float(self.term_available)
        diagnostics[:, 11] = float(self.term_config_valid)
        # Index 12 is reserved in the worker tail for spawn-hook readiness.
        diagnostics[:, 12] = 0.0

        if bool(wall_reset.any()):
            diagnostics[wall_reset] = previous[wall_reset]
            diagnostics[wall_reset, 5] = 1.0
            diagnostics[wall_reset, 7] = (
                seconds_since_push.to(self.device).reshape(-1)[wall_reset]
                < self.push_grace_s
            ).float()
        if bool(reset.any()):
            self.position_history_valid[:, reset] = False
            self.position_sample_count[reset] = 0
            self.candidate_active[reset] = False
            self.candidate_steps[reset] = 0
            self.eligible_steps[reset] = 0
            self.wall_evidence_age_steps[reset] = self.wall_latch_steps + 1
            self.wall_absence_steps[reset] = 0
            self.wall_sequence_steps[reset] = 0
            self.episode_age_steps[reset] = 0
            self.wall_contact_ema[reset] = 0.0
            self.candidate_wall_contact_ema[reset] = 0.0
            self.candidate_yaw[reset] = 0.0
            self.foot_jam_shadow[reset] = False
            self.triggered[reset] = False
            self.shadow_event_reported[reset] = False
            counter = getattr(self.env, "_nav_motion_stuck", None)
            if torch.is_tensor(counter) and counter.numel() == self.num_envs:
                counter[reset] = 0.0
        self.last_diagnostics = diagnostics
        return diagnostics


__all__ = ["MotionWallStuckTracker"]
