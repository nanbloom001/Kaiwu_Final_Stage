#!/usr/bin/env python3
"""P4-only confirmed wall-stuck tracker backed by Isaac termination manager."""

from __future__ import annotations

import copy
from typing import Any

import torch

from agent_ppo.feature import p2_contract, p4_contract


class MotionWallStuckTracker:
    """Maintain worker-owned motion confinement state without faking ``done``."""

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
        merged = p4_contract.normalize_stuck_reset_contract(config)
        self.enabled = bool(merged["enabled"])
        self.mode = str(merged.get("mode", "shadow"))
        if self.mode not in {"shadow", "active", "disabled"}:
            raise ValueError(f"unsupported P4 stuck-reset mode {self.mode!r}")
        self.confirmation_s = float(merged["confirmation_s"])
        self.radius_m = float(merged["radius_m"])
        self.min_goal_distance_m = float(merged["min_goal_distance_m"])
        self.body_collision_force_n = float(merged["body_collision_force_n"])
        self.wall_evidence_latch_s = float(merged["wall_evidence_latch_s"])
        self.episode_grace_s = float(merged["episode_grace_s"])
        self.push_grace_s = float(merged["push_grace_s"])
        self.episode_length_s = float(episode_length_s)
        self.dt_s = float(getattr(env, "step_dt", p2_contract.CONTROL_DT_S))
        self.dt_valid = abs(self.dt_s - p2_contract.CONTROL_DT_S) <= 1.0e-6
        self.confirmation_steps = max(1, round(self.confirmation_s / self.dt_s))
        self.wall_latch_steps = max(1, round(self.wall_evidence_latch_s / self.dt_s))
        self.anchor_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.anchor_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.confined_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.wall_evidence_age_steps = torch.full(
            (self.num_envs,), self.wall_latch_steps + 1,
            dtype=torch.long,
            device=self.device,
        )
        self.wall_sequence_steps = torch.zeros_like(self.confined_steps)
        self.episode_age_steps = torch.zeros_like(self.confined_steps)
        self.triggered = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.shadow_event_reported = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.term_available = False
        self.term_config_valid = False
        self._termination_config_attempts = 0
        self.last_diagnostics = torch.zeros(
            self.num_envs, 12, device=self.device
        )
        self._configure_termination()

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
            self._write_counter(torch.zeros_like(self.confined_steps))
            return
        try:
            original = getter("nav_stuck_timeout")
            cfg = copy.deepcopy(original)
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
                and
                getattr(readback, "time_out", False)
                and isinstance(readback_params, dict)
                and int(readback_params.get("max_stuck", -1))
                == self.confirmation_steps
            )
        except Exception:
            self.term_config_valid = False
        if not self.term_config_valid:
            self._write_counter(torch.zeros_like(self.confined_steps))

    def _write_counter(self, value: torch.Tensor) -> None:
        setattr(self.env, "_nav_motion_stuck", value.to(self.device).float())

    def termination_mask(self, reset: torch.Tensor) -> torch.Tensor:
        manager = getattr(self.env, "termination_manager", None)
        getter = getattr(manager, "get_term", None)
        if callable(getter) and self.term_available:
            try:
                value = getter("nav_stuck_timeout")
                if torch.is_tensor(value) and value.numel() == self.num_envs:
                    return value.to(self.device).reshape(-1).bool() & reset
            except Exception:
                pass
        return reset & self.triggered

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
    ) -> torch.Tensor:
        # Observation terms may run once while Isaac is still assembling the
        # termination manager. Retry on the first real frames instead of
        # permanently pinning the tracker to the early unavailable result.
        if (
            self.enabled
            and not self.term_config_valid
            and self._termination_config_attempts < 8
        ):
            self._configure_termination()
        raw_root_xy = root_xy.to(self.device)
        root_valid = torch.isfinite(raw_root_xy).all(dim=-1)
        root_xy = torch.nan_to_num(raw_root_xy)
        reset = reset.to(self.device).bool().reshape(-1)
        reason = terminal_reason.to(self.device).round().long().reshape(-1)
        previous = self.last_diagnostics.clone()

        wall_now = (
            collision_force.to(self.device).reshape(-1)
            >= self.body_collision_force_n
        ) & mapping_valid.to(self.device).bool().reshape(-1)
        self.wall_evidence_age_steps = torch.where(
            wall_now,
            torch.zeros_like(self.wall_evidence_age_steps),
            self.wall_evidence_age_steps + 1,
        )
        recent_wall = self.wall_evidence_age_steps <= self.wall_latch_steps
        self.episode_age_steps = torch.round(
            episode_age_s.to(self.device).reshape(-1) / self.dt_s
        ).long().clamp_min(0)
        eligible = (
            self.enabled
            & (self.mode != "disabled")
            & (goal_distance.to(self.device).reshape(-1) > self.min_goal_distance_m)
            & (episode_age_s.to(self.device).reshape(-1) >= self.episode_grace_s)
            & (seconds_since_push.to(self.device).reshape(-1) >= self.push_grace_s)
            & mapping_valid.to(self.device).bool().reshape(-1)
            & root_valid
        )
        displacement = torch.linalg.vector_norm(root_xy - self.anchor_xy, dim=-1)
        moved = self.anchor_valid & (displacement >= self.radius_m)
        initialize = ~self.anchor_valid
        clear = reset | ~eligible | moved | initialize
        self.anchor_xy = torch.where(clear.unsqueeze(-1), root_xy, self.anchor_xy)
        self.anchor_valid |= initialize
        self.confined_steps = torch.where(
            clear | ~recent_wall,
            torch.zeros_like(self.confined_steps),
            self.confined_steps + 1,
        )
        self.wall_sequence_steps = torch.where(
            clear | ~recent_wall,
            torch.zeros_like(self.wall_sequence_steps),
            self.wall_sequence_steps + 1,
        )
        sequence_clear = clear | ~recent_wall
        self.shadow_event_reported &= ~sequence_clear
        candidate = eligible & recent_wall & (self.confined_steps > 0)
        would_reset = candidate & (
            self.confined_steps >= self.confirmation_steps
        ) & ~self.shadow_event_reported
        self.shadow_event_reported |= would_reset
        active_trigger = (
            would_reset
            & (self.mode == "active")
            & self.term_available
            & self.term_config_valid
        )
        self.triggered |= active_trigger
        counter = (
            self.confined_steps
            if self.mode == "active" and self.term_config_valid
            else torch.zeros_like(self.confined_steps)
        )
        self._write_counter(counter)
        wall_reset = reset & (reason == 4)

        diagnostics = torch.zeros_like(self.last_diagnostics)
        diagnostics[:, 0] = (eligible & ~moved).float()
        diagnostics[:, 1] = recent_wall.float()
        diagnostics[:, 2] = candidate.float()
        diagnostics[:, 3] = self.confined_steps.float() * self.dt_s
        diagnostics[:, 4] = would_reset.float()
        diagnostics[:, 5] = 0.0
        diagnostics[:, 6] = mapping_valid.to(self.device).float().reshape(-1)
        diagnostics[:, 7] = (
            wall_reset
            & (
                seconds_since_push.to(self.device).reshape(-1)
                < self.push_grace_s
            )
        ).float()
        diagnostics[:, 8] = torch.where(
            would_reset,
            torch.clamp(
                self.episode_length_s
                - episode_age_s.to(self.device).reshape(-1),
                min=0.0,
            ),
            torch.zeros(self.num_envs, device=self.device),
        )
        diagnostics[:, 9] = self.wall_sequence_steps.float() * self.dt_s
        diagnostics[:, 10] = float(self.term_available)
        diagnostics[:, 11] = float(self.term_config_valid)

        if bool(wall_reset.any()):
            diagnostics[wall_reset] = previous[wall_reset]
            diagnostics[wall_reset, 5] = 1.0
            diagnostics[wall_reset, 7] = (
                seconds_since_push.to(self.device).reshape(-1)[wall_reset]
                < self.push_grace_s
            ).float()
        if bool(reset.any()):
            self.anchor_xy[reset] = root_xy[reset]
            self.anchor_valid[reset] = True
            self.confined_steps[reset] = 0
            self.wall_evidence_age_steps[reset] = self.wall_latch_steps + 1
            self.wall_sequence_steps[reset] = 0
            self.episode_age_steps[reset] = 0
            self.triggered[reset] = False
            self.shadow_event_reported[reset] = False
            counter = getattr(self.env, "_nav_motion_stuck", None)
            if torch.is_tensor(counter) and counter.numel() == self.num_envs:
                counter[reset] = 0.0
        self.last_diagnostics = diagnostics
        return diagnostics


__all__ = ["MotionWallStuckTracker"]
