#!/usr/bin/env python3
"""Deployment-shaped recurrent goal belief used by the P4 high-level actor."""

from __future__ import annotations

import math

import torch

from agent_ppo.feature import nav_contract, p4_contract


class GoalBeliefChainV2:
    """Robust one-second MAP goal estimate propagated by SportMode/IMU feedback.

    The caller may use privileged goal truth only to synthesize measurements.
    Between measurements the state is propagated exclusively with measured
    body velocity and IMU yaw rate; root pose/velocity are intentionally absent.
    """

    HISTORY = 8

    def __init__(self, num_envs: int, device, *, seed: int = 0):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        shape = (self.num_envs,)
        self.estimate = torch.zeros(self.num_envs, 2, device=self.device)
        self.has_estimate = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.age_s = torch.full(shape, 1.0e6, device=self.device)
        self.measurement_clock_s = torch.full(shape, 1.0e6, device=self.device)
        self.dropout_remaining_s = torch.zeros(shape, device=self.device)
        self.jump_remaining_s = torch.zeros(shape, device=self.device)
        self.jump_offset = torch.zeros(self.num_envs, 2, device=self.device)
        self.jump_radial_offset_m = torch.zeros(shape, device=self.device)
        self.jump_tangent_offset_m = torch.zeros(shape, device=self.device)
        self.goal_epoch = torch.full(shape, -1, dtype=torch.long, device=self.device)
        self.reacquire_pending = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.reacquire_elapsed_s = torch.zeros(shape, device=self.device)
        self.last_reacquisition_time_s = torch.zeros(shape, device=self.device)
        self.process_variance_m2 = torch.zeros(shape, device=self.device)
        self.candidate_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.candidate_valid = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.candidate_count = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self.history_xy = torch.zeros(
            self.HISTORY, self.num_envs, 2, device=self.device
        )
        self.history_valid = torch.zeros(
            self.HISTORY, self.num_envs, dtype=torch.bool, device=self.device
        )
        self.history_age_s = torch.full(
            (self.HISTORY, self.num_envs), 1.0e6, device=self.device
        )
        self.history_cursor = 0
        self.fault_scale = 0.0
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def _uniform(self, shape, *, device=None) -> torch.Tensor:
        value = torch.rand(shape, generator=self.generator, device="cpu")
        return value.to(device or self.device)

    def _normal(self, shape, *, device=None) -> torch.Tensor:
        value = torch.randn(shape, generator=self.generator, device="cpu")
        return value.to(device or self.device)

    def set_fault_scale(self, value: float) -> None:
        self.fault_scale = max(0.0, min(1.0, float(value)))

    def reset(self, mask: torch.Tensor) -> None:
        mask = mask.to(self.device).bool().reshape(-1)
        if not bool(mask.any()):
            return
        self.estimate[mask] = 0.0
        self.has_estimate[mask] = False
        self.age_s[mask] = 1.0e6
        self.measurement_clock_s[mask] = 1.0e6
        self.dropout_remaining_s[mask] = 0.0
        self.jump_remaining_s[mask] = 0.0
        self.jump_offset[mask] = 0.0
        self.jump_radial_offset_m[mask] = 0.0
        self.jump_tangent_offset_m[mask] = 0.0
        self.goal_epoch[mask] = -1
        self.reacquire_pending[mask] = False
        self.reacquire_elapsed_s[mask] = 0.0
        self.last_reacquisition_time_s[mask] = 0.0
        self.process_variance_m2[mask] = 0.0
        self.candidate_xy[mask] = 0.0
        self.candidate_valid[mask] = False
        self.candidate_count[mask] = 0
        self.history_xy[:, mask] = 0.0
        self.history_valid[:, mask] = False
        self.history_age_s[:, mask] = 1.0e6

    @staticmethod
    def _propagate(
        goal_xy: torch.Tensor,
        measured_velocity3: torch.Tensor,
        dt_s: float,
        xy_velocity_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        translation = measured_velocity3[:, :2] * float(dt_s)
        if xy_velocity_valid is not None:
            translation = torch.where(
                xy_velocity_valid.reshape(-1, 1),
                translation,
                torch.zeros_like(translation),
            )
        yaw = measured_velocity3[:, 2] * float(dt_s)
        shifted = goal_xy - translation
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        return torch.stack(
            (
                cosine * shifted[:, 0] + sine * shifted[:, 1],
                -sine * shifted[:, 0] + cosine * shifted[:, 1],
            ),
            dim=-1,
        )

    def _robust_map(self) -> torch.Tensor:
        valid = self.history_valid & (self.history_age_s <= 1.0)
        weights = valid.to(self.history_xy.dtype) * torch.exp(
            -self.history_age_s.clamp_min(0.0) / 0.50
        )
        center = (self.history_xy * weights.unsqueeze(-1)).sum(dim=0) / (
            weights.sum(dim=0, keepdim=False).clamp_min(1.0e-6).unsqueeze(-1)
        )
        residual = torch.linalg.vector_norm(
            self.history_xy - center.unsqueeze(0), dim=-1
        )
        huber = torch.where(
            residual <= 0.25,
            torch.ones_like(residual),
            0.25 / residual.clamp_min(1.0e-6),
        )
        robust = weights * huber
        return (self.history_xy * robust.unsqueeze(-1)).sum(dim=0) / (
            robust.sum(dim=0).clamp_min(1.0e-6).unsqueeze(-1)
        )

    def update(
        self,
        true_goal_xy: torch.Tensor,
        measured_velocity3: torch.Tensor,
        *,
        velocity_valid: torch.Tensor,
        dt_s: float,
        reset_mask: torch.Tensor | None = None,
        goal_epoch: torch.Tensor | None = None,
        deterministic: bool = False,
        fault_profile: str = "full",
        fault_allowed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        true_goal_xy = true_goal_xy.to(self.device)
        measured_velocity3 = torch.nan_to_num(measured_velocity3.to(self.device))
        valid_velocity = velocity_valid.to(self.device).bool().reshape(-1)
        if reset_mask is not None:
            self.reset(reset_mask)
        epoch_changed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if goal_epoch is not None:
            incoming = goal_epoch.to(self.device).long().reshape(-1)
            epoch_changed = (self.goal_epoch >= 0) & (incoming != self.goal_epoch)
            if bool(epoch_changed.any()):
                self.reset(epoch_changed)
            self.goal_epoch.copy_(incoming)

        propagated = self._propagate(
            self.estimate,
            measured_velocity3,
            dt_s,
            valid_velocity,
        )
        self.estimate = torch.where(
            self.has_estimate.unsqueeze(-1),
            propagated,
            self.estimate,
        )
        propagated_history = self._propagate(
            self.history_xy.reshape(-1, 2),
            measured_velocity3.unsqueeze(0)
            .expand(self.HISTORY, -1, -1)
            .reshape(-1, 3),
            dt_s,
            valid_velocity.unsqueeze(0)
            .expand(self.HISTORY, -1)
            .reshape(-1),
        ).reshape(self.HISTORY, self.num_envs, 2)
        self.history_xy = torch.where(
            self.history_valid.unsqueeze(-1),
            propagated_history,
            self.history_xy,
        )
        propagated_candidate = self._propagate(
            self.candidate_xy,
            measured_velocity3,
            dt_s,
            valid_velocity,
        )
        self.candidate_xy = torch.where(
            self.candidate_valid.unsqueeze(-1),
            propagated_candidate,
            self.candidate_xy,
        )
        estimate_distance = torch.linalg.vector_norm(self.estimate, dim=-1).clamp(
            0.0, 20.0
        )
        process_q = (
            p4_contract.GOAL_PROCESS_SIGMA_V_M_S * float(dt_s)
        ) ** 2 + (
            estimate_distance
            * p4_contract.GOAL_PROCESS_SIGMA_WZ_RAD_S
            * float(dt_s)
        ).square()
        process_q = torch.where(valid_velocity, process_q, process_q * 4.0)
        self.process_variance_m2 = torch.where(
            self.has_estimate,
            self.process_variance_m2 + process_q,
            self.process_variance_m2,
        )
        self.age_s.add_(float(dt_s))
        self.measurement_clock_s.add_(float(dt_s))
        self.history_age_s.add_(float(dt_s))
        was_dropout = self.dropout_remaining_s > 0.0
        self.dropout_remaining_s.sub_(float(dt_s)).clamp_min_(0.0)
        self.jump_remaining_s.sub_(float(dt_s)).clamp_min_(0.0)
        jump_ended = self.jump_remaining_s <= 0.0
        self.jump_offset[jump_ended] = 0.0
        self.jump_radial_offset_m[jump_ended] = 0.0
        self.jump_tangent_offset_m[jump_ended] = 0.0
        dropout_ended = was_dropout & (self.dropout_remaining_s <= 0.0)
        self.reacquire_pending |= dropout_ended
        self.reacquire_elapsed_s = torch.where(
            self.reacquire_pending,
            self.reacquire_elapsed_s + float(dt_s),
            self.reacquire_elapsed_s,
        )

        if fault_profile not in {"noise_only", "medium", "full", "stress"}:
            raise ValueError(f"unsupported P4 GoalBelief fault profile: {fault_profile!r}")
        fault_allowed = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if fault_allowed_mask is None
            else fault_allowed_mask.to(self.device).bool().reshape(-1)
        )
        fault_enabled = fault_profile != "noise_only"
        allow_long_dropout = fault_profile in {"full", "stress"}
        if not deterministic and self.fault_scale > 0.0 and fault_enabled:
            dropout_rate = 0.04 + (0.005 if allow_long_dropout else 0.0)
            drop_start = (
                self._uniform((self.num_envs,))
                < dropout_rate * float(dt_s) * self.fault_scale
            ) & (self.dropout_remaining_s <= 0.0) & fault_allowed
            if bool(drop_start.any()):
                count = int(drop_start.sum())
                long = (
                    self._uniform((count,)) < (0.005 / 0.045)
                    if allow_long_dropout
                    else torch.zeros(count, dtype=torch.bool, device=self.device)
                )
                short_duration = 0.3 + 0.9 * self._uniform((count,))
                long_duration = 1.2 + 1.3 * self._uniform((count,))
                self.dropout_remaining_s[drop_start] = torch.where(
                    long, long_duration, short_duration
                )
                self.reacquire_pending[drop_start] = False
                self.reacquire_elapsed_s[drop_start] = 0.0
            true_distance = torch.linalg.vector_norm(true_goal_xy, dim=-1)
            jump_start = (
                self._uniform((self.num_envs,))
                < p4_contract.GOAL_JUMP_RATE_PER_S * float(dt_s) * self.fault_scale
            ) & (self.jump_remaining_s <= 0.0) & fault_allowed & (
                true_distance >= p4_contract.GOAL_JUMP_MIN_DISTANCE_M
            )
            if bool(jump_start.any()):
                ids = jump_start.nonzero(as_tuple=False).reshape(-1)
                distance = true_distance[ids]
                capture_x = torch.clamp((distance - 1.5) / 1.5, 0.0, 1.0)
                capture = capture_x.square() * (3.0 - 2.0 * capture_x)
                radial_axis = capture * torch.clamp(
                    0.04 + 0.004 * distance,
                    p4_contract.GOAL_JUMP_RADIAL_RANGE_M[0],
                    p4_contract.GOAL_JUMP_RADIAL_RANGE_M[1],
                )
                tangent_axis = capture * torch.clamp(
                    0.10 + 0.05 * distance,
                    p4_contract.GOAL_JUMP_TANGENT_RANGE_M[0],
                    p4_contract.GOAL_JUMP_TANGENT_RANGE_M[1],
                )
                theta = 2.0 * math.pi * self._uniform((ids.numel(),))
                radius = torch.sqrt(self._uniform((ids.numel(),)))
                radial_component = radius * torch.cos(theta) * radial_axis
                tangent_component = radius * torch.sin(theta) * tangent_axis
                radial_unit = true_goal_xy[ids] / distance.clamp_min(1.0e-6).unsqueeze(-1)
                tangent_unit = torch.stack((-radial_unit[:, 1], radial_unit[:, 0]), dim=-1)
                self.jump_offset[ids] = (
                    radial_component.unsqueeze(-1) * radial_unit
                    + tangent_component.unsqueeze(-1) * tangent_unit
                )
                self.jump_radial_offset_m[ids] = radial_component
                self.jump_tangent_offset_m[ids] = tangent_component
                low, high = p4_contract.GOAL_JUMP_DURATION_S
                self.jump_remaining_s[ids] = low + (high - low) * self._uniform((ids.numel(),))

        due = (self.measurement_clock_s >= 0.20) & (
            self.dropout_remaining_s <= 0.0
        )
        accepted = torch.zeros_like(due)
        clipped = torch.zeros_like(due)
        rejected = torch.zeros_like(due)
        innovation_d2 = torch.zeros(self.num_envs, device=self.device)
        if bool(due.any()):
            ids = due.nonzero(as_tuple=False).reshape(-1)
            source_xy = true_goal_xy[ids]
            source_finite = torch.isfinite(source_xy).all(dim=-1)
            xy = torch.nan_to_num(source_xy)
            distance = torch.linalg.vector_norm(xy, dim=-1)
            bearing = torch.atan2(xy[:, 1], xy[:, 0])
            bearing_std = torch.clamp(0.010 + 0.0025 * distance, 0.010, 0.035)
            distance_std = torch.clamp(0.025 + 0.0125 * distance, 0.025, 0.150)
            if not deterministic:
                bearing = bearing + self._normal((ids.numel(),)) * bearing_std
                distance = torch.clamp_min(
                    distance + self._normal((ids.numel(),)) * distance_std, 0.0
                )
            measured = torch.stack(
                (distance * torch.cos(bearing), distance * torch.sin(bearing)), dim=-1
            )
            active_jump = self.jump_remaining_s[ids] > 0.0
            measured = measured + torch.where(
                active_jump.unsqueeze(-1), self.jump_offset[ids], torch.zeros_like(measured)
            )
            residual = measured - self.estimate[ids]
            variance = distance_std.square() + (distance * bearing_std).square()
            innovation_variance = variance + self.process_variance_m2[ids]
            d2 = residual.square().sum(dim=-1) / innovation_variance.clamp_min(
                1.0e-6
            )
            first = ~self.has_estimate[ids]
            normal = source_finite & (
                first | (d2 <= p4_contract.GOAL_NORMAL_D2) | epoch_changed[ids]
            )
            finite = (~normal) & (d2 <= p4_contract.GOAL_CLIPPED_D2)
            finite &= source_finite
            reject = source_finite & ~(normal | finite)

            # A persistent, internally consistent alternative is allowed to
            # replace a stale belief only after five 5 Hz observations.  This
            # is deliberately longer than the injected 0.2--0.8 s jump burst.
            reject_ids = ids[reject]
            if reject_ids.numel() > 0:
                newly_pending = ~self.reacquire_pending[reject_ids]
                self.reacquire_pending[reject_ids] = True
                self.reacquire_elapsed_s[reject_ids[newly_pending]] = 0.0
                reject_measured = measured[reject]
                candidate_delta = torch.linalg.vector_norm(
                    reject_measured - self.candidate_xy[reject_ids], dim=-1
                )
                consistency = torch.maximum(
                    torch.full_like(candidate_delta, 0.20),
                    3.0 * torch.sqrt(variance[reject].clamp_min(1.0e-6)),
                )
                same = self.candidate_valid[reject_ids] & (
                    candidate_delta <= consistency
                )
                previous_count = self.candidate_count[reject_ids]
                next_count = torch.where(
                    same, previous_count + 1, torch.ones_like(previous_count)
                )
                blend = 1.0 / next_count.to(reject_measured).clamp_min(1.0)
                blended = self.candidate_xy[reject_ids] + blend.unsqueeze(-1) * (
                    reject_measured - self.candidate_xy[reject_ids]
                )
                self.candidate_xy[reject_ids] = torch.where(
                    same.unsqueeze(-1), blended, reject_measured
                )
                self.candidate_valid[reject_ids] = True
                self.candidate_count[reject_ids] = next_count
                persistent = next_count >= p4_contract.GOAL_REACQUIRE_SAMPLES
                if bool(persistent.any()):
                    persistent_local = reject.nonzero(as_tuple=False).reshape(-1)[
                        persistent
                    ]
                    normal[persistent_local] = True
                    reject[persistent_local] = False
                    measured[persistent_local] = self.candidate_xy[
                        reject_ids[persistent]
                    ]
                    residual[persistent_local] = (
                        measured[persistent_local] - self.estimate[ids[persistent_local]]
                    )

            invalid_source = ~source_finite
            if bool(invalid_source.any()):
                invalid_ids = ids[invalid_source]
                self.candidate_valid[invalid_ids] = False
                self.candidate_count[invalid_ids] = 0
            clipped_residual = residual * torch.minimum(
                torch.ones_like(d2),
                torch.sqrt(
                    torch.full_like(d2, p4_contract.GOAL_NORMAL_D2)
                    / d2.clamp_min(1.0e-6)
                ),
            ).unsqueeze(-1)
            assimilated = torch.where(
                finite.unsqueeze(-1), self.estimate[ids] + clipped_residual, measured
            )
            use = normal | finite
            write_ids = ids[use]
            self.history_xy[self.history_cursor, write_ids] = assimilated[use]
            self.history_valid[self.history_cursor, write_ids] = True
            self.history_age_s[self.history_cursor, write_ids] = 0.0
            self.has_estimate[write_ids] = True
            self.age_s[write_ids] = 0.0
            self.process_variance_m2[write_ids] = variance[use]
            self.candidate_valid[write_ids] = False
            self.candidate_count[write_ids] = 0
            reacquired = use & self.reacquire_pending[ids]
            if bool(reacquired.any()):
                reacquired_ids = ids[reacquired]
                self.last_reacquisition_time_s[reacquired_ids] = (
                    self.reacquire_elapsed_s[reacquired_ids]
                )
                self.reacquire_pending[reacquired_ids] = False
                self.reacquire_elapsed_s[reacquired_ids] = 0.0
            self.measurement_clock_s[ids] = 0.0
            accepted[ids[normal]] = True
            clipped[ids[finite]] = True
            rejected[ids[reject | invalid_source]] = True
            innovation_d2[ids] = d2
            self.history_cursor = (self.history_cursor + 1) % self.HISTORY
            robust = self._robust_map()
            self.estimate = torch.where(
                self.has_estimate.unsqueeze(-1), robust, self.estimate
            )

        goal4 = self.goal4()
        self.last_diagnostics = {
            "goal_innovation_d2": innovation_d2.detach(),
            "goal_measurement_accepted": accepted.float(),
            "goal_measurement_clipped": clipped.float(),
            "goal_measurement_rejected": rejected.float(),
            "goal_age_s": self.age_s.detach().clone(),
            "goal_dropout_active": (self.dropout_remaining_s > 0.0).float(),
            "goal_jump_active": (self.jump_remaining_s > 0.0).float(),
            "goal_jump_radial_offset_m": self.jump_radial_offset_m.detach().clone(),
            "goal_jump_tangent_offset_m": self.jump_tangent_offset_m.detach().clone(),
            "goal_map_x_m": self.estimate[:, 0].detach().clone(),
            "goal_map_y_m": self.estimate[:, 1].detach().clone(),
            "goal_map_distance_m": torch.linalg.vector_norm(
                self.estimate, dim=-1
            ).detach(),
            "goal_propagated": self.has_estimate.float(),
            "goal_epoch_changed": epoch_changed.float(),
            "goal_reacquire_pending": self.reacquire_pending.float(),
            "goal_reacquisition_time_s": self.last_reacquisition_time_s.detach().clone(),
            "goal_fault_allowed": fault_allowed.float(),
            "goal_freshness": goal4[:, 3].detach().clone(),
            "goal_process_variance_m2": self.process_variance_m2.detach().clone(),
            "goal_candidate_count": self.candidate_count.float().detach().clone(),
            "goal_stale_low_speed_active": (
                self.has_estimate & (self.age_s > p4_contract.GOAL_AGE_HOLD_END_S)
            ).float(),
        }
        return goal4

    def goal4(self) -> torch.Tensor:
        encoded_xy = nav_contract.encode_goal_xy_direction_preserving(
            self.estimate
        )
        distance = torch.linalg.vector_norm(self.estimate, dim=-1)
        encoded_distance = torch.clamp(
            distance / nav_contract.GOAL_DIST_SCALE_M, 0.0, 1.0
        )
        freshness = torch.where(
            self.age_s <= p4_contract.GOAL_AGE_SLOW_START_S,
            torch.ones_like(self.age_s),
            (
                (p4_contract.GOAL_AGE_HOLD_END_S - self.age_s)
                / (
                    p4_contract.GOAL_AGE_HOLD_END_S
                    - p4_contract.GOAL_AGE_SLOW_START_S
                )
            ).clamp(0.0, 1.0),
        )
        freshness = torch.where(
            self.has_estimate, freshness, torch.zeros_like(freshness)
        )
        freshness = torch.where(
            self.has_estimate & (freshness <= 0.0),
            torch.full_like(freshness, p4_contract.GOAL_FRESHNESS_FLOOR),
            freshness,
        )
        result = torch.cat(
            (encoded_xy, encoded_distance.unsqueeze(-1), freshness.unsqueeze(-1)),
            dim=-1,
        )
        result[~self.has_estimate] = 0.0
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "version": p4_contract.GOAL_BELIEF_VERSION,
            "goal_encoding_version": nav_contract.GOAL_ENCODING_VERSION,
            "generator_state": self.generator.get_state(),
            "fault_scale": self.fault_scale,
            # Live per-environment estimator state is intentionally not exact-
            # resumed; workflow resets episodes and recurrent state together.
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("version") != p4_contract.GOAL_BELIEF_VERSION:
            raise ValueError("P4 GoalBelief checkpoint version mismatch")
        if state.get("goal_encoding_version") != nav_contract.GOAL_ENCODING_VERSION:
            raise ValueError("P4 GoalBelief goal-encoding version mismatch")
        generator_state = state.get("generator_state")
        if not torch.is_tensor(generator_state):
            raise ValueError("P4 GoalBelief checkpoint missing RNG state")
        self.generator.set_state(generator_state.cpu())
        self.fault_scale = float(state.get("fault_scale", 0.0))
        self.reset(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
