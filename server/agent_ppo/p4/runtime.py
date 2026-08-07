#!/usr/bin/env python3
"""Runtime transport, camera, capability, and command mapping for P4."""

from __future__ import annotations

from agent_ppo.p4.constants import *  # noqa: F403

import torch

from agent_ppo.feature import p2_contract, p3_contract, p4_contract
from agent_ppo.feature.p2_response_buffer import split_p2_transport
from agent_ppo.p4.profiles import PROFILE_MAZE_CLOSED_LOOP_V3


class P4RuntimeMixin:
    """Behavior-preserving methods extracted from AlgorithmP4NavPPO."""

    def _split_transport(self, critic_wire: torch.Tensor):
        if critic_wire.ndim != 2:
            raise ValueError("P4 privileged transport must be rank-2")
        if critic_wire.shape[1] == p2_contract.PRIVILEGED_WIRE_DIM:
            self._p4_extra.zero_()
            self._p4_worker_extra.zero_()
            return split_p2_transport(critic_wire)
        if critic_wire.shape[1] != p4_contract.P4_PRIVILEGED_WIRE_DIM:
            raise ValueError(
                "P4 privileged wire must be 385 eval columns or 519 training "
                f"columns, got {tuple(critic_wire.shape)}"
            )
        base = critic_wire[:, : p2_contract.PRIVILEGED_WIRE_DIM]
        extra = critic_wire[
            :,
            p2_contract.PRIVILEGED_WIRE_DIM : p3_contract.P3_PRIVILEGED_WIRE_DIM,
        ]
        p4_extra = critic_wire[:, p3_contract.P3_PRIVILEGED_WIRE_DIM :]
        critic, aux = split_p2_transport(base)
        self._p4_extra = extra.to(self.device)
        self._p4_worker_extra = p4_extra.to(self.device)
        worker_step = torch.nan_to_num(
            aux[:, 26], nan=-1.0, posinf=-1.0, neginf=-1.0
        ).long()
        fresh_transport = worker_step != self._last_push_transport_step
        push = (
            (extra[:, p3_contract.PUSH_EVENT_FLAG_INDEX] > 0.5)
            & fresh_transport
        )
        self.push_epoch += push.long()
        delta = extra[:, p3_contract.PUSH_DELTA_VELOCITY_SLICE].to(self.device)
        self._push_interval_count += push.float()
        self._push_lifetime_count += push.float()
        self._push_rollout_count += push.float()
        self._push_env_seen |= push
        self._push_interval_delta_sum += torch.where(
            push.unsqueeze(-1), delta, torch.zeros_like(delta)
        )
        self._push_interval_delta_abs_max = torch.maximum(
            self._push_interval_delta_abs_max,
            torch.where(push.unsqueeze(-1), delta.abs(), torch.zeros_like(delta)),
        )
        self._push_recovery_pending |= push
        self._push_recovery_time[push] = float("nan")
        self._last_push_transport_step.copy_(worker_step)
        self.seconds_since_push.copy_(
            torch.nan_to_num(
                extra[:, p3_contract.SECONDS_SINCE_PUSH_INDEX],
                nan=1.0e6,
                posinf=1.0e6,
                neginf=0.0,
            )
        )
        return critic, aux

    def _prepare_policy_parts(self, parts, critic_obs, aux, reset):
        worker_reset = aux[:, 24] > 0.5
        self._previous_policy_command[worker_reset | reset] = 0.0
        self._previous_exec_command[worker_reset | reset] = 0.0
        self._last_policy_command[worker_reset | reset] = 0.0
        self._last_limited_command[worker_reset | reset] = 0.0
        self._translation_alpha_prev[worker_reset | reset] = 1.0
        self.user_speed_cap.fill_(p4_contract.P4_MAX_VX)
        if self.training_enabled:
            true_xy = self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE]
            if true_xy.shape != (self.num_envs, 2):
                raise RuntimeError(
                    f"P4 raw metric goal transport drift: {tuple(true_xy.shape)}"
                )
            measured = aux[:, 6:9].clone()
            xy_valid = aux[:, 9] > 0.5
            measured[~xy_valid, :2] = 0.0
            schedule = p4_contract.training_schedule(
                self.session_effective_seconds,
                branch=self._effective_maze_branch(self.session_effective_seconds),
            )
            self.goal_belief.set_fault_scale(
                float(schedule["goal_fault_multiplier"])
                if self.goal_fault_course_enabled
                else 0.0
            )
            if self.session_effective_seconds < 1_800.0:
                goal_fault_profile = "noise_only"
            elif self.session_effective_seconds < 21_600.0:
                goal_fault_profile = "medium"
            elif self.session_effective_seconds < 27_000.0:
                goal_fault_profile = "full"
            else:
                goal_fault_profile = "stress"
            fault_allowed = torch.ones_like(xy_valid)
            if goal_fault_profile == "full":
                # During 5-7h, keep the 5% severe-camera bucket from also
                # receiving a long Goal dropout. Combined severe faults are
                # reserved for the final stress phase.
                fault_allowed &= self.camera_state.sequence_kind != 3
            parts["goal4"] = self.goal_belief.update(
                true_xy,
                measured,
                velocity_valid=xy_valid,
                dt_s=p2_contract.CONTROL_DT_S,
                reset_mask=worker_reset,
                deterministic=False,
                fault_profile=(
                    goal_fault_profile
                    if self.goal_fault_course_enabled
                    else "noise_only"
                ),
                fault_allowed_mask=fault_allowed,
            )
            self._goal_epoch_changed_since_tick |= (
                self.goal_belief.last_diagnostics["goal_epoch_changed"] > 0.5
            )
        delivered, diagnostics = self.camera_state.process(
            parts["depth"],
            reset_mask=worker_reset,
            session_effective_seconds=self.session_effective_seconds,
            training=(self.training_enabled and self.camera_fault_course_enabled),
        )
        self._clean_depth = self.camera_state.clean_capture.detach().clone()
        # This read-only view is consumed immediately by _map_policy_target()
        # before env.step() can recycle the observation buffer.
        self._delivered_depth = delivered.detach()
        parts["depth"] = delivered
        if self.command.command_transition_mode == "instant_hold_10hz":
            # R4 exposes the hard mapper range directly. Goal freshness and
            # predictive risk remain separate observations/diagnostics; they
            # must not advertise a smaller capability than the range that
            # actually maps policy_target directly to exec.
            self.safety_speed_cap.fill_(1.0)
            self.effective_speed_cap.copy_(self.user_speed_cap)
        else:
            self.effective_speed_cap = p4_contract.effective_speed_cap(
                self.user_speed_cap,
                parts["goal4"][:, 3],
                self.safety_speed_cap,
            )
        self._camera_diagnostics = diagnostics
        return parts

    def _low_level_frame(self, parts, critic_obs):
        del critic_obs
        frame_id = self._camera_diagnostics.get("camera_frame_id")
        if frame_id is None:
            frame_id = torch.full(
                (self.num_envs,), -1.0, device=self.device
            )
        frame_id = frame_id.long()
        changed = frame_id != self._cached_low_frame_id
        with torch.inference_mode():
            if bool(changed.any()):
                self._cached_low_cnn[changed] = self.low_level_encoder.cnn(
                    parts["depth"][changed]
                )
                self._cached_low_frame_id[changed] = frame_id[changed]
            latent = self.low_level_encoder.forward_from_cnn_features(
                self._cached_low_cnn,
                parts["proprio"],
                masks=None,
            )
            action = self.low_level_actor(
                torch.cat((parts["proprio"], latent), dim=-1)
            )
        return action, {
            "camera_frame_id": frame_id,
            "low_cnn_recomputed": changed.float(),
        }

    def _nav_capability(self, batch: int, *, dtype=torch.float32) -> torch.Tensor:
        if self.command.command_transition_mode == "instant_hold_10hz":
            cap = torch.full((batch,), p4_contract.P4_MAX_VX, device=self.device)
        elif batch != self.num_envs:
            cap = torch.full((batch,), p4_contract.P4_MAX_VX, device=self.device)
        else:
            cap = self.effective_speed_cap
        capability_values = (
            p4_contract.INSTANT_ACTOR_CAPABILITY_PROFILE15
            if self.command.command_transition_mode == "instant_hold_10hz"
            else p2_contract.NAV_CAPABILITY_PROFILE15
        )
        result = torch.tensor(
            capability_values,
            device=self.device,
            dtype=dtype,
        ).expand(batch, -1).clone()
        result[:, 3:6] = torch.tensor(
            (0.0, -p4_contract.P4_MAX_ABS_VY, -p4_contract.P4_MAX_ABS_WZ),
            device=self.device,
            dtype=dtype,
        )
        result[:, 6] = cap.to(dtype)
        result[:, 7] = p4_contract.P4_MAX_ABS_VY
        result[:, 8] = p4_contract.P4_MAX_ABS_WZ
        if self.command.command_transition_mode != "instant_hold_10hz":
            result[:, 9:12] = result.new_tensor(self.command_slew_rate)
            result[:, 12:15] = result.new_tensor(self.command_slew_release_rate)
        return result

    def _response_capability(self, batch: int, *, dtype=torch.float32) -> torch.Tensor:
        result = super()._response_capability(batch, dtype=dtype).clone()
        result[:, 3] = p4_contract.P4_MAX_VX
        result[:, 4] = p4_contract.P4_MAX_ABS_WZ
        result[:, 7] = p4_contract.P4_MAX_ABS_VY
        return result

    def _map_policy_target(self, normalized, legacy_target, *, goal4, aux):
        del legacy_target, aux
        self._previous_policy_command.copy_(self._last_policy_command)
        # The base algorithm may sanitize a non-finite sampled target after
        # this hook returns.  Keep a separate physical history for the instant
        # parent anchor without changing legacy policy-target diagnostics.
        self._previous_exec_command.copy_(self.command.exec_cmd)
        full_cap = torch.full(
            (normalized.shape[0],), p4_contract.P4_MAX_VX,
            device=normalized.device, dtype=normalized.dtype,
        )
        policy = p4_contract.map_normalized_action(
            normalized, full_cap, goal_freshness=None
        )
        instant_mode = self.command.command_transition_mode == "instant_hold_10hz"
        if instant_mode:
            if self._delivered_depth is None:
                raise RuntimeError("P4 predictive reward missing current delivered depth")
            with torch.inference_mode():
                current_risk = p2_contract.predictive_collision_risk_penalty(
                    self._delivered_depth, policy
                )[3]
            ones = torch.ones(policy.shape[0], device=policy.device, dtype=policy.dtype)
            zeros = torch.zeros_like(ones)
            self._safety_cap_predictive_risk = current_risk.detach()
            self.safety_speed_cap = ones
            self._translation_alpha_prev.fill_(1.0)
            self._translation_limiter_diagnostics = {
                "translation_safety_alpha_raw": ones,
                "translation_safety_alpha": ones,
                "translation_limiter_shadow": ones,
                "translation_limiter_disabled": ones,
            }
            self._near_goal_capture_diagnostics = {
                "near_goal_capture_candidate": zeros,
                "near_goal_capture_active": zeros,
                "near_goal_final_translation_alpha": ones,
            }
            self.effective_speed_cap = full_cap
            self._last_normalized_action = normalized.detach()
            self._last_policy_command = policy.detach()
            self._last_limited_command = policy.detach()
            self._last_mapped_command = policy.detach()
            self._last_goal_freshness = goal4[:, 3].detach()
            return policy
        stale_cap = p4_contract.stale_goal_cap(self.user_speed_cap, goal4[:, 3])
        stale_limited = p4_contract.map_normalized_action(
            normalized, stale_cap, goal4[:, 3]
        )
        if self._delivered_depth is None:
            raise RuntimeError("P4 translation limiter missing current delivered depth")
        # Risk must correspond to the candidate sampled for this transition.
        # Using command.active_target here would lag by one 10 Hz decision and
        # can protect the direction that the policy has already abandoned.
        with torch.inference_mode():
            candidate_arc = self._predictive_command(stale_limited)
            current_risk = p2_contract.predictive_collision_risk_penalty(
                self._delivered_depth, candidate_arc
            )[3]
        self._safety_cap_predictive_risk = current_risk.detach()
        command_contract = p4_contract.command_contract(self.training_profile)
        limiter_contract = command_contract["translation_vector_limiter"]
        limiter_limited, limiter = p4_contract.translation_vector_limiter(
            stale_limited,
            current_risk,
            self._translation_alpha_prev,
            reset_mask=self.reset_since_tick,
            risk_threshold=float(limiter_contract["risk_threshold"]),
            alpha_floor=float(limiter_contract["alpha_floor"]),
        )
        self.safety_speed_cap = limiter["translation_safety_alpha_raw"].detach()
        limiter_shadow = self.training_profile == PROFILE_MAZE_CLOSED_LOOP_V3
        applied_alpha = (
            torch.ones_like(limiter["translation_safety_alpha"])
            if limiter_shadow
            else limiter["translation_safety_alpha"]
        )
        if limiter_shadow:
            limited = stale_limited
            _, capture = p4_contract.near_goal_capture(
                limited,
                self.goal_belief.estimate,
                goal4[:, 3],
                applied_alpha,
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                self.reset_since_tick,
                self._goal_epoch_changed_since_tick,
            )
            capture["near_goal_capture_active"] = torch.zeros_like(
                capture["near_goal_capture_active"]
            )
            capture["near_goal_final_translation_alpha"] = applied_alpha
        else:
            limited, capture = p4_contract.near_goal_capture(
                limiter_limited,
                self.goal_belief.estimate,
                goal4[:, 3],
                applied_alpha,
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                self.reset_since_tick,
                self._goal_epoch_changed_since_tick,
            )
        self._translation_alpha_prev.copy_(applied_alpha)
        self._translation_limiter_diagnostics = {
            name: value.detach() for name, value in limiter.items()
        }
        self._translation_limiter_diagnostics["translation_safety_alpha"] = (
            applied_alpha.detach()
        )
        self._translation_limiter_diagnostics["translation_limiter_shadow"] = (
            torch.full_like(applied_alpha, float(limiter_shadow))
        )
        self._near_goal_capture_diagnostics = {
            name: value.detach() for name, value in capture.items()
        }
        self.effective_speed_cap = stale_cap * applied_alpha
        self._last_normalized_action = normalized.detach()
        self._last_policy_command = policy.detach()
        self._last_limited_command = limited.detach()
        self._last_mapped_command = limited.detach()
        self._last_goal_freshness = goal4[:, 3].detach()
        return limited

    def _command_rate_weight(self) -> float:
        if self.training_profile == p4_contract.INSTANT_REPAIR_PROFILE:
            return float(p4_contract.INSTANT_REPAIR_COMMAND_RATE_WEIGHT)
        return super()._command_rate_weight()

    def _tracking_error_weight(self) -> float:
        if self.training_profile == p4_contract.INSTANT_REPAIR_PROFILE:
            return float(p4_contract.INSTANT_REPAIR_TRACKING_ERROR_WEIGHT)
        return super()._tracking_error_weight()

    def _predictive_command(self, target: torch.Tensor) -> torch.Tensor:
        if self.command.command_transition_mode == "instant_hold_10hz":
            return target
        # Integrate the same 50 Hz slew/reversal rule as the live controller
        # and return the mean velocity over the 0.8 s arc.  The downstream
        # depth-sector model uses this mean for both travel and yaw curvature.
        current = self.command.exec_cmd.clone()
        up = torch.tensor(self.command_slew_rate, device=self.device).reshape(1, 3)
        release = torch.tensor(
            self.command_slew_release_rate, device=self.device
        ).reshape(1, 3)
        frames = max(
            1,
            int(round(
                p2_contract.PREDICTIVE_COLLISION_LOOKAHEAD_S
                / p2_contract.CONTROL_DT_S
            )),
        )
        accumulated = torch.zeros_like(current)
        for _ in range(frames):
            opposite = target * current < 0.0
            reducing = target.abs() < current.abs()
            rate = torch.where(opposite | reducing, release, up)
            effective_target = torch.where(opposite, torch.zeros_like(target), target)
            delta = torch.clamp(
                effective_target - current,
                -rate * p2_contract.CONTROL_DT_S,
                rate * p2_contract.CONTROL_DT_S,
            )
            previous = current
            current = current + delta
            crossed_zero = opposite & (current * previous <= 0.0)
            current = torch.where(crossed_zero, torch.zeros_like(current), current)
            accumulated += current
        return accumulated / float(frames)

    def _response_append_kwargs(self) -> dict[str, object]:
        return {
            "push_epoch": self.push_epoch,
            "seconds_since_push": self.seconds_since_push,
        }


def map_normalized_action(
    normalized_action: torch.Tensor,
    effective_max_vx: torch.Tensor | float,
    goal_freshness: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map normalized PPO coordinates without changing their log-probability."""
    if normalized_action.shape[-1] != 3:
        raise ValueError("P4 normalized action must end in three coordinates")
    bounded = torch.nan_to_num(
        torch.clamp(normalized_action, -1.0, 1.0), nan=0.0, posinf=1.0, neginf=-1.0
    )
    cap = torch.as_tensor(
        effective_max_vx, device=bounded.device, dtype=bounded.dtype
    )
    if cap.ndim == bounded.ndim - 1:
        cap = cap.unsqueeze(-1)
    cap = torch.clamp(cap, 0.0, P4_MAX_VX)
    vx = 0.5 * cap * (bounded[..., 0:1] + 1.0)
    vy = P4_MAX_ABS_VY * bounded[..., 1:2]
    wz = P4_MAX_ABS_WZ * bounded[..., 2:3]
    if goal_freshness is not None:
        freshness = torch.as_tensor(
            goal_freshness, device=bounded.device, dtype=bounded.dtype
        )
        if freshness.ndim == bounded.ndim - 1:
            freshness = freshness.unsqueeze(-1)
        waiting = freshness <= 0.0
        stale = (freshness > 0.0) & (
            freshness <= GOAL_FRESHNESS_FLOOR + 1.0e-6
        )
        vy = torch.where(waiting, torch.zeros_like(vy), vy)
        vy = torch.where(
            stale,
            torch.clamp(
                vy,
                -STALE_GOAL_WAIT_MAX_ABS_VY,
                STALE_GOAL_WAIT_MAX_ABS_VY,
            ),
            vy,
        )
        wz = torch.where(
            waiting | stale,
            torch.clamp(wz, -STALE_GOAL_WAIT_MAX_ABS_WZ, STALE_GOAL_WAIT_MAX_ABS_WZ),
            wz,
        )
    return torch.cat((vx, vy, wz), dim=-1)

def map_normalized_action_legacy(normalized_action: torch.Tensor) -> torch.Tensor:
    """Versioned parent mapper used only for migration/equality validation."""
    return p2_contract.map_normalized_action(normalized_action, hard_abs_vy=0.40)

def instant_parent_anchor_reachable(
    parent_target_cmd3: torch.Tensor,
    previous_exec_cmd3: torch.Tensor,
) -> torch.Tensor:
    """Return rows where the legacy parent target was reachable in one nav tick.

    The immutable parent Actor predicts a target that used to pass through the
    50 Hz slew controller.  Anchoring an instant policy to a target that the old
    controller could not physically reach in the same 10 Hz interval preserves
    the wrong transition semantics.  Only smooth, non-reversing parent targets
    are therefore eligible for the distribution anchor.
    """
    parent = torch.as_tensor(parent_target_cmd3)
    previous = torch.as_tensor(
        previous_exec_cmd3, device=parent.device, dtype=parent.dtype
    )
    if parent.ndim != 2 or parent.shape[1] != 3 or previous.shape != parent.shape:
        raise ValueError("P4 instant parent anchor expects matching [N,3] commands")
    finite = torch.isfinite(parent).all(dim=-1) & torch.isfinite(previous).all(dim=-1)
    opposite = parent * previous < 0.0
    growing = parent.abs() > previous.abs()
    up = parent.new_tensor(INSTANT_PARENT_ANCHOR_SLEW_UP) * P4_NAV_DT_S
    release = parent.new_tensor(INSTANT_PARENT_ANCHOR_SLEW_RELEASE) * P4_NAV_DT_S
    reachable_delta = torch.where(growing, up, release)
    within_delta = (parent - previous).abs() <= reachable_delta + 1.0e-6
    return finite & ~opposite.any(dim=-1) & within_delta.all(dim=-1)

def stale_goal_cap(user_cap: torch.Tensor, goal_freshness: torch.Tensor) -> torch.Tensor:
    """Apply the GoalBelief age contract encoded by goal4 freshness."""
    cap = torch.clamp(user_cap, 0.0, P4_MAX_VX)
    freshness = torch.clamp(goal_freshness, 0.0, 1.0)
    # freshness=1 through 0.5 s, then linearly reaches 0 at 2.5 s.
    age = GOAL_AGE_SLOW_START_S + (1.0 - freshness) * (
        GOAL_AGE_HOLD_END_S - GOAL_AGE_SLOW_START_S
    )
    slow_ratio = torch.clamp(
        (age - GOAL_AGE_SLOW_START_S)
        / (GOAL_AGE_SLOW_END_S - GOAL_AGE_SLOW_START_S),
        0.0,
        1.0,
    )
    slowing = cap + slow_ratio * (torch.minimum(cap, torch.full_like(cap, 0.25)) - cap)
    holding = torch.minimum(cap, torch.full_like(cap, 0.20))
    result = torch.where(age <= GOAL_AGE_SLOW_END_S, slowing, holding)
    return torch.where(freshness > 0.0, result, torch.zeros_like(result))

def effective_speed_cap(
    user_cap: torch.Tensor,
    goal_freshness: torch.Tensor,
    safety_cap: torch.Tensor | float,
) -> torch.Tensor:
    goal_cap = stale_goal_cap(user_cap, goal_freshness)
    safety = torch.as_tensor(safety_cap, device=user_cap.device, dtype=user_cap.dtype)
    return torch.minimum(
        torch.minimum(goal_cap, torch.clamp(safety, 0.0, P4_MAX_VX)),
        torch.full_like(goal_cap, P4_MAX_VX),
    )

def translation_vector_limiter(
    policy_cmd3: torch.Tensor,
    predictive_risk: torch.Tensor,
    alpha_prev: torch.Tensor,
    *,
    reset_mask: torch.Tensor | None = None,
    risk_threshold: float = TRANSLATION_LIMITER_RISK_THRESHOLD,
    alpha_floor: float = TRANSLATION_LIMITER_ALPHA_FLOOR,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Limit the complete translational vector using deployable depth risk.

    ``alpha_prev`` is caller-owned live state.  This function deliberately does
    not retain it, so reset/resume can restore the prescribed alpha=1.0 without
    putting limiter state in a checkpoint.
    """
    if policy_cmd3.ndim != 2 or policy_cmd3.shape[1] != 3:
        raise ValueError("P4 translation limiter expects policy_cmd3=[N,3]")
    count = policy_cmd3.shape[0]
    risk = torch.as_tensor(
        predictive_risk, device=policy_cmd3.device, dtype=policy_cmd3.dtype
    ).reshape(-1)
    previous = torch.as_tensor(
        alpha_prev, device=policy_cmd3.device, dtype=policy_cmd3.dtype
    ).reshape(-1)
    if risk.numel() != count or previous.numel() != count:
        raise ValueError("P4 translation limiter batch shape drift")
    if reset_mask is None:
        reset = torch.zeros(count, device=policy_cmd3.device, dtype=torch.bool)
    else:
        reset = torch.as_tensor(reset_mask, device=policy_cmd3.device).reshape(-1).bool()
        if reset.numel() != count:
            raise ValueError("P4 translation limiter reset shape drift")
    risk = torch.nan_to_num(risk, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    threshold = float(risk_threshold)
    floor = float(alpha_floor)
    if not 0.0 <= threshold < 1.0 or not 0.0 < floor <= 1.0:
        raise ValueError("P4 translation limiter threshold/floor are invalid")
    emergency = torch.clamp(
        (risk - threshold)
        / (1.0 - threshold),
        0.0,
        1.0,
    )
    raw_alpha = 1.0 - (1.0 - floor) * emergency
    previous = torch.nan_to_num(previous, nan=1.0, posinf=1.0, neginf=1.0).clamp(
        floor, 1.0
    )
    previous = torch.where(reset, torch.ones_like(previous), previous)
    # Tightening is immediate; only release is rate limited at the 10 Hz tick.
    alpha = torch.minimum(
        raw_alpha,
        previous + TRANSLATION_LIMITER_RELEASE_PER_TICK,
    )
    limited = policy_cmd3.clone()
    limited[:, :2] *= alpha.unsqueeze(-1)
    return limited, {
        "translation_safety_alpha_raw": raw_alpha,
        "translation_safety_alpha": alpha,
        "translation_safety_risk": risk,
        "translation_safety_emergency": emergency,
    }

def near_goal_capture(
    policy_cmd3: torch.Tensor,
    goal_xy_m: torch.Tensor,
    goal_freshness: torch.Tensor,
    safety_alpha: torch.Tensor,
    terminal: torch.Tensor,
    reset: torch.Tensor,
    goal_epoch_changed: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the near-goal translation cap without changing direction or yaw."""
    if policy_cmd3.ndim != 2 or policy_cmd3.shape[1] != 3:
        raise ValueError("P4 near-goal capture expects policy_cmd3=[N,3]")
    if goal_xy_m.shape != (policy_cmd3.shape[0], 2):
        raise ValueError("P4 near-goal capture expects goal_xy_m=[N,2]")
    count = policy_cmd3.shape[0]

    def _flat(value: torch.Tensor, name: str, *, boolean: bool = False) -> torch.Tensor:
        result = torch.as_tensor(value, device=policy_cmd3.device).reshape(-1)
        if result.numel() != count:
            raise ValueError(f"P4 near-goal capture {name} shape drift")
        return result.bool() if boolean else result.to(dtype=policy_cmd3.dtype)

    freshness = _flat(goal_freshness, "freshness").clamp(0.0, 1.0)
    alpha_safe = _flat(safety_alpha, "safety_alpha").clamp(0.0, 1.0)
    terminal_mask = _flat(terminal, "terminal", boolean=True)
    reset_mask = _flat(reset, "reset", boolean=True)
    epoch_changed = _flat(goal_epoch_changed, "goal_epoch_changed", boolean=True)
    goal = torch.nan_to_num(goal_xy_m.to(policy_cmd3), nan=0.0, posinf=0.0, neginf=0.0)
    policy_xy = policy_cmd3[:, :2]
    speed = torch.linalg.vector_norm(policy_xy, dim=-1)
    distance = torch.linalg.vector_norm(goal, dim=-1)
    goal_unit = goal / distance.unsqueeze(-1).clamp_min(1.0e-6)
    policy_unit = policy_xy / speed.unsqueeze(-1).clamp_min(1.0e-6)
    alignment = (policy_unit * goal_unit).sum(dim=-1)
    candidate = (
        (freshness >= NEAR_GOAL_CAPTURE_FRESHNESS_MIN)
        & ~terminal_mask
        & ~reset_mask
        & ~epoch_changed
        & (distance > NEAR_GOAL_CAPTURE_MIN_DISTANCE_M)
        & (distance < NEAR_GOAL_CAPTURE_MAX_DISTANCE_M)
        & (speed > NEAR_GOAL_CAPTURE_MIN_SPEED_M_S)
        & (alignment >= NEAR_GOAL_CAPTURE_GOAL_COSINE_MIN)
    )
    capture_cap = 0.10 + 0.35 * torch.clamp(
        (distance - 0.70) / 0.50, min=0.0, max=1.0
    )
    alpha_capture = torch.minimum(torch.ones_like(speed), capture_cap / speed.clamp_min(1.0e-6))
    applied_alpha = torch.where(
        candidate, torch.minimum(alpha_safe, alpha_capture), alpha_safe
    )
    limited = policy_cmd3.clone()
    limited[:, :2] *= applied_alpha.unsqueeze(-1)
    return limited, {
        "near_goal_capture_candidate": candidate.float(),
        "near_goal_capture_active": candidate.float(),
        "near_goal_capture_distance_m": distance,
        "near_goal_capture_alignment": alignment,
        "near_goal_capture_cap_m_s": capture_cap,
        "near_goal_capture_alpha": torch.where(
            candidate, alpha_capture, torch.ones_like(alpha_capture)
        ),
        "near_goal_final_translation_alpha": applied_alpha,
    }
