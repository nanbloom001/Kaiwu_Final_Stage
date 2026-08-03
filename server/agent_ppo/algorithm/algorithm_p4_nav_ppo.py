#!/usr/bin/env python3
"""P4 Track navigation PPO over a structurally complete frozen P3 parent."""

from __future__ import annotations

import copy
import os
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch import nn

from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
from agent_ppo.checkpoint_io import (
    normalize_kaiwu_train_bundle,
    validate_p3_eval_bundle,
    validate_state_dict_finite,
)
from agent_ppo.feature import nav_contract, p2_contract, p3_contract, p4_contract
from agent_ppo.feature.feedback_emulator import feedback_implementation_digest
from agent_ppo.feature.p2_response_buffer import split_p2_transport
from agent_ppo.feature.p4_camera import P4SharedCameraState
from agent_ppo.feature.p4_goal_belief import GoalBeliefChainV2
from agent_ppo.model.p2_high_level import (
    assemble_actor_input,
    navigation_actor_spec,
    navigation_encoder_spec,
    navigation_safety_head_spec,
)
from agent_ppo.model.response_adapter import response_adapter_spec


class AlgorithmP4NavPPO(AlgorithmP2NavPPO):
    STAGE_TYPE = p4_contract.STAGE_TYPE

    def __init__(self, *args, **kwargs):
        early_config = dict(kwargs.get("config") or {})
        early_config.setdefault(
            "nav_period_frames", p4_contract.P4_NAV_PERIOD_FRAMES
        )
        kwargs["config"] = early_config
        self.maze_training_branch = str(
            early_config.get("maze_training_branch", "actor_attack")
        )
        self._resolved_maze_training_branch = None
        self.session_wall_seconds = 0.0
        self.diagnostic_elapsed_seconds = 0.0
        self._training_clock_origin_seconds = None
        super().__init__(*args, **kwargs)
        runtime_segments = tuple(
            self.config.get("track_segment_labels", ("maze",))
        )
        if runtime_segments != ("maze",):
            raise ValueError(
                "P4 Maze-only training requires track_segment_labels=['maze']; "
                f"got {list(runtime_segments)!r}"
            )
        if self.nav_period_frames != p4_contract.P4_NAV_PERIOD_FRAMES:
            raise ValueError(
                "P4 runtime navigation period does not match the 10 Hz contract: "
                f"runtime={self.nav_period_frames} "
                f"contract={p4_contract.P4_NAV_PERIOD_FRAMES}"
            )
        runtime_slew = tuple(float(value) for value in self.command_slew_rate)
        runtime_release = tuple(
            float(value) for value in self.command_slew_release_rate
        )
        if (
            runtime_slew != p4_contract.P4_SLEW_RATE
            or runtime_release != p4_contract.P4_SLEW_RELEASE_RATE
        ):
            raise ValueError(
                "P4 runtime slew does not match command contract: "
                f"slew={runtime_slew!r} release={runtime_release!r}"
            )
        seed = int(self.config.get("p4_seed", 4100))
        self.stuck_reset_contract = p4_contract.normalize_stuck_reset_contract(
            self.config.get("stuck_reset")
        )
        self.goal_belief = GoalBeliefChainV2(
            self.num_envs, self.device, seed=seed + 1
        )
        self.camera_state = P4SharedCameraState(
            self.num_envs, self.device, seed=seed + 2
        )
        self.maze_training_branch = str(
            self.config.get("maze_training_branch", "actor_attack")
        )
        if not hasattr(self, "_resolved_maze_training_branch"):
            self._resolved_maze_training_branch = None
        self.user_speed_cap = torch.full(
            (self.num_envs,), p4_contract.P4_MAX_VX, device=self.device
        )
        self.safety_speed_cap = torch.ones(self.num_envs, device=self.device)
        self._safety_cap_predictive_risk = torch.zeros(
            self.num_envs, device=self.device
        )
        self.effective_speed_cap = self.user_speed_cap.clone()
        self._last_policy_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_limited_command = torch.zeros_like(self._last_policy_command)
        self._last_goal_freshness = torch.zeros(self.num_envs, device=self.device)
        self._zero_hidden_action_mae = torch.zeros(self.num_envs, device=self.device)
        self._zero_hidden_direction_disagreement = torch.zeros(
            self.num_envs, device=self.device
        )
        self._last_safety_head_risk3 = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._risk_event_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._risk_event_age_ticks = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._risk_event_baseline_policy_vx = torch.zeros(
            self.num_envs, device=self.device
        )
        self._risk_event_baseline_limited_vx = torch.zeros(
            self.num_envs, device=self.device
        )
        self._risk_condition_previous = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.push_epoch = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.seconds_since_push = torch.full(
            (self.num_envs,), 1.0e6, device=self.device
        )
        # The final transport of one rollout is reused as the first transport
        # of the next rollout.  Deduplicate worker-owned push pulses by the
        # worker step counter carried in the stable P2 aux prefix.
        self._last_push_transport_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._push_interval_count = torch.zeros(self.num_envs, device=self.device)
        self._push_interval_delta_sum = torch.zeros(self.num_envs, 2, device=self.device)
        self._push_interval_delta_abs_max = torch.zeros_like(
            self._push_interval_delta_sum
        )
        self._push_lifetime_count = torch.zeros(self.num_envs, device=self.device)
        self._push_rollout_count = torch.zeros(self.num_envs, device=self.device)
        self._push_env_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._push_recovery_pending = torch.zeros_like(self._push_env_seen)
        self._push_recovery_time = torch.full(
            (self.num_envs,), float("nan"), device=self.device
        )
        self._p4_extra = torch.zeros(
            self.num_envs,
            p3_contract.P3_WORKER_EXTRA_DIM,
            device=self.device,
        )
        self._p4_worker_extra = torch.zeros(
            self.num_envs,
            p4_contract.P4_WORKER_EXTRA_DIM,
            device=self.device,
        )
        self._cached_low_cnn = torch.zeros(
            self.num_envs, 32, device=self.device
        )
        self._cached_low_frame_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._clean_depth = None
        self._diagnostic_fault_depth = None
        self._diagnostic_fault_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._diagnostic_terminal_mask = torch.zeros_like(
            self._diagnostic_fault_mask
        )
        self._diagnostic_fault_nav_feat = torch.zeros(
            self.num_envs, 32, device=self.device
        )
        self._diagnostic_clean_fault_latent_cosine = torch.ones(
            self.num_envs, device=self.device
        )
        self._diagnostic_clean_fault_action_mae = torch.zeros(
            self.num_envs, device=self.device
        )
        self._camera_diagnostics: dict[str, torch.Tensor] = {}
        self._clean_action_mean = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._camera_aux_mask = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._yaw_window_ticks = max(
            2, int(round(p4_contract.YAW_WINDOW_SECONDS / self.nav_dt_s))
        )
        self._risk_response_ticks = max(
            1, int(round(p4_contract.RISK_RESPONSE_WINDOW_SECONDS / self.nav_dt_s))
        )
        self._yaw_exec_history = torch.zeros(
            self._yaw_window_ticks, self.num_envs, device=self.device
        )
        self._yaw_true_history = torch.zeros_like(self._yaw_exec_history)
        self._yaw_history_cursor = 0
        self._yaw_history_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._p4_reward_diagnostics: dict[str, torch.Tensor] = {}
        self._camera_aux_coefficient = 0.0
        self._camera_aux_gradient_ratio = 0.0
        self._camera_aux_calibration_pending = True
        self._camera_clean_live_latent_cosine = torch.ones(
            self.num_envs, device=self.device
        )
        self._goal_epoch_changed_since_tick = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._p4_episode_return = torch.zeros(self.num_envs, device=self.device)
        self._teacher_navigation_encoder = copy.deepcopy(self.navigation_encoder).to(
            self.device
        )
        self._teacher_actor = copy.deepcopy(self.actor).to(self.device)
        self._teacher_hidden = None
        self.parent_phase_label = None
        self._diagnostic_nav_risk_probe = nn.Linear(32, 3).to(self.device)
        self._diagnostic_nav_scene_probe = nn.Linear(32, 5).to(self.device)
        self._diagnostic_goal_risk_probe = nn.Linear(4, 3).to(self.device)
        self._diagnostic_goal_scene_probe = nn.Linear(4, 5).to(self.device)
        self._reset_maze_diagnostic_probes()
        self._reset_maze_diagnostic_state()
        for module in (self._teacher_navigation_encoder, self._teacher_actor):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self._initial_low_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        self.cnn_unfrozen = False
        self.current_vy_trusted_limit = p4_contract.P4_MAX_ABS_VY
        self.current_vy_hard_limit = p4_contract.P4_MAX_ABS_VY
        if self.training_enabled:
            self._apply_training_schedule(self.session_effective_seconds)
            if self.rollout is not None:
                self.rollout = self.rollout.reset(
                    store_depth=self.cnn_unfrozen
                )

    # ------------------------------------------------------------------
    # Versioned transport, GoalBelief, camera and command mapping
    # ------------------------------------------------------------------

    def _split_transport(self, critic_wire: torch.Tensor):
        if critic_wire.ndim != 2:
            raise ValueError("P4 privileged transport must be rank-2")
        if critic_wire.shape[1] == p2_contract.PRIVILEGED_WIRE_DIM:
            self._p4_extra.zero_()
            self._p4_worker_extra.zero_()
            return split_p2_transport(critic_wire)
        if critic_wire.shape[1] != p4_contract.P4_PRIVILEGED_WIRE_DIM:
            raise ValueError(
                "P4 privileged wire must be 385 eval columns or 507 training "
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
        del reset
        worker_reset = aux[:, 24] > 0.5
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
            )
            if self.session_effective_seconds < 3_600.0:
                goal_fault_profile = "noise_only"
            elif self.session_effective_seconds < 18_000.0:
                goal_fault_profile = "medium"
            elif self.session_effective_seconds < 25_200.0:
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
                fault_profile=goal_fault_profile,
                fault_allowed_mask=fault_allowed,
            )
            self._goal_epoch_changed_since_tick |= (
                self.goal_belief.last_diagnostics["goal_epoch_changed"] > 0.5
            )
        delivered, diagnostics = self.camera_state.process(
            parts["depth"],
            reset_mask=worker_reset,
            session_effective_seconds=self.session_effective_seconds,
            training=self.training_enabled,
        )
        self._clean_depth = self.camera_state.clean_capture.detach().clone()
        parts["depth"] = delivered
        if self.frame_count % self.nav_period_frames == 0:
            with torch.inference_mode():
                current_arc = self._predictive_command(
                    self.command.active_target
                )
                safety_probe = p2_contract.predictive_collision_risk_penalty(
                    delivered, current_arc
                )
                current_risk = safety_probe[3]
            self._safety_cap_predictive_risk = current_risk.detach()
            self.safety_speed_cap = torch.clamp(
                1.0 - 0.75 * current_risk, 0.25, 1.0
            )
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
        if batch != self.num_envs:
            cap = torch.full((batch,), p4_contract.P4_MAX_VX, device=self.device)
        else:
            cap = self.effective_speed_cap
        result = torch.tensor(
            p2_contract.NAV_CAPABILITY_PROFILE15,
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
        return result

    def _response_capability(self, batch: int, *, dtype=torch.float32) -> torch.Tensor:
        result = super()._response_capability(batch, dtype=dtype).clone()
        result[:, 3] = p4_contract.P4_MAX_VX
        result[:, 4] = p4_contract.P4_MAX_ABS_WZ
        result[:, 7] = p4_contract.P4_MAX_ABS_VY
        return result

    def _map_policy_target(self, normalized, legacy_target, *, goal4, aux):
        del legacy_target, aux
        full_cap = torch.full(
            (normalized.shape[0],), p4_contract.P4_MAX_VX,
            device=normalized.device, dtype=normalized.dtype,
        )
        policy = p4_contract.map_normalized_action(
            normalized, full_cap, goal_freshness=None
        )
        limited = p4_contract.map_normalized_action(
            normalized, self.effective_speed_cap, goal4[:, 3]
        )
        self._last_normalized_action = normalized.detach()
        self._last_policy_command = policy.detach()
        self._last_limited_command = limited.detach()
        self._last_mapped_command = limited.detach()
        self._last_goal_freshness = goal4[:, 3].detach()
        return limited

    def _predictive_command(self, target: torch.Tensor) -> torch.Tensor:
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

    # ------------------------------------------------------------------
    # Clean/live camera auxiliary and reward closure
    # ------------------------------------------------------------------

    def begin_rollout(self) -> None:
        """Refresh the clean teacher from the current live policy."""
        self._push_rollout_count.zero_()
        self.camera_state.begin_rollout(
            self.session_effective_seconds, training=self.training_enabled
        )
        self._teacher_navigation_encoder.load_state_dict(
            self.navigation_encoder.state_dict(), strict=True
        )
        self._teacher_actor.load_state_dict(self.actor.state_dict(), strict=True)
        self._camera_aux_calibration_pending = True

    def _update_policy_auxiliary_target(
        self, *, parts, nav_feat, nav_nonvisual, profile, confidence, reset
    ) -> None:
        if self._clean_depth is None:
            return
        diagnostic_active = (
            self.maze_training_branch == "auto"
            and self._resolved_maze_training_branch is None
            and self.diagnostic_elapsed_seconds < p4_contract.DIAGNOSTIC_SECONDS
        )
        if diagnostic_active:
            (
                self._diagnostic_fault_depth,
                self._diagnostic_fault_mask,
            ) = self.camera_state.diagnostic_fault_shadow(
                self._clean_depth, sample_share=0.10
            )
        else:
            self._diagnostic_fault_depth = None
            self._diagnostic_fault_mask.zero_()
        self._teacher_hidden = self._mask_hidden(self._teacher_hidden, reset)
        teacher_hidden_before = (
            tuple(item.clone() for item in self._teacher_hidden)
            if self._teacher_hidden is not None
            else None
        )
        with torch.inference_mode():
            clean_feat = self._teacher_navigation_encoder(self._clean_depth)
            self._camera_clean_live_latent_cosine = F.cosine_similarity(
                clean_feat, nav_feat, dim=-1
            ).detach()
            teacher_input = assemble_actor_input(
                clean_feat, nav_nonvisual, profile, confidence
            )
            _, normalized, _, _, self._teacher_hidden = self._teacher_actor.deterministic(
                teacher_input,
                self._teacher_hidden,
                reset,
                hard_abs_vy=p4_contract.P4_MAX_ABS_VY,
            )
            fault_normalized = normalized
            self._diagnostic_fault_nav_feat.copy_(clean_feat)
            if (
                self._diagnostic_fault_depth is not None
                and bool(self._diagnostic_fault_mask.any())
            ):
                selected = self._diagnostic_fault_mask
                fault_feat = self._teacher_navigation_encoder(
                    self._diagnostic_fault_depth[selected]
                )
                self._diagnostic_fault_nav_feat[selected] = fault_feat
                fault_input = assemble_actor_input(
                    self._diagnostic_fault_nav_feat,
                    nav_nonvisual,
                    profile,
                    confidence,
                )
                _, fault_normalized, _, _, _ = self._teacher_actor.deterministic(
                    fault_input,
                    teacher_hidden_before,
                    reset,
                    hard_abs_vy=p4_contract.P4_MAX_ABS_VY,
                )
                self._diagnostic_clean_fault_latent_cosine = F.cosine_similarity(
                    clean_feat,
                    self._diagnostic_fault_nav_feat,
                    dim=-1,
                ).detach()
                self._diagnostic_clean_fault_action_mae = (
                    normalized - fault_normalized
                ).abs().mean(dim=-1).detach()
            else:
                self._diagnostic_clean_fault_latent_cosine.fill_(1.0)
                self._diagnostic_clean_fault_action_mae.zero_()
            zero_hidden = (
                torch.zeros(
                    self._teacher_actor.num_layers,
                    teacher_input.shape[0],
                    self._teacher_actor.hidden_dim,
                    device=teacher_input.device,
                    dtype=teacher_input.dtype,
                ),
                torch.zeros(
                    self._teacher_actor.num_layers,
                    teacher_input.shape[0],
                    self._teacher_actor.hidden_dim,
                    device=teacher_input.device,
                    dtype=teacher_input.dtype,
                ),
            )
            _, zero_normalized, _, _, _ = self._teacher_actor.deterministic(
                teacher_input,
                zero_hidden,
                torch.ones_like(reset, dtype=torch.bool),
                hard_abs_vy=p4_contract.P4_MAX_ABS_VY,
            )
        self._clean_action_mean = normalized.detach()
        if self.safety_head is None:
            self._last_safety_head_risk3.zero_()
        else:
            self._last_safety_head_risk3 = torch.sigmoid(
                self.safety_head(nav_feat.detach())
            ).detach()
        zero_delta = (normalized - zero_normalized).abs()
        self._zero_hidden_action_mae = zero_delta.mean(dim=-1).detach()
        self._zero_hidden_direction_disagreement = (
            torch.sign(normalized[:, 2]) != torch.sign(zero_normalized[:, 2])
        ).float().detach()
        self._camera_aux_mask = torch.stack(
            (
                self._camera_diagnostics["camera_delay_only"],
                self._camera_diagnostics["camera_fault_only"],
                self._camera_diagnostics["camera_fault_delay_overlap"],
            ),
            dim=-1,
        ).detach()

    def _transition_extras(self) -> dict[str, torch.Tensor]:
        return {
            "clean_action_mean": self._clean_action_mean,
            "camera_aux_mask": self._camera_aux_mask,
        }

    @staticmethod
    def _grad_norm(grads) -> torch.Tensor:
        values = [grad.square().sum() for grad in grads if grad is not None]
        if not values:
            return torch.tensor(0.0)
        return torch.sqrt(torch.stack(values).sum())

    def _actor_auxiliary_loss(
        self, *, normalized_mean, batch, ppo_actor_loss
    ):
        mask3 = batch["camera_aux_mask"] > 0.5
        selected = mask3.any(dim=-1)
        if not bool(selected.any()):
            zero = normalized_mean.new_zeros(())
            return zero, {
                "camera_memory_loss": zero.detach(),
                "camera_aux_gradient_ratio": zero.detach(),
            }
        per_action = F.smooth_l1_loss(
            normalized_mean,
            batch["clean_action_mean"],
            reduction="none",
        ).mean(dim=-1)
        raw = per_action[selected].mean()
        target_ratio = float(
            p4_contract.training_schedule(
                self.session_effective_seconds,
                branch=self._effective_maze_branch(self.session_effective_seconds),
            )["camera_aux_ratio"]
        )
        if target_ratio <= 0.0:
            coefficient = 0.0
            actual_ratio = 0.0
        elif self._camera_aux_calibration_pending:
            parameters = [
                parameter
                for module in (self.navigation_encoder, self.actor)
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            aux_grads = torch.autograd.grad(
                raw, parameters, retain_graph=True, allow_unused=True
            )
            ppo_grads = torch.autograd.grad(
                ppo_actor_loss, parameters, retain_graph=True, allow_unused=True
            )
            aux_norm = float(self._grad_norm(aux_grads).detach().cpu())
            ppo_norm = float(self._grad_norm(ppo_grads).detach().cpu())
            coefficient = target_ratio * ppo_norm / max(aux_norm, 1.0e-12)
            coefficient = max(0.0, min(coefficient, 0.02 * ppo_norm / max(aux_norm, 1.0e-12)))
            actual_ratio = coefficient * aux_norm / max(ppo_norm, 1.0e-12)
            self._camera_aux_coefficient = coefficient
            self._camera_aux_gradient_ratio = min(actual_ratio, 0.02)
            self._camera_aux_calibration_pending = False
        else:
            coefficient = self._camera_aux_coefficient
            actual_ratio = self._camera_aux_gradient_ratio
        loss = raw * float(coefficient)
        return loss, {
            "camera_memory_loss": raw.detach(),
            "camera_clean_live_action_mae": (
                normalized_mean[selected]
                - batch["clean_action_mean"][selected]
            ).abs().mean().detach(),
            "camera_aux_coefficient": raw.new_tensor(float(coefficient)),
            "camera_aux_gradient_ratio": raw.new_tensor(float(actual_ratio)),
            "camera_delay_only_share": mask3[..., 0].float().mean().detach(),
            "camera_fault_only_share": mask3[..., 1].float().mean().detach(),
            "camera_fault_delay_overlap_share": mask3[..., 2].float().mean().detach(),
        }

    def _actor_auxiliary_metric_names(self) -> tuple[str, ...]:
        return (
            "camera_memory_loss",
            "camera_clean_live_action_mae",
            "camera_aux_coefficient",
            "camera_aux_gradient_ratio",
            "camera_delay_only_share",
            "camera_fault_only_share",
            "camera_fault_delay_overlap_share",
        )

    def _override_reward_components(self, components, **context):
        exec_cmd = context["reward_exec_cmd"]
        source_aux = context["reward_source_aux"]
        terminal = context["terminal"].bool()
        duration_frames = torch.as_tensor(
            context.get("duration_frames", self.nav_period_frames),
            device=self.device,
            dtype=torch.float32,
        ).reshape(-1)
        continuous_time_scale = torch.clamp(
            duration_frames / float(p2_contract.NAV_PERIOD_FRAMES), 0.0, 1.0
        )
        for name in ("crawl", "tracking", "gait_symmetry"):
            if name in components:
                components[name] = components[name] * continuous_time_scale
        wall_stuck_terminal = context["reason"].reshape(-1) == 4
        history_reset = terminal | self._goal_epoch_changed_since_tick
        if bool(history_reset.any()):
            self._yaw_exec_history[:, history_reset] = 0.0
            self._yaw_true_history[:, history_reset] = 0.0
            self._yaw_history_count[history_reset] = 0
        previous_index = (self._yaw_history_cursor - 1) % self._yaw_window_ticks
        previous_exec_wz = self._yaw_exec_history[previous_index]
        previous_true_wz = self._yaw_true_history[previous_index]
        exec_sign_flip = (
            (previous_exec_wz * exec_cmd[:, 2] < 0.0)
            & (self._yaw_history_count > 0)
        )
        true_sign_flip = (
            (previous_true_wz * source_aux[:, 14] < 0.0)
            & (self._yaw_history_count > 0)
        )
        self._yaw_exec_history[self._yaw_history_cursor] = exec_cmd[:, 2]
        self._yaw_true_history[self._yaw_history_cursor] = source_aux[:, 14]
        self._yaw_history_count = torch.where(
            terminal,
            torch.zeros_like(self._yaw_history_count),
            torch.clamp(self._yaw_history_count + 1, max=self._yaw_window_ticks),
        )
        self._yaw_history_cursor = (
            self._yaw_history_cursor + 1
        ) % self._yaw_window_ticks
        exec_cancel = p4_contract.yaw_cancellation(
            self._yaw_exec_history, dt_s=self.nav_dt_s
        )
        true_cancel = p4_contract.yaw_cancellation(
            self._yaw_true_history, dt_s=self.nav_dt_s
        )
        enough = self._yaw_history_count >= self._yaw_window_ticks
        yaw_raw = torch.where(
            enough,
            p4_contract.YAW_EXEC_WEIGHT * exec_cancel
            + p4_contract.YAW_TRUE_WEIGHT * true_cancel,
            torch.zeros_like(exec_cancel),
        ).clamp_min(p4_contract.YAW_TOTAL_FLOOR)
        predictive_raw = (
            components["predictive_collision_risk"]
            * p4_contract.PREDICTIVE_COLLISION_SCALE
        ).clamp_min(p4_contract.PREDICTIVE_RAW_FLOOR)
        safe3 = self.pending_tick.get(
            "safe3", torch.zeros(self.num_envs, 3, device=self.device)
        )
        safety_valid = self.pending_tick.get(
            "safety_valid", torch.zeros(self.num_envs, 1, device=self.device)
        ).reshape(-1) > 0.5
        missed_raw, missed_diagnostics = (
            p4_contract.maze_missed_safe_direction_penalty(
                safe3,
                self._last_policy_command,
                safety_valid,
                self.session_effective_seconds,
            )
        )
        schedule = p4_contract.training_schedule(
            self.session_effective_seconds,
            branch=self._effective_maze_branch(self.session_effective_seconds),
        )
        reward_multiplier = float(schedule["reward_multiplier"])
        predictive_raw *= reward_multiplier
        missed_raw *= reward_multiplier
        yaw_raw *= reward_multiplier
        (predictive, missed, yaw), scale = p4_contract.proportional_negative_cap(
            predictive_raw, missed_raw, yaw_raw
        )
        predictive *= continuous_time_scale
        missed *= continuous_time_scale
        yaw *= continuous_time_scale
        components["predictive_collision_risk"] = predictive
        components["missed_safe_direction"] = missed
        components["yaw_cancellation"] = yaw
        components["success"] = (context["reason"].reshape(-1) == 1).float() * float(
            p4_contract.SUCCESS_IMPULSE
        )
        stagnation_shadow = components["frontier_stagnation"].detach().clone()
        components["frontier_stagnation"] = torch.zeros_like(stagnation_shadow)
        # A confirmed wall-stuck reset is a single terminal event. Recharging
        # collision/predictive/stagnation penalties on that same tick would
        # count the same failure twice and make the new terminal dominate PPO.
        for name in (
            "body_collision",
            "predictive_collision_risk",
            "missed_safe_direction",
            "frontier_stagnation",
        ):
            components[name] = torch.where(
                wall_stuck_terminal,
                torch.zeros_like(components[name]),
                components[name],
            )
        components["stuck_reset"] = wall_stuck_terminal.float() * float(
            self.stuck_reset_contract["terminal_penalty"]
        )
        stuck_sustained, stuck_sustained_diag = (
            p4_contract.sustained_wall_stuck_penalty(
                self._p4_worker_extra[:, p4_contract.STUCK_DURATION_S_INDEX],
                self._p4_worker_extra[:, p4_contract.STUCK_CANDIDATE_INDEX] > 0.5,
                self._p4_worker_extra[:, p4_contract.STUCK_MAPPING_VALID_INDEX] > 0.5,
                terminal,
                confirmation_s=float(self.stuck_reset_contract["confirmation_s"]),
            )
        )
        components["stuck_sustained"] = (
            stuck_sustained * continuous_time_scale
        )
        goal_safe, goal_safe_diag = p4_contract.goal_safe_direction_penalty(
            safe3,
            self._last_policy_command,
            self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE],
            safety_valid,
            self._last_goal_freshness,
            terminal,
        )
        components["goal_safe_preference"] = (
            goal_safe * continuous_time_scale
        )
        path_length_m = context.get("path_length_m")
        if path_length_m is None:
            path_length_m = torch.zeros(self.num_envs, device=self.device)
        route_excess, route_diag = p4_contract.route_excess_penalty(
            torch.as_tensor(path_length_m, device=self.device).reshape(-1),
            context["start_goal_distance"].reshape(-1),
            context["end_goal_distance"].reshape(-1),
            terminal,
        )
        grace = self.seconds_since_push < 0.30
        components["route_excess"] = torch.where(
            grace, torch.zeros_like(route_excess), route_excess
        )
        soft_cruise, cruise_diag = p4_contract.soft_cruise_penalty(
            self._last_policy_command,
            self.pending_tick.get("safe3", torch.zeros(self.num_envs, 3, device=self.device)),
            self.pending_tick.get(
                "safety_valid",
                torch.zeros(self.num_envs, 1, device=self.device),
            ).reshape(-1) > 0.5,
            self._last_goal_freshness,
            terminal,
        )
        soft_cruise *= (
            float(schedule.get("cruise_multiplier", 1.0))
            * continuous_time_scale
        )
        components["soft_cruise"] = soft_cruise
        components["tracking"] = torch.where(
            grace, components["tracking"] * 0.5, components["tracking"]
        )
        response_mae = (
            source_aux[:, 12:15] - exec_cmd
        ).abs().mean(dim=-1)
        recovered = (
            self._push_recovery_pending
            & (self.seconds_since_push >= 0.30)
            & (response_mae <= 0.12)
            & ~terminal
        )
        self._push_recovery_time[recovered] = self.seconds_since_push[recovered]
        self._push_recovery_pending[recovered | terminal] = False
        self._p4_reward_diagnostics = {
            "reward_predictive_raw": predictive_raw.detach(),
            "reward_missed_safe_raw": missed_raw.detach(),
            "reward_yaw_raw": yaw_raw.detach(),
            "reward_continuous_time_scale": continuous_time_scale.detach(),
            "reward_safety_group_scale": scale.detach(),
            "reward_frontier_stagnation_shadow": stagnation_shadow,
            "yaw_exec_cancellation": exec_cancel.detach(),
            "yaw_true_cancellation": true_cancel.detach(),
            "yaw_exec_sign_flip": exec_sign_flip.float(),
            "yaw_true_sign_flip": true_sign_flip.float(),
            "yaw_true_overshoot": torch.clamp(
                source_aux[:, 14].abs()
                - self.pending_tick["target_cmd3"][:, 2].abs(),
                min=0.0,
            ),
            "push_grace_active": grace.float(),
            "push_tracking_response_mae": response_mae,
            **{name: value.detach() for name, value in stuck_sustained_diag.items()},
            **{name: value.detach() for name, value in goal_safe_diag.items()},
            **{name: value.detach() for name, value in route_diag.items()},
            **{name: value.detach() for name, value in missed_diagnostics.items()},
            **{name: value.detach() for name, value in cruise_diag.items()},
        }
        self._goal_epoch_changed_since_tick.zero_()
        return components

    @torch.no_grad()
    def finish_tick(self, *args, **kwargs):
        reason = torch.as_tensor(
            kwargs["terminal_reason"], device=self.device
        ).reshape(-1).round().long()
        hard = torch.as_tensor(
            kwargs["hard_terminated"], device=self.device
        ).reshape(-1).bool()
        timeout = torch.as_tensor(
            kwargs["timeout"], device=self.device
        ).reshape(-1).bool()
        self._diagnostic_terminal_mask.copy_((reason != 0) | hard | timeout)
        full = super().finish_tick(*args, **kwargs)
        tick_reward = self.last_tick_penalties["decomposed_total"].reshape(-1)
        self._p4_episode_return += tick_reward
        stuck = reason == 4
        stuck_return = torch.where(
            stuck, self._p4_episode_return, torch.zeros_like(self._p4_episode_return)
        )
        self.last_tick_diagnostics.update(
            {
                "stuck_terminal_count": stuck.float().reshape(-1, 1),
                "stuck_terminal_episode_return_sum": stuck_return.reshape(-1, 1),
                "stuck_terminal_nonnegative_count": (
                    stuck & (self._p4_episode_return >= 0.0)
                ).float().reshape(-1, 1),
            }
        )
        self._p4_episode_return[(reason != 0) | hard | timeout] = 0.0
        return full

    def _reset_maze_diagnostic_state(self) -> None:
        self._maze_diag_total = torch.zeros((), device=self.device)
        self._maze_diag_valid = torch.zeros((), device=self.device)
        self._maze_diag_wall_positive = torch.zeros((), device=self.device)
        self._maze_diag_wall_missed = torch.zeros((), device=self.device)
        self._maze_diag_top1_total = torch.zeros((), device=self.device)
        self._maze_diag_top1_correct = torch.zeros((), device=self.device)
        self._maze_diag_risk_positive_hist = torch.zeros(20, device=self.device)
        self._maze_diag_risk_negative_hist = torch.zeros(20, device=self.device)
        self._maze_diag_scene_confusion = torch.zeros(5, 6, device=self.device)
        self._maze_diag_latent_cosine_sum = torch.zeros((), device=self.device)
        self._maze_diag_latent_cosine_count = torch.zeros((), device=self.device)
        self._maze_diag_goal_risk_positive_hist = torch.zeros(20, device=self.device)
        self._maze_diag_goal_risk_negative_hist = torch.zeros(20, device=self.device)
        self._maze_diag_goal_top1_total = torch.zeros((), device=self.device)
        self._maze_diag_goal_top1_correct = torch.zeros((), device=self.device)
        self._maze_diag_goal_scene_confusion = torch.zeros(5, 6, device=self.device)
        self._maze_diag_fault_risk_positive_hist = torch.zeros(20, device=self.device)
        self._maze_diag_fault_risk_negative_hist = torch.zeros(20, device=self.device)
        self._maze_diag_fault_top1_total = torch.zeros((), device=self.device)
        self._maze_diag_fault_top1_correct = torch.zeros((), device=self.device)
        self._maze_diag_fault_scene_confusion = torch.zeros(5, 6, device=self.device)

    def _reset_maze_diagnostic_probes(self) -> None:
        probes = (
            self._diagnostic_nav_risk_probe,
            self._diagnostic_nav_scene_probe,
            self._diagnostic_goal_risk_probe,
            self._diagnostic_goal_scene_probe,
        )
        for probe in probes:
            nn.init.zeros_(probe.weight)
            nn.init.zeros_(probe.bias)
        self._diagnostic_probe_optimizer = torch.optim.SGD(
            [parameter for probe in probes for parameter in probe.parameters()],
            lr=0.05,
        )

    @staticmethod
    def _diagnostic_scene_class(
        safe3: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        diagnostics = p4_contract.safety_scene_diagnostics(safe3, valid)
        result = torch.full(
            (safe3.shape[0],), 5, dtype=torch.long, device=safe3.device
        )
        for index, name in enumerate(
            (
                "teacher_scene_corridor",
                "teacher_scene_left_open",
                "teacher_scene_right_open",
                "teacher_scene_junction",
                "teacher_scene_dead_end",
            )
        ):
            select = (result == 5) & (diagnostics[name] > 0.5)
            result[select] = index
        result[~valid.reshape(-1).bool()] = 5
        return result

    def _accumulate_maze_diagnostic(
        self,
        safe3: torch.Tensor,
        teacher_valid: torch.Tensor,
        nav_feat: torch.Tensor,
        goal4: torch.Tensor,
        fault_nav_feat: torch.Tensor | None = None,
        fault_mask: torch.Tensor | None = None,
    ) -> None:
        if self.maze_training_branch != "auto":
            return
        if self._resolved_maze_training_branch is not None:
            return
        if self.diagnostic_elapsed_seconds >= p4_contract.DIAGNOSTIC_SECONDS:
            return
        valid = teacher_valid.reshape(-1).bool()
        self._maze_diag_total += float(valid.numel())
        self._maze_diag_valid += valid.float().sum()
        self._maze_diag_latent_cosine_sum += torch.nan_to_num(
            self._camera_clean_live_latent_cosine, nan=0.0
        ).sum()
        self._maze_diag_latent_cosine_count += float(
            self._camera_clean_live_latent_cosine.numel()
        )
        if not bool(valid.any()):
            return

        teacher_risk = (1.0 - safe3).clamp(0.0, 1.0)
        nav_feat = nav_feat.detach()
        goal4 = goal4.detach()
        fault_nav_feat = (
            fault_nav_feat.detach()
            if torch.is_tensor(fault_nav_feat)
            else nav_feat
        )
        fault_mask = (
            fault_mask.reshape(-1).bool()
            if torch.is_tensor(fault_mask)
            else torch.zeros_like(valid)
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        train_mask = valid & ((env_ids % 5) != 0)
        teacher_class = self._diagnostic_scene_class(safe3, valid)
        scene_train = train_mask & (teacher_class < 5)
        # finish_tick() is intentionally inference-only and therefore invokes
        # this hook under torch.no_grad(). The training-only linear probes still
        # need their own small graph; detached inputs prevent gradients from
        # reaching NavigationEncoder, Actor, or the rollout collection path.
        with torch.enable_grad():
            self._diagnostic_probe_optimizer.zero_grad(set_to_none=True)
            losses = []
            if bool(train_mask.any()):
                losses.extend(
                    (
                        F.binary_cross_entropy_with_logits(
                            self._diagnostic_nav_risk_probe(nav_feat[train_mask]),
                            teacher_risk[train_mask],
                        ),
                        F.binary_cross_entropy_with_logits(
                            self._diagnostic_goal_risk_probe(goal4[train_mask]),
                            teacher_risk[train_mask],
                        ),
                    )
                )
            if bool(scene_train.any()):
                losses.extend(
                    (
                        F.cross_entropy(
                            self._diagnostic_nav_scene_probe(nav_feat[scene_train]),
                            teacher_class[scene_train],
                        ),
                        F.cross_entropy(
                            self._diagnostic_goal_scene_probe(goal4[scene_train]),
                            teacher_class[scene_train],
                        ),
                    )
                )
            if losses:
                torch.stack(losses).sum().backward()
                self._diagnostic_probe_optimizer.step()
            self._diagnostic_probe_optimizer.zero_grad(set_to_none=True)

        eval_mask = valid & ((env_ids % 5) == 0)
        if not bool(eval_mask.any()):
            eval_mask = valid
        with torch.no_grad():
            predicted_risk = torch.sigmoid(
                self._diagnostic_nav_risk_probe(nav_feat)
            ).clamp(0.0, 1.0)
            goal_predicted_risk = torch.sigmoid(
                self._diagnostic_goal_risk_probe(goal4)
            ).clamp(0.0, 1.0)
            predicted_scene = self._diagnostic_nav_scene_probe(nav_feat).argmax(dim=-1)
            goal_predicted_scene = self._diagnostic_goal_scene_probe(goal4).argmax(dim=-1)
            fault_predicted_risk = torch.sigmoid(
                self._diagnostic_nav_risk_probe(fault_nav_feat)
            ).clamp(0.0, 1.0)
            fault_predicted_scene = self._diagnostic_nav_scene_probe(
                fault_nav_feat
            ).argmax(dim=-1)

        valid3 = eval_mask.unsqueeze(-1).expand_as(teacher_risk)
        positive = valid3 & (teacher_risk >= 0.65)
        negative = valid3 & (teacher_risk <= 0.35)
        self._maze_diag_wall_positive += positive.float().sum()
        self._maze_diag_wall_missed += (
            positive & (predicted_risk < 0.50)
        ).float().sum()
        bins = torch.clamp((predicted_risk * 20.0).long(), 0, 19)
        if bool(positive.any()):
            self._maze_diag_risk_positive_hist += torch.bincount(
                bins[positive], minlength=20
            ).to(self._maze_diag_risk_positive_hist)
        if bool(negative.any()):
            self._maze_diag_risk_negative_hist += torch.bincount(
                bins[negative], minlength=20
            ).to(self._maze_diag_risk_negative_hist)
        goal_bins = torch.clamp((goal_predicted_risk * 20.0).long(), 0, 19)
        if bool(positive.any()):
            self._maze_diag_goal_risk_positive_hist += torch.bincount(
                goal_bins[positive], minlength=20
            ).to(self._maze_diag_goal_risk_positive_hist)
        if bool(negative.any()):
            self._maze_diag_goal_risk_negative_hist += torch.bincount(
                goal_bins[negative], minlength=20
            ).to(self._maze_diag_goal_risk_negative_hist)

        scene = p4_contract.safety_scene_diagnostics(safe3, valid)
        clear = (scene["teacher_safe_top1_clear"] > 0.5) & eval_mask
        teacher_top = safe3.argmax(dim=-1)
        predicted_safe = 1.0 - predicted_risk
        predicted_top = predicted_safe.argmax(dim=-1)
        goal_predicted_top = (1.0 - goal_predicted_risk).argmax(dim=-1)
        self._maze_diag_top1_total += clear.float().sum()
        self._maze_diag_top1_correct += (
            clear & (teacher_top == predicted_top)
        ).float().sum()
        self._maze_diag_goal_top1_total += clear.float().sum()
        self._maze_diag_goal_top1_correct += (
            clear & (teacher_top == goal_predicted_top)
        ).float().sum()

        labeled = (teacher_class < 5) & eval_mask
        if bool(labeled.any()):
            flat = teacher_class[labeled] * 6 + predicted_scene[labeled]
            self._maze_diag_scene_confusion += torch.bincount(
                flat, minlength=30
            ).reshape(5, 6).to(self._maze_diag_scene_confusion)
            goal_flat = teacher_class[labeled] * 6 + goal_predicted_scene[labeled]
            self._maze_diag_goal_scene_confusion += torch.bincount(
                goal_flat, minlength=30
            ).reshape(5, 6).to(self._maze_diag_goal_scene_confusion)
        fault_eval = eval_mask & fault_mask
        if bool(fault_eval.any()):
            fault_valid3 = fault_eval.unsqueeze(-1).expand_as(teacher_risk)
            fault_positive = fault_valid3 & (teacher_risk >= 0.65)
            fault_negative = fault_valid3 & (teacher_risk <= 0.35)
            fault_bins = torch.clamp((fault_predicted_risk * 20.0).long(), 0, 19)
            if bool(fault_positive.any()):
                self._maze_diag_fault_risk_positive_hist += torch.bincount(
                    fault_bins[fault_positive], minlength=20
                ).to(self._maze_diag_fault_risk_positive_hist)
            if bool(fault_negative.any()):
                self._maze_diag_fault_risk_negative_hist += torch.bincount(
                    fault_bins[fault_negative], minlength=20
                ).to(self._maze_diag_fault_risk_negative_hist)
            fault_clear = (scene["teacher_safe_top1_clear"] > 0.5) & fault_eval
            fault_top = (1.0 - fault_predicted_risk).argmax(dim=-1)
            self._maze_diag_fault_top1_total += fault_clear.float().sum()
            self._maze_diag_fault_top1_correct += (
                fault_clear & (teacher_top == fault_top)
            ).float().sum()
            fault_labeled = (teacher_class < 5) & fault_eval
            if bool(fault_labeled.any()):
                fault_flat = (
                    teacher_class[fault_labeled] * 6
                    + fault_predicted_scene[fault_labeled]
                )
                self._maze_diag_fault_scene_confusion += torch.bincount(
                    fault_flat, minlength=30
                ).reshape(5, 6).to(self._maze_diag_fault_scene_confusion)

    def _maze_diagnostic_summary(self) -> dict[str, float]:
        def histogram_auc(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
            negative_below = torch.cumsum(negative, dim=0) - negative
            numerator = (positive * (negative_below + 0.5 * negative)).sum()
            return numerator / (positive.sum() * negative.sum()).clamp_min(1.0)

        def scene_macro_f1(confusion: torch.Tensor) -> torch.Tensor:
            values = []
            for index in range(5):
                true_positive = confusion[index, index]
                false_negative = confusion[index, :].sum() - true_positive
                false_positive = confusion[:, index].sum() - true_positive
                denominator = 2.0 * true_positive + false_positive + false_negative
                if float(confusion[index, :].sum().detach().cpu()) > 0.0:
                    values.append(
                        2.0 * true_positive / denominator.clamp_min(1.0)
                    )
            return (
                torch.stack(values).mean()
                if values
                else confusion.new_zeros(())
            )

        auc = histogram_auc(
            self._maze_diag_risk_positive_hist,
            self._maze_diag_risk_negative_hist,
        )
        goal_auc = histogram_auc(
            self._maze_diag_goal_risk_positive_hist,
            self._maze_diag_goal_risk_negative_hist,
        )
        confusion = self._maze_diag_scene_confusion
        macro_f1 = scene_macro_f1(confusion)
        goal_macro_f1 = scene_macro_f1(self._maze_diag_goal_scene_confusion)
        fault_auc = histogram_auc(
            self._maze_diag_fault_risk_positive_hist,
            self._maze_diag_fault_risk_negative_hist,
        )
        fault_macro_f1 = scene_macro_f1(self._maze_diag_fault_scene_confusion)
        return {
            "teacher_coverage": float(
                (self._maze_diag_valid / self._maze_diag_total.clamp_min(1.0))
                .detach()
                .cpu()
            ),
            "wall_auroc": float(auc.detach().cpu()),
            "wall_miss_rate": float(
                (
                    self._maze_diag_wall_missed
                    / self._maze_diag_wall_positive.clamp_min(1.0)
                )
                .detach()
                .cpu()
            ),
            "safe_top1_accuracy": float(
                (
                    self._maze_diag_top1_correct
                    / self._maze_diag_top1_total.clamp_min(1.0)
                )
                .detach()
                .cpu()
            ),
            "scene_macro_f1": float(macro_f1.detach().cpu()),
            "goal_wall_auroc": float(goal_auc.detach().cpu()),
            "goal_safe_top1_accuracy": float(
                (
                    self._maze_diag_goal_top1_correct
                    / self._maze_diag_goal_top1_total.clamp_min(1.0)
                )
                .detach()
                .cpu()
            ),
            "goal_scene_macro_f1": float(goal_macro_f1.detach().cpu()),
            "fault_wall_auroc": float(fault_auc.detach().cpu()),
            "fault_safe_top1_accuracy": float(
                (
                    self._maze_diag_fault_top1_correct
                    / self._maze_diag_fault_top1_total.clamp_min(1.0)
                ).detach().cpu()
            ),
            "fault_scene_macro_f1": float(fault_macro_f1.detach().cpu()),
            "clean_live_latent_cosine": float(
                (
                    self._maze_diag_latent_cosine_sum
                    / self._maze_diag_latent_cosine_count.clamp_min(1.0)
                )
                .detach()
                .cpu()
            ),
            "wall_positive_samples": float(self._maze_diag_wall_positive.detach().cpu()),
            "safe_top1_samples": float(self._maze_diag_top1_total.detach().cpu()),
            "scene_samples": float(confusion.sum().detach().cpu()),
            "fault_samples": float(
                self._maze_diag_fault_scene_confusion.sum().detach().cpu()
            ),
        }

    def _extra_tick_diagnostics(self) -> dict[str, torch.Tensor]:
        event_denominator = self._push_interval_count.clamp_min(1.0).unsqueeze(-1)
        interval_delta_mean = self._push_interval_delta_sum / event_denominator
        raw_goal_distance = torch.linalg.vector_norm(
            self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE],
            dim=-1,
        )
        goal_diagnostics = self.goal_belief.last_diagnostics
        pending = getattr(self, "pending_tick", {}) or {}
        safe3 = pending.get("safe3", torch.zeros(self.num_envs, 3, device=self.device))
        teacher_valid = pending.get(
            "safety_valid", torch.zeros(self.num_envs, 1, device=self.device)
        ).reshape(-1) > 0.5
        scene_diag = p4_contract.safety_scene_diagnostics(
            safe3,
            teacher_valid,
        )
        self._accumulate_maze_diagnostic(
            safe3,
            teacher_valid,
            pending.get(
                "nav_feat", torch.zeros(self.num_envs, 32, device=self.device)
            ),
            pending.get(
                "nav_nonvisual",
                torch.zeros(
                    self.num_envs,
                    p2_contract.NAV_NONVISUAL_DIM,
                    device=self.device,
                ),
            )[:, :4],
            self._diagnostic_fault_nav_feat,
            self._diagnostic_fault_mask,
        )
        teacher_top = safe3.argmax(dim=-1)
        head_top = (1.0 - self._last_safety_head_risk3).argmax(dim=-1)
        policy_weights = p2_contract.command_direction_weights(self._last_policy_command)
        actor_top = policy_weights.argmax(dim=-1)
        teacher_clear = scene_diag["teacher_safe_top1_clear"] > 0.5
        head_correct = teacher_clear & (head_top == teacher_top)
        actor_wrong = head_correct & (actor_top != teacher_top)
        reset_now = pending.get(
            "reset_mask", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        ).reshape(-1).bool() | self._diagnostic_terminal_mask
        if bool(reset_now.any()):
            self._risk_event_active[reset_now] = False
            self._risk_event_age_ticks[reset_now] = 0
            self._risk_event_baseline_policy_vx[reset_now] = 0.0
            self._risk_event_baseline_limited_vx[reset_now] = 0.0
            self._risk_condition_previous[reset_now] = False
        center_risk_condition = (
            ~reset_now
            & teacher_valid
            & ((1.0 - safe3[:, 1]) >= 0.65)
            & (self._last_policy_command[:, 0] >= 0.50)
            & (self._last_goal_freshness >= 0.50)
        )
        new_risk_event = center_risk_condition & ~self._risk_condition_previous
        self._risk_condition_previous.copy_(center_risk_condition)
        self._risk_event_active |= new_risk_event
        self._risk_event_baseline_policy_vx[new_risk_event] = (
            self._last_policy_command[new_risk_event, 0]
        )
        self._risk_event_baseline_limited_vx[new_risk_event] = (
            self._last_limited_command[new_risk_event, 0]
        )
        self._risk_event_age_ticks[new_risk_event] = 0
        eligible_response = self._risk_event_active & ~new_risk_event
        policy_decel_mask = eligible_response & (
            self._last_policy_command[:, 0]
            <= self._risk_event_baseline_policy_vx - 0.15
        )
        limited_decel_mask = eligible_response & (
            self._last_limited_command[:, 0]
            <= self._risk_event_baseline_limited_vx - 0.15
        )
        self._risk_event_age_ticks[eligible_response] += 1
        no_decel_mask = (
            self._risk_event_active
            & (self._risk_event_age_ticks >= self._risk_response_ticks)
            & ~policy_decel_mask
            & ~limited_decel_mask
        )
        finished = policy_decel_mask | limited_decel_mask | no_decel_mask
        self._risk_event_active[finished] = False
        self._risk_event_age_ticks[finished] = 0
        policy_decel = policy_decel_mask.float()
        limited_decel = limited_decel_mask.float()
        no_decel = no_decel_mask.float()

        def expanded_quantile(value: torch.Tensor, quantile: float) -> torch.Tensor:
            finite = value.reshape(-1)[torch.isfinite(value.reshape(-1))]
            scalar = (
                torch.quantile(finite.float(), quantile)
                if finite.numel()
                else value.new_zeros(())
            )
            return scalar.expand(self.num_envs)

        goal_bucket_metrics: dict[str, torch.Tensor] = {}
        accepted = goal_diagnostics.get(
            "goal_measurement_accepted", raw_goal_distance.new_zeros(self.num_envs)
        )
        clipped = goal_diagnostics.get(
            "goal_measurement_clipped", raw_goal_distance.new_zeros(self.num_envs)
        )
        rejected = goal_diagnostics.get(
            "goal_measurement_rejected", raw_goal_distance.new_zeros(self.num_envs)
        )
        measurement_due = (accepted + clipped + rejected) > 0.5
        for label, bucket in (
            ("0_5", raw_goal_distance < 5.0),
            ("5_10", (raw_goal_distance >= 5.0) & (raw_goal_distance < 10.0)),
            ("10_plus", raw_goal_distance >= 10.0),
        ):
            denominator = (bucket & measurement_due).float().sum().clamp_min(1.0)
            goal_bucket_metrics[f"goal_distance_{label}_share"] = (
                bucket.float().mean().expand(self.num_envs)
            )
            for outcome, source in (
                ("accept", accepted),
                ("clipped", clipped),
                ("reject", rejected),
            ):
                rate = (source * bucket.float()).sum() / denominator
                goal_bucket_metrics[f"goal_{outcome}_{label}"] = rate.expand(
                    self.num_envs
                )

        stuck_duration = self._p4_worker_extra[
            :, p4_contract.STUCK_DURATION_S_INDEX
        ]
        stuck_candidate = self._p4_worker_extra[
            :, p4_contract.STUCK_CANDIDATE_INDEX
        ] > 0.5
        candidate_duration = stuck_duration[stuck_candidate]
        if not candidate_duration.numel():
            candidate_duration = stuck_duration.new_zeros(1)
        stuck_reset = self._p4_worker_extra[
            :, p4_contract.STUCK_RESET_TRIGGERED_INDEX
        ] > 0.5
        collision_delay = self._p4_worker_extra[
            :, p4_contract.STUCK_COLLISION_TO_RESET_S_INDEX
        ]
        collision_delay_mean = (
            collision_delay[stuck_reset].mean()
            if bool(stuck_reset.any())
            else collision_delay.new_zeros(())
        )
        diagnostic_summary = self._maze_diagnostic_summary()
        diagnostic_metrics = {
            f"diagnostic_{name}": raw_goal_distance.new_full(
                (self.num_envs,), float(value)
            )
            for name, value in diagnostic_summary.items()
            if name
            in {
                "teacher_coverage",
                "wall_auroc",
                "wall_miss_rate",
                "safe_top1_accuracy",
                "scene_macro_f1",
                "clean_live_latent_cosine",
                "goal_wall_auroc",
                "goal_safe_top1_accuracy",
                "goal_scene_macro_f1",
                "fault_wall_auroc",
                "fault_safe_top1_accuracy",
                "fault_scene_macro_f1",
            }
        }
        resolved_branch = self._effective_maze_branch(self.session_effective_seconds)
        schedule = p4_contract.training_schedule(
            self.session_effective_seconds, branch=resolved_branch
        )
        values = {
            **goal_diagnostics,
            **goal_bucket_metrics,
            **self._camera_diagnostics,
            **self._p4_reward_diagnostics,
            "camera_clean_live_latent_cosine": self._camera_clean_live_latent_cosine,
            "user_speed_cap": self.user_speed_cap,
            "effective_speed_cap": self.effective_speed_cap,
            "safety_speed_cap": self.safety_speed_cap,
            "safety_cap_predictive_risk": self._safety_cap_predictive_risk,
            **scene_diag,
            **diagnostic_metrics,
            "safety_head_risk_left": self._last_safety_head_risk3[:, 0],
            "safety_head_risk_center": self._last_safety_head_risk3[:, 1],
            "safety_head_risk_right": self._last_safety_head_risk3[:, 2],
            "head_correct_samples": head_correct.float(),
            "head_correct_actor_wrong_count": actor_wrong.float(),
            "risk_event_resolved_count": finished.float(),
            "risk_decel_policy_count": policy_decel,
            "risk_decel_limited_count": limited_decel,
            "risk_no_deceleration_count": no_decel,
            "diagnostic_fault_shadow_share": self._diagnostic_fault_mask.float(),
            "diagnostic_clean_fault_latent_cosine": (
                self._diagnostic_clean_fault_latent_cosine
            ),
            "diagnostic_clean_fault_action_mae": (
                self._diagnostic_clean_fault_action_mae
            ),
            "maze_branch_actor_attack": raw_goal_distance.new_full(
                (self.num_envs,), float(resolved_branch == "actor_attack")
            ),
            "maze_branch_visual_recovery": raw_goal_distance.new_full(
                (self.num_envs,), float(resolved_branch == "visual_recovery")
            ),
            "maze_phase_probe": raw_goal_distance.new_full(
                (self.num_envs,), float(schedule["phase"] == "mazeprobe")
            ),
            "maze_phase_attack": raw_goal_distance.new_full(
                (self.num_envs,), float(schedule["phase"] == "mazeattack")
            ),
            "maze_phase_hard": raw_goal_distance.new_full(
                (self.num_envs,), float(schedule["phase"] == "mazehard")
            ),
            "maze_phase_final": raw_goal_distance.new_full(
                (self.num_envs,), float(schedule["phase"] == "mazefinal")
            ),
            "zero_hidden_action_mae": self._zero_hidden_action_mae,
            "zero_hidden_direction_disagreement": self._zero_hidden_direction_disagreement,
            "push_epoch": self.push_epoch.float(),
            "seconds_since_push": self.seconds_since_push,
            "push_runtime_active": self._p4_extra[:, p3_contract.PUSH_RUNTIME_ACTIVE_INDEX],
            "push_telemetry_valid": self._p4_extra[:, p3_contract.PUSH_TELEMETRY_VALID_INDEX],
            "push_delta_vx": self._p4_extra[:, p3_contract.PUSH_DELTA_VELOCITY_SLICE.start],
            "push_delta_vy": self._p4_extra[:, p3_contract.PUSH_DELTA_VELOCITY_SLICE.start + 1],
            "push_event_count": self._push_interval_count,
            "push_lifetime_count": self._push_lifetime_count,
            "push_env_coverage": self._push_env_seen.float(),
            "push_actual_delta_vx_mean": interval_delta_mean[:, 0],
            "push_actual_delta_vy_mean": interval_delta_mean[:, 1],
            "push_actual_delta_vx_abs_max": self._push_interval_delta_abs_max[:, 0],
            "push_actual_delta_vy_abs_max": self._push_interval_delta_abs_max[:, 1],
            "push_recovery_pending": self._push_recovery_pending.float(),
            "push_tracking_recovery_time_s": torch.nan_to_num(
                self._push_recovery_time, nan=0.0
            ),
            "raw_goal_distance_gt10_share": (raw_goal_distance > 10.0).float(),
            "goal_innovation_d2_p50": expanded_quantile(
                goal_diagnostics["goal_innovation_d2"][measurement_due], 0.50
            ),
            "goal_innovation_d2_p90": expanded_quantile(
                goal_diagnostics["goal_innovation_d2"][measurement_due], 0.90
            ),
            "goal_innovation_d2_p99": expanded_quantile(
                goal_diagnostics["goal_innovation_d2"][measurement_due], 0.99
            ),
            "goal_age_s_p50": expanded_quantile(
                goal_diagnostics["goal_age_s"], 0.50
            ),
            "goal_age_s_p90": expanded_quantile(
                goal_diagnostics["goal_age_s"], 0.90
            ),
            "motion_confined_share": self._p4_worker_extra[
                :, p4_contract.STUCK_MOTION_CONFINED_INDEX
            ],
            "wall_evidence_share": self._p4_worker_extra[
                :, p4_contract.STUCK_WALL_EVIDENCE_INDEX
            ],
            "wall_stuck_candidate_share": self._p4_worker_extra[
                :, p4_contract.STUCK_CANDIDATE_INDEX
            ],
            "wall_stuck_duration_s": self._p4_worker_extra[
                :, p4_contract.STUCK_DURATION_S_INDEX
            ],
            "wall_stuck_would_reset": self._p4_worker_extra[
                :, p4_contract.STUCK_WOULD_RESET_INDEX
            ],
            "wall_stuck_reset_triggered": self._p4_worker_extra[
                :, p4_contract.STUCK_RESET_TRIGGERED_INDEX
            ],
            "wall_stuck_mapping_valid": self._p4_worker_extra[
                :, p4_contract.STUCK_MAPPING_VALID_INDEX
            ],
            "reset_after_push_share": self._p4_worker_extra[
                :, p4_contract.STUCK_RESET_AFTER_PUSH_INDEX
            ],
            "wall_stuck_saved_seconds": self._p4_worker_extra[
                :, p4_contract.STUCK_SAVED_SECONDS_INDEX
            ],
            "wall_stuck_duration_p50_s": expanded_quantile(
                candidate_duration, 0.50
            ),
            "wall_stuck_duration_p90_s": expanded_quantile(
                candidate_duration, 0.90
            ),
            "collision_to_stuck_reset_delay_s": collision_delay_mean.expand(
                self.num_envs
            ),
            "wall_stuck_term_available": self._p4_worker_extra[
                :, p4_contract.STUCK_TERM_AVAILABLE_INDEX
            ],
            "wall_stuck_term_config_valid": self._p4_worker_extra[
                :, p4_contract.STUCK_TERM_CONFIG_VALID_INDEX
            ],
        }
        normalized = getattr(self, "_last_normalized_action", None)
        mapped = getattr(self, "_last_mapped_command", None)
        if torch.is_tensor(normalized) and normalized.shape == (self.num_envs, 3):
            for index, axis in enumerate(("vx", "vy", "wz")):
                values[f"normalized_action_{axis}"] = normalized[:, index]
        if torch.is_tensor(mapped) and mapped.shape == (self.num_envs, 3):
            for index, axis in enumerate(("vx", "vy", "wz")):
                values[f"mapped_cmd_{axis}"] = mapped[:, index]
        for label, command in (
            ("policy_target", self._last_policy_command),
            ("limited_target", self._last_limited_command),
        ):
            if torch.is_tensor(command) and command.shape == (self.num_envs, 3):
                for index, axis in enumerate(("vx", "vy", "wz")):
                    values[f"{label}_{axis}"] = command[:, index]
        result = {
            name: torch.nan_to_num(value).detach().reshape(-1, 1).clone()
            for name, value in values.items()
            if torch.is_tensor(value) and value.numel() == self.num_envs
        }
        self._push_interval_count.zero_()
        self._push_interval_delta_sum.zero_()
        self._push_interval_delta_abs_max.zero_()
        return result

    def _response_append_kwargs(self) -> dict[str, object]:
        return {
            "push_epoch": self.push_epoch,
            "seconds_since_push": self.seconds_since_push,
        }

    # ------------------------------------------------------------------
    # P4 optimizer schedule and persistence
    # ------------------------------------------------------------------

    def _effective_maze_branch(self, seconds: float) -> str:
        del seconds
        configured = str(self.maze_training_branch or "actor_attack")
        if configured != "auto":
            return configured
        if self.diagnostic_elapsed_seconds < p4_contract.DIAGNOSTIC_SECONDS:
            return "auto"
        if self._resolved_maze_training_branch is None:
            summary = self._maze_diagnostic_summary()
            sufficient_samples = (
                summary["wall_positive_samples"] >= 100.0
                and summary["safe_top1_samples"] >= 100.0
                and summary["scene_samples"] >= 100.0
            )
            perception_passed = (
                sufficient_samples
                and summary["teacher_coverage"] >= 0.90
                and summary["wall_auroc"] >= 0.85
                and summary["wall_miss_rate"] <= 0.15
                and summary["safe_top1_accuracy"] >= 0.70
                and summary["scene_macro_f1"] >= 0.70
                and summary["clean_live_latent_cosine"] >= 0.90
                and summary["wall_auroc"] >= summary["goal_wall_auroc"] + 0.05
                and summary["safe_top1_accuracy"]
                >= summary["goal_safe_top1_accuracy"] + 0.10
                and summary["scene_macro_f1"]
                >= summary["goal_scene_macro_f1"] + 0.10
                and summary["wall_auroc"] >= 0.55
                and summary["safe_top1_accuracy"] >= (1.0 / 3.0) + 0.10
                and summary["scene_macro_f1"] >= 0.30
                and summary["fault_samples"] >= 50.0
                and summary["fault_wall_auroc"]
                >= summary["wall_auroc"] - 0.10
                and summary["fault_safe_top1_accuracy"]
                >= summary["safe_top1_accuracy"] - 0.15
            )
            self._resolved_maze_training_branch = (
                "actor_attack" if perception_passed else "visual_recovery"
            )
            if self.logger:
                self.logger.info(
                    f"[P4Maze] auto diagnostic selected "
                    f"branch={self._resolved_maze_training_branch} "
                    f"coverage={summary['teacher_coverage']:.3f} "
                    f"wall_auroc={summary['wall_auroc']:.3f} "
                    f"wall_miss={summary['wall_miss_rate']:.3f} "
                    f"top1={summary['safe_top1_accuracy']:.3f} "
                    f"scene_f1={summary['scene_macro_f1']:.3f} "
                    f"clean_live_cos={summary['clean_live_latent_cosine']:.3f} "
                    f"goal_auc={summary['goal_wall_auroc']:.3f} "
                    f"goal_top1={summary['goal_safe_top1_accuracy']:.3f} "
                    f"goal_scene_f1={summary['goal_scene_macro_f1']:.3f} "
                    f"fault_auc={summary['fault_wall_auroc']:.3f} "
                    f"fault_top1={summary['fault_safe_top1_accuracy']:.3f} "
                    f"fault_scene_f1={summary['fault_scene_macro_f1']:.3f} "
                    f"sufficient_samples={sufficient_samples}"
                )
        return self._resolved_maze_training_branch

    def _apply_training_schedule(self, session_effective_seconds, **_kwargs):
        self.session_effective_seconds = max(0.0, float(session_effective_seconds))
        self.effective_training_seconds = self.session_effective_seconds
        self.lifetime_effective_seconds = self.lifetime_base_seconds + self.session_effective_seconds
        schedule = p4_contract.training_schedule(
            self.session_effective_seconds,
            branch=self._effective_maze_branch(self.session_effective_seconds),
        )
        self.cnn_unfrozen = float(schedule["navigation_multiplier"]) > 0.0
        self.entropy_coefficient = float(schedule["entropy_coefficient"])
        self.optimizer_phase = str(schedule["phase"])
        if self.actor_optimizer is not None:
            for group in self.actor_optimizer.param_groups:
                name = str(group.get("name", ""))
                if name == "navigation_safety_head":
                    base = p4_contract.SAFETY_HEAD_LR
                    multiplier = float(schedule["safety_head_multiplier"])
                elif name.startswith("navigation_"):
                    layer = name.removeprefix("navigation_")
                    base = p4_contract.NAVIGATION_ENCODER_LRS[layer]
                    multiplier = float(schedule["navigation_multiplier"])
                else:
                    base = p4_contract.ACTOR_LR
                    multiplier = float(schedule["actor_multiplier"])
                group["base_lr"] = base
                group["lr"] = base * multiplier
                trainable = multiplier > 0.0
                for parameter in group["params"]:
                    parameter.requires_grad_(trainable)
                    if not trainable:
                        parameter.grad = None
        if self.critic_optimizer is not None:
            self.critic_optimizer.param_groups[0]["lr"] = (
                p4_contract.CRITIC_LR * float(schedule["critic_multiplier"])
            )
        if self.response_optimizer is not None:
            self.response_optimizer.param_groups[0]["lr"] = (
                p4_contract.ADAPTER_LR * float(schedule["adapter_multiplier"])
            )
        return schedule

    def maybe_unfreeze_cnn(self, effective_seconds: float) -> bool:
        del effective_seconds
        previous = self.cnn_unfrozen
        self._apply_training_schedule(self.session_effective_seconds)
        return (not previous) and self.cnn_unfrozen

    def update_training_clocks(self, session_wall_seconds: float) -> None:
        self.session_wall_seconds = max(0.0, float(session_wall_seconds))
        if self.maze_training_branch == "auto":
            self.diagnostic_elapsed_seconds = min(
                self.session_wall_seconds, p4_contract.DIAGNOSTIC_SECONDS
            )
            if (
                self._training_clock_origin_seconds is None
                and self.session_wall_seconds >= p4_contract.DIAGNOSTIC_SECONDS
            ):
                # The diagnostic decision is applied at this rollout boundary.
                # Start the gradient-training clock here so the diagnostic's
                # final partial rollout cannot consume the 28800-second budget.
                self._training_clock_origin_seconds = self.session_wall_seconds
            training_seconds = (
                0.0
                if self._training_clock_origin_seconds is None
                else max(
                    0.0,
                    self.session_wall_seconds - self._training_clock_origin_seconds,
                )
            )
        else:
            self.diagnostic_elapsed_seconds = 0.0
            self._training_clock_origin_seconds = 0.0
            training_seconds = self.session_wall_seconds
        previous = self.cnn_unfrozen
        self._apply_training_schedule(training_seconds)
        if (
            self.rollout is not None
            and self.rollout.step == 0
            and previous != self.cnn_unfrozen
        ):
            self.rollout = self.rollout.reset(store_depth=self.cnn_unfrozen)

    @property
    def current_phase(self) -> str:
        return str(
            p4_contract.training_schedule(
                self.session_effective_seconds,
                branch=self._effective_maze_branch(self.session_effective_seconds),
            )["phase"]
        )

    def update(self) -> dict[str, float]:
        if self.current_phase == "mazediag":
            self.current_iteration += 1
            metrics = {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "adapter_loss": 0.0,
                "adapter_updates": 0.0,
                "updates": 0.0,
                "diagnostic_read_only": 1.0,
            }
            metrics.update(self._training_monitor_metrics())
            self.rollout = self.rollout.reset(store_depth=self.cnn_unfrozen)
            self.rollout_invalid = False
            return metrics
        return super().update()

    def reset_live_state(self) -> None:
        super().reset_live_state()
        if hasattr(self, "goal_belief"):
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.goal_belief.reset(mask)
            self.camera_state.reset(mask, randomize_capture_phase=False)
            self._cached_low_frame_id.fill_(-1)
            self._teacher_hidden = None
            self._goal_epoch_changed_since_tick.zero_()
            self._risk_event_active.zero_()
            self._risk_event_age_ticks.zero_()
            self._risk_event_baseline_policy_vx.zero_()
            self._risk_event_baseline_limited_vx.zero_()
            self._risk_condition_previous.zero_()
            self._p4_episode_return.zero_()
            self._diagnostic_fault_depth = None
            self._diagnostic_fault_mask.zero_()
            self._diagnostic_terminal_mask.zero_()
            self._diagnostic_fault_nav_feat.zero_()
            self._diagnostic_clean_fault_latent_cosine.fill_(1.0)
            self._diagnostic_clean_fault_action_mae.zero_()

    def _feedback_contract(self) -> tuple[dict[str, object], str]:
        contract = {
            "profile": self.config.get("feedback_profile", {}),
            "implementation_sha256": feedback_implementation_digest(),
        }
        return contract, p4_contract.stable_digest(contract)

    def _configure_adapter_contract(self) -> None:
        if self.response_buffer is None or self.low_level_state_digest is None:
            return
        _, feedback_digest = self._feedback_contract()
        self.response_buffer.set_record_contract(
            p4_contract.adapter_record_contract(
                low_level_digest=self.low_level_state_digest,
                feedback_digest=feedback_digest,
            )
        )
        self.response_buffer.enable_p4_compatible_replay()

    def load_bundle(self, path: str, *, platform_model_id) -> str:
        raw = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(raw, dict):
            raise ValueError("P4 checkpoint payload must be a mapping")
        if raw.get("stage_type") == self.STAGE_TYPE:
            bundle, _ = normalize_kaiwu_train_bundle(raw)
            contracts = bundle.get("contracts", {})
            exact_training_contract = p4_contract.training_contract(
                self.stuck_reset_contract
            )
            exact_reward_contract = p4_contract.reward_contract(
                self.stuck_reset_contract
            )
            exact_command_contract = p4_contract.command_contract()
            saved_training = contracts.get("training")
            saved_reward = contracts.get("reward")
            saved_command = contracts.get("command")
            if (
                isinstance(saved_training, dict)
                and saved_training.get("version")
                == p4_contract.CHECKPOINT_CONTRACT_VERSION
                and saved_training != exact_training_contract
            ):
                raise ValueError("P4 exact resume training contract mismatch")
            if (
                isinstance(saved_training, dict)
                and saved_training.get("version")
                == p4_contract.CHECKPOINT_CONTRACT_VERSION
                and saved_reward != exact_reward_contract
            ):
                raise ValueError("P4 exact resume reward contract mismatch")
            if (
                isinstance(saved_training, dict)
                and saved_training.get("version")
                == p4_contract.CHECKPOINT_CONTRACT_VERSION
                and saved_command != exact_command_contract
            ):
                raise ValueError("P4 exact resume command contract mismatch")
            exact_compatible = (
                saved_training == exact_training_contract
                and saved_reward == exact_reward_contract
                and saved_command == exact_command_contract
            )
            original_p4_state = copy.deepcopy(
                bundle.get("training_states", {}).get("p4", {})
            )
            runtime_maze_branch = self.maze_training_branch
            if exact_compatible:
                self.maze_training_branch = str(
                    original_p4_state.get(
                        "maze_training_branch", self.maze_training_branch
                    )
                )
                saved_resolved = original_p4_state.get(
                    "resolved_maze_training_branch"
                )
                self._resolved_maze_training_branch = (
                    str(saved_resolved)
                    if isinstance(saved_resolved, str)
                    else None
                )
                self.diagnostic_elapsed_seconds = float(
                    original_p4_state.get("diagnostic_elapsed_seconds", 0.0)
                )
            else:
                # The parent P2 loader validates the optimizer phase before P4
                # state is restored.  Use a deterministic temporary phase only
                # for loading compatible optimizer state; the new diagnostic
                # branch is reset immediately after the warm start.
                self.maze_training_branch = runtime_maze_branch
                self._resolved_maze_training_branch = "actor_attack"
                self.diagnostic_elapsed_seconds = p4_contract.DIAGNOSTIC_SECONDS
            compatible = copy.deepcopy(bundle)
            compatible["contracts"]["reward"] = p2_contract.reward_contract()
            compatible["contracts"]["training"] = p2_contract.training_contract()
            compatible["contracts"]["command"] = p2_contract.command_contract()
            if not exact_compatible:
                high_state = compatible.get("training_states", {}).get("high_level", {})
                if isinstance(high_state, dict):
                    saved_seconds = float(high_state.get("session_effective_seconds", 0.0))
                    warm_schedule = p4_contract.training_schedule(
                        saved_seconds,
                        branch="actor_attack",
                    )
                    high_state["optimizer_phase"] = str(warm_schedule["phase"])
                    high_state["entropy_coefficient"] = float(
                        warm_schedule["entropy_coefficient"]
                    )
                    high_state["cnn_unfrozen"] = (
                        float(warm_schedule["navigation_multiplier"]) > 0.0
                    )
                    # A previous-contract package starts a new P4 session.  Its
                    # serialized generator states may belong to another device
                    # backend (CUDA and CPU generator formats are different), so
                    # keep the runtime's deterministic fresh seeds while still
                    # using the strict parent loader for modules and optimizers.
                    warm_rng_report = {}
                    for key, generator in (
                        ("shuffle_rng_state", self.ppo_generator),
                        ("action_rng_state", self.action_generator),
                        ("vy_action_rng_state", self.vy_action_generator),
                        ("neutral_rng_state", self.neutral_generator),
                    ):
                        high_state[key] = generator.get_state().cpu()
                        warm_rng_report[key] = {
                            "status": "fresh_seed",
                            "reason": "new_p4_session",
                        }
                    migration = high_state.get("optimizer_migration_report")
                    if not isinstance(migration, dict):
                        migration = {}
                    high_state["optimizer_migration_report"] = {
                        **migration,
                        "p4_warm_start_rng": warm_rng_report,
                    }
                response_state = compatible.get("training_states", {}).get(
                    "response_adapter", {}
                )
                if isinstance(response_state, dict):
                    response_state["rng_state"] = self.adapter_generator.get_state().cpu()
            mode = self._load_exact_resume(compatible, platform_model_id, path)
            saved_low_digest = (bundle.get("lineage") or {}).get(
                "low_level_frozen_digest"
            )
            actual_low_digest = self._module_digest(
                (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
            )
            if saved_low_digest != actual_low_digest:
                raise ValueError(
                    "P4 exact resume frozen low-level digest mismatch: "
                    f"saved={saved_low_digest!r} actual={actual_low_digest!r}"
                )
            self._initial_low_digest = actual_low_digest
            self.parent_phase_label = (bundle.get("lineage") or {}).get(
                "p4_parent_phase"
            )
            if exact_compatible:
                self._load_p4_state(original_p4_state)
            else:
                self.parent_phase_label = str(bundle.get("phase_label", "p4_previous"))
                self.lifetime_base_seconds = float(self.lifetime_effective_seconds)
                self.maze_training_branch = runtime_maze_branch
                self._resolved_maze_training_branch = None
                self.session_wall_seconds = 0.0
                self.diagnostic_elapsed_seconds = 0.0
                self._training_clock_origin_seconds = None
                self.session_effective_seconds = 0.0
                self.effective_training_seconds = 0.0
                self.lifetime_effective_seconds = self.lifetime_base_seconds
                self.frame_count = 0
                self.nav_ticks = 0
                self.current_iteration = 0
                self._reset_maze_diagnostic_probes()
                self._reset_maze_diagnostic_state()
                self._apply_training_schedule(0.0)
                self.reset_live_state()
                if self.logger:
                    self.logger.warning(
                        "[P4NavPPO] previous P4 contract loaded as maze warm start; "
                        "session clock and live state reset"
                    )
            self._configure_adapter_contract()
            return f"p4_{mode if exact_compatible else 'maze_warm_start'}"
        disposition = validate_p3_eval_bundle(raw, mode="track")
        if self.logger and not disposition.get("phase_label_known", False):
            self.logger.warning(
                "[P4NavPPO] parent phase label is not recognized; continuing "
                "because structural validation passed. phase=%r",
                disposition.get("phase_label"),
            )
        bundle, _ = normalize_kaiwu_train_bundle(raw)
        modules = bundle["modules"]
        low = modules["low_level"]
        high = modules["high_level"]
        self._load_leaf(
            low, "locomotion_encoder", self.low_level_encoder,
            class_name="VisionEncoder", spec=self._low_encoder_spec(self.low_level_encoder),
            context="P4 P3.5 parent low_level",
        )
        self._load_leaf(
            low, "actor", self.low_level_actor,
            class_name="Actor77Sequential", spec=self._low_actor_spec(),
            context="P4 P3.5 parent low_level",
        )
        for name, module, class_name, spec in (
            ("navigation_encoder", self.navigation_encoder, "NavigationEncoder", navigation_encoder_spec()),
            ("actor", self.actor, "P2NavigationActor", navigation_actor_spec()),
            ("response_adapter", self.response_adapter, "CommandResponseAdapter", response_adapter_spec()),
        ):
            self._load_leaf(
                high, name, module, class_name=class_name, spec=spec,
                context="P4 P3.5 parent high_level",
            )
        self._load_leaf(
            high, "navigation_safety_head", self.safety_head,
            class_name="NavigationSafetyHead", spec=navigation_safety_head_spec(),
            context="P4 P3.5 parent high_level",
        )
        self.low_level_payload = low
        self.parent_optimizer_payload = bundle.get("transparent_parent_optimizers", bundle.get("optimizers", {}))
        self.parent_scheduler_payload = bundle.get("transparent_parent_schedulers", bundle.get("schedulers", {}))
        self.parent_training_payload = bundle.get("transparent_parent_training_states", bundle.get("training_states", {}))
        self.source_parent_model_id = self._bundle_identity(bundle, platform_model_id, path)
        self.loaded_platform_model_id = self.source_parent_model_id
        self.parent_phase_label = str(disposition["phase_label"])
        self.parent_checkpoint_sha256 = self._sha256(path)
        self.low_level_state_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        self._initial_low_digest = self.low_level_state_digest
        self.response_buffer.set_low_level_version(self.low_level_state_digest, 0)
        self._configure_adapter_contract()
        parent_buffer = (
            bundle.get("training_states", {})
            .get("response_adapter", {})
            .get("buffer")
        )
        if isinstance(parent_buffer, dict):
            self.response_buffer.load_parent_completed_records(parent_buffer)
        parent_high_state = bundle.get("training_states", {}).get("high_level", {})
        self.lifetime_base_seconds = float(
            parent_high_state.get("lifetime_effective_seconds", 0.0)
        ) if isinstance(parent_high_state, dict) else 0.0
        self.session_wall_seconds = 0.0
        self.diagnostic_elapsed_seconds = 0.0
        self._training_clock_origin_seconds = None
        self._resolved_maze_training_branch = None
        self.session_effective_seconds = 0.0
        self.effective_training_seconds = 0.0
        self.lifetime_effective_seconds = self.lifetime_base_seconds
        self.return_statistics = {
            "count": 0, "mean": 0.0, "m2": 0.0,
            "value_normalization_enabled": False,
        }
        self._reset_maze_diagnostic_probes()
        self._reset_maze_diagnostic_state()
        self._apply_training_schedule(0.0)
        self.reset_live_state()
        if self.logger:
            self.logger.info(
                "[P4NavPPO] final P3 parent loaded; low/high actor/nav/adapter preserved, "
                f"critic/return/optimizers rebuilt phase={disposition['phase_label']}"
            )
        return "p4_parent_warm_start_rebuilt_critic"

    def _p4_state(self) -> dict[str, object]:
        frozen_groups = {}
        if self.actor_optimizer is not None:
            for group in self.actor_optimizer.param_groups:
                name = str(group.get("name", "unnamed"))
                frozen_groups[name] = {
                    "lr": float(group.get("lr", 0.0)),
                    "trainable": all(
                        parameter.requires_grad for parameter in group["params"]
                    ),
                }
        return {
            "goal_belief": self.goal_belief.state_dict(),
            "camera": self.camera_state.state_dict(),
            "session_wall_seconds": self.session_wall_seconds,
            "diagnostic_elapsed_seconds": self.diagnostic_elapsed_seconds,
            "training_clock_origin_seconds": self._training_clock_origin_seconds,
            "maze_training_branch": self.maze_training_branch,
            "resolved_maze_training_branch": self._resolved_maze_training_branch,
            "maze_diagnostic": {
                "total": self._maze_diag_total.detach().cpu(),
                "valid": self._maze_diag_valid.detach().cpu(),
                "wall_positive": self._maze_diag_wall_positive.detach().cpu(),
                "wall_missed": self._maze_diag_wall_missed.detach().cpu(),
                "top1_total": self._maze_diag_top1_total.detach().cpu(),
                "top1_correct": self._maze_diag_top1_correct.detach().cpu(),
                "risk_positive_hist": self._maze_diag_risk_positive_hist.detach().cpu(),
                "risk_negative_hist": self._maze_diag_risk_negative_hist.detach().cpu(),
                "scene_confusion": self._maze_diag_scene_confusion.detach().cpu(),
                "latent_cosine_sum": self._maze_diag_latent_cosine_sum.detach().cpu(),
                "latent_cosine_count": self._maze_diag_latent_cosine_count.detach().cpu(),
                "goal_risk_positive_hist": self._maze_diag_goal_risk_positive_hist.detach().cpu(),
                "goal_risk_negative_hist": self._maze_diag_goal_risk_negative_hist.detach().cpu(),
                "goal_top1_total": self._maze_diag_goal_top1_total.detach().cpu(),
                "goal_top1_correct": self._maze_diag_goal_top1_correct.detach().cpu(),
                "goal_scene_confusion": self._maze_diag_goal_scene_confusion.detach().cpu(),
                "fault_risk_positive_hist": self._maze_diag_fault_risk_positive_hist.detach().cpu(),
                "fault_risk_negative_hist": self._maze_diag_fault_risk_negative_hist.detach().cpu(),
                "fault_top1_total": self._maze_diag_fault_top1_total.detach().cpu(),
                "fault_top1_correct": self._maze_diag_fault_top1_correct.detach().cpu(),
                "fault_scene_confusion": self._maze_diag_fault_scene_confusion.detach().cpu(),
                "nav_risk_probe": self._diagnostic_nav_risk_probe.state_dict(),
                "nav_scene_probe": self._diagnostic_nav_scene_probe.state_dict(),
                "goal_risk_probe": self._diagnostic_goal_risk_probe.state_dict(),
                "goal_scene_probe": self._diagnostic_goal_scene_probe.state_dict(),
                "probe_optimizer": self._diagnostic_probe_optimizer.state_dict(),
            },
            "camera_aux_coefficient": self._camera_aux_coefficient,
            "camera_aux_gradient_ratio": self._camera_aux_gradient_ratio,
            "action_mapper_version": p4_contract.ACTION_MAPPER_VERSION,
            "push_phase": p4_contract.push_phase_config(self.session_effective_seconds),
            "push_lifetime_count": self._push_lifetime_count.detach().cpu(),
            "push_env_seen": self._push_env_seen.detach().cpu(),
            "goal_belief_contract": {
                "process_sigma_v_m_s": p4_contract.GOAL_PROCESS_SIGMA_V_M_S,
                "process_sigma_wz_rad_s": p4_contract.GOAL_PROCESS_SIGMA_WZ_RAD_S,
                "reacquire_samples": p4_contract.GOAL_REACQUIRE_SAMPLES,
            },
            "stuck_reset_contract": dict(self.stuck_reset_contract),
            "maze_soft_cruise": {
                "preferred_vx": [
                    p4_contract.SOFT_CRUISE_MIN_VX,
                    p4_contract.SOFT_CRUISE_MAX_VX,
                ],
                "training_branch": self.maze_training_branch,
            },
            "optimizer_group_freeze_summary": frozen_groups,
        }

    def _load_p4_state(self, state: dict[str, object]) -> None:
        if state.get("action_mapper_version") != p4_contract.ACTION_MAPPER_VERSION:
            raise ValueError("P4 exact resume state mapper mismatch")
        saved_stuck = state.get("stuck_reset_contract")
        if not isinstance(saved_stuck, dict):
            raise ValueError("P4 exact resume missing stuck-reset contract")
        saved_stuck = p4_contract.normalize_stuck_reset_contract(saved_stuck)
        if saved_stuck != self.stuck_reset_contract:
            raise ValueError(
                "P4 exact resume stuck-reset contract mismatch: "
                f"saved={saved_stuck!r} runtime={self.stuck_reset_contract!r}"
            )
        self.goal_belief.load_state_dict(state.get("goal_belief", {}))
        self.session_wall_seconds = float(
            state.get("session_wall_seconds", self.session_effective_seconds)
        )
        self.diagnostic_elapsed_seconds = float(
            state.get("diagnostic_elapsed_seconds", 0.0)
        )
        if "training_clock_origin_seconds" not in state:
            raise ValueError("P4 exact resume missing training clock origin")
        clock_origin = state.get("training_clock_origin_seconds")
        self._training_clock_origin_seconds = (
            float(clock_origin)
            if isinstance(clock_origin, (int, float))
            else None
        )
        self.maze_training_branch = str(
            state.get("maze_training_branch", self.maze_training_branch)
        )
        resolved = state.get("resolved_maze_training_branch")
        self._resolved_maze_training_branch = (
            str(resolved) if isinstance(resolved, str) else None
        )
        diagnostic_state = state.get("maze_diagnostic")
        if not isinstance(diagnostic_state, dict):
            raise ValueError("P4 exact resume missing maze diagnostic state")
        diagnostic_targets = {
            "total": self._maze_diag_total,
            "valid": self._maze_diag_valid,
            "wall_positive": self._maze_diag_wall_positive,
            "wall_missed": self._maze_diag_wall_missed,
            "top1_total": self._maze_diag_top1_total,
            "top1_correct": self._maze_diag_top1_correct,
            "risk_positive_hist": self._maze_diag_risk_positive_hist,
            "risk_negative_hist": self._maze_diag_risk_negative_hist,
            "scene_confusion": self._maze_diag_scene_confusion,
            "latent_cosine_sum": self._maze_diag_latent_cosine_sum,
            "latent_cosine_count": self._maze_diag_latent_cosine_count,
            "goal_risk_positive_hist": self._maze_diag_goal_risk_positive_hist,
            "goal_risk_negative_hist": self._maze_diag_goal_risk_negative_hist,
            "goal_top1_total": self._maze_diag_goal_top1_total,
            "goal_top1_correct": self._maze_diag_goal_top1_correct,
            "goal_scene_confusion": self._maze_diag_goal_scene_confusion,
            "fault_risk_positive_hist": self._maze_diag_fault_risk_positive_hist,
            "fault_risk_negative_hist": self._maze_diag_fault_risk_negative_hist,
            "fault_top1_total": self._maze_diag_fault_top1_total,
            "fault_top1_correct": self._maze_diag_fault_top1_correct,
            "fault_scene_confusion": self._maze_diag_fault_scene_confusion,
        }
        for name, target in diagnostic_targets.items():
            saved = diagnostic_state.get(name)
            if not torch.is_tensor(saved) or saved.shape != target.shape:
                raise ValueError(
                    f"P4 exact resume invalid maze diagnostic state {name}"
                )
            target.copy_(saved.to(device=self.device, dtype=target.dtype))
        for name, probe in (
            ("nav_risk_probe", self._diagnostic_nav_risk_probe),
            ("nav_scene_probe", self._diagnostic_nav_scene_probe),
            ("goal_risk_probe", self._diagnostic_goal_risk_probe),
            ("goal_scene_probe", self._diagnostic_goal_scene_probe),
        ):
            probe_state = diagnostic_state.get(name)
            if not isinstance(probe_state, dict):
                raise ValueError(f"P4 exact resume missing diagnostic probe {name}")
            probe.load_state_dict(probe_state, strict=True)
        probe_optimizer = diagnostic_state.get("probe_optimizer")
        if not isinstance(probe_optimizer, dict):
            raise ValueError("P4 exact resume missing diagnostic probe optimizer")
        self._diagnostic_probe_optimizer.load_state_dict(probe_optimizer)
        self.camera_state.load_state_dict(state.get("camera", {}))
        push_lifetime_count = state.get("push_lifetime_count")
        push_env_seen = state.get("push_env_seen")
        if torch.is_tensor(push_lifetime_count) and push_lifetime_count.numel() == self.num_envs:
            self._push_lifetime_count.copy_(
                push_lifetime_count.reshape(-1).to(self.device)
            )
        if torch.is_tensor(push_env_seen) and push_env_seen.numel() == self.num_envs:
            self._push_env_seen.copy_(push_env_seen.reshape(-1).to(self.device).bool())
        self._camera_aux_coefficient = float(state.get("camera_aux_coefficient", 0.0))
        self._camera_aux_gradient_ratio = float(state.get("camera_aux_gradient_ratio", 0.0))
        self._apply_training_schedule(self.session_effective_seconds)

    def save_training_bundle(self, path: str, *, platform_model_id) -> str:
        super().save_training_bundle(path, platform_model_id=platform_model_id)
        payload = torch.load(path, weights_only=False, map_location="cpu")
        feedback, feedback_digest = self._feedback_contract()
        payload["contracts"] = {
            **p4_contract.contract_metadata(self.stuck_reset_contract),
            "feedback": feedback,
            "feedback_digest": feedback_digest,
            "critic_transport": {
                "wire_dim": p4_contract.P4_PRIVILEGED_WIRE_DIM,
                "critic_dim": p2_contract.CRITIC_OBS_DIM,
                "response_aux_dim": p2_contract.RESPONSE_AUX_DIM,
                "worker_aux_dim": p2_contract.WORKER_AUX_DIM,
                "training_tail_dim": (
                    p3_contract.P3_WORKER_EXTRA_DIM
                    + p4_contract.P4_WORKER_EXTRA_DIM
                ),
                "p3_training_tail_dim": p3_contract.P3_WORKER_EXTRA_DIM,
                "p4_training_tail_dim": p4_contract.P4_WORKER_EXTRA_DIM,
            },
        }
        payload["training_states"]["p4"] = self._p4_state()
        payload["training_states"]["global"]["train_scope"] = "high_level_and_response_adapter"
        payload["modules"]["high_level"]["action_mapper_version"] = p4_contract.ACTION_MAPPER_VERSION
        payload["lineage"]["p4_parent_phase"] = self.parent_phase_label
        payload["lineage"]["low_level_frozen_digest"] = self._initial_low_digest
        payload["capabilities"].update(
            action_mapper_version=p4_contract.ACTION_MAPPER_VERSION,
            goal_belief_version=p4_contract.GOAL_BELIEF_VERSION,
            standard_low_level_eval=True,
            track_full_eval=True,
        )
        current_low = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        if current_low != self._initial_low_digest:
            raise RuntimeError("P4 frozen low-level digest drift before checkpoint save")
        for name in ("navigation_safety_head", "critic"):
            payload["modules"]["high_level"][name]["training_only"] = True
        directory = os.path.dirname(path) or "."
        temporary = os.path.join(directory, f".{os.path.basename(path)}.{uuid4().hex}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        return self._sha256(path)

    def load_evaluation_bundle(self, path: str, *, platform_model_id) -> str:
        raw = torch.load(path, weights_only=False, map_location="cpu")
        if raw.get("stage_type") != self.STAGE_TYPE:
            raise ValueError("P4 Track evaluation requires stage_type=p4_nav_ppo")
        mapper = (raw.get("contracts", {}).get("command") or {}).get("mapper_version")
        if mapper != p4_contract.ACTION_MAPPER_VERSION:
            raise ValueError("P4 Track evaluation action mapper mismatch")
        return super().load_evaluation_bundle(path, platform_model_id=platform_model_id)

    def memory_metrics(self) -> dict[str, float]:
        result = super().memory_metrics()
        current_low = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        result.update(
            low_digest_drift=float(current_low != self._initial_low_digest),
            low_optimizer_steps=0.0,
            high_updates=float(self.actor_gradient_steps),
            mapper_version_valid=1.0,
            p4_tbptt_sequences_per_env=2.0,
            p4_tbptt_nav_ticks=16.0,
            push_term_assembly_valid=float(
                bool(
                    torch.all(
                        self._p4_extra[
                            :, p3_contract.PUSH_TELEMETRY_VALID_INDEX
                        ]
                        > 0.5
                    )
                )
            ),
            push_rollout_event_count_total=float(self._push_rollout_count.sum()),
            push_lifetime_event_count_total=float(self._push_lifetime_count.sum()),
            push_env_coverage_rate=float(self._push_env_seen.float().mean()),
        )
        if self.response_buffer is not None:
            result.update(
                adapter_target_current_ratio=0.50,
                adapter_target_p35_ratio=0.25,
                adapter_target_earlier_ratio=0.25,
            )
            result["adapter_compat_rejected_records"] = float(
                sum(self.response_buffer.compatibility_rejections.values())
            )
            result["adapter_compat_migrated_legacy_parent_records"] = float(
                self.response_buffer.legacy_parent_records_migrated
            )
            result["adapter_legacy_parent_rejected_records"] = float(
                sum(self.response_buffer.legacy_parent_migration_rejections.values())
            )
            for reason in p4_contract.ADAPTER_COMPATIBILITY_REJECTION_REASONS:
                result[p4_contract.adapter_compatibility_metric_name(reason)] = float(
                    self.response_buffer.compatibility_rejections.get(reason, 0)
                )
            for index, label in enumerate(("02s", "06s", "10s")):
                result[f"adapter_push_rejected_{label}"] = float(
                    self.response_buffer.push_horizon_rejections[index]
                )
            result["adapter_push_rejected_pose"] = float(
                self.response_buffer.push_pose_rejections
            )
        return result
