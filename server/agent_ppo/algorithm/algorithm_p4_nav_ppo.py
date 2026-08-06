#!/usr/bin/env python3
"""P4 Track navigation PPO over a structurally complete frozen P3 parent."""

from __future__ import annotations

import copy
import math
import os
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch import nn

from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
from agent_ppo.algorithm.algorithm_visual_ppo import _calibrate_auxiliary_gradients
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
    P4ActorStuckHead,
    assemble_actor_input,
    navigation_actor_spec,
    navigation_encoder_spec,
    navigation_safety_head_spec,
    p4_actor_stuck_head_spec,
)
from agent_ppo.model.response_adapter import response_adapter_spec


class AlgorithmP4NavPPO(AlgorithmP2NavPPO):
    STAGE_TYPE = p4_contract.STAGE_TYPE
    MAZE_PROFILES = {"maze_credit_repair", "maze_closed_loop_v3"}

    def __init__(self, *args, **kwargs):
        early_config = dict(kwargs.get("config") or {})
        early_training_profile = str(
            early_config.get("training_profile", "full_track")
        )
        early_command_contract = p4_contract.command_contract(
            early_training_profile
        )
        early_config.setdefault(
            "nav_period_frames", p4_contract.P4_NAV_PERIOD_FRAMES
        )
        early_config.setdefault(
            "slew_rate", tuple(early_command_contract["slew_rate"])
        )
        early_config.setdefault(
            "slew_release_rate",
            tuple(early_command_contract["slew_release_rate"]),
        )
        kwargs["config"] = early_config
        self.maze_training_branch = str(
            early_config.get("maze_training_branch", "actor_attack")
        )
        self.training_profile = early_training_profile
        self._resolved_maze_training_branch = None
        self.session_wall_seconds = 0.0
        self.diagnostic_elapsed_seconds = 0.0
        self._training_clock_origin_seconds = None
        super().__init__(*args, **kwargs)
        runtime_segments = tuple(
            self.config.get(
                "track_segment_labels", p4_contract.FULL_TRACK_SEGMENT_LABELS
            )
        )
        expected_segments = (
            ("maze",)
            if self.training_profile in self.MAZE_PROFILES
            else p4_contract.FULL_TRACK_SEGMENT_LABELS
        )
        if runtime_segments != expected_segments:
            raise ValueError(
                f"P4 {self.training_profile} requires segments "
                f"{list(expected_segments)!r}; "
                f"got {list(runtime_segments)!r}"
            )
        self.track_segment_labels = runtime_segments
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
        expected_command_contract = p4_contract.command_contract(
            self.training_profile
        )
        expected_slew = tuple(expected_command_contract["slew_rate"])
        expected_release = tuple(expected_command_contract["slew_release_rate"])
        if runtime_slew != expected_slew or runtime_release != expected_release:
            raise ValueError(
                "P4 runtime slew does not match command contract: "
                f"slew={runtime_slew!r} release={runtime_release!r} "
                f"expected={expected_slew!r}/{expected_release!r}"
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
        self.camera_fault_course_enabled = bool(
            self.config.get("camera_fault_course_enabled", True)
        )
        self.goal_fault_course_enabled = bool(
            self.config.get("goal_fault_course_enabled", True)
        )
        if not hasattr(self, "_resolved_maze_training_branch"):
            self._resolved_maze_training_branch = None
        self.user_speed_cap = torch.full(
            (self.num_envs,), p4_contract.P4_MAX_VX, device=self.device
        )
        self.safety_speed_cap = torch.ones(self.num_envs, device=self.device)
        self._translation_alpha_prev = torch.ones(
            self.num_envs, device=self.device
        )
        self._translation_limiter_diagnostics: dict[str, torch.Tensor] = {}
        self._near_goal_capture_diagnostics: dict[str, torch.Tensor] = {}
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
        self._delivered_depth = None
        self._mirror_batch_schedule: list[object | None] = []
        self._mirror_batch_cursor = 0
        self._mirror_aux_eligible_sequence_count = 0
        self._mirror_aux_scheduled_sequence_share = 0.0
        self._p4_recovery_monitor_state = {
            "event_times": [],
            "success_lifetime_count": 0,
            "candidate_lifetime_count": 0,
            "terminal_lifetime_count": 0,
        }
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
        self._spawn_segment = torch.zeros(self.num_envs, device=self.device)
        self._max_segment_reached = torch.zeros(self.num_envs, device=self.device)
        self._segment_state_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._maze_credit_earned = torch.zeros(self.num_envs, device=self.device)
        self._teacher_navigation_encoder = copy.deepcopy(self.navigation_encoder).to(
            self.device
        )
        self._teacher_actor = copy.deepcopy(self.actor).to(self.device)
        self._teacher_hidden = None
        self.stuck_head = (
            P4ActorStuckHead(self.actor.hidden_dim).to(self.device)
            if self.training_enabled
            else None
        )
        self.mirror_generator = torch.Generator(device="cpu")
        self.mirror_generator.manual_seed(seed + 31)
        self._auxiliary_calibration = {
            "combined_ratio": 0.0,
            "teacher_ratio": 0.0,
            "camera_ratio": 0.0,
            "mirror_ratio": 0.0,
            "stuck_ratio": 0.0,
            "teacher_valid_steps": 0.0,
            "stuck_valid_steps": 0.0,
        }
        self._auxiliary_coefficients: dict[str, float] = {}
        self._teacher_update_enabled = True
        self.actor_stuck_positive_ema = 0.10
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
            self.actor_optimizer.add_param_group(
                {
                    "params": list(self.stuck_head.parameters()),
                    "lr": p4_contract.SAFETY_HEAD_LR,
                    "base_lr": p4_contract.SAFETY_HEAD_LR,
                    "name": "actor_stuck_head",
                }
            )
            self._assert_optimizer_isolation()
            self._apply_training_schedule(self.session_effective_seconds)
            self._credit_fresh_actor_optimizer_state = copy.deepcopy(
                self.actor_optimizer.state_dict()
            )
            self._credit_fresh_critic_state = copy.deepcopy(self.critic.state_dict())
            self._credit_fresh_critic_optimizer_state = copy.deepcopy(
                self.critic_optimizer.state_dict()
            )
            self._credit_fresh_critic_scheduler_state = (
                copy.deepcopy(self.critic_scheduler.state_dict())
                if self.critic_scheduler is not None
                else None
            )
            self._credit_fresh_actor_scheduler_state = (
                copy.deepcopy(self.actor_scheduler.state_dict())
                if self.actor_scheduler is not None
                else None
            )
            self._credit_fresh_response_scheduler_state = (
                copy.deepcopy(self.response_scheduler.state_dict())
                if self.response_scheduler is not None
                else None
            )
            self._credit_fresh_stuck_head_state = copy.deepcopy(
                self.stuck_head.state_dict()
            )
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
        full_cap = torch.full(
            (normalized.shape[0],), p4_contract.P4_MAX_VX,
            device=normalized.device, dtype=normalized.dtype,
        )
        policy = p4_contract.map_normalized_action(
            normalized, full_cap, goal_freshness=None
        )
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
        limiter_shadow = self.training_profile == "maze_closed_loop_v3"
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

    @torch.no_grad()
    def frame_begin(self, obs: torch.Tensor, critic_wire: torch.Tensor, *, deterministic=False):
        """Attach recovery labels after the parent has sampled its safe3 teacher."""
        result, critic_obs, aux = super().frame_begin(
            obs, critic_wire, deterministic=deterministic
        )
        if not (self.training_enabled and result.get("is_tick") and self.pending_tick):
            return result, critic_obs, aux

        pending = self.pending_tick
        critic_obs, _ = self._split_transport(critic_wire)
        safe5, safe5_valid, _safe5_diagnostics = (
            p4_contract.privileged_safe_directions5(critic_obs)
        )
        reset = pending["reset_mask"].reshape(-1).bool()
        safe3 = pending["safe3"]
        scanner_valid = pending["safety_valid"].reshape(-1) > 0.5
        closed_loop_profile = self.training_profile == "maze_closed_loop_v3"
        mapping_valid = (
            self._p4_worker_extra[:, p4_contract.STUCK_MAPPING_VALID_INDEX] > 0.5
        )
        wall_evidence = (
            self._p4_worker_extra[:, p4_contract.STUCK_WALL_EVIDENCE_INDEX] > 0.5
        )
        worker_candidate = (
            self._p4_worker_extra[:, p4_contract.STUCK_CANDIDATE_INDEX] > 0.5
        )
        duration_s = torch.nan_to_num(
            self._p4_worker_extra[:, p4_contract.STUCK_DURATION_S_INDEX], nan=0.0
        ).clamp_min(0.0)
        policy_xy = pending["target_cmd3"][:, :2]
        exec_xy = self.command.exec_cmd[:, :2]
        true_xy = aux[:, 12:14]
        policy_or_exec_intent = (
            torch.linalg.vector_norm(policy_xy, dim=-1) > 0.10
        ) | (torch.linalg.vector_norm(exec_xy, dim=-1) > 0.08)
        true_motion_low = torch.linalg.vector_norm(true_xy, dim=-1) < 0.08
        push_grace = self.seconds_since_push < float(
            self.stuck_reset_contract.get("push_grace_s", 0.0)
        )
        episode_grace = reset
        alive = torch.isfinite(safe3).all(dim=-1) & torch.isfinite(true_xy).all(dim=-1)
        stuck_mask = (
            alive
            & mapping_valid
            & wall_evidence
            & policy_or_exec_intent
            & ~push_grace
            & ~episode_grace
        )
        stuck_label = (
            worker_candidate
            & true_motion_low
            & policy_or_exec_intent
            & wall_evidence
            & mapping_valid
            & ~push_grace
            & ~episode_grace
            & (duration_s >= 0.8)
        )
        stuck_weight = torch.where(
            duration_s >= 2.0,
            torch.ones_like(duration_s),
            torch.full_like(duration_s, 0.25),
        )
        stuck_weight = torch.where(stuck_label, stuck_weight, torch.ones_like(stuck_weight))
        guidance = p4_contract.teacher_guidance_mask(
            alive=alive,
            scanner_valid=scanner_valid,
            mapping_valid=mapping_valid,
            terminal=torch.zeros_like(alive),
            reset=reset,
            push_grace=push_grace,
            episode_grace=episode_grace,
            goal_freshness=self._last_goal_freshness,
            safe3=safe3,
            safe5=safe5 if closed_loop_profile else None,
        )
        teacher_mask = (
            guidance["teacher_guidance_eligible"].reshape(-1).bool()
            & (safe5_valid if closed_loop_profile else scanner_valid)
        )
        goal_mask = (
            guidance["teacher_guidance_goal_eligible"].reshape(-1).bool()
            & (safe5_valid if closed_loop_profile else scanner_valid)
        )
        mirror_eligible = (
            alive & mapping_valid & ~push_grace & ~worker_candidate
        )
        pending.update(
            {
                "teacher_safe3": safe3.detach(),
                "teacher_safe5": safe5.detach(),
                "teacher_goal_xy": self.goal_belief.estimate.detach(),
                "teacher_predictive_risk": pending["predictive_collision_risk"].reshape(-1, 1),
                "teacher_mask": teacher_mask.float().unsqueeze(-1),
                "teacher_goal_mask": goal_mask.float().unsqueeze(-1),
                "teacher_weight": stuck_weight.unsqueeze(-1),
                "stuck_label": stuck_label.float().unsqueeze(-1),
                "stuck_mask": stuck_mask.float().unsqueeze(-1),
                "mirror_eligible": mirror_eligible.float().unsqueeze(-1),
            }
        )
        return result, critic_obs, aux

    def begin_rollout(self) -> None:
        """Refresh the clean teacher from the current live policy."""
        self._push_rollout_count.zero_()
        self.camera_state.begin_rollout(
            self.session_effective_seconds,
            training=(self.training_enabled and self.camera_fault_course_enabled),
        )
        self._teacher_navigation_encoder.load_state_dict(
            self.navigation_encoder.state_dict(), strict=True
        )
        self._teacher_actor.load_state_dict(self.actor.state_dict(), strict=True)
        self._camera_aux_calibration_pending = True
        self._auxiliary_coefficients = {}

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

    def _actor_auxiliary_loss(
        self,
        *,
        normalized_mean,
        actor_normalized_mean=None,
        batch,
        ppo_actor_loss,
        actor_features=None,
        nav_feat=None,
    ):
        zero = normalized_mean.new_zeros(())
        if actor_normalized_mean is None:
            actor_normalized_mean = normalized_mean
        schedule = p4_contract.training_schedule(
            self.session_effective_seconds,
            branch=self._effective_maze_branch(self.session_effective_seconds),
        )
        specs: list[tuple[str, torch.Tensor, float]] = []
        mask3 = batch["camera_aux_mask"] > 0.5
        camera_selected = mask3.any(dim=-1)
        camera_raw = zero
        if bool(camera_selected.any()):
            camera_raw = F.smooth_l1_loss(
                normalized_mean,
                batch["clean_action_mean"],
                reduction="none",
            ).mean(dim=-1)[camera_selected].mean()
            specs.append(("camera", camera_raw, float(schedule.get("camera_aux_ratio", 0.01))))

        teacher_raw = zero
        teacher = {
            "direction": zero, "speed": zero, "yaw": zero, "edge": zero,
            "recovery": zero,
            "teacher_valid_steps": zero, "teacher_loss_active": zero,
            "teacher_edge_active_share": zero,
            "teacher_recovery_active_share": zero,
        }
        teacher_mask = batch["teacher_mask"].reshape(-1) > 0.5
        if self._teacher_update_enabled and bool(teacher_mask.any()):
            full_cap = torch.full(
                actor_normalized_mean.shape[:-1], p4_contract.P4_MAX_VX,
                dtype=actor_normalized_mean.dtype,
                device=actor_normalized_mean.device,
            )
            mean_command = p4_contract.map_normalized_action(
                actor_normalized_mean, full_cap, goal_freshness=None
            ).reshape(-1, 3)
            teacher_safe5 = (
                batch["teacher_safe5"]
                if self.training_profile == "maze_closed_loop_v3"
                else batch["teacher_safe3"]
            )
            teacher = p4_contract.teacher_guidance_loss(
                mean_command,
                teacher_safe5.reshape(-1, teacher_safe5.shape[-1]),
                batch["teacher_goal_xy"].reshape(-1, 2),
                batch["teacher_predictive_risk"].reshape(-1),
                batch["stuck_label"].reshape(-1) > 0.5,
                batch["teacher_mask"].reshape(-1) > 0.5,
                batch["teacher_goal_mask"].reshape(-1) > 0.5,
                sample_weight=batch["teacher_weight"].reshape(-1),
                min_valid_steps=1,
                closed_loop_v3=(self.training_profile == "maze_closed_loop_v3"),
            )
            teacher_raw = teacher["loss"]
            specs.append((
                "teacher", teacher_raw,
                min(
                    float(schedule.get("teacher_gradient_target_ratio", 0.02)),
                    float(schedule.get("teacher_gradient_hard_cap", 0.03)),
                ),
            ))

        stuck_raw = zero
        stuck_logits = None
        stuck_precision = zero
        stuck_recall = zero
        stuck_f1 = zero
        stuck_pr_auc = zero
        stuck_threshold = zero.new_tensor(0.5)
        stuck_positive_share = zero
        stuck_mask = batch["stuck_mask"].reshape(-1) > 0.5
        if actor_features is not None and bool(stuck_mask.any()) and self.stuck_head is not None:
            stuck_frozen = self.training_profile == "maze_closed_loop_v3"
            if stuck_frozen:
                # The closed-loop run freezes StuckHead. Keep its quality
                # metrics without retaining a graph through the Actor.
                with torch.no_grad():
                    stuck_logits = self.stuck_head(
                        actor_features.detach()
                    ).reshape(-1)
            else:
                stuck_logits = self.stuck_head(actor_features).reshape(-1)
            labels = batch["stuck_label"].reshape(-1)
            selected_labels = labels[stuck_mask]
            positive = selected_labels.mean().detach()
            self.actor_stuck_positive_ema = float(
                0.99 * self.actor_stuck_positive_ema + 0.01 * positive
            )
            pos_weight = min(
                10.0,
                max(1.0, (1.0 - self.actor_stuck_positive_ema) / max(self.actor_stuck_positive_ema, 1.0e-4)),
            )
            if stuck_frozen:
                with torch.no_grad():
                    stuck_raw = F.binary_cross_entropy_with_logits(
                        stuck_logits[stuck_mask], selected_labels,
                        pos_weight=torch.tensor(
                            pos_weight, device=stuck_logits.device
                        ),
                    )
            else:
                stuck_raw = F.binary_cross_entropy_with_logits(
                    stuck_logits[stuck_mask], selected_labels,
                    pos_weight=torch.tensor(pos_weight, device=stuck_logits.device),
                )
            with torch.no_grad():
                probabilities = torch.sigmoid(stuck_logits[stuck_mask])
                labels_bool = selected_labels > 0.5
                predicted = probabilities >= 0.5
                true_positive = (predicted & labels_bool).float().sum()
                false_positive = (predicted & ~labels_bool).float().sum()
                false_negative = (~predicted & labels_bool).float().sum()
                stuck_precision = true_positive / (
                    true_positive + false_positive
                ).clamp_min(1.0)
                stuck_recall = true_positive / (
                    true_positive + false_negative
                ).clamp_min(1.0)
                stuck_f1 = (
                    2.0 * stuck_precision * stuck_recall
                    / (stuck_precision + stuck_recall).clamp_min(1.0e-6)
                )
                stuck_positive_share = labels_bool.float().mean()
                order = torch.argsort(probabilities, descending=True)
                sorted_labels = labels_bool[order].float()
                cumulative_positive = sorted_labels.cumsum(dim=0)
                ranks = torch.arange(
                    1,
                    sorted_labels.numel() + 1,
                    device=sorted_labels.device,
                    dtype=sorted_labels.dtype,
                )
                precision_curve = cumulative_positive / ranks
                positive_count = sorted_labels.sum()
                stuck_pr_auc = (
                    (precision_curve * sorted_labels).sum()
                    / positive_count.clamp_min(1.0)
                )
                recall_curve = cumulative_positive / positive_count.clamp_min(1.0)
                f1_curve = (
                    2.0 * precision_curve * recall_curve
                    / (precision_curve + recall_curve).clamp_min(1.0e-6)
                )
                if sorted_labels.numel() and float(positive_count) > 0.0:
                    best_index = int(f1_curve.argmax())
                    stuck_threshold = probabilities[order[best_index]]
            stuck_target = float(schedule.get("stuck_gradient_target_ratio", 0.0))
            if stuck_target > 0.0 and stuck_raw.requires_grad:
                specs.append(("stuck", stuck_raw, stuck_target))

        mirror_raw = zero
        mirror_sequence_count = 0
        mirror_batch = batch.get("mirror_batch")
        if isinstance(mirror_batch, dict) and "depth" in mirror_batch:
            depth = mirror_batch["depth"]
            mirror_sequence_count = int(depth.shape[1])
            if mirror_sequence_count:
                flat_depth = depth.reshape(
                    -1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
                )
                mirrored_depth = depth.flip(dims=(-2,)).reshape(
                    -1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
                )
                amp_enabled = self.device.type == "cuda"
                if not amp_enabled:
                    flat_depth = flat_depth.float()
                    mirrored_depth = mirrored_depth.float()
                with torch.no_grad(), torch.autocast(
                    device_type=self.device.type, enabled=amp_enabled
                ):
                    original_feat = self.navigation_encoder(flat_depth).float().reshape(
                        depth.shape[0], depth.shape[1], -1
                    )
                    mirrored_feat = self.navigation_encoder(
                        mirrored_depth
                    ).float().reshape(depth.shape[0], depth.shape[1], -1)
                original_input = assemble_actor_input(
                    original_feat.detach(),
                    mirror_batch["nav_nonvisual"],
                    mirror_batch["response_profile"],
                    mirror_batch["confidence"],
                )
                mirrored_input = assemble_actor_input(
                    mirrored_feat.detach(),
                    self._mirror_nav_nonvisual(mirror_batch["nav_nonvisual"]),
                    self._mirror_response_profile(
                        mirror_batch["response_profile"]
                    ),
                    mirror_batch["confidence"],
                )
                zero_hidden = (
                    torch.zeros(
                        self.actor.num_layers,
                        mirror_sequence_count,
                        self.actor.hidden_dim,
                        device=normalized_mean.device,
                        dtype=normalized_mean.dtype,
                    ),
                    torch.zeros(
                        self.actor.num_layers,
                        mirror_sequence_count,
                        self.actor.hidden_dim,
                        device=normalized_mean.device,
                        dtype=normalized_mean.dtype,
                    ),
                )
                with torch.no_grad():
                    _, _, original_mean, _, _, _ = self.actor.evaluate_actions(
                        original_input,
                        mirror_batch["pre_tanh_action"],
                        zero_hidden,
                        mirror_batch["reset_mask"],
                        return_features=True,
                    )
                _, _, mirrored_mean, _, _, _ = self.actor.evaluate_actions(
                    mirrored_input,
                    mirror_batch["pre_tanh_action"],
                    zero_hidden,
                    mirror_batch["reset_mask"],
                    return_features=True,
                )
                expected = torch.tanh(original_mean).detach().clone()
                expected[..., 1:].neg_()
                mirror_raw = F.smooth_l1_loss(torch.tanh(mirrored_mean), expected)
                specs.append((
                    "mirror",
                    mirror_raw,
                    min(
                        float(schedule.get("mirror_gradient_target_ratio", 0.005)),
                        float(schedule.get("mirror_gradient_hard_cap", 0.01)),
                    ),
                ))

        parameters = [
            parameter
            for module in (self.navigation_encoder, self.actor, self.stuck_head)
            if module is not None
            for parameter in module.parameters() if parameter.requires_grad
        ]
        stored_ratios = {
            name: float(self._auxiliary_calibration.get(f"{name}_ratio", 0.0))
            for name in ("camera", "teacher", "mirror", "stuck")
        }
        # A diagnostic/teacher term can legitimately be a detached constant:
        # for example when its privileged inputs are invalid, or when the
        # corresponding module is frozen during warm-up.  Do not pass such a
        # term to autograd.grad; doing so crashes the whole rollout update with
        # "does not require grad".  Keep it visible in diagnostics, but let the
        # PPO loss proceed without calibrating a coefficient for it.
        uncalibrated = [
            spec
            for spec in specs
            if spec[0] not in self._auxiliary_coefficients
            and bool(spec[1].requires_grad)
        ]
        head_only_calibrated = []
        head_only_names: set[str] = set()
        if uncalibrated and not ppo_actor_loss.requires_grad:
            # During creditwarm the Actor is frozen but the new StuckHead is
            # intentionally calibrated.  With no PPO reference gradient a
            # ratio is undefined, so train only the isolated head with BCE.
            # Do not persist this coefficient: after Actor unfreezes, stuck
            # supervision must be recalibrated against the real PPO gradient.
            for name, raw, _target in uncalibrated:
                if name == "stuck" and raw.requires_grad:
                    head_only_calibrated.append(
                        {
                            "name": name,
                            "loss": raw,
                            "multiplier": 1.0,
                            "component_ratio": 0.0,
                        }
                    )
                    head_only_names.add(name)
            uncalibrated = []
        if uncalibrated:
            remaining = max(
                0.0,
                float(schedule.get("auxiliary_gradient_hard_cap", 0.05))
                - sum(stored_ratios.values()),
            )
            newly_calibrated, _new_ratio = _calibrate_auxiliary_gradients(
                ppo_actor_loss,
                uncalibrated,
                parameters,
                remaining,
            )
            for item in newly_calibrated:
                name = str(item["name"])
                self._auxiliary_coefficients[name] = float(item["multiplier"])
                stored_ratios[name] = float(item["component_ratio"])
        self._camera_aux_calibration_pending = False
        calibrated = head_only_calibrated + [
            {
                "name": name,
                "loss": raw,
                "multiplier": float(self._auxiliary_coefficients.get(name, 0.0)),
                "component_ratio": stored_ratios.get(name, 0.0),
            }
            for name, raw, _target in specs
            if name not in head_only_names
        ]
        combined_ratio = min(
            float(schedule.get("auxiliary_gradient_hard_cap", 0.05)),
            sum(stored_ratios.values()),
        )
        multipliers = {item["name"]: float(item["multiplier"]) for item in calibrated}
        ratios = {item["name"]: float(item["component_ratio"]) for item in calibrated}
        loss = sum(
            (item["loss"] * item["multiplier"] for item in calibrated), zero
        )
        self._auxiliary_calibration = {
            "combined_ratio": float(combined_ratio),
            "teacher_ratio": stored_ratios.get("teacher", 0.0),
            "camera_ratio": stored_ratios.get("camera", 0.0),
            "mirror_ratio": stored_ratios.get("mirror", 0.0),
            "stuck_ratio": stored_ratios.get("stuck", 0.0),
            "teacher_valid_steps": float(teacher["teacher_valid_steps"].detach()),
            "stuck_valid_steps": float(stuck_mask.sum()),
        }
        self._camera_aux_coefficient = multipliers.get("camera", 0.0)
        self._camera_aux_gradient_ratio = ratios.get("camera", 0.0)
        return loss, {
            "camera_memory_loss": camera_raw.detach(),
            "camera_clean_live_action_mae": (
                (normalized_mean[camera_selected] - batch["clean_action_mean"][camera_selected])
                .abs().mean().detach() if bool(camera_selected.any()) else zero.detach()
            ),
            "camera_aux_coefficient": zero.new_tensor(self._camera_aux_coefficient),
            "camera_aux_gradient_ratio": zero.new_tensor(self._camera_aux_gradient_ratio),
            "camera_delay_only_share": mask3[..., 0].float().mean().detach(),
            "camera_fault_only_share": mask3[..., 1].float().mean().detach(),
            "camera_fault_delay_overlap_share": mask3[..., 2].float().mean().detach(),
            "teacher_guidance_loss": teacher_raw.detach(),
            "teacher_guidance_valid_steps": teacher["teacher_valid_steps"].detach(),
            "teacher_guidance_gradient_ratio": zero.new_tensor(ratios.get("teacher", 0.0)),
            "stuck_aux_loss": stuck_raw.detach(),
            "stuck_aux_valid_steps": zero.new_tensor(float(stuck_mask.sum())),
            "stuck_aux_gradient_ratio": zero.new_tensor(ratios.get("stuck", 0.0)),
            "actor_stuck_pr_auc": stuck_pr_auc.detach(),
            "actor_stuck_precision": stuck_precision.detach(),
            "actor_stuck_recall": stuck_recall.detach(),
            "actor_stuck_f1": stuck_f1.detach(),
            "actor_stuck_threshold": stuck_threshold.detach(),
            "actor_stuck_positive_share": stuck_positive_share.detach(),
            "mirror_aux_loss": mirror_raw.detach(),
            "mirror_aux_sequence_share": zero.new_tensor(
                float(mirror_sequence_count) / max(normalized_mean.shape[1], 1)
            ),
            "mirror_aux_eligible_sequence_count": zero.new_tensor(
                float(self._mirror_aux_eligible_sequence_count)
            ),
            "mirror_aux_scheduled_sequence_share": zero.new_tensor(
                float(self._mirror_aux_scheduled_sequence_share)
            ),
            "mirror_aux_gradient_ratio": zero.new_tensor(ratios.get("mirror", 0.0)),
            "auxiliary_gradient_ratio": zero.new_tensor(float(combined_ratio)),
            "teacher_direction_loss": teacher["direction"].detach(),
            "teacher_speed_loss": teacher["speed"].detach(),
            "teacher_yaw_loss": teacher["yaw"].detach(),
            "teacher_edge_loss": teacher.get("edge", zero).detach(),
            "teacher_edge_active_share": teacher.get(
                "teacher_edge_active_share", zero
            ).detach(),
            "teacher_recovery_loss": teacher.get("recovery", zero).detach(),
            "teacher_recovery_active_share": teacher.get(
                "teacher_recovery_active_share", zero
            ).detach(),
        }

    @staticmethod
    def _mirror_nav_nonvisual(values: torch.Tensor) -> torch.Tensor:
        mirrored = values.clone()
        # goal-y and target/executed/measured vy/wz channels.
        for index in (1, 5, 6, 8, 9, 11, 12):
            mirrored[..., index].neg_()
        # Angular velocity is an axial vector under a left/right reflection;
        # projected gravity is an ordinary vector.
        for index in (15, 17, 19):
            mirrored[..., index].neg_()
        return mirrored

    @staticmethod
    def _mirror_response_profile(values: torch.Tensor) -> torch.Tensor:
        mirrored = values.clone()
        for index in (1, 2, 4, 5, 7, 8):
            mirrored[..., index].neg_()
        mirrored[..., 10].neg_()
        mirrored[..., 11].neg_()
        return mirrored

    def _actor_micro_loss(self, batch):
        if "depth" in batch:
            depth = batch["depth"].reshape(
                -1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
            )
            amp_enabled = self.device.type == "cuda"
            if not amp_enabled:
                depth = depth.float()
            with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                feat = self.navigation_encoder(depth).float()
            feat = feat.reshape(batch["nav_nonvisual"].shape[0], -1, 32)
        else:
            feat = batch["nav_feat"]
        inputs = assemble_actor_input(
            feat, batch["nav_nonvisual"], batch["response_profile"], batch["confidence"]
        )
        log_prob, entropy, mean, _, _, _ = self.actor.evaluate_actions(
            inputs, batch["pre_tanh_action"], batch["actor_hidden"],
            batch["reset_mask"], return_features=True,
        )
        ratio = torch.exp(log_prob - batch["old_log_prob"])
        surrogate = -torch.minimum(
            ratio * batch["advantages"],
            torch.clamp(ratio, 0.8, 1.2) * batch["advantages"],
        ).mean()
        if getattr(self, "track_safety_enabled", True):
            safety_logits = self.safety_head(feat)
            safety_elements = F.binary_cross_entropy_with_logits(
                safety_logits, batch["safety_target"], reduction="none"
            )
            safety_mask = batch["safety_valid"].expand_as(safety_elements)
            hard_positive = (
                (batch["safety_target"] >= 0.65).any(dim=-1)
                | (batch["stuck_label"].squeeze(-1) > 0.5)
            )
            safety_weight = 1.0 + hard_positive.float()
            weighted_safety_mask = safety_mask * safety_weight.unsqueeze(-1)
            safety_loss = (
                safety_elements.mul(weighted_safety_mask).sum()
                / weighted_safety_mask.sum().clamp_min(1.0)
            )
            student_risk = torch.sigmoid(safety_logits).mean(dim=(0, 1))
        else:
            safety_loss = surrogate.new_zeros(())
            student_risk = torch.zeros(3, device=surrogate.device)
            hard_positive = torch.zeros(
                batch["safety_target"].shape[:-1],
                dtype=torch.bool,
                device=surrogate.device,
            )
        detached_inputs = assemble_actor_input(
            feat.detach(), batch["nav_nonvisual"], batch["response_profile"], batch["confidence"]
        )
        _, _, auxiliary_mean, _, _, auxiliary_features = self.actor.evaluate_actions(
            detached_inputs, batch["pre_tanh_action"], batch["actor_hidden"],
            batch["reset_mask"], return_features=True,
        )
        auxiliary_loss, auxiliary_metrics = self._actor_auxiliary_loss(
            normalized_mean=torch.tanh(mean),
            actor_normalized_mean=torch.tanh(auxiliary_mean),
            actor_features=auxiliary_features,
            nav_feat=feat,
            batch=batch, ppo_actor_loss=surrogate,
        )
        auxiliary_metrics["safety_hard_positive_share"] = (
            hard_positive.float().mean().detach()
        )
        total_loss = (
            surrogate - self.entropy_coefficient * entropy.mean()
            + p2_contract.SAFETY_BCE_WEIGHT * safety_loss + auxiliary_loss
        )
        with torch.no_grad():
            log_ratio = log_prob - batch["old_log_prob"]
            approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > 0.2).float().mean()
        return total_loss, {
            "surrogate_loss": surrogate.detach(), "entropy": entropy.mean().detach(),
            "approx_kl": approx_kl.detach(), "clip_fraction": clip_fraction.detach(),
            "safety_bce": safety_loss.detach(),
            "scanner_valid_share": batch["safety_valid"].float().mean().detach(),
            "safety_head_risk_left": student_risk[0].detach(),
            "safety_head_risk_center": student_risk[1].detach(),
            "safety_head_risk_right": student_risk[2].detach(),
            **auxiliary_metrics,
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
            "teacher_guidance_loss",
            "teacher_guidance_valid_steps",
            "teacher_guidance_gradient_ratio",
            "teacher_direction_loss",
            "teacher_speed_loss",
            "teacher_yaw_loss",
            "teacher_edge_loss",
            "teacher_edge_active_share",
            "teacher_recovery_loss",
            "teacher_recovery_active_share",
            "stuck_aux_loss",
            "stuck_aux_valid_steps",
            "stuck_aux_gradient_ratio",
            "actor_stuck_pr_auc",
            "actor_stuck_precision",
            "actor_stuck_recall",
            "actor_stuck_f1",
            "actor_stuck_threshold",
            "actor_stuck_positive_share",
            "mirror_aux_loss",
            "mirror_aux_sequence_share",
            "mirror_aux_eligible_sequence_count",
            "mirror_aux_scheduled_sequence_share",
            "mirror_aux_gradient_ratio",
            "auxiliary_gradient_ratio",
            "safety_hard_positive_share",
        )

    def _actor_sequence_batch(self, refs, *, advantage_mean, advantage_std):
        batch = super()._actor_sequence_batch(
            refs, advantage_mean=advantage_mean, advantage_std=advantage_std
        )
        for name in (
            "teacher_safe3", "teacher_safe5", "teacher_goal_xy", "teacher_predictive_risk",
            "teacher_mask", "teacher_goal_mask", "teacher_weight",
            "stuck_label", "stuck_mask", "mirror_eligible",
        ):
            batch[name] = self._stack_refs(name, refs)
        if self._mirror_batch_cursor < len(self._mirror_batch_schedule):
            mirror_ref = self._mirror_batch_schedule[self._mirror_batch_cursor]
            self._mirror_batch_cursor += 1
            if mirror_ref is not None:
                batch["mirror_batch"] = {
                    name: self._stack_refs(name, [mirror_ref])
                    for name in (
                        "depth",
                        "nav_nonvisual",
                        "response_profile",
                        "confidence",
                        "pre_tanh_action",
                        "reset_mask",
                    )
                }
        return batch

    def _prepare_mirror_batch_schedule(self) -> None:
        """Schedule episode-aligned mirror sequences across actor microbatches."""
        self._mirror_batch_schedule = []
        self._mirror_batch_cursor = 0
        self._mirror_aux_eligible_sequence_count = 0
        self._mirror_aux_scheduled_sequence_share = 0.0
        if self.rollout.store_depth:
            pool = self.rollout.episode_aligned_refs(
                eligibility=self.rollout.mirror_eligible[: self.rollout.step]
            )
            self._mirror_aux_eligible_sequence_count = len(pool)
            fixed_sequence_count = (
                self.num_envs * self.rollout.num_ticks
                // self.rollout.sequence_length
            )
            minibatch_sequences = max(
                1,
                (fixed_sequence_count + self.num_mini_batches - 1)
                // self.num_mini_batches,
            )
            micro_calls_per_epoch = sum(
                (
                    min(minibatch_sequences, fixed_sequence_count - start)
                    + self.micro_sequences - 1
                )
                // self.micro_sequences
                for start in range(0, fixed_sequence_count, minibatch_sequences)
            )
            total_micro_calls = micro_calls_per_epoch * self.num_learning_epochs
            target_sequences = min(
                total_micro_calls,
                int(round(
                    fixed_sequence_count
                    * self.num_learning_epochs
                    * float(p4_contract.training_schedule(
                        self.session_effective_seconds,
                        branch=self._effective_maze_branch(
                            self.session_effective_seconds
                        ),
                    ).get("mirror_sequence_share", 0.10))
                )),
            )
            if pool and target_sequences > 0:
                selected_slots = torch.randperm(
                    total_micro_calls, generator=self.mirror_generator
                )[:target_sequences].tolist()
                selected_refs = torch.randint(
                    len(pool),
                    (target_sequences,),
                    generator=self.mirror_generator,
                ).tolist()
                schedule: list[object | None] = [None] * total_micro_calls
                for slot, ref_index in zip(selected_slots, selected_refs):
                    schedule[int(slot)] = pool[int(ref_index)]
                self._mirror_batch_schedule = schedule
                self._mirror_aux_scheduled_sequence_share = (
                    float(target_sequences)
                    / max(
                        fixed_sequence_count * self.num_learning_epochs,
                        1,
                    )
                )

    def _run_ppo_epochs(self) -> dict[str, float]:
        valid_teacher_steps = int(
            self.rollout.teacher_mask[: self.rollout.step].sum().item()
        )
        schedule = p4_contract.training_schedule(
            self.session_effective_seconds,
            branch=self._effective_maze_branch(self.session_effective_seconds),
        )
        self._teacher_update_enabled = (
            valid_teacher_steps >= p4_contract.TEACHER_MIN_VALID_STEPS
            and float(schedule.get("teacher_gradient_target_ratio", 0.0)) > 0.0
        )
        self._camera_aux_calibration_pending = True
        self._auxiliary_coefficients = {}
        self._auxiliary_calibration.update(
            {
                "combined_ratio": 0.0,
                "teacher_ratio": 0.0,
                "camera_ratio": 0.0,
                "mirror_ratio": 0.0,
                "stuck_ratio": 0.0,
                "teacher_valid_steps": float(valid_teacher_steps),
            }
        )
        self._prepare_mirror_batch_schedule()
        return super()._run_ppo_epochs()

    def _actor_update_enabled(self) -> bool:
        schedule = p4_contract.training_schedule(
            self.session_effective_seconds,
            branch=self._effective_maze_branch(self.session_effective_seconds),
        )
        return float(schedule.get("actor_multiplier", 0.0)) > 0.0

    def _adapter_update(self) -> dict[str, float]:
        if self.training_profile in self.MAZE_PROFILES:
            return {
                "adapter_loss": 0.0,
                "adapter_updates": 0.0,
                "adapter_frozen": 1.0,
            }
        return super()._adapter_update()

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
        maze_profile = self.training_profile in self.MAZE_PROFILES
        closed_loop_profile = self.training_profile == "maze_closed_loop_v3"
        if maze_profile:
            components["time"] = (
                p2_contract.TIME_COST_PER_TICK
                * duration_frames
                / float(p4_contract.P4_NAV_PERIOD_FRAMES)
            )
            best_before = torch.where(
                torch.isfinite(self.best_goal_distance),
                self.best_goal_distance,
                context["start_goal_distance"].reshape(-1),
            )
            maze_credit_before = self._maze_credit_earned.clone()
            maze_credit, maze_credit_after, maze_credit_delta = (
                p4_contract.maze_new_best_credit(
                    best_before,
                    context["end_goal_distance"].reshape(-1),
                    self._maze_credit_earned,
                    weight_per_m=(
                        p4_contract.MAZE_NEW_BEST_WEIGHT_PER_M
                        if closed_loop_profile
                        else p4_contract.LEGACY_MAZE_NEW_BEST_WEIGHT_PER_M
                    ),
                    episode_cap=(
                        p4_contract.MAZE_NEW_BEST_EPISODE_CAP
                        if closed_loop_profile
                        else p4_contract.LEGACY_MAZE_NEW_BEST_EPISODE_CAP
                    ),
                )
            )
            components["frontier_shaping"] = maze_credit
            self._maze_credit_earned.copy_(maze_credit_after)
        else:
            maze_credit_before = self._maze_credit_earned
            maze_credit = torch.zeros(self.num_envs, device=self.device)
            maze_credit_after = self._maze_credit_earned
            maze_credit_delta = torch.zeros_like(maze_credit)
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
        yaw_exec_weight = (
            p4_contract.YAW_EXEC_WEIGHT
            if closed_loop_profile
            else p4_contract.LEGACY_YAW_EXEC_WEIGHT
        )
        yaw_true_weight = (
            p4_contract.YAW_TRUE_WEIGHT
            if closed_loop_profile
            else p4_contract.LEGACY_YAW_TRUE_WEIGHT
        )
        yaw_floor = (
            p4_contract.YAW_TOTAL_FLOOR
            if closed_loop_profile
            else p4_contract.LEGACY_YAW_TOTAL_FLOOR
        )
        yaw_raw = torch.where(
            enough,
            yaw_exec_weight * exec_cancel
            + yaw_true_weight * true_cancel,
            torch.zeros_like(exec_cancel),
        ).clamp_min(yaw_floor)
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
        goal_safe_raw, goal_safe_diag = p4_contract.goal_safe_direction_penalty(
            safe3,
            self._last_policy_command,
            self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE],
            safety_valid,
            self._last_goal_freshness,
            terminal,
        )
        yaw_exit_raw, yaw_exit_diag = p4_contract.yaw_exit_response_penalty(
            safe3,
            self._last_policy_command,
            self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE],
            safety_valid,
            self._last_goal_freshness,
            terminal,
        )
        goal_safe_raw *= reward_multiplier
        yaw_exit_raw *= reward_multiplier
        missed_shadow_raw = missed_raw.detach().clone()
        goal_safe_shadow_raw = goal_safe_raw.detach().clone()
        yaw_exit_shadow_raw = yaw_exit_raw.detach().clone()
        if closed_loop_profile:
            # The five-direction Actor teacher is the only privileged
            # directional learning signal in v3. Keep the legacy reward terms
            # as shadow diagnostics so they cannot duplicate or contradict the
            # teacher gradient, and keep the deployable depth-risk term active.
            missed_raw = torch.zeros_like(missed_raw)
            goal_safe_raw = torch.zeros_like(goal_safe_raw)
            yaw_exit_raw = torch.zeros_like(yaw_exit_raw)
        (predictive, missed, yaw, goal_safe, yaw_exit), scale = (
            p4_contract.proportional_negative_cap(
                predictive_raw,
                missed_raw,
                yaw_raw,
                goal_safe_raw=goal_safe_raw,
                yaw_exit_raw=yaw_exit_raw,
                floor=(
                    p4_contract.SAFETY_GROUP_FLOOR
                    if closed_loop_profile
                    else p4_contract.LEGACY_SAFETY_GROUP_FLOOR
                ),
            )
        )
        predictive *= continuous_time_scale
        missed *= continuous_time_scale
        yaw *= continuous_time_scale
        goal_safe *= continuous_time_scale
        yaw_exit *= continuous_time_scale
        components["predictive_collision_risk"] = predictive
        components["missed_safe_direction"] = missed
        # In r3 cancellation is diagnostic-only: rapid left/right changes can
        # be correct at adjacent Maze corners. Legacy profiles retain their
        # original cancellation reward and contract.
        components["yaw_cancellation"] = (
            torch.zeros_like(yaw) if closed_loop_profile else yaw
        )
        components["goal_safe_preference"] = goal_safe
        components["yaw_exit_response"] = yaw_exit
        components["success"] = (context["reason"].reshape(-1) == 1).float() * float(
            p4_contract.SUCCESS_IMPULSE
        )
        if closed_loop_profile:
            components["failure"] = (
                context["reason"].reshape(-1) == 2
            ).float() * float(p4_contract.FAILURE_IMPULSE)
        if "timeout" in components:
            components["timeout"] = (
                context["reason"].reshape(-1) == 3
            ).float() * float(
                p4_contract.TIMEOUT_IMPULSE
                if closed_loop_profile
                else p4_contract.LEGACY_TIMEOUT_IMPULSE
            )
        if closed_loop_profile:
            force = torch.nan_to_num(
                source_aux[:, p2_contract.BODY_COLLISION_FORCE_INDEX],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            collision_active = components["body_collision"] < 0.0
            collision_onset = components["body_collision"] < (
                p2_contract.BODY_COLLISION_PERSISTENT - 1.0e-6
            )
            severity = torch.clamp(
                (force - p2_contract.BODY_COLLISION_SOFT_FORCE_N)
                / (
                    p2_contract.BODY_COLLISION_HARD_FORCE_N
                    - p2_contract.BODY_COLLISION_SOFT_FORCE_N
                ),
                0.0,
                1.0,
            )
            components["body_collision"] = torch.where(
                collision_onset,
                p4_contract.P4_BODY_COLLISION_ONSET_BASE
                + p4_contract.P4_BODY_COLLISION_ONSET_SEVERITY * severity,
                torch.where(
                    collision_active,
                    torch.full_like(
                        severity, p4_contract.P4_BODY_COLLISION_PERSISTENT
                    ),
                    torch.zeros_like(severity),
                ),
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
            "yaw_cancellation",
            "yaw_exit_response",
            "goal_safe_preference",
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
                floor=(
                    p4_contract.STUCK_SUSTAINED_FLOOR
                    if closed_loop_profile
                    else p4_contract.LEGACY_STUCK_SUSTAINED_FLOOR
                ),
            )
        )
        stuck_time_scale = torch.clamp(
            duration_frames / float(p4_contract.P4_NAV_PERIOD_FRAMES), 0.0, 1.0
        )
        components["stuck_sustained"] = stuck_sustained * stuck_time_scale
        path_length_m = context.get("path_length_m")
        if path_length_m is None:
            path_length_m = torch.zeros(self.num_envs, device=self.device)
        scene = p4_contract.safety_scene_diagnostics(safe3, safety_valid)
        recovery_active = (
            (self._p4_worker_extra[:, p4_contract.STUCK_CANDIDATE_INDEX] > 0.5)
            & (self._p4_worker_extra[:, p4_contract.STUCK_MAPPING_VALID_INDEX] > 0.5)
        )
        route_excess, route_diag = p4_contract.route_excess_penalty(
            torch.as_tensor(path_length_m, device=self.device).reshape(-1),
            context["start_goal_distance"].reshape(-1),
            context["end_goal_distance"].reshape(-1),
            terminal,
            recovery_active=recovery_active,
            dead_end=scene["teacher_scene_dead_end"] > 0.5,
            goal_freshness=self._last_goal_freshness,
            contact_latch=self.previous_body_collision,
        )
        grace = self.seconds_since_push < 0.30
        components["route_excess"] = torch.where(
            grace, torch.zeros_like(route_excess), route_excess
        )
        current_segment = (
            torch.full(
                (self.num_envs,),
                float(len(p4_contract.FULL_TRACK_SEGMENT_LABELS) - 1),
                device=self.device,
            )
            if maze_profile
            else torch.nan_to_num(
                source_aux[:, p2_contract.CURRENT_SEGMENT_INDEX], nan=0.0
            ).round().clamp(0, len(p4_contract.FULL_TRACK_SEGMENT_LABELS) - 1)
        )
        initialize_segment = ~self._segment_state_initialized
        if bool(initialize_segment.any()):
            self._spawn_segment[initialize_segment] = current_segment[initialize_segment]
            self._max_segment_reached[initialize_segment] = current_segment[initialize_segment]
            self._segment_state_initialized[initialize_segment] = True
        segment_frontier, segment_phi_before, segment_phi_after, segment_max_after = (
            p4_contract.segment_frontier_potential(
                self._spawn_segment,
                self._max_segment_reached,
                current_segment,
                duration_frames,
                terminal,
            )
        )
        components["segment_frontier"] = segment_frontier
        self._max_segment_reached.copy_(segment_max_after)
        root_x_m = source_aux[:, 15]
        boundary_distance = p4_contract.track_boundary_distance_m(
            root_x_m, current_segment
        )
        positive_progress = torch.clamp(
            context["start_goal_distance"].reshape(-1)
            - context["end_goal_distance"].reshape(-1),
            min=0.0,
        )
        open_path_excess = torch.clamp(
            torch.as_tensor(path_length_m, device=self.device).reshape(-1)
            - positive_progress,
            min=0.0,
            max=0.20,
        )
        open_straight, open_straight_diag = p4_contract.open_straight_penalty(
            self._last_policy_command,
            source_aux[:, 12:15],
            self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE],
            safe3,
            safety_valid,
            current_segment,
            boundary_distance,
            terminal,
            junction=scene["teacher_scene_junction"] > 0.5,
            dead_end=scene["teacher_scene_dead_end"] > 0.5,
            contact_or_recovery=(self.previous_body_collision | recovery_active),
            goal_freshness=self._last_goal_freshness,
            yaw_cancellation_value=true_cancel,
            path_excess_m=open_path_excess,
        )
        components["open_straight"] = open_straight * continuous_time_scale
        if maze_profile:
            components["route_excess"] = torch.zeros_like(components["route_excess"])
            components["segment_frontier"] = torch.zeros_like(
                components["segment_frontier"]
            )
            components["open_straight"] = torch.zeros_like(components["open_straight"])
        soft_cruise, cruise_diag = p4_contract.soft_cruise_penalty(
            self._last_policy_command,
            self.pending_tick.get("safe3", torch.zeros(self.num_envs, 3, device=self.device)),
            self.pending_tick.get(
                "safety_valid",
                torch.zeros(self.num_envs, 1, device=self.device),
            ).reshape(-1) > 0.5,
            self._last_goal_freshness,
            terminal,
            capture_active=self._near_goal_capture_diagnostics.get(
                "near_goal_capture_active",
                torch.zeros(self.num_envs, device=self.device),
            ),
        )
        soft_cruise *= (
            float(schedule.get("cruise_multiplier", 1.0))
            * continuous_time_scale
        )
        components["soft_cruise"] = soft_cruise
        if closed_loop_profile:
            reward_allowlist = {
                "frame_safety",
                "frontier_shaping",
                "success",
                "failure",
                "timeout",
                "time",
                "crawl",
                "command_rate",
                "tracking",
                "body_collision",
                "predictive_collision_risk",
                "stuck_reset",
                "stuck_sustained",
            }
            for name in tuple(components):
                if name not in reward_allowlist:
                    components[name] = torch.zeros_like(components[name])
        capture_active = self._near_goal_capture_diagnostics.get(
            "near_goal_capture_active", torch.zeros(self.num_envs, device=self.device)
        ) > 0.5
        if "crawl" in components:
            components["crawl"] = torch.where(
                capture_active,
                torch.zeros_like(components["crawl"]),
                components["crawl"],
            )
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
            "reward_missed_safe_raw": missed_shadow_raw,
            "reward_yaw_raw": yaw_raw.detach(),
            "reward_goal_safe_raw": goal_safe_shadow_raw,
            "reward_yaw_exit_raw": yaw_exit_shadow_raw,
            "reward_continuous_time_scale": continuous_time_scale.detach(),
            "reward_safety_group_scale": scale.detach(),
            "reward_frontier_stagnation_shadow": stagnation_shadow,
            "maze_new_best_credit": maze_credit.detach(),
            "maze_new_best_delta_m": maze_credit_delta.detach(),
            "maze_new_best_episode_earned": maze_credit_after.detach(),
            "frontier_potential_before": maze_credit_before.detach(),
            "frontier_potential_after": maze_credit_after.detach(),
            "terminal_potential_clawback": torch.zeros_like(maze_credit),
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
            **{name: value.detach() for name, value in yaw_exit_diag.items()},
            **{name: value.detach() for name, value in route_diag.items()},
            "segment_frontier_phi_before": segment_phi_before.detach(),
            "segment_frontier_phi_after": segment_phi_after.detach(),
            "segment_frontier_spawn_segment": self._spawn_segment.detach().clone(),
            "segment_frontier_max_segment": segment_max_after.detach(),
            "current_segment_index": current_segment.detach(),
            **{name: value.detach() for name, value in open_straight_diag.items()},
            **{name: value.detach() for name, value in missed_diagnostics.items()},
            **{name: value.detach() for name, value in cruise_diag.items()},
            **self._translation_limiter_diagnostics,
            **self._near_goal_capture_diagnostics,
        }
        if bool(terminal.any()):
            self._segment_state_initialized[terminal] = False
            self._spawn_segment[terminal] = 0.0
            self._max_segment_reached[terminal] = 0.0
            self._maze_credit_earned[terminal] = 0.0
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
        terminal_for_aux = (reason != 0) | hard | timeout
        if self.pending_tick is not None and bool(terminal_for_aux.any()):
            # Terminal/reset transitions never provide recovery supervision.
            for name in ("teacher_mask", "teacher_goal_mask", "stuck_mask", "mirror_eligible"):
                value = self.pending_tick.get(name)
                if torch.is_tensor(value):
                    value = value.clone()
                    value[terminal_for_aux] = 0.0
                    self.pending_tick[name] = value
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
        safe5 = pending.get(
            "teacher_safe5", torch.zeros(self.num_envs, 5, device=self.device)
        )
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
            clean_bearing = torch.atan2(
                self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE.stop - 1],
                self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE.start],
            ).abs()
            fault_bearing = torch.atan2(
                self.goal_belief.estimate[:, 1], self.goal_belief.estimate[:, 0]
            ).abs()
            bucket_count = bucket.float().sum().clamp_min(1.0)
            goal_bucket_metrics[f"goal_bearing_clean_{label}_abs_rad"] = (
                (clean_bearing * bucket.float()).sum() / bucket_count
            ).expand(self.num_envs)
            goal_bucket_metrics[f"goal_bearing_fault_{label}_abs_rad"] = (
                (fault_bearing * bucket.float()).sum() / bucket_count
            ).expand(self.num_envs)

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
        current_segment = self._p4_reward_diagnostics.get(
            "current_segment_index", raw_goal_distance.new_zeros(self.num_envs)
        ).round().long().clamp(0, len(p4_contract.FULL_TRACK_SEGMENT_LABELS) - 1)
        spawn_segment = self._p4_worker_extra[
            :, p4_contract.SPAWN_SEGMENT_INDEX
        ].round().long()
        spawn_quartile = self._p4_worker_extra[
            :, p4_contract.SPAWN_QUARTILE_INDEX
        ].round().long()
        clean_goal = self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE]
        clean_bearing_signed = torch.atan2(clean_goal[:, 1], clean_goal[:, 0])
        side_goal_candidate = (
            teacher_valid
            & (clean_bearing_signed.abs() >= math.radians(15.0))
            & (clean_bearing_signed.abs() <= math.radians(90.0))
        )
        side_goal_selected = side_goal_candidate & (
            self._last_policy_command[:, 2] * clean_bearing_signed > 0.0
        )
        desired_yaw_sign = torch.sign(clean_bearing_signed)
        correct_yaw_response = (
            self._last_policy_command[:, 2] * desired_yaw_sign >= 0.10
        )
        vy_substitutes_yaw = (
            teacher_valid
            & (clean_bearing_signed.abs() >= math.radians(35.0))
            & (self._last_policy_command[:, 1].abs() >= 0.12)
            & ~correct_yaw_response
        )
        limiter_intervened = self._translation_limiter_diagnostics.get(
            "translation_safety_alpha",
            raw_goal_distance.new_ones(self.num_envs),
        ) < 0.999
        frontier_before = self._p4_reward_diagnostics.get(
            "segment_frontier_phi_before", raw_goal_distance.new_zeros(self.num_envs)
        )
        frontier_after = self._p4_reward_diagnostics.get(
            "segment_frontier_phi_after", raw_goal_distance.new_zeros(self.num_envs)
        )
        full_track_metrics: dict[str, torch.Tensor] = {}
        for index, label in enumerate(p4_contract.FULL_TRACK_SEGMENT_LABELS):
            current_mask = current_segment == index
            spawn_mask = spawn_segment == index
            full_track_metrics[f"current_segment_{label}_share"] = current_mask.float()
            full_track_metrics[f"spawn_segment_{label}_share"] = spawn_mask.float()
            denominator = spawn_mask.float().sum().clamp_min(1.0)
            full_track_metrics[f"frontier_potential_before_{label}"] = (
                (frontier_before * spawn_mask.float()).sum() / denominator
            ).expand(self.num_envs)
            full_track_metrics[f"frontier_potential_after_{label}"] = (
                (frontier_after * spawn_mask.float()).sum() / denominator
            ).expand(self.num_envs)
        for index in range(4):
            full_track_metrics[f"spawn_column_q{index + 1}_share"] = (
                spawn_quartile == index
            ).float()
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
            **full_track_metrics,
            "safety_head_risk_left": self._last_safety_head_risk3[:, 0],
            "safety_head_risk_center": self._last_safety_head_risk3[:, 1],
            "safety_head_risk_right": self._last_safety_head_risk3[:, 2],
            "teacher_safe5_far_left": safe5[:, 0],
            "teacher_safe5_left": safe5[:, 1],
            "teacher_safe5_center": safe5[:, 2],
            "teacher_safe5_right": safe5[:, 3],
            "teacher_safe5_far_right": safe5[:, 4],
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
            "full_phase_warm": raw_goal_distance.new_full(
                (self.num_envs,), float(schedule["phase"] == "fullwarm")
            ),
            "full_phase_adapt": raw_goal_distance.new_full(
                (self.num_envs,), float(schedule["phase"] == "fulladapt")
            ),
            "full_phase_train": raw_goal_distance.new_full(
                (self.num_envs,), float(schedule["phase"] == "fulltrain")
            ),
            "full_phase_stabilize": raw_goal_distance.new_full(
                (self.num_envs,), float(schedule["phase"] == "fullstabilize")
            ),
            "closedloop_phase_critic_warm": raw_goal_distance.new_full(
                (self.num_envs,), float(schedule["phase"] in {"closedwarm", "loopwarm"})
            ),
            "closedloop_phase_actor_adapt": raw_goal_distance.new_full(
                (self.num_envs,), float(schedule["phase"] in {"closedadapt", "loopadapt"})
            ),
            "closedloop_phase_train": raw_goal_distance.new_full(
                (self.num_envs,), float(schedule["phase"] in {"closedtrain", "looptrain"})
            ),
            "closedloop_phase_stabilize": raw_goal_distance.new_full(
                (self.num_envs,), float(schedule["phase"] in {"closedstable", "loopstable"})
            ),
            "legitimate_side_goal_candidate_count": side_goal_candidate.float(),
            "legitimate_side_goal_selected_count": side_goal_selected.float(),
            "legitimate_side_goal_bearing_abs_rad": clean_bearing_signed.abs(),
            "large_goal_correct_yaw_response": correct_yaw_response.float(),
            "vy_substitutes_yaw_count": vy_substitutes_yaw.float(),
            "translation_limiter_intervention": limiter_intervened.float(),
            "spawn_safe_point_share": self._p4_worker_extra[
                :, p4_contract.SPAWN_SAFE_POINT_INDEX
            ],
            "spawn_full_start_share": self._p4_worker_extra[
                :, p4_contract.SPAWN_FULL_START_INDEX
            ],
            "spawn_segment_index": spawn_segment.float(),
            "spawn_position_quartile": spawn_quartile.float(),
            "spawn_reason4_retry_count": self._p4_worker_extra[
                :, p4_contract.SPAWN_REASON4_RETRY_COUNT_INDEX
            ],
            "spawn_reason4_exhausted_count": self._p4_worker_extra[
                :, p4_contract.SPAWN_REASON4_EXHAUSTED_COUNT_INDEX
            ],
            "spawn_reason4_fallback_applied_count": self._p4_worker_extra[
                :, p4_contract.SPAWN_REASON4_FALLBACK_APPLIED_COUNT_INDEX
            ],
            "spawn_all_position_applied_count": self._p4_worker_extra[
                :, p4_contract.SPAWN_ALL_POSITION_APPLIED_COUNT_INDEX
            ],
            "spawn_validation_failure_count": self._p4_worker_extra[
                :, p4_contract.SPAWN_VALIDATION_FAILURE_COUNT_INDEX
            ],
            "spawn_write_failure_count": self._p4_worker_extra[
                :, p4_contract.SPAWN_WRITE_FAILURE_COUNT_INDEX
            ],
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
            "p4_spawn_hook_installed": self._p4_worker_extra[
                :, p4_contract.SPAWN_INSTALLED_INDEX
            ],
            "wall_stuck_raw_term": self._p4_worker_extra[
                :, p4_contract.STUCK_RAW_TERM_INDEX
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
        # The parent constructor applies the optimizer schedule before P4
        # diagnostic buffers are allocated.  Treat that initialization-only
        # call as unresolved auto; the first rollout-boundary schedule resolves
        # the branch after all diagnostics exist.
        if not hasattr(self, "_maze_diag_risk_positive_hist"):
            return "auto"
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
                elif name == "actor_stuck_head":
                    base = p4_contract.SAFETY_HEAD_LR
                    multiplier = float(
                        schedule.get("stuck_head_multiplier", schedule["safety_head_multiplier"])
                    )
                elif name.startswith("navigation_"):
                    layer = name.removeprefix("navigation_")
                    base = p4_contract.NAVIGATION_ENCODER_LRS[layer]
                    multiplier = float(schedule["navigation_multiplier"])
                else:
                    base = float(schedule.get("actor_lr", p4_contract.ACTOR_LR))
                    multiplier = float(schedule["actor_multiplier"])
                group["base_lr"] = base
                group["lr"] = base * multiplier
                trainable = multiplier > 0.0
                for parameter in group["params"]:
                    parameter.requires_grad_(trainable)
                    if not trainable:
                        parameter.grad = None
        if self.critic_optimizer is not None:
            self.critic_optimizer.param_groups[0]["lr"] = float(
                schedule.get(
                    "critic_lr",
                    p4_contract.CRITIC_LR
                    * float(schedule.get("critic_multiplier", 1.0)),
                )
            )
        if self.response_optimizer is not None:
            self.response_optimizer.param_groups[0]["lr"] = (
                p4_contract.ADAPTER_LR * float(schedule["adapter_multiplier"])
            )
        adapter_trainable = float(schedule["adapter_multiplier"]) > 0.0
        for parameter in self.response_adapter.parameters():
            parameter.requires_grad_(adapter_trainable)
            if not adapter_trainable:
                parameter.grad = None
        return schedule

    def maybe_unfreeze_cnn(self, effective_seconds: float) -> bool:
        del effective_seconds
        previous = self.cnn_unfrozen
        self._apply_training_schedule(self.session_effective_seconds)
        return (not previous) and self.cnn_unfrozen

    def update_training_clocks(
        self,
        session_wall_seconds: float,
        *,
        session_effective_seconds: float | None = None,
    ) -> None:
        self.session_wall_seconds = max(0.0, float(session_wall_seconds))
        supplied_effective_seconds = (
            None
            if session_effective_seconds is None
            else max(0.0, float(session_effective_seconds))
        )
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
            training_seconds = (
                self.session_wall_seconds
                if supplied_effective_seconds is None
                else supplied_effective_seconds
            )
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
            self._spawn_segment.zero_()
            self._max_segment_reached.zero_()
            self._segment_state_initialized.zero_()
            self._maze_credit_earned.zero_()
            self._translation_alpha_prev.fill_(1.0)
            self._translation_limiter_diagnostics = {}
            self._near_goal_capture_diagnostics = {}
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

    def _migrate_previous_p4_actor_optimizer(
        self, old_state: dict
    ) -> tuple[dict, dict[str, object]]:
        """Restore compatible previous P4 Adam groups by stable group name."""
        if not isinstance(old_state, dict):
            raise KeyError("P4 warm start missing high-level actor optimizer")
        validate_state_dict_finite(old_state, "P4 previous actor optimizer")
        old_groups = old_state.get("param_groups")
        old_states = old_state.get("state")
        if not isinstance(old_groups, list) or not isinstance(old_states, dict):
            raise ValueError("P4 previous actor optimizer is malformed")

        old_by_name = {str(group.get("name", "")): group for group in old_groups}
        current_state = self.actor_optimizer.state_dict()
        restored_parameters = 0
        restored_groups: list[str] = []
        fresh_groups: list[str] = []
        for object_group, serialized_group in zip(
            self.actor_optimizer.param_groups, current_state["param_groups"]
        ):
            name = str(object_group.get("name", ""))
            source_group = old_by_name.get(name)
            if name == "actor_stuck_head" and source_group is None:
                fresh_groups.append(name)
                continue
            if source_group is None:
                raise ValueError(
                    f"P4 previous actor optimizer missing group {name!r}"
                )
            source_ids = list(source_group.get("params", ()))
            target_ids = list(serialized_group.get("params", ()))
            if len(source_ids) != len(target_ids):
                raise ValueError(
                    f"P4 previous actor optimizer group mismatch {name!r}: "
                    f"source={len(source_ids)} target={len(target_ids)}"
                )
            for key, value in source_group.items():
                if key != "params":
                    serialized_group[key] = copy.deepcopy(value)
            serialized_group["params"] = target_ids
            for source_id, target_id, parameter in zip(
                source_ids, target_ids, object_group["params"]
            ):
                source = old_states.get(source_id)
                if not isinstance(source, dict):
                    continue
                copied = {}
                for key, value in source.items():
                    if torch.is_tensor(value):
                        if value.ndim and value.shape != parameter.shape:
                            raise ValueError(
                                "P4 previous actor optimizer tensor mismatch "
                                f"{name}.{key}: source={tuple(value.shape)} "
                                f"target={tuple(parameter.shape)}"
                            )
                        copied[key] = value.detach().clone()
                    else:
                        copied[key] = copy.deepcopy(value)
                current_state["state"][target_id] = copied
                restored_parameters += 1
            restored_groups.append(name)
        unexpected = sorted(set(old_by_name) - set(restored_groups))
        if unexpected:
            raise ValueError(
                "P4 previous actor optimizer has unexpected groups: "
                + ", ".join(unexpected)
            )
        validate_state_dict_finite(
            current_state, "P4 migrated actor optimizer"
        )
        return current_state, {
            "mapping_basis": "stable_optimizer_group_name_and_parameter_order",
            "restored_parameters": restored_parameters,
            "restored_groups": restored_groups,
            "fresh_groups": fresh_groups,
        }

    def load_bundle(self, path: str, *, platform_model_id) -> str:
        raw = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(raw, dict):
            raise ValueError("P4 checkpoint payload must be a mapping")
        if raw.get("stage_type") == self.STAGE_TYPE:
            bundle, _ = normalize_kaiwu_train_bundle(raw)
            contracts = bundle.get("contracts", {})
            exact_training_contract = p4_contract.training_contract(
                self.stuck_reset_contract,
                self.training_profile,
            )
            exact_reward_contract = p4_contract.reward_contract(
                self.stuck_reset_contract,
                self.training_profile,
            )
            exact_command_contract = p4_contract.command_contract(
                self.training_profile
            )
            saved_training = contracts.get("training")
            saved_reward = contracts.get("reward")
            saved_command = contracts.get("command")
            if (
                isinstance(saved_training, dict)
                and saved_training.get("version")
                == exact_training_contract["version"]
                and saved_training != exact_training_contract
            ):
                raise ValueError("P4 exact resume training contract mismatch")
            if (
                isinstance(saved_training, dict)
                and saved_training.get("version")
                == exact_training_contract["version"]
                and saved_reward != exact_reward_contract
            ):
                raise ValueError("P4 exact resume reward contract mismatch")
            if (
                isinstance(saved_training, dict)
                and saved_training.get("version")
                == exact_training_contract["version"]
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
            global_state = compatible.get("training_states", {}).get("global", {})
            if exact_compatible:
                saved_scope = global_state.get("train_scope")
                expected_scope = (
                    "maze_closed_loop_v3_actor_critic_only"
                    if self.training_profile == "maze_closed_loop_v3"
                    else (
                        "maze_actor_critic_only"
                        if self.training_profile == "maze_credit_repair"
                        else "high_level_and_response_adapter"
                    )
                )
                if saved_scope != expected_scope:
                    raise ValueError(
                        "P4 exact resume train_scope mismatch: "
                        f"saved={saved_scope!r} expected={expected_scope!r}"
                    )
            # The inherited loader validates P2's historical scope string.
            # P4 has already validated its stricter scope above, so only the
            # temporary compatibility view is translated here.
            if isinstance(global_state, dict):
                global_state["train_scope"] = "high_level_and_response_adapter"
            if not exact_compatible:
                optimizers = compatible.get("optimizers")
                if not isinstance(optimizers, dict):
                    raise KeyError("P4 warm start missing optimizer payload")
                maze_profile = self.training_profile in self.MAZE_PROFILES
                if "high_level_critic" not in optimizers:
                    raise KeyError("P4 closed-loop warm start missing Critic optimizer")
                if self.training_profile == "maze_closed_loop_v3":
                    optimizers["high_level_actor"] = copy.deepcopy(
                        self.actor_optimizer.state_dict()
                    )
                    optimizers["high_level_critic"] = copy.deepcopy(
                        self.critic_optimizer.state_dict()
                    )
                    actor_optimizer_report = {
                        "status": "reset",
                        "reason": "new_reward_and_command_dynamics_contract",
                    }
                    critic_optimizer_report = {
                        "status": "reset",
                        "reason": "critic_recalibration_without_weight_reset",
                    }
                elif self.training_profile == "maze_credit_repair":
                    # Preserve the historical credit-repair boundary: this
                    # profile changes the objective and horizon, so its Actor
                    # and Critic Adam state must start fresh, and the Critic
                    # value estimate/statistics must be rebuilt.  Do not let
                    # the new r3 warm-start policy silently change old
                    # checkpoint semantics.
                    optimizers["high_level_actor"] = copy.deepcopy(
                        self._credit_fresh_actor_optimizer_state
                    )
                    optimizers["high_level_critic"] = copy.deepcopy(
                        self._credit_fresh_critic_optimizer_state
                    )
                    actor_optimizer_report = {
                        "status": "reset",
                        "reason": "maze_credit_assignment_objective_change",
                    }
                    critic_optimizer_report = {
                        "status": "reset",
                        "reason": "maze_credit_reward_and_horizon_change",
                    }
                else:
                    migrated_actor_optimizer, actor_optimizer_report = (
                        self._migrate_previous_p4_actor_optimizer(
                            optimizers.get("high_level_actor")
                        )
                    )
                    optimizers["high_level_actor"] = migrated_actor_optimizer
                    critic_optimizer_report = {
                        "status": "preserved",
                        "reason": "critic_observation_and_value_contract_unchanged",
                    }
                high_state = compatible.get("training_states", {}).get("high_level", {})
                if isinstance(high_state, dict):
                    warm_schedule = p4_contract.training_schedule(
                        0.0,
                        branch=(
                            "closed_loop_v3"
                            if self.training_profile == "maze_closed_loop_v3"
                            else ("credit_repair" if maze_profile else "actor_attack")
                        ),
                    )
                    inherited_seconds = max(
                        0.0,
                        float(high_state.get("lifetime_effective_seconds", 0.0)),
                        float(high_state.get("effective_training_seconds", 0.0)),
                        float(high_state.get("session_effective_seconds", 0.0)),
                    )
                    high_state.update(
                        effective_training_seconds=0.0,
                        session_effective_seconds=0.0,
                        lifetime_effective_seconds=inherited_seconds,
                        lifetime_base_seconds=inherited_seconds,
                        frame_count=0,
                        iteration=0,
                        actor_gradient_steps=0,
                        critic_gradient_steps=0,
                        nav_ticks=0,
                        skipped_nonfinite=0,
                        nonfinite_action_fallbacks=0,
                        invalid_transition_count=0,
                    )
                    high_state["optimizer_phase"] = str(warm_schedule["phase"])
                    high_state["entropy_coefficient"] = float(
                        warm_schedule["entropy_coefficient"]
                    )
                    high_state["cnn_unfrozen"] = (
                        float(warm_schedule["navigation_multiplier"]) > 0.0
                    )
                    if self.training_profile == "maze_credit_repair":
                        high_state["return_statistics"] = {
                            "count": 0,
                            "mean": 0.0,
                            "m2": 0.0,
                            "value_normalization_enabled": False,
                        }
                    elif not isinstance(high_state.get("return_statistics"), dict):
                        raise KeyError(
                            "P4 closed-loop warm start missing Critic return statistics"
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
                        "p4_actor_optimizer": actor_optimizer_report,
                        "p4_critic_optimizer": critic_optimizer_report,
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
            high = (bundle.get("modules") or {}).get("high_level") or {}
            if exact_compatible:
                self._load_leaf(
                    high,
                    "actor_stuck_head",
                    self.stuck_head,
                    class_name="P4ActorStuckHead",
                    spec=p4_actor_stuck_head_spec(),
                    context="P4 exact resume high_level",
                )
                self._load_p4_state(original_p4_state)
            else:
                stuck_head_loaded = False
                if isinstance(high.get("actor_stuck_head"), dict):
                    self._load_leaf(
                        high,
                        "actor_stuck_head",
                        self.stuck_head,
                        class_name="P4ActorStuckHead",
                        spec=p4_actor_stuck_head_spec(),
                        context=(
                            "P4 legacy full-track warm start high_level"
                            if self.training_profile == "full_track"
                            else "P4 maze warm start diagnostic head"
                        ),
                    )
                    stuck_head_loaded = True
                if self.training_profile == "maze_credit_repair":
                    self.critic.load_state_dict(
                        self._credit_fresh_critic_state, strict=True
                    )
                    self.critic_optimizer.load_state_dict(
                        copy.deepcopy(self._credit_fresh_critic_optimizer_state)
                    )
                    self.actor_optimizer.load_state_dict(
                        copy.deepcopy(self._credit_fresh_actor_optimizer_state)
                    )
                    if not stuck_head_loaded:
                        self.stuck_head.load_state_dict(
                            self._credit_fresh_stuck_head_state, strict=True
                        )
                    self.return_statistics = {
                        "count": 0,
                        "mean": 0.0,
                        "m2": 0.0,
                        "value_normalization_enabled": False,
                    }
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
                    warm_profile = (
                        "Maze closed-loop v3"
                        if self.training_profile == "maze_closed_loop_v3"
                        else (
                            "Maze closed-loop v2"
                            if self.training_profile == "maze_credit_repair"
                            else "full-track legacy-contract"
                        )
                    )
                    optimizer_note = (
                        "Actor/Critic optimizer moments reset; weights and return "
                        "statistics preserved"
                        if self.training_profile == "maze_closed_loop_v3"
                        else "Actor/Critic optimizer moments and return statistics preserved"
                    )
                    self.logger.warning(
                        f"[P4NavPPO] previous P4 contract loaded as {warm_profile} "
                        f"warm start; {optimizer_note}"
                    )
            self._configure_adapter_contract()
            if exact_compatible:
                return f"p4_{mode}"
            warm_disposition = (
                "maze_closed_loop_v3_warm_start"
                if self.training_profile == "maze_closed_loop_v3"
                else (
                    "maze_credit_repair_warm_start"
                    if self.training_profile == "maze_credit_repair"
                    else "full_track_legacy_contract_warm_start"
                )
            )
            return f"p4_{warm_disposition}"
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
            "training_profile": self.training_profile,
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
            "mirror_rng_state": self.mirror_generator.get_state().cpu(),
            "auxiliary_calibration": dict(self._auxiliary_calibration),
            "actor_stuck_positive_ema": self.actor_stuck_positive_ema,
            "recovery_monitor": copy.deepcopy(self._p4_recovery_monitor_state),
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
        if str(state.get("training_profile", self.training_profile)) != self.training_profile:
            raise ValueError("P4 exact resume training profile mismatch")
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
        mirror_rng_state = state.get("mirror_rng_state")
        if not torch.is_tensor(mirror_rng_state):
            raise ValueError("P4 exact resume missing mirror RNG state")
        self.mirror_generator.set_state(mirror_rng_state.cpu())
        calibration = state.get("auxiliary_calibration")
        if not isinstance(calibration, dict):
            raise ValueError("P4 exact resume missing auxiliary calibration")
        self._auxiliary_calibration = {
            name: float(calibration.get(name, 0.0))
            for name in (
                "combined_ratio", "teacher_ratio", "camera_ratio", "mirror_ratio",
                "stuck_ratio", "teacher_valid_steps", "stuck_valid_steps",
            )
        }
        self.actor_stuck_positive_ema = float(
            state.get("actor_stuck_positive_ema", 0.10)
        )
        recovery_monitor = state.get("recovery_monitor")
        if not isinstance(recovery_monitor, dict):
            raise ValueError("P4 exact resume missing recovery monitor state")
        event_times = recovery_monitor.get("event_times")
        if not isinstance(event_times, (list, tuple)):
            raise ValueError("P4 exact resume invalid recovery event times")
        parsed_times = [float(value) for value in event_times]
        if not all(torch.isfinite(torch.tensor(parsed_times)).tolist()):
            raise ValueError("P4 exact resume non-finite recovery event time")
        self._p4_recovery_monitor_state = {
            "event_times": parsed_times,
            "success_lifetime_count": int(
                recovery_monitor.get("success_lifetime_count", 0)
            ),
            "candidate_lifetime_count": int(
                recovery_monitor.get("candidate_lifetime_count", 0)
            ),
            "terminal_lifetime_count": int(
                recovery_monitor.get("terminal_lifetime_count", 0)
            ),
        }
        self._apply_training_schedule(self.session_effective_seconds)

    def save_training_bundle(self, path: str, *, platform_model_id) -> str:
        super().save_training_bundle(path, platform_model_id=platform_model_id)
        payload = torch.load(path, weights_only=False, map_location="cpu")
        feedback, feedback_digest = self._feedback_contract()
        payload["contracts"] = {
            **p4_contract.contract_metadata(
                self.stuck_reset_contract,
                self.training_profile,
            ),
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
        payload["training_states"]["global"]["train_scope"] = (
            "maze_closed_loop_v3_actor_critic_only"
            if self.training_profile == "maze_closed_loop_v3"
            else (
                "maze_actor_critic_only"
                if self.training_profile == "maze_credit_repair"
                else "high_level_and_response_adapter"
            )
        )
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
        payload["modules"]["high_level"]["actor_stuck_head"] = self._leaf(
            "P4ActorStuckHead",
            p4_actor_stuck_head_spec(),
            self.stuck_head.state_dict(),
        )
        payload["modules"]["high_level"]["actor_stuck_head"]["training_only"] = True
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
