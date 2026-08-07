#!/usr/bin/env python3
"""P4 Track navigation PPO over a structurally complete frozen P3 parent."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
from agent_ppo.feature import p2_contract, p3_contract, p4_contract
from agent_ppo.feature.p4_camera import P4SharedCameraState
from agent_ppo.feature.p4_goal_belief import GoalBeliefChainV2
from agent_ppo.model.p2_high_level import P4ActorStuckHead

from agent_ppo.p4.checkpoint import P4CheckpointMixin
from agent_ppo.p4.diagnostics import P4DiagnosticsMixin
from agent_ppo.p4.rewards import P4RewardMixin
from agent_ppo.p4.runtime import P4RuntimeMixin
from agent_ppo.p4.teacher import P4TeacherMixin
from agent_ppo.p4.training import P4TrainingMixin
from agent_ppo.p4.profiles import (
    PROFILE_FULL_TRACK,
    PROFILE_MAZE_CLOSED_LOOP_V3,
    PROFILE_MAZE_CREDIT_REPAIR,
    PROFILE_MAZE_INSTANT_COMMAND_R4,
    PROFILE_MAZE_INSTANT_REPAIR2H,
    PROFILE_MAZE_STABLE_DIRECTION_SMOKE,
    PROFILE_MAZE_STABLE_DIRECTION_8H,
    get_training_profile,
)


class AlgorithmP4NavPPO(
    P4RuntimeMixin,
    P4TeacherMixin,
    P4RewardMixin,
    P4DiagnosticsMixin,
    P4TrainingMixin,
    P4CheckpointMixin,
    AlgorithmP2NavPPO,
):
    STAGE_TYPE = p4_contract.STAGE_TYPE
    MAZE_PROFILES = {
        PROFILE_MAZE_CREDIT_REPAIR,
        PROFILE_MAZE_CLOSED_LOOP_V3,
        PROFILE_MAZE_INSTANT_COMMAND_R4,
        PROFILE_MAZE_INSTANT_REPAIR2H,
        PROFILE_MAZE_STABLE_DIRECTION_SMOKE,
        PROFILE_MAZE_STABLE_DIRECTION_8H,
    }
    INSTANT_PROFILES = {
        PROFILE_MAZE_INSTANT_COMMAND_R4,
        PROFILE_MAZE_INSTANT_REPAIR2H,
        PROFILE_MAZE_STABLE_DIRECTION_SMOKE,
        PROFILE_MAZE_STABLE_DIRECTION_8H,
    }

    def __init__(self, *args, **kwargs):
        early_config, early_training_profile = self._prepare_early_config(kwargs)
        self._initialize_pre_parent_state(early_config, early_training_profile)
        super().__init__(*args, **kwargs)
        self._validate_runtime_contract()
        seed = int(self.config.get("p4_seed", 4100))
        self._initialize_navigation_runtime_state(seed)
        self._initialize_transport_camera_state()
        self._initialize_diagnostic_reward_state()
        self._initialize_teacher_state(seed)
        self._freeze_parent_models()
        self._initialize_training_state()

    @staticmethod
    def _prepare_early_config(kwargs) -> tuple[dict, str]:
        early_config = dict(kwargs.get("config") or {})
        early_training_profile = str(
            early_config.get("training_profile", PROFILE_FULL_TRACK)
        )
        profile_spec = get_training_profile(early_training_profile)
        early_command_contract = p4_contract.command_contract(
            early_training_profile
        )
        early_config.setdefault("maze_training_branch", profile_spec.schedule_branch)
        early_config.setdefault(
            "nav_period_frames", p4_contract.P4_NAV_PERIOD_FRAMES
        )
        early_config.setdefault(
            "command_transition_mode",
            str(early_command_contract.get("command_transition_mode", "slew")),
        )
        if "slew_rate" in early_command_contract:
            early_config.setdefault(
                "slew_rate", tuple(early_command_contract["slew_rate"])
            )
        if "slew_release_rate" in early_command_contract:
            early_config.setdefault(
                "slew_release_rate",
                tuple(early_command_contract["slew_release_rate"]),
            )
        kwargs["config"] = early_config
        return early_config, early_training_profile

    def _initialize_pre_parent_state(
        self, early_config: dict, early_training_profile: str
    ) -> None:
        self.maze_training_branch = str(
            early_config.get(
                "maze_training_branch",
                get_training_profile(early_training_profile).schedule_branch,
            )
        )
        self.training_profile = early_training_profile
        self._resolved_maze_training_branch = None
        self.session_wall_seconds = 0.0
        self.diagnostic_elapsed_seconds = 0.0
        self._training_clock_origin_seconds = None

    def _validate_runtime_contract(self) -> None:
        profile_spec = get_training_profile(self.training_profile)
        if self.maze_training_branch != profile_spec.schedule_branch:
            raise ValueError(
                f"P4 {self.training_profile} requires schedule branch "
                f"{profile_spec.schedule_branch!r}; got "
                f"{self.maze_training_branch!r}"
            )
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
        expected_command_contract = p4_contract.command_contract(
            self.training_profile
        )
        expected_transition_mode = str(
            expected_command_contract.get("command_transition_mode", "slew")
        )
        if self.command.command_transition_mode != expected_transition_mode:
            raise ValueError(
                "P4 runtime command transition mode does not match contract: "
                f"runtime={self.command.command_transition_mode!r} "
                f"expected={expected_transition_mode!r}"
            )
        if expected_transition_mode == "slew":
            runtime_slew = tuple(float(value) for value in self.command_slew_rate)
            runtime_release = tuple(
                float(value) for value in self.command_slew_release_rate
            )
            expected_slew = tuple(expected_command_contract["slew_rate"])
            expected_release = tuple(
                expected_command_contract["slew_release_rate"]
            )
            if runtime_slew != expected_slew or runtime_release != expected_release:
                raise ValueError(
                    "P4 runtime slew does not match command contract: "
                    f"slew={runtime_slew!r} release={runtime_release!r} "
                    f"expected={expected_slew!r}/{expected_release!r}"
                )
        else:
            expected_hold = int(expected_command_contract.get("hold_frames", 0))
            if int(self.command.hold_frames) != expected_hold:
                raise ValueError(
                    "P4 instant-hold frame count does not match command contract: "
                    f"runtime={self.command.hold_frames} expected={expected_hold}"
                )

    def _initialize_navigation_runtime_state(self, seed: int) -> None:
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
            self.config.get(
                "maze_training_branch",
                get_training_profile(self.training_profile).schedule_branch,
            )
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
        self._previous_policy_command = torch.zeros_like(self._last_policy_command)
        self._previous_exec_command = torch.zeros_like(self._last_policy_command)
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

    def _initialize_transport_camera_state(self) -> None:
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

    def _initialize_diagnostic_reward_state(self) -> None:
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

    def _initialize_teacher_state(self, seed: int) -> None:
        self._teacher_navigation_encoder = copy.deepcopy(self.navigation_encoder).to(
            self.device
        )
        self._teacher_actor = copy.deepcopy(self.actor).to(self.device)
        self._teacher_hidden = None
        self._parent_anchor_actor = copy.deepcopy(self.actor).to(self.device)
        self._parent_anchor_hidden = None
        self._parent_anchor_normalized_mean = torch.zeros(
            self.num_envs, p2_contract.ACTION_DIM, device=self.device
        )
        self._parent_anchor_log_std = torch.zeros_like(
            self._parent_anchor_normalized_mean
        )
        self._parent_anchor_mask = torch.zeros(
            self.num_envs, 1, device=self.device
        )
        self.parent_anchor_source_sha256 = None
        self.parent_anchor_digest = None
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
            "anchor_ratio": 0.0,
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

    def _freeze_parent_models(self) -> None:
        for module in (
            self._teacher_navigation_encoder,
            self._teacher_actor,
            self._parent_anchor_actor,
        ):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self._initial_low_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        self.cnn_unfrozen = False
        self.current_vy_trusted_limit = p4_contract.P4_MAX_ABS_VY
        self.current_vy_hard_limit = p4_contract.P4_MAX_ABS_VY

    def _initialize_training_state(self) -> None:
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
        closed_loop_profile = self.training_profile in {
            PROFILE_MAZE_CLOSED_LOOP_V3,
            PROFILE_MAZE_INSTANT_COMMAND_R4,
            PROFILE_MAZE_INSTANT_REPAIR2H,
            PROFILE_MAZE_STABLE_DIRECTION_SMOKE,
            PROFILE_MAZE_STABLE_DIRECTION_8H,
        }
        context = self._frame_stuck_context(pending, aux, reset, safe3)
        masks = self._frame_teacher_masks(
            pending,
            reset,
            safe3,
            safe5,
            safe5_valid,
            scanner_valid,
            closed_loop_profile,
            context,
        )
        anchor_mask = self._frame_parent_anchor_mask(
            pending,
            safe5_valid,
            scanner_valid,
            closed_loop_profile,
            context,
            masks,
        )
        self._parent_anchor_mask.copy_(anchor_mask.float().unsqueeze(-1))
        self._store_frame_teacher_labels(
            pending, safe3, safe5, context, masks
        )
        return result, critic_obs, aux

    def _frame_stuck_context(self, pending, aux, reset, safe3):
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
        policy_or_exec_intent |= self._last_policy_command[:, 2].abs() > 0.10
        policy_or_exec_intent |= self.command.exec_cmd[:, 2].abs() > 0.10
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
        return {
            "mapping_valid": mapping_valid,
            "wall_evidence": wall_evidence,
            "worker_candidate": worker_candidate,
            "policy_or_exec_intent": policy_or_exec_intent,
            "push_grace": push_grace,
            "episode_grace": episode_grace,
            "alive": alive,
            "stuck_mask": stuck_mask,
            "stuck_label": stuck_label,
            "stuck_weight": stuck_weight,
        }

    def _frame_teacher_masks(
        self,
        pending,
        reset,
        safe3,
        safe5,
        safe5_valid,
        scanner_valid,
        closed_loop_profile,
        context,
    ):
        guidance = p4_contract.teacher_guidance_mask(
            alive=context["alive"],
            scanner_valid=scanner_valid,
            mapping_valid=context["mapping_valid"],
            terminal=torch.zeros_like(context["alive"]),
            reset=reset,
            push_grace=context["push_grace"],
            episode_grace=context["episode_grace"],
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
        teacher_context_mask = (
            context["alive"]
            & context["mapping_valid"]
            & (safe5_valid if closed_loop_profile else scanner_valid)
            & ~reset
            & ~context["push_grace"]
            & ~context["episode_grace"]
        )
        teacher_values = safe5 if closed_loop_profile else safe3
        sector_angles = pending["target_cmd3"].new_tensor(
            (70.0, 35.0, 0.0, -35.0, -70.0)
            if closed_loop_profile
            else (35.0, 0.0, -35.0)
        ) * (math.pi / 180.0)
        policy_heading = torch.atan2(
            pending["target_cmd3"][:, 1],
            pending["target_cmd3"][:, 0].clamp_min(1.0e-4),
        )
        heading_weights = torch.softmax(
            8.0 * torch.cos(policy_heading[:, None] - sector_angles[None, :]),
            dim=-1,
        )
        heading_safe = (heading_weights * teacher_values).sum(dim=-1)
        best_safe = teacher_values.max(dim=-1).values
        policy_speed = torch.linalg.vector_norm(
            pending["target_cmd3"][:, :2], dim=-1
        )
        edge_mask = (
            teacher_mask
            & (best_safe >= 0.65)
            & ((best_safe - heading_safe) >= 0.15)
            & (heading_safe < 0.65)
            & (policy_speed > 0.10)
        )
        if closed_loop_profile:
            left_clearance = teacher_values[:, :2].amax(dim=-1)
            right_clearance = teacher_values[:, 3:].amax(dim=-1)
        else:
            left_clearance = teacher_values[:, 0]
            right_clearance = teacher_values[:, 2]
        recovery_mask = (
            context["stuck_label"]
            & (torch.maximum(left_clearance, right_clearance) >= 0.65)
            & ((left_clearance - right_clearance).abs() >= 0.12)
        )
        edge_mask &= ~recovery_mask
        normal_teacher_mask = teacher_mask & ~edge_mask & ~recovery_mask
        teacher_mask = normal_teacher_mask | edge_mask | recovery_mask
        return {
            "teacher_mask": teacher_mask,
            "goal_mask": goal_mask,
            "teacher_context_mask": teacher_context_mask,
            "normal_teacher_mask": normal_teacher_mask,
            "edge_mask": edge_mask,
            "recovery_mask": recovery_mask,
        }

    def _frame_parent_anchor_mask(
        self,
        pending,
        safe5_valid,
        scanner_valid,
        closed_loop_profile,
        context,
        masks,
    ):
        anchor_mask = (
            context["alive"]
            & (safe5_valid if closed_loop_profile else scanner_valid)
            & (self._last_goal_freshness >= 0.50)
            & (pending["predictive_collision_risk"].reshape(-1) < 0.35)
            & ~context["wall_evidence"]
            & ~context["worker_candidate"]
            & ~masks["edge_mask"]
            & ~masks["recovery_mask"]
            & ~context["push_grace"]
            & ~context["episode_grace"]
        )
        if self.command.command_transition_mode == "instant_hold_10hz":
            parent_target = p4_contract.map_normalized_action(
                self._parent_anchor_normalized_mean,
                p4_contract.P4_MAX_VX,
                goal_freshness=None,
            )
            anchor_mask &= p4_contract.instant_parent_anchor_reachable(
                parent_target,
                self._previous_exec_command,
            )
        return anchor_mask

    def _store_frame_teacher_labels(
        self, pending, safe3, safe5, context, masks
    ) -> None:
        mirror_eligible = (
            context["alive"]
            & context["mapping_valid"]
            & ~context["push_grace"]
            & ~context["worker_candidate"]
        )
        pending.update(
            {
                "teacher_safe3": safe3.detach(),
                "teacher_safe5": safe5.detach(),
                "teacher_goal_xy": self.goal_belief.estimate.detach(),
                "teacher_goal_freshness": self._last_goal_freshness.detach().unsqueeze(-1),
                "teacher_predictive_risk": pending["predictive_collision_risk"].reshape(-1, 1),
                "teacher_mask": masks["teacher_mask"].float().unsqueeze(-1),
                "teacher_context_mask": masks["teacher_context_mask"].float().unsqueeze(-1),
                "teacher_goal_mask": masks["goal_mask"].float().unsqueeze(-1),
                "teacher_weight": context["stuck_weight"].unsqueeze(-1),
                "stuck_label": context["stuck_label"].float().unsqueeze(-1),
                "stuck_motion_intent": context["policy_or_exec_intent"].float().unsqueeze(-1),
                "stuck_mask": context["stuck_mask"].float().unsqueeze(-1),
                "parent_anchor_mask": self._parent_anchor_mask.detach(),
                "teacher_normal_mask": masks["normal_teacher_mask"].float().unsqueeze(-1),
                "teacher_edge_mask": masks["edge_mask"].float().unsqueeze(-1),
                "teacher_recovery_mask": masks["recovery_mask"].float().unsqueeze(-1),
                "mirror_eligible": mirror_eligible.float().unsqueeze(-1),
            }
        )

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

    def _transition_extras(self) -> dict[str, torch.Tensor]:
        return {
            "clean_action_mean": self._clean_action_mean,
            "camera_aux_mask": self._camera_aux_mask,
            "parent_normalized_mean": self._parent_anchor_normalized_mean,
            "parent_log_std": self._parent_anchor_log_std,
            "parent_anchor_mask": self._parent_anchor_mask,
        }

    def _run_ppo_epochs(self) -> dict[str, float]:
        teacher_activation_mask = self.rollout.teacher_mask[: self.rollout.step] > 0.5
        if self.training_profile == PROFILE_MAZE_INSTANT_REPAIR2H:
            teacher_activation_mask = teacher_activation_mask | (
                self.rollout.teacher_context_mask[: self.rollout.step] > 0.5
            )
        teacher_activation_mask = teacher_activation_mask & (
            self.rollout.valid_mask[: self.rollout.step] > 0.5
        ) & (
            self.rollout.continuation_mask[: self.rollout.step] > 0.5
        )
        valid_teacher_steps = int(teacher_activation_mask.sum().item())
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
                "anchor_ratio": 0.0,
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
        if self.training_profile in self.INSTANT_PROFILES:
            schedule = p4_contract.training_schedule(
                self.session_effective_seconds,
                branch=self._effective_maze_branch(self.session_effective_seconds),
            )
            if float(schedule.get("adapter_multiplier", 0.0)) > 0.0:
                metrics = super()._adapter_update()
                metrics["adapter_frozen"] = 0.0
                return metrics
        if self.training_profile in self.MAZE_PROFILES:
            return {
                "adapter_loss": 0.0,
                "adapter_updates": 0.0,
                "adapter_frozen": 1.0,
            }
        return super()._adapter_update()

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
        unattributed = torch.as_tensor(
            kwargs.get("unattributed_boundary", torch.zeros_like(timeout)),
            device=self.device,
        ).reshape(-1).bool()
        terminal_for_aux = (reason != 0) | hard | timeout | unattributed
        if self.pending_tick is not None and bool(terminal_for_aux.any()):
            # Terminal/reset transitions never provide recovery supervision.
            for name in ("teacher_mask", "teacher_goal_mask", "stuck_mask", "mirror_eligible"):
                value = self.pending_tick.get(name)
                if torch.is_tensor(value):
                    value = value.clone()
                    value[terminal_for_aux] = 0.0
                    self.pending_tick[name] = value
        self._diagnostic_terminal_mask.copy_(terminal_for_aux)
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
        self._p4_episode_return[
            (reason != 0) | hard | timeout | unattributed
        ] = 0.0
        return full

    def update(self) -> dict[str, float]:
        return super().update()
