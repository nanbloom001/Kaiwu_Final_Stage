#!/usr/bin/env python3
"""P3 Standard joint-recovery algorithm coordinator."""

from __future__ import annotations
import hashlib
import math
import os
from uuid import uuid4
import torch
from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
from agent_ppo.checkpoint_io import (
    normalize_kaiwu_train_bundle,
    validate_state_dict_finite,
)
from agent_ppo.feature import p2_contract, p3_contract
from agent_ppo.feature.p3_depth_memory import (
    P35CameraFeatureTiming,
    P3DepthFaultAugmenter,
    P3MemoryAuxiliary,
)
from agent_ppo.feature.p3_gait import (
    P3ActionSmoothAuxiliary,
    P3GaitBaseline,
    P3MirrorAuxiliary,
    P35LowRewardShaper,
)
from agent_ppo.model.p2_high_level import (
    navigation_critic_spec,
    navigation_safety_head_spec,
)


P3_CONTRACT_WARM_START_MODES = frozenset(
    {
        "p3_radial_warm_start",
        "p3_stair_memory_warm_start",
        "p35_parent_selection_warm_start",
        "p35_previous_run_final_warm_start",
        "p35_fixed_parent_warm_start",
    }
)


def _validate_configured_parent_digest(path: str, expected_sha256: str | None) -> str:
    """Verify the explicitly selected warm-start artifact when a digest is set."""
    expected = str(expected_sha256 or "").strip().lower()
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if expected and actual != expected:
        raise ValueError(
            "P3 configured parent checkpoint SHA256 mismatch: "
            f"expected={expected} actual={actual} path={path}"
        )
    return actual


def _validate_resume_phase(raw: dict, global_state: dict, elapsed_s: float) -> str:
    expected = p3_contract.phase_for_elapsed(elapsed_s).name
    saved = raw.get("phase_label")
    state = global_state.get("compound_schedule_phase")
    if saved != expected or state != expected:
        raise RuntimeError(
            "P3 exact resume phase/time mismatch: "
            f"phase_label={saved!r} state_phase={state!r} expected={expected!r} "
            f"session_effective_seconds={elapsed_s}"
        )
    return expected


class AlgorithmP3HighPPO(AlgorithmP2NavPPO):
    STAGE_TYPE = "p3_standard_joint"

    def attach_low_algorithm(self, low_algorithm) -> None:
        self.p3_low_algorithm = low_algorithm
        self.defer_adapter_update = True
        self.track_safety_enabled = False
        for parameter in self.safety_head.parameters():
            parameter.requires_grad_(False)

    def _override_reward_components(self, components, **context):
        # P3 Standard keeps task progress and local-goal events as the high-
        # level objective. Track-specific safety, gait, collision, tracking and
        # stagnation terms remain diagnostics and must not train this policy.
        for name in (
            "success",
            "frontier_shaping",
            "tracking",
            "gait_symmetry",
            "body_collision",
            "predictive_collision_risk",
            "missed_safe_direction",
            "frontier_stagnation",
        ):
            components[name] = torch.zeros_like(components[name])
        reason = context.get("reason")
        if torch.is_tensor(reason):
            components["failure"] = (reason == 2).to(
                components["failure"]
            ) * -8.0
            components["timeout"] = (reason == 3).to(
                components["timeout"]
            ) * -4.0
        else:
            components["failure"] = torch.zeros_like(components["failure"])
            components["timeout"] = torch.zeros_like(components["timeout"])
        return components

    def _apply_external_reset(self, aux: torch.Tensor) -> None:
        reset = aux[:, 24] > 0.5
        super()._apply_external_reset(aux)
        low = getattr(self, "p3_low_algorithm", None)
        if low is not None and bool(reset.any()):
            ids = reset.nonzero(as_tuple=False).flatten()
            low.anchor_encoder.reset_hidden_state_for_envs(ids)

    def _low_level_frame(self, parts, critic_obs):
        del critic_obs
        low = getattr(self, "p3_low_algorithm", None)
        if low is None:
            raise RuntimeError("P3 high-level controller is missing its low-level executor")
        actions = low.actor_critic.act_from_proprio_depth(
            parts["proprio"], parts["depth"]
        )
        return actions, {}

    def _adapter_update(self):
        if getattr(self, "defer_adapter_update", False):
            return {"adapter_loss": 0.0, "adapter_updates": 0.0}
        return super()._adapter_update()

    def _actor_micro_loss(self, batch):
        batch = dict(batch)
        batch["safety_valid"] = torch.zeros_like(batch["safety_valid"])
        return super()._actor_micro_loss(batch)

    def _actor_update_enabled(self) -> bool:
        return False

    def update_adapter_after_policy(self):
        self.defer_adapter_update = False
        try:
            return super()._adapter_update()
        finally:
            self.defer_adapter_update = True

    def _apply_training_schedule(self, session_effective_seconds, *, allow_cnn_transition=True):
        del allow_cnn_transition
        elapsed = max(0.0, float(session_effective_seconds))
        self.session_effective_seconds = elapsed
        self.effective_training_seconds = elapsed
        self.lifetime_effective_seconds = self.lifetime_base_seconds + elapsed
        phase = p3_contract.phase_for_elapsed(elapsed)
        response_config = self.config.get("response_adapter") or {}
        base_adapter_lr = float(
            response_config.get("learning_rate", p2_contract.ADAPTER_LR)
        )
        calibration_adapter_lr = float(
            response_config.get("calibration_learning_rate", base_adapter_lr)
        )
        high_adapt_lr = float(
            response_config.get("high_adapt_learning_rate", base_adapter_lr)
        )
        self.current_vy_trusted_limit, self.current_vy_hard_limit = 0.20, 0.40
        if phase.name == "highadapt":
            nav, actor, critic, adapter_lr, entropy = (
                0.15,
                0.10,
                0.30,
                high_adapt_lr,
                0.006,
            )
        elif phase.name == "highslow":
            nav, actor, critic, adapter_lr, entropy = (
                0.20,
                0.15,
                0.25,
                high_adapt_lr,
                0.005,
            )
        elif phase.name in {"adaptercalib", "stable"}:
            nav, actor, critic, adapter_lr, entropy = (
                0.0,
                0.0,
                0.30,
                calibration_adapter_lr,
                0.0,
            )
        else:
            nav, actor, critic, adapter_lr, entropy = (
                0.0,
                0.0,
                0.0,
                base_adapter_lr,
                0.0,
            )
        self.cnn_unfrozen = nav > 0.0
        self.entropy_coefficient = entropy
        self.optimizer_phase = phase.name
        # This run never steps the high-level actor or critic.  Leave their
        # optimizer and scheduler state byte-for-byte inherited from the
        # parent; zero learning rates would still mutate the saved state.
        del nav, actor, critic
        self.response_optimizer.param_groups[0]["lr"] = adapter_lr
        for module in (
            getattr(self, "navigation_encoder", None),
            getattr(self, "actor", None),
            getattr(self, "critic", None),
            getattr(self, "safety_head", None),
        ):
            if module is not None:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        return {"phase": phase.name, "cnn_unfrozen": self.cnn_unfrozen}

    @property
    def current_phase(self):
        return p3_contract.phase_for_elapsed(self.session_effective_seconds).name

    def load_p2_parent_bundle(self, path: str, *, platform_model_id) -> str:
        """Load a P2 package as a new P3 session, tolerating RNG device changes."""
        raw = torch.load(path, weights_only=False, map_location="cpu")
        bundle, _ = normalize_kaiwu_train_bundle(raw)
        if raw.get("stage_type") != "p2_nav_ppo":
            raise ValueError("P3 warm start requires a P2 navigation parent")
        mode = self._load_p2_safe_direction_warm_start(
            bundle, path, platform_model_id
        )
        high = bundle.get("modules", {}).get("high_level", {})
        self._load_leaf(
            high,
            "navigation_safety_head",
            self.safety_head,
            class_name="NavigationSafetyHead",
            spec=navigation_safety_head_spec(),
            context="P3 warm start high_level",
        )
        return mode

    def load_p3_exact_bundle(self, raw: dict, path: str, *, platform_model_id) -> str:
        bundle, _ = normalize_kaiwu_train_bundle(raw)
        self._warn_platform_identity(bundle, platform_model_id)
        states = dict(bundle.get("training_states") or {})
        global_state = dict(states.get("global") or {})
        global_state["train_scope"] = "high_level_and_response_adapter"
        states["global"] = global_state
        bundle["training_states"] = states
        return self._load_exact_resume(bundle, platform_model_id, path)


class AlgorithmP3StandardJoint:
    STAGE_TYPE = "p3_standard_joint"

    def __init__(self, *, low_algorithm, high_algorithm, config, logger=None):
        self.low_algorithm = low_algorithm
        self.high_algorithm = high_algorithm
        self.config = dict(config or {})
        self.logger = logger
        self.session_effective_seconds = 0.0
        self.lifetime_base_seconds = 0.0
        self.current_iteration = 0
        self.low_updates = 0
        self.high_updates = 0
        self.parent_loaded = False
        if hasattr(self.high_algorithm, "attach_low_algorithm"):
            self.high_algorithm.attach_low_algorithm(self.low_algorithm)
        self.episode_subgoal_successes = torch.zeros(
            self.high_algorithm.num_envs,
            dtype=torch.long,
            device=self.high_algorithm.device,
        )
        self.episode_origin_xy = torch.zeros(
            self.high_algorithm.num_envs,
            2,
            dtype=torch.float32,
            device=self.high_algorithm.device,
        )
        self.episode_origin_valid = torch.zeros(
            self.high_algorithm.num_envs,
            dtype=torch.bool,
            device=self.high_algorithm.device,
        )
        self.episode_origin_reset_pending = torch.zeros_like(
            self.episode_origin_valid
        )
        self.best_radial_distance = torch.zeros(
            self.high_algorithm.num_envs,
            dtype=torch.float32,
            device=self.high_algorithm.device,
        )
        self.m3_proxy_latched = torch.zeros_like(self.episode_origin_valid)
        self.mirror_mapping_valid = False
        self.gait_baseline = P3GaitBaseline(self.high_algorithm.device)
        storage = getattr(self.low_algorithm, "storage", None)
        self.mirror_aux = P3MirrorAuxiliary(
            num_steps=int(
                getattr(storage, "num_transitions_per_env", self.config.get("num_steps_per_env", 80))
            ),
            num_envs=self.high_algorithm.num_envs,
            obs_dim=77,
            sequence_length=int(getattr(self.low_algorithm, "sequence_length", 16)),
            device=self.high_algorithm.device,
            seed=int(self.config.get("mirror_seed", 3197)),
        )
        self.low_algorithm.p3_mirror_aux = self.mirror_aux
        num_steps = int(
            getattr(storage, "num_transitions_per_env", self.config.get("num_steps_per_env", 128))
        )
        self.depth_fault = P3DepthFaultAugmenter(
            num_steps=num_steps,
            num_envs=self.high_algorithm.num_envs,
            depth_shape=(p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, p2_contract.DEPTH_CHANNELS),
            device=self.high_algorithm.device,
            seed=int(self.config.get("depth_fault_seed", 3329)),
        )
        self.memory_aux = P3MemoryAuxiliary(
            num_steps=num_steps,
            num_envs=self.high_algorithm.num_envs,
            obs_dim=77,
            device=self.high_algorithm.device,
            seed=int(self.config.get("memory_seed", 3331)),
        )
        camera_config = self.config.get("camera_timing") or {}
        self.camera_timing = P35CameraFeatureTiming(
            num_envs=self.high_algorithm.num_envs,
            feature_dim=32,
            device=self.high_algorithm.device,
            capacity=int(camera_config.get("feature_fifo_frames", 10)),
            seed=int(camera_config.get("capture_phase_seed", 3361)),
            control_dt_s=1.0 / float(camera_config.get("control_rate_hz", 50.0)),
            capture_rate_hz=float(camera_config.get("capture_rate_hz", 30.0)),
        )
        self.low_algorithm.p3_memory_aux = self.memory_aux
        self.action_smooth_aux = P3ActionSmoothAuxiliary()
        self.low_algorithm.p3_action_smooth_aux = self.action_smooth_aux
        self.low_reward_shaper = P35LowRewardShaper(
            self.high_algorithm.num_envs,
            self.high_algorithm.device,
            seed=int(self.config.get("p35_baseline_seed", 3373)),
        )
        terrain_size_x = float(self.config.get("terrain_size_x_m", 8.0))
        (
            self.platform_complete_radius_m,
            self.m3_target_radius_m,
            self.platform_boundary_radius_m,
        ) = p3_contract.platform_completion_radii(terrain_size_x)
        self._low_base_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(self.low_algorithm.optimizer.param_groups)
        }
        self._high_frozen_digest = None
        self._apply_phase()

    @property
    def current_phase(self):
        return p3_contract.phase_for_elapsed(self.session_effective_seconds).name

    def _apply_phase(self):
        phase = p3_contract.phase_for_elapsed(self.session_effective_seconds)
        self.low_algorithm.anchor_session_elapsed_hours = self.session_effective_seconds / 3600.0
        self.low_algorithm.current_phase = phase.name
        self.low_algorithm._set_trainable_phase(phase.name)
        self.low_algorithm.actor_critic.std.requires_grad_(False)
        low_lrs = {
            "calib": {"actor": 0.0, "lstm": 0.0, "critic": 3.0e-5},
            "gaitwarm": {"actor": 3.0e-6, "lstm": 1.5e-6, "critic": 3.0e-5},
            "gaitfull": {"actor": 3.0e-6, "lstm": 1.5e-6, "critic": 3.0e-5},
            "camfull": {"actor": 2.0e-6, "lstm": 1.0e-6, "critic": 2.0e-5},
            "pushwarm": {"actor": 1.5e-6, "lstm": 7.5e-7, "critic": 2.0e-5},
            "pushfull": {"actor": 1.0e-6, "lstm": 5.0e-7, "critic": 1.5e-5},
            "stable": {"actor": 0.0, "lstm": 0.0, "critic": 1.0e-5},
        }[phase.name]
        for index, group in enumerate(self.low_algorithm.optimizer.param_groups):
            name = str(group.get("name", index))
            group["lr"] = low_lrs.get(name, 0.0)
            trainable = group["lr"] > 0.0
            if name == "actor":
                for parameter in group["params"]:
                    parameter.requires_grad_(False)
                if trainable:
                    for parameter in self.low_algorithm.actor_critic.actor[-1].parameters():
                        parameter.requires_grad_(True)
            else:
                for parameter in group["params"]:
                    parameter.requires_grad_(trainable)
        self.low_algorithm.actor_critic.std.requires_grad_(False)
        self.high_algorithm.update_training_clocks(self.session_effective_seconds)

    def update_clock(self, elapsed_s):
        old = self.current_phase
        self.session_effective_seconds = max(0.0, float(elapsed_s))
        self._apply_phase()
        return self.current_phase != old

    def reset_live_state(self):
        self._ensure_radial_state()
        self.episode_subgoal_successes.zero_()
        self.episode_origin_xy.zero_()
        self.episode_origin_valid.zero_()
        self.episode_origin_reset_pending.zero_()
        self.best_radial_distance.zero_()
        self.m3_proxy_latched.zero_()
        self.high_algorithm.reset_live_state()
        self.low_algorithm.initialize_recurrent_states(self.high_algorithm.num_envs)
        self.camera_timing.reset(
            torch.ones(
                self.high_algorithm.num_envs,
                dtype=torch.bool,
                device=self.high_algorithm.device,
            )
        )

    def update_runtime_terrain_size(self, values: torch.Tensor) -> None:
        finite = values[torch.isfinite(values) & (values > 0.0)]
        if finite.numel() == 0:
            return
        terrain_size = float(finite.median())
        complete, target, boundary = p3_contract.platform_completion_radii(terrain_size)
        self.platform_complete_radius_m = complete
        self.m3_target_radius_m = target
        self.platform_boundary_radius_m = boundary

    def _ensure_radial_state(self, worker_aux: torch.Tensor | None = None) -> None:
        """Initialize transient radial state for compatibility construction paths."""
        if hasattr(self, "episode_origin_xy"):
            if not hasattr(self, "episode_origin_reset_pending"):
                self.episode_origin_reset_pending = torch.zeros_like(
                    self.episode_origin_valid
                )
            return
        if worker_aux is not None:
            num_envs = int(worker_aux.shape[0])
            device = worker_aux.device
        else:
            num_envs = int(self.high_algorithm.num_envs)
            device = self.high_algorithm.device
        self.episode_subgoal_successes = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.episode_origin_xy = torch.zeros(
            num_envs, 2, dtype=torch.float32, device=device
        )
        self.episode_origin_valid = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.episode_origin_reset_pending = torch.zeros_like(
            self.episode_origin_valid
        )
        self.best_radial_distance = torch.zeros(
            num_envs, dtype=torch.float32, device=device
        )
        self.m3_proxy_latched = torch.zeros(num_envs, dtype=torch.bool, device=device)
        terrain_size_x = float(getattr(self, "config", {}).get("terrain_size_x_m", 8.0))
        (
            self.platform_complete_radius_m,
            self.m3_target_radius_m,
            self.platform_boundary_radius_m,
        ) = p3_contract.platform_completion_radii(terrain_size_x)

    def sync_episode_origins(self, worker_aux: torch.Tensor) -> torch.Tensor:
        self._ensure_radial_state(worker_aux)
        pose_xy = worker_aux[:, 15:17].to(self.episode_origin_xy)
        reset = worker_aux[:, 24] > 0.5
        initialize = ~self.episode_origin_valid & ~(
            reset & self.episode_origin_reset_pending
        )
        if bool(initialize.any()):
            self.episode_origin_xy[initialize] = pose_xy[initialize]
            self.episode_origin_valid[initialize] = True
            self.episode_origin_reset_pending[initialize] = False
            self.best_radial_distance[initialize] = 0.0
            self.m3_proxy_latched[initialize] = False
        return reset

    def radial_snapshot(self, worker_aux: torch.Tensor) -> dict[str, torch.Tensor]:
        self._ensure_radial_state(worker_aux)
        self.sync_episode_origins(worker_aux)
        pose_xy = worker_aux[:, 15:17].to(self.episode_origin_xy)
        radius = p3_contract.radial_distance(pose_xy, self.episode_origin_xy)
        best_before = self.best_radial_distance.clone()
        reward, best = p3_contract.radial_new_best(radius, best_before)
        return {
            "radius": radius,
            "best_before": best_before,
            "best_after": best,
            "new_best_reward": reward,
            "hold": self.m3_proxy_latched | (radius >= self.platform_complete_radius_m),
        }

    def guard_high_commands(self, worker_aux: torch.Tensor) -> torch.Tensor:
        snapshot = self.radial_snapshot(worker_aux)
        root_xy = worker_aux[:, 15:17].to(self.episode_origin_xy)
        yaw = worker_aux[:, 17].to(self.episode_origin_xy)
        target, hold = p3_contract.cap_outward_body_command(
            self.high_algorithm.command.active_target,
            root_xy,
            self.episode_origin_xy,
            yaw,
            self.platform_complete_radius_m,
        )
        executed, _ = p3_contract.cap_outward_body_command(
            self.high_algorithm.command.exec_cmd,
            root_xy,
            self.episode_origin_xy,
            yaw,
            self.platform_complete_radius_m,
        )
        self.high_algorithm.command.active_target.copy_(target)
        self.high_algorithm.command.exec_cmd.copy_(executed)
        if bool(hold.any()):
            self.high_algorithm.command.active_target[hold] = 0.0
            self.high_algorithm.command.exec_cmd[hold] = 0.0
        return hold

    def observe_subgoal_and_terminal(
        self, worker_aux: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Maintain radial proxy and platform outcome accounting across rollouts."""
        self._ensure_radial_state(worker_aux)
        event = worker_aux[:, p2_contract.CURRENT_SEGMENT_INDEX].round().long()
        reached = event == p3_contract.SUBGOAL_EVENT_REACHED
        successes_before = self.episode_subgoal_successes.clone()
        m1_reached = reached & (successes_before == 0)
        m2_reached = reached & (successes_before == 1)
        self.episode_subgoal_successes += reached.long()
        reset = worker_aux[:, 24] > 0.5
        reason = worker_aux[:, 25].round().long()
        platform_success = reset & (reason == 1)
        previous_proxy = self.m3_proxy_latched.clone()
        previous_subgoals = self.episode_subgoal_successes.clone()
        pose_xy = worker_aux[:, 15:17].to(self.episode_origin_xy)
        initialize = ~self.episode_origin_valid & ~reset
        if bool(initialize.any()):
            self.episode_origin_xy[initialize] = pose_xy[initialize]
            self.episode_origin_valid[initialize] = True
            self.episode_origin_reset_pending[initialize] = False
        radius = p3_contract.radial_distance(pose_xy, self.episode_origin_xy)
        proxy_new = ~self.m3_proxy_latched & (
            radius >= self.platform_complete_radius_m
        )
        self.m3_proxy_latched |= proxy_new
        _, next_best = p3_contract.radial_new_best(
            radius, self.best_radial_distance
        )
        self.best_radial_distance.copy_(next_best)
        joint_success = p3_contract.joint_episode_success(
            platform_success,
            previous_subgoals,
            reset & (reason == 2),
        )
        agreement = platform_success & (previous_proxy | proxy_new)
        result = {
            "m1_reached_mask": m1_reached,
            "m2_reached_mask": m2_reached,
            "m3_proxy_new_mask": proxy_new,
            "p3_m1_success_count": m1_reached.float().sum(),
            "p3_m2_success_count": m2_reached.float().sum(),
            "p3_standard_success_count": proxy_new.float().sum(),
            "p3_platform_success_count": platform_success.float().sum(),
            "p3_joint_success_count": joint_success.float().sum(),
            "p3_proxy_platform_agreement_count": agreement.float().sum(),
            "p3_radial_distance_mean": radius.mean(),
            "p3_best_radial_distance_mean": self.best_radial_distance.mean(),
            "p3_m3_hold_share": self.m3_proxy_latched.float().mean(),
        }
        if bool(reset.any()):
            self.episode_subgoal_successes.masked_fill_(reset, 0)
            self.episode_origin_valid[reset] = False
            self.episode_origin_reset_pending[reset] = True
            self.best_radial_distance[reset] = 0.0
            self.m3_proxy_latched[reset] = False
        return result

    def _validate_anchor_cnn_reuse(self) -> None:
        live = self.low_algorithm.actor_critic.vision_encoder.cnn.state_dict()
        anchor = self.low_algorithm.anchor_encoder.cnn.state_dict()
        if live.keys() != anchor.keys() or any(
            not torch.equal(live[name].detach().cpu(), anchor[name].detach().cpu())
            for name in live
        ):
            raise RuntimeError(
                "P3 compact anchor replay requires identical frozen live/anchor CNNs"
            )

    def note_low_level_update(self) -> str:
        digest = self.high_algorithm._module_digest(
            (
                ("vision", self.high_algorithm.low_level_encoder),
                ("actor", self.high_algorithm.low_level_actor),
            )
        )
        self.high_algorithm.low_level_state_digest = digest
        self.high_algorithm.response_buffer.set_low_level_version(
            digest, self.low_updates
        )
        return digest

    def should_collect_low_rollout(self):
        return True

    def should_collect_high_rollout(self):
        return False

    def _current_high_frozen_digest(self) -> str:
        return self.high_algorithm._module_digest(
            (
                ("navigation_encoder", self.high_algorithm.navigation_encoder),
                ("actor", self.high_algorithm.actor),
                ("critic", self.high_algorithm.critic),
                ("safety_head", self.high_algorithm.safety_head),
            )
        )

    def _capture_high_frozen_digest(self) -> None:
        self._high_frozen_digest = self._current_high_frozen_digest()

    def _restore_high_value_state_for_short_adaptation(self, bundle) -> None:
        """Restore the parent value estimator after the policy-only reward loader."""
        high = (bundle.get("modules") or {}).get("high_level") or {}
        self.high_algorithm._load_leaf(
            high,
            "critic",
            self.high_algorithm.critic,
            class_name="P2NavigationCritic",
            spec=navigation_critic_spec(),
            context="P3 short high-level adaptation",
        )
        critic_optimizer = (bundle.get("optimizers") or {}).get(
            "high_level_critic"
        )
        if not isinstance(critic_optimizer, dict):
            raise KeyError(
                "P3 short high-level adaptation requires high_level_critic optimizer"
            )
        validate_state_dict_finite(
            critic_optimizer, "P3 short high-level adaptation critic optimizer"
        )
        self.high_algorithm.critic_optimizer.load_state_dict(critic_optimizer)

        high_state = (bundle.get("training_states") or {}).get("high_level") or {}
        return_statistics = high_state.get("return_statistics")
        if not isinstance(return_statistics, dict):
            raise KeyError(
                "P3 short high-level adaptation requires return statistics"
            )
        restored_statistics = {
            "count": int(return_statistics.get("count", 0)),
            "mean": float(return_statistics.get("mean", 0.0)),
            "m2": float(return_statistics.get("m2", 0.0)),
            "value_normalization_enabled": bool(
                return_statistics.get("value_normalization_enabled", False)
            ),
        }
        if (
            restored_statistics["count"] < 0
            or not math.isfinite(restored_statistics["mean"])
            or not math.isfinite(restored_statistics["m2"])
            or restored_statistics["m2"] < 0.0
        ):
            raise ValueError(
                "P3 short high-level adaptation return statistics are invalid"
            )
        self.high_algorithm.return_statistics = restored_statistics
        self.high_algorithm.actor_gradient_steps = int(
            high_state.get("actor_gradient_steps", 0)
        )
        self.high_algorithm.critic_gradient_steps = int(
            high_state.get("critic_gradient_steps", 0)
        )

    def _load_p3_stair_memory_warm_start(self, raw, path, *, platform_model_id):
        bundle, _ = normalize_kaiwu_train_bundle(raw)
        mode = self.high_algorithm._load_p2_reward_warm_start(
            bundle, path, platform_model_id
        )
        self._restore_high_value_state_for_short_adaptation(bundle)
        self._restore_low_state(raw, exact=False)
        optimizers = raw.get("optimizers") or {}
        low_optimizer = optimizers.get("low_level")
        if isinstance(low_optimizer, dict):
            self.low_algorithm.optimizer.load_state_dict(low_optimizer)
        actor_optimizer = optimizers.get("high_level_actor")
        if isinstance(actor_optimizer, dict):
            try:
                self.high_algorithm.actor_optimizer.load_state_dict(actor_optimizer)
            except (ValueError, RuntimeError) as exc:
                if self.logger:
                    self.logger.warning(
                        "[P3] radial warm start kept fresh actor optimizer because "
                        f"the parent state was incompatible: {exc}"
                    )
        # The clean-memory teacher is the selected F2 policy itself, not the
        # older anchor nested in that parent package.  Exact resume persists
        # and restores this new immutable snapshot.
        self.low_algorithm.anchor_encoder.load_state_dict(
            self.low_algorithm.actor_critic.vision_encoder.state_dict(), strict=True
        )
        self.low_algorithm.anchor_actor.load_state_dict(
            self.low_algorithm.actor_critic.actor.state_dict(), strict=True
        )
        low_state = (raw.get("training_states") or {}).get("low_level") or {}
        self.low_updates = int(low_state.get("gradient_steps", 0))
        global_state = (raw.get("training_states") or {}).get("global") or {}
        lifetime = float(
            global_state.get(
                "lifetime_effective_seconds",
                global_state.get("session_effective_seconds", 0.0),
            )
        )
        self.lifetime_base_seconds = max(0.0, lifetime)
        self.high_algorithm.lifetime_base_seconds = self.lifetime_base_seconds
        self.session_effective_seconds = 0.0
        self.current_iteration = 0
        self.high_updates = 0
        self.high_algorithm.session_effective_seconds = 0.0
        self.high_algorithm.effective_training_seconds = 0.0
        self.high_algorithm.lifetime_effective_seconds = self.lifetime_base_seconds
        self.note_low_level_update()
        self._validate_anchor_cnn_reuse()
        self.parent_loaded = True
        self.reset_live_state()
        self._capture_high_frozen_digest()
        self._apply_phase()
        return f"p3_stair_memory_warm_start:{mode}"

    def _restore_low_state(self, raw, *, exact: bool) -> None:
        low = ((raw.get("modules") or {}).get("low_level") or {})
        critic_name = "p3_critic"
        if not exact and critic_name not in low:
            critic_name = "critic"
            if self.logger:
                self.logger.warning(
                    "[P3] warm-start bundle has no p3_critic; using legacy low critic"
                )
        critic_leaf = low.get(critic_name) if isinstance(low, dict) else None
        critic_state = critic_leaf.get("state_dict") if isinstance(critic_leaf, dict) else None
        if not isinstance(critic_state, dict):
            raise KeyError(f"P3 warm start missing required low-level {critic_name}")
        self.low_algorithm.actor_critic.critic.load_state_dict(
            critic_state, strict=True
        )
        distribution = low.get("action_distribution", {})
        distribution_state = (
            distribution.get("state_dict", {})
            if isinstance(distribution, dict)
            else {}
        )
        std = (
            distribution_state.get("std")
            if isinstance(distribution_state, dict)
            else None
        )
        if not torch.is_tensor(std) and isinstance(distribution, dict):
            std = distribution.get("std")
        if isinstance(std, torch.Tensor):
            self.low_algorithm.actor_critic.std.data.copy_(
                std.to(self.low_algorithm.device)
            )
        if exact:
            anchor = low.get("p3_anchor") if isinstance(low, dict) else None
            if not isinstance(anchor, dict):
                raise KeyError("P3 exact resume missing immutable low-level anchor")
            encoder_state = anchor.get("encoder_state_dict")
            actor_state = anchor.get("actor_state_dict")
            if not isinstance(encoder_state, dict) or not isinstance(actor_state, dict):
                raise KeyError("P3 anchor is missing encoder or actor state")
            self.low_algorithm.anchor_encoder.load_state_dict(encoder_state, strict=True)
            self.low_algorithm.anchor_actor.load_state_dict(actor_state, strict=True)
            anchor_digest = self.high_algorithm._module_digest(
                (
                    ("anchor_encoder", self.low_algorithm.anchor_encoder),
                    ("anchor_actor", self.low_algorithm.anchor_actor),
                )
            )
            if anchor_digest != anchor.get("digest"):
                raise RuntimeError("P3 exact resume anchor digest mismatch")
            optimizer = (raw.get("optimizers") or {}).get("low_level")
            if not isinstance(optimizer, dict):
                raise KeyError("P3 exact resume missing low_level optimizer")
            self.low_algorithm.optimizer.load_state_dict(optimizer)
            state = (raw.get("training_states") or {}).get("low_level", {})
            if not isinstance(state, dict):
                raise ValueError("P3 exact resume low_level state must be a mapping")
            self.low_algorithm.current_iteration = int(
                state.get("current_iteration", 0)
            )
            self.low_updates = int(state.get("gradient_steps", 0))
            self.low_algorithm._restore_rng_state(
                state.get("rng_state"), self.logger
            )
            mirror_state = state.get("mirror_rng_state")
            if torch.is_tensor(mirror_state):
                self.mirror_aux.generator.set_state(mirror_state.cpu())
            depth_fault_state = state.get("depth_fault")
            memory_state = state.get("memory_auxiliary")
            camera_state = state.get("camera_timing")
            if (
                not isinstance(depth_fault_state, dict)
                or not isinstance(memory_state, dict)
                or not isinstance(camera_state, dict)
            ):
                raise KeyError("P3 exact resume missing stair-memory training state")
            self.depth_fault.load_state_dict(depth_fault_state)
            self.memory_aux.load_state_dict(memory_state)
            self.camera_timing.load_state_dict(camera_state)
            gait_state = state.get("gait_baseline")
            if isinstance(gait_state, dict):
                self.gait_baseline.load_state_dict(gait_state)
            reward_state = state.get("p35_reward_baseline")
            if isinstance(reward_state, dict):
                self.low_reward_shaper.load_state_dict(reward_state)

    def load_parent(self, path, *, platform_model_id):
        mode = self.high_algorithm.load_p2_parent_bundle(
            path, platform_model_id=platform_model_id
        )
        raw = torch.load(path, weights_only=False, map_location="cpu")
        self._restore_low_state(raw, exact=False)
        self.low_algorithm.anchor_encoder.load_state_dict(
            self.low_algorithm.actor_critic.vision_encoder.state_dict(), strict=True
        )
        self.low_algorithm.anchor_actor.load_state_dict(
            self.low_algorithm.actor_critic.actor.state_dict(), strict=True
        )
        self._validate_anchor_cnn_reuse()
        self.low_updates = 0
        self.note_low_level_update()
        self.lifetime_base_seconds = float(getattr(self.high_algorithm, "lifetime_effective_seconds", 0.0))
        self.high_algorithm.lifetime_base_seconds = self.lifetime_base_seconds
        self.session_effective_seconds = 0.0
        self.high_algorithm.reset_live_state()
        self.low_algorithm.initialize_recurrent_states(self.high_algorithm.num_envs)
        self.parent_loaded = True
        self._capture_high_frozen_digest()
        self._apply_phase()
        return f"p3_joint_warm_start:{mode}"

    def load_checkpoint(self, path, *, platform_model_id):
        raw = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(raw, dict):
            raise ValueError("P3 checkpoint must be a mapping")
        contracts = raw.get("contracts") or {}
        exact_resume = (
            raw.get("stage_type") == self.STAGE_TYPE
            and contracts.get("p3_standard_joint") == p3_contract.contract()
        )
        if (
            not exact_resume
            and str(self.config.get("load_mode", ""))
            == "p35_fixed_parent_warm_start"
        ):
            _validate_configured_parent_digest(
                path, self.config.get("parent_checkpoint_sha256")
            )
        if raw.get("stage_type") != self.STAGE_TYPE:
            return self.load_parent(path, platform_model_id=platform_model_id)
        if contracts.get("p3_standard_joint") != p3_contract.contract():
            if str(self.config.get("load_mode", "")) in P3_CONTRACT_WARM_START_MODES:
                return self._load_p3_stair_memory_warm_start(
                    raw, path, platform_model_id=platform_model_id
                )
            raise ValueError("P3 exact resume contract mismatch")
        mode = self.high_algorithm.load_p3_exact_bundle(
            raw, path, platform_model_id=platform_model_id
        )
        self._restore_low_state(raw, exact=True)
        global_state = (raw.get("training_states") or {}).get("global", {})
        self.session_effective_seconds = float(
            global_state.get("session_effective_seconds", 0.0)
        )
        _validate_resume_phase(raw, global_state, self.session_effective_seconds)
        lifetime = float(
            global_state.get(
                "lifetime_effective_seconds", self.session_effective_seconds
            )
        )
        self.lifetime_base_seconds = max(
            0.0, lifetime - self.session_effective_seconds
        )
        self.current_iteration = int(global_state.get("current_iteration", 0))
        self.high_updates = int(global_state.get("high_updates", 0))
        if self.high_updates != 0:
            raise RuntimeError("P3 stair-memory exact resume requires high_updates=0")
        runtime_m3 = global_state.get("runtime_m3_contract")
        if not isinstance(runtime_m3, dict):
            raise KeyError("P3 exact resume missing runtime M3 contract")
        required = (
            "platform_complete_radius_m",
            "m3_target_radius_m",
            "platform_boundary_radius_m",
        )
        values = [float(runtime_m3.get(name, float("nan"))) for name in required]
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("P3 exact resume runtime M3 contract is invalid")
        (
            self.platform_complete_radius_m,
            self.m3_target_radius_m,
            self.platform_boundary_radius_m,
        ) = values
        current_digest = self.high_algorithm._module_digest(
            (
                ("vision", self.high_algorithm.low_level_encoder),
                ("actor", self.high_algorithm.low_level_actor),
            )
        )
        saved_buffer = (
            ((raw.get("training_states") or {}).get("response_adapter") or {}).get(
                "buffer", {}
            )
        )
        if not isinstance(saved_buffer, dict):
            raise KeyError("P3 exact resume missing Adapter completed-record state")
        if str(saved_buffer.get("low_level_digest", "")) != current_digest:
            raise RuntimeError(
                "P3 exact resume Adapter records reference a different low-level digest"
            )
        if int(saved_buffer.get("low_level_iteration", -1)) != self.low_updates:
            raise RuntimeError(
                "P3 exact resume Adapter records reference a different low-level version"
            )
        self.high_algorithm.low_level_state_digest = current_digest
        self.high_algorithm.response_buffer.set_low_level_version(
            current_digest, self.low_updates
        )
        self._validate_anchor_cnn_reuse()
        self.parent_loaded = True
        self._capture_high_frozen_digest()
        saved_high_digest = str(global_state.get("high_level_frozen_digest", ""))
        if not saved_high_digest or saved_high_digest != self._high_frozen_digest:
            raise RuntimeError("P3 stair-memory exact resume high-level digest mismatch")
        self.high_algorithm.reset_live_state()
        self.low_algorithm.initialize_recurrent_states(self.high_algorithm.num_envs)
        self._apply_phase()
        return f"p3_exact_resume:{mode}"

    def save_training_bundle(self, path, *, platform_model_id):
        high_digest = self._current_high_frozen_digest()
        if self._high_frozen_digest is not None and high_digest != self._high_frozen_digest:
            raise RuntimeError("P3 stair-memory high-level frozen digest drifted")
        current_low_digest = self.high_algorithm._module_digest(
            (
                ("vision", self.high_algorithm.low_level_encoder),
                ("actor", self.high_algorithm.low_level_actor),
            )
        )
        if current_low_digest != self.high_algorithm.low_level_state_digest:
            self.high_algorithm.low_level_state_digest = current_low_digest
            self.high_algorithm.response_buffer.set_low_level_version(
                current_low_digest, self.low_updates
            )
        high_tmp = f"{path}.{uuid4().hex}.high.tmp"
        self.high_algorithm.save_training_bundle(high_tmp, platform_model_id=platform_model_id)
        payload = torch.load(high_tmp, weights_only=False, map_location="cpu")
        os.remove(high_tmp)
        payload["stage_type"] = self.STAGE_TYPE
        payload["phase_label"] = self.current_phase
        payload["modules"]["low_level"]["p3_critic"] = {
            "class_name": "VisualCritic",
            "spec": {"input_dim": 323, "hidden_dims": [512, 256, 128]},
            "state_dict": self.low_algorithm.actor_critic.critic.state_dict(),
        }
        anchor_digest = self.high_algorithm._module_digest(
            (
                ("anchor_encoder", self.low_algorithm.anchor_encoder),
                ("anchor_actor", self.low_algorithm.anchor_actor),
            )
        )
        payload["modules"]["low_level"]["p3_anchor"] = {
            "class_name": "P3ImmutableLowLevelAnchor",
            "spec": {"encoder": "VisionEncoder", "actor": "Actor77Sequential"},
            "encoder_state_dict": self.low_algorithm.anchor_encoder.state_dict(),
            "actor_state_dict": self.low_algorithm.anchor_actor.state_dict(),
            "digest": anchor_digest,
        }
        action_distribution = payload["modules"]["low_level"].get(
            "action_distribution"
        )
        if not isinstance(action_distribution, dict):
            raise KeyError("P3 save missing inherited low action distribution leaf")
        action_distribution_state = action_distribution.get("state_dict")
        if not isinstance(action_distribution_state, dict):
            raise KeyError("P3 save low action distribution is missing state_dict")
        current_std = self.low_algorithm.actor_critic.std.detach().cpu()
        action_distribution_state["std"] = current_std
        # Transitional compatibility for early P3 development checkpoints.
        action_distribution["std"] = current_std
        payload["optimizers"]["low_level"] = self.low_algorithm.optimizer.state_dict()
        realized_dr = p3_contract.materialize_environment_config(
            {"p3_standard_joint": self.config}, self.session_effective_seconds
        )
        payload["training_states"]["global"].update({
            "compound_schedule_phase": self.current_phase,
            "train_scope": "low_level_and_response_adapter",
            "session_effective_seconds": self.session_effective_seconds,
            "lifetime_effective_seconds": self.lifetime_base_seconds + self.session_effective_seconds,
            "current_iteration": self.current_iteration,
            "low_updates": self.low_updates,
            "high_updates": self.high_updates,
            "high_level_frozen_digest": high_digest,
            "p3_anchor_digest": anchor_digest,
            "low_level_version": self.low_updates,
            "domain_randomization_phase": p3_contract.domain_randomization_index(
                self.session_effective_seconds
            ),
            "domain_randomization_realized": {
                "domain_rand": realized_dr.get("domain_rand", {}),
                "noise": realized_dr.get("noise", {}),
                "p3_runtime": realized_dr.get("p3_runtime", {}),
            },
            "runtime_m3_contract": {
                "terrain_size_x_m": 2.0 * self.platform_boundary_radius_m,
                "platform_complete_radius_m": self.platform_complete_radius_m,
                "m3_target_radius_m": self.m3_target_radius_m,
                "platform_boundary_radius_m": self.platform_boundary_radius_m,
            },
            "worker_command_sampler_resume": "seeded_fresh_after_environment_reset",
            "p35_diagnostic_stop_reason": getattr(
                self, "p35_diagnostic_stop_reason", None
            ),
        })
        payload["training_states"]["low_level"].update({
            "frozen": not p3_contract.phase_for_elapsed(self.session_effective_seconds).low_level_trainable,
            "gradient_steps": self.low_updates,
            "current_iteration": self.low_algorithm.current_iteration,
            "rng_state": self.low_algorithm._capture_rng_state(),
            "mirror_rng_state": self.mirror_aux.generator.get_state(),
            "mirror_training_fraction": p3_contract.mirror_training_fraction(
                self.session_effective_seconds
            ),
            "depth_fault": self.depth_fault.state_dict(),
            "memory_auxiliary": self.memory_aux.state_dict(),
            "camera_timing": self.camera_timing.state_dict(),
            "memory_training_fraction": p3_contract.memory_training_fraction(
                self.session_effective_seconds
            ),
            "p35_reward_baseline": self.low_reward_shaper.state_dict(),
            "action_smooth_training_fraction": (
                p3_contract.action_smooth_training_fraction(
                    self.session_effective_seconds,
                    mapping_valid=self.mirror_mapping_valid,
                )
            ),
            "gait_baseline": self.gait_baseline.state_dict(),
        })
        payload["contracts"]["p3_standard_joint"] = p3_contract.contract()
        payload["deployable"] = False
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        tmp = os.path.join(directory, f".{os.path.basename(path)}.{uuid4().hex}.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def memory_metrics(self):
        return self.high_algorithm.memory_metrics()
