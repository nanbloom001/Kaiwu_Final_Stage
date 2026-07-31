#!/usr/bin/env python3
"""P3 Standard joint-recovery algorithm coordinator."""

from __future__ import annotations
import hashlib
import os
from uuid import uuid4
import torch
from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
from agent_ppo.checkpoint_io import normalize_kaiwu_train_bundle
from agent_ppo.feature import p2_contract, p3_contract
from agent_ppo.model.p2_high_level import (
    navigation_safety_head_spec,
)


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

    def _override_reward_components(self, components, **_context):
        # P3 Standard keeps task progress and local-goal events as the high-
        # level objective. Track-specific safety, gait, collision, tracking and
        # stagnation terms remain diagnostics and must not train this policy.
        for name in (
            "success",
            "timeout",
            "tracking",
            "gait_symmetry",
            "body_collision",
            "predictive_collision_risk",
            "missed_safe_direction",
            "frontier_stagnation",
        ):
            components[name] = torch.zeros_like(components[name])
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
                0.10,
                0.10,
                0.30,
                high_adapt_lr,
                0.006,
            )
        elif phase.name == "highslow":
            nav, actor, critic, adapter_lr, entropy = (
                0.15,
                0.15,
                0.20,
                high_adapt_lr,
                0.005,
            )
        elif phase.name == "adaptercalib":
            nav, actor, critic, adapter_lr, entropy = (
                0.0,
                0.0,
                0.0,
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
        for group in self.actor_optimizer.param_groups:
            name = str(group.get("name", ""))
            if name == "navigation_safety_head":
                group["lr"] = 0.0
            else:
                group["lr"] = self._actor_group_base_lr(group) * (
                    nav if name.startswith("navigation_") else actor
                )
        self.critic_optimizer.param_groups[0]["lr"] = p2_contract.CRITIC_LR * critic
        self.response_optimizer.param_groups[0]["lr"] = adapter_lr
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
        self._low_base_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(self.low_algorithm.optimizer.param_groups)
        }
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
        low_scales = {
            "lowbase": {"actor": 0.20, "lstm": 0.20, "critic": 1.00},
            "lowmild": {"actor": 0.50, "lstm": 0.50, "critic": 1.00},
            "lowmedium": {"actor": 1.00, "lstm": 1.00, "critic": 1.00},
            "lowfull": {"actor": 0.50, "lstm": 0.50, "critic": 0.50},
            "adaptercalib": {"actor": 0.00, "lstm": 0.00, "critic": 0.00},
            "highadapt": {"actor": 0.00, "lstm": 0.00, "critic": 0.00},
            "highslow": {"actor": 0.00, "lstm": 0.00, "critic": 0.00},
        }[phase.name]
        for index, group in enumerate(self.low_algorithm.optimizer.param_groups):
            name = str(group.get("name", index))
            group["lr"] = self._low_base_lrs[name] * low_scales.get(name, 0.0)
        self.high_algorithm.update_training_clocks(self.session_effective_seconds)

    def update_clock(self, elapsed_s):
        old = self.current_phase
        self.session_effective_seconds = max(0.0, float(elapsed_s))
        self._apply_phase()
        return self.current_phase != old

    def reset_live_state(self):
        self.episode_subgoal_successes.zero_()
        self.high_algorithm.reset_live_state()
        self.low_algorithm.initialize_recurrent_states(self.high_algorithm.num_envs)

    def observe_subgoal_and_terminal(
        self, worker_aux: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Maintain monitoring-only Standard/subgoal conjunction across rollouts."""
        event = worker_aux[:, p2_contract.CURRENT_SEGMENT_INDEX].round().long()
        reached = event == p3_contract.SUBGOAL_EVENT_REACHED
        self.episode_subgoal_successes += reached.long()
        reset = worker_aux[:, 24] > 0.5
        reason = worker_aux[:, 25].round().long()
        standard_success = reset & (reason == 1)
        joint_success = p3_contract.joint_episode_success(
            standard_success,
            self.episode_subgoal_successes,
            reset & (reason == 2),
        )
        result = {
            "p3_standard_success_count": standard_success.float().sum(),
            "p3_joint_success_count": joint_success.float().sum(),
        }
        self.episode_subgoal_successes.masked_fill_(reset, 0)
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
        return self.current_phase in {"lowbase", "lowmild", "lowmedium", "lowfull"}

    def should_collect_high_rollout(self):
        return self.current_phase in {"highadapt", "highslow"}

    def _restore_low_state(self, raw, *, exact: bool) -> None:
        low = ((raw.get("modules") or {}).get("low_level") or {})
        critic_name = "p3_critic" if exact else "critic"
        critic_leaf = low.get(critic_name) if isinstance(low, dict) else None
        critic_state = critic_leaf.get("state_dict") if isinstance(critic_leaf, dict) else None
        if isinstance(critic_state, dict):
            try:
                self.low_algorithm.actor_critic.critic.load_state_dict(critic_state, strict=True)
            except (RuntimeError, ValueError) as exc:
                if self.logger:
                    self.logger.warning(f"[P3] incompatible parent low critic; rebuilt: {exc}")
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
        self._apply_phase()
        return f"p3_joint_warm_start:{mode}"

    def load_checkpoint(self, path, *, platform_model_id):
        raw = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(raw, dict):
            raise ValueError("P3 checkpoint must be a mapping")
        if raw.get("stage_type") != self.STAGE_TYPE:
            return self.load_parent(path, platform_model_id=platform_model_id)
        contracts = raw.get("contracts") or {}
        if contracts.get("p3_standard_joint") != p3_contract.contract():
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
        self.high_algorithm.reset_live_state()
        self.low_algorithm.initialize_recurrent_states(self.high_algorithm.num_envs)
        self._apply_phase()
        return f"p3_exact_resume:{mode}"

    def save_training_bundle(self, path, *, platform_model_id):
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
            "train_scope": "p3_standard_joint",
            "session_effective_seconds": self.session_effective_seconds,
            "lifetime_effective_seconds": self.lifetime_base_seconds + self.session_effective_seconds,
            "current_iteration": self.current_iteration,
            "low_updates": self.low_updates,
            "high_updates": self.high_updates,
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
        })
        payload["training_states"]["low_level"].update({
            "frozen": not p3_contract.phase_for_elapsed(self.session_effective_seconds).low_level_trainable,
            "gradient_steps": self.low_updates,
            "current_iteration": self.low_algorithm.current_iteration,
            "rng_state": self.low_algorithm._capture_rng_state(),
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
