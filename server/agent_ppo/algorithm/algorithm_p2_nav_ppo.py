#!/usr/bin/env python3
"""Dedicated continuous recurrent high-level PPO for P2 Track training."""

from __future__ import annotations

import hashlib
import os
import time
from uuid import uuid4

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.checkpoint_io import (
    KAIWU_TRAIN_FORMAT,
    KAIWU_TRAIN_SCHEMA_V2,
    normalize_kaiwu_train_bundle,
    validate_p3_eval_bundle,
    validate_state_dict_finite,
)
from agent_ppo.feature import nav_contract, p15_contract, p2_contract
from agent_ppo.feature.feedback_emulator import feedback_implementation_digest
from agent_ppo.feature.p2_command_controller import P2CommandController
from agent_ppo.feature.p2_curriculum_probe import P2CurriculumAccumulator
from agent_ppo.feature.p2_gait import P2GaitBaseline
from agent_ppo.feature.p2_response_buffer import (
    patch_owned_commands,
    split_p2_transport,
)
from agent_ppo.feature.p2_rollout import P2RolloutStorage
from agent_ppo.feature.response_aux_buffer import build_response_observation
from agent_ppo.model.p2_high_level import (
    assemble_actor_input,
    assemble_critic_input,
    navigation_actor_spec,
    navigation_critic_spec,
    navigation_encoder_spec,
    navigation_safety_head_spec,
)
from agent_ppo.model.response_adapter import response_adapter_spec


class AlgorithmP2NavPPO:
    STAGE_TYPE = "p2_nav_ppo"

    @staticmethod
    def _adapter_replay_counts(
        metadata: dict,
    ) -> tuple[float, float, float, float, float]:
        """Return mutually exclusive replay-pool counts and their total."""
        if "p4_current_batch_envs" in metadata:
            latest = float(metadata.get("p4_current_batch_envs", 0.0))
            recent = float(metadata.get("p35_parent_batch_envs", 0.0))
            parent = float(metadata.get("earlier_lineage_batch_envs", 0.0))
            track = latest + recent
            return latest, recent, parent, track, track + parent
        if "latest_batch_envs" in metadata or "recent_batch_envs" in metadata:
            latest = float(metadata.get("latest_batch_envs", 0.0))
            recent = float(metadata.get("recent_batch_envs", 0.0))
            parent = float(metadata.get("parent_batch_envs", 0.0))
            track = latest + recent
            return latest, recent, parent, track, track + parent
        latest = float(metadata.get("latest_batch_envs", 0.0))
        recent = float(metadata.get("recent_batch_envs", 0.0))
        track = float(metadata.get("track_batch_envs", latest + recent))
        parent = float(metadata.get("parent_batch_envs", 0.0))
        return latest, recent, parent, track, track + parent

    def __init__(
        self,
        *,
        low_level_encoder: nn.Module,
        low_level_actor: nn.Module,
        navigation_encoder: nn.Module,
        safety_head: nn.Module | None,
        actor: nn.Module,
        critic: nn.Module | None,
        response_adapter: nn.Module,
        response_buffer,
        num_envs: int,
        device="cuda:0",
        config=None,
        logger=None,
        monitor=None,
        training: bool = True,
    ):
        self.device = torch.device(device)
        self.logger = logger
        self.monitor = monitor
        self.config = dict(config or {})
        self.nav_period_frames = int(
            self.config.get("nav_period_frames", p2_contract.NAV_PERIOD_FRAMES)
        )
        self.nav_rollout_ticks = int(
            self.config.get("nav_rollout_ticks", p2_contract.NAV_ROLLOUT_TICKS)
        )
        self.tbptt_sequence_length = int(
            self.config.get(
                "tbptt_sequence_length", p2_contract.TBPTT_SEQUENCE_LENGTH
            )
        )
        if self.nav_period_frames <= 0:
            raise ValueError("nav_period_frames must be positive")
        if self.nav_rollout_ticks <= 0 or self.tbptt_sequence_length <= 0:
            raise ValueError("navigation rollout and TBPTT lengths must be positive")
        if self.nav_rollout_ticks % self.tbptt_sequence_length:
            raise ValueError("navigation rollout ticks must be divisible by TBPTT length")
        self.nav_dt_s = p2_contract.CONTROL_DT_S * self.nav_period_frames
        self.load_mode = str(self.config.get("load_mode", "auto"))
        self.training_enabled = bool(training)
        self.num_envs = int(num_envs)
        self.low_level_encoder = low_level_encoder.to(self.device)
        self.low_level_actor = low_level_actor.to(self.device)
        self.navigation_encoder = navigation_encoder.to(self.device)
        self.safety_head = safety_head.to(self.device) if safety_head is not None else None
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device) if critic is not None else None
        self.response_adapter = response_adapter.to(self.device)
        self.response_buffer = response_buffer
        for module in (self.low_level_encoder, self.low_level_actor):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.actor_optimizer = None
        self.critic_optimizer = None
        self.response_optimizer = None
        self.actor_scheduler = None
        self.critic_scheduler = None
        self.response_scheduler = None
        if self.training_enabled:
            if self.critic is None or self.response_buffer is None or self.safety_head is None:
                raise ValueError("P2 training requires critic, response buffer, and SafetyHead")
            actor_groups = []
            for name, parameters in self.navigation_encoder.parameter_groups().items():
                base_lr = p2_contract.CNN_LAYER_LRS[name]
                actor_groups.append(
                    {
                        "params": list(parameters),
                        "lr": base_lr,
                        "base_lr": base_lr,
                        "name": f"navigation_{name}",
                    }
                )
            actor_groups.extend(
                (
                {
                    "params": list(self.actor.memory.parameters()),
                    "lr": p2_contract.ACTOR_LR,
                    "base_lr": p2_contract.ACTOR_LR,
                    "name": "actor_trunk",
                },
                {
                    "params": [
                        *self.actor.mean_head.parameters(),
                        self.actor.log_std,
                    ],
                    "lr": p2_contract.ACTOR_LR,
                    "base_lr": p2_contract.ACTOR_LR,
                    "name": "actor_main",
                },
                {
                    "params": [
                        *self.actor.vy_mean_head.parameters(),
                        self.actor.vy_log_std,
                    ],
                    "lr": p2_contract.ACTOR_LR,
                    "base_lr": p2_contract.ACTOR_LR,
                    "name": "actor_vy",
                },
                )
            )
            actor_groups.append(
                {
                    "params": list(self.safety_head.parameters()),
                    "lr": p2_contract.SAFETY_HEAD_LR,
                    "base_lr": p2_contract.SAFETY_HEAD_LR,
                    "name": "navigation_safety_head",
                }
            )
            self.actor_optimizer = torch.optim.Adam(actor_groups)
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=p2_contract.CRITIC_LR
            )
            self.response_optimizer = torch.optim.Adam(
                self.response_adapter.parameters(), lr=p2_contract.ADAPTER_LR
            )
            self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.actor_optimizer, lambda _step: 1.0
            )
            self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.critic_optimizer, lambda _step: 1.0
            )
            self.response_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.response_optimizer, lambda _step: 1.0
            )
            self._assert_optimizer_isolation()
        else:
            for module in (self.navigation_encoder, self.actor, self.response_adapter):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

        self.command_slew_rate = tuple(
            self.config.get("slew_rate", (0.30, 0.30, 1.00))
        )
        self.command_slew_release_rate = tuple(
            self.config.get("slew_release_rate", (0.30, 0.60, 2.50))
        )
        self.command_transition_mode = str(
            self.config.get("command_transition_mode", "slew")
        )
        self.command = P2CommandController(
            self.num_envs,
            self.device,
            slew_rate=self.command_slew_rate,
            slew_release_rate=self.command_slew_release_rate,
            command_transition_mode=self.command_transition_mode,
        )
        self.rollout = (
            P2RolloutStorage(
                self.num_envs,
                num_ticks=self.nav_rollout_ticks,
                sequence_length=self.tbptt_sequence_length,
                store_depth=True,
                pin_memory=True,
            )
            if self.training_enabled
            else None
        )
        self.actor_hidden = None
        self.critic_hidden = None
        self.adapter_hidden = None
        self.reset_since_tick = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.frame_count = 0
        self.nav_ticks = 0
        self.current_iteration = 0
        self.actor_gradient_steps = 0
        self.critic_gradient_steps = 0
        self.adapter_gradient_steps = 0
        self.skipped_nonfinite = 0
        # Keep the legacy name as the current-session clock because workflow
        # stop/save scheduling already consumes it. Lifetime is persisted
        # independently so the added session does not erase lineage.
        self.effective_training_seconds = 0.0
        self.session_effective_seconds = 0.0
        self.lifetime_effective_seconds = 0.0
        self.lifetime_base_seconds = 0.0
        self.cnn_unfrozen = False
        self.loaded_platform_model_id = None
        self.source_parent_model_id = None
        self.parent_checkpoint_sha256 = None
        self.low_level_state_digest = None
        self.low_level_payload = {}
        self.parent_optimizer_payload = {}
        self.parent_scheduler_payload = {}
        self.parent_training_payload = {}
        self.resume_loaded = False
        self.pending_tick = None
        self.rollout_invalid = False
        self.invalid_transition_count = 0
        self.last_tick_penalties = {}
        self.last_tick_diagnostics = {}
        self.adapter_generator = torch.Generator(device="cpu")
        self.adapter_generator.manual_seed(int(self.config.get("adapter_seed", 2501)))
        self.neutral_generator = torch.Generator(device=self.device)
        self.neutral_generator.manual_seed(int(self.config.get("neutral_seed", 2503)))
        self.ppo_generator = torch.Generator(device="cpu")
        self.ppo_generator.manual_seed(int(self.config.get("ppo_seed", 2502)))
        self.action_generator = torch.Generator(device=self.device)
        self.action_generator.manual_seed(int(self.config.get("action_seed", 2504)))
        self.vy_action_generator = torch.Generator(device=self.device)
        self.vy_action_generator.manual_seed(
            int(self.config.get("vy_action_seed", 2505))
        )
        self.optimizer_migration_report = {}
        self.micro_sequences = int(self.config.get("cnn_microbatch_sequences", 4))
        self.num_learning_epochs = int(self.config.get("num_learning_epochs", 4))
        if self.num_learning_epochs <= 0:
            raise ValueError("P2 num_learning_epochs must be positive")
        self.num_mini_batches = int(self.config.get("num_mini_batches", 4))
        self.adapter_batch_envs = int(
            (self.config.get("response_adapter") or {}).get("batch_envs", 64)
        )
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.stuck_positive_ema = 0.1
        self.curriculum_probe = P2CurriculumAccumulator(
            curriculum_enabled=bool(self.config.get("terrain_curriculum", False))
        )
        self.gait_baseline = P2GaitBaseline()
        self.best_goal_distance = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        self.episode_start_goal_distance = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        self._initialize_navigation_reward_state()
        self.entropy_coefficient = 0.020
        self.optimizer_phase = "navwarm"
        self.current_vy_trusted_limit = 0.20
        self.current_vy_hard_limit = 0.40
        self.return_statistics = {
            "count": 0,
            "mean": 0.0,
            "m2": 0.0,
            "value_normalization_enabled": False,
        }
        self._h2d_time_s = 0.0
        self.nonfinite_action_fallbacks = 0
        self.adapter_oom_skips = 0
        if self.training_enabled:
            self._apply_training_schedule(0.0)

    def _initialize_navigation_reward_state(self) -> None:
        self.frontier_history = torch.full(
            (p2_contract.FRONTIER_WINDOW_TICKS, self.num_envs),
            float("inf"),
            device=self.device,
        )
        self.frontier_history_index = 0
        self.frontier_stagnation_age = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.previous_body_collision = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _ensure_navigation_reward_state(self) -> None:
        if not hasattr(self, "frontier_history"):
            self._initialize_navigation_reward_state()
        if not hasattr(self, "best_goal_distance"):
            self.best_goal_distance = torch.full(
                (self.num_envs,), float("inf"), device=self.device
            )
        if not hasattr(self, "episode_start_goal_distance"):
            self.episode_start_goal_distance = torch.full(
                (self.num_envs,), float("inf"), device=self.device
            )

    def _reset_navigation_reward_state(self, reset: torch.Tensor) -> None:
        self._ensure_navigation_reward_state()
        reset = reset.to(self.device).reshape(-1).bool()
        if not bool(reset.any()):
            return
        self.frontier_history[:, reset] = float("inf")
        self.frontier_stagnation_age[reset] = 0
        self.previous_body_collision[reset] = False

    def _frontier_rewards(
        self,
        *,
        best_before: torch.Tensor,
        end_goal: torch.Tensor,
        terminal: torch.Tensor,
    ) -> torch.Tensor:
        """Settle the monotonic-frontier stagnation penalty."""
        self._ensure_navigation_reward_state()
        terminal = terminal.reshape(-1).bool()
        current_best = torch.minimum(best_before, end_goal)
        history_reference = self.frontier_history[
            self.frontier_history_index
        ].clone()
        history_valid = torch.isfinite(history_reference)
        frontier_advance = history_reference - current_best
        previous_age = self.frontier_stagnation_age.clone()

        stagnating = (
            history_valid
            & (frontier_advance < p2_contract.FRONTIER_MIN_ADVANCE_M)
            & (end_goal > 0.8)
            & ~terminal
        )
        self.frontier_stagnation_age = torch.where(
            stagnating,
            previous_age + 1,
            torch.zeros_like(previous_age),
        )
        stagnation_penalty = p2_contract.frontier_stagnation_penalty(
            self.frontier_stagnation_age
        )
        write_value = torch.where(
            terminal,
            torch.full_like(current_best, float("inf")),
            current_best,
        )
        self.frontier_history[self.frontier_history_index] = write_value
        self.frontier_history_index = (
            self.frontier_history_index + 1
        ) % p2_contract.FRONTIER_WINDOW_TICKS
        self._reset_navigation_reward_state(terminal)
        return stagnation_penalty

    def _assert_optimizer_isolation(self) -> None:
        sets = {
            "actor": {id(p) for g in self.actor_optimizer.param_groups for p in g["params"]},
            "critic": {id(p) for g in self.critic_optimizer.param_groups for p in g["params"]},
            "adapter": {id(p) for g in self.response_optimizer.param_groups for p in g["params"]},
            "low": {id(p) for m in (self.low_level_encoder, self.low_level_actor) for p in m.parameters()},
        }
        for left, right in (("actor", "critic"), ("actor", "adapter"), ("critic", "adapter")):
            if sets[left] & sets[right]:
                raise RuntimeError(f"P2 optimizer parameter overlap: {left}/{right}")
        if (sets["actor"] | sets["critic"] | sets["adapter"]) & sets["low"]:
            raise RuntimeError("P2 frozen low-level parameter leaked into optimizer")

    @staticmethod
    def _module_digest(modules) -> str:
        digest = hashlib.sha256()
        for prefix, module in modules:
            for name, value in sorted(module.state_dict().items()):
                digest.update(f"{prefix}.{name}".encode("ascii"))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _low_encoder_spec(module: nn.Module) -> dict[str, object]:
        return {
            "image_shape": list(getattr(module, "image_shape", (180, 320, 1))),
            "proprio_dim": int(getattr(module, "proprio_dim", 45)),
            "cnn_output_dim": int(getattr(module, "cnn_output_dim", 32)),
            "rnn_hidden_dim": int(getattr(module, "rnn_hidden_dim", 64)),
            "rnn_num_layers": int(getattr(module, "rnn_num_layers", 2)),
            "rnn_output_dim": int(getattr(module, "rnn_output_dim", 32)),
            "use_lstm": bool(getattr(module, "use_lstm", True)),
        }

    @staticmethod
    def _low_actor_spec() -> dict[str, object]:
        return {
            "input_dim": 77,
            "hidden_dims": [512, 256, 128],
            "output_dim": 12,
            "activation": "elu",
        }

    @staticmethod
    def _low_critic_spec() -> dict[str, object]:
        return {
            "input_dim": 316,
            "hidden_dims": [512, 256, 128],
            "output_dim": 1,
            "activation": "elu",
        }

    @staticmethod
    def _action_distribution_spec() -> dict[str, object]:
        return {
            "action_dim": 12,
            "parameterization": "diagonal_std",
        }

    @staticmethod
    def _leaf(class_name: str, spec: dict, state_dict: dict) -> dict:
        return {
            "class_name": class_name,
            "spec": dict(spec),
            "state_dict": state_dict,
        }

    @staticmethod
    def _load_leaf(
        group: dict,
        name: str,
        module: nn.Module,
        *,
        class_name: str,
        spec: dict,
        context: str,
    ) -> None:
        leaf = group.get(name)
        if not isinstance(leaf, dict):
            raise KeyError(f"{context} missing leaf {name}")
        if leaf.get("class_name") != class_name:
            raise ValueError(
                f"{context}.{name} class mismatch: {leaf.get('class_name')!r}"
            )
        if leaf.get("spec") != spec:
            raise ValueError(
                f"{context}.{name} spec mismatch: {leaf.get('spec')!r}"
            )
        state = leaf.get("state_dict")
        if not isinstance(state, dict):
            raise KeyError(f"{context}.{name} missing state_dict")
        validate_state_dict_finite(state, f"{context}.{name}")
        module.load_state_dict(state, strict=True)

    @staticmethod
    def _validate_opaque_leaf(
        leaf: dict,
        *,
        class_name: str,
        spec: dict,
        context: str,
    ) -> None:
        if not isinstance(leaf, dict):
            raise KeyError(f"{context} missing leaf")
        if leaf.get("class_name") != class_name or leaf.get("spec") != spec:
            raise ValueError(f"{context} class/spec mismatch")
        state = leaf.get("state_dict")
        if not isinstance(state, dict):
            raise KeyError(f"{context} missing state_dict")
        validate_state_dict_finite(state, context)

    def _migrate_parent_low_aux_modules(self, modules: dict, low: dict) -> None:
        critic = low.get("critic")
        if isinstance(critic, dict) and isinstance(critic.get("state_dict"), dict):
            low["critic"] = self._leaf(
                "VisualCritic",
                self._low_critic_spec(),
                critic["state_dict"],
            )
        action_distribution = modules.get("action_distribution")
        if "action_distribution" not in low and isinstance(action_distribution, dict):
            std = action_distribution.get("std")
            if torch.is_tensor(std):
                if tuple(std.shape) != (12,):
                    raise ValueError(
                        "P2 parent action distribution std must have shape [12]"
                    )
                low["action_distribution"] = self._leaf(
                    "DiagonalGaussianActionDistribution",
                    self._action_distribution_spec(),
                    {"std": std},
                )
        anchor = modules.get("s0_anchor")
        if "s0_anchor" not in low and isinstance(anchor, dict):
            encoder_state = anchor.get("vision_encoder_state_dict")
            actor_state = anchor.get("actor_state_dict")
            if isinstance(encoder_state, dict) and isinstance(actor_state, dict):
                low["s0_anchor"] = {
                    "contract_version": "low_level_s0_anchor_v1",
                    "locomotion_encoder": self._leaf(
                        "VisionEncoder",
                        self._low_encoder_spec(self.low_level_encoder),
                        encoder_state,
                    ),
                    "actor": self._leaf(
                        "Actor77Sequential",
                        self._low_actor_spec(),
                        actor_state,
                    ),
                }

    def _validate_low_aux_modules(self, low: dict, *, context: str) -> None:
        critic = low.get("critic")
        if critic is not None:
            self._validate_opaque_leaf(
                critic,
                class_name="VisualCritic",
                spec=self._low_critic_spec(),
                context=f"{context}.critic",
            )
        action_distribution = low.get("action_distribution")
        if action_distribution is not None:
            self._validate_opaque_leaf(
                action_distribution,
                class_name="DiagonalGaussianActionDistribution",
                spec=self._action_distribution_spec(),
                context=f"{context}.action_distribution",
            )
            std = action_distribution["state_dict"].get("std")
            if not torch.is_tensor(std) or tuple(std.shape) != (12,):
                raise ValueError(f"{context}.action_distribution std shape mismatch")
        anchor = low.get("s0_anchor")
        if anchor is not None:
            if (
                not isinstance(anchor, dict)
                or anchor.get("contract_version") != "low_level_s0_anchor_v1"
            ):
                raise ValueError(f"{context}.s0_anchor contract mismatch")
            self._validate_opaque_leaf(
                anchor.get("locomotion_encoder"),
                class_name="VisionEncoder",
                spec=self._low_encoder_spec(self.low_level_encoder),
                context=f"{context}.s0_anchor.locomotion_encoder",
            )
            self._validate_opaque_leaf(
                anchor.get("actor"),
                class_name="Actor77Sequential",
                spec=self._low_actor_spec(),
                context=f"{context}.s0_anchor.actor",
            )

    def _warn_platform_identity(self, bundle: dict, requested_id) -> None:
        payload_id = bundle.get("platform_model_id")
        if payload_id is not None and str(payload_id) != str(requested_id) and self.logger:
            # Platform selection is filename/request-ID owned. Keep historical
            # payload IDs diagnostic-only per the repository identity policy.
            self.logger.warning(
                "[P2NavPPO] requested ID differs from payload platform_model_id; "
                f"requested={requested_id} payload={payload_id}"
            )

    @staticmethod
    def _bundle_identity(bundle: dict, requested_id, path: str | None = None) -> str:
        """Return provenance identity without turning metadata into a gate."""
        lineage = bundle.get("lineage", {})
        for value in (
            bundle.get("platform_model_id"),
            lineage.get("platform_model_id") if isinstance(lineage, dict) else None,
        ):
            if value is not None:
                return str(value)
        if path:
            filename_id = os.path.basename(path).rsplit("-", 1)[-1].split(".", 1)[0]
            if filename_id.isdigit():
                return filename_id
        return str(requested_id)

    @staticmethod
    def _finite_rows(*values: torch.Tensor) -> torch.Tensor:
        result = None
        for value in values:
            current = torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1)
            result = current if result is None else result & current
        return result

    @staticmethod
    def _zero_invalid_rows(value: torch.Tensor, invalid: torch.Tensor) -> torch.Tensor:
        sanitized = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.where(
            invalid.reshape(-1, *([1] * (sanitized.ndim - 1))),
            torch.zeros_like(sanitized),
            sanitized,
        )

    @staticmethod
    def _sanitize_hidden_rows(hidden, invalid: torch.Tensor):
        if hidden is None:
            return None
        mask = invalid.reshape(1, -1, 1)
        return tuple(
            torch.where(
                mask,
                torch.zeros_like(value),
                torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
            )
            for value in hidden
        )

    @staticmethod
    def _hidden_or_zeros(hidden, layers: int, batch: int, size: int, device):
        if hidden is None:
            return (
                torch.zeros(layers, batch, size, device=device),
                torch.zeros(layers, batch, size, device=device),
            )
        return hidden

    @staticmethod
    def _mask_hidden(hidden, reset_mask):
        if hidden is None:
            return None
        reset = reset_mask.reshape(1, -1, 1)
        return (
            torch.where(reset, torch.zeros_like(hidden[0]), hidden[0]),
            torch.where(reset, torch.zeros_like(hidden[1]), hidden[1]),
        )

    def _split_policy(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if obs.shape[1] != nav_contract.POLICY_OBS_DIM:
            raise ValueError(f"P2 policy obs must be 57905, got {tuple(obs.shape)}")
        return {
            "proprio": obs[:, :45],
            "height_scan": obs[:, 45:301],
            "goal4": obs[:, 301:305],
            "depth": obs[:, 305:].reshape(
                obs.shape[0], p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
            ),
        }

    def _low_level_frame(self, parts, critic_obs):
        """Run one frozen low-level frame; P3 overrides this for joint capture."""
        del critic_obs
        with torch.inference_mode():
            latent = self.low_level_encoder(parts["depth"], parts["proprio"], masks=None)
            action = self.low_level_actor(
                torch.cat((parts["proprio"], latent), dim=-1)
            )
        return action, {}

    def _nav_capability(self, batch: int, *, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(
            p2_contract.NAV_CAPABILITY_PROFILE15,
            device=self.device,
            dtype=dtype,
        ).expand(batch, -1)

    def _response_capability(self, batch: int, *, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(
            p2_contract.RESPONSE_CAPABILITY_PROFILE15,
            device=self.device,
            dtype=dtype,
        ).expand(batch, -1)

    def _nav_nonvisual(self, goal4: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        capability = self._nav_capability(aux.shape[0], dtype=aux.dtype)
        result = torch.cat(
            (
                goal4,
                aux[:, 0:3],
                aux[:, 3:6],
                aux[:, 6:9],
                aux[:, 9:10],
                aux[:, 10:11],
                aux[:, 18:21],
                aux[:, 21:24],
                capability,
            ),
            dim=-1,
        )
        if result.shape[1] != p2_contract.NAV_NONVISUAL_DIM:
            raise AssertionError("P2 nav_nonvisual36 layout drift")
        return result

    def _split_transport(self, critic_wire: torch.Tensor):
        """Stage hook for a versioned training-only privileged tail."""
        return split_p2_transport(critic_wire)

    def _prepare_policy_parts(self, parts, critic_obs, aux, reset):
        del critic_obs, aux, reset
        return parts

    def _map_policy_target(
        self,
        normalized: torch.Tensor,
        legacy_target: torch.Tensor,
        *,
        goal4: torch.Tensor,
        aux: torch.Tensor,
    ) -> torch.Tensor:
        del normalized, goal4, aux
        return legacy_target

    def _predictive_command(self, target: torch.Tensor) -> torch.Tensor:
        return target

    def _command_rate_weight(self) -> float:
        return float(p2_contract.COMMAND_RATE_WEIGHT)

    def _tracking_error_weight(self) -> float:
        return float(p2_contract.TRACKING_ERROR_WEIGHT)

    def _response_append_kwargs(self) -> dict[str, object]:
        return {}

    def _transition_extras(self) -> dict[str, torch.Tensor]:
        return {}

    def _update_policy_auxiliary_target(
        self, *, parts, nav_feat, nav_nonvisual, profile, confidence, reset
    ) -> None:
        del parts, nav_feat, nav_nonvisual, profile, confidence, reset

    def _extra_tick_diagnostics(self) -> dict[str, torch.Tensor]:
        return {}

    def _actor_auxiliary_loss(
        self,
        *,
        normalized_mean: torch.Tensor,
        batch: dict[str, torch.Tensor],
        ppo_actor_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del batch, ppo_actor_loss
        return normalized_mean.new_zeros(()), {}

    def _actor_auxiliary_metric_names(self) -> tuple[str, ...]:
        """Scalar auxiliary metrics that must survive PPO aggregation."""
        return ()

    def _apply_external_reset(self, aux: torch.Tensor) -> None:
        reset = aux[:, 24] > 0.5
        if not bool(reset.any()):
            return
        ids = reset.nonzero(as_tuple=False).reshape(-1)
        self.low_level_encoder.reset_hidden_state_for_envs(ids)
        self.actor_hidden = self._mask_hidden(self.actor_hidden, reset)
        self.critic_hidden = self._mask_hidden(self.critic_hidden, reset)
        if self.adapter_hidden is not None:
            self.adapter_hidden[:, ids, :] = 0.0
        self.command.reset(ids)
        if self.pending_tick is None:
            self.best_goal_distance[reset] = float("inf")
            self.episode_start_goal_distance[reset] = float("inf")
            self._reset_navigation_reward_state(reset)
        self.reset_since_tick |= reset

    @torch.no_grad()
    def frame_begin(self, obs: torch.Tensor, critic_wire: torch.Tensor, *, deterministic=False):
        critic_obs, aux = self._split_transport(critic_wire)
        self.curriculum_probe.observe(
            aux,
            logger=self.logger,
            elapsed_s=self.session_effective_seconds,
        )
        self._apply_external_reset(aux)
        aux = patch_owned_commands(
            aux, self.command.active_target, self.command.exec_cmd, self.command.command_epoch
        )
        critic_obs = critic_obs.clone()
        parts = self._split_policy(obs)
        parts["proprio"] = parts["proprio"].clone()
        parts = self._prepare_policy_parts(
            parts, critic_obs, aux, self.reset_since_tick.clone()
        )
        p0, p1 = nav_contract.POLICY_CMD_SLICE
        c0, c1 = nav_contract.CRITIC_CMD_SLICE
        parts["proprio"][:, p0:p1] = self.command.exec_cmd.to(parts["proprio"])
        critic_obs[:, c0:c1] = self.command.exec_cmd.to(critic_obs)
        is_tick = False
        tick_penalty = None
        if self.frame_count % self.nav_period_frames == 0:
            reset = self.reset_since_tick.clone()
            self.actor_hidden = self._mask_hidden(self.actor_hidden, reset)
            self.critic_hidden = self._mask_hidden(self.critic_hidden, reset)
            actor_initial = self._hidden_or_zeros(
                self.actor_hidden, self.actor.num_layers, self.num_envs, self.actor.hidden_dim, self.device
            )
            critic_initial = (
                self._hidden_or_zeros(
                    self.critic_hidden,
                    self.critic.num_layers,
                    self.num_envs,
                    self.critic.hidden_dim,
                    self.device,
                )
                if self.training_enabled
                else None
            )
            nav_feat = self.navigation_encoder(parts["depth"])
            response_obs = build_response_observation(
                aux[:, : p2_contract.RESPONSE_AUX_DIM],
                self._response_capability(self.num_envs),
            )
            with torch.inference_mode():
                profile, next_adapter_hidden = self.response_adapter(
                    response_obs,
                    hidden=self.adapter_hidden,
                    reset_mask=reset.reshape(1, -1),
                )
            # Keep recurrent state mutable for per-environment resets performed
            # by frame_end() outside the inference-mode context.
            profile = profile.clone()
            self.adapter_hidden = next_adapter_hidden.clone()
            confidence = p2_contract.adapter_confidence(
                velocity_valid=aux[:, 9:10],
                velocity_age=p2_contract.feedback_age_seconds(
                    aux[:, 10:11],
                    age_clip_s=float(
                        (self.config.get("feedback_profile") or {}).get(
                            "age_clip_s", 0.8
                        )
                    ),
                ),
                velocity_log_sigma=profile[:, 13:16],
                target_cmd3=aux[:, 0:3],
            )
            if not deterministic:
                neutral = torch.rand(
                    self.num_envs, 1, generator=self.neutral_generator, device=self.device
                ) < p2_contract.NEUTRAL_PROFILE_PROBABILITY
                profile = torch.where(neutral, torch.zeros_like(profile), profile)
                confidence = torch.where(neutral, torch.zeros_like(confidence), confidence)
            nav_nonvisual = self._nav_nonvisual(parts["goal4"], aux)
            self._update_policy_auxiliary_target(
                parts=parts,
                nav_feat=nav_feat,
                nav_nonvisual=nav_nonvisual,
                profile=profile,
                confidence=confidence,
                reset=reset,
            )
            actor_input = assemble_actor_input(nav_feat, nav_nonvisual, profile, confidence)
            critic_input = None
            if deterministic:
                target, normalized, mean, log_std, self.actor_hidden = self.actor.deterministic(
                    actor_input,
                    self.actor_hidden,
                    reset,
                    hard_abs_vy=self.current_vy_hard_limit,
                )
                pre_tanh = mean
                log_prob = torch.zeros(self.num_envs, 1, device=self.device)
            else:
                target, pre_tanh, normalized, log_prob, mean, log_std, self.actor_hidden = self.actor.sample(
                    actor_input,
                    self.actor_hidden,
                    reset,
                    generator=self.action_generator,
                    vy_generator=self.vy_action_generator,
                    hard_abs_vy=self.current_vy_hard_limit,
                )
            target = self._map_policy_target(
                normalized,
                target,
                goal4=parts["goal4"],
                aux=aux,
            )
            value = None
            if self.training_enabled:
                if getattr(self, "track_safety_enabled", True):
                    safe3, scanner_available, safety_teacher_diagnostics = (
                        p2_contract.privileged_safe_directions(critic_obs)
                    )
                else:
                    safe3 = torch.zeros(self.num_envs, 3, device=self.device)
                    scanner_available = torch.zeros(
                        self.num_envs, dtype=torch.bool, device=self.device
                    )
                    safety_teacher_diagnostics = {
                        "nav_risk3": torch.zeros_like(safe3),
                        "terrain_passable3": torch.zeros_like(safe3),
                    }
                critic_input = assemble_critic_input(
                    critic_obs,
                    self.command.active_target,
                    self._nav_capability(self.num_envs),
                )
                value, self.critic_hidden = self.critic(
                    critic_input, self.critic_hidden, reset
                )
            finite_values = [
                nav_feat,
                nav_nonvisual,
                profile,
                confidence,
                mean,
                log_std,
                pre_tanh,
                normalized,
                log_prob,
                target,
            ]
            if value is not None:
                finite_values.extend((critic_input, value))
            output_finite = self._finite_rows(*finite_values)
            invalid = ~output_finite
            if bool(invalid.any()):
                invalid_count = int(invalid.sum())
                self.nonfinite_action_fallbacks += invalid_count
                self.invalid_transition_count += invalid_count
                self.rollout_invalid |= self.training_enabled
                target = self._zero_invalid_rows(target, invalid)
                pre_tanh = self._zero_invalid_rows(pre_tanh, invalid)
                log_prob = self._zero_invalid_rows(log_prob, invalid)
                nav_feat = self._zero_invalid_rows(nav_feat, invalid)
                nav_nonvisual = self._zero_invalid_rows(nav_nonvisual, invalid)
                profile = self._zero_invalid_rows(profile, invalid)
                confidence = self._zero_invalid_rows(confidence, invalid)
                parts["depth"] = self._zero_invalid_rows(parts["depth"], invalid)
                self.actor_hidden = self._mask_hidden(self.actor_hidden, invalid)
                self.critic_hidden = self._mask_hidden(self.critic_hidden, invalid)
                actor_initial = self._sanitize_hidden_rows(actor_initial, invalid)
                critic_initial = self._sanitize_hidden_rows(critic_initial, invalid)
                if self.adapter_hidden is not None:
                    self.adapter_hidden[:, invalid, :] = 0.0
                if value is not None:
                    value = self._zero_invalid_rows(value, invalid)
                    critic_input = self._zero_invalid_rows(critic_input, invalid)
                if self.logger and (
                    self.nonfinite_action_fallbacks == invalid_count
                    or self.nonfinite_action_fallbacks % 100 == 0
                ):
                    self.logger.error(
                        "[P2NavPPO] non-finite high-level transition sanitized; "
                        "PPO rollout will be skipped while training continues; "
                        f"fallbacks={self.nonfinite_action_fallbacks}"
                    )
            command_penalty = (
                self._command_rate_weight()
                * p2_contract.normalized_command_rate(
                    target, self.command.active_target
                )
            )
            command_scale = torch.tensor(
                p2_contract.COMMAND_NORMALIZATION,
                device=target.device,
                dtype=target.dtype,
            )
            command_axis_weights = torch.tensor(
                p2_contract.COMMAND_RATE_AXIS_WEIGHTS,
                device=target.device,
                dtype=target.dtype,
            )
            command_delta = torch.clamp(
                (target - self.command.active_target) / command_scale,
                -1.0,
                1.0,
            )
            command_rate_axis_penalty = (
                self._command_rate_weight()
                * command_delta.square()
                * command_axis_weights
            )
            if self.training_enabled:
                if getattr(self, "track_safety_enabled", True):
                    (
                        predictive_collision_penalty,
                        predictive_clearance_m,
                        predictive_stopping_distance_m,
                        predictive_collision_risk,
                        predictive_collision_legacy_risk,
                        predictive_collision_wallness,
                        predictive_collision_sector_risk,
                    ) = p2_contract.predictive_collision_risk_penalty(
                        parts["depth"], self._predictive_command(target)
                    )
                    missed_safe_penalty, missed_safe_diagnostics = (
                        p2_contract.missed_safe_direction_penalty(
                            safe3,
                            target,
                            scanner_available,
                            self.session_effective_seconds,
                        )
                    )
                else:
                    zeros = torch.zeros(self.num_envs, device=self.device)
                    predictive_collision_penalty = zeros
                    predictive_clearance_m = torch.full_like(
                        zeros, p2_contract.PREDICTIVE_COLLISION_MAX_DEPTH_M
                    )
                    predictive_stopping_distance_m = zeros
                    predictive_collision_risk = zeros
                    predictive_collision_legacy_risk = zeros
                    predictive_collision_wallness = torch.zeros(
                        self.num_envs, 3, device=self.device
                    )
                    predictive_collision_sector_risk = torch.zeros_like(
                        predictive_collision_wallness
                    )
                    missed_safe_penalty = zeros
                    missed_safe_diagnostics = {
                        "selected_safe": zeros,
                        "best_safe": zeros,
                        "safe_gap": zeros,
                        "active": zeros,
                        "selection_eligible": zeros,
                        "selected_safest": zeros,
                    }
                self.pending_tick = {
                    # Own the high-level visual sample immediately. Isaac may
                    # reuse its observation buffer on the next low-level step.
                    "depth": P2RolloutStorage.own_depth_sample(parts["depth"]),
                    "nav_feat": nav_feat.detach(),
                    "nav_nonvisual": nav_nonvisual.detach(),
                    "response_profile": profile.detach(),
                    "confidence": confidence.detach(),
                    "safety_target": (1.0 - safe3).detach(),
                    "safety_valid": scanner_available.float().reshape(-1, 1).detach(),
                    "critic_input": critic_input.detach(),
                    "pre_tanh_action": pre_tanh.detach(),
                    "old_log_prob": log_prob.detach(),
                    "old_value": value.detach(),
                    "reset_mask": reset.detach(),
                    "actor_hidden": tuple(item.detach() for item in actor_initial),
                    "critic_hidden": tuple(item.detach() for item in critic_initial),
                    "command_penalty": command_penalty,
                    "command_rate_axis_penalty": command_rate_axis_penalty.detach(),
                    "predictive_collision_penalty": (
                        predictive_collision_penalty.detach()
                    ),
                    "predictive_collision_clearance_m": (
                        predictive_clearance_m.detach()
                    ),
                    "predictive_collision_stopping_distance_m": (
                        predictive_stopping_distance_m.detach()
                    ),
                    "predictive_collision_risk": predictive_collision_risk.detach(),
                    "predictive_collision_legacy_risk": (
                        predictive_collision_legacy_risk.detach()
                    ),
                    "predictive_collision_wallness": (
                        predictive_collision_wallness.detach()
                    ),
                    "predictive_collision_sector_risk": (
                        predictive_collision_sector_risk.detach()
                    ),
                    "missed_safe_direction_penalty": missed_safe_penalty.detach(),
                    "safe3": safe3.detach(),
                    "safety_teacher_nav_risk3": safety_teacher_diagnostics["nav_risk3"].detach(),
                    "safety_teacher_passable3": safety_teacher_diagnostics["terrain_passable3"].detach(),
                    "selected_safe": missed_safe_diagnostics["selected_safe"].detach(),
                    "best_safe": missed_safe_diagnostics["best_safe"].detach(),
                    "safe_gap": missed_safe_diagnostics["safe_gap"].detach(),
                    "safe_direction_active": missed_safe_diagnostics["active"].detach(),
                    "safe_selection_eligible": missed_safe_diagnostics[
                        "selection_eligible"
                    ].detach(),
                    "selected_safest_direction": missed_safe_diagnostics["selected_safest"].detach(),
                    "tick_penalty": command_penalty + missed_safe_penalty.reshape(-1, 1),
                    "target_cmd3": target.detach(),
                    **self._transition_extras(),
                }
            self.command.set_target(target)
            self.reset_since_tick.zero_()
            self.nav_ticks += self.num_envs
            is_tick = True
            tick_penalty = command_penalty

        # A 10 Hz instant command belongs to the current boundary frame. Patch
        # the freshly selected exec command before the 50 Hz low-level policy
        # runs; legacy slew mode is unchanged because set_target() does not
        # advance its exec command.
        aux = patch_owned_commands(
            aux, self.command.active_target, self.command.exec_cmd, self.command.command_epoch
        )
        parts["proprio"][:, p0:p1] = self.command.exec_cmd.to(parts["proprio"])
        critic_obs[:, c0:c1] = self.command.exec_cmd.to(critic_obs)
        low_action, low_metadata = self._low_level_frame(parts, critic_obs)
        low_hidden = self.low_level_encoder.get_hidden_state()
        if low_hidden is not None:
            self.low_level_encoder.set_hidden_state(
                tuple(state.clone() for state in low_hidden)
            )
        low_finite = torch.isfinite(low_action).all(dim=-1)
        if not bool(low_finite.all()):
            invalid_low_count = int((~low_finite).sum())
            self.low_level_encoder.reset_hidden_state_for_envs(
                (~low_finite).nonzero(as_tuple=False).reshape(-1)
            )
            low_action = torch.where(
                low_finite.unsqueeze(-1), low_action, torch.zeros_like(low_action)
            )
            self.nonfinite_action_fallbacks += invalid_low_count
            self.invalid_transition_count += invalid_low_count
            self.rollout_invalid |= self.training_enabled
        result = {
            "actions": low_action,
            "is_tick": is_tick,
            "tick_penalty": tick_penalty,
            **low_metadata,
        }
        self.frame_count += 1
        return result, critic_obs, aux

    def eval_frame_advance(self) -> None:
        """Advance the 50 Hz slew clock when evaluation has no frame_end hook."""
        self.command.step()

    def frame_end(self, aux: torch.Tensor, dones: torch.Tensor) -> None:
        patched = patch_owned_commands(
            aux, self.command.active_target, self.command.exec_cmd, self.command.command_epoch
        )
        self.response_buffer.append(
            patched[:, : p2_contract.RESPONSE_AUX_DIM],
            dones,
            current_segment=patched[:, p2_contract.CURRENT_SEGMENT_INDEX],
            **self._response_append_kwargs(),
        )
        dones = dones.to(self.device).bool()
        if bool(dones.any()):
            ids = dones.nonzero(as_tuple=False).reshape(-1)
            self.low_level_encoder.reset_hidden_state_for_envs(ids)
            self.actor_hidden = self._mask_hidden(self.actor_hidden, dones)
            self.critic_hidden = self._mask_hidden(self.critic_hidden, dones)
            if self.adapter_hidden is not None:
                self.adapter_hidden[:, ids, :] = 0.0
            self.command.reset(ids)
            # Keep the old episode's best distance until finish_tick settles
            # the terminal-safe reward. Non-training/eval paths have no
            # pending transition, so they can reset the live state now.
            if self.pending_tick is None:
                self.best_goal_distance[dones] = float("inf")
                self.episode_start_goal_distance[dones] = float("inf")
                self._reset_navigation_reward_state(dones)
            self.reset_since_tick |= dones
        self.command.step()

    @torch.no_grad()
    def finish_tick(
        self,
        next_obs: torch.Tensor,
        next_critic_wire: torch.Tensor,
        *,
        frame_safety_reward: torch.Tensor,
        start_goal_distance: torch.Tensor,
        end_goal_distance: torch.Tensor,
        terminal_reason: torch.Tensor,
        duration_frames: torch.Tensor,
        hard_terminated: torch.Tensor,
        timeout: torch.Tensor,
        unattributed_boundary: torch.Tensor | None = None,
        terminal_safe_aux: torch.Tensor | None = None,
        terminal_safe_exec_cmd: torch.Tensor | None = None,
        frontier_settle_mask: torch.Tensor | None = None,
        path_length_m: torch.Tensor | None = None,
    ) -> bool:
        if self.pending_tick is None:
            raise RuntimeError("P2 finish_tick called without a pending nav transition")
        critic_obs, next_aux = self._split_transport(next_critic_wire)
        critic_obs = critic_obs.clone()
        c0, c1 = nav_contract.CRITIC_CMD_SLICE
        critic_obs[:, c0:c1] = self.command.exec_cmd.to(critic_obs)
        next_input = assemble_critic_input(
            critic_obs,
            self.command.active_target,
            self._nav_capability(self.num_envs),
        )
        bootstrap_value, _ = self.critic(next_input, self.critic_hidden, self.reset_since_tick)
        hard = hard_terminated.to(self.device).reshape(-1, 1).bool()
        timed = timeout.to(self.device).reshape(-1, 1).bool()
        unattributed = (
            torch.zeros_like(hard)
            if unattributed_boundary is None
            else unattributed_boundary.to(self.device).reshape(-1, 1).bool()
        )
        terminal = hard | timed | unattributed
        bootstrap_finite = torch.isfinite(bootstrap_value).reshape(-1, 1)
        invalid_rows = ~bootstrap_finite.reshape(-1)
        bootstrap_value = torch.where(
            terminal | ~bootstrap_finite,
            torch.zeros_like(bootstrap_value),
            torch.nan_to_num(bootstrap_value),
        )
        duration_raw = duration_frames.reshape(-1).to(self.device).to(torch.float32)
        duration_valid = (
            torch.isfinite(duration_raw)
            & (duration_raw >= 1.0)
            & (duration_raw <= float(self.nav_period_frames))
        )
        duration = torch.nan_to_num(
            duration_raw,
            nan=float(self.nav_period_frames),
            posinf=float(self.nav_period_frames),
            neginf=1.0,
        ).round().long().clamp(1, self.nav_period_frames)
        start_goal_raw = start_goal_distance.reshape(-1).to(self.device)
        end_goal_raw = end_goal_distance.reshape(-1).to(self.device)
        goal_valid = (
            torch.isfinite(start_goal_raw)
            & torch.isfinite(end_goal_raw)
            & (start_goal_raw >= 0.0)
            & (end_goal_raw >= 0.0)
        )
        start_goal = torch.nan_to_num(
            start_goal_raw, nan=0.0, posinf=0.0, neginf=0.0
        )
        end_goal = torch.nan_to_num(
            end_goal_raw, nan=0.0, posinf=0.0, neginf=0.0
        )
        safety_reward_raw = frame_safety_reward.reshape(-1).to(self.device)
        safety_valid = torch.isfinite(safety_reward_raw)
        safety_reward = torch.nan_to_num(
            safety_reward_raw, nan=0.0, posinf=0.0, neginf=0.0
        )
        reason_raw = terminal_reason.reshape(-1).to(self.device).to(torch.float32)
        reason_rounded = torch.round(
            torch.nan_to_num(reason_raw, nan=0.0, posinf=0.0, neginf=0.0)
        )
        reason_valid = (
            torch.isfinite(reason_raw)
            & (reason_raw == reason_rounded)
            & (reason_rounded >= 0.0)
            & (reason_rounded <= 4.0)
        )
        reason = reason_rounded.long()
        reward_source_aux = next_aux if terminal_safe_aux is None else terminal_safe_aux
        reward_exec_cmd = (
            self.command.exec_cmd
            if terminal_safe_exec_cmd is None
            else terminal_safe_exec_cmd
        )
        reward_source_aux = reward_source_aux.to(self.device)
        reward_exec_cmd = reward_exec_cmd.to(self.device)
        reward_aux = torch.cat(
            (
                reward_source_aux[:, 12:15],
                reward_source_aux[:, p2_contract.GAIT_DUTY_SLICE],
                reward_source_aux[:, p2_contract.GAIT_MEAN_SWING_SLICE],
                reward_source_aux[:, p2_contract.GAIT_PROLONGED_RATIO_SLICE],
                reward_source_aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE],
                reward_source_aux[:, p2_contract.GAIT_VALID_INDEX : p2_contract.GAIT_VALID_INDEX + 1],
                reward_source_aux[
                    :,
                    p2_contract.BODY_COLLISION_FORCE_INDEX :
                    p2_contract.BODY_COLLISION_FORCE_INDEX + 1,
                ],
                reward_source_aux[
                    :,
                    p2_contract.BODY_COLLISION_MAPPING_VALID_INDEX :
                    p2_contract.BODY_COLLISION_MAPPING_VALID_INDEX + 1,
                ],
            ),
            dim=1,
        )
        invalid_rows |= ~duration_valid
        invalid_rows |= ~goal_valid
        invalid_rows |= ~safety_valid
        invalid_rows |= ~reason_valid
        invalid_rows |= ~self._finite_rows(
            self.pending_tick["command_penalty"],
            self.pending_tick.get(
                "predictive_collision_penalty",
                torch.zeros(self.num_envs, device=self.device),
            ),
            self.pending_tick.get(
                "missed_safe_direction_penalty",
                torch.zeros(self.num_envs, device=self.device),
            ),
            self.pending_tick["target_cmd3"],
            reward_aux,
        )
        tracking_penalty = self._tracking_error_weight() * (
            p2_contract.normalized_true_tracking_error(
                reward_exec_cmd,
                reward_source_aux[:, 12:15],
            )
        )
        tracking_penalty = torch.where(
            terminal,
            torch.zeros_like(tracking_penalty),
            torch.nan_to_num(tracking_penalty),
        )
        command_penalty = self.pending_tick["command_penalty"]
        predictive_collision_penalty = self.pending_tick.get(
            "predictive_collision_penalty",
            torch.zeros(self.num_envs, device=self.device),
        ).reshape(-1)
        missed_safe_direction_penalty = self.pending_tick.get(
            "missed_safe_direction_penalty",
            torch.zeros(self.num_envs, device=self.device),
        ).reshape(-1)
        self._ensure_navigation_reward_state()
        episode_start = torch.where(
            torch.isfinite(self.episode_start_goal_distance),
            self.episode_start_goal_distance,
            start_goal,
        )
        best_before = torch.where(
            torch.isfinite(self.best_goal_distance),
            self.best_goal_distance,
            start_goal,
        )
        candidate_best = torch.minimum(best_before, end_goal)
        settle_mask = terminal.reshape(-1)
        if frontier_settle_mask is not None:
            settle_mask = settle_mask | frontier_settle_mask.to(self.device).reshape(-1).bool()
        (
            frontier_shaping,
            frontier_potential_before,
            frontier_potential_after,
        ) = p2_contract.frontier_potential_shaping(
            episode_start,
            best_before,
            candidate_best,
            duration,
            settle_mask,
        )
        success_reward = (reason == 1).float() * p2_contract.SUCCESS_IMPULSE
        failure_reward = (reason == 2).float() * p2_contract.FAILURE_IMPULSE
        timeout_reward = (reason == 3).float() * p2_contract.TIMEOUT_IMPULSE
        time_penalty = (
            p2_contract.TIME_COST_PER_TICK
            * duration.float()
            / float(p2_contract.NAV_PERIOD_FRAMES)
        )
        crawl_penalty = p2_contract.crawl_deadzone_penalty(
            self.pending_tick["target_cmd3"]
        )
        gait_penalty, gait_excess = self.gait_baseline.penalty(reward_source_aux)
        self._ensure_navigation_reward_state()
        previous_body_collision = self.previous_body_collision.clone()
        collision_penalty, next_body_collision = (
            p2_contract.body_collision_penalty(
                reward_source_aux[:, p2_contract.BODY_COLLISION_FORCE_INDEX],
                self.previous_body_collision,
                terminal.reshape(-1),
            )
        )
        collision_mapping_valid = (
            reward_source_aux[:, p2_contract.BODY_COLLISION_MAPPING_VALID_INDEX] > 0.5
        ) & ~invalid_rows
        collision_penalty = torch.where(
            collision_mapping_valid,
            collision_penalty,
            torch.zeros_like(collision_penalty),
        )
        self.previous_body_collision = torch.where(
            collision_mapping_valid,
            next_body_collision,
            torch.zeros_like(next_body_collision),
        )
        collision_onset = (
            next_body_collision & ~previous_body_collision & collision_mapping_valid
        )
        stagnation_penalty = self._frontier_rewards(
            best_before=best_before,
            end_goal=end_goal,
            terminal=terminal,
        )
        components = {
            "frame_safety": safety_reward,
            "frontier_shaping": frontier_shaping,
            "success": success_reward,
            "failure": failure_reward,
            "timeout": timeout_reward,
            "time": time_penalty,
            "crawl": crawl_penalty,
            "command_rate": command_penalty.reshape(-1),
            "tracking": tracking_penalty.reshape(-1),
            "gait_symmetry": gait_penalty,
            "body_collision": collision_penalty,
            "predictive_collision_risk": predictive_collision_penalty,
            "missed_safe_direction": missed_safe_direction_penalty,
            "frontier_stagnation": stagnation_penalty,
        }
        components = self._override_reward_components(
            components,
            settle_mask=settle_mask,
            terminal=terminal.reshape(-1),
            reason=reason,
            reward_source_aux=reward_source_aux,
            reward_exec_cmd=reward_exec_cmd,
            start_goal_distance=start_goal,
            end_goal_distance=end_goal,
            duration_frames=duration,
            path_length_m=path_length_m,
            invalid_rows=invalid_rows,
            unattributed=unattributed.reshape(-1),
        )
        component_names = tuple(components)
        component_stack = torch.stack(
            tuple(components[name] for name in component_names), dim=0
        )
        invalid_rows |= ~torch.isfinite(component_stack).all(dim=0)
        component_stack = torch.nan_to_num(
            component_stack, nan=0.0, posinf=0.0, neginf=0.0
        )
        # Wrapper-driven reason-0 resets terminate recurrence and GAE, but do
        # not invent a failure/timeout reward or retain partial old-episode
        # shaping. Keep the row usable as a neutral terminal transition.
        component_stack[:, unattributed.reshape(-1)] = 0.0
        component_stack[:, invalid_rows] = 0.0
        components = {
            name: component_stack[index]
            for index, name in enumerate(component_names)
        }
        if bool(invalid_rows.any()):
            self.rollout_invalid = True
            self.invalid_transition_count += int(invalid_rows.sum())
            self._reset_navigation_reward_state(invalid_rows)
        total_reward = component_stack.sum(dim=0)
        positive_total = torch.clamp(component_stack, min=0.0).sum(dim=0)
        negative_total = torch.clamp(component_stack, max=0.0).sum(dim=0)
        self.episode_start_goal_distance = torch.where(
            invalid_rows,
            self.episode_start_goal_distance,
            episode_start,
        )
        self.best_goal_distance = torch.where(
            invalid_rows,
            self.best_goal_distance,
            candidate_best,
        )
        self.best_goal_distance[terminal.reshape(-1)] = float("inf")
        self.episode_start_goal_distance[terminal.reshape(-1)] = float("inf")
        gait_excess = {
            name: torch.where(
                invalid_rows,
                torch.zeros_like(value),
                torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
            )
            for name, value in gait_excess.items()
        }
        self.last_tick_penalties = {
            **{name: value.detach().reshape(-1, 1) for name, value in components.items()},
            "positive_total": positive_total.detach().reshape(-1, 1),
            "negative_total": negative_total.detach().reshape(-1, 1),
            "decomposed_total": total_reward.detach().reshape(-1, 1),
        }
        self.last_tick_diagnostics = {
            "body_collision_force": torch.nan_to_num(
                reward_source_aux[:, p2_contract.BODY_COLLISION_FORCE_INDEX],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).detach().reshape(-1, 1),
            "body_collision_contact": self.previous_body_collision.float()
            .detach()
            .reshape(-1, 1),
            "body_collision_onset": collision_onset.float()
            .detach()
            .reshape(-1, 1),
            "predictive_collision_clearance_m": self.pending_tick.get(
                "predictive_collision_clearance_m",
                torch.full(
                    (self.num_envs,),
                    p2_contract.PREDICTIVE_COLLISION_MAX_DEPTH_M,
                    device=self.device,
                ),
            ).detach().reshape(-1, 1),
            "predictive_collision_stopping_distance_m": self.pending_tick.get(
                "predictive_collision_stopping_distance_m",
                torch.zeros(self.num_envs, device=self.device),
            ).detach().reshape(-1, 1),
            "predictive_collision_risk": self.pending_tick.get(
                "predictive_collision_risk",
                torch.zeros(self.num_envs, device=self.device),
            ).detach().reshape(-1, 1),
            "predictive_collision_legacy_risk": self.pending_tick.get(
                "predictive_collision_legacy_risk",
                torch.zeros(self.num_envs, device=self.device),
            ).detach().reshape(-1, 1),
            "scanner_available": self.pending_tick.get(
                "safety_valid",
                torch.zeros(self.num_envs, 1, device=self.device),
            ).detach().reshape(-1, 1),
            **{
                f"teacher_risk_{name}": self.pending_tick.get(
                    "safety_target",
                    torch.zeros(self.num_envs, 3, device=self.device),
                )[:, index].detach().reshape(-1, 1)
                for index, name in enumerate(("left", "center", "right"))
            },
            "selected_safe": self.pending_tick.get(
                "selected_safe", torch.zeros(self.num_envs, device=self.device)
            ).detach().reshape(-1, 1),
            "best_safe": self.pending_tick.get(
                "best_safe", torch.zeros(self.num_envs, device=self.device)
            ).detach().reshape(-1, 1),
            "safe_gap": self.pending_tick.get(
                "safe_gap", torch.zeros(self.num_envs, device=self.device)
            ).detach().reshape(-1, 1),
            "safe_alternative_available": self.pending_tick.get(
                "safe_selection_eligible",
                torch.zeros(self.num_envs, device=self.device),
            ).detach().reshape(-1, 1),
            "selected_safest_direction": self.pending_tick.get(
                "selected_safest_direction", torch.zeros(self.num_envs, device=self.device)
            ).detach().reshape(-1, 1),
            **{
                f"predictive_collision_wallness_{name}": self.pending_tick.get(
                    "predictive_collision_wallness",
                    torch.zeros(self.num_envs, 3, device=self.device),
                )[:, index].detach().reshape(-1, 1)
                for index, name in enumerate(("left", "center", "right"))
            },
            **{
                f"predictive_collision_risk_{name}": self.pending_tick.get(
                    "predictive_collision_sector_risk",
                    torch.zeros(self.num_envs, 3, device=self.device),
                )[:, index].detach().reshape(-1, 1)
                for index, name in enumerate(("left", "center", "right"))
            },
            "gait_duty_excess": gait_excess["duty_excess"].detach().reshape(-1, 1),
            "gait_swing_excess": gait_excess["swing_excess"].detach().reshape(-1, 1),
            "gait_prolonged_excess": gait_excess["prolonged_excess"].detach().reshape(-1, 1),
            "gait_frequency_excess": gait_excess["frequency_excess"].detach().reshape(-1, 1),
            "frontier_potential_before": frontier_potential_before.detach().reshape(-1, 1),
            "frontier_potential_after": frontier_potential_after.detach().reshape(-1, 1),
            "terminal_potential_clawback": torch.where(
                settle_mask,
                -frontier_potential_before,
                torch.zeros_like(frontier_potential_before),
            ).detach().reshape(-1, 1),
            "reward_conservation_error": (
                total_reward - torch.stack(tuple(components.values()), dim=0).sum(dim=0)
            ).abs().detach().reshape(-1, 1),
            "unattributed_reset_boundary": unattributed.float().detach(),
            **self._extra_tick_diagnostics(),
        }
        transition = dict(self.pending_tick)
        transition.pop("command_penalty")
        transition.pop("tick_penalty")
        transition.pop("target_cmd3")
        transition.pop("predictive_collision_penalty", None)
        transition.pop("predictive_collision_clearance_m", None)
        transition.pop("predictive_collision_stopping_distance_m", None)
        transition.pop("predictive_collision_risk", None)
        transition.pop("predictive_collision_legacy_risk", None)
        transition.pop("predictive_collision_wallness", None)
        transition.pop("predictive_collision_sector_risk", None)
        transition.pop("missed_safe_direction_penalty", None)
        transition.pop("safe3", None)
        transition.pop("safety_teacher_nav_risk3", None)
        transition.pop("safety_teacher_passable3", None)
        transition.pop("selected_safe", None)
        transition.pop("best_safe", None)
        transition.pop("safe_gap", None)
        transition.pop("safe_direction_active", None)
        transition.pop("safe_selection_eligible", None)
        transition.pop("selected_safest_direction", None)
        transition.update(
            {
                "reward": total_reward.reshape(-1, 1),
                "duration_frames": duration.reshape(-1, 1),
                "bootstrap_value": bootstrap_value,
                # The platform wrapper does not expose a terminal critic
                # observation. Timeout therefore uses the conservative
                # no-bootstrap fallback instead of the post-reset episode.
                "bootstrap_mask": (~terminal).float(),
                "continuation_mask": (~terminal).float(),
                # A reason-0 reset is a real recurrent boundary but has no
                # attributable outcome. Exclude that row from every PPO/value
                # statistic instead of training it toward an artificial zero
                # return.
                "valid_mask": (~unattributed.reshape(-1) & ~invalid_rows.reshape(-1))
                .float()
                .reshape(-1, 1),
            }
        )
        self.rollout.add(**transition)
        self.pending_tick = None
        return self.rollout.full

    def _override_reward_components(self, components, **_context):
        """Stage-specific reward profile hook; P2 keeps its current contract."""
        return components

    def maybe_unfreeze_cnn(self, effective_seconds: float) -> bool:
        previous = self.cnn_unfrozen
        desired = bool(p2_contract.training_schedule(effective_seconds)["cnn_unfrozen"])
        if desired and not previous and self.rollout.step != 0:
            self._apply_training_schedule(
                effective_seconds,
                allow_cnn_transition=False,
            )
            return False
        self._apply_training_schedule(effective_seconds)
        if previous or not self.cnn_unfrozen:
            return False
        self.rollout = self.rollout.reset(store_depth=True)
        if self.logger:
            self.logger.info(
                "[P2NavPPO] NavigationEncoder unfrozen at 20 session minutes "
                "with 0.2x layered learning rates"
            )
        return True

    def update_training_clocks(self, session_effective_seconds: float) -> None:
        """Advance all persisted clocks and apply one rollout-boundary schedule."""
        self._apply_training_schedule(session_effective_seconds)

    @staticmethod
    def _actor_group_base_lr(group: dict) -> float:
        name = str(group.get("name", ""))
        if "base_lr" in group:
            return float(group["base_lr"])
        if name == "navigation_safety_head":
            base_lr = p2_contract.SAFETY_HEAD_LR
        elif name in {"actor_trunk", "actor_main", "actor_vy"}:
            base_lr = p2_contract.ACTOR_LR
        elif name.startswith("navigation_"):
            layer = name.removeprefix("navigation_")
            if layer not in p2_contract.CNN_LAYER_LRS:
                raise RuntimeError(f"P2 unknown navigation optimizer group {name!r}")
            base_lr = p2_contract.CNN_LAYER_LRS[layer]
        else:
            raise RuntimeError(f"P2 unknown actor optimizer group {name!r}")
        group["base_lr"] = base_lr
        return float(base_lr)

    def _apply_training_schedule(
        self,
        session_effective_seconds: float,
        *,
        allow_cnn_transition: bool = True,
    ) -> dict[str, object]:
        self.session_effective_seconds = max(
            0.0, float(session_effective_seconds)
        )
        self.effective_training_seconds = self.session_effective_seconds
        self.lifetime_effective_seconds = (
            float(getattr(self, "lifetime_base_seconds", 0.0))
            + self.session_effective_seconds
        )
        schedule = p2_contract.training_schedule(self.session_effective_seconds)
        vy_limits = p2_contract.vy_action_limits()
        self.current_vy_trusted_limit = float(vy_limits["trusted_abs_vy"])
        self.current_vy_hard_limit = float(vy_limits["hard_abs_vy"])
        desired_cnn_unfrozen = bool(schedule["cnn_unfrozen"])
        if desired_cnn_unfrozen and not self.cnn_unfrozen and not allow_cnn_transition:
            schedule = dict(schedule)
            schedule["cnn_unfrozen"] = False
            schedule["navigation_multiplier"] = 0.0
        self.cnn_unfrozen = bool(schedule["cnn_unfrozen"])
        self.entropy_coefficient = float(schedule["entropy_coefficient"])
        self.optimizer_phase = str(schedule.get("optimizer_phase", schedule["phase"]))
        for group in self.actor_optimizer.param_groups:
            name = str(group.get("name", ""))
            if name == "navigation_safety_head":
                multiplier = float(schedule["safety_head_multiplier"])
            elif name.startswith("navigation_"):
                multiplier = float(schedule["navigation_multiplier"])
            elif name == "actor_trunk":
                multiplier = float(schedule["actor_trunk_multiplier"])
            elif name == "actor_main":
                multiplier = float(schedule["actor_multiplier"])
            elif name == "actor_vy":
                multiplier = float(schedule["vy_actor_multiplier"])
            else:
                raise RuntimeError(f"P2 unknown actor optimizer group {name!r}")
            group["lr"] = self._actor_group_base_lr(group) * multiplier
        if self.actor_scheduler is not None:
            self.actor_scheduler.base_lrs = [
                self._actor_group_base_lr(group)
                for group in self.actor_optimizer.param_groups
            ]
            self.actor_scheduler._last_lr = [
                float(group["lr"]) for group in self.actor_optimizer.param_groups
            ]
        self.critic_optimizer.param_groups[0]["lr"] = (
            p2_contract.CRITIC_LR * float(schedule["critic_multiplier"])
        )
        self.response_optimizer.param_groups[0]["lr"] = (
            p2_contract.ADAPTER_LR * float(schedule["adapter_multiplier"])
        )
        if self.critic_scheduler is not None:
            self.critic_scheduler.base_lrs = [p2_contract.CRITIC_LR]
            self.critic_scheduler._last_lr = [
                float(self.critic_optimizer.param_groups[0]["lr"])
            ]
        if self.response_scheduler is not None:
            self.response_scheduler.base_lrs = [p2_contract.ADAPTER_LR]
            self.response_scheduler._last_lr = [
                float(self.response_optimizer.param_groups[0]["lr"])
            ]
        return schedule

    def _stack_refs(self, name, refs):
        source = getattr(self.rollout, name)
        return torch.stack(
            [
                source[
                    ref.start : ref.start + self.rollout.sequence_length,
                    ref.env,
                ]
                for ref in refs
            ],
            dim=1,
        ).to(self.device, non_blocking=True)

    def _actor_sequence_batch(self, refs, *, advantage_mean, advantage_std):
        transfer_started = time.perf_counter()
        batch = {
            name: self._stack_refs(name, refs)
            for name in (
                "nav_nonvisual",
                "response_profile",
                "confidence",
                "pre_tanh_action",
                "old_log_prob",
                "advantages",
                "valid_mask",
                "reset_mask",
                "safety_target",
                "safety_valid",
                "clean_action_mean",
                "camera_aux_mask",
            )
        }
        batch["advantages"] = batch["valid_mask"] * (
            batch["advantages"] - float(advantage_mean)
        ) / float(advantage_std)
        if self.rollout.store_depth:
            batch["depth"] = self._stack_refs("depth", refs)
        else:
            batch["nav_feat"] = self._stack_refs("nav_feat", refs)
        batch["actor_hidden"] = (
            torch.stack(
                [self.rollout.actor_h[ref.start, :, ref.env] for ref in refs],
                dim=1,
            ).to(self.device, non_blocking=True),
            torch.stack(
                [self.rollout.actor_c[ref.start, :, ref.env] for ref in refs],
                dim=1,
            ).to(self.device, non_blocking=True),
        )
        self._h2d_time_s += time.perf_counter() - transfer_started
        return batch

    def _critic_sequence_batch(self, refs):
        transfer_started = time.perf_counter()
        batch = {
            name: self._stack_refs(name, refs)
            for name in ("critic_input", "returns", "valid_mask", "reset_mask")
        }
        batch["critic_hidden"] = (
            torch.stack(
                [self.rollout.critic_h[ref.start, :, ref.env] for ref in refs],
                dim=1,
            ).to(self.device, non_blocking=True),
            torch.stack(
                [self.rollout.critic_c[ref.start, :, ref.env] for ref in refs],
                dim=1,
            ).to(self.device, non_blocking=True),
        )
        self._h2d_time_s += time.perf_counter() - transfer_started
        return batch

    def _rollout_advantage_stats(self) -> tuple[float, float]:
        valid_storage = getattr(
            self.rollout,
            "valid_mask",
            torch.ones_like(self.rollout.advantages),
        )
        valid = valid_storage[: self.rollout.step] > 0.5
        values = self.rollout.advantages[: self.rollout.step].float()[valid]
        if not values.numel():
            return 0.0, 1.0
        return float(values.mean()), float(
            values.std(unbiased=False).clamp_min(1.0e-8)
        )

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
        log_prob, entropy, mean, _, _ = self.actor.evaluate_actions(
            inputs,
            batch["pre_tanh_action"],
            batch["actor_hidden"],
            batch["reset_mask"],
        )
        ratio = torch.exp(log_prob - batch["old_log_prob"])
        surrogate_element = -torch.minimum(
            ratio * batch["advantages"],
            torch.clamp(ratio, 0.8, 1.2) * batch["advantages"],
        )
        valid_mask = batch.get(
            "valid_mask", torch.ones_like(batch["old_log_prob"])
        ) > 0.5
        surrogate = self._masked_mean(surrogate_element, valid_mask)
        if getattr(self, "track_safety_enabled", True):
            safety_logits = self.safety_head(feat)
            safety_element_loss = F.binary_cross_entropy_with_logits(
                safety_logits,
                batch["safety_target"],
                reduction="none",
            )
            safety_mask = (
                batch["safety_valid"] * valid_mask.to(batch["safety_valid"].dtype)
            ).expand_as(safety_element_loss)
            safety_denominator = safety_mask.sum()
            safety_loss = (
                (safety_element_loss * safety_mask).sum()
                / safety_denominator.clamp_min(1.0)
            )
            risk_probability = torch.sigmoid(safety_logits)
            student_risk = torch.stack(
                tuple(
                    self._masked_mean(
                        risk_probability[..., index], valid_mask
                    )
                    for index in range(3)
                )
            )
        else:
            safety_loss = surrogate.new_zeros(())
            student_risk = torch.zeros(3, device=surrogate.device)
        auxiliary_loss, auxiliary_metrics = self._actor_auxiliary_loss(
            normalized_mean=torch.tanh(mean),
            batch=batch,
            ppo_actor_loss=surrogate,
        )
        total_loss = (
            surrogate
            - self.entropy_coefficient * self._masked_mean(entropy, valid_mask)
            + p2_contract.SAFETY_BCE_WEIGHT * safety_loss
            + auxiliary_loss
        )
        with torch.no_grad():
            log_ratio = log_prob - batch["old_log_prob"]
            approx_kl = self._masked_mean(
                (torch.exp(log_ratio) - 1.0) - log_ratio, valid_mask
            )
            clip_fraction = self._masked_mean(
                ((ratio - 1.0).abs() > 0.2).float(), valid_mask
            )
            entropy_mean = self._masked_mean(entropy, valid_mask)
        return total_loss, {
            "surrogate_loss": surrogate.detach(),
            "entropy": entropy_mean.detach(),
            "approx_kl": approx_kl.detach(),
            "clip_fraction": clip_fraction.detach(),
            "safety_bce": safety_loss.detach(),
            "scanner_valid_share": self._masked_mean(
                batch["safety_valid"].float(), valid_mask
            ).detach(),
            "safety_head_risk_left": student_risk[0].detach(),
            "safety_head_risk_center": student_risk[1].detach(),
            "safety_head_risk_right": student_risk[2].detach(),
            **auxiliary_metrics,
        }

    def _critic_micro_loss(self, batch):
        values, _ = self.critic(
            batch["critic_input"], batch["critic_hidden"], batch["reset_mask"]
        )
        return self._masked_mean(
            (values - batch["returns"]) ** 2,
            batch.get("valid_mask", torch.ones_like(batch["returns"])) > 0.5,
        )

    def _run_ppo_epochs(self) -> dict[str, float]:
        refs = self.rollout.sequence_refs(self.ppo_generator)
        expected_sequences = (
            self.num_envs
            * self.rollout.num_ticks
            // self.rollout.sequence_length
        )
        if len(refs) != expected_sequences:
            raise RuntimeError(
                f"P2 expected {expected_sequences} sequences, got {len(refs)}"
            )
        advantage_mean, advantage_std = self._rollout_advantage_stats()
        minibatch_sequences = max(
            1, (len(refs) + self.num_mini_batches - 1) // self.num_mini_batches
        )
        auxiliary_metric_names = self._actor_auxiliary_metric_names()
        totals = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "safety_bce": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "entropy": 0.0,
            "scanner_valid_share": 0.0,
            "safety_head_risk_left": 0.0,
            "safety_head_risk_center": 0.0,
            "safety_head_risk_right": 0.0,
            "updates": 0.0,
            "actor_update_time_s": 0.0,
            "critic_update_time_s": 0.0,
            **{name: 0.0 for name in auxiliary_metric_names},
        }
        epoch_metrics = []
        for _epoch in range(self.num_learning_epochs):
            epoch_started = time.perf_counter()
            epoch_totals = {
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "actor_loss": 0.0,
                "entropy": 0.0,
                "safety_bce": 0.0,
                "micro_weight": 0.0,
            }
            for minibatch_start in range(0, len(refs), minibatch_sequences):
                minibatch = refs[
                    minibatch_start : minibatch_start + minibatch_sequences
                ]
                if hasattr(self.rollout, "valid_mask"):
                    minibatch_valid_steps = sum(
                        int(
                            self.rollout.valid_mask[
                                ref.start : ref.start + self.rollout.sequence_length,
                                ref.env,
                            ].sum().item()
                        )
                        for ref in minibatch
                    )
                else:
                    minibatch_valid_steps = (
                        len(minibatch) * self.rollout.sequence_length
                    )
                if minibatch_valid_steps <= 0:
                    continue
                actor_started = time.perf_counter()
                actor_enabled = bool(self._actor_update_enabled())
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_finite = actor_enabled
                actor_has_grad = False
                for micro_start in range(0, len(minibatch), self.micro_sequences) if actor_enabled else ():
                    micro = minibatch[micro_start : micro_start + self.micro_sequences]
                    actor_batch = self._actor_sequence_batch(
                        micro,
                        advantage_mean=advantage_mean,
                        advantage_std=advantage_std,
                    )
                    micro_valid_steps = int(actor_batch["valid_mask"].sum().item())
                    if micro_valid_steps <= 0:
                        continue
                    loss, actor_metrics = self._actor_micro_loss(actor_batch)
                    scale = P2RolloutStorage.microbatch_loss_scale(
                        micro_valid_steps, minibatch_valid_steps
                    )
                    if not bool(torch.isfinite(loss)):
                        actor_finite = False
                        break
                    if loss.requires_grad:
                        (loss * scale).backward()
                        actor_has_grad = True
                    totals["actor_loss"] += float(loss.detach()) * scale
                    for name in (
                        "safety_bce", "approx_kl", "clip_fraction", "entropy",
                        "scanner_valid_share", "safety_head_risk_left",
                        "safety_head_risk_center", "safety_head_risk_right",
                    ):
                        totals[name] += float(actor_metrics[name]) * scale
                    for name in auxiliary_metric_names:
                        if name in actor_metrics:
                            totals[name] += float(actor_metrics[name]) * scale
                    epoch_totals["actor_loss"] += float(loss.detach()) * scale
                    for name in ("safety_bce", "approx_kl", "clip_fraction", "entropy"):
                        epoch_totals[name] += float(actor_metrics[name]) * scale
                    epoch_totals["micro_weight"] += scale
                if actor_enabled and actor_has_grad and actor_finite and all(
                    p.grad is None or bool(torch.isfinite(p.grad).all())
                    for g in self.actor_optimizer.param_groups for p in g["params"]
                ):
                    nn.utils.clip_grad_norm_(
                        [p for g in self.actor_optimizer.param_groups for p in g["params"]],
                        self.max_grad_norm,
                    )
                    self.actor_optimizer.step()
                    with torch.no_grad():
                        self.actor.log_std.clamp_(
                            p2_contract.LOG_STD_MIN, p2_contract.LOG_STD_MAX
                        )
                    self.actor_gradient_steps += 1
                elif actor_enabled and not actor_finite:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.skipped_nonfinite += 1
                totals["actor_update_time_s"] += time.perf_counter() - actor_started

                critic_started = time.perf_counter()
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_finite = True
                for micro_start in range(0, len(minibatch), self.micro_sequences):
                    micro = minibatch[micro_start : micro_start + self.micro_sequences]
                    critic_batch = self._critic_sequence_batch(micro)
                    micro_valid_steps = int(
                        critic_batch.get(
                            "valid_mask", torch.ones_like(critic_batch["returns"])
                        ).sum().item()
                    )
                    if micro_valid_steps <= 0:
                        continue
                    loss = self._critic_micro_loss(critic_batch)
                    scale = P2RolloutStorage.microbatch_loss_scale(
                        micro_valid_steps, minibatch_valid_steps
                    )
                    if not bool(torch.isfinite(loss)):
                        critic_finite = False
                        break
                    (loss * scale).backward()
                    totals["critic_loss"] += float(loss.detach()) * scale
                critic_finite = critic_finite and all(
                    parameter.grad is None
                    or bool(torch.isfinite(parameter.grad).all())
                    for parameter in self.critic.parameters()
                )
                if critic_finite:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    self.critic_gradient_steps += 1
                else:
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    self.skipped_nonfinite += 1
                totals["critic_update_time_s"] += time.perf_counter() - critic_started
                totals["updates"] += 1.0
            weight = max(epoch_totals.pop("micro_weight"), 1.0e-12)
            epoch_metrics.append(
                {
                    **{name: value / weight for name, value in epoch_totals.items()},
                    "update_time_s": time.perf_counter() - epoch_started,
                }
            )
        if totals["updates"]:
            totals["actor_loss"] /= totals["updates"]
            totals["critic_loss"] /= totals["updates"]
            for name in (
                "safety_bce", "approx_kl", "clip_fraction", "entropy",
                "scanner_valid_share", "safety_head_risk_left",
                "safety_head_risk_center", "safety_head_risk_right",
            ):
                totals[name] /= totals["updates"]
            for name in auxiliary_metric_names:
                totals[name] /= totals["updates"]
        for index, metrics in enumerate(epoch_metrics):
            for name, value in metrics.items():
                totals[f"epoch_{index + 1}_{name}"] = value
        return totals

    def _actor_update_enabled(self) -> bool:
        return True

    @staticmethod
    def _masked_mean(value, mask):
        mask = mask.to(value.dtype)
        while mask.ndim > value.ndim and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.ndim > value.ndim:
            raise ValueError(
                "P2 masked mean received a mask with incompatible rank: "
                f"value={tuple(value.shape)} mask={tuple(mask.shape)}"
            )
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1.0)

    @staticmethod
    def _masked_group_share(group, mask):
        valid = mask.to(device=group.device, dtype=torch.bool)
        selected = valid & group.to(dtype=torch.bool).unsqueeze(-1)
        return selected.float().sum() / valid.float().sum().clamp_min(1.0)

    def _adapter_update(self) -> dict[str, float]:
        batch = self.response_buffer.sample(
            batch_envs=self.adapter_batch_envs,
            generator=self.adapter_generator,
        )
        if batch is None:
            return {"adapter_loss": 0.0, "adapter_updates": 0.0}
        for field in (
            "burn_in_observations",
            "burn_in_reset_mask",
            "observations",
            "reset_mask",
            "velocity_labels",
            "pose_labels",
            "stuck_labels",
            "horizon_mask",
            "pose_mask",
        ):
            setattr(
                batch,
                field,
                getattr(batch, field).to(self.device, non_blocking=True),
            )
        hidden = None
        if batch.burn_in_observations.shape[0]:
            with torch.no_grad():
                _, hidden = self.response_adapter(
                    batch.burn_in_observations, reset_mask=batch.burn_in_reset_mask
                )
            hidden = hidden.detach()
        profile, _ = self.response_adapter(
            batch.observations, hidden=hidden, reset_mask=batch.reset_mask
        )
        parts = self.response_adapter.split_profile(profile)
        velocity_error = parts["velocity"] - batch.velocity_labels
        horizon_weights = torch.tensor((1.0, 0.75, 0.5), device=self.device).view(1, 1, 3)
        weighted = batch.horizon_mask.float() * horizon_weights
        velocity = (
            F.smooth_l1_loss(parts["velocity"], batch.velocity_labels, reduction="none").mean(-1)
            * weighted
        ).sum() / weighted.sum().clamp_min(1.0)
        pose = self._masked_mean(
            F.smooth_l1_loss(parts["pose_delta"], batch.pose_labels, reduction="none"),
            batch.pose_mask,
        )
        positive = float(
            (batch.stuck_labels * batch.pose_mask.float()).sum()
            / batch.pose_mask.float().sum().clamp_min(1.0)
        )
        self.stuck_positive_ema = 0.99 * self.stuck_positive_ema + 0.01 * positive
        pos_weight = min(10.0, max(1.0, (1.0 - self.stuck_positive_ema) / max(1e-4, self.stuck_positive_ema)))
        stuck = self._masked_mean(
            F.binary_cross_entropy_with_logits(
                parts["stuck_logit"], batch.stuck_labels,
                pos_weight=torch.tensor(pos_weight, device=self.device), reduction="none"
            ), batch.pose_mask
        )
        error_1s = velocity_error[..., 2, :]
        log_sigma = parts["velocity_log_sigma"]
        nll = self._masked_mean(
            0.5 * (error_1s.square() * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma),
            batch.horizon_mask[..., 2],
        )
        loss = velocity + 0.5 * pose + 0.25 * stuck + 0.1 * nll
        if not bool(torch.isfinite(loss)):
            self.skipped_nonfinite += 1
            return {"adapter_loss": float("nan"), "adapter_updates": 0.0}
        self.response_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.response_adapter.parameters()):
            self.response_optimizer.zero_grad(set_to_none=True)
            self.skipped_nonfinite += 1
            return {"adapter_loss": float("nan"), "adapter_updates": 0.0}
        adapter_grad_norm = nn.utils.clip_grad_norm_(
            self.response_adapter.parameters(), 0.5
        )
        self.response_optimizer.step()
        self.adapter_gradient_steps += 1
        horizon_mae = []
        for horizon in range(3):
            horizon_mae.append(
                self._masked_mean(
                    velocity_error[..., horizon, :].abs().mean(dim=-1),
                    batch.horizon_mask[..., horizon],
                )
            )
        metadata = batch.metadata if isinstance(batch.metadata, dict) else {}
        latest_batch, recent_batch, parent_batch, track_batch, replay_total = (
            self._adapter_replay_counts(metadata)
        )
        zero_baseline = self._masked_mean(
            batch.velocity_labels.abs().mean(dim=-1), batch.horizon_mask
        )
        exec_baseline = batch.observations[..., 3:6].unsqueeze(-2).expand_as(
            batch.velocity_labels
        )
        copy_baseline = self._masked_mean(
            (exec_baseline - batch.velocity_labels).abs().mean(dim=-1),
            batch.horizon_mask,
        )
        prediction_mae = self._masked_mean(
            velocity_error.abs().mean(dim=-1), batch.horizon_mask
        )
        sigma = torch.exp(log_sigma)
        one_sigma = self._masked_mean(
            (error_1s.abs() <= sigma).float(), batch.horizon_mask[..., 2]
        )
        two_sigma = self._masked_mean(
            (error_1s.abs() <= 2.0 * sigma).float(), batch.horizon_mask[..., 2]
        )
        stuck_valid = batch.pose_mask.float()
        stuck_pred = (torch.sigmoid(parts["stuck_logit"]) >= 0.5).float()
        true_positive = (stuck_pred * batch.stuck_labels * stuck_valid).sum()
        predicted_positive = (stuck_pred * stuck_valid).sum()
        actual_positive = (batch.stuck_labels * stuck_valid).sum()
        precision = true_positive / predicted_positive.clamp_min(1.0)
        recall = true_positive / actual_positive.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-6)
        axis_mae = []
        for horizon in range(3):
            axis_mae.append(
                [
                    self._masked_mean(
                        velocity_error[..., horizon, axis].abs(),
                        batch.horizon_mask[..., horizon],
                    )
                    for axis in range(3)
                ]
            )
        segment_labels = tuple(
            self.config.get(
                "track_segment_labels",
                p2_contract.TRACK_SEGMENT_METRIC_LABELS,
            )
        )
        segment_metric_labels = (
            p2_contract.TRACK_SEGMENT_METRIC_LABELS
            if all(
                label in p2_contract.TRACK_SEGMENT_METRIC_LABELS
                for label in segment_labels
            )
            else p2_contract.CANONICAL_TRACK_SEGMENT_METRIC_LABELS
        )
        grouped_metrics = {
            f"adapter_{label}_{suffix}": 0.0
            for label in segment_metric_labels
            for suffix in ("sample_share", "mae")
        }
        current_segment = metadata.get("current_segment")
        if (
            torch.is_tensor(current_segment)
            and current_segment.shape == batch.reset_mask.shape
        ):
            current_segment = current_segment.to(self.device).round().long()
            segment_valid = current_segment >= 0
            segment_horizon_mask = (
                batch.horizon_mask & segment_valid.unsqueeze(-1)
            )
            metric_segment = p2_contract.track_segment_metric_indices(
                current_segment, segment_labels
            )
            for row, label in enumerate(segment_metric_labels):
                group = metric_segment == row
                grouped_metrics[f"adapter_{label}_sample_share"] = float(
                    self._masked_group_share(group, segment_horizon_mask)
                )
                grouped_metrics[f"adapter_{label}_mae"] = float(
                    self._masked_mean(
                        velocity_error.abs().mean(dim=-1),
                        segment_horizon_mask & group.unsqueeze(-1),
                    ).detach()
                )
        target = batch.observations[..., 0:3]
        outer = (target[..., 0] > p2_contract.TRUSTED_CORE["vx"][1]) | (
            target[..., 1].abs() > p2_contract.TRUSTED_CORE["vy"][1]
        ) | (
            target[..., 2].abs() > p2_contract.TRUSTED_CORE["wz"][1]
        )
        for label, group in (("core", ~outer), ("outer", outer)):
            grouped_metrics[f"adapter_{label}_sample_share"] = float(
                self._masked_group_share(group, batch.horizon_mask)
            )
            grouped_metrics[f"adapter_{label}_mae"] = float(
                self._masked_mean(
                    velocity_error.abs().mean(dim=-1),
                    batch.horizon_mask & group.unsqueeze(-1),
                ).detach()
            )
        vy_specialty = (
            (target[..., 1].abs() > 0.05)
            & (target[..., 0].abs() <= 0.10)
            & (target[..., 2].abs() <= 0.10)
        )
        for label, group in (
            ("vy_specialty", vy_specialty),
            ("joint_outer", outer & ~vy_specialty),
        ):
            grouped_metrics[f"adapter_{label}_sample_share"] = float(
                self._masked_group_share(group, batch.horizon_mask)
            )
            grouped_metrics[f"adapter_{label}_mae"] = float(
                self._masked_mean(
                    velocity_error.abs().mean(dim=-1),
                    batch.horizon_mask & group.unsqueeze(-1),
                ).detach()
            )
        adapter_confidence = p2_contract.adapter_confidence(
            velocity_valid=batch.observations[..., 9:10],
            velocity_age=batch.observations[..., 10:11],
            velocity_log_sigma=log_sigma,
            target_cmd3=target,
        ).squeeze(-1)
        confidence_buckets = (
            ("low", adapter_confidence < 1.0 / 3.0),
            (
                "mid",
                (adapter_confidence >= 1.0 / 3.0)
                & (adapter_confidence < 2.0 / 3.0),
            ),
            ("high", adapter_confidence >= 2.0 / 3.0),
        )
        for label, group in confidence_buckets:
            grouped_metrics[f"adapter_confidence_{label}_share"] = float(
                group.float().mean()
            )
            grouped_metrics[f"adapter_confidence_{label}_mae"] = float(
                self._masked_mean(
                    velocity_error.abs().mean(dim=-1),
                    batch.horizon_mask & group.unsqueeze(-1),
                ).detach()
            )
        return {
            "adapter_loss": float(loss.detach()),
            "adapter_velocity_loss": float(velocity.detach()),
            "adapter_pose_loss": float(pose.detach()),
            "adapter_stuck_loss": float(stuck.detach()),
            "adapter_updates": 1.0,
            "adapter_velocity_mae_02s": float(horizon_mae[0].detach()),
            "adapter_velocity_mae_06s": float(horizon_mae[1].detach()),
            "adapter_velocity_mae_10s": float(horizon_mae[2].detach()),
            "adapter_nll_10s": float(nll.detach()),
            "adapter_zero_baseline_mae": float(zero_baseline.detach()),
            "adapter_copy_exec_baseline_mae": float(copy_baseline.detach()),
            "adapter_gain_vs_zero": float((zero_baseline - prediction_mae).detach()),
            "adapter_gain_vs_copy_exec": float((copy_baseline - prediction_mae).detach()),
            "adapter_sigma_mean": float(sigma.mean().detach()),
            "adapter_coverage_1sigma": float(one_sigma.detach()),
            "adapter_coverage_2sigma": float(two_sigma.detach()),
            "adapter_stuck_prevalence": positive,
            "adapter_stuck_precision": float(precision.detach()),
            "adapter_stuck_recall": float(recall.detach()),
            "adapter_stuck_f1": float(f1.detach()),
            "adapter_gradient_norm": float(adapter_grad_norm.detach()),
            "adapter_reset_mask_ratio": float(batch.reset_mask.float().mean()),
            "adapter_valid_02s": float(batch.horizon_mask[..., 0].float().mean()),
            "adapter_valid_06s": float(batch.horizon_mask[..., 1].float().mean()),
            "adapter_valid_10s": float(batch.horizon_mask[..., 2].float().mean()),
            "adapter_parent_replay_ratio": (
                parent_batch / replay_total if replay_total > 0.0 else 0.0
            ),
            "adapter_track_records": track_batch,
            "adapter_parent_records": parent_batch,
            "adapter_latest_records": latest_batch,
            "adapter_recent_records": recent_batch,
            "adapter_latest_replay_ratio": latest_batch / replay_total if replay_total > 0.0 else 0.0,
            "adapter_recent_replay_ratio": recent_batch / replay_total if replay_total > 0.0 else 0.0,
            "adapter_low_level_version_lag": float(metadata.get("active_version_lag", 0.0)),
            "adapter_compatible_current_records": float(metadata.get("compatible_current_records", 0.0)),
            "adapter_compatible_parent_records": float(metadata.get("compatible_parent_records", 0.0)),
            "adapter_compat_rejected_records": float(metadata.get("rejected_records", 0.0)),
            **{
                f"adapter_{axis}_mae_{label}": float(axis_mae[horizon][axis_index].detach())
                for horizon, label in enumerate(("02s", "06s", "10s"))
                for axis_index, axis in enumerate(("vx", "vy", "wz"))
            },
            **grouped_metrics,
        }

    def _training_monitor_metrics(self) -> dict[str, float]:
        valid = getattr(
            self.rollout,
            "valid_mask",
            torch.ones_like(self.rollout.rewards),
        ) > 0.5

        def valid_mean(value: torch.Tensor) -> float:
            selected = value.float()[valid]
            return float(selected.mean()) if selected.numel() else 0.0

        def valid_std(value: torch.Tensor) -> float:
            selected = value.float()[valid]
            return (
                float(selected.std(unbiased=False)) if selected.numel() else 0.0
            )

        main_action_std = torch.exp(
            torch.clamp(
                self.actor.log_std.detach(),
                p2_contract.LOG_STD_MIN,
                p2_contract.LOG_STD_MAX,
            )
        ).to(device="cpu")
        vy_action_std = float(
            torch.exp(
                torch.clamp(
                    self.actor.vy_log_std.detach(),
                    p2_contract.LOG_STD_MIN,
                    p2_contract.LOG_STD_MAX,
                )
            ).cpu()[0]
        )
        actor_lr = next(
            float(group["lr"])
            for group in self.actor_optimizer.param_groups
            if group.get("name") == "actor_main"
        )
        vy_actor_lr = next(
            float(group["lr"])
            for group in self.actor_optimizer.param_groups
            if group.get("name") == "actor_vy"
        )
        navigation_lr = max(
            float(group["lr"])
            for group in self.actor_optimizer.param_groups
            if str(group.get("name", "")).startswith("navigation_")
            and group.get("name") != "navigation_safety_head"
        )
        safety_head_lr = next(
            float(group["lr"])
            for group in self.actor_optimizer.param_groups
            if group.get("name") == "navigation_safety_head"
        )
        return {
            "rollout_reward_mean": valid_mean(self.rollout.rewards),
            "rollout_reward_std": valid_std(self.rollout.rewards),
            "rollout_return_mean": valid_mean(self.rollout.returns),
            "rollout_value_mean": valid_mean(self.rollout.old_value),
            "rollout_advantage_mean": valid_mean(self.rollout.advantages),
            "rollout_advantage_std": valid_std(self.rollout.advantages),
            "rollout_valid_share": float(valid.float().mean()),
            "action_std_vx": float(main_action_std[0]),
            "action_std_vy": vy_action_std,
            "action_std_wz": float(main_action_std[1]),
            "actor_learning_rate": actor_lr,
            "vy_actor_learning_rate": vy_actor_lr,
            "navigation_learning_rate": navigation_lr,
            "safety_head_learning_rate": safety_head_lr,
            "critic_learning_rate": float(self.critic_optimizer.param_groups[0]["lr"]),
            "adapter_learning_rate": float(self.response_optimizer.param_groups[0]["lr"]),
            "entropy_coefficient": self.entropy_coefficient,
            "vy_trusted_limit": self.current_vy_trusted_limit,
            "vy_hard_limit": self.current_vy_hard_limit,
            "gait_baseline_finalized": float(self.gait_baseline.finalized),
            "gait_baseline_samples": float(self.gait_baseline.samples),
            "return_stat_mean": float(self.return_statistics["mean"]),
            "return_stat_std": float(
                max(
                    0.0,
                    self.return_statistics["m2"]
                    / max(1, self.return_statistics["count"] - 1),
                )
                ** 0.5
            ),
            "skipped_nonfinite_total": float(self.skipped_nonfinite),
            "adapter_oom_skips": float(self.adapter_oom_skips),
        }

    def _update_return_statistics(self) -> None:
        valid = getattr(
            self.rollout,
            "valid_mask",
            torch.ones_like(self.rollout.returns),
        ).detach().to(device="cpu") > 0.5
        values = self.rollout.returns.detach().to(
            device="cpu", dtype=torch.float64
        )[valid]
        if not values.numel():
            return
        batch_count = int(values.numel())
        batch_mean = float(values.mean())
        batch_m2 = float(((values - batch_mean) ** 2).sum())
        count = int(self.return_statistics["count"])
        mean = float(self.return_statistics["mean"])
        m2 = float(self.return_statistics["m2"])
        total = count + batch_count
        delta = batch_mean - mean
        self.return_statistics["mean"] = mean + delta * batch_count / total
        self.return_statistics["m2"] = (
            m2 + batch_m2 + delta * delta * count * batch_count / total
        )
        self.return_statistics["count"] = total

    def update(self) -> dict[str, float]:
        self.rollout.compute_returns()
        self._update_return_statistics()
        started = time.perf_counter()
        actor_steps_before = self.actor_gradient_steps
        critic_steps_before = self.critic_gradient_steps
        if self.rollout_invalid:
            self.skipped_nonfinite += 1
            metrics = {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "updates": 0.0,
                "actor_update_time_s": 0.0,
                "critic_update_time_s": 0.0,
                "ppo_rollout_skipped_nonfinite": 1.0,
            }
        else:
            try:
                metrics = self._run_ppo_epochs()
            except torch.cuda.OutOfMemoryError:
                if (
                    self.micro_sequences <= 2
                    or self.actor_gradient_steps != actor_steps_before
                    or self.critic_gradient_steps != critic_steps_before
                ):
                    raise
                self.micro_sequences = 2
                self.navigation_encoder.activation_checkpointing = True
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                if self.logger:
                    self.logger.warning(
                        "[P2NavPPO] recoverable first-minibatch OOM; CNN microbatch "
                        "64 -> 32 frames and activation checkpointing enabled"
                    )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                metrics = self._run_ppo_epochs()
        for parameter in (
            *self.actor.parameters(),
            *self.critic.parameters(),
            *self.navigation_encoder.parameters(),
            *self.safety_head.parameters(),
        ):
            parameter.grad = None
        adapter_started = time.perf_counter()
        try:
            metrics.update(self._adapter_update())
        except torch.cuda.OutOfMemoryError:
            self.response_optimizer.zero_grad(set_to_none=True)
            previous_batch = self.adapter_batch_envs
            self.adapter_batch_envs = max(8, self.adapter_batch_envs // 2)
            self.adapter_oom_skips += 1
            metrics.update(
                {
                    "adapter_loss": float("nan"),
                    "adapter_updates": 0.0,
                    "adapter_oom_skips": float(self.adapter_oom_skips),
                }
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if self.logger:
                self.logger.warning(
                    "[P2NavPPO] response adapter update OOM; skipped this auxiliary "
                    f"step and reduced batch_envs {previous_batch}->{self.adapter_batch_envs}"
                )
        metrics["adapter_update_time_s"] = time.perf_counter() - adapter_started
        self.current_iteration += 1
        metrics["update_time_s"] = time.perf_counter() - started
        metrics["microbatch_frames"] = float(self.micro_sequences * 16)
        metrics["h2d_time_s"] = self._h2d_time_s
        self._h2d_time_s = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            total_memory = float(torch.cuda.get_device_properties(self.device).total_memory)
            peak_ratio = float(torch.cuda.max_memory_reserved(self.device)) / max(total_memory, 1.0)
            metrics["max_memory_reserved_ratio"] = peak_ratio
            if peak_ratio > 0.80 and self.micro_sequences > 2:
                self.micro_sequences = 2
                self.navigation_encoder.activation_checkpointing = True
                if self.logger:
                    self.logger.warning(
                        "[P2NavPPO] peak reserved memory exceeded 80%; next update "
                        "uses 32-frame CNN microbatches with activation checkpointing"
                    )
        metrics.update(self._training_monitor_metrics())
        self.rollout = self.rollout.reset(store_depth=self.cnn_unfrozen)
        self.rollout_invalid = False
        return metrics

    def reset_live_state(self) -> None:
        self.actor_hidden = None
        self.critic_hidden = None
        self.adapter_hidden = None
        self.low_level_encoder.reset_hidden_state(self.num_envs, self.device)
        self.command = P2CommandController(
            self.num_envs,
            self.device,
            slew_rate=self.command_slew_rate,
            slew_release_rate=self.command_slew_release_rate,
            command_transition_mode=self.command_transition_mode,
        )
        self.best_goal_distance.fill_(float("inf"))
        self.episode_start_goal_distance.fill_(float("inf"))
        self._initialize_navigation_reward_state()
        self.reset_since_tick.fill_(True)
        self.pending_tick = None
        self.last_tick_penalties = {}
        self.last_tick_diagnostics = {}
        if self.rollout is not None:
            self.rollout = self.rollout.reset(store_depth=self.cnn_unfrozen)
        if self.response_buffer is not None:
            self.response_buffer.clear_unfinished_history()
        self.curriculum_probe.reset_live_boundary()
        self.rollout_invalid = False

    def load_bundle(self, path: str, *, platform_model_id) -> str:
        if not self.training_enabled:
            raise RuntimeError("P2 eval runtime must use load_evaluation_bundle")
        raw = torch.load(path, weights_only=False, map_location="cpu")
        bundle, _ = normalize_kaiwu_train_bundle(raw)
        self._warn_platform_identity(bundle, platform_model_id)
        modules = bundle.get("modules", {})
        if raw.get("stage_type") == self.STAGE_TYPE:
            reward_version = (
                (bundle.get("contracts", {}).get("reward") or {}).get("version")
            )
            current_reward_version = p2_contract.reward_contract()["version"]
            command_version = (
                (bundle.get("contracts", {}).get("command") or {}).get("version")
            )
            current_command_version = p2_contract.command_contract()["version"]
            saved_training = bundle.get("contracts", {}).get("training")
            has_safety_head = isinstance(
                (modules.get("high_level", {}) or {}).get("navigation_safety_head"),
                dict,
            )
            safe_direction_exact = (
                reward_version == current_reward_version
                and command_version == current_command_version
                and saved_training == p2_contract.training_contract()
                and has_safety_head
            )
            if (
                self.load_mode == "p2_safe_direction_continue_warm_start"
                and not safe_direction_exact
            ):
                return self._load_p2_safe_direction_warm_start(
                    bundle, path, platform_model_id
                )
            if (
                self.load_mode == "p2_command_v2_expansion_warm_start"
                and command_version != current_command_version
            ):
                return self._load_p2_command_v2_warm_start(
                    bundle, path, platform_model_id
                )
            if (
                self.load_mode == "reward_v2_warm_start"
                and reward_version != current_reward_version
            ):
                return self._load_p2_reward_warm_start(bundle, path, platform_model_id)
            return self._load_exact_resume(bundle, platform_model_id, path)
        if (
            raw.get("stage_type") != "p15_response_adapter"
            or int(raw.get("schema_version", 0)) != KAIWU_TRAIN_SCHEMA_V2
        ):
            raise ValueError(
                "P2 bootstrap requires a P1.5 response adapter schema2 parent"
            )
        high = modules.get("high_level", {})
        if (
            high.get("component_status") != "adapter_only"
            or set(high) != {"component_status", "response_adapter"}
        ):
            raise ValueError("P2 bootstrap requires a P1.5 adapter_only schema2 parent")
        low = modules.get("low_level", {})
        self._migrate_parent_low_aux_modules(modules, low)
        encoder_state = (low.get("locomotion_encoder") or {}).get("state_dict")
        actor_state = (low.get("actor") or {}).get("state_dict")
        adapter_state = (high.get("response_adapter") or {}).get("state_dict")
        if not all(isinstance(state, dict) for state in (encoder_state, actor_state, adapter_state)):
            raise KeyError("P2 parent missing low-level encoder/actor or response adapter")
        expected_adapter_spec = response_adapter_spec()
        actual_adapter_spec = (high.get("response_adapter") or {}).get("spec")
        if actual_adapter_spec != expected_adapter_spec:
            raise ValueError(
                f"P2 parent response adapter spec mismatch: {actual_adapter_spec!r}"
            )
        validate_state_dict_finite(encoder_state, "P2 parent locomotion encoder")
        validate_state_dict_finite(actor_state, "P2 parent locomotion actor")
        validate_state_dict_finite(adapter_state, "P2 parent response adapter")
        self.low_level_encoder.load_state_dict(encoder_state, strict=True)
        self.low_level_actor.load_state_dict(actor_state, strict=True)
        self.response_adapter.load_state_dict(adapter_state, strict=True)
        self._validate_low_aux_modules(low, context="P2 parent low_level")
        self.navigation_encoder.copy_from_low_level_cnn(self.low_level_encoder.cnn)
        self.low_level_payload = low
        self.parent_optimizer_payload = bundle.get("optimizers", {})
        if not isinstance(self.parent_optimizer_payload, dict):
            raise ValueError("P2 parent optimizer payload must be a mapping")
        validate_state_dict_finite(
            self.parent_optimizer_payload,
            "P2 transparent parent optimizers",
        )
        self.parent_scheduler_payload = bundle.get("schedulers", {})
        parent_states = bundle.get("training_states", {})
        self.parent_training_payload = {
            key: value
            for key, value in parent_states.items()
            if key in {"global", "low_level"}
        }
        adapter_training = bundle.get("training_states", {}).get("response_adapter", {})
        parent_records = self.response_buffer.load_parent_completed_records(
            adapter_training.get("buffer", {}) if isinstance(adapter_training, dict) else {}
        )
        selected_identity = self._bundle_identity(bundle, platform_model_id, path)
        self.source_parent_model_id = selected_identity
        self.loaded_platform_model_id = selected_identity
        self.parent_checkpoint_sha256 = self._sha256(path)
        self.low_level_state_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        self.response_buffer.set_low_level_version(self.low_level_state_digest, 0)
        self.reset_live_state()
        if self.logger:
            message = (
                f"[P2NavPPO] bootstrap parent={selected_identity} "
                f"completed_parent_records={parent_records}"
            )
            if parent_records:
                self.logger.info(message)
            else:
                self.logger.warning(
                    message + "; parent replay unavailable, using Track-only Adapter batches"
                )
        return "bootstrap_high"

    def _load_incremental_actor_state(self, high: dict) -> dict[str, object]:
        leaf = high.get("actor")
        if not isinstance(leaf, dict):
            raise KeyError("P2 command-v2 warm start missing high-level actor")
        if leaf.get("class_name") != "P2NavigationActor":
            raise ValueError("P2 command-v2 warm start actor class mismatch")
        old_spec = leaf.get("spec")
        if not isinstance(old_spec, dict) or int(old_spec.get("action_dim", -1)) != 2:
            raise ValueError("P2 command-v2 warm start requires a two-axis P2 actor")
        old_state = leaf.get("state_dict")
        if not isinstance(old_state, dict):
            raise KeyError("P2 command-v2 warm start actor state_dict missing")
        validate_state_dict_finite(old_state, "P2 command-v2 source actor")
        current = self.actor.state_dict()
        new_keys = {"vy_log_std", "vy_mean_head.weight", "vy_mean_head.bias"}
        unexpected = set(old_state) - (set(current) - new_keys)
        missing = (set(current) - new_keys) - set(old_state)
        if unexpected or missing:
            raise ValueError(
                "P2 command-v2 actor migration key mismatch "
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        for name, value in old_state.items():
            if current[name].shape != value.shape:
                raise ValueError(
                    f"P2 command-v2 actor tensor shape mismatch {name}: "
                    f"saved={tuple(value.shape)} current={tuple(current[name].shape)}"
                )
            current[name] = value
        self.actor.load_state_dict(current, strict=True)
        return {
            "source_action_dim": 2,
            "target_action_dim": 3,
            "copied_actor_tensors": len(old_state),
            "initialized_actor_tensors": sorted(new_keys),
        }

    def _migrate_actor_optimizer_moments(self, old_state: dict) -> dict[str, object]:
        """Copy compatible Adam states by stable parameter name."""
        if not isinstance(old_state, dict):
            return {"restored": 0, "skipped": "source_optimizer_missing"}
        validate_state_dict_finite(old_state, "P2 command-v2 source actor optimizer")
        old_groups = old_state.get("param_groups")
        old_states = old_state.get("state")
        if not isinstance(old_groups, list) or not isinstance(old_states, dict):
            raise ValueError("P2 command-v2 source actor optimizer is malformed")

        new_name_by_object = {
            id(parameter): f"navigation_encoder.{name}"
            for name, parameter in self.navigation_encoder.named_parameters()
        }
        new_name_by_object.update(
            {
                id(parameter): f"actor.{name}"
                for name, parameter in self.actor.named_parameters()
            }
        )
        state_by_name: dict[str, dict] = {}
        skipped_source_groups: list[str] = []
        new_groups_by_name = {
            str(group.get("name")): group for group in self.actor_optimizer.param_groups
        }
        for old_group in old_groups:
            group_name = str(old_group.get("name", ""))
            old_ids = list(old_group.get("params", ()))
            if group_name.startswith("navigation_"):
                current_group = new_groups_by_name.get(group_name)
                if current_group is None or len(old_ids) != len(current_group["params"]):
                    raise ValueError(
                        f"P2 command-v2 navigation optimizer group mismatch {group_name}"
                    )
                names = [new_name_by_object[id(p)] for p in current_group["params"]]
            elif group_name == "actor":
                names = [
                    f"actor.{name}"
                    for name, _ in self.actor.named_parameters()
                    if name not in {"vy_log_std", "vy_mean_head.weight", "vy_mean_head.bias"}
                ]
                if len(old_ids) != len(names):
                    raise ValueError(
                        "P2 command-v2 legacy actor optimizer parameter count mismatch"
                    )
            else:
                if any(old_id in old_states for old_id in old_ids):
                    skipped_source_groups.append(group_name or "<unnamed>")
                continue
            for old_id, name in zip(old_ids, names):
                state = old_states.get(old_id)
                if isinstance(state, dict):
                    state_by_name[name] = state

        new_state = self.actor_optimizer.state_dict()
        restored_names = []
        for object_group, serialized_group in zip(
            self.actor_optimizer.param_groups, new_state["param_groups"]
        ):
            if str(object_group.get("name", "")) == "navigation_safety_head":
                continue
            for parameter, new_id in zip(
                object_group["params"], serialized_group["params"]
            ):
                name = new_name_by_object[id(parameter)]
                source = state_by_name.get(name)
                if source is None:
                    continue
                compatible = True
                copied = {}
                for key, value in source.items():
                    if torch.is_tensor(value):
                        if value.ndim and value.shape != parameter.shape:
                            compatible = False
                            break
                        copied[key] = value.detach().clone()
                    else:
                        copied[key] = value
                if compatible:
                    new_state["state"][new_id] = copied
                    restored_names.append(name)
        self.actor_optimizer.load_state_dict(new_state)
        return {
            "restored": len(restored_names),
            "restored_names": sorted(restored_names),
            "mapping_basis": "named_groups_and_legacy_module_parameter_order",
            "skipped_source_groups": sorted(skipped_source_groups),
            "new_zero_state_names": [
                "actor.vy_log_std",
                "actor.vy_mean_head.bias",
                "actor.vy_mean_head.weight",
            ],
        }

    @staticmethod
    def _restore_warm_start_rng(
        generator: torch.Generator,
        state: object,
        *,
        name: str,
    ) -> dict[str, str]:
        """Restore a compatible RNG without turning device format into a gate."""
        if not torch.is_tensor(state):
            return {"status": "fresh_seed", "reason": "source_state_missing"}
        try:
            generator.set_state(state.cpu())
        except RuntimeError as exc:
            return {
                "status": "fresh_seed",
                "reason": f"source_state_incompatible:{type(exc).__name__}",
            }
        return {"status": "restored", "source": name}

    def _migrate_same_shape_actor_optimizer_with_safety_head(
        self, old_state: dict
    ) -> dict[str, object]:
        """Restore existing named Adam groups while leaving SafetyHead fresh."""
        if not isinstance(old_state, dict):
            return {"restored": 0, "status": "source_optimizer_missing"}
        validate_state_dict_finite(old_state, "P2 safe-direction source actor optimizer")
        old_groups = old_state.get("param_groups")
        old_states = old_state.get("state")
        if not isinstance(old_groups, list) or not isinstance(old_states, dict):
            raise ValueError("P2 safe-direction source actor optimizer is malformed")
        old_by_name = {str(group.get("name", "")): group for group in old_groups}
        current_state = self.actor_optimizer.state_dict()
        restored = 0
        restored_groups = []
        for object_group, serialized_group in zip(
            self.actor_optimizer.param_groups, current_state["param_groups"]
        ):
            name = str(object_group.get("name", ""))
            if name == "navigation_safety_head":
                continue
            source_group = old_by_name.get(name)
            if source_group is None:
                raise ValueError(f"P2 safe-direction source optimizer missing group {name!r}")
            source_ids = list(source_group.get("params", ()))
            target_ids = list(serialized_group.get("params", ()))
            if len(source_ids) != len(target_ids):
                raise ValueError(
                    f"P2 safe-direction optimizer group shape mismatch {name!r}: "
                    f"source={len(source_ids)} target={len(target_ids)}"
                )
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
                                f"P2 safe-direction optimizer tensor mismatch {name}.{key}"
                            )
                        copied[key] = value.detach().clone()
                    else:
                        copied[key] = value
                current_state["state"][target_id] = copied
                restored += 1
            restored_groups.append(name)
        self.actor_optimizer.load_state_dict(current_state)
        return {
            "restored": restored,
            "restored_groups": restored_groups,
            "new_zero_state_group": "navigation_safety_head",
        }

    def _load_p2_safe_direction_warm_start(
        self, bundle: dict, path: str, platform_model_id
    ) -> str:
        """Continue a compatible three-axis P2 package with a fresh SafetyHead."""
        if bundle.get("bundle_kind") != "hierarchical_control_v3":
            raise ValueError("P2 safe-direction warm start bundle_kind mismatch")
        high = bundle.get("modules", {}).get("high_level", {})
        if high.get("component_status") != "complete":
            raise ValueError("P2 safe-direction warm start requires complete high-level state")
        for name, module, class_name, spec in (
            ("navigation_encoder", self.navigation_encoder, "NavigationEncoder", navigation_encoder_spec()),
            ("actor", self.actor, "P2NavigationActor", navigation_actor_spec()),
            ("critic", self.critic, "P2NavigationCritic", navigation_critic_spec()),
            ("response_adapter", self.response_adapter, "CommandResponseAdapter", response_adapter_spec()),
        ):
            self._load_leaf(
                high, name, module, class_name=class_name, spec=spec,
                context="P2 safe-direction warm start high_level",
            )
        low = bundle.get("modules", {}).get("low_level", {})
        if low.get("contract_version") != "low_level_v2":
            raise ValueError("P2 safe-direction warm start low-level contract mismatch")
        self._load_leaf(
            low, "locomotion_encoder", self.low_level_encoder,
            class_name="VisionEncoder", spec=self._low_encoder_spec(self.low_level_encoder),
            context="P2 safe-direction warm start low_level",
        )
        self._load_leaf(
            low, "actor", self.low_level_actor,
            class_name="Actor77Sequential", spec=self._low_actor_spec(),
            context="P2 safe-direction warm start low_level",
        )
        self._validate_low_aux_modules(low, context="P2 safe-direction warm start low_level")

        optimizers = bundle.get("optimizers", {})
        if not isinstance(optimizers, dict):
            raise ValueError("P2 safe-direction optimizer payload must be a mapping")
        migration = {
            "actor_optimizer": self._migrate_same_shape_actor_optimizer_with_safety_head(
                optimizers.get("high_level_actor")
            )
        }
        for name, optimizer in (
            ("high_level_critic", self.critic_optimizer),
            ("response_adapter", self.response_optimizer),
        ):
            state = optimizers.get(name)
            if not isinstance(state, dict):
                raise KeyError(f"P2 safe-direction warm start missing optimizer {name}")
            validate_state_dict_finite(state, f"P2 safe-direction optimizer {name}")
            optimizer.load_state_dict(state)

        states = bundle.get("training_states", {})
        high_state = states.get("high_level", {})
        response_state = states.get("response_adapter", {})
        if not isinstance(high_state, dict) or not isinstance(response_state, dict):
            raise ValueError("P2 safe-direction training states must be mappings")
        return_statistics = high_state.get("return_statistics")
        if not isinstance(return_statistics, dict):
            raise KeyError("P2 safe-direction warm start missing return statistics")
        self.return_statistics = dict(return_statistics)
        gait_state = high_state.get("gait_baseline")
        if isinstance(gait_state, dict):
            self.gait_baseline.load_state_dict(gait_state)
        self.actor_gradient_steps = int(high_state.get("actor_gradient_steps", 0))
        self.critic_gradient_steps = int(high_state.get("critic_gradient_steps", 0))
        self.adapter_gradient_steps = int(response_state.get("gradient_steps", 0))
        rng_report = {}
        for key, generator in (
            ("shuffle_rng_state", self.ppo_generator),
            ("action_rng_state", self.action_generator),
            ("vy_action_rng_state", self.vy_action_generator),
            ("neutral_rng_state", self.neutral_generator),
        ):
            rng_report[key] = self._restore_warm_start_rng(
                generator, high_state.get(key), name=f"high_level.{key}"
            )
        rng_report["adapter"] = self._restore_warm_start_rng(
            self.adapter_generator,
            response_state.get("rng_state"),
            name="response_adapter.rng_state",
        )
        migration["rng"] = rng_report
        buffer_state = response_state.get("buffer")
        if isinstance(buffer_state, dict) and buffer_state:
            self.response_buffer.load_checkpoint_state(buffer_state)
        self.stuck_positive_ema = float(
            response_state.get("stuck_positive_ema", self.stuck_positive_ema)
        )
        self.adapter_batch_envs = max(
            8, int(response_state.get("batch_envs", self.adapter_batch_envs))
        )
        self.adapter_oom_skips = int(response_state.get("oom_skips", 0))
        inherited_seconds = float(
            high_state.get(
                "lifetime_effective_seconds",
                high_state.get("effective_training_seconds", 0.0),
            )
        )
        self.lifetime_base_seconds = max(0.0, inherited_seconds)
        self.session_effective_seconds = 0.0
        self.effective_training_seconds = 0.0
        self.lifetime_effective_seconds = self.lifetime_base_seconds
        self.current_iteration = 0
        self.low_level_payload = low
        self.parent_optimizer_payload = bundle.get(
            "transparent_parent_optimizers", optimizers
        )
        self.parent_scheduler_payload = bundle.get(
            "transparent_parent_schedulers", bundle.get("schedulers", {})
        )
        self.parent_training_payload = bundle.get(
            "transparent_parent_training_states", states
        )
        selected_identity = self._bundle_identity(bundle, platform_model_id, path)
        self.source_parent_model_id = selected_identity
        self.loaded_platform_model_id = selected_identity
        self.parent_checkpoint_sha256 = self._sha256(path)
        self.low_level_state_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        self.response_buffer.set_low_level_version(self.low_level_state_digest, 0)
        self.optimizer_migration_report = migration
        self._apply_training_schedule(0.0)
        self.reset_live_state()
        if self.logger:
            self.logger.info(
                "[P2NavPPO] safe-direction warm start preserved policy, critic, "
                "adapter, return statistics and compatible optimizer moments; "
                f"lifetime_s={self.lifetime_base_seconds:.1f} SafetyHead=fresh"
            )
        return "p2_safe_direction_continue_warm_start"

    def _load_p2_command_v2_warm_start(
        self, bundle: dict, path: str, platform_model_id
    ) -> str:
        """Expand a compatible two-axis P2 package while rebuilding the critic."""
        if bundle.get("bundle_kind") != "hierarchical_control_v3":
            raise ValueError("P2 command-v2 warm start bundle_kind mismatch")
        high = bundle.get("modules", {}).get("high_level", {})
        if high.get("component_status") != "complete":
            raise ValueError("P2 command-v2 warm start requires complete high-level state")
        self._load_leaf(
            high,
            "navigation_encoder",
            self.navigation_encoder,
            class_name="NavigationEncoder",
            spec=navigation_encoder_spec(),
            context="P2 command-v2 warm start high_level",
        )
        migration = self._load_incremental_actor_state(high)
        self._load_leaf(
            high,
            "response_adapter",
            self.response_adapter,
            class_name="CommandResponseAdapter",
            spec=response_adapter_spec(),
            context="P2 command-v2 warm start high_level",
        )
        low = bundle.get("modules", {}).get("low_level", {})
        if low.get("contract_version") != "low_level_v2":
            raise ValueError("P2 command-v2 warm start low-level contract mismatch")
        self._load_leaf(
            low,
            "locomotion_encoder",
            self.low_level_encoder,
            class_name="VisionEncoder",
            spec=self._low_encoder_spec(self.low_level_encoder),
            context="P2 command-v2 warm start low_level",
        )
        self._load_leaf(
            low,
            "actor",
            self.low_level_actor,
            class_name="Actor77Sequential",
            spec=self._low_actor_spec(),
            context="P2 command-v2 warm start low_level",
        )
        self._validate_low_aux_modules(low, context="P2 command-v2 warm start low_level")

        old_optimizers = bundle.get("optimizers", {})
        migration["optimizer"] = self._migrate_actor_optimizer_moments(
            old_optimizers.get("high_level_actor")
            if isinstance(old_optimizers, dict)
            else None
        )
        response_state = bundle.get("training_states", {}).get("response_adapter", {})
        if isinstance(response_state, dict):
            buffer_state = response_state.get("buffer")
            if isinstance(buffer_state, dict) and buffer_state:
                self.response_buffer.load_checkpoint_state(buffer_state)
            response_optimizer = (
                old_optimizers.get("response_adapter")
                if isinstance(old_optimizers, dict)
                else None
            )
            if isinstance(response_optimizer, dict):
                validate_state_dict_finite(
                    response_optimizer, "P2 command-v2 source response optimizer"
                )
                self.response_optimizer.load_state_dict(response_optimizer)
            adapter_rng = response_state.get("rng_state")
            migration.setdefault("rng", {})["adapter"] = self._restore_warm_start_rng(
                self.adapter_generator,
                adapter_rng,
                name="response_adapter.rng_state",
            )
            self.adapter_gradient_steps = int(response_state.get("gradient_steps", 0))
            self.stuck_positive_ema = float(
                response_state.get("stuck_positive_ema", self.stuck_positive_ema)
            )
            self.adapter_batch_envs = max(
                8, int(response_state.get("batch_envs", self.adapter_batch_envs))
            )
            self.adapter_oom_skips = int(response_state.get("oom_skips", 0))

        old_high_state = bundle.get("training_states", {}).get("high_level", {})
        inherited_seconds = 0.0
        if isinstance(old_high_state, dict):
            inherited_seconds = float(
                old_high_state.get(
                    "lifetime_effective_seconds",
                    old_high_state.get("effective_training_seconds", 0.0),
                )
            )
            shuffle_rng = old_high_state.get("shuffle_rng_state")
            action_rng = old_high_state.get("action_rng_state")
            neutral_rng = old_high_state.get("neutral_rng_state")
            rng_report = migration.setdefault("rng", {})
            rng_report["shuffle"] = self._restore_warm_start_rng(
                self.ppo_generator,
                shuffle_rng,
                name="high_level.shuffle_rng_state",
            )
            rng_report["main_action"] = self._restore_warm_start_rng(
                self.action_generator,
                action_rng,
                name="high_level.action_rng_state",
            )
            rng_report["neutral"] = self._restore_warm_start_rng(
                self.neutral_generator,
                neutral_rng,
                name="high_level.neutral_rng_state",
            )
            rng_report["vy_action"] = {
                "status": "fresh_seed",
                "reason": "new_action_dimension",
            }
            self.actor_gradient_steps = int(old_high_state.get("actor_gradient_steps", 0))
        self.lifetime_base_seconds = max(0.0, inherited_seconds)
        self.session_effective_seconds = 0.0
        self.effective_training_seconds = 0.0
        self.lifetime_effective_seconds = self.lifetime_base_seconds
        self.current_iteration = 0
        self.critic_gradient_steps = 0
        self.return_statistics = {
            "count": 0,
            "mean": 0.0,
            "m2": 0.0,
            "value_normalization_enabled": False,
        }
        self.gait_baseline = P2GaitBaseline()
        self.low_level_payload = low
        self.parent_optimizer_payload = bundle.get(
            "transparent_parent_optimizers", bundle.get("optimizers", {})
        )
        self.parent_scheduler_payload = bundle.get(
            "transparent_parent_schedulers", bundle.get("schedulers", {})
        )
        self.parent_training_payload = bundle.get(
            "transparent_parent_training_states", bundle.get("training_states", {})
        )
        selected_identity = self._bundle_identity(bundle, platform_model_id, path)
        self.source_parent_model_id = selected_identity
        self.loaded_platform_model_id = selected_identity
        self.parent_checkpoint_sha256 = self._sha256(path)
        self.low_level_state_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        self.response_buffer.set_low_level_version(self.low_level_state_digest, 0)
        self.optimizer_migration_report = migration
        degraded_rngs = sorted(
            name
            for name, report in migration.get("rng", {}).items()
            if report.get("status") != "restored" and name != "vy_action"
        )
        if degraded_rngs and self.logger:
            self.logger.warning(
                "[P2NavPPO] warm-start RNG states were not device-compatible; "
                f"fresh configured seeds retained for {degraded_rngs}"
            )
        skipped_optimizer_groups = migration["optimizer"].get(
            "skipped_source_groups", []
        )
        if skipped_optimizer_groups and self.logger:
            self.logger.warning(
                "[P2NavPPO] warm-start optimizer groups could not be mapped by "
                f"the legacy parameter contract and were left fresh: {skipped_optimizer_groups}"
            )
        self._apply_training_schedule(0.0)
        self.reset_live_state()
        if self.logger:
            self.logger.info(
                "[P2NavPPO] command-v2 expansion warm start loaded two-axis "
                f"policy with lifetime_s={self.lifetime_base_seconds:.1f}; "
                "critic/return statistics rebuilt and vy head initialized"
            )
        return "p2_command_v2_expansion_warm_start"

    def _load_p2_reward_warm_start(
        self, bundle: dict, path: str, platform_model_id
    ) -> str:
        """Keep policy/adapter capability while discarding the old reward critic."""
        high = bundle.get("modules", {}).get("high_level", {})
        if (
            bundle.get("bundle_kind") != "hierarchical_control_v3"
            or high.get("component_status") != "complete"
        ):
            raise ValueError("P2 reward warm start requires a complete P2 schema2 bundle")
        for name, module, class_name, spec in (
            (
                "navigation_encoder",
                self.navigation_encoder,
                "NavigationEncoder",
                navigation_encoder_spec(),
            ),
            ("actor", self.actor, "P2NavigationActor", navigation_actor_spec()),
            (
                "response_adapter",
                self.response_adapter,
                "CommandResponseAdapter",
                response_adapter_spec(),
            ),
        ):
            self._load_leaf(
                high,
                name,
                module,
                class_name=class_name,
                spec=spec,
                context="P2 reward-v2 warm start high_level",
            )
        low = bundle.get("modules", {}).get("low_level", {})
        if low.get("contract_version") != "low_level_v2":
            raise ValueError("P2 reward warm start low-level contract mismatch")
        self._load_leaf(
            low,
            "locomotion_encoder",
            self.low_level_encoder,
            class_name="VisionEncoder",
            spec=self._low_encoder_spec(self.low_level_encoder),
            context="P2 reward-v2 warm start low_level",
        )
        self._load_leaf(
            low,
            "actor",
            self.low_level_actor,
            class_name="Actor77Sequential",
            spec=self._low_actor_spec(),
            context="P2 reward-v2 warm start low_level",
        )
        self._validate_low_aux_modules(low, context="P2 reward-v2 warm start low_level")
        response_state = bundle.get("training_states", {}).get("response_adapter", {})
        restored_adapter_gradient_steps = 0
        restored_stuck_positive_ema = self.stuck_positive_ema
        restored_adapter_batch_envs = self.adapter_batch_envs
        restored_adapter_oom_skips = self.adapter_oom_skips
        if isinstance(response_state, dict):
            buffer_state = response_state.get("buffer", {})
            if isinstance(buffer_state, dict) and buffer_state:
                self.response_buffer.load_checkpoint_state(buffer_state)
            optimizer_state = bundle.get("optimizers", {}).get("response_adapter")
            if isinstance(optimizer_state, dict):
                validate_state_dict_finite(
                    optimizer_state, "P2 reward-v2 warm start response optimizer"
                )
                self.response_optimizer.load_state_dict(optimizer_state)
            scheduler_state = bundle.get("schedulers", {}).get("response_adapter")
            if isinstance(scheduler_state, dict):
                self.response_scheduler.load_state_dict(scheduler_state)
            adapter_rng = response_state.get("rng_state")
            if torch.is_tensor(adapter_rng):
                self.adapter_generator.set_state(adapter_rng.cpu())
            restored_adapter_gradient_steps = int(
                response_state.get("gradient_steps", 0)
            )
            restored_stuck_positive_ema = float(
                response_state.get("stuck_positive_ema", self.stuck_positive_ema)
            )
            restored_adapter_batch_envs = max(
                8, int(response_state.get("batch_envs", self.adapter_batch_envs))
            )
            restored_adapter_oom_skips = int(response_state.get("oom_skips", 0))
        self.low_level_payload = low
        self.parent_optimizer_payload = bundle.get(
            "transparent_parent_optimizers", bundle.get("optimizers", {})
        )
        self.parent_scheduler_payload = bundle.get(
            "transparent_parent_schedulers", bundle.get("schedulers", {})
        )
        self.parent_training_payload = bundle.get(
            "transparent_parent_training_states", bundle.get("training_states", {})
        )
        selected_identity = self._bundle_identity(bundle, platform_model_id, path)
        self.source_parent_model_id = selected_identity
        self.loaded_platform_model_id = selected_identity
        self.parent_checkpoint_sha256 = self._sha256(path)
        self.low_level_state_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        self.response_buffer.set_low_level_version(self.low_level_state_digest, 0)
        self.current_iteration = 0
        self.actor_gradient_steps = 0
        self.critic_gradient_steps = 0
        self.adapter_gradient_steps = restored_adapter_gradient_steps
        self.stuck_positive_ema = restored_stuck_positive_ema
        self.adapter_batch_envs = restored_adapter_batch_envs
        self.adapter_oom_skips = restored_adapter_oom_skips
        self.effective_training_seconds = 0.0
        self.cnn_unfrozen = True
        self.gait_baseline = P2GaitBaseline()
        self.return_statistics = {
            "count": 0,
            "mean": 0.0,
            "m2": 0.0,
            "value_normalization_enabled": False,
        }
        self._apply_training_schedule(0.0)
        self.reset_live_state()
        if self.logger:
            self.logger.info(
                "[P2NavPPO] reward-v8 warm start loaded policy/adapter and rebuilt "
                "critic, actor/CNN optimizer schedule, return statistics and gait baseline"
            )
        return "p2_reward_v2_warm_start"

    def _load_exact_resume(
        self,
        bundle: dict,
        platform_model_id,
        path: str,
    ) -> str:
        if bundle.get("bundle_kind") != "hierarchical_control_v3":
            raise ValueError("P2 exact resume requires bundle_kind=hierarchical_control_v3")
        high = bundle.get("modules", {}).get("high_level", {})
        if high.get("component_status") != "complete":
            raise ValueError("P2 exact resume requires complete high-level component")
        if high.get("contract_version") != "high_level_continuous_v2":
            raise ValueError("P2 exact resume high-level contract mismatch")
        saved_reward = (bundle.get("contracts", {}).get("reward") or {}).get("version")
        if saved_reward != p2_contract.reward_contract()["version"]:
            raise ValueError(
                f"P2 exact resume reward contract mismatch: {saved_reward!r}"
            )
        saved_training = bundle.get("contracts", {}).get("training")
        if saved_training != p2_contract.training_contract():
            raise ValueError("P2 exact resume training contract mismatch")
        self._load_leaf(
            high,
            "navigation_encoder",
            self.navigation_encoder,
            class_name="NavigationEncoder",
            spec=navigation_encoder_spec(),
            context="P2 exact resume high_level",
        )
        self._load_leaf(
            high,
            "navigation_safety_head",
            self.safety_head,
            class_name="NavigationSafetyHead",
            spec=navigation_safety_head_spec(),
            context="P2 exact resume high_level",
        )
        self._load_leaf(
            high,
            "actor",
            self.actor,
            class_name="P2NavigationActor",
            spec=navigation_actor_spec(),
            context="P2 exact resume high_level",
        )
        self._load_leaf(
            high,
            "critic",
            self.critic,
            class_name="P2NavigationCritic",
            spec=navigation_critic_spec(),
            context="P2 exact resume high_level",
        )
        self._load_leaf(
            high,
            "response_adapter",
            self.response_adapter,
            class_name="CommandResponseAdapter",
            spec=response_adapter_spec(),
            context="P2 exact resume high_level",
        )
        low = bundle.get("modules", {}).get("low_level", {})
        if low.get("contract_version") != "low_level_v2":
            raise ValueError("P2 exact resume low-level contract mismatch")
        self._load_leaf(
            low,
            "locomotion_encoder",
            self.low_level_encoder,
            class_name="VisionEncoder",
            spec=self._low_encoder_spec(self.low_level_encoder),
            context="P2 exact resume low_level",
        )
        self._validate_low_aux_modules(low, context="P2 exact resume low_level")
        self._load_leaf(
            low,
            "actor",
            self.low_level_actor,
            class_name="Actor77Sequential",
            spec=self._low_actor_spec(),
            context="P2 exact resume low_level",
        )
        optimizers = bundle.get("optimizers", {})
        schedulers = bundle.get("schedulers", {})
        for name, optimizer in (
            ("high_level_actor", self.actor_optimizer),
            ("high_level_critic", self.critic_optimizer),
            ("response_adapter", self.response_optimizer),
        ):
            if not isinstance(optimizers.get(name), dict):
                raise KeyError(f"P2 exact resume missing optimizer {name}")
            validate_state_dict_finite(optimizers[name], f"P2 exact resume optimizer {name}")
            optimizer.load_state_dict(optimizers[name])
        for name, scheduler in (
            ("high_level_actor", self.actor_scheduler),
            ("high_level_critic", self.critic_scheduler),
            ("response_adapter", self.response_scheduler),
        ):
            if not isinstance(schedulers.get(name), dict):
                raise KeyError(f"P2 exact resume missing scheduler {name}")
            scheduler.load_state_dict(schedulers[name])
        states = bundle.get("training_states", {})
        global_state = states.get("global", {})
        high_state = states.get("high_level", {})
        response_state = states.get("response_adapter", {})
        if not isinstance(global_state, dict) or global_state.get("train_scope") != (
            "high_level_and_response_adapter"
        ):
            raise ValueError("P2 exact resume train_scope mismatch")
        for key in (
            "effective_training_seconds",
            "session_effective_seconds",
            "lifetime_effective_seconds",
            "lifetime_base_seconds",
            "frame_count",
            "iteration",
            "actor_gradient_steps",
            "critic_gradient_steps",
            "shuffle_rng_state",
            "action_rng_state",
            "vy_action_rng_state",
            "neutral_rng_state",
            "nav_ticks",
            "cnn_unfrozen",
            "skipped_nonfinite",
            "nonfinite_action_fallbacks",
            "microbatch_frames",
            "invalid_transition_count",
            "optimizer_phase",
            "entropy_coefficient",
            "return_statistics",
            "gait_baseline",
            "optimizer_migration_report",
        ):
            if key not in high_state:
                raise KeyError(f"P2 exact resume missing high-level training state {key}")
        for key in (
            "gradient_steps",
            "rng_state",
            "buffer",
            "stuck_positive_ema",
            "oom_skips",
            "batch_envs",
        ):
            if key not in response_state:
                raise KeyError(
                    f"P2 exact resume missing response adapter training state {key}"
                )
        self.session_effective_seconds = float(high_state["session_effective_seconds"])
        self.effective_training_seconds = self.session_effective_seconds
        self.lifetime_effective_seconds = float(high_state["lifetime_effective_seconds"])
        self.lifetime_base_seconds = float(high_state["lifetime_base_seconds"])
        if abs(
            self.lifetime_effective_seconds
            - (self.lifetime_base_seconds + self.session_effective_seconds)
        ) > 1.0e-3:
            raise ValueError("P2 exact resume training clocks are inconsistent")
        self.current_iteration = int(high_state["iteration"])
        self.frame_count = int(high_state["frame_count"])
        if self.frame_count < 0:
            raise ValueError("P2 exact resume frame_count must be non-negative")
        frame_remainder = self.frame_count % self.nav_period_frames
        if frame_remainder:
            alignment_delta = self.nav_period_frames - frame_remainder
            self.frame_count += alignment_delta
            if self.logger:
                self.logger.warning(
                    "[P2NavPPO] exact resume discarded an unfinished navigation "
                    f"tick and aligned frame_count by {alignment_delta} frames"
                )
        self.actor_gradient_steps = int(high_state["actor_gradient_steps"])
        self.critic_gradient_steps = int(high_state["critic_gradient_steps"])
        self.nav_ticks = int(high_state["nav_ticks"])
        self.skipped_nonfinite = int(high_state["skipped_nonfinite"])
        self.nonfinite_action_fallbacks = int(
            high_state["nonfinite_action_fallbacks"]
        )
        saved_microbatch_frames = int(high_state["microbatch_frames"])
        if saved_microbatch_frames not in (32, 64):
            raise ValueError(
                f"P2 exact resume invalid CNN microbatch frames: {saved_microbatch_frames}"
            )
        self.micro_sequences = saved_microbatch_frames // self.rollout.sequence_length
        self.navigation_encoder.activation_checkpointing = self.micro_sequences <= 2
        self.adapter_oom_skips = int(response_state["oom_skips"])
        self.adapter_batch_envs = max(8, int(response_state["batch_envs"]))
        self.adapter_gradient_steps = int(response_state["gradient_steps"])
        for key in (
            "shuffle_rng_state",
            "action_rng_state",
            "vy_action_rng_state",
            "neutral_rng_state",
        ):
            if not torch.is_tensor(high_state[key]):
                raise ValueError(f"P2 exact resume invalid RNG state {key}")
        self.ppo_generator.set_state(high_state["shuffle_rng_state"].cpu())
        self.action_generator.set_state(high_state["action_rng_state"].cpu())
        self.vy_action_generator.set_state(high_state["vy_action_rng_state"].cpu())
        self.neutral_generator.set_state(high_state["neutral_rng_state"].cpu())
        self.invalid_transition_count = int(high_state["invalid_transition_count"])
        return_statistics = high_state["return_statistics"]
        if not isinstance(return_statistics, dict):
            raise ValueError("P2 exact resume return statistics must be a mapping")
        self.return_statistics = {
            "count": int(return_statistics.get("count", 0)),
            "mean": float(return_statistics.get("mean", 0.0)),
            "m2": float(return_statistics.get("m2", 0.0)),
            "value_normalization_enabled": bool(
                return_statistics.get("value_normalization_enabled", False)
            ),
        }
        self.gait_baseline.load_state_dict(high_state["gait_baseline"])
        if not isinstance(high_state["optimizer_migration_report"], dict):
            raise ValueError("P2 exact resume optimizer migration report must be a mapping")
        self.optimizer_migration_report = dict(
            high_state["optimizer_migration_report"]
        )
        adapter_rng = response_state.get("rng_state")
        if not torch.is_tensor(adapter_rng):
            raise KeyError("P2 exact resume missing response adapter RNG")
        self.adapter_generator.set_state(adapter_rng.cpu())
        buffer_mode = self.response_buffer.load_checkpoint_state(
            response_state.get("buffer", {})
        )
        if buffer_mode != "completed_records_restored_history_reset":
            raise ValueError(f"P2 exact resume response buffer mismatch: {buffer_mode}")
        if self.response_buffer.resized_completed_records and self.logger:
            self.logger.warning(
                "[P2NavPPO] completed response records sliced for environment-count "
                f"fallback records={self.response_buffer.resized_completed_records} "
                f"num_envs={self.num_envs}"
            )
        self.stuck_positive_ema = float(response_state["stuck_positive_ema"])
        self._apply_training_schedule(self.session_effective_seconds)
        if self.cnn_unfrozen != bool(high_state["cnn_unfrozen"]):
            raise ValueError("P2 exact resume CNN schedule state mismatch")
        if self.optimizer_phase != str(high_state["optimizer_phase"]):
            raise ValueError("P2 exact resume optimizer phase mismatch")
        if abs(self.entropy_coefficient - float(high_state["entropy_coefficient"])) > 1e-9:
            raise ValueError("P2 exact resume entropy schedule mismatch")
        self.low_level_payload = low
        for key in (
            "transparent_parent_optimizers",
            "transparent_parent_schedulers",
            "transparent_parent_training_states",
        ):
            if not isinstance(bundle.get(key), dict):
                raise KeyError(f"P2 exact resume missing {key}")
        self.parent_optimizer_payload = bundle["transparent_parent_optimizers"]
        self.parent_scheduler_payload = bundle["transparent_parent_schedulers"]
        self.parent_training_payload = bundle["transparent_parent_training_states"]
        if not all(
            isinstance(payload, dict)
            for payload in (
                self.parent_optimizer_payload,
                self.parent_scheduler_payload,
                self.parent_training_payload,
            )
        ):
            raise ValueError("P2 exact resume transparent parent state must be mappings")
        validate_state_dict_finite(
            self.parent_optimizer_payload,
            "P2 exact resume transparent parent optimizers",
        )
        lineage = bundle.get("lineage", {})
        self.source_parent_model_id = lineage.get("source_parent_model_id")
        self.parent_checkpoint_sha256 = lineage.get("parent_checkpoint_sha256")
        saved_low_digest = lineage.get("low_level_state_digest")
        actual_low_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        if saved_low_digest not in (None, actual_low_digest) and self.logger:
            self.logger.warning(
                "[P2NavPPO] exact resume low-level digest mismatch; continuing "
                f"saved={saved_low_digest} actual={actual_low_digest}"
            )
        self.low_level_state_digest = actual_low_digest
        self.response_buffer.set_low_level_version(actual_low_digest, 0)
        self.curriculum_probe.load_state_dict(
            bundle.get("curriculum_diagnostics", {})
        )
        self.loaded_platform_model_id = self._bundle_identity(
            bundle, platform_model_id, path
        )
        self.resume_loaded = True
        self.reset_live_state()
        return "exact_resume_history_reset"

    def load_evaluation_bundle(self, path: str, *, platform_model_id) -> str:
        if self.training_enabled:
            raise RuntimeError("P2 training runtime cannot use evaluation-only loader")
        raw = torch.load(path, weights_only=False, map_location="cpu")
        bundle, _ = normalize_kaiwu_train_bundle(raw)
        self._warn_platform_identity(bundle, platform_model_id)
        if raw.get("stage_type") != self.STAGE_TYPE:
            raise ValueError("P2 evaluation requires a p2_nav_ppo checkpoint")
        if bundle.get("bundle_kind") != "hierarchical_control_v3":
            raise ValueError("P2 evaluation bundle_kind mismatch")
        modules = bundle.get("modules", {})
        low = modules.get("low_level", {})
        high = modules.get("high_level", {})
        if low.get("contract_version") != "low_level_v2":
            raise ValueError("P2 evaluation low-level contract mismatch")
        if (
            high.get("contract_version") != "high_level_continuous_v2"
            or high.get("component_status") != "complete"
        ):
            raise ValueError("P2 evaluation requires complete high-level contract")
        self._load_leaf(
            low,
            "locomotion_encoder",
            self.low_level_encoder,
            class_name="VisionEncoder",
            spec=self._low_encoder_spec(self.low_level_encoder),
            context="P2 evaluation low_level",
        )
        self._load_leaf(
            low,
            "actor",
            self.low_level_actor,
            class_name="Actor77Sequential",
            spec=self._low_actor_spec(),
            context="P2 evaluation low_level",
        )
        for name, module, class_name, spec in (
            (
                "navigation_encoder",
                self.navigation_encoder,
                "NavigationEncoder",
                navigation_encoder_spec(),
            ),
            ("actor", self.actor, "P2NavigationActor", navigation_actor_spec()),
            (
                "response_adapter",
                self.response_adapter,
                "CommandResponseAdapter",
                response_adapter_spec(),
            ),
        ):
            self._load_leaf(
                high,
                name,
                module,
                class_name=class_name,
                spec=spec,
                context="P2 evaluation high_level",
            )
        self.loaded_platform_model_id = self._bundle_identity(
            bundle, platform_model_id, path
        )
        self.low_level_state_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        self.reset_live_state()
        return "evaluate_full_modules_only"

    def load_p3_evaluation_bundle(self, path: str, *, platform_model_id) -> str:
        """Load a P3 ``p3_standard_joint`` package for Track evaluation.

        Mirrors ``load_evaluation_bundle`` but accepts the P3 package and runs
        the shared ``validate_p3_eval_bundle`` structural validator first.
        SafetyHead, high-level Critic, optimizers, schedulers, PPO storage and
        the ResponseBuffer are never created or loaded; only the frozen
        low-level VisionEncoder/Actor plus NavigationEncoder, three-axis Actor
        and ResponseAdapter are instantiated (evaluate_full module-only).
        """
        if self.training_enabled:
            raise RuntimeError("P3 evaluation must use a training=False runtime")
        raw = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(raw, dict):
            raise ValueError("P3 evaluation checkpoint must be a mapping")
        self._warn_platform_identity(raw, platform_model_id)
        disposition = validate_p3_eval_bundle(raw, mode="track")
        bundle, _ = normalize_kaiwu_train_bundle(raw)
        modules = bundle.get("modules", {})
        low = modules.get("low_level", {})
        high = modules.get("high_level", {})
        self._load_leaf(
            low,
            "locomotion_encoder",
            self.low_level_encoder,
            class_name="VisionEncoder",
            spec=self._low_encoder_spec(self.low_level_encoder),
            context="P3 track evaluation low_level",
        )
        self._load_leaf(
            low,
            "actor",
            self.low_level_actor,
            class_name="Actor77Sequential",
            spec=self._low_actor_spec(),
            context="P3 track evaluation low_level",
        )
        for name, module, class_name, spec in (
            (
                "navigation_encoder",
                self.navigation_encoder,
                "NavigationEncoder",
                navigation_encoder_spec(),
            ),
            ("actor", self.actor, "P2NavigationActor", navigation_actor_spec()),
            (
                "response_adapter",
                self.response_adapter,
                "CommandResponseAdapter",
                response_adapter_spec(),
            ),
        ):
            self._load_leaf(
                high,
                name,
                module,
                class_name=class_name,
                spec=spec,
                context="P3 track evaluation high_level",
            )
        self.loaded_platform_model_id = self._bundle_identity(
            bundle, platform_model_id, path
        )
        self.low_level_state_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        self._p3_eval_phase_label = disposition["phase_label"]
        self.reset_live_state()
        return "evaluate_full_modules_only"

    @property
    def current_phase(self) -> str:
        return str(
            p2_contract.training_schedule(self.session_effective_seconds)["phase"]
        )

    def save_training_bundle(self, path: str, *, platform_model_id) -> str:
        if not self.training_enabled:
            raise RuntimeError("P2 eval runtime cannot save a training bundle")
        module_states = {
            "low_level_encoder": self.low_level_encoder.state_dict(),
            "low_level_actor": self.low_level_actor.state_dict(),
            "navigation_encoder": self.navigation_encoder.state_dict(),
            "navigation_safety_head": self.safety_head.state_dict(),
            "high_level_actor": self.actor.state_dict(),
            "high_level_critic": self.critic.state_dict(),
            "response_adapter": self.response_adapter.state_dict(),
        }
        optimizer_states = {
            "high_level_actor": self.actor_optimizer.state_dict(),
            "high_level_critic": self.critic_optimizer.state_dict(),
            "response_adapter": self.response_optimizer.state_dict(),
        }
        for name, state in module_states.items():
            validate_state_dict_finite(state, f"P2 save module {name}")
        for name, state in optimizer_states.items():
            validate_state_dict_finite(state, f"P2 save optimizer {name}")
        validate_state_dict_finite(
            self.parent_optimizer_payload,
            "P2 save transparent parent optimizers",
        )
        current_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        low_drift = self.low_level_state_digest not in (None, current_digest)
        if low_drift and self.logger:
            self.logger.warning(
                "[P2NavPPO] WARNING frozen low-level digest drift detected; checkpoint marked"
            )
        self.low_level_state_digest = current_digest
        low = dict(self.low_level_payload)
        for legacy_key in ("actor_state_dict", "encoder_state_dict", "policy_state_dict"):
            low.pop(legacy_key, None)
        low["contract_version"] = "low_level_v2"
        low["locomotion_encoder"] = self._leaf(
            "VisionEncoder",
            self._low_encoder_spec(self.low_level_encoder),
            module_states["low_level_encoder"],
        )
        low["actor"] = self._leaf(
            "Actor77Sequential",
            self._low_actor_spec(),
            module_states["low_level_actor"],
        )
        if isinstance(low.get("critic"), dict):
            low["critic"] = self._leaf(
                "VisualCritic",
                self._low_critic_spec(),
                low["critic"].get("state_dict", {}),
            )
        self._validate_low_aux_modules(low, context="P2 save low_level")
        feedback_contract = {
            "profile": self.config.get("feedback_profile", {}),
            "implementation_sha256": feedback_implementation_digest(),
        }
        low_training_state = dict(
            self.parent_training_payload.get("low_level", {})
            if isinstance(self.parent_training_payload.get("low_level", {}), dict)
            else {}
        )
        low_training_state.update(
            {
                "frozen": True,
                "digest_drift": low_drift,
                "transparent_parent_state_preserved": True,
            }
        )
        payload = {
            "format": KAIWU_TRAIN_FORMAT,
            "schema_version": KAIWU_TRAIN_SCHEMA_V2,
            "bundle_kind": "hierarchical_control_v3",
            "stage_type": self.STAGE_TYPE,
            "model_spec": {
                "proprio_dim": 45,
                "scan_dim": 256,
                "depth_height": 180,
                "depth_width": 320,
                "depth_channels": 1,
                "latent_dim": 32,
                "action_dim": 12,
                "goal_dim": 0,
            },
            "phase_label": self.current_phase,
            "platform_model_id": str(platform_model_id),
            "deployable": False,
            "modules": {
                "low_level": low,
                "high_level": {
                    "contract_version": "high_level_continuous_v2",
                    "component_status": "complete",
                    "navigation_encoder": self._leaf(
                        "NavigationEncoder",
                        navigation_encoder_spec(),
                        module_states["navigation_encoder"],
                    ),
                    "navigation_safety_head": self._leaf(
                        "NavigationSafetyHead",
                        navigation_safety_head_spec(),
                        module_states["navigation_safety_head"],
                    ),
                    "actor": self._leaf(
                        "P2NavigationActor",
                        navigation_actor_spec(),
                        module_states["high_level_actor"],
                    ),
                    "critic": self._leaf(
                        "P2NavigationCritic",
                        navigation_critic_spec(),
                        module_states["high_level_critic"],
                    ),
                    "response_adapter": self._leaf(
                        "CommandResponseAdapter",
                        response_adapter_spec(),
                        module_states["response_adapter"],
                    ),
                },
            },
            "optimizers": {
                "high_level_actor": optimizer_states["high_level_actor"],
                "high_level_critic": optimizer_states["high_level_critic"],
                "response_adapter": optimizer_states["response_adapter"],
            },
            "transparent_parent_optimizers": self.parent_optimizer_payload,
            "transparent_parent_schedulers": self.parent_scheduler_payload,
            "transparent_parent_training_states": self.parent_training_payload,
            "schedulers": {
                "high_level_actor": self.actor_scheduler.state_dict(),
                "high_level_critic": self.critic_scheduler.state_dict(),
                "response_adapter": self.response_scheduler.state_dict(),
            },
            "training_states": {
                "global": {
                    "compound_schedule_phase": self.current_phase,
                    "last_active_scope": "high_level_and_response_adapter",
                    "train_scope": "high_level_and_response_adapter",
                },
                "low_level": low_training_state,
                "high_level": {
                    "iteration": self.current_iteration,
                    "actor_gradient_steps": self.actor_gradient_steps,
                    "critic_gradient_steps": self.critic_gradient_steps,
                    "nav_ticks": self.nav_ticks,
                    "frame_count": self.frame_count,
                    "effective_training_seconds": self.effective_training_seconds,
                    "session_effective_seconds": self.session_effective_seconds,
                    "lifetime_effective_seconds": self.lifetime_effective_seconds,
                    "lifetime_base_seconds": self.lifetime_base_seconds,
                    "cnn_unfrozen": self.cnn_unfrozen,
                    "shuffle_rng_state": self.ppo_generator.get_state(),
                    "action_rng_state": self.action_generator.get_state().cpu(),
                    "vy_action_rng_state": self.vy_action_generator.get_state().cpu(),
                    "neutral_rng_state": self.neutral_generator.get_state().cpu(),
                    "skipped_nonfinite": self.skipped_nonfinite,
                    "nonfinite_action_fallbacks": self.nonfinite_action_fallbacks,
                    "microbatch_frames": self.micro_sequences * 16,
                    "invalid_transition_count": self.invalid_transition_count,
                    "optimizer_phase": self.optimizer_phase,
                    "entropy_coefficient": self.entropy_coefficient,
                    "return_statistics": dict(self.return_statistics),
                    "gait_baseline": self.gait_baseline.state_dict(
                        parent_sha256=self.parent_checkpoint_sha256
                    ),
                    "optimizer_migration_report": dict(
                        self.optimizer_migration_report
                    ),
                },
                "response_adapter": {
                    "gradient_steps": self.adapter_gradient_steps,
                    "rng_state": self.adapter_generator.get_state().cpu(),
                    "stuck_positive_ema": self.stuck_positive_ema,
                    "oom_skips": self.adapter_oom_skips,
                    "batch_envs": self.adapter_batch_envs,
                    "buffer": self.response_buffer.checkpoint_state(),
                },
            },
            "contracts": {
                **p2_contract.contract_metadata(),
                "feedback": feedback_contract,
                "feedback_digest": p2_contract.stable_digest(feedback_contract),
                "critic_transport": {
                    "wire_dim": p2_contract.PRIVILEGED_WIRE_DIM,
                    "critic_dim": p2_contract.CRITIC_OBS_DIM,
                    "response_aux_dim": p2_contract.RESPONSE_AUX_DIM,
                    "diagnostic_aux_dim": p2_contract.DIAGNOSTIC_AUX_DIM,
                    "worker_aux_dim": p2_contract.WORKER_AUX_DIM,
                },
            },
            "lineage": {
                "source_parent_model_id": self.source_parent_model_id,
                "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
                "low_level_state_digest": self.low_level_state_digest,
                "navigation_encoder_state_digest": self._module_digest(
                    (("navigation", self.navigation_encoder),)
                ),
                "command_v2_migration": dict(self.optimizer_migration_report),
            },
            "capabilities": {
                "envelope_type": "piecewise_union_v1",
                "continuous_vx_vy_wz": True,
                "deployable": False,
            },
            "curriculum_diagnostics": self.curriculum_probe.state_dict(),
        }
        payload["modules"]["high_level"]["navigation_safety_head"][
            "training_only"
        ] = True
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        temporary = os.path.join(directory, f".{os.path.basename(path)}.{uuid4().hex}.tmp")
        try:
            torch.save(payload, temporary)
            if not os.path.isfile(temporary) or os.path.getsize(temporary) <= 0:
                raise IOError(f"P2 checkpoint temporary file invalid: {temporary}")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        return self._sha256(path)

    def memory_metrics(self) -> dict[str, float]:
        result = {"pinned_depth_bytes": float(0 if self.rollout.depth is None else self.rollout.depth.numel() * 2)}
        if self.device.type == "cuda" and torch.cuda.is_available():
            result.update(
                {
                    "memory_allocated": float(torch.cuda.memory_allocated(self.device)),
                    "memory_reserved": float(torch.cuda.memory_reserved(self.device)),
                    "max_memory_allocated": float(torch.cuda.max_memory_allocated(self.device)),
                    "max_memory_reserved": float(torch.cuda.max_memory_reserved(self.device)),
                }
            )
        return result
