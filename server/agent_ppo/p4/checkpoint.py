#!/usr/bin/env python3
"""P4 live-state reset, bundle migration, and checkpoint persistence."""

from __future__ import annotations

import copy
import os
from uuid import uuid4

import torch

from agent_ppo.checkpoint_io import (
    normalize_kaiwu_train_bundle,
    validate_p3_eval_bundle,
    validate_state_dict_finite,
)
from agent_ppo.feature import p2_contract, p3_contract, p4_contract
from agent_ppo.feature.feedback_emulator import feedback_implementation_digest
from agent_ppo.model.p2_high_level import (
    navigation_actor_spec,
    navigation_encoder_spec,
    navigation_safety_head_spec,
    p4_actor_stuck_head_spec,
)
from agent_ppo.model.response_adapter import response_adapter_spec
from agent_ppo.p4.profiles import (
    PROFILE_FULL_TRACK,
    PROFILE_MAZE_CLOSED_LOOP_V3,
    PROFILE_MAZE_CREDIT_REPAIR,
    PROFILE_MAZE_INSTANT_COMMAND_R4,
    PROFILE_MAZE_INSTANT_REPAIR2H,
)


class P4CheckpointMixin:
    """Behavior-preserving methods extracted from AlgorithmP4NavPPO."""

    def reset_live_state(self) -> None:
        super().reset_live_state()
        if hasattr(self, "goal_belief"):
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.goal_belief.reset(mask)
            self.camera_state.reset(mask, randomize_capture_phase=False)
            self._cached_low_frame_id.fill_(-1)
            self._teacher_hidden = None
            self._parent_anchor_hidden = None
            self._parent_anchor_normalized_mean.zero_()
            self._parent_anchor_log_std.zero_()
            self._parent_anchor_mask.zero_()
            self._previous_policy_command.zero_()
            self._previous_exec_command.zero_()
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
                training_profile=self.training_profile,
            )
        )
        if self.training_profile == PROFILE_MAZE_INSTANT_REPAIR2H:
            self.response_buffer.enable_p4_current_only_replay()
        else:
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
                raise ValueError(f"P4 previous actor optimizer missing group {name!r}")
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
        validate_state_dict_finite(current_state, "P4 migrated actor optimizer")
        return current_state, {
            "mapping_basis": "stable_optimizer_group_name_and_parameter_order",
            "restored_parameters": restored_parameters,
            "restored_groups": restored_groups,
            "fresh_groups": fresh_groups,
        }

    def _freeze_parent_anchor_from_current_actor(self, *, source_sha256: str) -> None:
        """Capture the warm-start policy once; never refresh it from online weights."""
        self._parent_anchor_actor.load_state_dict(self.actor.state_dict(), strict=True)
        self._parent_anchor_actor.eval()
        for parameter in self._parent_anchor_actor.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self._parent_anchor_hidden = None
        self.parent_anchor_source_sha256 = str(source_sha256)
        self.parent_anchor_digest = self._module_digest(
            (("parent_actor_anchor", self._parent_anchor_actor),)
        )

    def _validate_p4_resume_contract(self, bundle) -> bool:
        contracts = bundle.get("contracts", {})
        exact_training_contract = p4_contract.training_contract(
            self.stuck_reset_contract,
            self.training_profile,
        )
        exact_reward_contract = p4_contract.reward_contract(
            self.stuck_reset_contract,
            self.training_profile,
        )
        exact_command_contract = p4_contract.command_contract(self.training_profile)
        saved_training = contracts.get("training")
        saved_reward = contracts.get("reward")
        saved_command = contracts.get("command")
        if (
            isinstance(saved_training, dict)
            and saved_training.get("version") == exact_training_contract["version"]
            and saved_training != exact_training_contract
        ):
            raise ValueError("P4 exact resume training contract mismatch")
        if (
            isinstance(saved_training, dict)
            and saved_training.get("version") == exact_training_contract["version"]
            and saved_reward != exact_reward_contract
        ):
            raise ValueError("P4 exact resume reward contract mismatch")
        if (
            isinstance(saved_training, dict)
            and saved_training.get("version") == exact_training_contract["version"]
            and saved_command != exact_command_contract
        ):
            raise ValueError("P4 exact resume command contract mismatch")
        exact_compatible = (
            saved_training == exact_training_contract
            and saved_reward == exact_reward_contract
            and saved_command == exact_command_contract
        )
        return exact_compatible

    def _apply_p4_resume_branch_state(
        self, exact_compatible, original_p4_state, runtime_maze_branch
    ) -> None:
        if exact_compatible:
            self.maze_training_branch = str(
                original_p4_state.get("maze_training_branch", self.maze_training_branch)
            )
            saved_resolved = original_p4_state.get("resolved_maze_training_branch")
            self._resolved_maze_training_branch = (
                str(saved_resolved) if isinstance(saved_resolved, str) else None
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

    def _prepare_p4_warm_optimizers(self, compatible):
        optimizers = compatible.get("optimizers")
        if not isinstance(optimizers, dict):
            raise KeyError("P4 warm start missing optimizer payload")
        maze_profile = self.training_profile in self.MAZE_PROFILES
        if "high_level_critic" not in optimizers:
            raise KeyError("P4 closed-loop warm start missing Critic optimizer")
        if self.training_profile in {
            PROFILE_MAZE_CLOSED_LOOP_V3,
            PROFILE_MAZE_INSTANT_COMMAND_R4,
            PROFILE_MAZE_INSTANT_REPAIR2H,
        }:
            optimizers["high_level_actor"] = copy.deepcopy(
                self.actor_optimizer.state_dict()
            )
            optimizers["high_level_critic"] = copy.deepcopy(
                self.critic_optimizer.state_dict()
            )
            if self.training_profile in self.INSTANT_PROFILES:
                optimizers["response_adapter"] = copy.deepcopy(
                    self.response_optimizer.state_dict()
                )
            actor_optimizer_report = {
                "status": "reset",
                "reason": "new_reward_and_command_dynamics_contract",
            }
            critic_optimizer_report = {
                "status": "reset",
                "reason": "critic_recalibration_without_weight_reset",
            }
        elif self.training_profile == PROFILE_MAZE_CREDIT_REPAIR:
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
        return maze_profile, actor_optimizer_report, critic_optimizer_report

    def _prepare_p4_warm_training_state(
        self, compatible, maze_profile, actor_optimizer_report, critic_optimizer_report
    ) -> None:
        high_state = compatible.get("training_states", {}).get("high_level", {})
        if isinstance(high_state, dict):
            warm_schedule = p4_contract.training_schedule(
                0.0,
                branch=(
                    "instant_command_r4"
                    if self.training_profile == PROFILE_MAZE_INSTANT_COMMAND_R4
                    else (
                        "instant_repair2h"
                        if self.training_profile == PROFILE_MAZE_INSTANT_REPAIR2H
                        else (
                            "closed_loop_v3"
                            if self.training_profile == PROFILE_MAZE_CLOSED_LOOP_V3
                            else ("credit_repair" if maze_profile else "actor_attack")
                        )
                    )
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
            if self.training_profile in {
                PROFILE_MAZE_CREDIT_REPAIR,
                PROFILE_MAZE_INSTANT_COMMAND_R4,
                PROFILE_MAZE_INSTANT_REPAIR2H,
            }:
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

    def _prepare_p4_compatible_bundle(self, bundle, exact_compatible):
        compatible = copy.deepcopy(bundle)
        compatible["contracts"]["reward"] = p2_contract.reward_contract()
        compatible["contracts"]["training"] = p2_contract.training_contract()
        compatible["contracts"]["command"] = p2_contract.command_contract()
        global_state = compatible.get("training_states", {}).get("global", {})
        if exact_compatible:
            saved_scope = global_state.get("train_scope")
            expected_scope = (
                (
                    "maze_instant_repair2h_actor_critic_adapter_calibration"
                    if self.training_profile == PROFILE_MAZE_INSTANT_REPAIR2H
                    else "maze_instant_command_r4_actor_critic_adapter_calibration"
                )
                if self.training_profile in self.INSTANT_PROFILES
                else (
                    "maze_closed_loop_v3_actor_critic_only"
                    if self.training_profile == PROFILE_MAZE_CLOSED_LOOP_V3
                    else (
                        "maze_actor_critic_only"
                        if self.training_profile == PROFILE_MAZE_CREDIT_REPAIR
                        else "high_level_and_response_adapter"
                    )
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
            (
                maze_profile,
                actor_optimizer_report,
                critic_optimizer_report,
            ) = self._prepare_p4_warm_optimizers(compatible)
            self._prepare_p4_warm_training_state(
                compatible,
                maze_profile,
                actor_optimizer_report,
                critic_optimizer_report,
            )
        return compatible

    def _restore_p4_loaded_modules(
        self, bundle, exact_compatible, original_p4_state, path
    ) -> None:
        saved_low_digest = (bundle.get("lineage") or {}).get("low_level_frozen_digest")
        actual_low_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        if saved_low_digest != actual_low_digest:
            raise ValueError(
                "P4 exact resume frozen low-level digest mismatch: "
                f"saved={saved_low_digest!r} actual={actual_low_digest!r}"
            )
        self._initial_low_digest = actual_low_digest
        self.parent_phase_label = (bundle.get("lineage") or {}).get("p4_parent_phase")
        high = (bundle.get("modules") or {}).get("high_level") or {}
        if exact_compatible:
            if self.training_profile in self.INSTANT_PROFILES:
                self._load_leaf(
                    high,
                    "parent_actor_anchor",
                    self._parent_anchor_actor,
                    class_name="P2NavigationActor",
                    spec=navigation_actor_spec(),
                    context="P4 exact resume high_level",
                )
                self._parent_anchor_actor.eval()
                for parameter in self._parent_anchor_actor.parameters():
                    parameter.requires_grad_(False)
                self.parent_anchor_source_sha256 = str(
                    original_p4_state.get("parent_anchor_source_sha256", "")
                )
                self.parent_anchor_digest = self._module_digest(
                    (("parent_actor_anchor", self._parent_anchor_actor),)
                )
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
            if self.training_profile == PROFILE_MAZE_INSTANT_REPAIR2H:
                self.response_buffer.clear_completed_records_for_new_session()
            self._freeze_parent_anchor_from_current_actor(
                source_sha256=self._sha256(path)
            )
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
                        if self.training_profile == PROFILE_FULL_TRACK
                        else "P4 maze warm start diagnostic head"
                    ),
                )
                stuck_head_loaded = True
            if self.training_profile == PROFILE_MAZE_CREDIT_REPAIR:
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

    def _reset_p4_warm_session(self, bundle, runtime_maze_branch) -> None:
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
                (
                    "Maze instant repair2h"
                    if self.training_profile == PROFILE_MAZE_INSTANT_REPAIR2H
                    else "Maze instant-command r4"
                )
                if self.training_profile in self.INSTANT_PROFILES
                else (
                    "Maze closed-loop v3"
                    if self.training_profile == PROFILE_MAZE_CLOSED_LOOP_V3
                    else (
                        "Maze closed-loop v2"
                        if self.training_profile == PROFILE_MAZE_CREDIT_REPAIR
                        else "full-track legacy-contract"
                    )
                )
            )
            optimizer_note = (
                "Actor/Critic optimizer moments reset; weights and return "
                "statistics preserved"
                if self.training_profile == PROFILE_MAZE_CLOSED_LOOP_V3
                else (
                    "Actor/Critic/Adapter optimizer moments and return "
                    "statistics reset; network weights preserved"
                    if self.training_profile in self.INSTANT_PROFILES
                    else (
                        "Actor/Critic optimizer moments and return statistics reset; "
                        "network weights preserved"
                        if self.training_profile == PROFILE_MAZE_CREDIT_REPAIR
                        else "compatible optimizer moments and return statistics preserved"
                    )
                )
            )
            self.logger.warning(
                f"[P4NavPPO] previous P4 contract loaded as {warm_profile} "
                f"warm start; {optimizer_note}"
            )

    def _p4_warm_disposition(self) -> str:
        warm_disposition = (
            (
                "maze_instant_repair2h_warm_start"
                if self.training_profile == PROFILE_MAZE_INSTANT_REPAIR2H
                else "maze_instant_r4_warm_start"
            )
            if self.training_profile in self.INSTANT_PROFILES
            else (
                "maze_closed_loop_v3_warm_start"
                if self.training_profile == PROFILE_MAZE_CLOSED_LOOP_V3
                else (
                    "maze_credit_repair_warm_start"
                    if self.training_profile == PROFILE_MAZE_CREDIT_REPAIR
                    else "full_track_legacy_contract_warm_start"
                )
            )
        )
        return f"p4_{warm_disposition}"

    def _load_p4_checkpoint(self, raw, path, platform_model_id) -> str:
        bundle, _ = normalize_kaiwu_train_bundle(raw)
        exact_compatible = self._validate_p4_resume_contract(bundle)
        original_p4_state = copy.deepcopy(
            bundle.get("training_states", {}).get("p4", {})
        )
        runtime_maze_branch = self.maze_training_branch
        self._apply_p4_resume_branch_state(
            exact_compatible, original_p4_state, runtime_maze_branch
        )
        compatible = self._prepare_p4_compatible_bundle(bundle, exact_compatible)
        mode = self._load_exact_resume(compatible, platform_model_id, path)
        self._restore_p4_loaded_modules(
            bundle, exact_compatible, original_p4_state, path
        )
        if not exact_compatible:
            self._reset_p4_warm_session(bundle, runtime_maze_branch)
        self._configure_adapter_contract()
        if exact_compatible:
            return f"p4_{mode}"
        return self._p4_warm_disposition()

    def _load_p3_parent_checkpoint(self, raw, path, platform_model_id) -> str:
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
            low,
            "locomotion_encoder",
            self.low_level_encoder,
            class_name="VisionEncoder",
            spec=self._low_encoder_spec(self.low_level_encoder),
            context="P4 P3.5 parent low_level",
        )
        self._load_leaf(
            low,
            "actor",
            self.low_level_actor,
            class_name="Actor77Sequential",
            spec=self._low_actor_spec(),
            context="P4 P3.5 parent low_level",
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
                context="P4 P3.5 parent high_level",
            )
        self._load_leaf(
            high,
            "navigation_safety_head",
            self.safety_head,
            class_name="NavigationSafetyHead",
            spec=navigation_safety_head_spec(),
            context="P4 P3.5 parent high_level",
        )
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
        self.source_parent_model_id = self._bundle_identity(
            bundle, platform_model_id, path
        )
        self.loaded_platform_model_id = self.source_parent_model_id
        self.parent_phase_label = str(disposition["phase_label"])
        self.parent_checkpoint_sha256 = self._sha256(path)
        self._freeze_parent_anchor_from_current_actor(
            source_sha256=self.parent_checkpoint_sha256
        )
        self.low_level_state_digest = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        self._initial_low_digest = self.low_level_state_digest
        self.response_buffer.set_low_level_version(self.low_level_state_digest, 0)
        self._configure_adapter_contract()
        parent_buffer = (
            bundle.get("training_states", {}).get("response_adapter", {}).get("buffer")
        )
        if isinstance(parent_buffer, dict):
            self.response_buffer.load_parent_completed_records(parent_buffer)
        parent_high_state = bundle.get("training_states", {}).get("high_level", {})
        self.lifetime_base_seconds = (
            float(parent_high_state.get("lifetime_effective_seconds", 0.0))
            if isinstance(parent_high_state, dict)
            else 0.0
        )
        self.session_wall_seconds = 0.0
        self.diagnostic_elapsed_seconds = 0.0
        self._training_clock_origin_seconds = None
        self._resolved_maze_training_branch = None
        self.session_effective_seconds = 0.0
        self.effective_training_seconds = 0.0
        self.lifetime_effective_seconds = self.lifetime_base_seconds
        self.return_statistics = {
            "count": 0,
            "mean": 0.0,
            "m2": 0.0,
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

    def load_bundle(self, path: str, *, platform_model_id) -> str:
        raw = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(raw, dict):
            raise ValueError("P4 checkpoint payload must be a mapping")
        if raw.get("stage_type") == self.STAGE_TYPE:
            return self._load_p4_checkpoint(raw, path, platform_model_id)
        return self._load_p3_parent_checkpoint(raw, path, platform_model_id)

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
        state = {
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
        if self.training_profile in self.INSTANT_PROFILES:
            state.update(
                parent_anchor_source_sha256=self.parent_anchor_source_sha256,
                parent_anchor_digest=self.parent_anchor_digest,
            )
        return state

    def _load_p4_diagnostic_state(self, state: dict[str, object]) -> None:
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
        if self.training_profile in self.INSTANT_PROFILES:
            saved_anchor_sha = state.get("parent_anchor_source_sha256")
            saved_anchor_digest = state.get("parent_anchor_digest")
            if not isinstance(saved_anchor_sha, str) or not saved_anchor_sha:
                raise ValueError("P4 exact resume missing parent anchor source SHA256")
            if not isinstance(saved_anchor_digest, str) or not saved_anchor_digest:
                raise ValueError("P4 exact resume missing parent anchor digest")
            if self.parent_anchor_digest != saved_anchor_digest:
                raise ValueError(
                    "P4 exact resume parent anchor digest mismatch: "
                    f"saved={saved_anchor_digest!r} actual={self.parent_anchor_digest!r}"
                )
            self.parent_anchor_source_sha256 = saved_anchor_sha
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
            float(clock_origin) if isinstance(clock_origin, (int, float)) else None
        )
        self.maze_training_branch = str(
            state.get("maze_training_branch", self.maze_training_branch)
        )
        if (
            str(state.get("training_profile", self.training_profile))
            != self.training_profile
        ):
            raise ValueError("P4 exact resume training profile mismatch")
        resolved = state.get("resolved_maze_training_branch")
        self._resolved_maze_training_branch = (
            str(resolved) if isinstance(resolved, str) else None
        )
        self._load_p4_diagnostic_state(state)
        self.camera_state.load_state_dict(state.get("camera", {}))
        push_lifetime_count = state.get("push_lifetime_count")
        push_env_seen = state.get("push_env_seen")
        if (
            torch.is_tensor(push_lifetime_count)
            and push_lifetime_count.numel() == self.num_envs
        ):
            self._push_lifetime_count.copy_(
                push_lifetime_count.reshape(-1).to(self.device)
            )
        if torch.is_tensor(push_env_seen) and push_env_seen.numel() == self.num_envs:
            self._push_env_seen.copy_(push_env_seen.reshape(-1).to(self.device).bool())
        self._camera_aux_coefficient = float(state.get("camera_aux_coefficient", 0.0))
        self._camera_aux_gradient_ratio = float(
            state.get("camera_aux_gradient_ratio", 0.0)
        )
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
                "combined_ratio",
                "teacher_ratio",
                "anchor_ratio",
                "camera_ratio",
                "mirror_ratio",
                "stuck_ratio",
                "teacher_valid_steps",
                "stuck_valid_steps",
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
                    p3_contract.P3_WORKER_EXTRA_DIM + p4_contract.P4_WORKER_EXTRA_DIM
                ),
                "p3_training_tail_dim": p3_contract.P3_WORKER_EXTRA_DIM,
                "p4_training_tail_dim": p4_contract.P4_WORKER_EXTRA_DIM,
            },
        }
        payload["training_states"]["p4"] = self._p4_state()
        if self.training_profile in self.INSTANT_PROFILES:
            if not self.parent_anchor_source_sha256 or not self.parent_anchor_digest:
                raise RuntimeError(
                    "P4 instant-command checkpoint requires a loaded immutable parent anchor"
                )
            parent_anchor_leaf = self._leaf(
                "P2NavigationActor",
                navigation_actor_spec(),
                self._parent_anchor_actor.state_dict(),
            )
            parent_anchor_leaf["training_only"] = True
            payload["modules"]["high_level"]["parent_actor_anchor"] = parent_anchor_leaf
        payload["training_states"]["global"]["train_scope"] = (
            (
                "maze_instant_repair2h_actor_critic_adapter_calibration"
                if self.training_profile == PROFILE_MAZE_INSTANT_REPAIR2H
                else "maze_instant_command_r4_actor_critic_adapter_calibration"
            )
            if self.training_profile in self.INSTANT_PROFILES
            else (
                "maze_closed_loop_v3_actor_critic_only"
                if self.training_profile == PROFILE_MAZE_CLOSED_LOOP_V3
                else (
                    "maze_actor_critic_only"
                    if self.training_profile == PROFILE_MAZE_CREDIT_REPAIR
                    else "high_level_and_response_adapter"
                )
            )
        )
        payload["modules"]["high_level"][
            "action_mapper_version"
        ] = p4_contract.ACTION_MAPPER_VERSION
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
            raise RuntimeError(
                "P4 frozen low-level digest drift before checkpoint save"
            )
        for name in ("navigation_safety_head", "critic"):
            payload["modules"]["high_level"][name]["training_only"] = True
        payload["modules"]["high_level"]["actor_stuck_head"] = self._leaf(
            "P4ActorStuckHead",
            p4_actor_stuck_head_spec(),
            self.stuck_head.state_dict(),
        )
        payload["modules"]["high_level"]["actor_stuck_head"]["training_only"] = True
        directory = os.path.dirname(path) or "."
        temporary = os.path.join(
            directory, f".{os.path.basename(path)}.{uuid4().hex}.tmp"
        )
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
        saved_command = raw.get("contracts", {}).get("command") or {}
        expected_command = p4_contract.command_contract(self.training_profile)
        if saved_command != expected_command:
            raise ValueError(
                "P4 Track evaluation command contract does not match runtime profile: "
                f"saved={saved_command.get('version')!r}/"
                f"{saved_command.get('command_transition_mode', 'slew')!r} "
                f"runtime={expected_command.get('version')!r}/"
                f"{expected_command.get('command_transition_mode', 'slew')!r}"
            )
        return super().load_evaluation_bundle(path, platform_model_id=platform_model_id)
