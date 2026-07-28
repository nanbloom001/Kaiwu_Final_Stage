#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Recurrent visual PPO for the frozen Standard visual baseline.

``visual_anchor_anneal_v2`` is the Anchor R2 schedule.  The follow-up
``visual_command_generalization_v1`` schedule keeps the final anchor weights
fixed while the environment worker mixes commands before observation assembly.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO
from agent_ppo.checkpoint_io import (
    KAIWU_TRAIN_FORMAT,
    KAIWU_TRAIN_SCHEMA_VERSION,
    VISUAL_ANCHOR_R2_PHASE_LABELS,
    is_kaiwu_train_bundle,
    validate_low_level_spec,
)
from agent_ppo.feature.definition import RecurrentRolloutStorage

# Recognized schedule modes. ``visual_anchor_anneal_v2`` is the Anchor R2 active
# branch; the rest are kept as explicit branches so historical checkpoints can
# still be resumed without silently falling back to legacy rl* phases.
SUPPORTED_SCHEDULE_MODES = (
    "legacy_three_phase_v1",
    "anchor_anneal_v1",
    "visual_recovery_split_v1",
    "visual_anchor_anneal_v2",
    "visual_command_generalization_v1",
    "p15_response_adapter_v1",
)

# Fixed load_mode values used in the startup log (§6). Returned by
# load_training_bundle to tell the Agent which of the three restore paths ran.
LOAD_MODE_S0 = "s0"
LOAD_MODE_ANCHOR_RESUME = "anchor_resume"
LOAD_MODE_TRANSITION_RESUME = "transition_resume"
LOAD_MODE_SCHEDULE_MIGRATION = "schedule_migration"

# Legacy phase vocabulary kept for resume compatibility only.
_LEGACY_PHASES = ("rlcritic", "rlactor", "rlfull")


class AlgorithmVisualPPO(AlgorithmPPO):
    """PPO with TBPTT and frozen-S0 action/latent anchoring.

    The S0 model never drives the environment. It runs alongside rollout
    collection only to provide deterministic anchor targets at the exact
    recurrent state visited by the current student.
    """

    STAGE_TYPE = "standard_visual_ppo"

    def __init__(
        self,
        *,
        model: nn.Module,
        anchor_encoder: nn.Module,
        anchor_actor: nn.Module,
        optimizer: torch.optim.Optimizer,
        sequence_length: int,
        # Anchor R2 array-driven schedule (§4.1). Hours units.
        schedule_mode: str,
        run_name: str,
        source_parent_model_id: str | int | None,
        anchor_schedule_hours: list[float] | tuple[float, ...] | None,
        action_anchor_schedule: list[float] | tuple[float, ...] | None,
        latent_anchor_schedule: list[float] | tuple[float, ...] | None,
        anchor_phase_labels: list[str] | tuple[str, ...] | None,
        anchor_phase_end_hours: list[float] | tuple[float, ...] | None,
        critic_warmup_learning_rate: float | None,
        task_end_hours: float,
        warning_only_safety: bool,
        max_anchor_action_mse: float,
        max_hard_termination_delta: float,
        max_action_amplitude: float = 6.0,
        command_anchor_action: float = 0.35,
        command_anchor_latent: float = 0.10,
        # Legacy R1 scalar schedule (kept for legacy_three_phase_v1 resume only).
        critic_only_hours: float = 0.0,
        actor_only_end_hours: float = 0.0,
        action_anchor_start: float = 0.0,
        action_anchor_mid: float = 0.0,
        action_anchor_end: float = 0.0,
        latent_anchor_weight: float = 0.0,
        logger=None,
        monitor=None,
        device=None,
        **ppo_kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            logger=logger,
            monitor=monitor,
            device=device,
            schedule="fixed",
            **ppo_kwargs,
        )
        self.anchor_encoder = anchor_encoder
        self.anchor_actor = anchor_actor
        self.sequence_length = int(sequence_length)

        self.schedule_mode = str(schedule_mode)
        self.run_name = str(run_name)
        self.source_parent_model_id = (
            None if source_parent_model_id is None else str(source_parent_model_id)
        )
        self.task_end_hours = float(task_end_hours)
        self.critic_warmup_learning_rate = (
            None if critic_warmup_learning_rate is None
            else float(critic_warmup_learning_rate)
        )
        self.warning_only_safety = bool(warning_only_safety)
        self.max_anchor_action_mse = float(max_anchor_action_mse)
        self.max_hard_termination_delta = float(max_hard_termination_delta)
        self.max_action_amplitude = float(max_action_amplitude)
        self.command_anchor_action = float(command_anchor_action)
        self.command_anchor_latent = float(command_anchor_latent)

        # Legacy scalar schedule (used only when schedule_mode resolves to the
        # legacy branch; Anchor R2 ignores these).
        self.critic_only_hours = float(critic_only_hours)
        self.actor_only_end_hours = float(actor_only_end_hours)
        self.action_anchor_start = float(action_anchor_start)
        self.action_anchor_mid = float(action_anchor_mid)
        self.action_anchor_end = float(action_anchor_end)
        self.latent_anchor_weight = float(latent_anchor_weight)

        # Anchor R2 session clock (§4.3). Drives phase/anchor interpolation.
        # Distinct from elapsed_training_hours, which only records cumulative
        # training time and must NOT decide Anchor R2 phase.
        self.anchor_session_elapsed_hours = 0.0
        self.elapsed_training_hours = 0.0
        self.current_iteration = 0
        self.skipped_nonfinite_updates = 0
        if self.schedule_mode == "p15_response_adapter_v1":
            self.current_phase = "responsebase"
        elif self.schedule_mode == "visual_command_generalization_v1":
            self.current_phase = "commandbase"
        else:
            self.current_phase = "anchorcritic"
        self.action_anchor_weight = 0.0
        self.latent_anchor_weight_current = 0.0
        # Anchor R2 is warning-only: violations never pause actor updates or
        # freeze the schedule (§5.3). actor_updates_paused stays False; kept as
        # an attribute for metric/reporting compatibility.
        self.actor_updates_paused = False
        self.pause_reason = None
        self.anchor_schedule_frozen = False
        self.last_diagnostic_save_requested = False
        self.baseline_hard_termination_sum = 0.0
        self.baseline_hard_termination_windows = 0
        self.baseline_hard_termination_rate = None
        self.last_anchor_action_mse = 0.0
        self.s0_checkpoint_sha256 = "unknown"
        self.loaded_platform_model_id: str | None = None
        self.loaded_schedule_mode: str | None = None
        self.resume_loaded = False

        # Configure the active schedule. _configure_anchor_schedule validates
        # knot monotonicity, phase/knot count, label membership and boundaries.
        self._configure_anchor_schedule(
            anchor_schedule_hours=anchor_schedule_hours,
            action_anchor_schedule=action_anchor_schedule,
            latent_anchor_schedule=latent_anchor_schedule,
            anchor_phase_labels=anchor_phase_labels,
            anchor_phase_end_hours=anchor_phase_end_hours,
        )
        # Configure command metadata. Actual source/target ownership belongs to
        # the worker-side CommandSchedule and is not restored from checkpoints.
        self._configure_command_schedule()
        # Initial phase/weights derived from session clock = 0.
        self.current_phase = self._phase_for_elapsed(self.anchor_session_elapsed_hours)
        self.action_anchor_weight = self._anchor_weight_for_elapsed(
            self.anchor_session_elapsed_hours, kind="action"
        )
        self.latent_anchor_weight_current = self._anchor_weight_for_elapsed(
            self.anchor_session_elapsed_hours, kind="latent"
        )
        # Snapshot the normal critic LR (from optimizer construction) so
        # _set_phase_learning_rates can restore it after a warmup phase. The
        # warmup LR is read from critic_warmup_learning_rate; missing it falls
        # back to this baseline, never silently to 3e-4.
        self._critic_base_lr = next(
            (
                float(group["lr"])
                for group in self.optimizer.param_groups
                if group.get("name") == "critic"
            ),
            1e-4,
        )
        self._set_trainable_phase(self.current_phase)
        self._set_phase_learning_rates(self.current_phase)

        for module in (self.anchor_encoder, self.anchor_actor):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    # ------------------------------------------------------------------
    # Schedule configuration (§5.3)
    # ------------------------------------------------------------------

    def _configure_anchor_schedule(
        self,
        *,
        anchor_schedule_hours,
        action_anchor_schedule,
        latent_anchor_schedule,
        anchor_phase_labels,
        anchor_phase_end_hours,
    ) -> None:
        """Validate and store the active anchor schedule.

        ``visual_anchor_anneal_v2`` requires the full array-driven schedule
        (4 knots, 4 phases, 3 end hours). Legacy modes fall back to scalar
        breakpoints configured separately. An unknown mode or missing knot
        raises a configuration error; it never silently generates rl* phases.
        """
        if self.schedule_mode not in SUPPORTED_SCHEDULE_MODES:
            raise ValueError(
                f"Unsupported schedule_mode={self.schedule_mode!r}; "
                f"expected one of {SUPPORTED_SCHEDULE_MODES}"
            )
        if self.schedule_mode in {
            "visual_command_generalization_v1",
            "p15_response_adapter_v1",
        }:
            self.anchor_schedule_hours = None
            self.action_anchor_schedule = [self.command_anchor_action]
            self.latent_anchor_schedule = [self.command_anchor_latent]
            if self.schedule_mode == "p15_response_adapter_v1":
                self.anchor_phase_labels = [
                    "responsebase",
                    "responseexpand",
                    "responsefull",
                    "responsecalib",
                ]
                self.anchor_phase_end_hours = [0.5, 2.0, 7.0]
            else:
                self.anchor_phase_labels = ["commandbase", "commandblend", "commandfull"]
                self.anchor_phase_end_hours = [0.5, 3.0]
            return
        # Anchor R2: array-driven schedule is mandatory.
        if self.schedule_mode == "visual_anchor_anneal_v2":
            if anchor_schedule_hours is None:
                raise ValueError(
                    "visual_anchor_anneal_v2 requires anchor_schedule_hours"
                )
            hours = [float(h) for h in anchor_schedule_hours]
            actions = [float(a) for a in (action_anchor_schedule or [])]
            latents = [float(a) for a in (latent_anchor_schedule or [])]
            labels = list(anchor_phase_labels or [])
            ends = [float(e) for e in (anchor_phase_end_hours or [])]
            # Length checks: 4 knots, 4 phases, matching weight arrays, 3 ends.
            if len(hours) != 4:
                raise ValueError(
                    f"anchor_schedule_hours must have 4 knots, got {len(hours)}"
                )
            if len(labels) != 4:
                raise ValueError(
                    f"anchor_phase_labels must have 4 labels, got {len(labels)}"
                )
            if len(actions) != 4 or len(latents) != 4:
                raise ValueError(
                    "action_anchor_schedule and latent_anchor_schedule must "
                    f"each have 4 values; got {len(actions)}/{len(latents)}"
                )
            if len(ends) != 3:
                raise ValueError(
                    f"anchor_phase_end_hours must have 3 values, got {len(ends)}"
                )
            # Label membership (§5.5).
            for label in labels:
                if label not in VISUAL_ANCHOR_R2_PHASE_LABELS:
                    raise ValueError(
                        f"anchor phase label {label!r} not in "
                        f"{VISUAL_ANCHOR_R2_PHASE_LABELS}"
                    )
            if labels != list(VISUAL_ANCHOR_R2_PHASE_LABELS):
                raise ValueError(
                    f"anchor_phase_labels must be exactly "
                    f"{list(VISUAL_ANCHOR_R2_PHASE_LABELS)} in order, got {labels}"
                )
            # Knot monotonicity and non-negative start.
            if hours[0] < 0.0:
                raise ValueError(
                    f"anchor_schedule_hours must start >= 0, got {hours[0]}"
                )
            for prev, cur in zip(hours, hours[1:]):
                if not cur > prev:
                    raise ValueError(
                        f"anchor_schedule_hours must be strictly increasing: {hours}"
                    )
            # End hours must match interior knots (left-closed/right-open
            # boundaries in §4.1): ends == [hours[1], hours[2], hours[3]].
            if ends != hours[1:]:
                raise ValueError(
                    f"anchor_phase_end_hours {ends} must equal interior knots "
                    f"{hours[1:]}"
                )
            # Weight ranges in [0, 1] (anchors are mixing coefficients).
            for name, values in (("action", actions), ("latent", latents)):
                for value in values:
                    if not 0.0 <= value <= 1.0:
                        raise ValueError(
                            f"{name}_anchor_schedule values must be in [0,1], "
                            f"got {values}"
                        )
            self.anchor_schedule_hours = hours
            self.action_anchor_schedule = actions
            self.latent_anchor_schedule = latents
            self.anchor_phase_labels = labels
            self.anchor_phase_end_hours = ends
            return
        # Legacy modes: no array schedule. They use scalar breakpoints set via
        # the legacy kwargs; nothing to store here.
        self.anchor_schedule_hours = None
        self.action_anchor_schedule = None
        self.latent_anchor_schedule = None
        self.anchor_phase_labels = list(_LEGACY_PHASES)
        self.anchor_phase_end_hours = None

    def _configure_command_schedule(self) -> None:
        """Fix command scheduler state for the active mode.

        Anchor R2 returns target_probability=0 and requires no progressive
        command data. visual_recovery_split_v1 keeps its historical behavior
        (target_probability driven by worker bridge) but is not the active task.
        """
        if self.schedule_mode == "visual_anchor_anneal_v2":
            self.command_target_probability = 0.0
            return
        if self.schedule_mode == "visual_recovery_split_v1":
            # Historical R3 behavior retained for resume compatibility.
            self.command_target_probability = 0.0
            return
        if self.schedule_mode == "visual_command_generalization_v1":
            self.command_target_probability = 0.0
            return
        self.command_target_probability = 0.0

    def _phase_for_elapsed(self, elapsed_h: float) -> str:
        """Map elapsed session hours to the current phase label.

        Anchor R2 uses left-closed/right-open intervals (§4.1):
            [0.0, h1) = anchorcritic
            [h1, h2)  = anchoractor
            [h2, h3)  = anchoranneal
            [h3, +inf) = anchorfinal
        Legacy modes keep the scalar-breakpoint rlcritic/rlactor/rlfull mapping.
        """
        if self.schedule_mode == "visual_command_generalization_v1":
            if elapsed_h < 0.5:
                return "commandbase"
            if elapsed_h < 3.0:
                return "commandblend"
            return "commandfull"
        if self.schedule_mode == "p15_response_adapter_v1":
            if elapsed_h < 0.5:
                return "responsebase"
            if elapsed_h < 2.0:
                return "responseexpand"
            if elapsed_h < 7.0:
                return "responsefull"
            return "responsecalib"
        if self.schedule_mode == "visual_anchor_anneal_v2":
            ends = self.anchor_phase_end_hours
            if elapsed_h < ends[0]:
                return self.anchor_phase_labels[0]
            if elapsed_h < ends[1]:
                return self.anchor_phase_labels[1]
            if elapsed_h < ends[2]:
                return self.anchor_phase_labels[2]
            return self.anchor_phase_labels[3]
        # Legacy scalar mapping.
        if elapsed_h < self.critic_only_hours:
            return "rlcritic"
        if elapsed_h < self.actor_only_end_hours:
            return "rlactor"
        return "rlfull"

    def _anchor_weight_for_elapsed(self, elapsed_h: float, *, kind: str) -> float:
        """Linearly interpolate the action or latent anchor weight.

        All A -> B ranges are linear over wall-clock within each phase (§4.1 N7).
        ``t >= last knot`` clamps to the final value (no extrapolation, no
        fallback to legacy phases).
        """
        if self.schedule_mode in {
            "visual_command_generalization_v1",
            "p15_response_adapter_v1",
        }:
            return (
                self.command_anchor_action
                if kind == "action"
                else self.command_anchor_latent
            )
        if self.schedule_mode == "visual_anchor_anneal_v2":
            hours = self.anchor_schedule_hours
            values = (
                self.action_anchor_schedule
                if kind == "action"
                else self.latent_anchor_schedule
            )
            if elapsed_h <= hours[0]:
                return float(values[0])
            if elapsed_h >= hours[-1]:
                return float(values[-1])
            for left, right, v_left, v_right in zip(
                hours[:-1], hours[1:], values[:-1], values[1:]
            ):
                if left <= elapsed_h < right:
                    span = max(right - left, 1e-9)
                    progress = (elapsed_h - left) / span
                    return float(v_left + progress * (v_right - v_left))
            return float(values[-1])
        # Legacy scalar interpolation (action anchor only).
        if kind != "action":
            return self.latent_anchor_weight
        if elapsed_h <= self.critic_only_hours:
            return self.action_anchor_start
        if elapsed_h <= self.actor_only_end_hours:
            span = max(self.actor_only_end_hours - self.critic_only_hours, 1e-6)
            progress = (elapsed_h - self.critic_only_hours) / span
            return self.action_anchor_start + progress * (
                self.action_anchor_mid - self.action_anchor_start
            )
        span = max(self.task_end_hours - self.actor_only_end_hours, 1e-6)
        progress = min(
            1.0,
            max(0.0, (elapsed_h - self.actor_only_end_hours) / span),
        )
        return self.action_anchor_mid + progress * (
            self.action_anchor_end - self.action_anchor_mid
        )

    def _set_trainable_phase(self, phase: str) -> None:
        """Set requires_grad per the active schedule's phase table (§5.3).

        Anchor R2 phase table:
            anchorcritic:  Actor/std=false, RNN/output=false, CNN=false, Critic=true
            anchoractor:   Actor/std=true,  RNN/output=false, CNN=false, Critic=true
            anchoranneal:  Actor/std=true,  RNN/output=true,  CNN=false, Critic=true
            anchorfinal:   Actor/std=true,  RNN/output=true,  CNN=false, Critic=true
        CNN is permanently frozen regardless of phase.
        """
        if self.schedule_mode in {
            "visual_command_generalization_v1",
            "p15_response_adapter_v1",
        }:
            enabled = not (
                self.schedule_mode == "p15_response_adapter_v1"
                and phase == "responsecalib"
            )
            actor_enabled, recurrent_enabled = enabled, enabled
        elif self.schedule_mode == "visual_anchor_anneal_v2":
            table = {
                "anchorcritic": (False, False),
                "anchoractor": (True, False),
                "anchoranneal": (True, True),
                "anchorfinal": (True, True),
            }
            if phase not in table:
                raise ValueError(
                    f"Unknown Anchor R2 phase {phase!r}; expected one of "
                    f"{list(table)}"
                )
            actor_enabled, recurrent_enabled = table[phase]
        else:
            # Legacy behavior: rlcritic freezes both; rlactor trains actor;
            # rlfull trains actor + recurrent.
            actor_enabled = phase in {"rlactor", "rlfull"}
            recurrent_enabled = phase == "rlfull"

        for parameter in self.actor_critic.actor.parameters():
            parameter.requires_grad_(actor_enabled)
        critic_enabled = not (
            self.schedule_mode == "p15_response_adapter_v1"
            and phase == "responsecalib"
        )
        for parameter in self.actor_critic.critic.parameters():
            parameter.requires_grad_(critic_enabled)
        if hasattr(self.actor_critic, "std"):
            self.actor_critic.std.requires_grad_(actor_enabled)
        if hasattr(self.actor_critic, "log_std"):
            self.actor_critic.log_std.requires_grad_(actor_enabled)

        # CNN is permanently frozen and never enters any optimizer group.
        for parameter in self.actor_critic.vision_encoder.cnn.parameters():
            parameter.requires_grad_(False)
        for parameter in self.actor_critic.vision_encoder.rnn.parameters():
            parameter.requires_grad_(recurrent_enabled)
        for parameter in self.actor_critic.vision_encoder.rnn_output_layer.parameters():
            parameter.requires_grad_(recurrent_enabled)

    def _set_phase_learning_rates(self, phase: str) -> None:
        """Switch per-group learning rates based on the active phase (§4.2/§5.3).

        ``anchorcritic`` uses ``critic_warmup_learning_rate`` (3e-4) for the
        critic group; all other phases use the normal critic LR (1e-4). Actor
        and LSTM groups keep their configured LR throughout. If warmup LR is
        missing the method falls back to the critic group's current LR and
        emits a warning, never silently to 3e-4.
        """
        if self.schedule_mode != "visual_anchor_anneal_v2":
            return
        critic_group = None
        for group in self.optimizer.param_groups:
            if group.get("name") == "critic":
                critic_group = group
                break
        if critic_group is None:
            return
        if phase == "anchorcritic":
            if self.critic_warmup_learning_rate is not None:
                critic_group["lr"] = self.critic_warmup_learning_rate
            else:
                if self.logger:
                    self.logger.warning(
                        "[VisualPPO] critic_warmup_learning_rate is missing; "
                        "falling back to critic_learning_rate (1e-4), not 3e-4"
                    )
        else:
            # Normal critic LR. The TOML critic_learning_rate was applied at
            # optimizer construction; restore it from the stored baseline.
            critic_group["lr"] = self._critic_base_lr

    # ------------------------------------------------------------------
    # Storage / recurrent state
    # ------------------------------------------------------------------

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
        device=None,
    ):
        self.storage = RecurrentRolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            privileged_obs_shape=critic_obs_shape,
            actions_shape=action_shape,
            anchor_latent_dim=self.actor_critic.vision_encoder.rnn_output_dim,
            device=device or self.device,
        )

    def initialize_recurrent_states(self, num_envs: int) -> None:
        self.actor_critic.vision_encoder.reset_hidden_state(num_envs, self.device)
        self.anchor_encoder.reset_hidden_state(num_envs, self.device)

    @staticmethod
    def _clone_hidden(hidden_states):
        if hidden_states is None:
            return None
        return tuple(value.detach().clone() for value in hidden_states)

    def rollout_hidden_state(self):
        return self._clone_hidden(self.actor_critic.get_hidden_states())

    def anchor_inference(self, obs: torch.Tensor):
        """Run the frozen S0 anchor (eval, no grad) for action/latent targets."""
        proprio, depth = self.actor_critic._split_actor_observation(obs)
        latent = self.anchor_encoder(
            depth,
            proprio,
            masks=None,
            detach_hidden=True,
        )
        action = self.anchor_actor(torch.cat((proprio, latent), dim=-1))
        return action, latent

    def reset_recurrent_states(self, dones: torch.Tensor) -> None:
        self.actor_critic.reset(dones)
        done_ids = torch.nonzero(dones.reshape(-1).bool(), as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self.anchor_encoder.reset_hidden_state_for_envs(done_ids)

    # ------------------------------------------------------------------
    # Safety (§5.3 warning-only)
    # ------------------------------------------------------------------

    def _update_safety_state(self, elapsed_h: float) -> float:
        """Update safety diagnostics.

        Anchor R2 (warning_only_safety=True): hard termination, anchor MSE and
        action-amplitude violations only emit a warning, increment counters and
        request a diagnostic checkpoint. They never set actor_updates_paused,
        never freeze the schedule, never roll back memory snapshots. NaN/Inf in
        weights/loss is still a hard failure handled in ``learn``.
        """
        hard_rate = float(self.storage.hard_terminations.float().mean().item())
        # Baseline hard-termination is accumulated over the critic-only warmup
        # window (anchorcritic) for diagnostic comparison only.
        warmup_phase = (
            self.anchor_phase_labels[0]
            if self.schedule_mode == "visual_anchor_anneal_v2"
            else None
        )
        in_warmup = (
            (elapsed_h < self.anchor_phase_end_hours[0])
            if self.schedule_mode == "visual_anchor_anneal_v2"
            else (elapsed_h < self.critic_only_hours)
        )
        if in_warmup:
            self.baseline_hard_termination_sum += hard_rate
            self.baseline_hard_termination_windows += 1
            self.baseline_hard_termination_rate = (
                self.baseline_hard_termination_sum
                / max(1, self.baseline_hard_termination_windows)
            )
        if self.baseline_hard_termination_rate is None:
            self.baseline_hard_termination_rate = hard_rate

        reasons: list[str] = []
        hard_limit = (
            self.baseline_hard_termination_rate
            + self.max_hard_termination_delta
        )
        if hard_rate > hard_limit:
            reasons.append(f"hard_termination={hard_rate:.4f}>{hard_limit:.4f}")
        if self.last_anchor_action_mse > self.max_anchor_action_mse:
            reasons.append(
                f"anchor_action_mse={self.last_anchor_action_mse:.4f}>"
                f"{self.max_anchor_action_mse:.4f}"
            )
        if self.storage is not None and self.storage.actions.numel() > 0:
            action_amplitude = float(self.storage.actions.detach().abs().max().item())
            if not np.isfinite(action_amplitude):
                reasons.append("action_amplitude=nonfinite")
            elif action_amplitude > self.max_action_amplitude:
                reasons.append(
                    f"action_amplitude={action_amplitude:.4f}>"
                    f"{self.max_action_amplitude:.4f}"
                )
        if reasons:
            self.last_diagnostic_save_requested = True
            if self.warning_only_safety:
                # Anchor R2: warning only. Do not pause, do not freeze.
                if self.logger:
                    self.logger.warning(
                        "[VisualPPO] safety diagnostic (warning-only): "
                        + "; ".join(reasons)
                    )
            else:
                # Legacy behavior retained for historical modes.
                self.actor_updates_paused = True
                self.pause_reason = "; ".join(reasons)
        else:
            self.actor_updates_paused = False
            self.pause_reason = None
        return hard_rate

    # ------------------------------------------------------------------
    # Learn
    # ------------------------------------------------------------------

    def learn(self, elapsed_h: float | None = None) -> dict[str, float]:
        elapsed_h = (
            self.anchor_session_elapsed_hours if elapsed_h is None else float(elapsed_h)
        )
        # elapsed_training_hours is maintained by the workflow as a cumulative
        # record. Only the anchor session clock drives phase decisions.
        self.anchor_session_elapsed_hours = elapsed_h
        self.current_phase = self._phase_for_elapsed(elapsed_h)
        self.action_anchor_weight = self._anchor_weight_for_elapsed(
            elapsed_h, kind="action"
        )
        self.latent_anchor_weight_current = self._anchor_weight_for_elapsed(
            elapsed_h, kind="latent"
        )
        hard_rate = self._update_safety_state(elapsed_h)
        self._set_trainable_phase(self.current_phase)
        self._set_phase_learning_rates(self.current_phase)

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "action_anchor_loss": 0.0,
            "latent_anchor_loss": 0.0,
            "anchor_action_mse": 0.0,
        }
        applied_updates = 0
        generator = self.storage.recurrent_mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
            self.sequence_length,
        )
        for sample_index, sample in enumerate(generator):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_log_prob_batch,
                _old_mu_batch,
                _old_sigma_batch,
                hidden_batch,
                masks_batch,
                anchor_actions_batch,
                anchor_latents_batch,
                anchor_weights_batch,
            ) = sample

            self.actor_critic.update_distribution(
                obs_batch,
                hidden_states=hidden_batch,
                masks=masks_batch,
            )
            actions_log_prob = self.actor_critic.get_actions_log_prob(actions_batch)
            entropy = self.actor_critic.entropy.mean()
            values = self.actor_critic.evaluate(critic_obs_batch)
            action_mean = self.actor_critic.action_mean
            latent = self.actor_critic.last_latent

            ratio = torch.exp(
                actions_log_prob - old_log_prob_batch.squeeze(-1)
            )
            advantages = advantages_batch.squeeze(-1)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            policy_loss = torch.maximum(surrogate, surrogate_clipped).mean()
            value_loss = self._compute_value_loss(
                values, returns_batch, target_values_batch
            )
            anchor_weights_batch = anchor_weights_batch.to(
                device=action_mean.device, dtype=action_mean.dtype
            )
            weight_sum = anchor_weights_batch.sum().clamp_min(1.0)
            action_anchor_per_sample = F.smooth_l1_loss(
                action_mean, anchor_actions_batch, reduction="none"
            ).mean(dim=-1, keepdim=True)
            latent_anchor_per_sample = F.smooth_l1_loss(
                latent, anchor_latents_batch, reduction="none"
            ).mean(dim=-1, keepdim=True)
            action_mse_per_sample = (
                (action_mean - anchor_actions_batch).pow(2).mean(dim=-1, keepdim=True)
            )
            action_anchor_loss = (
                action_anchor_per_sample * anchor_weights_batch
            ).sum() / weight_sum
            latent_anchor_loss = (
                latent_anchor_per_sample * anchor_weights_batch
            ).sum() / weight_sum
            anchor_action_mse = (
                action_mse_per_sample * anchor_weights_batch
            ).sum() / weight_sum

            loss = (
                policy_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy
                + self.action_anchor_weight * action_anchor_loss
                + self.latent_anchor_weight_current * latent_anchor_loss
            )
            if not torch.isfinite(loss):
                self.skipped_nonfinite_updates += 1
                if self.logger:
                    self.logger.warning(
                        f"[VisualPPO] nonfinite loss in minibatch {sample_index}; "
                        "optimizer step skipped"
                    )
                continue

            self.optimizer.zero_grad()
            loss.backward()
            finite = all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in self.actor_critic.parameters()
            )
            if not finite:
                self.optimizer.zero_grad()
                self.skipped_nonfinite_updates += 1
                if self.logger:
                    self.logger.warning(
                        f"[VisualPPO] nonfinite gradient in minibatch {sample_index}; "
                        "optimizer step skipped"
                    )
                continue
            nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in self.actor_critic.parameters()
                    if parameter.requires_grad
                ],
                self.max_grad_norm,
            )
            self.optimizer.step()
            if hasattr(self.actor_critic, "std"):
                self.actor_critic.std.data.copy_(
                    torch.maximum(
                        torch.nan_to_num(
                            self.actor_critic.std.data,
                            nan=0.1,
                            posinf=1.0,
                            neginf=0.0,
                        ),
                        self.min_std,
                    )
                )

            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["entropy_loss"] += float(entropy.item())
            totals["action_anchor_loss"] += float(action_anchor_loss.item())
            totals["latent_anchor_loss"] += float(latent_anchor_loss.item())
            totals["anchor_action_mse"] += float(anchor_action_mse.item())
            applied_updates += 1

        divisor = max(1, applied_updates)
        metrics = {key: value / divisor for key, value in totals.items()}
        metrics.update(
            {
                "hard_termination_rate": hard_rate,
                "baseline_hard_termination_rate": float(
                    self.baseline_hard_termination_rate or 0.0
                ),
                "action_anchor_weight": self.action_anchor_weight,
                "latent_anchor_weight": self.latent_anchor_weight_current,
                "actor_updates_paused": float(self.actor_updates_paused),
                "anchor_schedule_frozen": float(self.anchor_schedule_frozen),
                "elapsed_training_hours": self.elapsed_training_hours,
                "applied_updates": float(applied_updates),
                "skipped_nonfinite_updates": float(
                    self.skipped_nonfinite_updates
                ),
                "anchor_weight_mean": float(
                    self.storage.anchor_weights[: self.storage.step].mean().item()
                    if self.storage.step else 1.0
                ),
            }
        )
        self.last_anchor_action_mse = metrics["anchor_action_mse"]
        self.current_iteration += 1
        self.train_step += 1
        return metrics

    # ------------------------------------------------------------------
    # RNG / hashing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: Any, logger=None) -> None:
        if not isinstance(state, dict):
            return
        try:
            if "python" in state:
                random.setstate(state["python"])
            if "numpy" in state:
                np.random.set_state(state["numpy"])
            if "torch_cpu" in state:
                torch.set_rng_state(state["torch_cpu"].cpu())
            if torch.cuda.is_available() and "torch_cuda" in state:
                torch.cuda.set_rng_state_all(
                    [value.cpu() for value in state["torch_cuda"]]
                )
        except (RuntimeError, ValueError, TypeError) as exc:
            if logger:
                logger.warning(
                    f"[VisualPPO] RNG state was not restored; continuing: {exc}"
                )

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def trainable_modules_snapshot(self) -> dict[str, bool]:
        """Return the current per-module trainable flags for logging/checkpoint."""
        return {
            "cnn": False,
            "actor": any(
                p.requires_grad for p in self.actor_critic.actor.parameters()
            ),
            "lstm": any(
                p.requires_grad
                for p in self.actor_critic.vision_encoder.rnn.parameters()
            ),
            "critic": any(
                p.requires_grad for p in self.actor_critic.critic.parameters()
            ),
        }

    # ------------------------------------------------------------------
    # Checkpoint save / load (§5.4)
    # ------------------------------------------------------------------

    def save_training_bundle(
        self,
        path: str,
        *,
        platform_model_id: str | int,
        phase_label: str,
        model_spec: dict[str, int],
    ) -> str:
        payload = {
            "format": KAIWU_TRAIN_FORMAT,
            "schema_version": KAIWU_TRAIN_SCHEMA_VERSION,
            "stage_type": self.STAGE_TYPE,
            "model_spec": dict(model_spec),
            "modules": {
                "vision_encoder": {
                    "state_dict": self.actor_critic.vision_encoder.state_dict()
                },
                "low_level": {
                    "actor_state_dict": self.actor_critic.actor.state_dict()
                },
                "critic": {"state_dict": self.actor_critic.critic.state_dict()},
                "action_distribution": {
                    "std": (
                        self.actor_critic.std.detach().cpu()
                        if hasattr(self.actor_critic, "std")
                        else None
                    )
                },
                "s0_anchor": {
                    "vision_encoder_state_dict": self.anchor_encoder.state_dict(),
                    "actor_state_dict": self.anchor_actor.state_dict(),
                },
            },
            "optimizers": {"visual_ppo": self.optimizer.state_dict()},
            "training_state": {
                "current_iteration": self.current_iteration,
                "iteration_semantics": "completed_outer_iterations_v1",
                # Anchor session clock drives Anchor R2 phase decisions (§4.3).
                # anchor_elapsed_hours is a legacy-reader alias with the same
                # value; loader prefers anchor_session_elapsed_hours.
                "anchor_session_elapsed_hours": self.anchor_session_elapsed_hours,
                "anchor_elapsed_hours": self.anchor_session_elapsed_hours,
                "elapsed_training_hours": self.elapsed_training_hours,
                "phase_label": phase_label,
                "schedule_mode": self.schedule_mode,
                "run_name": self.run_name,
                "source_parent_model_id": self.source_parent_model_id,
                "transition_parent_platform_model_id": (
                    self.loaded_platform_model_id
                ),
                "action_anchor_weight": self.action_anchor_weight,
                "latent_anchor_weight": self.latent_anchor_weight_current,
                "trainable_modules": self.trainable_modules_snapshot(),
                "actor_updates_paused": self.actor_updates_paused,
                "anchor_schedule_frozen": self.anchor_schedule_frozen,
                "pause_reason": self.pause_reason,
                "warning_only_safety": self.warning_only_safety,
                "max_action_amplitude": self.max_action_amplitude,
                "baseline_hard_termination_sum": self.baseline_hard_termination_sum,
                "baseline_hard_termination_windows": self.baseline_hard_termination_windows,
                "baseline_hard_termination_rate": self.baseline_hard_termination_rate,
                "last_anchor_action_mse": self.last_anchor_action_mse,
                "command_session_elapsed_hours": self.anchor_session_elapsed_hours,
                "command_runtime_owner": "worker_observation_bridge_v1",
                "command_resume_policy": "new_task_restart",
                "rng_state": self._capture_rng_state(),
                "skipped_nonfinite_updates": self.skipped_nonfinite_updates,
            },
            "lineage": {
                "s0_checkpoint_sha256": self.s0_checkpoint_sha256,
                "source_parent_model_id": self.source_parent_model_id,
                "transition_parent_platform_model_id": (
                    self.loaded_platform_model_id
                ),
            },
            "lstm_reset_contract": {
                "live_hidden_saved": False,
                "resume_action": "reset_env_and_zero_hidden",
                "done_mask_semantics": "true_means_continue",
                "tbptt_sequence_length": self.sequence_length,
            },
            "capabilities": {
                "critic_trained": True,
                "deployable": False,
                "requires_depth": True,
                "requires_height_scan_for_actor": False,
            },
            "platform_model_id": str(platform_model_id),
        }
        torch.save(payload, path)
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            raise IOError(f"Visual PPO checkpoint write failed: {path}")
        return self._sha256(path)

    def load_eval_bundle(
        self,
        path: str,
        *,
        expected_spec: dict[str, int],
        num_envs: int,
    ) -> dict[str, Any]:
        """Load only the deploy-shaped visual student for evaluation."""
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        if not is_kaiwu_train_bundle(checkpoint):
            raise ValueError(f"Expected kaiwu_train_v1 checkpoint: {path}")
        validate_low_level_spec(checkpoint, expected_spec)
        modules = checkpoint.get("modules", {})
        vision_state = modules.get("vision_encoder", {}).get("state_dict")
        actor_state = modules.get("low_level", {}).get("actor_state_dict")
        if not isinstance(vision_state, dict) or not isinstance(actor_state, dict):
            raise KeyError(
                "VisualPPO eval requires modules.vision_encoder.state_dict "
                "and modules.low_level.actor_state_dict"
            )
        self.actor_critic.vision_encoder.load_state_dict(vision_state, strict=True)
        self.actor_critic.actor.load_state_dict(actor_state, strict=True)
        std_value = modules.get("action_distribution", {}).get("std")
        if std_value is not None and hasattr(self.actor_critic, "std"):
            self.actor_critic.std.data.copy_(std_value.to(self.device))
        self.actor_critic.eval()
        self.actor_critic.vision_encoder.reset_hidden_state(num_envs, self.device)
        self.loaded_platform_model_id = checkpoint.get("platform_model_id")
        state = checkpoint.get("training_state", {})
        self.loaded_schedule_mode = (
            state.get("schedule_mode") if isinstance(state, dict) else None
        )
        lineage = checkpoint.get("lineage", {})
        if not isinstance(lineage, dict):
            lineage = {}
        checksum = self._sha256(path)
        self.s0_checkpoint_sha256 = checksum
        return {
            "format": checkpoint.get("format"),
            "schema_version": checkpoint.get("schema_version"),
            "platform_model_id": self.loaded_platform_model_id,
            "schedule_mode": self.loaded_schedule_mode,
            "sha256": checksum,
            "file_size_bytes": os.path.getsize(path),
            "lineage_source_parent_model_id": lineage.get("source_parent_model_id"),
            "lineage_transition_parent_platform_model_id": lineage.get(
                "transition_parent_platform_model_id"
            ),
            "uses_height_scan_at_inference": False,
        }

    def load_training_bundle(
        self,
        path: str,
        *,
        expected_spec: dict[str, int],
        env_seed: int | None = None,
    ) -> str:
        """Restore weights and return one of the three load_mode values (§5.4/§6).

        Paths:
          * exact S0 ``visionfull-28401``     -> LOAD_MODE_S0
          * same-mode Anchor R2 bundle         -> LOAD_MODE_ANCHOR_RESUME
          * other Visual PPO schedule bundle   -> LOAD_MODE_SCHEDULE_MIGRATION

        RNG: only same-mode resume restores the saved RNG. ``s0`` and
        ``schedule_migration`` ignore the bundle RNG and reinitialize from
        ``env_seed`` (the active TOML ``[env_conf].seed``). Live LSTM hidden is
        never restored; caller must reset the env and zero student/S0 hidden.
        """
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        if not is_kaiwu_train_bundle(checkpoint):
            raise ValueError(f"Expected kaiwu_train_v1 checkpoint: {path}")
        validate_low_level_spec(checkpoint, expected_spec)
        modules = checkpoint.get("modules", {})
        vision_state = modules.get("vision_encoder", {}).get("state_dict")
        actor_state = modules.get("low_level", {}).get("actor_state_dict")
        if not isinstance(vision_state, dict) or not isinstance(actor_state, dict):
            raise KeyError(
                "Visual PPO preload requires modules.vision_encoder.state_dict "
                "and modules.low_level.actor_state_dict"
            )
        self.actor_critic.vision_encoder.load_state_dict(vision_state, strict=True)
        self.actor_critic.actor.load_state_dict(actor_state, strict=True)

        is_visual_ppo_bundle = checkpoint.get("stage_type") in {
            "standard_visual_ppo",
            "p15_response_adapter",
        }
        saved_schedule_mode = (
            checkpoint.get("training_state", {}).get("schedule_mode")
            if isinstance(checkpoint.get("training_state"), dict)
            else None
        )
        same_mode = (
            is_visual_ppo_bundle
            and saved_schedule_mode == self.schedule_mode
        )
        # Other Visual PPO schedule (e.g. R3 visual_recovery_split_v1) loaded
        # while the active task is Anchor R2: treat as migration parent.
        other_visual_schedule = (
            is_visual_ppo_bundle and not same_mode
        )

        anchor_section = modules.get("s0_anchor", {}) if is_visual_ppo_bundle else {}
        anchor_vision = anchor_section.get(
            "vision_encoder_state_dict", vision_state
        )
        anchor_actor = anchor_section.get("actor_state_dict", actor_state)
        self.anchor_encoder.load_state_dict(anchor_vision, strict=True)
        self.anchor_actor.load_state_dict(anchor_actor, strict=True)

        transition_resume = (
            self.schedule_mode == "visual_command_generalization_v1"
            and is_visual_ppo_bundle
            and saved_schedule_mode == "visual_anchor_anneal_v2"
        )
        transition_resume = transition_resume or (
            self.schedule_mode == "p15_response_adapter_v1"
            and is_visual_ppo_bundle
            and saved_schedule_mode == "visual_command_generalization_v1"
        )

        if same_mode or transition_resume:
            # Full resume: restore critic, std, optimizer, anchor clock, RNG.
            critic_state = modules.get("critic", {}).get("state_dict")
            if not isinstance(critic_state, dict):
                raise KeyError("modules.critic.state_dict missing from visual PPO resume")
            self.actor_critic.critic.load_state_dict(critic_state, strict=True)
            std_value = modules.get("action_distribution", {}).get("std")
            if std_value is not None and hasattr(self.actor_critic, "std"):
                self.actor_critic.std.data.copy_(std_value.to(self.device))
            optimizer_state = checkpoint.get("optimizers", {}).get("visual_ppo")
            if not isinstance(optimizer_state, dict):
                raise KeyError("optimizers.visual_ppo missing from visual PPO resume")
            self.optimizer.load_state_dict(optimizer_state)
            state = checkpoint.get("training_state", {})
            self.current_iteration = int(state.get("current_iteration", 0))
            self.skipped_nonfinite_updates = int(
                state.get("skipped_nonfinite_updates", 0)
            )
            if self.schedule_mode == "visual_command_generalization_v1" or (
                self.schedule_mode == "p15_response_adapter_v1" and transition_resume
            ):
                # Preserve the learned PPO state but start a fresh command
                # session when entering a new command curriculum. Exact P1.5
                # schema-2 resumes restore their saved wall-clock phase below.
                self.anchor_session_elapsed_hours = 0.0
            else:
                clock = state.get("anchor_session_elapsed_hours")
                if clock is None:
                    clock = state.get("anchor_elapsed_hours")
                self.anchor_session_elapsed_hours = float(clock or 0.0)
            self.elapsed_training_hours = float(
                state.get("elapsed_training_hours", self.anchor_session_elapsed_hours)
            )
            self.action_anchor_weight = float(
                state.get("action_anchor_weight", self.action_anchor_weight)
            )
            self.latent_anchor_weight_current = float(
                state.get("latent_anchor_weight", self.latent_anchor_weight_current)
            )
            self.actor_updates_paused = bool(
                state.get("actor_updates_paused", False)
            )
            self.anchor_schedule_frozen = bool(
                state.get("anchor_schedule_frozen", False)
            )
            self.pause_reason = state.get("pause_reason")
            self.baseline_hard_termination_sum = float(
                state.get("baseline_hard_termination_sum", 0.0)
            )
            self.baseline_hard_termination_windows = int(
                state.get("baseline_hard_termination_windows", 0)
            )
            baseline_rate = state.get("baseline_hard_termination_rate")
            self.baseline_hard_termination_rate = (
                float(baseline_rate) if baseline_rate is not None else None
            )
            self.last_anchor_action_mse = float(
                state.get("last_anchor_action_mse", 0.0)
            )
            # Model/optimizer RNG resumes in aisrv. Worker command state is
            # intentionally task-local and starts from t=0 in the new worker.
            self._restore_rng_state(state.get("rng_state"), self.logger)
            self.s0_checkpoint_sha256 = str(
                checkpoint.get("lineage", {}).get(
                    "s0_checkpoint_sha256", "unknown"
                )
            )
            self.loaded_platform_model_id = checkpoint.get("platform_model_id")
            self.loaded_schedule_mode = saved_schedule_mode
            self.resume_loaded = True
            load_mode = (
                LOAD_MODE_TRANSITION_RESUME
                if transition_resume
                else LOAD_MODE_ANCHOR_RESUME
            )
        else:
            # S0 first-load or schedule migration: fresh critic/optimizer/clock.
            # RNG is reinitialized from env_seed (§5.4 N1). S0 has no independent
            # RNG state; frozen S0 only runs eval() and consumes no RNG.
            self._reseed(env_seed)
            self.anchor_session_elapsed_hours = 0.0
            self.elapsed_training_hours = 0.0
            self.actor_updates_paused = False
            self.anchor_schedule_frozen = False
            self.pause_reason = None
            self.baseline_hard_termination_sum = 0.0
            self.baseline_hard_termination_windows = 0
            self.baseline_hard_termination_rate = None
            self.last_anchor_action_mse = 0.0
            self.skipped_nonfinite_updates = 0
            self.resume_loaded = False
            if other_visual_schedule:
                # Migration: warn but accept VisionEncoder/Actor as new parent.
                self.loaded_schedule_mode = saved_schedule_mode
                self.loaded_platform_model_id = checkpoint.get("platform_model_id")
                if self.logger:
                    self.logger.warning(
                        f"[VisualPPO] schedule migration: bundle "
                        f"schedule_mode={saved_schedule_mode!r} differs from "
                        f"active {self.schedule_mode!r}; treating "
                        "VisionEncoder/Actor as a new parent, resetting critic/"
                        "optimizer/clock/RNG."
                    )
                load_mode = LOAD_MODE_SCHEDULE_MIGRATION
            else:
                # Exact S0 first-load.
                self.s0_checkpoint_sha256 = self._sha256(path)
                load_mode = LOAD_MODE_S0

        # Recompute phase/weights from the (possibly restored) session clock.
        self.current_phase = self._phase_for_elapsed(self.anchor_session_elapsed_hours)
        self.action_anchor_weight = self._anchor_weight_for_elapsed(
            self.anchor_session_elapsed_hours, kind="action"
        )
        self.latent_anchor_weight_current = self._anchor_weight_for_elapsed(
            self.anchor_session_elapsed_hours, kind="latent"
        )
        self._set_trainable_phase(self.current_phase)
        self._set_phase_learning_rates(self.current_phase)

        self.initialize_recurrent_states(
            self.storage.num_envs if self.storage is not None else 1
        )
        return load_mode

    def load_low_level_only_bundle(
        self,
        path: str,
        *,
        expected_spec: dict[str, int],
        env_seed: int | None = None,
        restore_optimizer: bool = False,
    ) -> str:
        """Explicitly restore Standard low-level state while ignoring adapters."""
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        if not is_kaiwu_train_bundle(checkpoint):
            raise ValueError(f"Expected kaiwu_train_v1 checkpoint: {path}")
        validate_low_level_spec(checkpoint, expected_spec)
        modules = checkpoint.get("modules", {})
        vision_state = modules.get("vision_encoder", {}).get("state_dict")
        actor_state = modules.get("low_level", {}).get("actor_state_dict")
        critic_state = modules.get("critic", {}).get("state_dict")
        if not all(isinstance(value, dict) for value in (vision_state, actor_state, critic_state)):
            raise KeyError(
                "low_level_only preload requires vision_encoder, low_level actor, and critic"
            )
        state_tensors = [
            value
            for state_dict in (vision_state, actor_state, critic_state)
            for value in state_dict.values()
            if torch.is_tensor(value)
        ]
        if not all(bool(torch.isfinite(value).all()) for value in state_tensors):
            raise ValueError("low_level_only checkpoint contains nonfinite weights")
        self.actor_critic.vision_encoder.load_state_dict(vision_state, strict=True)
        self.actor_critic.actor.load_state_dict(actor_state, strict=True)
        self.actor_critic.critic.load_state_dict(critic_state, strict=True)
        std_value = modules.get("action_distribution", {}).get("std")
        if std_value is not None and hasattr(self.actor_critic, "std"):
            if not bool(torch.isfinite(std_value).all()):
                raise ValueError("low_level_only checkpoint contains nonfinite action std")
            self.actor_critic.std.data.copy_(std_value.to(self.device))
        anchor = modules.get("s0_anchor", {})
        self.anchor_encoder.load_state_dict(
            anchor.get("vision_encoder_state_dict", vision_state), strict=True
        )
        self.anchor_actor.load_state_dict(
            anchor.get("actor_state_dict", actor_state), strict=True
        )

        state = checkpoint.get("training_state", {})
        if not isinstance(state, dict):
            state = {}
        optimizer_state = checkpoint.get("optimizers", {}).get("visual_ppo")
        optimizer_restored = restore_optimizer and isinstance(optimizer_state, dict)
        if optimizer_restored:
            self.optimizer.load_state_dict(optimizer_state)
            self.current_iteration = int(state.get("current_iteration", 0))
            self.skipped_nonfinite_updates = int(
                state.get("skipped_nonfinite_updates", 0)
            )
            self._restore_rng_state(state.get("rng_state"), self.logger)
        else:
            self._reseed(env_seed)
            self.current_iteration = int(state.get("current_iteration", 0))
            self.skipped_nonfinite_updates = 0
            if restore_optimizer and self.logger:
                self.logger.warning(
                    "[VisualPPO] low_level_only optimizer was requested but is missing; "
                    "continuing as weights-only warm start"
                )
        self.anchor_session_elapsed_hours = 0.0
        self.elapsed_training_hours = 0.0
        self.loaded_platform_model_id = checkpoint.get("platform_model_id")
        self.loaded_schedule_mode = state.get("schedule_mode")
        self.resume_loaded = optimizer_restored
        self.current_phase = self._phase_for_elapsed(0.0)
        self._set_trainable_phase(self.current_phase)
        self._set_phase_learning_rates(self.current_phase)
        self.initialize_recurrent_states(
            self.storage.num_envs if self.storage is not None else 1
        )
        if self.logger:
            self.logger.warning(
                "[VisualPPO] explicit low_level_only preload completed; "
                "modules.high_level was intentionally ignored "
                f"optimizer_restored={optimizer_restored}"
            )
        return (
            "low_level_only_resume"
            if optimizer_restored
            else "low_level_only_warm_start"
        )

    def _reseed(self, env_seed: int | None) -> None:
        """Reinitialize process-level RNG from the active TOML seed (§5.4 N1).

        Used on S0 first-load and schedule migration (paths that ignore the
        bundle RNG). No second seed config; env_seed comes from [env_conf].seed.
        """
        seed = 0 if env_seed is None else int(env_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
