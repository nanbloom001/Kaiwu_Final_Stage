#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Constrained recurrent PPO for the frozen Standard visual baseline."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO
from agent_ppo.checkpoint_io import (
    KAIWU_TRAIN_FORMAT,
    KAIWU_TRAIN_SCHEMA_VERSION,
    is_kaiwu_train_bundle,
    validate_low_level_spec,
)
from agent_ppo.feature.definition import RecurrentRolloutStorage


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
        critic_only_hours: float,
        actor_only_end_hours: float,
        task_end_hours: float,
        action_anchor_start: float,
        action_anchor_mid: float,
        action_anchor_end: float,
        latent_anchor_weight: float,
        max_anchor_action_mse: float,
        max_hard_termination_delta: float,
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
        self.critic_only_hours = float(critic_only_hours)
        self.actor_only_end_hours = float(actor_only_end_hours)
        self.task_end_hours = float(task_end_hours)
        self.action_anchor_start = float(action_anchor_start)
        self.action_anchor_mid = float(action_anchor_mid)
        self.action_anchor_end = float(action_anchor_end)
        self.latent_anchor_weight = float(latent_anchor_weight)
        self.max_anchor_action_mse = float(max_anchor_action_mse)
        self.max_hard_termination_delta = float(max_hard_termination_delta)

        self.elapsed_training_hours = 0.0
        self.current_iteration = 0
        self.current_phase = "rlcritic"
        self.action_anchor_weight = self.action_anchor_start
        self.actor_updates_paused = False
        self.pause_reason = None
        self.baseline_hard_termination_sum = 0.0
        self.baseline_hard_termination_windows = 0
        self.baseline_hard_termination_rate = None
        self.last_anchor_action_mse = 0.0
        self.s0_checkpoint_sha256 = "unknown"
        self.resume_loaded = False

        for module in (self.anchor_encoder, self.anchor_actor):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

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

    def _phase_for_elapsed(self, elapsed_h: float) -> str:
        if elapsed_h < self.critic_only_hours:
            return "rlcritic"
        if elapsed_h < self.actor_only_end_hours:
            return "rlactor"
        return "rlfull"

    def _anchor_weight_for_elapsed(self, elapsed_h: float) -> float:
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
        actor_enabled = phase in {"rlactor", "rlfull"} and not self.actor_updates_paused
        recurrent_enabled = phase == "rlfull" and not self.actor_updates_paused

        for parameter in self.actor_critic.actor.parameters():
            parameter.requires_grad_(actor_enabled)
        for parameter in self.actor_critic.critic.parameters():
            parameter.requires_grad_(True)
        if hasattr(self.actor_critic, "std"):
            self.actor_critic.std.requires_grad_(actor_enabled)
        if hasattr(self.actor_critic, "log_std"):
            self.actor_critic.log_std.requires_grad_(actor_enabled)

        for parameter in self.actor_critic.vision_encoder.cnn.parameters():
            parameter.requires_grad_(False)
        for parameter in self.actor_critic.vision_encoder.rnn.parameters():
            parameter.requires_grad_(recurrent_enabled)
        for parameter in self.actor_critic.vision_encoder.rnn_output_layer.parameters():
            parameter.requires_grad_(recurrent_enabled)

    def _update_safety_state(self, elapsed_h: float) -> float:
        hard_rate = float(self.storage.hard_terminations.float().mean().item())
        if elapsed_h < self.critic_only_hours:
            self.baseline_hard_termination_sum += hard_rate
            self.baseline_hard_termination_windows += 1
            self.baseline_hard_termination_rate = (
                self.baseline_hard_termination_sum
                / max(1, self.baseline_hard_termination_windows)
            )
            self.actor_updates_paused = False
            self.pause_reason = None
            return hard_rate

        if self.baseline_hard_termination_rate is None:
            self.baseline_hard_termination_rate = hard_rate
        hard_limit = (
            self.baseline_hard_termination_rate
            + self.max_hard_termination_delta
        )
        reasons = []
        if hard_rate > hard_limit:
            reasons.append(f"hard_termination={hard_rate:.4f}>{hard_limit:.4f}")
        if self.last_anchor_action_mse > self.max_anchor_action_mse:
            reasons.append(
                f"anchor_action_mse={self.last_anchor_action_mse:.4f}>"
                f"{self.max_anchor_action_mse:.4f}"
            )
        self.actor_updates_paused = bool(reasons)
        self.pause_reason = "; ".join(reasons) if reasons else None
        return hard_rate

    def learn(self, elapsed_h: float | None = None) -> dict[str, float]:
        elapsed_h = (
            self.elapsed_training_hours if elapsed_h is None else float(elapsed_h)
        )
        self.elapsed_training_hours = elapsed_h
        self.current_phase = self._phase_for_elapsed(elapsed_h)
        self.action_anchor_weight = self._anchor_weight_for_elapsed(elapsed_h)
        hard_rate = self._update_safety_state(elapsed_h)
        self._set_trainable_phase(self.current_phase)

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
            action_anchor_loss = F.smooth_l1_loss(
                action_mean, anchor_actions_batch
            )
            latent_anchor_loss = F.smooth_l1_loss(
                latent, anchor_latents_batch
            )
            anchor_action_mse = F.mse_loss(action_mean, anchor_actions_batch)

            latent_weight = (
                self.latent_anchor_weight
                if self.current_phase == "rlfull"
                and not self.actor_updates_paused
                else 0.0
            )
            loss = (
                policy_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy
                + self.action_anchor_weight * action_anchor_loss
                + latent_weight * latent_anchor_loss
            )
            if not torch.isfinite(loss):
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
                "actor_updates_paused": float(self.actor_updates_paused),
                "elapsed_training_hours": elapsed_h,
                "applied_updates": float(applied_updates),
            }
        )
        self.last_anchor_action_mse = metrics["anchor_action_mse"]
        self.current_iteration += 1
        self.train_step += 1
        return metrics

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
                "elapsed_training_hours": self.elapsed_training_hours,
                "phase_label": phase_label,
                "action_anchor_weight": self.action_anchor_weight,
                "actor_updates_paused": self.actor_updates_paused,
                "pause_reason": self.pause_reason,
                "baseline_hard_termination_sum": self.baseline_hard_termination_sum,
                "baseline_hard_termination_windows": self.baseline_hard_termination_windows,
                "baseline_hard_termination_rate": self.baseline_hard_termination_rate,
                "last_anchor_action_mse": self.last_anchor_action_mse,
                "rng_state": self._capture_rng_state(),
            },
            "lineage": {"s0_checkpoint_sha256": self.s0_checkpoint_sha256},
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

    def load_training_bundle(
        self,
        path: str,
        *,
        expected_spec: dict[str, int],
    ) -> str:
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

        is_resume = checkpoint.get("stage_type") == self.STAGE_TYPE
        anchor_section = modules.get("s0_anchor", {}) if is_resume else {}
        anchor_vision = anchor_section.get(
            "vision_encoder_state_dict", vision_state
        )
        anchor_actor = anchor_section.get("actor_state_dict", actor_state)
        self.anchor_encoder.load_state_dict(anchor_vision, strict=True)
        self.anchor_actor.load_state_dict(anchor_actor, strict=True)

        if is_resume:
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
            self.elapsed_training_hours = float(
                state.get("elapsed_training_hours", 0.0)
            )
            restored_phase = state.get("phase_label")
            expected_phase = self._phase_for_elapsed(
                self.elapsed_training_hours
            )
            self.current_phase = (
                restored_phase
                if restored_phase in {"rlcritic", "rlactor", "rlfull"}
                else expected_phase
            )
            if self.current_phase != expected_phase:
                if self.logger:
                    self.logger.warning(
                        "[VisualPPO] checkpoint phase does not match elapsed "
                        f"time: saved={self.current_phase}, "
                        f"expected={expected_phase}; using elapsed-time phase"
                    )
                self.current_phase = expected_phase
            self.action_anchor_weight = float(
                state.get("action_anchor_weight", self.action_anchor_start)
            )
            self.actor_updates_paused = bool(
                state.get("actor_updates_paused", False)
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
            self._restore_rng_state(state.get("rng_state"), self.logger)
            self.s0_checkpoint_sha256 = str(
                checkpoint.get("lineage", {}).get(
                    "s0_checkpoint_sha256", "unknown"
                )
            )
            self.resume_loaded = True
        else:
            self.s0_checkpoint_sha256 = self._sha256(path)
            self.resume_loaded = False

        self.initialize_recurrent_states(
            self.storage.num_envs if self.storage is not None else 1
        )
        return "resume" if is_resume else "s0"
