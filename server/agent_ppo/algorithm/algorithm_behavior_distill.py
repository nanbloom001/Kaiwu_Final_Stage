#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Guarded behavior distillation for the Standard 301-D reference policy.

The source teacher is the frozen flat Standard policy::

    [proprio45 | height_scan256] -> Actor301 -> action12

The student keeps the same observable inputs but introduces the modular
height-scan encoder required by the later visual LBC stage::

    height_scan256 -> encoder -> latent32
    [proprio45 | latent32] -> Actor77 -> action12

This module deliberately contains no reward or PPO update.  It supports
per-environment DAgger, post-step sample weighting and a complete resumable
checkpoint.  The resulting privileged teacher is not a deployable artifact.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.model.actor_critic import ActorCritic


BEHAVIOR_DISTILL_FORMAT = "behavior_distill_v2"
PRIVILEGED_TEACHER_FORMAT = "privileged_loco_teacher_v1"
CHECKPOINT_SCHEMA_VERSION = 1


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AlgorithmBehaviorDistill:
    """Action-supervised bridge from a flat teacher to ActorCriticEncoder."""

    def __init__(
        self,
        student: nn.Module,
        teacher_ckpt: str | None = None,
        device: str = "cuda:0",
        learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
        action_loss_weight: float = 1.0,
        num_obs: int = 301,
        num_critic_obs: int = 316,
        num_actions: int = 12,
        teacher_actor_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        teacher_critic_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        teacher_activation: str = "elu",
        expected_teacher_sha256: str | None = None,
        stage_name: str = "standard_bridge_r1",
        logger=None,
    ) -> None:
        self.student = student.to(device)
        self.actor_critic = self.student
        self.device = device
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.action_loss_weight = action_loss_weight
        self.num_obs = int(num_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.expected_teacher_sha256 = (
            str(expected_teacher_sha256).lower() if expected_teacher_sha256 else None
        )
        self.stage_name = stage_name
        self.logger = logger

        self.teacher = ActorCritic(
            num_obs=self.num_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            actor_hidden_dims=teacher_actor_hidden_dims,
            critic_hidden_dims=teacher_critic_hidden_dims,
            activation=teacher_activation,
        ).to(device)
        self.teacher_loaded = False
        self.teacher_source = None
        self.teacher_sha256 = None
        self._initial_teacher_state: dict[str, torch.Tensor] | None = None

        if not hasattr(self.student, "encoder") or not hasattr(self.student, "actor"):
            raise TypeError(
                "Behavior-distill student must expose encoder and actor modules."
            )
        student_parameters = [
            *self.student.encoder.parameters(),
            *self.student.actor.parameters(),
        ]
        self.optimizer = torch.optim.Adam(student_parameters, lr=learning_rate)

        self.current_iteration = 0
        self.total_steps = 0
        self.gradient_steps = 0
        self.dagger_phase_index = 0
        self.dagger_phase_iteration = 0
        self.student_drive_probability = 0.0
        self.safety_threshold = float("inf")
        self.previous_hard_termination_rate: float | None = None
        self.training_status = "initialized"
        self.gate_history: list[dict[str, Any]] = []
        self.recent_iteration_metrics: list[dict[str, float]] = []
        self.config_sha256 = "unknown"
        self.code_commit = "unknown"
        self.loss_buffer = deque(maxlen=100)

        if teacher_ckpt:
            self._load_teacher(teacher_ckpt)
        self._freeze_teacher()

    def _load_teacher(self, teacher_ckpt: str) -> None:
        checkpoint_sha = sha256_file(teacher_ckpt)
        ckpt = torch.load(teacher_ckpt, weights_only=False, map_location=self.device)
        self.load_teacher_state_dict(
            ckpt,
            source=teacher_ckpt,
            source_sha256=checkpoint_sha,
        )

    def load_teacher_state_dict(
        self,
        ckpt: dict,
        source: str = "<preload>",
        source_sha256: str | None = None,
    ) -> None:
        """Strictly load the original flat 301-D teacher.

        Packaged bridge/LBC/visual checkpoints are intentionally rejected here;
        resuming a bridge checkpoint uses :meth:`load_checkpoint_dict` instead.
        """
        if isinstance(ckpt, dict) and ckpt.get("format"):
            raise ValueError(
                f"Flat teacher must be a raw state dict, got format={ckpt.get('format')!r}"
            )
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        if not isinstance(ckpt, dict):
            raise TypeError(
                f"Flat standard teacher must be a state-dict, got {type(ckpt).__name__}"
            )

        expected = self.teacher.state_dict()
        missing = sorted(set(expected) - set(ckpt))
        unexpected = sorted(set(ckpt) - set(expected))
        mismatched = sorted(
            key
            for key in set(expected) & set(ckpt)
            if not hasattr(ckpt[key], "shape")
            or tuple(ckpt[key].shape) != tuple(expected[key].shape)
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "Flat standard teacher contract mismatch: "
                f"missing={missing}, unexpected={unexpected}, "
                f"shape_mismatch={mismatched}"
            )

        if source_sha256 is None and os.path.isfile(source):
            source_sha256 = sha256_file(source)
        if source_sha256:
            source_sha256 = source_sha256.lower()
        if (
            self.expected_teacher_sha256
            and source_sha256 != self.expected_teacher_sha256
        ):
            raise ValueError(
                "Flat teacher SHA256 mismatch: "
                f"expected={self.expected_teacher_sha256}, got={source_sha256}, "
                f"source={source}"
            )

        self.teacher.load_state_dict(ckpt, strict=True)
        self.teacher_loaded = True
        self.teacher_source = source
        self.teacher_sha256 = source_sha256 or "unknown"
        self._freeze_teacher()
        self._initial_teacher_state = {
            name: value.detach().clone()
            for name, value in self.teacher.state_dict().items()
        }
        if self.logger is not None:
            self.logger.info(
                "[BehaviorDistill] loaded strict flat Standard teacher "
                f"from {source}, sha256={self.teacher_sha256}"
            )

    def _freeze_teacher(self) -> None:
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

    def assert_teacher_ready(self) -> None:
        if not self.teacher_loaded:
            raise RuntimeError(
                "[BehaviorDistill] strict 301-D teacher was not loaded. Select the "
                "original Standard 10288 checkpoint before starting STD-BRIDGE-R1."
            )
        frozen = not self.teacher.training and all(
            not parameter.requires_grad for parameter in self.teacher.parameters()
        )
        optimizer_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        expected_ids = {
            id(parameter)
            for module in (self.student.encoder, self.student.actor)
            for parameter in module.parameters()
        }
        teacher_ids = {id(parameter) for parameter in self.teacher.parameters()}
        if not frozen or optimizer_ids != expected_ids or optimizer_ids & teacher_ids:
            raise RuntimeError(
                "[BehaviorDistill] freeze/optimizer contract failed: "
                f"teacher_frozen={frozen}, student_encoder_actor_only="
                f"{optimizer_ids == expected_ids}, teacher_in_optimizer="
                f"{bool(optimizer_ids & teacher_ids)}"
            )

    def teacher_max_abs_diff(self) -> float:
        if self._initial_teacher_state is None:
            return float("nan")
        return max(
            float(
                torch.max(
                    torch.abs(value.detach() - self._initial_teacher_state[name])
                ).item()
            )
            for name, value in self.teacher.state_dict().items()
        )

    def set_run_metadata(self, config_sha256: str, code_commit: str) -> None:
        self.config_sha256 = config_sha256
        self.code_commit = code_commit

    def prepare_update(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Build one differentiable student batch and frozen teacher labels."""
        self.assert_teacher_ready()
        obs = torch.as_tensor(obs, device=self.device)
        if obs.ndim != 2 or obs.shape[-1] != self.num_obs:
            raise ValueError(
                f"Behavior-distill obs must be [B,{self.num_obs}], got {tuple(obs.shape)}"
            )
        with torch.no_grad():
            teacher_action = self.teacher.act_inference(obs)
        if not bool(torch.isfinite(teacher_action).all().item()):
            raise FloatingPointError("Frozen teacher produced NaN/Inf action.")

        student_action = self.student.act_inference(obs)
        student_finite = torch.isfinite(student_action).all(dim=-1)
        safe_student_action = torch.where(
            student_finite.unsqueeze(-1), student_action, teacher_action.detach()
        )
        difference = safe_student_action - teacher_action
        per_sample_mse = difference.pow(2).mean(dim=-1)
        per_sample_l2 = difference.pow(2).sum(dim=-1).sqrt()
        per_sample_cos = F.cosine_similarity(
            safe_student_action, teacher_action, dim=-1
        )
        return {
            "obs": obs,
            "teacher_action": teacher_action,
            "student_action": student_action,
            "safe_student_action": safe_student_action,
            "student_finite": student_finite,
            "difference": difference,
            "per_sample_mse": per_sample_mse,
            "per_sample_l2": per_sample_l2,
            "per_sample_cos": per_sample_cos,
        }

    def select_driver_actions(
        self,
        batch: Dict[str, torch.Tensor],
        student_drive_probability: float,
        safety_threshold: float,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Sample a driver independently for every environment."""
        probability = float(student_drive_probability)
        if probability < 0.0 or probability > 1.0:
            raise ValueError("student_drive_probability must stay in [0, 1]")
        num_envs = batch["teacher_action"].shape[0]
        requested_student = torch.rand(num_envs, device=self.device) < probability
        excessive_disagreement = batch["per_sample_l2"] > float(safety_threshold)
        safety_takeover = requested_student & (
            ~batch["student_finite"] | excessive_disagreement
        )
        effective_student = requested_student & ~safety_takeover
        actions = torch.where(
            effective_student.unsqueeze(-1),
            batch["safe_student_action"].detach(),
            batch["teacher_action"].detach(),
        )
        selection = {
            "requested_student": requested_student,
            "effective_student": effective_student,
            "safety_takeover": safety_takeover,
        }
        return actions, selection

    def finish_update(
        self,
        batch: Dict[str, torch.Tensor],
        sample_weights: torch.Tensor,
    ) -> Dict[str, float]:
        """Apply post-step weighted action supervision."""
        weights = torch.as_tensor(
            sample_weights,
            device=self.device,
            dtype=batch["per_sample_mse"].dtype,
        ).reshape(-1)
        if weights.shape[0] != batch["per_sample_mse"].shape[0]:
            raise ValueError(
                "sample_weights size mismatch: "
                f"{weights.shape[0]} != {batch['per_sample_mse'].shape[0]}"
            )
        weights = torch.clamp(torch.nan_to_num(weights, nan=0.0), 0.0, 1.0)
        denominator = weights.sum()
        weighted_count = float(denominator.item())
        update_skipped_nonfinite = 0.0

        if weighted_count > 0.0:
            # Re-run only the rows that are allowed to contribute gradients.
            # Masking a NaN row after a full-batch forward is not sufficient:
            # zero upstream gradients can still meet non-finite activations in
            # shared Linear backward kernels and poison otherwise valid rows.
            train_mask = (weights > 0.0) & batch["student_finite"]
            train_obs = batch["obs"][train_mask]
            train_teacher = batch["teacher_action"][train_mask].detach()
            train_weights = weights[train_mask]
            train_student = (
                self.student.act_inference(train_obs)
                if train_obs.shape[0] > 0
                else None
            )
            if train_student is None or not bool(
                torch.isfinite(train_student).all().item()
            ):
                action_mse = batch["per_sample_mse"].sum() * 0.0
                loss = action_mse
                grad_norm = torch.zeros((), device=self.device)
                update_skipped_nonfinite = 1.0
            else:
                train_mse = (train_student - train_teacher).pow(2).mean(dim=-1)
                train_denominator = train_weights.sum()
                action_mse = (
                    train_weights * train_mse
                ).sum() / train_denominator
                loss = self.action_loss_weight * action_mse
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for group in self.optimizer.param_groups
                        for parameter in group["params"]
                    ],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.gradient_steps += 1
        else:
            action_mse = batch["per_sample_mse"].sum() * 0.0
            loss = action_mse
            grad_norm = torch.zeros((), device=self.device)

        self.total_steps += int(batch["teacher_action"].shape[0])
        loss_value = float(loss.detach().item())
        self.loss_buffer.append(loss_value)

        with torch.no_grad():
            safe_denominator = denominator.clamp_min(1.0)
            action_l2 = (weights * batch["per_sample_l2"]).sum() / safe_denominator
            action_cos = (weights * batch["per_sample_cos"]).sum() / safe_denominator
            teacher_abs = batch["teacher_action"].abs().mean()
            student_abs = batch["safe_student_action"].abs().mean()
            teacher_variance = batch["teacher_action"].var(
                dim=0, unbiased=False
            ).clamp_min(1.0e-6)
            per_dimension_mse = (
                weights.unsqueeze(-1) * batch["difference"].pow(2)
            ).sum(dim=0) / safe_denominator
            normalized_action_mse = (per_dimension_mse / teacher_variance).mean()
            action_l2_p95 = torch.quantile(batch["per_sample_l2"], 0.95)

        return {
            "loss": loss_value,
            "action_mse": float(action_mse.detach().item()),
            "normalized_action_mse": float(normalized_action_mse.item()),
            "action_l2": float(action_l2.item()),
            "action_l2_p95": float(action_l2_p95.item()),
            "action_cos": float(action_cos.item()),
            "teacher_abs": float(teacher_abs.item()),
            "student_abs": float(student_abs.item()),
            "grad_norm": float(grad_norm.detach().item()),
            "weighted_sample_rate": float((weights > 0.0).float().mean().item()),
            "mean_sample_weight": float(weights.mean().item()),
            "nonfinite_rate": float((~batch["student_finite"]).float().mean().item()),
            "update_skipped_nonfinite": update_skipped_nonfinite,
        }

    def act_teacher(self, obs: torch.Tensor) -> torch.Tensor:
        self.assert_teacher_ready()
        with torch.no_grad():
            return self.teacher.act_inference(torch.as_tensor(obs, device=self.device))

    def act_student(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.student.act_inference(torch.as_tensor(obs, device=self.device))

    def compute_metrics(self, obs: torch.Tensor) -> Dict[str, float]:
        batch = self.prepare_update(obs)
        weights = torch.ones(batch["teacher_action"].shape[0], device=self.device)
        with torch.no_grad():
            return self._metrics_without_update(batch, weights)

    def _metrics_without_update(
        self, batch: Dict[str, torch.Tensor], weights: torch.Tensor
    ) -> Dict[str, float]:
        weights = weights.to(device=self.device, dtype=batch["per_sample_mse"].dtype)
        denominator = weights.sum().clamp_min(1.0)
        teacher_variance = batch["teacher_action"].var(
            dim=0, unbiased=False
        ).clamp_min(1.0e-6)
        per_dimension_mse = (
            weights.unsqueeze(-1) * batch["difference"].pow(2)
        ).sum(dim=0) / denominator
        return {
            "loss": float((weights * batch["per_sample_mse"]).sum().item() / denominator.item()),
            "action_mse": float((weights * batch["per_sample_mse"]).sum().item() / denominator.item()),
            "normalized_action_mse": float((per_dimension_mse / teacher_variance).mean().item()),
            "action_l2": float((weights * batch["per_sample_l2"]).sum().item() / denominator.item()),
            "action_l2_p95": float(torch.quantile(batch["per_sample_l2"], 0.95).item()),
            "action_cos": float((weights * batch["per_sample_cos"]).sum().item() / denominator.item()),
            "teacher_abs": float(batch["teacher_action"].abs().mean().item()),
            "student_abs": float(batch["safe_student_action"].abs().mean().item()),
            "grad_norm": 0.0,
            "weighted_sample_rate": 1.0,
            "mean_sample_weight": 1.0,
            "nonfinite_rate": float((~batch["student_finite"]).float().mean().item()),
        }

    def update(self, obs: torch.Tensor) -> Dict[str, float]:
        """Backward-compatible all-sample supervised update."""
        batch = self.prepare_update(obs)
        weights = torch.ones(batch["teacher_action"].shape[0], device=self.device)
        return self.finish_update(batch, weights)

    def _rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict[str, Any] | None) -> None:
        if not isinstance(state, dict):
            return
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def model_spec(self) -> dict[str, Any]:
        return {
            "teacher_observation_dim": self.num_obs,
            "teacher_critic_observation_dim": self.num_critic_obs,
            "proprio_dim": int(getattr(self.student, "num_proprio", 45)),
            "scan_dim": int(getattr(self.student, "num_scan", 256)),
            "latent_dim": int(getattr(self.student, "latent_dim", 32)),
            "actor_input_dim": int(getattr(self.student, "num_proprio", 45))
            + int(getattr(self.student, "latent_dim", 32)),
            "action_dim": self.num_actions,
            "goal_dim": int(getattr(self.student, "num_goal_obs", 0)),
        }

    def checkpoint_payload(self, **kwargs: Any) -> dict[str, Any]:
        payload = {
            "format": BEHAVIOR_DISTILL_FORMAT,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage": self.stage_name,
            "model_spec": self.model_spec(),
            "student_model_state_dict": self.student.state_dict(),
            # Required for a true one-file resume on the platform.
            "teacher_model_state_dict": self.teacher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_iteration": self.current_iteration,
            "total_steps": self.total_steps,
            "gradient_steps": self.gradient_steps,
            "dagger_phase_index": self.dagger_phase_index,
            "dagger_phase_iteration": self.dagger_phase_iteration,
            "student_drive_probability": self.student_drive_probability,
            "safety_threshold": self.safety_threshold,
            "previous_hard_termination_rate": self.previous_hard_termination_rate,
            "gate_history": self.gate_history,
            "recent_iteration_metrics": self.recent_iteration_metrics,
            "rng_state": self._rng_state(),
            "teacher_source": self.teacher_source,
            "teacher_sha256": self.teacher_sha256,
            "config_sha256": self.config_sha256,
            "code_commit": self.code_commit,
            "critic_trained": False,
            "training_status": self.training_status,
        }
        payload.update(kwargs)
        return payload

    def save(self, path: str, **kwargs: Any) -> str:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.checkpoint_payload(**kwargs), path)
        return sha256_file(path)

    def save_privileged_teacher(
        self,
        path: str,
        bridge_checkpoint_sha256: str,
    ) -> str:
        payload = {
            "format": PRIVILEGED_TEACHER_FORMAT,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage": self.stage_name,
            "encoder_state_dict": self.student.encoder.state_dict(),
            "actor_state_dict": self.student.actor.state_dict(),
            "model_spec": self.model_spec(),
            "source_teacher_sha256": self.teacher_sha256,
            "bridge_checkpoint_sha256": bridge_checkpoint_sha256,
            "critic_trained": False,
            "deployable": False,
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(payload, path)
        return sha256_file(path)

    def load_checkpoint_dict(
        self,
        checkpoint: dict[str, Any],
        source: str,
        *,
        load_optimizer: bool,
        load_teacher: bool,
        restore_rng: bool,
    ) -> None:
        if checkpoint.get("format") != BEHAVIOR_DISTILL_FORMAT:
            raise ValueError(
                f"Expected {BEHAVIOR_DISTILL_FORMAT}, got {checkpoint.get('format')!r}"
            )
        if int(checkpoint.get("schema_version", 0)) != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported behavior checkpoint schema={checkpoint.get('schema_version')!r}"
            )
        state = checkpoint.get("student_model_state_dict")
        if not isinstance(state, dict):
            raise KeyError("student_model_state_dict missing from behavior checkpoint")
        self.student.load_state_dict(state, strict=True)

        if load_teacher:
            teacher_state = checkpoint.get("teacher_model_state_dict")
            if not isinstance(teacher_state, dict):
                raise KeyError(
                    "teacher_model_state_dict missing; this checkpoint cannot resume "
                    "without the original teacher"
                )
            teacher_sha = checkpoint.get("teacher_sha256")
            if self.expected_teacher_sha256 and teacher_sha != self.expected_teacher_sha256:
                raise ValueError(
                    "Resumed teacher SHA mismatch: "
                    f"expected={self.expected_teacher_sha256}, got={teacher_sha}"
                )
            expected = self.teacher.state_dict()
            if set(teacher_state) != set(expected) or any(
                teacher_state[key].shape != expected[key].shape for key in expected
            ):
                raise ValueError("Packaged flat teacher state does not match 301-D contract")
            self.teacher.load_state_dict(teacher_state, strict=True)
            self.teacher_loaded = True
            self.teacher_source = checkpoint.get("teacher_source", source)
            self.teacher_sha256 = teacher_sha or "unknown"
            self._freeze_teacher()
            self._initial_teacher_state = {
                name: value.detach().clone()
                for name, value in self.teacher.state_dict().items()
            }

        if load_optimizer:
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if not isinstance(optimizer_state, dict):
                raise KeyError("optimizer_state_dict missing from behavior checkpoint")
            self.optimizer.load_state_dict(optimizer_state)

        self.current_iteration = int(checkpoint.get("current_iteration", 0))
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.gradient_steps = int(checkpoint.get("gradient_steps", 0))
        self.dagger_phase_index = int(checkpoint.get("dagger_phase_index", 0))
        self.dagger_phase_iteration = int(checkpoint.get("dagger_phase_iteration", 0))
        self.student_drive_probability = float(
            checkpoint.get("student_drive_probability", 0.0)
        )
        self.safety_threshold = float(checkpoint.get("safety_threshold", float("inf")))
        previous_rate = checkpoint.get("previous_hard_termination_rate")
        self.previous_hard_termination_rate = (
            None if previous_rate is None else float(previous_rate)
        )
        self.gate_history = list(checkpoint.get("gate_history", []))
        self.recent_iteration_metrics = list(
            checkpoint.get("recent_iteration_metrics", [])
        )
        self.config_sha256 = str(checkpoint.get("config_sha256", "unknown"))
        self.code_commit = str(checkpoint.get("code_commit", "unknown"))
        self.training_status = str(checkpoint.get("training_status", "resumed"))
        if restore_rng:
            self._restore_rng_state(checkpoint.get("rng_state"))

    def load_checkpoint(
        self,
        path: str,
        *,
        load_optimizer: bool = True,
        load_teacher: bool = True,
        restore_rng: bool = True,
    ) -> None:
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Behavior checkpoint at {path} is not a dictionary")
        self.load_checkpoint_dict(
            checkpoint,
            source=path,
            load_optimizer=load_optimizer,
            load_teacher=load_teacher,
            restore_rng=restore_rng,
        )

    def load_student_weights(self, state: dict, source: str = "<legacy>") -> None:
        """Explicit weight-only import for old raw bridge artifacts."""
        if isinstance(state, dict) and state.get("format") == BEHAVIOR_DISTILL_FORMAT:
            state = state.get("student_model_state_dict")
        if not isinstance(state, dict):
            raise TypeError(f"Student weights from {source} are not a state dict")
        self.student.load_state_dict(state, strict=True)
