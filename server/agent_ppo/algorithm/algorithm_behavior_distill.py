#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Protected DAgger bridge from the flat Standard teacher to Actor77."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.checkpoint_io import (
    KAIWU_TRAIN_FORMAT,
    KAIWU_TRAIN_SCHEMA_VERSION,
    is_kaiwu_train_bundle,
    low_level_policy_state,
    phase_label,
    validate_low_level_spec,
)
from agent_ppo.model.actor_critic import ActorCritic


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_to_cpu(item) for item in value)
    return copy.deepcopy(value)


class AlgorithmBehaviorDistill:
    """Action-supervised DAgger with a frozen flat301 teacher."""

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
        replay_capacity: int = 8192,
        replay_batch_size: int = 256,
        replay_loss_ratio: float = 0.25,
        replay_add_per_step: int = 64,
        logger=None,
    ) -> None:
        self.student = student.to(device)
        self.actor_critic = self.student
        self.device = device
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = float(max_grad_norm)
        self.action_loss_weight = float(action_loss_weight)
        self.num_obs = int(num_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
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
        self.teacher_sha256 = "unknown"
        self._initial_teacher_state: dict[str, torch.Tensor] | None = None

        if not hasattr(self.student, "encoder") or not hasattr(self.student, "actor"):
            raise TypeError("Behavior-distill student must expose encoder and actor modules")
        student_parameters = [
            *self.student.encoder.parameters(),
            *self.student.actor.parameters(),
        ]
        self.optimizer = torch.optim.Adam(student_parameters, lr=self.learning_rate)

        self.current_iteration = 0
        self.total_steps = 0
        self.gradient_steps = 0
        self.dagger_phase_index = 0
        self.dagger_phase_iteration = 0
        self.student_drive_probability = 0.0
        self.safety_threshold = float("inf")
        self.previous_hard_termination_rate: float | None = None
        self.training_status = "initialized"
        self.final_selection: dict[str, Any] | None = None
        self.promotion_history: list[dict[str, Any]] = []
        self.recent_iteration_metrics: list[dict[str, float]] = []
        self.config_sha256 = "unknown"
        self.code_commit = "unknown"
        self.parent_model_id = "10288"
        self.loss_buffer = deque(maxlen=100)

        self.replay_capacity = int(replay_capacity)
        self.replay_batch_size = int(replay_batch_size)
        self.replay_loss_ratio = float(replay_loss_ratio)
        self.replay_add_per_step = int(replay_add_per_step)
        if self.replay_capacity <= 0 or self.replay_batch_size <= 0:
            raise ValueError("Replay capacity and batch size must be positive")
        if not 0.0 <= self.replay_loss_ratio <= 1.0:
            raise ValueError("replay_loss_ratio must stay in [0,1]")
        self.replay_obs: torch.Tensor | None = None
        self.replay_actions: torch.Tensor | None = None
        self.replay_size = 0
        self.replay_seen = 0

        self.phase_entry_snapshot: dict[str, Any] | None = None
        self.phase_best_snapshot: dict[str, Any] | None = None
        self.phase_exit_snapshot: dict[str, Any] | None = None
        self.phase_best_metrics: dict[str, float] | None = None
        self.phase_best_key: tuple[float, ...] | None = None

        if teacher_ckpt:
            self._load_teacher(teacher_ckpt)
        self._freeze_teacher()

    @property
    def current_phase_label(self) -> str:
        return phase_label(self.dagger_phase_index)

    def _load_teacher(self, teacher_ckpt: str) -> None:
        checkpoint_sha = sha256_file(teacher_ckpt)
        checkpoint = torch.load(
            teacher_ckpt, weights_only=False, map_location=self.device
        )
        self.load_teacher_state_dict(
            checkpoint,
            source=teacher_ckpt,
            source_sha256=checkpoint_sha,
        )

    def load_teacher_state_dict(
        self,
        checkpoint: dict,
        source: str = "<preload>",
        source_sha256: str | None = None,
    ) -> None:
        if isinstance(checkpoint, dict) and checkpoint.get("format"):
            raise ValueError(
                "Flat teacher must be a raw state dict, got "
                f"format={checkpoint.get('format')!r}"
            )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError("Flat Standard teacher must be a state dict")

        expected = self.teacher.state_dict()
        missing = sorted(set(expected) - set(checkpoint))
        unexpected = sorted(set(checkpoint) - set(expected))
        mismatched = sorted(
            key
            for key in set(expected) & set(checkpoint)
            if not hasattr(checkpoint[key], "shape")
            or tuple(checkpoint[key].shape) != tuple(expected[key].shape)
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "Flat Standard teacher contract mismatch: "
                f"missing={missing}, unexpected={unexpected}, "
                f"shape_mismatch={mismatched}"
            )

        if source_sha256 is None and os.path.isfile(source):
            source_sha256 = sha256_file(source)
        self.teacher.load_state_dict(checkpoint, strict=True)
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
                "[BehaviorDistill] no compatible flat301 teacher was loaded"
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
        teacher_frozen = not self.teacher.training and all(
            not parameter.requires_grad for parameter in self.teacher.parameters()
        )
        if (
            not teacher_frozen
            or optimizer_ids != expected_ids
            or bool(optimizer_ids & teacher_ids)
        ):
            raise RuntimeError(
                "[BehaviorDistill] freeze/optimizer contract failed: "
                f"teacher_frozen={teacher_frozen}, "
                f"student_encoder_actor_only={optimizer_ids == expected_ids}, "
                f"teacher_in_optimizer={bool(optimizer_ids & teacher_ids)}"
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

    def set_run_metadata(
        self, config_sha256: str, code_commit: str, parent_model_id: str = "10288"
    ) -> None:
        self.config_sha256 = str(config_sha256)
        self.code_commit = str(code_commit)
        self.parent_model_id = str(parent_model_id)

    def prepare_update(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.assert_teacher_ready()
        obs = torch.as_tensor(obs, device=self.device)
        if obs.ndim != 2 or obs.shape[-1] != self.num_obs:
            raise ValueError(
                f"Behavior-distill obs must be [B,{self.num_obs}], got {tuple(obs.shape)}"
            )
        with torch.no_grad():
            teacher_action = self.teacher.act_inference(obs)
            student_action = self.student.act_inference(obs)
        if not bool(torch.isfinite(teacher_action).all().item()):
            raise FloatingPointError("Frozen teacher produced NaN/Inf action")

        student_finite = torch.isfinite(student_action).all(dim=-1)
        safe_student_action = torch.where(
            student_finite.unsqueeze(-1), student_action, teacher_action.detach()
        )
        difference = safe_student_action - teacher_action
        return {
            "obs": obs,
            "teacher_action": teacher_action,
            "student_action": student_action,
            "safe_student_action": safe_student_action,
            "student_finite": student_finite,
            "difference": difference,
            "per_sample_mse": difference.pow(2).mean(dim=-1),
            "per_sample_l2": difference.pow(2).sum(dim=-1).sqrt(),
            "per_sample_cos": F.cosine_similarity(
                safe_student_action, teacher_action, dim=-1
            ),
        }

    def select_driver_actions(
        self,
        batch: Dict[str, torch.Tensor],
        student_drive_probability: float,
        safety_threshold: float,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        probability = float(student_drive_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("student_drive_probability must stay in [0,1]")
        requested_student = (
            torch.rand(batch["teacher_action"].shape[0], device=self.device)
            < probability
        )
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
        return actions, {
            "requested_student": requested_student,
            "effective_student": effective_student,
            "safety_takeover": safety_takeover,
        }

    def _add_replay(
        self,
        obs: torch.Tensor,
        teacher_actions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return
        if valid_indices.numel() > self.replay_add_per_step:
            order = torch.randperm(valid_indices.numel(), device=valid_indices.device)
            valid_indices = valid_indices[order[: self.replay_add_per_step]]
        candidate_obs = obs[valid_indices].detach().to("cpu", dtype=torch.float16)
        candidate_actions = (
            teacher_actions[valid_indices].detach().to("cpu", dtype=torch.float16)
        )
        if self.replay_obs is None:
            self.replay_obs = torch.empty(
                (self.replay_capacity, self.num_obs), dtype=torch.float16
            )
            self.replay_actions = torch.empty(
                (self.replay_capacity, self.num_actions), dtype=torch.float16
            )

        cursor = 0
        available = self.replay_capacity - self.replay_size
        if available > 0:
            count = min(available, candidate_obs.shape[0])
            end = self.replay_size + count
            self.replay_obs[self.replay_size : end].copy_(candidate_obs[:count])
            self.replay_actions[self.replay_size : end].copy_(
                candidate_actions[:count]
            )
            self.replay_size = end
            self.replay_seen += count
            cursor = count

        remaining = candidate_obs.shape[0] - cursor
        if remaining <= 0:
            return
        considered = torch.arange(
            1, remaining + 1, dtype=torch.float64
        ) + float(self.replay_seen)
        slots = torch.floor(torch.rand(remaining, dtype=torch.float64) * considered).long()
        accepted = slots < self.replay_capacity
        if bool(accepted.any().item()):
            accepted_slots = slots[accepted]
            self.replay_obs[accepted_slots] = candidate_obs[cursor:][accepted]
            self.replay_actions[accepted_slots] = candidate_actions[cursor:][accepted]
        self.replay_seen += remaining

    def _replay_loss(self) -> torch.Tensor | None:
        if (
            self.replay_obs is None
            or self.replay_actions is None
            or self.replay_size == 0
        ):
            return None
        count = min(self.replay_batch_size, self.replay_size)
        indices = torch.randint(0, self.replay_size, (count,))
        replay_obs = self.replay_obs[indices].to(self.device, dtype=torch.float32)
        replay_targets = self.replay_actions[indices].to(
            self.device, dtype=torch.float32
        )
        replay_student = self.student.act_inference(replay_obs)
        if not bool(torch.isfinite(replay_student).all().item()):
            raise FloatingPointError("Student produced NaN/Inf on replay batch")
        return F.mse_loss(replay_student, replay_targets)

    def finish_update(
        self,
        batch: Dict[str, torch.Tensor],
        sample_weights: torch.Tensor,
    ) -> Dict[str, float]:
        weights = torch.as_tensor(
            sample_weights,
            device=self.device,
            dtype=batch["per_sample_mse"].dtype,
        ).reshape(-1)
        if weights.shape[0] != batch["per_sample_mse"].shape[0]:
            raise ValueError("sample_weights size mismatch")
        weights = torch.clamp(torch.nan_to_num(weights, nan=0.0), 0.0, 1.0)
        train_mask = (weights > 0.0) & batch["student_finite"]
        grad_norm = torch.zeros((), device=self.device)
        current_loss = batch["per_sample_mse"].sum() * 0.0
        replay_loss = current_loss
        replay_ratio = 0.0

        if bool(train_mask.any().item()):
            train_obs = batch["obs"][train_mask]
            train_targets = batch["teacher_action"][train_mask].detach()
            train_weights = weights[train_mask]
            train_student = self.student.act_inference(train_obs)
            if not bool(torch.isfinite(train_student).all().item()):
                raise FloatingPointError("Student produced NaN/Inf during update")
            per_sample = (train_student - train_targets).pow(2).mean(dim=-1)
            current_loss = (
                train_weights * per_sample
            ).sum() / train_weights.sum().clamp_min(1.0)
            replay_value = self._replay_loss()
            if replay_value is not None:
                replay_loss = replay_value
                replay_ratio = self.replay_loss_ratio
            loss = self.action_loss_weight * (
                (1.0 - replay_ratio) * current_loss + replay_ratio * replay_loss
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Behavior-distill loss is NaN/Inf")
            self.optimizer.zero_grad()
            loss.backward()
            parameters = [
                parameter
                for group in self.optimizer.param_groups
                for parameter in group["params"]
            ]
            grad_norm = nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            if not bool(torch.isfinite(grad_norm).item()):
                raise FloatingPointError("Behavior-distill gradient norm is NaN/Inf")
            self.optimizer.step()
            self.gradient_steps += 1
        else:
            loss = current_loss

        self._add_replay(
            batch["obs"],
            batch["teacher_action"],
            valid_mask=(weights >= 1.0) & batch["student_finite"],
        )
        self.total_steps += int(batch["teacher_action"].shape[0])
        loss_value = float(loss.detach().item())
        self.loss_buffer.append(loss_value)

        denominator = weights.sum().clamp_min(1.0)
        with torch.no_grad():
            teacher_variance = batch["teacher_action"].var(
                dim=0, unbiased=False
            ).clamp_min(1.0e-6)
            per_dimension_mse = (
                weights.unsqueeze(-1) * batch["difference"].pow(2)
            ).sum(dim=0) / denominator
            normalized_action_mse = (per_dimension_mse / teacher_variance).mean()

        return {
            "loss": loss_value,
            "action_mse": float(current_loss.detach().item()),
            "replay_mse": float(replay_loss.detach().item()),
            "replay_loss_ratio": replay_ratio,
            "normalized_action_mse": float(normalized_action_mse.item()),
            "action_l2": float(
                ((weights * batch["per_sample_l2"]).sum() / denominator).item()
            ),
            "action_l2_p95": float(
                torch.quantile(batch["per_sample_l2"].detach(), 0.95).item()
            ),
            "action_cos": float(
                ((weights * batch["per_sample_cos"]).sum() / denominator).item()
            ),
            "teacher_abs": float(batch["teacher_action"].abs().mean().item()),
            "student_abs": float(batch["safe_student_action"].abs().mean().item()),
            "grad_norm": float(grad_norm.detach().item()),
            "weighted_sample_rate": float((weights > 0.0).float().mean().item()),
            "mean_sample_weight": float(weights.mean().item()),
            "nonfinite_rate": float(
                (~batch["student_finite"]).float().mean().item()
            ),
            "replay_size": float(self.replay_size),
        }

    def assert_student_parameters_finite(self) -> None:
        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        if not all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in parameters
        ):
            raise FloatingPointError("Student parameters became NaN/Inf")

    def act_teacher(self, obs: torch.Tensor) -> torch.Tensor:
        self.assert_teacher_ready()
        with torch.no_grad():
            return self.teacher.act_inference(
                torch.as_tensor(obs, device=self.device)
            )

    def act_student(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.student.act_inference(
                torch.as_tensor(obs, device=self.device)
            )

    def update(self, obs: torch.Tensor) -> Dict[str, float]:
        batch = self.prepare_update(obs)
        weights = torch.ones(batch["teacher_action"].shape[0], device=self.device)
        return self.finish_update(batch, weights)

    def start_phase(self, phase_index: int) -> None:
        self.dagger_phase_index = int(phase_index)
        self.dagger_phase_iteration = 0
        self.phase_entry_snapshot = self._training_snapshot()
        self.phase_best_snapshot = None
        self.phase_best_metrics = None
        self.phase_best_key = None

    def capture_phase_exit(self) -> None:
        self.phase_exit_snapshot = self._training_snapshot()

    @staticmethod
    def _quality_key(metrics: dict[str, float]) -> tuple[float, ...] | None:
        required = (
            "hard_termination_rate",
            "safety_takeover_rate",
            "normalized_action_mse",
            "action_cos",
            "nonfinite_rate",
            "teacher_ood_rate",
        )
        if any(
            key not in metrics or not math.isfinite(float(metrics[key]))
            for key in required
        ):
            return None
        if metrics["nonfinite_rate"] > 0.0 or metrics["teacher_ood_rate"] > 0.0:
            return None
        return (
            float(metrics["hard_termination_rate"]),
            float(metrics["safety_takeover_rate"]),
            float(metrics["normalized_action_mse"]),
            -float(metrics["action_cos"]),
        )

    def consider_phase_best(self, metrics: dict[str, float]) -> None:
        key = self._quality_key(metrics)
        if key is None:
            return
        if self.phase_best_key is None or key < self.phase_best_key:
            self.phase_best_key = key
            self.phase_best_metrics = dict(metrics)
            self.phase_best_snapshot = self._training_snapshot()

    def restore_phase_best_or_entry(self) -> str:
        if self.phase_best_snapshot is not None:
            self._restore_training_snapshot(self.phase_best_snapshot)
            return "phase_best"
        if self.phase_entry_snapshot is not None:
            self._restore_training_snapshot(self.phase_entry_snapshot)
            return "phase_entry"
        return "current"

    def _training_snapshot(self) -> dict[str, Any]:
        return {
            "student_model_state_dict": _copy_to_cpu(self.student.state_dict()),
            "optimizer_state_dict": _copy_to_cpu(self.optimizer.state_dict()),
        }

    def _restore_training_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.student.load_state_dict(snapshot["student_model_state_dict"], strict=True)
        self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])

    def model_spec(self) -> dict[str, Any]:
        return {
            "task": "standard",
            "proprio_dim": int(getattr(self.student, "num_proprio", 45)),
            "scan_dim": int(getattr(self.student, "num_scan", 256)),
            "latent_dim": int(getattr(self.student, "latent_dim", 32)),
            "actor_input_dim": int(getattr(self.student, "num_proprio", 45))
            + int(getattr(self.student, "latent_dim", 32)),
            "teacher_observation_dim": self.num_obs,
            "teacher_critic_observation_dim": self.num_critic_obs,
            "action_dim": self.num_actions,
            "goal_dim": int(getattr(self.student, "num_goal_obs", 0)),
        }

    @staticmethod
    def _rng_state() -> dict[str, Any]:
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

    def checkpoint_payload(
        self, *, platform_model_id: str | int | None = None, **extra: Any
    ) -> dict[str, Any]:
        replay = {
            "capacity": self.replay_capacity,
            "batch_size": self.replay_batch_size,
            "loss_ratio": self.replay_loss_ratio,
            "add_per_step": self.replay_add_per_step,
            "size": self.replay_size,
            "seen": self.replay_seen,
            "obs_fp16": (
                None
                if self.replay_obs is None
                else self.replay_obs[: self.replay_size].clone()
            ),
            "teacher_actions_fp16": (
                None
                if self.replay_actions is None
                else self.replay_actions[: self.replay_size].clone()
            ),
        }
        payload = {
            "format": KAIWU_TRAIN_FORMAT,
            "schema_version": KAIWU_TRAIN_SCHEMA_VERSION,
            "artifact_role": "training_bundle",
            "stage_type": "standard_bridge",
            "model_spec": self.model_spec(),
            "modules": {
                "low_level": {
                    "class_name": self.student.__class__.__name__,
                    "policy_state_dict": self.student.state_dict(),
                    "encoder_state_dict": self.student.encoder.state_dict(),
                    "actor_state_dict": self.student.actor.state_dict(),
                    "critic_trained": False,
                },
                "privileged_teacher": {
                    "class_name": self.teacher.__class__.__name__,
                    "state_dict": self.teacher.state_dict(),
                    "frozen": True,
                },
            },
            "optimizers": {"low_level_distill": self.optimizer.state_dict()},
            "training_state": {
                "current_iteration": self.current_iteration,
                "total_steps": self.total_steps,
                "gradient_steps": self.gradient_steps,
                "dagger_phase_index": self.dagger_phase_index,
                "dagger_phase_label": self.current_phase_label,
                "dagger_phase_iteration": self.dagger_phase_iteration,
                "student_drive_probability": self.student_drive_probability,
                "safety_threshold": self.safety_threshold,
                "previous_hard_termination_rate": self.previous_hard_termination_rate,
                "promotion_history": self.promotion_history,
                "recent_iteration_metrics": self.recent_iteration_metrics,
                "rng_state": self._rng_state(),
                "training_status": self.training_status,
                "final_selection": self.final_selection,
            },
            "replay": replay,
            "phase_snapshots": {
                "entry": self.phase_entry_snapshot,
                "best": self.phase_best_snapshot,
                "exit": self.phase_exit_snapshot,
                "best_metrics": self.phase_best_metrics,
                "best_key": self.phase_best_key,
            },
            "lineage": {
                "parent_model_id": self.parent_model_id,
                "teacher_source": self.teacher_source,
                "teacher_sha256": self.teacher_sha256,
                "config_sha256": self.config_sha256,
                "code_commit": self.code_commit,
                "platform_model_id": (
                    None if platform_model_id is None else str(platform_model_id)
                ),
            },
            "capabilities": {
                "task": "standard",
                "goal_dim": 0,
                "critic_trained": False,
                "deployable": False,
            },
        }
        payload.update(extra)
        return payload

    def save(
        self,
        path: str,
        *,
        platform_model_id: str | int | None = None,
        **extra: Any,
    ) -> str:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            self.checkpoint_payload(
                platform_model_id=platform_model_id,
                **extra,
            ),
            path,
        )
        return sha256_file(path)

    def load_checkpoint_dict(
        self,
        checkpoint: dict[str, Any],
        source: str,
        *,
        load_optimizer: bool = True,
        load_teacher: bool = True,
        restore_rng: bool = True,
    ) -> None:
        validate_low_level_spec(
            checkpoint,
            expected={
                "proprio_dim": int(getattr(self.student, "num_proprio", 45)),
                "scan_dim": int(getattr(self.student, "num_scan", 256)),
                "latent_dim": int(getattr(self.student, "latent_dim", 32)),
                "action_dim": self.num_actions,
                "goal_dim": 0,
            },
        )
        self.student.load_state_dict(low_level_policy_state(checkpoint), strict=True)
        modules = checkpoint["modules"]
        if load_teacher:
            teacher_section = modules.get("privileged_teacher")
            if not isinstance(teacher_section, dict):
                raise KeyError("modules.privileged_teacher missing")
            teacher_state = teacher_section.get("state_dict")
            if not isinstance(teacher_state, dict):
                raise KeyError("privileged teacher state_dict missing")
            self.load_teacher_state_dict(
                teacher_state,
                source=source,
                source_sha256=checkpoint.get("lineage", {}).get("teacher_sha256"),
            )
        if load_optimizer:
            optimizer_state = checkpoint.get("optimizers", {}).get(
                "low_level_distill"
            )
            if not isinstance(optimizer_state, dict):
                raise KeyError("optimizers.low_level_distill missing")
            self.optimizer.load_state_dict(optimizer_state)

        state = checkpoint.get("training_state", {})
        self.current_iteration = int(state.get("current_iteration", 0))
        self.total_steps = int(state.get("total_steps", 0))
        self.gradient_steps = int(state.get("gradient_steps", 0))
        self.dagger_phase_index = int(state.get("dagger_phase_index", 0))
        saved_phase_label = state.get("dagger_phase_label")
        if saved_phase_label != phase_label(self.dagger_phase_index):
            raise ValueError(
                "Checkpoint DAgger phase mismatch: "
                f"index={self.dagger_phase_index}, label={saved_phase_label!r}"
            )
        self.dagger_phase_iteration = int(state.get("dagger_phase_iteration", 0))
        self.student_drive_probability = float(
            state.get("student_drive_probability", 0.0)
        )
        self.safety_threshold = float(state.get("safety_threshold", float("inf")))
        previous_rate = state.get("previous_hard_termination_rate")
        self.previous_hard_termination_rate = (
            None if previous_rate is None else float(previous_rate)
        )
        self.promotion_history = list(state.get("promotion_history", []))
        self.recent_iteration_metrics = list(
            state.get("recent_iteration_metrics", [])
        )
        self.training_status = str(state.get("training_status", "resumed"))
        self.final_selection = state.get("final_selection")

        lineage = checkpoint.get("lineage", {})
        self.parent_model_id = str(lineage.get("parent_model_id", "10288"))
        self.config_sha256 = str(lineage.get("config_sha256", "unknown"))
        self.code_commit = str(lineage.get("code_commit", "unknown"))

        replay = checkpoint.get("replay", {})
        replay_obs = replay.get("obs_fp16")
        replay_actions = replay.get("teacher_actions_fp16")
        self.replay_capacity = int(replay.get("capacity", self.replay_capacity))
        self.replay_batch_size = int(
            replay.get("batch_size", self.replay_batch_size)
        )
        self.replay_loss_ratio = float(
            replay.get("loss_ratio", self.replay_loss_ratio)
        )
        self.replay_add_per_step = int(
            replay.get("add_per_step", self.replay_add_per_step)
        )
        if (
            self.replay_capacity <= 0
            or self.replay_batch_size <= 0
            or self.replay_add_per_step <= 0
            or not 0.0 <= self.replay_loss_ratio <= 1.0
        ):
            raise ValueError(
                "Invalid replay settings restored from checkpoint: "
                f"capacity={self.replay_capacity}, "
                f"batch_size={self.replay_batch_size}, "
                f"loss_ratio={self.replay_loss_ratio}, "
                f"add_per_step={self.replay_add_per_step}"
            )
        self.replay_size = int(replay.get("size", 0))
        self.replay_seen = int(replay.get("seen", self.replay_size))
        if self.replay_seen < self.replay_size:
            raise ValueError(
                f"Replay seen={self.replay_seen} is smaller than size={self.replay_size}"
            )
        if not 0 <= self.replay_size <= self.replay_capacity:
            raise ValueError(
                "Replay size exceeds configured capacity: "
                f"{self.replay_size}>{self.replay_capacity}"
            )
        if self.replay_size > 0:
            if not torch.is_tensor(replay_obs) or not torch.is_tensor(replay_actions):
                raise KeyError("Replay tensors missing from non-empty checkpoint")
            if tuple(replay_obs.shape) != (self.replay_size, self.num_obs):
                raise ValueError(
                    f"Replay obs shape mismatch: {tuple(replay_obs.shape)}"
                )
            if tuple(replay_actions.shape) != (
                self.replay_size,
                self.num_actions,
            ):
                raise ValueError(
                    "Replay action shape mismatch: "
                    f"{tuple(replay_actions.shape)}"
                )
            self.replay_obs = torch.empty(
                (self.replay_capacity, self.num_obs), dtype=torch.float16
            )
            self.replay_actions = torch.empty(
                (self.replay_capacity, self.num_actions), dtype=torch.float16
            )
            self.replay_obs[: self.replay_size].copy_(replay_obs)
            self.replay_actions[: self.replay_size].copy_(replay_actions)
        else:
            self.replay_obs = None
            self.replay_actions = None

        snapshots = checkpoint.get("phase_snapshots", {})
        self.phase_entry_snapshot = snapshots.get("entry")
        self.phase_best_snapshot = snapshots.get("best")
        self.phase_exit_snapshot = snapshots.get("exit")
        self.phase_best_metrics = snapshots.get("best_metrics")
        best_key = snapshots.get("best_key")
        self.phase_best_key = None if best_key is None else tuple(best_key)
        if restore_rng:
            self._restore_rng_state(state.get("rng_state"))
        if self.logger is not None:
            self.logger.info(
                f"[BehaviorDistill] resumed {KAIWU_TRAIN_FORMAT} from {source}, "
                f"iteration={self.current_iteration}, phase={self.current_phase_label}"
            )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        if not is_kaiwu_train_bundle(checkpoint):
            raise ValueError(f"Not a {KAIWU_TRAIN_FORMAT} checkpoint: {path}")
        self.load_checkpoint_dict(checkpoint, path)

    def load_student_weights(self, checkpoint: dict, source: str = "<legacy>") -> None:
        if is_kaiwu_train_bundle(checkpoint):
            checkpoint = low_level_policy_state(checkpoint)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Student weights from {source} are not a state dict")
        self.student.load_state_dict(checkpoint, strict=True)
