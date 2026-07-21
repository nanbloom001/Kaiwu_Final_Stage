#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Behavior distillation from a mature flat standard teacher to ActorCriticEncoder.

This is the bridge stage before lbc_loco:
    flat reference ActorCritic(obs301 -> action12)
        -> student ActorCriticEncoder(proprio45 + scan256 -> latent32 -> action12)
        -> saved as model.ckpt-locomotion-{id}.pkl

The produced locomotion ckpt has encoder.* / actor.* keys, so the existing
LBC depth-camera distillation path can split it as teacher_encoder + teacher_actor.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.model.actor_critic import ActorCritic


class AlgorithmBehaviorDistill:
    """Supervised action distillation for standard locomotion."""

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
        logger=None,
    ) -> None:
        self.student = student.to(device)
        self.actor_critic = self.student
        self.device = device
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.action_loss_weight = action_loss_weight
        self.logger = logger

        self.teacher = ActorCritic(
            num_obs=num_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=teacher_actor_hidden_dims,
            critic_hidden_dims=teacher_critic_hidden_dims,
            activation=teacher_activation,
        ).to(device)
        self.teacher_loaded = False
        self.teacher_source = None
        if teacher_ckpt:
            self._load_teacher(teacher_ckpt)
        self._freeze_teacher()

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=learning_rate)
        self.current_iteration = 0
        self.total_steps = 0
        self.loss_buffer = deque(maxlen=100)

    def _load_teacher(self, teacher_ckpt: str) -> None:
        ckpt = torch.load(teacher_ckpt, weights_only=False, map_location=self.device)
        self.load_teacher_state_dict(ckpt, source=teacher_ckpt)

    def load_teacher_state_dict(self, ckpt: dict, source: str = "<preload>") -> None:
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

        self.teacher.load_state_dict(ckpt, strict=True)
        self.teacher_loaded = True
        self.teacher_source = source
        if self.logger is not None:
            self.logger.info(f"[BehaviorDistill] loaded flat standard teacher from {source}")

    def _freeze_teacher(self) -> None:
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def act_teacher(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.teacher_loaded:
            raise RuntimeError(
                "[BehaviorDistill] teacher is not loaded. Select a matching flat "
                "pretrained teacher checkpoint before starting this stage."
            )
        with torch.no_grad():
            return self.teacher.act_inference(obs)

    def act_student(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.student.act_inference(obs)

    def compute_metrics(self, obs: torch.Tensor) -> Dict[str, float]:
        """Evaluate student-vs-teacher action agreement without updating weights."""
        if not self.teacher_loaded:
            raise RuntimeError(
                "[BehaviorDistill] teacher is not loaded. Select a matching flat "
                "pretrained teacher checkpoint before starting this stage."
            )
        obs = obs.to(self.device)
        with torch.no_grad():
            teacher_action = self.teacher.act_inference(obs)
            student_action = self.student.act_inference(obs)
            action_mse = F.mse_loss(student_action, teacher_action)
            l2_distance = (student_action - teacher_action).pow(2).sum(dim=-1).sqrt().mean()
            cos_sim = F.cosine_similarity(student_action, teacher_action, dim=-1).mean()
            teacher_abs = teacher_action.abs().mean()
            student_abs = student_action.abs().mean()

        return {
            "loss": float(action_mse.item()),
            "action_mse": float(action_mse.item()),
            "action_l2": float(l2_distance.item()),
            "action_cos": float(cos_sim.item()),
            "teacher_abs": float(teacher_abs.item()),
            "student_abs": float(student_abs.item()),
            "grad_norm": 0.0,
        }

    def update(self, obs: torch.Tensor) -> Dict[str, float]:
        """Run one supervised action-MSE update on the current observation batch."""
        if not self.teacher_loaded:
            raise RuntimeError(
                "[BehaviorDistill] teacher is not loaded. Select a matching flat "
                "pretrained teacher checkpoint before starting this stage."
            )
        obs = obs.to(self.device)
        with torch.no_grad():
            teacher_action = self.teacher.act_inference(obs)

        student_action = self.student.act_inference(obs)
        action_mse = F.mse_loss(student_action, teacher_action)
        loss = self.action_loss_weight * action_mse

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.total_steps += 1
        loss_val = float(loss.item())
        self.loss_buffer.append(loss_val)

        with torch.no_grad():
            l2_distance = (student_action - teacher_action).pow(2).sum(dim=-1).sqrt().mean()
            cos_sim = F.cosine_similarity(student_action, teacher_action, dim=-1).mean()
            teacher_abs = teacher_action.abs().mean()
            student_abs = student_action.abs().mean()

        return {
            "loss": loss_val,
            "action_mse": float(action_mse.item()),
            "action_l2": float(l2_distance.item()),
            "action_cos": float(cos_sim.item()),
            "teacher_abs": float(teacher_abs.item()),
            "student_abs": float(student_abs.item()),
            "grad_norm": float(grad_norm.item()),
        }

    def save(self, path: str, **kwargs: Any) -> None:
        torch.save(self.student.state_dict(), path)

    def load_student(self, path: str) -> None:
        state = torch.load(path, weights_only=False, map_location=self.device)
        self.student.load_state_dict(state)
