#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from __future__ import annotations

import os
import statistics
from collections import deque
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlgorithmLBC:
    """LBC (Learning by Cheating) training algorithm.
    LBC (Learning by Cheating) 训练算法。

    Teacher-student distillation for training the vision encoder against the
    frozen locomotion teacher (height_scan → latent).
    教师-学生蒸馏算法：学生 VisionEncoder 从 depth 图像回归教师 encoder(height_scan) 的 latent。

    Args:
        vision_encoder: 学生视觉编码器 (VisionEncoder)
        teacher_encoder: 教师特权编码器 (DmEncoder)
        teacher_actor: 教师 Actor 网络 (用于生成动作驱动环境)
        device: 设备 ("cuda:0" 或 "cpu")
        learning_rate: 学习率 (默认 1e-3)
        latent_dim: 潜在特征维度 (默认 32)
        max_grad_norm: 梯度裁剪范数 (默认 1.0)
        proprio_dim: 本体感知维度 (默认 45，对应 Go2 proprio)
        scan_dim: height_scan 维度 (默认 256，16×16 网格)
        depth_shape: 深度图形状 (H, W, C) (默认 (180, 320, 1))
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        teacher_encoder: nn.Module,
        teacher_actor: nn.Module,
        device: str = "cuda:0",
        learning_rate: float = 1e-3,
        latent_dim: int = 32,
        max_grad_norm: float = 1.0,
        proprio_dim: int = 45,
        scan_dim: int = 256,
        goal_dim: int = 0,
        depth_shape: Tuple[int, int, int] = (180, 320, 1),
    ):
        self.device = device
        self.mode = "loco"
        self.latent_dim = latent_dim
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.proprio_dim = proprio_dim
        self.scan_dim = scan_dim
        self.goal_dim = goal_dim
        self.depth_shape = depth_shape  # (H, W, C)

        # Student: vision encoder (trainable)
        self.vision_encoder = vision_encoder.to(device)
        self.vision_encoder.train()

        # Teacher: loco encoder + loco actor (frozen)
        self.teacher_encoder = teacher_encoder.to(device)
        self.teacher_actor = teacher_actor.to(device)

        self._freeze_teacher()

        # Optimizer: 全部 vision_encoder 参数可训
        self.optimizer = torch.optim.Adam(self.vision_encoder.parameters(), lr=learning_rate)

        # Training state
        self.current_iteration = 0
        self.total_steps = 0

        # Logging buffers
        self.mse_buffer = deque(maxlen=100)
        self.reward_buffer = deque(maxlen=100)
        self.episode_length_buffer = deque(maxlen=100)

    def _freeze_teacher(self):
        """Freeze teacher networks (encoder + actor)."""
        for module in (self.teacher_encoder, self.teacher_actor):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def _split_obs(self, obs) -> Dict[str, torch.Tensor]:
        """将 flat tensor obs 切分为 dict；dict 则直接返回。

        lbc_loco flat tensor layout:
            [ proprio | height_scan | optional_goal | depth(H*W*C) ]
        """
        if isinstance(obs, dict):
            return obs

        H, W, C = self.depth_shape
        proprio = obs[:, : self.proprio_dim]
        height_scan = obs[:, self.proprio_dim : self.proprio_dim + self.scan_dim]
        goal_start = self.proprio_dim + self.scan_dim
        depth_start = goal_start + self.goal_dim
        goal = obs[:, goal_start:depth_start] if self.goal_dim > 0 else None
        depth_flat = obs[:, depth_start:]
        depth_image = depth_flat.view(-1, H, W, C)
        return {
            "proprio": proprio,
            "height_scan": height_scan,
            "goal": goal,
            "depth_image": depth_image,
        }

    def _teacher_actor_input(self, obs: Dict[str, torch.Tensor], latent: torch.Tensor) -> torch.Tensor:
        parts = [obs["proprio"], latent]
        if self.goal_dim > 0:
            goal = obs.get("goal")
            if goal is None or goal.shape[-1] != self.goal_dim:
                got = None if goal is None else goal.shape[-1]
                raise ValueError(f"LBC teacher expects goal_dim={self.goal_dim}, got {got}")
            parts.append(goal)
        return torch.cat(parts, dim=-1)

    def act_teacher(self, obs) -> torch.Tensor:
        """教师网络生成动作 (用于驱动环境)。

        Args:
            obs: Observation dict 或 flat tensor，包含:
                 - "proprio": proprioception [num_envs, proprio_dim]
                 - "height_scan": height scan (privileged) [num_envs, height_scan_dim]

        Returns:
            actions: 动作 [num_envs, num_actions]。
        """
        obs = self._split_obs(obs)
        with torch.no_grad():
            # Teacher encoder: height_scan → latent
            teacher_latent = self.teacher_encoder(obs["height_scan"])
            # Concatenate proprio + latent
            policy_input = self._teacher_actor_input(obs, teacher_latent)
            # Actor generates actions
            actions = self.teacher_actor(policy_input)
        return actions

    def act_student(self, obs) -> torch.Tensor:
        """Student closed-loop inference: 用于可视化学生当前表现。

        depth → student_latent → teacher_actor(proprio, student_latent) → actions
        """
        obs = self._split_obs(obs)
        with torch.no_grad():
            student_latent = self.vision_encoder(
                depth_image=obs["depth_image"],
                proprio=obs["proprio"],
                masks=None,
            )
            policy_input = self._teacher_actor_input(obs, student_latent)
            actions = self.teacher_actor(policy_input)
        return actions

    def compute_latent_loss(
        self,
        obs,
        masks: torch.Tensor = None,
    ) -> Dict[str, Any]:
        """Compute latent feature distillation loss.
        计算潜在特征蒸馏损失。

        Args:
            obs: Observation dict 或 flat tensor，包含:
                 - "proprio":     [num_envs, proprio_dim]
                 - "depth_image": [num_envs, H, W, C]
                 - "height_scan": [num_envs, scan_dim]
            masks: Episode reset mask [num_envs,]，True=continue，False=reset。

        Returns:
            loss_dict:
                - "loss": 总损失（== mse_loss）
                - "mse_loss": MSE 损失
                - "distance": L2 距离
                - "teacher_latent" / "student_latent": 用于调试
        """
        obs = self._split_obs(obs)

        # Teacher forward (frozen)
        with torch.no_grad():
            teacher_latent = self.teacher_encoder(obs["height_scan"])

        # Student forward (trainable) — LSTM 吃 cnn_feat+proprio
        student_latent = self.vision_encoder(
            depth_image=obs["depth_image"],
            proprio=obs["proprio"],
            masks=masks,
        )

        # MSE loss
        # 不用默认的 reduction='mean'（会同时按 batch 和 dim 取均值，
        # 导致 loss 数值被 latent_dim 稀释 32 倍，难以观察）。
        # 改为：先在 dim 维 sum，再在 batch 维 mean，等价于"逐样本平方距离的均值"。
        per_sample_sq = (student_latent - teacher_latent).pow(2).sum(dim=-1)  # [B]
        mse_loss = per_sample_sq.mean()

        # L2 distance (for monitoring)
        with torch.no_grad():
            distance = per_sample_sq.sqrt().mean()

        return {
            "loss": mse_loss,
            "mse_loss": mse_loss,
            "distance": distance,
            "teacher_latent": teacher_latent.detach(),
            "student_latent": student_latent.detach(),
        }

    def update(
        self,
        obs,
        masks: torch.Tensor = None,
    ) -> Dict[str, float]:
        """Perform one gradient update (loco distillation)."""
        loss_dict = self.compute_latent_loss(obs, masks)
        loss = loss_dict["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.vision_encoder.parameters(),
            self.max_grad_norm,
        )
        self.optimizer.step()

        self.total_steps += 1
        mse_val = loss_dict["mse_loss"].item()
        distance = loss_dict["distance"].item()
        self.mse_buffer.append(mse_val)

        with torch.no_grad():
            t_lat = loss_dict["teacher_latent"]
            s_lat = loss_dict["student_latent"]
            cos_sim = F.cosine_similarity(s_lat, t_lat, dim=-1).mean().item()
            student_std = s_lat.std(dim=0).mean().item()
            teacher_std = t_lat.std(dim=0).mean().item()
            student_norm = s_lat.norm(dim=-1).mean().item()

        return {
            "mse_loss": mse_val,
            "distance": distance,
            "grad_norm": grad_norm.item(),
            "cos_sim": cos_sim,
            "student_std": student_std,
            "teacher_std": teacher_std,
            "student_norm": student_norm,
        }

    def reset_student_hidden_states(self, dones: torch.Tensor):
        """Reset LSTM hidden states of student encoder (for terminated envs)。

        Args:
            dones: Episode termination flag [num_envs,]，True=terminated。
        """
        if hasattr(self.vision_encoder, "reset_hidden_state_for_envs"):
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                self.vision_encoder.reset_hidden_state_for_envs(done_ids)

    def update_episode_stats(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        cur_reward_sum: torch.Tensor,
        cur_episode_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """更新 episode 统计信息。"""
        cur_reward_sum += rewards
        cur_episode_length += 1

        done_ids = (dones > 0).nonzero(as_tuple=False)
        if done_ids.numel() > 0:
            self.reward_buffer.extend(cur_reward_sum[done_ids][:, 0].cpu().numpy().tolist())
            self.episode_length_buffer.extend(cur_episode_length[done_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[done_ids] = 0
            cur_episode_length[done_ids] = 0

        return cur_reward_sum, cur_episode_length

    def get_training_stats(self) -> Dict[str, float]:
        """获取训练统计信息。"""
        stats = {}
        if len(self.mse_buffer) > 0:
            stats["mean_mse_loss"] = statistics.mean(self.mse_buffer)
        if len(self.reward_buffer) > 0:
            stats["mean_reward"] = statistics.mean(self.reward_buffer)
        if len(self.episode_length_buffer) > 0:
            stats["mean_episode_length"] = statistics.mean(self.episode_length_buffer)
        return stats

    def train_mode(self):
        """Set to training mode."""
        self.vision_encoder.train()

    def eval_mode(self):
        """Set to evaluation mode."""
        self.vision_encoder.eval()

    def save(self, path: str, format_tag: str = "lbc_loco", **kwargs):
        """保存训练期 ckpt（包含续训需要的全部 state）。

        ckpt 内容：
          - format                     : 固定 "lbc_loco"
          - vision_encoder_state_dict  : 学生，真机必需
          - teacher_actor_state_dict   : 教师 loco actor，真机必需（latent → joint）
          - teacher_encoder_state_dict : loco scan encoder，仅训练期续训
          - optimizer_state_dict       : 仅续训
          - current_iteration / total_steps / learning_rate
        """
        if format_tag != "lbc_loco":
            raise ValueError(
                f"AlgorithmLBC only supports format_tag='lbc_loco', got '{format_tag}'"
            )
        saved = {
            "format": format_tag,
            "vision_encoder_state_dict": self.vision_encoder.state_dict(),
            "teacher_encoder_state_dict": self.teacher_encoder.state_dict(),
            "teacher_actor_state_dict": self.teacher_actor.state_dict(),
            "goal_dim": self.goal_dim,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_iteration": self.current_iteration,
            "total_steps": self.total_steps,
            "learning_rate": self.learning_rate,
        }
        saved.update(kwargs)

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(saved, path)

    def load(self, path: str, expected_format: Optional[str] = None, load_optimizer: bool = True):
        """加载训练期 ckpt（续训用）。"""
        ckpt = torch.load(path, weights_only=False, map_location=self.device)

        # Format guard
        if expected_format is not None:
            got = ckpt.get("format")
            if got != expected_format:
                raise ValueError(
                    f"Ckpt format mismatch: expected '{expected_format}', "
                    f"got '{got}' at {path}. "
                    f"Keys in ckpt: {sorted(ckpt.keys())}"
                )

        # 学生 VisionEncoder
        if "vision_encoder_state_dict" in ckpt:
            self.vision_encoder.load_state_dict(ckpt["vision_encoder_state_dict"])

        # Loco 侧教师
        if "teacher_encoder_state_dict" in ckpt:
            self.teacher_encoder.load_state_dict(ckpt["teacher_encoder_state_dict"])
        if "teacher_actor_state_dict" in ckpt:
            self.teacher_actor.load_state_dict(ckpt["teacher_actor_state_dict"])

        # Optimizer
        if load_optimizer and "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError:
                # optimizer state 不匹配时静默忽略，不影响 student 权重
                pass

        if "current_iteration" in ckpt:
            self.current_iteration = ckpt["current_iteration"]
        if "total_steps" in ckpt:
            self.total_steps = ckpt["total_steps"]

    def load_teacher_state_dict(self, full_state: dict, source: str = "<platform>"):
        """Load frozen teacher encoder/actor from a platform-selected state dict."""
        if isinstance(full_state, dict) and "model_state_dict" in full_state:
            full_state = full_state["model_state_dict"]
        if isinstance(full_state, dict) and "state_dict" in full_state:
            full_state = full_state["state_dict"]
        if not isinstance(full_state, dict):
            raise ValueError(f"Unsupported teacher checkpoint from {source}: not a state dict")

        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in full_state.items()
            if isinstance(k, str) and k.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError(
                f"No 'encoder.*' keys found in {source}; "
                f"this does not look like an ActorCriticEncoder teacher ckpt."
            )
        if hasattr(self.teacher_encoder, "mlp"):
            self.teacher_encoder.mlp.load_state_dict(encoder_state)
        else:
            self.teacher_encoder.load_state_dict(encoder_state)

        actor_state = {
            k.replace("actor.", ""): v
            for k, v in full_state.items()
            if isinstance(k, str) and k.startswith("actor.")
        }
        if not actor_state:
            raise ValueError(
                f"No 'actor.*' keys found in {source}; ckpt structure unexpected."
            )
        self.teacher_actor.load_state_dict(actor_state)
        self._freeze_teacher()

    def load_teacher_from_locomotion_ckpt(self, ckpt_path: str):
        """Backward-compatible wrapper for platform-selected teacher files."""
        full_state = torch.load(ckpt_path, weights_only=False, map_location=self.device)
        self.load_teacher_state_dict(full_state, source=ckpt_path)
