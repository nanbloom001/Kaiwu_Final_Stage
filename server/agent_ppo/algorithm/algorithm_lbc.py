#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from __future__ import annotations

import hashlib
import os
import random
import statistics
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.checkpoint_io import (
    KAIWU_TRAIN_FORMAT,
    KAIWU_TRAIN_SCHEMA_VERSION,
    is_kaiwu_train_bundle,
    low_level_teacher_parts,
    validate_low_level_spec,
    vision_checkpoint_candidates,
    vision_parent_candidates,
    vision_phase_label,
)

_SAFETY_CALIBRATION_SAMPLES = 256


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
        depth_shape: Tuple[int, int, int] = (180, 320, 1),
    ):
        self.device = device
        self.mode = "loco"
        self.latent_dim = latent_dim
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.proprio_dim = proprio_dim
        self.scan_dim = scan_dim
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
        self.resume_loaded = False

        # 线性 ramp 调度状态（阶段 4）。每次启动 reset 环境 + 清零 LSTM hidden，
        # 因此这里只存 ramp 进度和软停留诊断，不存任何 LSTM 状态。
        # ramp_probability 由 workflow 按 elapsed_h 重算并写回；checkpoint 只记录
        # 断点，让续训从相同 ramp 比例继续（而不依赖墙钟时间）。
        self.ramp_probability = 0.0
        self.ramp_start_h = 0.50      # 默认：前 30min 教师预热后开始 ramp
        self.ramp_end_h = 5.00        # 默认：约 4.5h ramp 到 100%
        self.ramp_clock_h = 0.0
        self.soft_stay_frozen = False
        self.soft_stay_reason: Optional[str] = None
        self.training_status = "initialized"
        self.safety_threshold = float("inf")
        self.safety_fixed = False
        self.safety_calibration_l2: list[float] = []
        self.lr_scheduler_state: Optional[dict] = None

        # 血缘与审计（运行时不做字节级 SHA 门禁，只记录用于追溯）
        self.parent_checkpoint_sha256 = "unknown"
        self.teacher_low_level_sha256 = "unknown"
        self.config_sha256 = "unknown"
        self.code_commit = "unknown"

        # LSTM reset 契约记录（供 eval/导出对齐 reset mask 语义）
        self.lstm_reset_contract = {
            "per_env_hidden": True,          # 每环境独立 hidden/cell
            "reset_on_done": True,           # episode done 在同 env 索引清零
            "teacher_to_student_no_clear": True,  # 同 episode 内切换驱动不清空
            "eval_no_random_aug": True,      # eval 关闭随机增强
            "cross_run_not_restored": True,  # 跨运行不恢复旧 hidden
        }

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
            [ proprio(proprio_dim) | height_scan(scan_dim) | depth(H*W*C) ]
        """
        if isinstance(obs, dict):
            return obs

        H, W, C = self.depth_shape
        proprio = obs[:, : self.proprio_dim]
        height_scan = obs[:, self.proprio_dim : self.proprio_dim + self.scan_dim]
        depth_flat = obs[:, self.proprio_dim + self.scan_dim :]
        depth_image = depth_flat.view(-1, H, W, C)
        return {
            "proprio": proprio,
            "height_scan": height_scan,
            "depth_image": depth_image,
        }

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
            policy_input = torch.cat([obs["proprio"], teacher_latent], dim=-1)
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
            policy_input = torch.cat([obs["proprio"], student_latent], dim=-1)
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
                - "l2_distance": L2 距离
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
            l2_distance = per_sample_sq.sqrt().mean()

        return {
            "loss": mse_loss,
            "mse_loss": mse_loss,
            "l2_distance": l2_distance,
            "teacher_latent": teacher_latent.detach(),
            "student_latent": student_latent.detach(),
        }

    # ------------------------------------------------------------------
    # 单次 forward 的逐环境线性 DAgger（阶段 4）
    # ------------------------------------------------------------------

    def prepare_vision_update(self, obs, masks: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """单次 forward 缓存：一次跑出教师/学生的 latent 和 action。

        P1 第 5 条：动作选择和三路损失必须复用同一 forward 结果，否则
        学生驱动环境时 LSTM 会对同一观测推进两次（一次选动作，一次算 loss），
        导致 hidden 错位、latent 与动作不对应。

        返回的 batch 中：
          - teacher_latent / teacher_action：no_grad，监督目标
          - student_latent：带梯度，三路损失回传到此
          - student_action：teacher_actor(proprio, student_latent)，带梯度
            （梯度穿过冻结 teacher_actor 回传到 vision_encoder）
        """
        obs_dict = self._split_obs(obs)

        # Teacher forward (frozen, no_grad)
        with torch.no_grad():
            teacher_latent = self.teacher_encoder(obs_dict["height_scan"])
            teacher_action = self.teacher_actor(
                torch.cat([obs_dict["proprio"], teacher_latent], dim=-1)
            )

        # Student forward (trainable, 带梯度)。LSTM 在此推进一次。
        student_latent = self.vision_encoder(
            depth_image=obs_dict["depth_image"],
            proprio=obs_dict["proprio"],
            masks=masks,
        )
        # student_action 复用同一 student_latent（不再跑第二次 vision_encoder）
        student_action = self.teacher_actor(
            torch.cat([obs_dict["proprio"], student_latent], dim=-1)
        )

        return {
            "proprio": obs_dict["proprio"],
            "teacher_latent": teacher_latent,
            "teacher_action": teacher_action,
            "student_latent": student_latent,      # 带梯度
            "student_action": student_action,      # 带梯度（穿过冻结 actor）
            "student_finite": torch.isfinite(student_action).all(dim=-1),
        }

    def compute_three_way_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """三路蒸馏损失（阶段 4 计划 §6）。

            latent_loss = SmoothL1(student_latent, teacher_latent)
            cosine_loss = 1 - cosine(student_latent, teacher_latent).mean()
            action_loss = SmoothL1(student_action, teacher_action)
            total       = 0.5*latent + 0.1*cosine + 1.0*action

        action_loss 的梯度穿过冻结 teacher_actor 回传到 vision_encoder：
        teacher_actor 的权重 requires_grad=False（_freeze_teacher 已设），
        但 student_latent 带梯度，因此 autograd 会把 action_loss 的梯度
        经 teacher_actor 的线性变换回传到 student_latent，再到 vision_encoder。
        teacher_actor 自身不进入 optimizer。
        """
        t_lat = batch["teacher_latent"]
        s_lat = batch["student_latent"]
        t_act = batch["teacher_action"]
        s_act = batch["student_action"]

        latent_loss = F.smooth_l1_loss(s_lat, t_lat)
        cosine_loss = 1.0 - F.cosine_similarity(s_lat, t_lat, dim=-1).mean()
        action_loss = F.smooth_l1_loss(s_act, t_act)
        total = 0.5 * latent_loss + 0.1 * cosine_loss + 1.0 * action_loss

        return {
            "latent_loss": latent_loss,
            "cosine_loss": cosine_loss,
            "action_loss": action_loss,
            "total_loss": total,
        }

    def select_driver_actions(
        self,
        batch: Dict[str, torch.Tensor],
        student_drive_probability: float,
        safety_threshold: float,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """逐环境采样 driver：Bernoulli(probability) 决定学生驱动，安全接管兜底。

        学生动作 NaN/Inf 或与教师动作 L2 超过 safety_threshold 时，该环境
        由教师接管（但仍保留为有效训练样本，权重降为 0.25）。
        动作来源全部是 prepare_vision_update 缓存的结果，不重复跑 forward。
        """
        probability = float(student_drive_probability)
        requested_student = (
            torch.rand(batch["teacher_action"].shape[0], device=self.device)
            < probability
        )
        # 学生与教师动作的逐样本 L2（用于安全判定）
        with torch.no_grad():
            per_sample_l2 = (
                (batch["student_action"] - batch["teacher_action"])
                .pow(2).sum(dim=-1).sqrt()
            )
        excessive = per_sample_l2 > float(safety_threshold)
        safety_takeover = requested_student & (
            ~batch["student_finite"] | excessive
        )
        effective_student = requested_student & ~safety_takeover
        actions = torch.where(
            effective_student.unsqueeze(-1),
            batch["student_action"].detach(),
            batch["teacher_action"].detach(),
        )
        return actions, {
            "requested_student": requested_student,
            "effective_student": effective_student,
            "safety_takeover": safety_takeover,
            "per_sample_l2": per_sample_l2,
        }

    def finish_vision_update(
        self,
        batch: Dict[str, torch.Tensor],
        sample_weights: torch.Tensor,
    ) -> Dict[str, float]:
        """加权三路损失更新。sample_weights: 1.0/0.25/0（计划 §10.2）。

        首轮不启用 replay（P1 第 6 条）：LSTM 单帧 replay 会用错误的 hidden
        历史算 latent，训练到错误目标。replay 接口预留为 no-op。
        """
        weights = torch.as_tensor(
            sample_weights, device=self.device, dtype=batch["student_latent"].dtype
        ).reshape(-1)
        weights = torch.clamp(torch.nan_to_num(weights, nan=0.0), 0.0, 1.0)

        loss_dict = self.compute_three_way_loss(batch)
        # 加权：把 per-sample 权重作用到每一项（每项都是 mean，改写为加权 mean）
        # 这里简化：权重整体作用于 total_loss 的反向。若 train_mask 全 0 则跳过。
        train_mask = weights > 0.0
        grad_norm = torch.zeros((), device=self.device)

        if bool(train_mask.any().item()):
            # per-sample total loss
            t_lat = batch["teacher_latent"]
            s_lat = batch["student_latent"]
            t_act = batch["teacher_action"]
            s_act = batch["student_action"]
            per_latent = F.smooth_l1_loss(s_lat, t_lat, reduction="none").mean(dim=-1)
            per_cosine = 1.0 - F.cosine_similarity(s_lat, t_lat, dim=-1)
            per_action = F.smooth_l1_loss(s_act, t_act, reduction="none").mean(dim=-1)
            per_total = 0.5 * per_latent + 0.1 * per_cosine + 1.0 * per_action
            w = weights[train_mask]
            loss = (w * per_total[train_mask]).sum() / w.sum().clamp_min(1.0)

            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("vision distill loss is NaN/Inf")

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.vision_encoder.parameters(), self.max_grad_norm
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise FloatingPointError("vision distill gradient norm is NaN/Inf")
            self.optimizer.step()
        else:
            loss = loss_dict["total_loss"].detach()

        self.total_steps += int(batch["teacher_action"].shape[0])

        with torch.no_grad():
            t_lat = batch["teacher_latent"]
            s_lat = batch["student_latent"]
            t_act = batch["teacher_action"]
            s_act = batch["student_action"]
            cos_lat = F.cosine_similarity(s_lat, t_lat, dim=-1).mean().item()
            cos_act = F.cosine_similarity(s_act, t_act, dim=-1).mean().item()
            latent_mse = (s_lat - t_lat).pow(2).sum(dim=-1).mean().item()
            action_mse = (s_act - t_act).pow(2).sum(dim=-1).mean().item()
            # normalized action mse: 按维除以教师方差
            t_var = t_act.var(dim=0, unbiased=False).clamp_min(1e-6)
            per_dim = (s_act - t_act).pow(2).mean(dim=0)
            norm_action_mse = (per_dim / t_var).mean().item()

        return {
            "loss": float(loss.detach().item()),
            "latent_loss": float(loss_dict["latent_loss"].detach().item()),
            "cosine_loss": float(loss_dict["cosine_loss"].detach().item()),
            "action_loss": float(loss_dict["action_loss"].detach().item()),
            "latent_mse": latent_mse,
            "action_mse": action_mse,
            "normalized_action_mse": norm_action_mse,
            "latent_cosine": cos_lat,
            "action_cosine": cos_act,
            "grad_norm": float(grad_norm.detach().item()),
            "nonfinite_rate": float((~batch["student_finite"]).float().mean().item()),
        }

    def assert_student_parameters_finite(self):
        """断言 vision_encoder 参数全有限（每 outer iteration 检查一次）。"""
        params = list(self.vision_encoder.parameters())
        if not all(bool(torch.isfinite(p).all().item()) for p in params):
            raise FloatingPointError("vision_encoder parameters became NaN/Inf")

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
        l2_dist = loss_dict["l2_distance"].item()
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
            "l2_distance": l2_dist,
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

    def load_teacher_from_locomotion_ckpt(self, ckpt_path: str):
        """从 locomotion ckpt (ActorCriticEncoder.state_dict()) 按前缀拆分加载教师。

        locomotion ckpt 格式:
            encoder.0.weight, encoder.2.weight, encoder.4.weight ...  → teacher_encoder.mlp
            actor.0.weight,   actor.2.weight,   ...                    → teacher_actor
            critic.*                                                   (丢弃)
            log_std / std                                              (丢弃)

        说明:
            - ActorCriticEncoder.encoder 是 Sequential(Linear, ELU, Linear, ELU, Linear, L2Norm)，
              与 DmEncoder.mlp 结构一致，去掉 "encoder." 前缀即可对齐。
            - 教师 Actor 期望是 nn.Sequential(Linear, Act, Linear, Act, Linear, Act, Linear)，
              ActorCritic.actor 的 key 形如 actor.0.weight ...，去掉 "actor." 前缀后能对齐。

        Args:
            ckpt_path: locomotion 模型文件路径（model.ckpt-locomotion-{id}.pkl）。
        """
        full_state = torch.load(ckpt_path, weights_only=False, map_location=self.device)
        parent_sha = self._sha256_file(ckpt_path) if os.path.isfile(ckpt_path) else "unknown"
        if is_kaiwu_train_bundle(full_state):
            validate_low_level_spec(
                full_state,
                expected={
                    "proprio_dim": self.proprio_dim,
                    "scan_dim": self.scan_dim,
                    "latent_dim": self.latent_dim,
                    "action_dim": 12,
                    "goal_dim": 0,
                },
            )
            encoder_state, actor_state = low_level_teacher_parts(full_state)
            if hasattr(self.teacher_encoder, "mlp"):
                self.teacher_encoder.mlp.load_state_dict(
                    encoder_state, strict=True
                )
            else:
                self.teacher_encoder.load_state_dict(
                    encoder_state, strict=True
                )
            self.teacher_actor.load_state_dict(actor_state, strict=True)
            self._freeze_teacher()
            # 记录父血缘供追溯（P1 第 4 条）：打印实际命中路径、SHA、model_spec，
            # 让操作者核对是否真的是 daggerfull-16288 血缘。运行时不做字节级门禁。
            self.parent_checkpoint_sha256 = parent_sha
            self.teacher_low_level_sha256 = parent_sha
            spec = full_state.get("model_spec", {})
            print(
                f"[AlgorithmLBC] vision parent loaded from {ckpt_path}\n"
                f"  sha256={parent_sha}\n"
                f"  model_spec={spec}\n"
                f"  (operator must confirm this is daggerfull-16288 lineage)"
            )
            return

        # 拆分 encoder 权重 → teacher_encoder.mlp
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in full_state.items()
            if k.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError(
                f"No 'encoder.*' keys found in {ckpt_path}; "
                f"this does not look like a locomotion ActorCriticEncoder ckpt."
            )
        # DmEncoder 用 self.mlp
        if hasattr(self.teacher_encoder, "mlp"):
            self.teacher_encoder.mlp.load_state_dict(encoder_state)
        else:
            self.teacher_encoder.load_state_dict(encoder_state)

        # 拆分 actor 权重 → teacher_actor
        actor_state = {
            k.replace("actor.", ""): v
            for k, v in full_state.items()
            if k.startswith("actor.")
        }
        if not actor_state:
            raise ValueError(
                f"No 'actor.*' keys found in {ckpt_path}; ckpt structure unexpected."
            )
        self.teacher_actor.load_state_dict(actor_state)

        # critic.* 和 log_std 按设计丢弃

        # 记录父血缘（raw state dict 分支，历史格式）
        self.parent_checkpoint_sha256 = parent_sha
        self.teacher_low_level_sha256 = parent_sha
        print(
            f"[AlgorithmLBC] vision parent loaded (raw state dict) from {ckpt_path}\n"
            f"  sha256={parent_sha}\n"
            f"  (operator must confirm this is daggerfull-16288 lineage)"
        )

        # 确保教师冻结
        self._freeze_teacher()

    # ------------------------------------------------------------------
    # 视觉训练包 codec（阶段 4：kaiwu_train_v1 + modules.vision_encoder）
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _vision_model_spec(self) -> dict:
        """视觉学生的 model_spec。教师 low-level spec 由父包带入，这里只描述学生。"""
        return {
            "task": "standard",
            "proprio_dim": int(self.proprio_dim),
            "scan_dim": int(self.scan_dim),  # 教师用，学生推理不读
            "latent_dim": int(self.latent_dim),
            "depth_shape": list(self.depth_shape),
            "actor_input_dim": int(self.proprio_dim) + int(self.latent_dim),
            "action_dim": 12,
            "goal_dim": 0,
        }

    @staticmethod
    def _capture_rng_state() -> dict:
        """Capture driver-mask and augmentation RNG state for safe resume."""
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: Any) -> None:
        """Restore RNG state saved by _capture_rng_state()."""
        if not isinstance(state, dict):
            return
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

    def vision_bundle_payload(
        self,
        *,
        platform_model_id: Optional[str | int] = None,
        ramp_label: Optional[str] = None,
        **extra: Any,
    ) -> dict:
        """构造视觉训练包 payload（kaiwu_train_v1 + modules.vision_encoder）。

        与特权 DAgger 的 kaiwu_train_v1 区别：
          - modules.vision_encoder：视觉学生（本轮训练对象）
          - modules.low_level：冻结教师副本（Encoder + Actor77），供续训和血缘审计
          - 不含 modules.privileged_teacher（那是 flat301 教师，视觉阶段不需要）
          - 不含 LSTM hidden state（跨运行不恢复，见 lstm_reset_contract）

        capabilities.deployable=false：本训练包不是 Jetson 制品。
        """
        valid_label = vision_phase_label(ramp_label) if ramp_label else "visionfull"
        payload = {
            "format": KAIWU_TRAIN_FORMAT,
            "schema_version": KAIWU_TRAIN_SCHEMA_VERSION,
            "artifact_role": "vision_training_bundle",
            "stage_type": "standard_vision_distill",
            "ramp_label": valid_label,
            "model_spec": self._vision_model_spec(),
            "modules": {
                "vision_encoder": {
                    "class_name": self.vision_encoder.__class__.__name__,
                    "state_dict": self.vision_encoder.state_dict(),
                    "trainable": True,
                },
                # 冻结教师副本：Encoder(scan256→latent32) + Actor77(proprio+latent→action12)
                # 供续训时重建教师、血缘审计和未来 lbc_loco 导出拆分。
                "low_level": {
                    "class_name": "ActorCriticEncoder",
                    "encoder_state_dict": self.teacher_encoder.state_dict(),
                    "actor_state_dict": self.teacher_actor.state_dict(),
                    "frozen": True,
                    "source": "parent_daggerfull",
                },
            },
            "optimizers": {
                "vision_distill": self.optimizer.state_dict(),
            },
            "training_state": {
                "current_iteration": self.current_iteration,
                "iteration_semantics": "completed_outer_iterations_v1",
                "total_steps": self.total_steps,
                "ramp_probability": self.ramp_probability,
                "ramp_start_h": self.ramp_start_h,
                "ramp_end_h": self.ramp_end_h,
                "ramp_clock_h": self.ramp_clock_h,
                "soft_stay_frozen": self.soft_stay_frozen,
                "soft_stay_reason": self.soft_stay_reason,
                "safety_threshold": self.safety_threshold,
                "safety_fixed": self.safety_fixed,
                "safety_calibration_l2": list(
                    self.safety_calibration_l2[-_SAFETY_CALIBRATION_SAMPLES:]
                ),
                "lr_scheduler_state": self.lr_scheduler_state,
                "rng_state": self._capture_rng_state(),
                "training_status": self.training_status,
            },
            "replay": {
                # 首轮不启用单帧 replay（LSTM 单帧 replay 会训到错误 latent）。
                # 字段保留为空，待未来序列回放（8-16 帧 + burn-in）消融时填充。
                "enabled": False,
                "note": "single-frame replay invalid for LSTM; deferred to sequence-replay ablation",
            },
            "lineage": {
                "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
                "teacher_low_level_sha256": self.teacher_low_level_sha256,
                "config_sha256": self.config_sha256,
                "code_commit": self.code_commit,
                "platform_model_id": (
                    None if platform_model_id is None else str(platform_model_id)
                ),
            },
            "capabilities": {
                "task": "standard",
                "uses_depth": True,
                "uses_height_scan_at_inference": False,
                "goal_dim": 0,
                "deployable": False,
            },
            "lstm_reset_contract": dict(self.lstm_reset_contract),
        }
        payload.update(extra)
        return payload

    def save_vision_bundle(
        self,
        path: str,
        *,
        platform_model_id: Optional[str | int] = None,
        ramp_label: Optional[str] = None,
        **extra: Any,
    ) -> str:
        """保存视觉训练包到 path，返回文件 SHA256。"""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = self.vision_bundle_payload(
            platform_model_id=platform_model_id,
            ramp_label=ramp_label,
            **extra,
        )
        torch.save(payload, path)
        return self._sha256_file(path)

    def load_vision_bundle(self, path: str) -> None:
        """从视觉训练包恢复学生/教师/optimizer/ramp 状态。

        P1 第 7 条：不恢复 LSTM hidden。每次启动 workflow 会 reset 环境 + 清零 hidden
        （lbc_workflow 现状），这里只恢复可跨运行持久化的状态。
        """
        ckpt = torch.load(path, weights_only=False, map_location=self.device)
        if not is_kaiwu_train_bundle(ckpt):
            raise ValueError(
                f"Not a {KAIWU_TRAIN_FORMAT} vision bundle: {path} "
                f"(format={ckpt.get('format')!r})"
            )
        modules = ckpt.get("modules", {})

        # 视觉学生
        vision_section = modules.get("vision_encoder")
        if not isinstance(vision_section, dict) or "state_dict" not in vision_section:
            raise KeyError("modules.vision_encoder.state_dict missing from vision bundle")
        self.vision_encoder.load_state_dict(vision_section["state_dict"], strict=True)

        # 冻结教师副本（Encoder + Actor77）
        # save 写的是 teacher_encoder.state_dict()，key 形如 mlp.0.weight（DmEncoder
        # 内部有 self.mlp）。load 必须用 teacher_encoder.load_state_dict() 让 PyTorch
        # 按 mlp.* 前缀匹配，而不是剥前缀给 .mlp.load_state_dict()（后者期望 0.weight，
        # 会 key 不匹配）。eval 路径同理。
        low_level = modules.get("low_level", {})
        enc_state = low_level.get("encoder_state_dict")
        act_state = low_level.get("actor_state_dict")
        if not isinstance(enc_state, dict) or not isinstance(act_state, dict):
            raise KeyError(
                "modules.low_level.encoder_state_dict/actor_state_dict missing"
            )
        self.teacher_encoder.load_state_dict(enc_state, strict=True)
        self.teacher_actor.load_state_dict(act_state, strict=True)
        self._freeze_teacher()

        # Optimizer
        opt_state = ckpt.get("optimizers", {}).get("vision_distill")
        if isinstance(opt_state, dict):
            try:
                self.optimizer.load_state_dict(opt_state)
                self.learning_rate = float(
                    self.optimizer.param_groups[0]["lr"]
                )
            except ValueError as exc:
                print(
                    "[AlgorithmLBC] WARNING: optimizer state was not restored: "
                    f"{exc}"
                )

        # Ramp / training state（不恢复 LSTM hidden）
        state = ckpt.get("training_state", {})
        stored_iteration = int(state.get("current_iteration", 0))
        if state.get("iteration_semantics") == "completed_outer_iterations_v1":
            self.current_iteration = stored_iteration
        else:
            # 兼容本次修复前已生成的视觉包：旧 workflow 保存的是从 0 开始的
            # 当前 loop index，而不是已完成次数。转换后续训不会重复最后一轮。
            self.current_iteration = max(0, stored_iteration + 1)
        self.total_steps = int(state.get("total_steps", 0))
        self.ramp_probability = float(state.get("ramp_probability", 0.0))
        self.ramp_start_h = float(state.get("ramp_start_h", self.ramp_start_h))
        self.ramp_end_h = float(state.get("ramp_end_h", self.ramp_end_h))
        self.ramp_clock_h = float(
            state.get(
                "ramp_clock_h",
                self.ramp_start_h
                + self.ramp_probability * (self.ramp_end_h - self.ramp_start_h)
                if self.ramp_probability > 0.0
                else 0.0,
            )
        )
        self.soft_stay_frozen = bool(state.get("soft_stay_frozen", False))
        self.soft_stay_reason = state.get("soft_stay_reason")
        self.safety_threshold = float(
            state.get("safety_threshold", float("inf"))
        )
        self.safety_fixed = bool(state.get("safety_fixed", False))
        self.safety_calibration_l2 = [
            float(value)
            for value in state.get("safety_calibration_l2", [])
            if isinstance(value, (int, float))
        ][-_SAFETY_CALIBRATION_SAMPLES:]
        scheduler_state = state.get("lr_scheduler_state")
        self.lr_scheduler_state = (
            scheduler_state if isinstance(scheduler_state, dict) else None
        )
        self._restore_rng_state(state.get("rng_state"))
        self.training_status = str(state.get("training_status", "resumed"))
        self.resume_loaded = True

        # LSTM reset 契约（仅记录，不恢复 hidden）
        contract = ckpt.get("lstm_reset_contract")
        if isinstance(contract, dict):
            self.lstm_reset_contract = dict(contract)

        # 血缘
        lineage = ckpt.get("lineage", {})
        self.parent_checkpoint_sha256 = str(
            lineage.get("parent_checkpoint_sha256", "unknown")
        )
        self.teacher_low_level_sha256 = str(
            lineage.get("teacher_low_level_sha256", "unknown")
        )
        self.config_sha256 = str(lineage.get("config_sha256", "unknown"))
        self.code_commit = str(lineage.get("code_commit", "unknown"))

        print(
            f"[AlgorithmLBC] resumed vision bundle from {path}\n"
            f"  iteration={self.current_iteration}, ramp_probability={self.ramp_probability:.3f},\n"
            f"  parent_sha256={self.parent_checkpoint_sha256},\n"
            f"  LSTM hidden NOT restored (will reset on env.reset)"
        )

    def load_parent_bundle(self, path: str, model_id: str | int) -> str:
        """按视觉父候选顺序查找并加载特权父文件（daggerfull-16288）。

        P1 第 4 条：显式优先 daggerfull，再 locomotion，不靠偶然排序。
        返回实际命中的文件路径。
        """
        candidates = vision_parent_candidates(path, model_id)
        for candidate in candidates:
            if os.path.exists(candidate):
                self.load_teacher_from_locomotion_ckpt(candidate)
                return candidate
        raise FileNotFoundError(
            f"No vision parent checkpoint found in {path}/ for id={model_id}; "
            f"tried: {candidates[:4]}"
        )

    def load_vision_resume(self, path: str, model_id: str | int) -> Optional[str]:
        """按视觉候选顺序查找并加载续训视觉包。

        返回命中的文件路径；无候选时返回 None（调用方据此走首训父加载分支）。
        """
        candidates = vision_checkpoint_candidates(path, model_id)
        for candidate in candidates:
            if os.path.exists(candidate):
                payload = torch.load(
                    candidate, weights_only=False, map_location=self.device
                )
                modules = payload.get("modules", {}) if isinstance(payload, dict) else {}
                vision_section = (
                    modules.get("vision_encoder", {})
                    if isinstance(modules, dict)
                    else {}
                )
                if is_kaiwu_train_bundle(payload) and isinstance(
                    vision_section.get("state_dict"), dict
                ):
                    self.load_vision_bundle(candidate)
                    return candidate
                if (
                    isinstance(payload, dict)
                    and payload.get("format") == "lbc_loco"
                    and isinstance(payload.get("vision_encoder_state_dict"), dict)
                ):
                    self.load(
                        candidate,
                        expected_format="lbc_loco",
                        load_optimizer=True,
                    )
                    self.resume_loaded = True
                    return candidate
                # A low-level daggerfull/locomotion bundle is a parent, not a
                # visual resume. Skip it so _load_lbc_loco can reach
                # load_parent_bundle().
        return None
