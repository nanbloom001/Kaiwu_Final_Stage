#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any
import time
import os

from agent_ppo.feature.definition import RolloutStorage


class AlgorithmPPO:
    """
    PPO algorithm for training locomotion policies
    PPO算法，用于训练运动策略
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
        logger: Any = None,
        monitor: Any = None,
        # PPO hyperparameters
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        normalize_value_loss: bool = True,
        num_mini_batches: int = 4,
        num_learning_epochs: int = 5,
        desired_kl: float = 0.01,
        schedule: str = "adaptive",
        **kwargs,
    ):
        """
        Initialize PPO algorithm
        初始化PPO算法

        Args:
            model: Actor-critic network (ActorCritic or ActorCriticEncoder)
            model: Actor-Critic网络（ActorCritic或ActorCriticEncoder）
            optimizer: Optimizer for model parameters
            optimizer: 模型参数优化器
            device: Device for computation
            device: 计算设备
            logger: Logger for metrics
            logger: 指标日志记录器
            monitor: Monitor for performance tracking
            monitor: 性能监控器
            clip_param: PPO clip parameter (epsilon)
            clip_param: PPO裁剪参数（epsilon）
            gamma: Discount factor
            gamma: 折扣因子
            lam: GAE lambda parameter
            lam: GAE lambda参数
            value_loss_coef: Coefficient for value loss
            value_loss_coef: 价值损失系数
            entropy_coef: Coefficient for entropy bonus
            entropy_coef: 熵奖励系数
            learning_rate: Initial learning rate
            learning_rate: 初始学习率
            max_grad_norm: Max gradient norm for clipping
            max_grad_norm: 梯度裁剪最大范数
            use_clipped_value_loss: Whether to clip value loss
            use_clipped_value_loss: 是否裁剪价值损失
            normalize_value_loss: Whether to normalize value loss by returns variance
            normalize_value_loss: 是否按回报方差归一化价值损失
            num_mini_batches: Number of mini-batches per epoch
            num_mini_batches: 每个epoch的mini-batch数量
            num_learning_epochs: Number of epochs per update
            num_learning_epochs: 每次更新的epoch数量
            desired_kl: Target KL divergence for adaptive learning rate
            desired_kl: 自适应学习率的目标KL散度
            schedule: Learning rate schedule ("adaptive" or "fixed")
            schedule: 学习率调度策略（"adaptive"或"fixed"）
        """
        self.device = device
        self.actor_critic = model
        self.optimizer = optimizer
        self.logger = logger
        self.monitor = monitor

        # PPO hyperparameters
        # PPO超参数
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.normalize_value_loss = normalize_value_loss
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.desired_kl = desired_kl
        self.schedule = schedule

        # Minimum std clamp (prevents std from going negative / too small)
        # 标准差下限（防止标准差变为负值或过小）
        from agent_ppo.conf.conf import Config

        self.min_std = torch.tensor(Config.CURRENT.min_normalized_std, device=device)

        # Training state
        # 训练状态
        self.train_step = 0
        self.last_report_monitor_time = 0

        # Storage (to be initialized)
        # 存储（待初始化）
        self.storage = None

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: tuple,
        critic_obs_shape: tuple,
        action_shape: tuple,
        device: torch.device = None,
    ):
        """
        Initialize rollout storage buffer
        初始化rollout存储缓冲区

        Args:
            num_envs: Number of parallel environments
            num_envs: 并行环境数量
            num_transitions_per_env: Steps per rollout
            num_transitions_per_env: 每次rollout的步数
            actor_obs_shape: Shape of actor observations
            actor_obs_shape: Actor观测形状
            critic_obs_shape: Shape of critic observations
            critic_obs_shape: Critic观测形状
            action_shape: Shape of actions
            action_shape: 动作形状
            device: Device for storage tensors
            device: 存储张量的设备
        """
        device = device or self.device
        self.storage = RolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            privileged_obs_shape=critic_obs_shape,
            actions_shape=action_shape,
            device=device,
        )

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor = None) -> tuple:
        """
        Generate actions and compute values
        生成动作并计算值函数

        Args:
            obs: [B, num_obs] flat observation tensor (for actor)
            obs: [B, num_obs] Actor观测张量
            critic_obs: [B, num_critic_obs] critic observation tensor
            critic_obs: [B, num_critic_obs] Critic观测张量

        Returns:
            tuple: (actions, values, log_probs, action_mean, action_std)
            返回值：(动作, 值函数, 对数概率, 动作均值, 动作标准差)
        """
        if critic_obs is None:
            critic_obs = obs

        with torch.no_grad():
            # Sample actions using actor
            # 使用 actor 采样动作
            actions = self.actor_critic.act(obs)

            # Compute values using critic (with privileged observations)
            # 使用 critic 计算价值（使用特权观测）
            values = self.actor_critic.evaluate(critic_obs)

            # Get log probabilities and distribution parameters
            # 获取对数概率和分布参数
            log_probs = self.actor_critic.get_actions_log_prob(actions)
            action_mean = self.actor_critic.action_mean.detach()
            action_std = self.actor_critic.action_std.detach()

        return actions, values, log_probs, action_mean, action_std

    def compute_returns(self, last_obs: torch.Tensor):
        """
        Compute returns and advantages using GAE
        使用GAE方法计算回报和优势函数

        Args:
            last_obs: Observations from the last step (for bootstrap value)
            last_obs: 最后一步的观测（用于引导值计算）
        """
        with torch.no_grad():
            last_values = self.actor_critic.evaluate(last_obs)

        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def learn(self) -> tuple:
        """
        Train the policy using PPO algorithm
        使用PPO算法训练策略

        Returns:
            tuple: (mean_surrogate_loss, mean_value_loss, mean_entropy_loss)
            返回值：(平均替代损失, 平均价值损失, 平均熵损失)
        """
        # Initialize loss accumulators
        # 初始化损失累加器
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0

        # Get mini-batch generator
        # 获取mini-batch生成器
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Training loop over mini-batches
        # mini-batch训练循环
        for sample_idx, sample in enumerate(generator):
            # Unpack sample data
            # 解包样本数据
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                # Not used, placeholder
                # 未使用，占位符
                hid_states_batch,
                masks_batch,
            ) = sample

            # Forward pass through actor-critic
            # 前向传播计算actor-critic
            self.actor_critic.update_distribution(obs_batch)

            # Get action log probabilities
            # 获取动作对数概率
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            # Get entropy
            # 获取熵
            entropy_batch = self.actor_critic.entropy

            # Get value estimates
            # 获取价值估计
            value_batch = self.actor_critic.evaluate(critic_obs_batch)

            # Get distribution parameters
            # 获取分布参数
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std

            # Adaptive learning rate
            # 自适应学习率
            self._update_learning_rate(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch)

            # Compute losses
            # 计算损失
            surrogate_loss = self._compute_surrogate_loss(
                actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch
            )
            value_loss = self._compute_value_loss(value_batch, returns_batch, target_values_batch)

            # Combine losses
            # 组合损失
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # NaN/Inf guard: skip this mini-batch update entirely if loss is invalid
            # NaN/Inf 防护：如果 loss 非法则跳过此 mini-batch 更新，避免坏梯度写入参数
            if not torch.isfinite(loss):
                if self.logger:
                    self.logger.warning(
                        f"[PPO] NaN/Inf loss detected at step {self.train_step}, "
                        f"mini-batch {sample_idx}. Skipping this update. "
                        f"surrogate={surrogate_loss.item()}, value={value_loss.item()}"
                    )
                continue

            # Gradient update
            # 梯度更新
            self.optimizer.zero_grad()
            loss.backward()

            # Extra guard: if any gradient is NaN after backward, skip step
            # 额外防护：backward 后如果梯度含 NaN，跳过 optimizer.step
            grad_finite = True
            for p in self.actor_critic.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_finite = False
                    break

            if not grad_finite:
                if self.logger:
                    self.logger.warning(
                        f"[PPO] NaN/Inf gradient detected at step {self.train_step}, "
                        f"mini-batch {sample_idx}. Zeroing grads and skipping optimizer.step()."
                    )
                self.optimizer.zero_grad()
                continue

            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Clamp action std: replace NaN/Inf, then enforce [min_std, 1e6]
            # 清洗并夹住 std：替换 NaN/Inf，然后限制到 [min_std, 1e6]
            if hasattr(self.actor_critic, "std") and self.min_std is not None:
                max_std_t = torch.full_like(self.actor_critic.std.data, 1.0e6)
                safe_std = torch.nan_to_num(
                    self.actor_critic.std.data,
                    nan=1.0,
                    posinf=1.0e6,
                    neginf=0.0,
                )
                self.actor_critic.std.data.copy_(torch.clamp(safe_std, min=self.min_std, max=max_std_t))

            # Accumulate losses (use 0.0 for any remaining NaN as safety net)
            # 累加损失（对残留 NaN 兜底为 0.0）
            sl = surrogate_loss.item()
            vl = value_loss.item()
            el = entropy_batch.mean().item()
            mean_surrogate_loss += sl if not (sl != sl) else 0.0
            mean_value_loss += vl if not (vl != vl) else 0.0
            mean_entropy_loss += el if not (el != el) else 0.0

        # Average losses
        # 平均损失
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates

        # Report metrics
        # 上报指标
        self._report_training_metrics(mean_surrogate_loss, mean_value_loss, mean_entropy_loss)

        self.train_step += 1
        return mean_surrogate_loss, mean_value_loss, mean_entropy_loss

    def _update_learning_rate(
        self,
        mu_batch: torch.Tensor,
        sigma_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
    ):
        """
        Adaptively update learning rate based on KL divergence
        基于KL散度自适应更新学习率
        """
        if self.desired_kl is None or self.schedule != "adaptive":
            return

        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)

            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _compute_surrogate_loss(
        self,
        actions_log_prob_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PPO surrogate loss with clipping
        计算带裁剪的PPO替代损失
        """
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        return torch.max(surrogate, surrogate_clipped).mean()

    def _compute_value_loss(
        self,
        value_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value function loss with optional clipping and normalization.

        计算价值函数损失（可选裁剪和归一化）。
        """
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            raw_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            raw_loss = (returns_batch - value_batch).pow(2).mean()

        if self.normalize_value_loss:
            # Normalize by returns variance to keep value_loss scale-invariant
            # 按回报方差归一化，保持 value_loss 尺度不变
            returns_var = returns_batch.detach().var() + 1e-8
            return raw_loss / returns_var

        return raw_loss

    def _report_training_metrics(
        self,
        mean_surrogate_loss: float,
        mean_value_loss: float,
        mean_entropy_loss: float,
    ):
        """
        Report training metrics to monitor
        向监控系统上报训练指标
        """
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "policy_loss": mean_surrogate_loss,
                "value_loss": mean_value_loss,
                "entropy_loss": mean_entropy_loss,
                "total_loss": mean_surrogate_loss + mean_value_loss + mean_entropy_loss,
                "learning_rate": self.learning_rate,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now
