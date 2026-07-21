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
from torch.distributions import Normal
from typing import Any


def resolve_nn_activation(activation: str) -> nn.Module:
    """
    Get activation function by name
    根据名称获取激活函数
    """
    activation_map = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if activation not in activation_map:
        raise ValueError(f"Unknown activation: {activation}. Available: {list(activation_map.keys())}")
    return activation_map[activation]


class ActorCritic(nn.Module):
    """
    Actor-Critic network with flat tensor interface
    使用扁平张量接口的Actor-Critic网络
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float | list | tuple = 1.0,
        noise_std_type: str = "scalar",
        build_critic: bool = True,
        actor_last_layer_init_gain: float | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize ActorCritic / 初始化 ActorCritic.

        Args:
            num_obs: Actor 观测维度
            num_critic_obs: Critic 观测维度（build_critic=False 时忽略）
            num_actions: 动作维度
            actor_hidden_dims: Actor MLP 隐藏层大小
            critic_hidden_dims: Critic MLP 隐藏层大小
            activation: 激活函数名称
            init_noise_std: 探索噪声初始标准差
            noise_std_type: "scalar" 或 "log"
            build_critic: 是否构建 critic 网络。False 用于纯推理模型
                          （例如 hier_nav 阶段冻结的 loco_model），不创建 critic
                          也不允许 evaluate()。
        """
        super().__init__()

        activation_fn = resolve_nn_activation(activation)
        self.has_critic = build_critic

        # Build actor MLP
        # 构建策略网络
        actor_layers = []
        actor_layers.append(nn.Linear(num_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # Actor 末层 small-gain 正交初始化（可选）。
        # 动机：默认 kaiming 初始化下 actor 末层输出尺度偏大，叠加“无界输出 + 事后 clamp”，
        # 训练初期采样会大面积落在 cmd_range 之外，把 μ 持续往边界外推（实测漂到 36+）。
        # small gain（如 0.01）让初始 μ≈0，配合 per-dim init_noise_std 把工作点压在范围内。
        if actor_last_layer_init_gain is not None:
            last_linear = None
            for _m in self.actor:
                if isinstance(_m, nn.Linear):
                    last_linear = _m
            if last_linear is not None:
                torch.nn.init.orthogonal_(last_linear.weight, gain=actor_last_layer_init_gain)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

        # Build critic MLP (with LayerNorm) / 构建价值网络（含层标准化）
        if build_critic:
            critic_layers = []
            critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
            critic_layers.append(activation_fn)
            for i in range(len(critic_hidden_dims)):
                if i == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                    critic_layers.append(nn.LayerNorm(critic_hidden_dims[i + 1]))
                    critic_layers.append(activation_fn)
            self.critic = nn.Sequential(*critic_layers)
        else:
            self.critic = None

        # Action noise initialization
        # 动作噪声初始化
        self.noise_std_type = noise_std_type
        # init_noise_std 支持标量（三维共享）或 per-dim 列表/张量（每个动作维独立）。
        # per-dim 动机：nav 的 [vx,vy,wz] 量程差异大（如 vy 量程 0.3 vs wz 量程 2.0），
        # 共享 scalar=1.0 会让窄量程维（vy）噪声远超量程，从一开始就采样越界。
        if isinstance(init_noise_std, (list, tuple)):
            std_tensor = torch.tensor(init_noise_std, dtype=torch.float)
            assert std_tensor.numel() == num_actions, (
                f"init_noise_std list len {std_tensor.numel()} != num_actions {num_actions}"
            )
        else:
            std_tensor = float(init_noise_std) * torch.ones(num_actions)
        if noise_std_type == "scalar":
            self.std = nn.Parameter(std_tensor.clone())
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(std_tensor.clone()))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (set by update_distribution)
        # 动作分布（由update_distribution设置）
        self.distribution = None
        # Disable args validation for speedup
        # 禁用分布验证加速
        Normal.set_default_validate_args(False)

    @staticmethod
    def init_weights(sequential, scales):
        """
        Initialize weights using orthogonal initialization
        使用正交初始化方法初始化权重
        """
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """
        Reset hidden states for terminated episodes
        重置已终止episode的隐藏状态
        """
        pass

    def forward(self):
        """
        Forward pass (not implemented, use act/evaluate instead)
        前向传播（未实现，请使用act/evaluate方法）
        """
        raise NotImplementedError

    @property
    def action_mean(self):
        """
        Get mean of action distribution
        获取动作分布的均值
        """
        return self.distribution.mean

    @property
    def action_std(self):
        """
        Get standard deviation of action distribution
        获取动作分布的标准差
        """
        return self.distribution.stddev

    @property
    def entropy(self):
        """
        Get entropy of action distribution
        获取动作分布的熵
        """
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs: torch.Tensor):
        """
        Update action distribution based on observations
        基于观测更新动作分布

        Args:
            obs: [B, num_obs] flat actor observation tensor
            obs: [B, num_obs] Actor观测张量
        """
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.clamp(min=1e-6).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def act(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Sample actions from policy distribution
        从策略分布中采样动作

        Args:
            obs: [B, num_obs]
            obs: [B, num_obs] 观测张量

        Returns:
            actions: [B, num_actions]
            返回值：[B, num_actions] 动作张量
        """
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action (mean) for inference
        推理时的确定性动作（均值）

        Args:
            obs: [B, num_obs]
            obs: [B, num_obs] 观测张量

        Returns:
            actions: [B, num_actions]
            返回值：[B, num_actions] 动作张量
        """
        return self.actor(obs)

    def evaluate(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluate state value using critic network / 使用 critic 网络评估状态价值。"""
        if not self.has_critic:
            raise NotImplementedError(
                "Model was constructed with build_critic=False; no critic available."
            )
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of actions under current distribution
        计算动作在当前分布下的对数概率

        Args:
            actions: [B, num_actions]
            actions: [B, num_actions] 动作张量

        Returns:
            log_prob: [B]
            返回值：[B] 对数概率
        """
        return self.distribution.log_prob(actions).sum(dim=-1)
