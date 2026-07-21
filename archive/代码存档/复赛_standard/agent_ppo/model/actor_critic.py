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
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialize ActorCritic
        初始化ActorCritic

        Args:
            num_obs: Dimension of actor observation
            num_obs: Actor观测维度
            num_critic_obs: Dimension of critic observation
            num_critic_obs: Critic观测维度
            num_actions: Number of action dimensions
            num_actions: 动作维度
            actor_hidden_dims: Hidden layer sizes for actor MLP
            actor_hidden_dims: Actor MLP隐藏层大小
            critic_hidden_dims: Hidden layer sizes for critic MLP
            critic_hidden_dims: Critic MLP隐藏层大小
            activation: Activation function name
            activation: 激活函数名称
            init_noise_std: Initial noise std for exploration
            init_noise_std: 探索噪声初始标准差
            noise_std_type: "scalar" or "log"
            noise_std_type: 标准差类型，"scalar"或"log"
        """
        super().__init__()

        activation_fn = resolve_nn_activation(activation)

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

        # Build critic MLP (with LayerNorm)
        # 构建价值网络（含层标准化）
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

        # Action noise initialization
        # 动作噪声初始化
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
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
        """
        Evaluate state value using critic network
        使用critic网络评估状态价值

        Args:
            critic_obs: [B, num_critic_obs]
            critic_obs: [B, num_critic_obs] Critic观测张量

        Returns:
            values: [B, 1]
            返回值：[B, 1] 状态价值
        """
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
