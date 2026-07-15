#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
ActorCriticEncoder — Asymmetric actor-critic with scan-based terrain encoder.

Actor 始终走 encoder（部署时 LBC 蒸馏的目标）：
    actor: scan(原始) → actor_encoder → latent_a(32) → cat(proprio, latent_a) → π

Critic 有两种模式（由 critic_use_encoder 切换）：
  (a) critic_use_encoder=False：critic 直接吃 raw critic_obs（按 drop_slice 丢弃无关段）
        critic: critic_obs - drop → critic MLP → V
        优点：信息最全；缺点：高维 scan 直入易过拟合 + 收敛慢

  (b) critic_use_encoder=True（LocomotionConfig 采用）：
        critic 用独立的 critic_encoder（与 actor encoder 参数分离）
        critic: cat(critic_obs - drop - scan, critic_encoder(scan)) → critic MLP → V
        优点：scan 段先压缩到 latent，正则化效果显著；
              critic_encoder 与 actor_encoder 参数独立，梯度互不干扰；
              不影响 LBC 蒸馏（蒸馏的是 actor encoder）。

Stage 维度参考（LocomotionConfig, standard 地形）:
    policy obs : [proprio(45) | h_scan(256)]        = 301
    critic obs : [c_proprio(60) | h_scan(256)]      = 316
    actor input  : 45 + 32 = 77
    critic input (use_encoder=True):  60 + 32 = 92
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from agent_ppo.model.actor_critic import ActorCritic, resolve_nn_activation


class L2Norm(nn.Module):
    """L2 normalization layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2.0, dim=-1)


def _build_encoder(
    num_scan: int,
    encoder_hidden_dims: list[int] | tuple[int, ...],
    latent_dim: int,
    activation_fn: nn.Module,
) -> nn.Sequential:
    """Build a scan→latent MLP encoder with L2Norm output."""
    layers: list[nn.Module] = []
    layers.append(nn.Linear(num_scan, encoder_hidden_dims[0]))
    layers.append(activation_fn)
    for i in range(len(encoder_hidden_dims) - 1):
        layers.append(nn.Linear(encoder_hidden_dims[i], encoder_hidden_dims[i + 1]))
        layers.append(activation_fn)
    layers.append(nn.Linear(encoder_hidden_dims[-1], latent_dim))
    layers.append(L2Norm())
    return nn.Sequential(*layers)


class ActorCriticEncoder(ActorCritic):
    """Asymmetric actor-critic with optional independent critic encoder.

    Args:
        num_proprio:  proprio dim in policy obs (always at [0:num_proprio]).
        num_scan:     scan dim that BOTH actor encoder and (optional) critic encoder consume.
        num_critic_input: critic MLP first-layer input dim. 调用方负责按下面公式计算：
            critic_use_encoder=False:
                num_critic_input = critic_obs_total - drop_slice_len
            critic_use_encoder=True:
                num_critic_input = critic_obs_total - drop_slice_len - num_scan + latent_dim
        num_actions:  action dim.
        num_goal_obs: goal dim (拼接到 actor 输入末尾，从 policy obs 末尾切出)。
        scan_offset:  scan 在 policy obs 中的起点。默认 = num_proprio（Stage1/2）。
        critic_drop_slice: (start, end) 半开区间，从 critic_obs 丢弃的列段。
                           例如 nav_critic 想丢 height_scan。None 表示不丢。
        critic_use_encoder: 是否给 critic 配独立 encoder（与 actor encoder 参数分离）。
                            默认 False（critic 吃 raw critic_obs）。
        critic_scan_slice: (start, end) 半开区间，critic_obs 中 scan 段的位置。
                           critic_use_encoder=True 时必填；==False 时忽略。
                           注意：critic_scan_slice 不能与 critic_drop_slice 重叠。
        build_critic: 是否构建 critic 网络。
                      hier_nav 的 frozen loco_model 不参与 PPO update，可设 False。
    """

    def __init__(
        self,
        num_proprio: int,
        num_scan: int,
        num_critic_input: int,
        num_actions: int,
        num_goal_obs: int = 0,
        scan_offset: int | None = None,
        critic_drop_slice: tuple[int, int] | None = None,
        critic_use_encoder: bool = False,
        critic_scan_slice: tuple[int, int] | None = None,
        build_critic: bool = True,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        latent_dim: int = 32,
        actor_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        critic_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float | list | tuple = 1.0,
        noise_std_type: str = "scalar",
        actor_last_layer_init_gain: float | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        actor_input_dim = num_proprio + latent_dim + num_goal_obs

        # build_critic 透传给基类：False 时基类不创建 critic，self.critic = None
        super().__init__(
            num_obs=actor_input_dim,
            num_critic_obs=num_critic_input,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            build_critic=build_critic,
            actor_last_layer_init_gain=actor_last_layer_init_gain,
        )

        # Store split dims
        self.num_proprio = num_proprio
        self.num_scan = num_scan
        self.num_goal_obs = num_goal_obs
        self.latent_dim = latent_dim

        # Policy obs scan offset (default = num_proprio for Stage1/2)
        self.scan_offset = scan_offset if scan_offset is not None else num_proprio

        # Critic slice configuration
        # Critic 切片配置
        if critic_drop_slice is not None:
            s, e = critic_drop_slice
            assert 0 <= s < e, f"Invalid critic_drop_slice={critic_drop_slice}"
        self.critic_drop_slice = critic_drop_slice

        self.critic_use_encoder = critic_use_encoder and build_critic
        if self.critic_use_encoder:
            assert critic_scan_slice is not None, (
                "critic_use_encoder=True requires critic_scan_slice=(start, end)"
            )
            s, e = critic_scan_slice
            assert 0 <= s < e, f"Invalid critic_scan_slice={critic_scan_slice}"
            assert (e - s) == num_scan, (
                f"critic_scan_slice length {e - s} != num_scan {num_scan}"
            )
            # 不允许与 drop_slice 重叠
            if critic_drop_slice is not None:
                ds, de = critic_drop_slice
                assert e <= ds or de <= s, (
                    f"critic_scan_slice {critic_scan_slice} overlaps "
                    f"critic_drop_slice {critic_drop_slice}"
                )
        self.critic_scan_slice = critic_scan_slice if self.critic_use_encoder else None

        # Actor encoder (always built, actor must go through encoder)
        # Actor encoder（始终构建，actor 必须走 encoder）
        activation_fn = resolve_nn_activation(activation)
        self.encoder = _build_encoder(num_scan, encoder_hidden_dims, latent_dim, activation_fn)

        # Critic encoder (optional, independent from actor encoder)
        # Critic encoder（可选，参数独立于 actor encoder）
        if self.critic_use_encoder:
            self.critic_encoder = _build_encoder(
                num_scan, encoder_hidden_dims, latent_dim, activation_fn
            )
        else:
            self.critic_encoder = None

    # Policy (actor) path: 走 actor encoder
    # 策略（actor）路径：走 actor encoder
    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Split policy obs and produce actor input = cat(proprio, latent, [goal])."""
        expected_dim = self.scan_offset + self.num_scan + self.num_goal_obs
        if obs.shape[-1] < expected_dim:
            raise ValueError(
                f"Policy observation too short: expected at least {expected_dim}, "
                f"got {obs.shape[-1]}."
            )
        proprio = obs[:, : self.num_proprio]
        scan = obs[:, self.scan_offset : self.scan_offset + self.num_scan]
        latent = self.encoder(scan)

        if self.num_goal_obs > 0:
            goal = obs[:, -self.num_goal_obs :]
            return torch.cat([proprio, latent, goal], dim=-1)
        return torch.cat([proprio, latent], dim=-1)

    def update_distribution(self, obs: torch.Tensor):
        actor_obs = self._encode_obs(obs)
        super().update_distribution(actor_obs)

    def act(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs = self._encode_obs(obs)
        return self.actor(actor_obs)

    def get_encoder_latent(self, obs: torch.Tensor) -> torch.Tensor:
        """Return actor_encoder(scan) latent (for teacher-student distillation)."""
        scan = obs[:, self.scan_offset : self.scan_offset + self.num_scan]
        return self.encoder(scan)

    # Critic path
    # Critic 路径
    def _build_critic_input(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """根据 critic_drop_slice / critic_scan_slice 配置组装 critic 输入。

        critic_use_encoder=False:
            cat([critic_obs[:, :ds], critic_obs[:, de:]])    (drop_slice or pass-through)

        critic_use_encoder=True:
            将 critic_obs 中 scan 段抽出走 critic_encoder，其他段（去掉 drop_slice）拼接 latent。
        """
        if not self.critic_use_encoder:
            # 仅按 drop_slice 切除
            if self.critic_drop_slice is None:
                return critic_obs
            ds, de = self.critic_drop_slice
            return torch.cat([critic_obs[:, :ds], critic_obs[:, de:]], dim=-1)

        # critic_use_encoder=True：抽 scan、丢 drop、拼 latent
        ss, se = self.critic_scan_slice  # type: ignore[misc]
        expected_dim = se + self.num_goal_obs
        if critic_obs.shape[-1] < expected_dim:
            raise ValueError(
                f"Critic observation too short: expected at least {expected_dim}, "
                f"got {critic_obs.shape[-1]}."
            )
        scan = critic_obs[:, ss:se]
        latent = self.critic_encoder(scan)  # type: ignore[misc]

        # "非 scan 非 drop" 的段：把 critic_obs 减去两个区间
        # 用合并的不要区间表达：sorted [scan_slice, drop_slice]
        bad_intervals = [self.critic_scan_slice]
        if self.critic_drop_slice is not None:
            bad_intervals.append(self.critic_drop_slice)
        bad_intervals = sorted(bad_intervals, key=lambda x: x[0])  # type: ignore[arg-type]

        keep_parts = []
        cur = 0
        total = critic_obs.shape[-1]
        for s, e in bad_intervals:
            if cur < s:
                keep_parts.append(critic_obs[:, cur:s])
            cur = max(cur, e)
        if cur < total:
            keep_parts.append(critic_obs[:, cur:total])

        keep_parts.append(latent)
        return torch.cat(keep_parts, dim=-1)

    def evaluate(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self.has_critic:
            return super().evaluate(critic_obs, **kwargs)
        return self.critic(self._build_critic_input(critic_obs))
