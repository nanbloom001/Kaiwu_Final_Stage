#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls, Frame
import torch
import numpy as np
from agent_ppo.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls(
    "ActData",
    action=None,
)


class RolloutStorage:
    """
    Experience replay buffer for PPO algorithm.
    PPO 算法经验回放缓冲区。

    V3: no history buffer — obs contains proprio + height_scan directly.
    V3：无 history 缓冲区 —— obs 直接包含 proprio + height_scan。
    """

    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        device="cpu",
    ):
        self.device = device
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self._init_buffers(
            num_transitions_per_env,
            num_envs,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
        )

        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        self.step = 0

    def _init_buffers(
        self,
        num_transitions_per_env,
        num_envs,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
    ):
        """
        Initialize all tensor buffers.
        初始化所有张量缓冲区。
        """
        shape = (num_transitions_per_env, num_envs)

        # Observation buffers
        # 观测缓冲区
        self.observations = torch.zeros(*shape, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(*shape, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None

        # Action-related buffers
        # 动作相关缓冲区
        self.actions = torch.zeros(*shape, *actions_shape, device=self.device)
        self.actions_log_prob = torch.zeros(*shape, 1, device=self.device)
        self.mu = torch.zeros(*shape, *actions_shape, device=self.device)
        self.sigma = torch.zeros(*shape, *actions_shape, device=self.device)

        # Reward and value buffers
        # 奖励和价值缓冲区
        self.rewards = torch.zeros(*shape, 1, device=self.device)
        self.values = torch.zeros(*shape, 1, device=self.device)
        self.returns = torch.zeros(*shape, 1, device=self.device)
        self.advantages = torch.zeros(*shape, 1, device=self.device)

        # Done flags
        # 完成标志
        self.dones = torch.zeros(*shape, 1, device=self.device).byte()

    def add_transitions(self, transition):
        """
        Add a transition to the rollout buffer.
        向 rollout 缓冲区添加一个转移。
        """
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        """
        Save RNN hidden states.
        保存 RNN 隐藏状态。
        """
        if hidden_states is None or hidden_states == (None, None):
            return
        hid_a, hid_c = self._normalize_hidden_states(hidden_states)
        if self.saved_hidden_states_a is None:
            self._init_hidden_state_storage(hid_a, hid_c)
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def _normalize_hidden_states(self, hidden_states):
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        return hid_a, hid_c

    def _init_hidden_state_storage(self, hid_a, hid_c):
        self.saved_hidden_states_a = [
            torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
        ]
        self.saved_hidden_states_c = [
            torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
        ]

    def clear(self):
        """Reset buffer pointer.

        重置缓冲区指针。
        """
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """
        Calculate returns and advantages using GAE.
        使用 GAE 方法计算回报和优势函数。
        """
        # Sanitize inputs before backward GAE pass
        # GAE 反向传递前清洗输入数据
        last_values = torch.nan_to_num(last_values, nan=0.0, posinf=0.0, neginf=0.0)
        self.values.copy_(torch.nan_to_num(self.values, nan=0.0, posinf=0.0, neginf=0.0))
        self.rewards.copy_(torch.nan_to_num(self.rewards, nan=0.0, posinf=0.0, neginf=0.0))

        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Sanitize returns against NaN/Inf (extreme rewards or bad values)
        # 清洗 returns 中可能出现的 NaN/Inf（极端奖励或坏价值估计）
        self.returns.copy_(torch.nan_to_num(self.returns, nan=0.0, posinf=0.0, neginf=0.0))

        # Compute and normalize advantages
        # 计算并标准化优势函数
        self.advantages = self.returns - self.values
        adv_std = self.advantages.std()
        if not torch.isfinite(adv_std) or adv_std < 1e-6:
            # Variance is too small or invalid -> skip normalization, only subtract the
            # mean, to avoid division by zero that would produce NaN.
            # 方差过小或非法 → 跳过归一化，只去均值，避免除 0 产生 NaN
            adv_std = torch.ones_like(adv_std)
        self.advantages = (self.advantages - self.advantages.mean()) / (adv_std + 1e-8)
        self.advantages = torch.nan_to_num(self.advantages, nan=0.0, posinf=0.0, neginf=0.0)

    def get_statistics(self):
        """Get trajectory statistics.

        获取轨迹统计信息。
        """
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        Generate mini-batches for training.
        生成训练用的小批量数据。
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        flattened_data = self._flatten_buffers()

        for epoch in range(num_epochs):
            indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield self._create_mini_batch(flattened_data, batch_idx)

    def _flatten_buffers(self):
        """Flatten all buffer tensors for batch generation.

        将所有缓冲区张量展平以用于生成 batch。
        """
        observations = self.observations.flatten(0, 1)
        critic_observations = (
            self.privileged_observations.flatten(0, 1) if self.privileged_observations is not None else observations
        )
        return {
            "observations": observations,
            "critic_observations": critic_observations,
            "actions": self.actions.flatten(0, 1),
            "values": self.values.flatten(0, 1),
            "returns": self.returns.flatten(0, 1),
            "old_actions_log_prob": self.actions_log_prob.flatten(0, 1),
            "advantages": self.advantages.flatten(0, 1),
            "old_mu": self.mu.flatten(0, 1),
            "old_sigma": self.sigma.flatten(0, 1),
        }

    def _create_mini_batch(self, flattened_data, batch_idx):
        """Create a mini-batch from flattened data using batch indices.

        根据 batch_idx 从展平数据中创建一个 mini-batch。
        """
        return (
            flattened_data["observations"][batch_idx],
            flattened_data["critic_observations"][batch_idx],
            flattened_data["actions"][batch_idx],
            flattened_data["values"][batch_idx],
            flattened_data["advantages"][batch_idx],
            flattened_data["returns"][batch_idx],
            flattened_data["old_actions_log_prob"][batch_idx],
            flattened_data["old_mu"][batch_idx],
            flattened_data["old_sigma"][batch_idx],
            # hidden states placeholder
            # 隐藏状态占位符
            (None, None),
            # masks placeholder
            # 掩码占位符
            None,
        )
