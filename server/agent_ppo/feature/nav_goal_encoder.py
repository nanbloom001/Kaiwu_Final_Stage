#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""NavGoalChain — UWB 模拟测量链（真值 → 带噪 goal4，Actor 专用）。

Actor 输入的 goal4 必须经过本链；critic/Oracle 直接用真值（特权合法）。
链路（每 50Hz 帧调用 ``update``；常数全部来自 nav_contract 的成对常数）：

    真值体坐标 local_xy（goal_features.build_track_goal_raw，仅此一步用特权）
      → (bearing, planar_dist) 分解
      → 每 episode 采样的 bearing/distance 偏置 + 每次测量的高斯噪声
        （IMU heading 漂移等效折算进 bearing 噪声；pitch 分量折算进 distance
         噪声——真机 beta/pitch/distance 三元组的 planar 投影等效，配对
         校准时如需拆分再细化）
      → 低通滤波（time_filter 语义：alpha = 1 - exp(-age/tau)）
      → 等效 UWB 率 S&H（每 episode 采样 3-8Hz；样本间保持上次值）
      → 丢帧段（泊松到达 × 均匀时长；期间不更新测量、freshness 衰减）
      → freshness = uwb_freshness_scale(age)（stale 前为 1，hold 时衰减到 0）
      → goal4 = [clamp(x/10), clamp(y/10), clamp(dist/20), freshness]

到达与失效的输入空间分离：到达 = dist 小且 freshness 高；失效 = freshness
低（xy/dist 冻结在上次有效值，不清零）。freshness 随机化自首个训练任务生效
——未经历该分布的模型不得进入真机接管。

评估模式（env._is_eval）：确定性名义链——固定 5Hz、无噪声/偏置/丢帧，
滤波与 freshness 照常（保证评估可复现，且输入语义与训练同构）。
"""

from __future__ import annotations

import math

import torch

from agent_ppo.feature import nav_contract
from agent_ppo.feature.goal_features import build_track_goal_raw


class NavGoalChain:
    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = num_envs
        self.device = torch.device(device)

        n = num_envs
        dev = self.device
        # 每 episode 常量
        self.period_s = torch.full((n,), 0.2, device=dev)
        self.bearing_bias = torch.zeros(n, device=dev)
        self.distance_bias = torch.zeros(n, device=dev)
        # 运行状态
        self.age_s = torch.full((n,), 1.0e6, device=dev)     # 巨大年龄 → 首帧立即采样
        self.dropout_remaining_s = torch.zeros(n, device=dev)
        self.filtered_xy = torch.zeros(n, 2, device=dev)
        self.has_sample = torch.zeros(n, dtype=torch.bool, device=dev)

        self._resample_episode_params(torch.arange(n, device=dev), deterministic=False)

    # ------------------------------------------------------------------

    def _resample_episode_params(self, env_ids: torch.Tensor, deterministic: bool) -> None:
        if env_ids.numel() == 0:
            return
        if deterministic:
            self.period_s[env_ids] = 1.0 / 5.0
            self.bearing_bias[env_ids] = 0.0
            self.distance_bias[env_ids] = 0.0
            return
        lo, hi = nav_contract.UWB_RATE_RANGE_HZ
        rate = lo + (hi - lo) * torch.rand(env_ids.numel(), device=self.device)
        self.period_s[env_ids] = 1.0 / rate
        bearing_bias = (
            torch.randn(env_ids.numel(), device=self.device)
            * nav_contract.UWB_BEARING_BIAS_STD_RAD
        ).clamp(
            -nav_contract.UWB_BEARING_BIAS_CLIP_RAD, nav_contract.UWB_BEARING_BIAS_CLIP_RAD
        )
        distance_bias = (
            torch.randn(env_ids.numel(), device=self.device)
            * nav_contract.UWB_DISTANCE_BIAS_STD_M
        ).clamp(
            -nav_contract.UWB_DISTANCE_BIAS_CLIP_M, nav_contract.UWB_DISTANCE_BIAS_CLIP_M
        )
        self.bearing_bias[env_ids] = bearing_bias
        self.distance_bias[env_ids] = distance_bias

    def reset(self, env_ids: torch.Tensor, deterministic: bool = False) -> None:
        if env_ids.numel() == 0:
            return
        self.age_s[env_ids] = 1.0e6
        self.dropout_remaining_s[env_ids] = 0.0
        self.filtered_xy[env_ids] = 0.0
        self.has_sample[env_ids] = False
        self._resample_episode_params(env_ids, deterministic)

    # ------------------------------------------------------------------

    def update(self, env, dt_s: float) -> torch.Tensor:
        """推进一帧测量链并返回 goal4 [N, 4]。"""

        is_eval = bool(getattr(env, "_is_eval", False))
        dev = self.device

        self.age_s += dt_s
        self.dropout_remaining_s = (self.dropout_remaining_s - dt_s).clamp_min(0.0)

        # 丢帧段开始（训练模式）：逐帧泊松近似
        if not is_eval:
            start_prob = nav_contract.UWB_DROPOUT_RATE_PER_S * dt_s
            starts = (
                (torch.rand(self.num_envs, device=dev) < start_prob)
                & (self.dropout_remaining_s <= 0.0)
            )
            if bool(starts.any()):
                d_lo, d_hi = nav_contract.UWB_DROPOUT_DURATION_S
                duration = d_lo + (d_hi - d_lo) * torch.rand(
                    int(starts.sum().item()), device=dev
                )
                self.dropout_remaining_s[starts] = duration

        true_xy = build_track_goal_raw(env)  # [N, 2]，真值缺席时为全零
        goal_available = bool(getattr(env, "goal_positions", None) is not None)

        due = (self.age_s >= self.period_s) & (self.dropout_remaining_s <= 0.0)
        if goal_available and bool(due.any()):
            ids = due.nonzero(as_tuple=False).squeeze(-1)
            xy = true_xy[ids]
            bearing = torch.atan2(xy[:, 1], xy[:, 0])
            dist = torch.linalg.norm(xy, dim=1)
            if not is_eval:
                bearing = (
                    bearing
                    + self.bearing_bias[ids]
                    + torch.randn_like(bearing)
                    * math.hypot(
                        nav_contract.UWB_BEARING_NOISE_STD_RAD,
                        nav_contract.UWB_HEADING_NOISE_STD_RAD,
                    )
                )
                dist = (
                    dist
                    + self.distance_bias[ids]
                    + torch.randn_like(dist) * nav_contract.UWB_DISTANCE_NOISE_STD_M
                ).clamp_min(0.0)
            measured_xy = torch.stack(
                (dist * torch.cos(bearing), dist * torch.sin(bearing)), dim=1
            )
            # 低通滤波：alpha = 1 - exp(-age/tau)，age 为距上次有效样本的间隔
            alpha = 1.0 - torch.exp(
                -self.age_s[ids].clamp(max=10.0) / nav_contract.UWB_FILTER_TAU_S
            )
            first = ~self.has_sample[ids]
            alpha = torch.where(first, torch.ones_like(alpha), alpha)
            self.filtered_xy[ids] = (
                self.filtered_xy[ids] + alpha.unsqueeze(-1) * (measured_xy - self.filtered_xy[ids])
            )
            self.has_sample[ids] = True
            self.age_s[ids] = 0.0

        return self.goal4()

    def goal4(self) -> torch.Tensor:
        """当前 goal4 输出 [N, 4]（S&H：样本间保持上次滤波值）。"""

        xy = self.filtered_xy
        dist = torch.linalg.norm(xy, dim=1)
        encoded_xy = torch.clamp(xy / nav_contract.GOAL_XY_SCALE_M, -1.0, 1.0)
        encoded_dist = torch.clamp(dist / nav_contract.GOAL_DIST_SCALE_M, 0.0, 1.0)

        stale = nav_contract.UWB_STALE_TIMEOUT_S
        hold = nav_contract.UWB_HOLD_TIMEOUT_S
        freshness = ((hold - self.age_s) / (hold - stale)).clamp(0.0, 1.0)
        freshness = torch.where(self.age_s <= stale, torch.ones_like(freshness), freshness)
        freshness = torch.where(self.has_sample, freshness, torch.zeros_like(freshness))

        goal4 = torch.cat(
            (
                encoded_xy,
                encoded_dist.unsqueeze(-1),
                freshness.unsqueeze(-1),
            ),
            dim=1,
        )
        # 无样本（真值缺席 / episode 刚开始且未采到）→ 全零 + freshness 0
        goal4[~self.has_sample] = 0.0
        return goal4
