#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""NavOracle — 特权规则策略（DAgger 教师），只消费 critic_obs。

输入：critic_obs [N, 319] = [critic_proprio(60) | height_scan(256) | goal3(3)]
  - 真值 goal3（critic[316:319]，无噪声、无 S&H）→ 方位角与距离
  - 特权 height_scan（critic[60:316]，16×16 近场网格）→ 前方障碍减速
  - 真值 lin_vel（critic[0:3]）暂未使用（保留接口）

输出：词表 token [N] long（`nav_contract.VOCAB` 编号）。

规则（v1，全部阈值可由构造参数覆盖）：
  1. dist < stop_radius            → zero（到达驻停）
  2. |angle| > spin_threshold      → 纯转向 spin_left/right
  3. |angle| > turn_threshold      → 慢速转向 creep_left/right
  4. |angle| > veer_threshold      → 前进转向 forward_left/right
  5. 直行：前方近场障碍高 or dist 小 → forward_slow；
     dist < mid_range → forward_mid；否则 forward_fast

Oracle 是特权训练工具：真值/height_scan 只在训练侧存在，学生（HighLevelPolicy）
永远看不到它们——学生输入是 48 维真实可测契约（nav_contract）。
"""

from __future__ import annotations

import torch

from agent_ppo.feature import nav_contract


class NavOracle:
    def __init__(
        self,
        stop_radius_m: float = 0.5,
        spin_threshold_rad: float = 0.9,
        turn_threshold_rad: float = 0.35,
        veer_threshold_rad: float = 0.12,
        slow_range_m: float = 1.5,
        mid_range_m: float = 3.0,
        obstacle_threshold_m: float = 0.3,
    ):
        self.stop_radius_m = stop_radius_m
        self.spin_threshold_rad = spin_threshold_rad
        self.turn_threshold_rad = turn_threshold_rad
        self.veer_threshold_rad = veer_threshold_rad
        self.slow_range_m = slow_range_m
        self.mid_range_m = mid_range_m
        self.obstacle_threshold_m = obstacle_threshold_m

    def act(self, critic_obs: torch.Tensor) -> torch.Tensor:
        if critic_obs.ndim != 2 or critic_obs.shape[1] < nav_contract.CRITIC_OBS_DIM:
            raise ValueError(
                f"NavOracle expects critic obs [N, >={nav_contract.CRITIC_OBS_DIM}], "
                f"got {tuple(critic_obs.shape)}"
            )
        n = critic_obs.shape[0]
        device = critic_obs.device

        g_lo = nav_contract.CRITIC_GOAL3_START
        goal3 = critic_obs[:, g_lo : g_lo + 3]
        # goal3 编码为 [x/10, y/10, dist/20]；atan2 对同比缩放不变
        local_x = goal3[:, 0]
        local_y = goal3[:, 1]
        dist_m = goal3[:, 2] * nav_contract.GOAL_DIST_SCALE_M
        angle = torch.atan2(local_y, local_x)

        s_lo, s_hi = nav_contract.CRITIC_SCAN_SLICE
        scan = critic_obs[:, s_lo:s_hi].view(n, 16, 16)
        # 前方近场窗口——与 reward_process.py:321-322 的取窗惯例严格一致：
        # grid.view(N,16,16)[:, y, x]，dim1=y（机身横向，取中间 3..13），
        # dim2=x（机身前向，取前 10）。判据同样沿用仓内先例的带符号形式
        # （reward_process.py:323 用 `< -0.3`：击中点显著低于基线 = 障碍/墙），
        # 绝对量纲基线由平台 env cfg 决定，默认阈值待 S0a 探针核验后校准。
        front_window = scan[:, 3:13, :10]
        front_obstacle = front_window.amin(dim=(1, 2)) < -self.obstacle_threshold_m

        tokens = torch.full((n,), nav_contract.ZERO_TOKEN_INDEX, dtype=torch.long, device=device)

        arrived = dist_m < self.stop_radius_m
        spin = ~arrived & (angle.abs() > self.spin_threshold_rad)
        creep = ~arrived & ~spin & (angle.abs() > self.turn_threshold_rad)
        veer = ~arrived & ~spin & ~creep & (angle.abs() > self.veer_threshold_rad)
        straight = ~arrived & ~spin & ~creep & ~veer

        left = angle > 0

        tokens[spin & left] = 8       # spin_left
        tokens[spin & ~left] = 9      # spin_right
        tokens[creep & left] = 6      # creep_left
        tokens[creep & ~left] = 7     # creep_right
        tokens[veer & left] = 4       # forward_left
        tokens[veer & ~left] = 5      # forward_right

        slow = straight & (front_obstacle | (dist_m < self.slow_range_m))
        mid = straight & ~slow & (dist_m < self.mid_range_m)
        fast = straight & ~slow & ~mid
        tokens[slow] = 1              # forward_slow
        tokens[mid] = 2               # forward_mid
        tokens[fast] = 3              # forward_fast

        # 有效性：goal3 全零（真值缺席）视为无效标签 → zero + 调用方置 valid=0
        return tokens

    @staticmethod
    def label_validity(critic_obs: torch.Tensor) -> torch.Tensor:
        """[N] bool：Oracle 标签是否有效（真值 goal 存在且输入有限）。"""
        g_lo = nav_contract.CRITIC_GOAL3_START
        goal3 = critic_obs[:, g_lo : g_lo + 3]
        finite = torch.isfinite(critic_obs).all(dim=-1)
        goal_present = goal3.abs().sum(dim=-1) > 1.0e-8
        return finite & goal_present
