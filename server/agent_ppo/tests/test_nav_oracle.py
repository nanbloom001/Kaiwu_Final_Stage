#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""NavOracle 规则策略测试（只消费 critic_obs）。"""

import unittest

import torch

from agent_ppo.feature import nav_contract as nc
from agent_ppo.feature.nav_oracle import NavOracle


def _critic_obs(n=2, local_x=0.0, local_y=0.0, dist_m=0.0, front_height=0.0):
    obs = torch.zeros(n, nc.CRITIC_OBS_DIM)
    obs[:, nc.CRITIC_GOAL3_START + 0] = local_x / nc.GOAL_XY_SCALE_M
    obs[:, nc.CRITIC_GOAL3_START + 1] = local_y / nc.GOAL_XY_SCALE_M
    obs[:, nc.CRITIC_GOAL3_START + 2] = dist_m / nc.GOAL_DIST_SCALE_M
    if front_height:
        scan = obs[:, nc.CRITIC_SCAN_SLICE[0] : nc.CRITIC_SCAN_SLICE[1]].view(n, 16, 16)
        scan[:, 5, 8] = front_height  # 前方窗口内一点
    return obs


class TestNavOracle(unittest.TestCase):
    def setUp(self):
        self.oracle = NavOracle()

    def test_arrived_returns_zero(self):
        tokens = self.oracle.act(_critic_obs(local_x=0.3, dist_m=0.3))
        self.assertTrue(bool((tokens == 0).all()))

    def test_straight_speed_by_distance(self):
        self.assertTrue(
            bool((self.oracle.act(_critic_obs(local_x=1.0, dist_m=1.0)) == 1).all())
        )  # slow（近）
        self.assertTrue(
            bool((self.oracle.act(_critic_obs(local_x=2.0, dist_m=2.0)) == 2).all())
        )  # mid
        self.assertTrue(
            bool((self.oracle.act(_critic_obs(local_x=5.0, dist_m=5.0)) == 3).all())
        )  # fast

    def test_front_obstacle_forces_slow(self):
        tokens = self.oracle.act(
            _critic_obs(local_x=5.0, dist_m=5.0, front_height=0.3)
        )
        self.assertTrue(bool((tokens == 1).all()))

    def test_turning_bands(self):
        # 大角度 → 纯转向
        self.assertTrue(
            bool((self.oracle.act(_critic_obs(local_x=0.1, local_y=2.0, dist_m=2.0)) == 8).all())
        )
        self.assertTrue(
            bool((self.oracle.act(_critic_obs(local_x=0.1, local_y=-2.0, dist_m=2.0)) == 9).all())
        )
        # 中角度 → creep 转向
        self.assertTrue(
            bool((self.oracle.act(_critic_obs(local_x=2.0, local_y=1.2, dist_m=2.3)) == 6).all())
        )
        # 小角度 → forward 转向
        self.assertTrue(
            bool((self.oracle.act(_critic_obs(local_x=3.0, local_y=0.6, dist_m=3.1)) == 4).all())
        )

    def test_label_validity(self):
        valid = NavOracle.label_validity(_critic_obs(local_x=2.0, dist_m=2.0))
        self.assertTrue(bool(valid.all()))
        # goal 全零 → 无效
        invalid = NavOracle.label_validity(_critic_obs())
        self.assertFalse(bool(invalid.any()))
        # 非有限 → 无效
        obs = _critic_obs(local_x=2.0, dist_m=2.0)
        obs[0, 0] = float("nan")
        mixed = NavOracle.label_validity(obs)
        self.assertFalse(bool(mixed[0]))
        self.assertTrue(bool(mixed[1]))

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            self.oracle.act(torch.zeros(2, 100))


if __name__ == "__main__":
    unittest.main()
