#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Nav 观测 process 测试（stub ObservationProcess + mock env）。

严格 offset 断言：policy 57905、goal4 [301:305]、depth 起点 305、
critic 323、goal3 [316:319]、nav_priv [319:323]；resolver 双切换。
"""

import unittest
from types import SimpleNamespace

import agent_ppo.tests._nav_test_stubs  # noqa: F401  平台模块 stub，必须先于其他 agent_ppo import

import torch

from agent_ppo.feature import nav_contract as nc


class _Robot:
    def __init__(self, n):
        class _Data:
            pass

        self.data = _Data()
        self.data.root_pos_w = torch.zeros(n, 3)
        # 单位四元数（无旋转）
        quat = torch.zeros(n, 4)
        quat[:, 0] = 1.0
        self.data.root_quat_w = quat


class _Camera:
    def __init__(self, n, h=180, w=320):
        class _Data:
            pass

        self.data = _Data()
        self.data.output = {"depth": torch.full((n, h, w, 1), 2.5)}


class _RayScanner:
    def __init__(self, n, rays=143, pattern_cfg=None):
        class _Data:
            pass

        self.data = _Data()
        self.data.pos_w = torch.zeros(n, 3)
        self.data.ray_hits_w = torch.zeros(n, rays, 3)
        self.cfg = SimpleNamespace(pattern_cfg=pattern_cfg)


class _Scene:
    def __init__(self, n):
        self._robot = _Robot(n)
        self.sensors = {
            "depth_camera": _Camera(n),
            "nav_scanner": _RayScanner(n),
        }

    def __getitem__(self, key):
        if key == "robot":
            return self._robot
        raise KeyError(key)


class _Env:
    def __init__(self, n=2):
        self.num_envs = n
        self.device = torch.device("cpu")
        self.scene = _Scene(n)
        self.goal_positions = torch.tensor([[3.0, 0.0, 0.0]] * n)
        self.goal_yaw = torch.zeros(n)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self._is_eval = True  # 确定性名义测量链
        self.step_dt = 0.02
        self._depth_preprocess_conf = {}  # 绕过 Config 的 TOML 解析


def _mk_process(cls, env, default_dim):
    process = cls.__new__(cls)  # 绕过平台基类 __init__ 签名差异
    process.env = env
    process._default_obs = torch.zeros(env.num_envs, default_dim)
    process._goal_chain = None
    # stub 基类方法
    process.default_observation = lambda: process._default_obs
    process.concatenate_terms = lambda *terms: torch.cat(terms, dim=-1)
    return process


class TestNavObservationProcesses(unittest.TestCase):
    def test_policy_obs_layout(self):
        from agent_ppo.feature.nav_observation_process import (
            NavPolicyObservationProcess,
        )

        env = _Env()
        process = _mk_process(NavPolicyObservationProcess, env, 301)
        # 标记 proprio/scan 段，便于验证 goal4/depth 落位
        process._default_obs[:, 6:9] = 0.123
        obs = process.process()
        self.assertEqual(obs.shape, (2, nc.POLICY_OBS_DIM))
        goal4 = obs[:, nc.GOAL4_OBS_START : nc.GOAL4_OBS_END]
        # 真值 goal 在正前方 3m；确定性链首帧即采样
        self.assertTrue(bool((goal4[:, 0] > 0.0).all()))       # local_x/10 > 0
        self.assertAlmostEqual(float(goal4[0, 1]), 0.0, places=5)
        self.assertAlmostEqual(float(goal4[0, 2]), 3.0 / 20.0, places=4)
        self.assertAlmostEqual(float(goal4[0, 3]), 1.0, places=5)  # freshness
        depth = obs[:, nc.DEPTH_OBS_START :]
        self.assertEqual(depth.shape[-1], nc.DEPTH_DIM)
        # depth 归一化：2.5m / 5.0 = 0.5
        self.assertAlmostEqual(float(depth[0, 0]), 0.5, places=5)
        # proprio 标记原位
        self.assertAlmostEqual(float(obs[0, 6]), 0.123, places=5)

    def test_policy_obs_rejects_wrong_base_dim(self):
        from agent_ppo.feature.nav_observation_process import (
            NavPolicyObservationProcess,
        )

        env = _Env()
        process = _mk_process(NavPolicyObservationProcess, env, 300)
        with self.assertRaises(ValueError):
            process.process()

    def test_critic_obs_layout(self):
        from agent_ppo.feature.nav_observation_process import (
            NavCriticObservationProcess,
        )

        env = _Env()
        process = _mk_process(NavCriticObservationProcess, env, 316)
        obs = process.process()
        self.assertEqual(obs.shape, (2, nc.CRITIC_OBS_DIM))
        goal3 = obs[:, nc.CRITIC_GOAL3_START : nc.CRITIC_NAV_PRIV_START]
        # 真值编码：x=3m → 0.3；dist=3m → 0.15
        self.assertAlmostEqual(float(goal3[0, 0]), 0.3, places=5)
        self.assertAlmostEqual(float(goal3[0, 1]), 0.0, places=5)
        self.assertAlmostEqual(float(goal3[0, 2]), 0.15, places=5)
        nav_priv = obs[:, nc.CRITIC_NAV_PRIV_SLICE[0] : nc.CRITIC_NAV_PRIV_SLICE[1]]
        self.assertEqual(nav_priv.shape, (2, 4))
        self.assertTrue(bool((nav_priv[:, 0] == 1.0).all()))

    def test_critic_requires_nav_scanner(self):
        from agent_ppo.feature.nav_observation_process import (
            NavCriticObservationProcess,
        )

        env = _Env()
        del env.scene.sensors["nav_scanner"]
        process = _mk_process(NavCriticObservationProcess, env, 316)
        with self.assertRaises(RuntimeError):
            process.process()

    def test_critic_accepts_active_anisotropic_273_ray_scanner(self):
        from agent_ppo.feature.nav_observation_process import (
            NavCriticObservationProcess,
        )

        env = _Env()
        pattern = SimpleNamespace(
            size=(2.5, 2.0),
            resolution_x=0.2,
            resolution_y=0.1,
            resolution=None,
            ordering="xy",
        )
        env.scene.sensors["nav_scanner"] = _RayScanner(
            env.num_envs, rays=273, pattern_cfg=pattern
        )
        process = _mk_process(NavCriticObservationProcess, env, 316)
        obs = process.process()
        self.assertEqual(obs.shape, (env.num_envs, nc.CRITIC_OBS_DIM))
        self.assertTrue(bool((obs[:, nc.CRITIC_NAV_PRIV_START] == 1.0).all()))

    def test_resolvers_switch_policy_and_critic_together(self):
        import agent_ppo.feature as feature
        from agent_ppo.conf.conf import Config, NavDaggerConfig, StandardVisualPPOConfig

        saved = Config.CURRENT
        try:
            Config.CURRENT = NavDaggerConfig
            policy_cls = feature.PolicyObservationProcess
            critic_cls = feature.CriticObservationProcess
            self.assertEqual(policy_cls.__name__, "NavPolicyObservationProcess")
            self.assertEqual(critic_cls.__name__, "NavCriticObservationProcess")
            Config.CURRENT = StandardVisualPPOConfig
            self.assertNotEqual(
                feature.CriticObservationProcess.__name__,
                "NavCriticObservationProcess",
            )
        finally:
            Config.CURRENT = saved


if __name__ == "__main__":
    unittest.main()
