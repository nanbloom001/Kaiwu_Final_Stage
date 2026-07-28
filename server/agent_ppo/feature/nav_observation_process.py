#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Nav 阶段观测 processors（policy 与 critic 必须一起生效）。

obs layout（权威定义见 feature/nav_contract.py）：

    policy: [proprio(45) | height_scan(256) | goal4(4) | depth(57600)] = 57905 D
            goal4 经 UWB 模拟测量链（NavGoalChain，Actor 只准看测量值）
    critic: [critic_proprio(60) | height_scan(256) | goal3(3)
             | nav_priv(4)] = 323 D
            goal3 为真值编码（特权，仅供 Oracle / 监控）
            nav_priv 为 nav_scanner 压缩墙体特征（特权，仅供 Oracle）

注意：
  - nav 阶段 worker 原生指令为死值（TOML: curriculum=false、resampling 300s、
    worker_progressive 关闭），实际执行指令由 NavScheduler 在 aisrv 侧写入
    obs[:,6:9] / critic_obs[:,9:12]——本文件不调用 worker command bridge。
  - 测量链状态驻留在 process 实例里（worker 侧、跨步持久）；env reset 由
    episode_length_buf == 0 检测并逐环境重置（评估模式走确定性名义链）。
  - 全部维度/偏移施加严格断言：装错在第一步就炸，不允许静默错位。
"""

import torch

from tools.base_env.observation_process import ObservationProcess

from agent_ppo.feature import nav_contract, nav_observation_utils, nav_probe
from agent_ppo.feature.goal_features import build_track_goal_raw, encode_track_goal
from agent_ppo.feature.nav_goal_encoder import NavGoalChain


def _detect_resets(env, chain: NavGoalChain) -> None:
    """episode_length_buf == 0 的环境视为刚 reset，重置其测量链状态。"""
    episode_length = getattr(env, "episode_length_buf", None)
    if episode_length is None:
        return
    reset_ids = (episode_length == 0).nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        chain.reset(reset_ids, deterministic=bool(getattr(env, "_is_eval", False)))


class NavPolicyObservationProcess(ObservationProcess):
    """Nav 阶段 policy 观测（57905 D）。"""

    target_group = "policy"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._goal_chain = None

    def process(self):
        env = self.env
        nav_probe.probe_once(env)

        obs = self.default_observation()  # [proprio(45) | height_scan(256)]
        expected_base = nav_contract.POLICY_PROPRIO_DIM + nav_contract.SCAN_DIM
        if obs.shape[-1] != expected_base:
            raise ValueError(
                "NavPolicyObservationProcess: default policy obs dim "
                f"{obs.shape[-1]} != {expected_base} (proprio45 + scan256)"
            )

        if self._goal_chain is None or self._goal_chain.num_envs != obs.shape[0]:
            self._goal_chain = NavGoalChain(obs.shape[0], obs.device)
        _detect_resets(env, self._goal_chain)
        step_dt = float(getattr(env, "step_dt", nav_contract.FRAME_DT_S))
        goal4 = self._goal_chain.update(env, step_dt)
        if goal4.shape != (obs.shape[0], 4):
            raise ValueError(
                f"NavPolicyObservationProcess: goal4 shape {tuple(goal4.shape)} != "
                f"({obs.shape[0]}, 4)"
            )

        depth = nav_observation_utils.depth_camera_image(env)  # (N, 57600)
        if depth.shape[-1] != nav_contract.DEPTH_DIM:
            raise ValueError(
                f"NavPolicyObservationProcess: depth dim {depth.shape[-1]} != "
                f"{nav_contract.DEPTH_DIM}"
            )

        full = self.concatenate_terms(
            self.concatenate_terms(obs, goal4.to(obs.dtype)), depth
        )
        if full.shape[-1] != nav_contract.POLICY_OBS_DIM:
            raise ValueError(
                f"NavPolicyObservationProcess: assembled policy obs dim "
                f"{full.shape[-1]} != {nav_contract.POLICY_OBS_DIM} "
                "(45 proprio | 256 scan | 4 goal4 | 57600 depth)"
            )
        return full


class NavCriticObservationProcess(ObservationProcess):
    """Nav 阶段 critic 观测（323 D，含真值 goal3/nav scanner 特权）。"""

    target_group = "critic"

    def process(self):
        env = self.env

        obs = self.default_observation()  # [critic_proprio(60) | height_scan(256)]
        expected_base = nav_contract.CRITIC_PROPRIO_DIM + nav_contract.SCAN_DIM
        if obs.shape[-1] != expected_base:
            raise ValueError(
                "NavCriticObservationProcess: default critic obs dim "
                f"{obs.shape[-1]} != {expected_base} (critic_proprio60 + scan256)"
            )

        local_xy = build_track_goal_raw(env)          # [N, 2] 真值体坐标（米）
        goal3 = encode_track_goal(local_xy)           # [N, 3] 编码（/10, /20, clamp）
        if goal3.shape != (obs.shape[0], 3):
            raise ValueError(
                f"NavCriticObservationProcess: goal3 shape {tuple(goal3.shape)} != "
                f"({obs.shape[0]}, 3)"
            )

        nav_priv = nav_observation_utils.nav_scanner_privileged_features(env)
        if nav_priv.shape != (obs.shape[0], nav_contract.CRITIC_NAV_PRIV_DIM):
            raise ValueError(
                "NavCriticObservationProcess: nav privilege shape "
                f"{tuple(nav_priv.shape)} != ({obs.shape[0]}, "
                f"{nav_contract.CRITIC_NAV_PRIV_DIM})"
            )

        full = self.concatenate_terms(
            self.concatenate_terms(obs, goal3.to(obs.dtype)),
            nav_priv.to(obs.dtype),
        )
        if full.shape[-1] != nav_contract.CRITIC_OBS_DIM:
            raise ValueError(
                f"NavCriticObservationProcess: assembled critic obs dim "
                f"{full.shape[-1]} != {nav_contract.CRITIC_OBS_DIM} "
                "(60 critic_proprio | 256 scan | 3 goal3 | 4 nav privilege)"
            )
        return full
