#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

critic obs layout: [critic_proprio(60) | height_scan(256)] → 316 dim
critic 观测布局：[critic_proprio(60) | height_scan(256)] → 316 维

agent_diy only carries LocomotionConfig (standard locomotion), critic obs does not
include nav/goal terms; for track terrain extension, refer to
agent_ppo/feature/nav_observation_utils.py and compose manually.
agent_diy 仅承载 LocomotionConfig（standard locomotion），critic obs 不含 nav/goal
项；如需 track 地形扩展，参考 agent_ppo/feature/nav_observation_utils.py 自行拼接。
"""

from tools.base_env.observation_process import ObservationProcess


class CriticObservationProcess(ObservationProcess):
    """Critic observation processor.

    与 Isaac Lab CriticCfg 对齐的 critic 观测处理器。
    """

    target_group = "critic"

    def process(self):
        """Compute critic observation.

        计算 critic 观测。

        critic_obs = 316
        """
        return self.default_observation()
