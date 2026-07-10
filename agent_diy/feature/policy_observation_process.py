#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PolicyObservationProcess — custom policy observation processor.
PolicyObservationProcess — 自定义 policy 观测处理器。

obs layout: [proprio(45) | height_scan(256)] → 301 dim
观测布局：[proprio(45) | height_scan(256)] → 301 维

agent_diy only carries LocomotionConfig (standard locomotion), obs does not include
nav/goal terms; for track terrain extension, refer to
agent_ppo/feature/nav_observation_utils.py and compose manually.
agent_diy 仅承载 LocomotionConfig（standard locomotion），obs 不含 nav/goal
项；如需 track 地形扩展，参考 agent_ppo/feature/nav_observation_utils.py 自行拼接。
"""

from tools.base_env.observation_process import ObservationProcess


class PolicyObservationProcess(ObservationProcess):
    """Policy observation processor with height_scan.

    带 height_scan 的 policy 观测处理器。
    """

    target_group = "policy"

    def process(self):
        """Compute policy observation.

        计算 policy 观测。

        proprio(45) + height_scan(256) = 301
        """
        return self.default_observation()
