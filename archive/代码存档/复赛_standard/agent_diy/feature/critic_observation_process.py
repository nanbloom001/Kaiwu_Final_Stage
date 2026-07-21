# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

critic obs layout: [critic_proprio(60) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2: num_goal_obs=0  → critic_obs = 316 dim
- Stage3:   num_goal_obs=4  → critic_obs = 320 dim
critic 观测布局：[critic_proprio(60) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2：num_goal_obs=0  → critic_obs = 316 维
- Stage3：  num_goal_obs=4  → critic_obs = 320 维
"""

from tools.base_env.observation_process import ObservationProcess


class CriticObservationProcess(ObservationProcess):
    """Critic observation processor with optional goal obs.

    与 Isaac Lab CriticCfg 对齐的 critic 观测处理器，可选拼接 goal obs。
    """

    target_group = "critic"

    def process(self):
        """Compute critic observation.

        计算 critic 观测。

        Stage1/2: critic_obs = 316
        Stage3:   critic_obs = 316 + goal(4) = 320
        Stage1/2：critic_obs = 316
        Stage3：  critic_obs = 316 + goal(4) = 320
        """
        obs = self.default_observation()

        if self._get_num_goal_obs() > 0:
            goal_obs = self.goal_position_in_robot_frame()
            obs = self.concatenate_terms(obs, goal_obs)

        return obs
