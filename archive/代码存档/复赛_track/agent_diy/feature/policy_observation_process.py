# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PolicyObservationProcess — custom policy observation processor.
PolicyObservationProcess — 自定义 policy 观测处理器。

obs layout: [proprio(45) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2: num_goal_obs=0  → obs = proprio + scan = 301 dim
- Stage3:   num_goal_obs=4  → obs = proprio + scan + goal = 305 dim
观测布局：[proprio(45) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2：num_goal_obs=0  → obs = proprio + scan = 301 维
- Stage3：  num_goal_obs=4  → obs = proprio + scan + goal = 305 维
"""

from tools.base_env.observation_process import ObservationProcess


class PolicyObservationProcess(ObservationProcess):
    """Policy observation processor with height_scan and optional goal obs.

    带 height_scan 和可选 goal obs 的 policy 观测处理器。
    """

    target_group = "policy"

    def process(self):
        """Compute policy observation.

        计算 policy 观测。

        Stage1/2: proprio(45) + height_scan(256) = 301
        Stage3:   proprio(45) + height_scan(256) + goal(4) = 305
        """
        obs = self.default_observation()

        if self._get_num_goal_obs() > 0:
            goal_obs = self.goal_position_in_robot_frame()
            obs = self.concatenate_terms(obs, goal_obs)

        return obs
