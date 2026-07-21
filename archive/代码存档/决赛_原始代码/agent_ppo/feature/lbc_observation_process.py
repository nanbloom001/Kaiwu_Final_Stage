# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
LBCObservationProcess — LBC 蒸馏阶段的 policy observation processor。

obs layout:
    [proprio(45) | height_scan(256) | depth(57600)] = 57901 D

  - proprio       → 直接喂 teacher_actor
  - height_scan   → teacher_encoder → loco_latent_T   (监督 target)
  - depth         → student VisionEncoder → loco_latent_S

切分逻辑见 AlgorithmLBC._split_obs。
"""

from tools.base_env.observation_process import ObservationProcess

from agent_ppo.feature import nav_observation_utils


class LBCObservationProcess(ObservationProcess):
    """LBC 蒸馏阶段专用 policy 观测。"""

    target_group = "policy"

    def process(self):
        # default_observation() = [proprio | height_scan] by Isaac Lab ObsCfg
        obs = self.default_observation()
        depth = nav_observation_utils.depth_camera_image(self.env)   # (N, 57600)
        return self.concatenate_terms(obs, depth)
