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

from agent_ppo.feature import nav_observation_utils, nav_probe
from agent_ppo.feature.worker_command_bridge import (
    apply_worker_command,
    record_worker_command_observation,
)


class LBCObservationProcess(ObservationProcess):
    """LBC 蒸馏阶段专用 policy 观测。"""

    target_group = "policy"

    def process(self):
        apply_worker_command(self.env)
        # S0a 只读探针：默认关闭，由激活阶段 TOML 的 [nav_probe] enabled 门控
        nav_probe.probe_once(self.env)
        # default_observation() = [proprio | height_scan] by Isaac Lab ObsCfg
        obs = self.default_observation()
        record_worker_command_observation(self.env, "policy", obs)
        depth = nav_observation_utils.depth_camera_image(self.env)   # (N, 57600)
        return self.concatenate_terms(obs, depth)
