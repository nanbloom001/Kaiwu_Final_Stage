# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

critic obs layout: [critic_proprio(60) | height_scan(256)] → 316 dim
critic 观测布局：[critic_proprio(60) | height_scan(256)] → 316 维

When extending to track terrain, please refer to the extension guide in
policy_observation_process.py; the critic observation must stay in sync
with the policy on the task-information convention.
扩展到 track 地形时，请参考 policy_observation_process.py 的扩展指引；
critic 观测需保持与 policy 同步的任务信息约定。
"""

from tools.base_env.observation_process import ObservationProcess


class CriticObservationProcess(ObservationProcess):
    target_group = "critic"
    _EXPECTED_OBS_DIM = 316

    def process(self):
        obs = self.default_observation()
        if obs.shape[-1] != self._EXPECTED_OBS_DIM:
            raise ValueError(
                f"Critic observation dim mismatch: expected {self._EXPECTED_OBS_DIM}, got {obs.shape[-1]}. "
                "This usually means height_scan or privileged observation layout changed unexpectedly."
            )
        # TODO (track terrain): if the policy observation appends goal features,
        # the critic observation must keep the same task-information convention.
        # TODO (track 地形)：如果 policy 观测追加了 goal 特征，
        # critic 观测也需保持同步的任务信息约定。
        return obs
