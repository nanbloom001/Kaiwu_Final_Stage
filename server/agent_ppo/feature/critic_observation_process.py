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

When extending to track terrain, please refer to the extension guide in
policy_observation_process.py; the critic observation must stay in sync
with the policy on the task-information convention.
扩展到 track 地形时，请参考 policy_observation_process.py 的扩展指引；
critic 观测需保持与 policy 同步的任务信息约定。
"""

from tools.base_env.observation_process import ObservationProcess

from agent_ppo.feature.worker_command_bridge import (
    apply_worker_command,
    record_worker_command_observation,
)


class CriticObservationProcess(ObservationProcess):
    target_group = "critic"

    def process(self):
        apply_worker_command(self.env)
        obs = self.default_observation()
        record_worker_command_observation(self.env, "critic", obs)
        # TODO (track terrain): if the policy observation appends goal features,
        # the critic observation must keep the same task-information convention.
        # TODO (track 地形)：如果 policy 观测追加了 goal 特征，
        # critic 观测也需保持同步的任务信息约定。
        return obs
