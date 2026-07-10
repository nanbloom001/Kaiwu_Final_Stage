#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls, Frame
import torch
import numpy as np
import collections
from agent_diy.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls(
    "ActData",
    action=None,
)


def sample_process(collector):
    """
    Process samples from collector
    从收集器处理样本
    """
    return collector.sample_process()


def build_frame(frame_no, obs, actions, dones, rewards):
    """
    Create sample data for the current frame
    创建当前帧的样本
    """

    frame = Frame(
        frame_no=frame_no,
        obs=obs,
        actions=actions,
        done=dones,
        rewards=rewards,
    )
    return frame


def obs_normalizer(obs):
    """
    Observation normalizer function
    观测归一化函数
    """
    pass
