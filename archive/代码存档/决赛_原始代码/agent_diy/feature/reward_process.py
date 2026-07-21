#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):
    """Locomotion reward processor — inherits all base native reward terms.
    Locomotion reward processor — 仅继承 base 原生 reward 项。

    agent_diy only carries LocomotionConfig (standard locomotion), all reward terms
    are provided by RewardProcessBase (track_lin_vel_xy / track_ang_vel_z / lin_vel_z /
    ang_vel_xy / joint_acc / joint_torques / dof_pos_limits / action_rate /
    undesired_contacts / flat_orientation, etc.). TOML [rewards.*] only references
    terms already implemented in base; no override or addition is needed here.
    agent_diy 只承载 LocomotionConfig（standard locomotion），所有 reward 项均由
    RewardProcessBase 提供（track_lin_vel_xy / track_ang_vel_z / lin_vel_z /
    ang_vel_xy / joint_acc / joint_torques / dof_pos_limits / action_rate /
    undesired_contacts / flat_orientation 等）。TOML [rewards.*] 只引用 base 已实现的项，
    无需在此覆盖或新增。

    To customize, override with `_reward_{term_name}` convention in this subclass.
    如需自定义，按 `_reward_{term_name}` 约定在本子类覆盖。
    """

    pass
