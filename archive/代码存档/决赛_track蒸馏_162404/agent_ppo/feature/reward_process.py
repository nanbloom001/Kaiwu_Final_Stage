#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

RewardProcess — custom reward processor.
RewardProcess — 自定义奖励处理器。

Ships one example reward:
    _reward_termination — penalise real failures (terminated ∧ ¬time_out)
预置一个示例 reward：
    _reward_termination — 惩罚真正的失败（terminated ∧ ¬time_out）

Other generic locomotion rewards (track_lin_vel_xy / joint_acc / action_rate, etc.)
are inherited from RewardProcessBase (see tools/base_env/base_reward.py).
Activate them in the TOML; no need to re-implement here.
其余通用 locomotion reward（track_lin_vel_xy / joint_acc / action_rate 等）
继承自 RewardProcessBase（见 tools/base_env/base_reward.py），
在 TOML 中激活即可，无需在此重复实现。
"""

import torch

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):
    def _reward_termination(self):
        """Penalise real failures (terminated ∧ ¬time_out).
        惩罚真正的失败（terminated ∧ ¬time_out）。
        """
        term_mgr = self.env.termination_manager
        return (term_mgr.terminated & ~term_mgr.time_outs).float()
