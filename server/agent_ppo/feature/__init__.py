#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature module entry — exposes Policy/Critic observation processors and
RewardProcess. The PolicyObservationProcess actually used depends on the
current stage's ``algorithm`` attribute in ``Config.CURRENT``:

    - "lbc_loco"/"visual_ppo" → agent_ppo.feature.lbc_observation_process.LBCObservationProcess
                                 (policy obs 拼接 depth_image，供学生 VisionEncoder 使用)
    - anything else (default) → agent_ppo.feature.policy_observation_process.PolicyObservationProcess
                                 (proprio + height_scan)

根据当前阶段 ``Config.CURRENT.algorithm`` 动态选择 PolicyObservationProcess：
  - "lbc_loco" / "visual_ppo" → LBCObservationProcess（拼接 depth_image）
  - 其他（默认） → PolicyObservationProcess
"""

from agent_ppo.feature.critic_observation_process import CriticObservationProcess
from agent_ppo.feature.reward_process import RewardProcess


def _resolve_policy_observation_process():
    """根据 Config.CURRENT.algorithm 返回合适的 PolicyObservationProcess 类。

    Delayed import: Config 在 conf.py 中按模块级暴露，import 应在函数内进行
    以避免循环 import；algorithm 缺省为 "ppo"。
    """
    # 延迟 import：Config 是运行时状态，避免 feature 模块加载时就锁定实现
    from agent_ppo.conf.conf import Config

    algorithm = getattr(Config.CURRENT, "algorithm", "ppo")

    if algorithm in {"lbc_loco", "visual_ppo"}:
        from agent_ppo.feature.lbc_observation_process import (
            LBCObservationProcess as _Policy,
        )
        return _Policy

    from agent_ppo.feature.policy_observation_process import (
        PolicyObservationProcess as _Policy,
    )
    return _Policy


# 模块级导出：base_env.py 通过 importlib 读取 module.PolicyObservationProcess。
# 用 PEP 562 __getattr__ 做惰性解析，保证 Config.CURRENT 被运行时（如 eval
# 模式按 TOML 推导）覆盖后再读取仍能拿到正确实现，避免 import-time 锁定。
def __getattr__(name):
    if name == "PolicyObservationProcess":
        return _resolve_policy_observation_process()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CriticObservationProcess",
    "PolicyObservationProcess",
    "RewardProcess",
]
