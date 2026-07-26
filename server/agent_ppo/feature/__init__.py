#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature module entry — exposes Policy/Critic observation processors and
RewardProcess. The observation processes actually used depend on the current
stage's ``algorithm`` attribute in ``Config.CURRENT``:

    - "lbc_loco"/"visual_ppo"   → LBCObservationProcess（policy 拼接 depth）
                                   + CriticObservationProcess（默认 critic）
    - "nav_dagger"/"nav_eval"   → NavPolicyObservationProcess（57905 维，含 goal4+depth）
                                   + NavCriticObservationProcess（319 维，含真值 goal3）
    - anything else (default)   → PolicyObservationProcess + CriticObservationProcess

全部导出（Policy / Critic / RewardProcess）都经 PEP 562 ``__getattr__`` 惰性
解析：base_env.py 经 importlib 读取本模块属性，惰性解析保证 Config.CURRENT
被运行时（如 eval 按 TOML 推导）覆盖后再读取仍拿到正确实现；同时避免在
无平台 ``tools`` 包的本地测试环境里 import 本包即失败。

Nav 阶段的 Policy 与 Critic **必须一起生效**（policy 带 goal4、critic 带真值
goal3，两者的任务信息约定必须同步）——因此二者由同一个 algorithm 判定分发。
"""


def _current_algorithm() -> str:
    # 延迟 import：Config 是运行时状态，避免 feature 模块加载时锁定实现
    from agent_ppo.conf.conf import Config

    return getattr(Config.CURRENT, "algorithm", "ppo")


def _resolve_policy_observation_process():
    """根据 Config.CURRENT.algorithm 返回合适的 PolicyObservationProcess 类。"""

    algorithm = _current_algorithm()

    if algorithm in {"nav_dagger", "nav_eval"}:
        from agent_ppo.feature.nav_observation_process import (
            NavPolicyObservationProcess as _Policy,
        )
        return _Policy

    if algorithm in {"lbc_loco", "visual_ppo"}:
        from agent_ppo.feature.lbc_observation_process import (
            LBCObservationProcess as _Policy,
        )
        return _Policy

    from agent_ppo.feature.policy_observation_process import (
        PolicyObservationProcess as _Policy,
    )
    return _Policy


def _resolve_critic_observation_process():
    """根据 Config.CURRENT.algorithm 返回合适的 CriticObservationProcess 类。

    Nav 阶段 critic 必须与 policy 一起切换（真值 goal3 拼接），否则 Oracle
    拿不到目标信息而 policy 侧却带着 goal4 —— 任务信息约定被破坏。
    """

    algorithm = _current_algorithm()

    if algorithm in {"nav_dagger", "nav_eval"}:
        from agent_ppo.feature.nav_observation_process import (
            NavCriticObservationProcess as _Critic,
        )
        return _Critic

    from agent_ppo.feature.critic_observation_process import (
        CriticObservationProcess as _Critic,
    )
    return _Critic


# 模块级导出：base_env.py 通过 importlib 读取 module.PolicyObservationProcess /
# module.CriticObservationProcess / module.RewardProcess。全部惰性解析。
def __getattr__(name):
    if name == "PolicyObservationProcess":
        return _resolve_policy_observation_process()
    if name == "CriticObservationProcess":
        return _resolve_critic_observation_process()
    if name == "RewardProcess":
        from agent_ppo.feature.reward_process import RewardProcess as _Reward

        return _Reward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CriticObservationProcess",
    "PolicyObservationProcess",
    "RewardProcess",
]
