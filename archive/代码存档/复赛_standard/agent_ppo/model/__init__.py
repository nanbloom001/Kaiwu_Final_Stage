#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Model module for agent_ppo — lite baseline.
agent_ppo 模型模块 — lite baseline。

Exports:
    - ActorCritic: MLP actor + MLP critic + Gaussian action distribution
导出：
    - ActorCritic：MLP actor + MLP critic + 高斯动作分布

If players need a terrain-compression or vision model, they can add it under this
directory on their own and update agent_ppo/conf/conf.py::StageConfig.model_class.
选手如需要地形压缩或视觉模型，可自行在本目录下添加并更新
agent_ppo/conf/conf.py::StageConfig.model_class。
"""

from agent_ppo.model.actor_critic import ActorCritic, resolve_nn_activation

__all__ = [
    "ActorCritic",
    "resolve_nn_activation",
]
