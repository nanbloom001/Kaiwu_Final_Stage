#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Agent PPO Algorithm Module — lite baseline.
Agent PPO 算法模块 — lite baseline。

Lite baseline only ships PPO; if players need other algorithms
(e.g. distillation, LBC), they can add them on their own.
Lite baseline 仅预置 PPO；选手如需其他算法（如蒸馏、LBC），可自行添加。
"""

from .algorithm_ppo import AlgorithmPPO

__all__ = [
    "AlgorithmPPO",
]
