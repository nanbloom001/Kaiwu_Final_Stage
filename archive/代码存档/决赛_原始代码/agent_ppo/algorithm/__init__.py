#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Agent PPO Algorithm Module.
Agent PPO 算法模块。

Exports:
    - AlgorithmPPO: flat PPO
    - AlgorithmLBC: teacher-student MSE distillation
导出：
    - AlgorithmPPO：flat PPO
    - AlgorithmLBC：教师-学生 MSE 蒸馏
"""

from .algorithm_ppo import AlgorithmPPO
from .algorithm_lbc import AlgorithmLBC

__all__ = [
    "AlgorithmPPO",
    "AlgorithmLBC",
]
