#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Model module for agent_ppo.
agent_ppo 模型模块。

Exports:
    - ActorCritic:         MLP actor + MLP critic + Gaussian action distribution
    - ActorCriticEncoder:  height_scan(256) → latent(32) 编码器 + MLP actor/critic
    - L2Norm:              L2 归一化层（编码器末端）
    - VisionEncoder:       depth 图像 CNN + LSTM 编码器（LBC 蒸馏学生）
    - VisualActorCritic:   depth Actor + privileged-state Critic（视觉 PPO）
    - DmEncoder:           height_scan MLP 编码器（LBC 蒸馏教师）
    - CNNRNN:              VisionEncoder 的兼容外壳
    - create_cnn_encoder:  SimpleCNN 构造函数
"""

from agent_ppo.model.actor_critic import ActorCritic, resolve_nn_activation
from agent_ppo.model.actor_critic_encoder import ActorCriticEncoder, L2Norm
from agent_ppo.model.simple_cnn import create_cnn_encoder
from agent_ppo.model.vision_encoder import VisionEncoder, DmEncoder, CNNRNN
from agent_ppo.model.visual_actor_critic import VisualActorCritic

__all__ = [
    "ActorCritic",
    "resolve_nn_activation",
    "ActorCriticEncoder",
    "L2Norm",
    "create_cnn_encoder",
    "VisionEncoder",
    "DmEncoder",
    "CNNRNN",
    "VisualActorCritic",
]
