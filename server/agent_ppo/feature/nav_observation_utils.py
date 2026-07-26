#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Depth 观测业务方法（模块级函数）。

方法接收 env（Isaac Lab unwrapped env）而非 self。
"""

import math

import torch
import torch.nn.functional as F

from agent_ppo.conf.depth_config import resolve_depth_preprocess_conf


def depth_camera_image(env) -> torch.Tensor:
    """从 TiledCamera 读取 flatten 归一化深度图，并按配置施加增强。

    对应 `velocity_env_cfg.py :: DepthImageCfg` 里 `mdp.depth_camera_image` 调用的
    方法版本。默认参数保持历史行为，但可由
    ``[camera.depth_camera.augmentation]`` 覆盖：

      - sensor_name          = "depth_camera"
      - max_depth            = 5.0     # D435i 实用量程
      - pixel_dropout_prob   = 0.1     # 10% 像素置零，模拟空洞
      - gaussian_noise_scale = 0.02    # 深度比例高斯噪声
      - roll_range_deg       = 10.0    # roll 抖动 ±10°
      - pitch_shift_pix      = 10      # pitch 抖动 ±10 px

    Augmentation pipeline（仅训练模式且 augmentation.enabled=true 时启用）:
      1. 归一化：clamp [0, max_depth] 后除以 max_depth → [0, 1]
      2. 像素 dropout：随机将像素置零（模拟 D435i 空洞）
      3. 高斯噪声：std = scale * normalized_depth（越远越嘈杂）
      4. Roll 抖动：绕图像中心随机旋转
      5. Pitch 抖动：随机纵向平移

    Returns:
        torch.Tensor: shape (num_envs, H*W)，值域 [0, 1]。
                      D435i 配置 (180×320) 下为 (num_envs, 57600)。
    """

    preprocess_conf = resolve_depth_preprocess_conf(env)
    augmentation_conf = preprocess_conf.get("augmentation", {})
    if not isinstance(augmentation_conf, dict):
        augmentation_conf = {}

    # 未显式配置时保持历史训练行为；评估模式始终关闭随机增强。
    sensor_name = "depth_camera"
    max_depth = float(preprocess_conf.get("max_depth", 5.0))
    pixel_dropout_prob = float(
        augmentation_conf.get("pixel_dropout_prob", 0.1)
    )
    gaussian_noise_scale = float(
        augmentation_conf.get("gaussian_noise_scale", 0.02)
    )
    roll_range_deg = float(augmentation_conf.get("roll_range_deg", 10.0))
    pitch_shift_pix = int(augmentation_conf.get("pitch_shift_pix", 10))

    if env is None or not hasattr(env, "scene"):
        raise RuntimeError("当前 observation process 尚未绑定有效 env.scene，无法读取传感器数据。")

    sensors = env.scene.sensors
    if sensor_name not in sensors:
        available = list(sensors.keys())
        raise KeyError(f"传感器 '{sensor_name}' 不存在于场景中。可用传感器: {available}")

    camera = sensors[sensor_name]
    # TiledCamera depth output shape: [N, H, W, 1]
    depth = camera.data.output["depth"]
    depth = depth.squeeze(-1)  # [N, H, W]

    # 超出量程的像素置 0，匹配真实 D435i 行为（仿真器默认返回 max clip 值）。
    out_of_range = depth >= max_depth
    depth = torch.clamp(depth, 0.0, max_depth)
    depth = depth / max_depth  # [0, 1]
    depth[out_of_range] = 0.0

    N, H, W = depth.shape
    is_eval = bool(getattr(env, "_is_eval", False))
    training = bool(getattr(env, "_is_training", not is_eval)) and not is_eval
    augmentation_enabled = training and bool(
        augmentation_conf.get("enabled", True)
    )

    if augmentation_enabled:
        # --- Pixel dropout: 随机置零像素，模拟缺失深度 ---
        if pixel_dropout_prob > 0.0:
            mask = torch.rand(N, H, W, device=depth.device) > pixel_dropout_prob
            depth = depth * mask

        # --- 深度比例高斯噪声 ---
        if gaussian_noise_scale > 0.0:
            noise = torch.randn_like(depth) * gaussian_noise_scale * depth
            depth = torch.clamp(depth + noise, 0.0, 1.0)

        # --- Roll 抖动：绕图像中心旋转 ---
        if roll_range_deg > 0.0:
            angles = (torch.rand(N, device=depth.device) * 2 - 1) * roll_range_deg
            rad = angles * (math.pi / 180.0)
            cos_a = torch.cos(rad)
            sin_a = torch.sin(rad)
            theta = torch.zeros(N, 2, 3, device=depth.device)
            theta[:, 0, 0] = cos_a
            theta[:, 0, 1] = -sin_a
            theta[:, 1, 0] = sin_a
            theta[:, 1, 1] = cos_a
            grid = F.affine_grid(theta, [N, 1, H, W], align_corners=False)
            depth = F.grid_sample(
                depth.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=False
            ).squeeze(1)

        # --- Pitch 抖动：纵向平移 ---
        if pitch_shift_pix > 0:
            shifts = torch.randint(-pitch_shift_pix, pitch_shift_pix + 1, (N,), device=depth.device)
            theta = torch.zeros(N, 2, 3, device=depth.device)
            theta[:, 0, 0] = 1.0
            theta[:, 1, 1] = 1.0
            theta[:, 1, 2] = shifts.float() * (2.0 / H)  # 归一化像素偏移
            grid = F.affine_grid(theta, [N, 1, H, W], align_corners=False)
            depth = F.grid_sample(
                depth.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=False
            ).squeeze(1)

    return depth.reshape(N, -1)  # [N, H*W]
