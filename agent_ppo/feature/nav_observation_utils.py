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


_DEPTH_AUGMENTATION = {"enabled": False}


def configure_depth_augmentation(config=None, training: bool = False):
    """Set LBC image-space augmentation from the stage TOML before env.reset()."""
    global _DEPTH_AUGMENTATION
    _DEPTH_AUGMENTATION = dict(config or {})
    _DEPTH_AUGMENTATION["enabled"] = bool(_DEPTH_AUGMENTATION.get("enabled", False) and training)


def _range_pair(value, default):
    value = value if isinstance(value, (list, tuple)) and len(value) == 2 else default
    return int(value[0]), int(value[1])


def depth_camera_image(env) -> torch.Tensor:
    """从 TiledCamera 读取 flatten 归一化深度图，并施加 sim2real 增强。

    对应 `velocity_env_cfg.py :: DepthImageCfg` 里 `mdp.depth_camera_image` 调用的
    方法版本，参数与该 ObsTerm 的 `params` 字段一一对齐（全部硬编码在方法内）：

      - sensor_name          = "depth_camera"
      - max_depth            = 5.0     # D435i 实用量程
      - pixel_dropout_prob   = 0.1     # 10% 像素置零，模拟空洞
      - gaussian_noise_scale = 0.02    # 深度比例高斯噪声
      - roll_range_deg       = 10.0    # roll 抖动 ±10°
      - pitch_shift_pix      = 10      # pitch 抖动 ±10 px

    Augmentation pipeline（仅在 `env._is_training` 为真时启用，默认 True）:
      1. 归一化：clamp [0, max_depth] 后除以 max_depth → [0, 1]
      2. 像素 dropout：随机将像素置零（模拟 D435i 空洞）
      3. 高斯噪声：std = scale * normalized_depth（越远越嘈杂）
      4. Roll 抖动：绕图像中心随机旋转
      5. Pitch 抖动：随机纵向平移

    Returns:
        torch.Tensor: shape (num_envs, H*W)，值域 [0, 1]。
                      D435i 配置 (180×320) 下为 (num_envs, 57600)。
    """

    # Hardware range remains fixed; only image-space corruption is randomized.
    sensor_name = "depth_camera"
    max_depth = 5.0
    aug = _DEPTH_AUGMENTATION
    pixel_dropout_prob = float(aug.get("pixel_dropout_prob", 0.0))
    gaussian_noise_scale = float(aug.get("gaussian_noise_scale", 0.0))
    roll_range_deg = float(aug.get("roll_range_deg", 0.0))
    pitch_shift_pix = int(aug.get("pitch_shift_pix", 0))

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
    training = bool(getattr(env, "_is_training", False) and aug.get("enabled", False))

    if training:
        # --- Pixel dropout: 随机置零像素，模拟缺失深度 ---
        if pixel_dropout_prob > 0.0:
            mask = torch.rand(N, H, W, device=depth.device) > pixel_dropout_prob
            depth = depth * mask

        # --- 深度比例高斯噪声 ---
        if gaussian_noise_scale > 0.0:
            noise = torch.randn_like(depth) * gaussian_noise_scale * depth
            depth = torch.clamp(depth + noise, 0.0, 1.0)

        # Persistent rectangular holes model temporal invalid-depth regions.
        state = getattr(env, "_lbc_depth_aug_state", None)
        if not isinstance(state, dict) or state.get("shape") != (N, H, W) or state.get("device") != depth.device:
            state = {"shape": (N, H, W), "device": depth.device,
                     "hole_mask": torch.zeros(N, H, W, dtype=torch.bool, device=depth.device),
                     "hole_ttl": torch.zeros(N, dtype=torch.long, device=depth.device),
                     "hold_ttl": torch.zeros(N, dtype=torch.long, device=depth.device),
                     "last_depth": depth.clone()}
            env._lbc_depth_aug_state = state
        hole_prob = float(aug.get("hole_patch_prob", 0.0))
        hole_min, hole_max = _range_pair(aug.get("hole_patch_size"), (4, 16))
        ttl_min, ttl_max = _range_pair(aug.get("hole_hold_steps"), (2, 8))
        state["hole_ttl"] = torch.clamp(state["hole_ttl"] - 1, min=0)
        # A completed hole must disappear even when this frame does not spawn
        # a replacement; otherwise a temporary corruption becomes permanent.
        state["hole_mask"][state["hole_ttl"] == 0] = False
        renew = (state["hole_ttl"] == 0) & (torch.rand(N, device=depth.device) < hole_prob)
        if renew.any():
            state["hole_mask"][renew] = False
            for idx in renew.nonzero(as_tuple=False).squeeze(-1).tolist():
                size = int(torch.randint(hole_min, hole_max + 1, (1,), device=depth.device).item())
                top = int(torch.randint(0, max(1, H - size + 1), (1,), device=depth.device).item())
                left = int(torch.randint(0, max(1, W - size + 1), (1,), device=depth.device).item())
                state["hole_mask"][idx, top:top + size, left:left + size] = True
            state["hole_ttl"][renew] = torch.randint(ttl_min, ttl_max + 1, (int(renew.sum().item()),), device=depth.device)
        depth = depth.masked_fill(state["hole_mask"], 0.0)

        edge_prob = float(aug.get("edge_dropout_prob", 0.0))
        if edge_prob > 0.0:
            dy = F.pad((depth[:, 1:] - depth[:, :-1]).abs(), (0, 0, 0, 1))
            dx = F.pad((depth[:, :, 1:] - depth[:, :, :-1]).abs(), (0, 1, 0, 0))
            edge = (dx + dy) > 0.03
            depth = depth.masked_fill(edge & (torch.rand_like(depth) < edge_prob), 0.0)

        blur_kernel = int(aug.get("blur_kernel_size", 0))
        if blur_kernel > 1 and blur_kernel % 2 == 1:
            depth = F.avg_pool2d(depth.unsqueeze(1), blur_kernel, stride=1, padding=blur_kernel // 2).squeeze(1)

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

        # Frame hold is applied after all spatial corruption, matching a delayed
        # depth stream rather than a stale clean render.
        hold_prob = float(aug.get("frame_hold_prob", 0.0))
        hold_min, hold_max = _range_pair(aug.get("frame_hold_steps"), (1, 5))
        state["hold_ttl"] = torch.clamp(state["hold_ttl"] - 1, min=0)
        new_hold = (state["hold_ttl"] == 0) & (torch.rand(N, device=depth.device) < hold_prob)
        state["hold_ttl"][new_hold] = torch.randint(hold_min, hold_max + 1, (int(new_hold.sum().item()),), device=depth.device)
        use_last = state["hold_ttl"] > 0
        current = depth
        depth = torch.where(use_last.view(N, 1, 1), state["last_depth"], current)
        state["last_depth"] = torch.where(use_last.view(N, 1, 1), state["last_depth"], current).detach()

    return depth.reshape(N, -1)  # [N, H*W]
