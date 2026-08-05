#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Depth 观测业务方法（模块级函数）。

方法接收 env（Isaac Lab unwrapped env）而非 self。
"""

import math
import time

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


def nav_scanner_privileged_features(env, *, return_diagnostics: bool = False):
    """Return worker-only maze wall features from ``nav_scanner``.

    Layout is ``[available, front_score, left_score, right_score]``.  The
    feature math intentionally mirrors ``terrain_gate._nav_wall_features`` but
    lives in the nav observation module so the DAgger contract does not depend
    on the retired terrain-gate state machine. Missing or malformed sensors are
    a hard error: silently training the Oracle without wall visibility creates
    a teacher that cannot solve the maze.
    """

    try:
        sensor = env.scene.sensors["nav_scanner"]
        data = sensor.data
        ray_hits = data.ray_hits_w
        raw = data.pos_w[:, 2:3] - ray_hits[..., 2]
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "nav_scanner is required for nav DAgger privileged Oracle features"
        ) from exc
    if raw.ndim != 2 or raw.shape[0] != env.num_envs:
        raise ValueError(
            f"nav_scanner ray tensor must be [N,R], got {tuple(raw.shape)}"
        )
    if ray_hits.ndim != 3 or ray_hits.shape[:2] != raw.shape or ray_hits.shape[2] != 3:
        raise ValueError(
            f"nav_scanner hit tensor must be [N,R,3], got {tuple(ray_hits.shape)}"
        )

    finite_hit = torch.isfinite(ray_hits).all(dim=-1)
    positive_no_hit = torch.isposinf(ray_hits).all(dim=-1)
    well_formed = finite_hit | positive_no_hit
    well_formed_ratio = well_formed.float().mean(dim=1)
    finite_hit_ratio = finite_hit.float().mean(dim=1)
    available = (
        (well_formed_ratio >= 0.95)
        & (finite_hit_ratio >= 0.80)
    )
    strict = not bool(getattr(env, "_p2_allow_scanner_gaps", False))
    if strict and not bool(available.all()):
        raise RuntimeError(
            "nav_scanner strict Oracle validity failed: "
            f"available={int(available.sum())}/{available.numel()} "
            f"well_formed_min={float(well_formed_ratio.min()):.4f} "
            f"finite_hit_min={float(finite_hit_ratio.min()):.4f}"
        )

    rows, cols = _nav_scanner_grid_shape(sensor, raw.shape[1])

    grid = (-raw).view(raw.shape[0], rows, cols)
    finite_grid = finite_hit.view(raw.shape[0], rows, cols)
    well_formed_grid = well_formed.view(raw.shape[0], rows, cols)
    quantile_source = torch.where(finite_grid, grid, torch.full_like(grid, float("nan")))
    floor = torch.nanquantile(quantile_source.flatten(1), 0.20, dim=1).view(-1, 1, 1)
    floor = torch.nan_to_num(floor, nan=0.0, posinf=0.0, neginf=0.0)
    relative = torch.where(
        finite_grid,
        torch.clamp(grid - floor, min=0.0),
        torch.zeros_like(grid),
    )
    body_start = max(0, rows // 2 - 3)
    body_end = min(rows, rows // 2 + 3)
    front_cols = min(6, cols)
    side_width = max(1, min(3, rows // 2))

    def _wall_score(sector, valid_sector, finite_sector):
        score = torch.where(
            finite_sector,
            torch.sigmoid((sector - 0.24) / 0.08),
            torch.zeros_like(sector),
        )
        denominator = valid_sector.float().sum(dim=(1, 2)).clamp_min(1.0)
        return (score * valid_sector.float()).sum(dim=(1, 2)) / denominator

    def _sector(row_slice):
        return _wall_score(
            relative[:, row_slice, :front_cols],
            well_formed_grid[:, row_slice, :front_cols],
            finite_grid[:, row_slice, :front_cols],
        )

    front = _sector(slice(body_start, body_end))
    # ordering="xy" uses lateral y as the outer dimension, ordered -y to +y.
    # In the robot body frame +y is left, so low rows are right and high rows
    # are left.
    right = _sector(slice(0, side_width))
    left = _sector(slice(rows - side_width, rows))
    features = torch.stack((available.float(), front, left, right), dim=-1)
    features[:, 1:] = torch.where(available[:, None], features[:, 1:], torch.zeros_like(features[:, 1:]))
    pattern = getattr(getattr(sensor, "cfg", None), "pattern_cfg", None)
    diagnostics = {
        "available": available,
        "well_formed_ratio": well_formed_ratio,
        "finite_hit_ratio": finite_hit_ratio,
        "rows": rows,
        "cols": cols,
        "ordering": str(getattr(pattern, "ordering", "xy")),
        "pattern_size": tuple(getattr(pattern, "size", ()) or ()),
        "resolution_x": getattr(pattern, "resolution_x", None),
        "resolution_y": getattr(pattern, "resolution_y", None),
    }
    if not bool(getattr(env, "_p2_nav_scanner_contract_logged", False)):
        logger = getattr(env, "logger", None)
        message = (
            "[P2Scanner] rays=%d rows=%d lateral-y cols=%d forward-x ordering=%s "
            "size=%s resolution_x=%s resolution_y=%s right=low-row left=high-row"
            % (
                raw.shape[1], rows, cols, diagnostics["ordering"],
                diagnostics["pattern_size"], diagnostics["resolution_x"],
                diagnostics["resolution_y"],
            )
        )
        if logger is not None and hasattr(logger, "info"):
            logger.info(message)
        setattr(env, "_p2_nav_scanner_contract_logged", True)
    setattr(env, "_p2_nav_scanner_diagnostics", diagnostics)
    now = time.monotonic()
    last_log = float(getattr(env, "_p2_nav_scanner_ratio_log_at", 0.0))
    if now - last_log >= 120.0:
        message = (
            "[P2Scanner] available_share=%.4f well_formed_ratio=%.4f "
            "finite_hit_ratio=%.4f"
            % (
                float(available.float().mean()),
                float(well_formed_ratio.mean()),
                float(finite_hit_ratio.mean()),
            )
        )
        logger = getattr(env, "logger", None)
        if logger is not None and hasattr(logger, "info"):
            logger.info(message)
        elif bool(getattr(env, "_is_training", False)):
            print(message, flush=True)
        setattr(env, "_p2_nav_scanner_ratio_log_at", now)
    if return_diagnostics:
        return features, diagnostics
    return features


def _nav_scanner_grid_shape(sensor, num_rays: int) -> tuple[int, int]:
    """Infer the flattened RayCaster grid using the platform pattern contract."""

    pattern = getattr(getattr(sensor, "cfg", None), "pattern_cfg", None)
    size = getattr(pattern, "size", None)
    resolution_x = getattr(pattern, "resolution_x", None)
    resolution_y = getattr(pattern, "resolution_y", None)
    resolution = getattr(pattern, "resolution", None)
    ordering = getattr(pattern, "ordering", "xy")
    if size is not None and len(size) == 2:
        if resolution_x is None or resolution_y is None:
            resolution_x = resolution_y = resolution
        if resolution_x is not None and resolution_y is not None:
            try:
                resolution_x = float(resolution_x)
                resolution_y = float(resolution_y)
            except (TypeError, ValueError) as exc:
                raise ValueError("nav_scanner pattern resolution is not numeric") from exc
            if resolution_x <= 0.0 or resolution_y <= 0.0:
                raise ValueError("nav_scanner pattern resolution must be positive")
            count_x = int(math.floor(float(size[0]) / resolution_x + 1.0e-9)) + 1
            count_y = int(math.floor(float(size[1]) / resolution_y + 1.0e-9)) + 1
            if ordering == "xy":
                rows, cols = count_y, count_x
            elif ordering == "yx":
                rows, cols = count_x, count_y
            else:
                raise ValueError(f"unsupported ordering {ordering!r}")
            if rows * cols != num_rays:
                raise ValueError(
                    "nav_scanner pattern metadata disagrees with ray tensor: "
                    f"shape=({rows},{cols}), rays={num_rays}, size={size}, "
                    f"resolution_x={resolution_x}, resolution_y={resolution_y}, "
                    f"ordering={ordering}"
                )
            return rows, cols

    # Compatibility for older platform patterns whose cfg metadata is absent.
    known_shapes = {256: (16, 16), 143: (13, 11), 273: (21, 13)}
    if num_rays in known_shapes:
        return known_shapes[num_rays]
    side = int(num_rays ** 0.5)
    if side * side == num_rays:
        return side, side
    raise ValueError(
        f"unsupported nav_scanner ray count {num_rays}; "
        "pattern metadata is unavailable and no known grid shape matches"
    )
