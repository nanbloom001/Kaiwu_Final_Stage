# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Track phase and pre-maze terrain semantics from public observations."""

import torch


def estimate_maze_phase_from_goal_feature(obs, rl_nav_conf: dict):
    if obs is None or not hasattr(obs, "shape") or obs.shape[-1] < 304:
        return None

    goal_start = int(rl_nav_conf.get("goal_start", 301))
    if obs.shape[-1] < goal_start + 3:
        return None

    goal_dist_gate = float(rl_nav_conf.get("phase_maze_goal_dist_gate", 14.0))
    mode = str(rl_nav_conf.get("phase_maze_distance_mode", "euclidean")).lower()
    if mode in ("longitudinal", "x", "track_x") and obs.shape[-1] >= goal_start + 1:
        remaining = torch.clamp(obs[:, goal_start], -1.0, 1.0) * 10.0
        min_remaining = float(rl_nav_conf.get("phase_maze_goal_longitudinal_min", -1.0))
        return (remaining < goal_dist_gate) & (remaining > min_remaining)

    goal_dist = torch.clamp(obs[:, goal_start + 2], 0.0, 1.0) * 20.0
    return goal_dist < goal_dist_gate


def classify_pre_maze_terrain_from_height_scan(obs, rl_nav_conf: dict):
    if obs is None or not hasattr(obs, "shape"):
        return None

    scan_start = int(rl_nav_conf.get("scan_start", 45))
    scan_size = int(rl_nav_conf.get("scan_size", 256))
    if obs.shape[-1] < scan_start + scan_size:
        return None

    side = int(scan_size ** 0.5)
    if side * side != scan_size:
        return None
    grid = obs[:, scan_start:scan_start + scan_size].view(obs.shape[0], side, side)

    row_start = max(int(rl_nav_conf.get("terrain_row_start", 3)), 0)
    row_end = min(int(rl_nav_conf.get("terrain_row_end", 13)), grid.shape[1])
    front_cols = min(int(rl_nav_conf.get("terrain_front_cols", 8)), grid.shape[2])
    if row_end <= row_start or front_cols <= 1:
        return None

    sector = grid[:, row_start:row_end, :front_cols]
    if sector.shape[1] == 0 or sector.shape[2] <= 1:
        return None

    lateral_std = sector.std(dim=1, unbiased=False).mean(dim=1)
    dx = sector[:, :, 1:] - sector[:, :, :-1]
    abs_dx = dx.abs()
    if abs_dx.numel() == 0:
        return None

    q = float(rl_nav_conf.get("terrain_step_quantile", 0.85))
    q = min(max(q, 0.0), 1.0)
    step_strength = torch.quantile(abs_dx.flatten(1), q, dim=1)
    sign_consistency = dx.mean(dim=(1, 2)).abs() / (abs_dx.mean(dim=(1, 2)) + 1e-6)
    if dx.shape[2] > 1:
        second_diff = (dx[:, :, 1:] - dx[:, :, :-1]).abs().mean(dim=(1, 2))
    else:
        second_diff = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)

    is_uniform = lateral_std < float(rl_nav_conf.get("terrain_lateral_std_threshold", 0.18))
    not_wall = sector.amin(dim=(1, 2)) > float(rl_nav_conf.get("terrain_wall_height_threshold", -1.05))
    terrain_like = is_uniform & not_wall & (
        step_strength > float(rl_nav_conf.get("terrain_slope_delta_threshold", 0.035))
    )
    stair_like = terrain_like & (
        (step_strength > float(rl_nav_conf.get("terrain_stair_delta_threshold", 0.10)))
        | (second_diff > float(rl_nav_conf.get("terrain_stair_second_diff_threshold", 0.055)))
    )
    slope_like = terrain_like & ~stair_like & (
        sign_consistency > float(rl_nav_conf.get("terrain_slope_sign_consistency_threshold", 0.55))
    )

    terrain_id = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
    terrain_id = torch.where(slope_like, torch.ones_like(terrain_id), terrain_id)
    terrain_id = torch.where(stair_like, torch.full_like(terrain_id, 2), terrain_id)
    return terrain_id


def nan_terrain_gate_from_height_scan(obs, rl_nav_conf: dict):
    """Nan-style terrain gate using only policy-visible height scan.

    Returns a dict with terrain ids: flat=0, slope=1, stairs=2, wall=3.
    This function intentionally avoids env/raw sensor access so training and
    evaluation can use the same gate.
    """
    if obs is None or not hasattr(obs, "shape"):
        return None

    scan_start = int(rl_nav_conf.get("scan_start", 45))
    scan_size = int(rl_nav_conf.get("scan_size", 256))
    if obs.shape[-1] < scan_start + scan_size:
        return None

    side = int(scan_size ** 0.5)
    if side * side != scan_size:
        return None

    try:
        grid = obs[:, scan_start:scan_start + scan_size].view(obs.shape[0], side, side)
    except Exception:
        return None

    if grid.shape[1] < 2 or grid.shape[2] < 2:
        return None

    num_envs = grid.shape[0]
    body_y_start = max(0, int(rl_nav_conf.get("nan_gate_body_y_start", 5)))
    body_y_end = min(int(rl_nav_conf.get("nan_gate_body_y_end", 11)), grid.shape[1])
    x_start = max(0, int(rl_nav_conf.get("nan_gate_x_start", 0)))
    x_end = min(int(rl_nav_conf.get("nan_gate_x_end", 10)), grid.shape[2])
    if body_y_end <= body_y_start or x_end - x_start < 2:
        return None

    window = grid[:, body_y_start:body_y_end, x_start:x_end]
    dx = window[:, :, 1:] - window[:, :, :-1]
    dy = window[:, 1:, :] - window[:, :-1, :]
    abs_edges = torch.cat((dx.abs().reshape(num_envs, -1), dy.abs().reshape(num_envs, -1)), dim=1)
    if abs_edges.shape[1] == 0:
        return None

    edge_sharpness = abs_edges.amax(dim=1)
    edge_mean = abs_edges.mean(dim=1)
    slope_smoothness = torch.clamp(edge_mean / (edge_sharpness + 1.0e-6), 0.0, 1.0)
    edge_locality = torch.clamp(1.0 - slope_smoothness, 0.0, 1.0)

    near_x_end = min(int(rl_nav_conf.get("nan_gate_near_x_end", 4)), grid.shape[2])
    front_x_start = min(int(rl_nav_conf.get("nan_gate_front_x_start", 4)), grid.shape[2] - 1)
    front_x_end = min(int(rl_nav_conf.get("nan_gate_front_x_end", 10)), grid.shape[2])
    if near_x_end <= 0 or front_x_end <= front_x_start:
        step_delta = torch.zeros(num_envs, device=grid.device, dtype=grid.dtype)
    else:
        near_z = grid[:, body_y_start:body_y_end, :near_x_end].mean(dim=(1, 2))
        front_z = grid[:, body_y_start:body_y_end, front_x_start:front_x_end].mean(dim=(1, 2))
        step_delta = front_z - near_z

    wall_like = edge_sharpness > float(rl_nav_conf.get("nan_gate_wall_edge_threshold", 0.30))
    stair_like = (
        (edge_sharpness >= float(rl_nav_conf.get("nan_gate_stair_edge_threshold", 0.040)))
        & (edge_locality >= float(rl_nav_conf.get("nan_gate_stair_locality_threshold", 0.30)))
        & (slope_smoothness <= float(rl_nav_conf.get("nan_gate_stair_smoothness_max", 0.45)))
        & (~wall_like)
    )
    slope_like = (
        (edge_mean >= float(rl_nav_conf.get("nan_gate_slope_edge_mean_min", 0.006)))
        & (slope_smoothness >= float(rl_nav_conf.get("nan_gate_slope_smoothness_min", 0.60)))
        & (edge_sharpness <= float(rl_nav_conf.get("nan_gate_slope_edge_max", 0.070)))
        & (~wall_like)
        & (~stair_like)
    )

    difficulty_signal = torch.where(stair_like, edge_sharpness, step_delta.abs())
    stair_low = float(rl_nav_conf.get("stairs_difficulty_low_threshold", rl_nav_conf.get("nan_gate_difficulty_low_threshold", 0.09)))
    stair_high = float(rl_nav_conf.get("stairs_difficulty_high_threshold", rl_nav_conf.get("nan_gate_difficulty_high_threshold", 0.16)))
    slope_low = float(rl_nav_conf.get("slope_difficulty_low_threshold", rl_nav_conf.get("nan_gate_difficulty_low_threshold", 0.09)))
    slope_high = float(rl_nav_conf.get("slope_difficulty_high_threshold", rl_nav_conf.get("nan_gate_difficulty_high_threshold", 0.16)))
    low_thr = torch.where(
        stair_like,
        torch.full_like(difficulty_signal, stair_low),
        torch.full_like(difficulty_signal, slope_low),
    )
    high_thr = torch.where(
        stair_like,
        torch.full_like(difficulty_signal, stair_high),
        torch.full_like(difficulty_signal, slope_high),
    )

    terrain_id = torch.zeros(num_envs, dtype=torch.long, device=grid.device)
    terrain_id = torch.where(slope_like, torch.ones_like(terrain_id), terrain_id)
    terrain_id = torch.where(stair_like, torch.full_like(terrain_id, 2), terrain_id)
    terrain_id = torch.where(wall_like, torch.full_like(terrain_id, 3), terrain_id)

    active = slope_like | stair_like
    difficulty_band = torch.zeros(num_envs, dtype=torch.long, device=grid.device)
    difficulty_band = torch.where(active & (difficulty_signal >= low_thr), torch.ones_like(difficulty_band), difficulty_band)
    difficulty_band = torch.where(active & (difficulty_signal >= high_thr), torch.full_like(difficulty_band, 2), difficulty_band)

    return {
        "available": torch.ones(num_envs, device=grid.device, dtype=grid.dtype),
        "terrain_id": terrain_id,
        "difficulty_band": difficulty_band,
        "difficulty_active": active,
        "edge_sharpness": edge_sharpness,
        "edge_mean": edge_mean,
        "edge_locality": edge_locality,
        "slope_smoothness": slope_smoothness,
        "step_delta": step_delta,
        "difficulty_signal": difficulty_signal,
    }


# Backward-compatible aliases for existing call sites.
estimate_maze_phase_from_obs = estimate_maze_phase_from_goal_feature
classify_pre_maze_terrain = classify_pre_maze_terrain_from_height_scan
