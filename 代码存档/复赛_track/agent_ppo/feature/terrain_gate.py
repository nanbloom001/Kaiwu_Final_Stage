# -*- coding: UTF-8 -*-
"""Worker-side terrain gate and command probe for track short tests."""

from __future__ import annotations

import os

import torch

from agent_ppo.conf.conf import Config

try:
    import toml
except ModuleNotFoundError:  # pragma: no cover - container dependent
    toml = None
    import tomllib


_CONF_CACHE = None
_CONF_CACHE_PATH = None


def _load_toml(path):
    if toml is not None:
        with open(path, "r", encoding="utf-8") as f:
            return toml.load(f)
    with open(path, "rb") as f:
        return tomllib.load(f)


def _rl_conf():
    global _CONF_CACHE, _CONF_CACHE_PATH
    stage = Config.CURRENT
    rel_path = f"agent_ppo/conf/train_env_conf_{stage.task_type}_{stage.name}.toml"
    if _CONF_CACHE is None or _CONF_CACHE_PATH != rel_path:
        if os.path.exists(rel_path):
            _CONF_CACHE = _load_toml(rel_path)
        else:
            _CONF_CACHE = {}
        _CONF_CACHE_PATH = rel_path
    return (_CONF_CACHE or {}).get("rl_navigation", {})


def _diagnostics_enabled(conf):
    return bool(conf.get("gate_diagnostics_enabled", False))


def _zeros(env, dtype=torch.float32):
    return torch.zeros(env.num_envs, device=env.device, dtype=dtype)


def _bools(env):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def _range_midpoint(values, default):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return float(default)
    return 0.5 * (float(values[0]) + float(values[1]))


def _sample_range(values, shape, device, dtype):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        values = [0.0, 0.0]
    low, high = float(values[0]), float(values[1])
    if high < low:
        low, high = high, low
    if abs(high - low) < 1.0e-8:
        return torch.full(shape, low, device=device, dtype=dtype)
    return low + (high - low) * torch.rand(shape, device=device, dtype=dtype)


def _step_key(env):
    value = getattr(env, "common_step_counter", None)
    if value is None:
        value = getattr(env, "_common_step_counter", None)
    try:
        if torch.is_tensor(value):
            return int(value.item())
        if value is not None:
            return int(value)
    except Exception:
        pass
    return None


def _current_command(env):
    try:
        return env.command_manager.get_command("base_velocity")
    except Exception:
        return torch.zeros(env.num_envs, 3, device=env.device)


def _write_command(env, command):
    try:
        current = env.command_manager.get_command("base_velocity")
        if current.shape != command.shape:
            return False
        current.copy_(command.to(device=current.device, dtype=current.dtype))
        return True
    except Exception:
        return False


def _sensor_grid(env, sensor_name):
    try:
        sensor = env.scene.sensors.get(sensor_name)
    except Exception:
        return None
    if sensor is None:
        return None
    data = getattr(sensor, "data", None)
    if data is None or not hasattr(data, "pos_w") or not hasattr(data, "ray_hits_w"):
        return None
    try:
        scan = data.pos_w[:, 2:3] - data.ray_hits_w[..., 2]
        scan = torch.nan_to_num(scan, nan=0.0, posinf=0.0, neginf=0.0)
        num_rays = scan.shape[-1]
        if num_rays == 256:
            rows, cols = 16, 16
        elif num_rays == 143:
            rows, cols = 13, 11
        else:
            side = int(num_rays ** 0.5)
            if side * side != num_rays:
                return None
            rows, cols = side, side
        return (-scan).view(scan.shape[0], rows, cols)
    except Exception:
        return None


def _obs_grid(obs, group="policy"):
    if obs is None or obs.shape[-1] < 301:
        return None
    scan_start = 60 if group == "critic" else 45
    if obs.shape[-1] < scan_start + 256:
        return None
    try:
        return obs[:, scan_start:scan_start + 256].view(obs.shape[0], 16, 16)
    except Exception:
        return None


def _relative_height_grid(grid, floor_quantile=0.20):
    """Convert an elevation grid into height above the local floor estimate."""
    if grid is None:
        return None
    q = min(max(float(floor_quantile), 0.0), 1.0)
    floor = torch.quantile(grid.flatten(1), q, dim=1).view(-1, 1, 1)
    return torch.clamp(grid - floor, min=0.0)


def _empty_gate(env, source_id=0):
    terrain_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    difficulty = torch.zeros_like(terrain_id)
    return {
        "available": _zeros(env),
        "source_id": torch.full((env.num_envs,), float(source_id), device=env.device),
        "terrain_id": terrain_id,
        "difficulty_band": difficulty,
        "flat": terrain_id == 0,
        "slope": _bools(env),
        "stairs": _bools(env),
        "wall": _bools(env),
        "up": _bools(env),
        "down": _bools(env),
        "edge_sharpness": _zeros(env),
        "edge_mean": _zeros(env),
        "edge_locality": _zeros(env),
        "slope_smoothness": _zeros(env),
        "step_delta": _zeros(env),
        "difficulty_signal": _zeros(env),
        "nav_available": _zeros(env),
    }


def _copy_gate(env, gate, source_id=None):
    copied = {}
    for key, value in gate.items():
        if torch.is_tensor(value):
            copied[key] = value.detach().clone()
        else:
            copied[key] = value
    if source_id is not None:
        copied["source_id"] = torch.full((env.num_envs,), float(source_id), device=env.device)
    return copied


def _structure_gate(env, grid, conf, prefix, source_id):
    gate = _empty_gate(env, source_id)
    if grid is None or grid.shape[1] < 2 or grid.shape[2] < 2:
        return gate

    num_envs = grid.shape[0]
    body_y_start = max(0, int(conf.get(f"{prefix}_body_y_start", max(0, grid.shape[1] // 2 - 3))))
    body_y_end = min(int(conf.get(f"{prefix}_body_y_end", min(grid.shape[1], grid.shape[1] // 2 + 3))), grid.shape[1])
    x_start = max(0, int(conf.get(f"{prefix}_x_start", 0)))
    x_end = min(int(conf.get(f"{prefix}_x_end", min(grid.shape[2], 10))), grid.shape[2])
    if body_y_end <= body_y_start or x_end - x_start < 2:
        return gate

    window = grid[:, body_y_start:body_y_end, x_start:x_end]
    dx = window[:, :, 1:] - window[:, :, :-1]
    dy = window[:, 1:, :] - window[:, :-1, :]
    abs_edges = torch.cat((dx.abs().reshape(num_envs, -1), dy.abs().reshape(num_envs, -1)), dim=1)
    if abs_edges.shape[1] == 0:
        return gate

    edge_sharpness = abs_edges.amax(dim=1)
    edge_mean = abs_edges.mean(dim=1)
    slope_smoothness = torch.clamp(edge_mean / (edge_sharpness + 1.0e-6), 0.0, 1.0)
    edge_locality = torch.clamp(1.0 - slope_smoothness, 0.0, 1.0)

    near_x_end = min(int(conf.get(f"{prefix}_near_x_end", 4)), grid.shape[2])
    front_x_start = min(int(conf.get(f"{prefix}_front_x_start", 4)), grid.shape[2] - 1)
    front_x_end = min(int(conf.get(f"{prefix}_front_x_end", min(grid.shape[2], 10))), grid.shape[2])
    if near_x_end <= 0 or front_x_end <= front_x_start:
        step_delta = _zeros(env, dtype=grid.dtype)
    else:
        near_z = grid[:, body_y_start:body_y_end, :near_x_end].mean(dim=(1, 2))
        front_z = grid[:, body_y_start:body_y_end, front_x_start:front_x_end].mean(dim=(1, 2))
        step_delta = front_z - near_z

    wall_like = edge_sharpness > float(conf.get(f"{prefix}_wall_edge_threshold", 0.30))
    stair_like = (
        (edge_sharpness >= float(conf.get(f"{prefix}_stair_edge_threshold", 0.040)))
        & (edge_locality >= float(conf.get(f"{prefix}_stair_locality_threshold", 0.30)))
        & (slope_smoothness <= float(conf.get(f"{prefix}_stair_smoothness_max", 0.45)))
        & (~wall_like)
    )
    slope_like = (
        (edge_mean >= float(conf.get(f"{prefix}_slope_edge_mean_min", 0.006)))
        & (slope_smoothness >= float(conf.get(f"{prefix}_slope_smoothness_min", 0.60)))
        & (edge_sharpness <= float(conf.get(f"{prefix}_slope_edge_max", 0.080)))
        & (~wall_like)
        & (~stair_like)
    )
    direction_margin = float(conf.get(f"{prefix}_direction_margin", 0.02))
    up = stair_like & (step_delta > direction_margin)
    down = stair_like & (step_delta < -direction_margin)

    difficulty_signal = torch.where(stair_like, edge_sharpness, step_delta.abs())
    low_thr = float(conf.get(f"{prefix}_difficulty_low_threshold", conf.get("fused_gate_difficulty_low_threshold", 0.09)))
    high_thr = float(conf.get(f"{prefix}_difficulty_high_threshold", conf.get("fused_gate_difficulty_high_threshold", 0.16)))
    difficulty = torch.zeros(num_envs, dtype=torch.long, device=grid.device)
    active = stair_like | slope_like
    difficulty = torch.where(active & (difficulty_signal >= low_thr), torch.ones_like(difficulty), difficulty)
    difficulty = torch.where(active & (difficulty_signal >= high_thr), torch.full_like(difficulty, 2), difficulty)

    terrain_id = torch.zeros(num_envs, dtype=torch.long, device=grid.device)
    terrain_id = torch.where(slope_like, torch.ones_like(terrain_id), terrain_id)
    terrain_id = torch.where(stair_like, torch.full_like(terrain_id, 2), terrain_id)
    terrain_id = torch.where(wall_like, torch.full_like(terrain_id, 3), terrain_id)

    gate.update({
        "available": torch.ones(num_envs, device=grid.device, dtype=torch.float32),
        "terrain_id": terrain_id,
        "difficulty_band": difficulty,
        "flat": terrain_id == 0,
        "slope": slope_like,
        "stairs": stair_like,
        "wall": wall_like,
        "up": up,
        "down": down,
        "edge_sharpness": edge_sharpness,
        "edge_mean": edge_mean,
        "edge_locality": edge_locality,
        "slope_smoothness": slope_smoothness,
        "step_delta": step_delta,
        "difficulty_signal": difficulty_signal,
    })
    return gate


def _nan_structure_gate(env, grid, conf):
    """Nan-style height-scan structure gate, independent from nav fusion."""
    gate = _empty_gate(env, 1)
    if grid is None or grid.shape[1] < 2 or grid.shape[2] < 2:
        return gate

    num_envs = grid.shape[0]
    body_y_start = max(0, int(conf.get("nan_gate_body_y_start", 5)))
    body_y_end = min(int(conf.get("nan_gate_body_y_end", 11)), grid.shape[1])
    x_start = max(0, int(conf.get("nan_gate_x_start", 0)))
    x_end = min(int(conf.get("nan_gate_x_end", 10)), grid.shape[2])
    if body_y_end <= body_y_start or x_end - x_start < 2:
        return gate

    window = grid[:, body_y_start:body_y_end, x_start:x_end]
    dx = window[:, :, 1:] - window[:, :, :-1]
    dy = window[:, 1:, :] - window[:, :-1, :]
    abs_edges = torch.cat((dx.abs().reshape(num_envs, -1), dy.abs().reshape(num_envs, -1)), dim=1)
    if abs_edges.shape[1] == 0:
        return gate

    edge_sharpness = abs_edges.amax(dim=1)
    edge_mean = abs_edges.mean(dim=1)
    slope_smoothness = torch.clamp(edge_mean / (edge_sharpness + 1.0e-6), 0.0, 1.0)
    edge_locality = torch.clamp(1.0 - slope_smoothness, 0.0, 1.0)

    near_x_end = min(int(conf.get("nan_gate_near_x_end", 4)), grid.shape[2])
    front_x_start = min(int(conf.get("nan_gate_front_x_start", 4)), grid.shape[2] - 1)
    front_x_end = min(int(conf.get("nan_gate_front_x_end", 10)), grid.shape[2])
    if near_x_end <= 0 or front_x_end <= front_x_start:
        step_delta = _zeros(env, dtype=grid.dtype)
    else:
        near_z = grid[:, body_y_start:body_y_end, :near_x_end].mean(dim=(1, 2))
        front_z = grid[:, body_y_start:body_y_end, front_x_start:front_x_end].mean(dim=(1, 2))
        step_delta = front_z - near_z

    wall_like = edge_sharpness > float(conf.get("nan_gate_wall_edge_threshold", 0.30))
    stair_like = (
        (edge_sharpness >= float(conf.get("nan_gate_stair_edge_threshold", 0.040)))
        & (edge_locality >= float(conf.get("nan_gate_stair_locality_threshold", 0.30)))
        & (slope_smoothness <= float(conf.get("nan_gate_stair_smoothness_max", 0.45)))
        & (~wall_like)
    )
    slope_like = (
        (edge_mean >= float(conf.get("nan_gate_slope_edge_mean_min", 0.006)))
        & (slope_smoothness >= float(conf.get("nan_gate_slope_smoothness_min", 0.60)))
        & (edge_sharpness <= float(conf.get("nan_gate_slope_edge_max", 0.070)))
        & (~wall_like)
        & (~stair_like)
    )
    direction_margin = float(conf.get("nan_gate_direction_margin", 0.02))
    up = stair_like & (step_delta > direction_margin)
    down = stair_like & (step_delta < -direction_margin)

    difficulty_signal = torch.where(stair_like, edge_sharpness, step_delta.abs())
    stair_low_thr = float(conf.get("stairs_difficulty_low_threshold", conf.get("nan_gate_difficulty_low_threshold", 0.09)))
    stair_high_thr = float(conf.get("stairs_difficulty_high_threshold", conf.get("nan_gate_difficulty_high_threshold", 0.16)))
    slope_low_thr = float(conf.get("slope_difficulty_low_threshold", conf.get("nan_gate_difficulty_low_threshold", 0.09)))
    slope_high_thr = float(conf.get("slope_difficulty_high_threshold", conf.get("nan_gate_difficulty_high_threshold", 0.16)))
    low_thr = torch.where(
        stair_like,
        torch.full_like(difficulty_signal, stair_low_thr),
        torch.full_like(difficulty_signal, slope_low_thr),
    )
    high_thr = torch.where(
        stair_like,
        torch.full_like(difficulty_signal, stair_high_thr),
        torch.full_like(difficulty_signal, slope_high_thr),
    )
    difficulty = torch.zeros(num_envs, dtype=torch.long, device=grid.device)
    active = stair_like | slope_like
    difficulty = torch.where(active & (difficulty_signal >= low_thr), torch.ones_like(difficulty), difficulty)
    difficulty = torch.where(active & (difficulty_signal >= high_thr), torch.full_like(difficulty, 2), difficulty)

    terrain_id = torch.zeros(num_envs, dtype=torch.long, device=grid.device)
    terrain_id = torch.where(slope_like, torch.ones_like(terrain_id), terrain_id)
    terrain_id = torch.where(stair_like, torch.full_like(terrain_id, 2), terrain_id)
    terrain_id = torch.where(wall_like, torch.full_like(terrain_id, 3), terrain_id)

    gate.update({
        "available": torch.ones(num_envs, device=grid.device, dtype=torch.float32),
        "terrain_id": terrain_id,
        "difficulty_band": difficulty,
        "flat": terrain_id == 0,
        "slope": slope_like,
        "stairs": stair_like,
        "wall": wall_like,
        "up": up,
        "down": down,
        "edge_sharpness": edge_sharpness,
        "edge_mean": edge_mean,
        "edge_locality": edge_locality,
        "slope_smoothness": slope_smoothness,
        "step_delta": step_delta,
        "difficulty_signal": difficulty_signal,
    })
    return gate


def _nav_wall_features(env, grid, conf):
    zeros = _zeros(env)
    if grid is None or grid.shape[1] < 2 or grid.shape[2] < 2:
        return {
            "available": zeros,
            "front_wall_score": zeros,
            "left_wall_score": zeros,
            "right_wall_score": zeros,
            "front_blocked": zeros,
            "open_side": zeros,
        }

    body_y_start = max(0, int(conf.get("nav_wall_body_y_start", max(0, grid.shape[1] // 2 - 3))))
    body_y_end = min(int(conf.get("nav_wall_body_y_end", min(grid.shape[1], grid.shape[1] // 2 + 3))), grid.shape[1])
    front_cols = max(1, min(int(conf.get("nav_wall_front_cols", 6)), grid.shape[2]))
    side_width = max(1, min(int(conf.get("nav_wall_side_width", 3)), grid.shape[1] // 2))
    if body_y_end <= body_y_start:
        return {
            "available": zeros,
            "front_wall_score": zeros,
            "left_wall_score": zeros,
            "right_wall_score": zeros,
            "front_blocked": zeros,
            "open_side": zeros,
        }

    threshold = float(conf.get("nav_wall_height_threshold", 0.24))
    temperature = max(float(conf.get("nav_wall_temperature", 0.08)), 1.0e-6)
    block_threshold = float(conf.get("nav_wall_score_threshold", 0.35))
    rel_grid = _relative_height_grid(
        grid,
        floor_quantile=float(conf.get("nav_wall_floor_quantile", 0.20)),
    )
    if rel_grid is None:
        return {
            "available": zeros,
            "front_wall_score": zeros,
            "left_wall_score": zeros,
            "right_wall_score": zeros,
            "front_blocked": zeros,
            "open_side": zeros,
        }

    def wall_score(sector):
        return torch.sigmoid((sector - threshold) / temperature).mean(dim=(1, 2))

    front_sector = rel_grid[:, body_y_start:body_y_end, :front_cols]
    left_sector = rel_grid[:, :side_width, :front_cols]
    right_sector = rel_grid[:, -side_width:, :front_cols]
    front_score = wall_score(front_sector)
    left_score = wall_score(left_sector)
    right_score = wall_score(right_sector)
    open_side = (1.0 - right_score) - (1.0 - left_score)
    return {
        "available": torch.ones(grid.shape[0], device=grid.device, dtype=torch.float32),
        "front_wall_score": front_score,
        "left_wall_score": left_score,
        "right_wall_score": right_score,
        "front_blocked": (front_score > block_threshold).float(),
        "open_side": open_side,
    }


def _current_gate(env, obs, conf, group="policy"):
    gate = _empty_gate(env, 0)
    grid = _obs_grid(obs, group)
    if grid is None:
        return gate
    row_start = max(0, int(conf.get("terrain_row_start", 3)))
    row_end = min(int(conf.get("terrain_row_end", 13)), grid.shape[1])
    front_cols = min(int(conf.get("terrain_front_cols", 8)), grid.shape[2])
    if row_end <= row_start or front_cols <= 1:
        return gate
    sector = grid[:, row_start:row_end, :front_cols]
    dx = sector[:, :, 1:] - sector[:, :, :-1]
    abs_dx = dx.abs()
    if abs_dx.numel() == 0:
        return gate
    lateral_std = sector.std(dim=1, unbiased=False).mean(dim=1)
    q = min(max(float(conf.get("terrain_step_quantile", 0.85)), 0.0), 1.0)
    step_strength = torch.quantile(abs_dx.flatten(1), q, dim=1)
    sign_consistency = dx.mean(dim=(1, 2)).abs() / (abs_dx.mean(dim=(1, 2)) + 1.0e-6)
    if dx.shape[2] > 1:
        second_diff = (dx[:, :, 1:] - dx[:, :, :-1]).abs().mean(dim=(1, 2))
    else:
        second_diff = torch.zeros(env.num_envs, device=env.device)
    is_uniform = lateral_std < float(conf.get("terrain_lateral_std_threshold", 0.18))
    not_wall = sector.amin(dim=(1, 2)) > float(conf.get("terrain_wall_height_threshold", -1.05))
    terrain_like = is_uniform & not_wall & (step_strength > float(conf.get("terrain_slope_delta_threshold", 0.035)))
    stair_like = terrain_like & (
        (step_strength > float(conf.get("terrain_stair_delta_threshold", 0.10)))
        | (second_diff > float(conf.get("terrain_stair_second_diff_threshold", 0.055)))
    )
    slope_like = terrain_like & (~stair_like) & (
        sign_consistency > float(conf.get("terrain_slope_sign_consistency_threshold", 0.55))
    )
    mid_thr = float(conf.get("terrain_difficulty_mid_threshold", 0.16))
    high_thr = float(conf.get("terrain_difficulty_high_threshold", 0.28))
    active = stair_like | slope_like
    difficulty = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    difficulty = torch.where(active & (step_strength >= mid_thr), torch.ones_like(difficulty), difficulty)
    difficulty = torch.where(active & (step_strength >= high_thr), torch.full_like(difficulty, 2), difficulty)
    terrain_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    terrain_id = torch.where(slope_like, torch.ones_like(terrain_id), terrain_id)
    terrain_id = torch.where(stair_like, torch.full_like(terrain_id, 2), terrain_id)
    gate.update({
        "available": torch.ones(env.num_envs, device=env.device),
        "terrain_id": terrain_id,
        "difficulty_band": difficulty,
        "flat": terrain_id == 0,
        "slope": slope_like,
        "stairs": stair_like,
        "wall": torch.zeros_like(stair_like),
        "difficulty_signal": step_strength,
    })
    return gate


def _maze_remaining(env, obs, conf, group="policy"):
    gate = float(conf.get("phase_maze_goal_dist_gate", 14.0))
    mode = str(conf.get("phase_maze_distance_mode", "euclidean")).lower()
    try:
        robot = env.scene["robot"]
        delta_xy = env.goal_positions[:, :2] - robot.data.root_pos_w[:, :2]
        if mode in ("longitudinal", "x", "track_x"):
            goal_yaw = getattr(env, "goal_yaw", None)
            if goal_yaw is None:
                forward_x = torch.ones(delta_xy.shape[0], device=delta_xy.device, dtype=delta_xy.dtype)
                forward_y = torch.zeros_like(forward_x)
            else:
                goal_yaw = goal_yaw.to(device=delta_xy.device, dtype=delta_xy.dtype)
                forward_x = torch.cos(goal_yaw)
                forward_y = torch.sin(goal_yaw)
            remaining = delta_xy[:, 0] * forward_x + delta_xy[:, 1] * forward_y
            return remaining, gate
        return torch.linalg.norm(delta_xy, dim=1), gate
    except Exception:
        pass
    default_goal_start = 316 if group == "critic" else 301
    goal_start = int(conf.get("goal_start", default_goal_start))
    if obs is not None and obs.shape[-1] >= goal_start + 3:
        return torch.clamp(obs[:, goal_start + 2], 0.0, 1.0) * 20.0, gate
    return None, gate


def _maze_phase(env, obs, conf, group="policy"):
    remaining, gate = _maze_remaining(env, obs, conf, group)
    if remaining is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    min_remaining = float(conf.get("phase_maze_goal_longitudinal_min", -1.0))
    return (remaining < gate) & (remaining > min_remaining)


def _pitch_abs(env):
    try:
        gravity = env.scene["robot"].data.projected_gravity_b
        return torch.atan2(
            torch.abs(gravity[:, 0]),
            torch.clamp(torch.abs(gravity[:, 2]), min=1.0e-6),
        )
    except Exception:
        return _zeros(env)


def _bool_state(shape, device):
    return torch.zeros(shape, dtype=torch.bool, device=device)


def _long_state(shape, device):
    return torch.zeros(shape, dtype=torch.long, device=device)


def _float_state(shape, device):
    return torch.zeros(shape, dtype=torch.float, device=device)


def _update_sticky_gate(env, obs, conf, instant_gate, maze_instant, state, group="policy"):
    terrain_id = instant_gate["terrain_id"].to(device=env.device).long()
    difficulty = instant_gate["difficulty_band"].to(device=env.device).long()
    difficulty_signal = instant_gate["difficulty_signal"].to(device=env.device).float()
    up = instant_gate["up"].to(device=env.device).bool()
    down = instant_gate["down"].to(device=env.device).bool()
    num_envs = terrain_id.shape[0]

    if "sticky_terrain_id" not in state or state["sticky_terrain_id"].shape[0] != num_envs:
        state["sticky_terrain_id"] = _long_state(num_envs, env.device)
        state["sticky_difficulty_band"] = _long_state(num_envs, env.device)
        state["sticky_difficulty_signal"] = _float_state(num_envs, env.device)
        state["sticky_up"] = _bool_state(num_envs, env.device)
        state["sticky_down"] = _bool_state(num_envs, env.device)
        state["sticky_hold"] = _long_state(num_envs, env.device)
        state["pending_terrain_id"] = _long_state(num_envs, env.device)
        state["pending_difficulty_band"] = _long_state(num_envs, env.device)
        state["pending_up"] = _bool_state(num_envs, env.device)
        state["pending_down"] = _bool_state(num_envs, env.device)
        state["pending_count"] = _long_state(num_envs, env.device)
        state["flat_release_count"] = _long_state(num_envs, env.device)
        state["maze_confirm_count"] = _long_state(num_envs, env.device)
        state["maze_sticky"] = _bool_state(num_envs, env.device)

    same_pending = (
        (terrain_id == state["pending_terrain_id"])
        & (difficulty == state["pending_difficulty_band"])
        & (up == state["pending_up"])
        & (down == state["pending_down"])
    )
    state["pending_count"] = torch.where(
        same_pending,
        state["pending_count"] + 1,
        torch.ones_like(state["pending_count"]),
    )
    state["pending_terrain_id"] = terrain_id.detach().clone()
    state["pending_difficulty_band"] = difficulty.detach().clone()
    state["pending_up"] = up.detach().clone()
    state["pending_down"] = down.detach().clone()

    confirm_steps = max(int(conf.get("terrain_sticky_confirm_steps", 4)), 1)
    flat_release_steps = max(int(conf.get("terrain_sticky_flat_release_steps", 10)), 1)
    slope_hold_steps = max(int(conf.get("terrain_sticky_slope_hold_steps", 30)), 1)
    stairs_hold_steps = max(int(conf.get("terrain_sticky_stairs_hold_steps", 50)), 1)
    wall_hold_steps = max(int(conf.get("terrain_sticky_wall_hold_steps", 40)), 1)

    active = terrain_id != 0
    confirmed = state["pending_count"] >= confirm_steps
    hold_steps = torch.full_like(state["sticky_hold"], slope_hold_steps)
    hold_steps = torch.where(terrain_id == 2, torch.full_like(hold_steps, stairs_hold_steps), hold_steps)
    hold_steps = torch.where(terrain_id == 3, torch.full_like(hold_steps, wall_hold_steps), hold_steps)
    state["sticky_hold"] = torch.clamp(state["sticky_hold"] - 1, min=0)

    previous_sticky_terrain = state["sticky_terrain_id"].detach().clone()
    previous_sticky_difficulty = state["sticky_difficulty_band"].detach().clone()
    apply_active = confirmed & active
    preempt_switch = (
        apply_active
        & (previous_sticky_terrain != 0)
        & (
            (terrain_id != previous_sticky_terrain)
            | (difficulty != previous_sticky_difficulty)
        )
    )
    state["sticky_terrain_id"] = torch.where(apply_active, terrain_id, state["sticky_terrain_id"])
    state["sticky_difficulty_band"] = torch.where(apply_active, difficulty, state["sticky_difficulty_band"])
    state["sticky_difficulty_signal"] = torch.where(
        apply_active,
        difficulty_signal,
        state["sticky_difficulty_signal"],
    )
    state["sticky_up"] = torch.where(apply_active, up, state["sticky_up"])
    state["sticky_down"] = torch.where(apply_active, down, state["sticky_down"])
    state["sticky_hold"] = torch.where(
        apply_active,
        torch.maximum(state["sticky_hold"], hold_steps),
        state["sticky_hold"],
    )

    edge_max = float(conf.get("terrain_flat_release_edge_max", 0.025))
    delta_max = float(conf.get("terrain_flat_release_delta_max", 0.025))
    pitch_max = float(conf.get("terrain_flat_release_pitch_max", 0.10))
    pitch_abs = _pitch_abs(env)
    flat_confident = (
        (terrain_id == 0)
        & (instant_gate["edge_sharpness"].to(device=env.device).abs() <= edge_max)
        & (instant_gate["step_delta"].to(device=env.device).abs() <= delta_max)
        & (pitch_abs <= pitch_max)
    )
    state["flat_release_count"] = torch.where(
        flat_confident,
        state["flat_release_count"] + 1,
        torch.zeros_like(state["flat_release_count"]),
    )
    release = (
        (state["sticky_terrain_id"] != 0)
        & (state["flat_release_count"] >= flat_release_steps)
    )
    state["sticky_terrain_id"] = torch.where(release, torch.zeros_like(state["sticky_terrain_id"]), state["sticky_terrain_id"])
    state["sticky_difficulty_band"] = torch.where(release, torch.zeros_like(state["sticky_difficulty_band"]), state["sticky_difficulty_band"])
    state["sticky_difficulty_signal"] = torch.where(release, torch.zeros_like(state["sticky_difficulty_signal"]), state["sticky_difficulty_signal"])
    state["sticky_up"] = torch.where(release, torch.zeros_like(state["sticky_up"]), state["sticky_up"])
    state["sticky_down"] = torch.where(release, torch.zeros_like(state["sticky_down"]), state["sticky_down"])
    state["flat_confident"] = flat_confident.detach().clone()
    state["early_flat_release"] = release.detach().clone()
    state["active_preempt_switch"] = preempt_switch.detach().clone()

    maze_confirm_steps = max(int(conf.get("phase_maze_confirm_steps", 3)), 1)
    state["maze_confirm_count"] = torch.where(
        maze_instant,
        state["maze_confirm_count"] + 1,
        torch.zeros_like(state["maze_confirm_count"]),
    )
    maze_sticky = state["maze_sticky"] | (state["maze_confirm_count"] >= maze_confirm_steps)
    remaining, gate = _maze_remaining(env, obs, conf, group)
    if remaining is not None:
        release_margin = float(conf.get("phase_maze_release_margin", 2.0))
        maze_sticky = maze_sticky & ~(remaining > (float(gate) + release_margin))
    state["maze_sticky"] = maze_sticky.detach().clone()

    return _sticky_gate_from_state(env, instant_gate, state, conf)


def _sticky_gate_from_state(env, instant_gate, state, conf):
    sticky = _copy_gate(env, instant_gate, source_id=4)
    sticky_terrain = state["sticky_terrain_id"]
    sticky["terrain_id"] = sticky_terrain.detach().clone()
    sticky["difficulty_band"] = state["sticky_difficulty_band"].detach().clone()
    sticky["difficulty_signal"] = state["sticky_difficulty_signal"].detach().clone()
    sticky["flat"] = sticky_terrain == 0
    sticky["slope"] = sticky_terrain == 1
    sticky["stairs"] = sticky_terrain == 2
    sticky["wall"] = sticky_terrain == 3
    sticky["up"] = state["sticky_up"].detach().clone()
    sticky["down"] = state["sticky_down"].detach().clone()
    sticky["maze"] = state["maze_sticky"].detach().clone()
    sticky["sticky_hold"] = state["sticky_hold"].float().detach().clone()
    sticky["pending_count"] = state["pending_count"].float().detach().clone()
    sticky["flat_release_count"] = state.get(
        "flat_release_count", torch.zeros_like(state["sticky_hold"])
    ).float().detach().clone()
    sticky["flat_confident"] = state.get(
        "flat_confident", torch.zeros_like(state["sticky_hold"], dtype=torch.bool)
    ).detach().clone()
    sticky["early_flat_release"] = state.get(
        "early_flat_release", torch.zeros_like(state["sticky_hold"], dtype=torch.bool)
    ).detach().clone()
    sticky["active_preempt_switch"] = state.get(
        "active_preempt_switch", torch.zeros_like(state["sticky_hold"], dtype=torch.bool)
    ).detach().clone()
    sticky["maze_confirm_count"] = state["maze_confirm_count"].float().detach().clone()
    sticky["pitch_abs"] = _pitch_abs(env).detach()
    sticky["updown_confidence"] = torch.clamp(
        torch.abs(instant_gate["step_delta"].to(device=env.device))
        / max(float(conf.get("terrain_updown_full_delta", 0.12)), 1.0e-6),
        0.0,
        1.0,
    ).detach()
    return sticky


def _compute_gates(env, obs, conf, maze_phase=None, group="policy"):
    raw_height_grid = _sensor_grid(env, "height_scanner")
    raw_nav_grid = _sensor_grid(env, "nav_scanner")
    current = _current_gate(env, obs, conf, group)
    nan = _nan_structure_gate(env, raw_height_grid if raw_height_grid is not None else _obs_grid(obs, group), conf)
    raw = _structure_gate(env, raw_height_grid, conf, "fused_gate", 2)
    nav = _structure_gate(env, raw_nav_grid, conf, "nav_gate", 2)
    nav_features = _nav_wall_features(env, raw_nav_grid, conf)
    if raw["available"].mean() > 0 and nav["available"].mean() > 0:
        nav_front_wall = nav_features["front_blocked"] > 0.5
        nav_wall = nav["wall"]
        if bool(conf.get("nav_wall_gate_in_maze_only", True)):
            if maze_phase is None:
                nav_front_wall = torch.zeros_like(nav_front_wall)
                nav_wall = torch.zeros_like(nav_wall)
            else:
                maze_mask = maze_phase.to(device=nav_front_wall.device).bool()
                nav_front_wall = nav_front_wall & maze_mask
                nav_wall = nav_wall & maze_mask
        wall = raw["wall"] | nav_wall | nav_front_wall
        raw["wall"] = wall
        raw["stairs"] = raw["stairs"] & (~wall)
        raw["slope"] = raw["slope"] & (~wall)
        raw["terrain_id"] = torch.where(wall, torch.full_like(raw["terrain_id"], 3), raw["terrain_id"])
        raw["flat"] = raw["terrain_id"] == 0
    raw["nav_available"] = nav["available"]
    raw["nav_front_wall_score"] = nav_features["front_wall_score"]
    raw["nav_left_wall_score"] = nav_features["left_wall_score"]
    raw["nav_right_wall_score"] = nav_features["right_wall_score"]
    raw["nav_front_blocked"] = nav_features["front_blocked"]
    raw["nav_open_side"] = nav_features["open_side"]
    return {"current": current, "nan": nan, "raw_fused": raw}


def _select_gate(gates, mode):
    mode = {
        "nan_sticky": "sticky",
        "raw_fused_sticky": "sticky",
        "current_sticky": "sticky",
    }.get(mode, mode)
    if mode not in gates:
        mode = "raw_fused"
    return gates[mode], mode


def _sticky_source_name(conf, mode):
    if mode == "nan_sticky":
        return "nan"
    if mode == "raw_fused_sticky":
        return "raw_fused"
    if mode == "current_sticky":
        return "current"
    return str(conf.get("terrain_sticky_source", "nan"))


def _get_state(env):
    state = getattr(env, "_worker_terrain_gate_state", None)
    cmd = _current_command(env)
    if state is None or state.get("command", None) is None or state["command"].shape != cmd.shape:
        state = {
            "command": cmd.clone(),
            "timer": torch.zeros(cmd.shape[0], dtype=torch.long, device=cmd.device),
            "last_step": None,
            "maze_phase": torch.zeros(cmd.shape[0], dtype=torch.bool, device=cmd.device),
            "terrain_id": torch.zeros(cmd.shape[0], dtype=torch.long, device=cmd.device),
            "difficulty_band": torch.zeros(cmd.shape[0], dtype=torch.long, device=cmd.device),
        }
        setattr(env, "_worker_terrain_gate_state", state)
    return state


def _sample_command(env, conf, maze_phase, gate, state, dtype):
    command = state["command"].clone()
    timer = torch.clamp(state["timer"] - 1, min=0)
    terrain_id = gate["terrain_id"]
    terrain_changed = terrain_id != state["terrain_id"]
    phase_changed = maze_phase != state["maze_phase"]
    needs_sample = (timer <= 0) | phase_changed | terrain_changed | (command[:, 0] <= 0.0)
    if needs_sample.any():
        fallback_vx = float(conf.get("suggested_speed_fallback", conf.get("phase_command_fallback_vx", 0.62)))
        fallback = torch.full((env.num_envs,), fallback_vx, device=env.device, dtype=dtype)
        pre = _sample_range(conf.get("pre_maze_lin_vel_x", [fallback_vx, fallback_vx]), (env.num_envs,), env.device, dtype)
        slope = _sample_range(conf.get("slope_lin_vel_x", [fallback_vx, fallback_vx]), (env.num_envs,), env.device, dtype)
        stairs = _sample_range(conf.get("stairs_lin_vel_x", [fallback_vx, fallback_vx]), (env.num_envs,), env.device, dtype)
        maze = _sample_range(conf.get("maze_lin_vel_x", [fallback_vx, fallback_vx]), (env.num_envs,), env.device, dtype)
        non_maze = pre
        if bool(conf.get("terrain_phase_speed_enabled", True)):
            non_maze = torch.where(terrain_id == 1, slope, non_maze)
            non_maze = torch.where(terrain_id == 2, stairs, non_maze)
            non_maze = torch.where(terrain_id == 3, fallback, non_maze)
        command[:, 0] = torch.where(maze_phase, maze, non_maze)
        command[:, 1] = 0.0
        command[:, 2] = 0.0
    state["command"] = command
    state["timer"] = torch.where(
        needs_sample,
        torch.full_like(timer, max(int(conf.get("phase_command_resample_steps", 160)), 1)),
        timer,
    )
    state["maze_phase"] = maze_phase.detach().clone()
    state["terrain_id"] = terrain_id.detach().clone()
    return command


def apply_worker_gate_command(env, obs, group):
    """Compute all gates, optionally publish terrain-conditioned command, patch obs."""
    conf = _rl_conf()
    mode = str(conf.get("gate_test_mode", "raw_fused"))
    diagnostics_enabled = _diagnostics_enabled(conf)
    reward_metrics_enabled = bool(conf.get("gate_reward_metrics_enabled", True))
    enabled = (
        bool(conf.get("worker_phase_command_enabled", True))
        and bool(conf.get("phase_command_enabled", False))
        and bool(conf.get("gate_speed_advice_enabled", True))
    )
    if bool(getattr(env, "_is_eval", False)) and not enabled and not diagnostics_enabled:
        return obs
    if not enabled and not diagnostics_enabled and not reward_metrics_enabled:
        return obs

    state = _get_state(env)
    step = _step_key(env)
    cache = state.get("gate_cache")
    if step is not None and cache is not None and cache.get("step") == step:
        maze_instant = cache["maze_instant"]
        maze_phase = cache["maze_phase"]
        gates = cache["gates"]
        sticky = gates["sticky"]
    else:
        maze_instant = _maze_phase(env, obs, conf, group)
        gates = _compute_gates(env, obs, conf, maze_instant, group)
        sticky_source_name = _sticky_source_name(conf, mode)
        sticky_source = gates.get(sticky_source_name, gates["nan"])
        if step is None or state.get("last_gate_step") != step or "sticky_terrain_id" not in state:
            sticky = _update_sticky_gate(env, obs, conf, sticky_source, maze_instant, state, group)
            state["last_gate_step"] = step
        else:
            sticky = _sticky_gate_from_state(env, sticky_source, state, conf)
        maze_phase = sticky["maze"] if bool(conf.get("phase_maze_sticky_until_done", True)) else maze_instant
        if bool(conf.get("nav_wall_gate_in_maze_only", True)):
            gates = _compute_gates(env, obs, conf, maze_phase, group)
            sticky_source = gates.get(sticky_source_name, gates["nan"])
            sticky = _sticky_gate_from_state(env, sticky_source, state, conf)
        gates["sticky"] = sticky
        if step is not None:
            state["gate_cache"] = {
                "step": step,
                "maze_instant": maze_instant,
                "maze_phase": maze_phase,
                "gates": gates,
            }
    selected, selected_name = _select_gate(gates, mode if mode != "shadow" else str(conf.get("shadow_selected_gate", "sticky")))
    current_cmd = _current_command(env)
    command = current_cmd.clone()
    should_update = step is None or state.get("last_step") != step
    command_written = False
    if enabled and mode != "shadow" and should_update:
        command = _sample_command(env, conf, maze_phase, selected, state, obs.dtype)
        command_written = _write_command(env, command)
        state["last_step"] = step
    elif enabled and mode != "shadow":
        command = state["command"]
        command_written = _write_command(env, command)
    else:
        state["command"] = current_cmd.clone()

    patched = obs
    if enabled and mode != "shadow":
        patched = obs.clone()
        if group == "policy" and patched.shape[-1] >= 9:
            patched[:, 6:9] = command.to(device=patched.device, dtype=patched.dtype)
        elif group == "critic" and patched.shape[-1] >= 12:
            patched[:, 9:12] = command.to(device=patched.device, dtype=patched.dtype)

    if group == "policy" and patched.shape[-1] >= 9:
        state["policy_cmd_vx"] = patched[:, 6].detach().clone()
    if group == "critic" and patched.shape[-1] >= 12:
        state["critic_cmd_vx"] = patched[:, 9].detach().clone()

    source_value = {"current": 0.0, "nan": 1.0, "raw_fused": 2.0, "sticky": 4.0}.get(selected_name, -1.0)
    selected_maze = maze_phase.float()
    not_maze = (~maze_phase).float()
    selected_flat = selected["flat"].float() * not_maze
    selected_slope = selected["slope"].float() * not_maze
    selected_stairs = selected["stairs"].float() * not_maze
    selected_wall = selected["wall"].float() * not_maze
    final_flat = selected_flat
    final_slope = selected_slope
    final_stairs = selected_stairs
    final_maze = selected_maze
    final_invalid = selected_wall
    final_terrain_sum = final_flat + final_slope + final_stairs + final_maze + final_invalid
    final_non_maze_sum = final_flat + final_slope + final_stairs + final_invalid
    final_active = final_slope + final_stairs
    final_difficulty = selected["difficulty_band"].float() * final_active
    final_difficulty_signal = selected["difficulty_signal"].float() * final_active
    final_difficulty_low = (selected["difficulty_band"] == 0).float() * final_active
    final_difficulty_mid = (selected["difficulty_band"] == 1).float() * final_active
    final_difficulty_high = (selected["difficulty_band"] >= 2).float() * final_active
    final_difficulty_unknown = final_flat + final_maze + final_invalid
    final_source_current = torch.full((env.num_envs,), 1.0 if selected_name == "current" else 0.0, device=env.device)
    final_source_nan = torch.full((env.num_envs,), 1.0 if selected_name == "nan" else 0.0, device=env.device)
    final_source_raw = torch.full((env.num_envs,), 1.0 if selected_name == "raw_fused" else 0.0, device=env.device)
    final_source_sticky = torch.full((env.num_envs,), 1.0 if selected_name == "sticky" else 0.0, device=env.device)
    final_terrain_id = torch.where(
        maze_phase,
        torch.full_like(selected["terrain_id"], 3),
        selected["terrain_id"],
    )
    final_terrain_id = torch.where(selected["wall"] & (~maze_phase), torch.full_like(final_terrain_id, 4), final_terrain_id)
    final_difficulty_id = torch.full_like(selected["difficulty_band"], -1)
    final_difficulty_id = torch.where(final_active.bool(), selected["difficulty_band"], final_difficulty_id)
    previous_final_terrain = state.get("final_terrain_id")
    previous_final_difficulty = state.get("final_difficulty_id")
    if previous_final_terrain is None or previous_final_terrain.shape != final_terrain_id.shape:
        terrain_switch = torch.zeros(env.num_envs, device=env.device)
    else:
        terrain_switch = (previous_final_terrain != final_terrain_id).float()
    if previous_final_difficulty is None or previous_final_difficulty.shape != final_difficulty_id.shape:
        difficulty_switch = torch.zeros(env.num_envs, device=env.device)
    else:
        difficulty_switch = (previous_final_difficulty != final_difficulty_id).float()
    state["final_terrain_id"] = final_terrain_id.detach().clone()
    state["final_difficulty_id"] = final_difficulty_id.detach().clone()
    minimal_diagnostics = {
        "selected_stairs": selected_stairs.detach(),
        "selected_difficulty": selected["difficulty_band"].float().detach(),
        "current_stairs": gates["current"]["stairs"].float().detach(),
        "current_difficulty": gates["current"]["difficulty_band"].float().detach(),
        "final_flat": final_flat.detach(),
        "final_slope": final_slope.detach(),
        "final_stairs": final_stairs.detach(),
        "final_maze": final_maze.detach(),
        "final_invalid": final_invalid.detach(),
        "final_terrain_sum": final_terrain_sum.detach(),
        "final_non_maze_sum": final_non_maze_sum.detach(),
        "final_active": final_active.detach(),
        "final_difficulty": final_difficulty.detach(),
        "final_difficulty_signal": final_difficulty_signal.detach(),
        "final_difficulty_low": final_difficulty_low.detach(),
        "final_difficulty_mid": final_difficulty_mid.detach(),
        "final_difficulty_high": final_difficulty_high.detach(),
        "final_difficulty_unknown": final_difficulty_unknown.detach(),
        "final_difficulty_sum": final_active.detach(),
        "final_gate_valid": ((selected["available"].float() > 0.5) & (final_terrain_sum > 0.5)).float().detach(),
        "nav_wall_front_score": gates["raw_fused"]["nav_front_wall_score"].float().detach(),
        "nav_wall_left_score": gates["raw_fused"]["nav_left_wall_score"].float().detach(),
        "nav_wall_right_score": gates["raw_fused"]["nav_right_wall_score"].float().detach(),
        "nav_wall_front_blocked": gates["raw_fused"]["nav_front_blocked"].float().detach(),
        "high_stair_active": (
            selected["stairs"] & (selected["difficulty_band"] >= 2)
        ).float().detach(),
        "mode_shadow": torch.full((env.num_envs,), 1.0 if mode == "shadow" else 0.0, device=env.device),
        "mode_control": torch.full((env.num_envs,), 0.0 if mode == "shadow" else 1.0, device=env.device),
        "selected_source": torch.full((env.num_envs,), source_value, device=env.device),
        "selected_available": selected["available"].float().detach(),
        "selected_terrain_sum": final_terrain_sum.detach(),
        "selected_maze": selected_maze.detach(),
        "selected_slope": selected_slope.detach(),
        "selected_stairs": selected_stairs.detach(),
        "selected_wall": selected_wall.detach(),
        "selected_difficulty_low": final_difficulty_low.detach(),
        "selected_difficulty_mid": final_difficulty_mid.detach(),
        "selected_difficulty_high": final_difficulty_high.detach(),
        "selected_difficulty_sum": final_active.detach(),
        "sticky_hold_steps": selected.get("sticky_hold", torch.zeros(env.num_envs, device=env.device)).float().detach(),
        "sticky_pending_count": selected.get("pending_count", torch.zeros(env.num_envs, device=env.device)).float().detach(),
        "maze_instant": maze_instant.float().detach(),
        "maze_confirm_count": selected.get("maze_confirm_count", torch.zeros(env.num_envs, device=env.device)).float().detach(),
        "maze_phase": maze_phase.float().detach(),
        "command_written": torch.full((env.num_envs,), 1.0 if command_written else 0.0, device=env.device),
        "worker_cmd_vx": _current_command(env)[:, 0].float().detach(),
        "target_cmd_vx": command[:, 0].float().detach(),
        "raw_nav_available": gates["raw_fused"]["nav_available"].float().detach(),
        "raw_nav_front_wall_score": gates["raw_fused"]["nav_front_wall_score"].float().detach(),
        "raw_nav_left_wall_score": gates["raw_fused"]["nav_left_wall_score"].float().detach(),
        "raw_nav_right_wall_score": gates["raw_fused"]["nav_right_wall_score"].float().detach(),
        "raw_nav_front_blocked": gates["raw_fused"]["nav_front_blocked"].float().detach(),
        "raw_nav_open_side": gates["raw_fused"]["nav_open_side"].float().detach(),
    }
    if not diagnostics_enabled:
        state["diagnostics"] = minimal_diagnostics
        return patched

    state["diagnostics"] = {
        "mode_shadow": torch.full((env.num_envs,), 1.0 if mode == "shadow" else 0.0, device=env.device),
        "mode_control": torch.full((env.num_envs,), 0.0 if mode == "shadow" else 1.0, device=env.device),
        "selected_source": torch.full((env.num_envs,), source_value, device=env.device),
        "selected_available": selected["available"].float().detach(),
        "selected_flat": selected_flat.detach(),
        "selected_slope": selected_slope.detach(),
        "selected_stairs": selected_stairs.detach(),
        "selected_wall": selected_wall.detach(),
        "selected_maze": selected_maze.detach(),
        "selected_terrain_sum": final_terrain_sum.detach(),
        "selected_up": (selected["up"].float() * not_maze).detach(),
        "selected_down": (selected["down"].float() * not_maze).detach(),
        "selected_difficulty": selected["difficulty_band"].float().detach(),
        "selected_difficulty_signal": final_difficulty_signal.detach(),
        "selected_difficulty_low": final_difficulty_low.detach(),
        "selected_difficulty_mid": final_difficulty_mid.detach(),
        "selected_difficulty_high": final_difficulty_high.detach(),
        "selected_difficulty_sum": final_active.detach(),
        "final_flat": final_flat.detach(),
        "final_slope": final_slope.detach(),
        "final_stairs": final_stairs.detach(),
        "final_maze": final_maze.detach(),
        "final_invalid": final_invalid.detach(),
        "final_terrain_sum": final_terrain_sum.detach(),
        "final_non_maze_sum": final_non_maze_sum.detach(),
        "final_active": final_active.detach(),
        "final_difficulty": final_difficulty.detach(),
        "final_difficulty_signal": final_difficulty_signal.detach(),
        "final_difficulty_low": final_difficulty_low.detach(),
        "final_difficulty_mid": final_difficulty_mid.detach(),
        "final_difficulty_high": final_difficulty_high.detach(),
        "final_difficulty_unknown": final_difficulty_unknown.detach(),
        "final_difficulty_sum": final_active.detach(),
        "final_source_current": final_source_current.detach(),
        "final_source_nan": final_source_nan.detach(),
        "final_source_raw": final_source_raw.detach(),
        "final_source_sticky": final_source_sticky.detach(),
        "final_gate_valid": ((selected["available"].float() > 0.5) & (final_terrain_sum > 0.5)).float().detach(),
        "terrain_switch": terrain_switch.detach(),
        "difficulty_switch": difficulty_switch.detach(),
        "sticky_hold_steps": sticky["sticky_hold"].float().detach(),
        "sticky_pending_count": sticky["pending_count"].float().detach(),
        "sticky_flat_release_count": sticky["flat_release_count"].float().detach(),
        "sticky_flat_confident": sticky["flat_confident"].float().detach(),
        "sticky_early_flat_release": sticky["early_flat_release"].float().detach(),
        "sticky_active_preempt_switch": sticky["active_preempt_switch"].float().detach(),
        "maze_instant": maze_instant.float().detach().clone(),
        "maze_confirm_count": sticky["maze_confirm_count"].float().detach(),
        "stair_up": (selected["up"].float() * not_maze).detach(),
        "stair_down": (selected["down"].float() * not_maze).detach(),
        "stair_unknown_dir": (
            (selected["stairs"] & (~selected["up"]) & (~selected["down"])).float() * not_maze
        ).detach(),
        "pitch_abs": sticky["pitch_abs"].float().detach(),
        "updown_confidence": sticky["updown_confidence"].float().detach(),
        "command_written": torch.full((env.num_envs,), 1.0 if command_written else 0.0, device=env.device),
        "worker_cmd_vx": _current_command(env)[:, 0].detach().clone(),
        "target_cmd_vx": command[:, 0].detach().clone(),
        "maze_phase": maze_phase.float().detach().clone(),
    }
    state["diagnostics"].update(minimal_diagnostics)
    for name, gate in gates.items():
        prefix = f"{name}_"
        flat = gate["flat"].float()
        slope = gate["slope"].float()
        stairs = gate["stairs"].float()
        wall = gate["wall"].float()
        difficulty = gate["difficulty_band"]
        active = gate["slope"] | gate["stairs"] | gate["wall"]
        difficulty_active = gate["slope"] | gate["stairs"]
        state["diagnostics"][prefix + "available"] = gate["available"].float().detach()
        state["diagnostics"][prefix + "flat"] = flat.detach()
        state["diagnostics"][prefix + "slope"] = slope.detach()
        state["diagnostics"][prefix + "stairs"] = stairs.detach()
        state["diagnostics"][prefix + "wall"] = wall.detach()
        state["diagnostics"][prefix + "active"] = active.float().detach()
        state["diagnostics"][prefix + "terrain_sum"] = (flat + slope + stairs + wall).detach()
        state["diagnostics"][prefix + "up"] = gate["up"].float().detach()
        state["diagnostics"][prefix + "down"] = gate["down"].float().detach()
        state["diagnostics"][prefix + "difficulty"] = difficulty.float().detach()
        state["diagnostics"][prefix + "difficulty_low"] = (difficulty == 0).float().detach()
        state["diagnostics"][prefix + "difficulty_mid"] = (difficulty == 1).float().detach()
        state["diagnostics"][prefix + "difficulty_high"] = (difficulty >= 2).float().detach()
        state["diagnostics"][prefix + "difficulty_sum"] = torch.ones(env.num_envs, device=env.device)
        state["diagnostics"][prefix + "active_difficulty_low"] = (difficulty_active & (difficulty == 0)).float().detach()
        state["diagnostics"][prefix + "active_difficulty_mid"] = (difficulty_active & (difficulty == 1)).float().detach()
        state["diagnostics"][prefix + "active_difficulty_high"] = (difficulty_active & (difficulty >= 2)).float().detach()
        state["diagnostics"][prefix + "active_difficulty_sum"] = difficulty_active.float().detach()
        for key in (
            "edge_sharpness",
            "edge_mean",
            "edge_locality",
            "slope_smoothness",
            "step_delta",
            "difficulty_signal",
        ):
            state["diagnostics"][prefix + key] = gate[key].float().detach()
        if name == "raw_fused":
            state["diagnostics"]["raw_nav_available"] = gate["nav_available"].float().detach()
            state["diagnostics"]["raw_nav_front_wall_score"] = gate["nav_front_wall_score"].float().detach()
            state["diagnostics"]["raw_nav_left_wall_score"] = gate["nav_left_wall_score"].float().detach()
            state["diagnostics"]["raw_nav_right_wall_score"] = gate["nav_right_wall_score"].float().detach()
            state["diagnostics"]["raw_nav_front_blocked"] = gate["nav_front_blocked"].float().detach()
            state["diagnostics"]["raw_nav_open_side"] = gate["nav_open_side"].float().detach()
    current = gates["current"]
    nan = gates["nan"]
    raw = gates["raw_fused"]
    state["diagnostics"]["current_nan_stair_disagree"] = (
        current["stairs"] != nan["stairs"]
    ).float().detach()
    state["diagnostics"]["current_raw_stair_disagree"] = (
        current["stairs"] != raw["stairs"]
    ).float().detach()
    state["diagnostics"]["nan_raw_stair_disagree"] = (
        nan["stairs"] != raw["stairs"]
    ).float().detach()
    state["diagnostics"]["current_nan_slope_disagree"] = (
        current["slope"] != nan["slope"]
    ).float().detach()
    state["diagnostics"]["current_raw_slope_disagree"] = (
        current["slope"] != raw["slope"]
    ).float().detach()
    state["diagnostics"]["nan_raw_slope_disagree"] = (
        nan["slope"] != raw["slope"]
    ).float().detach()
    state["diagnostics"]["current_nan_wall_disagree"] = (
        current["wall"] != nan["wall"]
    ).float().detach()
    state["diagnostics"]["current_raw_wall_disagree"] = (
        current["wall"] != raw["wall"]
    ).float().detach()
    state["diagnostics"]["nan_raw_wall_disagree"] = (
        nan["wall"] != raw["wall"]
    ).float().detach()
    state["diagnostics"]["raw_wall_non_maze"] = (raw["wall"] & (~maze_phase)).float().detach()
    state["diagnostics"]["high_stair_active"] = (
        selected["stairs"] & (selected["difficulty_band"] >= 2)
    ).float().detach()
    return patched


def _diagnostic_value(env, diagnostics, name):
    value = diagnostics.get(name)
    if value is None:
        return _zeros(env)
    if not torch.is_tensor(value):
        return torch.full((env.num_envs,), float(value), device=env.device)
    return value.to(device=env.device).float()


def worker_gate_monitor_stats(env):
    """Return worker-side gate diagnostics as direct monitor metrics.

    These values are not reward terms, so they are not diluted by tiny diagnostic
    weights and can be read as raw ratios/means on the dashboard.
    """
    state = getattr(env, "_worker_terrain_gate_state", None)
    if not state:
        return {}
    diagnostics = state.get("diagnostics", {})
    if not diagnostics:
        return {}

    stats = {}

    def add(metric_name, diagnostic_name=None):
        diagnostic_name = diagnostic_name or metric_name
        stats[f"gate_worker_{metric_name}"] = _diagnostic_value(env, diagnostics, diagnostic_name)

    for metric_name, diagnostic_name in (
        ("mode_shadow_ratio", "mode_shadow"),
        ("selected_source", "selected_source"),
        ("selected_available_ratio", "selected_available"),
        ("selected_flat_ratio", "selected_flat"),
        ("selected_slope_ratio", "selected_slope"),
        ("selected_stairs_ratio", "selected_stairs"),
        ("selected_wall_ratio", "selected_wall"),
        ("selected_maze_ratio", "selected_maze"),
        ("selected_terrain_sum_ratio", "selected_terrain_sum"),
        ("selected_up_ratio", "selected_up"),
        ("selected_down_ratio", "selected_down"),
        ("selected_difficulty_mean", "selected_difficulty"),
        ("selected_difficulty_low_ratio", "selected_difficulty_low"),
        ("selected_difficulty_mid_ratio", "selected_difficulty_mid"),
        ("selected_difficulty_high_ratio", "selected_difficulty_high"),
        ("selected_difficulty_sum_ratio", "selected_difficulty_sum"),
        ("sticky_hold_steps", "sticky_hold_steps"),
        ("sticky_pending_count", "sticky_pending_count"),
        ("maze_instant_ratio", "maze_instant"),
        ("maze_confirm_count", "maze_confirm_count"),
        ("stair_up_ratio", "stair_up"),
        ("stair_down_ratio", "stair_down"),
        ("stair_unknown_dir_ratio", "stair_unknown_dir"),
        ("pitch_abs", "pitch_abs"),
        ("updown_confidence", "updown_confidence"),
        ("command_written_ratio", "command_written"),
        ("worker_cmd_vx", "worker_cmd_vx"),
        ("target_cmd_vx", "target_cmd_vx"),
        ("maze_phase_ratio", "maze_phase"),
        ("raw_nav_available_ratio", "raw_nav_available"),
        ("raw_nav_front_wall_score", "raw_nav_front_wall_score"),
        ("raw_nav_left_wall_score", "raw_nav_left_wall_score"),
        ("raw_nav_right_wall_score", "raw_nav_right_wall_score"),
        ("raw_nav_front_blocked_ratio", "raw_nav_front_blocked"),
        ("raw_nav_open_side", "raw_nav_open_side"),
    ):
        add(metric_name, diagnostic_name)

    for metric_name, diagnostic_name in (
        ("final_flat_ratio", "final_flat"),
        ("final_slope_ratio", "final_slope"),
        ("final_stairs_ratio", "final_stairs"),
        ("final_maze_ratio", "final_maze"),
        ("final_invalid_ratio", "final_invalid"),
        ("final_terrain_sum_ratio", "final_terrain_sum"),
        ("final_non_maze_sum_ratio", "final_non_maze_sum"),
        ("final_active_ratio", "final_active"),
        ("final_difficulty_mean", "final_difficulty"),
        ("final_difficulty_signal", "final_difficulty_signal"),
        ("final_difficulty_low_ratio", "final_difficulty_low"),
        ("final_difficulty_mid_ratio", "final_difficulty_mid"),
        ("final_difficulty_high_ratio", "final_difficulty_high"),
        ("final_difficulty_unknown_ratio", "final_difficulty_unknown"),
        ("final_difficulty_sum_ratio", "final_difficulty_sum"),
        ("final_gate_valid_ratio", "final_gate_valid"),
        ("final_source_current_ratio", "final_source_current"),
        ("final_source_nan_ratio", "final_source_nan"),
        ("final_source_raw_ratio", "final_source_raw"),
        ("final_source_sticky_ratio", "final_source_sticky"),
    ):
        add(metric_name, diagnostic_name)

    for gate_name, diagnostic_prefix in (
        ("current", "current"),
        ("nan", "nan"),
        ("raw", "raw_fused"),
        ("sticky", "sticky"),
    ):
        for key in (
            "available",
            "flat",
            "slope",
            "stairs",
            "wall",
            "active",
            "terrain_sum",
            "up",
            "down",
            "difficulty_low",
            "difficulty_mid",
            "difficulty_high",
            "difficulty_sum",
            "active_difficulty_low",
            "active_difficulty_mid",
            "active_difficulty_high",
            "active_difficulty_sum",
        ):
            add(f"{gate_name}_{key}_ratio", f"{diagnostic_prefix}_{key}")
        for key in (
            "difficulty",
            "edge_sharpness",
            "edge_mean",
            "edge_locality",
            "slope_smoothness",
            "step_delta",
            "difficulty_signal",
        ):
            add(f"{gate_name}_{key}_mean", f"{diagnostic_prefix}_{key}")

    for metric_name in (
        "current_nan_stair_disagree",
        "current_raw_stair_disagree",
        "nan_raw_stair_disagree",
        "current_nan_slope_disagree",
        "current_raw_slope_disagree",
        "nan_raw_slope_disagree",
        "current_nan_wall_disagree",
        "current_raw_wall_disagree",
        "nan_raw_wall_disagree",
        "raw_wall_non_maze",
        "high_stair_active",
    ):
        add(f"{metric_name}_ratio", metric_name)

    stats["gate_worker_policy_cmd_error"] = command_sync_error(env, "policy_cmd_vx")
    stats["gate_worker_critic_cmd_error"] = command_sync_error(env, "critic_cmd_vx")
    stats["gate_worker_actual_minus_cmd_vx"] = actual_minus_command(env)
    return stats


def gate_metric(env, name):
    state = getattr(env, "_worker_terrain_gate_state", None)
    if not state:
        return _zeros(env)
    diagnostics = state.get("diagnostics", {})
    value = diagnostics.get(name)
    if value is None:
        return _zeros(env)
    return value.float()


def command_sync_error(env, source):
    state = getattr(env, "_worker_terrain_gate_state", None)
    if not state:
        return _zeros(env)
    worker = _current_command(env)[:, 0]
    other = state.get(source)
    if other is None:
        return _zeros(env)
    return torch.abs(worker - other.to(device=worker.device, dtype=worker.dtype))


def actual_minus_command(env):
    try:
        robot = env.scene["robot"]
        return robot.data.root_lin_vel_b[:, 0] - _current_command(env)[:, 0]
    except Exception:
        return _zeros(env)
