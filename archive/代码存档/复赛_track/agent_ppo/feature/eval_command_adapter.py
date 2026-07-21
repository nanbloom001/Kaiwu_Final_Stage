# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Evaluation-time command adaptation for Track policy observations."""

import torch

from agent_ppo.feature.phase_semantics import (
    estimate_maze_phase_from_goal_feature,
    nan_terrain_gate_from_height_scan,
)


def build_track_eval_phase_command_profile(rl_nav_conf: dict, device):
    """Build a minimal evaluation command profile from TrackNav config."""
    profile = {
        "enabled": bool(rl_nav_conf.get("phase_command_enabled", False)),
        "rl_nav_conf": rl_nav_conf.copy(),
        "fallback_command": None,
        "pre_maze_command": None,
        "slope_command": None,
        "stairs_command": None,
        "maze_command": None,
    }

    fallback_vx = float(rl_nav_conf.get("suggested_speed_fallback", rl_nav_conf.get("phase_command_fallback_vx", 0.62)))
    profile["fallback_command"] = torch.tensor([fallback_vx, 0.0, 0.0], device=device, dtype=torch.float32)

    for key, field in (
        ("pre_maze_lin_vel_x", "pre_maze_command"),
        ("slope_lin_vel_x", "slope_command"),
        ("stairs_lin_vel_x", "stairs_command"),
        ("maze_lin_vel_x", "maze_command"),
    ):
        values = rl_nav_conf.get(key, None)
        if isinstance(values, (list, tuple)) and len(values) == 2:
            profile[field] = _build_midpoint_command(values, device)
    return profile


def _build_midpoint_command(range_values, device):
    return torch.tensor(
        [0.5 * (float(range_values[0]) + float(range_values[1])), 0.0, 0.0],
        device=device,
        dtype=torch.float32,
    )


def apply_track_eval_phase_command_to_obs(
    obs,
    rl_nav_conf: dict,
    pre_maze_command,
    slope_command,
    stairs_command,
    maze_command,
):
    """Patch only the policy-visible command observation for Track evaluation."""
    if obs is None or obs.shape[-1] < 9:
        return obs
    if pre_maze_command is None or maze_command is None:
        return obs

    maze_phase = estimate_maze_phase_from_goal_feature(obs, rl_nav_conf)
    if maze_phase is None:
        return obs

    fallback_vx = float(rl_nav_conf.get("suggested_speed_fallback", rl_nav_conf.get("phase_command_fallback_vx", 0.62)))
    fallback = torch.tensor([fallback_vx, 0.0, 0.0], device=obs.device, dtype=obs.dtype).expand(obs.shape[0], -1)
    command = fallback
    if bool(rl_nav_conf.get("terrain_phase_speed_enabled", False)):
        gate = nan_terrain_gate_from_height_scan(obs, rl_nav_conf)
        if gate is not None:
            terrain_id = gate["terrain_id"]
            valid_terrain = gate["available"].bool() & (terrain_id != 3)
            pre = pre_maze_command.to(device=obs.device, dtype=obs.dtype).expand(obs.shape[0], -1)
            command = command.where((~valid_terrain).unsqueeze(1), pre)
            if slope_command is not None:
                slope = slope_command.to(device=obs.device, dtype=obs.dtype).expand(obs.shape[0], -1)
                command = command.where((terrain_id != 1).unsqueeze(1), slope)
            if stairs_command is not None:
                stairs = stairs_command.to(device=obs.device, dtype=obs.dtype).expand(obs.shape[0], -1)
                command = command.where((terrain_id != 2).unsqueeze(1), stairs)
    else:
        command = pre_maze_command.to(device=obs.device, dtype=obs.dtype).expand(obs.shape[0], -1)

    maze = maze_command.to(device=obs.device, dtype=obs.dtype).expand(obs.shape[0], -1)
    command = command.where((~maze_phase).unsqueeze(1), maze)
    nav_obs = obs.clone()
    nav_obs[:, 6:9] = command
    return nav_obs


# Backward-compatible aliases for older Agent imports.
build_track_eval_command_profile = build_track_eval_phase_command_profile
apply_track_eval_command_to_obs = apply_track_eval_phase_command_to_obs
