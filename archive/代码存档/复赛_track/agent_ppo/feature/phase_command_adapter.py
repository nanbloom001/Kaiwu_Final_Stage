# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Track phase-command scheduling from public observations."""

import torch

from agent_ppo.feature.terrain_gate import worker_gate_monitor_stats
from agent_ppo.feature.phase_semantics import (
    estimate_maze_phase_from_goal_feature,
    nan_terrain_gate_from_height_scan,
)


def _range_midpoint(range_values, default_value: float) -> float:
    if not isinstance(range_values, (list, tuple)) or len(range_values) != 2:
        return default_value
    return 0.5 * (float(range_values[0]) + float(range_values[1]))


def _fallback_speed(rl_nav_conf) -> float:
    return float(rl_nav_conf.get("suggested_speed_fallback", rl_nav_conf.get("phase_command_fallback_vx", 0.62)))


def _sample_uniform_range(range_values, shape, device, dtype):
    low = float(range_values[0])
    high = float(range_values[1])
    if high < low:
        low, high = high, low
    if abs(high - low) <= 1e-8:
        return torch.full(shape, low, device=device, dtype=dtype)
    return low + (high - low) * torch.rand(shape, device=device, dtype=dtype)


def _build_midpoint_track_phase_command(obs, maze_phase, rl_nav_conf, dtype):
    fallback_vx = _fallback_speed(rl_nav_conf)
    pre_range = rl_nav_conf.get("pre_maze_lin_vel_x", [fallback_vx, fallback_vx])
    slope_range = rl_nav_conf.get("slope_lin_vel_x", pre_range)
    stairs_range = rl_nav_conf.get("stairs_lin_vel_x", pre_range)
    maze_range = rl_nav_conf.get("maze_lin_vel_x", [fallback_vx, fallback_vx])
    terrain_phase_speed_enabled = bool(rl_nav_conf.get("terrain_phase_speed_enabled", False))
    pre_vx = _range_midpoint(pre_range, fallback_vx)
    slope_vx = _range_midpoint(slope_range, pre_vx)
    stairs_vx = _range_midpoint(stairs_range, pre_vx)
    maze_vx = _range_midpoint(maze_range, fallback_vx)
    command = torch.zeros(maze_phase.shape[0], 3, device=maze_phase.device, dtype=dtype)
    pre_command = torch.full_like(command[:, 0], pre_vx)
    if terrain_phase_speed_enabled and obs is not None:
        pre_command = torch.full_like(command[:, 0], fallback_vx)
        gate = nan_terrain_gate_from_height_scan(obs, rl_nav_conf)
        if gate is not None:
            terrain_id = gate["terrain_id"]
            pre_command = torch.full_like(pre_command, pre_vx)
            pre_command = torch.where(terrain_id == 1, torch.full_like(pre_command, slope_vx), pre_command)
            pre_command = torch.where(terrain_id == 2, torch.full_like(pre_command, stairs_vx), pre_command)
    command[:, 0] = torch.where(maze_phase, torch.full_like(command[:, 0], maze_vx), pre_command)
    return command


def _get_track_phase_command_state(env, num_envs, device, dtype):
    state = getattr(env, "_rl_phase_command_state", None)
    if (
        state is None
        or state["command"].shape[0] != num_envs
        or state["command"].device != device
        or state["command"].dtype != dtype
    ):
        state = {
            "command": torch.zeros(num_envs, 3, device=device, dtype=dtype),
            "timer": torch.zeros(num_envs, dtype=torch.long, device=device),
            "maze_phase": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "terrain_id": torch.zeros(num_envs, dtype=torch.long, device=device),
            "gate_available": torch.zeros(num_envs, dtype=torch.bool, device=device),
        }
        setattr(env, "_rl_phase_command_state", state)
    return state


def reset_track_phase_command_state(env, dones):
    state = getattr(env, "_rl_phase_command_state", None)
    if state is None or dones is None:
        return
    done_mask = dones.bool().view(-1)
    if done_mask.any() and state["timer"].shape[0] == done_mask.shape[0]:
        state["timer"][done_mask] = 0
        state["command"][done_mask] = 0.0
        state["maze_phase"][done_mask] = False
        if "terrain_id" in state:
            state["terrain_id"][done_mask] = 0
        if "gate_available" in state:
            state["gate_available"][done_mask] = False


def apply_track_phase_command(
    obs,
    critic_obs,
    env,
    rl_nav_conf: dict,
    logger=None,
    update_state=True,
    update_env_command=True,
    set_env_command_fn=None,
):
    """Patch the Track velocity command anchor from public goal/scan observations."""
    if not bool(rl_nav_conf.get("phase_command_enabled", False)):
        return obs, critic_obs, {}
    if bool(rl_nav_conf.get("worker_phase_command_enabled", False)):
        stats = worker_gate_monitor_stats(env)
        if stats:
            return obs, critic_obs, stats
        return obs, critic_obs, _worker_gate_fallback_monitor_stats(obs, critic_obs, env, rl_nav_conf)

    maze_phase = estimate_maze_phase_from_goal_feature(obs, rl_nav_conf)
    if maze_phase is None:
        return obs, critic_obs, {}

    state = _get_track_phase_command_state(env, obs.shape[0], obs.device, obs.dtype)
    resample_steps = max(int(rl_nav_conf.get("phase_command_resample_steps", 96)), 1)
    fallback_vx = _fallback_speed(rl_nav_conf)
    pre_range = rl_nav_conf.get("pre_maze_lin_vel_x", [fallback_vx, fallback_vx])
    slope_range = rl_nav_conf.get("slope_lin_vel_x", pre_range)
    stairs_range = rl_nav_conf.get("stairs_lin_vel_x", pre_range)
    maze_range = rl_nav_conf.get("maze_lin_vel_x", [fallback_vx, fallback_vx])
    terrain_phase_speed_enabled = bool(rl_nav_conf.get("terrain_phase_speed_enabled", False))
    gate = nan_terrain_gate_from_height_scan(obs, rl_nav_conf)
    if gate is None:
        terrain_id = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        difficulty_band = torch.zeros_like(terrain_id)
        difficulty_active = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        gate_available = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        edge_sharpness = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        difficulty_signal = torch.zeros_like(edge_sharpness)
    else:
        terrain_id = gate["terrain_id"]
        difficulty_band = gate["difficulty_band"]
        difficulty_active = gate["difficulty_active"]
        gate_available = gate["available"].bool()
        edge_sharpness = gate["edge_sharpness"]
        difficulty_signal = gate["difficulty_signal"]

    command = state["command"]
    if update_state:
        timer = torch.clamp(state["timer"] - 1, min=0)
        phase_changed = maze_phase != state["maze_phase"]
        terrain_changed = terrain_phase_speed_enabled & (~maze_phase) & (terrain_id != state["terrain_id"])
        needs_sample = (timer <= 0) | phase_changed | terrain_changed | (command[:, 0] <= 0.0)

        if needs_sample.any():
            sampled_pre = _sample_uniform_range(pre_range, (obs.shape[0],), obs.device, obs.dtype)
            sampled_slope = _sample_uniform_range(slope_range, (obs.shape[0],), obs.device, obs.dtype)
            sampled_stairs = _sample_uniform_range(stairs_range, (obs.shape[0],), obs.device, obs.dtype)
            sampled_maze = _sample_uniform_range(maze_range, (obs.shape[0],), obs.device, obs.dtype)
            sampled_fallback = torch.full((obs.shape[0],), fallback_vx, device=obs.device, dtype=obs.dtype)
            invalid_terrain = (~gate_available) | (terrain_id == 3)
            sampled_non_maze = torch.where(invalid_terrain, sampled_fallback, sampled_pre)
            if terrain_phase_speed_enabled:
                sampled_non_maze = torch.where(terrain_id == 1, sampled_slope, sampled_non_maze)
                sampled_non_maze = torch.where(terrain_id == 2, sampled_stairs, sampled_non_maze)
            sampled_vx = torch.where(maze_phase, sampled_maze, sampled_non_maze)
            command = command.clone()
            command[needs_sample, 0] = sampled_vx[needs_sample]
            command[needs_sample, 1] = 0.0
            command[needs_sample, 2] = 0.0
            state["command"] = command

        state["timer"] = torch.where(needs_sample, torch.full_like(timer, resample_steps), timer)
        state["maze_phase"] = maze_phase.detach().clone()
        state["terrain_id"] = terrain_id.detach().clone()
        state["gate_available"] = gate_available.detach().clone()
    else:
        command = command.clone()
        phase_changed = maze_phase != state["maze_phase"]
        invalid = command[:, 0] <= 0.0
        fallback = _build_midpoint_track_phase_command(obs, maze_phase, rl_nav_conf, obs.dtype)
        command[phase_changed | invalid] = fallback[phase_changed | invalid]

    obs = obs.clone()
    if obs.shape[-1] >= 9:
        obs[:, 6:9] = command

    if critic_obs is not None:
        critic_obs = critic_obs.clone()
        if critic_obs.shape[-1] >= 12:
            critic_obs[:, 9:12] = command.to(device=critic_obs.device, dtype=critic_obs.dtype)

    if update_env_command and set_env_command_fn is not None:
        set_env_command_fn(env, command, logger)

    invalid_terrain = (~gate_available) | (terrain_id == 3)
    non_maze = ~maze_phase
    flat_mask = non_maze & (~invalid_terrain) & (terrain_id == 0)
    slope_mask = non_maze & (~invalid_terrain) & (terrain_id == 1)
    stairs_mask = non_maze & (~invalid_terrain) & (terrain_id == 2)
    invalid_mask = non_maze & invalid_terrain
    difficulty_mask = non_maze & difficulty_active

    return obs, critic_obs, {
        "rl_phase_command_vx": command[:, 0].detach(),
        "rl_suggested_speed_vx": command[:, 0].detach(),
        "rl_suggested_fallback_ratio": invalid_mask.float().detach(),
        "rl_phase_maze_ratio": maze_phase.float().detach(),
        "rl_phase_flat_ratio": flat_mask.float().detach(),
        "rl_phase_slope_ratio": slope_mask.float().detach(),
        "rl_phase_stairs_ratio": stairs_mask.float().detach(),
        "rl_phase_invalid_ratio": invalid_mask.float().detach(),
        "rl_nan_gate_available_ratio": gate_available.float().detach(),
        "rl_nan_gate_low_ratio": (difficulty_mask & (difficulty_band == 0)).float().detach(),
        "rl_nan_gate_mid_ratio": (difficulty_mask & (difficulty_band == 1)).float().detach(),
        "rl_nan_gate_high_ratio": (difficulty_mask & (difficulty_band == 2)).float().detach(),
        "rl_nan_gate_edge_sharpness": edge_sharpness.detach(),
        "rl_nan_gate_difficulty_signal": difficulty_signal.detach(),
    }


# Backward-compatible aliases for older workflow imports.
apply_rl_phase_command = apply_track_phase_command
reset_phase_command_state = reset_track_phase_command_state


def _current_env_command(env, obs):
    try:
        command = env.command_manager.get_command("base_velocity")
        if command is not None and command.shape[0] == obs.shape[0]:
            return command.to(device=obs.device, dtype=obs.dtype)
    except Exception:
        pass
    if obs is not None and obs.shape[-1] >= 9:
        return obs[:, 6:9]
    return torch.zeros(obs.shape[0], 3, device=obs.device, dtype=obs.dtype)


def _actual_minus_command(env, command):
    try:
        robot = env.scene["robot"]
        actual_vx = robot.data.root_lin_vel_b[:, 0].to(device=command.device, dtype=command.dtype)
        return actual_vx - command[:, 0]
    except Exception:
        return torch.zeros(command.shape[0], device=command.device, dtype=command.dtype)


def _worker_gate_fallback_monitor_stats(obs, critic_obs, env, rl_nav_conf):
    """Learner-side monitor fallback when worker observation state is not visible.

    Some platform layouts run observation processing in an env worker process,
    while the PPO workflow reports metrics from the learner process.  In that
    case ``worker_gate_monitor_stats(env)`` can be empty even though the gate is
    active.  This fallback only reports diagnostics from the public rollout obs;
    it does not patch obs or write commands.
    """
    if obs is None or obs.shape[-1] < 301:
        return {}
    maze_phase = estimate_maze_phase_from_goal_feature(obs, rl_nav_conf)
    if maze_phase is None:
        maze_phase = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)

    gate = nan_terrain_gate_from_height_scan(obs, rl_nav_conf)
    if gate is None:
        terrain_id = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        available = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        edge = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
    else:
        terrain_id = gate["terrain_id"].to(device=obs.device)
        available = gate["available"].float().to(device=obs.device, dtype=obs.dtype)
        edge = gate["edge_sharpness"].to(device=obs.device, dtype=obs.dtype)

    non_maze = ~maze_phase
    flat = non_maze & (terrain_id == 0)
    slope = non_maze & (terrain_id == 1)
    stairs = non_maze & (terrain_id == 2)
    invalid = non_maze & ((available <= 0.5) | (terrain_id == 3))
    target_command = _build_midpoint_track_phase_command(obs, maze_phase, rl_nav_conf, obs.dtype)
    env_command = _current_env_command(env, obs)
    policy_cmd_error = torch.abs(env_command[:, 0] - target_command[:, 0])
    critic_cmd_error = torch.zeros_like(policy_cmd_error)
    if critic_obs is not None and critic_obs.shape[-1] >= 12:
        critic_cmd_error = torch.abs(critic_obs[:, 9].to(device=obs.device, dtype=obs.dtype) - target_command[:, 0])

    terrain_sum = flat.float() + slope.float() + stairs.float() + maze_phase.float() + invalid.float()
    stats = {
        "gate_worker_target_cmd_vx": target_command[:, 0].detach(),
        "gate_worker_worker_cmd_vx": env_command[:, 0].detach(),
        "gate_worker_command_written_ratio": torch.zeros_like(target_command[:, 0]),
        "gate_worker_final_flat_ratio": flat.float().detach(),
        "gate_worker_final_slope_ratio": slope.float().detach(),
        "gate_worker_final_stairs_ratio": stairs.float().detach(),
        "gate_worker_final_maze_ratio": maze_phase.float().detach(),
        "gate_worker_final_invalid_ratio": invalid.float().detach(),
        "gate_worker_final_terrain_sum_ratio": terrain_sum.detach(),
        "gate_worker_final_gate_valid_ratio": ((available > 0.5) & (terrain_sum > 0.5)).float().detach(),
        "gate_worker_policy_cmd_error": policy_cmd_error.detach(),
        "gate_worker_critic_cmd_error": critic_cmd_error.detach(),
        "gate_worker_actual_minus_cmd_vx": _actual_minus_command(env, target_command).detach(),
        "gate_worker_raw_nav_available_ratio": torch.zeros_like(target_command[:, 0]),
        "gate_worker_raw_nav_front_wall_score": torch.zeros_like(target_command[:, 0]),
        "gate_worker_nan_edge_sharpness_mean": edge.detach(),
        "gate_worker_raw_edge_sharpness_mean": edge.detach(),
        "gate_worker_sticky_hold_steps": torch.zeros_like(target_command[:, 0]),
        "gate_worker_sticky_pending_count": torch.zeros_like(target_command[:, 0]),
        "gate_worker_maze_confirm_count": torch.zeros_like(target_command[:, 0]),
    }
    return stats
