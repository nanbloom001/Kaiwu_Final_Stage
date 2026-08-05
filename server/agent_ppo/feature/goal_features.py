# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Public goal-feature helpers for Track navigation observations."""

import torch

from agent_ppo.feature import nav_contract


def build_track_goal_raw(env) -> torch.Tensor:
    """Return the planar goal vector in the robot frame, in meters."""
    zeros = torch.zeros(env.num_envs, 2, device=env.device)
    # P3 deliberately owns a short-range private subgoal while the Standard
    # environment may still expose its native scoring goal.  Prefer the P3
    # target whenever present so the high-level observation and its reward
    # refer to the same objective.
    goal_positions = getattr(env, "_p3_goal_positions", None)
    if goal_positions is None:
        goal_positions = getattr(env, "goal_positions", None)
    if goal_positions is None:
        return zeros

    try:
        robot = env.scene["robot"]
        root_pos_w = robot.data.root_pos_w
        quat = robot.data.root_quat_w
    except Exception:
        return zeros

    delta_w = goal_positions[:, :2] - root_pos_w[:, :2]
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    heading = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cos_h = torch.cos(-heading)
    sin_h = torch.sin(-heading)
    local_x = cos_h * delta_w[:, 0] - sin_h * delta_w[:, 1]
    local_y = sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
    return torch.stack((local_x, local_y), dim=1)


def encode_track_goal(local_goal_xy: torch.Tensor) -> torch.Tensor:
    """Encode a metric robot-frame goal with the versioned Track contract."""
    if local_goal_xy.ndim != 2 or local_goal_xy.shape[-1] != 2:
        raise ValueError(
            "Track goal must have shape [num_envs, 2], "
            f"got {tuple(local_goal_xy.shape)}."
        )

    local_goal = nav_contract.encode_goal_xy_direction_preserving(local_goal_xy)
    goal_dist = torch.clamp(
        torch.linalg.norm(local_goal_xy, dim=1),
        0.0,
        nav_contract.GOAL_DIST_SCALE_M,
    ) / nav_contract.GOAL_DIST_SCALE_M
    return torch.cat((local_goal, goal_dist.unsqueeze(1)), dim=1)


def build_track_goal_features(env, feature_dim: int):
    """Build clean goal features while preserving the public ST7 interface."""
    if feature_dim <= 0:
        return None

    if feature_dim != 3:
        return torch.zeros(env.num_envs, feature_dim, device=env.device)

    return encode_track_goal(build_track_goal_raw(env))
