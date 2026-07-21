# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Public goal-feature helpers for Track navigation observations."""

import torch


def build_track_goal_features(env, feature_dim: int):
    """Build goal features from public env state only."""
    if feature_dim <= 0:
        return None

    zeros = torch.zeros(env.num_envs, feature_dim, device=env.device)
    if feature_dim != 3:
        return zeros

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
    local_goal = torch.stack((local_x, local_y), dim=1)
    local_goal = torch.clamp(local_goal / 10.0, -1.0, 1.0)
    goal_dist = torch.clamp(torch.linalg.norm(delta_w, dim=1), 0.0, 20.0) / 20.0
    return torch.cat((local_goal, goal_dist.unsqueeze(1)), dim=1)
