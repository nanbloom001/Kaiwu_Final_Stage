#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

RewardProcess — custom reward processor.
RewardProcess — 自定义奖励处理器。

The stage-1 locomotion configuration enables a set of custom gait, posture, and
progress rewards here. Generic Isaac Lab rewards remain inherited from
RewardProcessBase and are activated directly from TOML.
第一阶段 locomotion 配置会启用这里的自定义步态、姿态和推进奖励。
通用 Isaac Lab reward 仍继承自 RewardProcessBase，并直接由 TOML 激活。
"""

import torch

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):
    def _reward_flat_orientation(self):
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    def _reward_joint_vel(self):
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.joint_vel), dim=1)

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.5):
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not getattr(contact_sensor.cfg, "track_air_time", False):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
        is_moving = torch.norm(self.env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return reward * is_moving.float()

    def _reward_air_time_variance_penalty(self):
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not getattr(contact_sensor.cfg, "track_air_time", False):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
        return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
            torch.clip(last_contact_time, max=0.5), dim=1
        )

    def _reward_max_foot_air_time(self, threshold: float = 0.5):
        """Penalize any foot that stays in the air too long."""
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not getattr(contact_sensor.cfg, "track_air_time", False):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        over = torch.clip(air_time - threshold, min=0.0)
        return torch.sum(over, dim=1)

    def _reward_foot_symmetry(self, max_diff: float = 0.2):
        """Penalize left/right foot air-time asymmetry."""
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not getattr(contact_sensor.cfg, "track_air_time", False):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        front_diff = torch.abs(air_time[:, 0] - air_time[:, 1])
        rear_diff = torch.abs(air_time[:, 2] - air_time[:, 3])
        over_front = torch.clip(front_diff - max_diff, min=0.0)
        over_rear = torch.clip(rear_diff - max_diff, min=0.0)
        return over_front + over_rear

    def _reward_trot_gait(self):
        """Penalize bound-like gait and encourage diagonal trot rhythm."""
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not getattr(contact_sensor.cfg, "track_air_time", False):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        contact = (air_time == 0.0).float()
        front_sync = contact[:, 0] * contact[:, 1] + (1 - contact[:, 0]) * (1 - contact[:, 1])
        back_sync = contact[:, 2] * contact[:, 3] + (1 - contact[:, 2]) * (1 - contact[:, 3])
        return front_sync * back_sync

    def _reward_feet_slide(self):
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not hasattr(contact_sensor.data, "net_forces_w_history"):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        asset = self.env.scene[asset_cfg.name]
        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
        body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

    def _reward_joint_position_penalty(self, stand_still_scale: float = 5.0, velocity_threshold: float = 0.1):
        asset = self._get_robot_asset()
        cmd = torch.linalg.norm(self.env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        reward = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
            reward,
            stand_still_scale * reward,
        )

    def _reward_feet_stumble(self):
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not hasattr(contact_sensor.data, "net_forces_w"):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        return torch.any(forces_xy > 5 * forces_z, dim=1).float()

    def _reward_forward_velocity(self):
        """Forward velocity gated by posture and base-height quality."""
        robot = self._get_robot_asset()
        vel = robot.data.root_lin_vel_b[:, 0]
        gravity_xy = torch.norm(robot.data.projected_gravity_b[:, :2], dim=1)
        posture_quality = torch.exp(-gravity_xy / 0.1)
        height_err = torch.abs(robot.data.root_pos_w[:, 2] - 0.38)
        height_quality = torch.exp(-height_err / 0.05)
        return vel * posture_quality * height_quality

    def _reward_x_command_hip_regular(self):
        """Penalize asymmetric hip joints during forward commands."""
        asset = self._get_robot_asset()
        cmd = self.env.command_manager.get_command("base_velocity")

        command_x = cmd[:, 0]
        command_norm = torch.norm(cmd[:, :3], dim=1)
        x_command_ratio = torch.abs(command_x) / (command_norm + 1e-6)

        num_joints = asset.data.joint_pos.shape[1]
        leg_count = num_joints // 3
        hip_indices = list(range(0, num_joints, 3))[:leg_count]
        hip_pos = asset.data.joint_pos[:, hip_indices]

        penalty_raw = torch.abs(hip_pos[:, 0] + hip_pos[:, 1]) + torch.abs(hip_pos[:, 2] + hip_pos[:, 3])
        return penalty_raw * x_command_ratio

    def _reward_energy(self):
        asset = self._get_robot_asset()
        data = asset.data
        torque = getattr(data, "applied_torque", None)
        if torque is None:
            torque = getattr(data, "torque", None)
        if torque is None:
            torque = getattr(data, "joint_effort", None)
        if torque is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        return torch.sum(torch.abs(torque * data.joint_vel), dim=1)

    def _reward_correct_base_height(self, target_height: float = 0.38):
        """Only penalize when the base is below target height."""
        asset = self._get_robot_asset()
        below = torch.clip(target_height - asset.data.root_pos_w[:, 2], min=0.0)
        return torch.square(below)

    def _reward_hip_to_default(self):
        asset = self._get_robot_asset()
        num_joints = asset.data.joint_pos.shape[1]
        leg_count = num_joints // 3
        hip_indices = list(range(0, num_joints, 3))[:leg_count]
        hip_pos = asset.data.joint_pos[:, hip_indices]
        hip_default = asset.data.default_joint_pos[:, hip_indices]
        return torch.sum(torch.square(hip_pos - hip_default), dim=1)

    def _reward_feet_regulation(self, base_height_target: float = 0.38):
        """Penalize horizontal foot velocity while feet are close to the ground."""
        asset_cfg = self._get_foot_asset_cfg()
        asset = self.env.scene[asset_cfg.name]
        robot = self._get_robot_asset()

        feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        feet_speed_sq = torch.sum(torch.square(feet_vel_xy), dim=-1)

        root_z = robot.data.root_pos_w[:, 2:3]
        ground_z = root_z - base_height_target
        foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        feet_height = torch.clip(foot_z - ground_z, min=0.0)

        decay = torch.exp(-feet_height / (0.025 * base_height_target))
        return torch.sum(feet_speed_sq * decay, dim=1)

    def _reward_feet_contact_forces(self, max_force: float = 147.0):
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not hasattr(contact_sensor.data, "net_forces_w"):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
        over = torch.clip(forces - max_force, min=0.0)
        return torch.sum(over, dim=1)

    def _reward_legs_distance(self, min_distance: float = 0.1):
        asset_cfg = self._get_foot_asset_cfg()
        asset = self.env.scene[asset_cfg.name]
        robot = self._get_robot_asset()

        foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
        root_pos = robot.data.root_pos_w[:, :2]
        foot_pos_b = foot_pos_w - root_pos.unsqueeze(1)

        front_dy = torch.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])
        rear_dy = torch.abs(foot_pos_b[:, 2, 1] - foot_pos_b[:, 3, 1])

        penalty_front = torch.clip(min_distance - front_dy, min=0.0) ** 2
        penalty_rear = torch.clip(min_distance - rear_dy, min=0.0) ** 2
        return penalty_front + penalty_rear

    def _reward_obstacle_evasion(
        self,
        command_name: str = "base_velocity",
        obstacle_threshold: float = -0.3,
        near_x_end: int = 10,
        body_y_start: int = 3,
        body_y_end: int = 13,
        turn_std: float = 0.5,
    ):
        asset = self._get_robot_asset()
        sensor = self.env.scene.sensors["height_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)
        window = grid[:, body_y_start:body_y_end, :near_x_end]
        col_blocked = (window < obstacle_threshold).any(dim=-1).float()
        blocked = col_blocked.mean(dim=-1)
        yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
        not_evading = torch.exp(-yaw_rate / turn_std)
        cmd = self.env.command_manager.get_command(command_name)
        has_fwd_cmd = (cmd[:, 0] > 0.05).float()
        return blocked * not_evading * has_fwd_cmd

    def _reward_approach_goal(self):
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        current_dist = torch.norm(goal_pos - robot_pos, dim=1)
        if not hasattr(self.env, "_previous_goal_dist") or self.env._previous_goal_dist is None:
            self.env._previous_goal_dist = current_dist.clone()
            return torch.zeros(self.env.num_envs, device=self.env.device)
        delta_dist = current_dist - self.env._previous_goal_dist
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        delta_dist[reset_mask] = 0.0
        self.env._previous_goal_dist = current_dist.clone()
        return -delta_dist

    def _reward_heading_velocity(self):
        """Reward velocity projected toward the goal."""
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        robot = self._get_robot_asset()
        robot_vel = robot.data.root_lin_vel_w[:, :2]
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        direction = goal_pos - robot_pos
        unit_dir = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)
        return torch.sum(robot_vel * unit_dir, dim=1)

    def _reward_reach_goal(self, threshold: float = 0.6):
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()

    def _reward_navigation_time(self):
        return torch.ones(self.env.num_envs, device=self.env.device)

    def _reward_termination(self):
        """Penalise real failures (terminated ∧ ¬time_out).
        惩罚真正的失败（terminated ∧ ¬time_out）。
        """
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs
        active_terms = getattr(term_mgr, "active_terms", ())
        if "goal_reached" in active_terms:
            goal_done = term_mgr.get_term("goal_reached")
            failure = failure & ~goal_done
        return failure.float()
