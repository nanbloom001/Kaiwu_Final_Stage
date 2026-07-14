# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""RewardProcess — merged hjcnew/xtrack-style rewards on the current baseline."""

import torch

from agent_ppo.feature.terrain_gate import gate_metric
from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):
    def _tracking_command(self, command_name: str = "base_velocity"):
        return self.env.command_manager.get_command(command_name)

    def _gate_metric(self, name: str):
        """读取 terrain_gate worker_state 里的监控指标（speed_gate 探针用）。"""
        return gate_metric(self.env, name)

    def _goal_velocity_projection(self):
        """机器人前向速度在目标方向上的投影，返回 (projection, dist, goal_dir)。"""
        robot = self._get_robot_asset()
        _, dist, goal_dir = self._goal_vector_body()
        projection = torch.sum(robot.data.root_lin_vel_b[:, :2] * goal_dir, dim=1)
        return projection, dist, goal_dir

    @staticmethod
    def _quat_to_roll_pitch(quat: torch.Tensor):
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)
        return roll, pitch

    # -----------------------------------------------------------------------
    # Generic locomotion rewards
    # -----------------------------------------------------------------------

    def _reward_flat_orientation(self):
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    def _reward_joint_vel(self):
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.joint_vel), dim=1)

    def _reward_track_lin_vel_xy(self, std: float = 0.25, command_name: str = "base_velocity"):
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)
        error = cmd[:, :2] - asset.data.root_lin_vel_b[:, :2]
        return torch.exp(-torch.sum(torch.square(error), dim=1) / max(std * std, 1e-6))

    def _reward_track_ang_vel_z(self, std: float = 0.25, command_name: str = "base_velocity"):
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)
        error = cmd[:, 2] - asset.data.root_ang_vel_b[:, 2]
        return torch.exp(-torch.square(error / max(std, 1e-6)))

    def _reward_command_speed_advantage(
        self,
        command_name: str = "base_velocity",
        deadband: float = 0.03,
        surplus_scale: float = 0.35,
        lag_scale: float = 0.35,
        max_surplus: float = 0.60,
        max_lag: float = 0.60,
        lag_penalty_scale: float = 1.0,
        min_command: float = 0.10,
    ):
        asset = self._get_robot_asset()
        cmd_vx = self._tracking_command(command_name)[:, 0]
        actual_vx = asset.data.root_lin_vel_b[:, 0]

        surplus = actual_vx - cmd_vx
        faster = torch.clamp(surplus - deadband, min=0.0, max=max_surplus) / max(surplus_scale, 1e-6)
        slower = torch.clamp(-surplus - deadband, min=0.0, max=max_lag) / max(lag_scale, 1e-6)
        active = cmd_vx > min_command
        return active.float() * (faster - lag_penalty_scale * slower)

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.5):
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not getattr(contact_sensor.cfg, "track_air_time", False):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
        is_moving = torch.norm(self._tracking_command(command_name)[:, :2], dim=1) > 0.1
        return reward * is_moving.float()

    def _reward_air_time_variance_penalty(self):
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not getattr(contact_sensor.cfg, "track_air_time", False):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        last_contact_time = getattr(contact_sensor.data, "last_contact_time", None)
        if last_contact_time is None:
            return torch.var(torch.clamp(last_air_time, max=0.5), dim=1)
        return torch.var(torch.clamp(last_air_time, max=0.5), dim=1) + torch.var(
            torch.clamp(last_contact_time[:, sensor_cfg.body_ids], max=0.5),
            dim=1,
        )

    def _reward_max_foot_air_time(self, threshold: float = 0.5):
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not getattr(contact_sensor.cfg, "track_air_time", False):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        return torch.sum(torch.clamp(air_time - threshold, min=0.0), dim=1)

    def _reward_foot_symmetry(self, max_diff: float = 0.2):
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not getattr(contact_sensor.cfg, "track_air_time", False):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        front_diff = torch.abs(air_time[:, 0] - air_time[:, 1])
        rear_diff = torch.abs(air_time[:, 2] - air_time[:, 3])
        return torch.clamp(front_diff - max_diff, min=0.0) + torch.clamp(rear_diff - max_diff, min=0.0)

    def _reward_trot_gait(self):
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not getattr(contact_sensor.cfg, "track_air_time", False):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        contact = (air_time == 0.0).float()
        front_sync = contact[:, 0] * contact[:, 1] + (1.0 - contact[:, 0]) * (1.0 - contact[:, 1])
        back_sync = contact[:, 2] * contact[:, 3] + (1.0 - contact[:, 2]) * (1.0 - contact[:, 3])
        return front_sync * back_sync

    def _reward_feet_clearance(
        self,
        command_name: str = "base_velocity",
        target_height: float = 0.08,
        std: float = 0.05,
        terrain_height_scale: float = 0.6,
        max_terrain_extra_height: float = 0.08,
        speed_height_scale: float = 0.01,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 2,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
    ):
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        contact_forces = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )
        swing = contact_forces <= 1.0
        foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - asset.data.root_pos_w[:, 2].unsqueeze(1)
        command = self._tracking_command(command_name)
        command_speed = torch.norm(command[:, :2], dim=1)

        terrain_extra = torch.zeros(self.env.num_envs, device=self.env.device)
        height_scanner = self.env.scene.sensors.get("height_scanner")
        if height_scanner is not None:
            scan = height_scanner.data.pos_w[:, 2:3] - height_scanner.data.ray_hits_w[..., 2]
            grid = scan.view(self.env.num_envs, 16, 16)
            forward_window = grid[:, body_y_start:body_y_end, near_x_start:near_x_end]
            if forward_window.shape[-1] > 1 and forward_window.shape[1] > 0:
                step_deltas = torch.abs(forward_window[:, :, 1:] - forward_window[:, :, :-1]).flatten(1)
                local_step = torch.quantile(step_deltas, delta_quantile, dim=1)
                terrain_extra = torch.clamp(
                    terrain_height_scale * local_step,
                    0.0,
                    max_terrain_extra_height,
                )

        speed_extra = speed_height_scale * torch.clamp(command_speed, 0.0, 1.0)
        dynamic_target_height = target_height + terrain_extra + speed_extra
        height_error = (foot_height - dynamic_target_height.unsqueeze(1)) / max(std, 1e-6)
        clearance_reward = torch.exp(-torch.square(height_error))
        is_moving = command_speed > 0.1
        return (
            torch.sum(clearance_reward * swing.float(), dim=1)
            * is_moving.float()
            / max(len(asset_cfg.body_ids), 1)
        )

    def _reward_feet_swing_forward(
        self,
        command_name: str = "base_velocity",
        target_forward: float = 0.11,
        std: float = 0.08,
        min_command: float = 0.10,
    ):
        """鼓励摆动脚沿机器人当前机身前向迈出。

        足端位置先从世界坐标系旋转到机器人航向坐标系，
        避免机器人转弯后仍然被错误地要求沿世界 X 方向摆腿。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()

        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        # 使用现有接触判定方式识别摆动脚
        contact_forces = (
            contact_sensor.data
            .net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )
        swing = contact_forces <= 1.0

        # 足端相对机身的位置，当前仍在世界坐标系
        foot_delta_w = (
            asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
            - asset.data.root_pos_w[:, None, :2]
        )

        # root_quat_w 使用 w, x, y, z
        quat = asset.data.root_quat_w
        qw, qx, qy, qz = (
            quat[:, 0],
            quat[:, 1],
            quat[:, 2],
            quat[:, 3],
        )

        # 计算机器人世界坐标系下的 yaw
        heading = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        # 将世界坐标差值旋转到机器人航向坐标系
        cos_h = torch.cos(-heading).unsqueeze(1)
        sin_h = torch.sin(-heading).unsqueeze(1)

        foot_forward_b = (
            cos_h * foot_delta_w[..., 0]
            - sin_h * foot_delta_w[..., 1]
        )

        # 足端达到或超过目标前摆位置时奖励接近1；
        # 前摆不足时按高斯形式衰减
        shortfall = torch.clamp(
            target_forward - foot_forward_b,
            min=0.0,
        )
        forward_reward = torch.exp(
            -torch.square(shortfall / max(std, 1e-6))
        )

        command = self._tracking_command(command_name)
        has_forward_command = command[:, 0] > min_command

        return (
            torch.sum(
                forward_reward * swing.float(),
                dim=1,
            )
            * has_forward_command.float()
            / max(len(asset_cfg.body_ids), 1)
        )

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

    def _reward_joint_position_penalty(
        self,
        stand_still_scale: float = 5.0,
        velocity_threshold: float = 0.1,
        cmd_threshold: float = 0.1,
        ang_cmd_threshold: float = 0.2,
    ):
        asset = self._get_robot_asset()
        cmd = self._tracking_command("base_velocity")
        cmd_xy = torch.linalg.norm(cmd[:, :2], dim=1)
        cmd_yaw = torch.abs(cmd[:, 2])
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        deviation = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
        is_moving = torch.logical_or(
            torch.logical_or(cmd_xy > cmd_threshold, cmd_yaw > ang_cmd_threshold),
            body_vel > velocity_threshold,
        )
        return torch.where(is_moving, deviation, stand_still_scale * deviation)

    def _reward_stand_still_motion(
        self,
        command_name: str = "base_velocity",
        lin_cmd_threshold: float = 0.15,
        ang_cmd_threshold: float = 0.2,
        vertical_vel_scale: float = 0.5,
        ang_vel_scale: float = 0.5,
        joint_vel_scale: float = 0.1,
    ):
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)
        near_zero_cmd = (
            torch.linalg.norm(cmd[:, :2], dim=1) < lin_cmd_threshold
        ) & (torch.abs(cmd[:, 2]) < ang_cmd_threshold)

        base_lin_vel = asset.data.root_lin_vel_b
        base_ang_vel = asset.data.root_ang_vel_b
        mean_abs_joint_vel = torch.mean(torch.abs(asset.data.joint_vel), dim=1)

        motion_penalty = (
            torch.linalg.norm(base_lin_vel[:, :2], dim=1)
            + vertical_vel_scale * torch.abs(base_lin_vel[:, 2])
            + ang_vel_scale * torch.linalg.norm(base_ang_vel[:, :2], dim=1)
            + joint_vel_scale * mean_abs_joint_vel
        )
        return near_zero_cmd.float() * motion_penalty

    def _reward_commanded_still_penalty(
        self,
        command_name: str = "base_velocity",
        cmd_threshold: float = 0.20,
        still_speed_threshold: float = 0.08,
    ):
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)
        cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
        body_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

        commanded_to_move = cmd_speed > cmd_threshold
        stillness = torch.clamp(
            (still_speed_threshold - body_speed) / max(still_speed_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        return commanded_to_move.float() * stillness

    def _reward_feet_stumble(self):
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not hasattr(contact_sensor.data, "net_forces_w"):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        return torch.any(forces_xy > 5 * forces_z, dim=1).float()

    def _reward_forward_velocity(self):
        robot = self._get_robot_asset()
        vel = robot.data.root_lin_vel_b[:, 0]
        gravity_xy = torch.norm(robot.data.projected_gravity_b[:, :2], dim=1)
        posture_quality = torch.exp(-gravity_xy / 0.1)
        height_err = torch.abs(robot.data.root_pos_w[:, 2] - 0.38)
        height_quality = torch.exp(-height_err / 0.05)
        return vel * posture_quality * height_quality

    def _reward_forward_heading_velocity(self, target_speed: float = 0.55, max_reward: float = 1.0):
        robot = self._get_robot_asset()
        vx = robot.data.root_lin_vel_b[:, 0]
        return torch.clamp(vx / max(target_speed, 1e-6), min=0.0, max=max_reward)

    def _reward_backward_penalty(self, deadband: float = 0.03):
        robot = self._get_robot_asset()
        vx = robot.data.root_lin_vel_b[:, 0]
        return torch.clamp(-(vx + deadband), min=0.0)

    def _reward_x_command_hip_regular(self):
        asset = self._get_robot_asset()
        cmd = self._tracking_command("base_velocity")
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
        asset = self._get_robot_asset()
        return torch.square(asset.data.root_pos_w[:, 2] - target_height)

    def _reward_hip_to_default(self):
        asset = self._get_robot_asset()
        num_joints = asset.data.joint_pos.shape[1]
        leg_count = num_joints // 3
        hip_indices = list(range(0, num_joints, 3))[:leg_count]
        hip_pos = asset.data.joint_pos[:, hip_indices]
        hip_default = asset.data.default_joint_pos[:, hip_indices]
        return torch.sum(torch.square(hip_pos - hip_default), dim=1)

    def _reward_feet_regulation(self, base_height_target: float = 0.38):
        asset_cfg = self._get_foot_asset_cfg()
        asset = self.env.scene[asset_cfg.name]
        robot = self._get_robot_asset()
        feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        feet_speed_sq = torch.sum(torch.square(feet_vel_xy), dim=-1)
        root_z = robot.data.root_pos_w[:, 2:3]
        ground_z = root_z - base_height_target
        foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        feet_height = torch.clamp(foot_z - ground_z, min=0.0)
        decay = torch.exp(-feet_height / (0.025 * base_height_target))
        return torch.sum(feet_speed_sq * decay, dim=1)

    def _reward_feet_contact_forces(self, max_force: float = 147.0):
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not hasattr(contact_sensor.data, "net_forces_w"):
            return torch.zeros(self.env.num_envs, device=self.env.device)
        forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
        return torch.sum(torch.clamp(forces - max_force, min=0.0), dim=1)

    def _reward_legs_distance(self, min_distance: float = 0.1):
        asset_cfg = self._get_foot_asset_cfg()
        asset = self.env.scene[asset_cfg.name]
        robot = self._get_robot_asset()
        foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
        root_pos = robot.data.root_pos_w[:, :2]
        foot_pos_b = foot_pos_w - root_pos.unsqueeze(1)
        front_dy = torch.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])
        rear_dy = torch.abs(foot_pos_b[:, 2, 1] - foot_pos_b[:, 3, 1])
        return torch.clamp(min_distance - front_dy, min=0.0) ** 2 + torch.clamp(min_distance - rear_dy, min=0.0) ** 2

    def _reward_dof_vel(self):
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.joint_vel), dim=1)

    def _reward_base_lateral_vel(self, command_name: str = "base_velocity"):
        asset = self._get_robot_asset()
        cmd_vy = self._tracking_command(command_name)[:, 1]
        actual_vy = asset.data.root_lin_vel_b[:, 1]
        return torch.square(actual_vy - cmd_vy)

    def _reward_action_smoothness(self):
        curr = self.env.action_manager.action
        prev = self.env.action_manager.prev_action
        if not hasattr(self.env, "_smooth_prev_prev"):
            self.env._smooth_prev_prev = prev.clone()
        accel = curr - 2.0 * prev + self.env._smooth_prev_prev
        self.env._smooth_prev_prev = prev.clone()
        return torch.sum(torch.square(accel), dim=1)

    def _reward_score_guidance(
        self,
        command_name: str = "base_velocity",
        min_command: float = 0.15,
        tracking_std: float = 0.35,
        posture_std: float = 0.25,
        power_scale: float = 35.0,
        posture_weight: float = 0.6,
    ):
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)

        cmd_xy = cmd[:, :2]
        actual_xy = asset.data.root_lin_vel_b[:, :2]
        cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
        moving_cmd = cmd_speed > min_command

        vel_error = torch.sum(torch.square(cmd_xy - actual_xy), dim=1)
        tracking_score = torch.exp(-vel_error / max(tracking_std * tracking_std, 1e-6))

        roll, pitch = self._quat_to_roll_pitch(asset.data.root_quat_w)
        pose_deviation = torch.abs(roll) + torch.abs(pitch)
        pose_deviation = torch.nan_to_num(pose_deviation, nan=0.0, posinf=0.0, neginf=0.0)
        posture_score = torch.exp(-5.0 * pose_deviation)

        power = self._reward_energy()
        energy_score = torch.exp(-power / max(power_scale, 1e-6))

        posture_weight = min(max(posture_weight, 0.0), 1.0)
        score_hint = posture_weight * posture_score + (1.0 - posture_weight) * energy_score
        return moving_cmd.float() * tracking_score * score_hint

    def _reward_pose_score_formula(self):
        asset = self._get_robot_asset()
        roll, pitch = self._quat_to_roll_pitch(asset.data.root_quat_w)
        pose_deviation = torch.abs(roll) + torch.abs(pitch)
        pose_deviation = torch.nan_to_num(pose_deviation, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.exp(-5.0 * pose_deviation)

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
        sensor = self.env.scene.sensors.get("height_scanner")
        if sensor is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)
        window = grid[:, body_y_start:body_y_end, :near_x_end]
        col_blocked = (window < obstacle_threshold).any(dim=-1).float()
        blocked = col_blocked.mean(dim=-1)
        yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
        not_evading = torch.exp(-yaw_rate / turn_std)
        cmd = self._tracking_command(command_name)
        has_fwd_cmd = (cmd[:, 0] > 0.05).float()
        return blocked * not_evading * has_fwd_cmd

    # -----------------------------------------------------------------------
    # Goal / maze helpers
    # -----------------------------------------------------------------------

    def _goal_delta_body(self):
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            zeros = torch.zeros(self.env.num_envs, 2, device=self.env.device)
            return zeros, torch.zeros(self.env.num_envs, device=self.env.device)

        try:
            robot = self.env.scene["robot"]
            root_pos_w = robot.data.root_pos_w
            quat = robot.data.root_quat_w
        except Exception:
            zeros = torch.zeros(self.env.num_envs, 2, device=self.env.device)
            return zeros, torch.zeros(self.env.num_envs, device=self.env.device)

        delta_w = self.env.goal_positions[:, :2] - root_pos_w[:, :2]
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        heading = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        cos_h = torch.cos(-heading)
        sin_h = torch.sin(-heading)
        local_x = cos_h * delta_w[:, 0] - sin_h * delta_w[:, 1]
        local_y = sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
        return torch.stack((local_x, local_y), dim=1), torch.linalg.norm(delta_w, dim=1)

    def _height_grid(self):
        scanner = self.env.scene.sensors.get("height_scanner")
        if scanner is None:
            return None
        scan = scanner.data.pos_w[:, 2:3] - scanner.data.ray_hits_w[..., 2]
        return scan.view(self.env.num_envs, 16, 16)

    def _rough_terrain_gate(
        self,
        body_y_start: int = 4,
        body_y_end: int = 12,
        near_x_start: int = 1,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
        min_delta: float = 0.025,
        full_delta: float = 0.10,
    ):
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        window = grid[:, body_y_start:body_y_end, near_x_start:near_x_end]
        if window.shape[1] == 0 or window.shape[2] <= 1:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        deltas = torch.abs(window[:, :, 1:] - window[:, :, :-1])
        if deltas.numel() == 0:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        local_delta = torch.quantile(deltas.flatten(1), delta_quantile, dim=1)
        denom = max(full_delta - min_delta, 1e-6)
        return torch.clamp((local_delta - min_delta) / denom, 0.0, 1.0)

    def _goal_vector_body(self):
        local_goal, dist = self._goal_delta_body()
        denom = torch.clamp(dist, min=1e-6).unsqueeze(1)
        goal_dir = local_goal / denom
        return local_goal, dist, goal_dir

    def _reward_rough_score_guidance(
        self,
        command_name: str = "base_velocity",
        min_command: float = 0.15,
        tracking_std: float = 0.38,
        posture_weight: float = 0.75,
        body_y_start: int = 4,
        body_y_end: int = 12,
        near_x_start: int = 1,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
        min_delta: float = 0.025,
        full_delta: float = 0.10,
    ):
        robot = self._get_robot_asset()
        cmd = self._tracking_command(command_name)
        cmd_xy = cmd[:, :2]
        actual_xy = robot.data.root_lin_vel_b[:, :2]
        cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
        moving_cmd = cmd_speed > min_command

        vel_error = torch.sum(torch.square(cmd_xy - actual_xy), dim=1)
        tracking_score = torch.exp(-vel_error / max(tracking_std * tracking_std, 1e-6))

        roll, pitch = self._quat_to_roll_pitch(robot.data.root_quat_w)
        pose_deviation = torch.abs(roll) + torch.abs(pitch)
        pose_deviation = torch.nan_to_num(pose_deviation, nan=0.0, posinf=0.0, neginf=0.0)
        pose_score = torch.exp(-5.0 * pose_deviation)
        energy_score = torch.exp(-0.01 * self._reward_energy())

        rough_gate = self._rough_terrain_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            delta_quantile=delta_quantile,
            min_delta=min_delta,
            full_delta=full_delta,
        )
        posture_weight = min(max(posture_weight, 0.0), 1.0)
        score_hint = posture_weight * pose_score + (1.0 - posture_weight) * energy_score
        return rough_gate * moving_cmd.float() * tracking_score * score_hint

    def _reward_rough_ang_vel_xy(
        self,
        body_y_start: int = 4,
        body_y_end: int = 12,
        near_x_start: int = 1,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
        min_delta: float = 0.025,
        full_delta: float = 0.10,
    ):
        robot = self._get_robot_asset()
        rough_gate = self._rough_terrain_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            delta_quantile=delta_quantile,
            min_delta=min_delta,
            full_delta=full_delta,
        )
        return rough_gate * torch.linalg.norm(robot.data.root_ang_vel_b[:, :2], dim=1)

    def _reward_rough_roll_pitch_abs(
        self,
        body_y_start: int = 4,
        body_y_end: int = 12,
        near_x_start: int = 1,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
        min_delta: float = 0.015,
        full_delta: float = 0.075,
    ):
        robot = self._get_robot_asset()
        roll, pitch = self._quat_to_roll_pitch(robot.data.root_quat_w)
        pose_deviation = torch.abs(roll) + torch.abs(pitch)
        pose_deviation = torch.nan_to_num(pose_deviation, nan=0.0, posinf=0.0, neginf=0.0)
        rough_gate = self._rough_terrain_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            delta_quantile=delta_quantile,
            min_delta=min_delta,
            full_delta=full_delta,
        )
        return rough_gate * pose_deviation

    def _reward_rough_energy(
        self,
        body_y_start: int = 4,
        body_y_end: int = 12,
        near_x_start: int = 1,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
        min_delta: float = 0.015,
        full_delta: float = 0.075,
    ):
        rough_gate = self._rough_terrain_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            delta_quantile=delta_quantile,
            min_delta=min_delta,
            full_delta=full_delta,
        )
        return rough_gate * self._reward_energy()

    def _wall_score_from_sector(
        self,
        sector: torch.Tensor,
        obstacle_threshold: float = -0.75,
        temperature: float = 0.18,
    ):
        if sector.shape[1] == 0 or sector.shape[2] == 0:
            return torch.zeros(sector.shape[0], device=sector.device)
        return torch.sigmoid((obstacle_threshold - sector) / max(temperature, 1e-6)).mean(dim=(1, 2))

    def _maze_context_gate(
        self,
        grid: torch.Tensor,
        goal_dist_gate: float = 14.0,
        obstacle_threshold: float = -0.80,
        temperature: float = 0.18,
        front_cols: int = 10,
        side_width: int = 3,
        side_col_threshold: float = 0.32,
        side_depth_ratio: float = 0.55,
        front_col_threshold: float = 0.62,
        front_depth_ratio: float = 0.45,
        stair_uniformity_threshold: float = 0.16,
        stair_max_front_depth_ratio: float = 0.32,
    ):
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        num_envs = grid.shape[0]
        cols = max(1, min(int(front_cols), grid.shape[2]))
        side = max(1, min(int(side_width), grid.shape[1] // 2))
        temp = max(temperature, 1e-6)

        wall_prob = torch.sigmoid((obstacle_threshold - grid[:, :, :cols]) / temp)

        left_cols = wall_prob[:, :side, :].mean(dim=1)
        right_cols = wall_prob[:, -side:, :].mean(dim=1)
        left_cont = (left_cols > side_col_threshold).float().mean(dim=1)
        right_cont = (right_cols > side_col_threshold).float().mean(dim=1)
        side_corridor = (left_cont > side_depth_ratio) & (right_cont > side_depth_ratio)

        center = wall_prob[:, side:-side, :] if grid.shape[1] > 2 * side else wall_prob
        center_cols = center.mean(dim=1)
        dense_front_ratio = (center_cols > front_col_threshold).float().mean(dim=1)
        thick_front_wall = dense_front_ratio > front_depth_ratio

        raw_center = grid[:, side:-side, :cols] if grid.shape[1] > 2 * side else grid[:, :, :cols]
        lateral_uniformity = raw_center.std(dim=1).mean(dim=1)
        stair_or_slope_like = (
            (lateral_uniformity < stair_uniformity_threshold)
            & (dense_front_ratio < stair_max_front_depth_ratio)
        )

        front_gate = thick_front_wall.float() * (1.0 - stair_or_slope_like.float())
        visual_gate = torch.clamp(side_corridor.float() + front_gate, max=1.0)

        if hasattr(self.env, "goal_positions") and self.env.goal_positions is not None:
            _, goal_dist = self._goal_delta_body()
            phase_gate = (goal_dist < goal_dist_gate).float()
        else:
            phase_gate = torch.ones(num_envs, device=grid.device)

        return phase_gate * visual_gate

    def _reward_maze_context_gate(
        self,
        goal_dist_gate: float = 14.0,
        obstacle_threshold: float = -0.80,
        temperature: float = 0.18,
        front_cols: int = 10,
    ):
        grid = self._height_grid()
        return self._maze_context_gate(
            grid,
            goal_dist_gate=goal_dist_gate,
            obstacle_threshold=obstacle_threshold,
            temperature=temperature,
            front_cols=front_cols,
        )

    def _maze_front_wall_turn_features(
        self,
        obstacle_threshold: float = -0.72,
        temperature: float = 0.18,
        front_cols: int = 6,
        body_y_start: int = 3,
        body_y_end: int = 13,
        side_width: int = 4,
        wall_start: float = 0.28,
        wall_full: float = 0.72,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        grid = self._height_grid()
        if grid is None:
            zeros = torch.zeros(self.env.num_envs, device=self.env.device)
            return zeros, zeros

        front_cols = max(1, min(int(front_cols), grid.shape[2]))
        body_y_start = max(0, int(body_y_start))
        body_y_end = min(int(body_y_end), grid.shape[1])
        side_width = max(1, min(int(side_width), grid.shape[1] // 2))
        if body_y_end <= body_y_start:
            zeros = torch.zeros(self.env.num_envs, device=self.env.device)
            return zeros, zeros

        wall_prob = torch.sigmoid((obstacle_threshold - grid[:, :, :front_cols]) / max(temperature, 1e-6))
        center_wall = wall_prob[:, body_y_start:body_y_end, :].mean(dim=(1, 2))
        left_open = 1.0 - wall_prob[:, :side_width, :].mean(dim=(1, 2))
        right_open = 1.0 - wall_prob[:, -side_width:, :].mean(dim=(1, 2))
        open_delta = right_open - left_open

        _, _, goal_dir = self._goal_vector_body()
        goal_turn = torch.sign(goal_dir[:, 1])
        visual_turn = torch.sign(open_delta)
        turn_sign = torch.where(torch.abs(open_delta) > 0.08, visual_turn, goal_turn)

        wall_gate = torch.clamp((center_wall - wall_start) / max(wall_full - wall_start, 1e-6), 0.0, 1.0)
        maze_gate = self._maze_context_gate(
            grid,
            goal_dist_gate=maze_goal_dist_gate,
            obstacle_threshold=maze_gate_obstacle_threshold,
            temperature=temperature,
        )
        return wall_gate * maze_gate, turn_sign

    def _reward_maze_anticipatory_turn(
        self,
        obstacle_threshold: float = -0.72,
        temperature: float = 0.18,
        front_cols: int = 6,
        body_y_start: int = 3,
        body_y_end: int = 13,
        side_width: int = 4,
        wall_start: float = 0.28,
        wall_full: float = 0.72,
        target_yaw_rate: float = 0.75,
        target_forward_speed: float = 0.75,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
        near_goal_disable_dist: float = 4.6,
    ):
        robot = self._get_robot_asset()
        wall_gate, turn_sign = self._maze_front_wall_turn_features(
            obstacle_threshold=obstacle_threshold,
            temperature=temperature,
            front_cols=front_cols,
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            side_width=side_width,
            wall_start=wall_start,
            wall_full=wall_full,
            maze_goal_dist_gate=maze_goal_dist_gate,
            maze_gate_obstacle_threshold=maze_gate_obstacle_threshold,
        )
        yaw_toward_opening = torch.clamp(
            robot.data.root_ang_vel_b[:, 2] * turn_sign / max(target_yaw_rate, 1e-6),
            min=0.0,
            max=1.0,
        )
        speed_score = torch.clamp(
            robot.data.root_lin_vel_b[:, 0] / max(target_forward_speed, 1e-6),
            min=0.0,
            max=1.0,
        )
        reward = wall_gate * yaw_toward_opening * speed_score
        # 接近终点时关闭迷宫转向引导，避免终点附近乱转。
        if hasattr(self.env, "goal_positions") and self.env.goal_positions is not None:
            _, goal_dist = self._goal_delta_body()
            reward = reward * (goal_dist > near_goal_disable_dist).float()
        return reward

    # -----------------------------------------------------------------------
    # Goal / track rewards
    # -----------------------------------------------------------------------

    def _reward_goal_heading_alignment(self, std: float = 0.75):
        _, dist, goal_dir = self._goal_vector_body()
        angle_error = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])
        return torch.exp(-torch.square(angle_error / max(std, 1e-6))) * (dist > 0.6).float()

    def _reward_goal_velocity_projection(self, max_speed: float = 0.75):
        robot = self._get_robot_asset()
        _, dist, goal_dir = self._goal_vector_body()
        body_xy = robot.data.root_lin_vel_b[:, :2]
        projection = torch.sum(body_xy * goal_dir, dim=1)
        return torch.clamp(projection / max(max_speed, 1e-6), min=-1.0, max=1.0) * (dist > 0.6).float()

    def _reward_goal_backtrack_penalty(self, deadband: float = 0.02):
        robot = self._get_robot_asset()
        _, dist, goal_dir = self._goal_vector_body()
        projection = torch.sum(robot.data.root_lin_vel_b[:, :2] * goal_dir, dim=1)
        return torch.clamp(-(projection + deadband), min=0.0) * (dist > 0.8).float()

    def _reward_approach_goal(self):
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        _, current_dist = self._goal_delta_body()
        if (
            not hasattr(self.env, "_nav_previous_goal_dist")
            or self.env._nav_previous_goal_dist.shape != current_dist.shape
        ):
            self.env._nav_previous_goal_dist = current_dist.clone()
            self.env._nav_previous_goal_valid = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.env.device
            )

        if (
            not hasattr(self.env, "_nav_previous_goal_valid")
            or self.env._nav_previous_goal_valid.shape != current_dist.shape
        ):
            self.env._nav_previous_goal_valid = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.env.device
            )

        delta = current_dist - self.env._nav_previous_goal_dist
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        valid_mask = self.env._nav_previous_goal_valid & ~reset_mask
        delta = torch.where(valid_mask, delta, torch.zeros_like(delta))
        self.env._nav_previous_goal_dist = current_dist.clone()
        self.env._nav_previous_goal_valid = ~reset_mask
        return -delta

    def _reward_heading_velocity(self):
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        robot = self._get_robot_asset()
        robot_vel = robot.data.root_lin_vel_w[:, :2]
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        direction = goal_pos - robot_pos
        unit_dir = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)
        return torch.sum(robot_vel * unit_dir, dim=1)

    def _reward_goal_distance(self, scale: float = 8.0):
        _, dist = self._goal_delta_body()
        return torch.exp(-dist / max(scale, 1e-6))

    def _reward_task_complete(self, threshold: float = 0.6):
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()

    def _reward_reach_goal(self, threshold: float = 0.6):
        return self._reward_task_complete(threshold=threshold)

    def _reward_wall_proximity(
        self,
        obstacle_threshold: float = -0.55,
        front_cols: int = 7,
        body_y_start: int = 2,
        body_y_end: int = 14,
        wall_score_threshold: float = 0.18,
        temperature: float = 0.18,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        sector = grid[:, body_y_start:body_y_end, :front_cols]
        wall_score = self._wall_score_from_sector(sector, obstacle_threshold, temperature)
        gate = self._maze_context_gate(
            grid,
            goal_dist_gate=maze_goal_dist_gate,
            obstacle_threshold=maze_gate_obstacle_threshold,
            temperature=temperature,
        )
        return torch.clamp(wall_score - wall_score_threshold, min=0.0) * gate

    def _reward_wall_collision(
        self,
        obstacle_threshold: float = -0.75,
        front_cols: int = 3,
        body_y_start: int = 3,
        body_y_end: int = 13,
        wall_score_threshold: float = 0.55,
        temperature: float = 0.18,
        touch_penalty: float = 0.12,
        slow_speed: float = 0.15,
        impact_speed: float = 0.55,
        impact_penalty: float = 1.60,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        robot = self._get_robot_asset()
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        sector = grid[:, body_y_start:body_y_end, :front_cols]
        wall_score = self._wall_score_from_sector(sector, obstacle_threshold, temperature)
        forward_speed = torch.clamp(robot.data.root_lin_vel_b[:, 0], min=0.0)
        wall_intensity = torch.clamp(
            (wall_score - wall_score_threshold) / max(1.0 - wall_score_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        speed_ratio = torch.clamp(
            (forward_speed - slow_speed) / max(impact_speed - slow_speed, 1e-6),
            min=0.0,
            max=1.0,
        )
        penalty = touch_penalty + (impact_penalty - touch_penalty) * torch.square(speed_ratio)
        gate = self._maze_context_gate(
            grid,
            goal_dist_gate=maze_goal_dist_gate,
            obstacle_threshold=maze_gate_obstacle_threshold,
            temperature=temperature,
        )
        return wall_intensity * penalty * gate

    def _reward_wall_stall_penalty(
        self,
        obstacle_threshold: float = -0.70,
        front_cols: int = 5,
        body_y_start: int = 3,
        body_y_end: int = 13,
        wall_score_threshold: float = 0.38,
        temperature: float = 0.18,
        still_speed: float = 0.12,
        goal_dist_threshold: float = 0.8,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        robot = self._get_robot_asset()
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        front_cols = max(1, min(int(front_cols), grid.shape[2]))
        sector = grid[:, body_y_start:body_y_end, :front_cols]
        wall_score = self._wall_score_from_sector(sector, obstacle_threshold, temperature)
        wall_intensity = torch.clamp(
            (wall_score - wall_score_threshold) / max(1.0 - wall_score_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )

        body_speed = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
        _, goal_dist = self._goal_delta_body()
        stall_gate = (body_speed < still_speed).float() * (goal_dist > goal_dist_threshold).float()
        maze_gate = self._maze_context_gate(
            grid,
            goal_dist_gate=maze_goal_dist_gate,
            obstacle_threshold=maze_gate_obstacle_threshold,
            temperature=temperature,
        )
        return wall_intensity * stall_gate * maze_gate

    def _reward_open_space(
        self,
        obstacle_threshold: float = -0.35,
        front_cols: int = 8,
        body_y_start: int = 1,
        body_y_end: int = 15,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        sector = grid[:, body_y_start:body_y_end, :front_cols]
        gate = self._maze_context_gate(
            grid,
            goal_dist_gate=maze_goal_dist_gate,
            obstacle_threshold=maze_gate_obstacle_threshold,
        )
        return (sector > obstacle_threshold).float().mean(dim=(1, 2)) * gate

    def _reward_corridor_centering(
        self,
        obstacle_threshold: float = -0.55,
        front_cols: int = 8,
        wall_score_threshold: float = 0.20,
        temperature: float = 0.18,
        center_band_half_width: int = 1,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        front_cols = max(1, min(int(front_cols), grid.shape[2]))
        row_wall_score = torch.sigmoid(
            (obstacle_threshold - grid[:, :, :front_cols]) / max(temperature, 1e-6)
        ).mean(dim=2)

        num_rows = row_wall_score.shape[1]
        row_idx = torch.arange(num_rows, device=grid.device, dtype=grid.dtype)
        center = 0.5 * float(num_rows - 1)
        half_width = max(float(center_band_half_width), 0.0)
        left_mask = row_idx < center - half_width
        right_mask = row_idx > center + half_width

        left_score = torch.where(left_mask.unsqueeze(0), row_wall_score, torch.zeros_like(row_wall_score))
        right_score = torch.where(right_mask.unsqueeze(0), row_wall_score, torch.zeros_like(row_wall_score))
        left_strength = left_score.max(dim=1).values
        right_strength = right_score.max(dim=1).values
        corridor_gate = ((left_strength > wall_score_threshold) & (right_strength > wall_score_threshold)).float()

        dist_to_center = torch.abs(row_idx - center).unsqueeze(0)
        left_weight = left_score * left_score
        right_weight = right_score * right_score
        left_dist = torch.sum(left_weight * dist_to_center, dim=1) / torch.clamp(left_weight.sum(dim=1), min=1e-6)
        right_dist = torch.sum(right_weight * dist_to_center, dim=1) / torch.clamp(right_weight.sum(dim=1), min=1e-6)
        imbalance = torch.abs(left_dist - right_dist) / torch.clamp(left_dist + right_dist, min=1e-6)
        maze_gate = self._maze_context_gate(
            grid,
            goal_dist_gate=maze_goal_dist_gate,
            obstacle_threshold=maze_gate_obstacle_threshold,
            temperature=temperature,
        )
        return corridor_gate * imbalance * maze_gate

    def _reward_directed_exploration(
        self,
        radius: float = 0.55,
        memory_size: int = 96,
        goal_heading_std: float = 1.0,
    ):
        robot = self._get_robot_asset()
        pos = robot.data.root_pos_w[:, :2]
        num_envs = self.env.num_envs
        device = self.env.device

        if (
            not hasattr(self.env, "_rl_nav_visit_pos")
            or self.env._rl_nav_visit_pos.shape[0] != num_envs
            or self.env._rl_nav_visit_pos.shape[1] != memory_size
        ):
            self.env._rl_nav_visit_pos = torch.zeros(num_envs, memory_size, 2, device=device)
            self.env._rl_nav_visit_valid = torch.zeros(num_envs, memory_size, dtype=torch.bool, device=device)
            self.env._rl_nav_visit_ptr = torch.zeros(num_envs, dtype=torch.long, device=device)

        visit_pos = self.env._rl_nav_visit_pos
        valid = self.env._rl_nav_visit_valid
        dist_to_seen = torch.linalg.norm(visit_pos - pos.unsqueeze(1), dim=2)
        dist_to_seen = torch.where(valid, dist_to_seen, torch.full_like(dist_to_seen, 1e6))
        novel = dist_to_seen.min(dim=1).values > radius

        _, goal_dist, goal_dir = self._goal_vector_body()
        angle_error = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])
        toward_goal_gate = torch.exp(-torch.square(angle_error / max(goal_heading_std, 1e-6)))
        reward = novel.float() * toward_goal_gate * (goal_dist > 1.0).float()

        ptr = self.env._rl_nav_visit_ptr
        env_ids = torch.arange(num_envs, device=device)
        if novel.any():
            add_ids = env_ids[novel]
            add_ptr = ptr[novel]
            visit_pos[add_ids, add_ptr] = pos[novel]
            valid[add_ids, add_ptr] = True
            ptr[novel] = (add_ptr + 1) % memory_size

        try:
            done = self.env.termination_manager.terminated | self.env.termination_manager.time_outs
            if done.any():
                visit_pos[done] = 0.0
                valid[done] = False
                ptr[done] = 0
        except Exception:
            pass

        return reward

    def _reward_stuck_penalty(self, min_command: float = 0.15, still_speed: float = 0.08):
        robot = self._get_robot_asset()
        _, dist = self._goal_delta_body()
        cmd = self._tracking_command("base_velocity")
        cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
        body_speed = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
        return ((cmd_speed > min_command) & (body_speed < still_speed) & (dist > 0.8)).float()

    def _reward_navigation_time(self):
        return torch.ones(self.env.num_envs, device=self.env.device)

    def _reward_termination(self):
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs
        try:
            if "goal_reached" in term_mgr.active_terms:
                failure = failure & ~term_mgr.get_term("goal_reached")
        except Exception:
            pass
        return failure.float()

    # -----------------------------------------------------------------------
    # Suggested-speed gate monitor probes
    # 建议速度门控监控探针（weight=1e-9，纯监控用，不主导 reward）
    # -----------------------------------------------------------------------

    def _reward_speed_gate_flat(self):
        return self._gate_metric("final_flat")

    def _reward_speed_gate_slope(self):
        return self._gate_metric("final_slope")

    def _reward_speed_gate_stairs(self):
        return self._gate_metric("final_stairs")

    def _reward_speed_gate_maze(self):
        return self._gate_metric("final_maze")

    def _reward_speed_gate_invalid(self):
        return self._gate_metric("final_invalid")

    def _reward_speed_gate_sum(self):
        return self._gate_metric("final_terrain_sum")

    def _reward_speed_gate_valid(self):
        return self._gate_metric("final_gate_valid")

    def _reward_speed_gate_target_vx(self):
        return self._gate_metric("target_cmd_vx")

    def _reward_speed_gate_worker_vx(self):
        return self._gate_metric("worker_cmd_vx")

    def _reward_speed_gate_written(self):
        return self._gate_metric("command_written")

    def _reward_speed_gate_nav_front(self):
        return self._gate_metric("nav_wall_front_score")

    def _reward_speed_gate_nav_block(self):
        return self._gate_metric("nav_wall_front_blocked")

    def _reward_speed_gate_hold_steps(self):
        return self._gate_metric("sticky_hold_steps")

    def _reward_speed_gate_pending(self):
        return self._gate_metric("sticky_pending_count")

    def _reward_speed_gate_maze_confirm(self):
        return self._gate_metric("maze_confirm_count")

    # -----------------------------------------------------------------------
    # Energy / posture formula rewards（与平台评分公式对齐）
    # -----------------------------------------------------------------------

    def _reward_energy_score_formula(self):
        """平台对齐能耗评分：exp(-0.01 * sum(|torque × joint_vel|))。

        与平台评分公式严格对齐：系数 0.01 与 base_scorer.py 中
        energy_score = 100 * exp(-0.01 * mean_energy) 的指数核一致。
        输出范围 (0, 1]，功率越低奖励越接近 1，直接引导策略降低能耗。
        """
        power = self._reward_energy()
        return torch.exp(-0.01 * power)

    def _reward_posture_stability(self):
        """姿态稳定性惩罚：惩罚 roll/pitch 的快速变化（一阶差分）。

        直接指数奖励只惩罚当前偏角大小，无法抑制机身周期性震荡；
        本项惩罚角度的变化速率，鼓励机身平稳过渡而非来回摇摆。
        """
        asset = self._get_robot_asset()
        roll, pitch = self._quat_to_roll_pitch(asset.data.root_quat_w)

        if not hasattr(self.env, "_posture_prev_roll") or self.env._posture_prev_roll.shape != roll.shape:
            self.env._posture_prev_roll = roll.clone()
            self.env._posture_prev_pitch = pitch.clone()

        roll_rate = torch.abs(roll - self.env._posture_prev_roll)
        pitch_rate = torch.abs(pitch - self.env._posture_prev_pitch)

        try:
            done = self.env.termination_manager.terminated | self.env.termination_manager.time_outs
            if done.any():
                roll_rate[done] = 0.0
                pitch_rate[done] = 0.0
        except Exception:
            pass

        self.env._posture_prev_roll = roll.clone()
        self.env._posture_prev_pitch = pitch.clone()

        return roll_rate + pitch_rate

    def _reward_difficulty_pressure_complete(
        self,
        threshold: float = 0.6,
        ema_decay: float = 0.995,
        warmup_steps: int = 80,
        min_std: float = 0.02,
        std_scale: float = 2.0,
        energy_weight: float = 0.6,
        pressure_start: float = 0.55,
        curve_power: float = 1.5,
    ):
        """高压episode的额外完成奖励。

        Pressure由当前episode的mean energy/posture formula分数相对
        近期EMA统计推断。单独不奖励；只有完成的env可拿到bonus。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        num_envs = self.env.num_envs
        device = self.env.device
        asset = self._get_robot_asset()
        power = torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)
        energy_score = torch.exp(-0.01 * power)
        roll, pitch = self._quat_to_roll_pitch(asset.data.root_quat_w)
        pose_deviation = torch.nan_to_num(torch.abs(roll) + torch.abs(pitch), nan=0.0, posinf=0.0, neginf=0.0)
        pose_score = torch.exp(-5.0 * pose_deviation)

        if (
            not hasattr(self.env, "_difficulty_pressure_energy_sum")
            or self.env._difficulty_pressure_energy_sum.shape[0] != num_envs
        ):
            self.env._difficulty_pressure_energy_sum = torch.zeros(num_envs, device=device)
            self.env._difficulty_pressure_pose_sum = torch.zeros(num_envs, device=device)
            self.env._difficulty_pressure_step_count = torch.zeros(num_envs, device=device)

        energy_sum = self.env._difficulty_pressure_energy_sum
        pose_sum = self.env._difficulty_pressure_pose_sum
        step_count = self.env._difficulty_pressure_step_count
        energy_sum[:] = energy_sum + energy_score.detach()
        pose_sum[:] = pose_sum + pose_score.detach()
        step_count[:] = step_count + 1.0

        safe_count = torch.clamp(step_count, min=1.0)
        episode_energy_mean = energy_sum / safe_count
        episode_pose_mean = pose_sum / safe_count
        energy_mean_batch = episode_energy_mean.mean()
        pose_mean_batch = episode_pose_mean.mean()
        energy_var_batch = torch.mean(torch.square(episode_energy_mean - energy_mean_batch))
        pose_var_batch = torch.mean(torch.square(episode_pose_mean - pose_mean_batch))

        stats = getattr(self.env, "_difficulty_pressure_stats", None)
        if stats is None:
            stats = {
                "energy_mean": energy_mean_batch.clone(),
                "energy_var": torch.clamp(energy_var_batch, min=min_std * min_std).clone(),
                "pose_mean": pose_mean_batch.clone(),
                "pose_var": torch.clamp(pose_var_batch, min=min_std * min_std).clone(),
                "steps": 0,
            }
            setattr(self.env, "_difficulty_pressure_stats", stats)

        energy_std = torch.sqrt(torch.clamp(stats["energy_var"], min=min_std * min_std))
        pose_std = torch.sqrt(torch.clamp(stats["pose_var"], min=min_std * min_std))
        energy_pressure = torch.clamp((stats["energy_mean"] - episode_energy_mean) / max(std_scale, 1e-6) / energy_std, 0.0, 1.0)
        pose_pressure = torch.clamp((stats["pose_mean"] - episode_pose_mean) / max(std_scale, 1e-6) / pose_std, 0.0, 1.0)
        energy_weight = min(max(float(energy_weight), 0.0), 1.0)
        pressure = energy_weight * energy_pressure + (1.0 - energy_weight) * pose_pressure
        pressure_start = min(max(float(pressure_start), 0.0), 0.99)
        bonus_pressure = torch.clamp((pressure - pressure_start) / max(1.0 - pressure_start, 1e-6), 0.0, 1.0)
        if curve_power != 1.0:
            bonus_pressure = torch.pow(bonus_pressure, max(float(curve_power), 1.0))

        robot_pos = asset.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        complete = (torch.norm(goal_pos - robot_pos, dim=1) < threshold).float()
        reward = bonus_pressure.detach() * complete

        decay = min(max(float(ema_decay), 0.0), 0.9999)
        stats["energy_mean"] = (decay * stats["energy_mean"] + (1.0 - decay) * energy_mean_batch).detach()
        stats["pose_mean"] = (decay * stats["pose_mean"] + (1.0 - decay) * pose_mean_batch).detach()
        stats["energy_var"] = (
            decay * stats["energy_var"] + (1.0 - decay) * torch.clamp(energy_var_batch, min=min_std * min_std)
        ).detach()
        stats["pose_var"] = (
            decay * stats["pose_var"] + (1.0 - decay) * torch.clamp(pose_var_batch, min=min_std * min_std)
        ).detach()
        stats["steps"] = int(stats.get("steps", 0)) + 1
        self.env._difficulty_pressure_metrics = {
            "pressure": pressure.detach(),
            "bonus_pressure": bonus_pressure.detach(),
            "energy_pressure": energy_pressure.detach(),
            "pose_pressure": pose_pressure.detach(),
            "episode_energy_mean": episode_energy_mean.detach(),
            "episode_pose_mean": episode_pose_mean.detach(),
        }
        try:
            done = self.env.termination_manager.terminated | self.env.termination_manager.time_outs
            if done.any():
                energy_sum[done] = 0.0
                pose_sum[done] = 0.0
                step_count[done] = 0.0
        except Exception:
            pass
        if stats["steps"] < int(warmup_steps):
            return torch.zeros_like(reward)
        return reward

    # -----------------------------------------------------------------------
    # Near-goal navigation rewards（终点冲刺阶段的精细引导）
    # -----------------------------------------------------------------------

    def _reward_near_goal_circling_penalty(
        self,
        near_dist: float = 4.6,
        complete_dist: float = 0.6,
        min_progress_speed: float = 0.10,
        yaw_rate_threshold: float = 0.35,
        lateral_speed_threshold: float = 0.14,
    ):
        """到达终点附近但无正向目标进展时的转向/横移惩罚。"""
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        projection, dist, _ = self._goal_velocity_projection()
        yaw_rate = torch.abs(robot.data.root_ang_vel_b[:, 2])
        lateral_speed = torch.abs(robot.data.root_lin_vel_b[:, 1])

        near_gate = ((dist < near_dist) & (dist > complete_dist)).float()
        no_progress = torch.clamp((min_progress_speed - projection) / max(min_progress_speed, 1e-6), 0.0, 1.0)
        circling = torch.maximum(
            torch.clamp((yaw_rate - yaw_rate_threshold) / max(yaw_rate_threshold, 1e-6), 0.0, 1.0),
            torch.clamp((lateral_speed - lateral_speed_threshold) / max(lateral_speed_threshold, 1e-6), 0.0, 1.0),
        )
        return near_gate * no_progress * circling

    def _reward_near_goal_finish_drive(
        self,
        near_dist: float = 4.6,
        complete_dist: float = 0.6,
        target_speed: float = 0.35,
        yaw_rate_soft_limit: float = 0.35,
        lateral_speed_soft_limit: float = 0.16,
    ):
        """冲入goal捕获半径的最后直道推进奖励。"""
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        projection, dist, _ = self._goal_velocity_projection()
        yaw_rate = torch.abs(robot.data.root_ang_vel_b[:, 2])
        lateral_speed = torch.abs(robot.data.root_lin_vel_b[:, 1])

        near_gate = ((dist < near_dist) & (dist > complete_dist)).float()
        closeness = torch.clamp((near_dist - dist) / max(near_dist - complete_dist, 1e-6), 0.0, 1.0)
        progress = torch.clamp(projection / max(target_speed, 1e-6), 0.0, 1.0)
        steady_heading = 1.0 - torch.clamp(yaw_rate / max(yaw_rate_soft_limit, 1e-6), 0.0, 1.0)
        centered_motion = 1.0 - torch.clamp(lateral_speed / max(lateral_speed_soft_limit, 1e-6), 0.0, 1.0)
        return near_gate * closeness * progress * torch.clamp(0.5 * (steady_heading + centered_motion), 0.0, 1.0)

    def _reward_near_goal_retreat_penalty(
        self,
        near_dist: float = 4.6,
        complete_dist: float = 0.6,
        retreat_deadband: float = 0.03,
        target_speed: float = 0.25,
    ):
        """最后接近阶段后背离goal的强惩罚。"""
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        projection, dist, _ = self._goal_velocity_projection()
        near_gate = ((dist < near_dist) & (dist > complete_dist)).float()
        retreat = torch.clamp(-(projection + retreat_deadband) / max(target_speed, 1e-6), 0.0, 1.0)
        closeness = torch.clamp((near_dist - dist) / max(near_dist - complete_dist, 1e-6), 0.0, 1.0)
        return near_gate * retreat * (0.5 + 0.5 * closeness)

    def _reward_goal_miss_penalty(
        self,
        near_dist: float = 4.6,
        complete_dist: float = 0.6,
        miss_margin: float = 0.12,
        reset_dist: float = 5.0,
    ):
        """进入终点区后又飘走的惩罚。"""
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        _, current_dist = self._goal_delta_body()
        num_envs = self.env.num_envs
        device = self.env.device

        if (
            not hasattr(self.env, "_nav_near_goal_best_dist")
            or self.env._nav_near_goal_best_dist.shape != current_dist.shape
        ):
            self.env._nav_near_goal_best_dist = torch.full_like(current_dist, reset_dist)
            self.env._nav_near_goal_active = torch.zeros(num_envs, dtype=torch.bool, device=device)

        if (
            not hasattr(self.env, "_nav_near_goal_active")
            or self.env._nav_near_goal_active.shape != current_dist.shape
        ):
            self.env._nav_near_goal_active = torch.zeros(num_envs, dtype=torch.bool, device=device)

        active = self.env._nav_near_goal_active
        entered = current_dist < near_dist
        active[:] = active | entered
        active[:] = active & (current_dist < reset_dist)

        best_dist = self.env._nav_near_goal_best_dist
        best_dist[:] = torch.where(active, torch.minimum(best_dist, current_dist), torch.full_like(best_dist, reset_dist))
        miss = torch.clamp((current_dist - best_dist - miss_margin) / max(miss_margin, 1e-6), min=0.0, max=1.0)
        miss = miss * active.float() * (current_dist > complete_dist).float()

        try:
            done = self.env.termination_manager.terminated | self.env.termination_manager.time_outs
            if done.any():
                active[done] = False
                best_dist[done] = reset_dist
        except Exception:
            pass

        return miss

    def _reward_long_non_foot_contact(
        self,
        force_threshold: float = 5.0,
        duration_s: float = 1.0,
        step_dt: float = 0.02,
        max_penalty: float = 2.0,
        maze_only: bool = True,
        maze_goal_dist_gate: float = 14.0,
    ):
        """惩罚非脚部body的持续真实接触，对角落卡住尤其有效。"""
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w
        if forces is None or forces.ndim != 3:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        num_bodies = forces.shape[1]
        if num_bodies <= 0:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        non_foot_mask = torch.ones(num_bodies, dtype=torch.bool, device=forces.device)
        foot_ids = torch.as_tensor(sensor_cfg.body_ids, dtype=torch.long, device=forces.device)
        foot_ids = foot_ids[(foot_ids >= 0) & (foot_ids < num_bodies)]
        if foot_ids.numel() == 0:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        non_foot_mask[foot_ids] = False
        if not non_foot_mask.any():
            return torch.zeros(self.env.num_envs, device=self.env.device)

        contact_force = forces[:, non_foot_mask, :].norm(dim=-1).amax(dim=1)
        active_contact = contact_force > force_threshold
        if maze_only and hasattr(self.env, "goal_positions") and self.env.goal_positions is not None:
            _, goal_dist = self._goal_delta_body()
            active_contact = active_contact & (goal_dist < maze_goal_dist_gate)

        if (
            not hasattr(self.env, "_rl_non_foot_contact_steps")
            or self.env._rl_non_foot_contact_steps.shape[0] != self.env.num_envs
        ):
            self.env._rl_non_foot_contact_steps = torch.zeros(
                self.env.num_envs, dtype=torch.long, device=self.env.device
            )

        steps = self.env._rl_non_foot_contact_steps
        steps[:] = torch.where(active_contact, steps + 1, torch.zeros_like(steps))
        try:
            done = self.env.termination_manager.terminated | self.env.termination_manager.time_outs
            if done.any():
                steps[done] = 0
        except Exception:
            pass

        threshold_steps = max(int(duration_s / max(step_dt, 1e-6)), 1)
        over = torch.clamp((steps.float() - float(threshold_steps)) / float(threshold_steps), min=0.0)
        return torch.clamp(over + (steps >= threshold_steps).float(), min=0.0, max=max_penalty)
