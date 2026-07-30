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
    def _p2_goal_geometry(self):
        goal = getattr(self.env, "goal_positions", None)
        if not torch.is_tensor(goal):
            zeros = torch.zeros(self.env.num_envs, 2, device=self.env.device)
            return zeros, torch.zeros(self.env.num_envs, device=self.env.device), zeros
        robot = self._get_robot_asset()
        delta_w = goal[:, :2] - robot.data.root_pos_w[:, :2]
        distance = torch.linalg.vector_norm(delta_w, dim=-1)
        quat = robot.data.root_quat_w
        w, x, y, z = quat.unbind(-1)
        yaw = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y.square() + z.square()),
        )
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        local = torch.stack(
            (
                cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1],
                -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1],
            ),
            dim=-1,
        )
        direction = torch.nn.functional.normalize(local, dim=-1, eps=1.0e-6)
        return local, distance, direction

    def _p15_contact_statistics(self, ema_tau_s: float = 1.0):
        """Return current air time and a once-per-step contact-duty EMA."""
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if not getattr(contact_sensor.cfg, "track_air_time", False):
            zeros = torch.zeros(self.env.num_envs, 4, device=self.env.device)
            return zeros, zeros
        air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        contact = (air_time <= 0.0).to(torch.float32)
        step_key = int(getattr(self.env, "common_step_counter", -1))
        state = getattr(self.env, "_p15_contact_reward_state", None)
        if not isinstance(state, dict) or state.get("duty") is None:
            state = {"step": None, "duty": contact.clone()}
            self.env._p15_contact_reward_state = state
        if state["step"] != step_key:
            dt_s = float(
                getattr(self.env, "step_dt", getattr(self.env, "physics_dt", 0.02))
            )
            alpha = 1.0 - torch.exp(
                torch.tensor(
                    -max(0.0, dt_s) / max(1.0e-3, float(ema_tau_s)),
                    device=contact.device,
                )
            )
            state["duty"].mul_(1.0 - alpha).add_(contact * alpha)
            reset = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=contact.device
            )
            lengths = getattr(self.env, "episode_length_buf", None)
            if torch.is_tensor(lengths) and lengths.numel() == self.env.num_envs:
                reset |= lengths.to(contact.device).reshape(-1) == 0
            term_mgr = getattr(self.env, "termination_manager", None)
            terminated = getattr(term_mgr, "terminated", None)
            time_outs = getattr(term_mgr, "time_outs", None)
            for value in (terminated, time_outs):
                if torch.is_tensor(value) and value.numel() == self.env.num_envs:
                    reset |= value.to(contact.device).reshape(-1).bool()
            state["duty"][reset] = contact[reset]
            state["step"] = step_key
        return air_time, state["duty"]

    @staticmethod
    def _zero_stability_huber(value, scale: float):
        """Per-environment robust normalized penalty for zero-command stability."""
        normalized = torch.abs(value) / float(scale)
        penalty = torch.where(
            normalized <= 1.0,
            normalized.square(),
            2.0 * normalized - 1.0,
        )
        return penalty.mean(dim=-1) if penalty.ndim > 1 else penalty

    def _reward_zero_command_stability(
        self,
        command_name: str = "base_velocity",
        command_threshold: float = 0.05,
        grace_period_s: float = 0.4,
    ):
        """Penalize residual motion only after a verified zero-command grace period.

        The complete velocity command is tested, so a pure-yaw command remains
        a motion command and is never treated as a stop request. Action values
        come from the environment action manager after workflow clipping.
        """
        asset = self._get_robot_asset()
        command = self.env.command_manager.get_command(command_name)[:, :3]
        zero_mask = torch.linalg.vector_norm(command, dim=1) < command_threshold
        device = command.device
        num_envs = command.shape[0]
        elapsed = getattr(self.env, "_zero_command_elapsed_s", None)
        if not isinstance(elapsed, torch.Tensor) or elapsed.shape != (num_envs,):
            elapsed = torch.zeros(num_envs, device=device, dtype=command.dtype)
            self.env._zero_command_elapsed_s = elapsed
        dt_s = float(
            getattr(
                self.env,
                "step_dt",
                getattr(self.env, "physics_dt", 0.02),
            )
        )
        elapsed[zero_mask] += max(0.0, dt_s)
        elapsed[~zero_mask] = 0.0
        term_mgr = getattr(self.env, "termination_manager", None)
        terminated = getattr(term_mgr, "terminated", None)
        if isinstance(terminated, torch.Tensor) and terminated.shape == elapsed.shape:
            elapsed[terminated.bool()] = 0.0
        active = zero_mask & (elapsed >= float(grace_period_s))
        if not bool(active.any()):
            return torch.zeros(num_envs, device=device, dtype=command.dtype)

        lin_vel = asset.data.root_lin_vel_b
        ang_vel = asset.data.root_ang_vel_b
        action_manager = getattr(self.env, "action_manager", None)
        current_action = getattr(action_manager, "action", None)
        previous_action = getattr(action_manager, "prev_action", None)
        if not isinstance(current_action, torch.Tensor) or not isinstance(previous_action, torch.Tensor):
            action_delta = torch.zeros(num_envs, device=device, dtype=command.dtype)
            action_second_delta = action_delta
        else:
            previous_previous = getattr(self.env, "_zero_stability_previous_action", None)
            if (
                not isinstance(previous_previous, torch.Tensor)
                or previous_previous.shape != previous_action.shape
            ):
                previous_previous = previous_action.clone()
            action_delta = self._zero_stability_huber(
                current_action - previous_action, 0.25
            )
            action_second_delta = self._zero_stability_huber(
                current_action - 2.0 * previous_action + previous_previous, 0.25
            )
            self.env._zero_stability_previous_action = previous_action.clone()

        penalty = (
            self._zero_stability_huber(lin_vel[:, :2], 0.05)
            + 0.5 * self._zero_stability_huber(lin_vel[:, 2], 0.03)
            + 0.5 * self._zero_stability_huber(ang_vel[:, :2], 0.10)
            + 0.25 * self._zero_stability_huber(ang_vel[:, 2], 0.10)
            + 0.10 * action_delta
            + 0.05 * action_second_delta
        )
        return torch.where(active, penalty, torch.zeros_like(penalty))

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
        is_moving = torch.linalg.vector_norm(
            self.env.command_manager.get_command(command_name)[:, :3], dim=1
        ) > 0.1
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

    def _reward_max_continuous_air_time_penalty(
        self, soft_limit_s: float = 0.55, hard_limit_s: float = 1.20
    ):
        air_time, _ = self._p15_contact_statistics()
        scale = max(1.0e-3, float(hard_limit_s) - float(soft_limit_s))
        normalized = torch.clamp((air_time - float(soft_limit_s)) / scale, 0.0, 1.0)
        return normalized.max(dim=1).values

    def _reward_contact_duty_factor_symmetry(
        self,
        ema_tau_s: float = 1.0,
        full_symmetry_wz: float = 0.15,
        minimum_scale: float = 0.25,
        maximum_wz: float = 0.80,
    ):
        _, duty = self._p15_contact_statistics(ema_tau_s)
        front = torch.abs(duty[:, 0] - duty[:, 1])
        rear = torch.abs(duty[:, 2] - duty[:, 3])
        raw = 0.5 * (front + rear)
        command = self.env.command_manager.get_command("base_velocity")[:, :3]
        yaw = torch.abs(command[:, 2])
        span = max(1.0e-3, float(maximum_wz) - float(full_symmetry_wz))
        blend = torch.clamp(1.0 - (yaw - float(full_symmetry_wz)) / span, 0.0, 1.0)
        scale = float(minimum_scale) + (1.0 - float(minimum_scale)) * blend
        return torch.clamp(raw * scale, 0.0, 1.0)

    def _reward_foot_contact_participation(
        self, ema_tau_s: float = 1.0, minimum_duty_factor: float = 0.12
    ):
        _, duty = self._p15_contact_statistics(ema_tau_s)
        minimum = max(1.0e-3, float(minimum_duty_factor))
        deficit = torch.clamp((minimum - duty) / minimum, 0.0, 1.0)
        return deficit.mean(dim=1)

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
        command = self.env.command_manager.get_command("base_velocity")
        moving = torch.linalg.vector_norm(command[:, :3], dim=1) > 0.1
        return torch.where(
            moving,
            front_sync * back_sync,
            torch.zeros_like(front_sync),
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
        previous = getattr(self.env, "_p2_previous_goal_dist", None)
        valid = getattr(self.env, "_p2_previous_goal_valid", None)
        if not torch.is_tensor(previous) or previous.shape != current_dist.shape:
            self.env._p2_previous_goal_dist = current_dist.clone()
            self.env._p2_previous_goal_valid = torch.zeros_like(current_dist, dtype=torch.bool)
            self.env._p2_last_goal_progress = torch.zeros_like(current_dist)
            return torch.zeros(self.env.num_envs, device=self.env.device)
        if not torch.is_tensor(valid) or valid.shape != current_dist.shape:
            valid = torch.zeros_like(current_dist, dtype=torch.bool)
        raw_progress = previous - current_dist
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        progress = torch.where(
            valid & ~reset_mask,
            raw_progress,
            torch.zeros_like(raw_progress),
        )
        self.env._p2_previous_goal_dist = current_dist.clone()
        self.env._p2_previous_goal_valid = ~reset_mask
        # Curriculum diagnostics need the terminal-frame distance delta even
        # though the reward itself masks reset boundaries.
        self.env._p2_last_goal_progress = torch.where(
            valid,
            raw_progress,
            torch.zeros_like(raw_progress),
        ).detach().clone()
        return progress

    def _reward_goal_heading_alignment(self, std: float = 0.75):
        _, distance, direction = self._p2_goal_geometry()
        error = torch.atan2(direction[:, 1], direction[:, 0])
        return torch.exp(-torch.square(error / max(float(std), 1.0e-6))) * (distance > 0.6)

    def _reward_goal_velocity_projection(self, max_speed: float = 0.75):
        robot = self._get_robot_asset()
        _, distance, direction = self._p2_goal_geometry()
        projection = torch.sum(robot.data.root_lin_vel_b[:, :2] * direction, dim=-1)
        return torch.clamp(projection / max(float(max_speed), 1.0e-6), -1.0, 1.0) * (distance > 0.6)

    def _reward_goal_distance(self, scale: float = 8.0):
        _, distance, _ = self._p2_goal_geometry()
        return torch.exp(-distance / max(float(scale), 1.0e-6))

    def _reward_task_complete(self, threshold: float = 0.6):
        _, distance, _ = self._p2_goal_geometry()
        return (distance < float(threshold)).float()

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
