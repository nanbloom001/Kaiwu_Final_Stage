# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
RewardProcess — PPO locomotion and pure-RL maze navigation rewards.

The locomotion terms keep the pretrained gait stable.  The navigation terms
shape target-facing progress, completion, wall avoidance, and a very small
goal-directed exploration signal without using a hand-written planner.
"""

import torch

from agent_ppo.feature.terrain_gate import gate_metric
from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):

    def _gate_metric(self, name: str):
        return gate_metric(self.env, name)

    # -----------------------------------------------------------------------
    # Suggested-speed gate monitor probes
    # 建议速度门控监控探针
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

    def _tracking_command(self, command_name: str = "base_velocity"):
        command_manager = self.env.command_manager
        return command_manager.get_command(command_name)

    def _body_forward_speed(self):
        return self._get_robot_asset().data.root_lin_vel_b[:, 0]

    @staticmethod
    def _xy_norm(xy: torch.Tensor):
        return torch.linalg.norm(xy, dim=1)

    def _blank_reward(self):
        return torch.zeros(self.env.num_envs, device=self.env.device)

    def _unit_reward(self):
        return torch.empty(self.env.num_envs, device=self.env.device).fill_(1.0)

    @staticmethod
    def _contact_force_peak(contact_sensor, body_ids):
        return contact_sensor.data.net_forces_w_history[:, :, body_ids, :].norm(dim=-1).amax(dim=1)

    @staticmethod
    def _quat_to_roll_pitch(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract roll and pitch from WXYZ quaternions, matching BaseScorer."""
        w, x, y, z = quat.unbind(dim=1)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)
        return roll, pitch

    # -----------------------------------------------------------------------
    # Locomotion quality rewards
    # 运动质量奖励
    # -----------------------------------------------------------------------

    def _reward_track_lin_vel_xy(self, std: float = 0.25, command_name: str = "base_velocity"):
        asset = self._get_robot_asset()
        cmd_xy = self._tracking_command(command_name)[:, :2]
        vel_error = cmd_xy - asset.data.root_lin_vel_b[:, :2]
        squared_error = torch.linalg.vector_norm(vel_error, dim=1).square()
        return torch.exp(-squared_error / max(std * std, 1e-6))

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
        """Signed forward-speed reward around the published command.

        If actual vx is below commanded vx, this returns a negative penalty.
        If actual vx is above commanded vx, this returns a positive reward that
        grows with surplus speed, with a cap to avoid overwhelming posture.
        """
        asset = self._get_robot_asset()
        target_vx = self._tracking_command(command_name)[:, 0]
        speed_delta = asset.data.root_lin_vel_b[:, 0] - target_vx

        surplus_part = torch.clamp(speed_delta - deadband, min=0.0, max=max_surplus)
        lag_part = torch.clamp(-speed_delta - deadband, min=0.0, max=max_lag)
        reward_part = surplus_part / max(surplus_scale, 1e-6)
        penalty_part = lag_part / max(lag_scale, 1e-6)
        active = (target_vx > min_command).float()
        return active * (reward_part - lag_penalty_scale * penalty_part)

    def _reward_track_ang_vel_z(self, std: float = 0.25, command_name: str = "base_velocity"):
        asset = self._get_robot_asset()
        target_yaw_rate = self._tracking_command(command_name)[:, 2]
        yaw_rate_error = target_yaw_rate - asset.data.root_ang_vel_b[:, 2]
        inv_std = 1.0 / max(std, 1e-6)
        return torch.exp(-(yaw_rate_error * inv_std).square())

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.5):
        """Reward long steps (feet air time above threshold when moving).

        奖励长步幅（移动时脚部滞空时间超过阈值）。
        Ref: Rudin et al., "Learning to Walk in Minutes", RSS 2022 (legged_gym).
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")
        foot_ids = sensor_cfg.body_ids
        first_contact = contact_sensor.data.current_air_time[:, foot_ids].eq(0.0)
        air_time_margin = contact_sensor.data.last_air_time[:, foot_ids] - threshold
        reward = (air_time_margin * first_contact).sum(dim=1)
        is_moving = self._xy_norm(self._tracking_command(command_name)[:, :2]) > 0.1
        return reward * is_moving.float()

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
        """Reward terrain-aware swing-foot clearance to reduce stair-edge tripping.

        Active only for moving commands and swing feet. The Gaussian target keeps
        the reward bounded: it encourages enough clearance to step over edges,
        but does not reward unnecessarily high, energy-wasting leg lifts.

        The dynamic target is based on local adjacent height-scan deltas, not a
        full-window max-min range, so several stair levels in the scan window are
        less likely to be accumulated into one exaggerated step height.
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        swing = self._contact_force_peak(contact_sensor, sensor_cfg.body_ids) <= 1.0
        foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - asset.data.root_pos_w[:, 2].unsqueeze(1)
        command = self._tracking_command(command_name)
        command_speed = self._xy_norm(command[:, :2])

        terrain_extra = torch.zeros(self.env.num_envs, device=self.env.device)
        height_scanner = self.env.scene.sensors.get("height_scanner")
        if height_scanner is not None:
            scan = height_scanner.data.pos_w[:, 2:3] - height_scanner.data.ray_hits_w[..., 2]
            grid = scan.view(self.env.num_envs, 16, 16)
            forward_window = grid[:, body_y_start:body_y_end, near_x_start:near_x_end]
            if forward_window.shape[-1] > 1 and forward_window.shape[1] > 0:
                step_deltas = (forward_window[:, :, 1:] - forward_window[:, :, :-1]).abs().flatten(1)
                local_step = torch.quantile(step_deltas, delta_quantile, dim=1)
                terrain_extra = torch.clamp(
                    terrain_height_scale * local_step,
                    0.0,
                    max_terrain_extra_height,
                )

        speed_extra = speed_height_scale * torch.clamp(command_speed, 0.0, 1.0)
        dynamic_target_height = target_height + terrain_extra + speed_extra
        height_error = (foot_height - dynamic_target_height.unsqueeze(1)) / max(std, 1e-6)
        clearance_reward = torch.exp(-height_error.square())
        is_moving = command_speed > 0.1
        per_foot_reward = clearance_reward * swing.float()
        return per_foot_reward.sum(dim=1) * is_moving.float() / max(len(asset_cfg.body_ids), 1)

    def _reward_feet_swing_forward(
        self,
        command_name: str = "base_velocity",
        target_forward: float = 0.10,
        std: float = 0.08,
        min_command: float = 0.10,
    ):
        """Reward swing feet moving forward enough to follow the body on stairs.

        This complements feet_clearance: clearance helps the foot avoid stair
        edges vertically, while this term encourages the swing foot—especially
        the rear feet—to reach forward instead of taking a short high step that
        still lands on the stair edge.
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        swing = self._contact_force_peak(contact_sensor, sensor_cfg.body_ids) <= 1.0
        foot_forward = asset.data.body_pos_w[:, asset_cfg.body_ids, 0] - asset.data.root_pos_w[:, 0].unsqueeze(1)
        shortfall = torch.clamp(target_forward - foot_forward, min=0.0)
        forward_reward = torch.exp(-(shortfall / max(std, 1e-6)).square())

        command = self._tracking_command(command_name)
        has_forward_command = command[:, 0] > min_command
        active_reward = forward_reward * swing.float()
        return active_reward.sum(dim=1) * has_forward_command.float() / max(len(asset_cfg.body_ids), 1)

    def _reward_feet_slide(self):
        """Penalize feet sliding on the ground (velocity while in contact).

        惩罚脚部在地面上的滑动（接触时的速度）。
        使用 net_forces_w_history 的 3D 力范数 + 多帧取 max 判定接触。
        Ref: Miki et al., Science Robotics 2022.
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]
        contacts = self._contact_force_peak(contact_sensor, sensor_cfg.body_ids) > 1.0
        body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

    def _reward_joint_position_penalty(
        self,
        stand_still_scale: float = 5.0,
        velocity_threshold: float = 0.1,
        cmd_threshold: float = 0.1,
        ang_cmd_threshold: float = 0.2,
    ):
        """Penalize joint position deviation from default pose.

        惩罚关节位置偏离默认姿态；静止时惩罚倍增（鼓励站立保持默认姿势）。
        Ref: Kumar et al., "RMA: Rapid Motor Adaptation", RSS 2021.

        Bug fix: original code used `cmd > 0.0` (exact-zero check).
        Since velocity commands are sampled from continuous distributions, the
        probability of all three components being EXACTLY 0.0 simultaneously is
        essentially zero — stand_still_scale never fired in practice.
        Fixed to use `cmd > cmd_threshold` (default 0.1 m/s equivalent) so the
        scale activates whenever the robot is genuinely commanded to stand still.

        原始实现使用 `cmd > 0.0`（精确零判断）。由于速度指令来自连续分布，
        三个分量同时精确等于 0 的概率几乎为零，stand_still_scale 实际上从不生效。
        修复为 `cmd > cmd_threshold`（默认 0.1），让真正的零速指令时才放大惩罚。

        Args:
            stand_still_scale: Penalty multiplier when standing still (cmd ≈ 0).
                               静止（cmd ≈ 0）时的惩罚倍数。
            velocity_threshold: Body velocity threshold to confirm robot is not moving (m/s).
                                机体速度阈值，低于此值判定为静止 (m/s)。
            cmd_threshold: Command norm threshold below which robot is "standing still" (m/s).
                           速度指令模长阈值，低于此值视为零速指令 (m/s)。
            ang_cmd_threshold: Yaw-rate command threshold below which robot is treated as not turning.
                               偏航角速度指令阈值，低于此值视为未被要求转向。
        """
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
        return torch.where(
            is_moving,
            deviation,
            stand_still_scale * deviation,
        )

    def _reward_stand_still_motion(
        self,
        command_name: str = "base_velocity",
        lin_cmd_threshold: float = 0.15,
        ang_cmd_threshold: float = 0.2,
        vertical_vel_scale: float = 0.5,
        ang_vel_scale: float = 0.5,
        joint_vel_scale: float = 0.1,
    ):
        """Penalize body oscillation and leg fidgeting under near-zero commands.

        连续命令采样几乎不会精确落在 0 速度，因此“站立不动”往往训练不足。
        本奖励在近零线速度/角速度命令带内激活，直接惩罚机身上下晃动、
        pitch/roll 摇摆以及小腿频繁抬放造成的关节速度。

        Args:
            lin_cmd_threshold: Near-zero threshold for XY linear command norm (m/s).
            ang_cmd_threshold: Near-zero threshold for yaw command magnitude (rad/s).
            vertical_vel_scale: Weight for vertical body velocity in the penalty.
            ang_vel_scale: Weight for pitch/roll angular velocity in the penalty.
            joint_vel_scale: Weight for mean absolute joint velocity in the penalty.
        """
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)

        near_zero_cmd = (
            torch.linalg.norm(cmd[:, :2], dim=1) < lin_cmd_threshold
        ) & (
            torch.abs(cmd[:, 2]) < ang_cmd_threshold
        )

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
        """Penalize staying nearly still when an XY velocity command is present."""
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)
        commanded_to_move = self._xy_norm(cmd[:, :2]) > cmd_threshold
        body_speed = self._xy_norm(asset.data.root_lin_vel_b[:, :2])
        stillness = torch.clamp(
            (still_speed_threshold - body_speed) / max(still_speed_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        return commanded_to_move.float() * stillness

    def _reward_score_guidance(
        self,
        command_name: str = "base_velocity",
        min_command: float = 0.15,
        tracking_std: float = 0.35,
        posture_std: float = 0.25,
        power_scale: float = 35.0,
        posture_weight: float = 0.6,
    ):
        """Small bounded bonus aligned with time, posture, and energy scores.

        The tracking gate prevents the policy from earning the posture/energy bonus
        by standing still when it has a movement command.
        """
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)

        cmd_xy = cmd[:, :2]
        actual_xy = asset.data.root_lin_vel_b[:, :2]
        cmd_speed = self._xy_norm(cmd_xy)
        moving_cmd = cmd_speed > min_command

        vel_error = (cmd_xy - actual_xy).square().sum(dim=1)
        tracking_score = torch.exp(-vel_error / max(tracking_std * tracking_std, 1e-6))

        roll, pitch = self._quat_to_roll_pitch(asset.data.root_quat_w)
        pose_deviation = roll.abs() + pitch.abs()
        pose_deviation = torch.nan_to_num(pose_deviation, nan=0.0, posinf=0.0, neginf=0.0)
        posture_score = torch.exp(-5.0 * pose_deviation)

        power = (asset.data.applied_torque * asset.data.joint_vel).abs().sum(dim=1)
        energy_score = torch.exp(-power / max(power_scale, 1e-6))

        posture_weight = min(max(posture_weight, 0.0), 1.0)
        energy_weight = 1.0 - posture_weight
        score_hint = posture_weight * posture_score + energy_weight * energy_score
        return moving_cmd.float() * tracking_score * score_hint

    def _reward_feet_stumble(self):
        """Penalize feet hitting vertical surfaces (stair edges, walls).

        惩罚脚撞到垂直面（台阶边缘、墙壁）。水平力 > 5× 垂直力时触发。
        阈值 5× 对齐 legged_gym 原版。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        return torch.any(forces_xy > 5 * forces_z, dim=1).float()

    def _reward_termination(self):
        """Penalize real failures (terminated AND NOT timed-out).

        惩罚真正的失败（被终止 且 非超时截断），对应 legged_gym `reset_buf * ~time_out_buf` 逻辑。
        防止策略学会"倒地不起"以规避其他惩罚。
        """
        term_mgr = self.env.termination_manager
        episode_ended = term_mgr.terminated
        timeout = term_mgr.time_outs
        failure = episode_ended & ~timeout
        try:
            if "goal_reached" in term_mgr.active_terms:
                failure = failure & ~term_mgr.get_term("goal_reached")
        except Exception:
            pass
        return failure.float()

    def _reward_action_smoothness(self):
        """2nd-order action smoothness penalty (squared action acceleration).

        动作二阶平滑惩罚（动作加速度的平方和）。
        a_t - 2*a_{t-1} + a_{t-2} 的平方和，比一阶 action_rate 更能抑制抖动。
        在 env 上缓存 prev_prev_action 以跨调用维持状态。
        """
        action_mgr = self.env.action_manager
        curr = action_mgr.action
        prev = action_mgr.prev_action
        if not hasattr(self.env, "_smooth_prev_prev"):
            self.env._smooth_prev_prev = prev.clone()
        prev_prev = self.env._smooth_prev_prev
        accel = curr - prev - (prev - prev_prev)
        self.env._smooth_prev_prev = prev.clone()
        return torch.sum(torch.square(accel), dim=1)

    def _reward_energy(self):
        """Energy penalty: sum of |torque × joint_velocity|.

        能耗惩罚：扭矩绝对值与关节角速度的乘积之和。
        对应赛题 energy 评分项，鼓励高效步态。
        """
        asset = self._get_robot_asset()
        joint_power = (asset.data.applied_torque * asset.data.joint_vel).abs()
        return joint_power.sum(dim=1)

    def _reward_energy_score_formula(self):
        """Platform-aligned energy score: exp(-0.01 * sum(|torque × joint_vel|)).

        与平台评分公式严格对齐：系数 0.01 与 base_scorer.py 中
        energy_score = 100 * exp(-0.01 * mean_energy) 的指数核一致。
        输出范围 (0, 1]，功率越低奖励越接近 1，直接引导策略降低能耗。
        """
        power = self._reward_energy()
        return torch.exp(-0.01 * power)

    def _reward_pose_score_formula(self):
        """Platform-aligned posture score: exp(-5 * (|roll| + |pitch|))."""
        asset = self._get_robot_asset()
        roll, pitch = self._quat_to_roll_pitch(asset.data.root_quat_w)
        pose_deviation = roll.abs() + pitch.abs()
        pose_deviation = torch.nan_to_num(pose_deviation, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.exp(-5.0 * pose_deviation)

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
        """Extra completion bonus for high-pressure episodes.

        Pressure is inferred from the current episode mean energy/posture formula
        scores relative to recent EMA statistics. It is never rewarded by itself;
        only completed envs can receive the bonus.
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

    def _reward_posture_stability(self):
        """Penalize rapid changes in roll and pitch (1st-order finite difference).

        姿态稳定性惩罚：惩罚 roll/pitch 的快速变化（一阶差分）。
        直接指数奖励只惩罚当前偏角大小，无法抑制机身周期性震荡；
        本项惩罚角度的变化速率，鼓励机身平稳过渡而非来回摇摆。
        在楼梯、坡道过渡段尤其有效，可减少因步态节律不稳导致的
        roll/pitch 振荡，直接提升姿态评分。
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

    def _reward_correct_base_height(self, target_height: float = 0.38):
        """Penalize deviation of base height from target (squared).

        惩罚机身高度偏离目标高度（平方误差）。
        对应赛题 posture 评分项。Go2 标准站立高度 ≈ 0.38 m。

        Args:
            target_height: Target base height in meters. / 目标机身高度（米）。
        """
        asset = self._get_robot_asset()
        height_error = asset.data.root_pos_w[:, 2].sub(float(target_height))
        return height_error.square()

    def _reward_hip_to_default(self):
        """Penalize hip joint deviation from default angle (squared sum).

        惩罚髋关节偏离默认角度（平方和）。
        Go2 髋关节为每腿第 0 个关节：FL=0, FR=3, RL=6, RR=9。
        辅助保持自然站姿，防止外八或内八步态。
        """
        asset = self._get_robot_asset()
        hip_idx = [0, 3, 6, 9]
        hip_dev = asset.data.joint_pos[:, hip_idx] - asset.data.default_joint_pos[:, hip_idx]
        return torch.sum(torch.square(hip_dev), dim=1)

    # -----------------------------------------------------------------------
    # Gait quality rewards (Round-4 addition)
    # 步态质量奖励（第四轮新增）
    # -----------------------------------------------------------------------

    def _reward_dof_vel(self):
        """Penalize large joint velocities (L2 norm of joint velocity vector).

        惩罚过大的关节速度（关节速度向量的 L2 范数）。
        关节速度无约束是步态"鬼畜"的根本原因之一；该惩罚鼓励流畅、低速摆腿。
        Ref: agent_diy/feature/reward_process.py
        """
        asset = self._get_robot_asset()
        return asset.data.joint_vel.square().sum(dim=1)

    def _reward_base_lateral_vel(self, command_name: str = "base_velocity"):
        """Penalize untracked lateral (Y-axis) velocity to prevent crab walking.

        惩罚**未被指令覆盖**的侧向速度，防止螃蟹步或横向侧滑。

        注意：不能无条件惩罚 root_lin_vel_b[:,1]，因为速度指令中含有 lin_vel_y
        分量（[-0.3, 0.3]），机器人有时被要求侧向行走。无条件惩罚会与
        track_lin_vel_xy 正向奖励产生梯度冲突。

        正确做法：惩罚 (实际侧向速度 - 指令侧向速度)²，即"横移误差"。
        机器人照指令侧走时误差≈0，不罚；无指令自发侧漂时才受惩罚。
        Ref: custom_rewards.py penalize_base_lat_vel_l2 (error-based variant)
        """
        asset = self._get_robot_asset()
        cmd_vy = self._tracking_command(command_name)[:, 1]
        actual_vy = asset.data.root_lin_vel_b[:, 1]
        return torch.square(actual_vy - cmd_vy)

    def _reward_air_time_variance_penalty(self):
        """Penalize variance in per-foot air time to enforce gait rhythm symmetry.

        惩罚各脚滞空时间的方差，促进对称步态节律（如 trot）。
        对称步态更稳定、更节能，有助于提高 posture 与 energy 评分。
        使用 clamp(max=0.5s) 避免单脚异常值主导方差计算。
        Ref: agent_diy/feature/reward_process.py
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        return torch.var(torch.clamp(last_air_time, max=0.5), dim=1)

    # -----------------------------------------------------------------------
    # Goal-reaching rewards (activated only in track terrain)
    # 目标到达奖励（仅 track 地形时激活）
    # -----------------------------------------------------------------------

    def _zero_goal_state(self):
        """Return empty goal geometry tensors with the standard Track shapes."""
        zero_xy = torch.zeros(self.env.num_envs, 2, device=self.env.device)
        zero_dist = torch.zeros(self.env.num_envs, device=self.env.device)
        return zero_xy, zero_dist

    def _robot_root_pose_for_goal(self):
        try:
            robot = self.env.scene["robot"]
            return robot.data.root_pos_w, robot.data.root_quat_w
        except Exception:
            return None, None

    @staticmethod
    def _heading_from_quat_wxyz(quat: torch.Tensor):
        qw, qx, qy, qz = quat.unbind(dim=1)
        return torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    @staticmethod
    def _rotate_world_xy_to_body(delta_w: torch.Tensor, heading: torch.Tensor):
        cos_yaw = torch.cos(heading)
        sin_yaw = torch.sin(heading)
        body_x = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
        body_y = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
        return torch.stack((body_x, body_y), dim=1)

    def _goal_delta_body(self):
        goal_positions = getattr(self.env, "goal_positions", None)
        if goal_positions is None:
            return self._zero_goal_state()

        root_pos_w, root_quat_w = self._robot_root_pose_for_goal()
        if root_pos_w is None or root_quat_w is None:
            return self._zero_goal_state()

        goal_delta_w = goal_positions[:, :2] - root_pos_w[:, :2]
        heading = self._heading_from_quat_wxyz(root_quat_w)
        goal_delta_b = self._rotate_world_xy_to_body(goal_delta_w, heading)
        return goal_delta_b, torch.linalg.norm(goal_delta_w, dim=1)

    # -----------------------------------------------------------------------
    # Scan helper — pure height_scanner grid, no navigation.py dependency
    # -----------------------------------------------------------------------

    def _height_grid(self):
        scanner = self.env.scene.sensors.get("height_scanner")
        if scanner is None:
            return None
        sensor_height = scanner.data.pos_w[:, 2:3]
        hit_height = scanner.data.ray_hits_w[..., 2]
        return (sensor_height - hit_height).reshape(self.env.num_envs, 16, 16)

    def _goal_vector_body(self):
        local_goal, dist = self._goal_delta_body()
        goal_dir = torch.nn.functional.normalize(local_goal, p=2.0, dim=1, eps=1e-6)
        return local_goal, dist, goal_dir

    @staticmethod
    def _distance_gate(dist: torch.Tensor, threshold: float):
        return (dist > threshold).float()

    def _goal_velocity_projection(self):
        robot = self._get_robot_asset()
        _, dist, goal_dir = self._goal_vector_body()
        projection = torch.sum(robot.data.root_lin_vel_b[:, :2] * goal_dir, dim=1)
        return projection, dist, goal_dir

    def _wall_score_from_sector(
        self,
        sector: torch.Tensor,
        obstacle_threshold: float = -0.75,
        temperature: float = 0.18,
    ):
        if sector.shape[1] == 0 or sector.shape[2] == 0:
            return torch.zeros(sector.shape[0], device=sector.device)
        wall_logits = (obstacle_threshold - sector) / max(temperature, 1e-6)
        wall_prob = torch.sigmoid(wall_logits)
        return wall_prob.flatten(start_dim=1).mean(dim=1)

    def _maze_wall_gate(
        self,
        grid: torch.Tensor,
        goal_dist_gate: float,
        obstacle_threshold: float,
        temperature: float,
    ):
        return self._maze_context_gate(
            grid,
            goal_dist_gate=goal_dist_gate,
            obstacle_threshold=obstacle_threshold,
            temperature=temperature,
        )

    def _fractional_side_wall_mean(self, wall_prob: torch.Tensor, side_width: float, left: bool):
        """Mean side wall probability with fractional row support, e.g. width=4.5."""
        num_rows = wall_prob.shape[1]
        width = max(1.0, min(float(side_width), float(num_rows) * 0.5))
        row_idx = torch.arange(num_rows, device=wall_prob.device, dtype=wall_prob.dtype)
        if left:
            weights = torch.clamp(width - row_idx, min=0.0, max=1.0)
        else:
            weights = torch.clamp(width - (float(num_rows - 1) - row_idx), min=0.0, max=1.0)
        weights = weights.view(1, num_rows, 1)
        return (wall_prob * weights).sum(dim=(1, 2)) / torch.clamp(
            weights.sum() * wall_prob.shape[2], min=1e-6
        )

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
        """Gate maze-only wall rewards away from slopes/stairs.

        Full track is ordered as slopes/stairs first and maze last.  Height scan
        alone is not semantic: stair risers can look like short front walls.  We
        therefore require both a late-track phase (close enough to final goal)
        and a maze-like wall pattern: continuous side walls or a thick front
        blocker.  Thin row-uniform bands are treated as stairs/slopes.
        """
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

        # Stair/slope risers usually span almost the whole track width and only
        # occupy thin depth bands.  Maze walls are laterally localized or thick.
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
        """Diagnostic only: 1 when wall rewards are allowed to behave as maze logic."""
        grid = self._height_grid()
        gate_kwargs = {
            "goal_dist_gate": goal_dist_gate,
            "obstacle_threshold": obstacle_threshold,
            "temperature": temperature,
            "front_cols": front_cols,
        }
        return self._maze_context_gate(
            grid,
            **gate_kwargs,
        )

    def _maze_front_wall_turn_features(
        self,
        obstacle_threshold: float = -0.72,
        temperature: float = 0.18,
        front_cols: int = 6,
        body_y_start: int = 3,
        body_y_end: int = 13,
        side_width: float = 4.5,
        wall_start: float = 0.28,
        wall_full: float = 0.72,
        maze_goal_dist_gate: float = 14.0,
    ):
        grid = self._height_grid()
        if grid is None:
            zeros = torch.zeros(self.env.num_envs, device=self.env.device)
            return zeros, zeros

        front_cols = max(1, min(int(front_cols), grid.shape[2]))
        body_y_start = max(0, int(body_y_start))
        body_y_end = min(int(body_y_end), grid.shape[1])
        if body_y_end <= body_y_start:
            zeros = torch.zeros(self.env.num_envs, device=self.env.device)
            return zeros, zeros

        wall_prob = torch.sigmoid((obstacle_threshold - grid[:, :, :front_cols]) / max(temperature, 1e-6))
        center_wall = wall_prob[:, body_y_start:body_y_end, :].mean(dim=(1, 2))
        left_open = 1.0 - self._fractional_side_wall_mean(wall_prob, side_width, left=True)
        right_open = 1.0 - self._fractional_side_wall_mean(wall_prob, side_width, left=False)
        open_delta = right_open - left_open

        _, _, goal_dir = self._goal_vector_body()
        goal_turn = torch.sign(goal_dir[:, 1])
        visual_turn = torch.sign(open_delta)
        turn_sign = torch.where(torch.abs(open_delta) > 0.08, visual_turn, goal_turn)

        wall_gate = torch.clamp((center_wall - wall_start) / max(wall_full - wall_start, 1e-6), 0.0, 1.0)
        if hasattr(self.env, "goal_positions") and self.env.goal_positions is not None:
            _, goal_dist = self._goal_delta_body()
            phase_gate = (goal_dist < maze_goal_dist_gate).float()
        else:
            phase_gate = torch.ones(self.env.num_envs, device=self.env.device)
        return wall_gate * phase_gate, turn_sign

    def _reward_maze_anticipatory_turn(
        self,
        obstacle_threshold: float = -0.72,
        temperature: float = 0.18,
        front_cols: int = 6,
        body_y_start: int = 3,
        body_y_end: int = 13,
        side_width: float = 4.5,
        wall_start: float = 0.28,
        wall_full: float = 0.72,
        target_yaw_rate: float = 0.75,
        target_forward_speed: float = 0.75,
        maze_goal_dist_gate: float = 14.0,
        near_goal_disable_dist: float = 0.0,
    ):
        """Reward arcing turns toward the more open side before a maze wall."""
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
        )
        if near_goal_disable_dist > 0.0 and hasattr(self.env, "goal_positions") and self.env.goal_positions is not None:
            _, goal_dist = self._goal_delta_body()
            wall_gate = wall_gate * (goal_dist > near_goal_disable_dist).float()
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
        return wall_gate * yaw_toward_opening * speed_score

    def _reward_forward_heading_velocity(
        self,
        target_speed: float = 0.55,
        max_reward: float = 1.0,
    ):
        """Reward moving forward in the robot head/body direction."""
        normalized_vx = self._body_forward_speed() / max(target_speed, 1e-6)
        return torch.clamp(normalized_vx, min=0.0, max=max_reward)

    def _reward_backward_penalty(self, deadband: float = 0.03):
        """Penalize walking backward relative to the robot head direction."""
        reverse_speed = -(self._body_forward_speed() + deadband)
        return torch.clamp(reverse_speed, min=0.0)

    def _reward_goal_heading_alignment(self, std: float = 0.75):
        """Reward body-frame heading consistency with the Track goal direction."""
        _, dist, direction_b = self._goal_vector_body()
        heading_error = torch.atan2(direction_b[:, 1], direction_b[:, 0])
        scaled_error = heading_error / max(std, 1e-6)
        return torch.exp(-(scaled_error * scaled_error)) * self._distance_gate(dist, 0.6)

    def _reward_goal_velocity_projection(self, max_speed: float = 0.75):
        """Reward useful velocity along the current goal ray."""
        projection, dist, _ = self._goal_velocity_projection()
        bounded_projection = torch.clamp(projection / max(max_speed, 1e-6), min=-1.0, max=1.0)
        return bounded_projection * self._distance_gate(dist, 0.6)

    def _reward_goal_backtrack_penalty(self, deadband: float = 0.02):
        """Penalize movement whose body-frame projection increases goal distance."""
        projection, dist, _ = self._goal_velocity_projection()
        retreat_speed = torch.clamp(-(projection + deadband), min=0.0)
        return retreat_speed * self._distance_gate(dist, 0.8)

    def _reward_near_goal_circling_penalty(
        self,
        near_dist: float = 3.5,
        complete_dist: float = 0.6,
        min_progress_speed: float = 0.10,
        yaw_rate_threshold: float = 0.45,
        lateral_speed_threshold: float = 0.18,
    ):
        """Penalize moving/turning near the finish without positive goal progress."""
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
        near_dist: float = 1.6,
        complete_dist: float = 0.6,
        target_speed: float = 0.35,
        yaw_rate_soft_limit: float = 0.35,
        lateral_speed_soft_limit: float = 0.16,
    ):
        """Reward the final straight push into the goal capture radius."""
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
        near_dist: float = 1.6,
        complete_dist: float = 0.6,
        retreat_deadband: float = 0.03,
        target_speed: float = 0.25,
    ):
        """Strongly penalize moving away from the goal after the final approach starts."""
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
        near_dist: float = 3.5,
        complete_dist: float = 0.6,
        miss_margin: float = 0.45,
        reset_dist: float = 5.0,
    ):
        """Penalize drifting away after the robot has already entered the finish area."""
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

    def _reward_goal_distance(self, scale: float = 8.0):
        """Dense bounded reward that increases as the robot gets closer."""
        _, dist = self._goal_delta_body()
        return dist.neg().div(max(scale, 1e-6)).exp()

    def _reward_task_complete(self, threshold: float = 0.6):
        """Sparse completion reward using the same goal geometry helper as dense terms."""
        _, dist = self._goal_delta_body()
        return (dist < threshold).float()

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
        """Small penalty for being close to wall-like geometry."""
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        sector = grid[:, body_y_start:body_y_end, :front_cols]
        wall_score = self._wall_score_from_sector(sector, obstacle_threshold, temperature)
        gate = self._maze_wall_gate(
            grid,
            maze_goal_dist_gate,
            maze_gate_obstacle_threshold,
            temperature,
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
        """Speed-scaled wall penalty: touching slowly is cheap, ramming is expensive."""
        robot = self._get_robot_asset()
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        sector = grid[:, body_y_start:body_y_end, :front_cols]
        wall_score = self._wall_score_from_sector(sector, obstacle_threshold, temperature)
        forward_speed = torch.clamp(robot.data.root_lin_vel_b[:, 0], min=0.0)
        wall_intensity = torch.clamp((wall_score - wall_score_threshold) / max(1.0 - wall_score_threshold, 1e-6), 0.0, 1.0)
        speed_ratio = torch.clamp(
            (forward_speed - slow_speed) / max(impact_speed - slow_speed, 1e-6),
            min=0.0,
            max=1.0,
        )
        penalty = touch_penalty + (impact_penalty - touch_penalty) * torch.square(speed_ratio)
        gate = self._maze_wall_gate(
            grid,
            maze_goal_dist_gate,
            maze_gate_obstacle_threshold,
            temperature,
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
        """Penalty for waiting near a clear front wall or pillar.

        Slow contact is still allowed by wall_collision. This term only fires
        when the scan sees a strong blocker ahead and the body is barely moving,
        which matches the observed wall/pillar timeout failure mode.
        """
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

        body_speed = self._xy_norm(robot.data.root_lin_vel_b[:, :2])
        _, goal_dist = self._goal_delta_body()
        stall_gate = (body_speed < still_speed).float() * (goal_dist > goal_dist_threshold).float()
        maze_gate = self._maze_wall_gate(
            grid,
            maze_goal_dist_gate,
            maze_gate_obstacle_threshold,
            temperature,
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
        """Tiny reward for staying in locally open space."""
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        sector = grid[:, body_y_start:body_y_end, :front_cols]
        gate = self._maze_wall_gate(
            grid,
            maze_goal_dist_gate,
            maze_gate_obstacle_threshold,
            0.18,
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
        """Penalize off-center walking only when both corridor walls are visible."""
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
        left_dist = torch.sum(left_weight * dist_to_center, dim=1) / torch.clamp(
            left_weight.sum(dim=1), min=1e-6
        )
        right_dist = torch.sum(right_weight * dist_to_center, dim=1) / torch.clamp(
            right_weight.sum(dim=1), min=1e-6
        )
        imbalance = torch.abs(left_dist - right_dist) / torch.clamp(left_dist + right_dist, min=1e-6)
        maze_gate = self._maze_wall_gate(
            grid,
            maze_goal_dist_gate,
            maze_gate_obstacle_threshold,
            temperature,
        )
        return corridor_gate * imbalance * maze_gate

    def _reward_directed_exploration(
        self,
        radius: float = 0.55,
        memory_size: int = 96,
        goal_heading_std: float = 1.0,
    ):
        """Tiny novelty reward, gated by target-facing direction.

        This prevents the policy from getting paid for random wandering away
        from the maze goal.
        """
        robot = self._get_robot_asset()
        current_xy = robot.data.root_pos_w[:, :2]
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
        dist_to_seen = torch.linalg.norm(visit_pos - current_xy.unsqueeze(1), dim=2)
        dist_to_seen = torch.where(valid, dist_to_seen, torch.full_like(dist_to_seen, 1e6))
        novel = dist_to_seen.min(dim=1).values > radius

        _, goal_dist, goal_dir = self._goal_vector_body()
        angle_error = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])
        inv_heading_std = 1.0 / max(goal_heading_std, 1e-6)
        toward_goal_gate = torch.exp(-(angle_error * inv_heading_std).square())
        far_from_goal = self._distance_gate(goal_dist, 1.0)
        reward = novel.float() * toward_goal_gate * far_from_goal

        ptr = self.env._rl_nav_visit_ptr
        env_ids = torch.arange(num_envs, device=device)
        if novel.any():
            add_ids = env_ids[novel]
            add_ptr = ptr[novel]
            visit_pos[add_ids, add_ptr] = current_xy[novel]
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

    def _reward_approach_goal(self):
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return self._blank_reward()

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

        previous_dist = self.env._nav_previous_goal_dist
        delta = current_dist - previous_dist
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        valid_mask = self.env._nav_previous_goal_valid & ~reset_mask
        delta = torch.where(valid_mask, delta, torch.zeros_like(delta))
        self.env._nav_previous_goal_dist = current_dist.clone()
        self.env._nav_previous_goal_valid = ~reset_mask
        return -delta

    def _reward_stuck_penalty(self, min_command: float = 0.15, still_speed: float = 0.08):
        robot = self._get_robot_asset()
        _, dist = self._goal_delta_body()
        cmd = self._tracking_command("base_velocity")
        commanded_motion = self._xy_norm(cmd[:, :2]) > min_command
        nearly_static = self._xy_norm(robot.data.root_lin_vel_b[:, :2]) < still_speed
        still_relevant = self._distance_gate(dist, 0.8).bool()
        return (commanded_motion & nearly_static & still_relevant).float()

    def _reward_long_non_foot_contact(
        self,
        force_threshold: float = 5.0,
        duration_s: float = 1.0,
        step_dt: float = 0.02,
        max_penalty: float = 2.0,
        maze_only: bool = True,
        maze_goal_dist_gate: float = 14.0,
    ):
        """Penalize sustained real contact on non-foot bodies, useful for corner jams."""
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

    def _reward_navigation_time(self):
        return self._unit_reward()
