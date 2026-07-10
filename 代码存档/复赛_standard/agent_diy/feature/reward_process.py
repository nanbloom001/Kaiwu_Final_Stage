# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):
    """
    Custom reward processor with user-defined reward terms
    自定义奖励处理器，包含用户自定义的奖励项
    """

    def _reward_flat_orientation(self):
        """Penalize non-flat base orientation (deviation from upright).

        惩罚非平坦的基座朝向（偏离直立）。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    def _reward_joint_vel(self):
        """Penalize large joint velocities.

        惩罚大的关节速度。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.joint_vel), dim=1)

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.5):
        """Reward long steps (feet air time above threshold when moving).

        奖励长步幅（移动时脚部滞空时间超过阈值）。

        Args:
            command_name: Command term name. / 命令项名称。
            threshold: Minimum air time threshold. / 最小滞空时间阈值。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")
        # Compute reward
        # 计算奖励
        first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
        # No reward for zero commands
        # 当命令为零时不给奖励
        is_moving = torch.norm(self.env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return reward * is_moving.float()

    def _reward_air_time_variance_penalty(self):
        """Penalize variance in foot air/contact time (gait symmetry).

        惩罚脚部滞空/接触时间的方差（步态对称性）。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
        return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
            torch.clip(last_contact_time, max=0.5), dim=1
        )

    def _reward_feet_slide(self):
        """Penalize feet sliding on the ground (velocity while in contact).

        惩罚脚部在地面上的滑动（接触时的速度）。
        对齐 Isaac Lab feet_slide：用 net_forces_w_history 的 3D 力范数 + 多帧取 max 判定接触。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]
        # Check which feet are in contact (3D force norm, max over history frames)
        # 检查哪些脚在接触地面（3D 力的范数，历史帧取最大值）
        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
        # Get foot velocities (xy only)
        # 获取脚部速度（仅 xy 分量）
        body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        # Penalize velocity when in contact
        # 接触时惩罚速度
        reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
        return reward

    def _reward_joint_position_penalty(self, stand_still_scale: float = 5.0, velocity_threshold: float = 0.1):
        """Penalize joint position error from default pose.

        惩罚关节位置偏离默认姿态。

        Args:
            stand_still_scale: Scale factor when standing still. / 静止时的缩放因子。
            velocity_threshold: Velocity threshold to determine if moving. / 判断是否移动的速度阈值。
        """
        asset = self._get_robot_asset()
        cmd = torch.linalg.norm(self.env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        reward = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
            reward,
            stand_still_scale * reward,
        )

    # def _reward_standing_posture(self):
    #     """Reward maintaining default standing posture (for standing task)."""
    #     asset = self._get_robot_asset()
    #     joint_deviation = torch.sum(
    #         torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
    #     )
    #     return torch.exp(-joint_deviation * 10.0)

    # def _reward_feet_contact(self):
    #     """Reward four feet in contact (for standing task)."""
    #     contact = self._get_foot_contact()
    #     num_feet_in_contact = contact.sum(dim=1).float()
    #     return (num_feet_in_contact == 4).float()

    # def _reward_stand_velocity(self):
    #     """Penalize any movement (for standing task)."""
    #     asset = self._get_robot_asset()
    #     linear_vel_penalty = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    #     angular_vel_penalty = torch.square(asset.data.root_ang_vel_b[:, 2])
    #     return linear_vel_penalty + angular_vel_penalty

    def _reward_obstacle_evasion(
        self,
        command_name: str = "base_velocity",
        obstacle_threshold: float = -0.3,
        near_x_end: int = 10,
        body_y_start: int = 3,
        body_y_end: int = 13,
        turn_std: float = 0.5,
    ):
        """Penalize forward-blocked path when robot is not actively turning.

        惩罚前方被障碍阻挡时未主动转向。

        Uses height_scan near-field window to detect tall obstacles (pillars/walls)
        directly ahead, and angular velocity to detect evasion turning.
        使用 height_scan 近场窗口检测正前方高障碍物，用角速度检测转向。

        Returns: blocked * not_evading * has_fwd_cmd
        返回：blocked * not_evading * has_fwd_cmd

        Grid layout (16x16, offset 0.75m fwd, res=0.1m):
          reshaped (N, 16, 16) -> dim0=y_idx, dim1=x_idx
          y: -0.75m(idx0) .. +0.75m(idx15)
          x: 0.0m(idx0) .. 1.5m(idx15)
        网格布局（16x16，前方偏移 0.75m，分辨率 0.1m）：
          reshape 为 (N, 16, 16) -> dim0=y_idx，dim1=x_idx
          y：-0.75m(idx0) .. +0.75m(idx15)
          x：0.0m(idx0) .. 1.5m(idx15)

        Default window:
          Y [3:13] = -0.45m ~ +0.55m (≈ passage width, catches side walls)
          X [:10]  = 0.0m ~ 0.9m (≈1s reaction at 0.5~1.0 m/s)
        默认窗口：
          Y [3:13] = -0.45m ~ +0.55m（≈通道宽度，可捕捉侧壁）
          X [:10]  = 0.0m ~ 0.9m（在 0.5~1.0 m/s 下约 1s 反应距离）
        """
        asset = self._get_robot_asset()
        sensor = self.env.scene.sensors["height_scanner"]

        # raw height: base_z - hit_z (positive=ground below, negative=obstacle above)
        # 原始高度：base_z - hit_z（正值=下方为地面，负值=上方有障碍）
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)

        # near-field body-width window
        # 近场、身体宽度的窗口
        window = grid[:, body_y_start:body_y_end, :near_x_end]

        # column-projection: for each y-strip, any obstacle in forward range?
        # 列投影：每个 y 条带在前方范围内是否存在障碍物
        col_blocked = (window < obstacle_threshold).any(dim=-1).float()
        blocked = col_blocked.mean(dim=-1)

        # evasion signal: turning hard -> low penalty
        # 规避信号：转弯幅度大 -> 惩罚低
        yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
        not_evading = torch.exp(-yaw_rate / turn_std)

        # gate: only when forward command exists
        # 门控：仅在存在前进指令时生效
        cmd = self.env.command_manager.get_command(command_name)
        has_fwd_cmd = (cmd[:, 0] > 0.05).float()

        return blocked * not_evading * has_fwd_cmd

    def _reward_feet_stumble(self):
        """Penalize feet hitting vertical surfaces (stair edges, walls).

        惩罚脚撞到垂直面。阈值 5× 对齐 legged_gym 原版。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]

        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)

        return torch.any(forces_xy > 5 * forces_z, dim=1).float()

    # -----------------------------------------------------------------------
    # Navigation rewards (Stage 3)
    # 导航奖励（第三阶段）
    # -----------------------------------------------------------------------

    def _reward_approach_goal(self):
        """Reward approaching the maze exit: -(current_dist - previous_dist).

        接近迷宫出口奖励：距离减少→正奖励，距离增加→负奖励。

        Requires env.goal_positions to be set by TerrainExitManager
        (auto-initialized via observation_process.goal_position_in_robot_frame).
        需要 TerrainExitManager 设置 env.goal_positions
        （通过 observation_process.goal_position_in_robot_frame 自动初始化）。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]  # (N, 2)
        goal_pos = self.env.goal_positions[:, :2]  # (N, 2)

        current_dist = torch.norm(goal_pos - robot_pos, dim=1)  # (N,)

        # 首次调用：初始化 previous_dist，返回零
        if not hasattr(self.env, "_previous_goal_dist") or self.env._previous_goal_dist is None:
            self.env._previous_goal_dist = current_dist.clone()
            return torch.zeros(self.env.num_envs, device=self.env.device)

        # 距离变化（正=远离，负=接近）
        delta_dist = current_dist - self.env._previous_goal_dist

        # 重置的 env 不计算 delta（距离跳变）
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        delta_dist[reset_mask] = 0.0

        # 更新 previous
        self.env._previous_goal_dist = current_dist.clone()

        # 返回负的距离变化 = 接近→正奖励
        return -delta_dist

    def _reward_reach_goal(self, threshold: float = 0.5):
        """Reward reaching the maze exit (distance < threshold).

        到达迷宫出口奖励（距离 < 阈值时返回 1.0）。

        Args:
            threshold: Distance threshold to consider goal reached (m).
                       判定到达目标的距离阈值（米）。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]

        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()

    def _reward_navigation_time(self):
        """Per-step penalty to encourage fast navigation.

        每步固定惩罚，鼓励快速到达出口。返回固定值 1.0，由 weight 控制大小。
        """
        return torch.ones(self.env.num_envs, device=self.env.device)

    # --- Termination penalty / 终止惩罚 ---
    def _reward_termination(self):
        """Penalize real failures (terminated AND NOT timed-out AND NOT goal-reached).

        惩罚真正的失败（被终止且非超时截断且非到达目标），对应经典 legged_gym 的
        `reset_buf * ~time_out_buf` 逻辑，同时排除导航成功终止。

        Returns:
            Float tensor (num_envs,): 1.0 for real failures, 0.0 otherwise.
            浮点张量 (num_envs,)：真实失败返回 1.0，其他情况返回 0.0。
        """
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs

        # 排除 goal_reached（导航成功不应被惩罚）
        if "goal_reached" in term_mgr.active_terms:
            goal_done = term_mgr.get_term("goal_reached")
            failure = failure & ~goal_done

        return failure.float()
