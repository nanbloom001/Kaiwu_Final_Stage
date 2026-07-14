#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    #
    # Note: this builder only keeps metrics that are unique to algorithm training
    # (loss-series metrics, episode_reward, track traversal progress).
    # Other reward_* metrics (velocity tracking, posture, gait, navigation rewards, etc.)
    # are rendered by the project-side tools/conf/monitor_default.yaml and
    # tools/conf/monitor_default_track.yaml, and are no longer redefined here,
    # to avoid duplicated panels with the same name in the final merged dashboard.
    #
    # 注意：本 builder 只保留算法训练独有的指标（loss 类、episode_reward、赛道穿越进度）。
    # 其余 reward_* 指标（速度跟踪、姿态、步态、导航奖励等）由项目侧
    # tools/conf/monitor_default.yaml 与 tools/conf/monitor_default_track.yaml 负责展示，
    # 这里不再重复定义，避免最终合并后的监控面板出现同名指标重复绘制。

    Returns:
        dict: monitor configuration dictionary
        返回值：监控配置字典
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("四足机器人导航")
        # ==============================================================
        # Group 1: Algorithm training loss metrics (unique to this builder, not covered by yaml)
        # Group 1: 算法训练损失指标（本 builder 独有，yaml 未覆盖）
        # ==============================================================
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .end_group()
        # ==============================================================
        # Group 2: Reward metrics (examples, players can add more reward panels as needed)
        # Group 2: Reward 指标（示例，选手可按需补充更多 reward 面板）
        # ==============================================================
        .add_group(group_name="奖励指标", group_name_en="reward")
        # 平台默认已有（不重复注册）：track_lin_vel_xy / track_ang_vel_z
        # / undesired_contacts / dof_pos_limits / flat_orientation / termination

        # --- 导航 reward（第三组，自定义，平台无默认面板）---
        .add_panel(name="朝目标推进", name_en="reward_approach_goal", type="line")
            .add_metric(metrics_name="reward_approach_goal",
                        expr="avg(reward_approach_goal{})")
            .end_panel()
        .add_panel(name="完成奖励", name_en="reward_task_complete", type="line")
            .add_metric(metrics_name="reward_task_complete",
                        expr="avg(reward_task_complete{})")
            .end_panel()
        .add_panel(name="目标速度投影", name_en="reward_goal_velocity_projection", type="line")
            .add_metric(metrics_name="reward_goal_velocity_projection",
                        expr="avg(reward_goal_velocity_projection{})")
            .end_panel()
        .add_panel(name="目标朝向对齐", name_en="reward_goal_heading_alignment", type="line")
            .add_metric(metrics_name="reward_goal_heading_alignment",
                        expr="avg(reward_goal_heading_alignment{})")
            .end_panel()
        .add_panel(name="目标距离奖励", name_en="reward_goal_distance", type="line")
            .add_metric(metrics_name="reward_goal_distance",
                        expr="avg(reward_goal_distance{})")
            .end_panel()

        # --- 步态/姿态 reward（第一/二组，自定义，平台无默认面板）---
        .add_panel(name="姿态稳定惩罚", name_en="reward_posture_stability", type="line")
            .add_metric(metrics_name="reward_posture_stability",
                        expr="avg(reward_posture_stability{})")
            .end_panel()
        .add_panel(name="足部腾空时间", name_en="reward_feet_air_time", type="line")
            .add_metric(metrics_name="reward_feet_air_time",
                        expr="avg(reward_feet_air_time{})")
            .end_panel()
        .add_panel(name="足部离地间隙", name_en="reward_feet_clearance", type="line")
            .add_metric(metrics_name="reward_feet_clearance",
                        expr="avg(reward_feet_clearance{})")
            .end_panel()
        .add_panel(name="足部前摆", name_en="reward_feet_swing_forward", type="line")
            .add_metric(metrics_name="reward_feet_swing_forward",
                        expr="avg(reward_feet_swing_forward{})")
            .end_panel()
        .add_panel(name="步态方差惩罚", name_en="reward_air_time_variance_penalty", type="line")
            .add_metric(metrics_name="reward_air_time_variance_penalty",
                        expr="avg(reward_air_time_variance_penalty{})")
            .end_panel()
        .add_panel(name="足部滑动惩罚", name_en="reward_feet_slide", type="line")
            .add_metric(metrics_name="reward_feet_slide",
                        expr="avg(reward_feet_slide{})")
            .end_panel()
        .add_panel(name="足部绊碰惩罚", name_en="reward_feet_stumble", type="line")
            .add_metric(metrics_name="reward_feet_stumble",
                        expr="avg(reward_feet_stumble{})")
            .end_panel()

        # --- 运动质量 reward（Stage 3 重点：能耗/关节/动作平滑）---
        .add_panel(name="能耗", name_en="reward_energy", type="line")
            .add_metric(metrics_name="reward_energy",
                        expr="avg(reward_energy{})")
            .end_panel()
        .add_panel(name="关节加速度", name_en="reward_joint_acc", type="line")
            .add_metric(metrics_name="reward_joint_acc",
                        expr="avg(reward_joint_acc{})")
            .end_panel()
        .add_panel(name="关节力矩", name_en="reward_joint_torques", type="line")
            .add_metric(metrics_name="reward_joint_torques",
                        expr="avg(reward_joint_torques{})")
            .end_panel()
        .add_panel(name="动作变化率", name_en="reward_action_rate", type="line")
            .add_metric(metrics_name="reward_action_rate",
                        expr="avg(reward_action_rate{})")
            .end_panel()
        .add_panel(name="动作平滑度", name_en="reward_action_smoothness", type="line")
            .add_metric(metrics_name="reward_action_smoothness",
                        expr="avg(reward_action_smoothness{})")
            .end_panel()
        .add_panel(name="关节限位惩罚", name_en="reward_dof_pos_limits", type="line")
            .add_metric(metrics_name="reward_dof_pos_limits",
                        expr="avg(reward_dof_pos_limits{})")
            .end_panel()
        .end_group()
        # ==============================================================
        # Group 3: 命令诊断指标（Stage4A）
        # ==============================================================
        .add_group(group_name="命令诊断", group_name_en="command_diag")
        .add_panel(name="目标速度vx", name_en="obs_cmd_vel_x", type="line")
            .add_metric(metrics_name="obs_cmd_vel_x", expr="avg(obs_cmd_vel_x{})")
            .end_panel()
        .add_panel(name="目标速度vy", name_en="obs_cmd_vel_y", type="line")
            .add_metric(metrics_name="obs_cmd_vel_y", expr="avg(obs_cmd_vel_y{})")
            .end_panel()
        .add_panel(name="目标角速度wz", name_en="obs_cmd_yaw", type="line")
            .add_metric(metrics_name="obs_cmd_yaw", expr="avg(obs_cmd_yaw{})")
            .end_panel()
        .add_panel(name="实际速度vx", name_en="obs_actual_vel_x", type="line")
            .add_metric(metrics_name="obs_actual_vel_x", expr="avg(obs_actual_vel_x{})")
            .end_panel()
        .add_panel(name="速度跟踪误差vx", name_en="obs_lin_vel_x_error", type="line")
            .add_metric(metrics_name="obs_lin_vel_x_error", expr="avg(obs_lin_vel_x_error{})")
            .end_panel()
        .add_panel(name="速度跟踪误差yaw", name_en="obs_yaw_error", type="line")
            .add_metric(metrics_name="obs_yaw_error", expr="avg(obs_yaw_error{})")
            .end_panel()
        .add_panel(name="机身高度", name_en="obs_base_height", type="line")
            .add_metric(metrics_name="obs_base_height", expr="avg(obs_base_height{})")
            .end_panel()
        .add_panel(name="角速度xy范数", name_en="obs_ang_vel_xy", type="line")
            .add_metric(metrics_name="obs_ang_vel_xy", expr="avg(obs_ang_vel_xy{})")
            .end_panel()
        .end_group()
        # ==============================================================
        # Group 4: 门控诊断指标（Stage4A）
        # ==============================================================
        .add_group(group_name="门控诊断", group_name_en="gate_diag")
        .add_panel(name="门控地形-flat", name_en="gate_worker_final_flat_ratio", type="line")
            .add_metric(metrics_name="gate_worker_final_flat_ratio", expr="avg(gate_worker_final_flat_ratio{})")
            .end_panel()
        .add_panel(name="门控地形-slope", name_en="gate_worker_final_slope_ratio", type="line")
            .add_metric(metrics_name="gate_worker_final_slope_ratio", expr="avg(gate_worker_final_slope_ratio{})")
            .end_panel()
        .add_panel(name="门控地形-stairs", name_en="gate_worker_final_stairs_ratio", type="line")
            .add_metric(metrics_name="gate_worker_final_stairs_ratio", expr="avg(gate_worker_final_stairs_ratio{})")
            .end_panel()
        .add_panel(name="门控地形-maze", name_en="gate_worker_final_maze_ratio", type="line")
            .add_metric(metrics_name="gate_worker_final_maze_ratio", expr="avg(gate_worker_final_maze_ratio{})")
            .end_panel()
        .add_panel(name="门控目标vx", name_en="gate_worker_target_cmd_vx", type="line")
            .add_metric(metrics_name="gate_worker_target_cmd_vx", expr="avg(gate_worker_target_cmd_vx{})")
            .end_panel()
        .add_panel(name="门控写入vx", name_en="gate_worker_worker_cmd_vx", type="line")
            .add_metric(metrics_name="gate_worker_worker_cmd_vx", expr="avg(gate_worker_worker_cmd_vx{})")
            .end_panel()
        .add_panel(name="命令写入率", name_en="gate_worker_command_written_ratio", type="line")
            .add_metric(metrics_name="gate_worker_command_written_ratio", expr="avg(gate_worker_command_written_ratio{})")
            .end_panel()
        .add_panel(name="sticky保持步数", name_en="gate_worker_sticky_hold_steps", type="line")
            .add_metric(metrics_name="gate_worker_sticky_hold_steps", expr="avg(gate_worker_sticky_hold_steps{})")
            .end_panel()
        .add_panel(name="策略命令误差", name_en="gate_worker_policy_cmd_error", type="line")
            .add_metric(metrics_name="gate_worker_policy_cmd_error", expr="avg(gate_worker_policy_cmd_error{})")
            .end_panel()
        .add_panel(name="critic命令误差", name_en="gate_worker_critic_cmd_error", type="line")
            .add_metric(metrics_name="gate_worker_critic_cmd_error", expr="avg(gate_worker_critic_cmd_error{})")
            .end_panel()
        .end_group()
        .build()
    )
    return config_dict
