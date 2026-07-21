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
    # Panel organization (Phase-0: flat-ground gait shaping):
    # 面板组织（Phase-0 平地步态塑形）：
    #
    # Group 1: 算法指标   — PPO loss curves
    # Group 2: 速度跟踪   — velocity tracking rewards (primary positive signal)
    # Group 3: 姿态质量   — posture rewards (Phase-0 primary training objective)
    # Group 4: 步态质量   — gait quality rewards (Phase-0 secondary objective)
    # Group 5: 稳定/接触  — stability & contact penalties
    # Group 6: 关节/动作  — joint & action smoothness penalties
    # Group 7: 能耗       — energy / torque penalties (competition scoring)

    Returns:
        dict: monitor configuration dictionary
        返回值：监控配置字典
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("四足机器人导航")
        # ==============================================================
        # Group 0: Training progress — reward-weight-agnostic convergence signals
        # Group 0: 训练进展 — 与 reward 权重无关的收敛信号
        #
        # mean_episode_length: average steps per episode.
        #   Monotonically rising → robot survives longer → genuine convergence.
        #   Plateau at low value → robot keeps falling → weight or algo problem.
        #
        # mean_episode_reward: cumulative reward summed over one episode.
        #   Combines all weighted terms; the one true "fitness" curve.
        #   Use this to compare runs with different weight configurations.
        #
        # mean_episode_length: 每 episode 的平均存活步数。
        #   单调上升 → 机器人越来越耐摔 → 真正收敛。
        #   早期很低并长期平台 → 机器人一直倒 → 权重或算法有问题。
        #
        # mean_episode_reward: 每 episode 全部加权奖励之和。
        #   是唯一综合所有项的真实适应度曲线，用来跨配置对比训练。
        # ==============================================================
        .add_group(group_name="训练进展", group_name_en="training_progress")
        .add_panel(name="平均 episode 步数", name_en="mean_episode_length", type="line")
            .add_metric(metrics_name="mean_episode_length",
                        expr="avg(mean_episode_length{})")
            .end_panel()
        .add_panel(name="每 episode 累计奖励", name_en="mean_episode_reward", type="line")
            .add_metric(metrics_name="mean_episode_reward",
                        expr="avg(mean_episode_reward{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 1: PPO algorithm loss curves
        # Group 1: PPO 算法损失曲线
        # ==============================================================
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="总损失", name_en="total_loss", type="line")
            .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
            .end_panel()
        .add_panel(name="价值损失", name_en="value_loss", type="line")
            .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
            .end_panel()
        .add_panel(name="策略损失", name_en="policy_loss", type="line")
            .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
            .end_panel()
        .add_panel(name="熵损失", name_en="entropy_loss", type="line")
            .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 2: Velocity tracking (primary positive reward signal)
        # Group 2: 速度跟踪（主正向奖励信号）
        # ==============================================================
        .add_group(group_name="速度跟踪", group_name_en="velocity_tracking")
        .add_panel(name="线速度跟踪", name_en="reward_track_lin_vel_xy", type="line")
            .add_metric(metrics_name="reward_track_lin_vel_xy",
                        expr="avg(reward_track_lin_vel_xy{})")
            .end_panel()
        .add_panel(name="偏航角速度跟踪", name_en="reward_track_ang_vel_z", type="line")
            .add_metric(metrics_name="reward_track_ang_vel_z",
                        expr="avg(reward_track_ang_vel_z{})")
            .end_panel()
        .add_panel(name="课程追踪比例", name_en="vel_curriculum_tracking_ratio", type="line")
            .add_metric(metrics_name="vel_curriculum_tracking_ratio",
                        expr="avg(vel_curriculum_tracking_ratio{})")
            .end_panel()
        .add_panel(name="速度课程等级", name_en="vel_curriculum_stage", type="line")
            .add_metric(metrics_name="vel_curriculum_stage",
                        expr="avg(vel_curriculum_stage{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 3: Posture quality (Phase-0 PRIMARY objective)
        # Group 3: 姿态质量（Phase-0 核心训练目标）
        # ==============================================================
        .add_group(group_name="姿态质量", group_name_en="posture_quality")
        .add_panel(name="机身水平姿态惩罚", name_en="reward_flat_orientation", type="line")
            .add_metric(metrics_name="reward_flat_orientation",
                        expr="avg(reward_flat_orientation{})")
            .end_panel()
        .add_panel(name="机身高度惩罚", name_en="reward_correct_base_height", type="line")
            .add_metric(metrics_name="reward_correct_base_height",
                        expr="avg(reward_correct_base_height{})")
            .end_panel()
        .add_panel(name="全身关节偏离惩罚", name_en="reward_joint_position_penalty", type="line")
            .add_metric(metrics_name="reward_joint_position_penalty",
                        expr="avg(reward_joint_position_penalty{})")
            .end_panel()
        .add_panel(name="侧向漂移惩罚", name_en="reward_base_lateral_vel", type="line")
            .add_metric(metrics_name="reward_base_lateral_vel",
                        expr="avg(reward_base_lateral_vel{})")
            .end_panel()
        .add_panel(name="pitch roll 角速度惩罚", name_en="reward_ang_vel_xy", type="line")
            .add_metric(metrics_name="reward_ang_vel_xy",
                        expr="avg(reward_ang_vel_xy{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 4: Gait quality (Phase-0 secondary objective)
        # Group 4: 步态质量（Phase-0 次要训练目标）
        # ==============================================================
        .add_group(group_name="步态质量", group_name_en="gait_quality")
        .add_panel(name="脚部滞空时间奖励", name_en="reward_feet_air_time", type="line")
            .add_metric(metrics_name="reward_feet_air_time",
                        expr="avg(reward_feet_air_time{})")
            .end_panel()
        .add_panel(name="关节速度惩罚", name_en="reward_dof_vel", type="line")
            .add_metric(metrics_name="reward_dof_vel",
                        expr="avg(reward_dof_vel{})")
            .end_panel()
        .add_panel(name="步态对称性惩罚", name_en="reward_air_time_variance_penalty", type="line")
            .add_metric(metrics_name="reward_air_time_variance_penalty",
                        expr="avg(reward_air_time_variance_penalty{})")
            .end_panel()
        .add_panel(name="原地旋转惩罚", name_en="reward_pivot_turning", type="line")
            .add_metric(metrics_name="reward_pivot_turning",
                        expr="avg(reward_pivot_turning{})")
            .end_panel()
        .add_panel(name="脚部打滑惩罚", name_en="reward_feet_slide", type="line")
            .add_metric(metrics_name="reward_feet_slide",
                        expr="avg(reward_feet_slide{})")
            .end_panel()
        .add_panel(name="脚撞台阶边缘惩罚", name_en="reward_feet_stumble", type="line")
            .add_metric(metrics_name="reward_feet_stumble",
                        expr="avg(reward_feet_stumble{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 5: Stability & contact penalties
        # Group 5: 稳定性与接触惩罚
        # ==============================================================
        .add_group(group_name="稳定接触", group_name_en="stability_contact")
        .add_panel(name="垂直速度惩罚", name_en="reward_lin_vel_z", type="line")
            .add_metric(metrics_name="reward_lin_vel_z",
                        expr="avg(reward_lin_vel_z{})")
            .end_panel()
        .add_panel(name="非预期接触惩罚", name_en="reward_undesired_contacts", type="line")
            .add_metric(metrics_name="reward_undesired_contacts",
                        expr="avg(reward_undesired_contacts{})")
            .end_panel()
        .add_panel(name="终止惩罚", name_en="reward_termination", type="line")
            .add_metric(metrics_name="reward_termination",
                        expr="avg(reward_termination{})")
            .end_panel()
        .add_panel(name="关节位置极限惩罚", name_en="reward_dof_pos_limits", type="line")
            .add_metric(metrics_name="reward_dof_pos_limits",
                        expr="avg(reward_dof_pos_limits{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 6: Joint & action smoothness penalties
        # Group 6: 关节与动作平滑惩罚
        # ==============================================================
        .add_group(group_name="关节动作平滑", group_name_en="joint_action_smoothness")
        .add_panel(name="关节加速度惩罚", name_en="reward_joint_acc", type="line")
            .add_metric(metrics_name="reward_joint_acc",
                        expr="avg(reward_joint_acc{})")
            .end_panel()
        .add_panel(name="动作变化率惩罚 一阶", name_en="reward_action_rate", type="line")
            .add_metric(metrics_name="reward_action_rate",
                        expr="avg(reward_action_rate{})")
            .end_panel()
        .add_panel(name="动作平滑惩罚 二阶", name_en="reward_action_smoothness", type="line")
            .add_metric(metrics_name="reward_action_smoothness",
                        expr="avg(reward_action_smoothness{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 7: Energy / torque (competition scoring items)
        # Group 7: 能耗 / 扭矩（赛题评分项）
        # ==============================================================
        .add_group(group_name="能耗扭矩", group_name_en="energy_torque")
        .add_panel(name="能耗惩罚", name_en="reward_energy", type="line")
            .add_metric(metrics_name="reward_energy",
                        expr="avg(reward_energy{})")
            .end_panel()
        .add_panel(name="关节扭矩惩罚", name_en="reward_joint_torques", type="line")
            .add_metric(metrics_name="reward_joint_torques",
                        expr="avg(reward_joint_torques{})")
            .end_panel()
        .end_group()

        # ==============================================================
        # Group 8: Physical observations (SI units, weight-independent)
        # Group 8: 物理观测量（SI 单位，与 reward 权重无关）
        #
        # These panels show the *physical* quantities that the reward functions
        # operate on, NOT the weighted reward values.  They let you answer
        # "is the robot actually converging in terms of real physics?"
        # independently of whether you've tuned the reward weights well.
        #
        # 这些面板显示奖励函数实际计算所用的物理量（SI 单位），
        # 而非加权后的奖励值。用于独立于权重设置判断"机器人物理收敛了吗"。
        #
        # obs_lin_vel_x_error: mean |cmd_vx - actual_vx| (m/s)
        #   → 速度追踪误差（前向）；收敛后应趋近 0 m/s。
        #     值持续 >0.3 m/s 且 reward 已平台 → 权重可能有矛盾梯度。
        #
        # obs_lin_vel_y_error: mean |cmd_vy - actual_vy| (m/s)
        #   → 速度追踪误差（侧向）；正常约 0.1 m/s 以内。
        #
        # obs_actual_vel_x: mean actual forward speed (m/s)
        #   → 机器人实际平均前向速度；Stage-0 目标 0.25 m/s，随阶段升高。
        #
        # obs_base_height: mean base height (m)
        #   → 机身高度均值；目标 0.38 m。
        #     偏低说明腿弯曲不足（或摔倒率高）；偏高说明过度伸展。
        #
        # obs_ang_vel_xy: mean |ω_pitch, ω_roll| magnitude (rad/s)
        #   → pitch/roll 角速度幅值；反映机身倾斜程度。
        #     正常行走约 0.3–0.8 rad/s；持续 >1.5 说明姿态不稳定。
        # ==============================================================
        .add_group(group_name="物理观测量", group_name_en="physics_obs")
        .add_panel(name="前向速度追踪误差", name_en="obs_lin_vel_x_error", type="line")
            .add_metric(metrics_name="obs_lin_vel_x_error",
                        expr="avg(obs_lin_vel_x_error{})")
            .end_panel()
        .add_panel(name="侧向速度追踪误差", name_en="obs_lin_vel_y_error", type="line")
            .add_metric(metrics_name="obs_lin_vel_y_error",
                        expr="avg(obs_lin_vel_y_error{})")
            .end_panel()
        .add_panel(name="实际前向速度", name_en="obs_actual_vel_x", type="line")
            .add_metric(metrics_name="obs_actual_vel_x",
                        expr="avg(obs_actual_vel_x{})")
            .end_panel()
        .add_panel(name="机身高度", name_en="obs_base_height", type="line")
            .add_metric(metrics_name="obs_base_height",
                        expr="avg(obs_base_height{})")
            .end_panel()
        .add_panel(name="pitch roll 角速度", name_en="obs_ang_vel_xy", type="line")
            .add_metric(metrics_name="obs_ang_vel_xy",
                        expr="avg(obs_ang_vel_xy{})")
            .end_panel()
        .end_group()
        .build()
    )
    return config_dict
