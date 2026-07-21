#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Compact R1 Standard PPO monitoring panels."""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _add_line_panel(monitor, display_name, panel_name, metric_name):
    return (
        monitor.add_panel(name=display_name, name_en=panel_name, type="line")
        .add_metric(metrics_name=metric_name, expr=f"avg({metric_name}{{}})")
        .end_panel()
    )


def _build_lbc_monitor(title="LBC_Loco"):
    monitor = MonitorConfigBuilder().title(title)
    groups = (
        (
            "蒸馏",
            "distill",
            (
                ("加权总损失", "distill_total", "total_loss"),
                ("潜变量损失", "latent_loss", "latent_loss"),
                ("余弦损失", "cosine_loss", "cosine_loss"),
                ("余弦相似度", "cosine_sim", "cosine_similarity"),
                ("夹角", "angle_deg", "angle_deg"),
                ("梯度范数", "grad_norm", "grad_norm"),
                ("学习率", "learning_rate", "learning_rate"),
                ("动作损失", "action_loss", "action_loss"),
                ("学生驱动率", "student_drive", "student_drive_ratio"),
            ),
        ),
        (
            "Goal噪声",
            "goal_noise",
            (
                ("启用率", "goal_active", "goal_noise_active_ratio"),
                ("方向噪声", "goal_bear_noise", "goal_bearing_noise_abs_mean"),
                ("距离噪声", "goal_dist_noise", "goal_distance_noise_abs_mean"),
                ("方向偏差", "goal_bear_bias", "goal_bearing_bias_abs_mean"),
                ("距离偏差", "goal_dist_bias", "goal_distance_bias_abs_mean"),
            ),
        ),
        (
            "深度增强",
            "depth_aug",
            (
                ("散点空洞", "pixel_dropout", "depth_random_dropout_ratio"),
                ("块帧占比", "block_frames", "depth_block_dropout_frame_ratio"),
                ("块面积占比", "block_area", "depth_block_dropout_area_ratio"),
                ("活动块数", "active_blocks", "active_block_count"),
                ("块持续帧", "block_persist", "block_persistence_mean"),
                ("增强前有效率", "valid_before", "valid_depth_ratio_before"),
                ("增强后有效率", "valid_after", "valid_depth_ratio_after"),
            ),
        ),
    )
    for group_name, group_name_en, panels in groups:
        monitor.add_group(group_name=group_name, group_name_en=group_name_en)
        for display_name, panel_name, metric_name in panels:
            _add_line_panel(monitor, display_name, panel_name, metric_name)
        monitor.end_group()
    return monitor.build()


def _build_behavior_distill_monitor():
    monitor = MonitorConfigBuilder().title("Standard_Distill_1")
    monitor.add_group(group_name="行为蒸馏", group_name_en="behavior_distill")
    panels = (
        ("动作均方差", "action_mse", "action_mse"),
        ("动作距离", "action_l2", "action_l2"),
        ("动作余弦", "action_cos", "action_cos"),
        ("教师动作幅值", "teacher_abs", "teacher_abs"),
        ("学生动作幅值", "student_abs", "student_abs"),
        ("梯度范数", "grad_norm", "grad_norm"),
        ("训练轮次", "iteration", "iteration"),
        ("累计步数", "total_steps", "total_steps"),
    )
    for display_name, panel_name, metric_name in panels:
        _add_line_panel(monitor, display_name, panel_name, metric_name)
    monitor.end_group()
    return monitor.build()


def build_monitor():
    """Build panels whose display and metric names stay within 20 characters."""
    from agent_ppo.conf.conf import Config

    algorithm = getattr(Config.CURRENT, "algorithm", "ppo")
    if algorithm == "lbc_loco":
        title = (
            "Standard_Distill_1"
            if Config.CURRENT.name == "standard_distill_1"
            else "ST9_Opt3_D2"
        )
        return _build_lbc_monitor(title)
    if algorithm == "behavior_distill":
        return _build_behavior_distill_monitor()

    monitor = MonitorConfigBuilder()
    return (
        monitor.title("Standard_PPO")
        .add_group(group_name="训练", group_name_en="train")
        .add_panel(name="回合步数", name_en="ep_steps", type="line")
            .add_metric(metrics_name="ep_steps", expr="avg(ep_steps{})")
            .end_panel()
        .add_panel(name="回合奖励", name_en="ep_reward", type="line")
            .add_metric(metrics_name="ep_reward", expr="avg(ep_reward{})")
            .end_panel()
        .add_panel(name="回合数", name_en="ep_count", type="line")
            .add_metric(metrics_name="ep_count", expr="max(ep_count{})")
            .end_panel()
        .end_group()

        .add_group(group_name="优化", group_name_en="ppo_opt")
        .add_panel(name="总损失", name_en="total_loss", type="line")
            .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
            .end_panel()
        .add_panel(name="策略损失", name_en="policy_loss", type="line")
            .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
            .end_panel()
        .add_panel(name="价值损失", name_en="value_loss", type="line")
            .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
            .end_panel()
        .add_panel(name="熵损失", name_en="entropy_loss", type="line")
            .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
            .end_panel()
        .add_panel(name="学习率", name_en="learning_rate", type="line")
            .add_metric(metrics_name="learning_rate", expr="avg(learning_rate{})")
            .end_panel()
        .end_group()

        .add_group(group_name="速度", group_name_en="speed")
        .add_panel(name="速度课程", name_en="vel_stage", type="line")
            .add_metric(metrics_name="vel_stage", expr="avg(vel_stage{})")
            .end_panel()
        .add_panel(name="速度追踪", name_en="vel_track", type="line")
            .add_metric(metrics_name="vel_track", expr="avg(vel_track{})")
            .end_panel()
        .add_panel(name="速度上限", name_en="vel_max", type="line")
            .add_metric(metrics_name="vel_max", expr="avg(vel_max{})")
            .end_panel()
        .add_panel(name="前向奖励", name_en="rew_vx", type="line")
            .add_metric(metrics_name="rew_vx", expr="avg(rew_vx{})")
            .end_panel()
        .add_panel(name="偏航奖励", name_en="rew_yaw", type="line")
            .add_metric(metrics_name="rew_yaw", expr="avg(rew_yaw{})")
            .end_panel()
        .add_panel(name="速度对照", name_en="vx_stat", type="stat")
            .add_metric(metrics_name="vx_cmd", expr="avg(vx_cmd{})")
            .add_metric(metrics_name="vx_real", expr="avg(vx_real{})")
            .end_panel()
        .add_panel(name="速度误差", name_en="vx_err", type="line")
            .add_metric(metrics_name="vx_err", expr="avg(vx_err{})")
            .end_panel()
        .end_group()

        .add_group(group_name="地形", group_name_en="terrain")
        .add_panel(name="地形难度", name_en="ter_level", type="line")
            .add_metric(metrics_name="ter_level", expr="avg(ter_level{})")
            .end_panel()
        .add_panel(name="地形范围", name_en="ter_range", type="stat")
            .add_metric(metrics_name="ter_min", expr="avg(ter_min{})")
            .add_metric(metrics_name="ter_max", expr="avg(ter_max{})")
            .end_panel()
        .add_panel(name="地形切换", name_en="ter_switch", type="stat")
            .add_metric(metrics_name="ter_up", expr="avg(ter_up{})")
            .add_metric(metrics_name="ter_down", expr="avg(ter_down{})")
            .end_panel()
        .end_group()

        .add_group(group_name="困难段回放", group_name_en="hard_replay")
        .add_panel(name="起点比例", name_en="start_ratio", type="stat")
            .add_metric(metrics_name="full_track_ratio", expr="avg(full_track_ratio{})")
            .add_metric(metrics_name="hard_start_ratio", expr="avg(hard_start_ratio{})")
            .end_panel()
        .add_panel(name="困难起点1", name_en="start_count_1", type="stat")
            .add_metric(metrics_name="stairs_down_starts", expr="max(stairs_down_starts{})")
            .add_metric(metrics_name="slope_down_starts", expr="max(slope_down_starts{})")
            .end_panel()
        .add_panel(name="困难起点2", name_en="start_count_2", type="stat")
            .add_metric(metrics_name="maze_entry_starts", expr="max(maze_entry_starts{})")
            .add_metric(metrics_name="stairs_up_starts", expr="max(stairs_up_starts{})")
            .end_panel()
        .add_panel(name="分段成功率", name_en="hard_success", type="line")
            .add_metric(metrics_name="stairs_down_success", expr="avg(stairs_down_success{})")
            .add_metric(metrics_name="slope_down_success", expr="avg(slope_down_success{})")
            .add_metric(metrics_name="maze_entry_success", expr="avg(maze_entry_success{})")
            .add_metric(metrics_name="stairs_up_success", expr="avg(stairs_up_success{})")
            .end_panel()
        .add_panel(name="姿态异常1", name_en="bad_segment_1", type="stat")
            .add_metric(metrics_name="bad_stairs_down", expr="max(bad_stairs_down{})")
            .add_metric(metrics_name="bad_slope_down", expr="max(bad_slope_down{})")
            .end_panel()
        .add_panel(name="姿态异常2", name_en="bad_segment_2", type="stat")
            .add_metric(metrics_name="bad_maze_entry", expr="max(bad_maze_entry{})")
            .add_metric(metrics_name="bad_stairs_up", expr="max(bad_stairs_up{})")
            .end_panel()
        .add_panel(name="机身接触1", name_en="base_segment_1", type="stat")
            .add_metric(metrics_name="base_stairs_down", expr="max(base_stairs_down{})")
            .add_metric(metrics_name="base_slope_down", expr="max(base_slope_down{})")
            .end_panel()
        .add_panel(name="机身接触2", name_en="base_segment_2", type="stat")
            .add_metric(metrics_name="base_maze_entry", expr="max(base_maze_entry{})")
            .add_metric(metrics_name="base_stairs_up", expr="max(base_stairs_up{})")
            .end_panel()
        .add_panel(name="超时分段1", name_en="timeout_segment_1", type="stat")
            .add_metric(metrics_name="timeout_stairs_down", expr="max(timeout_stairs_down{})")
            .add_metric(metrics_name="timeout_slope_down", expr="max(timeout_slope_down{})")
            .end_panel()
        .add_panel(name="超时分段2", name_en="timeout_segment_2", type="stat")
            .add_metric(metrics_name="timeout_maze_entry", expr="max(timeout_maze_entry{})")
            .add_metric(metrics_name="timeout_stairs_up", expr="max(timeout_stairs_up{})")
            .end_panel()
        .add_panel(name="早期异常1", name_en="early_failure_1", type="stat")
            .add_metric(metrics_name="early_bad_total", expr="max(early_bad_total{})")
            .add_metric(metrics_name="early_base_total", expr="max(early_base_total{})")
            .end_panel()
        .add_panel(name="早期终止", name_en="early_failure_2", type="stat")
            .add_metric(metrics_name="early_term_total", expr="max(early_term_total{})")
            .add_metric(metrics_name="surface_query_fail", expr="max(surface_query_fail{})")
            .end_panel()
        .add_panel(name="出生净空", name_en="spawn_clearance", type="line")
            .add_metric(metrics_name="spawn_clearance", expr="avg(spawn_clearance{})")
            .end_panel()
        .end_group()

        .add_group(group_name="稳定", group_name_en="stability")
        .add_panel(name="姿态奖励", name_en="rew_flat", type="line")
            .add_metric(metrics_name="rew_flat", expr="avg(rew_flat{})")
            .end_panel()
        .add_panel(name="接触惩罚", name_en="rew_contact", type="line")
            .add_metric(metrics_name="rew_contact", expr="avg(rew_contact{})")
            .end_panel()
        .add_panel(name="终止惩罚", name_en="rew_term", type="line")
            .add_metric(metrics_name="rew_term", expr="avg(rew_term{})")
            .end_panel()
        .add_panel(name="机身高度", name_en="body_h", type="line")
            .add_metric(metrics_name="body_h", expr="avg(body_h{})")
            .end_panel()
        .add_panel(name="倾斜速度", name_en="tilt_rate", type="line")
            .add_metric(metrics_name="tilt_rate", expr="avg(tilt_rate{})")
            .end_panel()
        .add_panel(name="动态倾斜风险", name_en="reward_dynamic_tilt_risk", type="line")
            .add_metric(
                metrics_name="reward_dynamic_tilt_risk",
                expr="avg(reward_dynamic_tilt_risk{})",
            )
            .end_panel()
        .end_group()

        .add_group(group_name="能耗动作", group_name_en="energy_act")
        .add_panel(name="能耗惩罚", name_en="rew_energy", type="line")
            .add_metric(metrics_name="rew_energy", expr="avg(rew_energy{})")
            .end_panel()
        .add_panel(name="扭矩惩罚", name_en="rew_torque", type="line")
            .add_metric(metrics_name="rew_torque", expr="avg(rew_torque{})")
            .end_panel()
        .add_panel(name="动作惩罚", name_en="rew_action", type="line")
            .add_metric(metrics_name="rew_action", expr="avg(rew_action{})")
            .end_panel()
        .build()
    )
