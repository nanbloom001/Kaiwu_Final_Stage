#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Stage-aware dashboard for P1.5 while preserving the Nav monitor."""

from pathlib import Path

import toml
from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


P15_PANEL_SPECS = (
    ("策略损失", "p15_policy_loss", "policy_loss"),
    ("价值损失", "p15_value_loss", "value_loss"),
    ("动作锚点损失", "p15_action_anchor", "action_anchor_loss"),
    ("潜变量锚点损失", "p15_latent_anchor", "latent_anchor_loss"),
    ("响应器总损失", "p15_adapter_loss", "adapter_loss"),
    ("速度预测损失", "p15_adapter_velocity", "adapter_velocity_loss"),
    ("位姿预测损失", "p15_adapter_pose", "adapter_pose_loss"),
    ("卡滞预测损失", "p15_adapter_stuck", "adapter_stuck_loss"),
    ("速度预测误差", "p15_adapter_mae", "adapter_velocity_mae"),
    ("零值基线误差", "p15_zero_baseline", "adapter_zero_baseline_mae"),
    ("指令基线误差", "p15_copy_baseline", "adapter_copy_exec_baseline_mae"),
    ("相对零值提升", "p15_gain_zero", "adapter_gain_vs_zero"),
    ("相对指令提升", "p15_gain_copy", "adapter_gain_vs_copy_exec"),
    ("响应器更新数", "p15_adapter_updates", "adapter_applied_updates"),
    ("低层冻结状态", "p15_low_frozen", "low_level_frozen"),
    ("低层梯度步数", "p15_low_steps", "low_level_gradient_steps"),
    ("累计环境样本", "p15_env_steps", "total_env_steps"),
    ("响应器跳过数", "p15_nonfinite", "adapter_skipped_nonfinite"),
    ("低层跳过数", "p15_low_nonfinite", "low_level_skipped_nonfinite"),
    ("硬终止率", "p15_hard_term", "hard_termination_rate"),
    ("锚点动作误差", "p15_anchor_mse", "anchor_action_mse"),
    ("目标执行误差", "p15_target_exec", "target_exec_error"),
    ("执行实际误差", "p15_exec_actual", "exec_actual_error"),
    ("反馈有效率", "p15_feedback_valid", "response_valid_rate"),
    ("联合域覆盖率", "p15_coverage", "command_coverage_ratio"),
    ("指令课程阶段", "p15_phase", "command_phase"),
)

TRACK_PANEL_SPECS = (
    ("赛道完成数", "track_completed", "completed_count_track_l", "sum"),
    ("赛道失败数", "track_abnormal", "abnormal_count_track_l", "sum"),
    ("赛道超时数", "track_timeout", "timeout_count_track_l", "sum"),
    ("赛道总分", "track_total_score", "total_score_track_l", "avg"),
    ("赛道能耗分", "track_energy_score", "energy_score_track_l", "avg"),
    ("赛道姿态分", "track_pose_score", "pose_score_track_l", "avg"),
    ("赛道时间分", "track_time_score", "time_score_track_l", "avg"),
)

NAV_PANEL_SPECS = (
    ("交叉熵", "nav_ce_loss", "ce_loss"),
    ("Oracle 模仿准确率", "nav_top1_accuracy", "top1_accuracy"),
    ("学生与 Oracle 分歧率", "nav_disagreement", "disagreement_rate"),
    ("Token 熵", "nav_token_entropy", "token_entropy"),
    ("Token 切换率", "nav_switch_rate", "switch_rate"),
    ("目标有效率", "nav_goal_valid", "goal_valid_rate"),
    ("目标新鲜率", "nav_goal_fresh", "goal4_fresh_rate"),
    ("目标前进进度", "nav_goal_progress", "goal_progress_m_per_frame"),
    ("非超时终止率", "nav_non_timeout", "non_timeout_termination_rate"),
    ("超时率", "nav_timeout", "timeout_rate"),
    ("DAgger 学生比例", "nav_ramp", "ramp_probability"),
    ("梯度范数", "nav_grad_norm", "grad_norm"),
    ("生命周期成功回调", "nav_lifecycle_callbacks", "platform_lifecycle_callbacks"),
    ("生命周期失败回调", "nav_lifecycle_failures", "platform_lifecycle_failures"),
    ("累计低层批量步", "nav_low_level_steps", "total_low_level_steps"),
    ("累计环境帧", "nav_total_env_frames", "total_env_frames"),
    ("距下次发布回调数", "nav_until_dump", "callbacks_until_next_dump"),
    ("学生驱动比例", "nav_student_drive", "student_drive_ratio"),
    ("Oracle 驱动比例", "nav_oracle_drive", "oracle_drive_ratio"),
    ("请求与执行不一致", "nav_requested_effective", "requested_effective_mismatch_ratio"),
    ("指令执行误差", "nav_worker_exec_error", "worker_exec_cmd_linf_mean"),
    ("指令执行匹配率", "nav_worker_exec_match", "worker_exec_cmd_match_ratio"),
    ("实际线速度", "nav_actual_vx", "actual_lin_vel_x_mean"),
    ("执行线速度误差", "nav_exec_vx_error", "exec_vx_tracking_error_mean"),
    ("原始线速度误差", "nav_worker_vx_error", "worker_vx_tracking_error_mean"),
    ("执行更接近实际比例", "nav_exec_vx_closer", "exec_vx_closer_ratio"),
    ("低层动作幅度", "nav_action_abs", "low_level_action_abs_mean"),
    ("低层动作最大幅度", "nav_action_abs_max", "low_level_action_abs_max"),
    ("低层动作变化", "nav_action_delta", "low_level_action_delta_abs_mean"),
    ("低层动作非有限数", "nav_action_nonfinite", "low_level_action_nonfinite_count"),
    ("切换响应样本数", "nav_switch_samples", "switch_response_sample_count"),
    ("速度切换样本数", "nav_switch_vx_samples", "switch_response_vx_sample_count"),
    ("切换后动作变化", "nav_switch_action", "switch_response_action_delta_abs_mean"),
    ("切换后速度变化", "nav_switch_vx", "switch_response_vx_delta_abs_mean"),
    ("切换后动作无响应率", "nav_switch_no_action", "switch_response_no_action_ratio"),
    ("切换后速度无响应率", "nav_switch_no_velocity", "switch_response_no_velocity_ratio"),
    ("驻留 tick 均值", "nav_dwell_mean", "scheduler_dwell_ticks_mean"),
    ("驻留 tick 最大值", "nav_dwell_max", "scheduler_dwell_ticks_max"),
    ("Oracle 目标距离", "nav_oracle_goal_dist", "oracle_goal_dist_mean"),
    ("Oracle 目标方位", "nav_oracle_goal_angle", "oracle_goal_angle_abs_mean"),
    ("Oracle 前方阻塞率", "nav_oracle_blocked", "oracle_front_blocked_ratio"),
    ("Oracle 前方分数", "nav_oracle_front", "oracle_front_score_mean"),
    ("Oracle 左侧分数", "nav_oracle_left", "oracle_left_score_mean"),
    ("Oracle 右侧分数", "nav_oracle_right", "oracle_right_score_mean"),
    ("worker vx", "nav_worker_cmd_vx", "worker_cmd_vx_mean"),
    ("worker vy", "nav_worker_cmd_vy", "worker_cmd_vy_mean"),
    ("worker wz", "nav_worker_cmd_wz", "worker_cmd_wz_mean"),
    ("held vx", "nav_held_cmd_vx", "held_cmd_vx_mean"),
    ("held vy", "nav_held_cmd_vy", "held_cmd_vy_mean"),
    ("held wz", "nav_held_cmd_wz", "held_cmd_wz_mean"),
    ("exec vx", "nav_exec_cmd_vx", "exec_cmd_vx_mean"),
    ("exec vy", "nav_exec_cmd_vy", "exec_cmd_vy_mean"),
    ("exec wz", "nav_exec_cmd_wz", "exec_cmd_wz_mean"),
)

ORACLE_MODE_NAMES = (
    "arrived",
    "spin",
    "creep",
    "veer",
    "wall_avoid",
    "forward_slow",
    "forward_mid",
    "forward_fast",
)


def _configured_policy_entry() -> str:
    path = Path(__file__).resolve().parents[2] / "conf" / "configure_app.toml"
    try:
        config = toml.load(path)
    except (OSError, TypeError, ValueError):
        return "nav_dagger"
    return str(config.get("app", {}).get("policy_entry", "nav_dagger")).strip().lower()


def _add_line_panel(monitor, name, name_en, metric):
    (
        monitor.add_panel(name=name, name_en=name_en, type="line")
        .add_metric(metrics_name=metric, expr=f"avg({metric}{{}})")
        .end_panel()
    )


def _add_track_panel(monitor, name, name_en, metric_prefix, aggregation):
    monitor.add_panel(name=name, name_en=name_en, type="line")
    for level in range(10):
        metric = f"{metric_prefix}{level}"
        monitor.add_metric(metrics_name=metric, expr=f"{aggregation}({metric}{{}})")
    monitor.end_panel()


def _add_multi_line_panel(monitor, name, name_en, metrics):
    monitor.add_panel(name=name, name_en=name_en, type="line")
    for metric in metrics:
        monitor.add_metric(metrics_name=metric, expr=f"avg({metric}{{}})")
    monitor.end_panel()


def _build_p15_monitor():
    monitor = MonitorConfigBuilder()
    monitor.title("P15动态响应训练")
    monitor.add_group(group_name="联合训练状态", group_name_en="p15_training")
    for panel in P15_PANEL_SPECS:
        _add_line_panel(monitor, *panel)
    monitor.end_group()
    return monitor.build()


def _build_nav_monitor():
    from agent_ppo.feature.nav_contract import TOKEN_NAMES

    monitor = MonitorConfigBuilder()
    monitor.title("Track 高层导航训练")
    monitor.add_group(group_name="Track 赛道结果", group_name_en="track_outcomes")
    for panel in TRACK_PANEL_SPECS:
        _add_track_panel(monitor, *panel)
    monitor.end_group()
    monitor.add_group(group_name="Nav DAgger", group_name_en="nav_dagger")
    for panel in NAV_PANEL_SPECS:
        _add_line_panel(monitor, *panel)
    _add_multi_line_panel(
        monitor,
        "学生 Token 分布",
        "nav_student_token_distribution",
        [f"student_token_{name}_ratio" for name in TOKEN_NAMES],
    )
    _add_multi_line_panel(
        monitor,
        "Oracle Token 分布",
        "nav_oracle_token_distribution",
        [f"oracle_token_{name}_ratio" for name in TOKEN_NAMES],
    )
    _add_multi_line_panel(
        monitor,
        "请求 Token 分布",
        "nav_requested_token_distribution",
        [f"requested_token_{name}_ratio" for name in TOKEN_NAMES],
    )
    _add_multi_line_panel(
        monitor,
        "执行 Token 分布",
        "nav_effective_token_distribution",
        [f"effective_token_{name}_ratio" for name in TOKEN_NAMES],
    )
    _add_multi_line_panel(
        monitor,
        "Oracle 规则分支",
        "nav_oracle_rule_modes",
        [f"oracle_mode_{name}_ratio" for name in ORACLE_MODE_NAMES],
    )
    monitor.end_group()
    return monitor.build()


def build_monitor():
    if _configured_policy_entry() == "p15_response":
        return _build_p15_monitor()
    return _build_nav_monitor()
