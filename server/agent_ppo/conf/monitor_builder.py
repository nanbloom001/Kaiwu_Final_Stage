#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Track-first monitor panels for the active hier-nav training stage."""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder

from agent_ppo.feature.nav_contract import TOKEN_NAMES


TRACK_PANEL_SPECS = (
    ("赛道完成数", "track_completed", "completed_count_track_l", "sum"),
    ("赛道失败数", "track_abnormal", "abnormal_count_track_l", "sum"),
    ("赛道超时数", "track_timeout", "timeout_count_track_l", "sum"),
    ("赛道总分", "track_total_score", "total_score_track_l", "avg"),
    ("赛道能耗分", "track_energy_score", "energy_score_track_l", "avg"),
    ("赛道姿态分", "track_pose_score", "pose_score_track_l", "avg"),
    # EnvMonitor exports the scorer's internal step score under the documented
    # public ``time_score_*`` metric family.
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
    ("非超时终止率", "nav_non_timeout_termination", "non_timeout_termination_rate"),
    ("超时率", "nav_timeout", "timeout_rate"),
    ("DAgger 学生比例", "nav_ramp", "ramp_probability"),
    ("梯度范数", "nav_grad_norm", "grad_norm"),
    ("平台生命周期成功回调", "nav_lifecycle_callbacks", "platform_lifecycle_callbacks"),
    ("平台生命周期失败回调", "nav_lifecycle_failures", "platform_lifecycle_failures"),
    ("累计低层批量步", "nav_low_level_steps", "total_low_level_steps"),
    ("累计环境帧", "nav_total_env_frames", "total_env_frames"),
    ("距下次平台发布回调数", "nav_callbacks_until_dump", "callbacks_until_next_dump"),
    ("学生驱动比例", "nav_student_drive", "student_drive_ratio"),
    ("Oracle 驱动比例", "nav_oracle_drive", "oracle_drive_ratio"),
    ("requested/effective 不一致", "nav_requested_effective_mismatch", "requested_effective_mismatch_ratio"),
    ("worker/exec 指令误差", "nav_worker_exec_command_error", "worker_exec_cmd_linf_mean"),
    ("worker/exec 指令匹配率", "nav_worker_exec_command_match", "worker_exec_cmd_match_ratio"),
    ("实际线速度", "nav_actual_vx", "actual_lin_vel_x_mean"),
    ("exec 线速度误差", "nav_exec_vx_error", "exec_vx_tracking_error_mean"),
    ("worker 线速度误差", "nav_worker_vx_error", "worker_vx_tracking_error_mean"),
    ("exec 更接近实际速度比例", "nav_exec_vx_closer", "exec_vx_closer_ratio"),
    ("低层 action 幅度", "nav_action_abs", "low_level_action_abs_mean"),
    ("低层 action 最大幅度", "nav_action_abs_max", "low_level_action_abs_max"),
    ("低层 action 变化", "nav_action_delta", "low_level_action_delta_abs_mean"),
    ("低层 action 非有限数", "nav_action_nonfinite", "low_level_action_nonfinite_count"),
    ("切换响应样本数", "nav_switch_response_samples", "switch_response_sample_count"),
    ("vx 切换响应样本数", "nav_switch_vx_response_samples", "switch_response_vx_sample_count"),
    ("切换后 action 变化", "nav_switch_action_delta", "switch_response_action_delta_abs_mean"),
    ("切换后真实速度变化", "nav_switch_vx_delta", "switch_response_vx_delta_abs_mean"),
    ("切换后 action 无响应比例", "nav_switch_no_action", "switch_response_no_action_ratio"),
    ("切换后速度无响应比例", "nav_switch_no_velocity", "switch_response_no_velocity_ratio"),
    ("驻留 tick 均值", "nav_dwell_mean", "scheduler_dwell_ticks_mean"),
    ("驻留 tick 最大值", "nav_dwell_max", "scheduler_dwell_ticks_max"),
    ("Oracle 目标距离", "nav_oracle_goal_dist", "oracle_goal_dist_mean"),
    ("Oracle 目标方位", "nav_oracle_goal_angle", "oracle_goal_angle_abs_mean"),
    ("Oracle 前方阻塞比例", "nav_oracle_front_blocked", "oracle_front_blocked_ratio"),
    ("Oracle 前方分数", "nav_oracle_front_score", "oracle_front_score_mean"),
    ("Oracle 左侧分数", "nav_oracle_left_score", "oracle_left_score_mean"),
    ("Oracle 右侧分数", "nav_oracle_right_score", "oracle_right_score_mean"),
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
    "arrived", "spin", "creep", "veer", "wall_avoid",
    "forward_slow", "forward_mid", "forward_fast",
)


def _add_track_panel(monitor, name, name_en, metric_prefix, aggregation):
    monitor.add_panel(name=name, name_en=name_en, type="line")
    for level in range(10):
        metric = f"{metric_prefix}{level}"
        monitor.add_metric(
            metrics_name=metric,
            expr=f"{aggregation}({metric}{{}})",
        )
    monitor.end_panel()


def _add_line_panel(monitor, name, name_en, metric):
    (
        monitor.add_panel(name=name, name_en=name_en, type="line")
        .add_metric(metrics_name=metric, expr=f"avg({metric}{{}})")
        .end_panel()
    )


def _add_multi_line_panel(monitor, name, name_en, metrics):
    monitor.add_panel(name=name, name_en=name_en, type="line")
    for metric in metrics:
        monitor.add_metric(metrics_name=metric, expr=f"avg({metric}{{}})")
    monitor.end_panel()


def build_monitor():
    """Build an explicit Track dashboard independent of frontend fallback rules."""

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
        "requested Token 分布",
        "nav_requested_token_distribution",
        [f"requested_token_{name}_ratio" for name in TOKEN_NAMES],
    )
    _add_multi_line_panel(
        monitor,
        "实际执行 Token 分布",
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
