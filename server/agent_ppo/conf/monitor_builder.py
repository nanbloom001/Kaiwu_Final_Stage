#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Track-first monitor panels for the active hier-nav training stage."""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


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
    ("异常终止率", "nav_hard_termination", "hard_termination_rate"),
    ("超时率", "nav_timeout", "timeout_rate"),
    ("DAgger 学生比例", "nav_ramp", "ramp_probability"),
    ("梯度范数", "nav_grad_norm", "grad_norm"),
    ("平台生命周期成功回调", "nav_lifecycle_callbacks", "platform_lifecycle_callbacks"),
    ("平台生命周期失败回调", "nav_lifecycle_failures", "platform_lifecycle_failures"),
    ("累计低层批量步", "nav_low_level_steps", "total_low_level_steps"),
    ("累计环境帧", "nav_total_env_frames", "total_env_frames"),
    ("距下次平台发布回调数", "nav_callbacks_until_dump", "callbacks_until_next_dump"),
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
    monitor.end_group()

    return monitor.build()
