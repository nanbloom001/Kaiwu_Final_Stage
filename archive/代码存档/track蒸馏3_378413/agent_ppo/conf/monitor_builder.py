#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Custom monitor panels for Track Camera LBC training."""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _add_line_panel(monitor, name, metric):
    return (
        monitor.add_panel(name=name, name_en=metric, type="line")
        .add_metric(metrics_name=metric, expr=f"avg({metric}{{}})")
        .end_panel()
    )


def build_monitor():
    """Build monitor config using only Chinese, English and underscores in names."""
    monitor = MonitorConfigBuilder()

    monitor.title("legged_robot_nav")

    monitor.add_group(group_name="algorithm", group_name_en="algorithm")
    _add_line_panel(monitor, "total_loss", "total_loss")
    _add_line_panel(monitor, "value_loss", "value_loss")
    _add_line_panel(monitor, "policy_loss", "policy_loss")
    _add_line_panel(monitor, "entropy_loss", "entropy_loss")
    _add_line_panel(monitor, "iteration", "iteration")
    _add_line_panel(monitor, "total_steps", "total_steps")
    monitor.end_group()

    monitor.add_group(group_name="distill", group_name_en="distill")
    _add_line_panel(monitor, "angle", "angle")
    _add_line_panel(monitor, "cos_sim", "cos_sim")
    _add_line_panel(monitor, "mse_loss", "mse_loss")
    _add_line_panel(monitor, "distance", "distance")
    _add_line_panel(monitor, "student_std", "student_std")
    _add_line_panel(monitor, "teacher_std", "teacher_std")
    _add_line_panel(monitor, "grad_norm", "grad_norm")
    _add_line_panel(monitor, "student_drive_ratio", "student_drive_ratio")
    _add_line_panel(monitor, "action_loss", "action_loss")
    _add_line_panel(monitor, "p_student", "p_student")
    monitor.end_group()

    monitor.add_group(group_name="reward", group_name_en="reward")
    _add_line_panel(monitor, "reward_lin_vel_xy", "reward_track_lin_vel_xy")
    monitor.end_group()

    return monitor.build()

