#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Monitor configuration for the current track-navigation training line."""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


DISPLAY_NAME_LIMIT = 20

DISPLAY_NAMES = {
    "track_navigation_training": "track_nav",
    "direct_score_and_completion": "score_complete",
    "time_and_progress_proxy": "time_progress",
    "posture_energy_support": "pose_energy",
    "gait_and_stillness": "gait_still",
    "maze_safety_and_finish": "maze_safety",
    "suggested_speed_gate_probes": "speed_gate_probe",
    "suggested_speed_gate_summary": "speed_gate",
    "velocity_curriculum": "vel_curriculum",
    "training_progress": "train_progress",
    "difficulty_pressure_complete": "diff_pressure",
    "goal_velocity_projection": "goal_vel_proj",
    "forward_heading_velocity": "heading_vel",
    "command_speed_advantage": "cmd_speed_adv",
    "joint_position_penalty": "joint_pos_penalty",
    "action_smoothness": "action_smooth",
    "air_time_variance_penalty": "air_time_var",
    "commanded_still_penalty": "cmd_still_penalty",
    "goal_heading_alignment": "goal_heading",
    "near_goal_finish_drive": "near_finish_drive",
    "near_goal_retreat_penalty": "near_retreat",
    "near_goal_circling_penalty": "near_circling",
    "maze_anticipatory_turn": "maze_turn",
    "directed_exploration": "direct_explore",
    "long_non_foot_contact": "non_foot_contact",
    "speed_gate_target_vx": "gate_target_vx",
    "speed_gate_worker_vx": "gate_worker_vx",
    "speed_gate_nav_front": "gate_nav_front",
    "speed_gate_nav_block": "gate_nav_block",
    "speed_gate_hold_steps": "gate_hold_steps",
    "speed_gate_maze_confirm": "gate_maze_confirm",
    "vel_curriculum_tracking_ratio": "vel_track_ratio",
    "vel_curriculum_tracking_reward": "vel_track_reward",
    "level_abnormal_pct": "level_abn_pct",
    "level_timeout_pct": "level_timeout_pct",
    "command_vx_and_write_pct": "cmd_vx_write",
    "wall_gate_signal": "wall_gate",
    "sticky_gate_state": "sticky_gate",
    "velocity_error_stat": "vel_error_stat",
    "obs_lin_vel_x_error": "obs_vx_error",
    "obs_lin_vel_y_error": "obs_vy_error",
}


def _display_name(name: str) -> str:
    display = DISPLAY_NAMES.get(name, name)
    return display[:DISPLAY_NAME_LIMIT]


def _avg(metric_name: str) -> str:
    return f"avg({metric_name}{{}})"


def _reward(reward_name: str) -> str:
    return f"reward_{reward_name}"


def _track_rate(outcome_name: str, level: int | None = None) -> str:
    if level is None:
        done = "completed_count"
        abnormal = "abnormal_count"
        timeout = "timeout_count"
    else:
        suffix = f"_track_l{level}"
        done = f"completed_count{suffix}"
        abnormal = f"abnormal_count{suffix}"
        timeout = f"timeout_count{suffix}"
    return (
        f"100 * avg({outcome_name}{{}}) / "
        f"clamp_min(avg({done}{{}}) + avg({abnormal}{{}}) + avg({timeout}{{}}), 1)"
    )


def _add_panel(builder, panel_name: str, metric_name: str, expr: str | None = None, panel_type: str = "line"):
    builder.add_panel(name=_display_name(panel_name), name_en=panel_name, type=panel_type)
    builder.add_metric(metrics_name=metric_name, expr=expr or _avg(metric_name))
    builder.end_panel()


def _add_multi_panel(builder, panel_name: str, metrics: tuple[tuple[str, str], ...], panel_type: str = "line"):
    builder.add_panel(name=_display_name(panel_name), name_en=panel_name, type=panel_type)
    for metric_name, expr in metrics:
        builder.add_metric(metrics_name=metric_name, expr=expr)
    builder.end_panel()


def _add_group(builder, group_name: str, panels: tuple[tuple, ...]):
    builder.add_group(group_name=_display_name(group_name), group_name_en=group_name)
    for panel in panels:
        if len(panel) == 2:
            panel_name, metric_name = panel
            _add_panel(builder, panel_name, metric_name)
        elif len(panel) == 3 and isinstance(panel[1], tuple):
            panel_name, metrics, panel_type = panel
            _add_multi_panel(builder, panel_name, metrics, panel_type)
        else:
            panel_name, metric_name, expr = panel
            _add_panel(builder, panel_name, metric_name, expr)
    builder.end_group()


def _level_outcome_metrics(kind: str) -> tuple[tuple[str, str], ...]:
    source = "abnormal_count" if kind == "abnormal" else "timeout_count"
    suffix = "abn" if kind == "abnormal" else "timeout"
    return tuple(
        (
            f"l{level}_{suffix}",
            _track_rate(f"{source}_track_l{level}", level),
        )
        for level in range(10)
    )


REWARD_GROUPS = (
    (
        "direct_score_and_completion",
        (
            ("task_complete", _reward("task_complete")),
            ("difficulty_pressure_complete", _reward("difficulty_pressure_complete")),
            ("pose_score_formula", _reward("pose_score_formula")),
            ("energy_score_formula", _reward("energy_score_formula")),
            ("termination", _reward("termination")),
            ("undesired_contacts", _reward("undesired_contacts")),
        ),
    ),
    (
        "time_and_progress_proxy",
        (
            ("goal_velocity_projection", _reward("goal_velocity_projection")),
            ("forward_heading_velocity", _reward("forward_heading_velocity")),
            ("track_lin_vel_xy", _reward("track_lin_vel_xy")),
            ("command_speed_advantage", _reward("command_speed_advantage")),
            ("approach_goal", _reward("approach_goal")),
            ("goal_distance", _reward("goal_distance")),
            ("navigation_time", _reward("navigation_time")),
            ("backward_penalty", _reward("backward_penalty")),
            ("goal_backtrack_penalty", _reward("goal_backtrack_penalty")),
        ),
    ),
    (
        "posture_energy_support",
        (
            ("flat_orientation", _reward("flat_orientation")),
            ("correct_base_height", _reward("correct_base_height")),
            ("posture_stability", _reward("posture_stability")),
            ("ang_vel_xy", _reward("ang_vel_xy")),
            ("base_lateral_vel", _reward("base_lateral_vel")),
            ("lin_vel_z", _reward("lin_vel_z")),
            ("hip_to_default", _reward("hip_to_default")),
            ("joint_position_penalty", _reward("joint_position_penalty")),
            ("dof_pos_limits", _reward("dof_pos_limits")),
            ("energy", _reward("energy")),
            ("joint_torques", _reward("joint_torques")),
            ("joint_acc", _reward("joint_acc")),
            ("dof_vel", _reward("dof_vel")),
            ("action_rate", _reward("action_rate")),
            ("action_smoothness", _reward("action_smoothness")),
            ("score_guidance", _reward("score_guidance")),
        ),
    ),
    (
        "gait_and_stillness",
        (
            ("track_ang_vel_z", _reward("track_ang_vel_z")),
            ("feet_air_time", _reward("feet_air_time")),
            ("feet_clearance", _reward("feet_clearance")),
            ("feet_swing_forward", _reward("feet_swing_forward")),
            ("feet_slide", _reward("feet_slide")),
            ("feet_stumble", _reward("feet_stumble")),
            ("air_time_variance_penalty", _reward("air_time_variance_penalty")),
            ("stand_still_motion", _reward("stand_still_motion")),
            ("commanded_still_penalty", _reward("commanded_still_penalty")),
        ),
    ),
    (
        "maze_safety_and_finish",
        (
            ("goal_heading_alignment", _reward("goal_heading_alignment")),
            ("near_goal_finish_drive", _reward("near_goal_finish_drive")),
            ("near_goal_retreat_penalty", _reward("near_goal_retreat_penalty")),
            ("near_goal_circling_penalty", _reward("near_goal_circling_penalty")),
            ("goal_miss_penalty", _reward("goal_miss_penalty")),
            ("maze_context_gate", _reward("maze_context_gate")),
            ("maze_anticipatory_turn", _reward("maze_anticipatory_turn")),
            ("wall_collision", _reward("wall_collision")),
            ("wall_stall_penalty", _reward("wall_stall_penalty")),
            ("wall_proximity", _reward("wall_proximity")),
            ("open_space", _reward("open_space")),
            ("corridor_centering", _reward("corridor_centering")),
            ("directed_exploration", _reward("directed_exploration")),
            ("stuck_penalty", _reward("stuck_penalty")),
            ("long_non_foot_contact", _reward("long_non_foot_contact")),
        ),
    ),
    (
        "suggested_speed_gate_probes",
        (
            ("speed_gate_flat", _reward("speed_gate_flat")),
            ("speed_gate_slope", _reward("speed_gate_slope")),
            ("speed_gate_stairs", _reward("speed_gate_stairs")),
            ("speed_gate_maze", _reward("speed_gate_maze")),
            ("speed_gate_invalid", _reward("speed_gate_invalid")),
            ("speed_gate_sum", _reward("speed_gate_sum")),
            ("speed_gate_valid", _reward("speed_gate_valid")),
            ("speed_gate_target_vx", _reward("speed_gate_target_vx")),
            ("speed_gate_worker_vx", _reward("speed_gate_worker_vx")),
            ("speed_gate_written", _reward("speed_gate_written")),
            ("speed_gate_nav_front", _reward("speed_gate_nav_front")),
            ("speed_gate_nav_block", _reward("speed_gate_nav_block")),
            ("speed_gate_hold_steps", _reward("speed_gate_hold_steps")),
            ("speed_gate_pending", _reward("speed_gate_pending")),
            ("speed_gate_maze_confirm", _reward("speed_gate_maze_confirm")),
        ),
    ),
)


def _add_reward_groups(builder):
    for group_name, rewards in REWARD_GROUPS:
        panels = tuple((panel_name, metric_name) for panel_name, metric_name in rewards)
        _add_group(builder, group_name, panels)


def _add_track_results(builder):
    _add_group(
        builder,
        "track_results",
        (
            (
                "outcome_pct",
                (
                    ("completion_pct", _track_rate("completed_count")),
                    ("abnormal_pct", _track_rate("abnormal_count")),
                    ("timeout_pct", _track_rate("timeout_count")),
                ),
                "line",
            ),
            ("level_abnormal_pct", _level_outcome_metrics("abnormal"), "line"),
            ("level_timeout_pct", _level_outcome_metrics("timeout"), "line"),
        ),
    )


def _add_gate_summary(builder):
    denom = "clamp_min(avg(reward_speed_gate_sum{}), 1e-12)"
    _add_group(
        builder,
        "suggested_speed_gate_summary",
        (
            (
                "command_vx_and_write_pct",
                (
                    ("target_vx", "avg(reward_speed_gate_target_vx{}) * 1000000000"),
                    ("worker_vx", "avg(reward_speed_gate_worker_vx{}) * 1000000000"),
                    ("write_pct", "avg(reward_speed_gate_written{}) * 100000000000"),
                ),
                "line",
            ),
            (
                "terrain_pct",
                (
                    ("flat_pct", f"100 * avg(reward_speed_gate_flat{{}}) / {denom}"),
                    ("slope_pct", f"100 * avg(reward_speed_gate_slope{{}}) / {denom}"),
                    ("stairs_pct", f"100 * avg(reward_speed_gate_stairs{{}}) / {denom}"),
                    ("maze_pct", f"100 * avg(reward_speed_gate_maze{{}}) / {denom}"),
                    ("invalid_pct", f"100 * avg(reward_speed_gate_invalid{{}}) / {denom}"),
                ),
                "line",
            ),
            (
                "gate_status",
                (
                    ("valid_pct", f"100 * avg(reward_speed_gate_valid{{}}) / {denom}"),
                    ("terrain_sum", "avg(reward_speed_gate_sum{}) * 1000000000"),
                ),
                "line",
            ),
            (
                "wall_gate_signal",
                (
                    ("front_wall", "avg(reward_speed_gate_nav_front{}) * 1000000000"),
                    ("front_block_pct", "avg(reward_speed_gate_nav_block{}) * 100000000000"),
                ),
                "line",
            ),
            (
                "sticky_gate_state",
                (
                    ("hold_steps", "avg(reward_speed_gate_hold_steps{}) * 1000000000"),
                    ("pending", "avg(reward_speed_gate_pending{}) * 1000000000"),
                    ("maze_confirm", "avg(reward_speed_gate_maze_confirm{}) * 1000000000"),
                ),
                "line",
            ),
        ),
    )


def build_monitor():
    """Build monitor panels without tying dashboard structure to old experiments."""
    monitor = MonitorConfigBuilder()
    monitor.title(_display_name("track_navigation_training"))

    _add_group(
        monitor,
        "training_progress",
        (
            ("mean_episode_length", "mean_episode_length"),
            ("mean_episode_reward", "mean_episode_reward"),
            ("episode_reward", "episode_reward"),
        ),
    )
    _add_group(
        monitor,
        "algorithm",
        (
            ("total_loss", "total_loss"),
            ("value_loss", "value_loss"),
            ("policy_loss", "policy_loss"),
            ("entropy_loss", "entropy_loss"),
        ),
    )
    _add_group(
        monitor,
        "velocity_curriculum",
        (
            ("vel_curriculum_stage", "vel_curriculum_stage"),
            ("vel_curriculum_tracking_ratio", "vel_curriculum_tracking_ratio"),
            ("vel_curriculum_tracking_reward", "vel_curriculum_tracking_reward"),
        ),
    )

    _add_track_results(monitor)
    _add_reward_groups(monitor)
    _add_gate_summary(monitor)

    _add_group(
        monitor,
        "physics_obs",
        (
            (
                "velocity_x_stat",
                (
                    ("cmd_vx", _avg("obs_cmd_vel_x")),
                    ("actual_vx", _avg("obs_actual_vel_x")),
                ),
                "stat",
            ),
            (
                "velocity_y_stat",
                (
                    ("cmd_vy", _avg("obs_cmd_vel_y")),
                    ("actual_vy", _avg("obs_actual_vel_y")),
                ),
                "stat",
            ),
            (
                "velocity_yaw_stat",
                (
                    ("cmd_yaw", _avg("obs_cmd_yaw")),
                    ("actual_yaw", _avg("obs_actual_yaw")),
                ),
                "stat",
            ),
            (
                "velocity_error_stat",
                (
                    ("vx_error", _avg("obs_lin_vel_x_error")),
                    ("yaw_error", _avg("obs_yaw_error")),
                ),
                "stat",
            ),
            ("obs_lin_vel_x_error", "obs_lin_vel_x_error"),
            ("obs_lin_vel_y_error", "obs_lin_vel_y_error"),
            ("obs_actual_vel_x", "obs_actual_vel_x"),
            ("obs_base_height", "obs_base_height"),
            ("obs_ang_vel_xy", "obs_ang_vel_xy"),
        ),
    )

    return monitor.build()
