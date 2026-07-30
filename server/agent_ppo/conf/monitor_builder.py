#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Stage-aware dashboards for P1.5 and P2 while preserving Nav panels."""

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

P2_MONITOR_GROUPS = (
    (
        "训练收敛",
        "p2_optimization",
        (
            ("损失趋势", "p2_losses", ("actor_loss", "critic_loss", "adapter_loss")),
            (
                "回报统计",
                "p2_returns",
                ("rollout_reward_mean", "rollout_reward_std", "rollout_return_mean"),
            ),
            (
                "价值优势统计",
                "p2_value_advantage",
                ("rollout_value_mean", "rollout_advantage_mean", "rollout_advantage_std"),
            ),
            (
                "动作探索标准差",
                "p2_action_std",
                ("action_std_vx", "action_std_vy", "action_std_wz"),
            ),
            (
                "优化器学习率",
                "p2_learning_rates",
                (
                    "navigation_learning_rate",
                    "actor_learning_rate",
                    "vy_actor_learning_rate",
                    "safety_head_learning_rate",
                    "critic_learning_rate",
                    "adapter_learning_rate",
                ),
            ),
            (
                "PPO逐轮质量",
                "p2_epoch_quality",
                (
                    "approx_kl", "clip_fraction", "entropy", "safety_bce",
                    "epoch_1_approx_kl", "epoch_2_approx_kl",
                    "epoch_3_approx_kl", "epoch_4_approx_kl",
                ),
            ),
            ("熵课程", "p2_entropy_schedule", ("entropy_coefficient",)),
            (
                "更新与跳过",
                "p2_update_health",
                (
                    "updates",
                    "adapter_updates",
                    "ppo_rollout_skipped_nonfinite",
                    "skipped_nonfinite_total",
                    "adapter_oom_skips",
                ),
            ),
            (
                "训练时钟",
                "p2_training_clocks",
                (
                    "session_effective_seconds",
                    "lifetime_effective_seconds",
                ),
            ),
            (
                "横向能力边界",
                "p2_vy_limits",
                ("vy_trusted_limit", "vy_hard_limit"),
            ),
            ("网络训练状态", "p2_network_state", ("cnn_unfrozen", "microbatch_frames")),
            (
                "生命周期回调",
                "p2_lifecycle",
                ("lifecycle_success", "lifecycle_failures"),
            ),
        ),
    ),
    (
        "奖励贡献",
        "p2_rewards",
        (
            (
                "终点势函数",
                "p2_progress_rewards",
                (
                    "reward_frontier_shaping",
                    "frontier_potential_before",
                    "frontier_potential_after",
                    "terminal_potential_clawback",
                ),
            ),
            (
                "终止事件",
                "p2_terminal_rewards",
                ("reward_success", "reward_failure", "reward_timeout"),
            ),
            (
                "持续代价",
                "p2_tick_costs",
                (
                    "reward_time",
                    "reward_crawl",
                    "reward_command_rate",
                    "reward_tracking",
                ),
            ),
            (
                "避障与脱困",
                "p2_obstacle_recovery_rewards",
                (
                    "reward_body_collision",
                    "reward_predictive_collision_risk",
                    "reward_missed_safe_direction",
                    "reward_frontier_stagnation",
                ),
            ),
            (
                "步态诊断不计奖励",
                "p2_gait_reward",
                (
                    "reward_gait_symmetry",
                    "gait_duty_excess",
                    "gait_swing_excess",
                    "gait_prolonged_excess",
                    "gait_frequency_excess",
                ),
            ),
            ("50Hz安全小计", "p2_frame_safety", ("reward_frame_safety",)),
            (
                "奖励守恒",
                "p2_reward_totals",
                (
                    "reward_positive_total",
                    "reward_negative_total",
                    "reward_decomposed_total",
                    "rollout_reward_mean",
                ),
            ),
        ),
    ),
    (
        "导航成效",
        "p2_navigation_effect",
        (
            (
                "即时结果率",
                "p2_outcome_rates",
                ("success_rate", "failure_rate", "timeout_rate"),
            ),
            (
                "窗口结束结果",
                "p2_terminal_outcomes",
                (
                    "rollout_success_count",
                    "rollout_failure_count",
                    "rollout_timeout_count",
                    "rollout_terminal_count",
                    "episode_success_fraction",
                    "episode_timeout_fraction",
                ),
            ),
            (
                "结束时机",
                "p2_episode_timing",
                ("early_end_rate", "transition_duration_frames"),
            ),
            ("目标距离", "p2_goal_distance", ("goal_distance",)),
            (
                "目标推进效率",
                "p2_goal_progress",
                ("goal_progress", "goal_progress_m_per_s", "goal_progress_positive_rate"),
            ),
            (
                "卡滞与终止",
                "p2_termination",
                (
                    "stuck",
                    "reward_frontier_stagnation",
                    "reward_body_collision",
                    "reward_predictive_collision_risk",
                    "hard_termination",
                    "timeout_rate",
                ),
            ),
            (
                "方向探索率",
                "p2_turn_distribution",
                ("target_left", "target_right", "target_straight"),
            ),
            (
                "探索域使用率",
                "p2_domain_usage",
                (
                    "command_core_overflow_rate",
                    "vx_near_hard_boundary_rate",
                    "wz_near_hard_boundary_rate",
                ),
            ),
        ),
    ),
    (
        "控制与反馈",
        "p2_control_feedback",
        (
            (
                "前进速度链",
                "p2_vx_response",
                ("target_vx", "exec_vx", "measured_vx", "true_vx"),
            ),
            (
                "转向速度链",
                "p2_wz_response",
                ("target_wz", "exec_wz", "measured_wz", "true_wz"),
            ),
            (
                "横向速度链",
                "p2_vy_response",
                ("target_vy", "exec_vy", "measured_vy", "true_vy"),
            ),
            (
                "Slew跟随误差",
                "p2_slew_error",
                (
                    "target_exec_vx_error", "target_exec_vy_error",
                    "target_exec_wz_error", "slew_saturation_rate",
                ),
            ),
            (
                "执行跟踪误差",
                "p2_tracking_error",
                (
                    "vx_tracking_abs_error", "vy_tracking_abs_error",
                    "wz_tracking_abs_error",
                ),
            ),
            (
                "反馈来源比例",
                "p2_feedback_sources",
                (
                    "feedback_source_invalid",
                    "feedback_source_sport",
                    "feedback_source_contact",
                ),
            ),
            (
                "反馈质量",
                "p2_feedback_quality",
                ("feedback_valid", "feedback_age_s", "feedback_true_velocity_error"),
            ),
            ("响应器可信度", "p2_adapter_confidence", ("adapter_confidence",)),
            (
                "前进速度分位数",
                "p2_vx_quantiles",
                (
                    "target_vx_p10", "target_vx_p50", "target_vx_p90",
                    "exec_vx_p10", "exec_vx_p50", "exec_vx_p90",
                    "measured_vx_p10", "measured_vx_p50", "measured_vx_p90",
                    "true_vx_p10", "true_vx_p50", "true_vx_p90",
                ),
            ),
            (
                "转向速度分位数",
                "p2_wz_quantiles",
                (
                    "target_wz_p10", "target_wz_p50", "target_wz_p90",
                    "exec_wz_p10", "exec_wz_p50", "exec_wz_p90",
                    "measured_wz_p10", "measured_wz_p50", "measured_wz_p90",
                    "true_wz_p10", "true_wz_p50", "true_wz_p90",
                ),
            ),
            (
                "横向速度分位数",
                "p2_vy_quantiles",
                (
                    "target_vy_p10", "target_vy_p50", "target_vy_p90",
                    "exec_vy_p10", "exec_vy_p50", "exec_vy_p90",
                    "measured_vy_p10", "measured_vy_p50", "measured_vy_p90",
                    "true_vy_p10", "true_vy_p50", "true_vy_p90",
                ),
            ),
            (
                "指令速度区域",
                "p2_command_speed_regions",
                ("zero_command_rate", "creep_command_rate", "stable_command_rate"),
            ),
            (
                "横向指令使用率",
                "p2_vy_usage",
                (
                    "vy_nonzero_rate",
                    "vy_positive_rate",
                    "vy_negative_rate",
                    "vy_over_core_rate",
                    "vy_over_specialty_rate",
                    "vy_near_hard_boundary_rate",
                ),
            ),
            (
                "指令组合类型",
                "p2_command_modes",
                (
                    "target_straight", "target_left", "target_right",
                    "target_pure_yaw", "target_joint_turn",
                ),
            ),
        ),
    ),
    (
        "运动安全",
        "p2_motion_safety",
        (
            (
                "机身稳定性",
                "p2_body_stability",
                ("tilt_xy_norm", "true_lateral_speed_abs"),
            ),
            (
                "机身碰撞事件",
                "p2_body_collision_events",
                (
                    "body_collision_force",
                    "body_collision_force_max",
                    "body_collision_contact",
                    "body_collision_onset",
                    "body_collision_mapping_valid",
                ),
            ),
            (
                "预测碰撞风险",
                "p2_predictive_collision",
                (
                    "predictive_collision_clearance_m",
                    "predictive_collision_stopping_distance_m",
                    "predictive_collision_risk",
                    "predictive_collision_legacy_risk",
                    "predictive_collision_wallness_left",
                    "predictive_collision_wallness_center",
                    "predictive_collision_wallness_right",
                    "predictive_collision_risk_left",
                    "predictive_collision_risk_center",
                    "predictive_collision_risk_right",
                    "reward_predictive_collision_risk",
                ),
            ),
            (
                "特权安全教师",
                "p2_safe_direction_teacher",
                (
                    "scanner_available",
                    "teacher_risk_left", "teacher_risk_center", "teacher_risk_right",
                    "selected_safe", "best_safe", "safe_gap",
                    "safe_alternative_available",
                    "selected_safest_direction",
                    "selected_safest_direction_rate",
                    "student_risk_left", "student_risk_center", "student_risk_right",
                    "reward_missed_safe_direction",
                ),
            ),
            (
                "碰撞恢复时延",
                "p2_collision_recovery",
                (
                    "collision_onset_count",
                    "collision_onset_rate",
                    "collision_trace_events",
                    "collision_wz_zero_latency_s",
                    "collision_wz_reverse_latency_s",
                    "collision_progress_recovery_latency_s",
                    "collision_target_to_exec_wz_reverse_latency_s",
                    "collision_exec_to_true_wz_zero_latency_s",
                    "collision_target_to_exec_vy_reverse_latency_s",
                    "collision_exec_to_true_vy_zero_latency_s",
                    "collision_risk_lead_time_s",
                    "collision_terminal_overlap_rate",
                ),
            ),
            (
                "低层动作负载",
                "p2_low_action_load",
                ("low_level_action_abs_mean", "low_level_action_peak_mean"),
            ),
            (
                "低层动作饱和",
                "p2_low_action_saturation",
                ("low_level_action_saturation_rate",),
            ),
            (
                "四足接触占空比",
                "p2_gait_duty",
                ("fl_duty_factor", "fr_duty_factor", "rl_duty_factor", "rr_duty_factor"),
            ),
            (
                "四足平均摆动时间",
                "p2_gait_swing",
                (
                    "fl_mean_swing_time", "fr_mean_swing_time",
                    "rl_mean_swing_time", "rr_mean_swing_time",
                ),
            ),
            (
                "四足最长悬空",
                "p2_gait_max_air",
                ("fl_max_air_time", "fr_max_air_time", "rl_max_air_time", "rr_max_air_time"),
            ),
            (
                "四足长期悬空比例",
                "p2_gait_prolonged",
                (
                    "fl_prolonged_air_ratio", "fr_prolonged_air_ratio",
                    "rl_prolonged_air_ratio", "rr_prolonged_air_ratio",
                ),
            ),
            (
                "四足步频",
                "p2_gait_frequency",
                ("fl_step_frequency", "fr_step_frequency", "rl_step_frequency", "rr_step_frequency"),
            ),
            (
                "四足接触滑移",
                "p2_gait_slip",
                ("fl_slip_speed", "fr_slip_speed", "rl_slip_speed", "rr_slip_speed"),
            ),
            (
                "步态基线状态",
                "p2_gait_baseline",
                (
                    "gait_window_valid",
                    "gait_sensor_mapping_valid",
                    "body_collision_mapping_valid",
                    "gait_baseline_finalized",
                    "gait_baseline_samples",
                ),
            ),
        ),
    ),
    (
        "响应预测",
        "p2_response",
        (
            (
                "速度预测误差",
                "p2_adapter_mae",
                (
                    "adapter_velocity_mae_02s",
                    "adapter_velocity_mae_06s",
                    "adapter_velocity_mae_10s",
                ),
            ),
            (
                "标签有效率",
                "p2_adapter_valid",
                ("adapter_valid_02s", "adapter_valid_06s", "adapter_valid_10s"),
            ),
            (
                "长期预测质量",
                "p2_adapter_quality",
                (
                    "adapter_nll_10s", "adapter_sigma_mean",
                    "adapter_coverage_1sigma", "adapter_coverage_2sigma",
                ),
            ),
            (
                "响应损失分解",
                "p2_adapter_losses",
                (
                    "adapter_velocity_loss", "adapter_pose_loss",
                    "adapter_stuck_loss", "adapter_nll_10s",
                ),
            ),
            (
                "响应基线提升",
                "p2_adapter_baselines",
                (
                    "adapter_zero_baseline_mae", "adapter_copy_exec_baseline_mae",
                    "adapter_gain_vs_zero", "adapter_gain_vs_copy_exec",
                ),
            ),
            (
                "卡滞分类质量",
                "p2_adapter_stuck",
                (
                    "adapter_stuck_prevalence", "adapter_stuck_precision",
                    "adapter_stuck_recall", "adapter_stuck_f1",
                ),
            ),
            (
                "响应样本与梯度",
                "p2_adapter_samples",
                (
                    "adapter_track_records", "adapter_parent_records",
                    "adapter_parent_replay_ratio", "adapter_gradient_norm",
                    "adapter_reset_mask_ratio",
                ),
            ),
            (
                "分赛段响应误差",
                "p2_adapter_rows",
                (
                    "adapter_slope_inv_mae", "adapter_slope_inv_sample_share",
                    "adapter_stairs_inv_mae", "adapter_stairs_inv_sample_share",
                    "adapter_maze_mae", "adapter_maze_sample_share",
                ),
            ),
            (
                "分能力域响应误差",
                "p2_adapter_domains",
                (
                    "adapter_core_mae", "adapter_core_sample_share",
                    "adapter_outer_mae", "adapter_outer_sample_share",
                ),
            ),
            (
                "横向联合域响应",
                "p2_adapter_vy_domains",
                (
                    "adapter_vy_specialty_mae",
                    "adapter_vy_specialty_sample_share",
                    "adapter_joint_outer_mae",
                    "adapter_joint_outer_sample_share",
                ),
            ),
            (
                "分可信度响应误差",
                "p2_adapter_confidence_bins",
                (
                    "adapter_confidence_low_mae", "adapter_confidence_low_share",
                    "adapter_confidence_mid_mae", "adapter_confidence_mid_share",
                    "adapter_confidence_high_mae", "adapter_confidence_high_share",
                ),
            ),
        ),
    ),
    (
        "课程诊断",
        "p2_curriculum",
        (
            (
                "赛道行变化",
                "p2_row_moves",
                (
                    "curriculum_row_promotions",
                    "curriculum_row_demotions",
                    "curriculum_row_unchanged",
                ),
            ),
            (
                "难度列变化",
                "p2_column_moves",
                ("curriculum_column_changed", "curriculum_column_unchanged"),
            ),
            (
                "课程结果累计",
                "p2_curriculum_outcomes",
                ("curriculum_successes", "curriculum_failures", "curriculum_timeouts"),
            ),
            (
                "重置起点累计",
                "p2_curriculum_starts",
                (
                    "curriculum_slope_inv_starts",
                    "curriculum_stairs_inv_starts",
                    "curriculum_maze_entry_starts",
                ),
            ),
            (
                "20列静态覆盖",
                "p2_static_columns",
                (
                    "terrain_column_l0_share", "terrain_column_l1_share",
                    "terrain_column_l2_share", "terrain_column_l3_share",
                    "terrain_column_l4_share", "terrain_column_l5_share",
                    "terrain_column_l6_share", "terrain_column_l7_share",
                    "terrain_column_l8_share", "terrain_column_l9_share",
                    "terrain_column_l10_share", "terrain_column_l11_share",
                    "terrain_column_l12_share", "terrain_column_l13_share",
                    "terrain_column_l14_share", "terrain_column_l15_share",
                    "terrain_column_l16_share", "terrain_column_l17_share",
                    "terrain_column_l18_share", "terrain_column_l19_share",
                ),
            ),
            (
                "出生行分布",
                "p2_spawn_rows",
                (
                    "terrain_spawn_row_0_share",
                    "terrain_spawn_row_1_share",
                    "terrain_spawn_row_2_share",
                ),
            ),
        ),
    ),
    (
        "指令联合域",
        "p2_command_bins",
        (
            (
                "vx分桶0覆盖",
                "p2_cmd_v0_coverage",
                (
                    "cmd_v0_w0_count", "cmd_v0_w0_share",
                    "cmd_v0_w1_count", "cmd_v0_w1_share",
                    "cmd_v0_w2_count", "cmd_v0_w2_share",
                    "cmd_v0_w3_count", "cmd_v0_w3_share",
                ),
            ),
            (
                "vx分桶1覆盖",
                "p2_cmd_v1_coverage",
                (
                    "cmd_v1_w0_count", "cmd_v1_w0_share",
                    "cmd_v1_w1_count", "cmd_v1_w1_share",
                    "cmd_v1_w2_count", "cmd_v1_w2_share",
                    "cmd_v1_w3_count", "cmd_v1_w3_share",
                ),
            ),
            (
                "vx分桶2覆盖",
                "p2_cmd_v2_coverage",
                (
                    "cmd_v2_w0_count", "cmd_v2_w0_share",
                    "cmd_v2_w1_count", "cmd_v2_w1_share",
                    "cmd_v2_w2_count", "cmd_v2_w2_share",
                    "cmd_v2_w3_count", "cmd_v2_w3_share",
                ),
            ),
            (
                "vx分桶3覆盖",
                "p2_cmd_v3_coverage",
                (
                    "cmd_v3_w0_count", "cmd_v3_w0_share",
                    "cmd_v3_w1_count", "cmd_v3_w1_share",
                    "cmd_v3_w2_count", "cmd_v3_w2_share",
                    "cmd_v3_w3_count", "cmd_v3_w3_share",
                ),
            ),
            (
                "vx分桶4覆盖",
                "p2_cmd_v4_coverage",
                (
                    "cmd_v4_w0_count", "cmd_v4_w0_share",
                    "cmd_v4_w1_count", "cmd_v4_w1_share",
                    "cmd_v4_w2_count", "cmd_v4_w2_share",
                    "cmd_v4_w3_count", "cmd_v4_w3_share",
                ),
            ),
            (
                "vx分桶0横向",
                "p2_cmd_v0_vy",
                (
                    "cmd_v0_w0_vy_nonzero_share", "cmd_v0_w0_target_abs_vy", "cmd_v0_w0_vy_outer_share",
                    "cmd_v0_w1_vy_nonzero_share", "cmd_v0_w1_target_abs_vy", "cmd_v0_w1_vy_outer_share",
                    "cmd_v0_w2_vy_nonzero_share", "cmd_v0_w2_target_abs_vy", "cmd_v0_w2_vy_outer_share",
                    "cmd_v0_w3_vy_nonzero_share", "cmd_v0_w3_target_abs_vy", "cmd_v0_w3_vy_outer_share",
                ),
            ),
            (
                "vx分桶1横向",
                "p2_cmd_v1_vy",
                (
                    "cmd_v1_w0_vy_nonzero_share", "cmd_v1_w0_target_abs_vy", "cmd_v1_w0_vy_outer_share",
                    "cmd_v1_w1_vy_nonzero_share", "cmd_v1_w1_target_abs_vy", "cmd_v1_w1_vy_outer_share",
                    "cmd_v1_w2_vy_nonzero_share", "cmd_v1_w2_target_abs_vy", "cmd_v1_w2_vy_outer_share",
                    "cmd_v1_w3_vy_nonzero_share", "cmd_v1_w3_target_abs_vy", "cmd_v1_w3_vy_outer_share",
                ),
            ),
            (
                "vx分桶2横向",
                "p2_cmd_v2_vy",
                (
                    "cmd_v2_w0_vy_nonzero_share", "cmd_v2_w0_target_abs_vy", "cmd_v2_w0_vy_outer_share",
                    "cmd_v2_w1_vy_nonzero_share", "cmd_v2_w1_target_abs_vy", "cmd_v2_w1_vy_outer_share",
                    "cmd_v2_w2_vy_nonzero_share", "cmd_v2_w2_target_abs_vy", "cmd_v2_w2_vy_outer_share",
                    "cmd_v2_w3_vy_nonzero_share", "cmd_v2_w3_target_abs_vy", "cmd_v2_w3_vy_outer_share",
                ),
            ),
            (
                "vx分桶3横向",
                "p2_cmd_v3_vy",
                (
                    "cmd_v3_w0_vy_nonzero_share", "cmd_v3_w0_target_abs_vy", "cmd_v3_w0_vy_outer_share",
                    "cmd_v3_w1_vy_nonzero_share", "cmd_v3_w1_target_abs_vy", "cmd_v3_w1_vy_outer_share",
                    "cmd_v3_w2_vy_nonzero_share", "cmd_v3_w2_target_abs_vy", "cmd_v3_w2_vy_outer_share",
                    "cmd_v3_w3_vy_nonzero_share", "cmd_v3_w3_target_abs_vy", "cmd_v3_w3_vy_outer_share",
                ),
            ),
            (
                "vx分桶4横向",
                "p2_cmd_v4_vy",
                (
                    "cmd_v4_w0_vy_nonzero_share", "cmd_v4_w0_target_abs_vy", "cmd_v4_w0_vy_outer_share",
                    "cmd_v4_w1_vy_nonzero_share", "cmd_v4_w1_target_abs_vy", "cmd_v4_w1_vy_outer_share",
                    "cmd_v4_w2_vy_nonzero_share", "cmd_v4_w2_target_abs_vy", "cmd_v4_w2_vy_outer_share",
                    "cmd_v4_w3_vy_nonzero_share", "cmd_v4_w3_target_abs_vy", "cmd_v4_w3_vy_outer_share",
                ),
            ),
            (
                "vx分桶0效果",
                "p2_cmd_v0_effect",
                (
                    "cmd_v0_w0_progress", "cmd_v0_w0_tracking_mae", "cmd_v0_w0_gait_penalty",
                    "cmd_v0_w1_progress", "cmd_v0_w1_tracking_mae", "cmd_v0_w1_gait_penalty",
                    "cmd_v0_w2_progress", "cmd_v0_w2_tracking_mae", "cmd_v0_w2_gait_penalty",
                    "cmd_v0_w3_progress", "cmd_v0_w3_tracking_mae", "cmd_v0_w3_gait_penalty",
                ),
            ),
            (
                "vx分桶1效果",
                "p2_cmd_v1_effect",
                (
                    "cmd_v1_w0_progress", "cmd_v1_w0_tracking_mae", "cmd_v1_w0_gait_penalty",
                    "cmd_v1_w1_progress", "cmd_v1_w1_tracking_mae", "cmd_v1_w1_gait_penalty",
                    "cmd_v1_w2_progress", "cmd_v1_w2_tracking_mae", "cmd_v1_w2_gait_penalty",
                    "cmd_v1_w3_progress", "cmd_v1_w3_tracking_mae", "cmd_v1_w3_gait_penalty",
                ),
            ),
            (
                "vx分桶2效果",
                "p2_cmd_v2_effect",
                (
                    "cmd_v2_w0_progress", "cmd_v2_w0_tracking_mae", "cmd_v2_w0_gait_penalty",
                    "cmd_v2_w1_progress", "cmd_v2_w1_tracking_mae", "cmd_v2_w1_gait_penalty",
                    "cmd_v2_w2_progress", "cmd_v2_w2_tracking_mae", "cmd_v2_w2_gait_penalty",
                    "cmd_v2_w3_progress", "cmd_v2_w3_tracking_mae", "cmd_v2_w3_gait_penalty",
                ),
            ),
            (
                "vx分桶3效果",
                "p2_cmd_v3_effect",
                (
                    "cmd_v3_w0_progress", "cmd_v3_w0_tracking_mae", "cmd_v3_w0_gait_penalty",
                    "cmd_v3_w1_progress", "cmd_v3_w1_tracking_mae", "cmd_v3_w1_gait_penalty",
                    "cmd_v3_w2_progress", "cmd_v3_w2_tracking_mae", "cmd_v3_w2_gait_penalty",
                    "cmd_v3_w3_progress", "cmd_v3_w3_tracking_mae", "cmd_v3_w3_gait_penalty",
                ),
            ),
            (
                "vx分桶4效果",
                "p2_cmd_v4_effect",
                (
                    "cmd_v4_w0_progress", "cmd_v4_w0_tracking_mae", "cmd_v4_w0_gait_penalty",
                    "cmd_v4_w1_progress", "cmd_v4_w1_tracking_mae", "cmd_v4_w1_gait_penalty",
                    "cmd_v4_w2_progress", "cmd_v4_w2_tracking_mae", "cmd_v4_w2_gait_penalty",
                    "cmd_v4_w3_progress", "cmd_v4_w3_tracking_mae", "cmd_v4_w3_gait_penalty",
                ),
            ),
            (
                "vx分桶0前进链",
                "p2_cmd_v0_vx_chain",
                (
                    "cmd_v0_w0_target_vx", "cmd_v0_w0_exec_vx", "cmd_v0_w0_true_vx",
                    "cmd_v0_w1_target_vx", "cmd_v0_w1_exec_vx", "cmd_v0_w1_true_vx",
                    "cmd_v0_w2_target_vx", "cmd_v0_w2_exec_vx", "cmd_v0_w2_true_vx",
                    "cmd_v0_w3_target_vx", "cmd_v0_w3_exec_vx", "cmd_v0_w3_true_vx",
                ),
            ),
            (
                "vx分桶0转向链",
                "p2_cmd_v0_wz_chain",
                (
                    "cmd_v0_w0_target_abs_wz", "cmd_v0_w0_exec_abs_wz", "cmd_v0_w0_true_abs_wz",
                    "cmd_v0_w1_target_abs_wz", "cmd_v0_w1_exec_abs_wz", "cmd_v0_w1_true_abs_wz",
                    "cmd_v0_w2_target_abs_wz", "cmd_v0_w2_exec_abs_wz", "cmd_v0_w2_true_abs_wz",
                    "cmd_v0_w3_target_abs_wz", "cmd_v0_w3_exec_abs_wz", "cmd_v0_w3_true_abs_wz",
                ),
            ),
            (
                "vx分桶0结果",
                "p2_cmd_v0_outcomes",
                (
                    "cmd_v0_w0_success", "cmd_v0_w0_failure", "cmd_v0_w0_timeout",
                    "cmd_v0_w1_success", "cmd_v0_w1_failure", "cmd_v0_w1_timeout",
                    "cmd_v0_w2_success", "cmd_v0_w2_failure", "cmd_v0_w2_timeout",
                    "cmd_v0_w3_success", "cmd_v0_w3_failure", "cmd_v0_w3_timeout",
                ),
            ),
            (
                "vx分桶1前进链",
                "p2_cmd_v1_vx_chain",
                (
                    "cmd_v1_w0_target_vx", "cmd_v1_w0_exec_vx", "cmd_v1_w0_true_vx",
                    "cmd_v1_w1_target_vx", "cmd_v1_w1_exec_vx", "cmd_v1_w1_true_vx",
                    "cmd_v1_w2_target_vx", "cmd_v1_w2_exec_vx", "cmd_v1_w2_true_vx",
                    "cmd_v1_w3_target_vx", "cmd_v1_w3_exec_vx", "cmd_v1_w3_true_vx",
                ),
            ),
            (
                "vx分桶1转向链",
                "p2_cmd_v1_wz_chain",
                (
                    "cmd_v1_w0_target_abs_wz", "cmd_v1_w0_exec_abs_wz", "cmd_v1_w0_true_abs_wz",
                    "cmd_v1_w1_target_abs_wz", "cmd_v1_w1_exec_abs_wz", "cmd_v1_w1_true_abs_wz",
                    "cmd_v1_w2_target_abs_wz", "cmd_v1_w2_exec_abs_wz", "cmd_v1_w2_true_abs_wz",
                    "cmd_v1_w3_target_abs_wz", "cmd_v1_w3_exec_abs_wz", "cmd_v1_w3_true_abs_wz",
                ),
            ),
            (
                "vx分桶1结果",
                "p2_cmd_v1_outcomes",
                (
                    "cmd_v1_w0_success", "cmd_v1_w0_failure", "cmd_v1_w0_timeout",
                    "cmd_v1_w1_success", "cmd_v1_w1_failure", "cmd_v1_w1_timeout",
                    "cmd_v1_w2_success", "cmd_v1_w2_failure", "cmd_v1_w2_timeout",
                    "cmd_v1_w3_success", "cmd_v1_w3_failure", "cmd_v1_w3_timeout",
                ),
            ),
            (
                "vx分桶2前进链",
                "p2_cmd_v2_vx_chain",
                (
                    "cmd_v2_w0_target_vx", "cmd_v2_w0_exec_vx", "cmd_v2_w0_true_vx",
                    "cmd_v2_w1_target_vx", "cmd_v2_w1_exec_vx", "cmd_v2_w1_true_vx",
                    "cmd_v2_w2_target_vx", "cmd_v2_w2_exec_vx", "cmd_v2_w2_true_vx",
                    "cmd_v2_w3_target_vx", "cmd_v2_w3_exec_vx", "cmd_v2_w3_true_vx",
                ),
            ),
            (
                "vx分桶2转向链",
                "p2_cmd_v2_wz_chain",
                (
                    "cmd_v2_w0_target_abs_wz", "cmd_v2_w0_exec_abs_wz", "cmd_v2_w0_true_abs_wz",
                    "cmd_v2_w1_target_abs_wz", "cmd_v2_w1_exec_abs_wz", "cmd_v2_w1_true_abs_wz",
                    "cmd_v2_w2_target_abs_wz", "cmd_v2_w2_exec_abs_wz", "cmd_v2_w2_true_abs_wz",
                    "cmd_v2_w3_target_abs_wz", "cmd_v2_w3_exec_abs_wz", "cmd_v2_w3_true_abs_wz",
                ),
            ),
            (
                "vx分桶2结果",
                "p2_cmd_v2_outcomes",
                (
                    "cmd_v2_w0_success", "cmd_v2_w0_failure", "cmd_v2_w0_timeout",
                    "cmd_v2_w1_success", "cmd_v2_w1_failure", "cmd_v2_w1_timeout",
                    "cmd_v2_w2_success", "cmd_v2_w2_failure", "cmd_v2_w2_timeout",
                    "cmd_v2_w3_success", "cmd_v2_w3_failure", "cmd_v2_w3_timeout",
                ),
            ),
            (
                "vx分桶3前进链",
                "p2_cmd_v3_vx_chain",
                (
                    "cmd_v3_w0_target_vx", "cmd_v3_w0_exec_vx", "cmd_v3_w0_true_vx",
                    "cmd_v3_w1_target_vx", "cmd_v3_w1_exec_vx", "cmd_v3_w1_true_vx",
                    "cmd_v3_w2_target_vx", "cmd_v3_w2_exec_vx", "cmd_v3_w2_true_vx",
                    "cmd_v3_w3_target_vx", "cmd_v3_w3_exec_vx", "cmd_v3_w3_true_vx",
                ),
            ),
            (
                "vx分桶3转向链",
                "p2_cmd_v3_wz_chain",
                (
                    "cmd_v3_w0_target_abs_wz", "cmd_v3_w0_exec_abs_wz", "cmd_v3_w0_true_abs_wz",
                    "cmd_v3_w1_target_abs_wz", "cmd_v3_w1_exec_abs_wz", "cmd_v3_w1_true_abs_wz",
                    "cmd_v3_w2_target_abs_wz", "cmd_v3_w2_exec_abs_wz", "cmd_v3_w2_true_abs_wz",
                    "cmd_v3_w3_target_abs_wz", "cmd_v3_w3_exec_abs_wz", "cmd_v3_w3_true_abs_wz",
                ),
            ),
            (
                "vx分桶3结果",
                "p2_cmd_v3_outcomes",
                (
                    "cmd_v3_w0_success", "cmd_v3_w0_failure", "cmd_v3_w0_timeout",
                    "cmd_v3_w1_success", "cmd_v3_w1_failure", "cmd_v3_w1_timeout",
                    "cmd_v3_w2_success", "cmd_v3_w2_failure", "cmd_v3_w2_timeout",
                    "cmd_v3_w3_success", "cmd_v3_w3_failure", "cmd_v3_w3_timeout",
                ),
            ),
            (
                "vx分桶4前进链",
                "p2_cmd_v4_vx_chain",
                (
                    "cmd_v4_w0_target_vx", "cmd_v4_w0_exec_vx", "cmd_v4_w0_true_vx",
                    "cmd_v4_w1_target_vx", "cmd_v4_w1_exec_vx", "cmd_v4_w1_true_vx",
                    "cmd_v4_w2_target_vx", "cmd_v4_w2_exec_vx", "cmd_v4_w2_true_vx",
                    "cmd_v4_w3_target_vx", "cmd_v4_w3_exec_vx", "cmd_v4_w3_true_vx",
                ),
            ),
            (
                "vx分桶4转向链",
                "p2_cmd_v4_wz_chain",
                (
                    "cmd_v4_w0_target_abs_wz", "cmd_v4_w0_exec_abs_wz", "cmd_v4_w0_true_abs_wz",
                    "cmd_v4_w1_target_abs_wz", "cmd_v4_w1_exec_abs_wz", "cmd_v4_w1_true_abs_wz",
                    "cmd_v4_w2_target_abs_wz", "cmd_v4_w2_exec_abs_wz", "cmd_v4_w2_true_abs_wz",
                    "cmd_v4_w3_target_abs_wz", "cmd_v4_w3_exec_abs_wz", "cmd_v4_w3_true_abs_wz",
                ),
            ),
            (
                "vx分桶4结果",
                "p2_cmd_v4_outcomes",
                (
                    "cmd_v4_w0_success", "cmd_v4_w0_failure", "cmd_v4_w0_timeout",
                    "cmd_v4_w1_success", "cmd_v4_w1_failure", "cmd_v4_w1_timeout",
                    "cmd_v4_w2_success", "cmd_v4_w2_failure", "cmd_v4_w2_timeout",
                    "cmd_v4_w3_success", "cmd_v4_w3_failure", "cmd_v4_w3_timeout",
                ),
            ),
        ),
    ),
    (
        "三段赛道",
        "p2_track_segments",
        (
            (
                "各段推进速度",
                "p2_segment_progress",
                ("slope_inv_progress_mps", "stairs_inv_progress_mps", "maze_progress_mps"),
            ),
            (
                "各段真实前进速度",
                "p2_segment_true_vx",
                ("slope_inv_true_vx", "stairs_inv_true_vx", "maze_true_vx"),
            ),
            (
                "各段目标与执行速度",
                "p2_segment_command_vx",
                (
                    "slope_inv_target_vx", "slope_inv_exec_vx",
                    "stairs_inv_target_vx", "stairs_inv_exec_vx",
                    "maze_target_vx", "maze_exec_vx",
                ),
            ),
            (
                "各段结果率",
                "p2_segment_outcomes",
                (
                    "slope_inv_success", "slope_inv_failure", "slope_inv_timeout",
                    "stairs_inv_success", "stairs_inv_failure", "stairs_inv_timeout",
                    "maze_success", "maze_failure", "maze_timeout",
                ),
            ),
            (
                "各段样本占比",
                "p2_segment_share",
                (
                    "slope_inv_sample_share",
                    "stairs_inv_sample_share",
                    "maze_sample_share",
                    "current_segment_valid",
                ),
            ),
            (
                "各段低速与旋转",
                "p2_segment_motion_modes",
                (
                    "slope_inv_stop_ratio", "slope_inv_creep_ratio", "slope_inv_spin_ratio",
                    "stairs_inv_stop_ratio", "stairs_inv_creep_ratio", "stairs_inv_spin_ratio",
                    "maze_stop_ratio", "maze_creep_ratio", "maze_spin_ratio",
                ),
            ),
            (
                "各段步态代价",
                "p2_segment_gait",
                ("slope_inv_gait_penalty", "stairs_inv_gait_penalty", "maze_gait_penalty"),
            ),
            (
                "各段预测碰撞",
                "p2_segment_predictive_collision",
                (
                    "slope_inv_predictive_clearance_m",
                    "stairs_inv_predictive_clearance_m",
                    "maze_predictive_clearance_m",
                    "slope_inv_predictive_risk",
                    "stairs_inv_predictive_risk",
                    "maze_predictive_risk",
                    "slope_inv_predictive_penalty",
                    "stairs_inv_predictive_penalty",
                    "maze_predictive_penalty",
                ),
            ),
            (
                "各段安全教师风险",
                "p2_segment_teacher_risk",
                (
                    "slope_inv_teacher_risk", "slope_inv_teacher_high_risk_rate",
                    "stairs_inv_teacher_risk", "stairs_inv_teacher_high_risk_rate",
                    "maze_teacher_risk", "maze_teacher_high_risk_rate",
                ),
            ),
        ),
    ),
    (
        "性能资源",
        "p2_runtime",
        (
            (
                "阶段耗时",
                "p2_stage_timing",
                ("rollout_time_s", "update_time_s", "env_step_time_s"),
            ),
            (
                "更新耗时",
                "p2_update_timing",
                ("actor_update_time_s", "critic_update_time_s", "adapter_update_time_s"),
            ),
            ("采样吞吐", "p2_throughput", ("samples_per_s",)),
            ("图像传输耗时", "p2_h2d", ("h2d_time_s",)),
            (
                "显存当前值",
                "p2_memory_current",
                ("memory_allocated", "memory_reserved"),
            ),
            (
                "显存峰值",
                "p2_memory_peak",
                ("max_memory_allocated", "max_memory_reserved"),
            ),
            (
                "显存与深度缓存",
                "p2_memory_guard",
                ("max_memory_reserved_ratio", "pinned_depth_bytes"),
            ),
        ),
    ),
)

TRACK_PANEL_SPECS = (
    ("赛道窗口完成数", "track_completed", "completed_count_track_l", "sum"),
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


def _build_p2_monitor():
    monitor = MonitorConfigBuilder()
    monitor.title("P2连续导航训练")
    for group_name, group_name_en, panels in P2_MONITOR_GROUPS:
        monitor.add_group(group_name=group_name, group_name_en=group_name_en)
        for name, name_en, metrics in panels:
            _add_multi_line_panel(monitor, name, name_en, metrics)
        monitor.end_group()
    return monitor.build()


def build_monitor():
    policy_entry = _configured_policy_entry()
    if policy_entry == "p15_response":
        return _build_p15_monitor()
    if policy_entry in {"p2_nav_ppo", "p2_nav_eval"}:
        return _build_p2_monitor()
    return _build_nav_monitor()
