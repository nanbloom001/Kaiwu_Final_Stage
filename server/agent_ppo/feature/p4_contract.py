#!/usr/bin/env python3
"""Versioned contracts for the P4 Track robustness stages."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch

from agent_ppo.feature import p2_contract


RUN_NAME = "p4maze8h-closedloop-r3"
STAGE_NAME = "p4_nav_ppo"
STAGE_TYPE = "p4_nav_ppo"
TRAINING_HOURS = 8.0
TARGET_EFFECTIVE_SECONDS = 28_800.0
LEGACY_MAZE_TRAINING_HOURS = 2.0
LEGACY_MAZE_TARGET_EFFECTIVE_SECONDS = 7_200.0
DIAGNOSTIC_SECONDS = 0.0
PLATFORM_WALL_MARGIN_SECONDS = 900.0
PLATFORM_WALL_SECONDS = (
    TARGET_EFFECTIVE_SECONDS
    + DIAGNOSTIC_SECONDS
    + PLATFORM_WALL_MARGIN_SECONDS
)
PLATFORM_WALL_HOURS = PLATFORM_WALL_SECONDS / 3_600.0
SCHEDULE_BOUNDARIES_SECONDS = (1_800.0, 7_200.0, 21_600.0, 28_800.0)
LEGACY_MAZE_SCHEDULE_BOUNDARIES_SECONDS = (
    600.0, 1_800.0, 3_600.0, 5_400.0, 6_300.0, 7_200.0
)

LEGACY_ACTION_MAPPER_VERSION = "p2_legacy_action_mapper_v1"
ACTION_MAPPER_VERSION = "p4_capability_action_mapper_v1"
GOAL_BELIEF_VERSION = "p4_goal_belief_v4_full_track_metric_raw"
CAMERA_CONTRACT_VERSION = "p4_shared_camera_v4_recovery_nominal_light"
ADAPTER_RECORD_CONTRACT_VERSION = "p4_adapter_record_v1"
SAFETY_REWARD_RAMP_VERSION = "p4_maze_credit_repair_safety_group_v1"
CHECKPOINT_CONTRACT_VERSION = "p4_maze_credit_repair_v1"
MAZE_CLOSED_LOOP_SAFETY_REWARD_VERSION = "p4_maze_closed_loop_recovery_translation_v4"
MAZE_CLOSED_LOOP_CHECKPOINT_VERSION = "p4_maze_closed_loop_v3"
FULL_TRACK_CHECKPOINT_CONTRACT_VERSION = "p4_full_track_v2"
FULL_TRACK_REWARD_CONTRACT_VERSION = "p4_full_track_reward_v2_potential_straight"
FULL_TRACK_COMMAND_CONTRACT_VERSION = "p4_full_track_command_v2"
WORKER_WIRE_VERSION = "p4_worker_wire_v6_full_track_spawn"
LEGACY_STUCK_RESET_CONTRACT_VERSION = "p4_stuck_reset_v3_sliding_window_10s"
STUCK_RESET_CONTRACT_VERSION = "p4_stuck_reset_v5_shadow_12s_to_active_10s"
LEGACY_ACTOR_MEAN_GUIDANCE_CONTRACT_VERSION = "p4_actor_mean_guidance_v1"
ACTOR_MEAN_GUIDANCE_CONTRACT_VERSION = "p4_actor_mean_guidance_v5_safe5_recovery_translation"
LEGACY_TRANSLATION_VECTOR_LIMITER_CONTRACT_VERSION = (
    "p4_translation_vector_limiter_v2_emergency_only"
)
TRANSLATION_VECTOR_LIMITER_CONTRACT_VERSION = "p4_translation_vector_limiter_v4_emergency_only"
LEGACY_NEAR_GOAL_CAPTURE_CONTRACT_VERSION = "p4_near_goal_capture_v1"
NEAR_GOAL_CAPTURE_CONTRACT_VERSION = "p4_near_goal_capture_v2_shadow_only"
STUCK_AUX_CONTRACT_VERSION = "p4_stuck_aux_v1"
# Training-only transport appended after the stable P3 493-column wire.
# These fields never enter either learned network and are absent from eval.
P4_WORKER_EXTRA_DIM = 26
P4_PRIVILEGED_WIRE_DIM = 519
RAW_GOAL_XY_SLICE = slice(0, 2)
STUCK_MOTION_CONFINED_INDEX = 2
STUCK_WALL_EVIDENCE_INDEX = 3
STUCK_CANDIDATE_INDEX = 4
STUCK_DURATION_S_INDEX = 5
STUCK_WOULD_RESET_INDEX = 6
STUCK_RESET_TRIGGERED_INDEX = 7
STUCK_MAPPING_VALID_INDEX = 8
STUCK_RESET_AFTER_PUSH_INDEX = 9
STUCK_SAVED_SECONDS_INDEX = 10
STUCK_COLLISION_TO_RESET_S_INDEX = 11
STUCK_TERM_AVAILABLE_INDEX = 12
STUCK_TERM_CONFIG_VALID_INDEX = 13
SPAWN_INSTALLED_INDEX = 14
STUCK_RAW_TERM_INDEX = 15
SPAWN_FULL_START_INDEX = 16
SPAWN_SEGMENT_INDEX = 17
SPAWN_QUARTILE_INDEX = 18
SPAWN_SAFE_POINT_INDEX = 19
SPAWN_REASON4_RETRY_COUNT_INDEX = 20
SPAWN_REASON4_EXHAUSTED_COUNT_INDEX = 21
SPAWN_REASON4_FALLBACK_APPLIED_COUNT_INDEX = 22
SPAWN_ALL_POSITION_APPLIED_COUNT_INDEX = 23
SPAWN_VALIDATION_FAILURE_COUNT_INDEX = 24
SPAWN_WRITE_FAILURE_COUNT_INDEX = 25

SOFT_CRUISE_MIN_VX = 0.60
SOFT_CRUISE_MAX_VX = 0.75
SOFT_CRUISE_LOW_WEIGHT = -0.03
SOFT_CRUISE_HIGH_WEIGHT = -0.02

P4_MAX_ABS_VY = 0.30
P4_MAX_ABS_WZ = 0.90
P4_MAX_VX = 1.00
P4_NAV_PERIOD_FRAMES = 5
P4_NAV_DT_S = p2_contract.CONTROL_DT_S * P4_NAV_PERIOD_FRAMES
P4_SLEW_RATE = (0.60, 0.60, 2.00)
P4_SLEW_RELEASE_RATE = (1.20, 1.20, 4.00)
LEGACY_P4_SLEW_RATE = (0.30, 0.40, 1.50)
LEGACY_P4_SLEW_RELEASE_RATE = (0.30, 0.80, 3.00)
STALE_GOAL_WAIT_MAX_ABS_VY = 0.10
STALE_GOAL_WAIT_MAX_ABS_WZ = 0.25

GOAL_NORMAL_D2 = 9.21
GOAL_CLIPPED_D2 = 25.0
GOAL_AGE_SLOW_START_S = 0.5
GOAL_AGE_SLOW_END_S = 2.0
GOAL_AGE_HOLD_END_S = 2.5
GOAL_FRESHNESS_FLOOR = 0.05
GOAL_PROCESS_SIGMA_V_M_S = 0.03
GOAL_PROCESS_SIGMA_WZ_RAD_S = 0.011
GOAL_REACQUIRE_SAMPLES = 5

STUCK_RESET_TERMINAL_PENALTY = -75.0
SUCCESS_IMPULSE = 200.0
FAILURE_IMPULSE = -60.0
TIMEOUT_IMPULSE = -40.0
LEGACY_STUCK_RESET_TERMINAL_PENALTY = -25.0
LEGACY_TIMEOUT_IMPULSE = -25.0
STUCK_SUSTAINED_GRACE_S = 0.8
STUCK_SUSTAINED_FULL_S = 2.0
STUCK_SUSTAINED_BASE = -0.010
STUCK_SUSTAINED_FLOOR = -0.05
LEGACY_STUCK_SUSTAINED_FLOOR = -0.02
P4_BODY_COLLISION_ONSET_BASE = -0.20
P4_BODY_COLLISION_ONSET_SEVERITY = -0.30
P4_BODY_COLLISION_PERSISTENT = -0.08
GOAL_SAFE_PREFERENCE_WEIGHT = -0.012
GOAL_SAFE_PREFERENCE_MARGIN = 0.08
GOAL_SAFE_PREFERENCE_SCALE = 0.35
ROUTE_EXCESS_WEIGHT = 0.0
ROUTE_EXCESS_CAP_M = 0.0
YAW_EXIT_RESPONSE_RAW_FLOOR = -0.006
TRANSLATION_LIMITER_RISK_THRESHOLD = 0.95
TRANSLATION_LIMITER_ALPHA_FLOOR = 0.75
TRANSLATION_LIMITER_RELEASE_PER_TICK = 0.20
LEGACY_TRANSLATION_LIMITER_RISK_THRESHOLD = 0.75
LEGACY_TRANSLATION_LIMITER_ALPHA_FLOOR = 0.60
NEAR_GOAL_CAPTURE_MIN_DISTANCE_M = 0.65
NEAR_GOAL_CAPTURE_MAX_DISTANCE_M = 1.20
NEAR_GOAL_CAPTURE_MIN_SPEED_M_S = 0.10
NEAR_GOAL_CAPTURE_GOAL_COSINE_MIN = 0.70
NEAR_GOAL_CAPTURE_FRESHNESS_MIN = 0.90
TEACHER_SAFE_MIN = 0.65
TEACHER_SAFE_MARGIN_MIN = 0.20
TEACHER_GOAL_SAFE_TIE_MARGIN = 0.10
TEACHER_GOAL_FRESHNESS_MIN = 0.75
TEACHER_DIRECTION_TOLERANCE_DEG = 35.0
TEACHER_EDGE_DIRECTION_TOLERANCE_DEG = 20.0
TEACHER_EDGE_SAFE_MIN = 0.65
TEACHER_EDGE_SPEED_CAP_MIN = 0.15
TEACHER_EDGE_SPEED_CAP_RANGE = 0.55
TEACHER_RECOVERY_MAX_VX = 0.12
TEACHER_RECOVERY_MIN_ABS_VY = 0.12
TEACHER_RECOVERY_SIDE_MARGIN = 0.10
TEACHER_SPEED_RISK_MIN = 0.90
TEACHER_MIN_VALID_STEPS = 64
TEACHER_SAFE5_ANGLES_DEG = (60.0, 30.0, 0.0, -30.0, -60.0)
TEACHER_SAFE5_HEIGHT_SECTORS = ((12, 16), (9, 14), (5, 11), (2, 7), (0, 4))
STUCK_RESET_DEFAULTS = {
    "enabled": True,
    "mode": "shadow",
    "schedule_enabled": False,
    "confirmation_s": 10.0,
    "initial_confirmation_s": 12.0,
    "activation_delay_s": 1_800.0,
    "tighten_after_s": 7_200.0,
    "resume_offset_s": 0.0,
    "radius_m": 0.50,
    "min_goal_distance_m": 0.80,
    "body_collision_force_n": 30.0,
    "wall_evidence_latch_s": 2.0,
    "episode_grace_s": 5.0,
    "push_grace_s": 2.0,
    "max_true_motion_speed_m_s": 0.08,
    "terminal_penalty": LEGACY_STUCK_RESET_TERMINAL_PENALTY,
}

FULL_TRACK_SEGMENT_LABELS = (
    "slope",
    "slope_inv",
    "stairs",
    "stairs_inv",
    "maze",
)
FULL_TRACK_SEGMENT_LENGTH_M = 8.0
SEGMENT_FRONTIER_WEIGHT = 1.0
MAZE_NEW_BEST_WEIGHT_PER_M = 1.0
MAZE_NEW_BEST_EPISODE_CAP = 6.0
LEGACY_MAZE_NEW_BEST_WEIGHT_PER_M = 2.0
LEGACY_MAZE_NEW_BEST_EPISODE_CAP = 12.0
OPEN_STRAIGHT_LATERAL_WEIGHT = -0.005
OPEN_STRAIGHT_S_TURN_WEIGHT = -0.005
OPEN_STRAIGHT_EXTRA_PATH_WEIGHT = -0.0025
OPEN_STRAIGHT_TOTAL_FLOOR = -0.0125
OPEN_STRAIGHT_TRUE_VY_DEADBAND_M_S = 0.08
OPEN_STRAIGHT_GOAL_BEARING_MAX_DEG = 15.0
OPEN_STRAIGHT_CENTER_SAFE_MIN = 0.75
OPEN_STRAIGHT_CENTER_BEST_MARGIN = 0.05
OPEN_STRAIGHT_BOUNDARY_MARGIN_M = 0.80

GOAL_JUMP_RATE_PER_S = 0.015
GOAL_JUMP_DURATION_S = (0.20, 0.80)
GOAL_JUMP_MIN_DISTANCE_M = 3.0
GOAL_JUMP_RADIAL_RANGE_M = (0.04, 0.20)
GOAL_JUMP_TANGENT_RANGE_M = (0.10, 1.00)


def normalize_stuck_reset_contract(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the canonical runtime contract used by worker and checkpoint."""
    supplied = dict(config or {})
    unknown = set(supplied) - set(STUCK_RESET_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown P4 stuck-reset fields: {sorted(unknown)}")
    merged = dict(STUCK_RESET_DEFAULTS)
    merged.update(supplied)
    mode = str(merged["mode"])
    if mode not in {"shadow", "active", "disabled"}:
        raise ValueError(f"unsupported P4 stuck-reset mode {mode!r}")
    result = {
        "enabled": bool(merged["enabled"]),
        "mode": mode,
        "schedule_enabled": bool(merged["schedule_enabled"]),
        "confirmation_s": float(merged["confirmation_s"]),
        "initial_confirmation_s": float(merged["initial_confirmation_s"]),
        "activation_delay_s": float(merged["activation_delay_s"]),
        "tighten_after_s": float(merged["tighten_after_s"]),
        "resume_offset_s": float(merged["resume_offset_s"]),
        "radius_m": float(merged["radius_m"]),
        "min_goal_distance_m": float(merged["min_goal_distance_m"]),
        "body_collision_force_n": float(merged["body_collision_force_n"]),
        "wall_evidence_latch_s": float(merged["wall_evidence_latch_s"]),
        "episode_grace_s": float(merged["episode_grace_s"]),
        "push_grace_s": float(merged["push_grace_s"]),
        "max_true_motion_speed_m_s": float(
            merged["max_true_motion_speed_m_s"]
        ),
        "terminal_penalty": float(merged["terminal_penalty"]),
    }
    numeric = tuple(
        value
        for key, value in result.items()
        if key not in {"enabled", "mode", "schedule_enabled"}
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("P4 stuck-reset numeric fields must be finite")
    if any(
        result[key] < 0.0
        for key in (
            "confirmation_s",
            "initial_confirmation_s",
            "activation_delay_s",
            "tighten_after_s",
            "resume_offset_s",
            "radius_m",
            "min_goal_distance_m",
            "body_collision_force_n",
            "wall_evidence_latch_s",
            "episode_grace_s",
            "push_grace_s",
            "max_true_motion_speed_m_s",
        )
    ):
        raise ValueError("P4 stuck-reset durations, distances and force must be non-negative")
    if result["terminal_penalty"] > 0.0:
        raise ValueError("P4 stuck-reset terminal_penalty must be non-positive")
    if result["tighten_after_s"] < result["activation_delay_s"]:
        raise ValueError(
            "P4 stuck-reset tighten_after_s must not precede activation_delay_s"
        )
    return result

YAW_WINDOW_SECONDS = 1.0
RISK_RESPONSE_WINDOW_SECONDS = 1.0
# Global yaw cancellation is shadow-only. It previously penalized legitimate
# rapid obstacle-avoidance reversals without feeding the intervention state
# back into the policy.
YAW_EXEC_WEIGHT = 0.0
YAW_TRUE_WEIGHT = 0.0
YAW_TOTAL_FLOOR = 0.0
SAFETY_GROUP_FLOOR = -0.070
LEGACY_YAW_EXEC_WEIGHT = -0.012
LEGACY_YAW_TRUE_WEIGHT = -0.008
LEGACY_YAW_TOTAL_FLOOR = -0.020
LEGACY_SAFETY_GROUP_FLOOR = -0.060

PREDICTIVE_COLLISION_SCALE = 1.25
PREDICTIVE_RAW_FLOOR = -0.0300
MISSED_SAFE_RAW_FLOOR = -0.0400
SAFE_DIRECTION_GAP_SCALE = 0.35

NAVIGATION_ENCODER_LRS = {
    "conv1": 3.0e-6,
    "conv2": 1.0e-5,
    "conv3": 3.0e-5,
    "fc": 3.0e-5,
}
ACTOR_LR = 3.0e-4
CRITIC_LR = 3.0e-4
SAFETY_HEAD_LR = 3.0e-4
ADAPTER_LR = 1.0e-5

MONITOR_REQUIRED_METRICS = (
    "session_wall_seconds",
    "session_effective_seconds",
    "teacher_guidance_loss",
    "teacher_direction_loss",
    "teacher_speed_loss",
    "teacher_yaw_loss",
    "teacher_edge_loss",
    "teacher_edge_active_share",
    "teacher_recovery_loss",
    "teacher_recovery_active_share",
    "teacher_guidance_valid_steps",
    "teacher_guidance_gradient_ratio",
    "mirror_aux_gradient_ratio",
    "mirror_aux_sequence_share",
    "mirror_aux_eligible_sequence_count",
    "mirror_aux_scheduled_sequence_share",
    "stuck_aux_gradient_ratio",
    "stuck_aux_loss",
    "stuck_aux_valid_steps",
    "auxiliary_gradient_ratio",
    "safety_hard_positive_share",
    "actor_stuck_pr_auc",
    "actor_stuck_precision",
    "actor_stuck_recall",
    "actor_stuck_f1",
    "actor_stuck_threshold",
    "actor_stuck_positive_share",
    "teacher_risk_left",
    "teacher_risk_center",
    "teacher_risk_right",
    "safe_alternative_available",
    "selected_safest_direction",
    "reward_goal_safe_raw",
    "reward_yaw_exit_raw",
    "reward_yaw_exit_response",
    "translation_safety_risk",
    "translation_safety_alpha_raw",
    "translation_safety_alpha",
    "translation_safety_emergency",
    "near_goal_final_translation_alpha",
    "near_goal_capture_candidate",
    "near_goal_capture_active",
    "near_goal_capture_distance_m",
    "near_goal_capture_alignment",
    "near_goal_capture_cap_m_s",
    "near_goal_capture_alpha",
    "near_goal_capture_candidate_count",
    "near_goal_capture_entry_count",
    "near_goal_capture_exit_count",
    "near_goal_capture_zone_success_count",
    "near_goal_capture_zone_collision_count",
    "near_goal_capture_zone_timeout_count",
    "near_goal_capture_zone_reset_count",
    "near_goal_capture_reset_counted_as_completion_error",
    "near_goal_capture_entry_to_platform_success_latency_s",
    "recovery_event_count_60s",
    "recovery_event_lifetime_count",
    "recovery_candidate_entry_count",
    "recovery_success_count",
    "recovery_terminal_count",
    "recovery_unverified_exit_count",
    "recovery_success_rate",
    "recovery_time_s",
    "recovery_early_stuck_sample_share",
    "recovery_confirmed_stuck_sample_share",
    "recovery_safe_exit_share",
    "recovery_candidate_lifetime_count",
    "recovery_terminal_lifetime_count",
    "goal_map_x_m",
    "goal_map_y_m",
    "goal_map_distance_m",
    "goal_age_s",
    "goal_innovation_d2",
    "goal_measurement_accepted",
    "goal_measurement_clipped",
    "goal_measurement_rejected",
    "goal_dropout_active",
    "goal_jump_active",
    "goal_propagated",
    "goal_epoch_changed",
    "goal_reacquire_pending",
    "goal_reacquisition_time_s",
    "goal_fault_allowed",
    "goal_freshness",
    "goal_process_variance_m2",
    "goal_candidate_count",
    "goal_stale_low_speed_active",
    "raw_goal_distance_gt10_share",
    "goal_innovation_d2_p50",
    "goal_innovation_d2_p90",
    "goal_innovation_d2_p99",
    "goal_age_s_p50",
    "goal_age_s_p90",
    "goal_distance_0_5_share",
    "goal_accept_0_5",
    "goal_clipped_0_5",
    "goal_reject_0_5",
    "goal_distance_5_10_share",
    "goal_accept_5_10",
    "goal_clipped_5_10",
    "goal_reject_5_10",
    "goal_distance_10_plus_share",
    "goal_accept_10_plus",
    "goal_clipped_10_plus",
    "goal_reject_10_plus",
    "user_speed_cap",
    "effective_speed_cap",
    "safety_speed_cap",
    "safety_cap_predictive_risk",
    "mapper_version_valid",
    "normalized_action_vx",
    "normalized_action_vy",
    "normalized_action_wz",
    "mapped_cmd_vx",
    "mapped_cmd_vy",
    "mapped_cmd_wz",
    "policy_target_vx",
    "policy_target_vy",
    "policy_target_wz",
    "limited_target_vx",
    "limited_target_vy",
    "limited_target_wz",
    "soft_cruise_clear_factor",
    "soft_cruise_low_error",
    "soft_cruise_high_error",
    "reward_soft_cruise",
    "teacher_scene_corridor",
    "teacher_scene_left_open",
    "teacher_scene_right_open",
    "teacher_scene_junction",
    "teacher_scene_dead_end",
    "teacher_scene_fuzzy",
    "teacher_safe_top1_clear",
    "diagnostic_teacher_coverage",
    "diagnostic_wall_auroc",
    "diagnostic_wall_miss_rate",
    "diagnostic_safe_top1_accuracy",
    "diagnostic_scene_macro_f1",
    "diagnostic_clean_live_latent_cosine",
    "diagnostic_goal_wall_auroc",
    "diagnostic_goal_safe_top1_accuracy",
    "diagnostic_goal_scene_macro_f1",
    "diagnostic_fault_wall_auroc",
    "diagnostic_fault_safe_top1_accuracy",
    "diagnostic_fault_scene_macro_f1",
    "diagnostic_fault_shadow_share",
    "diagnostic_clean_fault_latent_cosine",
    "diagnostic_clean_fault_action_mae",
    "scanner_valid_share",
    "safety_bce",
    "safety_head_risk_left",
    "safety_head_risk_center",
    "safety_head_risk_right",
    "head_correct_samples",
    "head_correct_actor_wrong_count",
    "head_correct_actor_wrong_rate",
    "risk_event_resolved_count",
    "risk_decel_policy_count",
    "risk_decel_limited_count",
    "risk_no_deceleration_count",
    "risk_decel_policy_rate",
    "risk_decel_limited_rate",
    "risk_no_deceleration_rate",
    "missed_safe_event_active",
    "missed_safe_event_severity",
    "missed_safe_weight",
    "body_collision_onset",
    "predictive_collision_risk",
    "zero_hidden_action_mae",
    "zero_hidden_direction_disagreement",
    "yaw_exec_cancellation",
    "yaw_true_cancellation",
    "yaw_exec_sign_flip",
    "yaw_true_sign_flip",
    "yaw_true_overshoot",
    "reward_predictive_raw",
    "reward_missed_safe_raw",
    "reward_yaw_raw",
    "reward_continuous_time_scale",
    "reward_predictive_collision_risk",
    "reward_missed_safe_direction",
    "reward_yaw_cancellation",
    "reward_safety_group_scale",
    "reward_decomposed_total",
    "reward_conservation_error",
    "camera_frame_id",
    "camera_capture",
    "camera_frame_changed",
    "camera_age_s",
    "camera_raw_hole_rate",
    "camera_near_clip_added_hole_rate",
    "camera_delivered_hole_rate",
    "camera_center_hole_rate",
    "camera_lower_hole_rate",
    "near_clip_m",
    "near_clip_normalized",
    "camera_delay_only",
    "camera_fault_only",
    "camera_fault_delay_overlap",
    "camera_fault_kind",
    "camera_shadow_age_250ms",
    "motion_confined_share",
    "wall_evidence_share",
    "wall_stuck_candidate_share",
    "wall_stuck_duration_s",
    "wall_stuck_would_reset",
    "wall_stuck_reset_triggered",
    "wall_stuck_raw_term",
    "wall_stuck_reset_rate",
    "rollout_wall_stuck_reset_count",
    "wall_stuck_mapping_valid",
    "reset_after_push_share",
    "wall_stuck_saved_seconds",
    "rollout_wall_stuck_saved_seconds",
    "wall_stuck_duration_p50_s",
    "wall_stuck_duration_p90_s",
    "collision_to_stuck_reset_delay_s",
    "wall_stuck_term_available",
    "wall_stuck_term_config_valid",
    "p4_spawn_hook_installed",
    "reward_stuck_reset",
    "reward_stuck_sustained",
    "reward_goal_safe_preference",
    "reward_route_excess",
    "reward_open_straight",
    "open_straight_eligible",
    "open_straight_lateral_penalty",
    "open_straight_s_turn_penalty",
    "open_straight_extra_path_penalty",
    "current_segment_slope_share",
    "current_segment_slope_inv_share",
    "current_segment_stairs_share",
    "current_segment_stairs_inv_share",
    "current_segment_maze_share",
    "spawn_segment_slope_event_share",
    "spawn_segment_slope_inv_event_share",
    "spawn_segment_stairs_event_share",
    "spawn_segment_stairs_inv_event_share",
    "spawn_segment_maze_event_share",
    "spawn_safe_point_event_share",
    "spawn_hard_position_event_share",
    "spawn_full_start_event_share",
    "spawn_segment_start_event_share",
    "spawn_reset_event_count",
    "spawn_reason4_retry_count",
    "spawn_reason4_exhausted_count",
    "spawn_reason4_fallback_applied_count",
    "spawn_validation_failure_count",
    "spawn_write_failure_count",
    "goal_jump_radial_offset_m",
    "goal_jump_tangent_offset_m",
    "legitimate_side_goal_selection_rate",
    "policy_target_vy_positive_mean",
    "policy_target_vy_negative_mean",
    "policy_target_wz_positive_mean",
    "policy_target_wz_negative_mean",
    "wall_stuck_sustained_active",
    "wall_stuck_sustained_severity",
    "goal_safe_preference_eligible",
    "goal_safe_preference_gap",
    "goal_safe_preference_selected",
    "goal_safe_preference_best",
    "route_path_length_m",
    "route_positive_progress_m",
    "route_excess_distance_m",
    "route_efficiency",
    "reward_frontier_stagnation",
    "reward_frontier_stagnation_shadow",
    "maze_new_best_credit",
    "maze_new_best_delta_m",
    "maze_new_best_episode_earned",
    "terminal_potential_clawback",
    "stuck_terminal_count",
    "stuck_terminal_episode_return_mean",
    "stuck_terminal_nonnegative_rate",
    "episode_starts_per_hour",
    "camera_memory_loss",
    "camera_clean_live_action_mae",
    "camera_clean_live_latent_cosine",
    "camera_aux_coefficient",
    "camera_aux_gradient_ratio",
    "camera_delay_only_share",
    "camera_fault_only_share",
    "camera_fault_delay_overlap_share",
    "push_term_assembly_valid",
    "push_runtime_active",
    "push_telemetry_valid",
    "push_epoch",
    "push_event_count",
    "push_lifetime_count",
    "push_env_coverage",
    "seconds_since_push",
    "push_actual_delta_vx_mean",
    "push_actual_delta_vy_mean",
    "push_actual_delta_vx_abs_max",
    "push_actual_delta_vy_abs_max",
    "push_rollout_event_count_total",
    "push_lifetime_event_count_total",
    "push_env_coverage_rate",
    "push_grace_active",
    "push_tracking_response_mae",
    "push_recovery_pending",
    "push_tracking_recovery_time_s",
    "adapter_push_rejected_02s",
    "adapter_push_rejected_06s",
    "adapter_push_rejected_10s",
    "adapter_push_rejected_pose",
    "adapter_compatible_current_records",
    "adapter_compatible_parent_records",
    "adapter_compat_migrated_legacy_parent_records",
    "adapter_legacy_parent_rejected_records",
    "adapter_compat_rejected_records",
    "adapter_target_current_ratio",
    "adapter_target_p35_ratio",
    "adapter_target_earlier_ratio",
    "adapter_latest_replay_ratio",
    "adapter_recent_replay_ratio",
    "adapter_parent_replay_ratio",
    "adapter_frozen",
    "navigation_learning_rate",
    "safety_head_learning_rate",
    "adapter_learning_rate",
    "adapter_compat_rejected_missing_contract",
    "adapter_compat_rejected_mismatch_version",
    "adapter_compat_rejected_mismatch_schema",
    "adapter_compat_rejected_mismatch_low_level_digest",
    "adapter_compat_rejected_mismatch_feedback_digest",
    "adapter_compat_rejected_mismatch_capability_digest",
    "adapter_compat_rejected_mismatch_response_profile15",
    "adapter_compat_rejected_mismatch_action_mapper",
    "adapter_compat_rejected_mismatch_observation_layout",
    "adapter_compat_rejected_mismatch_label_layout",
    "low_digest_drift",
    "low_optimizer_steps",
    "high_updates",
    "p4_tbptt_sequences_per_env",
    "p4_tbptt_nav_ticks",
    "rollout_reward_mean",
    "memory_allocated",
    "memory_reserved",
    "max_memory_allocated",
    "max_memory_reserved",
    "samples_per_s",
    "full_phase_warm",
    "full_phase_adapt",
    "full_phase_train",
    "full_phase_stabilize",
    "closedloop_phase_critic_warm",
    "closedloop_phase_actor_adapt",
    "closedloop_phase_train",
    "closedloop_phase_stabilize",
    "teacher_safe5_far_left",
    "teacher_safe5_left",
    "teacher_safe5_center",
    "teacher_safe5_right",
    "teacher_safe5_far_right",
    "large_goal_correct_yaw_response",
    "vy_substitutes_yaw_count",
    "translation_limiter_intervention",
    "translation_limiter_shadow",
    "legitimate_side_goal_candidate_count",
    "legitimate_side_goal_selected_count",
    "exec_vy",
    "exec_wz",
    "true_vy",
    "true_wz",
)

# P4 retains the broad P2 dashboard for diagnostics.  These inherited panels
# are intentionally excluded from the P4 health denominator; each prefix is a
# complete P2 metric family rather than a fallback for an unknown P4 key.
MONITOR_OPTIONAL_METRICS = (
    "maze_branch_actor_attack",
    "maze_branch_visual_recovery",
    "maze_phase_probe",
    "maze_phase_attack",
    "maze_phase_hard",
    "maze_phase_final",
    "abnormal_count_track_l",
    "completed_count_track_l",
    "energy_score_track_l",
    "p4_monitor_empty_metric_count",
    "p4_monitor_expected_metric_count",
    "p4_monitor_longest_data_age_s",
    "p4_monitor_metric_with_data_count",
    "pose_score_track_l",
    "stuck",
    "time_score_track_l",
    "timeout_rate",
    "timeout_count_track_l",
    "total_score_track_l",
)
MONITOR_OPTIONAL_METRIC_PREFIXES = (
    "completed_count_track_l",
    "abnormal_count_track_l",
    "timeout_count_track_l",
    "total_score_track_l",
    "energy_score_track_l",
    "pose_score_track_l",
    "time_score_track_l",
)

ADAPTER_COMPATIBILITY_REJECTION_REASONS = (
    "missing_contract",
    "mismatch_version",
    "mismatch_schema",
    "mismatch_low_level_digest",
    "mismatch_feedback_digest",
    "mismatch_capability_digest",
    "mismatch_response_capability_profile15",
    "mismatch_action_mapper",
    "mismatch_observation_layout",
    "mismatch_label_layout",
)


def adapter_compatibility_metric_name(reason: str) -> str:
    """Return a platform-safe metric name for one compatibility reason."""
    suffix = (
        "mismatch_response_profile15"
        if reason == "mismatch_response_capability_profile15"
        else str(reason)
    )
    return f"adapter_compat_rejected_{suffix}"


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def map_normalized_action(
    normalized_action: torch.Tensor,
    effective_max_vx: torch.Tensor | float,
    goal_freshness: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map normalized PPO coordinates without changing their log-probability."""
    if normalized_action.shape[-1] != 3:
        raise ValueError("P4 normalized action must end in three coordinates")
    bounded = torch.nan_to_num(
        torch.clamp(normalized_action, -1.0, 1.0), nan=0.0, posinf=1.0, neginf=-1.0
    )
    cap = torch.as_tensor(
        effective_max_vx, device=bounded.device, dtype=bounded.dtype
    )
    if cap.ndim == bounded.ndim - 1:
        cap = cap.unsqueeze(-1)
    cap = torch.clamp(cap, 0.0, P4_MAX_VX)
    vx = 0.5 * cap * (bounded[..., 0:1] + 1.0)
    vy = P4_MAX_ABS_VY * bounded[..., 1:2]
    wz = P4_MAX_ABS_WZ * bounded[..., 2:3]
    if goal_freshness is not None:
        freshness = torch.as_tensor(
            goal_freshness, device=bounded.device, dtype=bounded.dtype
        )
        if freshness.ndim == bounded.ndim - 1:
            freshness = freshness.unsqueeze(-1)
        waiting = freshness <= 0.0
        stale = (freshness > 0.0) & (
            freshness <= GOAL_FRESHNESS_FLOOR + 1.0e-6
        )
        vy = torch.where(waiting, torch.zeros_like(vy), vy)
        vy = torch.where(
            stale,
            torch.clamp(
                vy,
                -STALE_GOAL_WAIT_MAX_ABS_VY,
                STALE_GOAL_WAIT_MAX_ABS_VY,
            ),
            vy,
        )
        wz = torch.where(
            waiting | stale,
            torch.clamp(wz, -STALE_GOAL_WAIT_MAX_ABS_WZ, STALE_GOAL_WAIT_MAX_ABS_WZ),
            wz,
        )
    return torch.cat((vx, vy, wz), dim=-1)


def map_normalized_action_legacy(normalized_action: torch.Tensor) -> torch.Tensor:
    """Versioned parent mapper used only for migration/equality validation."""
    return p2_contract.map_normalized_action(normalized_action, hard_abs_vy=0.40)


def stale_goal_cap(user_cap: torch.Tensor, goal_freshness: torch.Tensor) -> torch.Tensor:
    """Apply the GoalBelief age contract encoded by goal4 freshness."""
    cap = torch.clamp(user_cap, 0.0, P4_MAX_VX)
    freshness = torch.clamp(goal_freshness, 0.0, 1.0)
    # freshness=1 through 0.5 s, then linearly reaches 0 at 2.5 s.
    age = GOAL_AGE_SLOW_START_S + (1.0 - freshness) * (
        GOAL_AGE_HOLD_END_S - GOAL_AGE_SLOW_START_S
    )
    slow_ratio = torch.clamp(
        (age - GOAL_AGE_SLOW_START_S)
        / (GOAL_AGE_SLOW_END_S - GOAL_AGE_SLOW_START_S),
        0.0,
        1.0,
    )
    slowing = cap + slow_ratio * (torch.minimum(cap, torch.full_like(cap, 0.25)) - cap)
    holding = torch.minimum(cap, torch.full_like(cap, 0.20))
    result = torch.where(age <= GOAL_AGE_SLOW_END_S, slowing, holding)
    return torch.where(freshness > 0.0, result, torch.zeros_like(result))


def effective_speed_cap(
    user_cap: torch.Tensor,
    goal_freshness: torch.Tensor,
    safety_cap: torch.Tensor | float,
) -> torch.Tensor:
    goal_cap = stale_goal_cap(user_cap, goal_freshness)
    safety = torch.as_tensor(safety_cap, device=user_cap.device, dtype=user_cap.dtype)
    return torch.minimum(
        torch.minimum(goal_cap, torch.clamp(safety, 0.0, P4_MAX_VX)),
        torch.full_like(goal_cap, P4_MAX_VX),
    )


def translation_vector_limiter(
    policy_cmd3: torch.Tensor,
    predictive_risk: torch.Tensor,
    alpha_prev: torch.Tensor,
    *,
    reset_mask: torch.Tensor | None = None,
    risk_threshold: float = TRANSLATION_LIMITER_RISK_THRESHOLD,
    alpha_floor: float = TRANSLATION_LIMITER_ALPHA_FLOOR,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Limit the complete translational vector using deployable depth risk.

    ``alpha_prev`` is caller-owned live state.  This function deliberately does
    not retain it, so reset/resume can restore the prescribed alpha=1.0 without
    putting limiter state in a checkpoint.
    """
    if policy_cmd3.ndim != 2 or policy_cmd3.shape[1] != 3:
        raise ValueError("P4 translation limiter expects policy_cmd3=[N,3]")
    count = policy_cmd3.shape[0]
    risk = torch.as_tensor(
        predictive_risk, device=policy_cmd3.device, dtype=policy_cmd3.dtype
    ).reshape(-1)
    previous = torch.as_tensor(
        alpha_prev, device=policy_cmd3.device, dtype=policy_cmd3.dtype
    ).reshape(-1)
    if risk.numel() != count or previous.numel() != count:
        raise ValueError("P4 translation limiter batch shape drift")
    if reset_mask is None:
        reset = torch.zeros(count, device=policy_cmd3.device, dtype=torch.bool)
    else:
        reset = torch.as_tensor(reset_mask, device=policy_cmd3.device).reshape(-1).bool()
        if reset.numel() != count:
            raise ValueError("P4 translation limiter reset shape drift")
    risk = torch.nan_to_num(risk, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    threshold = float(risk_threshold)
    floor = float(alpha_floor)
    if not 0.0 <= threshold < 1.0 or not 0.0 < floor <= 1.0:
        raise ValueError("P4 translation limiter threshold/floor are invalid")
    emergency = torch.clamp(
        (risk - threshold)
        / (1.0 - threshold),
        0.0,
        1.0,
    )
    raw_alpha = 1.0 - (1.0 - floor) * emergency
    previous = torch.nan_to_num(previous, nan=1.0, posinf=1.0, neginf=1.0).clamp(
        floor, 1.0
    )
    previous = torch.where(reset, torch.ones_like(previous), previous)
    # Tightening is immediate; only release is rate limited at the 10 Hz tick.
    alpha = torch.minimum(
        raw_alpha,
        previous + TRANSLATION_LIMITER_RELEASE_PER_TICK,
    )
    limited = policy_cmd3.clone()
    limited[:, :2] *= alpha.unsqueeze(-1)
    return limited, {
        "translation_safety_alpha_raw": raw_alpha,
        "translation_safety_alpha": alpha,
        "translation_safety_risk": risk,
        "translation_safety_emergency": emergency,
    }


def near_goal_capture(
    policy_cmd3: torch.Tensor,
    goal_xy_m: torch.Tensor,
    goal_freshness: torch.Tensor,
    safety_alpha: torch.Tensor,
    terminal: torch.Tensor,
    reset: torch.Tensor,
    goal_epoch_changed: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the near-goal translation cap without changing direction or yaw."""
    if policy_cmd3.ndim != 2 or policy_cmd3.shape[1] != 3:
        raise ValueError("P4 near-goal capture expects policy_cmd3=[N,3]")
    if goal_xy_m.shape != (policy_cmd3.shape[0], 2):
        raise ValueError("P4 near-goal capture expects goal_xy_m=[N,2]")
    count = policy_cmd3.shape[0]

    def _flat(value: torch.Tensor, name: str, *, boolean: bool = False) -> torch.Tensor:
        result = torch.as_tensor(value, device=policy_cmd3.device).reshape(-1)
        if result.numel() != count:
            raise ValueError(f"P4 near-goal capture {name} shape drift")
        return result.bool() if boolean else result.to(dtype=policy_cmd3.dtype)

    freshness = _flat(goal_freshness, "freshness").clamp(0.0, 1.0)
    alpha_safe = _flat(safety_alpha, "safety_alpha").clamp(0.0, 1.0)
    terminal_mask = _flat(terminal, "terminal", boolean=True)
    reset_mask = _flat(reset, "reset", boolean=True)
    epoch_changed = _flat(goal_epoch_changed, "goal_epoch_changed", boolean=True)
    goal = torch.nan_to_num(goal_xy_m.to(policy_cmd3), nan=0.0, posinf=0.0, neginf=0.0)
    policy_xy = policy_cmd3[:, :2]
    speed = torch.linalg.vector_norm(policy_xy, dim=-1)
    distance = torch.linalg.vector_norm(goal, dim=-1)
    goal_unit = goal / distance.unsqueeze(-1).clamp_min(1.0e-6)
    policy_unit = policy_xy / speed.unsqueeze(-1).clamp_min(1.0e-6)
    alignment = (policy_unit * goal_unit).sum(dim=-1)
    candidate = (
        (freshness >= NEAR_GOAL_CAPTURE_FRESHNESS_MIN)
        & ~terminal_mask
        & ~reset_mask
        & ~epoch_changed
        & (distance > NEAR_GOAL_CAPTURE_MIN_DISTANCE_M)
        & (distance < NEAR_GOAL_CAPTURE_MAX_DISTANCE_M)
        & (speed > NEAR_GOAL_CAPTURE_MIN_SPEED_M_S)
        & (alignment >= NEAR_GOAL_CAPTURE_GOAL_COSINE_MIN)
    )
    capture_cap = 0.10 + 0.35 * torch.clamp(
        (distance - 0.70) / 0.50, min=0.0, max=1.0
    )
    alpha_capture = torch.minimum(torch.ones_like(speed), capture_cap / speed.clamp_min(1.0e-6))
    applied_alpha = torch.where(
        candidate, torch.minimum(alpha_safe, alpha_capture), alpha_safe
    )
    limited = policy_cmd3.clone()
    limited[:, :2] *= applied_alpha.unsqueeze(-1)
    return limited, {
        "near_goal_capture_candidate": candidate.float(),
        "near_goal_capture_active": candidate.float(),
        "near_goal_capture_distance_m": distance,
        "near_goal_capture_alignment": alignment,
        "near_goal_capture_cap_m_s": capture_cap,
        "near_goal_capture_alpha": torch.where(
            candidate, alpha_capture, torch.ones_like(alpha_capture)
        ),
        "near_goal_final_translation_alpha": applied_alpha,
    }


def teacher_guidance_mask(
    *,
    alive: torch.Tensor,
    scanner_valid: torch.Tensor,
    mapping_valid: torch.Tensor,
    terminal: torch.Tensor,
    reset: torch.Tensor,
    push_grace: torch.Tensor,
    episode_grace: torch.Tensor,
    goal_freshness: torch.Tensor,
    safe3: torch.Tensor,
    safe5: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build rollout-time masks for the non-privileged Actor mean guidance."""
    if safe3.ndim != 2 or safe3.shape[1] != 3:
        raise ValueError("P4 teacher guidance expects safe3=[N,3]")
    count = safe3.shape[0]

    def _flat(value: torch.Tensor, name: str, *, boolean: bool = True) -> torch.Tensor:
        result = torch.as_tensor(value, device=safe3.device).reshape(-1)
        if result.numel() != count:
            raise ValueError(f"P4 teacher guidance {name} shape drift")
        return result.bool() if boolean else result.to(dtype=safe3.dtype)

    values = torch.nan_to_num(safe3, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if safe5 is not None:
        safe5_values = torch.as_tensor(safe5, device=safe3.device)
        if safe5_values.shape != (count, 5):
            raise ValueError("P4 teacher guidance expects safe5=[N,5]")
        values = torch.nan_to_num(
            safe5_values, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
    top2 = torch.topk(values, k=2, dim=-1).values
    best_safe = top2[:, 0]
    margin = best_safe - top2[:, 1]
    freshness = _flat(goal_freshness, "goal_freshness", boolean=False)
    valid = (
        _flat(alive, "alive")
        & _flat(scanner_valid, "scanner_valid")
        & _flat(mapping_valid, "mapping_valid")
        & ~_flat(terminal, "terminal")
        & ~_flat(reset, "reset")
        & ~_flat(push_grace, "push_grace")
        & ~_flat(episode_grace, "episode_grace")
        & (best_safe >= TEACHER_SAFE_MIN)
    )
    clear_best = margin >= TEACHER_SAFE_MARGIN_MIN
    fresh_goal = freshness >= TEACHER_GOAL_FRESHNESS_MIN
    goal_tie_break = fresh_goal if safe5 is not None else torch.zeros_like(fresh_goal)
    base = valid & (clear_best | goal_tie_break)
    goal_eligible = (
        valid & goal_tie_break
        if safe5 is not None
        else base & fresh_goal
    )
    return {
        "teacher_guidance_eligible": base,
        "teacher_guidance_goal_eligible": goal_eligible,
        "teacher_best_safe": best_safe,
        "teacher_safe_margin": margin,
    }


def privileged_safe_directions5(
    critic_obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build a five-direction training-only safety teacher from existing sensors.

    The nav scanner contract remains three-directional. Its conservative wall
    risk is interpolated to five sectors, while the 16x16 height scan provides
    distinct overlapping terrain-continuity evidence for each direction. This
    does not change Actor/Critic observations or any deployment interface.
    """
    if critic_obs.ndim != 2 or critic_obs.shape[1] < p2_contract.CRITIC_OBS_DIM:
        raise ValueError("P4 safe5 teacher critic observation shape drift")
    height = critic_obs[:, 60:316].reshape(-1, 16, 16)
    nav_priv = critic_obs[:, 319:323]
    scanner_available = nav_priv[:, 0] > 0.5
    left = nav_priv[:, 2]
    center = nav_priv[:, 1]
    right = nav_priv[:, 3]
    nav_risk5 = torch.stack(
        (
            left,
            torch.maximum(left, center),
            center,
            torch.maximum(center, right),
            right,
        ),
        dim=-1,
    )
    nav_risk5 = torch.nan_to_num(
        nav_risk5, nan=1.0, posinf=1.0, neginf=1.0
    ).clamp(0.0, 1.0)

    passable = []
    sector_valid = []
    jump90 = []
    for y0, y1 in TEACHER_SAFE5_HEIGHT_SECTORS:
        region = height[:, y0:y1, : p2_contract.SAFETY_HEIGHT_FORWARD_COLS]
        finite = torch.isfinite(region)
        finite_ratio = finite.float().mean(dim=(1, 2))
        pair_valid = finite[:, :, 1:] & finite[:, :, :-1]
        diff = torch.where(
            pair_valid,
            torch.abs(torch.diff(region, dim=2)),
            torch.full_like(region[:, :, 1:], float("nan")),
        )
        jump = torch.nanquantile(diff.flatten(1), 0.90, dim=1)
        valid = (
            (finite_ratio >= p2_contract.SAFETY_HEIGHT_FINITE_RATIO_MIN)
            & (pair_valid.sum(dim=(1, 2)) >= p2_contract.SAFETY_HEIGHT_MIN_VALID_DIFFS)
            & torch.isfinite(jump)
        )
        jump = torch.nan_to_num(
            jump, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
        )
        excess = torch.relu(jump - p2_contract.SAFETY_HEIGHT_JUMP_FREE_M)
        passable.append(
            torch.exp(-torch.square(excess / p2_contract.SAFETY_HEIGHT_JUMP_SCALE_M))
        )
        sector_valid.append(valid)
        jump90.append(jump)
    terrain_passable5 = torch.stack(passable, dim=-1)
    sector_valid5 = torch.stack(sector_valid, dim=-1)
    valid = scanner_available & sector_valid5.all(dim=-1)
    safe5 = ((1.0 - nav_risk5) * terrain_passable5).clamp(0.0, 1.0)
    safe5 = torch.where(valid[:, None], safe5, torch.zeros_like(safe5))
    return safe5, valid, {
        "nav_risk5": nav_risk5,
        "terrain_passable5": terrain_passable5,
        "height_jump90_5": torch.stack(jump90, dim=-1),
        "height_sector_valid5": sector_valid5.float(),
    }


def teacher_guidance_loss(
    policy_mean_cmd3: torch.Tensor,
    safe3: torch.Tensor,
    goal_xy_m: torch.Tensor,
    predictive_risk: torch.Tensor,
    stuck_active: torch.Tensor,
    teacher_mask: torch.Tensor,
    goal_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    *,
    min_valid_steps: int = TEACHER_MIN_VALID_STEPS,
    closed_loop_v3: bool = False,
) -> dict[str, torch.Tensor]:
    """Return tolerant direction, speed and yaw guidance without a full action teacher."""
    if policy_mean_cmd3.ndim != 2 or policy_mean_cmd3.shape[1] != 3:
        raise ValueError("P4 teacher loss expects policy_mean_cmd3=[N,3]")
    expected_sectors = 5 if closed_loop_v3 else 3
    if safe3.shape != (policy_mean_cmd3.shape[0], expected_sectors):
        raise ValueError(
            f"P4 teacher loss safe{expected_sectors} shape drift"
        )
    if goal_xy_m.shape != (policy_mean_cmd3.shape[0], 2):
        raise ValueError("P4 teacher loss goal_xy_m shape drift")
    count = policy_mean_cmd3.shape[0]

    def _flat(value: torch.Tensor, name: str, *, boolean: bool = False) -> torch.Tensor:
        result = torch.as_tensor(value, device=policy_mean_cmd3.device).reshape(-1)
        if result.numel() != count:
            raise ValueError(f"P4 teacher loss {name} shape drift")
        return result.bool() if boolean else result.to(dtype=policy_mean_cmd3.dtype)

    base = _flat(teacher_mask, "teacher_mask", boolean=True)
    goal_valid = _flat(goal_mask, "goal_mask", boolean=True)
    weights = (
        torch.ones(count, device=policy_mean_cmd3.device, dtype=policy_mean_cmd3.dtype)
        if sample_weight is None
        else _flat(sample_weight, "sample_weight").clamp_min(0.0)
    )
    values = torch.nan_to_num(safe3.to(policy_mean_cmd3), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    best_safe, pure_best_index = values.max(dim=-1)
    sector_angles = policy_mean_cmd3.new_tensor(
        TEACHER_SAFE5_ANGLES_DEG if closed_loop_v3 else (35.0, 0.0, -35.0)
    ) * (math.pi / 180.0)
    goal = torch.nan_to_num(
        goal_xy_m.to(policy_mean_cmd3), nan=0.0, posinf=0.0, neginf=0.0
    )
    goal_distance = torch.linalg.vector_norm(goal, dim=-1)
    bearing = torch.atan2(goal[:, 1], goal[:, 0]).clamp(
        min=math.radians(-75.0), max=math.radians(75.0)
    )
    if closed_loop_v3:
        safe_candidate = values >= torch.maximum(
            best_safe[:, None] - TEACHER_GOAL_SAFE_TIE_MARGIN,
            torch.full_like(values, TEACHER_SAFE_MIN),
        )
        goal_distance_to_sector = torch.abs(
            bearing[:, None] - sector_angles[None, :]
        )
        goal_distance_to_sector = torch.where(
            safe_candidate,
            goal_distance_to_sector,
            torch.full_like(goal_distance_to_sector, 1.0e6),
        )
        goal_best_index = goal_distance_to_sector.argmin(dim=-1)
        best_index = torch.where(goal_valid, goal_best_index, pure_best_index)
    else:
        best_index = pure_best_index
    safe_angle = sector_angles[best_index]
    safe_direction = torch.stack((torch.cos(safe_angle), torch.sin(safe_angle)), dim=-1)
    mean_xy = policy_mean_cmd3[:, :2]
    mean_speed = torch.linalg.vector_norm(mean_xy, dim=-1)
    mean_direction = mean_xy / mean_speed.unsqueeze(-1).clamp_min(1.0e-6)
    direction_cosine = (mean_direction * safe_direction).sum(dim=-1)
    direction_error = torch.relu(
        math.cos(math.radians(TEACHER_DIRECTION_TOLERANCE_DEG)) - direction_cosine
    )
    direction_loss = direction_error.square()

    # The regular direction term tolerates a 35 degree deviation so the Actor
    # can follow a goal through a wide corridor. At a wall edge that tolerance
    # permits forward motion aimed between a safe sector and an unsafe one.
    # Compute the safety value of the actual translation heading and use a
    # narrower, continuous correction only when that heading lacks clearance.
    heading = torch.atan2(mean_xy[:, 1], mean_xy[:, 0])
    heading_alignment = torch.cos(
        heading[:, None] - sector_angles[None, :]
    )
    heading_weights = torch.softmax(8.0 * heading_alignment, dim=-1)
    heading_safe = (heading_weights * values).sum(dim=-1)
    edge_danger = torch.relu(TEACHER_EDGE_SAFE_MIN - heading_safe) / TEACHER_EDGE_SAFE_MIN
    edge_mask = base & (mean_speed > 0.10) & (heading_safe < TEACHER_EDGE_SAFE_MIN)
    edge_direction_error = torch.relu(
        math.cos(math.radians(TEACHER_EDGE_DIRECTION_TOLERANCE_DEG))
        - direction_cosine
    )
    edge_speed_cap = TEACHER_EDGE_SPEED_CAP_MIN + (
        TEACHER_EDGE_SPEED_CAP_RANGE * heading_safe
    )
    edge_loss = edge_danger * (
        edge_direction_error.square()
        + torch.relu(mean_speed - edge_speed_cap).square()
    )

    # ``stuck_active`` is a delayed label built from the executed command and
    # measured motion.  During those samples continuing to command forward
    # motion only presses the body further into the wall.  If safe5 identifies
    # one lateral side as clearly clearer than the other, train the Actor mean
    # to reduce vx and issue a small vy escape command on that side.  This is
    # training-only guidance: no contact, scanner, or command override enters
    # the deployed Actor path.
    stuck = _flat(stuck_active, "stuck_active", boolean=True)
    recovery_mask = torch.zeros_like(base)
    recovery_loss = torch.zeros_like(mean_speed)
    if closed_loop_v3:
        left_clearance = values[:, :2].amax(dim=-1)
        right_clearance = values[:, 3:].amax(dim=-1)
        side_difference = left_clearance - right_clearance
        side_sign = torch.sign(side_difference)
        side_clear = (
            torch.maximum(left_clearance, right_clearance) >= TEACHER_SAFE_MIN
        ) & (side_difference.abs() >= TEACHER_RECOVERY_SIDE_MARGIN)
        recovery_mask = base & stuck & side_clear
        signed_vy = side_sign * policy_mean_cmd3[:, 1]
        recovery_loss = (
            torch.relu(policy_mean_cmd3[:, 0] - TEACHER_RECOVERY_MAX_VX).square()
            + torch.relu(TEACHER_RECOVERY_MIN_ABS_VY - signed_vy).square()
        )

    risk = _flat(predictive_risk, "predictive_risk").clamp(0.0, 1.0)
    speed_risk_min = TEACHER_SPEED_RISK_MIN if closed_loop_v3 else 0.65
    speed_mask = base & ((risk >= speed_risk_min) | stuck)
    speed_cap = 0.20 + 0.45 * best_safe
    speed_loss = torch.relu(mean_speed - speed_cap).square()

    goal_direction = goal / goal_distance.unsqueeze(-1).clamp_min(1.0e-6)
    compatible = (goal_direction * safe_direction).sum(dim=-1) >= math.cos(
        math.radians(TEACHER_DIRECTION_TOLERANCE_DEG)
    )
    desired_yaw = safe_angle if closed_loop_v3 else bearing
    bearing_abs_deg = desired_yaw.abs() * (180.0 / math.pi)
    required_wz = torch.where(
        bearing_abs_deg <= 15.0 + 1.0e-4,
        torch.zeros_like(bearing_abs_deg),
        torch.where(
            bearing_abs_deg <= 35.0 + 1.0e-4,
            torch.full_like(bearing_abs_deg, 0.06),
            torch.where(
                bearing_abs_deg <= 60.0 + 1.0e-4,
                torch.full_like(bearing_abs_deg, 0.12),
                torch.where(
                    bearing_abs_deg <= 90.0 + 1.0e-4,
                    torch.full_like(bearing_abs_deg, 0.18),
                    torch.zeros_like(bearing_abs_deg),
                ),
            ),
        ),
    )
    yaw_mask = (
        base & (required_wz > 0.0)
        if closed_loop_v3
        else goal_valid & compatible & (required_wz > 0.0)
    )
    signed_wz = torch.sign(desired_yaw) * policy_mean_cmd3[:, 2]
    yaw_response_loss = torch.relu(required_wz - signed_wz).square()
    if closed_loop_v3:
        vy_substitution = torch.where(
            bearing_abs_deg >= 35.0,
            torch.relu(policy_mean_cmd3[:, 1].abs() - 0.12).square(),
            torch.zeros_like(signed_wz),
        )
        yaw_loss = yaw_response_loss + 0.25 * vy_substitution
    else:
        yaw_loss = yaw_response_loss

    valid_steps = base.sum()
    active = valid_steps >= int(min_valid_steps)

    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected_weight = weights * mask.to(weights.dtype)
        return (value * selected_weight).sum() / selected_weight.sum().clamp_min(1.0)

    direction = _masked_mean(direction_loss, base)
    speed = _masked_mean(speed_loss, speed_mask)
    yaw = _masked_mean(yaw_loss, yaw_mask)
    edge = _masked_mean(edge_loss, edge_mask)
    recovery = _masked_mean(recovery_loss, recovery_mask)
    total = (
        0.40 * direction + 0.10 * speed + 0.25 * yaw + 0.15 * edge + 0.10 * recovery
        if closed_loop_v3
        else 0.45 * direction + 0.20 * speed + 0.35 * yaw
    )
    active_float = active.to(dtype=policy_mean_cmd3.dtype)
    total = total * active_float
    return {
        "loss": total,
        "direction": direction * active_float,
        "speed": speed * active_float,
        "yaw": yaw * active_float,
        "edge": edge * active_float,
        "recovery": recovery * active_float,
        "teacher_valid_steps": valid_steps.to(dtype=policy_mean_cmd3.dtype),
        "teacher_loss_active": active_float,
        "teacher_direction_mask": base.to(dtype=policy_mean_cmd3.dtype),
        "teacher_speed_mask": speed_mask.to(dtype=policy_mean_cmd3.dtype),
        "teacher_yaw_mask": yaw_mask.to(dtype=policy_mean_cmd3.dtype),
        "teacher_edge_mask": edge_mask.to(dtype=policy_mean_cmd3.dtype),
        "teacher_edge_active_share": edge_mask.to(dtype=policy_mean_cmd3.dtype).mean(),
        "teacher_recovery_mask": recovery_mask.to(dtype=policy_mean_cmd3.dtype),
        "teacher_recovery_active_share": recovery_mask.to(
            dtype=policy_mean_cmd3.dtype
        ).mean(),
    }


def soft_cruise_penalty(
    policy_target_cmd3: torch.Tensor,
    safe3: torch.Tensor,
    teacher_valid: torch.Tensor,
    goal_freshness: torch.Tensor,
    terminal: torch.Tensor,
    capture_active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Softly discourage clear-road cruising outside the preferred vx band."""
    vx = policy_target_cmd3[..., 0]
    center_safe = torch.clamp(safe3[..., 1], 0.0, 1.0)
    best_safe = torch.clamp(safe3.max(dim=-1).values, 0.0, 1.0)
    valid = teacher_valid.to(dtype=vx.dtype).reshape(-1)
    freshness = torch.clamp(goal_freshness.to(dtype=vx.dtype).reshape(-1), 0.0, 1.0)
    clear_factor = valid * freshness * center_safe * torch.clamp(
        center_safe / (best_safe + 1.0e-6), 0.0, 1.0
    )
    low_error = torch.relu((SOFT_CRUISE_MIN_VX - vx) / SOFT_CRUISE_MIN_VX)
    high_error = torch.relu((vx - SOFT_CRUISE_MAX_VX) / (P4_MAX_VX - SOFT_CRUISE_MAX_VX))
    penalty = (
        SOFT_CRUISE_LOW_WEIGHT * clear_factor * low_error.square()
        + SOFT_CRUISE_HIGH_WEIGHT * high_error.square()
    )
    disabled = terminal.reshape(-1).bool()
    if capture_active is not None:
        disabled |= torch.as_tensor(
            capture_active, device=disabled.device
        ).reshape(-1).bool()
    penalty = torch.where(disabled, torch.zeros_like(penalty), penalty)
    return penalty, {
        "soft_cruise_clear_factor": clear_factor,
        "soft_cruise_low_error": low_error,
        "soft_cruise_high_error": high_error,
    }


def safety_scene_diagnostics(
    safe3: torch.Tensor, teacher_valid: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return deployment-free labels used only for maze perception diagnostics."""
    valid = teacher_valid.to(dtype=torch.bool).reshape(-1)
    left, center, right = safe3[:, 0], safe3[:, 1], safe3[:, 2]
    safe = safe3 >= 0.65
    unsafe = safe3 <= 0.35
    top2 = torch.topk(safe3, k=2, dim=-1).values
    clear_top1 = valid & ((top2[:, 0] - top2[:, 1]) >= 0.15)
    corridor = valid & safe[:, 1] & unsafe[:, 0] & unsafe[:, 2]
    left_open = valid & safe[:, 0] & ((left - torch.maximum(center, right)) >= 0.20)
    right_open = valid & safe[:, 2] & ((right - torch.maximum(left, center)) >= 0.20)
    junction = valid & (safe.sum(dim=-1) >= 2)
    dead_end = valid & unsafe.all(dim=-1)
    labeled = corridor | left_open | right_open | junction | dead_end
    return {
        "teacher_scene_corridor": corridor.float(),
        "teacher_scene_left_open": left_open.float(),
        "teacher_scene_right_open": right_open.float(),
        "teacher_scene_junction": junction.float(),
        "teacher_scene_dead_end": dead_end.float(),
        "teacher_scene_fuzzy": (valid & ~labeled).float(),
        "teacher_safe_top1_clear": clear_top1.float(),
    }


def sustained_wall_stuck_penalty(
    duration_s: torch.Tensor,
    candidate: torch.Tensor,
    mapping_valid: torch.Tensor,
    terminal: torch.Tensor,
    *,
    confirmation_s: float,
    floor: float = STUCK_SUSTAINED_FLOOR,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize confirmed wall confinement before the terminal reset fires."""
    duration = torch.nan_to_num(duration_s.float(), nan=0.0, posinf=0.0, neginf=0.0)
    active = (
        candidate.reshape(-1).bool()
        & mapping_valid.reshape(-1).bool()
        & ~terminal.reshape(-1).bool()
        & (duration >= STUCK_SUSTAINED_GRACE_S)
    )
    del confirmation_s
    ramp_span = max(STUCK_SUSTAINED_FULL_S - STUCK_SUSTAINED_GRACE_S, 1.0e-6)
    severity = torch.clamp(
        (duration - STUCK_SUSTAINED_GRACE_S) / ramp_span, 0.0, 1.0
    )
    magnitude = abs(STUCK_SUSTAINED_BASE) + severity * (
        abs(float(floor)) - abs(STUCK_SUSTAINED_BASE)
    )
    penalty = torch.where(active, -magnitude, torch.zeros_like(magnitude))
    return penalty, {
        "wall_stuck_sustained_active": active.float(),
        "wall_stuck_sustained_severity": severity,
    }


def maze_new_best_credit(
    best_distance_before: torch.Tensor,
    end_distance: torch.Tensor,
    episode_credit_before: torch.Tensor,
    *,
    weight_per_m: float = MAZE_NEW_BEST_WEIGHT_PER_M,
    episode_cap: float = MAZE_NEW_BEST_EPISODE_CAP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reward each newly reached Maze distance once, without terminal clawback."""
    best_raw = best_distance_before.float()
    end_raw = end_distance.float()
    earned_raw = episode_credit_before.float()
    valid = (
        torch.isfinite(best_raw)
        & torch.isfinite(end_raw)
        & torch.isfinite(earned_raw)
    )
    best = torch.where(valid, best_raw.clamp_min(0.0), torch.zeros_like(best_raw))
    end = torch.where(valid, end_raw.clamp_min(0.0), best)
    earned = torch.nan_to_num(
        earned_raw, nan=0.0, posinf=float(episode_cap), neginf=0.0
    ).clamp(0.0, float(episode_cap))
    delta = torch.clamp(best - torch.minimum(best, end), min=0.0)
    raw_reward = float(weight_per_m) * delta
    reward = torch.minimum(
        raw_reward,
        torch.clamp(float(episode_cap) - earned, min=0.0),
    )
    reward = torch.where(valid, reward, torch.zeros_like(reward))
    return reward, earned + reward, delta


def goal_safe_direction_penalty(
    safe3: torch.Tensor,
    target_cmd3: torch.Tensor,
    goal_xy_m: torch.Tensor,
    teacher_valid: torch.Tensor,
    goal_freshness: torch.Tensor,
    terminal: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer the goal-facing option only among directions the teacher deems safe."""
    if safe3.ndim != 2 or safe3.shape[1] != 3:
        raise ValueError("P4 goal-safe preference expects safe3=[N,3]")
    if goal_xy_m.ndim != 2 or goal_xy_m.shape[1] != 2:
        raise ValueError("P4 goal-safe preference expects goal_xy_m=[N,2]")
    command_weights = p2_contract.command_direction_weights(target_cmd3)
    goal = torch.nan_to_num(goal_xy_m.float(), nan=0.0, posinf=0.0, neginf=0.0)
    bearing = torch.atan2(goal[:, 1], goal[:, 0]).clamp(
        min=math.radians(-80.0), max=math.radians(80.0)
    )
    centers = torch.deg2rad(
        torch.tensor(
            p2_contract.PREDICTIVE_COLLISION_SECTOR_CENTERS_DEG,
            device=goal.device,
            dtype=goal.dtype,
        )
    )
    width = math.radians(p2_contract.PREDICTIVE_COLLISION_SECTOR_WIDTH_DEG)
    goal_weights = torch.softmax(
        -0.5 * ((bearing[:, None] - centers[None, :]) / width).square(), dim=-1
    )
    score3 = torch.clamp(safe3, 0.0, 1.0) * goal_weights
    selected = (command_weights * score3).sum(dim=-1)
    best = score3.max(dim=-1).values
    gap = torch.relu(best - selected - GOAL_SAFE_PREFERENCE_MARGIN)
    top2 = torch.topk(score3, k=2, dim=-1).values
    distinct = (top2[:, 0] - top2[:, 1]) > GOAL_SAFE_PREFERENCE_MARGIN
    speed_scale = target_cmd3.new_tensor(
        (1.0, 1.0, p2_contract.CRAWL_BODY_RADIUS_M)
    )
    moving = torch.linalg.vector_norm(target_cmd3 * speed_scale, dim=-1) > 0.05
    goal_distance = torch.linalg.vector_norm(goal, dim=-1)
    eligible = (
        teacher_valid.reshape(-1).bool()
        & (goal_freshness.reshape(-1) >= TEACHER_GOAL_FRESHNESS_MIN)
        & moving
        & distinct
        & (safe3.max(dim=-1).values >= p2_contract.SAFE_DIRECTION_MIN_BEST_SAFE)
        & (goal_distance >= 0.60)
        & ~terminal.reshape(-1).bool()
        & (gap > 0.0)
    )
    normalized_gap = torch.clamp(
        gap / GOAL_SAFE_PREFERENCE_SCALE, 0.0, 1.0
    )
    penalty = torch.where(
        eligible,
        GOAL_SAFE_PREFERENCE_WEIGHT * normalized_gap,
        torch.zeros_like(normalized_gap),
    )
    return penalty, {
        "goal_safe_preference_eligible": eligible.float(),
        "goal_safe_preference_gap": gap,
        "goal_safe_preference_selected": selected,
        "goal_safe_preference_best": best,
    }


def yaw_exit_response_penalty(
    safe3: torch.Tensor,
    target_cmd3: torch.Tensor,
    goal_xy_m: torch.Tensor,
    teacher_valid: torch.Tensor,
    goal_freshness: torch.Tensor,
    terminal: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Require a small correctly signed yaw response at a clear side exit."""
    if safe3.ndim != 2 or safe3.shape[1] != 3:
        raise ValueError("P4 yaw-exit response expects safe3=[N,3]")
    if target_cmd3.shape != (safe3.shape[0], 3):
        raise ValueError("P4 yaw-exit response expects target_cmd3=[N,3]")
    if goal_xy_m.shape != (safe3.shape[0], 2):
        raise ValueError("P4 yaw-exit response expects goal_xy_m=[N,2]")

    values = torch.nan_to_num(
        safe3.to(target_cmd3), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    goal = torch.nan_to_num(
        goal_xy_m.to(target_cmd3), nan=0.0, posinf=0.0, neginf=0.0
    )
    bearing = torch.atan2(goal[:, 1], goal[:, 0]).clamp(
        min=math.radians(-80.0), max=math.radians(80.0)
    )
    centers = target_cmd3.new_tensor(
        p2_contract.PREDICTIVE_COLLISION_SECTOR_CENTERS_DEG
    ) * (math.pi / 180.0)
    width = math.radians(p2_contract.PREDICTIVE_COLLISION_SECTOR_WIDTH_DEG)
    goal_weights = torch.softmax(
        -0.5 * ((bearing[:, None] - centers[None, :]) / width).square(), dim=-1
    )
    score3 = values * goal_weights
    top2 = torch.topk(score3, k=2, dim=-1).values
    best_index = score3.argmax(dim=-1)
    side_exit = best_index != 1
    desired_sign = torch.where(
        best_index == 0,
        torch.ones_like(bearing),
        -torch.ones_like(bearing),
    )
    signed_wz = desired_sign * target_cmd3[:, 2]
    response_error = torch.clamp((0.10 - signed_wz) / 0.20, 0.0, 1.0)
    goal_distance = torch.linalg.vector_norm(goal, dim=-1)
    eligible = (
        teacher_valid.reshape(-1).bool()
        & (goal_freshness.reshape(-1) >= TEACHER_GOAL_FRESHNESS_MIN)
        & ~terminal.reshape(-1).bool()
        & side_exit
        & (values.max(dim=-1).values >= TEACHER_SAFE_MIN)
        & ((top2[:, 0] - top2[:, 1]) >= GOAL_SAFE_PREFERENCE_MARGIN)
        & (goal_distance >= NEAR_GOAL_CAPTURE_MAX_DISTANCE_M)
        & (target_cmd3[:, 0] >= 0.15)
        & (target_cmd3[:, 1].abs() <= 0.10)
        & (response_error > 0.0)
    )
    penalty = torch.where(
        eligible,
        YAW_EXIT_RESPONSE_RAW_FLOOR * response_error,
        torch.zeros_like(response_error),
    )
    return penalty, {
        "yaw_exit_response_eligible": eligible.float(),
        "yaw_exit_response_error": response_error,
        "yaw_exit_response_desired_sign": torch.where(
            side_exit, desired_sign, torch.zeros_like(desired_sign)
        ),
        "yaw_exit_response_signed_wz": signed_wz,
    }


def route_excess_penalty(
    path_length_m: torch.Tensor,
    start_goal_distance_m: torch.Tensor,
    end_goal_distance_m: torch.Tensor,
    terminal: torch.Tensor,
    *,
    recovery_active: torch.Tensor | None = None,
    dead_end: torch.Tensor | None = None,
    goal_freshness: torch.Tensor | None = None,
    contact_latch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply a small cost only outside recovery, dead-end and stale-goal windows."""
    path = torch.nan_to_num(path_length_m.float(), nan=0.0, posinf=0.0, neginf=0.0)
    progress = torch.clamp(
        start_goal_distance_m.float() - end_goal_distance_m.float(), min=0.0
    )
    excess = torch.clamp(path - progress, min=0.0, max=ROUTE_EXCESS_CAP_M)
    valid = (
        torch.isfinite(path_length_m)
        & torch.isfinite(start_goal_distance_m)
        & torch.isfinite(end_goal_distance_m)
        & ~terminal.reshape(-1).bool()
    )
    count = path.numel()

    def _optional_mask(value: torch.Tensor | None, name: str) -> torch.Tensor:
        if value is None:
            return torch.zeros(count, device=path.device, dtype=torch.bool)
        result = torch.as_tensor(value, device=path.device).reshape(-1)
        if result.numel() != count:
            raise ValueError(f"P4 route-excess {name} shape drift")
        return result.bool()

    if goal_freshness is not None:
        freshness = torch.as_tensor(
            goal_freshness, device=path.device, dtype=path.dtype
        ).reshape(-1)
        if freshness.numel() != count:
            raise ValueError("P4 route-excess goal_freshness shape drift")
        valid &= torch.isfinite(freshness) & (freshness > GOAL_FRESHNESS_FLOOR)
    suspended = (
        _optional_mask(recovery_active, "recovery_active")
        | _optional_mask(dead_end, "dead_end")
        | _optional_mask(contact_latch, "contact_latch")
    )
    eligible = valid & ~suspended
    penalty = torch.where(
        eligible,
        ROUTE_EXCESS_WEIGHT * excess,
        torch.zeros_like(excess),
    )
    efficiency = torch.where(
        path > 1.0e-6,
        torch.clamp(progress / path, 0.0, 1.0),
        torch.ones_like(path),
    )
    return penalty, {
        "route_path_length_m": path,
        "route_positive_progress_m": progress,
        "route_excess_distance_m": excess,
        "route_efficiency": efficiency,
        "route_excess_eligible": eligible.float(),
        "route_excess_suspended": suspended.float(),
    }


def segment_frontier_potential(
    spawn_segment: torch.Tensor,
    max_segment_before: torch.Tensor,
    current_segment: torch.Tensor,
    duration_frames: torch.Tensor,
    terminal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Potential shaping for first-time segment progress with terminal clawback."""
    spawn = torch.nan_to_num(spawn_segment.float(), nan=0.0).round().clamp(0, 4)
    before_max = torch.maximum(
        torch.nan_to_num(max_segment_before.float(), nan=0.0).round(), spawn
    ).clamp(0, 4)
    current = torch.nan_to_num(current_segment.float(), nan=0.0).round().clamp(0, 4)
    after_max = torch.maximum(before_max, current)
    phi_before = SEGMENT_FRONTIER_WEIGHT * (before_max - spawn)
    phi_after = SEGMENT_FRONTIER_WEIGHT * (after_max - spawn)
    settled_after = torch.where(
        terminal.reshape(-1).bool(), torch.zeros_like(phi_after), phi_after
    )
    discount = torch.pow(
        torch.full_like(phi_before, p2_contract.GAMMA_FRAME),
        duration_frames.float().reshape(-1).clamp(1.0, float(P4_NAV_PERIOD_FRAMES)),
    )
    return discount * settled_after - phi_before, phi_before, settled_after, after_max


def track_boundary_distance_m(
    root_x_m: torch.Tensor,
    current_segment: torch.Tensor,
    *,
    segment_length_m: float = FULL_TRACK_SEGMENT_LENGTH_M,
    segment_count: int = 5,
) -> torch.Tensor:
    """Distance to the closest Track segment boundary in the centered world frame."""
    root_x = torch.nan_to_num(root_x_m.float(), nan=0.0)
    segment = current_segment.float().round().clamp(0, segment_count - 1)
    offset = -0.5 * float(segment_count) * float(segment_length_m)
    local_x = root_x - (offset + segment * float(segment_length_m))
    return torch.minimum(local_x, float(segment_length_m) - local_x).clamp_min(0.0)


def open_straight_penalty(
    policy_cmd3: torch.Tensor,
    true_velocity3: torch.Tensor,
    clean_goal_xy_m: torch.Tensor,
    safe3: torch.Tensor,
    teacher_valid: torch.Tensor,
    current_segment: torch.Tensor,
    boundary_distance_m: torch.Tensor,
    terminal: torch.Tensor,
    *,
    junction: torch.Tensor,
    dead_end: torch.Tensor,
    contact_or_recovery: torch.Tensor,
    goal_freshness: torch.Tensor,
    yaw_cancellation_value: torch.Tensor,
    path_excess_m: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Small anti-S-turn cost restricted to verified open slope straightaways."""
    if policy_cmd3.ndim != 2 or policy_cmd3.shape[-1] != 3:
        raise ValueError("P4 open-straight penalty expects policy_cmd3=[N,3]")
    if true_velocity3.shape != policy_cmd3.shape:
        raise ValueError("P4 open-straight true velocity shape drift")
    n = policy_cmd3.shape[0]
    for name, value in (
        ("clean_goal_xy_m", clean_goal_xy_m),
        ("safe3", safe3),
    ):
        expected = (n, 2) if name == "clean_goal_xy_m" else (n, 3)
        if value.shape != expected:
            raise ValueError(f"P4 open-straight {name} shape drift")
    safe = torch.nan_to_num(safe3.to(policy_cmd3), nan=0.0).clamp(0.0, 1.0)
    best_safe = safe.max(dim=-1).values
    center_safe = safe[:, 1]
    goal = torch.nan_to_num(clean_goal_xy_m.to(policy_cmd3), nan=0.0)
    bearing = torch.atan2(goal[:, 1], goal[:, 0]).abs()
    segment = current_segment.reshape(-1).round().long()
    open_slope = (segment == 0) | (segment == 1)
    eligible = (
        teacher_valid.reshape(-1).bool()
        & open_slope
        & (center_safe >= OPEN_STRAIGHT_CENTER_SAFE_MIN)
        & ((best_safe - center_safe) <= OPEN_STRAIGHT_CENTER_BEST_MARGIN)
        & (bearing <= math.radians(OPEN_STRAIGHT_GOAL_BEARING_MAX_DEG))
        & (boundary_distance_m.reshape(-1) > OPEN_STRAIGHT_BOUNDARY_MARGIN_M)
        & ~junction.reshape(-1).bool()
        & ~dead_end.reshape(-1).bool()
        & ~contact_or_recovery.reshape(-1).bool()
        & (goal_freshness.reshape(-1) >= TEACHER_GOAL_FRESHNESS_MIN)
        & ~terminal.reshape(-1).bool()
    )
    lateral_error = torch.relu(
        true_velocity3[:, 1].abs() - OPEN_STRAIGHT_TRUE_VY_DEADBAND_M_S
    ) / max(P4_MAX_ABS_VY, 1.0e-6)
    lateral = OPEN_STRAIGHT_LATERAL_WEIGHT * lateral_error.square()
    s_turn = OPEN_STRAIGHT_S_TURN_WEIGHT * torch.clamp(
        yaw_cancellation_value.reshape(-1) - 0.15, min=0.0, max=1.0
    )
    # ``yaw_cancellation_value`` is already close to zero for legitimate
    # one-direction steering and high only when recent yaw repeatedly cancels
    # itself.  Scaling it down by the current |wz| would let large alternating
    # commands escape the anti-S-turn term.
    extra_path = OPEN_STRAIGHT_EXTRA_PATH_WEIGHT * torch.clamp(
        path_excess_m.reshape(-1) / 0.20, 0.0, 1.0
    )
    raw = torch.where(eligible, lateral + s_turn + extra_path, torch.zeros_like(lateral))
    total = raw.clamp_min(OPEN_STRAIGHT_TOTAL_FLOOR)
    return total, {
        "open_straight_eligible": eligible.float(),
        "open_straight_lateral_penalty": torch.where(eligible, lateral, torch.zeros_like(lateral)),
        "open_straight_s_turn_penalty": torch.where(eligible, s_turn, torch.zeros_like(s_turn)),
        "open_straight_extra_path_penalty": torch.where(eligible, extra_path, torch.zeros_like(extra_path)),
        "open_straight_goal_bearing_abs_rad": bearing,
        "open_straight_boundary_distance_m": boundary_distance_m.reshape(-1),
    }


def yaw_cancellation(x: torch.Tensor, *, dt_s: float = P4_NAV_DT_S) -> torch.Tensor:
    """Return cancellation in [0,1] for a [T,N] yaw-rate window."""
    if x.ndim != 2:
        raise ValueError("yaw cancellation expects [T,N]")
    absolute_integral = x.abs().sum(dim=0) * float(dt_s)
    signed_integral = (x.sum(dim=0) * float(dt_s)).abs()
    activity = torch.clamp(absolute_integral / 0.25, 0.0, 1.0)
    cancellation = 1.0 - signed_integral / (absolute_integral + 1.0e-6)
    return activity * torch.clamp(cancellation, 0.0, 1.0)


def proportional_negative_cap(
    predictive_raw: torch.Tensor,
    missed_raw: torch.Tensor,
    yaw_raw: torch.Tensor,
    *,
    goal_safe_raw: torch.Tensor | None = None,
    yaw_exit_raw: torch.Tensor | None = None,
    floor: float = SAFETY_GROUP_FLOOR,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Proportionally cap a group of non-positive reward terms."""
    terms = [predictive_raw, missed_raw, yaw_raw]
    if goal_safe_raw is not None:
        terms.append(goal_safe_raw)
    if yaw_exit_raw is not None:
        terms.append(yaw_exit_raw)
    if any(term.shape != predictive_raw.shape for term in terms):
        raise ValueError("P4 safety group tensor shape drift")
    raw_sum = sum(terms)
    magnitude = torch.clamp(-raw_sum, min=0.0)
    scale = torch.minimum(
        torch.ones_like(magnitude),
        torch.full_like(magnitude, abs(float(floor))) / magnitude.clamp_min(1.0e-9),
    )
    return tuple(term * scale for term in terms), scale


def push_phase_config(session_effective_seconds: float) -> dict[str, float | str | bool]:
    del session_effective_seconds
    return {
        "name": "p4recovery_no_push",
        "active": False,
        "max_velocity_xy_m_s": 0.0,
        "min_interval_s": 30.0,
        "max_interval_s": 45.0,
    }


def camera_mix(session_effective_seconds: float) -> dict[str, float]:
    seconds = max(0.0, float(session_effective_seconds))
    if seconds < 7_200.0:
        return {"nominal": 0.90, "light": 0.10, "delayed": 0.0, "severe": 0.0}
    return {"nominal": 0.80, "light": 0.20, "delayed": 0.0, "severe": 0.0}


def safe_direction_weight(session_effective_seconds: float) -> float:
    """Keep the verified parent safety weight fixed during credit repair."""
    del session_effective_seconds
    return 0.012


def maze_missed_safe_direction_penalty(
    safe3: torch.Tensor,
    target_cmd3: torch.Tensor,
    teacher_valid: torch.Tensor,
    session_effective_seconds: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize a clear missed alternative with event-normalized severity."""
    weights = p2_contract.command_direction_weights(target_cmd3)
    selected_safe = (weights * safe3).sum(dim=-1)
    best_safe = safe3.max(dim=-1).values
    chosen_risk = torch.clamp(1.0 - selected_safe, 0.0, 1.0)
    safe_gap = torch.relu(
        best_safe - selected_safe - p2_contract.SAFE_DIRECTION_GAP_MARGIN
    )
    normalized_gap = torch.clamp(safe_gap / SAFE_DIRECTION_GAP_SCALE, 0.0, 1.0)
    speed_scale = target_cmd3.new_tensor(
        (1.0, 1.0, p2_contract.CRAWL_BODY_RADIUS_M)
    )
    moving = torch.linalg.vector_norm(target_cmd3 * speed_scale, dim=-1) > 0.05
    active = (
        teacher_valid.reshape(-1).bool()
        & moving
        & (best_safe >= p2_contract.SAFE_DIRECTION_MIN_BEST_SAFE)
        & (safe_gap > 0.0)
    )
    weight = safe_direction_weight(session_effective_seconds)
    severity = chosen_risk * normalized_gap
    penalty = torch.where(
        active,
        torch.clamp(-weight * severity, min=-weight, max=0.0),
        torch.zeros_like(severity),
    )
    return penalty, {
        "missed_safe_event_active": active.float(),
        "missed_safe_event_severity": severity,
        "missed_safe_weight": severity.new_full(severity.shape, weight),
    }


def training_schedule(
    session_effective_seconds: float,
    *,
    branch: str = "actor_attack",
) -> dict[str, float | str | bool]:
    seconds = max(0.0, float(session_effective_seconds))
    branch = str(branch or "actor_attack")
    if branch == "auto":
        branch = "actor_attack"
    if branch not in {
        "actor_attack", "visual_recovery", "credit_repair", "closed_loop_v3"
    }:
        branch = "actor_attack"
    if branch == "closed_loop_v3":
        common = {
            "training_branch": branch,
            "reward_multiplier": 1.0,
            "goal_fault_multiplier": 0.0,
            "camera_aux_ratio": 0.0,
            "stuck_gradient_target_ratio": 0.0,
            "mirror_sequence_share": 0.0,
            "cruise_multiplier": 0.0,
            "teacher_gradient_hard_cap": 0.03,
            "auxiliary_gradient_hard_cap": 0.03,
            "mirror_gradient_hard_cap": 0.0,
            "navigation_multiplier": 0.0,
            "safety_head_multiplier": 0.0,
            "adapter_multiplier": 0.0,
            "stuck_head_multiplier": 0.0,
            "mirror_gradient_target_ratio": 0.0,
        }
        if seconds < 1_800.0:
            return {
                **common,
                "phase": "loopwarm",
                "actor_multiplier": 0.0,
                "actor_lr": 3.0e-5,
                "critic_lr": 6.0e-5,
                "teacher_gradient_target_ratio": 0.0,
                "entropy_coefficient": 0.004,
            }
        if seconds < 7_200.0:
            return {
                **common,
                "phase": "loopadapt",
                "actor_multiplier": 1.0,
                "actor_lr": 3.0e-5,
                "critic_lr": 6.0e-5,
                "teacher_gradient_target_ratio": (
                    0.010 * (seconds - 1_800.0) / 5_400.0
                ),
                "entropy_coefficient": 0.004,
            }
        if seconds < 21_600.0:
            return {
                **common,
                "phase": "looptrain",
                "actor_multiplier": 1.0,
                "actor_lr": 5.0e-5,
                "critic_lr": 5.0e-5,
                "teacher_gradient_target_ratio": 0.020,
                "entropy_coefficient": 0.003,
            }
        return {
            **common,
            "phase": "loopstable",
            "actor_multiplier": 1.0,
            "actor_lr": 2.5e-5,
            "critic_lr": 3.0e-5,
            "teacher_gradient_target_ratio": 0.010,
            "entropy_coefficient": 0.002,
        }
    if branch == "credit_repair":
        common = {
            "training_branch": branch,
            "reward_multiplier": 1.0,
            "goal_fault_multiplier": 0.0,
            "camera_aux_ratio": 0.0,
            "stuck_gradient_target_ratio": 0.005,
            "mirror_sequence_share": 0.0,
            "cruise_multiplier": 1.0,
            "teacher_gradient_hard_cap": 0.05,
            "auxiliary_gradient_hard_cap": 0.05,
            "mirror_gradient_hard_cap": 0.0,
            "navigation_multiplier": 0.0,
            "safety_head_multiplier": 0.0,
            "adapter_multiplier": 0.0,
            "stuck_head_multiplier": 1.0,
            "mirror_gradient_target_ratio": 0.0,
            "entropy_coefficient": 0.005,
        }
        if seconds < 600.0:
            return {
                **common,
                "phase": "creditwarm",
                "actor_multiplier": 0.0,
                "critic_lr": 1.2e-4,
                "teacher_gradient_target_ratio": 0.0,
            }
        if seconds < 1_800.0:
            return {
                **common,
                "phase": "creditadapt",
                "actor_multiplier": 1.0,
                "actor_lr": 7.5e-5,
                "critic_lr": 1.2e-4,
                "teacher_gradient_target_ratio": (
                    0.025 * (seconds - 600.0) / 1_200.0
                ),
            }
        if seconds < 6_300.0:
            return {
                **common,
                "phase": "credittrain",
                "actor_multiplier": 1.0,
                "actor_lr": 1.0e-4,
                "critic_lr": 1.0e-4,
                "teacher_gradient_target_ratio": 0.035,
            }
        return {
            **common,
            "phase": "creditfinal",
            "actor_multiplier": 1.0,
            "actor_lr": 5.0e-5,
            "critic_lr": 6.0e-5,
            "teacher_gradient_target_ratio": 0.020,
        }
    common = {
        "training_branch": branch,
        "reward_multiplier": 1.0,
        "goal_fault_multiplier": min(1.0, max(0.0, (seconds - 1_800.0) / 5_400.0)),
        "camera_aux_ratio": 0.01,
        "stuck_gradient_target_ratio": 0.005,
        "mirror_sequence_share": 0.10,
        "cruise_multiplier": 1.0,
        "teacher_gradient_hard_cap": 0.03,
        "auxiliary_gradient_hard_cap": 0.05,
        "mirror_gradient_hard_cap": 0.01,
    }
    if seconds < 1_800.0:
        teacher_ratio = 0.015 * seconds / 1_800.0
        return {
            **common,
            "phase": "fullwarm",
            "navigation_multiplier": 0.15,
            "actor_multiplier": 0.20,
            "critic_multiplier": 0.60,
            "safety_head_multiplier": 1.00,
            "stuck_head_multiplier": 1.00,
            "adapter_multiplier": 0.50,
            "teacher_gradient_target_ratio": teacher_ratio,
            "mirror_gradient_target_ratio": 0.005,
            "entropy_coefficient": 0.006,
        }
    if seconds < 7_200.0:
        entropy = 0.006 + (seconds - 1_800.0) / 5_400.0 * (0.005 - 0.006)
        return {
            **common,
            "phase": "fulladapt",
            "navigation_multiplier": 0.35,
            "actor_multiplier": 0.55,
            "critic_multiplier": 0.80,
            "safety_head_multiplier": 1.00,
            "stuck_head_multiplier": 1.00,
            "adapter_multiplier": 0.50,
            "teacher_gradient_target_ratio": 0.0225,
            "mirror_gradient_target_ratio": 0.005,
            "entropy_coefficient": entropy,
        }
    if seconds < 21_600.0:
        return {
            **common,
            "phase": "fulltrain",
            "navigation_multiplier": 0.30,
            "actor_multiplier": 0.45,
            "critic_multiplier": 0.60,
            "safety_head_multiplier": 0.75,
            "stuck_head_multiplier": 0.75,
            "adapter_multiplier": 0.50,
            "teacher_gradient_target_ratio": 0.020,
            "mirror_gradient_target_ratio": 0.005,
            "entropy_coefficient": 0.005,
        }
    return {
        **common,
        "phase": "fullstabilize",
        "navigation_multiplier": 0.15,
        "actor_multiplier": 0.20,
        "critic_multiplier": 0.35,
        "safety_head_multiplier": 0.50,
        "stuck_head_multiplier": 0.50,
        "adapter_multiplier": 0.25,
        "teacher_gradient_target_ratio": 0.010,
        "mirror_gradient_target_ratio": 0.005,
        "entropy_coefficient": 0.004,
    }


def _normalized_training_profile(training_profile: str) -> str:
    profile = str(training_profile).strip().lower()
    if profile not in {"maze_credit_repair", "maze_closed_loop_v3", "full_track"}:
        raise ValueError(f"unsupported P4 training profile {training_profile!r}")
    return profile


def _is_maze_profile(profile: str) -> bool:
    return profile in {"maze_credit_repair", "maze_closed_loop_v3"}


def command_contract(
    training_profile: str = "maze_credit_repair",
) -> dict[str, Any]:
    profile = _normalized_training_profile(training_profile)
    closed_loop = profile == "maze_closed_loop_v3"
    slew_rate = P4_SLEW_RATE if closed_loop else LEGACY_P4_SLEW_RATE
    slew_release_rate = (
        P4_SLEW_RELEASE_RATE if closed_loop else LEGACY_P4_SLEW_RELEASE_RATE
    )
    limiter_threshold = (
        TRANSLATION_LIMITER_RISK_THRESHOLD
        if closed_loop
        else LEGACY_TRANSLATION_LIMITER_RISK_THRESHOLD
    )
    limiter_floor = (
        TRANSLATION_LIMITER_ALPHA_FLOOR
        if closed_loop
        else LEGACY_TRANSLATION_LIMITER_ALPHA_FLOOR
    )
    return {
        "version": (
            "p4_maze_closed_loop_command_v3"
            if closed_loop
            else (
                "p4_maze_credit_repair_command_v1"
                if profile == "maze_credit_repair"
                else FULL_TRACK_COMMAND_CONTRACT_VERSION
            )
        ),
        "mapper_version": ACTION_MAPPER_VERSION,
        "legacy_mapper_version": LEGACY_ACTION_MAPPER_VERSION,
        "normalized_action": "unchanged_tanh_gaussian_v2",
        "mapped_ranges": {"vx": [0.0, 1.0], "vy": [-0.30, 0.30], "wz": [-0.90, 0.90]},
        "policy_target_vx": [0.0, 1.0],
        "limited_target": (
            "stale_goal_cap_with_translation_limiter_shadow_only"
            if closed_loop
            else "stale_goal_cap_then_translation_vector_limiter_then_near_goal_capture"
        ),
        "stale_goal_wait": {
            "no_estimate": {
                "vx": 0.0,
                "vy": 0.0,
                "max_abs_wz": STALE_GOAL_WAIT_MAX_ABS_WZ,
            },
            "stale_estimate": {
                "max_vx": 0.20,
                "max_abs_vy": STALE_GOAL_WAIT_MAX_ABS_VY,
                "max_abs_wz": STALE_GOAL_WAIT_MAX_ABS_WZ,
                "freshness_floor": GOAL_FRESHNESS_FLOOR,
            },
        },
        "speed_tier_contract": "removed",
        "soft_cruise": {
            "preferred_vx": [SOFT_CRUISE_MIN_VX, SOFT_CRUISE_MAX_VX],
            "low_weight": SOFT_CRUISE_LOW_WEIGHT,
            "high_weight": SOFT_CRUISE_HIGH_WEIGHT,
            "clear_factor": "teacher_valid*goal_freshness*center_safe*center_safe/best_safe",
            **({"status": "disabled"} if closed_loop else {}),
        },
        "slew_rate": list(slew_rate),
        "slew_release_rate": list(slew_release_rate),
        "nav_period_frames": P4_NAV_PERIOD_FRAMES,
        "nav_frequency_hz": 1.0 / P4_NAV_DT_S,
        "slew_semantics": p2_contract.command_contract()["slew_semantics"],
        "translation_vector_limiter": {
            "version": (
                TRANSLATION_VECTOR_LIMITER_CONTRACT_VERSION
                if closed_loop
                else LEGACY_TRANSLATION_VECTOR_LIMITER_CONTRACT_VERSION
            ),
            "source": "deployment_available_predictive_depth_risk",
            "alpha_raw": (
                f"risk<={limiter_threshold:.2f}:1; "
                f"risk=1:{limiter_floor:.2f}; linear between"
            ),
            "risk_threshold": limiter_threshold,
            "alpha_floor": limiter_floor,
            "release_per_10hz_tick": TRANSLATION_LIMITER_RELEASE_PER_TICK,
            "state": "live_only_reset_to_one_not_checkpointed",
            "axes": "scale_vx_vy_preserve_wz",
            **({"status": "shadow_only"} if closed_loop else {}),
        },
        "near_goal_capture": {
            "version": (
                NEAR_GOAL_CAPTURE_CONTRACT_VERSION
                if closed_loop
                else LEGACY_NEAR_GOAL_CAPTURE_CONTRACT_VERSION
            ),
            "distance_m": [NEAR_GOAL_CAPTURE_MIN_DISTANCE_M, NEAR_GOAL_CAPTURE_MAX_DISTANCE_M],
            "goal_freshness_min": NEAR_GOAL_CAPTURE_FRESHNESS_MIN,
            "policy_goal_cosine_min": NEAR_GOAL_CAPTURE_GOAL_COSINE_MIN,
            **(
                {"status": "shadow_diagnostic_only_no_command_rewrite"}
                if closed_loop
                else {"axes": "translation_only_preserve_yaw"}
            ),
        },
    }


def reward_contract(
    stuck_reset: dict[str, Any] | None = None,
    training_profile: str = "maze_credit_repair",
) -> dict[str, Any]:
    profile = _normalized_training_profile(training_profile)
    closed_loop = profile == "maze_closed_loop_v3"
    stuck = normalize_stuck_reset_contract(stuck_reset)
    stuck_term = {
        "mode": stuck["mode"],
        "terminal_penalty": stuck["terminal_penalty"],
        "semantics": (
            "shadow_only_no_reason4_terminal"
            if stuck["mode"] == "shadow"
            else "active_reason4_terminal"
        ),
    }
    if stuck["schedule_enabled"]:
        stuck_term["schedule"] = {
            "shadow_until_s": stuck["activation_delay_s"],
            "initial_confirmation_s": stuck["initial_confirmation_s"],
            "tighten_after_s": stuck["tighten_after_s"],
            "tight_confirmation_s": stuck["confirmation_s"],
        }
    contract = {
        "version": (
            "p4_maze_closed_loop_reward_v3_single_signal"
            if closed_loop
            else (
                "p4_maze_credit_repair_reward_v1"
                if profile == "maze_credit_repair"
                else FULL_TRACK_REWARD_CONTRACT_VERSION
            )
        ),
        "inherits": p2_contract.reward_contract()["version"],
        "tick_time_scaling": {
            "reference_period_frames": p2_contract.NAV_PERIOD_FRAMES,
            "runtime_period_frames": P4_NAV_PERIOD_FRAMES,
            "continuous_terms": "duration_frames/reference_period_frames",
            "distance_terms": "per_meter_unscaled",
            "command_rate": "per_policy_decision_unscaled",
            "body_collision": "per_10hz_tick_unscaled_for_stronger_enforcement",
        },
        "new_terms": {
            "predictive_collision_raw_floor": PREDICTIVE_RAW_FLOOR,
            "predictive_collision_scale": PREDICTIVE_COLLISION_SCALE,
            "missed_safe_direction_raw_floor": MISSED_SAFE_RAW_FLOOR,
            "missed_safe_direction_gap_scale": SAFE_DIRECTION_GAP_SCALE,
            "frontier_stagnation": "shadow_only_zero_ppo_weight",
            "yaw_exec_weight": (
                YAW_EXEC_WEIGHT if closed_loop else LEGACY_YAW_EXEC_WEIGHT
            ),
            "yaw_true_weight": (
                YAW_TRUE_WEIGHT if closed_loop else LEGACY_YAW_TRUE_WEIGHT
            ),
            "yaw_total_floor": (
                YAW_TOTAL_FLOOR if closed_loop else LEGACY_YAW_TOTAL_FLOOR
            ),
            "yaw_exit_response_raw_floor": YAW_EXIT_RESPONSE_RAW_FLOOR,
            "safety_group_floor": (
                SAFETY_GROUP_FLOOR if closed_loop else LEGACY_SAFETY_GROUP_FLOOR
            ),
            "safety_group_terms": (
                ["predictive_collision"]
                if closed_loop
                else [
                    "predictive_collision",
                    "missed_safe_direction",
                    "yaw_cancellation",
                    "yaw_exit_response",
                    "goal_safe_preference",
                ]
            ),
            **(
                {"yaw_cancellation": "shadow_diagnostic_only_zero_ppo_weight"}
                if closed_loop
                else {}
            ),
            "cap_semantics": "proportional_no_hidden_adjustment_5hz_reference",
            "confirmed_wall_stuck_reset": stuck_term,
            "success_impulse": SUCCESS_IMPULSE,
            **({"failure_impulse": FAILURE_IMPULSE} if closed_loop else {}),
            "timeout_impulse": (
                TIMEOUT_IMPULSE if closed_loop else LEGACY_TIMEOUT_IMPULSE
            ),
            "sustained_wall_stuck": {
                "grace_s": STUCK_SUSTAINED_GRACE_S,
                "full_penalty_s": STUCK_SUSTAINED_FULL_S,
                "base": STUCK_SUSTAINED_BASE,
                "floor": (
                    STUCK_SUSTAINED_FLOOR
                    if closed_loop
                    else LEGACY_STUCK_SUSTAINED_FLOOR
                ),
            },
            "closed_loop_collision": {
                "onset_base": P4_BODY_COLLISION_ONSET_BASE,
                "onset_severity": P4_BODY_COLLISION_ONSET_SEVERITY,
                "persistent": P4_BODY_COLLISION_PERSISTENT,
            },
            "recovery_translation_teacher": {
                "status": (
                    "active_training_only_safe5_lateral_egress"
                    if closed_loop
                    else "disabled"
                ),
                "max_vx_m_s": TEACHER_RECOVERY_MAX_VX,
                "min_abs_vy_m_s": TEACHER_RECOVERY_MIN_ABS_VY,
                "side_clearance_margin": TEACHER_RECOVERY_SIDE_MARGIN,
            },
            "goal_safe_preference": {
                "weight": GOAL_SAFE_PREFERENCE_WEIGHT,
                "margin": GOAL_SAFE_PREFERENCE_MARGIN,
                "scale": GOAL_SAFE_PREFERENCE_SCALE,
                "semantics": "goal_preference_only_among_privileged_safe_sectors",
            },
            "route_excess": {
                "weight_per_m": ROUTE_EXCESS_WEIGHT,
                "per_tick_cap_m": ROUTE_EXCESS_CAP_M,
                "status": "disabled_global_term",
            },
            "maze_new_best_credit": {
                "weight_per_m": (
                    MAZE_NEW_BEST_WEIGHT_PER_M
                    if closed_loop
                    else LEGACY_MAZE_NEW_BEST_WEIGHT_PER_M
                ),
                "episode_cap": (
                    MAZE_NEW_BEST_EPISODE_CAP
                    if closed_loop
                    else LEGACY_MAZE_NEW_BEST_EPISODE_CAP
                ),
                "terminal": "retain_earned_credit_no_clawback",
            },
            "open_straight": {
                "segments": ["slope", "slope_inv"],
                "lateral_weight": OPEN_STRAIGHT_LATERAL_WEIGHT,
                "s_turn_weight": OPEN_STRAIGHT_S_TURN_WEIGHT,
                "extra_path_weight": OPEN_STRAIGHT_EXTRA_PATH_WEIGHT,
                "total_floor": OPEN_STRAIGHT_TOTAL_FLOOR,
                "requires": "teacher_open_center_clean_goal_near_axis_boundary_margin",
                "zero_on": ["stairs", "maze", "junction", "dead_end", "contact", "recovery", "stale", "terminal"],
            },
            "soft_cruise": {
                "preferred_vx": [SOFT_CRUISE_MIN_VX, SOFT_CRUISE_MAX_VX],
                "low_weight": SOFT_CRUISE_LOW_WEIGHT,
                "high_weight": SOFT_CRUISE_HIGH_WEIGHT,
            },
        },
        "goal_truth_consumers": ["critic", "reward", "terminal", "scorer"],
        "goal_belief_consumers": ["actor", "speed_cap"],
    }
    if closed_loop:
        contract["reward_allowlist"] = [
            "frame_safety",
            "frontier_shaping",
            "success",
            "failure",
            "timeout",
            "time",
            "crawl",
            "command_rate",
            "tracking",
            "body_collision",
            "predictive_collision_risk",
            "stuck_reset",
            "stuck_sustained",
        ]
        contract["nonterminal_positive_shaping_cap"] = MAZE_NEW_BEST_EPISODE_CAP
        contract["new_terms"]["missed_safe_direction"] = "shadow_only_zero_ppo_weight"
        contract["new_terms"]["goal_safe_preference"] = "shadow_only_zero_ppo_weight"
        contract["new_terms"]["yaw_exit_response"] = "shadow_only_zero_ppo_weight"
        contract["new_terms"]["soft_cruise"]["status"] = "disabled"
    if profile == "full_track":
        contract["new_terms"]["maze_new_best_credit"] = {
            "status": "disabled_outside_maze_credit_repair",
        }
        contract["new_terms"]["segment_frontier"] = {
            "weight": SEGMENT_FRONTIER_WEIGHT,
            "semantics": "first_segment_progress_terminal_clawed_potential",
        }
        contract["new_terms"]["open_straight"]["status"] = (
            "enabled_on_teacher_confirmed_slope_and_slope_inv_only"
        )
    return contract


def training_contract(
    stuck_reset: dict[str, Any] | None = None,
    training_profile: str = "maze_credit_repair",
) -> dict[str, Any]:
    profile = _normalized_training_profile(training_profile)
    maze_profile = _is_maze_profile(profile)
    closed_loop = profile == "maze_closed_loop_v3"
    legacy_maze = profile == "maze_credit_repair"
    stuck = normalize_stuck_reset_contract(stuck_reset)
    serialized_stuck = dict(stuck)
    if not closed_loop:
        # Keep the byte-level shape of the historical v1/v2 contracts.  The
        # schedule flag belongs only to the new v3 contract; adding it to an
        # unchanged legacy version would make valid exact-resume packages look
        # like contract drift.
        for name in (
            "schedule_enabled",
            "initial_confirmation_s",
            "activation_delay_s",
            "tighten_after_s",
            "resume_offset_s",
        ):
            serialized_stuck.pop(name, None)
    return {
        "version": (
            MAZE_CLOSED_LOOP_CHECKPOINT_VERSION
            if closed_loop
            else (
            CHECKPOINT_CONTRACT_VERSION
                if profile == "maze_credit_repair"
                else FULL_TRACK_CHECKPOINT_CONTRACT_VERSION
            )
        ),
        "training_profile": profile,
        "run_name": (
            RUN_NAME
            if closed_loop
            else ("p4maze2h-credit-repair" if legacy_maze else "p4full8h-r2")
        ),
        "training_hours": (
            TRAINING_HOURS
            if closed_loop
            else (LEGACY_MAZE_TRAINING_HOURS if legacy_maze else 8.0)
        ),
        "target_effective_seconds": (
            int(TARGET_EFFECTIVE_SECONDS)
            if closed_loop
            else (
                int(LEGACY_MAZE_TARGET_EFFECTIVE_SECONDS)
                if legacy_maze
                else 28_800
            )
        ),
        "diagnostic_seconds": int(DIAGNOSTIC_SECONDS),
        "required_platform_wall_seconds": (
            int(PLATFORM_WALL_SECONDS)
            if closed_loop
            else (8_100 if legacy_maze else 29_700)
        ),
        "required_platform_wall_hours": (
            PLATFORM_WALL_HOURS
            if closed_loop
            else (2.25 if legacy_maze else 8.25)
        ),
        "clock_semantics": {
            "diagnostic": "wall_seconds_before_training_not_counted_in_session",
            "session_effective_seconds": "gradient_training_seconds_only",
            "session_wall_seconds": (
                "task_wall_including_rollout_update_checkpoint_monitor_and_logging"
            ),
            "platform_task": (
                "must_cover diagnostic plus target effective seconds plus bounded "
                "rollout/save shutdown margin"
            ),
            "platform_wall_margin_seconds": int(PLATFORM_WALL_MARGIN_SECONDS),
        },
        "schedule_boundaries_seconds": (
            list(SCHEDULE_BOUNDARIES_SECONDS)
            if closed_loop
            else (
                list(LEGACY_MAZE_SCHEDULE_BOUNDARIES_SECONDS)
                if legacy_maze
                else [1_800.0, 7_200.0, 21_600.0, 28_800.0]
            )
        ),
        "safety_reward_ramp": {
            "version": (
                MAZE_CLOSED_LOOP_SAFETY_REWARD_VERSION
                if closed_loop
                else (
                    SAFETY_REWARD_RAMP_VERSION
                    if maze_profile
                    else "p4_full_track_safety_group_v2"
                )
            ),
            "segments": [
                {
                    "seconds": [
                        0,
                        28_800 if closed_loop or not legacy_maze else 7_200,
                    ],
                    "weight": (
                        [0.0, 0.0]
                        if closed_loop
                        else [0.012, 0.012]
                    ),
                },
            ],
        },
        "goal_fault_ramp": {
            "semantics": (
                "disabled_for_closed_loop_maze_run"
                if closed_loop
                else (
                    "disabled_for_credit_assignment_run"
                    if legacy_maze
                    else "ramp_after_30m_to_full_at_2h"
                )
            ),
            "segments": (
                [{"seconds": [0, 28_800], "multiplier": [0.0, 0.0]}]
                if closed_loop
                else (
                    [{"seconds": [0, 7_200], "multiplier": [0.0, 0.0]}]
                    if legacy_maze
                    else [
                        {"seconds": [0, 1_800], "multiplier": [0.0, 0.0]},
                        {"seconds": [1_800, 7_200], "multiplier": [0.0, 1.0]},
                        {"seconds": [7_200, 28_800], "multiplier": [1.0, 1.0]},
                    ]
                )
            ),
        },
        "rollout_nav_ticks": 32,
        "tbptt_nav_ticks": 16,
        "nav_period_frames": P4_NAV_PERIOD_FRAMES,
        "nav_frequency_hz": 1.0 / P4_NAV_DT_S,
        "frozen_low_level": ["cnn", "lstm", "actor", "std", "critic"],
        "trainable": (
            ["high_actor_lstm", "high_actor_head", "high_critic"]
            if closed_loop
            else (
                ["high_actor_lstm", "high_actor_head", "high_critic", "stuck_head"]
                if legacy_maze
                else [
                    "navigation_encoder", "high_actor_lstm", "high_actor_head",
                    "high_critic", "safety_head", "stuck_head", "response_adapter",
                ]
            )
        ),
        "frozen_high_level": (
            ["navigation_encoder", "safety_head", "stuck_head", "response_adapter"]
            if closed_loop
            else (
                ["navigation_encoder", "safety_head", "response_adapter"]
                if legacy_maze
                else []
            )
        ),
        "goal_belief_version": GOAL_BELIEF_VERSION,
        "camera_contract_version": CAMERA_CONTRACT_VERSION,
        "worker_wire_version": WORKER_WIRE_VERSION,
        "worker_wire_dim": P4_PRIVILEGED_WIRE_DIM,
        "stuck_reset_contract_version": (
            STUCK_RESET_CONTRACT_VERSION
            if closed_loop
            else LEGACY_STUCK_RESET_CONTRACT_VERSION
        ),
        "stuck_reset": serialized_stuck,
        "adapter_record_contract_version": ADAPTER_RECORD_CONTRACT_VERSION,
        "actor_mean_guidance": {
            "version": (
                ACTOR_MEAN_GUIDANCE_CONTRACT_VERSION
                if closed_loop
                else LEGACY_ACTOR_MEAN_GUIDANCE_CONTRACT_VERSION
            ),
            "minimum_valid_steps": TEACHER_MIN_VALID_STEPS,
            **(
                {
                    "teacher_directions": [
                        "far_left", "left", "center", "right", "far_right"
                    ],
                    "weights": {"direction": 0.55, "speed": 0.10, "yaw": 0.35},
                    "goal_safe_tie_margin": TEACHER_GOAL_SAFE_TIE_MARGIN,
                }
                if closed_loop
                else {"weights": {"direction": 0.45, "speed": 0.20, "yaw": 0.35}}
            ),
            "gradient_target_ratio": (
                [0.0, 0.020] if closed_loop else (
                    [0.0, 0.025] if maze_profile else [0.0, 0.0225]
                )
            ),
            "gradient_hard_cap": 0.03 if closed_loop else (
                0.05 if maze_profile else 0.03
            ),
            "nav_feat_detached": True,
            "rollout_time_labels_required": True,
        },
        "mirror": {
            "requested_eligible_sequence_share": 0.0 if maze_profile else 0.10,
            "eligibility": "episode_start_zero_hidden_no_reset_crossing",
            "gradient_target_ratio": 0.0 if maze_profile else 0.005,
            "gradient_hard_cap": 0.0 if maze_profile else 0.01,
        },
        "stuck_aux": {
            "version": STUCK_AUX_CONTRACT_VERSION,
            "balanced_positive_negative": True,
            "classifier": "actor_lstm_to_stuck_logit_training_only",
            **({"status": "frozen_diagnostic_only"} if closed_loop else {}),
        },
        "closed_loop_guards": {
            "post_actor_command_rewrite": (
                "none_translation_limiter_shadow_only"
                if closed_loop
                else False
            ),
            "near_goal_capture": "shadow_diagnostic_only",
            "translation_limiter": {
                "risk_threshold": TRANSLATION_LIMITER_RISK_THRESHOLD,
                "minimum_xy_scale": TRANSLATION_LIMITER_ALPHA_FLOOR,
                "preserves_wz": True,
                "status": "shadow_only" if closed_loop else "active",
            },
            "global_yaw_cancellation_reward": False,
            "teacher_is_training_only": True,
        },
        "maze_only": maze_profile,
        "track_segment_labels": (
            ["maze"] if maze_profile else list(FULL_TRACK_SEGMENT_LABELS)
        ),
        "track_length": 1 if maze_profile else 5,
        "episode_length_s": 120.0 if closed_loop or not maze_profile else 75.0,
        "spawn": {
            "enabled": not maze_profile,
            "semantics": (
                "platform_default_single_maze_spawn"
                if maze_profile
                else "full_track_static_quota_with_runtime_safe_hard_validation"
            ),
        },
        "goal_jump": {
            "enabled": not maze_profile,
            "semantics": (
                "base_feedback_only_no_extra_goal_fault_course"
                if maze_profile
                else "distance_scaled_zero_mean_elliptical_actor_belief_fault"
            ),
        },
        "soft_cruise": command_contract(profile)["soft_cruise"],
        "exact_resume": (
                "p4_maze_closed_loop_v3_only"
            if closed_loop
            else (
                "p4_maze_credit_repair_v1_only"
                if legacy_maze
                else "p4_full_track_v2_only_worker_spawn_rng_reseeded"
            )
        ),
    }


def contract_metadata(
    stuck_reset: dict[str, Any] | None = None,
    training_profile: str = "maze_credit_repair",
) -> dict[str, Any]:
    profile = _normalized_training_profile(training_profile)
    command = command_contract(profile)
    reward = reward_contract(stuck_reset, profile)
    training = training_contract(stuck_reset, profile)
    return {
        "stage": STAGE_NAME,
        "actor_input_dim": p2_contract.ACTOR_INPUT_DIM,
        "critic_input_dim": p2_contract.CRITIC_INPUT_DIM,
        "command": command,
        "command_digest": stable_digest(command),
        "reward": reward,
        "reward_digest": stable_digest(reward),
        "training": training,
        "training_digest": stable_digest(training),
    }


def adapter_record_contract(
    *, low_level_digest: str, feedback_digest: str
) -> dict[str, Any]:
    response_capability = list(p2_contract.RESPONSE_CAPABILITY_PROFILE15)
    response_capability[3] = P4_MAX_VX
    response_capability[4] = P4_MAX_ABS_WZ
    response_capability[7] = P4_MAX_ABS_VY
    return {
        "version": ADAPTER_RECORD_CONTRACT_VERSION,
        "schema": "response_aux30_axis_specific_stuck_labels_v3",
        "low_level_digest": str(low_level_digest),
        "feedback_digest": str(feedback_digest),
        "capability_digest": stable_digest(response_capability),
        "response_capability_profile15": response_capability,
        "action_mapper": ACTION_MAPPER_VERSION,
        "observation_layout": "response_obs45_profile16",
        "label_layout": "velocity_horizons_0p2_0p6_1p0_pose_stuck",
    }
