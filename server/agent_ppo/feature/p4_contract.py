"""Compatibility facade for the canonical :mod:`agent_ppo.p4` APIs."""

from __future__ import annotations

from typing import Any

from agent_ppo.feature import p2_contract
from agent_ppo.p4 import constants as _constants
from agent_ppo.p4 import contracts as _canonical_contracts
from agent_ppo.p4.runtime import (
    map_normalized_action, map_normalized_action_legacy, instant_parent_anchor_reachable, stale_goal_cap, effective_speed_cap, translation_vector_limiter, near_goal_capture,
)
from agent_ppo.p4.primitives import (
    adapter_compatibility_metric_name, teacher_guidance_mask, privileged_safe_directions5, teacher_guidance_loss, instant_r4_teacher_masks, instant_r4_teacher_guidance_loss,
)
from agent_ppo.p4.rewards import (
    soft_cruise_penalty, sustained_wall_stuck_penalty, maze_new_best_credit, goal_safe_direction_penalty, yaw_exit_response_penalty, route_excess_penalty, segment_frontier_potential, track_boundary_distance_m, open_straight_penalty, yaw_cancellation, proportional_negative_cap, maze_missed_safe_direction_penalty,
)
from agent_ppo.p4.diagnostics import (
    safety_scene_diagnostics,
)
from agent_ppo.p4.training import (
    push_phase_config, camera_mix, safe_direction_weight, training_schedule,
)
from agent_ppo.p4.contracts import (
    normalize_stuck_reset_contract, stable_digest, adapter_record_contract,
)
from agent_ppo.p4.profiles import LEGACY_COMPATIBILITY_DEFAULT

for _name in _constants.__all__:
    globals()[_name] = getattr(_constants, _name)
del _name

MONITOR_REQUIRED_METRICS = (
    'session_wall_seconds', 'session_effective_seconds', 'teacher_guidance_loss', 'teacher_direction_loss',
    'teacher_speed_loss', 'teacher_yaw_loss', 'teacher_edge_loss', 'teacher_edge_active_share',
    'teacher_recovery_loss', 'teacher_recovery_active_share', 'teacher_stale_goal_loss', 'teacher_stale_goal_active_share',
    'teacher_near_goal_loss', 'teacher_near_goal_active_share', 'teacher_guidance_valid_steps', 'teacher_guidance_gradient_ratio',
    'mirror_aux_gradient_ratio', 'mirror_aux_sequence_share', 'mirror_aux_eligible_sequence_count', 'mirror_aux_scheduled_sequence_share',
    'stuck_aux_gradient_ratio', 'stuck_aux_loss', 'stuck_aux_valid_steps', 'auxiliary_gradient_ratio',
    'safety_hard_positive_share', 'actor_stuck_pr_auc', 'actor_stuck_precision', 'actor_stuck_recall',
    'actor_stuck_f1', 'actor_stuck_threshold', 'actor_stuck_positive_share', 'parent_anchor_valid_share',
    'teacher_risk_left', 'teacher_risk_center', 'teacher_risk_right', 'safe_alternative_available',
    'selected_safest_direction', 'reward_command_rate_vx', 'reward_command_rate_vy', 'reward_command_rate_wz',
    'reward_tracking_vx', 'reward_tracking_vy', 'reward_tracking_wz', 'wall_stuck_candidate_with_motion_intent_share',
    'wall_stuck_candidate_without_motion_intent_share', 'reward_goal_safe_raw', 'reward_yaw_exit_raw', 'reward_yaw_exit_response',
    'translation_safety_risk', 'translation_safety_alpha_raw', 'translation_safety_alpha', 'translation_safety_emergency',
    'near_goal_final_translation_alpha', 'near_goal_capture_candidate', 'near_goal_capture_active', 'near_goal_capture_distance_m',
    'near_goal_capture_alignment', 'near_goal_capture_cap_m_s', 'near_goal_capture_alpha', 'near_goal_capture_candidate_count',
    'near_goal_capture_entry_count', 'near_goal_capture_exit_count', 'near_goal_capture_zone_success_count', 'near_goal_capture_zone_collision_count',
    'near_goal_capture_zone_timeout_count', 'near_goal_capture_zone_reset_count', 'near_goal_capture_reset_counted_as_completion_error', 'near_goal_capture_entry_to_platform_success_latency_s',
    'recovery_event_count_60s', 'recovery_event_lifetime_count', 'recovery_candidate_entry_count', 'recovery_success_count',
    'recovery_terminal_count', 'recovery_unverified_exit_count', 'recovery_success_rate', 'recovery_time_s',
    'recovery_early_stuck_sample_share', 'recovery_confirmed_stuck_sample_share', 'recovery_safe_exit_share', 'recovery_candidate_lifetime_count',
    'recovery_terminal_lifetime_count', 'goal_map_x_m', 'goal_map_y_m', 'goal_map_distance_m',
    'goal_age_s', 'goal_innovation_d2', 'goal_measurement_accepted', 'goal_measurement_clipped',
    'goal_measurement_rejected', 'goal_dropout_active', 'goal_jump_active', 'goal_propagated',
    'goal_epoch_changed', 'goal_reacquire_pending', 'goal_reacquisition_time_s', 'goal_fault_allowed',
    'goal_freshness', 'goal_process_variance_m2', 'goal_candidate_count', 'goal_stale_low_speed_active',
    'raw_goal_distance_gt10_share', 'goal_innovation_d2_p50', 'goal_innovation_d2_p90', 'goal_innovation_d2_p99',
    'goal_age_s_p50', 'goal_age_s_p90', 'goal_distance_0_5_share', 'goal_accept_0_5',
    'goal_clipped_0_5', 'goal_reject_0_5', 'goal_distance_5_10_share', 'goal_accept_5_10',
    'goal_clipped_5_10', 'goal_reject_5_10', 'goal_distance_10_plus_share', 'goal_accept_10_plus',
    'goal_clipped_10_plus', 'goal_reject_10_plus', 'user_speed_cap', 'effective_speed_cap',
    'safety_speed_cap', 'safety_cap_predictive_risk', 'mapper_version_valid', 'normalized_action_vx',
    'normalized_action_vy', 'normalized_action_wz', 'mapped_cmd_vx', 'mapped_cmd_vy',
    'mapped_cmd_wz', 'policy_target_vx', 'policy_target_vy', 'policy_target_wz',
    'limited_target_vx', 'limited_target_vy', 'limited_target_wz', 'soft_cruise_clear_factor',
    'soft_cruise_low_error', 'soft_cruise_high_error', 'reward_soft_cruise', 'teacher_scene_corridor',
    'teacher_scene_left_open', 'teacher_scene_right_open', 'teacher_scene_junction', 'teacher_scene_dead_end',
    'teacher_scene_fuzzy', 'teacher_safe_top1_clear', 'diagnostic_teacher_coverage', 'diagnostic_wall_auroc',
    'diagnostic_wall_miss_rate', 'diagnostic_safe_top1_accuracy', 'diagnostic_scene_macro_f1', 'diagnostic_clean_live_latent_cosine',
    'diagnostic_goal_wall_auroc', 'diagnostic_goal_safe_top1_accuracy', 'diagnostic_goal_scene_macro_f1', 'diagnostic_fault_wall_auroc',
    'diagnostic_fault_safe_top1_accuracy', 'diagnostic_fault_scene_macro_f1', 'diagnostic_fault_shadow_share', 'diagnostic_clean_fault_latent_cosine',
    'diagnostic_clean_fault_action_mae', 'scanner_valid_share', 'safety_bce', 'safety_head_risk_left',
    'safety_head_risk_center', 'safety_head_risk_right', 'head_correct_samples', 'head_correct_actor_wrong_count',
    'head_correct_actor_wrong_rate', 'risk_event_resolved_count', 'risk_decel_policy_count', 'risk_decel_limited_count',
    'risk_no_deceleration_count', 'risk_decel_policy_rate', 'risk_decel_limited_rate', 'risk_no_deceleration_rate',
    'missed_safe_event_active', 'missed_safe_event_severity', 'missed_safe_weight', 'body_collision_onset',
    'predictive_collision_risk', 'zero_hidden_action_mae', 'zero_hidden_direction_disagreement', 'yaw_exec_cancellation',
    'yaw_true_cancellation', 'yaw_exec_sign_flip', 'yaw_true_sign_flip', 'yaw_true_overshoot',
    'reward_predictive_raw', 'reward_missed_safe_raw', 'reward_yaw_raw', 'reward_continuous_time_scale',
    'reward_predictive_collision_risk', 'reward_missed_safe_direction', 'reward_yaw_cancellation', 'reward_safety_group_scale',
    'reward_decomposed_total', 'reward_conservation_error', 'camera_frame_id', 'camera_capture',
    'camera_frame_changed', 'camera_age_s', 'camera_raw_hole_rate', 'camera_near_clip_added_hole_rate',
    'camera_delivered_hole_rate', 'camera_center_hole_rate', 'camera_lower_hole_rate', 'near_clip_m',
    'near_clip_normalized', 'camera_delay_only', 'camera_fault_only', 'camera_fault_delay_overlap',
    'camera_fault_kind', 'camera_shadow_age_250ms', 'motion_confined_share', 'wall_evidence_share',
    'wall_stuck_candidate_share', 'wall_stuck_duration_s', 'wall_stuck_would_reset', 'wall_stuck_reset_triggered',
    'wall_stuck_raw_term', 'wall_stuck_reset_rate', 'rollout_wall_stuck_reset_count', 'wall_stuck_mapping_valid',
    'reset_after_push_share', 'wall_stuck_saved_seconds', 'rollout_wall_stuck_saved_seconds', 'wall_stuck_duration_p50_s',
    'wall_stuck_duration_p90_s', 'collision_to_stuck_reset_delay_s', 'wall_stuck_term_available', 'wall_stuck_term_config_valid',
    'p4_spawn_hook_installed', 'reward_stuck_reset', 'reward_stuck_sustained', 'reward_goal_safe_preference',
    'reward_route_excess', 'reward_open_straight', 'open_straight_eligible', 'open_straight_lateral_penalty',
    'open_straight_s_turn_penalty', 'open_straight_extra_path_penalty', 'current_segment_slope_share', 'current_segment_slope_inv_share',
    'current_segment_stairs_share', 'current_segment_stairs_inv_share', 'current_segment_maze_share', 'spawn_segment_slope_event_share',
    'spawn_segment_slope_inv_event_share', 'spawn_segment_stairs_event_share', 'spawn_segment_stairs_inv_event_share', 'spawn_segment_maze_event_share',
    'spawn_safe_point_event_share', 'spawn_hard_position_event_share', 'spawn_full_start_event_share', 'spawn_segment_start_event_share',
    'spawn_reset_event_count', 'spawn_reason4_retry_count', 'spawn_reason4_exhausted_count', 'spawn_reason4_fallback_applied_count',
    'spawn_validation_failure_count', 'spawn_write_failure_count', 'goal_jump_radial_offset_m', 'goal_jump_tangent_offset_m',
    'legitimate_side_goal_selection_rate', 'policy_target_vy_positive_mean', 'policy_target_vy_negative_mean', 'limited_target_vy_positive_mean',
    'limited_target_vy_negative_mean', 'mapped_cmd_vy_positive_mean', 'mapped_cmd_vy_negative_mean', 'exec_vy_positive_mean',
    'exec_vy_negative_mean', 'true_vy_positive_mean', 'true_vy_negative_mean', 'policy_target_wz_positive_mean',
    'policy_target_wz_negative_mean', 'limited_target_wz_positive_mean', 'limited_target_wz_negative_mean', 'mapped_cmd_wz_positive_mean',
    'mapped_cmd_wz_negative_mean', 'exec_wz_positive_mean', 'exec_wz_negative_mean', 'true_wz_positive_mean',
    'true_wz_negative_mean', 'reward_command_rate_vy_positive', 'reward_command_rate_vy_negative', 'reward_command_rate_wz_positive',
    'reward_command_rate_wz_negative', 'reward_tracking_vy_positive', 'reward_tracking_vy_negative', 'reward_tracking_wz_positive',
    'reward_tracking_wz_negative', 'wall_stuck_sustained_active', 'wall_stuck_sustained_severity', 'goal_safe_preference_eligible',
    'goal_safe_preference_gap', 'goal_safe_preference_selected', 'goal_safe_preference_best', 'route_path_length_m',
    'route_positive_progress_m', 'route_excess_distance_m', 'route_efficiency', 'reward_frontier_stagnation',
    'reward_frontier_stagnation_shadow', 'maze_new_best_credit', 'maze_new_best_delta_m', 'maze_new_best_episode_earned',
    'terminal_potential_clawback', 'stuck_terminal_count', 'stuck_terminal_episode_return_mean', 'stuck_terminal_nonnegative_rate',
    'episode_starts_per_hour', 'camera_memory_loss', 'camera_clean_live_action_mae', 'camera_clean_live_latent_cosine',
    'camera_aux_coefficient', 'camera_aux_gradient_ratio', 'camera_delay_only_share', 'camera_fault_only_share',
    'camera_fault_delay_overlap_share', 'push_term_assembly_valid', 'push_runtime_active', 'push_telemetry_valid',
    'push_epoch', 'push_event_count', 'push_lifetime_count', 'push_env_coverage',
    'seconds_since_push', 'push_actual_delta_vx_mean', 'push_actual_delta_vy_mean', 'push_actual_delta_vx_abs_max',
    'push_actual_delta_vy_abs_max', 'push_rollout_event_count_total', 'push_lifetime_event_count_total', 'push_env_coverage_rate',
    'push_grace_active', 'push_tracking_response_mae', 'push_recovery_pending', 'push_tracking_recovery_time_s',
    'adapter_push_rejected_02s', 'adapter_push_rejected_06s', 'adapter_push_rejected_10s', 'adapter_push_rejected_pose',
    'adapter_compatible_current_records', 'adapter_compatible_parent_records', 'adapter_compat_migrated_legacy_parent_records', 'adapter_legacy_parent_rejected_records',
    'adapter_compat_rejected_records', 'adapter_target_current_ratio', 'adapter_target_p35_ratio', 'adapter_target_earlier_ratio',
    'adapter_latest_replay_ratio', 'adapter_recent_replay_ratio', 'adapter_parent_replay_ratio', 'adapter_frozen',
    'navigation_learning_rate', 'safety_head_learning_rate', 'adapter_learning_rate', 'adapter_compat_rejected_missing_contract',
    'adapter_compat_rejected_mismatch_version', 'adapter_compat_rejected_mismatch_schema', 'adapter_compat_rejected_mismatch_low_level_digest', 'adapter_compat_rejected_mismatch_feedback_digest',
    'adapter_compat_rejected_mismatch_capability_digest', 'adapter_compat_rejected_mismatch_response_profile15', 'adapter_compat_rejected_mismatch_action_mapper', 'adapter_compat_rejected_mismatch_observation_layout',
    'adapter_compat_rejected_mismatch_label_layout', 'low_digest_drift', 'low_optimizer_steps', 'high_updates',
    'p4_tbptt_sequences_per_env', 'p4_tbptt_nav_ticks', 'rollout_reward_mean', 'memory_allocated',
    'memory_reserved', 'max_memory_allocated', 'max_memory_reserved', 'samples_per_s',
    'full_phase_warm', 'full_phase_adapt', 'full_phase_train', 'full_phase_stabilize',
    'closedloop_phase_critic_warm', 'closedloop_phase_actor_adapt', 'closedloop_phase_train', 'closedloop_phase_stabilize',
    'teacher_safe5_far_left', 'teacher_safe5_left', 'teacher_safe5_center', 'teacher_safe5_right',
    'teacher_safe5_far_right', 'large_goal_correct_yaw_response', 'vy_substitutes_yaw_count', 'translation_limiter_intervention',
    'translation_limiter_shadow', 'legitimate_side_goal_candidate_count', 'legitimate_side_goal_selected_count', 'exec_vy',
    'exec_wz', 'true_vy', 'true_wz',
)

MONITOR_OPTIONAL_METRICS = (
    'maze_branch_actor_attack', 'maze_branch_visual_recovery', 'maze_phase_probe', 'maze_phase_attack',
    'maze_phase_hard', 'maze_phase_final', 'abnormal_count_track_l', 'completed_count_track_l',
    'energy_score_track_l', 'p4_monitor_empty_metric_count', 'p4_monitor_expected_metric_count', 'p4_monitor_longest_data_age_s',
    'p4_monitor_metric_with_data_count', 'p4_monitor_producer_longest_age_s', 'p4_monitor_upload_age_s', 'p4_monitor_upload_failure_count',
    'pose_score_track_l', 'stuck', 'time_score_track_l',
    'timeout_rate', 'timeout_count_track_l', 'total_score_track_l',
)

MONITOR_OPTIONAL_METRIC_PREFIXES = (
    'completed_count_track_l', 'abnormal_count_track_l', 'timeout_count_track_l', 'total_score_track_l',
    'energy_score_track_l', 'pose_score_track_l', 'time_score_track_l',
)

def command_contract(
    training_profile: str = LEGACY_COMPATIBILITY_DEFAULT,
) -> dict[str, Any]:
    """Return the canonical P4 command contract."""
    return _canonical_contracts.command_contract(training_profile)


def reward_contract(
    stuck_reset: dict[str, Any] | None = None,
    training_profile: str = LEGACY_COMPATIBILITY_DEFAULT,
) -> dict[str, Any]:
    """Return the canonical P4 reward contract."""
    return _canonical_contracts.reward_contract(stuck_reset, training_profile)


def training_contract(
    stuck_reset: dict[str, Any] | None = None,
    training_profile: str = LEGACY_COMPATIBILITY_DEFAULT,
) -> dict[str, Any]:
    """Return the canonical P4 training contract."""
    return _canonical_contracts.training_contract(stuck_reset, training_profile)


def contract_metadata(
    stuck_reset: dict[str, Any] | None = None,
    training_profile: str = LEGACY_COMPATIBILITY_DEFAULT,
) -> dict[str, Any]:
    """Return canonical P4 contracts with deterministic digests."""
    return _canonical_contracts.contract_metadata(stuck_reset, training_profile)


__all__ = ('RUN_NAME', 'STAGE_NAME', 'STAGE_TYPE', 'TRAINING_HOURS', 'TARGET_EFFECTIVE_SECONDS', 'LEGACY_MAZE_TRAINING_HOURS', 'LEGACY_MAZE_TARGET_EFFECTIVE_SECONDS', 'DIAGNOSTIC_SECONDS', 'PLATFORM_WALL_MARGIN_SECONDS', 'PLATFORM_WALL_SECONDS', 'PLATFORM_WALL_HOURS', 'SCHEDULE_BOUNDARIES_SECONDS', 'LEGACY_MAZE_SCHEDULE_BOUNDARIES_SECONDS', 'LEGACY_ACTION_MAPPER_VERSION', 'ACTION_MAPPER_VERSION', 'GOAL_BELIEF_VERSION', 'CAMERA_CONTRACT_VERSION', 'ADAPTER_RECORD_CONTRACT_VERSION', 'INSTANT_ADAPTER_RECORD_CONTRACT_VERSION', 'SAFETY_REWARD_RAMP_VERSION', 'CHECKPOINT_CONTRACT_VERSION', 'MAZE_CLOSED_LOOP_SAFETY_REWARD_VERSION', 'MAZE_CLOSED_LOOP_CHECKPOINT_VERSION', 'INSTANT_COMMAND_REWARD_CONTRACT_VERSION', 'INSTANT_COMMAND_CHECKPOINT_CONTRACT_VERSION', 'INSTANT_REPAIR_REWARD_CONTRACT_VERSION', 'INSTANT_REPAIR_CHECKPOINT_CONTRACT_VERSION', 'INSTANT_COMMAND_STUCK_RESET_CONTRACT_VERSION', 'INSTANT_COMMAND_PHASE_LABELS', 'INSTANT_REPAIR_PHASE_LABELS', 'INSTANT_REPAIR_SCHEDULE_BOUNDARIES_SECONDS', 'INSTANT_REPAIR_CHECKPOINT_BOUNDARIES_SECONDS', 'INSTANT_COMMAND_SCHEDULE_BOUNDARIES_SECONDS', 'FULL_TRACK_CHECKPOINT_CONTRACT_VERSION', 'FULL_TRACK_REWARD_CONTRACT_VERSION', 'FULL_TRACK_COMMAND_CONTRACT_VERSION', 'WORKER_WIRE_VERSION', 'LEGACY_STUCK_RESET_CONTRACT_VERSION', 'STUCK_RESET_CONTRACT_VERSION', 'LEGACY_ACTOR_MEAN_GUIDANCE_CONTRACT_VERSION', 'ACTOR_MEAN_GUIDANCE_CONTRACT_VERSION', 'LEGACY_TRANSLATION_VECTOR_LIMITER_CONTRACT_VERSION', 'TRANSLATION_VECTOR_LIMITER_CONTRACT_VERSION', 'LEGACY_NEAR_GOAL_CAPTURE_CONTRACT_VERSION', 'NEAR_GOAL_CAPTURE_CONTRACT_VERSION', 'STUCK_AUX_CONTRACT_VERSION', 'P4_WORKER_EXTRA_DIM', 'P4_PRIVILEGED_WIRE_DIM', 'RAW_GOAL_XY_SLICE', 'STUCK_MOTION_CONFINED_INDEX', 'STUCK_WALL_EVIDENCE_INDEX', 'STUCK_CANDIDATE_INDEX', 'STUCK_DURATION_S_INDEX', 'STUCK_WOULD_RESET_INDEX', 'STUCK_RESET_TRIGGERED_INDEX', 'STUCK_MAPPING_VALID_INDEX', 'STUCK_RESET_AFTER_PUSH_INDEX', 'STUCK_SAVED_SECONDS_INDEX', 'STUCK_COLLISION_TO_RESET_S_INDEX', 'STUCK_TERM_AVAILABLE_INDEX', 'STUCK_TERM_CONFIG_VALID_INDEX', 'SPAWN_INSTALLED_INDEX', 'STUCK_RAW_TERM_INDEX', 'SPAWN_FULL_START_INDEX', 'SPAWN_SEGMENT_INDEX', 'SPAWN_QUARTILE_INDEX', 'SPAWN_SAFE_POINT_INDEX', 'SPAWN_REASON4_RETRY_COUNT_INDEX', 'SPAWN_REASON4_EXHAUSTED_COUNT_INDEX', 'SPAWN_REASON4_FALLBACK_APPLIED_COUNT_INDEX', 'SPAWN_ALL_POSITION_APPLIED_COUNT_INDEX', 'SPAWN_VALIDATION_FAILURE_COUNT_INDEX', 'SPAWN_WRITE_FAILURE_COUNT_INDEX', 'SOFT_CRUISE_MIN_VX', 'SOFT_CRUISE_MAX_VX', 'SOFT_CRUISE_LOW_WEIGHT', 'SOFT_CRUISE_HIGH_WEIGHT', 'P4_MAX_ABS_VY', 'P4_MAX_ABS_WZ', 'P4_MAX_VX', 'P4_NAV_PERIOD_FRAMES', 'P4_NAV_DT_S', 'INSTANT_ACTOR_CAPABILITY_PROFILE15', 'INSTANT_CAPABILITY_CHANGE_RATE', 'INSTANT_PARENT_ANCHOR_SLEW_UP', 'INSTANT_PARENT_ANCHOR_SLEW_RELEASE', 'P4_SLEW_RATE', 'P4_SLEW_RELEASE_RATE', 'LEGACY_P4_SLEW_RATE', 'LEGACY_P4_SLEW_RELEASE_RATE', 'STALE_GOAL_WAIT_MAX_ABS_VY', 'STALE_GOAL_WAIT_MAX_ABS_WZ', 'GOAL_NORMAL_D2', 'GOAL_CLIPPED_D2', 'GOAL_AGE_SLOW_START_S', 'GOAL_AGE_SLOW_END_S', 'GOAL_AGE_HOLD_END_S', 'GOAL_FRESHNESS_FLOOR', 'GOAL_PROCESS_SIGMA_V_M_S', 'GOAL_PROCESS_SIGMA_WZ_RAD_S', 'GOAL_REACQUIRE_SAMPLES', 'STUCK_RESET_TERMINAL_PENALTY', 'SUCCESS_IMPULSE', 'FAILURE_IMPULSE', 'TIMEOUT_IMPULSE', 'LEGACY_STUCK_RESET_TERMINAL_PENALTY', 'LEGACY_TIMEOUT_IMPULSE', 'STUCK_SUSTAINED_GRACE_S', 'STUCK_SUSTAINED_FULL_S', 'STUCK_SUSTAINED_BASE', 'STUCK_SUSTAINED_FLOOR', 'LEGACY_STUCK_SUSTAINED_FLOOR', 'P4_BODY_COLLISION_ONSET_BASE', 'P4_BODY_COLLISION_ONSET_SEVERITY', 'P4_BODY_COLLISION_PERSISTENT', 'INSTANT_COMMAND_COLLISION_ONSET_BASE', 'INSTANT_COMMAND_COLLISION_ONSET_SEVERITY', 'INSTANT_COMMAND_COLLISION_PERSISTENT', 'INSTANT_COMMAND_STUCK_SUSTAINED_GRACE_S', 'INSTANT_COMMAND_STUCK_SUSTAINED_FULL_S', 'INSTANT_COMMAND_STUCK_SUSTAINED_BASE', 'INSTANT_COMMAND_STUCK_SUSTAINED_FLOOR', 'INSTANT_REPAIR_COMMAND_RATE_WEIGHT', 'INSTANT_REPAIR_TRACKING_ERROR_WEIGHT', 'INSTANT_REPAIR_STALE_TRANSLATION_CAP_M_S', 'INSTANT_REPAIR_STALE_YAW_CAP_RAD_S', 'INSTANT_REPAIR_NEAR_GOAL_MAX_SPEED_M_S', 'INSTANT_REPAIR_NEAR_GOAL_MIN_SPEED_M_S', 'GOAL_SAFE_PREFERENCE_WEIGHT', 'GOAL_SAFE_PREFERENCE_MARGIN', 'GOAL_SAFE_PREFERENCE_SCALE', 'ROUTE_EXCESS_WEIGHT', 'ROUTE_EXCESS_CAP_M', 'YAW_EXIT_RESPONSE_RAW_FLOOR', 'TRANSLATION_LIMITER_RISK_THRESHOLD', 'TRANSLATION_LIMITER_ALPHA_FLOOR', 'TRANSLATION_LIMITER_RELEASE_PER_TICK', 'LEGACY_TRANSLATION_LIMITER_RISK_THRESHOLD', 'LEGACY_TRANSLATION_LIMITER_ALPHA_FLOOR', 'NEAR_GOAL_CAPTURE_MIN_DISTANCE_M', 'NEAR_GOAL_CAPTURE_MAX_DISTANCE_M', 'NEAR_GOAL_CAPTURE_MIN_SPEED_M_S', 'NEAR_GOAL_CAPTURE_GOAL_COSINE_MIN', 'NEAR_GOAL_CAPTURE_FRESHNESS_MIN', 'TEACHER_SAFE_MIN', 'TEACHER_SAFE_MARGIN_MIN', 'TEACHER_GOAL_SAFE_TIE_MARGIN', 'TEACHER_GOAL_FRESHNESS_MIN', 'TEACHER_DIRECTION_TOLERANCE_DEG', 'TEACHER_EDGE_DIRECTION_TOLERANCE_DEG', 'TEACHER_EDGE_SAFE_MIN', 'TEACHER_EDGE_SPEED_CAP_MIN', 'TEACHER_EDGE_SPEED_CAP_RANGE', 'TEACHER_RECOVERY_MAX_VX', 'TEACHER_RECOVERY_MIN_ABS_VY', 'TEACHER_RECOVERY_SIDE_MARGIN', 'TEACHER_SPEED_RISK_MIN', 'TEACHER_MIN_VALID_STEPS', 'TEACHER_SAFE5_ANGLES_DEG', 'TEACHER_SAFE5_HEIGHT_SECTORS', 'STUCK_RESET_DEFAULTS', 'FULL_TRACK_SEGMENT_LABELS', 'FULL_TRACK_SEGMENT_LENGTH_M', 'SEGMENT_FRONTIER_WEIGHT', 'MAZE_NEW_BEST_WEIGHT_PER_M', 'MAZE_NEW_BEST_EPISODE_CAP', 'LEGACY_MAZE_NEW_BEST_WEIGHT_PER_M', 'LEGACY_MAZE_NEW_BEST_EPISODE_CAP', 'OPEN_STRAIGHT_LATERAL_WEIGHT', 'OPEN_STRAIGHT_S_TURN_WEIGHT', 'OPEN_STRAIGHT_EXTRA_PATH_WEIGHT', 'OPEN_STRAIGHT_TOTAL_FLOOR', 'OPEN_STRAIGHT_TRUE_VY_DEADBAND_M_S', 'OPEN_STRAIGHT_GOAL_BEARING_MAX_DEG', 'OPEN_STRAIGHT_CENTER_SAFE_MIN', 'OPEN_STRAIGHT_CENTER_BEST_MARGIN', 'OPEN_STRAIGHT_BOUNDARY_MARGIN_M', 'GOAL_JUMP_RATE_PER_S', 'GOAL_JUMP_DURATION_S', 'GOAL_JUMP_MIN_DISTANCE_M', 'GOAL_JUMP_RADIAL_RANGE_M', 'GOAL_JUMP_TANGENT_RANGE_M', 'YAW_WINDOW_SECONDS', 'RISK_RESPONSE_WINDOW_SECONDS', 'YAW_EXEC_WEIGHT', 'YAW_TRUE_WEIGHT', 'YAW_TOTAL_FLOOR', 'SAFETY_GROUP_FLOOR', 'LEGACY_YAW_EXEC_WEIGHT', 'LEGACY_YAW_TRUE_WEIGHT', 'LEGACY_YAW_TOTAL_FLOOR', 'LEGACY_SAFETY_GROUP_FLOOR', 'PREDICTIVE_COLLISION_SCALE', 'PREDICTIVE_RAW_FLOOR', 'MISSED_SAFE_RAW_FLOOR', 'SAFE_DIRECTION_GAP_SCALE', 'NAVIGATION_ENCODER_LRS', 'ACTOR_LR', 'CRITIC_LR', 'SAFETY_HEAD_LR', 'ADAPTER_LR', 'ADAPTER_COMPATIBILITY_REJECTION_REASONS', 'INSTANT_REPAIR_PROFILE', 'LEGACY_INSTANT_PROFILE', 'MONITOR_REQUIRED_METRICS', 'MONITOR_OPTIONAL_METRICS', 'MONITOR_OPTIONAL_METRIC_PREFIXES', 'map_normalized_action', 'map_normalized_action_legacy', 'instant_parent_anchor_reachable', 'stale_goal_cap', 'effective_speed_cap', 'translation_vector_limiter', 'near_goal_capture', 'adapter_compatibility_metric_name', 'teacher_guidance_mask', 'privileged_safe_directions5', 'teacher_guidance_loss', 'instant_r4_teacher_masks', 'instant_r4_teacher_guidance_loss', 'soft_cruise_penalty', 'sustained_wall_stuck_penalty', 'maze_new_best_credit', 'goal_safe_direction_penalty', 'yaw_exit_response_penalty', 'route_excess_penalty', 'segment_frontier_potential', 'track_boundary_distance_m', 'open_straight_penalty', 'yaw_cancellation', 'proportional_negative_cap', 'maze_missed_safe_direction_penalty', 'safety_scene_diagnostics', 'push_phase_config', 'camera_mix', 'safe_direction_weight', 'training_schedule', 'normalize_stuck_reset_contract', 'stable_digest', 'adapter_record_contract', 'command_contract', 'reward_contract', 'training_contract', 'contract_metadata')
