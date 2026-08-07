"""Canonical P4 contract builders and deterministic contract digests."""

from __future__ import annotations
import hashlib
import json
import math
from typing import Any
from agent_ppo.feature import p2_contract
from agent_ppo.p4 import profiles as _profile_registry
from agent_ppo.p4.constants import *
from agent_ppo.p4.profiles import (
    LEGACY_COMPATIBILITY_DEFAULT,
    PROFILE_FULL_TRACK,
    PROFILE_MAZE_CLOSED_LOOP_V3,
    PROFILE_MAZE_CREDIT_REPAIR,
    PROFILE_MAZE_STABLE_DIRECTION_SMOKE,
    PROFILE_MAZE_STABLE_DIRECTION_8H,
    get_training_profile,
)


def stable_digest(value: Any) -> str:
    """Return the stable SHA256 used in checkpoints and compatibility gates."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_stable_direction(profile: str) -> bool:
    return profile in {
        PROFILE_MAZE_STABLE_DIRECTION_SMOKE,
        PROFILE_MAZE_STABLE_DIRECTION_8H,
    }


def _normalize_profile_argument(
    stuck_reset: dict[str, Any] | str | None, training_profile: str
) -> tuple[dict[str, Any] | None, str]:
    if isinstance(stuck_reset, str):
        if training_profile != LEGACY_COMPATIBILITY_DEFAULT:
            raise TypeError(
                "P4 profile was supplied both positionally and by training_profile"
            )
        return (None, stuck_reset)
    return (stuck_reset, training_profile)


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
        "max_true_motion_speed_m_s": float(merged["max_true_motion_speed_m_s"]),
        "terminal_penalty": float(merged["terminal_penalty"]),
    }
    numeric = tuple(
        (
            value
            for key, value in result.items()
            if key not in {"enabled", "mode", "schedule_enabled"}
        )
    )
    if not all((math.isfinite(value) for value in numeric)):
        raise ValueError("P4 stuck-reset numeric fields must be finite")
    if any(
        (
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
        )
    ):
        raise ValueError(
            "P4 stuck-reset durations, distances and force must be non-negative"
        )
    if result["terminal_penalty"] > 0.0:
        raise ValueError("P4 stuck-reset terminal_penalty must be non-positive")
    if result["tighten_after_s"] < result["activation_delay_s"]:
        raise ValueError(
            "P4 stuck-reset tighten_after_s must not precede activation_delay_s"
        )
    return result


def _build_command_contract(
    training_profile: str = PROFILE_MAZE_CREDIT_REPAIR,
) -> dict[str, Any]:
    profile = _profile_registry.normalize_training_profile(training_profile)
    if _profile_registry.is_instant_profile(profile):
        return {
            "version": "p4_maze_instant_command_r4_inputfix",
            "mapper_version": ACTION_MAPPER_VERSION,
            "legacy_mapper_version": LEGACY_ACTION_MAPPER_VERSION,
            "normalized_action": "unchanged_tanh_gaussian_v2",
            "mapped_ranges": {"vx": [0.0, 1.0], "vy": [-0.3, 0.3], "wz": [-0.9, 0.9]},
            "policy_target_vx": [0.0, 1.0],
            "command_transition_mode": "instant_hold_10hz",
            "hold_frames": P4_NAV_PERIOD_FRAMES,
            "nav_period_frames": P4_NAV_PERIOD_FRAMES,
            "nav_frequency_hz": 1.0 / P4_NAV_DT_S,
            "policy_target_equals_exec": "at_10hz_tick_boundary",
            "instant_physical_change_rate_per_s": list(INSTANT_CAPABILITY_CHANGE_RATE),
            "actor_capability_profile15": {
                "values": list(INSTANT_ACTOR_CAPABILITY_PROFILE15),
                "semantics": "frozen_parent_observation_compatibility_not_controller_rate",
            },
            "command_rewrites": {
                "slew": "disabled",
                "zero_cross_guard": "disabled",
                "reversal_guard": "disabled",
                "runtime_limiter": "disabled",
                "near_goal_rewrite": "disabled",
                "recovery_override": "disabled",
            },
            "hard_action_ranges_only": True,
            "speed_tier_contract": "removed",
            "soft_cruise": {"status": "disabled"},
            "translation_vector_limiter": {"status": "disabled_no_runtime_limiter"},
            "near_goal_capture": {"status": "disabled_no_command_rewrite"},
        }
    closed_loop = profile == PROFILE_MAZE_CLOSED_LOOP_V3
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
                if profile == PROFILE_MAZE_CREDIT_REPAIR
                else FULL_TRACK_COMMAND_CONTRACT_VERSION
            )
        ),
        "mapper_version": ACTION_MAPPER_VERSION,
        "legacy_mapper_version": LEGACY_ACTION_MAPPER_VERSION,
        "normalized_action": "unchanged_tanh_gaussian_v2",
        "mapped_ranges": {"vx": [0.0, 1.0], "vy": [-0.3, 0.3], "wz": [-0.9, 0.9]},
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
                "max_vx": 0.2,
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
            "alpha_raw": f"risk<={limiter_threshold:.2f}:1; risk=1:{limiter_floor:.2f}; linear between",
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
            "distance_m": [
                NEAR_GOAL_CAPTURE_MIN_DISTANCE_M,
                NEAR_GOAL_CAPTURE_MAX_DISTANCE_M,
            ],
            "goal_freshness_min": NEAR_GOAL_CAPTURE_FRESHNESS_MIN,
            "policy_goal_cosine_min": NEAR_GOAL_CAPTURE_GOAL_COSINE_MIN,
            **(
                {"status": "shadow_diagnostic_only_no_command_rewrite"}
                if closed_loop
                else {"axes": "translation_only_preserve_yaw"}
            ),
        },
    }


def _reward_new_terms_part_1(
    profile, closed_loop, instant_command, instant_repair, stuck, stuck_term
) -> dict[str, Any]:
    stable_direction = _is_stable_direction(profile)
    return {
        "predictive_collision_raw_floor": PREDICTIVE_RAW_FLOOR,
        "predictive_collision_scale": PREDICTIVE_COLLISION_SCALE,
        "missed_safe_direction_raw_floor": MISSED_SAFE_RAW_FLOOR,
        "missed_safe_direction_gap_scale": SAFE_DIRECTION_GAP_SCALE,
        "frontier_stagnation": "shadow_only_zero_ppo_weight",
        "yaw_exec_weight": (
            YAW_EXEC_WEIGHT
            if closed_loop or instant_command
            else LEGACY_YAW_EXEC_WEIGHT
        ),
        "yaw_true_weight": (
            YAW_TRUE_WEIGHT
            if closed_loop or instant_command
            else LEGACY_YAW_TRUE_WEIGHT
        ),
        "yaw_total_floor": (
            YAW_TOTAL_FLOOR
            if closed_loop or instant_command
            else LEGACY_YAW_TOTAL_FLOOR
        ),
        "yaw_exit_response_raw_floor": YAW_EXIT_RESPONSE_RAW_FLOOR,
        "safety_group_floor": (
            SAFETY_GROUP_FLOOR
            if closed_loop or instant_command
            else LEGACY_SAFETY_GROUP_FLOOR
        ),
        "safety_group_terms": (
            [
                "predictive_collision",
                "missed_safe_direction",
                "goal_safe_preference",
                "yaw_exit_response",
            ]
            if stable_direction
            else ["predictive_collision"]
            if closed_loop or instant_command
            else [
                "predictive_collision",
                "missed_safe_direction",
                "yaw_cancellation",
                "yaw_exit_response",
                "goal_safe_preference",
            ]
        ),
        **(
            {"yaw_cancellation": "disabled_no_reward"}
            if instant_command
            else (
                {"yaw_cancellation": "shadow_diagnostic_only_zero_ppo_weight"}
                if closed_loop
                else {}
            )
        ),
        "cap_semantics": "proportional_no_hidden_adjustment_5hz_reference",
        "confirmed_wall_stuck_reset": stuck_term,
    }


def _reward_new_terms_part_2(
    profile, closed_loop, instant_command, instant_repair, stuck, stuck_term
) -> dict[str, Any]:
    stable_direction = _is_stable_direction(profile)
    return {
        "success_impulse": SUCCESS_IMPULSE,
        **(
            {"failure_impulse": FAILURE_IMPULSE}
            if closed_loop or instant_command
            else {}
        ),
        "timeout_impulse": (
            TIMEOUT_IMPULSE
            if closed_loop or instant_command
            else LEGACY_TIMEOUT_IMPULSE
        ),
        "sustained_wall_stuck": {
            "grace_s": (
                INSTANT_COMMAND_STUCK_SUSTAINED_GRACE_S
                if instant_command
                else STUCK_SUSTAINED_GRACE_S
            ),
            "full_penalty_s": (
                INSTANT_COMMAND_STUCK_SUSTAINED_FULL_S
                if instant_command
                else STUCK_SUSTAINED_FULL_S
            ),
            "base": (
                INSTANT_COMMAND_STUCK_SUSTAINED_BASE
                if instant_command
                else STUCK_SUSTAINED_BASE
            ),
            "floor": (
                INSTANT_COMMAND_STUCK_SUSTAINED_FLOOR
                if instant_command
                else (
                    STUCK_SUSTAINED_FLOOR
                    if closed_loop
                    else LEGACY_STUCK_SUSTAINED_FLOOR
                )
            ),
        },
        "closed_loop_collision": {
            "onset_base": (
                INSTANT_COMMAND_COLLISION_ONSET_BASE
                if instant_command
                else P4_BODY_COLLISION_ONSET_BASE
            ),
            "onset_severity": (
                INSTANT_COMMAND_COLLISION_ONSET_SEVERITY
                if instant_command
                else P4_BODY_COLLISION_ONSET_SEVERITY
            ),
            "persistent": (
                INSTANT_COMMAND_COLLISION_PERSISTENT
                if instant_command
                else P4_BODY_COLLISION_PERSISTENT
            ),
        },
        "recovery_translation_teacher": {
            "status": (
                "active_training_only_safe5_lateral_egress"
                if closed_loop
                else "disabled_no_recovery_bonus" if instant_command else "disabled"
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
                if closed_loop or instant_command
                else LEGACY_MAZE_NEW_BEST_WEIGHT_PER_M
            ),
            "episode_cap": (
                MAZE_NEW_BEST_EPISODE_CAP
                if closed_loop or instant_command
                else LEGACY_MAZE_NEW_BEST_EPISODE_CAP
            ),
            "terminal": (
                "retain_earned_credit_no_clawback"
                if instant_repair
                else (
                    "exact_clawback_on_failure_timeout_reason4"
                    if instant_command
                    else "retain_earned_credit_no_clawback"
                )
            ),
        },
        "open_straight": {
            "segments": ["slope", "slope_inv"],
            "lateral_weight": OPEN_STRAIGHT_LATERAL_WEIGHT,
            "s_turn_weight": OPEN_STRAIGHT_S_TURN_WEIGHT,
            "extra_path_weight": OPEN_STRAIGHT_EXTRA_PATH_WEIGHT,
            "total_floor": OPEN_STRAIGHT_TOTAL_FLOOR,
            "requires": "teacher_open_center_clean_goal_near_axis_boundary_margin",
            "zero_on": [
                "stairs",
                "maze",
                "junction",
                "dead_end",
                "contact",
                "recovery",
                "stale",
                "terminal",
            ],
        },
        "soft_cruise": {
            "preferred_vx": [SOFT_CRUISE_MIN_VX, SOFT_CRUISE_MAX_VX],
            "low_weight": SOFT_CRUISE_LOW_WEIGHT,
            "high_weight": SOFT_CRUISE_HIGH_WEIGHT,
        },
    }


def _reward_new_terms(
    profile, closed_loop, instant_command, instant_repair, stuck, stuck_term
) -> dict[str, Any]:
    terms: dict[str, Any] = {}
    terms.update(
        _reward_new_terms_part_1(
            profile, closed_loop, instant_command, instant_repair, stuck, stuck_term
        )
    )
    terms.update(
        _reward_new_terms_part_2(
            profile, closed_loop, instant_command, instant_repair, stuck, stuck_term
        )
    )
    return terms


def _reward_contract_part_1(
    profile, closed_loop, instant_command, instant_repair, stuck, stuck_term
) -> dict[str, Any]:
    return {
        "version": (
            INSTANT_REPAIR_REWARD_CONTRACT_VERSION
            if instant_repair
            else (
                INSTANT_COMMAND_REWARD_CONTRACT_VERSION
                if instant_command
                else (
                    "p4_maze_closed_loop_reward_v3_single_signal"
                    if closed_loop
                    else (
                        "p4_maze_credit_repair_reward_v1"
                        if profile == PROFILE_MAZE_CREDIT_REPAIR
                        else FULL_TRACK_REWARD_CONTRACT_VERSION
                    )
                )
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
        "profile_weights": {
            "command_rate": (
                INSTANT_REPAIR_COMMAND_RATE_WEIGHT
                if instant_repair
                else p2_contract.COMMAND_RATE_WEIGHT
            ),
            "tracking_error": (
                INSTANT_REPAIR_TRACKING_ERROR_WEIGHT
                if instant_repair
                else p2_contract.TRACKING_ERROR_WEIGHT
            ),
            "command_rate_axis_weights": list(p2_contract.COMMAND_RATE_AXIS_WEIGHTS),
            "tracking_error_axis_weights": list(
                p2_contract.TRACKING_ERROR_AXIS_WEIGHTS
            ),
        },
        "new_terms": _reward_new_terms(
            profile, closed_loop, instant_command, instant_repair, stuck, stuck_term
        ),
        "goal_truth_consumers": ["critic", "reward", "terminal", "scorer"],
        "goal_belief_consumers": ["actor", "speed_cap"],
    }


def _training_contract_part_1(
    profile,
    maze_profile,
    closed_loop,
    instant_command,
    instant_repair,
    legacy_maze,
    profile_spec,
    serialized_stuck,
) -> dict[str, Any]:
    stable_direction = _is_stable_direction(profile)
    return {
        "version": (
            "p4_maze_stable_direction_v1"
            if stable_direction
            else INSTANT_REPAIR_CHECKPOINT_CONTRACT_VERSION
            if instant_repair
            else (
                INSTANT_COMMAND_CHECKPOINT_CONTRACT_VERSION
                if instant_command
                else (
                    MAZE_CLOSED_LOOP_CHECKPOINT_VERSION
                    if closed_loop
                    else (
                        CHECKPOINT_CONTRACT_VERSION
                        if profile == PROFILE_MAZE_CREDIT_REPAIR
                        else FULL_TRACK_CHECKPOINT_CONTRACT_VERSION
                    )
                )
            )
        ),
        "training_profile": profile,
        "run_name": profile_spec.run_name,
        "training_hours": profile_spec.training_hours,
        "target_effective_seconds": profile_spec.target_effective_seconds,
        "diagnostic_seconds": int(DIAGNOSTIC_SECONDS),
        "required_platform_wall_seconds": profile_spec.required_platform_wall_seconds,
        "required_platform_wall_hours": profile_spec.required_platform_wall_hours,
        "clock_semantics": {
            "diagnostic": "wall_seconds_before_training_not_counted_in_session",
            "session_effective_seconds": "gradient_training_seconds_only",
            "session_wall_seconds": "task_wall_including_rollout_update_checkpoint_monitor_and_logging",
            "platform_task": "must_cover diagnostic plus target effective seconds plus bounded rollout/save shutdown margin",
            "platform_wall_margin_seconds": int(PLATFORM_WALL_MARGIN_SECONDS),
        },
        "schedule_boundaries_seconds": list(profile_spec.schedule_boundaries_seconds),
        **(
            {
                "checkpoint_phase_labels": list(
                    (
                        [
                            "stable_warm",
                            "stable_early",
                            "stable_mid",
                            "stable_late",
                            "stable_stabilize",
                        ]
                        if stable_direction
                        else INSTANT_REPAIR_PHASE_LABELS
                        if instant_repair
                        else INSTANT_COMMAND_PHASE_LABELS
                    )
                )
            }
            if instant_command
            else {}
        ),
        "safety_reward_ramp": {
            "version": (
                "p4_maze_reward_v6_stable_direction"
                if stable_direction
                else INSTANT_REPAIR_REWARD_CONTRACT_VERSION
                if instant_repair
                else (
                    INSTANT_COMMAND_REWARD_CONTRACT_VERSION
                    if instant_command
                    else (
                        MAZE_CLOSED_LOOP_SAFETY_REWARD_VERSION
                        if closed_loop
                        else (
                            SAFETY_REWARD_RAMP_VERSION
                            if maze_profile
                            else "p4_full_track_safety_group_v2"
                        )
                    )
                )
            ),
            "segments": [
                {
                    "seconds": [0, profile_spec.target_effective_seconds],
                    "weight": (
                        [-0.01, -0.02]
                        if stable_direction
                        else [0.0, 0.0]
                        if closed_loop or instant_command
                        else [0.012, 0.012]
                    ),
                }
            ],
        },
        "goal_fault_ramp": {
            "semantics": (
                "disabled_for_instant_command_maze_run"
                if instant_command
                else (
                    "disabled_for_closed_loop_maze_run"
                    if closed_loop
                    else (
                        "disabled_for_credit_assignment_run"
                        if legacy_maze
                        else "ramp_after_30m_to_full_at_2h"
                    )
                )
            ),
            "segments": (
                [
                    {
                        "seconds": [0, profile_spec.target_effective_seconds],
                        "multiplier": [0.0, 0.0],
                    }
                ]
                if closed_loop or instant_command
                else (
                    [{"seconds": [0, 7200], "multiplier": [0.0, 0.0]}]
                    if legacy_maze
                    else [
                        {"seconds": [0, 1800], "multiplier": [0.0, 0.0]},
                        {"seconds": [1800, 7200], "multiplier": [0.0, 1.0]},
                        {"seconds": [7200, 28800], "multiplier": [1.0, 1.0]},
                    ]
                )
            ),
        },
        "rollout_nav_ticks": 32,
    }


def _training_contract_part_2(
    profile,
    maze_profile,
    closed_loop,
    instant_command,
    instant_repair,
    legacy_maze,
    profile_spec,
    serialized_stuck,
) -> dict[str, Any]:
    stable_direction = _is_stable_direction(profile)
    return {
        "tbptt_nav_ticks": 16,
        "nav_period_frames": P4_NAV_PERIOD_FRAMES,
        "nav_frequency_hz": 1.0 / P4_NAV_DT_S,
        "frozen_low_level": ["cnn", "lstm", "actor", "std", "critic"],
        "trainable": (
            ["high_actor_lstm", "high_actor_head", "high_critic"]
            + (
                []
                if stable_direction
                else [
                    (
                        "response_adapter_phase_5m_to_90m_only"
                        if instant_repair
                        else "response_adapter_phase_30m_to_2h_only"
                    )
                ]
            )
            if instant_command
            else (
                ["high_actor_lstm", "high_actor_head", "high_critic"]
                if closed_loop
                else (
                    ["high_actor_lstm", "high_actor_head", "high_critic", "stuck_head"]
                    if legacy_maze
                    else [
                        "navigation_encoder",
                        "high_actor_lstm",
                        "high_actor_head",
                        "high_critic",
                        "safety_head",
                        "stuck_head",
                        "response_adapter",
                    ]
                )
            )
        ),
        "frozen_high_level": (
            ["navigation_encoder", "safety_head", "stuck_head", "response_adapter"]
            if stable_direction
            else (
                ["navigation_encoder", "safety_head", "stuck_head"]
                if instant_command
                else (
                    ["navigation_encoder", "safety_head", "stuck_head", "response_adapter"]
                    if closed_loop
                    else (
                        ["navigation_encoder", "safety_head", "response_adapter"]
                        if legacy_maze
                        else []
                    )
                )
            )
        ),
        "goal_belief_version": GOAL_BELIEF_VERSION,
        "camera_contract_version": CAMERA_CONTRACT_VERSION,
        "worker_wire_version": WORKER_WIRE_VERSION,
        "worker_wire_dim": P4_PRIVILEGED_WIRE_DIM,
        "stuck_reset_contract_version": (
            INSTANT_COMMAND_STUCK_RESET_CONTRACT_VERSION
            if instant_command
            else (
                STUCK_RESET_CONTRACT_VERSION
                if closed_loop
                else LEGACY_STUCK_RESET_CONTRACT_VERSION
            )
        ),
        "stuck_reset": serialized_stuck,
        "adapter_record_contract_version": (
            INSTANT_ADAPTER_RECORD_CONTRACT_VERSION
            if instant_command
            else ADAPTER_RECORD_CONTRACT_VERSION
        ),
        "adapter_replay_policy": (
            "frozen_no_adapter_update"
            if stable_direction
            else (
                "instant_compatible_current_only_parent_ratio_0"
                if instant_repair
                else (
                    "p4_compatible_50_25_25"
                    if instant_command
                    else "legacy_profile_default"
                )
            )
        ),
    }


def _training_contract_part_5(
    profile,
    maze_profile,
    closed_loop,
    instant_command,
    instant_repair,
    legacy_maze,
    profile_spec,
    serialized_stuck,
) -> dict[str, Any]:
    return {
        "neutral_profile_dropout_ratio": 0.1,
        "actor_mean_guidance": {
            "version": (
                ACTOR_MEAN_GUIDANCE_CONTRACT_VERSION
                if closed_loop or instant_command
                else LEGACY_ACTOR_MEAN_GUIDANCE_CONTRACT_VERSION
            ),
            "minimum_valid_steps": TEACHER_MIN_VALID_STEPS,
            **(
                {
                    "teacher_directions": [
                        "far_left",
                        "left",
                        "center",
                        "right",
                        "far_right",
                    ],
                    "weights": (
                        {
                            "direction": 0.34,
                            "speed": 0.085,
                            "yaw": 0.2125,
                            "edge": 0.1275,
                            "recovery": 0.085,
                            "stale_goal": 0.075,
                            "near_goal": 0.075,
                        }
                        if instant_repair
                        else {"direction": 0.55, "speed": 0.1, "yaw": 0.35}
                    ),
                    "goal_safe_tie_margin": TEACHER_GOAL_SAFE_TIE_MARGIN,
                }
                if closed_loop or instant_command
                else {"weights": {"direction": 0.45, "speed": 0.2, "yaw": 0.35}}
            ),
            "gradient_target_ratio": (
                [0.0, 0.015]
                if instant_repair
                else (
                    [0.0, 0.0175]
                    if instant_command
                    else (
                        [0.0, 0.02]
                        if closed_loop
                        else [0.0, 0.025] if maze_profile else [0.0, 0.0225]
                    )
                )
            ),
            "gradient_hard_cap": (
                0.03
                if closed_loop or instant_command
                else 0.05 if maze_profile else 0.03
            ),
            "nav_feat_detached": True,
            "rollout_time_labels_required": True,
        },
        "mirror": {
            "requested_eligible_sequence_share": 0.0 if maze_profile else 0.1,
            "eligibility": "episode_start_zero_hidden_no_reset_crossing",
            "gradient_target_ratio": 0.0 if maze_profile else 0.005,
            "gradient_hard_cap": 0.0 if maze_profile else 0.01,
        },
        **(
            {
                "anchor": {
                    "multiplier": "explicit_per_instant_phase",
                    "target_ratio": "explicit_per_instant_phase",
                    "phase_labels": list(
                        INSTANT_REPAIR_PHASE_LABELS
                        if instant_repair
                        else INSTANT_COMMAND_PHASE_LABELS
                    ),
                    "eligibility": "low_risk_and_parent_target_reachable_by_legacy_slew_in_one_nav_tick",
                },
                "exact_resume_incompatible_command_transition_modes": ["slew"],
            }
            if instant_command
            else {}
        ),
        "stuck_aux": {
            "version": STUCK_AUX_CONTRACT_VERSION,
            "balanced_positive_negative": True,
            "classifier": "actor_lstm_to_stuck_logit_training_only",
            **(
                {"status": "frozen_diagnostic_only"}
                if closed_loop or instant_command
                else {}
            ),
        },
        "closed_loop_guards": {
            "post_actor_command_rewrite": (
                "none_policy_target_equals_exec_at_10hz_tick_boundary"
                if instant_command
                else "none_translation_limiter_shadow_only" if closed_loop else False
            ),
            "near_goal_capture": (
                "disabled_no_command_rewrite"
                if instant_command
                else "shadow_diagnostic_only"
            ),
            "translation_limiter": {
                **(
                    {"status": "disabled_no_runtime_limiter"}
                    if instant_command
                    else {
                        "risk_threshold": TRANSLATION_LIMITER_RISK_THRESHOLD,
                        "minimum_xy_scale": TRANSLATION_LIMITER_ALPHA_FLOOR,
                        "preserves_wz": True,
                        "status": "shadow_only" if closed_loop else "active",
                    }
                )
            },
            "global_yaw_cancellation_reward": False,
            "teacher_is_training_only": True,
            **({"recovery_command_override": "disabled"} if instant_command else {}),
        },
        "maze_only": maze_profile,
    }


def _training_contract_part_6(
    profile,
    maze_profile,
    closed_loop,
    instant_command,
    instant_repair,
    legacy_maze,
    profile_spec,
    serialized_stuck,
) -> dict[str, Any]:
    stable_direction = _is_stable_direction(profile)
    return {
        "track_segment_labels": (
            ["maze"] if maze_profile else list(FULL_TRACK_SEGMENT_LABELS)
        ),
        "track_length": 1 if maze_profile else 5,
        "episode_length_s": (
            75.0
            if instant_repair or legacy_maze
            else 120.0 if closed_loop or instant_command or (not maze_profile) else 75.0
        ),
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
            "p4_maze_stable_direction_v1_only"
            if stable_direction
            else "p4_maze_instant_repair2h_v1_only"
            if instant_repair
            else (
                "p4_maze_instant_command_r4_inputfix_only"
                if instant_command
                else (
                    "p4_maze_closed_loop_v3_only"
                    if closed_loop
                    else (
                        "p4_maze_credit_repair_v1_only"
                        if legacy_maze
                        else "p4_full_track_v2_only_worker_spawn_rng_reseeded"
                    )
                )
            )
        ),
    }


def _build_reward_contract(
    stuck_reset: dict[str, Any] | None = None,
    training_profile: str = PROFILE_MAZE_CREDIT_REPAIR,
) -> dict[str, Any]:
    profile = _profile_registry.normalize_training_profile(training_profile)
    stable_direction = _is_stable_direction(profile)
    closed_loop = profile == PROFILE_MAZE_CLOSED_LOOP_V3
    instant_command = _profile_registry.is_instant_profile(profile)
    profile_spec = _profile_registry.get_training_profile(profile)
    instant_repair = profile_spec.trainable and profile_spec.instant_command
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
    contract: dict[str, Any] = {}
    contract.update(
        _reward_contract_part_1(
            profile, closed_loop, instant_command, instant_repair, stuck, stuck_term
        )
    )
    if closed_loop or instant_command:
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
        if stable_direction:
            contract["reward_allowlist"].extend(
                [
                    "missed_safe_direction",
                    "goal_safe_preference",
                    "yaw_exit_response",
                ]
            )
            contract["new_terms"]["missed_safe_direction"] = {
                "status": "active_negative_ppo_reward",
                "eligibility": "teacher_valid_moving_clear_alternative_nonterminal",
            }
            contract["new_terms"]["goal_safe_preference"] = {
                "status": "active_negative_ppo_reward",
                "eligibility": "fresh_goal_only_among_safe_exits",
            }
            contract["new_terms"]["yaw_exit_response"] = {
                "status": "active_negative_ppo_reward",
                "eligibility": "fresh_goal_clear_side_exit_nonterminal",
            }
            contract["safety_group_cap"] = {
                "seconds": [0, 1800, profile_spec.target_effective_seconds],
                "floor": [-0.01, -0.02],
                "semantics": "proportional_cap_no_positive_shaping",
            }
        else:
            contract["new_terms"]["missed_safe_direction"] = "shadow_only_zero_ppo_weight"
            contract["new_terms"]["goal_safe_preference"] = "shadow_only_zero_ppo_weight"
            contract["new_terms"]["yaw_exit_response"] = "shadow_only_zero_ppo_weight"
        contract["new_terms"]["soft_cruise"]["status"] = "disabled"
    if profile == PROFILE_FULL_TRACK:
        contract["new_terms"]["maze_new_best_credit"] = {
            "status": "disabled_outside_maze_credit_repair"
        }
        contract["new_terms"]["segment_frontier"] = {
            "weight": SEGMENT_FRONTIER_WEIGHT,
            "semantics": "first_segment_progress_terminal_clawed_potential",
        }
        contract["new_terms"]["open_straight"][
            "status"
        ] = "enabled_on_teacher_confirmed_slope_and_slope_inv_only"
    return contract


def _build_training_contract(
    stuck_reset: dict[str, Any] | None = None,
    training_profile: str = PROFILE_MAZE_CREDIT_REPAIR,
) -> dict[str, Any]:
    profile = _profile_registry.normalize_training_profile(training_profile)
    maze_profile = _profile_registry.is_maze_profile(profile)
    closed_loop = profile == PROFILE_MAZE_CLOSED_LOOP_V3
    instant_command = _profile_registry.is_instant_profile(profile)
    profile_spec = _profile_registry.get_training_profile(profile)
    instant_repair = profile_spec.trainable and profile_spec.instant_command
    legacy_maze = profile == PROFILE_MAZE_CREDIT_REPAIR
    stuck = normalize_stuck_reset_contract(stuck_reset)
    serialized_stuck = dict(stuck)
    if not (closed_loop or instant_command):
        for name in (
            "schedule_enabled",
            "initial_confirmation_s",
            "activation_delay_s",
            "tighten_after_s",
            "resume_offset_s",
        ):
            serialized_stuck.pop(name, None)
    contract: dict[str, Any] = {}
    contract.update(
        _training_contract_part_1(
            profile,
            maze_profile,
            closed_loop,
            instant_command,
            instant_repair,
            legacy_maze,
            profile_spec,
            serialized_stuck,
        )
    )
    contract.update(
        _training_contract_part_2(
            profile,
            maze_profile,
            closed_loop,
            instant_command,
            instant_repair,
            legacy_maze,
            profile_spec,
            serialized_stuck,
        )
    )
    contract.update(
        _training_contract_part_5(
            profile,
            maze_profile,
            closed_loop,
            instant_command,
            instant_repair,
            legacy_maze,
            profile_spec,
            serialized_stuck,
        )
    )
    contract.update(
        _training_contract_part_6(
            profile,
            maze_profile,
            closed_loop,
            instant_command,
            instant_repair,
            legacy_maze,
            profile_spec,
            serialized_stuck,
        )
    )
    return contract


def command_contract(
    training_profile: str = LEGACY_COMPATIBILITY_DEFAULT, *, mode: str | None = None
) -> dict[str, Any]:
    spec = get_training_profile(training_profile, mode=mode)
    return _build_command_contract(spec.name)


def reward_contract(
    stuck_reset: dict[str, Any] | str | None = None,
    training_profile: str = LEGACY_COMPATIBILITY_DEFAULT,
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    stuck_reset, training_profile = _normalize_profile_argument(
        stuck_reset, training_profile
    )
    spec = get_training_profile(training_profile, mode=mode)
    return _build_reward_contract(stuck_reset, spec.name)


def training_contract(
    stuck_reset: dict[str, Any] | str | None = None,
    training_profile: str = LEGACY_COMPATIBILITY_DEFAULT,
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    stuck_reset, training_profile = _normalize_profile_argument(
        stuck_reset, training_profile
    )
    spec = get_training_profile(training_profile, mode=mode)
    return _build_training_contract(stuck_reset, spec.name)


def contract_metadata(
    stuck_reset: dict[str, Any] | str | None = None,
    training_profile: str = LEGACY_COMPATIBILITY_DEFAULT,
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    stuck_reset, training_profile = _normalize_profile_argument(
        stuck_reset, training_profile
    )
    spec = get_training_profile(training_profile, mode=mode)
    command = command_contract(spec.name)
    reward = reward_contract(stuck_reset, spec.name)
    training = training_contract(stuck_reset, spec.name)
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
    *,
    low_level_digest: str,
    feedback_digest: str,
    training_profile: str = PROFILE_FULL_TRACK,
) -> dict[str, Any]:
    profile = _profile_registry.normalize_training_profile(training_profile)
    command = command_contract(profile)
    instant_command = _profile_registry.is_instant_profile(profile)
    response_capability = list(p2_contract.RESPONSE_CAPABILITY_PROFILE15)
    response_capability[3] = P4_MAX_VX
    response_capability[4] = P4_MAX_ABS_WZ
    response_capability[7] = P4_MAX_ABS_VY
    return {
        "version": (
            INSTANT_ADAPTER_RECORD_CONTRACT_VERSION
            if instant_command
            else ADAPTER_RECORD_CONTRACT_VERSION
        ),
        "schema": "response_aux30_axis_specific_stuck_labels_v3",
        "low_level_digest": str(low_level_digest),
        "feedback_digest": str(feedback_digest),
        "capability_digest": stable_digest(response_capability),
        "response_capability_profile15": response_capability,
        "action_mapper": ACTION_MAPPER_VERSION,
        "observation_layout": "response_obs45_profile16",
        "label_layout": "velocity_horizons_0p2_0p6_1p0_pose_stuck",
        **(
            {
                "command_contract_digest": stable_digest(command),
                "command_transition_mode": command["command_transition_mode"],
                "command_hold_frames": int(command["hold_frames"]),
            }
            if instant_command
            else {}
        ),
    }


canonical_contracts = contract_metadata
