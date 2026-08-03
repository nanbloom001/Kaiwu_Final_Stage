#!/usr/bin/env python3
"""Versioned contracts for the P4 Track robustness stages."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch

from agent_ppo.feature import p2_contract


RUN_NAME = "p4maze2h-attack"
STAGE_NAME = "p4_nav_ppo"
STAGE_TYPE = "p4_nav_ppo"
TRAINING_HOURS = 2.0
TARGET_EFFECTIVE_SECONDS = 7_200.0
DIAGNOSTIC_SECONDS = 600.0
SCHEDULE_BOUNDARIES_SECONDS = (600.0, 5_400.0)

LEGACY_ACTION_MAPPER_VERSION = "p2_legacy_action_mapper_v1"
ACTION_MAPPER_VERSION = "p4_capability_action_mapper_v1"
GOAL_BELIEF_VERSION = "p4_goal_belief_v3_metric_raw"
CAMERA_CONTRACT_VERSION = "p4_shared_camera_v2_normalized_clip"
ADAPTER_RECORD_CONTRACT_VERSION = "p4_adapter_record_v1"
SAFETY_REWARD_RAMP_VERSION = "p4_maze_safety_reward_ramp_v1"
CHECKPOINT_CONTRACT_VERSION = "p4_maze_soft_cruise_v2_training_clock"
WORKER_WIRE_VERSION = "p4_worker_wire_v2"
STUCK_RESET_CONTRACT_VERSION = "p4_stuck_reset_v1"
# Training-only transport appended after the stable P3 493-column wire.
# These fields never enter either learned network and are absent from eval.
P4_WORKER_EXTRA_DIM = 14
P4_PRIVILEGED_WIRE_DIM = 507
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

SPEED_TIERS_ENABLED = False
SPEED_CAPS_MPS = (1.00,)
SPEED_CAP_PROBABILITIES = (1.00,)
SPEED_CAP_LABELS = ("full",)
SOFT_CRUISE_MIN_VX = 0.60
SOFT_CRUISE_MAX_VX = 0.75
SOFT_CRUISE_LOW_WEIGHT = -0.03
SOFT_CRUISE_HIGH_WEIGHT = -0.02

P4_MAX_ABS_VY = 0.30
P4_MAX_ABS_WZ = 0.90
P4_MAX_VX = 1.00
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

STUCK_RESET_TERMINAL_PENALTY = -6.0
STUCK_RESET_DEFAULTS = {
    "enabled": True,
    "mode": "shadow",
    "confirmation_s": 10.0,
    "radius_m": 0.30,
    "min_goal_distance_m": 0.80,
    "body_collision_force_n": 30.0,
    "wall_evidence_latch_s": 2.0,
    "episode_grace_s": 5.0,
    "push_grace_s": 2.0,
    "terminal_penalty": STUCK_RESET_TERMINAL_PENALTY,
}


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
        "confirmation_s": float(merged["confirmation_s"]),
        "radius_m": float(merged["radius_m"]),
        "min_goal_distance_m": float(merged["min_goal_distance_m"]),
        "body_collision_force_n": float(merged["body_collision_force_n"]),
        "wall_evidence_latch_s": float(merged["wall_evidence_latch_s"]),
        "episode_grace_s": float(merged["episode_grace_s"]),
        "push_grace_s": float(merged["push_grace_s"]),
        "terminal_penalty": float(merged["terminal_penalty"]),
    }
    numeric = tuple(value for key, value in result.items() if key not in {"enabled", "mode"})
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("P4 stuck-reset numeric fields must be finite")
    if any(
        result[key] < 0.0
        for key in (
            "confirmation_s",
            "radius_m",
            "min_goal_distance_m",
            "body_collision_force_n",
            "wall_evidence_latch_s",
            "episode_grace_s",
            "push_grace_s",
        )
    ):
        raise ValueError("P4 stuck-reset durations, distances and force must be non-negative")
    if result["terminal_penalty"] > 0.0:
        raise ValueError("P4 stuck-reset terminal_penalty must be non-positive")
    return result

YAW_WINDOW_TICKS = 5
YAW_EXEC_WEIGHT = -0.012
YAW_TRUE_WEIGHT = -0.008
YAW_TOTAL_FLOOR = -0.020
SAFETY_GROUP_FLOOR = -0.050

PREDICTIVE_RAW_FLOOR = -0.0225
MISSED_SAFE_RAW_FLOOR = -0.0200

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
    "speed_tier",
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
    "scanner_valid_share",
    "safety_bce",
    "safety_head_risk_left",
    "safety_head_risk_center",
    "safety_head_risk_right",
    "head_correct_actor_wrong",
    "risk_decel_policy_vx",
    "risk_decel_limited_vx",
    "risk_no_deceleration",
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
    "reward_stuck_reset",
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


def soft_cruise_penalty(
    policy_target_cmd3: torch.Tensor,
    safe3: torch.Tensor,
    teacher_valid: torch.Tensor,
    goal_freshness: torch.Tensor,
    terminal: torch.Tensor,
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
    penalty = torch.where(terminal.reshape(-1).bool(), torch.zeros_like(penalty), penalty)
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


def yaw_cancellation(x: torch.Tensor, *, dt_s: float = p2_contract.NAV_DT_S) -> torch.Tensor:
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
    floor: float = SAFETY_GROUP_FLOOR,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Proportionally cap a group of non-positive reward terms."""
    raw_sum = predictive_raw + missed_raw + yaw_raw
    magnitude = torch.clamp(-raw_sum, min=0.0)
    scale = torch.minimum(
        torch.ones_like(magnitude),
        torch.full_like(magnitude, abs(float(floor))) / magnitude.clamp_min(1.0e-9),
    )
    return (
        predictive_raw * scale,
        missed_raw * scale,
        yaw_raw * scale,
    ), scale


def push_phase_config(session_effective_seconds: float) -> dict[str, float | str | bool]:
    seconds = max(0.0, float(session_effective_seconds))
    if seconds < 7_200.0:
        return {
            "name": "pnavwarm",
            "active": False,
            "max_velocity_xy_m_s": 0.0,
            "min_interval_s": 30.0,
            "max_interval_s": 45.0,
        }
    if seconds < 21_600.0:
        return {
            "name": "pnavrobust",
            "active": True,
            "max_velocity_xy_m_s": 0.04,
            "min_interval_s": 30.0,
            "max_interval_s": 45.0,
        }
    return {
        "name": "pnavfull",
        "active": True,
        "max_velocity_xy_m_s": 0.05,
        "min_interval_s": 25.0,
        "max_interval_s": 40.0,
    }


def camera_mix(session_effective_seconds: float) -> dict[str, float]:
    seconds = max(0.0, float(session_effective_seconds))
    if seconds < 1_800.0:
        return {"nominal": 1.0, "light": 0.0, "delayed": 0.0, "severe": 0.0}
    if seconds < 7_200.0:
        return {"nominal": 0.70, "light": 0.25, "delayed": 0.05, "severe": 0.0}
    if seconds < 21_600.0:
        return {"nominal": 0.55, "light": 0.25, "delayed": 0.15, "severe": 0.05}
    return {"nominal": 0.50, "light": 0.35, "delayed": 0.10, "severe": 0.05}


def training_schedule(
    session_effective_seconds: float,
    *,
    branch: str = "actor_attack",
) -> dict[str, float | str | bool]:
    seconds = max(0.0, float(session_effective_seconds))
    branch = str(branch or "actor_attack")
    if branch == "auto" and seconds < DIAGNOSTIC_SECONDS:
        return {
            "phase": "mazediag",
            "training_branch": "auto",
            "navigation_multiplier": 0.0,
            "actor_multiplier": 0.0,
            "critic_multiplier": 0.0,
            "safety_head_multiplier": 0.0,
            "adapter_multiplier": 0.0,
            "reward_multiplier": 0.0,
            "goal_fault_multiplier": 0.0,
            "camera_aux_ratio": 0.0,
            "cruise_multiplier": 0.0,
            "entropy_coefficient": 0.0,
        }
    if branch == "auto":
        branch = "actor_attack"
    if branch not in {"actor_attack", "visual_recovery"}:
        branch = "actor_attack"
    if branch == "visual_recovery":
        if seconds < 1_800.0:
            return {
                "phase": "mazeprobe",
                "training_branch": branch,
                "navigation_multiplier": 0.75,
                "actor_multiplier": 0.15,
                "critic_multiplier": 1.0,
                "safety_head_multiplier": 1.5,
                "adapter_multiplier": 0.5,
                "reward_multiplier": 1.0,
                "goal_fault_multiplier": 0.25,
                "camera_aux_ratio": 0.02,
                "cruise_multiplier": 0.0,
                "entropy_coefficient": 0.006,
            }
        if seconds < 5_400.0:
            cruise = min(1.0, max(0.0, (seconds - 1_800.0) / 1_800.0))
            return {
                "phase": "mazefull",
                "training_branch": branch,
                "navigation_multiplier": 0.60,
                "actor_multiplier": 0.35,
                "critic_multiplier": 0.8,
                "safety_head_multiplier": 1.0,
                "adapter_multiplier": 0.5,
                "reward_multiplier": 1.0,
                "goal_fault_multiplier": 0.25,
                "camera_aux_ratio": 0.015,
                "cruise_multiplier": cruise,
                "entropy_coefficient": 0.005,
            }
        return {
            "phase": "mazefinal",
            "training_branch": branch,
            "navigation_multiplier": 0.30,
            "actor_multiplier": 0.30,
            "critic_multiplier": 0.5,
            "safety_head_multiplier": 0.5,
            "adapter_multiplier": 0.25,
            "reward_multiplier": 1.0,
            "goal_fault_multiplier": 0.25,
            "camera_aux_ratio": 0.01,
            "cruise_multiplier": 1.0,
            "entropy_coefficient": 0.004,
        }
    if seconds < 600.0:
        return {
            "phase": "mazeprobe",
            "training_branch": branch,
            "navigation_multiplier": 0.25,
            "actor_multiplier": 0.35,
            "critic_multiplier": 1.0,
            "safety_head_multiplier": 1.0,
            "adapter_multiplier": 0.5,
            "reward_multiplier": 1.0,
            "goal_fault_multiplier": 0.25,
            "camera_aux_ratio": 0.01,
            "cruise_multiplier": seconds / 600.0,
            "entropy_coefficient": 0.006,
        }
    if seconds < 5_400.0:
        return {
            "phase": "mazefull",
            "training_branch": branch,
            "navigation_multiplier": 0.60,
            "actor_multiplier": 0.75,
            "critic_multiplier": 0.8,
            "safety_head_multiplier": 1.0,
            "adapter_multiplier": 0.5,
            "reward_multiplier": 1.0,
            "goal_fault_multiplier": 0.25,
            "camera_aux_ratio": 0.015,
            "cruise_multiplier": 1.0,
            "entropy_coefficient": 0.005,
        }
    return {
        "phase": "mazefinal",
        "training_branch": branch,
        "navigation_multiplier": 0.30,
        "actor_multiplier": 0.40,
        "critic_multiplier": 0.5,
        "safety_head_multiplier": 0.25,
        "adapter_multiplier": 0.25,
        "reward_multiplier": 1.0,
        "goal_fault_multiplier": 0.25,
        "camera_aux_ratio": 0.01,
        "cruise_multiplier": 1.0,
        "entropy_coefficient": 0.004,
    }


def command_contract() -> dict[str, Any]:
    return {
        "version": "p4_maze_soft_cruise_command_v1",
        "mapper_version": ACTION_MAPPER_VERSION,
        "legacy_mapper_version": LEGACY_ACTION_MAPPER_VERSION,
        "normalized_action": "unchanged_tanh_gaussian_v2",
        "mapped_ranges": {"vx": [0.0, 1.0], "vy": [-0.30, 0.30], "wz": [-0.90, 0.90]},
        "policy_target_vx": [0.0, 1.0],
        "limited_target_vx": "min(policy_target, stale_goal_cap, safety_cap)",
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
        "speed_tiers_enabled": SPEED_TIERS_ENABLED,
        "soft_cruise": {
            "preferred_vx": [SOFT_CRUISE_MIN_VX, SOFT_CRUISE_MAX_VX],
            "low_weight": SOFT_CRUISE_LOW_WEIGHT,
            "high_weight": SOFT_CRUISE_HIGH_WEIGHT,
            "clear_factor": "teacher_valid*goal_freshness*center_safe*center_safe/best_safe",
        },
        "slew_semantics": p2_contract.command_contract()["slew_semantics"],
    }


def reward_contract(
    stuck_reset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stuck = normalize_stuck_reset_contract(stuck_reset)
    return {
        "version": "p4_maze_reward_v1_soft_cruise",
        "inherits": p2_contract.reward_contract()["version"],
        "new_terms": {
            "predictive_collision_raw_floor": PREDICTIVE_RAW_FLOOR,
            "missed_safe_direction_raw_floor": MISSED_SAFE_RAW_FLOOR,
            "yaw_exec_weight": YAW_EXEC_WEIGHT,
            "yaw_true_weight": YAW_TRUE_WEIGHT,
            "yaw_total_floor": YAW_TOTAL_FLOOR,
            "safety_group_floor": SAFETY_GROUP_FLOOR,
            "cap_semantics": "proportional_no_hidden_adjustment",
            "confirmed_wall_stuck_reset": stuck["terminal_penalty"],
            "soft_cruise": {
                "preferred_vx": [SOFT_CRUISE_MIN_VX, SOFT_CRUISE_MAX_VX],
                "low_weight": SOFT_CRUISE_LOW_WEIGHT,
                "high_weight": SOFT_CRUISE_HIGH_WEIGHT,
            },
        },
        "goal_truth_consumers": ["critic", "reward", "terminal", "scorer"],
        "goal_belief_consumers": ["actor", "speed_cap"],
    }


def training_contract(
    stuck_reset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stuck = normalize_stuck_reset_contract(stuck_reset)
    return {
        "version": CHECKPOINT_CONTRACT_VERSION,
        "run_name": RUN_NAME,
        "training_hours": TRAINING_HOURS,
        "target_effective_seconds": int(TARGET_EFFECTIVE_SECONDS),
        "diagnostic_seconds": int(DIAGNOSTIC_SECONDS),
        "clock_semantics": {
            "diagnostic": "wall_seconds_before_training_not_counted_in_session",
            "session_effective_seconds": "gradient_training_seconds_only",
            "session_wall_seconds": "diagnostic_plus_training",
        },
        "schedule_boundaries_seconds": list(SCHEDULE_BOUNDARIES_SECONDS),
        "safety_reward_ramp": {
            "version": SAFETY_REWARD_RAMP_VERSION,
            "segments": [
                {"seconds": [0, 7_200], "multiplier": [1.0, 1.0]},
            ],
        },
        "goal_fault_ramp": {
            "semantics": "independent_from_safety_reward_multiplier",
            "segments": [
                {"seconds": [0, 7_200], "multiplier": [0.25, 0.25]},
            ],
        },
        "rollout_nav_ticks": 32,
        "tbptt_nav_ticks": 16,
        "frozen_low_level": ["cnn", "lstm", "actor", "std", "critic"],
        "trainable": ["navigation_encoder", "high_actor_lstm", "high_critic", "safety_head", "response_adapter"],
        "goal_belief_version": GOAL_BELIEF_VERSION,
        "camera_contract_version": CAMERA_CONTRACT_VERSION,
        "worker_wire_version": WORKER_WIRE_VERSION,
        "worker_wire_dim": P4_PRIVILEGED_WIRE_DIM,
        "stuck_reset_contract_version": STUCK_RESET_CONTRACT_VERSION,
        "stuck_reset": stuck,
        "adapter_record_contract_version": ADAPTER_RECORD_CONTRACT_VERSION,
        "maze_only": True,
        "soft_cruise": command_contract()["soft_cruise"],
        "exact_resume": "p4_maze_soft_cruise_v2_training_clock_only",
    }


def contract_metadata(
    stuck_reset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command = command_contract()
    reward = reward_contract(stuck_reset)
    training = training_contract(stuck_reset)
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
        "schema": "response_aux30_future_labels_v2",
        "low_level_digest": str(low_level_digest),
        "feedback_digest": str(feedback_digest),
        "capability_digest": stable_digest(response_capability),
        "response_capability_profile15": response_capability,
        "action_mapper": ACTION_MAPPER_VERSION,
        "observation_layout": "response_obs45_profile16",
        "label_layout": "velocity_horizons_0p2_0p6_1p0_pose_stuck",
    }
