#!/usr/bin/env python3
"""Stable contracts for P2 continuous high-level Track navigation."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch


RUN_NAME = "p2nav2hsafedir"
STAGE_NAME = "p2_nav_ppo"
TRAINING_HOURS = 2.0
CONTROL_DT_S = 0.02
NAV_PERIOD_FRAMES = 10
NAV_DT_S = CONTROL_DT_S * NAV_PERIOD_FRAMES
NAV_ROLLOUT_TICKS = 32
TBPTT_SEQUENCE_LENGTH = 16

DEPTH_HEIGHT = 180
DEPTH_WIDTH = 320
DEPTH_CHANNELS = 1
DEPTH_DIM = DEPTH_HEIGHT * DEPTH_WIDTH * DEPTH_CHANNELS
NAV_FEATURE_DIM = 32
NAV_NONVISUAL_DIM = 36
RESPONSE_PROFILE_DIM = 16
ADAPTER_CONFIDENCE_DIM = 1
ACTOR_INPUT_DIM = 85
CRITIC_OBS_DIM = 323
RESPONSE_AUX_DIM = 30
GAIT_DIAGNOSTIC_DIM = 25
DIAGNOSTIC_AUX_DIM = 32
WORKER_AUX_DIM = RESPONSE_AUX_DIM + DIAGNOSTIC_AUX_DIM
PRIVILEGED_WIRE_DIM = CRITIC_OBS_DIM + WORKER_AUX_DIM
CRITIC_INPUT_DIM = 341
ACTION_DIM = 3
TERRAIN_NUM_COLUMNS = 20
TRACK_SEGMENT_METRIC_LABELS = ("slope_inv", "stairs_inv", "maze")
TRACK_TERRAIN_TO_METRIC_LABEL = {
    "pyramid_slope_inv": "slope_inv",
    "pyramid_stairs_inv": "stairs_inv",
    "open_entry_maze": "maze",
}


def canonical_track_segment_labels(sub_terrains) -> tuple[str, ...]:
    """Map configured Track terrain names to stable metric bucket labels."""
    if sub_terrains is None or sub_terrains == ():
        return TRACK_SEGMENT_METRIC_LABELS
    if not isinstance(sub_terrains, (list, tuple)) or not sub_terrains:
        raise ValueError("Track sub_terrains must be a non-empty sequence")
    labels = []
    for terrain in sub_terrains:
        terrain_name = str(terrain)
        label = TRACK_TERRAIN_TO_METRIC_LABEL.get(terrain_name)
        if label is None:
            raise ValueError(
                f"unsupported navigation Track terrain {terrain_name!r}; "
                f"supported={sorted(TRACK_TERRAIN_TO_METRIC_LABEL)}"
            )
        labels.append(label)
    return tuple(labels)


def track_segment_metric_indices(
    physical_segment: torch.Tensor,
    segment_labels: tuple[str, ...] | list[str],
) -> torch.Tensor:
    """Translate physical Track segment indices into stable metric bucket IDs."""
    labels = tuple(str(label) for label in segment_labels)
    result = torch.full_like(physical_segment.round().long(), -1)
    for physical_index, label in enumerate(labels):
        if label not in TRACK_SEGMENT_METRIC_LABELS:
            continue
        metric_index = TRACK_SEGMENT_METRIC_LABELS.index(label)
        result[physical_segment.round().long() == physical_index] = metric_index
    return result

# The first 30 worker-owned slots remain the stable ResponseAdapter contract.
# P2-only diagnostics are appended so P1.5 completed records stay loadable.
GAIT_DUTY_SLICE = slice(30, 34)
GAIT_MEAN_SWING_SLICE = slice(34, 38)
GAIT_MAX_AIR_SLICE = slice(38, 42)
GAIT_PROLONGED_RATIO_SLICE = slice(42, 46)
GAIT_STEP_FREQUENCY_SLICE = slice(46, 50)
GAIT_SLIP_SPEED_SLICE = slice(50, 54)
GAIT_VALID_INDEX = 54
PRE_STEP_TERRAIN_TYPE_INDEX = 55
PRE_STEP_TERRAIN_LEVEL_INDEX = 56
PRE_STEP_GOAL_DISTANCE_INDEX = 57
BODY_COLLISION_FORCE_INDEX = 58
CURRENT_SEGMENT_INDEX = 59
GAIT_SENSOR_MAPPING_VALID_INDEX = 60
BODY_COLLISION_MAPPING_VALID_INDEX = 61

# The Arena eval workflow forwards only the policy observation. P2 does not
# consume height_scan256 in either the frozen low level or the high-level
# policy, so eval reuses a small prefix of that stable 256-slot region to carry
# worker-owned response/reset fields without changing the 57905-D contract.
EVAL_AUX_POLICY_START = 45
EVAL_AUX_POLICY_END = EVAL_AUX_POLICY_START + RESPONSE_AUX_DIM
EVAL_AUX_MARKER_INDEX = EVAL_AUX_POLICY_END
EVAL_AUX_MARKER_VALUE = 27182.0

TRUSTED_CORE = {
    "vx": (0.0, 1.0),
    "vy": (-0.20, 0.20),
    "wz": (-0.8, 0.8),
}
EXPLORATION_HARD_BOUNDARY = {
    "vx": (0.0, 1.25),
    "vy": (-0.40, 0.40),
    "wz": (-1.0, 1.0),
}
NAV_CAPABILITY_PROFILE15_LAYOUT = (
    "active_mask_vx",
    "active_mask_vy",
    "active_mask_wz",
    "command_min_vx",
    "command_min_vy",
    "command_min_wz",
    "command_max_vx",
    "command_max_vy",
    "command_max_wz",
    "slew_up_vx",
    "slew_up_vy",
    "slew_up_wz",
    "slew_down_vx",
    "slew_down_vy",
    "slew_down_wz",
)
NAV_CAPABILITY_PROFILE15 = (
    1.0, 1.0, 1.0,       # active_mask3
    0.0, -0.40, -1.0,    # cmd_min3
    1.25, 0.40, 1.0,     # cmd_max3
    0.30, 0.30, 1.00,    # slew_up3
    0.30, 0.60, 2.50,    # slew_down3
)

# ResponseAdapter was pretrained with P1.5's piecewise-union layout. Keep that
# layout stable while describing P2's joint three-axis command domain.
RESPONSE_CAPABILITY_PROFILE15 = (
    1.0,  # piecewise_union_enabled
    1.0,  # main_enabled
    0.0,  # main_vx_min
    1.25, # main_vx_max
    1.0,  # main_abs_wz_max
    0.0,  # main_vy_fixed (legacy piecewise branch)
    1.0,  # vy_specialty_enabled
    0.3,  # vy_specialty_abs_vy_max
    0.0,  # vy_specialty_vx_fixed
    0.0,  # vy_specialty_wz_fixed
    0.0,  # source_replay_enabled in the live command domain
    0.0,  # source_replay_vx_min
    0.0,  # source_replay_vx_max
    0.0,  # source_replay_abs_vy_max
    0.0,  # source_replay_abs_wz_max
)

INITIAL_VX_MPS = 0.30
INITIAL_VX_NORMALIZED = 2.0 * INITIAL_VX_MPS / 1.25 - 1.0
INITIAL_VX_PRE_TANH = math.atanh(INITIAL_VX_NORMALIZED)
INITIAL_LOG_STD = -0.7
INITIAL_VY_LOG_STD = -1.1
LOG_STD_MIN = -2.0
LOG_STD_MAX = 0.0

GAMMA_NAV = 0.995
GAMMA_FRAME = GAMMA_NAV ** (1.0 / NAV_PERIOD_FRAMES)
GAE_LAMBDA = 0.95

SAFETY_WARM_END_SECONDS = 30.0 * 60.0
SAFETY_STABILIZE_SECONDS = 1.5 * 3600.0
CNN_LAYER_LRS = {
    "conv1": 3.0e-6,
    "conv2": 1.0e-5,
    "conv3": 3.0e-5,
    "fc": 3.0e-5,
}
ACTOR_LR = 3.0e-4
CRITIC_LR = 3.0e-4
ADAPTER_LR = 3.0e-5

TARGET_STABILITY_TOLERANCE = (0.05, 0.02, 0.08)
PARENT_RESPONSE_REPLAY_RATIO = 0.25
NEUTRAL_PROFILE_PROBABILITY = 0.10

SUCCESS_IMPULSE = 50.0
FAILURE_IMPULSE = -25.0
TIMEOUT_IMPULSE = -22.5
FRONTIER_POTENTIAL_WEIGHT = 2.0
TIME_COST_PER_TICK = -0.02
COMMAND_RATE_WEIGHT = -0.008
COMMAND_RATE_AXIS_WEIGHTS = (1.0, 0.5, 1.5)
TRACKING_ERROR_WEIGHT = -0.004
TRACKING_ERROR_AXIS_WEIGHTS = (1.0, 0.75, 1.5)
COMMAND_NORMALIZATION = (1.25, 0.40, 1.0)
CRAWL_BODY_RADIUS_M = 0.30
CRAWL_STABLE_MIN_MPS = 0.22
CRAWL_WEIGHT = -0.03
GAIT_WINDOW_SECONDS = 1.5
GAIT_PROLONGED_AIR_SECONDS = 0.60
GAIT_PENALTY_CAP = 0.0
GAIT_BASELINE_VERSION = "p2_gait_parent_baseline_v2"
GAIT_BASELINE_PARENT_MODEL_ID = "37953"
GAIT_BASELINE_PARENT_LABEL = "p15resp8h-r1_37953-F"
GAIT_BASELINE_THRESHOLDS = {
    "duty_imbalance": 0.30,
    "swing_imbalance_s": 0.20,
    "prolonged_air_ratio": 0.10,
    "step_frequency_imbalance_hz": 1.50,
}

BODY_COLLISION_SOFT_FORCE_N = 30.0
BODY_COLLISION_HARD_FORCE_N = 150.0
BODY_COLLISION_ONSET_BASE = -0.08
BODY_COLLISION_ONSET_SEVERITY = -0.12
BODY_COLLISION_PERSISTENT = -0.03
FRONTIER_WINDOW_TICKS = 15
FRONTIER_MIN_ADVANCE_M = 0.03
STAGNATION_INITIAL_PENALTY = -0.015
STAGNATION_RAMP_PER_TICK = -0.003
STAGNATION_PENALTY_CAP = -0.06
PREDICTIVE_COLLISION_WEIGHT = -0.02
PREDICTIVE_COLLISION_MAX_DEPTH_M = 5.0
PREDICTIVE_COLLISION_BODY_MARGIN_M = 0.30
PREDICTIVE_COLLISION_BRAKE_ACCEL_MPS2 = 0.80
PREDICTIVE_COLLISION_RISK_BAND_M = 0.40
PREDICTIVE_COLLISION_ACTIVE_SPEED_MPS = 0.05
PREDICTIVE_COLLISION_LOOKAHEAD_S = 0.80
PREDICTIVE_COLLISION_YAW_RADIUS_M = 0.30
PREDICTIVE_COLLISION_NEAR_MARGIN_M = 0.20
PREDICTIVE_COLLISION_GAP_MARGIN_M = 0.15
PREDICTIVE_COLLISION_NEAR_SCALE_M = 0.15
PREDICTIVE_COLLISION_FLATNESS_SCALE_M = 0.35
PREDICTIVE_COLLISION_QUANTILE = 0.20
PREDICTIVE_COLLISION_VERTICAL_BANDS = {
    "upper": (25, 60),
    "middle": (60, 100),
    "lower": (100, 145),
}
PREDICTIVE_COLLISION_HORIZONTAL_SECTORS = {
    "left": (48, 144),
    "center": (112, 208),
    "right": (176, 272),
}
PREDICTIVE_COLLISION_SECTOR_CENTERS_DEG = (35.0, 0.0, -35.0)
PREDICTIVE_COLLISION_SECTOR_WIDTH_DEG = 25.0
SAFETY_BCE_WEIGHT = 0.03
SAFETY_HEAD_LR = 3.0e-4
SAFE_DIRECTION_MAX_WEIGHT = 0.03
SAFE_DIRECTION_INITIAL_WEIGHT = 0.015
SAFE_DIRECTION_MIN_SPEED_MPS = 0.05
SAFE_DIRECTION_MIN_BEST_SAFE = 0.35
SAFE_DIRECTION_GAP_MARGIN = 0.15
SAFETY_SCANNER_WELL_FORMED_RATIO = 0.95
SAFETY_SCANNER_FINITE_HIT_RATIO = 0.80
SAFETY_HEIGHT_JUMP_FREE_M = 0.20
SAFETY_HEIGHT_JUMP_SCALE_M = 0.15
# Grid rows are ordered from body -y (right) to +y (left). Public teacher
# vectors remain [left, center, right].
SAFETY_HEIGHT_SECTORS = ((9, 16), (4, 12), (0, 7))
SAFETY_HEIGHT_FORWARD_COLS = 12
SAFETY_HEIGHT_FINITE_RATIO_MIN = 0.95
SAFETY_HEIGHT_MIN_VALID_DIFFS = 32

LEGACY_PREDICTIVE_COLLISION_BODY_MARGIN_M = 0.35
LEGACY_PREDICTIVE_COLLISION_BRAKE_ACCEL_MPS2 = 0.60
LEGACY_PREDICTIVE_COLLISION_RISK_BAND_M = 0.45
LEGACY_PREDICTIVE_COLLISION_ROI = (30, 105, 96, 224)
LEGACY_PREDICTIVE_COLLISION_QUANTILE = 0.10
def vy_action_limits() -> dict[str, float]:
    """Return the fixed three-axis command domain used from the first rollout."""
    return {"trusted_abs_vy": 0.20, "hard_abs_vy": 0.40}


def training_schedule(session_effective_seconds: float) -> dict[str, float | str | bool]:
    """Return the two-hour safe-direction optimizer contract."""
    seconds = max(0.0, float(session_effective_seconds))
    if seconds < SAFETY_WARM_END_SECONDS:
        return {
            "phase": "safewarm",
            "cnn_unfrozen": True,
            "navigation_multiplier": 0.15,
            "actor_trunk_multiplier": 0.10,
            "actor_multiplier": 0.10,
            "vy_actor_multiplier": 0.10,
            "safety_head_multiplier": 1.0,
            "critic_multiplier": 0.30,
            "adapter_multiplier": 0.5,
            "entropy_coefficient": 0.006,
        }
    if seconds < SAFETY_STABILIZE_SECONDS:
        return {
            "phase": "safefull",
            "cnn_unfrozen": True,
            "navigation_multiplier": 0.25,
            "actor_trunk_multiplier": 0.15,
            "actor_multiplier": 0.15,
            "vy_actor_multiplier": 0.15,
            "safety_head_multiplier": 1.0,
            "critic_multiplier": 0.30,
            "adapter_multiplier": 0.5,
            "entropy_coefficient": 0.005,
        }
    return {
        "phase": "safestable",
        "optimizer_phase": "stabilize",
        "cnn_unfrozen": True,
        "navigation_multiplier": 0.15,
        "actor_trunk_multiplier": 0.10,
        "actor_multiplier": 0.10,
        "vy_actor_multiplier": 0.10,
        "safety_head_multiplier": 0.5,
        "critic_multiplier": 0.20,
        "adapter_multiplier": 0.5,
        "entropy_coefficient": 0.004,
    }


def safe_direction_weight(session_effective_seconds: float) -> float:
    seconds = max(0.0, min(float(session_effective_seconds), TRAINING_HOURS * 3600.0))
    if seconds <= SAFETY_WARM_END_SECONDS:
        return SAFE_DIRECTION_INITIAL_WEIGHT * seconds / SAFETY_WARM_END_SECONDS
    remaining = TRAINING_HOURS * 3600.0 - SAFETY_WARM_END_SECONDS
    ratio = (seconds - SAFETY_WARM_END_SECONDS) / max(remaining, 1.0)
    return SAFE_DIRECTION_INITIAL_WEIGHT + ratio * (
        SAFE_DIRECTION_MAX_WEIGHT - SAFE_DIRECTION_INITIAL_WEIGHT
    )


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def pack_eval_response_aux(
    policy_obs: torch.Tensor, response_aux: torch.Tensor
) -> torch.Tensor:
    """Embed aux30 into the unused P2 scan prefix for single-stream eval."""
    if policy_obs.ndim != 2 or policy_obs.shape[1] <= EVAL_AUX_MARKER_INDEX:
        raise ValueError(
            "P2 eval policy observation must be rank-2 and contain the scan region"
        )
    if response_aux.ndim != 2 or response_aux.shape != (
        policy_obs.shape[0],
        RESPONSE_AUX_DIM,
    ):
        raise ValueError(
            "P2 eval response aux must match policy batch and have 30 columns"
        )
    packed = policy_obs.clone()
    packed[:, EVAL_AUX_POLICY_START:EVAL_AUX_POLICY_END] = response_aux.to(
        device=packed.device, dtype=packed.dtype
    )
    packed[:, EVAL_AUX_MARKER_INDEX] = EVAL_AUX_MARKER_VALUE
    return packed


def unpack_eval_response_aux(policy_obs: torch.Tensor) -> torch.Tensor:
    """Recover aux30 and reject a policy tensor that was not eval-packed."""
    if policy_obs.ndim != 2 or policy_obs.shape[1] <= EVAL_AUX_MARKER_INDEX:
        raise ValueError(
            "P2 eval policy observation must be rank-2 and contain the scan region"
        )
    marker = policy_obs[:, EVAL_AUX_MARKER_INDEX]
    expected = torch.full_like(marker, EVAL_AUX_MARKER_VALUE)
    if not bool(torch.isfinite(marker).all()) or not bool(
        torch.isclose(marker, expected, rtol=0.0, atol=0.5).all()
    ):
        raise RuntimeError(
            "P2 eval response transport marker is missing; worker policy aux30 "
            "packing is not active"
        )
    return policy_obs[:, EVAL_AUX_POLICY_START:EVAL_AUX_POLICY_END].clone()


def command_contract() -> dict[str, Any]:
    return {
        "version": "p2_continuous_command_v2",
        "action_dim": ACTION_DIM,
        "distribution": "diagonal_tanh_squashed_gaussian",
        "coordinates": ["vx", "vy", "wz"],
        "incremental_heads": {"main": ["vx", "wz"], "lateral": ["vy"]},
        "initial_log_std": {
            "vx_wz": INITIAL_LOG_STD,
            "vy": INITIAL_VY_LOG_STD,
        },
        "trusted_core": TRUSTED_CORE,
        "exploration_hard_boundary": EXPLORATION_HARD_BOUNDARY,
        "vy_action_domain": {
            "schedule": "fully_open_from_first_rollout",
            "trusted_abs_vy": 0.20,
            "hard_abs_vy": 0.40,
        },
        "target_hz": 5,
        "control_hz": 50,
        "nav_period_frames": NAV_PERIOD_FRAMES,
        "slew_semantics": {
            "vx": {"up": 0.30, "down_or_reverse": 0.30},
            "vy": {"up": 0.30, "down_or_reverse": 0.60},
            "wz": {"up": 1.00, "down_or_reverse": 2.50},
            "reversal": "release_to_zero_before_opposite_direction",
        },
        "nav_capability_profile15": {
            "layout": list(NAV_CAPABILITY_PROFILE15_LAYOUT),
            "values": list(NAV_CAPABILITY_PROFILE15),
        },
        "response_capability_profile15": {
            "layout": "p15_piecewise_union_profile_v1",
            "values": list(RESPONSE_CAPABILITY_PROFILE15),
            "joint_command_extension": {
                "coordinates": ["vx", "vy", "wz"],
                "trusted_abs_vy": 0.20,
                "hard_abs_vy": 0.40,
                "note": "profile15 layout stays backward-compatible; live target/exec carry joint commands",
            },
        },
    }


def confidence_contract() -> dict[str, Any]:
    return {
        "version": "p2_adapter_confidence_v2",
        "age_tau_s": 0.20,
        "sigma_floor": 0.10,
        "core_vx_max": 1.0,
        "hard_vx_max": 1.25,
        "core_abs_wz_max": 0.8,
        "hard_abs_wz_max": 1.0,
        "core_abs_vy_max": 0.20,
        "hard_abs_vy_max": 0.40,
        "hard_boundary_domain_factor": 0.5,
        "neutral_profile_probability": NEUTRAL_PROFILE_PROBABILITY,
    }


def reward_contract() -> dict[str, Any]:
    return {
        "version": "p2_track_reward_v9_safe_direction_teacher",
        "goal_reached_threshold_m": 0.6,
        "low_level_frame_terms": {
            "flat_orientation": -0.05,
            "energy": -5.0e-6,
            "undesired_contacts": 0.0,
            "track_lin_vel_xy": 0.0,
            "track_ang_vel_z": 0.0,
        },
        "high_level_tick_terms": {
            "frontier_potential": {
                "weight_per_meter": FRONTIER_POTENTIAL_WEIGHT,
                "discount": "gamma_frame^duration_frames",
                "terminal_potential": 0.0,
                "frontier": "episode_start_distance_minus_best_distance",
            },
            "success": SUCCESS_IMPULSE,
            "failure": FAILURE_IMPULSE,
            "timeout": TIMEOUT_IMPULSE,
            "time": TIME_COST_PER_TICK,
            "command_rate": {
                "weight": COMMAND_RATE_WEIGHT,
                "normalization": list(COMMAND_NORMALIZATION),
                "axis_weights": list(COMMAND_RATE_AXIS_WEIGHTS),
            },
            "tracking_error": {
                "weight": TRACKING_ERROR_WEIGHT,
                "normalization": list(COMMAND_NORMALIZATION),
                "axis_weights": list(TRACKING_ERROR_AXIS_WEIGHTS),
            },
            "crawl": CRAWL_WEIGHT,
            "gait_symmetry": {
                "weight": 0.0,
                "mode": "monitor_only",
            },
            "body_collision": {
                "soft_force_n": BODY_COLLISION_SOFT_FORCE_N,
                "hard_force_n": BODY_COLLISION_HARD_FORCE_N,
                "onset_base": BODY_COLLISION_ONSET_BASE,
                "onset_severity": BODY_COLLISION_ONSET_SEVERITY,
                "persistent": BODY_COLLISION_PERSISTENT,
            },
            "frontier_stagnation": {
                "window_ticks": FRONTIER_WINDOW_TICKS,
                "min_advance_m": FRONTIER_MIN_ADVANCE_M,
                "initial_penalty": STAGNATION_INITIAL_PENALTY,
                "ramp_per_tick": STAGNATION_RAMP_PER_TICK,
                "cap": STAGNATION_PENALTY_CAP,
            },
            "predictive_collision_risk": {
                "weight": PREDICTIVE_COLLISION_WEIGHT,
                "max_depth_m": PREDICTIVE_COLLISION_MAX_DEPTH_M,
                "body_margin_m": PREDICTIVE_COLLISION_BODY_MARGIN_M,
                "brake_acceleration_mps2": PREDICTIVE_COLLISION_BRAKE_ACCEL_MPS2,
                "risk_band_m": PREDICTIVE_COLLISION_RISK_BAND_M,
                "active_speed_mps": PREDICTIVE_COLLISION_ACTIVE_SPEED_MPS,
                "vertical_bands_y0_y1": {
                    name: list(bounds)
                    for name, bounds in PREDICTIVE_COLLISION_VERTICAL_BANDS.items()
                },
                "horizontal_sectors_x0_x1": {
                    name: list(bounds)
                    for name, bounds in PREDICTIVE_COLLISION_HORIZONTAL_SECTORS.items()
                },
                "sector_centers_deg": list(PREDICTIVE_COLLISION_SECTOR_CENTERS_DEG),
                "sector_width_deg": PREDICTIVE_COLLISION_SECTOR_WIDTH_DEG,
                "robust_quantile": PREDICTIVE_COLLISION_QUANTILE,
                "near_margin_m": PREDICTIVE_COLLISION_NEAR_MARGIN_M,
                "gap_margin_m": PREDICTIVE_COLLISION_GAP_MARGIN_M,
                "near_scale_m": PREDICTIVE_COLLISION_NEAR_SCALE_M,
                "flatness_scale_m": PREDICTIVE_COLLISION_FLATNESS_SCALE_M,
                "lookahead_s": PREDICTIVE_COLLISION_LOOKAHEAD_S,
                "yaw_radius_m": PREDICTIVE_COLLISION_YAW_RADIUS_M,
                "zero_depth_semantics": "invalid_or_out_of_range_treated_as_max_depth",
                "positive_clearance_reward": False,
                "legacy_risk": "shadow_diagnostic_only",
            },
            "missed_safe_direction": {
                "weight_schedule": {
                    "0s": 0.0,
                    "1800s": SAFE_DIRECTION_INITIAL_WEIGHT,
                    "7200s": SAFE_DIRECTION_MAX_WEIGHT,
                },
                "minimum_command_equivalent_speed_mps": SAFE_DIRECTION_MIN_SPEED_MPS,
                "minimum_best_safe": SAFE_DIRECTION_MIN_BEST_SAFE,
                "safe_gap_margin": SAFE_DIRECTION_GAP_MARGIN,
                "teacher": "scanner_wall_risk_times_height_continuity",
                "goal_or_heading_in_teacher": False,
                "positive_reward": False,
            },
        },
        "reward_inputs": [
            "simulation_goal_terminal_and_true_velocity",
            "deployment_available_normalized_depth",
            "target_and_exec_command",
            "validated_contact_and_gait_diagnostics",
        ],
        "adapter_in_reward": False,
        "uwb_in_reward": False,
        "contact_sensor_contract": {
            "sensor_name": "contact_forces",
            "robot_foot_ids": "articulation_body_ids",
            "sensor_foot_ids": "contact_sensor_local_columns_by_exact_name",
            "invalid_mapping": "disable_gait_and_body_collision_rewards_with_warning",
        },
    }


def training_contract() -> dict[str, Any]:
    return {
        "version": "p2_track_training_2h_safe_direction_v2",
        "run_name": RUN_NAME,
        "training_hours": TRAINING_HOURS,
        "target_effective_seconds": int(TRAINING_HOURS * 3600.0),
        "schedule_boundaries_seconds": [
            SAFETY_WARM_END_SECONDS,
            SAFETY_STABILIZE_SECONDS,
        ],
        "clock_contract": {
            "lifetime_effective_seconds": "inherited_plus_current_session",
            "session_effective_seconds": "current_two_hour_run_only",
        },
        "safety_teacher": {
            "head_input": "navigation_depth_feature32",
            "target": "one_minus_privileged_safe3",
            "loss": "masked_bce_with_logits",
            "loss_weight": SAFETY_BCE_WEIGHT,
            "training_only": True,
            "scanner_well_formed_ratio_min": SAFETY_SCANNER_WELL_FORMED_RATIO,
            "scanner_finite_hit_ratio_min": SAFETY_SCANNER_FINITE_HIT_RATIO,
        },
        "gait_baseline_version": GAIT_BASELINE_VERSION,
        "gait_baseline_parent_model_id": GAIT_BASELINE_PARENT_MODEL_ID,
        "gait_baseline_parent_label": GAIT_BASELINE_PARENT_LABEL,
        "gait_baseline_thresholds": dict(GAIT_BASELINE_THRESHOLDS),
        "response_aux_dim": RESPONSE_AUX_DIM,
        "diagnostic_aux_dim": DIAGNOSTIC_AUX_DIM,
        "worker_aux_dim": WORKER_AUX_DIM,
        "terrain_telemetry": {
            "spawn_row": "terrain_levels",
            "difficulty_column": "terrain_types",
            "current_segment": "root_pos_w_x_track_boundaries",
            "invalid_segment": -1,
        },
    }


def privileged_safe_directions(critic_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build the goal-independent privileged safety target [left, center, right]."""
    if critic_obs.ndim != 2 or critic_obs.shape[1] < CRITIC_OBS_DIM:
        raise ValueError(
            f"P2 safety teacher expects [N,{CRITIC_OBS_DIM}] critic observations"
        )
    scan_start, scan_end = 60, 316
    nav_start = 319
    height = critic_obs[:, scan_start:scan_end].reshape(-1, 16, 16)
    nav_priv = critic_obs[:, nav_start : nav_start + 4]
    scanner_available = nav_priv[:, 0] > 0.5
    # nav_priv is [available, front, left, right]. The teacher contract is
    # [left, center, right].
    nav_risk = torch.stack((nav_priv[:, 2], nav_priv[:, 1], nav_priv[:, 3]), dim=1)
    nav_risk = torch.clamp(torch.nan_to_num(nav_risk, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)

    passable = []
    jump_values = []
    sector_valid_values = []
    sector_finite_ratios = []
    for y0, y1 in SAFETY_HEIGHT_SECTORS:
        region = height[:, y0:y1, :SAFETY_HEIGHT_FORWARD_COLS]
        finite = torch.isfinite(region)
        finite_ratio = finite.float().mean(dim=(1, 2))
        pair_valid = finite[:, :, 1:] & finite[:, :, :-1]
        diff = torch.abs(torch.diff(region, dim=2))
        diff = torch.where(pair_valid, diff, torch.full_like(diff, float("nan")))
        jump = torch.nanquantile(diff.flatten(1), 0.90, dim=1)
        sector_valid = (
            (finite_ratio >= SAFETY_HEIGHT_FINITE_RATIO_MIN)
            & (pair_valid.sum(dim=(1, 2)) >= SAFETY_HEIGHT_MIN_VALID_DIFFS)
            & torch.isfinite(jump)
        )
        jump = torch.nan_to_num(jump, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        excess = torch.relu(jump - SAFETY_HEIGHT_JUMP_FREE_M)
        passable.append(torch.exp(-torch.square(excess / SAFETY_HEIGHT_JUMP_SCALE_M)))
        jump_values.append(jump)
        sector_valid_values.append(sector_valid)
        sector_finite_ratios.append(finite_ratio)
    terrain_passable = torch.stack(passable, dim=1)
    jump90 = torch.stack(jump_values, dim=1)
    height_sector_valid = torch.stack(sector_valid_values, dim=1)
    height_finite_ratio3 = torch.stack(sector_finite_ratios, dim=1)
    teacher_valid = scanner_available & height_sector_valid.all(dim=1)
    safe = torch.clamp((1.0 - nav_risk) * terrain_passable, 0.0, 1.0)
    safe = torch.where(teacher_valid[:, None], safe, torch.zeros_like(safe))
    return safe, teacher_valid, {
        "nav_risk3": nav_risk,
        "terrain_passable3": terrain_passable,
        "height_jump90_3": jump90,
        "height_finite_ratio3": height_finite_ratio3,
        "height_sector_valid3": height_sector_valid.float(),
        "scanner_available": scanner_available.float(),
    }


def command_direction_weights(target_cmd3: torch.Tensor) -> torch.Tensor:
    """Map a three-axis command to smooth [left, center, right] sector weights."""
    if target_cmd3.ndim != 2 or target_cmd3.shape[1] != 3:
        raise ValueError("P2 direction weights expect [N,3] commands")
    command = torch.nan_to_num(target_cmd3, nan=0.0, posinf=0.0, neginf=0.0)
    theta = torch.atan2(command[:, 1], torch.clamp(command[:, 0], min=0.05))
    theta = theta + 0.5 * command[:, 2] * PREDICTIVE_COLLISION_LOOKAHEAD_S
    centers = torch.deg2rad(torch.tensor(
        PREDICTIVE_COLLISION_SECTOR_CENTERS_DEG,
        device=command.device,
        dtype=command.dtype,
    ))
    width = math.radians(PREDICTIVE_COLLISION_SECTOR_WIDTH_DEG)
    return torch.softmax(-0.5 * ((theta[:, None] - centers[None, :]) / width).square(), dim=1)


def missed_safe_direction_penalty(
    safe3: torch.Tensor,
    target_cmd3: torch.Tensor,
    scanner_available: torch.Tensor,
    session_effective_seconds: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize choosing a risky sector when a materially safer one exists."""
    if safe3.ndim != 2 or safe3.shape[1] != 3:
        raise ValueError("P2 safe-direction teacher expects [N,3]")
    weights = command_direction_weights(target_cmd3)
    selected_safe = (weights * safe3).sum(dim=1)
    best_safe = safe3.max(dim=1).values
    chosen_risk = torch.clamp(1.0 - selected_safe, 0.0, 1.0)
    safe_gap = torch.relu(best_safe - selected_safe - SAFE_DIRECTION_GAP_MARGIN)
    speed_scale = torch.tensor(
        (1.0, 1.0, CRAWL_BODY_RADIUS_M),
        device=target_cmd3.device,
        dtype=target_cmd3.dtype,
    )
    equivalent_speed = torch.linalg.vector_norm(
        torch.nan_to_num(target_cmd3, nan=0.0, posinf=0.0, neginf=0.0) * speed_scale,
        dim=1,
    )
    reward_active = (
        scanner_available.bool()
        & (equivalent_speed > SAFE_DIRECTION_MIN_SPEED_MPS)
        & (best_safe >= SAFE_DIRECTION_MIN_BEST_SAFE)
        & (safe_gap > 0.0)
    )
    penalty = -safe_direction_weight(session_effective_seconds) * chosen_risk * torch.clamp(safe_gap, 0.0, 1.0)
    penalty = torch.where(reward_active, penalty, torch.zeros_like(penalty))
    top2 = torch.topk(safe3, k=2, dim=1).values
    unique_best = (top2[:, 0] - top2[:, 1]) > 1.0e-4
    selected_is_best = weights.argmax(dim=1) == safe3.argmax(dim=1)
    selection_eligible = reward_active & unique_best
    return penalty, {
        "selected_safe": selected_safe,
        "best_safe": best_safe,
        "chosen_risk": chosen_risk,
        "safe_gap": safe_gap,
        "direction_weights": weights,
        "active": reward_active.float(),
        "selection_eligible": selection_eligible.float(),
        "selected_safest": (selection_eligible & selected_is_best).float(),
    }


def normalized_true_tracking_error(
    exec_cmd3: torch.Tensor, true_velocity3: torch.Tensor
) -> torch.Tensor:
    if exec_cmd3.shape != true_velocity3.shape or exec_cmd3.shape[-1] != 3:
        raise ValueError("P2 true tracking error expects matching [...,3] tensors")
    scale = torch.tensor(COMMAND_NORMALIZATION, device=exec_cmd3.device, dtype=exec_cmd3.dtype)
    axis_weights = torch.tensor(
        TRACKING_ERROR_AXIS_WEIGHTS, device=exec_cmd3.device, dtype=exec_cmd3.dtype
    )
    normalized = torch.clamp((exec_cmd3 - true_velocity3) / scale, -1.0, 1.0)
    return (normalized.square() * axis_weights).sum(dim=-1, keepdim=True)


def normalized_command_rate(
    target_cmd3: torch.Tensor, previous_target_cmd3: torch.Tensor
) -> torch.Tensor:
    if target_cmd3.shape != previous_target_cmd3.shape or target_cmd3.shape[-1] != 3:
        raise ValueError("P2 command rate expects matching [...,3] tensors")
    scale = torch.tensor(
        COMMAND_NORMALIZATION, device=target_cmd3.device, dtype=target_cmd3.dtype
    )
    axis_weights = torch.tensor(
        COMMAND_RATE_AXIS_WEIGHTS, device=target_cmd3.device, dtype=target_cmd3.dtype
    )
    normalized = torch.clamp((target_cmd3 - previous_target_cmd3) / scale, -1.0, 1.0)
    return (normalized.square() * axis_weights).sum(dim=-1, keepdim=True)


def frontier_potential_shaping(
    episode_start_distance: torch.Tensor,
    best_distance_before: torch.Tensor,
    best_distance_after: torch.Tensor,
    duration_frames: torch.Tensor,
    terminal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Potential shaping that cannot subsidize a failed or timed-out episode."""
    start = episode_start_distance.float()
    before = torch.clamp(start - best_distance_before.float(), min=0.0)
    after = torch.clamp(start - best_distance_after.float(), min=0.0)
    phi_before = FRONTIER_POTENTIAL_WEIGHT * before
    phi_after = FRONTIER_POTENTIAL_WEIGHT * after
    terminal = terminal.reshape(-1).bool()
    settled_after = torch.where(terminal, torch.zeros_like(phi_after), phi_after)
    discount = torch.pow(
        torch.full_like(phi_before, GAMMA_FRAME),
        duration_frames.float().clamp(1.0, float(NAV_PERIOD_FRAMES)),
    )
    shaping = discount * settled_after - phi_before
    return shaping, phi_before, settled_after


def crawl_deadzone_penalty(target_cmd3: torch.Tensor) -> torch.Tensor:
    if target_cmd3.shape[-1] != 3:
        raise ValueError("P2 crawl deadzone expects [...,3] target commands")
    equivalent_speed = torch.sqrt(
        target_cmd3[..., 0].square()
        + target_cmd3[..., 1].square()
        + (CRAWL_BODY_RADIUS_M * target_cmd3[..., 2]).square()
    )
    z = torch.clamp(equivalent_speed / CRAWL_STABLE_MIN_MPS, 0.0, 1.0)
    return CRAWL_WEIGHT * 4.0 * z * (1.0 - z)


def predictive_collision_risk_penalty(
    normalized_depth: torch.Tensor,
    target_cmd3: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return directional wall risk plus the legacy central-ROI shadow metric."""
    if normalized_depth.ndim != 4 or normalized_depth.shape[1:] != (
        DEPTH_HEIGHT,
        DEPTH_WIDTH,
        DEPTH_CHANNELS,
    ):
        raise ValueError(
            "P2 predictive collision expects depth [N,180,320,1], got "
            f"{tuple(normalized_depth.shape)}"
        )
    if target_cmd3.shape != (normalized_depth.shape[0], 3):
        raise ValueError(
            "P2 predictive collision expects target [N,3], got "
            f"{tuple(target_cmd3.shape)}"
        )
    raw_depth = normalized_depth[..., 0].float()
    valid = torch.isfinite(raw_depth) & (raw_depth > 1.0e-4) & (raw_depth <= 1.0)
    depth_m = torch.where(
        valid,
        torch.clamp(raw_depth, 0.0, 1.0) * PREDICTIVE_COLLISION_MAX_DEPTH_M,
        torch.full_like(raw_depth, PREDICTIVE_COLLISION_MAX_DEPTH_M),
    )

    def quantile_region(y_bounds, x_bounds, quantile):
        y0, y1 = y_bounds
        x0, x1 = x_bounds
        flat = depth_m[:, y0:y1, x0:x1].flatten(1)
        kth = max(1, int(math.ceil(float(quantile) * flat.shape[1])))
        return torch.kthvalue(flat, kth, dim=1).values

    band_depths = []
    for x_bounds in PREDICTIVE_COLLISION_HORIZONTAL_SECTORS.values():
        band_depths.append(
            torch.stack(
                tuple(
                    quantile_region(y_bounds, x_bounds, PREDICTIVE_COLLISION_QUANTILE)
                    for y_bounds in PREDICTIVE_COLLISION_VERTICAL_BANDS.values()
                ),
                dim=1,
            )
        )
    sector_depths = torch.stack(band_depths, dim=1)
    upper = sector_depths[:, :, 0]
    middle = sector_depths[:, :, 1]
    lower = sector_depths[:, :, 2]

    command = torch.nan_to_num(target_cmd3, nan=0.0, posinf=0.0, neginf=0.0)
    velocity_scale = torch.tensor(
        (1.0, 1.0, PREDICTIVE_COLLISION_YAW_RADIUS_M),
        device=target_cmd3.device,
        dtype=target_cmd3.dtype,
    )
    speed = torch.linalg.vector_norm(command * velocity_scale, dim=1)
    finite_command = torch.isfinite(target_cmd3).all(dim=1)
    active = finite_command & (speed > PREDICTIVE_COLLISION_ACTIVE_SPEED_MPS)
    stopping_distance_m = (
        PREDICTIVE_COLLISION_BODY_MARGIN_M
        + speed.square()
        / (2.0 * PREDICTIVE_COLLISION_BRAKE_ACCEL_MPS2)
    )
    stopping_distance_m = torch.where(
        active,
        stopping_distance_m,
        torch.zeros_like(stopping_distance_m),
    )
    flatness = torch.exp(
        -((upper - lower) / PREDICTIVE_COLLISION_FLATNESS_SCALE_M).square()
    )
    near_upper = torch.sigmoid(
        (stopping_distance_m[:, None] + PREDICTIVE_COLLISION_NEAR_MARGIN_M - upper)
        / PREDICTIVE_COLLISION_NEAR_SCALE_M
    )
    near_middle = torch.sigmoid(
        (stopping_distance_m[:, None] + PREDICTIVE_COLLISION_NEAR_MARGIN_M - middle)
        / PREDICTIVE_COLLISION_NEAR_SCALE_M
    )
    wallness = flatness * near_upper * near_middle
    clearance_m = 0.5 * (upper + middle)
    gap_x = torch.clamp(
        (
            stopping_distance_m[:, None]
            + PREDICTIVE_COLLISION_GAP_MARGIN_M
            - clearance_m
        )
        / PREDICTIVE_COLLISION_RISK_BAND_M,
        0.0,
        1.0,
    )
    gap = gap_x.square() * (3.0 - 2.0 * gap_x)
    sector_risk = wallness * gap

    theta = torch.atan2(
        command[:, 1], torch.clamp(command[:, 0], min=0.05)
    ) + 0.5 * command[:, 2] * PREDICTIVE_COLLISION_LOOKAHEAD_S
    centers = torch.deg2rad(
        torch.tensor(
            PREDICTIVE_COLLISION_SECTOR_CENTERS_DEG,
            device=target_cmd3.device,
            dtype=target_cmd3.dtype,
        )
    )
    width = math.radians(PREDICTIVE_COLLISION_SECTOR_WIDTH_DEG)
    direction_logits = -0.5 * ((theta[:, None] - centers[None, :]) / width).square()
    direction_weights = torch.softmax(direction_logits, dim=1)
    risk = (direction_weights * sector_risk).sum(dim=1).clamp(0.0, 1.0)
    risk = torch.where(active, risk, torch.zeros_like(risk))
    selected_clearance = (direction_weights * clearance_m).sum(dim=1)

    y0, y1, x0, x1 = LEGACY_PREDICTIVE_COLLISION_ROI
    legacy_flat = depth_m[:, y0:y1, x0:x1].flatten(1)
    legacy_kth = max(
        1, int(math.ceil(LEGACY_PREDICTIVE_COLLISION_QUANTILE * legacy_flat.shape[1]))
    )
    legacy_clearance = torch.kthvalue(legacy_flat, legacy_kth, dim=1).values
    legacy_speed = torch.clamp(command[:, 0], min=0.0)
    legacy_stop = (
        LEGACY_PREDICTIVE_COLLISION_BODY_MARGIN_M
        + legacy_speed.square() / (2.0 * LEGACY_PREDICTIVE_COLLISION_BRAKE_ACCEL_MPS2)
    )
    legacy_risk = torch.clamp(
        (legacy_stop - legacy_clearance) / LEGACY_PREDICTIVE_COLLISION_RISK_BAND_M,
        0.0,
        1.0,
    ).square()
    legacy_active = finite_command & (
        legacy_speed > PREDICTIVE_COLLISION_ACTIVE_SPEED_MPS
    )
    legacy_risk = torch.where(legacy_active, legacy_risk, torch.zeros_like(legacy_risk))

    penalty = PREDICTIVE_COLLISION_WEIGHT * risk
    return (
        penalty,
        selected_clearance,
        stopping_distance_m,
        risk,
        legacy_risk,
        wallness,
        sector_risk,
    )


def body_collision_penalty(
    max_force_n: torch.Tensor,
    previous_contact: torch.Tensor,
    terminal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one 5 Hz body-collision penalty and the next contact state."""
    force = torch.nan_to_num(max_force_n.float(), nan=0.0, posinf=0.0, neginf=0.0)
    terminal = terminal.bool()
    contact = (force > BODY_COLLISION_SOFT_FORCE_N) & ~terminal
    severity = torch.clamp(
        (force - BODY_COLLISION_SOFT_FORCE_N)
        / (BODY_COLLISION_HARD_FORCE_N - BODY_COLLISION_SOFT_FORCE_N),
        0.0,
        1.0,
    )
    onset = contact & ~previous_contact.bool()
    onset_penalty = (
        BODY_COLLISION_ONSET_BASE
        + BODY_COLLISION_ONSET_SEVERITY * severity
    )
    penalty = torch.where(
        onset,
        onset_penalty,
        torch.where(
            contact,
            torch.full_like(force, BODY_COLLISION_PERSISTENT),
            torch.zeros_like(force),
        ),
    )
    return penalty, contact


def frontier_stagnation_penalty(stagnation_age: torch.Tensor) -> torch.Tensor:
    """Penalize a confirmed lack of monotonic frontier improvement."""
    age = stagnation_age.to(dtype=torch.float32)
    ramp = STAGNATION_INITIAL_PENALTY + STAGNATION_RAMP_PER_TICK * torch.clamp(
        age - 1.0, min=0.0
    )
    penalty = torch.maximum(
        ramp,
        torch.full_like(ramp, STAGNATION_PENALTY_CAP),
    )
    return torch.where(age > 0.0, penalty, torch.zeros_like(penalty))


def map_normalized_action(
    normalized_action: torch.Tensor, *, hard_abs_vy: float = 0.40
) -> torch.Tensor:
    """Map [vx,vy,wz] policy coordinates into body-frame target_cmd3."""
    if normalized_action.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"P2 action must end in {ACTION_DIM} dimensions, got "
            f"{tuple(normalized_action.shape)}"
        )
    bounded = torch.clamp(normalized_action, -1.0, 1.0)
    vx_normalized = torch.where(
        torch.isfinite(bounded[..., 0]),
        bounded[..., 0],
        torch.full_like(bounded[..., 0], -1.0),
    )
    vy_normalized = torch.where(
        torch.isfinite(bounded[..., 1]),
        bounded[..., 1],
        torch.zeros_like(bounded[..., 1]),
    )
    wz_normalized = torch.where(
        torch.isfinite(bounded[..., 2]),
        bounded[..., 2],
        torch.zeros_like(bounded[..., 2]),
    )
    vx = 0.625 * (vx_normalized + 1.0)
    vy = max(0.0, min(float(hard_abs_vy), 0.40)) * vy_normalized
    wz = wz_normalized
    return torch.stack((vx, vy, wz), dim=-1)


def adapter_confidence(
    *,
    velocity_valid: torch.Tensor,
    velocity_age: torch.Tensor,
    velocity_log_sigma: torch.Tensor,
    target_cmd3: torch.Tensor,
) -> torch.Tensor:
    """Compute confidence only from deployment-available physical fields."""
    if velocity_log_sigma.shape[-1] != 3 or target_cmd3.shape[-1] != 3:
        raise ValueError("P2 confidence expects log_sigma3 and target_cmd3")
    valid = velocity_valid.to(dtype=velocity_log_sigma.dtype)
    if valid.ndim == velocity_log_sigma.ndim - 1:
        valid = valid.unsqueeze(-1)
    age = velocity_age.to(dtype=velocity_log_sigma.dtype)
    if age.ndim == velocity_log_sigma.ndim - 1:
        age = age.unsqueeze(-1)
    age_factor = torch.exp(-torch.clamp(age, min=0.0) / 0.20)
    sigma = torch.exp(torch.clamp(velocity_log_sigma, -4.0, 1.0)).mean(
        dim=-1, keepdim=True
    )
    sigma_factor = torch.clamp(torch.exp(-sigma), 0.10, 1.0)
    vx_overflow = torch.clamp((target_cmd3[..., 0:1] - 1.0) / 0.25, 0.0, 1.0)
    wz_overflow = torch.clamp(
        (target_cmd3[..., 2:3].abs() - 0.8) / 0.2, 0.0, 1.0
    )
    vy_overflow = torch.clamp(
        (target_cmd3[..., 1:2].abs() - 0.20) / 0.20, 0.0, 1.0
    )
    domain_factor = 1.0 - 0.5 * torch.maximum(
        torch.maximum(vx_overflow, vy_overflow), wz_overflow
    )
    return torch.clamp(valid * age_factor * sigma_factor * domain_factor, 0.0, 1.0)


def feedback_age_seconds(normalized_age: torch.Tensor, *, age_clip_s: float) -> torch.Tensor:
    """Convert the shared P1.5 normalized age field back to physical seconds."""
    return torch.clamp(normalized_age, 0.0, 1.0) * max(0.0, float(age_clip_s))


def normalized_tracking_error(
    exec_cmd3: torch.Tensor,
    measured_velocity3: torch.Tensor,
    velocity_valid: torch.Tensor,
) -> torch.Tensor:
    """Mask missing SportMode xy feedback while retaining IMU yaw feedback."""
    if exec_cmd3.shape != measured_velocity3.shape or exec_cmd3.shape[-1] != 3:
        raise ValueError("P2 tracking error expects matching [...,3] tensors")
    valid = velocity_valid.to(device=exec_cmd3.device, dtype=exec_cmd3.dtype)
    if valid.ndim == exec_cmd3.ndim - 1:
        valid = valid.unsqueeze(-1)
    error = torch.clamp(
        (exec_cmd3 - measured_velocity3)
        / torch.tensor(
            COMMAND_NORMALIZATION, device=exec_cmd3.device, dtype=exec_cmd3.dtype
        ),
        -1.0,
        1.0,
    )
    weights = torch.tensor(
        TRACKING_ERROR_AXIS_WEIGHTS, device=exec_cmd3.device, dtype=exec_cmd3.dtype
    )
    xy = (error[..., :2].square() * weights[:2]).sum(dim=-1, keepdim=True) * valid
    return xy + error[..., 2:3].square() * weights[2]


def contract_metadata() -> dict[str, Any]:
    command = command_contract()
    confidence = confidence_contract()
    reward = reward_contract()
    training = training_contract()
    return {
        "stage": STAGE_NAME,
        "actor_input_dim": ACTOR_INPUT_DIM,
        "critic_input_dim": CRITIC_INPUT_DIM,
        "command": command,
        "command_digest": stable_digest(command),
        "confidence": confidence,
        "confidence_digest": stable_digest(confidence),
        "reward": reward,
        "reward_digest": stable_digest(reward),
        "training": training,
        "training_digest": stable_digest(training),
        "gamma_nav": GAMMA_NAV,
        "gamma_frame": GAMMA_FRAME,
        "gae_lambda": GAE_LAMBDA,
    }
