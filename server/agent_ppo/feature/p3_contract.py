#!/usr/bin/env python3
"""P3 Standard joint-recovery contracts."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import torch

POLICY_OBS_DIM = 57905
LOW_LEVEL_POLICY_OBS_DIM = 57901
PROPRIO_SCAN_DIM = 301
GOAL4_SLICE = slice(301, 305)
DEPTH_SLICE = slice(305, 57905)
SESSION_TARGET_SECONDS = 7200.0
P3_WORKER_EXTRA_DIM = 108
P3_PRIVILEGED_WIRE_DIM = 493
RUNTIME_TERRAIN_SIZE_INDEX = 0
GAIT_CONTACT_ONSET_SLICE = slice(1, 5)
GAIT_IMPACT_SPEED_SLICE = slice(5, 9)
GAIT_TOUCHDOWN_Y_SLICE = slice(9, 13)
GAIT_CONTINUOUS_STANCE_SLICE = slice(13, 17)
GAIT_COMPLETED_SLIP_SLICE = slice(17, 21)
GAIT_COMPLETED_SLIP_EVENT_SLICE = slice(21, 25)
JOINT_TORQUE_SLICE = slice(25, 37)
MECHANICAL_POWER_INDEX = 37
JOINT_MAPPING_VALID_INDEX = 38
COMMAND_BUCKET_INDEX = 39
COMMAND_ANCHOR_WEIGHT_INDEX = 40
SIM2REAL_COMPONENT_SLICE = slice(41, 45)
SIM2REAL_COMPONENT_VALID_INDEX = 45
JOINT_ACCELERATION_SLICE = slice(46, 58)
CONTACT_FORCE_SLICE = slice(58, 72)
CONTACT_ONSET_SLICE = slice(72, 86)
CONTACT_OVER_THRESHOLD_DURATION_SLICE = slice(86, 100)
JOINT_ACCELERATION_MAPPING_VALID_INDEX = 100
CONTACT_REWARD_MAPPING_VALID_INDEX = 101
PUSH_EVENT_FLAG_INDEX = 102
PUSH_DELTA_VELOCITY_SLICE = slice(103, 105)
SECONDS_SINCE_PUSH_INDEX = 105
PUSH_RUNTIME_ACTIVE_INDEX = 106
PUSH_TELEMETRY_VALID_INDEX = 107
SUBGOAL_SUCCESS_DISTANCE_M = 0.50
SUBGOAL_MIN_DISTANCE_M = 1.30
SUBGOAL_MAX_DISTANCE_M = 2.90
TILE_HALF_EXTENT_M = 4.0
TILE_INNER_MARGIN_M = 1.0
TILE_RESET_LOCAL_ABS_M = 3.20
PLATFORM_COMPLETE_MARGIN_M = 0.10
M3_TARGET_OVERSHOOT_M = 0.03
M3_BOUNDARY_MARGIN_M = 0.03
MILESTONE_RADIUS_RANGES_M = ((1.30, 1.60), (2.50, 2.90))
REPLAN_ANGLE_OFFSETS_DEG = (-60.0, -30.0, 30.0, 60.0)
SOFT_BRAKE_START_MARGIN_M = 0.20
SOFT_BRAKE_FINAL_MARGIN_M = 0.04
SOFT_BRAKE_MID_SPEED_M_S = 0.30
SOFT_BRAKE_FINAL_SPEED_M_S = 0.12
P3_ACTION_DIM = 12
P3_COMMAND_SEED = 3187
# M1 and M2 are local-goal events. Platform completion supplies M3.
JOINT_SUCCESS_MIN_SUBGOALS = 2
SUBGOAL_EVENT_NONE = 0
SUBGOAL_EVENT_REACHED = 1
SUBGOAL_EVENT_TIMEOUT = 2


@dataclass(frozen=True)
class P3Phase:
    name: str
    start_s: float
    end_s: float
    low_level_trainable: bool
    adapter_trainable: bool
    high_level_trainable: bool


PHASES = (
    P3Phase("gaitfixcalib", 0.0, 900.0, True, True, False),
    P3Phase("repair", 900.0, 4500.0, True, True, False),
    P3Phase("pushwarm", 4500.0, 5400.0, True, True, False),
    P3Phase("pushfull", 5400.0, 6300.0, True, True, False),
    P3Phase("stable", 6300.0, SESSION_TARGET_SECONDS, False, True, False),
)

MIRROR_SEQUENCE_SHARE = 0.25
MIRROR_TARGET_GRADIENT_RATIO = 0.005
MIRROR_MAX_GRADIENT_RATIO = 0.05
MEMORY_TARGET_GRADIENT_RATIO = 0.015
MEMORY_MAX_GRADIENT_RATIO = 0.03
ACTION_SMOOTH_TARGET_GRADIENT_RATIO = 0.0
ACTION_RANGE_TARGET_GRADIENT_RATIO = 0.0
AUXILIARY_MAX_GRADIENT_RATIO = 0.05
ANCHOR_TARGET_GRADIENT_RATIO = 0.030
ACTION_MEAN_SOFT_LIMIT = 5.0
ACTION_TO_JOINT_SCALE = 0.25
GAIT_CONTACT_REWARD_CAP = 0.18
GAIT_CROSS_REWARD_CAP = 0.14
GAIT_STARVATION_REWARD_CAP = 0.08
GAIT_TOTAL_REWARD_CAP = 0.40
GAIT_BASELINE_CONTINUOUS_STRIDE = 5
GAIT_BASELINE_VERSION = 3
TERRAIN_COLUMN_BUCKET_BOUNDARIES = (3, 6, 13, 20)


def gait_training_fraction(elapsed_s: float) -> float:
    del elapsed_s
    # Event gait terms remain shadow diagnostics in the stair-memory run.
    return 0.0


def mirror_training_fraction(elapsed_s: float) -> float:
    return 0.0 if float(elapsed_s) < 2700.0 else 1.0


def depth_fault_strength(elapsed_s: float) -> float:
    # P3.5 inherits the parent model's final 50% fault mixture. Camera timing,
    # rather than stronger pixel corruption, is the active curriculum.
    return 0.0 if float(elapsed_s) < 900.0 else 0.5


def memory_training_fraction(elapsed_s: float) -> float:
    return depth_fault_strength(elapsed_s)


def adapter_replay_ratios(elapsed_s: float) -> tuple[float, float, float]:
    """Return latest, recent and parent completed-record replay shares."""
    elapsed = max(0.0, float(elapsed_s))
    if elapsed < 4500.0:
        return 0.50, 0.25, 0.25
    if elapsed < 6300.0:
        return 0.60, 0.25, 0.15
    return 0.75, 0.15, 0.10


def adapter_updates_per_low_rollout(elapsed_s: float) -> int:
    return int(phase_for_elapsed(elapsed_s).adapter_trainable)


def camera_delay_probabilities(elapsed_s: float) -> tuple[float, float, float]:
    """Return nominal, 40-100 ms and 100-150 ms active-delay shares."""
    if float(elapsed_s) < 900.0:
        return 1.0, 0.0, 0.0
    if float(elapsed_s) < 4500.0:
        return 0.70, 0.30, 0.0
    return 0.50, 0.35, 0.15


def push_phase_config(elapsed_s: float) -> dict[str, float | bool | str]:
    elapsed = max(0.0, float(elapsed_s))
    if elapsed < 4500.0:
        return {
            "name": "disabled",
            "active": False,
            "max_velocity_xy_m_s": 0.0,
            "min_interval_s": 12.0,
            "max_interval_s": 18.0,
        }
    if elapsed < 5400.0:
        maximum = 0.05
        name = "pushwarm"
    else:
        maximum = 0.08
        name = "pushfull"
    return {
        "name": name,
        "active": True,
        "max_velocity_xy_m_s": maximum,
        "min_interval_s": 12.0,
        "max_interval_s": 18.0,
    }


def action_smooth_training_fraction(
    elapsed_s: float, *, mapping_valid: bool = True
) -> float:
    del elapsed_s, mapping_valid
    # Kept for diagnostics only; it must never enter the optimizer.
    return 0.0


def asymmetric_local_progress(
    start_distance: torch.Tensor,
    end_distance: torch.Tensor,
    duration_frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    duration = duration_frames.to(start_distance).clamp_min(1.0)
    normalized = (start_distance - end_distance) * 10.0 / duration
    positive = 1.5 * torch.clamp(normalized, 0.0, 0.25)
    negative = 0.4 * torch.clamp(normalized, -0.25, 0.0)
    finite = torch.isfinite(start_distance) & torch.isfinite(end_distance)
    return torch.where(finite, positive, 0.0), torch.where(finite, negative, 0.0)


def phase_for_elapsed(elapsed_s: float) -> P3Phase:
    elapsed = min(max(float(elapsed_s), 0.0), SESSION_TARGET_SECONDS)
    for phase in PHASES:
        if elapsed < phase.end_s:
            return phase
    return PHASES[-1]


def domain_randomization_index(elapsed_s: float) -> int:
    del elapsed_s
    # Platform BaseEnv creates Isaac once per process, so P3 uses one startup
    # randomization contract instead of advertising an unrealizable schedule.
    return 0


def materialize_environment_config(usr_conf: dict, elapsed_s: float) -> dict:
    """Materialize the process-start P3 environment configuration."""
    result = copy.deepcopy(usr_conf)
    p3 = result.get("p3_standard_joint", {})
    config = p3.get("domain_randomization", {}) if isinstance(p3, dict) else {}
    friction = config.get("friction_range", [0.65, 1.25])
    added_mass = float(config.get("base_added_mass_kg", 0.40))
    noise_level = float(config.get("noise_level", 0.35))
    restitution = config.get("restitution_range", [0.0, 0.05])
    push_enabled = bool(config.get("push_robots", True))
    min_push_interval = float(config.get("min_push_interval_s", 12.0))
    max_push_interval = float(config.get("push_interval_s", 18.0))
    if not 0.0 < min_push_interval <= max_push_interval:
        raise ValueError("P3 push interval must be finite, positive, and ordered")

    result["domain_rand"] = {
        **dict(result.get("domain_rand", {})),
        "enable_domain_rand": True,
        "randomize_friction": True,
        "friction_range": list(friction),
        "randomize_base_mass": added_mass > 0.0,
        "added_mass_range": [-added_mass, added_mass],
        "restitution_range": list(restitution),
        "push_robots": push_enabled,
        "min_push_interval_s": min_push_interval,
        "push_interval_s": max_push_interval,
        "max_push_vel_xy": 0.0,
    }
    result["noise"] = {
        **dict(result.get("noise", {})),
        "add_noise": noise_level > 0.0,
        "noise_level": noise_level,
        "dof_pos": 0.01,
        "dof_vel": 0.50,
        "ang_vel": 0.10,
        "gravity": 0.05,
    }
    result["p3_runtime"] = {
        "phase_index": domain_randomization_index(elapsed_s),
        "environment_contract": "p35_static_dr_dynamic_push_v1",
        "base_added_mass_kg": added_mass,
        "push_enabled": push_enabled,
        "push_velocity_m_s": 0.0,
        "worker_command_override": True,
        "low_phase_command_owner": "p3_worker_recovery_sampler_2_to_8_seconds",
    }
    stage = result.setdefault("p3_standard_joint", {})
    push_schedule = stage.setdefault("push_schedule", {})
    push_schedule["resume_offset_s"] = max(0.0, float(elapsed_s))
    return result


def platform_completion_radii(terrain_size_x_m: float = 8.0) -> tuple[float, float, float]:
    """Return scorer threshold, safe virtual target, and terrain boundary."""
    boundary = 0.5 * float(terrain_size_x_m)
    if not math.isfinite(boundary) or boundary <= PLATFORM_COMPLETE_MARGIN_M:
        raise ValueError("terrain size must provide a finite positive half extent")
    complete = boundary - PLATFORM_COMPLETE_MARGIN_M
    target = min(
        complete + M3_TARGET_OVERSHOOT_M,
        boundary - M3_BOUNDARY_MARGIN_M,
    )
    if not complete < target < boundary:
        raise ValueError("P3 M3 target must remain between completion and boundary")
    return complete, target, boundary


def radial_distance(root_xy: torch.Tensor, episode_origin_xy: torch.Tensor) -> torch.Tensor:
    if root_xy.shape != episode_origin_xy.shape or root_xy.shape[-1] != 2:
        raise ValueError("root_xy and episode_origin_xy must have matching [N,2] shape")
    return torch.linalg.vector_norm(root_xy - episode_origin_xy, dim=-1)


def radial_new_best(
    radius: torch.Tensor,
    best_radius: torch.Tensor,
    *,
    scale: float = 4.0,
    max_delta_m: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor]:
    if radius.shape != best_radius.shape:
        raise ValueError("radius and best_radius must have matching shape")
    finite_radius = torch.nan_to_num(radius, nan=0.0, posinf=0.0, neginf=0.0)
    finite_best = torch.nan_to_num(best_radius, nan=0.0, posinf=0.0, neginf=0.0)
    delta = torch.clamp(finite_radius - finite_best, 0.0, float(max_delta_m))
    return float(scale) * delta, torch.maximum(finite_best, finite_radius)


def radial_outward_speed_limit(
    radius: torch.Tensor,
    complete_radius_m: float,
) -> torch.Tensor:
    """Per-env outward speed cap; zero means the completion hold is active."""
    complete = float(complete_radius_m)
    result = torch.full_like(radius, float("inf"))
    middle = radius >= complete - SOFT_BRAKE_START_MARGIN_M
    final = radius >= complete - SOFT_BRAKE_FINAL_MARGIN_M
    result[middle] = SOFT_BRAKE_MID_SPEED_M_S
    result[final] = SOFT_BRAKE_FINAL_SPEED_M_S
    result[radius >= complete] = 0.0
    return result


def cap_outward_body_command(
    command: torch.Tensor,
    root_xy: torch.Tensor,
    episode_origin_xy: torch.Tensor,
    yaw: torch.Tensor,
    complete_radius_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap only the outward world-frame linear component near M3."""
    if command.ndim != 2 or command.shape[-1] != 3:
        raise ValueError("command must have shape [N,3]")
    radius = radial_distance(root_xy, episode_origin_xy)
    limit = radial_outward_speed_limit(radius, complete_radius_m).to(command)
    delta = root_xy - episode_origin_xy
    radial_unit = torch.nn.functional.normalize(delta, dim=-1, eps=1.0e-6)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    world_linear = torch.stack(
        (
            cos_yaw * command[:, 0] - sin_yaw * command[:, 1],
            sin_yaw * command[:, 0] + cos_yaw * command[:, 1],
        ),
        dim=-1,
    )
    outward = (world_linear * radial_unit).sum(dim=-1)
    excess = torch.clamp(outward - limit, min=0.0)
    guarded_world = world_linear - excess.unsqueeze(-1) * radial_unit
    guarded = command.clone()
    guarded[:, 0] = cos_yaw * guarded_world[:, 0] + sin_yaw * guarded_world[:, 1]
    guarded[:, 1] = -sin_yaw * guarded_world[:, 0] + cos_yaw * guarded_world[:, 1]
    hold = radius >= float(complete_radius_m)
    guarded[hold] = 0.0
    return guarded, hold


def split_policy_observation(observation: torch.Tensor):
    """Drop only goal4 for the unchanged low-level transport contract."""
    if observation.ndim != 2 or observation.shape[1] != POLICY_OBS_DIM:
        raise ValueError(
            f"P3 policy observation must be [N,{POLICY_OBS_DIM}], got {tuple(observation.shape)}"
        )
    low = torch.cat((observation[:, :PROPRIO_SCAN_DIM], observation[:, DEPTH_SLICE]), dim=-1)
    if low.shape[1] != LOW_LEVEL_POLICY_OBS_DIM:
        raise AssertionError("P3 low-level observation adapter shape drift")
    return low, observation[:, GOAL4_SLICE], observation[:, DEPTH_SLICE]


def sample_local_subgoals_with_validity(
    root_xy,
    tile_origin_xy,
    *,
    generator,
    min_distance_m=SUBGOAL_MIN_DISTANCE_M,
    max_distance_m=SUBGOAL_MAX_DISTANCE_M,
    tile_inner_margin_m=TILE_INNER_MARGIN_M,
    max_attempts=64,
):
    if root_xy.ndim != 2 or root_xy.shape[-1] != 2 or tile_origin_xy.shape != root_xy.shape:
        raise ValueError("root_xy and tile_origin_xy must have matching [N,2] shape")
    minimum = float(min_distance_m)
    maximum = float(max_distance_m)
    inner = TILE_HALF_EXTENT_M - float(tile_inner_margin_m)
    if not (0.0 < minimum <= maximum and 0.0 < inner < TILE_HALF_EXTENT_M):
        raise ValueError("P3 subgoal distance or tile margin is invalid")
    count = root_xy.shape[0]
    result = torch.zeros_like(root_xy)
    valid = torch.zeros(count, dtype=torch.bool, device=root_xy.device)
    for _ in range(max(1, int(max_attempts))):
        pending = ~valid
        if not bool(pending.any()):
            break
        ids = pending.nonzero(as_tuple=False).flatten()
        distance = torch.empty(ids.numel(), 1, device=root_xy.device, dtype=root_xy.dtype)
        distance.uniform_(minimum, maximum, generator=generator)
        angle = torch.empty_like(distance).uniform_(-math.pi, math.pi, generator=generator)
        candidate = root_xy[ids] + torch.cat(
            (torch.cos(angle), torch.sin(angle)), dim=-1
        ) * distance
        inside = ((candidate - tile_origin_xy[ids]).abs() <= inner).all(dim=-1)
        if bool(inside.any()):
            accepted = ids[inside]
            result[accepted] = candidate[inside]
            valid[accepted] = True
    result[~valid] = tile_origin_xy[~valid] + (inner + 1.0)
    return result, valid


def sample_local_subgoals(root_xy, tile_origin_xy, *, generator):
    result, valid = sample_local_subgoals_with_validity(
        root_xy, tile_origin_xy, generator=generator
    )
    if not bool(valid.all()):
        raise RuntimeError("P3 could not sample a legal local subgoal in 64 attempts")
    return result


def local_out_of_bounds(
    root_xy, tile_origin_xy, *, threshold_m=TILE_RESET_LOCAL_ABS_M
):
    if root_xy.shape != tile_origin_xy.shape or root_xy.shape[-1] != 2:
        raise ValueError("root_xy and tile_origin_xy must have matching [N,2] shape")
    return (root_xy - tile_origin_xy).abs().amax(dim=-1) > float(threshold_m)


def subgoal_reached(
    root_xy, goal_xy, *, threshold_m=SUBGOAL_SUCCESS_DISTANCE_M
):
    if root_xy.shape != goal_xy.shape or root_xy.shape[-1] != 2:
        raise ValueError("root_xy and goal_xy must have matching [N,2] shape")
    return torch.linalg.vector_norm(goal_xy - root_xy, dim=-1) <= float(threshold_m)


def joint_episode_success(standard_completed, subgoal_success_count, hard_failure):
    """Monitoring-only conjunction; never drives environment termination."""
    return (
        standard_completed.bool()
        & (subgoal_success_count >= JOINT_SUCCESS_MIN_SUBGOALS)
        & ~hard_failure.bool()
    )


def contract():
    return {
        "name": "p35_low_speed_gait_push_v1",
        "policy_observation_dim": POLICY_OBS_DIM,
        "low_level_policy_observation_dim": LOW_LEVEL_POLICY_OBS_DIM,
        "high_level_actor_input_dim": 85,
        "subgoal_success_distance_m": SUBGOAL_SUCCESS_DISTANCE_M,
        "subgoal_distance_m": [SUBGOAL_MIN_DISTANCE_M, SUBGOAL_MAX_DISTANCE_M],
        "milestone_radius_ranges_m": [list(item) for item in MILESTONE_RADIUS_RANGES_M],
        "platform_completion_margin_m": PLATFORM_COMPLETE_MARGIN_M,
        "m3_target_overshoot_m": M3_TARGET_OVERSHOOT_M,
        "m3_boundary_margin_m": M3_BOUNDARY_MARGIN_M,
        "replan_angle_offsets_deg": list(REPLAN_ANGLE_OFFSETS_DEG),
        "m3_soft_brake": {
            "start_margin_m": SOFT_BRAKE_START_MARGIN_M,
            "final_margin_m": SOFT_BRAKE_FINAL_MARGIN_M,
            "mid_speed_m_s": SOFT_BRAKE_MID_SPEED_M_S,
            "final_speed_m_s": SOFT_BRAKE_FINAL_SPEED_M_S,
        },
        "tile_inner_margin_m": TILE_INNER_MARGIN_M,
        "tile_reset_local_abs_m": TILE_RESET_LOCAL_ABS_M,
        "standard_success_owner": "platform_standard_scorer",
        "subgoal_success_owner": "p3_high_level",
        "high_level_goal_precedence": ["_p3_goal_positions", "goal_positions"],
        "subgoal_success_terminates_environment": False,
        "joint_success_min_subgoals": JOINT_SUCCESS_MIN_SUBGOALS,
        "low_level_storage": "proprio45+frozen_cnn_feature32",
        "anchor_cnn_reuse_requires_identical_frozen_weights": True,
        "physics_randomization": "platform_process_start_friction_base_mass_restitution_dynamic_push",
        "observation_noise": "platform_process_start_explicit_term_bounds",
        "step_transport": {
            "action_dim": P3_ACTION_DIM,
            "privileged_wire_dim": P3_PRIVILEGED_WIRE_DIM,
            "runtime_extra_dim": P3_WORKER_EXTRA_DIM,
            "training_only_extra": {
                "joint_acc12": [46, 58],
                "contact_force14": [58, 72],
                "contact_onset14": [72, 86],
                "contact_over_threshold_duration14": [86, 100],
                "joint_mapping_valid": 100,
                "contact_mapping_valid": 101,
                "push_event_flag": 102,
                "push_delta_vx_vy": [103, 105],
                "seconds_since_push": 105,
                "push_runtime_active": 106,
                "push_telemetry_valid": 107,
            },
        },
        "low_phase_command_owner": "p3_worker_recovery_sampler_2_to_8_seconds",
        "high_level_training": "disabled_full_session",
        "high_level_frozen_modules": [
            "navigation_encoder",
            "actor_lstm",
            "critic",
            "safety_head",
        ],
        "training_schedule": [
            {
                "name": phase.name,
                "start_s": phase.start_s,
                "end_s": phase.end_s,
                "low_level_trainable": phase.low_level_trainable,
                "adapter_trainable": phase.adapter_trainable,
                "high_level_trainable": phase.high_level_trainable,
            }
            for phase in PHASES
        ],
        "local_out_of_bounds": "diagnostic_only_platform_reset_unavailable",
        "local_progress": {
            "positive_scale": 1.5,
            "negative_scale": 0.4,
            "normalized_by_duration_frames": True,
        },
        "mirror_consistency": {
            "sequence_share": MIRROR_SEQUENCE_SHARE,
            "target_gradient_ratio": MIRROR_TARGET_GRADIENT_RATIO,
            "hard_gradient_ratio": MIRROR_MAX_GRADIENT_RATIO,
            "training_only": True,
            "preflight": "joint_order+action_scale+pd+effort+contact_mapping",
        },
        "auxiliary_gradient_targets": {
            "anchor": ANCHOR_TARGET_GRADIENT_RATIO,
            "memory": MEMORY_TARGET_GRADIENT_RATIO,
            "mirror": MIRROR_TARGET_GRADIENT_RATIO,
            "combined_hard_cap": AUXILIARY_MAX_GRADIENT_RATIO,
        },
        "deterministic_mean_smoothing": {
            "version": "p3_action_smooth_v1",
            "joint_target_scale": ACTION_TO_JOINT_SCALE,
            "rate_threshold_rad_per_frame": {
                "hip_thigh": 0.20,
                "calf": 0.25,
            },
            "jerk_threshold_rad_per_frame2": {
                "hip_thigh": 0.15,
                "calf": 0.20,
            },
            "target_gradient_ratio": ACTION_SMOOTH_TARGET_GRADIENT_RATIO,
            "mean_soft_limit": ACTION_MEAN_SOFT_LIMIT,
            "range_target_gradient_ratio": ACTION_RANGE_TARGET_GRADIENT_RATIO,
            "combined_auxiliary_hard_ratio": AUXILIARY_MAX_GRADIENT_RATIO,
            "training_only": True,
        },
        "command_anchor_by_bucket": {
            "straight": 0.35,
            "reserved_reverse": 0.0,
            "vx_vy": 0.30,
            "vx_wz": 0.25,
            "pure_yaw": 0.15,
            "brake_restart": 0.10,
            "zero": 0.10,
        },
        "command_domain": {
            "vx_m_s": [0.0, 1.0],
            "vy_m_s": [-0.30, 0.30],
            "wz_rad_s": [-0.90, 0.90],
            "bucket_weights": [0.25, 0.0, 0.10, 0.35, 0.08, 0.15, 0.07],
            "reverse_recovery_enabled": False,
            "reason": "high_level_contract_cannot_issue_negative_vx",
        },
        "worker_command_sampler": {
            "seed": P3_COMMAND_SEED,
            "rng": "dedicated_torch_generator",
            "resume": "seeded_fresh_after_environment_reset",
        },
        "touchdown_frame": "full_root_quaternion_inverse_body_y",
        "slip_measurement": "per_stance_accumulated_world_xy_distance",
        "worker_command_sampler_resume": "seeded_fresh_after_environment_reset",
        "gait_event_caps": {
            "contact": GAIT_CONTACT_REWARD_CAP,
            "cross": GAIT_CROSS_REWARD_CAP,
            "starvation": GAIT_STARVATION_REWARD_CAP,
            "total": GAIT_TOTAL_REWARD_CAP,
        },
        "stair_memory": {
            "contract": "p35_camera_timing_v1",
            "rollout_frames": 128,
            "tbptt_frames": 128,
            "near_clip_m": [0.10, 0.25],
            "near_clip_distribution": "0.10+0.15*Beta(1,4)",
            "memory_target_gradient_ratio": MEMORY_TARGET_GRADIENT_RATIO,
            "memory_hard_gradient_ratio": MEMORY_MAX_GRADIENT_RATIO,
            "high_level_frozen": True,
            "low_level_cnn_frozen": True,
            "clean_teacher": "selected_f2_low_policy_snapshot",
            "capture_rate_hz": 30.0,
            "control_rate_hz": 50.0,
            "feature_fifo_frames": 10,
            "active_delay_max_ms": 150.0,
            "shadow_delay_max_ms": 250.0,
            "teacher_selection": "pixel_fault_or_delivered_feature_age_positive",
        },
        "adapter_replay_schedule": [
            {"start_s": 0.0, "end_s": 4500.0, "latest_recent_parent": [0.50, 0.25, 0.25]},
            {"start_s": 4500.0, "end_s": 6300.0, "latest_recent_parent": [0.60, 0.25, 0.15]},
            {"start_s": 6300.0, "end_s": 7200.0, "latest_recent_parent": [0.75, 0.15, 0.10]},
        ],
        "push": {
            "event_term": "push_robot",
            "implementation": "event_manager_public_get_set_wrapper",
            "warm_start_s": 4500.0,
            "full_start_s": 5400.0,
            "interval_s": [12.0, 18.0],
            "warm_velocity_xy_m_s": 0.05,
            "full_velocity_xy_m_s": 0.08,
            "post_push_reward_grace_s": 0.40,
        },
        "gait_baseline_continuous_stride": GAIT_BASELINE_CONTINUOUS_STRIDE,
        "gait_baseline": {
            "version": GAIT_BASELINE_VERSION,
            "terrain_buckets": ["slope", "slope_inv", "stairs", "stairs_inv"],
            "terrain_column_boundaries": list(TERRAIN_COLUMN_BUCKET_BOUNDARIES),
            "motion_buckets": ["low_speed", "forward", "turn_lateral"],
            "excluded_command_buckets": ["brake_restart", "zero"],
            "fallback_reward_scale": {
                "exact": 1.0,
                "same_terrain": 0.5,
                "global": 0.0,
                "disabled": 0.0,
            },
        },
    }
