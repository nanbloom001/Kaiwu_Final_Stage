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
SESSION_TARGET_SECONDS = 9000.0
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
    P3Phase("lowbase", 0.0, 1800.0, True, True, False),
    P3Phase("lowmild", 1800.0, 3600.0, True, True, False),
    P3Phase("lowmedium", 3600.0, 5400.0, True, True, False),
    P3Phase("adaptercalib", 5400.0, 6000.0, False, True, True),
    P3Phase("highadapt", 6000.0, 7200.0, False, True, True),
    P3Phase("highslow", 7200.0, SESSION_TARGET_SECONDS, False, True, True),
)


def phase_for_elapsed(elapsed_s: float) -> P3Phase:
    elapsed = min(max(float(elapsed_s), 0.0), SESSION_TARGET_SECONDS)
    for phase in PHASES:
        if elapsed < phase.end_s:
            return phase
    return PHASES[-1]


def domain_randomization_index(elapsed_s: float) -> int:
    elapsed = max(0.0, float(elapsed_s))
    if elapsed < 1800.0:
        return 0
    return 1


def materialize_environment_config(usr_conf: dict, elapsed_s: float) -> dict:
    """Expand the P3 wall-clock DR table into worker-consumed configuration."""
    result = copy.deepcopy(usr_conf)
    p3 = result.get("p3_standard_joint", {})
    table = p3.get("domain_randomization", {}) if isinstance(p3, dict) else {}
    index = domain_randomization_index(elapsed_s)

    def item(name, default):
        values = table.get(name, default)
        if not isinstance(values, (tuple, list)) or len(values) <= index:
            raise ValueError(f"P3 domain randomization table {name} is incomplete")
        return values[index]

    friction = item("friction_ranges", [[0.6, 1.3], [0.55, 1.35]])
    added_mass = float(item("base_added_mass_kg", [0.75, 0.85]))
    noise_level = float(item("noise_levels", [0.5, 0.55]))

    result["domain_rand"] = {
        **dict(result.get("domain_rand", {})),
        "enable_domain_rand": True,
        "randomize_friction": True,
        "friction_range": list(friction),
        "randomize_base_mass": added_mass > 0.0,
        "added_mass_range": [-added_mass, added_mass],
        # The platform API cannot express per-env share or reset grace.
        "push_robots": False,
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
        "phase_index": index,
        "base_added_mass_kg": added_mass,
        "worker_command_override": False,
        "low_phase_command_owner": "platform_native_2_to_8_seconds",
    }
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
        "name": "p3_standard_radial_v4",
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
        "physics_randomization": "platform_friction_and_base_mass_only",
        "observation_noise": "platform_explicit_term_bounds_by_phase_level",
        "step_transport": {"action_dim": P3_ACTION_DIM},
        "low_phase_command_owner": "platform_native_2_to_8_seconds",
        "high_phase_command_owner": "p3_high_level_policy",
        "low_level_frozen_while_high_level_owns_command": True,
        "local_out_of_bounds": "diagnostic_only_platform_reset_unavailable",
    }
