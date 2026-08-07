"""Pure tensor primitives shared by P4 teacher and compatibility APIs."""

from __future__ import annotations

import math

import torch

from agent_ppo.feature import p2_contract
from agent_ppo.p4.constants import *  # noqa: F403


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
    severity = torch.clamp((duration - STUCK_SUSTAINED_GRACE_S) / ramp_span, 0.0, 1.0)
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
        torch.isfinite(best_raw) & torch.isfinite(end_raw) & torch.isfinite(earned_raw)
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


def adapter_compatibility_metric_name(reason: str) -> str:
    """Return a platform-safe metric name for one compatibility reason."""
    suffix = (
        "mismatch_response_profile15"
        if reason == "mismatch_response_capability_profile15"
        else str(reason)
    )
    return f"adapter_compat_rejected_{suffix}"


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
        values = torch.nan_to_num(safe5_values, nan=0.0, posinf=1.0, neginf=0.0).clamp(
            0.0, 1.0
        )
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
    goal_eligible = valid & goal_tie_break if safe5 is not None else base & fresh_goal
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
    nav_risk5 = torch.nan_to_num(nav_risk5, nan=1.0, posinf=1.0, neginf=1.0).clamp(
        0.0, 1.0
    )

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
    return (
        safe5,
        valid,
        {
            "nav_risk5": nav_risk5,
            "terrain_passable5": terrain_passable5,
            "height_jump90_5": torch.stack(jump90, dim=-1),
            "height_sector_valid5": sector_valid5.float(),
        },
    )


def teacher_guidance_loss(
    policy_mean_cmd3: torch.Tensor,
    safe3: torch.Tensor,
    goal_xy_m: torch.Tensor,
    predictive_risk: torch.Tensor,
    stuck_active: torch.Tensor,
    teacher_mask: torch.Tensor,
    goal_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    goal_freshness: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
    *,
    min_valid_steps: int = TEACHER_MIN_VALID_STEPS,
    closed_loop_v3: bool = False,
    instant_r4: bool = False,
    instant_repair: bool = False,
) -> dict[str, torch.Tensor]:
    """Return tolerant direction, speed and yaw guidance without a full action teacher."""
    if instant_r4:
        return instant_r4_teacher_guidance_loss(
            policy_mean_cmd3,
            safe3,
            goal_xy_m,
            predictive_risk,
            stuck_active,
            teacher_mask,
            goal_mask,
            sample_weight,
            goal_freshness,
            context_mask,
            min_valid_steps=min_valid_steps,
            instant_repair=instant_repair,
        )
    if policy_mean_cmd3.ndim != 2 or policy_mean_cmd3.shape[1] != 3:
        raise ValueError("P4 teacher loss expects policy_mean_cmd3=[N,3]")
    expected_sectors = 5 if closed_loop_v3 else 3
    if safe3.shape != (policy_mean_cmd3.shape[0], expected_sectors):
        raise ValueError(f"P4 teacher loss safe{expected_sectors} shape drift")
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
    values = torch.nan_to_num(
        safe3.to(policy_mean_cmd3), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
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
        goal_distance_to_sector = torch.abs(bearing[:, None] - sector_angles[None, :])
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
    heading_alignment = torch.cos(heading[:, None] - sector_angles[None, :])
    heading_weights = torch.softmax(8.0 * heading_alignment, dim=-1)
    heading_safe = (heading_weights * values).sum(dim=-1)
    edge_danger = (
        torch.relu(TEACHER_EDGE_SAFE_MIN - heading_safe) / TEACHER_EDGE_SAFE_MIN
    )
    edge_mask = base & (mean_speed > 0.10) & (heading_safe < TEACHER_EDGE_SAFE_MIN)
    edge_direction_error = torch.relu(
        math.cos(math.radians(TEACHER_EDGE_DIRECTION_TOLERANCE_DEG)) - direction_cosine
    )
    edge_speed_cap = TEACHER_EDGE_SPEED_CAP_MIN + (
        TEACHER_EDGE_SPEED_CAP_RANGE * heading_safe
    )
    edge_loss = edge_danger * (
        edge_direction_error.square() + torch.relu(mean_speed - edge_speed_cap).square()
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


def instant_r4_teacher_masks(
    policy_mean_cmd3: torch.Tensor,
    safe5: torch.Tensor,
    teacher_mask: torch.Tensor,
    stuck_active: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Classify mutually exclusive R4 normal, edge and recovery teacher samples."""
    if policy_mean_cmd3.ndim != 2 or policy_mean_cmd3.shape[1] != 3:
        raise ValueError("P4 instant R4 teacher expects policy_mean_cmd3=[N,3]")
    if safe5.shape != (policy_mean_cmd3.shape[0], 5):
        raise ValueError("P4 instant R4 teacher expects safe5=[N,5]")
    count = policy_mean_cmd3.shape[0]

    def _flat(value: torch.Tensor, name: str, *, boolean: bool) -> torch.Tensor:
        result = torch.as_tensor(value, device=policy_mean_cmd3.device).reshape(-1)
        if result.numel() != count:
            raise ValueError(f"P4 instant R4 teacher {name} shape drift")
        return result.bool() if boolean else result.to(dtype=policy_mean_cmd3.dtype)

    base = _flat(teacher_mask, "teacher_mask", boolean=True)
    stuck = _flat(stuck_active, "stuck_active", boolean=True)
    values = torch.nan_to_num(
        safe5.to(policy_mean_cmd3), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    sector_angles = policy_mean_cmd3.new_tensor(TEACHER_SAFE5_ANGLES_DEG) * (
        math.pi / 180.0
    )
    best_safe, best_index = values.max(dim=-1)
    top2 = torch.topk(values, k=2, dim=-1).values
    safe_margin = top2[:, 0] - top2[:, 1]
    mean_xy = policy_mean_cmd3[:, :2]
    mean_speed = torch.linalg.vector_norm(mean_xy, dim=-1)
    heading = torch.atan2(mean_xy[:, 1], mean_xy[:, 0])
    heading_weights = torch.softmax(
        8.0 * torch.cos(heading[:, None] - sector_angles[None, :]), dim=-1
    )
    heading_safe = (heading_weights * values).sum(dim=-1)
    best_heading_safe_gap = best_safe - heading_safe
    edge_candidate = (
        (best_safe >= TEACHER_SAFE_MIN)
        & (best_heading_safe_gap >= 0.15)
        & (heading_safe < TEACHER_SAFE_MIN)
        & (mean_speed > 0.10)
    )
    left_clearance = values[:, :2].amax(dim=-1)
    right_clearance = values[:, 3:].amax(dim=-1)
    side_difference = left_clearance - right_clearance
    recovery_candidate = (
        stuck
        & (torch.maximum(left_clearance, right_clearance) >= TEACHER_SAFE_MIN)
        & (side_difference.abs() >= TEACHER_RECOVERY_SIDE_MARGIN)
    )
    recovery = base & recovery_candidate
    edge = base & edge_candidate & ~recovery
    normal = base & ~edge & ~recovery
    far_side = (best_index == 0) | (best_index == 4)
    recovery_wz = recovery & far_side
    recovery_vy = recovery & ~far_side
    return {
        "teacher_normal_mask": normal,
        "teacher_edge_mask": edge,
        "teacher_recovery_mask": recovery,
        "teacher_recovery_vy_mask": recovery_vy,
        "teacher_recovery_wz_mask": recovery_wz,
        "teacher_best_safe": best_safe,
        "teacher_safe_margin": safe_margin,
        "teacher_heading_safe": heading_safe,
        "teacher_best_heading_safe_gap": best_heading_safe_gap,
        "teacher_recovery_side_sign": torch.sign(side_difference),
        "teacher_recovery_far_side": far_side.to(dtype=policy_mean_cmd3.dtype),
    }


def instant_r4_teacher_guidance_loss(
    policy_mean_cmd3: torch.Tensor,
    safe5: torch.Tensor,
    goal_xy_m: torch.Tensor,
    predictive_risk: torch.Tensor,
    stuck_active: torch.Tensor,
    teacher_mask: torch.Tensor,
    goal_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    goal_freshness: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
    *,
    min_valid_steps: int = TEACHER_MIN_VALID_STEPS,
    instant_repair: bool = False,
) -> dict[str, torch.Tensor]:
    """Return R4 teacher loss with one mutually exclusive recovery escape axis."""
    if policy_mean_cmd3.ndim != 2 or policy_mean_cmd3.shape[1] != 3:
        raise ValueError("P4 instant R4 teacher loss expects policy_mean_cmd3=[N,3]")
    if safe5.shape != (policy_mean_cmd3.shape[0], 5):
        raise ValueError("P4 instant R4 teacher loss expects safe5=[N,5]")
    if goal_xy_m.shape != (policy_mean_cmd3.shape[0], 2):
        raise ValueError("P4 instant R4 teacher loss goal_xy_m shape drift")
    count = policy_mean_cmd3.shape[0]

    def _flat(value: torch.Tensor, name: str, *, boolean: bool = False) -> torch.Tensor:
        result = torch.as_tensor(value, device=policy_mean_cmd3.device).reshape(-1)
        if result.numel() != count:
            raise ValueError(f"P4 instant R4 teacher loss {name} shape drift")
        return result.bool() if boolean else result.to(dtype=policy_mean_cmd3.dtype)

    masks = instant_r4_teacher_masks(
        policy_mean_cmd3, safe5, teacher_mask, stuck_active
    )
    base = _flat(teacher_mask, "teacher_mask", boolean=True)
    goal_valid = _flat(goal_mask, "goal_mask", boolean=True)
    context = (
        base
        if context_mask is None
        else _flat(context_mask, "context_mask", boolean=True)
    )
    freshness = (
        torch.ones(count, device=policy_mean_cmd3.device, dtype=policy_mean_cmd3.dtype)
        if goal_freshness is None
        else _flat(goal_freshness, "goal_freshness").clamp(0.0, 1.0)
    )
    weights = (
        torch.ones(count, device=policy_mean_cmd3.device, dtype=policy_mean_cmd3.dtype)
        if sample_weight is None
        else _flat(sample_weight, "sample_weight").clamp_min(0.0)
    )
    values = torch.nan_to_num(
        safe5.to(policy_mean_cmd3), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    best_safe, pure_best_index = values.max(dim=-1)
    top2 = torch.topk(values, k=2, dim=-1).values
    sector_angles = policy_mean_cmd3.new_tensor(TEACHER_SAFE5_ANGLES_DEG) * (
        math.pi / 180.0
    )
    goal = torch.nan_to_num(
        goal_xy_m.to(policy_mean_cmd3), nan=0.0, posinf=0.0, neginf=0.0
    )
    goal_distance = torch.linalg.vector_norm(goal, dim=-1)
    bearing = torch.atan2(goal[:, 1], goal[:, 0]).clamp(
        min=math.radians(-75.0), max=math.radians(75.0)
    )
    tied_safe = (values >= TEACHER_SAFE_MIN) & (
        values >= best_safe[:, None] - TEACHER_GOAL_SAFE_TIE_MARGIN
    )
    goal_distance_to_sector = torch.where(
        tied_safe,
        torch.abs(bearing[:, None] - sector_angles[None, :]),
        torch.full_like(values, 1.0e6),
    )
    goal_best_index = goal_distance_to_sector.argmin(dim=-1)
    goal_sector_index = torch.abs(bearing[:, None] - sector_angles[None, :]).argmin(
        dim=-1
    )
    goal_direction_safe = (
        values.gather(1, goal_sector_index.unsqueeze(-1)).squeeze(-1)
        >= TEACHER_SAFE_MIN
    )
    goal_tie = goal_valid & ((top2[:, 0] - top2[:, 1]) <= TEACHER_GOAL_SAFE_TIE_MARGIN)
    best_index = torch.where(goal_tie, goal_best_index, pure_best_index)
    safe_angle = sector_angles[best_index]
    safe_direction = torch.stack((torch.cos(safe_angle), torch.sin(safe_angle)), dim=-1)
    mean_xy = policy_mean_cmd3[:, :2]
    mean_speed = torch.linalg.vector_norm(mean_xy, dim=-1)
    mean_direction = mean_xy / mean_speed.unsqueeze(-1).clamp_min(1.0e-6)
    direction_cosine = (mean_direction * safe_direction).sum(dim=-1)
    direction_loss = torch.relu(
        math.cos(math.radians(TEACHER_DIRECTION_TOLERANCE_DEG)) - direction_cosine
    ).square()
    edge_direction_error = torch.relu(
        math.cos(math.radians(TEACHER_EDGE_DIRECTION_TOLERANCE_DEG)) - direction_cosine
    )
    heading_safe = masks["teacher_heading_safe"]
    edge_speed_cap = TEACHER_EDGE_SPEED_CAP_MIN + (
        TEACHER_EDGE_SPEED_CAP_RANGE * heading_safe
    )
    edge_loss = torch.relu(TEACHER_SAFE_MIN - heading_safe).div(TEACHER_SAFE_MIN) * (
        edge_direction_error.square() + torch.relu(mean_speed - edge_speed_cap).square()
    )
    risk = _flat(predictive_risk, "predictive_risk").clamp(0.0, 1.0)
    normal = masks["teacher_normal_mask"]
    edge = masks["teacher_edge_mask"]
    recovery = masks["teacher_recovery_mask"]
    speed_mask = normal & (risk >= TEACHER_SPEED_RISK_MIN)
    speed_cap = 0.20 + 0.45 * best_safe
    speed_loss = torch.relu(mean_speed - speed_cap).square()
    desired_yaw = safe_angle
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
                torch.full_like(bearing_abs_deg, 0.18),
            ),
        ),
    )
    yaw_mask = normal & (required_wz > 0.0)
    signed_wz = torch.sign(desired_yaw) * policy_mean_cmd3[:, 2]
    yaw_loss = torch.relu(required_wz - signed_wz).square()
    high_risk_cap = torch.where(
        risk >= TEACHER_SPEED_RISK_MIN,
        torch.full_like(mean_speed, 0.12),
        torch.full_like(mean_speed, 0.25),
    )
    recovery_translation_loss = torch.relu(mean_speed - high_risk_cap).square()
    side_sign = masks["teacher_recovery_side_sign"]
    signed_vy = side_sign * policy_mean_cmd3[:, 1]
    recovery_vy_loss = torch.relu(TEACHER_RECOVERY_MIN_ABS_VY - signed_vy).square()
    recovery_wz_target = torch.where(
        masks["teacher_recovery_far_side"].bool(),
        torch.full_like(signed_wz, 0.18),
        torch.full_like(signed_wz, 0.12),
    )
    recovery_wz_loss = torch.relu(
        recovery_wz_target - side_sign * policy_mean_cmd3[:, 2]
    ).square()
    recovery_axis_loss = torch.where(
        masks["teacher_recovery_vy_mask"], recovery_vy_loss, recovery_wz_loss
    )
    recovery_loss = recovery_translation_loss + recovery_axis_loss
    stale_mask = torch.zeros_like(base)
    near_goal_mask = torch.zeros_like(base)
    stale_loss = torch.zeros_like(mean_speed)
    near_goal_loss = torch.zeros_like(mean_speed)
    if instant_repair:
        has_goal = goal_distance > 1.0e-3
        stale_mask = context & (freshness < 0.25)
        stale_speed_cap = torch.where(
            has_goal,
            torch.full_like(mean_speed, INSTANT_REPAIR_STALE_TRANSLATION_CAP_M_S),
            torch.zeros_like(mean_speed),
        )
        stale_loss = (
            torch.relu(mean_speed - stale_speed_cap).square()
            + 0.25
            * torch.relu(
                policy_mean_cmd3[:, 2].abs() - INSTANT_REPAIR_STALE_YAW_CAP_RAD_S
            ).square()
        )

        near_goal_mask = (
            context
            & goal_valid
            & (freshness >= NEAR_GOAL_CAPTURE_FRESHNESS_MIN)
            & (goal_distance >= NEAR_GOAL_CAPTURE_MIN_DISTANCE_M)
            & (goal_distance <= NEAR_GOAL_CAPTURE_MAX_DISTANCE_M)
            & goal_direction_safe
        )
        goal_direction = goal / goal_distance.unsqueeze(-1).clamp_min(1.0e-6)
        parallel_speed = (mean_xy * goal_direction).sum(dim=-1)
        perpendicular = mean_xy - parallel_speed.unsqueeze(-1) * goal_direction
        perpendicular_speed = torch.linalg.vector_norm(perpendicular, dim=-1)
        near_progress = torch.clamp(
            (goal_distance - NEAR_GOAL_CAPTURE_MIN_DISTANCE_M)
            / max(
                NEAR_GOAL_CAPTURE_MAX_DISTANCE_M - NEAR_GOAL_CAPTURE_MIN_DISTANCE_M,
                1.0e-6,
            ),
            0.0,
            1.0,
        )
        near_speed_cap = INSTANT_REPAIR_NEAR_GOAL_MIN_SPEED_M_S + near_progress * (
            INSTANT_REPAIR_NEAR_GOAL_MAX_SPEED_M_S
            - INSTANT_REPAIR_NEAR_GOAL_MIN_SPEED_M_S
        )
        near_goal_loss = (
            perpendicular_speed.square()
            + torch.relu(mean_speed - near_speed_cap).square()
            + 0.25 * torch.relu(-parallel_speed).square()
        )

    valid_steps = (base | stale_mask | near_goal_mask).sum()
    active = valid_steps >= int(min_valid_steps)

    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected_weight = weights * mask.to(weights.dtype)
        return (value * selected_weight).sum() / selected_weight.sum().clamp_min(1.0)

    direction = _masked_mean(direction_loss, normal)
    speed = _masked_mean(speed_loss, speed_mask)
    yaw = _masked_mean(yaw_loss, yaw_mask)
    edge_value = _masked_mean(edge_loss, edge)
    recovery_value = _masked_mean(recovery_loss, recovery)
    stale_value = _masked_mean(stale_loss, stale_mask)
    near_goal_value = _masked_mean(near_goal_loss, near_goal_mask)
    total = (
        0.34 * direction
        + 0.085 * speed
        + 0.2125 * yaw
        + 0.1275 * edge_value
        + 0.085 * recovery_value
        + 0.075 * stale_value
        + 0.075 * near_goal_value
        if instant_repair
        else (
            0.40 * direction
            + 0.10 * speed
            + 0.25 * yaw
            + 0.15 * edge_value
            + 0.10 * recovery_value
        )
    )
    active_float = active.to(dtype=policy_mean_cmd3.dtype)
    result = {
        "loss": total * active_float,
        "direction": direction * active_float,
        "speed": speed * active_float,
        "yaw": yaw * active_float,
        "edge": edge_value * active_float,
        "recovery": recovery_value * active_float,
        "stale_goal": stale_value * active_float,
        "near_goal": near_goal_value * active_float,
        "teacher_valid_steps": valid_steps.to(dtype=policy_mean_cmd3.dtype),
        "teacher_loss_active": active_float,
        "teacher_direction_mask": normal.to(dtype=policy_mean_cmd3.dtype),
        "teacher_speed_mask": speed_mask.to(dtype=policy_mean_cmd3.dtype),
        "teacher_yaw_mask": yaw_mask.to(dtype=policy_mean_cmd3.dtype),
        "teacher_edge_mask": edge.to(dtype=policy_mean_cmd3.dtype),
        "teacher_edge_active_share": edge.to(dtype=policy_mean_cmd3.dtype).mean(),
        "teacher_recovery_mask": recovery.to(dtype=policy_mean_cmd3.dtype),
        "teacher_recovery_active_share": recovery.to(
            dtype=policy_mean_cmd3.dtype
        ).mean(),
        "teacher_stale_goal_mask": stale_mask.to(dtype=policy_mean_cmd3.dtype),
        "teacher_stale_goal_active_share": stale_mask.to(
            dtype=policy_mean_cmd3.dtype
        ).mean(),
        "teacher_near_goal_mask": near_goal_mask.to(dtype=policy_mean_cmd3.dtype),
        "teacher_near_goal_active_share": near_goal_mask.to(
            dtype=policy_mean_cmd3.dtype
        ).mean(),
        "teacher_goal_tie_mask": goal_tie.to(dtype=policy_mean_cmd3.dtype),
        "teacher_recovery_vy_mask": masks["teacher_recovery_vy_mask"].to(
            dtype=policy_mean_cmd3.dtype
        ),
        "teacher_recovery_wz_mask": masks["teacher_recovery_wz_mask"].to(
            dtype=policy_mean_cmd3.dtype
        ),
    }
    result.update(masks)
    return result


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
    raw = torch.where(
        eligible, lateral + s_turn + extra_path, torch.zeros_like(lateral)
    )
    total = raw.clamp_min(OPEN_STRAIGHT_TOTAL_FLOOR)
    return total, {
        "open_straight_eligible": eligible.float(),
        "open_straight_lateral_penalty": torch.where(
            eligible, lateral, torch.zeros_like(lateral)
        ),
        "open_straight_s_turn_penalty": torch.where(
            eligible, s_turn, torch.zeros_like(s_turn)
        ),
        "open_straight_extra_path_penalty": torch.where(
            eligible, extra_path, torch.zeros_like(extra_path)
        ),
        "open_straight_goal_bearing_abs_rad": bearing,
        "open_straight_boundary_distance_m": boundary_distance_m.reshape(-1),
    }
