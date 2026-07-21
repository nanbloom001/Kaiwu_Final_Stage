# -*- coding: UTF-8 -*-
"""Training-only hard-segment reset sampling for ST7-Opt5.

The hook replaces only the configured ``reset_base`` event.  It first runs the
platform reset function so difficulty sampling and the normal reset pipeline
remain intact, then moves selected environments to a safe point immediately
before a configured track segment.  Evaluation never installs this hook.
"""

from __future__ import annotations

import functools
import inspect
import math
from typing import Any

import torch


START_NAMES = (
    "full_track",
    "stairs_down",
    "slope_down",
    "maze_entry",
    "stairs_up",
)


_HARD_START_HOOK_CONFIGURED = False


def is_hard_start_hook_configured() -> bool:
    return _HARD_START_HOOK_CONFIGURED


def _log_initialized_state_once(env, state) -> None:
    if bool(getattr(env, "_hard_start_replay_startup_logged", False)):
        return
    counts = state["start_counts"].detach().cpu().tolist()
    print(
        "[Opt5HardStart] initialized in Isaac worker: "
        f"start_counts={dict(zip(START_NAMES, counts))}"
    )
    env._hard_start_replay_startup_logged = True


def _normalized_weights(values: list[float], expected: int) -> torch.Tensor:
    weights = torch.as_tensor(values, dtype=torch.float32)
    if weights.numel() != expected or torch.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError(
            f"hard_start_replay.hard_weights must contain {expected} non-negative values"
        )
    return weights / weights.sum()


def _sample_start_kinds(num_samples: int, config: dict, device) -> torch.Tensor:
    """Return 0=full track, 1..4=hard-start type."""
    full_ratio = float(config.get("full_track_ratio", 0.5))
    if not 0.0 <= full_ratio <= 1.0:
        raise ValueError("hard_start_replay.full_track_ratio must be in [0, 1]")

    kinds = torch.zeros(num_samples, dtype=torch.long, device=device)
    hard_mask = torch.rand(num_samples, device=device) >= full_ratio
    hard_count = int(hard_mask.sum().item())
    if hard_count:
        weights = _normalized_weights(
            list(config.get("hard_weights", [0.45, 0.25, 0.20, 0.10])),
            expected=4,
        ).to(device)
        kinds[hard_mask] = torch.multinomial(
            weights,
            hard_count,
            replacement=True,
        ) + 1
    return kinds


def _quat_from_roll_pitch_yaw(roll, pitch, yaw):
    """Build WXYZ quaternions without importing Isaac modules at import time."""
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=1,
    )


def _resolve_robot(env, asset_cfg):
    asset_name = getattr(asset_cfg, "name", "robot") if asset_cfg is not None else "robot"
    return env.scene[asset_name]


def _resolve_track_sequence(config: dict) -> list[str]:
    sequence = config.get("track_sequence", [])
    if not isinstance(sequence, (list, tuple)) or not sequence:
        raise ValueError("hard_start_replay.track_sequence must be a non-empty list")
    return [str(name) for name in sequence]


def _segment_rows(config: dict) -> torch.Tensor:
    sequence = _resolve_track_sequence(config)
    requested = config.get(
        "hard_segments",
        [
            "pyramid_stairs_inv",
            "pyramid_slope_inv",
            "open_entry_maze",
            "pyramid_stairs",
        ],
    )
    if not isinstance(requested, (list, tuple)) or len(requested) != 4:
        raise ValueError("hard_start_replay.hard_segments must contain four names")
    missing = [name for name in requested if name not in sequence]
    if missing:
        raise ValueError(f"hard-start segments are absent from track_sequence: {missing}")
    return torch.tensor([sequence.index(name) for name in requested], dtype=torch.long)


def _state_for_env(env, num_envs: int, device):
    state = getattr(env, "_hard_start_replay_state", None)
    if state is None or state["start_kind"].numel() != num_envs:
        state = {
            "start_kind": torch.zeros(num_envs, dtype=torch.long, device=device),
            "start_counts": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            "episode_counts": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            "success_counts": torch.zeros(len(START_NAMES), dtype=torch.long, device=device),
            "reason_counts": {},
        }
        env._hard_start_replay_state = state
    return state


def _set_env_origins(env, terrain, env_ids, origins):
    for owner in (terrain, getattr(env, "scene", None)):
        target = getattr(owner, "env_origins", None)
        if torch.is_tensor(target) and target.shape[0] >= env.num_envs:
            target[env_ids] = origins.to(device=target.device, dtype=target.dtype)


def _apply_hard_start_poses(env, env_ids, kinds, config: dict, asset_cfg=None):
    terrain = env.scene.terrain
    terrain_origins = getattr(terrain, "terrain_origins", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if not (torch.is_tensor(terrain_origins) and terrain_origins.ndim == 3):
        raise RuntimeError("Track terrain_origins[row, col] is unavailable")
    if not (torch.is_tensor(terrain_types) and torch.is_tensor(terrain_levels)):
        raise RuntimeError("Track terrain_levels/terrain_types are unavailable")

    device = terrain_origins.device
    env_ids = env_ids.to(device=device, dtype=torch.long)
    kinds = kinds.to(device=device, dtype=torch.long)
    hard_rows = _segment_rows(config).to(device)
    if torch.any(kinds <= 0):
        raise ValueError("_apply_hard_start_poses accepts hard-start kinds only")
    target_rows = hard_rows[kinds - 1]

    difficulty_cols = terrain_types[env_ids].long().clamp(
        min=0,
        max=terrain_origins.shape[1] - 1,
    )
    target_rows = target_rows.clamp(min=0, max=terrain_origins.shape[0] - 1)
    target_origins = terrain_origins[target_rows, difficulty_cols]

    previous_rows = target_rows - 1
    previous_origins = terrain_origins[previous_rows, difficulty_cols]
    direction_xy = target_origins[:, :2] - previous_origins[:, :2]

    segment_length = torch.linalg.norm(direction_xy, dim=1).clamp(min=1.0)
    direction_xy = direction_xy / segment_length.unsqueeze(1)
    lateral_xy = torch.stack((-direction_xy[:, 1], direction_xy[:, 0]), dim=1)

    boundary_xy = 0.5 * (previous_origins[:, :2] + target_origins[:, :2])
    # Track tiles meet at their zero-height border. ``terrain_origins[..., 2]``
    # is the center/platform spawn height for pyramid terrain, so interpolating
    # it here would place a boundary start in mid-air.
    boundary_z = torch.full_like(
        target_origins[:, 2],
        float(config.get("track_ground_z", 0.0)),
    )
    offset_range = config.get("approach_offset_m", [0.5, 1.2])
    offset_low, offset_high = float(offset_range[0]), float(offset_range[1])
    approach_offset = offset_low + (offset_high - offset_low) * torch.rand(
        env_ids.numel(), device=device
    )
    lateral_range = config.get("lateral_offset_m", [0.05, 0.08])
    lateral_min, lateral_max = float(lateral_range[0]), float(lateral_range[1])
    lateral_magnitude = lateral_min + (lateral_max - lateral_min) * torch.rand(
        env_ids.numel(), device=device
    )
    lateral_sign = torch.where(
        torch.rand(env_ids.numel(), device=device) < 0.5,
        -torch.ones(env_ids.numel(), device=device),
        torch.ones(env_ids.numel(), device=device),
    )
    lateral_offset = lateral_sign * lateral_magnitude
    spawn_xy = (
        boundary_xy
        - approach_offset.unsqueeze(1) * direction_xy
        + lateral_offset.unsqueeze(1) * lateral_xy
    )

    robot = _resolve_robot(env, asset_cfg)
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :2] = spawn_xy.to(root_state.dtype)
    root_state[:, 2] = boundary_z.to(root_state.dtype) + robot.data.default_root_state[
        env_ids, 2
    ]

    yaw_center = torch.atan2(direction_xy[:, 1], direction_xy[:, 0])
    yaw_range = config.get("yaw_offset_deg", [3.0, 5.0])
    yaw_min = math.radians(float(yaw_range[0]))
    yaw_max = math.radians(float(yaw_range[1]))
    yaw_magnitude = yaw_min + (yaw_max - yaw_min) * torch.rand_like(yaw_center)
    yaw_sign = torch.where(
        torch.rand_like(yaw_center) < 0.5,
        -torch.ones_like(yaw_center),
        torch.ones_like(yaw_center),
    )
    yaw = yaw_center + yaw_sign * yaw_magnitude
    roll_pitch_max = float(config.get("roll_pitch_offset_max_rad", 0.025))
    roll = (2.0 * torch.rand_like(yaw) - 1.0) * roll_pitch_max
    pitch = (2.0 * torch.rand_like(yaw) - 1.0) * roll_pitch_max
    root_state[:, 3:7] = _quat_from_roll_pitch_yaw(roll, pitch, yaw).to(root_state.dtype)

    entry_speeds = torch.as_tensor(
        config.get("entry_speeds_mps", [0.0, 0.60, 1.00, 0.70, 0.60]),
        device=device,
        dtype=root_state.dtype,
    )
    if entry_speeds.numel() != len(START_NAMES):
        raise ValueError(f"entry_speeds_mps must contain {len(START_NAMES)} values")
    speed_jitter = float(config.get("entry_speed_jitter_mps", 0.04))
    speeds = entry_speeds[kinds] + (
        2.0 * torch.rand(env_ids.numel(), device=device, dtype=root_state.dtype) - 1.0
    ) * speed_jitter
    speeds = torch.clamp(speeds, min=0.0)
    root_state[:, 7:9] = direction_xy.to(root_state.dtype) * speeds.unsqueeze(1)
    root_state[:, 9:13] = 0.0

    terrain_levels[env_ids] = target_rows.to(terrain_levels.dtype)
    _set_env_origins(env, terrain, env_ids, target_origins)
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def _record_start_assignments(env, env_ids, kinds):
    state = _state_for_env(env, env.num_envs, env_ids.device)
    state["start_kind"][env_ids] = kinds
    state["start_counts"] += torch.bincount(kinds, minlength=len(START_NAMES))


def _build_reset_wrapper(original, config: dict):
    if getattr(original, "_is_hard_start_replay_wrapper", False):
        return original

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        bound = inspect.signature(original).bind_partial(*args, **kwargs)
        env = bound.arguments.get("env", args[0] if args else None)
        env_ids = bound.arguments.get("env_ids", args[1] if len(args) > 1 else None)
        asset_cfg = bound.arguments.get("asset_cfg")
        if env is None or bool(getattr(env, "_is_eval", False)):
            return result
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=env.device)
        else:
            env_ids = env_ids.to(env.device)
        env_ids = env_ids.long().view(-1)
        if env_ids.numel() == 0:
            return result

        kinds = _sample_start_kinds(env_ids.numel(), config, env_ids.device)
        full_ids = env_ids[kinds == 0]
        if full_ids.numel():
            # The platform reset function already owns the exact full-track
            # start semantics. Reuse its evaluation path for this subset.
            full_bound = inspect.signature(original).bind_partial(*args, **kwargs)
            full_bound.arguments["env_ids"] = full_ids
            had_is_eval = hasattr(env, "_is_eval")
            old_is_eval = getattr(env, "_is_eval", False)
            env._is_eval = True
            try:
                original(*full_bound.args, **full_bound.kwargs)
            finally:
                if had_is_eval:
                    env._is_eval = old_is_eval
                else:
                    delattr(env, "_is_eval")

        hard_mask = kinds > 0
        if hard_mask.any():
            _apply_hard_start_poses(
                env,
                env_ids[hard_mask],
                kinds[hard_mask],
                config,
                asset_cfg=asset_cfg,
            )
        _record_start_assignments(env, env_ids, kinds)
        return result

    wrapped._is_hard_start_replay_wrapper = True
    return wrapped


def install_hard_start_replay_event(env_cfg, config: dict) -> bool:
    """Wrap ``env_cfg.events.reset_base`` before EventManager construction."""
    global _HARD_START_HOOK_CONFIGURED
    if not bool(config.get("enabled", False)):
        return False
    events = getattr(env_cfg, "events", None)
    reset_term = getattr(events, "reset_base", None)
    original = getattr(reset_term, "func", None)
    if original is None:
        raise RuntimeError("Cannot install hard-start replay: reset_base event is missing")
    reset_term.func = _build_reset_wrapper(original, config)
    # Keep a copy on cfg for startup diagnostics without changing the public API.
    env_cfg._hard_start_replay_config = dict(config)
    _HARD_START_HOOK_CONFIGURED = True
    return True


def initialize_hard_start_replay(env) -> bool:
    """Initialize the first hard-start batch inside the Isaac worker process."""
    if not _HARD_START_HOOK_CONFIGURED or bool(getattr(env, "_is_eval", False)):
        return False
    state = getattr(env, "_hard_start_replay_state", None)
    if state is not None:
        _log_initialized_state_once(env, state)
        return False

    event_manager = getattr(env, "event_manager", None)
    get_term_cfg = getattr(event_manager, "get_term_cfg", None)
    if not callable(get_term_cfg):
        raise RuntimeError("Cannot initialize hard starts: EventManager is unavailable")
    try:
        reset_term = get_term_cfg("reset_base")
    except Exception as exc:
        raise RuntimeError("Cannot initialize hard starts: reset_base is missing") from exc
    reset_func = getattr(reset_term, "func", None)
    if not getattr(reset_func, "_is_hard_start_replay_wrapper", False):
        raise RuntimeError(
            "Cannot initialize hard starts: reset_base wrapper was not preserved"
        )

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    reset_func(
        env,
        env_ids,
        **dict(getattr(reset_term, "params", {}) or {}),
    )
    state = getattr(env, "_hard_start_replay_state", None)
    if state is None:
        raise RuntimeError("Cannot initialize hard starts: reset state was not created")
    _log_initialized_state_once(env, state)
    return True


def publish_hard_start_metrics(env) -> None:
    """Publish reset-distribution metrics through Isaac Lab step extras."""
    state = getattr(env, "_hard_start_replay_state", None)
    extras = getattr(env, "extras", None)
    if state is None or not isinstance(extras, dict):
        return
    kinds = state["start_kind"]
    starts = state["start_counts"]
    extras["hard_start_metrics"] = {
        "full_track_ratio": (kinds == 0).float().mean().item(),
        "hard_start_ratio": (kinds > 0).float().mean().item(),
        "stairs_down_starts": starts[1].item(),
        "slope_down_starts": starts[2].item(),
        "maze_entry_starts": starts[3].item(),
        "stairs_up_starts": starts[4].item(),
    }
