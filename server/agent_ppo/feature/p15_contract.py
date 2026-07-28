#!/usr/bin/env python3
"""Stable contracts for the P1.5 response-adapter training stage."""

from __future__ import annotations

import hashlib
import json
from typing import Any


P15_SCHEDULE_MODE = "p15_response_adapter_v1"
P15_RUN_NAME = "p15resp8h"
P15_TASK_END_HOURS = 8.0
P15_LOW_LEVEL_FREEZE_HOURS = 7.0

CRITIC_OBS_DIM = 316
RESPONSE_AUX_DIM = 30
PRIVILEGED_WIRE_DIM = CRITIC_OBS_DIM + RESPONSE_AUX_DIM
RESPONSE_OBSERVATION_DIM = 32
RESPONSE_PROFILE_DIM = 16

P15_PHASE_LABELS = (
    "responsebase",
    "responseexpand",
    "responsefull",
    "responsecalib",
)

PHASE_BOUNDARIES_H = (0.5, 1.25, 2.0, 7.0, 8.0)
PHASE_REPLAY_PROBABILITY = (1.0, 0.35, 0.30, 0.25, 0.25)

COMMAND_FAMILIES = (
    "joint",
    "straight",
    "pureyaw",
    "transition",
    "lateral",
)
COMMAND_FAMILY_WEIGHTS = (0.65, 0.12, 0.12, 0.06, 0.05)

TRAJECTORY_MODES = ("smooth", "step", "hold")
TRAJECTORY_MODE_WEIGHTS = (0.50, 0.30, 0.20)

CHANGE_MODES = ("vx_only", "wz_only", "both")
CHANGE_MODE_WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

SLEW_RATE = (0.30, 0.30, 1.00)
CONTROL_DT_S = 0.02
TARGET_PERIOD_FRAMES = 10

# worker privileged wire layout; all slices are half-open.
RESPONSE_AUX_LAYOUT = {
    "active_target_cmd3": (0, 3),
    "exec_cmd3": (3, 6),
    "measured_velocity3": (6, 9),
    "velocity_valid": (9, 10),
    "velocity_age": (10, 11),
    "feedback_source": (11, 12),
    "true_velocity3": (12, 15),
    "true_pose3": (15, 18),
    "ang_vel3": (18, 21),
    "projected_gravity3": (21, 24),
    "command_family": (24, 25),
    "trajectory_mode": (25, 26),
    "command_epoch": (26, 27),
    "command_phase": (27, 28),
    "terrain_family": (28, 29),
    "terrain_level": (29, 30),
}

CAPABILITY_PROFILE_VERSION = "piecewise_union_profile_v1"
CAPABILITY_PROFILE15_LAYOUT = (
    "piecewise_union_enabled",
    "main_enabled",
    "main_vx_min",
    "main_vx_max",
    "main_abs_wz_max",
    "main_vy_fixed",
    "vy_specialty_enabled",
    "vy_specialty_abs_vy_max",
    "vy_specialty_vx_fixed",
    "vy_specialty_wz_fixed",
    "source_replay_enabled",
    "source_replay_vx_min",
    "source_replay_vx_max",
    "source_replay_abs_vy_max",
    "source_replay_abs_wz_max",
)
CAPABILITY_PROFILE15 = (
    1.0,
    1.0,
    0.0,
    1.0,
    0.8,
    0.0,
    1.0,
    0.3,
    0.0,
    0.0,
    1.0,
    0.3,
    1.3,
    0.2,
    0.3,
)

FEEDBACK_PROFILE: dict[str, Any] = {
    "version": "p15_feedback_v2",
    "age_clip_s": 0.8,
    "sport": {
        "rate_hz": [35.0, 50.0],
        "delay_s": [0.0, 0.06],
        "noise_std": [0.025, 0.025, 0.035],
        "bias_std": [0.015, 0.015, 0.020],
        "dropout_probability": [0.0, 0.08],
        "freshness_timeout_s": [0.10, 0.22],
    },
    "uwb": {
        "rate_hz": [4.0, 6.0],
        "delay_s": [0.08, 0.30],
        "noise_std": [0.06, 0.06, 0.05],
        "bias_std": [0.03, 0.03, 0.02],
        "low_pass_tau_s": [0.18, 0.45],
        "dropout_probability": [0.0, 0.18],
        "freshness_timeout_s": [0.30, 0.70],
    },
    "imu": {
        "gyro_noise_std": [0.006, 0.006, 0.009],
        "gyro_bias_std": [0.004, 0.004, 0.006],
        "gravity_noise_std": 0.004,
    },
    "short_horizon_velocity_sources": ["sport_mode_xy", "imu_gyro_z"],
    "uwb_role": "long_horizon_only_not_in_response_observation",
    "complete_failure_probability": 0.015,
    "complete_failure_duration_s": [0.10, 0.60],
}


def phase_index(elapsed_h: float) -> int:
    value = max(0.0, float(elapsed_h))
    if value < 0.5:
        return 0
    if value < 1.25:
        return 1
    if value < 2.0:
        return 2
    if value < 7.0:
        return 3
    return 4


def phase_label(elapsed_h: float) -> str:
    index = phase_index(elapsed_h)
    if index == 0:
        return "responsebase"
    if index in (1, 2):
        return "responseexpand"
    if index == 3:
        return "responsefull"
    return "responsecalib"


def command_limits(elapsed_h: float) -> dict[str, tuple[float, float]]:
    index = phase_index(elapsed_h)
    if index == 0:
        return {"vx": (0.0, 1.3), "vy": (-0.2, 0.2), "wz": (-0.3, 0.3)}
    if index == 1:
        return {"vx": (0.0, 1.0), "vy": (-0.2, 0.2), "wz": (-0.5, 0.5)}
    if index == 2:
        return {"vx": (0.0, 1.0), "vy": (-0.25, 0.25), "wz": (-0.65, 0.65)}
    return {"vx": (0.0, 1.0), "vy": (-0.3, 0.3), "wz": (-0.8, 0.8)}


def original_replay_probability(elapsed_h: float) -> float:
    return PHASE_REPLAY_PROBABILITY[phase_index(elapsed_h)]


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def command_contract() -> dict[str, Any]:
    return {
        "version": "p15_command_v1",
        "envelope_type": "piecewise_union_v1",
        "control_hz": 50,
        "target_hz": 5,
        "target_period_frames": TARGET_PERIOD_FRAMES,
        "slew_rate": list(SLEW_RATE),
        "command_family_weights": dict(zip(COMMAND_FAMILIES, COMMAND_FAMILY_WEIGHTS)),
        "trajectory_mode_weights": dict(zip(TRAJECTORY_MODES, TRAJECTORY_MODE_WEIGHTS)),
        "change_mode_weights": dict(zip(CHANGE_MODES, CHANGE_MODE_WEIGHTS)),
        "phase_boundaries_h": list(PHASE_BOUNDARIES_H),
        "command_envelope": {
            "type": "piecewise_union_v1",
            "branches": {
                "main_vx_wz": {
                    "vx": [0.0, 1.0],
                    "vy": [0.0, 0.0],
                    "wz": [-0.8, 0.8],
                },
                "vy_specialty": {
                    "vx": [0.0, 0.0],
                    "vy": [-0.3, 0.3],
                    "wz": [0.0, 0.0],
                },
                "source_replay": {
                    "vx": [0.3, 1.3],
                    "vy": [-0.2, 0.2],
                    "wz": [-0.3, 0.3],
                },
            },
        },
        "capability_profile": {
            "version": CAPABILITY_PROFILE_VERSION,
            "layout": list(CAPABILITY_PROFILE15_LAYOUT),
            "values": list(CAPABILITY_PROFILE15),
        },
    }


def p15_anchor_weights_from_commands(commands):
    """Preserve the parent throughout its supported command envelope.

    The inherited generic helper only anchors positive-forward samples. P1.5
    also replays zero, pure-yaw and lateral commands that commandfull-34728
    already learned, so those samples must retain the S0 anchor as well.
    """
    import torch

    if commands.ndim != 2 or commands.shape[1] < 3:
        raise ValueError(f"expected command tensor [N,3+], got {tuple(commands.shape)}")
    command = commands[:, :3]
    if not bool(torch.isfinite(command).all()):
        raise RuntimeError("P1.5 command tensor contains non-finite values")
    source_supported = (
        (command[:, 0] >= -1.0e-6)
        & (command[:, 0] <= 1.3 + 1.0e-6)
        & (command[:, 1].abs() <= 0.2 + 1.0e-6)
        & (command[:, 2].abs() <= 0.3 + 1.0e-6)
    )
    weights = torch.full(
        (command.shape[0], 1), 0.25, device=command.device, dtype=torch.float32
    )
    weights[source_supported] = 1.0
    return weights


def feedback_contract() -> dict[str, Any]:
    return {
        "version": "p15_feedback_contract_v1",
        "response_observation_dim": RESPONSE_OBSERVATION_DIM,
        "response_profile_dim": RESPONSE_PROFILE_DIM,
        "aux_layout": RESPONSE_AUX_LAYOUT,
        "profile": FEEDBACK_PROFILE,
    }
