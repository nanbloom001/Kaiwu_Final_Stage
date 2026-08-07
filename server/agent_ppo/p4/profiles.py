"""Authoritative P4 training-profile registry and lifecycle policy."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


TRAIN_MODE = "train"
WARM_START_MODE = "warm_start"
EVAL_MODE = "eval"
PROFILE_MODES = frozenset({TRAIN_MODE, WARM_START_MODE, EVAL_MODE})

PROFILE_MAZE_INSTANT_REPAIR2H = "maze_instant_repair2h"
PROFILE_FULL_TRACK = "full_track"
PROFILE_MAZE_CREDIT_REPAIR = "maze_credit_repair"
PROFILE_MAZE_CLOSED_LOOP_V3 = "maze_closed_loop_v3"
PROFILE_MAZE_INSTANT_COMMAND_R4 = "maze_instant_command_r4"


@dataclass(frozen=True)
class TrainingProfileSpec:
    """Immutable identity and allowed lifecycle modes for one P4 profile."""

    name: str
    allowed_modes: frozenset[str]
    maze_only: bool
    instant_command: bool
    schedule_branch: str
    wall_stuck_precedes_success: bool
    run_name: str
    training_hours: float
    target_effective_seconds: int
    task_end_hours: float
    required_platform_wall_seconds: int
    schedule_boundaries_seconds: tuple[float, ...]
    checkpoint_boundaries_seconds: tuple[float, ...]
    legacy: bool = False

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip().lower()
        if normalized_name != self.name or not normalized_name:
            raise ValueError(f"P4 profile name must be canonical: {self.name!r}")
        unknown_modes = set(self.allowed_modes) - PROFILE_MODES
        if unknown_modes:
            raise ValueError(
                f"unsupported P4 profile modes for {self.name!r}: "
                f"{sorted(unknown_modes)!r}"
            )
        if not self.allowed_modes:
            raise ValueError(f"P4 profile {self.name!r} has no allowed modes")
        if self.legacy and TRAIN_MODE in self.allowed_modes:
            raise ValueError(
                f"legacy P4 profile {self.name!r} cannot be an active train profile"
            )
        if not self.run_name:
            raise ValueError(f"P4 profile {self.name!r} has no run name")
        if not self.schedule_branch:
            raise ValueError(f"P4 profile {self.name!r} has no schedule branch")
        if self.training_hours <= 0.0 or self.target_effective_seconds <= 0:
            raise ValueError(f"P4 profile {self.name!r} has an invalid training clock")
        if self.required_platform_wall_seconds < self.target_effective_seconds:
            raise ValueError(
                f"P4 profile {self.name!r} platform wall clock is too short"
            )
        expected_task_hours = self.required_platform_wall_seconds / 3_600.0
        if abs(self.task_end_hours - expected_task_hours) > 1.0e-9:
            raise ValueError(
                f"P4 profile {self.name!r} task_end_hours does not match "
                "required_platform_wall_seconds"
            )
        for label, boundaries in (
            ("schedule", self.schedule_boundaries_seconds),
            ("checkpoint", self.checkpoint_boundaries_seconds),
        ):
            if not boundaries or tuple(sorted(boundaries)) != boundaries:
                raise ValueError(
                    f"P4 profile {self.name!r} has invalid {label} boundaries"
                )
            if boundaries[-1] != float(self.target_effective_seconds):
                raise ValueError(
                    f"P4 profile {self.name!r} {label} boundaries must end at "
                    "target_effective_seconds"
                )

    @property
    def trainable(self) -> bool:
        return TRAIN_MODE in self.allowed_modes

    @property
    def allowed_uses(self) -> frozenset[str]:
        """Compatibility alias for callers that describe modes as uses."""
        return self.allowed_modes

    @property
    def required_platform_wall_hours(self) -> float:
        return self.required_platform_wall_seconds / 3_600.0

    def allows(self, mode: str) -> bool:
        return normalize_profile_mode(mode) in self.allowed_modes


_ALL_MODES = frozenset({TRAIN_MODE, WARM_START_MODE, EVAL_MODE})
_LEGACY_MODES = frozenset({WARM_START_MODE, EVAL_MODE})

_PROFILE_SPECS = (
    TrainingProfileSpec(
        name=PROFILE_MAZE_INSTANT_REPAIR2H,
        allowed_modes=_ALL_MODES,
        maze_only=True,
        instant_command=True,
        schedule_branch="instant_repair2h",
        wall_stuck_precedes_success=False,
        run_name="p4maze2h-instant-repair-r1",
        training_hours=2.0,
        target_effective_seconds=7_200,
        task_end_hours=2.25,
        required_platform_wall_seconds=8_100,
        schedule_boundaries_seconds=(300.0, 1_800.0, 5_400.0, 7_200.0),
        checkpoint_boundaries_seconds=(
            300.0,
            900.0,
            1_800.0,
            3_600.0,
            5_400.0,
            7_200.0,
        ),
    ),
    TrainingProfileSpec(
        name=PROFILE_FULL_TRACK,
        allowed_modes=_ALL_MODES,
        maze_only=False,
        instant_command=False,
        schedule_branch="auto",
        wall_stuck_precedes_success=True,
        run_name="p4full8h-r2",
        training_hours=8.0,
        target_effective_seconds=28_800,
        task_end_hours=8.25,
        required_platform_wall_seconds=29_700,
        schedule_boundaries_seconds=(1_800.0, 7_200.0, 21_600.0, 28_800.0),
        checkpoint_boundaries_seconds=(
            1_800.0,
            7_200.0,
            21_600.0,
            28_800.0,
        ),
    ),
    TrainingProfileSpec(
        name=PROFILE_MAZE_CREDIT_REPAIR,
        allowed_modes=_LEGACY_MODES,
        maze_only=True,
        instant_command=False,
        schedule_branch="credit_repair",
        wall_stuck_precedes_success=True,
        run_name="p4maze2h-credit-repair",
        training_hours=2.0,
        target_effective_seconds=7_200,
        task_end_hours=2.25,
        required_platform_wall_seconds=8_100,
        schedule_boundaries_seconds=(
            600.0,
            1_800.0,
            3_600.0,
            5_400.0,
            6_300.0,
            7_200.0,
        ),
        checkpoint_boundaries_seconds=(
            600.0,
            1_800.0,
            3_600.0,
            5_400.0,
            6_300.0,
            7_200.0,
        ),
        legacy=True,
    ),
    TrainingProfileSpec(
        name=PROFILE_MAZE_CLOSED_LOOP_V3,
        allowed_modes=_LEGACY_MODES,
        maze_only=True,
        instant_command=False,
        schedule_branch="closed_loop_v3",
        wall_stuck_precedes_success=True,
        run_name="p4maze8h-closedloop-r3",
        training_hours=8.0,
        target_effective_seconds=28_800,
        task_end_hours=8.25,
        required_platform_wall_seconds=29_700,
        schedule_boundaries_seconds=(1_800.0, 7_200.0, 21_600.0, 28_800.0),
        checkpoint_boundaries_seconds=(
            1_800.0,
            7_200.0,
            21_600.0,
            28_800.0,
        ),
        legacy=True,
    ),
    TrainingProfileSpec(
        name=PROFILE_MAZE_INSTANT_COMMAND_R4,
        allowed_modes=_LEGACY_MODES,
        maze_only=True,
        instant_command=True,
        schedule_branch="instant_command_r4",
        wall_stuck_precedes_success=True,
        run_name="p4maze8h-instant-r4-inputfix",
        training_hours=8.0,
        target_effective_seconds=28_800,
        task_end_hours=8.25,
        required_platform_wall_seconds=29_700,
        schedule_boundaries_seconds=(
            1_800.0,
            7_200.0,
            10_800.0,
            12_600.0,
            28_800.0,
        ),
        checkpoint_boundaries_seconds=(
            1_800.0,
            7_200.0,
            10_800.0,
            12_600.0,
            28_800.0,
        ),
        legacy=True,
    ),
)

PROFILE_REGISTRY: Mapping[str, TrainingProfileSpec] = MappingProxyType(
    {spec.name: spec for spec in _PROFILE_SPECS}
)
ACTIVE_TRAIN_PROFILES = frozenset(
    name for name, spec in PROFILE_REGISTRY.items() if spec.trainable
)
LEGACY_PROFILE_NAMES = frozenset(
    name for name, spec in PROFILE_REGISTRY.items() if spec.legacy
)
DEFAULT_TRAIN_PROFILE = PROFILE_MAZE_INSTANT_REPAIR2H
DEFAULT_EVAL_PROFILE = PROFILE_FULL_TRACK
LEGACY_COMPATIBILITY_DEFAULT = PROFILE_MAZE_CREDIT_REPAIR


def normalize_profile_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in PROFILE_MODES:
        raise ValueError(f"unsupported P4 profile mode {mode!r}")
    return normalized


def normalize_training_profile(training_profile: str) -> str:
    profile = str(training_profile).strip().lower()
    if profile not in PROFILE_REGISTRY:
        raise ValueError(f"unsupported P4 training profile {training_profile!r}")
    return profile


def get_training_profile(
    training_profile: str,
    *,
    mode: str | None = None,
) -> TrainingProfileSpec:
    profile = normalize_training_profile(training_profile)
    spec = PROFILE_REGISTRY[profile]
    if mode is not None:
        required_mode = normalize_profile_mode(mode)
        if required_mode not in spec.allowed_modes:
            raise ValueError(
                f"P4 profile {profile!r} is not allowed for {required_mode!r}; "
                f"allowed modes are {sorted(spec.allowed_modes)!r}"
            )
    return spec


def require_training_profile(training_profile: str) -> TrainingProfileSpec:
    return get_training_profile(training_profile, mode=TRAIN_MODE)


def is_maze_profile(training_profile: str) -> bool:
    return get_training_profile(training_profile).maze_only


def is_instant_profile(training_profile: str) -> bool:
    return get_training_profile(training_profile).instant_command
