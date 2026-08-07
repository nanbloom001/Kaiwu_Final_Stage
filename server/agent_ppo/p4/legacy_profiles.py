"""Explicit compatibility-only view of retired P4 training profiles."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from agent_ppo.p4.profiles import (
    LEGACY_PROFILE_NAMES,
    PROFILE_REGISTRY,
    TrainingProfileSpec,
    get_training_profile,
)


LEGACY_PROFILE_REGISTRY: Mapping[str, TrainingProfileSpec] = MappingProxyType(
    {name: PROFILE_REGISTRY[name] for name in sorted(LEGACY_PROFILE_NAMES)}
)


def get_legacy_profile(training_profile: str) -> TrainingProfileSpec:
    spec = get_training_profile(training_profile)
    if not spec.legacy:
        raise ValueError(f"P4 profile {spec.name!r} is not a legacy profile")
    return spec
