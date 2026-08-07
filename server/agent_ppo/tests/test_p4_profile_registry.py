"""Focused checks for the engineered P4 profile and contract registry."""

from dataclasses import FrozenInstanceError

import pytest

from agent_ppo.feature import p4_contract
from agent_ppo.p4 import contracts, profiles
from agent_ppo.p4.legacy_profiles import LEGACY_PROFILE_REGISTRY


def test_registry_exposes_only_two_active_train_profiles():
    assert profiles.ACTIVE_TRAIN_PROFILES == frozenset(
        {
            "maze_instant_repair2h",
            "full_track",
            "maze_stable_direction_smoke",
            "maze_stable_direction_8h",
        }
    )
    assert profiles.LEGACY_PROFILE_NAMES == frozenset(
        {
            "maze_credit_repair",
            "maze_closed_loop_v3",
            "maze_instant_command_r4",
        }
    )
    assert set(LEGACY_PROFILE_REGISTRY) == profiles.LEGACY_PROFILE_NAMES


@pytest.mark.parametrize("profile", sorted(profiles.LEGACY_PROFILE_NAMES))
def test_legacy_profiles_are_warm_start_and_eval_only(profile):
    spec = profiles.get_training_profile(profile)
    assert spec.legacy is True
    assert spec.allowed_modes == frozenset({"warm_start", "eval"})
    with pytest.raises(ValueError, match="not allowed for 'train'"):
        profiles.get_training_profile(profile, mode="train")
    assert profiles.get_training_profile(profile, mode="warm_start") is spec
    assert profiles.get_training_profile(profile, mode="eval") is spec


def test_profile_specs_and_registry_are_frozen():
    spec = profiles.PROFILE_REGISTRY["maze_instant_repair2h"]
    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"
    with pytest.raises(TypeError):
        profiles.PROFILE_REGISTRY["changed"] = spec


def test_active_profile_specs_own_runtime_clock_and_boundaries():
    repair = profiles.require_training_profile("maze_instant_repair2h")
    assert repair.run_name == "p4maze2h-instant-repair-r1"
    assert repair.training_hours == 2.0
    assert repair.target_effective_seconds == 7_200
    assert repair.task_end_hours == 2.25
    assert repair.required_platform_wall_seconds == 8_100
    assert repair.schedule_branch == "instant_repair2h"
    assert repair.wall_stuck_precedes_success is False
    assert repair.schedule_boundaries_seconds == (
        300.0,
        1_800.0,
        5_400.0,
        7_200.0,
    )
    assert repair.checkpoint_boundaries_seconds == (
        300.0,
        900.0,
        1_800.0,
        3_600.0,
        5_400.0,
        7_200.0,
    )

    full_track = profiles.require_training_profile("full_track")
    assert full_track.run_name == "p4full8h-r2"
    assert full_track.target_effective_seconds == 28_800
    assert full_track.task_end_hours == 8.25
    assert full_track.required_platform_wall_seconds == 29_700
    assert full_track.schedule_branch == "auto"
    assert full_track.wall_stuck_precedes_success is True
    assert full_track.schedule_boundaries_seconds[-1] == 28_800.0
    assert full_track.checkpoint_boundaries_seconds[-1] == 28_800.0

    smoke = profiles.require_training_profile("maze_stable_direction_smoke")
    assert smoke.run_name == "p4maze30m-stable-smoke"
    assert smoke.target_effective_seconds == 1_800
    assert smoke.task_end_hours == 0.75
    assert smoke.schedule_branch == "stable_direction_smoke"

    stable = profiles.require_training_profile("maze_stable_direction_8h")
    assert stable.run_name == "p4maze8h-stable-direction"
    assert stable.target_effective_seconds == 28_800
    assert stable.task_end_hours == 8.25
    assert stable.schedule_branch == "stable_direction_8h"


@pytest.mark.parametrize("profile", sorted(profiles.PROFILE_REGISTRY))
def test_training_contract_clock_is_sourced_from_profile_spec(profile):
    spec = profiles.get_training_profile(profile)
    training = contracts.training_contract(training_profile=profile)
    assert training["run_name"] == spec.run_name
    assert training["training_hours"] == spec.training_hours
    assert training["target_effective_seconds"] == spec.target_effective_seconds
    assert (
        training["required_platform_wall_seconds"]
        == spec.required_platform_wall_seconds
    )
    assert training["required_platform_wall_hours"] == (
        spec.required_platform_wall_hours
    )
    assert training["schedule_boundaries_seconds"] == list(
        spec.schedule_boundaries_seconds
    )


def test_profile_normalization_and_classification_are_centralized():
    assert profiles.normalize_training_profile(" FULL_TRACK ") == "full_track"
    assert profiles.is_maze_profile("maze_instant_repair2h") is True
    assert profiles.is_maze_profile("full_track") is False
    assert profiles.is_instant_profile("maze_instant_command_r4") is True
    with pytest.raises(ValueError, match="unsupported P4 training profile"):
        profiles.get_training_profile("unknown")


@pytest.mark.parametrize("profile", sorted(profiles.PROFILE_REGISTRY))
def test_legacy_facade_matches_canonical_contracts_and_digests(profile):
    assert p4_contract.command_contract(profile) == contracts.command_contract(profile)
    assert p4_contract.reward_contract(training_profile=profile) == (
        contracts.reward_contract(training_profile=profile)
    )
    assert p4_contract.training_contract(training_profile=profile) == (
        contracts.training_contract(training_profile=profile)
    )
    metadata = contracts.contract_metadata(training_profile=profile)
    assert metadata == p4_contract.contract_metadata(training_profile=profile)
    for name in ("command", "reward", "training"):
        assert metadata[f"{name}_digest"] == contracts.stable_digest(metadata[name])
        assert len(metadata[f"{name}_digest"]) == 64


def test_canonical_contract_access_can_enforce_profile_mode():
    contracts.training_contract("maze_instant_repair2h", mode="train")
    contracts.training_contract("full_track", mode="train")
    with pytest.raises(ValueError, match="not allowed for 'train'"):
        contracts.training_contract("maze_closed_loop_v3", mode="train")
    contracts.training_contract("maze_closed_loop_v3", mode="warm_start")
