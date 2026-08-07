"""Regression checks for the stable-direction smoke and eight-hour profiles."""

from __future__ import annotations

import pytest
import torch

from agent_ppo.feature import p4_contract
from agent_ppo.p4 import contracts, profiles, rewards


@pytest.mark.parametrize(
    ("profile", "target", "phase0"),
    [
        ("maze_stable_direction_smoke", 1_800, "stable_smoke_calibrate"),
        ("maze_stable_direction_8h", 28_800, "stable_warm"),
    ],
)
def test_stable_profile_contract_and_clock(profile, target, phase0):
    spec = profiles.require_training_profile(profile)
    assert spec.target_effective_seconds == target
    assert contracts.training_contract(profile, mode="train")["run_name"] == spec.run_name
    assert contracts.command_contract(profile)["command_transition_mode"] == (
        "instant_hold_10hz"
    )
    reward = contracts.reward_contract(training_profile=profile, mode="train")
    assert reward["new_terms"]["missed_safe_direction"]["status"] == (
        "active_negative_ppo_reward"
    )
    assert p4_contract.training_schedule(0.0, branch=spec.schedule_branch)["phase"] == phase0


def test_stable_direction_schedule_has_exact_boundaries_and_cap():
    branch = "stable_direction_8h"
    warm = p4_contract.training_schedule(0.0, branch=branch)
    early = p4_contract.training_schedule(600.0, branch=branch)
    mid = p4_contract.training_schedule(3_600.0, branch=branch)
    late = p4_contract.training_schedule(10_800.0, branch=branch)
    final = p4_contract.training_schedule(21_600.0, branch=branch)
    assert [row["phase"] for row in (warm, early, mid, late, final)] == [
        "stable_warm",
        "stable_early",
        "stable_mid",
        "stable_late",
        "stable_stabilize",
    ]
    assert [row["actor_lr"] for row in (warm, early, mid, late, final)] == [
        0.0,
        1.0e-5,
        7.5e-6,
        5.0e-6,
        2.5e-6,
    ]
    assert warm["safety_group_floor"] == pytest.approx(-0.01)
    assert final["safety_group_floor"] == pytest.approx(-0.02)
    assert final["teacher_gradient_target_ratio"] == pytest.approx(0.01)


def test_stable_direction_reward_spec_activates_only_directional_terms():
    stable = rewards.REWARD_SPECS[profiles.PROFILE_MAZE_STABLE_DIRECTION_8H]
    old = rewards.REWARD_SPECS[profiles.PROFILE_MAZE_INSTANT_REPAIR2H]
    for name in ("missed_safe_direction", "goal_safe_preference", "yaw_exit_response"):
        assert name in stable.enabled_components
        assert name in old.shadow_components
    assert stable.safety_cap_components == frozenset(
        {
            "predictive_collision_risk",
            "missed_safe_direction",
            "goal_safe_preference",
            "yaw_exit_response",
        }
    )


def test_proportional_cap_is_dynamic_and_preserves_reward_conservation():
    values = [torch.tensor([-0.02]), torch.tensor([-0.01]), torch.tensor([-0.01])]
    active, scale = p4_contract.proportional_negative_cap(
        *values[:3], floor=-0.02
    )
    assert scale.item() == pytest.approx(0.5)
    assert sum(value.item() for value in active) == pytest.approx(-0.02)
    invalid = torch.tensor([0.0])
    _, no_cap = p4_contract.proportional_negative_cap(
        invalid, invalid, invalid, floor=-0.01
    )
    assert no_cap.item() == pytest.approx(1.0)
