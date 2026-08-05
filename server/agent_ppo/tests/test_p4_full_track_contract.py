#!/usr/bin/env python3
"""Goal and terrain contracts shared by P4's complete five-stage Track run."""

import math

import pytest
import torch

from agent_ppo.feature import nav_contract, p2_contract, p4_contract
from agent_ppo.feature.goal_features import encode_track_goal
from agent_ppo.feature.p2_worker_bridge import P2WorkerBridge
from agent_ppo.feature.p4_goal_belief import GoalBeliefChainV2


@pytest.mark.parametrize(
    ("distance_m", "bearing_rad", "expected_xy"),
    (
        (3.0, 0.0, (0.3, 0.0)),
        (10.0, math.pi / 4.0, (math.sqrt(0.5), math.sqrt(0.5))),
        (20.0, math.pi / 2.0, (0.0, 1.0)),
        (40.0, math.atan2(4.0, 3.0), (0.6, 0.8)),
    ),
)
def test_goal_encoding_preserves_bearing_and_actor_critic_agree(
    distance_m, bearing_rad, expected_xy
):
    local_xy = torch.tensor(
        ((distance_m * math.cos(bearing_rad), distance_m * math.sin(bearing_rad)),),
        dtype=torch.float32,
    )
    critic_goal = encode_track_goal(local_xy)
    belief = GoalBeliefChainV2(1, "cpu")
    belief.estimate.copy_(local_xy)
    belief.has_estimate.fill_(True)
    belief.age_s.zero_()
    actor_goal = belief.goal4()

    torch.testing.assert_close(actor_goal[:, :3], critic_goal)
    torch.testing.assert_close(
        actor_goal[0, :2], torch.tensor(expected_xy, dtype=torch.float32)
    )
    assert actor_goal[0, 2].item() == pytest.approx(min(distance_m / 20.0, 1.0))
    assert actor_goal.shape == (1, 4)
    assert critic_goal.shape == (1, 3)


def test_near_goal_xy_encoding_is_pointwise_legacy_compatible():
    local_xy = torch.tensor(((0.0, 0.0), (3.0, -4.0), (-6.0, 8.0)))
    encoded = nav_contract.encode_goal_xy_direction_preserving(local_xy)
    expected = torch.clamp(local_xy / 10.0, -1.0, 1.0)
    assert torch.equal(encoded, expected)


def test_goal_encoding_version_is_checkpoint_metadata_without_shape_change():
    belief = GoalBeliefChainV2(1, "cpu")
    state = belief.state_dict()
    checkpoint_contract = nav_contract.high_level_checkpoint_contract()

    assert state["goal_encoding_version"] == nav_contract.GOAL_ENCODING_VERSION
    assert checkpoint_contract["goal_encoding"]["version"] == nav_contract.GOAL_ENCODING_VERSION
    assert checkpoint_contract["goal_encoding"]["xy"] == (
        "normalize(local_xy)*min(distance_m/10,1)"
    )
    state["goal_encoding_version"] = "legacy_axis_clamp_v1"
    with pytest.raises(ValueError, match="goal-encoding version"):
        belief.load_state_dict(state)


def test_five_stage_track_mapping_and_legacy_three_stage_compatibility():
    full_chain = (
        "pyramid_slope",
        "pyramid_slope_inv",
        "pyramid_stairs",
        "pyramid_stairs_inv",
        "open_entry_maze",
    )
    assert p2_contract.canonical_track_segment_labels(full_chain) == (
        "slope",
        "slope_inv",
        "stairs",
        "stairs_inv",
        "maze",
    )
    assert p2_contract.track_segment_metric_indices(
        torch.arange(5, dtype=torch.float32),
        p2_contract.canonical_track_segment_labels(full_chain),
    ).tolist() == [0, 1, 2, 3, 4]

    legacy_chain = (
        "pyramid_slope_inv",
        "pyramid_stairs_inv",
        "open_entry_maze",
    )
    legacy_labels = p2_contract.canonical_track_segment_labels(legacy_chain)
    assert legacy_labels == p2_contract.TRACK_SEGMENT_METRIC_LABELS
    assert p2_contract.track_segment_metric_indices(
        torch.arange(3, dtype=torch.float32), legacy_labels
    ).tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("stage", "expected"),
    (("p4_nav_ppo", True), ("p4_track_eval", False), ("p2_nav_ppo", False)),
)
def test_worker_bridge_p4_training_wire_is_derived_from_runtime_stage(stage, expected):
    bridge = object.__new__(P2WorkerBridge)
    bridge.runtime_stage_type = stage
    assert bridge._p4_enabled is expected


def test_open_straight_large_alternating_yaw_cannot_escape_s_turn_penalty():
    penalty, diagnostics = p4_contract.open_straight_penalty(
        torch.tensor([[0.70, 0.0, 0.80]]),
        torch.tensor([[0.70, 0.0, 0.0]]),
        torch.tensor([[8.0, 0.0]]),
        torch.tensor([[0.85, 0.95, 0.85]]),
        torch.tensor([True]),
        torch.tensor([0]),
        torch.tensor([2.0]),
        torch.tensor([False]),
        junction=torch.tensor([False]),
        dead_end=torch.tensor([False]),
        contact_or_recovery=torch.tensor([False]),
        goal_freshness=torch.tensor([1.0]),
        yaw_cancellation_value=torch.tensor([0.90]),
        path_excess_m=torch.tensor([0.0]),
    )
    assert diagnostics["open_straight_eligible"].item() == 1.0
    assert diagnostics["open_straight_s_turn_penalty"].item() < 0.0
    assert penalty.item() < 0.0


def test_open_straight_never_penalizes_maze_or_segment_boundary():
    common = dict(
        policy_cmd3=torch.tensor([[0.70, 0.15, 0.70]]),
        true_velocity3=torch.tensor([[0.70, 0.15, 0.0]]),
        clean_goal_xy_m=torch.tensor([[8.0, 0.0]]),
        safe3=torch.tensor([[0.85, 0.95, 0.85]]),
        teacher_valid=torch.tensor([True]),
        terminal=torch.tensor([False]),
        junction=torch.tensor([False]),
        dead_end=torch.tensor([False]),
        contact_or_recovery=torch.tensor([False]),
        goal_freshness=torch.tensor([1.0]),
        yaw_cancellation_value=torch.tensor([0.90]),
        path_excess_m=torch.tensor([0.20]),
    )
    maze, _ = p4_contract.open_straight_penalty(
        current_segment=torch.tensor([4]),
        boundary_distance_m=torch.tensor([2.0]),
        **common,
    )
    boundary, _ = p4_contract.open_straight_penalty(
        current_segment=torch.tensor([0]),
        boundary_distance_m=torch.tensor([0.25]),
        **common,
    )
    assert maze.item() == 0.0
    assert boundary.item() == 0.0
