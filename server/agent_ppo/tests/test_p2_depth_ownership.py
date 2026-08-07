#!/usr/bin/env python3
"""Ownership checks for one-copy P2 depth rollout capture."""

import torch

import agent_ppo.tests._nav_test_stubs  # noqa: F401
from agent_ppo.feature import p2_contract
from agent_ppo.feature.p2_rollout import P2RolloutStorage


def _transition(num_envs: int, depth: torch.Tensor) -> dict[str, torch.Tensor]:
    hidden = (
        torch.zeros(2, num_envs, 64),
        torch.zeros(2, num_envs, 64),
    )
    return {
        "depth": depth,
        "nav_nonvisual": torch.zeros(num_envs, p2_contract.NAV_NONVISUAL_DIM),
        "response_profile": torch.zeros(num_envs, p2_contract.RESPONSE_PROFILE_DIM),
        "confidence": torch.ones(num_envs, 1),
        "safety_target": torch.zeros(num_envs, 3),
        "safety_valid": torch.ones(num_envs, 1),
        "critic_input": torch.zeros(num_envs, p2_contract.CRITIC_INPUT_DIM),
        "pre_tanh_action": torch.zeros(num_envs, p2_contract.ACTION_DIM),
        "old_log_prob": torch.zeros(num_envs, 1),
        "old_value": torch.zeros(num_envs, 1),
        "reward": torch.zeros(num_envs, 1),
        "duration_frames": torch.ones(num_envs, 1, dtype=torch.long),
        "bootstrap_value": torch.zeros(num_envs, 1),
        "bootstrap_mask": torch.ones(num_envs, 1),
        "continuation_mask": torch.ones(num_envs, 1),
        "reset_mask": torch.zeros(num_envs, dtype=torch.bool),
        "actor_hidden": hidden,
        "critic_hidden": hidden,
    }


def test_pending_depth_uses_current_rollout_slot_and_survives_buffer_reuse():
    storage = P2RolloutStorage(
        1, num_ticks=2, sequence_length=1, store_depth=True, pin_memory=False
    )
    observation_buffer = torch.full(
        (1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1), 0.25
    )

    pending_depth = storage.own_current_depth_slot(observation_buffer)
    assert pending_depth.data_ptr() == storage.depth[0].data_ptr()
    assert pending_depth.dtype == torch.float16
    observation_buffer.fill_(0.75)
    assert torch.all(pending_depth == 0.25)

    storage.add(**_transition(1, pending_depth))
    assert storage.step == 1
    assert torch.all(storage.depth[0] == 0.25)

    next_pending_depth = storage.own_current_depth_slot(observation_buffer)
    assert next_pending_depth.data_ptr() == storage.depth[1].data_ptr()
    assert next_pending_depth.data_ptr() != pending_depth.data_ptr()
    assert torch.all(next_pending_depth == 0.75)


def test_prepared_depth_slot_is_not_copied_again_by_add():
    storage = P2RolloutStorage(
        1, num_ticks=1, sequence_length=1, store_depth=True, pin_memory=False
    )
    pending_depth = storage.own_current_depth_slot(
        torch.full((1, p2_contract.DEPTH_DIM), 0.25)
    )
    copied_depth_slots: list[int] = []
    original_copy = storage._copy

    def record_copy(target: torch.Tensor, value: torch.Tensor) -> None:
        if target.data_ptr() == storage.depth[storage.step].data_ptr():
            copied_depth_slots.append(target.data_ptr())
        original_copy(target, value)

    storage._copy = record_copy
    storage.add(**_transition(1, pending_depth))

    assert copied_depth_slots == []


def test_depth_copy_writes_directly_without_intermediate_tensor_to(monkeypatch):
    storage = P2RolloutStorage(
        1, num_ticks=1, sequence_length=1, store_depth=True, pin_memory=False
    )
    source = torch.full((1, p2_contract.DEPTH_DIM), 0.375)
    target = storage.depth[0]

    def forbidden_to(*_args, **_kwargs):
        raise AssertionError("depth ownership copy must not allocate through Tensor.to")

    monkeypatch.setattr(torch.Tensor, "to", forbidden_to)
    storage._copy(target, source)
    assert torch.all(target == torch.tensor(0.375, dtype=target.dtype))


def test_frozen_nav_feature_rollout_does_not_expose_depth_slots():
    storage = P2RolloutStorage(
        1, num_ticks=1, sequence_length=1, store_depth=False, pin_memory=False
    )
    try:
        storage.own_current_depth_slot(torch.zeros(1, p2_contract.DEPTH_DIM))
    except RuntimeError as error:
        assert "depth slots are unavailable" in str(error)
    else:
        raise AssertionError("frozen nav-feature storage unexpectedly exposed depth slots")

    nav_feature = torch.full((1, p2_contract.NAV_FEATURE_DIM), 0.5)
    transition = _transition(1, torch.zeros(1, p2_contract.DEPTH_DIM))
    transition["nav_feat"] = nav_feature
    storage.add(**transition)
    assert storage.step == 1
    assert torch.equal(storage.nav_feat[0], nav_feature)
