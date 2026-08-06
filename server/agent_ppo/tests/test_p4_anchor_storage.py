"""Direct CPU regressions for P4 parent-anchor rollout transport."""

from __future__ import annotations

import torch

from agent_ppo.feature import p2_contract
from agent_ppo.feature.p2_rollout import P2RolloutStorage, SequenceRef


def _transition(storage: P2RolloutStorage) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    count = storage.num_envs
    hidden = (
        torch.zeros(2, count, 64),
        torch.zeros(2, count, 64),
    )
    return {
        "nav_feat": torch.zeros(count, p2_contract.NAV_FEATURE_DIM),
        "nav_nonvisual": torch.zeros(count, p2_contract.NAV_NONVISUAL_DIM),
        "response_profile": torch.zeros(count, p2_contract.RESPONSE_PROFILE_DIM),
        "confidence": torch.zeros(count, 1),
        "safety_target": torch.zeros(count, 3),
        "safety_valid": torch.zeros(count, 1),
        "critic_input": torch.zeros(count, p2_contract.CRITIC_INPUT_DIM),
        "pre_tanh_action": torch.zeros(count, 3),
        "old_log_prob": torch.zeros(count, 1),
        "old_value": torch.zeros(count, 1),
        "reward": torch.zeros(count, 1),
        "duration_frames": torch.ones(count, 1, dtype=torch.long),
        "bootstrap_value": torch.zeros(count, 1),
        "bootstrap_mask": torch.zeros(count, 1),
        "continuation_mask": torch.zeros(count, 1),
        "reset_mask": torch.zeros(count, dtype=torch.bool),
        "actor_hidden": hidden,
        "critic_hidden": hidden,
    }


def _stack_refs(storage: P2RolloutStorage, name: str, refs: list[SequenceRef]) -> torch.Tensor:
    """Match AlgorithmP2NavPPO._stack_refs' generic storage access."""
    source = getattr(storage, name)
    return torch.stack(
        [
            source[ref.start : ref.start + storage.sequence_length, ref.env]
            for ref in refs
        ],
        dim=1,
    )


def test_parent_anchor_storage_copies_cpu_values_and_stacks_tbptt_sequences():
    storage = P2RolloutStorage(
        2, num_ticks=2, sequence_length=2, store_depth=False, pin_memory=False
    )
    normalized_mean = torch.tensor([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]])
    log_std = torch.tensor([[-1.0, -0.9, -0.8], [-0.7, -0.6, -0.5]])
    mask = torch.tensor([[1.0], [0.0]])
    first = _transition(storage)
    first.update(
        parent_normalized_mean=normalized_mean,
        parent_log_std=log_std,
        parent_anchor_mask=mask,
    )
    storage.add(**first)
    normalized_mean.fill_(99.0)
    log_std.fill_(99.0)
    mask.fill_(99.0)

    second = _transition(storage)
    second.update(
        parent_normalized_mean=torch.full((2, 3), 2.0),
        parent_log_std=torch.full((2, 3), -2.0),
        parent_anchor_mask=torch.ones(2, 1),
    )
    storage.add(**second)

    assert storage.parent_normalized_mean.device.type == "cpu"
    assert storage.parent_normalized_mean.shape == (2, 2, 3)
    assert storage.parent_log_std.shape == (2, 2, 3)
    assert storage.parent_anchor_mask.shape == (2, 2, 1)
    assert torch.allclose(
        storage.parent_normalized_mean[0],
        torch.tensor([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]]),
    )
    assert torch.allclose(
        storage.parent_log_std[0],
        torch.tensor([[-1.0, -0.9, -0.8], [-0.7, -0.6, -0.5]]),
    )
    assert torch.equal(storage.parent_anchor_mask[0], torch.tensor([[1.0], [0.0]]))

    refs = [SequenceRef(start=0, env=1)]
    assert torch.allclose(
        _stack_refs(storage, "parent_normalized_mean", refs),
        torch.tensor([[[0.4, -0.5, 0.6]], [[2.0, 2.0, 2.0]]]),
    )
    assert torch.allclose(
        _stack_refs(storage, "parent_log_std", refs),
        torch.tensor([[[-0.7, -0.6, -0.5]], [[-2.0, -2.0, -2.0]]]),
    )
    assert torch.equal(
        _stack_refs(storage, "parent_anchor_mask", refs),
        torch.tensor([[[0.0]], [[1.0]]]),
    )


def test_parent_anchor_defaults_are_zero_after_reset_and_omission():
    storage = P2RolloutStorage(
        1, num_ticks=1, sequence_length=1, store_depth=False, pin_memory=False
    )
    populated = _transition(storage)
    populated.update(
        parent_normalized_mean=torch.ones(1, 3),
        parent_log_std=torch.ones(1, 3),
        parent_anchor_mask=torch.ones(1, 1),
    )
    storage.add(**populated)
    assert storage.parent_anchor_mask[0, 0, 0] == 1.0

    assert storage.reset() is storage
    storage.add(**_transition(storage))
    assert storage.step == 1
    assert torch.count_nonzero(storage.parent_normalized_mean) == 0
    assert torch.count_nonzero(storage.parent_log_std) == 0
    assert torch.count_nonzero(storage.parent_anchor_mask) == 0
