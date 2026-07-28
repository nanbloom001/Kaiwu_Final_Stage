#!/usr/bin/env python3

import torch

from agent_ppo.feature.response_aux_buffer import (
    ResponseAuxBuffer,
    build_response_observation,
    split_privileged_transport,
)
from agent_ppo.model.response_adapter import CommandResponseAdapter


def _aux(num_envs, step, *, epoch=1.0):
    value = torch.zeros(num_envs, 30)
    value[:, 0] = 0.4
    value[:, 3] = min(0.4, step * 0.006)
    value[:, 6] = 0.2
    value[:, 9] = 1.0
    value[:, 12] = 0.2
    value[:, 15] = step * 0.004
    value[:, 21:24] = torch.tensor([0.0, 0.0, -1.0])
    value[:, 26] = epoch
    return value


def test_transport_splits_before_ppo_storage():
    wire = torch.randn(8, 346)
    critic, aux = split_privileged_transport(wire)
    assert critic.shape == (8, 316)
    assert aux.shape == (8, 30)
    assert torch.equal(critic, wire[:, :316])
    assert torch.equal(aux, wire[:, 316:])


def test_future_labels_mask_target_change_and_done():
    buffer = ResponseAuxBuffer(
        4, "cpu", capacity_steps=128, sequence_length=1, burn_in_steps=0
    )
    for step in range(51):
        epoch = 2.0 if step >= 30 else 1.0
        dones = torch.zeros(4, dtype=torch.bool)
        if step == 20:
            dones[1] = True
        buffer.append(_aux(4, step, epoch=epoch), dones)
    batch = buffer.sample(batch_envs=4, generator=torch.Generator().manual_seed(1))
    assert batch is not None
    # Target changes at 0.6s, so only the 0.2s horizon remains valid.
    assert torch.all(batch.horizon_mask[..., 1:] == 0)
    assert int(batch.horizon_mask[..., 0].sum()) <= 4
    assert torch.all(batch.pose_mask == 0)
    assert batch.metadata["low_level_digest"] == ("unknown",)
    assert batch.metadata["low_level_iteration"].shape == (1,)


def test_adapter_shape_log_sigma_and_backward():
    observation = build_response_observation(
        _aux(6, 10), torch.ones(15)
    ).unsqueeze(0).repeat(4, 1, 1)
    adapter = CommandResponseAdapter()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=3.0e-4)
    profile, _ = adapter(observation)
    parts = adapter.split_profile(profile)
    assert profile.shape == (4, 6, 16)
    assert parts["velocity"].shape == (4, 6, 3, 3)
    assert float(parts["velocity_log_sigma"].detach().min()) >= -4.0
    assert float(parts["velocity_log_sigma"].detach().max()) <= 1.0
    loss = profile.square().mean()
    optimizer.zero_grad()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in adapter.parameters()
    )


def test_adapter_reset_mask_matches_fresh_hidden_replay():
    torch.manual_seed(11)
    adapter = CommandResponseAdapter().eval()
    observation = torch.randn(6, 2, 32)
    reset_mask = torch.zeros(6, 2, dtype=torch.bool)
    reset_mask[3, 0] = True
    full, _ = adapter(observation, reset_mask=reset_mask)
    replay, _ = adapter(observation[3:, 0:1])
    assert torch.allclose(full[3:, 0:1], replay, atol=1.0e-6, rtol=1.0e-5)


def test_buffer_burn_in_and_training_window_shapes():
    buffer = ResponseAuxBuffer(
        4, "cpu", capacity_steps=128, sequence_length=4, burn_in_steps=2
    )
    buffer.set_low_level_version("digest-a", 1)
    for step in range(56):
        buffer.append(_aux(4, step), torch.zeros(4, dtype=torch.bool))
    batch = buffer.sample(batch_envs=3, generator=torch.Generator().manual_seed(2))
    assert batch is not None
    assert batch.burn_in_observations.shape == (2, 3, 32)
    assert batch.burn_in_reset_mask.shape == (2, 3)
    assert batch.observations.shape == (4, 3, 32)
    assert batch.reset_mask.shape == (4, 3)
    assert set(batch.metadata["low_level_digest"]) == {"digest-a"}


def test_low_level_version_change_clears_unfinished_labels_and_sequences_do_not_mix():
    buffer = ResponseAuxBuffer(
        2, "cpu", capacity_steps=128, sequence_length=2, burn_in_steps=0
    )
    buffer.set_low_level_version("digest-a", 1)
    for step in range(51):
        buffer.append(_aux(2, step), torch.zeros(2, dtype=torch.bool))
    assert not buffer.ready

    buffer.set_low_level_version("digest-b", 2)
    for step in range(51):
        buffer.append(_aux(2, 100 + step), torch.zeros(2, dtype=torch.bool))
    assert not buffer.ready
    buffer.append(_aux(2, 151), torch.zeros(2, dtype=torch.bool))
    batch = buffer.sample(batch_envs=2, generator=torch.Generator().manual_seed(3))
    assert batch is not None
    assert set(batch.metadata["low_level_digest"]) == {"digest-b"}
    assert torch.all(batch.metadata["low_level_iteration"] == 2)


def test_eighty_frame_rollout_forms_records_before_version_boundary():
    buffer = ResponseAuxBuffer(
        2, "cpu", capacity_steps=128, sequence_length=16, burn_in_steps=8
    )
    buffer.set_low_level_version("digest-a", 1)
    for step in range(80):
        buffer.append(_aux(2, step), torch.zeros(2, dtype=torch.bool))

    state = buffer.state_dict()
    assert state["append_calls"] == 80
    assert state["max_history_length"] == 51
    assert state["record_steps"] == 30
    assert state["version_reset_count"] == 0
    assert buffer.ready

    buffer.set_low_level_version("digest-b", 2)
    assert buffer.state_dict()["version_reset_count"] == 1
    assert buffer.state_dict()["current_history_length"] == 0


def test_resume_with_different_env_count_discards_only_warm_history():
    source = ResponseAuxBuffer(
        4, "cpu", capacity_steps=128, sequence_length=1, burn_in_steps=0
    )
    for step in range(51):
        source.append(_aux(4, step), torch.zeros(4, dtype=torch.bool))
    state = source.checkpoint_state()

    restored = ResponseAuxBuffer(
        8, "cpu", capacity_steps=128, sequence_length=1, burn_in_steps=0
    )
    assert restored.load_checkpoint_state(state) == "buffer_reset_shape_mismatch"
    assert not restored.ready
    assert restored.total_sequences == source.total_sequences
    assert restored.low_level_digest == source.low_level_digest
    restored.append(_aux(8, 0), torch.zeros(8, dtype=torch.bool))


def test_resume_discards_unfinished_future_history_but_restores_completed_records():
    source = ResponseAuxBuffer(
        2, "cpu", capacity_steps=128, sequence_length=1, burn_in_steps=0
    )
    source.set_low_level_version("digest-a", 4)
    for step in range(55):
        source.append(_aux(2, step), torch.zeros(2, dtype=torch.bool))
    state = source.checkpoint_state()

    restored = ResponseAuxBuffer(
        2, "cpu", capacity_steps=128, sequence_length=1, burn_in_steps=0
    )
    assert (
        restored.load_checkpoint_state(state)
        == "completed_records_restored_history_reset"
    )
    assert restored.ready
    assert len(restored._history_aux) == 0
    restored.set_low_level_version("digest-a", 4)
    for step in range(50):
        restored.append(_aux(2, 100 + step), torch.zeros(2, dtype=torch.bool))
    assert len(restored._history_aux) == 50
