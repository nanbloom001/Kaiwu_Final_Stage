import copy

import pytest
import torch

from agent_ppo.checkpoint_io import P3_STANDARD_JOINT_PHASE_LABELS
from agent_ppo.feature import p3_contract
from agent_ppo.feature.p3_depth_memory import (
    P35CameraFeatureTiming,
    P3DepthFaultAugmenter,
    P3MemoryAuxiliary,
)
from agent_ppo.model.visual_actor_critic import VisualActorCritic


def test_near_clip_beta_distribution_and_rng_are_bounded():
    first = P3DepthFaultAugmenter(
        num_steps=128,
        num_envs=4096,
        depth_shape=(180, 320, 1),
        device="cpu",
        seed=7,
    )
    second = P3DepthFaultAugmenter(
        num_steps=128,
        num_envs=4096,
        depth_shape=(180, 320, 1),
        device="cpu",
        seed=7,
    )
    assert torch.equal(first.near_clip_m, second.near_clip_m)
    assert float(first.near_clip_m.min()) >= 0.10
    assert float(first.near_clip_m.max()) <= 0.25
    assert float(first.near_clip_m.mean()) == pytest.approx(0.13, abs=0.004)


def test_near_clip_environment_initialization_is_idempotent():
    augmenter = P3DepthFaultAugmenter(
        num_steps=128,
        num_envs=32,
        depth_shape=(8, 8, 1),
        device="cpu",
        seed=9,
    )
    initial = augmenter.near_clip_m.clone()
    rng = augmenter.generator.get_state().clone()
    augmenter.ensure_environment_initialized()
    assert torch.equal(augmenter.near_clip_m, initial)
    assert torch.equal(augmenter.generator.get_state(), rng)


def test_p35_camera_timing_holds_30hz_features_at_50hz_and_bounds_delay():
    timing = P35CameraFeatureTiming(
        num_envs=8, feature_dim=32, device="cpu", capacity=10, seed=11
    )
    timing.begin_rollout(5000.0)
    captures = []
    timing_changed = []
    for step in range(20):
        output, capture = timing.step(torch.full((8, 32), float(step)))
        assert output.shape == (8, 32)
        captures.append(capture)
        timing_changed.append(timing.last_timing_changed.clone())
    metrics = timing.diagnostics()
    assert torch.stack(captures).float().mean() == pytest.approx(0.60, abs=0.10)
    assert 0.0 < metrics["camera_hold_ratio"] < 1.0
    assert metrics["camera_active_delay_p95_ms"] <= 150.0 + 1.0e-4
    assert metrics["camera_shadow_delay_p95_ms"] <= 250.0 + 1.0e-4
    assert bool(torch.stack(timing_changed).any())
    timing.reset(torch.tensor([True, False, False, False, False, False, False, False]))
    assert bool((timing.timestamps[:, 0] < -1.0e8).all())


def test_p35_camera_timing_rng_round_trip():
    first = P35CameraFeatureTiming(
        num_envs=4, feature_dim=32, device="cpu", seed=19
    )
    state = first.state_dict()
    first.begin_rollout(5000.0)
    expected = first.delay_s.clone()
    second = P35CameraFeatureTiming(
        num_envs=4, feature_dim=32, device="cpu", seed=999
    )
    second.load_state_dict(state)
    second.begin_rollout(5000.0)
    assert torch.equal(expected, second.delay_s)


def test_severe_fault_is_persistent_and_stays_in_requested_hole_range():
    augmenter = P3DepthFaultAugmenter(
        num_steps=128,
        num_envs=8,
        depth_shape=(180, 320, 1),
        device="cpu",
        seed=11,
    )
    augmenter.begin_rollout(1.0)
    augmenter.enabled.fill_(True)
    augmenter.mode.fill_(3)
    augmenter.prefix.zero_()
    augmenter.duration.fill_(75)
    augmenter.severe_ratio.fill_(0.65)
    depth = torch.ones(8, 180, 320, 1)
    first, changed_first = augmenter.apply(depth, 0)
    second, changed_second = augmenter.apply(depth, 50)
    inactive, changed_inactive = augmenter.apply(depth, 76)
    for value in (first, second):
        holes = (value <= 0.0).float().mean(dim=(1, 2, 3))
        assert bool(((holes >= 0.60) & (holes <= 0.85)).all())
    assert torch.equal(first, second)
    assert bool(changed_first.all() and changed_second.all())
    assert not bool(changed_inactive.any())
    assert bool((inactive > 0.0).all())


@pytest.mark.parametrize("mode", (1, 3, 4))
def test_random_fault_masks_are_reused_during_the_active_interval(mode):
    augmenter = P3DepthFaultAugmenter(
        num_steps=128,
        num_envs=4,
        depth_shape=(180, 320, 1),
        device="cpu",
        seed=15 + mode,
    )
    augmenter.begin_rollout(1.0)
    augmenter.enabled.fill_(True)
    augmenter.mode.fill_(mode)
    augmenter.prefix.zero_()
    augmenter.duration.fill_(75)
    clean = torch.ones(4, 180, 320, 1)
    first, _ = augmenter.apply(clean, 0)
    second, _ = augmenter.apply(clean, 50)
    assert torch.equal(first, second)


def test_severe_fault_does_not_stack_on_top_of_existing_eighty_percent_holes():
    augmenter = P3DepthFaultAugmenter(
        num_steps=128,
        num_envs=4,
        depth_shape=(180, 320, 1),
        device="cpu",
        seed=19,
    )
    augmenter.begin_rollout(1.0)
    augmenter.enabled.fill_(True)
    augmenter.mode.fill_(3)
    augmenter.prefix.zero_()
    augmenter.duration.fill_(75)
    augmenter.severe_ratio.fill_(0.65)
    depth = torch.zeros(4, 180, 320, 1)
    depth.reshape(4, -1)[:, ::5] = 1.0
    augmented, changed = augmenter.apply(depth, 0)
    holes = (augmented <= 0.0).float().mean(dim=(1, 2, 3))
    assert not bool(changed.any())
    assert bool((holes >= 0.80).all())
    assert bool((holes <= 0.85).all())

    concentrated = torch.ones(4, 180, 320, 1)
    concentrated.reshape(4, -1)[:, : int(0.80 * 180 * 320)] = 0.0
    augmented, _ = augmenter.apply(concentrated, 1)
    concentrated_holes = (augmented <= 0.0).float().mean(dim=(1, 2, 3))
    assert bool((concentrated_holes >= 0.80).all())
    assert bool((concentrated_holes <= 0.85).all())


def test_fault_strength_controls_enabled_sequences_without_near_clip_leakage():
    augmenter = P3DepthFaultAugmenter(
        num_steps=128,
        num_envs=4096,
        depth_shape=(8, 8, 1),
        device="cpu",
        seed=17,
    )
    augmenter.begin_rollout(0.25)
    assert float(augmenter.enabled.float().mean()) == pytest.approx(0.25, abs=0.025)
    depth = torch.full((4096, 8, 8, 1), 0.02)
    augmented, changed = augmenter.apply(depth, 0)
    assert torch.equal(changed, augmenter.enabled)
    assert bool((augmented[~augmenter.enabled] == depth[~augmenter.enabled]).all())


def test_fault_diagnostics_report_severe_and_blackout_events():
    augmenter = P3DepthFaultAugmenter(
        num_steps=128,
        num_envs=4,
        depth_shape=(8, 8, 1),
        device="cpu",
        seed=21,
    )
    augmenter.begin_rollout(1.0)
    augmenter.enabled.fill_(True)
    augmenter.mode.copy_(torch.tensor((3, 3, 4, 0)))
    augmenter.duration.copy_(torch.tensor((20, 40, 10, 30)))
    metrics = augmenter.diagnostics()
    assert metrics["depth_fault_planned_duration_s"] == pytest.approx(0.5)
    assert "depth_fault_duration_s" not in metrics
    assert metrics["depth_fault_severe_event_count"] == 2.0
    assert metrics["depth_fault_severe_duration_s"] == pytest.approx(0.6)
    assert metrics["depth_fault_blackout_event_count"] == 1.0
    assert metrics["depth_fault_blackout_duration_s"] == pytest.approx(0.2)
    assert metrics["depth_fault_recovery_telemetry_available"] == 0.0


def test_depth_fault_restore_does_not_consume_rng_before_environment_rebuild():
    source = P3DepthFaultAugmenter(
        num_steps=128,
        num_envs=16,
        depth_shape=(8, 8, 1),
        device="cpu",
        seed=23,
    )
    state = source.state_dict()
    restored = P3DepthFaultAugmenter(
        num_steps=128,
        num_envs=16,
        depth_shape=(8, 8, 1),
        device="cpu",
        seed=29,
    )
    restored.load_state_dict(state)
    assert torch.equal(restored.generator.get_state(), state["generator_state"])

    expected = P3DepthFaultAugmenter(
        num_steps=128,
        num_envs=16,
        depth_shape=(8, 8, 1),
        device="cpu",
        seed=31,
    )
    expected.generator.set_state(state["generator_state"])
    expected.reset_environment()
    restored.reset_environment()
    assert torch.equal(restored.near_clip_m, expected.near_clip_m)
    assert torch.equal(restored.generator.get_state(), expected.generator.get_state())


def test_memory_auxiliary_only_updates_recurrent_and_final_action_head():
    model = VisualActorCritic(
        num_proprio=45,
        num_scan=256,
        depth_shape=(180, 320, 1),
        latent_dim=32,
        cnn_output_dim=32,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        num_critic_obs=323,
        num_actions=12,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=0.15,
    )
    anchor_encoder = copy.deepcopy(model.vision_encoder).eval()
    anchor_actor = copy.deepcopy(model.actor).eval()
    auxiliary = P3MemoryAuxiliary(
        num_steps=4, num_envs=2, obs_dim=77, device="cpu", seed=13
    )
    auxiliary.begin_rollout(1.0)
    clean = torch.randn(4, 2, 77)
    fault = clean.clone()
    fault[:, :, 45:] = 0.0
    for step in range(4):
        auxiliary.store(step, clean[step], torch.ones(2, dtype=torch.bool))
    loss, metrics = auxiliary.loss(
        model,
        anchor_encoder,
        anchor_actor,
        fault,
        torch.zeros(4, 2, 1, dtype=torch.bool),
    )
    loss.backward()
    assert float(loss.detach()) > 0.0
    assert torch.isfinite(torch.tensor(metrics["memory_hidden_advantage"]))
    assert any(parameter.grad is not None for parameter in model.vision_encoder.rnn.parameters())
    assert model.actor[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.vision_encoder.cnn.parameters())
    assert all(parameter.grad is None for parameter in model.critic.parameters())
    assert all(parameter.grad is None for parameter in model.actor[:-1].parameters())


def test_memory_auxiliary_includes_delay_only_frames():
    auxiliary = P3MemoryAuxiliary(
        num_steps=2, num_envs=3, obs_dim=77, device="cpu", seed=17
    )
    auxiliary.begin_rollout(1.0)
    clean = torch.zeros(3, 77)
    auxiliary.store(
        0,
        clean,
        torch.tensor((False, True, True)),
        torch.tensor((True, False, True)),
    )
    assert auxiliary.selected[0].tolist() == [True, True, True]
    metrics = auxiliary._selection_metrics()
    assert metrics["memory_timing_only_frame_share"] == pytest.approx(1.0 / 6.0)
    assert metrics["memory_fault_only_frame_share"] == pytest.approx(1.0 / 6.0)
    assert metrics["memory_fault_timing_overlap_share"] == pytest.approx(1.0 / 6.0)


def test_stair_memory_schedule_and_checkpoint_labels_are_explicit():
    assert p3_contract.depth_fault_strength(0.0) == 0.0
    assert p3_contract.depth_fault_strength(2700.0) == pytest.approx(0.5)
    assert p3_contract.depth_fault_strength(7200.0) == 0.5
    assert p3_contract.depth_fault_strength(21600.0) == 0.5
    assert p3_contract.mirror_training_fraction(2699.0) == 0.0
    assert p3_contract.mirror_training_fraction(2700.0) == 1.0
    assert P3_STANDARD_JOINT_PHASE_LABELS[-7:] == (
        "calib",
        "gaitwarm",
        "gaitfull",
        "camfull",
        "pushwarm",
        "pushfull",
        "stable",
    )
