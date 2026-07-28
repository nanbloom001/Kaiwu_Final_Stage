#!/usr/bin/env python3

from types import SimpleNamespace

import torch

from agent_ppo.feature.feedback_emulator import (
    FeedbackEmulator,
    feedback_implementation_digest,
)
from agent_ppo.feature import p15_contract
from agent_ppo.feature.p15_worker_bridge import P15WorkerBridge


def _feedback_rollout(seed: int):
    emulator = FeedbackEmulator(8, "cpu", seed=seed)
    velocity = torch.tensor([[0.4, 0.0, 0.2]]).repeat(8, 1)
    angular = torch.tensor([[0.0, 0.0, 0.2]]).repeat(8, 1)
    gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(8, 1)
    outputs = []
    for step in range(20):
        sample = emulator.step(
            velocity,
            angular,
            gravity,
            dt_s=0.02,
            reset_mask=torch.ones(8, dtype=torch.bool) if step == 0 else None,
        )
        outputs.append(
            (
                sample.measured_velocity.clone(),
                sample.velocity_valid.clone(),
                sample.velocity_age.clone(),
                sample.feedback_source.clone(),
            )
        )
    return outputs


def test_feedback_emulator_is_seed_deterministic_and_freshness_is_bounded():
    first = _feedback_rollout(9)
    second = _feedback_rollout(9)
    for lhs, rhs in zip(first, second):
        for left, right in zip(lhs, rhs):
            assert torch.equal(left, right)
    for _, valid, age, _ in first:
        assert torch.all((age >= 0.0) & (age <= 1.0))
        assert torch.all(age[valid <= 0.5] == 1.0)
    for measured, _, _, source in first:
        assert torch.all((source == 0.0) | (source == 1.0))
        assert not torch.any(source == 2.0)
        assert torch.isfinite(measured[:, 2]).all()


def test_uwb_profile_changes_do_not_change_short_horizon_feedback():
    first_profile = dict(p15_contract.FEEDBACK_PROFILE)
    first_profile["uwb"] = dict(first_profile["uwb"])
    second_profile = dict(first_profile)
    second_profile["uwb"] = {
        **first_profile["uwb"],
        "rate_hz": [0.1, 0.2],
        "delay_s": [3.0, 5.0],
        "noise_std": [9.0, 9.0, 9.0],
    }
    velocity = torch.tensor([[0.4, 0.1, 0.2]]).repeat(4, 1)
    angular = torch.tensor([[0.0, 0.0, 0.2]]).repeat(4, 1)
    gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(4, 1)
    first = FeedbackEmulator(4, "cpu", seed=77, profile=first_profile)
    second = FeedbackEmulator(4, "cpu", seed=77, profile=second_profile)
    for step in range(20):
        reset = torch.ones(4, dtype=torch.bool) if step == 0 else None
        lhs = first.step(velocity, angular, gravity, dt_s=0.02, reset_mask=reset)
        rhs = second.step(velocity, angular, gravity, dt_s=0.02, reset_mask=reset)
        assert torch.equal(lhs.measured_velocity, rhs.measured_velocity)
        assert torch.equal(lhs.feedback_source, rhs.feedback_source)


def test_feedback_implementation_digest_is_stable_sha256():
    first = feedback_implementation_digest()
    assert first == feedback_implementation_digest()
    assert len(first) == 64
    int(first, 16)


def test_runtime_friction_boundary_applies_updated_event_term():
    calls = []

    def apply_material(_env, env_ids, **params):
        calls.append((env_ids, dict(params)))

    cfg = SimpleNamespace(
        params={
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
        },
        func=apply_material,
    )

    class Manager:
        def get_term_cfg(self, name):
            assert name == "physics_material"
            return cfg

        def set_term_cfg(self, name, value):
            assert name == "physics_material"
            assert value is cfg

    bridge = object.__new__(P15WorkerBridge)
    bridge.env = SimpleNamespace(event_manager=Manager())
    bridge._friction_enabled = False
    bridge._friction_last_attempt_step = -500
    bridge._friction_activation_attempts = 0
    bridge._friction_activation_failures = 0
    bridge._try_enable_friction(1.99, 1000)
    assert not calls
    bridge._try_enable_friction(2.0, 1000)
    assert bridge._friction_enabled
    assert calls[0][1]["static_friction_range"] == (0.6, 1.2)
    assert calls[0][1]["dynamic_friction_range"] == (0.6, 1.2)
    assert bridge._friction_activation_attempts == 1
    assert bridge._friction_activation_failures == 0


def test_gait_probe_uses_air_and_swing_sample_denominators():
    air = torch.tensor(
        [[0.2, 0.0, 0.1, 0.0], [0.4, 0.0, 0.3, 0.0]], dtype=torch.float32
    )
    sensor = SimpleNamespace(data=SimpleNamespace(current_air_time=air))
    robot_data = SimpleNamespace(
        body_lin_vel_w=torch.zeros(2, 4, 3),
        body_pos_w=torch.full((2, 4, 3), 0.72),
        root_pos_w=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    bridge = object.__new__(P15WorkerBridge)
    bridge.num_envs = 2
    bridge.device = torch.device("cpu")
    bridge._gait_probe = {
        "body_ids": torch.arange(4),
        "sensor": sensor,
        "frames": 0,
        "contact_sum": torch.zeros(4),
        "contact_events": torch.zeros(4),
        "air_sum": torch.zeros(4),
        "air_samples": torch.zeros(4),
        "air_max": torch.zeros(4),
        "slip_distance": torch.zeros(4),
        "swing_height_sum": torch.zeros(4),
        "swing_samples": torch.zeros(4),
        "previous_contact": torch.ones(2, 4, dtype=torch.bool),
    }
    bridge._resolve_gait_probe = lambda: bridge._gait_probe
    bridge._robot = lambda: SimpleNamespace(data=robot_data)
    metrics = bridge._update_gait_probe()
    assert abs(metrics["fl_mean_air_time"] - 0.3) < 1.0e-6
    assert abs(metrics["rl_mean_air_time"] - 0.2) < 1.0e-6
    assert abs(metrics["fl_swing_height"] - 0.1) < 1.0e-6
    assert metrics["fr_mean_air_time"] == 0.0
