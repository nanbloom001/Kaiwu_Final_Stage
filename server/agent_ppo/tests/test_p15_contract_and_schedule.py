#!/usr/bin/env python3

import torch

from agent_ppo.feature import p15_contract
from agent_ppo.feature.p15_command_schedule import P15CommandSchedule


def test_p15_contract_dimensions_and_phases():
    assert p15_contract.CRITIC_OBS_DIM == 316
    assert p15_contract.RESPONSE_AUX_DIM == 30
    assert p15_contract.PRIVILEGED_WIRE_DIM == 346
    assert p15_contract.RESPONSE_OBSERVATION_DIM == 32
    assert [p15_contract.phase_label(value) for value in (0.0, 0.6, 1.5, 3.0, 7.1)] == [
        "responsebase",
        "responseexpand",
        "responseexpand",
        "responsefull",
        "responsecalib",
    ]


def test_command_distribution_and_independent_joint_axes():
    num_envs = 20000
    schedule = P15CommandSchedule(num_envs, "cpu", seed=17)
    state = schedule.step(torch.zeros(num_envs, 3), elapsed_h=3.0)
    metrics = schedule.metrics()
    expected = dict(
        zip(p15_contract.COMMAND_FAMILIES, p15_contract.COMMAND_FAMILY_WEIGHTS)
    )
    for name, probability in expected.items():
        assert abs(metrics["resampled_family_ratio"][name] - probability) < 0.02
    for name, probability in zip(
        p15_contract.TRAJECTORY_MODES, p15_contract.TRAJECTORY_MODE_WEIGHTS
    ):
        assert abs(metrics["trajectory_ratio"][name] - probability) < 0.02
    assert abs(metrics["original_replay_ratio"] - 0.25) < 0.02
    assert abs(sum(metrics["active_family_ratio"].values()) - 1.0) < 1.0e-6
    assert metrics["active_frame_count"] == num_envs
    expanded_joint = (
        (state.family == 0)
        & (state.trajectory_mode == 1)
        & (~state.original_replay)
        & (state.change_mode == 2)
    )
    assert bool(expanded_joint.any())
    assert float(state.active_target[expanded_joint, 0].min()) < 0.05
    assert float(state.active_target[expanded_joint, 2].abs().min()) < 0.03

    source_schedule = P15CommandSchedule(num_envs, "cpu", seed=19)
    source = source_schedule.step(torch.zeros(num_envs, 3), elapsed_h=0.0)
    joint = (
        (source.family == 0)
        & (source.trajectory_mode == 1)
        & (source.change_mode == 2)
    )
    vx = source.active_target[joint, 0]
    wz = source.active_target[joint, 2].abs()
    correlation = torch.corrcoef(torch.stack((vx, wz)))[0, 1]
    assert abs(float(correlation)) < 0.08
    assert float(vx.max()) <= 1.3 + 1.0e-6
    assert float(wz.max()) <= 0.3 + 1.0e-6


def test_target_is_5hz_and_exec_is_50hz_slew_limited():
    schedule = P15CommandSchedule(256, "cpu", seed=3)
    native = torch.zeros(256, 3)
    previous_target = None
    previous_exec = None
    changed_frames = []
    for frame in range(31):
        state = schedule.step(native, elapsed_h=3.0)
        if previous_target is not None and not torch.equal(
            state.active_target, previous_target
        ):
            changed_frames.append(frame)
        if previous_exec is not None:
            normal = ~state.emergency_stop
            delta = (state.exec_command[normal] - previous_exec[normal]).abs().amax(dim=0)
            assert float(delta[0]) <= 0.0060001
            assert float(delta[1]) <= 0.0060001
            assert float(delta[2]) <= 0.0200001
            assert torch.all(state.exec_command[state.emergency_stop] == 0.0)
        previous_target = state.active_target
        previous_exec = state.exec_command
    assert changed_frames
    assert all(frame % 10 == 0 for frame in changed_frames)


def test_transition_brake_can_bypass_slew_as_emergency_stop():
    schedule = P15CommandSchedule(4096, "cpu", seed=113)
    native = torch.full((4096, 3), 0.8)
    state = schedule.step(native, elapsed_h=3.0)
    emergency_brake = state.emergency_stop
    assert bool(emergency_brake.any())
    assert torch.all(state.family[emergency_brake] == 3)
    assert torch.all(state.trajectory_mode[emergency_brake] == 1)
    assert torch.all(state.active_target[emergency_brake] == 0.0)
    assert torch.all(state.exec_command[emergency_brake] == 0.0)


def test_command_phase_boundaries_and_piecewise_union_contract():
    expected = (
        (0.499, 1.3, 0.2, 0.3, "responsebase"),
        (0.5, 1.0, 0.2, 0.5, "responseexpand"),
        (1.25, 1.0, 0.25, 0.65, "responseexpand"),
        (2.0, 1.0, 0.3, 0.8, "responsefull"),
        (7.0, 1.0, 0.3, 0.8, "responsecalib"),
    )
    for elapsed, vx, vy, wz, label in expected:
        limits = p15_contract.command_limits(elapsed)
        assert limits["vx"][1] == vx
        assert limits["vy"] == (-vy, vy)
        assert limits["wz"] == (-wz, wz)
        assert p15_contract.phase_label(elapsed) == label
    envelope = p15_contract.command_contract()["command_envelope"]
    assert envelope["type"] == "piecewise_union_v1"
    assert envelope["branches"]["main_vx_wz"]["vy"] == [0.0, 0.0]
    capability = p15_contract.command_contract()["capability_profile"]
    assert capability["version"] == "piecewise_union_profile_v1"
    assert len(capability["layout"]) == len(capability["values"]) == 15
    assert "vy_specialty_abs_vy_max" in capability["layout"]
    assert "source_replay_abs_wz_max" in capability["layout"]


def test_smooth_target_updates_epoch_for_future_label_masking():
    schedule = P15CommandSchedule(512, "cpu", seed=71)
    native = torch.zeros(512, 3)
    first = schedule.step(native, elapsed_h=3.0)
    smooth = first.trajectory_mode == 0
    assert bool(smooth.any())
    epoch_before = first.command_epoch.clone()
    for _ in range(10):
        state = schedule.step(native, elapsed_h=3.0)
    changed = smooth & (state.active_target - first.active_target).abs().amax(dim=1).gt(0)
    assert bool(changed.any())
    assert torch.all(state.command_epoch[changed] > epoch_before[changed])


def test_p15_anchor_keeps_parent_supported_yaw_lateral_and_zero():
    commands = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.25],
            [0.0, 0.18, 0.0],
            [0.8, 0.0, 0.25],
            [0.8, 0.0, 0.60],
        ]
    )
    weights = p15_contract.p15_anchor_weights_from_commands(commands).flatten()
    assert torch.equal(weights[:4], torch.ones(4))
    assert weights[4] == 0.25
