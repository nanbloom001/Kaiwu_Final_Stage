import pytest
import torch
import time

from agent_ppo.algorithm.algorithm_visual_ppo import _calibrate_auxiliary_gradients
from agent_ppo.algorithm.algorithm_p3_standard_joint import AlgorithmP3HighPPO
from agent_ppo.feature import p2_contract, p3_contract
from agent_ppo.feature.p3_command_sampler import P3RecoveryCommandSampler
from agent_ppo.feature.p3_gait import (
    P35LowRewardShaper,
    P3ActionSmoothAuxiliary,
    P3GaitBaseline,
    mirror_action,
    per_leg_action_mse,
    mirror_proprio,
    mirror_scan_from_coordinates,
    validate_mirror_assembly,
    validate_joint_order,
)
from agent_ppo.feature.p2_worker_bridge import P2WorkerBridge
from agent_ppo.feature.reward_process import p3_normalized_torque_excess
from agent_ppo.workflow.p3_standard_joint_workflow import (
    _accumulate_action_bucket,
    _accumulate_command,
    _accumulate_outcomes,
    _command_accumulator,
    _command_metrics,
)


def test_torque_soft_constraint_starts_at_80_percent_and_is_squared():
    torque = torch.tensor([17.6, 19.8, 22.0, 43.0, 47.3])
    soft = torch.tensor([17.6, 17.6, 17.6, 34.4, 34.4])
    hard = torch.tensor([22.0, 22.0, 22.0, 43.0, 43.0])
    excess = p3_normalized_torque_excess(torque, soft, hard)
    assert excess.tolist() == pytest.approx([0.0, 0.25, 1.0, 1.0, 2.25])


def _ready_p35_reward_shaper(num_envs=2):
    shaper = P35LowRewardShaper(num_envs, "cpu")
    shaper.finalized = True
    shaper.valid = True
    shaper.thresholds = {
        "joint_pos": torch.ones(12),
        "joint_acc_noncontact": torch.full((12,), 10.0),
        "joint_acc_onset": torch.full((12,), 15.0),
        "posture": torch.ones(4),
        "frequency": torch.full((4,), 0.5),
    }
    return shaper


def test_p35_reward_shaper_is_zero_inside_parent_envelope():
    shaper = _ready_p35_reward_shaper()
    proprio = torch.zeros(2, 45)
    aux = torch.zeros(2, p2_contract.WORKER_AUX_DIM)
    next_aux = aux.clone()
    aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE] = 0.6
    extra = torch.zeros(2, p3_contract.P3_WORKER_EXTRA_DIM)
    extra[:, p3_contract.JOINT_ACCELERATION_MAPPING_VALID_INDEX] = 1.0
    extra[:, p3_contract.CONTACT_REWARD_MAPPING_VALID_INDEX] = 1.0
    extra[:, p3_contract.SECONDS_SINCE_PUSH_INDEX] = 10.0
    reward, components = shaper.rewards(
        proprio=proprio,
        aux=aux,
        next_aux=next_aux,
        extra=extra,
        command=torch.zeros(2, 3),
        dones=torch.zeros(2, dtype=torch.bool),
    )
    assert torch.equal(reward, torch.zeros_like(reward))
    assert all(
        torch.equal(value, torch.zeros_like(value))
        for value in components.values()
    )


def test_p35_reward_shaper_contact_and_push_grace_are_bounded():
    normal = _ready_p35_reward_shaper(1)
    grace = _ready_p35_reward_shaper(1)
    proprio = torch.zeros(1, 45)
    aux = torch.zeros(1, p2_contract.WORKER_AUX_DIM)
    aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE] = 0.6
    extra = torch.zeros(1, p3_contract.P3_WORKER_EXTRA_DIM)
    extra[:, p3_contract.JOINT_ACCELERATION_MAPPING_VALID_INDEX] = 1.0
    extra[:, p3_contract.CONTACT_REWARD_MAPPING_VALID_INDEX] = 1.0
    extra[:, p3_contract.JOINT_ACCELERATION_SLICE] = 20.0
    extra[:, p3_contract.CONTACT_FORCE_SLICE.start] = 20.0
    extra[:, p3_contract.CONTACT_ONSET_SLICE.start] = 1.0
    extra[:, p3_contract.SECONDS_SINCE_PUSH_INDEX] = 1.0
    _, normal_components = normal.rewards(
        proprio=proprio,
        aux=aux,
        next_aux=aux,
        extra=extra,
        command=torch.zeros(1, 3),
        dones=torch.zeros(1, dtype=torch.bool),
    )
    extra[:, p3_contract.SECONDS_SINCE_PUSH_INDEX] = 0.2
    _, grace_components = grace.rewards(
        proprio=proprio,
        aux=aux,
        next_aux=aux,
        extra=extra,
        command=torch.zeros(1, 3),
        dones=torch.zeros(1, dtype=torch.bool),
    )
    assert -0.06 <= float(normal_components["contact"]) < 0.0
    assert grace_components["contact"] == pytest.approx(
        normal_components["contact"]
    )
    assert grace_components["joint_acc"] == pytest.approx(
        0.5 * normal_components["joint_acc"]
    )


def test_p35_push_wrapper_records_actual_delta_and_switches_public_term():
    root_velocity = torch.zeros(3, 6)

    def push_by_setting_velocity(_env, env_ids, **_kwargs):
        root_velocity[env_ids, 0] += 0.03
        root_velocity[env_ids, 1] -= 0.02

    cfg = type("Cfg", (), {})()
    cfg.func = push_by_setting_velocity
    cfg.params = {"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}
    cfg.interval_range_s = (12.0, 18.0)

    class Manager:
        active_terms = {"interval": ["push_robot"]}

        def __init__(self):
            self.reset_count = 0

        def get_term_cfg(self, name):
            assert name == "push_robot"
            return cfg

        def set_term_cfg(self, name, value):
            assert name == "push_robot"
            assert value is cfg

        def reset(self, _ids=None):
            self.reset_count += 1

    manager = Manager()
    bridge = object.__new__(P2WorkerBridge)
    bridge.num_envs = 3
    bridge.device = torch.device("cpu")
    bridge.config = {"push_schedule": {"term_name": "push_robot"}}
    bridge.env = type("Env", (), {"event_manager": manager})()
    bridge._robot = lambda: type(
        "Robot", (), {"data": type("Data", (), {"root_vel_w": root_velocity})()}
    )()
    bridge._p35_push_event_flag = torch.zeros(3, dtype=torch.bool)
    bridge._p35_push_delta = torch.zeros(3, 2)
    bridge._p35_seconds_since_push = torch.full((3,), 1.0e6)
    bridge._p35_push_runtime_active = False
    bridge._p35_push_telemetry_valid = False
    bridge._p35_push_event_count = 0
    bridge._p35_push_phase_name = None
    bridge._p35_resume_offset_s = 4500.0
    bridge._p35_started_monotonic = time.monotonic()
    bridge._p35_install_push_wrapper()
    assert bridge._p35_push_telemetry_valid
    cfg.func(bridge.env, torch.tensor([0, 2]), **cfg.params)
    assert bridge._p35_push_event_flag.tolist() == [True, False, True]
    assert torch.allclose(
        bridge._p35_push_delta[[0, 2]], torch.tensor([[0.03, -0.02], [0.03, -0.02]])
    )
    bridge._p35_update_push_phase()
    assert bridge._p35_push_runtime_active
    assert cfg.params["velocity_range"]["x"] == (-0.05, 0.05)
    assert manager.reset_count == 1


def test_mirror_contract_is_an_involution():
    action = torch.randn(7, 12)
    proprio = torch.randn(7, 45)
    assert torch.allclose(mirror_action(mirror_action(action)), action)
    assert torch.allclose(mirror_proprio(mirror_proprio(proprio)), proprio)
    lateral = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    scan = torch.arange(5.0).repeat(2, 1)
    assert torch.equal(
        mirror_scan_from_coordinates(
            mirror_scan_from_coordinates(scan, lateral), lateral
        ),
        scan,
    )


def test_joint_order_validation_is_explicit_and_symmetric():
    names = [
        f"{leg}_{axis}_joint"
        for axis in ("hip", "thigh", "calf")
        for leg in ("FL", "FR", "RL", "RR")
    ]
    assert validate_joint_order(names)
    leg_major = [
        f"{leg}_{axis}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for axis in ("hip", "thigh", "calf")
    ]
    assert not validate_joint_order(leg_major)
    names[3], names[6] = names[6], names[3]
    assert not validate_joint_order(names)


def test_mirror_assembly_requires_symmetric_pd_effort_and_action_scale():
    names = [
        f"{leg}_{axis}_joint"
        for axis in ("hip", "thigh", "calf")
        for leg in ("FL", "FR", "RL", "RR")
    ]
    data = type(
        "Data",
        (),
        {
            "joint_names": names,
            "joint_stiffness": torch.full((1, 12), 20.0),
            "joint_damping": torch.full((1, 12), 0.5),
            "joint_effort_limits": torch.full((1, 12), 25.0),
        },
    )()
    robot = type("Robot", (), {"data": data})()
    term = type(
        "JointPositionAction",
        (),
        {"action_dim": 12, "cfg": type("Cfg", (), {"scale": 0.25})()},
    )()
    env = type(
        "Env",
        (),
        {"action_manager": type("Manager", (), {"_terms": {"JointPositionAction": term}})()},
    )()
    valid, checks = validate_mirror_assembly(robot, env)
    assert valid and all(checks.values())
    data.joint_names = None
    robot.joint_names = names
    valid, checks = validate_mirror_assembly(robot, env)
    assert valid and all(checks.values())
    data.joint_damping[0, 3] = 0.75
    valid, checks = validate_mirror_assembly(robot, env)
    assert not valid and not checks["damping"]


def test_mirror_assembly_ignores_unrelated_action_terms_and_checks_joint_contract():
    names = [
        f"{leg}_{axis}_joint"
        for axis in ("hip", "thigh", "calf")
        for leg in ("FL", "FR", "RL", "RR")
    ]
    data = type(
        "Data",
        (),
        {
            "joint_names": names,
            "joint_stiffness": torch.full((1, 12), 20.0),
            "joint_damping": torch.full((1, 12), 0.5),
            "joint_effort_limits": torch.full((1, 12), 25.0),
        },
    )()
    robot = type("Robot", (), {"data": data})()
    unrelated = type(
        "GripperAction",
        (),
        {"action_dim": 1, "cfg": type("Cfg", (), {"scale": 0.25})()},
    )()
    joint = type(
        "JointPositionAction",
        (),
        {"action_dim": 12, "cfg": type("Cfg", (), {"scale": 0.20})()},
    )()
    manager = type(
        "Manager",
        (),
        {"_terms": {"GripperAction": unrelated, "JointPositionAction": joint}},
    )()
    valid, checks = validate_mirror_assembly(
        robot, type("Env", (), {"action_manager": manager})()
    )
    assert not valid
    assert not checks["action_scale"]


def test_per_leg_mirror_error_uses_axis_major_action_layout():
    prediction = torch.zeros(1, 12)
    target = torch.zeros_like(prediction)
    prediction[:, (0, 4, 8)] = 2.0
    errors = per_leg_action_mse(prediction, target)
    assert errors["fl"] == pytest.approx(4.0)
    assert errors["fr"] == 0.0
    assert errors["rl"] == 0.0
    assert errors["rr"] == 0.0


def test_p3_command_sampler_ranges_and_sign_balance():
    sampler = P3RecoveryCommandSampler(num_envs=20000, device="cpu")
    commands, buckets = sampler._sample_target_commands(20000)
    assert commands[:, 0].min() >= 0.0
    assert commands[:, 0].max() <= 1.0
    assert commands[:, 1].abs().max() <= 0.30
    assert commands[:, 2].abs().max() <= 0.90
    lateral = buckets == 2
    turn = buckets == 3
    assert abs(float((commands[lateral, 1] > 0).float().mean()) - 0.5) < 0.04
    assert abs(float((commands[turn, 2] > 0).float().mean()) - 0.5) < 0.04
    assert not bool((buckets == 1).any())


def test_p35_vx_bands_and_restart_share_one_distribution():
    sampler = P3RecoveryCommandSampler(num_envs=20000, device="cpu")
    values = sampler._sample_vx(20000)
    assert float(values.min()) >= 0.10
    assert float(values.max()) <= 1.00
    assert float((values < 0.35).float().mean()) == pytest.approx(0.55, abs=0.02)
    assert float(((values >= 0.35) & (values < 0.70)).float().mean()) == pytest.approx(
        0.30, abs=0.02
    )
    restart = P3RecoveryCommandSampler(
        num_envs=12000,
        device="cpu",
        config={"bucket_weights": [0, 0, 0, 0, 0, 1, 0]},
    )
    commands, _ = restart._sample_target_commands(12000)
    moving = commands[:, 0] > 0.0
    assert float(moving.float().mean()) == pytest.approx(0.50, abs=0.03)
    assert float(commands[moving, 0].min()) >= 0.10
    assert bool((commands[moving, 0] >= 0.70).any())


def test_p35_joint_acceleration_uses_normal_foot_touchdown():
    shaper = _ready_p35_reward_shaper(1)
    proprio = torch.zeros(1, 45)
    aux = torch.zeros(1, p2_contract.WORKER_AUX_DIM)
    aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE] = 0.6
    extra = torch.zeros(1, p3_contract.P3_WORKER_EXTRA_DIM)
    extra[:, p3_contract.JOINT_ACCELERATION_MAPPING_VALID_INDEX] = 1.0
    extra[:, p3_contract.CONTACT_REWARD_MAPPING_VALID_INDEX] = 1.0
    extra[:, p3_contract.SECONDS_SINCE_PUSH_INDEX] = 10.0
    joint_acc = extra[:, p3_contract.JOINT_ACCELERATION_SLICE]
    joint_acc[:, (0, 4, 8)] = 12.0
    _, noncontact = shaper.rewards(
        proprio=proprio,
        aux=aux,
        next_aux=aux,
        extra=extra,
        command=torch.zeros(1, 3),
        dones=torch.zeros(1, dtype=torch.bool),
    )
    extra[:, p3_contract.GAIT_CONTACT_ONSET_SLICE.start] = 1.0
    _, touchdown = shaper.rewards(
        proprio=proprio,
        aux=aux,
        next_aux=aux,
        extra=extra,
        command=torch.zeros(1, 3),
        dones=torch.zeros(1, dtype=torch.bool),
    )
    assert float(noncontact["joint_acc"]) < 0.0
    assert touchdown["joint_acc"] == 0.0


def test_p35_joint_acceleration_baseline_does_not_leak_between_legs():
    shaper = P35LowRewardShaper(128, "cpu")
    proprio = torch.zeros(128, 45)
    aux = torch.zeros(128, p2_contract.WORKER_AUX_DIM)
    extra = torch.zeros(128, p3_contract.P3_WORKER_EXTRA_DIM)
    extra[:, p3_contract.JOINT_ACCELERATION_MAPPING_VALID_INDEX] = 1.0
    extra[:, p3_contract.CONTACT_REWARD_MAPPING_VALID_INDEX] = 1.0
    healthy = torch.ones(128, dtype=torch.bool)
    extra[:, p3_contract.JOINT_ACCELERATION_SLICE] = 10.0
    shaper.observe(proprio, aux, extra, healthy)

    extra[:, p3_contract.GAIT_CONTACT_ONSET_SLICE.start] = 1.0
    onset_acc = extra[:, p3_contract.JOINT_ACCELERATION_SLICE]
    onset_acc[:, :] = 1000.0
    onset_acc[:, (0, 4, 8)] = 20.0
    shaper.observe(proprio, aux, extra, healthy)
    shaper.finalize()

    threshold = shaper.thresholds["joint_acc_onset"]
    assert threshold[[0, 4, 8]].tolist() == pytest.approx([20.0, 20.0, 20.0])
    assert threshold[[1, 5, 9]].tolist() == pytest.approx([15.0, 15.0, 15.0])


def test_p35_posture_uses_roll_angle_as_primary_axis():
    shaper = _ready_p35_reward_shaper(1)
    proprio = torch.zeros(1, 45)
    aux = torch.zeros(1, p2_contract.WORKER_AUX_DIM)
    aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE] = 0.6
    extra = torch.zeros(1, p3_contract.P3_WORKER_EXTRA_DIM)
    extra[:, p3_contract.JOINT_ACCELERATION_MAPPING_VALID_INDEX] = 1.0
    extra[:, p3_contract.CONTACT_REWARD_MAPPING_VALID_INDEX] = 1.0
    extra[:, p3_contract.SECONDS_SINCE_PUSH_INDEX] = 10.0
    roll = aux.clone()
    roll[:, 22] = 2.0
    pitch = aux.clone()
    pitch[:, 21] = 2.0
    _, roll_components = shaper.rewards(
        proprio=proprio, aux=roll, next_aux=roll, extra=extra,
        command=torch.zeros(1, 3), dones=torch.zeros(1, dtype=torch.bool),
    )
    _, pitch_components = shaper.rewards(
        proprio=proprio, aux=pitch, next_aux=pitch, extra=extra,
        command=torch.zeros(1, 3), dones=torch.zeros(1, dtype=torch.bool),
    )
    assert float(roll_components["posture"]) < float(pitch_components["posture"])


def test_p3_command_sampler_rejects_enabling_unreachable_reverse_domain():
    with pytest.raises(ValueError, match="reverse_recovery"):
        P3RecoveryCommandSampler(
            num_envs=8,
            device="cpu",
            config={"bucket_weights": [0.25, 0.15, 0.20, 0.15, 0.10, 0.10, 0.05]},
        )


def test_p3_contract_records_forward_only_command_and_gait_fallback_semantics():
    value = p3_contract.contract()
    assert value["command_domain"]["vx_m_s"] == [0.0, 1.0]
    assert value["command_domain"]["bucket_weights"][1] == 0.0
    assert value["command_domain"]["reverse_recovery_enabled"] is False
    assert value["gait_baseline"]["version"] == 3
    assert value["gait_baseline"]["motion_buckets"] == [
        "low_speed",
        "forward",
        "turn_lateral",
    ]
    assert value["gait_baseline"]["fallback_reward_scale"]["global"] == 0.0
    assert [
        (phase["name"], phase["start_s"], phase["end_s"])
        for phase in value["training_schedule"]
    ] == [
        ("gaitfixcalib", 0.0, 900.0),
        ("repair", 900.0, 4500.0),
        ("pushwarm", 4500.0, 5400.0),
        ("pushfull", 5400.0, 6300.0),
        ("stable", 6300.0, 7200.0),
    ]
    assert value["high_level_training"] == "disabled_full_session"
    assert "short_high_adaptation_preserves" not in value


def test_p3_command_sampler_is_seeded_and_zero_hold_uses_bucket_six():
    first = P3RecoveryCommandSampler(
        num_envs=64, device="cpu", config={"seed": 91, "zero_hold_s": [0.4, 0.6]}
    )
    second = P3RecoveryCommandSampler(
        num_envs=64, device="cpu", config={"seed": 91, "zero_hold_s": [0.4, 0.6]}
    )
    first_commands, first_buckets = first._sample_target_commands(64)
    second_commands, second_buckets = second._sample_target_commands(64)
    assert torch.equal(first_buckets, second_buckets)
    assert torch.equal(first_commands, second_commands)

    buckets = torch.tensor([0, 2, 6, 6])
    holds = first._sample_hold(target=True, buckets=buckets)
    assert bool(((holds[2:] >= 0.4) & (holds[2:] <= 0.6)).all())
    weights = first._anchor_weight_for_targets(
        torch.arange(7), torch.zeros(7, 3)
    )
    assert weights.tolist() == pytest.approx([0.35, 0.0, 0.30, 0.25, 0.15, 0.10, 0.10])


def test_p3_command_sampler_does_not_advance_global_torch_rng():
    sampler = P3RecoveryCommandSampler(num_envs=16, device="cpu", config={"seed": 91})
    before = torch.random.get_rng_state().clone()
    plan = sampler.plan(torch.zeros(16, 3))
    after = torch.random.get_rng_state()
    assert plan.requested_target_ids.numel() == 16
    assert torch.equal(before, after)


def test_combined_auxiliary_gradient_uses_actual_hard_cap():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    policy_loss = parameter[0]
    first = parameter[0] + parameter[1]
    second = parameter[0] - parameter[1]
    calibrated, combined_ratio = _calibrate_auxiliary_gradients(
        policy_loss,
        [("first", first, 0.08), ("second", second, 0.08)],
        [parameter],
        0.10,
    )
    auxiliary_loss = sum(item["loss"] * item["multiplier"] for item in calibrated)
    auxiliary_grad = torch.autograd.grad(auxiliary_loss, parameter)[0]
    policy_grad = torch.autograd.grad(policy_loss, parameter)[0]
    actual_ratio = auxiliary_grad.norm() / policy_grad.norm()
    assert combined_ratio == pytest.approx(0.10, abs=1.0e-6)
    assert actual_ratio == pytest.approx(0.10, abs=1.0e-6)


def test_action_mean_smoothing_masks_reset_and_soft_range_is_inactive_inside_limit():
    auxiliary = P3ActionSmoothAuxiliary()
    auxiliary.begin_rollout(1.0)
    means = torch.zeros(3, 1, 12)
    means[1:] = 4.0
    continuation = torch.ones(3, 1, 1, dtype=torch.bool)
    continuation[0] = False
    smooth, range_loss, metrics = auxiliary.loss(means, continuation)
    assert smooth == 0.0
    assert range_loss == 0.0
    assert metrics["action_mean_rate_loss"] == 0.0

    oscillating = torch.zeros(4, 1, 12)
    oscillating[1] = 4.0
    oscillating[2] = -4.0
    oscillating[3] = 4.0
    smooth, range_loss, _ = auxiliary.loss(
        oscillating, torch.ones(4, 1, 1, dtype=torch.bool)
    )
    assert smooth > 0.0
    assert range_loss == 0.0

    extreme = torch.full((3, 1, 12), 6.0)
    _, range_loss, _ = auxiliary.loss(
        extreme, torch.ones(3, 1, 1, dtype=torch.bool)
    )
    assert range_loss > 0.0


def test_action_diagnostics_detect_high_frequency_and_clipping():
    steps = 80
    time = torch.arange(steps, dtype=torch.float32) * 0.02
    alternating = torch.sin(2.0 * torch.pi * 20.0 * time).reshape(steps, 1, 1)
    mean = alternating.repeat(1, 2, 12)
    sampled = mean.clone()
    sampled[0, 0, 0] = 7.0
    metrics = P3ActionSmoothAuxiliary.diagnostics(
        mean, sampled, torch.zeros(steps, 2, 1, dtype=torch.bool)
    )
    assert metrics["action_clip_rate"] > 0.0
    assert metrics["action_15_25hz_power_ratio_hip"] > 0.9
    assert metrics["action_spectrum_valid_env_share"] == 1.0


def test_action_smooth_aux_detaches_actor_body_but_updates_features_and_head():
    actor = torch.nn.Sequential(
        torch.nn.Linear(6, 8),
        torch.nn.ELU(),
        torch.nn.Linear(8, 12),
    )
    features = torch.randn(4, 2, 6, requires_grad=True)
    output = P3ActionSmoothAuxiliary.action_mean_with_detached_body(actor, features)
    output.square().mean().backward()
    assert features.grad is not None
    assert actor[0].weight.grad is None
    assert actor[0].bias.grad is None
    assert actor[2].weight.grad is not None
    assert actor[2].bias.grad is not None


def test_p3_monitor_keeps_signed_velocity_and_conditioned_gait_statistics():
    accumulator = _command_accumulator(torch.device("cpu"))
    target = torch.tensor(
        ((0.1, 0.0, 0.0), (0.4, 0.0, 0.0), (0.1, -0.2, 0.0), (0.3, 0.2, -0.4))
    )
    aux = torch.zeros(4, p2_contract.WORKER_AUX_DIM)
    aux[:, 3:6] = target
    aux[:, 6:9] = target
    aux[:, 12:15] = target
    aux[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX] = torch.tensor((0.0, 3.0, 6.0, 13.0))
    aux[:, p2_contract.GAIT_VALID_INDEX] = 1.0
    extra = torch.zeros(4, p3_contract.P3_WORKER_EXTRA_DIM)
    extra[:, p3_contract.GAIT_COMPLETED_SLIP_SLICE] = 0.2
    extra[:, p3_contract.GAIT_COMPLETED_SLIP_EVENT_SLICE] = 1.0
    extra[:, p3_contract.GAIT_IMPACT_SPEED_SLICE] = 0.4
    extra[:, p3_contract.GAIT_CONTACT_ONSET_SLICE] = 1.0
    extra[:, p3_contract.GAIT_CONTINUOUS_STANCE_SLICE] = 0.6
    _accumulate_command(accumulator, target, target, aux, extra)
    metrics = _command_metrics(accumulator)
    assert metrics["target_vy_positive_mean"] == pytest.approx(0.2)
    assert metrics["target_vy_negative_mean"] == pytest.approx(-0.2)
    assert metrics["p3_slope_low_speed_fl_slip_distance"] == pytest.approx(0.2)
    assert metrics["p3_stairs_turn_lateral_fl_impact_speed"] == pytest.approx(0.4)
    assert metrics["p3_slope_low_speed_sample_share"] == pytest.approx(0.25)
    assert metrics["p3_slope_inv_forward_sample_share"] == pytest.approx(0.25)
    assert metrics["p3_stairs_turn_lateral_sample_share"] == pytest.approx(0.25)


def test_p35_terrain_buckets_match_15_15_35_35_columns():
    aux = torch.zeros(20, p2_contract.WORKER_AUX_DIM)
    aux[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX] = torch.arange(20)
    buckets = P3GaitBaseline._terrain_bucket(aux)
    assert torch.bincount(buckets, minlength=4).tolist() == [3, 3, 7, 7]


def test_near_clip_stair_completion_rate_joins_terminal_and_clip_bucket():
    accumulator = _command_accumulator(torch.device("cpu"))
    aux = torch.zeros(4, p2_contract.WORKER_AUX_DIM)
    aux[:, 24] = 1.0
    aux[:, 25] = torch.tensor((1.0, 2.0, 1.0, 3.0))
    aux[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX] = torch.tensor(
        (8.0, 14.0, 0.0, 8.0)
    )
    near_clip = torch.tensor((0.101, 0.249, 0.120, 0.101))
    _accumulate_outcomes(accumulator, aux, near_clip)
    metrics = _command_metrics(accumulator)
    assert metrics["near_clip_bin_0_stair_attempt_count"] == 2.0
    assert metrics["near_clip_bin_0_stair_completion_rate"] == pytest.approx(0.5)
    assert metrics["near_clip_bin_9_stair_attempt_count"] == 1.0
    assert metrics["near_clip_bin_9_stair_completion_rate"] == 0.0


def test_p3_gait_metrics_exclude_invalid_frames_and_average_impact_per_onset():
    accumulator = _command_accumulator(torch.device("cpu"))
    target = torch.tensor(((0.1, 0.0, 0.0),) * 3)
    aux = torch.zeros(3, p2_contract.WORKER_AUX_DIM)
    aux[:, 3:6] = target
    aux[:, 6:9] = target
    aux[:, 12:15] = target
    aux[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX] = 0.0
    aux[:, p2_contract.GAIT_VALID_INDEX] = torch.tensor((1.0, 1.0, 0.0))
    extra = torch.zeros(3, p3_contract.P3_WORKER_EXTRA_DIM)
    extra[:, p3_contract.GAIT_COMPLETED_SLIP_SLICE] = torch.tensor(
        ((0.2,) * 4, (0.4,) * 4, (9.0,) * 4)
    )
    extra[:2, p3_contract.GAIT_COMPLETED_SLIP_EVENT_SLICE] = 1.0
    extra[:, p3_contract.GAIT_CONTINUOUS_STANCE_SLICE] = torch.tensor(
        ((0.3,) * 4, (0.5,) * 4, (9.0,) * 4)
    )
    extra[0, p3_contract.GAIT_CONTACT_ONSET_SLICE] = 1.0
    extra[0, p3_contract.GAIT_IMPACT_SPEED_SLICE] = 0.6
    extra[1, p3_contract.GAIT_IMPACT_SPEED_SLICE] = 8.0
    extra[2, p3_contract.GAIT_CONTACT_ONSET_SLICE] = 1.0
    extra[2, p3_contract.GAIT_IMPACT_SPEED_SLICE] = 9.0
    _accumulate_command(accumulator, target, target, aux, extra)
    metrics = _command_metrics(accumulator)
    assert metrics["fl_slip_distance"] == pytest.approx(0.3)
    assert metrics["fl_impact_speed"] == pytest.approx(0.6)
    assert metrics["fl_continuous_stance"] == pytest.approx(0.4)
    assert metrics["p3_slope_low_speed_fl_slip_distance"] == pytest.approx(0.3)
    assert metrics["p3_slope_low_speed_fl_impact_speed"] == pytest.approx(0.6)
    assert metrics["p3_slope_low_speed_fl_stance_s"] == pytest.approx(0.4)


def test_command_bucket_invalid_anchor_diagnostic_fails_safe_to_zero():
    accumulator = _command_accumulator(torch.device("cpu"))
    extra = torch.zeros(2, p3_contract.P3_WORKER_EXTRA_DIM)
    extra[:, p3_contract.COMMAND_BUCKET_INDEX] = 0.0
    extra[:, p3_contract.COMMAND_ANCHOR_WEIGHT_INDEX] = torch.tensor(
        [float("nan"), float("inf")]
    )
    _accumulate_action_bucket(accumulator, extra, torch.zeros(2, 12))
    metrics = _command_metrics(accumulator)
    assert metrics["command_bucket_straight_anchor_mean"] == 0.0


def test_p3_torque_monitor_uses_axis_major_joint_groups():
    accumulator = _command_accumulator(torch.device("cpu"))
    accumulator["count"] = torch.tensor(1.0)
    accumulator["torque_samples"].append(
        torch.arange(1.0, 13.0).reshape(1, 12)
    )
    accumulator["mechanical_power"] = torch.tensor(6.0)
    accumulator["mechanical_power_samples"].append(torch.tensor([2.0, 4.0]))
    accumulator["count"] = torch.tensor(2.0)
    metrics = _command_metrics(accumulator)
    assert metrics["hip_torque_max"] == 4.0
    assert metrics["thigh_torque_max"] == 8.0
    assert metrics["calf_torque_max"] == 12.0
    assert metrics["mechanical_power_mean"] == pytest.approx(3.0)
    assert metrics["mechanical_power_p50"] == pytest.approx(3.0)
    assert metrics["mechanical_power_p95"] == pytest.approx(3.9)
    assert metrics["mechanical_power_max"] == pytest.approx(4.0)


def test_p3_sim2real_component_monitor_excludes_invalid_rows():
    accumulator = _command_accumulator(torch.device("cpu"))
    target = torch.zeros(2, 3)
    aux = torch.zeros(2, p2_contract.WORKER_AUX_DIM)
    extra = torch.zeros(2, p3_contract.P3_WORKER_EXTRA_DIM)
    extra[:, p3_contract.SIM2REAL_COMPONENT_SLICE] = torch.tensor(
        [[0.2, 0.3, 0.4, 0.5], [9.0, 9.0, 9.0, 9.0]]
    )
    extra[0, p3_contract.SIM2REAL_COMPONENT_VALID_INDEX] = 1.0
    _accumulate_command(accumulator, target, target, aux, extra)
    metrics = _command_metrics(accumulator)
    assert metrics["p3_sim2real_component_valid_share"] == pytest.approx(0.5)
    assert metrics["p3_sim2real_sustained_torque_raw"] == pytest.approx(0.2)
    assert metrics["p3_sim2real_torque_peak_raw"] == pytest.approx(0.3)
    assert metrics["p3_sim2real_action_rate_raw"] == pytest.approx(0.4)
    assert metrics["p3_sim2real_action_jerk_raw"] == pytest.approx(0.5)


def test_asymmetric_local_progress_prefers_forward_without_overpunishing_reverse():
    positive, negative = p3_contract.asymmetric_local_progress(
        torch.tensor([2.0, 2.0]),
        torch.tensor([1.9, 2.1]),
        torch.tensor([10, 10]),
    )
    assert positive.tolist() == pytest.approx([0.15, 0.0])
    assert negative[0] == 0.0
    assert torch.isclose(negative[1], torch.tensor(-0.040000010281801224))
    assert positive[0] > negative[1].abs()


def test_gait_rewards_are_zero_inside_baseline_and_trigger_independently():
    baseline = P3GaitBaseline("cpu")
    baseline.finalized = True
    thresholds = {
        "slip": 0.2,
        "impact": 0.5,
        "margin": 0.05,
        "stance": 1.0,
        "frequency": 0.5,
        "duty": 0.85,
    }
    baseline.values = {(0, 0, name): value for name, value in thresholds.items()}
    baseline.valid = {(0, 0, name): True for name in thresholds}
    baseline.fallback_levels = {(0, 0, name): 0 for name in thresholds}
    aux = torch.zeros(4, p2_contract.WORKER_AUX_DIM)
    aux[:, p2_contract.GAIT_VALID_INDEX] = 1.0
    aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE] = 0.7
    aux[:, p2_contract.GAIT_DUTY_SLICE] = 0.7
    extra = torch.zeros(4, p3_contract.P3_WORKER_EXTRA_DIM)
    extra[:, p3_contract.GAIT_COMPLETED_SLIP_SLICE] = 0.1
    extra[:, p3_contract.GAIT_COMPLETED_SLIP_EVENT_SLICE] = 1.0
    extra[:, p3_contract.GAIT_CONTINUOUS_STANCE_SLICE] = 0.8
    contact, crossing, starvation = baseline.rewards(aux, extra, 1.0)
    assert torch.equal(contact, torch.zeros_like(contact))
    assert torch.equal(crossing, torch.zeros_like(crossing))
    assert torch.equal(starvation, torch.zeros_like(starvation))

    extra[0, p3_contract.GAIT_COMPLETED_SLIP_SLICE.start] = 1.0
    extra[0, p3_contract.GAIT_COMPLETED_SLIP_EVENT_SLICE.start] = 1.0
    extra[1, p3_contract.GAIT_CONTACT_ONSET_SLICE.start] = 1.0
    extra[1, p3_contract.GAIT_IMPACT_SPEED_SLICE.start] = 1.5
    extra[1, p3_contract.GAIT_TOUCHDOWN_Y_SLICE.start] = 0.10
    extra[2, p3_contract.GAIT_CONTACT_ONSET_SLICE.start] = 1.0
    extra[2, p3_contract.GAIT_TOUCHDOWN_Y_SLICE.start] = -0.1
    extra[3, p3_contract.GAIT_CONTINUOUS_STANCE_SLICE.start] = 2.0
    contact, crossing, starvation = baseline.rewards(aux, extra, 1.0)
    assert contact[0] < 0 and crossing[0] == 0 and starvation[0] == 0
    assert contact[1] < 0 and crossing[1] == 0
    assert crossing[2] < 0 and contact[2] == 0
    assert starvation[3] < 0 and contact[3] == 0 and crossing[3] == 0
    assert torch.all(contact >= -p3_contract.GAIT_CONTACT_REWARD_CAP)
    assert torch.all(crossing >= -p3_contract.GAIT_CROSS_REWARD_CAP)
    assert torch.all(starvation >= -p3_contract.GAIT_STARVATION_REWARD_CAP)
    extra[0, p3_contract.GAIT_COMPLETED_SLIP_EVENT_SLICE.start] = 0.0
    repeated_contact, _, _ = baseline.rewards(aux, extra, 1.0)
    assert repeated_contact[0] == 0.0


def test_gait_baseline_falls_back_by_terrain_then_disables_missing_metrics():
    baseline = P3GaitBaseline("cpu")
    key = (0, 0, "slip")
    baseline.samples[key] = [torch.linspace(0.1, 0.3, baseline.MIN_SAMPLES)]
    baseline.sample_counts[key] = baseline.MIN_SAMPLES
    baseline.total_samples = baseline.MIN_SAMPLES
    baseline.finalize()
    assert baseline.valid[(0, 0, "slip")]
    assert baseline.fallback_levels[(0, 1, "slip")] == 1
    assert baseline.fallback_levels[(1, 0, "slip")] == 2
    assert not baseline.valid[(0, 0, "impact")]


def test_gait_baseline_global_fallback_is_diagnostic_only():
    baseline = P3GaitBaseline("cpu")
    key = (0, 0, "slip")
    baseline.samples[key] = [torch.linspace(0.1, 0.3, baseline.MIN_SAMPLES)]
    baseline.sample_counts[key] = baseline.MIN_SAMPLES
    baseline.total_samples = baseline.MIN_SAMPLES
    baseline.finalize()

    aux = torch.zeros(1, p2_contract.WORKER_AUX_DIM)
    aux[:, p2_contract.GAIT_VALID_INDEX] = 1.0
    aux[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX] = 8.0
    extra = torch.zeros(1, p3_contract.P3_WORKER_EXTRA_DIM)
    extra[:, p3_contract.GAIT_COMPLETED_SLIP_SLICE] = 2.0
    extra[:, p3_contract.GAIT_COMPLETED_SLIP_EVENT_SLICE] = 1.0
    contact, crossing, starvation = baseline.rewards(aux, extra, 1.0)
    assert torch.equal(contact, torch.zeros_like(contact))
    assert torch.equal(crossing, torch.zeros_like(crossing))
    assert torch.equal(starvation, torch.zeros_like(starvation))


def test_unfinished_gait_baseline_checkpoint_preserves_samples():
    baseline = P3GaitBaseline("cpu")
    key = (2, 2, "impact")
    baseline.samples[key] = [torch.tensor([0.4, 0.5])]
    baseline.sample_counts[key] = 2
    baseline.total_samples = 2
    restored = P3GaitBaseline("cpu")
    restored.load_state_dict(baseline.state_dict())
    assert restored.sample_counts[key] == 2
    assert torch.equal(restored.samples[key][0], torch.tensor([0.4, 0.5]))


def test_adapter_calibration_skips_high_actor_optimizer_step():
    algorithm = object.__new__(AlgorithmP3HighPPO)
    algorithm.session_effective_seconds = 7500.0
    assert not algorithm._actor_update_enabled()
    algorithm.session_effective_seconds = 8100.0
    assert not algorithm._actor_update_enabled()


def test_p3_reset_keeps_static_joint_mapping_valid():
    bridge = object.__new__(P2WorkerBridge)
    bridge.num_envs = 2
    bridge.device = torch.device("cpu")
    bridge.last_p3_extra = torch.zeros(2, p3_contract.P3_WORKER_EXTRA_DIM)
    bridge.env = type(
        "Env",
        (),
        {
            "action_manager": type(
                "ActionManager",
                (),
                {
                    "_terms": {
                        "JointPositionAction": type(
                            "JointPositionAction",
                            (),
                            {
                                "action_dim": 12,
                                "cfg": type("Cfg", (), {"scale": 0.25})(),
                            },
                        )()
                    }
                },
            )()
        },
    )()
    bridge._gait_window = type(
        "Gait",
        (),
            {"valid": True, "last_p3_detail": torch.zeros(2, 24)},
    )()
    bridge._runtime_terrain_size_x = lambda: 8.0
    bridge.env._agent_ppo_worker_command_bridge = type(
        "CommandBridge",
        (),
        {
            "scheduler": type(
                "Scheduler",
                (),
                {
                    "bucket": torch.tensor([0, 6]),
                    "anchor_weights": torch.tensor([0.10, 0.0]),
                },
            )()
        },
    )()
    bridge.env._p3_sim2real_components = {
        "sustained_torque": torch.tensor([0.1, 0.2]),
        "torque_peak": torch.tensor([0.3, 0.4]),
        "action_rate": torch.tensor([0.5, 0.6]),
        "action_jerk": torch.tensor([0.7, 0.8]),
    }
    robot = type(
        "Robot",
        (),
        {
            "data": type(
                "Data",
                (),
                {
                        "joint_names": [
                            f"{leg}_{axis}_joint"
                            for axis in ("hip", "thigh", "calf")
                            for leg in ("FL", "FR", "RL", "RR")
                        ],
                    "applied_torque": torch.ones(2, 12),
                    "joint_vel": torch.ones(2, 12),
                    "joint_stiffness": torch.full((2, 12), 20.0),
                    "joint_damping": torch.full((2, 12), 0.5),
                    "joint_effort_limits": torch.full((2, 12), 25.0),
                },
            )()
        },
    )()
    extra = bridge._p3_extra(robot, torch.tensor([True, False]))
    assert extra[0, p3_contract.JOINT_MAPPING_VALID_INDEX] == 1.0
    assert extra[0, p3_contract.GAIT_CONTACT_ONSET_SLICE].sum() == 0.0
    assert extra[:, p3_contract.COMMAND_BUCKET_INDEX].tolist() == [0.0, 6.0]
    assert extra[:, p3_contract.COMMAND_ANCHOR_WEIGHT_INDEX].tolist() == pytest.approx(
        [0.10, 0.0]
    )
    assert extra[0, p3_contract.SIM2REAL_COMPONENT_VALID_INDEX] == 0.0
    assert extra[1, p3_contract.SIM2REAL_COMPONENT_VALID_INDEX] == 1.0
    assert extra[1, p3_contract.SIM2REAL_COMPONENT_SLICE].tolist() == pytest.approx(
        [0.2, 0.4, 0.6, 0.8]
    )

    bridge.env._p3_torque_mapping_valid = torch.tensor([False, True])
    robot.data.joint_damping[0, 3] = 0.75
    cached = bridge._p3_extra(robot, torch.tensor([False, False]))
    assert cached[:, p3_contract.JOINT_MAPPING_VALID_INDEX].tolist() == [1.0, 1.0]
    assert cached[:, p3_contract.SIM2REAL_COMPONENT_VALID_INDEX].tolist() == [
        0.0,
        1.0,
    ]

    del bridge.env._agent_ppo_worker_command_bridge
    missing = bridge._p3_extra(robot, torch.tensor([False, False]))
    assert missing[:, p3_contract.COMMAND_BUCKET_INDEX].tolist() == [-1.0, -1.0]
    assert missing[:, p3_contract.COMMAND_ANCHOR_WEIGHT_INDEX].tolist() == [0.0, 0.0]
