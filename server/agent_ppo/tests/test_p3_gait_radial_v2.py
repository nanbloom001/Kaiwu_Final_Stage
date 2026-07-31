import pytest
import torch

from agent_ppo.algorithm.algorithm_p3_standard_joint import AlgorithmP3HighPPO
from agent_ppo.feature import p2_contract, p3_contract
from agent_ppo.feature.p3_command_sampler import P3RecoveryCommandSampler
from agent_ppo.feature.p3_gait import (
    P3GaitBaseline,
    mirror_action,
    mirror_proprio,
    mirror_scan_from_coordinates,
    validate_joint_order,
)
from agent_ppo.feature.p2_worker_bridge import P2WorkerBridge


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
        for leg in ("FL", "FR", "RL", "RR")
        for axis in ("hip", "thigh", "calf")
    ]
    assert validate_joint_order(names)
    names[3], names[6] = names[6], names[3]
    assert not validate_joint_order(names)


def test_p3_command_sampler_ranges_and_sign_balance():
    sampler = P3RecoveryCommandSampler(num_envs=20000, device="cpu")
    commands, buckets = sampler._sample_target_commands(20000)
    assert commands[:, 0].min() >= -0.25
    assert commands[:, 0].max() <= 1.0
    assert commands[:, 1].abs().max() <= 0.30
    assert commands[:, 2].abs().max() <= 0.90
    lateral = buckets == 4
    turn = buckets == 3
    assert abs(float((commands[lateral, 1] > 0).float().mean()) - 0.5) < 0.04
    assert abs(float((commands[turn, 2] > 0).float().mean()) - 0.5) < 0.04


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
    aux = torch.zeros(4, p2_contract.WORKER_AUX_DIM)
    aux[:, p2_contract.GAIT_VALID_INDEX] = 1.0
    aux[:, p2_contract.GAIT_SLIP_SPEED_SLICE] = 0.1
    aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE] = 0.7
    aux[:, p2_contract.GAIT_DUTY_SLICE] = 0.7
    extra = torch.zeros(4, p3_contract.P3_WORKER_EXTRA_DIM)
    extra[:, p3_contract.GAIT_CONTINUOUS_STANCE_SLICE] = 0.8
    contact, crossing, starvation = baseline.rewards(aux, extra, 1.0)
    assert torch.equal(contact, torch.zeros_like(contact))
    assert torch.equal(crossing, torch.zeros_like(crossing))
    assert torch.equal(starvation, torch.zeros_like(starvation))

    aux[0, p2_contract.GAIT_SLIP_SPEED_SLICE.start] = 1.0
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


def test_unfinished_gait_baseline_checkpoint_preserves_samples():
    baseline = P3GaitBaseline("cpu")
    key = (2, 3, "impact")
    baseline.samples[key] = [torch.tensor([0.4, 0.5])]
    baseline.sample_counts[key] = 2
    baseline.total_samples = 2
    restored = P3GaitBaseline("cpu")
    restored.load_state_dict(baseline.state_dict())
    assert restored.sample_counts[key] == 2
    assert torch.equal(restored.samples[key][0], torch.tensor([0.4, 0.5]))


def test_adapter_calibration_skips_high_actor_optimizer_step():
    algorithm = object.__new__(AlgorithmP3HighPPO)
    algorithm.session_effective_seconds = 5500.0
    assert not algorithm._actor_update_enabled()
    algorithm.session_effective_seconds = 6100.0
    assert algorithm._actor_update_enabled()


def test_p3_reset_keeps_static_joint_mapping_valid():
    bridge = object.__new__(P2WorkerBridge)
    bridge.num_envs = 2
    bridge.device = torch.device("cpu")
    bridge.last_p3_extra = torch.zeros(2, p3_contract.P3_WORKER_EXTRA_DIM)
    bridge._gait_window = type(
        "Gait",
        (),
        {"valid": False, "last_p3_detail": torch.zeros(2, 20)},
    )()
    bridge._runtime_terrain_size_x = lambda: 8.0
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
                        for leg in ("FL", "FR", "RL", "RR")
                        for axis in ("hip", "thigh", "calf")
                    ],
                    "applied_torque": torch.ones(2, 12),
                    "joint_vel": torch.ones(2, 12),
                },
            )()
        },
    )()
    extra = bridge._p3_extra(robot, torch.tensor([True, False]))
    assert extra[0, p3_contract.JOINT_MAPPING_VALID_INDEX] == 1.0
    assert extra[0, p3_contract.GAIT_CONTACT_ONSET_SLICE].sum() == 0.0
