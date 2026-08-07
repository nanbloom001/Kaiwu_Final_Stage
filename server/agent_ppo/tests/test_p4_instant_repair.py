from __future__ import annotations

from types import SimpleNamespace

import torch

from agent_ppo.algorithm.algorithm_p4_nav_ppo import AlgorithmP4NavPPO
from agent_ppo.feature import p2_contract, p4_contract
from agent_ppo.feature.p2_response_buffer import P2ResponseAuxBuffer
from agent_ppo.feature.p2_worker_bridge import _termination_reason_codes


PROFILE = "maze_instant_repair2h"


def test_repair_schedule_and_contract_are_two_hour_instant_profile():
    phases = [
        p4_contract.training_schedule(seconds, branch="instant_repair2h")
        for seconds in (0.0, 300.0, 1_800.0, 5_400.0)
    ]
    assert [phase["phase"] for phase in phases] == [
        "repaircollect",
        "repairadapt",
        "repairtrain",
        "repairstable",
    ]
    assert [phase["actor_lr"] for phase in phases] == [
        0.0,
        1.0e-5,
        2.0e-5,
        7.5e-6,
    ]
    assert [phase["adapter_lr_override"] for phase in phases] == [
        0.0,
        1.25e-6,
        2.5e-6,
        0.0,
    ]
    training = p4_contract.training_contract(
        {
            "enabled": True,
            "mode": "shadow",
            "schedule_enabled": False,
            "confirmation_s": 12.0,
            "terminal_penalty": -75.0,
        },
        PROFILE,
    )
    assert training["version"] == "p4_maze_instant_repair2h_v1"
    assert training["run_name"] == "p4maze2h-instant-repair-r1"
    assert training["target_effective_seconds"] == 7_200
    assert training["adapter_replay_policy"] == (
        "instant_compatible_current_only_parent_ratio_0"
    )
    assert p4_contract.command_contract(PROFILE)["version"] == (
        "p4_maze_instant_command_r4_inputfix"
    )
    reward = p4_contract.reward_contract(
        {
            "enabled": True,
            "mode": "shadow",
            "schedule_enabled": False,
            "confirmation_s": 12.0,
            "terminal_penalty": -75.0,
        },
        PROFILE,
    )
    assert reward["version"] == "p4_maze_reward_v5_instant_credit_repair"
    assert reward["profile_weights"]["command_rate"] == -0.003
    assert reward["profile_weights"]["tracking_error"] == -0.002
    assert reward["new_terms"]["maze_new_best_credit"] == {
        "weight_per_m": 1.0,
        "episode_cap": 6.0,
        "terminal": "retain_earned_credit_no_clawback",
    }
    maximum_timeout_return = 6.0 + 750 * p2_contract.TIME_COST_PER_TICK - 40.0
    assert maximum_timeout_return == -49.0


def test_repair_reward_weights_only_override_p4_profile():
    algorithm = object.__new__(AlgorithmP4NavPPO)
    algorithm.training_profile = PROFILE
    assert algorithm._command_rate_weight() == -0.003
    assert algorithm._tracking_error_weight() == -0.002
    algorithm.training_profile = "maze_instant_command_r4"
    assert algorithm._command_rate_weight() == p2_contract.COMMAND_RATE_WEIGHT
    assert algorithm._tracking_error_weight() == p2_contract.TRACKING_ERROR_WEIGHT


def test_repair_teacher_stale_and_near_goal_terms_are_finite_and_context_gated():
    command = torch.tensor(
        [[0.8, 0.2, 0.5], [0.8, 0.2, 0.0], [0.8, 0.2, 0.0]],
        dtype=torch.float32,
    )
    safe5 = torch.tensor(
        [[0.8, 0.9, 0.9, 0.9, 0.8]] * 3, dtype=torch.float32
    )
    result = p4_contract.instant_r4_teacher_guidance_loss(
        command,
        safe5,
        torch.tensor([[2.0, 0.0], [0.9, 0.0], [0.9, 0.0]]),
        torch.zeros(3),
        torch.zeros(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        goal_freshness=torch.tensor([0.0, 1.0, 1.0]),
        context_mask=torch.tensor([True, True, False]),
        min_valid_steps=1,
        instant_repair=True,
    )
    assert torch.isfinite(result["loss"])
    assert result["teacher_stale_goal_mask"].tolist() == [1.0, 0.0, 0.0]
    assert result["teacher_near_goal_mask"].tolist() == [0.0, 1.0, 0.0]


def test_repair_adapter_policy_routes_to_current_only_sampler(monkeypatch):
    buffer = object.__new__(P2ResponseAuxBuffer)
    buffer.replay_policy = "track_parent_75_25"
    buffer.enable_p4_current_only_replay()
    sentinel = object()
    monkeypatch.setattr(
        buffer,
        "_sample_p4_current_only",
        lambda **_kwargs: sentinel,
    )
    assert buffer.sample(batch_envs=4) is sentinel


def test_success_has_priority_over_failure_wall_stuck_and_timeout():
    class Manager:
        active_terms = ("goal_reached",)
        terminated = torch.tensor([True, True, False])
        time_outs = torch.tensor([True, True, True])

        @staticmethod
        def get_term(name):
            assert name == "goal_reached"
            return torch.tensor([True, False, False])

    env = SimpleNamespace(termination_manager=Manager())
    reason = _termination_reason_codes(
        env,
        torch.ones(3, dtype=torch.bool),
        wall_stuck=torch.tensor([True, True, True]),
    )
    assert reason.tolist() == [1.0, 2.0, 4.0]
