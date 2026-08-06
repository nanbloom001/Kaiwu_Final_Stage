"""Focused CPU checks for the P4 Maze instant-command R4 contract."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import torch

from agent_ppo.feature import p4_contract


R4_PROFILE = "maze_instant_command_r4"


def _stuck_reset() -> dict[str, float | bool | str]:
    return {
        "enabled": True,
        "mode": "active",
        "schedule_enabled": True,
        "confirmation_s": 10.0,
        "initial_confirmation_s": 12.0,
        "activation_delay_s": 1_800.0,
        "tighten_after_s": 7_200.0,
        "terminal_penalty": -75.0,
    }


def test_r4_config_and_command_contract_are_instant_only():
    path = Path(__file__).parents[1] / "conf" / "train_env_conf_track_p4_nav_ppo.toml"
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    stage = config["p4_nav_ppo"]

    assert config["env"] == {
        "num_envs": 128,
        "episode_length_s": 120.0,
        "task": "track",
    }
    assert config["terrain"]["num_cols"] == 20
    assert config["terrain"]["curriculum"] is False
    assert config["terrain"]["track"] == {
        "track_length": 1,
        "sub_terrains": ["open_entry_maze"],
        "num_parallel_tracks": 20,
    }
    assert stage["run_name"] == "p4maze8h-instant-r4"
    assert stage["parent_model_id"] == 1416926
    assert stage["parent_model_label"] == "p4maze8h10hz_1416926-mazefinal"
    assert stage["load_mode"] == "p4_maze_instant_r4_warm_start"
    assert stage["training_profile"] == R4_PROFILE
    assert stage["maze_training_branch"] == "instant_command_r4"
    assert stage["task_end_hours"] == pytest.approx(8.25)
    assert stage["target_effective_seconds"] == 28_800
    assert stage["command_transition_mode"] == "instant_hold_10hz"
    assert stage["hold_frames"] == 5
    assert "slew_rate" not in stage
    assert "slew_release_rate" not in stage

    command = p4_contract.command_contract(R4_PROFILE)
    assert command["version"] == "p4_maze_instant_command_r4"
    assert command["mapped_ranges"] == {
        "vx": [0.0, 1.0],
        "vy": [-0.30, 0.30],
        "wz": [-0.90, 0.90],
    }
    assert command["command_transition_mode"] == "instant_hold_10hz"
    assert command["hold_frames"] == 5
    assert command["policy_target_equals_exec"] == "at_10hz_tick_boundary"
    assert command["command_rewrites"] == {
        "slew": "disabled",
        "zero_cross_guard": "disabled",
        "reversal_guard": "disabled",
        "runtime_limiter": "disabled",
        "near_goal_rewrite": "disabled",
        "recovery_override": "disabled",
    }
    assert command["hard_action_ranges_only"] is True
    assert "slew_rate" not in command
    assert "slew_release_rate" not in command
    assert "slew_rate" in p4_contract.command_contract("maze_closed_loop_v3")


@pytest.mark.parametrize(
    ("seconds", "phase", "actor_lr", "critic_lr", "teacher", "anchor", "adapter_lr", "entropy"),
    (
        (0.0, "instantwarm", 0.0, 6.0e-5, 0.0, 0.0, 0.0, 0.004),
        (1_800.0, "instantadapt", 1.5e-5, 6.0e-5, 0.0125, 0.005, 5.0e-6, 0.004),
        (7_200.0, "instantcorrect", 1.0e-5, 4.0e-5, 0.0175, 0.010, 0.0, 0.0035),
        (10_800.0, "instantstable", 5.0e-6, 3.0e-5, 0.010, 0.0125, 0.0, 0.0035),
        (12_600.0, "instantfrozen", 0.0, 1.0e-5, 0.0, 0.0, 0.0, 0.0035),
    ),
)
def test_r4_schedule_has_explicit_anchor_and_adapter_controls(
    seconds: float,
    phase: str,
    actor_lr: float,
    critic_lr: float,
    teacher: float,
    anchor: float,
    adapter_lr: float,
    entropy: float,
):
    schedule = p4_contract.training_schedule(seconds, branch="instant_command_r4")
    assert schedule["phase"] == phase
    assert schedule["actor_lr"] == pytest.approx(actor_lr)
    assert schedule["critic_lr"] == pytest.approx(critic_lr)
    assert schedule["teacher_gradient_target_ratio"] == pytest.approx(teacher)
    assert schedule["anchor_target_ratio"] == pytest.approx(anchor)
    assert schedule["adapter_lr_override"] == pytest.approx(adapter_lr)
    assert schedule["entropy_coefficient"] == pytest.approx(entropy)
    assert schedule["navigation_multiplier"] == 0.0
    assert schedule["safety_head_multiplier"] == 0.0
    assert schedule["stuck_head_multiplier"] == 0.0
    assert schedule["critic_multiplier"] == 1.0
    if phase == "instantwarm":
        assert schedule["actor_multiplier"] == 0.0
        assert schedule["anchor_multiplier"] == 0.0
        assert schedule["adapter_multiplier"] == 0.0
    elif phase == "instantadapt":
        assert schedule["actor_multiplier"] == 1.0
        assert schedule["anchor_multiplier"] == 1.0
        assert schedule["adapter_multiplier"] == 1.0
    elif phase in {"instantcorrect", "instantstable"}:
        assert schedule["actor_multiplier"] == 1.0
        assert schedule["anchor_multiplier"] == 1.0
        assert schedule["adapter_multiplier"] == 0.0
    else:
        assert schedule["actor_multiplier"] == 0.0
        assert schedule["anchor_multiplier"] == 0.0
        assert schedule["adapter_multiplier"] == 0.0


def test_r4_reward_checkpoint_and_stuck_contract_are_separate_from_slew_profiles():
    stuck = _stuck_reset()
    reward = p4_contract.reward_contract(stuck, R4_PROFILE)
    terms = reward["new_terms"]
    assert reward["version"] == p4_contract.INSTANT_COMMAND_REWARD_CONTRACT_VERSION
    assert terms["success_impulse"] == 200.0
    assert terms["failure_impulse"] == -60.0
    assert terms["timeout_impulse"] == -40.0
    assert terms["confirmed_wall_stuck_reset"] == {
        "mode": "active",
        "terminal_penalty": -75.0,
        "semantics": "active_reason4_terminal",
        "schedule": {
            "shadow_until_s": 1_800.0,
            "initial_confirmation_s": 12.0,
            "tighten_after_s": 7_200.0,
            "tight_confirmation_s": 10.0,
        },
    }
    assert terms["closed_loop_collision"] == {
        "onset_base": -0.16,
        "onset_severity": -0.24,
        "persistent": -0.06,
    }
    assert terms["sustained_wall_stuck"] == {
        "grace_s": 0.8,
        "full_penalty_s": 2.0,
        "base": -0.010,
        "floor": -0.04,
    }
    assert terms["maze_new_best_credit"] == {
        "weight_per_m": 1.0,
        "episode_cap": 6.0,
        "terminal": "exact_clawback_on_failure_timeout_reason4",
    }
    assert terms["recovery_translation_teacher"]["status"] == "disabled_no_recovery_bonus"
    assert terms["yaw_cancellation"] == "disabled_no_reward"

    training = p4_contract.training_contract(stuck, R4_PROFILE)
    assert training["version"] == p4_contract.INSTANT_COMMAND_CHECKPOINT_CONTRACT_VERSION
    assert training["checkpoint_phase_labels"] == list(
        p4_contract.INSTANT_COMMAND_PHASE_LABELS
    )
    assert training["schedule_boundaries_seconds"] == [
        1_800.0,
        7_200.0,
        10_800.0,
        12_600.0,
        28_800.0,
    ]
    assert training["stuck_reset_contract_version"] == (
        p4_contract.INSTANT_COMMAND_STUCK_RESET_CONTRACT_VERSION
    )
    assert training["exact_resume"] == (
        "p4_maze_instant_command_r4_only_incompatible_with_slew_profiles"
    )
    assert training["exact_resume_incompatible_command_transition_modes"] == ["slew"]
    assert p4_contract.training_contract(
        training_profile="maze_closed_loop_v3"
    )["stuck_reset_contract_version"] == p4_contract.STUCK_RESET_CONTRACT_VERSION


def test_r4_teacher_masks_are_mutually_exclusive_and_choose_one_recovery_axis():
    policy = torch.tensor(
        (
            (0.30, 0.0, 0.0),
            (0.55, 0.0, 0.0),
            (0.50, 0.0, 0.0),
        )
    )
    safe5 = torch.tensor(
        (
            (0.70, 0.70, 0.90, 0.70, 0.70),
            (0.90, 0.10, 0.20, 0.10, 0.05),
            (0.20, 0.10, 0.10, 0.75, 0.95),
        )
    )
    masks = p4_contract.instant_r4_teacher_masks(
        policy,
        safe5,
        torch.ones(3, dtype=torch.bool),
        torch.tensor((False, False, True)),
    )
    partition = (
        masks["teacher_normal_mask"].int()
        + masks["teacher_edge_mask"].int()
        + masks["teacher_recovery_mask"].int()
    )
    assert partition.tolist() == [1, 1, 1]
    assert masks["teacher_normal_mask"].tolist() == [True, False, False]
    assert masks["teacher_edge_mask"].tolist() == [False, True, False]
    assert masks["teacher_recovery_wz_mask"].tolist() == [False, False, True]
    assert not masks["teacher_recovery_vy_mask"].any()

    result = p4_contract.teacher_guidance_loss(
        policy,
        safe5,
        torch.tensor(((1.0, 0.0),) * 3),
        torch.tensor((0.0, 0.0, 1.0)),
        torch.tensor((False, False, True)),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        min_valid_steps=3,
        instant_r4=True,
    )
    assert result["teacher_edge_mask"].tolist() == [0.0, 1.0, 0.0]
    assert result["teacher_recovery_wz_mask"].tolist() == [0.0, 0.0, 1.0]
    assert result["teacher_recovery_vy_mask"].sum().item() == 0.0
    assert result["teacher_goal_tie_mask"].sum().item() == 0.0
