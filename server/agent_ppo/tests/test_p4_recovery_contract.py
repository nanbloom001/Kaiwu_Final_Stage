#!/usr/bin/env python3
"""Focused regression tests for the P4 Maze closed-loop contract."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import pytest
import torch

from agent_ppo.feature import p2_contract, p4_contract
from agent_ppo.feature.p2_command_controller import P2CommandController


def test_eight_hour_closed_loop_schedule_and_config_are_aligned():
    config_path = Path(__file__).parents[1] / "conf" / "train_env_conf_track_p4_nav_ppo.toml"
    config = tomllib.loads(config_path.read_text())
    stage = config["p4_nav_ppo"]

    assert p4_contract.TARGET_EFFECTIVE_SECONDS == 28_800.0
    assert p4_contract.TRAINING_HOURS == 8.0
    legacy_training = p4_contract.training_contract(
        stage["stuck_reset"], "maze_closed_loop_v3"
    )
    legacy_command = p4_contract.command_contract("maze_closed_loop_v3")
    assert legacy_training["required_platform_wall_seconds"] == 29_700
    assert stage["task_end_hours"] == 8.25
    assert stage["parent_model_id"] == 1416926
    assert tuple(legacy_command["slew_rate"]) == p4_contract.P4_SLEW_RATE
    assert tuple(legacy_command["slew_release_rate"]) == p4_contract.P4_SLEW_RELEASE_RATE
    assert stage["stuck_reset"]["confirmation_s"] == 10.0
    assert stage["stuck_reset"]["schedule_enabled"] is True
    assert legacy_training["stuck_reset"]["schedule_enabled"] is True
    assert config["domain_rand"]["push_robots"] is False

    warm = p4_contract.training_schedule(0.0, branch="closed_loop_v3")
    middle = p4_contract.training_schedule(3_600.0, branch="closed_loop_v3")
    train = p4_contract.training_schedule(7_200.0, branch="closed_loop_v3")
    final = p4_contract.training_schedule(25_200.0, branch="closed_loop_v3")
    assert (warm["phase"], middle["phase"], train["phase"], final["phase"]) == (
        "loopwarm",
        "loopadapt",
        "looptrain",
        "loopstable",
    )
    assert warm["actor_multiplier"] == 0.0
    assert middle["actor_lr"] == pytest.approx(3.0e-5)
    assert train["actor_lr"] == pytest.approx(5.0e-5)
    assert final["actor_lr"] == pytest.approx(2.5e-5)
    assert warm["navigation_multiplier"] == 0.0
    assert warm["adapter_multiplier"] == 0.0
    assert warm["teacher_gradient_target_ratio"] == pytest.approx(0.0)
    assert p4_contract.training_schedule(4_500.0, branch="closed_loop_v3")[
        "teacher_gradient_target_ratio"
    ] == pytest.approx(0.005)
    assert train["teacher_gradient_target_ratio"] == pytest.approx(0.020)
    assert final["teacher_gradient_target_ratio"] == pytest.approx(0.010)
    assert config["terrain"]["track"]["sub_terrains"] == ["open_entry_maze"]
    assert config["terrain"]["track"]["track_length"] == 1
    assert config["env"]["episode_length_s"] == 120.0
    assert stage["stuck_reset"]["mode"] == "active"
    assert stage["stuck_reset"]["terminal_penalty"] == -75.0
    assert stage["camera_fault_course_enabled"] is False
    assert stage["goal_fault_course_enabled"] is False


def test_recovery_slew_increase_release_and_zero_crossing():
    controller = P2CommandController(
        1,
        "cpu",
        slew_rate=p4_contract.P4_SLEW_RATE,
        slew_release_rate=p4_contract.P4_SLEW_RELEASE_RATE,
    )
    controller.set_target(torch.tensor(((1.0, 1.0, 1.0),)))
    controller.step()
    assert torch.allclose(controller.exec_cmd, torch.tensor(((0.012, 0.012, 0.040))))

    controller.exec_cmd[:] = torch.tensor(((0.20, 0.20, 0.20)))
    controller.set_target(torch.tensor(((-1.0, -1.0, -1.0),)))
    controller.step()
    assert torch.allclose(controller.exec_cmd, torch.tensor(((0.176, 0.176, 0.120))))
    previous = controller.exec_cmd.clone()
    hit_zero = torch.zeros(3, dtype=torch.bool)
    entered_reverse_after_zero = torch.zeros(3, dtype=torch.bool)
    for _ in range(40):
        controller.step()
        current = controller.exec_cmd[0].clone()
        crossed_without_zero = (previous[0] > 0.0) & (current < 0.0)
        assert not bool(crossed_without_zero.any())
        hit_zero |= current == 0.0
        entered_reverse_after_zero |= hit_zero & (current < 0.0)
        previous = controller.exec_cmd.clone()
    assert bool(hit_zero.all())
    assert bool(entered_reverse_after_zero.all())


def test_translation_vector_limiter_scales_xy_and_releases_only_by_tick_budget():
    policy = torch.tensor(((1.0, 0.20, 0.70),))
    limited, diagnostics = p4_contract.translation_vector_limiter(
        policy, torch.ones(1), torch.ones(1)
    )
    assert diagnostics["translation_safety_alpha"].item() == pytest.approx(0.75)
    assert torch.allclose(limited, torch.tensor(((0.75, 0.15, 0.70))))

    released, diagnostics = p4_contract.translation_vector_limiter(
        policy, torch.zeros(1), diagnostics["translation_safety_alpha"]
    )
    assert diagnostics["translation_safety_alpha"].item() == pytest.approx(0.95)
    assert torch.allclose(released, torch.tensor(((0.95, 0.19, 0.70))))

    reset, diagnostics = p4_contract.translation_vector_limiter(
        policy, torch.zeros(1), torch.tensor((0.60,)), reset_mask=torch.ones(1, dtype=torch.bool)
    )
    assert diagnostics["translation_safety_alpha"].item() == pytest.approx(1.0)
    assert torch.allclose(reset, policy)


def test_maze_new_best_credit_is_non_repeatable_and_capped_without_clawback():
    reward, earned, delta = p4_contract.maze_new_best_credit(
        torch.tensor((5.0, 5.0, 5.0)),
        torch.tensor((4.5, 5.2, 2.0)),
        torch.tensor((0.0, 0.0, 5.0)),
    )
    assert torch.allclose(reward, torch.tensor((0.5, 0.0, 1.0)))
    assert torch.allclose(earned, torch.tensor((0.5, 0.0, 6.0)))
    assert torch.allclose(delta, torch.tensor((0.5, 0.0, 3.0)))

    repeated, repeated_earned, repeated_delta = p4_contract.maze_new_best_credit(
        torch.tensor((4.5,)), torch.tensor((4.5,)), torch.tensor((1.0,))
    )
    assert repeated.item() == 0.0
    assert repeated_earned.item() == pytest.approx(1.0)
    assert repeated_delta.item() == 0.0

    invalid, invalid_earned, invalid_delta = p4_contract.maze_new_best_credit(
        torch.tensor((4.5, float("nan"))),
        torch.tensor((float("inf"), 2.0)),
        torch.tensor((1.0, 1.0)),
    )
    assert torch.equal(invalid, torch.zeros(2))
    assert torch.equal(invalid_delta, torch.zeros(2))
    assert torch.allclose(invalid_earned, torch.tensor((1.0, 1.0)))


def test_maximum_credit_timeout_episode_remains_negative():
    ticks = int(120.0 / p4_contract.P4_NAV_DT_S)
    maximum_timeout_return = (
        p4_contract.MAZE_NEW_BEST_EPISODE_CAP
        + p4_contract.TIMEOUT_IMPULSE
        + ticks * p2_contract.TIME_COST_PER_TICK
    )
    assert maximum_timeout_return < 0.0


def test_maximum_credit_wall_reset_episode_remains_strictly_negative():
    maximum_reset_return = (
        p4_contract.MAZE_NEW_BEST_EPISODE_CAP
        + p4_contract.STUCK_RESET_TERMINAL_PENALTY
    )
    assert maximum_reset_return < p4_contract.TIMEOUT_IMPULSE < 0.0


def test_closed_loop_active_stuck_contract_is_reason4_and_worse_than_timeout():
    contract = p4_contract.reward_contract(
        {"mode": "active", "terminal_penalty": -75.0},
        "maze_closed_loop_v3",
    )
    reset = contract["new_terms"]["confirmed_wall_stuck_reset"]
    assert reset["mode"] == "active"
    assert reset["semantics"] == "active_reason4_terminal"
    assert reset["terminal_penalty"] < p4_contract.TIMEOUT_IMPULSE


def test_yaw_exit_response_is_small_signed_and_inactive_near_goal():
    safe3 = torch.tensor(((0.95, 0.20, 0.10),))
    target = torch.tensor(((0.40, 0.00, -0.05),))
    goal = torch.tensor(((2.0, 2.0),))
    penalty, diagnostics = p4_contract.yaw_exit_response_penalty(
        safe3,
        target,
        goal,
        torch.ones(1),
        torch.ones(1),
        torch.zeros(1, dtype=torch.bool),
    )
    assert diagnostics["yaw_exit_response_eligible"].item() == 1.0
    assert -0.006 <= penalty.item() < 0.0

    correct_target = target.clone()
    correct_target[:, 2] = 0.12
    correct, _ = p4_contract.yaw_exit_response_penalty(
        safe3,
        correct_target,
        goal,
        torch.ones(1),
        torch.ones(1),
        torch.zeros(1, dtype=torch.bool),
    )
    assert correct.item() == 0.0

    near_goal, _ = p4_contract.yaw_exit_response_penalty(
        safe3,
        target,
        torch.tensor(((0.8, 0.8),)),
        torch.ones(1),
        torch.ones(1),
        torch.zeros(1, dtype=torch.bool),
    )
    assert near_goal.item() == 0.0


def test_near_goal_capture_never_stops_before_platform_radius_and_stale_goal_disables_it():
    policy = torch.tensor(((0.80, 0.0, 0.55),))
    captured, diagnostics = p4_contract.near_goal_capture(
        policy,
        torch.tensor(((0.80, 0.0),)),
        torch.ones(1),
        torch.ones(1),
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
    )
    assert diagnostics["near_goal_capture_active"].item() == 1.0
    assert captured[0, 0].item() == pytest.approx(0.17)
    assert captured[0, 2].item() == policy[0, 2].item()

    stale, diagnostics = p4_contract.near_goal_capture(
        policy,
        torch.tensor(((0.80, 0.0),)),
        torch.zeros(1),
        torch.tensor((0.50,)),
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
    )
    assert diagnostics["near_goal_capture_active"].item() == 0.0
    assert torch.allclose(stale, torch.tensor(((0.40, 0.0, 0.55))))


def test_teacher_masks_and_tolerant_mean_loss_are_rollout_pure():
    count = 64
    safe3 = torch.tensor(((0.90, 0.40, 0.10),)).repeat(count, 1)
    masks = p4_contract.teacher_guidance_mask(
        alive=torch.ones(count, dtype=torch.bool),
        scanner_valid=torch.ones(count, dtype=torch.bool),
        mapping_valid=torch.ones(count, dtype=torch.bool),
        terminal=torch.zeros(count, dtype=torch.bool),
        reset=torch.zeros(count, dtype=torch.bool),
        push_grace=torch.zeros(count, dtype=torch.bool),
        episode_grace=torch.zeros(count, dtype=torch.bool),
        goal_freshness=torch.ones(count),
        safe3=safe3,
    )
    assert masks["teacher_guidance_eligible"].all()
    mean = torch.tensor(((0.06, 0.10, 0.12),)).repeat(count, 1)
    kwargs = dict(
        policy_mean_cmd3=mean,
        safe3=safe3,
        goal_xy_m=torch.tensor(((1.0, 1.0),)).repeat(count, 1),
        predictive_risk=torch.full((count,), 0.70),
        stuck_active=torch.zeros(count, dtype=torch.bool),
        teacher_mask=masks["teacher_guidance_eligible"],
        goal_mask=masks["teacher_guidance_goal_eligible"],
    )
    accepted = p4_contract.teacher_guidance_loss(**kwargs)
    assert accepted["loss"].item() == pytest.approx(0.0, abs=1.0e-7)

    rejected = p4_contract.teacher_guidance_loss(
        **{**kwargs, "policy_mean_cmd3": torch.tensor(((0.40, -0.30, -0.12),)).repeat(count, 1)}
    )
    assert rejected["direction"].item() > 0.0
    assert rejected["yaw"].item() > 0.0

    stale_masks = p4_contract.teacher_guidance_mask(
        alive=torch.ones(count, dtype=torch.bool),
        scanner_valid=torch.ones(count, dtype=torch.bool),
        mapping_valid=torch.ones(count, dtype=torch.bool),
        terminal=torch.zeros(count, dtype=torch.bool),
        reset=torch.zeros(count, dtype=torch.bool),
        push_grace=torch.zeros(count, dtype=torch.bool),
        episode_grace=torch.zeros(count, dtype=torch.bool),
        goal_freshness=torch.zeros(count),
        safe3=safe3,
    )
    assert stale_masks["teacher_guidance_eligible"].all()
    assert not stale_masks["teacher_guidance_goal_eligible"].any()
    too_small = p4_contract.teacher_guidance_loss(**kwargs, min_valid_steps=count + 1)
    assert too_small["loss"].item() == 0.0


def test_teacher_mask_uses_safe5_margin_for_closed_loop_instead_of_safe3_tie():
    common = dict(
        alive=torch.ones(1, dtype=torch.bool),
        scanner_valid=torch.ones(1, dtype=torch.bool),
        mapping_valid=torch.ones(1, dtype=torch.bool),
        terminal=torch.zeros(1, dtype=torch.bool),
        reset=torch.zeros(1, dtype=torch.bool),
        push_grace=torch.zeros(1, dtype=torch.bool),
        episode_grace=torch.zeros(1, dtype=torch.bool),
        goal_freshness=torch.ones(1),
        safe3=torch.tensor([[0.80, 0.79, 0.78]]),
    )
    masks = p4_contract.teacher_guidance_mask(**common)
    assert not masks["teacher_guidance_eligible"].item()
    closed = p4_contract.teacher_guidance_mask(
        **{
            **common,
            "safe3": torch.tensor([[0.80, 0.79, 0.78]]),
            "safe5": torch.tensor([[0.95, 0.30, 0.30, 0.30, 0.30]]),
        }
    )
    assert closed["teacher_guidance_eligible"].item()
    assert closed["teacher_best_safe"].item() == pytest.approx(0.95)


def test_closed_loop_goal_tie_break_remains_reachable_for_similarly_safe_exits():
    count = 64
    safe3 = torch.tensor([[0.90, 0.85, 0.10]]).repeat(count, 1)
    safe5 = torch.tensor([[0.90, 0.85, 0.10, 0.05, 0.05]]).repeat(count, 1)
    masks = p4_contract.teacher_guidance_mask(
        alive=torch.ones(count, dtype=torch.bool),
        scanner_valid=torch.ones(count, dtype=torch.bool),
        mapping_valid=torch.ones(count, dtype=torch.bool),
        terminal=torch.zeros(count, dtype=torch.bool),
        reset=torch.zeros(count, dtype=torch.bool),
        push_grace=torch.zeros(count, dtype=torch.bool),
        episode_grace=torch.zeros(count, dtype=torch.bool),
        goal_freshness=torch.ones(count),
        safe3=safe3,
        safe5=safe5,
    )
    assert masks["teacher_safe_margin"].max().item() < 0.10
    assert masks["teacher_guidance_eligible"].all()
    assert masks["teacher_guidance_goal_eligible"].all()
    guided = p4_contract.teacher_guidance_loss(
        policy_mean_cmd3=torch.tensor([[0.30, -0.20, -0.20]]).repeat(count, 1),
        safe3=safe5,
        goal_xy_m=torch.tensor([[1.0, 0.577]]).repeat(count, 1),
        predictive_risk=torch.zeros(count),
        stuck_active=torch.zeros(count, dtype=torch.bool),
        teacher_mask=masks["teacher_guidance_eligible"],
        goal_mask=masks["teacher_guidance_goal_eligible"],
        min_valid_steps=count,
        closed_loop_v3=True,
    )
    assert guided["loss"].item() > 0.0


def test_closed_loop_teacher_requires_yaw_toward_safe_detour_even_when_goal_disagrees():
    count = 64
    result = p4_contract.teacher_guidance_loss(
        policy_mean_cmd3=torch.tensor([[0.20, 0.35, -0.20]]).repeat(count, 1),
        safe3=torch.tensor([[0.95, 0.20, 0.10, 0.10, 0.05]]).repeat(count, 1),
        goal_xy_m=torch.tensor([[1.0, -1.0]]).repeat(count, 1),
        predictive_risk=torch.zeros(count),
        stuck_active=torch.zeros(count, dtype=torch.bool),
        teacher_mask=torch.ones(count, dtype=torch.bool),
        goal_mask=torch.ones(count, dtype=torch.bool),
        min_valid_steps=count,
        closed_loop_v3=True,
    )
    assert result["teacher_yaw_mask"].all()
    assert result["yaw"].item() > 0.0


def test_closed_loop_teacher_penalizes_motion_aimed_at_a_low_clearance_edge():
    count = 64
    # The left sector is safe, but the current straight-ahead travel direction
    # has poor clearance. The 35 degree normal direction tolerance accepts the
    # 30 degree difference, so this specifically exercises edge clearance.
    result = p4_contract.teacher_guidance_loss(
        policy_mean_cmd3=torch.tensor([[0.55, 0.0, 0.10]]).repeat(count, 1),
        safe3=torch.tensor([[0.40, 0.95, 0.30, 0.20, 0.20]]).repeat(count, 1),
        goal_xy_m=torch.tensor([[1.0, 0.0]]).repeat(count, 1),
        predictive_risk=torch.zeros(count),
        stuck_active=torch.zeros(count, dtype=torch.bool),
        teacher_mask=torch.ones(count, dtype=torch.bool),
        goal_mask=torch.zeros(count, dtype=torch.bool),
        min_valid_steps=count,
        closed_loop_v3=True,
    )
    assert result["direction"].item() == pytest.approx(0.0, abs=1.0e-7)
    assert result["edge"].item() > 0.0
    assert result["teacher_edge_active_share"].item() == pytest.approx(1.0)


def test_closed_loop_teacher_uses_lateral_translation_to_leave_confirmed_stuck_state():
    count = 64
    common = dict(
        safe3=torch.tensor([[0.95, 0.80, 0.20, 0.10, 0.05]]).repeat(count, 1),
        goal_xy_m=torch.tensor([[1.0, 0.0]]).repeat(count, 1),
        predictive_risk=torch.ones(count),
        stuck_active=torch.ones(count, dtype=torch.bool),
        teacher_mask=torch.ones(count, dtype=torch.bool),
        goal_mask=torch.zeros(count, dtype=torch.bool),
        min_valid_steps=count,
        closed_loop_v3=True,
    )
    trapped = p4_contract.teacher_guidance_loss(
        policy_mean_cmd3=torch.tensor([[0.50, 0.0, 0.0]]).repeat(count, 1),
        **common,
    )
    escaped = p4_contract.teacher_guidance_loss(
        policy_mean_cmd3=torch.tensor([[0.10, 0.15, 0.0]]).repeat(count, 1),
        **common,
    )
    assert trapped["teacher_recovery_active_share"].item() == pytest.approx(1.0)
    assert trapped["recovery"].item() > 0.0
    assert escaped["recovery"].item() == pytest.approx(0.0, abs=1.0e-7)


def test_closed_loop_stuck_and_collision_penalties_are_stronger_than_previous_contract():
    assert p4_contract.STUCK_SUSTAINED_BASE <= -0.010
    assert p4_contract.STUCK_SUSTAINED_FLOOR <= -0.05
    assert p4_contract.P4_BODY_COLLISION_ONSET_BASE <= -0.20
    assert p4_contract.P4_BODY_COLLISION_ONSET_SEVERITY <= -0.30
    assert p4_contract.P4_BODY_COLLISION_PERSISTENT <= -0.08


def test_teacher_loss_applies_recovery_weights_per_timestep():
    count = 64
    safe3 = torch.tensor(((0.90, 0.40, 0.10),)).repeat(count, 1)
    accepted = torch.tensor(((0.40, 0.20, 0.12),)).repeat(count // 2, 1)
    rejected = torch.tensor(((0.40, -0.30, -0.12),)).repeat(count // 2, 1)
    policy = torch.cat((accepted, rejected), dim=0)
    common = dict(
        policy_mean_cmd3=policy,
        safe3=safe3,
        goal_xy_m=torch.tensor(((1.0, 1.0),)).repeat(count, 1),
        predictive_risk=torch.full((count,), 0.70),
        stuck_active=torch.zeros(count, dtype=torch.bool),
        teacher_mask=torch.ones(count, dtype=torch.bool),
        goal_mask=torch.ones(count, dtype=torch.bool),
        min_valid_steps=count,
    )
    unweighted = p4_contract.teacher_guidance_loss(**common)
    sample_weight = torch.ones(count)
    sample_weight[count // 2 :] = 0.25
    weighted = p4_contract.teacher_guidance_loss(
        **common, sample_weight=sample_weight
    )
    assert 0.0 < weighted["loss"].item() < unweighted["loss"].item()


def test_safety_group_and_route_eligibility_include_new_recovery_boundaries():
    raw = torch.full((1,), -0.03)
    applied, scale = p4_contract.proportional_negative_cap(
        raw,
        raw,
        raw,
        goal_safe_raw=raw,
        yaw_exit_raw=raw,
    )
    assert sum(term.item() for term in applied) == pytest.approx(-0.07)
    assert scale.item() == pytest.approx(7.0 / 15.0)

    penalty, diagnostics = p4_contract.route_excess_penalty(
        torch.tensor((0.20,)),
        torch.tensor((2.0,)),
        torch.tensor((1.90,)),
        torch.zeros(1, dtype=torch.bool),
        recovery_active=torch.ones(1, dtype=torch.bool),
        goal_freshness=torch.ones(1),
    )
    assert penalty.item() == 0.0
    assert diagnostics["route_excess_eligible"].item() == 0.0
