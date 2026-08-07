"""Runtime checks for the P4 instant-command R4 training profile."""

from __future__ import annotations

from pathlib import Path
import copy
import tempfile

import pytest
import torch
from torch import nn

from agent_ppo.algorithm.algorithm_p4_nav_ppo import AlgorithmP4NavPPO
from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
from agent_ppo.feature import nav_contract, p2_contract, p4_contract
from agent_ppo.feature.p2_response_buffer import P2ResponseAuxBuffer
from agent_ppo.model.p2_high_level import (
    NavigationEncoder,
    NavigationSafetyHead,
    P2NavigationActor,
    P2NavigationCritic,
)
from agent_ppo.model.response_adapter import CommandResponseAdapter
from agent_ppo.model.vision_encoder import VisionEncoder


def _algorithm(profile: str = "maze_instant_command_r4") -> AlgorithmP4NavPPO:
    low_actor = nn.Sequential(
        nn.Linear(77, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 12),
    )
    return AlgorithmP4NavPPO(
        low_level_encoder=VisionEncoder(),
        low_level_actor=low_actor,
        navigation_encoder=NavigationEncoder(),
        safety_head=NavigationSafetyHead(),
        actor=P2NavigationActor(),
        critic=P2NavigationCritic(),
        response_adapter=CommandResponseAdapter(),
        response_buffer=P2ResponseAuxBuffer(1, "cpu"),
        num_envs=1,
        device="cpu",
        config={
            "p4_seed": 17,
            "num_learning_epochs": 4,
            "training_profile": profile,
            "maze_training_branch": (
                "instant_repair2h"
                if profile == "maze_instant_repair2h"
                else "instant_command_r4"
            ),
            "track_segment_labels": ["maze"],
            "command_transition_mode": "instant_hold_10hz",
            "camera_fault_course_enabled": False,
            "goal_fault_course_enabled": False,
            "stuck_reset": {
                "enabled": True,
                "mode": "shadow" if profile == "maze_instant_repair2h" else "active",
                "schedule_enabled": profile != "maze_instant_repair2h",
                "confirmation_s": 12.0 if profile == "maze_instant_repair2h" else 10.0,
                "initial_confirmation_s": 12.0,
                "activation_delay_s": 1_800.0,
                "tighten_after_s": 7_200.0,
                "terminal_penalty": -75.0,
            },
        },
    )


def _completed_response_record(buffer: P2ResponseAuxBuffer) -> dict[str, object]:
    return {
        "aux": torch.zeros(1, p2_contract.RESPONSE_AUX_DIM),
        "episode_start": torch.ones(1, dtype=torch.bool),
        "current_segment": torch.zeros(1),
        "velocity": torch.zeros(1, 3, 3),
        "pose": torch.zeros(1, 3),
        "stuck": torch.zeros(1, 1),
        "horizon_mask": torch.ones(1, 3, dtype=torch.bool),
        "pose_mask": torch.ones(1, 1, dtype=torch.bool),
        "record_contract": copy.deepcopy(buffer.current_record_contract),
    }


def _eval_algorithm(profile: str) -> AlgorithmP4NavPPO:
    low_actor = nn.Sequential(
        nn.Linear(77, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 12),
    )
    config = {
        "p4_seed": 17,
        "training_profile": profile,
        "track_segment_labels": (
            ["maze"]
            if profile in AlgorithmP4NavPPO.MAZE_PROFILES
            else list(p4_contract.FULL_TRACK_SEGMENT_LABELS)
        ),
    }
    if profile == "maze_instant_command_r4":
        config["maze_training_branch"] = "instant_command_r4"
    return AlgorithmP4NavPPO(
        low_level_encoder=VisionEncoder(),
        low_level_actor=low_actor,
        navigation_encoder=NavigationEncoder(),
        safety_head=None,
        actor=P2NavigationActor(),
        critic=None,
        response_adapter=CommandResponseAdapter(),
        response_buffer=None,
        num_envs=1,
        device="cpu",
        config=config,
        training=False,
    )


def _components() -> dict[str, torch.Tensor]:
    names = (
        "frame_safety",
        "frontier_shaping",
        "success",
        "failure",
        "timeout",
        "time",
        "crawl",
        "command_rate",
        "tracking",
        "gait_symmetry",
        "body_collision",
        "predictive_collision_risk",
        "missed_safe_direction",
        "frontier_stagnation",
    )
    return {name: torch.zeros(1) for name in names}


def _reward_context(
    *, start: float, end: float, terminal: bool = False, reason: int = 0
) -> dict[str, torch.Tensor]:
    return {
        "reward_exec_cmd": torch.zeros(1, 3),
        "reward_source_aux": torch.zeros(1, p2_contract.WORKER_AUX_DIM),
        "terminal": torch.tensor((terminal,)),
        "reason": torch.tensor((reason,)),
        "duration_frames": torch.tensor((p4_contract.P4_NAV_PERIOD_FRAMES,)),
        "start_goal_distance": torch.tensor((start,)),
        "end_goal_distance": torch.tensor((end,)),
        "path_length_m": torch.zeros(1),
    }


def test_r4_policy_target_is_the_exec_command_for_all_held_low_frames():
    algorithm = _algorithm()
    algorithm._delivered_depth = torch.ones(1, 180, 320, 1)
    normalized = torch.tensor(((0.4, -0.5, 0.8),))
    goal4 = torch.tensor(((0.2, 0.1, 0.3, 1.0),))
    target = algorithm._map_policy_target(
        normalized,
        torch.zeros(1, 3),
        goal4=goal4,
        aux=torch.zeros(1, p2_contract.RESPONSE_AUX_DIM),
    )
    expected = p4_contract.map_normalized_action(normalized, 1.0)
    assert torch.equal(target, expected)
    assert torch.equal(algorithm._last_limited_command, expected)
    algorithm.command.set_target(target)
    for _ in range(p4_contract.P4_NAV_PERIOD_FRAMES):
        algorithm.command.step()
        assert torch.equal(algorithm.command.exec_cmd, expected)

    reversed_target = target.clone()
    reversed_target[:, 1:] *= -1.0
    algorithm.command.set_target(reversed_target)
    assert torch.equal(algorithm.command.exec_cmd, reversed_target)


def test_r4_actor_capability_preserves_parent_rate_inputs_and_hard_mapper_range():
    algorithm = _algorithm()
    algorithm.effective_speed_cap.fill_(0.05)
    algorithm.safety_speed_cap.fill_(0.10)

    capability = algorithm._nav_capability(1)

    assert capability[0, 6].item() == pytest.approx(p4_contract.P4_MAX_VX)
    expected = torch.tensor(p4_contract.INSTANT_ACTOR_CAPABILITY_PROFILE15)
    assert torch.equal(capability[0, 9:15], expected[9:15])
    assert not torch.equal(
        capability[0, 9:12],
        torch.tensor(p4_contract.INSTANT_CAPABILITY_CHANGE_RATE),
    )


def test_r4_parent_anchor_rejects_unreachable_or_reversing_legacy_targets():
    previous = torch.tensor(
        (
            (0.40, 0.10, 0.20),
            (0.40, 0.10, 0.20),
            (0.40, 0.10, 0.20),
            (0.40, 0.10, 0.20),
        )
    )
    parent = torch.tensor(
        (
            (0.42, 0.12, 0.25),
            (0.50, 0.12, 0.25),
            (0.42, -0.10, 0.25),
            (0.38, 0.05, 0.00),
        )
    )

    reachable = p4_contract.instant_parent_anchor_reachable(parent, previous)

    assert reachable.tolist() == [True, False, False, True]


def test_r4_next_tick_anchors_from_the_command_that_was_actually_executed():
    algorithm = _algorithm()
    algorithm.command.exec_cmd.copy_(torch.tensor(((0.20, 0.0, 0.0),)))
    # Model a sampled target that was later rejected by the base finite-row
    # fallback and therefore never reached the controller.
    algorithm._last_policy_command.fill_(float("nan"))
    algorithm._delivered_depth = torch.ones(1, 180, 320, 1)

    normalized = torch.tensor(((0.0, 0.0, 0.0),))
    goal4 = torch.tensor(((1.0, 0.0, 0.5, 1.0),))
    algorithm._map_policy_target(normalized, None, goal4=goal4, aux={})

    assert torch.isnan(algorithm._previous_policy_command).all()
    assert torch.equal(
        algorithm._previous_exec_command,
        torch.tensor(((0.20, 0.0, 0.0),)),
    )
    parent_target = torch.tensor(((0.22, 0.0, 0.0),))
    assert p4_contract.instant_parent_anchor_reachable(
        parent_target,
        algorithm._previous_exec_command,
    ).item()


def test_slew_profile_keeps_policy_target_history_separate_from_exec_history():
    algorithm = _eval_algorithm("maze_closed_loop_v3")
    algorithm._delivered_depth = torch.ones(1, 180, 320, 1)
    algorithm.goal_belief.estimate[:] = torch.tensor(((4.0, 0.0),))
    algorithm.command.exec_cmd.copy_(torch.tensor(((0.20, 0.0, 0.0),)))
    algorithm._last_policy_command.copy_(torch.tensor(((0.70, 0.10, 0.20),)))

    algorithm._map_policy_target(
        torch.zeros(1, 3),
        None,
        goal4=torch.tensor(((4.0, 0.0, 0.5, 1.0),)),
        aux={},
    )

    assert torch.equal(
        algorithm._previous_policy_command,
        torch.tensor(((0.70, 0.10, 0.20),)),
    )
    assert torch.equal(
        algorithm._previous_exec_command,
        torch.tensor(((0.20, 0.0, 0.0),)),
    )


def test_r4_boundary_frame_low_policy_reads_new_command_without_one_frame_delay():
    algorithm = _algorithm()
    algorithm.training_enabled = False
    old_command = torch.tensor(((0.7, 0.2, 0.6),))
    new_command = torch.tensor(((0.3, -0.3, -0.9),))
    algorithm.command.set_target(old_command)
    observed_commands: list[torch.Tensor] = []

    algorithm._map_policy_target = (
        lambda _normalized, _target, **_kwargs: new_command.clone()
    )

    def capture_low_command(parts, _critic_obs):
        p0, p1 = nav_contract.POLICY_CMD_SLICE
        observed_commands.append(parts["proprio"][:, p0:p1].clone())
        return torch.zeros(1, 12), {}

    algorithm._low_level_frame = capture_low_command
    observation = torch.zeros(1, nav_contract.POLICY_OBS_DIM)
    observation[:, 305:] = 0.5
    wire = torch.zeros(1, p4_contract.P4_PRIVILEGED_WIRE_DIM)
    wire[:, p2_contract.CRITIC_OBS_DIM + 9] = 1.0

    for frame in range(p4_contract.P4_NAV_PERIOD_FRAMES):
        result, _, aux = algorithm.frame_begin(
            observation, wire, deterministic=True
        )
        assert result["is_tick"] is (frame == 0)
        assert torch.equal(aux[:, 0:3], new_command)
        assert torch.equal(aux[:, 3:6], new_command)

    assert len(observed_commands) == p4_contract.P4_NAV_PERIOD_FRAMES
    assert all(torch.equal(command, new_command) for command in observed_commands)


@pytest.mark.parametrize("reason, impulse", ((2, -60.0), (3, -40.0), (4, -75.0)))
def test_r4_terminal_exactly_claws_back_earned_maze_credit(
    reason: int, impulse: float
):
    algorithm = _algorithm()
    algorithm.pending_tick = {
        "target_cmd3": torch.zeros(1, 3),
        "safe3": torch.zeros(1, 3),
        "safety_valid": torch.zeros(1, 1),
    }
    earned = algorithm._override_reward_components(
        _components(), **_reward_context(start=5.0, end=4.0)
    )
    assert earned["frontier_shaping"].item() == pytest.approx(1.0)
    assert algorithm._maze_credit_earned.item() == pytest.approx(1.0)

    terminal = algorithm._override_reward_components(
        _components(),
        **_reward_context(start=4.0, end=4.0, terminal=True, reason=reason),
    )
    assert terminal["frontier_shaping"].item() == pytest.approx(-1.0)
    terminal_name = {2: "failure", 3: "timeout", 4: "stuck_reset"}[reason]
    assert terminal[terminal_name].item() == pytest.approx(impulse)
    assert algorithm._maze_credit_earned.item() == 0.0


def test_r4_exact_resume_restores_instant_contract_and_parent_anchor():
    algorithm = _algorithm()
    algorithm._initial_low_digest = algorithm._module_digest(
        (("vision", algorithm.low_level_encoder), ("actor", algorithm.low_level_actor))
    )
    algorithm.low_level_state_digest = algorithm._initial_low_digest
    algorithm._configure_adapter_contract()
    algorithm._freeze_parent_anchor_from_current_actor(source_sha256="a" * 64)
    algorithm.update_training_clocks(8_000.0)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.ckpt-instantcorrect-42.pkl"
        algorithm.save_training_bundle(str(path), platform_model_id="42")
        payload = torch.load(path, weights_only=False, map_location="cpu")
        command = payload["contracts"]["command"]
        assert command["command_transition_mode"] == "instant_hold_10hz"
        assert "slew_rate" not in command
        assert "parent_actor_anchor" in payload["modules"]["high_level"]

        resumed = _algorithm()
        mode = resumed.load_bundle(str(path), platform_model_id="42")

    assert mode == "p4_exact_resume_history_reset"
    assert resumed.command.command_transition_mode == "instant_hold_10hz"
    assert resumed.parent_anchor_source_sha256 == "a" * 64
    assert resumed.parent_anchor_digest == algorithm.parent_anchor_digest


def test_r4_eval_rejects_default_slew_runtime_and_accepts_matching_runtime():
    algorithm = _algorithm()
    algorithm._initial_low_digest = algorithm._module_digest(
        (("vision", algorithm.low_level_encoder), ("actor", algorithm.low_level_actor))
    )
    algorithm.low_level_state_digest = algorithm._initial_low_digest
    algorithm._configure_adapter_contract()
    algorithm._freeze_parent_anchor_from_current_actor(source_sha256="b" * 64)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.ckpt-instantadapt-42.pkl"
        algorithm.save_training_bundle(str(path), platform_model_id="42")

        with pytest.raises(ValueError, match="command contract"):
            _eval_algorithm("full_track").load_evaluation_bundle(
                str(path), platform_model_id="42"
            )

        mode = _eval_algorithm("maze_instant_command_r4").load_evaluation_bundle(
            str(path), platform_model_id="42"
        )

    assert mode == "evaluate_full_modules_only"


def _assert_nested_equal(left, right):
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right):
            _assert_nested_equal(lhs, rhs)
    else:
        assert left == right


def _seed_optimizer(optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = sum(
        (
            parameter.square().mean()
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ),
        torch.zeros(()),
    )
    assert loss.requires_grad
    loss.backward()
    optimizer.step()


class _FrozenPhaseRollout:
    num_ticks = 32
    sequence_length = 16

    @staticmethod
    def sequence_refs(_generator):
        return [0, 1]


def _assert_frozen_phase_updates_critic_only(algorithm):
    actor_parameters = [
        parameter.detach().clone()
        for group in algorithm.actor_optimizer.param_groups
        for parameter in group["params"]
    ]
    adapter_parameters = [
        parameter.detach().clone() for parameter in algorithm.response_adapter.parameters()
    ]
    actor_optimizer = copy.deepcopy(algorithm.actor_optimizer.state_dict())
    adapter_optimizer = copy.deepcopy(algorithm.response_optimizer.state_dict())
    critic_before = [parameter.detach().clone() for parameter in algorithm.critic.parameters()]

    algorithm.rollout = _FrozenPhaseRollout()
    algorithm.num_learning_epochs = 1
    algorithm.num_mini_batches = 1
    algorithm._rollout_advantage_stats = lambda: (torch.tensor(0.0), torch.tensor(1.0))
    algorithm._critic_sequence_batch = lambda refs: {
        "returns": torch.zeros(len(refs), algorithm.rollout.sequence_length, 1)
    }
    algorithm._critic_micro_loss = lambda _batch: sum(
        (parameter.square().mean() for parameter in algorithm.critic.parameters()),
        torch.zeros(()),
    )

    AlgorithmP2NavPPO._run_ppo_epochs(algorithm)
    adapter_metrics = algorithm._adapter_update()

    assert adapter_metrics["adapter_frozen"] == 1.0
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            actor_parameters,
            (
                parameter
                for group in algorithm.actor_optimizer.param_groups
                for parameter in group["params"]
            ),
        )
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(adapter_parameters, algorithm.response_adapter.parameters())
    )
    _assert_nested_equal(actor_optimizer, algorithm.actor_optimizer.state_dict())
    _assert_nested_equal(adapter_optimizer, algorithm.response_optimizer.state_dict())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, algorithm.critic.parameters())
    )


def test_r4_frozen_phase_preserves_actor_adapter_adam_before_and_after_resume():
    algorithm = _algorithm()
    algorithm._initial_low_digest = algorithm._module_digest(
        (("vision", algorithm.low_level_encoder), ("actor", algorithm.low_level_actor))
    )
    algorithm.low_level_state_digest = algorithm._initial_low_digest
    algorithm._configure_adapter_contract()
    algorithm._freeze_parent_anchor_from_current_actor(source_sha256="c" * 64)
    algorithm.update_training_clocks(1_800.0)
    _seed_optimizer(algorithm.actor_optimizer)
    _seed_optimizer(algorithm.response_optimizer)
    algorithm.update_training_clocks(12_600.0)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.ckpt-instantfrozen-42.pkl"
        algorithm.save_training_bundle(str(path), platform_model_id="42")
        resumed = _algorithm()
        assert resumed.load_bundle(str(path), platform_model_id="42") == (
            "p4_exact_resume_history_reset"
        )

    _assert_frozen_phase_updates_critic_only(algorithm)
    _assert_frozen_phase_updates_critic_only(resumed)


def test_repair_warm_start_drops_inherited_completed_adapter_records():
    parent = _algorithm()
    parent._initial_low_digest = parent._module_digest(
        (("vision", parent.low_level_encoder), ("actor", parent.low_level_actor))
    )
    parent.low_level_state_digest = parent._initial_low_digest
    parent._configure_adapter_contract()
    parent._freeze_parent_anchor_from_current_actor(source_sha256="d" * 64)
    parent.response_buffer._records.append(
        _completed_response_record(parent.response_buffer)
    )
    parent.response_buffer.total_sequences = 1

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.ckpt-instantfrozen-42.pkl"
        parent.save_training_bundle(str(path), platform_model_id="42")
        resumed = _algorithm("maze_instant_repair2h")
        mode = resumed.load_bundle(str(path), platform_model_id="42")

    assert mode == "p4_maze_instant_repair2h_warm_start"
    assert len(resumed.response_buffer._records) == 0
    assert len(resumed.response_buffer._parent_records) == 0
    assert resumed.response_buffer.total_sequences == 0
    assert resumed.response_buffer.replay_policy == "p4_current_only"


def test_repair_exact_resume_preserves_current_session_completed_records():
    algorithm = _algorithm("maze_instant_repair2h")
    algorithm._initial_low_digest = algorithm._module_digest(
        (("vision", algorithm.low_level_encoder), ("actor", algorithm.low_level_actor))
    )
    algorithm.low_level_state_digest = algorithm._initial_low_digest
    algorithm._configure_adapter_contract()
    algorithm._freeze_parent_anchor_from_current_actor(source_sha256="e" * 64)
    algorithm.response_buffer._records.append(
        _completed_response_record(algorithm.response_buffer)
    )
    algorithm.response_buffer.total_sequences = 1

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.ckpt-repairadapt-42.pkl"
        algorithm.save_training_bundle(str(path), platform_model_id="42")
        resumed = _algorithm("maze_instant_repair2h")
        mode = resumed.load_bundle(str(path), platform_model_id="42")

    assert mode == "p4_exact_resume_history_reset"
    assert len(resumed.response_buffer._records) == 1
    assert resumed.response_buffer.total_sequences == 1
    assert resumed.response_buffer.replay_policy == "p4_current_only"


def test_repair_teacher_runs_for_stale_context_without_direction_mask():
    algorithm = _algorithm("maze_instant_repair2h")
    algorithm.update_training_clocks(300.0)
    with torch.no_grad():
        algorithm.actor.mean_head.bias.copy_(torch.tensor((2.0, 1.0)))
        algorithm.actor.vy_mean_head.bias.fill_(1.0)
    ticks = 2
    batch = {
        "nav_feat": torch.zeros(ticks, 1, p2_contract.NAV_FEATURE_DIM),
        "nav_nonvisual": torch.zeros(ticks, 1, p2_contract.NAV_NONVISUAL_DIM),
        "response_profile": torch.zeros(ticks, 1, p2_contract.RESPONSE_PROFILE_DIM),
        "confidence": torch.ones(ticks, 1, 1),
        "pre_tanh_action": torch.zeros(ticks, 1, p2_contract.ACTION_DIM),
        "actor_hidden": (
            torch.zeros(algorithm.actor.num_layers, 1, algorithm.actor.hidden_dim),
            torch.zeros(algorithm.actor.num_layers, 1, algorithm.actor.hidden_dim),
        ),
        "reset_mask": torch.zeros(ticks, 1, dtype=torch.bool),
        "old_log_prob": torch.zeros(ticks, 1, 1),
        "advantages": torch.zeros(ticks, 1, 1),
        "valid_mask": torch.ones(ticks, 1, 1),
        "safety_target": torch.zeros(ticks, 1, 3),
        "safety_valid": torch.zeros(ticks, 1, 1),
        "camera_aux_mask": torch.zeros(ticks, 1, 3),
        "clean_action_mean": torch.zeros(ticks, 1, 3),
        "teacher_safe3": torch.ones(ticks, 1, 3),
        "teacher_safe5": torch.ones(ticks, 1, 5),
        "teacher_goal_xy": torch.tensor([2.0, 0.0]).repeat(ticks, 1, 1),
        "teacher_goal_freshness": torch.zeros(ticks, 1, 1),
        "teacher_predictive_risk": torch.zeros(ticks, 1, 1),
        "teacher_mask": torch.zeros(ticks, 1, 1),
        "teacher_context_mask": torch.ones(ticks, 1, 1),
        "teacher_goal_mask": torch.ones(ticks, 1, 1),
        "teacher_weight": torch.ones(ticks, 1, 1),
        "stuck_label": torch.zeros(ticks, 1, 1),
        "stuck_mask": torch.zeros(ticks, 1, 1),
        "parent_normalized_mean": torch.zeros(ticks, 1, 3),
        "parent_log_std": torch.zeros(ticks, 1, 3),
        "parent_anchor_mask": torch.zeros(ticks, 1, 1),
    }
    loss, metrics = algorithm._actor_micro_loss(batch)
    loss.backward()
    assert metrics["teacher_stale_goal_active_share"].item() == 1.0
    assert metrics["teacher_guidance_loss"].item() > 0.0
    assert any(parameter.grad is not None for parameter in algorithm.actor.parameters())


def test_repair_epoch_gate_counts_stale_context_without_direction_mask(monkeypatch):
    algorithm = _algorithm("maze_instant_repair2h")
    algorithm.update_training_clocks(300.0)
    algorithm.rollout.teacher_mask = torch.zeros(32, 4, 1)
    algorithm.rollout.teacher_context_mask = torch.ones(32, 4, 1)
    algorithm.rollout.valid_mask = torch.ones(32, 4, 1)
    algorithm.rollout.continuation_mask = torch.ones(32, 4, 1)
    algorithm.rollout.step = 32
    monkeypatch.setattr(
        AlgorithmP2NavPPO,
        "_run_ppo_epochs",
        lambda _self: {"updates": 0.0},
    )
    algorithm._run_ppo_epochs()
    assert algorithm._teacher_update_enabled

    algorithm.rollout.continuation_mask.zero_()
    algorithm._run_ppo_epochs()
    assert not algorithm._teacher_update_enabled
    algorithm.rollout.continuation_mask.fill_(1.0)
    algorithm.rollout.valid_mask.zero_()
    algorithm._run_ppo_epochs()
    assert not algorithm._teacher_update_enabled

    legacy = _algorithm()
    legacy.update_training_clocks(1_800.0)
    legacy.rollout.teacher_mask = torch.zeros(32, 4, 1)
    legacy.rollout.teacher_context_mask = torch.ones(32, 4, 1)
    legacy.rollout.valid_mask = torch.ones(32, 4, 1)
    legacy.rollout.continuation_mask = torch.ones(32, 4, 1)
    legacy.rollout.step = 32
    legacy._run_ppo_epochs()
    assert not legacy._teacher_update_enabled


def test_r4_actor_loss_and_gradient_ignore_invalid_reason_zero_row():
    algorithm = _algorithm()
    algorithm.update_training_clocks(1_800.0)
    algorithm.track_safety_enabled = False
    algorithm._actor_auxiliary_loss = lambda **kwargs: (
        kwargs["ppo_actor_loss"].new_zeros(()),
        {},
    )

    def batch(ticks: int, valid_mask: torch.Tensor):
        return {
            "nav_feat": torch.zeros(ticks, 1, p2_contract.NAV_FEATURE_DIM),
            "nav_nonvisual": torch.zeros(
                ticks, 1, p2_contract.NAV_NONVISUAL_DIM
            ),
            "response_profile": torch.zeros(
                ticks, 1, p2_contract.RESPONSE_PROFILE_DIM
            ),
            "confidence": torch.ones(ticks, 1, 1),
            "pre_tanh_action": torch.zeros(
                ticks, 1, p2_contract.ACTION_DIM
            ),
            "actor_hidden": (
                torch.zeros(algorithm.actor.num_layers, 1, algorithm.actor.hidden_dim),
                torch.zeros(algorithm.actor.num_layers, 1, algorithm.actor.hidden_dim),
            ),
            "reset_mask": torch.zeros(ticks, 1, dtype=torch.bool),
            "old_log_prob": torch.zeros(ticks, 1, 1),
            "advantages": torch.cat(
                (torch.ones(1, 1, 1), torch.full((max(ticks - 1, 0), 1, 1), 999.0)),
                dim=0,
            ),
            "valid_mask": valid_mask,
            "safety_target": torch.zeros(ticks, 1, 3),
            "safety_valid": torch.ones(ticks, 1, 1),
        }

    def loss_and_gradient(input_batch):
        algorithm.actor.zero_grad(set_to_none=True)
        loss, metrics = algorithm._actor_micro_loss(input_batch)
        loss.backward()
        gradient = torch.cat(
            tuple(
                parameter.grad.detach().reshape(-1)
                for parameter in algorithm.actor.parameters()
                if parameter.grad is not None
            )
        )
        return loss.detach(), metrics, gradient

    valid_only = batch(1, torch.ones(1, 1, 1))
    mixed = batch(2, torch.tensor([[[1.0]], [[0.0]]]))
    valid_loss, valid_metrics, valid_gradient = loss_and_gradient(valid_only)
    mixed_loss, mixed_metrics, mixed_gradient = loss_and_gradient(mixed)

    assert torch.allclose(mixed_loss, valid_loss, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(mixed_gradient, valid_gradient, atol=1.0e-7, rtol=0.0)
    for name in ("surrogate_loss", "entropy", "approx_kl", "clip_fraction"):
        assert torch.allclose(
            mixed_metrics[name], valid_metrics[name], atol=1.0e-7, rtol=0.0
        )


def test_r4_real_safety_and_auxiliary_metrics_ignore_invalid_reason_zero_row():
    algorithm = _algorithm()
    algorithm.update_training_clocks(1_800.0)
    algorithm._auxiliary_coefficients.update(camera=0.10, teacher=0.10)

    def batch(ticks: int, valid_mask: torch.Tensor):
        camera_mask = torch.zeros(ticks, 1, 3)
        camera_mask[0, 0, 0] = 1.0
        if ticks > 1:
            camera_mask[1, 0, 2] = 1.0
        return {
            "nav_feat": torch.zeros(ticks, 1, p2_contract.NAV_FEATURE_DIM),
            "nav_nonvisual": torch.zeros(
                ticks, 1, p2_contract.NAV_NONVISUAL_DIM
            ),
            "response_profile": torch.zeros(
                ticks, 1, p2_contract.RESPONSE_PROFILE_DIM
            ),
            "confidence": torch.ones(ticks, 1, 1),
            "pre_tanh_action": torch.zeros(ticks, 1, p2_contract.ACTION_DIM),
            "actor_hidden": (
                torch.zeros(algorithm.actor.num_layers, 1, algorithm.actor.hidden_dim),
                torch.zeros(algorithm.actor.num_layers, 1, algorithm.actor.hidden_dim),
            ),
            "reset_mask": torch.zeros(ticks, 1, dtype=torch.bool),
            "old_log_prob": torch.zeros(ticks, 1, 1),
            "advantages": torch.cat(
                (torch.ones(1, 1, 1), torch.full((max(ticks - 1, 0), 1, 1), 999.0)),
                dim=0,
            ),
            "valid_mask": valid_mask,
            "safety_target": torch.tensor([1.0, 0.0, 0.0]).repeat(ticks, 1, 1),
            "safety_valid": torch.ones(ticks, 1, 1),
            "camera_aux_mask": camera_mask,
            "clean_action_mean": torch.zeros(ticks, 1, 3),
            "teacher_safe3": torch.tensor([1.0, 0.0, 0.0]).repeat(ticks, 1, 1),
            "teacher_safe5": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]).repeat(
                ticks, 1, 1
            ),
            "teacher_goal_xy": torch.tensor([1.0, 0.0]).repeat(ticks, 1, 1),
            "teacher_predictive_risk": torch.ones(ticks, 1, 1),
            "teacher_mask": torch.ones(ticks, 1, 1),
            "teacher_goal_mask": torch.ones(ticks, 1, 1),
            "teacher_weight": torch.ones(ticks, 1, 1),
            "stuck_label": torch.zeros(ticks, 1, 1),
            "stuck_mask": torch.zeros(ticks, 1, 1),
        }

    valid_loss, valid_metrics = algorithm._actor_micro_loss(
        batch(1, torch.ones(1, 1, 1))
    )
    mixed_loss, mixed_metrics = algorithm._actor_micro_loss(
        batch(2, torch.tensor([[[1.0]], [[0.0]]]))
    )

    assert torch.allclose(mixed_loss, valid_loss, atol=1.0e-7, rtol=0.0)
    for name in (
        "safety_bce",
        "scanner_valid_share",
        "safety_head_risk_left",
        "safety_head_risk_center",
        "safety_head_risk_right",
        "camera_delay_only_share",
        "camera_fault_only_share",
        "camera_fault_delay_overlap_share",
        "teacher_edge_active_share",
        "teacher_recovery_active_share",
    ):
        assert torch.allclose(
            mixed_metrics[name], valid_metrics[name], atol=1.0e-7, rtol=0.0
        ), name
    assert mixed_metrics["camera_delay_only_share"].item() == pytest.approx(1.0)
    assert mixed_metrics["camera_fault_delay_overlap_share"].item() == 0.0
    assert mixed_metrics["teacher_edge_active_share"].item() == pytest.approx(1.0)
