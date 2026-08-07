"""Focused P4 recovery-auxiliary transport and gradient-bound regressions."""

import tempfile
from pathlib import Path

import torch
from torch import nn
import pytest

from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
from agent_ppo.algorithm.algorithm_p4_nav_ppo import AlgorithmP4NavPPO
from agent_ppo.feature import p2_contract, p4_contract
from agent_ppo.feature.p2_response_buffer import P2ResponseAuxBuffer
from agent_ppo.feature.p2_rollout import P2RolloutStorage
from agent_ppo.model.p2_high_level import (
    NavigationEncoder,
    NavigationSafetyHead,
    P2NavigationActor,
    P2NavigationCritic,
    assemble_actor_input,
)
from agent_ppo.model.response_adapter import CommandResponseAdapter
from agent_ppo.model.vision_encoder import VisionEncoder


def _algorithm(*, training=True, config=None):
    low_actor = nn.Sequential(
        nn.Linear(77, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12),
    )
    runtime_config = {"p4_seed": 71, "num_learning_epochs": 1}
    runtime_config.update(config or {})
    return AlgorithmP4NavPPO(
        low_level_encoder=VisionEncoder(),
        low_level_actor=low_actor,
        navigation_encoder=NavigationEncoder(),
        safety_head=NavigationSafetyHead() if training else None,
        actor=P2NavigationActor(),
        critic=P2NavigationCritic() if training else None,
        response_adapter=CommandResponseAdapter(),
        response_buffer=P2ResponseAuxBuffer(1, "cpu") if training else None,
        num_envs=1,
        device="cpu",
        config=runtime_config,
        training=training,
    )


def _transition(storage: P2RolloutStorage, *, marker: float):
    count = storage.num_envs
    hidden = (
        torch.zeros(2, count, 64),
        torch.zeros(2, count, 64),
    )
    observation = (
        {"depth": torch.zeros(count, p2_contract.DEPTH_DIM)}
        if storage.store_depth
        else {"nav_feat": torch.zeros(count, p2_contract.NAV_FEATURE_DIM)}
    )
    storage.add(
        **observation,
        nav_nonvisual=torch.zeros(count, p2_contract.NAV_NONVISUAL_DIM),
        response_profile=torch.zeros(count, p2_contract.RESPONSE_PROFILE_DIM),
        confidence=torch.ones(count, 1),
        safety_target=torch.zeros(count, 3),
        safety_valid=torch.ones(count, 1),
        critic_input=torch.zeros(count, p2_contract.CRITIC_INPUT_DIM),
        pre_tanh_action=torch.zeros(count, 3),
        old_log_prob=torch.zeros(count, 1),
        old_value=torch.zeros(count, 1),
        reward=torch.zeros(count, 1),
        duration_frames=torch.ones(count, 1, dtype=torch.long),
        bootstrap_value=torch.zeros(count, 1),
        bootstrap_mask=torch.ones(count, 1),
        continuation_mask=torch.ones(count, 1),
        reset_mask=torch.zeros(count, dtype=torch.bool),
        actor_hidden=hidden,
        critic_hidden=hidden,
        teacher_safe3=torch.full((count, 3), marker),
        teacher_goal_xy=torch.full((count, 2), marker),
        teacher_predictive_risk=torch.full((count, 1), marker),
        teacher_mask=torch.ones(count, 1),
        teacher_goal_mask=torch.ones(count, 1),
        teacher_weight=torch.full((count, 1), marker),
        stuck_label=torch.ones(count, 1),
        stuck_mask=torch.ones(count, 1),
        mirror_eligible=torch.ones(count, 1),
    )


def test_recovery_targets_remain_aligned_with_tbptt_storage():
    storage = P2RolloutStorage(
        1, num_ticks=16, sequence_length=16, store_depth=True
    )
    for tick in range(16):
        _transition(storage, marker=float(tick))
    ref = storage.sequence_refs(torch.Generator().manual_seed(4))[0]
    assert ref.start == 0
    assert storage.teacher_safe3[:, ref.env, 0].tolist() == list(range(16))
    assert storage.teacher_goal_xy[:, ref.env, 0].tolist() == list(range(16))
    assert storage.teacher_weight[:, ref.env, 0].tolist() == list(range(16))
    assert storage.stuck_label[:, ref.env, 0].all()
    assert storage.mirror_eligible[:, ref.env, 0].all()


def test_episode_aligned_mirror_refs_start_at_arbitrary_resets():
    storage = P2RolloutStorage(
        2, num_ticks=32, sequence_length=16, store_depth=True
    )
    storage.step = 32
    storage.reset_mask.zero_()
    storage.mirror_eligible.fill_(1.0)
    storage.reset_mask[3, 0] = True
    storage.reset_mask[9, 1] = True
    refs = storage.episode_aligned_refs(eligibility=storage.mirror_eligible)
    assert {(ref.start, ref.env) for ref in refs} == {(3, 0), (9, 1)}

    storage.reset_mask[12, 0] = True
    refs = storage.episode_aligned_refs(eligibility=storage.mirror_eligible)
    assert {(ref.start, ref.env) for ref in refs} == {(12, 0), (9, 1)}


def test_mirror_schedule_reaches_rollout_level_target_share():
    algorithm = _algorithm(training=True)
    algorithm.num_learning_epochs = 4
    algorithm.rollout.step = algorithm.rollout.num_ticks
    algorithm.rollout.reset_mask.zero_()
    algorithm.rollout.reset_mask[3, 0] = True
    algorithm.rollout.mirror_eligible.fill_(1.0)
    algorithm._prepare_mirror_batch_schedule()
    scheduled = sum(ref is not None for ref in algorithm._mirror_batch_schedule)
    assert algorithm._mirror_aux_eligible_sequence_count == 1
    assert scheduled == 1
    assert algorithm._mirror_aux_scheduled_sequence_share == 0.125
    scheduled_ref = next(
        ref for ref in algorithm._mirror_batch_schedule if ref is not None
    )
    assert (scheduled_ref.start, scheduled_ref.env) == (3, 0)


def test_recovery_label_requires_rollout_motion_intent(monkeypatch):
    algorithm = _algorithm(training=True)

    def fake_parent_frame_begin(self, _obs, _wire, *, deterministic=False):
        del deterministic
        self.pending_tick = {
            "reset_mask": torch.zeros(1, dtype=torch.bool),
            "safe3": torch.tensor([[0.90, 0.10, 0.10]]),
            "safety_valid": torch.ones(1, 1),
            "target_cmd3": torch.zeros(1, 3),
            "predictive_collision_risk": torch.zeros(1),
        }
        return {"is_tick": True}, torch.zeros(1, 323), torch.zeros(1, 32)

    monkeypatch.setattr(AlgorithmP2NavPPO, "frame_begin", fake_parent_frame_begin)
    critic_obs = torch.zeros(1, p2_contract.CRITIC_OBS_DIM)
    critic_obs[:, 319] = 1.0
    monkeypatch.setattr(algorithm, "_split_transport", lambda _wire: (critic_obs, None))
    tail = algorithm._p4_worker_extra
    tail[:, p4_contract.STUCK_MAPPING_VALID_INDEX] = 1.0
    tail[:, p4_contract.STUCK_WALL_EVIDENCE_INDEX] = 1.0
    tail[:, p4_contract.STUCK_CANDIDATE_INDEX] = 1.0
    tail[:, p4_contract.STUCK_DURATION_S_INDEX] = 2.5
    algorithm.goal_belief.estimate[:] = torch.tensor([[1.0, 0.0]])
    algorithm.frame_begin(torch.zeros(1, 1), torch.zeros(1, 1))
    assert algorithm.pending_tick["stuck_mask"].item() == 0.0
    assert algorithm.pending_tick["stuck_label"].item() == 0.0

    algorithm.command.exec_cmd[:, 0] = 0.20
    algorithm.frame_begin(torch.zeros(1, 1), torch.zeros(1, 1))
    assert algorithm.pending_tick["stuck_mask"].item() == 1.0
    assert algorithm.pending_tick["stuck_label"].item() == 1.0


def test_auxiliary_gradients_are_actor_only_and_hard_capped():
    algorithm = _algorithm(
        training=True,
        config={
            "training_profile": "maze_credit_repair",
            "maze_training_branch": "credit_repair",
            "track_segment_labels": ["maze"],
            "camera_fault_course_enabled": False,
            "goal_fault_course_enabled": False,
        },
    )
    algorithm._apply_training_schedule(7_200.0)
    timesteps, batch_size = 16, 4  # exactly the 64 teacher-step activation floor
    inputs = torch.randn(timesteps, batch_size, p2_contract.ACTOR_INPUT_DIM)
    hidden = (
        torch.zeros(2, batch_size, 64),
        torch.zeros(2, batch_size, 64),
    )
    pre_tanh = torch.zeros(timesteps, batch_size, 3)
    _, _, mean, _, _, features = algorithm.actor.evaluate_actions(
        inputs, pre_tanh, hidden, torch.zeros(timesteps, batch_size, dtype=torch.bool),
        return_features=True,
    )
    batch = {
        "camera_aux_mask": torch.zeros(timesteps, batch_size, 3),
        "clean_action_mean": torch.zeros(timesteps, batch_size, 3),
        "teacher_safe3": torch.tensor([0.10, 0.90, 0.10]).repeat(timesteps, batch_size, 1),
        "teacher_safe5": torch.tensor([0.90, 0.40, 0.10, 0.10, 0.10]).repeat(timesteps, batch_size, 1),
        "teacher_goal_xy": torch.tensor([1.0, 0.2]).repeat(timesteps, batch_size, 1),
        "teacher_predictive_risk": torch.full((timesteps, batch_size, 1), 0.8),
        "teacher_mask": torch.ones(timesteps, batch_size, 1),
        "teacher_goal_mask": torch.ones(timesteps, batch_size, 1),
        "teacher_weight": torch.ones(timesteps, batch_size, 1),
        "stuck_label": torch.arange(timesteps * batch_size).reshape(timesteps, batch_size, 1).remainder(2).float(),
        "stuck_mask": torch.ones(timesteps, batch_size, 1),
    }
    ppo_loss = mean.square().mean()
    loss, metrics = algorithm._actor_auxiliary_loss(
        normalized_mean=torch.tanh(mean), actor_features=features,
        batch=batch, ppo_actor_loss=ppo_loss,
    )
    loss.backward()
    assert metrics["teacher_guidance_valid_steps"].item() == 64.0
    assert metrics["auxiliary_gradient_ratio"].item() <= 0.050001
    assert any(parameter.grad is not None for parameter in algorithm.actor.parameters())
    assert any(parameter.grad is not None for parameter in algorithm.stuck_head.parameters())
    assert all(parameter.grad is None for parameter in algorithm.navigation_encoder.parameters())


def test_camera_auxiliary_still_updates_navigation_encoder():
    algorithm = _algorithm(training=True)
    timesteps, batch_size = 16, 4
    depth = torch.rand(
        timesteps * batch_size,
        p2_contract.DEPTH_HEIGHT,
        p2_contract.DEPTH_WIDTH,
        1,
    )
    nav_feat = algorithm.navigation_encoder(depth).reshape(timesteps, batch_size, 32)
    actor_input = assemble_actor_input(
        nav_feat,
        torch.zeros(timesteps, batch_size, p2_contract.NAV_NONVISUAL_DIM),
        torch.zeros(timesteps, batch_size, p2_contract.RESPONSE_PROFILE_DIM),
        torch.ones(timesteps, batch_size, 1),
    )
    hidden = (
        torch.zeros(2, batch_size, 64),
        torch.zeros(2, batch_size, 64),
    )
    pre_tanh = torch.zeros(timesteps, batch_size, 3)
    _, _, mean, _, _, features = algorithm.actor.evaluate_actions(
        actor_input,
        pre_tanh,
        hidden,
        torch.zeros(timesteps, batch_size, dtype=torch.bool),
        return_features=True,
    )
    batch = {
        "camera_aux_mask": torch.ones(timesteps, batch_size, 3),
        "clean_action_mean": torch.zeros(timesteps, batch_size, 3),
        "teacher_safe3": torch.zeros(timesteps, batch_size, 3),
        "teacher_goal_xy": torch.zeros(timesteps, batch_size, 2),
        "teacher_predictive_risk": torch.zeros(timesteps, batch_size, 1),
        "teacher_mask": torch.zeros(timesteps, batch_size, 1),
        "teacher_goal_mask": torch.zeros(timesteps, batch_size, 1),
        "teacher_weight": torch.ones(timesteps, batch_size, 1),
        "stuck_label": torch.zeros(timesteps, batch_size, 1),
        "stuck_mask": torch.zeros(timesteps, batch_size, 1),
    }
    ppo_loss = mean.square().mean()
    loss, _metrics = algorithm._actor_auxiliary_loss(
        normalized_mean=torch.tanh(mean),
        actor_normalized_mean=torch.tanh(mean.detach()),
        actor_features=features,
        batch=batch,
        ppo_actor_loss=ppo_loss,
    )
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in algorithm.navigation_encoder.parameters()
    )


def test_teacher_activation_floor_is_rollout_wide(monkeypatch):
    algorithm = _algorithm(training=True)
    algorithm._apply_training_schedule(1_800.0)
    algorithm.rollout.teacher_mask = torch.zeros(32, 4, 1)
    algorithm.rollout.teacher_mask[:16] = 1.0
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
    algorithm.rollout.teacher_mask[15, 3] = 0.0
    algorithm._run_ppo_epochs()
    assert not algorithm._teacher_update_enabled


def test_creditwarm_valid_stuck_batch_updates_head_without_actor():
    algorithm = _algorithm(
        training=True,
        config={
            "training_profile": "maze_credit_repair",
            "maze_training_branch": "credit_repair",
            "track_segment_labels": ["maze"],
            "camera_fault_course_enabled": False,
            "goal_fault_course_enabled": False,
        },
    )
    algorithm._apply_training_schedule(0.0)
    timesteps = 16
    normalized = torch.zeros(timesteps, 1, 3)
    batch = {
        "camera_aux_mask": torch.zeros(timesteps, 1, 3),
        "clean_action_mean": torch.zeros(timesteps, 1, 3),
        "teacher_mask": torch.zeros(timesteps, 1, 1),
        "teacher_goal_mask": torch.zeros(timesteps, 1, 1),
        "teacher_safe3": torch.zeros(timesteps, 1, 3),
        "teacher_goal_xy": torch.zeros(timesteps, 1, 2),
        "teacher_predictive_risk": torch.zeros(timesteps, 1, 1),
        "teacher_weight": torch.ones(timesteps, 1, 1),
        "stuck_mask": torch.ones(timesteps, 1, 1),
        "stuck_label": torch.cat(
            (torch.zeros(timesteps // 2), torch.ones(timesteps // 2))
        ).reshape(timesteps, 1, 1),
    }
    loss, _metrics = algorithm._actor_auxiliary_loss(
        normalized_mean=normalized,
        actor_normalized_mean=normalized,
        actor_features=torch.randn(timesteps, 1, algorithm.actor.hidden_dim),
        batch=batch,
        ppo_actor_loss=torch.zeros(()),
    )
    assert loss.requires_grad
    loss.backward()
    actor_before = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
    assert any(parameter.grad is not None for parameter in algorithm.stuck_head.parameters())
    assert all(parameter.grad is None for parameter in algorithm.actor.parameters())
    for before, parameter in zip(actor_before, algorithm.actor.parameters()):
        torch.testing.assert_close(parameter, before)
        assert parameter not in algorithm.actor_optimizer.state


def test_closed_loop_profile_keeps_stuck_head_diagnostic_only():
    algorithm = _algorithm(
        training=True,
        config={
            "training_profile": "maze_closed_loop_v3",
            "maze_training_branch": "closed_loop_v3",
            "track_segment_labels": ["maze"],
            "camera_fault_course_enabled": False,
            "goal_fault_course_enabled": False,
        },
    )
    algorithm._apply_training_schedule(7_200.0)
    timesteps = 16
    normalized = torch.zeros(timesteps, 1, 3)
    batch = {
        "camera_aux_mask": torch.zeros(timesteps, 1, 3),
        "clean_action_mean": torch.zeros(timesteps, 1, 3),
        "teacher_mask": torch.zeros(timesteps, 1, 1),
        "teacher_goal_mask": torch.zeros(timesteps, 1, 1),
        "teacher_safe3": torch.zeros(timesteps, 1, 3),
        "teacher_safe5": torch.zeros(timesteps, 1, 5),
        "teacher_goal_xy": torch.zeros(timesteps, 1, 2),
        "teacher_predictive_risk": torch.zeros(timesteps, 1, 1),
        "teacher_weight": torch.ones(timesteps, 1, 1),
        "stuck_mask": torch.ones(timesteps, 1, 1),
        "stuck_label": torch.cat(
            (torch.zeros(timesteps // 2), torch.ones(timesteps // 2))
        ).reshape(timesteps, 1, 1),
    }
    loss, _metrics = algorithm._actor_auxiliary_loss(
        normalized_mean=normalized,
        actor_normalized_mean=normalized,
        actor_features=torch.randn(timesteps, 1, algorithm.actor.hidden_dim),
        batch=batch,
        ppo_actor_loss=torch.zeros(()),
    )
    assert not loss.requires_grad
    assert all(parameter.grad is None for parameter in algorithm.stuck_head.parameters())
    assert "stuck" not in algorithm._auxiliary_coefficients


def test_creditwarm_empty_stuck_batch_skips_actor_backward_and_updates_critic():
    algorithm = _algorithm(
        training=True,
        config={
            "training_profile": "maze_credit_repair",
            "maze_training_branch": "credit_repair",
            "track_segment_labels": ["maze"],
            "camera_fault_course_enabled": False,
            "goal_fault_course_enabled": False,
        },
    )
    algorithm._apply_training_schedule(0.0)
    algorithm.rollout = P2RolloutStorage(
        1,
        num_ticks=32,
        sequence_length=16,
        store_depth=False,
        pin_memory=False,
    )
    for tick in range(32):
        _transition(algorithm.rollout, marker=float(tick))
    algorithm.rollout.safety_valid.zero_()
    algorithm.rollout.teacher_mask.zero_()
    algorithm.rollout.teacher_goal_mask.zero_()
    algorithm.rollout.stuck_mask.zero_()
    algorithm.rollout.mirror_eligible.zero_()
    algorithm.rollout.compute_returns()

    actor_steps_before = algorithm.actor_gradient_steps
    critic_steps_before = algorithm.critic_gradient_steps
    actor_state_before = {
        parameter: {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in state.items()
        }
        for parameter, state in algorithm.actor_optimizer.state.items()
    }

    metrics = algorithm._run_ppo_epochs()

    assert metrics["updates"] > 0.0
    assert algorithm.actor_gradient_steps == actor_steps_before
    assert algorithm.critic_gradient_steps > critic_steps_before
    assert algorithm.skipped_nonfinite == 0
    assert set(algorithm.actor_optimizer.state) == set(actor_state_before)
    for parameter, state in actor_state_before.items():
        for key, value in state.items():
            actual = algorithm.actor_optimizer.state[parameter][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(actual, value)
            else:
                assert actual == value


def test_detached_teacher_loss_is_skipped_instead_of_crashing_gradient_calibration(
    monkeypatch,
):
    algorithm = _algorithm(
        training=True,
        config={
            "training_profile": "maze_closed_loop_v3",
            "maze_training_branch": "closed_loop_v3",
            "track_segment_labels": ["maze"],
            "camera_fault_course_enabled": False,
            "goal_fault_course_enabled": False,
        },
    )
    algorithm._apply_training_schedule(3_600.0)
    steps = 64
    mean = torch.zeros(steps, 3, requires_grad=True)
    detached = torch.tensor(2.5)
    monkeypatch.setattr(
        p4_contract,
        "teacher_guidance_loss",
        lambda *args, **kwargs: {
            "loss": detached,
            "direction": detached,
            "speed": detached,
            "yaw": detached,
            "teacher_valid_steps": torch.tensor(float(steps)),
            "teacher_loss_active": torch.tensor(1.0),
            "teacher_direction_mask": torch.ones(steps),
            "teacher_speed_mask": torch.ones(steps),
            "teacher_yaw_mask": torch.ones(steps),
        },
    )
    batch = {
        "camera_aux_mask": torch.zeros(steps, 3),
        "clean_action_mean": torch.zeros(steps, 3),
        "teacher_safe3": torch.zeros(steps, 3),
        "teacher_safe5": torch.zeros(steps, 5),
        "teacher_goal_xy": torch.zeros(steps, 2),
        "teacher_predictive_risk": torch.zeros(steps, 1),
        "teacher_mask": torch.ones(steps, 1),
        "teacher_goal_mask": torch.ones(steps, 1),
        "teacher_weight": torch.ones(steps, 1),
        "stuck_label": torch.zeros(steps, 1),
        "stuck_mask": torch.zeros(steps, 1),
    }
    loss, metrics = algorithm._actor_auxiliary_loss(
        normalized_mean=torch.tanh(mean),
        actor_normalized_mean=torch.tanh(mean),
        actor_features=torch.zeros(steps, algorithm.actor.hidden_dim),
        batch=batch,
        ppo_actor_loss=mean.square().mean(),
    )
    assert loss.item() == 0.0
    assert metrics["teacher_guidance_loss"] == pytest.approx(2.5)
    assert "teacher" not in algorithm._auxiliary_coefficients


def test_eval_assembly_does_not_create_training_stuck_head():
    assert _algorithm(training=True).stuck_head is not None
    assert _algorithm(training=False).stuck_head is None


def test_high_level_mirror_is_an_involution_and_preserves_goal_distance():
    nonvisual = torch.arange(36, dtype=torch.float32).reshape(1, 1, 36)
    response = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16)
    mirrored_nonvisual = AlgorithmP4NavPPO._mirror_nav_nonvisual(nonvisual)
    mirrored_response = AlgorithmP4NavPPO._mirror_response_profile(response)
    assert mirrored_nonvisual[..., 2].item() == nonvisual[..., 2].item()
    torch.testing.assert_close(
        AlgorithmP4NavPPO._mirror_nav_nonvisual(mirrored_nonvisual), nonvisual
    )
    torch.testing.assert_close(
        AlgorithmP4NavPPO._mirror_response_profile(mirrored_response), response
    )


def test_episode_aligned_mirror_microbatch_keeps_cnn_frozen(
    monkeypatch,
):
    algorithm = _algorithm(training=True)
    original_schedule = p4_contract.training_schedule

    def mirror_schedule(seconds, *, branch="actor_attack"):
        result = dict(original_schedule(seconds, branch=branch))
        result["mirror_sequence_share"] = 1.0
        return result

    monkeypatch.setattr(p4_contract, "training_schedule", mirror_schedule)
    timesteps, batch_size = 16, 1
    nav_feat = torch.randn(timesteps, batch_size, 32)
    actor_input = torch.randn(
        timesteps, batch_size, p2_contract.ACTOR_INPUT_DIM
    )
    hidden = (
        torch.zeros(2, batch_size, 64),
        torch.zeros(2, batch_size, 64),
    )
    pre_tanh = torch.zeros(timesteps, batch_size, 3)
    _, _, mean, _, _, features = algorithm.actor.evaluate_actions(
        actor_input,
        pre_tanh,
        hidden,
        torch.zeros(timesteps, batch_size, dtype=torch.bool),
        return_features=True,
    )
    batch = {
        "nav_nonvisual": torch.zeros(
            timesteps, batch_size, p2_contract.NAV_NONVISUAL_DIM
        ),
        "response_profile": torch.zeros(
            timesteps, batch_size, p2_contract.RESPONSE_PROFILE_DIM
        ),
        "confidence": torch.ones(timesteps, batch_size, 1),
        "pre_tanh_action": pre_tanh,
        "reset_mask": torch.cat(
            (
                torch.ones(1, batch_size, dtype=torch.bool),
                torch.zeros(timesteps - 1, batch_size, dtype=torch.bool),
            ),
            dim=0,
        ),
        "actor_hidden": hidden,
        "camera_aux_mask": torch.zeros(timesteps, batch_size, 3),
        "clean_action_mean": torch.zeros(timesteps, batch_size, 3),
        "teacher_safe3": torch.zeros(timesteps, batch_size, 3),
        "teacher_goal_xy": torch.zeros(timesteps, batch_size, 2),
        "teacher_predictive_risk": torch.zeros(timesteps, batch_size, 1),
        "teacher_mask": torch.zeros(timesteps, batch_size, 1),
        "teacher_goal_mask": torch.zeros(timesteps, batch_size, 1),
        "teacher_weight": torch.ones(timesteps, batch_size, 1),
        "stuck_label": torch.zeros(timesteps, batch_size, 1),
        "stuck_mask": torch.zeros(timesteps, batch_size, 1),
        "mirror_eligible": torch.ones(timesteps, batch_size, 1),
    }
    batch["mirror_batch"] = {
        "depth": torch.zeros(
            timesteps, batch_size, p2_contract.DEPTH_HEIGHT,
            p2_contract.DEPTH_WIDTH, 1, dtype=torch.float16,
        ),
        "nav_nonvisual": batch["nav_nonvisual"],
        "response_profile": batch["response_profile"],
        "confidence": batch["confidence"],
        "pre_tanh_action": batch["pre_tanh_action"],
        "reset_mask": batch["reset_mask"],
    }
    ppo_loss = mean.square().mean()
    loss, metrics = algorithm._actor_auxiliary_loss(
        normalized_mean=torch.tanh(mean),
        actor_features=features,
        nav_feat=nav_feat,
        batch=batch,
        ppo_actor_loss=ppo_loss,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["mirror_aux_sequence_share"].item() == 1.0
    assert all(
        parameter.grad is None for parameter in algorithm.navigation_encoder.parameters()
    )


def test_mirror_microbatch_rejects_mid_episode_nonzero_hidden(monkeypatch):
    algorithm = _algorithm(training=True)
    original_schedule = p4_contract.training_schedule

    def mirror_schedule(seconds, *, branch="actor_attack"):
        result = dict(original_schedule(seconds, branch=branch))
        result["mirror_sequence_share"] = 1.0
        return result

    monkeypatch.setattr(p4_contract, "training_schedule", mirror_schedule)
    timesteps, batch_size = 16, 1
    actor_input = torch.randn(timesteps, batch_size, p2_contract.ACTOR_INPUT_DIM)
    hidden = (torch.ones(2, batch_size, 64), torch.ones(2, batch_size, 64))
    pre_tanh = torch.zeros(timesteps, batch_size, 3)
    _, _, mean, _, _, features = algorithm.actor.evaluate_actions(
        actor_input,
        pre_tanh,
        hidden,
        torch.zeros(timesteps, batch_size, dtype=torch.bool),
        return_features=True,
    )
    batch = {
        "depth": torch.zeros(
            timesteps, batch_size, p2_contract.DEPTH_HEIGHT,
            p2_contract.DEPTH_WIDTH, 1,
        ),
        "nav_nonvisual": torch.zeros(timesteps, batch_size, p2_contract.NAV_NONVISUAL_DIM),
        "response_profile": torch.zeros(timesteps, batch_size, p2_contract.RESPONSE_PROFILE_DIM),
        "confidence": torch.ones(timesteps, batch_size, 1),
        "pre_tanh_action": pre_tanh,
        "reset_mask": torch.zeros(timesteps, batch_size, dtype=torch.bool),
        "actor_hidden": hidden,
        "camera_aux_mask": torch.zeros(timesteps, batch_size, 3),
        "clean_action_mean": torch.zeros(timesteps, batch_size, 3),
        "teacher_safe3": torch.zeros(timesteps, batch_size, 3),
        "teacher_goal_xy": torch.zeros(timesteps, batch_size, 2),
        "teacher_predictive_risk": torch.zeros(timesteps, batch_size, 1),
        "teacher_mask": torch.zeros(timesteps, batch_size, 1),
        "teacher_goal_mask": torch.zeros(timesteps, batch_size, 1),
        "teacher_weight": torch.ones(timesteps, batch_size, 1),
        "stuck_label": torch.zeros(timesteps, batch_size, 1),
        "stuck_mask": torch.zeros(timesteps, batch_size, 1),
        "mirror_eligible": torch.ones(timesteps, batch_size, 1),
    }
    loss, metrics = algorithm._actor_auxiliary_loss(
        normalized_mean=torch.tanh(mean),
        actor_features=features,
        nav_feat=torch.randn(timesteps, batch_size, 32),
        batch=batch,
        ppo_actor_loss=mean.square().mean(),
    )
    assert torch.isfinite(loss)
    assert metrics["mirror_aux_sequence_share"].item() == 0.0


def test_algorithm_mapper_keeps_near_goal_capture_shadow_only():
    algorithm = _algorithm(
        training=False,
        config={
            "training_profile": "maze_closed_loop_v3",
            "maze_training_branch": "closed_loop_v3",
            "track_segment_labels": ["maze"],
            "camera_fault_course_enabled": False,
            "goal_fault_course_enabled": False,
        },
    )
    algorithm._delivered_depth = torch.ones(
        1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
    )
    algorithm.goal_belief.estimate[:] = torch.tensor([[0.80, 0.0]])
    algorithm._safety_cap_predictive_risk.zero_()
    algorithm._translation_alpha_prev.fill_(1.0)
    algorithm.reset_since_tick.zero_()
    algorithm._goal_epoch_changed_since_tick.zero_()

    normalized = torch.tensor([[1.0, 0.60, 0.60]])
    mapped = algorithm._map_policy_target(
        normalized,
        torch.zeros_like(normalized),
        goal4=torch.tensor([[0.80, 0.0, 0.0, 1.0]]),
        aux=torch.empty(1, 0),
    )

    policy = algorithm._last_policy_command
    diagnostics = algorithm._near_goal_capture_diagnostics
    assert diagnostics["near_goal_capture_candidate"].item() == 1.0
    assert diagnostics["near_goal_capture_active"].item() == 0.0
    torch.testing.assert_close(mapped, policy)


def test_translation_limiter_risk_uses_current_candidate_not_previous_target(monkeypatch):
    algorithm = _algorithm(training=False)
    algorithm._delivered_depth = torch.ones(
        1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
    )
    algorithm.command.active_target[:] = torch.tensor([[0.8, 0.0, 0.0]])
    algorithm.goal_belief.estimate[:] = torch.tensor([[4.0, 0.0]])
    algorithm.reset_since_tick.zero_()
    algorithm._goal_epoch_changed_since_tick.zero_()
    monkeypatch.setattr(algorithm, "_predictive_command", lambda target: target)

    def candidate_risk(_depth, command):
        risk = (command[:, 1].abs() / p4_contract.P4_MAX_ABS_VY).clamp(0.0, 1.0)
        zero = torch.zeros_like(risk)
        return zero, zero, zero, risk

    monkeypatch.setattr(
        p2_contract, "predictive_collision_risk_penalty", candidate_risk
    )
    normalized = torch.tensor([[0.0, 1.0, 0.0]])
    mapped = algorithm._map_policy_target(
        normalized,
        torch.zeros_like(normalized),
        goal4=torch.tensor([[4.0, 0.0, 0.0, 1.0]]),
        aux=torch.empty(1, 0),
    )
    assert algorithm._translation_limiter_diagnostics[
        "translation_safety_risk"
    ].item() > 0.75
    assert mapped[0, 1].abs().item() < algorithm._last_policy_command[0, 1].abs().item()


def test_closed_loop_limiter_is_shadow_only(monkeypatch):
    algorithm = _algorithm(
        training=True,
        config={
            "training_profile": "maze_closed_loop_v3",
            "maze_training_branch": "closed_loop_v3",
            "track_segment_labels": ["maze"],
            "camera_fault_course_enabled": False,
            "goal_fault_course_enabled": False,
        },
    )
    algorithm._delivered_depth = torch.ones(
        1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
    )
    algorithm.goal_belief.estimate[:] = torch.tensor([[4.0, 0.0]])
    algorithm.reset_since_tick.zero_()
    algorithm._goal_epoch_changed_since_tick.zero_()
    monkeypatch.setattr(algorithm, "_predictive_command", lambda target: target)

    def full_risk(_depth, command):
        risk = torch.ones(command.shape[0])
        zero = torch.zeros_like(risk)
        return zero, zero, zero, risk

    monkeypatch.setattr(
        p2_contract, "predictive_collision_risk_penalty", full_risk
    )
    normalized = torch.tensor([[0.5, 0.5, 0.2]])
    goal4 = torch.tensor([[4.0, 0.0, 0.0, 1.0]])
    shadow = algorithm._map_policy_target(
        normalized, torch.zeros_like(normalized), goal4=goal4, aux=torch.empty(1, 0)
    )
    torch.testing.assert_close(shadow, algorithm._last_policy_command)
    assert algorithm._translation_limiter_diagnostics[
        "translation_limiter_shadow"
    ].item() == 1.0

    algorithm.session_effective_seconds = 7_200.0
    algorithm._translation_alpha_prev.fill_(1.0)
    later = algorithm._map_policy_target(
        normalized, torch.zeros_like(normalized), goal4=goal4, aux=torch.empty(1, 0)
    )
    assert algorithm._translation_limiter_diagnostics[
        "translation_limiter_shadow"
    ].item() == 1.0
    torch.testing.assert_close(later, algorithm._last_policy_command)


def test_exact_resume_restores_training_only_stuck_head_and_mirror_rng():
    config = {
        "training_profile": "maze_credit_repair",
        "maze_training_branch": "credit_repair",
        "track_segment_labels": ["maze"],
        "camera_fault_course_enabled": False,
        "goal_fault_course_enabled": False,
    }
    algorithm = _algorithm(training=True, config=config)
    algorithm._initial_low_digest = algorithm._module_digest(
        (("vision", algorithm.low_level_encoder), ("actor", algorithm.low_level_actor))
    )
    algorithm.low_level_state_digest = algorithm._initial_low_digest
    algorithm._configure_adapter_contract()
    algorithm.stuck_head.logit.bias.data.fill_(0.37)
    _ = torch.rand(4, generator=algorithm.mirror_generator)
    expected_rng = algorithm.mirror_generator.get_state().clone()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.ckpt-mazeprobe-42.pkl"
        algorithm.save_training_bundle(str(path), platform_model_id="42")
        resumed = _algorithm(training=True, config=config)
        mode = resumed.load_bundle(str(path), platform_model_id="42")

    assert mode == "p4_exact_resume_history_reset"
    torch.testing.assert_close(resumed.stuck_head.logit.bias, algorithm.stuck_head.logit.bias)
    assert resumed.mirror_generator.get_state().equal(expected_rng)
