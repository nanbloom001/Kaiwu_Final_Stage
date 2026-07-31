#!/usr/bin/env python3

import copy
from types import SimpleNamespace

import torch
import pytest

from agent_ppo.algorithm.algorithm_p3_standard_joint import (
    AlgorithmP3HighPPO,
    AlgorithmP3StandardJoint,
    _validate_resume_phase,
)
from agent_ppo.feature import p2_contract, p3_contract
from agent_ppo.algorithm.algorithm_visual_ppo import AlgorithmVisualPPO
from agent_ppo.checkpoint_io import CheckpointSaveError
from agent_ppo.model.visual_actor_critic import VisualActorCritic
from agent_ppo.workflow.p3_standard_joint_workflow import (
    _advance_platform_lifecycle,
    _finalize_low_level_update,
    _high_adapter_update_due,
    _low_policy_action,
    _native_command_epoch,
    _requires_environment_rebuild,
    _storage_bytes,
    _zero_inactive_high_commands,
)


def _module():
    return torch.nn.Linear(1, 1)


def test_p3_visual_schedule_enables_expected_low_modules():
    algorithm = object.__new__(AlgorithmVisualPPO)
    algorithm.schedule_mode = "p3_low_recovery_v1"
    algorithm.actor_critic = SimpleNamespace(
        actor=_module(),
        critic=_module(),
        std=torch.nn.Parameter(torch.ones(1)),
        vision_encoder=SimpleNamespace(
            cnn=_module(), rnn=_module(), rnn_output_layer=_module()
        ),
    )
    algorithm._set_trainable_phase("lowbase")
    assert all(parameter.requires_grad for parameter in algorithm.actor_critic.actor.parameters())
    assert all(parameter.requires_grad for parameter in algorithm.actor_critic.vision_encoder.rnn.parameters())
    assert not any(parameter.requires_grad for parameter in algorithm.actor_critic.vision_encoder.cnn.parameters())
    algorithm._set_trainable_phase("highslow")
    assert not any(parameter.requires_grad for parameter in algorithm.actor_critic.actor.parameters())
    assert not any(parameter.requires_grad for parameter in algorithm.actor_critic.vision_encoder.rnn.parameters())


def test_visual_ppo_training_mode_keeps_only_frozen_modules_in_eval():
    algorithm = object.__new__(AlgorithmVisualPPO)
    algorithm.actor_critic = torch.nn.Module()
    algorithm.actor_critic.vision_encoder = torch.nn.Module()
    algorithm.actor_critic.vision_encoder.cnn = torch.nn.Sequential(
        torch.nn.BatchNorm2d(1)
    )
    algorithm.actor_critic.vision_encoder.rnn = torch.nn.LSTM(1, 1, batch_first=True)
    algorithm.actor_critic.actor = _module()
    algorithm.actor_critic.critic = _module()
    algorithm.anchor_encoder = _module()
    algorithm.anchor_actor = _module()

    algorithm.actor_critic.eval()
    algorithm.anchor_encoder.train()
    algorithm.anchor_actor.train()
    algorithm._prepare_training_modules()

    assert algorithm.actor_critic.training
    assert algorithm.actor_critic.vision_encoder.rnn.training
    assert algorithm.actor_critic.actor.training
    assert algorithm.actor_critic.critic.training
    assert not algorithm.actor_critic.vision_encoder.cnn.training
    assert not algorithm.anchor_encoder.training
    assert not algorithm.anchor_actor.training


def test_compact_low_level_replay_matches_full_observation():
    torch.manual_seed(5)
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
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
    ).eval()
    full = torch.randn(1, 57901)
    model.vision_encoder.reset_hidden_state(1, torch.device("cpu"))
    model.update_distribution(full)
    full_mean = model.action_mean.detach().clone()
    compact = torch.cat((full[:, :45], model.last_cnn_features.detach()), dim=-1)
    model.vision_encoder.reset_hidden_state(1, torch.device("cpu"))
    model.update_distribution(compact)
    assert compact.shape == (1, 77)
    assert torch.allclose(model.action_mean, full_mean, atol=1.0e-6, rtol=1.0e-6)


def test_proprio_depth_fast_path_matches_full_observation():
    torch.manual_seed(11)
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
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
    ).eval()
    full = torch.randn(2, 57901)
    model.vision_encoder.reset_hidden_state(2, torch.device("cpu"))
    model.update_distribution(full)
    expected = model.action_mean.detach().clone()
    depth = full[:, 301:].reshape(2, 180, 320, 1)
    model.vision_encoder.reset_hidden_state(2, torch.device("cpu"))
    model.update_distribution_from_proprio_depth(full[:, :45], depth)
    assert torch.allclose(model.action_mean, expected, atol=1.0e-6, rtol=1.0e-6)


def test_anchor_reuses_identical_frozen_cnn_features():
    torch.manual_seed(17)
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
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
    ).eval()
    algorithm = object.__new__(AlgorithmVisualPPO)
    algorithm.actor_critic = model
    algorithm.anchor_encoder = copy.deepcopy(model.vision_encoder).eval()
    algorithm.anchor_actor = copy.deepcopy(model.actor).eval()
    observation = torch.randn(2, 57901)

    algorithm.anchor_encoder.reset_hidden_state(2, torch.device("cpu"))
    full_action, full_latent = algorithm.anchor_inference(observation)
    algorithm.anchor_encoder.reset_hidden_state(2, torch.device("cpu"))
    model.vision_encoder.reset_hidden_state(2, torch.device("cpu"))
    model.act(observation)
    reused_action, reused_latent = algorithm.anchor_inference_from_cnn_features(
        observation[:, :45], model.last_cnn_features
    )

    assert torch.allclose(reused_latent, full_latent, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(reused_action, full_action, atol=1.0e-6, rtol=1.0e-6)


def test_adapter_calibration_low_action_skips_ppo_metadata():
    class Actor:
        def __init__(self):
            self.calls = 0

        def act_from_proprio_depth(self, proprio, depth):
            self.calls += 1
            assert proprio.shape == (2, 45)
            assert depth.shape == (2, 180, 320, 1)
            return torch.zeros(2, 12)

        def evaluate(self, *_args, **_kwargs):
            raise AssertionError("adapter calibration must not evaluate the critic")

    actor = Actor()
    agent = SimpleNamespace(
        low_level_algorithm=SimpleNamespace(actor_critic=actor)
    )
    actions = _low_policy_action(
        agent,
        torch.zeros(2, 45),
        torch.zeros(2, 180, 320, 1),
    )
    assert actor.calls == 1
    assert actions.shape == (2, 12)


def test_coordinator_keeps_std_frozen_and_scales_low_lrs():
    visual = object.__new__(AlgorithmVisualPPO)
    visual.anchor_session_elapsed_hours = 0.0
    visual.current_phase = ""
    visual.actor_critic = SimpleNamespace(
        std=torch.nn.Parameter(torch.ones(1))
    )
    visual._set_trainable_phase = lambda phase: visual.actor_critic.std.requires_grad_(True)
    visual.optimizer = torch.optim.Adam(
        [
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 1e-5, "name": "actor"},
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 5e-6, "name": "lstm"},
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 1e-4, "name": "critic"},
        ]
    )
    high = SimpleNamespace(
        update_training_clocks=lambda elapsed: None,
        num_envs=2,
        device="cpu",
    )
    joint = AlgorithmP3StandardJoint(
        low_algorithm=visual,
        high_algorithm=high,
        config={},
        logger=None,
    )
    assert not visual.actor_critic.std.requires_grad
    assert [group["lr"] for group in visual.optimizer.param_groups] == pytest.approx([2e-6, 1e-6, 1e-4])
    joint.update_clock(6.5 * 3600)
    assert [group["lr"] for group in visual.optimizer.param_groups] == pytest.approx([0.0, 0.0, 0.0])


def test_only_domain_randomization_boundaries_rebuild_environment():
    assert _requires_environment_rebuild(1799.0, 1800.0)
    assert _requires_environment_rebuild(7199.0, 7200.0)
    assert _requires_environment_rebuild(12599.0, 12600.0)
    assert not _requires_environment_rebuild(17999.0, 18000.0)
    assert not _requires_environment_rebuild(21599.0, 21600.0)
    assert not _requires_environment_rebuild(23399.0, 23400.0)


def test_high_adapter_updates_every_second_policy_rollout():
    config = {"high_update_interval": 2}
    assert not _high_adapter_update_due(1, config)
    assert _high_adapter_update_due(2, config)
    assert not _high_adapter_update_due(3, config)
    assert _high_adapter_update_due(4, config)


def test_native_command_epoch_tracks_each_environment_independently():
    agent = SimpleNamespace()
    first = torch.tensor([[0.2, 0.0, 0.0], [0.3, 0.0, 0.1]])
    assert _native_command_epoch(agent, first).tolist() == [0, 0]
    same = first.clone()
    assert _native_command_epoch(agent, same).tolist() == [0, 0]
    changed = same.clone()
    changed[1, 2] = -0.1
    assert _native_command_epoch(agent, changed).tolist() == [0, 1]
    changed[0, 0] = 0.4
    assert _native_command_epoch(agent, changed).tolist() == [1, 1]


def test_storage_bytes_counts_all_distinct_tensor_buffers():
    shared = torch.zeros(4, dtype=torch.float32)
    storage = SimpleNamespace(
        observations=torch.zeros(2, 3, dtype=torch.float32),
        values=torch.zeros(2, 3, dtype=torch.float16),
        hidden=(shared, shared),
        step=0,
    )
    expected = 6 * 4 + 6 * 2 + 4 * 4
    assert _storage_bytes(storage) == expected


def test_low_level_version_advances_only_after_optimizer_step():
    coordinator = SimpleNamespace(low_updates=4, note_low_level_update=lambda: None)
    agent = SimpleNamespace(algorithm=coordinator)
    assert not _finalize_low_level_update(agent, {"applied_updates": 0.0})
    assert coordinator.low_updates == 4

    calls = []
    coordinator.note_low_level_update = lambda: calls.append("updated")
    assert _finalize_low_level_update(agent, {"applied_updates": 3.0})
    assert coordinator.low_updates == 5
    assert calls == ["updated"]


def test_p3_lifecycle_advances_once_per_successful_step():
    calls = []
    agent = SimpleNamespace(
        learn=lambda list_sample_data=None: calls.append(list_sample_data),
        logger=SimpleNamespace(info=lambda *_args: None, error=lambda *_args: None),
    )
    assert _advance_platform_lifecycle(agent)
    assert _advance_platform_lifecycle(agent)
    assert calls == [None, None]
    assert agent._p3_lifecycle_attempt_callbacks == 2
    assert agent._p3_lifecycle_success_callbacks == 2
    assert getattr(agent, "_p3_lifecycle_failure_callbacks", 0) == 0


def test_p3_lifecycle_counts_transient_failure_and_continues():
    def fail_once(list_sample_data=None):
        del list_sample_data
        if not getattr(agent, "failed", False):
            agent.failed = True
            raise RuntimeError("synthetic lifecycle failure")

    agent = SimpleNamespace(
        learn=fail_once,
        logger=SimpleNamespace(info=lambda *_args: None, error=lambda *_args: None),
    )
    assert not _advance_platform_lifecycle(agent)
    assert _advance_platform_lifecycle(agent)
    assert agent._p3_lifecycle_attempt_callbacks == 2
    assert agent._p3_lifecycle_success_callbacks == 1
    assert agent._p3_lifecycle_failure_callbacks == 1


def test_p3_lifecycle_checkpoint_failure_stops_training():
    def fail_checkpoint(list_sample_data=None):
        del list_sample_data
        raise CheckpointSaveError("synthetic checkpoint failure")

    agent = SimpleNamespace(
        learn=fail_checkpoint,
        logger=SimpleNamespace(info=lambda *_args: None, error=lambda *_args: None),
    )
    with pytest.raises(CheckpointSaveError, match="synthetic checkpoint failure"):
        _advance_platform_lifecycle(agent)
    assert agent._p3_lifecycle_attempt_callbacks == 1
    assert getattr(agent, "_p3_lifecycle_success_callbacks", 0) == 0


def test_exact_resume_phase_must_match_effective_clock():
    raw = {"phase_label": "lowmild"}
    state = {"compound_schedule_phase": "lowmild"}
    assert _validate_resume_phase(raw, state, 1800.0) == "lowmild"
    with pytest.raises(RuntimeError, match="phase/time mismatch"):
        _validate_resume_phase(
            {"phase_label": "highslow"},
            {"compound_schedule_phase": "highslow"},
            1800.0,
        )


def test_joint_success_monitor_persists_across_rollouts_and_resets():
    coordinator = object.__new__(AlgorithmP3StandardJoint)
    coordinator.episode_subgoal_successes = torch.zeros(2, dtype=torch.long)
    aux = torch.zeros(2, p2_contract.WORKER_AUX_DIM)
    aux[0, p2_contract.CURRENT_SEGMENT_INDEX] = p3_contract.SUBGOAL_EVENT_REACHED
    coordinator.observe_subgoal_and_terminal(aux)
    coordinator.observe_subgoal_and_terminal(aux)
    terminal = torch.zeros(2, p2_contract.WORKER_AUX_DIM)
    terminal[0, 24] = 1
    terminal[0, 25] = 1
    metrics = coordinator.observe_subgoal_and_terminal(terminal)
    assert metrics["p3_standard_success_count"].item() == 1.0
    assert metrics["p3_joint_success_count"].item() == 1.0
    assert coordinator.episode_subgoal_successes.tolist() == [0, 0]


def test_inactive_high_commands_are_zeroed_without_touching_live_envs():
    command = SimpleNamespace(
        active_target=torch.tensor([[0.4, 0.1, -0.2], [0.7, -0.2, 0.3]]),
        exec_cmd=torch.tensor([[0.3, 0.1, -0.1], [0.6, -0.1, 0.2]]),
    )
    _zero_inactive_high_commands(command, torch.tensor([False, True]))
    assert torch.equal(command.active_target[0], torch.zeros(3))
    assert torch.equal(command.exec_cmd[0], torch.zeros(3))
    assert torch.equal(command.active_target[1], torch.tensor([0.7, -0.2, 0.3]))
    assert torch.equal(command.exec_cmd[1], torch.tensor([0.6, -0.1, 0.2]))


def test_high_schedule_applies_real_adapter_learning_rates():
    algorithm = object.__new__(AlgorithmP3HighPPO)
    algorithm.config = {
        "response_adapter": {
            "learning_rate": 3.0e-5,
            "calibration_learning_rate": 2.0e-4,
            "high_adapt_learning_rate": 1.0e-5,
        }
    }
    algorithm.lifetime_base_seconds = 0.0
    algorithm.actor_optimizer = torch.optim.Adam(
        [
            {
                "params": [torch.nn.Parameter(torch.ones(1))],
                "lr": 3.0e-6,
                "base_lr": 3.0e-6,
                "name": "navigation_conv1",
            },
            {
                "params": [torch.nn.Parameter(torch.ones(1))],
                "lr": 3.0e-4,
                "base_lr": 3.0e-4,
                "name": "actor",
            },
        ]
    )
    algorithm.critic_optimizer = torch.optim.Adam(
        [torch.nn.Parameter(torch.ones(1))], lr=3.0e-4
    )
    algorithm.response_optimizer = torch.optim.Adam(
        [torch.nn.Parameter(torch.ones(1))], lr=3.0e-5
    )

    algorithm._apply_training_schedule(0.0)
    assert algorithm.response_optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-5)

    algorithm._apply_training_schedule(5.0 * 3600)
    assert algorithm.current_phase == "adaptercalib"
    assert algorithm.response_optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-4)
    assert all(group["lr"] == 0.0 for group in algorithm.actor_optimizer.param_groups)
    assert algorithm.critic_optimizer.param_groups[0]["lr"] == 0.0

    algorithm._apply_training_schedule(6.0 * 3600)
    assert algorithm.current_phase == "highadapt"
    assert algorithm.response_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5)

    algorithm._apply_training_schedule(6.5 * 3600)
    assert algorithm.current_phase == "highslow"
    assert algorithm.response_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5)


def test_p3_high_reward_profile_disables_track_only_shaping():
    algorithm = object.__new__(AlgorithmP3HighPPO)
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
    components = {name: torch.ones(2) for name in names}
    result = algorithm._override_reward_components(components)
    for name in (
        "success",
        "timeout",
        "tracking",
        "gait_symmetry",
        "body_collision",
        "predictive_collision_risk",
        "missed_safe_direction",
        "frontier_stagnation",
    ):
        assert torch.equal(result[name], torch.zeros(2))
    assert torch.equal(result["frontier_shaping"], torch.ones(2))
    assert torch.equal(result["frame_safety"], torch.ones(2))
