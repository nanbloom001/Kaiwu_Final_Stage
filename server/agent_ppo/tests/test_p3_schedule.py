#!/usr/bin/env python3

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from agent_ppo.algorithm.algorithm_p3_standard_joint import (
    AlgorithmP3HighPPO,
    AlgorithmP3StandardJoint,
    P3_CONTRACT_WARM_START_MODES,
    _validate_configured_parent_digest,
    _validate_resume_phase,
)
from agent_ppo.feature import p2_contract, p3_contract
from agent_ppo.algorithm.algorithm_visual_ppo import AlgorithmVisualPPO
from agent_ppo.checkpoint_io import CheckpointSaveError
from agent_ppo.model.visual_actor_critic import VisualActorCritic
from agent_ppo.workflow.p3_standard_joint_workflow import (
    P35_REQUIRED_MONITOR_METRICS,
    _advance_platform_lifecycle,
    _finalize_low_level_update,
    _high_adapter_update_due,
    _low_policy_action,
    _monitor_contract_metrics,
    _native_command_epoch,
    _p35_baseline_gate_reason,
    _run_adapter_updates,
    _storage_bytes,
    _zero_inactive_high_commands,
)
from agent_ppo.tools.p3_joint_rollout_smoke import (
    _assert_p35_runtime_health,
    _load_config,
    _smoke_num_mini_batches,
)


def _module():
    return torch.nn.Linear(1, 1)


def test_p35_previous_run_final_is_an_explicit_contract_warm_start():
    assert "p35_previous_run_final_warm_start" in P3_CONTRACT_WARM_START_MODES
    assert "p35_fixed_parent_warm_start" in P3_CONTRACT_WARM_START_MODES


def test_p35_fixed_parent_digest_is_a_hard_gate(tmp_path):
    artifact = tmp_path / "parent.pkl"
    artifact.write_bytes(b"fixed-p35-parent")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert _validate_configured_parent_digest(str(artifact), expected) == expected
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _validate_configured_parent_digest(str(artifact), "0" * 64)


def test_p35_integrated_smoke_uses_current_gaitwarm_phase():
    source = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "p3_joint_rollout_smoke.py"
    ).read_text(encoding="utf-8")
    assert 'PHASE_BOUNDARIES["gaitwarm"]' in source
    assert 'PHASE_BOUNDARIES["stairrobust"]' not in source


def test_p35_monitor_names_avoid_platform_forbidden_periods():
    source = (
        Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
    ).read_text(encoding="utf-8")
    assert 'monitor.title("P35 低速楼梯与后期Push")' in source
    assert '"P35训练侧奖励"' in source
    assert 'monitor.title("P3.5' not in source
    assert '"P3.5训练侧奖励"' not in source


def test_p35_baseline_correctness_gate_rejects_empty_and_invalid_calibration():
    shaper = SimpleNamespace(
        sample_count={
            "joint_pos": 0,
            "joint_acc_noncontact": 0,
            "joint_acc_onset": 0,
            "posture": 0,
            "frequency": 0,
        },
        eligibility={
            name: torch.tensor(0.0)
            for name in ("base", "joint", "contact", "gait")
        },
        finalized=False,
        valid=False,
        component_valid={
            name: False
            for name in (
                "progress", "default_posture", "joint_acc",
                "contact", "gait", "posture",
            )
        },
    )
    agent = SimpleNamespace(algorithm=SimpleNamespace(low_reward_shaper=shaper))
    assert "five_minute_empty_samples" in _p35_baseline_gate_reason(agent, 300.0)
    shaper.sample_count = {name: 1 for name in shaper.sample_count}
    shaper.eligibility = {
        name: torch.tensor(1.0) for name in shaper.eligibility
    }
    assert _p35_baseline_gate_reason(agent, 300.0) is None
    shaper.finalized = True
    assert "fifteen_minute_invalid_components" in _p35_baseline_gate_reason(
        agent, 900.0
    )


@pytest.mark.parametrize(
    ("num_envs", "expected"), ((1, 1), (2, 2), (3, 3), (4, 4), (6, 3))
)
def test_p35_smoke_uses_a_divisible_minibatch_count(num_envs, expected):
    assert _smoke_num_mini_batches(num_envs, 4) == expected


def test_p35_smoke_config_enables_bounded_real_push_only_for_smoke():
    path = (
        Path(__file__).resolve().parents[1]
        / "conf"
        / "train_env_conf_standard_p3_standard_joint.toml"
    )
    config = _load_config(path, 1, True)
    schedule = config["p3_standard_joint"]["push_schedule"]
    phase = p3_contract.push_phase_config(900.0, schedule)
    assert phase == {
        "name": "smoke",
        "active": True,
        "max_velocity_xy_m_s": 0.03,
        "min_interval_s": 0.5,
        "max_interval_s": 1.0,
    }
    assert p3_contract.push_phase_config(900.0)["active"] is False


def test_p35_production_push_intervals_fit_forty_second_episode():
    disabled = p3_contract.push_phase_config(0.0)
    warm = p3_contract.push_phase_config(14400.0)
    full = p3_contract.push_phase_config(21600.0)
    assert (disabled["min_interval_s"], disabled["max_interval_s"]) == (20.0, 30.0)
    assert (warm["min_interval_s"], warm["max_interval_s"]) == (20.0, 30.0)
    assert (full["min_interval_s"], full["max_interval_s"]) == (17.0, 27.0)
    assert full["max_interval_s"] < 40.0


def test_p35_smoke_runtime_health_rejects_invalid_mapping_or_missing_push():
    extra = torch.ones(2, p3_contract.P3_WORKER_EXTRA_DIM)
    agent = SimpleNamespace(_p3_extra=extra)
    healthy = {
        "gait_sensor_mapping_valid": 1.0,
        "gait_window_valid": 1.0,
        "p35_push_telemetry_valid_share": 1.0,
        "p35_push_runtime_active_share": 1.0,
        "p35_push_event_count": 1.0,
        "p35_push_delta_vx_max": 0.02,
        "p35_push_delta_vy_max": 0.01,
        "p35_push_config_violation_count": 0.0,
    }
    _assert_p35_runtime_health(agent, healthy)
    invalid_mapping = dict(healthy)
    extra[:, p3_contract.CONTACT_REWARD_MAPPING_VALID_INDEX] = 0.0
    with pytest.raises(AssertionError, match="mapping invalid"):
        _assert_p35_runtime_health(agent, invalid_mapping)
    extra[:, p3_contract.CONTACT_REWARD_MAPPING_VALID_INDEX] = 1.0
    missing_push = dict(healthy, p35_push_event_count=0.0)
    with pytest.raises(AssertionError, match="no real Push"):
        _assert_p35_runtime_health(agent, missing_push)


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


def test_p35_visual_schedule_trains_only_action_head_and_recurrent_path():
    algorithm = object.__new__(AlgorithmVisualPPO)
    algorithm.schedule_mode = "p35_gaitfix_v1"
    actor = torch.nn.Sequential(
        torch.nn.Linear(1, 2), torch.nn.ELU(), torch.nn.Linear(2, 1)
    )
    algorithm.actor_critic = SimpleNamespace(
        actor=actor,
        critic=_module(),
        std=torch.nn.Parameter(torch.ones(1)),
        vision_encoder=SimpleNamespace(
            cnn=_module(), rnn=_module(), rnn_output_layer=_module()
        ),
    )
    algorithm._set_trainable_phase("calib")
    assert not any(parameter.requires_grad for parameter in actor.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in algorithm.actor_critic.vision_encoder.rnn.parameters()
    )
    algorithm._set_trainable_phase("gaitwarm")
    assert not any(parameter.requires_grad for parameter in actor[0].parameters())
    assert all(parameter.requires_grad for parameter in actor[-1].parameters())
    assert all(
        parameter.requires_grad
        for parameter in algorithm.actor_critic.vision_encoder.rnn.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in algorithm.actor_critic.vision_encoder.cnn.parameters()
    )
    algorithm._set_trainable_phase("stable")
    assert not any(parameter.requires_grad for parameter in actor.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in algorithm.actor_critic.vision_encoder.rnn.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in algorithm.actor_critic.critic.parameters()
    )


def test_p35_fixed_parent_digest_gate_precedes_stage_type_dispatch(tmp_path):
    artifact = tmp_path / "parent.pkl"
    torch.save({"stage_type": "p2_nav_ppo"}, artifact)
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    algorithm = object.__new__(AlgorithmP3StandardJoint)
    algorithm.config = {
        "load_mode": "p35_fixed_parent_warm_start",
        "parent_checkpoint_sha256": expected,
    }
    algorithm.load_parent = lambda path, platform_model_id: "loaded-parent"
    assert algorithm.load_checkpoint(
        str(artifact), platform_model_id="1013548"
    ) == "loaded-parent"
    algorithm.config["parent_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        algorithm.load_checkpoint(str(artifact), platform_model_id="1013548")


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


def test_p35_monitor_contract_tracks_registration_and_data_age():
    agent = SimpleNamespace()
    metrics = {name: 1.0 for name in P35_REQUIRED_MONITOR_METRICS}
    first = _monitor_contract_metrics(agent, metrics, 10.0)
    assert first["p35_monitor_registered_metric_count"] == first[
        "p35_monitor_expected_metric_count"
    ]
    assert first["p35_monitor_empty_metric_count"] == 0.0
    missing = dict(metrics)
    missing.pop("p35_reward_posture")
    second = _monitor_contract_metrics(agent, missing, 14.0)
    assert second["p35_monitor_registered_metric_count"] == second[
        "p35_monitor_expected_metric_count"
    ] - 1.0
    assert second["p35_monitor_longest_data_age_s"] == pytest.approx(4.0)


def test_coordinator_keeps_std_frozen_and_scales_low_lrs():
    visual = object.__new__(AlgorithmVisualPPO)
    visual.anchor_session_elapsed_hours = 0.0
    visual.current_phase = ""
    actor = torch.nn.Sequential(
        torch.nn.Linear(1, 2), torch.nn.ELU(), torch.nn.Linear(2, 1)
    )
    visual.actor_critic = SimpleNamespace(
        std=torch.nn.Parameter(torch.ones(1)), actor=actor
    )
    visual._set_trainable_phase = lambda phase: visual.actor_critic.std.requires_grad_(True)
    visual.optimizer = torch.optim.Adam(
        [
            {"params": list(actor.parameters()), "lr": 1e-5, "name": "actor"},
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
    assert [group["lr"] for group in visual.optimizer.param_groups] == pytest.approx([0.0, 0.0, 3e-5])
    joint.update_clock(4500.0)
    assert [group["lr"] for group in visual.optimizer.param_groups] == pytest.approx([3e-6, 1.5e-6, 3e-5])
    assert all(not parameter.requires_grad for parameter in actor[0].parameters())
    assert all(parameter.requires_grad for parameter in actor[-1].parameters())


def test_high_adapter_updates_every_second_policy_rollout():
    config = {"high_update_interval": 2}
    assert not _high_adapter_update_due(1, config)
    assert _high_adapter_update_due(2, config)
    assert not _high_adapter_update_due(3, config)
    assert _high_adapter_update_due(4, config)


def test_adapter_update_metrics_distinguish_attempts_applied_and_skipped():
    results = iter(
        (
            {"adapter_loss": 0.2, "adapter_updates": 1.0},
            {"adapter_loss": 0.0, "adapter_updates": 0.0},
        )
    )
    replay_ratios = []
    agent = SimpleNamespace(
        algorithm=SimpleNamespace(session_effective_seconds=15000.0),
        response_aux_buffer=SimpleNamespace(
            set_p3_replay_ratios=lambda *values: replay_ratios.append(values)
        ),
        high_level_algorithm=SimpleNamespace(
            update_adapter_after_policy=lambda: next(results)
        )
    )
    metrics = _run_adapter_updates(agent, 2)
    assert replay_ratios == [(0.60, 0.25, 0.15)]
    assert metrics["adapter_update_attempts"] == 2.0
    assert metrics["adapter_updates"] == 1.0
    assert metrics["adapter_skipped_updates"] == 1.0


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
    raw = {"phase_label": "gaitwarm"}
    state = {"compound_schedule_phase": "gaitwarm"}
    assert _validate_resume_phase(raw, state, 1800.0) == "gaitwarm"
    with pytest.raises(RuntimeError, match="phase/time mismatch"):
        _validate_resume_phase(
            {"phase_label": "stable"},
            {"compound_schedule_phase": "stable"},
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
    terminal[0, 15] = 3.91
    terminal[0, 24] = 1
    terminal[0, 25] = 1
    metrics = coordinator.observe_subgoal_and_terminal(terminal)
    assert metrics["p3_standard_success_count"].item() == 1.0
    assert metrics["p3_platform_success_count"].item() == 1.0
    assert metrics["p3_proxy_platform_agreement_count"].item() == 1.0
    assert metrics["p3_joint_success_count"].item() == 1.0
    assert coordinator.episode_subgoal_successes.tolist() == [0, 0]


def test_terminal_safe_pose_is_not_reused_as_next_episode_origin():
    coordinator = object.__new__(AlgorithmP3StandardJoint)
    coordinator.config = {}
    terminal = torch.zeros(1, p2_contract.WORKER_AUX_DIM)
    terminal[0, 15] = 3.91
    terminal[0, 24] = 1
    terminal[0, 25] = 1
    first = coordinator.observe_subgoal_and_terminal(terminal)
    assert first["p3_standard_success_count"].item() == 1.0
    coordinator.sync_episode_origins(terminal)
    assert not bool(coordinator.episode_origin_valid[0])

    next_episode = torch.zeros(1, p2_contract.WORKER_AUX_DIM)
    next_episode[0, 15:17] = torch.tensor([0.25, -0.10])
    second = coordinator.observe_subgoal_and_terminal(next_episode)
    assert second["p3_standard_success_count"].item() == 0.0
    assert coordinator.episode_origin_valid.tolist() == [True]
    assert torch.allclose(
        coordinator.episode_origin_xy[0], torch.tensor([0.25, -0.10])
    )
    assert second["p3_radial_distance_mean"].item() == pytest.approx(0.0)


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


def test_stair_memory_schedule_only_changes_adapter_learning_rate():
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
    actor_lrs = [group["lr"] for group in algorithm.actor_optimizer.param_groups]
    critic_lr = algorithm.critic_optimizer.param_groups[0]["lr"]

    algorithm._apply_training_schedule(0.0)
    assert algorithm.response_optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-5)

    algorithm._apply_training_schedule(21600.0)
    assert algorithm.current_phase == "pushfull"
    assert algorithm.response_optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-5)
    assert [group["lr"] for group in algorithm.actor_optimizer.param_groups] == actor_lrs
    assert algorithm.critic_optimizer.param_groups[0]["lr"] == critic_lr

    algorithm._apply_training_schedule(27000.0)
    assert algorithm.current_phase == "stable"
    assert algorithm.response_optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-4)
    assert [group["lr"] for group in algorithm.actor_optimizer.param_groups] == actor_lrs
    assert algorithm.critic_optimizer.param_groups[0]["lr"] == critic_lr


def test_short_high_adaptation_restores_parent_critic_optimizer_and_statistics():
    source_critic = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.Adam(source_critic.parameters(), lr=3.0e-4)
    source_critic(torch.ones(2, 2)).square().mean().backward()
    source_optimizer.step()

    target_critic = torch.nn.Linear(2, 1)
    target_optimizer = torch.optim.Adam(target_critic.parameters(), lr=3.0e-4)

    def load_leaf(container, name, module, **_kwargs):
        module.load_state_dict(container[name]["state_dict"], strict=True)

    high = SimpleNamespace(
        critic=target_critic,
        critic_optimizer=target_optimizer,
        _load_leaf=load_leaf,
        return_statistics={},
        actor_gradient_steps=0,
        critic_gradient_steps=0,
    )
    joint = object.__new__(AlgorithmP3StandardJoint)
    joint.high_algorithm = high
    bundle = {
        "modules": {
            "high_level": {
                "critic": {"state_dict": source_critic.state_dict()},
            }
        },
        "optimizers": {"high_level_critic": source_optimizer.state_dict()},
        "training_states": {
            "high_level": {
                "actor_gradient_steps": 123,
                "critic_gradient_steps": 456,
                "return_statistics": {
                    "count": 789,
                    "mean": 1.25,
                    "m2": 3.5,
                    "value_normalization_enabled": False,
                },
            }
        },
    }
    joint._restore_high_value_state_for_short_adaptation(bundle)

    for source, restored in zip(
        source_critic.parameters(), target_critic.parameters()
    ):
        assert torch.equal(source, restored)
        source_state = source_optimizer.state[source]
        restored_state = target_optimizer.state[restored]
        assert torch.equal(source_state["exp_avg"], restored_state["exp_avg"])
        assert torch.equal(source_state["exp_avg_sq"], restored_state["exp_avg_sq"])
    assert high.return_statistics["count"] == 789
    assert high.return_statistics["mean"] == pytest.approx(1.25)
    assert high.return_statistics["m2"] == pytest.approx(3.5)
    assert high.actor_gradient_steps == 123
    assert high.critic_gradient_steps == 456


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
    assert torch.equal(result["frontier_shaping"], torch.zeros(2))
    assert torch.equal(result["frame_safety"], torch.ones(2))
