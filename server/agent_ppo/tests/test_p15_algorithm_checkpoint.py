#!/usr/bin/env python3

import copy

import torch

from agent_ppo.algorithm.algorithm_p15_response import AlgorithmP15Response
from agent_ppo.feature.response_aux_buffer import ResponseAuxBuffer
from agent_ppo.model.response_adapter import CommandResponseAdapter
from agent_ppo.model.visual_actor_critic import VisualActorCritic


def _algorithm(seed=1):
    torch.manual_seed(seed)
    model = VisualActorCritic(
        num_proprio=45,
        num_scan=256,
        depth_shape=(180, 320, 1),
        latent_dim=32,
        cnn_output_dim=32,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        num_critic_obs=316,
        num_actions=12,
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
        init_noise_std=0.15,
    )
    anchor_encoder = copy.deepcopy(model.vision_encoder)
    anchor_actor = copy.deepcopy(model.actor)
    for parameter in model.vision_encoder.cnn.parameters():
        parameter.requires_grad_(False)
    low_optimizer = torch.optim.Adam(
        [
            {
                "params": [*model.actor.parameters(), model.std],
                "lr": 1.0e-5,
                "name": "actor",
            },
            {
                "params": [
                    *model.vision_encoder.rnn.parameters(),
                    *model.vision_encoder.rnn_output_layer.parameters(),
                ],
                "lr": 5.0e-6,
                "name": "lstm",
            },
            {"params": model.critic.parameters(), "lr": 1.0e-4, "name": "critic"},
        ]
    )
    adapter = CommandResponseAdapter()
    adapter_optimizer = torch.optim.Adam(adapter.parameters(), lr=3.0e-4)
    adapter_scheduler = torch.optim.lr_scheduler.LambdaLR(
        adapter_optimizer, lr_lambda=lambda _step: 1.0
    )
    low_level_scheduler = torch.optim.lr_scheduler.LambdaLR(
        low_optimizer, lr_lambda=lambda _step: 1.0
    )
    algorithm = AlgorithmP15Response(
        model=model,
        anchor_encoder=anchor_encoder,
        anchor_actor=anchor_actor,
        optimizer=low_optimizer,
        sequence_length=16,
        schedule_mode="p15_response_adapter_v1",
        run_name="p15resp8h",
        source_parent_model_id=34728,
        anchor_schedule_hours=None,
        action_anchor_schedule=None,
        latent_anchor_schedule=None,
        anchor_phase_labels=None,
        anchor_phase_end_hours=None,
        critic_warmup_learning_rate=3.0e-4,
        task_end_hours=8.0,
        warning_only_safety=True,
        max_anchor_action_mse=0.05,
        max_hard_termination_delta=0.02,
        command_anchor_action=0.35,
        command_anchor_latent=0.10,
        response_adapter=adapter,
        response_optimizer=adapter_optimizer,
        response_scheduler=adapter_scheduler,
        low_level_scheduler=low_level_scheduler,
        response_buffer=ResponseAuxBuffer(4, "cpu", sequence_length=2),
        response_config={"batch_envs": 2, "seed": seed},
        p15_config={
            "command_schedule": {"target_period_frames": 10},
            "feedback_profile": {"version": "test_feedback_profile"},
        },
        device="cpu",
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1.0e-5,
        max_grad_norm=1.0,
        num_mini_batches=1,
        num_learning_epochs=1,
        desired_kl=None,
    )
    return algorithm


def test_optimizer_sets_are_disjoint_and_calibration_freezes_low_level():
    algorithm = _algorithm()
    low_ids = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    adapter_ids = {
        id(parameter)
        for group in algorithm.response_optimizer.param_groups
        for parameter in group["params"]
    }
    assert low_ids.isdisjoint(adapter_ids)
    algorithm._set_trainable_phase("responsecalib")
    assert not any(parameter.requires_grad for parameter in algorithm.actor_critic.actor.parameters())
    assert not any(parameter.requires_grad for parameter in algorithm.actor_critic.critic.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in algorithm.actor_critic.vision_encoder.rnn.parameters()
    )
    assert all(parameter.requires_grad for parameter in algorithm.response_adapter.parameters())


def test_schema2_round_trip_restores_both_training_states(tmp_path):
    algorithm = _algorithm(seed=4)
    adapter_loss = sum(parameter.square().mean() for parameter in algorithm.response_adapter.parameters())
    algorithm.response_optimizer.zero_grad()
    adapter_loss.backward()
    algorithm.response_optimizer.step()
    algorithm.adapter_iteration = 9
    algorithm.adapter_gradient_steps = 7
    algorithm.low_level_gradient_steps = 11
    algorithm.low_level_skipped_nonfinite = 2
    algorithm.current_iteration = 34728
    algorithm.anchor_session_elapsed_hours = 2.5
    algorithm.elapsed_training_hours = 6.5
    algorithm.total_env_steps = 123456
    for step in range(60):
        aux = torch.zeros(4, 30)
        aux[:, 3] = 0.3
        aux[:, 9] = 1.0
        aux[:, 12] = 0.2
        aux[:, 15] = step * 0.004
        aux[:, 26] = 1.0
        algorithm.response_buffer.append(aux, torch.zeros(4, dtype=torch.bool))
    assert algorithm.response_buffer.ready
    path = tmp_path / "model.ckpt-responsefull-40000.pkl"
    checksum = algorithm.save_training_bundle(
        str(path),
        platform_model_id=40000,
        phase_label="responsefull",
        model_spec={
            "proprio_dim": 45,
            "scan_dim": 256,
            "depth_height": 180,
            "depth_width": 320,
            "depth_channels": 1,
            "latent_dim": 32,
            "action_dim": 12,
            "goal_dim": 0,
        },
    )
    payload = torch.load(path, weights_only=False, map_location="cpu")
    assert len(checksum) == 64
    assert payload["schema_version"] == 2
    assert payload["modules"]["high_level"]["component_status"] == "adapter_only"
    assert "visual_ppo" in payload["optimizers"]
    assert "response_adapter" in payload["optimizers"]
    assert isinstance(payload["schedulers"]["low_level"], dict)
    assert payload["training_states"]["response_adapter"]["iteration"] == 9
    assert payload["contracts"]["command"]["runtime_config"] == {
        "target_period_frames": 10
    }
    feedback_contract = payload["contracts"]["feedback"]
    assert feedback_contract["profile"]["version"] == "test_feedback_profile"
    assert feedback_contract["implementation"]["module"] == (
        "agent_ppo.feature.feedback_emulator"
    )
    assert len(feedback_contract["implementation"]["sha256"]) == 64

    restored = _algorithm(seed=99)
    load_mode = restored.load_training_bundle(
        str(path),
        expected_spec={
            "proprio_dim": 45,
            "scan_dim": 256,
            "latent_dim": 32,
            "action_dim": 12,
            "goal_dim": 0,
        },
        env_seed=99,
    )
    assert load_mode == "weights_optim_rng_resume_with_history_reset"
    assert restored.current_iteration == 34728
    assert restored.anchor_session_elapsed_hours == 2.5
    assert restored.adapter_iteration == 9
    assert restored.adapter_gradient_steps == 7
    assert restored.low_level_gradient_steps == 11
    assert restored.low_level_skipped_nonfinite == 2
    assert restored.total_env_steps == 123456
    assert restored.response_buffer.ready
    for expected, actual in zip(
        algorithm.response_adapter.parameters(), restored.response_adapter.parameters()
    ):
        assert torch.equal(expected, actual)


def test_schema2_missing_scheduler_and_rng_is_explicit_warm_start(tmp_path):
    source = _algorithm(seed=12)
    path = tmp_path / "model.ckpt-responsebase-41000.pkl"
    source.save_training_bundle(
        str(path),
        platform_model_id=41000,
        phase_label="responsebase",
        model_spec={
            "proprio_dim": 45,
            "scan_dim": 256,
            "depth_height": 180,
            "depth_width": 320,
            "depth_channels": 1,
            "latent_dim": 32,
            "action_dim": 12,
            "goal_dim": 0,
        },
    )
    payload = torch.load(path, weights_only=False, map_location="cpu")
    payload["schedulers"].pop("response_adapter", None)
    payload["training_states"]["response_adapter"].pop("rng_state", None)
    torch.save(payload, path)

    restored = _algorithm(seed=13)
    load_mode = restored.load_training_bundle(
        str(path),
        expected_spec={
            "proprio_dim": 45,
            "scan_dim": 256,
            "latent_dim": 32,
            "action_dim": 12,
            "goal_dim": 0,
        },
        env_seed=13,
    )
    assert load_mode == "warm_start"


def test_explicit_low_level_only_load_ignores_response_adapter(tmp_path):
    source = _algorithm(seed=21)
    path = tmp_path / "model.ckpt-responsefull-42000.pkl"
    source.save_training_bundle(
        str(path),
        platform_model_id=42000,
        phase_label="responsefull",
        model_spec={
            "proprio_dim": 45,
            "scan_dim": 256,
            "depth_height": 180,
            "depth_width": 320,
            "depth_channels": 1,
            "latent_dim": 32,
            "action_dim": 12,
            "goal_dim": 0,
        },
    )
    restored = _algorithm(seed=22)
    adapter_before = {
        key: value.clone() for key, value in restored.response_adapter.state_dict().items()
    }
    load_mode = restored.load_low_level_only_bundle(
        str(path),
        expected_spec={
            "proprio_dim": 45,
            "scan_dim": 256,
            "latent_dim": 32,
            "action_dim": 12,
            "goal_dim": 0,
        },
        env_seed=22,
        restore_optimizer=False,
    )
    assert load_mode == "low_level_only_warm_start"
    for key, value in restored.response_adapter.state_dict().items():
        assert torch.equal(value, adapter_before[key])


def test_schema1_command_parent_restores_low_level_and_starts_adapter_fresh(tmp_path):
    source = _algorithm(seed=8)
    source.current_iteration = 34728
    source_path = tmp_path / "model.ckpt-responsebase-34728.pkl"
    source.save_training_bundle(
        str(source_path),
        platform_model_id=34728,
        phase_label="responsebase",
        model_spec={
            "proprio_dim": 45,
            "scan_dim": 256,
            "depth_height": 180,
            "depth_width": 320,
            "depth_channels": 1,
            "latent_dim": 32,
            "action_dim": 12,
            "goal_dim": 0,
        },
    )
    parent = torch.load(source_path, weights_only=False, map_location="cpu")
    parent["schema_version"] = 1
    parent["stage_type"] = "standard_visual_ppo"
    parent["training_state"]["schedule_mode"] = "visual_command_generalization_v1"
    parent["training_state"]["current_iteration"] = 34728
    parent["modules"].pop("high_level", None)
    parent["optimizers"].pop("response_adapter", None)
    parent.pop("schedulers", None)
    parent.pop("training_states", None)
    parent_path = tmp_path / "model.ckpt-commandfull-34728.pkl"
    torch.save(parent, parent_path)

    restored = _algorithm(seed=9)
    load_mode = restored.load_training_bundle(
        str(parent_path),
        expected_spec={
            "proprio_dim": 45,
            "scan_dim": 256,
            "latent_dim": 32,
            "action_dim": 12,
            "goal_dim": 0,
        },
        env_seed=9,
    )
    assert load_mode == "transition_resume"
    assert restored.current_iteration == 34728
    assert restored.adapter_iteration == 0
    assert restored.anchor_session_elapsed_hours == 0.0
