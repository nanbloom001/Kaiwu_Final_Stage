#!/usr/bin/env python3
"""Verify P2-to-P3 warm start and P3 exact resume with a real bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import tomllib
from pathlib import Path

import torch

import agent_ppo.tests._nav_test_stubs  # noqa: F401
from agent_ppo.algorithm.algorithm_p3_standard_joint import (
    AlgorithmP3HighPPO,
    AlgorithmP3StandardJoint,
)
from agent_ppo.algorithm.algorithm_visual_ppo import AlgorithmVisualPPO
from agent_ppo.feature.p2_response_buffer import P2ResponseAuxBuffer
from agent_ppo.feature import p3_contract
from agent_ppo.feature.definition import RolloutStorage
from agent_ppo.model.p2_high_level import (
    NavigationEncoder,
    NavigationSafetyHead,
    P2NavigationActor,
    P2NavigationCritic,
)
from agent_ppo.model.response_adapter import CommandResponseAdapter
from agent_ppo.model.visual_actor_critic import VisualActorCritic


def _optimizer_state_snapshot(optimizer, parameters) -> str:
    digest = hashlib.sha256()
    for index, parameter in enumerate(parameters):
        digest.update(str(index).encode("ascii"))
        state = optimizer.state.get(parameter, {})
        for key in sorted(state, key=str):
            digest.update(str(key).encode("utf-8"))
            value = state[key]
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _nested_state_equal(left, right) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_state_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _build_joint(config: dict) -> AlgorithmP3StandardJoint:
    device = torch.device("cpu")
    low = VisualActorCritic(
        num_proprio=45,
        num_scan=256,
        depth_shape=(180, 320, 1),
        latent_dim=32,
        cnn_output_dim=32,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        num_critic_obs=323,
        num_actions=12,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=0.15,
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": low.actor.parameters(), "lr": 1e-5, "name": "actor"},
            {
                "params": [
                    *low.vision_encoder.rnn.parameters(),
                    *low.vision_encoder.rnn_output_layer.parameters(),
                ],
                "lr": 5e-6,
                "name": "lstm",
            },
            {"params": low.critic.parameters(), "lr": 1e-4, "name": "critic"},
        ]
    )
    low_algorithm = AlgorithmVisualPPO(
        model=low,
        anchor_encoder=copy.deepcopy(low.vision_encoder),
        anchor_actor=copy.deepcopy(low.actor),
        optimizer=optimizer,
        sequence_length=int(config.get("tbptt_sequence_length", 128)),
        schedule_mode="p3_stair_memory_v1",
        run_name=str(config.get("run_name", "p3-parent-smoke")),
        source_parent_model_id=config.get("parent_model_id"),
        anchor_schedule_hours=[],
        action_anchor_schedule=None,
        latent_anchor_schedule=None,
        anchor_phase_labels=None,
        anchor_phase_end_hours=[],
        critic_warmup_learning_rate=None,
        task_end_hours=float(config.get("task_end_hours", 8.0)),
        warning_only_safety=True,
        max_anchor_action_mse=0.1,
        max_hard_termination_delta=0.05,
        max_action_amplitude=6.0,
        command_anchor_action=0.2,
        command_anchor_latent=0.05,
        device=device,
        logger=None,
        monitor=None,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        num_mini_batches=1,
        num_learning_epochs=1,
        desired_kl=None,
    )
    low_algorithm.init_storage(
        2,
        int(config.get("num_steps_per_env", 128)),
        actor_obs_shape=(77,),
        critic_obs_shape=(323,),
        action_shape=(12,),
        device=device,
    )
    low_algorithm.initialize_recurrent_states(2)
    high_algorithm = AlgorithmP3HighPPO(
        low_level_encoder=low.vision_encoder,
        low_level_actor=low.actor,
        navigation_encoder=NavigationEncoder(),
        safety_head=NavigationSafetyHead(),
        actor=P2NavigationActor(),
        critic=P2NavigationCritic(),
        response_adapter=CommandResponseAdapter(),
        response_buffer=P2ResponseAuxBuffer(
            2,
            "cpu",
            capacity_steps=64,
            sequence_length=16,
            burn_in_steps=8,
        ),
        num_envs=2,
        device=device,
        config=config,
        logger=None,
        monitor=None,
    )
    return AlgorithmP3StandardJoint(
        low_algorithm=low_algorithm,
        high_algorithm=high_algorithm,
        config=config,
        logger=None,
    )


def _run_low_update(joint: AlgorithmP3StandardJoint) -> dict[str, object]:
    algorithm = joint.low_algorithm
    joint.action_smooth_aux.begin_rollout(
        p3_contract.action_smooth_training_fraction(
            joint.session_effective_seconds,
            mapping_valid=joint.mirror_mapping_valid,
        )
    )
    joint.memory_aux.begin_rollout(
        p3_contract.memory_training_fraction(joint.session_effective_seconds)
    )
    storage = algorithm.storage
    storage.clear()
    torch.manual_seed(7)
    actor_before = {
        name: value.detach().clone()
        for name, value in algorithm.actor_critic.actor.state_dict().items()
    }
    critic_before = {
        name: value.detach().clone()
        for name, value in algorithm.actor_critic.critic.state_dict().items()
    }
    policy_parameters = [
        *algorithm.actor_critic.actor.parameters(),
        *algorithm.actor_critic.vision_encoder.rnn.parameters(),
        *algorithm.actor_critic.vision_encoder.rnn_output_layer.parameters(),
    ]
    optimizer_before = _optimizer_state_snapshot(algorithm.optimizer, policy_parameters)
    for tick in range(storage.num_transitions_per_env):
        obs = torch.zeros(2, 57901)
        obs[:, :45] = torch.randn(2, 45) * 0.05
        critic_obs = torch.randn(2, 323) * 0.05
        hidden = algorithm.rollout_hidden_state()
        with torch.no_grad():
            anchor_action, anchor_latent = algorithm.anchor_inference(obs)
            depth = obs[:, 301:].reshape(2, 180, 320, 1)
            clean_feature = algorithm.actor_critic.vision_encoder.cnn(depth)
            clean_compact = torch.cat(
                (obs[:, :45], clean_feature), dim=-1
            )
            compact_obs = clean_compact.clone()
            changed = torch.zeros(2, dtype=torch.bool)
            if joint.memory_aux.scale > 0.0:
                compact_obs[:, 45:] = 0.0
                changed.fill_(True)
            joint.memory_aux.store(tick, clean_compact, changed)
            actions = algorithm.actor_critic.act(compact_obs)
            values = algorithm.actor_critic.evaluate(critic_obs)
            log_prob = algorithm.actor_critic.get_actions_log_prob(actions)
        transition = RolloutStorage.Transition()
        transition.observations = compact_obs.detach()
        transition.critic_observations = critic_obs
        transition.actions = actions
        transition.values = values
        transition.actions_log_prob = log_prob
        transition.action_mean = algorithm.actor_critic.action_mean.detach()
        transition.action_sigma = algorithm.actor_critic.action_std.detach()
        transition.hidden_states = hidden
        transition.anchor_actions = anchor_action.detach()
        transition.anchor_latents = anchor_latent.detach()
        transition.anchor_weights = torch.ones(2, 1)
        transition.rewards = torch.full((2,), 0.02 + 0.001 * tick)
        transition.dones = torch.zeros(2, dtype=torch.bool)
        transition.hard_terminations = torch.zeros(2, dtype=torch.bool)
        storage.add_transitions(transition)
    with torch.no_grad():
        last_values = algorithm.actor_critic.evaluate(
            torch.randn(2, 323) * 0.05
        )
    storage.compute_returns(last_values, algorithm.gamma, algorithm.lam)
    metrics = algorithm.learn(
        joint.session_effective_seconds / 3600.0,
        phase_override=joint.current_phase,
    )
    joint.low_updates += 1
    joint.note_low_level_update()
    actor_changed = any(
        not torch.equal(value, actor_before[name])
        for name, value in algorithm.actor_critic.actor.state_dict().items()
    )
    critic_changed = any(
        not torch.equal(value, critic_before[name])
        for name, value in algorithm.actor_critic.critic.state_dict().items()
    )
    policy_optimizer_changed = (
        optimizer_before
        != _optimizer_state_snapshot(algorithm.optimizer, policy_parameters)
    )
    storage.clear()
    return {
        "applied_updates": metrics["applied_updates"],
        "actor_changed": actor_changed,
        "critic_changed": critic_changed,
        "policy_optimizer_changed": policy_optimizer_changed,
        "action_smooth_scale": metrics["action_smooth_scale"],
        "action_aux_combined_gradient_ratio": metrics[
            "action_aux_combined_gradient_ratio"
        ],
        "memory_loss": metrics["memory_loss"],
        "memory_gradient_ratio": metrics["memory_gradient_ratio"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)["p3_standard_joint"]
    joint = _build_joint(config)
    parent_model_id = str(config.get("parent_model_id", "")) or None
    warm_start = joint.load_checkpoint(
        args.checkpoint,
        platform_model_id=parent_model_id,
    )
    parent = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    expected_std = parent["modules"]["low_level"]["action_distribution"][
        "state_dict"
    ]["std"]
    if not torch.equal(joint.low_algorithm.actor_critic.std.detach().cpu(), expected_std):
        raise AssertionError("P3 warm start did not restore parent low action std")
    if not all(
        torch.equal(value, joint.low_algorithm.actor_critic.actor.state_dict()[name])
        for name, value in joint.low_algorithm.anchor_actor.state_dict().items()
    ):
        raise AssertionError("P3 stair-memory anchor is not the selected F2 actor")
    parent_high = parent["modules"]["high_level"]
    high_critic_restored = all(
        torch.equal(value.detach().cpu(), parent_high["critic"]["state_dict"][name])
        for name, value in joint.high_algorithm.critic.state_dict().items()
    )
    high_critic_optimizer_restored = _nested_state_equal(
        joint.high_algorithm.critic_optimizer.state_dict()["state"],
        parent["optimizers"]["high_level_critic"]["state"],
    )
    high_return_statistics_restored = (
        joint.high_algorithm.return_statistics
        == parent["training_states"]["high_level"]["return_statistics"]
    )
    high_actor_optimizer_before = copy.deepcopy(
        joint.high_algorithm.actor_optimizer.state_dict()
    )
    high_critic_optimizer_before = copy.deepcopy(
        joint.high_algorithm.critic_optimizer.state_dict()
    )
    frozen_low_update = _run_low_update(joint)
    joint.update_clock(901.0)
    trainable_low_update = _run_low_update(joint)
    high_actor_optimizer_unchanged = _nested_state_equal(
        high_actor_optimizer_before,
        joint.high_algorithm.actor_optimizer.state_dict(),
    )
    high_critic_optimizer_unchanged = _nested_state_equal(
        high_critic_optimizer_before,
        joint.high_algorithm.critic_optimizer.state_dict(),
    )
    with tempfile.TemporaryDirectory(prefix="p3-parent-smoke-") as temp_dir:
        saved = Path(temp_dir) / "model.ckpt-lowbase-1.pkl"
        joint.save_training_bundle(saved, platform_model_id="1")
        exact_resume = joint.load_checkpoint(saved, platform_model_id="1")
    result = {
        "warm_start": warm_start,
        "exact_resume": exact_resume,
        "phase": joint.current_phase,
        "parent_loaded": joint.parent_loaded,
        "low_action_std_restored": True,
        "high_critic_restored": high_critic_restored,
        "high_critic_optimizer_restored": high_critic_optimizer_restored,
        "high_return_statistics_restored": high_return_statistics_restored,
        "high_actor_optimizer_unchanged": high_actor_optimizer_unchanged,
        "high_critic_optimizer_unchanged": high_critic_optimizer_unchanged,
        "low_level_digest_present": (
            joint.high_algorithm.low_level_state_digest is not None
        ),
        "session_effective_seconds": joint.session_effective_seconds,
        "frozen_low_update": frozen_low_update,
        "trainable_low_update": trainable_low_update,
    }
    print(json.dumps(result, sort_keys=True))
    if (
        not result["parent_loaded"]
        or not result["high_critic_restored"]
        or not result["high_critic_optimizer_restored"]
        or not result["high_return_statistics_restored"]
        or not result["high_actor_optimizer_unchanged"]
        or not result["high_critic_optimizer_unchanged"]
        or result["session_effective_seconds"] != 901.0
        or frozen_low_update["applied_updates"] <= 0
        or frozen_low_update["actor_changed"]
        or frozen_low_update["policy_optimizer_changed"]
        or not frozen_low_update["critic_changed"]
        or trainable_low_update["applied_updates"] <= 0
        or not trainable_low_update["actor_changed"]
        or not trainable_low_update["policy_optimizer_changed"]
        or not trainable_low_update["critic_changed"]
        or trainable_low_update["action_smooth_scale"] != 0.0
        or trainable_low_update["memory_loss"] <= 0.0
        or trainable_low_update["memory_gradient_ratio"] > 0.03 + 1.0e-6
        or trainable_low_update["action_aux_combined_gradient_ratio"] > 0.05 + 1.0e-6
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
