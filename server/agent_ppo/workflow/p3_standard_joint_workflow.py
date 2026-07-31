#!/usr/bin/env python3
"""P3 Standard joint recovery with independent 50 Hz and 5 Hz rollouts."""

from __future__ import annotations

import math
import os
import signal
import threading
import time

import torch

from agent_ppo.checkpoint_io import CheckpointSaveError
from agent_ppo.conf.conf import Config
from agent_ppo.feature import nav_contract, p2_contract, p3_contract
from agent_ppo.feature.p3_gait import mirror_proprio
from agent_ppo.feature.definition import RolloutStorage
from agent_ppo.feature.p2_response_buffer import (
    patch_owned_commands,
    split_p2_transport,
)
from agent_ppo.workflow.p2_nav_ppo_workflow import (
    _extract_step,
    _frame_done_masks,
    _resolve_terminal_outcome,
)


def _as_device(value, device):
    return torch.as_tensor(value, device=device)


def _reset_env(env, agent, usr_conf):
    runtime_conf = p3_contract.materialize_environment_config(
        usr_conf, agent.algorithm.session_effective_seconds
    )
    data = env.reset(runtime_conf)
    if data is None:
        raise RuntimeError("P3 env.reset returned None")
    obs, critic_wire = data
    obs = _as_device(obs, agent.device).clone()
    critic_wire = _as_device(critic_wire, agent.device).clone()
    if obs.shape != (agent.num_envs, p3_contract.POLICY_OBS_DIM):
        raise ValueError(f"P3 reset policy shape drift: {tuple(obs.shape)}")
    if critic_wire.shape != (agent.num_envs, p3_contract.P3_PRIVILEGED_WIRE_DIM):
        raise ValueError(f"P3 reset critic wire shape drift: {tuple(critic_wire.shape)}")
    agent._p3_extra = critic_wire[:, p2_contract.PRIVILEGED_WIRE_DIM :].clone()
    agent.algorithm.update_runtime_terrain_size(agent._p3_extra[:, 0])
    agent.algorithm.mirror_mapping_valid = bool(
        (agent._p3_extra[:, p3_contract.JOINT_MAPPING_VALID_INDEX] > 0.5).all()
    )
    critic_wire = critic_wire[:, : p2_contract.PRIVILEGED_WIRE_DIM]
    agent.algorithm.reset_live_state()
    agent._p3_native_target_cmd = None
    agent._p3_native_command_epoch = None
    return obs, critic_wire


def _consume_p3_wire(agent, wire: torch.Tensor) -> torch.Tensor:
    if wire.shape != (agent.num_envs, p3_contract.P3_PRIVILEGED_WIRE_DIM):
        raise ValueError(f"P3 critic wire shape drift: {tuple(wire.shape)}")
    agent._p3_extra = wire[:, p2_contract.PRIVILEGED_WIRE_DIM :].clone()
    agent.algorithm.update_runtime_terrain_size(agent._p3_extra[:, 0])
    agent.algorithm.mirror_mapping_valid = bool(
        (agent._p3_extra[:, p3_contract.JOINT_MAPPING_VALID_INDEX] > 0.5).all()
    )
    return wire[:, : p2_contract.PRIVILEGED_WIRE_DIM]


def _requires_environment_rebuild(before_s: float, after_s: float) -> bool:
    return p3_contract.domain_randomization_index(
        before_s
    ) != p3_contract.domain_randomization_index(after_s)


def _high_adapter_update_due(high_updates: int, response_config: dict) -> bool:
    interval = max(1, int(response_config.get("high_update_interval", 2)))
    return int(high_updates) % interval == 0


def _run_adapter_updates(agent, count: int) -> dict[str, float]:
    attempts = max(0, int(count))
    latest: dict[str, float] = {}
    applied = 0.0
    for _ in range(attempts):
        current = agent.high_level_algorithm.update_adapter_after_policy()
        if isinstance(current, dict):
            latest.update(current)
            applied += float(current.get("adapter_updates", 0.0))
    latest["adapter_update_attempts"] = float(attempts)
    latest["adapter_updates"] = applied
    latest["adapter_skipped_updates"] = float(attempts) - applied
    return latest


def _native_command_epoch(agent, target_command: torch.Tensor) -> torch.Tensor:
    """Track asynchronous platform-native command changes per environment."""
    previous = getattr(agent, "_p3_native_target_cmd", None)
    epoch = getattr(agent, "_p3_native_command_epoch", None)
    if (
        not torch.is_tensor(previous)
        or previous.shape != target_command.shape
        or not torch.is_tensor(epoch)
        or epoch.shape != target_command.shape[:1]
    ):
        epoch = torch.zeros(
            target_command.shape[0], device=target_command.device, dtype=torch.long
        )
    else:
        previous = previous.to(target_command)
        epoch = epoch.to(device=target_command.device, dtype=torch.long)
        changed = (target_command - previous).abs().amax(dim=-1) > 1.0e-6
        epoch = epoch + changed.long()
    agent._p3_native_target_cmd = target_command.detach().clone()
    agent._p3_native_command_epoch = epoch.detach().clone()
    return epoch


def _storage_bytes(storage) -> int:
    """Return bytes owned by all tensor buffers in a rollout storage object."""
    total = 0
    seen = set()

    def visit(value):
        nonlocal total
        if torch.is_tensor(value):
            key = (value.device.type, value.device.index, value.data_ptr())
            if key not in seen:
                seen.add(key)
                total += value.numel() * value.element_size()
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for value in vars(storage).values():
        visit(value)
    return total


def _finalize_low_level_update(agent, metrics: dict) -> bool:
    """Advance the low-level version only after a real optimizer step."""
    if float(metrics.get("applied_updates", 0.0)) <= 0.0:
        return False
    agent.algorithm.low_updates += 1
    agent.algorithm.note_low_level_update()
    return True


def _advance_platform_lifecycle(agent) -> bool:
    """Advance Kaiwu's model-publication clock after one successful env step."""
    attempts = int(getattr(agent, "_p3_lifecycle_attempt_callbacks", 0)) + 1
    successes = int(getattr(agent, "_p3_lifecycle_success_callbacks", 0))
    failures = int(getattr(agent, "_p3_lifecycle_failure_callbacks", 0))
    agent._p3_lifecycle_attempt_callbacks = attempts
    logger = getattr(agent, "logger", None)
    if attempts == 1 and logger is not None:
        logger.info("[P3Lifecycle] first callback begin")
    try:
        # P3 gradients are updated inside this workflow. Agent.learn(None) is a
        # deliberate no-op whose platform wrapper advances train_global_step,
        # dump_model_freq and the training-health watchdog.
        agent.learn(list_sample_data=None)
    except CheckpointSaveError as exc:
        if logger is not None:
            logger.error(
                "[P3Lifecycle] checkpoint save failed inside platform callback; "
                f"stopping training: {exc}"
            )
        raise
    except Exception as exc:
        failures += 1
        agent._p3_lifecycle_failure_callbacks = failures
        if logger is not None and (failures == 1 or failures % 100 == 0):
            logger.error(
                "[P3Lifecycle] callback failed; training continues but model "
                "publication does not advance: "
                f"failures={failures} error={type(exc).__name__}: {exc}"
            )
        return False
    successes += 1
    agent._p3_lifecycle_success_callbacks = successes
    if attempts == 1 and logger is not None:
        logger.info("[P3Lifecycle] first callback complete")
    return True


def _low_policy_step(agent, proprio, depth, critic_obs):
    algorithm = agent.low_level_algorithm
    hidden = algorithm.rollout_hidden_state()
    actions = algorithm.actor_critic.act_from_proprio_depth(proprio, depth)
    anchor_action, anchor_latent = algorithm.anchor_inference_from_cnn_features(
        proprio,
        algorithm.actor_critic.last_cnn_features,
    )
    compact_obs = torch.cat(
        (
            proprio,
            algorithm.actor_critic.last_cnn_features,
        ),
        dim=-1,
    )
    values = algorithm.actor_critic.evaluate(critic_obs)
    log_prob = algorithm.actor_critic.get_actions_log_prob(actions)
    return {
        "obs": compact_obs,
        "actions": actions,
        "values": values,
        "log_prob": log_prob,
        "mean": algorithm.actor_critic.action_mean.detach(),
        "std": algorithm.actor_critic.action_std.detach(),
        "hidden": hidden,
        "anchor_action": anchor_action,
        "anchor_latent": anchor_latent,
    }


def _low_policy_action(agent, proprio, depth):
    """Run the frozen low-level executor without constructing PPO metadata."""
    return agent.low_level_algorithm.actor_critic.act_from_proprio_depth(
        proprio, depth
    )


def _record_adapter_frame(
    agent,
    critic_wire,
    target_command,
    exec_command,
    command_epoch,
    dones,
):
    _, aux = split_p2_transport(critic_wire)
    patched = patch_owned_commands(
        aux,
        target_command,
        exec_command,
        command_epoch,
    )
    agent.response_aux_buffer.append(
        patched[:, : p2_contract.RESPONSE_AUX_DIM],
        dones,
        current_segment=torch.zeros_like(dones, dtype=torch.float32),
    )


def _command_accumulator(device):
    return {
        "count": torch.zeros((), device=device),
        "target": torch.zeros(3, device=device),
        "exec": torch.zeros(3, device=device),
        "measured": torch.zeros(3, device=device),
        "true": torch.zeros(3, device=device),
        "tracking": torch.zeros(3, device=device),
        "positive": torch.zeros(3, device=device),
        "negative": torch.zeros(3, device=device),
        "gait_duty": torch.zeros(4, device=device),
        "gait_air": torch.zeros(4, device=device),
        "gait_frequency": torch.zeros(4, device=device),
        "gait_slip": torch.zeros(4, device=device),
        "gait_valid": torch.zeros((), device=device),
        "body_collision_mapping_valid": torch.zeros((), device=device),
        "feedback_valid": torch.zeros((), device=device),
        "feedback_age_s": torch.zeros((), device=device),
        "feedback_true_error": torch.zeros((), device=device),
        "terrain_columns": torch.zeros(20, device=device),
        "terrain_levels": torch.zeros(10, device=device),
        "outcomes": torch.zeros(3, device=device),
        "standard_successes": torch.zeros((), device=device),
        "platform_successes": torch.zeros((), device=device),
        "joint_successes": torch.zeros((), device=device),
        "proxy_platform_agreements": torch.zeros((), device=device),
        "m1_successes": torch.zeros((), device=device),
        "m2_successes": torch.zeros((), device=device),
        "radial_distance": torch.zeros((), device=device),
        "best_radial_distance": torch.zeros((), device=device),
        "m3_hold": torch.zeros((), device=device),
        "torque_samples": [],
        "mechanical_power": torch.zeros((), device=device),
        "gait_contact_onset": torch.zeros(4, device=device),
        "gait_impact": torch.zeros(4, device=device),
        "gait_touchdown_y": torch.zeros(4, device=device),
        "gait_stance": torch.zeros(4, device=device),
    }


def _accumulate_command(accumulator, target, exec_command, aux, extra=None):
    batch = float(target.shape[0])
    measured = aux[:, 6:9]
    true = aux[:, 12:15]
    accumulator["count"] += batch
    accumulator["target"] += target.sum(dim=0)
    accumulator["exec"] += exec_command.sum(dim=0)
    accumulator["measured"] += measured.sum(dim=0)
    accumulator["true"] += true.sum(dim=0)
    accumulator["tracking"] += torch.abs(exec_command - true).sum(dim=0)
    accumulator["positive"] += (target > 0.05).float().sum(dim=0)
    accumulator["negative"] += (target < -0.05).float().sum(dim=0)
    accumulator["gait_duty"] += aux[:, p2_contract.GAIT_DUTY_SLICE].sum(dim=0)
    accumulator["gait_air"] += aux[:, p2_contract.GAIT_MAX_AIR_SLICE].sum(dim=0)
    accumulator["gait_frequency"] += aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE].sum(dim=0)
    accumulator["gait_slip"] += aux[:, p2_contract.GAIT_SLIP_SPEED_SLICE].sum(dim=0)
    accumulator["gait_valid"] += (
        aux[:, p2_contract.GAIT_VALID_INDEX] > 0.5
    ).float().sum()
    accumulator["body_collision_mapping_valid"] += (
        aux[:, p2_contract.BODY_COLLISION_MAPPING_VALID_INDEX] > 0.5
    ).float().sum()
    accumulator["feedback_valid"] += aux[:, 9].clamp(0.0, 1.0).sum()
    accumulator["feedback_age_s"] += p2_contract.feedback_age_seconds(
        aux[:, 10:11], age_clip_s=0.8
    ).sum()
    accumulator["feedback_true_error"] += torch.linalg.vector_norm(
        measured - true, dim=1
    ).sum()
    if torch.is_tensor(extra) and extra.shape[1] == p3_contract.P3_WORKER_EXTRA_DIM:
        accumulator["torque_samples"].append(
            extra[:, p3_contract.JOINT_TORQUE_SLICE].abs()
        )
        accumulator["mechanical_power"] += extra[:, p3_contract.MECHANICAL_POWER_INDEX].sum()
        accumulator["gait_contact_onset"] += extra[:, p3_contract.GAIT_CONTACT_ONSET_SLICE].sum(0)
        accumulator["gait_impact"] += extra[:, p3_contract.GAIT_IMPACT_SPEED_SLICE].sum(0)
        accumulator["gait_touchdown_y"] += extra[:, p3_contract.GAIT_TOUCHDOWN_Y_SLICE].sum(0)
        accumulator["gait_stance"] += extra[:, p3_contract.GAIT_CONTINUOUS_STANCE_SLICE].sum(0)
    columns = aux[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX].round().long()
    levels = aux[:, p2_contract.PRE_STEP_TERRAIN_LEVEL_INDEX].round().long()
    valid_columns = (columns >= 0) & (columns < accumulator["terrain_columns"].numel())
    valid_levels = (levels >= 0) & (levels < accumulator["terrain_levels"].numel())
    accumulator["terrain_columns"] += torch.bincount(
        columns[valid_columns], minlength=accumulator["terrain_columns"].numel()
    ).to(accumulator["terrain_columns"])
    accumulator["terrain_levels"] += torch.bincount(
        levels[valid_levels], minlength=accumulator["terrain_levels"].numel()
    ).to(accumulator["terrain_levels"])


def _accumulate_outcomes(accumulator, aux):
    reset = aux[:, 24] > 0.5
    reason = aux[:, 25].round().long()
    for index, code in enumerate((1, 2, 3)):
        accumulator["outcomes"][index] += (reset & (reason == code)).float().sum()


def _accumulate_joint_success(agent, accumulator, aux):
    metrics = agent.algorithm.observe_subgoal_and_terminal(aux)
    batch = float(aux.shape[0])
    accumulator["standard_successes"] += metrics["p3_standard_success_count"]
    accumulator["platform_successes"] += metrics["p3_platform_success_count"]
    accumulator["joint_successes"] += metrics["p3_joint_success_count"]
    accumulator["proxy_platform_agreements"] += metrics[
        "p3_proxy_platform_agreement_count"
    ]
    accumulator["m1_successes"] += metrics["p3_m1_success_count"]
    accumulator["m2_successes"] += metrics["p3_m2_success_count"]
    accumulator["radial_distance"] += metrics["p3_radial_distance_mean"] * batch
    accumulator["best_radial_distance"] += metrics[
        "p3_best_radial_distance_mean"
    ] * batch
    accumulator["m3_hold"] += metrics["p3_m3_hold_share"] * batch
    return metrics


def _zero_inactive_high_commands(command, active: torch.Tensor) -> None:
    """Keep completed envs on a zero-command path until the next nav tick."""
    mask = active.to(device=command.exec_cmd.device, dtype=command.exec_cmd.dtype)
    mask = mask.unsqueeze(-1)
    command.active_target.mul_(mask)
    command.exec_cmd.mul_(mask)


def _command_metrics(accumulator):
    count = accumulator["count"].clamp_min(1.0)
    result = {}
    for axis, index in zip(("vx", "vy", "wz"), range(3)):
        for source in ("target", "exec", "measured", "true"):
            result[f"{source}_{axis}"] = accumulator[source][index] / count
        result[f"{axis}_tracking_abs_error"] = accumulator["tracking"][index] / count
        result[f"target_{axis}_positive_share"] = accumulator["positive"][index] / count
        result[f"target_{axis}_negative_share"] = accumulator["negative"][index] / count
    for index, leg in enumerate(("fl", "fr", "rl", "rr")):
        result[f"{leg}_duty_factor"] = accumulator["gait_duty"][index] / count
        result[f"{leg}_max_air_time"] = accumulator["gait_air"][index] / count
        result[f"{leg}_step_frequency"] = accumulator["gait_frequency"][index] / count
        result[f"{leg}_slip_speed"] = accumulator["gait_slip"][index] / count
    result["gait_window_valid"] = accumulator["gait_valid"] / count
    result["gait_sensor_mapping_valid"] = result["gait_window_valid"]
    result["body_collision_mapping_valid"] = (
        accumulator["body_collision_mapping_valid"] / count
    )
    result["feedback_valid"] = accumulator["feedback_valid"] / count
    result["feedback_age_s"] = accumulator["feedback_age_s"] / count
    result["feedback_true_velocity_error"] = accumulator["feedback_true_error"] / count
    terrain_total = accumulator["terrain_columns"].sum().clamp_min(1.0)
    level_total = accumulator["terrain_levels"].sum().clamp_min(1.0)
    for index in range(accumulator["terrain_columns"].numel()):
        result[f"p3_terrain_column_l{index}_share"] = (
            accumulator["terrain_columns"][index] / terrain_total
        )
    for index in range(accumulator["terrain_levels"].numel()):
        result[f"p3_terrain_level_l{index}_share"] = (
            accumulator["terrain_levels"][index] / level_total
        )
    for label, start, end in (
        ("slope", 0, 4),
        ("slope_inv", 4, 8),
        ("stairs", 8, 14),
        ("stairs_inv", 14, 20),
    ):
        result[f"p3_terrain_{label}_share"] = (
            accumulator["terrain_columns"][start:end].sum() / terrain_total
        )
    for index, label in enumerate(("completed", "failure", "timeout")):
        result[f"p3_window_{label}_count"] = accumulator["outcomes"][index]
    result["p3_standard_success_count"] = accumulator["standard_successes"]
    result["p3_platform_success_count"] = accumulator["platform_successes"]
    result["p3_joint_success_count"] = accumulator["joint_successes"]
    result["p3_proxy_platform_agreement_count"] = accumulator[
        "proxy_platform_agreements"
    ]
    result["p3_m1_success_count"] = accumulator["m1_successes"]
    result["p3_m2_success_count"] = accumulator["m2_successes"]
    result["p3_radial_distance_mean"] = accumulator["radial_distance"] / count
    result["p3_best_radial_distance_mean"] = (
        accumulator["best_radial_distance"] / count
    )
    result["p3_m3_hold_share"] = accumulator["m3_hold"] / count
    result["mechanical_power_mean"] = accumulator["mechanical_power"] / count
    if accumulator["torque_samples"]:
        torque = torch.cat(accumulator["torque_samples"], dim=0)
        for label, ids in (
            ("hip", (0, 3, 6, 9)),
            ("thigh", (1, 4, 7, 10)),
            ("calf", (2, 5, 8, 11)),
        ):
            values = torque[:, ids].reshape(-1)
            result[f"{label}_torque_p50"] = torch.quantile(values, 0.50)
            result[f"{label}_torque_p95"] = torch.quantile(values, 0.95)
            result[f"{label}_torque_max"] = values.max()
    for index, leg in enumerate(("fl", "fr", "rl", "rr")):
        result[f"{leg}_contact_onset_rate"] = accumulator["gait_contact_onset"][index] / count
        result[f"{leg}_impact_speed"] = accumulator["gait_impact"][index] / count
        result[f"{leg}_touchdown_y"] = accumulator["gait_touchdown_y"][index] / count
        result[f"{leg}_continuous_stance"] = accumulator["gait_stance"][index] / count
    keys = tuple(result)
    values = torch.stack([result[key].reshape(()) for key in keys]).detach().cpu().tolist()
    return dict(zip(keys, values))


def _collect_low_rollout(env, agent, obs, critic_wire, *, train_low):
    algorithm = agent.low_level_algorithm
    # Collection is inference-only.  AlgorithmVisualPPO.learn() switches the
    # recurrent path back to train mode before CUDA recurrent replay.
    algorithm.actor_critic.eval()
    storage = algorithm.storage
    storage.clear()
    reward_sum = torch.zeros((), device=agent.device)
    done_sum = torch.zeros((), device=agent.device)
    command_accumulator = _command_accumulator(agent.device)
    gait_fraction = p3_contract.gait_training_fraction(
        agent.algorithm.session_effective_seconds
    )
    if not agent.algorithm.mirror_mapping_valid:
        gait_fraction = 0.0
    if agent.algorithm.session_effective_seconds >= 900.0:
        agent.algorithm.gait_baseline.finalize()
    agent.algorithm.mirror_aux.begin_rollout(gait_fraction)
    gait_reward_sums = torch.zeros(3, device=agent.device)
    for _ in range(storage.num_transitions_per_env):
        critic_obs, aux = split_p2_transport(critic_wire)
        p0, p1 = nav_contract.POLICY_CMD_SLICE
        target_command = obs[:, p0:p1].clone()
        exec_command = target_command.clone()
        command_epoch = _native_command_epoch(agent, target_command)
        _accumulate_command(
            command_accumulator, target_command, exec_command, aux, agent._p3_extra
        )
        # The worker-owned native command is already present in proprio45. Only
        # the trainable PPO path needs an owned copy for transition storage;
        # Adapter calibration keeps a read-only view and avoids critic/anchor
        # work entirely.
        proprio = obs[:, :45].clone() if train_low else obs[:, :45]
        if train_low:
            proprio[
                :,
                nav_contract.POLICY_CMD_SLICE[0] : nav_contract.POLICY_CMD_SLICE[1],
            ] = exec_command.to(proprio)
        depth = obs[:, p3_contract.DEPTH_SLICE].reshape(
            obs.shape[0], p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, p2_contract.DEPTH_CHANNELS
        )
        with torch.no_grad():
            if train_low:
                critic_obs = critic_obs.clone()
                critic_obs[
                    :,
                    nav_contract.CRITIC_CMD_SLICE[0] : nav_contract.CRITIC_CMD_SLICE[1],
                ] = exec_command.to(critic_obs)
                step = _low_policy_step(agent, proprio, depth, critic_obs)
                actions = step["actions"]
                mirror_mask = agent.algorithm.mirror_aux.select_block(
                    storage.step, agent.num_envs
                )
                if bool(mirror_mask.any()):
                    mirror_ids = mirror_mask.nonzero(as_tuple=False).flatten()
                    mirror_proprio_values = mirror_proprio(proprio[mirror_ids])
                    mirror_features = algorithm.actor_critic.vision_encoder.cnn(
                        depth[mirror_ids].flip(dims=(2,))
                    )
                    agent.algorithm.mirror_aux.store(
                        storage.step,
                        mirror_mask,
                        torch.cat((mirror_proprio_values, mirror_features), dim=-1),
                    )
            else:
                step = None
                actions = _low_policy_action(agent, proprio, depth)

        transition = None
        if train_low:
            transition = RolloutStorage.Transition()
            transition.observations = step["obs"].detach()
            transition.critic_observations = critic_obs.detach()
            transition.actions = actions.detach()
            transition.values = step["values"].detach()
            transition.actions_log_prob = step["log_prob"].detach()
            transition.action_mean = step["mean"]
            transition.action_sigma = step["std"]
            transition.hidden_states = step["hidden"]
            transition.anchor_actions = step["anchor_action"].detach()
            transition.anchor_latents = step["anchor_latent"].detach()
            transition.anchor_weights = torch.ones(
                agent.num_envs, 1, device=agent.device
            )

        step_data = env.step(torch.clamp(actions, -6.0, 6.0))
        _, next_obs, rewards, terminated, truncated, infos, next_wire = _extract_step(
            step_data
        )
        next_obs = _as_device(next_obs, agent.device)
        next_wire = _consume_p3_wire(agent, _as_device(next_wire, agent.device))
        rewards = _as_device(rewards, agent.device).reshape(-1)
        next_aux = next_wire[:, p2_contract.CRITIC_OBS_DIM :]
        dones, timeouts = _frame_done_masks(
            terminated,
            truncated,
            infos,
            agent.device,
            worker_aux=next_aux,
        )
        _accumulate_outcomes(
            command_accumulator,
            next_aux,
        )
        _accumulate_joint_success(
            agent,
            command_accumulator,
            next_aux,
        )
        hard = dones & ~timeouts
        extra = agent._p3_extra
        healthy = (
            (next_aux[:, p2_contract.GAIT_VALID_INDEX] > 0.5)
            & ~dones
            & (next_aux[:, p2_contract.BODY_COLLISION_FORCE_INDEX] < 30.0)
            & (next_aux[:, 21:23].abs().amax(dim=-1) < 0.35)
            & (torch.linalg.vector_norm(next_aux[:, 12:15] - exec_command, dim=-1) < 0.50)
        )
        if agent.algorithm.session_effective_seconds < 900.0:
            agent.algorithm.gait_baseline.observe(
                next_aux,
                extra,
                healthy,
                collect_continuous=(
                    storage.step % p3_contract.GAIT_BASELINE_CONTINUOUS_STRIDE == 0
                ),
            )
        contact_reward, cross_reward, starvation_reward = (
            agent.algorithm.gait_baseline.rewards(next_aux, extra, gait_fraction)
        )
        if storage.step % p2_contract.NAV_PERIOD_FRAMES != p2_contract.NAV_PERIOD_FRAMES - 1:
            starvation_reward.zero_()
        gait_rewards = contact_reward + cross_reward + starvation_reward
        rewards = rewards + gait_rewards
        gait_reward_sums += torch.stack(
            (contact_reward.mean(), cross_reward.mean(), starvation_reward.mean())
        )
        if train_low:
            transition.rewards = rewards.clone()
            transition.dones = dones
            transition.hard_terminations = hard
            # The wrapper exposes only the post-reset observation.
            # Conservatively terminate GAE at timeout instead of bootstrapping
            # from either the pre-step value or the next episode.
            storage.add_transitions(transition)
        _record_adapter_frame(
            agent,
            critic_wire,
            target_command,
            exec_command,
            command_epoch,
            dones,
        )
        algorithm.reset_recurrent_states(dones)
        obs, critic_wire = next_obs, next_wire
        reward_sum += rewards.mean()
        done_sum += dones.float().mean()
        _advance_platform_lifecycle(agent)

    if train_low:
        with torch.no_grad():
            next_critic, _ = split_p2_transport(critic_wire)
            last_values = algorithm.actor_critic.evaluate(next_critic).detach()
        storage.compute_returns(last_values, algorithm.gamma, algorithm.lam)
        metrics = algorithm.learn(agent.training_elapsed_h)
        low_updated = _finalize_low_level_update(agent, metrics)
        if low_updated:
            update_count = 2 if agent.algorithm.current_phase == "lowmedium" else 1
            metrics.update(_run_adapter_updates(agent, update_count))
        else:
            metrics.update(_run_adapter_updates(agent, 0))
    else:
        metrics = {"applied_updates": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    storage_bytes = _storage_bytes(storage)
    storage.clear()
    metrics.update(
        {
            "low_reward_mean": float(reward_sum / storage.num_transitions_per_env),
            "low_done_rate": float(done_sum / storage.num_transitions_per_env),
            "reward_p3_contact_quality": float(gait_reward_sums[0] / storage.num_transitions_per_env),
            "reward_p3_crossing": float(gait_reward_sums[1] / storage.num_transitions_per_env),
            "reward_p3_starvation": float(gait_reward_sums[2] / storage.num_transitions_per_env),
            "p3_gait_baseline_finalized": float(agent.algorithm.gait_baseline.finalized),
            "p3_gait_baseline_fallback_share": float(agent.algorithm.gait_baseline.fallback_share),
            "gait_baseline_samples": float(agent.algorithm.gait_baseline.total_samples),
            "p3_low_storage_bytes": float(storage_bytes),
            **_command_metrics(command_accumulator),
        }
    )
    return obs, critic_wire, metrics


def _collect_high_rollout(env, agent, obs, critic_wire):
    algorithm = agent.high_level_algorithm
    # The high-level controller shares the frozen low-level executor with the
    # low PPO path.  Reassert inference mode after every possible low update.
    agent.low_level_algorithm.actor_critic.eval()
    local_successes = torch.zeros((), device=agent.device)
    local_timeouts = torch.zeros((), device=agent.device)
    command_accumulator = _command_accumulator(agent.device)
    reward_component_sum = {}
    reward_component_ticks = 0
    frontier_clawback = torch.zeros((), device=agent.device)
    frontier_clawback_count = torch.zeros((), device=agent.device)
    radial_reward_sum = torch.zeros((), device=agent.device)
    local_positive_sum = torch.zeros((), device=agent.device)
    local_negative_sum = torch.zeros((), device=agent.device)
    for _ in range(p2_contract.NAV_ROLLOUT_TICKS):
        _, start_aux = split_p2_transport(critic_wire)
        agent.algorithm.sync_episode_origins(start_aux)
        tick_origin = agent.algorithm.episode_origin_xy.clone()
        tick_best_radius = agent.algorithm.best_radial_distance.clone()
        start_goal = start_aux[:, p2_contract.PRE_STEP_GOAL_DISTANCE_INDEX].clone()
        event_goal = torch.full_like(start_goal, float("nan"))
        local_success = torch.zeros(agent.num_envs, dtype=torch.bool, device=agent.device)
        local_timeout = torch.zeros_like(local_success)
        m1_success = torch.zeros_like(local_success)
        m2_success = torch.zeros_like(local_success)
        m3_success = torch.zeros_like(local_success)
        duration = torch.zeros(agent.num_envs, dtype=torch.long, device=agent.device)
        hard = torch.zeros(agent.num_envs, dtype=torch.bool, device=agent.device)
        timeout = torch.zeros_like(hard)
        reason = torch.zeros(agent.num_envs, dtype=torch.long, device=agent.device)
        active = torch.ones_like(hard)
        frame_reward = torch.zeros(agent.num_envs, device=agent.device)
        terminal_aux = torch.zeros(
            agent.num_envs, p2_contract.WORKER_AUX_DIM, device=agent.device
        )
        terminal_exec = torch.zeros(agent.num_envs, 3, device=agent.device)
        for _frame in range(p2_contract.NAV_PERIOD_FRAMES):
            _, current_aux = split_p2_transport(critic_wire)
            agent.algorithm.guard_high_commands(current_aux)
            if _frame > 0:
                # A vectorized environment may reset one member while the other
                # members are still completing this high-level transition. Feed
                # the reset episode a zero command instead of the old episode's
                # command. This is agent-side and never patches platform BaseEnv.
                _zero_inactive_high_commands(algorithm.command, active)
            result, _, aux = algorithm.frame_begin(obs, critic_wire)
            agent.algorithm.guard_high_commands(aux)
            if _frame > 0:
                # frame_begin can open a new 5 Hz tick and sample a target. The
                # inactive member still belongs to the previous transition.
                _zero_inactive_high_commands(algorithm.command, active)
            _accumulate_command(
                command_accumulator,
                algorithm.command.active_target,
                algorithm.command.exec_cmd,
                aux,
                agent._p3_extra,
            )
            step_data = env.step(torch.clamp(result["actions"], -6.0, 6.0))
            _, next_obs, _, terminated, truncated, infos, next_wire = _extract_step(
                step_data
            )
            next_obs = _as_device(next_obs, agent.device)
            next_wire = _consume_p3_wire(agent, _as_device(next_wire, agent.device))
            next_aux = next_wire[:, p2_contract.CRITIC_OBS_DIM :]
            frame_done, frame_timeout = _frame_done_masks(
                terminated,
                truncated,
                infos,
                agent.device,
                worker_aux=next_aux,
            )
            _accumulate_outcomes(command_accumulator, next_aux)
            p3_events = _accumulate_joint_success(
                agent, command_accumulator, next_aux
            )
            m1_success |= active & p3_events["m1_reached_mask"]
            m2_success |= active & p3_events["m2_reached_mask"]
            m3_success |= active & p3_events["m3_proxy_new_mask"]
            duration += active.long()
            new_done = active & frame_done
            resolved, new_hard, new_timeout = _resolve_terminal_outcome(
                new_done, frame_timeout, next_aux[:, 25].round().long()
            )
            reason[new_done] = resolved[new_done]
            hard |= new_hard
            timeout |= new_timeout
            event = next_aux[:, p2_contract.CURRENT_SEGMENT_INDEX].round().long()
            reached = active & (event == p3_contract.SUBGOAL_EVENT_REACHED)
            expired = active & (event == p3_contract.SUBGOAL_EVENT_TIMEOUT)
            first_event = (reached | expired) & ~torch.isfinite(event_goal)
            event_goal[first_event] = next_aux[
                first_event, p2_contract.PRE_STEP_GOAL_DISTANCE_INDEX
            ]
            local_success |= reached
            local_timeout |= expired
            if bool(new_done.any()):
                terminal_aux[new_done] = aux[new_done]
                terminal_exec[new_done] = algorithm.command.exec_cmd[new_done]
            active &= ~frame_done
            algorithm.frame_end(aux, frame_done)
            obs, critic_wire = next_obs, next_wire
            _advance_platform_lifecycle(agent)
        _, live_aux = split_p2_transport(critic_wire)
        live_goal = live_aux[:, p2_contract.PRE_STEP_GOAL_DISTANCE_INDEX]
        end_goal = torch.where(torch.isfinite(event_goal), event_goal, live_goal)
        transition_done = hard | timeout
        diagnostic_pose = torch.where(
            transition_done.unsqueeze(-1), terminal_aux[:, 15:17], live_aux[:, 15:17]
        )
        end_radius = p3_contract.radial_distance(diagnostic_pose, tick_origin)
        local_positive, local_negative = p3_contract.asymmetric_local_progress(
            start_goal, end_goal, duration
        )
        radial_reward, _ = p3_contract.radial_new_best(
            end_radius, tick_best_radius
        )
        radial_allowed = ~transition_done | (reason == 1)
        radial_reward = torch.where(radial_allowed, radial_reward, 0.0)
        frame_reward += local_positive + local_negative + radial_reward
        frame_reward += (m1_success | m2_success).float() * float(
            agent.algorithm.config.get("milestone_success_reward", 1.5)
        )
        frame_reward += m3_success.float() * float(
            agent.algorithm.config.get("m3_success_reward", 15.0)
        )
        frame_reward += local_timeout.float() * float(
            agent.algorithm.config.get("subgoal_timeout_penalty", -0.5)
        )
        diagnostic_aux = torch.where(
            transition_done.unsqueeze(-1), terminal_aux, live_aux
        )
        exec_cmd = torch.where(
            transition_done.unsqueeze(-1),
            terminal_exec,
            algorithm.command.exec_cmd,
        )
        full = algorithm.finish_tick(
            obs,
            critic_wire,
            frame_safety_reward=frame_reward,
            start_goal_distance=start_goal,
            end_goal_distance=end_goal,
            terminal_reason=reason,
            duration_frames=duration,
            hard_terminated=hard,
            timeout=timeout,
            terminal_safe_aux=diagnostic_aux,
            terminal_safe_exec_cmd=exec_cmd,
            frontier_settle_mask=local_timeout,
        )
        reward_component_ticks += 1
        for name, value in algorithm.last_tick_penalties.items():
            contribution = value.float().mean().detach()
            reward_component_sum[name] = (
                reward_component_sum.get(name, torch.zeros_like(contribution))
                + contribution
            )
        frontier_value = algorithm.last_tick_penalties.get("frontier_shaping")
        if torch.is_tensor(frontier_value):
            frontier_clawback += torch.clamp(
                frontier_value.reshape(-1)[local_timeout], max=0.0
            ).sum().detach()
            frontier_clawback_count += local_timeout.float().sum().detach()
        reset_reward_state = local_success | local_timeout
        if bool(reset_reward_state.any()):
            algorithm.best_goal_distance[reset_reward_state] = float("inf")
            algorithm.episode_start_goal_distance[reset_reward_state] = float("inf")
            algorithm._reset_navigation_reward_state(reset_reward_state)
        local_successes += local_success.float().sum()
        local_timeouts += local_timeout.float().sum()
        radial_reward_sum += radial_reward.mean().detach()
        local_positive_sum += local_positive.mean().detach()
        local_negative_sum += local_negative.mean().detach()
        if full and algorithm.rollout.step != p2_contract.NAV_ROLLOUT_TICKS:
            raise RuntimeError("P3 high rollout filled at an invalid boundary")
    if not algorithm.rollout.full:
        raise RuntimeError("P3 high rollout did not fill at 32 ticks")
    metrics = algorithm.update()
    metrics["high_reward_mean"] = float(metrics.get("rollout_reward_mean", 0.0))
    agent.algorithm.high_updates += 1
    response_config = agent.algorithm.config.get("response_adapter", {})
    if agent.algorithm.current_phase == "adaptercalib":
        metrics.update(
            _run_adapter_updates(
                agent,
                int(response_config.get("calibration_updates_per_iteration", 4)),
            )
        )
    elif _high_adapter_update_due(agent.algorithm.high_updates, response_config):
        metrics.update(_run_adapter_updates(agent, 1))
    else:
        metrics.update(_run_adapter_updates(agent, 0))
    reward_metric_keys = tuple(reward_component_sum)
    reward_metric_values = (
        torch.stack(
            [
                reward_component_sum[name] / max(1, reward_component_ticks)
                for name in reward_metric_keys
            ]
        ).detach().cpu().tolist()
        if reward_metric_keys
        else []
    )
    frontier_settlement_mean = frontier_clawback / frontier_clawback_count.clamp_min(1.0)
    terminal_values = torch.stack(
        (local_successes, local_timeouts, frontier_settlement_mean)
    ).detach().cpu().tolist()
    metrics.update(
        {
            "subgoal_success_count": terminal_values[0],
            "subgoal_timeout_count": terminal_values[1],
            "p3_low_storage_bytes": float(
                _storage_bytes(agent.low_level_algorithm.storage)
            ),
            **_command_metrics(command_accumulator),
            **{
                f"reward_{name}": value
                for name, value in zip(reward_metric_keys, reward_metric_values)
            },
            "reward_frontier_settlement_mean": terminal_values[2],
            "reward_radial_new_best": float(
                radial_reward_sum / p2_contract.NAV_ROLLOUT_TICKS
            ),
            "reward_local_positive_progress": float(
                local_positive_sum / p2_contract.NAV_ROLLOUT_TICKS
            ),
            "reward_local_negative_progress": float(
                local_negative_sum / p2_contract.NAV_ROLLOUT_TICKS
            ),
        }
    )
    return obs, critic_wire, metrics


def _monitor_put(monitor, metrics):
    if monitor is not None:
        monitor.put_data({os.getpid(): metrics})


def _install_sigterm_handler():
    if (
        not hasattr(signal, "SIGTERM")
        or threading.current_thread() is not threading.main_thread()
    ):
        return None
    previous = signal.getsignal(signal.SIGTERM)

    def _handle(signum, frame):
        if previous is not signal.SIG_DFL and callable(previous):
            previous(signum, frame)
        raise SystemExit(f"SIGTERM({signum})")

    signal.signal(signal.SIGTERM, _handle)
    return previous


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    del args, kwargs
    env, agent = envs[0], agents[0]
    if not getattr(agent, "is_p3_joint", False):
        raise RuntimeError("P3 workflow requires Agent.is_p3_joint")
    usr_conf, _, _, _ = Config.load_conf(logger)
    conf = usr_conf.get("p3_standard_joint", {})
    obs, critic_wire = _reset_env(env, agent, usr_conf)
    resumed = float(agent.algorithm.session_effective_seconds)
    target = float(conf.get("target_effective_seconds", p3_contract.SESSION_TARGET_SECONDS))
    save_interval = 60.0 * float(conf.get("save_interval_minutes", 10.0))
    first_save = 60.0 * float(conf.get("first_save_minutes", 5.0))
    next_save = first_save if resumed <= 0.0 else (
        math.floor(resumed / save_interval) + 1
    ) * save_interval
    last_log = 0.0
    agent._p3_lifecycle_attempt_callbacks = 0
    agent._p3_lifecycle_success_callbacks = 0
    agent._p3_lifecycle_failure_callbacks = 0
    previous_sigterm = _install_sigterm_handler()
    try:
        while agent.algorithm.session_effective_seconds < target:
            loop_started = time.monotonic()
            phase_before = agent.algorithm.current_phase
            elapsed_before = float(agent.algorithm.session_effective_seconds)
            collect_high = agent.algorithm.should_collect_high_rollout()
            if collect_high:
                obs, critic_wire, metrics = _collect_high_rollout(
                    env, agent, obs, critic_wire
                )
            else:
                train_low = p3_contract.phase_for_elapsed(
                    agent.algorithm.session_effective_seconds
                ).low_level_trainable
                obs, critic_wire, metrics = _collect_low_rollout(
                    env, agent, obs, critic_wire, train_low=train_low
                )
            now = time.monotonic()
            elapsed = elapsed_before + (now - loop_started)
            changed = agent.algorithm.update_clock(elapsed)
            if elapsed >= 900.0 and not agent.algorithm.gait_baseline.finalized:
                agent.algorithm.gait_baseline.finalize()
            agent.training_elapsed_h = elapsed / 3600.0
            agent.algorithm.current_iteration += 1
            metrics.update(
                {
                    "p3_session_effective_seconds": elapsed,
                    "p3_phase": float(
                        tuple(item.name for item in p3_contract.PHASES).index(
                            agent.algorithm.current_phase
                        )
                    ),
                    "p3_low_updates": float(agent.algorithm.low_updates),
                    "p3_high_updates": float(agent.algorithm.high_updates),
                    "platform_lifecycle_callbacks": float(
                        agent._p3_lifecycle_success_callbacks
                    ),
                    "platform_lifecycle_failures": float(
                        agent._p3_lifecycle_failure_callbacks
                    ),
                    "p3_rollout_time_s": now - loop_started,
                    **agent.algorithm.memory_metrics(),
                }
            )
            rollout_frames = (
                p2_contract.NAV_ROLLOUT_TICKS * p2_contract.NAV_PERIOD_FRAMES
                if collect_high
                else int(conf.get("num_steps_per_env", 80))
            )
            metrics["samples_per_s"] = (
                float(agent.num_envs * rollout_frames)
                / max(now - loop_started, 1.0e-6)
            )
            realized = p3_contract.materialize_environment_config(usr_conf, elapsed)
            domain_rand = realized.get("domain_rand", {})
            runtime = realized.get("p3_runtime", {})
            metrics.update(
                {
                    "p3_dr_phase": float(runtime.get("phase_index", 0)),
                    "p3_friction_min": float(domain_rand.get("friction_range", [1, 1])[0]),
                    "p3_friction_max": float(domain_rand.get("friction_range", [1, 1])[1]),
                    "p3_base_added_mass_kg": float(runtime.get("base_added_mass_kg", 0.0)),
                    "p3_noise_level": float((realized.get("noise", {}) or {}).get("noise_level", 0.0)),
                }
            )
            if changed:
                agent.save_model()
                domain_randomization_changed = _requires_environment_rebuild(
                    elapsed_before, elapsed
                )
                logger.info(
                    f"[P3] rollout-boundary phase change {phase_before} -> "
                    f"{agent.algorithm.current_phase}; "
                    "environment_reset=True; "
                    f"domain_randomization_changed={domain_randomization_changed}"
                )
                # Every responsibility boundary starts from a fresh episode.
                # This uses the public platform reset API and does not depend on
                # patching the platform-owned BaseEnv implementation.
                obs, critic_wire = _reset_env(env, agent, usr_conf)
            if elapsed >= next_save:
                agent.save_model()
                while next_save <= elapsed:
                    next_save += save_interval
            if now - last_log >= 60.0 or agent.algorithm.current_iteration == 1:
                logger.info(
                    f"[P3] iter={agent.algorithm.current_iteration} "
                    f"session_h={elapsed / 3600.0:.3f} "
                    f"phase={agent.algorithm.current_phase} "
                    f"low_updates={agent.algorithm.low_updates} "
                    f"high_updates={agent.algorithm.high_updates}"
                )
                _monitor_put(monitor, metrics)
                last_log = now
        agent.save_model()
    except (KeyboardInterrupt, SystemExit):
        agent.save_model()
        raise
    finally:
        if hasattr(signal, "SIGTERM") and previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
