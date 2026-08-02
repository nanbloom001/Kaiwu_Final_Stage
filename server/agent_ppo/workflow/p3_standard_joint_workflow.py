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
from agent_ppo.feature.p3_gait import P3GaitBaseline, gait_bucket_indices, mirror_proprio
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


P35_REQUIRED_MONITOR_METRICS = (
    "p35_push_telemetry_valid_share",
    "p35_push_runtime_active_share",
    "p35_push_event_count",
    "p35_push_delta_vx_p95",
    "p35_push_delta_vy_p95",
    "p35_push_post_roll_peak",
    "p35_push_post_speed_mae_peak",
    "camera_hold_ratio",
    "camera_feature_age_p95_ms",
    "camera_active_delay_p95_ms",
    "p35_reward_progress",
    "p35_reward_joint_acc",
    "p35_reward_contact",
    "p35_reward_gait",
    "p35_reward_posture",
    "p35_reward_baseline_valid",
    "p35_baseline_samples_joint_pos",
    "p35_baseline_samples_joint_acc_noncontact",
    "p35_baseline_samples_posture",
    "p35_baseline_samples_frequency",
    "p35_baseline_eligible_contact_share",
    "p3_low_updates",
    "p3_high_updates",
)


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
    agent.algorithm.depth_fault.ensure_environment_initialized()
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


def _high_adapter_update_due(high_updates: int, response_config: dict) -> bool:
    interval = max(1, int(response_config.get("high_update_interval", 2)))
    return int(high_updates) % interval == 0


def _run_adapter_updates(agent, count: int) -> dict[str, float]:
    agent.response_aux_buffer.set_p3_replay_ratios(
        *p3_contract.adapter_replay_ratios(
            agent.algorithm.session_effective_seconds
        )
    )
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


def _low_policy_step(
    agent, proprio, depth, critic_obs, *, clean_depth=None, cnn_features=None
):
    algorithm = agent.low_level_algorithm
    hidden = algorithm.rollout_hidden_state()
    actions = (
        algorithm.actor_critic.act_from_proprio_depth(proprio, depth)
        if cnn_features is None
        else algorithm.actor_critic.act_from_proprio_cnn_features(
            proprio, cnn_features
        )
    )
    clean_features = algorithm.actor_critic.last_cnn_features
    if clean_depth is not None:
        clean_features = algorithm.anchor_encoder.cnn(clean_depth)
    anchor_action, anchor_latent = algorithm.anchor_inference_from_cnn_features(
        proprio, clean_features
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
        "clean_obs": torch.cat((proprio, clean_features), dim=-1),
        "actions": actions,
        "values": values,
        "log_prob": log_prob,
        "mean": algorithm.actor_critic.action_mean.detach(),
        "std": algorithm.actor_critic.action_std.detach(),
        "hidden": hidden,
        "anchor_action": anchor_action,
        "anchor_latent": anchor_latent,
    }


def _low_policy_action(agent, proprio, depth, *, cnn_features=None):
    """Run the frozen low-level executor without constructing PPO metadata."""
    if cnn_features is None:
        return agent.low_level_algorithm.actor_critic.act_from_proprio_depth(
            proprio, depth
        )
    return agent.low_level_algorithm.actor_critic.act_from_proprio_cnn_features(
        proprio, cnn_features
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


def _command_accumulator(device, *, push_limit_m_s: float = 0.0):
    gait_bucket_count = len(P3GaitBaseline.TERRAIN_BUCKETS) * len(
        P3GaitBaseline.MOTION_BUCKETS
    )
    return {
        "count": torch.zeros((), device=device),
        "target": torch.zeros(3, device=device),
        "exec": torch.zeros(3, device=device),
        "measured": torch.zeros(3, device=device),
        "true": torch.zeros(3, device=device),
        "tracking": torch.zeros(3, device=device),
        "positive": torch.zeros(3, device=device),
        "negative": torch.zeros(3, device=device),
        "signed_count": torch.zeros(3, 2, device=device),
        "signed_target": torch.zeros(3, 2, device=device),
        "signed_exec": torch.zeros(3, 2, device=device),
        "signed_true": torch.zeros(3, 2, device=device),
        "signed_tracking": torch.zeros(3, 2, device=device),
        "gait_duty": torch.zeros(4, device=device),
        "gait_air": torch.zeros(4, device=device),
        "gait_frequency": torch.zeros(4, device=device),
        "gait_slip_speed": torch.zeros(4, device=device),
        "gait_completed_slip": torch.zeros(4, device=device),
        "gait_completed_slip_count": torch.zeros(4, device=device),
        "gait_valid": torch.zeros((), device=device),
        "body_collision_mapping_valid": torch.zeros((), device=device),
        "mirror_mapping_valid": torch.zeros((), device=device),
        "feedback_valid": torch.zeros((), device=device),
        "feedback_age_s": torch.zeros((), device=device),
        "feedback_true_error": torch.zeros((), device=device),
        "terrain_columns": torch.zeros(20, device=device),
        "terrain_levels": torch.zeros(10, device=device),
        # Platform-aligned outcomes count a latched/reached 3.9 m proxy as
        # completion even if the worker later emits a time-limit or fall code.
        # Preserve the raw worker reason codes as a separate diagnostic.
        "outcomes": torch.zeros(3, device=device),
        "raw_outcomes": torch.zeros(3, device=device),
        "timeout_after_completion": torch.zeros((), device=device),
        "hard_after_completion": torch.zeros((), device=device),
        "near_clip_stair_outcomes": torch.zeros(10, 3, device=device),
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
        "torque_near_hard_frames": None,
        "torque_near_hard_max_frames": None,
        "torque_near_hard_count": torch.zeros((), device=device),
        "mechanical_power": torch.zeros((), device=device),
        "mechanical_power_samples": [],
        "gait_contact_onset": torch.zeros(4, device=device),
        "gait_impact": torch.zeros(4, device=device),
        "gait_touchdown_y": torch.zeros(4, device=device),
        "gait_stance": torch.zeros(4, device=device),
        "gait_bucket_count": torch.zeros(gait_bucket_count, device=device),
        "gait_bucket_impact_count": torch.zeros(gait_bucket_count, 4, device=device),
        "gait_bucket_slip_count": torch.zeros(gait_bucket_count, 4, device=device),
        "gait_bucket_slip": torch.zeros(gait_bucket_count, 4, device=device),
        "gait_bucket_impact": torch.zeros(gait_bucket_count, 4, device=device),
        "gait_bucket_stance": torch.zeros(gait_bucket_count, 4, device=device),
        "gait_bucket_duty": torch.zeros(gait_bucket_count, 4, device=device),
        "gait_bucket_frequency": torch.zeros(gait_bucket_count, 4, device=device),
        "command_bucket_count": torch.zeros(7, device=device),
        "command_bucket_clipped": torch.zeros(7, device=device),
        "command_anchor_sum": torch.zeros(7, device=device),
        "sim2real_component_sum": torch.zeros(4, device=device),
        "sim2real_component_valid": torch.zeros((), device=device),
        "push_event_count": torch.zeros((), device=device),
        "push_delta_samples": [],
        "push_env_seen": None,
        "push_runtime_active": torch.zeros((), device=device),
        "push_telemetry_valid": torch.zeros((), device=device),
        "push_seconds_since_min": torch.full((), 1.0e6, device=device),
        "push_post_count": torch.zeros((), device=device),
        "push_post_roll_peak": torch.zeros((), device=device),
        "push_post_pitch_peak": torch.zeros((), device=device),
        "push_post_speed_mae_peak": torch.zeros((), device=device),
        "push_post_fall_count": torch.zeros((), device=device),
        "push_recovered": None,
        "push_recovery_time_samples": [],
        "push_terrain_count": torch.zeros(4, device=device),
        "push_level_count": torch.zeros(10, device=device),
        "push_command_count": torch.zeros(7, device=device),
        "push_speed_count": torch.zeros(3, device=device),
        "push_direction_count": torch.zeros(4, device=device),
        "push_config_limit_m_s": float(push_limit_m_s),
        "push_config_violation_count": torch.zeros((), device=device),
        "push_post_timeout_count": torch.zeros((), device=device),
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
    for axis in range(3):
        for sign_index, selected in enumerate((target[:, axis] > 0.05, target[:, axis] < -0.05)):
            selected_f = selected.float()
            accumulator["signed_count"][axis, sign_index] += selected_f.sum()
            accumulator["signed_target"][axis, sign_index] += (target[:, axis] * selected_f).sum()
            accumulator["signed_exec"][axis, sign_index] += (exec_command[:, axis] * selected_f).sum()
            accumulator["signed_true"][axis, sign_index] += (true[:, axis] * selected_f).sum()
            accumulator["signed_tracking"][axis, sign_index] += (
                torch.abs(exec_command[:, axis] - true[:, axis]) * selected_f
            ).sum()
    gait_valid = aux[:, p2_contract.GAIT_VALID_INDEX] > 0.5
    gait_valid_f = gait_valid.to(aux).unsqueeze(-1)
    accumulator["gait_duty"] += (
        aux[:, p2_contract.GAIT_DUTY_SLICE] * gait_valid_f
    ).sum(dim=0)
    accumulator["gait_air"] += (
        aux[:, p2_contract.GAIT_MAX_AIR_SLICE] * gait_valid_f
    ).sum(dim=0)
    accumulator["gait_frequency"] += (
        aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE] * gait_valid_f
    ).sum(dim=0)
    accumulator["gait_slip_speed"] += (
        aux[:, p2_contract.GAIT_SLIP_SPEED_SLICE] * gait_valid_f
    ).sum(dim=0)
    accumulator["gait_valid"] += gait_valid_f.sum()
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
        absolute_torque = extra[:, p3_contract.JOINT_TORQUE_SLICE].abs()
        hard = absolute_torque.new_tensor([22.0] * 8 + [43.0] * 4)
        near_hard = (absolute_torque >= 0.95 * hard).any(dim=-1)
        if accumulator["torque_near_hard_frames"] is None:
            accumulator["torque_near_hard_frames"] = torch.zeros_like(
                near_hard, dtype=torch.float32
            )
            accumulator["torque_near_hard_max_frames"] = torch.zeros_like(
                near_hard, dtype=torch.float32
            )
        accumulator["torque_near_hard_frames"] = (
            accumulator["torque_near_hard_frames"] + 1.0
        ) * near_hard.float()
        accumulator["torque_near_hard_max_frames"] = torch.maximum(
            accumulator["torque_near_hard_max_frames"],
            accumulator["torque_near_hard_frames"],
        )
        accumulator["torque_near_hard_count"] += near_hard.float().sum()
        mechanical_power = extra[:, p3_contract.MECHANICAL_POWER_INDEX]
        accumulator["mechanical_power"] += mechanical_power.sum()
        accumulator["mechanical_power_samples"].append(mechanical_power.detach())
        onset = (extra[:, p3_contract.GAIT_CONTACT_ONSET_SLICE] > 0.5) & gait_valid.unsqueeze(-1)
        onset_f = onset.to(extra)
        slip_event = (
            extra[:, p3_contract.GAIT_COMPLETED_SLIP_EVENT_SLICE] > 0.5
        ) & gait_valid.unsqueeze(-1)
        slip_event_f = slip_event.to(extra)
        accumulator["gait_contact_onset"] += onset_f.sum(0)
        accumulator["gait_impact"] += (
            extra[:, p3_contract.GAIT_IMPACT_SPEED_SLICE] * onset_f
        ).sum(0)
        accumulator["gait_touchdown_y"] += (
            extra[:, p3_contract.GAIT_TOUCHDOWN_Y_SLICE] * onset_f
        ).sum(0)
        accumulator["gait_stance"] += (
            extra[:, p3_contract.GAIT_CONTINUOUS_STANCE_SLICE] * gait_valid_f
        ).sum(0)
        accumulator["gait_completed_slip"] += (
            extra[:, p3_contract.GAIT_COMPLETED_SLIP_SLICE] * slip_event_f
        ).sum(0)
        accumulator["gait_completed_slip_count"] += slip_event_f.sum(0)
        accumulator["mirror_mapping_valid"] += (
            extra[:, p3_contract.JOINT_MAPPING_VALID_INDEX] > 0.5
        ).float().sum()
        sim2real_valid = (
            extra[:, p3_contract.SIM2REAL_COMPONENT_VALID_INDEX] > 0.5
        )
        accumulator["sim2real_component_valid"] += sim2real_valid.float().sum()
        accumulator["sim2real_component_sum"] += (
            extra[:, p3_contract.SIM2REAL_COMPONENT_SLICE]
            * sim2real_valid.to(extra).unsqueeze(-1)
        ).sum(dim=0)
        push_valid = extra[:, p3_contract.PUSH_TELEMETRY_VALID_INDEX] > 0.5
        push_active = extra[:, p3_contract.PUSH_RUNTIME_ACTIVE_INDEX] > 0.5
        push_delta = extra[:, p3_contract.PUSH_DELTA_VELOCITY_SLICE]
        push_event = (
            (extra[:, p3_contract.PUSH_EVENT_FLAG_INDEX] > 0.5)
            & push_valid
            & push_active
            & torch.isfinite(push_delta).all(dim=-1)
            & (push_delta.abs().amax(dim=-1) > 1.0e-6)
        )
        accumulator["push_runtime_active"] += push_active.float().sum()
        accumulator["push_telemetry_valid"] += push_valid.float().sum()
        if bool(push_event.any()):
            accumulator["push_event_count"] += push_event.float().sum()
            accumulator["push_delta_samples"].append(
                extra[push_event, p3_contract.PUSH_DELTA_VELOCITY_SLICE].detach()
            )
            limit = float(accumulator["push_config_limit_m_s"])
            if limit > 0.0:
                accumulator["push_config_violation_count"] += (
                    extra[push_event, p3_contract.PUSH_DELTA_VELOCITY_SLICE].abs()
                    > limit + 1.0e-5
                ).any(dim=-1).float().sum()
            ids = push_event.nonzero(as_tuple=False).flatten()
            if accumulator["push_env_seen"] is None:
                accumulator["push_env_seen"] = torch.zeros(
                    extra.shape[0], dtype=torch.bool, device=extra.device
                )
            accumulator["push_env_seen"][ids] = True
            if accumulator["push_recovered"] is None:
                accumulator["push_recovered"] = torch.zeros(
                    extra.shape[0], dtype=torch.bool, device=extra.device
                )
            accumulator["push_recovered"][ids] = False
            terrain_columns = aux[push_event, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX].round().long()
            terrain_groups = torch.full_like(terrain_columns, -1)
            first, second, third, fourth = p3_contract.TERRAIN_COLUMN_BUCKET_BOUNDARIES
            terrain_groups[(terrain_columns >= 0) & (terrain_columns < first)] = 0
            terrain_groups[(terrain_columns >= first) & (terrain_columns < second)] = 1
            terrain_groups[(terrain_columns >= second) & (terrain_columns < third)] = 2
            terrain_groups[(terrain_columns >= third) & (terrain_columns < fourth)] = 3
            valid_terrain = terrain_groups >= 0
            accumulator["push_terrain_count"] += torch.bincount(
                terrain_groups[valid_terrain], minlength=4
            ).to(accumulator["push_terrain_count"])
            levels = aux[push_event, p2_contract.PRE_STEP_TERRAIN_LEVEL_INDEX].round().long()
            valid_level = (levels >= 0) & (levels < 10)
            accumulator["push_level_count"] += torch.bincount(
                levels[valid_level], minlength=10
            ).to(accumulator["push_level_count"])
            buckets = extra[push_event, p3_contract.COMMAND_BUCKET_INDEX].round().long()
            valid_bucket = (buckets >= 0) & (buckets < 7)
            accumulator["push_command_count"] += torch.bincount(
                buckets[valid_bucket], minlength=7
            ).to(accumulator["push_command_count"])
            vx = exec_command[push_event, 0]
            speed_bucket = torch.where(vx < 0.35, 0, torch.where(vx < 0.70, 1, 2)).long()
            accumulator["push_speed_count"] += torch.bincount(
                speed_bucket, minlength=3
            ).to(accumulator["push_speed_count"])
            delta = extra[push_event, p3_contract.PUSH_DELTA_VELOCITY_SLICE]
            accumulator["push_direction_count"] += torch.stack(
                ((delta[:, 0] > 0).sum(), (delta[:, 0] < 0).sum(),
                 (delta[:, 1] > 0).sum(), (delta[:, 1] < 0).sum())
            ).to(accumulator["push_direction_count"])
        seconds_since = extra[:, p3_contract.SECONDS_SINCE_PUSH_INDEX]
        finite_age = seconds_since[
            push_valid & push_active & torch.isfinite(seconds_since)
        ]
        if finite_age.numel():
            accumulator["push_seconds_since_min"] = torch.minimum(
                accumulator["push_seconds_since_min"], finite_age.min()
            )
        post = (
            push_valid
            & push_active
            & (seconds_since >= 0.0)
            & (seconds_since <= 2.0)
        )
        if bool(post.any()):
            accumulator["push_post_count"] += post.float().sum()
            accumulator["push_post_roll_peak"] = torch.maximum(
                accumulator["push_post_roll_peak"], aux[post, 21].abs().max()
            )
            accumulator["push_post_pitch_peak"] = torch.maximum(
                accumulator["push_post_pitch_peak"], aux[post, 22].abs().max()
            )
            accumulator["push_post_speed_mae_peak"] = torch.maximum(
                accumulator["push_post_speed_mae_peak"],
                (true[post] - exec_command[post]).abs().amax(dim=-1).max(),
            )
            accumulator["push_post_fall_count"] += (
                (aux[post, 24] > 0.5) & (aux[post, 25].round() == 2)
            ).float().sum()
            accumulator["push_post_timeout_count"] += (
                (aux[post, 24] > 0.5) & (aux[post, 25].round() == 3)
            ).float().sum()
            if accumulator["push_recovered"] is None:
                accumulator["push_recovered"] = torch.zeros(
                    extra.shape[0], dtype=torch.bool, device=extra.device
                )
            speed_error = (true - exec_command).abs().amax(dim=-1)
            posture_error = aux[:, 21:23].abs().amax(dim=-1)
            recovered = post & ~accumulator["push_recovered"] & (speed_error < 0.10) & (posture_error < 0.12)
            if bool(recovered.any()):
                accumulator["push_recovery_time_samples"].append(
                    seconds_since[recovered].detach()
                )
                accumulator["push_recovered"][recovered] = True
        terrain_bucket, motion_bucket = gait_bucket_indices(aux, extra)
        valid_bucket = (terrain_bucket >= 0) & (motion_bucket >= 0) & gait_valid
        flat_bucket = terrain_bucket * len(P3GaitBaseline.MOTION_BUCKETS) + motion_bucket
        ids = flat_bucket[valid_bucket]
        accumulator["gait_bucket_count"].scatter_add_(
            0, ids, torch.ones_like(ids, dtype=accumulator["gait_bucket_count"].dtype)
        )
        accumulator["gait_bucket_impact_count"].scatter_add_(
            0,
            ids.unsqueeze(-1).expand(-1, 4),
            onset[valid_bucket].to(accumulator["gait_bucket_impact_count"]),
        )
        accumulator["gait_bucket_slip_count"].scatter_add_(
            0,
            ids.unsqueeze(-1).expand(-1, 4),
            slip_event[valid_bucket].to(accumulator["gait_bucket_slip_count"]),
        )
        for name, values in (
            (
                "gait_bucket_slip",
                extra[:, p3_contract.GAIT_COMPLETED_SLIP_SLICE] * slip_event_f,
            ),
            (
                "gait_bucket_impact",
                extra[:, p3_contract.GAIT_IMPACT_SPEED_SLICE] * onset_f,
            ),
            ("gait_bucket_stance", extra[:, p3_contract.GAIT_CONTINUOUS_STANCE_SLICE]),
            ("gait_bucket_duty", aux[:, p2_contract.GAIT_DUTY_SLICE]),
            ("gait_bucket_frequency", aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE]),
        ):
            accumulator[name].scatter_add_(
                0,
                ids.unsqueeze(-1).expand(-1, 4),
                values[valid_bucket],
            )
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


def _accumulate_outcomes(agent, accumulator, aux, near_clip_m=None):
    reset = aux[:, 24] > 0.5
    reason = aux[:, 25].round().long()
    for index, code in enumerate((1, 2, 3)):
        accumulator["raw_outcomes"][index] += (
            reset & (reason == code)
        ).float().sum()

    algorithm = agent.algorithm
    proxy_complete = algorithm.m3_proxy_latched.clone()
    origin_valid = algorithm.episode_origin_valid
    if bool(origin_valid.any()):
        pose_xy = aux[:, 15:17].to(algorithm.episode_origin_xy)
        radius = p3_contract.radial_distance(pose_xy, algorithm.episode_origin_xy)
        proxy_complete |= origin_valid & (
            radius >= float(algorithm.platform_complete_radius_m)
        )
    completed = reset & ((reason == 1) | proxy_complete)
    failure = reset & (reason == 2) & ~completed
    timeout = reset & (reason == 3) & ~completed
    accumulator["outcomes"][0] += completed.float().sum()
    accumulator["outcomes"][1] += failure.float().sum()
    accumulator["outcomes"][2] += timeout.float().sum()
    accumulator["hard_after_completion"] += (
        reset & (reason == 2) & completed
    ).float().sum()
    accumulator["timeout_after_completion"] += (
        reset & (reason == 3) & completed
    ).float().sum()
    if torch.is_tensor(near_clip_m) and near_clip_m.shape == reset.shape:
        columns = aux[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX].round().long()
        stairs = (columns >= 8) & (columns < 20)
        bins = torch.floor((near_clip_m.to(aux) - 0.10) / 0.015).long().clamp(0, 9)
        for index, selected in enumerate((completed, failure, timeout)):
            selected = selected & stairs
            accumulator["near_clip_stair_outcomes"][:, index].scatter_add_(
                0,
                bins[selected],
                torch.ones_like(
                    bins[selected],
                    dtype=accumulator["near_clip_stair_outcomes"].dtype,
                ),
            )


def _accumulate_action_bucket(accumulator, extra, actions):
    if not torch.is_tensor(extra) or extra.shape[1] != p3_contract.P3_WORKER_EXTRA_DIM:
        return
    bucket = extra[:, p3_contract.COMMAND_BUCKET_INDEX].round().long()
    valid = (bucket >= 0) & (bucket < accumulator["command_bucket_count"].numel())
    if not bool(valid.any()):
        return
    ids = bucket[valid]
    ones = torch.ones_like(ids, dtype=accumulator["command_bucket_count"].dtype)
    accumulator["command_bucket_count"].scatter_add_(0, ids, ones)
    clipped = (actions.detach().abs() > 6.0).any(dim=-1).float()[valid]
    accumulator["command_bucket_clipped"].scatter_add_(0, ids, clipped)
    anchor = extra[:, p3_contract.COMMAND_ANCHOR_WEIGHT_INDEX][valid].float()
    anchor = torch.nan_to_num(anchor, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    accumulator["command_anchor_sum"].scatter_add_(0, ids, anchor)


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
        for sign_index, sign in enumerate(("positive", "negative")):
            signed_count = accumulator["signed_count"][index, sign_index].clamp_min(1.0)
            result[f"target_{axis}_{sign}_mean"] = accumulator["signed_target"][index, sign_index] / signed_count
            result[f"exec_{axis}_{sign}_mean"] = accumulator["signed_exec"][index, sign_index] / signed_count
            result[f"true_{axis}_{sign}_mean"] = accumulator["signed_true"][index, sign_index] / signed_count
            result[f"{axis}_{sign}_tracking_abs_error"] = accumulator["signed_tracking"][index, sign_index] / signed_count
    gait_count = accumulator["gait_valid"].clamp_min(1.0)
    for index, leg in enumerate(("fl", "fr", "rl", "rr")):
        result[f"{leg}_duty_factor"] = accumulator["gait_duty"][index] / gait_count
        result[f"{leg}_max_air_time"] = accumulator["gait_air"][index] / gait_count
        result[f"{leg}_step_frequency"] = accumulator["gait_frequency"][index] / gait_count
        result[f"{leg}_slip_speed"] = (
            accumulator["gait_slip_speed"][index] / gait_count
        )
        result[f"{leg}_slip_distance"] = (
            accumulator["gait_completed_slip"][index]
            / accumulator["gait_completed_slip_count"][index].clamp_min(1.0)
        )
    result["gait_window_valid"] = accumulator["gait_valid"] / count
    result["gait_sensor_mapping_valid"] = result["gait_window_valid"]
    result["body_collision_mapping_valid"] = (
        accumulator["body_collision_mapping_valid"] / count
    )
    result["mirror_mapping_valid"] = accumulator["mirror_mapping_valid"] / count
    sim2real_count = accumulator["sim2real_component_valid"].clamp_min(1.0)
    result["p3_sim2real_component_valid_share"] = (
        accumulator["sim2real_component_valid"] / count
    )
    for index, name in enumerate(
        ("sustained_torque", "torque_peak", "action_rate", "action_jerk")
    ):
        result[f"p3_sim2real_{name}_raw"] = (
            accumulator["sim2real_component_sum"][index] / sim2real_count
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
    outcome_total = accumulator["outcomes"].sum()
    outcome_denominator = outcome_total.clamp_min(1.0)
    result["p3_window_episode_count"] = outcome_total
    result["p3_window_completion_rate"] = (
        accumulator["outcomes"][0] / outcome_denominator
    )
    result["p3_window_abnormal_rate"] = (
        accumulator["outcomes"][1] / outcome_denominator
    )
    result["p3_window_timeout_rate"] = (
        accumulator["outcomes"][2] / outcome_denominator
    )
    for index, label in enumerate(("completed", "failure", "timeout")):
        result[f"p3_window_raw_{label}_count"] = accumulator[
            "raw_outcomes"
        ][index]
    raw_total = accumulator["raw_outcomes"].sum()
    raw_denominator = raw_total.clamp_min(1.0)
    result["p3_window_raw_episode_count"] = raw_total
    result["p3_window_raw_completion_rate"] = (
        accumulator["raw_outcomes"][0] / raw_denominator
    )
    result["p3_window_raw_abnormal_rate"] = (
        accumulator["raw_outcomes"][1] / raw_denominator
    )
    result["p3_window_raw_timeout_rate"] = (
        accumulator["raw_outcomes"][2] / raw_denominator
    )
    result["p3_window_true_timeout_count"] = accumulator["outcomes"][2]
    result["p3_window_timeout_after_completion_count"] = accumulator[
        "timeout_after_completion"
    ]
    result["p3_window_hard_after_completion_count"] = accumulator[
        "hard_after_completion"
    ]
    for index in range(10):
        attempts = accumulator["near_clip_stair_outcomes"][index].sum()
        result[f"near_clip_bin_{index}_stair_attempt_count"] = attempts
        result[f"near_clip_bin_{index}_stair_completion_rate"] = (
            accumulator["near_clip_stair_outcomes"][index, 0]
            / attempts.clamp_min(1.0)
        )
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
    if accumulator["mechanical_power_samples"]:
        power = torch.cat(accumulator["mechanical_power_samples"]).float()
        result["mechanical_power_p50"] = torch.quantile(power, 0.50)
        result["mechanical_power_p95"] = torch.quantile(power, 0.95)
        result["mechanical_power_max"] = power.max()
    bucket_total = accumulator["command_bucket_count"].sum().clamp_min(1.0)
    for index, name in enumerate(
        (
            "straight", "reserved_reverse", "vx_vy", "vx_wz", "pure_yaw",
            "brake_restart", "zero",
        )
    ):
        bucket_count = accumulator["command_bucket_count"][index].clamp_min(1.0)
        result[f"command_bucket_{name}_share"] = (
            accumulator["command_bucket_count"][index] / bucket_total
        )
        result[f"command_bucket_{name}_clip_rate"] = (
            accumulator["command_bucket_clipped"][index] / bucket_count
        )
        result[f"command_bucket_{name}_anchor_mean"] = (
            accumulator["command_anchor_sum"][index] / bucket_count
        )
    if accumulator["torque_samples"]:
        torque = torch.cat(accumulator["torque_samples"], dim=0)
        for label, ids in (
            ("hip", (0, 1, 2, 3)),
            ("thigh", (4, 5, 6, 7)),
            ("calf", (8, 9, 10, 11)),
        ):
            values = torque[:, ids].reshape(-1)
            result[f"{label}_torque_p50"] = torch.quantile(values, 0.50)
            result[f"{label}_torque_p95"] = torch.quantile(values, 0.95)
            result[f"{label}_torque_max"] = values.max()
        hard = torque.new_tensor([22.0] * 8 + [43.0] * 4)
        margin = hard - torque
        result["torque_margin_p50"] = torch.quantile(margin, 0.50)
        result["torque_margin_p05"] = torch.quantile(margin, 0.05)
        result["torque_margin_min"] = margin.min()
        result["torque_hard_violation_rate"] = (torque > hard).float().mean()
        result["torque_near_hard_rate"] = accumulator["torque_near_hard_count"] / count
        near_max = accumulator["torque_near_hard_max_frames"]
        result["torque_near_hard_max_duration_s"] = (
            near_max.max() * 0.02 if torch.is_tensor(near_max) else torque.new_zeros(())
        )
    result["p35_push_event_count"] = accumulator["push_event_count"]
    result["p35_push_events_per_min_per_env"] = (
        accumulator["push_event_count"] / count
    ) * (60.0 / p2_contract.CONTROL_DT_S)
    result["p35_push_env_coverage"] = (
        accumulator["push_env_seen"].float().mean()
        if torch.is_tensor(accumulator["push_env_seen"])
        else count.new_zeros(())
    )
    result["p35_push_runtime_active_share"] = accumulator["push_runtime_active"] / count
    result["p35_push_telemetry_valid_share"] = accumulator["push_telemetry_valid"] / count
    result["p35_seconds_since_push_min"] = torch.where(
        accumulator["push_seconds_since_min"] < 1.0e5,
        accumulator["push_seconds_since_min"],
        count.new_full((), -1.0),
    )
    post_count = accumulator["push_post_count"].clamp_min(1.0)
    result["p35_push_post_roll_peak"] = accumulator["push_post_roll_peak"]
    result["p35_push_post_pitch_peak"] = accumulator["push_post_pitch_peak"]
    result["p35_push_post_speed_mae_peak"] = accumulator["push_post_speed_mae_peak"]
    result["p35_push_post_fall_rate"] = accumulator["push_post_fall_count"] / post_count
    result["p35_push_post_timeout_rate"] = accumulator["push_post_timeout_count"] / post_count
    result["p35_push_config_violation_count"] = accumulator["push_config_violation_count"]
    if accumulator["push_delta_samples"]:
        delta = torch.cat(accumulator["push_delta_samples"], dim=0)
        for index, axis in enumerate(("vx", "vy")):
            values = delta[:, index].abs()
            result[f"p35_push_delta_{axis}_p50"] = torch.quantile(values, 0.50)
            result[f"p35_push_delta_{axis}_p95"] = torch.quantile(values, 0.95)
            result[f"p35_push_delta_{axis}_max"] = values.max()
            result[f"p35_push_delta_{axis}_positive_share"] = (delta[:, index] > 0).float().mean()
    else:
        for axis in ("vx", "vy"):
            for statistic in ("p50", "p95", "max", "positive_share"):
                result[f"p35_push_delta_{axis}_{statistic}"] = count.new_zeros(())
    if accumulator["push_recovery_time_samples"]:
        recovery = torch.cat(accumulator["push_recovery_time_samples"])
        result["p35_push_recovery_time_p50_s"] = torch.quantile(recovery, 0.50)
        result["p35_push_recovery_time_p95_s"] = torch.quantile(recovery, 0.95)
    else:
        result["p35_push_recovery_time_p50_s"] = count.new_full((), -1.0)
        result["p35_push_recovery_time_p95_s"] = count.new_full((), -1.0)
    for index, name in enumerate(("slope", "slope_inv", "stairs", "stairs_inv")):
        result[f"p35_push_terrain_{name}_count"] = accumulator["push_terrain_count"][index]
    for index in range(10):
        result[f"p35_push_level_{index}_count"] = accumulator["push_level_count"][index]
    for index, name in enumerate(("straight", "reserved_reverse", "vx_vy", "vx_wz", "pure_yaw", "brake_restart", "zero")):
        result[f"p35_push_command_{name}_count"] = accumulator["push_command_count"][index]
    for index, name in enumerate(("low", "medium", "high")):
        result[f"p35_push_speed_{name}_count"] = accumulator["push_speed_count"][index]
    for index, name in enumerate(("vx_positive", "vx_negative", "vy_positive", "vy_negative")):
        result[f"p35_push_direction_{name}_count"] = accumulator["push_direction_count"][index]
    for index, leg in enumerate(("fl", "fr", "rl", "rr")):
        onset_count = accumulator["gait_contact_onset"][index].clamp_min(1.0)
        result[f"{leg}_contact_onset_rate"] = accumulator["gait_contact_onset"][index] / count
        result[f"{leg}_impact_speed"] = accumulator["gait_impact"][index] / onset_count
        result[f"{leg}_touchdown_y"] = accumulator["gait_touchdown_y"][index] / onset_count
        result[f"{leg}_continuous_stance"] = accumulator["gait_stance"][index] / gait_count
    gait_bucket_total = accumulator["gait_bucket_count"].sum().clamp_min(1.0)
    for terrain_index, terrain in enumerate(P3GaitBaseline.TERRAIN_BUCKETS):
        for motion_index, motion in enumerate(P3GaitBaseline.MOTION_BUCKETS):
            bucket = terrain_index * len(P3GaitBaseline.MOTION_BUCKETS) + motion_index
            bucket_count = accumulator["gait_bucket_count"][bucket].clamp_min(1.0)
            result[f"p3_{terrain}_{motion}_sample_share"] = (
                accumulator["gait_bucket_count"][bucket] / gait_bucket_total
            )
            for leg_index, leg in enumerate(("fl", "fr", "rl", "rr")):
                result[f"p3_{terrain}_{motion}_{leg}_duty_factor"] = (
                    accumulator["gait_bucket_duty"][bucket, leg_index] / bucket_count
                )
                result[f"p3_{terrain}_{motion}_{leg}_step_frequency"] = (
                    accumulator["gait_bucket_frequency"][bucket, leg_index]
                    / bucket_count
                )
                result[f"p3_{terrain}_{motion}_{leg}_slip_distance"] = (
                    accumulator["gait_bucket_slip"][bucket, leg_index]
                    / accumulator["gait_bucket_slip_count"][bucket, leg_index].clamp_min(1.0)
                )
                result[f"p3_{terrain}_{motion}_{leg}_impact_speed"] = (
                    accumulator["gait_bucket_impact"][bucket, leg_index]
                    / accumulator["gait_bucket_impact_count"][bucket, leg_index].clamp_min(1.0)
                )
                result[f"p3_{terrain}_{motion}_{leg}_stance_s"] = (
                    accumulator["gait_bucket_stance"][bucket, leg_index] / bucket_count
                )
            frequency = accumulator["gait_bucket_frequency"][bucket] / bucket_count
            result[f"p3_{terrain}_{motion}_step_frequency_ratio"] = (
                frequency.max() / frequency.min().clamp_min(1.0e-6)
            )
            diagonal_a = 0.5 * (frequency[0] + frequency[3])
            diagonal_b = 0.5 * (frequency[1] + frequency[2])
            result[f"p3_{terrain}_{motion}_diagonal_frequency_relative"] = (
                (diagonal_a - diagonal_b).abs()
                / (0.5 * (diagonal_a + diagonal_b)).clamp_min(0.20)
            )
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
    command_accumulator = _command_accumulator(
        agent.device,
        push_limit_m_s=float(
            p3_contract.push_phase_config(
                agent.algorithm.session_effective_seconds,
                agent.algorithm.config.get("push_schedule") or {},
            )["max_velocity_xy_m_s"]
        ),
    )
    env_step_elapsed = 0.0
    if agent.algorithm.session_effective_seconds >= 900.0:
        agent.algorithm.gait_baseline.finalize()
        agent.algorithm.low_reward_shaper.finalize()
    mirror_fraction = p3_contract.mirror_training_fraction(
        agent.algorithm.session_effective_seconds
    )
    if not agent.algorithm.mirror_mapping_valid:
        mirror_fraction = 0.0
    agent.algorithm.mirror_aux.begin_rollout(mirror_fraction)
    agent.algorithm.depth_fault.begin_rollout(
        p3_contract.depth_fault_strength(agent.algorithm.session_effective_seconds)
    )
    agent.algorithm.memory_aux.begin_rollout(
        p3_contract.memory_training_fraction(agent.algorithm.session_effective_seconds)
    )
    agent.algorithm.camera_timing.begin_rollout(
        agent.algorithm.session_effective_seconds
    )
    agent.algorithm.action_smooth_aux.begin_rollout(
        p3_contract.action_smooth_training_fraction(
            agent.algorithm.session_effective_seconds,
            mapping_valid=agent.algorithm.mirror_mapping_valid,
        )
    )
    gait_reward_sums = torch.zeros(3, device=agent.device)
    p35_reward_sums = {
        name: torch.zeros((), device=agent.device)
        for name in (
            "progress", "default_posture", "joint_acc", "contact", "gait", "posture"
        )
    }
    p35_reward_raw_sums = {name: torch.zeros((), device=agent.device) for name in p35_reward_sums}
    p35_reward_eligible_sums = {
        name: torch.zeros((), device=agent.device) for name in p35_reward_sums
    }
    p35_reward_cap_correction = torch.zeros((), device=agent.device)
    for _ in range(storage.num_transitions_per_env):
        critic_obs, aux = split_p2_transport(critic_wire)
        p0, p1 = nav_contract.POLICY_CMD_SLICE
        target_command = obs[:, p0:p1].clone()
        exec_command = target_command.clone()
        command_epoch = _native_command_epoch(agent, target_command)
        aux = patch_owned_commands(
            aux, target_command, exec_command, command_epoch
        )
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
        clean_depth = obs[:, p3_contract.DEPTH_SLICE].reshape(
            obs.shape[0], p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, p2_contract.DEPTH_CHANNELS
        )
        depth, fault_changed = agent.algorithm.depth_fault.apply(
            clean_depth, storage.step
        )
        with torch.no_grad():
            fault_features = algorithm.actor_critic.vision_encoder.cnn(depth)
            timed_features, _ = agent.algorithm.camera_timing.step(
                fault_features
            )
            if train_low:
                critic_obs = critic_obs.clone()
                critic_obs[
                    :,
                    nav_contract.CRITIC_CMD_SLICE[0] : nav_contract.CRITIC_CMD_SLICE[1],
                ] = exec_command.to(critic_obs)
                step = _low_policy_step(
                    agent,
                    proprio,
                    depth,
                    critic_obs,
                    clean_depth=clean_depth,
                    cnn_features=timed_features,
                )
                actions = step["actions"]
                agent.algorithm.memory_aux.store(
                    storage.step,
                    step["clean_obs"],
                    fault_changed,
                    agent.algorithm.camera_timing.last_timing_changed,
                )
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
                actions = _low_policy_action(
                    agent, proprio, depth, cnn_features=timed_features
                )

        _accumulate_action_bucket(command_accumulator, agent._p3_extra, actions)

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
            anchor_weights = agent._p3_extra[
                :, p3_contract.COMMAND_ANCHOR_WEIGHT_INDEX
            ].reshape(-1, 1)
            valid_anchor = torch.isfinite(anchor_weights) & (
                (anchor_weights >= 0.0) & (anchor_weights <= 1.0)
            )
            transition.anchor_weights = torch.where(
                valid_anchor,
                anchor_weights,
                torch.zeros_like(anchor_weights),
            )

        env_step_started = time.perf_counter()
        step_data = env.step(torch.clamp(actions, -6.0, 6.0))
        env_step_elapsed += time.perf_counter() - env_step_started
        _, next_obs, rewards, terminated, truncated, infos, next_wire = _extract_step(
            step_data
        )
        next_obs = _as_device(next_obs, agent.device)
        next_wire = _consume_p3_wire(agent, _as_device(next_wire, agent.device))
        rewards = _as_device(rewards, agent.device).reshape(-1)
        next_aux = next_wire[:, p2_contract.CRITIC_OBS_DIM :]
        next_aux = patch_owned_commands(
            next_aux, target_command, exec_command, command_epoch
        )
        dones, timeouts = _frame_done_masks(
            terminated,
            truncated,
            infos,
            agent.device,
            worker_aux=next_aux,
        )
        _accumulate_outcomes(
            agent,
            command_accumulator,
            next_aux,
            agent.algorithm.depth_fault.near_clip_m,
        )
        _accumulate_joint_success(
            agent,
            command_accumulator,
            next_aux,
        )
        hard = dones & ~timeouts
        extra = agent._p3_extra
        base_healthy = (
            ~dones
            & (next_aux[:, p2_contract.BODY_COLLISION_FORCE_INDEX] < 30.0)
            & (next_aux[:, 21:23].abs().amax(dim=-1) < 0.35)
            & (torch.linalg.vector_norm(next_aux[:, 12:15] - exec_command, dim=-1) < 0.50)
            & torch.isfinite(next_aux).all(dim=-1)
            & torch.isfinite(next_obs[:, :45]).all(dim=-1)
        )
        joint_healthy = base_healthy & (
            extra[:, p3_contract.JOINT_ACCELERATION_MAPPING_VALID_INDEX] > 0.5
        ) & torch.isfinite(
            extra[:, p3_contract.JOINT_ACCELERATION_SLICE]
        ).all(dim=-1)
        contact_healthy = base_healthy & (
            extra[:, p3_contract.CONTACT_REWARD_MAPPING_VALID_INDEX] > 0.5
        ) & torch.isfinite(
            extra[:, p3_contract.CONTACT_FORCE_SLICE]
        ).all(dim=-1) & torch.isfinite(
            extra[:, p3_contract.CONTACT_ONSET_SLICE]
        ).all(dim=-1) & torch.isfinite(
            extra[:, p3_contract.CONTACT_OVER_THRESHOLD_DURATION_SLICE]
        ).all(dim=-1)
        gait_healthy = base_healthy & (
            next_aux[:, p2_contract.GAIT_VALID_INDEX] > 0.5
        ) & torch.isfinite(
            next_aux[:, p2_contract.GAIT_DUTY_SLICE]
        ).all(dim=-1) & torch.isfinite(
            next_aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE]
        ).all(dim=-1)
        if agent.algorithm.session_effective_seconds < 900.0:
            agent.algorithm.gait_baseline.observe(
                next_aux,
                extra,
                gait_healthy,
                collect_continuous=(
                    storage.step % p3_contract.GAIT_BASELINE_CONTINUOUS_STRIDE == 0
                ),
            )
            agent.algorithm.low_reward_shaper.observe(
                next_obs[:, :45],
                next_aux,
                extra,
                base_healthy,
                joint_healthy=joint_healthy,
                contact_healthy=contact_healthy,
                gait_healthy=gait_healthy,
                collect_continuous=(
                    storage.step % p3_contract.GAIT_BASELINE_CONTINUOUS_STRIDE == 0
                ),
            )
        contact_reward, cross_reward, starvation_reward = (
            agent.algorithm.gait_baseline.rewards(
                next_aux,
                extra,
                1.0 if agent.algorithm.gait_baseline.finalized else 0.0,
            )
        )
        if storage.step % p2_contract.NAV_PERIOD_FRAMES != p2_contract.NAV_PERIOD_FRAMES - 1:
            starvation_reward.zero_()
        gait_reward_sums += torch.stack(
            (contact_reward.mean(), cross_reward.mean(), starvation_reward.mean())
        )
        p35_reward, p35_components = agent.algorithm.low_reward_shaper.rewards(
            proprio=next_obs[:, :45],
            aux=aux,
            next_aux=next_aux,
            extra=extra,
            command=exec_command,
            dones=dones,
            scale=p3_contract.p35_reward_training_fraction(
                agent.algorithm.session_effective_seconds
            ),
        )
        for name, values in p35_components.items():
            p35_reward_sums[name] += values.mean()
            p35_reward_raw_sums[name] += agent.algorithm.low_reward_shaper.last_raw_components[
                name
            ].mean()
            p35_reward_eligible_sums[name] += agent.algorithm.low_reward_shaper.last_eligible[
                name
            ].float().mean()
        p35_reward_cap_correction += (
            agent.algorithm.low_reward_shaper.last_cap_correction.mean()
        )
        rewards = rewards + p35_reward
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
        agent.algorithm.camera_timing.reset(dones)
        obs, critic_wire = next_obs, next_wire
        reward_sum += rewards.mean()
        done_sum += dones.float().mean()
        _advance_platform_lifecycle(agent)

    if train_low:
        with torch.no_grad():
            next_critic, _ = split_p2_transport(critic_wire)
            last_values = algorithm.actor_critic.evaluate(next_critic).detach()
        storage.compute_returns(last_values, algorithm.gamma, algorithm.lam)
        metrics = algorithm.learn(
            agent.training_elapsed_h,
            phase_override=agent.algorithm.current_phase,
        )
        low_updated = _finalize_low_level_update(agent, metrics)
        if low_updated:
            metrics.update(_run_adapter_updates(agent, 1))
        else:
            metrics.update(_run_adapter_updates(agent, 0))
    else:
        metrics = {
            "applied_updates": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "low_update_time_s": 0.0,
            "low_actor_update_active": 0.0,
            "low_critic_update_active": 0.0,
        }
        metrics.update(
            _run_adapter_updates(
                agent,
                p3_contract.adapter_updates_per_low_rollout(
                    agent.algorithm.session_effective_seconds
                ),
            )
        )
    storage_bytes = _storage_bytes(storage)
    storage.clear()
    fallback_shares = agent.algorithm.gait_baseline.fallback_level_shares()
    metrics.update(
        {
            "p3_low_policy_loss": float(metrics.get("policy_loss", 0.0)),
            "p3_low_value_loss": float(metrics.get("value_loss", 0.0)),
            "p3_low_entropy": float(metrics.get("entropy_loss", 0.0)),
            "p3_low_approx_kl": float(metrics.get("approx_kl", 0.0)),
            "p3_low_clip_fraction": float(metrics.get("clip_fraction", 0.0)),
            "p3_low_update_time_s": float(metrics.get("low_update_time_s", 0.0)),
            "p3_low_actor_update_active": float(
                metrics.get("low_actor_update_active", 0.0)
            ),
            "p3_low_critic_update_active": float(
                metrics.get("low_critic_update_active", 0.0)
            ),
            "low_reward_mean": float(reward_sum / storage.num_transitions_per_env),
            "low_done_rate": float(done_sum / storage.num_transitions_per_env),
            "shadow_p3_contact_quality": float(gait_reward_sums[0] / storage.num_transitions_per_env),
            "shadow_p3_crossing": float(gait_reward_sums[1] / storage.num_transitions_per_env),
            "shadow_p3_starvation": float(gait_reward_sums[2] / storage.num_transitions_per_env),
            "p3_gait_baseline_finalized": float(agent.algorithm.gait_baseline.finalized),
            "p3_gait_baseline_fallback_share": float(agent.algorithm.gait_baseline.fallback_share),
            **{
                f"p3_gait_baseline_{name}_share": float(value)
                for name, value in fallback_shares.items()
            },
            "gait_baseline_samples": float(agent.algorithm.gait_baseline.total_samples),
            "p35_reward_baseline_valid": float(agent.algorithm.low_reward_shaper.valid),
            "p35_reward_scale": float(
                p3_contract.p35_reward_training_fraction(
                    agent.algorithm.session_effective_seconds
                )
            ),
            "p35_reward_cap_correction": float(
                p35_reward_cap_correction / storage.num_transitions_per_env
            ),
            **{
                f"p35_reward_{name}": float(
                    value / storage.num_transitions_per_env
                )
                for name, value in p35_reward_sums.items()
            },
            **{
                f"p35_reward_raw_{name}": float(
                    value / storage.num_transitions_per_env
                )
                for name, value in p35_reward_raw_sums.items()
            },
            **{
                f"p35_reward_eligible_{name}": float(
                    value / storage.num_transitions_per_env
                )
                for name, value in p35_reward_eligible_sums.items()
            },
            **{
                f"p35_baseline_samples_{name}": float(value)
                for name, value in agent.algorithm.low_reward_shaper.sample_count.items()
            },
            **{
                f"p35_baseline_seen_{name}": float(value)
                for name, value in agent.algorithm.low_reward_shaper.sample_seen_count.items()
            },
            **{
                f"p35_baseline_component_valid_{name}": float(value)
                for name, value in agent.algorithm.low_reward_shaper.component_valid.items()
            },
            **agent.algorithm.low_reward_shaper.diagnostics(),
            "p3_low_storage_bytes": float(storage_bytes),
            "env_step_time_s": env_step_elapsed,
            **agent.algorithm.depth_fault.diagnostics(),
            **agent.algorithm.camera_timing.diagnostics(),
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
    command_accumulator = _command_accumulator(
        agent.device,
        push_limit_m_s=float(
            p3_contract.push_phase_config(
                agent.algorithm.session_effective_seconds,
                agent.algorithm.config.get("push_schedule") or {},
            )["max_velocity_xy_m_s"]
        ),
    )
    env_step_elapsed = 0.0
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
            env_step_started = time.perf_counter()
            step_data = env.step(torch.clamp(result["actions"], -6.0, 6.0))
            env_step_elapsed += time.perf_counter() - env_step_started
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
            _accumulate_outcomes(
                agent,
                command_accumulator,
                next_aux,
                agent.algorithm.depth_fault.near_clip_m,
            )
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
    metrics.update(
        {
            "p3_high_actor_loss": float(metrics.get("actor_loss", 0.0)),
            "p3_high_critic_loss": float(metrics.get("critic_loss", 0.0)),
            "p3_high_entropy": float(metrics.get("entropy", 0.0)),
            "p3_high_approx_kl": float(metrics.get("approx_kl", 0.0)),
            "p3_high_clip_fraction": float(metrics.get("clip_fraction", 0.0)),
            "p3_high_reward_mean": float(metrics.get("rollout_reward_mean", 0.0)),
            "p3_high_update_time_s": float(
                metrics.get("actor_update_time_s", 0.0)
                + metrics.get("critic_update_time_s", 0.0)
            ),
            "p3_high_actor_update_active": float(algorithm._actor_update_enabled()),
        }
    )
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
            "env_step_time_s": env_step_elapsed,
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


def _monitor_contract_metrics(agent, metrics: dict, elapsed_s: float) -> dict[str, float]:
    last_finite = getattr(agent, "_p35_monitor_last_finite_s", None)
    if not isinstance(last_finite, dict):
        last_finite = {}
        agent._p35_monitor_last_finite_s = last_finite
    registered = 0
    with_data = 0
    ages = []
    for name in P35_REQUIRED_MONITOR_METRICS:
        if name in metrics:
            registered += 1
        value = metrics.get(name)
        finite = False
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            pass
        if finite:
            with_data += 1
            last_finite[name] = float(elapsed_s)
        last_seen = last_finite.get(name)
        ages.append(
            float(elapsed_s) if last_seen is None else max(0.0, float(elapsed_s) - last_seen)
        )
    expected = len(P35_REQUIRED_MONITOR_METRICS)
    return {
        "p35_monitor_expected_metric_count": float(expected),
        "p35_monitor_registered_metric_count": float(registered),
        "p35_monitor_metric_with_data_count": float(with_data),
        "p35_monitor_empty_metric_count": float(expected - with_data),
        "p35_monitor_longest_data_age_s": max(ages, default=0.0),
        "p35_monitor_semantic_health": float(
            elapsed_s < 900.0 or agent.algorithm.low_reward_shaper.valid
        ),
    }


def _p35_baseline_gate_reason(agent, elapsed_s: float) -> str | None:
    shaper = agent.algorithm.low_reward_shaper
    if elapsed_s >= 300.0 and not getattr(agent, "_p35_baseline_5m_gate_passed", False):
        required_continuous = (
            "joint_pos", "joint_acc_noncontact", "posture", "frequency"
        )
        empty = [
            name for name in required_continuous
            if shaper.sample_count.get(name, 0) <= 0
        ]
        eligibility_empty = [
            name for name in ("base", "joint", "contact", "gait")
            if shaper.eligibility.get(name, 0) <= 0
        ]
        if empty or eligibility_empty:
            return f"five_minute_empty_samples={empty},eligibility={eligibility_empty}"
        agent._p35_baseline_5m_gate_passed = True
    if elapsed_s >= 900.0 and shaper.finalized and not shaper.valid:
        disabled = [
            name for name, valid in shaper.component_valid.items() if not valid
        ]
        return f"fifteen_minute_invalid_components={disabled}"
    return None


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
            baseline_finalized_now = False
            if elapsed >= 900.0 and not agent.algorithm.gait_baseline.finalized:
                agent.algorithm.gait_baseline.finalize()
                baseline_finalized_now = True
            if elapsed >= 900.0 and not agent.algorithm.low_reward_shaper.finalized:
                agent.algorithm.low_reward_shaper.finalize()
                baseline_finalized_now = True
            if baseline_finalized_now:
                logger.info(
                    "[P35Baseline] "
                    f"valid={int(agent.algorithm.low_reward_shaper.valid)} "
                    f"samples={agent.algorithm.low_reward_shaper.sample_count} "
                    "threshold_ranges="
                    f"{ {str(name): [float(value.min()), float(value.max())] for name, value in agent.algorithm.low_reward_shaper.thresholds.items()} } "
                    "component_valid="
                    f"{agent.algorithm.low_reward_shaper.component_valid} "
                    "disabled_rewards="
                    f"{[name for name, valid in agent.algorithm.low_reward_shaper.component_valid.items() if not valid]}"
                )
            agent.training_elapsed_h = elapsed / 3600.0
            agent.algorithm.current_iteration += 1
            metrics.update(
                {
                    "p3_session_effective_seconds": elapsed,
                    "p3_lifetime_effective_seconds": (
                        agent.algorithm.lifetime_base_seconds + elapsed
                    ),
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
                    "p3_m3_target_boundary_gap_m": (
                        agent.algorithm.platform_boundary_radius_m
                        - agent.algorithm.m3_target_radius_m
                    ),
                    "p3_m3_proxy_target_gap_m": (
                        agent.algorithm.m3_target_radius_m
                        - agent.algorithm.platform_complete_radius_m
                    ),
                    **agent.algorithm.memory_metrics(),
                }
            )
            rollout_frames = (
                p2_contract.NAV_ROLLOUT_TICKS * p2_contract.NAV_PERIOD_FRAMES
                if collect_high
                else int(
                    getattr(
                        agent.low_level_algorithm.storage,
                        "num_transitions_per_env",
                        conf.get("num_steps_per_env", 128),
                    )
                )
            )
            metrics["samples_per_s"] = (
                float(agent.num_envs * rollout_frames)
                / max(now - loop_started, 1.0e-6)
            )
            realized = p3_contract.materialize_environment_config(usr_conf, elapsed)
            domain_rand = realized.get("domain_rand", {})
            runtime = realized.get("p3_runtime", {})
            push_phase = p3_contract.push_phase_config(
                elapsed, conf.get("push_schedule") or {}
            )
            metrics.update(
                {
                    "p3_dr_phase": float(runtime.get("phase_index", 0)),
                    "p3_friction_min": float(domain_rand.get("friction_range", [1, 1])[0]),
                    "p3_friction_max": float(domain_rand.get("friction_range", [1, 1])[1]),
                    "p3_base_added_mass_kg": float(runtime.get("base_added_mass_kg", 0.0)),
                    "p3_noise_level": float((realized.get("noise", {}) or {}).get("noise_level", 0.0)),
                    "p3_restitution_max": float(domain_rand.get("restitution_range", [0, 0])[1]),
                    "p3_push_enabled": float(push_phase["active"]),
                    "p3_push_velocity_m_s": float(push_phase["max_velocity_xy_m_s"]),
                    "p3_push_runtime_telemetry_available": float(
                        metrics.get("p35_push_telemetry_valid_share", 0.0) > 0.99
                    ),
                    "p3_dr_runtime_telemetry_available": 0.0,
                }
            )
            metrics.update(_monitor_contract_metrics(agent, metrics, elapsed))
            baseline_gate_reason = _p35_baseline_gate_reason(agent, elapsed)
            if baseline_gate_reason is not None:
                agent.algorithm.p35_diagnostic_stop_reason = baseline_gate_reason
                agent.algorithm.checkpoint_label_override = "baselinefault"
                logger.error(
                    "[P35Baseline] hard_gate=1 reason="
                    f"{baseline_gate_reason} samples="
                    f"{agent.algorithm.low_reward_shaper.sample_count} eligibility="
                    f"{agent.algorithm.low_reward_shaper.eligibility}"
                )
                agent.save_model()
                raise RuntimeError(f"P35 baseline correctness gate failed: {baseline_gate_reason}")
            push_preflight_valid = float(
                metrics.get("p35_push_telemetry_valid_share", 0.0) > 0.99
            )
            metrics.update(
                {
                    "p35_push_term_exists": push_preflight_valid,
                    "p35_push_mode_correct": push_preflight_valid,
                    "p35_push_wrapper_installed": push_preflight_valid,
                    "p35_push_runtime_api_available": push_preflight_valid,
                }
            )
            if changed:
                agent.save_model()
                logger.info(
                    f"[P3] rollout-boundary phase change {phase_before} -> "
                    f"{agent.algorithm.current_phase}; "
                    "episode_reset=True; environment_contract=p35_gaitfix8h_static_dr_dynamic_push_v1"
                )
                # Phase boundaries clear recurrent/live state, but platform-owned
                # physics and EventManager objects remain the startup instances.
                obs, critic_wire = _reset_env(env, agent, usr_conf)
            if elapsed >= next_save:
                agent.save_model()
                while next_save <= elapsed:
                    next_save += save_interval
            if now - last_log >= 60.0 or agent.algorithm.current_iteration == 1:
                logger.info(
                    f"[P35TrainState] iter={agent.algorithm.current_iteration} "
                    f"session_h={elapsed / 3600.0:.3f} "
                    f"phase={agent.algorithm.current_phase} "
                    f"low_updates={agent.algorithm.low_updates} "
                    f"high_updates={agent.algorithm.high_updates}"
                )
                logger.info(
                    "[P35PushSummary] "
                    f"events={metrics.get('p35_push_event_count', 0.0)} "
                    f"telemetry={metrics.get('p35_push_telemetry_valid_share', 0.0)} "
                    f"delta_vx_p95={metrics.get('p35_push_delta_vx_p95', 0.0)} "
                    f"delta_vy_p95={metrics.get('p35_push_delta_vy_p95', 0.0)} "
                    f"post_fall_rate={metrics.get('p35_push_post_fall_rate', 0.0)}"
                )
                logger.info(
                    "[P35MonitorContract] "
                    f"registered={int(metrics['p35_monitor_registered_metric_count'])} "
                    f"expected={int(metrics['p35_monitor_expected_metric_count'])} "
                    f"with_data={int(metrics['p35_monitor_metric_with_data_count'])} "
                    f"empty={int(metrics['p35_monitor_empty_metric_count'])} "
                    f"max_age_s={metrics['p35_monitor_longest_data_age_s']:.1f}"
                )
                _monitor_put(monitor, metrics)
                last_log = now
        agent.save_model()
    except BaseException as exc:
        if not getattr(agent.algorithm, "p35_diagnostic_stop_reason", None):
            agent.algorithm.p35_diagnostic_stop_reason = (
                f"{type(exc).__name__}:{exc}"
            )
        if not getattr(agent.algorithm, "checkpoint_label_override", None):
            agent.algorithm.checkpoint_label_override = "emergency"
        try:
            agent.save_model()
            logger.error(
                "[P35EmergencySave] saved=1 exception="
                f"{type(exc).__name__}:{exc}"
            )
        except Exception as save_exc:
            logger.error(
                "[P35EmergencySave] saved=0 exception="
                f"{type(exc).__name__}:{exc} save_error="
                f"{type(save_exc).__name__}:{save_exc}"
            )
        raise
    finally:
        if hasattr(signal, "SIGTERM") and previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
