#!/usr/bin/env python3
"""Shared P2/P4 navigation PPO orchestration."""

from __future__ import annotations

import math
import signal
import time
from collections import deque
from types import SimpleNamespace

import torch

from agent_ppo.checkpoint_io import CheckpointSaveError
from agent_ppo.conf.conf import Config
from agent_ppo.feature import nav_contract, p2_contract
from agent_ppo.workflow.nav_ppo_metrics import build_nav_rollout_metrics
from agent_ppo.workflow.nav_ppo_support import (
    MonitorHealthTracker,
    NavWorkflowHooks,
    WorkflowSpec,
    _CollisionTraceRecorder,
    _advance_runtime_clocks_and_saves,
    _extract_step,
    _final_save,
    _frame_done_masks,
    _goal_history_stuck,
    _install_sigterm_handler,
    _maybe_report_monitor,
    _resolve_terminal_outcome,
    _terminal_safe_segment,
    _terminal_safe_tensor,
    _tick_diagnostic_values,
)


def _segment_labels(usr_conf: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    terrain_track = (usr_conf.get("terrain") or {}).get("track") or {}
    labels = p2_contract.canonical_track_segment_labels(
        terrain_track.get("sub_terrains", ())
    )
    metric_labels = (
        p2_contract.TRACK_SEGMENT_METRIC_LABELS
        if all(label in p2_contract.TRACK_SEGMENT_METRIC_LABELS for label in labels)
        else p2_contract.CANONICAL_TRACK_SEGMENT_METRIC_LABELS
    )
    return labels, metric_labels


def _next_save_clock(resumed: float, first: float, interval: float) -> float:
    if first <= 0.0 or interval <= 0.0:
        raise ValueError("P2 checkpoint intervals must be positive")
    if resumed <= 0.0:
        return first
    return (math.floor(resumed / interval) + 1) * interval


def _prepare_runtime(envs, agents, logger, monitor, spec, hooks):
    agent = agents[0]
    env = envs[0]
    if not spec.accepts(agent):
        raise RuntimeError(
            f"{spec.name} workflow requires agent flag {spec.agent_flag!r}"
        )
    usr_conf, conf_path, _, stage = Config.load_conf(logger)
    del stage
    stage_conf = usr_conf.get(spec.config_key, {})
    segment_labels, segment_metric_labels = _segment_labels(usr_conf)
    feedback_conf = stage_conf.get("feedback_profile", {})
    if not isinstance(feedback_conf, dict):
        feedback_conf = {}
    algorithm = agent.algorithm
    target = hooks.resolve_target(spec, algorithm, stage_conf)
    logger.info(
        "[P2NavPPO] start "
        f"conf={conf_path} envs={agent.num_envs} parent={agent._p2_parent_model_id} "
        "rollout=32 ticks tbptt=16 minibatch=64seq microbatch=4seq "
        f"target_hours={float(target.target_hours):.1f} performance_gates=none"
    )
    hooks.configure_worker_resume(algorithm, stage_conf, logger)
    data = env.reset(usr_conf)
    if data is None:
        raise RuntimeError("P2 env.reset returned None")
    obs, critic_wire = data
    obs = torch.as_tensor(obs).to(agent.device).clone()
    critic_wire = torch.as_tensor(critic_wire).to(agent.device).clone()
    if obs.shape != (agent.num_envs, nav_contract.POLICY_OBS_DIM):
        raise ValueError(f"P2 reset policy shape drift: {tuple(obs.shape)}")
    expected_wire_dim = int(spec.expected_wire_dim)
    if critic_wire.shape != (agent.num_envs, expected_wire_dim):
        raise ValueError(
            f"navigation reset critic wire shape drift: {tuple(critic_wire.shape)}; "
            f"expected=({agent.num_envs},{expected_wire_dim})"
        )
    algorithm.reset_live_state()
    resumed_seconds = float(algorithm.session_effective_seconds)
    resumed_clock_seconds = float(
        getattr(algorithm, "session_wall_seconds", resumed_seconds)
    )
    session_started = time.monotonic()
    first_save = float(stage_conf.get("first_save_minutes", 5.0)) * 60.0
    save_interval = float(stage_conf.get("save_interval_minutes", 10.0)) * 60.0
    resumed_save = hooks.resumed_save_clock(resumed_seconds, resumed_clock_seconds)
    schedule_boundaries = tuple(hooks.schedule_boundaries(algorithm))
    required_metrics = tuple(hooks.monitor_required_metrics())
    nav_dt_s = float(algorithm.nav_dt_s)
    runtime = SimpleNamespace(
        agent=agent,
        env=env,
        logger=logger,
        monitor=monitor,
        hooks=hooks,
        target=target,
        obs=obs,
        critic_wire=critic_wire,
        algorithm=algorithm,
        feedback_age_clip_s=float(feedback_conf.get("age_clip_s", 0.8)),
        segment_labels=segment_labels,
        segment_metric_labels=segment_metric_labels,
        resumed_seconds=resumed_seconds,
        resumed_clock_seconds=resumed_clock_seconds,
        session_started=session_started,
        variant_training_seconds=0.0,
        save_interval_s=save_interval,
        next_save_clock=_next_save_clock(resumed_save, first_save, save_interval),
        retry_save_at=None,
        unfreeze_save_done=bool(algorithm.cnn_unfrozen),
        lifecycle_success=0,
        lifecycle_failures=0,
        nav_period_frames=int(algorithm.nav_period_frames),
        nav_rollout_ticks=int(algorithm.nav_rollout_ticks),
        nav_dt_s=nav_dt_s,
        collision_traces=_CollisionTraceRecorder(
            nav_dt_s=nav_dt_s,
            pre_ticks=max(1, int(round(1.0 / nav_dt_s))),
            post_ticks=max(1, int(round(2.0 / nav_dt_s))),
        ),
        goal_history=deque(maxlen=max(2, int(round(2.0 / nav_dt_s)) + 1)),
        schedule_boundaries=schedule_boundaries,
        saved_schedule_boundaries={
            boundary for boundary in schedule_boundaries if boundary <= resumed_seconds
        },
        last_log=time.monotonic(),
        monitor_health=(
            MonitorHealthTracker(required_metrics, started_at=session_started)
            if required_metrics
            else None
        ),
        cumulative_counter_previous={},
        variant_state=hooks.initialize_variant_state(agent, algorithm),
    )
    runtime.previous_sigterm = _install_sigterm_handler(logger)
    agent._p2_training_started = True
    agent._p2_final_save_done = False
    return runtime


def _new_command_buffers(device) -> dict[str, torch.Tensor]:
    counts = torch.zeros(5, 4, device=device)
    return {
        name: torch.zeros_like(counts)
        for name in (
            "counts",
            "progress",
            "tracking",
            "tracking_counts",
            "target_vx",
            "exec_vx",
            "true_vx",
            "target_wz",
            "exec_wz",
            "true_wz",
            "gait",
            "success",
            "failure",
            "timeout",
            "vy_nonzero",
            "abs_vy",
            "vy_outer",
        )
    }


def _new_segment_buffers(count: int, device) -> dict[str, torch.Tensor]:
    rows = torch.zeros(count, device=device)
    return {
        name: torch.zeros_like(rows)
        for name in (
            "counts",
            "progress",
            "target_vx",
            "exec_vx",
            "true_vx",
            "target_wz",
            "stop",
            "creep",
            "spin",
            "gait",
            "success",
            "failure",
            "timeout",
            "reason4",
            "predictive_clearance",
            "predictive_risk",
            "predictive_penalty",
            "teacher_risk",
            "teacher_high_risk",
        )
    }


def _new_rollout(runtime):
    agent = runtime.agent
    quantile_names = (
        "target_vx", "target_vy", "target_wz",
        "exec_vx", "exec_vy", "exec_wz",
        "true_vx", "true_vy", "true_wz",
        "measured_vx", "measured_vy", "measured_wz",
    )
    return SimpleNamespace(
        rollout_started=time.monotonic(),
        env_step_time_s=0.0,
        diagnostic_sums={},
        diagnostic_maxima={},
        diagnostic_valid_sums={},
        diagnostic_count=0,
        diagnostic_valid_count=torch.zeros((), device=agent.device),
        command_bins=_new_command_buffers(agent.device),
        segment_rows=_new_segment_buffers(
            len(runtime.segment_metric_labels), agent.device
        ),
        segment_diagnostic_count=torch.zeros((), device=agent.device),
        quantile_samples={
            name: torch.full(
                (runtime.nav_rollout_ticks, agent.num_envs),
                float("nan"),
                device=agent.device,
            )
            for name in quantile_names
        },
        cumulative_counter_names=runtime.hooks.cumulative_counter_names(),
        variant=runtime.hooks.initialize_rollout_variant(
            agent, len(runtime.segment_metric_labels)
        ),
    )


def _new_tick(runtime):
    agent = runtime.agent
    hard = torch.zeros(agent.num_envs, dtype=torch.bool, device=agent.device)
    return SimpleNamespace(
        start_goal=(
            runtime.critic_wire[:, nav_contract.CRITIC_GOAL3_START + 2]
            * nav_contract.GOAL_DIST_SCALE_M
        ).detach(),
        frame_safety_reward=torch.zeros(agent.num_envs, device=agent.device),
        terminal_goal=torch.full(
            (agent.num_envs,), float("nan"), device=agent.device
        ),
        terminal_segment=torch.full(
            (agent.num_envs,), -1.0, device=agent.device
        ),
        terminal_target=torch.zeros(agent.num_envs, 3, device=agent.device),
        terminal_executed=torch.zeros(agent.num_envs, 3, device=agent.device),
        terminal_aux=torch.zeros(
            agent.num_envs, p2_contract.WORKER_AUX_DIM, device=agent.device
        ),
        terminal_extra=runtime.hooks.make_terminal_extra(agent),
        duration=torch.zeros(agent.num_envs, dtype=torch.long, device=agent.device),
        path_length_m=torch.zeros(agent.num_envs, device=agent.device),
        hard=hard,
        timeout=torch.zeros_like(hard),
        unattributed_boundary=torch.zeros_like(hard),
        terminal_reason=torch.zeros(
            agent.num_envs, dtype=torch.long, device=agent.device
        ),
        active=torch.ones_like(hard),
    )


def _record_lifecycle(runtime) -> None:
    try:
        runtime.agent.learn(None)
    except Exception as exc:
        runtime.lifecycle_failures += 1
        if isinstance(exc, CheckpointSaveError) and runtime.retry_save_at is None:
            runtime.retry_save_at = time.monotonic() + 60.0
        if runtime.lifecycle_failures == 1 or runtime.lifecycle_failures % 100 == 0:
            runtime.logger.error(
                "[P2NavPPO] lifecycle callback failed; training continues "
                f"failures={runtime.lifecycle_failures} "
                f"error={type(exc).__name__}: {exc}"
            )
    else:
        runtime.lifecycle_success += 1


def _capture_terminal_rows(runtime, tick, frame_target, frame_executed, frame_extra, next_critic, next_aux, new_done):
    if not bool(new_done.any()):
        return
    tick.terminal_target[new_done] = frame_target[new_done]
    tick.terminal_executed[new_done] = frame_executed[new_done]
    tick.terminal_aux[new_done] = next_aux[new_done]
    runtime.hooks.capture_terminal_extra(
        tick.terminal_extra, frame_extra, next_critic, new_done
    )
    tick.terminal_goal[new_done] = next_aux[
        new_done, p2_contract.PRE_STEP_GOAL_DISTANCE_INDEX
    ]
    tick.terminal_segment[new_done] = next_aux[
        new_done, p2_contract.CURRENT_SEGMENT_INDEX
    ]


def _run_control_frames(runtime, rollout, tick) -> None:
    algorithm = runtime.algorithm
    agent = runtime.agent
    for frame in range(runtime.nav_period_frames):
        result, _critic_obs, aux = algorithm.frame_begin(
            runtime.obs, runtime.critic_wire
        )
        actions = torch.clamp(result["actions"], -6.0, 6.0)
        frame_target = algorithm.command.active_target
        frame_executed = algorithm.command.exec_cmd
        frame_true_xy = torch.nan_to_num(
            aux[:, 12:14], nan=0.0, posinf=0.0, neginf=0.0
        )
        tick.path_length_m += (
            torch.linalg.vector_norm(frame_true_xy, dim=-1)
            * p2_contract.CONTROL_DT_S
            * tick.active.float()
        )
        frame_extra = runtime.hooks.frame_extra(algorithm)
        if frame > 0 and bool((~tick.active).any()):
            algorithm.command.active_target[~tick.active] = 0.0
            algorithm.command.exec_cmd[~tick.active] = 0.0
        step_started = time.perf_counter()
        step_data = runtime.env.step(actions)
        rollout.env_step_time_s += time.perf_counter() - step_started
        _, next_obs, rewards, terminated, truncated, infos, next_critic = _extract_step(
            step_data
        )
        next_obs = torch.as_tensor(next_obs).to(agent.device)
        next_critic = torch.as_tensor(next_critic).to(agent.device)
        rewards = torch.as_tensor(rewards).to(agent.device).reshape(-1)
        next_aux = next_critic[
            :, p2_contract.CRITIC_OBS_DIM : p2_contract.PRIVILEGED_WIRE_DIM
        ]
        frame_done, frame_timeout = _frame_done_masks(
            terminated, truncated, infos, agent.device, worker_aux=next_aux
        )
        tick.frame_safety_reward += (
            rewards * tick.active.float() * (p2_contract.GAMMA_FRAME ** frame)
        )
        tick.duration += tick.active.long()
        new_done = tick.active & frame_done
        reason, new_hard, new_timeout = _resolve_terminal_outcome(
            new_done, frame_timeout, next_aux[:, 25].round().long()
        )
        tick.terminal_reason[new_done] = reason[new_done]
        _capture_terminal_rows(
            runtime,
            tick,
            frame_target,
            frame_executed,
            frame_extra,
            next_critic,
            next_aux,
            new_done,
        )
        tick.hard |= new_hard
        tick.timeout |= new_timeout
        tick.unattributed_boundary |= new_done & (reason == 0)
        tick.active &= ~frame_done
        algorithm.frame_end(aux, frame_done)
        runtime.obs, runtime.critic_wire = next_obs, next_critic
        _record_lifecycle(runtime)
    tick.actions = actions
    tick.next_aux = next_aux


def _resolve_tick_values(runtime, tick) -> None:
    live_end_goal = (
        runtime.critic_wire[:, nav_contract.CRITIC_GOAL3_START + 2]
        * nav_contract.GOAL_DIST_SCALE_M
    )
    tick.transition_done = tick.hard | tick.timeout | tick.unattributed_boundary
    tick.end_goal = torch.where(
        tick.transition_done & torch.isfinite(tick.terminal_goal),
        tick.terminal_goal,
        live_end_goal,
    )
    if bool(tick.transition_done.any()):
        for previous_goal in runtime.goal_history:
            previous_goal[tick.transition_done] = float("nan")
    runtime.goal_history.append(tick.end_goal.detach().clone())
    stuck = _goal_history_stuck(runtime.goal_history, tick.end_goal)
    pending = runtime.algorithm.pending_tick
    if pending is None:
        raise RuntimeError("P2 pending transition disappeared before finish_tick")
    tick.target = _terminal_safe_tensor(
        runtime.algorithm.command.active_target,
        tick.terminal_target,
        tick.transition_done,
    )
    tick.executed = _terminal_safe_tensor(
        runtime.algorithm.command.exec_cmd,
        tick.terminal_executed,
        tick.transition_done,
    )
    tick.diagnostic_aux = _terminal_safe_tensor(
        tick.next_aux, tick.terminal_aux, tick.transition_done
    )
    tick.values, tick.valid_values, tick.valid_mask = _tick_diagnostic_values(
        target=tick.target,
        executed=tick.executed,
        response_aux=tick.diagnostic_aux,
        confidence=pending["confidence"],
        actions=tick.actions,
        start_goal=tick.start_goal,
        end_goal=tick.end_goal,
        done=tick.transition_done,
        hard=tick.hard,
        timeout=tick.timeout,
        terminal_reason=tick.terminal_reason,
        duration_frames=tick.duration,
        stuck=stuck,
        feedback_age_clip_s=runtime.feedback_age_clip_s,
        nav_period_frames=runtime.nav_period_frames,
    )


def _add_sum(target: dict, name: str, value: torch.Tensor, device) -> None:
    target[name] = target.get(name, torch.zeros((), device=device)) + value.sum()


def _accumulate_base_diagnostics(runtime, rollout, tick, tick_index: int) -> None:
    for name, value in tick.values.items():
        _add_sum(rollout.diagnostic_sums, name, value, runtime.agent.device)
    rollout.diagnostic_valid_count += tick.valid_mask.float().sum()
    for name, value in tick.valid_values.items():
        _add_sum(
            rollout.diagnostic_valid_sums,
            name,
            torch.where(tick.valid_mask, value, torch.zeros_like(value)),
            runtime.agent.device,
        )
    for name in (
        "target_vx", "target_vy", "target_wz",
        "exec_vx", "exec_vy", "exec_wz",
        "true_vx", "true_vy", "true_wz",
    ):
        rollout.quantile_samples[name][tick_index].copy_(tick.values[name].detach())
    for name in ("measured_vx", "measured_vy", "measured_wz"):
        rollout.quantile_samples[name][tick_index].copy_(
            torch.where(
                tick.valid_mask,
                tick.valid_values[name].detach(),
                torch.full_like(tick.valid_values[name], float("nan")),
            )
        )


def _scatter_flat(buffer: torch.Tensor, index: torch.Tensor, source) -> None:
    buffer.view(-1).scatter_add_(0, index, source.float())


def _accumulate_command_bins(runtime, rollout, tick) -> None:
    buffers = rollout.command_bins
    vx_bin = torch.bucketize(
        tick.target[:, 0], torch.tensor((0.1, 0.4, 0.8, 1.0), device=runtime.agent.device)
    )
    wz_bin = torch.bucketize(
        tick.target[:, 2].abs(),
        torch.tensor((0.1, 0.4, 0.8), device=runtime.agent.device),
    )
    flat_bin = vx_bin * 4 + wz_bin
    tick.command_flat_bin = flat_bin
    tracking = (
        (tick.executed - tick.diagnostic_aux[:, 12:15]).abs()
        / torch.tensor(p2_contract.COMMAND_NORMALIZATION, device=runtime.agent.device)
    ).mean(dim=-1)
    valid = (~tick.transition_done).float()
    sources = {
        "counts": torch.ones_like(tick.target[:, 0]),
        "vy_nonzero": tick.target[:, 1].abs() > 0.05,
        "abs_vy": tick.target[:, 1].abs(),
        "vy_outer": tick.target[:, 1].abs() > 0.20,
        "progress": tick.values["goal_progress_m_per_s"],
        "tracking": tracking * valid,
        "tracking_counts": valid,
        "target_vx": tick.target[:, 0],
        "exec_vx": tick.executed[:, 0],
        "true_vx": tick.diagnostic_aux[:, 12],
        "target_wz": tick.target[:, 2].abs(),
        "exec_wz": tick.executed[:, 2].abs(),
        "true_wz": tick.diagnostic_aux[:, 14].abs(),
        "success": tick.values["success_rate"],
        "failure": tick.values["failure_rate"],
        "timeout": tick.values["timeout_rate"],
    }
    for name, source in sources.items():
        _scatter_flat(buffers[name], flat_bin, source)


def _scatter_segment(buffer, index, valid, source) -> None:
    buffer.scatter_add_(0, index, source.float() * valid.float())


def _accumulate_segments(runtime, rollout, tick) -> None:
    live_segment = tick.diagnostic_aux[:, p2_contract.CURRENT_SEGMENT_INDEX]
    raw_segment = _terminal_safe_segment(
        live_segment, tick.terminal_segment, tick.transition_done
    )
    row_index = p2_contract.track_segment_metric_indices(
        raw_segment, runtime.segment_labels
    )
    valid = row_index >= 0
    rollout.segment_diagnostic_count += valid.float().sum()
    index = row_index.clamp(0, len(runtime.segment_metric_labels) - 1)
    rows = rollout.segment_rows
    sources = {
        "counts": torch.ones_like(tick.target[:, 0]),
        "progress": tick.values["goal_progress_m_per_s"],
        "target_vx": tick.target[:, 0],
        "exec_vx": tick.executed[:, 0],
        "true_vx": tick.diagnostic_aux[:, 12],
        "target_wz": tick.target[:, 2].abs(),
        "stop": tick.values["zero_command_rate"],
        "creep": tick.values["creep_command_rate"],
        "spin": tick.values["target_pure_yaw"],
        "success": tick.terminal_reason == 1,
        "failure": tick.terminal_reason == 2,
        "timeout": tick.terminal_reason == 3,
        "reason4": tick.terminal_reason == 4,
    }
    for name, source in sources.items():
        _scatter_segment(rows[name], index, valid, source)
    runtime.hooks.record_current_outcomes(
        rollout.variant, valid, tick.terminal_reason, index
    )
    tick.segment_valid = valid
    tick.valid_row_index = index


def _accumulate_algorithm_outputs(runtime, rollout) -> None:
    device = runtime.agent.device
    algorithm = runtime.algorithm
    for component, value in algorithm.last_tick_penalties.items():
        _add_sum(rollout.diagnostic_sums, f"reward_{component}", value, device)
    for name, value in algorithm.last_tick_diagnostics.items():
        if name in rollout.cumulative_counter_names:
            rollout.diagnostic_maxima[name] = torch.maximum(
                rollout.diagnostic_maxima.get(name, torch.zeros((), device=device)),
                value.max(),
            )
            continue
        _add_sum(rollout.diagnostic_sums, name, value, device)
        if name == "body_collision_force":
            maximum_name = f"{name}_max"
            rollout.diagnostic_maxima[maximum_name] = torch.maximum(
                rollout.diagnostic_maxima.get(
                    maximum_name, torch.zeros((), device=device)
                ),
                value.max(),
            )


def _record_collision_and_risk(runtime, rollout, tick, tick_index: int) -> None:
    algorithm = runtime.algorithm
    runtime.collision_traces.observe(
        iteration=algorithm.current_iteration,
        tick=tick_index,
        onset=algorithm.last_tick_diagnostics["body_collision_onset"],
        target_vx=tick.target[:, 0],
        target_vy=tick.target[:, 1],
        target_wz=tick.target[:, 2],
        exec_vx=tick.executed[:, 0],
        exec_vy=tick.executed[:, 1],
        exec_wz=tick.executed[:, 2],
        measured_vx=tick.diagnostic_aux[:, 6],
        measured_vy=tick.diagnostic_aux[:, 7],
        measured_wz=tick.diagnostic_aux[:, 8],
        true_vx=tick.diagnostic_aux[:, 12],
        true_vy=tick.diagnostic_aux[:, 13],
        true_wz=tick.diagnostic_aux[:, 14],
        collision_force=tick.diagnostic_aux[:, p2_contract.BODY_COLLISION_FORCE_INDEX],
        progress=tick.start_goal - tick.end_goal,
        segment=tick.diagnostic_aux[:, p2_contract.CURRENT_SEGMENT_INDEX],
        column=tick.diagnostic_aux[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX],
        reason=tick.terminal_reason,
        risk=algorithm.last_tick_diagnostics["predictive_collision_risk"],
    )
    gait = algorithm.last_tick_penalties["gait_symmetry"].reshape(-1)
    risk_sources = {
        "gait": gait,
        "predictive_clearance": algorithm.last_tick_diagnostics[
            "predictive_collision_clearance_m"
        ].reshape(-1),
        "predictive_risk": algorithm.last_tick_diagnostics[
            "predictive_collision_risk"
        ].reshape(-1),
        "predictive_penalty": algorithm.last_tick_penalties[
            "predictive_collision_risk"
        ].reshape(-1),
    }
    teacher_risk = torch.stack(
        tuple(
            algorithm.last_tick_diagnostics[f"teacher_risk_{name}"].reshape(-1)
            for name in ("left", "center", "right")
        ),
        dim=1,
    ).amax(dim=1)
    risk_sources["teacher_risk"] = teacher_risk
    risk_sources["teacher_high_risk"] = teacher_risk >= 0.5
    _scatter_flat(rollout.command_bins["gait"], tick.command_flat_bin, gait)
    for name, source in risk_sources.items():
        _scatter_segment(
            rollout.segment_rows[name],
            tick.valid_row_index,
            tick.segment_valid,
            source,
        )


def _finish_nav_tick(runtime, rollout, tick, tick_index: int) -> bool:
    _resolve_tick_values(runtime, tick)
    _accumulate_base_diagnostics(runtime, rollout, tick, tick_index)
    _accumulate_command_bins(runtime, rollout, tick)
    _accumulate_segments(runtime, rollout, tick)
    finish_wire = runtime.hooks.terminal_safe_critic_wire(
        runtime.critic_wire, tick.terminal_extra, tick.transition_done
    )
    full = runtime.algorithm.finish_tick(
        runtime.obs,
        finish_wire,
        frame_safety_reward=tick.frame_safety_reward,
        start_goal_distance=tick.start_goal,
        end_goal_distance=tick.end_goal,
        terminal_reason=tick.terminal_reason,
        duration_frames=tick.duration,
        hard_terminated=tick.hard,
        timeout=tick.timeout,
        unattributed_boundary=tick.unattributed_boundary,
        terminal_safe_aux=tick.diagnostic_aux,
        terminal_safe_exec_cmd=tick.executed,
        path_length_m=tick.path_length_m,
    )
    _accumulate_algorithm_outputs(runtime, rollout)
    runtime.hooks.after_tick(
        SimpleNamespace(runtime=runtime, rollout=rollout, tick=tick)
    )
    _record_collision_and_risk(runtime, rollout, tick, tick_index)
    rollout.diagnostic_count += runtime.agent.num_envs
    return bool(full)


def _run_rollout(runtime):
    if hasattr(runtime.algorithm, "begin_rollout"):
        runtime.algorithm.begin_rollout()
    rollout = _new_rollout(runtime)
    for tick_index in range(runtime.nav_rollout_ticks):
        tick = _new_tick(runtime)
        _run_control_frames(runtime, rollout, tick)
        full = _finish_nav_tick(runtime, rollout, tick, tick_index)
        if tick_index < runtime.nav_rollout_ticks - 1 and full:
            raise RuntimeError("P2 rollout filled before the 32-tick boundary")
        if tick_index == runtime.nav_rollout_ticks - 1 and not full:
            raise RuntimeError("P2 rollout did not fill at the 32-tick boundary")
    return rollout


def _advance_and_report(runtime, rollout, metrics) -> None:
    runtime.hooks.after_update(
        runtime.algorithm, runtime.agent, runtime.nav_rollout_ticks
    )
    curriculum_snapshot = runtime.algorithm.curriculum_probe.state_dict()
    (
        now,
        runtime.variant_training_seconds,
        runtime.unfreeze_save_done,
        runtime.next_save_clock,
        runtime.retry_save_at,
    ) = _advance_runtime_clocks_and_saves(
        agent=runtime.agent,
        algorithm=runtime.algorithm,
        logger=runtime.logger,
        hooks=runtime.hooks,
        rollout_started=rollout.rollout_started,
        session_started=runtime.session_started,
        resumed_seconds=runtime.resumed_seconds,
        resumed_clock_seconds=runtime.resumed_clock_seconds,
        variant_training_seconds=runtime.variant_training_seconds,
        unfreeze_save_done=runtime.unfreeze_save_done,
        schedule_boundaries=runtime.schedule_boundaries,
        saved_schedule_boundaries=runtime.saved_schedule_boundaries,
        next_save_clock=runtime.next_save_clock,
        save_interval_s=runtime.save_interval_s,
        retry_save_at=runtime.retry_save_at,
    )
    state = {
        "runtime": runtime,
        "rollout": rollout,
        "now": now,
        "curriculum_snapshot": curriculum_snapshot,
    }
    build_nav_rollout_metrics(metrics, state)
    runtime.last_log = _maybe_report_monitor(
        metrics,
        {
            "now": now,
            "last_log": runtime.last_log,
            "algorithm": runtime.algorithm,
            "logger": runtime.logger,
            "monitor": runtime.monitor,
            "monitor_health": runtime.monitor_health,
            "lifecycle_success": runtime.lifecycle_success,
            "lifecycle_failures": runtime.lifecycle_failures,
        },
    )


def _training_loop(runtime) -> None:
    target_seconds = float(runtime.target.target_seconds)
    while runtime.algorithm.session_effective_seconds < target_seconds:
        rollout = _run_rollout(runtime)
        _advance_and_report(runtime, rollout, runtime.algorithm.update())
    _final_save(
        runtime.agent,
        runtime.logger,
        runtime.hooks.final_save_reason(runtime.target),
    )


def run_nav_ppo_workflow(
    envs,
    agents,
    logger=None,
    monitor=None,
    *args,
    spec: WorkflowSpec,
    hooks: NavWorkflowHooks,
    **kwargs,
):
    """Run one navigation PPO workflow through an explicit wrapper contract."""
    del args, kwargs
    runtime = _prepare_runtime(envs, agents, logger, monitor, spec, hooks)
    try:
        _training_loop(runtime)
    except (SystemExit, KeyboardInterrupt) as exc:
        _final_save(runtime.agent, runtime.logger, type(exc).__name__)
        raise
    finally:
        if runtime.previous_sigterm is not None:
            signal.signal(signal.SIGTERM, runtime.previous_sigterm)
