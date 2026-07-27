#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Wall-clock workflow for constrained recurrent visual PPO schedules.

``visual_anchor_anneal_v2`` uses the Anchor R2 phase labels; command
generalization uses ``commandbase`` -> ``commandblend`` -> ``commandfull``.
Both use a session clock, not the cumulative-training-hours record.
"""

from __future__ import annotations

import os
import signal
import threading
import time

from agent_ppo.conf.conf import Config
from agent_ppo.workflow.train_workflow import (
    _initialize_training_state,
    report_monitor_data,
    run_episodes_,
)
from agent_ppo.workflow.save_schedule import (
    SAVE_DEDUP_DELTA_S,
    should_deduplicate_save,
)

# Phase -> monitor int code. Values are stable within each schedule vocabulary.
_PHASE_CODES = {
    "anchorcritic": 0,
    "anchoractor": 1,
    "anchoranneal": 2,
    "anchorfinal": 3,
    # Legacy codes retained for resume compatibility of historical bundles.
    "rlcritic": 0,
    "rlactor": 1,
    "rlfull": 2,
    "commandbase": 0,
    "commandblend": 1,
    "commandfull": 2,
}


def _save_final_checkpoint(agent, logger, *, reason: str) -> bool:
    """Best-effort final save for a graceful workflow exit.

    This helper intentionally does not catch a failed normal save in the main
    loop: checkpoint write failures remain hard failures. It is only used from
    the graceful-exit path after ``SystemExit``/``KeyboardInterrupt`` where a
    best-effort save must not hide the platform's original shutdown signal.
    """
    if (
        not getattr(agent, "is_visual_ppo", False)
        or not getattr(agent, "_visual_ppo_training_started", False)
        or getattr(agent, "_visual_ppo_final_save_done", False)
    ):
        return False
    try:
        logger.warning(
            "[VisualPPO] graceful workflow exit; saving final checkpoint "
            f"(reason={reason})"
        )
        agent.save_model()
    except Exception as exc:  # Preserve the original shutdown exception.
        logger.error(
            "[VisualPPO] final checkpoint save failed during graceful exit: "
            f"{exc}"
        )
        return False
    agent._visual_ppo_final_save_done = True
    agent._visual_ppo_final_save_reason = reason
    return True


def _install_sigterm_checkpoint_handler(logger):
    """Convert the default SIGTERM action into the existing graceful path.

    The training platform normally stops a completed wall-clock task with a
    signal rather than an iteration-cap return.  Only replace Python's default
    SIGTERM action; a framework-installed handler remains authoritative.  The
    handler raises ``SystemExit`` so ``workflow`` can perform exactly one
    best-effort final save before restoring the original disposition.
    """
    if (
        not hasattr(signal, "SIGTERM")
        or threading.current_thread() is not threading.main_thread()
    ):
        return None
    previous = signal.getsignal(signal.SIGTERM)
    if previous is not signal.SIG_DFL:
        logger.info(
            "[VisualPPO] preserving existing SIGTERM handler; "
            "final checkpoint depends on its graceful-exit path"
        )
        return None

    def _handle_sigterm(signum, _frame):
        raise SystemExit(f"SIGTERM({signum})")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    logger.info("[VisualPPO] installed default SIGTERM graceful-save handler")
    return previous


def _restore_sigterm_handler(previous) -> None:
    if previous is not None:
        signal.signal(signal.SIGTERM, previous)


def _workflow_impl(envs, agents, logger=None, monitor=None, *args, **kwargs):
    del args, kwargs
    agent = agents[0]
    env = envs[0]
    if not getattr(agent, "is_visual_ppo", False):
        raise RuntimeError("visual_ppo_workflow requires Agent.is_visual_ppo")

    (
        storage,
        obs,
        critic_obs,
        ep_infos,
        rewbuffer,
        lenbuffer,
        cur_reward_sum,
        cur_episode_length,
        reward_keys,
        usr_conf,
    ) = _initialize_training_state(env, agent, logger)
    agent._visual_ppo_training_started = True

    algorithm = agent.algorithm
    stage_conf = usr_conf.get(Config.CURRENT.name, {})
    save_interval_s = 60.0 * float(
        stage_conf.get("save_interval_minutes", 10.0)
    )
    # Resume first-save delay (§5.8 N3): resume deliberately re-runs a 2-minute
    # first save even if it lands within 10 min of the last boundary save. This
    # is a safety save, not a dedup defect.
    resume_first_save_s = 60.0 * float(
        stage_conf.get("resume_first_save_minutes", save_interval_s / 60.0)
    )
    log_interval = max(1, int(stage_conf.get("log_interval", 10)))
    max_iterations = int(stage_conf.get("max_iterations", 50000))
    # Phase milestone saves (minutes -> seconds). Command generalization uses
    # its own boundary key; Anchor R2 keeps the historical anchor key.
    milestone_key = (
        "command_checkpoint_minutes"
        if algorithm.schedule_mode == "visual_command_generalization_v1"
        else "anchor_checkpoint_minutes"
    )
    anchor_checkpoint_s = sorted(
        60.0 * float(value)
        for value in stage_conf.get(milestone_key, [])
    )

    command_schedule = (
        algorithm.schedule_mode == "visual_command_generalization_v1"
    )
    session_clock_name = "command_session_h" if command_schedule else "anchor_h"
    resume_session_h = float(algorithm.anchor_session_elapsed_hours)
    resume_cumulative_h = float(algorithm.elapsed_training_hours)
    session_start = time.monotonic()
    resume_loaded = bool(getattr(algorithm, "resume_loaded", False))
    # Periodic save deadline. On a fresh start the first save is one full
    # cadence out; on resume it is resume_first_save_s out (§5.8 N3).
    first_save_delay_s = resume_first_save_s if resume_loaded else save_interval_s
    next_periodic_save = session_start + first_save_delay_s
    last_save_time = session_start
    last_monitor_time = 0.0
    diagnostic_checkpoint_saved = False
    # Anchor milestone pointer: skip milestones already passed at resume.
    next_anchor_index = 0
    while (
        next_anchor_index < len(anchor_checkpoint_s)
        and anchor_checkpoint_s[next_anchor_index] <= resume_session_h * 3600.0
    ):
        next_anchor_index += 1
    iteration = int(algorithm.current_iteration)
    last_obs = obs.clone()
    last_critic_obs = critic_obs.clone()

    logger.info(
        "[VisualPPO] start command generalization schedule: "
        if command_schedule else "[VisualPPO] start anchor r2 schedule: "
    )
    logger.info(
        f"run={algorithm.run_name}, schedule={algorithm.schedule_mode}, "
        f"resume_iter={iteration}, resume_{session_clock_name}="
        f"{resume_session_h:.3f}, "
        f"save_interval_min={save_interval_s / 60.0:.1f}, "
        f"first_save_delay_min={first_save_delay_s / 60.0:.1f}, "
        f"checkpoint_minutes={[round(s / 60.0, 1) for s in anchor_checkpoint_s]}, "
        f"tbptt={algorithm.sequence_length}, "
        "command_runtime_owner=environment_worker"
    )
    env_conf = usr_conf.get("env", {})
    terrain_conf = usr_conf.get("terrain", {})
    camera_conf = usr_conf.get("camera", {}).get("depth_camera", {})
    command_conf = usr_conf.get("commands", {})
    custom_parameters = usr_conf.get("custom_parameters", {})
    standard_terrain = terrain_conf.get("standard", {})
    terrain_proportions = {
        name: values.get("proportion")
        for name, values in standard_terrain.items()
        if isinstance(values, dict) and "proportion" in values
    }
    logger.info(
        "[VisualPPO] runtime contract: "
        f"config_path={getattr(agent, 'usr_conf_file', 'unknown')}, "
        f"continuous_training={bool(custom_parameters.get('continuous_training', False))}, "
        "iteration_semantics=completed_outer_iterations_v1, "
        "outer_iteration=one_rollout_plus_one_optimizer_update, "
        f"inner_steps_per_outer={agent.num_steps_per_env}, "
        f"num_envs={agent.num_envs}, "
        f"task_name={usr_conf.get('env_conf', {}).get('task_name')}, "
        f"policy_entry={usr_conf.get('env_conf', {}).get('policy_entry')}, "
        f"curriculum={terrain_conf.get('curriculum')}, "
        f"max_init_terrain_level={terrain_conf.get('max_init_terrain_level')}, "
        f"terrain_proportions={terrain_proportions}, "
        f"native_resampling_time={command_conf.get('resampling_time')}, "
        f"depth_offset_pos={camera_conf.get('offset_pos')}, "
        f"depth_offset_rot={camera_conf.get('offset_rot')}, "
        f"depth_augmentation={camera_conf.get('augmentation', {}).get('enabled')}, "
        "command_runtime_owner=worker_observation_bridge_v1, "
        "command_effective_metrics_source=worker_log"
    )

    while iteration < max_iterations:
        loop_start = time.monotonic()
        # The active schedule uses its own session clock; elapsed_training_hours
        # only records cumulative training time.
        session_h = resume_session_h + (loop_start - session_start) / 3600.0
        agent.training_elapsed_h = session_h
        algorithm.anchor_session_elapsed_hours = session_h
        algorithm.elapsed_training_hours = (
            resume_cumulative_h + (loop_start - session_start) / 3600.0
        )

        last_obs, last_critic_obs, storage_stats = run_episodes_(
            env,
            agent,
            storage,
            logger,
            last_obs,
            last_critic_obs,
            iteration,
            ep_infos,
            cur_reward_sum,
            cur_episode_length,
            rewbuffer,
            lenbuffer,
        )
        metrics = agent.learn(list_sample_data=None)
        storage.clear()
        iteration = int(algorithm.current_iteration)

        # The first update is diagnostic only. Quality thresholds are
        # warning-only in Anchor R2 and must not abort iteration 1.
        if iteration == 1 and not resume_loaded:
            logger.info(
                "[VisualPPO] first-update diagnostic: "
                f"applied_updates={metrics.get('applied_updates')}, "
                f"anchor_action_mse={metrics.get('anchor_action_mse')}"
            )

        now = time.monotonic()
        if iteration == 1 or iteration % log_interval == 0:
            logger.info(
                "[VisualPPO] "
                f"iter={iteration}, {session_clock_name}={session_h:.3f}, "
                f"phase={algorithm.current_phase}, "
                f"policy={metrics.get('policy_loss', 0.0):.5f}, "
                f"value={metrics.get('value_loss', 0.0):.5f}, "
                f"action_anchor_loss={metrics.get('action_anchor_loss', 0.0):.5f}, "
                f"latent_anchor_loss={metrics.get('latent_anchor_loss', 0.0):.5f}, "
                f"action_anchor_w={algorithm.action_anchor_weight:.3f}, "
                f"latent_anchor_w={algorithm.latent_anchor_weight_current:.3f}, "
                f"anchor_mse={metrics.get('anchor_action_mse', 0.0):.5f}, "
                f"hard_term={metrics.get('hard_termination_rate', 0.0):.4f}, "
                f"frozen={algorithm.anchor_schedule_frozen}, "
                f"cost_s={now - loop_start:.2f}"
            )
            command_metrics = getattr(agent, "_last_command_metrics", {})
            if command_metrics:
                logger.info(
                    "[VisualPPO] command observation telemetry: "
                    "runtime_owner=worker_observation_bridge_v1, "
                    f"anchor_weight_mean={command_metrics.get('anchor_weight_mean')}, "
                    f"min={command_metrics.get('command_observation_min')}, "
                    f"max={command_metrics.get('command_observation_max')}, "
                    f"mean={command_metrics.get('command_observation_mean')}, "
                    "effective_source_target_metrics=worker_log_only"
                )
            zero_metrics = getattr(agent, "_last_zero_command_telemetry", {})
            if zero_metrics:
                logger.info(
                    "[VisualPPO] zero-command telemetry: "
                    f"source={zero_metrics.get('zero_telemetry_source')}, "
                    f"samples={zero_metrics.get('zero_telemetry_window_samples', 0)}, "
                    f"action_delta_mean={zero_metrics.get('zero_action_delta_mean')}, "
                    f"action_delta_p95={zero_metrics.get('zero_action_delta_p95')}, "
                    "action_second_delta_mean="
                    f"{zero_metrics.get('zero_action_second_delta_mean')}, "
                    "action_second_delta_p95="
                    f"{zero_metrics.get('zero_action_second_delta_p95')}, "
                    f"root_v_xy={zero_metrics.get('zero_root_v_xy')}, "
                    f"root_omega_xy={zero_metrics.get('zero_root_omega_xy')}, "
                    f"foot_slide={zero_metrics.get('zero_foot_slide')}, "
                    f"stability_penalty={zero_metrics.get('zero_stability_penalty')}"
                )

        if monitor is not None and now - last_monitor_time >= 60.0:
            report_monitor_data(
                ep_infos,
                reward_keys,
                agent,
                monitor,
                iteration,
                storage_stats,
            )
            phase_code = _PHASE_CODES.get(algorithm.current_phase, 0)
            monitor_metrics = {
                **metrics,
                "visual_ppo_phase": phase_code,
            }
            for key in (
                "anchor_weight_mean",
            ):
                value = getattr(agent, "_last_command_metrics", {}).get(key)
                if isinstance(value, (int, float)):
                    monitor_metrics[key] = float(value)
            for key in (
                "zero_telemetry_window_samples",
                "zero_action_delta_mean",
                "zero_action_delta_p95",
                "zero_action_second_delta_mean",
                "zero_action_second_delta_p95",
            ):
                value = getattr(agent, "_last_zero_command_telemetry", {}).get(key)
                if isinstance(value, (int, float)):
                    monitor_metrics[key] = float(value)
            monitor.put_data({os.getpid(): monitor_metrics})
            last_monitor_time = now
        ep_infos.clear()

        # Diagnostic save on warning-only safety trigger (§5.3/§5.8). Anchor R2
        # never pauses actor updates; this only persists an extra checkpoint for
        # inspection and does not end the task.
        if (
            getattr(algorithm, "last_diagnostic_save_requested", False)
            and not diagnostic_checkpoint_saved
        ):
            logger.warning(
                "[VisualPPO] safety diagnostic triggered; saving diagnostic "
                f"checkpoint (warning-only, task continues). "
                f"hard_term={metrics.get('hard_termination_rate', 0.0):.4f}, "
                f"anchor_mse={metrics.get('anchor_action_mse', 0.0):.5f}"
            )
            agent.save_model()
            last_save_time = now
            diagnostic_checkpoint_saved = True
            algorithm.last_diagnostic_save_requested = False
        elif not getattr(algorithm, "last_diagnostic_save_requested", False):
            diagnostic_checkpoint_saved = False

        # --- Save trigger resolution (§5.8 N3/N4) ---
        # Two trigger types: periodic (cadence) and anchor milestone (boundary).
        # When both fire in the same loop, only one save happens (the first
        # branch to act). When a trigger fires within SAVE_DEDUP_DELTA_S of
        # the last save, it is skipped and the next periodic deadline shifts to
        # last_save_time + cadence. This dedup is process-local; resume re-runs
        # the 2-minute first save by design (§5.8 N3).
        periodic_due = now >= next_periodic_save
        milestone_due = (
            next_anchor_index < len(anchor_checkpoint_s)
            and session_h * 3600.0 >= anchor_checkpoint_s[next_anchor_index]
        )
        if periodic_due or milestone_due:
            delta_since_save = now - last_save_time
            # session_start sentinel: no save yet this session, never dedup.
            first_save_this_session = last_save_time == session_start
            if should_deduplicate_save(
                delta_since_save,
                first_save_this_session=first_save_this_session,
            ):
                # Collapse: skip this save, but still advance the milestone
                # pointer and reschedule periodic from the last save so we do
                # not re-trigger every loop.
                logger.info(
                    "[VisualPPO] save trigger within dedup window skipped "
                    f"(delta={delta_since_save:.1f}s<={SAVE_DEDUP_DELTA_S}s); "
                    "next periodic rescheduled"
                )
                if milestone_due:
                    next_anchor_index += 1
                next_periodic_save = last_save_time + save_interval_s
            else:
                save_reasons = []
                if periodic_due:
                    save_reasons.append("periodic")
                if milestone_due:
                    save_reasons.append(
                        f"milestone_"
                        f"{round(anchor_checkpoint_s[next_anchor_index] / 60.0, 1)}min"
                    )
                agent.save_model()
                last_save_time = now
                # Reschedule next periodic deadline from the actual save time.
                next_periodic_save = now + save_interval_s
                if milestone_due:
                    next_anchor_index += 1
                logger.info(
                    f"[VisualPPO] checkpoint saved ({'+'.join(save_reasons)})"
                )

    logger.warning(
        f"[VisualPPO] reached max_iterations={max_iterations}; final save"
    )
    agent.save_model()
    agent._visual_ppo_final_save_done = True
    agent._visual_ppo_final_save_reason = "iteration_cap"


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """Run visual PPO and always release the simulator on failure or exit."""
    agent = agents[0]
    agent._visual_ppo_training_started = False
    agent._visual_ppo_final_save_done = False
    agent._visual_ppo_final_save_reason = None
    previous_sigterm_handler = _install_sigterm_checkpoint_handler(logger)
    try:
        return _workflow_impl(
            envs,
            agents,
            logger=logger,
            monitor=monitor,
            *args,
            **kwargs,
        )
    except (KeyboardInterrupt, SystemExit):
        _save_final_checkpoint(agent, logger, reason="graceful_platform_exit")
        raise
    finally:
        try:
            envs[0].close()
        finally:
            _restore_sigterm_handler(previous_sigterm_handler)
