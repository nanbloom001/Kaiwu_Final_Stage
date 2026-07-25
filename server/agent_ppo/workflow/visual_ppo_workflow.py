#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Anchor R2 four-hour workflow for constrained recurrent visual PPO.

Schedule ``visual_anchor_anneal_v2``: anchorcritic -> anchoractor ->
anchoranneal -> anchorfinal, driven by the anchor session clock (not the
cumulative training-hours counter). See
shared/分析记录/2026-07-25_StandardAnchorR2四小时实施计划.md §4/§5.8.
"""

from __future__ import annotations

import os
import time

from agent_ppo.conf.conf import Config
from agent_ppo.workflow.train_workflow import (
    _initialize_training_state,
    report_monitor_data,
    run_episodes_,
)

# Phase -> monitor int code. Anchor R2 phase vocabulary.
_PHASE_CODES = {
    "anchorcritic": 0,
    "anchoractor": 1,
    "anchoranneal": 2,
    "anchorfinal": 3,
    # Legacy codes retained for resume compatibility of historical bundles.
    "rlcritic": 0,
    "rlactor": 1,
    "rlfull": 2,
}

# Two save triggers within delta seconds collapse into one save call (§5.8 N4).
_SAVE_DEDUP_DELTA_S = 60.0


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
    # Anchor milestone saves (minutes -> seconds). First three are phase
    # boundaries; the last (230 min) is a pre-reclamation safety save.
    anchor_checkpoint_s = sorted(
        60.0 * float(value)
        for value in stage_conf.get("anchor_checkpoint_minutes", [])
    )

    algorithm = agent.algorithm
    resume_anchor_h = float(algorithm.anchor_session_elapsed_hours)
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
        and anchor_checkpoint_s[next_anchor_index] <= resume_anchor_h * 3600.0
    ):
        next_anchor_index += 1
    iteration = int(algorithm.current_iteration)
    last_obs = obs.clone()
    last_critic_obs = critic_obs.clone()

    logger.info(
        "[VisualPPO] start anchor r2 schedule: "
        f"run={algorithm.run_name}, schedule={algorithm.schedule_mode}, "
        f"resume_iter={iteration}, resume_anchor_h={resume_anchor_h:.3f}, "
        f"save_interval_min={save_interval_s / 60.0:.1f}, "
        f"first_save_delay_min={first_save_delay_s / 60.0:.1f}, "
        f"anchor_checkpoints_min={[round(s / 60.0, 1) for s in anchor_checkpoint_s]}, "
        f"tbptt={algorithm.sequence_length}"
    )

    while iteration < max_iterations:
        loop_start = time.monotonic()
        # Anchor session clock drives Anchor R2 phase decisions (§4.3).
        # elapsed_training_hours only records cumulative training time.
        anchor_h = resume_anchor_h + (loop_start - session_start) / 3600.0
        agent.training_elapsed_h = anchor_h
        algorithm.anchor_session_elapsed_hours = anchor_h
        algorithm.elapsed_training_hours = anchor_h

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

        # First Critic-only update is the in-task smoke (§10.2). Skipped on
        # resume so a resumed run is not falsely failed.
        if iteration == 1 and not resume_loaded:
            if (
                metrics.get("applied_updates", 0.0) <= 0.0
                or metrics.get("anchor_action_mse", float("inf")) > 1.0e-6
            ):
                agent.save_model()
                raise RuntimeError(
                    "[VisualPPO] in-task smoke failed: "
                    f"applied_updates={metrics.get('applied_updates')}, "
                    f"anchor_action_mse={metrics.get('anchor_action_mse')}"
                )

        now = time.monotonic()
        if iteration == 1 or iteration % log_interval == 0:
            logger.info(
                "[VisualPPO] "
                f"iter={iteration}, anchor_h={anchor_h:.3f}, "
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
            monitor.put_data(
                {
                    os.getpid(): {
                        **metrics,
                        "visual_ppo_phase": phase_code,
                    }
                }
            )
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
        # branch to act). When a trigger fires within _SAVE_DEDUP_DELTA_S of
        # the last save, it is skipped and the next periodic deadline shifts to
        # last_save_time + cadence. This dedup is process-local; resume re-runs
        # the 2-minute first save by design (§5.8 N3).
        periodic_due = now >= next_periodic_save
        milestone_due = (
            next_anchor_index < len(anchor_checkpoint_s)
            and anchor_h * 3600.0 >= anchor_checkpoint_s[next_anchor_index]
        )
        if periodic_due or milestone_due:
            delta_since_save = now - last_save_time
            # session_start sentinel: no save yet this session, never dedup.
            first_save_this_session = last_save_time == session_start
            if (
                not first_save_this_session
                and delta_since_save <= _SAVE_DEDUP_DELTA_S
            ):
                # Collapse: skip this save, but still advance the milestone
                # pointer and reschedule periodic from the last save so we do
                # not re-trigger every loop.
                logger.info(
                    "[VisualPPO] save trigger within dedup window skipped "
                    f"(delta={delta_since_save:.1f}s<={_SAVE_DEDUP_DELTA_S}s); "
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


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """Run visual PPO and always release the simulator on failure or exit."""
    try:
        return _workflow_impl(
            envs,
            agents,
            logger=logger,
            monitor=monitor,
            *args,
            **kwargs,
        )
    finally:
        envs[0].close()
