#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Single-run three-hour workflow for constrained recurrent visual PPO."""

from __future__ import annotations

import os
import time

from agent_ppo.conf.conf import Config
from agent_ppo.workflow.train_workflow import (
    _initialize_training_state,
    report_monitor_data,
    run_episodes_,
)


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
    log_interval = max(1, int(stage_conf.get("log_interval", 10)))
    max_iterations = int(stage_conf.get("max_iterations", 50000))

    resume_elapsed_h = float(agent.algorithm.elapsed_training_hours)
    session_start = time.monotonic()
    last_save_time = session_start
    last_monitor_time = 0.0
    pause_checkpoint_saved = False
    iteration = int(agent.algorithm.current_iteration)
    last_obs = obs.clone()
    last_critic_obs = critic_obs.clone()

    logger.info(
        "[VisualPPO] start single-run schedule: "
        f"resume_iter={iteration}, resume_elapsed_h={resume_elapsed_h:.3f}, "
        f"save_interval_min={save_interval_s / 60.0:.1f}, "
        f"tbptt={agent.algorithm.sequence_length}"
    )

    while iteration < max_iterations:
        loop_start = time.monotonic()
        elapsed_h = resume_elapsed_h + (loop_start - session_start) / 3600.0
        agent.training_elapsed_h = elapsed_h
        agent.algorithm.elapsed_training_hours = elapsed_h

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
        iteration = int(agent.algorithm.current_iteration)

        # The first Critic-only update is also the in-task smoke. Since current
        # actor and frozen S0 are loaded from the same payload, deterministic
        # action means must initially agree to numerical precision.
        if iteration == 1 and not agent.algorithm.resume_loaded:
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
                f"iter={iteration}, elapsed_h={elapsed_h:.3f}, "
                f"phase={agent.algorithm.current_phase}, "
                f"policy={metrics.get('policy_loss', 0.0):.5f}, "
                f"value={metrics.get('value_loss', 0.0):.5f}, "
                f"action_anchor={metrics.get('action_anchor_loss', 0.0):.5f}, "
                f"latent_anchor={metrics.get('latent_anchor_loss', 0.0):.5f}, "
                f"anchor_mse={metrics.get('anchor_action_mse', 0.0):.5f}, "
                f"hard_term={metrics.get('hard_termination_rate', 0.0):.4f}, "
                f"paused={agent.algorithm.actor_updates_paused}, "
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
            monitor.put_data(
                {
                    os.getpid(): {
                        **metrics,
                        "visual_ppo_phase": {
                            "rlcritic": 0,
                            "rlactor": 1,
                            "rlfull": 2,
                        }[agent.algorithm.current_phase],
                    }
                }
            )
            last_monitor_time = now
        ep_infos.clear()

        if (
            agent.algorithm.actor_updates_paused
            and not pause_checkpoint_saved
        ):
            logger.warning(
                "[VisualPPO] Actor/LSTM updates paused; Critic continues. "
                f"reason={agent.algorithm.pause_reason}"
            )
            agent.save_model()
            pause_checkpoint_saved = True
        elif not agent.algorithm.actor_updates_paused:
            pause_checkpoint_saved = False

        if now - last_save_time >= save_interval_s:
            agent.save_model()
            last_save_time = now

    logger.warning(
        f"[VisualPPO] reached safety max_iterations={max_iterations}; final save"
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
