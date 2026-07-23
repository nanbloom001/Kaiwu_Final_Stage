#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Workflow for standard_ref_distill behavior cloning."""

from __future__ import annotations

import os
import time
from collections import deque

import torch

from agent_ppo.conf.conf import Config


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """Run flat-standard-teacher -> ActorCriticEncoder student distillation."""
    agent = agents[0]
    env = envs[0]

    assert getattr(agent, "is_behavior_distill", False), (
        "behavior_distill_workflow called but agent.is_behavior_distill is False"
    )

    stage = agent.stage
    algorithm = agent.algorithm
    usr_conf, _usr_conf_file, _is_eval, _stage = Config.load_conf(logger)
    section = stage.name
    distill_conf = usr_conf.get(section, {}) if isinstance(usr_conf, dict) else {}

    max_iterations = int(distill_conf.get("max_iterations", stage.max_iterations))
    log_interval = int(distill_conf.get("log_interval", stage.log_interval))
    save_interval = int(distill_conf.get("save_interval", stage.model_save_interval))
    num_steps_per_env = int(distill_conf.get("num_steps_per_env", stage.num_steps_per_env))
    student_drive = bool(distill_conf.get("student_drive", False))

    override_lr = distill_conf.get("learning_rate")
    if override_lr is not None:
        override_lr = float(override_lr)
        for pg in algorithm.optimizer.param_groups:
            pg["lr"] = override_lr
        algorithm.learning_rate = override_lr

    logger.info(
        f"[BehaviorDistill] Start: max_iterations={max_iterations}, "
        f"num_steps_per_env={num_steps_per_env}, save_interval={save_interval}, "
        f"student_drive={student_drive}, lr={algorithm.optimizer.param_groups[0]['lr']:.2e}"
    )

    if not getattr(algorithm, "teacher_loaded", False):
        preload_dir = os.environ.get("KAIWU_MODEL_CKPT_DIR", "/data/pre_model/ckpt")
        logger.info(
            "[BehaviorDistill] teacher not loaded yet; trying the explicit "
            f"10288 preload from {preload_dir}"
        )
        try:
            agent.load_model(path=preload_dir, id="10288")
        except Exception as exc:
            logger.warning(
                "[BehaviorDistill] 10288 preload fallback did not complete; "
                "continuing because this minimal run delegates pretrained-model "
                f"selection to the platform operator. Error: {exc}"
            )
    logger.warning(
        "[BehaviorDistill] teacher status after preload: "
        f"loaded={algorithm.teacher_loaded}. Verify the task selected the original "
        "flat Standard 10288 checkpoint."
    )

    data = env.reset(usr_conf)
    if data is None:
        raise RuntimeError("[BehaviorDistill] env.reset returned None")
    obs, _critic_obs = data
    obs = torch.as_tensor(obs, device=agent.device)

    if student_drive:
        agent.model.eval()
    else:
        agent.model.train()

    metric_windows = {
        "loss": deque(maxlen=log_interval),
        "action_mse": deque(maxlen=log_interval),
        "action_l2": deque(maxlen=log_interval),
        "action_cos": deque(maxlen=log_interval),
        "teacher_abs": deque(maxlen=log_interval),
        "student_abs": deque(maxlen=log_interval),
        "grad_norm": deque(maxlen=log_interval),
    }

    loop_start = time.time()
    for iteration in range(max_iterations):
        iter_id = iteration + 1
        algorithm.current_iteration = iter_id
        iter_start = time.time()

        for _step in range(num_steps_per_env):
            if student_drive:
                actions = algorithm.act_student(obs)
                metrics = algorithm.compute_metrics(obs)
            else:
                metrics = algorithm.update(obs)
                actions = algorithm.act_teacher(obs)

            for key, window in metric_windows.items():
                if key in metrics:
                    window.append(metrics[key])

            step_data = env.step(torch.clip(actions, -6.0, 6.0).to(agent.device))
            if step_data is None:
                raise RuntimeError("[BehaviorDistill] env.step returned None")
            _frame_no, next_obs, _rewards, _terminated, _truncated, (_infos, _privileged_obs) = step_data
            obs = torch.as_tensor(next_obs, device=agent.device)

        # The supervised update above does not pass through BaseAgent.learn().
        # Keep one no-op callback per outer iteration so Kaiwu can advance
        # train_global_step and generate a fresh model-pool checkpoint id.
        if not student_drive:
            agent.learn(list_sample_data=None)

        if iter_id % log_interval == 0 or iteration == 0:
            def _avg(buf):
                return sum(buf) / max(1, len(buf))

            log_data = {key: _avg(window) for key, window in metric_windows.items()}
            logger.info(
                f"[BehaviorDistill] iter={iter_id}/{max_iterations} "
                f"mse={log_data['action_mse']:.6f} "
                f"l2={log_data['action_l2']:.4f} "
                f"cos={log_data['action_cos']:.4f} "
                f"student_abs={log_data['student_abs']:.4f} "
                f"teacher_abs={log_data['teacher_abs']:.4f} "
                f"grad={log_data['grad_norm']:.3f} "
                f"steps={algorithm.total_steps} "
                f"iter_time={time.time() - iter_start:.2f}s"
            )
            if monitor is not None:
                try:
                    payload = {"iteration": iter_id, "total_steps": algorithm.total_steps}
                    payload.update(log_data)
                    monitor.put_data({os.getpid(): payload})
                except Exception as exc:
                    logger.warning(f"[BehaviorDistill] monitor.put_data failed: {exc}")

        # Match the proven Standard workflow: call save_model without a business
        # id and let the platform inject its current, monotonically increasing id.
        if not student_drive and iter_id % save_interval == 0:
            agent.save_model()

    if not student_drive and max_iterations % save_interval != 0:
        agent.save_model()
    logger.info(
        f"[BehaviorDistill] finished in {time.time() - loop_start:.1f}s, "
        f"total steps={algorithm.total_steps}"
    )
    env.close()
