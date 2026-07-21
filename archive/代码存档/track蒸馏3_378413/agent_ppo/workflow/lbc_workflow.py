#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""LBC distillation workflow (DAgger + scheduled sampling + action distill).

lbc_loco:
    teacher = teacher_encoder(height_scan) + teacher_actor  (FROZEN)
    student = VisionEncoder (CNN + LSTM + head)            (TRAIN)
    loss = MSE(student_latent, teacher_latent)
         + action_loss_weight * MSE(student_action, teacher_action)  (方案B)

Env driver:
    act_teacher (teacher full chain)
    student_drive: DAgger - student drives env AND update trains on student
    closed-loop states. scheduled sampling (方案A): after teacher_warmup,
    p_student linearly anneals 0 -> 1 so student gradually adapts to its own
    closed-loop distribution (avoids the hard-switch score drop).

DAgger note: when student_drive, update() is called once (one vision_encoder
forward -> LSTM hidden advances one step, in sync with obs). The computed
student_latent is reused to drive env, avoiding double LSTM update.
"""

from __future__ import annotations

import math
import os
import random
import time
from collections import deque

import torch

from agent_ppo.conf.conf import Config


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """LBC main workflow."""
    agent = agents[0]
    env = envs[0]

    assert getattr(agent, "is_lbc", False), (
        "lbc_workflow.workflow called but agent.is_lbc is False; "
        "check Config.CURRENT and agent.__init__."
    )

    stage = agent.stage
    algorithm = agent.algorithm

    usr_conf, usr_conf_file, is_eval, _stage = Config.load_conf(logger)
    section = stage.name
    lbc_conf = usr_conf.get(section, {}) if isinstance(usr_conf, dict) else {}

    max_iterations = int(lbc_conf.get("max_iterations", stage.max_iterations))
    log_interval = int(lbc_conf.get("log_interval", stage.log_interval))
    save_interval = int(lbc_conf.get("save_interval", stage.model_save_interval))
    num_steps_per_env = int(lbc_conf.get("num_steps_per_env", stage.num_steps_per_env))

    student_drive_conf = bool(lbc_conf.get("student_drive", False))
    teacher_warmup = int(lbc_conf.get("teacher_warmup_iterations", 0))
    # p_student anneals 0->1 over the first anneal_fraction of training (post-warmup),
    # then holds at 1 (full student closed-loop) for the rest. Old behavior annealed
    # linearly across the whole run so p_student only reached 1 at the last iter (zero closed-loop).
    anneal_fraction = float(lbc_conf.get("p_student_anneal_fraction", 0.5))
    anneal_fraction = min(1.0, max(0.0, anneal_fraction))
    # action-level distillation weight (方案B); 0 = off (pure latent distill)
    action_loss_weight = float(lbc_conf.get("action_loss_weight", 0.0))
    algorithm.action_loss_weight = action_loss_weight

    override_lr = lbc_conf.get("learning_rate")
    if override_lr is not None:
        override_lr = float(override_lr)
        for pg in algorithm.optimizer.param_groups:
            pg["lr"] = override_lr
        if hasattr(algorithm, "learning_rate"):
            algorithm.learning_rate = override_lr
        logger.info(
            f"[LBC] toml override learning_rate: stage.lr={stage.lr} -> {override_lr}"
        )
    lr_min = float(lbc_conf.get("lr_min", getattr(stage, "lr_min", 1e-5)))

    # DAgger: both teacher-warmup and student-drive phases train -> always schedule LR.
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        algorithm.optimizer,
        T_max=max_iterations,
        eta_min=lr_min,
    )

    logger.info(
        f"[LBC-Loco] Start: "
        f"max_iterations={max_iterations}, log_interval={log_interval}, "
        f"save_interval={save_interval}, num_steps_per_env={num_steps_per_env}, "
        f"student_drive={student_drive_conf}, teacher_warmup={teacher_warmup}, "
        f"action_loss_weight={action_loss_weight}, "
        f"p_student_anneal_fraction={anneal_fraction}, "
        f"init_lr={algorithm.optimizer.param_groups[0]['lr']:.2e}, "
        f"lr_scheduler=cosine(eta_min={lr_min:.0e})"
    )

    data = env.reset(usr_conf)
    if data is None:
        raise RuntimeError("[LBC] env.reset returned None, check env configuration.")
    obs, _critic_obs = data
    obs = torch.as_tensor(obs).to(agent.device)

    if hasattr(algorithm.vision_encoder, "reset_hidden_state"):
        algorithm.vision_encoder.reset_hidden_state(batch_size=agent.num_envs, device=agent.device)

    algorithm.train_mode()

    metric_windows = {
        "grad_norm": deque(maxlen=log_interval),
        "mse_loss": deque(maxlen=log_interval),
        "action_loss": deque(maxlen=log_interval),
        "distance": deque(maxlen=log_interval),
        "cos_sim": deque(maxlen=log_interval),
        "student_std": deque(maxlen=log_interval),
        "teacher_std": deque(maxlen=log_interval),
        "student_drive_ratio": deque(maxlen=log_interval),
    }

    loop_start = time.time()
    last_save_iter = 0

    for iteration in range(max_iterations):
        algorithm.current_iteration = iteration
        ep_start = time.time()

        # scheduled sampling (方案A): warmup -> p_student 0; then linearly anneal to 1
        # over the first anneal_fraction of training; hold at 1 (full closed-loop) for the rest.
        if teacher_warmup > 0 and iteration < teacher_warmup:
            p_student = 0.0
        else:
            _anneal_span = max(1, int(anneal_fraction * (max_iterations - teacher_warmup)))
            _anneal_end = teacher_warmup + _anneal_span
            if iteration < _anneal_end:
                p_student = (iteration - teacher_warmup) / _anneal_span
            else:
                p_student = 1.0
        p_student = min(1.0, max(0.0, p_student))

        for _step in range(num_steps_per_env):
            student_drive = student_drive_conf and (random.random() < p_student)
            obs, step_metrics = _lbc_step(env, agent, algorithm, obs, student_drive=student_drive)
            for key, window in metric_windows.items():
                if key in step_metrics:
                    window.append(step_metrics[key])
            metric_windows["student_drive_ratio"].append(1.0 if student_drive else 0.0)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if (iteration + 1) % log_interval == 0 or iteration == 0:

            def _avg(buf):
                return sum(buf) / max(1, len(buf))

            cur_lr = algorithm.optimizer.param_groups[0]["lr"]
            dt = time.time() - ep_start

            mean_mse = _avg(metric_windows["mse_loss"])
            mean_action_loss = _avg(metric_windows["action_loss"])
            mean_distance = _avg(metric_windows["distance"])
            mean_cos = _avg(metric_windows["cos_sim"])
            mean_s_std = _avg(metric_windows["student_std"])
            mean_t_std = _avg(metric_windows["teacher_std"])
            mean_grad = _avg(metric_windows["grad_norm"])
            mean_sd_ratio = _avg(metric_windows["student_drive_ratio"])
            angle = math.degrees(math.acos(max(-1.0, min(1.0, mean_cos))))
            logger.info(
                f"[LBC-Loco] iter={iteration+1}/{max_iterations}  "
                f"angle={angle:.2f}deg  cos={mean_cos:.4f}  "
                f"mse={mean_mse:.5f}  act={mean_action_loss:.5f}  "
                f"distance={mean_distance:.4f}  "
                f"s_lat_bstd={mean_s_std:.4f}  t_lat_bstd={mean_t_std:.4f}  "
                f"grad={mean_grad:.3f}  lr={cur_lr:.2e}  "
                f"p_student={p_student:.2f}  sd_ratio={mean_sd_ratio:.2f}  "
                f"total_steps={algorithm.total_steps}  iter_time={dt:.2f}s"
            )
            if monitor is not None:
                try:
                    monitor.put_data(
                        {
                            os.getpid(): {
                                "iteration": iteration + 1,
                                "angle": angle,
                                "cos_sim": mean_cos,
                                "mse_loss": mean_mse,
                                "action_loss": mean_action_loss,
                                "distance": mean_distance,
                                "student_std": mean_s_std,
                                "teacher_std": mean_t_std,
                                "grad_norm": mean_grad,
                                "student_drive_ratio": mean_sd_ratio,
                                "p_student": p_student,
                                "total_steps": algorithm.total_steps,
                            }
                        }
                    )
                except Exception as e:
                    logger.warning(f"[LBC-Loco] monitor.put_data failed: {e}")

        if (iteration + 1) % save_interval == 0 and iteration != last_save_iter:
            agent.learn(list_sample_data=None)
            agent.save_model(id=str(iteration + 1))
            last_save_iter = iteration

    agent.save_model(id=str(max_iterations))
    total_time = time.time() - loop_start
    logger.info(
        f"[LBC-Loco] Training finished in {total_time:.1f}s, total steps={algorithm.total_steps}"
    )

    env.close()


def _lbc_step(env, agent, algorithm, obs, student_drive=False):
    """LBC single step.

    DAgger: student_drive=true -> update (trains on student closed-loop obs) and
    reuse the computed student_latent to drive env (one vision_encoder call,
    keeps LSTM hidden in sync with obs).
    """
    device = agent.device

    if student_drive:
        step_metrics = algorithm.update(obs)
        agent.learn(list_sample_data=None)
        with torch.no_grad():
            student_latent = step_metrics["student_latent"]
            obs_dict = algorithm._split_obs(obs)
            policy_input = algorithm._teacher_actor_input(obs_dict, student_latent)
            actions = algorithm.teacher_actor(policy_input)
    else:
        actions = algorithm.act_teacher(obs)
        step_metrics = algorithm.update(obs)
        agent.learn(list_sample_data=None)

    actions_clipped = torch.clip(actions, -6.0, 6.0).to(device)

    step_data = env.step(actions_clipped)
    if step_data is None:
        raise RuntimeError("[LBC] env.step returned None")

    frame_no, next_obs, rewards, terminated, truncated, (infos, privileged_obs) = step_data
    next_obs = torch.as_tensor(next_obs).to(device)

    dones = torch.logical_or(
        torch.as_tensor(terminated).to(device),
        torch.as_tensor(truncated).to(device),
    )
    if dones.any():
        algorithm.reset_student_hidden_states(dones)

    return next_obs, step_metrics
