#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
LBC 蒸馏训练 workflow。

lbc_loco:
    教师 = teacher_encoder(height_scan) + teacher_actor  (FROZEN)
    学生 = VisionEncoder (CNN + LSTM + head)             (TRAIN)
    loss = MSE(student_latent, teacher_latent)

驱动 env 的 actor:
    act_teacher (纯教师全链 — encoder/actor 都吃 scan)
"""

from __future__ import annotations

import math
import os
import time
from collections import deque

import torch

from agent_ppo.conf.conf import Config


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """LBC 主训练 workflow。"""
    agent = agents[0]
    env = envs[0]

    assert getattr(agent, "is_lbc", False), (
        "lbc_workflow.workflow called but agent.is_lbc is False; "
        "check Config.CURRENT and agent.__init__."
    )

    stage = agent.stage
    algorithm = agent.algorithm  # AlgorithmLBC

    # 读取配置（TOML 段名按 stage.name）
    usr_conf, usr_conf_file, is_eval, _stage = Config.load_conf(logger)
    section = stage.name  # "lbc_loco"
    lbc_conf = usr_conf.get(section, {}) if isinstance(usr_conf, dict) else {}

    max_iterations = int(lbc_conf.get("max_iterations", stage.max_iterations))
    log_interval = int(lbc_conf.get("log_interval", stage.log_interval))
    save_interval = int(lbc_conf.get("save_interval", stage.model_save_interval))
    num_steps_per_env = int(lbc_conf.get("num_steps_per_env", stage.num_steps_per_env))

    student_drive = bool(lbc_conf.get("student_drive", False))

    # LR override：toml 优先于 conf.py 的 stage.lr。
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

    lr_scheduler = None
    if not student_drive:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            algorithm.optimizer,
            T_max=max_iterations,
            eta_min=lr_min,
        )

    logger.info(
        f"[LBC-Loco] Start: "
        f"max_iterations={max_iterations}, log_interval={log_interval}, "
        f"save_interval={save_interval}, num_steps_per_env={num_steps_per_env}, "
        f"student_drive={student_drive}, "
        f"init_lr={algorithm.optimizer.param_groups[0]['lr']:.2e}, "
        f"lr_scheduler={f'cosine(eta_min={lr_min:.0e})' if lr_scheduler else 'none'}"
    )

    # env reset
    data = env.reset(usr_conf)
    if data is None:
        raise RuntimeError("[LBC] env.reset returned None, check env configuration.")
    obs, _critic_obs = data
    obs = torch.as_tensor(obs).to(agent.device)

    # 初始化 LSTM 隐状态
    if hasattr(algorithm.vision_encoder, "reset_hidden_state"):
        algorithm.vision_encoder.reset_hidden_state(batch_size=agent.num_envs, device=agent.device)

    if student_drive:
        algorithm.eval_mode()
    else:
        algorithm.train_mode()

    # 指标 windows
    metric_windows = {
        "grad_norm": deque(maxlen=log_interval),
        "mse_loss": deque(maxlen=log_interval),
        "l2_distance": deque(maxlen=log_interval),
        "cos_sim": deque(maxlen=log_interval),
        "student_std": deque(maxlen=log_interval),
        "teacher_std": deque(maxlen=log_interval),
    }

    loop_start = time.time()
    last_save_iter = 0

    # Main loop
    for iteration in range(max_iterations):
        algorithm.current_iteration = iteration
        ep_start = time.time()

        for _step in range(num_steps_per_env):
            obs, step_metrics = _lbc_step(env, agent, algorithm, obs, student_drive=student_drive)
            for key, window in metric_windows.items():
                if key in step_metrics:
                    window.append(step_metrics[key])

        # LR scheduler 步进
        if lr_scheduler is not None:
            lr_scheduler.step()

        # 日志
        if (iteration + 1) % log_interval == 0 or iteration == 0:

            def _avg(buf):
                return sum(buf) / max(1, len(buf))

            cur_lr = algorithm.optimizer.param_groups[0]["lr"]
            dt = time.time() - ep_start

            mean_mse = _avg(metric_windows["mse_loss"])
            mean_l2 = _avg(metric_windows["l2_distance"])
            mean_cos = _avg(metric_windows["cos_sim"])
            mean_s_std = _avg(metric_windows["student_std"])
            mean_t_std = _avg(metric_windows["teacher_std"])
            mean_grad = _avg(metric_windows["grad_norm"])
            # angle_deg: 学生/教师 latent 的夹角（度）。两侧均 L2-normed，
            # 故 mse ≡ 2(1-cos)，真正可读指标是角度，越小越贴合。
            angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, mean_cos))))
            logger.info(
                f"[LBC-Loco] iter={iteration+1}/{max_iterations}  "
                f"angle={angle_deg:.2f}deg  cos={mean_cos:.4f}  "
                f"mse={mean_mse:.5f}  l2={mean_l2:.4f}  "
                f"s_lat_bstd={mean_s_std:.4f}  t_lat_bstd={mean_t_std:.4f}  "
                f"grad={mean_grad:.3f}  lr={cur_lr:.2e}  "
                f"total_steps={algorithm.total_steps}  iter_time={dt:.2f}s"
            )
            if monitor is not None:
                try:
                    monitor.put_data(
                        {
                            os.getpid(): {
                                "iteration": iteration + 1,
                                "angle_deg": angle_deg,
                                "cos_sim": mean_cos,
                                "mse_loss": mean_mse,
                                "l2_distance": mean_l2,
                                "student_std": mean_s_std,
                                "teacher_std": mean_t_std,
                                "grad_norm": mean_grad,
                                "total_steps": algorithm.total_steps,
                            }
                        }
                    )
                except Exception as e:
                    logger.warning(f"[LBC-Loco] monitor.put_data failed: {e}")

        # 保存（仅训练模式；student_drive=true 是纯评估，参数未更新，避免覆盖 preload 的 ckpt）
        if not student_drive:
            if (iteration + 1) % save_interval == 0 and iteration != last_save_iter:
                agent.learn(list_sample_data=None)
                agent.save_model(id=str(iteration + 1))
                last_save_iter = iteration

    # 最终保存（仅训练模式）
    if not student_drive:
        agent.save_model(id=str(max_iterations))
    total_time = time.time() - loop_start
    logger.info(
        f"[LBC-Loco] "
        f"{'Training' if not student_drive else 'Student-drive eval'} "
        f"finished in {total_time:.1f}s, total steps={algorithm.total_steps}"
    )

    env.close()


def _lbc_step(env, agent, algorithm, obs, student_drive=False):
    """LBC 单步。按 student_drive 分发教师/学生驱动 env，update 返回 metrics。"""
    device = agent.device

    # 1. 选择驱动源
    if student_drive:
        actions = algorithm.act_student(obs)
    else:
        actions = algorithm.act_teacher(obs)
    actions_clipped = torch.clip(actions, -6.0, 6.0).to(device)

    # 2. update 或纯评估
    if student_drive:
        # 纯 eval 模式：算一次 loss 作 metric，不做 backward
        with torch.no_grad():
            loss_dict = algorithm.compute_latent_loss(obs)
            t_lat = loss_dict["teacher_latent"]
            s_lat = loss_dict["student_latent"]
            step_metrics = {
                "mse_loss": loss_dict["mse_loss"].item(),
                "l2_distance": loss_dict["l2_distance"].item(),
                "grad_norm": 0.0,
                "cos_sim": torch.nn.functional.cosine_similarity(s_lat, t_lat, dim=-1).mean().item(),
                "student_std": s_lat.std(dim=0).mean().item(),
                "teacher_std": t_lat.std(dim=0).mean().item(),
            }
    else:
        step_metrics = algorithm.update(obs)
        agent.learn(list_sample_data=None)

    # 3. env.step
    step_data = env.step(actions_clipped)
    if step_data is None:
        raise RuntimeError("[LBC] env.step returned None")

    frame_no, next_obs, rewards, terminated, truncated, (infos, privileged_obs) = step_data
    next_obs = torch.as_tensor(next_obs).to(device)

    # 4. LSTM 隐状态管理
    dones = torch.logical_or(
        torch.as_tensor(terminated).to(device),
        torch.as_tensor(truncated).to(device),
    )
    if dones.any():
        algorithm.reset_student_hidden_states(dones)

    return next_obs, step_metrics
