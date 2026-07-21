#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""LBC distillation workflow (latent/action matching + closed-loop DAgger).

lbc_loco:
    teacher = teacher_encoder(height_scan) + teacher_actor  (FROZEN)
    student = VisionEncoder (CNN + LSTM + head)            (TRAIN)
    D2 loss = 0.5 * SmoothL1(student_latent, teacher_latent)
            + 0.1 * cosine distance
            + 1.0 * SmoothL1(student_action, teacher_action)

Env driver:
    act_teacher (teacher full chain)
    D2 samples the driver independently for every environment. The student
    drives 50%, 75%, then 100% of environments across the three train phases,
    while the frozen teacher continues to label every observed state.

DAgger note: when student_drive, update() is called once (one vision_encoder
forward -> LSTM hidden advances one step, in sync with obs). The computed
student_latent is reused to drive env, avoiding double LSTM update.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque

import torch

from agent_ppo.conf.conf import Config
from agent_ppo.feature.nav_observation_utils import configure_depth_augmentation


def _snapshot_teacher(algorithm):
    return {
        f"encoder.{name}": value.detach().clone()
        for name, value in algorithm.teacher_encoder.state_dict().items()
    } | {
        f"actor.{name}": value.detach().clone()
        for name, value in algorithm.teacher_actor.state_dict().items()
    }


def _teacher_max_abs_diff(algorithm, initial_state):
    current = {
        f"encoder.{name}": value.detach()
        for name, value in algorithm.teacher_encoder.state_dict().items()
    } | {
        f"actor.{name}": value.detach()
        for name, value in algorithm.teacher_actor.state_dict().items()
    }
    return max(
        float(torch.max(torch.abs(current[name] - initial)).item())
        for name, initial in initial_state.items()
    )


def _env_metrics(env, attribute):
    for source in (env, getattr(env, "unwrapped", None)):
        metrics = getattr(source, attribute, None)
        if isinstance(metrics, dict):
            return metrics
    return {}


def _student_drive_probability(progress, phase_fractions, phase_ratios):
    """Return the piecewise-constant DAgger ratio for normalized progress."""
    if len(phase_fractions) != len(phase_ratios) or not phase_fractions:
        raise ValueError("student drive phase fractions/ratios must be non-empty and aligned")
    if any(fraction <= 0.0 for fraction in phase_fractions):
        raise ValueError("student drive phase fractions must be positive")
    if abs(sum(phase_fractions) - 1.0) > 1.0e-6:
        raise ValueError("student drive phase fractions must sum to 1.0")
    if any(ratio < 0.0 or ratio > 1.0 for ratio in phase_ratios):
        raise ValueError("student drive ratios must stay in [0, 1]")

    progress = min(1.0, max(0.0, float(progress)))
    boundary = 0.0
    for fraction, ratio in zip(phase_fractions, phase_ratios):
        boundary += fraction
        if progress < boundary:
            return float(ratio)
    return float(phase_ratios[-1])


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
    logger.info(f"[ST9-Opt3-D2] workflow={os.path.abspath(__file__)}")
    logger.info(
        f"[ST9-Opt3-D2] stage={stage.__name__}, config={usr_conf_file}, "
        f"task_type={stage.task_type}"
    )

    max_iterations = int(lbc_conf.get("max_iterations", stage.max_iterations))
    log_interval = int(lbc_conf.get("log_interval", stage.log_interval))
    save_interval = int(lbc_conf.get("save_interval", stage.model_save_interval))
    num_steps_per_env = int(lbc_conf.get("num_steps_per_env", stage.num_steps_per_env))
    sequence_length = max(
        1,
        min(
            num_steps_per_env,
            int(lbc_conf.get("sequence_length", lbc_conf.get("bptt_steps", 1))),
        ),
    )

    student_drive_conf = bool(lbc_conf.get("student_drive", False))
    phase_fractions = [
        float(value)
        for value in lbc_conf.get("student_drive_phase_fractions", [0.20, 0.40, 0.40])
    ]
    phase_ratios = [
        float(value)
        for value in lbc_conf.get("student_drive_ratios", [0.50, 0.75, 1.00])
    ]
    _student_drive_probability(0.0, phase_fractions, phase_ratios)

    algorithm.latent_loss_weight = float(lbc_conf.get("latent_loss_weight", 1.0))
    algorithm.cosine_loss_weight = float(lbc_conf.get("cosine_loss_weight", 0.0))
    algorithm.action_loss_weight = float(lbc_conf.get("action_loss_weight", 0.0))
    algorithm.latent_loss_type = str(lbc_conf.get("latent_loss_type", "legacy_mse"))
    algorithm.action_loss_type = str(lbc_conf.get("action_loss_type", "mse"))

    # Some platform launch paths inject the selected checkpoint only after the
    # agent is constructed. Preserve that documented compatibility path, then
    # fail hard if it did not provide a complete frozen teacher.
    if not algorithm.teacher_loaded:
        logger.info("[LBC-Loco] trying to load platform-selected locomotion teacher")
        try:
            agent.load_model(id="latest")
        except Exception as exc:
            logger.warning(f"[LBC-Loco] platform teacher load failed: {exc}")
    algorithm.assert_teacher_ready()
    if bool(lbc_conf.get("require_student_resume", False)):
        algorithm.assert_student_ready()
    teacher_encoder_frozen = (
        not algorithm.teacher_encoder.training
        and all(not parameter.requires_grad for parameter in algorithm.teacher_encoder.parameters())
    )
    teacher_actor_frozen = (
        not algorithm.teacher_actor.training
        and all(not parameter.requires_grad for parameter in algorithm.teacher_actor.parameters())
    )
    optimizer_ids = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    vision_ids = {id(parameter) for parameter in algorithm.vision_encoder.parameters()}
    teacher_ids = {
        id(parameter)
        for module in (algorithm.teacher_encoder, algorithm.teacher_actor)
        for parameter in module.parameters()
    }
    student_only_optimizer = optimizer_ids == vision_ids and not (optimizer_ids & teacher_ids)
    if not (teacher_encoder_frozen and teacher_actor_frozen and student_only_optimizer):
        raise RuntimeError(
            "[LBC] teacher freeze/optimizer contract failed: "
            f"encoder_frozen={teacher_encoder_frozen}, "
            f"actor_frozen={teacher_actor_frozen}, "
            f"student_only_optimizer={student_only_optimizer}."
        )
    initial_teacher_state = _snapshot_teacher(algorithm)
    logger.info(
        f"[ST9-Opt3-D2] teacher_checkpoint={algorithm.teacher_source}, "
        f"student_checkpoint={algorithm.student_source}, "
        f"teacher_encoder_frozen={teacher_encoder_frozen}, "
        f"teacher_actor_frozen={teacher_actor_frozen}, "
        f"vision_encoder_in_optimizer={student_only_optimizer}"
    )
    # ObservationProcess is invoked during reset, so configure image augmentation
    # before the formal env.reset(usr_conf) call.
    env._is_training = not is_eval
    env._is_eval = is_eval
    configure_depth_augmentation(
        usr_conf.get("depth_aug", {}),
        usr_conf.get("depth_block_dropout", {}),
        training=not is_eval,
    )
    logger.info(f"[ST9-Opt3-D2] goal_noise={usr_conf.get('goal_noise', {})}")
    logger.info(
        f"[ST9-Opt3-D2] depth_aug={usr_conf.get('depth_aug', {})}, "
        f"depth_block_dropout={usr_conf.get('depth_block_dropout', {})}"
    )

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
        f"sequence_length={sequence_length}, student_drive={student_drive_conf}, "
        f"student_drive_phase_fractions={phase_fractions}, "
        f"student_drive_ratios={phase_ratios}, "
        f"loss={algorithm.latent_loss_weight}*{algorithm.latent_loss_type}_latent + "
        f"{algorithm.cosine_loss_weight}*cosine + "
        f"{algorithm.action_loss_weight}*{algorithm.action_loss_type}_action, "
        f"init_lr={algorithm.optimizer.param_groups[0]['lr']:.2e}, "
        f"lr_scheduler=cosine(eta_min={lr_min:.0e})"
    )

    data = env.reset(usr_conf)
    if data is None:
        raise RuntimeError("[LBC] env.reset returned None, check env configuration.")
    obs, _critic_obs = data
    obs = torch.as_tensor(obs).to(agent.device)
    obs_dict = algorithm._split_obs(obs)
    shape_summary = {
        key: None if value is None else tuple(value.shape)
        for key, value in obs_dict.items()
    }
    expected_shapes = {
        "proprio": (agent.num_envs, stage.proprio_dim),
        "height_scan": (agent.num_envs, stage.scan_dim),
        "goal": (agent.num_envs, getattr(stage, "num_goal_obs", 0)),
        "depth_image": (
            agent.num_envs,
            stage.depth_height,
            stage.depth_width,
            stage.depth_channels,
        ),
    }
    for key, expected in expected_shapes.items():
        actual = shape_summary.get(key)
        if actual != expected:
            raise ValueError(f"[LBC] {key} shape mismatch: expected {expected}, got {actual}.")
    logger.info(
        f"[ST9-Opt3-D2] shapes={shape_summary}, latent={stage.latent_dim}, "
        f"action={stage.num_actions}"
    )

    if hasattr(algorithm.vision_encoder, "reset_hidden_state"):
        algorithm.vision_encoder.reset_hidden_state(batch_size=agent.num_envs, device=agent.device)

    algorithm.train_mode()

    metric_windows = {
        "grad_norm": deque(maxlen=log_interval),
        "total_loss": deque(maxlen=log_interval),
        "latent_loss": deque(maxlen=log_interval),
        "cosine_loss": deque(maxlen=log_interval),
        "mse_loss": deque(maxlen=log_interval),
        "action_loss": deque(maxlen=log_interval),
        "distance": deque(maxlen=log_interval),
        "cos_sim": deque(maxlen=log_interval),
        "student_std": deque(maxlen=log_interval),
        "teacher_std": deque(maxlen=log_interval),
        "student_drive_ratio": deque(maxlen=log_interval),
        "episode_return": deque(maxlen=log_interval),
        "episode_length": deque(maxlen=log_interval),
        "terminated_rate": deque(maxlen=log_interval),
        "truncated_rate": deque(maxlen=log_interval),
        "goal_noise_active_ratio": deque(maxlen=log_interval),
        "goal_bearing_noise_abs_mean": deque(maxlen=log_interval),
        "goal_distance_noise_abs_mean": deque(maxlen=log_interval),
        "goal_bearing_bias_abs_mean": deque(maxlen=log_interval),
        "goal_distance_bias_abs_mean": deque(maxlen=log_interval),
        "depth_random_dropout_ratio": deque(maxlen=log_interval),
        "depth_block_dropout_frame_ratio": deque(maxlen=log_interval),
        "depth_block_dropout_area_ratio": deque(maxlen=log_interval),
        "active_block_count": deque(maxlen=log_interval),
        "block_persistence_mean": deque(maxlen=log_interval),
        "valid_depth_ratio_before": deque(maxlen=log_interval),
        "valid_depth_ratio_after": deque(maxlen=log_interval),
    }
    cur_reward_sum = torch.zeros(agent.num_envs, device=agent.device)
    cur_episode_length = torch.zeros(agent.num_envs, device=agent.device)
    continuation_masks = None

    loop_start = time.time()
    last_save_iter = 0
    loss_scale_logged = False

    for iteration in range(max_iterations):
        algorithm.current_iteration = iteration
        ep_start = time.time()

        progress = iteration / max(1, max_iterations)
        p_student = (
            _student_drive_probability(progress, phase_fractions, phase_ratios)
            if student_drive_conf
            else 0.0
        )

        for sequence_start in range(0, num_steps_per_env, sequence_length):
            sequence_len = min(sequence_length, num_steps_per_env - sequence_start)
            algorithm.begin_sequence()
            last_step_metrics = None
            for _step in range(sequence_len):
                obs, step_metrics, dones, rewards, terminated, truncated = _lbc_step(
                    env,
                    agent,
                    algorithm,
                    obs,
                    student_drive_probability=p_student,
                    masks=continuation_masks,
                )
                if not loss_scale_logged:
                    logger.info(
                        "[ST9-Opt3-D2 LossScale] "
                        f"latent_unweighted={step_metrics['latent_loss']:.8f}, "
                        f"cosine_unweighted={step_metrics['cosine_loss']:.8f}, "
                        f"action_unweighted={step_metrics['action_loss']:.8f}, "
                        f"total_weighted={step_metrics['total_loss']:.8f}"
                    )
                    loss_scale_logged = True
                continuation_masks = ~dones
                cur_reward_sum, cur_episode_length = algorithm.update_episode_stats(
                    rewards, dones, cur_reward_sum, cur_episode_length
                )
                stats = algorithm.get_training_stats()
                step_metrics.update({
                    "episode_return": stats.get("mean_reward", 0.0),
                    "episode_length": stats.get("mean_episode_length", 0.0),
                    "terminated_rate": terminated.float().mean().item(),
                    "truncated_rate": truncated.float().mean().item(),
                })
                step_metrics.update(_env_metrics(env, "_goal_noise_metrics"))
                step_metrics.update(_env_metrics(env, "_depth_aug_metrics"))
                for key, window in metric_windows.items():
                    if key in step_metrics:
                        window.append(step_metrics[key])
                last_step_metrics = step_metrics
            grad_norm = algorithm.finish_sequence()
            if last_step_metrics is not None:
                metric_windows["grad_norm"].append(grad_norm)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Keep the platform training lifecycle aligned with PPO: one learn
        # callback per completed rollout.  Agent.learn() is deliberately a
        # no-op for LBC, because finish_sequence() already updated the student,
        # but the framework uses this callback to advance cumulative training
        # progress.
        agent.learn(list_sample_data=None)

        if (iteration + 1) % log_interval == 0 or iteration == 0:

            def _avg(buf):
                return sum(buf) / max(1, len(buf))

            cur_lr = algorithm.optimizer.param_groups[0]["lr"]
            dt = time.time() - ep_start

            mean_total = _avg(metric_windows["total_loss"])
            mean_latent = _avg(metric_windows["latent_loss"])
            mean_cosine_loss = _avg(metric_windows["cosine_loss"])
            mean_mse = _avg(metric_windows["mse_loss"])
            mean_action_loss = _avg(metric_windows["action_loss"])
            mean_distance = _avg(metric_windows["distance"])
            mean_cos = _avg(metric_windows["cos_sim"])
            mean_s_std = _avg(metric_windows["student_std"])
            mean_t_std = _avg(metric_windows["teacher_std"])
            mean_grad = _avg(metric_windows["grad_norm"])
            mean_sd_ratio = _avg(metric_windows["student_drive_ratio"])
            mean_return = _avg(metric_windows["episode_return"])
            mean_ep_len = _avg(metric_windows["episode_length"])
            mean_terminated = _avg(metric_windows["terminated_rate"])
            mean_truncated = _avg(metric_windows["truncated_rate"])
            sensor_metrics = {
                key: _avg(metric_windows[key])
                for key in (
                    "goal_noise_active_ratio",
                    "goal_bearing_noise_abs_mean",
                    "goal_distance_noise_abs_mean",
                    "goal_bearing_bias_abs_mean",
                    "goal_distance_bias_abs_mean",
                    "depth_random_dropout_ratio",
                    "depth_block_dropout_frame_ratio",
                    "depth_block_dropout_area_ratio",
                    "active_block_count",
                    "block_persistence_mean",
                    "valid_depth_ratio_before",
                    "valid_depth_ratio_after",
                )
            }
            teacher_max_abs_diff = _teacher_max_abs_diff(
                algorithm,
                initial_teacher_state,
            )
            if teacher_max_abs_diff != 0.0:
                raise RuntimeError(
                    "[LBC] frozen teacher parameters changed during distillation: "
                    f"max_abs_diff={teacher_max_abs_diff}."
                )
            angle = math.degrees(math.acos(max(-1.0, min(1.0, mean_cos))))
            logger.info(
                f"[LBC-Loco] iter={iteration+1}/{max_iterations}  "
                f"angle={angle:.2f}deg  cos={mean_cos:.4f}  "
                f"total={mean_total:.6f} latent={mean_latent:.6f} "
                f"cos_loss={mean_cosine_loss:.6f} act={mean_action_loss:.6f} "
                f"legacy_mse={mean_mse:.5f} "
                f"distance={mean_distance:.4f}  "
                f"s_lat_bstd={mean_s_std:.4f}  t_lat_bstd={mean_t_std:.4f}  "
                f"grad={mean_grad:.3f}  lr={cur_lr:.2e}  "
                f"p_student={p_student:.2f}  sd_ratio={mean_sd_ratio:.2f}  "
                f"ep_return={mean_return:.3f} ep_len={mean_ep_len:.1f} "
                f"terminated={mean_terminated:.3f} truncated={mean_truncated:.3f} "
                f"teacher_max_abs_diff={teacher_max_abs_diff:.3g} "
                f"total_steps={algorithm.total_steps}  iter_time={dt:.2f}s"
            )
            logger.info(
                "[ST9-Opt3-D2 Sensors] "
                + ", ".join(f"{key}={value:.6f}" for key, value in sensor_metrics.items())
            )
            if monitor is not None:
                try:
                    monitor.put_data(
                        {
                            os.getpid(): {
                                "iteration": iteration + 1,
                                "total_loss": mean_total,
                                "latent_loss": mean_latent,
                                "cosine_loss": mean_cosine_loss,
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
                                "episode_return": mean_return,
                                "episode_length": mean_ep_len,
                                "terminated_rate": mean_terminated,
                                "truncated_rate": mean_truncated,
                                "total_steps": algorithm.total_steps,
                                "latent_mse": mean_mse,
                                "cosine_similarity": mean_cos,
                                "angle_deg": angle,
                                "learning_rate": cur_lr,
                                "teacher_student_action_mse": mean_action_loss,
                                "teacher_max_abs_diff": teacher_max_abs_diff,
                                **sensor_metrics,
                            }
                        }
                    )
                except Exception as e:
                    logger.warning(f"[LBC-Loco] monitor.put_data failed: {e}")

        # The platform probes model_dir for model.ckpt-<label>-<id>.* while
        # training is still running.  Emit the first valid student artifact
        # after one complete rollout instead of waiting for save_interval.
        if iteration == 0 or (
            (iteration + 1) % save_interval == 0 and iteration != last_save_iter
        ):
            agent.save_model(id=str(iteration + 1))
            last_save_iter = iteration

    agent.save_model(id=str(max_iterations))
    teacher_max_abs_diff = _teacher_max_abs_diff(algorithm, initial_teacher_state)
    logger.info(
        f"[ST9-Opt3-D2] teacher_parameter_max_abs_diff={teacher_max_abs_diff:.9g}"
    )
    if teacher_max_abs_diff != 0.0:
        raise RuntimeError(
            "[LBC] frozen teacher parameters changed during distillation: "
            f"max_abs_diff={teacher_max_abs_diff}."
        )
    total_time = time.time() - loop_start
    logger.info(
        f"[LBC-Loco] Training finished in {total_time:.1f}s, total steps={algorithm.total_steps}"
    )

    env.close()


def _lbc_step(
    env,
    agent,
    algorithm,
    obs,
    student_drive_probability=0.0,
    masks=None,
):
    """LBC single step.

    DAgger samples the driver independently per environment and reuses the
    already-computed raw teacher/student actions. This keeps one VisionEncoder
    forward per observation and preserves LSTM time alignment.
    """
    device = agent.device

    step_metrics = algorithm.accumulate(obs, masks=masks)
    teacher_actions = step_metrics["teacher_action"]
    student_actions = step_metrics["student_action"]
    drive_mask = torch.rand(
        teacher_actions.shape[0],
        device=teacher_actions.device,
    ) < float(student_drive_probability)
    actions = torch.where(drive_mask.unsqueeze(-1), student_actions, teacher_actions)
    step_metrics["student_drive_ratio"] = drive_mask.float().mean().item()

    actions_clipped = torch.clip(actions, -6.0, 6.0).to(device)

    step_data = env.step(actions_clipped)
    if step_data is None:
        raise RuntimeError("[LBC] env.step returned None")

    frame_no, next_obs, rewards, terminated, truncated, (infos, privileged_obs) = step_data
    next_obs = torch.as_tensor(next_obs).to(device)

    terminated = torch.as_tensor(terminated, device=device).reshape(-1).bool()
    truncated = torch.as_tensor(truncated, device=device).reshape(-1).bool()
    rewards = torch.as_tensor(rewards, device=device).reshape(-1)
    dones = torch.logical_or(terminated, truncated)
    if dones.any():
        algorithm.reset_student_hidden_states(dones)

    return next_obs, step_metrics, dones, rewards, terminated, truncated
