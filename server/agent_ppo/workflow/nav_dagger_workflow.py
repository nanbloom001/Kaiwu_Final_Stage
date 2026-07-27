#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""nav_dagger_workflow — hier-nav 高层 DAgger 主循环（Track + Camera）。

仿 lbc_workflow 的骨架（线性 ramp、soft-stay、iteration 计数、monitor 惯例），
两处结构差异：

  1. 训练更新时机 = NavTickBuffer 凑满 T=16 个 nav tick（TBPTT 序列 CE），
     不是每 env 步；一个 outer iteration = 160 低层帧 = 16 tick = 恰好一段。
  2. soft-stay 与监控使用分类指标（ce/top1/disagreement/entropy/switch +
     hard termination + nonfinite 回退），不移植 action-MSE safety takeover。

另从 visual_ppo_workflow 复制墙钟任务必需的 SIGTERM 优雅保存三件套为私有
副本（lbc/visual workflow 本体零改动）。ramp = 0 即 Oracle 空转轮（N0 门）。
"""

import math
import os
import signal
import threading
import time
from collections import defaultdict

import torch

from agent_ppo.conf.conf import Config
from agent_ppo.checkpoint_io import CheckpointSaveError
from agent_ppo.feature import nav_contract
from agent_ppo.feature.nav_event_log import emit_nav_event


# soft-stay 阈值（每 _SOFT_STAY_WINDOW iterations 检查一次，连续两个窗口）
_SOFT_STAY_WINDOW = 50
_SOFT_STAY_DISAGREE_DELTA = 0.05    # disagreement 相对前窗口恶化超此值触发
_SOFT_STAY_HARD_DELTA = 0.02        # hard_termination 相对前窗口恶化超此值触发
_SOFT_STAY_PROGRESS_DELTA = 0.003   # goal progress (m/frame) 相对前窗口恶化超此值触发
# 解冻的绝对质量门槛（不只是"不再恶化"）
_SOFT_STAY_UNFREEZE_MIN_TOP1 = 0.85
_SOFT_STAY_UNFREEZE_MAX_HARD = 0.05
_DEFAULT_PLATFORM_DUMP_MODEL_FREQ = 3600


def _platform_dump_model_freq() -> int:
    """Read the framework dump cadence without making it a Nav config knob."""

    try:
        from common_python.config.config_control import CONFIG as app_config

        value = int(app_config.dump_model_freq)
    except (AttributeError, ImportError, TypeError, ValueError):
        value = _DEFAULT_PLATFORM_DUMP_MODEL_FREQ
    if value <= 0:
        raise ValueError(f"dump_model_freq must be positive, got {value}")
    return value


def _callbacks_until_next_dump(completed_callbacks: int, dump_frequency: int) -> int:
    """Return 0 on a dump boundary, otherwise successful callbacks remaining."""

    if completed_callbacks < 0:
        raise ValueError("completed_callbacks must be non-negative")
    if dump_frequency <= 0:
        raise ValueError("dump_frequency must be positive")
    if completed_callbacks == 0:
        return dump_frequency
    remainder = completed_callbacks % dump_frequency
    return 0 if remainder == 0 else dump_frequency - remainder


def _mean_metrics(rows: list) -> dict:
    values = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            if k.startswith("_"):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                values[k].append(fv)
    return {k: sum(vs) / len(vs) for k, vs in values.items() if vs}


def _episode_outcome_rates(
    hard_events: float, timeout_events: float, completed_episodes: float
) -> dict:
    """Episode outcome rates; never divide terminations by environment frames."""

    completed = float(completed_episodes)
    result = {
        "hard_termination_count": float(hard_events),
        "timeout_count": float(timeout_events),
        "completed_episode_count": completed,
    }
    if completed > 0.0:
        result["hard_termination_rate"] = float(hard_events) / completed
        result["timeout_rate"] = float(timeout_events) / completed
    return result


def _quality_window_metrics(rows: list[dict]) -> dict:
    """Average dense metrics and aggregate sparse episode outcomes."""

    metrics = _mean_metrics(rows)
    hard = sum(float(row.get("hard_termination_count", 0.0)) for row in rows)
    timeout = sum(float(row.get("timeout_count", 0.0)) for row in rows)
    completed = sum(float(row.get("completed_episode_count", 0.0)) for row in rows)
    metrics.update(_episode_outcome_rates(hard, timeout, completed))
    return metrics


def _aggregate_goal_metrics(rows: list[dict]) -> dict:
    valid = sum(float(row.get("oracle_valid_count", 0.0)) for row in rows)
    samples = sum(float(row.get("oracle_sample_count", 0.0)) for row in rows)
    fresh = sum(float(row.get("goal4_fresh_count", 0.0)) for row in rows)
    fresh_samples = sum(float(row.get("goal4_sample_count", 0.0)) for row in rows)
    return {
        "goal_valid_count": valid,
        "goal_sample_count": samples,
        "goal_valid_rate": valid / samples if samples > 0 else float("nan"),
        "goal4_fresh_count": fresh,
        "goal4_sample_count": fresh_samples,
        "goal4_fresh_rate": fresh / fresh_samples if fresh_samples > 0 else float("nan"),
    }


def _ramp_probability(elapsed_h: float, ramp_start_h: float, ramp_end_h: float) -> float:
    """线性 ramp：elapsed < ramp_start 时 0，到 ramp_end 时 1。"""
    if elapsed_h < ramp_start_h:
        return 0.0
    if elapsed_h >= ramp_end_h:
        return 1.0
    return (elapsed_h - ramp_start_h) / (ramp_end_h - ramp_start_h)


def _ramp_elapsed_from_probability(p: float, ramp_start_h: float, ramp_end_h: float) -> float:
    """从 ramp_probability 反推已用 ramp 时长（续训恢复用）。"""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return ramp_end_h
    return ramp_start_h + p * (ramp_end_h - ramp_start_h)


def _save_final_checkpoint(agent, logger, *, reason: str) -> bool:
    """优雅退出的 best-effort 收尾保存（主循环内的保存失败仍是硬失败）。"""
    if (
        not getattr(agent, "is_nav_dagger", False)
        or not getattr(agent, "_nav_training_started", False)
        or getattr(agent, "_nav_final_save_done", False)
    ):
        return False
    try:
        logger.warning(
            f"[NavDAgger] graceful workflow exit; saving final checkpoint (reason={reason})"
        )
        agent.save_model()
    except Exception as exc:  # 保留原始退出信号
        logger.error(
            f"[NavDAgger] final checkpoint save failed during graceful exit: {exc}"
        )
        return False
    agent._nav_final_save_done = True
    agent._nav_final_save_reason = reason
    return True


def _install_sigterm_checkpoint_handler(logger):
    """平台按信号结束墙钟任务；仅替换默认 SIGTERM 动作，框架 handler 优先。"""
    if (
        not hasattr(signal, "SIGTERM")
        or threading.current_thread() is not threading.main_thread()
    ):
        return None
    previous = signal.getsignal(signal.SIGTERM)
    if previous is not signal.SIG_DFL:
        logger.info(
            "[NavDAgger] preserving existing SIGTERM handler; "
            "final checkpoint depends on its graceful-exit path"
        )
        return None

    def _handle_sigterm(signum, _frame):
        raise SystemExit(f"SIGTERM({signum})")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    logger.info("[NavDAgger] installed default SIGTERM graceful-save handler")
    return previous


def _restore_sigterm_handler(previous) -> None:
    if previous is not None:
        signal.signal(signal.SIGTERM, previous)


def _extract_step(step_data):
    frame_no, next_obs, rewards, terminated, truncated, extra = step_data
    infos, privileged_obs = extra
    return frame_no, next_obs, rewards, terminated, truncated, infos, privileged_obs


def _soft_stay_check(window: dict, prev_window: dict | None) -> str | None:
    if window.get("nonfinite_fallback", 0.0) > 0.0:
        return "nonfinite_fallback"
    if prev_window is not None:
        if (
            window.get("disagreement_rate", 0.0)
            > prev_window.get("disagreement_rate", 0.0) + _SOFT_STAY_DISAGREE_DELTA
        ):
            return "disagreement_worsened"
        if (
            "hard_termination_rate" in window
            and "hard_termination_rate" in prev_window
            and window["hard_termination_rate"]
            > prev_window["hard_termination_rate"] + _SOFT_STAY_HARD_DELTA
        ):
            return "hard_termination_worsened"
        if (
            window.get("goal_progress_m_per_frame", 0.0)
            < prev_window.get("goal_progress_m_per_frame", 0.0)
            - _SOFT_STAY_PROGRESS_DELTA
        ):
            return "goal_progress_worsened"
    return None


def _quality_absolutely_ok(window: dict) -> bool:
    return (
        window.get("top1_accuracy", 0.0) >= _SOFT_STAY_UNFREEZE_MIN_TOP1
        and "hard_termination_rate" in window
        and window["hard_termination_rate"] <= _SOFT_STAY_UNFREEZE_MAX_HARD
    )


def _workflow_impl(envs, agents, logger=None, monitor=None, *args, **kwargs):
    agent = agents[0]
    env = envs[0]
    stage = agent.stage
    algorithm = agent.algorithm  # AlgorithmNavDagger

    logger.info(
        "[LifecycleProbe] nav_workflow_impl enter "
        f"pid={os.getpid()} stage={stage.name} "
        f"parent_loaded={algorithm.low_level_state_digest is not None} "
        f"resume={algorithm.resume_loaded}"
    )
    emit_nav_event(
        "workflow_enter",
        role=getattr(agent, "_process_role", "unknown"),
        stage=stage.name,
        resume=bool(algorithm.resume_loaded),
    )
    usr_conf, usr_conf_file, is_eval, _stage = Config.load_conf(logger)
    logger.info(
        "[LifecycleProbe] nav_workflow config_resolved "
        f"conf={usr_conf_file} is_eval={is_eval} resolved_stage={_stage.name}"
    )
    section = stage.name  # "nav_dagger"
    nav_conf = usr_conf.get(section, {}) if isinstance(usr_conf, dict) else {}

    # ---- 超参（TOML 优先、类默认兜底）----
    max_iterations = int(nav_conf.get("max_iterations", stage.max_iterations))
    log_interval = int(nav_conf.get("log_interval", stage.log_interval))
    num_steps_per_env = int(
        nav_conf.get("num_steps_per_env", stage.num_steps_per_env)
    )
    tbptt_T = int(
        nav_conf.get("tbptt_sequence_length", stage.tbptt_sequence_length)
    )
    lr = float(nav_conf.get("learning_rate", stage.lr))
    lr_min = float(nav_conf.get("lr_min", stage.lr_min))
    lr_scheduler_iterations = int(
        nav_conf.get("lr_scheduler_iterations", stage.lr_scheduler_iterations)
    )
    dump_model_freq = _platform_dump_model_freq()

    # ---- 契约一致性启动断言（TOML 镜像 == nav_contract 权威值）----
    # 镜像键必填：缺键会让 get(默认=契约值) 自比自、断言必过——静默绕过
    for required_key in ("nav_period_frames", "min_dwell_ticks", "tbptt_sequence_length"):
        if required_key not in nav_conf:
            raise ValueError(
                f"TOML [{section}] missing required contract-mirror key: "
                f"{required_key} (must be present and equal to nav_contract)"
            )
    toml_period = int(nav_conf["nav_period_frames"])
    toml_dwell = int(nav_conf["min_dwell_ticks"])
    if toml_period != nav_contract.NAV_PERIOD_FRAMES:
        raise ValueError(
            f"TOML nav_period_frames={toml_period} != contract "
            f"{nav_contract.NAV_PERIOD_FRAMES} — nav_contract.py is authoritative"
        )
    if toml_dwell != nav_contract.MIN_DWELL_TICKS:
        raise ValueError(
            f"TOML min_dwell_ticks={toml_dwell} != contract "
            f"{nav_contract.MIN_DWELL_TICKS} — nav_contract.py is authoritative"
        )
    if tbptt_T != nav_contract.TBPTT_T:
        raise ValueError(
            f"TOML tbptt_sequence_length={tbptt_T} != contract {nav_contract.TBPTT_T}"
        )
    if num_steps_per_env != tbptt_T * nav_contract.NAV_PERIOD_FRAMES:
        raise ValueError(
            "num_steps_per_env must equal tbptt_T * nav_period_frames "
            f"({tbptt_T} * {nav_contract.NAV_PERIOD_FRAMES} = "
            f"{tbptt_T * nav_contract.NAV_PERIOD_FRAMES}), got {num_steps_per_env}"
        )

    # ---- ramp 调度（TOML 覆盖 checkpoint 旧调度，写回 algorithm）----
    ramp_start_h = float(nav_conf.get("ramp_start_h", algorithm.ramp_start_h))
    ramp_end_h = float(nav_conf.get("ramp_end_h", algorithm.ramp_end_h))
    if ramp_end_h <= ramp_start_h:
        raise ValueError(f"ramp_end_h({ramp_end_h}) must be > ramp_start_h({ramp_start_h})")
    algorithm.ramp_start_h = ramp_start_h
    algorithm.ramp_end_h = ramp_end_h

    # ---- LR / scheduler（续训保留 optimizer 状态，强制覆写 T_max）----
    if not algorithm.resume_loaded:
        for group in algorithm.optimizer.param_groups:
            group["lr"] = lr
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        algorithm.optimizer, T_max=lr_scheduler_iterations, eta_min=lr_min
    )
    if algorithm.resume_loaded and algorithm.lr_scheduler_state:
        try:
            lr_scheduler.load_state_dict(algorithm.lr_scheduler_state)
            lr_scheduler.T_max = lr_scheduler_iterations
        except (KeyError, ValueError, RuntimeError) as exc:
            logger.warning(f"[NavDAgger] lr scheduler restore failed: {exc}")

    # 续训：优先沿用 checkpoint 里保存的 ramp_clock_h（load_nav_resume 已恢复）；
    # 仅当 clock 缺失/为零而比例已非零时才从比例反推（lbc 先例的 fallback 语义）
    if algorithm.resume_loaded and (
        algorithm.ramp_clock_h <= 0.0 and algorithm.ramp_probability > 0.0
    ):
        algorithm.ramp_clock_h = _ramp_elapsed_from_probability(
            algorithm.ramp_probability, ramp_start_h, ramp_end_h
        )

    logger.info(
        f"[NavDAgger] start: conf={usr_conf_file} max_iter={max_iterations} "
        f"steps/iter={num_steps_per_env} T={tbptt_T} "
        f"platform_dump_model_freq={dump_model_freq} "
        f"ramp=[{ramp_start_h},{ramp_end_h}]h resume={algorithm.resume_loaded} "
        f"ramp_p={algorithm.ramp_probability:.3f} "
        f"low_level_digest={algorithm.low_level_state_digest}"
    )

    # ---- env reset ----
    logger.info(
        "[LifecycleProbe] nav_env_reset begin "
        f"pid={os.getpid()} env_type={type(env).__module__}.{type(env).__name__}"
    )
    reset_started = time.monotonic()
    data = env.reset(usr_conf)
    if data is None:
        raise RuntimeError("env.reset returned None")
    obs, critic_obs = data
    # clone：inject 会就地写观测副本；env 若返回同设备张量，.to(device) 是
    # 恒等别名，必须显式拷贝（与 eval 路径的 obs.clone() 对称）
    obs = torch.as_tensor(obs).to(agent.device).clone()
    critic_obs = torch.as_tensor(critic_obs).to(agent.device).clone()
    logger.info(
        f"[NavDAgger] reset ok: obs={tuple(obs.shape)} critic={tuple(critic_obs.shape)} "
        f"elapsed_s={time.monotonic() - reset_started:.3f}"
    )
    emit_nav_event(
        "reset_complete",
        obs_shape=list(obs.shape),
        critic_shape=list(critic_obs.shape),
        num_envs=int(obs.shape[0]),
    )

    algorithm.train_mode()

    start_iteration = int(algorithm.current_iteration)
    iter_metric_history: list[dict] = []
    prev_window: dict | None = None
    iter_end_prev = time.monotonic()
    platform_lifecycle_callbacks = 0
    platform_lifecycle_failures = 0
    total_low_level_steps = 0
    session_total_env_frames = 0
    first_lifecycle_success_logged = False
    agent._nav_platform_dump_model_freq = dump_model_freq
    agent._nav_lifecycle_success_callbacks = 0
    agent._nav_lifecycle_failure_callbacks = 0
    agent._nav_lifecycle_attempt_callbacks = 0

    try:
        for iteration_index in range(start_iteration, max_iterations):
            iter_start = time.monotonic()
            first_iteration_probe = iteration_index == start_iteration
            if first_iteration_probe:
                logger.info(
                    "[LifecycleProbe] nav_iteration first_begin "
                    f"iteration={iteration_index + 1} steps={num_steps_per_env}"
                )

            # ramp clock：冻结时暂停
            if not algorithm.soft_stay_frozen:
                algorithm.ramp_clock_h += max(0.0, iter_start - iter_end_prev) / 3600.0
            iter_end_prev = iter_start

            requested_p = _ramp_probability(
                algorithm.ramp_clock_h, ramp_start_h, ramp_end_h
            )
            if algorithm.soft_stay_frozen:
                effective_p = algorithm.ramp_probability  # 冻结保持
            else:
                effective_p = max(requested_p, algorithm.ramp_probability)  # 单调不回退
                algorithm.ramp_probability = effective_p

            step_rows: list[dict] = []
            # 全帧聚合指标（不能只在 tick 帧采样：换 token 后的失稳集中在
            # tick+1..tick+9，恰好全是非 tick 帧）
            hard_events = 0.0
            timeout_events = 0.0
            completed_episodes = 0.0
            progress_sum = 0.0
            progress_count = 0
            total_env_frames = 0
            g_dist_idx = nav_contract.CRITIC_GOAL3_START + 2
            for step_index in range(num_steps_per_env):
                pre_goal_dist = critic_obs[:, g_dist_idx] * nav_contract.GOAL_DIST_SCALE_M
                if first_iteration_probe and step_index == 0:
                    logger.info("[LifecycleProbe] nav_first_frame algorithm begin")
                result = algorithm.frame_begin(obs, critic_obs)
                actions = torch.clip(result["actions"], -6.0, 6.0)
                if first_iteration_probe and step_index == 0:
                    logger.info(
                        "[LifecycleProbe] nav_first_frame algorithm complete "
                        f"actions={tuple(actions.shape)} is_tick={result['is_tick']}"
                    )
                    emit_nav_event(
                        "first_frame",
                        actions_shape=list(actions.shape),
                        is_nav_tick=bool(result["is_tick"]),
                    )

                if first_iteration_probe and step_index == 0:
                    logger.info("[LifecycleProbe] nav_first_frame env_step begin")
                step_data = env.step(actions)
                if first_iteration_probe and step_index == 0:
                    logger.info("[LifecycleProbe] nav_first_frame env_step complete")
                (
                    _frame_no,
                    next_obs,
                    _rewards,
                    terminated,
                    truncated,
                    infos,
                    privileged_obs,
                ) = _extract_step(step_data)

                next_obs = torch.as_tensor(next_obs).to(agent.device).clone()
                privileged_obs = torch.as_tensor(privileged_obs).to(agent.device).clone()
                terminated = torch.as_tensor(terminated).to(agent.device).bool()
                truncated = torch.as_tensor(truncated).to(agent.device).bool()
                dones = terminated | truncated
                if isinstance(infos, dict) and "time_outs" in infos:
                    time_outs = torch.as_tensor(infos["time_outs"]).to(agent.device).bool()
                else:
                    time_outs = truncated
                timeout_termination = time_outs & dones
                hard_termination = terminated & ~time_outs

                algorithm.frame_end(dones)

                # 全帧指标累计
                num_envs_now = int(dones.shape[0])
                total_env_frames += num_envs_now
                hard_events += float(hard_termination.float().sum().item())
                timeout_events += float(timeout_termination.float().sum().item())
                completed_episodes += float(dones.float().sum().item())
                post_goal_dist = (
                    privileged_obs[:, g_dist_idx] * nav_contract.GOAL_DIST_SCALE_M
                )
                keep = ~dones
                if bool(keep.any()):
                    progress = pre_goal_dist[keep] - post_goal_dist[keep]
                    progress_sum += float(progress.sum().item())
                    progress_count += int(keep.sum().item())

                if result["is_tick"]:
                    row = dict(result["tick_metrics"])
                    if row.pop("buffer_full", False):
                        first_update_probe = not agent._nav_training_started
                        if first_update_probe:
                            logger.info("[LifecycleProbe] nav_first_update begin")
                        update_metrics = algorithm.finish_nav_sequence_update()
                        agent._nav_training_started = True
                        if first_update_probe:
                            logger.info(
                                "[LifecycleProbe] nav_first_update complete "
                                f"metrics={update_metrics}"
                            )
                            if (
                                float(update_metrics.get("valid_ticks", 0.0)) > 0.0
                                and not update_metrics.get("update_skipped_no_valid", 0.0)
                            ):
                                emit_nav_event(
                                    "first_update_complete", **update_metrics
                                )
                            else:
                                emit_nav_event(
                                    "first_update_skipped", **update_metrics
                                )
                        row.update(update_metrics)
                    step_rows.append(row)

                obs = next_obs
                critic_obs = privileged_obs

                # BaseAgent.learn is a Nav no-op for gradients, but each successful
                # call advances the platform lifecycle and its dump_model_freq clock.
                # Keep it after all frame/TBPTT processing so automatic dumps observe
                # the latest stable weights. A failed env.step never reaches here.
                total_low_level_steps += 1
                session_total_env_frames += num_envs_now
                first_lifecycle_attempt = (
                    platform_lifecycle_callbacks == 0
                    and platform_lifecycle_failures == 0
                )
                agent._nav_lifecycle_attempt_callbacks = (
                    platform_lifecycle_callbacks + 1
                )
                if first_lifecycle_attempt:
                    logger.info("[LifecycleProbe] nav_first_lifecycle_callback begin")
                try:
                    agent.learn(list_sample_data=None)
                except CheckpointSaveError as exc:
                    logger.error(
                        "[NavDAgger] checkpoint save failed inside platform "
                        f"lifecycle callback; stopping training: {exc}"
                    )
                    emit_nav_event(
                        "checkpoint_save_failure",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise
                except Exception as exc:
                    platform_lifecycle_failures += 1
                    agent._nav_lifecycle_failure_callbacks = platform_lifecycle_failures
                    if (
                        platform_lifecycle_failures == 1
                        or platform_lifecycle_failures % 100 == 0
                    ):
                        logger.error(
                            "[NavDAgger] platform lifecycle callback failed; "
                            "training continues but failed callbacks do not advance "
                            "model publication: "
                            f"failures={platform_lifecycle_failures} "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        emit_nav_event(
                            "platform_lifecycle_failure",
                            failures=platform_lifecycle_failures,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                else:
                    platform_lifecycle_callbacks += 1
                    agent._nav_lifecycle_success_callbacks = platform_lifecycle_callbacks
                    if not first_lifecycle_success_logged:
                        first_lifecycle_success_logged = True
                        logger.info(
                            "[LifecycleProbe] nav_first_lifecycle_callback complete"
                        )
                    callbacks_until_next_dump = _callbacks_until_next_dump(
                        platform_lifecycle_callbacks, dump_model_freq
                    )
                    if callbacks_until_next_dump == 0:
                        logger.info(
                            "[NavDAgger] platform lifecycle dump boundary reached "
                            f"callbacks={platform_lifecycle_callbacks} "
                            f"dump_model_freq={dump_model_freq}"
                        )
                        emit_nav_event(
                            "platform_dump_boundary",
                            platform_lifecycle_callbacks=platform_lifecycle_callbacks,
                            dump_model_freq=dump_model_freq,
                        )

            completed_iteration = iteration_index + 1
            algorithm.current_iteration = completed_iteration
            lr_scheduler.step()
            algorithm.lr_scheduler_state = lr_scheduler.state_dict()
            algorithm.assert_high_level_parameters_finite()

            iteration_metrics = _mean_metrics(step_rows)
            iteration_metrics.update(_aggregate_goal_metrics(step_rows))
            # 全帧聚合指标覆盖（soft-stay 消费全覆盖信号，非 tick 帧欠采样版）
            iteration_metrics.update(
                _episode_outcome_rates(
                    hard_events, timeout_events, completed_episodes
                )
            )
            # Retain the per-frame exposure denominator as an explicitly named
            # diagnostic. Soft-stay never consumes these hazard values.
            iteration_metrics["hard_termination_per_frame"] = hard_events / max(
                1, total_env_frames
            )
            iteration_metrics["timeout_per_frame"] = timeout_events / max(
                1, total_env_frames
            )
            iteration_metrics["goal_progress_m_per_frame"] = progress_sum / max(
                1, progress_count
            )
            iteration_metrics.update(
                {
                    "platform_lifecycle_callbacks": platform_lifecycle_callbacks,
                    "platform_lifecycle_failures": platform_lifecycle_failures,
                    "total_low_level_steps": total_low_level_steps,
                    "total_env_frames": session_total_env_frames,
                    "callbacks_until_next_dump": _callbacks_until_next_dump(
                        platform_lifecycle_callbacks, dump_model_freq
                    ),
                }
            )
            iter_metric_history.append(iteration_metrics)
            emit_nav_event(
                "iteration",
                iteration=completed_iteration,
                ramp_probability=float(algorithm.ramp_probability),
                **iteration_metrics,
            )
            if len(iter_metric_history) > 2 * _SOFT_STAY_WINDOW:
                del iter_metric_history[: -2 * _SOFT_STAY_WINDOW]

            # ---- soft-stay：每 WINDOW 检查一次，两窗口对比 ----
            if len(iter_metric_history) >= 2 * _SOFT_STAY_WINDOW and (
                completed_iteration % _SOFT_STAY_WINDOW == 0
            ):
                w1 = _quality_window_metrics(
                    iter_metric_history[-2 * _SOFT_STAY_WINDOW : -_SOFT_STAY_WINDOW]
                )
                w2 = _quality_window_metrics(iter_metric_history[-_SOFT_STAY_WINDOW:])
                if not algorithm.soft_stay_frozen:
                    reason = _soft_stay_check(w2, prev_window or w1)
                    if reason is not None:
                        algorithm.soft_stay_frozen = True
                        algorithm.soft_stay_reason = reason
                        algorithm.training_status = "soft_stay_frozen"
                        logger.warning(
                            f"[NavDAgger] soft-stay FROZEN at ramp="
                            f"{algorithm.ramp_probability:.3f} (reason={reason})"
                        )
                else:
                    no_new_failure = _soft_stay_check(w2, w1) is None
                    if no_new_failure and _quality_absolutely_ok(w2):
                        algorithm.soft_stay_frozen = False
                        algorithm.soft_stay_reason = ""
                        algorithm.training_status = "running"
                        logger.info("[NavDAgger] soft-stay unfrozen; ramp resumes")
                prev_window = w2

            # ---- 日志 / monitor ----
            if completed_iteration % log_interval == 0 or iteration_index == start_iteration:
                m = iteration_metrics
                logger.info(
                    f"[NavDAgger] iter={completed_iteration} "
                    f"ramp={algorithm.ramp_probability:.3f}"
                    f"{' (FROZEN)' if algorithm.soft_stay_frozen else ''} "
                    f"req={requested_p:.3f} "
                    f"ce={m.get('ce_loss', float('nan')):.4f} "
                    f"top1={m.get('top1_accuracy', float('nan')):.3f} "
                    f"disagree={m.get('disagreement_rate', float('nan')):.3f} "
                    f"entropy={m.get('token_entropy', float('nan')):.3f} "
                    f"switch={m.get('switch_rate', float('nan')):.3f} "
                    f"hard={m.get('hard_termination_rate', float('nan')):.4f} "
                    f"timeout={m.get('timeout_rate', float('nan')):.4f} "
                    f"goal_valid={m.get('goal_valid_rate', float('nan')):.3f} "
                    f"goal_fresh={m.get('goal4_fresh_rate', float('nan')):.3f} "
                    f"progress={m.get('goal_progress_m_per_frame', float('nan')):.4f} "
                    f"nonfinite={algorithm.nonfinite_fallback_count} "
                    f"grad={m.get('grad_norm', float('nan')):.3f} "
                    f"lr={algorithm.optimizer.param_groups[0]['lr']:.2e} "
                    f"nav_ticks={algorithm.total_nav_ticks} "
                    f"lifecycle={platform_lifecycle_callbacks} "
                    f"lifecycle_fail={platform_lifecycle_failures} "
                    f"until_dump={m['callbacks_until_next_dump']} "
                    f"iter_time={time.monotonic() - iter_start:.2f}s"
                )
            if monitor is not None:
                try:
                    monitor.put_data(
                        {
                            os.getpid(): {
                                "iteration": completed_iteration,
                                "ramp_probability": algorithm.ramp_probability,
                                "soft_stay_frozen": int(algorithm.soft_stay_frozen),
                                "nonfinite_fallback": algorithm.nonfinite_fallback_count,
                                **iteration_metrics,
                            }
                        }
                    )
                except Exception as exc:
                    logger.warning(f"[NavDAgger] monitor.put_data failed: {exc}")

        algorithm.training_status = (
            "completed_with_warnings" if algorithm.soft_stay_frozen else "completed"
        )
        _save_final_checkpoint(agent, logger, reason="normal_completion")
        logger.info(
            f"[NavDAgger] done: status={algorithm.training_status} "
            f"iterations={algorithm.current_iteration}"
        )
    finally:
        pass  # env.close 由外层 workflow 的 finally 统一处理


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """Nav DAgger 主入口（含 SIGTERM 优雅保存）。"""
    agent = agents[0]
    logger.info(
        "[LifecycleProbe] nav_workflow wrapper_enter "
        f"pid={os.getpid()} is_nav_dagger={getattr(agent, 'is_nav_dagger', False)}"
    )
    assert getattr(agent, "is_nav_dagger", False), (
        "nav_dagger_workflow.workflow called but agent.is_nav_dagger is False; "
        "check Config.CURRENT and agent.__init__."
    )
    agent._nav_training_started = getattr(agent, "_nav_training_started", False)
    agent._nav_final_save_done = False
    agent._nav_final_save_reason = None

    previous = _install_sigterm_checkpoint_handler(logger)
    try:
        _workflow_impl(envs, agents, logger=logger, monitor=monitor, *args, **kwargs)
    except (KeyboardInterrupt, SystemExit):
        _save_final_checkpoint(agent, logger, reason="graceful_platform_exit")
        raise
    except Exception as exc:
        emit_nav_event(
            "fatal_error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        try:
            envs[0].close()
        except Exception as exc:
            logger.warning(f"[NavDAgger] env.close failed: {exc}")
        _restore_sigterm_handler(previous)
