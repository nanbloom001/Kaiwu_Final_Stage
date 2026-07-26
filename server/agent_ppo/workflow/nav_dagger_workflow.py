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
from agent_ppo.feature import nav_contract


# soft-stay 阈值（每 _SOFT_STAY_WINDOW iterations 检查一次，连续两个窗口）
_SOFT_STAY_WINDOW = 50
_SOFT_STAY_DISAGREE_DELTA = 0.05    # disagreement 相对前窗口恶化超此值触发
_SOFT_STAY_HARD_DELTA = 0.02        # hard_termination 相对前窗口恶化超此值触发
# 解冻的绝对质量门槛（不只是"不再恶化"）
_SOFT_STAY_UNFREEZE_MIN_TOP1 = 0.85
_SOFT_STAY_UNFREEZE_MAX_HARD = 0.05


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
            window.get("hard_termination_rate", 0.0)
            > prev_window.get("hard_termination_rate", 0.0) + _SOFT_STAY_HARD_DELTA
        ):
            return "hard_termination_worsened"
    return None


def _quality_absolutely_ok(window: dict) -> bool:
    return (
        window.get("top1_accuracy", 0.0) >= _SOFT_STAY_UNFREEZE_MIN_TOP1
        and window.get("hard_termination_rate", 1.0) <= _SOFT_STAY_UNFREEZE_MAX_HARD
    )


def _workflow_impl(envs, agents, logger=None, monitor=None, *args, **kwargs):
    agent = agents[0]
    env = envs[0]
    stage = agent.stage
    algorithm = agent.algorithm  # AlgorithmNavDagger

    usr_conf, usr_conf_file, is_eval, _stage = Config.load_conf(logger)
    section = stage.name  # "nav_dagger"
    nav_conf = usr_conf.get(section, {}) if isinstance(usr_conf, dict) else {}

    # ---- 超参（TOML 优先、类默认兜底）----
    max_iterations = int(nav_conf.get("max_iterations", stage.max_iterations))
    log_interval = int(nav_conf.get("log_interval", stage.log_interval))
    save_interval = int(nav_conf.get("save_interval", stage.model_save_interval))
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

    # ---- 契约一致性启动断言（TOML 镜像 == nav_contract 权威值）----
    toml_period = int(nav_conf.get("nav_period_frames", nav_contract.NAV_PERIOD_FRAMES))
    toml_dwell = int(nav_conf.get("min_dwell_ticks", nav_contract.MIN_DWELL_TICKS))
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

    # 续训：从保存的 ramp 比例反推 clock（断点续接，不按墙钟追赶）
    if algorithm.resume_loaded:
        algorithm.ramp_clock_h = _ramp_elapsed_from_probability(
            algorithm.ramp_probability, ramp_start_h, ramp_end_h
        )

    logger.info(
        f"[NavDAgger] start: conf={usr_conf_file} max_iter={max_iterations} "
        f"steps/iter={num_steps_per_env} T={tbptt_T} save_interval={save_interval} "
        f"ramp=[{ramp_start_h},{ramp_end_h}]h resume={algorithm.resume_loaded} "
        f"ramp_p={algorithm.ramp_probability:.3f} "
        f"low_level_digest={algorithm.low_level_state_digest}"
    )

    # ---- env reset ----
    data = env.reset(usr_conf)
    if data is None:
        raise RuntimeError("env.reset returned None")
    obs, critic_obs = data
    obs = torch.as_tensor(obs).to(agent.device)
    critic_obs = torch.as_tensor(critic_obs).to(agent.device)
    logger.info(
        f"[NavDAgger] reset ok: obs={tuple(obs.shape)} critic={tuple(critic_obs.shape)}"
    )

    algorithm.train_mode()

    start_iteration = int(algorithm.current_iteration)
    last_save_iter = start_iteration
    iter_metric_history: list[dict] = []
    prev_window: dict | None = None
    iter_end_prev = time.monotonic()

    try:
        for iteration_index in range(start_iteration, max_iterations):
            iter_start = time.monotonic()

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
            for _ in range(num_steps_per_env):
                result = algorithm.frame_begin(obs, critic_obs)
                actions = torch.clip(result["actions"], -6.0, 6.0)

                step_data = env.step(actions)
                (
                    _frame_no,
                    next_obs,
                    _rewards,
                    terminated,
                    truncated,
                    infos,
                    privileged_obs,
                ) = _extract_step(step_data)

                next_obs = torch.as_tensor(next_obs).to(agent.device)
                privileged_obs = torch.as_tensor(privileged_obs).to(agent.device)
                terminated = torch.as_tensor(terminated).to(agent.device).bool()
                truncated = torch.as_tensor(truncated).to(agent.device).bool()
                dones = terminated | truncated
                if isinstance(infos, dict) and "time_outs" in infos:
                    time_outs = torch.as_tensor(infos["time_outs"]).to(agent.device).bool()
                else:
                    time_outs = truncated
                hard_termination = terminated & ~time_outs

                algorithm.frame_end(dones)

                if result["is_tick"]:
                    row = dict(result["tick_metrics"])
                    row["hard_termination_rate"] = float(
                        hard_termination.float().mean().item()
                    )
                    if row.pop("buffer_full", False):
                        update_metrics = algorithm.finish_nav_sequence_update()
                        row.update(update_metrics)
                        agent._nav_training_started = True
                    step_rows.append(row)

                obs = next_obs
                critic_obs = privileged_obs

            completed_iteration = iteration_index + 1
            algorithm.current_iteration = completed_iteration
            lr_scheduler.step()
            algorithm.lr_scheduler_state = lr_scheduler.state_dict()
            algorithm.assert_high_level_parameters_finite()

            # 平台 lifecycle 推进（每 outer iteration 恰一次；nav learn 为 no-op）
            agent.learn(list_sample_data=None)

            iteration_metrics = _mean_metrics(step_rows)
            iter_metric_history.append(iteration_metrics)

            # ---- soft-stay：每 WINDOW 检查一次，两窗口对比 ----
            if len(iter_metric_history) >= 2 * _SOFT_STAY_WINDOW and (
                completed_iteration % _SOFT_STAY_WINDOW == 0
            ):
                w1 = _mean_metrics(
                    iter_metric_history[-2 * _SOFT_STAY_WINDOW : -_SOFT_STAY_WINDOW]
                )
                w2 = _mean_metrics(iter_metric_history[-_SOFT_STAY_WINDOW:])
                if not algorithm.soft_stay_frozen:
                    reason = _soft_stay_check(w2, prev_window or w1)
                    if reason is not None:
                        algorithm.soft_stay_frozen = True
                        algorithm.soft_stay_reason = reason
                        logger.warning(
                            f"[NavDAgger] soft-stay FROZEN at ramp="
                            f"{algorithm.ramp_probability:.3f} (reason={reason})"
                        )
                else:
                    no_new_failure = _soft_stay_check(w2, w1) is None
                    if no_new_failure and _quality_absolutely_ok(w2):
                        algorithm.soft_stay_frozen = False
                        algorithm.soft_stay_reason = ""
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
                    f"nonfinite={algorithm.nonfinite_fallback_count} "
                    f"grad={m.get('grad_norm', float('nan')):.3f} "
                    f"lr={algorithm.optimizer.param_groups[0]['lr']:.2e} "
                    f"nav_ticks={algorithm.total_nav_ticks} "
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

            # ---- 按 iteration 保存（禁止用 iteration 当 id；id 由平台注入）----
            if (
                completed_iteration % save_interval == 0
                and completed_iteration != last_save_iter
            ):
                agent.save_model()
                last_save_iter = completed_iteration

        algorithm.training_status = (
            "completed_with_warnings" if algorithm.soft_stay_frozen else "completed"
        )
        agent.save_model()
        logger.info(
            f"[NavDAgger] done: status={algorithm.training_status} "
            f"iterations={algorithm.current_iteration}"
        )
    finally:
        pass  # env.close 由外层 workflow 的 finally 统一处理


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """Nav DAgger 主入口（含 SIGTERM 优雅保存）。"""
    agent = agents[0]
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
    finally:
        try:
            envs[0].close()
        except Exception as exc:
            logger.warning(f"[NavDAgger] env.close failed: {exc}")
        _restore_sigterm_handler(previous)
