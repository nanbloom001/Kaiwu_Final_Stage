#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
LBC 视觉蒸馏 workflow（阶段 4：线性 ramp DAgger）。

lbc_loco (stage 4):
    教师 = teacher_encoder(height_scan) + teacher_actor           (FROZEN)
    学生 = VisionEncoder (CNN + LSTM + head)                      (TRAIN)
    loss = 0.5*SmoothL1(latent) + 0.1*(1-cos) + 1.0*SmoothL1(action)

调度：学生驱动比例从 0 线性 ramp 到 100%（约 4.5h），无离散档位、无强制晋升。
      质量恶化时 soft-stay 冻结当前比例（不回退、不强制升档），恢复后继续 ramp。

单次 forward：动作选择和三路损失复用 prepare_vision_update 的同一缓存结果，
              避免学生驱动时 LSTM 对同一观测推进两次（P1 第 5 条）。
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict

import torch

from agent_ppo.conf.conf import Config


# soft-stay 触发阈值（每 _SOFT_STAY_WINDOW iterations 检查一次，连续两个窗口）
_SOFT_STAY_WINDOW = 50
_SOFT_STAY_NONFINITE = 0.0          # 任一 nonfinite 立即触发
_SOFT_STAY_EFFECTIVE_DELTA = 0.10   # effective < requested - 0.10 触发
_SOFT_STAY_NMSE_DELTA = 0.05        # normalized_action_mse 相对前窗口恶化超此值触发
_SOFT_STAY_HARD_DELTA = 0.02        # hard_termination 相对前窗口恶化超此值触发
# 解冻的绝对质量门槛（不只是"不再恶化"，必须质量绝对达标才继续 ramp）
_SOFT_STAY_UNFREEZE_MAX_NMSE = 0.15      # normalized_action_mse 必须低于此值
_SOFT_STAY_UNFREEZE_MAX_HARD = 0.05      # hard_termination_rate 必须低于此值
_SOFT_STAY_UNFREEZE_MIN_ACTION_COS = 0.90  # action_cosine 必须高于此值


def _mean_metrics(rows: list[dict]) -> dict:
    values: dict[str, list[float]] = defaultdict(list)
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
    """从 ramp_probability 反推已用 ramp 时长（续训恢复用）。

    ramp_clock 在冻结时暂停；续训时从保存的 ramp_probability 反推 clock，
    保证从断点的 ramp 比例继续，而不是按墙钟时间追赶。
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return ramp_end_h
    return ramp_start_h + p * (ramp_end_h - ramp_start_h)


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """LBC 视觉蒸馏主 workflow（线性 ramp DAgger）。"""
    agent = agents[0]
    env = envs[0]

    assert getattr(agent, "is_lbc", False), (
        "lbc_workflow.workflow called but agent.is_lbc is False; "
        "check Config.CURRENT and agent.__init__."
    )

    stage = agent.stage
    algorithm = agent.algorithm  # AlgorithmLBC

    usr_conf, usr_conf_file, is_eval, _stage = Config.load_conf(logger)
    section = stage.name  # "lbc_loco"
    lbc_conf = usr_conf.get(section, {}) if isinstance(usr_conf, dict) else {}

    max_iterations = int(lbc_conf.get("max_iterations", stage.max_iterations))
    log_interval = int(lbc_conf.get("log_interval", stage.log_interval))
    save_interval = int(lbc_conf.get("save_interval", stage.model_save_interval))
    num_steps_per_env = int(lbc_conf.get("num_steps_per_env", stage.num_steps_per_env))

    # 线性 ramp 配置（TOML 可配，默认 0.33h–4.83h = 4.5h ramp）
    ramp_start_h = float(lbc_conf.get("ramp_start_h", algorithm.ramp_start_h))
    ramp_end_h = float(lbc_conf.get("ramp_end_h", algorithm.ramp_end_h))
    if ramp_end_h <= ramp_start_h:
        raise ValueError(f"ramp_end_h must exceed ramp_start_h: {ramp_end_h}<={ramp_start_h}")

    # 安全阈值固定时机：ramp 达到约 10% 后用 P95 固定
    safety_threshold = float("inf")
    safety_min_threshold = float(lbc_conf.get("safety_min_threshold", 0.25))
    safety_p95_multiplier = float(lbc_conf.get("safety_p95_multiplier", 2.0))
    safety_fixed = False

    # LR override：toml 优先于 conf.py 的 stage.lr。
    override_lr = lbc_conf.get("learning_rate")
    if override_lr is not None:
        override_lr = float(override_lr)
        for pg in algorithm.optimizer.param_groups:
            pg["lr"] = override_lr
        if hasattr(algorithm, "learning_rate"):
            algorithm.learning_rate = override_lr
        logger.info(f"[LBC] toml override learning_rate: stage.lr={stage.lr} -> {override_lr}")
    lr_min = float(lbc_conf.get("lr_min", getattr(stage, "lr_min", 1e-5)))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        algorithm.optimizer, T_max=max_iterations, eta_min=lr_min,
    )

    logger.info(
        f"[LBC-Vision] Start linear-ramp DAgger: "
        f"max_iterations={max_iterations}, ramp={ramp_start_h}h->{ramp_end_h}h, "
        f"num_steps_per_env={num_steps_per_env}, save_interval={save_interval}, "
        f"init_lr={algorithm.optimizer.param_groups[0]['lr']:.2e}, "
        f"lr_scheduler=cosine(eta_min={lr_min:.0e})"
    )

    # env reset + 清零 LSTM hidden（每次启动都重置，不恢复旧 hidden）
    data = env.reset(usr_conf)
    if data is None:
        raise RuntimeError("[LBC] env.reset returned None, check env configuration.")
    obs, _critic_obs = data
    obs = torch.as_tensor(obs).to(agent.device)

    if hasattr(algorithm.vision_encoder, "reset_hidden_state"):
        algorithm.vision_encoder.reset_hidden_state(batch_size=agent.num_envs, device=agent.device)

    algorithm.train_mode()
    algorithm.training_status = "running"

    # soft-stay 用 per-iteration 聚合指标（不是 per-step），每 _SOFT_STAY_WINDOW
    # 个 iteration 形成一个窗口。避免 24 steps/iteration 下 100 条 step 记录
    # 只覆盖约 4 个 iteration 的失真。
    iter_metric_history: list[dict] = []
    prev_window_metrics: dict | None = None
    # ramp clock：冻结时暂停累加，恢复后继续，避免解冻后追赶墙钟时间突然跳高。
    # 续训时从 algorithm.ramp_probability 反推已用 ramp 时长，保证从断点继续。
    ramp_clock_h = _ramp_elapsed_from_probability(
        algorithm.ramp_probability, ramp_start_h, ramp_end_h
    )
    loop_start = time.time()
    iter_start_prev = loop_start
    last_save_iter = -1

    try:
        for iteration in range(max_iterations):
            algorithm.current_iteration = iteration
            iter_start = time.time()

            # 推进 ramp clock（只在未冻结时累加，冻结时暂停）
            if not algorithm.soft_stay_frozen:
                ramp_clock_h += (iter_start - iter_start_prev) / 3600.0
            iter_start_prev = iter_start

            # 计算 ramp probability
            requested_p = _ramp_probability(ramp_clock_h, ramp_start_h, ramp_end_h)
            if algorithm.soft_stay_frozen:
                # 冻结：保持冻结时的比例，不上升
                effective_p = algorithm.ramp_probability
            else:
                # 续训恢复后从断点继续，不回退
                effective_p = max(requested_p, algorithm.ramp_probability)
                algorithm.ramp_probability = effective_p

            # safety_threshold 在 ramp 达到约 10% 后用 P95 固定（只固定一次）
            step_rows: list[dict] = []
            for _step in range(num_steps_per_env):
                step_metrics = _vision_dagger_step(
                    env, agent, algorithm, obs,
                    probability=effective_p,
                    safety_threshold=safety_threshold,
                )
                step_rows.append(step_metrics)
                obs = step_metrics["_next_obs"]  # 内部传递

            if not safety_fixed and effective_p >= 0.10:
                recent_l2 = [
                    r["_action_l2_p95"] for r in step_rows
                    if math.isfinite(r.get("_action_l2_p95", float("nan")))
                ]
                if recent_l2:
                    p95 = sorted(recent_l2)[int(0.95 * len(recent_l2))]
                    safety_threshold = max(safety_min_threshold, safety_p95_multiplier * p95)
                    safety_fixed = True
                    logger.info(
                        f"[LBC-Vision] safety_threshold fixed at {safety_threshold:.4f} "
                        f"(p95={p95:.4f}, ramp_p={effective_p:.3f})"
                    )

            # LR scheduler 步进
            lr_scheduler.step()
            algorithm.assert_student_parameters_finite()

            # 收集本 iteration 的聚合指标（per-iteration，不是 per-step）
            iteration_metrics = _mean_metrics(step_rows)
            iter_metric_history.append(iteration_metrics)

            # soft-stay 检查（每 _SOFT_STAY_WINDOW iterations，两个连续窗口）
            if (iteration + 1) % _SOFT_STAY_WINDOW == 0:
                iter_metric_history = iter_metric_history[-(2 * _SOFT_STAY_WINDOW):]
                if len(iter_metric_history) >= 2 * _SOFT_STAY_WINDOW:
                    w1_rows = iter_metric_history[-(2 * _SOFT_STAY_WINDOW):-_SOFT_STAY_WINDOW]
                    w2_rows = iter_metric_history[-_SOFT_STAY_WINDOW:]
                    # _mean_metrics 接受 list[dict]；这里每个 w*_rows 是 list[dict]
                    # 但每个元素本身是 iteration_metrics（dict），需要对窗口内多个
                    # iteration 的 dict 再聚合一次。直接传 list[dict] 即可。
                    w1 = _mean_metrics(w1_rows)
                    w2 = _mean_metrics(w2_rows)
                    reasons = _soft_stay_check(w1, w2, prev_window_metrics, effective_p)
                    if reasons and not algorithm.soft_stay_frozen:
                        algorithm.soft_stay_frozen = True
                        algorithm.soft_stay_reason = "; ".join(reasons)
                        algorithm.training_status = "soft_stay_frozen"
                        logger.warning(
                            "[LBC-Vision] soft-stay FROZEN at ramp_p="
                            f"{effective_p:.3f}: {algorithm.soft_stay_reason}"
                        )
                    elif algorithm.soft_stay_frozen and not reasons and _quality_absolutely_ok(w2):
                        # 已冻结：连续两个窗口既无新故障、且质量绝对达标后才解冻。
                        # 不只是"不再恶化"——NMSE 在差水平稳定不能解冻。
                        algorithm.soft_stay_frozen = False
                        algorithm.soft_stay_reason = None
                        algorithm.training_status = "running"
                        logger.info(
                            "[LBC-Vision] soft-stay released, ramp resumes from "
                            f"{effective_p:.3f}"
                        )
                    prev_window_metrics = w2

            # 日志
            if (iteration + 1) % log_interval == 0 or iteration == 0:
                cur_lr = algorithm.optimizer.param_groups[0]["lr"]
                dt = time.time() - iter_start
                cos_lat = iteration_metrics.get("latent_cosine", 0.0)
                angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_lat))))
                logger.info(
                    f"[LBC-Vision] iter={iteration+1}/{max_iterations}  "
                    f"ramp_p={effective_p:.3f}{'(FROZEN)' if algorithm.soft_stay_frozen else ''}  "
                    f"req={requested_p:.3f}  "
                    f"angle={angle_deg:.2f}deg  cos_lat={cos_lat:.4f}  "
                    f"lat_mse={iteration_metrics.get('latent_mse', 0):.5f}  "
                    f"act_mse={iteration_metrics.get('action_mse', 0):.5f}  "
                    f"nmse={iteration_metrics.get('normalized_action_mse', 0):.5f}  "
                    f"takeover={iteration_metrics.get('safety_takeover_rate', 0):.3f}  "
                    f"eff_ratio={iteration_metrics.get('effective_student_ratio', 0):.3f}  "
                    f"nonfinite={iteration_metrics.get('nonfinite_rate', 0):.4f}  "
                    f"grad={iteration_metrics.get('grad_norm', 0):.3f}  lr={cur_lr:.2e}  "
                    f"total_steps={algorithm.total_steps}  iter_time={dt:.2f}s"
                )
                if monitor is not None:
                    try:
                        monitor.put_data({os.getpid(): {
                            "iteration": iteration + 1,
                            "ramp_probability": effective_p,
                            "requested_probability": requested_p,
                            "soft_stay_frozen": int(algorithm.soft_stay_frozen),
                            "angle_deg": angle_deg,
                            **iteration_metrics,
                        }})
                    except Exception as e:
                        logger.warning(f"[LBC-Vision] monitor.put_data failed: {e}")

            # 定时保存。不传 id，让平台框架注入当前模型池 ID（R2 验证过的做法）；
            # 用 iteration 作 id 会让平台模型池 ID / 文件名 / 任务页记录不一致。
            # 标签由 _current_ramp_label 按 ramp 进度自动决定，无需显式里程碑保存。
            if (iteration + 1) % save_interval == 0 and iteration != last_save_iter:
                agent.learn(list_sample_data=None)
                agent.save_model()
                last_save_iter = iteration

        # 最终保存。不传 id，平台注入。标签由 _current_ramp_label 自动决定
        # （完成 100% → visionfull；soft-stay 未恢复 → visionblocked）。
        algorithm.training_status = (
            "completed_with_warnings" if algorithm.soft_stay_frozen else "completed"
        )
        agent.save_model()
        total_time = time.time() - loop_start
        logger.info(
            f"[LBC-Vision] Training finished in {total_time:.1f}s, "
            f"total steps={algorithm.total_steps}, "
            f"final ramp_p={algorithm.ramp_probability:.3f}, "
            f"status={algorithm.training_status}"
        )
    finally:
        env.close()


def _soft_stay_check(
    w1: dict, w2: dict, prev: dict | None, requested_p: float
) -> list[str]:
    """检查最近两个窗口是否触发 soft-stay 冻结。返回原因列表（空=正常）。"""
    reasons: list[str] = []
    for name, win in (("w1", w1), ("w2", w2)):
        nonfinite = win.get("nonfinite_rate", 0.0)
        if nonfinite > _SOFT_STAY_NONFINITE:
            reasons.append(f"{name} nonfinite_rate={nonfinite:.4f}>0")
        eff = win.get("effective_student_ratio", requested_p)
        if requested_p > 0 and eff < requested_p - _SOFT_STAY_EFFECTIVE_DELTA:
            reasons.append(
                f"{name} effective_ratio={eff:.3f}<{requested_p - _SOFT_STAY_EFFECTIVE_DELTA:.3f}"
            )
    # 相对前一窗口恶化（只在 w2 上判）
    if prev is not None:
        nmse_prev = prev.get("normalized_action_mse")
        nmse_now = w2.get("normalized_action_mse")
        if nmse_prev is not None and nmse_now is not None:
            if nmse_now - nmse_prev > _SOFT_STAY_NMSE_DELTA:
                reasons.append(f"nmse worsened {nmse_prev:.4f}->{nmse_now:.4f}")
        hard_prev = prev.get("hard_termination_rate")
        hard_now = w2.get("hard_termination_rate")
        if hard_prev is not None and hard_now is not None:
            if hard_now - hard_prev > _SOFT_STAY_HARD_DELTA:
                reasons.append(f"hard_term worsened {hard_prev:.4f}->{hard_now:.4f}")
    return reasons


def _quality_absolutely_ok(w: dict) -> bool:
    """解冻的绝对质量门槛：不只是"不再恶化"，必须质量绝对达标。

    避免 NMSE 在较差水平稳定下来后被自动解冻（"不再恶化"≠"质量够好"）。
    """
    nmse = w.get("normalized_action_mse")
    hard = w.get("hard_termination_rate")
    cos = w.get("action_cosine")
    if nmse is None or hard is None or cos is None:
        return False
    return (
        nmse <= _SOFT_STAY_UNFREEZE_MAX_NMSE
        and hard <= _SOFT_STAY_UNFREEZE_MAX_HARD
        and cos >= _SOFT_STAY_UNFREEZE_MIN_ACTION_COS
    )


def _vision_dagger_step(
    env, agent, algorithm, obs, probability: float, safety_threshold: float
) -> dict:
    """视觉 DAgger 单步：单次 forward → 选驱动 → env.step → 加权三路损失更新。"""
    device = agent.device

    # 1. 单次 forward 缓存（teacher + student latent/action）
    batch = algorithm.prepare_vision_update(obs)

    # 2. 逐环境选驱动 + 安全接管
    actions, selection = algorithm.select_driver_actions(
        batch, probability, safety_threshold
    )
    actions_clipped = torch.clip(actions, -6.0, 6.0).to(device)

    # 3. env.step（用选定的驱动动作）
    step_data = env.step(actions_clipped)
    if step_data is None:
        raise RuntimeError("[LBC] env.step returned None")
    frame_no, next_obs, rewards, terminated, truncated, (infos, privileged_obs) = step_data
    next_obs = torch.as_tensor(next_obs).to(device)

    # 4. LSTM 隐状态管理（done 的环境清零）
    dones = torch.logical_or(
        torch.as_tensor(terminated).to(device),
        torch.as_tensor(truncated).to(device),
    )
    if dones.any():
        algorithm.reset_student_hidden_states(dones)

    # 5. 样本权重（计划 §10.2）：正常 1.0，安全接管 0.25，非有限/硬终止 0
    weights = torch.ones(obs.shape[0], dtype=torch.float32, device=device)
    weights = torch.where(
        selection["safety_takeover"],
        torch.full_like(weights, 0.25),
        weights,
    )
    hard_term = torch.as_tensor(terminated, device=device).reshape(-1).bool()
    if isinstance(infos, dict):
        timeouts = torch.as_tensor(
            infos.get("time_outs", truncated), device=device
        ).reshape(-1).bool()
    else:
        timeouts = torch.as_tensor(truncated, device=device).reshape(-1).bool()
    hard_failure = hard_term & ~timeouts
    invalid = (~batch["student_finite"]) | hard_failure
    weights = torch.where(invalid, torch.zeros_like(weights), weights)

    # 6. 加权三路损失更新
    update_metrics = algorithm.finish_vision_update(batch, weights)

    # 7. 平台 lifecycle callback（no-op，推进 train_global_step / 模型池 ID）
    agent.learn(list_sample_data=None)

    # 8. 收集诊断指标
    requested_n = int(selection["requested_student"].sum().item())
    takeover_n = int(selection["safety_takeover"].sum().item())
    per_l2 = selection["per_sample_l2"]
    p95_val = (
        float(torch.quantile(per_l2.float().cpu(), 0.95).item())
        if per_l2.numel() else 0.0
    )

    update_metrics.update({
        "_next_obs": next_obs,
        "_action_l2_p95": p95_val,
        "requested_student_ratio": float(selection["requested_student"].float().mean().item()),
        "effective_student_ratio": float(selection["effective_student"].float().mean().item()),
        "safety_takeover_rate": (takeover_n / requested_n) if requested_n else 0.0,
        "hard_termination_rate": float(hard_failure.float().mean().item()),
    })
    return update_metrics
