#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Adaptive three-hour DAgger workflow for the Standard 10288 bridge."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from agent_ppo.checkpoint_io import DAGGER_PHASE_LABELS
from agent_ppo.conf.conf import Config


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_commit() -> str:
    configured = os.environ.get("KAIWU_CODE_COMMIT", "").strip()
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            value = float(value)
            if math.isfinite(value):
                values[key].append(value)
    return {
        key: sum(samples) / len(samples)
        for key, samples in values.items()
        if samples
    }


def _unpack_step_result(step_data, device):
    """Normalize Tencent wrapper 6/7-item step payloads."""
    if step_data is None:
        raise RuntimeError("[StandardDAgger] env.step returned None")
    if not isinstance(step_data, (tuple, list)):
        raise TypeError(
            f"Unexpected env.step return type: {type(step_data).__name__}"
        )
    if len(step_data) == 6:
        frame_no, next_obs, rewards, terminated, truncated, extra = step_data
        if isinstance(extra, (tuple, list)) and len(extra) >= 2:
            infos, privileged_obs = extra[:2]
        elif isinstance(extra, dict):
            infos = extra
            privileged_obs = extra.get("privileged_obs", extra.get("critic_obs"))
        else:
            raise TypeError(
                f"Unexpected env.step extra type: {type(extra).__name__}"
            )
    elif len(step_data) >= 7:
        frame_no, next_obs, rewards, terminated, truncated = step_data[:5]
        infos_or_extra = step_data[5]
        if isinstance(infos_or_extra, (tuple, list)) and len(infos_or_extra) >= 2:
            infos, privileged_obs = infos_or_extra[:2]
        else:
            infos = infos_or_extra
            privileged_obs = step_data[6]
    else:
        raise ValueError(f"Unexpected env.step return length: {len(step_data)}")
    if not isinstance(infos, dict):
        infos = {}
    return (
        frame_no,
        torch.as_tensor(next_obs, device=device),
        torch.as_tensor(rewards, device=device).reshape(-1),
        torch.as_tensor(terminated, device=device).reshape(-1).bool(),
        torch.as_tensor(truncated, device=device).reshape(-1).bool(),
        infos,
        privileged_obs,
    )


def _command_ood_mask(
    obs: torch.Tensor, command_guard: dict[str, Any]
) -> torch.Tensor:
    if obs.ndim != 2 or obs.shape[-1] < 9:
        raise ValueError(
            "Command guard requires commands at obs[:,6:9], "
            f"got {tuple(obs.shape)}"
        )
    bounds = (
        command_guard.get("lin_vel_x", [0.3, 1.3]),
        command_guard.get("lin_vel_y", [-0.2, 0.2]),
        command_guard.get("ang_vel_z", [-0.3, 0.3]),
    )
    tolerance = float(command_guard.get("tolerance", 1.0e-5))
    in_domain = torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
    for index, bound in zip((6, 7, 8), bounds):
        if not isinstance(bound, (tuple, list)) or len(bound) != 2:
            raise ValueError(f"Invalid command guard bound: {bound!r}")
        low, high = float(bound[0]), float(bound[1])
        in_domain &= (obs[:, index] >= low - tolerance) & (
            obs[:, index] <= high + tolerance
        )
    return ~in_domain


def _validate_schedule(
    labels: list[str],
    ratios: list[float],
    minimums: list[int],
    maximums: list[int],
    max_iterations: int,
    window_size: int,
) -> None:
    expected = list(DAGGER_PHASE_LABELS)
    if labels != expected:
        raise ValueError(f"phase_labels must be exactly {expected}, got {labels}")
    if not (len(labels) == len(ratios) == len(minimums) == len(maximums)):
        raise ValueError("DAgger phase arrays must have the same length")
    if ratios != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError(f"Unexpected DAgger ratios: {ratios}")
    if any(low <= 0 or high < low for low, high in zip(minimums, maximums)):
        raise ValueError("Each phase requires 0 < minimum <= maximum")
    if any(
        value % window_size != 0
        for value in [*minimums[:-1], *maximums[:-1]]
    ):
        raise ValueError(
            "Pre-full phase minimums/maximums must align with quality windows"
        )
    if sum(maximums[:-1]) + minimums[-1] > max_iterations:
        raise ValueError(
            "Schedule cannot guarantee the minimum daggerfull budget: "
            f"{sum(maximums[:-1])}+{minimums[-1]}>{max_iterations}"
        )


def _quality_result(
    recent: list[dict[str, float]],
    window_size: int,
    thresholds: dict[str, Any],
    previous_hard_rate: float | None,
) -> tuple[bool, dict[str, Any]]:
    required = 2 * window_size
    if len(recent) < required:
        return False, {
            "reasons": [f"need {required} iterations, have {len(recent)}"],
            "windows": [],
        }
    windows = [
        _mean_metrics(recent[-required:-window_size]),
        _mean_metrics(recent[-window_size:]),
    ]
    minimum_cosine = float(thresholds.get("minimum_action_cosine", 0.98))
    maximum_mse = float(
        thresholds.get("maximum_normalized_action_mse", 0.10)
    )
    maximum_takeover = float(
        thresholds.get("maximum_safety_takeover_rate", 0.10)
    )
    maximum_hard_delta = float(
        thresholds.get("maximum_hard_termination_delta", 0.01)
    )
    reasons: list[str] = []
    required_metrics = (
        "action_cos",
        "normalized_action_mse",
        "nonfinite_rate",
        "teacher_ood_rate",
        "safety_takeover_rate",
        "hard_termination_rate",
        "action_l2_p95",
    )
    for number, window in enumerate(windows, start=1):
        for name in required_metrics:
            value = window.get(name)
            if value is None or not math.isfinite(float(value)):
                reasons.append(f"window{number} {name} is not finite: {value}")
        if window.get("action_cos", float("-inf")) < minimum_cosine:
            reasons.append(f"window{number} action_cos below {minimum_cosine}")
        if window.get("normalized_action_mse", float("inf")) > maximum_mse:
            reasons.append(f"window{number} normalized_action_mse above {maximum_mse}")
        if window.get("nonfinite_rate", 1.0) != 0.0:
            reasons.append(f"window{number} nonfinite_rate is not zero")
        if window.get("teacher_ood_rate", 1.0) != 0.0:
            reasons.append(f"window{number} teacher_ood_rate is not zero")
        if window.get("safety_takeover_rate", 1.0) > maximum_takeover:
            reasons.append(f"window{number} safety_takeover_rate above {maximum_takeover}")
        if previous_hard_rate is not None:
            allowed = previous_hard_rate + maximum_hard_delta
            if window.get("hard_termination_rate", 1.0) > allowed:
                reasons.append(
                    f"window{number} hard_termination_rate above {allowed}"
                )
    summary = _mean_metrics(windows)
    summary["reasons"] = reasons
    summary["windows"] = windows
    return not reasons, summary


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """Run original flat 10288 teacher -> modular low-level DAgger."""
    agent = agents[0]
    env = envs[0]
    if not getattr(agent, "is_behavior_distill", False):
        raise RuntimeError("Standard DAgger workflow requires behavior_distill")

    stage = agent.stage
    algorithm = agent.algorithm
    usr_conf, usr_conf_file, _is_eval, _stage = Config.load_conf(logger)
    distill = usr_conf.get(stage.name, {}) if isinstance(usr_conf, dict) else {}
    max_iterations = int(distill.get("max_iterations", stage.max_iterations))
    num_steps = int(
        distill.get("num_steps_per_env", stage.num_steps_per_env)
    )
    log_interval = int(distill.get("log_interval", stage.log_interval))
    save_interval = int(
        distill.get("save_interval", stage.model_save_interval)
    )
    window_size = int(distill.get("quality_window_iterations", 50))
    labels = [str(value) for value in distill.get("phase_labels", DAGGER_PHASE_LABELS)]
    ratios = [
        float(value)
        for value in distill.get(
            "student_drive_ratios", [0.0, 0.25, 0.5, 0.75, 1.0]
        )
    ]
    minimums = [
        int(value)
        for value in distill.get(
            "phase_min_iterations", [900, 600, 600, 600, 1800]
        )
    ]
    maximums = [
        int(value)
        for value in distill.get(
            "phase_max_iterations", [1500, 900, 900, 900, 6000]
        )
    ]
    if save_interval <= 0 or window_size <= 0:
        raise ValueError("save_interval and quality_window_iterations must be positive")
    _validate_schedule(
        labels, ratios, minimums, maximums, max_iterations, window_size
    )

    quality_conf = distill.get("quality_thresholds", {})
    safety_conf = distill.get("safety", {})
    command_guard = distill.get("command_guard", {})
    minimum_threshold = float(
        safety_conf.get("minimum_action_l2_threshold", 0.25)
    )
    p95_multiplier = float(safety_conf.get("p95_multiplier", 2.0))

    learning_rate = float(distill.get("learning_rate", algorithm.learning_rate))
    algorithm.learning_rate = learning_rate
    for group in algorithm.optimizer.param_groups:
        group["lr"] = learning_rate

    if not algorithm.teacher_loaded:
        preload_dir = os.environ.get(
            "KAIWU_MODEL_CKPT_DIR", "/data/pre_model/ckpt"
        )
        logger.info(
            "[StandardDAgger] loading platform-selected original 10288 teacher "
            f"from {preload_dir}"
        )
        agent.load_model(path=preload_dir, id="10288")
    algorithm.assert_teacher_ready()
    algorithm.set_run_metadata(_sha256_file(usr_conf_file), _code_commit(), "10288")

    start_iteration = int(algorithm.current_iteration)
    if start_iteration < 0 or start_iteration >= max_iterations:
        raise ValueError(
            f"Invalid resume iteration {start_iteration}/{max_iterations}"
        )
    if not 0 <= algorithm.dagger_phase_index < len(labels):
        raise ValueError(
            f"Invalid resume phase index: {algorithm.dagger_phase_index}"
        )
    if start_iteration == 0:
        if (
            algorithm.dagger_phase_index != 0
            or algorithm.dagger_phase_iteration != 0
            or algorithm.promotion_history
            or not math.isclose(
                algorithm.student_drive_probability,
                ratios[0],
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError(
                "Iteration-zero checkpoint has non-initial DAgger state"
            )
    else:
        phase_index = algorithm.dagger_phase_index
        phase_iteration = algorithm.dagger_phase_iteration
        if algorithm.phase_entry_snapshot is None:
            raise ValueError("Resume checkpoint is missing phase entry snapshot")
        if phase_iteration < 0:
            raise ValueError(
                f"Resume phase iteration is negative: {phase_iteration}"
            )
        if phase_index < len(labels) - 1 and phase_iteration > maximums[phase_index]:
            raise ValueError(
                "Resume phase iteration exceeds phase maximum: "
                f"{phase_iteration}>{maximums[phase_index]}"
            )
        expected_probability = ratios[phase_index]
        if not math.isclose(
            algorithm.student_drive_probability,
            expected_probability,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "Resume student ratio does not match phase: "
                f"{algorithm.student_drive_probability}!={expected_probability}"
            )
        if phase_index == 0:
            expected_iteration = phase_iteration
            if algorithm.promotion_history:
                raise ValueError("Phase zero resume unexpectedly has promotion history")
        else:
            if not algorithm.promotion_history:
                raise ValueError("Promoted phase resume has no promotion history")
            last_promotion = algorithm.promotion_history[-1]
            if int(last_promotion.get("to_phase_index", -1)) != phase_index:
                raise ValueError(
                    "Resume promotion history does not end at current phase"
                )
            expected_iteration = (
                int(last_promotion.get("global_iteration", -1))
                + phase_iteration
            )
        if expected_iteration != start_iteration:
            raise ValueError(
                "Resume iteration/phase state mismatch: "
                f"global={start_iteration}, expected={expected_iteration}"
            )
    if start_iteration == 0 and algorithm.phase_entry_snapshot is None:
        algorithm.start_phase(0)
    warning_seen = (
        algorithm.training_status in {
            "running_with_warnings",
            "completed_with_warnings",
        }
        or any(
            record.get("promotion_mode") == "forced"
            for record in algorithm.promotion_history
        )
    )
    algorithm.training_status = (
        "running_with_warnings" if warning_seen else "running"
    )
    max_recent = 2 * window_size
    algorithm.recent_iteration_metrics = algorithm.recent_iteration_metrics[
        -max_recent:
    ]

    reset_data = env.reset(usr_conf)
    if not isinstance(reset_data, (tuple, list)) or not reset_data:
        raise RuntimeError("[StandardDAgger] env.reset returned invalid data")
    obs = torch.as_tensor(reset_data[0], device=agent.device)
    agent.model.train()

    logger.info(
        "[StandardDAgger] std-dagger-r2 start: "
        f"iteration={start_iteration}/{max_iterations}, phase="
        f"{algorithm.current_phase_label}, phase_iteration="
        f"{algorithm.dagger_phase_iteration}, ratios={ratios}, "
        f"minimums={minimums}, maximums={maximums}, "
        f"teacher={algorithm.teacher_source}, "
        f"teacher_sha256={algorithm.teacher_sha256}"
    )
    logger.warning(
        "[StandardDAgger] supervised updates happen inside this workflow; "
        "learner_proxy sample succ_cnt may remain zero. Confirm progress with "
        "StandardDAgger logs, train_global_step, and saved model files."
    )

    loop_start = time.time()
    last_saved_iteration: int | None = None
    try:
        for iteration in range(start_iteration, max_iterations):
            iter_start = time.time()
            phase_index = algorithm.dagger_phase_index
            executed_phase_label = labels[phase_index]
            probability = ratios[phase_index]
            algorithm.student_drive_probability = probability
            step_rows: list[dict[str, float]] = []

            for _step in range(num_steps):
                observation_finite = torch.isfinite(obs).all(dim=-1)
                teacher_ood = _command_ood_mask(obs, command_guard)
                safe_obs = torch.nan_to_num(
                    obs, nan=0.0, posinf=0.0, neginf=0.0
                )
                batch = algorithm.prepare_update(safe_obs)
                actions, selection = algorithm.select_driver_actions(
                    batch,
                    student_drive_probability=probability,
                    safety_threshold=algorithm.safety_threshold,
                )
                result = env.step(
                    torch.clip(actions, -6.0, 6.0).to(agent.device)
                )
                (
                    _frame_no,
                    next_obs,
                    _rewards,
                    terminated,
                    truncated,
                    infos,
                    _privileged_obs,
                ) = _unpack_step_result(result, agent.device)
                timeouts = torch.as_tensor(
                    infos.get("time_outs", truncated), device=agent.device
                ).reshape(-1).bool()
                hard_failure = terminated & ~timeouts

                weights = torch.ones(
                    obs.shape[0], dtype=torch.float32, device=agent.device
                )
                weights = torch.where(
                    selection["safety_takeover"],
                    torch.full_like(weights, 0.25),
                    weights,
                )
                invalid = (
                    ~observation_finite
                    | ~batch["student_finite"]
                    | teacher_ood
                    | hard_failure
                )
                weights = torch.where(
                    invalid, torch.zeros_like(weights), weights
                )
                metrics = algorithm.finish_update(batch, weights)

                requested_count = int(selection["requested_student"].sum().item())
                takeover_count = int(selection["safety_takeover"].sum().item())
                metrics.update(
                    {
                        "requested_student_ratio": float(
                            selection["requested_student"].float().mean().item()
                        ),
                        "effective_student_ratio": float(
                            selection["effective_student"].float().mean().item()
                        ),
                        "safety_takeover_rate": (
                            takeover_count / requested_count
                            if requested_count
                            else 0.0
                        ),
                        "teacher_ood_rate": float(
                            teacher_ood.float().mean().item()
                        ),
                        "hard_termination_rate": float(
                            hard_failure.float().mean().item()
                        ),
                        "nonfinite_rate": max(
                            float(metrics.get("nonfinite_rate", 0.0)),
                            float(
                                (~observation_finite).float().mean().item()
                            ),
                        ),
                        "terminated_rate": float(terminated.float().mean().item()),
                        "truncated_rate": float(truncated.float().mean().item()),
                    }
                )
                step_rows.append(metrics)
                obs = next_obs

            iteration_metrics = _mean_metrics(step_rows)
            algorithm.assert_student_parameters_finite()
            algorithm.recent_iteration_metrics.append(iteration_metrics)
            algorithm.recent_iteration_metrics = algorithm.recent_iteration_metrics[
                -max_recent:
            ]
            algorithm.current_iteration = iteration + 1
            algorithm.dagger_phase_iteration += 1

            # Preserve the verified Kaiwu lifecycle. The algorithm already updated
            # above; this no-op callback advances train_global_step/model-pool ID.
            agent.learn(list_sample_data=None)

            phase_boundary_saved = False
            local_iteration = algorithm.dagger_phase_iteration
            should_check = (
                local_iteration >= minimums[phase_index]
                and local_iteration % window_size == 0
            )
            quality_ok = False
            quality_summary: dict[str, Any] = {}
            if should_check:
                quality_ok, quality_summary = _quality_result(
                    algorithm.recent_iteration_metrics,
                    window_size,
                    quality_conf,
                    algorithm.previous_hard_termination_rate,
                )
                algorithm.consider_phase_best(quality_summary)

            forced = (
                phase_index < len(labels) - 1
                and local_iteration >= maximums[phase_index]
                and not quality_ok
            )
            if phase_index < len(labels) - 1 and (quality_ok or forced):
                if algorithm.teacher_max_abs_diff() != 0.0:
                    raise RuntimeError(
                        "Frozen 10288 teacher changed before phase promotion"
                    )
                restore_source = "current"
                promotion_mode = "normal" if quality_ok else "forced"
                if forced:
                    restore_source = algorithm.restore_phase_best_or_entry()
                    warning_seen = True
                if phase_index == 0:
                    p95 = float(quality_summary.get("action_l2_p95", 0.0))
                    algorithm.safety_threshold = max(
                        minimum_threshold, p95_multiplier * p95
                    )
                hard_rate = quality_summary.get("hard_termination_rate")
                if hard_rate is not None:
                    algorithm.previous_hard_termination_rate = float(hard_rate)
                algorithm.promotion_history.append(
                    {
                        "from_phase_index": phase_index,
                        "from_phase_label": labels[phase_index],
                        "to_phase_index": phase_index + 1,
                        "to_phase_label": labels[phase_index + 1],
                        "global_iteration": algorithm.current_iteration,
                        "phase_iteration": local_iteration,
                        "promotion_mode": promotion_mode,
                        "restored_from": restore_source,
                        "quality": quality_summary,
                    }
                )
                algorithm.capture_phase_exit()
                algorithm.training_status = (
                    "running_with_warnings" if warning_seen else "running"
                )
                algorithm.start_phase(phase_index + 1)
                algorithm.student_drive_probability = ratios[phase_index + 1]
                algorithm.recent_iteration_metrics = []
                # Save the resumable entry point of the newly active phase.
                agent.save_model()
                last_saved_iteration = algorithm.current_iteration
                phase_boundary_saved = True
                logger.info(
                    "[StandardDAgger] promoted "
                    f"{labels[phase_index]} -> {labels[phase_index + 1]}, "
                    f"mode={promotion_mode}, restored={restore_source}, "
                    f"safety_threshold={algorithm.safety_threshold:.6f}"
                )

            if (
                algorithm.current_iteration % save_interval == 0
                and not phase_boundary_saved
            ):
                agent.save_model()
                last_saved_iteration = algorithm.current_iteration

            if (
                algorithm.current_iteration % log_interval == 0
                or iteration == start_iteration
            ):
                logger.info(
                    "[StandardDAgger] "
                    f"iter={algorithm.current_iteration}/{max_iterations} "
                    f"executed_phase={executed_phase_label} "
                    f"executed_phase_iter={local_iteration} "
                    f"next_phase={algorithm.current_phase_label} "
                    f"requested={probability:.2f} "
                    f"effective={iteration_metrics.get('effective_student_ratio', 0):.3f} "
                    f"takeover={iteration_metrics.get('safety_takeover_rate', 0):.3f} "
                    f"mse={iteration_metrics.get('action_mse', 0):.6f} "
                    f"nmse={iteration_metrics.get('normalized_action_mse', 0):.6f} "
                    f"cos={iteration_metrics.get('action_cos', 0):.4f} "
                    f"hard_term={iteration_metrics.get('hard_termination_rate', 0):.4f} "
                    f"replay={int(iteration_metrics.get('replay_size', 0))} "
                    f"env_steps={algorithm.total_steps} "
                    f"iter_time={time.time() - iter_start:.2f}s"
                )
                if monitor is not None:
                    try:
                        monitor.put_data(
                            {
                                os.getpid(): {
                                    "iteration": algorithm.current_iteration,
                                    "total_steps": algorithm.total_steps,
                                    "dagger_phase": algorithm.dagger_phase_index,
                                    **iteration_metrics,
                                }
                            }
                        )
                    except Exception as exc:
                        logger.warning(
                            f"[StandardDAgger] monitor.put_data failed: {exc}"
                        )

        if algorithm.dagger_phase_index != len(labels) - 1:
            raise RuntimeError(
                "Run exhausted before entering daggerfull; schedule is inconsistent"
            )
        if algorithm.dagger_phase_iteration < minimums[-1]:
            raise RuntimeError(
                "daggerfull did not receive its minimum 1800-iteration budget"
            )
        if algorithm.teacher_max_abs_diff() != 0.0:
            raise RuntimeError("Frozen 10288 teacher changed during daggerfull")
        final_quality = _mean_metrics(algorithm.recent_iteration_metrics)
        algorithm.consider_phase_best(final_quality)
        final_key = algorithm._quality_key(final_quality)
        final_selection = "current"
        if (
            algorithm.phase_best_key is not None
            and (final_key is None or final_key > algorithm.phase_best_key)
        ):
            final_selection = algorithm.restore_phase_best_or_entry()
            warning_seen = True
        algorithm.final_selection = {
            "global_iteration": algorithm.current_iteration,
            "selected": final_selection,
            "final_quality": final_quality,
            "phase_best_metrics": algorithm.phase_best_metrics,
        }
        algorithm.training_status = (
            "completed_with_warnings" if warning_seen else "completed"
        )
        # Persist the final status and any best-point rollback even when iteration
        # 6000 was already a periodic save using the same framework-supplied ID.
        agent.save_model()
        logger.info(
            "[StandardDAgger] completed: "
            f"status={algorithm.training_status}, "
            f"iteration={algorithm.current_iteration}, "
            f"full_phase_iterations={algorithm.dagger_phase_iteration}, "
            f"teacher_diff={algorithm.teacher_max_abs_diff():.3e}, "
            f"elapsed={time.time() - loop_start:.1f}s"
        )
    finally:
        env.close()
