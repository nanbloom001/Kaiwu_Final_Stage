#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""STD-BRIDGE-R1: guarded one-run behavior distillation workflow."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

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


def _validate_schedule(
    phase_end_iterations: list[int],
    phase_ratios: list[float],
    max_iterations: int,
) -> None:
    if not phase_end_iterations or len(phase_end_iterations) != len(phase_ratios):
        raise ValueError("DAgger phase endpoints and ratios must be non-empty and aligned")
    if any(end <= 0 for end in phase_end_iterations):
        raise ValueError("DAgger phase endpoints must be positive")
    if phase_end_iterations != sorted(set(phase_end_iterations)):
        raise ValueError("DAgger phase endpoints must be strictly increasing")
    if phase_end_iterations[-1] != max_iterations:
        raise ValueError(
            "Final DAgger phase endpoint must equal max_iterations: "
            f"{phase_end_iterations[-1]} != {max_iterations}"
        )
    if any(ratio < 0.0 or ratio > 1.0 for ratio in phase_ratios):
        raise ValueError("DAgger student-drive ratios must stay in [0, 1]")


def _phase_for_iteration(iteration: int, phase_end_iterations: list[int]) -> int:
    for phase_index, phase_end in enumerate(phase_end_iterations):
        if iteration < phase_end:
            return phase_index
    return len(phase_end_iterations) - 1


def _platform_model_id(current_iteration: int, platform_model_id_base: int) -> int:
    """Map the true R1 iteration to a monotonic platform liveness ID."""
    iteration = int(current_iteration)
    base = int(platform_model_id_base)
    if iteration <= 0:
        raise ValueError(
            f"Platform checkpoint mapping requires iteration > 0, got {iteration}"
        )
    if base < 0:
        raise ValueError(
            f"platform_model_id_base must be non-negative, got {base}"
        )
    return base + iteration


def _unpack_step_result(step_data, device):
    """Normalize Tencent wrapper 6/7-item step payloads."""
    if step_data is None:
        raise RuntimeError("[BehaviorDistill] env.step returned None")
    if not isinstance(step_data, (tuple, list)):
        raise TypeError(
            f"Unexpected env.step return type: {type(step_data).__name__}"
        )
    if len(step_data) == 6:
        frame_no, next_obs, rewards, terminated, truncated, extra = step_data
        if isinstance(extra, (tuple, list)):
            if len(extra) < 2:
                raise ValueError(f"Unexpected env.step extra length: {len(extra)}")
            infos, privileged_obs = extra[0], extra[1]
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

    if infos is None or not isinstance(infos, dict):
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


def _command_ood_mask(obs: torch.Tensor, command_guard: dict[str, Any]) -> torch.Tensor:
    if obs.ndim != 2 or obs.shape[-1] < 9:
        raise ValueError(
            "Command guard requires proprio command at obs[:, 6:9], "
            f"got shape={tuple(obs.shape)}"
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
        lower, upper = float(bound[0]), float(bound[1])
        in_domain &= (obs[:, index] >= lower - tolerance) & (
            obs[:, index] <= upper + tolerance
        )
    return ~in_domain


def _mean_metrics(step_metrics: list[dict[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for metrics in step_metrics:
        for key, value in metrics.items():
            values[key].append(float(value))
    return {
        key: sum(samples) / max(1, len(samples))
        for key, samples in values.items()
    }


def _window_average(rows: list[dict[str, float]]) -> dict[str, float]:
    return _mean_metrics(rows)


def _evaluate_phase_gate(
    recent_metrics: list[dict[str, float]],
    gate_window_iterations: int,
    gate_conf: dict[str, Any],
    previous_hard_termination_rate: float | None,
) -> tuple[bool, dict[str, Any]]:
    required = 2 * gate_window_iterations
    if len(recent_metrics) < required:
        return False, {
            "reasons": [
                f"need {required} gate iterations, only have {len(recent_metrics)}"
            ],
            "windows": [],
        }
    selected = recent_metrics[-required:]
    windows = [
        _window_average(selected[:gate_window_iterations]),
        _window_average(selected[gate_window_iterations:]),
    ]
    minimum_cosine = float(gate_conf.get("minimum_action_cosine", 0.98))
    maximum_normalized_mse = float(
        gate_conf.get("maximum_normalized_action_mse", 0.10)
    )
    maximum_takeover = float(gate_conf.get("maximum_safety_takeover_rate", 0.10))
    maximum_hard_delta = float(
        gate_conf.get("maximum_hard_termination_delta", 0.01)
    )
    reasons: list[str] = []
    for index, window in enumerate(windows, start=1):
        if window.get("action_cos", float("-inf")) < minimum_cosine:
            reasons.append(
                f"window{index} action_cos={window.get('action_cos')} < {minimum_cosine}"
            )
        if window.get("normalized_action_mse", float("inf")) > maximum_normalized_mse:
            reasons.append(
                "window"
                f"{index} normalized_action_mse={window.get('normalized_action_mse')} "
                f"> {maximum_normalized_mse}"
            )
        if window.get("nonfinite_rate", 1.0) > 0.0:
            reasons.append(
                f"window{index} nonfinite_rate={window.get('nonfinite_rate')} > 0"
            )
        if window.get("teacher_ood_rate", 1.0) > 0.0:
            reasons.append(
                f"window{index} teacher_ood_rate={window.get('teacher_ood_rate')} > 0"
            )
        if window.get("safety_takeover_rate", 1.0) > maximum_takeover:
            reasons.append(
                f"window{index} safety_takeover_rate="
                f"{window.get('safety_takeover_rate')} > {maximum_takeover}"
            )
        if previous_hard_termination_rate is not None:
            hard_rate = window.get("hard_termination_rate", 1.0)
            allowed = previous_hard_termination_rate + maximum_hard_delta
            if hard_rate > allowed:
                reasons.append(
                    f"window{index} hard_termination_rate={hard_rate} > {allowed}"
                )
    return not reasons, {
        "reasons": reasons,
        "windows": windows,
        "hard_termination_rate": sum(
            window.get("hard_termination_rate", 0.0) for window in windows
        )
        / 2.0,
        "action_l2_p95": sum(
            window.get("action_l2_p95", 0.0) for window in windows
        )
        / 2.0,
    }


def _validate_resume_contract(algorithm, current_config_sha: str) -> None:
    """Reject config drift and blocked-checkpoint auto-progression."""
    if (
        algorithm.current_iteration > 0
        and algorithm.config_sha256 not in {"", "unknown", current_config_sha}
    ):
        raise RuntimeError(
            "Refusing to resume STD-BRIDGE-R1 with a changed config: "
            f"checkpoint={algorithm.config_sha256}, current={current_config_sha}"
        )
    if algorithm.training_status == "blocked":
        raise RuntimeError(
            "Refusing to auto-resume a blocked STD-BRIDGE-R1 checkpoint. "
            "Review the failed gate and start an explicitly approved recovery run; "
            "the next DAgger ratio will not be entered automatically."
        )


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """Run flat Standard 10288 -> modular privileged teacher distillation."""
    agent = agents[0]
    env = envs[0]
    if not getattr(agent, "is_behavior_distill", False):
        raise RuntimeError(
            "behavior_distill_workflow called while agent.is_behavior_distill is false"
        )

    stage = agent.stage
    algorithm = agent.algorithm
    usr_conf, usr_conf_file, _is_eval, _stage = Config.load_conf(logger)
    distill_conf = usr_conf.get(stage.name, {}) if isinstance(usr_conf, dict) else {}
    max_iterations = int(distill_conf.get("max_iterations", stage.max_iterations))
    log_interval = int(distill_conf.get("log_interval", stage.log_interval))
    save_interval = int(distill_conf.get("save_interval", stage.model_save_interval))
    if save_interval <= 0:
        raise ValueError("save_interval must be positive")
    platform_model_id_base = int(
        distill_conf.get("platform_model_id_base", 10288)
    )
    initial_probe_save_iteration = int(
        distill_conf.get("initial_probe_save_iteration", 1)
    )
    if not 1 <= initial_probe_save_iteration <= max_iterations:
        raise ValueError(
            "initial_probe_save_iteration must stay within the R1 run: "
            f"{initial_probe_save_iteration} not in [1, {max_iterations}]"
        )
    num_steps_per_env = int(
        distill_conf.get("num_steps_per_env", stage.num_steps_per_env)
    )
    phase_end_iterations = [
        int(value)
        for value in distill_conf.get(
            "phase_end_iterations", [1500, 2250, 3000, 3750, 5000]
        )
    ]
    phase_ratios = [
        float(value)
        for value in distill_conf.get(
            "student_drive_ratios", [0.0, 0.25, 0.50, 0.75, 1.0]
        )
    ]
    _validate_schedule(phase_end_iterations, phase_ratios, max_iterations)
    gate_window_iterations = int(distill_conf.get("gate_window_iterations", 50))
    if gate_window_iterations <= 0:
        raise ValueError("gate_window_iterations must be positive")
    gate_conf = distill_conf.get("gates", {})
    safety_conf = distill_conf.get("safety", {})
    command_guard = distill_conf.get("command_guard", {})
    minimum_safety_threshold = float(
        safety_conf.get("minimum_action_l2_threshold", 0.25)
    )
    threshold_multiplier = float(safety_conf.get("p95_multiplier", 2.0))

    override_lr = distill_conf.get("learning_rate")
    if override_lr is not None:
        override_lr = float(override_lr)
        for param_group in algorithm.optimizer.param_groups:
            param_group["lr"] = override_lr
        algorithm.learning_rate = override_lr

    # Platform-selected checkpoints can be injected after Agent construction.
    if not algorithm.teacher_loaded:
        logger.info("[BehaviorDistill] loading selected teacher/resume checkpoint")
        agent.load_model(id="latest")
    algorithm.assert_teacher_ready()

    current_config_sha = _sha256_file(usr_conf_file)
    _validate_resume_contract(algorithm, current_config_sha)
    algorithm.set_run_metadata(current_config_sha, _code_commit())

    start_iteration = int(algorithm.current_iteration)
    if start_iteration < 0 or start_iteration >= max_iterations:
        raise ValueError(
            f"Invalid resume iteration {start_iteration} for max={max_iterations}"
        )
    expected_phase = _phase_for_iteration(start_iteration, phase_end_iterations)
    if algorithm.dagger_phase_index not in {expected_phase, 0} and start_iteration > 0:
        raise RuntimeError(
            "Checkpoint DAgger phase is inconsistent with iteration: "
            f"phase={algorithm.dagger_phase_index}, expected={expected_phase}, "
            f"iteration={start_iteration}"
        )
    algorithm.dagger_phase_index = expected_phase
    algorithm.training_status = "running"
    max_recent = 2 * gate_window_iterations
    algorithm.recent_iteration_metrics = algorithm.recent_iteration_metrics[-max_recent:]

    logger.info(
        "[BehaviorDistill] STD-BRIDGE-R1 start: "
        f"iteration={start_iteration}/{max_iterations}, steps_per_env={num_steps_per_env}, "
        f"phase_ends={phase_end_iterations}, ratios={phase_ratios}, "
        f"platform_model_id_base={platform_model_id_base}, "
        f"initial_probe_save_iteration={initial_probe_save_iteration}, "
        f"teacher={algorithm.teacher_source}, teacher_sha256={algorithm.teacher_sha256}, "
        f"config_sha256={current_config_sha}, code_commit={algorithm.code_commit}"
    )

    data = env.reset(usr_conf)
    if data is None or not isinstance(data, (tuple, list)) or len(data) < 1:
        raise RuntimeError("[BehaviorDistill] env.reset returned an invalid payload")
    obs = torch.as_tensor(data[0], device=agent.device)
    agent.model.train()

    loop_start = time.time()
    blocked = False
    last_saved_iteration: int | None = None

    def save_r1_checkpoint(
        current_iteration: int,
        *,
        checkpoint_label: str | None = None,
        publish_privileged_teacher: bool = False,
        release_status: str | None = None,
    ) -> int:
        platform_id = _platform_model_id(
            current_iteration, platform_model_id_base
        )
        agent.save_model(
            id=str(platform_id),
            checkpoint_label=checkpoint_label,
            publish_privileged_teacher=publish_privileged_teacher,
            release_status=release_status,
        )
        logger.info(
            "[BehaviorDistill] checkpoint mapping: "
            f"current_iteration={current_iteration} -> platform_model_id="
            f"{platform_id}, label={checkpoint_label or 'periodic'}, "
            f"publish_teacher={publish_privileged_teacher}"
        )
        return platform_id

    try:
        for iteration in range(start_iteration, max_iterations):
            iter_start = time.time()
            phase_index = _phase_for_iteration(iteration, phase_end_iterations)
            phase_start = 0 if phase_index == 0 else phase_end_iterations[phase_index - 1]
            student_probability = phase_ratios[phase_index]
            algorithm.dagger_phase_index = phase_index
            algorithm.dagger_phase_iteration = iteration - phase_start
            algorithm.student_drive_probability = student_probability
            step_metric_rows: list[dict[str, float]] = []

            for _step in range(num_steps_per_env):
                teacher_ood = _command_ood_mask(obs, command_guard)
                teacher_ood_rate = float(teacher_ood.float().mean().item())
                if bool(teacher_ood.any().item()):
                    offending = obs[teacher_ood, 6:9][:5].detach().cpu().tolist()
                    raise RuntimeError(
                        "Teacher command-domain violation. Refusing to label OOD "
                        f"commands; examples={offending}, guard={command_guard}"
                    )

                batch = algorithm.prepare_update(obs)
                actions, selection = algorithm.select_driver_actions(
                    batch,
                    student_drive_probability=student_probability,
                    safety_threshold=algorithm.safety_threshold,
                )
                step_data = env.step(torch.clip(actions, -6.0, 6.0).to(agent.device))
                (
                    _frame_no,
                    next_obs,
                    _rewards,
                    terminated,
                    truncated,
                    infos,
                    _privileged_obs,
                ) = _unpack_step_result(step_data, agent.device)

                timeouts = infos.get("time_outs", truncated)
                timeouts = torch.as_tensor(
                    timeouts, device=agent.device
                ).reshape(-1).bool()
                hard_failure = terminated & ~timeouts
                sample_weights = torch.ones(
                    obs.shape[0], device=agent.device, dtype=torch.float32
                )
                sample_weights = torch.where(
                    selection["safety_takeover"],
                    torch.full_like(sample_weights, 0.25),
                    sample_weights,
                )
                sample_weights = torch.where(
                    ~batch["student_finite"] | hard_failure,
                    torch.zeros_like(sample_weights),
                    sample_weights,
                )
                metrics = algorithm.finish_update(batch, sample_weights)

                requested_count = int(selection["requested_student"].sum().item())
                takeover_count = int(selection["safety_takeover"].sum().item())
                metrics.update(
                    {
                        "student_drive_probability": student_probability,
                        "requested_student_drive_ratio": float(
                            selection["requested_student"].float().mean().item()
                        ),
                        "effective_student_drive_ratio": float(
                            selection["effective_student"].float().mean().item()
                        ),
                        "safety_takeover_rate": (
                            takeover_count / requested_count
                            if requested_count > 0
                            else 0.0
                        ),
                        "teacher_ood_rate": teacher_ood_rate,
                        "hard_termination_rate": float(
                            hard_failure.float().mean().item()
                        ),
                        "terminated_rate": float(terminated.float().mean().item()),
                        "truncated_rate": float(truncated.float().mean().item()),
                    }
                )
                step_metric_rows.append(metrics)
                obs = next_obs

            iteration_metrics = _mean_metrics(step_metric_rows)
            iteration_metrics["teacher_max_abs_diff"] = (
                algorithm.teacher_max_abs_diff()
            )
            algorithm.recent_iteration_metrics.append(iteration_metrics)
            algorithm.recent_iteration_metrics = algorithm.recent_iteration_metrics[
                -max_recent:
            ]
            iter_id = iteration + 1
            algorithm.current_iteration = iter_id
            algorithm.dagger_phase_iteration = iter_id - phase_start

            # The platform probe needs a student artifact immediately.  Its
            # numeric filename ID must already outrank the source teacher 10288.
            if (
                iter_id == initial_probe_save_iteration
                and last_saved_iteration != iter_id
            ):
                save_r1_checkpoint(iter_id)
                last_saved_iteration = iter_id

            if iter_id % log_interval == 0 or iteration == start_iteration:
                logger.info(
                    "[BehaviorDistill] "
                    f"iter={iter_id}/{max_iterations} phase={phase_index} "
                    f"p_student={student_probability:.2f} "
                    f"effective={iteration_metrics.get('effective_student_drive_ratio', 0):.3f} "
                    f"takeover={iteration_metrics.get('safety_takeover_rate', 0):.3f} "
                    f"mse={iteration_metrics.get('action_mse', 0):.6f} "
                    f"nmse={iteration_metrics.get('normalized_action_mse', 0):.6f} "
                    f"cos={iteration_metrics.get('action_cos', 0):.4f} "
                    f"hard_term={iteration_metrics.get('hard_termination_rate', 0):.4f} "
                    f"teacher_diff={iteration_metrics['teacher_max_abs_diff']:.3e} "
                    f"env_steps={algorithm.total_steps} "
                    f"iter_time={time.time() - iter_start:.2f}s"
                )
                if monitor is not None:
                    try:
                        payload = {
                            "iteration": iter_id,
                            "total_steps": algorithm.total_steps,
                            "dagger_phase": phase_index,
                            **iteration_metrics,
                        }
                        monitor.put_data({os.getpid(): payload})
                    except Exception as exc:
                        logger.warning(
                            f"[BehaviorDistill] monitor.put_data failed: {exc}"
                        )

            phase_end = phase_end_iterations[phase_index]
            if iter_id == phase_end:
                gate_passed, gate_details = _evaluate_phase_gate(
                    algorithm.recent_iteration_metrics,
                    gate_window_iterations,
                    gate_conf,
                    algorithm.previous_hard_termination_rate,
                )
                teacher_unchanged = algorithm.teacher_max_abs_diff() == 0.0
                if not teacher_unchanged:
                    gate_passed = False
                    gate_details.setdefault("reasons", []).append(
                        "frozen teacher parameters changed"
                    )
                gate_record = {
                    "phase_index": phase_index,
                    "phase_end_iteration": iter_id,
                    "platform_model_id": _platform_model_id(
                        iter_id, platform_model_id_base
                    ),
                    "student_drive_probability": student_probability,
                    "passed": gate_passed,
                    **gate_details,
                }
                algorithm.gate_history.append(gate_record)
                if not gate_passed:
                    algorithm.training_status = "blocked"
                    blocked = True
                    logger.error(
                        "[BehaviorDistill] phase gate blocked progression: "
                        f"{gate_record}"
                    )
                    save_r1_checkpoint(
                        iter_id,
                        checkpoint_label="blocked",
                        publish_privileged_teacher=False,
                        release_status="blocked",
                    )
                    last_saved_iteration = iter_id
                    break

                algorithm.previous_hard_termination_rate = float(
                    gate_details["hard_termination_rate"]
                )
                if phase_index == 0:
                    algorithm.safety_threshold = max(
                        minimum_safety_threshold,
                        threshold_multiplier * float(gate_details["action_l2_p95"]),
                    )
                    logger.info(
                        "[BehaviorDistill] calibrated safety threshold="
                        f"{algorithm.safety_threshold:.6f} from phase-0 P95"
                    )
                if phase_index + 1 < len(phase_ratios):
                    algorithm.dagger_phase_index = phase_index + 1
                    algorithm.dagger_phase_iteration = 0
                else:
                    algorithm.training_status = "completed"
                logger.info(f"[BehaviorDistill] phase gate passed: {gate_record}")
                release_status = (
                    "completed"
                    if algorithm.training_status == "completed"
                    else "phase_passed"
                )
                save_r1_checkpoint(
                    iter_id,
                    checkpoint_label="bridge",
                    publish_privileged_teacher=True,
                    release_status=release_status,
                )
                last_saved_iteration = iter_id

            if (
                iter_id % save_interval == 0
                and not blocked
                and last_saved_iteration != iter_id
            ):
                save_r1_checkpoint(iter_id)
                last_saved_iteration = iter_id

        if not blocked:
            algorithm.training_status = "completed"
            if last_saved_iteration != max_iterations:
                save_r1_checkpoint(
                    max_iterations,
                    checkpoint_label="bridge",
                    publish_privileged_teacher=True,
                    release_status="completed",
                )
            logger.info(
                "[BehaviorDistill] completed in "
                f"{time.time() - loop_start:.1f}s, env_steps={algorithm.total_steps}, "
                f"teacher_max_abs_diff={algorithm.teacher_max_abs_diff():.3e}"
            )
        else:
            logger.warning(
                "[BehaviorDistill] stopped at a blocked phase gate; no later "
                "student-drive ratio was entered."
            )
    finally:
        env.close()
