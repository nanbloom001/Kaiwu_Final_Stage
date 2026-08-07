#!/usr/bin/env python3
"""Rollout metric reduction for the shared navigation PPO runtime."""

import torch

from agent_ppo.workflow.nav_ppo_support import (
    _curriculum_metrics,
    _event_share,
    _p4_conditional_metrics,
    _terminal_outcome_count,
)


def _tensor_total(values, name, device) -> float:
    value = values.get(name, torch.zeros((), device=device))
    return float(value.detach().cpu())


def _add_diagnostic_metrics(metrics: dict[str, float], state: dict) -> None:
    runtime = state["runtime"]
    rollout = state["rollout"]
    device = runtime.agent.device
    metrics.update(runtime.algorithm.memory_metrics())
    metrics.update(runtime.collision_traces.drain(runtime.logger))
    if rollout.diagnostic_count:
        success = _tensor_total(rollout.diagnostic_sums, "success_rate", device)
        failure = _tensor_total(rollout.diagnostic_sums, "failure_rate", device)
        timeout = _tensor_total(rollout.diagnostic_sums, "timeout_rate", device)
        wall_reset = _tensor_total(
            rollout.diagnostic_sums, "wall_stuck_reset_rate", device
        )
        terminal_count = _terminal_outcome_count(success, failure, timeout)
        metrics.update(
            {
                "rollout_success_count": success,
                "rollout_failure_count": failure,
                "rollout_timeout_count": timeout,
                "rollout_wall_stuck_reset_count": wall_reset,
                "rollout_terminal_count": terminal_count,
                "episode_success_fraction": success / max(terminal_count, 1.0),
                "episode_timeout_fraction": timeout / max(terminal_count, 1.0),
                "rollout_wall_stuck_saved_seconds": _tensor_total(
                    rollout.diagnostic_sums, "wall_stuck_saved_seconds", device
                ),
            }
        )
        metrics.update(
            {
                name: float(value.detach().cpu()) / rollout.diagnostic_count
                for name, value in rollout.diagnostic_sums.items()
            }
        )
        runtime.hooks.extend_diagnostic_metrics(metrics, state)
        eligible = _tensor_total(
            rollout.diagnostic_sums, "safe_alternative_available", device
        )
        selected = _tensor_total(
            rollout.diagnostic_sums, "selected_safest_direction", device
        )
        metrics["selected_safest_direction_rate"] = selected / max(eligible, 1.0)
        for name, value in rollout.diagnostic_maxima.items():
            current = float(value.detach().cpu())
            if name in rollout.cumulative_counter_names:
                previous = runtime.cumulative_counter_previous.get(name, 0.0)
                metrics[name] = max(0.0, current - previous)
                metrics[f"{name}_lifetime"] = current
                runtime.cumulative_counter_previous[name] = current
            else:
                metrics[name] = current
    valid_count = float(rollout.diagnostic_valid_count.detach().cpu())
    if valid_count > 0.0:
        metrics.update(
            {
                name: float(value.detach().cpu()) / valid_count
                for name, value in rollout.diagnostic_valid_sums.items()
            }
        )


def _add_quantile_metrics(metrics: dict[str, float], state: dict) -> None:
    for name, buffer in state["rollout"].quantile_samples.items():
        sample = buffer.reshape(-1)
        sample = sample[torch.isfinite(sample)]
        if not sample.numel():
            continue
        for label, quantile in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
            metrics[f"{name}_{label}"] = float(
                torch.quantile(sample.float(), quantile)
            )


def _add_command_bin_metrics(metrics: dict[str, float], state: dict) -> None:
    rollout = state["rollout"]
    buffers = rollout.command_bins
    sources = (
        ("progress", "progress"),
        ("tracking_mae", "tracking"),
        ("target_vx", "target_vx"),
        ("exec_vx", "exec_vx"),
        ("true_vx", "true_vx"),
        ("target_abs_wz", "target_wz"),
        ("exec_abs_wz", "exec_wz"),
        ("true_abs_wz", "true_wz"),
        ("gait_penalty", "gait"),
        ("success", "success"),
        ("failure", "failure"),
        ("timeout", "timeout"),
        ("vy_nonzero_share", "vy_nonzero"),
        ("target_abs_vy", "abs_vy"),
        ("vy_outer_share", "vy_outer"),
    )
    for vx_index in range(5):
        for wz_index in range(4):
            count = float(buffers["counts"][vx_index, wz_index])
            prefix = f"cmd_v{vx_index}_w{wz_index}"
            metrics[f"{prefix}_count"] = count
            metrics[f"{prefix}_share"] = count / max(1.0, rollout.diagnostic_count)
            for suffix, key in sources:
                denominator = max(1.0, count)
                if suffix == "tracking_mae":
                    denominator = max(
                        1.0,
                        float(buffers["tracking_counts"][vx_index, wz_index]),
                    )
                metrics[f"{prefix}_{suffix}"] = (
                    float(buffers[key][vx_index, wz_index]) / denominator
                )


def _add_segment_metrics(metrics: dict[str, float], state: dict) -> None:
    runtime = state["runtime"]
    rollout = state["rollout"]
    rows = rollout.segment_rows
    sources = (
        ("progress_mps", "progress"),
        ("target_vx", "target_vx"),
        ("exec_vx", "exec_vx"),
        ("true_vx", "true_vx"),
        ("target_abs_wz", "target_wz"),
        ("stop_ratio", "stop"),
        ("creep_ratio", "creep"),
        ("spin_ratio", "spin"),
        ("gait_penalty", "gait"),
        ("success", "success"),
        ("failure", "failure"),
        ("timeout", "timeout"),
        ("reason4", "reason4"),
        ("predictive_clearance_m", "predictive_clearance"),
        ("predictive_risk", "predictive_risk"),
        ("predictive_penalty", "predictive_penalty"),
        ("teacher_risk", "teacher_risk"),
        ("teacher_high_risk_rate", "teacher_high_risk"),
    )
    for index, label in enumerate(runtime.segment_metric_labels):
        count = float(rows["counts"][index])
        denominator = max(1.0, count)
        metrics[f"{label}_sample_share"] = count / max(
            1.0, float(rollout.segment_diagnostic_count)
        )
        for suffix, key in sources:
            metrics[f"{label}_{suffix}"] = float(rows[key][index]) / denominator


def _add_runtime_metrics(metrics: dict[str, float], state: dict) -> None:
    runtime = state["runtime"]
    rollout = state["rollout"]
    now = state["now"]
    algorithm = runtime.algorithm
    metrics.update(
        {
            "effective_training_seconds": algorithm.effective_training_seconds,
            "session_effective_seconds": algorithm.session_effective_seconds,
            "lifetime_effective_seconds": algorithm.lifetime_effective_seconds,
            "lifecycle_success": float(runtime.lifecycle_success),
            "platform_lifecycle_callbacks": float(runtime.lifecycle_success),
            "lifecycle_failures": float(runtime.lifecycle_failures),
            "rollout_time_s": now - rollout.rollout_started,
            "env_step_time_s": rollout.env_step_time_s,
            "samples_per_s": (
                runtime.nav_rollout_ticks * runtime.agent.num_envs
                / max(now - rollout.rollout_started, 1.0e-6)
            ),
            "cnn_unfrozen": float(algorithm.cnn_unfrozen),
        }
    )
    metrics["episode_starts_per_hour"] = (
        float(metrics.get("rollout_terminal_count", 0.0))
        * 3600.0
        / max(now - rollout.rollout_started, 1.0e-6)
    )
    if runtime.monitor_health is not None:
        metrics.update(runtime.monitor_health.observe_producer(metrics, now=now))


def build_nav_rollout_metrics(metrics: dict[str, float], state: dict) -> dict[str, float]:
    """Compose common metrics, then delegate variant-only series to hooks."""
    _add_diagnostic_metrics(metrics, state)
    metrics.update(_curriculum_metrics(state["curriculum_snapshot"]))
    _add_quantile_metrics(metrics, state)
    _add_command_bin_metrics(metrics, state)
    _add_segment_metrics(metrics, state)
    state["runtime"].hooks.extend_rollout_metrics(metrics, state)
    _add_runtime_metrics(metrics, state)
    return metrics


def add_p4_diagnostic_metrics(metrics: dict[str, float], state: dict) -> None:
    runtime = state["runtime"]
    metrics.update(
        _p4_conditional_metrics(
            state["rollout"].diagnostic_sums,
            runtime.agent.device,
        )
    )


def add_p4_spawn_metrics(metrics: dict[str, float], state: dict) -> None:
    runtime = state["runtime"]
    variant = state["rollout"].variant
    reset_count = float(variant.spawn_reset_event_count)
    segment_count = float(
        variant.spawn_safe_point_event_count
        + variant.spawn_hard_position_event_count
    )
    metrics.update(
        {
            "spawn_reset_event_count": reset_count,
            "spawn_full_start_event_share": _event_share(
                variant.spawn_full_start_event_count, reset_count
            ),
            "spawn_segment_start_event_share": _event_share(segment_count, reset_count),
            "spawn_safe_point_event_share": _event_share(
                variant.spawn_safe_point_event_count, segment_count
            ),
            "spawn_hard_position_event_share": _event_share(
                variant.spawn_hard_position_event_count, segment_count
            ),
            "episode_start_count": float(variant.episode_start_counts.sum()),
        }
    )
    for index in range(4):
        metrics[f"spawn_position_q{index + 1}_event_share"] = _event_share(
            variant.spawn_quartile_event_counts[index], segment_count
        )
    for axis in ("vy", "wz"):
        for sign in ("positive", "negative"):
            denominator = max(1.0, float(variant.signed_chain_counts[axis][sign]))
            for stage in ("policy_target", "limited_target", "mapped_cmd", "exec", "true"):
                metrics[f"{stage}_{axis}_{sign}_mean"] = (
                    float(variant.signed_chain_sums[axis][sign][stage]) / denominator
                )
    for index, label in enumerate(runtime.segment_metric_labels):
        metrics[f"spawn_segment_{label}_event_share"] = _event_share(
            variant.episode_start_counts[index], reset_count
        )
        metrics[f"episode_start_{label}_count"] = float(
            variant.episode_start_counts[index]
        )
        for outcomes, prefix in (
            (variant.spawn_outcomes, "spawn_segment"),
            (variant.current_outcomes, "current_segment"),
        ):
            total = max(1.0, float(outcomes[index].sum()))
            for outcome_index, suffix in enumerate(
                ("success", "failure", "timeout", "reason4")
            ):
                metrics[f"{prefix}_{label}_{suffix}_rate"] = (
                    float(outcomes[index, outcome_index]) / total
                )


def add_p4_recovery_metrics(metrics: dict[str, float], state: dict) -> None:
    runtime = state["runtime"]
    rollout_variant = state["rollout"].variant
    variant = runtime.variant_state
    wall_seconds = float(runtime.algorithm.session_wall_seconds)
    while variant.recovery_event_times and variant.recovery_event_times[0] < wall_seconds - 60.0:
        variant.recovery_event_times.popleft()
    metrics.update(
        {
            "session_wall_seconds": wall_seconds,
            "near_goal_capture_candidate_count": float(rollout_variant.capture_candidate_count),
            "near_goal_capture_entry_count": float(rollout_variant.capture_entry_count),
            "near_goal_capture_exit_count": float(rollout_variant.capture_exit_count),
            "near_goal_capture_zone_success_count": float(rollout_variant.capture_zone_success_count),
            "near_goal_capture_zone_collision_count": float(rollout_variant.capture_zone_collision_count),
            "near_goal_capture_zone_timeout_count": float(rollout_variant.capture_zone_timeout_count),
            "near_goal_capture_zone_reset_count": float(rollout_variant.capture_zone_reset_count),
            "near_goal_capture_reset_counted_as_completion_error": float(
                rollout_variant.capture_reset_completion_error_count
            ),
            "near_goal_capture_entry_to_platform_success_latency_s": sum(
                rollout_variant.capture_success_latency_s
            ) / max(len(rollout_variant.capture_success_latency_s), 1),
            "recovery_event_count_60s": float(len(variant.recovery_event_times)),
            "recovery_event_lifetime_count": float(variant.recovery_event_lifetime_count),
            "recovery_candidate_entry_count": float(rollout_variant.recovery_candidate_entry_count),
            "recovery_success_count": float(rollout_variant.recovery_success_count),
            "recovery_terminal_count": float(rollout_variant.recovery_terminal_count),
            "recovery_unverified_exit_count": float(rollout_variant.recovery_unverified_exit_count),
            "recovery_success_rate": float(rollout_variant.recovery_success_count)
            / max(
                rollout_variant.recovery_success_count
                + rollout_variant.recovery_terminal_count,
                1,
            ),
            "recovery_time_s": sum(rollout_variant.recovery_time_s)
            / max(len(rollout_variant.recovery_time_s), 1),
            "recovery_early_stuck_sample_share": float(
                rollout_variant.recovery_early_sample_count
            ) / max(runtime.nav_rollout_ticks * runtime.agent.num_envs, 1),
            "recovery_confirmed_stuck_sample_share": float(
                rollout_variant.recovery_confirmed_sample_count
            ) / max(runtime.nav_rollout_ticks * runtime.agent.num_envs, 1),
            "recovery_safe_exit_share": float(rollout_variant.recovery_success_count)
            / max(rollout_variant.recovery_candidate_entry_count, 1),
            "recovery_candidate_lifetime_count": float(
                variant.recovery_candidate_lifetime_count
            ),
            "recovery_terminal_lifetime_count": float(
                variant.recovery_terminal_lifetime_count
            ),
        }
    )
    runtime.algorithm._p4_recovery_monitor_state = {
        "event_times": list(variant.recovery_event_times),
        "success_lifetime_count": variant.recovery_event_lifetime_count,
        "candidate_lifetime_count": variant.recovery_candidate_lifetime_count,
        "terminal_lifetime_count": variant.recovery_terminal_lifetime_count,
    }
