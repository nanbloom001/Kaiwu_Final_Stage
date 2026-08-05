#!/usr/bin/env python3
"""Four-session-hour P2 Track semi-MDP PPO workflow."""

from __future__ import annotations

import math
import json
import os
import signal
import threading
import time
from collections import deque

import torch

from agent_ppo.checkpoint_io import CheckpointSaveError
from agent_ppo.conf.conf import Config
from agent_ppo.feature import nav_contract, p2_contract, p3_contract, p4_contract
from agent_ppo.feature.nav_event_log import emit_nav_event


def _p4_worker_push_resume_offset(algorithm) -> float:
    """Translate the restored learner clock to the worker's monotonic clock."""
    if str(getattr(algorithm, "maze_training_branch", "")) == "auto":
        diagnostic_elapsed = float(
            getattr(algorithm, "diagnostic_elapsed_seconds", 0.0)
        )
        if diagnostic_elapsed < p4_contract.DIAGNOSTIC_SECONDS:
            return diagnostic_elapsed - p4_contract.DIAGNOSTIC_SECONDS
    return float(getattr(algorithm, "session_effective_seconds", 0.0))


def _p4_worker_stuck_resume_offset(algorithm) -> float:
    """Resume worker wall-clock safety phases without mixing clock domains."""
    return float(
        getattr(
            algorithm,
            "session_wall_seconds",
            getattr(algorithm, "session_effective_seconds", 0.0),
        )
    )


def _goal_history_stuck(
    goal_history: deque, end_goal: torch.Tensor
) -> torch.Tensor:
    """Evaluate the configured full window at either 5 Hz or 10 Hz."""
    if goal_history.maxlen is None or len(goal_history) != goal_history.maxlen:
        return torch.zeros_like(end_goal, dtype=torch.bool)
    window = torch.stack(tuple(goal_history))
    progress = window[0] - window[-1]
    history_valid = window.isfinite().all(dim=0)
    return history_valid & (end_goal > 0.8) & (progress < 0.05)


def _event_share(numerator: float | torch.Tensor, denominator: float | torch.Tensor) -> float:
    """Return a truthful event ratio; an empty event window reports zero."""
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        return 0.0
    return numerator_value / denominator_value


def _p4_conditional_metrics(
    diagnostic_sums: dict[str, torch.Tensor], device: torch.device
) -> dict[str, float]:
    """Reduce P4 event counters using their actual eligible-event denominators."""

    def count(name: str) -> float:
        return float(
            diagnostic_sums.get(name, torch.zeros((), device=device)).detach().cpu()
        )

    head_samples = count("head_correct_samples")
    head_wrong = count("head_correct_actor_wrong_count")
    resolved = count("risk_event_resolved_count")
    policy_decel = count("risk_decel_policy_count")
    limited_decel = count("risk_decel_limited_count")
    no_decel = count("risk_no_deceleration_count")
    stuck_terminal = count("stuck_terminal_count")
    stuck_return_sum = count("stuck_terminal_episode_return_sum")
    stuck_nonnegative = count("stuck_terminal_nonnegative_count")
    side_candidates = count("legitimate_side_goal_candidate_count")
    side_selected = count("legitimate_side_goal_selected_count")
    return {
        "head_correct_samples": head_samples,
        "head_correct_actor_wrong_count": head_wrong,
        "head_correct_actor_wrong_rate": head_wrong / max(head_samples, 1.0),
        "risk_event_resolved_count": resolved,
        "risk_decel_policy_count": policy_decel,
        "risk_decel_limited_count": limited_decel,
        "risk_no_deceleration_count": no_decel,
        "risk_decel_policy_rate": policy_decel / max(resolved, 1.0),
        "risk_decel_limited_rate": limited_decel / max(resolved, 1.0),
        "risk_no_deceleration_rate": no_decel / max(resolved, 1.0),
        "stuck_terminal_count": stuck_terminal,
        "stuck_terminal_episode_return_mean": (
            stuck_return_sum / max(stuck_terminal, 1.0)
        ),
        "stuck_terminal_nonnegative_rate": (
            stuck_nonnegative / max(stuck_terminal, 1.0)
        ),
        "legitimate_side_goal_candidate_count": side_candidates,
        "legitimate_side_goal_selected_count": side_selected,
        "legitimate_side_goal_selection_rate": (
            side_selected / max(side_candidates, 1.0)
        ),
    }


def _p4_recovery_event_masks(
    active: torch.Tensor,
    awaiting_evidence: torch.Tensor,
    candidate: torch.Tensor,
    duration_s: torch.Tensor,
    transition_done: torch.Tensor,
    evidence_valid: torch.Tensor,
    escape_evidence: torch.Tensor,
    completion: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Classify P4 recovery state without treating reset as a recovery."""
    active = torch.as_tensor(active).bool().reshape(-1)
    awaiting_evidence = torch.as_tensor(
        awaiting_evidence, device=active.device
    ).bool().reshape(-1)
    candidate = torch.as_tensor(candidate, device=active.device).bool().reshape(-1)
    duration_s = torch.as_tensor(
        duration_s, device=active.device, dtype=torch.float32
    ).reshape(-1)
    transition_done = torch.as_tensor(
        transition_done, device=active.device
    ).bool().reshape(-1)
    evidence_valid = torch.as_tensor(
        evidence_valid, device=active.device
    ).bool().reshape(-1)
    escape_evidence = torch.as_tensor(
        escape_evidence, device=active.device
    ).bool().reshape(-1)
    completion = torch.as_tensor(
        completion, device=active.device
    ).bool().reshape(-1)
    if not (
        active.shape
        == awaiting_evidence.shape
        == candidate.shape
        == duration_s.shape
        == transition_done.shape
        == evidence_valid.shape
        == escape_evidence.shape
        == completion.shape
    ):
        raise ValueError("P4 recovery tensors must share the same flat shape")
    entry = candidate & ~active
    physical_exit = (
        active
        & ~candidate
        & ~transition_done
        & evidence_valid
        & escape_evidence
    )
    success = physical_exit | (active & transition_done & completion)
    terminal = active & transition_done & ~completion
    unverified_exit = (
        active
        & ~candidate
        & ~transition_done
        & ~physical_exit
        & ~awaiting_evidence
    )
    next_awaiting = (
        (awaiting_evidence | unverified_exit)
        & ~candidate
        & ~physical_exit
        & ~transition_done
    )
    return {
        "entry": entry,
        "success": success,
        "terminal": terminal,
        "unverified_exit": unverified_exit,
        "early": candidate & (duration_s >= 0.8) & (duration_s < 2.0),
        "confirmed": candidate & (duration_s >= 2.0),
        "next_active": (candidate | next_awaiting) & ~transition_done,
        "next_awaiting": next_awaiting,
    }


def _p4_reset_completion_mismatch(
    raw_wall_reset: torch.Tensor,
    completion: torch.Tensor,
) -> torch.Tensor:
    """Compare independent wall-reset evidence with scorer completion."""
    raw_wall_reset = torch.as_tensor(raw_wall_reset).bool().reshape(-1)
    completion = torch.as_tensor(
        completion, device=raw_wall_reset.device
    ).bool().reshape(-1)
    if raw_wall_reset.shape != completion.shape:
        raise ValueError("P4 reset/completion tensors must share shape")
    return raw_wall_reset & completion


class _CollisionTraceRecorder:
    """Bounded high-level event traces; aggregate every event, log few examples."""

    def __init__(
        self,
        *,
        nav_dt_s=p2_contract.NAV_DT_S,
        pre_ticks=5,
        post_ticks=10,
        max_raw_events=12,
    ):
        self.nav_dt_s = float(nav_dt_s)
        self.history = deque(maxlen=int(pre_ticks))
        self.post_ticks = int(post_ticks)
        self.max_raw_events = int(max_raw_events)
        self.pending = []
        self.completed = []
        self.total_onsets = 0
        self.rollout_onsets = 0
        self.rollout_observations = 0

    @staticmethod
    def _row(snapshot, env_id: int) -> dict[str, float | int]:
        return {
            name: int(value[env_id]) if name in {"segment", "column", "reason"}
            else float(value[env_id])
            for name, value in snapshot.items()
        }

    def observe(self, *, iteration: int, tick: int, onset: torch.Tensor, **values) -> None:
        snapshot = {
            name: torch.as_tensor(value).detach().reshape(-1).cpu()
            for name, value in values.items()
        }
        for event in self.pending:
            event["post"].append(self._row(snapshot, event["env_id"]))
        ready = [event for event in self.pending if len(event["post"]) >= self.post_ticks]
        self.pending = [event for event in self.pending if len(event["post"]) < self.post_ticks]
        self.completed.extend(ready)

        onset_ids = torch.as_tensor(onset).detach().reshape(-1).bool().nonzero(
            as_tuple=False
        ).reshape(-1).tolist()
        self.total_onsets += len(onset_ids)
        self.rollout_onsets += len(onset_ids)
        self.rollout_observations += int(torch.as_tensor(onset).numel())
        for env_id in onset_ids:
            if len(self.pending) + len(self.completed) >= self.max_raw_events:
                break
            self.pending.append(
                {
                    "iteration": int(iteration),
                    "tick": int(tick),
                    "env_id": int(env_id),
                    "pre": [self._row(old, env_id) for old in self.history],
                    "onset": self._row(snapshot, env_id),
                    "post": [],
                }
            )
        self.history.append(snapshot)

    def drain(self, logger=None) -> dict[str, float]:
        events = self.completed[: self.max_raw_events]
        self.completed.clear()
        if events and logger:
            logger.info(
                "[P2CollisionTrace] "
                + json.dumps(events, ensure_ascii=True, separators=(",", ":"))
            )
        zero_latencies = []
        reverse_latencies = []
        progress_latencies = []
        target_to_exec_wz_latencies = []
        exec_to_true_wz_latencies = []
        target_to_exec_vy_latencies = []
        exec_to_true_vy_latencies = []
        risk_lead_times = []
        terminal_overlaps = 0
        for event in events:
            onset_wz = float(event["onset"]["exec_wz"])
            onset_true_wz = float(event["onset"]["true_wz"])
            onset_vy = float(event["onset"]["exec_vy"])
            onset_true_vy = float(event["onset"]["true_vy"])
            for index, row in enumerate(event["post"], start=1):
                if not zero_latencies or abs(float(row["exec_wz"])) <= 0.05:
                    if abs(float(row["exec_wz"])) <= 0.05:
                        zero_latencies.append(index)
                        break
            if abs(onset_wz) > 0.05:
                target_reverse_index = None
                exec_reverse_index = None
                for index, row in enumerate(event["post"], start=1):
                    if target_reverse_index is None and float(row["target_wz"]) * onset_wz < 0.0:
                        target_reverse_index = index
                    if float(row["exec_wz"]) * onset_wz < 0.0:
                        reverse_latencies.append(index)
                        exec_reverse_index = index
                        break
                if target_reverse_index is not None and exec_reverse_index is not None:
                    target_to_exec_wz_latencies.append(
                        max(0, exec_reverse_index - target_reverse_index)
                    )
                if exec_reverse_index is not None and abs(onset_true_wz) > 0.05:
                    for index, row in enumerate(event["post"], start=1):
                        if index >= exec_reverse_index and float(row["true_wz"]) * onset_true_wz <= 0.0:
                            exec_to_true_wz_latencies.append(index - exec_reverse_index)
                            break
            if abs(onset_vy) > 0.05:
                target_reverse_index = None
                exec_reverse_index = None
                for index, row in enumerate(event["post"], start=1):
                    if target_reverse_index is None and float(row["target_vy"]) * onset_vy < 0.0:
                        target_reverse_index = index
                    if float(row["exec_vy"]) * onset_vy < 0.0:
                        exec_reverse_index = index
                        break
                if target_reverse_index is not None and exec_reverse_index is not None:
                    target_to_exec_vy_latencies.append(
                        max(0, exec_reverse_index - target_reverse_index)
                    )
                if exec_reverse_index is not None and abs(onset_true_vy) > 0.05:
                    for index, row in enumerate(event["post"], start=1):
                        if index >= exec_reverse_index and float(row["true_vy"]) * onset_true_vy <= 0.0:
                            exec_to_true_vy_latencies.append(index - exec_reverse_index)
                            break
            pre_risks = [float(row.get("risk", 0.0)) for row in event["pre"]]
            high_risk_indices = [index for index, value in enumerate(pre_risks) if value >= 0.5]
            if high_risk_indices:
                risk_lead_times.append(len(pre_risks) - high_risk_indices[0])
            for index, row in enumerate(event["post"], start=1):
                if float(row["progress"]) > 0.0:
                    progress_latencies.append(index)
                    break
            terminal_overlaps += int(
                any(int(row["reason"]) != 0 for row in event["post"])
            )
        count = max(len(events), 1)
        rollout_onsets = self.rollout_onsets
        rollout_observations = self.rollout_observations
        self.rollout_onsets = 0
        self.rollout_observations = 0
        return {
            "collision_trace_events": float(len(events)),
            "collision_onset_count": float(rollout_onsets),
            "collision_onset_rate": float(rollout_onsets) / max(rollout_observations, 1),
            "collision_onset_total": float(self.total_onsets),
            "collision_wz_zero_latency_s": (
                sum(zero_latencies) / len(zero_latencies) * self.nav_dt_s
                if zero_latencies else 0.0
            ),
            "collision_wz_reverse_latency_s": (
                sum(reverse_latencies) / len(reverse_latencies) * self.nav_dt_s
                if reverse_latencies else 0.0
            ),
            "collision_progress_recovery_latency_s": (
                sum(progress_latencies) / len(progress_latencies) * self.nav_dt_s
                if progress_latencies else 0.0
            ),
            "collision_target_to_exec_wz_reverse_latency_s": (
                sum(target_to_exec_wz_latencies) / len(target_to_exec_wz_latencies) * self.nav_dt_s
                if target_to_exec_wz_latencies else 0.0
            ),
            "collision_exec_to_true_wz_zero_latency_s": (
                sum(exec_to_true_wz_latencies) / len(exec_to_true_wz_latencies) * self.nav_dt_s
                if exec_to_true_wz_latencies else 0.0
            ),
            "collision_target_to_exec_vy_reverse_latency_s": (
                sum(target_to_exec_vy_latencies) / len(target_to_exec_vy_latencies) * self.nav_dt_s
                if target_to_exec_vy_latencies else 0.0
            ),
            "collision_exec_to_true_vy_zero_latency_s": (
                sum(exec_to_true_vy_latencies) / len(exec_to_true_vy_latencies) * self.nav_dt_s
                if exec_to_true_vy_latencies else 0.0
            ),
            "collision_risk_lead_time_s": (
                sum(risk_lead_times) / len(risk_lead_times) * self.nav_dt_s
                if risk_lead_times else 0.0
            ),
            "collision_terminal_overlap_rate": terminal_overlaps / count,
        }


def _extract_step(step_data):
    if step_data is None:
        raise RuntimeError("P2 env.step returned None")
    frame_no, next_obs, rewards, terminated, truncated, extra = step_data
    infos, privileged_obs = extra
    return frame_no, next_obs, rewards, terminated, truncated, infos, privileged_obs


def _expected_critic_wire_dim(*, is_p4: bool) -> int:
    return (
        p4_contract.P4_PRIVILEGED_WIRE_DIM
        if is_p4
        else p2_contract.PRIVILEGED_WIRE_DIM
    )


def _frame_done_masks(terminated, truncated, infos, device, *, worker_aux=None):
    terminated = torch.as_tensor(terminated, device=device).bool().reshape(-1)
    truncated = torch.as_tensor(truncated, device=device).bool().reshape(-1)
    if isinstance(infos, dict) and "time_outs" in infos:
        timeout = torch.as_tensor(infos["time_outs"], device=device).bool().reshape(-1)
    else:
        timeout = truncated
    if worker_aux is not None:
        worker_aux = torch.as_tensor(worker_aux, device=device)
        reset = worker_aux[:, 24] > 0.5
        reason = worker_aux[:, 25].round().long()
        worker_timeout = (reason == 3) | (reason == 4)
        worker_hard = (reason == 1) | (reason == 2)
        # Any reset not explained by a hard terminal is a timeout. This
        # recovers the platform wrapper's known truncated/time_outs erasure.
        timeout |= worker_timeout | (reset & ~worker_hard & ~terminated)
        terminated |= worker_hard
    done = terminated | truncated | timeout
    return done, timeout


def _resolve_terminal_outcome(
    new_done: torch.Tensor,
    frame_timeout: torch.Tensor,
    raw_reason: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return authoritative reason and mutually exclusive hard/timeout masks."""
    fallback_reason = torch.where(
        frame_timeout,
        torch.full_like(raw_reason, 3),
        torch.full_like(raw_reason, 2),
    )
    valid_reason = (raw_reason >= 1) & (raw_reason <= 4)
    reason = torch.where(valid_reason, raw_reason, fallback_reason)
    hard = new_done & ((reason == 1) | (reason == 2))
    # Reason 4 is represented as a truncation by the platform, but it is a
    # policy-caused terminal boundary and must not bootstrap like time limit 3.
    timeout = new_done & ((reason == 3) | (reason == 4))
    return reason, hard, timeout


def _terminal_safe_segment(
    live_segment: torch.Tensor,
    terminal_segment: torch.Tensor,
    transition_done: torch.Tensor,
) -> torch.Tensor:
    """Keep the old episode's segment after auto-reset advances a new episode."""
    return torch.where(
        transition_done & (terminal_segment >= 0.0),
        terminal_segment,
        live_segment,
    )


def _terminal_safe_tensor(
    live_value: torch.Tensor,
    terminal_value: torch.Tensor,
    done: torch.Tensor,
) -> torch.Tensor:
    """Select the frozen old-episode row after an in-window reset."""
    if live_value.shape != terminal_value.shape:
        raise ValueError("terminal-safe tensors must have matching shapes")
    if live_value.shape[0] != done.numel():
        raise ValueError("terminal-safe mask must match the tensor batch")
    mask = done.reshape((-1,) + (1,) * (live_value.ndim - 1))
    return torch.where(mask, terminal_value, live_value)


def _terminal_safe_p4_critic_wire(
    live_critic_wire: torch.Tensor,
    terminal_p4_extra: torch.Tensor,
    done: torch.Tensor,
) -> torch.Tensor:
    """Restore the old episode's P4-only tail after an automatic reset."""
    if live_critic_wire.ndim != 2 or live_critic_wire.shape[1] != p4_contract.P4_PRIVILEGED_WIRE_DIM:
        raise ValueError("P4 terminal-safe critic wire has an invalid shape")
    if terminal_p4_extra.shape != (
        live_critic_wire.shape[0],
        p4_contract.P4_WORKER_EXTRA_DIM,
    ):
        raise ValueError("P4 terminal-safe extra has an invalid shape")
    if done.numel() != live_critic_wire.shape[0]:
        raise ValueError("P4 terminal-safe mask must match the wire batch")
    result = live_critic_wire.clone()
    done = done.reshape(-1).bool()
    result[
        done,
        p3_contract.P3_PRIVILEGED_WIRE_DIM : p4_contract.P4_PRIVILEGED_WIRE_DIM,
    ] = terminal_p4_extra[done]
    return result


def _curriculum_metrics(snapshot: dict[str, object]) -> dict[str, float]:
    """Flatten the worker probe state into stable scalar monitor metrics."""
    if not isinstance(snapshot, dict):
        return {}
    row_moves = snapshot.get("row_moves", ())
    col_moves = snapshot.get("col_moves", ())
    outcomes = snapshot.get("outcomes", ())
    starts = snapshot.get("start_counts", ())
    try:
        outcome_totals = [
            sum(int(cell[index]) for row in outcomes for cell in row)
            for index in range(3)
        ]
        metrics = {
            "curriculum_row_demotions": float(row_moves[0]),
            "curriculum_row_unchanged": float(row_moves[1]),
            "curriculum_row_promotions": float(row_moves[2]),
            "curriculum_column_unchanged": float(col_moves[0]),
            "curriculum_column_changed": float(col_moves[1]),
            "curriculum_successes": float(outcome_totals[0]),
            "curriculum_failures": float(outcome_totals[1]),
            "curriculum_timeouts": float(outcome_totals[2]),
        }
        row_labels = (
            p4_contract.FULL_TRACK_SEGMENT_LABELS
            if len(starts) >= len(p4_contract.FULL_TRACK_SEGMENT_LABELS)
            else p2_contract.TRACK_SEGMENT_METRIC_LABELS
        )
        for index, label in enumerate(row_labels):
            if index >= len(starts):
                break
            count = float(sum(int(v) for v in starts[index]))
            metrics[f"curriculum_{label}_starts"] = count
        if len(row_labels) == 3:
            metrics["curriculum_maze_entry_starts"] = metrics.get(
                "curriculum_maze_starts", 0.0
            )
        column_histogram = snapshot.get("last_terrain_types_histogram", ())
        column_total = max(1, sum(int(value) for value in column_histogram))
        for index, value in enumerate(column_histogram):
            metrics[f"terrain_column_l{index}_share"] = float(value) / column_total
        low_columns = sum(int(value) for value in column_histogram[:10])
        high_columns = sum(int(value) for value in column_histogram[10:20])
        metrics["difficulty_l0_l9_share"] = float(low_columns) / column_total
        metrics["difficulty_l10_l19_share"] = float(high_columns) / column_total
        row_histogram = snapshot.get("last_terrain_levels_histogram", ())
        row_total = max(1, sum(int(value) for value in row_histogram))
        for index, value in enumerate(row_histogram):
            metrics[f"terrain_spawn_row_{index}_share"] = float(value) / row_total
        return metrics
    except (IndexError, TypeError, ValueError):
        return {}


def _tick_diagnostic_values(
    *,
    target: torch.Tensor,
    executed: torch.Tensor,
    response_aux: torch.Tensor,
    confidence: torch.Tensor,
    actions: torch.Tensor,
    start_goal: torch.Tensor,
    end_goal: torch.Tensor,
    done: torch.Tensor,
    hard: torch.Tensor,
    timeout: torch.Tensor,
    terminal_reason: torch.Tensor,
    duration_frames: torch.Tensor,
    stuck: torch.Tensor,
    feedback_age_clip_s: float,
    nav_period_frames: int = p2_contract.NAV_PERIOD_FRAMES,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Return per-env P2 telemetry without changing training semantics."""
    measured = response_aux[:, 6:9]
    valid = (response_aux[:, 9] > 0.5) & ~done
    source = response_aux[:, 11].round().long()
    true_velocity = response_aux[:, 12:15]
    progress = start_goal - end_goal
    duration_s = duration_frames.float().clamp_min(1.0) * p2_contract.CONTROL_DT_S
    action_peak = actions.abs().amax(dim=-1)
    equivalent_command_speed = torch.sqrt(
        target[:, 0].square()
        + target[:, 1].square()
        + (p2_contract.CRAWL_BODY_RADIUS_M * target[:, 2]).square()
    )
    values = {
        "target_vx": target[:, 0],
        "target_vy": target[:, 1],
        "target_wz": target[:, 2],
        "target_abs_wz": target[:, 2].abs(),
        "target_left": (target[:, 2] > 1.0e-4).float(),
        "target_right": (target[:, 2] < -1.0e-4).float(),
        "target_straight": (target[:, 2].abs() <= 0.05).float(),
        "target_pure_yaw": (
            (target[:, 0] <= 0.10) & (target[:, 2].abs() > 0.10)
        ).float(),
        "target_joint_turn": (
            (target[:, 0] > 0.10) & (target[:, 2].abs() > 0.10)
        ).float(),
        "exec_vx": executed[:, 0],
        "exec_vy": executed[:, 1],
        "exec_wz": executed[:, 2],
        "true_vx": true_velocity[:, 0],
        "true_vy": true_velocity[:, 1],
        "true_wz": true_velocity[:, 2],
        "true_lateral_speed_abs": true_velocity[:, 1].abs(),
        "target_exec_vx_error": (target[:, 0] - executed[:, 0]).abs(),
        "target_exec_vy_error": (target[:, 1] - executed[:, 1]).abs(),
        "target_exec_wz_error": (target[:, 2] - executed[:, 2]).abs(),
        "slew_saturation_rate": (
            (target - executed).abs().amax(dim=-1) > 1.0e-4
        ).float(),
        "feedback_valid": valid.float(),
        "feedback_source_invalid": (source == 0).float(),
        "feedback_source_sport": (source == 1).float(),
        "feedback_source_contact": (source == 2).float(),
        "adapter_confidence": confidence.reshape(-1),
        "success_rate": (terminal_reason == 1).float(),
        "failure_rate": (
            (terminal_reason == 2)
            | (terminal_reason == 4)
            | (hard & (terminal_reason != 1))
        ).float(),
        "timeout_rate": (done & (terminal_reason == 3)).float(),
        "wall_stuck_reset_rate": (done & (terminal_reason == 4)).float(),
        "hard_termination": hard.float(),
        "early_end_rate": (
            done & (duration_frames < int(nav_period_frames))
        ).float(),
        "transition_duration_frames": duration_frames.float(),
        "goal_distance": end_goal,
        "goal_progress": progress,
        "goal_progress_m_per_s": progress / duration_s,
        "goal_progress_positive_rate": (progress > 0.0).float(),
        "command_core_overflow_rate": (
            (target[:, 0] > p2_contract.TRUSTED_CORE["vx"][1])
            | (target[:, 1].abs() > p2_contract.TRUSTED_CORE["vy"][1])
            | (target[:, 2].abs() > p2_contract.TRUSTED_CORE["wz"][1])
        ).float(),
        "vx_near_hard_boundary_rate": (target[:, 0] >= 1.20).float(),
        "vy_nonzero_rate": (target[:, 1].abs() > 0.05).float(),
        "vy_over_core_rate": (target[:, 1].abs() > 0.20).float(),
        "vy_over_specialty_rate": (target[:, 1].abs() > 0.30).float(),
        "vy_near_hard_boundary_rate": (target[:, 1].abs() >= 0.38).float(),
        "vy_positive_rate": (target[:, 1] > 0.05).float(),
        "vy_negative_rate": (target[:, 1] < -0.05).float(),
        "wz_near_hard_boundary_rate": (target[:, 2].abs() >= 0.95).float(),
        "stuck": stuck.float(),
        "stuck_penalty": torch.zeros_like(stuck, dtype=torch.float32),
        "zero_command_rate": (equivalent_command_speed <= 1.0e-4).float(),
        "creep_command_rate": (
            (equivalent_command_speed > 1.0e-4)
            & (equivalent_command_speed < p2_contract.CRAWL_STABLE_MIN_MPS)
        ).float(),
        "stable_command_rate": (
            equivalent_command_speed >= p2_contract.CRAWL_STABLE_MIN_MPS
        ).float(),
        "terrain_start_column": response_aux[:, 28],
        "terrain_start_row": response_aux[:, 29],
        "current_segment": response_aux[:, p2_contract.CURRENT_SEGMENT_INDEX],
        "current_segment_valid": (
            response_aux[:, p2_contract.CURRENT_SEGMENT_INDEX] >= 0.0
        ).float(),
        "gait_sensor_mapping_valid": response_aux[
            :, p2_contract.GAIT_SENSOR_MAPPING_VALID_INDEX
        ],
        "body_collision_mapping_valid": response_aux[
            :, p2_contract.BODY_COLLISION_MAPPING_VALID_INDEX
        ],
        "tilt_xy_norm": torch.linalg.vector_norm(response_aux[:, 21:23], dim=-1),
        "low_level_action_abs_mean": actions.abs().mean(dim=-1),
        "low_level_action_peak_mean": action_peak,
        "low_level_action_saturation_rate": (action_peak >= 5.95).float(),
    }
    gait_valid = response_aux[:, p2_contract.GAIT_VALID_INDEX] > 0.5
    for index, leg in enumerate(("fl", "fr", "rl", "rr")):
        values[f"{leg}_duty_factor"] = response_aux[
            :, p2_contract.GAIT_DUTY_SLICE.start + index
        ]
        values[f"{leg}_mean_swing_time"] = response_aux[
            :, p2_contract.GAIT_MEAN_SWING_SLICE.start + index
        ]
        values[f"{leg}_max_air_time"] = response_aux[
            :, p2_contract.GAIT_MAX_AIR_SLICE.start + index
        ]
        values[f"{leg}_prolonged_air_ratio"] = response_aux[
            :, p2_contract.GAIT_PROLONGED_RATIO_SLICE.start + index
        ]
        values[f"{leg}_step_frequency"] = response_aux[
            :, p2_contract.GAIT_STEP_FREQUENCY_SLICE.start + index
        ]
        values[f"{leg}_slip_speed"] = response_aux[
            :, p2_contract.GAIT_SLIP_SPEED_SLICE.start + index
        ]
    values["gait_window_valid"] = gait_valid.float()
    valid_values = {
        "measured_vx": measured[:, 0],
        "measured_vy": measured[:, 1],
        "measured_wz": measured[:, 2],
        "feedback_age_s": p2_contract.feedback_age_seconds(
            response_aux[:, 10], age_clip_s=feedback_age_clip_s
        ),
        "vx_tracking_abs_error": (executed[:, 0] - measured[:, 0]).abs(),
        "vy_tracking_abs_error": (executed[:, 1] - measured[:, 1]).abs(),
        "wz_tracking_abs_error": (executed[:, 2] - measured[:, 2]).abs(),
        "feedback_true_velocity_error": torch.linalg.vector_norm(
            measured - true_velocity, dim=-1
        ),
    }
    return values, valid_values, valid


def _terminal_outcome_count(
    success_count: float,
    failure_count: float,
    timeout_count: float,
) -> float:
    """Count mutually exclusive terminal outcomes.

    Wall-stuck reason 4 is included in ``failure_count`` and remains a
    separate diagnostic metric; it must not be added to this denominator again.
    """
    return float(success_count + failure_count + timeout_count)


def _install_sigterm_handler(logger):
    if not hasattr(signal, "SIGTERM") or threading.current_thread() is not threading.main_thread():
        return None
    previous = signal.getsignal(signal.SIGTERM)
    if previous is signal.SIG_IGN:
        return None

    def handler(signum, frame):
        if previous is not signal.SIG_DFL and callable(previous):
            previous(signum, frame)
        raise SystemExit(f"SIGTERM({signum})")

    signal.signal(signal.SIGTERM, handler)
    logger.info(f"[P2NavPPO] installed chained SIGTERM handler previous={previous!r}")
    return previous


def _final_save(agent, logger, reason: str) -> bool:
    if getattr(agent, "_p2_final_save_done", False):
        return False
    try:
        logger.warning(f"[P2NavPPO] final platform archive request reason={reason}")
        agent.save_model()
    except Exception as exc:
        logger.error(
            "[P2NavPPO] final save failed without replacing the original exit reason: "
            f"{type(exc).__name__}: {exc}"
        )
        return False
    agent._p2_final_save_done = True
    return True


def _request_periodic_save(agent, logger, reason: str) -> bool:
    try:
        logger.info(
            f"[P2NavPPO] periodic platform archive request begin reason={reason}; "
            "platform must inject path and model ID"
        )
        agent.save_model()
    except Exception as exc:
        logger.error(
            "[P2NavPPO] checkpoint save failed; previous valid package remains and "
            f"retry is due in 60s: {type(exc).__name__}: {exc}"
        )
        return False
    return True


def _monitor_put(monitor, metrics: dict[str, float], logger=None) -> bool:
    if monitor is None:
        return False
    try:
        monitor.put_data({os.getpid(): metrics})
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "[P2NavPPO] monitor upload failed; training continues: "
                f"{type(exc).__name__}: {exc}"
            )
        return False
    return True


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    del args, kwargs
    agent = agents[0]
    env = envs[0]
    is_p4 = bool(getattr(agent, "is_p4_nav", False))
    is_p2_training = bool(getattr(agent, "is_p2_nav", False)) and not bool(
        getattr(agent, "is_p2_nav_eval", False)
    )
    if not (is_p2_training or is_p4):
        raise RuntimeError("navigation PPO workflow requires a P2/P4 training assembly")
    usr_conf, conf_path, _, stage = Config.load_conf(logger)
    p2_conf = usr_conf.get("p4_nav_ppo" if is_p4 else "p2_nav_ppo", {})
    terrain_track = (usr_conf.get("terrain") or {}).get("track") or {}
    segment_labels = p2_contract.canonical_track_segment_labels(
        terrain_track.get("sub_terrains", ())
    )
    segment_metric_labels = (
        p2_contract.TRACK_SEGMENT_METRIC_LABELS
        if all(
            label in p2_contract.TRACK_SEGMENT_METRIC_LABELS
            for label in segment_labels
        )
        else p2_contract.CANONICAL_TRACK_SEGMENT_METRIC_LABELS
    )
    feedback_conf = p2_conf.get("feedback_profile", {})
    if not isinstance(feedback_conf, dict):
        feedback_conf = {}
    feedback_age_clip_s = float(feedback_conf.get("age_clip_s", 0.8))
    algorithm = agent.algorithm
    logger.info(
        "[P2NavPPO] start "
        f"conf={conf_path} envs={agent.num_envs} parent={agent._p2_parent_model_id} "
        "rollout=32 ticks tbptt=16 minibatch=64seq microbatch=4seq "
        f"target_hours={(p4_contract.TRAINING_HOURS if is_p4 else p2_contract.TRAINING_HOURS):.1f} "
        "performance_gates=none"
    )
    if is_p4:
        # The worker owns EventManager and receives its stage configuration in
        # env.reset().  Feed the restored learner clock back into that config so
        # an exact-resume task restores the current Push phase instead of
        # replaying the initial two-hour no-Push phase.
        push_schedule = p2_conf.setdefault("push_schedule", {})
        push_schedule["resume_offset_s"] = _p4_worker_push_resume_offset(algorithm)
        stuck_reset = p2_conf.setdefault("stuck_reset", {})
        stuck_reset["resume_offset_s"] = _p4_worker_stuck_resume_offset(algorithm)
        logger.info(
            "[P4Resume] worker push/stuck resume_offset_s=%.3f "
            "diagnostic_elapsed_s=%.3f",
            push_schedule["resume_offset_s"],
            float(getattr(algorithm, "diagnostic_elapsed_seconds", 0.0)),
        )
    data = env.reset(usr_conf)
    if data is None:
        raise RuntimeError("P2 env.reset returned None")
    obs, critic_wire = data
    obs = torch.as_tensor(obs).to(agent.device).clone()
    critic_wire = torch.as_tensor(critic_wire).to(agent.device).clone()
    if obs.shape != (agent.num_envs, nav_contract.POLICY_OBS_DIM):
        raise ValueError(f"P2 reset policy shape drift: {tuple(obs.shape)}")
    expected_wire_dim = _expected_critic_wire_dim(is_p4=is_p4)
    if critic_wire.shape != (agent.num_envs, expected_wire_dim):
        raise ValueError(
            f"navigation reset critic wire shape drift: {tuple(critic_wire.shape)}; "
            f"expected=({agent.num_envs},{expected_wire_dim})"
        )
    algorithm.reset_live_state()

    resumed_seconds = float(algorithm.session_effective_seconds)
    resumed_clock_seconds = float(
        getattr(algorithm, "session_wall_seconds", resumed_seconds)
    )
    session_started = time.monotonic()
    p4_active_training_seconds = 0.0
    first_save_s = float(p2_conf.get("first_save_minutes", 5.0)) * 60.0
    save_interval_s = float(p2_conf.get("save_interval_minutes", 10.0)) * 60.0
    if first_save_s <= 0.0 or save_interval_s <= 0.0:
        raise ValueError("P2 checkpoint intervals must be positive")
    resumed_save_clock = resumed_clock_seconds if is_p4 else resumed_seconds
    if resumed_save_clock <= 0.0:
        next_save_clock = first_save_s
    else:
        next_save_clock = (
            math.floor(resumed_save_clock / save_interval_s) + 1
        ) * save_interval_s
    retry_save_at = None
    unfreeze_save_done = bool(algorithm.cnn_unfrozen)
    lifecycle_success = 0
    lifecycle_failures = 0
    nav_period_frames = int(algorithm.nav_period_frames)
    nav_rollout_ticks = int(algorithm.nav_rollout_ticks)
    nav_dt_s = float(algorithm.nav_dt_s)
    collision_traces = _CollisionTraceRecorder(
        nav_dt_s=nav_dt_s,
        pre_ticks=max(1, int(round(1.0 / nav_dt_s))),
        post_ticks=max(1, int(round(2.0 / nav_dt_s))),
    )
    goal_history = deque(maxlen=max(2, int(round(2.0 / nav_dt_s)) + 1))
    schedule_boundaries = (
        p4_contract.SCHEDULE_BOUNDARIES_SECONDS
        if is_p4
        else (
            p2_contract.SAFETY_WARM_END_SECONDS,
            p2_contract.SAFETY_STABILIZE_SECONDS,
        )
    )
    saved_schedule_boundaries = {
        boundary for boundary in schedule_boundaries if boundary <= resumed_seconds
    }
    previous_sigterm = _install_sigterm_handler(logger)
    agent._p2_training_started = True
    agent._p2_final_save_done = False
    last_log = time.monotonic()
    p4_metric_last_seen: dict[str, float] = {}
    p4_cumulative_counter_previous: dict[str, float] = {}
    p4_capture_active = torch.zeros(agent.num_envs, dtype=torch.bool, device=agent.device)
    p4_capture_entry_tick = torch.full(
        (agent.num_envs,), -1, dtype=torch.long, device=agent.device
    )
    p4_recovery_active = torch.zeros(
        agent.num_envs, dtype=torch.bool, device=agent.device
    )
    p4_recovery_awaiting_evidence = torch.zeros(
        agent.num_envs, dtype=torch.bool, device=agent.device
    )
    p4_recovery_entry_tick = torch.full(
        (agent.num_envs,), -1, dtype=torch.long, device=agent.device
    )
    p4_tick_index = 0
    restored_recovery = getattr(algorithm, "_p4_recovery_monitor_state", {})
    p4_recovery_event_times: deque[float] = deque(
        float(value) for value in restored_recovery.get("event_times", ())
    )
    p4_recovery_event_lifetime_count = int(
        restored_recovery.get("success_lifetime_count", 0)
    )
    p4_recovery_candidate_lifetime_count = int(
        restored_recovery.get("candidate_lifetime_count", 0)
    )
    p4_recovery_terminal_lifetime_count = int(
        restored_recovery.get("terminal_lifetime_count", 0)
    )
    target_seconds = (
        p4_contract.TARGET_EFFECTIVE_SECONDS
        if is_p4
        else p2_contract.TRAINING_HOURS * 3600.0
    )

    try:
        while algorithm.session_effective_seconds < target_seconds:
            if hasattr(algorithm, "begin_rollout"):
                algorithm.begin_rollout()
            rollout_started = time.monotonic()
            env_step_time_s = 0.0
            diagnostic_sums: dict[str, torch.Tensor] = {}
            diagnostic_maxima: dict[str, torch.Tensor] = {}
            diagnostic_valid_sums: dict[str, torch.Tensor] = {}
            diagnostic_count = 0
            diagnostic_valid_count = torch.zeros((), device=agent.device)
            p4_capture_candidate_count = 0
            p4_capture_entry_count = 0
            p4_capture_exit_count = 0
            p4_capture_zone_success_count = 0
            p4_capture_zone_collision_count = 0
            p4_capture_zone_timeout_count = 0
            p4_capture_zone_reset_count = 0
            p4_capture_reset_completion_error_count = 0
            p4_capture_success_latency_s: list[float] = []
            p4_recovery_candidate_entry_count = 0
            p4_recovery_success_count = 0
            p4_recovery_terminal_count = 0
            p4_recovery_unverified_exit_count = 0
            p4_recovery_early_sample_count = 0
            p4_recovery_confirmed_sample_count = 0
            p4_recovery_time_s: list[float] = []
            command_bin_counts = torch.zeros(5, 4, device=agent.device)
            command_bin_progress = torch.zeros_like(command_bin_counts)
            command_bin_tracking = torch.zeros_like(command_bin_counts)
            command_bin_tracking_counts = torch.zeros_like(command_bin_counts)
            command_bin_target_vx = torch.zeros_like(command_bin_counts)
            command_bin_exec_vx = torch.zeros_like(command_bin_counts)
            command_bin_true_vx = torch.zeros_like(command_bin_counts)
            command_bin_target_wz = torch.zeros_like(command_bin_counts)
            command_bin_exec_wz = torch.zeros_like(command_bin_counts)
            command_bin_true_wz = torch.zeros_like(command_bin_counts)
            command_bin_gait = torch.zeros_like(command_bin_counts)
            command_bin_success = torch.zeros_like(command_bin_counts)
            command_bin_failure = torch.zeros_like(command_bin_counts)
            command_bin_timeout = torch.zeros_like(command_bin_counts)
            command_bin_vy_nonzero = torch.zeros_like(command_bin_counts)
            command_bin_abs_vy = torch.zeros_like(command_bin_counts)
            command_bin_vy_outer = torch.zeros_like(command_bin_counts)
            row_counts = torch.zeros(len(segment_metric_labels), device=agent.device)
            row_progress = torch.zeros_like(row_counts)
            row_target_vx = torch.zeros_like(row_counts)
            row_exec_vx = torch.zeros_like(row_counts)
            row_true_vx = torch.zeros_like(row_counts)
            row_target_wz = torch.zeros_like(row_counts)
            row_stop = torch.zeros_like(row_counts)
            row_creep = torch.zeros_like(row_counts)
            row_spin = torch.zeros_like(row_counts)
            row_gait = torch.zeros_like(row_counts)
            row_predictive_clearance = torch.zeros_like(row_counts)
            row_predictive_risk = torch.zeros_like(row_counts)
            row_predictive_penalty = torch.zeros_like(row_counts)
            row_teacher_risk = torch.zeros_like(row_counts)
            row_teacher_high_risk = torch.zeros_like(row_counts)
            row_success = torch.zeros_like(row_counts)
            row_failure = torch.zeros_like(row_counts)
            row_timeout = torch.zeros_like(row_counts)
            row_reason4 = torch.zeros_like(row_counts)
            p4_spawn_outcomes = torch.zeros(
                len(segment_metric_labels), 4, device=agent.device
            )
            p4_current_outcomes = torch.zeros_like(p4_spawn_outcomes)
            p4_episode_start_counts = torch.zeros_like(row_counts)
            p4_spawn_reset_event_count = torch.zeros((), device=agent.device)
            p4_spawn_full_start_event_count = torch.zeros((), device=agent.device)
            p4_spawn_safe_point_event_count = torch.zeros((), device=agent.device)
            p4_spawn_hard_position_event_count = torch.zeros((), device=agent.device)
            p4_spawn_quartile_event_counts = torch.zeros(4, device=agent.device)
            signed_chain_sums = {
                axis: {
                    sign: {
                        stage: torch.zeros((), device=agent.device)
                        for stage in (
                            "policy_target",
                            "limited_target",
                            "mapped_cmd",
                            "exec",
                            "true",
                        )
                    }
                    for sign in ("positive", "negative")
                }
                for axis in ("vy", "wz")
            }
            signed_chain_counts = {
                axis: {
                    sign: torch.zeros((), device=agent.device)
                    for sign in ("positive", "negative")
                }
                for axis in ("vy", "wz")
            }
            p4_cumulative_counter_names = {
                "spawn_reason4_retry_count",
                "spawn_reason4_exhausted_count",
                "spawn_reason4_fallback_applied_count",
                "spawn_all_position_applied_count",
                "spawn_validation_failure_count",
                "spawn_write_failure_count",
            }
            segment_diagnostic_count = torch.zeros((), device=agent.device)
            quantile_names = (
                "target_vx", "target_vy", "target_wz",
                "exec_vx", "exec_vy", "exec_wz",
                "true_vx", "true_vy", "true_wz",
                "measured_vx", "measured_vy", "measured_wz",
            )
            quantile_samples = {
                name: torch.full(
                    (nav_rollout_ticks, agent.num_envs),
                    float("nan"),
                    device=agent.device,
                )
                for name in quantile_names
            }
            for _tick in range(nav_rollout_ticks):
                start_goal = (
                    critic_wire[:, nav_contract.CRITIC_GOAL3_START + 2]
                    * nav_contract.GOAL_DIST_SCALE_M
                ).detach()
                frame_safety_reward = torch.zeros(agent.num_envs, device=agent.device)
                terminal_goal = torch.full(
                    (agent.num_envs,), float("nan"), device=agent.device
                )
                terminal_segment = torch.full(
                    (agent.num_envs,), -1.0, device=agent.device
                )
                terminal_target = torch.zeros(
                    agent.num_envs, 3, device=agent.device
                )
                terminal_executed = torch.zeros_like(terminal_target)
                terminal_aux = torch.zeros(
                    agent.num_envs,
                    p2_contract.WORKER_AUX_DIM,
                    device=agent.device,
                )
                terminal_p4_extra = (
                    torch.zeros(
                        agent.num_envs,
                        p4_contract.P4_WORKER_EXTRA_DIM,
                        device=agent.device,
                    )
                    if is_p4
                    else None
                )
                duration = torch.zeros(agent.num_envs, dtype=torch.long, device=agent.device)
                path_length_m = torch.zeros(agent.num_envs, device=agent.device)
                hard = torch.zeros(agent.num_envs, dtype=torch.bool, device=agent.device)
                timeout = torch.zeros_like(hard)
                terminal_reason = torch.zeros(
                    agent.num_envs, dtype=torch.long, device=agent.device
                )
                active = torch.ones_like(hard)
                result = None
                for frame in range(nav_period_frames):
                    result, _critic_obs, aux = algorithm.frame_begin(obs, critic_wire)
                    actions = torch.clamp(result["actions"], -6.0, 6.0)
                    frame_target = algorithm.command.active_target
                    frame_executed = algorithm.command.exec_cmd
                    frame_aux = aux
                    frame_true_xy = torch.nan_to_num(
                        frame_aux[:, 12:14], nan=0.0, posinf=0.0, neginf=0.0
                    )
                    path_length_m += (
                        torch.linalg.vector_norm(frame_true_xy, dim=-1)
                        * p2_contract.CONTROL_DT_S
                        * active.float()
                    )
                    frame_p4_extra = (
                        algorithm._p4_worker_extra.detach().clone()
                        if is_p4
                        else None
                    )
                    if frame > 0 and bool((~active).any()):
                        # Done envs stay on the reset zero-command path for the
                        # remainder of this transition; their rewards are masked.
                        algorithm.command.active_target[~active] = 0.0
                        algorithm.command.exec_cmd[~active] = 0.0
                    env_step_started = time.perf_counter()
                    step_data = env.step(actions)
                    env_step_time_s += time.perf_counter() - env_step_started
                    _, next_obs, rewards, terminated, truncated, infos, next_critic = _extract_step(step_data)
                    next_obs = torch.as_tensor(next_obs).to(agent.device)
                    next_critic = torch.as_tensor(next_critic).to(agent.device)
                    rewards = torch.as_tensor(rewards).to(agent.device).reshape(-1)
                    # P4 appends a training-only tail after the stable aux62.
                    # Terminal/reset logic consumes only that stable prefix.
                    next_aux = next_critic[
                        :,
                        p2_contract.CRITIC_OBS_DIM : p2_contract.PRIVILEGED_WIRE_DIM,
                    ]
                    frame_done, frame_timeout = _frame_done_masks(
                        terminated,
                        truncated,
                        infos,
                        agent.device,
                        worker_aux=next_aux,
                    )
                    weight = p2_contract.GAMMA_FRAME ** frame
                    frame_safety_reward += rewards * active.float() * weight
                    duration += active.long()
                    new_done = active & frame_done
                    reason, new_hard, new_timeout = _resolve_terminal_outcome(
                        new_done,
                        frame_timeout,
                        next_aux[:, 25].round().long(),
                    )
                    terminal_reason[new_done] = reason[new_done]
                    if bool(new_done.any()):
                        terminal_target[new_done] = frame_target[new_done]
                        terminal_executed[new_done] = frame_executed[new_done]
                        terminal_aux[new_done] = frame_aux[new_done]
                        if is_p4:
                            terminal_p4_extra[
                                new_done, p4_contract.RAW_GOAL_XY_SLICE
                            ] = frame_p4_extra[
                                new_done, p4_contract.RAW_GOAL_XY_SLICE
                            ]
                            terminal_p4_extra[new_done, 2:] = next_critic[
                                new_done,
                                p3_contract.P3_PRIVILEGED_WIRE_DIM + 2 :
                                p4_contract.P4_PRIVILEGED_WIRE_DIM,
                            ]
                        terminal_goal[new_done] = next_aux[
                            new_done, p2_contract.PRE_STEP_GOAL_DISTANCE_INDEX
                        ]
                        terminal_segment[new_done] = next_aux[
                            new_done, p2_contract.CURRENT_SEGMENT_INDEX
                        ]
                    hard |= new_hard
                    timeout |= new_timeout
                    active &= ~frame_done
                    algorithm.frame_end(aux, frame_done)
                    obs, critic_wire = next_obs, next_critic
                    try:
                        agent.learn(None)
                    except Exception as exc:
                        lifecycle_failures += 1
                        if isinstance(exc, CheckpointSaveError) and retry_save_at is None:
                            retry_save_at = time.monotonic() + 60.0
                        if lifecycle_failures == 1 or lifecycle_failures % 100 == 0:
                            logger.error(
                                "[P2NavPPO] lifecycle callback failed; training continues "
                                f"failures={lifecycle_failures} error={type(exc).__name__}: {exc}"
                            )
                    else:
                        lifecycle_success += 1

                live_end_goal = (
                    critic_wire[:, nav_contract.CRITIC_GOAL3_START + 2]
                    * nav_contract.GOAL_DIST_SCALE_M
                )
                transition_done = hard | timeout
                end_goal = torch.where(
                    transition_done & torch.isfinite(terminal_goal),
                    terminal_goal,
                    live_end_goal,
                )
                if bool(transition_done.any()):
                    for previous_goal in goal_history:
                        previous_goal[transition_done] = float("nan")
                goal_history.append(end_goal.detach().clone())
                stuck = _goal_history_stuck(goal_history, end_goal)
                pending = algorithm.pending_tick
                if pending is None:
                    raise RuntimeError("P2 pending transition disappeared before finish_tick")
                target = _terminal_safe_tensor(
                    algorithm.command.active_target,
                    terminal_target,
                    transition_done,
                )
                executed = _terminal_safe_tensor(
                    algorithm.command.exec_cmd,
                    terminal_executed,
                    transition_done,
                )
                diagnostic_aux = _terminal_safe_tensor(
                    next_aux, terminal_aux, transition_done
                )
                values, valid_values, valid_mask = _tick_diagnostic_values(
                    target=target,
                    executed=executed,
                    response_aux=diagnostic_aux,
                    confidence=pending["confidence"],
                    actions=actions,
                    start_goal=start_goal,
                    end_goal=end_goal,
                    done=transition_done,
                    hard=hard,
                    timeout=timeout,
                    terminal_reason=terminal_reason,
                    duration_frames=duration,
                    stuck=stuck,
                    feedback_age_clip_s=feedback_age_clip_s,
                    nav_period_frames=nav_period_frames,
                )
                for name, value in values.items():
                    diagnostic_sums[name] = diagnostic_sums.get(
                        name, torch.zeros((), device=agent.device)
                    ) + value.sum()
                diagnostic_valid_count += valid_mask.float().sum()
                for name, value in valid_values.items():
                    diagnostic_valid_sums[name] = diagnostic_valid_sums.get(
                        name, torch.zeros((), device=agent.device)
                    ) + torch.where(valid_mask, value, torch.zeros_like(value)).sum()
                for name in (
                    "target_vx",
                    "target_vy",
                    "target_wz",
                    "exec_vx",
                    "exec_vy",
                    "exec_wz",
                    "true_vx",
                    "true_vy",
                    "true_wz",
                ):
                    quantile_samples[name][_tick].copy_(values[name].detach())
                for name in ("measured_vx", "measured_vy", "measured_wz"):
                    quantile_samples[name][_tick].copy_(
                        torch.where(
                            valid_mask,
                            valid_values[name].detach(),
                            torch.full_like(valid_values[name], float("nan")),
                        )
                    )
                vx_bin = torch.bucketize(
                    target[:, 0],
                    torch.tensor((0.1, 0.4, 0.8, 1.0), device=agent.device),
                )
                wz_bin = torch.bucketize(
                    target[:, 2].abs(),
                    torch.tensor((0.1, 0.4, 0.8), device=agent.device),
                )
                true_tracking = (
                    (executed - diagnostic_aux[:, 12:15]).abs()
                    / torch.tensor(
                        p2_contract.COMMAND_NORMALIZATION, device=agent.device
                    )
                ).mean(dim=-1)
                flat_bin = vx_bin * 4 + wz_bin
                def add_command_bin(target_buffer, source):
                    target_buffer.view(-1).scatter_add_(0, flat_bin, source.float())

                add_command_bin(command_bin_counts, torch.ones_like(target[:, 0]))
                add_command_bin(command_bin_vy_nonzero, target[:, 1].abs() > 0.05)
                add_command_bin(command_bin_abs_vy, target[:, 1].abs())
                add_command_bin(command_bin_vy_outer, target[:, 1].abs() > 0.20)
                add_command_bin(command_bin_progress, values["goal_progress_m_per_s"])
                tracking_valid = (~transition_done).float()
                add_command_bin(command_bin_tracking, true_tracking * tracking_valid)
                add_command_bin(command_bin_tracking_counts, tracking_valid)
                add_command_bin(command_bin_target_vx, target[:, 0])
                add_command_bin(command_bin_exec_vx, executed[:, 0])
                add_command_bin(command_bin_true_vx, diagnostic_aux[:, 12])
                add_command_bin(command_bin_target_wz, target[:, 2].abs())
                add_command_bin(command_bin_exec_wz, executed[:, 2].abs())
                add_command_bin(command_bin_true_wz, diagnostic_aux[:, 14].abs())
                add_command_bin(command_bin_success, values["success_rate"])
                add_command_bin(command_bin_failure, values["failure_rate"])
                add_command_bin(command_bin_timeout, values["timeout_rate"])
                live_segment = diagnostic_aux[:, p2_contract.CURRENT_SEGMENT_INDEX]
                raw_segment = _terminal_safe_segment(
                    live_segment,
                    terminal_segment,
                    transition_done,
                )
                row_index = p2_contract.track_segment_metric_indices(
                    raw_segment, segment_labels
                )
                segment_valid = row_index >= 0
                segment_diagnostic_count += segment_valid.float().sum()
                valid_row_index = row_index.clamp(0, len(segment_metric_labels) - 1)
                def add_segment(target_buffer, source):
                    target_buffer.scatter_add_(
                        0, valid_row_index, source.float() * segment_valid.float()
                    )

                add_segment(row_counts, torch.ones_like(target[:, 0]))
                add_segment(row_progress, values["goal_progress_m_per_s"])
                add_segment(row_target_vx, target[:, 0])
                add_segment(row_exec_vx, executed[:, 0])
                add_segment(row_true_vx, diagnostic_aux[:, 12])
                add_segment(row_target_wz, target[:, 2].abs())
                add_segment(row_stop, values["zero_command_rate"])
                add_segment(row_creep, values["creep_command_rate"])
                add_segment(row_spin, values["target_pure_yaw"])
                add_segment(row_success, terminal_reason == 1)
                add_segment(row_failure, terminal_reason == 2)
                add_segment(row_timeout, terminal_reason == 3)
                add_segment(row_reason4, terminal_reason == 4)
                if is_p4:
                    for outcome_index, reason_code in enumerate((1, 2, 3, 4)):
                        outcome_mask = segment_valid & (terminal_reason == reason_code)
                        p4_current_outcomes[:, outcome_index].scatter_add_(
                            0, valid_row_index, outcome_mask.float()
                        )
                diagnostic_count += agent.num_envs
                finish_critic_wire = critic_wire
                if is_p4 and bool(transition_done.any()):
                    finish_critic_wire = _terminal_safe_p4_critic_wire(
                        critic_wire,
                        terminal_p4_extra,
                        transition_done,
                    )
                full = algorithm.finish_tick(
                    obs,
                    finish_critic_wire,
                    frame_safety_reward=frame_safety_reward,
                    start_goal_distance=start_goal,
                    end_goal_distance=end_goal,
                    terminal_reason=terminal_reason,
                    duration_frames=duration,
                    hard_terminated=hard,
                    timeout=timeout,
                    terminal_safe_aux=diagnostic_aux,
                    terminal_safe_exec_cmd=executed,
                    path_length_m=path_length_m,
                )
                for component, value in algorithm.last_tick_penalties.items():
                    name = f"reward_{component}"
                    diagnostic_sums[name] = diagnostic_sums.get(
                        name, torch.zeros((), device=agent.device)
                    ) + value.sum()
                for name, value in algorithm.last_tick_diagnostics.items():
                    if name in p4_cumulative_counter_names:
                        diagnostic_maxima[name] = torch.maximum(
                            diagnostic_maxima.get(
                                name, torch.zeros((), device=agent.device)
                            ),
                            value.max(),
                        )
                        continue
                    diagnostic_sums[name] = diagnostic_sums.get(
                        name, torch.zeros((), device=agent.device)
                    ) + value.sum()
                    if name == "body_collision_force":
                        diagnostic_maxima[f"{name}_max"] = torch.maximum(
                            diagnostic_maxima.get(
                                f"{name}_max",
                                torch.zeros((), device=agent.device),
                            ),
                            value.max(),
                        )
                if is_p4:
                    diagnostics = algorithm.last_tick_diagnostics
                    spawn_segment = diagnostics.get(
                        "segment_frontier_spawn_segment",
                        torch.full(
                            (agent.num_envs,), -1.0, device=agent.device
                        ),
                    ).reshape(-1).round().long()
                    spawn_valid = (
                        (spawn_segment >= 0)
                        & (spawn_segment < len(segment_metric_labels))
                    )
                    spawn_index = spawn_segment.clamp(
                        0, len(segment_metric_labels) - 1
                    )
                    for outcome_index, reason_code in enumerate((1, 2, 3, 4)):
                        outcome_mask = spawn_valid & (terminal_reason == reason_code)
                        p4_spawn_outcomes[:, outcome_index].scatter_add_(
                            0, spawn_index, outcome_mask.float()
                        )

                    new_episode = transition_done
                    new_full_start = diagnostics.get(
                        "spawn_full_start_share",
                        torch.zeros(agent.num_envs, device=agent.device),
                    ).reshape(-1) > 0.5
                    new_safe_point = diagnostics.get(
                        "spawn_safe_point_share",
                        torch.zeros(agent.num_envs, device=agent.device),
                    ).reshape(-1) > 0.5
                    new_quartile = diagnostics.get(
                        "spawn_position_quartile",
                        torch.full(
                            (agent.num_envs,), -1.0, device=agent.device
                        ),
                    ).reshape(-1).round().long()
                    new_spawn_segment = diagnostics.get(
                        "spawn_segment_index",
                        torch.full(
                            (agent.num_envs,), -1.0, device=agent.device
                        ),
                    ).reshape(-1).round().long()
                    new_spawn_valid = (
                        new_episode
                        & (new_spawn_segment >= 0)
                        & (new_spawn_segment < len(segment_metric_labels))
                    )
                    p4_episode_start_counts.scatter_add_(
                        0,
                        new_spawn_segment.clamp(0, len(segment_metric_labels) - 1),
                        new_spawn_valid.float(),
                    )
                    p4_spawn_reset_event_count += new_episode.float().sum()
                    p4_spawn_full_start_event_count += (
                        new_episode & new_full_start
                    ).float().sum()
                    segment_start = new_episode & ~new_full_start
                    p4_spawn_safe_point_event_count += (
                        segment_start & new_safe_point
                    ).float().sum()
                    p4_spawn_hard_position_event_count += (
                        segment_start & ~new_safe_point
                    ).float().sum()
                    quartile_valid = (
                        segment_start & (new_quartile >= 0) & (new_quartile < 4)
                    )
                    p4_spawn_quartile_event_counts.scatter_add_(
                        0,
                        new_quartile.clamp(0, 3),
                        quartile_valid.float(),
                    )

                    chain_values = {
                        "policy_target": {
                            axis: diagnostics.get(
                                f"policy_target_{axis}",
                                torch.zeros(agent.num_envs, device=agent.device),
                            ).reshape(-1)
                            for axis in ("vy", "wz")
                        },
                        "limited_target": {
                            axis: diagnostics.get(
                                f"limited_target_{axis}",
                                torch.zeros(agent.num_envs, device=agent.device),
                            ).reshape(-1)
                            for axis in ("vy", "wz")
                        },
                        "mapped_cmd": {
                            axis: diagnostics.get(
                                f"mapped_cmd_{axis}",
                                torch.zeros(agent.num_envs, device=agent.device),
                            ).reshape(-1)
                            for axis in ("vy", "wz")
                        },
                        "exec": {"vy": executed[:, 1], "wz": executed[:, 2]},
                        "true": {
                            "vy": diagnostic_aux[:, 13],
                            "wz": diagnostic_aux[:, 14],
                        },
                    }
                    for axis in ("vy", "wz"):
                        policy_axis = chain_values["policy_target"][axis]
                        for sign, mask in (
                            ("positive", policy_axis > 1.0e-4),
                            ("negative", policy_axis < -1.0e-4),
                        ):
                            signed_chain_counts[axis][sign] += mask.float().sum()
                            for stage, by_axis in chain_values.items():
                                signed_chain_sums[axis][sign][stage] += (
                                    by_axis[axis] * mask.float()
                                ).sum()
                    capture_active = diagnostics.get(
                        "near_goal_capture_active",
                        torch.zeros(agent.num_envs, device=agent.device),
                    ).reshape(-1) > 0.5
                    capture_candidate = diagnostics.get(
                        "near_goal_capture_candidate",
                        torch.zeros(agent.num_envs, device=agent.device),
                    ).reshape(-1) > 0.5
                    capture_entry = capture_active & ~p4_capture_active
                    capture_exit = p4_capture_active & (
                        ~capture_active | transition_done
                    )
                    capture_zone = p4_capture_active | capture_active
                    capture_success = capture_zone & (terminal_reason == 1)
                    capture_timeout = capture_zone & (terminal_reason == 3)
                    raw_wall_reset = diagnostics.get(
                        "wall_stuck_raw_term",
                        torch.zeros(agent.num_envs, device=agent.device),
                    ).reshape(-1) > 0.5
                    capture_reset = capture_zone & raw_wall_reset
                    capture_collision = capture_zone & (terminal_reason == 2)
                    p4_capture_candidate_count += int(capture_candidate.sum().item())
                    p4_capture_entry_count += int(capture_entry.sum().item())
                    p4_capture_exit_count += int(capture_exit.sum().item())
                    p4_capture_zone_success_count += int(capture_success.sum().item())
                    p4_capture_zone_collision_count += int(capture_collision.sum().item())
                    p4_capture_zone_timeout_count += int(capture_timeout.sum().item())
                    p4_capture_zone_reset_count += int(capture_reset.sum().item())
                    # Terminal reasons are exclusive.  Keep this visible so a
                    # reset can never silently inflate the completion series.
                    p4_capture_reset_completion_error_count += int(
                        _p4_reset_completion_mismatch(
                            raw_wall_reset, terminal_reason == 1
                        ).sum().item()
                    )
                    capture_entry_ticks = torch.where(
                        capture_entry,
                        torch.full_like(p4_capture_entry_tick, p4_tick_index),
                        p4_capture_entry_tick,
                    )
                    success_entries = capture_entry_ticks[capture_success]
                    p4_capture_success_latency_s.extend(
                        (
                            (p4_tick_index - success_entries + 1).clamp_min(0)
                            .float()
                            .mul(nav_dt_s)
                            .detach()
                            .cpu()
                            .tolist()
                        )
                    )
                    p4_capture_entry_tick[capture_entry] = p4_tick_index
                    p4_capture_entry_tick[transition_done] = -1
                    p4_capture_active = capture_active & ~transition_done
                    recovery_candidate = diagnostics.get(
                        "wall_stuck_candidate_share",
                        torch.zeros(agent.num_envs, device=agent.device),
                    ).reshape(-1) > 0.5
                    recovery_duration = diagnostics.get(
                        "wall_stuck_duration_s",
                        torch.zeros(agent.num_envs, device=agent.device),
                    ).reshape(-1)
                    recovery_mapping_valid = diagnostics.get(
                        "wall_stuck_mapping_valid",
                        torch.zeros(agent.num_envs, device=agent.device),
                    ).reshape(-1) > 0.5
                    recovery_true_xy = diagnostic_aux[:, 12:14]
                    recovery_true_speed = torch.linalg.vector_norm(
                        recovery_true_xy, dim=-1
                    )
                    recovery_progress = start_goal - end_goal
                    recovery_evidence_valid = (
                        recovery_mapping_valid
                        & torch.isfinite(recovery_true_xy).all(dim=-1)
                        & torch.isfinite(recovery_progress)
                    )
                    recovery_escape_evidence = (
                        (recovery_true_speed > 0.08)
                        | (recovery_progress > 0.02)
                    )
                    recovery_masks = _p4_recovery_event_masks(
                        p4_recovery_active,
                        p4_recovery_awaiting_evidence,
                        recovery_candidate,
                        recovery_duration,
                        transition_done,
                        recovery_evidence_valid,
                        recovery_escape_evidence,
                        terminal_reason == 1,
                    )
                    recovery_entry = recovery_masks["entry"]
                    recovery_success = recovery_masks["success"]
                    recovery_terminal = recovery_masks["terminal"]
                    recovery_unverified_exit = recovery_masks["unverified_exit"]
                    recovery_early = recovery_masks["early"]
                    recovery_confirmed = recovery_masks["confirmed"]

                    p4_recovery_candidate_entry_count += int(
                        recovery_entry.sum().item()
                    )
                    p4_recovery_success_count += int(recovery_success.sum().item())
                    p4_recovery_terminal_count += int(recovery_terminal.sum().item())
                    p4_recovery_unverified_exit_count += int(
                        recovery_unverified_exit.sum().item()
                    )
                    p4_recovery_early_sample_count += int(recovery_early.sum().item())
                    p4_recovery_confirmed_sample_count += int(
                        recovery_confirmed.sum().item()
                    )
                    p4_recovery_candidate_lifetime_count += int(
                        recovery_entry.sum().item()
                    )
                    p4_recovery_terminal_lifetime_count += int(
                        recovery_terminal.sum().item()
                    )

                    recovery_entry_ticks = torch.where(
                        recovery_entry,
                        torch.full_like(p4_recovery_entry_tick, p4_tick_index),
                        p4_recovery_entry_tick,
                    )
                    successful_entries = recovery_entry_ticks[recovery_success]
                    p4_recovery_time_s.extend(
                        (
                            (p4_tick_index - successful_entries + 1).clamp_min(0)
                            .float()
                            .mul(nav_dt_s)
                            .detach()
                            .cpu()
                            .tolist()
                        )
                    )
                    p4_recovery_entry_tick[recovery_entry] = p4_tick_index
                    p4_recovery_entry_tick[recovery_success | recovery_terminal] = -1
                    p4_recovery_active = recovery_masks["next_active"]
                    p4_recovery_awaiting_evidence = recovery_masks[
                        "next_awaiting"
                    ]

                    recovery_count = int(recovery_success.sum().item())
                    if recovery_count:
                        event_time = resumed_clock_seconds + (
                            time.monotonic() - session_started
                        )
                        p4_recovery_event_times.extend([event_time] * recovery_count)
                        p4_recovery_event_lifetime_count += recovery_count
                    algorithm._p4_recovery_monitor_state = {
                        "event_times": list(p4_recovery_event_times),
                        "success_lifetime_count": p4_recovery_event_lifetime_count,
                        "candidate_lifetime_count": p4_recovery_candidate_lifetime_count,
                        "terminal_lifetime_count": p4_recovery_terminal_lifetime_count,
                    }
                    p4_tick_index += 1
                collision_traces.observe(
                    iteration=algorithm.current_iteration,
                    tick=_tick,
                    onset=algorithm.last_tick_diagnostics[
                        "body_collision_onset"
                    ],
                    target_vx=target[:, 0],
                    target_vy=target[:, 1],
                    target_wz=target[:, 2],
                    exec_vx=executed[:, 0],
                    exec_vy=executed[:, 1],
                    exec_wz=executed[:, 2],
                    measured_vx=diagnostic_aux[:, 6],
                    measured_vy=diagnostic_aux[:, 7],
                    measured_wz=diagnostic_aux[:, 8],
                    true_vx=diagnostic_aux[:, 12],
                    true_vy=diagnostic_aux[:, 13],
                    true_wz=diagnostic_aux[:, 14],
                    collision_force=diagnostic_aux[
                        :, p2_contract.BODY_COLLISION_FORCE_INDEX
                    ],
                    progress=start_goal - end_goal,
                    segment=diagnostic_aux[:, p2_contract.CURRENT_SEGMENT_INDEX],
                    column=diagnostic_aux[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX],
                    reason=terminal_reason,
                    risk=algorithm.last_tick_diagnostics[
                        "predictive_collision_risk"
                    ],
                )
                gait_reward = algorithm.last_tick_penalties["gait_symmetry"].reshape(-1)
                predictive_clearance = algorithm.last_tick_diagnostics[
                    "predictive_collision_clearance_m"
                ].reshape(-1)
                predictive_risk = algorithm.last_tick_diagnostics[
                    "predictive_collision_risk"
                ].reshape(-1)
                predictive_penalty = algorithm.last_tick_penalties[
                    "predictive_collision_risk"
                ].reshape(-1)
                teacher_risk = torch.stack(
                    tuple(
                        algorithm.last_tick_diagnostics[f"teacher_risk_{name}"].reshape(-1)
                        for name in ("left", "center", "right")
                    ),
                    dim=1,
                ).amax(dim=1)
                add_command_bin(command_bin_gait, gait_reward)
                add_segment(row_gait, gait_reward)
                add_segment(row_predictive_clearance, predictive_clearance)
                add_segment(row_predictive_risk, predictive_risk)
                add_segment(row_predictive_penalty, predictive_penalty)
                add_segment(row_teacher_risk, teacher_risk)
                add_segment(row_teacher_high_risk, teacher_risk >= 0.5)
                if _tick < nav_rollout_ticks - 1 and full:
                    raise RuntimeError("P2 rollout filled before the 32-tick boundary")
                if _tick == nav_rollout_ticks - 1 and not full:
                    raise RuntimeError("P2 rollout did not fill at the 32-tick boundary")

            metrics = algorithm.update()
            if is_p4:
                emit_nav_event(
                    "iteration",
                    role="aisrv",
                    iteration=int(algorithm.current_iteration),
                    valid_ticks=float(
                        nav_rollout_ticks * agent.num_envs
                    ),
                    update_skipped_no_valid=0.0,
                )
            now = time.monotonic()
            if is_p4:
                # P4's persisted effective clock is the time spent collecting
                # the rollout and applying its updates.  Checkpoint I/O,
                # monitoring and logging below must not consume the 7200s
                # gradient-training budget.
                p4_active_training_seconds += max(0.0, now - rollout_started)
                algorithm.update_training_clocks(
                    resumed_clock_seconds + (now - session_started),
                    session_effective_seconds=(
                        resumed_seconds + p4_active_training_seconds
                    ),
                )
            else:
                algorithm.update_training_clocks(
                    resumed_clock_seconds + (now - session_started)
                )
            curriculum_snapshot = algorithm.curriculum_probe.state_dict()
            agent.training_elapsed_h = algorithm.effective_training_seconds / 3600.0
            unfrozen_now = algorithm.maybe_unfreeze_cnn(algorithm.session_effective_seconds)
            if unfrozen_now and not unfreeze_save_done:
                _request_periodic_save(agent, logger, "cnn_unfreeze_boundary")
                unfreeze_save_done = True
                if not is_p4:
                    saved_schedule_boundaries.add(p2_contract.SAFETY_WARM_END_SECONDS)
            for boundary in schedule_boundaries:
                if (
                    boundary <= algorithm.session_effective_seconds
                    and boundary not in saved_schedule_boundaries
                ):
                    _request_periodic_save(
                        agent, logger, f"schedule_boundary_{int(boundary)}s"
                    )
                    saved_schedule_boundaries.add(boundary)

            save_clock = (
                float(algorithm.session_wall_seconds)
                if is_p4
                else float(algorithm.session_effective_seconds)
            )
            due = save_clock >= next_save_clock
            retry_due = retry_save_at is not None and now >= retry_save_at
            if due or retry_due:
                if _request_periodic_save(agent, logger, "wall_clock" if due else "retry"):
                    retry_save_at = None
                    while next_save_clock <= save_clock:
                        next_save_clock += save_interval_s
                else:
                    retry_save_at = now + 60.0

            metrics.update(algorithm.memory_metrics())
            metrics.update(collision_traces.drain(logger))
            if diagnostic_count:
                success_count = float(
                    diagnostic_sums.get(
                        "success_rate", torch.zeros((), device=agent.device)
                    ).detach().cpu()
                )
                failure_count = float(
                    diagnostic_sums.get(
                        "failure_rate", torch.zeros((), device=agent.device)
                    ).detach().cpu()
                )
                timeout_count = float(
                    diagnostic_sums.get(
                        "timeout_rate", torch.zeros((), device=agent.device)
                    ).detach().cpu()
                )
                wall_stuck_reset_count = float(
                    diagnostic_sums.get(
                        "wall_stuck_reset_rate",
                        torch.zeros((), device=agent.device),
                    ).detach().cpu()
                )
                terminal_count = _terminal_outcome_count(
                    success_count,
                    failure_count,
                    timeout_count,
                )
                metrics.update(
                    {
                        "rollout_success_count": success_count,
                        "rollout_failure_count": failure_count,
                        "rollout_timeout_count": timeout_count,
                        "rollout_wall_stuck_reset_count": wall_stuck_reset_count,
                        "rollout_terminal_count": terminal_count,
                        "episode_success_fraction": success_count
                        / max(terminal_count, 1.0),
                        "episode_timeout_fraction": timeout_count
                        / max(terminal_count, 1.0),
                        "rollout_wall_stuck_saved_seconds": float(
                            diagnostic_sums.get(
                                "wall_stuck_saved_seconds",
                                torch.zeros((), device=agent.device),
                            ).detach().cpu()
                        ),
                    }
                )
                metrics.update(
                    {
                        name: float(value.detach().cpu()) / diagnostic_count
                        for name, value in diagnostic_sums.items()
                    }
                )
                if is_p4:
                    metrics.update(
                        _p4_conditional_metrics(diagnostic_sums, agent.device)
                    )
                eligible_safe_choices = float(
                    diagnostic_sums.get(
                        "safe_alternative_available",
                        torch.zeros((), device=agent.device),
                    ).detach().cpu()
                )
                selected_safest_choices = float(
                    diagnostic_sums.get(
                        "selected_safest_direction",
                        torch.zeros((), device=agent.device),
                    ).detach().cpu()
                )
                metrics["selected_safest_direction_rate"] = (
                    selected_safest_choices / max(eligible_safe_choices, 1.0)
                )
                for name, value in diagnostic_maxima.items():
                    current = float(value.detach().cpu())
                    if name in p4_cumulative_counter_names:
                        previous = p4_cumulative_counter_previous.get(name, 0.0)
                        metrics[name] = max(0.0, current - previous)
                        metrics[f"{name}_lifetime"] = current
                        p4_cumulative_counter_previous[name] = current
                    else:
                        metrics[name] = current
            valid_count = float(diagnostic_valid_count.detach().cpu())
            if valid_count > 0.0:
                metrics.update(
                    {
                        name: float(value.detach().cpu()) / valid_count
                        for name, value in diagnostic_valid_sums.items()
                    }
                )
            metrics.update(_curriculum_metrics(curriculum_snapshot))
            for name, buffer in quantile_samples.items():
                sample = buffer.reshape(-1)
                sample = sample[torch.isfinite(sample)]
                if not sample.numel():
                    continue
                for label, quantile in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
                    metrics[f"{name}_{label}"] = float(torch.quantile(sample.float(), quantile))
            for vx_index in range(5):
                for wz_index in range(4):
                    count = float(command_bin_counts[vx_index, wz_index])
                    prefix = f"cmd_v{vx_index}_w{wz_index}"
                    metrics[f"{prefix}_count"] = count
                    metrics[f"{prefix}_share"] = count / max(1.0, diagnostic_count)
                    denominator = max(1.0, count)
                    for suffix, source in (
                        ("progress", command_bin_progress),
                        ("tracking_mae", command_bin_tracking),
                        ("target_vx", command_bin_target_vx),
                        ("exec_vx", command_bin_exec_vx),
                        ("true_vx", command_bin_true_vx),
                        ("target_abs_wz", command_bin_target_wz),
                        ("exec_abs_wz", command_bin_exec_wz),
                        ("true_abs_wz", command_bin_true_wz),
                        ("gait_penalty", command_bin_gait),
                        ("success", command_bin_success),
                        ("failure", command_bin_failure),
                        ("timeout", command_bin_timeout),
                        ("vy_nonzero_share", command_bin_vy_nonzero),
                        ("target_abs_vy", command_bin_abs_vy),
                        ("vy_outer_share", command_bin_vy_outer),
                    ):
                        metric_denominator = denominator
                        if suffix == "tracking_mae":
                            metric_denominator = max(
                                1.0,
                                float(command_bin_tracking_counts[vx_index, wz_index]),
                            )
                        metrics[f"{prefix}_{suffix}"] = float(
                            source[vx_index, wz_index]
                        ) / metric_denominator
            for index, label in enumerate(segment_metric_labels):
                denominator = max(1.0, float(row_counts[index]))
                metrics[f"{label}_sample_share"] = float(row_counts[index]) / max(
                    1.0, float(segment_diagnostic_count)
                )
                for suffix, source in (
                    ("progress_mps", row_progress),
                    ("target_vx", row_target_vx),
                    ("exec_vx", row_exec_vx),
                    ("true_vx", row_true_vx),
                    ("target_abs_wz", row_target_wz),
                    ("stop_ratio", row_stop),
                    ("creep_ratio", row_creep),
                    ("spin_ratio", row_spin),
                    ("gait_penalty", row_gait),
                    ("success", row_success),
                    ("failure", row_failure),
                    ("timeout", row_timeout),
                    ("reason4", row_reason4),
                    ("predictive_clearance_m", row_predictive_clearance),
                    ("predictive_risk", row_predictive_risk),
                    ("predictive_penalty", row_predictive_penalty),
                    ("teacher_risk", row_teacher_risk),
                    ("teacher_high_risk_rate", row_teacher_high_risk),
                ):
                    metrics[f"{label}_{suffix}"] = float(source[index]) / denominator
            if is_p4:
                reset_event_count = float(p4_spawn_reset_event_count)
                segment_event_count = float(
                    p4_spawn_safe_point_event_count
                    + p4_spawn_hard_position_event_count
                )
                metrics["spawn_reset_event_count"] = float(
                    p4_spawn_reset_event_count
                )
                metrics["spawn_full_start_event_share"] = _event_share(
                    p4_spawn_full_start_event_count, reset_event_count
                )
                metrics["spawn_segment_start_event_share"] = _event_share(
                    segment_event_count, reset_event_count
                )
                metrics["spawn_safe_point_event_share"] = _event_share(
                    p4_spawn_safe_point_event_count, segment_event_count
                )
                metrics["spawn_hard_position_event_share"] = _event_share(
                    p4_spawn_hard_position_event_count, segment_event_count
                )
                for quartile_index in range(4):
                    metrics[f"spawn_position_q{quartile_index + 1}_event_share"] = (
                        _event_share(
                            p4_spawn_quartile_event_counts[quartile_index],
                            segment_event_count,
                        )
                    )
                metrics["episode_start_count"] = float(
                    p4_episode_start_counts.sum()
                )
                for axis in ("vy", "wz"):
                    for sign in ("positive", "negative"):
                        denominator = max(
                            1.0, float(signed_chain_counts[axis][sign])
                        )
                        for stage in (
                            "policy_target",
                            "limited_target",
                            "mapped_cmd",
                            "exec",
                            "true",
                        ):
                            metrics[f"{stage}_{axis}_{sign}_mean"] = float(
                                signed_chain_sums[axis][sign][stage]
                            ) / denominator
                for index, label in enumerate(segment_metric_labels):
                    metrics[f"spawn_segment_{label}_event_share"] = _event_share(
                        p4_episode_start_counts[index], reset_event_count
                    )
                    metrics[f"episode_start_{label}_count"] = float(
                        p4_episode_start_counts[index]
                    )
                    outcome_total = max(
                        1.0, float(p4_spawn_outcomes[index].sum())
                    )
                    for outcome_index, suffix in enumerate(
                        ("success", "failure", "timeout", "reason4")
                    ):
                        metrics[f"spawn_segment_{label}_{suffix}_rate"] = (
                            float(p4_spawn_outcomes[index, outcome_index])
                            / outcome_total
                        )
                    current_outcome_total = max(
                        1.0, float(p4_current_outcomes[index].sum())
                    )
                    for outcome_index, suffix in enumerate(
                        ("success", "failure", "timeout", "reason4")
                    ):
                        metrics[f"current_segment_{label}_{suffix}_rate"] = (
                            float(p4_current_outcomes[index, outcome_index])
                            / current_outcome_total
                        )
            metrics.update(
                {
                    "effective_training_seconds": algorithm.effective_training_seconds,
                    "session_effective_seconds": algorithm.session_effective_seconds,
                    "lifetime_effective_seconds": algorithm.lifetime_effective_seconds,
                    "lifecycle_success": float(lifecycle_success),
                    "platform_lifecycle_callbacks": float(lifecycle_success),
                    "lifecycle_failures": float(lifecycle_failures),
                    "rollout_time_s": now - rollout_started,
                    "env_step_time_s": env_step_time_s,
                    "samples_per_s": (
                        nav_rollout_ticks * agent.num_envs
                        / max(now - rollout_started, 1.0e-6)
                    ),
                    "cnn_unfrozen": float(algorithm.cnn_unfrozen),
                }
            )
            if is_p4:
                session_wall_seconds = float(algorithm.session_wall_seconds)
                while (
                    p4_recovery_event_times
                    and p4_recovery_event_times[0] < session_wall_seconds - 60.0
                ):
                    p4_recovery_event_times.popleft()
                metrics.update(
                    {
                        "session_wall_seconds": session_wall_seconds,
                        "near_goal_capture_candidate_count": float(
                            p4_capture_candidate_count
                        ),
                        "near_goal_capture_entry_count": float(p4_capture_entry_count),
                        "near_goal_capture_exit_count": float(p4_capture_exit_count),
                        "near_goal_capture_zone_success_count": float(
                            p4_capture_zone_success_count
                        ),
                        "near_goal_capture_zone_collision_count": float(
                            p4_capture_zone_collision_count
                        ),
                        "near_goal_capture_zone_timeout_count": float(
                            p4_capture_zone_timeout_count
                        ),
                        "near_goal_capture_zone_reset_count": float(
                            p4_capture_zone_reset_count
                        ),
                        "near_goal_capture_reset_counted_as_completion_error": float(
                            p4_capture_reset_completion_error_count
                        ),
                        "near_goal_capture_entry_to_platform_success_latency_s": (
                            sum(p4_capture_success_latency_s)
                            / max(len(p4_capture_success_latency_s), 1)
                        ),
                        "recovery_event_count_60s": float(
                            len(p4_recovery_event_times)
                        ),
                        "recovery_event_lifetime_count": float(
                            p4_recovery_event_lifetime_count
                        ),
                        "recovery_candidate_entry_count": float(
                            p4_recovery_candidate_entry_count
                        ),
                        "recovery_success_count": float(p4_recovery_success_count),
                        "recovery_terminal_count": float(p4_recovery_terminal_count),
                        "recovery_unverified_exit_count": float(
                            p4_recovery_unverified_exit_count
                        ),
                        "recovery_success_rate": float(p4_recovery_success_count)
                        / max(
                            p4_recovery_success_count + p4_recovery_terminal_count,
                            1,
                        ),
                        "recovery_time_s": sum(p4_recovery_time_s)
                        / max(len(p4_recovery_time_s), 1),
                        "recovery_early_stuck_sample_share": float(
                            p4_recovery_early_sample_count
                        )
                        / max(nav_rollout_ticks * agent.num_envs, 1),
                        "recovery_confirmed_stuck_sample_share": float(
                            p4_recovery_confirmed_sample_count
                        )
                        / max(nav_rollout_ticks * agent.num_envs, 1),
                        "recovery_safe_exit_share": float(p4_recovery_success_count)
                        / max(p4_recovery_candidate_entry_count, 1),
                        "recovery_candidate_lifetime_count": float(
                            p4_recovery_candidate_lifetime_count
                        ),
                        "recovery_terminal_lifetime_count": float(
                            p4_recovery_terminal_lifetime_count
                        ),
                    }
                )
                algorithm._p4_recovery_monitor_state = {
                    "event_times": list(p4_recovery_event_times),
                    "success_lifetime_count": p4_recovery_event_lifetime_count,
                    "candidate_lifetime_count": p4_recovery_candidate_lifetime_count,
                    "terminal_lifetime_count": p4_recovery_terminal_lifetime_count,
                }
            metrics["episode_starts_per_hour"] = (
                float(metrics.get("rollout_terminal_count", 0.0))
                * 3600.0
                / max(now - rollout_started, 1.0e-6)
            )
            if is_p4:
                for name in p4_contract.MONITOR_REQUIRED_METRICS:
                    value = metrics.get(name)
                    try:
                        finite = math.isfinite(float(value))
                    except (TypeError, ValueError):
                        finite = False
                    if finite:
                        p4_metric_last_seen[name] = now
                with_data = sum(
                    name in p4_metric_last_seen
                    for name in p4_contract.MONITOR_REQUIRED_METRICS
                )
                ages = [
                    now - timestamp for timestamp in p4_metric_last_seen.values()
                ]
                metrics.update(
                    p4_monitor_expected_metric_count=float(
                        len(p4_contract.MONITOR_REQUIRED_METRICS)
                    ),
                    p4_monitor_metric_with_data_count=float(with_data),
                    p4_monitor_empty_metric_count=float(
                        len(p4_contract.MONITOR_REQUIRED_METRICS) - with_data
                    ),
                    p4_monitor_longest_data_age_s=max(ages, default=0.0),
                )
            if now - last_log >= 60.0 or algorithm.current_iteration == 1:
                logger.info(
                    "[P2NavPPO] "
                    f"iter={algorithm.current_iteration} effective_min={algorithm.effective_training_seconds / 60.0:.1f} "
                    f"phase={algorithm.current_phase} actor={metrics.get('actor_loss', 0.0):.4f} "
                    f"critic={metrics.get('critic_loss', 0.0):.4f} adapter={metrics.get('adapter_loss', 0.0):.4f} "
                    f"micro_frames={algorithm.micro_sequences * 16} lifecycle={lifecycle_success} "
                    f"lifecycle_fail={lifecycle_failures} memory={algorithm.memory_metrics()}"
                )
                _monitor_put(monitor, metrics, logger)
                last_log = now
        _final_save(
            agent,
            logger,
            "p4_full_eight_session_hours_complete"
            if is_p4
            else "four_session_hours_complete",
        )
    except (SystemExit, KeyboardInterrupt) as exc:
        _final_save(agent, logger, type(exc).__name__)
        raise
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
