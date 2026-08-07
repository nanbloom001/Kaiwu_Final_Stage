#!/usr/bin/env python3
"""Shared P2/P4 Track semi-MDP PPO runtime."""

from __future__ import annotations

import math
import json
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import torch

from agent_ppo.checkpoint_io import CheckpointSaveError
from agent_ppo.conf.conf import Config
from agent_ppo.feature import nav_contract, p2_contract, p3_contract, p4_contract
from agent_ppo.feature.nav_event_log import emit_nav_event


@dataclass(frozen=True)
class WorkflowSpec:
    """Static wrapper contract for one navigation PPO stage."""

    name: str
    config_key: str
    agent_flag: str
    expected_wire_dim: int
    target_seconds: float
    target_hours: float
    eval_flag: str | None = None

    def accepts(self, agent) -> bool:
        if not bool(getattr(agent, self.agent_flag, False)):
            return False
        return not (self.eval_flag and bool(getattr(agent, self.eval_flag, False)))


@dataclass(frozen=True)
class WorkflowTarget:
    target_seconds: float
    target_hours: float
    training_contract: dict[str, object] | None = None


class NavWorkflowHooks:
    """Optional stage behavior injected by a thin workflow wrapper."""

    def resolve_target(self, spec: WorkflowSpec, algorithm, stage_conf: dict) -> WorkflowTarget:
        del algorithm, stage_conf
        return WorkflowTarget(spec.target_seconds, spec.target_hours)

    def configure_worker_resume(self, algorithm, stage_conf: dict, logger) -> None:
        del algorithm, stage_conf, logger

    def schedule_boundaries(self, algorithm) -> Sequence[float]:
        del algorithm
        return (
            p2_contract.SAFETY_WARM_END_SECONDS,
            p2_contract.SAFETY_STABILIZE_SECONDS,
        )

    def monitor_required_metrics(self) -> Sequence[str]:
        return ()

    def resumed_save_clock(self, resumed_seconds: float, resumed_clock_seconds: float) -> float:
        del resumed_clock_seconds
        return resumed_seconds

    def initialize_variant_state(self, agent, algorithm):
        del agent, algorithm
        return None

    def initialize_rollout_variant(self, agent, segment_count: int):
        del agent, segment_count
        return None

    def cumulative_counter_names(self) -> set[str]:
        return set()

    def make_terminal_extra(self, agent):
        del agent
        return None

    def frame_extra(self, algorithm):
        del algorithm
        return None

    def capture_terminal_extra(
        self, terminal_extra, frame_extra, next_critic, new_done
    ) -> None:
        del terminal_extra, frame_extra, next_critic, new_done

    def terminal_safe_critic_wire(self, critic_wire, terminal_extra, done):
        del terminal_extra, done
        return critic_wire

    def record_current_outcomes(
        self, variant, segment_valid, terminal_reason, valid_row_index
    ) -> None:
        del variant, segment_valid, terminal_reason, valid_row_index

    def after_tick(self, context) -> None:
        del context

    def extend_diagnostic_metrics(
        self, metrics: dict[str, float], state: dict
    ) -> None:
        del metrics, state

    def extend_rollout_metrics(
        self, metrics: dict[str, float], state: dict
    ) -> None:
        del metrics, state

    def advance_training_clocks(
        self,
        algorithm,
        *,
        now: float,
        rollout_started: float,
        session_started: float,
        resumed_seconds: float,
        resumed_clock_seconds: float,
        variant_training_seconds: float,
    ) -> float:
        del rollout_started, resumed_seconds
        algorithm.update_training_clocks(
            resumed_clock_seconds + (now - session_started)
        )
        return variant_training_seconds

    def record_cnn_unfreeze_boundary(self, saved_schedule_boundaries: set[float]) -> None:
        saved_schedule_boundaries.add(p2_contract.SAFETY_WARM_END_SECONDS)

    def save_clock(self, algorithm) -> float:
        return float(algorithm.session_effective_seconds)

    def after_update(self, algorithm, agent, nav_rollout_ticks: int) -> None:
        del algorithm, agent, nav_rollout_ticks

    def final_save_reason(self, target: WorkflowTarget) -> str:
        del target
        return "four_session_hours_complete"


@dataclass
class MonitorHealthTracker:
    """Keep producer freshness separate from monitor delivery health."""

    required_metrics: tuple[str, ...]
    started_at: float
    producer_last_seen: dict[str, float] = field(default_factory=dict)
    upload_last_success: float | None = None
    upload_failure_count: int = 0

    def observe_producer(self, metrics: dict[str, float], *, now: float) -> dict[str, float]:
        for name in self.required_metrics:
            value = metrics.get(name)
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if finite:
                self.producer_last_seen[name] = now
        with_data = sum(name in self.producer_last_seen for name in self.required_metrics)
        producer_ages = [
            now - self.producer_last_seen.get(name, self.started_at)
            for name in self.required_metrics
        ]
        producer_age = max(producer_ages, default=max(0.0, now - self.started_at))
        upload_age = max(
            0.0,
            now
            - (
                self.upload_last_success
                if self.upload_last_success is not None
                else self.started_at
            ),
        )
        return {
            "p4_monitor_expected_metric_count": float(len(self.required_metrics)),
            "p4_monitor_metric_with_data_count": float(with_data),
            "p4_monitor_empty_metric_count": float(len(self.required_metrics) - with_data),
            "p4_monitor_producer_longest_age_s": producer_age,
            "p4_monitor_upload_age_s": upload_age,
            "p4_monitor_upload_failure_count": float(self.upload_failure_count),
            # Compatibility alias retained until all platform panels migrate.
            "p4_monitor_longest_data_age_s": producer_age,
        }

    def record_upload(self, *, success: bool, attempted: bool, now: float) -> None:
        if not attempted:
            return
        if success:
            self.upload_last_success = now
        else:
            self.upload_failure_count += 1


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
    """Compatibility helper retained for older tests and tooling."""
    return {
        False: p2_contract.PRIVILEGED_WIRE_DIM,
        True: p4_contract.P4_PRIVILEGED_WIRE_DIM,
    }[bool(is_p4)]


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
        # The worker snapshot is authoritative for rows that actually reset.
        # A reason-0 reset is an unattributed episode boundary, not a timeout.
        timeout = torch.where(reset, worker_timeout, timeout)
        terminated = torch.where(reset, worker_hard, terminated)
    else:
        reset = torch.zeros_like(terminated)
    done = terminated | truncated | timeout | reset
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
        torch.zeros_like(raw_reason),
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


def _gait_diagnostic_values(response_aux: torch.Tensor) -> dict[str, torch.Tensor]:
    values = {}
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
    values["gait_window_valid"] = (
        response_aux[:, p2_contract.GAIT_VALID_INDEX] > 0.5
    ).float()
    return values


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
    values.update(_gait_diagnostic_values(response_aux))
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


def _advance_runtime_clocks_and_saves(
    *,
    agent,
    algorithm,
    logger,
    hooks: NavWorkflowHooks,
    rollout_started: float,
    session_started: float,
    resumed_seconds: float,
    resumed_clock_seconds: float,
    variant_training_seconds: float,
    unfreeze_save_done: bool,
    schedule_boundaries: Sequence[float],
    saved_schedule_boundaries: set[float],
    next_save_clock: float,
    save_interval_s: float,
    retry_save_at: float | None,
) -> tuple[float, float, bool, float, float | None]:
    now = time.monotonic()
    variant_training_seconds = hooks.advance_training_clocks(
        algorithm,
        now=now,
        rollout_started=rollout_started,
        session_started=session_started,
        resumed_seconds=resumed_seconds,
        resumed_clock_seconds=resumed_clock_seconds,
        variant_training_seconds=variant_training_seconds,
    )
    agent.training_elapsed_h = algorithm.effective_training_seconds / 3600.0
    unfrozen_now = algorithm.maybe_unfreeze_cnn(algorithm.session_effective_seconds)
    if unfrozen_now and not unfreeze_save_done:
        _request_periodic_save(agent, logger, "cnn_unfreeze_boundary")
        unfreeze_save_done = True
        hooks.record_cnn_unfreeze_boundary(saved_schedule_boundaries)
    for boundary in schedule_boundaries:
        if (
            boundary <= algorithm.session_effective_seconds
            and boundary not in saved_schedule_boundaries
        ):
            _request_periodic_save(agent, logger, f"schedule_boundary_{int(boundary)}s")
            saved_schedule_boundaries.add(boundary)
    save_clock = hooks.save_clock(algorithm)
    due = save_clock >= next_save_clock
    retry_due = retry_save_at is not None and now >= retry_save_at
    if due or retry_due:
        if _request_periodic_save(agent, logger, "wall_clock" if due else "retry"):
            retry_save_at = None
            while next_save_clock <= save_clock:
                next_save_clock += save_interval_s
        else:
            retry_save_at = now + 60.0
    return (
        now,
        variant_training_seconds,
        unfreeze_save_done,
        next_save_clock,
        retry_save_at,
    )


def _maybe_report_monitor(metrics: dict[str, float], state: dict) -> float:
    now = state["now"]
    last_log = state["last_log"]
    algorithm = state["algorithm"]
    if now - last_log < 60.0 and algorithm.current_iteration != 1:
        return last_log
    logger = state["logger"]
    logger.info(
        "[P2NavPPO] "
        f"iter={algorithm.current_iteration} effective_min={algorithm.effective_training_seconds / 60.0:.1f} "
        f"phase={algorithm.current_phase} actor={metrics.get('actor_loss', 0.0):.4f} "
        f"critic={metrics.get('critic_loss', 0.0):.4f} adapter={metrics.get('adapter_loss', 0.0):.4f} "
        f"micro_frames={algorithm.micro_sequences * 16} lifecycle={state['lifecycle_success']} "
        f"lifecycle_fail={state['lifecycle_failures']} memory={algorithm.memory_metrics()}"
    )
    monitor = state["monitor"]
    upload_success = _monitor_put(monitor, metrics, logger)
    upload_now = time.monotonic()
    monitor_health = state["monitor_health"]
    if monitor_health is not None:
        monitor_health.record_upload(
            success=upload_success,
            attempted=monitor is not None,
            now=upload_now,
        )
    return upload_now
