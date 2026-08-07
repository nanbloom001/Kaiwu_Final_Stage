#!/usr/bin/env python3
"""Thin P4 wrapper and hooks for the shared navigation PPO runtime."""

from collections import deque
from types import SimpleNamespace
import time

import torch

from agent_ppo.feature import p3_contract, p4_contract
from agent_ppo.feature.nav_event_log import emit_nav_event
from agent_ppo.p4 import profiles as p4_profiles
from agent_ppo.workflow.nav_ppo_support import (
    NavWorkflowHooks,
    WorkflowSpec,
    WorkflowTarget,
    _p4_worker_push_resume_offset,
    _p4_worker_stuck_resume_offset,
    _p4_recovery_event_masks,
    _p4_reset_completion_mismatch,
    _terminal_safe_p4_critic_wire,
)
from agent_ppo.workflow.nav_ppo_metrics import (
    add_p4_diagnostic_metrics,
    add_p4_recovery_metrics,
    add_p4_spawn_metrics,
)
from agent_ppo.workflow.nav_ppo_runtime import run_nav_ppo_workflow


def _diagnostic(context, name, fill=0.0):
    runtime = context.runtime
    return runtime.algorithm.last_tick_diagnostics.get(
        name,
        torch.full(
            (runtime.agent.num_envs,),
            float(fill),
            device=runtime.agent.device,
        ),
    ).reshape(-1)


def _record_spawn_outcomes(context):
    tick = context.tick
    variant = context.rollout.variant
    segment_count = len(context.runtime.segment_metric_labels)
    segment = _diagnostic(context, "segment_frontier_spawn_segment", -1.0)
    segment = segment.round().long()
    valid = (segment >= 0) & (segment < segment_count)
    index = segment.clamp(0, segment_count - 1)
    for outcome_index, reason_code in enumerate((1, 2, 3, 4)):
        mask = valid & (tick.terminal_reason == reason_code)
        variant.spawn_outcomes[:, outcome_index].scatter_add_(
            0, index, mask.float()
        )


def _record_spawn_events(context):
    tick = context.tick
    variant = context.rollout.variant
    segment_count = len(context.runtime.segment_metric_labels)
    new_episode = tick.transition_done
    full_start = _diagnostic(context, "spawn_full_start_share") > 0.5
    safe_point = _diagnostic(context, "spawn_safe_point_share") > 0.5
    quartile = _diagnostic(context, "spawn_position_quartile", -1.0).round().long()
    segment = _diagnostic(context, "spawn_segment_index", -1.0).round().long()
    segment_valid = new_episode & (segment >= 0) & (segment < segment_count)
    variant.episode_start_counts.scatter_add_(
        0, segment.clamp(0, segment_count - 1), segment_valid.float()
    )
    variant.spawn_reset_event_count += new_episode.float().sum()
    variant.spawn_full_start_event_count += (new_episode & full_start).float().sum()
    segment_start = new_episode & ~full_start
    variant.spawn_safe_point_event_count += (segment_start & safe_point).float().sum()
    variant.spawn_hard_position_event_count += (
        segment_start & ~safe_point
    ).float().sum()
    quartile_valid = segment_start & (quartile >= 0) & (quartile < 4)
    variant.spawn_quartile_event_counts.scatter_add_(
        0, quartile.clamp(0, 3), quartile_valid.float()
    )


def _record_signed_chain(context):
    tick = context.tick
    variant = context.rollout.variant
    chain = {
        "policy_target": {
            axis: _diagnostic(context, f"policy_target_{axis}")
            for axis in ("vy", "wz")
        },
        "limited_target": {
            axis: _diagnostic(context, f"limited_target_{axis}")
            for axis in ("vy", "wz")
        },
        "mapped_cmd": {
            axis: _diagnostic(context, f"mapped_cmd_{axis}")
            for axis in ("vy", "wz")
        },
        "exec": {"vy": tick.executed[:, 1], "wz": tick.executed[:, 2]},
        "true": {
            "vy": tick.diagnostic_aux[:, 13],
            "wz": tick.diagnostic_aux[:, 14],
        },
    }
    for axis in ("vy", "wz"):
        policy_axis = chain["policy_target"][axis]
        for sign, mask in (
            ("positive", policy_axis > 1.0e-4),
            ("negative", policy_axis < -1.0e-4),
        ):
            variant.signed_chain_counts[axis][sign] += mask.float().sum()
            for stage, by_axis in chain.items():
                variant.signed_chain_sums[axis][sign][stage] += (
                    by_axis[axis] * mask.float()
                ).sum()


def _record_capture_state(context):
    runtime = context.runtime
    tick = context.tick
    state = runtime.variant_state
    rollout = context.rollout.variant
    capture_active = _diagnostic(context, "near_goal_capture_active") > 0.5
    capture_candidate = _diagnostic(context, "near_goal_capture_candidate") > 0.5
    entry = capture_active & ~state.capture_active
    exit_mask = state.capture_active & (~capture_active | tick.transition_done)
    capture_zone = state.capture_active | capture_active
    success = capture_zone & (tick.terminal_reason == 1)
    timeout = capture_zone & (tick.terminal_reason == 3)
    assigned_reset = tick.terminal_reason == 4
    reset = capture_zone & assigned_reset
    collision = capture_zone & (tick.terminal_reason == 2)
    rollout.capture_candidate_count += int(capture_candidate.sum().item())
    rollout.capture_entry_count += int(entry.sum().item())
    rollout.capture_exit_count += int(exit_mask.sum().item())
    rollout.capture_zone_success_count += int(success.sum().item())
    rollout.capture_zone_collision_count += int(collision.sum().item())
    rollout.capture_zone_timeout_count += int(timeout.sum().item())
    rollout.capture_zone_reset_count += int(reset.sum().item())
    rollout.capture_reset_completion_error_count += int(
        _p4_reset_completion_mismatch(
            assigned_reset, tick.terminal_reason == 1
        ).sum().item()
    )
    entry_ticks = torch.where(
        entry,
        torch.full_like(state.capture_entry_tick, state.tick_index),
        state.capture_entry_tick,
    )
    success_entries = entry_ticks[success]
    rollout.capture_success_latency_s.extend(
        (
            (state.tick_index - success_entries + 1).clamp_min(0)
            .float()
            .mul(runtime.nav_dt_s)
            .detach()
            .cpu()
            .tolist()
        )
    )
    state.capture_entry_tick[entry] = state.tick_index
    state.capture_entry_tick[tick.transition_done] = -1
    state.capture_active = capture_active & ~tick.transition_done


def _recovery_inputs(context):
    tick = context.tick
    candidate = _diagnostic(context, "wall_stuck_candidate_share") > 0.5
    duration = _diagnostic(context, "wall_stuck_duration_s")
    mapping_valid = _diagnostic(context, "wall_stuck_mapping_valid") > 0.5
    true_xy = tick.diagnostic_aux[:, 12:14]
    progress = tick.start_goal - tick.end_goal
    evidence_valid = (
        mapping_valid
        & torch.isfinite(true_xy).all(dim=-1)
        & torch.isfinite(progress)
    )
    escape = (torch.linalg.vector_norm(true_xy, dim=-1) > 0.08) | (progress > 0.02)
    return candidate, duration, evidence_valid, escape


def _record_recovery_counts(rollout, state, masks):
    entry = masks["entry"]
    success = masks["success"]
    terminal = masks["terminal"]
    rollout.recovery_candidate_entry_count += int(entry.sum().item())
    rollout.recovery_success_count += int(success.sum().item())
    rollout.recovery_terminal_count += int(terminal.sum().item())
    rollout.recovery_unverified_exit_count += int(
        masks["unverified_exit"].sum().item()
    )
    rollout.recovery_early_sample_count += int(masks["early"].sum().item())
    rollout.recovery_confirmed_sample_count += int(
        masks["confirmed"].sum().item()
    )
    state.recovery_candidate_lifetime_count += int(entry.sum().item())
    state.recovery_terminal_lifetime_count += int(terminal.sum().item())


def _record_recovery_state(context):
    runtime = context.runtime
    tick = context.tick
    state = runtime.variant_state
    rollout = context.rollout.variant
    candidate, duration, evidence_valid, escape = _recovery_inputs(context)
    masks = _p4_recovery_event_masks(
        state.recovery_active,
        state.recovery_awaiting_evidence,
        candidate,
        duration,
        tick.transition_done,
        evidence_valid,
        escape,
        tick.terminal_reason == 1,
    )
    _record_recovery_counts(rollout, state, masks)
    entry_ticks = torch.where(
        masks["entry"],
        torch.full_like(state.recovery_entry_tick, state.tick_index),
        state.recovery_entry_tick,
    )
    successful_entries = entry_ticks[masks["success"]]
    rollout.recovery_time_s.extend(
        (
            (state.tick_index - successful_entries + 1).clamp_min(0)
            .float()
            .mul(runtime.nav_dt_s)
            .detach()
            .cpu()
            .tolist()
        )
    )
    state.recovery_entry_tick[masks["entry"]] = state.tick_index
    state.recovery_entry_tick[masks["success"] | masks["terminal"]] = -1
    state.recovery_active = masks["next_active"]
    state.recovery_awaiting_evidence = masks["next_awaiting"]
    recovery_count = int(masks["success"].sum().item())
    if recovery_count:
        event_time = runtime.resumed_clock_seconds + (
            time.monotonic() - runtime.session_started
        )
        state.recovery_event_times.extend([event_time] * recovery_count)
        state.recovery_event_lifetime_count += recovery_count
    runtime.algorithm._p4_recovery_monitor_state = {
        "event_times": list(state.recovery_event_times),
        "success_lifetime_count": state.recovery_event_lifetime_count,
        "candidate_lifetime_count": state.recovery_candidate_lifetime_count,
        "terminal_lifetime_count": state.recovery_terminal_lifetime_count,
    }


class P4WorkflowHooks(NavWorkflowHooks):
    def resolve_target(self, spec, algorithm, stage_conf):
        training_profile = getattr(
            algorithm,
            "training_profile",
            stage_conf.get("training_profile", p4_profiles.DEFAULT_TRAIN_PROFILE),
        )
        profile_spec = p4_profiles.require_training_profile(training_profile)
        contract = p4_contract.training_contract(
            getattr(algorithm, "stuck_reset_contract", stage_conf.get("stuck_reset", {})),
            training_profile,
        )
        target_seconds = float(contract["target_effective_seconds"])
        configured_target = float(
            stage_conf.get("target_effective_seconds", target_seconds)
        )
        if configured_target != target_seconds:
            raise ValueError(
                "P4 target_effective_seconds conflicts with the active profile: "
                f"configured={configured_target} contract={target_seconds}"
            )
        configured_run_name = str(
            stage_conf.get("run_name", profile_spec.run_name)
        )
        if configured_run_name != profile_spec.run_name:
            raise ValueError(
                "P4 run_name conflicts with the active profile: "
                f"configured={configured_run_name!r} "
                f"profile={profile_spec.run_name!r}"
            )
        configured_wall_hours = float(
            stage_conf.get("task_end_hours", profile_spec.task_end_hours)
        )
        if abs(configured_wall_hours - profile_spec.task_end_hours) > 1.0e-9:
            raise ValueError(
                "P4 task_end_hours conflicts with the active profile: "
                f"configured={configured_wall_hours} "
                f"profile={profile_spec.task_end_hours}"
            )
        return WorkflowTarget(
            target_seconds=target_seconds,
            target_hours=float(contract["training_hours"]),
            training_contract=contract,
        )

    def configure_worker_resume(self, algorithm, stage_conf, logger):
        # EventManager receives its configuration through env.reset(). Resume
        # offsets keep worker-owned monotonic schedules aligned with the learner.
        push_schedule = stage_conf.setdefault("push_schedule", {})
        push_schedule["resume_offset_s"] = _p4_worker_push_resume_offset(algorithm)
        stuck_reset = stage_conf.setdefault("stuck_reset", {})
        stuck_reset["resume_offset_s"] = _p4_worker_stuck_resume_offset(algorithm)
        logger.info(
            "[P4Resume] worker push/stuck resume_offset_s=%.3f "
            "diagnostic_elapsed_s=%.3f",
            push_schedule["resume_offset_s"],
            float(getattr(algorithm, "diagnostic_elapsed_seconds", 0.0)),
        )

    def schedule_boundaries(self, algorithm):
        profile = p4_profiles.require_training_profile(algorithm.training_profile)
        if profile.instant_command:
            return p4_contract.INSTANT_REPAIR_CHECKPOINT_BOUNDARIES_SECONDS
        return p4_contract.SCHEDULE_BOUNDARIES_SECONDS

    def monitor_required_metrics(self):
        return p4_contract.MONITOR_REQUIRED_METRICS

    def resumed_save_clock(self, resumed_seconds, resumed_clock_seconds):
        del resumed_seconds
        return resumed_clock_seconds

    def initialize_variant_state(self, agent, algorithm):
        restored = getattr(algorithm, "_p4_recovery_monitor_state", {})
        return SimpleNamespace(
            capture_active=torch.zeros(
                agent.num_envs, dtype=torch.bool, device=agent.device
            ),
            capture_entry_tick=torch.full(
                (agent.num_envs,), -1, dtype=torch.long, device=agent.device
            ),
            recovery_active=torch.zeros(
                agent.num_envs, dtype=torch.bool, device=agent.device
            ),
            recovery_awaiting_evidence=torch.zeros(
                agent.num_envs, dtype=torch.bool, device=agent.device
            ),
            recovery_entry_tick=torch.full(
                (agent.num_envs,), -1, dtype=torch.long, device=agent.device
            ),
            tick_index=0,
            recovery_event_times=deque(
                float(value) for value in restored.get("event_times", ())
            ),
            recovery_event_lifetime_count=int(
                restored.get("success_lifetime_count", 0)
            ),
            recovery_candidate_lifetime_count=int(
                restored.get("candidate_lifetime_count", 0)
            ),
            recovery_terminal_lifetime_count=int(
                restored.get("terminal_lifetime_count", 0)
            ),
        )

    def initialize_rollout_variant(self, agent, segment_count):
        rows = torch.zeros(segment_count, device=agent.device)
        outcomes = torch.zeros(segment_count, 4, device=agent.device)
        signed_sums = {
            axis: {
                sign: {
                    stage: torch.zeros((), device=agent.device)
                    for stage in (
                        "policy_target", "limited_target", "mapped_cmd", "exec", "true"
                    )
                }
                for sign in ("positive", "negative")
            }
            for axis in ("vy", "wz")
        }
        signed_counts = {
            axis: {
                sign: torch.zeros((), device=agent.device)
                for sign in ("positive", "negative")
            }
            for axis in ("vy", "wz")
        }
        return SimpleNamespace(
            spawn_outcomes=outcomes,
            current_outcomes=torch.zeros_like(outcomes),
            episode_start_counts=torch.zeros_like(rows),
            spawn_reset_event_count=torch.zeros((), device=agent.device),
            spawn_full_start_event_count=torch.zeros((), device=agent.device),
            spawn_safe_point_event_count=torch.zeros((), device=agent.device),
            spawn_hard_position_event_count=torch.zeros((), device=agent.device),
            spawn_quartile_event_counts=torch.zeros(4, device=agent.device),
            signed_chain_sums=signed_sums,
            signed_chain_counts=signed_counts,
            capture_candidate_count=0,
            capture_entry_count=0,
            capture_exit_count=0,
            capture_zone_success_count=0,
            capture_zone_collision_count=0,
            capture_zone_timeout_count=0,
            capture_zone_reset_count=0,
            capture_reset_completion_error_count=0,
            capture_success_latency_s=[],
            recovery_candidate_entry_count=0,
            recovery_success_count=0,
            recovery_terminal_count=0,
            recovery_unverified_exit_count=0,
            recovery_early_sample_count=0,
            recovery_confirmed_sample_count=0,
            recovery_time_s=[],
        )

    def cumulative_counter_names(self):
        return {
            "spawn_reason4_retry_count",
            "spawn_reason4_exhausted_count",
            "spawn_reason4_fallback_applied_count",
            "spawn_all_position_applied_count",
            "spawn_validation_failure_count",
            "spawn_write_failure_count",
        }

    def make_terminal_extra(self, agent):
        return torch.zeros(
            agent.num_envs,
            p4_contract.P4_WORKER_EXTRA_DIM,
            device=agent.device,
        )

    def frame_extra(self, algorithm):
        return algorithm._p4_worker_extra.detach().clone()

    def capture_terminal_extra(
        self, terminal_extra, frame_extra, next_critic, new_done
    ):
        if terminal_extra is None or not bool(new_done.any()):
            return
        del frame_extra
        terminal_extra[new_done] = next_critic[
            new_done,
            p3_contract.P3_PRIVILEGED_WIRE_DIM :
            p4_contract.P4_PRIVILEGED_WIRE_DIM,
        ]

    def terminal_safe_critic_wire(self, critic_wire, terminal_extra, done):
        if not bool(done.any()):
            return critic_wire
        return _terminal_safe_p4_critic_wire(critic_wire, terminal_extra, done)

    def record_current_outcomes(
        self, variant, segment_valid, terminal_reason, valid_row_index
    ):
        for outcome_index, reason_code in enumerate((1, 2, 3, 4)):
            outcome_mask = segment_valid & (terminal_reason == reason_code)
            variant.current_outcomes[:, outcome_index].scatter_add_(
                0, valid_row_index, outcome_mask.float()
            )

    def after_tick(self, context):
        _record_spawn_outcomes(context)
        _record_spawn_events(context)
        _record_signed_chain(context)
        _record_capture_state(context)
        _record_recovery_state(context)
        context.runtime.variant_state.tick_index += 1

    def extend_diagnostic_metrics(self, metrics, state):
        add_p4_diagnostic_metrics(metrics, state)

    def extend_rollout_metrics(self, metrics, state):
        add_p4_spawn_metrics(metrics, state)
        add_p4_recovery_metrics(metrics, state)

    def advance_training_clocks(
        self,
        algorithm,
        *,
        now,
        rollout_started,
        session_started,
        resumed_seconds,
        resumed_clock_seconds,
        variant_training_seconds,
    ):
        variant_training_seconds += max(0.0, now - rollout_started)
        algorithm.update_training_clocks(
            resumed_clock_seconds + (now - session_started),
            session_effective_seconds=resumed_seconds + variant_training_seconds,
        )
        return variant_training_seconds

    def record_cnn_unfreeze_boundary(self, saved_schedule_boundaries):
        del saved_schedule_boundaries

    def save_clock(self, algorithm):
        return float(algorithm.session_wall_seconds)

    def after_update(self, algorithm, agent, nav_rollout_ticks):
        emit_nav_event(
            "iteration",
            role="aisrv",
            iteration=int(algorithm.current_iteration),
            valid_ticks=float(nav_rollout_ticks * agent.num_envs),
            update_skipped_no_valid=0.0,
        )

    def final_save_reason(self, target):
        if target.training_contract is None:
            raise RuntimeError("P4 workflow target is missing its training contract")
        return f"{target.training_contract['run_name']}_session_complete"


P4_WORKFLOW_SPEC = WorkflowSpec(
    name="p4_nav_ppo",
    config_key="p4_nav_ppo",
    agent_flag="is_p4_nav",
    expected_wire_dim=p4_contract.P4_PRIVILEGED_WIRE_DIM,
    target_seconds=p4_contract.TARGET_EFFECTIVE_SECONDS,
    target_hours=p4_contract.TRAINING_HOURS,
)


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    return run_nav_ppo_workflow(
        envs,
        agents,
        logger=logger,
        monitor=monitor,
        *args,
        spec=P4_WORKFLOW_SPEC,
        hooks=P4WorkflowHooks(),
        **kwargs,
    )


__all__ = ["P4_WORKFLOW_SPEC", "P4WorkflowHooks", "workflow"]
