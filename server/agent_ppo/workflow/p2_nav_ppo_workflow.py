#!/usr/bin/env python3
"""Thin P2 wrapper around the shared navigation PPO runtime."""

import os

from agent_ppo.feature import p2_contract
from agent_ppo.workflow.nav_ppo_support import (
    MonitorHealthTracker,
    NavWorkflowHooks,
    WorkflowSpec,
    _CollisionTraceRecorder,
    _curriculum_metrics,
    _event_share,
    _expected_critic_wire_dim,
    _extract_step,
    _final_save,
    _frame_done_masks,
    _goal_history_stuck,
    _install_sigterm_handler,
    _monitor_put,
    _p4_conditional_metrics,
    _p4_recovery_event_masks,
    _p4_reset_completion_mismatch,
    _p4_worker_push_resume_offset,
    _p4_worker_stuck_resume_offset,
    _request_periodic_save,
    _resolve_terminal_outcome,
    _terminal_outcome_count,
    _terminal_safe_p4_critic_wire,
    _terminal_safe_segment,
    _terminal_safe_tensor,
    _tick_diagnostic_values,
)
from agent_ppo.workflow.nav_ppo_runtime import run_nav_ppo_workflow


P2_WORKFLOW_SPEC = WorkflowSpec(
    name="p2_nav_ppo",
    config_key="p2_nav_ppo",
    agent_flag="is_p2_nav",
    eval_flag="is_p2_nav_eval",
    expected_wire_dim=p2_contract.PRIVILEGED_WIRE_DIM,
    target_seconds=p2_contract.TRAINING_HOURS * 3600.0,
    target_hours=p2_contract.TRAINING_HOURS,
)


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    return run_nav_ppo_workflow(
        envs,
        agents,
        logger=logger,
        monitor=monitor,
        *args,
        spec=P2_WORKFLOW_SPEC,
        hooks=NavWorkflowHooks(),
        **kwargs,
    )


__all__ = [
    "MonitorHealthTracker",
    "NavWorkflowHooks",
    "P2_WORKFLOW_SPEC",
    "WorkflowSpec",
    "workflow",
]
