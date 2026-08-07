#!/usr/bin/env python3
"""Focused contracts for the shared P2/P4 workflow runtime."""

from pathlib import Path
from types import SimpleNamespace
import ast

import pytest

from agent_ppo.feature import p2_contract, p4_contract
from agent_ppo.p4 import profiles as p4_profiles
from agent_ppo.workflow.nav_ppo_support import MonitorHealthTracker
from agent_ppo.workflow.p2_nav_ppo_workflow import P2_WORKFLOW_SPEC
from agent_ppo.workflow.p4_nav_ppo_workflow import (
    P4_WORKFLOW_SPEC,
    P4WorkflowHooks,
)


def test_workflow_wrappers_expose_explicit_stage_specs():
    assert P2_WORKFLOW_SPEC.config_key == "p2_nav_ppo"
    assert P2_WORKFLOW_SPEC.expected_wire_dim == p2_contract.PRIVILEGED_WIRE_DIM
    assert P4_WORKFLOW_SPEC.config_key == "p4_nav_ppo"
    assert P4_WORKFLOW_SPEC.expected_wire_dim == p4_contract.P4_PRIVILEGED_WIRE_DIM


def test_monitor_health_splits_producer_and_upload_age_from_startup():
    tracker = MonitorHealthTracker(("ready", "missing"), started_at=10.0)
    first = tracker.observe_producer({"ready": 1.0}, now=13.0)
    assert first["p4_monitor_metric_with_data_count"] == 1.0
    assert first["p4_monitor_empty_metric_count"] == 1.0
    assert first["p4_monitor_producer_longest_age_s"] == pytest.approx(3.0)
    assert first["p4_monitor_longest_data_age_s"] == pytest.approx(3.0)
    assert first["p4_monitor_upload_age_s"] == pytest.approx(3.0)
    assert first["p4_monitor_upload_failure_count"] == 0.0

    tracker.record_upload(success=False, attempted=True, now=13.0)
    failed = tracker.observe_producer({"ready": 2.0}, now=15.0)
    assert failed["p4_monitor_producer_longest_age_s"] == pytest.approx(5.0)
    assert failed["p4_monitor_upload_age_s"] == pytest.approx(5.0)
    assert failed["p4_monitor_upload_failure_count"] == 1.0

    tracker.record_upload(success=True, attempted=True, now=15.0)
    delivered = tracker.observe_producer({"ready": 3.0}, now=17.0)
    assert delivered["p4_monitor_upload_age_s"] == pytest.approx(2.0)


def test_p4_profile_schedule_uses_registry_capability(monkeypatch):
    hooks = P4WorkflowHooks()
    algorithm = SimpleNamespace(training_profile="registry_profile")
    monkeypatch.setattr(
        "agent_ppo.workflow.p4_nav_ppo_workflow.p4_profiles.require_training_profile",
        lambda profile: SimpleNamespace(instant_command=profile == "registry_profile"),
    )


def _repair_profile_stage_conf():
    spec = p4_profiles.require_training_profile(
        p4_profiles.PROFILE_MAZE_INSTANT_REPAIR2H
    )
    return {
        "training_profile": spec.name,
        "run_name": spec.run_name,
        "target_effective_seconds": spec.target_effective_seconds,
        "task_end_hours": spec.task_end_hours,
        "stuck_reset": dict(p4_contract.STUCK_RESET_DEFAULTS),
    }


def test_p4_workflow_target_matches_the_active_profile_exactly():
    hooks = P4WorkflowHooks()
    stage_conf = _repair_profile_stage_conf()
    algorithm = SimpleNamespace(
        training_profile=stage_conf["training_profile"],
        stuck_reset_contract=stage_conf["stuck_reset"],
    )
    target = hooks.resolve_target(P4_WORKFLOW_SPEC, algorithm, stage_conf)
    assert target.target_seconds == 7200.0
    assert target.training_contract["run_name"] == stage_conf["run_name"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("run_name", "wrong-run"),
        ("target_effective_seconds", 28_800),
        ("task_end_hours", 8.25),
    ),
)
def test_p4_workflow_rejects_toml_clock_or_identity_drift(field, value):
    hooks = P4WorkflowHooks()
    stage_conf = _repair_profile_stage_conf()
    stage_conf[field] = value
    algorithm = SimpleNamespace(
        training_profile=stage_conf["training_profile"],
        stuck_reset_contract=stage_conf["stuck_reset"],
    )
    with pytest.raises(ValueError, match=field):
        hooks.resolve_target(P4_WORKFLOW_SPEC, algorithm, stage_conf)
    assert (
        hooks.schedule_boundaries(algorithm)
        == p4_contract.INSTANT_REPAIR_CHECKPOINT_BOUNDARIES_SECONDS
    )


def _function_lengths(path):
    tree = ast.parse(path.read_text())
    return {
        node.name: node.end_lineno - node.lineno + 1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_runtime_architecture_delegates_all_variant_behavior_to_hooks():
    workflow_dir = Path(__file__).parents[1] / "workflow"
    runtime = (workflow_dir / "nav_ppo_runtime.py").read_text()
    p2_wrapper = (workflow_dir / "p2_nav_ppo_workflow.py").read_text()
    p4_wrapper = (workflow_dir / "p4_nav_ppo_workflow.py").read_text()

    assert len(runtime.splitlines()) <= 1000
    assert "maze_instant_repair2h" not in runtime
    assert "is_p4" not in runtime
    assert "is_p4" not in (workflow_dir / "nav_ppo_metrics.py").read_text()
    assert "while algorithm.session_effective_seconds" not in p2_wrapper
    assert "while algorithm.session_effective_seconds" not in p4_wrapper
    for name in ("nav_ppo_support.py", "nav_ppo_metrics.py"):
        assert len((workflow_dir / name).read_text().splitlines()) <= 1200
    for name in (
        "nav_ppo_runtime.py",
        "nav_ppo_support.py",
        "nav_ppo_metrics.py",
        "p2_nav_ppo_workflow.py",
        "p4_nav_ppo_workflow.py",
    ):
        lengths = _function_lengths(workflow_dir / name)
        assert max(lengths.values()) <= 150, (name, lengths)
