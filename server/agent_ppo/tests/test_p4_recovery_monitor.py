"""Static contract checks for the P4 recovery monitor panels."""

import ast
from pathlib import Path


MONITOR_PATH = Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
CONTRACT_PATH = Path(__file__).resolve().parents[1] / "feature" / "p4_contract.py"


def _p4_groups():
    module = ast.parse(MONITOR_PATH.read_text(encoding="utf-8"))
    build_p4 = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_p4_monitor"
    )
    assignment = next(
        node
        for node in build_p4.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "groups"
            for target in node.targets
        )
    )
    return ast.literal_eval(assignment.value)


def _shared_groups():
    module = ast.parse(MONITOR_PATH.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "P2_MONITOR_GROUPS"
            for target in node.targets
        )
    )
    return ast.literal_eval(assignment.value)


def _contract_declaration(name):
    module = ast.parse(CONTRACT_PATH.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    )
    return ast.literal_eval(assignment.value)


def _panels_by_name():
    return {
        panel_name: metrics
        for _group_name, _group_name_en, panels in _p4_groups()
        for _name, panel_name, metrics in panels
    }


def test_p4_recovery_panels_keep_existing_window_and_lifetime_metric_keys():
    panels = _panels_by_name()

    assert panels["p4_push_event_scopes"] == (
        "push_rollout_event_count_total",
        "push_lifetime_event_count_total",
    )
    assert panels["p4_push_window"] == (
        "push_epoch",
        "push_event_count",
        "seconds_since_push",
        "push_runtime_active",
        "push_telemetry_valid",
        "push_actual_delta_vx_mean",
        "push_actual_delta_vy_mean",
        "push_actual_delta_vx_abs_max",
        "push_actual_delta_vy_abs_max",
    )
    assert panels["p4_push_lifetime"] == (
        "push_lifetime_count",
        "push_env_coverage",
        "push_env_coverage_rate",
    )
    assert panels["p4_stuck_window_outcome"] == (
        "wall_stuck_would_reset",
        "wall_stuck_reset_triggered",
        "wall_stuck_raw_term",
        "wall_stuck_reset_rate",
        "rollout_wall_stuck_reset_count",
        "wall_stuck_saved_seconds",
        "rollout_wall_stuck_saved_seconds",
        "collision_to_stuck_reset_delay_s",
        "reset_after_push_share",
        "episode_starts_per_hour",
    )

    shared_panels = {
        panel_name: metrics
        for _group_name, _group_name_en, group_panels in _shared_groups()
        for _name, panel_name, metrics in group_panels
    }
    assert "collision_onset_total" not in shared_panels["p2_collision_recovery"]
    assert shared_panels["p2_collision_lifetime"] == ("collision_onset_total",)


def test_p4_recovery_panels_explain_all_health_exemptions():
    panels = _panels_by_name()
    required_metrics = _contract_declaration("MONITOR_REQUIRED_METRICS")
    optional_metrics = _contract_declaration("MONITOR_OPTIONAL_METRICS")
    optional_prefixes = _contract_declaration("MONITOR_OPTIONAL_METRIC_PREFIXES")
    panel_metrics = {
        metric for metrics in panels.values() for metric in metrics
    }
    # The inherited P2 dashboard is intentionally outside the P4 health
    # denominator. Every metric declared by a P4-specific panel, however, must
    # be explicitly required or named as a narrow exemption.
    # Track outcome keys are added dynamically from TRACK_PANEL_SPECS.
    panel_metrics.update(
        {
            "completed_count_track_l",
            "abnormal_count_track_l",
            "timeout_count_track_l",
            "total_score_track_l",
            "energy_score_track_l",
            "pose_score_track_l",
            "time_score_track_l",
        }
    )
    missing = {
        metric
        for metric in panel_metrics
        if metric not in required_metrics
        and metric not in optional_metrics
        and not any(
            metric.startswith(prefix)
            for prefix in optional_prefixes
        )
    }
    assert not missing


def test_p4_recovery_panels_include_event_level_capture_and_recovery_metrics():
    panels = _panels_by_name()
    required_metrics = _contract_declaration("MONITOR_REQUIRED_METRICS")
    assert panels["p4_near_goal_capture_window"] == (
        "near_goal_capture_candidate_count",
        "near_goal_capture_entry_count",
        "near_goal_capture_exit_count",
        "near_goal_capture_zone_success_count",
        "near_goal_capture_zone_collision_count",
        "near_goal_capture_zone_timeout_count",
        "near_goal_capture_zone_reset_count",
        "near_goal_capture_reset_counted_as_completion_error",
    )
    assert panels["p4_near_goal_capture_platform_latency"] == (
        "near_goal_capture_entry_to_platform_success_latency_s",
    )
    assert panels["p4_recovery_events_60s"] == ("recovery_event_count_60s",)
    assert panels["p4_recovery_events_lifetime"] == (
        "recovery_event_lifetime_count",
    )
    assert panels["p4_recovery_phase_samples"] == (
        "recovery_early_stuck_sample_share",
        "recovery_confirmed_stuck_sample_share",
        "recovery_safe_exit_share",
    )
    assert panels["p4_recovery_window_outcome"] == (
        "recovery_candidate_entry_count",
        "recovery_success_count",
        "recovery_terminal_count",
        "recovery_unverified_exit_count",
        "recovery_success_rate",
        "recovery_time_s",
    )
    assert panels["p4_recovery_lifetime_outcome"] == (
        "recovery_candidate_lifetime_count",
        "recovery_event_lifetime_count",
        "recovery_terminal_lifetime_count",
    )
    assert "session_wall_seconds" in required_metrics


def test_p4_recovery_panels_remain_bounded_and_do_not_duplicate_metrics():
    panels = _panels_by_name()
    for name, metrics in panels.items():
        assert len(metrics) <= 20, name
        assert len(metrics) == len(set(metrics)), name
