"""Static UI contract for the P4 five-segment full-Track dashboard."""

import ast
from pathlib import Path
import re


MONITOR_PATH = Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"


def _literal_assignment(name):
    module = ast.parse(MONITOR_PATH.read_text(encoding="utf-8"))
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


def _panels():
    return {
        panel_name: metrics
        for _label, panel_name, metrics in _literal_assignment(
            "P4_FULL_TRACK_PANEL_SPECS"
        )
    }


def test_p4_full_track_panel_labels_match_platform_validation_contract():
    labels = [
        label
        for label, _panel_name, _metrics in _literal_assignment(
            "P4_FULL_TRACK_PANEL_SPECS"
        )
    ]

    assert all(1 <= len(label) <= 20 for label in labels)
    assert all(re.fullmatch(r"[A-Za-z0-9_\-\u4e00-\u9fff ]+", label) for label in labels)


def test_p4_full_track_panels_cover_all_requested_diagnostic_boundaries():
    panels = _panels()

    assert _literal_assignment("P4_FULL_TRACK_SEGMENT_LABELS") == (
        "slope",
        "slope_inv",
        "stairs",
        "stairs_inv",
        "maze",
    )
    assert set(panels) >= {
        "p4_full_current_segments",
        "p4_full_spawn_segments",
        "p4_full_spawn_column_quartiles",
        "p4_full_safe_hard_coverage",
        "p4_full_reason4_retry_fallback",
        "p4_full_goal_bearing_clean_fault",
        "p4_full_goal_jump_components",
        "p4_full_legitimate_side_goal",
        "p4_full_segment_frontier_potential",
        "p4_full_open_straight_gate",
        "p4_full_open_straight_penalties",
        "p4_full_vy_positive_chain",
        "p4_full_vy_negative_chain",
        "p4_full_wz_positive_chain",
        "p4_full_wz_negative_chain",
        "p4_full_stuck_sliding_window",
        "p4_full_episode_starts",
        "p4_full_spawn_segment_outcomes",
        "p4_full_current_segment_outcomes",
    }


def test_p4_full_track_panels_keep_metric_scope_unambiguous_and_bounded():
    panels = _panels()
    metrics = [metric for panel_metrics in panels.values() for metric in panel_metrics]

    assert all(len(panel_metrics) <= 20 for panel_metrics in panels.values())
    assert all(len(panel_metrics) == len(set(panel_metrics)) for panel_metrics in panels.values())
    assert len(metrics) == len(set(metrics))
    assert panels["p4_full_spawn_column_quartiles"] == (
        "spawn_reset_event_count",
        "spawn_full_start_event_share",
        "spawn_segment_start_event_share",
        "spawn_position_q1_event_share",
        "spawn_position_q2_event_share",
        "spawn_position_q3_event_share",
        "spawn_position_q4_event_share",
    )
    assert panels["p4_full_goal_bearing_clean_fault"] == (
        "goal_bearing_clean_0_5_abs_rad",
        "goal_bearing_fault_0_5_abs_rad",
        "goal_bearing_clean_5_10_abs_rad",
        "goal_bearing_fault_5_10_abs_rad",
        "goal_bearing_clean_10_plus_abs_rad",
        "goal_bearing_fault_10_plus_abs_rad",
    )


def test_p4_full_track_outcomes_keep_spawn_and_current_segment_attribution_separate():
    panels = _panels()

    for attribution in ("spawn", "current"):
        assert panels[f"p4_full_{attribution}_segment_outcomes"] == tuple(
            f"{attribution}_segment_{segment}_{outcome}_rate"
            for segment in _literal_assignment("P4_FULL_TRACK_SEGMENT_LABELS")
            for outcome in ("success", "failure", "timeout", "reason4")
        )
    assert "level_count=20" in MONITOR_PATH.read_text(encoding="utf-8")
