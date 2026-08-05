"""Static UI contract for the P4 five-segment full-Track dashboard."""

import ast
import importlib.util
from pathlib import Path
import re
import sys
import types


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


class _FakeMonitorConfigBuilder:
    def __init__(self):
        self.payload = {"title": "", "groups": []}
        self.group = None
        self.panel = None

    def title(self, value):
        self.payload["title"] = value
        return self

    def add_group(self, *, group_name, group_name_en):
        self.group = {
            "group_name": group_name,
            "group_name_en": group_name_en,
            "panels": [],
        }
        self.payload["groups"].append(self.group)
        return self

    def add_panel(self, *, name, name_en, type):
        self.panel = {
            "name": name,
            "name_en": name_en,
            "type": type,
            "metrics": [],
        }
        self.group["panels"].append(self.panel)
        return self

    def add_metric(self, *, metrics_name, expr):
        self.panel["metrics"].append({"metrics_name": metrics_name, "expr": expr})
        return self

    def end_panel(self):
        self.panel = None
        return self

    def end_group(self):
        self.group = None
        return self

    def build(self):
        return self.payload


def _load_monitor_builder():
    module_name = "_p4_monitor_builder_test"
    kaiwu = types.ModuleType("kaiwudrl")
    common = types.ModuleType("kaiwudrl.common")
    monitor = types.ModuleType("kaiwudrl.common.monitor")
    builder = types.ModuleType("kaiwudrl.common.monitor.monitor_config_builder")
    builder.MonitorConfigBuilder = _FakeMonitorConfigBuilder
    previous = {
        name: sys.modules.get(name)
        for name in (
            "kaiwudrl",
            "kaiwudrl.common",
            "kaiwudrl.common.monitor",
            "kaiwudrl.common.monitor.monitor_config_builder",
        )
    }
    sys.modules["kaiwudrl"] = kaiwu
    sys.modules["kaiwudrl.common"] = common
    sys.modules["kaiwudrl.common.monitor"] = monitor
    sys.modules["kaiwudrl.common.monitor.monitor_config_builder"] = builder
    try:
        spec = importlib.util.spec_from_file_location(module_name, MONITOR_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


def test_p4_monitor_build_routes_full_track_and_credit_profiles():
    module = _load_monitor_builder()
    full = module._build_p4_monitor("full_track")
    credit = module._build_p4_monitor("maze_credit_repair")
    full_groups = {group["group_name_en"] for group in full["groups"]}
    credit_groups = {group["group_name_en"] for group in credit["groups"]}
    assert full["title"] == "P4五段全赛道训练"
    assert "p4_full_track_diagnostics" in full_groups
    assert credit["title"] == "P4迷宫归因修复训练"
    assert "p4_full_track_diagnostics" not in credit_groups
