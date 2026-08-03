#!/usr/bin/env python3
"""Offline regression tests for the Arena network exporter."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOL_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = TOOL_ROOT / "network_export.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(TOOL_ROOT))
SPEC = importlib.util.spec_from_file_location("arena_network_export_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def configure_runtime(root: Path) -> None:
    MODULE.RUNTIME_DIR = root
    MODULE.NETWORK_DIR = root / "network_capture"
    MODULE.SESSIONS_DIR = MODULE.NETWORK_DIR / "sessions"
    MODULE.INDEX_JSONL = MODULE.NETWORK_DIR / "index.jsonl"


class HarParsingTests(unittest.TestCase):
    def test_metric_and_log_requests_parse_from_har(self):
        metrics, logs = MODULE.parse_har_entries(FIXTURES / "sample.har")
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["query_names"], ["completed_count"])
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["query_type"], "query_log")
        self.assertEqual(MODULE.build_log_view(logs)["entry_count"], 1)

    def test_malformed_har_fails_explicitly(self):
        with self.assertRaises(json.JSONDecodeError):
            MODULE.parse_har_entries(FIXTURES / "malformed.har")

    def test_missing_har_fails_explicitly(self):
        with self.assertRaises(FileNotFoundError):
            MODULE.parse_har_entries(FIXTURES / "missing.har")

    def test_coverage_report_uses_page_inventory_and_metric_queries(self):
        metrics, _ = MODULE.parse_har_entries(FIXTURES / "sample.har")
        inventory = MODULE.build_page_metric_inventory(
            "训练进展 (1)\ncompleted count\n查看\n"
        )
        report = MODULE.build_coverage_report(inventory, metrics)
        self.assertEqual(report["page_metric_count"], 1)
        self.assertEqual(report["unique_query_name_count"], 1)


class ExitStatusTests(unittest.TestCase):
    def _args(self, *extra):
        return MODULE.build_parser().parse_args(list(extra))

    def test_fatal_error_returns_nonzero_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            configure_runtime(Path(temp_dir))
            with mock.patch.object(
                MODULE,
                "switch_to_monitor_tab",
                side_effect=MODULE.CaptureError("missing monitor tab"),
            ):
                status = MODULE.export_capture(self._args())
            self.assertEqual(status, 1)
            summaries = list(MODULE.SESSIONS_DIR.glob("*/summary.json"))
            self.assertEqual(len(summaries), 1)
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertFalse(summary["capture_success"])
            self.assertIn("missing monitor tab", summary["errors"])

    def test_empty_metrics_only_fail_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            configure_runtime(Path(temp_dir))
            patches = (
                mock.patch.object(MODULE, "switch_to_monitor_tab"),
                mock.patch.object(
                    MODULE,
                    "ensure_monitor_ready",
                    return_value=("https://example/p/v5/exp/monitor", {}),
                ),
                mock.patch.object(MODULE, "ensure_auto_refresh", return_value=True),
                mock.patch.object(MODULE, "prepare_dense_overview_view"),
                mock.patch.object(MODULE, "get_body_text", return_value=""),
                mock.patch.object(
                    MODULE,
                    "capture_requests",
                    return_value=([], [], "none", {}),
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                self.assertEqual(MODULE.export_capture(self._args()), 0)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                self.assertEqual(
                    MODULE.export_capture(self._args("--fail-on-empty-metrics")), 1
                )

    def test_cli_main_propagates_export_failure(self):
        with mock.patch.object(sys, "argv", ["network_export.py"]), mock.patch.object(
            MODULE, "export_capture", return_value=1
        ):
            self.assertEqual(MODULE.main(), 1)


class PreflightTests(unittest.TestCase):
    def test_preflight_is_offline_and_reports_missing_browser(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            configure_runtime(Path(temp_dir))
            with mock.patch.object(MODULE.shutil, "which", return_value=None):
                self.assertEqual(MODULE.offline_preflight(), 1)

    def test_preflight_checks_version_without_browser_navigation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            configure_runtime(Path(temp_dir))
            with mock.patch.object(
                MODULE.shutil, "which", return_value=sys.executable
            ), mock.patch.object(MODULE.subprocess, "run") as run:
                run.return_value = mock.Mock(
                    returncode=0,
                    stdout="Python test-version\n",
                    stderr="",
                )
                self.assertEqual(MODULE.offline_preflight(), 0)
                self.assertEqual(run.call_count, 6)
                self.assertEqual(
                    run.call_args_list[0].args[0], [sys.executable, "--version"]
                )
                for call in run.call_args_list[1:]:
                    self.assertEqual(call.args[0][0], sys.executable)
                    self.assertEqual(call.args[0][-1], "--help")


if __name__ == "__main__":
    unittest.main()
