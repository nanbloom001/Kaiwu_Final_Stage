#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "realsense_latency_test", TOOLS / "realsense_latency_test.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DepthLatencyTest(unittest.TestCase):
    def test_summary_reports_average_maximum_and_percentiles(self):
        summary = MODULE.summarize([10.0, 20.0, 30.0, float("nan")])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["average_ms"], 20.0)
        self.assertEqual(summary["maximum_ms"], 30.0)
        self.assertEqual(summary["p50_ms"], 20.0)

    def test_yaml_merges_defaults_and_preserves_requested_filter(self):
        payload = {
            "filters": {"temporal": {"enabled": True, "persistence": 7}},
            "test": {"duration_seconds": 10},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latency.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            config = MODULE.load_config(path)
        self.assertEqual(config["camera"]["fps"], 30)
        self.assertTrue(config["filters"]["temporal"]["enabled"])
        self.assertEqual(config["filters"]["temporal"]["persistence"], 7)
        self.assertEqual(config["test"]["duration_seconds"], 10)

    def test_unknown_sensor_option_is_rejected(self):
        payload = {"sensor_options": {"not_a_real_option": 1}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latency.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported sensor_options"):
                MODULE.load_config(path)

    def test_filter_error_identifies_parameter_and_allowed_range(self):
        config = MODULE.merge_dict(
            MODULE.DEFAULT_CONFIG,
            {"filters": {"hole_fill": {"enabled": True, "mode": 3}}},
        )
        with self.assertRaisesRegex(
            ValueError,
            r"filters\.hole_fill\.mode=3.*allowed \[0, 2\]",
        ):
            MODULE.configure_filters(config)


if __name__ == "__main__":
    unittest.main()
