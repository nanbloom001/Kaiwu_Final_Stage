#!/usr/bin/env python3

from __future__ import annotations

import concurrent.futures
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from strict_contract import (  # noqa: E402
    ContractError,
    WatchdogState,
    apply_action_chain,
    band_energy,
    load_contract,
    normalize_depth,
    validate_onnx,
    verify_hashes,
)
from analyze_strict_log import analyze  # noqa: E402


class StrictSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract(ROOT)

    def test_artifact_hashes_and_exact_onnx_abi(self):
        self.assertEqual(len(verify_hashes(ROOT, self.contract)), 2)
        summary = validate_onnx(ROOT, self.contract)
        self.assertEqual(len(summary["inputs"]), 8)
        self.assertEqual(len(summary["outputs"]), 8)

    def test_fault_injection_rejects_modified_artifact_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "model.bin"
            artifact.write_bytes(b"expected")
            contract = {"checkpoint": {"path": "model.bin", "sha256": "0" * 64}}
            with self.assertRaisesRegex(ContractError, "SHA256 mismatch"):
                verify_hashes(root, contract)

    def test_fault_injection_rejects_modified_onnx_abi(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = ROOT / self.contract["onnx"]["path"]
            target = root / "policy.onnx"
            model = onnx.load(str(source))
            model.graph.output.pop()
            onnx.save(model, str(target))
            contract = json.loads(json.dumps(self.contract))
            contract["onnx"]["path"] = "policy.onnx"
            with self.assertRaisesRegex(ContractError, "output count mismatch"):
                validate_onnx(root, contract)

    def test_depth_golden_vector(self):
        actual = normalize_depth([0.0, -1.0, 1.0, 4.999, 5.0, np.nan, np.inf])
        np.testing.assert_allclose(actual, [0.0, 0.0, 0.2, 0.9998, 0.0, 0.0, 0.0], atol=1e-6)

    def test_action_chain_clip_slew_limits_and_feedback(self):
        offset = np.asarray(self.contract["action"]["offset"])
        raw = np.asarray([10.0, -10.0] + [0.2] * 10)
        result = apply_action_chain(raw, offset, self.contract)
        self.assertEqual(result.clipped_raw_action[0], 6.0)
        self.assertEqual(result.clipped_raw_action[1], -6.0)
        self.assertLessEqual(np.max(np.abs(result.applied_joint_target - offset)), 0.0500001)
        np.testing.assert_allclose(
            result.executed_raw_action,
            (result.applied_joint_target - offset) / 0.25,
        )
        self.assertTrue(result.safety_modified.any())

    def test_nonfinite_action_is_hard_rejected(self):
        offset = self.contract["action"]["offset"]
        with self.assertRaises(ContractError):
            apply_action_chain([np.nan] + [0.0] * 11, offset, self.contract)

    def test_watchdog_faults_immediately_and_freezes(self):
        state = WatchdogState().evaluate(
            {"lowstate": 51, "depth": 1, "policy_target": 1, "command": 1, "inference": 1},
            self.contract,
        )
        self.assertTrue(state.faulted)
        self.assertTrue(state.frozen)
        self.assertEqual(state.reason, "lowstate_stale")
        self.assertEqual(state.takeover, "Passive")

    def test_inference_deadline_miss_fault(self):
        state = WatchdogState().evaluate(
            {"lowstate": 1, "depth": 1, "policy_target": 1, "command": 1, "inference": 50},
            self.contract,
        )
        self.assertEqual(state.reason, "inference_deadline_miss")

    def test_depth_age_is_warning_only(self):
        state = WatchdogState().evaluate(
            {"lowstate": 1, "depth": float("inf"), "policy_target": 1,
             "command": 1, "inference": 1},
            self.contract,
        )
        self.assertFalse(state.faulted)
        self.assertEqual(state.reason, "none")
        self.assertEqual(
            self.contract["diagnostic_threshold_ms"]["depth_age_warning"], 150
        )

    def test_frequency_diagnostic_detects_alternation(self):
        values = np.tile([1.0, -1.0], 128)
        metrics = band_energy(values)
        self.assertGreater(metrics["15_25_hz"], metrics["5_10_hz"])
        self.assertGreater(metrics["alternating_sign_ratio"], 0.95)

    def test_concurrent_immutable_contract_reads(self):
        def read_once(_):
            return apply_action_chain([0.0] * 12, self.contract["action"]["offset"], self.contract).executed_raw_action.tolist()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(read_once, range(2000)))
        self.assertTrue(all(result == results[0] for result in results))

    def test_training_reference_identity_and_camera_contract(self):
        reference = self.contract["training_reference"]
        self.assertEqual(reference["checkpoint_sha256"], self.contract["checkpoint"]["sha256"])
        self.assertIn("torch.clamp(actions, -6.0, 6.0)", reference["workflow_action_transport"])
        extrinsics = self.contract["depth"]["extrinsics"]
        np.testing.assert_allclose(
            extrinsics["position_m"], [0.339871, 0.034697, 0.075010], atol=1e-9
        )
        self.assertFalse(extrinsics["calibrated"])
        depth = self.contract["depth"]
        self.assertEqual(depth["shape"], [1, 180, 320, 1])
        self.assertEqual(depth["acquisition"]["preferred_profile"], [424, 240, 30])
        self.assertEqual(
            depth["acquisition"]["allowed_fallback_profiles"], [[480, 270, 30]]
        )
        self.assertEqual(self.contract["network_goal_dim"], 0)
        self.assertIn("modules.high_level.actor", self.contract["ignored_modules"])

    def test_torque_warning_only_until_hard_fault(self):
        config_path = ROOT / self.contract["configuration"]["controller"]["path"]
        with config_path.open("r", encoding="utf-8") as handle:
            strict = yaml.safe_load(handle)["FSM"]["VisionLoco"]["strict_safety"]
        self.assertEqual(strict["tau_soft_stop"], strict["tau_hard_fault"])
        self.assertEqual(strict["tau_hard_fault"][:8], [22.0] * 8)
        self.assertEqual(strict["tau_hard_fault"][8:], [43.0] * 4)
        self.assertTrue(
            all(warning < hard for warning, hard in zip(
                strict["tau_warning"], strict["tau_hard_fault"]
            ))
        )

    def test_depth_camera_density_and_fault_capture_config(self):
        config_path = ROOT / self.contract["configuration"]["controller"]["path"]
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        depth = config["FSM"]["VisionLoco"]["depth"]
        self.assertNotIn("max_invalid_fraction", depth)
        self.assertNotIn("max_front_invalid_fraction", depth)
        self.assertEqual(depth["camera_options"]["visual_preset"], "high_density")
        self.assertTrue(depth["camera_options"]["emitter_enabled"])
        self.assertEqual(depth["camera_options"]["laser_power"], "max")
        self.assertTrue(depth["camera_options"]["auto_exposure"])
        strict = config["FSM"]["VisionLoco"]["strict_safety"]
        self.assertTrue(strict["depth_fault_capture_enabled"])
        self.assertEqual(strict["depth_fault_capture_frames"], 90)
        self.assertFalse(strict["depth_capture_dump_on_exit"])
        self.assertFalse(strict["startup_shadow_enabled"])
        self.assertEqual(self.contract["depth"]["normalization"],
                         "meters in (0,5) divided by 5; <=0, >=5, NaN and Inf map to 0")
        self.assertIn("warning-only", self.contract["safety"]["depth_health_policy"])

    def test_persistent_fault_diagnostics_are_part_of_contract(self):
        logging_contract = self.contract["safety"]["persistent_fault_log"]
        self.assertIn("strict_controller_TIMESTAMP.log", logging_contract)
        source_root = (ROOT / self.contract["configuration"]["controller"]["path"]).parent.parent
        main_source = (source_root / "main.cpp").read_text(encoding="utf-8")
        state_source = (source_root / "src" / "State_VisionLoco.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("basic_file_sink_mt", main_source)
        self.assertNotIn("[VisionLoco][DEPTH_FAULT]", state_source)
        for marker in (
            "[VisionLoco][FAULT_SNAPSHOT]",
            "[VisionLoco][MECHANICAL_LIMIT]",
            "[VisionLoco][DEPTH_WARNING]",
            "[VisionLoco][LOWSTATE_FAULT]",
            "[VisionLoco][POLICY_TARGET_STALE]",
            "[VisionLoco][BAD_ORIENTATION]",
        ):
            self.assertIn(marker, state_source)

    def test_log_analyzer_separates_raw_and_applied_frequency(self):
        names = ["frame", "safety_modified_rate"]
        for prefix in ("model_raw_action", "clipped_raw_action", "requested_target", "applied_target", "q", "dq", "tau"):
            names.extend(f"{prefix}{i}" for i in range(12))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "strict.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=names)
                writer.writeheader()
                for frame in range(256):
                    raw = 2.0 if frame % 2 else -2.0
                    row = {"frame": frame, "safety_modified_rate": 1.0}
                    for i in range(12):
                        row[f"model_raw_action{i}"] = raw if i in (0, 4, 8) else 0.01
                        row[f"clipped_raw_action{i}"] = row[f"model_raw_action{i}"]
                        row[f"requested_target{i}"] = 0.25 * row[f"clipped_raw_action{i}"]
                        row[f"applied_target{i}"] = 0.02 * raw if i in (0, 4, 8) else 0.0
                        row[f"q{i}"] = row[f"applied_target{i}"]
                        row[f"dq{i}"] = 0.0
                        row[f"tau{i}"] = 0.0
                    writer.writerow(row)
            report = analyze(str(path))
            self.assertEqual(report["layers"]["model_raw_action"]["worst_leg"], "FL")
            self.assertGreater(
                report["layers"]["model_raw_action"]["joints"][0]["15_25_hz"], 1.0
            )
            self.assertLess(report["tracking_error_rad"]["max"], 1e-9)


if __name__ == "__main__":
    unittest.main()
