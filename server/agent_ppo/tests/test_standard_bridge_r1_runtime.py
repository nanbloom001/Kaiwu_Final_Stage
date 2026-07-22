#!/usr/bin/env python3
"""Runtime contracts for STD-BRIDGE-R1 (PyTorch tests skip if unavailable)."""

import os
import sys
import tempfile
import unittest
import copy
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local docs-only environments
    torch = None


if torch is not None:
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )

    from agent_ppo.algorithm.algorithm_behavior_distill import (
        BEHAVIOR_DISTILL_FORMAT,
        PRIVILEGED_TEACHER_FORMAT,
        AlgorithmBehaviorDistill,
    )
    from agent_ppo.algorithm.algorithm_lbc import AlgorithmLBC
    from agent_ppo.model.actor_critic_encoder import ActorCriticEncoder
    from agent_ppo.model.vision_encoder import DmEncoder
    from agent_ppo.workflow.behavior_distill_workflow import (
        _evaluate_phase_quality,
        _phase_for_iteration,
        _platform_model_id,
        _resume_warnings,
        _validate_schedule,
    )


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class StandardBridgeR1RuntimeTests(unittest.TestCase):
    def _algorithm(self):
        student = ActorCriticEncoder(
            num_proprio=45,
            num_scan=256,
            num_critic_input=92,
            num_actions=12,
            critic_use_encoder=True,
            critic_scan_slice=(60, 316),
            encoder_hidden_dims=[512, 256],
            latent_dim=32,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        algorithm = AlgorithmBehaviorDistill(student=student, device="cpu")
        algorithm.load_teacher_state_dict(
            {
                key: value.detach().clone()
                for key, value in algorithm.teacher.state_dict().items()
            },
            source="unit-test",
            source_sha256="test-sha",
        )
        return algorithm

    def test_schedule_boundaries_are_exact(self):
        ends = [1500, 2250, 3000, 3750, 5000]
        ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        _validate_schedule(ends, ratios, 5000)
        self.assertEqual(
            [_phase_for_iteration(value, ends) for value in (0, 1499, 1500, 2249, 2250, 4999)],
            [0, 0, 1, 1, 2, 4],
        )
        self.assertEqual(_platform_model_id(1, 10288), 10289)
        self.assertEqual(_platform_model_id(1500, 10288), 11788)
        self.assertEqual(_platform_model_id(5000, 10288), 15288)

    def test_per_environment_driver_and_zero_weight_guard(self):
        algorithm = self._algorithm()
        obs = torch.randn(8, 301)
        batch = algorithm.prepare_update(obs)
        actions, selection = algorithm.select_driver_actions(
            batch, student_drive_probability=1.0, safety_threshold=float("inf")
        )
        self.assertEqual(tuple(actions.shape), (8, 12))
        self.assertTrue(bool(selection["requested_student"].all().item()))
        before = {
            key: value.detach().clone()
            for key, value in algorithm.student.state_dict().items()
        }
        metrics = algorithm.finish_update(batch, torch.zeros(8))
        self.assertEqual(metrics["weighted_sample_rate"], 0.0)
        self.assertTrue(
            all(
                torch.equal(before[key], value)
                for key, value in algorithm.student.state_dict().items()
            )
        )
        self.assertEqual(algorithm.teacher_max_abs_diff(), 0.0)

    def test_teacher_stays_frozen_after_weighted_update(self):
        algorithm = self._algorithm()
        batch = algorithm.prepare_update(torch.randn(8, 301))
        metrics = algorithm.finish_update(
            batch, torch.tensor([1.0, 0.25, 1.0, 0.0, 1.0, 0.25, 1.0, 1.0])
        )
        self.assertEqual(metrics["update_skipped_nonfinite"], 0.0)
        self.assertEqual(algorithm.teacher_max_abs_diff(), 0.0)

    def test_flat_teacher_shape_contract_is_strict(self):
        algorithm = self._algorithm()
        bad_state = {
            key: value.detach().clone()
            for key, value in algorithm.teacher.state_dict().items()
        }
        bad_state["actor.0.weight"] = bad_state["actor.0.weight"][:, :-1]
        with self.assertRaisesRegex(ValueError, "shape_mismatch"):
            algorithm.load_teacher_state_dict(
                bad_state, source="bad-unit-test", source_sha256="test-sha"
            )
        algorithm.expected_teacher_sha256 = "expected-sha"
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            algorithm.load_teacher_state_dict(
                algorithm.teacher.state_dict(),
                source="wrong-sha-unit-test",
                source_sha256="other-sha",
            )

    def test_checkpoint_round_trip_and_privileged_side_artifact(self):
        algorithm = self._algorithm()
        algorithm.current_iteration = 2250
        algorithm.total_steps = 1234
        algorithm.gradient_steps = 17
        algorithm.dagger_phase_index = 2
        algorithm.dagger_phase_iteration = 11
        algorithm.student_drive_probability = 0.5
        algorithm.safety_threshold = 0.42
        algorithm.training_status = "running"
        algorithm.set_run_metadata("config-sha", "commit-sha")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bridge.pkl"
            bridge_sha = algorithm.save(str(path))

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["format"], BEHAVIOR_DISTILL_FORMAT)
            self.assertIs(checkpoint["critic_trained"], False)
            resumed = self._algorithm()
            resumed.load_checkpoint_dict(
                checkpoint,
                source=str(path),
                load_optimizer=True,
                load_teacher=True,
                restore_rng=False,
            )
            self.assertEqual(resumed.current_iteration, 2250)
            self.assertEqual(resumed.dagger_phase_index, 2)
            self.assertEqual(resumed.dagger_phase_iteration, 11)
            self.assertEqual(resumed.student_drive_probability, 0.5)
            self.assertAlmostEqual(resumed.safety_threshold, 0.42)
            self.assertEqual(resumed.training_status, "running")

            side_path = Path(temp_dir) / "teacher.pkl"
            algorithm.save_privileged_teacher(
                str(side_path),
                bridge_sha,
                platform_model_id=12538,
                source_training_status="phase_passed",
            )
            side = torch.load(side_path, map_location="cpu", weights_only=False)
            self.assertEqual(side["format"], PRIVILEGED_TEACHER_FORMAT)
            self.assertIs(side["deployable"], False)
            self.assertEqual(side["model_spec"]["actor_input_dim"], 77)
            self.assertEqual(side["source_iteration"], 2250)
            self.assertEqual(side["source_training_status"], "phase_passed")
            self.assertEqual(side["platform_model_id"], 12538)

            lbc_target = SimpleNamespace(
                proprio_dim=45,
                scan_dim=256,
                latent_dim=32,
                goal_dim=0,
                teacher_encoder=DmEncoder(
                    input_dim=256, hidden_dims=(512, 256), output_dim=32
                ),
                teacher_actor=copy.deepcopy(algorithm.student.actor),
                teacher_loaded=False,
                teacher_source=None,
            )
            lbc_target._freeze_teacher = lambda: None
            AlgorithmLBC.load_teacher_state_dict(
                lbc_target, side, source=str(side_path)
            )
            self.assertTrue(lbc_target.teacher_loaded)
            self.assertEqual(lbc_target.teacher_source, str(side_path))

    def test_resume_drift_is_reported_without_blocking(self):
        resumed = SimpleNamespace(
            current_iteration=1500,
            config_sha256="old-config",
            training_status="running",
        )
        warnings = _resume_warnings(resumed, "new-config")
        self.assertTrue(any("changed config" in warning for warning in warnings))
        resumed.config_sha256 = "new-config"
        resumed.training_status = "blocked"
        warnings = _resume_warnings(resumed, "new-config")
        self.assertTrue(any("legacy checkpoint" in warning for warning in warnings))

    def test_two_quality_windows_produce_advisory_result(self):
        passing = {
            "action_cos": 0.99,
            "normalized_action_mse": 0.05,
            "nonfinite_rate": 0.0,
            "teacher_ood_rate": 0.0,
            "safety_takeover_rate": 0.01,
            "hard_termination_rate": 0.02,
            "action_l2_p95": 0.1,
        }
        passed, details = _evaluate_phase_quality(
            [passing.copy() for _ in range(100)],
            50,
            {},
            previous_hard_termination_rate=0.02,
        )
        self.assertTrue(passed)
        self.assertEqual(len(details["windows"]), 2)
        failing = [passing.copy() for _ in range(100)]
        for row in failing[50:]:
            row["action_cos"] = 0.90
        passed, details = _evaluate_phase_quality(failing, 50, {}, 0.02)
        self.assertFalse(passed)
        self.assertTrue(
            any("window2 action_cos" in reason for reason in details["reasons"])
        )

        nonfinite = [passing.copy() for _ in range(100)]
        nonfinite[-1]["action_cos"] = float("nan")
        passed, details = _evaluate_phase_quality(nonfinite, 50, {}, 0.02)
        self.assertFalse(passed)
        self.assertTrue(
            any("action_cos is not finite" in reason for reason in details["reasons"])
        )


if __name__ == "__main__":
    unittest.main()
