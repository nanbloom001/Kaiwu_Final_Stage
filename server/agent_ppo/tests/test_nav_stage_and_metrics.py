#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Nav bootstrap/eval routing and episode-outcome metric regressions."""

import ast
import os
import pathlib
import unittest
from unittest import mock

import agent_ppo.tests._nav_test_stubs  # noqa: F401

import toml

from agent_ppo.conf.conf import (
    Config,
    LBCLocoConfig,
    NavDaggerConfig,
    NavEvalConfig,
    _configured_training_stage,
    _infer_stage_from_task_name,
)
from agent_ppo.workflow.nav_dagger_workflow import (
    _aggregate_goal_metrics,
    _callbacks_until_next_dump,
    _episode_outcome_rates,
    _quality_window_metrics,
)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


class TestNavStageSelection(unittest.TestCase):
    def test_active_branch_bootstraps_worker_and_aisrv_to_nav(self):
        self.assertIs(Config.CURRENT, NavDaggerConfig)

    def test_literal_stage_default_is_track_before_runtime_bootstrap(self):
        conf_path = pathlib.Path(__file__).resolve().parent.parent / "conf" / "conf.py"
        tree = ast.parse(conf_path.read_text(encoding="utf-8"))
        config_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Config"
        )
        current_assignment = next(
            node
            for node in config_class.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "CURRENT"
                for target in node.targets
            )
        )
        self.assertIsInstance(current_assignment.value, ast.Name)
        self.assertEqual(current_assignment.value.id, "NavDaggerConfig")

    def test_training_bootstrap_selects_nav_before_stage_toml(self):
        logger = _Logger()
        with mock.patch(
            "agent_ppo.conf.conf.toml.load",
            return_value={"app": {"policy_entry": "nav_dagger"}},
        ):
            self.assertIs(_configured_training_stage(logger), NavDaggerConfig)

    def test_training_bootstrap_rejects_unknown_stage(self):
        with mock.patch(
            "agent_ppo.conf.conf.toml.load",
            return_value={"app": {"policy_entry": "typo_nav"}},
        ):
            with self.assertRaises(ValueError):
                _configured_training_stage(_Logger())

    def test_track_camera_without_forwarded_entry_selects_nav_eval(self):
        usr_conf = {
            "env_conf": {"task_name": "Unitree-Go2-Velocity-Camera"},
            "terrain": {"mode": "track"},
        }
        self.assertIs(
            _infer_stage_from_task_name(usr_conf, _Logger()), NavEvalConfig
        )

    def test_standard_camera_still_selects_lbc(self):
        usr_conf = {
            "env_conf": {"task_name": "Unitree-Go2-Velocity-Camera"},
            "terrain": {"mode": "standard"},
        }
        self.assertIs(
            _infer_stage_from_task_name(usr_conf, _Logger()), LBCLocoConfig
        )

    def test_explicit_nav_dagger_is_eval_only_during_eval_inference(self):
        usr_conf = {
            "env_conf": {
                "task_name": "Unitree-Go2-Velocity-Camera",
                "policy_entry": "nav_dagger",
            },
            "terrain": {"mode": "track"},
        }
        self.assertIs(
            _infer_stage_from_task_name(usr_conf, _Logger()), NavEvalConfig
        )

    def test_active_track_config_has_no_fake_level_mix(self):
        config_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "conf"
            / "train_env_conf_track_nav_dagger.toml"
        )
        config = toml.load(config_path)
        self.assertNotIn("level_mix", config["terrain"])
        self.assertEqual(config["terrain"]["track"]["num_parallel_tracks"], 10)
        self.assertEqual(config["env"]["num_envs"], 128)
        for removed_key in (
            "save_interval",
            "initial_save_after_iterations",
            "save_interval_minutes",
        ):
            self.assertNotIn(removed_key, config["nav_dagger"])

    def test_production_platform_dump_frequency_is_about_five_minutes(self):
        config_path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "conf"
            / "configure_app.toml"
        )
        config = toml.load(config_path)
        self.assertEqual(config["app"]["dump_model_freq"], 3600)

    def test_monitor_builder_declares_track_outcomes_and_nav_metrics(self):
        monitor_source = (
            pathlib.Path(__file__).resolve().parent.parent
            / "conf"
            / "monitor_builder.py"
        ).read_text(encoding="utf-8")
        for metric in (
            "completed_count_track_l",
            "abnormal_count_track_l",
            "timeout_count_track_l",
            "total_score_track_l",
            "time_score_track_l",
            "ce_loss",
            "top1_accuracy",
            "goal_progress_m_per_frame",
            "platform_lifecycle_callbacks",
            "platform_lifecycle_failures",
            "total_low_level_steps",
            "total_env_frames",
            "callbacks_until_next_dump",
        ):
            self.assertIn(metric, monitor_source)
        self.assertNotIn("step_score_track_l", monitor_source)
        self.assertNotIn('name_en="value_loss"', monitor_source)


class TestNavPlatformDumpCadence(unittest.TestCase):
    def test_callbacks_until_next_dump_boundaries(self):
        self.assertEqual(_callbacks_until_next_dump(0, 3600), 3600)
        self.assertEqual(_callbacks_until_next_dump(3599, 3600), 1)
        self.assertEqual(_callbacks_until_next_dump(3600, 3600), 0)
        self.assertEqual(_callbacks_until_next_dump(3601, 3600), 3599)
        self.assertEqual(_callbacks_until_next_dump(7200, 3600), 0)

    def test_invalid_callback_counters_are_rejected(self):
        with self.assertRaises(ValueError):
            _callbacks_until_next_dump(-1, 1000)
        with self.assertRaises(ValueError):
            _callbacks_until_next_dump(1, 0)


class TestEpisodeOutcomeMetrics(unittest.TestCase):
    def test_rates_use_completed_episodes_not_frames(self):
        metrics = _episode_outcome_rates(2, 6, 8)
        self.assertEqual(metrics["hard_termination_rate"], 0.25)
        self.assertEqual(metrics["timeout_rate"], 0.75)

    def test_no_completed_episode_has_no_fake_zero_rate(self):
        metrics = _episode_outcome_rates(0, 0, 0)
        self.assertNotIn("hard_termination_rate", metrics)
        self.assertNotIn("timeout_rate", metrics)

    def test_window_aggregates_sparse_episode_counts(self):
        rows = [
            {
                "top1_accuracy": 0.8,
                "hard_termination_count": 1,
                "timeout_count": 3,
                "completed_episode_count": 4,
            },
            {
                "top1_accuracy": 1.0,
                "hard_termination_count": 1,
                "timeout_count": 3,
                "completed_episode_count": 4,
            },
        ]
        metrics = _quality_window_metrics(rows)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.9)
        self.assertEqual(metrics["hard_termination_rate"], 0.25)
        self.assertEqual(metrics["timeout_rate"], 0.75)

    def test_goal_validity_aggregates_counts_across_ticks(self):
        metrics = _aggregate_goal_metrics(
            [
                {
                    "oracle_valid_count": 6,
                    "oracle_sample_count": 8,
                    "goal4_fresh_count": 4,
                    "goal4_sample_count": 8,
                },
                {
                    "oracle_valid_count": 8,
                    "oracle_sample_count": 8,
                    "goal4_fresh_count": 6,
                    "goal4_sample_count": 8,
                },
            ]
        )
        self.assertEqual(metrics["goal_valid_count"], 14)
        self.assertEqual(metrics["goal_valid_rate"], 14 / 16)
        self.assertEqual(metrics["goal4_fresh_rate"], 10 / 16)


class TestNavSmokeConfigOverride(unittest.TestCase):
    def test_smoke_num_envs_override_is_in_memory_only(self):
        from agent_ppo.conf.conf import _apply_runtime_env_overrides

        config = {"env": {"num_envs": 256}}
        logger = _Logger()
        with mock.patch.dict(
            os.environ,
            {"NAV_FULL_SMOKE": "1", "NAV_FULL_SMOKE_NUM_ENVS": "8"},
            clear=True,
        ):
            _apply_runtime_env_overrides(config, logger)
        self.assertEqual(config["env"]["num_envs"], 8)
        self.assertTrue(any(level == "warning" for level, _ in logger.messages))

    def test_smoke_num_envs_rejects_out_of_range(self):
        from agent_ppo.conf.conf import _apply_runtime_env_overrides

        config = {"env": {"num_envs": 256}}
        with mock.patch.dict(
            os.environ,
            {"NAV_FULL_SMOKE": "1", "NAV_FULL_SMOKE_NUM_ENVS": "0"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                _apply_runtime_env_overrides(config, _Logger())


class TestNavLifecycleDiagnostics(unittest.TestCase):
    def test_startup_boundaries_keep_one_shot_lifecycle_probes(self):
        agent_dir = pathlib.Path(__file__).resolve().parent.parent
        expected = {
            agent_dir / "agent.py": (
                "[LifecycleProbe] agent_init enter",
                "[LifecycleProbe] agent_init config_resolved",
                "[LifecycleProbe] nav_init_dagger enter",
                "[LifecycleProbe] nav_init_common begin",
                "[LifecycleProbe] nav_init_common complete",
                "[LifecycleProbe] nav_algorithm_construct begin",
                "[LifecycleProbe] nav_algorithm_construct complete",
                "[LifecycleProbe] agent_init before_base_agent",
                "[LifecycleProbe] agent_init complete",
                "[LifecycleProbe] load_model enter",
                "[LifecycleProbe] nav_load_model inventory",
                "[LifecycleProbe] load_model complete",
                "[LifecycleProbe] nav_agent_learn first_call",
                "[LifecycleProbe] nav_save_model first_call",
                "skip framework bootstrap save id=0 before parent preload",
            ),
            agent_dir / "workflow" / "train_workflow.py": (
                "[LifecycleProbe] train_workflow enter",
                "[LifecycleProbe] train_workflow nav_import begin",
                "[LifecycleProbe] train_workflow nav_import complete",
                "[LifecycleProbe] train_workflow dispatch=nav_dagger",
            ),
            agent_dir / "workflow" / "nav_dagger_workflow.py": (
                "[LifecycleProbe] nav_workflow wrapper_enter",
                "[LifecycleProbe] nav_workflow_impl enter",
                "[LifecycleProbe] nav_workflow config_resolved",
                "[LifecycleProbe] nav_env_reset begin",
                "[NavDAgger] reset ok",
                "[LifecycleProbe] nav_iteration first_begin",
                "[LifecycleProbe] nav_first_frame algorithm begin",
                "[LifecycleProbe] nav_first_frame algorithm complete",
                "[LifecycleProbe] nav_first_frame env_step begin",
                "[LifecycleProbe] nav_first_frame env_step complete",
                "[LifecycleProbe] nav_first_update begin",
                "[LifecycleProbe] nav_first_update complete",
                "[LifecycleProbe] nav_first_lifecycle_callback begin",
                "[LifecycleProbe] nav_first_lifecycle_callback complete",
            ),
            agent_dir / "algorithm" / "algorithm_nav_dagger.py": (
                "[LifecycleProbe] nav_algorithm_init",
                "vision_to_device begin",
                "vision_to_device complete",
                "low_level_to_device begin",
                "low_level_to_device complete",
                "high_level_to_device begin",
                "high_level_to_device complete",
                "architecture_check begin",
                "architecture_check complete",
                "freeze_low_level begin",
                "freeze_low_level complete",
                "optimizer_create begin",
                "optimizer_create complete",
                "optimizer_assert begin",
                "optimizer_assert complete",
                "oracle_create begin",
                "oracle_create complete",
                "[LifecycleProbe] nav_frame_begin first_call",
                "[LifecycleProbe] nav_frame_begin per_env_state_ready",
                "[LifecycleProbe] nav_frame_begin command_injected",
                "[LifecycleProbe] nav_frame_begin vision_encoder begin",
                "[LifecycleProbe] nav_frame_begin vision_encoder complete",
                "[LifecycleProbe] nav_frame_begin low_level complete",
                "[LifecycleProbe] nav_frame_begin nav_tick begin",
                "[LifecycleProbe] nav_tick cnn begin",
                "[LifecycleProbe] nav_tick cnn complete",
                "[LifecycleProbe] nav_tick high_level begin",
                "[LifecycleProbe] nav_tick high_level complete",
                "[LifecycleProbe] nav_tick oracle begin",
                "[LifecycleProbe] nav_tick oracle complete",
                "[LifecycleProbe] nav_tick buffer_add complete",
                "[LifecycleProbe] nav_frame_begin nav_tick complete",
                "[LifecycleProbe] nav_frame_begin first_call complete",
            ),
        }
        for path, markers in expected.items():
            source = path.read_text(encoding="utf-8")
            for marker in markers:
                self.assertIn(marker, source, f"missing lifecycle marker in {path}")

        algorithm_source = (
            agent_dir / "algorithm" / "algorithm_nav_dagger.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "first_tick_probe = not self._lifecycle_tick_probe_done",
            algorithm_source,
            "resume runs must still emit the process-local first-tick probes",
        )


if __name__ == "__main__":
    unittest.main()
