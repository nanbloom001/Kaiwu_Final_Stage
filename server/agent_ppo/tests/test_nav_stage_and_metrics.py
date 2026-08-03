#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Nav bootstrap/eval routing and episode-outcome metric regressions."""

import ast
import os
import pathlib
import sys
import types
import unittest
from unittest import mock

import agent_ppo.tests._nav_test_stubs  # noqa: F401

import toml

from agent_ppo.conf.conf import (
    Config,
    LBCLocoConfig,
    NavDaggerConfig,
    NavEvalConfig,
    P15ResponseConfig,
    P2NavEvalConfig,
    P2NavPPOConfig,
    P3StandardEvalConfig,
    P3StandardJointConfig,
    P3TrackEvalConfig,
    P4NavPPOConfig,
    StandardVisualPPOConfig,
    _configured_training_stage,
    _infer_stage_from_task_name,
)
from agent_ppo.workflow.nav_dagger_workflow import (
    _aggregate_goal_metrics,
    _aggregate_switch_response_metrics,
    _callbacks_until_next_dump,
    _episode_outcome_rates,
    _quality_window_metrics,
    _quality_absolutely_ok,
    _soft_stay_check,
)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


class TestNavStageSelection(unittest.TestCase):
    def test_p4_training_agent_init_resolves_single_maze_segment_contract(self):
        kaiwu_mod = types.ModuleType("kaiwudrl")
        interface_mod = types.ModuleType("kaiwudrl.interface")
        agent_mod = types.ModuleType("kaiwudrl.interface.agent")
        agent_mod.BaseAgent = type("BaseAgent", (), {})
        interface_mod.agent = agent_mod
        kaiwu_mod.interface = interface_mod
        sys.modules.setdefault("kaiwudrl", kaiwu_mod)
        sys.modules.setdefault("kaiwudrl.interface", interface_mod)
        sys.modules.setdefault("kaiwudrl.interface.agent", agent_mod)
        validate_mod = types.ModuleType("tools.train_env_conf_validate")
        validate_mod.check_usr_conf = lambda *_args, **_kwargs: (True, "ok")
        sys.modules.setdefault("tools.train_env_conf_validate", validate_mod)

        from agent_ppo.agent import Agent

        agent = Agent.__new__(Agent)
        agent.is_p2_nav_eval = False
        agent.is_p4_nav = True
        agent.device = "cpu"
        agent.num_envs = 1
        agent.num_actions = 12
        agent.logger = _Logger()
        agent.monitor = None
        agent._init_p2_nav(
            P4NavPPOConfig,
            {
                "p4_nav_ppo": {"p4_seed": 1, "num_learning_epochs": 4},
                "terrain": {"track": {"sub_terrains": ["open_entry_maze"]}},
            },
        )
        self.assertEqual(agent.algorithm.config["track_segment_labels"], ["maze"])

    def test_active_branch_bootstraps_worker_and_aisrv_from_configure_app(self):
        config_path = pathlib.Path(__file__).resolve().parents[2] / "conf" / "configure_app.toml"
        policy_entry = toml.load(config_path)["app"]["policy_entry"]
        expected = {
            "nav_dagger": NavDaggerConfig,
            "p15_response": P15ResponseConfig,
            "p2_nav_ppo": P2NavPPOConfig,
            "p3_standard_joint": P3StandardJointConfig,
            "p4_nav_ppo": P4NavPPOConfig,
        }[policy_entry]
        self.assertIs(Config.CURRENT, expected)

    def test_literal_stage_default_matches_active_branch(self):
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
        config_path = pathlib.Path(__file__).resolve().parents[2] / "conf" / "configure_app.toml"
        policy_entry = toml.load(config_path)["app"]["policy_entry"]
        expected_name = {
            "nav_dagger": "NavDaggerConfig",
            "p15_response": "P15ResponseConfig",
            "p2_nav_ppo": "P2NavPPOConfig",
            "p3_standard_joint": "P3StandardJointConfig",
            "p4_nav_ppo": "P4NavPPOConfig",
        }[policy_entry]
        self.assertEqual(current_assignment.value.id, expected_name)

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

    def test_track_camera_without_forwarded_entry_preserves_p2_lineage(self):
        usr_conf = {
            "env_conf": {"task_name": "Unitree-Go2-Velocity-Camera"},
            "terrain": {"mode": "track"},
        }
        with mock.patch.object(Config, "CURRENT", P2NavPPOConfig):
            self.assertIs(
                _infer_stage_from_task_name(usr_conf, _Logger()), P2NavEvalConfig
            )

    def test_track_camera_without_forwarded_entry_keeps_legacy_nav_lineage(self):
        usr_conf = {
            "env_conf": {"task_name": "Unitree-Go2-Velocity-Camera"},
            "terrain": {"mode": "track"},
        }
        with mock.patch.object(Config, "CURRENT", NavDaggerConfig):
            self.assertIs(
                _infer_stage_from_task_name(usr_conf, _Logger()), NavEvalConfig
            )

    def test_standard_camera_without_forwarded_entry_keeps_p3_lineage(self):
        usr_conf = {
            "env_conf": {"task_name": "Unitree-Go2-Velocity-Camera"},
            "terrain": {"mode": "standard"},
        }
        # 当前分支 bootstrap 为 P3 血缘：Standard+Camera 不能回退到 lbc_loco，
        # 必须保持 p3_standard_eval（否则 P3 包找不到 highslow 候选）。
        with mock.patch.object(Config, "CURRENT", P3StandardJointConfig):
            self.assertIs(
                _infer_stage_from_task_name(usr_conf, _Logger()),
                P3StandardEvalConfig,
            )

    def test_standard_camera_without_forwarded_entry_keeps_legacy_lbc(self):
        usr_conf = {
            "env_conf": {"task_name": "Unitree-Go2-Velocity-Camera"},
            "terrain": {"mode": "standard"},
        }
        # 非 P3 血缘（历史 StandardVisualPPO 基线）仍保持历史 lbc_loco 回退。
        with mock.patch.object(Config, "CURRENT", StandardVisualPPOConfig):
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

    def test_explicit_p2_training_entry_is_eval_only_during_eval_inference(self):
        usr_conf = {
            "env_conf": {
                "task_name": "Unitree-Go2-Velocity-Camera",
                "policy_entry": "p2_nav_ppo",
            },
            "terrain": {"mode": "track"},
        }
        self.assertIs(
            _infer_stage_from_task_name(usr_conf, _Logger()), P2NavEvalConfig
        )

    def test_p2_eval_single_stream_unpacks_aux_and_builds_wire(self):
        import torch

        kaiwu_mod = types.ModuleType("kaiwudrl")
        interface_mod = types.ModuleType("kaiwudrl.interface")
        agent_mod = types.ModuleType("kaiwudrl.interface.agent")

        class _BaseAgent:
            pass

        agent_mod.BaseAgent = _BaseAgent
        interface_mod.agent = agent_mod
        kaiwu_mod.interface = interface_mod
        sys.modules.setdefault("kaiwudrl", kaiwu_mod)
        sys.modules.setdefault("kaiwudrl.interface", interface_mod)
        sys.modules.setdefault("kaiwudrl.interface.agent", agent_mod)
        validate_mod = types.ModuleType("tools.train_env_conf_validate")
        validate_mod.check_usr_conf = lambda *_args, **_kwargs: (True, "ok")
        sys.modules.setdefault("tools.train_env_conf_validate", validate_mod)

        from agent_ppo.agent import Agent
        from agent_ppo.feature import nav_contract, p2_contract

        class _Algorithm:
            def __init__(self):
                self.received = None
                self.advanced = 0

            def frame_begin(self, obs, wire, *, deterministic):
                self.received = (obs.clone(), wire.clone(), deterministic)
                return {"actions": torch.ones(obs.shape[0], 12)}, None, None

            def eval_frame_advance(self):
                self.advanced += 1

        agent = Agent.__new__(Agent)
        agent.device = "cpu"
        agent.is_p2_nav = True
        agent.is_p2_nav_eval = True
        agent.is_lbc = False
        agent.is_nav_dagger = False
        agent.is_nav_eval = False
        agent.is_visual_ppo = False
        agent._p2_eval_checkpoint_path = "/tmp/model.ckpt-navadapt-61633.pkl"
        agent._lifecycle_probe_exploit_logged = True
        agent.algorithm = _Algorithm()
        obs = torch.zeros(2, nav_contract.POLICY_OBS_DIM)
        aux = torch.arange(60, dtype=torch.float32).reshape(2, 30)
        packed = p2_contract.pack_eval_response_aux(obs, aux)
        result = agent.exploit(packed)
        received_obs, received_wire, deterministic = agent.algorithm.received
        self.assertTrue(torch.equal(received_obs, packed))
        worker_aux = received_wire[:, p2_contract.CRITIC_OBS_DIM :]
        self.assertTrue(
            torch.equal(worker_aux[:, : p2_contract.RESPONSE_AUX_DIM], aux)
        )
        self.assertTrue(
            torch.equal(
                worker_aux[:, p2_contract.RESPONSE_AUX_DIM :],
                torch.zeros(2, p2_contract.DIAGNOSTIC_AUX_DIM),
            )
        )
        self.assertEqual(received_wire.shape, (2, p2_contract.PRIVILEGED_WIRE_DIM))
        self.assertTrue(deterministic)
        self.assertEqual(agent.algorithm.advanced, 1)
        self.assertEqual(tuple(result[0].action.shape), (2, 12))

    def test_p2_eval_ensure_load_uses_runtime_model_location(self):
        kaiwu_mod = types.ModuleType("kaiwudrl")
        interface_mod = types.ModuleType("kaiwudrl.interface")
        agent_mod = types.ModuleType("kaiwudrl.interface.agent")
        agent_mod.BaseAgent = type("BaseAgent", (), {})
        interface_mod.agent = agent_mod
        kaiwu_mod.interface = interface_mod
        sys.modules.setdefault("kaiwudrl", kaiwu_mod)
        sys.modules.setdefault("kaiwudrl.interface", interface_mod)
        sys.modules.setdefault("kaiwudrl.interface.agent", agent_mod)
        validate_mod = types.ModuleType("tools.train_env_conf_validate")
        validate_mod.check_usr_conf = lambda *_args, **_kwargs: (True, "ok")
        sys.modules.setdefault("tools.train_env_conf_validate", validate_mod)
        config_mod = types.ModuleType("common_python.config.config_control")
        config_mod.CONFIG = types.SimpleNamespace(
            eval_model_dir="/tmp/p2-eval", eval_model_id="61633"
        )
        config_pkg = types.ModuleType("common_python.config")
        config_pkg.config_control = config_mod
        sys.modules["common_python.config"] = config_pkg
        sys.modules["common_python.config.config_control"] = config_mod

        from agent_ppo.agent import Agent

        agent = Agent.__new__(Agent)
        agent.is_p2_nav_eval = True
        agent._p2_eval_checkpoint_path = None
        calls = []

        def _load(path, model_id):
            calls.append((path, model_id))
            agent._p2_eval_checkpoint_path = f"{path}/model.ckpt-navadapt-{model_id}.pkl"

        agent._load_p2_nav = _load
        agent._ensure_p2_eval_checkpoint_loaded()
        self.assertEqual(calls, [("/tmp/p2-eval", "61633")])

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
        self.assertEqual(
            config["nav_dagger"]["platform_archive_interval_minutes"], 5.0
        )
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
        nav_panel_source = monitor_source.split("NAV_PANEL_SPECS = (", 1)[1].split(
            "\n)\n\nORACLE_MODE_NAMES", 1
        )[0]
        for metric in (
            "completed_count_track_l",
            "abnormal_count_track_l",
            "timeout_count_track_l",
            "total_score_track_l",
            "time_score_track_l",
            "ce_loss",
            "top1_accuracy",
            "goal_progress_m_per_frame",
            "worker_exec_cmd_linf_mean",
            "worker_exec_cmd_match_ratio",
            "actual_lin_vel_x_mean",
            "exec_vx_tracking_error_mean",
            "worker_vx_tracking_error_mean",
            "low_level_action_delta_abs_mean",
            "low_level_action_nonfinite_count",
            "switch_response_sample_count",
            "switch_response_vx_sample_count",
            "switch_response_action_delta_abs_mean",
            "switch_response_vx_delta_abs_mean",
            "switch_response_no_action_ratio",
            "switch_response_no_velocity_ratio",
            "scheduler_dwell_ticks_mean",
            "scheduler_dwell_ticks_max",
            "platform_lifecycle_callbacks",
            "platform_lifecycle_failures",
            "total_low_level_steps",
            "total_env_frames",
            "callbacks_until_next_dump",
            "student_drive_ratio",
            "oracle_front_score_mean",
            "oracle_left_score_mean",
            "oracle_right_score_mean",
            "low_level_action_abs_max",
        ):
            self.assertIn(metric, monitor_source)
        for prefix in ("student", "oracle", "requested", "effective"):
            self.assertIn(f'f"{prefix}_token_{{name}}_ratio"', monitor_source)
        self.assertIn('f"oracle_mode_{name}_ratio"', monitor_source)
        self.assertNotIn("step_score_track_l", monitor_source)
        self.assertNotIn("hard_termination_rate", nav_panel_source)
        self.assertNotIn('name_en="value_loss"', monitor_source)

    def test_p4_monitor_declares_all_twenty_track_columns(self):
        monitor_source = (
            pathlib.Path(__file__).resolve().parent.parent
            / "conf"
            / "monitor_builder.py"
        ).read_text(encoding="utf-8")
        self.assertIn("level_count=20", monitor_source)
        self.assertIn('group_name_en="p4_track_outcomes"', monitor_source)

    def test_monitor_panel_names_use_platform_supported_characters(self):
        monitor_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "conf"
            / "monitor_builder.py"
        )
        module = ast.parse(monitor_path.read_text(encoding="utf-8"))
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"TRACK_PANEL_SPECS", "NAV_PANEL_SPECS"}
        }
        for specs in assignments.values():
            for name, *_ in specs:
                self.assertLessEqual(len(name), 20, name)
                self.assertTrue(
                    name and all(char.isalnum() or char in "*-_ " for char in name),
                    name,
                )


class TestNavPlatformDumpCadence(unittest.TestCase):
    def test_callbacks_until_next_dump_boundaries(self):
        self.assertEqual(_callbacks_until_next_dump(0, 3600), 3600)
        self.assertEqual(_callbacks_until_next_dump(3599, 3600), 1)
        self.assertEqual(_callbacks_until_next_dump(3600, 3600), 0)
        self.assertEqual(_callbacks_until_next_dump(3601, 3600), 3599)
        self.assertEqual(_callbacks_until_next_dump(7200, 3600), 0)
        self.assertEqual(_callbacks_until_next_dump(0, 3600, 34728), 1272)
        self.assertEqual(_callbacks_until_next_dump(1272, 3600, 34728), 0)

    def test_invalid_callback_counters_are_rejected(self):
        with self.assertRaises(ValueError):
            _callbacks_until_next_dump(-1, 1000)
        with self.assertRaises(ValueError):
            _callbacks_until_next_dump(1, 0)


class TestEpisodeOutcomeMetrics(unittest.TestCase):
    def test_rates_use_completed_episodes_not_frames(self):
        metrics = _episode_outcome_rates(2, 6, 8)
        self.assertEqual(metrics["non_timeout_termination_rate"], 0.25)
        self.assertEqual(metrics["timeout_rate"], 0.75)
        self.assertEqual(metrics["ended_episode_count"], 8)

    def test_no_completed_episode_has_no_fake_zero_rate(self):
        metrics = _episode_outcome_rates(0, 0, 0)
        self.assertNotIn("non_timeout_termination_rate", metrics)
        self.assertNotIn("completed_episode_count", metrics)
        self.assertNotIn("timeout_rate", metrics)

    def test_window_aggregates_sparse_episode_counts(self):
        rows = [
            {
                "top1_accuracy": 0.8,
                "non_timeout_termination_count": 1,
                "timeout_count": 3,
                "ended_episode_count": 4,
            },
            {
                "top1_accuracy": 1.0,
                "non_timeout_termination_count": 1,
                "timeout_count": 3,
                "ended_episode_count": 4,
            },
        ]
        metrics = _quality_window_metrics(rows)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.9)
        self.assertEqual(metrics["non_timeout_termination_rate"], 0.25)
        self.assertEqual(metrics["timeout_rate"], 0.75)

    def test_soft_stay_uses_per_frame_hazard_not_episode_composition(self):
        previous = {
            "non_timeout_termination_rate": 0.0,
            "non_timeout_termination_per_frame": 0.001,
        }
        current = {
            "non_timeout_termination_rate": 1.0,
            "non_timeout_termination_per_frame": 0.001,
        }
        self.assertIsNone(_soft_stay_check(current, previous))

        current["non_timeout_termination_per_frame"] = 0.03
        self.assertEqual(
            _soft_stay_check(current, previous),
            "non_timeout_termination_worsened",
        )

    def test_soft_stay_unfreeze_requires_per_frame_quality(self):
        self.assertTrue(
            _quality_absolutely_ok(
                {
                    "top1_accuracy": 0.9,
                    "non_timeout_termination_rate": 1.0,
                    "non_timeout_termination_per_frame": 0.01,
                }
            )
        )

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

    def test_switch_response_aggregation_ignores_zero_sample_ticks(self):
        metrics = _aggregate_switch_response_metrics(
            [
                {
                    "switch_response_sample_count": 0,
                    "switch_response_no_action_ratio": 0.0,
                },
                {
                    "switch_response_sample_count": 4,
                    "switch_response_vx_sample_count": 4,
                    "switch_response_action_delta_abs_mean": 0.0,
                    "switch_response_vx_delta_abs_mean": 0.01,
                    "switch_response_no_action_ratio": 1.0,
                    "switch_response_no_velocity_ratio": 0.75,
                },
            ]
        )
        self.assertEqual(metrics["switch_response_sample_count"], 4)
        self.assertEqual(metrics["switch_response_vx_sample_count"], 4)
        self.assertEqual(metrics["switch_response_no_action_ratio"], 1.0)
        self.assertEqual(metrics["switch_response_no_velocity_ratio"], 0.75)


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
