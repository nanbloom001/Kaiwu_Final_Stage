#!/usr/bin/env python3
"""Static contract tests for STD-BRIDGE-R1."""

from pathlib import Path
import re
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "agent_ppo"
    / "conf"
    / "train_env_conf_standard_standard_bridge_r1.toml"
)
SOURCE_CONFIG_PATH = (
    ROOT
    / "agent_ppo"
    / "conf"
    / "train_env_conf_standard_stair_inv_finetune.toml"
)


def _config():
    with CONFIG_PATH.open("rb") as stream:
        return tomllib.load(stream)


class StandardBridgeR1ConfigTests(unittest.TestCase):
    def test_schedule_and_clean_transfer_contract(self):
        config = _config()
        stage = config["standard_bridge_r1"]

        self.assertEqual(stage["max_iterations"], 5000)
        self.assertEqual(
            stage["phase_end_iterations"], [1500, 2250, 3000, 3750, 5000]
        )
        self.assertEqual(
            stage["student_drive_ratios"], [0.0, 0.25, 0.5, 0.75, 1.0]
        )
        self.assertEqual(stage["quality_window_iterations"], 50)
        self.assertEqual(stage["save_interval"], 500)
        self.assertEqual(stage["platform_model_id_base"], 10288)
        self.assertEqual(stage["initial_probe_save_iteration"], 1)
        self.assertIs(config["domain_rand"]["enable_domain_rand"], False)
        self.assertIs(config["domain_rand"]["randomize_friction"], False)
        self.assertIs(config["domain_rand"]["push_robots"], False)
        self.assertIs(config["noise"]["add_noise"], False)

    def test_reuses_frozen_teacher_command_and_terrain_domain(self):
        config = _config()
        expected_commands = {
            "lin_vel_x": [0.3, 1.3],
            "lin_vel_y": [-0.2, 0.2],
            "ang_vel_z": [-0.3, 0.3],
        }
        self.assertEqual(config["commands"]["limit"], expected_commands)
        self.assertEqual(
            config["commands"]["ranges"],
            {
                "lin_vel_x": [0.3, 1.3],
                "lin_vel_y": [-0.2, 0.2],
                "ang_vel_yaw": [-0.3, 0.3],
            },
        )
        self.assertEqual(
            config["standard_bridge_r1"]["command_guard"],
            {**expected_commands, "tolerance": 1e-5},
        )
        terrain = config["terrain"]["standard"]
        self.assertEqual(terrain["pyramid_slope"]["proportion"], 0.20)
        self.assertEqual(terrain["pyramid_slope_inv"]["proportion"], 0.20)
        self.assertEqual(terrain["pyramid_stairs"]["proportion"], 0.20)
        self.assertEqual(terrain["pyramid_stairs_inv"]["proportion"], 0.40)
        self.assertEqual(terrain["maze"]["proportion"], 0.0)
        self.assertEqual(config["terrain"]["max_init_terrain_level"], 9)

    def test_source_environment_tables_are_preserved(self):
        config = _config()
        with SOURCE_CONFIG_PATH.open("rb") as stream:
            source = tomllib.load(stream)
        self.assertEqual(config["rewards"], source["rewards"])
        self.assertEqual(
            config["velocity_curriculum"], source["velocity_curriculum"]
        )
        self.assertEqual(config["commands"], source["commands"])
        self.assertEqual(
            config["terrain"]["standard"], source["terrain"]["standard"]
        )

    def test_manual_teacher_identity_and_stage_selector_are_explicit(self):
        config = _config()
        self.assertEqual(
            config["standard_bridge_r1"]["expected_teacher_sha256"],
            "",
        )
        conf_source = (ROOT / "agent_ppo" / "conf" / "conf.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "class StandardBridgeR1Config(StandardRefDistillConfig):", conf_source
        )
        self.assertIn("CURRENT = StandardBridgeR1Config", conf_source)
        self.assertIn('ckpt_name = "model.ckpt-bridge"', conf_source)

    def test_platform_probe_names_and_ids_are_monotonic(self):
        probe = re.compile(
            r".*model\.ckpt-[a-z]*-*([0-9][0-9]*)\..*"
        )

        def checkpoint_id(filename):
            match = probe.fullmatch(filename)
            return int(match.group(1)) if match else None

        expected = {
            "model.ckpt-10289.pkl": 10289,
            "model.ckpt-bridge-11788.pkl": 11788,
            "model.ckpt-teacher-11788.pkl": 11788,
        }
        for filename, model_id in expected.items():
            self.assertEqual(checkpoint_id(filename), model_id)

        self.assertIsNone(
            checkpoint_id("model.ckpt-standard-bridge-r1-1500.pkl")
        )
        self.assertIsNone(
            checkpoint_id("model.ckpt-privileged-loco-teacher-1500.pkl")
        )
        self.assertGreater(
            checkpoint_id("model.ckpt-10289.pkl"),
            checkpoint_id("model.ckpt-10288.pkl"),
        )

    def test_quality_checks_are_advisory(self):
        workflow_source = (
            ROOT / "agent_ppo" / "workflow" / "behavior_distill_workflow.py"
        ).read_text(encoding="utf-8")
        self.assertIn("quality_thresholds", workflow_source)
        self.assertIn(
            "continuing with the configured DAgger schedule", workflow_source
        )
        self.assertNotIn('checkpoint_label="blocked"', workflow_source)

    def test_behavior_evaluation_explicitly_uses_student_model(self):
        agent_source = (ROOT / "agent_ppo" / "agent.py").read_text(
            encoding="utf-8"
        )
        expected = (
            "if self.is_behavior_distill:\n"
            "                return [ActData(action=self.model.act_inference(obs))]"
        )
        self.assertIn(expected, agent_source)


if __name__ == "__main__":
    unittest.main()
