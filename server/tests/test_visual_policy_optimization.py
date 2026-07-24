#!/usr/bin/env python3
"""Static contract tests for Standard Stage-5 visual policy optimization."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
CONFIG = (
    SERVER
    / "agent_ppo"
    / "conf"
    / "train_env_conf_standard_visual_policy_optimization.toml"
)


class VisualPolicyConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG.open("rb") as stream:
            cls.config = tomllib.load(stream)

    def test_single_run_schedule_and_tbptt_contract(self):
        stage = self.config["visual_policy_optimization"]
        self.assertEqual(stage["num_steps_per_env"], 48)
        self.assertEqual(stage["tbptt_sequence_length"], 16)
        self.assertEqual(48 % 16, 0)
        self.assertEqual(stage["critic_only_hours"], 0.5)
        self.assertEqual(stage["actor_only_end_hours"], 1.5)
        self.assertEqual(stage["task_end_hours"], 3.0)
        self.assertEqual(stage["save_interval_minutes"], 10.0)

    def test_command_buckets_are_bounded_and_sum_to_one(self):
        buckets = self.config["commands"]["buckets"]
        weights = [
            value for key, value in buckets.items() if key.endswith("_weight")
        ]
        self.assertAlmostEqual(sum(weights), 1.0, places=8)
        self.assertEqual(buckets["hold_time_s"], [3.0, 6.0])
        limits = self.config["commands"]["limit"]
        self.assertEqual(limits["lin_vel_x"], [0.0, 0.7])
        self.assertEqual(limits["lin_vel_y"], [-0.2, 0.2])
        self.assertEqual(limits["ang_vel_z"], [-0.3, 0.3])

    def test_controlled_first_run_keeps_randomization_off(self):
        self.assertFalse(self.config["domain_rand"]["enable_domain_rand"])
        self.assertFalse(self.config["domain_rand"]["push_robots"])
        self.assertFalse(self.config["noise"]["add_noise"])
        self.assertFalse(
            self.config["camera"]["depth_camera"]["augmentation"]["enabled"]
        )
        self.assertEqual(self.config["rewards"]["forward_velocity"]["weight"], 0.0)


class VisualPolicySourceTests(unittest.TestCase):
    def test_visual_bundle_save_is_not_executed_during_agent_init(self):
        source = (
            SERVER / "agent_ppo" / "agent.py"
        ).read_text(encoding="utf-8")
        init_block = source.split("def __init__(", 1)[1].split(
            "def _init_flat(", 1
        )[0]
        save_block = source.split("def save_model(", 1)[1].split(
            "def save_vision_at_ramp_label(", 1
        )[0]
        self.assertNotIn("save_training_bundle(", init_block)
        self.assertIn("self._init_visual_ppo(", init_block)
        self.assertIn("save_training_bundle(", save_block)

    def test_camera_eval_inference_is_available_in_env_subprocess(self):
        conf_source = (
            SERVER / "agent_ppo" / "conf" / "conf.py"
        ).read_text(encoding="utf-8")
        env_source = (
            SERVER / "isaac_env" / "base_env.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def _infer_stage_from_task_name(", conf_source)
        self.assertIn(
            'getattr(mod, "_infer_stage_from_task_name", None)',
            env_source,
        )

    def test_recurrent_generator_does_not_use_flattened_minibatches(self):
        source = (
            SERVER / "agent_ppo" / "feature" / "definition.py"
        ).read_text(encoding="utf-8")
        block = source.split("class RecurrentRolloutStorage", 1)[1]
        self.assertIn("recurrent_mini_batch_generator", block)
        self.assertIn("sequence_length", block)
        self.assertIn("continuation_masks", block)
        self.assertNotIn("_flatten_buffers()", block)

    def test_phase_checkpoint_labels_are_probe_safe(self):
        source = (
            SERVER / "agent_ppo" / "checkpoint_io.py"
        ).read_text(encoding="utf-8")
        labels = re.findall(r'\"(rl[a-z]+)\"', source)
        self.assertTrue({"rlcritic", "rlactor", "rlfull"}.issubset(labels))
        for label in labels:
            self.assertRegex(f"model.ckpt-{label}-28401.pkl", r"^model\.ckpt-[a-z]+-[0-9]+\.pkl$")

    def test_new_stage_does_not_import_failed_d4_d5_implementations(self):
        paths = [
            SERVER / "agent_ppo" / "algorithm" / "algorithm_visual_ppo.py",
            SERVER / "agent_ppo" / "workflow" / "visual_ppo_workflow.py",
            CONFIG,
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn("standard_visual_ppo_1_heading", combined)
        self.assertNotIn("standard_visual_ppo_2_d5", combined)
        self.assertNotIn('format = "visual_ppo"', combined)


@unittest.skipIf(torch is None, "torch not installed; run on platform/CI")
class VisualPolicyTensorTests(unittest.TestCase):
    def test_actor_ignores_privileged_scan_and_backpropagates_sequences(self):
        from agent_ppo.model.visual_actor_critic import VisualActorCritic

        torch.manual_seed(7)
        model = VisualActorCritic(
            num_proprio=45,
            num_scan=256,
            depth_shape=(64, 64, 1),
            latent_dim=32,
            cnn_output_dim=16,
            lstm_hidden_size=24,
            lstm_num_layers=2,
            num_critic_obs=316,
            num_actions=12,
            actor_hidden_dims=(32, 24),
            critic_hidden_dims=(32, 24),
        )
        model.eval()

        batch = 2
        proprio = torch.randn(batch, 45)
        depth = torch.randn(batch, 64 * 64)
        obs_a = torch.cat((proprio, torch.zeros(batch, 256), depth), dim=-1)
        obs_b = torch.cat((proprio, torch.randn(batch, 256), depth), dim=-1)

        model.reset()
        action_a = model.act_inference(obs_a)
        model.reset()
        action_b = model.act_inference(obs_b)
        self.assertTrue(torch.allclose(action_a, action_b, atol=1e-6, rtol=1e-6))
        self.assertEqual(model.evaluate(torch.randn(batch, 316)).shape, (batch, 1))

        sequence = obs_a.unsqueeze(0).repeat(3, 1, 1).requires_grad_(True)
        hidden = (
            torch.zeros(2, batch, 24),
            torch.zeros(2, batch, 24),
        )
        output = model.act_inference(
            sequence,
            hidden_states=hidden,
            masks=torch.ones(3, batch, 1),
        )
        self.assertEqual(output.shape, (3, batch, 12))
        output.square().mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.vision_encoder.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
