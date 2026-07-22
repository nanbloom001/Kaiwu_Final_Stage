#!/usr/bin/env python3
"""Contracts for STD-D4A visual PPO heading fine-tuning."""

import os
import re
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agent_ppo.conf.conf import StandardVisualPPO1HeadingConfig


CONF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conf"))
CONFIG_NAME = "train_env_conf_standard_standard_visual_ppo_1_heading.toml"


def _load_config():
    with open(os.path.join(CONF_DIR, CONFIG_NAME), "rb") as config_file:
        return tomllib.load(config_file)


def test_d4a_remains_a_reproducible_visual_ppo_stage():
    stage = StandardVisualPPO1HeadingConfig
    assert stage.name == "standard_visual_ppo_1_heading"
    assert stage.algorithm == "visual_ppo"
    assert stage.model_class == "VisualActorCritic"
    assert stage.task_type == "standard"
    assert stage.lr == pytest.approx(1e-4)
    assert stage.schedule == "fixed"
    assert stage.num_steps_per_env == 8
    assert stage.action_anchor_ema == 0.0


def test_d4a_clean_stair_distribution_and_command_contract():
    config = _load_config()
    terrain = config["terrain"]
    assert terrain["curriculum"] is True
    assert terrain["max_init_terrain_level"] == 4
    assert terrain["standard"] == {
        "pyramid_slope": {"proportion": 0.15},
        "pyramid_slope_inv": {"proportion": 0.15},
        "pyramid_stairs": {"proportion": 0.35},
        "pyramid_stairs_inv": {"proportion": 0.35},
        "maze": {"proportion": 0.0},
    }
    assert config["commands"] == {
        "resampling_time": [25.0, 25.0],
        "limit": {
            "lin_vel_x": [0.40, 0.60],
            "lin_vel_y": [0.0, 0.0],
            "ang_vel_z": [0.0, 0.0],
        },
        "ranges": {
            "lin_vel_x": [0.40, 0.60],
            "lin_vel_y": [0.0, 0.0],
            "ang_vel_yaw": [0.0, 0.0],
        },
    }
    assert config["domain_rand"]["enable_domain_rand"] is False
    assert config["domain_rand"]["randomize_friction"] is False
    assert config["domain_rand"]["push_robots"] is False
    assert config["noise"]["add_noise"] is False
    assert config["depth_aug"]["enabled"] is False


def test_d4a_losses_heading_and_camera_contract():
    config = _load_config()
    stage = config["standard_visual_ppo_1_heading"]
    assert stage["learning_rate"] == pytest.approx(1e-4)
    assert stage["latent_loss_weight"] == 0.0
    assert stage["latent_cosine_weight"] == 0.0
    assert stage["action_loss_weight"] == pytest.approx(0.05)
    assert stage["action_loss_decay_updates"] == 20

    assert config["rewards"]["straight_heading"] == {
        "weight": -0.12,
        "params": {
            "command_name": "base_velocity",
            "max_yaw_command": 0.10,
            "min_forward_command": 0.20,
        },
    }
    assert "lateral_drift" not in config["rewards"]
    assert "foot_symmetry" not in config["rewards"]
    assert "feet_clearance" not in config["rewards"]
    assert config["camera"]["depth_camera"] == {
        "offset_pos": [0.339871, 0.034697, 0.075010],
        "offset_rot": [0.982631, -0.007085, 0.184337, -0.020153],
    }


def test_platform_probe_uses_unlabelled_alias():
    probe = re.compile(r"model\.ckpt-[a-z]*-*(\d+)\..+")
    descriptive = "model.ckpt-standard-visual-ppo-1-heading-50.pkl"
    alias = "model.ckpt-50.pkl"
    assert probe.search(descriptive) is None
    assert probe.search(alias).group(1) == "50"


def test_visual_actor_ignores_privileged_scan_and_supports_sequences():
    torch = pytest.importorskip("torch")
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
    assert torch.allclose(action_a, action_b, atol=1e-6, rtol=1e-6)
    assert model.evaluate(torch.randn(batch, 316)).shape == (batch, 1)

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
    assert output.shape == (3, batch, 12)
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.vision_encoder.parameters())
