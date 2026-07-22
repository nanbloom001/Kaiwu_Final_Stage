#!/usr/bin/env python3
"""Contracts for STD-D5 scheduled action anchoring."""

import copy
import os
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agent_ppo.conf.conf import (
    Config,
    StandardVisualPPO1HeadingConfig,
    StandardVisualPPO2D5Config,
)


CONF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conf"))


def _load(filename):
    with open(os.path.join(CONF_DIR, filename), "rb") as config_file:
        return tomllib.load(config_file)


def test_d5_is_active_and_inherits_visual_model_contract():
    stage = StandardVisualPPO2D5Config
    assert Config.CURRENT is stage
    assert issubclass(stage, StandardVisualPPO1HeadingConfig)
    assert stage.name == "standard_visual_ppo_2_d5"
    assert stage.algorithm == "visual_ppo"
    assert stage.model_class == "VisualActorCritic"
    assert stage.ckpt_name == "model.ckpt-standard-visual-ppo-2-d5"
    assert stage.lr == pytest.approx(2e-5)
    assert stage.min_learning_rate == pytest.approx(2e-5)
    assert stage.max_learning_rate == pytest.approx(2e-5)
    assert stage.action_anchor_coef == pytest.approx(1.0)
    assert stage.action_anchor_ema == 0.0
    assert stage.action_anchor_schedule_total_steps == 1000
    assert stage.model_save_interval == 100


def test_d5_toml_changes_only_requested_training_contract():
    d4 = _load("train_env_conf_standard_standard_visual_ppo_1_heading.toml")
    d5 = _load("train_env_conf_standard_standard_visual_ppo_2_d5.toml")
    assert d5["env"] == d4["env"]
    assert d5["env_conf"] == d4["env_conf"]
    assert d5["terrain"] == d4["terrain"]
    assert d5["camera"] == d4["camera"]
    assert d5["domain_rand"] == d4["domain_rand"]
    assert d5["noise"] == d4["noise"]
    assert d5["depth_aug"] == d4["depth_aug"]

    assert d5["commands"] == {
        "resampling_time": [25.0, 25.0],
        "limit": {
            "lin_vel_x": [0.45, 0.70],
            "lin_vel_y": [0.0, 0.0],
            "ang_vel_z": [0.0, 0.0],
        },
        "ranges": {
            "lin_vel_x": [0.45, 0.70],
            "lin_vel_y": [0.0, 0.0],
            "ang_vel_yaw": [0.0, 0.0],
        },
    }
    stage_conf = d5["standard_visual_ppo_2_d5"]
    assert stage_conf == {
        "learning_rate": 2e-5,
        "latent_loss_weight": 0.0,
        "latent_cosine_weight": 0.0,
    }
    assert "action_loss_weight" not in stage_conf
    assert "action_loss_decay_updates" not in stage_conf
    assert d5["rewards"]["straight_heading"]["weight"] == pytest.approx(-0.03)

    d4_rewards = copy.deepcopy(d4["rewards"])
    d5_rewards = copy.deepcopy(d5["rewards"])
    d4_rewards.pop("straight_heading")
    d5_rewards.pop("straight_heading")
    assert d5_rewards == d4_rewards


def test_d5_schedule_interpolates_at_all_key_regions():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO

    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    algorithm = AlgorithmPPO(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        learning_rate=2e-5,
        schedule="fixed",
        min_learning_rate=2e-5,
        max_learning_rate=2e-5,
    )
    algorithm.set_reference_policy(copy.deepcopy(model), action_anchor_coef=1.0)

    expected = {
        0: 1.00,
        200: 1.00,
        300: 0.875,
        400: 0.75,
        500: 0.625,
        600: 0.50,
        700: 0.375,
        800: 0.25,
        900: 0.175,
        1000: 0.10,
        1200: 0.10,
    }
    for train_step, coefficient in expected.items():
        algorithm.train_step = train_step
        assert algorithm._get_scheduled_action_anchor_coef() == pytest.approx(
            coefficient
        )


def test_d4_legacy_linear_decay_remains_available():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO

    model = nn.Linear(4, 2)
    algorithm = AlgorithmPPO(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        device=torch.device("cpu"),
        learning_rate=1e-4,
        schedule="fixed",
    )
    algorithm.action_anchor_schedule_enabled = False
    algorithm.set_reference_policy(
        copy.deepcopy(model),
        action_anchor_coef=0.05,
        decay_updates=20,
    )
    algorithm.train_step = 10
    assert algorithm._effective_action_anchor_coef() == pytest.approx(0.025)


def test_d5_monitor_exposes_raw_weighted_and_progress_metrics():
    monitor_path = os.path.join(CONF_DIR, "monitor_builder.py")
    with open(monitor_path, encoding="utf-8") as monitor_file:
        source = monitor_file.read()
    for metric in (
        "action_anchor_coef",
        "action_anchor_loss",
        "weighted_action_anchor_loss",
        "action_anchor_progress",
    ):
        assert f'metrics_name="{metric}"' in source

