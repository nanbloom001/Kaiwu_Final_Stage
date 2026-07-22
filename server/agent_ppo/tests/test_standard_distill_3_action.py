#!/usr/bin/env python3
"""Contracts for the D2-to-D3A stair action-alignment continuation."""

import os
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agent_ppo.conf.conf import (
    Config,
    LBCLocoConfig,
    StandardDistill2StairConfig,
    StandardDistill3ActionConfig,
)


CONF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conf"))


def _load(filename):
    with open(os.path.join(CONF_DIR, filename), "rb") as config_file:
        return tomllib.load(config_file)


def test_d3a_is_the_active_lbc_continuation_stage():
    assert Config.CURRENT is StandardDistill3ActionConfig
    assert issubclass(StandardDistill3ActionConfig, LBCLocoConfig)
    assert StandardDistill3ActionConfig is not StandardDistill2StairConfig
    assert StandardDistill3ActionConfig.task_type == "standard"
    assert StandardDistill3ActionConfig.algorithm == "lbc_loco"
    assert StandardDistill3ActionConfig.num_goal_obs == 0
    assert (
        StandardDistill3ActionConfig.ckpt_name
        == "model.ckpt-standard-distill-3-action"
    )


def test_d3a_keeps_d2_terrain_and_camera_contract():
    d2 = _load("train_env_conf_standard_standard_distill_2_stair.toml")
    d3 = _load("train_env_conf_standard_standard_distill_3_action.toml")

    assert d3["terrain"] == d2["terrain"]
    assert d3["camera"] == d2["camera"]
    assert d3["terrain"]["standard"] == {
        "pyramid_slope": {"proportion": 0.05},
        "pyramid_slope_inv": {"proportion": 0.05},
        "pyramid_stairs": {"proportion": 0.30},
        "pyramid_stairs_inv": {"proportion": 0.60},
        "maze": {"proportion": 0.0},
    }


def test_d3a_action_alignment_and_clean_rollout_settings():
    config = _load("train_env_conf_standard_standard_distill_3_action.toml")
    stage = config["standard_distill_3_action"]

    assert stage["learning_rate"] == 2e-4
    assert stage["max_iterations"] == 10000
    assert stage["num_steps_per_env"] == 24
    assert stage["bptt_steps"] == 8
    assert stage["require_student_resume"] is True
    assert stage["student_drive"] is False
    assert stage["action_loss_weight"] == 1.0
    assert stage["latent_cosine_weight"] == 0.0
    assert stage["cosine_loss_weight"] == 0.0

    assert config["commands"] == {
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
    assert config["domain_rand"]["enable_domain_rand"] is False
    assert config["domain_rand"]["randomize_friction"] is False
    assert config["domain_rand"]["push_robots"] is False
    assert config["noise"]["add_noise"] is False
    assert config["depth_aug"]["enabled"] is False
    assert "rewards" not in config


def test_platform_probe_is_satisfied_by_the_unlabelled_alias():
    # This mirrors the platform's documented sed probe. The requested D3A
    # descriptive filename is retained, while Agent.save_model also emits the
    # simple alias that the probe can discover reliably.
    probe = re.compile(r"model\.ckpt-[a-z]*-*(\d+)\..+")
    descriptive = "model.ckpt-standard-distill-3-action-250.pkl"
    alias = "model.ckpt-250.pkl"
    assert probe.search(descriptive) is None
    assert probe.search(alias).group(1) == "250"


def test_workflow_logs_the_verified_student_resume_source():
    workflow_path = os.path.join(
        os.path.dirname(__file__), "..", "workflow", "lbc_workflow.py"
    )
    with open(workflow_path, encoding="utf-8") as workflow_file:
        source = workflow_file.read()
    assert "algorithm.assert_student_ready()" in source
    assert "resumed visual student checkpoint" in source

