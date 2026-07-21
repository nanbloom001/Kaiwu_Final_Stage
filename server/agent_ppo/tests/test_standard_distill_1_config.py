#!/usr/bin/env python3
"""Configuration contracts for standard-distill-1."""

import math
import os
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agent_ppo.conf.conf import Config, StandardDistill1Config


CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "conf",
        "train_env_conf_standard_standard_distill_1.toml",
    )
)


def _load_config():
    with open(CONFIG_PATH, "rb") as config_file:
        return tomllib.load(config_file)


def test_active_stage_preserves_standard_teacher_contract():
    assert Config.CURRENT is StandardDistill1Config
    assert StandardDistill1Config.task_type == "standard"
    assert StandardDistill1Config.algorithm == "lbc_loco"
    assert StandardDistill1Config.num_goal_obs == 0
    assert StandardDistill1Config.proprio_dim + StandardDistill1Config.latent_dim == 77
    assert StandardDistill1Config.ckpt_name == "model.ckpt-standard"


def test_all_direction_low_speed_curriculum_and_push_phase():
    config = _load_config()
    ranges = config["commands"]["ranges"]
    limits = config["commands"]["limit"]

    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw"):
        assert ranges[key][0] < 0.0 < ranges[key][1]
    assert limits["lin_vel_x"] == [-0.20, 0.65]
    assert limits["lin_vel_y"] == [-0.15, 0.15]
    assert limits["ang_vel_z"] == [-1.30, 1.30]
    assert config["terrain"]["curriculum"] is True
    assert config["domain_rand"]["randomize_friction"] is True
    assert config["domain_rand"]["push_robots"] is False


def test_camera_and_sequence_distillation_contract():
    config = _load_config()
    camera = config["camera"]["depth_camera"]
    quaternion = camera["offset_rot"]
    norm = math.sqrt(sum(component * component for component in quaternion))

    assert camera["offset_pos"] == [0.339871, 0.034697, 0.075010]
    assert quaternion == [0.982631, -0.007085, 0.184337, -0.020153]
    assert norm == pytest.approx(1.0, abs=1e-6)

    stage = config["standard_distill_1"]
    assert stage["sequence_length"] == 8
    assert stage["student_drive_phase_fractions"] == [0.03, 0.47, 0.50]
    assert stage["student_drive_ratios"] == [0.0, 0.5, 1.0]
    assert stage["action_loss_weight"] == 0.2
