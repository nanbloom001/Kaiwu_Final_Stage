#!/usr/bin/env python3
"""Configuration contracts for standard-distill-1."""

import math
import os
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agent_ppo.conf.conf import (
    Config,
    StandardDistill1Config,
    StandardVisualDistill1Config,
)
from agent_ppo.agent import _checkpoint_candidates, _checkpoint_id_from_name


CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "conf",
        "train_env_conf_standard_standard_distill_1.toml",
    )
)
VISUAL_CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "conf",
        "train_env_conf_standard_standard_visual_distill_1.toml",
    )
)


def _load_config():
    with open(CONFIG_PATH, "rb") as config_file:
        return tomllib.load(config_file)


def test_active_stage_is_flat_to_encoder_behavior_bridge():
    assert Config.CURRENT is StandardDistill1Config
    assert StandardDistill1Config.task_type == "standard"
    assert StandardDistill1Config.algorithm == "behavior_distill"
    assert StandardDistill1Config.num_goal_obs == 0
    assert StandardDistill1Config.teacher_num_obs == 301
    assert StandardDistill1Config.num_proprio_obs + StandardDistill1Config.latent_dim == 77
    assert StandardDistill1Config.ckpt_name == "model.ckpt-locomotion"


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
    with open(VISUAL_CONFIG_PATH, "rb") as config_file:
        config = tomllib.load(config_file)
    camera = config["camera"]["depth_camera"]
    quaternion = camera["offset_rot"]
    norm = math.sqrt(sum(component * component for component in quaternion))

    assert camera["offset_pos"] == [0.339871, 0.034697, 0.075010]
    assert quaternion == [0.982631, -0.007085, 0.184337, -0.020153]
    assert norm == pytest.approx(1.0, abs=1e-6)

    stage = config["standard_visual_distill_1"]
    assert stage["sequence_length"] == 8
    assert stage["student_drive_phase_fractions"] == [0.03, 0.47, 0.50]
    assert stage["student_drive_ratios"] == [0.0, 0.5, 1.0]
    assert stage["action_loss_weight"] == 0.2
    assert StandardVisualDistill1Config.algorithm == "lbc_loco"
    assert StandardVisualDistill1Config.num_goal_obs == 0


def test_behavior_bridge_uses_flat_teacher_and_teacher_driven_states():
    config = _load_config()
    assert config["env_conf"]["task_name"] == "Unitree-Go2-Velocity"
    assert "camera" not in config
    stage = config["standard_distill_1"]
    assert stage["action_loss_weight"] == 1.0
    assert stage["student_drive"] is False


def test_platform_checkpoint_discovery_accepts_labels_and_extensions(tmp_path):
    expected = {
        "model.ckpt-10288.pkl",
        "model.ckpt-hjcnew-10288.pkl",
        "model.ckpt-lbc-loco-10288.pth",
    }
    for filename in expected | {"model.ckpt-hjcnew-9999.pkl", "notes.txt"}:
        (tmp_path / filename).touch()

    candidates = _checkpoint_candidates(str(tmp_path), 10288)
    assert {Path(path).name for path in candidates} == expected
    assert _checkpoint_id_from_name("model.ckpt-standard-10288.pkl") == 10288
    assert _checkpoint_id_from_name("model.ckpt-standard-latest.pkl") is None
