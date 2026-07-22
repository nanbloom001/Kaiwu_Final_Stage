#!/usr/bin/env python3
"""Contracts for the stair-only Standard visual continuation."""

import copy
import os
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agent_ppo.conf.conf import (
    StandardDistill1Config,
    StandardDistill2StairConfig,
)


CONF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conf"))


def _load(filename):
    with open(os.path.join(CONF_DIR, filename), "rb") as config_file:
        return tomllib.load(config_file)


def test_stair_stage_remains_a_reproducible_visual_continuation():
    assert issubclass(StandardDistill2StairConfig, StandardDistill1Config)
    assert StandardDistill2StairConfig.algorithm == "lbc_loco"
    assert StandardDistill2StairConfig.ckpt_name == "model.ckpt-standard"
    assert StandardDistill2StairConfig.num_goal_obs == 0


def test_only_distribution_and_resume_guard_differ_from_d1():
    base = _load("train_env_conf_standard_standard_distill_1.toml")
    stair = _load("train_env_conf_standard_standard_distill_2_stair.toml")

    base_stage = base.pop("standard_distill_1")
    stair_stage = stair.pop("standard_distill_2_stair")
    assert stair_stage.pop("require_student_resume") is True
    assert stair_stage == base_stage

    base_terrain = base.pop("terrain")
    stair_terrain = stair.pop("terrain")
    assert base == stair

    assert stair_terrain["mode"] == base_terrain["mode"] == "standard"
    assert stair_terrain["curriculum"] is base_terrain["curriculum"] is True
    assert stair_terrain["max_init_terrain_level"] == 4

    base_terrain = copy.deepcopy(base_terrain)
    stair_terrain = copy.deepcopy(stair_terrain)
    stair_terrain["max_init_terrain_level"] = base_terrain["max_init_terrain_level"]
    stair_terrain["standard"] = base_terrain["standard"]
    assert stair_terrain == base_terrain


def test_stair_distribution_matches_experiment():
    config = _load("train_env_conf_standard_standard_distill_2_stair.toml")
    standard = config["terrain"]["standard"]
    proportions = {
        name: values["proportion"]
        for name, values in standard.items()
    }
    assert proportions == {
        "pyramid_slope": 0.05,
        "pyramid_slope_inv": 0.05,
        "pyramid_stairs": 0.30,
        "pyramid_stairs_inv": 0.60,
        "maze": 0.0,
    }
    assert sum(proportions.values()) == 1.0
    assert "rewards" not in config
