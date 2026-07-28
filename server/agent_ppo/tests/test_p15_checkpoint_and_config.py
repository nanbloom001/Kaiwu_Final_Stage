#!/usr/bin/env python3

from pathlib import Path
import ast
import re
import tomllib

import torch

from agent_ppo.checkpoint_io import (
    KAIWU_TRAIN_FORMAT,
    normalize_kaiwu_train_bundle,
    p15_response_checkpoint_candidates,
    low_level_only_parent_candidates,
    validate_probe_filename,
)
from agent_ppo.conf.conf import P15ResponseConfig


SERVER_ROOT = Path(__file__).resolve().parents[2]


def test_schema1_normalizes_without_fabricating_adapter_state():
    actor = {"0.weight": torch.ones(2, 2)}
    critic = {"0.weight": torch.ones(1, 2)}
    vision = {"cnn.0.weight": torch.ones(1, 1)}
    bundle = {
        "format": KAIWU_TRAIN_FORMAT,
        "schema_version": 1,
        "modules": {
            "vision_encoder": {"state_dict": vision},
            "low_level": {"actor_state_dict": actor},
            "critic": {"state_dict": critic},
        },
        "optimizers": {"visual_ppo": {"state": {}, "param_groups": []}},
        "training_state": {"current_iteration": 34728},
    }
    normalized, report = normalize_kaiwu_train_bundle(bundle)
    assert report["source_schema"] == 1
    assert normalized["training_states"]["low_level"]["current_iteration"] == 34728
    assert normalized["modules"]["low_level"]["actor"]["state_dict"] is actor
    assert "high_level" not in normalized["modules"]


def test_p15_labels_are_probe_compatible():
    candidates = p15_response_checkpoint_candidates("/tmp", 123)
    assert len(candidates) == 4
    assert all(validate_probe_filename(path) for path in candidates)


def test_formal_config_is_eight_hour_standard_camera():
    path = SERVER_ROOT / "agent_ppo/conf/train_env_conf_standard_p15_response.toml"
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    assert config["env"]["num_envs"] == 128
    assert config["env"]["task"] == "standard"
    assert config["env_conf"]["task_name"] == "Unitree-Go2-Velocity-Camera"
    assert config["env_conf"]["policy_entry"] == "p15_response"
    stage = config["p15_response"]
    assert stage["run_name"] == "p15resp8h"
    assert stage["transition_parent_model_id"] == 34728
    assert stage["task_end_hours"] == 8.0
    assert stage["num_steps_per_env"] == 80
    assert P15ResponseConfig.num_steps_per_env == stage["num_steps_per_env"]
    assert config["env"]["num_envs"] * stage["num_steps_per_env"] == 10240
    assert stage["first_save_minutes"] == 10.0
    assert stage["resume_first_save_minutes"] == 10.0
    assert stage["save_interval_minutes"] == 10.0
    assert stage["response_checkpoint_minutes"] == []
    assert config["rewards"]["approach_goal"]["weight"] == 0.0
    assert config["rewards"]["reach_goal"]["weight"] == 0.0
    assert "pivot_turning" not in config["rewards"]
    assert config["domain_rand"]["enable_domain_rand"] is True
    assert config["domain_rand"]["randomize_friction"] is True
    assert config["domain_rand"]["friction_range"] == [1.0, 1.0]
    assert config["domain_rand"]["randomize_base_mass"] is False
    assert config["domain_rand"]["push_robots"] is False
    assert config["terrain"]["curriculum"] is False
    assert stage["response_adapter"]["burn_in_steps"] == 8
    assert stage["feedback_profile"]["version"] == "p15_feedback_v2"


def test_low_level_only_candidates_are_explicit_and_include_response_packages():
    candidates = low_level_only_parent_candidates("/tmp/models", 123)
    assert candidates[0].endswith("model.ckpt-responsecalib-123.pkl")
    assert any("commandfull-123.pkl" in path for path in candidates)


def test_p15_monitor_names_match_platform_constraints():
    path = SERVER_ROOT / "agent_ppo/conf/monitor_builder.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert 'monitor.title("P15动态响应训练")' in source
    assert 'monitor.title("P1.5动态响应训练")' not in source
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "P15_PANEL_SPECS"
            for target in node.targets
        )
    )
    panel_specs = ast.literal_eval(assignment.value)
    allowed = re.compile(r"^[A-Za-z0-9_\-*\u4e00-\u9fff ]{1,20}$")
    for chinese_name, _, _ in panel_specs:
        assert allowed.fullmatch(chinese_name), chinese_name
