#!/usr/bin/env python3
"""Architecture guards for the P4 algorithm service split."""

import ast
from pathlib import Path

from agent_ppo.algorithm.algorithm_p4_nav_ppo import AlgorithmP4NavPPO
from agent_ppo.p4.checkpoint import P4CheckpointMixin
from agent_ppo.p4.diagnostics import P4DiagnosticsMixin
from agent_ppo.p4.rewards import P4RewardMixin
from agent_ppo.p4.runtime import P4RuntimeMixin
from agent_ppo.p4.teacher import P4TeacherMixin
from agent_ppo.p4.training import P4TrainingMixin


ALGORITHM_PATH = (
    Path(__file__).resolve().parents[1]
    / "algorithm"
    / "algorithm_p4_nav_ppo.py"
)


def test_p4_algorithm_keeps_only_orchestration_hooks():
    source = ALGORITHM_PATH.read_text()
    tree = ast.parse(source)
    algorithm = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AlgorithmP4NavPPO"
    )
    methods = {
        node.name
        for node in algorithm.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert methods == {
        "__init__",
        "_prepare_early_config",
        "_initialize_pre_parent_state",
        "_validate_runtime_contract",
        "_initialize_navigation_runtime_state",
        "_initialize_transport_camera_state",
        "_initialize_diagnostic_reward_state",
        "_initialize_teacher_state",
        "_freeze_parent_models",
        "_initialize_training_state",
        "frame_begin",
        "_frame_stuck_context",
        "_frame_teacher_masks",
        "_frame_parent_anchor_mask",
        "_store_frame_teacher_labels",
        "begin_rollout",
        "_transition_extras",
        "_run_ppo_epochs",
        "_actor_update_enabled",
        "_adapter_update",
        "finish_tick",
        "update",
    }
    assert len(source.splitlines()) < 1_000
    for node in algorithm.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start_line = min(
            [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
        )
        assert node.end_lineno - start_line + 1 <= 150, node.name


def test_p4_algorithm_uses_registry_constants_for_profile_identity():
    profile_literals = {
        "full_track",
        "maze_credit_repair",
        "maze_closed_loop_v3",
        "maze_instant_command_r4",
        "maze_instant_repair2h",
    }
    tree = ast.parse(ALGORITHM_PATH.read_text())
    used_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value in profile_literals
    }
    assert not used_literals


def test_p4_service_mixins_have_stable_mro_and_method_ownership():
    assert AlgorithmP4NavPPO.__mro__[1:7] == (
        P4RuntimeMixin,
        P4TeacherMixin,
        P4RewardMixin,
        P4DiagnosticsMixin,
        P4TrainingMixin,
        P4CheckpointMixin,
    )
    expected_modules = {
        "_map_policy_target": "agent_ppo.p4.runtime",
        "_actor_auxiliary_loss": "agent_ppo.p4.teacher",
        "_override_reward_components": "agent_ppo.p4.rewards",
        "_extra_tick_diagnostics": "agent_ppo.p4.diagnostics",
        "_apply_training_schedule": "agent_ppo.p4.training",
        "load_bundle": "agent_ppo.p4.checkpoint",
        "frame_begin": "agent_ppo.algorithm.algorithm_p4_nav_ppo",
        "finish_tick": "agent_ppo.algorithm.algorithm_p4_nav_ppo",
        "_run_ppo_epochs": "agent_ppo.algorithm.algorithm_p4_nav_ppo",
    }
    for method_name, module_name in expected_modules.items():
        assert getattr(AlgorithmP4NavPPO, method_name).__module__ == module_name
