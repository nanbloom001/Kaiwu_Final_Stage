"""Compatibility and ownership checks for the decomposed P4 contract facade."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from agent_ppo.feature import p4_contract
from agent_ppo.p4 import constants, contracts, diagnostics, profiles, rewards, teacher

P4_ROOT = Path(__file__).resolve().parents[1] / "p4"
FACADE_PATH = Path(__file__).resolve().parents[1] / "feature" / "p4_contract.py"


def _logical_line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
OWNED_MODULES = (
    "checkpoint",
    "contracts",
    "diagnostics",
    "primitives",
    "rewards",
    "teacher",
    "training",
)
TENSORIZED_KERNEL_ALLOWLIST = {
    "agent_ppo.p4.primitives.teacher_guidance_loss",
    "agent_ppo.p4.primitives.instant_r4_teacher_guidance_loss",
}


def test_facade_and_canonical_modules_stay_bounded():
    assert _logical_line_count(FACADE_PATH) <= 300
    for name in OWNED_MODULES:
        path = P4_ROOT / f"{name}.py"
        assert _logical_line_count(path) <= 1_200, path


def test_owned_p4_kernels_stay_bounded_with_exact_tensor_allowlist():
    over_limit = set()
    for module_name in OWNED_MODULES:
        path = P4_ROOT / f"{module_name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.end_lineno - node.lineno + 1 > 150:
                    over_limit.add(f"agent_ppo.p4.{module_name}.{node.name}")
    assert over_limit == TENSORIZED_KERNEL_ALLOWLIST


def test_owned_p4_modules_do_not_embed_registered_profile_names():
    embedded = set()
    names = set(profiles.PROFILE_REGISTRY)
    for module_name in OWNED_MODULES:
        tree = ast.parse((P4_ROOT / f"{module_name}.py").read_text(encoding="utf-8"))
        embedded.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in names
        )
    assert embedded == set()


def test_reward_and_diagnostic_specs_cover_the_profile_registry():
    assert set(rewards.REWARD_SPECS) == set(profiles.PROFILE_REGISTRY)
    assert set(diagnostics.DIAGNOSTIC_SPECS) == set(profiles.PROFILE_REGISTRY)
    for spec in rewards.REWARD_SPECS.values():
        assert spec.enabled_components.isdisjoint(spec.shadow_components)
    assert diagnostics.DIAGNOSTIC_SPECS[profiles.PROFILE_FULL_TRACK].full_track_segments
    assert not diagnostics.DIAGNOSTIC_SPECS[profiles.PROFILE_FULL_TRACK].maze_probe


def test_auxiliary_parameter_ownership_rejects_cross_module_gradients():
    owner = object.__new__(teacher.P4TeacherMixin)
    owner.navigation_encoder = torch.nn.Linear(1, 1)
    owner.actor = torch.nn.Linear(1, 1)
    owner.safety_head = torch.nn.Linear(1, 1)
    owner.stuck_head = torch.nn.Linear(1, 1)
    loss = owner.navigation_encoder.weight.sum()
    owner._assert_loss_parameter_ownership("camera", loss)
    with pytest.raises(RuntimeError, match="teacher loss crossed"):
        owner._assert_loss_parameter_ownership("teacher", loss)


def test_facade_defines_only_contract_compatibility_wrappers():
    tree = ast.parse(FACADE_PATH.read_text(encoding="utf-8"))
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert functions == {
        "__getattr__",
        "command_contract",
        "reward_contract",
        "training_contract",
        "contract_metadata",
    }


def test_facade_reexports_constants_and_stateless_implementations():
    for name in constants.__all__:
        assert getattr(p4_contract, name) is getattr(constants, name)
    assert p4_contract.map_normalized_action.__module__ == "agent_ppo.p4.runtime"
    assert p4_contract.teacher_guidance_loss.__module__ == "agent_ppo.p4.primitives"
    assert p4_contract.soft_cruise_penalty.__module__ == "agent_ppo.p4.rewards"
    assert p4_contract.training_schedule.__module__ == "agent_ppo.p4.training"
    assert p4_contract.stable_digest is contracts.stable_digest


@pytest.mark.parametrize(
    "profile",
    sorted(contracts._profile_registry.PROFILE_REGISTRY),
)
def test_facade_contracts_match_canonical_values(profile):
    assert p4_contract.command_contract(profile) == contracts.command_contract(profile)
    assert p4_contract.reward_contract(training_profile=profile) == (
        contracts.reward_contract(training_profile=profile)
    )
    assert p4_contract.training_contract(training_profile=profile) == (
        contracts.training_contract(training_profile=profile)
    )
    assert p4_contract.contract_metadata(training_profile=profile) == (
        contracts.contract_metadata(training_profile=profile)
    )
