"""Static ownership checks for the active P4 training baseline.

These checks deliberately inspect source only.  They must remain runnable from
the repository root without importing PyTorch, Isaac, archived experiments, or
shared documentation.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SERVER_ROOT = REPOSITORY_ROOT / "server"
CANONICAL_COMMAND_CONTRACT = SERVER_ROOT / "agent_ppo" / "p4" / "contracts.py"
COMPATIBILITY_FACADE = SERVER_ROOT / "agent_ppo" / "feature" / "p4_contract.py"
P4_PACKAGE_ROOT = SERVER_ROOT / "agent_ppo" / "p4"
P4_ORCHESTRATION_FILES = (
    SERVER_ROOT / "agent_ppo" / "algorithm" / "algorithm_p4_nav_ppo.py",
    SERVER_ROOT / "agent_ppo" / "workflow" / "nav_ppo_runtime.py",
    SERVER_ROOT / "agent_ppo" / "workflow" / "p4_nav_ppo_workflow.py",
)
P4_SUPPORT_FILES = (
    SERVER_ROOT / "agent_ppo" / "workflow" / "nav_ppo_metrics.py",
    SERVER_ROOT / "agent_ppo" / "workflow" / "nav_ppo_support.py",
)
TENSORIZED_KERNEL_ALLOWLIST = {
    "server/agent_ppo/p4/primitives.py:teacher_guidance_loss",
    "server/agent_ppo/p4/primitives.py:instant_r4_teacher_guidance_loss",
}

# These are active runtime owners.  Tests, developer tools, archive snapshots,
# and shared documentation are intentionally outside this boundary.
ACTIVE_RUNTIME_ROOTS = (
    SERVER_ROOT / "agent_ppo" / "algorithm",
    SERVER_ROOT / "agent_ppo" / "feature",
    SERVER_ROOT / "agent_ppo" / "workflow",
    SERVER_ROOT / "agent_ppo" / "conf",
    SERVER_ROOT / "agent_ppo" / "p4",
)
ACTIVE_RUNTIME_FILES = (SERVER_ROOT / "agent_ppo" / "checkpoint_io.py",)

# P4 profiles are compatibility contracts, not free-form experiment labels.
# Runtime code must consume the registry API.  Profile literals are permitted
# only in the registry and legacy compatibility declaration; tests are outside
# this production-source scan and may use literals as fixtures.
P4_PROFILE_LITERALS = {
    "maze_credit_repair",
    "maze_closed_loop_v3",
    "maze_instant_command_r4",
    "maze_instant_repair2h",
    "full_track",
}
PROFILE_LITERAL_OWNERS = {
    "server/agent_ppo/p4/profiles.py",
    "server/agent_ppo/p4/legacy_profiles.py",
}
NON_PROFILE_LITERAL_OWNERS = {
    # This is a reset-position bucket name, not a P4 training profile.
    "server/agent_ppo/feature/hard_start_replay.py": {"full_track"},
}


def _active_python_paths() -> list[Path]:
    result: list[Path] = []
    for root in ACTIVE_RUNTIME_ROOTS:
        result.extend(
            path
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    result.extend(ACTIVE_RUNTIME_FILES)
    return sorted(result)


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_active_runtime_does_not_import_archive_or_shared():
    violations: list[str] = []
    for path in _active_python_paths():
        for node in ast.walk(_tree(path)):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(
                name == "archive"
                or name.startswith("archive.")
                or name == "shared"
                or name.startswith("shared.")
                for name in names
            ):
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
    assert not violations, "active P4 runtime imports archival/shared code: " + ", ".join(violations)


def _p4_contract_definitions(name: str):
    definitions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for path in (CANONICAL_COMMAND_CONTRACT, COMPATIBILITY_FACADE):
        for node in ast.walk(_tree(path)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                definitions[path.relative_to(REPOSITORY_ROOT).as_posix()] = node
    return definitions


def test_p4_contracts_have_one_canonical_definition_and_thin_facade():
    canonical = CANONICAL_COMMAND_CONTRACT.relative_to(REPOSITORY_ROOT).as_posix()
    facade = COMPATIBILITY_FACADE.relative_to(REPOSITORY_ROOT).as_posix()
    for name in (
        "command_contract",
        "reward_contract",
        "training_contract",
        "contract_metadata",
    ):
        definitions = _p4_contract_definitions(name)
        assert set(definitions) == {canonical, facade}
        facade_returns = [
            node.value
            for node in ast.walk(definitions[facade])
            if isinstance(node, ast.Return)
        ]
        assert len(facade_returns) == 1
        returned = facade_returns[0]
        assert isinstance(returned, ast.Call)
        assert isinstance(returned.func, ast.Attribute)
        assert isinstance(returned.func.value, ast.Name)
        assert returned.func.value.id == "_canonical_contracts"
        assert returned.func.attr == name


def test_production_profile_literals_are_centralized():
    violations: list[str] = []
    for path in _active_python_paths():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        for node in ast.walk(_tree(path)):
            if (
                isinstance(node, ast.Constant)
                and node.value in P4_PROFILE_LITERALS
                and relative not in PROFILE_LITERAL_OWNERS
                and node.value not in NON_PROFILE_LITERAL_OWNERS.get(relative, set())
            ):
                violations.append(f"{relative}:{node.lineno}:{node.value}")
    assert not violations, "P4 profile literals must use the registry: " + ", ".join(violations)


def test_p4_modules_and_functions_stay_reviewable():
    violations: list[str] = []
    for path in P4_PACKAGE_ROOT.glob("*.py"):
        if len(path.read_text(encoding="utf-8").splitlines()) > 1_200:
            violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:module>1200")
    for path in (*P4_ORCHESTRATION_FILES, *P4_SUPPORT_FILES):
        limit = 1_000 if path in P4_ORCHESTRATION_FILES else 1_200
        if len(path.read_text(encoding="utf-8").splitlines()) > limit:
            violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:module>{limit}")
    checked = [*P4_PACKAGE_ROOT.glob("*.py"), *P4_ORCHESTRATION_FILES, *P4_SUPPORT_FILES]
    for path in checked:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        for node in ast.walk(_tree(path)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            key = f"{relative}:{node.name}"
            if node.end_lineno - node.lineno + 1 > 150 and key not in TENSORIZED_KERNEL_ALLOWLIST:
                violations.append(f"{key}:function>150")
    assert not violations, "P4 reviewability boundary drift: " + ", ".join(violations)
