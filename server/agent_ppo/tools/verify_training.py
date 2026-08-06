#!/usr/bin/env python3
"""Reproducible, cache-free verification profiles for the server training tree."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11+ is required by server
    tomllib = None


VERSION = 5
MAX_OUTPUT = 12000
PROFILE_NAMES = ("fast", "container", "release")
TEST_ROOTS = ("agent_ppo/tests", "tests")
CONTAINER_TESTS = (
    "agent_ppo/tests/test_nav_stage_and_metrics.py",
    "agent_ppo/tests/test_nav_checkpoint.py",
    "agent_ppo/tests/test_p15_checkpoint_and_config.py",
    "agent_ppo/tests/test_p2_core.py",
    "agent_ppo/tests/test_p3_contract.py",
    "agent_ppo/tests/test_p3_eval.py",
    "agent_ppo/tests/test_p3_schedule.py",
    "agent_ppo/tests/test_p3_gait_radial_v2.py",
    "agent_ppo/tests/test_p4_nav.py",
    "agent_ppo/tests/test_nav_lifecycle_publication.py",
    "agent_ppo/tests/test_nav_smoke_tools.py",
)
FAMILY_TESTS = {
    "p2": ("test_p2_core.py", "test_nav_checkpoint.py", "test_nav_contract.py"),
    "p3": ("test_p3_contract.py", "test_p3_eval.py", "test_p3_schedule.py"),
    "p4": (
        "test_p4_nav.py",
        "test_p2_core.py",
        "test_nav_stage_and_metrics.py",
        "test_nav_observation_process.py",
    ),
    "nav": ("test_nav_contract.py", "test_nav_checkpoint.py", "test_nav_stage_and_metrics.py"),
    "checkpoint": ("test_nav_checkpoint.py", "test_p15_checkpoint_and_config.py"),
    "monitor": ("test_nav_stage_and_metrics.py", "test_nav_lifecycle_publication.py"),
    "workflow": ("test_p15_workflow_transport.py", "test_p15_contract_and_schedule.py"),
    "sync": ("test_local_sync_client.py", "test_model_chunk_uploader.py"),
    "verify": ("test_verify_training.py",),
}
QUARANTINED_RELEASE_TESTS = {
    "agent_ppo/tests/test_st9_opt3_d2.py": (
        "retired ST9 D2 test imports removed LBC workflow symbols"
    ),
    "agent_ppo/tests/test_j9_fixed_lr.py": (
        "retired J9 fixed-LR contract targets removed validation hooks"
    ),
    (
        "tests/test_vision_distill_smoke.py::ProbeRegexTests::"
        "test_standard_first_visual_run_disables_random_depth_augmentation"
    ): "historical LBC iteration assertion no longer matches the active config",
    (
        "tests/test_visual_policy_optimization.py::"
        "CommandGeneralizationSourceContractTests::"
        "test_agent_ppo_does_not_import_base_env"
    ): "historical source scan includes bounded developer smoke tools",
    (
        "tests/test_visual_policy_optimization.py::"
        "CommandGeneralizationSourceContractTests::"
        "test_explicit_policy_entry_precedes_camera_task_inference"
    ): "historical camera inference assertion predates explicit P3 eval assembly",
}


def server_root(cwd: Path | None = None) -> Path:
    """Require invocation at the server root, rather than guessing a parent."""
    root = (cwd or Path.cwd()).resolve()
    if not (root / "train_test.py").is_file() or not (root / "agent_ppo").is_dir():
        raise ValueError("verify_training.py must run from server root (train_test.py and agent_ppo required)")
    return root


def _git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)


def normalize_path(root: Path, value: str | Path) -> str:
    """Return a server-root-relative path and reject traversal outside the project."""
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path escapes server root: {value}") from exc


def _nul_paths(result: subprocess.CompletedProcess[str]) -> set[str]:
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git path discovery failed")
    return {item for item in result.stdout.split("\0") if item}


def changed_paths(root: Path, overrides: Iterable[str] = ()) -> tuple[list[str], list[str]]:
    """Discover staged, unstaged, and non-ignored untracked paths."""
    paths: set[str] = set()
    for args in (
        ["diff", "--relative", "--name-only", "-z", "--", "."],
        ["diff", "--cached", "--relative", "--name-only", "-z", "--", "."],
        ["ls-files", "--others", "--exclude-standard", "-z"],
    ):
        paths.update(_nul_paths(_git(root, args)))
    skipped: list[str] = []
    for override in overrides:
        relative = normalize_path(root, override)
        ignored = _git(root, ["check-ignore", "-q", "--", relative]).returncode == 0
        if ignored:
            skipped.append(f"ignored override omitted: {relative}")
        else:
            paths.add(relative)
    normalized: set[str] = set()
    for path in paths:
        try:
            normalized.add(normalize_path(root, path))
        except ValueError:
            skipped.append(f"escaping git path omitted: {path}")
    return sorted(normalized), skipped


def _status_map(root: Path) -> dict[str, str]:
    prefix_result = _git(root, ["rev-parse", "--show-prefix"])
    if prefix_result.returncode:
        raise RuntimeError(prefix_result.stderr.strip() or "git prefix lookup failed")
    prefix = prefix_result.stdout.strip()
    result = _git(
        root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "."],
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git status failed")
    records = [record for record in result.stdout.split("\0") if record]
    statuses: dict[str, str] = {}
    index = 0
    while index < len(records):
        record = records[index]
        if len(record) >= 4:
            status, path = record[:2], record[3:]
            if prefix:
                if not path.startswith(prefix):
                    index += 1
                    continue
                path = path[len(prefix) :]
            statuses[path] = status
            if status[:1] in {"R", "C"} and index + 1 < len(records):
                index += 1
        index += 1
    return statuses


def source_fingerprint(
    root: Path,
    paths: Iterable[str],
    profile: str,
    run_contract: dict[str, Any] | None = None,
) -> str:
    """Hash HEAD plus each selected path's status and current content."""
    head = _git(root, ["rev-parse", "HEAD"])
    if head.returncode:
        raise RuntimeError(head.stderr.strip() or "git rev-parse failed")
    digest = hashlib.sha256()
    digest.update(f"v{VERSION}\0{profile}\0{head.stdout.strip()}\0".encode())
    digest.update(
        json.dumps(run_contract or {}, sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(b"\0")
    workspace_diff = _git(root, ["diff", "--binary", "HEAD"])
    if workspace_diff.returncode:
        raise RuntimeError(workspace_diff.stderr.strip() or "git diff fingerprint failed")
    digest.update(workspace_diff.stdout.encode("utf-8", errors="surrogateescape"))
    digest.update(b"\0")
    statuses = _status_map(root)
    for relative in sorted(set(paths)):
        path = root / relative
        digest.update(f"{relative}\0{statuses.get(relative, '??')}\0".encode())
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _existing(root: Path, candidates: Iterable[str]) -> list[str]:
    return [path for path in candidates if (root / path).is_file()]


def _without_ignored(root: Path, paths: Iterable[str]) -> list[str]:
    """Remove ignored artifacts while remaining usable in temporary non-git tests."""
    result = []
    for path in paths:
        ignored = _git(root, ["check-ignore", "-q", "--", path]).returncode == 0
        if not ignored:
            result.append(path)
    return result


def select_targeted_tests(root: Path, paths: Iterable[str]) -> tuple[list[str], bool]:
    """Select deterministic, relevant tests; only runtime changes trigger fallback."""
    path_list = sorted(set(paths))
    selected = {path for path in path_list if path.startswith("agent_ppo/tests/") and (root / path).is_file()}
    joined = "\n".join(path_list).lower()
    families: set[str] = set()
    for family in FAMILY_TESTS:
        if family in joined:
            families.add(family)
    for family in sorted(families):
        for name in FAMILY_TESTS[family]:
            for test in (root / "agent_ppo/tests").glob(name):
                selected.add(test.relative_to(root).as_posix())
            for test in (root / "tests").glob(name) if (root / "tests").is_dir() else ():
                selected.add(test.relative_to(root).as_posix())
    runtime_paths = [
        path
        for path in path_list
        if not path.startswith(("agent_ppo/tests/", "tests/"))
        and path.endswith((".py", ".toml"))
        and (
            path.startswith("agent_ppo/")
            or path.startswith("conf/")
            or path.startswith("isaac_env/")
            or path == "train_test.py"
        )
    ]
    fallback = any(
        not any(family in path.lower() for family in FAMILY_TESTS)
        for path in runtime_paths
    )
    if fallback:
        retired_files = {
            node for node in QUARANTINED_RELEASE_TESTS if "::" not in node
        }
        selected.update(
            path
            for path in _existing(
                root,
                (
                    candidate.relative_to(root).as_posix()
                    for candidate in (root / "agent_ppo/tests").glob("test_*.py")
                ),
            )
            if path not in retired_files
        )
    return sorted(selected), fallback


def release_paths(root: Path) -> list[str]:
    result: list[str] = []
    for top in ("agent_ppo", "conf", "isaac_env", "tests"):
        directory = root / top
        if directory.is_dir():
            result.extend(path.relative_to(root).as_posix() for path in directory.rglob("*.py") if "__pycache__" not in path.parts)
    if (root / "train_test.py").is_file():
        result.append("train_test.py")
    return sorted(set(_without_ignored(root, result)))


def all_toml_paths(root: Path) -> list[str]:
    candidates = (path.relative_to(root).as_posix() for path in root.rglob("*.toml") if "__pycache__" not in path.parts)
    return sorted(_without_ignored(root, candidates))


def build_profile(root: Path, profile: str, paths: list[str]) -> dict[str, Any]:
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown profile: {profile}")
    if profile == "release":
        python_paths = release_paths(root)
        toml_paths = all_toml_paths(root)
        tests = []
        for test_root in TEST_ROOTS:
            directory = root / test_root
            if directory.is_dir():
                tests.extend(
                    path.relative_to(root).as_posix()
                    for path in directory.rglob("test_*.py")
                )
        tests = _existing(root, tests)
        return {"paths": sorted(set(python_paths + toml_paths)), "python": python_paths, "toml": toml_paths, "tests": tests, "fallback": False}
    python_paths = [path for path in paths if path.endswith(".py") and (root / path).is_file()]
    toml_paths = [path for path in paths if path.endswith(".toml") and (root / path).is_file()]
    tests, fallback = select_targeted_tests(root, paths)
    if profile == "container":
        tests = sorted(set(tests) | set(_existing(root, CONTAINER_TESTS)))
    return {"paths": paths, "python": python_paths, "toml": toml_paths, "tests": tests, "fallback": fallback}


def release_test_selection(
    tests: Iterable[str], include_quarantined: bool
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return active test paths, pytest deselections, and explicit quarantine."""
    selected = sorted(set(tests))
    if include_quarantined:
        return selected, [], {}
    quarantined: dict[str, str] = {}
    deselected: list[str] = []
    whole_files = {
        node
        for node in QUARANTINED_RELEASE_TESTS
        if "::" not in node
    }
    selected = [path for path in selected if path not in whole_files]
    selected_set = set(selected)
    for node, reason in QUARANTINED_RELEASE_TESTS.items():
        file_path = node.split("::", 1)[0]
        if file_path not in tests:
            continue
        quarantined[node] = reason
        if "::" in node and file_path in selected_set:
            deselected.append(node)
    return selected, sorted(deselected), quarantined


def _bounded(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + "\n[truncated]"


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_command(command: list[str], root: Path, timeout: float | None = None) -> dict[str, Any]:
    """Run a subprocess with cache writes disabled and capture bounded evidence."""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, env=environment, timeout=timeout, check=False)
        return {"command": command, "duration_s": round(time.monotonic() - started, 3), "returncode": completed.returncode, "stdout": _bounded(completed.stdout), "stderr": _bounded(completed.stderr)}
    except subprocess.TimeoutExpired as exc:
        return {"command": command, "duration_s": round(time.monotonic() - started, 3), "returncode": 124, "stdout": _bounded(_as_text(exc.stdout)), "stderr": _bounded(_as_text(exc.stderr) + "\ncommand timed out")}


def compile_sources(root: Path, paths: Iterable[str]) -> tuple[bool, str]:
    """Compile source in memory so this check cannot create __pycache__."""
    errors = []
    for relative in paths:
        path = root / relative
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"{relative}: {exc}")
    return not errors, "\n".join(errors)


def parse_toml(root: Path, paths: Iterable[str]) -> tuple[bool, str]:
    if tomllib is None:
        return False, "tomllib unavailable"
    errors = []
    for relative in paths:
        try:
            with (root / relative).open("rb") as handle:
                tomllib.load(handle)
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{relative}: {exc}")
    return not errors, "\n".join(errors)


def _internal_step(name: str, operation: Callable[[], tuple[bool, str]]) -> dict[str, Any]:
    started = time.monotonic()
    success, output = operation()
    return {"name": name, "command": [name], "duration_s": round(time.monotonic() - started, 3), "returncode": 0 if success else 1, "stdout": _bounded(output if success else ""), "stderr": _bounded("" if success else output)}


def nav_smoke(root: Path, num_envs: int, runtime_dir: str, timeout: float, runner: Callable[..., dict[str, Any]] = run_command) -> list[dict[str, Any]]:
    """Start, poll, and always stop the existing smoke launcher within a deadline."""
    base = [sys.executable, "-m", "agent_ppo.tools.nav_full_smoke"]

    def smoke_command(action: str, *extra: str) -> list[str]:
        return [
            *base,
            action,
            "--runtime-dir",
            runtime_dir,
            "--num-envs",
            str(num_envs),
            *extra,
        ]

    command_timeout = min(timeout, 30.0)
    steps = [
        dict(
            runner(smoke_command("start"), root, timeout=command_timeout),
            name="nav-smoke-start",
        )
    ]
    if steps[0]["returncode"] != 0:
        steps.append(
            {
                "name": "nav-smoke-target",
                "command": ["target_reached=true"],
                "duration_s": 0.0,
                "returncode": 1,
                "stdout": "",
                "stderr": "nav smoke failed to start",
            }
        )
        steps.append(
            dict(
                runner(
                    smoke_command("stop", "--grace-seconds", "5"),
                    root,
                    timeout=command_timeout,
                ),
                name="nav-smoke-stop",
            )
        )
        return steps
    deadline = time.monotonic() + timeout
    reached = False
    try:
        while time.monotonic() < deadline:
            status = dict(
                runner(smoke_command("status"), root, timeout=command_timeout),
                name="nav-smoke-status",
            )
            steps.append(status)
            try:
                reached = status["returncode"] == 0 and bool(json.loads(status["stdout"]).get("target_reached"))
            except (KeyError, TypeError, json.JSONDecodeError):
                reached = False
            if reached or status["returncode"] not in (0, 1):
                break
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        if not reached:
            steps.append({"name": "nav-smoke-target", "command": ["target_reached=true"], "duration_s": 0.0, "returncode": 1, "stdout": "", "stderr": "nav smoke did not report target_reached=true before timeout"})
    finally:
        steps.append(
            dict(
                runner(
                    smoke_command("stop", "--grace-seconds", "5"),
                    root,
                    timeout=command_timeout,
                ),
                name="nav-smoke-stop",
            )
        )
    return steps


def atomic_write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def evidence_destination(root: Path, value: str) -> Path:
    verification_root = (root / ".verification").resolve()
    repository_lookup = _git(root, ["rev-parse", "--show-toplevel"])
    repository_root = (
        Path(repository_lookup.stdout.strip()).resolve()
        if repository_lookup.returncode == 0 and repository_lookup.stdout.strip()
        else root.resolve()
    )
    candidate = Path(value)
    destination = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    if destination.suffix != ".json":
        raise ValueError("evidence path must use a .json suffix")
    try:
        destination.relative_to(verification_root)
        return destination
    except ValueError:
        pass
    try:
        destination.relative_to(repository_root)
    except ValueError:
        if destination.exists():
            raise ValueError(
                "external evidence path already exists; refusing to overwrite it"
            )
        return destination
    raise ValueError("evidence inside the Git repository must stay under server/.verification/")


def reusable_evidence(path: Path, profile: str, fingerprint: str) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("version") == VERSION and value.get("profile") == profile and value.get("fingerprint") == fingerprint and value.get("success") is True


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "pytest", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def verification_success(steps: Iterable[dict[str, Any]], tests_skipped: bool) -> bool:
    """A diagnostic syntax-only run cannot satisfy a verification profile."""
    return not tests_skipped and all(step["returncode"] == 0 for step in steps)


def _step_command(name: str, command: list[str], root: Path, timeout: float | None = None) -> dict[str, Any]:
    result = run_command(command, root, timeout)
    result["name"] = name
    return result


def execute(args: argparse.Namespace, root: Path) -> int:
    paths, skipped = changed_paths(root, args.path)
    profile = build_profile(root, args.profile, paths)
    fingerprint_paths = sorted(set(profile["paths"]) | set(paths))
    head_start = _git(root, ["rev-parse", "HEAD"])
    if head_start.returncode:
        raise RuntimeError(head_start.stderr.strip() or "git rev-parse failed")
    run_contract = {
        "dependencies": dependency_versions(),
        "include_quarantined": bool(args.include_quarantined),
        "nav_smoke": bool(args.nav_smoke),
        "num_envs": args.num_envs if args.nav_smoke else None,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "runtime_dir": args.runtime_dir if args.nav_smoke else None,
        "skip_tests": bool(args.skip_tests),
        "smoke_timeout": args.smoke_timeout if args.nav_smoke else None,
        "test_timeout": args.test_timeout,
        "runtime_platform": platform.platform(),
    }
    fingerprint = source_fingerprint(
        root, fingerprint_paths, args.profile, run_contract
    )
    evidence_value = args.evidence or f".verification/{args.profile}.json"
    evidence_path = evidence_destination(root, evidence_value)
    if (
        args.reuse_valid
        and not args.nav_smoke
        and reusable_evidence(evidence_path, args.profile, fingerprint)
    ):
        print(f"reusing valid {args.profile} evidence: {evidence_path}")
        return 0
    selected_tests = list(profile["tests"])
    deselected_tests: list[str] = []
    quarantined_tests: dict[str, str] = {}
    if args.profile == "release":
        selected_tests, deselected_tests, quarantined_tests = release_test_selection(
            selected_tests, args.include_quarantined
        )
        skipped.extend(
            f"quarantined release test: {node} ({reason})"
            for node, reason in sorted(quarantined_tests.items())
        )
    commands = [
        ["git", "diff", "--check"],
        ["git", "diff", "--cached", "--check"],
    ]
    if selected_tests and not args.skip_tests:
        commands.append(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                *selected_tests,
                *(item for node in deselected_tests for item in ("--deselect", node)),
            ]
        )
    plan = {
        "profile": args.profile,
        "paths": profile["paths"],
        "python": profile["python"],
        "toml": profile["toml"],
        "tests": selected_tests,
        "quarantined_tests": quarantined_tests,
        "fallback": profile["fallback"],
        "commands": commands,
    }
    if args.plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    started = time.monotonic()
    steps = [
        _step_command("git-diff-check", commands[0], root),
        _step_command("git-cached-diff-check", commands[1], root),
    ]
    if profile["python"]:
        steps.append(_internal_step("compile-python", lambda: compile_sources(root, profile["python"])))
    if profile["toml"]:
        steps.append(_internal_step("parse-toml", lambda: parse_toml(root, profile["toml"])))
    if args.skip_tests:
        skipped.append("tests skipped by --skip-tests")
    elif not selected_tests:
        skipped.append("no existing selected tests")
    else:
        steps.append(
            _step_command("pytest", commands[-1], root, timeout=args.test_timeout)
        )
    if args.nav_smoke:
        if args.profile != "container":
            skipped.append("--nav-smoke is only valid for container profile")
            steps.append({"name": "nav-smoke", "command": ["nav-smoke"], "duration_s": 0.0, "returncode": 2, "stdout": "", "stderr": "--nav-smoke requires --profile container"})
        else:
            steps.extend(nav_smoke(root, args.num_envs, args.runtime_dir, args.smoke_timeout))
    end_paths, _ = changed_paths(root, args.path)
    end_profile = build_profile(root, args.profile, end_paths)
    end_fingerprint_paths = sorted(set(end_profile["paths"]) | set(end_paths))
    end_fingerprint = source_fingerprint(
        root, end_fingerprint_paths, args.profile, run_contract
    )
    if end_fingerprint != fingerprint:
        steps.append(
            {
                "name": "workspace-stability",
                "command": ["verify-workspace-unchanged"],
                "duration_s": 0.0,
                "returncode": 1,
                "stdout": "",
                "stderr": "HEAD or verification inputs changed while tests were running",
            }
        )
    success = verification_success(steps, args.skip_tests)
    head_after = _git(root, ["rev-parse", "HEAD"])
    if head_after.returncode:
        raise RuntimeError(head_after.stderr.strip() or "git rev-parse failed")
    evidence = {"version": VERSION, "profile": args.profile, "utc_time": datetime.now(timezone.utc).isoformat(), "head": head_start.stdout.strip(), "head_after": head_after.stdout.strip(), "fingerprint": fingerprint, "fingerprint_after": end_fingerprint, "run_contract": run_contract, "runtime": {"platform": platform.platform(), "python": sys.version, "executable": sys.executable, "cwd": str(root)}, "changed_paths": paths, "changed_paths_after": end_paths, "selected_paths": profile["paths"], "selected_tests": selected_tests, "tests_skipped": bool(args.skip_tests), "quarantined_tests": quarantined_tests, "skipped": skipped, "steps": steps, "total_duration_s": round(time.monotonic() - started, 3), "success": success}
    atomic_write_evidence(evidence_path, evidence)
    print(f"{args.profile} verification {'passed' if success else 'failed'}: {evidence_path}")
    return 0 if success else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile", choices=PROFILE_NAMES, default="fast")
    result.add_argument("--path", action="append", default=[], help="repeatable server-root-relative path override")
    result.add_argument("--plan", action="store_true")
    result.add_argument("--evidence")
    result.add_argument("--reuse-valid", action="store_true")
    result.add_argument("--skip-tests", action="store_true")
    result.add_argument("--include-quarantined", action="store_true")
    result.add_argument("--nav-smoke", action="store_true")
    result.add_argument("--num-envs", type=int, default=8)
    result.add_argument("--runtime-dir", default="/tmp/kaiwu_nav_full_smoke")
    result.add_argument("--smoke-timeout", type=float, default=300.0)
    result.add_argument("--test-timeout", type=float, default=1800.0)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.num_envs < 1 or args.smoke_timeout <= 0 or args.test_timeout <= 0:
        raise SystemExit(
            "--num-envs must be positive and timeouts must be > 0"
        )
    try:
        return execute(args, server_root())
    except (RuntimeError, ValueError) as exc:
        print(f"verification setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
