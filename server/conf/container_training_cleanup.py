#!/usr/bin/env python3
"""Prepare a Tencent Kaiwu container workspace for training-task snapshotting."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


DEFAULT_CODE_ROOT = Path("/workspace/code")
DEFAULT_PROJECT_ROOT = Path("/data/projects/legged_robot_competition_26")
DEFAULT_TMP_ROOT = Path("/tmp")
CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def validate_roots(code_root: Path, project_root: Path) -> None:
    if not code_root.is_absolute() or not project_root.is_absolute():
        raise ValueError("cleanup roots must be absolute")
    if not (code_root / "agent_ppo" / "agent.py").is_file():
        raise ValueError(f"not a Kaiwu code root: {code_root}")
    if not (code_root / "conf").is_dir():
        raise ValueError(f"missing protected conf directory: {code_root / 'conf'}")
    if not (project_root / "kaiwudrl").is_dir():
        raise ValueError(f"not a Kaiwu project root: {project_root}")


def _assert_safe_target(
    target: Path,
    roots: tuple[Path, ...],
    protected_paths: tuple[Path, ...],
) -> None:
    absolute = target.absolute()
    if absolute in roots:
        raise ValueError(f"refusing to delete cleanup root: {absolute}")
    if not any(absolute.is_relative_to(root) for root in roots):
        raise ValueError(f"cleanup target escaped allowed roots: {absolute}")
    if ".env" in absolute.parts or absolute in protected_paths:
        raise ValueError(f"refusing to delete protected config path: {absolute}")


def collect_targets(
    code_root: Path,
    project_root: Path,
    tmp_root: Path,
    *,
    remove_dev_files: bool,
) -> list[Path]:
    roots = tuple(path.absolute() for path in (code_root, project_root, tmp_root))
    protected_paths = ((code_root / "conf").absolute(),)
    targets: set[Path] = set()

    for directory in code_root.rglob("*"):
        if directory.is_dir() and directory.name in CACHE_DIR_NAMES:
            targets.add(directory)

    for fixed in (
        code_root / "agent_ppo" / "test_artifacts",
        tmp_root / "IsaacLab",
    ):
        if _lexists(fixed):
            targets.add(fixed)

    for pattern in (".ide-sync-*.tar.gz", "*.sync-tmp"):
        targets.update(path for path in project_root.glob(pattern) if _lexists(path))
    for pattern in ("p3*.log", "p3-*.pkl"):
        targets.update(path for path in tmp_root.glob(pattern) if _lexists(path))

    if remove_dev_files:
        for fixed in (
            code_root / ".git",
            code_root / ".vscode",
            code_root / "agent_ppo" / "tests",
            code_root / "agent_ppo" / "tools",
        ):
            if _lexists(fixed):
                targets.add(fixed)

    ordered = sorted(targets, key=lambda path: (len(path.parts), str(path)), reverse=True)
    for target in ordered:
        _assert_safe_target(target, roots, protected_paths)
    return ordered


def path_size(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        try:
            return path.lstat().st_size
        except FileNotFoundError:
            return 0
    total = 0
    for root, directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in files:
            try:
                total += (root_path / name).lstat().st_size
            except FileNotFoundError:
                pass
        directories[:] = [
            name for name in directories if not (root_path / name).is_symlink()
        ]
    return total


def remove_target(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def run_cleanup(targets: list[Path], *, apply: bool) -> tuple[int, int]:
    reclaimable = sum(path_size(path) for path in targets)
    if apply:
        for target in targets:
            remove_target(target)
    return len(targets), reclaimable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--tmp-root", type=Path, default=DEFAULT_TMP_ROOT)
    parser.add_argument("--apply", action="store_true", help="delete listed targets")
    parser.add_argument(
        "--remove-dev-files",
        action="store_true",
        help="also remove .git, .vscode, agent_ppo/tests and agent_ppo/tools",
    )
    args = parser.parse_args()

    code_root = args.code_root.absolute()
    project_root = args.project_root.absolute()
    tmp_root = args.tmp_root.absolute()
    validate_roots(code_root, project_root)
    targets = collect_targets(
        code_root,
        project_root,
        tmp_root,
        remove_dev_files=args.remove_dev_files,
    )
    count, reclaimable = run_cleanup(targets, apply=args.apply)
    print(f"mode={'apply' if args.apply else 'dry-run'}")
    print(f"targets={count}")
    print(f"reclaimable_bytes={reclaimable}")
    for target in targets:
        print(f"{'deleted' if args.apply else 'candidate'}={target}")
    print(f"protected={code_root / 'conf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
