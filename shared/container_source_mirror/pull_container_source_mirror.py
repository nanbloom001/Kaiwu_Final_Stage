#!/usr/bin/env python3
"""Build an incremental, read-only mirror of Tencent Kaiwu source/config files."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MIRROR_FORMAT = "tencent_kaiwu_source_mirror_v1"
DEFAULT_REMOTE_ROOTS = {
    "isaaclab": "/workspace/isaaclab",
    "unitree_rl_lab": "/workspace/unitree_rl_lab",
    "unitree_ros": "/workspace/unitree_ros",
    "kaiwu_runtime": "/data/projects/legged_robot_competition_26/isaac_env",
}
ALLOWED_SUFFIXES = {
    ".action",
    ".bash",
    ".bazel",
    ".bzl",
    ".c",
    ".cc",
    ".cfg",
    ".cmake",
    ".cpp",
    ".cu",
    ".cuh",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".ini",
    ".j2",
    ".jinja",
    ".json",
    ".launch",
    ".md",
    ".msg",
    ".py",
    ".proto",
    ".sdf",
    ".sh",
    ".srv",
    ".toml",
    ".txt",
    ".urdf",
    ".world",
    ".xml",
    ".xacro",
    ".yaml",
    ".yml",
    ".zsh",
}
ALLOWED_EXTENSIONLESS = {
    ".dockerignore",
    ".gitignore",
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "README",
}
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "ckpt",
    "dist",
    "log",
    "logs",
    "outputs",
    "runs",
    "runtime",
    "thirdparty",
    "venv",
}
SKIP_NAMES = {
    ".env",
    ".DS_Store",
}
SENSITIVE_NAME_FRAGMENTS = (
    "cookie",
    "credential",
    "private_key",
    "secret",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _eligible(rel_path: str, size: int, max_bytes: int) -> bool:
    path = Path(rel_path)
    if size < 0 or size > max_bytes:
        return False
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    if path.name in SKIP_NAMES:
        return False
    lower_name = path.name.lower()
    if any(fragment in lower_name for fragment in SENSITIVE_NAME_FRAGMENTS):
        return False
    return (
        path.suffix.lower() in ALLOWED_SUFFIXES
        or path.name in ALLOWED_EXTENSIONLESS
    )


def _decode_exec_output(result: dict[str, Any], field: str) -> bytes:
    value = result.get(field)
    if not isinstance(value, str) or not value:
        return b""
    return base64.b64decode(value, validate=True)


def _exec_remote(client, command: str, *, timeout: int = 120) -> bytes:
    encoded = base64.urlsafe_b64encode(command.encode("utf-8")).decode("ascii")
    result = client.get(
        "/exec_b64",
        cmd=encoded,
        cwd=".",
        timeout=str(timeout),
    )
    stdout = _decode_exec_output(result, "stdout_base64")
    stderr = _decode_exec_output(result, "stderr_base64")
    if result.get("stdout_truncated") or result.get("stderr_truncated"):
        raise RuntimeError("remote exec output was truncated")
    if result.get("timed_out"):
        raise RuntimeError(f"remote exec timed out: {stderr.decode(errors='replace')}")
    if int(result.get("returncode", 1)) != 0:
        raise RuntimeError(
            f"remote exec failed ({result.get('returncode')}): "
            f"{stderr.decode(errors='replace')}"
        )
    return stdout


def _platform_manifest(client, root: str, max_bytes: int) -> dict[str, Any]:
    payload = {
        "root": root,
        "max_bytes": int(max_bytes),
        "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
        "allowed_extensionless": sorted(ALLOWED_EXTENSIONLESS),
        "skip_dirs": sorted(SKIP_DIRS),
        "skip_names": sorted(SKIP_NAMES),
        "sensitive_fragments": list(SENSITIVE_NAME_FRAGMENTS),
    }
    remote_code = """
import hashlib
import json
import os
from pathlib import Path

cfg = json.loads(%s)
root = Path(cfg["root"]).resolve()
files = {}
for directory, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(name for name in dirnames if name not in cfg["skip_dirs"])
    directory_path = Path(directory)
    for filename in sorted(filenames):
        path = directory_path / filename
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path.name in cfg["skip_names"]:
            continue
        lower_name = path.name.lower()
        if any(fragment in lower_name for fragment in cfg["sensitive_fragments"]):
            continue
        stat = path.stat()
        if stat.st_size > cfg["max_bytes"]:
            continue
        if path.suffix.lower() not in cfg["allowed_suffixes"] and path.name not in cfg["allowed_extensionless"]:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[rel] = {"size": stat.st_size, "mtime": stat.st_mtime, "sha256": digest}
print(json.dumps({"root": str(root), "files": files}, separators=(",", ":")))
""" % repr(json.dumps(payload, separators=(",", ":")))
    command = f"python3 -c {shlex.quote(remote_code)}"
    return json.loads(_exec_remote(client, command, timeout=300).decode("utf-8"))


def _read_project_file(client, rel_path: str, expected_sha: str) -> bytes:
    result = client.get("/read", path=rel_path)
    encoded = result.get("content_base64")
    if not isinstance(encoded, str):
        raise RuntimeError(f"missing content for project file: {rel_path}")
    data = base64.b64decode(encoded, validate=True)
    actual = _sha256(data)
    if result.get("sha256") != actual or actual != expected_sha:
        raise RuntimeError(f"project file changed during mirror read: {rel_path}")
    return data


def _read_platform_file(
    client,
    platform_root: str,
    rel_path: str,
    expected_sha: str,
) -> bytes:
    target = str(Path(platform_root) / rel_path)
    remote_code = (
        "import base64, pathlib, sys; "
        f"sys.stdout.write(base64.b64encode(pathlib.Path({target!r}).read_bytes()).decode('ascii'))"
    )
    command = f"python3 -c {shlex.quote(remote_code)}"
    encoded = _exec_remote(client, command, timeout=120)
    data = base64.b64decode(encoded, validate=True)
    if _sha256(data) != expected_sha:
        raise RuntimeError(f"platform file changed during mirror read: {rel_path}")
    return data


def _load_previous_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if value.get("format") == MIRROR_FORMAT else {}


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.mirror-tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _prune_stale_files(mirror_root: Path, stale_paths: list[str]) -> int:
    removed = 0
    root = mirror_root.resolve()
    for local_rel in stale_paths:
        relative = Path(local_rel)
        if not relative.parts or relative.parts[0] == "_tools":
            continue
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise RuntimeError(f"refusing to prune path outside mirror: {local_rel}")
        if target.is_file() or target.is_symlink():
            target.unlink()
            removed += 1
    for directory in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if directory == root or "_tools" in directory.relative_to(root).parts:
            continue
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed


def _mirror_source(
    *,
    client,
    mirror_root: Path,
    source_name: str,
    remote_root: str,
    entries: dict[str, dict[str, Any]],
    reader,
    workers: int,
    previous_files: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, int]:
    records: list[dict[str, Any]] = []
    pending: list[tuple[str, dict[str, Any], Path, str]] = []
    reused = 0
    for rel_path, info in sorted(entries.items()):
        local_rel = f"{source_name}/{rel_path}"
        target = mirror_root / local_rel
        expected_sha = str(info["sha256"])
        previous = previous_files.get(local_rel, {})
        if (
            target.is_file()
            and previous.get("sha256") == expected_sha
            and _sha256(target.read_bytes()) == expected_sha
        ):
            records.append(
                {
                    "source": source_name,
                    "remote_root": remote_root,
                    "remote_path": rel_path,
                    "local_path": local_rel,
                    "size": int(info["size"]),
                    "mtime": float(info["mtime"]),
                    "sha256": expected_sha,
                    "status": "reused",
                }
            )
            reused += 1
            continue
        pending.append((rel_path, info, target, local_rel))

    downloaded = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(reader, rel_path, str(info["sha256"])): (
                rel_path,
                info,
                target,
                local_rel,
            )
            for rel_path, info, target, local_rel in pending
        }
        for future in as_completed(futures):
            rel_path, info, target, local_rel = futures[future]
            data = future.result()
            _write_file(target, data)
            records.append(
                {
                    "source": source_name,
                    "remote_root": remote_root,
                    "remote_path": rel_path,
                    "local_path": local_rel,
                    "size": len(data),
                    "mtime": float(info["mtime"]),
                    "sha256": str(info["sha256"]),
                    "status": "downloaded",
                }
            )
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"[{source_name}] downloaded {downloaded}/{len(pending)}")
    return records, downloaded, reused


def main() -> int:
    default_repo_root = Path(__file__).resolve().parents[2]
    default_mirror_root = (
        default_repo_root
        / "shared"
        / "arena_frontend_monitor"
        / "runtime"
        / "container_source_mirror"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror-root", type=Path, default=default_mirror_root)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root)
    parser.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--max-files", type=int, default=5000)
    parser.add_argument("--max-total-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prune-stale", action="store_true")
    parser.add_argument("--url", default=os.environ.get("IDE_SYNC_URL", ""))
    parser.add_argument("--token", default=None)
    parser.add_argument("--no-cookie-prompt", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("--workers must be in [1, 12]")
    if args.max_bytes <= 0:
        parser.error("--max-bytes must be positive")
    if args.max_files <= 0 or args.max_total_bytes <= 0:
        parser.error("--max-files and --max-total-bytes must be positive")

    repo_root = args.repo_root.resolve()
    server_root = repo_root / "server"
    sys.path.insert(0, str(server_root))
    from local_sync_client import (  # pylint: disable=import-error,import-outside-toplevel
        DEFAULT_COOKIE_FILE,
        DEFAULT_SYNC_URL,
        SyncClient,
        USER_PROXY_COOKIE,
        load_proxy_cookie,
        normalize_base_url,
        resolve_sync_token,
    )

    token, source = resolve_sync_token(cli_token=args.token, root=server_root)
    if not token:
        raise RuntimeError(
            "missing IDE sync token; configure server/conf/.env or IDE_SYNC_TOKEN"
        )
    cookie = load_proxy_cookie(
        explicit_cookie="",
        fallback_cookie=USER_PROXY_COOKIE,
        cookie_file=DEFAULT_COOKIE_FILE,
        cookie_name="kaiwu-token",
        no_save_cookie=False,
        no_cookie_prompt=args.no_cookie_prompt,
    )
    client = SyncClient(
        normalize_base_url(args.url or DEFAULT_SYNC_URL),
        token,
        timeout=180,
        proxy_cookie=cookie.value,
    )
    health = client.get("/health")
    mirror_root = args.mirror_root.resolve()
    mirror_root.mkdir(parents=True, exist_ok=True)
    manifest_path = mirror_root / "mirror_manifest.json"
    previous = _load_previous_manifest(manifest_path)
    previous_files = {
        item["local_path"]: item for item in previous.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("local_path"), str)
    }

    remote_source_roots = dict(DEFAULT_REMOTE_ROOTS)
    remote_manifests = {
        source_name: _platform_manifest(client, remote_root, args.max_bytes)
        for source_name, remote_root in remote_source_roots.items()
    }
    total_files = sum(
        len(manifest.get("files", {})) for manifest in remote_manifests.values()
    )
    total_bytes = sum(
        int(info.get("size", 0))
        for manifest in remote_manifests.values()
        for info in manifest.get("files", {}).values()
    )
    if total_files > args.max_files:
        raise RuntimeError(
            f"platform mirror file limit exceeded: {total_files} > {args.max_files}"
        )
    if total_bytes > args.max_total_bytes:
        raise RuntimeError(
            "platform mirror byte limit exceeded: "
            f"{total_bytes} > {args.max_total_bytes}"
        )
    external_records: list[dict[str, Any]] = []
    external_downloaded = 0
    external_reused = 0
    for source_name, remote_root in remote_source_roots.items():
        source_manifest = remote_manifests[source_name]
        source_entries = {
            path: info
            for path, info in source_manifest.get("files", {}).items()
            if _eligible(path, int(info.get("size", -1)), args.max_bytes)
        }
        source_records, source_downloaded, source_reused = _mirror_source(
            client=client,
            mirror_root=mirror_root,
            source_name=source_name,
            remote_root=str(source_manifest.get("root", remote_root)),
            entries=source_entries,
            reader=lambda path, sha, root=remote_root: _read_platform_file(
                client, root, path, sha
            ),
            workers=args.workers,
            previous_files=previous_files,
        )
        external_records.extend(source_records)
        external_downloaded += source_downloaded
        external_reused += source_reused
    records = sorted(
        external_records,
        key=lambda item: item["local_path"],
    )
    current_paths = {item["local_path"] for item in records}
    stale_paths = sorted(set(previous_files) - current_paths)
    pruned = _prune_stale_files(mirror_root, stale_paths) if args.prune_stale else 0
    manifest = {
        "format": MIRROR_FORMAT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "credential_source": source,
        "rpc_root": health.get("root"),
        "remote_roots": {
            source_name: remote_manifests[source_name].get("root", remote_root)
            for source_name, remote_root in remote_source_roots.items()
        },
        "filters": {
            "excluded_user_training_roots": [
                "/workspace/code/agent_diy",
                "/workspace/code/agent_ppo",
                "/workspace/code/conf",
                "/data/projects/legged_robot_competition_26/agent_diy",
                "/data/projects/legged_robot_competition_26/agent_ppo",
                "/data/projects/legged_robot_competition_26/conf",
            ],
            "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
            "max_bytes": args.max_bytes,
            "max_files": args.max_files,
            "max_total_bytes": args.max_total_bytes,
            "skip_dirs": sorted(SKIP_DIRS),
            "sensitive_names_excluded": True,
        },
        "stats": {
            "files": len(records),
            "bytes": sum(int(item["size"]) for item in records),
            "downloaded": external_downloaded,
            "reused": external_reused,
            "stale_not_deleted": len(stale_paths) - pruned,
            "stale_pruned": pruned,
        },
        "stale_not_deleted": [] if args.prune_stale else stale_paths,
        "files": records,
    }
    _write_file(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "mirror_root": str(mirror_root),
                "manifest": str(manifest_path),
                "stats": manifest["stats"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
