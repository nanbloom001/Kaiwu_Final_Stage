#!/usr/bin/env python3
"""Upload a large model to the Tencent Kaiwu container with resumable parts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from local_sync_client import (
    DEFAULT_COOKIE_FILE,
    DEFAULT_PROXY_COOKIE_NAME,
    DEFAULT_SYNC_URL,
    SyncClient,
    TencentProxyAuthError,
    USER_PROXY_COOKIE,
    build_bundle,
    load_proxy_cookie,
    normalize_base_url,
    resolve_sync_token,
)


DEFAULT_REMOTE_PREFIX = PurePosixPath("agent_ppo/test_artifacts")
DEFAULT_PART_BYTES = 1 * 1024 * 1024
DEFAULT_PART_WORKERS = 2
DEFAULT_CHUNK_WORKERS = 2
DEFAULT_RETRIES = 3


@dataclass(frozen=True)
class FilePart:
    index: int
    remote_path: str
    data: bytes
    sha256: str
    mtime: float


def validate_remote_path(raw: str) -> str:
    normalized = PurePosixPath(raw.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError("remote path must be relative and cannot contain '..'")
    if len(normalized.parts) <= len(DEFAULT_REMOTE_PREFIX.parts):
        raise ValueError(
            f"remote path must be below {DEFAULT_REMOTE_PREFIX.as_posix()}/"
        )
    if normalized.parts[: len(DEFAULT_REMOTE_PREFIX.parts)] != (
        DEFAULT_REMOTE_PREFIX.parts
    ):
        raise ValueError(
            f"remote path must be below {DEFAULT_REMOTE_PREFIX.as_posix()}/"
        )
    return normalized.as_posix()


def manifest_file_entry(
    manifest: dict[str, Any], remote_path: str
) -> dict[str, Any]:
    """Read a file entry from either rooted or scope-relative manifests."""
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        return {}
    entry = files.get(remote_path)
    if isinstance(entry, dict):
        return entry
    scope_prefix = "agent_ppo/"
    if remote_path.startswith(scope_prefix):
        entry = files.get(remote_path[len(scope_prefix) :])
        if isinstance(entry, dict):
            return entry
    return {}


def split_file(
    source: Path,
    remote_path: str,
    part_bytes: int,
) -> tuple[list[FilePart], str, int]:
    if part_bytes < 64 * 1024:
        raise ValueError("part size must be at least 64 KiB")
    stat = source.stat()
    whole_hash = hashlib.sha256()
    parts: list[FilePart] = []
    lineage = hashlib.sha256(
        f"{source.name}:{stat.st_size}".encode("utf-8")
    ).hexdigest()[:12]
    part_root = f"{remote_path}.parts/{lineage}"
    with source.open("rb") as stream:
        index = 0
        while True:
            data = stream.read(part_bytes)
            if not data:
                break
            whole_hash.update(data)
            parts.append(
                FilePart(
                    index=index,
                    remote_path=f"{part_root}/part-{index:05d}",
                    data=data,
                    sha256=hashlib.sha256(data).hexdigest(),
                    mtime=stat.st_mtime,
                )
            )
            index += 1
    if not parts:
        raise ValueError("source file is empty")
    return parts, whole_hash.hexdigest(), stat.st_size


def build_part_bundle(part: FilePart) -> bytes:
    item = {
        "size": len(part.data),
        "mtime": part.mtime,
        "sha256": part.sha256,
        "data": part.data,
    }
    return build_bundle({part.remote_path: item}, [part.remote_path])


def upload_part(
    client: SyncClient,
    part: FilePart,
    chunk_workers: int,
    retries: int,
) -> None:
    bundle = build_part_bundle(part)
    failure: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = client.upload_bundle_get(bundle, workers=chunk_workers)
            remote = result.get("files", {}).get(part.remote_path, {})
            if remote.get("sha256") != part.sha256:
                raise RuntimeError(
                    f"remote hash mismatch for part {part.index}: {result}"
                )
            return
        except (RuntimeError, TencentProxyAuthError) as exc:
            failure = exc
            if attempt == retries:
                break
            time.sleep(min(2.0, 0.25 * (2**attempt)))
    assert failure is not None
    raise failure


def build_merge_command(
    remote_path: str,
    parts: list[FilePart],
    expected_size: int,
    expected_sha256: str,
    keep_parts: bool,
) -> str:
    script = """\
import hashlib
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_size = int(sys.argv[2])
expected_sha256 = sys.argv[3]
keep_parts = sys.argv[4] == "1"
parts = [Path(value) for value in sys.argv[5:]]
output.parent.mkdir(parents=True, exist_ok=True)
staging = output.with_name(output.name + ".uploading")
digest = hashlib.sha256()
written = 0
with staging.open("wb") as destination:
    for part in parts:
        with part.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                destination.write(chunk)
                digest.update(chunk)
                written += len(chunk)
    destination.flush()
    os.fsync(destination.fileno())
actual_sha256 = digest.hexdigest()
if written != expected_size or actual_sha256 != expected_sha256:
    staging.unlink(missing_ok=True)
    raise SystemExit(
        f"merge verification failed: size={written}/{expected_size} "
        f"sha256={actual_sha256}/{expected_sha256}"
    )
os.replace(staging, output)
if not keep_parts:
    for part in parts:
        part.unlink(missing_ok=True)
    for directory in sorted({part.parent for part in parts}, reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
print(f"merged={output} bytes={written} sha256={actual_sha256}")
"""
    argv = [
        "python3",
        "-c",
        script,
        remote_path,
        str(expected_size),
        expected_sha256,
        "1" if keep_parts else "0",
        *(part.remote_path for part in parts),
    ]
    return " ".join(shlex.quote(value) for value in argv)


def run_remote_command(
    client: SyncClient,
    command: str,
    timeout: int,
) -> dict[str, Any]:
    encoded = base64.urlsafe_b64encode(command.encode("utf-8")).decode("ascii")
    result = client.get(
        "/exec_b64",
        cmd=encoded,
        cwd=".",
        timeout=str(timeout),
    )
    if int(result.get("returncode", 1)) != 0:
        stderr_raw = result.get("stderr_base64", "")
        stderr = (
            base64.b64decode(stderr_raw).decode("utf-8", errors="replace")
            if isinstance(stderr_raw, str) and stderr_raw
            else ""
        )
        raise RuntimeError(f"container merge failed: {stderr.strip()}")
    return result


def upload_model(
    client: SyncClient,
    source: Path,
    remote_path: str,
    part_bytes: int,
    part_workers: int,
    chunk_workers: int,
    retries: int,
    keep_parts: bool,
    dry_run: bool,
    merge_timeout: int,
) -> tuple[str, int]:
    parts, whole_sha256, whole_size = split_file(
        source, remote_path, part_bytes
    )
    manifest = client.get("/manifest", scope="agent_ppo")
    if manifest_file_entry(manifest, remote_path).get("sha256") == whole_sha256:
        print(
            f"remote file already verified: {remote_path} "
            f"bytes={whole_size} sha256={whole_sha256}"
        )
        return whole_sha256, whole_size
    pending = [
        part
        for part in parts
        if manifest_file_entry(manifest, part.remote_path).get("sha256")
        != part.sha256
    ]
    print(
        f"source={source} bytes={whole_size} parts={len(parts)} "
        f"pending={len(pending)} sha256={whole_sha256}"
    )
    print(
        f"concurrency=part_workers:{part_workers} x "
        f"chunk_workers:{chunk_workers} (max requests {part_workers * chunk_workers})"
    )
    if dry_run:
        return whole_sha256, whole_size
    if pending:
        completed = len(parts) - len(pending)
        with ThreadPoolExecutor(
            max_workers=min(part_workers, len(pending)),
            thread_name_prefix="model-part",
        ) as executor:
            futures = {
                executor.submit(
                    upload_part,
                    client,
                    part,
                    chunk_workers,
                    retries,
                ): part
                for part in pending
            }
            for future in as_completed(futures):
                part = futures[future]
                future.result()
                completed += 1
                print(f"verified part {completed}/{len(parts)}: {part.index}")
    command = build_merge_command(
        remote_path,
        parts,
        whole_size,
        whole_sha256,
        keep_parts,
    )
    run_remote_command(client, command, merge_timeout)
    final_manifest = client.get("/manifest", scope="agent_ppo")
    remote_sha256 = manifest_file_entry(final_manifest, remote_path).get(
        "sha256"
    )
    if remote_sha256 != whole_sha256:
        raise RuntimeError(
            f"final manifest hash mismatch: {remote_sha256} != {whole_sha256}"
        )
    print(
        f"upload complete: {remote_path} bytes={whole_size} "
        f"sha256={whole_sha256}"
    )
    return whole_sha256, whole_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a large model through resumable verified bundles and "
            "reassemble it atomically in the Kaiwu development container."
        )
    )
    parser.add_argument("source", help="Local model/checkpoint file")
    parser.add_argument(
        "--remote-path",
        required=True,
        help="Target below agent_ppo/test_artifacts/ in the container project",
    )
    parser.add_argument("--part-bytes", type=int, default=DEFAULT_PART_BYTES)
    parser.add_argument(
        "--part-workers", type=int, default=DEFAULT_PART_WORKERS
    )
    parser.add_argument(
        "--chunk-workers", type=int, default=DEFAULT_CHUNK_WORKERS
    )
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--keep-parts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--merge-timeout", type=int, default=120)
    parser.add_argument("--url", default=os.environ.get("IDE_SYNC_URL", DEFAULT_SYNC_URL))
    parser.add_argument("--token", default=None)
    parser.add_argument("--root", default=".", help="Local server project root")
    parser.add_argument(
        "--proxy-cookie", default=os.environ.get("IDE_PROXY_COOKIE", "")
    )
    parser.add_argument(
        "--proxy-cookie-name",
        default=os.environ.get("IDE_PROXY_COOKIE_NAME", DEFAULT_PROXY_COOKIE_NAME),
    )
    parser.add_argument("--cookie-file", default=str(DEFAULT_COOKIE_FILE))
    parser.add_argument("--no-cookie-prompt", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.part_workers <= 8:
        parser.error("--part-workers must be in [1, 8]")
    if not 1 <= args.chunk_workers <= 8:
        parser.error("--chunk-workers must be in [1, 8]")
    if args.part_workers * args.chunk_workers > 16:
        parser.error("part_workers * chunk_workers must not exceed 16")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        parser.error(f"source file does not exist: {source}")
    try:
        remote_path = validate_remote_path(args.remote_path)
    except ValueError as exc:
        parser.error(str(exc))

    root = Path(args.root).resolve()
    token, token_source = resolve_sync_token(cli_token=args.token, root=root)
    if not token:
        print("missing IDE_SYNC_TOKEN", file=sys.stderr)
        return 2
    print(f"token source: {token_source}")
    cookie = load_proxy_cookie(
        explicit_cookie=args.proxy_cookie,
        fallback_cookie=USER_PROXY_COOKIE,
        cookie_file=Path(args.cookie_file).expanduser(),
        cookie_name=args.proxy_cookie_name,
        no_save_cookie=False,
        no_cookie_prompt=args.no_cookie_prompt,
    )
    client = SyncClient(
        normalize_base_url(args.url),
        token,
        timeout=args.timeout,
        proxy_cookie=cookie.value,
    )
    try:
        upload_model(
            client,
            source,
            remote_path,
            args.part_bytes,
            args.part_workers,
            args.chunk_workers,
            args.retries,
            args.keep_parts,
            args.dry_run,
            args.merge_timeout,
        )
    except (RuntimeError, TencentProxyAuthError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
