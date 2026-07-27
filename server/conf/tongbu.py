# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
IDE side sync server.
放在网页 IDE 的项目根目录运行，用于接收本地代码同步请求。

Start example:
启动示例：
    python3 ide_sync_server.py --root .
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse


BODY_LIMIT = int(os.environ.get("IDE_SYNC_MAX_BODY", 64 * 1024 * 1024))
BUNDLE_LIMIT = int(os.environ.get("IDE_SYNC_MAX_BUNDLE", 32 * 1024 * 1024))
BUNDLE_UNPACKED_LIMIT = int(
    os.environ.get("IDE_SYNC_MAX_BUNDLE_UNPACKED", 64 * 1024 * 1024)
)
BUNDLE_FILE_LIMIT = int(os.environ.get("IDE_SYNC_MAX_BUNDLE_FILES", 4096))
EXEC_TIMEOUT_LIMIT = int(os.environ.get("IDE_SYNC_EXEC_TIMEOUT_LIMIT", 600))
EXEC_OUTPUT_LIMIT = int(os.environ.get("IDE_SYNC_EXEC_OUTPUT_LIMIT", 4 * 1024 * 1024))
SECRET_KEY = ""
BIND_ADDRESS = "0.0.0.0"
BIND_PORT = 8765
EXTERNAL_ENDPOINT = "https://tencentarena.com/p5/ide/18005/proxy/8765"
AUTH_BYPASS = os.environ.get("IDE_SYNC_DISABLE_AUTH", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
IGNORED_FOLDERS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
}
# Platform bootstrap owns this file and may replace it on startup.  Do not let
# a sync client write or delete it; other isaac_env files remain synchronizable.
PROTECTED_SYNC_PATHS = frozenset({"isaac_env/base_env.py"})


def pack_json(payload: Any, code: int = 200) -> Tuple[int, bytes]:
    return code, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def digest_file(fp: Path) -> str:
    hasher = hashlib.sha256()
    with fp.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


@dataclass
class Workspace:
    base_dir: Path
    api_key: str

    def __post_init__(self) -> None:
        # Canonicalize the configured project root once. Child paths are also
        # canonicalized below so a symlink cannot escape the workspace or
        # alias a protected platform-owned file.
        self.base_dir = self.base_dir.expanduser().resolve()

    def resolve(self, raw: Optional[str]) -> Path:
        cleaned = unquote(raw or "").replace("\\", "/").lstrip("/")
        destination = (self.base_dir / cleaned).resolve(strict=False)
        try:
            destination.relative_to(self.base_dir)
        except ValueError as exc:
            raise ValueError(f"path escapes root: {raw}")
        return destination


@dataclass
class BundleTransfer:
    path: Path
    size: int
    chunk_size: int
    received_offsets: set[int]
    lock: threading.Lock


HandlerSig = Callable[
    ["RequestDispatcher", Workspace, Dict[str, List[str]], Optional[Dict[str, Any]]],
    None,
]


class RequestDispatcher(BaseHTTPRequestHandler):
    server_version = "IdeSyncServer/1.0"

    @property
    def workspace(self) -> Workspace:
        return self.server.workspace  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, X-Sync-Token, Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, verb: str) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        try:
            self._authenticate(params)
            route = (verb, parsed.path)
            handler = _ROUTE_TABLE.get(route)
            if handler is None:
                self._reply({"error": "not found"}, 404)
                return
            body: Optional[Dict[str, Any]] = None
            if verb == "POST":
                body = self._read_body()
            handler(self, self.workspace, params, body)
        except Exception as exc:  # noqa: BLE001 - return readable remote errors.
            self._reply({"error": str(exc)}, 400)

    def _authenticate(self, params: Dict[str, List[str]]) -> None:
        if AUTH_BYPASS:
            return
        auth = self.headers.get("Authorization", "")
        bearer = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
        header_token = self.headers.get("X-Sync-Token", "")
        query_token = params.get("token", [""])[0]
        query_sync_token = params.get("sync_token", [""])[0]
        if self.workspace.api_key not in {bearer, header_token, query_token, query_sync_token}:
            raise PermissionError("unauthorized")

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > BODY_LIMIT:
            raise ValueError(f"body too large: {length} bytes")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _reply(self, payload: Any, status: int = 200) -> None:
        code, body = pack_json(payload, status)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# 路由处理器
# ---------------------------------------------------------------------------

def _staging_path(fp: Path) -> Path:
    return fp.with_name(fp.name + ".sync-tmp")


def _relative_path(ws: Workspace, target: Path) -> str:
    return os.path.normpath(target.relative_to(ws.base_dir).as_posix()).replace("\\", "/")


def _assert_writable_sync_path(ws: Workspace, target: Path) -> None:
    relative_path = _relative_path(ws, target)
    if relative_path in PROTECTED_SYNC_PATHS:
        raise PermissionError(f"platform-owned sync path is protected: {relative_path}")


def _health(dispatcher: RequestDispatcher, ws: Workspace, qs: Dict[str, List[str]], body: Optional[Dict[str, Any]]) -> None:
    dispatcher._reply(
        {
            "ok": True,
            "root": str(ws.base_dir),
            "time": time.time(),
            "capabilities": ["bundle_get_v2", "exec_b64_v1"],
        }
    )


def _manifest(dispatcher: RequestDispatcher, ws: Workspace, qs: Dict[str, List[str]], body: Optional[Dict[str, Any]]) -> None:
    raw_scope = qs.get("scope", [""])[0]
    scopes = [
        item
        for item in raw_scope.split(",")
        if item and Path(item).name == item
    ]
    scan_roots = (
        [ws.resolve(item) for item in dict.fromkeys(scopes)]
        if scopes
        else [ws.base_dir]
    )
    entries: Dict[str, Dict[str, Any]] = {}
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            continue
        for item in scan_root.rglob("*"):
            if not item.is_file():
                continue
            rel = item.relative_to(ws.base_dir).as_posix()
            if any(part in IGNORED_FOLDERS for part in Path(rel).parts):
                continue
            info = item.stat()
            entries[rel] = {
                "size": info.st_size,
                "mtime": info.st_mtime,
                "sha256": digest_file(item),
            }
    dispatcher._reply(
        {
            "root": str(ws.base_dir),
            "capabilities": ["bundle_get_v2", "exec_b64_v1"],
            "files": entries,
        }
    )


def _read(dispatcher: RequestDispatcher, ws: Workspace, qs: Dict[str, List[str]], body: Optional[Dict[str, Any]]) -> None:
    target = ws.resolve(qs.get("path", [""])[0])
    payload = target.read_bytes()
    dispatcher._reply(
        {
            "path": target.relative_to(ws.base_dir).as_posix(),
            "content_base64": base64.b64encode(payload).decode("ascii"),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )


def _decode_urlsafe_b64(value: str, *, field: str) -> bytes:
    if not value:
        raise ValueError(f"missing {field}")
    try:
        padded = value + "=" * (-len(value) % 4)
        return base64.b64decode(padded, altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"invalid base64url {field}") from exc


def _limited_output(value: bytes | None) -> Tuple[bytes, bool]:
    payload = value or b""
    if len(payload) <= EXEC_OUTPUT_LIMIT:
        return payload, False
    return payload[:EXEC_OUTPUT_LIMIT], True


def _exec_b64(
    dispatcher: RequestDispatcher,
    ws: Workspace,
    qs: Dict[str, List[str]],
    body: Optional[Dict[str, Any]],
) -> None:
    """Execute one diagnostic command inside the workspace.

    The existing IDE_SYNC_TOKEN is the only authentication layer.  The Tencent
    IDE proxy already isolates the container; this endpoint only adds cwd,
    timeout, and output bounds so a diagnostic cannot escape the project or
    wedge the sync service indefinitely.
    """

    del body
    command = _decode_urlsafe_b64(
        qs.get("cmd", [""])[0], field="cmd"
    ).decode("utf-8")
    raw_cwd = qs.get("cwd", [""])[0] or "."
    cwd = ws.resolve(raw_cwd)
    if not cwd.is_dir():
        raise NotADirectoryError(f"exec cwd is not a directory: {raw_cwd}")

    raw_timeout = qs.get("timeout", ["60"])[0]
    try:
        timeout = int(raw_timeout)
    except ValueError as exc:
        raise ValueError(f"invalid exec timeout: {raw_timeout}") from exc
    if not 1 <= timeout <= EXEC_TIMEOUT_LIMIT:
        raise ValueError(
            f"exec timeout must be in [1, {EXEC_TIMEOUT_LIMIT}]: {timeout}"
        )

    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        returncode = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout
        stderr = exc.stderr

    stdout, stdout_truncated = _limited_output(stdout)
    stderr, stderr_truncated = _limited_output(stderr)
    dispatcher._reply(
        {
            "ok": returncode == 0 and not timed_out,
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_s": time.monotonic() - started,
            "cwd": cwd.relative_to(ws.base_dir).as_posix() or ".",
            "stdout_base64": base64.b64encode(stdout).decode("ascii"),
            "stderr_base64": base64.b64encode(stderr).decode("ascii"),
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
    )


def _write(dispatcher: RequestDispatcher, ws: Workspace, qs: Dict[str, List[str]], body: Optional[Dict[str, Any]]) -> None:
    if body is None:
        raise ValueError("missing body")
    rel_path = body.get("path")
    if not rel_path:
        raise ValueError("missing path")
    target = ws.resolve(str(rel_path))
    _assert_writable_sync_path(ws, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(body.get("content_base64", ""))

    staging = _staging_path(target)
    staging.write_bytes(raw)
    os.replace(staging, target)

    mtime = body.get("mtime")
    if isinstance(mtime, (int, float)):
        os.utime(target, (mtime, mtime))
    dispatcher._reply(
        {
            "ok": True,
            "path": target.relative_to(ws.base_dir).as_posix(),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _write_begin(dispatcher: RequestDispatcher, ws: Workspace, qs: Dict[str, List[str]], body: Optional[Dict[str, Any]]) -> None:
    rel_path = qs.get("path", [""])[0]
    if not rel_path:
        raise ValueError("missing path")
    target = ws.resolve(rel_path)
    _assert_writable_sync_path(ws, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _staging_path(target).write_bytes(b"")
    dispatcher._reply({"ok": True, "path": target.relative_to(ws.base_dir).as_posix()})


def _write_chunk(dispatcher: RequestDispatcher, ws: Workspace, qs: Dict[str, List[str]], body: Optional[Dict[str, Any]]) -> None:
    rel_path = qs.get("path", [""])[0]
    if not rel_path:
        raise ValueError("missing path")
    target = ws.resolve(rel_path)
    _assert_writable_sync_path(ws, target)
    chunk_b64 = qs.get("data", [""])[0]
    decoded = base64.urlsafe_b64decode(chunk_b64.encode("ascii"))
    with _staging_path(target).open("ab") as f:
        f.write(decoded)
    dispatcher._reply({"ok": True, "bytes": len(decoded)})


def _write_finish(dispatcher: RequestDispatcher, ws: Workspace, qs: Dict[str, List[str]], body: Optional[Dict[str, Any]]) -> None:
    rel_path = qs.get("path", [""])[0]
    if not rel_path:
        raise ValueError("missing path")
    target = ws.resolve(rel_path)
    _assert_writable_sync_path(ws, target)
    staging = _staging_path(target)
    os.replace(staging, target)
    mtime_raw = qs.get("mtime", [""])[0]
    if mtime_raw:
        mtime = float(mtime_raw)
        os.utime(target, (mtime, mtime))
    payload = target.read_bytes()
    dispatcher._reply(
        {
            "ok": True,
            "path": target.relative_to(ws.base_dir).as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )


def _delete(dispatcher: RequestDispatcher, ws: Workspace, qs: Dict[str, List[str]], body: Optional[Dict[str, Any]]) -> None:
    if body is None:
        raise ValueError("missing body")
    deleted: List[str] = []
    for rel_path in body.get("paths", []):
        target = ws.resolve(str(rel_path))
        _assert_writable_sync_path(ws, target)
        if target.exists() and target.is_file():
            target.unlink()
            deleted.append(target.relative_to(ws.base_dir).as_posix())
    dispatcher._reply({"ok": True, "deleted": deleted})


def _bundle_id(raw: str) -> str:
    bundle_id = raw.strip()
    if len(bundle_id) != 32 or any(
        char not in "0123456789abcdef" for char in bundle_id
    ):
        raise ValueError("invalid bundle_id")
    return bundle_id


def _bundle_path(ws: Workspace, bundle_id: str) -> Path:
    return ws.resolve(f".ide-sync-{_bundle_id(bundle_id)}.tar.gz")


def _bundle_transfer(
    dispatcher: RequestDispatcher, bundle_id: str
) -> BundleTransfer:
    with dispatcher.server.bundle_transfers_lock:  # type: ignore[attr-defined]
        transfer = dispatcher.server.bundle_transfers.get(bundle_id)  # type: ignore[attr-defined]
    if transfer is None:
        raise ValueError("bundle not initialized")
    return transfer


def _remove_bundle_transfer(
    dispatcher: RequestDispatcher, bundle_id: str
) -> Optional[BundleTransfer]:
    with dispatcher.server.bundle_transfers_lock:  # type: ignore[attr-defined]
        return dispatcher.server.bundle_transfers.pop(bundle_id, None)  # type: ignore[attr-defined]


def _bundle_begin(
    dispatcher: RequestDispatcher,
    ws: Workspace,
    qs: Dict[str, List[str]],
    body: Optional[Dict[str, Any]],
) -> None:
    del body
    bundle_id = _bundle_id(qs.get("bundle_id", [""])[0])
    size = int(qs.get("size", ["0"])[0])
    chunk_size = int(qs.get("chunk_size", ["0"])[0])
    if not 0 < size <= BUNDLE_LIMIT:
        raise ValueError("invalid bundle size")
    if not 1 <= chunk_size <= 4096:
        raise ValueError("invalid bundle chunk size")
    old_transfer = _remove_bundle_transfer(dispatcher, bundle_id)
    if old_transfer is not None:
        old_transfer.path.unlink(missing_ok=True)
    target = _bundle_path(ws, bundle_id)
    with target.open("wb") as stream:
        stream.truncate(size)
    transfer = BundleTransfer(
        path=target,
        size=size,
        chunk_size=chunk_size,
        received_offsets=set(),
        lock=threading.Lock(),
    )
    with dispatcher.server.bundle_transfers_lock:  # type: ignore[attr-defined]
        dispatcher.server.bundle_transfers[bundle_id] = transfer  # type: ignore[attr-defined]
    dispatcher._reply({"ok": True})


def _bundle_chunk(
    dispatcher: RequestDispatcher,
    ws: Workspace,
    qs: Dict[str, List[str]],
    body: Optional[Dict[str, Any]],
) -> None:
    del ws, body
    bundle_id = _bundle_id(qs.get("bundle_id", [""])[0])
    transfer = _bundle_transfer(dispatcher, bundle_id)
    offset = int(qs.get("offset", ["-1"])[0])
    chunk = base64.urlsafe_b64decode(qs.get("data", [""])[0].encode("ascii"))
    expected_size = min(transfer.chunk_size, transfer.size - offset)
    if (
        offset < 0
        or offset % transfer.chunk_size != 0
        or len(chunk) != expected_size
    ):
        raise ValueError("invalid bundle chunk range")
    with transfer.lock:
        with transfer.path.open("r+b") as stream:
            stream.seek(offset)
            stream.write(chunk)
        transfer.received_offsets.add(offset)
    dispatcher._reply({"ok": True, "bytes": len(chunk)})


def _bundle_finish(
    dispatcher: RequestDispatcher,
    ws: Workspace,
    qs: Dict[str, List[str]],
    body: Optional[Dict[str, Any]],
) -> None:
    del body
    bundle_id = _bundle_id(qs.get("bundle_id", [""])[0])
    transfer = _bundle_transfer(dispatcher, bundle_id)
    expected_offsets = set(range(0, transfer.size, transfer.chunk_size))
    with transfer.lock:
        if transfer.received_offsets != expected_offsets:
            raise ValueError("bundle is missing chunks")
    bundle_path = transfer.path
    try:
        pending: List[Tuple[Path, bytes, int]] = []
        seen: set[str] = set()
        unpacked_bytes = 0
        with tarfile.open(bundle_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    raise ValueError(
                        f"bundle member is not a regular file: {member.name}"
                    )
                if member.name in seen:
                    raise ValueError(f"duplicate bundle member: {member.name}")
                seen.add(member.name)
                if len(seen) > BUNDLE_FILE_LIMIT:
                    raise ValueError("bundle has too many files")
                target = ws.resolve(member.name)
                _assert_writable_sync_path(ws, target)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read bundle member: {member.name}")
                data = source.read()
                unpacked_bytes += len(data)
                if unpacked_bytes > BUNDLE_UNPACKED_LIMIT:
                    raise ValueError("bundle unpacked data exceeds maximum size")
                pending.append((target, data, int(member.mtime)))

        written: Dict[str, Dict[str, str]] = {}
        for target, data, mtime in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = _staging_path(target)
            staging.write_bytes(data)
            os.replace(staging, target)
            os.utime(target, (mtime, mtime))
            written[target.relative_to(ws.base_dir).as_posix()] = {
                "sha256": hashlib.sha256(data).hexdigest()
            }
        dispatcher._reply({"ok": True, "files": written})
    finally:
        bundle_path.unlink(missing_ok=True)
        _remove_bundle_transfer(dispatcher, bundle_id)


def _bundle_abort(
    dispatcher: RequestDispatcher,
    ws: Workspace,
    qs: Dict[str, List[str]],
    body: Optional[Dict[str, Any]],
) -> None:
    del body
    bundle_id = _bundle_id(qs.get("bundle_id", [""])[0])
    transfer = _remove_bundle_transfer(dispatcher, bundle_id)
    target = transfer.path if transfer is not None else _bundle_path(ws, bundle_id)
    target.unlink(missing_ok=True)
    dispatcher._reply({"ok": True})


def _delete_one(
    dispatcher: RequestDispatcher,
    ws: Workspace,
    qs: Dict[str, List[str]],
    body: Optional[Dict[str, Any]],
) -> None:
    del body
    target = ws.resolve(qs.get("path", [""])[0])
    _assert_writable_sync_path(ws, target)
    if target.is_file():
        target.unlink()
    dispatcher._reply({"ok": True})


_ROUTE_TABLE: Dict[Tuple[str, str], HandlerSig] = {
    ("GET", "/health"): _health,
    ("GET", "/manifest"): _manifest,
    ("GET", "/read"): _read,
    ("GET", "/exec_b64"): _exec_b64,
    ("GET", "/write_begin"): _write_begin,
    ("GET", "/write_chunk"): _write_chunk,
    ("GET", "/write_finish"): _write_finish,
    ("GET", "/bundle_begin"): _bundle_begin,
    ("GET", "/bundle_chunk"): _bundle_chunk,
    ("GET", "/bundle_finish"): _bundle_finish,
    ("GET", "/bundle_abort"): _bundle_abort,
    ("GET", "/delete_one"): _delete_one,
    ("POST", "/write"): _write,
    ("POST", "/delete"): _delete,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive code sync requests from local_sync_client.py.")
    parser.add_argument("--host", default=BIND_ADDRESS)
    parser.add_argument("--port", type=int, default=BIND_PORT)
    parser.add_argument("--root", default=os.environ.get("IDE_SYNC_ROOT", "."))
    args = parser.parse_args()

    root = Path(args.root).expanduser().absolute()
    token = os.environ.get("IDE_SYNC_TOKEN") or SECRET_KEY
    if not token and not AUTH_BYPASS:
        parser.error(
            "missing IDE_SYNC_TOKEN; set the same random value for the server "
            "and local_sync_client.py"
        )
    server = ThreadingHTTPServer((args.host, args.port), RequestDispatcher)
    server.workspace = Workspace(base_dir=root, api_key=token)  # type: ignore[attr-defined]
    server.bundle_transfers = {}  # type: ignore[attr-defined]
    server.bundle_transfers_lock = threading.Lock()  # type: ignore[attr-defined]

    print(f"IDE sync server: http://{args.host}:{args.port}")
    print(f"Root: {root}")
    print(f"Token auth: {'disabled by environment' if AUTH_BYPASS else 'enabled'}")
    print(f"Fixed outside URL: {EXTERNAL_ENDPOINT}/")
    print(f"Fixed health URL: {EXTERNAL_ENDPOINT}/health")
    server.serve_forever()


if __name__ == "__main__":
    main()
