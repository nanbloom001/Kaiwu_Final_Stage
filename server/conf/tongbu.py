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
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

try:
    from conf.sync_env import load_env_file, setting
except ModuleNotFoundError:  # Supports `python conf/tongbu.py` from the IDE root.
    from sync_env import load_env_file, setting


BODY_LIMIT = int(os.environ.get("IDE_SYNC_MAX_BODY", 64 * 1024 * 1024))
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
        # Canonicalize once, then resolve every target below it. A lexical
        # relative_to() check is insufficient when an existing symlink under
        # the sync root points outside that root.
        self.base_dir = self.base_dir.expanduser().resolve(strict=False)

    def resolve(self, raw: Optional[str]) -> Path:
        cleaned = unquote(raw or "").replace("\\", "/").lstrip("/")
        destination = (self.base_dir / cleaned).resolve(strict=False)
        try:
            destination.relative_to(self.base_dir)
        except ValueError as exc:
            raise ValueError(f"path escapes root: {raw}")
        return destination


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


def _health(dispatcher: RequestDispatcher, ws: Workspace, qs: Dict[str, List[str]], body: Optional[Dict[str, Any]]) -> None:
    dispatcher._reply({"ok": True, "root": str(ws.base_dir), "time": time.time()})


def _manifest(dispatcher: RequestDispatcher, ws: Workspace, qs: Dict[str, List[str]], body: Optional[Dict[str, Any]]) -> None:
    entries: Dict[str, Dict[str, Any]] = {}
    for item in ws.base_dir.rglob("*"):
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
    dispatcher._reply({"root": str(ws.base_dir), "files": entries})


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


def _write(dispatcher: RequestDispatcher, ws: Workspace, qs: Dict[str, List[str]], body: Optional[Dict[str, Any]]) -> None:
    if body is None:
        raise ValueError("missing body")
    rel_path = body.get("path")
    if not rel_path:
        raise ValueError("missing path")
    target = ws.resolve(str(rel_path))
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
    target.parent.mkdir(parents=True, exist_ok=True)
    _staging_path(target).write_bytes(b"")
    dispatcher._reply({"ok": True, "path": target.relative_to(ws.base_dir).as_posix()})


def _write_chunk(dispatcher: RequestDispatcher, ws: Workspace, qs: Dict[str, List[str]], body: Optional[Dict[str, Any]]) -> None:
    rel_path = qs.get("path", [""])[0]
    if not rel_path:
        raise ValueError("missing path")
    target = ws.resolve(rel_path)
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
        if target.exists() and target.is_file():
            target.unlink()
            deleted.append(target.relative_to(ws.base_dir).as_posix())
    dispatcher._reply({"ok": True, "deleted": deleted})


_ROUTE_TABLE: Dict[Tuple[str, str], HandlerSig] = {
    ("GET", "/health"): _health,
    ("GET", "/manifest"): _manifest,
    ("GET", "/read"): _read,
    ("GET", "/write_begin"): _write_begin,
    ("GET", "/write_chunk"): _write_chunk,
    ("GET", "/write_finish"): _write_finish,
    ("POST", "/write"): _write,
    ("POST", "/delete"): _delete,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive code sync requests from local_sync_client.py.")
    parser.add_argument("--host", default=BIND_ADDRESS)
    parser.add_argument("--port", type=int, default=BIND_PORT)
    parser.add_argument("--root", default=os.environ.get("IDE_SYNC_ROOT", "."))
    parser.add_argument("--env-file", default=os.environ.get("IDE_SYNC_ENV_FILE"), help="Optional dotenv credentials file")
    args = parser.parse_args()

    root = Path(args.root).expanduser().absolute()
    try:
        env_values = load_env_file(args.env_file)
    except ValueError as exc:
        parser.error(str(exc))
    token = setting(None, "IDE_SYNC_TOKEN", env_values, SECRET_KEY)
    if not token and not AUTH_BYPASS:
        parser.error(
            "missing IDE_SYNC_TOKEN; set the same random value for the server "
            "and local_sync_client.py"
        )
    server = ThreadingHTTPServer((args.host, args.port), RequestDispatcher)
    server.workspace = Workspace(base_dir=root, api_key=token)  # type: ignore[attr-defined]

    print(f"IDE sync server: http://{args.host}:{args.port}")
    print(f"Root: {root}")
    print(f"Token auth: {'disabled by environment' if AUTH_BYPASS else 'enabled'}")
    print(f"Fixed outside URL: {EXTERNAL_ENDPOINT}/")
    print(f"Fixed health URL: {EXTERNAL_ENDPOINT}/health")
    server.serve_forever()


if __name__ == "__main__":
    main()
