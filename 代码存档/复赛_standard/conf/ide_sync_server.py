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
import secrets
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


MAX_BODY_BYTES = int(os.environ.get("IDE_SYNC_MAX_BODY", 64 * 1024 * 1024))
DEFAULT_SYNC_TOKEN = "fwwb-new-codex-sync-20260512-5a7f58a98ddf4e9c9b4c9e1b6a2d8f41"
FIXED_HOST = "0.0.0.0"
FIXED_PORT = 8765
FIXED_PUBLIC_URL = "https://tencentarena.com/p5/ide/11428/proxy/8765"
DISABLE_SYNC_TOKEN_AUTH = True
SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
}


def json_response(data: Any, status: int = 200) -> tuple[int, bytes]:
    return status, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SyncState:
    def __init__(self, root: Path, token: str) -> None:
        # Keep root lexical/absolute instead of resolving every child symlink.
        # 保持 root 为文本绝对路径，不解析子路径符号链接，兼容 IDE 中的挂载目录。
        self.root = root.expanduser().absolute()
        self.token = token

    def safe_path(self, raw_path: str | None) -> Path:
        rel = unquote(raw_path or "").replace("\\", "/").lstrip("/")
        target = (self.root / rel).absolute()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"path escapes root: {raw_path}")
        return target


class SyncHandler(BaseHTTPRequestHandler):
    server_version = "IdeSyncServer/1.0"

    @property
    def state(self) -> SyncState:
        return self.server.state  # type: ignore[attr-defined]

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
        self.dispatch("GET")

    def do_POST(self) -> None:
        self.dispatch("POST")

    def dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            self.require_auth(query)
            if method == "GET" and parsed.path == "/health":
                self.send_json({"ok": True, "root": str(self.state.root), "time": time.time()})
            elif method == "GET" and parsed.path == "/manifest":
                self.handle_manifest()
            elif method == "GET" and parsed.path == "/read":
                self.handle_read(query)
            elif method == "GET" and parsed.path == "/write_begin":
                self.handle_write_begin(query)
            elif method == "GET" and parsed.path == "/write_chunk":
                self.handle_write_chunk(query)
            elif method == "GET" and parsed.path == "/write_finish":
                self.handle_write_finish(query)
            elif method == "POST" and parsed.path == "/write":
                self.handle_write()
            elif method == "POST" and parsed.path == "/delete":
                self.handle_delete()
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001 - return readable remote errors.
            self.send_json({"error": str(exc)}, 400)

    def require_auth(self, query: dict[str, list[str]]) -> None:
        if DISABLE_SYNC_TOKEN_AUTH:
            return
        auth = self.headers.get("Authorization", "")
        bearer = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
        header_token = self.headers.get("X-Sync-Token", "")
        query_token = query.get("token", [""])[0]
        query_sync_token = query.get("sync_token", [""])[0]
        if self.state.token not in {bearer, header_token, query_token, query_sync_token}:
            raise PermissionError("unauthorized")

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError(f"body too large: {length} bytes")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, data: Any, status: int = 200) -> None:
        code, body = json_response(data, status)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_manifest(self) -> None:
        files: dict[str, dict[str, Any]] = {}
        for path in self.state.root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.state.root).as_posix()
            if any(part in SKIP_DIR_NAMES for part in Path(rel).parts):
                continue
            stat = path.stat()
            files[rel] = {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256": file_sha256(path),
            }
        self.send_json({"root": str(self.state.root), "files": files})

    def handle_read(self, query: dict[str, list[str]]) -> None:
        path = self.state.safe_path(query.get("path", [""])[0])
        data = path.read_bytes()
        self.send_json(
            {
                "path": path.relative_to(self.state.root).as_posix(),
                "content_base64": base64.b64encode(data).decode("ascii"),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    def handle_write(self) -> None:
        body = self.read_json_body()
        rel_path = body.get("path")
        if not rel_path:
            raise ValueError("missing path")
        path = self.state.safe_path(str(rel_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(body.get("content_base64", ""))

        tmp_path = path.with_name(path.name + ".sync-tmp")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)

        mtime = body.get("mtime")
        if isinstance(mtime, (int, float)):
            os.utime(path, (mtime, mtime))
        self.send_json(
            {
                "ok": True,
                "path": path.relative_to(self.state.root).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    def tmp_path_for(self, path: Path) -> Path:
        return path.with_name(path.name + ".sync-tmp")

    def handle_write_begin(self, query: dict[str, list[str]]) -> None:
        rel_path = query.get("path", [""])[0]
        if not rel_path:
            raise ValueError("missing path")
        path = self.state.safe_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_path_for(path).write_bytes(b"")
        self.send_json({"ok": True, "path": path.relative_to(self.state.root).as_posix()})

    def handle_write_chunk(self, query: dict[str, list[str]]) -> None:
        rel_path = query.get("path", [""])[0]
        if not rel_path:
            raise ValueError("missing path")
        path = self.state.safe_path(rel_path)
        chunk_b64 = query.get("data", [""])[0]
        data = base64.urlsafe_b64decode(chunk_b64.encode("ascii"))
        with self.tmp_path_for(path).open("ab") as f:
            f.write(data)
        self.send_json({"ok": True, "bytes": len(data)})

    def handle_write_finish(self, query: dict[str, list[str]]) -> None:
        rel_path = query.get("path", [""])[0]
        if not rel_path:
            raise ValueError("missing path")
        path = self.state.safe_path(rel_path)
        tmp_path = self.tmp_path_for(path)
        os.replace(tmp_path, path)
        mtime_raw = query.get("mtime", [""])[0]
        if mtime_raw:
            mtime = float(mtime_raw)
            os.utime(path, (mtime, mtime))
        data = path.read_bytes()
        self.send_json(
            {
                "ok": True,
                "path": path.relative_to(self.state.root).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    def handle_delete(self) -> None:
        body = self.read_json_body()
        deleted: list[str] = []
        for rel_path in body.get("paths", []):
            path = self.state.safe_path(str(rel_path))
            if path.exists() and path.is_file():
                path.unlink()
                deleted.append(path.relative_to(self.state.root).as_posix())
        self.send_json({"ok": True, "deleted": deleted})


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive code sync requests from local_sync_client.py.")
    parser.add_argument("--host", default=FIXED_HOST)
    parser.add_argument("--port", type=int, default=FIXED_PORT)
    parser.add_argument("--root", default=os.environ.get("IDE_SYNC_ROOT", "."))
    args = parser.parse_args()

    root = Path(args.root).expanduser().absolute()
    token = os.environ.get("IDE_SYNC_TOKEN") or DEFAULT_SYNC_TOKEN
    server = ThreadingHTTPServer((args.host, args.port), SyncHandler)
    server.state = SyncState(root=root, token=token)  # type: ignore[attr-defined]

    print(f"IDE sync server: http://{args.host}:{args.port}")
    print(f"Root: {root}")
    print(f"Token: {token}")
    print(f"Fixed outside URL: {FIXED_PUBLIC_URL}/")
    print(f"Fixed health URL: {FIXED_PUBLIC_URL}/health?sync_token={token}")
    server.serve_forever()


if __name__ == "__main__":
    main()
