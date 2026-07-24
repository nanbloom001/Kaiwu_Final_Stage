# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Local side sync client.
放在本地项目根目录运行，将 agent_diy / agent_ppo / conf / isaac_env
同步到网页 IDE。

Run example:
运行示例：
    python local_sync_client.py
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from conf.sync_env import load_env_file, setting


SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "ckpt",
    "log",
    "logs",
    "outputs",
    "runs",
}

SKIP_SUFFIXES = {
    ".ckpt",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
}
SYNC_DIR_NAMES = ("agent_diy", "agent_ppo", "conf", "isaac_env")
GET_UPLOAD_CHUNK_SIZE = 4096
DEFAULT_SYNC_URL = "https://tencentarena.com/p5/ide/18005/proxy/8765"
DEFAULT_SYNC_TOKEN = ""
DEFAULT_COOKIE_FILE = Path.home() / ".fwwb_ide_proxy_cookie"
DEFAULT_PROXY_COOKIE_NAME = "kaiwu-token"
PROXY_COOKIE_NAME_ALIASES = ("kaiwu-token", "kaiwu_token")

# Credentials must come from CLI/environment variables, the local cache, or
# the interactive prompt. Never paste a real token or Cookie into this file.
USER_PROXY_COOKIE = ""


class TencentProxyAuthError(RuntimeError):
    """Raised when Tencent proxy rejects browser session Cookie."""


@dataclass(frozen=True)
class CookieSelection:
    """Normalized Cookie plus the source used for safe invalidation."""

    value: str
    source: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def should_skip(path: Path, root: Path, max_bytes: int) -> bool:
    rel = path.relative_to(root)
    if any(part in SKIP_DIR_NAMES for part in rel.parts):
        return True
    if path.name.endswith(".sync-tmp"):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    try:
        return path.stat().st_size > max_bytes
    except OSError:
        return True


def is_in_sync_scope(rel_path: str) -> bool:
    first_part = rel_path.replace("\\", "/").split("/", 1)[0]
    return first_part in SYNC_DIR_NAMES


def collect_local_files(root: Path, max_bytes: int) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for dir_name in SYNC_DIR_NAMES:
        sync_root = root / dir_name
        if not sync_root.exists():
            print(f"skip missing sync dir: {dir_name}", file=sys.stderr)
            continue
        for path in sync_root.rglob("*"):
            if not path.is_file() or should_skip(path, root, max_bytes):
                continue
            rel = path.relative_to(root).as_posix()
            data = path.read_bytes()
            stat = path.stat()
            files[rel] = {
                "path": path,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256": sha256_bytes(data),
                "data": data,
            }
    return files


class SyncClient:
    def __init__(self, base_url: str, token: str, timeout: int, proxy_cookie: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.proxy_cookie = proxy_cookie

    def get(self, endpoint: str, **params: str) -> dict[str, Any]:
        params["sync_token"] = self.token
        suffix = f"?{urlencode(params)}" if params else ""
        req = Request(f"{self.base_url}{endpoint}{suffix}")
        req.add_header("X-Sync-Token", self.token)
        if self.proxy_cookie:
            req.add_header("Cookie", self.proxy_cookie)
        return self._open(req)

    def post(self, endpoint: str, payload: dict[str, Any], expect_json: bool = True) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(f"{self.base_url}{endpoint}?sync_token={self.token}", data=body, method="POST")
        req.add_header("X-Sync-Token", self.token)
        req.add_header("Content-Type", "application/json; charset=utf-8")
        if self.proxy_cookie:
            req.add_header("Cookie", self.proxy_cookie)
        return self._open(req, expect_json=expect_json)

    def upload_file_get(self, rel_path: str, data: bytes, mtime: float) -> dict[str, Any]:
        self.get("/write_begin", path=rel_path)
        for offset in range(0, len(data), GET_UPLOAD_CHUNK_SIZE):
            chunk = data[offset : offset + GET_UPLOAD_CHUNK_SIZE]
            encoded = base64.urlsafe_b64encode(chunk).decode("ascii")
            self.get("/write_chunk", path=rel_path, data=encoded)
        result = self.get("/write_finish", path=rel_path, mtime=str(mtime))
        if result.get("empty_response"):
            expected_sha256 = sha256_bytes(data)
            readback = self.get("/read", path=rel_path)
            if readback.get("sha256") == expected_sha256:
                return {
                    "ok": True,
                    "sha256": expected_sha256,
                    "verified_after_empty_response": True,
                }
        return result

    def _open(self, req: Request, expect_json: bool = True) -> dict[str, Any]:
        try:
            with urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - user-provided sync URL.
                raw = resp.read()
                if not raw:
                    return {"ok": True, "empty_response": True, "status": resp.status}
                text = raw.decode("utf-8", errors="replace")
                if not expect_json:
                    return {"ok": True, "status": resp.status, "response_bytes": len(raw)}
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    content_type = resp.headers.get("Content-Type", "")
                    preview = text[:500].replace("\r", "\\r").replace("\n", "\\n")
                    raise RuntimeError(
                        "server returned non-JSON response.\n"
                        f"status={resp.status}, content_type={content_type}, preview={preview!r}"
                    ) from exc
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 401 and "TOKEN_NOT_VALID" in detail:
                raise TencentProxyAuthError(
                    "Tencent proxy rejected the request before it reached the IDE sync server.\n"
                    "腾讯代理在转发前拒绝了请求，需要浏览器登录态/cookie；这不是 IDE_SYNC_TOKEN 的问题。\n"
                    "解决办法：首次运行时直接执行 python local_sync_client.py，然后按提示粘贴 Cookie 或 kaiwu_token 值。\n"
                    "脚本会自动缓存，后续直接一键运行即可。"
                ) from exc
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                "server unreachable: "
                f"{exc}\n\n"
                "请检查：\n"
                "1. 网页 IDE 终端里是否已经启动：python3 ide_sync_server.py --root .\n"
                "2. 服务端是否显示监听 http://0.0.0.0:8765。\n"
                "3. 浏览器是否能打开固定地址：https://tencentarena.com/p5/ide/18005/proxy/8765/health。"
            ) from exc


def normalize_base_url(raw_url: str) -> str:
    url = raw_url.strip()
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    endpoint_names = {"/health", "/manifest", "/read", "/write", "/delete"}
    for endpoint in endpoint_names:
        if path.endswith(endpoint):
            path = path[: -len(endpoint)].rstrip("/")
            break
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")).rstrip("/")


def normalize_cookie_input(raw_cookie: str, cookie_name: str) -> str:
    cookie = raw_cookie.strip()
    if cookie.lower().startswith("cookie:"):
        cookie = cookie.split(":", 1)[1].strip()
    cookie = cookie.strip().strip('"').strip("'")

    # Full Cookie header / 完整 Cookie 头：原样使用。
    if ";" in cookie:
        return cookie

    # Single name=value / 单个 name=value：名字像正常 Cookie 名时原样使用。
    if "=" in cookie:
        name, value = cookie.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name in PROXY_COOKIE_NAME_ALIASES or (name and "." not in name and len(name) <= 80):
            return f"{name}={value}"

    # Bare value / 只有 value：同时带两种常见名字，兼容 kaiwu-token / kaiwu_token。
    names = tuple(dict.fromkeys((cookie_name, *PROXY_COOKIE_NAME_ALIASES)))
    return "; ".join(f"{name}={cookie}" for name in names)


def cookie_summary(cookie: str) -> str:
    parts = [part.strip() for part in cookie.split(";") if part.strip()]
    names = [part.split("=", 1)[0].strip() for part in parts if "=" in part]
    digest = hashlib.sha256(cookie.encode("utf-8")).hexdigest()[:12]
    if not names:
        return f"cookie_parts=0, length={len(cookie)}, sha256={digest}"
    preview = ", ".join(names[:5])
    if len(names) > 5:
        preview += f", ...(+{len(names) - 5})"
    return f"cookie_names=[{preview}], length={len(cookie)}, sha256={digest}"


def print_cookie_feedback(source: str, cookie: str, cookie_file: Path | None = None) -> None:
    print(f"Cookie loaded from {source}: {cookie_summary(cookie)}")
    if cookie_file is not None:
        print(f"Cookie cache file: {cookie_file}")


def load_proxy_cookie(
    explicit_cookie: str,
    fallback_cookie: str,
    cookie_file: Path,
    cookie_name: str,
    no_save_cookie: bool,
    no_cookie_prompt: bool,
    refresh_cookie: bool = False,
) -> CookieSelection:
    if refresh_cookie and no_cookie_prompt:
        raise ValueError("--refresh-cookie cannot be combined with --no-cookie-prompt")

    if not refresh_cookie and explicit_cookie.strip():
        cookie = normalize_cookie_input(explicit_cookie, cookie_name)
        print_cookie_feedback("argument/env", cookie)
        return CookieSelection(cookie, "argument/env")

    if not refresh_cookie and cookie_file.exists():
        cached_cookie = cookie_file.read_text(encoding="utf-8").strip()
        if cached_cookie:
            cookie = normalize_cookie_input(cached_cookie, cookie_name)
            print_cookie_feedback("cache", cookie, cookie_file)
            return CookieSelection(cookie, "cache")

    if not refresh_cookie and fallback_cookie.strip():
        cookie = normalize_cookie_input(fallback_cookie, cookie_name)
        print_cookie_feedback("source fallback", cookie)
        return CookieSelection(cookie, "source fallback")

    if no_cookie_prompt:
        return CookieSelection("", "none")

    print("需要腾讯网页 IDE 的代理 Cookie。")
    print("你可以粘贴完整 Cookie，也可以只粘贴 kaiwu_token 的 Cookie Value。")
    print("浏览器 DevTools -> Application/Cookies 或 Network/Request Headers 都可以复制。")
    cookie = normalize_cookie_input(getpass.getpass("Paste Cookie or kaiwu_token value here, input is hidden: "), cookie_name)
    print_cookie_feedback("prompt", cookie)
    if cookie and not no_save_cookie:
        cookie_file.parent.mkdir(parents=True, exist_ok=True)
        cookie_file.write_text(cookie, encoding="utf-8")
        print(f"Cookie saved to: {cookie_file}")
    return CookieSelection(cookie, "prompt")


def check_local_scope(root: Path, max_bytes: int) -> int:
    """Validate local sync scope without constructing a network client."""
    if not root.is_dir():
        print(f"local root is not a directory: {root}", file=sys.stderr)
        return 2
    existing_dirs = [name for name in SYNC_DIR_NAMES if (root / name).is_dir()]
    missing_dirs = [name for name in SYNC_DIR_NAMES if name not in existing_dirs]
    if not existing_dirs:
        print(
            f"no sync directories found under {root}; expected {SYNC_DIR_NAMES}",
            file=sys.stderr,
        )
        return 2
    files = collect_local_files(root, max_bytes)
    total_bytes = sum(int(item["size"]) for item in files.values())
    print(f"local root: {root}")
    print(f"sync dirs: {', '.join(SYNC_DIR_NAMES)}")
    print(f"present dirs: {', '.join(existing_dirs)}")
    print(f"missing optional dirs: {', '.join(missing_dirs) if missing_dirs else 'none'}")
    print(f"eligible files: {len(files)}")
    print(f"eligible bytes: {total_bytes}")
    for rel_path in sorted(files)[:20]:
        print(f"  local: {rel_path} sha256={files[rel_path]['sha256']}")
    return 0


def sync_files(
    client: SyncClient,
    root: Path,
    max_bytes: int,
    delete_remote: bool,
    dry_run: bool,
    skip_unchanged: bool,
) -> None:
    health = client.get("/health")
    print(f"remote root: {health.get('root')}")

    local_files = collect_local_files(root, max_bytes)
    inspect_manifest = skip_unchanged or delete_remote or dry_run
    remote_files_all = (
        client.get("/manifest").get("files", {}) if inspect_manifest else {}
    )
    remote_files = {rel: meta for rel, meta in remote_files_all.items() if is_in_sync_scope(rel)}
    local_paths = set(local_files)
    remote_paths = set(remote_files)

    if skip_unchanged or dry_run:
        changed = [
            rel
            for rel, item in local_files.items()
            if rel not in remote_files or remote_files[rel].get("sha256") != item["sha256"]
        ]
    else:
        changed = sorted(local_files)
    delete_paths = sorted(remote_paths - local_paths) if delete_remote else []

    print(f"sync dirs: {', '.join(SYNC_DIR_NAMES)}")
    print(f"local files: {len(local_files)}")
    print(f"files to overwrite: {len(changed)}")
    print(f"remote delete candidates: {len(delete_paths)}")
    if dry_run:
        for rel in changed[:20]:
            print(f"  upload: {rel}")
        for rel in delete_paths[:20]:
            print(f"  delete: {rel}")
        return

    started = time.time()
    uploaded_bytes = 0
    for index, rel in enumerate(changed, start=1):
        item = local_files[rel]
        data = item["data"]
        result = client.upload_file_get(rel, data, item["mtime"])
        if not result.get("ok") or result.get("sha256") != item["sha256"]:
            raise RuntimeError(f"failed to upload {rel}: {result}")
        if result.get("verified_after_empty_response"):
            print(f"verified by readback after empty response: {rel}")
        uploaded_bytes += len(data)
        if index == 1 or index % 20 == 0 or index == len(changed):
            print(f"uploaded {index}/{len(changed)} files, {uploaded_bytes / 1024:.1f} KiB")

    if delete_paths:
        result = client.post("/delete", {"paths": delete_paths})
        print(f"deleted remote files: {len(result.get('deleted', []))}")

    print(f"sync complete in {time.time() - started:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync this local directory to the IDE sync server.")
    parser.add_argument("--env-file", default=os.environ.get("IDE_SYNC_ENV_FILE"), help="Optional dotenv credentials file")
    parser.add_argument("--url", default=None, help="IDE forwarded URL")
    parser.add_argument("--token", default=None, help="Shared sync token")
    parser.add_argument("--proxy-cookie", default=None, help="Tencent proxy Cookie")
    parser.add_argument(
        "--proxy-cookie-name",
        default=None,
        help="Cookie name used when only the value is pasted",
    )
    parser.add_argument("--cookie-file", default=None, help="Cached Tencent proxy Cookie file")
    parser.add_argument("--no-save-cookie", action="store_true", help="Do not save pasted Cookie")
    parser.add_argument("--clear-cookie", action="store_true", help="Delete cached Cookie and exit")
    parser.add_argument("--no-cookie-prompt", action="store_true", help="Do not prompt for Cookie when missing")
    parser.add_argument(
        "--refresh-cookie",
        action="store_true",
        help="Ignore explicit/cache/source Cookie and prompt for a fresh value",
    )
    parser.add_argument("--root", default=".", help="Local project root")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--delete", action="store_true", help="Delete remote files absent locally")
    parser.add_argument("--skip-unchanged", action="store_true", help="Compare remote hashes and upload changed files only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-local",
        action="store_true",
        help="Validate local sync scope without Cookie or network access",
    )
    args = parser.parse_args()

    try:
        env_values = load_env_file(args.env_file)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sync_url = setting(args.url, "IDE_SYNC_URL", env_values, DEFAULT_SYNC_URL)
    token = setting(args.token, "IDE_SYNC_TOKEN", env_values, DEFAULT_SYNC_TOKEN)
    proxy_cookie = setting(args.proxy_cookie, "IDE_PROXY_COOKIE", env_values)
    proxy_cookie_name = setting(
        args.proxy_cookie_name,
        "IDE_PROXY_COOKIE_NAME",
        env_values,
        DEFAULT_PROXY_COOKIE_NAME,
    )
    cookie_file_raw = setting(
        args.cookie_file,
        "IDE_PROXY_COOKIE_FILE",
        env_values,
        str(DEFAULT_COOKIE_FILE),
    )

    root = Path(args.root).resolve()
    if args.check_local:
        if args.delete:
            print("--check-local cannot be combined with --delete", file=sys.stderr)
            return 2
        return check_local_scope(root, args.max_bytes)

    if not token:
        print("missing token: pass --token or set IDE_SYNC_TOKEN", file=sys.stderr)
        return 2

    cookie_file = Path(cookie_file_raw).expanduser()
    if args.clear_cookie:
        if cookie_file.exists():
            cookie_file.unlink()
            print(f"Deleted cached Cookie: {cookie_file}")
        else:
            print(f"No cached Cookie found: {cookie_file}")
        if USER_PROXY_COOKIE.strip():
            print("Note: USER_PROXY_COOKIE is set in code; --clear-cookie does not modify source code.")
        return 0

    try:
        cookie_selection = load_proxy_cookie(
            explicit_cookie=proxy_cookie,
            fallback_cookie=USER_PROXY_COOKIE,
            cookie_file=cookie_file,
            cookie_name=proxy_cookie_name,
            no_save_cookie=args.no_save_cookie,
            no_cookie_prompt=args.no_cookie_prompt,
            refresh_cookie=args.refresh_cookie,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    client = SyncClient(
        normalize_base_url(sync_url),
        token,
        args.timeout,
        cookie_selection.value,
    )
    try:
        sync_files(client, root, args.max_bytes, args.delete, args.dry_run, args.skip_unchanged)
    except TencentProxyAuthError as exc:
        print(str(exc), file=sys.stderr)
        print(
            f"Rejected Cookie source: {cookie_selection.source}",
            file=sys.stderr,
        )
        if cookie_selection.source == "cache" and cookie_file.exists():
            cookie_file.unlink()
            print(
                f"Cached Cookie was rejected and has been deleted: {cookie_file}",
                file=sys.stderr,
            )
        print(
            "请使用 --refresh-cookie 重新录入；非缓存来源不会删除现有缓存。",
            file=sys.stderr,
        )
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
