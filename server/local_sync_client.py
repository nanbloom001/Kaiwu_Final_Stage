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
import io
import json
import os
import posixpath
import sys
import tarfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


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
# The platform recreates this implementation during container start-up.  It
# must never be overwritten or removed by a repository sync, while the rest
# of isaac_env remains in scope.
PROTECTED_SYNC_PATHS = frozenset({"isaac_env/base_env.py"})
GET_UPLOAD_CHUNK_SIZE = 4096
DEFAULT_UPLOAD_WORKERS = 4
MAX_UPLOAD_WORKERS = 16
DEFAULT_BUNDLE_WORKERS = 8
DEFAULT_UPLOAD_RETRIES = 2
DEFAULT_SYNC_TRANSPORT = "auto"
BUNDLE_CAPABILITY = "bundle_get_v2"
DEFAULT_SYNC_URL = "https://tencentarena.com/p5/ide/18005/proxy/8765"
DEFAULT_SYNC_TOKEN = ""
SYNC_TOKEN_ENV_NAME = "IDE_SYNC_TOKEN"
SYNC_TOKEN_ENV_RELATIVE_PATH = Path("conf") / ".env"
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


def load_dotenv_value(path: Path, key: str) -> str:
    """Return one dotenv value without evaluating shell syntax.

    The sync token is a local credential, so the client accepts only a simple
    ``KEY=value`` (or ``export KEY=value``) entry.  This deliberately avoids
    ``source``/shell execution and makes an arbitrary local .env inert.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        if separator != "=" or name.strip() != key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return ""


def resolve_sync_token(
    *,
    cli_token: str | None,
    root: Path,
    environ: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve the token with CLI/environment taking precedence over .env."""
    if cli_token and cli_token.strip():
        return cli_token.strip(), "--token"
    values = os.environ if environ is None else environ
    environment_token = values.get(SYNC_TOKEN_ENV_NAME, "").strip()
    if environment_token:
        return environment_token, SYNC_TOKEN_ENV_NAME
    env_path = root / SYNC_TOKEN_ENV_RELATIVE_PATH
    dotenv_token = load_dotenv_value(env_path, SYNC_TOKEN_ENV_NAME).strip()
    if dotenv_token:
        return dotenv_token, str(env_path)
    return DEFAULT_SYNC_TOKEN, ""


def should_skip(path: Path, root: Path, max_bytes: int) -> bool:
    rel = path.relative_to(root)
    if is_protected_sync_path(rel.as_posix()):
        return True
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


def is_protected_sync_path(rel_path: str) -> bool:
    """Return whether a normalized relative path is platform-owned."""
    normalized = posixpath.normpath(rel_path.replace("\\", "/").lstrip("/"))
    return normalized in PROTECTED_SYNC_PATHS


def is_in_sync_scope(rel_path: str) -> bool:
    normalized = posixpath.normpath(rel_path.replace("\\", "/").lstrip("/"))
    first_part = normalized.split("/", 1)[0]
    return first_part in SYNC_DIR_NAMES and not is_protected_sync_path(normalized)


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

    def upload_bundle_get(
        self,
        bundle: bytes,
        workers: int = DEFAULT_BUNDLE_WORKERS,
    ) -> dict[str, Any]:
        """Send a compressed source bundle with URL-safe GET chunks only.

        Tencent's IDE proxy reliably forwards query parameters but may drop
        POST bodies. The bundle turns a multi-file update into a small number
        of independently retriable GET chunk requests.
        """
        bundle_id = uuid.uuid4().hex
        chunk_count = (len(bundle) + GET_UPLOAD_CHUNK_SIZE - 1) // GET_UPLOAD_CHUNK_SIZE
        try:
            self.get(
                "/bundle_begin",
                bundle_id=bundle_id,
                size=str(len(bundle)),
                chunk_size=str(GET_UPLOAD_CHUNK_SIZE),
            )
            with ThreadPoolExecutor(
                max_workers=min(workers, chunk_count)
            ) as executor:
                futures = [
                    executor.submit(
                        self.get,
                        "/bundle_chunk",
                        bundle_id=bundle_id,
                        offset=str(offset),
                        data=base64.urlsafe_b64encode(
                            bundle[offset : offset + GET_UPLOAD_CHUNK_SIZE]
                        ).decode("ascii"),
                    )
                    for offset in range(0, len(bundle), GET_UPLOAD_CHUNK_SIZE)
                ]
                for future in as_completed(futures):
                    future.result()
            return self.get("/bundle_finish", bundle_id=bundle_id)
        except Exception:
            try:
                self.get("/bundle_abort", bundle_id=bundle_id)
            except RuntimeError:
                pass
            raise

    def delete_file_get(self, rel_path: str) -> None:
        self.get("/delete_one", path=rel_path)

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
    endpoint_names = {
        "/health",
        "/manifest",
        "/read",
        "/write",
        "/delete",
        "/bundle_begin",
        "/bundle_chunk",
        "/bundle_finish",
    }
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


def build_bundle(local_files: dict[str, dict[str, Any]], paths: list[str]) -> bytes:
    """Create a deterministic gzip tar bundle of the changed regular files."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for rel_path in paths:
            item = local_files[rel_path]
            info = tarfile.TarInfo(name=rel_path)
            info.size = int(item["size"])
            info.mtime = int(item["mtime"])
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(item["data"]))
    return buffer.getvalue()


def verify_remote_hashes(
    remote_files: dict[str, Any],
    local_files: dict[str, dict[str, Any]],
    paths: list[str],
) -> None:
    mismatched = [
        rel_path
        for rel_path in paths
        if remote_files.get(rel_path, {}).get("sha256")
        != local_files[rel_path]["sha256"]
    ]
    if mismatched:
        examples = ", ".join(mismatched[:5])
        raise RuntimeError(
            f"bundle verification failed for {len(mismatched)} files; "
            f"examples: {examples}"
        )


def upload_with_retry(
    client: SyncClient,
    rel_path: str,
    item: dict[str, Any],
    retries: int,
) -> dict[str, Any]:
    """Upload and verify one independent GET chunk transfer."""
    expected_sha256 = item["sha256"]
    failure: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = client.upload_file_get(
                rel_path, item["data"], item["mtime"]
            )
            if result.get("ok") and result.get("sha256") == expected_sha256:
                return result
            raise RuntimeError(f"failed to upload {rel_path}: {result}")
        except RuntimeError as exc:
            failure = exc
            if attempt == retries:
                break
            # /write_finish replaces a completed staging file atomically, so
            # retrying the same path does not corrupt a successful write.
            time.sleep(0.25 * (attempt + 1))
    assert failure is not None
    raise failure


def sync_files(
    client: SyncClient,
    root: Path,
    max_bytes: int,
    delete_remote: bool,
    dry_run: bool,
    skip_unchanged: bool,
    upload_workers: int = DEFAULT_UPLOAD_WORKERS,
    transport: str = DEFAULT_SYNC_TRANSPORT,
    upload_retries: int = DEFAULT_UPLOAD_RETRIES,
    only_paths: tuple[str, ...] = (),
) -> None:
    local_files = collect_local_files(root, max_bytes)
    manifest = client.get("/manifest", scope=",".join(SYNC_DIR_NAMES))
    print(f"remote root: {manifest.get('root')}")
    capabilities = set(manifest.get("capabilities", []))
    bundle_supported = BUNDLE_CAPABILITY in capabilities
    if transport == "auto":
        transport = "bundle" if bundle_supported else "chunk"
    if transport == "bundle" and not bundle_supported:
        raise RuntimeError(
            "remote sync service does not support bundle_get_v2 yet; "
            "run once with --transport chunk, then restart conf/start_tongbu.sh"
        )

    remote_files_all = manifest.get("files", {})
    remote_files = {
        rel: meta
        for rel, meta in remote_files_all.items()
        if is_in_sync_scope(rel)
    }
    local_paths = set(local_files)
    remote_paths = set(remote_files)

    if skip_unchanged or dry_run:
        changed = sorted(
            rel
            for rel, item in local_files.items()
            if rel not in remote_files or remote_files[rel].get("sha256") != item["sha256"]
        )
    else:
        changed = sorted(local_files)
    if only_paths:
        requested = set(only_paths)
        unknown = sorted(path for path in requested if path not in local_files)
        if unknown:
            raise RuntimeError(
                f"--only path is outside the local sync scope: {unknown[0]}"
            )
        changed = [path for path in changed if path in requested]
    delete_paths = sorted(remote_paths - local_paths) if delete_remote else []

    print(f"sync dirs: {', '.join(SYNC_DIR_NAMES)}")
    print(f"local files: {len(local_files)}")
    print(f"files to overwrite: {len(changed)}")
    print(f"remote delete candidates: {len(delete_paths)}")
    print(f"sync transport: {transport}")
    if dry_run:
        for rel in changed[:20]:
            print(f"  upload: {rel}")
        for rel in delete_paths[:20]:
            print(f"  delete: {rel}")
        return

    started = time.time()
    uploaded_bytes = 0
    if transport == "bundle" and changed:
        bundle = build_bundle(local_files, changed)
        print(
            f"bundle: {len(changed)} files, {len(bundle) / 1024:.1f} KiB, "
            "GET chunks: "
            f"{(len(bundle) + GET_UPLOAD_CHUNK_SIZE - 1) // GET_UPLOAD_CHUNK_SIZE}"
        )
        bundle_result = client.upload_bundle_get(bundle)
        verify_remote_hashes(bundle_result.get("files", {}), local_files, changed)
        uploaded_bytes = sum(int(local_files[rel]["size"]) for rel in changed)
        print(
            f"bundle verified: {len(changed)} files, "
            f"{uploaded_bytes / 1024:.1f} KiB"
        )
    elif changed:
        active_workers = min(upload_workers, len(changed))
        print(
            f"chunk workers: {active_workers}; "
            f"retries per file: {upload_retries}"
        )
        with ThreadPoolExecutor(
            max_workers=active_workers,
            thread_name_prefix="ide-sync",
        ) as executor:
            pending = {
                executor.submit(
                    upload_with_retry,
                    client,
                    rel,
                    local_files[rel],
                    upload_retries,
                ): rel
                for rel in changed
            }
            try:
                for index, future in enumerate(as_completed(pending), start=1):
                    rel = pending[future]
                    future.result()
                    uploaded_bytes += int(local_files[rel]["size"])
                    if index == 1 or index % 20 == 0 or index == len(changed):
                        print(
                            f"uploaded {index}/{len(changed)} files, "
                            f"{uploaded_bytes / 1024:.1f} KiB"
                        )
            except Exception:
                for future in pending:
                    future.cancel()
                raise

    if delete_paths:
        if not bundle_supported:
            raise RuntimeError(
                "safe GET deletion requires bundle_get_v2; "
                "restart the upgraded sync service first"
            )
        for rel_path in delete_paths:
            client.delete_file_get(rel_path)
        print(f"deleted remote files: {len(delete_paths)}")

    print(f"sync complete in {time.time() - started:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync this local directory to the IDE sync server.")
    parser.add_argument("--url", default=os.environ.get("IDE_SYNC_URL", DEFAULT_SYNC_URL), help="IDE forwarded URL")
    parser.add_argument(
        "--token",
        default=None,
        help="Shared sync token (overrides IDE_SYNC_TOKEN and conf/.env)",
    )
    parser.add_argument("--proxy-cookie", default=os.environ.get("IDE_PROXY_COOKIE", ""), help="Tencent proxy Cookie")
    parser.add_argument(
        "--proxy-cookie-name",
        default=os.environ.get("IDE_PROXY_COOKIE_NAME", DEFAULT_PROXY_COOKIE_NAME),
        help="Cookie name used when only the value is pasted",
    )
    parser.add_argument("--cookie-file", default=str(DEFAULT_COOKIE_FILE), help="Cached Tencent proxy Cookie file")
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
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=DEFAULT_UPLOAD_WORKERS,
        help=(
            "Parallel GET chunk uploads "
            f"(1-{MAX_UPLOAD_WORKERS}, default: {DEFAULT_UPLOAD_WORKERS})"
        ),
    )
    parser.add_argument(
        "--transport",
        "--upload-mode",
        dest="transport",
        choices=("auto", "bundle", "chunk"),
        default=DEFAULT_SYNC_TRANSPORT,
        help="auto prefers compressed GET bundle; chunk bootstraps old services",
    )
    parser.add_argument(
        "--upload-retries",
        type=int,
        default=DEFAULT_UPLOAD_RETRIES,
        help=f"Retries for an individual file (default: {DEFAULT_UPLOAD_RETRIES})",
    )
    parser.add_argument("--delete", action="store_true", help="Delete remote files absent locally")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="REL_PATH",
        help="Sync only this relative path; repeatable, useful for service bootstrap",
    )
    sync_mode = parser.add_mutually_exclusive_group()
    sync_mode.add_argument(
        "--skip-unchanged",
        dest="skip_unchanged",
        action="store_true",
        default=True,
        help="Compare remote hashes and upload changed files only (default)",
    )
    sync_mode.add_argument(
        "--force-all",
        dest="skip_unchanged",
        action="store_false",
        help="Upload every eligible local file, even when its SHA256 already matches",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-local",
        action="store_true",
        help="Validate local sync scope without Cookie or network access",
    )
    args = parser.parse_args()

    if not 1 <= args.upload_workers <= MAX_UPLOAD_WORKERS:
        parser.error(f"--upload-workers must be in [1, {MAX_UPLOAD_WORKERS}]")
    if args.upload_retries < 0:
        parser.error("--upload-retries must be non-negative")
    if args.delete and args.only:
        parser.error("--delete cannot be combined with --only")

    root = Path(args.root).resolve()
    if args.check_local:
        if args.delete:
            print("--check-local cannot be combined with --delete", file=sys.stderr)
            return 2
        return check_local_scope(root, args.max_bytes)

    args.token, token_source = resolve_sync_token(
        cli_token=args.token,
        root=root,
    )
    if not args.token:
        print(
            "missing token: pass --token, set IDE_SYNC_TOKEN, or add "
            "IDE_SYNC_TOKEN to conf/.env",
            file=sys.stderr,
        )
        return 2
    print(
        f"Sync token source: {token_source}; "
        f"sha256={sha256_bytes(args.token.encode('utf-8'))[:12]}"
    )

    cookie_file = Path(args.cookie_file).expanduser()
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
            explicit_cookie=args.proxy_cookie,
            fallback_cookie=USER_PROXY_COOKIE,
            cookie_file=cookie_file,
            cookie_name=args.proxy_cookie_name,
            no_save_cookie=args.no_save_cookie,
            no_cookie_prompt=args.no_cookie_prompt,
            refresh_cookie=args.refresh_cookie,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    client = SyncClient(
        normalize_base_url(args.url),
        args.token,
        args.timeout,
        cookie_selection.value,
    )
    try:
        sync_files(
            client,
            root,
            args.max_bytes,
            args.delete,
            args.dry_run,
            args.skip_unchanged,
            args.upload_workers,
            args.transport,
            args.upload_retries,
            tuple(args.only),
        )
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
