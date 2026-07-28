#!/usr/bin/env python3
"""Run a bounded diagnostic command in the Tencent Kaiwu IDE container."""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from local_sync_client import (
    DEFAULT_COOKIE_FILE,
    DEFAULT_PROXY_COOKIE_NAME,
    DEFAULT_SYNC_URL,
    SyncClient,
    TencentProxyAuthError,
    USER_PROXY_COOKIE,
    load_proxy_cookie,
    normalize_base_url,
    resolve_sync_token,
)


def _decode_output(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        return b""
    return base64.b64decode(value, validate=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute one diagnostic command through the IDE sync RPC service."
    )
    parser.add_argument("command", help="Shell command to execute in the container")
    parser.add_argument("--cwd", default=".", help="Container cwd relative to project root")
    parser.add_argument("--timeout", type=int, default=60)
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

    if args.timeout < 1:
        parser.error("--timeout must be positive")

    root = Path(args.root).resolve()
    token, _source = resolve_sync_token(cli_token=args.token, root=root)
    if not token:
        print(
            "missing token: pass --token, set IDE_SYNC_TOKEN, or add it to conf/.env",
            file=sys.stderr,
        )
        return 2

    cookie_file = Path(args.cookie_file).expanduser()
    cookie = load_proxy_cookie(
        explicit_cookie=args.proxy_cookie,
        fallback_cookie=USER_PROXY_COOKIE,
        cookie_file=cookie_file,
        cookie_name=args.proxy_cookie_name,
        no_save_cookie=False,
        no_cookie_prompt=args.no_cookie_prompt,
    )
    client = SyncClient(
        normalize_base_url(args.url),
        token,
        timeout=args.timeout + 15,
        proxy_cookie=cookie.value,
    )
    encoded_command = base64.urlsafe_b64encode(args.command.encode("utf-8")).decode(
        "ascii"
    )
    try:
        result = client.get(
            "/exec_b64",
            cmd=encoded_command,
            cwd=args.cwd,
            timeout=str(args.timeout),
        )
    except (RuntimeError, TencentProxyAuthError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    stdout = _decode_output(result.get("stdout_base64"))
    stderr = _decode_output(result.get("stderr_base64"))
    if stdout:
        sys.stdout.buffer.write(stdout)
        sys.stdout.buffer.flush()
    if stderr:
        sys.stderr.buffer.write(stderr)
        sys.stderr.buffer.flush()
    if result.get("stdout_truncated") or result.get("stderr_truncated"):
        print("\n[container-rpc] output truncated", file=sys.stderr)
    if result.get("timed_out"):
        print(
            f"[container-rpc] timed out after {result.get('elapsed_s', 0):.3f}s",
            file=sys.stderr,
        )
    returncode = int(result.get("returncode", 1))
    return returncode if 0 <= returncode <= 255 else 1


if __name__ == "__main__":
    raise SystemExit(main())
