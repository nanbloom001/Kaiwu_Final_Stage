#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = Path(
    os.environ.get("ARENA_MONITOR_RUNTIME_DIR", ROOT / "runtime")
).expanduser().resolve() / "probe"
SNAPSHOT_JSON = RUNTIME_DIR / "latest_snapshot.json"
REQUESTS_JSON = RUNTIME_DIR / "latest_metric_requests.json"

DEFAULT_SESSION = os.environ.get("AGENT_BROWSER_SESSION", "tencent-arena")
DEFAULT_SESSION_NAME = os.environ.get("AGENT_BROWSER_SESSION_NAME", DEFAULT_SESSION)


def decode_output(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", sys.getdefaultencoding()):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def run_agent_browser(args: list[str], timeout: int = 30) -> str:
    env = os.environ.copy()
    env["AGENT_BROWSER_SESSION"] = DEFAULT_SESSION
    env["AGENT_BROWSER_SESSION_NAME"] = DEFAULT_SESSION_NAME
    executable = shutil.which("agent-browser") or shutil.which("agent-browser.cmd") or "agent-browser"
    proc = subprocess.run(
        [executable] + args,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=False,
        timeout=timeout,
        check=True,
    )
    return decode_output(proc.stdout).strip()


def wait_ms(ms: int) -> None:
    run_agent_browser(["wait", str(ms)], timeout=max(10, ms // 1000 + 10))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe visible Tencent Arena metric cards and requests."
    )
    parser.add_argument(
        "--tab",
        help="Optional agent-browser tab id, for example t3. Defaults to the active tab.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    if args.tab:
        run_agent_browser(["tab", args.tab], timeout=20)
    run_agent_browser(["find", "text", "监控总览", "click", "--exact"], timeout=20)
    wait_ms(1500)
    run_agent_browser(["network", "requests", "--clear"], timeout=20)

    raw = run_agent_browser(["snapshot", "-i", "--json"], timeout=40)
    SNAPSHOT_JSON.write_text(raw, encoding="utf-8")
    payload = json.loads(raw)
    refs = payload["data"]["refs"]
    clickable_view_refs = []
    for ref_id, ref in refs.items():
        if ref.get("role") == "generic" and ref.get("name") == "查看":
            clickable_view_refs.append(ref_id)

    # Click every visible "查看" card trigger once. Re-snapshot after each click
    # because refs may change after the page updates.
    clicked = 0
    for _ in range(min(48, len(clickable_view_refs))):
        raw = run_agent_browser(["snapshot", "-i", "--json"], timeout=40)
        payload = json.loads(raw)
        refs = payload["data"]["refs"]
        target = None
        for ref_id, ref in refs.items():
            if ref.get("role") == "generic" and ref.get("name") == "查看":
                target = ref_id
                break
        if not target:
            break
        run_agent_browser(["click", f"@{target}"], timeout=20)
        wait_ms(1800)
        clicked += 1
        run_agent_browser(["scroll", "down", "300"], timeout=15)
        wait_ms(600)

    requests = run_agent_browser(
        ["network", "requests", "--filter", "GetTrainMetricRange", "--method", "POST", "--status", "2xx", "--json"],
        timeout=30,
    )
    REQUESTS_JSON.write_text(requests, encoding="utf-8")
    print(f"clicked={clicked}")
    print(SNAPSHOT_JSON)
    print(REQUESTS_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
