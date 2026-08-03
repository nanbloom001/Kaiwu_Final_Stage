#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote

from browser_auth import agent_browser_env


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = Path(
    os.environ.get("ARENA_MONITOR_RUNTIME_DIR", ROOT / "runtime")
).expanduser().resolve()
HISTORY_DIR = RUNTIME_DIR / "history"
LATEST_JSON = RUNTIME_DIR / "latest_snapshot.json"
INDEX_HTML = RUNTIME_DIR / "index.html"

DEFAULT_SESSION = os.environ.get("AGENT_BROWSER_SESSION", "tencent-arena")
DEFAULT_SESSION_NAME = os.environ.get("AGENT_BROWSER_SESSION_NAME", DEFAULT_SESSION)

MONITOR_URL_HINT = "/p/v5/exp/monitor"
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
GROUP_RE = re.compile(r"^(.+?)\(\s*(\d+)\s*\)$")


def run_agent_browser(args: list[str], session: str, session_name: str, timeout: int = 30) -> str:
    env = agent_browser_env(session, session_name)
    executable = shutil.which("agent-browser") or shutil.which("agent-browser.cmd") or "agent-browser"
    cmd = [executable] + args
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=False,
        timeout=timeout,
        check=True,
    )
    return decode_output(proc.stdout).strip()


def decode_output(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", sys.getdefaultencoding()):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def ensure_runtime_dirs() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def split_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def find_monitor_tab(session: str, session_name: str) -> str | None:
    output = run_agent_browser(["tab"], session, session_name, timeout=15)
    for line in output.splitlines():
        if MONITOR_URL_HINT in line:
            match = re.search(r"\[(t\d+)\]", line)
            if match:
                return match.group(1)
    return None


def switch_to_monitor_tab(session: str, session_name: str, monitor_url: str | None) -> None:
    tab = find_monitor_tab(session, session_name)
    if tab:
        run_agent_browser(["tab", tab], session, session_name, timeout=15)
        return

    if not monitor_url:
        raise RuntimeError(
            "No Tencent Arena monitor tab found in the current browser session. "
            "Open the monitor page first, or pass --monitor-url."
        )
    run_agent_browser(["open", monitor_url], session, session_name, timeout=30)
    time.sleep(2)


def safe_click_tab(label: str, session: str, session_name: str) -> None:
    run_agent_browser(["find", "text", label, "click", "--exact"], session, session_name, timeout=20)
    time.sleep(1.5)


def get_body_text(session: str, session_name: str) -> str:
    return run_agent_browser(["get", "text", "body"], session, session_name, timeout=30)


def wait_ms(ms: int, session: str, session_name: str) -> None:
    run_agent_browser(["wait", str(ms)], session, session_name, timeout=max(10, ms // 1000 + 10))


def scroll_page(direction: str, px: int, session: str, session_name: str) -> None:
    run_agent_browser(["scroll", direction, str(px)], session, session_name, timeout=15)


def prime_overview_lazy_sections(session: str, session_name: str) -> None:
    # Some Tencent Arena metric cards only issue data requests after their section
    # has entered the viewport, so a top-to-bottom sweep is part of collection.
    scroll_page("up", 30000, session, session_name)
    wait_ms(800, session, session_name)

    for _ in range(10):
        scroll_page("down", 1600, session, session_name)
        wait_ms(900, session, session_name)

    scroll_page("up", 30000, session, session_name)
    wait_ms(1200, session, session_name)


def parse_header(lines: list[str]) -> dict[str, str]:
    keys = {
        "状态",
        "ID",
        "版本",
        "运行时长 (任务时长)",
        "算法",
        "训练模式",
        "错误日志数量",
        "日志数量",
    }
    result: dict[str, str] = {}
    for idx, line in enumerate(lines[:-1]):
        if line in keys:
            result[line] = lines[idx + 1]
    return result


def parse_overview(text: str) -> dict[str, Any]:
    lines = split_lines(text)
    header = parse_header(lines)

    groups: list[dict[str, Any]] = []
    current_group: dict[str, Any] | None = None
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        group_match = GROUP_RE.match(line)
        if group_match:
            current_group = {
                "name": group_match.group(1).strip(),
                "count": int(group_match.group(2)),
                "metrics": [],
            }
            groups.append(current_group)
            idx += 1
            continue

        if current_group is not None:
            next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
            if next_line == "查看":
                metric = {"name": line, "value": None}
                value_idx = idx + 2
                if value_idx < len(lines):
                    maybe_value = lines[value_idx]
                    value_followed_by_view = value_idx + 1 < len(lines) and lines[value_idx + 1] == "查看"
                    if (
                        maybe_value != "查看"
                        and not GROUP_RE.match(maybe_value)
                        and not TIMESTAMP_RE.match(maybe_value)
                        and not value_followed_by_view
                    ):
                        metric["value"] = maybe_value
                        idx = value_idx
                current_group["metrics"].append(metric)
        idx += 1

    return {
        "header": header,
        "groups": groups,
        "raw_text": text,
    }


def parse_log_entry(chunk_lines: list[str]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "timestamp": chunk_lines[0],
        "message": chunk_lines[1] if len(chunk_lines) > 1 else "",
        "fields": {},
        "raw_lines": chunk_lines,
    }
    i = 2
    while i + 1 < len(chunk_lines):
        key = chunk_lines[i]
        val = chunk_lines[i + 1]
        if TIMESTAMP_RE.match(key):
            break
        entry["fields"][key] = val
        i += 2
    return entry


def parse_logs(text: str) -> dict[str, Any]:
    lines = split_lines(text)
    header = parse_header(lines)
    entries: list[dict[str, Any]] = []
    current: list[str] = []

    for line in lines:
        if TIMESTAMP_RE.match(line):
            if current:
                entries.append(parse_log_entry(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        entries.append(parse_log_entry(current))

    return {
        "header": header,
        "entries": entries,
        "raw_text": text,
    }


def collect_once(session: str, session_name: str, monitor_url: str | None) -> dict[str, Any]:
    switch_to_monitor_tab(session, session_name, monitor_url)

    safe_click_tab("监控总览", session, session_name)
    prime_overview_lazy_sections(session, session_name)
    overview_text = get_body_text(session, session_name)

    safe_click_tab("训练日志", session, session_name)
    logs_text = get_body_text(session, session_name)

    safe_click_tab("监控总览", session, session_name)

    now = dt.datetime.now(dt.timezone.utc)
    snapshot = {
        "captured_at_utc": now.isoformat(),
        "captured_at_local": dt.datetime.now().isoformat(timespec="seconds"),
        "session": session,
        "session_name": session_name,
        "overview": parse_overview(overview_text),
        "logs": parse_logs(logs_text),
    }
    return snapshot


def write_snapshot(snapshot: dict[str, Any]) -> Path:
    ensure_runtime_dirs()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = HISTORY_DIR / f"{stamp}.json"
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2)
    out_path.write_text(payload, encoding="utf-8")
    LATEST_JSON.write_text(payload, encoding="utf-8")
    return out_path


def build_html(snapshot: dict[str, Any], interval: int) -> str:
    overview = snapshot.get("overview", {})
    logs = snapshot.get("logs", {})
    groups = overview.get("groups", [])
    log_entries = logs.get("entries", [])[:120]
    header = overview.get("header", {})

    header_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in header.items()
    )

    group_blocks = []
    for group in groups:
        metrics = group.get("metrics", [])
        metric_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(metric.get('name', '')))}</td>"
            f"<td>{html.escape(str(metric.get('value', '')))}</td>"
            "</tr>"
            for metric in metrics
        )
        group_blocks.append(
            "<section class='group'>"
            f"<h3>{html.escape(group.get('name', ''))} ({group.get('count', 0)})</h3>"
            "<table><thead><tr><th>Metric</th><th>Visible Value</th></tr></thead>"
            f"<tbody>{metric_rows}</tbody></table>"
            "</section>"
        )

    log_rows = []
    for entry in log_entries:
        fields = entry.get("fields", {})
        log_rows.append(
            "<tr>"
            f"<td>{html.escape(entry.get('timestamp', ''))}</td>"
            f"<td>{html.escape(fields.get('level', ''))}</td>"
            f"<td>{html.escape(fields.get('module', ''))}</td>"
            f"<td>{html.escape(entry.get('message', ''))}</td>"
            "</tr>"
        )

    group_html = "\n".join(group_blocks)
    logs_html = "\n".join(log_rows)
    captured = html.escape(str(snapshot.get("captured_at_local", "")))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{max(interval, 5)}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Arena Frontend Monitor</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 20px;
      color: #111;
      background: #fafafa;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    .meta, .group, .logs {{
      background: #fff;
      border: 1px solid #ddd;
      padding: 16px;
      margin-bottom: 16px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #e2e2e2;
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f4f4f4;
      white-space: nowrap;
    }}
    .muted {{
      color: #666;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <h1>Arena Frontend Monitor</h1>
  <p class="muted">Captured at {captured}. Auto-refresh every {max(interval, 5)} seconds.</p>

  <section class="meta">
    <h2>Task Metadata</h2>
    <table>
      <tbody>{header_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Monitor Overview</h2>
    {group_html}
  </section>

  <section class="logs">
    <h2>Recent Training Logs</h2>
    <table>
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Level</th>
          <th>Module</th>
          <th>Message</th>
        </tr>
      </thead>
      <tbody>{logs_html}</tbody>
    </table>
  </section>
</body>
</html>
"""


def render_snapshot(snapshot: dict[str, Any], interval: int) -> None:
    ensure_runtime_dirs()
    INDEX_HTML.write_text(build_html(snapshot, interval), encoding="utf-8")


class RuntimeDirHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(RUNTIME_DIR), **kwargs)


def serve_http(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), RuntimeDirHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def serve_loop(args: argparse.Namespace) -> None:
    ensure_runtime_dirs()
    server = serve_http(args.port)
    print(f"Serving dashboard at http://127.0.0.1:{args.port}")
    try:
        while True:
            snapshot = collect_once(args.session, args.session_name, args.monitor_url)
            out_path = write_snapshot(snapshot)
            render_snapshot(snapshot, args.interval)
            print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] wrote {out_path}")
            time.sleep(args.interval)
    finally:
        server.shutdown()


def collect_once_cmd(args: argparse.Namespace) -> None:
    snapshot = collect_once(args.session, args.session_name, args.monitor_url)
    out_path = write_snapshot(snapshot)
    render_snapshot(snapshot, args.interval)
    print(out_path)
    print(INDEX_HTML)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tencent Arena frontend monitor collector")
    parser.set_defaults(func=None)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--session", default=DEFAULT_SESSION)
    common.add_argument("--session-name", default=DEFAULT_SESSION_NAME)
    common.add_argument("--monitor-url", default=None)
    common.add_argument("--interval", type=int, default=15)

    collect_parser = parser.add_subparsers(dest="command")

    once = collect_parser.add_parser("collect-once", parents=[common], help="collect one frontend snapshot")
    once.set_defaults(func=collect_once_cmd)

    serve = collect_parser.add_parser("serve", parents=[common], help="collect snapshots and serve local HTML")
    serve.add_argument("--port", type=int, default=8877)
    serve.set_defaults(func=serve_loop)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command or not args.func:
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
