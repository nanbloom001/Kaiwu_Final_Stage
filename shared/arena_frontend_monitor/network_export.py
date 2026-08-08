#!/usr/bin/env python3
"""Export Tencent Arena monitor logs and metric curves from the web frontend.

The exporter intentionally stays independent of the development-container RPC
bridge. It drives an already logged-in agent-browser session, triggers the
monitor page's lazy-loaded cards, captures frontend API traffic, and writes
AI-readable JSON artifacts under the configured runtime directory.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from browser_auth import agent_browser_command
from browser_auth import default_session_from_env, default_session_name_from_env


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = Path(
    os.environ.get("ARENA_MONITOR_RUNTIME_DIR", ROOT / "runtime")
).expanduser().resolve()
NETWORK_DIR = RUNTIME_DIR / "network_capture"
SESSIONS_DIR = NETWORK_DIR / "sessions"
INDEX_JSONL = NETWORK_DIR / "index.jsonl"

DEFAULT_SESSION = default_session_from_env()
DEFAULT_SESSION_NAME = default_session_name_from_env(DEFAULT_SESSION)

MONITOR_URL_HINT = "/p/v5/exp/monitor"
IDE_URL_HINT = "/p/common/competition/ide/"
API_LOG = "GetTrainLog"
API_METRIC = "GetTrainMetricRange"

OVERVIEW_TAB = "\u76d1\u63a7\u603b\u89c8"
LOGS_TAB = "\u8bad\u7ec3\u65e5\u5fd7"
VIEW_TOKEN = "\u67e5\u770b"
NO_DATA_TOKEN = "\u6682\u65e0\u6570\u636e"
DISABLE_AUTO_REFRESH = "\u7981\u7528\u81ea\u52a8\u5237\u65b0"
DEFAULT_AUTO_REFRESH = "\u6bcf 5 \u79d2\u81ea\u52a8\u5237\u65b0"

HEADER_KEYS = {
    "\u72b6\u6001",
    "ID",
    "\u7248\u672c",
    "\u8fd0\u884c\u65f6\u957f (\u4efb\u52a1\u65f6\u957f)",
    "\u7b97\u6cd5",
    "\u8bad\u7ec3\u6a21\u5f0f",
    "\u9519\u8bef\u65e5\u5fd7\u6570\u91cf",
    "\u65e5\u5fd7\u6570\u91cf",
}

GLOBAL_ENV_RESULT_IDS = {
    "completed_count_0": "completed_count",
    "abnormal_count_1": "abnormal_count",
    "timeout_count_2": "timeout_count",
    "total_score_0": "total_score",
    "forward_score_1": "forward_score",
    "step_score_2": "step_score",
    "energy_score_3": "energy_score",
    "pose_score_4": "pose_score",
    "step_avg_0": "step_avg",
}


class CaptureError(RuntimeError):
    """A recoverable capture failure that should still produce summary output."""


class AgentBrowserError(RuntimeError):
    """Wrap agent-browser failures with stdout/stderr for capture summaries."""


def decode_output(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", sys.getdefaultencoding()):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def run_agent_browser(args: list[str], session: str, session_name: str, timeout: int = 30) -> str:
    cmd, env = agent_browser_command(args, session, session_name)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        stdout = decode_output(proc.stdout or b"").strip()
        stderr = decode_output(proc.stderr or b"").strip()
        detail = stderr or stdout or f"exit code {proc.returncode}"
        raise AgentBrowserError(f"agent-browser {' '.join(args)} failed: {detail}")
    return decode_output(proc.stdout).strip()


def run_agent_browser_safe(
    args: list[str],
    session: str,
    session_name: str,
    timeout: int = 30,
) -> tuple[bool, str]:
    try:
        return True, run_agent_browser(args, session, session_name, timeout=timeout)
    except AgentBrowserError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001 - CLI boundary should report every failure.
        return False, str(exc)


def ensure_dirs() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def wait_ms(ms: int, session: str, session_name: str) -> None:
    run_agent_browser(["wait", str(ms)], session, session_name, timeout=max(10, ms // 1000 + 10))


def click_text(text: str, session: str, session_name: str, exact: bool = True) -> None:
    args = ["find", "text", text, "click"]
    if exact:
        args.append("--exact")
    run_agent_browser(args, session, session_name, timeout=20)


def press_key(key: str, session: str, session_name: str) -> None:
    run_agent_browser(["press", key], session, session_name, timeout=15)


def scroll_down(px: int, session: str, session_name: str) -> None:
    run_agent_browser(["scroll", "down", str(px)], session, session_name, timeout=15)


def scroll_up(px: int, session: str, session_name: str) -> None:
    run_agent_browser(["scroll", "up", str(px)], session, session_name, timeout=15)


def get_body_text(session: str, session_name: str) -> str:
    return run_agent_browser(["get", "text", "body"], session, session_name, timeout=30)


def get_current_url(session: str, session_name: str) -> str:
    return run_agent_browser(["get", "url"], session, session_name, timeout=15)


def list_tabs(session: str, session_name: str) -> str:
    return run_agent_browser(["tab", "list"], session, session_name, timeout=15)


def _tab_id_from_line(line: str) -> str | None:
    match = re.search(r"\[(t\d+)\]", line)
    return match.group(1) if match else None


def find_monitor_tab(session: str, session_name: str, monitor_url: str | None = None) -> str | None:
    output = list_tabs(session, session_name)
    if monitor_url:
        for line in output.splitlines():
            if monitor_url in line:
                return _tab_id_from_line(line)
    for line in output.splitlines():
        if MONITOR_URL_HINT in line:
            return _tab_id_from_line(line)
    return None


def switch_to_monitor_tab(session: str, session_name: str, monitor_url: str | None) -> None:
    tab = find_monitor_tab(session, session_name, monitor_url)
    if tab:
        run_agent_browser(["tab", tab], session, session_name, timeout=15)
        return
    if not monitor_url:
        raise CaptureError(
            "No Tencent Arena monitor tab found. Open the monitor page first or pass --monitor-url."
        )
    run_agent_browser(["open", monitor_url], session, session_name, timeout=45)
    wait_ms(2500, session, session_name)


def ensure_monitor_ready(session: str, session_name: str) -> tuple[str, str]:
    url = get_current_url(session, session_name)
    if IDE_URL_HINT in url:
        raise CaptureError(f"Current tab is the IDE page, not the monitor page: {url}")
    if MONITOR_URL_HINT not in url:
        raise CaptureError(f"Current tab is not a Tencent Arena monitor page: {url}")
    body = get_body_text(session, session_name)
    if OVERVIEW_TAB not in body and LOGS_TAB not in body:
        raise CaptureError("Monitor page did not expose expected tabs in visible text.")
    return url, body


def ensure_auto_refresh(
    session: str,
    session_name: str,
    target_label: str,
    errors: list[str],
) -> bool:
    for attempt in range(1, 4):
        body = get_body_text(session, session_name)
        if target_label in body and DISABLE_AUTO_REFRESH not in body:
            return True
        if target_label in body and DISABLE_AUTO_REFRESH in body:
            # The dropdown may include both current value and options. Prefer a
            # cheap click attempt before deciding it is already enabled.
            pass
        ok, output = run_agent_browser_safe(
            ["find", "text", DISABLE_AUTO_REFRESH, "click", "--exact"],
            session,
            session_name,
            timeout=20,
        )
        if not ok:
            ok, output = run_agent_browser_safe(
                ["find", "text", target_label, "click", "--exact"],
                session,
                session_name,
                timeout=20,
            )
            if ok:
                wait_ms(5000, session, session_name)
                return True
            errors.append(f"auto refresh open attempt {attempt} failed: {output}")
            wait_ms(1200, session, session_name)
            continue

        wait_ms(1200, session, session_name)
        ok, output = run_agent_browser_safe(
            ["find", "text", target_label, "click", "--exact"],
            session,
            session_name,
            timeout=20,
        )
        if ok:
            wait_ms(5000, session, session_name)
            return True
        errors.append(f"auto refresh select attempt {attempt} failed: {output}")
        wait_ms(1200, session, session_name)
    return False


def prepare_dense_overview_view(
    session: str,
    session_name: str,
    zoom_out_steps: int,
    settle_ms: int,
) -> None:
    click_text(OVERVIEW_TAB, session, session_name)
    wait_ms(1200, session, session_name)
    press_key("Control+0", session, session_name)
    wait_ms(800, session, session_name)
    for _ in range(max(0, zoom_out_steps)):
        press_key("Control+-", session, session_name)
        wait_ms(700, session, session_name)
    wait_ms(settle_ms, session, session_name)


def prime_overview_lazy_cards(
    session: str,
    session_name: str,
    scroll_steps: int,
    scroll_px: int,
    dwell_ms: int,
    passes: int,
) -> None:
    for pass_index in range(max(1, passes)):
        scroll_up(max(scroll_steps, 1) * scroll_px, session, session_name)
        wait_ms(2000, session, session_name)
        if pass_index % 2 == 0:
            for _ in range(scroll_steps):
                scroll_down(scroll_px, session, session_name)
                wait_ms(dwell_ms, session, session_name)
        else:
            scroll_down(max(scroll_steps, 1) * scroll_px, session, session_name)
            wait_ms(2000, session, session_name)
            for _ in range(scroll_steps):
                scroll_up(scroll_px, session, session_name)
                wait_ms(dwell_ms, session, session_name)
        wait_ms(5000, session, session_name)


def parse_json_or_none(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def decode_har_content(content: dict[str, Any]) -> str:
    text = content.get("text") or ""
    if content.get("encoding") == "base64" and text:
        try:
            return base64.b64decode(text).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return text


def normalize_lines(text: str) -> list[str]:
    return [line.strip().lstrip("\ufeff") for line in text.splitlines() if line.strip()]


def build_page_metric_inventory(body_text: str) -> dict[str, Any]:
    lines = normalize_lines(body_text)
    header: dict[str, str] = {}
    for idx, line in enumerate(lines[:-1]):
        if line in HEADER_KEYS:
            header[line] = lines[idx + 1]

    groups: list[dict[str, Any]] = []
    current_group: dict[str, Any] | None = None
    skip_tokens = {VIEW_TOKEN, NO_DATA_TOKEN, OVERVIEW_TAB, LOGS_TAB, DEFAULT_AUTO_REFRESH, DISABLE_AUTO_REFRESH}

    for idx, line in enumerate(lines):
        if "(" in line and line.endswith(")"):
            left, right = line.rsplit("(", 1)
            count = right[:-1].strip()
            if count.isdigit():
                current_group = {"name": left.strip(), "declared_count": int(count), "metrics": []}
                groups.append(current_group)
                continue

        if current_group is None or line in skip_tokens or line in HEADER_KEYS:
            continue

        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        if next_line == VIEW_TOKEN:
            status = "has_card"
            if idx + 2 < len(lines) and lines[idx + 2] == NO_DATA_TOKEN:
                status = "no_data"
            current_group["metrics"].append({"name": line, "page_status": status})

    return {
        "header": header,
        "group_count": len(groups),
        "metric_count": sum(len(group["metrics"]) for group in groups),
        "groups": groups,
        "raw_text": body_text,
    }


def as_number(value: Any) -> float | int | None:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return int(parsed) if parsed.is_integer() else parsed
    return None


def summarize_metric_series(item: dict[str, Any]) -> dict[str, Any]:
    values = item.get("values", [])
    latest = values[-1] if values else None
    first = values[0] if values else None
    previous = values[-2] if len(values) > 1 else None
    numeric_values = [as_number(point.get("value")) for point in values]
    numeric_values = [value for value in numeric_values if value is not None]
    latest_value = None if latest is None else as_number(latest.get("value"))
    previous_value = None if previous is None else as_number(previous.get("value"))
    first_value = None if first is None else as_number(first.get("value"))

    trend = None
    if first_value is not None and latest_value is not None:
        trend = "up" if latest_value > first_value else "down" if latest_value < first_value else "flat"

    return {
        "labels": item.get("labels", {}),
        "point_count": len(values),
        "nonzero_point_count": len([value for value in numeric_values if value != 0]),
        "first_value": first_value,
        "first_timestamp": None if first is None else first.get("timestamp"),
        "latest_value": latest_value,
        "latest_timestamp": None if latest is None else latest.get("timestamp"),
        "previous_value": previous_value,
        "previous_timestamp": None if previous is None else previous.get("timestamp"),
        "min_value": None if not numeric_values else min(numeric_values),
        "max_value": None if not numeric_values else max(numeric_values),
        "delta_from_first": None if first_value is None or latest_value is None else latest_value - first_value,
        "delta_from_previous": None
        if previous_value is None or latest_value is None
        else latest_value - previous_value,
        "trend_from_first": trend,
        "sample_head": values[:5],
        "sample_tail": values[-5:] if len(values) > 5 else values,
    }


def metric_result_to_compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": result.get("id"),
        "item_count": len(result.get("items", [])),
        "items": [summarize_metric_series(item) for item in result.get("items", [])],
    }


def normalize_log_entry(raw: str) -> dict[str, Any]:
    parsed = parse_json_or_none(raw)
    return parsed if isinstance(parsed, dict) else {"raw": raw}


def build_log_view(log_requests: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in log_requests:
        raw_logs = ((item.get("response_json") or {}).get("data") or {}).get("logs") or []
        for raw in raw_logs:
            entry = normalize_log_entry(raw)
            key = (
                entry.get("time"),
                entry.get("level"),
                entry.get("module"),
                entry.get("file"),
                entry.get("function"),
                entry.get("line"),
                entry.get("message"),
                entry.get("raw"),
            )
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
    entries.sort(key=lambda entry: str(entry.get("time") or ""), reverse=True)

    by_level: dict[str, int] = {}
    by_module: dict[str, int] = {}
    for entry in entries:
        level = str(entry.get("level", "UNKNOWN"))
        module = str(entry.get("module", "UNKNOWN"))
        by_level[level] = by_level.get(level, 0) + 1
        by_module[module] = by_module.get(module, 0) + 1
    return {"entry_count": len(entries), "levels": by_level, "modules": by_module, "entries": entries}


def build_metric_view(metric_requests: list[dict[str, Any]]) -> dict[str, Any]:
    request_views = []
    series_by_metric: dict[str, list[dict[str, Any]]] = {}
    for item in metric_requests:
        results = ((item.get("response_json") or {}).get("data") or {}).get("results", [])
        compact_results = [metric_result_to_compact(result) for result in results]
        result_ids = [str(result.get("id")) for result in results if result.get("id") is not None]
        request_views.append(
            {
                "source": item.get("source"),
                "request_id": item.get("request_id"),
                "request_url": item.get("url"),
                "query_count": item.get("query_count"),
                "query_names": item.get("query_names"),
                "result_ids": result_ids,
                "result_count": len(compact_results),
                "results": compact_results,
            }
        )
        for result in compact_results:
            series_by_metric.setdefault(str(result.get("id", "unknown")), []).append(result)
    return {"request_count": len(metric_requests), "request_views": request_views, "series_by_metric": series_by_metric}


def build_global_env_metrics_view(metric_requests: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for request in sorted(metric_requests, key=lambda item: item.get("timestamp") or 0):
        results = ((request.get("response_json") or {}).get("data") or {}).get("results", [])
        for result in results:
            result_id = result.get("id")
            metric_name = GLOBAL_ENV_RESULT_IDS.get(str(result_id))
            items = result.get("items") or []
            if not metric_name or not items:
                continue
            metrics[metric_name] = {
                "result_id": result_id,
                "source": request.get("source"),
                "source_request_id": request.get("request_id"),
                "captured_request_timestamp": request.get("timestamp"),
                "series": summarize_metric_series(items[0]),
            }
    return {"metric_count": len(metrics), "metrics": metrics}


def get_request_list(api_name: str, session: str, session_name: str) -> list[dict[str, Any]]:
    output = run_agent_browser(
        ["network", "requests", "--filter", api_name, "--method", "POST", "--status", "2xx", "--json"],
        session,
        session_name,
        timeout=20,
    )
    payload = json.loads(output)
    return payload.get("data", {}).get("requests", [])


def get_request_detail(request_id: str, session: str, session_name: str) -> dict[str, Any]:
    output = run_agent_browser(["network", "request", request_id, "--json"], session, session_name, timeout=20)
    payload = json.loads(output)
    return payload.get("data", {})


def clear_network_requests(session: str, session_name: str) -> None:
    run_agent_browser(["network", "requests", "--clear"], session, session_name, timeout=15)


def _fetch_api_details(api_name: str, session: str, session_name: str, source: str) -> list[dict[str, Any]]:
    details = []
    for req in get_request_list(api_name, session, session_name):
        request_id = req.get("requestId") or req.get("id")
        if not request_id:
            continue
        detail = get_request_detail(request_id, session, session_name)
        post_data = parse_json_or_none(detail.get("postData")) or {}
        response_json = parse_json_or_none(detail.get("responseBody")) or {}
        queries = post_data.get("queries") or []
        details.append(
            {
                "source": source,
                "api_name": api_name,
                "request_id": detail.get("requestId") or request_id,
                "timestamp": detail.get("timestamp"),
                "url": detail.get("url"),
                "status": detail.get("status"),
                "post_data": post_data,
                "response_json": response_json,
                "query_names": [query.get("name") for query in queries if query.get("name")],
                "query_count": len(queries),
                "query_type": "metric_range" if api_name == API_METRIC else post_data.get("query"),
                "keyword": None,
            }
        )
    return details


def collect_network_requests(
    session: str,
    session_name: str,
    log_settle_ms: int,
    log_scroll_steps: int,
    scroll_px: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_requests = _fetch_api_details(API_METRIC, session, session_name, "network")
    click_text(LOGS_TAB, session, session_name)
    wait_ms(log_settle_ms, session, session_name)
    for _ in range(log_scroll_steps):
        scroll_down(scroll_px, session, session_name)
        wait_ms(log_settle_ms, session, session_name)
    log_requests = _fetch_api_details(API_LOG, session, session_name, "network")
    return metric_requests, log_requests


def start_har(har_path: Path, session: str, session_name: str) -> None:
    run_agent_browser(["network", "har", "start", str(har_path)], session, session_name, timeout=15)


def stop_har(session: str, session_name: str) -> tuple[str | None, str]:
    output = run_agent_browser(["network", "har", "stop"], session, session_name, timeout=45)
    for line in output.splitlines():
        if ".har" not in line:
            continue
        match = re.search(r"([A-Za-z]:\\[^\r\n]+?\.har|/[^\s]+\.har)", line)
        if match:
            return match.group(1).strip(), output
    return None, output


def _har_timestamp(entry: dict[str, Any]) -> int | None:
    started = entry.get("startedDateTime") or ""
    if not started:
        return None
    try:
        return int(dt.datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def parse_har_entries(har_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    har = json.loads(har_path.read_text(encoding="utf-8"))
    metric_requests: list[dict[str, Any]] = []
    log_requests: list[dict[str, Any]] = []
    for entry in har.get("log", {}).get("entries", []):
        request = entry.get("request", {})
        response = entry.get("response", {})
        url = request.get("url", "")
        if API_METRIC not in url and API_LOG not in url:
            continue
        post_data = parse_json_or_none((request.get("postData") or {}).get("text")) or {}
        response_text = decode_har_content(response.get("content") or {})
        response_json = parse_json_or_none(response_text) or {}
        queries = post_data.get("queries") or []
        is_metric = API_METRIC in url
        collection = metric_requests if is_metric else log_requests
        collection.append(
            {
                "source": "har",
                "api_name": API_METRIC if is_metric else API_LOG,
                "request_id": f"har-{API_METRIC if is_metric else API_LOG}-{len(collection)}",
                "timestamp": _har_timestamp(entry),
                "url": url,
                "status": response.get("status"),
                "post_data": post_data,
                "response_json": response_json,
                "query_names": [query.get("name") for query in queries if query.get("name")],
                "query_count": len(queries),
                "query_type": "metric_range" if is_metric else post_data.get("query"),
                "keyword": None,
            }
        )
    return metric_requests, log_requests


def copy_har_to_session(source_path: Path, raw_dir: Path) -> Path | None:
    if not source_path.is_file():
        return None
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "capture.har"
    shutil.copy2(source_path, target)
    return target


def collect_har_requests(
    session: str,
    session_name: str,
    raw_dir: Path,
    scroll_steps: int,
    scroll_px: int,
    dwell_ms: int,
    passes: int,
    log_settle_ms: int,
    log_scroll_steps: int,
    errors: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    requested_har_path = raw_dir / "requested-capture.har"
    har_meta: dict[str, Any] = {"requested_har_path": str(requested_har_path)}
    start_har(requested_har_path, session, session_name)
    try:
        prime_overview_lazy_cards(session, session_name, scroll_steps, scroll_px, dwell_ms, passes)
        click_text(LOGS_TAB, session, session_name)
        wait_ms(log_settle_ms, session, session_name)
        for _ in range(log_scroll_steps):
            scroll_down(scroll_px, session, session_name)
            wait_ms(log_settle_ms, session, session_name)
    finally:
        actual_path_text, stop_output = stop_har(session, session_name)
        har_meta["stop_output"] = stop_output
        har_meta["actual_har_path"] = actual_path_text

    candidates = [Path(actual_path_text)] if actual_path_text else []
    candidates.append(requested_har_path)
    har_source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if not har_source:
        errors.append(f"HAR file not found. stop output: {stop_output}")
        return [], [], har_meta
    saved = copy_har_to_session(har_source, raw_dir)
    har_meta["saved_har_file"] = None if saved is None else str(saved.relative_to(NETWORK_DIR))
    try:
        return (*parse_har_entries(har_source), har_meta)
    except Exception as exc:  # noqa: BLE001 - preserve capture summary on malformed HAR.
        errors.append(f"HAR parse failed: {exc}")
        return [], [], har_meta


def request_summary_line(capture_id: str, seq: int, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "capture_id": capture_id,
        "seq": seq,
        "source": item.get("source"),
        "timestamp": item.get("timestamp"),
        "api_name": item.get("api_name"),
        "request_id": item.get("request_id"),
        "url": item.get("url"),
        "status": item.get("status"),
        "query_type": item.get("query_type"),
        "query_names": item.get("query_names"),
        "query_count": item.get("query_count"),
        "request_file": item.get("request_file"),
        "response_file": item.get("response_file"),
        "meta_file": item.get("meta_file"),
    }


def unique_query_names(metric_requests: list[dict[str, Any]]) -> list[str]:
    names = {name for item in metric_requests for name in item.get("query_names", []) if name}
    return sorted(names)


def unique_result_ids(metric_requests: list[dict[str, Any]]) -> list[str]:
    result_ids: set[str] = set()
    for item in metric_requests:
        for result in ((item.get("response_json") or {}).get("data") or {}).get("results", []):
            if result.get("id") is not None:
                result_ids.add(str(result.get("id")))
    return sorted(result_ids)


def build_coverage_report(page_inventory: dict[str, Any], metric_requests: list[dict[str, Any]]) -> dict[str, Any]:
    query_names = unique_query_names(metric_requests)
    result_ids = unique_result_ids(metric_requests)
    query_name_set = set(query_names)
    heuristic_matches = []
    missing_cards = []
    group_summary = []

    for group in page_inventory.get("groups", []):
        metrics = group.get("metrics", [])
        heuristic_count = 0
        group_missing = []
        for metric in metrics:
            name = metric.get("name", "")
            normalized_name = name.lower().replace(" ", "_")
            matched = [
                query_name
                for query_name in query_name_set
                if normalized_name and normalized_name in query_name.lower()
            ]
            if matched:
                heuristic_count += 1
                heuristic_matches.append({"page_card": name, "query_names": matched})
            else:
                group_missing.append(name)
                missing_cards.append({"group": group.get("name", ""), "name": name})
        group_summary.append(
            {
                "group_name": group.get("name", ""),
                "declared_count": group.get("declared_count", 0),
                "page_metric_count": len(metrics),
                "name_match_heuristic_count": heuristic_count,
                "missing_by_name_heuristic_count": len(group_missing),
            }
        )

    return {
        "page_group_count": page_inventory.get("group_count", 0),
        "page_metric_count": page_inventory.get("metric_count", 0),
        "captured_query_names": query_names,
        "captured_result_ids": result_ids,
        "unique_query_name_count": len(query_names),
        "unique_result_id_count": len(result_ids),
        "missing_page_cards": missing_cards,
        "group_summary": group_summary,
        "name_match_heuristic": heuristic_matches,
    }


def write_capture(
    capture_id: str,
    metric_requests: list[dict[str, Any]],
    log_requests: list[dict[str, Any]],
    page_inventory: dict[str, Any],
    coverage_report: dict[str, Any],
    summary_extra: dict[str, Any],
) -> Path:
    ensure_dirs()
    session_dir = SESSIONS_DIR / capture_id
    requests_dir = session_dir / "requests"
    views_dir = session_dir / "views"
    requests_dir.mkdir(parents=True, exist_ok=True)
    views_dir.mkdir(parents=True, exist_ok=True)

    index_lines = []
    for seq, item in enumerate(metric_requests + log_requests, start=1):
        prefix = f"{seq:04d}-{item['api_name']}"
        request_file = requests_dir / f"{prefix}.request.json"
        response_file = requests_dir / f"{prefix}.response.json"
        meta_file = requests_dir / f"{prefix}.meta.json"
        request_file.write_text(json.dumps(item.get("post_data"), ensure_ascii=False, indent=2), encoding="utf-8")
        response_file.write_text(json.dumps(item.get("response_json"), ensure_ascii=False, indent=2), encoding="utf-8")
        meta = request_summary_line(capture_id, seq, item)
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        item["request_file"] = str(request_file.relative_to(NETWORK_DIR))
        item["response_file"] = str(response_file.relative_to(NETWORK_DIR))
        item["meta_file"] = str(meta_file.relative_to(NETWORK_DIR))
        index_lines.append(request_summary_line(capture_id, seq, item))

    metric_view = build_metric_view(metric_requests)
    global_env_metrics_view = build_global_env_metrics_view(metric_requests)
    log_query_requests = [item for item in log_requests if item.get("query_type") in {None, "query_log"}]
    log_view = build_log_view(log_query_requests)

    files = {
        "metric_view_file": views_dir / "latest_metrics_compact.json",
        "global_env_metrics_view_file": views_dir / "global_env_metrics_compact.json",
        "log_view_file": views_dir / "latest_logs_compact.json",
        "page_metric_inventory_file": views_dir / "page_metric_inventory.json",
        "coverage_report_file": views_dir / "coverage_report.json",
    }
    files["metric_view_file"].write_text(json.dumps(metric_view, ensure_ascii=False, indent=2), encoding="utf-8")
    files["global_env_metrics_view_file"].write_text(
        json.dumps(global_env_metrics_view, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files["log_view_file"].write_text(json.dumps(log_view, ensure_ascii=False, indent=2), encoding="utf-8")
    files["page_metric_inventory_file"].write_text(
        json.dumps(page_inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files["coverage_report_file"].write_text(
        json.dumps(coverage_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    errors = summary_extra.get("errors", [])
    summary = {
        "capture_id": capture_id,
        "captured_at_local": dt.datetime.now().isoformat(timespec="seconds"),
        "capture_success": len(metric_requests) > 0 and not summary_extra.get("fatal_error"),
        "capture_mode": summary_extra.get("capture_mode"),
        "capture_source_used": summary_extra.get("capture_source_used"),
        "monitor_url": summary_extra.get("monitor_url"),
        "auto_refresh_target": summary_extra.get("auto_refresh_target"),
        "auto_refresh_enabled": summary_extra.get("auto_refresh_enabled", False),
        "page_group_count": page_inventory.get("group_count", 0),
        "page_metric_count": page_inventory.get("metric_count", 0),
        "metric_request_count": len(metric_requests),
        "metric_query_total": sum(item.get("query_count", 0) for item in metric_requests),
        "unique_query_name_count": len(unique_query_names(metric_requests)),
        "unique_result_id_count": len(unique_result_ids(metric_requests)),
        "log_request_count": len(log_requests),
        "log_query_request_count": len(log_query_requests),
        "log_stat_request_count": len([item for item in log_requests if item.get("query_type") == "stat_log"]),
        "log_entry_count": log_view.get("entry_count", 0),
        "errors": errors,
    }
    for key, path in files.items():
        summary[key] = str(path.relative_to(NETWORK_DIR))
    if summary_extra.get("har_meta"):
        summary["har_meta"] = summary_extra["har_meta"]

    (session_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with INDEX_JSONL.open("a", encoding="utf-8") as fh:
        for line in index_lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return session_dir


def capture_requests(args: argparse.Namespace, raw_dir: Path, errors: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    mode = args.capture_mode
    har_metric: list[dict[str, Any]] = []
    har_log: list[dict[str, Any]] = []
    network_metric: list[dict[str, Any]] = []
    network_log: list[dict[str, Any]] = []
    har_meta: dict[str, Any] = {}

    if mode in {"auto", "har", "both"}:
        try:
            har_metric, har_log, har_meta = collect_har_requests(
                args.session,
                args.session_name,
                raw_dir,
                args.overview_scroll_steps,
                args.scroll_px,
                args.dwell_ms,
                args.passes,
                args.logs_settle_ms,
                args.log_scroll_steps,
                errors,
            )
        except Exception as exc:  # noqa: BLE001 - auto mode should still try network fallback.
            errors.append(f"HAR capture failed: {exc}")

    if mode in {"auto", "network", "both"}:
        if mode != "auto" or not har_metric:
            try:
                network_metric, network_log = collect_network_requests(
                    args.session,
                    args.session_name,
                    args.logs_settle_ms,
                    args.log_scroll_steps,
                    args.scroll_px,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"network capture failed: {exc}")

    if mode == "har":
        return har_metric, har_log, "har", har_meta
    if mode == "network":
        return network_metric, network_log, "network", har_meta
    if mode == "both":
        return har_metric + network_metric, har_log + network_log, "both", har_meta
    if har_metric:
        return har_metric, har_log, "har", har_meta
    return network_metric, network_log, "network" if network_metric else "none", har_meta


def export_capture(args: argparse.Namespace) -> int:
    capture_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = SESSIONS_DIR / capture_id
    raw_dir = session_dir / "raw"
    errors: list[str] = []
    fatal_error = None
    monitor_url = None
    auto_refresh_enabled = False
    metric_requests: list[dict[str, Any]] = []
    log_requests: list[dict[str, Any]] = []
    capture_source_used = "none"
    har_meta: dict[str, Any] = {}
    page_inventory: dict[str, Any] = {"header": {}, "group_count": 0, "metric_count": 0, "groups": [], "raw_text": ""}

    try:
        switch_to_monitor_tab(args.session, args.session_name, args.monitor_url)
        monitor_url, _ = ensure_monitor_ready(args.session, args.session_name)
        auto_refresh_enabled = ensure_auto_refresh(
            args.session,
            args.session_name,
            args.auto_refresh_interval,
            errors,
        )
        if not auto_refresh_enabled:
            raise CaptureError(f"Could not enable auto refresh: {args.auto_refresh_interval}")
        prepare_dense_overview_view(
            args.session,
            args.session_name,
            zoom_out_steps=args.zoom_out_steps,
            settle_ms=args.zoom_settle_ms,
        )
        overview_body_text = get_body_text(args.session, args.session_name)
        page_inventory = build_page_metric_inventory(overview_body_text)
        metric_requests, log_requests, capture_source_used, har_meta = capture_requests(args, raw_dir, errors)
        if args.fail_on_empty_metrics and not metric_requests:
            raise CaptureError("No metric requests captured.")
    except Exception as exc:  # noqa: BLE001 - still write a failure session.
        fatal_error = str(exc)
        errors.append(fatal_error)

    coverage_report = build_coverage_report(page_inventory, metric_requests)
    session_dir = write_capture(
        capture_id,
        metric_requests,
        log_requests,
        page_inventory,
        coverage_report,
        {
            "capture_mode": args.capture_mode,
            "capture_source_used": capture_source_used,
            "monitor_url": monitor_url or args.monitor_url,
            "auto_refresh_target": args.auto_refresh_interval,
            "auto_refresh_enabled": auto_refresh_enabled,
            "errors": errors,
            "fatal_error": fatal_error,
            "har_meta": har_meta,
        },
    )

    print(session_dir)
    print(session_dir / "summary.json")
    print(session_dir / "views" / "latest_metrics_compact.json")
    print(session_dir / "views" / "page_metric_inventory.json")
    print(session_dir / "views" / "coverage_report.json")
    print(session_dir / "views" / "latest_logs_compact.json")
    return 1 if fatal_error else 0


def offline_preflight() -> int:
    """Check local prerequisites without opening or controlling a browser tab."""
    errors: list[str] = []
    if sys.version_info < (3, 9):
        errors.append(f"Python 3.9+ required, got {sys.version.split()[0]}")

    executable = shutil.which("agent-browser") or shutil.which("agent-browser.cmd")
    browser_version = "unavailable"
    if executable is None:
        errors.append("agent-browser executable is not available in PATH")
    else:
        try:
            result = subprocess.run(
                [executable, "--version"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            browser_version = (result.stdout or result.stderr).strip()
            if result.returncode != 0:
                errors.append(
                    f"agent-browser --version failed with exit code {result.returncode}"
                )
        except Exception as exc:  # noqa: BLE001 - preflight must report all failures.
            errors.append(f"agent-browser version check failed: {exc}")

    companion_files = (
        "collect_monitor_overview.py",
        "frontend_monitor.py",
        "manual_metric_recorder.py",
        "metric_probe.py",
        "postprocess_monitor_capture.py",
    )
    missing = [name for name in companion_files if not (ROOT / name).is_file()]
    if missing:
        errors.append(f"missing companion CLI files: {missing}")
    for name in companion_files:
        cli_path = ROOT / name
        if not cli_path.is_file():
            continue
        try:
            result = subprocess.run(
                [sys.executable, str(cli_path), "--help"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                errors.append(
                    f"{name} --help failed with exit code {result.returncode}"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name} --help failed: {exc}")

    fixture_path = ROOT / "tests" / "fixtures" / "sample.har"
    try:
        fixture_metrics, fixture_logs = parse_har_entries(fixture_path)
        if not fixture_metrics or not fixture_logs:
            errors.append(
                "sample HAR fixture must contain metric and log requests"
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sample HAR fixture parse failed: {exc}")

    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="arena-monitor-preflight-", dir=RUNTIME_DIR, delete=True
        ) as stream:
            stream.write(b"ok")
            stream.flush()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"runtime directory is not writable: {RUNTIME_DIR}: {exc}")

    print(f"Python: {sys.version.split()[0]}")
    print(f"agent-browser: {executable or 'missing'}")
    print(f"agent-browser version: {browser_version or 'unknown'}")
    print(f"companion CLI checks: {len(companion_files) - len(missing)}")
    print(f"HAR fixture: {fixture_path}")
    print(f"runtime directory: {RUNTIME_DIR}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("Static preflight failed. No browser tab was opened or modified.")
        return 1
    print("Static preflight passed. No browser tab was opened or modified.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Tencent Arena frontend API data")
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--session-name", default=DEFAULT_SESSION_NAME)
    parser.add_argument("--monitor-url", default=None)
    parser.add_argument("--capture-mode", choices=["auto", "har", "network", "both"], default="auto")
    parser.add_argument("--auto-refresh-interval", default=DEFAULT_AUTO_REFRESH)
    parser.add_argument("--overview-scroll-steps", type=int, default=14)
    parser.add_argument("--log-scroll-steps", type=int, default=3)
    parser.add_argument("--scroll-px", type=int, default=700)
    parser.add_argument("--dwell-ms", type=int, default=8000)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--logs-settle-ms", type=int, default=4000)
    parser.add_argument("--zoom-out-steps", type=int, default=4)
    parser.add_argument("--zoom-settle-ms", type=int, default=6000)
    parser.add_argument("--fail-on-empty-metrics", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run offline dependency/output checks without controlling a browser tab",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.check:
        return offline_preflight()
    return export_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
