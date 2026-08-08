#!/usr/bin/env python3
"""Record Tencent Arena monitor metric curves while the user scrolls manually.

This collector intentionally does not click, expand, or scroll monitor cards.
Open the target monitor page, start this script, then manually expand groups,
scroll, and wait for refresh.  The script watches frontend
``GetTrainMetricRange`` network responses, prints each newly recorded curve
name, and writes the raw request plus compact series data to local JSONL files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from browser_auth import agent_browser_command
from browser_auth import default_session_from_env, default_session_name_from_env


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = Path(
    os.environ.get("ARENA_MONITOR_RUNTIME_DIR", ROOT / "runtime")
).expanduser().resolve() / "manual_metric_recorder"
DEFAULT_SESSION = default_session_from_env()
DEFAULT_SESSION_NAME = default_session_name_from_env(DEFAULT_SESSION)
API_METRIC = "GetTrainMetricRange"
MONITOR_URL_HINT = "/p/v5/exp/monitor"


def decode_output(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", sys.getdefaultencoding()):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def run_agent_browser(args: list[str], session: str, session_name: str, timeout: int = 30) -> str:
    command, env = agent_browser_command(args, session, session_name)
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        stdout = decode_output(proc.stdout or b"").strip()
        stderr = decode_output(proc.stderr or b"").strip()
        raise RuntimeError(stderr or stdout or f"agent-browser exited with {proc.returncode}")
    return decode_output(proc.stdout).strip()


def parse_json_or_none(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def metric_requests(session: str, session_name: str) -> list[dict[str, Any]]:
    raw = run_agent_browser(
        ["network", "requests", "--filter", API_METRIC, "--method", "POST", "--json"],
        session,
        session_name,
        timeout=30,
    )
    payload = json.loads(raw)
    return payload.get("data", {}).get("requests", [])


def request_detail(request_id: str, session: str, session_name: str) -> dict[str, Any]:
    raw = run_agent_browser(["network", "request", request_id, "--json"], session, session_name, timeout=30)
    return json.loads(raw).get("data", {})


def current_url(session: str, session_name: str) -> str:
    return run_agent_browser(["get", "url"], session, session_name, timeout=15)


def open_or_check_monitor(args: argparse.Namespace) -> None:
    if args.monitor_url:
        run_agent_browser(["open", args.monitor_url], args.session, args.session_name, timeout=45)
        run_agent_browser(["wait", str(args.open_wait_ms)], args.session, args.session_name, timeout=20)
        return
    url = current_url(args.session, args.session_name)
    if MONITOR_URL_HINT not in url:
        raise RuntimeError(
            f"current tab is not a Tencent Arena monitor page: {url}; "
            "pass --monitor-url or switch the browser tab manually"
        )


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    items = result.get("items") or []
    point_count = 0
    first = None
    latest = None
    for item in items:
        values = item.get("values") or []
        point_count += len(values)
        if values and first is None:
            first = values[0]
        if values:
            latest = values[-1]
    return {
        "id": result.get("id"),
        "item_count": len(items),
        "point_count": point_count,
        "first_timestamp": first.get("timestamp") if isinstance(first, dict) else None,
        "first_value": first.get("value") if isinstance(first, dict) else None,
        "latest_timestamp": latest.get("timestamp") if isinstance(latest, dict) else None,
        "latest_value": latest.get("value") if isinstance(latest, dict) else None,
    }


def sortable_timestamp(value: Any) -> tuple[int, str]:
    text = "" if value is None else str(value)
    try:
        return int(text), text
    except ValueError:
        return -1, text


def compact_detail(detail: dict[str, Any]) -> dict[str, Any]:
    post_data = parse_json_or_none(detail.get("postData")) or {}
    response_json = parse_json_or_none(detail.get("responseBody")) or {}
    queries = post_data.get("queries") or []
    results = ((response_json.get("data") or {}).get("results") or []) if isinstance(response_json, dict) else []
    return {
        "request_id": detail.get("requestId") or detail.get("id"),
        "timestamp": detail.get("timestamp"),
        "url": detail.get("url"),
        "status": detail.get("status"),
        "queries": [
            {
                "name": query.get("name"),
                "id": query.get("id"),
                "expr": query.get("expr"),
                "step": query.get("step"),
            }
            for query in queries
        ],
        "query_names": [query.get("name") for query in queries if query.get("name")],
        "query_count": len(queries),
        "results": [summarize_result(result) for result in results],
        "post_data": post_data,
        "response_json": response_json,
    }


def safe_file_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return stem[:120] or "metric"


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_new_points(
    out_dir: Path,
    query_name: str,
    raw_result: dict[str, Any] | None,
    seen_points: dict[str, set[str]],
) -> int:
    if not isinstance(raw_result, dict):
        return 0
    metric_seen = seen_points.setdefault(query_name, set())
    new_count = 0
    for item_index, item in enumerate(raw_result.get("items") or []):
        if not isinstance(item, dict):
            continue
        labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
        labels_key = json.dumps(labels, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for value in item.get("values") or []:
            if not isinstance(value, dict):
                continue
            timestamp = value.get("timestamp")
            key = f"{item_index}|{labels_key}|{timestamp}"
            if key in metric_seen:
                continue
            metric_seen.add(key)
            new_count += 1
            append_jsonl(
                out_dir / "points" / f"{safe_file_stem(query_name)}.jsonl",
                {
                    "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "name": query_name,
                    "item_index": item_index,
                    "labels": labels,
                    "timestamp": timestamp,
                    "value": value.get("value"),
                },
            )
    return new_count


def write_series_files(
    out_dir: Path,
    detail: dict[str, Any],
    metric_state: dict[str, dict[str, Any]],
    seen_points: dict[str, set[str]],
) -> list[str]:
    printed = []
    result_by_id = {str(result.get("id")): result for result in detail.get("results", [])}
    raw_results = (((detail.get("response_json") or {}).get("data") or {}).get("results") or [])
    raw_by_id = {str(result.get("id")): result for result in raw_results if isinstance(result, dict)}
    raw_by_index = [result for result in raw_results if isinstance(result, dict)]
    result_by_index = detail.get("results", [])
    for index, query in enumerate(detail.get("queries") or []):
        query_name = query.get("name")
        if not query_name:
            continue
        query_id = query.get("id")
        result = result_by_id.get(str(query_id)) or result_by_id.get(query_name)
        raw_result = raw_by_id.get(str(query_id)) or raw_by_id.get(query_name)
        if result is None and index < len(result_by_index):
            result = result_by_index[index]
        if raw_result is None and index < len(raw_by_index):
            raw_result = raw_by_index[index]
        if result is None:
            result = {
                "id": query_id or query_name,
                "item_count": 0,
                "point_count": 0,
                "first_timestamp": None,
                "first_value": None,
                "latest_timestamp": None,
                "latest_value": None,
            }

        point_count = int(result.get("point_count") or 0)
        latest_timestamp = result.get("latest_timestamp")
        latest_value = result.get("latest_value")
        new_point_count = write_new_points(out_dir, query_name, raw_result, seen_points)
        previous = metric_state.get(query_name)
        latest_key = sortable_timestamp(latest_timestamp)
        previous_key = sortable_timestamp(previous.get("latest_timestamp")) if previous else (-1, "")
        first_seen = previous is None
        has_newer_timestamp = latest_key > previous_key
        has_more_points = bool(previous) and point_count > int(previous.get("point_count") or 0)
        should_print = (
            (first_seen and (point_count > 0 or latest_timestamp is not None))
            or has_newer_timestamp
            or has_more_points
            or new_point_count > 0
        )
        if first_seen and not should_print:
            metric_state[query_name] = {
                "point_count": point_count,
                "latest_timestamp": latest_timestamp,
                "latest_value": latest_value,
                "unique_point_count": len(seen_points.get(query_name, set())),
                "update_count": 0,
                "last_request_id": detail.get("request_id"),
            }
        if should_print:
            update_count = 1 if first_seen else int(previous.get("update_count") or 1) + 1
            metric_state[query_name] = {
                "point_count": point_count,
                "latest_timestamp": latest_timestamp,
                "latest_value": latest_value,
                "unique_point_count": len(seen_points.get(query_name, set())),
                "update_count": update_count,
                "last_request_id": detail.get("request_id"),
            }
            status = "new" if first_seen else "update"
            delta_points = point_count if first_seen else point_count - int(previous.get("point_count") or 0)
            print(
                f"[metric:{status}] {query_name} points={point_count} "
                f"delta={delta_points} new_points={new_point_count} "
                f"latest_ts={latest_timestamp} latest={latest_value}",
                flush=True,
            )
            printed.append(query_name)
        append_jsonl(
            out_dir / "series" / f"{safe_file_stem(query_name)}.jsonl",
            {
                "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
                "request_id": detail.get("request_id"),
                "name": query_name,
                "summary": result,
                "new_point_count": new_point_count,
                "metric_state": metric_state.get(query_name),
                "raw_result": raw_result,
            },
        )
    return printed


def main() -> int:
    parser = argparse.ArgumentParser(description="Manually scroll Tencent Arena monitor and record metric data.")
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--session-name", default=DEFAULT_SESSION_NAME)
    parser.add_argument("--monitor-url")
    parser.add_argument("--poll-interval", type=float, default=1.5)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means until Ctrl-C.")
    parser.add_argument("--clear-network", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-wait-ms", type=int, default=2500)
    parser.add_argument("--heartbeat-interval", type=float, default=10.0)
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else RUNTIME_DIR / "sessions" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # When opening a fresh monitor URL, clear the old network log before the
    # navigation.  Clearing after navigation would drop the initial metric
    # requests and make short live-run tests look empty.
    if args.clear_network and args.monitor_url:
        run_agent_browser(["network", "requests", "--clear"], args.session, args.session_name, timeout=20)
    open_or_check_monitor(args)
    if args.clear_network and not args.monitor_url:
        run_agent_browser(["network", "requests", "--clear"], args.session, args.session_name, timeout=20)

    print(f"[manual_metric_recorder] output={out_dir}", flush=True)
    print("[manual_metric_recorder] 手动展开/滚动监控卡片；脚本会打印新曲线英文名。按 Ctrl-C 结束。", flush=True)

    seen_requests: set[str] = set()
    metric_state: dict[str, dict[str, Any]] = {}
    seen_points: dict[str, set[str]] = {}
    metric_update_count = 0
    errors: list[str] = []
    start = time.monotonic()
    last_heartbeat = start
    try:
        while True:
            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break
            try:
                requests = metric_requests(args.session, args.session_name)
            except Exception as exc:  # noqa: BLE001 - recorder should keep running where possible.
                errors.append(str(exc))
                time.sleep(args.poll_interval)
                continue
            new_request_count = 0
            for request in requests:
                request_id = request.get("requestId") or request.get("id")
                if not request_id or request_id in seen_requests:
                    continue
                seen_requests.add(request_id)
                new_request_count += 1
                try:
                    detail = compact_detail(request_detail(request_id, args.session, args.session_name))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{request_id}: {exc}")
                    continue
                append_jsonl(out_dir / "metric_requests.jsonl", detail)
                printed = write_series_files(out_dir, detail, metric_state, seen_points)
                metric_update_count += len(printed)
            now = time.monotonic()
            if args.heartbeat_interval > 0 and now - last_heartbeat >= args.heartbeat_interval:
                print(
                    f"[manual_metric_recorder] heartbeat requests={len(seen_requests)} "
                    f"metrics={len(metric_state)} updates={metric_update_count} "
                    f"last_new_requests={new_request_count}",
                    flush=True,
                )
                last_heartbeat = now
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n[manual_metric_recorder] stopped by user", flush=True)

    summary = {
        "output": str(out_dir),
        "request_count": len(seen_requests),
        "unique_metric_count": len(metric_state),
        "metric_update_count": metric_update_count,
        "unique_point_count": sum(len(points) for points in seen_points.values()),
        "metric_names": sorted(metric_state),
        "metric_state": metric_state,
        "errors": errors,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[manual_metric_recorder] requests={len(seen_requests)} "
        f"metrics={len(metric_state)} updates={metric_update_count}",
        flush=True,
    )
    print(f"[manual_metric_recorder] output={out_dir}", flush=True)
    print(f"[manual_metric_recorder] summary={out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
