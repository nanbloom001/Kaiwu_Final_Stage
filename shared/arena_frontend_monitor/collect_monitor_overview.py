#!/usr/bin/env python3
"""Collect Tencent Arena monitor overview data with lazy loading handled.

This script is intentionally browser-driven. Tencent Arena only requests many
overview cards after their group is expanded and brought into view. The script:

1. reuses an existing logged-in agent-browser session;
2. enables "每 5 秒自动刷新";
3. collapses overview groups, then expands each group one by one;
4. clicks every "查看" trigger exposed under each group so lazy cards request data;
5. reads GetTrainMetricRange response bodies through agent-browser request
   details, because HAR files may omit response text;
6. reads the training log tab and extracts important lines.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from browser_auth import agent_browser_env


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = Path(
    os.environ.get("ARENA_MONITOR_RUNTIME_DIR", ROOT / "runtime")
).expanduser().resolve() / "overview_capture"
SESSIONS_DIR = RUNTIME_DIR / "sessions"

DEFAULT_SESSION = os.environ.get("AGENT_BROWSER_SESSION", "tencent-arena")
DEFAULT_SESSION_NAME = os.environ.get("AGENT_BROWSER_SESSION_NAME", DEFAULT_SESSION)

MONITOR_URL_HINT = "/p/v5/exp/monitor"
OVERVIEW_TAB = "监控总览"
LOGS_TAB = "训练日志"
DISABLE_AUTO_REFRESH = "禁用自动刷新"
DEFAULT_AUTO_REFRESH = "每 5 秒自动刷新"
API_METRIC = "GetTrainMetricRange"
API_LOG = "GetTrainLog"

GROUP_LINE_RE = re.compile(
    r'- button "([^"]+\(\s*(\d+)\s*\))" \[expanded=(true|false), ref=(e\d+)\]'
)
GROUP_NAME_RE = re.compile(r"^(?P<name>.+?)\(\s*(?P<count>\d+)\s*\)$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
TASK_STATUS_LABELS = ("进行中", "自动释放", "已完成", "成功", "失败", "已停止", "停止", "超时", "异常", "已终止", "终止")
RUNNING_STATUS_LABELS = {"进行中"}

DEFAULT_IMPORTANT_LOG_KEYWORDS = (
    "ERROR",
    "WARNING",
    "Traceback",
    "Exception",
    "MonitorDebug",
    "VelocityCurriculumDebug",
    "EnvMonitor",
    "Episode",
    "terrain",
    "mode",
    "reward",
    "tracking",
    "curriculum",
)

GROUP_QUERY_ALLOWLIST: dict[str, tuple[tuple[str, ...], ...]] = {
    "基础指标": (
        ("episode_cnt",),
        ("load_model_succ_cnt",),
        ("predict_succ_cnt",),
        ("sample_production_and_consumption_ratio",),
        ("sample_receive_cnt",),
        ("train_global_step",),
    ),
    "硬件指标": (
        ("aisrv_cpu_usage", "learner_cpu_usage"),
        ("gpu_usage",),
        ("gpu_mem_usage",),
        ("ram_usage",),
    ),
    "全局环境指标": (
        ("abnormal_count", "completed_count", "timeout_count"),
        ("energy_score", "forward_score", "pose_score", "step_score", "total_score"),
        ("step_avg",),
    ),
    "Reward指标": (
        ("reward_mean", "reward_std"),
        ("reward_track_lin_vel_xy", "reward_track_ang_vel_z"),
    ),
    "稳定接触": (
        ("reward_lin_vel_z",),
        ("reward_undesired_contacts",),
        ("reward_termination",),
        ("reward_dof_pos_limits",),
    ),
    "关节动作平滑": (
        ("reward_joint_acc",),
        ("reward_action_rate",),
        ("reward_action_smoothness",),
    ),
    "地形高度剖面": (
        ("terrain_passability",),
        ("terrain_wall_score",),
        ("terrain_slope_stair_score",),
        ("profile_total_delta",),
        ("profile_max_up_slope",),
        ("profile_peak_drop",),
        ("profile_roughness",),
        ("profile_monotonic_score",),
    ),
}

GROUP_COUNT_MODE: dict[str, str] = {
    "Reward指标": "results",
}


class CaptureError(RuntimeError):
    pass


def decode_output(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", sys.getdefaultencoding()):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def run_agent_browser(args: list[str], session: str, session_name: str, timeout: int = 30) -> str:
    env = agent_browser_env(session, session_name)
    executable = shutil.which("agent-browser") or shutil.which("agent-browser.cmd") or "agent-browser"
    proc = subprocess.run(
        [executable] + args,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        stdout = decode_output(proc.stdout or b"").strip()
        stderr = decode_output(proc.stderr or b"").strip()
        raise CaptureError(stderr or stdout or f"agent-browser {' '.join(args)} failed")
    return decode_output(proc.stdout).strip()


def run_agent_browser_safe(
    args: list[str],
    session: str,
    session_name: str,
    timeout: int = 30,
) -> tuple[bool, str]:
    try:
        return True, run_agent_browser(args, session, session_name, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - CLI should preserve partial captures.
        return False, str(exc)


def wait_ms(ms: int, session: str, session_name: str) -> None:
    time.sleep(ms / 1000)


def click_text(text: str, session: str, session_name: str, exact: bool = True) -> None:
    args = ["find", "text", text, "click"]
    if exact:
        args.append("--exact")
    run_agent_browser(args, session, session_name, timeout=20)


def click_ref(ref: str, session: str, session_name: str) -> None:
    run_agent_browser(["click", f"@{ref}"], session, session_name, timeout=20)


def press_key(key: str, session: str, session_name: str) -> None:
    run_agent_browser(["press", key], session, session_name, timeout=15)


def eval_js(script: str, session: str, session_name: str) -> str:
    return run_agent_browser(["eval", script], session, session_name, timeout=20)


def scroll_up(px: int, session: str, session_name: str) -> None:
    run_agent_browser(["scroll", "up", str(px)], session, session_name, timeout=15)


def scroll_down(px: int, session: str, session_name: str) -> None:
    run_agent_browser(["scroll", "down", str(px)], session, session_name, timeout=15)


def dom_scroll_monitor(args: argparse.Namespace, dy: int) -> tuple[bool, str]:
    script = f"""
    (() => {{
      const dy = {int(dy)};
      const tabpanel = document.querySelector('[role="tabpanel"]');
      const elements = [...document.querySelectorAll('*')];
      const candidates = elements.filter(el => {{
        const style = getComputedStyle(el);
        const overflow = `${{style.overflow}} ${{style.overflowY}}`;
        const rect = el.getBoundingClientRect();
        return el.scrollHeight > el.clientHeight + 20
          && rect.height > 120
          && rect.width > 300
          && /(auto|scroll|overlay)/.test(overflow);
      }});
      let best = null;
      for (const el of candidates) {{
        const rect = el.getBoundingClientRect();
        const containsPanel = tabpanel ? (el === tabpanel || el.contains(tabpanel) || tabpanel.contains(el)) : true;
        const score = (containsPanel ? 100000 : 0) + rect.height * 10 + rect.width + Math.min(el.scrollHeight - el.clientHeight, 5000);
        if (!best || score > best.score) best = {{el, score, rect}};
      }}
      const target = best?.el || document.scrollingElement || document.documentElement || document.body;
      const before = target.scrollTop || 0;
      target.scrollTop = Math.max(0, Math.min(before + dy, target.scrollHeight - target.clientHeight));
      if (Math.abs((target.scrollTop || 0) - before) < 1) window.scrollBy(0, dy);
      const after = target.scrollTop || 0;
      return {{
        ok: Math.abs(after - before) >= 1,
        before,
        after,
        dy,
        tag: target.tagName,
        className: String(target.className || '').slice(0, 120),
        maxScroll: Math.max(0, target.scrollHeight - target.clientHeight)
      }};
    }})()
    """
    return run_agent_browser_safe(["eval", script], args.session, args.session_name, timeout=20)


def drag_page_scrollbar_down(args: argparse.Namespace, step_index: int) -> None:
    session = args.session
    session_name = args.session_name
    ok, _ = dom_scroll_monitor(args, args.scrollbar_drag_dy)
    if ok:
        return
    start_y = args.scrollbar_start_y + step_index * args.scrollbar_step_y
    end_y = start_y + args.scrollbar_drag_dy
    run_agent_browser(["mouse", "move", str(args.scrollbar_x), str(start_y)], session, session_name, timeout=15)
    run_agent_browser(["mouse", "down"], session, session_name, timeout=15)
    run_agent_browser(["mouse", "move", str(args.scrollbar_x), str(end_y)], session, session_name, timeout=15)
    run_agent_browser(["mouse", "up"], session, session_name, timeout=15)


def drag_group_scrollbar_once(args: argparse.Namespace, drag_index: int, declared_count: int) -> tuple[bool, str]:
    session = args.session
    session_name = args.session_name
    drag_dy = args.scrollbar_drag_dy
    if declared_count <= args.small_group_threshold:
        drag_dy = min(drag_dy, args.small_group_scrollbar_drag_dy)
    ok, output = dom_scroll_monitor(args, drag_dy)
    if ok:
        return True, output
    ok, output = run_agent_browser_safe(
        ["mouse", "move", str(args.scrollbar_x), str(args.scrollbar_start_y + drag_index * args.scrollbar_step_y)],
        session,
        session_name,
        timeout=15,
    )
    if ok:
        ok, output = run_agent_browser_safe(["mouse", "down"], session, session_name, timeout=15)
    if ok:
        ok, output = run_agent_browser_safe(
            [
                "mouse",
                "move",
                str(args.scrollbar_x),
                str(args.scrollbar_start_y + drag_index * args.scrollbar_step_y + drag_dy),
            ],
            session,
            session_name,
            timeout=15,
        )
    if ok:
        ok, output = run_agent_browser_safe(["mouse", "up"], session, session_name, timeout=15)
    else:
        run_agent_browser_safe(["mouse", "up"], session, session_name, timeout=15)
    return ok, output


def drag_page_scrollbar_to_top(args: argparse.Namespace) -> None:
    session = args.session
    session_name = args.session_name
    script = """
    (() => {
      const roots = [document.scrollingElement, document.documentElement, document.body, ...document.querySelectorAll('*')];
      let changed = 0;
      for (const el of roots) {
        if (!el) continue;
        try {
          if (el.scrollTop) changed += 1;
          el.scrollTop = 0;
        } catch {}
      }
      window.scrollTo(0, 0);
      return {ok:true, changed};
    })()
    """
    ok, output = run_agent_browser_safe(["eval", script], session, session_name, timeout=20)
    if not ok:
        run_agent_browser(["mouse", "move", str(args.scrollbar_x), str(args.scrollbar_top_drag_start_y)], session, session_name, timeout=15)
        run_agent_browser(["mouse", "down"], session, session_name, timeout=15)
        run_agent_browser(["mouse", "move", str(args.scrollbar_x), str(args.scrollbar_top_y)], session, session_name, timeout=15)
        run_agent_browser(["mouse", "up"], session, session_name, timeout=15)


def set_page_zoom(args: argparse.Namespace, errors: list[str]) -> None:
    if args.page_zoom <= 0:
        return
    script = (
        "(() => {"
        f"const zoom = '{args.page_zoom}';"
        "document.documentElement.style.zoom = zoom;"
        "document.body.style.zoom = zoom;"
        "return {html: document.documentElement.style.zoom, body: document.body.style.zoom};"
        "})()"
    )
    ok, output = run_agent_browser_safe(["eval", script], args.session, args.session_name, timeout=20)
    if not ok:
        errors.append(f"set page zoom failed: {output}")
    wait_ms(args.zoom_wait_ms, args.session, args.session_name)


def get_body_text(session: str, session_name: str) -> str:
    return run_agent_browser(["get", "text", "body"], session, session_name, timeout=30)


def get_snapshot(session: str, session_name: str) -> dict[str, Any]:
    raw = run_agent_browser(["snapshot", "-i", "--json"], session, session_name, timeout=45)
    return json.loads(raw)


def visible_text_tail(session: str, session_name: str, line_count: int = 18) -> list[str]:
    lines = [line.strip() for line in get_body_text(session, session_name).splitlines() if line.strip()]
    return lines[-line_count:]


def list_tabs(session: str, session_name: str) -> str:
    return run_agent_browser(["tab"], session, session_name, timeout=15)


def switch_to_monitor_tab(
    session: str,
    session_name: str,
    monitor_url: str | None,
    allow_monitor_tab_fallback: bool,
) -> str:
    if monitor_url:
        tabs = list_tabs(session, session_name)
        for line in tabs.splitlines():
            if monitor_url in line:
                tab = re.search(r"\[(t\d+)\]", line)
                if tab:
                    run_agent_browser(["tab", tab.group(1)], session, session_name, timeout=15)
                    return monitor_url
        run_agent_browser(["open", monitor_url], session, session_name, timeout=45)
        wait_ms(2500, session, session_name)
        return monitor_url

    current_url = run_agent_browser(["get", "url"], session, session_name, timeout=15).strip()
    if MONITOR_URL_HINT in current_url:
        return current_url

    if not allow_monitor_tab_fallback:
        raise CaptureError(
            "The active browser tab is not a Tencent Arena monitor page. "
            "Switch to the intended monitor tab first, or pass --monitor-url explicitly."
        )

    tabs = list_tabs(session, session_name)
    for line in tabs.splitlines():
        if MONITOR_URL_HINT in line:
            tab = re.search(r"\[(t\d+)\]", line)
            if tab:
                run_agent_browser(["tab", tab.group(1)], session, session_name, timeout=15)
                return line.rsplit(" - ", 1)[-1].strip()
    raise CaptureError("No Tencent Arena monitor tab found. Open the monitor page or pass --monitor-url.")


def auto_refresh_label_selected(session: str, session_name: str, target_label: str) -> bool:
    script = f"""
    (() => {{
      const target = {json.dumps(target_label)};
      const labels = [...document.querySelectorAll('.ant-select-selection-item')]
        .map(el => (el.textContent || '').trim())
        .filter(Boolean);
      return {{ok: labels.some(label => label.includes(target)), labels}};
    }})()
    """
    ok, output = run_agent_browser_safe(["eval", script], session, session_name, timeout=10)
    if not ok:
        return False
    try:
        result = json.loads(output)
    except json.JSONDecodeError:
        return target_label in output
    return bool(result.get("ok"))


def ensure_auto_refresh(session: str, session_name: str, target_label: str, errors: list[str]) -> bool:
    if auto_refresh_label_selected(session, session_name, target_label):
        return True

    attempt_errors: list[str] = []
    for attempt in range(1, 4):
        body = get_body_text(session, session_name)
        opener_label = target_label if target_label in body else DISABLE_AUTO_REFRESH
        ok, output = run_agent_browser_safe(["find", "text", opener_label, "click", "--exact"], session, session_name, timeout=20)
        if ok:
            wait_ms(1200, session, session_name)
        else:
            attempt_errors.append(f"auto-refresh open attempt {attempt} failed for {opener_label}: {output}")

        ok, output = run_agent_browser_safe(
            ["find", "text", target_label, "click", "--exact"],
            session,
            session_name,
            timeout=20,
        )
        if ok:
            wait_ms(1200, session, session_name)
            if auto_refresh_label_selected(session, session_name, target_label):
                wait_ms(5500, session, session_name)
                return True
            attempt_errors.append(f"auto-refresh select attempt {attempt} did not confirm selected label {target_label}")
        else:
            attempt_errors.append(f"auto-refresh select attempt {attempt} failed: {output}")
        wait_ms(1200, session, session_name)

    script = f"""
    (async () => {{
      const target = {json.dumps(target_label)};
      const fire = (el, type) => el && el.dispatchEvent(new MouseEvent(type, {{bubbles:true, cancelable:true, view:window}}));
      const select = [...document.querySelectorAll('.ant-select')].find(el => (el.innerText || '').includes('自动刷新'));
      if (!select) return {{ok:false, reason:'no_ant_select'}};
      fire(select, 'mousedown'); fire(select, 'mouseup'); fire(select, 'click');
      await new Promise(resolve => setTimeout(resolve, 250));
      const option = [...document.querySelectorAll('.ant-select-item-option')].find(el => ((el.getAttribute('title') || el.innerText || '') === target));
      if (!option) return {{ok:false, reason:'no_target_option'}};
      fire(option, 'mousedown'); fire(option, 'mouseup'); fire(option, 'click');
      await new Promise(resolve => setTimeout(resolve, 500));
      const selected = document.querySelector('.ant-select-selection-item')?.textContent || '';
      return {{ok:selected.includes(target), selected}};
    }})()
    """
    ok, output = run_agent_browser_safe(["eval", script], session, session_name, timeout=20)
    if ok:
        try:
            result = json.loads(output)
        except json.JSONDecodeError:
            result = {"ok": target_label in output, "raw": output}
        if result.get("ok"):
            wait_ms(5500, session, session_name)
            return True
        attempt_errors.append(f"auto-refresh DOM fallback failed: {result}")
    else:
        attempt_errors.append(f"auto-refresh DOM fallback error: {output}")
    errors.extend(attempt_errors)
    return False


def parse_groups_from_snapshot(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    text = snapshot.get("data", {}).get("snapshot", "")
    groups: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = GROUP_LINE_RE.search(line)
        if not match:
            continue
        label = match.group(1)
        name_match = GROUP_NAME_RE.match(label)
        groups.append(
            {
                "label": label,
                "name": name_match.group("name").strip() if name_match else label,
                "declared_count": int(match.group(2)),
                "expanded": match.group(3) == "true",
                "ref": match.group(4),
            }
        )
    return groups


def group_view_refs_from_snapshot(snapshot: dict[str, Any], group_ref: str) -> list[str]:
    lines = snapshot.get("data", {}).get("snapshot", "").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if f"ref={group_ref}" in line and "- button " in line:
            start = idx
            break
    if start is None:
        return []

    refs = []
    for line in lines[start + 1 :]:
        if "- button " in line and "expanded=" in line:
            break
        match = re.search(r'- generic "查看" \[ref=(e\d+)\]', line)
        if match:
            refs.append(match.group(1))
    return refs


def parse_page_group_inventory(body_text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in body_text.splitlines() if line.strip()]
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        group_match = GROUP_NAME_RE.match(line)
        if group_match:
            current = {
                "name": group_match.group("name").strip(),
                "declared_count": int(group_match.group("count")),
                "cards": [],
            }
            groups.append(current)
            idx += 1
            continue
        if current is not None and idx + 1 < len(lines) and lines[idx + 1] == "查看":
            card = {"name": line, "page_status": "unknown"}
            if idx + 2 < len(lines) and lines[idx + 2] == "暂无数据":
                card["page_status"] = "no_data"
            current["cards"].append(card)
        idx += 1
    return groups


def detect_task_status(body_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in body_text.splitlines() if line.strip()]
    status = None
    for idx, line in enumerate(lines):
        if line == "状态" and idx + 1 < len(lines):
            status = lines[idx + 1]
            break
    if status is None:
        for label in TASK_STATUS_LABELS:
            if label in lines:
                status = label
                break
    is_running = status in RUNNING_STATUS_LABELS
    return {
        "status": status,
        "is_running": is_running,
        "detected": status is not None,
        "mode": "running" if is_running else "history",
    }


def clear_network(session: str, session_name: str) -> None:
    run_agent_browser(["network", "requests", "--clear"], session, session_name, timeout=20)


def list_metric_requests(session: str, session_name: str) -> list[dict[str, Any]]:
    raw = run_agent_browser(
        ["network", "requests", "--filter", API_METRIC, "--method", "POST", "--json"],
        session,
        session_name,
        timeout=30,
    )
    payload = json.loads(raw)
    return payload.get("data", {}).get("requests", [])


def metric_request_signatures(requests: list[dict[str, Any]]) -> set[str]:
    signatures = set()
    for request in requests:
        query_names = request_query_names(request)
        if query_names:
            signatures.add(request_signature_from_names(query_names))
    return signatures


def metric_request_signatures_from_details(details: list[dict[str, Any]]) -> set[str]:
    signatures = set()
    for detail in details:
        query_names = detail.get("query_names", [])
        if query_names:
            signatures.add(request_signature_from_names(query_names))
    return signatures


def effective_card_count(group_name: str, details: list[dict[str, Any]]) -> int:
    mode = GROUP_COUNT_MODE.get(group_name)
    if mode == "details":
        return len(details)
    if mode == "results":
        counts_by_signature = {}
        for detail in details:
            signature = request_signature_from_names(detail.get("query_names", []))
            counts_by_signature[signature] = max(counts_by_signature.get(signature, 0), len(detail.get("results", [])) or 1)
        return sum(counts_by_signature.values())
    return len(metric_request_signatures_from_details(details))


def latest_result_timestamp_ms(details: list[dict[str, Any]]) -> int | None:
    latest = None
    for detail in details:
        for result in detail.get("results", []):
            ts = result.get("latest_timestamp")
            if ts is None:
                continue
            try:
                value = int(float(ts))
            except (TypeError, ValueError):
                continue
            latest = value if latest is None else max(latest, value)
    return latest


def timestamp_validation(
    details: list[dict[str, Any]],
    baseline_latest_ms: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    latest_ms = latest_result_timestamp_ms(details)
    now_ms = int(time.time() * 1000)
    if latest_ms is None:
        return {"ok": False, "latest_timestamp_ms": None, "reason": "no_latest_timestamp"}
    if args.history_mode:
        if baseline_latest_ms is None:
            return {"ok": True, "latest_timestamp_ms": latest_ms, "mode": "history", "delta_from_baseline_ms": 0}
        delta = abs(latest_ms - baseline_latest_ms)
        return {
            "ok": delta <= args.history_timestamp_tolerance_ms,
            "latest_timestamp_ms": latest_ms,
            "mode": "history",
            "delta_from_baseline_ms": delta,
            "tolerance_ms": args.history_timestamp_tolerance_ms,
        }
    age = now_ms - latest_ms
    return {
        "ok": age <= args.running_timestamp_max_age_ms,
        "latest_timestamp_ms": latest_ms,
        "mode": "running",
        "age_ms": age,
        "max_age_ms": args.running_timestamp_max_age_ms,
    }


def locate_group_by_dragging(group: dict[str, Any], args: argparse.Namespace, errors: list[str]) -> dict[str, Any] | None:
    session = args.session
    session_name = args.session_name
    for attempt in range(args.max_group_locate_drags + 1):
        current = find_group(session, session_name, group["name"])
        if current:
            return current
        if attempt >= args.max_group_locate_drags:
            break
        ok, output = run_agent_browser_safe(
            ["mouse", "move", str(args.scrollbar_x), str(args.locate_scrollbar_start_y + attempt * args.locate_scrollbar_step_y)],
            session,
            session_name,
            timeout=15,
        )
        if ok:
            ok, output = run_agent_browser_safe(["mouse", "down"], session, session_name, timeout=15)
        if ok:
            ok, output = run_agent_browser_safe(
                [
                    "mouse",
                    "move",
                    str(args.scrollbar_x),
                    str(args.locate_scrollbar_start_y + attempt * args.locate_scrollbar_step_y + args.locate_scrollbar_drag_dy),
                ],
                session,
                session_name,
                timeout=15,
            )
        if ok:
            run_agent_browser_safe(["mouse", "up"], session, session_name, timeout=15)
        else:
            run_agent_browser_safe(["mouse", "up"], session, session_name, timeout=15)
            errors.append(f"locate group {group['name']} drag {attempt + 1} failed: {output}")
        wait_ms(args.locate_wait_ms, session, session_name)
    return None


def wait_for_overview_refresh(
    group_name: str,
    args: argparse.Namespace,
    errors: list[str],
) -> dict[str, Any]:
    session = args.session
    session_name = args.session_name
    clear_network(session, session_name)
    attempts = []
    deadline = time.monotonic() + args.refresh_confirm_timeout_ms / 1000
    attempt_index = 0
    while time.monotonic() < deadline:
        attempt_index += 1
        wait_ms(args.refresh_confirm_poll_ms, session, session_name)
        signatures = metric_request_signatures(list_metric_requests(session, session_name))
        attempts.append(
            {
                "attempt": attempt_index,
                "signature_count": len(signatures),
                "signatures": sorted(signatures),
            }
        )
        if len(signatures) >= args.refresh_confirm_min_signatures:
            return {"ok": True, "attempts": attempts}
    errors.append(
        f"group {group_name} refresh confirmation timed out: "
        f"expected at least {args.refresh_confirm_min_signatures} fresh signatures"
    )
    return {"ok": False, "attempts": attempts}


def list_log_requests(session: str, session_name: str) -> list[dict[str, Any]]:
    raw = run_agent_browser(
        ["network", "requests", "--filter", API_LOG, "--method", "POST", "--json"],
        session,
        session_name,
        timeout=30,
    )
    payload = json.loads(raw)
    return payload.get("data", {}).get("requests", [])


def get_request_detail(request_id: str, session: str, session_name: str) -> dict[str, Any]:
    raw = run_agent_browser(["network", "request", request_id, "--json"], session, session_name, timeout=30)
    return json.loads(raw).get("data", {})


def parse_json_or_none(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def request_query_names(request: dict[str, Any]) -> list[str]:
    post_data = parse_json_or_none(request.get("postData")) or {}
    return [query.get("name") for query in post_data.get("queries", []) if query.get("name")]


def request_signature_from_names(query_names: list[str]) -> str:
    # One monitor card normally emits one GetTrainMetricRange request. Terrain
    # cards can contain l0-l9 query series, so count unique request signatures
    # instead of raw query-name count when checking lazy-load coverage.
    return "|".join(sorted(query_names))


def filter_metric_details_for_group(
    group_name: str,
    details: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    allowlist = GROUP_QUERY_ALLOWLIST.get(group_name)
    if not allowlist:
        return details, [], []

    allowed_signatures = {request_signature_from_names(list(names)) for names in allowlist}
    matched: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for detail in details:
        signature = request_signature_from_names(detail.get("query_names", []))
        if signature in allowed_signatures:
            matched.append(detail)
        else:
            filtered.append(detail)
    missing = sorted(allowed_signatures - {request_signature_from_names(d.get("query_names", [])) for d in matched})
    return matched, filtered, missing


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    items = result.get("items") or []
    point_count = 0
    latest_value = None
    latest_timestamp = None
    first_value = None
    for item in items:
        values = item.get("values") or []
        point_count += len(values)
        if values and latest_timestamp is None:
            first_value = values[0].get("value")
        if values:
            latest_value = values[-1].get("value")
            latest_timestamp = values[-1].get("timestamp")
    return {
        "id": result.get("id"),
        "item_count": len(items),
        "point_count": point_count,
        "first_value": first_value,
        "latest_value": latest_value,
        "latest_timestamp": latest_timestamp,
    }


def collect_metric_details(
    requests: list[dict[str, Any]],
    session: str,
    session_name: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    details = []
    for request in requests:
        request_id = request.get("requestId") or request.get("id")
        if not request_id:
            continue
        ok, raw = run_agent_browser_safe(
            ["network", "request", request_id, "--json"],
            session,
            session_name,
            timeout=30,
        )
        if not ok:
            errors.append(f"metric request detail failed for {request_id}: {raw}")
            continue
        detail = json.loads(raw).get("data", {})
        post_data = parse_json_or_none(detail.get("postData")) or {}
        response_json = parse_json_or_none(detail.get("responseBody")) or {}
        results = ((response_json.get("data") or {}).get("results") or []) if isinstance(response_json, dict) else []
        details.append(
            {
                "request_id": request_id,
                "status": detail.get("status"),
                "query_names": [query.get("name") for query in post_data.get("queries", []) if query.get("name")],
                "query_count": len(post_data.get("queries", [])),
                "result_count": len(results),
                "results": [summarize_result(result) for result in results],
                "post_data": post_data,
                "response_json": response_json,
            }
        )
    return details


def inspect_metric_coverage(
    group_name: str,
    requests: list[dict[str, Any]],
    session: str,
    session_name: str,
    errors: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int]:
    raw_details = collect_metric_details(requests, session, session_name, errors)
    details, filtered_details, missing = filter_metric_details_for_group(group_name, raw_details)
    return details, filtered_details, missing, effective_card_count(group_name, details)


def poll_metric_coverage(
    group_name: str,
    expected_card_requests: int,
    args: argparse.Namespace,
    errors: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int, list[dict[str, Any]]]:
    attempts = []
    deadline = time.monotonic() + args.coverage_poll_timeout_ms / 1000
    best_details: list[dict[str, Any]] = []
    best_filtered: list[dict[str, Any]] = []
    best_missing: list[str] = []
    best_count = -1
    while True:
        requests = list_metric_requests(args.session, args.session_name)
        details, filtered, missing, count = inspect_metric_coverage(group_name, requests, args.session, args.session_name, errors)
        attempts.append(
            {
                "elapsed_ms": int((args.coverage_poll_timeout_ms / 1000 - max(0, deadline - time.monotonic())) * 1000),
                "raw_request_count": len(requests),
                "filtered_card_count": count,
                "filtered_signature_count": len(metric_request_signatures_from_details(details)),
                "filtered_out_count": len(filtered),
                "missing_allowlist_signatures": missing,
            }
        )
        if count > best_count:
            best_details, best_filtered, best_missing, best_count = details, filtered, missing, count
        if count >= expected_card_requests or time.monotonic() >= deadline:
            return best_details, best_filtered, best_missing, max(best_count, 0), attempts
        wait_ms(args.coverage_poll_interval_ms, args.session, args.session_name)


def scroll_result_bottom_reached(result: Any, tolerance_px: int) -> bool:
    if not isinstance(result, dict):
        return False
    after = result.get("after")
    max_scroll = result.get("maxScroll")
    try:
        return float(max_scroll) - float(after) <= tolerance_px
    except (TypeError, ValueError):
        return False


def result_has_points(detail: dict[str, Any]) -> bool:
    return any(result.get("point_count", 0) > 0 for result in detail.get("results", []))


def classify_group_capture(
    expected_card_requests: int,
    view_ref_count: int,
    request_signature_count: int,
    nonempty_request_count: int,
    ts_validation: dict[str, Any],
) -> str:
    if view_ref_count < expected_card_requests or request_signature_count < expected_card_requests:
        return "incomplete"
    if request_signature_count > expected_card_requests:
        return "polluted"
    if request_signature_count == expected_card_requests and nonempty_request_count == 0:
        return "empty"
    if not ts_validation.get("ok"):
        return "stale"
    return "ok"


def collapse_all_groups(session: str, session_name: str, errors: list[str]) -> None:
    script = """
    (() => {
      const clicked = [];
      const buttons = [...document.querySelectorAll('[aria-expanded="true"]')];
      for (const button of buttons) {
        const text = (button.innerText || button.textContent || '').trim();
        if (!/\\(\\s*\\d+\\s*\\)/.test(text)) continue;
        button.dispatchEvent(new MouseEvent('mousedown', {bubbles:true, cancelable:true, view:window}));
        button.dispatchEvent(new MouseEvent('mouseup', {bubbles:true, cancelable:true, view:window}));
        button.click();
        clicked.push(text);
      }
      return {clicked};
    })()
    """
    ok, output = run_agent_browser_safe(["eval", script], session, session_name, timeout=20)
    if ok:
        try:
            if json.loads(output).get("clicked"):
                wait_ms(250, session, session_name)
        except json.JSONDecodeError:
            pass
        return
    errors.append(f"DOM collapse failed, falling back to snapshot clicks: {output}")
    for _ in range(3):
        snapshot = get_snapshot(session, session_name)
        expanded = [group for group in parse_groups_from_snapshot(snapshot) if group["expanded"]]
        if not expanded:
            return
        for group in expanded:
            ok, output = run_agent_browser_safe(["click", f"@{group['ref']}"], session, session_name, timeout=20)
            if not ok:
                errors.append(f"collapse {group['label']} failed: {output}")
            wait_ms(250, session, session_name)


def expanded_group_count(session: str, session_name: str) -> int | None:
    script = """
    (() => [...document.querySelectorAll('[aria-expanded="true"]')]
      .filter(el => /\\(\\s*\\d+\\s*\\)/.test((el.innerText || el.textContent || '').trim()))
      .length)()
    """
    ok, output = run_agent_browser_safe(["eval", script], session, session_name, timeout=20)
    if not ok:
        return None
    try:
        return int(json.loads(output))
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            return int(output.strip())
        except ValueError:
            return None


def collapse_all_groups_across_page(args: argparse.Namespace, errors: list[str]) -> None:
    """Collapse every expanded overview group exposed while scanning the page.

    The monitor page virtualizes/lazy-loads content, so a single snapshot only
    sees the current viewport. Scan from top to bottom and collapse visible
    expanded groups at each stop before collecting any specific group.
    """
    drag_page_scrollbar_to_top(args)
    wait_ms(args.collapse_scan_wait_ms, args.session, args.session_name)
    for step in range(args.collapse_scan_steps + 1):
        collapse_all_groups(args.session, args.session_name, errors)
        if expanded_group_count(args.session, args.session_name) == 0:
            break
        if step >= args.collapse_scan_steps:
            break
        drag_page_scrollbar_down(args, step)
        wait_ms(args.collapse_scan_wait_ms, args.session, args.session_name)
    drag_page_scrollbar_to_top(args)
    wait_ms(args.collapse_scan_wait_ms, args.session, args.session_name)


def find_group(session: str, session_name: str, group_name: str) -> dict[str, Any] | None:
    snapshot = get_snapshot(session, session_name)
    for group in parse_groups_from_snapshot(snapshot):
        if group["name"] == group_name:
            return group
    return None


def find_overview_panel_ref(session: str, session_name: str) -> str | None:
    snapshot = get_snapshot(session, session_name)
    for ref_id, ref in snapshot.get("data", {}).get("refs", {}).items():
        if ref.get("role") == "tabpanel" and ref.get("name") == OVERVIEW_TAB:
            return ref_id
    return None


def trigger_group_lazy_load(
    group: dict[str, Any],
    args: argparse.Namespace,
    errors: list[str],
    baseline_latest_ms: int | None,
) -> dict[str, Any]:
    session = args.session
    session_name = args.session_name
    clear_network(session, session_name)

    current = locate_group_by_dragging(group, args, errors)
    if not current:
        errors.append(f"group not found before expand: {group['name']}")
        return {"group": group, "metric_details": [], "request_count": 0, "error": "group not found"}
    # Locating a lower group can scroll past lazy-loaded charts from earlier
    # groups. Clear after locating so coverage belongs to the target group.
    clear_network(session, session_name)
    if not current["expanded"]:
        ok, output = run_agent_browser_safe(["click", f"@{current['ref']}"], session, session_name, timeout=20)
        if not ok:
            errors.append(f"expand {current['label']} failed: {output}")
    wait_ms(args.settle_ms, session, session_name)

    refreshed = find_group(session, session_name, group["name"])
    if refreshed:
        current = refreshed

    expected_card_requests = group["declared_count"]
    expanded_snapshot = get_snapshot(session, session_name)
    view_refs = group_view_refs_from_snapshot(expanded_snapshot, current["ref"])
    if len(view_refs) < expected_card_requests:
        errors.append(
            f"group {group['name']} declared {expected_card_requests} cards, "
            f"but snapshot exposed only {len(view_refs)} 查看 refs"
        )
    if not view_refs:
        return {
            "group": group,
            "expected_card_request_count": expected_card_requests,
            "view_ref_count": 0,
            "view_refs": [],
            "trigger_trace": [],
            "refresh_confirmation": {"ok": False, "attempts": []},
            "request_count": 0,
            "detail_count": 0,
            "query_names": [],
            "unique_query_name_count": 0,
            "request_signatures": [],
            "request_signature_count": 0,
            "coverage_ok": False,
            "exact_coverage_ok": False,
            "data_status": "incomplete",
            "nonempty_request_count": 0,
            "metric_details": [],
            "error": "no view refs exposed for group",
        }

    observed_signatures: set[str] = set()
    trigger_trace = []
    coverage_requests: list[dict[str, Any]] = []
    coverage_details: list[dict[str, Any]] = []
    ts_validation: dict[str, Any] = {"ok": False, "latest_timestamp_ms": None, "reason": "not_checked"}
    no_new_data_count = 0
    bottom_reached = False
    wait_ms(args.initial_dwell_ms, session, session_name)
    coverage_details, filtered_details, missing_signatures, filtered_card_count, poll_attempts = poll_metric_coverage(
        group["name"], expected_card_requests, args, errors
    )
    observed_signatures.update(metric_request_signatures(list_metric_requests(session, session_name)))
    trigger_trace.append(
        {
            "step": 0,
            "signature_count": len(observed_signatures),
            "filtered_card_count": filtered_card_count,
            "poll_attempts": poll_attempts,
            "action": "initial",
        }
    )

    for idx in range(args.max_scrollbar_drags_per_group + 1):
        coverage_requests = list_metric_requests(session, session_name)
        observed_signatures.update(metric_request_signatures(coverage_requests))
        filtered_card_count = effective_card_count(group["name"], coverage_details)
        if filtered_card_count >= expected_card_requests:
            ts_validation = timestamp_validation(coverage_details, baseline_latest_ms, args)
            nonempty_checked = [detail for detail in coverage_details if result_has_points(detail)]
            if not ts_validation["ok"] and filtered_card_count >= expected_card_requests and not nonempty_checked and args.accept_empty_metric_groups:
                ts_validation = {
                    "ok": True,
                    "latest_timestamp_ms": None,
                    "mode": "empty",
                    "reason": "all_captured_metric_responses_empty",
                }
            trigger_trace.append(
                {
                    "step": idx,
                    "signature_count": len(observed_signatures),
                    "filtered_signature_count": len(metric_request_signatures_from_details(coverage_details)),
                    "filtered_card_count": filtered_card_count,
                    "filtered_out_count": len(filtered_details),
                    "missing_allowlist_signatures": missing_signatures,
                    "action": "coverage_check",
                    "timestamp_ok": ts_validation["ok"],
                    "latest_timestamp_ms": ts_validation.get("latest_timestamp_ms"),
                }
            )
            if ts_validation["ok"] or args.skip_timestamp_gate:
                break
        if idx >= args.max_scrollbar_drags_per_group:
            break
        if bottom_reached and no_new_data_count >= args.no_new_data_patience:
            trigger_trace.append(
                {
                    "step": idx,
                    "action": "stop_at_bottom_no_new_data",
                    "filtered_card_count": filtered_card_count,
                    "no_new_data_count": no_new_data_count,
                }
            )
            break
        before = len(observed_signatures)
        before_card_count = filtered_card_count
        ok, output = drag_group_scrollbar_once(args, idx, expected_card_requests)
        if not ok:
            errors.append(f"group {group['name']} scrollbar drag {idx + 1} failed: {output}")
        coverage_details, filtered_details, missing_signatures, filtered_card_count, poll_attempts = poll_metric_coverage(
            group["name"], expected_card_requests, args, errors
        )
        observed_signatures.update(metric_request_signatures(list_metric_requests(session, session_name)))
        after = len(observed_signatures)
        scroll_result = parse_json_or_none(output)
        bottom_reached = scroll_result_bottom_reached(scroll_result, args.scroll_bottom_tolerance_px)
        if filtered_card_count <= before_card_count:
            no_new_data_count += 1
        else:
            no_new_data_count = 0
        trigger_trace.append(
            {
                "step": idx + 1,
                "signature_count": after,
                "new_signature_count": after - before,
                "action": "drag_scrollbar",
                "ok": ok,
                "scroll_result": scroll_result,
                "bottom_reached": bottom_reached,
                "no_new_data_count": no_new_data_count,
                "filtered_card_count": filtered_card_count,
                "poll_attempts": poll_attempts,
            }
        )
    refresh_confirmation = {"ok": False, "attempts": []}
    if args.confirm_refresh and (len(observed_signatures) >= expected_card_requests or args.confirm_refresh_even_if_incomplete):
        refresh_confirmation = wait_for_overview_refresh(group["name"], args, errors)

    raw_detail_count = len(coverage_details)
    details, filtered_out_details, missing_allowlist_signatures = filter_metric_details_for_group(group["name"], coverage_details)
    query_names = sorted({name for detail in details for name in detail.get("query_names", [])})
    request_signatures = sorted(metric_request_signatures_from_details(details))
    nonempty_details = [detail for detail in details if result_has_points(detail)]
    request_signature_count = len(request_signatures)
    card_count = effective_card_count(group["name"], details)
    coverage_ok = len(view_refs) >= expected_card_requests and card_count >= expected_card_requests
    exact_coverage_ok = len(view_refs) == expected_card_requests and card_count == expected_card_requests
    if ts_validation.get("reason") == "not_checked":
        ts_validation = timestamp_validation(details, baseline_latest_ms, args)
        if (
            not ts_validation["ok"]
            and exact_coverage_ok
            and not nonempty_details
            and args.accept_empty_metric_groups
        ):
            ts_validation = {
                "ok": True,
                "latest_timestamp_ms": None,
                "mode": "empty",
                "reason": "all_captured_metric_responses_empty",
            }
    data_status = classify_group_capture(
        expected_card_requests,
        len(view_refs),
        card_count,
        len(nonempty_details),
        ts_validation,
    )
    if not ts_validation["ok"]:
        errors.append(f"group {group['name']} timestamp validation failed: {ts_validation}")
    if not coverage_ok:
        errors.append(
            f"group {group['name']} declared {expected_card_requests} cards, "
            f"captured {card_count} effective metric card requests"
        )
    if coverage_ok and not exact_coverage_ok:
        errors.append(
            f"group {group['name']} declared {expected_card_requests} cards, "
            f"captured {card_count} effective metric card requests; possible adjacent-group pollution"
        )

    return {
        "group": group,
        "expected_card_request_count": expected_card_requests,
        "view_ref_count": len(view_refs),
        "view_refs": view_refs,
        "trigger_trace": trigger_trace,
        "refresh_confirmation": refresh_confirmation,
        "request_count": len(coverage_requests),
        "raw_detail_count": raw_detail_count,
        "detail_count": len(details),
        "filtered_out_detail_count": len(filtered_out_details),
        "filtered_out_request_signatures": sorted(metric_request_signatures_from_details(filtered_out_details)),
        "missing_allowlist_signatures": missing_allowlist_signatures,
        "query_names": query_names,
        "unique_query_name_count": len(query_names),
        "request_signatures": request_signatures,
        "request_signature_count": request_signature_count,
        "effective_card_count": card_count,
        "coverage_ok": coverage_ok,
        "exact_coverage_ok": exact_coverage_ok,
        "data_status": data_status,
        "timestamp_validation": ts_validation,
        "nonempty_request_count": len(nonempty_details),
        "metric_details": details,
    }


def normalize_log_entry(raw: str) -> dict[str, Any]:
    parsed = parse_json_or_none(raw)
    return parsed if isinstance(parsed, dict) else {"raw": raw}


def collect_training_logs(args: argparse.Namespace, errors: list[str]) -> dict[str, Any]:
    session = args.session
    session_name = args.session_name
    clear_network(session, session_name)
    click_text(LOGS_TAB, session, session_name)
    wait_ms(args.log_wait_ms, session, session_name)
    body_text = get_body_text(session, session_name)

    for _ in range(args.log_scroll_steps):
        scroll_down(args.scroll_px, session, session_name)
        wait_ms(args.log_wait_ms, session, session_name)

    requests = list_log_requests(session, session_name)
    entries: list[dict[str, Any]] = []
    for request in requests:
        request_id = request.get("requestId") or request.get("id")
        if not request_id:
            continue
        ok, raw = run_agent_browser_safe(["network", "request", request_id, "--json"], session, session_name, timeout=30)
        if not ok:
            errors.append(f"log request detail failed for {request_id}: {raw}")
            continue
        detail = json.loads(raw).get("data", {})
        response_json = parse_json_or_none(detail.get("responseBody")) or {}
        for raw_log in ((response_json.get("data") or {}).get("logs") or []):
            entries.append(normalize_log_entry(raw_log))

    if not entries:
        entries = parse_log_entries_from_text(body_text)

    keywords = [item.strip() for item in args.log_keywords.split(",") if item.strip()]
    important = []
    seen = set()
    for entry in entries:
        text = json.dumps(entry, ensure_ascii=False)
        if keywords and not any(keyword.lower() in text.lower() for keyword in keywords):
            continue
        key = (entry.get("time"), entry.get("level"), entry.get("module"), entry.get("message"), entry.get("raw"))
        if key in seen:
            continue
        seen.add(key)
        important.append(entry)
        if len(important) >= args.max_important_logs:
            break

    return {
        "request_count": len(requests),
        "entry_count": len(entries),
        "important_count": len(important),
        "keywords": keywords,
        "important_entries": important,
        "raw_visible_text": body_text,
    }


def parse_log_entries_from_text(text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    entries = []
    current: list[str] = []
    for line in lines:
        if TIMESTAMP_RE.match(line):
            if current:
                entries.append({"time": current[0], "raw": "\n".join(current)})
            current = [line]
        elif current:
            current.append(line)
    if current:
        entries.append({"time": current[0], "raw": "\n".join(current)})
    return entries


def write_outputs(capture: dict[str, Any]) -> Path:
    capture_id = capture["capture_id"]
    session_dir = SESSIONS_DIR / capture_id
    groups_dir = session_dir / "groups"
    session_dir.mkdir(parents=True, exist_ok=True)
    groups_dir.mkdir(parents=True, exist_ok=True)

    group_index = []
    for idx, group_capture in enumerate(capture["overview_groups"], start=1):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", group_capture["group"]["name"]).strip("_")
        path = groups_dir / f"{idx:02d}_{safe_name or 'group'}.json"
        path.write_text(json.dumps(group_capture, ensure_ascii=False, indent=2), encoding="utf-8")
        group_index.append(
            {
                "name": group_capture["group"]["name"],
                "declared_count": group_capture["group"]["declared_count"],
                "unique_query_name_count": group_capture["unique_query_name_count"],
                "request_signature_count": group_capture["request_signature_count"],
                "effective_card_count": group_capture.get("effective_card_count", group_capture["request_signature_count"]),
                "coverage_ok": group_capture["coverage_ok"],
                "exact_coverage_ok": group_capture.get("exact_coverage_ok", False),
                "data_status": group_capture.get("data_status", "unknown"),
                "refresh_confirmed": group_capture["refresh_confirmation"]["ok"],
                "timestamp_ok": group_capture.get("timestamp_validation", {}).get("ok", False),
                "latest_timestamp_ms": group_capture.get("timestamp_validation", {}).get("latest_timestamp_ms"),
                "nonempty_request_count": group_capture["nonempty_request_count"],
                "filtered_out_detail_count": group_capture.get("filtered_out_detail_count", 0),
                "file": str(path.relative_to(session_dir)),
            }
        )

    summary = {
        "capture_id": capture_id,
        "captured_at_local": capture["captured_at_local"],
        "monitor_url": capture["monitor_url"],
        "task_status": capture.get("task_status", {}),
        "auto_refresh_enabled": capture["auto_refresh_enabled"],
        "page_group_count": len(capture["page_inventory"]),
        "captured_group_count": len(capture["overview_groups"]),
        "group_index": group_index,
        "log_entry_count": capture["training_logs"]["entry_count"],
        "important_log_count": capture["training_logs"]["important_count"],
        "errors": capture["errors"],
    }
    (session_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (session_dir / "page_inventory.json").write_text(
        json.dumps(capture["page_inventory"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (session_dir / "training_logs.json").write_text(
        json.dumps(capture["training_logs"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (session_dir / "capture.json").write_text(json.dumps(capture, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_dir


def collect(args: argparse.Namespace) -> Path:
    errors: list[str] = []
    task_status: dict[str, Any] = {}
    page_inventory: list[dict[str, Any]] = []
    overview_groups: list[dict[str, Any]] = []
    training_logs: dict[str, Any] = {"request_count": 0, "entry_count": 0, "important_count": 0, "important_entries": []}
    monitor_url = switch_to_monitor_tab(
        args.session,
        args.session_name,
        args.monitor_url,
        args.allow_monitor_tab_fallback,
    )
    auto_refresh_enabled = False
    try:
        set_page_zoom(args, errors)
        click_text(OVERVIEW_TAB, args.session, args.session_name)
        wait_ms(1500, args.session, args.session_name)
        set_page_zoom(args, errors)
        auto_refresh_enabled = ensure_auto_refresh(args.session, args.session_name, args.auto_refresh_interval, errors)
        if not auto_refresh_enabled and args.require_auto_refresh:
            raise CaptureError(f"Could not enable auto refresh: {args.auto_refresh_interval}")

        scroll_up(50000, args.session, args.session_name)
        wait_ms(1200, args.session, args.session_name)
        body_text = get_body_text(args.session, args.session_name)
        task_status = detect_task_status(body_text)
        if args.task_state_mode == "running":
            task_status["mode"] = "running"
            task_status["is_running"] = True
        elif args.task_state_mode == "history":
            task_status["mode"] = "history"
            task_status["is_running"] = False
        args.history_mode = task_status["mode"] == "history"
        page_inventory = parse_page_group_inventory(body_text)

        collapse_all_groups_across_page(args, errors)
        groups = [
            {
                "label": f"{item['name']}( {item['declared_count']} )",
                "name": item["name"],
                "declared_count": item["declared_count"],
                "expanded": False,
                "ref": "",
            }
            for item in page_inventory
        ]
        if not groups:
            snapshot = get_snapshot(args.session, args.session_name)
            groups = parse_groups_from_snapshot(snapshot)
        if args.group:
            wanted = set(args.group)
            groups = [group for group in groups if group["name"] in wanted or group["label"] in wanted]

        baseline_latest_ms = None
        for index, group in enumerate(groups):
            print(
                f"[{index + 1}/{len(groups)}] collecting {group['name']} "
                f"({group['declared_count']} cards)",
                flush=True,
            )
            if index == 0 or args.reset_to_top_each_group:
                drag_page_scrollbar_to_top(args)
                wait_ms(500, args.session, args.session_name)
            if args.full_collapse_each_group:
                collapse_all_groups_across_page(args, errors)
            else:
                collapse_all_groups(args.session, args.session_name, errors)
            group_error_start = len(errors)
            group_capture = trigger_group_lazy_load(group, args, errors, baseline_latest_ms)
            if group_capture.get("error") and args.recover_with_full_collapse:
                first_attempt_errors = errors[group_error_start:]
                del errors[group_error_start:]
                collapse_all_groups_across_page(args, errors)
                retry_error_start = len(errors)
                group_capture = trigger_group_lazy_load(group, args, errors, baseline_latest_ms)
                if not group_capture.get("error"):
                    group_capture["recovered_after_full_collapse"] = True
                    group_capture["first_attempt_errors"] = first_attempt_errors
                else:
                    retry_errors = errors[retry_error_start:]
                    del errors[retry_error_start:]
                    errors.extend(first_attempt_errors)
                    errors.extend(retry_errors)
            group_latest = group_capture.get("timestamp_validation", {}).get("latest_timestamp_ms")
            if baseline_latest_ms is None and group_latest is not None:
                baseline_latest_ms = group_latest
            overview_groups.append(group_capture)
            print(
                f"[{index + 1}/{len(groups)}] {group['name']}: "
                f"status={group_capture.get('data_status')} "
                f"effective={group_capture.get('effective_card_count')}/"
                f"{group['declared_count']}",
                flush=True,
            )
            if args.collapse_after_group:
                current = find_group(args.session, args.session_name, group["name"])
                if current and current["expanded"]:
                    run_agent_browser_safe(["click", f"@{current['ref']}"], args.session, args.session_name, timeout=20)
                    wait_ms(300, args.session, args.session_name)

        print("[logs] collecting training logs", flush=True)
        training_logs = collect_training_logs(args, errors)
    finally:
        if args.final_collapse:
            run_agent_browser_safe(["find", "text", OVERVIEW_TAB, "click", "--exact"], args.session, args.session_name, timeout=20)
            try:
                wait_ms(800, args.session, args.session_name)
                collapse_all_groups_across_page(args, errors)
                remaining = expanded_group_count(args.session, args.session_name)
                if remaining:
                    errors.append(f"final collapse left {remaining} expanded groups")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"final collapse failed: {exc}")
    capture = {
        "capture_id": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "captured_at_local": dt.datetime.now().isoformat(timespec="seconds"),
        "monitor_url": monitor_url,
        "task_status": task_status,
        "auto_refresh_enabled": auto_refresh_enabled,
        "page_inventory": page_inventory,
        "overview_groups": overview_groups,
        "training_logs": training_logs,
        "errors": errors,
    }
    return write_outputs(capture)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Tencent Arena lazy-loaded monitor overview data")
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--session-name", default=DEFAULT_SESSION_NAME)
    parser.add_argument("--monitor-url", default=None)
    parser.add_argument(
        "--allow-monitor-tab-fallback",
        action="store_true",
        help="When no --monitor-url is provided and the active tab is not a monitor page, switch to another open monitor tab.",
    )
    parser.add_argument("--auto-refresh-interval", default=DEFAULT_AUTO_REFRESH)
    parser.add_argument("--require-auto-refresh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group", action="append", help="Only collect this overview group name/label; repeatable")
    parser.add_argument("--page-zoom", type=float, default=0.75, help="CSS zoom applied before capture. Set <=0 to disable.")
    parser.add_argument("--zoom-wait-ms", type=int, default=600)
    parser.add_argument("--settle-ms", type=int, default=1200)
    parser.add_argument("--refresh-wait-ms", type=int, default=5500)
    parser.add_argument("--dwell-ms", type=int, default=5500)
    parser.add_argument("--initial-dwell-ms", type=int, default=1200)
    parser.add_argument("--scroll-px", type=int, default=650, help="Only used for optional training-log scrolling.")
    parser.add_argument(
        "--card-click-wait-ms",
        type=int,
        default=1800,
        help="Wait time after clicking each overview card's 查看 trigger.",
    )
    parser.add_argument("--drag-wait-ms", type=int, default=1800)
    parser.add_argument("--coverage-poll-timeout-ms", type=int, default=1400)
    parser.add_argument("--coverage-poll-interval-ms", type=int, default=250)
    parser.add_argument("--scrollbar-x", type=int, default=1195)
    parser.add_argument("--scrollbar-start-y", type=int, default=300)
    parser.add_argument("--scrollbar-step-y", type=int, default=40)
    parser.add_argument("--scrollbar-drag-dy", type=int, default=220)
    parser.add_argument("--small-group-threshold", type=int, default=4)
    parser.add_argument("--small-group-scrollbar-drag-dy", type=int, default=110)
    parser.add_argument("--scrollbar-top-drag-start-y", type=int, default=650)
    parser.add_argument("--scrollbar-top-y", type=int, default=170)
    parser.add_argument("--max-scrollbar-drags-per-group", type=int, default=20)
    parser.add_argument("--no-new-data-patience", type=int, default=2)
    parser.add_argument("--scroll-bottom-tolerance-px", type=int, default=16)
    parser.add_argument("--max-group-locate-drags", type=int, default=8)
    parser.add_argument("--locate-scrollbar-start-y", type=int, default=260)
    parser.add_argument("--locate-scrollbar-step-y", type=int, default=35)
    parser.add_argument("--locate-scrollbar-drag-dy", type=int, default=160)
    parser.add_argument("--locate-wait-ms", type=int, default=700)
    parser.add_argument("--collapse-scan-steps", type=int, default=10)
    parser.add_argument("--collapse-scan-wait-ms", type=int, default=350)
    parser.add_argument("--full-collapse-each-group", action="store_true")
    parser.add_argument("--recover-with-full-collapse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-collapse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-to-top-each-group", action="store_true")
    parser.add_argument(
        "--refresh-confirm-timeout-ms",
        type=int,
        default=15000,
        help="After lazy-load coverage, wait this long for a fresh auto-refresh request batch.",
    )
    parser.add_argument(
        "--refresh-confirm-poll-ms",
        type=int,
        default=5500,
        help="Polling interval for post-scroll auto-refresh confirmation.",
    )
    parser.add_argument(
        "--refresh-confirm-min-signatures",
        type=int,
        default=1,
        help="Minimum fresh metric request signatures required after post-scroll wait.",
    )
    parser.add_argument("--confirm-refresh", action="store_true")
    parser.add_argument("--confirm-refresh-even-if-incomplete", action="store_true")
    parser.add_argument("--skip-timestamp-gate", action="store_true")
    parser.add_argument("--accept-empty-metric-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--task-state-mode",
        choices=("auto", "running", "history"),
        default="auto",
        help="auto reads the page status before capture; running enforces freshness; history compares group timestamps.",
    )
    parser.add_argument("--history-mode", action="store_true")
    parser.add_argument("--running-timestamp-max-age-ms", type=int, default=180000)
    parser.add_argument("--history-timestamp-tolerance-ms", type=int, default=120000)
    parser.add_argument("--collapse-after-group", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-wait-ms", type=int, default=2500)
    parser.add_argument("--log-scroll-steps", type=int, default=2)
    parser.add_argument("--log-keywords", default=",".join(DEFAULT_IMPORTANT_LOG_KEYWORDS))
    parser.add_argument("--max-important-logs", type=int, default=80)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        session_dir = collect(args)
    except CaptureError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(session_dir)
    print(session_dir / "summary.json")
    print(session_dir / "training_logs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
