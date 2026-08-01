#!/usr/bin/env python3
"""Analyze Go2 VisionNav CSV logs with Python standard library only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "frame", "t_ms", "loop_ms", "inference_ms", "deadline_misses",
    "vx", "vy", "wz", "vx_raw", "vy_raw", "wz_raw",
    "clr_L", "clr_F", "clr_R",
    "dep_inval", "dep_meanv", "front_inval", "front_min", "front_mean",
    "pgx", "pgy", "pgz",
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: math.nan for key in ("min", "mean", "p50", "p95", "p99", "max", "std")}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def finite_float(text: str, column: str, row_number: int) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"第 {row_number} 行 {column} 不是数字: {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"第 {row_number} 行 {column} 不是有限值: {text!r}")
    return value


def discover_csv(paths: Iterable[str], recursive: bool) -> list[Path]:
    found: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_file() and path.suffix.lower() == ".csv":
            found.add(path.resolve())
        elif path.is_dir():
            pattern = "**/visnav_diag_*.csv" if recursive else "visnav_diag_*.csv"
            found.update(item.resolve() for item in path.glob(pattern) if item.is_file())
        else:
            found.update(item.resolve() for item in Path().glob(raw) if item.is_file())
    return sorted(found)


def load_columns(path: Path) -> tuple[dict[str, list[float]], int, dict[str, int]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        sanitized_lines = (line.replace("\0", "") for line in handle)
        reader = csv.DictReader(sanitized_lines)
        if not reader.fieldnames:
            raise ValueError("CSV 没有表头")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"缺少列: {', '.join(missing)}")

        columns = {name: [] for name in reader.fieldnames}
        row_count = 0
        skipped_rows = 0
        for row_number, row in enumerate(reader, start=2):
            if not row or not row.get("frame", "").strip():
                continue
            if any(row.get(name) in (None, "") for name in columns):
                skipped_rows += 1
                continue
            try:
                parsed = {
                    name: finite_float(row[name], name, row_number)
                    for name in columns
                }
            except ValueError:
                skipped_rows += 1
                continue
            for name, value in parsed.items():
                columns[name].append(value)
            row_count += 1
    if row_count == 0:
        raise ValueError("CSV 没有数据行")
    return columns, row_count, {"skipped_rows": skipped_rows}


def fraction(values: list[float], predicate) -> float:
    return sum(1 for value in values if predicate(value)) / len(values) if values else math.nan


def max_abs(columns: dict[str, list[float]], names: Iterable[str]) -> float:
    return max(abs(value) for name in names for value in columns.get(name, []))


def masked(values: list[float], mask: list[float]) -> list[float]:
    return [value for value, keep in zip(values, mask) if keep >= 0.5]


def analyze(path: Path) -> dict:
    c, frames, data_quality = load_columns(path)
    duration_s = max((c["t_ms"][-1] - c["t_ms"][0]) / 1000.0, 1e-9)
    effective_hz = (frames - 1) / duration_s if frames > 1 else 0.0

    tilt_deg = []
    for x, y, z in zip(c["pgx"], c["pgy"], c["pgz"]):
        norm = max(math.sqrt(x * x + y * y + z * z), 1e-9)
        cosine = max(-1.0, min(1.0, -z / norm))
        tilt_deg.append(math.degrees(math.acos(cosine)))

    action_names = [f"action{i}" for i in range(12) if f"action{i}" in c]
    target_names = [f"target{i}" for i in range(12) if f"target{i}" in c]
    applied_target_names = [f"applied_target{i}" for i in range(12) if f"applied_target{i}" in c]
    q_names = [f"q{i}" for i in range(12) if f"q{i}" in c]
    qerr_names = [f"qerr{i}" for i in range(12) if f"qerr{i}" in c]
    tau_names = [f"tau{i}" for i in range(12) if f"tau{i}" in c]
    pd_tau_names = [f"pd_tau{i}" for i in range(12) if f"pd_tau{i}" in c]

    tracking_errors: list[float] = []
    if len(qerr_names) == 12:
        for name in qerr_names:
            tracking_errors.extend(abs(value) for value in c[name])
    elif len(applied_target_names) == 12 and len(q_names) == 12:
        for i in range(12):
            tracking_errors.extend(
                abs(t - q) for t, q in zip(c[f"applied_target{i}"], c[f"q{i}"])
            )
    elif len(target_names) == 12 and len(q_names) == 12:
        for i in range(12):
            tracking_errors.extend(abs(t - q) for t, q in zip(c[f"target{i}"], c[f"q{i}"]))

    action_steps: list[float] = []
    if len(action_names) == 12 and frames > 1:
        for row in range(1, frames):
            action_steps.append(max(
                abs(c[f"action{i}"][row] - c[f"action{i}"][row - 1])
                for i in range(12)
            ))

    cmd = {axis: stats(c[axis]) for axis in ("vx", "vy", "wz")}
    override_detected = (
        cmd["vx"]["std"] < 1e-5 and cmd["vy"]["std"] < 1e-5 and cmd["wz"]["std"] < 1e-5
    )
    extended = "theory_vx" in c and "cmd_source" in c
    external_control = None
    if extended:
        source_code = int(round(statistics.fmean(c["cmd_source"])))
        mode = {0: "fixed", 1: "uwb", 2: "nav"}.get(source_code, f"unknown({source_code})")
        steady_rows = [
            row for row, t_ms in enumerate(c["t_ms"])
            if t_ms - c["t_ms"][0] >= 1000.0
        ]
        if not steady_rows:
            steady_rows = list(range(frames))
        theory_error = {
            axis: stats([
                c[axis][row] - c[f"theory_{axis}"][row]
                for row in steady_rows
            ])
            for axis in ("vx", "vy", "wz")
        }
        theory_abs_error = {
            axis: stats([
                abs(c[axis][row] - c[f"theory_{axis}"][row])
                for row in steady_rows
            ])
            for axis in ("vx", "vy", "wz")
        }
        external_control = {
            "mode": mode,
            "tracking_window_start_s": 1.0 if frames > len(steady_rows) else 0.0,
            "theory_cmd": {
                axis: stats(c[f"theory_{axis}"]) for axis in ("vx", "vy", "wz")
            },
            "theory_to_exec_error": theory_error,
            "theory_to_exec_abs_error": theory_abs_error,
            "max_exec_vx": max(c["vx"]),
        }
        if "uwb_valid" in c:
            valid = c["uwb_valid"]
            external_control["uwb"] = {
                "valid_fraction": statistics.fmean(valid),
                "age_s": stats(masked(c["uwb_age_s"], valid)),
                "distance_m": stats(masked(c["uwb_distance"], valid)),
                "bearing_rad": stats(masked(c["uwb_beta"], valid)),
                "closing_speed_mps": stats(masked(c["uwb_closing"], valid)),
                "expected_closing_mps": stats(masked(c["expected_closing"], valid)),
                "closing_error_mps": stats(masked(c["closing_error"], valid)),
                "closing_abs_error_mps": stats(
                    [abs(value) for value in masked(c["closing_error"], valid)]
                ),
                "error_frame_fraction": fraction(c["uwb_error"], lambda value: value != 0),
                "disabled_frame_fraction": fraction(c["uwb_enabled"], lambda value: value != 1),
            }
        if "sport_valid" in c:
            valid = c["sport_valid"]
            external_control["sport_feedback"] = {
                "valid_fraction": statistics.fmean(valid),
                "age_s": stats(masked(c["sport_age_s"], valid)),
                "velocity": {
                    axis: stats(masked(c[f"sport_v{axis}"], valid))
                    for axis in ("x", "y", "z")
                },
                "yaw_speed": stats(masked(c["sport_wz"], valid)),
                "tracking_error": {
                    axis: stats(masked(c[f"sport_err_{axis}"], valid))
                    for axis in ("vx", "vy", "wz")
                },
                "tracking_abs_error": {
                    axis: stats([
                        abs(value) for value in masked(c[f"sport_err_{axis}"], valid)
                    ])
                    for axis in ("vx", "vy", "wz")
                },
            }
        if "feedback_valid" in c:
            valid = c["feedback_valid"]
            valid_sources = [
                int(round(source))
                for source, keep in zip(c["feedback_source"], valid)
                if keep >= 0.5
            ]
            source_counts = {
                name: valid_sources.count(code)
                for code, name in ((1, "sport"), (2, "uwb"))
            }
            external_control["motion_feedback"] = {
                "valid_fraction": statistics.fmean(valid),
                "source_counts": source_counts,
                "age_s": stats(masked(c["feedback_age_s"], valid)),
                "velocity": {
                    axis: stats(masked(c[f"feedback_v{axis}"], valid))
                    for axis in ("x", "y", "z")
                },
                "yaw_speed": stats(masked(c["feedback_wz"], valid)),
                "tracking_error": {
                    axis: stats(masked(c[f"feedback_err_{axis}"], valid))
                    for axis in ("vx", "vy", "wz")
                },
                "tracking_abs_error": {
                    axis: stats([
                        abs(value) for value in masked(c[f"feedback_err_{axis}"], valid)
                    ])
                    for axis in ("vx", "vy", "wz")
                },
            }

    report = {
        "file": str(path),
        "data_quality": data_quality,
        "frames": frames,
        "duration_s": duration_s,
        "effective_hz": effective_hz,
        "loop_ms": stats(c["loop_ms"]),
        "inference_ms": stats(c["inference_ms"]),
        "deadline_misses": int(c["deadline_misses"][-1]),
        "deadline_miss_fraction": int(c["deadline_misses"][-1]) / frames,
        "cmd": cmd,
        "fixed_override_detected": override_detected,
        "cmd_raw_abs_max": {
            axis: max(abs(value) for value in c[f"{axis}_raw"])
            for axis in ("vx", "vy", "wz")
        },
        "clearance": {
            side: {
                **stats(c[name]),
                "above_0_98_fraction": fraction(c[name], lambda value: value >= 0.98),
            }
            for side, name in (("left", "clr_L"), ("front", "clr_F"), ("right", "clr_R"))
        },
        "depth": {
            "invalid_fraction": stats(c["dep_inval"]),
            "mean_valid_normalized": stats(c["dep_meanv"]),
            "front_invalid_fraction": stats(c["front_inval"]),
            "front_min_normalized": stats(c["front_min"]),
            "front_mean_normalized": stats(c["front_mean"]),
        },
        "tilt_deg": stats(tilt_deg),
        "max_abs_action": max_abs(c, action_names) if action_names else math.nan,
        "max_abs_joint_effort": max_abs(c, tau_names) if tau_names else math.nan,
        "joint_effort_abs": {
            name: stats([abs(value) for value in c[name]]) for name in tau_names
        },
        "max_abs_pd_torque": max_abs(c, pd_tau_names) if pd_tau_names else math.nan,
        "pd_torque_abs": {
            name: stats([abs(value) for value in c[name]]) for name in pd_tau_names
        },
        "mechanical_power_abs_sum": (
            stats(c["mechanical_power_abs_sum"])
            if "mechanical_power_abs_sum" in c else stats([])
        ),
        "action_step": stats(action_steps),
        "joint_tracking_abs_error": stats(tracking_errors),
        "external_control": external_control,
    }
    add_verdicts(report)
    return report


def add_issue(target: list[dict], level: str, message: str) -> None:
    target.append({"level": level, "message": message})


def add_verdicts(report: dict) -> None:
    control: list[dict] = []
    nav: list[dict] = []

    if report["effective_hz"] < 45.0:
        add_issue(control, "STOP", f"策略频率仅 {report['effective_hz']:.1f} Hz")
    elif report["effective_hz"] < 48.0:
        add_issue(control, "WARN", f"策略频率偏低: {report['effective_hz']:.1f} Hz")

    infer_p95 = report["inference_ms"]["p95"]
    if infer_p95 > 20.0:
        add_issue(control, "STOP", f"推理 P95={infer_p95:.2f} ms，超过 20 ms 周期")
    elif infer_p95 > 15.0:
        add_issue(control, "WARN", f"推理 P95 偏高: {infer_p95:.2f} ms")

    miss_rate = report["deadline_miss_fraction"]
    if miss_rate > 0.01:
        add_issue(control, "STOP", f"deadline miss 比例 {miss_rate:.1%}")
    elif report["deadline_misses"] > 3:
        add_issue(control, "WARN", f"deadline miss 共 {report['deadline_misses']} 次")

    tilt_max = report["tilt_deg"]["max"]
    if tilt_max > 45.0:
        add_issue(control, "STOP", f"最大倾角 {tilt_max:.1f}°")
    elif tilt_max > 20.0:
        add_issue(control, "WARN", f"最大倾角偏大: {tilt_max:.1f}°")

    action_max = report["max_abs_action"]
    if action_max > 10.0:
        add_issue(control, "STOP", f"原始关节动作绝对值达到 {action_max:.2f}")
    elif action_max > 5.0:
        add_issue(control, "WARN", f"原始关节动作偏大: {action_max:.2f}")

    tracking_p95 = report["joint_tracking_abs_error"]["p95"]
    if math.isfinite(tracking_p95):
        if tracking_p95 > 0.8:
            add_issue(control, "STOP", f"关节目标跟踪误差 P95={tracking_p95:.3f} rad")
        elif tracking_p95 > 0.4:
            add_issue(control, "WARN", f"关节目标跟踪误差 P95={tracking_p95:.3f} rad")

    external = report["external_control"]
    if external:
        if external["max_exec_vx"] > 0.3001:
            add_issue(control, "STOP", f"执行 vx 超过 0.3m/s: {external['max_exec_vx']:.3f}")
        slew_p95 = external["theory_to_exec_abs_error"]["vx"]["p95"]
        if slew_p95 > 0.10:
            add_issue(control, "WARN", f"理论到执行 vx 偏差 P95={slew_p95:.3f}m/s")

        feedback = external.get("motion_feedback") or external.get("sport_feedback")
        if feedback:
            if feedback["valid_fraction"] < 0.50:
                add_issue(
                    control, "WARN",
                    f"运动反馈有效帧仅 {feedback['valid_fraction']:.1%}，无法可靠评估轨迹",
                )
            else:
                vx_err_p95 = feedback["tracking_abs_error"]["vx"]["p95"]
                vy_mean = feedback["velocity"]["y"]["mean"]
                wz_err_mean = feedback["tracking_error"]["wz"]["mean"]
                if math.isfinite(vx_err_p95) and vx_err_p95 > 0.20:
                    add_issue(control, "WARN", f"实际 vx 跟踪误差 P95={vx_err_p95:.3f}m/s")
                if math.isfinite(vy_mean) and abs(vy_mean) > 0.05:
                    add_issue(control, "WARN", f"实际横向速度均值 {vy_mean:.3f}m/s，存在斜走")
                if math.isfinite(wz_err_mean) and abs(wz_err_mean) > 0.15:
                    add_issue(
                        control, "WARN",
                        f"实际偏航速度误差均值 {wz_err_mean:.3f}rad/s",
                    )

        uwb = external.get("uwb")
        if external["mode"] == "uwb" and uwb:
            if uwb["valid_fraction"] < 0.80:
                add_issue(control, "STOP", f"UWB 有效帧仅 {uwb['valid_fraction']:.1%}")
            elif uwb["valid_fraction"] < 0.95:
                add_issue(control, "WARN", f"UWB 有效帧 {uwb['valid_fraction']:.1%}")

    if not control:
        add_issue(control, "PASS", "外部命令控制链路的时序、姿态和动作量级未发现明显异常")

    raw_max = max(report["cmd_raw_abs_max"].values())
    if raw_max > 100.0:
        add_issue(nav, "STOP", f"导航 cmd_raw 极端异常，绝对值最大 {raw_max:.1f}")
    elif raw_max > 20.0:
        add_issue(nav, "WARN", f"导航 cmd_raw 异常偏大，绝对值最大 {raw_max:.1f}")

    clr = report["clearance"]
    clr_saturated = all(clr[side]["above_0_98_fraction"] > 0.80 for side in clr)
    if clr_saturated:
        add_issue(nav, "STOP", "三路 clearance 超过 80% 帧处于 >=0.98 饱和区")
    elif max(clr[side]["std"] for side in clr) < 0.01:
        add_issue(nav, "WARN", "clearance 变化很小，需确认是否响应障碍")

    depth_invalid = report["depth"]["invalid_fraction"]
    front_invalid = report["depth"]["front_invalid_fraction"]
    if front_invalid["p95"] > 0.50:
        add_issue(nav, "STOP", f"前方深度无效率 P95={front_invalid['p95']:.1%}")
    elif front_invalid["p95"] > 0.25:
        add_issue(nav, "WARN", f"前方深度无效率 P95={front_invalid['p95']:.1%}")
    if depth_invalid["mean"] > 0.40:
        add_issue(nav, "WARN", f"全图平均无效率较高: {depth_invalid['mean']:.1%}")

    if report["fixed_override_detected"]:
        vx = report["cmd"]["vx"]["mean"]
        add_issue(nav, "INFO", f"检测到固定 cmd override，vx={vx:.3f} m/s；导航输出未接管动作")

    if not nav:
        add_issue(nav, "PASS", "未发现阻止导航闭环的明显指标")

    report["control_issues"] = control
    report["navigation_issues"] = nav
    report["control_status"] = worst_level(control)
    report["navigation_status"] = worst_level(nav)
    report["limitations"] = [
        (
            "UWB 速度反馈假设 tag 静止；移动 tag 会污染 vx/vy 估计。"
            if external and external.get("motion_feedback") else
            "SportModeState 在 lowcmd 模式通常不发布，旧日志无法提供实际平面速度。"
            if external else
            "旧 CSV 未记录机器人机体线速度，不能证明实际速度跟踪了 cmd_vel。"
        ),
        (
            "UWB 距离差分只能衡量沿目标方向的接近速度，不等于完整机体速度。"
            if external and external["mode"] == "uwb" else
            "独立 nav 使用固定 goal，不构成随机器人位移更新的到点闭环。"
            if external and external["mode"] == "nav" else
            "固定命令模式只验证 loco，不验证目标导航。"
        ),
        "CSV 未记录 nav/loco latent，不能直接量化 sim-real latent OOD。",
    ]
def worst_level(issues: list[dict]) -> str:
    rank = {"PASS": 0, "INFO": 0, "WARN": 1, "STOP": 2}
    return max(issues, key=lambda item: rank[item["level"]])["level"]


def fmt(value: float, digits: int = 2) -> str:
    return (
        "n/a"
        if not isinstance(value, (int, float)) or not math.isfinite(value)
        else f"{value:.{digits}f}"
    )


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def print_report(report: dict, compact: bool = False) -> None:
    name = Path(report["file"]).name
    print(f"\n=== {name} ===")
    print(
        f"帧数 {report['frames']} | 时长 {report['duration_s']:.1f}s | "
        f"{report['effective_hz']:.2f}Hz | 推理P95 {report['inference_ms']['p95']:.2f}ms | "
        f"miss {report['deadline_misses']}"
    )
    if report["data_quality"]["skipped_rows"]:
        print(f"数据质量: 跳过 {report['data_quality']['skipped_rows']} 个损坏/不完整行")
    print(
        f"cmd均值 [{report['cmd']['vx']['mean']:.3f}, "
        f"{report['cmd']['vy']['mean']:.3f}, {report['cmd']['wz']['mean']:.3f}] | "
        f"raw最大 {max(report['cmd_raw_abs_max'].values()):.1f}"
    )
    external = report["external_control"]
    if external:
        print(
            f"命令模式 {external['mode']} | 执行vx最大 {external['max_exec_vx']:.3f} | "
            f"理论→执行 vx偏差P95 "
            f"{fmt(external['theory_to_exec_abs_error']['vx']['p95'], 3)}m/s"
        )
        sport = external.get("sport_feedback")
        feedback = external.get("motion_feedback")
        if feedback:
            source_counts = feedback["source_counts"]
            source = max(source_counts, key=source_counts.get) if sum(source_counts.values()) else "none"
            print(
                f"运动反馈 {source} 有效 {feedback['valid_fraction']:.1%} | "
                f"vx误差P95 {fmt(feedback['tracking_abs_error']['vx']['p95'], 3)}m/s | "
                f"vy均值 {fmt(feedback['velocity']['y']['mean'], 3)}m/s | "
                f"wz误差P95 {fmt(feedback['tracking_abs_error']['wz']['p95'], 3)}rad/s"
            )
        elif sport:
            print(
                f"Sport反馈有效 {sport['valid_fraction']:.1%} | "
                f"vx误差P95 {fmt(sport['tracking_abs_error']['vx']['p95'], 3)}m/s | "
                f"wz误差P95 {fmt(sport['tracking_abs_error']['wz']['p95'], 3)}rad/s"
            )
        uwb = external.get("uwb")
        if external["mode"] == "uwb" and uwb:
            print(
                f"UWB有效 {uwb['valid_fraction']:.1%} | "
                f"距离均值 {fmt(uwb['distance_m']['mean'], 2)}m | "
                f"接近速度误差P95 {fmt(uwb['closing_abs_error_mps']['p95'], 3)}m/s"
            )
    print(
        f"深度无效 mean/P95 {report['depth']['invalid_fraction']['mean']:.1%}/"
        f"{report['depth']['invalid_fraction']['p95']:.1%} | "
        f"前方无效 mean/P95 {report['depth']['front_invalid_fraction']['mean']:.1%}/"
        f"{report['depth']['front_invalid_fraction']['p95']:.1%}"
    )
    print(
        f"最大倾角 {report['tilt_deg']['max']:.1f}° | "
        f"最大动作 {fmt(report['max_abs_action'], 2)} | "
        f"关节误差P95 {fmt(report['joint_tracking_abs_error']['p95'], 3)} rad"
    )
    if math.isfinite(report["max_abs_joint_effort"]):
        print(
            f"Joint effort max {report['max_abs_joint_effort']:.2f} Nm | "
            f"nominal PD torque max {report['max_abs_pd_torque']:.2f} Nm | "
            f"absolute mechanical power P95 "
            f"{fmt(report['mechanical_power_abs_sum']['p95'], 2)} W"
        )
    print(f"命令与运控: {report['control_status']}")
    for issue in report["control_issues"]:
        print(f"  [{issue['level']}] {issue['message']}")
    nav_label = "nav闭环健康" if external and external["mode"] == "nav" else "nav影子输出健康"
    print(f"{nav_label}: {report['navigation_status']}")
    for issue in report["navigation_issues"]:
        print(f"  [{issue['level']}] {issue['message']}")
    if not compact:
        print("  限制:")
        for limitation in report["limitations"]:
            print(f"    - {limitation}")


def aggregate(reports: list[dict]) -> dict:
    return {
        "files": len(reports),
        "total_frames": sum(report["frames"] for report in reports),
        "total_duration_s": sum(report["duration_s"] for report in reports),
        "control_status_counts": {
            level: sum(report["control_status"] == level for report in reports)
            for level in ("PASS", "WARN", "STOP")
        },
        "navigation_status_counts": {
            level: sum(report["navigation_status"] == level for report in reports)
            for level in ("PASS", "WARN", "STOP")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="分析 Go2 VisionNav 实机 CSV；仅使用 Python 标准库。"
    )
    parser.add_argument("paths", nargs="+", help="CSV、目录或 glob")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归搜索目录")
    parser.add_argument("--json-out", type=Path, help="保存完整 JSON 报告")
    parser.add_argument("--compact", action="store_true", help="隐藏每份日志的限制说明")
    args = parser.parse_args()

    paths = discover_csv(args.paths, args.recursive)
    if not paths:
        print("没有找到 visnav_diag_*.csv", file=sys.stderr)
        return 2

    reports = []
    failures = []
    for path in paths:
        try:
            report = analyze(path)
            reports.append(report)
            print_report(report, compact=args.compact)
        except Exception as exc:
            failures.append({"file": str(path), "error": str(exc)})
            print(f"\n=== {path.name} ===\n[ERROR] {exc}", file=sys.stderr)

    summary = aggregate(reports)
    print("\n=== 总结 ===")
    print(
        f"{summary['files']} 份有效日志，{summary['total_frames']} 帧，"
        f"总时长 {summary['total_duration_s']:.1f}s"
    )
    print(f"外部命令运控状态: {summary['control_status_counts']}")
    print(f"nav影子输出状态: {summary['navigation_status_counts']}")
    if failures:
        print(f"解析失败: {len(failures)} 份")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                json_safe({"summary": summary, "reports": reports, "failures": failures}),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"JSON 已保存: {args.json_out}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
