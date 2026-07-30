#!/usr/bin/env python3
"""Compare historical walking logs with a hardened-runtime failure log."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


JOINT_COUNT = 12
DEFAULT_JOINT_POS = [
    0.1, -0.1, 0.1, -0.1,
    0.8, 0.8, 1.0, 1.0,
    -1.5, -1.5, -1.5, -1.5,
]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def load_rows(path: Path) -> list[dict[str, float]]:
    required = {"t_ms", "vx", "vy", "wz"}
    required.update(f"q{i}" for i in range(JOINT_COUNT))
    required.update(f"action{i}" for i in range(JOINT_COUNT))
    required.update(f"target{i}" for i in range(JOINT_COUNT))

    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(line.replace("\0", "") for line in handle)
        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV has no header")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if not row or not row.get("t_ms", "").strip():
                continue
            parsed: dict[str, float] = {}
            try:
                for key, text in row.items():
                    if key is None or text in (None, ""):
                        continue
                    value = float(text)
                    if not math.isfinite(value):
                        raise ValueError(f"non-finite {key}={text!r}")
                    parsed[key] = value
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            rows.append(parsed)
    if not rows:
        raise ValueError(f"{path}: CSV has no data rows")
    return rows


def command_magnitude(row: dict[str, float]) -> float:
    return max(abs(row[axis]) for axis in ("vx", "vy", "wz"))


def motion_indices(rows: list[dict[str, float]], threshold: float) -> list[int]:
    return [index for index, row in enumerate(rows) if command_magnitude(row) > threshold]


def max_joint_abs(row: dict[str, float], prefix: str) -> float:
    return max(abs(row[f"{prefix}{joint}"]) for joint in range(JOINT_COUNT))


def max_joint_delta(
    current: dict[str, float], previous: dict[str, float], prefix: str
) -> float:
    return max(
        abs(current[f"{prefix}{joint}"] - previous[f"{prefix}{joint}"])
        for joint in range(JOINT_COUNT)
    )


def runtime_distribution(
    rows: list[dict[str, float]],
    threshold: float,
    raw_action_limit: float,
    target_step_limit: float,
) -> dict:
    indices = motion_indices(rows, threshold)
    if not indices:
        raise ValueError("log contains no non-zero motion frames")

    raw_abs = [max_joint_abs(rows[index], "action") for index in indices]
    target_steps = [
        max_joint_delta(rows[current], rows[previous], "target")
        for previous, current in zip(indices, indices[1:])
        if current == previous + 1
    ]
    first = rows[indices[0]]
    deviations = [
        first[f"q{joint}"] - DEFAULT_JOINT_POS[joint]
        for joint in range(JOINT_COUNT)
    ]
    return {
        "frames": len(rows),
        "motion_frames": len(indices),
        "duration_s": (rows[-1]["t_ms"] - rows[0]["t_ms"]) / 1000.0,
        "raw_action_abs": summary(raw_abs),
        "raw_action_frames_over_limit": sum(
            value > raw_action_limit for value in raw_abs
        ),
        "raw_action_fraction_over_limit": sum(
            value > raw_action_limit for value in raw_abs
        ) / len(raw_abs),
        "target_step": summary(target_steps),
        "target_step_frames_over_limit": sum(
            value > target_step_limit + 1.0e-9 for value in target_steps
        ),
        "target_step_fraction_over_limit": (
            sum(value > target_step_limit + 1.0e-9 for value in target_steps)
            / len(target_steps)
            if target_steps else None
        ),
        "motion_start": {
            "row_index": indices[0],
            "t_ms": first["t_ms"],
            "command": [first[axis] for axis in ("vx", "vy", "wz")],
            "max_joint_deviation_from_default_rad": max(map(abs, deviations)),
            "joint_deviation_from_default_rad": deviations,
            "front_calf_q_rad": [first["q8"], first["q9"]],
            "front_calf_deviation_rad": [deviations[8], deviations[9]],
        },
    }


def clamp_delta(previous: float, candidate: float, limit: float) -> float:
    return previous + max(-limit, min(limit, candidate - previous))


def first_motion_segment(
    rows: list[dict[str, float]], threshold: float
) -> list[int]:
    indices = motion_indices(rows, threshold)
    if not indices:
        raise ValueError("failed log contains no non-zero motion frames")
    segment = [indices[0]]
    for index in indices[1:]:
        if index != segment[-1] + 1:
            break
        segment.append(index)
    return segment


def counterfactual_distortion(
    rows: list[dict[str, float]],
    threshold: float,
    action_scale: float,
    entry_blend_s: float,
    target_step_limit: float,
) -> dict:
    segment = first_motion_segment(rows, threshold)
    start = segment[0]
    entry_row = rows[max(0, start - 1)]
    entry_target = [entry_row[f"target{joint}"] for joint in range(JOINT_COUNT)]
    limited_target = list(entry_target)
    start_ms = rows[start]["t_ms"]

    blend_distortion: list[float] = []
    limiter_distortion: list[float] = []
    combined_distortion: list[float] = []
    simulated_log_error: list[float] = []
    prelimit_steps: list[float] = []

    for index in segment:
        row = rows[index]
        elapsed_s = max(0.0, (row["t_ms"] - start_ms) / 1000.0)
        if entry_blend_s <= 0.0:
            alpha = 1.0
        else:
            blend_x = min(elapsed_s / entry_blend_s, 1.0)
            alpha = blend_x * blend_x * (3.0 - 2.0 * blend_x)
        policy_target = [
            DEFAULT_JOINT_POS[joint] + action_scale * row[f"action{joint}"]
            for joint in range(JOINT_COUNT)
        ]
        blended_target = [
            entry_target[joint] + alpha * (policy_target[joint] - entry_target[joint])
            for joint in range(JOINT_COUNT)
        ]
        prelimit_steps.append(max(
            abs(blended_target[joint] - limited_target[joint])
            for joint in range(JOINT_COUNT)
        ))
        next_limited = [
            clamp_delta(limited_target[joint], blended_target[joint], target_step_limit)
            for joint in range(JOINT_COUNT)
        ]
        logged_target = [row[f"target{joint}"] for joint in range(JOINT_COUNT)]
        blend_distortion.append(max(
            abs(policy_target[joint] - blended_target[joint])
            for joint in range(JOINT_COUNT)
        ))
        limiter_distortion.append(max(
            abs(blended_target[joint] - next_limited[joint])
            for joint in range(JOINT_COUNT)
        ))
        combined_distortion.append(max(
            abs(policy_target[joint] - next_limited[joint])
            for joint in range(JOINT_COUNT)
        ))
        simulated_log_error.append(max(
            abs(next_limited[joint] - logged_target[joint])
            for joint in range(JOINT_COUNT)
        ))
        limited_target = next_limited

    return {
        "motion_frames": len(segment),
        "entry_blend_s": entry_blend_s,
        "target_step_limit_rad": target_step_limit,
        "prelimit_step": summary(prelimit_steps),
        "prelimit_fraction_over_limiter": sum(
            value > target_step_limit + 1.0e-9 for value in prelimit_steps
        ) / len(prelimit_steps),
        "blend_only_distortion_rad": summary(blend_distortion),
        "limiter_only_distortion_rad": summary(limiter_distortion),
        "combined_policy_target_distortion_rad": summary(combined_distortion),
        "simulation_vs_logged_target_error_rad": summary(simulated_log_error),
    }


def front_calf_progression(
    rows: list[dict[str, float]], threshold: float
) -> dict:
    segment = first_motion_segment(rows, threshold)
    start_ms = rows[segment[0]]["t_ms"]
    samples = []
    wanted_s = [0.0, 0.25, 0.5, 1.0]
    chosen: set[int] = set()
    for elapsed in wanted_s:
        index = min(
            segment,
            key=lambda item: abs((rows[item]["t_ms"] - start_ms) / 1000.0 - elapsed),
        )
        chosen.add(index)
    chosen.add(segment[-1])
    for index in sorted(chosen):
        row = rows[index]
        sample = {
            "elapsed_s": (row["t_ms"] - start_ms) / 1000.0,
            "action": [row["action8"], row["action9"]],
            "q_rad": [row["q8"], row["q9"]],
            "target_rad": [row["target8"], row["target9"]],
            "tracking_error_rad": [
                abs(row["target8"] - row["q8"]),
                abs(row["target9"] - row["q9"]),
            ],
        }
        if "tau8" in row and "tau9" in row:
            sample["tau_nm"] = [row["tau8"], row["tau9"]]
        samples.append(sample)
    return {"joint_indices": [8, 9], "samples": samples}


def aggregate_historical(items: Iterable[dict]) -> dict:
    items = list(items)
    raw_max = [item["raw_action_abs"]["max"] for item in items]
    step_p95 = [item["target_step"]["p95"] for item in items]
    step_max = [item["target_step"]["max"] for item in items]
    return {
        "runs": len(items),
        "raw_action_max_range": [min(raw_max), max(raw_max)],
        "target_step_p95_range": [min(step_p95), max(step_p95)],
        "target_step_max_range": [min(step_max), max(step_max)],
        "weighted_raw_action_fraction_over_limit": (
            sum(item["raw_action_frames_over_limit"] for item in items)
            / sum(item["motion_frames"] for item in items)
        ),
        "weighted_target_step_fraction_over_limit": (
            sum(item["target_step_frames_over_limit"] for item in items)
            / sum(item["motion_frames"] - 1 for item in items)
        ),
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_text(report: dict) -> None:
    print("Historical successful runtime")
    print("log                              raw max   target p95/max   >0.03 frames")
    for path, item in report["historical"].items():
        name = Path(path).name
        fraction = item["target_step_fraction_over_limit"]
        print(
            f"{name:32} {fmt(item['raw_action_abs']['max']):>7}   "
            f"{fmt(item['target_step']['p95'])}/{fmt(item['target_step']['max'])}      "
            f"{fmt(100.0 * fraction, 1)}%"
        )

    failed = report["failed"]
    start = failed["motion_start"]
    distortion = report["failed_counterfactual"]
    print("\nHardened failed runtime")
    print(
        f"raw max={fmt(failed['raw_action_abs']['max'])}, "
        f"target step p95/max={fmt(failed['target_step']['p95'])}/"
        f"{fmt(failed['target_step']['max'])} rad/frame"
    )
    print(
        "motion-start front calves q="
        f"[{fmt(start['front_calf_q_rad'][0])}, {fmt(start['front_calf_q_rad'][1])}] rad, "
        "deviation from defaults="
        f"[{fmt(start['front_calf_deviation_rad'][0])}, "
        f"{fmt(start['front_calf_deviation_rad'][1])}] rad"
    )
    print(
        "blend+limiter combined policy-target distortion p95/max="
        f"{fmt(distortion['combined_policy_target_distortion_rad']['p95'])}/"
        f"{fmt(distortion['combined_policy_target_distortion_rad']['max'])} rad"
    )
    print(
        "pre-limit frames exceeding 0.03 rad="
        f"{fmt(100.0 * distortion['prelimit_fraction_over_limiter'], 1)}%, "
        "simulation-vs-log max error="
        f"{fmt(distortion['simulation_vs_logged_target_error_rad']['max'], 6)} rad"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical", action="append", required=True, type=Path,
        help="historical successful visloco CSV; repeat for multiple runs",
    )
    parser.add_argument("--failed", required=True, type=Path)
    parser.add_argument("--raw-action-limit", type=float, default=6.0)
    parser.add_argument("--target-step-limit", type=float, default=0.03)
    parser.add_argument("--entry-blend-s", type=float, default=1.0)
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--motion-threshold", type=float, default=0.02)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.raw_action_limit <= 0.0 or args.target_step_limit <= 0.0:
        raise ValueError("safety limits must be positive")
    historical: dict[str, dict] = {}
    for path in args.historical:
        rows = load_rows(path)
        historical[str(path)] = runtime_distribution(
            rows, args.motion_threshold,
            args.raw_action_limit, args.target_step_limit,
        )
    failed_rows = load_rows(args.failed)
    failed = runtime_distribution(
        failed_rows, args.motion_threshold,
        args.raw_action_limit, args.target_step_limit,
    )
    report = {
        "parameters": {
            "raw_action_limit": args.raw_action_limit,
            "target_step_limit_rad": args.target_step_limit,
            "entry_blend_s": args.entry_blend_s,
            "action_scale": args.action_scale,
            "motion_threshold": args.motion_threshold,
            "default_joint_pos": DEFAULT_JOINT_POS,
        },
        "historical": historical,
        "historical_aggregate": aggregate_historical(historical.values()),
        "failed": failed,
        "failed_counterfactual": counterfactual_distortion(
            failed_rows, args.motion_threshold, args.action_scale,
            args.entry_blend_s, args.target_step_limit,
        ),
        "failed_front_calf_progression": front_calf_progression(
            failed_rows, args.motion_threshold
        ),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
