#!/usr/bin/env python3
"""Build compact, analysis-friendly views from a monitor capture.

The collector stores full frontend responses so no information is lost, but the
raw capture is too large for routine comparison. This postprocessor extracts all
metric series, keeps a full normalized copy, and adds shape-preserving downsampled
and rolling-smoothed views.

Supported inputs:
- overview_capture/sessions/<timestamp>, with groups/*.json
- manual_metric_recorder/sessions/<timestamp>, with points/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


DEPRECATED_EMPTY_METRICS = {
    "reward_deadend_escape",
    "reward_heuristic_navigation",
    "reward_obstacle_evasion",
    "reward_wall_proximity_brake",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path, help="monitor capture session directory")
    parser.add_argument("--downsample-points", type=int, default=80)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--cycle-min-period", type=int, default=2)
    parser.add_argument("--cycle-max-period", type=int, default=8)
    parser.add_argument("--cycle-smooth-window", type=int, default=3)
    parser.add_argument(
        "--no-full-series",
        action="store_true",
        help="skip writing all_metric_series.json; raw group responses still remain unchanged",
    )
    return parser.parse_args()


def _metric_name(result_id: str | None) -> str:
    if not result_id:
        return ""
    stem, sep, suffix = result_id.rpartition("_")
    if sep and suffix.isdigit():
        return stem
    return result_id


def extract_series(capture_dir: Path) -> tuple[dict[str, list[tuple[int, float]]], dict[str, dict[str, Any]]]:
    if (capture_dir / "points").is_dir() or (capture_dir / "metric_requests.jsonl").is_file():
        return extract_manual_series(capture_dir)
    return extract_overview_series(capture_dir)


def label_key(labels: dict[str, Any]) -> str:
    return json.dumps(labels, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_manual_series(capture_dir: Path) -> tuple[dict[str, list[tuple[int, float]]], dict[str, dict[str, Any]]]:
    points_dir = capture_dir / "points"
    if not points_dir.is_dir():
        return {}, {}

    buckets: dict[str, dict[int, list[float]]] = {}
    label_buckets: dict[str, dict[str, dict[int, list[float]]]] = {}
    labels_by_key: dict[str, dict[str, Any]] = {}
    meta: dict[str, dict[str, Any]] = {}
    label_signatures: dict[str, set[str]] = {}

    for point_file in sorted(points_dir.glob("*.jsonl")):
        for line in point_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                name = str(item["name"])
                timestamp = int(item["timestamp"])
                value = float(item["value"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

            buckets.setdefault(name, {}).setdefault(timestamp, []).append(value)
            labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
            label_signature = label_key(labels)
            labels_by_key[label_signature] = labels
            label_buckets.setdefault(name, {}).setdefault(label_signature, {}).setdefault(timestamp, []).append(value)
            label_signatures.setdefault(name, set()).add(label_signature)
            meta.setdefault(
                name,
                {
                    "group": "manual_metric_recorder",
                    "expr": "",
                    "result_ids": [],
                    "source": "points_jsonl",
                },
            )

    series: dict[str, list[tuple[int, float]]] = {}
    for name, timestamp_values in buckets.items():
        series[name] = sorted(
            (timestamp, statistics.fmean(values))
            for timestamp, values in timestamp_values.items()
        )
        meta[name]["label_series_count"] = len(label_signatures.get(name, set()))
        meta[name]["label_series"] = build_label_series_summary(
            label_buckets.get(name, {}),
            labels_by_key,
        )
    return series, meta


def build_label_series_summary(
    label_bucket: dict[str, dict[int, list[float]]],
    labels_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for signature, timestamp_values in sorted(label_bucket.items()):
        points = sorted(
            (timestamp, statistics.fmean(values))
            for timestamp, values in timestamp_values.items()
        )
        if not points:
            continue
        item = summarize(points, {"group": "label_series", "source": "points_jsonl"})
        item["labels"] = labels_by_key.get(signature, {})
        item["label_signature"] = signature
        summaries.append(item)
    return summaries


def extract_overview_series(capture_dir: Path) -> tuple[dict[str, list[tuple[int, float]]], dict[str, dict[str, Any]]]:
    groups_dir = capture_dir / "groups"
    if not groups_dir.is_dir():
        raise FileNotFoundError(f"missing groups directory: {groups_dir}")

    series: dict[str, list[tuple[int, float]]] = {}
    meta: dict[str, dict[str, Any]] = {}

    for group_file in sorted(groups_dir.glob("*.json")):
        data = json.loads(group_file.read_text())
        group_name = data.get("group", {}).get("name") or group_file.stem

        query_expr_by_id: dict[str, str] = {}
        query_name_by_id: dict[str, str] = {}
        for detail in data.get("metric_details", []):
            post_data = detail.get("post_data") or {}
            for query in post_data.get("queries") or []:
                qid = query.get("id")
                if qid:
                    query_expr_by_id[qid] = query.get("expr", "")
                    query_name_by_id[qid] = query.get("name", _metric_name(qid))

            response_json = detail.get("response_json") or {}
            for result in response_json.get("data", {}).get("results", []) or []:
                result_id = result.get("id") or ""
                name = query_name_by_id.get(result_id) or _metric_name(result_id)
                if not name:
                    continue

                values: list[tuple[int, float]] = []
                for item in result.get("items") or []:
                    for point in item.get("values") or []:
                        try:
                            values.append((int(point["timestamp"]), float(point["value"])))
                        except (KeyError, TypeError, ValueError):
                            continue
                if not values:
                    continue

                series.setdefault(name, []).extend(values)
                meta.setdefault(
                    name,
                    {
                        "group": group_name,
                        "expr": query_expr_by_id.get(result_id, ""),
                        "result_ids": [],
                    },
                )
                if result_id and result_id not in meta[name]["result_ids"]:
                    meta[name]["result_ids"].append(result_id)

    for name, values in list(series.items()):
        dedup = {timestamp: value for timestamp, value in values}
        series[name] = sorted(dedup.items())

    return series, meta


def lttb_downsample(points: list[tuple[int, float]], threshold: int) -> list[tuple[int, float]]:
    """Largest-Triangle-Three-Buckets downsampling.

    Keeps visual shape substantially better than uniform sampling while still
    preserving the first and last points.
    """
    data_length = len(points)
    if threshold <= 0 or threshold >= data_length:
        return list(points)
    if threshold < 3:
        return [points[0], points[-1]][:threshold]

    sampled = [points[0]]
    bucket_size = (data_length - 2) / (threshold - 2)
    a = 0

    for i in range(threshold - 2):
        avg_range_start = int(math.floor((i + 1) * bucket_size)) + 1
        avg_range_end = int(math.floor((i + 2) * bucket_size)) + 1
        avg_range_end = min(avg_range_end, data_length)
        avg_range = points[avg_range_start:avg_range_end]
        if avg_range:
            avg_x = sum(p[0] for p in avg_range) / len(avg_range)
            avg_y = sum(p[1] for p in avg_range) / len(avg_range)
        else:
            avg_x, avg_y = points[-1]

        range_offs = int(math.floor(i * bucket_size)) + 1
        range_to = int(math.floor((i + 1) * bucket_size)) + 1
        range_to = min(range_to, data_length - 1)

        point_a_x, point_a_y = points[a]
        max_area = -1.0
        next_a = range_offs
        for idx in range(range_offs, range_to):
            point_x, point_y = points[idx]
            area = abs(
                (point_a_x - avg_x) * (point_y - point_a_y)
                - (point_a_x - point_x) * (avg_y - point_a_y)
            ) * 0.5
            if area > max_area:
                max_area = area
                next_a = idx

        sampled.append(points[next_a])
        a = next_a

    sampled.append(points[-1])
    return sampled


def centered_moving_average(points: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    if window <= 1 or len(points) <= 2:
        return list(points)
    half = window // 2
    smoothed: list[tuple[int, float]] = []
    values = [value for _, value in points]
    for idx, (timestamp, _) in enumerate(points):
        start = max(0, idx - half)
        end = min(len(points), idx + half + 1)
        smoothed.append((timestamp, sum(values[start:end]) / (end - start)))
    return smoothed


def _safe_corr(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    var_a = sum((value - mean_a) ** 2 for value in a)
    var_b = sum((value - mean_b) ** 2 for value in b)
    if var_a <= 1e-24 or var_b <= 1e-24:
        return 0.0
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    return cov / math.sqrt(var_a * var_b)


def infer_cycle_period(
    points: list[tuple[int, float]],
    min_period: int,
    max_period: int,
    trend_window: int,
) -> tuple[int | None, float]:
    """Infer a short periodic sampling artifact from detrended residuals.

    The monitor often emits a low/low/low/high pattern when some terrain buckets
    finish near max episode length while others terminate early.  Detecting the
    period on detrended residuals avoids falsely treating slow learning curves as
    cyclic.
    """
    if len(points) < max(18, max_period * 4):
        return None, 0.0

    trend = centered_moving_average(points, max(trend_window, max_period * 3))
    residual = [value - trend_value for (_, value), (_, trend_value) in zip(points, trend)]
    if max(residual) - min(residual) <= 1e-12:
        return None, 0.0

    best_period: int | None = None
    best_corr = 0.0
    for period in range(max(2, min_period), max_period + 1):
        corr = _safe_corr(residual[:-period], residual[period:])
        if corr > best_corr:
            best_corr = corr
            best_period = period

    if best_period is None or best_corr < 0.45:
        return None, best_corr
    return best_period, best_corr


def cycle_bias_corrected_smoothing(
    points: list[tuple[int, float]],
    min_period: int,
    max_period: int,
    trend_window: int,
    smooth_window: int,
) -> tuple[list[tuple[int, float]], dict[str, Any]]:
    """Remove a detected short-period phase bias, then lightly smooth.

    This is deliberately conservative: it writes a separate analysis view and
    preserves the original series.  It subtracts a robust per-phase residual
    median rather than using a wide moving average, so real long-horizon changes
    are retained better than with a large smoothing window.
    """
    if len(points) <= 2:
        return list(points), {"method": "copy", "reason": "too_few_points"}

    period, corr = infer_cycle_period(points, min_period, max_period, trend_window)
    if period is None:
        smoothed = centered_moving_average(points, smooth_window)
        return smoothed, {
            "method": "centered_moving_average",
            "reason": "no_strong_cycle",
            "best_corr": corr,
            "smooth_window": smooth_window,
        }

    trend = centered_moving_average(points, max(trend_window, period * 3))
    residual = [value - trend_value for (_, value), (_, trend_value) in zip(points, trend)]
    phase_residuals: list[list[float]] = [[] for _ in range(period)]
    for idx, value in enumerate(residual):
        phase_residuals[idx % period].append(value)

    phase_bias = [
        statistics.median(values) if values else 0.0
        for values in phase_residuals
    ]
    # Keep the global level unchanged; only remove relative phase bias.
    center = statistics.fmean(phase_bias)
    phase_bias = [value - center for value in phase_bias]

    corrected = [
        (timestamp, value - phase_bias[idx % period])
        for idx, (timestamp, value) in enumerate(points)
    ]
    smoothed = centered_moving_average(corrected, smooth_window)
    return smoothed, {
        "method": "cycle_bias_corrected",
        "period_points": period,
        "period_seconds": int((points[1][0] - points[0][0]) * period / 1000) if len(points) > 1 else None,
        "residual_lag_corr": corr,
        "trend_window": max(trend_window, period * 3),
        "smooth_window": smooth_window,
        "phase_bias": phase_bias,
    }


def cycle_block_aggregate(
    points: list[tuple[int, float]],
    min_period: int,
    max_period: int,
    trend_window: int,
) -> tuple[list[tuple[int, float]], dict[str, Any]]:
    """Aggregate one detected monitor cycle into one point.

    For sawtooth counters, each phase can represent a different completion
    bucket.  Averaging a full period is less distorting than smoothing across
    arbitrary windows because it keeps every phase exactly once.
    """
    if len(points) <= 2:
        return list(points), {"method": "copy", "reason": "too_few_points"}

    period, corr = infer_cycle_period(points, min_period, max_period, trend_window)
    if period is None:
        return list(points), {
            "method": "copy",
            "reason": "no_strong_cycle",
            "best_corr": corr,
        }

    aggregated: list[tuple[int, float]] = []
    for start in range(0, len(points), period):
        block = points[start : start + period]
        if len(block) < max(2, period // 2):
            continue
        timestamp = block[len(block) // 2][0]
        value = statistics.fmean(value for _, value in block)
        aggregated.append((timestamp, value))

    return aggregated, {
        "method": "cycle_block_mean",
        "period_points": period,
        "period_seconds": int((points[1][0] - points[0][0]) * period / 1000) if len(points) > 1 else None,
        "residual_lag_corr": corr,
        "raw_points": len(points),
        "aggregated_points": len(aggregated),
    }


def summarize(points: list[tuple[int, float]], meta: dict[str, Any]) -> dict[str, Any]:
    values = [value for _, value in points]
    first20 = values[: min(20, len(values))]
    last20 = values[-min(20, len(values)) :]
    max_idx = max(range(len(values)), key=values.__getitem__)
    min_idx = min(range(len(values)), key=values.__getitem__)

    def ts_text(timestamp: int) -> str:
        return datetime.fromtimestamp(timestamp / 1000).isoformat(timespec="seconds")

    return {
        "group": meta.get("group", ""),
        "expr": meta.get("expr", ""),
        "source": meta.get("source", "overview_groups"),
        "label_series_count": meta.get("label_series_count"),
        "count": len(points),
        "start_time": ts_text(points[0][0]),
        "end_time": ts_text(points[-1][0]),
        "first": values[0],
        "mid": values[len(values) // 2],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "first20_avg": statistics.fmean(first20),
        "last20_avg": statistics.fmean(last20),
        "delta": values[-1] - values[0],
        "last20_delta": last20[-1] - last20[0] if len(last20) > 1 else 0.0,
        "max_time": ts_text(points[max_idx][0]),
        "min_time": ts_text(points[min_idx][0]),
    }


def empty_summary(name: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or {}
    return {
        "group": meta.get("group", ""),
        "expr": meta.get("expr", ""),
        "source": meta.get("source", "missing_points"),
        "label_series_count": meta.get("label_series_count", 0),
        "status": "empty",
        "count": 0,
        "start_time": None,
        "end_time": None,
        "first": None,
        "mid": None,
        "last": None,
        "min": None,
        "max": None,
        "mean": None,
        "first20_avg": None,
        "last20_avg": None,
        "delta": None,
        "last20_delta": None,
        "max_time": None,
        "min_time": None,
    }


def encode_points(points: list[tuple[int, float]]) -> list[dict[str, float | int]]:
    return [{"ts": timestamp, "value": value} for timestamp, value in points]


def recorder_metric_names(capture_dir: Path) -> list[str]:
    summary_path = capture_dir / "summary.json"
    if not summary_path.is_file():
        return []
    try:
        data = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return []
    names = data.get("metric_names")
    if not isinstance(names, list):
        return []
    return sorted(str(name) for name in names if name)


def build_metric_inventory(
    metric_names: list[str],
    summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    all_names = sorted(set(metric_names) | set(summary))
    metrics = {}
    for name in all_names:
        item = summary.get(name)
        count = int(item.get("count") or 0) if item else 0
        metrics[name] = {
            "status": "has_points" if count > 0 else "empty",
            "count": count,
            "start_time": item.get("start_time") if item else None,
            "end_time": item.get("end_time") if item else None,
            "label_series_count": item.get("label_series_count") if item else 0,
        }
    return {
        "metric_count": len(all_names),
        "has_points_count": sum(1 for item in metrics.values() if item["status"] == "has_points"),
        "empty_count": sum(1 for item in metrics.values() if item["status"] == "empty"),
        "metrics": metrics,
    }


def build_ai_readable_metrics(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for name, item in sorted(summary.items()):
        metrics[name] = {
            "metric_name": name,
            "status": item.get("status", "has_points" if item.get("count", 0) > 0 else "empty"),
            "count": item.get("count", 0),
            "start_time": item.get("start_time"),
            "end_time": item.get("end_time"),
            "first": item.get("first"),
            "last": item.get("last"),
            "min": item.get("min"),
            "max": item.get("max"),
            "mean": item.get("mean"),
            "delta": item.get("delta"),
            "recent_delta": item.get("last20_delta"),
            "first20_avg": item.get("first20_avg"),
            "last20_avg": item.get("last20_avg"),
            "label_series_count": item.get("label_series_count"),
        }
    return {
        "format": "fact_table",
        "metric_count": len(metrics),
        "metrics": metrics,
    }


def build_label_series_output(meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for name, item in sorted(meta.items()):
        label_series = item.get("label_series") or []
        if label_series:
            metrics[name] = {
                "label_series_count": len(label_series),
                "series": label_series,
            }
    return {
        "metric_count": len(metrics),
        "metrics": metrics,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6g}"
        return str(value)
    return str(value)


def build_markdown_report(capture_dir: Path, summary: dict[str, dict[str, Any]], overview: dict[str, Any]) -> str:
    non_empty = {name: item for name, item in summary.items() if item.get("count", 0) > 0}
    zero_metrics = sorted(
        name for name, item in summary.items()
        if item.get("count", 0) == 0 and name not in DEPRECATED_EMPTY_METRICS
    )
    recent_delta = sorted(non_empty.items(), key=lambda kv: abs(float(kv[1].get("last20_delta") or 0.0)), reverse=True)[:12]
    latest = sorted(non_empty.items(), key=lambda kv: str(kv[1].get("end_time", "")), reverse=True)[:12]

    focus_prefixes = (
        "total_score",
        "time_score",
        "pose_score",
        "energy_score",
        "completed_count",
        "timeout_count",
        "abnormal_count",
        "reward_",
        "vel_curriculum",
        "levelmix",
        "obs_",
        "mean_episode",
        "train_global_step",
    )
    focus = [
        (name, item)
        for name, item in non_empty.items()
        if name.startswith(focus_prefixes)
    ]
    focus = sorted(focus, key=lambda kv: kv[0])

    lines = [
        "# Preprocessed Metrics Summary",
        "",
        f"- capture_dir: `{capture_dir}`",
        f"- series_count: {overview['series_count']}",
        f"- source: {overview.get('source')}",
        f"- outputs: `{overview['outputs']['ai_readable']}`, `{overview['outputs']['inventory']}`, `{overview['outputs']['summary']}`, `{overview['outputs']['lttb']}`, `{overview['outputs']['smoothed']}`, `{overview['outputs']['cycle_smoothed']}`, `{overview['outputs']['cycle_blocks']}`",
        "",
        "This report is a factual preprocessing view only. It does not classify training quality or make optimization recommendations.",
        "",
        "## Latest Samples",
        "",
        "| metric | count | last | delta | end_time |",
        "|---|---:|---:|---:|---|",
    ]
    for name, item in latest:
        lines.append(
            f"| `{name}` | {item['count']} | {_fmt(item['last'])} | {_fmt(item['delta'])} | {item['end_time']} |"
        )

    lines += [
        "",
        "## Largest Absolute Recent Delta",
        "",
        "| metric | recent_delta | last | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in recent_delta:
        lines.append(
            f"| `{name}` | {_fmt(item['last20_delta'])} | {_fmt(item['last'])} | {_fmt(item['min'])} | {_fmt(item['max'])} |"
        )

    lines += [
        "",
        "## Focus Metrics",
        "",
        "| metric | count | first | last | delta | last20_avg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in focus[:80]:
        lines.append(
            f"| `{name}` | {item['count']} | {_fmt(item['first'])} | {_fmt(item['last'])} | "
            f"{_fmt(item['delta'])} | {_fmt(item['last20_avg'])} |"
        )

    if zero_metrics:
        lines += [
            "",
            "## Empty Metrics",
            "",
            ", ".join(f"`{name}`" for name in zero_metrics),
        ]
    return "\n".join(lines) + "\n"


def print_terminal_summary(summary: dict[str, dict[str, Any]], overview: dict[str, Any]) -> None:
    non_empty = {name: item for name, item in summary.items() if item.get("count", 0) > 0}
    zero_count = len(summary) - len(non_empty)
    print("\n[postprocess] preprocessing outputs generated")
    print(f"[postprocess] series={len(summary)} non_empty={len(non_empty)} empty={zero_count}")
    for name in ("train_global_step", "total_score", "completed_count", "timeout_count", "abnormal_count"):
        item = summary.get(name)
        if item:
            print(
                f"[postprocess] {name}: count={item['count']} "
                f"first={_fmt(item['first'])} last={_fmt(item['last'])} delta={_fmt(item['delta'])}"
            )
    deltas = sorted(non_empty.items(), key=lambda kv: abs(float(kv[1].get("last20_delta") or 0.0)), reverse=True)[:8]
    print("[postprocess] largest absolute recent_delta:")
    for name, item in deltas:
        print(f"  {name}: recent_delta={_fmt(item['last20_delta'])} last={_fmt(item['last'])}")
    print(f"[postprocess] ai_readable={overview['outputs']['ai_readable']}")
    print(f"[postprocess] inventory={overview['outputs']['inventory']}")
    print(f"[postprocess] report={overview['outputs']['report']}")


def main() -> None:
    args = parse_args()
    capture_dir = args.capture_dir.resolve()
    series, meta = extract_series(capture_dir)

    summary = {
        name: summarize(points, meta.get(name, {}))
        for name, points in sorted(series.items())
    }
    for item in summary.values():
        item["status"] = "has_points"
    for name in recorder_metric_names(capture_dir):
        summary.setdefault(
            name,
            empty_summary(
                name,
                meta.get(name) or {"group": "manual_metric_recorder", "source": "recorder_summary"},
            ),
        )

    inventory = build_metric_inventory(recorder_metric_names(capture_dir), summary)
    ai_readable = build_ai_readable_metrics(summary)
    label_series_output = build_label_series_output(meta)
    downsampled = {
        name: encode_points(lttb_downsample(points, args.downsample_points))
        for name, points in sorted(series.items())
    }
    smoothed = {
        name: encode_points(centered_moving_average(points, args.smooth_window))
        for name, points in sorted(series.items())
    }
    cycle_smoothed: dict[str, list[dict[str, float | int]]] = {}
    cycle_diagnostics: dict[str, dict[str, Any]] = {}
    cycle_blocks: dict[str, list[dict[str, float | int]]] = {}
    cycle_block_diagnostics: dict[str, dict[str, Any]] = {}
    for name, points in sorted(series.items()):
        processed, diagnostic = cycle_bias_corrected_smoothing(
            points,
            min_period=args.cycle_min_period,
            max_period=args.cycle_max_period,
            trend_window=args.smooth_window,
            smooth_window=args.cycle_smooth_window,
        )
        cycle_smoothed[name] = encode_points(processed)
        cycle_diagnostics[name] = diagnostic
        block_points, block_diagnostic = cycle_block_aggregate(
            points,
            min_period=args.cycle_min_period,
            max_period=args.cycle_max_period,
            trend_window=args.smooth_window,
        )
        cycle_blocks[name] = encode_points(block_points)
        cycle_block_diagnostics[name] = block_diagnostic

    (capture_dir / "all_metric_series_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    (capture_dir / "ai_readable_metrics.json").write_text(
        json.dumps(ai_readable, ensure_ascii=False, indent=2)
    )
    (capture_dir / "metric_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2)
    )
    (capture_dir / "label_series_summary.json").write_text(
        json.dumps(label_series_output, ensure_ascii=False, indent=2)
    )
    (capture_dir / "all_metric_series_lttb.json").write_text(
        json.dumps(downsampled, ensure_ascii=False)
    )
    (capture_dir / "all_metric_series_smoothed.json").write_text(
        json.dumps(smoothed, ensure_ascii=False)
    )
    (capture_dir / "all_metric_series_cycle_smoothed.json").write_text(
        json.dumps(cycle_smoothed, ensure_ascii=False)
    )
    (capture_dir / "cycle_smoothing_diagnostics.json").write_text(
        json.dumps(cycle_diagnostics, ensure_ascii=False, indent=2)
    )
    (capture_dir / "all_metric_series_cycle_blocks.json").write_text(
        json.dumps(cycle_blocks, ensure_ascii=False)
    )
    (capture_dir / "cycle_block_diagnostics.json").write_text(
        json.dumps(cycle_block_diagnostics, ensure_ascii=False, indent=2)
    )
    if not args.no_full_series:
        (capture_dir / "all_metric_series.json").write_text(
            json.dumps(
                {name: encode_points(points) for name, points in sorted(series.items())},
                ensure_ascii=False,
            )
        )

    group_counts: dict[str, int] = {}
    for item in summary.values():
        group_counts[item["group"]] = group_counts.get(item["group"], 0) + 1
    overview = {
        "capture_dir": str(capture_dir),
        "source": "manual_points" if (capture_dir / "metric_requests.jsonl").is_file() else "overview_groups",
        "series_count": len(series),
        "metric_count": len(summary),
        "empty_metric_count": sum(1 for item in summary.values() if item.get("count", 0) == 0),
        "downsample_points": args.downsample_points,
        "smooth_window": args.smooth_window,
        "cycle_min_period": args.cycle_min_period,
        "cycle_max_period": args.cycle_max_period,
        "cycle_smooth_window": args.cycle_smooth_window,
        "group_counts": dict(sorted(group_counts.items())),
        "outputs": {
            "ai_readable": "ai_readable_metrics.json",
            "inventory": "metric_inventory.json",
            "label_series": "label_series_summary.json",
            "summary": "all_metric_series_summary.json",
            "full_series": None if args.no_full_series else "all_metric_series.json",
            "lttb": "all_metric_series_lttb.json",
            "smoothed": "all_metric_series_smoothed.json",
            "cycle_smoothed": "all_metric_series_cycle_smoothed.json",
            "cycle_diagnostics": "cycle_smoothing_diagnostics.json",
            "cycle_blocks": "all_metric_series_cycle_blocks.json",
            "cycle_block_diagnostics": "cycle_block_diagnostics.json",
            "report": "analysis_report.md",
        },
    }
    report = build_markdown_report(capture_dir, summary, overview)
    (capture_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    (capture_dir / "postprocess_summary.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2)
    )
    print(json.dumps(overview, ensure_ascii=False, indent=2))
    print_terminal_summary(summary, overview)


if __name__ == "__main__":
    main()
