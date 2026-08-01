#!/usr/bin/env python3
"""Strict sim2real action, frequency, symmetry, and tracking diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from strict_contract import ContractError, band_energy


LEGS = {"FL": (0, 4, 8), "FR": (1, 5, 9), "RL": (2, 6, 10), "RR": (3, 7, 11)}
LAYERS = (
    "model_raw_action",
    "clipped_raw_action",
    "requested_target",
    "slew_target",
    "physical_target",
    "applied_target",
    "executed_raw_action",
)


def _load(path: str) -> tuple[list[str], dict[str, np.ndarray]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ContractError(f"empty CSV: {path}")
    columns: dict[str, np.ndarray] = {}
    for name in fields:
        values = []
        for row in rows:
            try:
                values.append(float(row[name]))
            except (TypeError, ValueError):
                values.append(float("nan"))
        columns[name] = np.asarray(values, dtype=np.float64)
    return fields, columns


def _matrix(columns: dict[str, np.ndarray], prefix: str) -> np.ndarray | None:
    names = [f"{prefix}{i}" for i in range(12)]
    if not all(name in columns for name in names):
        return None
    values = np.stack([columns[name] for name in names], axis=1)
    if not np.isfinite(values).all():
        raise ContractError(f"non-finite values in {prefix} layer")
    return values


def _layer_report(values: np.ndarray) -> dict:
    joints = []
    for joint in range(12):
        metrics = band_energy(values[:, joint])
        delta = np.diff(values[:, joint])
        metrics.update(
            joint=joint,
            max_abs=float(np.max(np.abs(values[:, joint]))),
            max_abs_delta=float(np.max(np.abs(delta))) if delta.size else 0.0,
            p95_abs_delta=float(np.percentile(np.abs(delta), 95)) if delta.size else 0.0,
        )
        joints.append(metrics)
    leg_delta_rms = {}
    leg_hf_energy = {}
    for leg, indices in LEGS.items():
        delta = np.diff(values[:, indices], axis=0)
        leg_delta_rms[leg] = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
        leg_hf_energy[leg] = float(sum(joints[index]["10_15_hz"] + joints[index]["15_25_hz"] for index in indices))
    leg_values = np.asarray(list(leg_hf_energy.values()))
    median_leg = float(np.median(leg_values)) if leg_values.size else 0.0
    worst_leg = max(leg_hf_energy, key=leg_hf_energy.get)
    left = leg_hf_energy["FL"] + leg_hf_energy["RL"]
    right = leg_hf_energy["FR"] + leg_hf_energy["RR"]
    return {
        "joints": joints,
        "leg_delta_rms": leg_delta_rms,
        "leg_high_frequency_energy": leg_hf_energy,
        "worst_leg": worst_leg,
        "single_leg_energy_ratio_to_median": (
            leg_hf_energy[worst_leg] / median_leg if median_leg > 1e-12 else 0.0
        ),
        "left_right_high_frequency_asymmetry": abs(left - right) / max(left + right, 1e-12),
    }


def _scalar_stats(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"count": 0, "min": None, "max": None, "mean": None, "p95": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
    }


def analyze(path: str) -> dict:
    fields, columns = _load(path)
    report: dict = {"source": str(path), "frames": len(next(iter(columns.values()))), "layers": {}}
    for layer in LAYERS:
        values = _matrix(columns, layer)
        if values is not None:
            report["layers"][layer] = _layer_report(values)
    # Successful historical sim2real_test_loco logs used action*/target* names.
    if "model_raw_action" not in report["layers"]:
        legacy = _matrix(columns, "action")
        if legacy is not None:
            report["layers"]["model_raw_action"] = _layer_report(legacy)
    if "applied_target" not in report["layers"]:
        legacy = _matrix(columns, "target")
        if legacy is not None:
            report["layers"]["applied_target"] = _layer_report(legacy)

    q = _matrix(columns, "q")
    applied = _matrix(columns, "applied_target")
    if q is not None and applied is not None:
        error = np.abs(applied - q)
        report["tracking_error_rad"] = {
            "max": float(error.max()),
            "mean": float(error.mean()),
            "p95": float(np.percentile(error, 95)),
            "per_joint_max": error.max(axis=0).tolist(),
        }
    dq = _matrix(columns, "dq")
    tau = _matrix(columns, "tau")
    if dq is not None:
        report["joint_velocity_abs_peak"] = np.max(np.abs(dq), axis=0).tolist()
    if tau is not None:
        report["joint_effort_abs_peak"] = np.max(np.abs(tau), axis=0).tolist()

    if "safety_modified_rate" in columns:
        report["safety_modified_rate"] = _scalar_stats(columns["safety_modified_rate"])
    elif "safety_modified_count" in columns:
        report["safety_modified_rate"] = _scalar_stats(columns["safety_modified_count"] / 12.0)
    for name in (
        "lowstate_age_ms",
        "depth_age_ms",
        "depth_invalid_fraction",
        "front_invalid_fraction",
        "inference_ms",
    ):
        if name in columns:
            report[name] = _scalar_stats(columns[name])
    if "depth_frame_number" in columns:
        numbers = columns["depth_frame_number"]
        finite = np.isfinite(numbers)
        numbers = numbers[finite]
        steps = np.diff(numbers)
        report["depth"] = {
            "unique_frames": int(np.unique(numbers).size),
            "repeated_policy_frame_ratio": float(np.mean(steps == 0)) if steps.size else 0.0,
            "backward_frame_count": int(np.sum(steps < 0)) if steps.size else 0,
            "max_forward_frame_step": int(np.max(steps)) if steps.size else 0,
        }
        if "depth_sensor_timestamp_ms" in columns and finite.size:
            timestamps = columns["depth_sensor_timestamp_ms"][finite]
            changed = steps > 0
            timestamp_steps = np.diff(timestamps)[changed]
            report["depth"]["sensor_timestamp_delta_ms"] = _scalar_stats(timestamp_steps)

    if "t_ms" in columns:
        elapsed = columns["t_ms"]
        elapsed = elapsed[np.isfinite(elapsed)]
        if elapsed.size > 1 and elapsed[-1] > elapsed[0]:
            duration_s = float((elapsed[-1] - elapsed[0]) / 1000.0)
            report["duration_s"] = duration_s
            report["policy_sample_hz"] = float((elapsed.size - 1) / duration_s)
            if "depth" in report:
                report["depth"]["unique_frame_hz"] = float(
                    max(report["depth"]["unique_frames"] - 1, 0) / duration_s
                )

    for name in ("sensor_sequence", "lowstate_tick"):
        if name not in columns:
            continue
        values = columns[name]
        values = values[np.isfinite(values)]
        steps = np.diff(values)
        report[name] = {
            "backward_count": int(np.sum(steps < 0)) if steps.size else 0,
            "repeated_count": int(np.sum(steps == 0)) if steps.size else 0,
            "step": _scalar_stats(steps),
        }
    return report


def compare(current: dict, baseline: dict) -> dict:
    result = {}
    for layer in ("model_raw_action", "applied_target"):
        if layer not in current.get("layers", {}) or layer not in baseline.get("layers", {}):
            continue
        cur = current["layers"][layer]["joints"]
        base = baseline["layers"][layer]["joints"]
        result[layer] = {
            "delta_rms_ratio_per_joint": [
                cur[i]["delta_rms"] / max(base[i]["delta_rms"], 1e-12) for i in range(12)
            ],
            "high_frequency_ratio_per_joint": [
                (cur[i]["10_15_hz"] + cur[i]["15_25_hz"])
                / max(base[i]["10_15_hz"] + base[i]["15_25_hz"], 1e-12)
                for i in range(12)
            ],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--baseline", help="successful sim2real_test_loco CSV with matching columns")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    try:
        report = analyze(args.csv)
        if args.baseline:
            baseline = analyze(args.baseline)
            report["baseline"] = baseline["source"]
            report["comparison"] = compare(report, baseline)
    except (OSError, ValueError, ContractError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 1
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
