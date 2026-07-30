#!/usr/bin/env python3
"""Replay recorded VisionLoco CSVs against an offline guard candidate."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def fail(message: str) -> None:
    raise SystemExit(f"transparent guard replay failed: {message}")


def load_profile(path: Path) -> Dict:
    with path.open(encoding="utf-8") as handle:
        profile = json.load(handle)
    if profile.get("schema") != 1 or profile.get("status") != "offline_only":
        fail("profile must be schema 1 and offline_only")
    for key in (
        "joint_order", "raw_action_abs_max", "raw_action_step_max",
        "target_min_rad", "target_max_rad",
    ):
        values = profile.get(key)
        if not isinstance(values, list) or len(values) != 12:
            fail(f"{key} must contain 12 values")
    numeric_keys = (
        "raw_action_abs_max", "raw_action_step_max",
        "target_min_rad", "target_max_rad",
    )
    for key in numeric_keys:
        profile[key] = [float(value) for value in profile[key]]
        if not all(math.isfinite(value) for value in profile[key]):
            fail(f"{key} contains a non-finite value")
    for low, high in zip(profile["target_min_rad"], profile["target_max_rad"]):
        if low >= high:
            fail("each target minimum must be below its maximum")
    return profile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_motion(row: Dict[str, str], threshold: float) -> bool:
    return any(abs(float(row[name])) > threshold for name in ("vx", "vy", "wz"))


def load_motion_rows(path: Path, threshold: float) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"t_ms", "vx", "vy", "wz"}
    required.update(f"action{index}" for index in range(12))
    required.update(f"target{index}" for index in range(12))
    if not rows or not required.issubset(rows[0]):
        fail(f"{path} is missing required VisionLoco columns")
    motion = [row for row in rows if is_motion(row, threshold)]
    if not motion:
        fail(f"{path} contains no non-zero motion rows")
    return motion


def first_trip(
    values: Sequence[Sequence[float]], limit: float, trip_frames: int
) -> Optional[Tuple[int, int, float]]:
    counts = [0] * 12
    for frame_index, frame in enumerate(values):
        for joint, value in enumerate(frame):
            counts[joint] = counts[joint] + 1 if abs(value) > limit else 0
            if counts[joint] >= trip_frames:
                return frame_index, joint, value
    return None


def replay(path: Path, profile: Dict) -> Dict:
    threshold = float(profile["derivation"]["motion_threshold"])
    rows = load_motion_rows(path, threshold)
    previous_action = [0.0] * 12
    violations = {"raw_abs": 0, "raw_step": 0, "target_range": 0}
    maxima = {"raw_abs": 0.0, "raw_step": 0.0}
    effort_frames: List[List[float]] = []
    have_effort = all(f"tau{index}" in rows[0] for index in range(12))

    for row in rows:
        action = [float(row[f"action{index}"]) for index in range(12)]
        target = [float(row[f"target{index}"]) for index in range(12)]
        if not all(math.isfinite(value) for value in action + target):
            fail(f"{path} contains a non-finite action or target")
        frame_raw_abs = max(abs(value) for value in action)
        frame_raw_step = max(
            abs(action[index] - previous_action[index]) for index in range(12)
        )
        maxima["raw_abs"] = max(maxima["raw_abs"], frame_raw_abs)
        maxima["raw_step"] = max(maxima["raw_step"], frame_raw_step)
        if any(
            abs(action[index]) > profile["raw_action_abs_max"][index] + 1.0e-9
            for index in range(12)
        ):
            violations["raw_abs"] += 1
        if any(
            abs(action[index] - previous_action[index])
            > profile["raw_action_step_max"][index] + 1.0e-9
            for index in range(12)
        ):
            violations["raw_step"] += 1
        if any(
            target[index] < profile["target_min_rad"][index] - 1.0e-9
            or target[index] > profile["target_max_rad"][index] + 1.0e-9
            for index in range(12)
        ):
            violations["target_range"] += 1
        previous_action = action
        if have_effort:
            effort_frames.append(
                [float(row[f"tau{index}"]) for index in range(12)]
            )

    load = profile["provisional_load_guard"]
    effort_trip = (
        first_trip(
            effort_frames,
            float(load["joint_effort_abs_nm"]),
            int(load["trip_frames"]),
        )
        if have_effort else None
    )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "motion_frames": len(rows),
        "output_guard_pass": not any(violations.values()),
        "violations": violations,
        "maxima": maxima,
        "effort_available": have_effort,
        "provisional_effort_trip": effort_trip,
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_profile = (
        script_dir.parent / "runtime" / "unitree_rl_lab_test" / "deploy" /
        "robots" / "go2_loco" / "config" / "transparent_guard_candidate.json"
    )
    parser = argparse.ArgumentParser(
        description="Replay recorded logs against the offline 378413 guard candidate."
    )
    parser.add_argument("--profile", type=Path, default=default_profile)
    parser.add_argument("--historical", type=Path, action="append", default=[])
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--negative", type=Path, action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.historical:
        fail("at least one --historical log is required")
    profile = load_profile(args.profile)
    source_hashes = {
        item["name"]: item["sha256"] for item in profile.get("source_logs", [])
    }
    results = []
    for role, paths in (
        ("historical", args.historical),
        ("candidate", args.candidate),
        ("negative", args.negative),
    ):
        for path in paths:
            result = replay(path, profile)
            result["role"] = role
            if role == "historical":
                expected = source_hashes.get(path.name)
                if expected is None or result["sha256"] != expected:
                    fail(f"historical source identity mismatch for {path.name}")
                if not result["output_guard_pass"]:
                    fail(f"historical source does not pass its derived envelope: {path}")
            results.append(result)

    names = profile["joint_order"]
    for result in results:
        trip = result["provisional_effort_trip"]
        if trip is None:
            effort = "n/a" if not result["effort_available"] else "no"
        else:
            effort = f"yes(frame={trip[0]},joint={names[trip[1]]},value={trip[2]:.2f})"
        print(
            f"{result['role']:10} {Path(result['path']).name:32} "
            f"frames={result['motion_frames']:4d} "
            f"output_pass={str(result['output_guard_pass']).lower():5} "
            f"violations={result['violations']} "
            f"raw={result['maxima']['raw_abs']:.3f} "
            f"d_raw={result['maxima']['raw_step']:.3f} "
            f"effort_trip={effort}"
        )

    negative_results = [item for item in results if item["role"] == "negative"]
    if negative_results and all(item["output_guard_pass"] for item in negative_results):
        print(
            "LIMITATION: known negative logs pass the policy-output envelope; "
            "output bounds alone cannot qualify ground safety."
        )
    if any(
        item["role"] == "historical" and not item["effort_available"]
        for item in results
    ):
        print(
            "LIMITATION: historical successful logs have no effort columns; "
            "the provisional effort guard requires harness validation."
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(
                {"profile": str(args.profile), "results": results},
                handle, indent=2, sort_keys=True,
            )
            handle.write("\n")


if __name__ == "__main__":
    main()
