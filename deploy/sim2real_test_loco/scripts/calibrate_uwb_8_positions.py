#!/usr/bin/env python3
"""Interactive, read-only calibration for eight UWB beacon positions.

The robot must remain stationary. For each requested direction, the operator
places the beacon at a measured horizontal radius and presses Enter. The tool
then collects a window of valid UWB samples, converts them to planar X/Y, and
writes raw and summary reports. It never publishes robot commands or enables
tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from monitor_uwb import load_sdk


@dataclass(frozen=True)
class Direction:
    name: str
    label: str
    angle_deg: float


DIRECTIONS = (
    Direction("front", "FRONT", 0.0),
    Direction("front_left", "FRONT-LEFT", 45.0),
    Direction("left", "LEFT", 90.0),
    Direction("back_left", "BACK-LEFT", 135.0),
    Direction("back", "BACK", 180.0),
    Direction("back_right", "BACK-RIGHT", -135.0),
    Direction("right", "RIGHT", -90.0),
    Direction("front_right", "FRONT-RIGHT", -45.0),
)


@dataclass(frozen=True)
class Sample:
    received_at: float
    wall_time: str
    beta: float
    pitch: float
    distance_m: float
    yaw: float
    base_yaw: float
    error_state: int
    enabled_from_app: int
    channel: int

    @property
    def valid(self) -> bool:
        values = (self.beta, self.pitch, self.distance_m, self.yaw, self.base_yaw)
        return (
            self.error_state == 0
            and self.enabled_from_app == 1
            and self.distance_m >= 0.0
            and all(math.isfinite(value) for value in values)
        )

    @property
    def planar_distance_m(self) -> float:
        return max(0.0, self.distance_m * math.cos(self.pitch))

    @property
    def local_x_m(self) -> float:
        return self.planar_distance_m * math.cos(self.beta)

    @property
    def local_y_m(self) -> float:
        return self.planar_distance_m * math.sin(self.beta)


class SampleCollector:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.latest: Sample | None = None
        self.capture_active = False
        self.captured: list[Sample] = []

    def callback(self, msg) -> None:
        sample = Sample(
            received_at=time.monotonic(),
            wall_time=datetime.now().isoformat(timespec="milliseconds"),
            beta=float(msg.orientation_est),
            pitch=float(msg.pitch_est),
            distance_m=float(msg.distance_est),
            yaw=float(msg.yaw_est),
            base_yaw=float(msg.base_yaw),
            error_state=int(msg.error_state),
            enabled_from_app=int(msg.enabled_from_app),
            channel=int(msg.channel),
        )
        with self.condition:
            self.latest = sample
            if self.capture_active:
                self.captured.append(sample)
            self.condition.notify_all()

    def wait_for_first_valid(self, timeout_s: float) -> Sample | None:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while True:
                if self.latest is not None and self.latest.valid:
                    return self.latest
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self.condition.wait(timeout=remaining)

    def capture(self, valid_count: int, settle_s: float, timeout_s: float) -> list[Sample]:
        time.sleep(settle_s)
        deadline = time.monotonic() + timeout_s
        with self.condition:
            self.captured = []
            self.capture_active = True
            try:
                while sum(sample.valid for sample in self.captured) < valid_count:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    self.condition.wait(timeout=remaining)
                return list(self.captured)
            finally:
                self.capture_active = False


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def circular_mean(values: list[float]) -> float:
    return math.atan2(
        statistics.fmean(math.sin(value) for value in values),
        statistics.fmean(math.cos(value) for value in values),
    )


def circular_std(values: list[float]) -> float:
    mean_sin = statistics.fmean(math.sin(value) for value in values)
    mean_cos = statistics.fmean(math.cos(value) for value in values)
    resultant = min(1.0, max(1e-12, math.hypot(mean_sin, mean_cos)))
    return math.sqrt(max(0.0, -2.0 * math.log(resultant)))


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize(
    direction: Direction,
    expected_radius_m: float,
    samples: list[Sample],
    stale_threshold_s: float,
) -> dict:
    valid = [sample for sample in samples if sample.valid]
    if not valid:
        raise ValueError("capture contains no valid UWB samples")

    expected_angle = math.radians(direction.angle_deg)
    expected_x = expected_radius_m * math.cos(expected_angle)
    expected_y = expected_radius_m * math.sin(expected_angle)

    beta_mean = circular_mean([sample.beta for sample in valid])
    beta_std = circular_std([sample.beta for sample in valid])
    pitch_mean, pitch_std = mean_std([sample.pitch for sample in valid])
    distance_mean, distance_std = mean_std([sample.distance_m for sample in valid])
    planar_mean, planar_std = mean_std([sample.planar_distance_m for sample in valid])
    x_mean, x_std = mean_std([sample.local_x_m for sample in valid])
    y_mean, y_std = mean_std([sample.local_y_m for sample in valid])

    gaps = [
        current.received_at - previous.received_at
        for previous, current in zip(samples, samples[1:])
    ]
    capture_duration = max(valid[-1].received_at - valid[0].received_at, 1e-9)
    receive_hz = (len(valid) - 1) / capture_duration if len(valid) > 1 else 0.0

    return {
        "direction_index": DIRECTIONS.index(direction) + 1,
        "direction": direction.name,
        "label": direction.label,
        "expected_angle_deg": direction.angle_deg,
        "expected_radius_m": expected_radius_m,
        "expected_x_m": expected_x,
        "expected_y_m": expected_y,
        "samples_total": len(samples),
        "samples_valid": len(valid),
        "samples_invalid": len(samples) - len(valid),
        "receive_hz": receive_hz,
        "gap_p95_s": percentile(gaps, 0.95),
        "gap_max_s": max(gaps, default=0.0),
        "stale_gap_count": sum(gap > stale_threshold_s for gap in gaps),
        "beta_mean_deg": math.degrees(beta_mean),
        "beta_std_deg": math.degrees(beta_std),
        "bearing_error_deg": math.degrees(wrap_angle(beta_mean - expected_angle)),
        "pitch_mean_deg": math.degrees(pitch_mean),
        "pitch_std_deg": math.degrees(pitch_std),
        "distance_3d_mean_m": distance_mean,
        "distance_3d_std_m": distance_std,
        "planar_distance_mean_m": planar_mean,
        "planar_distance_std_m": planar_std,
        "planar_radius_error_m": planar_mean - expected_radius_m,
        "local_x_mean_m": x_mean,
        "local_x_std_m": x_std,
        "local_y_mean_m": y_mean,
        "local_y_std_m": y_std,
        "position_error_m": math.hypot(x_mean - expected_x, y_mean - expected_y),
        "position_error_y_flipped_m": math.hypot(x_mean - expected_x, -y_mean - expected_y),
    }


def raw_rows(direction: Direction, radius_m: float, samples: list[Sample]) -> list[dict]:
    angle = math.radians(direction.angle_deg)
    expected_x = radius_m * math.cos(angle)
    expected_y = radius_m * math.sin(angle)
    rows = []
    for index, sample in enumerate(samples, start=1):
        row = {
            "direction": direction.name,
            "label": direction.label,
            "expected_angle_deg": direction.angle_deg,
            "expected_radius_m": radius_m,
            "expected_x_m": expected_x,
            "expected_y_m": expected_y,
            "sample_index": index,
            **asdict(sample),
            "valid": int(sample.valid),
            "planar_distance_m": sample.planar_distance_m,
            "local_x_m": sample.local_x_m,
            "local_y_m": sample.local_y_m,
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output_dir: Path, summaries: list[dict], raw: list[dict], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "raw_samples.csv", raw)

    report = {"metadata": metadata, "positions": summaries}
    if summaries:
        default_rmse = math.sqrt(
            statistics.fmean(item["position_error_m"] ** 2 for item in summaries)
        )
        flipped_rmse = math.sqrt(
            statistics.fmean(item["position_error_y_flipped_m"] ** 2 for item in summaries)
        )
        report["overall"] = {
            "position_rmse_beta_positive_left_m": default_rmse,
            "position_rmse_beta_positive_right_m": flipped_rmse,
            "suggested_beta_positive_side": "left" if default_rmse <= flipped_rmse else "right",
            "mean_abs_bearing_error_deg": statistics.fmean(
                abs(item["bearing_error_deg"]) for item in summaries
            ),
        }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def controller_processes() -> str:
    result = subprocess.run(
        ["pgrep", "-a", "go2_loco_ctrl"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def print_summary(summary: dict) -> None:
    print(
        "  "
        f"beta={summary['beta_mean_deg']:+.2f} +/- {summary['beta_std_deg']:.2f} deg, "
        f"pitch={summary['pitch_mean_deg']:+.2f} +/- {summary['pitch_std_deg']:.2f} deg"
    )
    print(
        "  "
        f"distance3d={summary['distance_3d_mean_m']:.3f} +/- "
        f"{summary['distance_3d_std_m']:.3f} m, "
        f"planar={summary['planar_distance_mean_m']:.3f} +/- "
        f"{summary['planar_distance_std_m']:.3f} m"
    )
    print(
        "  "
        f"xy=({summary['local_x_mean_m']:+.3f}, {summary['local_y_mean_m']:+.3f}) m, "
        f"bearing_error={summary['bearing_error_deg']:+.2f} deg, "
        f"position_error={summary['position_error_m']:.3f} m"
    )
    print(
        "  "
        f"valid={summary['samples_valid']}/{summary['samples_total']}, "
        f"rate={summary['receive_hz']:.2f} Hz, max_gap={summary['gap_max_s']:.3f} s"
    )


def run_self_test() -> int:
    now = time.monotonic()
    direction = DIRECTIONS[1]
    samples = [
        Sample(
            received_at=now + index * 0.1,
            wall_time="test",
            beta=math.radians(45.0),
            pitch=0.0,
            distance_m=1.0,
            yaw=0.0,
            base_yaw=0.0,
            error_state=0,
            enabled_from_app=1,
            channel=0,
        )
        for index in range(20)
    ]
    result = summarize(direction, 1.0, samples, 0.5)
    assert abs(result["local_x_mean_m"] - math.sqrt(0.5)) < 1e-6
    assert abs(result["local_y_mean_m"] - math.sqrt(0.5)) < 1e-6
    assert abs(result["bearing_error_deg"]) < 1e-6
    assert result["position_error_m"] < 1e-6
    print("Self-test passed.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Go2 UWB at eight planar positions.")
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--topic", default="rt/uwbstate")
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="measured horizontal radius for all positions in meters (default: 1.0)",
    )
    parser.add_argument("--samples", type=int, default=20, help="valid samples per position")
    parser.add_argument("--settle", type=float, default=0.5, help="settling time after Enter")
    parser.add_argument("--capture-timeout", type=float, default=15.0)
    parser.add_argument("--stale-threshold", type=float, default=0.5)
    parser.add_argument("--output-root", type=Path, default=Path("logs/uwb_calibration"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.radius <= 0.0:
        parser.error("--radius must be > 0")
    if args.samples < 2:
        parser.error("--samples must be >= 2")
    if args.settle < 0.0 or args.capture_timeout <= 0.0 or args.stale_threshold <= 0.0:
        parser.error("timing arguments must be positive (settle may be zero)")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    processes = controller_processes()
    if processes:
        print("ERROR: go2_loco_ctrl is running:", file=sys.stderr)
        print(processes, file=sys.stderr)
        print(
            "Put the robot in Passive with LT+B, stop it with Ctrl+C in its original terminal, "
            "then run this calibration again.",
            file=sys.stderr,
        )
        return 3

    session_name = datetime.now().strftime("%Y%m%d_%H%M%S_uwb_8_positions")
    output_dir = args.output_root / session_name
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "network": args.network,
        "topic": args.topic,
        "expected_horizontal_radius_m": args.radius,
        "valid_samples_per_position": args.samples,
        "coordinate_convention": "+X front, +Y left",
        "note": "Read-only DDS subscriber; robot remained stationary.",
    }

    print("Go2 UWB eight-position calibration (read-only)")
    print("Coordinate convention: +X front, +Y left")
    print(f"Measured horizontal radius: {args.radius:.3f} m")
    print(f"Samples per position: {args.samples}")
    print(f"Output directory: {output_dir}")
    print("Keep the robot and its heading completely stationary for all eight measurements.")
    print("Measure radius horizontally from the robot UWB antenna projection, not as a 3D slant.\n")

    ChannelFactoryInitialize, ChannelSubscriber, UwbState_ = load_sdk()
    try:
        ChannelFactoryInitialize(0, args.network)
    except Exception as exc:
        print(f"ERROR: DDS initialization failed: {exc}", file=sys.stderr)
        return 4

    collector = SampleCollector()
    subscriber = ChannelSubscriber(args.topic, UwbState_)
    subscriber.Init(collector.callback, 10)
    summaries: list[dict] = []
    raw: list[dict] = []

    try:
        first = collector.wait_for_first_valid(5.0)
        if first is None:
            print("ERROR: no valid UWB frame within 5 seconds.", file=sys.stderr)
            return 5
        print(
            f"UWB ready: beta={math.degrees(first.beta):+.1f} deg, "
            f"pitch={math.degrees(first.pitch):+.1f} deg, distance={first.distance_m:.3f} m\n"
        )

        for index, direction in enumerate(DIRECTIONS, start=1):
            while True:
                print(
                    f"[{index}/8] Place beacon at {direction.label} "
                    f"({direction.angle_deg:+.0f} deg), horizontal radius {args.radius:.3f} m."
                )
                command = input("Press Enter to capture, 's' to skip, or 'q' to finish: ").strip().lower()
                if command == "q":
                    raise KeyboardInterrupt
                if command == "s":
                    print("Skipped.\n")
                    break
                if command:
                    print("Unknown command.\n")
                    continue

                print(f"Collecting {args.samples} valid samples...")
                samples = collector.capture(args.samples, args.settle, args.capture_timeout)
                valid_count = sum(sample.valid for sample in samples)
                if valid_count < args.samples:
                    print(
                        f"Capture incomplete: {valid_count}/{args.samples} valid samples "
                        f"within {args.capture_timeout:.1f} s."
                    )
                    retry = input("Press Enter to retry, or 's' to skip: ").strip().lower()
                    if retry == "s":
                        print("Skipped.\n")
                        break
                    continue

                summary = summarize(direction, args.radius, samples, args.stale_threshold)
                print_summary(summary)
                decision = input("Press Enter to accept, 'r' to retry, or 'q' to finish: ").strip().lower()
                if decision == "q":
                    raise KeyboardInterrupt
                if decision == "r":
                    print("Retrying this position.\n")
                    continue
                if decision:
                    print("Unknown command; retrying this position.\n")
                    continue

                summaries.append(summary)
                raw.extend(raw_rows(direction, args.radius, samples))
                write_reports(output_dir, summaries, raw, metadata)
                print("Accepted and saved.\n")
                break
    except (KeyboardInterrupt, EOFError):
        print("\nCalibration stopped by operator.")
    finally:
        subscriber.Close()
        write_reports(output_dir, summaries, raw, metadata)

    print(f"Saved {len(summaries)}/8 positions to: {output_dir}")
    if summaries:
        report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
        overall = report.get("overall", {})
        print(
            "RMSE (beta positive=left): "
            f"{overall.get('position_rmse_beta_positive_left_m', math.nan):.3f} m"
        )
        print(
            "RMSE (beta positive=right): "
            f"{overall.get('position_rmse_beta_positive_right_m', math.nan):.3f} m"
        )
        print(f"Suggested beta-positive side: {overall.get('suggested_beta_positive_side', 'n/a')}")
    return 0 if len(summaries) == len(DIRECTIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
