#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pyrealsense2 as rs
import yaml

from realsense_depth_tuner import absolute_frame_age_ms, read_frame_metadata


SENSOR_OPTIONS = {
    "visual_preset": rs.option.visual_preset,
    "emitter": rs.option.emitter_enabled,
    "laser": rs.option.laser_power,
    "frames_queue": rs.option.frames_queue_size,
    "global_time": rs.option.global_time_enabled,
    "error_polling": rs.option.error_polling_enabled,
    "auto_exposure": rs.option.enable_auto_exposure,
    "auto_exposure_limit_toggle": rs.option.auto_exposure_limit_toggle,
    "auto_exposure_limit": rs.option.auto_exposure_limit,
    "auto_gain_limit_toggle": rs.option.auto_gain_limit_toggle,
    "auto_gain_limit": rs.option.auto_gain_limit,
    "exposure": rs.option.exposure,
    "gain": rs.option.gain,
}

FRAME_FIELDS = [
    "wall_time", "elapsed_s", "frame_number", "frame_timestamp_ms",
    "timestamp_domain", "latency_clock_valid", "frame_age_receive_ms",
    "frame_age_processed_ms", "host_processing_ms", "wait_ms", "host_gap_ms",
    "sensor_gap_ms", "frame_delta", "actual_exposure_us", "actual_gain",
    "actual_laser", "actual_emitter_mode", "metadata_fps",
]


DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "serial": "",
        "width": 480,
        "height": 270,
        "fps": 30,
        "timeout_ms": 1000,
    },
    "sensor_options": {
        "global_time": 1,
        "frames_queue": 2,
    },
    "filters": {
        "spatial": {
            "enabled": False,
            "magnitude": 2,
            "alpha": 0.5,
            "delta": 20,
            "holes": 0,
        },
        "temporal": {
            "enabled": True,
            "alpha": 0.5,
            "delta": 20,
            "persistence": 7,
        },
        "hole_fill": {
            "enabled": False,
            "mode": 1,
        },
    },
    "test": {
        "warmup_seconds": 2.0,
        "duration_seconds": 10.0,
        "require_synchronized_timestamps": True,
        "strict_sensor_options": True,
        "output_dir": "logs/depth_latency_test",
    },
}


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in base.items():
        result[key] = merge_dict(value, {}) if isinstance(value, dict) else value
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("latency test YAML root must be a mapping")
    config = merge_dict(DEFAULT_CONFIG, loaded)
    camera = config["camera"]
    test = config["test"]
    if any(int(camera[key]) <= 0 for key in ("width", "height", "fps", "timeout_ms")):
        raise ValueError("camera width, height, fps and timeout_ms must be positive")
    if float(test["warmup_seconds"]) < 0 or float(test["duration_seconds"]) <= 0:
        raise ValueError("warmup_seconds must be non-negative and duration_seconds positive")
    unknown = sorted(set(config["sensor_options"]) - set(SENSOR_OPTIONS))
    if unknown:
        raise ValueError(f"unsupported sensor_options: {', '.join(unknown)}")
    return config


def summarize(values: Iterable[float]) -> Dict[str, Optional[float]]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0, "average_ms": None, "maximum_ms": None,
            "p50_ms": None, "p95_ms": None, "p99_ms": None,
        }
    return {
        "count": int(array.size),
        "average_ms": float(np.mean(array)),
        "maximum_ms": float(np.max(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def option_range_dict(target: Any, option: rs.option) -> Dict[str, float]:
    option_range = target.get_option_range(option)
    return {
        "min": float(option_range.min),
        "max": float(option_range.max),
        "step": float(option_range.step),
        "default": float(option_range.default),
    }


def format_option_range(option_range: Dict[str, float]) -> str:
    return (
        f"[{option_range['min']:g}, {option_range['max']:g}] "
        f"step={option_range['step']:g} default={option_range['default']:g}"
    )


def set_filter_option(
    target: Any, option: rs.option, value: float, name: str
) -> float:
    requested = float(value)
    option_range = option_range_dict(target, option)
    tolerance = max(1e-6, abs(option_range["step"]) * 1e-4)
    if requested < option_range["min"] - tolerance or requested > option_range["max"] + tolerance:
        raise ValueError(
            f"filter option {name}={requested:g} is out of range; "
            f"allowed {format_option_range(option_range)}"
        )
    try:
        target.set_option(option, requested)
        return float(target.get_option(option))
    except RuntimeError as exc:
        raise RuntimeError(
            f"filter option {name}={requested:g} rejected; "
            f"allowed {format_option_range(option_range)}: {exc}"
        ) from exc


def configure_filters(config: Dict[str, Any]) -> Dict[str, Any]:
    filters = config["filters"]
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_fill = rs.hole_filling_filter()

    spatial_cfg = filters["spatial"]
    spatial_actual = {
        "enabled": bool(spatial_cfg["enabled"]),
        "magnitude": set_filter_option(
            spatial, rs.option.filter_magnitude, spatial_cfg["magnitude"],
            "filters.spatial.magnitude",
        ),
        "alpha": set_filter_option(
            spatial, rs.option.filter_smooth_alpha, spatial_cfg["alpha"],
            "filters.spatial.alpha",
        ),
        "delta": set_filter_option(
            spatial, rs.option.filter_smooth_delta, spatial_cfg["delta"],
            "filters.spatial.delta",
        ),
        "holes": set_filter_option(
            spatial, rs.option.holes_fill, spatial_cfg["holes"],
            "filters.spatial.holes",
        ),
    }

    temporal_cfg = filters["temporal"]
    temporal_actual = {
        "enabled": bool(temporal_cfg["enabled"]),
        "alpha": set_filter_option(
            temporal, rs.option.filter_smooth_alpha, temporal_cfg["alpha"],
            "filters.temporal.alpha",
        ),
        "delta": set_filter_option(
            temporal, rs.option.filter_smooth_delta, temporal_cfg["delta"],
            "filters.temporal.delta",
        ),
        "persistence": set_filter_option(
            temporal, rs.option.holes_fill, temporal_cfg["persistence"],
            "filters.temporal.persistence",
        ),
    }

    hole_cfg = filters["hole_fill"]
    hole_actual = {
        "enabled": bool(hole_cfg["enabled"]),
        "mode": set_filter_option(
            hole_fill, rs.option.holes_fill, hole_cfg["mode"],
            "filters.hole_fill.mode",
        ),
    }
    return {
        "instances": (spatial, temporal, hole_fill),
        "actual": {
            "spatial": spatial_actual,
            "temporal": temporal_actual,
            "hole_fill": hole_actual,
        },
    }


def apply_sensor_options(
    sensor: Any, requested: Dict[str, Any], strict: bool
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for key, requested_value in requested.items():
        option = SENSOR_OPTIONS[key]
        result: Dict[str, Any] = {"requested": float(requested_value)}
        try:
            if not sensor.supports(option):
                raise RuntimeError("option not supported by selected depth sensor")
            option_range = option_range_dict(sensor, option)
            result["range"] = option_range
            requested_float = float(requested_value)
            if not option_range["min"] <= requested_float <= option_range["max"]:
                raise RuntimeError(
                    f"requested {requested_float:g}, allowed "
                    f"{format_option_range(option_range)}"
                )
            sensor.set_option(option, requested_float)
            result["applied"] = float(sensor.get_option(option))
            result["status"] = "applied"
        except RuntimeError as exc:
            result["status"] = "rejected"
            result["error"] = str(exc)
            if strict:
                raise RuntimeError(
                    f"sensor option sensor_options.{key}={float(requested_value):g} "
                    f"rejected: {exc}"
                ) from exc
        results[key] = result
    return results


def create_session_dir(output_dir: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    session = output_dir.expanduser().resolve() / f"depth_latency_{stamp}"
    session.mkdir(parents=True, exist_ok=False)
    return session


def run_test(config: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    camera = config["camera"]
    test_cfg = config["test"]
    session_dir = create_session_dir(Path(test_cfg["output_dir"]))
    (session_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    context = rs.context()
    pipeline = rs.pipeline(context)
    rs_config = rs.config()
    if camera["serial"]:
        rs_config.enable_device(str(camera["serial"]))
    rs_config.enable_stream(
        rs.stream.depth,
        int(camera["width"]),
        int(camera["height"]),
        rs.format.z16,
        int(camera["fps"]),
    )

    profile: Optional[rs.pipeline_profile] = None
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "status": "failed",
        "source_config": str(source_path.resolve()),
        "session_dir": str(session_dir),
    }
    try:
        profile = pipeline.start(rs_config)
        device = profile.get_device()
        sensor = device.first_depth_sensor()
        identity = {
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "usb": device.get_info(rs.camera_info.usb_type_descriptor),
            "profile": [int(camera["width"]), int(camera["height"]), int(camera["fps"])],
        }
        print("DEVICE " + json.dumps(identity, ensure_ascii=False))
        if identity["usb"].startswith("2"):
            print("WARNING USB2_LINK policy=warning_only")

        sensor_results = apply_sensor_options(
            sensor,
            config["sensor_options"],
            bool(test_cfg["strict_sensor_options"]),
        )
        filter_bundle = configure_filters(config)
        spatial, temporal, hole_fill = filter_bundle["instances"]
        filter_actual = filter_bundle["actual"]

        warmup_seconds = float(test_cfg["warmup_seconds"])
        duration_seconds = float(test_cfg["duration_seconds"])
        warmup_end = time.monotonic() + warmup_seconds
        while time.monotonic() < warmup_end:
            pipeline.wait_for_frames(int(camera["timeout_ms"]))

        start = time.monotonic()
        end = start + duration_seconds
        previous_arrival_ns: Optional[int] = None
        previous_sensor_ms: Optional[float] = None
        previous_frame: Optional[int] = None
        while time.monotonic() < end:
            wait_start_ns = time.monotonic_ns()
            frames = pipeline.wait_for_frames(int(camera["timeout_ms"]))
            arrival_ns = time.monotonic_ns()
            arrival_wall_ms = time.time_ns() / 1e6
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            frame_number = int(depth_frame.get_frame_number())
            frame_timestamp_ms = float(depth_frame.get_timestamp())
            timestamp_domain = str(depth_frame.get_frame_timestamp_domain())
            receive_age_ms = absolute_frame_age_ms(
                frame_timestamp_ms, timestamp_domain, arrival_wall_ms
            )

            processed: Any = depth_frame
            if filter_actual["spatial"]["enabled"]:
                processed = spatial.process(processed)
            if filter_actual["temporal"]["enabled"]:
                processed = temporal.process(processed)
            if filter_actual["hole_fill"]["enabled"]:
                processed = hole_fill.process(processed)
            np.asanyarray(processed.get_data())
            processed_ns = time.monotonic_ns()
            processed_wall_ms = time.time_ns() / 1e6
            processed_age_ms = absolute_frame_age_ms(
                frame_timestamp_ms, timestamp_domain, processed_wall_ms
            )
            metadata = read_frame_metadata(depth_frame)

            row: Dict[str, Any] = {
                "wall_time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "elapsed_s": time.monotonic() - start,
                "frame_number": frame_number,
                "frame_timestamp_ms": frame_timestamp_ms,
                "timestamp_domain": timestamp_domain,
                "latency_clock_valid": int(math.isfinite(receive_age_ms)),
                "frame_age_receive_ms": receive_age_ms,
                "frame_age_processed_ms": processed_age_ms,
                "host_processing_ms": (processed_ns - arrival_ns) / 1e6,
                "wait_ms": (arrival_ns - wait_start_ns) / 1e6,
                "host_gap_ms": (
                    (arrival_ns - previous_arrival_ns) / 1e6
                    if previous_arrival_ns is not None else 0.0
                ),
                "sensor_gap_ms": (
                    frame_timestamp_ms - previous_sensor_ms
                    if previous_sensor_ms is not None else 0.0
                ),
                "frame_delta": (
                    frame_number - previous_frame if previous_frame is not None else 0
                ),
                **metadata,
            }
            rows.append(row)
            previous_arrival_ns = arrival_ns
            previous_sensor_ms = frame_timestamp_ms
            previous_frame = frame_number

        with (session_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in FRAME_FIELDS} for row in rows)

        receive_values = [float(row["frame_age_receive_ms"]) for row in rows]
        processed_values = [float(row["frame_age_processed_ms"]) for row in rows]
        valid_count = sum(math.isfinite(value) for value in receive_values)
        valid_fraction = valid_count / len(rows) if rows else 0.0
        require_sync = bool(test_cfg["require_synchronized_timestamps"])
        if require_sync and (not rows or valid_fraction < 0.95):
            raise RuntimeError(
                f"synchronized timestamp coverage {valid_fraction:.1%}, expected at least 95%"
            )

        elapsed = time.monotonic() - start
        dropped_frames = sum(max(0, int(row["frame_delta"]) - 1) for row in rows)
        summary.update({
            "status": "passed",
            "device": identity,
            "requested_config": config,
            "sensor_options": sensor_results,
            "filters_actual": filter_actual,
            "measurement": {
                "requested_duration_s": duration_seconds,
                "actual_duration_s": elapsed,
                "frames": len(rows),
                "effective_fps": len(rows) / elapsed if elapsed > 0 else 0.0,
                "dropped_frames": dropped_frames,
                "timestamp_valid_fraction": valid_fraction,
            },
            "frame_age_receive": summarize(receive_values),
            "frame_age_processed": summarize(processed_values),
            "host_processing": summarize(
                float(row["host_processing_ms"]) for row in rows
            ),
            "outputs": {
                "frames_csv": str(session_dir / "frames.csv"),
                "summary_json": str(session_dir / "summary.json"),
            },
        })
    except Exception as exc:
        summary["error"] = str(exc)
        raise
    finally:
        if profile is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        (session_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless RealSense latency measurement using a YAML configuration"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        if args.duration is not None:
            if args.duration <= 0:
                raise ValueError("duration override must be positive")
            config["test"]["duration_seconds"] = args.duration
        if args.output_dir is not None:
            config["test"]["output_dir"] = str(args.output_dir)
        summary = run_test(config, args.config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    result = {
        "status": summary["status"],
        "duration_s": summary["measurement"]["actual_duration_s"],
        "frames": summary["measurement"]["frames"],
        "receive_average_ms": summary["frame_age_receive"]["average_ms"],
        "receive_maximum_ms": summary["frame_age_receive"]["maximum_ms"],
        "processed_average_ms": summary["frame_age_processed"]["average_ms"],
        "processed_maximum_ms": summary["frame_age_processed"]["maximum_ms"],
        "summary_json": summary["outputs"]["summary_json"],
    }
    print(
        "LATENCY "
        f"receive_avg={result['receive_average_ms']:.3f}ms "
        f"receive_max={result['receive_maximum_ms']:.3f}ms "
        f"processed_avg={result['processed_average_ms']:.3f}ms "
        f"processed_max={result['processed_maximum_ms']:.3f}ms"
    )
    print("RESULT " + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
