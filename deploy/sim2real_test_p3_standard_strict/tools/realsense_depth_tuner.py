#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import pyrealsense2 as rs


WINDOW_DEPTH = "D435i Depth Diagnostic and Tuner"
POLICY_MAX_DEPTH_M = 5.0
INVALID_BGR = (255, 0, 255)
ABSOLUTE_TIMESTAMP_DOMAINS = {
    str(rs.timestamp_domain.global_time),
    str(rs.timestamp_domain.system_time),
}

COMPETITION_SENSOR_OPTION_SPECS = (
    ("visual_preset", "Visual preset", rs.option.visual_preset, False),
    ("emitter", "Emitter mode", rs.option.emitter_enabled, False),
    ("laser", "Laser power", rs.option.laser_power, False),
    ("frames_queue", "Frames queue size", rs.option.frames_queue_size, False),
    ("global_time", "Global time", rs.option.global_time_enabled, False),
    ("error_polling", "Internal error polling", rs.option.error_polling_enabled, False),
    ("auto_exposure", "Auto exposure", rs.option.enable_auto_exposure, False),
    ("auto_exposure_limit_toggle", "Exposure limit enabled",
     rs.option.auto_exposure_limit_toggle, True),
    ("auto_exposure_limit", "Exposure limit (us)", rs.option.auto_exposure_limit, True),
    ("auto_gain_limit_toggle", "Gain limit enabled",
     rs.option.auto_gain_limit_toggle, True),
    ("auto_gain_limit", "Gain limit", rs.option.auto_gain_limit, True),
    ("exposure", "Manual exposure (us)", rs.option.exposure, False),
    ("gain", "Manual gain", rs.option.gain, False),
)

FRAME_METADATA_SPECS = (
    ("actual_exposure_us", rs.frame_metadata_value.actual_exposure),
    ("actual_gain", rs.frame_metadata_value.gain_level),
    ("actual_laser", rs.frame_metadata_value.frame_laser_power),
    ("actual_emitter_mode", rs.frame_metadata_value.frame_emitter_mode),
    ("metadata_fps", rs.frame_metadata_value.actual_fps),
)

SAFE_REAPPLY_AFTER_REENUMERATION = {"frames_queue", "global_time", "error_polling"}


def read_frame_metadata(frame: Any) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, metadata in FRAME_METADATA_SPECS:
        try:
            if frame.supports_frame_metadata(metadata):
                values[key] = float(frame.get_frame_metadata(metadata))
        except RuntimeError:
            continue
    return values


def absolute_frame_age_ms(
    frame_timestamp_ms: float,
    timestamp_domain: str,
    host_wall_ms: float,
    maximum_age_ms: float = 60_000.0,
) -> float:
    if timestamp_domain not in ABSOLUTE_TIMESTAMP_DOMAINS:
        return float("nan")
    age_ms = float(host_wall_ms) - float(frame_timestamp_ms)
    if not np.isfinite(age_ms) or age_ms < -5.0 or age_ms > maximum_age_ms:
        return float("nan")
    return max(0.0, age_ms)


def compute_invalid_mask(
    depth_units: np.ndarray,
    depth_scale: float,
    max_depth_m: float = POLICY_MAX_DEPTH_M,
) -> np.ndarray:
    meters = depth_units.astype(np.float32) * float(depth_scale)
    return (~np.isfinite(meters)) | (meters <= 0.0) | (meters >= max_depth_m)


def compute_invalid_stats(invalid: np.ndarray) -> Dict[str, float]:
    if invalid.ndim != 2 or invalid.size == 0:
        raise ValueError("invalid mask must be a non-empty HxW array")
    width = invalid.shape[1]
    x0, x1 = width // 3, 2 * width // 3
    return {
        "whole": float(np.mean(invalid)),
        "central_third": float(np.mean(invalid[:, x0:x1])),
    }


def render_depth(
    depth_units: np.ndarray,
    depth_scale: float,
    min_depth_m: float,
    max_depth_m: float,
    policy_max_depth_m: float = POLICY_MAX_DEPTH_M,
) -> Tuple[np.ndarray, np.ndarray]:
    meters = depth_units.astype(np.float32) * float(depth_scale)
    invalid = compute_invalid_mask(depth_units, depth_scale, policy_max_depth_m)
    span = max(max_depth_m - min_depth_m, 1e-3)
    normalized = np.clip((meters - min_depth_m) / span, 0.0, 1.0)
    gray = np.asarray((1.0 - normalized) * 255.0, dtype=np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[invalid] = INVALID_BGR
    return color, invalid


def render_invalid_changes(raw_invalid: np.ndarray, filtered_invalid: np.ndarray) -> np.ndarray:
    image = np.zeros((*raw_invalid.shape, 3), dtype=np.uint8)
    unchanged_invalid = raw_invalid & filtered_invalid
    recovered = raw_invalid & ~filtered_invalid
    newly_invalid = ~raw_invalid & filtered_invalid
    image[unchanged_invalid] = (255, 255, 255)
    image[recovered] = (0, 220, 0)
    image[newly_invalid] = (0, 0, 255)
    return image


def _put_label(image: np.ndarray, text: str, row: int, color: Tuple[int, int, int]) -> None:
    y = 25 + row * 24
    cv2.putText(
        image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        color, 1, cv2.LINE_AA,
    )


def compose_dashboard(
    raw_depth: np.ndarray,
    filtered_depth: np.ndarray,
    depth_scale: float,
    display_min_m: float,
    display_max_m: float,
    fps: float,
    frame_number: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    raw_color, raw_invalid = render_depth(
        raw_depth, depth_scale, display_min_m, display_max_m
    )
    filtered_color, filtered_invalid = render_depth(
        filtered_depth, depth_scale, display_min_m, display_max_m
    )
    raw_stats = compute_invalid_stats(raw_invalid)
    filtered_stats = compute_invalid_stats(filtered_invalid)
    mask_view = np.zeros((*raw_invalid.shape, 3), dtype=np.uint8)
    mask_view[raw_invalid] = (255, 255, 255)
    changes = render_invalid_changes(raw_invalid, filtered_invalid)

    for panel in (raw_color, filtered_color, mask_view, changes):
        x0, x1 = panel.shape[1] // 3, 2 * panel.shape[1] // 3
        cv2.line(panel, (x0, 0), (x0, panel.shape[0] - 1), (0, 255, 255), 1)
        cv2.line(panel, (x1, 0), (x1, panel.shape[0] - 1), (0, 255, 255), 1)

    _put_label(raw_color, "RAW DEPTH", 0, (255, 255, 255))
    _put_label(
        raw_color,
        f"invalid {raw_stats['whole']:.1%}  center {raw_stats['central_third']:.1%}",
        1,
        (255, 255, 255),
    )
    _put_label(filtered_color, "FILTERED DEPTH", 0, (255, 255, 255))
    _put_label(
        filtered_color,
        f"invalid {filtered_stats['whole']:.1%}  center {filtered_stats['central_third']:.1%}",
        1,
        (255, 255, 255),
    )
    _put_label(mask_view, "RAW INVALID MASK", 0, (0, 255, 255))
    _put_label(mask_view, "white=invalid", 1, (0, 255, 255))
    _put_label(changes, "FILTER EFFECT", 0, (0, 255, 255))
    _put_label(changes, "green=filled red=lost white=still invalid", 1, (0, 255, 255))

    dashboard = np.vstack((np.hstack((raw_color, filtered_color)),
                           np.hstack((mask_view, changes))))
    stats = {
        "raw_invalid_fraction": raw_stats["whole"],
        "raw_central_invalid_fraction": raw_stats["central_third"],
        "filtered_invalid_fraction": filtered_stats["whole"],
        "filtered_central_invalid_fraction": filtered_stats["central_third"],
        "recovered_fraction": float(np.mean(raw_invalid & ~filtered_invalid)),
        "newly_invalid_fraction": float(np.mean(~raw_invalid & filtered_invalid)),
    }
    return dashboard, stats


def classify_stream_discontinuity(
    previous_frame: int,
    frame_number: int,
    host_gap_ms: float,
    sensor_gap_ms: float,
    threshold_ms: float,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    frame_delta = frame_number - previous_frame
    if frame_delta <= 0:
        events.append({"kind": "FRAME_NON_MONOTONIC", "frame_delta": frame_delta})
    elif frame_delta > 1:
        events.append({
            "kind": "FRAME_JUMP",
            "frame_delta": frame_delta,
            "dropped_frames": frame_delta - 1,
        })
    if host_gap_ms >= threshold_ms:
        if sensor_gap_ms >= threshold_ms:
            cause = "camera_or_usb_stall"
        else:
            cause = "host_processing_or_delivery_stall"
        events.append({
            "kind": "STREAM_GAP",
            "suspected_cause": cause,
            "host_gap_ms": host_gap_ms,
            "sensor_gap_ms": sensor_gap_ms,
        })
    return events


class DiagnosticLogger:
    METRIC_FIELDS = [
        "wall_time", "monotonic_s", "frame_number", "fps", "wait_ms",
        "host_gap_ms", "sensor_gap_ms", "relative_queue_ms", "frame_timestamp_ms",
        "timestamp_domain", "latency_clock_valid", "frame_age_receive_ms",
        "frame_age_ready_ms", "filter_ms", "render_ms", "app_latency_ms",
        "gui_latency_ms", "gui_convert_ms", "frame_age_gui_submit_ms",
        "raw_invalid_fraction", "raw_central_invalid_fraction",
        "filtered_invalid_fraction", "filtered_central_invalid_fraction",
        "recovered_fraction", "newly_invalid_fraction", "process_cpu_percent",
        "system_cpu_percent", "process_rss_mb", "system_ram_percent",
        "gpu_percent", "gpu_temp_c", "timeouts", "frame_jumps", "stream_gaps",
        "actual_exposure_us", "actual_gain", "actual_laser", "actual_emitter_mode",
        "metadata_fps", "frames_queue_size", "usb_type",
        "laser_requested", "laser_applied", "control_state", "device_generation",
    ]
    DISPLAY_METRIC_FIELDS = [
        "wall_time", "monotonic_s", "frame_number", "frame_timestamp_ms",
        "timestamp_domain", "latency_clock_valid", "frame_age_receive_ms",
        "frame_age_ready_ms", "frame_age_gui_submit_ms", "app_latency_ms",
        "gui_schedule_ms", "gui_convert_ms", "actual_exposure_us",
        "frames_queue_size", "temporal_enabled", "temporal_alpha",
        "temporal_delta", "temporal_persistence",
    ]

    def __init__(self, session_dir: Path) -> None:
        self.lock = threading.Lock()
        self.recent: deque = deque(maxlen=80)
        self.events_path = session_dir / "events.jsonl"
        self.metrics_path = session_dir / "metrics.csv"
        self.display_metrics_path = session_dir / "display_latency.csv"
        self.events_file = self.events_path.open("a", encoding="utf-8", buffering=1)
        self.metrics_file = self.metrics_path.open("w", newline="", encoding="utf-8")
        self.display_metrics_file = self.display_metrics_path.open(
            "w", newline="", encoding="utf-8"
        )
        self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=self.METRIC_FIELDS)
        self.display_metrics_writer = csv.DictWriter(
            self.display_metrics_file, fieldnames=self.DISPLAY_METRIC_FIELDS
        )
        self.last_display_flush_at = 0.0
        self.metrics_writer.writeheader()
        self.display_metrics_writer.writeheader()
        self.metrics_file.flush()
        self.display_metrics_file.flush()

    def event(self, kind: str, severity: str = "INFO", **fields: Any) -> None:
        record = {
            "wall_time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "monotonic_s": time.monotonic(),
            "severity": severity,
            "kind": kind,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.lock:
            self.events_file.write(line + "\n")
            self.events_file.flush()
            self.recent.append(record)
        if severity != "DEBUG":
            print(f"[{severity}] {kind} " + json.dumps(fields, ensure_ascii=False, default=str))

    def metric(self, values: Dict[str, Any]) -> None:
        row = {field: values.get(field, "") for field in self.METRIC_FIELDS}
        row["wall_time"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        row["monotonic_s"] = time.monotonic()
        with self.lock:
            self.metrics_writer.writerow(row)
            self.metrics_file.flush()

    def display_metric(self, values: Dict[str, Any]) -> None:
        row = {field: values.get(field, "") for field in self.DISPLAY_METRIC_FIELDS}
        row["wall_time"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        row["monotonic_s"] = time.monotonic()
        with self.lock:
            self.display_metrics_writer.writerow(row)
            now = time.monotonic()
            if now - self.last_display_flush_at >= 1.0:
                self.display_metrics_file.flush()
                self.last_display_flush_at = now

    def recent_records(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.recent)

    def close(self) -> None:
        with self.lock:
            self.events_file.close()
            self.metrics_file.close()
            self.display_metrics_file.close()


class TegrastatsSampler:
    def __init__(self, logger: DiagnosticLogger) -> None:
        self.logger = logger
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.running = threading.Event()
        self.lock = threading.Lock()
        self.values: Dict[str, float] = {"gpu_percent": float("nan"), "gpu_temp_c": float("nan")}

    def start(self) -> None:
        executable = shutil.which("tegrastats")
        if not executable:
            self.logger.event("TEGRastats_UNAVAILABLE", severity="WARNING")
            return
        try:
            self.process = subprocess.Popen(
                [executable, "--interval", "1000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.running.set()
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
        except OSError as exc:
            self.logger.event("TEGRastats_START_FAILED", severity="WARNING", error=str(exc))

    def _read_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            if not self.running.is_set():
                break
            gpu = re.search(r"GR3D_FREQ\s+(\d+)%", line)
            temperatures = [float(value) for value in re.findall(r"@([\d.]+)C", line)]
            with self.lock:
                if gpu:
                    self.values["gpu_percent"] = float(gpu.group(1))
                if temperatures:
                    self.values["gpu_temp_c"] = max(temperatures)

    def snapshot(self) -> Dict[str, float]:
        with self.lock:
            return dict(self.values)

    def stop(self) -> None:
        self.running.clear()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread is not None:
            self.thread.join(timeout=2)


@dataclass
class NumericParameter:
    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    initial: float
    default_value: Optional[float] = None
    target: Optional[Any] = None
    option: Optional[rs.option] = None
    restart_required: bool = False

    def __post_init__(self) -> None:
        self.lock = threading.Lock()
        if self.default_value is None:
            self.default_value = float(self.initial)
        self.device_maximum = float(self.maximum)
        self.desired = float(self.initial)
        self.applied = float(self.initial)
        self.desired_changed_at = 0.0
        self.last_apply_attempt_at = 0.0
        self.last_error = ""

    @classmethod
    def from_target(
        cls, key: str, label: str, target: Any, option: rs.option,
        restart_required: bool = False,
    ) -> Optional["NumericParameter"]:
        try:
            value_range = target.get_option_range(option)
            initial = float(target.get_option(option))
        except (RuntimeError, TypeError):
            return None
        step = float(value_range.step) if value_range.step > 0 else 1.0
        return cls(
            key=key,
            label=label,
            minimum=float(value_range.min),
            maximum=float(value_range.max),
            step=step,
            initial=initial,
            default_value=float(value_range.default),
            target=target,
            option=option,
            restart_required=restart_required,
        )

    def set_desired(self, value: float) -> float:
        clamped = float(np.clip(value, self.minimum, self.maximum))
        quantized = self.minimum + round((clamped - self.minimum) / self.step) * self.step
        quantized = float(np.clip(quantized, self.minimum, self.maximum))
        with self.lock:
            if abs(quantized - self.desired) >= self.step * 0.5:
                self.desired = quantized
                self.desired_changed_at = time.monotonic()
        return quantized

    def desired_value(self) -> float:
        with self.lock:
            return self.desired

    def reset(self) -> float:
        assert self.default_value is not None
        return self.set_desired(self.default_value)

    def cap_maximum(self, maximum: float, clamp_desired: bool = True) -> None:
        with self.lock:
            self.maximum = max(self.minimum, min(float(maximum), self.device_maximum))
        if clamp_desired:
            self.set_desired(self.desired_value())

    def rebind(self, target: Any, preserve_desired: bool) -> bool:
        if self.option is None:
            return False
        try:
            value_range = target.get_option_range(self.option)
            current = float(target.get_option(self.option))
        except (RuntimeError, TypeError):
            with self.lock:
                self.target = None
                self.applied = float("nan")
                self.last_error = "option unavailable after device rebind"
            return False
        step = float(value_range.step) if value_range.step > 0 else 1.0
        with self.lock:
            previous_desired = self.desired
            self.target = target
            self.minimum = float(value_range.min)
            self.maximum = float(value_range.max)
            self.device_maximum = float(value_range.max)
            self.step = step
            self.default_value = float(value_range.default)
            self.initial = current
            if preserve_desired:
                clamped = float(np.clip(previous_desired, self.minimum, self.maximum))
                self.desired = self.minimum + round(
                    (clamped - self.minimum) / self.step
                ) * self.step
            else:
                self.desired = current
            self.applied = current
            self.desired_changed_at = 0.0
            self.last_apply_attempt_at = 0.0
            self.last_error = ""
        return True

    def invalidate_applied(self) -> None:
        with self.lock:
            self.applied = float("nan")

    def applied_value(self) -> float:
        with self.lock:
            return float(self.applied)

    def has_pending_change(self) -> bool:
        with self.lock:
            if np.isfinite(self.applied):
                return abs(self.desired - self.applied) >= self.step * 0.5
            return self.desired_changed_at > self.last_apply_attempt_at

    def confirm_from_frame_metadata(self, actual: float) -> bool:
        with self.lock:
            if not np.isfinite(actual) or abs(actual - self.desired) >= self.step * 0.5:
                return False
            changed = not np.isfinite(self.applied) or abs(actual - self.applied) >= self.step * 0.5
            self.applied = float(actual)
            self.desired = float(actual)
            self.last_error = ""
            return changed

    def apply(
        self,
        logger: DiagnosticLogger,
        enabled: bool = True,
        settle_s: float = 0.0,
        retry_s: float = 2.0,
        now_s: Optional[float] = None,
    ) -> bool:
        if not enabled or self.target is None or self.option is None:
            return False
        with self.lock:
            desired = self.desired
            applied = self.applied
            desired_changed_at = self.desired_changed_at
            last_apply_attempt_at = self.last_apply_attempt_at
        if np.isfinite(applied) and abs(desired - applied) < self.step * 0.5:
            return False
        now = time.monotonic() if now_s is None else now_s
        if settle_s > 0.0 and now - desired_changed_at < settle_s:
            return False
        if not np.isfinite(applied) and now - last_apply_attempt_at < retry_s:
            return False
        with self.lock:
            self.last_apply_attempt_at = now
        try:
            self.target.set_option(self.option, desired)
            actual = float(self.target.get_option(self.option))
            with self.lock:
                self.applied = actual
                self.desired = actual
                self.last_error = ""
            logger.event("OPTION_APPLIED", key=self.key, requested=desired, actual=actual)
            return True
        except RuntimeError as exc:
            with self.lock:
                self.applied = float("nan")
                self.last_error = str(exc)
            logger.event(
                "OPTION_REJECTED", severity="WARNING", key=self.key,
                requested=desired, error=str(exc),
            )
            return False

    def describe(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "desired": float(self.desired),
                "applied": float(self.applied),
                "initial": self.initial,
                "default": self.default_value,
                "min": self.minimum,
                "max": self.maximum,
                "device_max": self.device_maximum,
                "step": self.step,
                "restart_required": self.restart_required,
                "last_error": self.last_error,
            }


class DepthTuner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        timestamp_ms = int(time.time() * 1000)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{timestamp_ms % 1000:03d}"
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.output_dir / f"depth_tuner_{stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.logger = DiagnosticLogger(self.session_dir)
        self.tegrastats = TegrastatsSampler(self.logger)
        self.context = rs.context()
        self.pipeline = rs.pipeline(self.context)
        self.profile: Optional[rs.pipeline_profile] = None
        self.depth_sensor: Any = None
        self.depth_scale = 0.001
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.filter_options: Dict[str, NumericParameter] = {}
        self.sensor_options: Dict[str, NumericParameter] = {}
        self.display_options = {
            "display_min_m": NumericParameter(
                "display_min_m", "Display minimum (m)", 0.01, 2.0, 0.01, 0.10
            ),
            "display_max_m": NumericParameter(
                "display_max_m", "Display maximum (m)", 0.20, 10.0, 0.01, 5.0
            ),
            "gap_threshold_ms": NumericParameter(
                "gap_threshold_ms", "Gap warning (ms)", 40.0, 2000.0, 1.0,
                float(args.gap_threshold_ms),
            ),
        }
        self.filter_lock = threading.Lock()
        self.filter_enabled = {
            "spatial": bool(args.spatial),
            "temporal": bool(args.temporal),
            "hole_fill": bool(args.hole_fill),
        }
        self.running = threading.Event()
        self.restart_requested = threading.Event()
        self.rebind_requested = threading.Event()
        self.restart_in_progress = threading.Event()
        self.sensor_controls_ready = threading.Event()
        self.control_state_lock = threading.Lock()
        self.control_state = "STARTING"
        self.device_generation = 0
        self.device_lost_at: Optional[float] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.latest_lock = threading.Lock()
        self.latest_dashboard: Optional[np.ndarray] = None
        self.latest_raw: Optional[np.ndarray] = None
        self.latest_filtered: Optional[np.ndarray] = None
        self.latest_frame_number = -1
        self.latest_ready_ns = 0
        self.latest_metrics: Dict[str, Any] = {}
        self.last_gui_latency_ms = 0.0
        self.last_gui_convert_ms = 0.0
        self.last_frame_age_gui_submit_ms = float("nan")
        self.capture_exception: Optional[str] = None
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(None)
        self.timeouts = 0
        self.frame_jumps = 0
        self.stream_gaps = 0
        self.frame_count = 0
        self.sensor_offset_min_ms: Optional[float] = None
        self.device_serial = ""
        self.usb_type = ""
        self.last_reconnect_wait_log = 0.0

    @staticmethod
    def list_devices() -> int:
        devices = rs.context().query_devices()
        if not devices:
            print("No RealSense device detected", file=sys.stderr)
            return 1
        for index, device in enumerate(devices):
            print(json.dumps({
                "index": index,
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
                "usb": device.get_info(rs.camera_info.usb_type_descriptor),
            }, ensure_ascii=False))
        return 0

    def _notification_callback(self, notification: Any) -> None:
        try:
            self.logger.event(
                "DEVICE_NOTIFICATION", severity="WARNING",
                category=str(notification.get_category()),
                notification_severity=str(notification.get_severity()),
                description=notification.get_description(),
                timestamp=notification.get_timestamp(),
                serialized=notification.get_serialized_data(),
            )
        except Exception as exc:
            self.logger.event(
                "DEVICE_NOTIFICATION_PARSE_FAILED", severity="WARNING", error=str(exc)
            )

    def _devices_changed_callback(self, info: Any) -> None:
        try:
            new_devices = list(info.get_new_devices())
            new_identities = []
            for device in new_devices:
                new_identities.append({
                    "name": device.get_info(rs.camera_info.name),
                    "serial": device.get_info(rs.camera_info.serial_number),
                    "usb": device.get_info(rs.camera_info.usb_type_descriptor),
                })
            active_serials = [
                device.get_info(rs.camera_info.serial_number)
                for device in self.context.query_devices()
            ]
            selected_present = (
                self.device_serial in active_serials if self.device_serial else None
            )
            self.logger.event(
                "DEVICE_TOPOLOGY_CHANGED", severity="WARNING",
                new_device_count=len(new_devices),
                new_devices=new_identities,
                active_serials=active_serials,
                selected_serial=self.device_serial,
                selected_device_present=selected_present,
            )
            if self.device_serial and not self.restart_in_progress.is_set():
                if not selected_present:
                    if self.device_lost_at is None:
                        self.device_lost_at = time.monotonic()
                    self.sensor_controls_ready.clear()
                    self._set_control_state("DEVICE_LOST")
                elif any(item["serial"] == self.device_serial for item in new_identities):
                    self.device_lost_at = None
                    self.sensor_controls_ready.clear()
                    self._set_control_state("RECONNECT_PENDING")
                    self.rebind_requested.set()
        except Exception as exc:
            self.logger.event(
                "DEVICE_TOPOLOGY_CALLBACK_FAILED", severity="WARNING", error=str(exc)
            )

    def _set_control_state(self, state: str) -> None:
        with self.control_state_lock:
            self.control_state = state

    def control_status(self) -> str:
        with self.control_state_lock:
            return self.control_state

    def start(self) -> None:
        self.context.set_devices_changed_callback(self._devices_changed_callback)
        config = rs.config()
        if self.args.serial:
            config.enable_device(self.args.serial)
        config.enable_stream(
            rs.stream.depth, self.args.width, self.args.height,
            rs.format.z16, self.args.fps,
        )
        self.profile = self.pipeline.start(config)
        device = self.profile.get_device()
        self.depth_sensor = device.first_depth_sensor()
        self.depth_sensor.set_notifications_callback(self._notification_callback)
        self.depth_scale = float(self.depth_sensor.get_depth_scale())
        stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intrinsics = stream.get_intrinsics()
        identity = {
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "usb": device.get_info(rs.camera_info.usb_type_descriptor),
            "profile": [intrinsics.width, intrinsics.height, self.args.fps],
            "depth_scale": self.depth_scale,
            "intrinsics": {
                "fx": intrinsics.fx, "fy": intrinsics.fy,
                "cx": intrinsics.ppx, "cy": intrinsics.ppy,
            },
        }
        self.device_serial = identity["serial"]
        self.usb_type = identity["usb"]
        self.logger.event("DEVICE_OPENED", **identity)
        self.logger.event(
            "CAMERA_CONTROL_POLICY",
            debounce_ms=self.args.camera_debounce_ms,
            message="hardware options apply after the requested value settles",
        )
        if identity["usb"].startswith("2"):
            self.logger.event(
                "USB2_LINK", severity="WARNING",
                policy="warning_only",
                message="USB 2.x is supported; link instability is diagnostic-only",
            )
        self._build_option_bindings()
        self.device_generation = 1
        self.sensor_controls_ready.set()
        self._set_control_state("ONLINE")
        self.tegrastats.start()

    def _add_option(
        self,
        collection: Dict[str, NumericParameter],
        key: str,
        label: str,
        target: Any,
        option: rs.option,
        restart_required: bool = False,
    ) -> None:
        parameter = NumericParameter.from_target(
            key, label, target, option, restart_required=restart_required
        )
        if parameter is not None:
            collection[key] = parameter

    def _build_option_bindings(self) -> None:
        for key, label, target, option in (
            ("spatial_magnitude", "Magnitude", self.spatial, rs.option.filter_magnitude),
            ("spatial_alpha", "Smooth alpha", self.spatial, rs.option.filter_smooth_alpha),
            ("spatial_delta", "Smooth delta", self.spatial, rs.option.filter_smooth_delta),
            ("spatial_holes", "Internal hole radius", self.spatial, rs.option.holes_fill),
            ("temporal_alpha", "Smooth alpha", self.temporal, rs.option.filter_smooth_alpha),
            ("temporal_delta", "Smooth delta", self.temporal, rs.option.filter_smooth_delta),
            ("temporal_persistence", "Persistence mode", self.temporal, rs.option.holes_fill),
            ("hole_mode", "Fill source mode", self.hole_filling, rs.option.holes_fill),
        ):
            self._add_option(self.filter_options, key, label, target, option)
        for key, label, option, restart_required in COMPETITION_SENSOR_OPTION_SPECS:
            self._add_option(
                self.sensor_options, key, label, self.depth_sensor, option,
                restart_required=restart_required,
            )

    def set_filter_enabled(self, key: str, enabled: bool) -> None:
        with self.filter_lock:
            self.filter_enabled[key] = bool(enabled)
        self.logger.event("FILTER_TOGGLED", key=key, enabled=bool(enabled))

    def filter_state(self) -> Dict[str, bool]:
        with self.filter_lock:
            return dict(self.filter_enabled)

    def reset_parameters(self) -> None:
        for parameter in (*self.filter_options.values(), *self.sensor_options.values(),
                          *self.display_options.values()):
            parameter.reset()
        for key in self.filter_enabled:
            self.set_filter_enabled(key, False)
        self.logger.event("PARAMETERS_RESET_TO_DEFAULTS")

    def _apply_options(
        self,
        force: bool = False,
        sensor_keys: Optional[set] = None,
        include_restart_required: bool = False,
    ) -> None:
        enabled = self.filter_state()
        for key, parameter in self.filter_options.items():
            active = (
                (key.startswith("spatial_") and enabled["spatial"])
                or (key.startswith("temporal_") and enabled["temporal"])
                or (key == "hole_mode" and enabled["hole_fill"])
            )
            parameter.apply(self.logger, active)

        if not self.sensor_controls_ready.is_set():
            return
        auto_exposure = self.sensor_options.get("auto_exposure")
        auto_enabled = auto_exposure is not None and auto_exposure.desired_value() >= 0.5
        for key, parameter in self.sensor_options.items():
            if sensor_keys is not None and key not in sensor_keys:
                continue
            if parameter.restart_required and not include_restart_required:
                continue
            changed = parameter.apply(
                self.logger,
                key not in ("exposure", "gain") or not auto_enabled,
                settle_s=0.0 if force else self.args.camera_debounce_ms / 1000.0,
            )
            if changed and key == "visual_preset":
                for other_key, other in self.sensor_options.items():
                    if other_key != key and not other.restart_required:
                        other.invalidate_applied()

    def _pending_restart_option_keys(self) -> set:
        pending = set()
        for key, parameter in self.sensor_options.items():
            if not parameter.restart_required:
                continue
            if parameter.has_pending_change():
                pending.add(key)
        return pending

    def request_stream_restart(self) -> None:
        if (
            not self.running.is_set()
            or not self.sensor_controls_ready.is_set()
            or self.control_status() != "ONLINE"
        ):
            self.logger.event(
                "STREAM_RESTART_REQUEST_IGNORED",
                severity="WARNING",
                reason="camera_controls_not_online",
                control_state=self.control_status(),
            )
            return
        if not self._pending_restart_option_keys():
            self.logger.event(
                "STREAM_RESTART_REQUEST_IGNORED",
                reason="no_pending_restart_required_options",
            )
            return
        if not self.restart_requested.is_set():
            self.restart_requested.set()
            self.sensor_controls_ready.clear()
            self._set_control_state("RESTART_REQUESTED")
            self.logger.event(
                "STREAM_RESTART_REQUESTED",
                message="capture thread will apply pending options and reopen the stream",
            )

    def _restart_stream(self, reason: str, preserve_all: bool) -> bool:
        self.restart_in_progress.set()
        self.sensor_controls_ready.clear()
        self._set_control_state("RECONNECTING")
        self.logger.event("STREAM_RESTART_BEGIN", reason=reason)
        try:
            preserve_keys = set(SAFE_REAPPLY_AFTER_REENUMERATION)
            if preserve_all:
                restart_keys = self._pending_restart_option_keys()
                if not restart_keys:
                    self.restart_requested.clear()
                    self.sensor_controls_ready.set()
                    self._set_control_state("ONLINE")
                    self.logger.event(
                        "STREAM_RESTART_SKIPPED",
                        reason="no_pending_restart_required_options",
                    )
                    return True
                self.sensor_controls_ready.set()
                self._apply_options(
                    force=True,
                    sensor_keys=restart_keys,
                    include_restart_required=True,
                )
                self.sensor_controls_ready.clear()
                applied_restart_keys = {
                    key for key in restart_keys
                    if np.isfinite(self.sensor_options[key].applied_value())
                    and abs(
                        self.sensor_options[key].desired_value()
                        - self.sensor_options[key].applied_value()
                    ) < self.sensor_options[key].step * 0.5
                }
                rejected_restart_keys = restart_keys - applied_restart_keys
                if rejected_restart_keys:
                    self.logger.event(
                        "RESTART_REQUIRED_OPTIONS_REJECTED",
                        severity="WARNING",
                        keys=sorted(rejected_restart_keys),
                        message="unsupported controls were not retried and do not justify reopening",
                    )
                if not applied_restart_keys:
                    self.restart_requested.clear()
                    self.sensor_controls_ready.set()
                    self._set_control_state("ONLINE")
                    self.logger.event(
                        "STREAM_RESTART_SKIPPED",
                        reason="all_pending_options_rejected",
                    )
                    return True
                preserve_keys.update(applied_restart_keys)
            try:
                self.pipeline.stop()
            except RuntimeError as exc:
                self.logger.event(
                    "PIPELINE_STOP_DURING_RESTART_FAILED", severity="WARNING", error=str(exc)
                )
            time.sleep(0.10)
            config = rs.config()
            config.enable_device(self.device_serial)
            config.enable_stream(
                rs.stream.depth, self.args.width, self.args.height,
                rs.format.z16, self.args.fps,
            )
            try:
                self.profile = self.pipeline.start(config)
            except RuntimeError as exc:
                self.profile = None
                self.restart_requested.clear()
                self.rebind_requested.set()
                self.device_lost_at = self.device_lost_at or time.monotonic()
                self._set_control_state("WAITING_FOR_DEVICE")
                self.logger.event(
                    "STREAM_RESTART_DEFERRED",
                    severity="WARNING",
                    reason=reason,
                    error=str(exc),
                    **self._device_presence(),
                )
                return False
            device = self.profile.get_device()
            serial = device.get_info(rs.camera_info.serial_number)
            if serial != self.device_serial:
                raise RuntimeError(
                    f"stream restart opened serial {serial}, expected {self.device_serial}"
                )
            self.depth_sensor = device.first_depth_sensor()
            self.depth_sensor.set_notifications_callback(self._notification_callback)
            self.depth_scale = float(self.depth_sensor.get_depth_scale())
            self.usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
            for key, parameter in self.sensor_options.items():
                parameter.rebind(
                    self.depth_sensor,
                    preserve_desired=(key in preserve_keys),
                )
            self.sensor_controls_ready.set()
            self._apply_options(
                force=True, sensor_keys=SAFE_REAPPLY_AFTER_REENUMERATION
            )
            self.sensor_offset_min_ms = None
            self.device_lost_at = None
            self.device_generation += 1
            self.restart_requested.clear()
            self.rebind_requested.clear()
            self._set_control_state("ONLINE")
            self.logger.event(
                "DEVICE_CONTROL_REBOUND", reason=reason, serial=serial,
                usb=self.usb_type, generation=self.device_generation,
                safe_reapplied=sorted(SAFE_REAPPLY_AFTER_REENUMERATION),
                unsafe_options_source="device_current_values",
            )
            return True
        finally:
            self.restart_in_progress.clear()

    def _relative_queue_delay(self, arrival_ms: float, sensor_timestamp_ms: float) -> float:
        offset = arrival_ms - sensor_timestamp_ms
        if self.sensor_offset_min_ms is None or offset < self.sensor_offset_min_ms:
            self.sensor_offset_min_ms = offset
        return max(0.0, offset - self.sensor_offset_min_ms)

    def _device_presence(self) -> Dict[str, Any]:
        try:
            devices = self.context.query_devices()
            serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
            return {"device_count": len(devices), "serials": serials}
        except Exception as exc:
            return {"device_query_error": str(exc)}

    def _capture_loop(self) -> None:
        previous_frame: Optional[int] = None
        previous_sensor_ms: Optional[float] = None
        previous_arrival_ns: Optional[int] = None
        fps = 0.0
        last_metric_log = 0.0
        consecutive_timeouts = 0
        try:
            while self.running.is_set():
                if self.rebind_requested.is_set():
                    presence = self._device_presence()
                    if self.device_serial not in presence.get("serials", []):
                        self.sensor_controls_ready.clear()
                        self._set_control_state("WAITING_FOR_DEVICE")
                        now = time.monotonic()
                        if now - self.last_reconnect_wait_log >= 2.0:
                            self.last_reconnect_wait_log = now
                            self.logger.event(
                                "WAITING_FOR_DEVICE",
                                severity="WARNING",
                                selected_serial=self.device_serial,
                                **presence,
                            )
                        time.sleep(0.20)
                        continue
                    self._restart_stream("device_reenumerated", preserve_all=False)
                    previous_frame = None
                    previous_sensor_ms = None
                    previous_arrival_ns = None
                    fps = 0.0
                    consecutive_timeouts = 0
                    continue
                if self.restart_requested.is_set():
                    self._restart_stream("manual_restart", preserve_all=True)
                    previous_frame = None
                    previous_sensor_ms = None
                    previous_arrival_ns = None
                    fps = 0.0
                    consecutive_timeouts = 0
                    continue
                wait_start_ns = time.monotonic_ns()
                try:
                    frames = self.pipeline.wait_for_frames(self.args.timeout_ms)
                except RuntimeError as exc:
                    self.timeouts += 1
                    consecutive_timeouts += 1
                    now_ns = time.monotonic_ns()
                    presence = self._device_presence()
                    self.logger.event(
                        "STREAM_TIMEOUT", severity="ERROR",
                        error=str(exc), timeout_ms=self.args.timeout_ms,
                        consecutive_timeouts=consecutive_timeouts,
                        since_last_frame_ms=(
                            (now_ns - previous_arrival_ns) / 1e6
                            if previous_arrival_ns is not None else None
                        ),
                        last_frame=previous_frame,
                        **presence,
                    )
                    if (
                        consecutive_timeouts >= self.args.disconnect_abort_count
                        and presence.get("device_count") == 0
                        and self.device_lost_at is not None
                        and time.monotonic() - self.device_lost_at
                            >= self.args.reconnect_grace_seconds
                    ):
                        self.logger.event(
                            "STREAM_ENTERED_RECONNECT_WAIT",
                            severity="WARNING",
                            consecutive_timeouts=consecutive_timeouts,
                            last_frame=previous_frame,
                        )
                        try:
                            self.pipeline.stop()
                        except RuntimeError as stop_exc:
                            self.logger.event(
                                "PIPELINE_STOP_FOR_RECONNECT_FAILED",
                                severity="WARNING",
                                error=str(stop_exc),
                            )
                        self.profile = None
                        self.rebind_requested.set()
                        self.sensor_controls_ready.clear()
                        self._set_control_state("WAITING_FOR_DEVICE")
                        continue
                    continue
                arrival_ns = time.monotonic_ns()
                arrival_wall_ms = time.time_ns() / 1e6
                wait_ms = (arrival_ns - wait_start_ns) / 1e6
                consecutive_timeouts = 0
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    self.logger.event("FRAMESET_WITHOUT_DEPTH", severity="WARNING")
                    continue
                frame_metadata = read_frame_metadata(depth_frame)
                laser_parameter = self.sensor_options.get("laser")
                actual_laser = frame_metadata.get("actual_laser")
                if (
                    laser_parameter is not None and actual_laser is not None and
                    laser_parameter.confirm_from_frame_metadata(actual_laser)
                ):
                    self.logger.event(
                        "OPTION_CONFIRMED_BY_FRAME_METADATA", key="laser",
                        actual=actual_laser,
                    )

                frame_number = int(depth_frame.get_frame_number())
                sensor_ms = float(depth_frame.get_timestamp())
                timestamp_domain = str(depth_frame.get_frame_timestamp_domain())
                frame_age_receive_ms = absolute_frame_age_ms(
                    sensor_ms, timestamp_domain, arrival_wall_ms
                )
                arrival_ms = arrival_ns / 1e6
                host_gap_ms = (
                    (arrival_ns - previous_arrival_ns) / 1e6
                    if previous_arrival_ns is not None else 0.0
                )
                sensor_gap_ms = (
                    sensor_ms - previous_sensor_ms
                    if previous_sensor_ms is not None else 0.0
                )
                if previous_frame is not None:
                    threshold = self.display_options["gap_threshold_ms"].desired_value()
                    for event in classify_stream_discontinuity(
                        previous_frame, frame_number, host_gap_ms, sensor_gap_ms, threshold
                    ):
                        kind = event.pop("kind")
                        if kind == "FRAME_JUMP":
                            self.frame_jumps += 1
                        if kind == "STREAM_GAP":
                            self.stream_gaps += 1
                        self.logger.event(kind, severity="WARNING", frame=frame_number, **event)
                previous_frame = frame_number
                previous_sensor_ms = sensor_ms
                previous_arrival_ns = arrival_ns
                if host_gap_ms > 0:
                    instantaneous = 1000.0 / host_gap_ms
                    fps = instantaneous if fps <= 0 else 0.9 * fps + 0.1 * instantaneous

                self._apply_options()
                filtered: Any = depth_frame
                filter_start_ns = time.monotonic_ns()
                state = self.filter_state()
                if state["spatial"]:
                    filtered = self.spatial.process(filtered)
                if state["temporal"]:
                    filtered = self.temporal.process(filtered)
                if state["hole_fill"]:
                    filtered = self.hole_filling.process(filtered)
                raw_depth = np.asanyarray(depth_frame.get_data()).copy()
                filtered_depth = np.asanyarray(filtered.get_data()).copy()
                filter_ms = (time.monotonic_ns() - filter_start_ns) / 1e6

                display_min = self.display_options["display_min_m"].desired_value()
                display_max = max(
                    display_min + 0.01,
                    self.display_options["display_max_m"].desired_value(),
                )
                render_start_ns = time.monotonic_ns()
                dashboard, stats = compose_dashboard(
                    raw_depth, filtered_depth, self.depth_scale,
                    display_min, display_max, fps, frame_number,
                )
                _put_label(
                    dashboard,
                    f"frame {frame_number}  fps {fps:.1f}  range {display_min:.2f}-{display_max:.2f}m",
                    4,
                    (255, 255, 255),
                )
                _put_label(
                    dashboard,
                    f"spatial={int(state['spatial'])} temporal={int(state['temporal'])} "
                    f"hole_fill={int(state['hole_fill'])}",
                    5,
                    (255, 255, 255),
                )
                render_ms = (time.monotonic_ns() - render_start_ns) / 1e6
                ready_ns = time.monotonic_ns()
                ready_wall_ms = time.time_ns() / 1e6
                app_latency_ms = (ready_ns - arrival_ns) / 1e6
                frame_age_ready_ms = absolute_frame_age_ms(
                    sensor_ms, timestamp_domain, ready_wall_ms
                )
                relative_queue_ms = self._relative_queue_delay(arrival_ms, sensor_ms)
                gpu = self.tegrastats.snapshot()
                memory = psutil.virtual_memory()
                queue_parameter = self.sensor_options.get("frames_queue")
                queue_size = (
                    queue_parameter.applied_value()
                    if queue_parameter is not None else float("nan")
                )
                laser_requested = (
                    laser_parameter.desired_value()
                    if laser_parameter is not None else float("nan")
                )
                laser_applied = (
                    laser_parameter.applied_value()
                    if laser_parameter is not None else float("nan")
                )
                metrics: Dict[str, Any] = {
                    "frame_number": frame_number,
                    "fps": fps,
                    "wait_ms": wait_ms,
                    "host_gap_ms": host_gap_ms,
                    "sensor_gap_ms": sensor_gap_ms,
                    "relative_queue_ms": relative_queue_ms,
                    "frame_timestamp_ms": sensor_ms,
                    "timestamp_domain": timestamp_domain,
                    "latency_clock_valid": int(np.isfinite(frame_age_receive_ms)),
                    "frame_age_receive_ms": frame_age_receive_ms,
                    "frame_age_ready_ms": frame_age_ready_ms,
                    "filter_ms": filter_ms,
                    "render_ms": render_ms,
                    "app_latency_ms": app_latency_ms,
                    "gui_latency_ms": self.last_gui_latency_ms,
                    "gui_convert_ms": self.last_gui_convert_ms,
                    "frame_age_gui_submit_ms": self.last_frame_age_gui_submit_ms,
                    "process_cpu_percent": self.process.cpu_percent(None),
                    "system_cpu_percent": psutil.cpu_percent(None),
                    "process_rss_mb": self.process.memory_info().rss / (1024 * 1024),
                    "system_ram_percent": memory.percent,
                    "gpu_percent": gpu.get("gpu_percent", float("nan")),
                    "gpu_temp_c": gpu.get("gpu_temp_c", float("nan")),
                    "timeouts": self.timeouts,
                    "frame_jumps": self.frame_jumps,
                    "stream_gaps": self.stream_gaps,
                    "frames_queue_size": queue_size,
                    "usb_type": self.usb_type,
                    "laser_requested": laser_requested,
                    "laser_applied": laser_applied,
                    "control_state": self.control_status(),
                    "device_generation": self.device_generation,
                    **frame_metadata,
                    **stats,
                }
                with self.latest_lock:
                    self.latest_dashboard = dashboard
                    self.latest_raw = raw_depth
                    self.latest_filtered = filtered_depth
                    self.latest_frame_number = frame_number
                    self.latest_ready_ns = ready_ns
                    self.latest_metrics = metrics
                self.frame_count += 1
                now = time.monotonic()
                if now - last_metric_log >= 1.0:
                    self.logger.metric(metrics)
                    if self.args.headless:
                        print(
                            f"frame={frame_number} fps={fps:.1f} wait={wait_ms:.1f}ms "
                            f"app={app_latency_ms:.1f}ms queue_rel={relative_queue_ms:.1f}ms "
                            f"cpu={metrics['process_cpu_percent']:.1f}% "
                            f"gpu={metrics['gpu_percent']}% invalid={stats['raw_invalid_fraction']:.3f}"
                        )
                    last_metric_log = now
                if self.args.frames > 0 and self.frame_count >= self.args.frames:
                    self.running.clear()
        except Exception as exc:
            self.capture_exception = str(exc)
            self.logger.event(
                "CAPTURE_EXCEPTION", severity="ERROR", error=str(exc),
                last_frame=previous_frame,
            )
            self.running.clear()

    def metrics_snapshot(self) -> Dict[str, Any]:
        with self.latest_lock:
            return dict(self.latest_metrics)

    def latest_image(self) -> Tuple[Optional[np.ndarray], int, int, Dict[str, Any]]:
        with self.latest_lock:
            return (
                self.latest_dashboard,
                self.latest_frame_number,
                self.latest_ready_ns,
                dict(self.latest_metrics),
            )

    def update_gui_timing(
        self,
        frame_number: int,
        schedule_ms: float,
        convert_ms: float,
        frame_age_gui_submit_ms: float,
    ) -> None:
        self.last_gui_latency_ms = max(0.0, float(schedule_ms))
        self.last_gui_convert_ms = max(0.0, float(convert_ms))
        self.last_frame_age_gui_submit_ms = float(frame_age_gui_submit_ms)
        with self.latest_lock:
            if self.latest_frame_number == frame_number:
                self.latest_metrics["gui_latency_ms"] = self.last_gui_latency_ms
                self.latest_metrics["gui_convert_ms"] = self.last_gui_convert_ms
                self.latest_metrics["frame_age_gui_submit_ms"] = (
                    self.last_frame_age_gui_submit_ms
                )

    def settings(self) -> Dict[str, Any]:
        return {
            "profile": [self.args.width, self.args.height, self.args.fps],
            "policy_max_depth_m": POLICY_MAX_DEPTH_M,
            "filters_enabled": self.filter_state(),
            "filters": {key: value.describe() for key, value in self.filter_options.items()},
            "sensor_options": {
                key: value.describe() for key, value in self.sensor_options.items()
            },
            "display": {key: value.describe() for key, value in self.display_options.items()},
            "metrics": self.metrics_snapshot(),
            "logs": {
                "events": str(self.logger.events_path),
                "metrics": str(self.logger.metrics_path),
                "display_latency": str(self.logger.display_metrics_path),
            },
        }

    def save_snapshot(self) -> Optional[Path]:
        with self.latest_lock:
            if self.latest_raw is None or self.latest_filtered is None or self.latest_dashboard is None:
                self.logger.event("SNAPSHOT_SKIPPED", severity="WARNING", reason="no_frame")
                return None
            raw_depth = self.latest_raw.copy()
            filtered_depth = self.latest_filtered.copy()
            dashboard = self.latest_dashboard.copy()
            frame_number = self.latest_frame_number
        destination = self.session_dir / (
            f"frame_{frame_number:010d}_{int(time.time() * 1000) % 1000:03d}"
        )
        destination.mkdir(parents=True, exist_ok=False)
        raw_invalid = compute_invalid_mask(raw_depth, self.depth_scale)
        filtered_invalid = compute_invalid_mask(filtered_depth, self.depth_scale)
        cv2.imwrite(str(destination / "raw_depth_units.png"), raw_depth)
        cv2.imwrite(str(destination / "filtered_depth_units.png"), filtered_depth)
        cv2.imwrite(str(destination / "raw_invalid_mask.png"), raw_invalid.astype(np.uint8) * 255)
        cv2.imwrite(
            str(destination / "filtered_invalid_mask.png"),
            filtered_invalid.astype(np.uint8) * 255,
        )
        cv2.imwrite(str(destination / "dashboard.jpg"), dashboard)
        np.save(destination / "raw_depth_units.npy", raw_depth)
        np.save(destination / "filtered_depth_units.npy", filtered_depth)
        (destination / "settings.json").write_text(
            json.dumps(self.settings(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.logger.event("SNAPSHOT_SAVED", frame=frame_number, path=str(destination))
        return destination

    def run(self) -> int:
        try:
            self.start()
            self.running.set()
            if self.args.headless:
                self._capture_loop()
            else:
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.capture_thread.start()
                DepthTunerGui(self).run()
                self.running.clear()
                if self.capture_thread.is_alive():
                    self.capture_thread.join(timeout=max(2.0, self.args.timeout_ms / 1000 + 1.0))
            if self.args.save_last:
                self.save_snapshot()
            return 1 if self.capture_exception else 0
        finally:
            self.running.clear()
            if self.profile is not None:
                try:
                    self.pipeline.stop()
                except RuntimeError as exc:
                    self.logger.event("PIPELINE_STOP_FAILED", severity="WARNING", error=str(exc))
            self.tegrastats.stop()
            (self.session_dir / "session.json").write_text(
                json.dumps(self.settings(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.logger.event(
                "SESSION_STOP", frames=self.frame_count, timeouts=self.timeouts,
                frame_jumps=self.frame_jumps, stream_gaps=self.stream_gaps,
            )
            self.logger.close()


class NumericControl:
    def __init__(
        self,
        parent: ttk.Frame,
        row: int,
        parameter: NumericParameter,
    ) -> None:
        self.parameter = parameter
        self.scale_var = tk.DoubleVar(value=parameter.desired_value())
        self.entry_var = tk.StringVar(value=self._format(parameter.desired_value()))
        ttk.Label(parent, text=parameter.label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        self.scale = ttk.Scale(
            parent, from_=parameter.minimum, to=parameter.maximum,
            variable=self.scale_var, command=self._from_scale,
        )
        self.scale.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        self.entry = ttk.Entry(parent, width=11, textvariable=self.entry_var)
        self.entry.grid(row=row, column=2, sticky="ew", padx=4, pady=4)
        self.entry.bind("<Return>", self._from_entry)
        self.entry.bind("<FocusOut>", self._from_entry)
        self.range_label = ttk.Label(
            parent,
            text=self._range_text(parameter),
            foreground="#666666",
        )
        self.range_label.grid(row=row, column=3, sticky="w", padx=4, pady=4)
        self.reset_button = ttk.Button(parent, text="Reset", width=7, command=self.reset)
        self.reset_button.grid(
            row=row, column=4, sticky="e", padx=4, pady=4
        )
        parent.columnconfigure(1, weight=1)

    @staticmethod
    def _format(value: float) -> str:
        return f"{value:.5g}"

    @staticmethod
    def _range_text(parameter: NumericParameter) -> str:
        text = (
            f"{parameter.minimum:g}..{parameter.maximum:g}\n"
            f"default {parameter.default_value:g}"
        )
        if parameter.maximum < parameter.device_maximum:
            text += f" | device max {parameter.device_maximum:g}"
        if parameter.restart_required:
            text += "\nrestart stream"
        return text

    def _from_scale(self, value: str) -> None:
        actual = self.parameter.set_desired(float(value))
        self.entry_var.set(self._format(actual))

    def _from_entry(self, _: Any = None) -> None:
        try:
            actual = self.parameter.set_desired(float(self.entry_var.get()))
        except ValueError:
            actual = self.parameter.desired_value()
        self.scale_var.set(actual)
        self.entry_var.set(self._format(actual))

    def reset(self) -> None:
        actual = self.parameter.reset()
        self.scale_var.set(actual)
        self.entry_var.set(self._format(actual))

    def refresh(self) -> None:
        self.scale.configure(from_=self.parameter.minimum, to=self.parameter.maximum)
        self.range_label.configure(text=self._range_text(self.parameter))
        actual = self.parameter.desired_value()
        if abs(self.scale_var.get() - actual) > self.parameter.step * 0.5:
            self.scale_var.set(actual)
            self.entry_var.set(self._format(actual))

    def set_enabled(self, enabled: bool) -> None:
        widgets = (self.scale, self.entry, self.reset_button)
        for widget in widgets:
            if enabled:
                widget.state(["!disabled"])
            else:
                widget.state(["disabled"])


class DepthTunerGui:
    def __init__(self, tuner: DepthTuner) -> None:
        self.tuner = tuner
        self.root = tk.Tk()
        self.root.title(WINDOW_DEPTH)
        self.root.geometry("1540x900")
        self.root.minsize(1180, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("q", lambda _: self.close())
        self.root.bind("<Escape>", lambda _: self.close())
        self.root.bind("s", lambda _: self.tuner.save_snapshot())
        self.root.bind("p", lambda _: self.print_settings())
        self.root.bind("r", lambda _: self.reset())
        self.controls: List[NumericControl] = []
        self.sensor_controls: List[NumericControl] = []
        self.filter_vars: Dict[str, tk.BooleanVar] = {}
        self.status_vars: Dict[str, tk.StringVar] = {}
        self.last_frame_number = -1
        self.photo: Optional[tk.PhotoImage] = None
        self.last_event_count = 0
        self._build()

    def _build(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Section.TLabelframe.Label", font=("Sans", 10, "bold"))
        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.Frame(paned, width=455)
        right = ttk.Frame(paned)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        notebook = ttk.Notebook(left)
        notebook.pack(fill=tk.BOTH, expand=True)
        filters_tab = ttk.Frame(notebook)
        camera_tab = ttk.Frame(notebook)
        monitor_tab = ttk.Frame(notebook)
        notebook.add(filters_tab, text="Filters")
        notebook.add(camera_tab, text="Camera")
        notebook.add(monitor_tab, text="Monitor & Logs")
        self._build_filters(filters_tab)
        self._build_camera(camera_tab)
        self._build_monitor(monitor_tab)

        status = ttk.LabelFrame(right, text="Live timing and compute", style="Section.TLabelframe")
        status.pack(fill=tk.X, padx=4, pady=(0, 6))
        status_keys = [
            ("stream", "Stream"), ("fps", "FPS"), ("wait_ms", "Capture wait"),
            ("timestamp_domain", "Timestamp domain"),
            ("frame_age_receive_ms", "Frame age @ receive"),
            ("frame_age_ready_ms", "Frame age @ app ready"),
            ("frame_age_gui_submit_ms", "Frame age @ GUI submit"),
            ("gui_convert_ms", "GUI conversion"),
            ("host_gap_ms", "Host gap"), ("sensor_gap_ms", "Sensor gap"),
            ("relative_queue_ms", "Queue extra (relative)"), ("filter_ms", "Filter"),
            ("render_ms", "Render"), ("gui_latency_ms", "GUI scheduling"),
            ("process_cpu_percent", "Process CPU"), ("gpu_percent", "GPU"),
            ("process_rss_mb", "Process RAM"), ("raw_invalid_fraction", "Raw invalid"),
            ("timeouts", "Timeouts"), ("frame_jumps", "Frame jumps"),
            ("stream_gaps", "Stream gaps"),
            ("actual_exposure_us", "Actual exposure"),
            ("actual_gain", "Actual gain"),
            ("laser_requested", "Laser requested"),
            ("laser_applied", "Laser applied"),
            ("actual_laser", "Laser frame actual"),
            ("frames_queue_size", "Frame queue"), ("usb_type", "USB link"),
            ("control_state", "Camera controls"),
            ("device_generation", "Device generation"),
        ]
        for index, (key, label) in enumerate(status_keys):
            row, column = divmod(index, 4)
            cell = ttk.Frame(status)
            cell.grid(row=row, column=column, sticky="ew", padx=6, pady=4)
            ttk.Label(cell, text=label, foreground="#666666").pack(anchor="w")
            variable = tk.StringVar(value="--")
            ttk.Label(cell, textvariable=variable, font=("Sans", 10, "bold")).pack(anchor="w")
            self.status_vars[key] = variable
            status.columnconfigure(column, weight=1)
        self.image_label = ttk.Label(right, anchor="center")
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def _filter_toggle(self, parent: ttk.Frame, key: str, text: str) -> None:
        variable = tk.BooleanVar(value=self.tuner.filter_state()[key])
        self.filter_vars[key] = variable
        ttk.Checkbutton(
            parent, text=text, variable=variable,
            command=lambda: self.tuner.set_filter_enabled(key, variable.get()),
        ).pack(anchor="w", padx=8, pady=(6, 2))

    def _parameter_group(
        self,
        parent: ttk.Frame,
        title: str,
        keys: List[str],
        source: Dict[str, NumericParameter],
        sensor_controls: bool = False,
    ) -> None:
        group = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe")
        group.pack(fill=tk.X, padx=8, pady=6)
        for row, key in enumerate(keys):
            if key in source:
                control = NumericControl(group, row, source[key])
                self.controls.append(control)
                if sensor_controls:
                    self.sensor_controls.append(control)

    def _build_filters(self, parent: ttk.Frame) -> None:
        self._filter_toggle(parent, "spatial", "Enable spatial filter")
        self._parameter_group(
            parent, "Spatial filter",
            ["spatial_magnitude", "spatial_alpha", "spatial_delta", "spatial_holes"],
            self.tuner.filter_options,
        )
        self._filter_toggle(parent, "temporal", "Enable temporal filter")
        self._parameter_group(
            parent, "Temporal filter",
            ["temporal_alpha", "temporal_delta", "temporal_persistence"],
            self.tuner.filter_options,
        )
        self._filter_toggle(parent, "hole_fill", "Enable independent hole filling")
        self._parameter_group(
            parent, "Independent hole filling", ["hole_mode"], self.tuner.filter_options
        )

    def _build_camera(self, parent: ttk.Frame) -> None:
        self._parameter_group(
            parent, "Stereo projector and preset",
            ["visual_preset", "emitter", "laser"],
            self.tuner.sensor_options,
            sensor_controls=True,
        )
        self._parameter_group(
            parent, "Automatic exposure constraints",
            [
                "auto_exposure", "auto_exposure_limit_toggle", "auto_exposure_limit",
                "auto_gain_limit_toggle", "auto_gain_limit", "exposure", "gain",
            ],
            self.tuner.sensor_options,
            sensor_controls=True,
        )
        self._parameter_group(
            parent, "Latency, timestamps and device diagnostics",
            ["frames_queue", "global_time", "error_polling"],
            self.tuner.sensor_options,
            sensor_controls=True,
        )
        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, padx=8, pady=6)
        self.restart_button = ttk.Button(
            actions, text="Apply limits & restart stream",
            command=self.tuner.request_stream_restart,
        )
        self.restart_button.pack(side=tk.LEFT, padx=3)
        ttk.Label(
            actions,
            text=(
                "Applies only pending exposure/gain limits, then reopens the stream. "
                "Rejected controls are not retried; disconnects wait for automatic recovery."
            ),
            foreground="#666666", wraplength=300,
        ).pack(side=tk.LEFT, padx=8)

    def _build_monitor(self, parent: ttk.Frame) -> None:
        self._parameter_group(
            parent, "Display and detection",
            ["display_min_m", "display_max_m", "gap_threshold_ms"],
            self.tuner.display_options,
        )
        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(buttons, text="Save snapshot", command=self.tuner.save_snapshot).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(buttons, text="Print settings", command=self.print_settings).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(buttons, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=3)
        ttk.Button(buttons, text="Quit", command=self.close).pack(side=tk.RIGHT, padx=3)
        paths = ttk.LabelFrame(parent, text="Persistent diagnostics", style="Section.TLabelframe")
        paths.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(paths, text=f"Events: {self.tuner.logger.events_path}", wraplength=400).pack(
            anchor="w", padx=6, pady=3
        )
        ttk.Label(paths, text=f"Metrics: {self.tuner.logger.metrics_path}", wraplength=400).pack(
            anchor="w", padx=6, pady=3
        )
        ttk.Label(
            paths,
            text=f"Display latency: {self.tuner.logger.display_metrics_path}",
            wraplength=400,
        ).pack(anchor="w", padx=6, pady=3)
        event_group = ttk.LabelFrame(parent, text="Recent stream events", style="Section.TLabelframe")
        event_group.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.event_text = tk.Text(event_group, height=18, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(event_group, orient=tk.VERTICAL, command=self.event_text.yview)
        self.event_text.configure(yscrollcommand=scrollbar.set)
        self.event_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def print_settings(self) -> None:
        print(json.dumps(self.tuner.settings(), indent=2, ensure_ascii=False))

    def reset(self) -> None:
        self.tuner.reset_parameters()
        for key, variable in self.filter_vars.items():
            variable.set(self.tuner.filter_state()[key])
        for control in self.controls:
            control.refresh()
        self._sync_camera_control_state()

    def _sync_camera_control_state(self) -> None:
        controls_ready = (
            self.tuner.sensor_controls_ready.is_set()
            and self.tuner.control_status() == "ONLINE"
        )
        for control in self.sensor_controls:
            control.set_enabled(controls_ready)
        restart_ready = (
            controls_ready
            and self.tuner.running.is_set()
            and not self.tuner.restart_in_progress.is_set()
            and bool(self.tuner._pending_restart_option_keys())
        )
        if restart_ready:
            self.restart_button.state(["!disabled"])
        else:
            self.restart_button.state(["disabled"])

    @staticmethod
    def _metric_text(key: str, value: Any) -> str:
        if value is None or value == "" or (isinstance(value, float) and not np.isfinite(value)):
            return "--"
        if key == "timestamp_domain":
            return str(value).rsplit(".", 1)[-1]
        if key in ("fps",):
            return f"{float(value):.1f}"
        if key.endswith("_ms"):
            return f"{float(value):.1f} ms"
        if key.endswith("_us"):
            return f"{float(value):.0f} us"
        if key in ("process_cpu_percent", "gpu_percent"):
            return f"{float(value):.1f}%"
        if key == "process_rss_mb":
            return f"{float(value):.0f} MB"
        if key.endswith("fraction"):
            return f"{float(value):.1%}"
        return str(value)

    def _update_status(self, metrics: Dict[str, Any]) -> None:
        stream_status = "RUNNING" if self.tuner.running.is_set() else "STOPPED"
        if self.tuner.capture_exception:
            stream_status = "ERROR"
        self.status_vars["stream"].set(stream_status)
        for key, variable in self.status_vars.items():
            if key != "stream":
                variable.set(self._metric_text(key, metrics.get(key)))

    def _update_events(self) -> None:
        records = self.tuner.logger.recent_records()
        if len(records) == self.last_event_count:
            return
        self.last_event_count = len(records)
        lines = []
        for record in records[-30:]:
            fields = {k: v for k, v in record.items() if k not in ("wall_time", "monotonic_s")}
            lines.append(json.dumps(fields, ensure_ascii=False, default=str))
        self.event_text.configure(state="normal")
        self.event_text.delete("1.0", tk.END)
        self.event_text.insert(tk.END, "\n".join(lines))
        self.event_text.see(tk.END)
        self.event_text.configure(state="disabled")

    def _update(self) -> None:
        if not self.root.winfo_exists():
            return
        dashboard, frame_number, ready_ns, metrics = self.tuner.latest_image()
        metrics["control_state"] = self.tuner.control_status()
        metrics["device_generation"] = self.tuner.device_generation
        laser_parameter = self.tuner.sensor_options.get("laser")
        if laser_parameter is not None:
            metrics["laser_requested"] = laser_parameter.desired_value()
            metrics["laser_applied"] = laser_parameter.applied_value()
        if dashboard is not None and frame_number != self.last_frame_number:
            gui_start_ns = time.monotonic_ns()
            gui_latency_ms = (gui_start_ns - ready_ns) / 1e6
            rgb = cv2.cvtColor(dashboard, cv2.COLOR_BGR2RGB)
            available_width = max(640, self.image_label.winfo_width() - 8)
            available_height = max(360, self.image_label.winfo_height() - 8)
            scale = min(
                1.0,
                available_width / rgb.shape[1],
                available_height / rgb.shape[0],
            )
            size = (max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale)))
            resized = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
            header = f"P6\n{size[0]} {size[1]}\n255\n".encode("ascii")
            self.photo = tk.PhotoImage(data=header + resized.tobytes(), format="PPM")
            self.image_label.configure(image=self.photo)
            gui_submit_ns = time.monotonic_ns()
            gui_submit_wall_ms = time.time_ns() / 1e6
            gui_convert_ms = (gui_submit_ns - gui_start_ns) / 1e6
            frame_age_gui_submit_ms = absolute_frame_age_ms(
                float(metrics.get("frame_timestamp_ms", float("nan"))),
                str(metrics.get("timestamp_domain", "")),
                gui_submit_wall_ms,
            )
            self.tuner.update_gui_timing(
                frame_number,
                gui_latency_ms,
                gui_convert_ms,
                frame_age_gui_submit_ms,
            )
            self.last_frame_number = frame_number
            metrics["gui_latency_ms"] = gui_latency_ms
            metrics["gui_convert_ms"] = gui_convert_ms
            metrics["frame_age_gui_submit_ms"] = frame_age_gui_submit_ms
            state = self.tuner.filter_state()
            temporal = self.tuner.filter_options
            self.tuner.logger.display_metric({
                **metrics,
                "gui_schedule_ms": gui_latency_ms,
                "temporal_enabled": int(state["temporal"]),
                "temporal_alpha": (
                    temporal["temporal_alpha"].desired_value()
                    if "temporal_alpha" in temporal else ""
                ),
                "temporal_delta": (
                    temporal["temporal_delta"].desired_value()
                    if "temporal_delta" in temporal else ""
                ),
                "temporal_persistence": (
                    temporal["temporal_persistence"].desired_value()
                    if "temporal_persistence" in temporal else ""
                ),
            })
        self._update_status(metrics)
        self._update_events()
        for control in self.controls:
            control.refresh()
        self._sync_camera_control_state()
        if self.tuner.running.is_set():
            self.root.after(50, self._update)
        elif self.tuner.capture_exception:
            self.root.after(1500, self.close)
        elif self.tuner.args.frames > 0:
            self.root.after(300, self.close)

    def close(self) -> None:
        self.tuner.running.clear()
        if self.root.winfo_exists():
            self.root.destroy()

    def run(self) -> None:
        self.root.after(20, self._update)
        self.root.mainloop()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live RealSense depth/invalid-pixel viewer and filter tuner"
    )
    parser.add_argument("--serial", default="", help="RealSense serial number")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--gap-threshold-ms", type=float, default=120.0)
    parser.add_argument(
        "--camera-debounce-ms",
        type=float,
        default=400.0,
        help="wait after the last hardware-option change before applying it",
    )
    parser.add_argument(
        "--disconnect-abort-count",
        type=int,
        default=3,
        help="exit after this many consecutive timeouts while the device is absent",
    )
    parser.add_argument(
        "--reconnect-grace-seconds",
        type=float,
        default=10.0,
        help="wait this long for the selected serial to re-enumerate before aborting",
    )
    parser.add_argument("--display", default="", help="X11 display, for example :1")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--hole-fill", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="0 means run until quit")
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "logs" / "depth_tuner"),
    )
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_devices:
        return DepthTuner.list_devices()
    if args.display:
        os.environ["DISPLAY"] = args.display
    if not args.headless and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        raise RuntimeError(
            "DISPLAY is empty. Run from the robot desktop, use --display :1, "
            "or use --headless."
        )
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("width, height and fps must be positive")
    if args.timeout_ms <= 0 or args.gap_threshold_ms <= 0:
        raise ValueError("timeout and gap threshold must be positive")
    if args.camera_debounce_ms < 0:
        raise ValueError("camera debounce must be non-negative")
    if args.disconnect_abort_count <= 0:
        raise ValueError("disconnect abort count must be positive")
    if args.reconnect_grace_seconds <= 0:
        raise ValueError("reconnect grace period must be positive")
    return DepthTuner(args).run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
