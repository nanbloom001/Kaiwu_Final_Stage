#!/usr/bin/env python3
"""Buffered, read-only visualization of the Go2 rt/uwbstate topic.

The viewer never enables tracking, switches robot modes, or publishes motion
commands. Pitch is displayed as a diagnostic value only; it is not presented
as a reliable signed height estimate.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from monitor_uwb import load_sdk


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def circular_mean(values: list[float]) -> float:
    return math.atan2(
        statistics.fmean(math.sin(value) for value in values),
        statistics.fmean(math.cos(value) for value in values),
    )


def smooth_angle(previous: float, target: float, alpha: float) -> float:
    return wrap_angle(previous + alpha * wrap_angle(target - previous))


def planar_position(
    bearing: float, pitch: float, distance: float, distance_mode: str
) -> tuple[float, float, float]:
    """Return base-frame X/Y and the radius used by the navigation view.

    ``raw`` treats distance_est as a ground-plane range. This is intentionally
    a 2D navigation approximation. ``projected`` uses the spherical projection
    d*cos(pitch) from Unitree's field definitions.
    """
    radius = distance
    if distance_mode == "projected":
        radius = max(0.0, distance * math.cos(pitch))
    return radius * math.cos(bearing), radius * math.sin(bearing), radius


@dataclass(frozen=True)
class UwbSample:
    received_at: float
    bearing: float
    pitch: float
    distance: float
    yaw: float
    tag_roll: float
    tag_pitch: float
    tag_yaw: float
    base_roll: float
    base_pitch: float
    base_yaw: float
    error_state: int
    enabled_from_app: int
    joy_mode: int
    channel: int

    @classmethod
    def from_message(cls, msg) -> "UwbSample":
        return cls(
            received_at=time.monotonic(),
            bearing=float(msg.orientation_est), pitch=float(msg.pitch_est),
            distance=float(msg.distance_est), yaw=float(msg.yaw_est),
            tag_roll=float(msg.tag_roll), tag_pitch=float(msg.tag_pitch),
            tag_yaw=float(msg.tag_yaw), base_roll=float(msg.base_roll),
            base_pitch=float(msg.base_pitch), base_yaw=float(msg.base_yaw),
            error_state=int(msg.error_state), enabled_from_app=int(msg.enabled_from_app),
            joy_mode=int(msg.joy_mode), channel=int(msg.channel),
        )

    @property
    def valid_measurement(self) -> bool:
        values = (
            self.bearing, self.pitch, self.distance, self.yaw, self.tag_roll,
            self.tag_pitch, self.tag_yaw, self.base_roll, self.base_pitch, self.base_yaw,
        )
        return (
            self.error_state == 0 and self.distance >= 0.0
            and all(math.isfinite(value) for value in values)
        )


@dataclass(frozen=True)
class FilteredPose:
    updated_at: float
    x: float
    y: float
    radius: float
    bearing: float
    pitch: float
    distance: float
    yaw: float
    vx: float
    vy: float
    yaw_rate: float


@dataclass(frozen=True)
class ReceiverSnapshot:
    raw: UwbSample | None
    filtered: FilteredPose | None
    received: int
    accepted: int
    hz: float
    queue_size: int
    queue_span: float
    trail: list[tuple[float, float]]
    gap_p95: float
    jitter: float
    raw_speed: float
    quality: str


class UwbReceiver:
    """Keep raw frames and produce a robust, time-aware filtered pose."""

    def __init__(
        self, distance_mode: str, buffer_seconds: float,
        median_window: float, filter_tau: float,
    ) -> None:
        self.distance_mode = distance_mode
        self.buffer_seconds = buffer_seconds
        self.median_window = median_window
        self.filter_tau = filter_tau
        self.lock = threading.Lock()
        self.frames: deque[UwbSample] = deque()
        self.recent_times: deque[float] = deque()
        self.trail: deque[tuple[float, float, float]] = deque()
        self.raw: UwbSample | None = None
        self.filtered: FilteredPose | None = None
        self.received = 0
        self.accepted = 0

    def callback(self, msg) -> None:
        self.add(UwbSample.from_message(msg))

    def add(self, sample: UwbSample) -> None:
        with self.lock:
            self.raw = sample
            self.received += 1
            self.frames.append(sample)
            self.recent_times.append(sample.received_at)
            self._expire(sample.received_at)
            if not sample.valid_measurement:
                return

            self.accepted += 1
            window = [
                item for item in self.frames
                if item.valid_measurement
                and sample.received_at - item.received_at <= self.median_window
            ]
            positions = [
                planar_position(item.bearing, item.pitch, item.distance, self.distance_mode)
                for item in window
            ]
            target_x = statistics.median(item[0] for item in positions)
            target_y = statistics.median(item[1] for item in positions)
            target_radius = math.hypot(target_x, target_y)
            target_bearing = math.atan2(target_y, target_x)
            target_pitch = circular_mean([item.pitch for item in window])
            target_distance = statistics.median(item.distance for item in window)
            target_yaw = circular_mean([item.yaw for item in window])

            if self.filtered is None:
                filtered = FilteredPose(
                    sample.received_at, target_x, target_y, target_radius,
                    target_bearing, target_pitch, target_distance, target_yaw,
                    0.0, 0.0, 0.0,
                )
            else:
                dt = max(0.0, sample.received_at - self.filtered.updated_at)
                alpha = 1.0 - math.exp(-dt / self.filter_tau)
                x = self.filtered.x + alpha * (target_x - self.filtered.x)
                y = self.filtered.y + alpha * (target_y - self.filtered.y)
                yaw = smooth_angle(self.filtered.yaw, target_yaw, alpha)
                velocity_alpha = 1.0 - math.exp(-dt / 0.25)
                instant_vx = max(-3.0, min(3.0, (x - self.filtered.x) / max(dt, 1e-3)))
                instant_vy = max(-3.0, min(3.0, (y - self.filtered.y) / max(dt, 1e-3)))
                instant_yaw_rate = max(
                    -4.0, min(4.0, wrap_angle(yaw - self.filtered.yaw) / max(dt, 1e-3))
                )
                filtered = FilteredPose(
                    sample.received_at, x, y, math.hypot(x, y), math.atan2(y, x),
                    smooth_angle(self.filtered.pitch, target_pitch, alpha),
                    self.filtered.distance + alpha * (target_distance - self.filtered.distance),
                    yaw,
                    self.filtered.vx + velocity_alpha * (instant_vx - self.filtered.vx),
                    self.filtered.vy + velocity_alpha * (instant_vy - self.filtered.vy),
                    self.filtered.yaw_rate
                    + velocity_alpha * (instant_yaw_rate - self.filtered.yaw_rate),
                )
            self.filtered = filtered
            self.trail.append((sample.received_at, filtered.x, filtered.y))
            self._expire(sample.received_at)

    def _expire(self, now: float) -> None:
        while self.frames and now - self.frames[0].received_at > self.buffer_seconds:
            self.frames.popleft()
        while self.trail and now - self.trail[0][0] > self.buffer_seconds:
            self.trail.popleft()
        while self.recent_times and now - self.recent_times[0] > 2.0:
            self.recent_times.popleft()

    def snapshot(self) -> ReceiverSnapshot:
        with self.lock:
            self._expire(time.monotonic())
            hz = 0.0
            if len(self.recent_times) > 1:
                span = self.recent_times[-1] - self.recent_times[0]
                if span > 0.0:
                    hz = (len(self.recent_times) - 1) / span
            queue_span = 0.0
            if len(self.frames) > 1:
                queue_span = self.frames[-1].received_at - self.frames[0].received_at
            gaps = [
                self.recent_times[index] - self.recent_times[index - 1]
                for index in range(1, len(self.recent_times))
            ]
            gap_p95 = 0.0
            if gaps:
                ordered_gaps = sorted(gaps)
                gap_p95 = ordered_gaps[round(0.95 * (len(ordered_gaps) - 1))]
            valid_frames = [item for item in self.frames if item.valid_measurement]
            recent_positions = [
                planar_position(item.bearing, item.pitch, item.distance, self.distance_mode)
                for item in valid_frames[-12:]
            ]
            jitter = 0.0
            if self.filtered is not None and recent_positions:
                jitter = statistics.median(
                    math.hypot(item[0] - self.filtered.x, item[1] - self.filtered.y)
                    for item in recent_positions
                )
            raw_speed = 0.0
            if len(valid_frames) >= 2:
                previous, current = valid_frames[-2:]
                previous_xy = planar_position(
                    previous.bearing, previous.pitch, previous.distance, self.distance_mode
                )
                current_xy = planar_position(
                    current.bearing, current.pitch, current.distance, self.distance_mode
                )
                dt = current.received_at - previous.received_at
                if dt > 0.0:
                    raw_speed = math.hypot(
                        current_xy[0] - previous_xy[0], current_xy[1] - previous_xy[1]
                    ) / dt
            if self.raw is None or not self.raw.valid_measurement:
                quality = "INVALID"
            elif len(valid_frames) < 5 or queue_span < 0.5:
                quality = "WARMUP"
            elif hz < 2.0 or gap_p95 > 0.5:
                quality = "DEGRADED"
            elif jitter > 0.50:
                quality = "NOISY"
            elif jitter > 0.20:
                quality = "FAIR"
            else:
                quality = "STABLE"
            return ReceiverSnapshot(
                self.raw, self.filtered, self.received, self.accepted, hz,
                len(self.frames), queue_span,
                [(x, y) for _, x, y in self.trail],
                gap_p95, jitter, raw_speed, quality,
            )


class UwbVisualizer:
    def __init__(
        self, receiver: UwbReceiver, live_timeout: float,
        hold_timeout: float, plot_range: float, render_tau: float,
        predict_horizon: float,
    ) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.receiver = receiver
        self.live_timeout = live_timeout
        self.hold_timeout = hold_timeout
        self.plot_range = plot_range
        self.view_range = plot_range
        self.shrink_candidate_since: float | None = None
        self.render_tau = render_tau
        self.predict_horizon = predict_horizon
        self.rendered: FilteredPose | None = None
        self.last_render_at = time.monotonic()
        self.last_diagnostics_at = 0.0
        self.render_times: deque[float] = deque()

        bg, panel, grid, text = "#101316", "#181c20", "#394047", "#e7eaed"
        self.colours = {"raw": "#ff8a65", "filtered": "#4dd0e1", "yaw": "#a5d66a"}
        plt.rcParams.update({
            "figure.facecolor": bg, "axes.facecolor": panel, "axes.edgecolor": grid,
            "axes.labelcolor": text, "xtick.color": "#aeb5bc", "ytick.color": "#aeb5bc",
            "text.color": text, "font.size": 10,
        })
        self.figure = plt.figure(figsize=(14.4, 8.0), facecolor=bg)
        layout = self.figure.add_gridspec(
            2, 3, width_ratios=(1.5, 1.5, 1.28), height_ratios=(0.88, 1.12), hspace=0.25
        )
        self.map_ax = self.figure.add_subplot(layout[:, :2])
        self.compass_ax = self.figure.add_subplot(layout[0, 2])
        self.info_ax = self.figure.add_subplot(layout[1, 2])
        self.info_ax.axis("off")
        self.figure.canvas.manager.set_window_title("Go2 UWB Diagnostics")
        self.figure.suptitle("GO2 UWB | buffered planar diagnostics", fontsize=15, weight="bold")

        self.map_ax.set_title("Navigation view | robot forward is up", loc="left", pad=12)
        self.map_ax.set_xlabel("LEFT  <------------------------------>  RIGHT")
        self.map_ax.set_ylabel("BACK  <----------------------------->  FRONT")
        self.map_ax.set_aspect("equal", adjustable="box")
        self.map_ax.set_xlim(-plot_range, plot_range)
        self.map_ax.set_ylim(-plot_range, plot_range)
        self.map_ax.grid(True, color=grid, linewidth=0.7, alpha=0.55)
        self.map_ax.axhline(0.0, color="#687078", linewidth=1.0)
        self.map_ax.axvline(0.0, color="#687078", linewidth=1.0)
        (self.robot_marker,) = self.map_ax.plot(
            (0.0,), (0.0,), marker="^", markersize=19, markerfacecolor="#68737d",
            markeredgecolor="white", markeredgewidth=1.6, linestyle="none", zorder=7,
        )
        self.map_ax.annotate(
            "GO2", (0.0, 0.0), xytext=(0, -20), textcoords="offset points",
            ha="center", va="top", fontsize=8, color="#cbd1d6",
        )
        (self.raw_line,) = self.map_ax.plot([], [], "--", color=self.colours["raw"], linewidth=1.2, alpha=0.7)
        (self.raw_point,) = self.map_ax.plot([], [], "o", markerfacecolor="none", markeredgecolor=self.colours["raw"],
                                             markeredgewidth=2.0, markersize=11, label="raw")
        (self.filtered_line,) = self.map_ax.plot([], [], color=self.colours["filtered"], linewidth=2.6)
        (self.filtered_point,) = self.map_ax.plot([], [], "o", color=self.colours["filtered"], markersize=10,
                                                  label="filtered")
        (self.trail_line,) = self.map_ax.plot([], [], color=self.colours["filtered"], linewidth=1.4, alpha=0.38)
        (self.yaw_line,) = self.map_ax.plot([], [], color=self.colours["yaw"], linewidth=3.0, label="yaw_est")
        self.map_ax.legend(loc="lower left", framealpha=0.15, ncol=3, fontsize=8)
        self.position_text = self.map_ax.text(
            0.02, 0.98, "Waiting for data", transform=self.map_ax.transAxes,
            ha="left", va="top", fontsize=12, weight="bold",
        )
        self.target_text = self.map_ax.annotate(
            "", (0.0, 0.0), xytext=(10, 10), textcoords="offset points",
            ha="left", va="bottom", fontsize=8.5, color="#d9f6fa",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#101316",
                  "edgecolor": "#4dd0e1", "alpha": 0.78},
        )

        self.compass_ax.set_title("Angles in the base frame")
        self.compass_ax.set_aspect("equal", adjustable="box")
        self.compass_ax.set_xlim(-1.15, 1.15)
        self.compass_ax.set_ylim(-1.15, 1.15)
        self.compass_ax.axis("off")
        circle = [index * math.pi / 36.0 for index in range(73)]
        self.compass_ax.plot([math.cos(a) for a in circle], [math.sin(a) for a in circle], color="#687078")
        self.compass_ax.axhline(0.0, color=grid, linewidth=0.8)
        self.compass_ax.axvline(0.0, color=grid, linewidth=0.8)
        self.compass_ax.text(0, 1.08, "FRONT", ha="center", va="bottom", fontsize=8)
        self.compass_ax.text(-1.08, 0, "LEFT", ha="right", va="center", fontsize=8)
        self.compass_ax.text(1.08, 0, "RIGHT", ha="left", va="center", fontsize=8)
        (self.bearing_ray,) = self.compass_ax.plot([], [], color=self.colours["filtered"], linewidth=3.0,
                                                   label="bearing")
        (self.compass_yaw_ray,) = self.compass_ax.plot([], [], color=self.colours["yaw"], linewidth=3.0,
                                                       label="yaw_est")
        self.compass_ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, framealpha=0.15, fontsize=8)

        self.info_text = self.info_ax.text(0.0, 1.0, "Waiting for UWB frames...", ha="left", va="top",
                                           family="monospace", fontsize=8.8, linespacing=1.25)
        self.status_text = self.map_ax.text(
            0.5, -0.115, "WAITING | no UWB frames", transform=self.map_ax.transAxes,
            ha="center", va="top", clip_on=False,
            color="white", backgroundcolor="#5f676f", fontsize=10.5,
        )
        self.figure.subplots_adjust(left=0.065, right=0.975, bottom=0.12, top=0.90,
                                    wspace=0.30, hspace=0.28)

    @staticmethod
    def screen_xy(x: float, y: float) -> tuple[float, float]:
        return -y, x

    def _continuous_pose(self, target: FilteredPose, now: float) -> FilteredPose:
        if self.rendered is None:
            self.rendered = target
            self.last_render_at = now
            return target
        dt = max(0.0, now - self.last_render_at)
        alpha = 1.0 - math.exp(-dt / self.render_tau)
        x = self.rendered.x + alpha * (target.x - self.rendered.x)
        y = self.rendered.y + alpha * (target.y - self.rendered.y)
        self.rendered = FilteredPose(
            target.updated_at, x, y, math.hypot(x, y), math.atan2(y, x),
            smooth_angle(self.rendered.pitch, target.pitch, alpha),
            self.rendered.distance + alpha * (target.distance - self.rendered.distance),
            smooth_angle(self.rendered.yaw, target.yaw, alpha),
            target.vx, target.vy, target.yaw_rate,
        )
        self.last_render_at = now
        return self.rendered

    def _predicted_pose(self, pose: FilteredPose, age: float, allow_prediction: bool) -> FilteredPose:
        horizon = min(max(age, 0.0), self.predict_horizon) if allow_prediction else 0.0
        x = pose.x + pose.vx * horizon
        y = pose.y + pose.vy * horizon
        yaw = wrap_angle(pose.yaw + pose.yaw_rate * horizon)
        return FilteredPose(
            pose.updated_at, x, y, math.hypot(x, y), math.atan2(y, x),
            pose.pitch, pose.distance, yaw, pose.vx, pose.vy, pose.yaw_rate,
        )

    def _update_view_range(self, needed_range: float, now: float) -> None:
        levels = (2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 80.0)
        minimum = max(self.plot_range, needed_range)
        desired = next((level for level in levels if level >= minimum), minimum * 1.05)
        changed = False
        if desired > self.view_range:
            self.view_range = desired
            self.shrink_candidate_since = None
            changed = True
        elif desired < self.view_range:
            if self.shrink_candidate_since is None:
                self.shrink_candidate_since = now
            elif now - self.shrink_candidate_since >= 3.0:
                self.view_range = desired
                self.shrink_candidate_since = None
                changed = True
        else:
            self.shrink_candidate_since = None
        if changed:
            self.map_ax.set_xlim(-self.view_range, self.view_range)
            self.map_ax.set_ylim(-self.view_range, self.view_range)

    def update(self, _frame=None):
        data = self.receiver.snapshot()
        now = time.monotonic()
        self.render_times.append(now)
        while self.render_times and now - self.render_times[0] > 2.0:
            self.render_times.popleft()
        display_fps = 0.0
        if len(self.render_times) > 1:
            display_fps = (len(self.render_times) - 1) / (
                self.render_times[-1] - self.render_times[0]
            )
        if data.raw is None or data.filtered is None:
            self._set_status("WAITING | no valid UWB measurement", "#5f676f")
            return self.artists

        age = max(0.0, now - data.raw.received_at)
        if not data.raw.valid_measurement:
            state, colour = "INVALID", "#c62828"
        elif age <= self.live_timeout:
            state, colour = "LIVE", "#207a54"
        elif age <= self.hold_timeout:
            state, colour = "HOLD", "#b26a00"
        else:
            state, colour = "LOST", "#b3261e"
        predicted = self._predicted_pose(
            data.filtered, age, data.raw.valid_measurement,
        )
        pose = self._continuous_pose(predicted, now)
        raw_x, raw_y, raw_radius = planar_position(
            data.raw.bearing, data.raw.pitch, data.raw.distance, self.receiver.distance_mode
        )
        raw_sx, raw_sy = self.screen_xy(raw_x, raw_y)
        sx, sy = self.screen_xy(pose.x, pose.y)
        trail_extent = 0.0
        if data.trail:
            trail_extent = max(max(abs(x), abs(y)) for x, y in data.trail)
        needed_range = max(abs(raw_sx), abs(raw_sy), abs(sx), abs(sy), trail_extent) * 1.18
        self._update_view_range(needed_range, now)
        self.raw_line.set_data((0.0, raw_sx), (0.0, raw_sy))
        self.raw_point.set_data((raw_sx,), (raw_sy,))
        self.filtered_line.set_data((0.0, sx), (0.0, sy))
        self.filtered_point.set_data((sx,), (sy,))
        if data.trail:
            screen_trail = [self.screen_xy(x, y) for x, y in data.trail]
            self.trail_line.set_data([p[0] for p in screen_trail], [p[1] for p in screen_trail])
        heading_length = max(0.35, min(1.4, self.view_range * 0.09))
        hx = pose.x + heading_length * math.cos(pose.yaw)
        hy = pose.y + heading_length * math.sin(pose.yaw)
        hsx, hsy = self.screen_xy(hx, hy)
        self.yaw_line.set_data((sx, hsx), (sy, hsy))
        self.position_text.set_text(
            f"filtered: forward {pose.x:+.2f} m | left {pose.y:+.2f} m | range {pose.radius:.2f} m"
        )
        self.target_text.xy = (sx, sy)
        self.target_text.set_text(
            f"{pose.radius:.2f} m\n{math.degrees(pose.bearing):+.1f} deg"
        )

        # Compass uses screen coordinates: forward is up and left is left.
        self.bearing_ray.set_data((0.0, -math.sin(pose.bearing)), (0.0, math.cos(pose.bearing)))
        self.compass_yaw_ray.set_data((0.0, -0.82 * math.sin(pose.yaw)),
                                      (0.0, 0.82 * math.cos(pose.yaw)))
        if now - self.last_diagnostics_at >= 0.25:
            self.last_diagnostics_at = now
            raw_delta = math.hypot(raw_x - pose.x, raw_y - pose.y)
            draw_rate_text = f"{display_fps:5.1f}" if display_fps > 0.0 else "   --"
            self.info_text.set_text(
                "FILTERED PLANAR POSE\n"
                f"forward / left {pose.x:+6.2f} / {pose.y:+6.2f} m\n"
                f"range / bearing {pose.radius:6.2f} m / {math.degrees(pose.bearing):+6.1f} deg\n\n"
                "RAW UWB\n"
                f"beta / pitch   {math.degrees(data.raw.bearing):+6.1f} / "
                f"{math.degrees(data.raw.pitch):+6.1f} deg\n"
                f"distance / yaw {data.raw.distance:6.2f} m / "
                f"{math.degrees(data.raw.yaw):+6.1f} deg\n"
                f"raw-filter gap {raw_delta:6.3f} m\n\n"
                "TAG IMU (RAW DIAGNOSTIC)\n"
                f"roll / pitch   {math.degrees(data.raw.tag_roll):+6.1f} / "
                f"{math.degrees(data.raw.tag_pitch):+6.1f} deg\n"
                f"yaw            {math.degrees(data.raw.tag_yaw):+6.1f} deg\n\n"
                "BUFFER / DDS\n"
                f"quality        {data.quality:>9s}\n"
                f"queue / rx     {data.queue_size:4d} / {data.received:6d}\n"
                f"input / draw   {data.hz:5.1f} / {draw_rate_text} Hz\n"
                f"age / gap95    {age:5.3f} / {data.gap_p95:5.3f} s\n"
                f"innovation     {data.jitter:6.3f} m\n"
                f"raw speed      {data.raw_speed:6.2f} m/s\n"
                f"error/app/mode {data.raw.error_state:3d} / {data.raw.enabled_from_app:3d} / "
                f"{data.raw.joy_mode:3d}\n"
                f"range source   {self.receiver.distance_mode:>9s}\n"
                "height         disabled (pitch diagnostic only)\n"
                "antenna        KEEP UP"
            )
            if state == "LIVE" and data.quality in ("WARMUP", "NOISY", "DEGRADED"):
                colour = "#b26a00"
            self._set_status(
                f"{state}/{data.quality} | input {data.hz:4.1f} Hz | "
                f"draw {draw_rate_text.strip()} FPS | age {age:5.3f} s", colour,
            )
        return self.artists

    @property
    def artists(self):
        return (
            self.raw_line, self.raw_point, self.filtered_line, self.filtered_point,
            self.trail_line, self.yaw_line, self.position_text, self.target_text, self.bearing_ray,
            self.compass_yaw_ray, self.info_text, self.status_text,
        )

    def _set_status(self, text: str, colour: str) -> None:
        self.status_text.set_text(text)
        self.status_text.set_backgroundcolor(colour)

    def show(self, interval_ms: int, use_blit: bool) -> None:
        from matplotlib.animation import FuncAnimation
        self.animation = FuncAnimation(self.figure, self.update, interval=interval_ms,
                                       blit=use_blit and self.figure.canvas.supports_blit,
                                       cache_frame_data=False)
        self.plt.show()

    def save(self, path: Path) -> None:
        self.update()
        self.figure.savefig(str(path), dpi=140, facecolor=self.figure.get_facecolor())


def demo_sample(elapsed: float) -> UwbSample:
    bearing = 0.85 * math.sin(elapsed * 0.42)
    distance = 2.0 + 0.5 * math.sin(elapsed * 0.24)
    return UwbSample(
        time.monotonic(), bearing, math.radians(38 + 15 * math.sin(elapsed * 0.31)),
        distance, wrap_angle(bearing + 0.5), 0.04, 0.15, -0.6, 0.0, 0.0, 0.0,
        0, 1, 1, 0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buffered read-only Go2 UWB pose viewer.")
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--topic", default="rt/uwbstate")
    parser.add_argument("--range", dest="plot_range", type=float, default=3.0)
    parser.add_argument("--distance-mode", choices=("raw", "projected"), default="projected",
                        help="raw: treat distance_est as 2D range; projected: use d*cos(pitch)")
    parser.add_argument("--buffer-seconds", type=float, default=6.0)
    parser.add_argument("--median-window", type=float, default=0.45)
    parser.add_argument("--filter-tau", type=float, default=0.25)
    parser.add_argument("--render-tau", type=float, default=0.08)
    parser.add_argument("--predict-horizon", type=float, default=0.25,
                        help="display-only extrapolation limit in seconds")
    parser.add_argument("--live-timeout", type=float, default=0.5)
    parser.add_argument("--hold-timeout", type=float, default=1.5)
    parser.add_argument("--interval-ms", type=int, default=33)
    parser.add_argument("--no-blit", action="store_true",
                        help="disable partial canvas redraw for compatibility")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--snapshot-wait", type=float, default=5.0)
    args = parser.parse_args()
    positive = ("plot_range", "buffer_seconds", "median_window", "filter_tau", "render_tau",
                "predict_horizon", "live_timeout", "hold_timeout", "snapshot_wait")
    for name in positive:
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if args.hold_timeout <= args.live_timeout:
        parser.error("--hold-timeout must be greater than --live-timeout")
    if args.interval_ms < 20:
        parser.error("--interval-ms must be >= 20")
    return args


def main() -> int:
    args = parse_args()
    if args.snapshot is not None and not os.environ.get("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")
    elif args.snapshot is None and not os.environ.get("DISPLAY"):
        print("ERROR: DISPLAY is not set; use a desktop terminal or --snapshot.", file=sys.stderr)
        return 4

    receiver = UwbReceiver(args.distance_mode, args.buffer_seconds,
                           args.median_window, args.filter_tau)
    subscriber = None
    stop_event = threading.Event()
    demo_thread = None
    print("Go2 UWB buffered pose viewer")
    print("Read-only: no tracking switch, mode change, or robot command is published.")
    print(f"distance_mode={args.distance_mode} buffer={args.buffer_seconds:.1f}s "
          f"live/hold={args.live_timeout:.1f}/{args.hold_timeout:.1f}s")

    if args.demo:
        started = time.monotonic()
        def run_demo() -> None:
            while not stop_event.is_set():
                receiver.add(demo_sample(time.monotonic() - started))
                stop_event.wait(0.12)
        demo_thread = threading.Thread(target=run_demo, daemon=True)
        demo_thread.start()
    else:
        ChannelFactoryInitialize, ChannelSubscriber, UwbState_ = load_sdk()
        try:
            ChannelFactoryInitialize(0, args.network)
        except Exception as exc:
            print(f"ERROR: failed to initialize DDS on {args.network!r}: {exc}", file=sys.stderr)
            return 3
        subscriber = ChannelSubscriber(args.topic, UwbState_)
        subscriber.Init(receiver.callback, 10)

    def stop(_signum=None, _frame=None) -> None:
        stop_event.set()
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        deadline = time.monotonic() + args.snapshot_wait
        if args.snapshot is not None:
            while receiver.snapshot().filtered is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if receiver.snapshot().filtered is None:
                print("ERROR: no valid UWB frame before snapshot timeout.", file=sys.stderr)
                return 2
        viewer = UwbVisualizer(
            receiver, args.live_timeout, args.hold_timeout,
            args.plot_range, args.render_tau, args.predict_horizon,
        )
        if args.snapshot is not None:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            viewer.save(args.snapshot)
            print(f"Saved snapshot: {args.snapshot.resolve()}")
        else:
            viewer.show(args.interval_ms, not args.no_blit)
    finally:
        stop_event.set()
        if subscriber is not None:
            subscriber.Close()
        if demo_thread is not None:
            demo_thread.join(timeout=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
