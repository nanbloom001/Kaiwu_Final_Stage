#!/usr/bin/env python3
"""Read-only monitor for the Go2 UWB DDS topic.

This process only creates a ChannelSubscriber. It does not enable UWB tracking,
publish robot commands, or change the controller configuration.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path


DEFAULT_SDK_PATH = Path("/home/unitree/Coding/unitree_sdk2_python-master")


def load_sdk():
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import UwbState_
    except ImportError:
        sdk_path = Path(os.environ.get("UNITREE_SDK2_PYTHON_PATH", DEFAULT_SDK_PATH))
        if not sdk_path.is_dir():
            raise RuntimeError(
                "unitree_sdk2py is not installed and the SDK source directory does not exist: "
                f"{sdk_path}"
            )
        sys.path.insert(0, str(sdk_path))
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import UwbState_

    return ChannelFactoryInitialize, ChannelSubscriber, UwbState_


class UwbMonitor:
    def __init__(self, stale_timeout: float) -> None:
        self.stale_timeout = stale_timeout
        self.lock = threading.Lock()
        self.received = 0
        self.last_received_at: float | None = None
        self.recent_times: deque[float] = deque()

    def callback(self, msg) -> None:
        now = time.monotonic()
        values = (
            msg.orientation_est,
            msg.pitch_est,
            msg.distance_est,
            msg.yaw_est,
            msg.base_yaw,
        )
        valid = (
            msg.error_state == 0
            and msg.enabled_from_app == 1
            and all(math.isfinite(value) for value in values)
            and msg.distance_est >= 0.0
        )

        with self.lock:
            self.received += 1
            self.last_received_at = now
            self.recent_times.append(now)
            while self.recent_times and now - self.recent_times[0] > 2.0:
                self.recent_times.popleft()
            hz = 0.0
            if len(self.recent_times) > 1:
                hz = (len(self.recent_times) - 1) / (
                    self.recent_times[-1] - self.recent_times[0]
                )

            status = "VALID" if valid else "INVALID"
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(
                f"{timestamp} n={self.received:6d} hz={hz:5.1f} {status:7s} "
                f"beta={msg.orientation_est:+8.4f} "
                f"pitch={msg.pitch_est:+8.4f} "
                f"distance={msg.distance_est:7.3f}m "
                f"yaw={msg.yaw_est:+8.4f} "
                f"base_yaw={msg.base_yaw:+8.4f} "
                f"error={int(msg.error_state):3d} "
                f"enabled={int(msg.enabled_from_app)} "
                f"channel={int(msg.channel):3d}",
                flush=True,
            )

    def snapshot(self) -> tuple[int, float | None]:
        with self.lock:
            return self.received, self.last_received_at


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read and print rt/uwbstate without sending robot commands."
    )
    parser.add_argument("--network", default="eth0", help="DDS network interface (default: eth0)")
    parser.add_argument("--topic", default="rt/uwbstate", help="DDS topic (default: rt/uwbstate)")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop after N seconds; 0 runs until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--stale-timeout",
        type=float,
        default=0.5,
        help="warn after N seconds without a new frame (default: 0.5)",
    )
    args = parser.parse_args()
    if args.duration < 0.0:
        parser.error("--duration must be >= 0")
    if args.stale_timeout <= 0.0:
        parser.error("--stale-timeout must be > 0")
    return args


def main() -> int:
    args = parse_args()
    ChannelFactoryInitialize, ChannelSubscriber, UwbState_ = load_sdk()

    print("Go2 UWB read-only monitor")
    print(f"network={args.network} topic={args.topic} stale_timeout={args.stale_timeout:.2f}s")
    print("This process does not publish commands or enable UWB tracking.")
    print("Press Ctrl+C to stop.\n")

    try:
        ChannelFactoryInitialize(0, args.network)
    except Exception as exc:
        print(
            f"ERROR: failed to initialize DDS on network interface {args.network!r}: {exc}",
            file=sys.stderr,
        )
        print("Check the interface name and run `ip -4 -o address show`.", file=sys.stderr)
        return 3

    monitor = UwbMonitor(args.stale_timeout)
    subscriber = ChannelSubscriber(args.topic, UwbState_)
    subscriber.Init(monitor.callback, 10)

    started_at = time.monotonic()
    last_warning_at = 0.0
    try:
        while args.duration == 0.0 or time.monotonic() - started_at < args.duration:
            time.sleep(0.1)
            now = time.monotonic()
            received, last_received_at = monitor.snapshot()
            age = now - started_at if last_received_at is None else now - last_received_at
            if age >= args.stale_timeout and now - last_warning_at >= 1.0:
                reason = "no UWB frames received" if received == 0 else f"last frame age={age:.3f}s"
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"{timestamp} STALE {reason}", flush=True)
                last_warning_at = now
    except KeyboardInterrupt:
        print("\nStopping UWB monitor...", flush=True)
    finally:
        subscriber.Close()

    received, _ = monitor.snapshot()
    print(f"Done. Received {received} UWB frames.")
    return 0 if received > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
