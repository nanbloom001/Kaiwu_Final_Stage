#!/usr/bin/env python3
"""Optional atomic JSONL events for cross-process nav smoke diagnostics."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any


EVENT_LOG_ENV = "NAV_SMOKE_EVENT_LOG"
_MAX_EVENT_BYTES = 4096


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def emit_nav_event(event: str, **fields: Any) -> bool:
    """Append one compact event with a single O_APPEND write.

    The facility is deliberately best-effort and disabled unless the smoke
    launcher provides ``NAV_SMOKE_EVENT_LOG``. Diagnostic I/O must never change
    training behavior.
    """

    path = os.environ.get(EVENT_LOG_ENV, "").strip()
    if not path:
        return False
    record = _json_safe({
        "event": str(event),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "time_unix_s": time.time(),
        "monotonic_s": time.monotonic(),
        **fields,
    })
    try:
        payload = (
            json.dumps(record, ensure_ascii=True, separators=(",", ":"), default=str)
            + "\n"
        ).encode("utf-8")
        if len(payload) > _MAX_EVENT_BYTES:
            record = {
                "event": str(event),
                "pid": os.getpid(),
                "time_unix_s": time.time(),
                "event_truncated": True,
            }
            payload = (
                json.dumps(record, separators=(",", ":")) + "\n"
            ).encode("utf-8")
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return True
    except (OSError, TypeError, ValueError):
        return False
