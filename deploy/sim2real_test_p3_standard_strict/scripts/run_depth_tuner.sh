#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${DISPLAY:-}" || -z "${XAUTHORITY:-}" ]]; then
    desktop_pid="$(pgrep -u "$(id -u)" -x gnome-shell | tail -n 1 || true)"
    if [[ -n "${desktop_pid}" && -r "/proc/${desktop_pid}/environ" ]]; then
        desktop_display="$(tr '\0' '\n' < "/proc/${desktop_pid}/environ" \
            | sed -n 's/^DISPLAY=//p' | head -n 1)"
        desktop_xauthority="$(tr '\0' '\n' < "/proc/${desktop_pid}/environ" \
            | sed -n 's/^XAUTHORITY=//p' | head -n 1)"
        [[ -n "${DISPLAY:-}" ]] || export DISPLAY="${desktop_display}"
        [[ -n "${XAUTHORITY:-}" ]] || export XAUTHORITY="${desktop_xauthority}"
    fi
fi

for module in cv2 numpy psutil pyrealsense2 tkinter; do
    if ! python3 -c "import ${module}" >/dev/null 2>&1; then
        echo "ERROR: missing Python module: ${module}" >&2
        exit 1
    fi
done

exec python3 "${ROOT_DIR}/tools/realsense_depth_tuner.py" "$@"
