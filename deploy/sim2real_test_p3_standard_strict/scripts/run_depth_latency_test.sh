#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/depth_latency_test.yaml"

if [[ "${1:-}" == "--config" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "ERROR: --config requires a YAML path" >&2
        exit 2
    fi
    CONFIG="$2"
    shift 2
fi

for module in numpy pyrealsense2 yaml; do
    if ! python3 -c "import ${module}" >/dev/null 2>&1; then
        echo "ERROR: missing Python module: ${module}" >&2
        exit 1
    fi
done

exec python3 "${ROOT_DIR}/tools/realsense_latency_test.py" --config "${CONFIG}" "$@"
