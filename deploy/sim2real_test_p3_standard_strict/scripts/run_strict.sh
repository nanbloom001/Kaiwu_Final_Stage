#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---preflight}"
shift || true
CTRL_ROOT="${ROOT}/runtime/unitree_rl_lab_test/deploy/robots/go2_loco"
BIN="${CTRL_ROOT}/build-strict/go2_loco_ctrl"

case "${MODE}" in
  --preflight)
    exec python3 "${ROOT}/tools/preflight.py" "$@"
    ;;
  --sensor-only|--shadow)
    [[ -x "${BIN}" ]] || { echo "ERROR: missing ${BIN}; build first." >&2; exit 3; }
    exec "${BIN}" "${MODE}" "$@"
    ;;
  --fixed-zero|--fixed-vx)
    [[ -x "${BIN}" ]] || { echo "ERROR: missing ${BIN}; build first." >&2; exit 3; }
    exec "${BIN}" "${MODE}" "$@"
    ;;
  *)
    echo "Usage: $0 --preflight [--json|--require-camera]" >&2
    echo "       $0 --sensor-only|--shadow [controller options]" >&2
    echo "       $0 --fixed-zero --arm [--suspended-test] [--record-depth] [controller options]" >&2
    echo "       $0 --fixed-zero --arm --ground-test [--record-depth] [controller options]" >&2
    echo "       $0 --fixed-vx --vx VALUE --arm [--record-depth] [controller options]  # VALUE in [0,1.00] m/s" >&2
    exit 2
    ;;
esac
