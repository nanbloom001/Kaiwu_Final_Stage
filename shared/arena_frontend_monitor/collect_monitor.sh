#!/usr/bin/env bash
set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AGENT_BROWSER_SESSION="${AGENT_BROWSER_SESSION:-tencent-arena}"
export AGENT_BROWSER_SESSION_NAME="${AGENT_BROWSER_SESSION_NAME:-$AGENT_BROWSER_SESSION}"

if [[ -n "${MONITOR_URL:-}" ]]; then
  set -- --monitor-url "$MONITOR_URL" "$@"
fi

python3 "$TOOL_DIR/collect_monitor_overview.py" \
  --initial-dwell-ms "${INITIAL_DWELL_MS:-300}" \
  --coverage-poll-timeout-ms "${COVERAGE_POLL_TIMEOUT_MS:-1200}" \
  --coverage-poll-interval-ms "${COVERAGE_POLL_INTERVAL_MS:-200}" \
  --max-scrollbar-drags-per-group "${MAX_SCROLLBAR_DRAGS_PER_GROUP:-20}" \
  --no-new-data-patience "${NO_NEW_DATA_PATIENCE:-2}" \
  --scroll-bottom-tolerance-px "${SCROLL_BOTTOM_TOLERANCE_PX:-16}" \
  --log-scroll-steps "${LOG_SCROLL_STEPS:-0}" \
  --log-wait-ms "${LOG_WAIT_MS:-500}" \
  "$@"
