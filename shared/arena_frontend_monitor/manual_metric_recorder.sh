#!/usr/bin/env bash
set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$TOOL_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$TOOL_DIR/.env"
  set +a
fi

usage() {
  cat <<'EOF'
Usage:
  shared/arena_frontend_monitor/manual_metric_recorder.sh <monitor-url> [recorder args...]
  shared/arena_frontend_monitor/manual_metric_recorder.sh --clipboard [recorder args...]
  shared/arena_frontend_monitor/manual_metric_recorder.sh -c [recorder args...]

在可视化 agent-browser 新标签页中打开腾讯竞技平台监控页面，等待终端回车，
然后使用 --no-clear-network 启动 manual_metric_recorder.py。

Examples:
  shared/arena_frontend_monitor/manual_metric_recorder.sh 'https://tencentarena.com/p/v5/exp/monitor?...'
  shared/arena_frontend_monitor/manual_metric_recorder.sh -c
  shared/arena_frontend_monitor/manual_metric_recorder.sh 'https://tencentarena.com/p/v5/exp/monitor?...' --poll-interval 1

Environment:
  AGENT_BROWSER_SESSION       Default: tencent-arena
  AGENT_BROWSER_SESSION_NAME  Default: same as AGENT_BROWSER_SESSION
  AGENT_BROWSER_PROFILE       Default: runtime/browser_profile (stores login locally)
  AGENT_BROWSER_HEADED        Default: 1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

if [[ "$1" == "-c" || "$1" == "--clipboard" ]]; then
  shift
  if command -v pbpaste >/dev/null 2>&1; then
    MONITOR_URL="$(pbpaste | tr -d '\r' | sed -n '1p')"
  elif command -v powershell.exe >/dev/null 2>&1; then
    MONITOR_URL="$(powershell.exe -NoProfile -Command Get-Clipboard | tr -d '\r' | sed -n '1p')"
  elif command -v xclip >/dev/null 2>&1; then
    MONITOR_URL="$(xclip -selection clipboard -o | tr -d '\r' | sed -n '1p')"
  elif command -v wl-paste >/dev/null 2>&1; then
    MONITOR_URL="$(wl-paste | tr -d '\r' | sed -n '1p')"
  else
    echo "no clipboard reader found: expected pbpaste, powershell.exe, xclip, or wl-paste" >&2
    exit 127
  fi
else
  MONITOR_URL="$1"
  shift
fi

if [[ -z "$MONITOR_URL" ]]; then
  echo "clipboard/URL is empty" >&2
  exit 2
fi
if [[ "$MONITOR_URL" != http://* && "$MONITOR_URL" != https://* ]]; then
  echo "clipboard/URL does not look like a URL: $MONITOR_URL" >&2
  exit 2
fi

export AGENT_BROWSER_SESSION="${AGENT_BROWSER_SESSION:-tencent-arena}"
export AGENT_BROWSER_SESSION_NAME="${AGENT_BROWSER_SESSION_NAME:-$AGENT_BROWSER_SESSION}"
export AGENT_BROWSER_PROFILE="${AGENT_BROWSER_PROFILE:-$TOOL_DIR/runtime/browser_profile}"
if [[ "$AGENT_BROWSER_PROFILE" != /* ]]; then
  export AGENT_BROWSER_PROFILE="$TOOL_DIR/$AGENT_BROWSER_PROFILE"
fi
export AGENT_BROWSER_HEADED="${AGENT_BROWSER_HEADED:-1}"

if ! command -v agent-browser >/dev/null 2>&1; then
  echo "missing agent-browser command on PATH" >&2
  exit 127
fi

echo "[manual_metric_recorder] opening new headed browser tab session=${AGENT_BROWSER_SESSION}"
# Clear stale requests before opening the target page.  The recorder itself
# still runs with --no-clear-network, so requests loaded between page open and
# pressing Enter are preserved.
agent-browser network requests --clear >/dev/null || true
agent-browser tab new "$MONITOR_URL"

cat <<EOF

[manual_metric_recorder] 页面已打开。
请在可视化浏览器里手动展开/滚动/刷新需要记录的监控卡片。
准备开始记录后，回到这个终端按 Enter。

注意：本脚本默认使用 --no-clear-network，不会清空你按回车前已经加载出的曲线请求。
EOF

IFS= read -r _

python3 "$TOOL_DIR/manual_metric_recorder.py" \
  --monitor-url "$MONITOR_URL" \
  --no-clear-network \
  "$@"

LATEST_SESSION="$(
  find "${ARENA_MONITOR_RUNTIME_DIR:-$TOOL_DIR/runtime}/manual_metric_recorder/sessions" \
    -maxdepth 1 -type d -name '20*' -print 2>/dev/null | sort | tail -n 1
)"

if [[ -n "$LATEST_SESSION" && -f "$LATEST_SESSION/summary.json" ]]; then
  echo
  echo "[manual_metric_recorder] running postprocess: $LATEST_SESSION"
  if ! python3 "$TOOL_DIR/postprocess_monitor_capture.py" "$LATEST_SESSION" --no-full-series; then
    echo "[manual_metric_recorder] postprocess failed; raw recorder output is still available: $LATEST_SESSION" >&2
  fi
else
  echo "[manual_metric_recorder] postprocess skipped: latest session not found" >&2
fi
