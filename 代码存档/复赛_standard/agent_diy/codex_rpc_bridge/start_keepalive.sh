#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || pwd)"
STATE_DIR="$REPO_ROOT/.codex_rpc"
PID_FILE="$STATE_DIR/keepalive_tencent_arena.pid"
LOG_FILE="$STATE_DIR/keepalive_tencent_arena.log"

IDE_ID="${CODEX_TENCENT_IDE_ID:-11428}"
DOMAIN_ID="${CODEX_TENCENT_DOMAIN_ID:-447}"
TEAM_ID="${CODEX_TENCENT_TEAM_ID:-11585}"
DEFAULT_IDE_URL="${CODEX_TENCENT_IDE_URL:-https://tencentarena.com/p/common/competition/ide/447/11585/11428}"
INTERVAL_SECONDS="${CODEX_KEEPALIVE_INTERVAL:-180}"
SESSION="${AGENT_BROWSER_SESSION:-tencent-arena}"
AGENT_BROWSER="${AGENT_BROWSER:-agent-browser}"
IDE_URL=""
HEALTH_URL=""
STATUS_URL="https://tencentarena.com/api/v5/Competition/GetWebIDE"
ACTION="start"
TERMINAL_KEEPALIVE=0
INTERACTIVE_RECOVERY=0
SYNTHETIC_ACTIVITY=1
SAFE_CLICK_KEEPALIVE=0
SAFE_CLICK_X=8
SAFE_CLICK_Y=8
OPEN_IDE_ON_START="${CODEX_OPEN_IDE_ON_START:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash agent_diy/codex_rpc_bridge/start_keepalive.sh [options]

Start, stop, restart, or inspect the macOS/Linux Tencent Arena keepalive loop.

Common:
  bash agent_diy/codex_rpc_bridge/start_keepalive.sh
  bash agent_diy/codex_rpc_bridge/start_keepalive.sh --status
  bash agent_diy/codex_rpc_bridge/start_keepalive.sh --stop
  bash agent_diy/codex_rpc_bridge/start_keepalive.sh --restart

Options:
  --ide-id ID             Tencent Arena IDE id. Default: 11428.
  --domain-id ID          Tencent Arena domain id. Default: 447.
  --team-id ID            Tencent Arena team id. Default: 11585.
  --ide-url URL           Full IDE entry URL. Overrides --ide-id/--domain-id/--team-id.
  --health-url URL        Full RPC health URL. Overrides --ide-id.
  --status-url URL        Tencent Arena WebIDE status URL.
  --interval SECONDS      Keepalive interval. Default: 180.
  --session NAME          agent-browser session. Default: tencent-arena.
  --agent-browser PATH    agent-browser executable. Default: agent-browser.
  --log-file PATH         Log file. Default: .codex_rpc/keepalive_tencent_arena.log.
  --pid-file PATH         Pid file. Default: .codex_rpc/keepalive_tencent_arena.pid.
  --status                Print process status and recent logs.
  --stop                  Stop the keepalive process.
  --restart               Stop, then start.
  --interactive-recovery  Allow switching/opening the IDE tab and clicking
                          reconnect/reload/restart buttons. Disabled by default.
  --no-synthetic-activity Do not dispatch focus-safe mousemove activity events.
  --safe-click-keepalive  Send one CDP mouse click to a safe page coordinate
                          each cycle. Does not move the OS mouse, but may
                          focus the browser page.
  --safe-click-x X        X coordinate for --safe-click-keepalive. Default: 8.
  --safe-click-y Y        Y coordinate for --safe-click-keepalive. Default: 8.
  --no-open-ide           Do not open the default IDE URL before starting or
                          reporting an existing keepalive process.
  --terminal-keepalive    Also type a command into the code-server terminal.
                          Disabled by default to avoid stealing input focus.
  -h, --help              Show help.

Environment:
  CODEX_TENCENT_IDE_ID        Default IDE id.
  CODEX_TENCENT_DOMAIN_ID     Default domain id.
  CODEX_TENCENT_TEAM_ID       Default team id.
  CODEX_TENCENT_IDE_URL       Default full IDE entry URL.
  CODEX_KEEPALIVE_INTERVAL    Default interval seconds.
  CODEX_OPEN_IDE_ON_START     Open default IDE URL before start/status. Default: 1.
  AGENT_BROWSER               agent-browser executable.
  AGENT_BROWSER_SESSION       agent-browser session name.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ide-id)
      IDE_ID="${2:?missing value for --ide-id}"
      shift 2
      ;;
    --domain-id)
      DOMAIN_ID="${2:?missing value for --domain-id}"
      shift 2
      ;;
    --team-id)
      TEAM_ID="${2:?missing value for --team-id}"
      shift 2
      ;;
    --ide-url)
      IDE_URL="${2:?missing value for --ide-url}"
      shift 2
      ;;
    --health-url)
      HEALTH_URL="${2:?missing value for --health-url}"
      shift 2
      ;;
    --status-url)
      STATUS_URL="${2:?missing value for --status-url}"
      shift 2
      ;;
    --interval)
      INTERVAL_SECONDS="${2:?missing value for --interval}"
      shift 2
      ;;
    --session)
      SESSION="${2:?missing value for --session}"
      shift 2
      ;;
    --agent-browser)
      AGENT_BROWSER="${2:?missing value for --agent-browser}"
      shift 2
      ;;
    --log-file)
      LOG_FILE="${2:?missing value for --log-file}"
      shift 2
      ;;
    --pid-file)
      PID_FILE="${2:?missing value for --pid-file}"
      shift 2
      ;;
    --status)
      ACTION="status"
      shift
      ;;
    --stop)
      ACTION="stop"
      shift
      ;;
    --restart)
      ACTION="restart"
      shift
      ;;
    --interactive-recovery)
      INTERACTIVE_RECOVERY=1
      shift
      ;;
    --no-synthetic-activity)
      SYNTHETIC_ACTIVITY=0
      shift
      ;;
    --safe-click-keepalive)
      SAFE_CLICK_KEEPALIVE=1
      shift
      ;;
    --safe-click-x)
      SAFE_CLICK_X="${2:?missing value for --safe-click-x}"
      shift 2
      ;;
    --safe-click-y)
      SAFE_CLICK_Y="${2:?missing value for --safe-click-y}"
      shift 2
      ;;
    --no-open-ide)
      OPEN_IDE_ON_START=0
      shift
      ;;
    --terminal-keepalive)
      TERMINAL_KEEPALIVE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$IDE_URL" ]; then
  if [ -n "${CODEX_TENCENT_IDE_URL:-}" ]; then
    IDE_URL="$DEFAULT_IDE_URL"
  else
    IDE_URL="https://tencentarena.com/p/common/competition/ide/${DOMAIN_ID}/${TEAM_ID}/${IDE_ID}"
  fi
fi
if [ -z "$HEALTH_URL" ]; then
  HEALTH_URL="https://tencentarena.com/p5/ide/${IDE_ID}/proxy/8765/api/health"
fi

mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"

is_alive() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

print_status() {
  echo "pid_file=$PID_FILE"
  echo "log_file=$LOG_FILE"
  echo "ide_url=$IDE_URL"
  echo "health_url=$HEALTH_URL"
  echo "status_url=$STATUS_URL"
  echo "interactive_recovery=$INTERACTIVE_RECOVERY"
  echo "terminal_keepalive=$TERMINAL_KEEPALIVE"
  echo "synthetic_activity=$SYNTHETIC_ACTIVITY"
  echo "safe_click_keepalive=$SAFE_CLICK_KEEPALIVE"
  echo "open_ide_on_start=$OPEN_IDE_ON_START"
  if is_alive; then
    echo "status=alive"
    echo "pid=$(cat "$PID_FILE")"
  else
    echo "status=not_alive"
    if [ -f "$PID_FILE" ]; then
      echo "last_pid=$(cat "$PID_FILE" 2>/dev/null || true)"
    fi
  fi
  if [ -f "$LOG_FILE" ]; then
    echo "--- recent log ---"
    tail -n 40 "$LOG_FILE"
  fi
}

open_default_ide() {
  [ "$OPEN_IDE_ON_START" -eq 1 ] || return 0

  command -v "$AGENT_BROWSER" >/dev/null 2>&1 || {
    echo "missing agent-browser command: $AGENT_BROWSER" >&2
    exit 127
  }

  AGENT_BROWSER_SESSION="$SESSION" \
  AGENT_BROWSER_SESSION_NAME="$SESSION" \
  AGENT_BROWSER_HEADED="${AGENT_BROWSER_HEADED:-1}" \
  "$AGENT_BROWSER" open "$IDE_URL" >/dev/null || {
    echo "warning: failed to open IDE URL: $IDE_URL" >&2
    return 1
  }
}

stop_keepalive() {
  local stopped=0
  if is_alive; then
    local pid
    pid="$(cat "$PID_FILE")"
    kill "$pid"
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      echo "warning: keepalive pid $pid is still alive after SIGTERM" >&2
    else
      echo "stopped keepalive pid $pid"
      stopped=1
    fi
  fi

  local extra_pids
  extra_pids="$(pgrep -f 'keepalive_tencent_arena\.sh' 2>/dev/null || true)"
  if [ -n "$extra_pids" ]; then
    local extra_pid
    for extra_pid in $extra_pids; do
      if [ "$extra_pid" = "$$" ]; then
        continue
      fi
      kill "$extra_pid" 2>/dev/null || true
      echo "stopped stale keepalive pid $extra_pid"
      stopped=1
    done
  fi

  if [ "$stopped" -eq 0 ]; then
    echo "keepalive is not running"
  fi
}

start_keepalive() {
  open_default_ide || true

  if is_alive; then
    echo "keepalive already running: pid=$(cat "$PID_FILE")"
    print_status
    return 0
  fi

  command -v "$AGENT_BROWSER" >/dev/null 2>&1 || {
    echo "missing agent-browser command: $AGENT_BROWSER" >&2
    exit 127
  }

  : > "$LOG_FILE"
  local args=(
    --interval "$INTERVAL_SECONDS"
    --agent-browser "$AGENT_BROWSER"
    --session "$SESSION"
    --ide-url "$IDE_URL"
    --health-url "$HEALTH_URL"
    --status-url "$STATUS_URL"
  )
  if [ "$INTERACTIVE_RECOVERY" -eq 1 ]; then
    args+=(--interactive-recovery)
  fi
  if [ "$SYNTHETIC_ACTIVITY" -eq 0 ]; then
    args+=(--no-synthetic-activity)
  fi
  if [ "$SAFE_CLICK_KEEPALIVE" -eq 1 ]; then
    args+=(--safe-click-keepalive --safe-click-x "$SAFE_CLICK_X" --safe-click-y "$SAFE_CLICK_Y")
  fi
  if [ "$TERMINAL_KEEPALIVE" -eq 1 ]; then
    args+=(--terminal-keepalive)
  fi

  AGENT_BROWSER_SESSION="$SESSION" \
  AGENT_BROWSER_SESSION_NAME="$SESSION" \
  AGENT_BROWSER="$AGENT_BROWSER" \
  nohup bash "$SCRIPT_DIR/keepalive_tencent_arena.sh" "${args[@]}" > "$LOG_FILE" 2>&1 &

  local pid="$!"
  echo "$pid" > "$PID_FILE"
  sleep 3

  if kill -0 "$pid" 2>/dev/null; then
    echo "started keepalive pid $pid"
  else
    echo "keepalive exited immediately; see log: $LOG_FILE" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
  fi
  print_status
}

case "$ACTION" in
  start)
    start_keepalive
    ;;
  status)
    open_default_ide || true
    print_status
    ;;
  stop)
    stop_keepalive
    ;;
  restart)
    stop_keepalive
    start_keepalive
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
