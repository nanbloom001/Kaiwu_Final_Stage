#!/usr/bin/env bash
set -euo pipefail

AGENT_BROWSER="${AGENT_BROWSER:-agent-browser}"
SESSION="${AGENT_BROWSER_SESSION:-tencent-arena}"
IDE_ID="${CODEX_TENCENT_IDE_ID:-11428}"
DURATION_SECONDS=60
HEALTH_URL="https://tencentarena.com/p5/ide/${IDE_ID}/proxy/8765/api/health"
OUT_DIR=""
STOP_OLD=0

usage() {
  cat <<'EOF'
Usage:
  bash agent_diy/codex_rpc_bridge/validate_keepalive_candidates.sh [options]

Validate focus-safe keepalive candidates for Tencent Arena IDE.

This script is intentionally read-only from the browser UI perspective:
it does not click, type, focus the terminal, or switch tabs. It only reads
the active browser context, probes RPC health with fetch, and records network
requests that the already-open page emits naturally.

Options:
  --duration SECONDS       Passive network observation window. Default: 60.
  --agent-browser PATH     agent-browser executable. Default: agent-browser.
  --session NAME           agent-browser session. Default: tencent-arena.
  --health-url URL         RPC health URL.
  --ide-id ID              Tencent Arena IDE id. Default: 11428.
  --out-dir DIR            Output directory. Default: .codex_rpc/keepalive_probe_<timestamp>.
  --stop-old               Kill old keepalive_tencent_arena.sh processes first.
  -h, --help               Show this help.

Outputs:
  before_url.txt           Active tab URL before the probe.
  after_url.txt            Active tab URL after the probe.
  health.json              Result of background fetch(health_url).
  runtime_resources.json   Browser performance/resource hints.
  network_requests.txt     agent-browser network request log.
  candidates.txt           Filtered heartbeat/activity/websocket candidates.

Interpretation:
  - If before_url and after_url differ, the probe was not focus-neutral.
  - heartbeat/activity candidates support plan B.
  - websocket/terminal candidates support plan A.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --duration)
      DURATION_SECONDS="${2:?missing value for --duration}"
      shift 2
      ;;
    --agent-browser)
      AGENT_BROWSER="${2:?missing value for --agent-browser}"
      shift 2
      ;;
    --session)
      SESSION="${2:?missing value for --session}"
      shift 2
      ;;
    --health-url)
      HEALTH_URL="${2:?missing value for --health-url}"
      shift 2
      ;;
    --ide-id)
      IDE_ID="${2:?missing value for --ide-id}"
      HEALTH_URL="https://tencentarena.com/p5/ide/${IDE_ID}/proxy/8765/api/health"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --stop-old)
      STOP_OLD=1
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

if ! [[ "$DURATION_SECONDS" =~ ^[0-9]+$ ]] || [ "$DURATION_SECONDS" -lt 1 ]; then
  echo "duration must be a positive integer" >&2
  exit 2
fi

if [ -z "$OUT_DIR" ]; then
  REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  OUT_DIR="$REPO_ROOT/.codex_rpc/keepalive_probe_$(date '+%Y%m%d_%H%M%S')"
fi
mkdir -p "$OUT_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

base64_one_line() {
  base64 | tr -d '\n'
}

run_browser() {
  AGENT_BROWSER_SESSION="$SESSION" \
  AGENT_BROWSER_SESSION_NAME="$SESSION" \
  "$AGENT_BROWSER" "$@"
}

stop_old_keepalive() {
  local pids=""
  pids="$(pgrep -f 'keepalive_tencent_arena\.sh' 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    log "no old keepalive_tencent_arena.sh process found"
    return 0
  fi

  local pid
  for pid in $pids; do
    if [ "$pid" = "$$" ]; then
      continue
    fi
    log "killing old keepalive process pid=$pid"
    kill "$pid" 2>/dev/null || true
  done
}

eval_js_b64() {
  local js="$1"
  local b64
  b64="$(printf '%s' "$js" | base64_one_line)"
  run_browser eval -b "$b64"
}

if [ "$STOP_OLD" -eq 1 ]; then
  stop_old_keepalive
fi

command -v "$AGENT_BROWSER" >/dev/null 2>&1 || {
  echo "missing agent-browser command: $AGENT_BROWSER" >&2
  exit 127
}

log "output directory: $OUT_DIR"
log "session: $SESSION"
log "health url: $HEALTH_URL"
log "reading active URL before probe"
run_browser get url > "$OUT_DIR/before_url.txt"

log "clearing agent-browser network log"
run_browser network requests --clear > "$OUT_DIR/network_clear.txt" 2>&1 || true

health_url_b64="$(printf '%s' "$HEALTH_URL" | base64_one_line)"
health_js='(() => {
  const healthUrl = new TextDecoder().decode(Uint8Array.from(atob("'"$health_url_b64"'"), c => c.charCodeAt(0)));
  return fetch(healthUrl, { credentials: "include", cache: "no-store" })
    .then(async (response) => ({
      ok: true,
      status: response.status,
      url: response.url,
      text: await response.text(),
    }))
    .catch((error) => ({ ok: false, error: String(error) }));
})()'

log "probing RPC health through current browser context"
eval_js_b64 "$health_js" > "$OUT_DIR/health.json" 2>&1 || true

runtime_js='(() => {
  const interesting = /(heart|keep|alive|activity|active|ping|pong|session|ide|websocket|socket|terminal|proxy|health|reconnect|code-server|vscode|workbench)/i;
  const resources = performance.getEntriesByType("resource")
    .map((entry) => ({
      name: entry.name,
      initiatorType: entry.initiatorType,
      duration: Math.round(entry.duration),
      startTime: Math.round(entry.startTime),
    }))
    .filter((entry) => interesting.test(entry.name) || interesting.test(entry.initiatorType || ""))
    .slice(-300);

  const scripts = Array.from(document.scripts || [])
    .map((script) => script.src)
    .filter((src) => src && interesting.test(src))
    .slice(-100);

  const iframeUrls = Array.from(document.querySelectorAll("iframe"))
    .map((iframe) => iframe.src || iframe.getAttribute("src") || "")
    .filter(Boolean);

  return {
    location: location.href,
    visibilityState: document.visibilityState,
    hasFocus: document.hasFocus(),
    iframeUrls,
    scripts,
    resources,
  };
})()'

log "collecting runtime resource hints"
eval_js_b64 "$runtime_js" > "$OUT_DIR/runtime_resources.json" 2>&1 || true

log "passively observing network for ${DURATION_SECONDS}s; no click/focus/type/tab switch"
sleep "$DURATION_SECONDS"

log "dumping network requests"
run_browser network requests > "$OUT_DIR/network_requests.txt" 2>&1 || true

log "reading active URL after probe"
run_browser get url > "$OUT_DIR/after_url.txt" 2>&1 || true

grep -Ei 'heart|keep|alive|activity|active|ping|pong|session|ide|websocket|socket|terminal|proxy|health|reconnect|code-server|vscode|workbench|/api/' \
  "$OUT_DIR/network_requests.txt" \
  "$OUT_DIR/runtime_resources.json" \
  "$OUT_DIR/health.json" \
  > "$OUT_DIR/candidates.txt" 2>/dev/null || true

before_url="$(tr -d '\r\n' < "$OUT_DIR/before_url.txt" 2>/dev/null || true)"
after_url="$(tr -d '\r\n' < "$OUT_DIR/after_url.txt" 2>/dev/null || true)"

echo "out_dir=$OUT_DIR"
echo "before_url=$before_url"
echo "after_url=$after_url"
if [ "$before_url" = "$after_url" ]; then
  echo "focus_neutral=likely"
else
  echo "focus_neutral=failed_url_changed"
fi

echo "--- health ---"
sed -n '1,40p' "$OUT_DIR/health.json" || true

echo "--- candidates ---"
sed -n '1,120p' "$OUT_DIR/candidates.txt" || true
