#!/usr/bin/env bash
set -u

INTERVAL_SECONDS=180
AGENT_BROWSER="${AGENT_BROWSER:-agent-browser}"
SESSION="tencent-arena"
IDE_URL="https://tencentarena.com/p/common/competition/ide/447/11585/11428"
HEALTH_URL="https://tencentarena.com/p5/ide/11428/proxy/8765/api/health"
STATUS_URL="https://tencentarena.com/api/v5/Competition/GetWebIDE"
KEEPALIVE_COMMAND='printf '"'"'[codex-keepalive] %s\n'"'"' "$(date -Iseconds)"; pwd >/dev/null'
ONCE=0
TERMINAL_KEEPALIVE=0
INTERACTIVE_RECOVERY=0
SYNTHETIC_ACTIVITY=1
SAFE_CLICK_KEEPALIVE=0
SAFE_CLICK_X=8
SAFE_CLICK_Y=8

usage() {
  cat <<'EOF'
Usage:
  bash agent_diy/codex_rpc_bridge/keepalive_tencent_arena.sh [options]

Options:
  --interval SECONDS          Keepalive interval. Default: 180
  --agent-browser PATH        agent-browser executable. Default: agent-browser
  --session NAME              agent-browser session name. Default: tencent-arena
  --ide-url URL               Tencent Arena IDE entry URL.
  --health-url URL            RPC health URL.
  --status-url URL            Tencent Arena WebIDE status URL.
                              Default: https://tencentarena.com/api/v5/Competition/GetWebIDE
  --interactive-recovery      Allow switching/opening the IDE tab and clicking
                              reconnect/reload/restart recovery buttons.
                              Disabled by default to avoid stealing keyboard focus.
  --no-synthetic-activity     Do not dispatch focus-safe mousemove activity
                              events in the current page/iframe.
  --safe-click-keepalive      Also send one CDP mouse click to a safe page
                              coordinate each cycle. This does not move the
                              OS mouse, but may focus the browser page.
  --safe-click-x X            X coordinate for --safe-click-keepalive. Default: 8.
  --safe-click-y Y            Y coordinate for --safe-click-keepalive. Default: 8.
  --terminal-keepalive        Also type a command into the code-server terminal.
                              Disabled by default to avoid stealing input focus.
  --keepalive-command CMD     Command typed into the terminal when
                              --terminal-keepalive is enabled.
  --once                      Run one keepalive cycle and exit.
  -h, --help                  Show this help.

Environment:
  AGENT_BROWSER               Alternative way to set agent-browser executable.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --interval)
      INTERVAL_SECONDS="${2:?missing value for --interval}"
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
    --keepalive-command)
      KEEPALIVE_COMMAND="${2:?missing value for --keepalive-command}"
      shift 2
      ;;
    --terminal-keepalive)
      TERMINAL_KEEPALIVE=1
      shift
      ;;
    --once)
      ONCE=1
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

export AGENT_BROWSER_SESSION="$SESSION"
export AGENT_BROWSER_SESSION_NAME="$SESSION"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

base64_one_line() {
  base64 | tr -d '\n'
}

run_agent_browser() {
  local max_attempts=4
  local attempt=1
  local output
  local status

  while [ "$attempt" -le "$max_attempts" ]; do
    output="$("$AGENT_BROWSER" "$@" 2>&1)"
    status=$?
    if [ "$status" -eq 0 ]; then
      printf '%s\n' "$output"
      return 0
    fi

    log "agent-browser failed (attempt ${attempt}/${max_attempts}): ${output}"
    if [ "$attempt" -lt "$max_attempts" ]; then
      sleep_seconds=$((attempt * 2))
      if [ "$sleep_seconds" -gt 8 ]; then
        sleep_seconds=8
      fi
      sleep "$sleep_seconds"
    fi
    attempt=$((attempt + 1))
  done

  log "agent-browser command failed: $*"
  return 1
}

select_ide_tab() {
  local tab_output
  local tab_id

  tab_output="$(run_agent_browser tab list)" || return 1
  tab_id="$(printf '%s\n' "$tab_output" | awk -v url="$IDE_URL" 'index($0, url) { if (match($0, /\[(t[0-9]+)\]/)) { print substr($0, RSTART + 1, RLENGTH - 2); exit } }')"

  if [ -n "$tab_id" ]; then
    log "switching to IDE tab: $tab_id"
    run_agent_browser tab "$tab_id" >/dev/null || return 1
    return 0
  fi

  log "IDE tab not found; opening IDE URL"
  run_agent_browser open "$IDE_URL" || return 1
  sleep 8
}

ensure_page() {
  local url_text

  select_ide_tab || return 1
  url_text="$(run_agent_browser get url | tr -d '\r' | tail -n 1)"
  log "url: $url_text"

  if [ -z "$url_text" ] || [ "$url_text" = "about:blank" ] || [ "$url_text" != "$IDE_URL" ]; then
    log "opening IDE URL"
    run_agent_browser open "$IDE_URL" || return 1
    sleep 8
  fi
}

eval_js_b64() {
  local js="$1"
  local b64
  b64="$(printf '%s' "$js" | base64_one_line)"
  run_agent_browser eval -b "$b64"
}

handle_recovery_buttons() {
  local js
  local result

  run_agent_browser snapshot -i >/dev/null || true
  js='(() => {
  const iframe = document.querySelector("iframe");
  const docs = [document];
  if (iframe && iframe.contentDocument) docs.push(iframe.contentDocument);

  const actions = [
    { key: "reconnect", label: "\u7acb\u5373\u91cd\u65b0\u8fde\u63a5", waitMs: 10000 },
    { key: "reload_window", label: "\u91cd\u65b0\u52a0\u8f7d\u7a97\u53e3", waitMs: 15000 },
    { key: "restart", label: "\u91cd\u542f", waitMs: 30000 },
  ];

  function normalize(value) {
    return String(value || "").replace(/\s+/g, "");
  }

  for (const action of actions) {
    const target = normalize(action.label);
    for (const doc of docs) {
      const nodes = Array.from(doc.querySelectorAll("button,a,[role=\"button\"],[onclick]"));
      const found = nodes.find((node) => normalize(node.innerText || node.textContent || node.getAttribute("aria-label")).includes(target));
      if (found) {
        found.click();
        return { clicked: action.key, waitMs: action.waitMs };
      }
    }
  }

  return { clicked: null };
})()'

  result="$(eval_js_b64 "$js" 2>&1)"
  log "recovery: $result"

  case "$result" in
    *'"clicked":"reconnect"'*|*'"clicked": "reconnect"'*) sleep 10 ;;
    *'"clicked":"reload_window"'*|*'"clicked": "reload_window"'*) sleep 15 ;;
    *'"clicked":"restart"'*|*'"clicked": "restart"'*) sleep 30 ;;
  esac
}

send_terminal_keepalive() {
  local command_b64
  local js
  local result

  command_b64="$(printf '%s' "$KEEPALIVE_COMMAND" | base64_one_line)"
  js='(() => {
  const commandB64 = "'"$command_b64"'";
  const text = new TextDecoder().decode(Uint8Array.from(atob(commandB64), c => c.charCodeAt(0)));
  const iframe = document.querySelector("iframe");
  if (!iframe || !iframe.contentDocument) {
    return { ok: false, reason: "iframe_not_ready" };
  }

  const doc = iframe.contentDocument;
  const terminalInput =
    doc.querySelector("textarea[aria-label*=\"终端\"]") ||
    doc.querySelector(".terminal-wrapper textarea") ||
    doc.querySelector(".xterm-helper-textarea");

  if (!terminalInput) {
    return { ok: false, reason: "terminal_input_not_found" };
  }

  terminalInput.focus();
  terminalInput.value = "";
  terminalInput.dispatchEvent(new InputEvent("input", { bubbles: true, data: "" }));

  for (const ch of text) {
    terminalInput.dispatchEvent(new KeyboardEvent("keydown", { key: ch, bubbles: true }));
    terminalInput.value += ch;
    terminalInput.dispatchEvent(new InputEvent("input", { bubbles: true, data: ch }));
    terminalInput.dispatchEvent(new KeyboardEvent("keyup", { key: ch, bubbles: true }));
  }

  terminalInput.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", code: "Enter", bubbles: true }));
  terminalInput.dispatchEvent(new KeyboardEvent("keypress", { key: "Enter", code: "Enter", bubbles: true }));
  terminalInput.dispatchEvent(new KeyboardEvent("keyup", { key: "Enter", code: "Enter", bubbles: true }));

  return { ok: true, command: text };
})()'

  result="$(eval_js_b64 "$js" 2>&1)"
  log "terminal-keepalive: $result"
  printf '%s\n' "$result"
}

handle_restart_only() {
  local js
  local result

  js='(() => {
  const iframe = document.querySelector("iframe");
  const docs = [document];
  if (iframe && iframe.contentDocument) docs.push(iframe.contentDocument);

  function normalize(value) {
    return String(value || "").replace(/\s+/g, "");
  }

  const target = normalize("\u91cd\u542f");
  for (const doc of docs) {
    const nodes = Array.from(doc.querySelectorAll("button,a,[role=\"button\"],[onclick]"));
    const found = nodes.find((node) => normalize(node.innerText || node.textContent || node.getAttribute("aria-label")).includes(target));
    if (found) {
      found.click();
      return { clicked: "restart", waitMs: 30000 };
    }
  }

  return { clicked: null };
})()'

  result="$(eval_js_b64 "$js" 2>&1)"
  log "hard-recovery: $result"

  case "$result" in
    *'"clicked":"restart"'*|*'"clicked": "restart"'*) sleep 30 ;;
  esac
}

test_health() {
  local health_url_b64
  local js
  local result

  health_url_b64="$(printf '%s' "$HEALTH_URL" | base64_one_line)"
  js='(() => {
  const healthUrl = new TextDecoder().decode(Uint8Array.from(atob("'"$health_url_b64"'"), c => c.charCodeAt(0)));
  return fetch(healthUrl)
    .then(async r => ({ status: r.status, text: await r.text() }))
    .catch(e => ({ error: String(e) }));
})()'

  result="$(eval_js_b64 "$js" 2>&1)"
  log "health: $result"
  printf '%s\n' "$result"
}

probe_background_keepalive() {
  local health_url_b64
  local status_url_b64
  local synthetic_activity
  local js
  local result

  health_url_b64="$(printf '%s' "$HEALTH_URL" | base64_one_line)"
  status_url_b64="$(printf '%s' "$STATUS_URL" | base64_one_line)"
  synthetic_activity="$SYNTHETIC_ACTIVITY"
  js='(() => {
  const decode = (value) => new TextDecoder().decode(Uint8Array.from(atob(value), c => c.charCodeAt(0)));
  const statusUrl = decode("'"$status_url_b64"'");
  const healthUrl = decode("'"$health_url_b64"'");
  const syntheticActivity = "'"$synthetic_activity"'" === "1";
  const activity = {
    enabled: syntheticActivity,
    dispatched: 0,
    errors: [],
  };
  const dispatchActivity = (targetWindow, targetDocument, label) => {
    if (!targetWindow || !targetDocument) return;
    try {
      const event = new targetWindow.MouseEvent("mousemove", {
        bubbles: true,
        cancelable: true,
        clientX: 1 + Math.floor(Math.random() * 20),
        clientY: 1 + Math.floor(Math.random() * 20),
        movementX: 1,
        movementY: 0,
      });
      targetWindow.dispatchEvent(event);
      targetDocument.dispatchEvent(event);
      activity.dispatched += 2;
    } catch (error) {
      activity.errors.push(`${label}: ${String(error)}`);
    }
  };
  if (syntheticActivity) {
    dispatchActivity(window, document, "top");
    for (const iframe of Array.from(document.querySelectorAll("iframe"))) {
      try {
        dispatchActivity(iframe.contentWindow, iframe.contentDocument, iframe.src || "iframe");
      } catch (error) {
        activity.errors.push(`iframe: ${String(error)}`);
      }
    }
  }
  const getText = async (url) => {
    try {
      const response = await fetch(url, { credentials: "include", cache: "no-store" });
      const text = await response.text();
      return { ok: true, status: response.status, url: response.url, text };
    } catch (error) {
      return { ok: false, url, error: String(error) };
    }
  };
  return Promise.all([getText(statusUrl), getText(healthUrl)]).then(([webide, health]) => ({
    location: location.href,
    visibilityState: document.visibilityState,
    hasFocus: document.hasFocus(),
    activity,
    webide,
    health,
  }));
})()'

  result="$(eval_js_b64 "$js" 2>&1)"
  log "background-probe: $result"
  printf '%s\n' "$result"
}

send_safe_click_keepalive() {
  local url_text

  url_text="$(run_agent_browser get url 2>&1 | tr -d '\r' | tail -n 1 || true)"
  if [ "$url_text" != "$IDE_URL" ]; then
    log "safe-click skipped: active url is not IDE url: $url_text"
    return 0
  fi

  run_agent_browser mouse move "$SAFE_CLICK_X" "$SAFE_CLICK_Y" >/dev/null || return 1
  run_agent_browser mouse down left >/dev/null || return 1
  run_agent_browser mouse up left >/dev/null || return 1
  log "safe-click: x=${SAFE_CLICK_X}, y=${SAFE_CLICK_Y}"
}

log "keepalive started; interval=${INTERVAL_SECONDS}s; interactive_recovery=${INTERACTIVE_RECOVERY}; terminal_keepalive=${TERMINAL_KEEPALIVE}; synthetic_activity=${SYNTHETIC_ACTIVITY}; safe_click=${SAFE_CLICK_KEEPALIVE}"

while :; do
  health_text=""

  if [ "$INTERACTIVE_RECOVERY" -eq 1 ] || [ "$TERMINAL_KEEPALIVE" -eq 1 ]; then
    if ! ensure_page; then
      log "could not ensure IDE page"
      if [ "$ONCE" -eq 1 ]; then
        log "keepalive once completed"
        break
      fi
      sleep "$INTERVAL_SECONDS"
      continue
    fi
    handle_recovery_buttons || true
    if [ "$TERMINAL_KEEPALIVE" -eq 1 ]; then
      terminal_result="$(send_terminal_keepalive || true)"
      case "$terminal_result" in
        *'"ok":false'*|*'"ok": false'*) log "terminal keepalive did not confirm terminal input path" ;;
      esac
    else
      log "terminal-keepalive: disabled"
    fi
    health_text="$(test_health || true)"

    case "$health_text" in
      *WEBIDE_RECORD_NOT_FOUND*)
        log "health indicates reclaimed IDE; attempting restart"
        if [ "$INTERACTIVE_RECOVERY" -eq 1 ]; then
          handle_restart_only || true
        else
          log "interactive recovery disabled; not clicking restart"
        fi
        ;;
    esac
  else
    health_text="$(probe_background_keepalive || true)"
    if [ "$SAFE_CLICK_KEEPALIVE" -eq 1 ]; then
      send_safe_click_keepalive || log "safe-click failed"
    fi
    case "$health_text" in
      *WEBIDE_RECORD_NOT_FOUND*)
        log "health indicates reclaimed IDE; interactive recovery disabled"
        ;;
    esac
  fi

  if [ "$ONCE" -eq 1 ]; then
    log "keepalive once completed"
    break
  fi

  sleep "$INTERVAL_SECONDS"
done
