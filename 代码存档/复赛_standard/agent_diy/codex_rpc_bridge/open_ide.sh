#!/usr/bin/env bash
set -euo pipefail

IDE_URL="${CODEX_TENCENT_IDE_URL:-https://tencentarena.com/p/common/competition/ide/447/11585/11428}"
SESSION="${AGENT_BROWSER_SESSION:-tencent-arena}"
AGENT_BROWSER="${AGENT_BROWSER:-agent-browser}"

export AGENT_BROWSER_SESSION="$SESSION"
export AGENT_BROWSER_SESSION_NAME="${AGENT_BROWSER_SESSION_NAME:-$SESSION}"
export AGENT_BROWSER_HEADED="${AGENT_BROWSER_HEADED:-1}"

exec "$AGENT_BROWSER" open "$IDE_URL"
