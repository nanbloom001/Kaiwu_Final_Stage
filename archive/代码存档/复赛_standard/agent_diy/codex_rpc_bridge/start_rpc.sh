#!/usr/bin/env bash
set -euo pipefail

ROOT="${CODEX_RPC_ROOT:-/data/projects/legged_robot_competition_26}"
HOST="${CODEX_RPC_HOST:-0.0.0.0}"
PORT="${CODEX_RPC_PORT:-8765}"
BRIDGE_DIR="${CODEX_RPC_BRIDGE_DIR:-agent_diy/codex_rpc_bridge}"
RUNTIME_DIR="${CODEX_RPC_RUNTIME_DIR:-agent_diy/codex_rpc_bridge_runtime}"
LOG_FILE="$RUNTIME_DIR/codex_file_rpc.log"
MAX_LOG_BYTES="${CODEX_RPC_MAX_LOG_BYTES:-65536}"

cd "$ROOT"
mkdir -p "$RUNTIME_DIR"

python3 - <<'PY'
from pathlib import Path
import os
import secrets

runtime = Path(os.environ.get("CODEX_RPC_RUNTIME_DIR", "agent_diy/codex_rpc_bridge_runtime"))
runtime.mkdir(parents=True, exist_ok=True)
for name in ("token", "admin_token"):
    path = runtime / name
    if not path.exists() or not path.read_text().strip():
        path.write_text(secrets.token_urlsafe(32) + "\n")
    print(f"{name}={path.read_text().strip()}")
PY

export CODEX_RPC_TOKEN
export CODEX_RPC_ADMIN_TOKEN
CODEX_RPC_TOKEN="$(cat "$RUNTIME_DIR/token")"
CODEX_RPC_ADMIN_TOKEN="$(cat "$RUNTIME_DIR/admin_token")"

python3 - <<'PY'
import os
import signal
from pathlib import Path

for proc_dir in Path("/proc").iterdir():
    if not proc_dir.name.isdigit():
        continue
    try:
        cmd = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except Exception:
        continue
    if "codex_file_rpc.py" not in cmd:
        continue
    try:
        os.kill(int(proc_dir.name), signal.SIGTERM)
    except ProcessLookupError:
        pass
PY

sleep 1

python3 - <<'PY'
import os
from pathlib import Path

runtime = Path(os.environ.get("CODEX_RPC_RUNTIME_DIR", "agent_diy/codex_rpc_bridge_runtime"))
log_file = runtime / "codex_file_rpc.log"
max_bytes = int(os.environ.get("CODEX_RPC_MAX_LOG_BYTES", "65536"))
if log_file.exists() and log_file.stat().st_size > max_bytes:
    data = log_file.read_bytes()[-max(max_bytes // 2, 4096):]
    marker = b"--- codex_file_rpc.log rolled by start_rpc.sh ---\n"
    log_file.write_bytes(marker + data)
PY

nohup python3 "$BRIDGE_DIR/codex_file_rpc.py" \
  --host "$HOST" \
  --port "$PORT" \
  --root "$ROOT" \
  --token "$CODEX_RPC_TOKEN" \
  --admin-token "$CODEX_RPC_ADMIN_TOKEN" \
  > "$LOG_FILE" 2>&1 &

sleep 1

tail -c "$MAX_LOG_BYTES" "$LOG_FILE"
echo
curl -s "http://127.0.0.1:${PORT}/api/health" || true
echo
