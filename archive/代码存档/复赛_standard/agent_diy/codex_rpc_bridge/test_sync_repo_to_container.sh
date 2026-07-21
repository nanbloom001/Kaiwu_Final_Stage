#!/usr/bin/env bash
set -euo pipefail

BASE="${CODEX_RPC_BASE:-https://tencentarena.com/p5/ide/11428/proxy/8765}"
SESSION="${AGENT_BROWSER_SESSION:-tencent-arena}"
AGENT_BROWSER_BIN="${AGENT_BROWSER:-agent-browser}"
SYNC_SCRIPT="agent_diy/codex_rpc_bridge/sync_repo_to_container.sh"
MODE="local"
KEEP_REMOTE=0
RUN_EFFICIENCY=0

usage() {
  cat <<'EOF'
Usage:
  bash agent_diy/codex_rpc_bridge/test_sync_repo_to_container.sh [options]

Test sync_repo_to_container.sh safety, completeness, and RPC transfer behavior.
Default mode is --local, which does not write to the container.

Options:
  --local                 Run local dry-run and safety tests only. Default.
  --remote                Run remote RPC end-to-end tests on a temp directory.
  --all                   Run local and remote tests.
  --efficiency            Include larger transfer/chunk-size timing tests.
  --base URL              RPC proxy base URL. Default: CODEX_RPC_BASE or IDE 11428.
  --session NAME          agent-browser session. Default: tencent-arena.
  --agent-browser PATH    agent-browser executable. Default: agent-browser.
  --keep-remote           Do not delete remote temp test directory.
  -h, --help              Show help.

Environment for --remote/--all:
  CODEX_RPC_TOKEN         Normal token.
  CODEX_RPC_ADMIN_TOKEN   Admin token.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --local)
      MODE="local"
      shift
      ;;
    --remote)
      MODE="remote"
      shift
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --efficiency)
      RUN_EFFICIENCY=1
      shift
      ;;
    --base)
      BASE="${2:?missing value for --base}"
      shift 2
      ;;
    --session)
      SESSION="${2:?missing value for --session}"
      shift 2
      ;;
    --agent-browser)
      AGENT_BROWSER_BIN="${2:?missing value for --agent-browser}"
      shift 2
      ;;
    --keep-remote)
      KEEP_REMOTE=1
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

pass() {
  log "PASS: $*"
}

fail() {
  log "FAIL: $*"
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

need_cmd bash
need_cmd python3
need_cmd git

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

TEST_ROOT="$REPO_ROOT/.sync_repo_to_container_test"
REMOTE_ROOT="tmp_sync_repo_to_container_tests/$(date '+%Y%m%d-%H%M%S')"

cleanup_local() {
  rm -rf "$TEST_ROOT"
}
trap cleanup_local EXIT

json_last_object() {
  python3 -c '
import json, sys
text = sys.stdin.read()
decoder = json.JSONDecoder()
idx = 0
objects = []
while idx < len(text):
    start = text.find("{", idx)
    if start < 0:
        break
    try:
        obj, end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        idx = start + 1
        continue
    objects.append(obj)
    idx = start + end
if not objects:
    raise SystemExit("no JSON object found")
print(json.dumps(objects[-1], ensure_ascii=False, sort_keys=True))
'
}

json_get() {
  local expr="$1"
  python3 -c "import json,sys; data=json.load(sys.stdin); print($expr)"
}

make_payload() {
  local dir="$1"
  rm -rf "$dir"
  mkdir -p "$dir/dir_a" "$dir/dir_b/nested" "$dir/space name" "$dir/unicode_中文"
  printf 'alpha\n' > "$dir/dir_a/a.txt"
  printf 'bravo\n' > "$dir/dir_b/b.txt"
  printf 'charlie\n' > "$dir/dir_b/nested/c.txt"
  printf 'space\n' > "$dir/space name/file with space.txt"
  printf '中文内容\n' > "$dir/unicode_中文/测试.txt"
  python3 - "$dir/binary.bin" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).write_bytes(bytes(range(256)) * 4)
PY
}

make_protected_payload() {
  local dir="$1"
  make_payload "$dir"
  printf 'secret\n' > "$dir/.env"
  printf 'key\n' > "$dir/secret.key"
  printf 'model\n' > "$dir/model.pt"
  printf 'zip\n' > "$dir/archive.zip"
  printf 'har\n' > "$dir/capture.har"
  mkdir -p "$dir/node_modules/pkg" "$dir/__pycache__"
  printf 'pkg\n' > "$dir/node_modules/pkg/index.js"
  printf 'pyc\n' > "$dir/__pycache__/x.pyc"
}

run_sync_dry_json() {
  local source="$1"
  local dest="$2"
  shift 2
  local out
  out="$(bash "$SYNC_SCRIPT" --dry-run --source "$source" --dest "$dest" "$@" 2>&1)"
  printf '%s\n' "$out" | json_last_object
}

rpc_eval() {
  local js="$1"
  printf '%s' "$js" | "$AGENT_BROWSER_BIN" eval --stdin
}

fetch_json_expr() {
  local url="$1"
  local header_name="${2:-}"
  local header_value="${3:-}"
  python3 - "$url" "$header_name" "$header_value" <<'PY'
import json, sys
url, header_name, header_value = sys.argv[1:]
headers = "{}" if not header_name else "{" + json.dumps(header_name) + ": " + json.dumps(header_value) + "}"
print(
    "(() => fetch(" + json.dumps(url) + ", {headers: " + headers + "})"
    ".then(async r => { const text = await r.text(); try { return JSON.parse(text); } "
    "catch (e) { return {ok:false, status:r.status, content_type:r.headers.get('content-type'), text:text.slice(0,300)}; } })"
    ".catch(e => ({ok:false, error:String(e)})))()"
)
PY
}

remote_json() {
  local url="$1"
  local header_name="${2:-}"
  local header_value="${3:-}"
  rpc_eval "$(fetch_json_expr "$url" "$header_name" "$header_value")"
}

b64url_text() {
  python3 -c 'import base64,sys; print(base64.urlsafe_b64encode(sys.stdin.buffer.read()).decode().rstrip("="))'
}

remote_exec() {
  local cmd="$1"
  local cmd_b64
  cmd_b64="$(printf '%s' "$cmd" | b64url_text)"
  remote_json "$BASE/api/exec_b64?cwd=.&timeout=300&cmd=$cmd_b64" "X-Codex-Admin-Token" "$CODEX_RPC_ADMIN_TOKEN"
}

assert_json_true() {
  local json="$1"
  local expr="$2"
  python3 - "$json" "$expr" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
expr = sys.argv[2]
if not eval(expr, {}, {"data": data}):
    raise SystemExit(f"assertion failed: {expr}; data={json.dumps(data, ensure_ascii=False)[:1000]}")
PY
}

assert_apply_stdout_true() {
  local json="$1"
  local expr="$2"
  python3 - "$json" "$expr" <<'PY'
import json, sys

outer = json.loads(sys.argv[1])
expr = sys.argv[2]
stdout = outer.get("stdout") or ""
try:
    data = json.loads(stdout)
except json.JSONDecodeError as exc:
    raise SystemExit(f"remote stdout is not JSON: {exc}: {stdout[:300]!r}")
if not eval(expr, {}, {"data": data}):
    raise SystemExit(f"assertion failed: {expr}; data={json.dumps(data, ensure_ascii=False)[:1000]}")
PY
}

assert_rpc_health() {
  local json="$1"
  python3 - "$json" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
if not data.get("ok"):
    text = data.get("text") or data.get("error") or json.dumps(data, ensure_ascii=False)
    if "WEBIDE_RECORD_NOT_FOUND" in text:
        raise SystemExit("RPC health failed: WEBIDE_RECORD_NOT_FOUND; reopen/restart the Tencent Arena IDE and restart RPC")
    if "ECONNREFUSED" in text:
        raise SystemExit("RPC health failed: ECONNREFUSED; start RPC in the container with start_rpc.sh")
    raise SystemExit("RPC health failed: " + text[:500])
if data.get("root") != "/data/projects/legged_robot_competition_26":
    raise SystemExit(f"unexpected RPC root: {data.get('root')}")
PY
}

run_local_tests() {
  log "running local tests"

  bash -n "$SYNC_SCRIPT"
  pass "sync script shell syntax"

  bash "$SYNC_SCRIPT" --help | grep -q "Container persistence note"
  pass "help includes persistence note"

  local payload="$TEST_ROOT/local_payload"
  make_payload "$payload"
  local summary
  summary="$(run_sync_dry_json "$payload" "tmp_sync_repo_to_container_tests/local-basic")"
  assert_json_true "$summary" 'data["files"] == 6'
  assert_json_true "$summary" 'data["total_bytes"] > 1000'
  pass "multi-directory dry-run includes all regular files"

  local protected="$TEST_ROOT/protected_payload"
  make_protected_payload "$protected"
  summary="$(run_sync_dry_json "$protected" "tmp_sync_repo_to_container_tests/protected" --include-ignored)"
  assert_json_true "$summary" 'data["files"] == 6'
  assert_json_true "$summary" 'data["skipped"] >= 5'
  assert_json_true "$summary" 'any(item.get("reason") == "protected:.env" for item in data.get("sample_skipped", []))'
  assert_json_true "$summary" 'any(item.get("reason") == "protected:*.key" for item in data.get("sample_skipped", []))'
  pass "protected files are skipped even with --include-ignored"

  if bash "$SYNC_SCRIPT" --delete-extra --dry-run --source "$payload" --dest tmp_sync_repo_to_container_tests/delete 2>/dev/null; then
    fail "--delete-extra unexpectedly succeeded"
  fi
  pass "--delete-extra is rejected"

  if bash "$SYNC_SCRIPT" --dry-run --source "$payload" --dest tmp_sync_repo_to_container_tests/chunk --chunk-size 65536 >/dev/null 2>&1; then
    fail "oversized chunk unexpectedly succeeded"
  fi
  pass "oversized chunk-size is rejected before upload"

  if bash "$SYNC_SCRIPT" --dry-run --source "$TEST_ROOT/missing" --dest tmp_sync_repo_to_container_tests/missing >/dev/null 2>&1; then
    fail "missing source unexpectedly succeeded"
  fi
  pass "missing source is rejected"

  mkdir -p "$TEST_ROOT/empty"
  if bash "$SYNC_SCRIPT" --dry-run --source "$TEST_ROOT/empty" --dest tmp_sync_repo_to_container_tests/empty >/dev/null 2>&1; then
    fail "empty source unexpectedly succeeded"
  fi
  pass "empty source is rejected"
}

remote_preflight() {
  : "${CODEX_RPC_TOKEN:?CODEX_RPC_TOKEN is required for remote tests}"
  : "${CODEX_RPC_ADMIN_TOKEN:?CODEX_RPC_ADMIN_TOKEN is required for remote tests}"
  need_cmd "$AGENT_BROWSER_BIN"

  local health
  health="$(remote_json "$BASE/api/health")"
  assert_rpc_health "$health"
  pass "RPC health/root preflight"

  local normal admin
  normal="$(remote_json "$BASE/api/token-check" "X-Codex-Token" "$CODEX_RPC_TOKEN")"
  admin="$(remote_json "$BASE/api/token-check" "X-Codex-Admin-Token" "$CODEX_RPC_ADMIN_TOKEN")"
  assert_json_true "$normal" 'data.get("role") == "normal"'
  assert_json_true "$admin" 'data.get("role") == "admin"'
  pass "normal/admin token preflight"
}

run_sync_apply_capture() {
  local source="$1"
  local dest="$2"
  local chunk_size="$3"
  local out
  if ! out="$(
    CODEX_RPC_TOKEN="$CODEX_RPC_TOKEN" \
    CODEX_RPC_ADMIN_TOKEN="$CODEX_RPC_ADMIN_TOKEN" \
    bash "$SYNC_SCRIPT" --apply --source "$source" --dest "$dest" --chunk-size "$chunk_size" 2>&1
  )"; then
    printf '%s\n' "$out" >&2
    return 1
  fi
  printf '%s\n' "$out"
}

remote_read_text() {
  local path="$1"
  remote_json "$BASE/api/read/$path" "X-Codex-Token" "$CODEX_RPC_TOKEN"
}

run_remote_tests() {
  log "running remote tests under $REMOTE_ROOT"
  remote_preflight

  local payload="$TEST_ROOT/remote_payload"
  make_payload "$payload"

  local out apply
  out="$(run_sync_apply_capture "$payload" "$REMOTE_ROOT/basic" 4096)"
  apply="$(printf '%s\n' "$out" | json_last_object)"
  assert_json_true "$apply" 'data.get("returncode") == 0'
  assert_apply_stdout_true "$apply" 'data.get("files_written") == 6'
  pass "remote first upload writes all files"

  local read_back
  read_back="$(remote_read_text "$REMOTE_ROOT/basic/dir_a/a.txt")"
  assert_json_true "$read_back" 'data.get("text") == "alpha\n"'
  pass "remote read-back content matches"

  printf 'alpha changed\n' > "$payload/dir_a/a.txt"
  out="$(run_sync_apply_capture "$payload" "$REMOTE_ROOT/basic" 4096)"
  apply="$(printf '%s\n' "$out" | json_last_object)"
  assert_json_true "$apply" 'data.get("returncode") == 0'
  assert_apply_stdout_true "$apply" 'data.get("files_backed_up") == 1'
  pass "remote overwrite creates backup and skips unchanged files"

  read_back="$(remote_read_text "$REMOTE_ROOT/basic/dir_a/a.txt")"
  assert_json_true "$read_back" 'data.get("text") == "alpha changed\n"'
  pass "remote overwrite content matches"

  local binary_payload="$TEST_ROOT/binary_payload"
  mkdir -p "$binary_payload"
  python3 - "$binary_payload/big.bin" <<'PY'
from pathlib import Path
import hashlib
data = bytes((i * 37 + 11) % 256 for i in range(256 * 1024))
Path(__import__("sys").argv[1]).write_bytes(data)
PY
  local local_sha remote_file
  local_sha="$(python3 -c 'import hashlib,pathlib; print(hashlib.sha256(pathlib.Path("'"$binary_payload"'/big.bin").read_bytes()).hexdigest())')"
  out="$(run_sync_apply_capture "$binary_payload" "$REMOTE_ROOT/binary-4096" 4096)"
  apply="$(printf '%s\n' "$out" | json_last_object)"
  assert_json_true "$apply" 'data.get("returncode") == 0'
  remote_file="$(remote_json "$BASE/api/read/$REMOTE_ROOT/binary-4096/big.bin?encoding=base64" "X-Codex-Token" "$CODEX_RPC_TOKEN")"
  assert_json_true "$remote_file" 'data.get("sha256") == "'"$local_sha"'"'
  pass "remote binary sha256 matches with default chunk"

  if [ "$RUN_EFFICIENCY" -eq 1 ]; then
    local start end elapsed
    for chunk in 1024 2048 4096; do
      start="$(date +%s)"
      out="$(run_sync_apply_capture "$binary_payload" "$REMOTE_ROOT/binary-chunk-$chunk" "$chunk")"
      end="$(date +%s)"
      elapsed=$((end - start))
      apply="$(printf '%s\n' "$out" | json_last_object)"
      assert_json_true "$apply" 'data.get("returncode") == 0'
      remote_file="$(remote_json "$BASE/api/read/$REMOTE_ROOT/binary-chunk-$chunk/big.bin?encoding=base64" "X-Codex-Token" "$CODEX_RPC_TOKEN")"
      assert_json_true "$remote_file" 'data.get("sha256") == "'"$local_sha"'"'
      pass "chunk-size $chunk preserves binary sha256 in ${elapsed}s"
    done
  fi

  local symlink_cmd
  symlink_cmd="mkdir -p '$REMOTE_ROOT/symlink_case' && ln -sf /tmp/nowhere '$REMOTE_ROOT/symlink_case/link.txt'"
  remote_exec "$symlink_cmd" >/dev/null
  local symlink_payload="$TEST_ROOT/symlink_payload"
  mkdir -p "$symlink_payload"
  printf 'must fail\n' > "$symlink_payload/link.txt"
  if run_sync_apply_capture "$symlink_payload" "$REMOTE_ROOT/symlink_case" 4096 >/tmp/sync_symlink_test.log 2>&1; then
    fail "symlink overwrite unexpectedly succeeded"
  fi
  grep -q "refusing to overwrite symlink" /tmp/sync_symlink_test.log
  pass "remote symlink overwrite is rejected"

  local dir_conflict_payload="$TEST_ROOT/dir_conflict_payload"
  mkdir -p "$dir_conflict_payload"
  printf 'must fail\n' > "$dir_conflict_payload/conflict.txt"
  remote_exec "mkdir -p '$REMOTE_ROOT/dir_conflict/conflict.txt'" >/dev/null
  if run_sync_apply_capture "$dir_conflict_payload" "$REMOTE_ROOT/dir_conflict" 4096 >/tmp/sync_dir_conflict_test.log 2>&1; then
    fail "directory conflict unexpectedly succeeded"
  fi
  grep -q "refusing to overwrite directory" /tmp/sync_dir_conflict_test.log
  pass "remote directory/file conflict is rejected"

  remote_exec "mkdir -p agent_diy/codex_rpc_bridge_runtime && printf locked > agent_diy/codex_rpc_bridge_runtime/sync.lock" >/dev/null
  if run_sync_apply_capture "$payload" "$REMOTE_ROOT/lock_case" 4096 >/tmp/sync_lock_test.log 2>&1; then
    remote_exec "rm -f agent_diy/codex_rpc_bridge_runtime/sync.lock" >/dev/null
    fail "fresh lock unexpectedly allowed sync"
  fi
  remote_exec "rm -f agent_diy/codex_rpc_bridge_runtime/sync.lock" >/dev/null
  grep -q "sync lock exists" /tmp/sync_lock_test.log
  pass "fresh remote sync lock is rejected"

  if [ "$KEEP_REMOTE" -eq 0 ]; then
    remote_exec "rm -rf '$REMOTE_ROOT'" >/dev/null || true
    pass "remote temp directory cleaned"
  else
    log "kept remote temp directory: $REMOTE_ROOT"
  fi
}

case "$MODE" in
  local)
    run_local_tests
    ;;
  remote)
    run_remote_tests
    ;;
  all)
    run_local_tests
    run_remote_tests
    ;;
  *)
    fail "unknown mode: $MODE"
    ;;
esac

log "all requested sync tests passed"
