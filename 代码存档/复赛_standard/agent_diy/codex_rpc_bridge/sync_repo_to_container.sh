#!/usr/bin/env bash
set -euo pipefail

BASE="${CODEX_RPC_BASE:-https://tencentarena.com/p5/ide/11428/proxy/8765}"
SESSION="${AGENT_BROWSER_SESSION:-tencent-arena}"
AGENT_BROWSER_BIN="${AGENT_BROWSER:-agent-browser}"
SOURCE="."
DEST="."
CHUNK_SIZE=4096
APPLY=0
INCLUDE_IGNORED=0
DELETE_EXTRA=0
RUN_PY_COMPILE=0

usage() {
  cat <<'EOF'
Usage:
  bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh [options]

Safely sync the local repository to the Tencent Arena development container
through the existing RPC bridge. Defaults to dry-run; pass --apply to write.

Container persistence note:
  Only top-level agent_diy, agent_ppo, conf, and log are expected to persist
  across container restarts. Other destination directories are suitable only
  for temporary uploads or disposable tests.

Options:
  --apply                    Upload and apply changes to the container.
  --dry-run                  Show selected files and archive summary only.
  --base URL                 RPC proxy base URL. Default: CODEX_RPC_BASE or IDE 11428.
  --session NAME             agent-browser session. Default: tencent-arena.
  --agent-browser PATH       agent-browser executable. Default: agent-browser.
  --source DIR               Local source directory. Default: repository root.
  --dest DIR                 Container destination under RPC root. Default: .
  --chunk-size BYTES         Raw upload chunk size. Default: 4096, max: 4096.
  --include-ignored          Include ignored files. Not recommended.
  --delete-extra             Refuse for now; mirror-delete is intentionally disabled.
  --py-compile               Run targeted py_compile after syncing.
  -h, --help                 Show help.

Environment:
  CODEX_RPC_TOKEN            Required with --apply.
  CODEX_RPC_ADMIN_TOKEN      Required with --apply.
  CODEX_RPC_BASE             Optional default for --base.
  AGENT_BROWSER              Optional default for --agent-browser.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --dry-run)
      APPLY=0
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
    --source)
      SOURCE="${2:?missing value for --source}"
      shift 2
      ;;
    --dest)
      DEST="${2:?missing value for --dest}"
      shift 2
      ;;
    --chunk-size)
      CHUNK_SIZE="${2:?missing value for --chunk-size}"
      shift 2
      ;;
    --include-ignored)
      INCLUDE_IGNORED=1
      shift
      ;;
    --delete-extra)
      DELETE_EXTRA=1
      shift
      ;;
    --py-compile)
      RUN_PY_COMPILE=1
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

if [ "$DELETE_EXTRA" -eq 1 ]; then
  echo "error: --delete-extra is intentionally disabled in this safe sync tool" >&2
  exit 2
fi

case "$CHUNK_SIZE" in
  ''|*[!0-9]*)
    echo "error: --chunk-size must be a positive integer" >&2
    exit 2
    ;;
esac
if [ "$CHUNK_SIZE" -lt 1 ] || [ "$CHUNK_SIZE" -gt 4096 ]; then
  echo "error: --chunk-size must be between 1 and 4096 because upload chunks are sent through URL query parameters" >&2
  exit 2
fi

export AGENT_BROWSER_SESSION="$SESSION"
export AGENT_BROWSER_SESSION_NAME="$SESSION"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 127
  }
}

need_cmd python3
need_cmd git

if [ "$APPLY" -eq 1 ]; then
  : "${CODEX_RPC_TOKEN:?CODEX_RPC_TOKEN is required with --apply}"
  : "${CODEX_RPC_ADMIN_TOKEN:?CODEX_RPC_ADMIN_TOKEN is required with --apply}"
  need_cmd "$AGENT_BROWSER_BIN"
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/codex-rpc-sync.XXXXXX")"
cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

ARCHIVE="$WORKDIR/repo.tar.gz"
MANIFEST="$WORKDIR/manifest.json"
REMOTE_APPLY_PY="$WORKDIR/apply_sync.py"
SUMMARY="$WORKDIR/summary.json"

log "building safe archive from source=$SOURCE dest=$DEST"

python3 - "$SOURCE" "$DEST" "$ARCHIVE" "$MANIFEST" "$REMOTE_APPLY_PY" "$SUMMARY" "$INCLUDE_IGNORED" <<'PY'
from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

source = Path(sys.argv[1]).resolve()
dest = sys.argv[2].replace("\\", "/").rstrip("/") or "."
archive_path = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])
remote_apply_path = Path(sys.argv[5])
summary_path = Path(sys.argv[6])
include_ignored = bool(int(sys.argv[7]))

if not source.is_dir():
    raise SystemExit(f"source is not a directory: {source}")

if dest in {"workspace", "workspace/code"} or dest.startswith("workspace/code/"):
    raise SystemExit(
        "refusing unsafe relative workspace destination. "
        "Use --dest /workspace/code for the platform workspace."
    )

repo_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=source, text=True).strip()).resolve()
try:
    source.relative_to(repo_root)
except ValueError:
    raise SystemExit(f"source must be inside git repository: {source}")
if source != repo_root:
    raise SystemExit(
        f"refusing partial source sync from {source}. "
        "Run from the repository root so the sync allowlist is evaluated against the real top-level paths."
    )

protected_exact = {
    ".gitignore",
    ".git",
    ".env",
    ".vscode",
    "AGENTS.md",
    "CLAUDE.md",
    "CHANGES.md",
    "README.md",
    "tencent-arena-auth.json",
    "agent_diy",
    "agent_diy/codex_rpc_bridge_runtime",
    "codex_rpc_bridge_runtime",
    ".codex_rpc",
    "workspace",
    "hjc3-6",
    "nan10-8750",
    "nan10 8750",
    "wk",
    "yjy",
    "skills/tencent-kaiwu-training-task",
    ".arena_frontend_monitor",
    "arena_frontend_monitor_runtime",
    "log",
    "logs",
    "runs",
    "checkpoints",
    "saved_models",
    "tensorboard",
    "node_modules",
}
protected_globs = {
    "*.key",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.zip",
    "*.har",
    "*.pyc",
    "tmp_training_*.png",
}
protected_parts = {"__pycache__", "codex_rpc_bridge_runtime", ".codex_rpc"}
sync_allowlist = {
    "agent_ppo",
    "conf",
    "isaac_env",
    "kaiwu.json",
    "train_test.py",
}
bridge_root = "agent_diy/codex_rpc_bridge"
bridge_allowlist = {
    "agent_diy/codex_rpc_bridge/codex_file_rpc.py",
    "agent_diy/codex_rpc_bridge/start_rpc.sh",
}

def posix_rel(path: Path) -> str:
    return path.relative_to(source).as_posix()

def has_bad_path(rel: str) -> bool:
    p = PurePosixPath(rel)
    return p.is_absolute() or any(part in ("", ".", "..") for part in p.parts)

def is_allowed_for_sync(rel: str) -> tuple[bool, str]:
    parts = PurePosixPath(rel).parts
    if not parts:
        return False, "empty_path"
    root = parts[0]
    if root in sync_allowlist:
        return True, root
    return False, root

def is_protected(rel: str) -> tuple[bool, str]:
    parts = PurePosixPath(rel).parts
    for idx in range(1, len(parts) + 1):
        prefix = "/".join(parts[:idx])
        if prefix in protected_exact:
            return True, prefix
    if rel.startswith(f"{bridge_root}/"):
        return True, "bridge_whitelist"
    if any(part in protected_parts for part in parts):
        return True, "__pycache__"
    name = parts[-1] if parts else rel
    for pattern in protected_globs:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
            return True, pattern
    return False, ""

def git_selected_files() -> list[Path]:
    if include_ignored:
        files = []
        for root, dirs, names in os.walk(source):
            root_path = Path(root)
            if root_path == source:
                dirs[:] = [
                    d for d in dirs
                    if is_allowed_for_sync(d)[0]
                ]
            dirs[:] = [
                d for d in dirs
                if not is_protected((root_path / d).relative_to(source).as_posix())[0]
            ]
            for name in names:
                files.append(root_path / name)
        return files

    output = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z", "--", str(source.relative_to(repo_root) or ".")],
        cwd=repo_root,
    )
    files = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        path = (repo_root / raw.decode("utf-8", errors="surrogateescape")).resolve()
        try:
            path.relative_to(source)
        except ValueError:
            continue
        files.append(path)
    return files

selected = []
skipped = []
total_bytes = 0
for path in sorted(set(git_selected_files())):
    if not path.is_file() or path.is_symlink():
        skipped.append({"path": str(path), "reason": "not_regular_file_or_symlink"})
        continue
    rel = posix_rel(path)
    if has_bad_path(rel):
        skipped.append({"path": rel, "reason": "unsafe_path"})
        continue
    allowed, allow_reason = is_allowed_for_sync(rel)
    if not allowed:
        skipped.append({"path": rel, "reason": f"not_in_allowlist:{allow_reason}"})
        continue
    protected, reason = is_protected(rel)
    if protected:
        skipped.append({"path": rel, "reason": f"protected:{reason}"})
        continue
    data = path.read_bytes()
    total_bytes += len(data)
    mode = stat.S_IMODE(path.stat().st_mode)
    selected.append({
        "path": rel,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mode": mode,
    })

if not selected:
    raise SystemExit("no files selected for sync")

sync_id = time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
manifest = {
    "version": 1,
    "sync_id": sync_id,
    "source": str(source),
    "dest": dest,
    "files": selected,
    "skipped": skipped,
    "total_files": len(selected),
    "total_bytes": total_bytes,
}
manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(manifest_path, arcname="manifest.json")
    for item in selected:
        tar.add(source / item["path"], arcname=f"files/{item['path']}", recursive=False)

archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
manifest["archive_sha256"] = archive_sha
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

summary = {
    "sync_id": sync_id,
    "archive": str(archive_path),
    "archive_size": archive_path.stat().st_size,
    "archive_sha256": archive_sha,
    "manifest": str(manifest_path),
    "files": len(selected),
    "total_bytes": total_bytes,
    "skipped": len(skipped),
    "dest": dest,
    "sample_files": [item["path"] for item in selected[:20]],
    "sample_skipped": skipped[:20],
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

remote_apply_path.write_text(r'''
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path.cwd().resolve()
ARCHIVE = Path(sys.argv[1]).resolve()
EXPECTED_ARCHIVE_SHA = sys.argv[2]
DEST_RAW = sys.argv[3].replace("\\", "/").rstrip("/") or "."
SYNC_ID = sys.argv[4]
RUN_PY_COMPILE = sys.argv[5] == "1"

if DEST_RAW in {"workspace", "workspace/code"} or DEST_RAW.startswith("workspace/code/"):
    raise SystemExit(
        "refusing unsafe relative workspace destination. "
        "Use /workspace/code instead of workspace/code."
    )

RUNTIME = ROOT / "agent_diy" / "codex_rpc_bridge_runtime"
UPLOAD_DIR = RUNTIME / "uploads" / SYNC_ID
STAGING = UPLOAD_DIR / "staging"
BACKUP_DIR = RUNTIME / "backups" / "sync" / SYNC_ID
LOCK = RUNTIME / "sync.lock"
MAX_SYNC_BACKUP_DIRS = 0
MAX_RPC_BACKUP_DIRS = 0
MAX_UPLOAD_DIRS = 0

protected_exact = {
    ".gitignore",
    ".git",
    ".env",
    ".vscode",
    "AGENTS.md",
    "CLAUDE.md",
    "CHANGES.md",
    "README.md",
    "tencent-arena-auth.json",
    "agent_diy",
    "agent_diy/codex_rpc_bridge_runtime",
    "codex_rpc_bridge_runtime",
    ".codex_rpc",
    "workspace",
    "hjc3-6",
    "nan10-8750",
    "nan10 8750",
    "wk",
    "yjy",
    "skills/tencent-kaiwu-training-task",
    ".arena_frontend_monitor",
    "arena_frontend_monitor_runtime",
    "log",
    "logs",
    "runs",
    "checkpoints",
    "saved_models",
    "tensorboard",
    "node_modules",
}
bridge_root = "agent_diy/codex_rpc_bridge"
bridge_allowlist = {
    "agent_diy/codex_rpc_bridge/codex_file_rpc.py",
    "agent_diy/codex_rpc_bridge/start_rpc.sh",
}
protected_suffixes = (".key", ".pt", ".pth", ".ckpt", ".zip", ".har", ".pyc")
sync_allowlist = {
    "agent_ppo",
    "conf",
    "isaac_env",
    "kaiwu.json",
    "train_test.py",
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_rel(rel: str) -> PurePosixPath:
    p = PurePosixPath(rel)
    if p.is_absolute() or any(part in ("", ".", "..") for part in p.parts):
        raise ValueError(f"unsafe relative path: {rel}")
    return p

def is_protected(rel: str) -> bool:
    p = safe_rel(rel)
    for idx in range(1, len(p.parts) + 1):
        if "/".join(p.parts[:idx]) in protected_exact:
            return True
    if rel.startswith(f"{bridge_root}/"):
        return True
    if "__pycache__" in p.parts or "codex_rpc_bridge_runtime" in p.parts or ".codex_rpc" in p.parts:
        return True
    name = p.parts[-1]
    return name.endswith(protected_suffixes) or name.startswith("tmp_training_") and name.endswith(".png")

def is_allowed_for_sync(rel: str) -> bool:
    p = safe_rel(rel)
    return bool(p.parts) and p.parts[0] in sync_allowlist

def workspace_roots() -> list[Path]:
    roots = [ROOT.resolve()]
    workspace_code = Path("/workspace/code")
    if workspace_code.exists():
        roots.append(workspace_code.resolve())
    return roots

def under_workspace(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    for root in workspace_roots():
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            pass
    raise ValueError(
        f"{str(resolved)!r} is not in the allowed workspace roots: "
        + ", ".join(str(root) for root in workspace_roots())
    )

def prune_backup_dirs(category: str, max_keep: int) -> list[str]:
    """Keep only the newest backup directories for one backup category."""
    backups_root = RUNTIME / "backups" / category
    if max_keep < 0 or not backups_root.exists():
        return []

    candidates = [
        item for item in backups_root.iterdir()
        if item.is_dir()
    ]
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)

    removed = []
    for old in candidates[max_keep:]:
        shutil.rmtree(old, ignore_errors=True)
        removed.append(str(old))
    return removed

def prune_upload_dirs(max_keep: int = MAX_UPLOAD_DIRS) -> list[str]:
    """Remove container-side uploaded archives after apply to avoid package bloat."""
    uploads_root = RUNTIME / "uploads"
    if max_keep < 0 or not uploads_root.exists():
        return []

    candidates = [
        item for item in uploads_root.iterdir()
        if item.is_dir()
    ]
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)

    removed = []
    for old in candidates[max_keep:]:
        shutil.rmtree(old, ignore_errors=True)
        removed.append(str(old))
    return removed

if LOCK.exists():
    age = time.time() - LOCK.stat().st_mtime
    if age < 3600:
        raise SystemExit(f"sync lock exists and is fresh: {LOCK}")
LOCK.parent.mkdir(parents=True, exist_ok=True)
LOCK.write_text(json.dumps({"sync_id": SYNC_ID, "pid": os.getpid(), "time": time.time()}) + "\n")

try:
    if sha256_file(ARCHIVE) != EXPECTED_ARCHIVE_SHA:
        raise SystemExit("archive sha256 mismatch")

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        for member in tar.getmembers():
            safe_rel(member.name)
            if member.name != "manifest.json" and not member.name.startswith("files/"):
                raise SystemExit(f"unexpected tar member: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise SystemExit(f"refusing non-regular tar member: {member.name}")
        tar.extractall(STAGING)

    manifest = json.loads((STAGING / "manifest.json").read_text())
    if manifest["sync_id"] != SYNC_ID:
        raise SystemExit("manifest sync_id mismatch")
    dest_root = under_workspace(ROOT / DEST_RAW)
    dest_root.mkdir(parents=True, exist_ok=True)

    written = []
    backed_up = []
    skipped = []
    for item in manifest["files"]:
        rel = item["path"]
        rel_path = safe_rel(rel)
        if not is_allowed_for_sync(rel):
            skipped.append({"path": rel, "reason": "not_in_allowlist_on_remote"})
            continue
        if is_protected(rel):
            skipped.append({"path": rel, "reason": "protected_on_remote"})
            continue
        src = STAGING / "files" / rel
        dst = dest_root / rel_path
        if not src.is_file():
            raise SystemExit(f"missing staged file: {rel}")
        if sha256_file(src) != item["sha256"]:
            raise SystemExit(f"staged sha256 mismatch: {rel}")

        current = dest_root
        for part in rel_path.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise SystemExit(f"refusing to write through symlink directory: {rel}")

        if dst.is_symlink():
            raise SystemExit(f"refusing to overwrite symlink: {rel}")
        if dst.exists() and dst.is_file():
            current_sha = sha256_file(dst)
            if current_sha == item["sha256"]:
                skipped.append({"path": rel, "reason": "unchanged"})
                continue
            if MAX_SYNC_BACKUP_DIRS > 0:
                backup = BACKUP_DIR / rel
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, backup)
                backed_up.append(rel)
        elif dst.exists() and dst.is_dir():
            raise SystemExit(f"refusing to overwrite directory with file: {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.sync-{SYNC_ID}.tmp")
        shutil.copy2(src, tmp)
        os.chmod(tmp, int(item.get("mode", 0o644)) & 0o777)
        os.replace(tmp, dst)
        written.append(rel)

    mismatches = []
    for item in manifest["files"]:
        rel = item["path"]
        if not is_allowed_for_sync(rel):
            continue
        if is_protected(rel):
            continue
        dst = dest_root / rel
        if not dst.is_file() or sha256_file(dst) != item["sha256"]:
            mismatches.append(rel)
    if mismatches:
        raise SystemExit("post-sync verification failed: " + ", ".join(mismatches[:20]))

    py_compile = None
    if RUN_PY_COMPILE:
        import py_compile as _py_compile
        if (dest_root / "conf" / "conf.py").exists():
            targets = [
                "conf/conf.py",
                "agent.py",
                "feature/policy_observation_process.py",
                "feature/critic_observation_process.py",
                "feature/reward_process.py",
                "workflow/train_workflow.py",
            ]
        else:
            targets = [
                "agent_ppo/conf/conf.py",
                "agent_ppo/agent.py",
                "agent_ppo/feature/policy_observation_process.py",
                "agent_ppo/feature/critic_observation_process.py",
                "agent_ppo/feature/reward_process.py",
                "agent_ppo/workflow/train_workflow.py",
            ]
        ok = []
        errors = []
        for rel in targets:
            path = dest_root / rel
            if path.exists():
                try:
                    _py_compile.compile(str(path), doraise=True)
                    ok.append(rel)
                except Exception as exc:
                    errors.append({"path": rel, "error": str(exc)})
        py_compile = {"ok": ok, "errors": errors}
        if errors:
            raise SystemExit("py_compile failed: " + json.dumps(errors[:5], ensure_ascii=False))

    pruned_sync_backups = prune_backup_dirs("sync", MAX_SYNC_BACKUP_DIRS)
    pruned_rpc_backups = prune_backup_dirs("rpc", MAX_RPC_BACKUP_DIRS)
    pruned_uploads = prune_upload_dirs()

    print(json.dumps({
        "ok": True,
        "sync_id": SYNC_ID,
        "dest": str(dest_root),
        "files_total": len(manifest["files"]),
        "files_written": len(written),
        "files_backed_up": len(backed_up),
        "files_skipped": len(skipped),
        "backup_dir": str(BACKUP_DIR) if backed_up else None,
        "sync_backups_kept": MAX_SYNC_BACKUP_DIRS,
        "sync_backups_pruned": pruned_sync_backups,
        "rpc_backups_kept": MAX_RPC_BACKUP_DIRS,
        "rpc_backups_pruned": pruned_rpc_backups,
        "uploads_kept": MAX_UPLOAD_DIRS,
        "uploads_pruned": pruned_uploads,
        "py_compile": py_compile,
    }, ensure_ascii=False, indent=2))
finally:
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass
''')
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY

cat "$SUMMARY"

if [ "$APPLY" -eq 0 ]; then
  log "dry-run only; pass --apply to upload and apply this archive"
  exit 0
fi

rpc_eval() {
  local js="$1"
  printf '%s' "$js" | "$AGENT_BROWSER_BIN" eval --stdin
}

rpc_get_json() {
  local js="$1"
  local wrapped
  local raw
  wrapped="$(python3 - "$js" <<'PY'
import json
import sys
js = sys.argv[1]
print(
    "Promise.resolve("
    + js
    + ").then(value => typeof value === 'string' ? value : JSON.stringify(value))"
)
PY
)"
  raw="$("$AGENT_BROWSER_BIN" eval --json --stdin <<<"$wrapped")"
  python3 - "$raw" <<'PY'
import json
import sys
raw = sys.argv[1]
try:
    outer = json.loads(raw)
except Exception:
    print(raw, end="")
    raise SystemExit(0)
if isinstance(outer, dict) and "data" in outer and isinstance(outer["data"], dict) and "result" in outer["data"]:
    result = outer["data"]["result"]
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, ensure_ascii=False))
else:
    print(raw, end="")
PY
}

rpc_get_json_retry() {
  local js="$1"
  local attempts="${2:-4}"
  local out
  local ok
  local attempt
  for attempt in $(seq 1 "$attempts"); do
    out="$(rpc_get_json "$js")" || out='{"ok":false,"error":"agent-browser eval failed"}'
    ok="$(python3 - "$out" <<'PY'
import json
import sys
try:
    data = json.loads(sys.argv[1])
except Exception:
    print("retry")
    raise SystemExit
if data.get("ok") is False:
    text = str(data.get("text") or data.get("error") or data)
    if "WEBIDE_RECORD_NOT_FOUND" in text or "Internal Server Error" in text or "ECONNREFUSED" in text:
        print("retry")
    else:
        print("done")
else:
    print("done")
PY
)"
    if [ "$ok" = "done" ]; then
      printf '%s\n' "$out"
      return 0
    fi
    log "RPC transient failure (attempt ${attempt}/${attempts}); retrying"
    sleep "$((attempt * 2))"
  done
  printf '%s\n' "$out"
  return 0
}

fetch_json_expr() {
  local url="$1"
  local header_name="$2"
  local header_value="$3"
  python3 - "$url" "$header_name" "$header_value" <<'PY'
import json
import sys
url, header_name, header_value = sys.argv[1:]
print(
    "(() => fetch("
    + json.dumps(url)
    + ", {headers: {"
    + json.dumps(header_name)
    + ": "
    + json.dumps(header_value)
    + "}}).then(async r => { const text = await r.text(); "
    + "try { return JSON.parse(text); } "
    + "catch (e) { return {ok:false, status:r.status, content_type:r.headers.get('content-type'), text:text.slice(0, 300)}; } "
    + "}).catch(e => ({ok:false, error:String(e)})))()"
)
PY
}

b64url_file() {
  python3 - "$1" <<'PY'
import base64, pathlib, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
print(base64.urlsafe_b64encode(data).decode().rstrip("="))
PY
}

b64url_text() {
  python3 -c 'import base64,sys; print(base64.urlsafe_b64encode(sys.stdin.buffer.read()).decode().rstrip("="))'
}

SUMMARY_JSON="$(cat "$SUMMARY")"
SYNC_ID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["sync_id"])' < "$SUMMARY")"
ARCHIVE_SHA="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["archive_sha256"])' < "$SUMMARY")"
REMOTE_DIR="agent_diy/codex_rpc_bridge_runtime/uploads/${SYNC_ID}"
REMOTE_ARCHIVE="${REMOTE_DIR}/repo_archive"
REMOTE_MANIFEST="${REMOTE_DIR}/manifest_json"
REMOTE_APPLY="${REMOTE_DIR}/apply_sync_py"

log "checking RPC health at $BASE"
HEALTH_JS='(() => fetch("'"$BASE"'/api/health").then(async r => { const text = await r.text(); try { return JSON.parse(text); } catch (e) { return {ok:false, status:r.status, error:String(e), text:text.slice(0, 300)}; } }).catch(e => ({ok:false,error:String(e)})))()'
HEALTH_OUT="$(rpc_get_json "$HEALTH_JS")"
printf '%s\n' "$HEALTH_OUT"
python3 - "$HEALTH_OUT" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
if not data.get("ok"):
    raise SystemExit("RPC health failed")
root = data.get("root", "")
if root != "/data/projects/legged_robot_competition_26":
    raise SystemExit(f"unexpected RPC root: {root}")
PY

log "creating remote upload directory: $REMOTE_DIR"
MKDIR_JS="$(fetch_json_expr "$BASE/api/mkdir_get/$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "$REMOTE_DIR")" "X-Codex-Token" "$CODEX_RPC_TOKEN")"
  rpc_get_json_retry "$MKDIR_JS" >/dev/null

upload_remote_file_chunked() {
  local local_path="$1"
  local remote_path="$2"
  local remote_url_path
  local remote_sha=""
  local chunk_dir="$WORKDIR/chunks-$(basename "$remote_path" | tr -c 'A-Za-z0-9_.-' '_')"
  local chunk_count
  local idx=0
  local data
  local query
  local out

  remote_url_path="$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "$remote_path")"
  python3 - "$local_path" "$CHUNK_SIZE" "$chunk_dir" <<'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1])
chunk_size = int(sys.argv[2])
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
with src.open("rb") as f:
    idx = 0
    while True:
        data = f.read(chunk_size)
        if not data:
            break
        (out / f"{idx:06d}.chunk").write_bytes(data)
        idx += 1
print(idx)
PY
  chunk_count="$(find "$chunk_dir" -type f -name '*.chunk' | wc -l | tr -d ' ')"
  for chunk in "$chunk_dir"/*.chunk; do
    data="$(b64url_file "$chunk")"
    query="data=${data}"
    if [ -n "$remote_sha" ]; then
      query="${query}&expected_sha256=${remote_sha}"
    fi
    out="$(rpc_get_json_retry "$(fetch_json_expr "$BASE/api/append_b64/$remote_url_path?$query" "X-Codex-Token" "$CODEX_RPC_TOKEN")")"
    remote_sha="$(python3 -c 'import json,sys; data=json.load(sys.stdin); assert data.get("ok"), data; print(data["sha256"])' <<< "$out")"
    idx=$((idx + 1))
    log "uploaded $(basename "$remote_path") chunk ${idx}/${chunk_count}"
  done
  python3 - "$local_path" "$remote_sha" <<'PY'
import hashlib, pathlib, sys
local = hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()
remote = sys.argv[2]
if local != remote:
    raise SystemExit(f"uploaded sha mismatch: local={local} remote={remote}")
PY
}

log "uploading manifest and remote apply script"
upload_remote_file_chunked "$MANIFEST" "$REMOTE_MANIFEST" >/dev/null
upload_remote_file_chunked "$REMOTE_APPLY_PY" "$REMOTE_APPLY" >/dev/null

log "uploading archive in chunks: $ARCHIVE"
upload_remote_file_chunked "$ARCHIVE" "$REMOTE_ARCHIVE" >/dev/null

log "applying archive on container"
REMOTE_CMD="$(python3 - "$REMOTE_ARCHIVE" "$ARCHIVE_SHA" "$DEST" "$SYNC_ID" "$RUN_PY_COMPILE" <<'PY'
import shlex, sys
archive, sha, dest, sync_id, pyc = sys.argv[1:]
cmd = "python3 " + " ".join(shlex.quote(x) for x in [
    f"agent_diy/codex_rpc_bridge_runtime/uploads/{sync_id}/apply_sync_py",
    archive,
    sha,
    dest,
    sync_id,
    pyc,
])
print(cmd)
PY
)"
REMOTE_CMD_B64="$(printf '%s' "$REMOTE_CMD" | b64url_text)"
APPLY_JS='(() => fetch("'"$BASE"'/api/exec_b64?cwd=.&timeout=300&cmd='"$REMOTE_CMD_B64"'", {headers: {"X-Codex-Admin-Token": "'"$CODEX_RPC_ADMIN_TOKEN"'"}}).then(r => r.json()))()'
APPLY_OUT="$(rpc_get_json "$APPLY_JS")"
printf '%s\n' "$APPLY_OUT"
python3 - "$APPLY_OUT" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
if not data.get("ok") or data.get("returncode") != 0:
    raise SystemExit("remote apply failed")
stdout = data.get("stdout", "")
parsed = json.loads(stdout)
if not parsed.get("ok"):
    raise SystemExit("remote apply did not report ok")
PY

log "sync completed"
