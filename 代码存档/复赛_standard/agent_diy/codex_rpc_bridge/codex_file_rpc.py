#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = None
TOKEN = None
ADMIN_TOKEN = None
SERVER = None

MAX_READ_BYTES = 2 * 1024 * 1024
MAX_EXEC_OUTPUT_CHARS = 200000
MAX_TREE_DEPTH = 6
MAX_LOG_BYTES = 64 * 1024
LOG_TRUNCATE_CHECK_INTERVAL = 32
SKIP_DIRS = {"/proc", "/sys", "/dev", "/run"}
RUNTIME_DIR_NAME = "codex_rpc_bridge_runtime"
DEFAULT_RUNTIME_DIR = Path("agent_diy") / RUNTIME_DIR_NAME
BACKUP_ROOT = DEFAULT_RUNTIME_DIR / "backups"
RPC_BACKUP_CATEGORY = "rpc"
MAX_RPC_BACKUP_DIRS = 0
SECRET_PREFIXES = {
    f"{DEFAULT_RUNTIME_DIR.as_posix()}/token",
    f"{DEFAULT_RUNTIME_DIR.as_posix()}/admin_token",
    f"{RUNTIME_DIR_NAME}/token",
    f"{RUNTIME_DIR_NAME}/admin_token",
}
_LOG_WRITE_COUNT = 0


def b64url_decode(value):
    value = str(value or "").replace("-", "+").replace("_", "/")
    value += "=" * ((4 - len(value) % 4) % 4)
    return base64.b64decode(value)


def query_value(query, name, default=""):
    prefix = name + "="
    for part in str(query or "").split("&"):
        if part.startswith(prefix):
            return unquote(part.split("=", 1)[1])
    return default


def send_json(handler, obj, status=200):
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header(
        "Access-Control-Allow-Headers",
        "authorization, content-type, x-codex-token, x-codex-admin-token",
    )
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.end_headers()
    handler.wfile.write(data)


def short_text(value, limit=MAX_EXEC_OUTPUT_CHARS):
    value = "" if value is None else str(value)
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]..."


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_path(raw):
    raw = unquote(str(raw or ".")).replace("\\", "/")
    for prefix in ("/workspace/code/", "workspace/code/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    return "." if raw in ("", ".", "/") else raw


def path_after(prefix, request_path, default="."):
    if request_path == prefix:
        return default
    if request_path.startswith(prefix + "/"):
        return request_path[len(prefix) + 1:]
    return default


def workspace_path(raw):
    raw = normalize_path(raw)
    if raw.startswith("__abs__/"):
        raise PermissionError("absolute path requires global resolver")
    if raw.startswith("/"):
        raw = raw[1:]
    path = ROOT / raw
    Path(str(path.absolute())).relative_to(ROOT)
    return path


def global_path(raw):
    raw = normalize_path(raw)
    if raw.startswith("__abs__/"):
        return Path("/" + raw[len("__abs__/"):])
    return workspace_path(raw)


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return "__abs__" + str(path)


def is_in_workspace(path):
    try:
        Path(str(path.absolute())).relative_to(ROOT)
        return True
    except ValueError:
        return False


def resolved_in_workspace(path):
    roots = [ROOT.resolve()]
    workspace_code = Path("/workspace/code")
    if workspace_code.exists():
        roots.append(workspace_code.resolve())
    try:
        resolved = path.resolve(strict=False)
    except Exception:
        resolved = Path(str(path.absolute()))
    return any(str(resolved) == str(root) or str(resolved).startswith(str(root) + "/") for root in roots)


def is_secret_path(path):
    display = display_path(path)
    return any(display == prefix or display.startswith(prefix + "/") for prefix in SECRET_PREFIXES)


def is_skipped_dir(path):
    path_string = str(path)
    return any(path_string == item or path_string.startswith(item + "/") for item in SKIP_DIRS)


def file_info(path):
    stat = path.stat()
    return {
        "name": path.name,
        "path": display_path(path),
        "type": "dir" if path.is_dir() else "file",
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "mode": oct(stat.st_mode & 0o777),
    }


def prune_backup_dirs(category: str = RPC_BACKUP_CATEGORY, max_keep: int = MAX_RPC_BACKUP_DIRS):
    backups_root = ROOT / BACKUP_ROOT / category
    if max_keep < 0 or not backups_root.exists():
        return []

    candidates = [item for item in backups_root.iterdir() if item.is_dir()]
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)

    removed = []
    for old in candidates[max_keep:]:
        shutil.rmtree(old, ignore_errors=True)
        removed.append(old)
    return removed


def backup_file(path, category: str = RPC_BACKUP_CATEGORY, max_keep: int = MAX_RPC_BACKUP_DIRS):
    if max_keep <= 0:
        prune_backup_dirs(category=category, max_keep=max_keep)
        return None
    if not path.exists() or not path.is_file():
        return None
    safe = display_path(path).replace("__abs__/", "abs/").lstrip("/")
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("-%03d" % int((time.time() % 1) * 1000))
    backup = ROOT / BACKUP_ROOT / category / stamp / (safe + ".bak")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_bytes(path.read_bytes())
    prune_backup_dirs(category=category, max_keep=max_keep)
    return backup


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.codex-tmp-{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        try:
            f.flush()
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(path)


def check_expected_sha(path, expected):
    expected = str(expected or "").strip()
    if not expected:
        return
    actual = sha256_file(path) if path.exists() and path.is_file() else ""
    if actual != expected:
        raise ValueError(f"file changed since read: expected_sha256={expected} actual_sha256={actual}")


def trim_current_log_if_needed(fd: int = 2, max_bytes: int = MAX_LOG_BYTES):
    """Bound the redirected runtime log size with an in-place rolling overwrite.

    The bridge is typically launched as:

        nohup python3 codex_file_rpc.py > codex_file_rpc.log 2>&1 &

    so stdout/stderr both point at the same regular file.  We keep this simple:
    if the file grows beyond ``max_bytes``, keep only the newest half-window and
    continue writing at the end.  This preserves recent diagnostics while
    avoiding unbounded runtime growth inside the container, which can later
    break platform packaging.
    """
    try:
        stat = os.fstat(fd)
        if stat.st_size <= max_bytes:
            return
        keep_bytes = max(max_bytes // 2, 4096)
        os.lseek(fd, max(0, stat.st_size - keep_bytes), os.SEEK_SET)
        tail = os.read(fd, keep_bytes)
        marker = (
            f"\n--- codex_file_rpc.log rolled at {time.strftime('%Y-%m-%d %H:%M:%S')}; "
            f"kept {len(tail)} bytes ---\n"
        ).encode("utf-8")
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, marker + tail)
        os.lseek(fd, 0, os.SEEK_END)
    except Exception:
        return


class Handler(BaseHTTPRequestHandler):
    server_version = "CodexFileRPC/0.7.1"

    def log_message(self, fmt, *args):
        global _LOG_WRITE_COUNT
        _LOG_WRITE_COUNT += 1
        if _LOG_WRITE_COUNT % LOG_TRUNCATE_CHECK_INTERVAL == 0:
            trim_current_log_if_needed()
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))

    def role(self):
        query = urlparse(self.path).query
        normal = self.headers.get("X-Codex-Token") or query_value(query, "token")
        admin = self.headers.get("X-Codex-Admin-Token") or query_value(query, "admin_token")
        if ADMIN_TOKEN and admin == ADMIN_TOKEN:
            return "admin"
        if TOKEN and normal == TOKEN:
            return "normal"
        return "none"

    def require_role(self):
        role = self.role()
        if role == "none":
            send_json(self, {"ok": False, "error": "unauthorized"}, 401)
            return None
        return role

    def read_path(self, raw, role):
        path = global_path(raw)
        if role != "admin" and is_secret_path(path):
            raise PermissionError("normal token cannot read rpc secret files")
        return path

    def write_path(self, raw, role):
        path = global_path(raw)
        if role != "admin":
            if not is_in_workspace(path):
                raise PermissionError("write outside workspace requires admin token")
            if is_secret_path(path):
                raise PermissionError("normal token cannot write rpc secret files")
            if path.exists() and path.is_symlink():
                raise PermissionError("refusing to write through symlink")
            if not resolved_in_workspace(path):
                raise PermissionError("resolved write target escapes workspace")
        return path

    def do_OPTIONS(self):
        send_json(self, {"ok": True})

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            request_path = parsed.path.rstrip("/") or "/"

            if request_path in ("/health", "/api/health"):
                send_json(
                    self,
                    {
                        "ok": True,
                        "version": "0.7.1",
                        "root": str(ROOT),
                        "pid": os.getpid(),
                        "auth": bool(TOKEN),
                        "admin_auth": bool(ADMIN_TOKEN),
                    },
                )
                return

            role = self.require_role()
            if role is None:
                return

            if request_path == "/api/token-check":
                send_json(self, {"ok": True, "role": role})
                return

            if request_path == "/api/list" or request_path.startswith("/api/list/"):
                path = self.read_path(path_after("/api/list", request_path, "."), role)
                if is_skipped_dir(path):
                    send_json(self, {"ok": False, "error": "refusing to list special directory"}, 400)
                    return
                if not path.exists():
                    send_json(self, {"ok": False, "error": "not found", "path": display_path(path)}, 404)
                    return
                if path.is_file():
                    send_json(self, {"ok": True, "entry": file_info(path)})
                    return
                entries = []
                for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
                    try:
                        entries.append(file_info(child))
                    except Exception as exc:
                        entries.append({"name": child.name, "path": display_path(child), "error": str(exc)})
                send_json(self, {"ok": True, "path": display_path(path), "entries": entries})
                return

            if request_path == "/api/tree" or request_path.startswith("/api/tree/"):
                root = self.read_path(path_after("/api/tree", request_path, "."), role)
                depth = max(0, min(int(query_value(parsed.query, "depth", "3") or "3"), MAX_TREE_DEPTH))
                if is_skipped_dir(root):
                    send_json(self, {"ok": False, "error": "refusing to tree special directory"}, 400)
                    return
                entries = []

                def walk(current, level):
                    if level > depth:
                        return
                    try:
                        children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
                    except Exception as exc:
                        entries.append({"name": current.name, "path": display_path(current), "error": str(exc)})
                        return
                    for child in children:
                        try:
                            entries.append(file_info(child))
                        except Exception as exc:
                            entries.append({"name": child.name, "path": display_path(child), "error": str(exc)})
                        if child.is_dir() and not is_skipped_dir(child):
                            walk(child, level + 1)

                if not root.exists():
                    send_json(self, {"ok": False, "error": "not found", "path": display_path(root)}, 404)
                    return
                if root.is_file():
                    send_json(self, {"ok": True, "root": display_path(root), "entries": [file_info(root)]})
                    return
                walk(root, 1)
                send_json(self, {"ok": True, "root": display_path(root), "depth": depth, "entries": entries})
                return

            if request_path == "/api/stat" or request_path.startswith("/api/stat/"):
                path = self.read_path(path_after("/api/stat", request_path, ""), role)
                if not path.exists():
                    send_json(self, {"ok": False, "error": "not found", "path": display_path(path)}, 404)
                    return
                send_json(self, {"ok": True, "entry": file_info(path)})
                return

            if request_path == "/api/read" or request_path.startswith("/api/read/"):
                path = self.read_path(path_after("/api/read", request_path, ""), role)
                if not path.is_file():
                    send_json(self, {"ok": False, "error": "not a file", "path": display_path(path)}, 400)
                    return
                size = path.stat().st_size
                if size > MAX_READ_BYTES:
                    send_json(self, {"ok": False, "error": "file too large", "size": size, "max_read_bytes": MAX_READ_BYTES}, 413)
                    return
                data = path.read_bytes()
                if "encoding=base64" in parsed.query:
                    send_json(
                        self,
                        {
                            "ok": True,
                            "path": display_path(path),
                            "size": size,
                            "sha256": sha256_bytes(data),
                            "encoding": "base64",
                            "data": base64.b64encode(data).decode("ascii"),
                        },
                    )
                else:
                    send_json(
                        self,
                        {
                            "ok": True,
                            "path": display_path(path),
                            "size": size,
                            "sha256": sha256_bytes(data),
                            "encoding": "utf-8",
                            "text": data.decode("utf-8", errors="replace"),
                        },
                    )
                return

            if request_path == "/api/write_text" or request_path.startswith("/api/write_text/"):
                path = self.write_path(path_after("/api/write_text", request_path, ""), role)
                data = query_value(parsed.query, "text", "").encode("utf-8")
                check_expected_sha(path, query_value(parsed.query, "expected_sha256", ""))
                saved_backup = backup_file(path)
                atomic_write(path, data)
                send_json(self, {"ok": True, "path": display_path(path), "size": path.stat().st_size, "sha256": sha256_file(path), "backup": display_path(saved_backup) if saved_backup else None, "role": role})
                return

            if request_path == "/api/write_b64" or request_path.startswith("/api/write_b64/"):
                path = self.write_path(path_after("/api/write_b64", request_path, ""), role)
                data = b64url_decode(query_value(parsed.query, "data", ""))
                check_expected_sha(path, query_value(parsed.query, "expected_sha256", ""))
                saved_backup = backup_file(path)
                atomic_write(path, data)
                send_json(self, {"ok": True, "path": display_path(path), "size": path.stat().st_size, "sha256": sha256_file(path), "backup": display_path(saved_backup) if saved_backup else None, "role": role})
                return

            if request_path == "/api/append_b64" or request_path.startswith("/api/append_b64/"):
                path = self.write_path(path_after("/api/append_b64", request_path, ""), role)
                check_expected_sha(path, query_value(parsed.query, "expected_sha256", ""))
                current = path.read_bytes() if path.exists() and path.is_file() else b""
                saved_backup = backup_file(path)
                atomic_write(path, current + b64url_decode(query_value(parsed.query, "data", "")))
                send_json(self, {"ok": True, "path": display_path(path), "size": path.stat().st_size, "sha256": sha256_file(path), "backup": display_path(saved_backup) if saved_backup else None, "role": role})
                return

            if request_path == "/api/mkdir_get" or request_path.startswith("/api/mkdir_get/"):
                path = self.write_path(path_after("/api/mkdir_get", request_path, ""), role)
                path.mkdir(parents=True, exist_ok=True)
                send_json(self, {"ok": True, "path": display_path(path), "role": role})
                return

            if request_path == "/api/touch" or request_path.startswith("/api/touch/"):
                path = self.write_path(path_after("/api/touch", request_path, ""), role)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                send_json(self, {"ok": True, "path": display_path(path), "size": path.stat().st_size, "role": role})
                return

            if request_path == "/api/delete_get" or request_path.startswith("/api/delete_get/"):
                raw_delete = path_after("/api/delete_get", request_path, "")
                path = global_path(raw_delete)
                if role != "admin":
                    if not is_in_workspace(path):
                        raise PermissionError("delete outside workspace requires admin token")
                    if is_secret_path(path):
                        raise PermissionError("normal token cannot delete rpc secret files")
                    if not path.is_symlink() and not resolved_in_workspace(path):
                        raise PermissionError("resolved delete target escapes workspace")
                if not path.exists() and not path.is_symlink():
                    send_json(self, {"ok": False, "error": "not found", "path": display_path(path)}, 404)
                    return
                saved_backup = backup_file(path) if path.exists() and path.is_file() and not path.is_symlink() else None
                if path.is_dir() and not path.is_symlink():
                    try:
                        path.rmdir()
                    except OSError:
                        send_json(self, {"ok": False, "error": "directory is not empty; refusing recursive delete", "path": display_path(path)}, 400)
                        return
                else:
                    path.unlink()
                send_json(self, {"ok": True, "path": display_path(path), "backup": display_path(saved_backup) if saved_backup else None, "role": role})
                return

            if request_path == "/api/exec_b64":
                if role != "admin":
                    send_json(self, {"ok": False, "error": "exec requires admin token"}, 403)
                    return
                cmd = b64url_decode(query_value(parsed.query, "cmd", "")).decode("utf-8", errors="replace")
                cwd = global_path(query_value(parsed.query, "cwd", "."))
                timeout = max(1, min(int(query_value(parsed.query, "timeout", "30") or "30"), 300))
                result = subprocess.run(cmd, shell=True, cwd=str(cwd), text=True, capture_output=True, timeout=timeout, executable="/bin/bash")
                send_json(self, {"ok": True, "cwd": display_path(cwd), "cmd": cmd, "timeout": timeout, "returncode": result.returncode, "stdout": short_text(result.stdout), "stderr": short_text(result.stderr), "role": role})
                return

            send_json(self, {"ok": False, "error": "unknown endpoint", "path": request_path}, 404)

        except subprocess.TimeoutExpired as exc:
            send_json(self, {"ok": False, "error": "timeout", "stdout": short_text(exc.stdout), "stderr": short_text(exc.stderr)}, 504)
        except PermissionError as exc:
            send_json(self, {"ok": False, "error": str(exc)}, 403)
        except Exception as exc:
            send_json(self, {"ok": False, "error": str(exc)}, 500)

    def do_POST(self):
        role = self.require_role()
        if role != "admin":
            return
        if urlparse(self.path).path.rstrip("/") == "/api/stop":
            send_json(self, {"ok": True, "message": "stopping", "pid": os.getpid()})
            threading.Thread(target=lambda: (time.sleep(0.2), SERVER.shutdown()), daemon=True).start()
        else:
            send_json(self, {"ok": False, "error": "POST disabled; use GET endpoints"}, 405)


def main():
    global ROOT, TOKEN, ADMIN_TOKEN, SERVER
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", default="/data/projects/legged_robot_competition_26")
    parser.add_argument("--token", required=True)
    parser.add_argument("--admin-token", required=True)
    args = parser.parse_args()
    ROOT = Path(args.root).absolute()
    TOKEN = args.token
    ADMIN_TOKEN = args.admin_token
    SERVER = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"ok": True, "version": "0.7.1", "listening": f"http://{args.host}:{args.port}", "root": str(ROOT), "pid": os.getpid()}, ensure_ascii=False), flush=True)
    SERVER.serve_forever()


if __name__ == "__main__":
    main()
