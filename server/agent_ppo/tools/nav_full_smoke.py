#!/usr/bin/env python3
"""Start, inspect and stop a bounded full-stack nav training smoke.

Run this inside the Kaiwu development container from the ``server`` root. The
launcher intentionally does not set ``KAIWU_TRAIN_TEST``: it exercises preload,
the real nav workflow, rollout, the first TBPTT update and checkpoint lifecycle.
Production TOML files are never edited.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


DEFAULT_RUNTIME_DIR = Path("/tmp/kaiwu_nav_full_smoke")
SMOKE_DUMP_MODEL_FREQ = 160


def _paths(runtime_dir: Path) -> dict[str, Path]:
    return {
        "pid": runtime_dir / "launcher.pid",
        "status": runtime_dir / "status.json",
        "events": runtime_dir / "events.jsonl",
        "output": runtime_dir / "train.log",
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _proc_command(pid: int | None) -> tuple[str, ...]:
    if not _alive(pid):
        return ()
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        # The launcher is intended for the Linux Kaiwu container. Refuse
        # ownership-sensitive operations when procfs cannot prove identity.
        return ()
    try:
        return tuple(
            part.decode("utf-8", errors="replace")
            for part in cmdline_path.read_bytes().split(b"\x00")
            if part
        )
    except OSError:
        return ()


def _owned_smoke_process(pid: int | None, runtime_dir: Path | None = None) -> bool:
    command = _proc_command(pid)
    if not command:
        return False
    joined = " ".join(command)
    if "nav_full_smoke.py" not in joined or "_run" not in command:
        return False
    if runtime_dir is None:
        return True
    expected = str(runtime_dir.resolve())
    return any(
        command[index] == "--runtime-dir" and command[index + 1] == expected
        for index in range(len(command) - 1)
    )


def _owned_smoke_pids(runtime_dir: Path) -> list[int]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []
    result = []
    for entry in proc_root.iterdir():
        if entry.name.isdigit():
            pid = int(entry.name)
            if _owned_smoke_process(pid, runtime_dir):
                result.append(pid)
    return result


def _events(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _target_reached(rows: list[dict]) -> bool:
    update_completed = any(
        row.get("event") == "iteration"
        and float(row.get("valid_ticks") or 0.0) > 0.0
        and not float(row.get("update_skipped_no_valid") or 0.0)
        for row in rows
    )
    dump_boundary_reached = any(
        row.get("event") == "platform_dump_boundary" for row in rows
    )
    return update_completed and dump_boundary_reached


def _watch_for_smoke_target(event_path: Path, process_group: int) -> None:
    while True:
        if _target_reached(_events(event_path)):
            try:
                os.killpg(process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
        time.sleep(0.25)


def _run(runtime_dir: Path, num_envs: int, stop_after_first_update: bool) -> int:
    paths = _paths(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    paths["pid"].write_text(f"{os.getpid()}\n", encoding="utf-8")
    _write_json(
        paths["status"],
        {"state": "running", "pid": os.getpid(), "num_envs": num_envs},
    )
    os.environ.pop("KAIWU_TRAIN_TEST", None)
    os.environ["NAV_FULL_SMOKE"] = "1"
    os.environ["NAV_FULL_SMOKE_NUM_ENVS"] = str(num_envs)
    os.environ["NAV_SMOKE_EVENT_LOG"] = str(paths["events"])

    if stop_after_first_update:
        watcher = threading.Thread(
            target=_watch_for_smoke_target,
            args=(paths["events"], os.getpgrp()),
            name="nav-smoke-first-update-watcher",
            daemon=True,
        )
        watcher.start()

    from kaiwudrl.common.utils.train_test_utils import run_train_test

    try:
        run_train_test(
            algorithm_name="ppo",
            algorithm_name_list=["ppo", "diy"],
            env_vars={
                "replay_buffer_capacity": "10",
                "preload_ratio": "1",
                "train_batch_size": "2",
                # One automatic dump after the first complete 160-frame TBPTT
                # iteration. Per-frame lifecycle callbacks would make a value of
                # 1 write 160 redundant checkpoints before the smoke can stop.
                "dump_model_freq": str(SMOKE_DUMP_MODEL_FREQ),
                "max_frame_no": "100000000",
                "NAV_FULL_SMOKE": "1",
                "NAV_FULL_SMOKE_NUM_ENVS": str(num_envs),
                "NAV_SMOKE_EVENT_LOG": str(paths["events"]),
            },
            shell="bash",
            skip_aisrv_alive_check=True,
            skip_error_scan=True,
            check_train_success_flag=False,
            check_model_method="glob_stage_pkl",
        )
    except BaseException as exc:
        _write_json(
            paths["status"],
            {
                "state": "stopped",
                "pid": os.getpid(),
                "target_reached": _target_reached(_events(paths["events"])),
                "reason": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    _write_json(
        paths["status"],
        {
            "state": "completed",
            "pid": os.getpid(),
            "target_reached": _target_reached(_events(paths["events"])),
        },
    )
    return 0


def _start(args) -> int:
    runtime_dir = Path(args.runtime_dir).resolve()
    paths = _paths(runtime_dir)
    launch_cwd = Path.cwd()
    if not (launch_cwd / "train_test.py").is_file():
        print(
            "nav full smoke must be started from the server project root "
            "containing train_test.py",
            file=sys.stderr,
        )
        return 2
    existing = _read_pid(paths["pid"])
    if _owned_smoke_pids(runtime_dir):
        print(f"nav smoke already running: pid={existing}")
        return 2
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for key in ("events", "output", "status", "pid"):
        try:
            paths[key].unlink()
        except FileNotFoundError:
            pass
    output = open(paths["output"], "ab", buffering=0)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_run",
        "--runtime-dir",
        str(runtime_dir),
        "--num-envs",
        str(args.num_envs),
    ]
    if args.keep_running:
        command.append("--keep-running")
    process = subprocess.Popen(
        command,
        # In the IDE, agent_ppo is a symlink into /workspace/code. Resolving
        # __file__ therefore loses the project-local kaiwudrl/tools tree.
        cwd=str(launch_cwd),
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    output.close()
    paths["pid"].write_text(f"{process.pid}\n", encoding="utf-8")
    print(
        f"started nav full smoke: pid={process.pid} num_envs={args.num_envs} "
        f"events={paths['events']} output={paths['output']}"
    )
    return 0


def _status(args) -> int:
    runtime_dir = Path(args.runtime_dir).resolve()
    paths = _paths(runtime_dir)
    pid = _read_pid(paths["pid"])
    rows = _events(paths["events"])
    last = rows[-1] if rows else None
    summary = {
        "pid": pid,
        "pid_alive": _alive(pid),
        "running": bool(_owned_smoke_pids(runtime_dir)),
        "event_count": len(rows),
        "target_reached": _target_reached(rows),
        "last_event": last,
        "output": str(paths["output"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["running"] or summary["target_reached"] else 1


def _stop(args) -> int:
    runtime_dir = Path(args.runtime_dir).resolve()
    paths = _paths(runtime_dir)
    pid = _read_pid(paths["pid"])
    owned_pids = _owned_smoke_pids(runtime_dir)
    if not owned_pids:
        if _alive(pid) and not _owned_smoke_process(pid, runtime_dir):
            print(
                f"refusing to stop pid={pid}: procfs does not identify it as "
                "agent_ppo.tools.nav_full_smoke _run",
                file=sys.stderr,
            )
            return 2
        print("nav smoke is not running")
        return 0
    process_groups = set()
    for owned_pid in owned_pids:
        try:
            process_groups.add(os.getpgid(owned_pid))
        except ProcessLookupError:
            continue
    for process_group in process_groups:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + args.grace_seconds
    while _owned_smoke_pids(runtime_dir) and time.monotonic() < deadline:
        time.sleep(0.2)
    remaining_pids = _owned_smoke_pids(runtime_dir)
    remaining_groups = set()
    for remaining_pid in remaining_pids:
        try:
            remaining_groups.add(os.getpgid(remaining_pid))
        except ProcessLookupError:
            continue
    for process_group in remaining_groups:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print(
        "stopped nav full smoke process groups: "
        f"pgids={sorted(process_groups | remaining_groups)}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "stop", "_run"))
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument("--grace-seconds", type=float, default=15.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.num_envs <= 256:
        raise SystemExit("--num-envs must be in [1, 256]")
    if args.command == "start":
        return _start(args)
    if args.command == "status":
        return _status(args)
    if args.command == "stop":
        return _stop(args)
    return _run(Path(args.runtime_dir).resolve(), args.num_envs, not args.keep_running)


if __name__ == "__main__":
    raise SystemExit(main())
