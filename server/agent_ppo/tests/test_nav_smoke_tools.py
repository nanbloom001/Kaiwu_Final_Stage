#!/usr/bin/env python3
"""Structured nav smoke event and launcher regressions."""

from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_ppo.feature.nav_event_log import EVENT_LOG_ENV, emit_nav_event
from agent_ppo.tools import nav_full_smoke


def _write_events(path: str, worker: int, count: int) -> None:
    os.environ[EVENT_LOG_ENV] = path
    for index in range(count):
        emit_nav_event("parallel", worker=worker, index=index)


class TestNavEventLog(unittest.TestCase):
    def test_disabled_event_log_is_noop(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(emit_nav_event("disabled"))

    def test_multi_process_events_remain_one_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "events.jsonl")
            context = multiprocessing.get_context("fork")
            workers = [
                context.Process(target=_write_events, args=(path, worker, 25))
                for worker in range(4)
            ]
            for process in workers:
                process.start()
            for process in workers:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            rows = [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 100)
            self.assertTrue(all(row["event"] == "parallel" for row in rows))

    def test_nonfinite_fields_are_json_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with mock.patch.dict(os.environ, {EVENT_LOG_ENV: str(path)}):
                self.assertTrue(emit_nav_event("metric", value=float("nan")))
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(row["value"])


class TestNavFullSmokeLauncher(unittest.TestCase):
    def test_smoke_uses_one_dump_per_complete_tbptt_iteration(self):
        self.assertEqual(nav_full_smoke.SMOKE_DUMP_MODEL_FREQ, 160)

    def test_paths_are_runtime_only(self):
        paths = nav_full_smoke._paths(Path("/tmp/example-nav-smoke"))
        self.assertTrue(all(str(path).startswith("/tmp/") for path in paths.values()))

    def test_event_reader_ignores_partial_last_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text('{"event":"ok"}\n{"event":', encoding="utf-8")
            self.assertEqual(nav_full_smoke._events(path), [{"event": "ok"}])

    def test_target_requires_completed_iteration_with_valid_ticks(self):
        self.assertFalse(
            nav_full_smoke._target_reached(
                [{"event": "first_update_complete", "valid_ticks": 8}]
            )
        )
        self.assertFalse(
            nav_full_smoke._target_reached(
                [{"event": "iteration", "valid_ticks": 0}]
            )
        )
        self.assertFalse(
            nav_full_smoke._target_reached(
                [{"event": "iteration", "valid_ticks": 8}]
            )
        )
        self.assertTrue(
            nav_full_smoke._target_reached(
                [
                    {"event": "platform_dump_boundary"},
                    {"event": "iteration", "valid_ticks": 8},
                ]
            )
        )

    def test_stop_refuses_live_pid_without_smoke_process_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = nav_full_smoke._paths(Path(directory))
            paths["pid"].write_text("4321\n", encoding="utf-8")
            args = mock.Mock(runtime_dir=directory, grace_seconds=0.0)
            with mock.patch.object(nav_full_smoke, "_alive", return_value=True), mock.patch.object(
                nav_full_smoke, "_owned_smoke_process", return_value=False
            ), mock.patch.object(nav_full_smoke.os, "killpg") as killpg:
                self.assertEqual(nav_full_smoke._stop(args), 2)
            killpg.assert_not_called()

    def test_start_uses_detached_process_group_without_editing_toml(self):
        with tempfile.TemporaryDirectory() as directory:
            args = mock.Mock(
                runtime_dir=directory,
                num_envs=8,
                keep_running=False,
            )
            process = mock.Mock(pid=4321)
            with mock.patch.object(
                nav_full_smoke.subprocess, "Popen", return_value=process
            ) as popen:
                self.assertEqual(nav_full_smoke._start(args), 0)
            kwargs = popen.call_args.kwargs
            self.assertTrue(kwargs["start_new_session"])
            command = popen.call_args.args[0]
            self.assertIn("_run", command)
            self.assertIn("--num-envs", command)
            self.assertFalse(any(item.endswith(".toml") for item in command))


if __name__ == "__main__":
    unittest.main()
