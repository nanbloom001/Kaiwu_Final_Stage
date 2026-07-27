#!/usr/bin/env python3
"""Offline tests for local_sync_client.py."""

import contextlib
import base64
import importlib.util
import io
import os
import sys
import tarfile
import tempfile
import threading
import types
from http.server import ThreadingHTTPServer
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "local_sync_client.py"
SPEC = importlib.util.spec_from_file_location("local_sync_client_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TONGBU_PATH = Path(__file__).resolve().parents[1] / "conf" / "tongbu.py"
TONGBU_SPEC = importlib.util.spec_from_file_location("tongbu_under_test", TONGBU_PATH)
TONGBU = importlib.util.module_from_spec(TONGBU_SPEC)
assert TONGBU_SPEC.loader is not None
sys.modules[TONGBU_SPEC.name] = TONGBU
TONGBU_SPEC.loader.exec_module(TONGBU)


class CookieSelectionTests(unittest.TestCase):
    def test_explicit_then_cache_then_source_fallback_precedence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "cookie"
            cache.write_text("cache-value", encoding="utf-8")
            explicit = MODULE.load_proxy_cookie(
                "explicit-value",
                "source-value",
                cache,
                "kaiwu-token",
                False,
                True,
            )
            self.assertEqual(explicit.source, "argument/env")

            cached = MODULE.load_proxy_cookie(
                "",
                "source-value",
                cache,
                "kaiwu-token",
                False,
                True,
            )
            self.assertEqual(cached.source, "cache")
            cache.unlink()
            fallback = MODULE.load_proxy_cookie(
                "",
                "source-value",
                cache,
                "kaiwu-token",
                False,
                True,
            )
            self.assertEqual(fallback.source, "source fallback")

    def test_refresh_bypasses_all_existing_sources_and_saves_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "cookie"
            cache.write_text("old-cache", encoding="utf-8")
            with mock.patch.object(MODULE.getpass, "getpass", return_value="fresh-value"):
                selected = MODULE.load_proxy_cookie(
                    "explicit-value",
                    "source-value",
                    cache,
                    "kaiwu-token",
                    False,
                    False,
                    refresh_cookie=True,
                )
            self.assertEqual(selected.source, "prompt")
            self.assertIn("fresh-value", selected.value)
            self.assertEqual(cache.read_text(encoding="utf-8"), selected.value)


class SyncTokenResolutionTests(unittest.TestCase):
    def test_cli_and_environment_take_precedence_over_dotenv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / "conf" / ".env"
            env_path.parent.mkdir()
            env_path.write_text("IDE_SYNC_TOKEN=dotenv-token\n", encoding="utf-8")

            token, source = MODULE.resolve_sync_token(
                cli_token="cli-token",
                root=root,
                environ={"IDE_SYNC_TOKEN": "environment-token"},
            )
            self.assertEqual((token, source), ("cli-token", "--token"))

            token, source = MODULE.resolve_sync_token(
                cli_token=None,
                root=root,
                environ={"IDE_SYNC_TOKEN": "environment-token"},
            )
            self.assertEqual((token, source), ("environment-token", "IDE_SYNC_TOKEN"))

    def test_dotenv_token_is_loaded_without_executing_shell_syntax(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / "conf" / ".env"
            env_path.parent.mkdir()
            env_path.write_text(
                "# local-only credential\n"
                "export IDE_SYNC_TOKEN='dotenv-token'\n"
                "UNRELATED=$(must-not-run)\n",
                encoding="utf-8",
            )

            token, source = MODULE.resolve_sync_token(
                cli_token=None,
                root=root,
                environ={},
            )
            self.assertEqual(token, "dotenv-token")
            self.assertEqual(source, str(env_path))

    def test_main_uses_root_dotenv_without_printing_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / "conf" / ".env"
            env_path.parent.mkdir()
            env_path.write_text("IDE_SYNC_TOKEN=dotenv-token\n", encoding="utf-8")
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "local_sync_client.py",
                        "--root",
                        str(root),
                        "--no-cookie-prompt",
                    ],
                ),
                mock.patch.object(
                    MODULE,
                    "load_proxy_cookie",
                    return_value=MODULE.CookieSelection("test-cookie", "test"),
                ),
                mock.patch.object(MODULE, "sync_files") as sync_files,
                contextlib.redirect_stdout(stdout),
            ):
                result = MODULE.main()

            self.assertEqual(result, 0)
            self.assertEqual(sync_files.call_args.args[0].token, "dotenv-token")
            output = stdout.getvalue()
            self.assertIn(f"Sync token source: {env_path.resolve()}", output)
            self.assertNotIn("dotenv-token", output)


class LocalCheckTests(unittest.TestCase):
    def test_check_local_never_constructs_network_client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "agent_ppo").mkdir()
            (root / "agent_ppo" / "model.py").write_text("x = 1\n", encoding="utf-8")
            stdout = io.StringIO()
            with (
                mock.patch.object(sys, "argv", ["local_sync_client.py", "--check-local", "--root", str(root)]),
                mock.patch.object(MODULE, "SyncClient", side_effect=AssertionError("network client constructed")),
                contextlib.redirect_stdout(stdout),
            ):
                result = MODULE.main()
            self.assertEqual(result, 0)
            self.assertIn("eligible files: 1", stdout.getvalue())


class AuthInvalidationTests(unittest.TestCase):
    def _run_auth_failure(self, argv):
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            MODULE, "sync_files", side_effect=MODULE.TencentProxyAuthError("rejected")
        ):
            return MODULE.main()

    def test_argument_rejection_does_not_delete_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "cookie"
            cache.write_text("valid-cache", encoding="utf-8")
            result = self._run_auth_failure(
                [
                    "local_sync_client.py",
                    "--root",
                    temp_dir,
                    "--token",
                    "test-token",
                    "--cookie-file",
                    str(cache),
                    "--proxy-cookie",
                    "bad-explicit",
                    "--no-cookie-prompt",
                ]
            )
            self.assertEqual(result, 1)
            self.assertTrue(cache.exists())

    def test_cache_rejection_deletes_only_that_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "cookie"
            cache.write_text("bad-cache", encoding="utf-8")
            result = self._run_auth_failure(
                [
                    "local_sync_client.py",
                    "--root",
                    temp_dir,
                    "--token",
                    "test-token",
                    "--cookie-file",
                    str(cache),
                    "--no-cookie-prompt",
                ]
            )
            self.assertEqual(result, 1)
            self.assertFalse(cache.exists())


class DryRunTests(unittest.TestCase):
    def test_dry_run_reads_scoped_manifest_without_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "agent_ppo").mkdir()
            (root / "agent_ppo" / "model.py").write_text("x = 1\n", encoding="utf-8")
            client = mock.Mock()
            client.get.return_value = {
                "root": "/remote",
                "files": {"agent_ppo/model.py": {"sha256": "old"}},
            }
            MODULE.sync_files(
                client,
                root,
                max_bytes=1024,
                delete_remote=False,
                dry_run=True,
                skip_unchanged=False,
            )
            self.assertEqual(client.get.call_args_list[0].args, ("/manifest",))
            self.assertEqual(
                client.get.call_args_list[0].kwargs,
                {"scope": "agent_diy,agent_ppo,conf,isaac_env"},
            )
            client.upload_file_get.assert_not_called()
            client.post.assert_not_called()


class BundleSyncTests(unittest.TestCase):
    class _BundleDispatcher:
        def __init__(self):
            self.payloads = []
            self.server = types.SimpleNamespace(
                bundle_transfers={},
                bundle_transfers_lock=threading.Lock(),
            )

        def _reply(self, payload, status=200):
            self.payloads.append((payload, status))

    @staticmethod
    def _item(data: bytes):
        return {
            "size": len(data),
            "mtime": 1.0,
            "sha256": MODULE.sha256_bytes(data),
            "data": data,
        }

    def test_client_prefers_get_bundle_and_checks_returned_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "agent_ppo" / "model.py"
            path.parent.mkdir()
            data = b"bundle-source"
            path.write_bytes(data)
            item = self._item(data)
            item["path"] = path
            client = mock.Mock()
            client.get.return_value = {
                "root": "/remote",
                "capabilities": ["bundle_get_v2"],
                "files": {"agent_ppo/model.py": {"sha256": "old"}},
            }
            client.upload_bundle_get.return_value = {
                "ok": True,
                "files": {"agent_ppo/model.py": {"sha256": item["sha256"]}},
            }

            MODULE.sync_files(
                client,
                root,
                max_bytes=1024,
                delete_remote=False,
                dry_run=False,
                skip_unchanged=True,
            )

            self.assertEqual(client.upload_bundle_get.call_count, 1)
            bundle_payload = client.upload_bundle_get.call_args.args[0]
            with tarfile.open(fileobj=io.BytesIO(bundle_payload), mode="r:gz") as archive:
                member = archive.getmember("agent_ppo/model.py")
                self.assertEqual(archive.extractfile(member).read(), data)
            client.upload_file_get.assert_not_called()

    def test_client_falls_back_to_parallel_get_chunks_without_capability(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "agent_ppo" / "model.py"
            path.parent.mkdir()
            data = b"legacy-source"
            path.write_bytes(data)
            client = mock.Mock()
            client.get.return_value = {
                "root": "/remote",
                "files": {"agent_ppo/model.py": {"sha256": "old"}},
            }
            client.upload_file_get.return_value = {
                "ok": True,
                "sha256": MODULE.sha256_bytes(data),
            }

            MODULE.sync_files(
                client,
                root,
                max_bytes=1024,
                delete_remote=False,
                dry_run=False,
                skip_unchanged=True,
            )

            client.upload_bundle_get.assert_not_called()
            self.assertEqual(client.upload_file_get.call_count, 1)

    def test_server_bundle_accepts_out_of_order_chunks_and_reports_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ws = TONGBU.Workspace(root, "test-token")
            dispatcher = self._BundleDispatcher()
            data = os.urandom(7000)
            local_files = {"agent_ppo/model.py": self._item(data)}
            bundle = MODULE.build_bundle(local_files, ["agent_ppo/model.py"])
            bundle_id = "a" * 32
            chunk_size = 4096
            TONGBU._bundle_begin(
                dispatcher,
                ws,
                {
                    "bundle_id": [bundle_id],
                    "size": [str(len(bundle))],
                    "chunk_size": [str(chunk_size)],
                },
                None,
            )
            for offset in reversed(range(0, len(bundle), chunk_size)):
                TONGBU._bundle_chunk(
                    dispatcher,
                    ws,
                    {
                        "bundle_id": [bundle_id],
                        "offset": [str(offset)],
                        "data": [
                            base64.urlsafe_b64encode(
                                bundle[offset : offset + chunk_size]
                            ).decode("ascii")
                        ],
                    },
                    None,
                )
            TONGBU._bundle_finish(
                dispatcher, ws, {"bundle_id": [bundle_id]}, None
            )

            self.assertEqual((root / "agent_ppo" / "model.py").read_bytes(), data)
            result = dispatcher.payloads[-1][0]
            self.assertEqual(
                result["files"]["agent_ppo/model.py"]["sha256"],
                MODULE.sha256_bytes(data),
            )
            self.assertEqual(dispatcher.server.bundle_transfers, {})
            self.assertFalse((root / f".ide-sync-{bundle_id}.tar.gz").exists())

    def test_server_bundle_rejects_platform_owned_base_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ws = TONGBU.Workspace(root, "test-token")
            dispatcher = self._BundleDispatcher()
            target = root / "isaac_env" / "base_env.py"
            target.parent.mkdir()
            target.write_bytes(b"platform")
            data = b"must-not-write"
            bundle = MODULE.build_bundle(
                {"isaac_env/base_env.py": self._item(data)},
                ["isaac_env/base_env.py"],
            )
            bundle_id = "b" * 32
            TONGBU._bundle_begin(
                dispatcher,
                ws,
                {
                    "bundle_id": [bundle_id],
                    "size": [str(len(bundle))],
                    "chunk_size": ["4096"],
                },
                None,
            )
            TONGBU._bundle_chunk(
                dispatcher,
                ws,
                {
                    "bundle_id": [bundle_id],
                    "offset": ["0"],
                    "data": [base64.urlsafe_b64encode(bundle).decode("ascii")],
                },
                None,
            )

            with self.assertRaises(PermissionError):
                TONGBU._bundle_finish(
                    dispatcher, ws, {"bundle_id": [bundle_id]}, None
                )
            self.assertEqual(target.read_bytes(), b"platform")

    def test_get_bundle_round_trip_through_actual_http_routes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            local_root = Path(temp_dir) / "local"
            remote_root = Path(temp_dir) / "remote"
            local_file = local_root / "agent_ppo" / "model.py"
            local_file.parent.mkdir(parents=True)
            remote_root.mkdir()
            payload = os.urandom(9000)
            local_file.write_bytes(payload)

            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), TONGBU.RequestDispatcher
            )
            server.workspace = TONGBU.Workspace(remote_root, "test-token")
            server.bundle_transfers = {}
            server.bundle_transfers_lock = threading.Lock()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                client = MODULE.SyncClient(
                    f"http://127.0.0.1:{port}", "test-token", 10
                )
                MODULE.sync_files(
                    client,
                    local_root,
                    max_bytes=32 * 1024,
                    delete_remote=False,
                    dry_run=False,
                    skip_unchanged=True,
                )
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

            self.assertEqual(
                (remote_root / "agent_ppo" / "model.py").read_bytes(), payload
            )


class PlatformOwnedSyncPathTests(unittest.TestCase):
    """The platform bootstrap owns base_env.py, not the repository sync."""

    class _ReplyCapture:
        def __init__(self):
            self.payloads = []

        def _reply(self, payload, status=200):
            self.payloads.append((payload, status))

    def test_client_excludes_only_platform_owned_base_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            isaac_env = root / "isaac_env"
            isaac_env.mkdir()
            (isaac_env / "base_env.py").write_text("platform\n", encoding="utf-8")
            (isaac_env / "overlay.py").write_text("local\n", encoding="utf-8")

            files = MODULE.collect_local_files(root, max_bytes=1024)

            self.assertNotIn("isaac_env/base_env.py", files)
            self.assertIn("isaac_env/overlay.py", files)
            self.assertFalse(MODULE.is_in_sync_scope("isaac_env/base_env.py"))
            self.assertFalse(MODULE.is_in_sync_scope("isaac_env/subdir/../base_env.py"))
            self.assertTrue(MODULE.is_in_sync_scope("isaac_env/overlay.py"))

    def test_client_dry_run_neither_uploads_nor_deletes_base_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            isaac_env = root / "isaac_env"
            isaac_env.mkdir()
            (isaac_env / "overlay.py").write_text("local\n", encoding="utf-8")
            client = mock.Mock()
            client.get.return_value = {
                "root": "/remote",
                "files": {
                    "isaac_env/base_env.py": {"sha256": "platform"},
                    "isaac_env/stale.py": {"sha256": "stale"},
                    "isaac_env/overlay.py": {"sha256": "old"},
                },
            }
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                MODULE.sync_files(
                    client,
                    root,
                    max_bytes=1024,
                    delete_remote=True,
                    dry_run=True,
                    skip_unchanged=True,
                )

            output = stdout.getvalue()
            self.assertIn("isaac_env/overlay.py", output)
            self.assertIn("isaac_env/stale.py", output)
            self.assertNotIn("isaac_env/base_env.py", output)
            client.upload_file_get.assert_not_called()
            client.post.assert_not_called()

    def test_server_rejects_protected_path_in_all_mutation_endpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = TONGBU.Workspace(Path(temp_dir), "test-token")
            dispatcher = self._ReplyCapture()
            for raw_path in (
                "./isaac_env\\base_env.py",
                "isaac_env/subdir/../base_env.py",
                "isaac_env%2Fbase_env.py",
            ):
                handlers = (
                    lambda: TONGBU._write(
                        dispatcher,
                        workspace,
                        {},
                        {"path": raw_path, "content_base64": ""},
                    ),
                    lambda: TONGBU._write_begin(
                        dispatcher, workspace, {"path": [raw_path]}, None
                    ),
                    lambda: TONGBU._write_chunk(
                        dispatcher, workspace, {"path": [raw_path], "data": [""]}, None
                    ),
                    lambda: TONGBU._write_finish(
                        dispatcher, workspace, {"path": [raw_path]}, None
                    ),
                    lambda: TONGBU._delete(
                        dispatcher, workspace, {}, {"paths": [raw_path]}
                    ),
                )

                for handler in handlers:
                    with self.assertRaises(PermissionError):
                        handler()

    def test_server_allows_other_isaac_env_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = TONGBU.Workspace(Path(temp_dir), "test-token")
            dispatcher = self._ReplyCapture()
            payload = base64.b64encode(b"overlay\n").decode("ascii")

            TONGBU._write(
                dispatcher,
                workspace,
                {},
                {"path": "isaac_env/overlay.py", "content_base64": payload},
            )
            target = Path(temp_dir) / "isaac_env" / "overlay.py"
            self.assertEqual(target.read_bytes(), b"overlay\n")
            TONGBU._delete(
                dispatcher,
                workspace,
                {},
                {"paths": ["isaac_env/overlay.py"]},
            )
            self.assertFalse(target.exists())

    def test_workspace_rejects_symlink_escape(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.TemporaryDirectory() as outside_dir,
        ):
            root = Path(temp_dir)
            (root / "outside_link").symlink_to(
                Path(outside_dir), target_is_directory=True
            )
            workspace = TONGBU.Workspace(root, "test-token")

            with self.assertRaisesRegex(ValueError, "path escapes root"):
                workspace.resolve("outside_link/payload.py")

    def test_protected_path_rejects_symlink_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            isaac_env = root / "isaac_env"
            isaac_env.mkdir()
            (root / "alias").symlink_to(isaac_env, target_is_directory=True)
            workspace = TONGBU.Workspace(root, "test-token")

            with self.assertRaises(PermissionError):
                TONGBU._write(
                    self._ReplyCapture(),
                    workspace,
                    {},
                    {
                        "path": "alias/base_env.py",
                        "content_base64": base64.b64encode(b"blocked").decode(
                            "ascii"
                        ),
                    },
                )


class ContainerExecRpcTests(unittest.TestCase):
    class _ReplyCapture:
        def __init__(self):
            self.payloads = []

        def _reply(self, payload, status=200):
            self.payloads.append((payload, status))

    @staticmethod
    def _encoded(command: str) -> str:
        return base64.urlsafe_b64encode(command.encode("utf-8")).decode("ascii")

    def test_exec_runs_inside_requested_workspace_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "agent_ppo"
            target.mkdir()
            dispatcher = self._ReplyCapture()

            TONGBU._exec_b64(
                dispatcher,
                TONGBU.Workspace(root, "test-token"),
                {
                    "cmd": [self._encoded("printf rpc-ok")],
                    "cwd": ["agent_ppo"],
                    "timeout": ["5"],
                },
                None,
            )

            payload, status = dispatcher.payloads[-1]
            self.assertEqual(status, 200)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["returncode"], 0)
            self.assertEqual(payload["cwd"], "agent_ppo")
            self.assertEqual(
                base64.b64decode(payload["stdout_base64"]), b"rpc-ok"
            )

    def test_exec_rejects_cwd_outside_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = TONGBU.Workspace(Path(temp_dir), "test-token")
            with self.assertRaises(ValueError):
                TONGBU._exec_b64(
                    self._ReplyCapture(),
                    workspace,
                    {
                        "cmd": [self._encoded("true")],
                        "cwd": ["../outside"],
                        "timeout": ["5"],
                    },
                    None,
                )

    def test_health_advertises_exec_capability(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dispatcher = self._ReplyCapture()
            TONGBU._health(
                dispatcher,
                TONGBU.Workspace(Path(temp_dir), "test-token"),
                {},
                None,
            )
            payload, _status = dispatcher.payloads[-1]
            self.assertIn("exec_b64_v1", payload["capabilities"])


class EmptyFinishResponseTests(unittest.TestCase):
    def test_upload_accepts_empty_finish_only_after_matching_readback(self):
        client = MODULE.SyncClient("https://example.invalid", "token", 30)
        payload = b"bridge-sync"
        client.get = mock.Mock(
            side_effect=[
                {"ok": True},
                {"ok": True},
                {"ok": True, "empty_response": True, "status": 200},
                {"sha256": MODULE.sha256_bytes(payload)},
            ]
        )

        result = client.upload_file_get("agent_ppo/model.py", payload, 1.0)

        self.assertTrue(result["ok"])
        self.assertTrue(result["verified_after_empty_response"])
        self.assertEqual(result["sha256"], MODULE.sha256_bytes(payload))
        self.assertEqual(client.get.call_args_list[-1].args, ("/read",))

    def test_upload_keeps_empty_finish_failure_when_readback_mismatches(self):
        client = MODULE.SyncClient("https://example.invalid", "token", 30)
        client.get = mock.Mock(
            side_effect=[
                {"ok": True},
                {"ok": True},
                {"ok": True, "empty_response": True, "status": 200},
                {"sha256": "different"},
            ]
        )

        result = client.upload_file_get("agent_ppo/model.py", b"bridge-sync", 1.0)

        self.assertTrue(result["empty_response"])
        self.assertNotIn("sha256", result)


if __name__ == "__main__":
    unittest.main()
