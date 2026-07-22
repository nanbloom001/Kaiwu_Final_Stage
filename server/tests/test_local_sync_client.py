#!/usr/bin/env python3
"""Offline tests for local_sync_client.py."""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "local_sync_client.py"
SPEC = importlib.util.spec_from_file_location("local_sync_client_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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
                    "--cookie-file",
                    str(cache),
                    "--no-cookie-prompt",
                ]
            )
            self.assertEqual(result, 1)
            self.assertFalse(cache.exists())


class DryRunTests(unittest.TestCase):
    def test_dry_run_reads_health_and_manifest_without_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "agent_ppo").mkdir()
            (root / "agent_ppo" / "model.py").write_text("x = 1\n", encoding="utf-8")
            client = mock.Mock()
            client.get.side_effect = [
                {"root": "/remote"},
                {"files": {"agent_ppo/model.py": {"sha256": "old"}}},
            ]
            MODULE.sync_files(
                client,
                root,
                max_bytes=1024,
                delete_remote=False,
                dry_run=True,
                skip_unchanged=False,
            )
            self.assertEqual(client.get.call_args_list[0].args, ("/health",))
            self.assertEqual(client.get.call_args_list[1].args, ("/manifest",))
            client.upload_file_get.assert_not_called()
            client.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
