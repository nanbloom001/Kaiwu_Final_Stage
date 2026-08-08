#!/usr/bin/env python3
"""Regression tests for local browser authentication configuration."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

import browser_auth  # noqa: E402


class BrowserAuthTests(unittest.TestCase):
    def test_default_session_bypasses_legacy_daemon(self):
        self.assertEqual(browser_auth.DEFAULT_SESSION, "tencent-arena-persistent-v2")

    def test_load_local_env_preserves_explicit_process_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "AGENT_BROWSER_SESSION=from-file\nAGENT_BROWSER_HEADED='1' # comment\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"AGENT_BROWSER_SESSION": "from-process"},
                clear=True,
            ):
                browser_auth.load_local_env(env_file)
                self.assertEqual(os.environ["AGENT_BROWSER_SESSION"], "from-process")
                self.assertEqual(os.environ["AGENT_BROWSER_HEADED"], "1")

    def test_agent_browser_env_uses_private_persistent_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile = Path(temp_dir) / "profile"
            with mock.patch.dict(
                os.environ,
                {"AGENT_BROWSER_PROFILE": str(profile)},
                clear=True,
            ):
                env = browser_auth.agent_browser_env("session-a", "saved-a")
            self.assertEqual(env["AGENT_BROWSER_SESSION"], "session-a")
            self.assertEqual(env["AGENT_BROWSER_SESSION_NAME"], "saved-a")
            self.assertEqual(env["AGENT_BROWSER_HEADED"], "1")
            self.assertEqual(Path(env["AGENT_BROWSER_PROFILE"]), profile.resolve())
            self.assertTrue(profile.is_dir())
            self.assertEqual(profile.stat().st_mode & 0o777, 0o700)

    def test_relative_profile_is_resolved_from_tool_directory(self):
        relative = Path("runtime") / "test-browser-profile"
        expected = (TOOL_ROOT / relative).resolve()
        with mock.patch.dict(
            os.environ,
            {"AGENT_BROWSER_PROFILE": str(relative)},
            clear=True,
        ), mock.patch.object(Path, "mkdir") as mkdir, mock.patch.object(Path, "chmod"):
            env = browser_auth.agent_browser_env("session-a", "saved-a")
        self.assertEqual(Path(env["AGENT_BROWSER_PROFILE"]), expected)
        mkdir.assert_called_once_with(parents=True, exist_ok=True, mode=0o700)

    def test_empty_profile_uses_safe_default(self):
        with mock.patch.dict(
            os.environ,
            {"AGENT_BROWSER_PROFILE": ""},
            clear=True,
        ), mock.patch.object(Path, "mkdir") as mkdir, mock.patch.object(Path, "chmod"):
            env = browser_auth.agent_browser_env("session-a", "saved-a")
        self.assertEqual(
            Path(env["AGENT_BROWSER_PROFILE"]),
            browser_auth.DEFAULT_PROFILE_DIR.resolve(),
        )
        mkdir.assert_called_once_with(parents=True, exist_ok=True, mode=0o700)

    def test_profile_cannot_be_tool_or_repository_ancestor(self):
        for unsafe_path in (browser_auth.ROOT, browser_auth.ROOT.parents[1]):
            with self.subTest(path=unsafe_path), mock.patch.dict(
                os.environ,
                {"AGENT_BROWSER_PROFILE": str(unsafe_path)},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "dedicated subdirectory"):
                    browser_auth.agent_browser_env("session-a", "saved-a")

    def test_repository_local_profile_must_be_under_runtime(self):
        unsafe_path = browser_auth.ROOT / "custom_profile"
        with mock.patch.dict(
            os.environ,
            {"AGENT_BROWSER_PROFILE": str(unsafe_path)},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "Git-ignored runtime"):
                browser_auth.agent_browser_env("session-a", "saved-a")

    def test_legacy_env_session_is_migrated_with_opt_out(self):
        with mock.patch.dict(
            os.environ,
            {
                "AGENT_BROWSER_SESSION": "tencent-arena",
                "AGENT_BROWSER_SESSION_NAME": "tencent-arena",
            },
            clear=True,
        ), mock.patch("sys.stderr"):
            session = browser_auth.default_session_from_env()
            session_name = browser_auth.default_session_name_from_env(session)
        self.assertEqual(session, browser_auth.DEFAULT_SESSION)
        self.assertEqual(session_name, browser_auth.DEFAULT_SESSION)

        with mock.patch.dict(
            os.environ,
            {
                "AGENT_BROWSER_SESSION": "tencent-arena",
                "AGENT_BROWSER_ALLOW_LEGACY_SESSION": "1",
            },
            clear=True,
        ):
            self.assertEqual(browser_auth.default_session_from_env(), "tencent-arena")

    def test_agent_browser_command_pins_auth_as_cli_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile = Path(temp_dir) / "profile"
            with mock.patch.dict(
                os.environ,
                {"AGENT_BROWSER_PROFILE": str(profile)},
                clear=True,
            ), mock.patch.object(
                browser_auth.shutil,
                "which",
                return_value="/test/agent-browser",
            ):
                command, env = browser_auth.agent_browser_command(
                    ["get", "url"],
                    "session-a",
                    "saved-a",
                )
        self.assertEqual(
            command,
            [
                "/test/agent-browser",
                "--session",
                "session-a",
                "--session-name",
                "saved-a",
                "--profile",
                str(profile.resolve()),
                "get",
                "url",
            ],
        )
        self.assertEqual(env["AGENT_BROWSER_PROFILE"], str(profile.resolve()))


if __name__ == "__main__":
    unittest.main()
