"""Regression tests for IDE-sync credential loading and path containment."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SERVER_ROOT = Path(__file__).resolve().parents[1]
SYNC_ENV_PATH = SERVER_ROOT / "conf" / "sync_env.py"
TONGBU_PATH = SERVER_ROOT / "conf" / "tongbu.py"
LOCAL_CLIENT_PATH = SERVER_ROOT / "local_sync_client.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SYNC_ENV = _load_module("sync_env_under_test", SYNC_ENV_PATH)
LOCAL_CLIENT = _load_module("local_sync_client_env_under_test", LOCAL_CLIENT_PATH)


class EnvFileTests(unittest.TestCase):
    def test_file_is_not_shell_evaluated_and_process_env_has_precedence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "IDE_SYNC_TOKEN=file-token\nIGNORED=$(do-not-run)\n",
                encoding="utf-8",
            )
            values = SYNC_ENV.load_env_file(str(env_file))
            self.assertEqual(values["IGNORED"], "$(do-not-run)")
            with mock.patch.dict("os.environ", {"IDE_SYNC_TOKEN": "process-token"}, clear=False):
                self.assertEqual(
                    SYNC_ENV.setting(None, "IDE_SYNC_TOKEN", values),
                    "process-token",
                )
            self.assertEqual(
                SYNC_ENV.setting("cli-token", "IDE_SYNC_TOKEN", values),
                "cli-token",
            )

    def test_empty_process_value_falls_back_to_file(self):
        with mock.patch.dict("os.environ", {"IDE_SYNC_TOKEN": ""}, clear=False):
            self.assertEqual(
                SYNC_ENV.setting(None, "IDE_SYNC_TOKEN", {"IDE_SYNC_TOKEN": "file-token"}),
                "file-token",
            )

    def test_rejects_invalid_syntax(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("not a dotenv assignment\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                SYNC_ENV.load_env_file(str(env_file))


class WorkspaceContainmentTests(unittest.TestCase):
    def test_rejects_existing_symlink_that_escapes_sync_root(self):
        tongbu = _load_module("tongbu_under_test", TONGBU_PATH)
        with tempfile.TemporaryDirectory() as root_dir, tempfile.TemporaryDirectory() as outside_dir:
            root = Path(root_dir)
            outside = Path(outside_dir)
            (root / "agent_ppo").mkdir()
            (root / "agent_ppo" / "inside.py").write_text("ok", encoding="utf-8")
            (root / "agent_ppo" / "escape").symlink_to(outside, target_is_directory=True)
            workspace = tongbu.Workspace(base_dir=root, api_key="test")
            self.assertEqual(
                workspace.resolve("agent_ppo/inside.py"),
                (root / "agent_ppo" / "inside.py").resolve(),
            )
            with self.assertRaisesRegex(ValueError, "escapes root"):
                workspace.resolve("agent_ppo/escape/outside.py")


class LocalClientEnvFileTests(unittest.TestCase):
    def test_env_file_supplies_client_token_without_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "agent_ppo").mkdir()
            env_file = root / ".env"
            env_file.write_text("IDE_SYNC_TOKEN=file-token\n", encoding="utf-8")
            cookie_file = root / "cookie"
            with (
                mock.patch.dict("os.environ", {}, clear=True),
                mock.patch.object(sys, "argv", [
                    "local_sync_client.py",
                    "--env-file", str(env_file),
                    "--root", str(root),
                    "--cookie-file", str(cookie_file),
                    "--no-cookie-prompt",
                ]),
                mock.patch.object(LOCAL_CLIENT, "sync_files") as sync_files,
                mock.patch.object(LOCAL_CLIENT, "SyncClient") as client_cls,
            ):
                self.assertEqual(LOCAL_CLIENT.main(), 0)
            self.assertEqual(client_cls.call_args.args[1], "file-token")
            sync_files.assert_called_once()


if __name__ == "__main__":
    unittest.main()
