#!/usr/bin/env python3
"""Safety tests for the container training cleanup tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SERVER_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = SERVER_ROOT / "conf" / "container_training_cleanup.py"
SPEC = importlib.util.spec_from_file_location("container_training_cleanup", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ContainerTrainingCleanupTests(unittest.TestCase):
    def _workspace(self, root: Path):
        code = root / "code"
        project = root / "project"
        tmp = root / "tmp"
        (code / "agent_ppo" / "tests").mkdir(parents=True)
        (code / "agent_ppo" / "tools").mkdir(parents=True)
        (code / "agent_ppo" / "test_artifacts" / "parent").mkdir(parents=True)
        (code / "agent_ppo" / "__pycache__").mkdir(parents=True)
        (code / "conf" / "__pycache__").mkdir(parents=True)
        (code / ".git" / "objects").mkdir(parents=True)
        (code / ".vscode").mkdir(parents=True)
        (project / "kaiwudrl").mkdir(parents=True)
        (tmp / "IsaacLab").mkdir(parents=True)
        (code / "agent_ppo" / "agent.py").write_text("# runtime\n")
        (code / "conf" / ".env").write_text("SECRET=preserved\n")
        (code / ".git" / "objects" / "pack").write_bytes(b"git")
        (code / "agent_ppo" / "tests" / "test_x.py").write_text("pass\n")
        (code / "agent_ppo" / "tools" / "smoke.py").write_text("pass\n")
        (code / "agent_ppo" / "test_artifacts" / "parent" / "model.pkl").write_bytes(b"model")
        (code / "agent_ppo" / "__pycache__" / "agent.pyc").write_bytes(b"cache")
        (code / "conf" / "__pycache__" / "cleanup.pyc").write_bytes(b"cache")
        (project / ".ide-sync-stale.tar.gz").write_bytes(b"bundle")
        (tmp / "p3_joint.log").write_text("log\n")
        return code, project, tmp

    def test_default_dry_run_preserves_dev_files_and_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            code, project, tmp = self._workspace(Path(directory))
            MODULE.validate_roots(code, project)
            targets = MODULE.collect_targets(
                code, project, tmp, remove_dev_files=False
            )
            count, reclaimable = MODULE.run_cleanup(targets, apply=False)
            self.assertGreater(count, 0)
            self.assertGreater(reclaimable, 0)
            self.assertTrue((code / ".git").exists())
            self.assertTrue((code / "agent_ppo" / "tests").exists())
            self.assertTrue((code / "agent_ppo" / "test_artifacts").exists())
            self.assertTrue((code / "conf" / ".env").exists())

    def test_apply_with_dev_files_matches_verified_cleanup_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            code, project, tmp = self._workspace(Path(directory))
            targets = MODULE.collect_targets(
                code, project, tmp, remove_dev_files=True
            )
            MODULE.run_cleanup(targets, apply=True)
            for relative in (
                ".git",
                ".vscode",
                "agent_ppo/tests",
                "agent_ppo/tools",
                "agent_ppo/test_artifacts",
                "agent_ppo/__pycache__",
                "conf/__pycache__",
            ):
                self.assertFalse((code / relative).exists(), relative)
            self.assertFalse((project / ".ide-sync-stale.tar.gz").exists())
            self.assertFalse((tmp / "IsaacLab").exists())
            self.assertFalse((tmp / "p3_joint.log").exists())
            self.assertTrue((code / "agent_ppo" / "agent.py").exists())
            self.assertTrue((code / "conf" / ".env").exists())

    def test_root_validation_rejects_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                MODULE.validate_roots(root / "code", root / "project")


if __name__ == "__main__":
    unittest.main()
