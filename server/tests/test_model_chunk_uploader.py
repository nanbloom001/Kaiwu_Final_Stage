#!/usr/bin/env python3
"""Offline tests for model_chunk_uploader.py."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_ROOT))
MODULE_PATH = SERVER_ROOT / "model_chunk_uploader.py"
SPEC = importlib.util.spec_from_file_location("model_chunk_uploader_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ModelChunkUploaderTests(unittest.TestCase):
    def test_default_concurrency_is_two_by_two(self):
        self.assertEqual(MODULE.DEFAULT_PART_BYTES, 1024 * 1024)
        self.assertEqual(MODULE.DEFAULT_PART_WORKERS, 2)
        self.assertEqual(MODULE.DEFAULT_CHUNK_WORKERS, 2)

    def test_manifest_entry_accepts_scope_relative_paths(self):
        digest = "abc123"
        manifest = {
            "files": {
                "test_artifacts/model.pkl": {"sha256": digest},
            }
        }
        self.assertEqual(
            MODULE.manifest_file_entry(
                manifest, "agent_ppo/test_artifacts/model.pkl"
            )["sha256"],
            digest,
        )

    def test_remote_path_is_confined_to_test_artifacts(self):
        self.assertEqual(
            MODULE.validate_remote_path("agent_ppo/test_artifacts/model.pkl"),
            "agent_ppo/test_artifacts/model.pkl",
        )
        for invalid in (
            "/tmp/model.pkl",
            "../model.pkl",
            "agent_ppo/model.pkl",
            "agent_ppo/test_artifacts",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MODULE.validate_remote_path(invalid)

    def test_split_file_preserves_order_size_and_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.pkl"
            payload = bytes(range(251)) * 1000
            source.write_bytes(payload)
            parts, digest, size = MODULE.split_file(
                source,
                "agent_ppo/test_artifacts/model.pkl",
                64 * 1024,
            )
        self.assertEqual(b"".join(part.data for part in parts), payload)
        self.assertEqual(size, len(payload))
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
        self.assertEqual([part.index for part in parts], list(range(len(parts))))

    def test_upload_model_resumes_matching_parts_then_merges(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.pkl"
            source.write_bytes(b"a" * (64 * 1024) + b"b" * (64 * 1024))
            parts, digest, _size = MODULE.split_file(
                source,
                "agent_ppo/test_artifacts/model.pkl",
                64 * 1024,
            )
            client = mock.Mock()
            client.get.side_effect = [
                {
                    "files": {
                        parts[0].remote_path: {"sha256": parts[0].sha256},
                    }
                },
                {
                    "files": {
                        "agent_ppo/test_artifacts/model.pkl": {
                            "sha256": digest
                        }
                    }
                },
            ]
            with (
                mock.patch.object(MODULE, "upload_part") as upload_part,
                mock.patch.object(MODULE, "run_remote_command") as merge,
            ):
                MODULE.upload_model(
                    client,
                    source,
                    "agent_ppo/test_artifacts/model.pkl",
                    64 * 1024,
                    2,
                    2,
                    1,
                    False,
                    False,
                    30,
                )
        upload_part.assert_called_once()
        self.assertEqual(upload_part.call_args.args[1].index, 1)
        merge.assert_called_once()

    def test_upload_model_skips_scope_relative_verified_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.pkl"
            payload = b"verified-model"
            source.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            client = mock.Mock()
            client.get.return_value = {
                "files": {
                    "test_artifacts/model.pkl": {"sha256": digest},
                }
            }
            with (
                mock.patch.object(MODULE, "upload_part") as upload_part,
                mock.patch.object(MODULE, "run_remote_command") as merge,
            ):
                result = MODULE.upload_model(
                    client,
                    source,
                    "agent_ppo/test_artifacts/model.pkl",
                    64 * 1024,
                    2,
                    2,
                    1,
                    False,
                    False,
                    30,
                )
        self.assertEqual(result, (digest, len(payload)))
        upload_part.assert_not_called()
        merge.assert_not_called()

    def test_merge_command_contains_atomic_verification_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.pkl"
            source.write_bytes(b"x" * (64 * 1024))
            parts, digest, size = MODULE.split_file(
                source,
                "agent_ppo/test_artifacts/model.pkl",
                64 * 1024,
            )
        command = MODULE.build_merge_command(
            "agent_ppo/test_artifacts/model.pkl",
            parts,
            size,
            digest,
            False,
        )
        self.assertIn(".uploading", command)
        self.assertIn("os.replace", command)
        self.assertIn(digest, command)


if __name__ == "__main__":
    unittest.main()
