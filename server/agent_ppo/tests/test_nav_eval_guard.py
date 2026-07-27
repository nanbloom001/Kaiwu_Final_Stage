#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""nav eval 硬门测试：validate_nav_eval_bundle 负例 + agent 布线源码断言。

源码文本断言沿用 server/tests/test_visual_policy_optimization.py 的先例
（对不便在本地完整构造的 Agent 类，断言关键防线存在于源码中）。
"""

import os
import pathlib
import tempfile
import unittest

import agent_ppo.tests._nav_test_stubs  # noqa: F401  平台模块 stub，必须先于其他 agent_ppo import

import torch
import torch.nn as nn

from agent_ppo import checkpoint_io as cio
from agent_ppo.algorithm.algorithm_nav_dagger import AlgorithmNavDagger
from agent_ppo.feature import nav_contract as nc
from agent_ppo.model.high_level_policy import HighLevelPolicy
from agent_ppo.model.vision_encoder import VisionEncoder

_AGENT_SRC = (
    pathlib.Path(__file__).resolve().parent.parent / "agent.py"
).read_text(encoding="utf-8")

EXPECTED_LOW = {
    "proprio_dim": 45,
    "scan_dim": 256,
    "latent_dim": 32,
    "action_dim": 12,
    "goal_dim": 0,
}
EXPECTED_HIGH = {
    "input_layout_version": nc.INPUT_LAYOUT_VERSION,
    "vocab": [list(v) for v in nc.VOCAB],
    "nav_period_frames": nc.NAV_PERIOD_FRAMES,
    "min_dwell_ticks": nc.MIN_DWELL_TICKS,
}


def _mk_nav_bundle_file(directory, model_id="777"):
    actor = nn.Sequential(
        nn.Linear(77, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12),
    )
    algo = AlgorithmNavDagger(
        vision_encoder=VisionEncoder(),
        low_level_actor=actor,
        high_level=HighLevelPolicy(),
        device="cpu",
        low_level_parent_model_id="34728",
    )
    algo.low_level_state_digest = cio.compute_low_level_state_digest(
        algo._live_low_level_bundle_view()
    )
    algo.source_parent_model_id = "34728"
    path = os.path.join(directory, f"model.ckpt-navfull-{model_id}.pkl")
    algo.save_nav_bundle(path, platform_model_id=model_id, phase_label="navfull")
    return path


class TestValidateNavEvalBundle(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = _mk_nav_bundle_file(self.dir, "777")
        self.bundle = torch.load(self.path, weights_only=False)

    def test_positive_chain_passes(self):
        info = cio.validate_nav_eval_bundle(
            self.bundle, "777", EXPECTED_LOW, EXPECTED_HIGH
        )
        self.assertTrue(info["high_level_present"])
        self.assertEqual(len(info["low_level_state_digest"]), 64)

    def test_wrong_id_hard_stops(self):
        with self.assertRaises(ValueError):
            cio.validate_nav_eval_bundle(
                self.bundle, "888", EXPECTED_LOW, EXPECTED_HIGH
            )

    def test_missing_high_level_hard_stops(self):
        bundle = dict(self.bundle)
        bundle["modules"] = {
            k: v for k, v in bundle["modules"].items() if k != "high_level"
        }
        with self.assertRaises(KeyError):
            cio.validate_nav_eval_bundle(bundle, "777", EXPECTED_LOW, EXPECTED_HIGH)

    def test_tampered_low_level_digest_hard_stops(self):
        bundle = torch.load(self.path, weights_only=False)
        actor_state = bundle["modules"]["low_level"]["actor_state_dict"]
        first_key = next(iter(actor_state))
        actor_state[first_key] = actor_state[first_key] + 1.0
        with self.assertRaises(ValueError):
            cio.validate_nav_eval_bundle(bundle, "777", EXPECTED_LOW, EXPECTED_HIGH)

    def test_vocab_mismatch_hard_stops(self):
        wrong_high = dict(EXPECTED_HIGH)
        wrong_high["vocab"] = [[9.9, 9.9, 9.9]]
        with self.assertRaises(ValueError):
            cio.validate_nav_eval_bundle(self.bundle, "777", EXPECTED_LOW, wrong_high)

    def test_missing_lineage_digest_hard_stops(self):
        bundle = torch.load(self.path, weights_only=False)
        bundle["lineage"] = dict(bundle["lineage"])
        bundle["lineage"]["low_level_state_digest"] = None
        with self.assertRaises(ValueError):
            cio.validate_nav_eval_bundle(bundle, "777", EXPECTED_LOW, EXPECTED_HIGH)

    def test_eval_candidates_never_fall_back_to_loco_labels(self):
        # 同目录放一个 command 低层包；nav eval 候选不得包含它
        torch.save(
            {"format": cio.KAIWU_TRAIN_FORMAT, "schema_version": 1},
            os.path.join(self.dir, "model.ckpt-commandfull-777.pkl"),
        )
        candidates = cio.nav_eval_checkpoint_candidates(self.dir, "777")
        self.assertEqual(candidates, [self.path])
        # 没有 nav 文件的 ID → 空（宁可硬失败，不静默评 loco-only）
        self.assertEqual(cio.nav_eval_checkpoint_candidates(self.dir, "999"), [])


class TestAgentWiringSource(unittest.TestCase):
    """Agent 类依赖平台运行时，本地不整机构造；按仓库先例做源码防线断言。"""

    def test_lbc_eval_guard_refuses_nav_bundles(self):
        self.assertIn('if "high_level" in modules:', _AGENT_SRC)
        self.assertIn("Refusing to silently drop the high", _AGENT_SRC)

    def test_nav_dispatch_present(self):
        self.assertIn('self.is_nav_dagger = self.algorithm_name == "nav_dagger"', _AGENT_SRC)
        self.assertIn('self.is_nav_eval = self.algorithm_name == "nav_eval"', _AGENT_SRC)
        self.assertIn("_init_nav_dagger", _AGENT_SRC)
        self.assertIn("_load_nav_for_eval", _AGENT_SRC)

    def test_storage_exclusion_covers_nav(self):
        self.assertIn("or self.is_nav_dagger", _AGENT_SRC)
        self.assertIn("or self.is_nav_eval", _AGENT_SRC)

    def test_no_side_locomotion_for_nav(self):
        # save_model :811 排除集必须含 nav（禁别名文件）
        marker = "self.is_lbc\n            or self.is_visual_ppo\n            or self.is_nav_dagger"
        self.assertIn(marker, _AGENT_SRC)

    def test_eval_refuses_unloaded_checkpoint(self):
        self.assertIn("checkpoint not loaded; refusing to run inference", _AGENT_SRC)

    def test_eval_uses_raw_cnn_feature(self):
        # cnn_feat32_raw 语义：eval 前向必须走 vision_encoder.cnn(depth)
        self.assertIn("self.vision_encoder.cnn(depth)", _AGENT_SRC)

    def test_nav_load_records_selected_checkpoint(self):
        self.assertIn("self.cur_model_name = hit", _AGENT_SRC)
        self.assertIn("self.cur_model_name = ckpt_path", _AGENT_SRC)


if __name__ == "__main__":
    unittest.main()
