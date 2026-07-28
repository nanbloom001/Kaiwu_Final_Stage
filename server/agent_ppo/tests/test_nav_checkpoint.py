#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""nav checkpoint 命名空间与 AlgorithmNavDagger 存取回路测试。

覆盖：首载低层父 / 保存单一 nav 文件（无别名）/ resume / 跨 ID 拒绝 /
visual latest 不受 nav 文件污染 / digest 漂移 warning-only。
"""

import os
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

PARENT_ID = "34728"


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, _message):
        pass

    def warning(self, message):
        self.warnings.append(str(message))


def _mk_actor():
    return nn.Sequential(
        nn.Linear(77, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12),
    )


def _mk_algorithm(logger=None):
    return AlgorithmNavDagger(
        vision_encoder=VisionEncoder(),
        low_level_actor=_mk_actor(),
        high_level=HighLevelPolicy(),
        device="cpu",
        low_level_parent_model_id=PARENT_ID,
        logger=logger,
    )


def _write_parent_bundle(directory, model_id=PARENT_ID):
    """伪造一个 command 标签族的低层父包（visual_ppo 形态的 modules 布局）。"""
    source_ve = VisionEncoder()
    source_actor = _mk_actor()
    bundle = {
        "format": cio.KAIWU_TRAIN_FORMAT,
        "schema_version": cio.KAIWU_TRAIN_SCHEMA_VERSION,
        "stage_type": "standard_visual_ppo",
        "platform_model_id": str(model_id),
        "model_spec": {
            "proprio_dim": 45,
            "scan_dim": 256,
            "latent_dim": 32,
            "action_dim": 12,
            "goal_dim": 0,
        },
        "modules": {
            "vision_encoder": {"state_dict": source_ve.state_dict()},
            "low_level": {"actor_state_dict": source_actor.state_dict()},
        },
        "lineage": {},
    }
    path = os.path.join(directory, f"model.ckpt-commandfull-{model_id}.pkl")
    torch.save(bundle, path)
    return path, source_ve, source_actor


class TestNavCheckpointRoundtrip(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.parent_path, self.src_ve, self.src_actor = _write_parent_bundle(self.dir)

    def test_first_load_save_resume_chain(self):
        algo = _mk_algorithm()

        # 分支 1：首载低层父（高层保持随机初始化，digest 当场计算）
        hit = algo.load_parent_bundle(self.dir, PARENT_ID)
        self.assertEqual(hit, self.parent_path)
        self.assertIsNotNone(algo.low_level_state_digest)
        self.assertFalse(algo.resume_loaded)
        # 低层权重逐位等于父包
        for k, v in self.src_ve.state_dict().items():
            self.assertTrue(torch.equal(algo.vision_encoder.state_dict()[k], v))

        # 分支 2：首次保存 → 有且只有一个 nav* 文件，无别名
        algo.ramp_probability = 0.05
        nav_path = os.path.join(self.dir, "model.ckpt-navbc-777.pkl")
        checksum = algo.save_nav_bundle(
            nav_path, platform_model_id="777", phase_label="navbc"
        )
        self.assertEqual(len(checksum), 64)
        self.assertTrue(cio.validate_probe_filename(nav_path))
        files = sorted(os.listdir(self.dir))
        self.assertNotIn("model.ckpt-locomotion-777.pkl", files)
        self.assertNotIn("model.ckpt-lbc-loco-777.pkl", files)
        self.assertEqual(
            [f for f in files if "777" in f], ["model.ckpt-navbc-777.pkl"]
        )

        # visual latest 不受 nav 文件污染（独立命名空间的核心断言）
        self.assertEqual(cio.visual_latest_model_id(self.dir), int(PARENT_ID))
        self.assertEqual(cio.nav_latest_model_id(self.dir), 777)

        # 跨 ID 拒绝
        self.assertEqual(cio.nav_checkpoint_candidates(self.dir, 999), [])

        # 分支 3：resume（新 algorithm 全量恢复）
        algo2 = _mk_algorithm()
        hit2 = algo2.load_nav_resume(self.dir, "777")
        self.assertEqual(hit2, nav_path)
        self.assertTrue(algo2.resume_loaded)
        self.assertEqual(algo2.low_level_state_digest, algo.low_level_state_digest)
        self.assertAlmostEqual(algo2.ramp_probability, 0.05)
        for k, v in algo.high_level.state_dict().items():
            self.assertTrue(torch.equal(algo2.high_level.state_dict()[k], v))
        # per-env 活状态不恢复
        self.assertIsNone(algo2.scheduler)

    def test_resume_rejects_high_level_contract_tamper(self):
        algo = _mk_algorithm()
        algo.load_parent_bundle(self.dir, PARENT_ID)
        nav_path = os.path.join(self.dir, "model.ckpt-navbc-779.pkl")
        algo.save_nav_bundle(nav_path, platform_model_id="779", phase_label="navbc")
        bundle = torch.load(nav_path, weights_only=False)
        bundle["modules"]["high_level"]["cmd_clamp_max"] = [9.0, 9.0, 9.0]
        torch.save(bundle, nav_path)
        with self.assertRaises(ValueError):
            _mk_algorithm().load_nav_resume(self.dir, "779")

    def test_resume_preserves_checkpoint_parent_lineage(self):
        algo = _mk_algorithm()
        algo.load_parent_bundle(self.dir, PARENT_ID)
        nav_path = os.path.join(self.dir, "model.ckpt-navbc-780.pkl")
        algo.save_nav_bundle(nav_path, platform_model_id="780", phase_label="navbc")

        resumed = AlgorithmNavDagger(
            vision_encoder=VisionEncoder(),
            low_level_actor=_mk_actor(),
            high_level=HighLevelPolicy(),
            device="cpu",
            low_level_parent_model_id="99999",
        )
        resumed.load_nav_resume(self.dir, "780")
        self.assertEqual(resumed.low_level_parent_model_id, PARENT_ID)

    def test_save_before_load_is_forbidden(self):
        algo = _mk_algorithm()
        with self.assertRaises(RuntimeError):
            algo.save_nav_bundle(
                os.path.join(self.dir, "model.ckpt-navbc-1.pkl"),
                platform_model_id="1",
                phase_label="navbc",
            )

    def test_digest_tamper_warns_and_saves_actual_digest(self):
        logger = _Logger()
        algo = _mk_algorithm(logger=logger)
        algo.load_parent_bundle(self.dir, PARENT_ID)
        # 污染冻结低层（模拟事故）
        with torch.no_grad():
            next(algo.low_level_actor.parameters()).add_(1.0)
        path = os.path.join(self.dir, "model.ckpt-navbc-778.pkl")
        algo.save_nav_bundle(
            path,
            platform_model_id="778",
            phase_label="navbc",
        )
        saved = torch.load(path, weights_only=False)
        actual = cio.compute_low_level_state_digest(saved)
        self.assertEqual(saved["lineage"]["low_level_state_digest"], actual)
        self.assertEqual(algo.low_level_state_digest, actual)
        self.assertTrue(any("WARNING-ONLY" in item for item in logger.warnings))

    def test_resume_digest_mismatch_warns_and_uses_loaded_tensors(self):
        algo = _mk_algorithm()
        algo.load_parent_bundle(self.dir, PARENT_ID)
        path = os.path.join(self.dir, "model.ckpt-navbc-781.pkl")
        algo.save_nav_bundle(path, platform_model_id="781", phase_label="navbc")
        bundle = torch.load(path, weights_only=False)
        bundle["lineage"]["low_level_state_digest"] = "0" * 64
        torch.save(bundle, path)

        logger = _Logger()
        resumed = _mk_algorithm(logger=logger)
        self.assertEqual(resumed.load_nav_resume(self.dir, "781"), path)
        self.assertTrue(resumed.resume_loaded)
        self.assertTrue(any("digest" in item for item in logger.warnings))

    def test_training_candidates_branching(self):
        # 首载：id == 父 ID → 只有低层父候选
        first = cio.nav_training_candidates(
            self.dir, PARENT_ID, low_level_parent_model_id=PARENT_ID
        )
        self.assertEqual(first, [self.parent_path])
        # 非父 ID 且无 nav 文件 → 仅同 ID 父候选（此处为空）
        other = cio.nav_training_candidates(
            self.dir, "555", low_level_parent_model_id=PARENT_ID
        )
        self.assertEqual(other, [])

    def test_optimizer_covers_high_level_only(self):
        algo = _mk_algorithm()
        opt_ids = {
            id(p) for g in algo.optimizer.param_groups for p in g["params"]
        }
        hl_ids = {id(p) for p in algo.high_level.parameters()}
        self.assertEqual(opt_ids, hl_ids)
        for p in algo.vision_encoder.parameters():
            self.assertFalse(p.requires_grad)
        for p in algo.low_level_actor.parameters():
            self.assertFalse(p.requires_grad)


if __name__ == "__main__":
    unittest.main()
