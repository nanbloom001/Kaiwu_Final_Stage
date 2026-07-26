#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""五视角审核后修复的回归测试。

覆盖：Oracle 标签投影过驻留纪律（CRITICAL）、非有限 tick 的 NaN 中和与
容错继续（CRITICAL）、损坏 resume 硬停（MAJOR）、全无效段跳过更新、
visual 诊断函数不受 nav 文件污染（逐字节回归）。
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

N = 2


def _mk_actor():
    return nn.Sequential(
        nn.Linear(77, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12),
    )


def _mk_algorithm():
    torch.manual_seed(3)
    return AlgorithmNavDagger(
        vision_encoder=VisionEncoder(),
        low_level_actor=_mk_actor(),
        high_level=HighLevelPolicy(),
        device="cpu",
        low_level_parent_model_id="34728",
    )


def _obs(nan_goal=False):
    obs = torch.zeros(N, nc.POLICY_OBS_DIM)
    if nan_goal:
        obs[:, nc.GOAL4_OBS_START] = float("nan")
    return obs


def _critic(local_x=5.0, local_y=0.0, dist_m=5.0):
    critic = torch.zeros(N, nc.CRITIC_OBS_DIM)
    critic[:, nc.CRITIC_GOAL3_START + 0] = local_x / nc.GOAL_XY_SCALE_M
    critic[:, nc.CRITIC_GOAL3_START + 1] = local_y / nc.GOAL_XY_SCALE_M
    critic[:, nc.CRITIC_GOAL3_START + 2] = dist_m / nc.GOAL_DIST_SCALE_M
    return critic


class TestOracleLabelProjection(unittest.TestCase):
    def test_blocked_oracle_label_projects_to_held(self):
        algo = _mk_algorithm()
        algo.ramp_probability = 0.0  # 纯 Oracle 驱动

        # tick 0：直行远目标 → Oracle=3（forward_fast），切换生效 dwell 清零
        result = algo.frame_begin(_obs(), _critic(local_x=5.0, dist_m=5.0))
        self.assertTrue(result["is_tick"])
        self.assertEqual(int(algo.scheduler.held_token[0].item()), 3)
        self.assertEqual(int(algo.tick_buffer.tokens[0, 0].item()), 3)
        for _ in range(nc.NAV_PERIOD_FRAMES - 1):
            algo.frame_end(torch.zeros(N, dtype=torch.bool))
            algo.frame_begin(_obs(), _critic(local_x=5.0, dist_m=5.0))
        algo.frame_end(torch.zeros(N, dtype=torch.bool))

        # tick 1（驻留未满）：目标切到大角度侧向 → Oracle 原始请求 spin_left(8)
        # 被驻留掩码屏蔽 → 落库标签必须投影为保持 held(3)，绝不能是 8
        algo.frame_begin(_obs(), _critic(local_x=0.1, local_y=2.0, dist_m=2.0))
        label = int(algo.tick_buffer.tokens[1, 0].item())
        self.assertEqual(label, 3)
        # held 也未被切换（scheduler 同样执行驻留纪律）
        self.assertEqual(int(algo.scheduler.held_token[0].item()), 3)
        # zero 急停例外仍然直通：再过一 tick 请求 zero
        for _ in range(nc.NAV_PERIOD_FRAMES - 1):
            algo.frame_end(torch.zeros(N, dtype=torch.bool))
            algo.frame_begin(_obs(), _critic(local_x=0.1, local_y=2.0, dist_m=2.0))
        algo.frame_end(torch.zeros(N, dtype=torch.bool))
        algo.frame_begin(_obs(), _critic(local_x=0.1, dist_m=0.2))  # 到达 → zero
        self.assertEqual(int(algo.tick_buffer.tokens[2, 0].item()), 0)
        self.assertEqual(int(algo.scheduler.held_token[0].item()), 0)


class TestNaNTickNeutralization(unittest.TestCase):
    def test_one_nan_tick_does_not_crash_the_segment(self):
        algo = _mk_algorithm()
        algo.ramp_probability = 0.0
        nan_tick = 4
        for frame in range(nc.TBPTT_T * nc.NAV_PERIOD_FRAMES):
            tick_index = frame // nc.NAV_PERIOD_FRAMES
            is_tick_frame = frame % nc.NAV_PERIOD_FRAMES == 0
            obs = _obs(nan_goal=(is_tick_frame and tick_index == nan_tick))
            result = algo.frame_begin(obs, _critic())
            if is_tick_frame and tick_index == nan_tick:
                self.assertGreater(result["tick_metrics"]["nonfinite_fallback"], 0)
            algo.frame_end(torch.zeros(N, dtype=torch.bool))
        self.assertTrue(algo.tick_buffer.is_full)
        # 落库输入必须已消毒（无 NaN），NaN tick 的 valid=0
        self.assertTrue(bool(torch.isfinite(algo.tick_buffer.inputs).all()))
        self.assertFalse(bool(algo.tick_buffer.valid_masks[nan_tick].any()))
        # 序列更新不崩、loss 有限
        metrics = algo.finish_nav_sequence_update()
        self.assertTrue(metrics["ce_loss"] == metrics["ce_loss"])  # not NaN
        self.assertLess(metrics["ce_loss"], 1.0e6)  # 未被掩码标签的 1e9 级 CE 污染
        self.assertGreater(algo.nonfinite_fallback_count, 0)

    def test_all_invalid_segment_skips_update(self):
        algo = _mk_algorithm()
        algo.frame_begin(_obs(), _critic())  # 触发 per-env 构建
        algo.frame_end(torch.zeros(N, dtype=torch.bool))
        algo.tick_buffer.clear()
        algo.tick_buffer.start_segment(None)
        for _ in range(nc.TBPTT_T):
            algo.tick_buffer.add(
                torch.zeros(N, 48),
                torch.zeros(N, dtype=torch.long),
                torch.zeros(N, dtype=torch.bool),
                torch.zeros(N, dtype=torch.bool),  # 全无效
                torch.ones(N, 10, dtype=torch.bool),
            )
        before = {k: v.clone() for k, v in algo.high_level.state_dict().items()}
        metrics = algo.finish_nav_sequence_update()
        self.assertEqual(metrics.get("update_skipped_no_valid"), 1.0)
        for k, v in algo.high_level.state_dict().items():
            self.assertTrue(torch.equal(before[k], v))  # 参数纹丝不动


class TestResumeHardStops(unittest.TestCase):
    def test_corrupted_resume_candidate_raises(self):
        directory = tempfile.mkdtemp()
        torch.save({"garbage": True}, os.path.join(directory, "model.ckpt-navfull-321.pkl"))
        algo = _mk_algorithm()
        with self.assertRaises(ValueError):
            algo.load_nav_resume(directory, "321")

    def test_wrong_stage_type_resume_raises(self):
        directory = tempfile.mkdtemp()
        torch.save(
            {
                "format": cio.KAIWU_TRAIN_FORMAT,
                "schema_version": cio.KAIWU_TRAIN_SCHEMA_VERSION,
                "stage_type": "standard_visual_ppo",
            },
            os.path.join(directory, "model.ckpt-navfull-322.pkl"),
        )
        algo = _mk_algorithm()
        with self.assertRaises(ValueError):
            algo.load_nav_resume(directory, "322")


class TestVisualDiagnosticsUnpolluted(unittest.TestCase):
    def test_visual_diagnostics_byte_equal_with_and_without_nav_files(self):
        directory = tempfile.mkdtemp()
        open(os.path.join(directory, "model.ckpt-commandfull-34728.pkl"), "wb").close()
        before_diag = cio.visual_eval_checkpoint_diagnostics(directory, "34728")
        before_latest = cio.visual_latest_model_id(directory)
        for name in ("model.ckpt-navbc-900.pkl", "model.ckpt-navfull-901.pkl"):
            open(os.path.join(directory, name), "wb").close()
        after_diag = cio.visual_eval_checkpoint_diagnostics(directory, "34728")
        after_latest = cio.visual_latest_model_id(directory)
        self.assertEqual(before_diag, after_diag)
        self.assertEqual(before_latest, after_latest)


if __name__ == "__main__":
    unittest.main()
