#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""nav_contract 契约测试：词表落域、布局算术、golden vectors 回归。"""

import json
import pathlib
import unittest

from agent_ppo.feature import nav_contract as nc

DATA = pathlib.Path(__file__).parent / "data" / "nav_golden_vectors.json"


class TestNavContract(unittest.TestCase):
    def test_vocab_within_command_34728_domain(self):
        nc.validate_vocab_in_domain()  # 越域即 raise

    def test_vocab_shape_and_order(self):
        self.assertEqual(len(nc.VOCAB), 10)
        self.assertEqual(nc.VOCAB_SIZE, 10)
        self.assertEqual(nc.ZERO_TOKEN_INDEX, 0)
        self.assertEqual(nc.VOCAB[0], (0.0, 0.0, 0.0))
        # v1 词表不含横移（vy 全零；lateral 待 N0 单独验收）
        for vx, vy, wz in nc.VOCAB:
            self.assertEqual(vy, 0.0)
        # vx 上锚 0.70（0.8 是 normal_forward 开区间从未采到的边界）
        self.assertEqual(max(v[0] for v in nc.VOCAB), 0.70)
        self.assertEqual(len(nc.TOKEN_NAMES), 10)

    def test_input_layout_arithmetic(self):
        self.assertEqual(nc.NAV_INPUT_DIM, 48)
        slices = [
            nc.CNN_FEAT_SLICE,
            nc.GOAL4_SLICE,
            nc.EXEC_CMD_SLICE,
            nc.HELD_CMD_SLICE,
            nc.ANG_VEL_SLICE,
            nc.PROJ_GRAV_SLICE,
        ]
        # 依次相接、覆盖 [0, 48)
        cursor = 0
        for lo, hi in slices:
            self.assertEqual(lo, cursor)
            cursor = hi
        self.assertEqual(cursor, nc.NAV_INPUT_DIM)

    def test_obs_layout_arithmetic(self):
        self.assertEqual(
            nc.POLICY_OBS_DIM,
            nc.POLICY_PROPRIO_DIM + nc.SCAN_DIM + 4 + nc.DEPTH_DIM,
        )
        self.assertEqual(nc.POLICY_OBS_DIM, 57905)
        self.assertEqual(nc.GOAL4_OBS_START, 301)
        self.assertEqual(nc.DEPTH_OBS_START, 305)
        self.assertEqual(nc.CRITIC_OBS_DIM, 319)
        self.assertEqual(nc.CRITIC_GOAL3_START, 316)
        self.assertEqual(nc.POLICY_CMD_SLICE, (6, 9))
        self.assertEqual(nc.CRITIC_CMD_SLICE, (9, 12))

    def test_timing_constants(self):
        self.assertEqual(nc.NAV_PERIOD_FRAMES, 10)
        self.assertEqual(nc.TBPTT_T, 16)
        self.assertEqual(nc.MIN_DWELL_TICKS, 10)
        self.assertAlmostEqual(nc.NAV_TICK_HZ, 5.0)

    def test_golden_vectors_match_committed_file(self):
        regenerated = nc.generate_golden_vectors(400)
        with DATA.open("r", encoding="utf-8") as stream:
            committed = json.load(stream)
        self.assertEqual(regenerated["contract"], committed["contract"])
        self.assertEqual(
            [list(item) for item in regenerated["token_script"]],
            [list(item) for item in committed["token_script"]],
        )
        self.assertEqual(regenerated["frames"], committed["frames"])

    def test_golden_vectors_one_frame_delay_and_zero_bypass(self):
        frames = nc.generate_golden_vectors(400)["frames"]
        # frame 0 exec 为零（tick0 的决策 frame1 才生效 —— 一帧确定性延迟）
        self.assertEqual(frames[0]["exec_cmd"], [0.0, 0.0, 0.0])
        self.assertEqual(frames[1]["held_token"], 3)
        # zero token 急停旁路：tick20 决策后下一帧 exec 立即为零
        self.assertEqual(frames[201]["held_token"], 0)
        self.assertEqual(frames[201]["exec_cmd"], [0.0, 0.0, 0.0])
        # 驻留压制：tick22 的请求被压制（held 保持 0 直到 tick30 生效）
        self.assertEqual(frames[300]["held_token"], 0)
        self.assertEqual(frames[301]["held_token"], 2)


if __name__ == "__main__":
    unittest.main()
