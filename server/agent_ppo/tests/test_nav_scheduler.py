#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""NavScheduler 测试：golden vectors 逐帧比对、注入偏移、驻留、reset、有限性。"""

import unittest

import torch

from agent_ppo.feature import nav_contract as nc
from agent_ppo.feature.nav_scheduler import NavScheduler


class TestNavSchedulerGoldenParity(unittest.TestCase):
    def test_frame_exact_against_golden_vectors(self):
        gv = nc.generate_golden_vectors(400)
        script = dict(gv["token_script"])
        sched = NavScheduler(num_envs=1, device="cpu")
        for frame in gv["frames"]:
            t = frame["frame"]
            exec_now = sched.exec_cmd[0].tolist()
            self.assertEqual(int(sched.held_token[0].item()), frame["held_token"], f"frame {t}")
            for got, want in zip(exec_now, frame["exec_cmd"]):
                self.assertAlmostEqual(got, want, places=5, msg=f"frame {t}")
            if t % nc.NAV_PERIOD_FRAMES == 0:
                tick = t // nc.NAV_PERIOD_FRAMES
                requested = script.get(tick, int(sched.held_token[0].item()))
                sched.request_tokens(torch.tensor([requested]))
            sched.step_exec()


class TestNavSchedulerSemantics(unittest.TestCase):
    def setUp(self):
        self.sched = NavScheduler(num_envs=3, device="cpu")

    def test_inject_offsets(self):
        self.sched.exec_cmd = torch.tensor(
            [[0.1, 0.0, 0.2]] * 3, dtype=torch.float32
        )
        obs = torch.zeros(3, nc.POLICY_OBS_DIM)
        critic = torch.zeros(3, nc.CRITIC_OBS_DIM)
        self.sched.inject(obs, critic)
        # policy [6:9] 与 critic [9:12] 拿到同一个 exec_cmd
        self.assertTrue(torch.equal(obs[:, 6:9], self.sched.exec_cmd))
        self.assertTrue(torch.equal(critic[:, 9:12], self.sched.exec_cmd))
        self.assertEqual(float(obs[:, :6].abs().sum()), 0.0)
        self.assertEqual(float(obs[:, 9:].abs().sum()), 0.0)

    def test_inject_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            self.sched.inject(torch.zeros(3, 8))
        with self.assertRaises(ValueError):
            self.sched.inject(
                torch.zeros(3, nc.POLICY_OBS_DIM), torch.zeros(3, 11)
            )

    def test_dwell_mask_blocks_switch_until_satisfied(self):
        # 切到 token3 后驻留清零 → 下一 tick 只允许保持或 zero
        self.sched.request_tokens(torch.tensor([3, 3, 3]))
        mask = self.sched.dwell_mask()
        self.assertTrue(bool(mask[:, 3].all()))       # 保持允许
        self.assertTrue(bool(mask[:, nc.ZERO_TOKEN_INDEX].all()))  # 急停允许
        self.assertFalse(bool(mask[:, 5].any()))      # 其他切换禁止
        # 驻留满后全放开
        for _ in range(nc.MIN_DWELL_TICKS):
            self.sched.request_tokens(torch.tensor([3, 3, 3]))
        self.assertTrue(bool(self.sched.dwell_mask().all()))

    def test_zero_token_overrides_dwell(self):
        self.sched.request_tokens(torch.tensor([3, 3, 3]))
        effective = self.sched.request_tokens(torch.tensor([0, 3, 3]))
        self.assertEqual(int(effective[0].item()), 0)
        self.assertEqual(int(effective[1].item()), 3)

    def test_reset_clears_all_state(self):
        self.sched.request_tokens(torch.tensor([3, 4, 5]))
        for _ in range(30):
            self.sched.step_exec()
        self.sched.reset(torch.tensor([0, 2]))
        self.assertEqual(int(self.sched.held_token[0].item()), nc.ZERO_TOKEN_INDEX)
        self.assertEqual(float(self.sched.exec_cmd[0].abs().sum()), 0.0)
        self.assertEqual(int(self.sched.dwell_ticks[0].item()), nc.MIN_DWELL_TICKS)
        # 未 reset 的 env 保持
        self.assertEqual(int(self.sched.held_token[1].item()), 4)

    def test_nonfinite_exec_falls_back_to_zero(self):
        self.sched.request_tokens(torch.tensor([3, 3, 3]))
        self.sched.exec_cmd[0, 0] = float("nan")
        stepped = self.sched.step_exec()
        self.assertTrue(bool(torch.isfinite(stepped).all()))
        self.assertEqual(float(stepped[0].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
