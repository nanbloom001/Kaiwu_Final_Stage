#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""NavTickBuffer 与 HighLevelPolicy TBPTT 行为测试。

关键断言：done 边界经 reset_mask 逐步清零 hidden——前后 episode 不得拼成
同一序列（重放中 reset 之后的输出必须等于"从零 hidden 重新开始"的输出）。
"""

import unittest

import torch

from agent_ppo.feature.nav_tick_buffer import NavTickBuffer
from agent_ppo.model.high_level_policy import HighLevelPolicy


class TestNavTickBuffer(unittest.TestCase):
    def _mk(self, B=3, T=4):
        return NavTickBuffer(B, "cpu", seq_len=T)

    def test_lifecycle_guards(self):
        buf = self._mk()
        with self.assertRaises(RuntimeError):
            buf.add(
                torch.zeros(3, 48),
                torch.zeros(3, dtype=torch.long),
                torch.zeros(3, dtype=torch.bool),
                torch.ones(3, dtype=torch.bool),
                torch.ones(3, 10, dtype=torch.bool),
            )
        buf.start_segment(None)
        with self.assertRaises(RuntimeError):
            buf.get()
        for _ in range(4):
            full = buf.add(
                torch.zeros(3, 48),
                torch.zeros(3, dtype=torch.long),
                torch.zeros(3, dtype=torch.bool),
                torch.ones(3, dtype=torch.bool),
                torch.ones(3, 10, dtype=torch.bool),
            )
        self.assertTrue(full)
        with self.assertRaises(RuntimeError):
            buf.add(
                torch.zeros(3, 48),
                torch.zeros(3, dtype=torch.long),
                torch.zeros(3, dtype=torch.bool),
                torch.ones(3, dtype=torch.bool),
                torch.ones(3, 10, dtype=torch.bool),
            )
        seq = buf.get()
        self.assertEqual(seq["inputs"].shape, (4, 3, 48))
        buf.clear()
        buf.start_segment(None)  # clear 后可开新段

    def test_initial_hidden_snapshot_detached(self):
        buf = self._mk()
        h = torch.randn(2, 3, 64, requires_grad=True)
        c = torch.randn(2, 3, 64, requires_grad=True)
        buf.start_segment((h, c))
        for _ in range(4):
            buf.add(
                torch.zeros(3, 48),
                torch.zeros(3, dtype=torch.long),
                torch.zeros(3, dtype=torch.bool),
                torch.ones(3, dtype=torch.bool),
                torch.ones(3, 10, dtype=torch.bool),
            )
        h0, c0 = buf.get()["initial_hidden"]
        self.assertFalse(h0.requires_grad)
        self.assertTrue(torch.equal(h0, h.detach()))
        self.assertTrue(torch.equal(c0, c.detach()))


class TestForwardSequenceResetBoundary(unittest.TestCase):
    def test_reset_mask_prevents_episode_stitching(self):
        torch.manual_seed(7)
        pol = HighLevelPolicy()
        pol.eval()
        T, B = 6, 2
        inputs = torch.randn(T, B, 48)
        h0 = torch.randn(2, B, 64)
        c0 = torch.randn(2, B, 64)

        # env0 在 t=3 处 reset；env1 从不 reset
        reset_masks = torch.zeros(T, B, dtype=torch.bool)
        reset_masks[3, 0] = True

        with torch.no_grad():
            logits = pol.forward_sequence(inputs, (h0, c0), reset_masks)
            # 参照：env0 的 t>=3 段以零 hidden 重新开始
            ref = pol.forward_sequence(
                inputs[3:, 0:1],
                (torch.zeros(2, 1, 64), torch.zeros(2, 1, 64)),
                torch.zeros(T - 3, 1, dtype=torch.bool),
            )
        self.assertTrue(torch.allclose(logits[3:, 0:1], ref, atol=1e-6))
        # env1 不受影响：与无 reset 的整段重放一致
        with torch.no_grad():
            ref_full = pol.forward_sequence(
                inputs[:, 1:2],
                (h0[:, 1:2], c0[:, 1:2]),
                torch.zeros(T, 1, dtype=torch.bool),
            )
        self.assertTrue(torch.allclose(logits[:, 1:2], ref_full, atol=1e-6))

    def test_sequence_gradients_flow_to_head_and_lstm(self):
        pol = HighLevelPolicy()
        T, B = 4, 2
        seqbuf = NavTickBuffer(B, "cpu", seq_len=T)
        seqbuf.start_segment(None)
        for _ in range(T):
            seqbuf.add(
                torch.randn(B, 48),
                torch.randint(0, 10, (B,)),
                torch.zeros(B, dtype=torch.bool),
                torch.ones(B, dtype=torch.bool),
                torch.ones(B, 10, dtype=torch.bool),
            )
        seq = seqbuf.get()
        logits = pol.forward_sequence(
            seq["inputs"], seq["initial_hidden"], seq["reset_masks"], seq["dwell_masks"]
        )
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(T * B, 10), seq["tokens"].reshape(T * B)
        )
        loss.backward()
        self.assertIsNotNone(pol.head.weight.grad)
        self.assertIsNotNone(pol.rnn.weight_ih_l0.grad)
        self.assertTrue(bool((pol.rnn.weight_ih_l0.grad != 0).any()))

    def test_rollout_forward_dwell_mask(self):
        pol = HighLevelPolicy()
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[:, 4] = False
        with torch.no_grad():
            logits = pol(torch.randn(2, 48), dwell_mask=mask)
        self.assertTrue(bool((logits[:, 4] < -1e8).all()))
        self.assertTrue(bool(torch.isfinite(logits[:, :4]).all()))


if __name__ == "__main__":
    unittest.main()
