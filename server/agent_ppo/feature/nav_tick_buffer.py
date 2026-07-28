#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""NavTickBuffer — 高层 LSTM TBPTT 行为克隆的序列缓存。

只在 5Hz nav tick 收集；凑满 T（默认 nav_contract.TBPTT_T=16）触发一次
序列交叉熵更新。禁止 LBC 式逐步即时反向——那无法训练高层 LSTM 的跨时刻
记忆（路线纪律：保留 LSTM 就必须 TBPTT；否则唯一合法降级是 MLP）。

存储（T 满时由 ``get()`` 一次取出）：
  - inputs      [T, B, 48]   nav tick 时的 48 维契约输入
  - tokens      [T, B]       Oracle 标签（long）
  - reset_masks [T, B]       该 tick 与前一 tick 之间是否发生过 env reset
                             （TBPTT 重放按它逐步清零 hidden——done 后前后
                              episode 不得拼成同一序列）
  - valid_masks [T, B]       非有限输入 / 无效 Oracle 标签 → False（CE 加权 0）
  - dwell_masks [T, B, V]    rollout 时实际施加的驻留 logits 掩码（落库不重算）
  - (h0, c0)    [L, B, H]    段首 rollout hidden 快照（detach 存储）
"""

from __future__ import annotations

from typing import Tuple

import torch

from agent_ppo.feature import nav_contract


class NavTickBuffer:
    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        seq_len: int = nav_contract.TBPTT_T,
        input_dim: int = nav_contract.NAV_INPUT_DIM,
        vocab_size: int = nav_contract.VOCAB_SIZE,
        rnn_num_layers: int = 2,
        rnn_hidden_dim: int = 64,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.seq_len = seq_len

        self.inputs = torch.zeros(seq_len, num_envs, input_dim, device=self.device)
        self.tokens = torch.zeros(seq_len, num_envs, dtype=torch.long, device=self.device)
        self.reset_masks = torch.zeros(seq_len, num_envs, dtype=torch.bool, device=self.device)
        self.valid_masks = torch.zeros(seq_len, num_envs, dtype=torch.bool, device=self.device)
        self.dwell_masks = torch.zeros(
            seq_len, num_envs, vocab_size, dtype=torch.bool, device=self.device
        )
        self._h0 = torch.zeros(rnn_num_layers, num_envs, rnn_hidden_dim, device=self.device)
        self._c0 = torch.zeros_like(self._h0)
        self._ptr = 0
        self._segment_open = False

    # ------------------------------------------------------------------

    @property
    def is_full(self) -> bool:
        return self._ptr >= self.seq_len

    @property
    def size(self) -> int:
        return self._ptr

    def start_segment(self, hidden: Tuple[torch.Tensor, torch.Tensor] | None) -> None:
        """段首快照当前 rollout hidden（detach clone）；hidden 为 None 时置零。"""
        if self._segment_open:
            raise RuntimeError("NavTickBuffer.start_segment called on an open segment")
        if hidden is None:
            self._h0.zero_()
            self._c0.zero_()
        else:
            h, c = hidden
            self._h0.copy_(h.detach())
            self._c0.copy_(c.detach())
        self._ptr = 0
        self._segment_open = True

    def add(
        self,
        inputs: torch.Tensor,
        oracle_tokens: torch.Tensor,
        reset_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        dwell_mask: torch.Tensor,
    ) -> bool:
        """收集一个 nav tick；返回是否已满（满后须 get()+clear() 再收集）。"""
        if not self._segment_open:
            raise RuntimeError("NavTickBuffer.add called before start_segment")
        if self.is_full:
            raise RuntimeError("NavTickBuffer.add called on a full buffer")
        t = self._ptr
        self.inputs[t].copy_(inputs.detach())
        self.tokens[t].copy_(oracle_tokens.detach())
        self.reset_masks[t].copy_(reset_mask.detach().bool())
        self.valid_masks[t].copy_(valid_mask.detach().bool())
        self.dwell_masks[t].copy_(dwell_mask.detach().bool())
        self._ptr += 1
        return self.is_full

    def get(self) -> dict:
        if not self.is_full:
            raise RuntimeError(
                f"NavTickBuffer.get called with {self._ptr}/{self.seq_len} ticks collected"
            )
        return {
            "inputs": self.inputs,
            "tokens": self.tokens,
            "reset_masks": self.reset_masks,
            "valid_masks": self.valid_masks,
            "dwell_masks": self.dwell_masks,
            "initial_hidden": (self._h0, self._c0),
        }

    def clear(self) -> None:
        self._ptr = 0
        self._segment_open = False
