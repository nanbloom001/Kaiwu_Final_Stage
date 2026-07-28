#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""HighLevelPolicy — hier-nav 高层导航网络（可训练的唯一模块）。

结构（接口契约 v1，`feature/nav_contract.py`）：

    48 维输入（cnn_feat32_raw ⊕ goal4 ⊕ exec_cmd3 ⊕ held_cmd3 ⊕ ang_vel3 ⊕ proj_grav3）
      → LSTM(2 层, hidden=64) → Linear(64 → 10) → 驻留掩码 → categorical logits

隐状态管理接口对齐 `model/vision_encoder.py::VisionEncoder` 的惯例
（reset_hidden_state / reset_hidden_state_for_envs / get / set）。

两种前向：
  - ``forward``：rollout 单 tick 前向（推进内部 hidden，detach）。
  - ``forward_sequence``：TBPTT 重放（外部提供 (h0,c0) 与逐步 reset_mask，
    不触碰内部 rollout hidden；done 边界按 reset_mask 逐步清零，前后
    episode 不会拼成同一序列）。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from agent_ppo.feature import nav_contract

_MASKED_LOGIT = -1.0e9


class HighLevelPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int = nav_contract.NAV_INPUT_DIM,
        vocab_size: int = nav_contract.VOCAB_SIZE,
        rnn_hidden_dim: int = nav_contract.NAV_LSTM_HIDDEN_SIZE,
        rnn_num_layers: int = nav_contract.NAV_LSTM_NUM_LAYERS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(rnn_hidden_dim, vocab_size)
        self._hidden_state = None

    # ------------------------------------------------------------------
    # rollout hidden 管理（惯例对齐 VisionEncoder）
    # ------------------------------------------------------------------

    def reset_hidden_state(self, batch_size=None, device=None):
        if batch_size is not None and device is not None:
            self._hidden_state = (
                torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device),
                torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device),
            )
        else:
            self._hidden_state = None

    def reset_hidden_state_for_envs(self, env_ids: torch.Tensor):
        if self._hidden_state is None:
            return
        h, c = self._hidden_state
        h[:, env_ids, :] = 0
        c[:, env_ids, :] = 0

    def get_hidden_state(self):
        return self._hidden_state

    def set_hidden_state(self, hidden_state: Tuple[torch.Tensor, torch.Tensor]):
        self._hidden_state = hidden_state

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_dwell_mask(logits: torch.Tensor, dwell_mask: torch.Tensor) -> torch.Tensor:
        """dwell_mask: [B, V] bool，True=允许选择。禁止项置大负值。"""
        if dwell_mask is None:
            return logits
        return logits.masked_fill(~dwell_mask.bool(), _MASKED_LOGIT)

    def forward(
        self,
        inputs: torch.Tensor,
        dwell_mask: torch.Tensor = None,
        detach_hidden: bool = True,
    ) -> torch.Tensor:
        """Rollout 单 tick 前向。inputs: [B, 48] → masked logits [B, V]。"""

        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"HighLevelPolicy expects [B, {self.input_dim}] inputs, got {tuple(inputs.shape)}"
            )
        batch_size = inputs.shape[0]
        device = inputs.device

        if self._hidden_state is None:
            self.reset_hidden_state(batch_size, device)

        rnn_out, self._hidden_state = self.rnn(inputs.unsqueeze(1), self._hidden_state)
        if detach_hidden:
            self._hidden_state = (
                self._hidden_state[0].detach(),
                self._hidden_state[1].detach(),
            )
        logits = self.head(rnn_out.squeeze(1))
        return self._apply_dwell_mask(logits, dwell_mask)

    def forward_sequence(
        self,
        inputs: torch.Tensor,
        initial_hidden: Tuple[torch.Tensor, torch.Tensor],
        reset_masks: torch.Tensor,
        dwell_masks: torch.Tensor = None,
    ) -> torch.Tensor:
        """TBPTT 重放。不触碰内部 rollout hidden。

        Args:
            inputs:         [T, B, 48]
            initial_hidden: (h0, c0)，各 [num_layers, B, hidden]
            reset_masks:    [T, B] bool，True = 该 tick 与前一 tick 之间发生过
                            env reset —— 前向该步之前先清零对应 env 的 hidden。
            dwell_masks:    [T, B, V] bool（rollout 落库的驻留掩码），可为 None。

        Returns:
            logits [T, B, V]（已施加 dwell 掩码）。
        """

        T, batch_size, _ = inputs.shape
        h, c = initial_hidden
        h = h.clone()
        c = c.clone()
        outputs = []
        for t in range(T):
            keep = (~reset_masks[t].bool()).reshape(1, batch_size, 1).to(h.dtype)
            h = h * keep
            c = c * keep
            rnn_out, (h, c) = self.rnn(inputs[t].unsqueeze(1), (h, c))
            logits = self.head(rnn_out.squeeze(1))
            if dwell_masks is not None:
                logits = self._apply_dwell_mask(logits, dwell_masks[t])
            outputs.append(logits)
        return torch.stack(outputs, dim=0)
