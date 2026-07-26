#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""NavScheduler — hold / slew / dwell 的唯一实现（per-env 向量化）。

训练侧（aisrv workflow 循环）与部署侧（C++ 镜像）共同遵守的指令交接状态机。
worker 不持有任何 nav 状态；learner 只消费落库张量。C++ 侧不是第二实现，
而是本状态机的契约镜像——用 `nav_contract.generate_golden_vectors` 逐帧比对。

冻结逐帧时序中的角色（frame t）：
    1. ``inject(obs, critic_obs)``   —— 当帧 exec_cmd 写入观测副本
    2. （低层前向，调用方负责）
    3. （nav tick 时高层前向，调用方负责）→ ``request_tokens(tokens)``
    4. —— 新目标自下一帧生效：
    5. （env.step 之后）``step_exec()`` —— slew 演化一帧；done 环境先 ``reset``
"""

from __future__ import annotations

import torch

from agent_ppo.feature import nav_contract


class NavScheduler:
    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = num_envs
        self.device = torch.device(device)

        vocab = torch.tensor(nav_contract.VOCAB, dtype=torch.float32, device=self.device)
        self._vocab = vocab  # [V, 3]
        clamp_min = torch.tensor(nav_contract.CMD_CLAMP_MIN, device=self.device)
        clamp_max = torch.tensor(nav_contract.CMD_CLAMP_MAX, device=self.device)
        # 词表值先过训练域白名单 clamp（词表本身在域内，此处是防御性双保险）
        self._vocab_cmd = torch.clamp(vocab, clamp_min, clamp_max)

        self._rate_up = torch.tensor(nav_contract.SLEW_RATE_UP, device=self.device)
        self._rate_down = torch.tensor(nav_contract.SLEW_RATE_DOWN, device=self.device)
        self._dt = nav_contract.FRAME_DT_S

        self.held_token = torch.full(
            (num_envs,), nav_contract.ZERO_TOKEN_INDEX, dtype=torch.long, device=self.device
        )
        self.exec_cmd = torch.zeros(num_envs, 3, device=self.device)
        # 起始视为驻留已满（episode 开局即可自由选择）
        self.dwell_ticks = torch.full(
            (num_envs,), nav_contract.MIN_DWELL_TICKS, dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------
    # 状态访问
    # ------------------------------------------------------------------

    @property
    def held_cmd(self) -> torch.Tensor:
        """当前词表目标值 [N, 3]。"""
        return self._vocab_cmd[self.held_token]

    def reset(self, env_ids: torch.Tensor) -> None:
        """done 环境全清零：held=zero、exec=0、驻留视为已满。

        与部署 episode 起始语义对齐（enter() 清零）。
        """
        if env_ids.numel() == 0:
            return
        self.held_token[env_ids] = nav_contract.ZERO_TOKEN_INDEX
        self.exec_cmd[env_ids] = 0.0
        self.dwell_ticks[env_ids] = nav_contract.MIN_DWELL_TICKS

    # ------------------------------------------------------------------
    # 步骤 1：观测注入（当帧 exec_cmd，policy [6:9] / critic [9:12] 同一个值）
    # ------------------------------------------------------------------

    def inject(self, obs: torch.Tensor, critic_obs: torch.Tensor = None) -> None:
        p_lo, p_hi = nav_contract.POLICY_CMD_SLICE
        if obs.ndim != 2 or obs.shape[0] != self.num_envs or obs.shape[1] < p_hi:
            raise ValueError(
                f"NavScheduler.inject: bad policy obs shape {tuple(obs.shape)}"
            )
        obs[:, p_lo:p_hi] = self.exec_cmd
        if critic_obs is not None:
            c_lo, c_hi = nav_contract.CRITIC_CMD_SLICE
            if (
                critic_obs.ndim != 2
                or critic_obs.shape[0] != self.num_envs
                or critic_obs.shape[1] < c_hi
            ):
                raise ValueError(
                    f"NavScheduler.inject: bad critic obs shape {tuple(critic_obs.shape)}"
                )
            critic_obs[:, c_lo:c_hi] = self.exec_cmd

    # ------------------------------------------------------------------
    # nav tick：驻留掩码与 token 交接
    # ------------------------------------------------------------------

    def dwell_mask(self) -> torch.Tensor:
        """[N, V] bool，True=本 tick 允许选择该 token。

        驻留未满时只允许：保持当前 token，或切到 zero（急停例外）。
        """
        mask = torch.zeros(
            self.num_envs, nav_contract.VOCAB_SIZE, dtype=torch.bool, device=self.device
        )
        satisfied = self.dwell_ticks >= nav_contract.MIN_DWELL_TICKS
        mask[satisfied] = True
        mask[torch.arange(self.num_envs, device=self.device), self.held_token] = True
        mask[:, nav_contract.ZERO_TOKEN_INDEX] = True
        return mask

    def request_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """nav tick 交接：施加驻留纪律后更新 held token；返回实际生效 token。

        新目标不影响当帧 exec（一帧确定性延迟由调用顺序保证：inject 已在
        本帧完成，slew 到下一帧的 ``step_exec`` 才朝新目标演化）。
        """
        tokens = tokens.to(device=self.device, dtype=torch.long)
        if tokens.shape != (self.num_envs,):
            raise ValueError(f"request_tokens expects [{self.num_envs}], got {tuple(tokens.shape)}")

        is_switch = tokens != self.held_token
        allowed = (
            (self.dwell_ticks >= nav_contract.MIN_DWELL_TICKS)
            | (tokens == nav_contract.ZERO_TOKEN_INDEX)
        )
        apply = is_switch & allowed
        self.held_token = torch.where(apply, tokens, self.held_token)
        self.dwell_ticks = torch.where(
            apply, torch.zeros_like(self.dwell_ticks), self.dwell_ticks
        )
        self.dwell_ticks += 1
        return self.held_token

    # ------------------------------------------------------------------
    # 步骤 5 之后：slew 演化一帧（下一帧的 exec_cmd）
    # ------------------------------------------------------------------

    def step_exec(self) -> torch.Tensor:
        target = self.held_cmd  # [N, 3]
        current = self.exec_cmd

        delta = target - current
        # 加速（|cmd| 增大且不换向）用保守速率；减速/过零用快速率
        same_sign = (current == 0.0) | ((current > 0) == (target > 0))
        moving_away = (target.abs() > current.abs()) & same_sign
        rate = torch.where(
            moving_away, self._rate_up.expand_as(current), self._rate_down.expand_as(current)
        )
        step = rate * self._dt
        stepped = current + torch.clamp(delta, -step, step)

        if nav_contract.ZERO_TOKEN_BYPASSES_SLEW:
            zero_envs = self.held_token == nav_contract.ZERO_TOKEN_INDEX
            stepped[zero_envs] = 0.0

        # 有限性防线：任何非有限值立即回落到零指令（域内最安全形态）
        finite = torch.isfinite(stepped).all(dim=-1)
        if not bool(finite.all()):
            stepped[~finite] = 0.0

        self.exec_cmd = stepped
        return self.exec_cmd
