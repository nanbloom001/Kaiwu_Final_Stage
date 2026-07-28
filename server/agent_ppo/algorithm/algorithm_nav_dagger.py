#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""AlgorithmNavDagger — 高层导航 DAgger（冻结低层，TBPTT 序列行为克隆）。

结构仿 `algorithm_lbc.py` 的三段式接口，但有三处刻意差异（计划修订四/九）：

  1. **TBPTT 序列更新替代逐步即时反向**：只在 5Hz nav tick 收集
     （NavTickBuffer），凑满 T=16 才做一次序列交叉熵更新——逐步 BC 无法
     训练高层 LSTM 的跨时刻记忆。禁止"保留 LSTM 却单帧 BC"。
  2. **不移植 action-MSE safety takeover**：词表只保证低层 command 不越域，
     不保证楼梯上错误转向不会摔倒——安全监控改为分类指标（CE / top1 /
     disagreement / token entropy / switch rate + 外部 hard termination），
     由 workflow 的 soft-stay 用它们冻结 ramp。
  3. **样本语义**：训练所有输入有限且 Oracle 标签有效的 nav tick（weight=1）；
     不因数帧后的 hard termination 把当前样本权重置 0（无法归因）。
     非有限 logits/token → 该 env 本 tick 立即回退 Oracle 驱动并计数，
     该 tick valid_mask=0。

解耦的物理保证：optimizer 只含 HighLevelPolicy 参数（构造后断言参数 ID
集合严格相等）；vision_encoder 与 low_level_actor 双保险冻结（.eval() +
requires_grad=False），前向全程 no_grad。

逐帧时序（冻结契约，nav_contract 模块 docstring）由 workflow 按序调用：
    frame_begin(obs, critic_obs)   # 步骤 1-3：注入 exec、低层前向、tick 决策
    → env.step(actions)
    → frame_end(dones)             # 步骤 5 后：done 三方清零 + slew 演化
buffer 满时 workflow 调 finish_nav_sequence_update()。
"""

from __future__ import annotations

import copy
import hashlib
import os
import time

import torch
import torch.nn.functional as F

from agent_ppo.checkpoint_io import (
    KAIWU_TRAIN_FORMAT,
    KAIWU_TRAIN_SCHEMA_VERSION,
    compute_low_level_state_digest,
    high_level_parts,
    validate_high_level_spec,
    is_kaiwu_train_bundle,
    nav_checkpoint_candidates,
    nav_parent_candidates,
    validate_low_level_spec,
)
from agent_ppo.feature import nav_contract
from agent_ppo.feature.nav_event_log import emit_nav_event
from agent_ppo.feature.nav_oracle import NavOracle
from agent_ppo.feature.nav_scheduler import NavScheduler
from agent_ppo.feature.nav_tick_buffer import NavTickBuffer

STAGE_TYPE = "nav_dagger_v1"
_COMMAND_MATCH_TOLERANCE = 1.0e-5
_ACTION_RESPONSE_MIN = 1.0e-3
_VELOCITY_RESPONSE_MIN_MPS = 0.02


def _token_ratio_metrics(prefix: str, tokens: torch.Tensor) -> dict[str, float]:
    """Return a stable dashboard series for every token in the frozen vocabulary."""

    tokens = tokens.detach().long().reshape(-1)
    count = max(1, int(tokens.numel()))
    return {
        f"{prefix}_token_{name}_ratio": float((tokens == index).sum().item()) / count
        for index, name in enumerate(nav_contract.TOKEN_NAMES)
    }


def _dominant_token(tokens: torch.Tensor) -> str:
    counts = torch.bincount(
        tokens.detach().long().reshape(-1), minlength=nav_contract.VOCAB_SIZE
    )
    return nav_contract.TOKEN_NAMES[int(counts.argmax().item())]


def _control_alignment_metrics(
    worker_cmd: torch.Tensor,
    exec_cmd: torch.Tensor,
    held_cmd: torch.Tensor,
    critic_obs: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, float]:
    """Quantify which command the low-level motion actually follows.

    ``worker_cmd`` is captured from the worker-produced observation before the
    aisrv patch. ``exec_cmd`` is the NavScheduler command seen by the frozen
    low-level policy. The environment reward still consumes ``worker_cmd``.
    """

    worker_cmd = worker_cmd.detach()
    exec_cmd = exec_cmd.detach()
    delta = (worker_cmd - exec_cmd).abs()
    linf = delta.amax(dim=-1)

    lin_lo, lin_hi = nav_contract.CRITIC_LIN_VEL_SLICE
    actual_vx = critic_obs[:, lin_lo:lin_hi].detach()[:, 0]
    worker_vx_error = (actual_vx - worker_cmd[:, 0]).abs()
    exec_vx_error = (actual_vx - exec_cmd[:, 0]).abs()
    finite_actions = torch.nan_to_num(actions.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    nonfinite_action_count = int((~torch.isfinite(actions.detach())).sum().item())

    metrics = {
        "worker_exec_cmd_linf_mean": float(linf.mean().item()),
        "worker_exec_cmd_linf_max": float(linf.max().item()),
        "worker_exec_cmd_match_ratio": float(
            (linf <= _COMMAND_MATCH_TOLERANCE).float().mean().item()
        ),
        "worker_cmd_vx_mean": float(worker_cmd[:, 0].mean().item()),
        "worker_cmd_vy_mean": float(worker_cmd[:, 1].mean().item()),
        "worker_cmd_wz_mean": float(worker_cmd[:, 2].mean().item()),
        "exec_cmd_vx_mean": float(exec_cmd[:, 0].mean().item()),
        "exec_cmd_vy_mean": float(exec_cmd[:, 1].mean().item()),
        "exec_cmd_wz_mean": float(exec_cmd[:, 2].mean().item()),
        "held_cmd_vx_mean": float(held_cmd[:, 0].mean().item()),
        "held_cmd_vy_mean": float(held_cmd[:, 1].mean().item()),
        "held_cmd_wz_mean": float(held_cmd[:, 2].mean().item()),
        "actual_lin_vel_x_mean": float(actual_vx.mean().item()),
        "worker_vx_tracking_error_mean": float(worker_vx_error.mean().item()),
        "exec_vx_tracking_error_mean": float(exec_vx_error.mean().item()),
        "exec_vx_closer_ratio": float(
            (exec_vx_error < worker_vx_error).float().mean().item()
        ),
        "low_level_action_abs_mean": float(finite_actions.abs().mean().item()),
        "low_level_action_abs_max": float(finite_actions.abs().max().item()),
        "low_level_action_nonfinite_count": nonfinite_action_count,
    }
    return metrics


def _switch_response_metrics(
    previous_switched: torch.Tensor,
    previous_actions: torch.Tensor,
    current_actions: torch.Tensor,
    previous_actual_vx: torch.Tensor,
    current_actual_vx: torch.Tensor,
    previous_vx_switched: torch.Tensor | None = None,
) -> dict[str, float]:
    """Measure low-level response one nav period after a token switch."""

    mask = previous_switched.detach().bool().reshape(-1)
    sample_count = int(mask.sum().item())
    vx_mask = (
        mask
        if previous_vx_switched is None
        else previous_vx_switched.detach().bool().reshape(-1)
    )
    vx_sample_count = int(vx_mask.sum().item())
    if sample_count == 0:
        return {
            "switch_response_sample_count": 0,
            "switch_response_vx_sample_count": 0,
            "switch_response_action_delta_abs_mean": 0.0,
            "switch_response_vx_delta_abs_mean": 0.0,
            "switch_response_no_action_ratio": 0.0,
            "switch_response_no_velocity_ratio": 0.0,
        }

    previous_actions = torch.nan_to_num(
        previous_actions.detach(), nan=0.0, posinf=0.0, neginf=0.0
    )
    current_actions = torch.nan_to_num(
        current_actions.detach(), nan=0.0, posinf=0.0, neginf=0.0
    )
    previous_actual_vx = torch.nan_to_num(
        previous_actual_vx.detach(), nan=0.0, posinf=0.0, neginf=0.0
    )
    current_actual_vx = torch.nan_to_num(
        current_actual_vx.detach(), nan=0.0, posinf=0.0, neginf=0.0
    )
    action_delta = (current_actions - previous_actions).abs().mean(dim=-1)[mask]
    velocity_delta = (current_actual_vx - previous_actual_vx).abs()[vx_mask]
    return {
        "switch_response_sample_count": sample_count,
        "switch_response_vx_sample_count": vx_sample_count,
        "switch_response_action_delta_abs_mean": float(action_delta.mean().item()),
        "switch_response_vx_delta_abs_mean": (
            float(velocity_delta.mean().item()) if vx_sample_count > 0 else 0.0
        ),
        "switch_response_no_action_ratio": float(
            (action_delta < _ACTION_RESPONSE_MIN).float().mean().item()
        ),
        "switch_response_no_velocity_ratio": (
            float(
                (velocity_delta < _VELOCITY_RESPONSE_MIN_MPS).float().mean().item()
            )
            if vx_sample_count > 0
            else 0.0
        ),
    }


class AlgorithmNavDagger:
    STAGE_TYPE = STAGE_TYPE

    def __init__(
        self,
        vision_encoder,
        low_level_actor,
        high_level,
        device: str = "cuda:0",
        learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
        proprio_dim: int = 45,
        scan_dim: int = 256,
        depth_shape: tuple = (180, 320, 1),
        low_level_parent_model_id: int | str | None = None,
        logger=None,
        process_role: str = "unknown",
    ):
        init_started = time.monotonic()
        self.device = torch.device(device)
        self.logger = logger
        self.process_role = str(process_role)
        self.max_grad_norm = float(max_grad_norm)

        def probe(message: str) -> None:
            if self.logger is not None:
                self.logger.info(
                    "[LifecycleProbe] nav_algorithm_init "
                    f"pid={os.getpid()} role={self.process_role} {message}"
                )

        def cuda_state() -> str:
            if self.device.type != "cuda" or not torch.cuda.is_available():
                return "cuda=unavailable"
            try:
                return (
                    f"cuda_allocated={torch.cuda.memory_allocated(self.device)} "
                    f"cuda_reserved={torch.cuda.memory_reserved(self.device)}"
                )
            except Exception as exc:
                return f"cuda_state_error={type(exc).__name__}:{exc}"

        def module_state(module) -> str:
            parameters = list(module.parameters())
            first_device = parameters[0].device if parameters else "none"
            return (
                f"type={type(module).__name__} params="
                f"{sum(parameter.numel() for parameter in parameters)} "
                f"trainable={sum(parameter.numel() for parameter in parameters if parameter.requires_grad)} "
                f"first_device={first_device}"
            )

        probe(f"enter device={self.device} {cuda_state()}")

        self.proprio_dim = proprio_dim
        self.scan_dim = scan_dim
        self.depth_shape = tuple(depth_shape)

        transfer_started = time.monotonic()
        probe(f"vision_to_device begin {module_state(vision_encoder)} {cuda_state()}")
        self.vision_encoder = vision_encoder.to(self.device)
        probe(
            "vision_to_device complete "
            f"elapsed_s={time.monotonic() - transfer_started:.3f} "
            f"{module_state(self.vision_encoder)} {cuda_state()}"
        )
        transfer_started = time.monotonic()
        probe(f"low_level_to_device begin {module_state(low_level_actor)} {cuda_state()}")
        self.low_level_actor = low_level_actor.to(self.device)
        probe(
            "low_level_to_device complete "
            f"elapsed_s={time.monotonic() - transfer_started:.3f} "
            f"{module_state(self.low_level_actor)} {cuda_state()}"
        )
        transfer_started = time.monotonic()
        probe(f"high_level_to_device begin {module_state(high_level)} {cuda_state()}")
        self.high_level = high_level.to(self.device)
        probe(
            "high_level_to_device complete "
            f"elapsed_s={time.monotonic() - transfer_started:.3f} "
            f"{module_state(self.high_level)} {cuda_state()}"
        )

        probe("architecture_check begin")
        actual_high_level = {
            "input_dim": getattr(self.high_level, "input_dim", None),
            "vocab_size": getattr(self.high_level, "vocab_size", None),
            "rnn_hidden_dim": getattr(self.high_level, "rnn_hidden_dim", None),
            "rnn_num_layers": getattr(self.high_level, "rnn_num_layers", None),
        }
        expected_high_level = {
            "input_dim": nav_contract.NAV_INPUT_DIM,
            "vocab_size": nav_contract.VOCAB_SIZE,
            "rnn_hidden_dim": nav_contract.NAV_LSTM_HIDDEN_SIZE,
            "rnn_num_layers": nav_contract.NAV_LSTM_NUM_LAYERS,
        }
        if actual_high_level != expected_high_level:
            raise ValueError(
                "HighLevelPolicy architecture differs from nav_contract: "
                f"actual={actual_high_level}, expected={expected_high_level}"
            )
        probe("architecture_check complete")

        probe("freeze_low_level begin")
        self._freeze_low_level()
        probe("freeze_low_level complete")

        optimizer_started = time.monotonic()
        probe(
            "optimizer_create begin "
            f"high_level_trainable_params="
            f"{sum(parameter.numel() for parameter in self.high_level.parameters() if parameter.requires_grad)}"
        )
        self.optimizer = torch.optim.Adam(
            self.high_level.parameters(), lr=float(learning_rate)
        )
        probe(
            "optimizer_create complete "
            f"elapsed_s={time.monotonic() - optimizer_started:.3f}"
        )
        probe("optimizer_assert begin")
        self._assert_optimizer_covers_high_level_only()
        probe("optimizer_assert complete")

        probe("oracle_create begin")
        self.oracle = NavOracle()
        probe("oracle_create complete")
        # per-env 组件按首个 batch 尺寸惰性构建
        self.scheduler: NavScheduler | None = None
        self.tick_buffer: NavTickBuffer | None = None
        self._reset_since_last_tick: torch.Tensor | None = None
        self._frame_count = 0

        # ---- 可持久化训练状态（照 lbc 清单）----
        self.current_iteration = 0
        self.total_env_steps = 0
        self.total_nav_ticks = 0
        self.resume_loaded = False
        self.ramp_probability = 0.0
        self.ramp_start_h = 0.5
        self.ramp_end_h = 4.0
        self.ramp_clock_h = 0.0
        self.soft_stay_frozen = False
        self.soft_stay_reason = ""
        self.training_status = "running"
        self.lr_scheduler_state = None
        self.rng_state_saved = None
        # ---- 血缘 ----
        self.low_level_parent_model_id = (
            str(low_level_parent_model_id) if low_level_parent_model_id is not None else None
        )
        self.source_parent_model_id = None
        self.parent_checkpoint_sha256 = None
        self.low_level_state_digest = None
        self.freshness_randomized = True  # 测量链自首训生效（nav_goal_encoder）
        # ---- 运行计数 ----
        self.nonfinite_fallback_count = 0
        self._previous_actions: torch.Tensor | None = None
        self._previous_action_valid: torch.Tensor | None = None
        self._previous_nav_tick_actions: torch.Tensor | None = None
        self._previous_nav_tick_actual_vx: torch.Tensor | None = None
        self._previous_nav_tick_switched: torch.Tensor | None = None
        self._previous_nav_tick_vx_switched: torch.Tensor | None = None
        self.loaded_platform_model_id = None
        self._lifecycle_frame_probe_done = False
        self._lifecycle_tick_probe_done = False
        probe(
            "complete "
            f"elapsed_s={time.monotonic() - init_started:.3f} {cuda_state()}"
        )

    # ------------------------------------------------------------------
    # 冻结与解耦断言
    # ------------------------------------------------------------------

    def _freeze_low_level(self) -> None:
        for module in (self.vision_encoder, self.low_level_actor):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def _assert_optimizer_covers_high_level_only(self) -> None:
        opt_ids = {
            id(p) for group in self.optimizer.param_groups for p in group["params"]
        }
        hl_ids = {id(p) for p in self.high_level.parameters()}
        if opt_ids != hl_ids:
            raise RuntimeError(
                "nav optimizer parameter set must equal HighLevelPolicy parameters "
                f"exactly (optimizer={len(opt_ids)}, high_level={len(hl_ids)})"
            )
        frozen_ids = {id(p) for p in self.vision_encoder.parameters()} | {
            id(p) for p in self.low_level_actor.parameters()
        }
        if opt_ids & frozen_ids:
            raise RuntimeError("frozen low-level parameters leaked into nav optimizer")

    def assert_high_level_parameters_finite(self) -> None:
        for name, param in self.high_level.named_parameters():
            if not torch.isfinite(param).all():
                raise FloatingPointError(f"non-finite high_level parameter: {name}")

    # ------------------------------------------------------------------
    # obs 切分（nav 布局：proprio45 | scan256 | goal4 | depth57600）
    # ------------------------------------------------------------------

    def _split_obs(self, obs: torch.Tensor) -> dict:
        if obs.shape[-1] != nav_contract.POLICY_OBS_DIM:
            raise ValueError(
                f"nav policy obs dim {obs.shape[-1]} != {nav_contract.POLICY_OBS_DIM}"
            )
        n = obs.shape[0]
        h, w, c = self.depth_shape
        proprio = obs[:, : self.proprio_dim]
        goal4 = obs[:, nav_contract.GOAL4_OBS_START : nav_contract.GOAL4_OBS_END]
        depth = obs[:, nav_contract.DEPTH_OBS_START :].reshape(n, h, w, c)
        return {"proprio": proprio, "goal4": goal4, "depth": depth}

    def _ensure_per_env(self, num_envs: int) -> None:
        if self.scheduler is not None and self.scheduler.num_envs == num_envs:
            return
        self.scheduler = NavScheduler(num_envs, self.device)
        self.tick_buffer = NavTickBuffer(
            num_envs,
            self.device,
            seq_len=nav_contract.TBPTT_T,
            rnn_num_layers=self.high_level.rnn_num_layers,
            rnn_hidden_dim=self.high_level.rnn_hidden_dim,
        )
        self._reset_since_last_tick = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_actions = None
        self._previous_action_valid = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_nav_tick_actions = None
        self._previous_nav_tick_actual_vx = None
        self._previous_nav_tick_switched = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_nav_tick_vx_switched = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self.high_level.reset_hidden_state(num_envs, self.device)
        self.vision_encoder.reset_hidden_state(num_envs, self.device)

    # ------------------------------------------------------------------
    # 逐帧滚动（冻结时序步骤 1-3）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def frame_begin(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> dict:
        """步骤 1-3：注入 exec → 低层前向一次 → nav tick 决策。

        返回 dict：{"actions": [N,12], "is_tick": bool, "tick_metrics": {...}}
        """
        first_frame_probe = not self._lifecycle_frame_probe_done
        if first_frame_probe and self.logger is not None:
            self.logger.info(
                "[LifecycleProbe] nav_frame_begin first_call "
                f"obs={tuple(obs.shape)} critic={tuple(critic_obs.shape)} device={obs.device}"
            )
        num_envs = obs.shape[0]
        self._ensure_per_env(num_envs)
        if first_frame_probe and self.logger is not None:
            self.logger.info("[LifecycleProbe] nav_frame_begin per_env_state_ready")

        # Preserve the worker-produced command before the aisrv patch. This is
        # the command consumed by worker-side command-tracking rewards.
        p_lo, p_hi = nav_contract.POLICY_CMD_SLICE
        worker_cmd = obs[:, p_lo:p_hi].detach().clone()

        # 步骤 1：当帧 exec_cmd 写入观测副本（policy [6:9] / critic [9:12] 同值）
        self.scheduler.inject(obs, critic_obs)
        if first_frame_probe and self.logger is not None:
            self.logger.info("[LifecycleProbe] nav_frame_begin command_injected")

        parts = self._split_obs(obs)

        # 步骤 2：低层前向恰一次（VisionEncoder LSTM 每帧只推进一帧）
        if first_frame_probe and self.logger is not None:
            self.logger.info("[LifecycleProbe] nav_frame_begin vision_encoder begin")
        latent = self.vision_encoder(parts["depth"], parts["proprio"], masks=None)
        if first_frame_probe and self.logger is not None:
            self.logger.info(
                f"[LifecycleProbe] nav_frame_begin vision_encoder complete latent={tuple(latent.shape)}"
            )
        actions = self.low_level_actor(torch.cat((parts["proprio"], latent), dim=-1))
        if (
            self._previous_actions is None
            or tuple(self._previous_actions.shape) != tuple(actions.shape)
        ):
            action_delta = torch.zeros_like(actions)
            action_delta_valid = torch.zeros(
                actions.shape[0], dtype=torch.bool, device=actions.device
            )
        else:
            action_delta = actions - self._previous_actions
            action_delta_valid = self._previous_action_valid
        self._previous_actions = actions.detach().clone()
        self._previous_action_valid = torch.ones(
            actions.shape[0], dtype=torch.bool, device=actions.device
        )
        if first_frame_probe and self.logger is not None:
            self.logger.info(
                f"[LifecycleProbe] nav_frame_begin low_level complete actions={tuple(actions.shape)}"
            )

        result = {"actions": actions, "is_tick": False, "tick_metrics": {}}

        # 步骤 3：nav tick（每 nav_period_frames 帧一次）
        if self._frame_count % nav_contract.NAV_PERIOD_FRAMES == 0:
            if first_frame_probe and self.logger is not None:
                self.logger.info("[LifecycleProbe] nav_frame_begin nav_tick begin")
            result["is_tick"] = True
            result["tick_metrics"] = self._nav_tick(
                parts,
                obs,
                critic_obs,
                worker_cmd=worker_cmd,
                actions=actions,
                action_delta=action_delta,
                action_delta_valid=action_delta_valid,
            )
            if first_frame_probe and self.logger is not None:
                self.logger.info("[LifecycleProbe] nav_frame_begin nav_tick complete")

        self._frame_count += 1
        self.total_env_steps += num_envs
        if first_frame_probe:
            self._lifecycle_frame_probe_done = True
            if self.logger is not None:
                self.logger.info("[LifecycleProbe] nav_frame_begin first_call complete")
        return result

    @torch.no_grad()
    def _nav_tick(
        self,
        parts: dict,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        *,
        worker_cmd: torch.Tensor,
        actions: torch.Tensor,
        action_delta: torch.Tensor,
        action_delta_valid: torch.Tensor,
    ) -> dict:
        num_envs = obs.shape[0]
        # This is deliberately process-local rather than derived from the
        # persisted total_nav_ticks counter, so resume smoke runs retain the
        # full first-tick diagnostic chain.
        first_tick_probe = not self._lifecycle_tick_probe_done

        # 段首快照：TBPTT 重放的 (h0, c0) 必须是本段第一个 tick 前向前的 hidden
        if self.tick_buffer.size == 0 and not self.tick_buffer._segment_open:
            self.tick_buffer.start_segment(self.high_level.get_hidden_state())

        # cnn_feat32_raw：同帧 depth 过冻结 CNN（不经 LSTM，冻结语义 v2）
        if first_tick_probe and self.logger is not None:
            self.logger.info("[LifecycleProbe] nav_tick cnn begin")
        cnn_feat_raw = self.vision_encoder.cnn(parts["depth"])
        if first_tick_probe and self.logger is not None:
            self.logger.info(
                f"[LifecycleProbe] nav_tick cnn complete features={tuple(cnn_feat_raw.shape)}"
            )

        nav_inputs_raw = torch.cat(
            (
                cnn_feat_raw,
                parts["goal4"],
                self.scheduler.exec_cmd,
                self.scheduler.held_cmd,  # 更新前的 held_cmd
                parts["proprio"][:, 0:3],  # ang_vel3
                parts["proprio"][:, 3:6],  # proj_grav3
            ),
            dim=-1,
        )
        if nav_inputs_raw.shape[-1] != nav_contract.NAV_INPUT_DIM:
            raise ValueError(
                f"nav input dim {nav_inputs_raw.shape[-1]} != {nav_contract.NAV_INPUT_DIM}"
            )
        inputs_finite = torch.isfinite(nav_inputs_raw).all(dim=-1)
        # NaN 消毒（CRITICAL 修复）：先消毒再前向/落库——非有限值一旦进入
        # 高层 LSTM 或 TBPTT 重放会污染 hidden 且 NaN*0=NaN 使 valid 掩码失效。
        nav_inputs = torch.nan_to_num(nav_inputs_raw, nan=0.0, posinf=0.0, neginf=0.0)

        dwell_mask = self.scheduler.dwell_mask()
        if first_tick_probe and self.logger is not None:
            self.logger.info("[LifecycleProbe] nav_tick high_level begin")
        logits = self.high_level(nav_inputs, dwell_mask=dwell_mask)
        if first_tick_probe and self.logger is not None:
            self.logger.info(
                f"[LifecycleProbe] nav_tick high_level complete logits={tuple(logits.shape)}"
            )
        student_tokens = logits.argmax(dim=-1)

        if first_tick_probe and self.logger is not None:
            self.logger.info("[LifecycleProbe] nav_tick oracle begin")
        oracle_tokens = self.oracle.act(critic_obs)
        oracle_valid = NavOracle.label_validity(critic_obs)
        if first_tick_probe and self.logger is not None:
            self.logger.info("[LifecycleProbe] nav_tick oracle complete")

        logits_finite = torch.isfinite(logits).all(dim=-1)
        student_ok = inputs_finite & logits_finite
        fallback = ~student_ok
        if bool(fallback.any()):
            # 非有限 → 立即回退 Oracle 驱动并记录（该 tick 标签无效）；
            # 同时重置这些 env 的高层 rollout hidden（消毒前的前向可能已受污染）
            self.nonfinite_fallback_count += int(fallback.sum().item())
            fallback_ids = fallback.nonzero(as_tuple=False).squeeze(-1)
            self.high_level.reset_hidden_state_for_envs(fallback_ids)

        # 标签投影（CRITICAL 修复）：Oracle 不感知驻留纪律，其请求可能落在
        # dwell 掩码之外——被屏蔽标签的 logit 为 -1e9，CE 会爆到 ~1e9 且梯度
        # 退化。把标签投影为"教师在驻留纪律下实际会执行的动作"：允许则用
        # Oracle 请求，否则保持更新前的 held token（zero 恒被允许，急停语义
        # 不受影响）。
        prev_held = self.scheduler.held_token.clone()
        oracle_allowed = dwell_mask.gather(1, oracle_tokens.unsqueeze(1)).squeeze(1)
        oracle_labels = torch.where(oracle_allowed, oracle_tokens, prev_held)

        # DAgger 混合（nav tick 粒度 per-env Bernoulli）
        student_drive = (
            torch.rand(num_envs, device=self.device) < float(self.ramp_probability)
        ) & student_ok
        executed = torch.where(student_drive, student_tokens, oracle_tokens)
        effective = self.scheduler.request_tokens(executed)
        switched = effective != prev_held

        lin_lo, lin_hi = nav_contract.CRITIC_LIN_VEL_SLICE
        actual_vx = critic_obs[:, lin_lo:lin_hi].detach()[:, 0]
        if (
            self._previous_nav_tick_actions is None
            or self._previous_nav_tick_actual_vx is None
            or self._previous_nav_tick_switched is None
            or self._previous_nav_tick_vx_switched is None
        ):
            response_metrics = _switch_response_metrics(
                torch.zeros(num_envs, dtype=torch.bool, device=self.device),
                actions,
                actions,
                actual_vx,
                actual_vx,
                previous_vx_switched=torch.zeros(
                    num_envs, dtype=torch.bool, device=self.device
                ),
            )
        else:
            response_metrics = _switch_response_metrics(
                self._previous_nav_tick_switched,
                self._previous_nav_tick_actions,
                actions,
                self._previous_nav_tick_actual_vx,
                actual_vx,
                previous_vx_switched=self._previous_nav_tick_vx_switched,
            )
        vocab = torch.as_tensor(
            nav_contract.VOCAB, dtype=actions.dtype, device=self.device
        )
        vx_switched = switched & (
            (vocab[effective, 0] - vocab[prev_held, 0]).abs()
            > _COMMAND_MATCH_TOLERANCE
        )
        self._previous_nav_tick_actions = actions.detach().clone()
        self._previous_nav_tick_actual_vx = actual_vx.detach().clone()
        self._previous_nav_tick_switched = switched.detach().clone()
        self._previous_nav_tick_vx_switched = vx_switched.detach().clone()

        valid = oracle_valid & inputs_finite & logits_finite

        goal3_start = nav_contract.CRITIC_GOAL3_START
        goal3 = critic_obs[:, goal3_start : goal3_start + 3]
        goal3_abs = torch.nan_to_num(goal3.abs(), nan=0.0, posinf=0.0, neginf=0.0)
        goal4_fresh = torch.isfinite(parts["goal4"][:, 3]) & (
            parts["goal4"][:, 3] > 0.0
        )
        goal_metrics = {
            "oracle_valid_count": int(oracle_valid.sum().item()),
            "oracle_sample_count": int(num_envs),
            "oracle_valid_ratio": float(oracle_valid.float().mean().item()),
            "goal3_abs_mean": float(goal3_abs.mean().item()),
            "goal3_abs_max": float(goal3_abs.max().item()),
            "goal4_fresh_count": int(goal4_fresh.sum().item()),
            "goal4_sample_count": int(num_envs),
            "goal4_fresh_ratio": float(goal4_fresh.float().mean().item()),
        }
        if first_tick_probe:
            if self.logger is not None:
                self.logger.info(
                    "[NavGoalProbe] first_nav_tick "
                    + " ".join(f"{key}={value}" for key, value in goal_metrics.items())
                )
            emit_nav_event(
                "first_nav_tick",
                role=self.process_role,
                **goal_metrics,
            )

        # 收集 TBPTT 样本（消毒后的输入 + 投影后的标签；
        # 旧段 reset 信息随本 tick 落库后清零）
        self.tick_buffer.add(
            nav_inputs,
            oracle_labels,
            self._reset_since_last_tick,
            valid,
            dwell_mask,
        )
        if first_tick_probe and self.logger is not None:
            self.logger.info("[LifecycleProbe] nav_tick buffer_add complete")
        if first_tick_probe:
            self._lifecycle_tick_probe_done = True
        self._reset_since_last_tick = torch.zeros_like(self._reset_since_last_tick)
        self.total_nav_ticks += num_envs

        disagreement = (student_tokens != oracle_labels) & oracle_valid
        metrics = {
            "buffer_full": self.tick_buffer.is_full,
            "student_drive_ratio": float(student_drive.float().mean().item()),
            "oracle_drive_ratio": float((~student_drive).float().mean().item()),
            "requested_effective_mismatch_ratio": float(
                (executed != effective).float().mean().item()
            ),
            "disagreement_rate": float(
                disagreement.float().sum().item() / max(1.0, oracle_valid.float().sum().item())
            ),
            "switch_rate": float(switched.float().mean().item()),
            "valid_ratio": float(valid.float().mean().item()),
            "nonfinite_fallback": int(fallback.sum().item()),
            **goal_metrics,
        }
        metrics.update(_token_ratio_metrics("student", student_tokens))
        metrics.update(_token_ratio_metrics("oracle", oracle_tokens))
        metrics.update(_token_ratio_metrics("requested", executed))
        metrics.update(_token_ratio_metrics("effective", effective))
        metrics.update(
            _control_alignment_metrics(
                worker_cmd,
                self.scheduler.exec_cmd,
                self.scheduler.held_cmd,
                critic_obs,
                actions,
            )
        )
        valid_delta_mask = action_delta_valid.to(device=action_delta.device).bool()
        valid_delta = valid_delta_mask.float().sum().clamp_min(1.0)
        masked_action_delta = action_delta[valid_delta_mask]
        if masked_action_delta.numel() == 0:
            masked_action_delta = action_delta.new_zeros((1, action_delta.shape[-1]))
        metrics["low_level_action_delta_abs_mean"] = float(
            masked_action_delta.abs().sum().item()
            / valid_delta.item()
            / max(1, action_delta.shape[-1])
        )
        metrics["scheduler_dwell_ticks_mean"] = float(
            self.scheduler.dwell_ticks.float().mean().item()
        )
        metrics["scheduler_dwell_ticks_max"] = float(
            self.scheduler.dwell_ticks.max().item()
        )
        metrics.update(response_metrics)
        metrics.update(getattr(self.oracle, "last_metrics", {}))
        if first_tick_probe and self.logger is not None:
            self.logger.info(
                "[NavControlProbe] first_nav_tick "
                f"driver_student={metrics['student_drive_ratio']:.3f} "
                f"driver_oracle={metrics['oracle_drive_ratio']:.3f} "
                f"student_dom={_dominant_token(student_tokens)} "
                f"oracle_dom={_dominant_token(oracle_tokens)} "
                f"requested_dom={_dominant_token(executed)} "
                f"effective_dom={_dominant_token(effective)} "
                f"worker_cmd={metrics['worker_cmd_vx_mean']:.3f},"
                f"{metrics['worker_cmd_vy_mean']:.3f},"
                f"{metrics['worker_cmd_wz_mean']:.3f} "
                f"exec_cmd={metrics['exec_cmd_vx_mean']:.3f},"
                f"{metrics['exec_cmd_vy_mean']:.3f},"
                f"{metrics['exec_cmd_wz_mean']:.3f} "
                f"actual_vx={metrics['actual_lin_vel_x_mean']:.3f} "
                f"action_abs={metrics['low_level_action_abs_mean']:.3f}"
            )
        return metrics

    def frame_end(self, dones: torch.Tensor) -> None:
        """步骤 5 后：done 三方清零（低层 hidden / 高层 hidden / scheduler），
        再 slew 演化一帧（下一帧的 exec_cmd）。"""
        if self.scheduler is None:
            return
        dones = dones.to(self.device).bool()
        if bool(dones.any()):
            ids = dones.nonzero(as_tuple=False).squeeze(-1)
            self.vision_encoder.reset_hidden_state_for_envs(ids)
            self.high_level.reset_hidden_state_for_envs(ids)
            self.scheduler.reset(ids)
            if self._previous_action_valid is not None:
                self._previous_action_valid[ids] = False
            if self._previous_nav_tick_switched is not None:
                self._previous_nav_tick_switched[ids] = False
            if self._previous_nav_tick_vx_switched is not None:
                self._previous_nav_tick_vx_switched[ids] = False
            self._reset_since_last_tick = self._reset_since_last_tick | dones
        self.scheduler.step_exec()

    # ------------------------------------------------------------------
    # TBPTT 序列更新
    # ------------------------------------------------------------------

    def finish_nav_sequence_update(self) -> dict:
        """buffer 满 T 时的一次序列交叉熵更新；返回指标 dict。"""
        seq = self.tick_buffer.get()
        inputs = seq["inputs"]                    # [T, B, 48]
        tokens = seq["tokens"]                    # [T, B]
        valid_bool = seq["valid_masks"].bool()    # [T, B]
        valid = valid_bool.float()

        # 全段无有效样本：跳过更新（loss=0 反传仍会让 Adam 动量微动参数）
        weight_sum_raw = valid.sum()
        if float(weight_sum_raw.item()) <= 0.0:
            self.tick_buffer.clear()
            return {
                "ce_loss": 0.0,
                "top1_accuracy": 0.0,
                "token_entropy": 0.0,
                "grad_norm": 0.0,
                "valid_ticks": 0.0,
                "update_skipped_no_valid": 1.0,
            }

        logits = self.high_level.forward_sequence(
            inputs, seq["initial_hidden"], seq["reset_masks"], seq["dwell_masks"]
        )                                          # [T, B, V]

        T, B, V = logits.shape
        per_ce = F.cross_entropy(
            logits.reshape(T * B, V), tokens.reshape(T * B), reduction="none"
        ).reshape(T, B)
        # NaN 中和（CRITICAL 修复）：valid=0 的样本用 where 归零而非乘法
        # （NaN * 0 = NaN 会击穿掩码并炸掉整段更新）。
        per_ce = torch.where(valid_bool, per_ce, torch.zeros_like(per_ce))
        weight_sum = weight_sum_raw.clamp_min(1.0)
        loss = per_ce.sum() / weight_sum
        if not torch.isfinite(loss):
            raise FloatingPointError("nav DAgger sequence loss is non-finite")

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.high_level.parameters(), self.max_grad_norm
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("nav DAgger grad norm is non-finite")
        self.optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = ((pred == tokens).float() * valid).sum() / weight_sum
            probs = F.softmax(logits.reshape(T * B, V), dim=-1)
            entropy = (
                -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1).reshape(T, B)
            )
            mean_entropy = (entropy * valid).sum() / weight_sum

        self.tick_buffer.clear()
        return {
            "ce_loss": float(loss.item()),
            "top1_accuracy": float(correct.item()),
            "token_entropy": float(mean_entropy.item()),
            "grad_norm": float(grad_norm.item()),
            "valid_ticks": float(valid.sum().item()),
        }

    def train_mode(self) -> None:
        self.high_level.train()
        # 冻结件永远保持 eval

    def eval_mode(self) -> None:
        self.high_level.eval()

    # ------------------------------------------------------------------
    # checkpoint：保存（单一 nav* 文件；digest 漂移仅诊断）
    # ------------------------------------------------------------------

    def _sha256(self, path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _live_low_level_bundle_view(self) -> dict:
        return {
            "modules": {
                "vision_encoder": {"state_dict": self.vision_encoder.state_dict()},
                "low_level": {"actor_state_dict": self.low_level_actor.state_dict()},
            }
        }

    def save_nav_bundle(self, path: str, *, platform_model_id, phase_label: str) -> str:
        current_digest = compute_low_level_state_digest(self._live_low_level_bundle_view())
        if self.low_level_state_digest is None:
            raise RuntimeError(
                "low_level_state_digest not initialized — load_parent_bundle / "
                "load_nav_resume must run before the first save"
            )
        if current_digest != self.low_level_state_digest:
            if self.logger is not None:
                self.logger.warning(
                    "[CheckpointIdentity] WARNING-ONLY: frozen low-level digest "
                    "changed before save; writing the actual recomputed digest: "
                    f"previous={self.low_level_state_digest}, actual={current_digest}"
                )
            self.low_level_state_digest = current_digest

        payload = {
            "format": KAIWU_TRAIN_FORMAT,
            "schema_version": KAIWU_TRAIN_SCHEMA_VERSION,
            "stage_type": self.STAGE_TYPE,
            "phase_label": phase_label,
            "platform_model_id": str(platform_model_id),
            "model_spec": {
                "proprio_dim": self.proprio_dim,
                "scan_dim": self.scan_dim,
                "latent_dim": self.vision_encoder.rnn_output_dim,
                "action_dim": 12,
                "goal_dim": 0,  # 低层 spec 恒为 0（写 3 会让现有加载器硬拒绝）
            },
            "modules": {
                "vision_encoder": {
                    "class_name": type(self.vision_encoder).__name__,
                    "state_dict": copy.deepcopy(self.vision_encoder.state_dict()),
                    "trainable": False,
                },
                "low_level": {
                    "class_name": "Actor77Sequential",
                    "actor_state_dict": copy.deepcopy(self.low_level_actor.state_dict()),
                    "frozen": True,
                    "source": self.source_parent_model_id,
                },
                "high_level": {
                    "class_name": type(self.high_level).__name__,
                    "state_dict": copy.deepcopy(self.high_level.state_dict()),
                    "trainable": True,
                    **nav_contract.high_level_checkpoint_contract(),
                },
            },
            "optimizers": {
                "high_level_dagger": self.optimizer.state_dict(),
            },
            "training_state": {
                "current_iteration": self.current_iteration,
                "iteration_semantics": "completed_outer_iterations_v1",
                "total_env_steps": self.total_env_steps,
                "total_nav_ticks": self.total_nav_ticks,
                "ramp_probability": self.ramp_probability,
                "ramp_start_h": self.ramp_start_h,
                "ramp_end_h": self.ramp_end_h,
                "ramp_clock_h": self.ramp_clock_h,
                "soft_stay_frozen": self.soft_stay_frozen,
                "soft_stay_reason": self.soft_stay_reason,
                "training_status": self.training_status,
                "lr_scheduler_state": self.lr_scheduler_state,
                "nonfinite_fallback_count": self.nonfinite_fallback_count,
                "schedule_mode": "nav_dagger_tbptt_v1",
            },
            "lineage": {
                "source_parent_model_id": self.source_parent_model_id,
                "low_level_parent_model_id": self.low_level_parent_model_id,
                "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
                "low_level_state_digest": current_digest,
            },
            "capabilities": {
                "task": "track_nav",
                "uses_depth": True,
                "uses_height_scan_at_inference": False,
                "goal_dim": 0,
                "nav_goal_dim": 4,
                "deployable": False,  # 部署另立任务；导出工具据此拒绝
                "freshness_randomized": self.freshness_randomized,
            },
            "lstm_reset_contract": {
                "low_level": "reset per-env on done; never restored from checkpoint",
                "high_level": "reset per-env on done; never restored from checkpoint",
            },
        }

        torch.save(payload, path)
        if os.path.getsize(path) <= 0:
            raise RuntimeError(f"nav bundle write produced empty file: {path}")
        return self._sha256(path)

    # ------------------------------------------------------------------
    # checkpoint：加载（首载低层父 / nav resume）
    # ------------------------------------------------------------------

    def _load_low_level_from_bundle(self, bundle: dict) -> None:
        validate_low_level_spec(
            bundle,
            {
                "proprio_dim": self.proprio_dim,
                "scan_dim": self.scan_dim,
                "latent_dim": self.vision_encoder.rnn_output_dim,
                "action_dim": 12,
                "goal_dim": 0,
            },
        )
        modules = bundle.get("modules", {})
        vision_section = modules.get("vision_encoder", {})
        vision_state = vision_section.get("state_dict")
        if not isinstance(vision_state, dict):
            raise KeyError("modules.vision_encoder.state_dict missing")
        self.vision_encoder.load_state_dict(vision_state, strict=True)
        # 注意：visual_ppo 父包与 nav 包的 modules.low_level 只有 actor_state_dict
        # （教师 DmEncoder 不属于低层部署形态），不能用 low_level_teacher_parts。
        low_level = modules.get("low_level", {})
        actor_state = (
            low_level.get("actor_state_dict") if isinstance(low_level, dict) else None
        )
        if not isinstance(actor_state, dict):
            raise KeyError("modules.low_level.actor_state_dict missing")
        self.low_level_actor.load_state_dict(actor_state, strict=True)
        self._freeze_low_level()
        self.low_level_state_digest = compute_low_level_state_digest(
            self._live_low_level_bundle_view()
        )

    def load_parent_bundle(self, path: str, model_id) -> str | None:
        """首载低层父（分支 1）：低层进权重、高层保持随机初始化、当场算 digest。"""
        for candidate in nav_parent_candidates(path, model_id):
            if not os.path.isfile(candidate):
                continue
            bundle = torch.load(candidate, map_location=self.device, weights_only=False)
            if not is_kaiwu_train_bundle(bundle):
                continue
            bundle_id = bundle.get("platform_model_id")
            if self.logger is not None and (
                bundle_id in (None, "") or str(bundle_id) != str(model_id)
            ):
                self.logger.warning(
                    "[CheckpointIdentity] WARNING-ONLY: low-level parent bundle "
                    "platform_model_id is missing or differs from the requested "
                    f"preload; continuing after structural validation: "
                    f"requested={model_id}, bundle={bundle_id!r}, path={candidate}"
                )
            self._load_low_level_from_bundle(bundle)
            self.source_parent_model_id = str(model_id)
            self.parent_checkpoint_sha256 = self._sha256(candidate)
            self.resume_loaded = False
            self.loaded_platform_model_id = str(model_id)
            if self.logger is not None:
                self.logger.info(
                    f"[nav] first-load low-level parent: {candidate} "
                    f"sha256={self.parent_checkpoint_sha256} "
                    f"low_level_digest={self.low_level_state_digest} "
                    "(high_level stays randomly initialized)"
                )
            return candidate
        return None

    def load_nav_resume(self, path: str, model_id) -> str | None:
        """nav resume（分支 3）：全量恢复；per-env 活状态不恢复（全 reset）。

        损坏的同 ID resume 候选是硬失败（宁硬停不静默）：若存在 nav 标签
        文件但格式/stage_type 不符，raise 而非跳过——静默跳过叠加任何父包
        回退会在无告警的情况下丢弃全部高层训练进度。
        """
        for candidate in nav_checkpoint_candidates(path, model_id):
            bundle = torch.load(candidate, map_location=self.device, weights_only=False)
            if not is_kaiwu_train_bundle(bundle):
                raise ValueError(
                    f"[nav] corrupted resume candidate (not a kaiwu_train_v1 "
                    f"bundle): {candidate} — refusing to silently skip"
                )
            if bundle.get("stage_type") != self.STAGE_TYPE:
                raise ValueError(
                    f"[nav] resume candidate has stage_type="
                    f"{bundle.get('stage_type')!r}, expected {self.STAGE_TYPE!r}: "
                    f"{candidate} — refusing to silently skip"
                )
            bundle_id = bundle.get("platform_model_id")
            if self.logger is not None and (
                bundle_id in (None, "") or str(bundle_id) != str(model_id)
            ):
                self.logger.warning(
                    "[CheckpointIdentity] WARNING-ONLY: nav resume platform_model_id "
                    "is missing or differs from the requested preload; continuing "
                    f"after structural validation: requested={model_id}, "
                    f"bundle={bundle_id!r}, path={candidate}"
                )
            self._load_low_level_from_bundle(bundle)

            hl_state, _hl_meta = high_level_parts(bundle)
            validate_high_level_spec(
                bundle, nav_contract.high_level_checkpoint_contract()
            )
            self.high_level.load_state_dict(hl_state, strict=True)

            optimizers = bundle.get("optimizers", {})
            opt_state = optimizers.get("high_level_dagger")
            if isinstance(opt_state, dict):
                try:
                    self.optimizer.load_state_dict(opt_state)
                except (ValueError, KeyError, RuntimeError) as exc:
                    if self.logger is not None:
                        self.logger.warning(f"[nav] optimizer restore failed: {exc}")

            state = bundle.get("training_state", {})
            self.current_iteration = int(state.get("current_iteration", 0))
            self.total_env_steps = int(state.get("total_env_steps", 0))
            self.total_nav_ticks = int(state.get("total_nav_ticks", 0))
            self.ramp_probability = float(state.get("ramp_probability", 0.0))
            self.ramp_start_h = float(state.get("ramp_start_h", self.ramp_start_h))
            self.ramp_end_h = float(state.get("ramp_end_h", self.ramp_end_h))
            self.ramp_clock_h = float(state.get("ramp_clock_h", 0.0))
            self.soft_stay_frozen = bool(state.get("soft_stay_frozen", False))
            self.soft_stay_reason = str(state.get("soft_stay_reason", ""))
            self.lr_scheduler_state = state.get("lr_scheduler_state")
            self.nonfinite_fallback_count = int(
                state.get("nonfinite_fallback_count", 0)
            )

            lineage_value = bundle.get("lineage", {})
            if isinstance(lineage_value, dict):
                lineage = lineage_value
            else:
                lineage = {}
                if self.logger is not None:
                    self.logger.warning(
                        "[CheckpointIdentity] WARNING-ONLY: nav resume lineage "
                        f"metadata is not a dict; ignoring it: {candidate}"
                    )
            self.source_parent_model_id = lineage.get("source_parent_model_id")
            self.parent_checkpoint_sha256 = lineage.get("parent_checkpoint_sha256")
            stored_low_level_parent = lineage.get("low_level_parent_model_id")
            configured_low_level_parent = self.low_level_parent_model_id
            if stored_low_level_parent in (None, ""):
                stored_low_level_parent = configured_low_level_parent
                if self.logger is not None:
                    self.logger.warning(
                        "[CheckpointIdentity] WARNING-ONLY: nav resume bundle has "
                        "no lineage.low_level_parent_model_id; preserving the "
                        f"configured value={configured_low_level_parent!r}: {candidate}"
                    )
            if (
                configured_low_level_parent not in (None, "")
                and str(configured_low_level_parent) != str(stored_low_level_parent)
                and self.logger is not None
            ):
                self.logger.warning(
                    "[nav] configured low-level parent differs from resume lineage; "
                    "preserving the checkpoint's truthful lineage: "
                    f"configured={configured_low_level_parent}, "
                    f"checkpoint={stored_low_level_parent}"
                )
            self.low_level_parent_model_id = (
                str(stored_low_level_parent)
                if stored_low_level_parent not in (None, "")
                else None
            )
            stored_digest = lineage.get("low_level_state_digest")
            if not stored_digest:
                if self.logger is not None:
                    self.logger.warning(
                        "[CheckpointIdentity] WARNING-ONLY: nav resume bundle has "
                        "no lineage.low_level_state_digest; using recomputed digest="
                        f"{self.low_level_state_digest}: {candidate}"
                    )
            elif stored_digest != self.low_level_state_digest and self.logger is not None:
                self.logger.warning(
                    "[CheckpointIdentity] WARNING-ONLY: nav resume low-level digest "
                    "differs from lineage; continuing with loaded tensors: "
                    f"lineage={stored_digest} recomputed={self.low_level_state_digest}"
                )

            capabilities = bundle.get("capabilities", {})
            if isinstance(capabilities, dict):
                self.freshness_randomized = bool(
                    capabilities.get("freshness_randomized", self.freshness_randomized)
                )

            # per-env 活状态（scheduler/buffer/hidden）不恢复：留待 _ensure_per_env
            # 在下一次 rollout 全新构建，环境整体 reset（计划 §2 分支 3 语义）。
            self.scheduler = None
            self.tick_buffer = None
            self._frame_count = 0

            self.resume_loaded = True
            self.loaded_platform_model_id = str(model_id)
            if self.logger is not None:
                self.logger.info(
                    f"[nav] resume: {candidate} iteration={self.current_iteration} "
                    f"ramp={self.ramp_probability:.3f} "
                    f"low_level_digest={self.low_level_state_digest}"
                )
            return candidate
        return None
