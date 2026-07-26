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

import torch
import torch.nn.functional as F

from agent_ppo.checkpoint_io import (
    KAIWU_TRAIN_FORMAT,
    KAIWU_TRAIN_SCHEMA_VERSION,
    compute_low_level_state_digest,
    high_level_parts,
    is_kaiwu_train_bundle,
    nav_checkpoint_candidates,
    nav_parent_candidates,
    validate_low_level_spec,
)
from agent_ppo.feature import nav_contract
from agent_ppo.feature.nav_oracle import NavOracle
from agent_ppo.feature.nav_scheduler import NavScheduler
from agent_ppo.feature.nav_tick_buffer import NavTickBuffer

STAGE_TYPE = "nav_dagger_v1"


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
    ):
        self.device = torch.device(device)
        self.logger = logger
        self.max_grad_norm = float(max_grad_norm)

        self.proprio_dim = proprio_dim
        self.scan_dim = scan_dim
        self.depth_shape = tuple(depth_shape)

        self.vision_encoder = vision_encoder.to(self.device)
        self.low_level_actor = low_level_actor.to(self.device)
        self.high_level = high_level.to(self.device)

        self._freeze_low_level()

        self.optimizer = torch.optim.Adam(
            self.high_level.parameters(), lr=float(learning_rate)
        )
        self._assert_optimizer_covers_high_level_only()

        self.oracle = NavOracle()
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
        self.loaded_platform_model_id = None

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
        num_envs = obs.shape[0]
        self._ensure_per_env(num_envs)

        # 步骤 1：当帧 exec_cmd 写入观测副本（policy [6:9] / critic [9:12] 同值）
        self.scheduler.inject(obs, critic_obs)

        parts = self._split_obs(obs)

        # 步骤 2：低层前向恰一次（VisionEncoder LSTM 每帧只推进一帧）
        latent = self.vision_encoder(parts["depth"], parts["proprio"], masks=None)
        actions = self.low_level_actor(torch.cat((parts["proprio"], latent), dim=-1))

        result = {"actions": actions, "is_tick": False, "tick_metrics": {}}

        # 步骤 3：nav tick（每 nav_period_frames 帧一次）
        if self._frame_count % nav_contract.NAV_PERIOD_FRAMES == 0:
            result["is_tick"] = True
            result["tick_metrics"] = self._nav_tick(parts, obs, critic_obs)

        self._frame_count += 1
        self.total_env_steps += num_envs
        return result

    @torch.no_grad()
    def _nav_tick(self, parts: dict, obs: torch.Tensor, critic_obs: torch.Tensor) -> dict:
        num_envs = obs.shape[0]

        # 段首快照：TBPTT 重放的 (h0, c0) 必须是本段第一个 tick 前向前的 hidden
        if self.tick_buffer.size == 0 and not self.tick_buffer._segment_open:
            self.tick_buffer.start_segment(self.high_level.get_hidden_state())

        # cnn_feat32_raw：同帧 depth 过冻结 CNN（不经 LSTM，冻结语义 v2）
        cnn_feat_raw = self.vision_encoder.cnn(parts["depth"])

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
        logits = self.high_level(nav_inputs, dwell_mask=dwell_mask)
        student_tokens = logits.argmax(dim=-1)

        oracle_tokens = self.oracle.act(critic_obs)
        oracle_valid = NavOracle.label_validity(critic_obs)

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

        valid = oracle_valid & inputs_finite & logits_finite

        # 收集 TBPTT 样本（消毒后的输入 + 投影后的标签；
        # 旧段 reset 信息随本 tick 落库后清零）
        self.tick_buffer.add(
            nav_inputs,
            oracle_labels,
            self._reset_since_last_tick,
            valid,
            dwell_mask,
        )
        self._reset_since_last_tick = torch.zeros_like(self._reset_since_last_tick)
        self.total_nav_ticks += num_envs

        disagreement = (student_tokens != oracle_labels) & oracle_valid
        return {
            "buffer_full": self.tick_buffer.is_full,
            "student_drive_ratio": float(student_drive.float().mean().item()),
            "disagreement_rate": float(
                disagreement.float().sum().item() / max(1.0, oracle_valid.float().sum().item())
            ),
            "switch_rate": float(switched.float().mean().item()),
            "valid_ratio": float(valid.float().mean().item()),
            "nonfinite_fallback": int(fallback.sum().item()),
        }

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
    # checkpoint：保存（单一 nav* 文件；digest 硬校验）
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
        # digest 硬校验：低层被污染 = 事故级硬失败（不是 warning）
        current_digest = compute_low_level_state_digest(self._live_low_level_bundle_view())
        if self.low_level_state_digest is None:
            raise RuntimeError(
                "low_level_state_digest not initialized — load_parent_bundle / "
                "load_nav_resume must run before the first save"
            )
        if current_digest != self.low_level_state_digest:
            raise RuntimeError(
                "frozen low level was modified during nav training: "
                f"expected digest {self.low_level_state_digest}, got {current_digest}"
            )

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
                    "vocab": [list(v) for v in nav_contract.VOCAB],
                    "input_layout_version": nav_contract.INPUT_LAYOUT_VERSION,
                    "nav_period_frames": nav_contract.NAV_PERIOD_FRAMES,
                    "min_dwell_ticks": nav_contract.MIN_DWELL_TICKS,
                    "slew_rate_up": list(nav_contract.SLEW_RATE_UP),
                    "slew_rate_down": list(nav_contract.SLEW_RATE_DOWN),
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
                "low_level_state_digest": self.low_level_state_digest,
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
            self._load_low_level_from_bundle(bundle)

            hl_state, hl_meta = high_level_parts(bundle)
            if hl_meta.get("input_layout_version") != nav_contract.INPUT_LAYOUT_VERSION:
                raise ValueError(
                    "nav resume input_layout_version mismatch: "
                    f"{hl_meta.get('input_layout_version')} != "
                    f"{nav_contract.INPUT_LAYOUT_VERSION}"
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

            lineage = bundle.get("lineage", {})
            self.source_parent_model_id = lineage.get("source_parent_model_id")
            self.parent_checkpoint_sha256 = lineage.get("parent_checkpoint_sha256")
            stored_digest = lineage.get("low_level_state_digest")
            # 与 eval 侧对称：缺失 digest 同样硬失败（被剥除 digest 的包
            # 不得无检通过 resume）。
            if not stored_digest:
                raise RuntimeError(
                    f"[nav] resume bundle has no lineage.low_level_state_digest: "
                    f"{candidate}"
                )
            if stored_digest != self.low_level_state_digest:
                raise RuntimeError(
                    "nav resume low-level digest mismatch: "
                    f"lineage={stored_digest} recomputed={self.low_level_state_digest}"
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
