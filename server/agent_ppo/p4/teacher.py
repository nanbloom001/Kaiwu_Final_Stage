#!/usr/bin/env python3
"""Training-only teacher, mirror, and auxiliary-loss services for P4."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import torch
import torch.nn.functional as F

from agent_ppo.algorithm.algorithm_visual_ppo import _calibrate_auxiliary_gradients
from agent_ppo.feature import p2_contract, p4_contract
from agent_ppo.model.p2_high_level import assemble_actor_input
from agent_ppo.p4.profiles import (
    PROFILE_MAZE_CLOSED_LOOP_V3,
    PROFILE_MAZE_INSTANT_COMMAND_R4,
    PROFILE_MAZE_INSTANT_REPAIR2H,
    PROFILE_MAZE_STABLE_DIRECTION_SMOKE,
    PROFILE_MAZE_STABLE_DIRECTION_8H,
)


@dataclass(frozen=True)
class AuxiliaryParameterSpec:
    """Trainable module groups an auxiliary loss may reach."""

    name: str
    allowed_groups: frozenset[str]


AUXILIARY_PARAMETER_SPECS = MappingProxyType(
    {
        "safety": AuxiliaryParameterSpec(
            "safety", frozenset({"navigation_encoder", "safety_head"})
        ),
        "goal": AuxiliaryParameterSpec("goal", frozenset({"actor"})),
        "teacher": AuxiliaryParameterSpec("teacher", frozenset({"actor"})),
        "stuck": AuxiliaryParameterSpec("stuck", frozenset({"actor", "stuck_head"})),
        "anchor": AuxiliaryParameterSpec("anchor", frozenset({"actor"})),
        "camera": AuxiliaryParameterSpec(
            "camera", frozenset({"navigation_encoder", "actor"})
        ),
        "mirror": AuxiliaryParameterSpec("mirror", frozenset({"actor"})),
    }
)


class P4TeacherMixin:
    """Behavior-preserving methods extracted from AlgorithmP4NavPPO."""

    def _update_policy_auxiliary_target(
        self, *, parts, nav_feat, nav_nonvisual, profile, confidence, reset
    ) -> None:
        if self._clean_depth is None:
            return
        diagnostic_active = (
            self.maze_training_branch == "auto"
            and self._resolved_maze_training_branch is None
            and self.diagnostic_elapsed_seconds < p4_contract.DIAGNOSTIC_SECONDS
        )
        if diagnostic_active:
            (
                self._diagnostic_fault_depth,
                self._diagnostic_fault_mask,
            ) = self.camera_state.diagnostic_fault_shadow(
                self._clean_depth, sample_share=0.10
            )
        else:
            self._diagnostic_fault_depth = None
            self._diagnostic_fault_mask.zero_()
        self._teacher_hidden = self._mask_hidden(self._teacher_hidden, reset)
        teacher_hidden_before = (
            tuple(item.clone() for item in self._teacher_hidden)
            if self._teacher_hidden is not None
            else None
        )
        with torch.inference_mode():
            clean_feat = self._teacher_navigation_encoder(self._clean_depth)
            self._camera_clean_live_latent_cosine = F.cosine_similarity(
                clean_feat, nav_feat, dim=-1
            ).detach()
            teacher_input = assemble_actor_input(
                clean_feat, nav_nonvisual, profile, confidence
            )
            _, normalized, _, _, self._teacher_hidden = (
                self._teacher_actor.deterministic(
                    teacher_input,
                    self._teacher_hidden,
                    reset,
                    hard_abs_vy=p4_contract.P4_MAX_ABS_VY,
                )
            )
            fault_normalized = normalized
            self._diagnostic_fault_nav_feat.copy_(clean_feat)
            if self._diagnostic_fault_depth is not None and bool(
                self._diagnostic_fault_mask.any()
            ):
                selected = self._diagnostic_fault_mask
                fault_feat = self._teacher_navigation_encoder(
                    self._diagnostic_fault_depth[selected]
                )
                self._diagnostic_fault_nav_feat[selected] = fault_feat
                fault_input = assemble_actor_input(
                    self._diagnostic_fault_nav_feat,
                    nav_nonvisual,
                    profile,
                    confidence,
                )
                _, fault_normalized, _, _, _ = self._teacher_actor.deterministic(
                    fault_input,
                    teacher_hidden_before,
                    reset,
                    hard_abs_vy=p4_contract.P4_MAX_ABS_VY,
                )
                self._diagnostic_clean_fault_latent_cosine = F.cosine_similarity(
                    clean_feat,
                    self._diagnostic_fault_nav_feat,
                    dim=-1,
                ).detach()
                self._diagnostic_clean_fault_action_mae = (
                    (normalized - fault_normalized).abs().mean(dim=-1).detach()
                )
            else:
                self._diagnostic_clean_fault_latent_cosine.fill_(1.0)
                self._diagnostic_clean_fault_action_mae.zero_()
            zero_hidden = (
                torch.zeros(
                    self._teacher_actor.num_layers,
                    teacher_input.shape[0],
                    self._teacher_actor.hidden_dim,
                    device=teacher_input.device,
                    dtype=teacher_input.dtype,
                ),
                torch.zeros(
                    self._teacher_actor.num_layers,
                    teacher_input.shape[0],
                    self._teacher_actor.hidden_dim,
                    device=teacher_input.device,
                    dtype=teacher_input.dtype,
                ),
            )
            _, zero_normalized, _, _, _ = self._teacher_actor.deterministic(
                teacher_input,
                zero_hidden,
                torch.ones_like(reset, dtype=torch.bool),
                hard_abs_vy=p4_contract.P4_MAX_ABS_VY,
            )
            self._parent_anchor_hidden = self._mask_hidden(
                self._parent_anchor_hidden, reset
            )
            parent_input = assemble_actor_input(
                nav_feat.detach(), nav_nonvisual, profile, confidence
            )
            (
                _,
                parent_normalized,
                _parent_mean,
                parent_log_std,
                self._parent_anchor_hidden,
            ) = self._parent_anchor_actor.deterministic(
                parent_input,
                self._parent_anchor_hidden,
                reset,
                hard_abs_vy=p4_contract.P4_MAX_ABS_VY,
            )
            self._parent_anchor_normalized_mean.copy_(parent_normalized)
            self._parent_anchor_log_std.copy_(parent_log_std)
        self._clean_action_mean = normalized.detach()
        if self.safety_head is None:
            self._last_safety_head_risk3.zero_()
        else:
            self._last_safety_head_risk3 = torch.sigmoid(
                self.safety_head(nav_feat.detach())
            ).detach()
        zero_delta = (normalized - zero_normalized).abs()
        self._zero_hidden_action_mae = zero_delta.mean(dim=-1).detach()
        self._zero_hidden_direction_disagreement = (
            (torch.sign(normalized[:, 2]) != torch.sign(zero_normalized[:, 2]))
            .float()
            .detach()
        )
        self._camera_aux_mask = torch.stack(
            (
                self._camera_diagnostics["camera_delay_only"],
                self._camera_diagnostics["camera_fault_only"],
                self._camera_diagnostics["camera_fault_delay_overlap"],
            ),
            dim=-1,
        ).detach()

    def _auxiliary_base_context(
        self, normalized_mean, actor_normalized_mean, actor_log_std, batch
    ) -> dict[str, object]:
        zero = normalized_mean.new_zeros(())
        if actor_normalized_mean is None:
            actor_normalized_mean = normalized_mean
        if actor_log_std is None:
            actor_log_std = torch.zeros_like(actor_normalized_mean)
        valid_rows = batch.get("valid_mask")
        if valid_rows is None:
            valid_rows = torch.ones(
                actor_normalized_mean.shape[:-1],
                dtype=torch.bool,
                device=actor_normalized_mean.device,
            )
        else:
            valid_rows = (
                valid_rows.reshape(actor_normalized_mean.shape[:-1]).to(
                    device=actor_normalized_mean.device
                )
                > 0.5
            )
        valid_flat = valid_rows.reshape(-1)
        schedule = p4_contract.training_schedule(
            self.session_effective_seconds,
            branch=self._effective_maze_branch(self.session_effective_seconds),
        )
        specs: list[tuple[str, torch.Tensor, float]] = []
        mask3 = (batch["camera_aux_mask"] > 0.5) & valid_rows.unsqueeze(-1)
        camera_selected = mask3.any(dim=-1)
        camera_raw = zero
        if bool(camera_selected.any()):
            camera_raw = (
                F.smooth_l1_loss(
                    normalized_mean,
                    batch["clean_action_mean"],
                    reduction="none",
                )
                .mean(dim=-1)[camera_selected]
                .mean()
            )
            specs.append(
                ("camera", camera_raw, float(schedule.get("camera_aux_ratio", 0.01)))
            )
        return {
            "zero": zero,
            "normalized_mean": normalized_mean,
            "actor_normalized_mean": actor_normalized_mean,
            "actor_log_std": actor_log_std,
            "batch": batch,
            "valid_rows": valid_rows,
            "valid_flat": valid_flat,
            "schedule": schedule,
            "specs": specs,
            "mask3": mask3,
            "camera_selected": camera_selected,
            "camera_raw": camera_raw,
        }

    def _teacher_auxiliary_context(self, context: dict[str, object]) -> None:
        zero = context["zero"]
        batch = context["batch"]
        valid_rows = context["valid_rows"]
        valid_flat = context["valid_flat"]
        actor_normalized_mean = context["actor_normalized_mean"]
        schedule = context["schedule"]
        specs = context["specs"]
        teacher_raw = zero
        teacher = {
            "direction": zero,
            "speed": zero,
            "yaw": zero,
            "edge": zero,
            "recovery": zero,
            "stale_goal": zero,
            "near_goal": zero,
            "teacher_valid_steps": zero,
            "teacher_loss_active": zero,
            "teacher_edge_active_share": zero,
            "teacher_recovery_active_share": zero,
            "teacher_stale_goal_active_share": zero,
            "teacher_near_goal_active_share": zero,
        }
        teacher_mask = (batch["teacher_mask"].reshape(-1) > 0.5) & valid_flat
        teacher_goal_mask = (batch["teacher_goal_mask"].reshape(-1) > 0.5) & valid_flat
        teacher_context_mask = (
            batch.get("teacher_context_mask", batch["teacher_mask"]).reshape(-1) > 0.5
        ) & valid_flat
        teacher_update_mask = teacher_mask
        if self.training_profile == PROFILE_MAZE_INSTANT_REPAIR2H:
            teacher_update_mask = teacher_update_mask | teacher_context_mask
        if self._teacher_update_enabled and bool(teacher_update_mask.any()):
            full_cap = torch.full(
                actor_normalized_mean.shape[:-1],
                p4_contract.P4_MAX_VX,
                dtype=actor_normalized_mean.dtype,
                device=actor_normalized_mean.device,
            )
            mean_command = p4_contract.map_normalized_action(
                actor_normalized_mean, full_cap, goal_freshness=None
            ).reshape(-1, 3)
            teacher_safe5 = (
                batch["teacher_safe5"]
                if self.training_profile
                in {
                    PROFILE_MAZE_CLOSED_LOOP_V3,
                    PROFILE_MAZE_INSTANT_COMMAND_R4,
                    PROFILE_MAZE_INSTANT_REPAIR2H,
                    PROFILE_MAZE_STABLE_DIRECTION_SMOKE,
                    PROFILE_MAZE_STABLE_DIRECTION_8H,
                }
                else batch["teacher_safe3"]
            )
            teacher = p4_contract.teacher_guidance_loss(
                mean_command,
                teacher_safe5.reshape(-1, teacher_safe5.shape[-1]),
                batch["teacher_goal_xy"].reshape(-1, 2),
                batch["teacher_predictive_risk"].reshape(-1),
                batch["stuck_label"].reshape(-1) > 0.5,
                teacher_mask,
                teacher_goal_mask,
                sample_weight=batch["teacher_weight"].reshape(-1),
                goal_freshness=batch.get(
                    "teacher_goal_freshness",
                    torch.ones_like(batch["teacher_mask"]),
                ).reshape(-1),
                context_mask=batch.get(
                    "teacher_context_mask", batch["teacher_mask"]
                ).reshape(-1)
                > 0.5,
                min_valid_steps=1,
                closed_loop_v3=(self.training_profile == PROFILE_MAZE_CLOSED_LOOP_V3),
                instant_r4=(self.training_profile in self.INSTANT_PROFILES),
                instant_repair=(self.training_profile == PROFILE_MAZE_INSTANT_REPAIR2H),
            )
            edge_mask = teacher.get("teacher_edge_mask")
            if edge_mask is not None:
                teacher["teacher_edge_active_share"] = self._masked_mean(
                    edge_mask.reshape(valid_rows.shape), valid_rows
                )
            recovery_mask = teacher.get("teacher_recovery_mask")
            if recovery_mask is not None:
                teacher["teacher_recovery_active_share"] = self._masked_mean(
                    recovery_mask.reshape(valid_rows.shape), valid_rows
                )
            teacher_raw = teacher["loss"]
            specs.append(
                (
                    "teacher",
                    teacher_raw,
                    min(
                        float(schedule.get("teacher_gradient_target_ratio", 0.02)),
                        float(schedule.get("teacher_gradient_hard_cap", 0.03)),
                    ),
                )
            )
        context.update(teacher=teacher, teacher_raw=teacher_raw)

    def _anchor_auxiliary_context(self, context: dict[str, object]) -> None:
        zero = context["zero"]
        batch = context["batch"]
        actor_normalized_mean = context["actor_normalized_mean"]
        actor_log_std = context["actor_log_std"]
        valid_flat = context["valid_flat"]
        schedule = context["schedule"]
        specs = context["specs"]
        anchor_raw = zero
        anchor_mask = (
            batch["parent_anchor_mask"].reshape(-1) > 0.5
            if "parent_anchor_mask" in batch
            else torch.zeros(
                actor_normalized_mean.numel() // actor_normalized_mean.shape[-1],
                dtype=torch.bool,
                device=actor_normalized_mean.device,
            )
        )
        anchor_mask &= valid_flat
        if bool(anchor_mask.any()):
            current_mean = actor_normalized_mean.reshape(-1, 3)[anchor_mask]
            parent_mean = batch["parent_normalized_mean"].reshape(-1, 3)[anchor_mask]
            current_log_std = actor_log_std.reshape(-1, 3)[anchor_mask]
            parent_log_std = batch["parent_log_std"].reshape(-1, 3)[anchor_mask]
            anchor_raw = F.smooth_l1_loss(current_mean, parent_mean)
            anchor_raw = anchor_raw + 0.10 * F.smooth_l1_loss(
                current_log_std, parent_log_std
            )
            specs.append(
                (
                    "anchor",
                    anchor_raw,
                    min(
                        float(schedule.get("anchor_target_ratio", 0.0)),
                        float(schedule.get("anchor_gradient_hard_cap", 0.015)),
                    ),
                )
            )
        context.update(anchor_raw=anchor_raw, anchor_mask=anchor_mask)

    def _stuck_auxiliary_context(
        self, context: dict[str, object], actor_features
    ) -> None:
        zero = context["zero"]
        batch = context["batch"]
        valid_flat = context["valid_flat"]
        schedule = context["schedule"]
        specs = context["specs"]
        stuck_raw = zero
        stuck_precision = zero
        stuck_recall = zero
        stuck_f1 = zero
        stuck_pr_auc = zero
        stuck_threshold = zero.new_tensor(0.5)
        stuck_positive_share = zero
        stuck_mask = (batch["stuck_mask"].reshape(-1) > 0.5) & valid_flat
        if (
            actor_features is not None
            and bool(stuck_mask.any())
            and self.stuck_head is not None
        ):
            stuck_frozen = self.training_profile in {
                PROFILE_MAZE_CLOSED_LOOP_V3,
                PROFILE_MAZE_INSTANT_COMMAND_R4,
                PROFILE_MAZE_INSTANT_REPAIR2H,
                PROFILE_MAZE_STABLE_DIRECTION_SMOKE,
                PROFILE_MAZE_STABLE_DIRECTION_8H,
            }
            if stuck_frozen:
                # Keep frozen-head quality metrics without an Actor graph.
                with torch.no_grad():
                    stuck_logits = self.stuck_head(actor_features.detach()).reshape(-1)
            else:
                stuck_logits = self.stuck_head(actor_features).reshape(-1)
            labels = batch["stuck_label"].reshape(-1)
            selected_labels = labels[stuck_mask]
            positive = selected_labels.mean().detach()
            self.actor_stuck_positive_ema = float(
                0.99 * self.actor_stuck_positive_ema + 0.01 * positive
            )
            pos_weight = min(
                10.0,
                max(
                    1.0,
                    (1.0 - self.actor_stuck_positive_ema)
                    / max(self.actor_stuck_positive_ema, 1.0e-4),
                ),
            )
            if stuck_frozen:
                with torch.no_grad():
                    stuck_raw = F.binary_cross_entropy_with_logits(
                        stuck_logits[stuck_mask],
                        selected_labels,
                        pos_weight=torch.tensor(pos_weight, device=stuck_logits.device),
                    )
            else:
                stuck_raw = F.binary_cross_entropy_with_logits(
                    stuck_logits[stuck_mask],
                    selected_labels,
                    pos_weight=torch.tensor(pos_weight, device=stuck_logits.device),
                )
            with torch.no_grad():
                probabilities = torch.sigmoid(stuck_logits[stuck_mask])
                labels_bool = selected_labels > 0.5
                predicted = probabilities >= 0.5
                true_positive = (predicted & labels_bool).float().sum()
                false_positive = (predicted & ~labels_bool).float().sum()
                false_negative = (~predicted & labels_bool).float().sum()
                stuck_precision = true_positive / (
                    true_positive + false_positive
                ).clamp_min(1.0)
                stuck_recall = true_positive / (
                    true_positive + false_negative
                ).clamp_min(1.0)
                stuck_f1 = (
                    2.0
                    * stuck_precision
                    * stuck_recall
                    / (stuck_precision + stuck_recall).clamp_min(1.0e-6)
                )
                stuck_positive_share = labels_bool.float().mean()
                order = torch.argsort(probabilities, descending=True)
                sorted_labels = labels_bool[order].float()
                cumulative_positive = sorted_labels.cumsum(dim=0)
                ranks = torch.arange(
                    1,
                    sorted_labels.numel() + 1,
                    device=sorted_labels.device,
                    dtype=sorted_labels.dtype,
                )
                precision_curve = cumulative_positive / ranks
                positive_count = sorted_labels.sum()
                stuck_pr_auc = (
                    precision_curve * sorted_labels
                ).sum() / positive_count.clamp_min(1.0)
                recall_curve = cumulative_positive / positive_count.clamp_min(1.0)
                f1_curve = (
                    2.0
                    * precision_curve
                    * recall_curve
                    / (precision_curve + recall_curve).clamp_min(1.0e-6)
                )
                if sorted_labels.numel() and float(positive_count) > 0.0:
                    best_index = int(f1_curve.argmax())
                    stuck_threshold = probabilities[order[best_index]]
            stuck_target = float(schedule.get("stuck_gradient_target_ratio", 0.0))
            if stuck_target > 0.0 and stuck_raw.requires_grad:
                specs.append(("stuck", stuck_raw, stuck_target))
        context.update(
            stuck_raw=stuck_raw,
            stuck_precision=stuck_precision,
            stuck_recall=stuck_recall,
            stuck_f1=stuck_f1,
            stuck_pr_auc=stuck_pr_auc,
            stuck_threshold=stuck_threshold,
            stuck_positive_share=stuck_positive_share,
            stuck_mask=stuck_mask,
        )

    def _mirror_auxiliary_context(self, context: dict[str, object]) -> None:
        zero = context["zero"]
        normalized_mean = context["normalized_mean"]
        batch = context["batch"]
        schedule = context["schedule"]
        specs = context["specs"]
        mirror_raw = zero
        mirror_sequence_count = 0
        mirror_batch = batch.get("mirror_batch")
        if isinstance(mirror_batch, dict) and "depth" in mirror_batch:
            depth = mirror_batch["depth"]
            mirror_sequence_count = int(depth.shape[1])
            if mirror_sequence_count:
                flat_depth = depth.reshape(
                    -1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
                )
                mirrored_depth = depth.flip(dims=(-2,)).reshape(
                    -1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
                )
                amp_enabled = self.device.type == "cuda"
                if not amp_enabled:
                    flat_depth = flat_depth.float()
                    mirrored_depth = mirrored_depth.float()
                with torch.no_grad(), torch.autocast(
                    device_type=self.device.type, enabled=amp_enabled
                ):
                    original_feat = (
                        self.navigation_encoder(flat_depth)
                        .float()
                        .reshape(depth.shape[0], depth.shape[1], -1)
                    )
                    mirrored_feat = (
                        self.navigation_encoder(mirrored_depth)
                        .float()
                        .reshape(depth.shape[0], depth.shape[1], -1)
                    )
                original_input = assemble_actor_input(
                    original_feat.detach(),
                    mirror_batch["nav_nonvisual"],
                    mirror_batch["response_profile"],
                    mirror_batch["confidence"],
                )
                mirrored_input = assemble_actor_input(
                    mirrored_feat.detach(),
                    self._mirror_nav_nonvisual(mirror_batch["nav_nonvisual"]),
                    self._mirror_response_profile(mirror_batch["response_profile"]),
                    mirror_batch["confidence"],
                )
                zero_hidden = (
                    torch.zeros(
                        self.actor.num_layers,
                        mirror_sequence_count,
                        self.actor.hidden_dim,
                        device=normalized_mean.device,
                        dtype=normalized_mean.dtype,
                    ),
                    torch.zeros(
                        self.actor.num_layers,
                        mirror_sequence_count,
                        self.actor.hidden_dim,
                        device=normalized_mean.device,
                        dtype=normalized_mean.dtype,
                    ),
                )
                with torch.no_grad():
                    _, _, original_mean, _, _, _ = self.actor.evaluate_actions(
                        original_input,
                        mirror_batch["pre_tanh_action"],
                        zero_hidden,
                        mirror_batch["reset_mask"],
                        return_features=True,
                    )
                _, _, mirrored_mean, _, _, _ = self.actor.evaluate_actions(
                    mirrored_input,
                    mirror_batch["pre_tanh_action"],
                    zero_hidden,
                    mirror_batch["reset_mask"],
                    return_features=True,
                )
                expected = torch.tanh(original_mean).detach().clone()
                expected[..., 1:].neg_()
                mirror_raw = F.smooth_l1_loss(torch.tanh(mirrored_mean), expected)
                specs.append(
                    (
                        "mirror",
                        mirror_raw,
                        min(
                            float(schedule.get("mirror_gradient_target_ratio", 0.005)),
                            float(schedule.get("mirror_gradient_hard_cap", 0.01)),
                        ),
                    )
                )
        context.update(
            mirror_raw=mirror_raw, mirror_sequence_count=mirror_sequence_count
        )

    def _calibrated_auxiliary_context(
        self, context: dict[str, object], ppo_actor_loss
    ) -> None:
        zero = context["zero"]
        schedule = context["schedule"]
        specs = context["specs"]
        teacher = context["teacher"]
        stuck_mask = context["stuck_mask"]
        for name, raw, _target in specs:
            self._assert_loss_parameter_ownership(name, raw)
        parameters = [
            parameter
            for module in (self.navigation_encoder, self.actor, self.stuck_head)
            if module is not None
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        stored_ratios = {
            name: float(self._auxiliary_calibration.get(f"{name}_ratio", 0.0))
            for name in ("camera", "teacher", "anchor", "mirror", "stuck")
        }
        # Detached diagnostic terms remain visible but are not calibrated.
        uncalibrated = [
            spec
            for spec in specs
            if spec[0] not in self._auxiliary_coefficients
            and bool(spec[1].requires_grad)
        ]
        head_only_calibrated = []
        head_only_names: set[str] = set()
        if uncalibrated and not ppo_actor_loss.requires_grad:
            # During creditwarm the Actor is frozen but StuckHead still trains.
            for name, raw, _target in uncalibrated:
                if name == "stuck" and raw.requires_grad:
                    head_only_calibrated.append(
                        {
                            "name": name,
                            "loss": raw,
                            "multiplier": 1.0,
                            "component_ratio": 0.0,
                        }
                    )
                    head_only_names.add(name)
            uncalibrated = []
        if uncalibrated:
            remaining = max(
                0.0,
                float(schedule.get("auxiliary_gradient_hard_cap", 0.05))
                - sum(stored_ratios.values()),
            )
            newly_calibrated, _new_ratio = _calibrate_auxiliary_gradients(
                ppo_actor_loss,
                uncalibrated,
                parameters,
                remaining,
            )
            for item in newly_calibrated:
                name = str(item["name"])
                self._auxiliary_coefficients[name] = float(item["multiplier"])
                stored_ratios[name] = float(item["component_ratio"])
        self._camera_aux_calibration_pending = False
        calibrated = head_only_calibrated + [
            {
                "name": name,
                "loss": raw,
                "multiplier": float(self._auxiliary_coefficients.get(name, 0.0)),
                "component_ratio": stored_ratios.get(name, 0.0),
            }
            for name, raw, _target in specs
            if name not in head_only_names
        ]
        combined_ratio = min(
            float(schedule.get("auxiliary_gradient_hard_cap", 0.05)),
            sum(stored_ratios.values()),
        )
        multipliers = {item["name"]: float(item["multiplier"]) for item in calibrated}
        ratios = {item["name"]: float(item["component_ratio"]) for item in calibrated}
        loss = sum((item["loss"] * item["multiplier"] for item in calibrated), zero)
        self._auxiliary_calibration = {
            "combined_ratio": float(combined_ratio),
            "teacher_ratio": stored_ratios.get("teacher", 0.0),
            "anchor_ratio": stored_ratios.get("anchor", 0.0),
            "camera_ratio": stored_ratios.get("camera", 0.0),
            "mirror_ratio": stored_ratios.get("mirror", 0.0),
            "stuck_ratio": stored_ratios.get("stuck", 0.0),
            "teacher_valid_steps": float(teacher["teacher_valid_steps"].detach()),
            "stuck_valid_steps": float(stuck_mask.sum()),
        }
        self._camera_aux_coefficient = multipliers.get("camera", 0.0)
        self._camera_aux_gradient_ratio = ratios.get("camera", 0.0)
        context.update(loss=loss, ratios=ratios, combined_ratio=combined_ratio)

    def _trainable_parameter_groups(self) -> dict[str, tuple[torch.nn.Parameter, ...]]:
        groups = {}
        for name in ("navigation_encoder", "actor", "safety_head", "stuck_head"):
            module = getattr(self, name, None)
            groups[name] = (
                tuple(
                    parameter
                    for parameter in module.parameters()
                    if parameter.requires_grad
                )
                if module is not None
                else ()
            )
        return groups

    def _assert_loss_parameter_ownership(self, name: str, loss: torch.Tensor) -> None:
        if not loss.requires_grad or not torch.is_grad_enabled():
            return
        spec = AUXILIARY_PARAMETER_SPECS[name]
        groups = self._trainable_parameter_groups()
        forbidden = [
            (group_name, parameter)
            for group_name, parameters in groups.items()
            if group_name not in spec.allowed_groups
            for parameter in parameters
        ]
        if not forbidden:
            return
        gradients = torch.autograd.grad(
            loss,
            [parameter for _group_name, parameter in forbidden],
            allow_unused=True,
            retain_graph=True,
        )
        crossed = sorted(
            {
                group_name
                for (group_name, _parameter), gradient in zip(forbidden, gradients)
                if gradient is not None
            }
        )
        if crossed:
            raise RuntimeError(
                f"{name} loss crossed auxiliary parameter boundary: {crossed}"
            )

    def _actor_auxiliary_loss(
        self,
        *,
        normalized_mean,
        actor_normalized_mean=None,
        actor_log_std=None,
        batch,
        ppo_actor_loss,
        actor_features=None,
        nav_feat=None,
    ):
        del nav_feat
        context = self._auxiliary_base_context(
            normalized_mean, actor_normalized_mean, actor_log_std, batch
        )
        zero = context["zero"]
        actor_normalized_mean = context["actor_normalized_mean"]
        actor_log_std = context["actor_log_std"]
        valid_rows = context["valid_rows"]
        valid_flat = context["valid_flat"]
        schedule = context["schedule"]
        specs = context["specs"]
        mask3 = context["mask3"]
        camera_selected = context["camera_selected"]
        camera_raw = context["camera_raw"]

        self._teacher_auxiliary_context(context)
        teacher = context["teacher"]
        teacher_raw = context["teacher_raw"]

        self._anchor_auxiliary_context(context)
        anchor_raw = context["anchor_raw"]
        anchor_mask = context["anchor_mask"]

        self._stuck_auxiliary_context(context, actor_features)
        stuck_raw = context["stuck_raw"]
        stuck_precision = context["stuck_precision"]
        stuck_recall = context["stuck_recall"]
        stuck_f1 = context["stuck_f1"]
        stuck_pr_auc = context["stuck_pr_auc"]
        stuck_threshold = context["stuck_threshold"]
        stuck_positive_share = context["stuck_positive_share"]
        stuck_mask = context["stuck_mask"]

        self._mirror_auxiliary_context(context)
        mirror_raw = context["mirror_raw"]
        mirror_sequence_count = context["mirror_sequence_count"]

        self._calibrated_auxiliary_context(context, ppo_actor_loss)
        ratios = context["ratios"]
        combined_ratio = context["combined_ratio"]
        return context["loss"], {
            "camera_memory_loss": camera_raw.detach(),
            "camera_clean_live_action_mae": (
                (
                    normalized_mean[camera_selected]
                    - batch["clean_action_mean"][camera_selected]
                )
                .abs()
                .mean()
                .detach()
                if bool(camera_selected.any())
                else zero.detach()
            ),
            "camera_aux_coefficient": zero.new_tensor(self._camera_aux_coefficient),
            "camera_aux_gradient_ratio": zero.new_tensor(
                self._camera_aux_gradient_ratio
            ),
            "camera_delay_only_share": self._masked_mean(
                mask3[..., 0].float(), valid_rows
            ).detach(),
            "camera_fault_only_share": self._masked_mean(
                mask3[..., 1].float(), valid_rows
            ).detach(),
            "camera_fault_delay_overlap_share": self._masked_mean(
                mask3[..., 2].float(), valid_rows
            ).detach(),
            "teacher_guidance_loss": teacher_raw.detach(),
            "teacher_guidance_valid_steps": teacher["teacher_valid_steps"].detach(),
            "teacher_guidance_gradient_ratio": zero.new_tensor(
                ratios.get("teacher", 0.0)
            ),
            "parent_anchor_loss": anchor_raw.detach(),
            "parent_anchor_valid_steps": anchor_mask.to(zero.dtype).sum().detach(),
            "parent_anchor_valid_share": (
                anchor_mask.to(zero.dtype).sum()
                / valid_flat.to(zero.dtype).sum().clamp_min(1.0)
            ).detach(),
            "parent_anchor_gradient_ratio": zero.new_tensor(ratios.get("anchor", 0.0)),
            "stuck_aux_loss": stuck_raw.detach(),
            "stuck_aux_valid_steps": zero.new_tensor(float(stuck_mask.sum())),
            "stuck_aux_gradient_ratio": zero.new_tensor(ratios.get("stuck", 0.0)),
            "actor_stuck_pr_auc": stuck_pr_auc.detach(),
            "actor_stuck_precision": stuck_precision.detach(),
            "actor_stuck_recall": stuck_recall.detach(),
            "actor_stuck_f1": stuck_f1.detach(),
            "actor_stuck_threshold": stuck_threshold.detach(),
            "actor_stuck_positive_share": stuck_positive_share.detach(),
            "mirror_aux_loss": mirror_raw.detach(),
            "mirror_aux_sequence_share": zero.new_tensor(
                float(mirror_sequence_count) / max(normalized_mean.shape[1], 1)
            ),
            "mirror_aux_eligible_sequence_count": zero.new_tensor(
                float(self._mirror_aux_eligible_sequence_count)
            ),
            "mirror_aux_scheduled_sequence_share": zero.new_tensor(
                float(self._mirror_aux_scheduled_sequence_share)
            ),
            "mirror_aux_gradient_ratio": zero.new_tensor(ratios.get("mirror", 0.0)),
            "auxiliary_gradient_ratio": zero.new_tensor(float(combined_ratio)),
            "teacher_direction_loss": teacher["direction"].detach(),
            "teacher_speed_loss": teacher["speed"].detach(),
            "teacher_yaw_loss": teacher["yaw"].detach(),
            "teacher_edge_loss": teacher.get("edge", zero).detach(),
            "teacher_edge_active_share": teacher.get(
                "teacher_edge_active_share", zero
            ).detach(),
            "teacher_recovery_loss": teacher.get("recovery", zero).detach(),
            "teacher_recovery_active_share": teacher.get(
                "teacher_recovery_active_share", zero
            ).detach(),
            "teacher_stale_goal_loss": teacher.get("stale_goal", zero).detach(),
            "teacher_stale_goal_active_share": teacher.get(
                "teacher_stale_goal_active_share", zero
            ).detach(),
            "teacher_near_goal_loss": teacher.get("near_goal", zero).detach(),
            "teacher_near_goal_active_share": teacher.get(
                "teacher_near_goal_active_share", zero
            ).detach(),
        }

    @staticmethod
    def _mirror_nav_nonvisual(values: torch.Tensor) -> torch.Tensor:
        mirrored = values.clone()
        # goal-y and target/executed/measured vy/wz channels.
        for index in (1, 5, 6, 8, 9, 11, 12):
            mirrored[..., index].neg_()
        # Angular velocity is an axial vector under a left/right reflection;
        # projected gravity is an ordinary vector.
        for index in (15, 17, 19):
            mirrored[..., index].neg_()
        return mirrored

    @staticmethod
    def _mirror_response_profile(values: torch.Tensor) -> torch.Tensor:
        mirrored = values.clone()
        for index in (1, 2, 4, 5, 7, 8):
            mirrored[..., index].neg_()
        mirrored[..., 10].neg_()
        mirrored[..., 11].neg_()
        return mirrored

    def _actor_micro_loss(self, batch):
        if "depth" in batch:
            depth = batch["depth"].reshape(
                -1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
            )
            amp_enabled = self.device.type == "cuda"
            if not amp_enabled:
                depth = depth.float()
            with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                feat = self.navigation_encoder(depth).float()
            feat = feat.reshape(batch["nav_nonvisual"].shape[0], -1, 32)
        else:
            feat = batch["nav_feat"]
        inputs = assemble_actor_input(
            feat, batch["nav_nonvisual"], batch["response_profile"], batch["confidence"]
        )
        log_prob, entropy, mean, _, _, _ = self.actor.evaluate_actions(
            inputs,
            batch["pre_tanh_action"],
            batch["actor_hidden"],
            batch["reset_mask"],
            return_features=True,
        )
        ratio = torch.exp(log_prob - batch["old_log_prob"])
        valid_mask = (
            batch.get("valid_mask", torch.ones_like(batch["old_log_prob"])) > 0.5
        )
        surrogate_element = -torch.minimum(
            ratio * batch["advantages"],
            torch.clamp(ratio, 0.8, 1.2) * batch["advantages"],
        )
        surrogate = self._masked_mean(surrogate_element, valid_mask)
        if getattr(self, "track_safety_enabled", True):
            safety_logits = self.safety_head(feat)
            safety_elements = F.binary_cross_entropy_with_logits(
                safety_logits, batch["safety_target"], reduction="none"
            )
            safety_mask = batch["safety_valid"].expand_as(safety_elements)
            safety_mask = safety_mask * valid_mask.expand_as(safety_elements)
            hard_positive = (batch["safety_target"] >= 0.65).any(dim=-1) | (
                batch["stuck_label"].squeeze(-1) > 0.5
            )
            safety_weight = 1.0 + hard_positive.float()
            weighted_safety_mask = safety_mask * safety_weight.unsqueeze(-1)
            safety_loss = safety_elements.mul(
                weighted_safety_mask
            ).sum() / weighted_safety_mask.sum().clamp_min(1.0)
            risk_probability = torch.sigmoid(safety_logits)
            student_risk = torch.stack(
                tuple(
                    self._masked_mean(risk_probability[..., index], valid_mask)
                    for index in range(3)
                )
            )
        else:
            safety_loss = surrogate.new_zeros(())
            student_risk = torch.zeros(3, device=surrogate.device)
            hard_positive = torch.zeros(
                batch["safety_target"].shape[:-1],
                dtype=torch.bool,
                device=surrogate.device,
            )
        self._assert_loss_parameter_ownership("safety", safety_loss)
        detached_inputs = assemble_actor_input(
            feat.detach(),
            batch["nav_nonvisual"],
            batch["response_profile"],
            batch["confidence"],
        )
        (
            _,
            _,
            auxiliary_mean,
            auxiliary_log_std,
            _,
            auxiliary_features,
        ) = self.actor.evaluate_actions(
            detached_inputs,
            batch["pre_tanh_action"],
            batch["actor_hidden"],
            batch["reset_mask"],
            return_features=True,
        )
        auxiliary_loss, auxiliary_metrics = self._actor_auxiliary_loss(
            normalized_mean=torch.tanh(mean),
            actor_normalized_mean=torch.tanh(auxiliary_mean),
            actor_log_std=auxiliary_log_std,
            actor_features=auxiliary_features,
            nav_feat=feat,
            batch=batch,
            ppo_actor_loss=surrogate,
        )
        auxiliary_metrics["safety_hard_positive_share"] = self._masked_mean(
            hard_positive.float(), valid_mask
        ).detach()
        total_loss = (
            surrogate
            - self.entropy_coefficient * self._masked_mean(entropy, valid_mask)
            + p2_contract.SAFETY_BCE_WEIGHT * safety_loss
            + auxiliary_loss
        )
        with torch.no_grad():
            log_ratio = log_prob - batch["old_log_prob"]
            approx_kl = self._masked_mean(
                (torch.exp(log_ratio) - 1.0) - log_ratio, valid_mask
            )
            clip_fraction = self._masked_mean(
                ((ratio - 1.0).abs() > 0.2).float(), valid_mask
            )
            entropy_mean = self._masked_mean(entropy, valid_mask)
        return total_loss, {
            "surrogate_loss": surrogate.detach(),
            "entropy": entropy_mean.detach(),
            "approx_kl": approx_kl.detach(),
            "clip_fraction": clip_fraction.detach(),
            "safety_bce": safety_loss.detach(),
            "scanner_valid_share": self._masked_mean(
                batch["safety_valid"].float(), valid_mask
            ).detach(),
            "safety_head_risk_left": student_risk[0].detach(),
            "safety_head_risk_center": student_risk[1].detach(),
            "safety_head_risk_right": student_risk[2].detach(),
            **auxiliary_metrics,
        }

    def _actor_auxiliary_metric_names(self) -> tuple[str, ...]:
        return (
            "camera_memory_loss",
            "camera_clean_live_action_mae",
            "camera_aux_coefficient",
            "camera_aux_gradient_ratio",
            "camera_delay_only_share",
            "camera_fault_only_share",
            "camera_fault_delay_overlap_share",
            "teacher_guidance_loss",
            "teacher_guidance_valid_steps",
            "teacher_guidance_gradient_ratio",
            "parent_anchor_loss",
            "parent_anchor_valid_steps",
            "parent_anchor_valid_share",
            "parent_anchor_gradient_ratio",
            "teacher_direction_loss",
            "teacher_speed_loss",
            "teacher_yaw_loss",
            "teacher_edge_loss",
            "teacher_edge_active_share",
            "teacher_recovery_loss",
            "teacher_recovery_active_share",
            "teacher_stale_goal_loss",
            "teacher_stale_goal_active_share",
            "teacher_near_goal_loss",
            "teacher_near_goal_active_share",
            "stuck_aux_loss",
            "stuck_aux_valid_steps",
            "stuck_aux_gradient_ratio",
            "actor_stuck_pr_auc",
            "actor_stuck_precision",
            "actor_stuck_recall",
            "actor_stuck_f1",
            "actor_stuck_threshold",
            "actor_stuck_positive_share",
            "mirror_aux_loss",
            "mirror_aux_sequence_share",
            "mirror_aux_eligible_sequence_count",
            "mirror_aux_scheduled_sequence_share",
            "mirror_aux_gradient_ratio",
            "auxiliary_gradient_ratio",
            "safety_hard_positive_share",
        )

    def _actor_sequence_batch(self, refs, *, advantage_mean, advantage_std):
        batch = super()._actor_sequence_batch(
            refs, advantage_mean=advantage_mean, advantage_std=advantage_std
        )
        for name in (
            "teacher_safe3",
            "teacher_safe5",
            "teacher_goal_xy",
            "teacher_goal_freshness",
            "teacher_predictive_risk",
            "teacher_mask",
            "teacher_context_mask",
            "teacher_goal_mask",
            "teacher_weight",
            "stuck_label",
            "stuck_mask",
            "mirror_eligible",
            "parent_normalized_mean",
            "parent_log_std",
            "parent_anchor_mask",
        ):
            batch[name] = self._stack_refs(name, refs)
        valid = batch["valid_mask"]
        for name in (
            "safety_valid",
            "camera_aux_mask",
            "teacher_mask",
            "teacher_context_mask",
            "teacher_goal_mask",
            "teacher_weight",
            "stuck_mask",
            "mirror_eligible",
            "parent_anchor_mask",
        ):
            batch[name] = batch[name] * valid
        if "continuation_mask" in batch:
            nonterminal = batch["continuation_mask"] > 0.5
            batch["teacher_mask"] = batch["teacher_mask"] * nonterminal
            batch["teacher_context_mask"] = batch["teacher_context_mask"] * nonterminal
            batch["teacher_goal_mask"] = batch["teacher_goal_mask"] * nonterminal
        if self._mirror_batch_cursor < len(self._mirror_batch_schedule):
            mirror_ref = self._mirror_batch_schedule[self._mirror_batch_cursor]
            self._mirror_batch_cursor += 1
            if mirror_ref is not None:
                batch["mirror_batch"] = {
                    name: self._stack_refs(name, [mirror_ref])
                    for name in (
                        "depth",
                        "nav_nonvisual",
                        "response_profile",
                        "confidence",
                        "pre_tanh_action",
                        "reset_mask",
                    )
                }
        return batch

    def _prepare_mirror_batch_schedule(self) -> None:
        """Schedule episode-aligned mirror sequences across actor microbatches."""
        self._mirror_batch_schedule = []
        self._mirror_batch_cursor = 0
        self._mirror_aux_eligible_sequence_count = 0
        self._mirror_aux_scheduled_sequence_share = 0.0
        if self.rollout.store_depth:
            pool = self.rollout.episode_aligned_refs(
                eligibility=self.rollout.mirror_eligible[: self.rollout.step]
            )
            self._mirror_aux_eligible_sequence_count = len(pool)
            fixed_sequence_count = (
                self.num_envs * self.rollout.num_ticks // self.rollout.sequence_length
            )
            minibatch_sequences = max(
                1,
                (fixed_sequence_count + self.num_mini_batches - 1)
                // self.num_mini_batches,
            )
            micro_calls_per_epoch = sum(
                (
                    min(minibatch_sequences, fixed_sequence_count - start)
                    + self.micro_sequences
                    - 1
                )
                // self.micro_sequences
                for start in range(0, fixed_sequence_count, minibatch_sequences)
            )
            total_micro_calls = micro_calls_per_epoch * self.num_learning_epochs
            target_sequences = min(
                total_micro_calls,
                int(
                    round(
                        fixed_sequence_count
                        * self.num_learning_epochs
                        * float(
                            p4_contract.training_schedule(
                                self.session_effective_seconds,
                                branch=self._effective_maze_branch(
                                    self.session_effective_seconds
                                ),
                            ).get("mirror_sequence_share", 0.10)
                        )
                    )
                ),
            )
            if pool and target_sequences > 0:
                selected_slots = torch.randperm(
                    total_micro_calls, generator=self.mirror_generator
                )[:target_sequences].tolist()
                selected_refs = torch.randint(
                    len(pool),
                    (target_sequences,),
                    generator=self.mirror_generator,
                ).tolist()
                schedule: list[object | None] = [None] * total_micro_calls
                for slot, ref_index in zip(selected_slots, selected_refs):
                    schedule[int(slot)] = pool[int(ref_index)]
                self._mirror_batch_schedule = schedule
                self._mirror_aux_scheduled_sequence_share = float(
                    target_sequences
                ) / max(
                    fixed_sequence_count * self.num_learning_epochs,
                    1,
                )
