#!/usr/bin/env python3
"""P4 diagnostic probes, tick metrics, and memory metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType

import torch
import torch.nn.functional as F
from torch import nn

from agent_ppo.feature import p2_contract, p3_contract, p4_contract
from agent_ppo.p4.profiles import (
    PROFILE_FULL_TRACK,
    PROFILE_MAZE_CLOSED_LOOP_V3,
    PROFILE_MAZE_CREDIT_REPAIR,
    PROFILE_MAZE_INSTANT_COMMAND_R4,
    PROFILE_MAZE_INSTANT_REPAIR2H,
    PROFILE_MAZE_STABLE_DIRECTION_SMOKE,
    PROFILE_MAZE_STABLE_DIRECTION_8H,
)
from agent_ppo.p4.primitives import safety_scene_diagnostics


@dataclass(frozen=True)
class DiagnosticSpec:
    maze_probe: bool
    full_track_segments: bool


DIAGNOSTIC_SPECS = MappingProxyType(
    {
        PROFILE_FULL_TRACK: DiagnosticSpec(False, True),
        PROFILE_MAZE_CREDIT_REPAIR: DiagnosticSpec(True, False),
        PROFILE_MAZE_CLOSED_LOOP_V3: DiagnosticSpec(True, False),
        PROFILE_MAZE_INSTANT_COMMAND_R4: DiagnosticSpec(True, False),
        PROFILE_MAZE_INSTANT_REPAIR2H: DiagnosticSpec(True, False),
        PROFILE_MAZE_STABLE_DIRECTION_SMOKE: DiagnosticSpec(True, False),
        PROFILE_MAZE_STABLE_DIRECTION_8H: DiagnosticSpec(True, False),
    }
)


class P4DiagnosticsMixin:
    """Behavior-preserving methods extracted from AlgorithmP4NavPPO."""

    def _reset_maze_diagnostic_state(self) -> None:
        self._maze_diag_total = torch.zeros((), device=self.device)
        self._maze_diag_valid = torch.zeros((), device=self.device)
        self._maze_diag_wall_positive = torch.zeros((), device=self.device)
        self._maze_diag_wall_missed = torch.zeros((), device=self.device)
        self._maze_diag_top1_total = torch.zeros((), device=self.device)
        self._maze_diag_top1_correct = torch.zeros((), device=self.device)
        self._maze_diag_risk_positive_hist = torch.zeros(20, device=self.device)
        self._maze_diag_risk_negative_hist = torch.zeros(20, device=self.device)
        self._maze_diag_scene_confusion = torch.zeros(5, 6, device=self.device)
        self._maze_diag_latent_cosine_sum = torch.zeros((), device=self.device)
        self._maze_diag_latent_cosine_count = torch.zeros((), device=self.device)
        self._maze_diag_goal_risk_positive_hist = torch.zeros(20, device=self.device)
        self._maze_diag_goal_risk_negative_hist = torch.zeros(20, device=self.device)
        self._maze_diag_goal_top1_total = torch.zeros((), device=self.device)
        self._maze_diag_goal_top1_correct = torch.zeros((), device=self.device)
        self._maze_diag_goal_scene_confusion = torch.zeros(5, 6, device=self.device)
        self._maze_diag_fault_risk_positive_hist = torch.zeros(20, device=self.device)
        self._maze_diag_fault_risk_negative_hist = torch.zeros(20, device=self.device)
        self._maze_diag_fault_top1_total = torch.zeros((), device=self.device)
        self._maze_diag_fault_top1_correct = torch.zeros((), device=self.device)
        self._maze_diag_fault_scene_confusion = torch.zeros(5, 6, device=self.device)

    def _reset_maze_diagnostic_probes(self) -> None:
        probes = (
            self._diagnostic_nav_risk_probe,
            self._diagnostic_nav_scene_probe,
            self._diagnostic_goal_risk_probe,
            self._diagnostic_goal_scene_probe,
        )
        for probe in probes:
            nn.init.zeros_(probe.weight)
            nn.init.zeros_(probe.bias)
        self._diagnostic_probe_optimizer = torch.optim.SGD(
            [parameter for probe in probes for parameter in probe.parameters()],
            lr=0.05,
        )

    @staticmethod
    def _diagnostic_scene_class(
        safe3: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        diagnostics = p4_contract.safety_scene_diagnostics(safe3, valid)
        result = torch.full((safe3.shape[0],), 5, dtype=torch.long, device=safe3.device)
        for index, name in enumerate(
            (
                "teacher_scene_corridor",
                "teacher_scene_left_open",
                "teacher_scene_right_open",
                "teacher_scene_junction",
                "teacher_scene_dead_end",
            )
        ):
            select = (result == 5) & (diagnostics[name] > 0.5)
            result[select] = index
        result[~valid.reshape(-1).bool()] = 5
        return result

    def _train_maze_diagnostic_probes(
        self, nav_feat, goal4, teacher_risk, train_mask, scene_train, teacher_class
    ) -> None:
        with torch.enable_grad():
            self._diagnostic_probe_optimizer.zero_grad(set_to_none=True)
            losses = []
            if bool(train_mask.any()):
                losses.extend(
                    (
                        F.binary_cross_entropy_with_logits(
                            self._diagnostic_nav_risk_probe(nav_feat[train_mask]),
                            teacher_risk[train_mask],
                        ),
                        F.binary_cross_entropy_with_logits(
                            self._diagnostic_goal_risk_probe(goal4[train_mask]),
                            teacher_risk[train_mask],
                        ),
                    )
                )
            if bool(scene_train.any()):
                losses.extend(
                    (
                        F.cross_entropy(
                            self._diagnostic_nav_scene_probe(nav_feat[scene_train]),
                            teacher_class[scene_train],
                        ),
                        F.cross_entropy(
                            self._diagnostic_goal_scene_probe(goal4[scene_train]),
                            teacher_class[scene_train],
                        ),
                    )
                )
            if losses:
                torch.stack(losses).sum().backward()
                self._diagnostic_probe_optimizer.step()
            self._diagnostic_probe_optimizer.zero_grad(set_to_none=True)

    def _evaluate_maze_diagnostic_probes(
        self, valid, env_ids, nav_feat, goal4, fault_nav_feat
    ):
        eval_mask = valid & ((env_ids % 5) == 0)
        if not bool(eval_mask.any()):
            eval_mask = valid
        with torch.no_grad():
            predicted_risk = torch.sigmoid(
                self._diagnostic_nav_risk_probe(nav_feat)
            ).clamp(0.0, 1.0)
            goal_predicted_risk = torch.sigmoid(
                self._diagnostic_goal_risk_probe(goal4)
            ).clamp(0.0, 1.0)
            predicted_scene = self._diagnostic_nav_scene_probe(nav_feat).argmax(dim=-1)
            goal_predicted_scene = self._diagnostic_goal_scene_probe(goal4).argmax(
                dim=-1
            )
            fault_predicted_risk = torch.sigmoid(
                self._diagnostic_nav_risk_probe(fault_nav_feat)
            ).clamp(0.0, 1.0)
            fault_predicted_scene = self._diagnostic_nav_scene_probe(
                fault_nav_feat
            ).argmax(dim=-1)
        return (
            eval_mask,
            predicted_risk,
            goal_predicted_risk,
            predicted_scene,
            goal_predicted_scene,
            fault_predicted_risk,
            fault_predicted_scene,
        )

    def _accumulate_fault_diagnostics(
        self,
        eval_mask,
        fault_mask,
        teacher_risk,
        fault_predicted_risk,
        scene,
        teacher_top,
        teacher_class,
        fault_predicted_scene,
    ) -> None:
        fault_eval = eval_mask & fault_mask
        if bool(fault_eval.any()):
            fault_valid3 = fault_eval.unsqueeze(-1).expand_as(teacher_risk)
            fault_positive = fault_valid3 & (teacher_risk >= 0.65)
            fault_negative = fault_valid3 & (teacher_risk <= 0.35)
            fault_bins = torch.clamp((fault_predicted_risk * 20.0).long(), 0, 19)
            if bool(fault_positive.any()):
                self._maze_diag_fault_risk_positive_hist += torch.bincount(
                    fault_bins[fault_positive], minlength=20
                ).to(self._maze_diag_fault_risk_positive_hist)
            if bool(fault_negative.any()):
                self._maze_diag_fault_risk_negative_hist += torch.bincount(
                    fault_bins[fault_negative], minlength=20
                ).to(self._maze_diag_fault_risk_negative_hist)
            fault_clear = (scene["teacher_safe_top1_clear"] > 0.5) & fault_eval
            fault_top = (1.0 - fault_predicted_risk).argmax(dim=-1)
            self._maze_diag_fault_top1_total += fault_clear.float().sum()
            self._maze_diag_fault_top1_correct += (
                (fault_clear & (teacher_top == fault_top)).float().sum()
            )
            fault_labeled = (teacher_class < 5) & fault_eval
            if bool(fault_labeled.any()):
                fault_flat = (
                    teacher_class[fault_labeled] * 6
                    + fault_predicted_scene[fault_labeled]
                )
                self._maze_diag_fault_scene_confusion += (
                    torch.bincount(fault_flat, minlength=30)
                    .reshape(5, 6)
                    .to(self._maze_diag_fault_scene_confusion)
                )

    def _accumulate_maze_diagnostic(
        self,
        safe3: torch.Tensor,
        teacher_valid: torch.Tensor,
        nav_feat: torch.Tensor,
        goal4: torch.Tensor,
        fault_nav_feat: torch.Tensor | None = None,
        fault_mask: torch.Tensor | None = None,
    ) -> None:
        if self.maze_training_branch != "auto":
            return
        if self._resolved_maze_training_branch is not None:
            return
        if self.diagnostic_elapsed_seconds >= p4_contract.DIAGNOSTIC_SECONDS:
            return
        valid = teacher_valid.reshape(-1).bool()
        self._maze_diag_total += float(valid.numel())
        self._maze_diag_valid += valid.float().sum()
        self._maze_diag_latent_cosine_sum += torch.nan_to_num(
            self._camera_clean_live_latent_cosine, nan=0.0
        ).sum()
        self._maze_diag_latent_cosine_count += float(
            self._camera_clean_live_latent_cosine.numel()
        )
        if not bool(valid.any()):
            return

        teacher_risk = (1.0 - safe3).clamp(0.0, 1.0)
        nav_feat = nav_feat.detach()
        goal4 = goal4.detach()
        fault_nav_feat = (
            fault_nav_feat.detach() if torch.is_tensor(fault_nav_feat) else nav_feat
        )
        fault_mask = (
            fault_mask.reshape(-1).bool()
            if torch.is_tensor(fault_mask)
            else torch.zeros_like(valid)
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        train_mask = valid & ((env_ids % 5) != 0)
        teacher_class = self._diagnostic_scene_class(safe3, valid)
        scene_train = train_mask & (teacher_class < 5)
        # finish_tick() is intentionally inference-only and therefore invokes
        # this hook under torch.no_grad(). The training-only linear probes still
        # need their own small graph; detached inputs prevent gradients from
        # reaching NavigationEncoder, Actor, or the rollout collection path.
        self._train_maze_diagnostic_probes(
            nav_feat, goal4, teacher_risk, train_mask, scene_train, teacher_class
        )
        (
            eval_mask,
            predicted_risk,
            goal_predicted_risk,
            predicted_scene,
            goal_predicted_scene,
            fault_predicted_risk,
            fault_predicted_scene,
        ) = self._evaluate_maze_diagnostic_probes(
            valid, env_ids, nav_feat, goal4, fault_nav_feat
        )

        valid3 = eval_mask.unsqueeze(-1).expand_as(teacher_risk)
        positive = valid3 & (teacher_risk >= 0.65)
        negative = valid3 & (teacher_risk <= 0.35)
        self._maze_diag_wall_positive += positive.float().sum()
        self._maze_diag_wall_missed += (
            (positive & (predicted_risk < 0.50)).float().sum()
        )
        bins = torch.clamp((predicted_risk * 20.0).long(), 0, 19)
        if bool(positive.any()):
            self._maze_diag_risk_positive_hist += torch.bincount(
                bins[positive], minlength=20
            ).to(self._maze_diag_risk_positive_hist)
        if bool(negative.any()):
            self._maze_diag_risk_negative_hist += torch.bincount(
                bins[negative], minlength=20
            ).to(self._maze_diag_risk_negative_hist)
        goal_bins = torch.clamp((goal_predicted_risk * 20.0).long(), 0, 19)
        if bool(positive.any()):
            self._maze_diag_goal_risk_positive_hist += torch.bincount(
                goal_bins[positive], minlength=20
            ).to(self._maze_diag_goal_risk_positive_hist)
        if bool(negative.any()):
            self._maze_diag_goal_risk_negative_hist += torch.bincount(
                goal_bins[negative], minlength=20
            ).to(self._maze_diag_goal_risk_negative_hist)

        scene = p4_contract.safety_scene_diagnostics(safe3, valid)
        clear = (scene["teacher_safe_top1_clear"] > 0.5) & eval_mask
        teacher_top = safe3.argmax(dim=-1)
        predicted_safe = 1.0 - predicted_risk
        predicted_top = predicted_safe.argmax(dim=-1)
        goal_predicted_top = (1.0 - goal_predicted_risk).argmax(dim=-1)
        self._maze_diag_top1_total += clear.float().sum()
        self._maze_diag_top1_correct += (
            (clear & (teacher_top == predicted_top)).float().sum()
        )
        self._maze_diag_goal_top1_total += clear.float().sum()
        self._maze_diag_goal_top1_correct += (
            (clear & (teacher_top == goal_predicted_top)).float().sum()
        )

        labeled = (teacher_class < 5) & eval_mask
        if bool(labeled.any()):
            flat = teacher_class[labeled] * 6 + predicted_scene[labeled]
            self._maze_diag_scene_confusion += (
                torch.bincount(flat, minlength=30)
                .reshape(5, 6)
                .to(self._maze_diag_scene_confusion)
            )
            goal_flat = teacher_class[labeled] * 6 + goal_predicted_scene[labeled]
            self._maze_diag_goal_scene_confusion += (
                torch.bincount(goal_flat, minlength=30)
                .reshape(5, 6)
                .to(self._maze_diag_goal_scene_confusion)
            )
        self._accumulate_fault_diagnostics(
            eval_mask,
            fault_mask,
            teacher_risk,
            fault_predicted_risk,
            scene,
            teacher_top,
            teacher_class,
            fault_predicted_scene,
        )

    def _maze_diagnostic_summary(self) -> dict[str, float]:
        def histogram_auc(
            positive: torch.Tensor, negative: torch.Tensor
        ) -> torch.Tensor:
            negative_below = torch.cumsum(negative, dim=0) - negative
            numerator = (positive * (negative_below + 0.5 * negative)).sum()
            return numerator / (positive.sum() * negative.sum()).clamp_min(1.0)

        def scene_macro_f1(confusion: torch.Tensor) -> torch.Tensor:
            values = []
            for index in range(5):
                true_positive = confusion[index, index]
                false_negative = confusion[index, :].sum() - true_positive
                false_positive = confusion[:, index].sum() - true_positive
                denominator = 2.0 * true_positive + false_positive + false_negative
                if float(confusion[index, :].sum().detach().cpu()) > 0.0:
                    values.append(2.0 * true_positive / denominator.clamp_min(1.0))
            return torch.stack(values).mean() if values else confusion.new_zeros(())

        auc = histogram_auc(
            self._maze_diag_risk_positive_hist,
            self._maze_diag_risk_negative_hist,
        )
        goal_auc = histogram_auc(
            self._maze_diag_goal_risk_positive_hist,
            self._maze_diag_goal_risk_negative_hist,
        )
        confusion = self._maze_diag_scene_confusion
        macro_f1 = scene_macro_f1(confusion)
        goal_macro_f1 = scene_macro_f1(self._maze_diag_goal_scene_confusion)
        fault_auc = histogram_auc(
            self._maze_diag_fault_risk_positive_hist,
            self._maze_diag_fault_risk_negative_hist,
        )
        fault_macro_f1 = scene_macro_f1(self._maze_diag_fault_scene_confusion)
        return {
            "teacher_coverage": float(
                (self._maze_diag_valid / self._maze_diag_total.clamp_min(1.0))
                .detach()
                .cpu()
            ),
            "wall_auroc": float(auc.detach().cpu()),
            "wall_miss_rate": float(
                (
                    self._maze_diag_wall_missed
                    / self._maze_diag_wall_positive.clamp_min(1.0)
                )
                .detach()
                .cpu()
            ),
            "safe_top1_accuracy": float(
                (
                    self._maze_diag_top1_correct
                    / self._maze_diag_top1_total.clamp_min(1.0)
                )
                .detach()
                .cpu()
            ),
            "scene_macro_f1": float(macro_f1.detach().cpu()),
            "goal_wall_auroc": float(goal_auc.detach().cpu()),
            "goal_safe_top1_accuracy": float(
                (
                    self._maze_diag_goal_top1_correct
                    / self._maze_diag_goal_top1_total.clamp_min(1.0)
                )
                .detach()
                .cpu()
            ),
            "goal_scene_macro_f1": float(goal_macro_f1.detach().cpu()),
            "fault_wall_auroc": float(fault_auc.detach().cpu()),
            "fault_safe_top1_accuracy": float(
                (
                    self._maze_diag_fault_top1_correct
                    / self._maze_diag_fault_top1_total.clamp_min(1.0)
                )
                .detach()
                .cpu()
            ),
            "fault_scene_macro_f1": float(fault_macro_f1.detach().cpu()),
            "clean_live_latent_cosine": float(
                (
                    self._maze_diag_latent_cosine_sum
                    / self._maze_diag_latent_cosine_count.clamp_min(1.0)
                )
                .detach()
                .cpu()
            ),
            "wall_positive_samples": float(
                self._maze_diag_wall_positive.detach().cpu()
            ),
            "safe_top1_samples": float(self._maze_diag_top1_total.detach().cpu()),
            "scene_samples": float(confusion.sum().detach().cpu()),
            "fault_samples": float(
                self._maze_diag_fault_scene_confusion.sum().detach().cpu()
            ),
        }

    def _risk_response_diagnostic_context(self) -> dict[str, object]:
        event_denominator = self._push_interval_count.clamp_min(1.0).unsqueeze(-1)
        interval_delta_mean = self._push_interval_delta_sum / event_denominator
        raw_goal_distance = torch.linalg.vector_norm(
            self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE],
            dim=-1,
        )
        goal_diagnostics = self.goal_belief.last_diagnostics
        pending = getattr(self, "pending_tick", {}) or {}
        stuck_motion_intent = (
            pending.get(
                "stuck_motion_intent",
                torch.zeros(self.num_envs, 1, device=self.device),
            ).reshape(-1)
            > 0.5
        )
        stuck_candidate = (
            self._p4_worker_extra[:, p4_contract.STUCK_CANDIDATE_INDEX] > 0.5
        )
        safe3 = pending.get("safe3", torch.zeros(self.num_envs, 3, device=self.device))
        safe5 = pending.get(
            "teacher_safe5", torch.zeros(self.num_envs, 5, device=self.device)
        )
        teacher_valid = (
            pending.get(
                "safety_valid", torch.zeros(self.num_envs, 1, device=self.device)
            ).reshape(-1)
            > 0.5
        )
        scene_diag = p4_contract.safety_scene_diagnostics(
            safe3,
            teacher_valid,
        )
        if DIAGNOSTIC_SPECS[self.training_profile].maze_probe:
            self._accumulate_maze_diagnostic(
                safe3,
                teacher_valid,
                pending.get(
                    "nav_feat", torch.zeros(self.num_envs, 32, device=self.device)
                ),
                pending.get(
                    "nav_nonvisual",
                    torch.zeros(
                        self.num_envs,
                        p2_contract.NAV_NONVISUAL_DIM,
                        device=self.device,
                    ),
                )[:, :4],
                self._diagnostic_fault_nav_feat,
                self._diagnostic_fault_mask,
            )
        teacher_top = safe3.argmax(dim=-1)
        head_top = (1.0 - self._last_safety_head_risk3).argmax(dim=-1)
        policy_weights = p2_contract.command_direction_weights(
            self._last_policy_command
        )
        actor_top = policy_weights.argmax(dim=-1)
        teacher_clear = scene_diag["teacher_safe_top1_clear"] > 0.5
        head_correct = teacher_clear & (head_top == teacher_top)
        actor_wrong = head_correct & (actor_top != teacher_top)
        reset_now = (
            pending.get(
                "reset_mask",
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            )
            .reshape(-1)
            .bool()
            | self._diagnostic_terminal_mask
        )
        if bool(reset_now.any()):
            self._risk_event_active[reset_now] = False
            self._risk_event_age_ticks[reset_now] = 0
            self._risk_event_baseline_policy_vx[reset_now] = 0.0
            self._risk_event_baseline_limited_vx[reset_now] = 0.0
            self._risk_condition_previous[reset_now] = False
        center_risk_condition = (
            ~reset_now
            & teacher_valid
            & ((1.0 - safe3[:, 1]) >= 0.65)
            & (self._last_policy_command[:, 0] >= 0.50)
            & (self._last_goal_freshness >= 0.50)
        )
        new_risk_event = center_risk_condition & ~self._risk_condition_previous
        self._risk_condition_previous.copy_(center_risk_condition)
        self._risk_event_active |= new_risk_event
        self._risk_event_baseline_policy_vx[new_risk_event] = self._last_policy_command[
            new_risk_event, 0
        ]
        self._risk_event_baseline_limited_vx[new_risk_event] = (
            self._last_limited_command[new_risk_event, 0]
        )
        self._risk_event_age_ticks[new_risk_event] = 0
        eligible_response = self._risk_event_active & ~new_risk_event
        policy_decel_mask = eligible_response & (
            self._last_policy_command[:, 0]
            <= self._risk_event_baseline_policy_vx - 0.15
        )
        limited_decel_mask = eligible_response & (
            self._last_limited_command[:, 0]
            <= self._risk_event_baseline_limited_vx - 0.15
        )
        self._risk_event_age_ticks[eligible_response] += 1
        no_decel_mask = (
            self._risk_event_active
            & (self._risk_event_age_ticks >= self._risk_response_ticks)
            & ~policy_decel_mask
            & ~limited_decel_mask
        )
        finished = policy_decel_mask | limited_decel_mask | no_decel_mask
        self._risk_event_active[finished] = False
        self._risk_event_age_ticks[finished] = 0
        policy_decel = policy_decel_mask.float()
        limited_decel = limited_decel_mask.float()
        no_decel = no_decel_mask.float()
        return {
            "event_denominator": event_denominator,
            "interval_delta_mean": interval_delta_mean,
            "raw_goal_distance": raw_goal_distance,
            "goal_diagnostics": goal_diagnostics,
            "pending": pending,
            "stuck_motion_intent": stuck_motion_intent,
            "stuck_candidate": stuck_candidate,
            "safe3": safe3,
            "safe5": safe5,
            "teacher_valid": teacher_valid,
            "scene_diag": scene_diag,
            "teacher_top": teacher_top,
            "head_top": head_top,
            "policy_weights": policy_weights,
            "actor_top": actor_top,
            "teacher_clear": teacher_clear,
            "head_correct": head_correct,
            "actor_wrong": actor_wrong,
            "reset_now": reset_now,
            "center_risk_condition": center_risk_condition,
            "new_risk_event": new_risk_event,
            "eligible_response": eligible_response,
            "policy_decel_mask": policy_decel_mask,
            "limited_decel_mask": limited_decel_mask,
            "no_decel_mask": no_decel_mask,
            "finished": finished,
            "policy_decel": policy_decel,
            "limited_decel": limited_decel,
            "no_decel": no_decel,
        }

    def _expanded_quantile(self, values, q):
        finite = values.reshape(-1)[torch.isfinite(values.reshape(-1))]
        scalar = (
            torch.quantile(finite.float(), q)
            if finite.numel()
            else values.new_zeros(())
        )
        return scalar.expand(self.num_envs)

    def _goal_bucket_diagnostic_context(
        self, goal_diagnostics, raw_goal_distance
    ) -> dict[str, object]:
        goal_bucket_metrics: dict[str, torch.Tensor] = {}
        accepted = goal_diagnostics.get(
            "goal_measurement_accepted", raw_goal_distance.new_zeros(self.num_envs)
        )
        clipped = goal_diagnostics.get(
            "goal_measurement_clipped", raw_goal_distance.new_zeros(self.num_envs)
        )
        rejected = goal_diagnostics.get(
            "goal_measurement_rejected", raw_goal_distance.new_zeros(self.num_envs)
        )
        measurement_due = (accepted + clipped + rejected) > 0.5
        for label, bucket in (
            ("0_5", raw_goal_distance < 5.0),
            ("5_10", (raw_goal_distance >= 5.0) & (raw_goal_distance < 10.0)),
            ("10_plus", raw_goal_distance >= 10.0),
        ):
            denominator = (bucket & measurement_due).float().sum().clamp_min(1.0)
            goal_bucket_metrics[f"goal_distance_{label}_share"] = (
                bucket.float().mean().expand(self.num_envs)
            )
            for outcome, source in (
                ("accept", accepted),
                ("clipped", clipped),
                ("reject", rejected),
            ):
                rate = (source * bucket.float()).sum() / denominator
                goal_bucket_metrics[f"goal_{outcome}_{label}"] = rate.expand(
                    self.num_envs
                )
            clean_bearing = torch.atan2(
                self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE.stop - 1],
                self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE.start],
            ).abs()
            fault_bearing = torch.atan2(
                self.goal_belief.estimate[:, 1], self.goal_belief.estimate[:, 0]
            ).abs()
            bucket_count = bucket.float().sum().clamp_min(1.0)
            goal_bucket_metrics[f"goal_bearing_clean_{label}_abs_rad"] = (
                (clean_bearing * bucket.float()).sum() / bucket_count
            ).expand(self.num_envs)
            goal_bucket_metrics[f"goal_bearing_fault_{label}_abs_rad"] = (
                (fault_bearing * bucket.float()).sum() / bucket_count
            ).expand(self.num_envs)
        return {
            "goal_bucket_metrics": goal_bucket_metrics,
            "accepted": accepted,
            "clipped": clipped,
            "rejected": rejected,
            "measurement_due": measurement_due,
        }

    def _stuck_diagnostic_context(self) -> dict[str, object]:
        stuck_duration = self._p4_worker_extra[:, p4_contract.STUCK_DURATION_S_INDEX]
        stuck_candidate = (
            self._p4_worker_extra[:, p4_contract.STUCK_CANDIDATE_INDEX] > 0.5
        )
        candidate_duration = stuck_duration[stuck_candidate]
        if not candidate_duration.numel():
            candidate_duration = stuck_duration.new_zeros(1)
        stuck_reset = (
            self._p4_worker_extra[:, p4_contract.STUCK_RESET_TRIGGERED_INDEX] > 0.5
        )
        collision_delay = self._p4_worker_extra[
            :, p4_contract.STUCK_COLLISION_TO_RESET_S_INDEX
        ]
        collision_delay_mean = (
            collision_delay[stuck_reset].mean()
            if bool(stuck_reset.any())
            else collision_delay.new_zeros(())
        )
        return {
            "stuck_duration": stuck_duration,
            "stuck_candidate": stuck_candidate,
            "candidate_duration": candidate_duration,
            "stuck_reset": stuck_reset,
            "collision_delay": collision_delay,
            "collision_delay_mean": collision_delay_mean,
        }

    def _branch_diagnostic_context(self, raw_goal_distance) -> dict[str, object]:
        diagnostic_summary = (
            self._maze_diagnostic_summary()
            if DIAGNOSTIC_SPECS[self.training_profile].maze_probe
            else {}
        )
        diagnostic_metrics = {
            f"diagnostic_{name}": raw_goal_distance.new_full(
                (self.num_envs,), float(value)
            )
            for name, value in diagnostic_summary.items()
            if name
            in {
                "teacher_coverage",
                "wall_auroc",
                "wall_miss_rate",
                "safe_top1_accuracy",
                "scene_macro_f1",
                "clean_live_latent_cosine",
                "goal_wall_auroc",
                "goal_safe_top1_accuracy",
                "goal_scene_macro_f1",
                "fault_wall_auroc",
                "fault_safe_top1_accuracy",
                "fault_scene_macro_f1",
            }
        }
        resolved_branch = self._effective_maze_branch(self.session_effective_seconds)
        schedule = p4_contract.training_schedule(
            self.session_effective_seconds, branch=resolved_branch
        )
        return {
            "diagnostic_summary": diagnostic_summary,
            "diagnostic_metrics": diagnostic_metrics,
            "resolved_branch": resolved_branch,
            "schedule": schedule,
        }

    def _track_diagnostic_context(
        self, pending, safe5, raw_goal_distance, teacher_valid
    ) -> dict[str, object]:
        current_segment = (
            self._p4_reward_diagnostics.get(
                "current_segment_index", raw_goal_distance.new_zeros(self.num_envs)
            )
            .round()
            .long()
            .clamp(0, len(p4_contract.FULL_TRACK_SEGMENT_LABELS) - 1)
        )
        spawn_segment = (
            self._p4_worker_extra[:, p4_contract.SPAWN_SEGMENT_INDEX].round().long()
        )
        spawn_quartile = (
            self._p4_worker_extra[:, p4_contract.SPAWN_QUARTILE_INDEX].round().long()
        )
        clean_goal = self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE]
        clean_bearing_signed = torch.atan2(clean_goal[:, 1], clean_goal[:, 0])
        side_goal_candidate = (
            teacher_valid
            & (clean_bearing_signed.abs() >= math.radians(15.0))
            & (clean_bearing_signed.abs() <= math.radians(90.0))
        )
        side_goal_selected = side_goal_candidate & (
            self._last_policy_command[:, 2] * clean_bearing_signed > 0.0
        )
        desired_yaw_sign = torch.sign(clean_bearing_signed)
        correct_yaw_response = (
            self._last_policy_command[:, 2] * desired_yaw_sign >= 0.10
        )
        vy_substitutes_yaw = (
            teacher_valid
            & (clean_bearing_signed.abs() >= math.radians(35.0))
            & (self._last_policy_command[:, 1].abs() >= 0.12)
            & ~correct_yaw_response
        )
        limiter_intervened = (
            self._translation_limiter_diagnostics.get(
                "translation_safety_alpha",
                raw_goal_distance.new_ones(self.num_envs),
            )
            < 0.999
        )
        command_delta = self._last_policy_command - self._previous_policy_command
        command_reversal = (
            (self._last_policy_command * self._previous_policy_command < 0.0)
            & (self._last_policy_command.abs() >= 0.05)
            & (self._previous_policy_command.abs() >= 0.05)
        )
        instant_command_active = (
            self.command.command_transition_mode == "instant_hold_10hz"
        )
        frontier_before = self._p4_reward_diagnostics.get(
            "segment_frontier_phi_before", raw_goal_distance.new_zeros(self.num_envs)
        )
        frontier_after = self._p4_reward_diagnostics.get(
            "segment_frontier_phi_after", raw_goal_distance.new_zeros(self.num_envs)
        )
        full_track_metrics: dict[str, torch.Tensor] = {}
        if DIAGNOSTIC_SPECS[self.training_profile].full_track_segments:
            for index, label in enumerate(p4_contract.FULL_TRACK_SEGMENT_LABELS):
                current_mask = current_segment == index
                spawn_mask = spawn_segment == index
                full_track_metrics[f"current_segment_{label}_share"] = (
                    current_mask.float()
                )
                full_track_metrics[f"spawn_segment_{label}_share"] = spawn_mask.float()
                denominator = spawn_mask.float().sum().clamp_min(1.0)
                full_track_metrics[f"frontier_potential_before_{label}"] = (
                    (frontier_before * spawn_mask.float()).sum() / denominator
                ).expand(self.num_envs)
                full_track_metrics[f"frontier_potential_after_{label}"] = (
                    (frontier_after * spawn_mask.float()).sum() / denominator
                ).expand(self.num_envs)
            for index in range(4):
                full_track_metrics[f"spawn_column_q{index + 1}_share"] = (
                    spawn_quartile == index
                ).float()
        return {
            "current_segment": current_segment,
            "spawn_segment": spawn_segment,
            "spawn_quartile": spawn_quartile,
            "clean_goal": clean_goal,
            "clean_bearing_signed": clean_bearing_signed,
            "side_goal_candidate": side_goal_candidate,
            "side_goal_selected": side_goal_selected,
            "desired_yaw_sign": desired_yaw_sign,
            "correct_yaw_response": correct_yaw_response,
            "vy_substitutes_yaw": vy_substitutes_yaw,
            "limiter_intervened": limiter_intervened,
            "command_delta": command_delta,
            "command_reversal": command_reversal,
            "instant_command_active": instant_command_active,
            "frontier_before": frontier_before,
            "frontier_after": frontier_after,
            "full_track_metrics": full_track_metrics,
        }

    def _tick_diagnostic_values_part_1(self, context) -> dict[str, torch.Tensor]:
        return {
            **context["goal_diagnostics"],
            **context["goal_bucket_metrics"],
            **self._camera_diagnostics,
            **self._p4_reward_diagnostics,
            "camera_clean_live_latent_cosine": self._camera_clean_live_latent_cosine,
            "user_speed_cap": self.user_speed_cap,
            "effective_speed_cap": self.effective_speed_cap,
            "safety_speed_cap": self.safety_speed_cap,
            "safety_cap_predictive_risk": self._safety_cap_predictive_risk,
            **context["scene_diag"],
            **context["diagnostic_metrics"],
            **context["full_track_metrics"],
            "safety_head_risk_left": self._last_safety_head_risk3[:, 0],
            "safety_head_risk_center": self._last_safety_head_risk3[:, 1],
            "safety_head_risk_right": self._last_safety_head_risk3[:, 2],
            "teacher_safe5_far_left": context["safe5"][:, 0],
            "teacher_safe5_left": context["safe5"][:, 1],
            "teacher_safe5_center": context["safe5"][:, 2],
            "teacher_safe5_right": context["safe5"][:, 3],
            "teacher_safe5_far_right": context["safe5"][:, 4],
            "head_correct_samples": context["head_correct"].float(),
            "head_correct_actor_wrong_count": context["actor_wrong"].float(),
            "risk_event_resolved_count": context["finished"].float(),
            "risk_decel_policy_count": context["policy_decel"],
        }

    def _tick_diagnostic_values_part_2(self, context) -> dict[str, torch.Tensor]:
        return {
            "risk_decel_limited_count": context["limited_decel"],
            "risk_no_deceleration_count": context["no_decel"],
            "diagnostic_fault_shadow_share": self._diagnostic_fault_mask.float(),
            "diagnostic_clean_fault_latent_cosine": self._diagnostic_clean_fault_latent_cosine,
            "diagnostic_clean_fault_action_mae": self._diagnostic_clean_fault_action_mae,
            "maze_branch_actor_attack": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["resolved_branch"] == "actor_attack")
            ),
            "maze_branch_visual_recovery": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["resolved_branch"] == "visual_recovery")
            ),
            "full_phase_warm": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["schedule"]["phase"] == "fullwarm")
            ),
            "full_phase_adapt": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["schedule"]["phase"] == "fulladapt")
            ),
            "full_phase_train": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["schedule"]["phase"] == "fulltrain")
            ),
            "full_phase_stabilize": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["schedule"]["phase"] == "fullstabilize")
            ),
            "closedloop_phase_critic_warm": context["raw_goal_distance"].new_full(
                (self.num_envs,),
                float(context["schedule"]["phase"] in {"closedwarm", "loopwarm"}),
            ),
            "closedloop_phase_actor_adapt": context["raw_goal_distance"].new_full(
                (self.num_envs,),
                float(context["schedule"]["phase"] in {"closedadapt", "loopadapt"}),
            ),
            "closedloop_phase_train": context["raw_goal_distance"].new_full(
                (self.num_envs,),
                float(context["schedule"]["phase"] in {"closedtrain", "looptrain"}),
            ),
            "closedloop_phase_stabilize": context["raw_goal_distance"].new_full(
                (self.num_envs,),
                float(context["schedule"]["phase"] in {"closedstable", "loopstable"}),
            ),
            "legitimate_side_goal_candidate_count": context[
                "side_goal_candidate"
            ].float(),
            "legitimate_side_goal_selected_count": context[
                "side_goal_selected"
            ].float(),
            "legitimate_side_goal_bearing_abs_rad": context[
                "clean_bearing_signed"
            ].abs(),
            "large_goal_correct_yaw_response": context["correct_yaw_response"].float(),
            "vy_substitutes_yaw_count": context["vy_substitutes_yaw"].float(),
            "translation_limiter_intervention": context["limiter_intervened"].float(),
            "instant_command_active": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["instant_command_active"])
            ),
            "policy_limited_command_mae": (
                self._last_policy_command - self._last_limited_command
            )
            .abs()
            .mean(dim=-1),
            "policy_exec_command_mae": (
                self._last_policy_command - self.command.exec_cmd
            )
            .abs()
            .mean(dim=-1),
        }

    def _tick_diagnostic_values_part_3(self, context) -> dict[str, torch.Tensor]:
        return {
            "command_hold_frames": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(self.command.hold_frames)
            ),
            "instant_phase_warm": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["schedule"]["phase"] == "instantwarm")
            ),
            "instant_phase_adapt": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["schedule"]["phase"] == "instantadapt")
            ),
            "instant_phase_correct": context["raw_goal_distance"].new_full(
                (self.num_envs,),
                float(context["schedule"]["phase"] == "instantcorrect"),
            ),
            "instant_phase_stable": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["schedule"]["phase"] == "instantstable")
            ),
            "instant_phase_frozen": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["schedule"]["phase"] == "instantfrozen")
            ),
            "repair_phase_collect": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["schedule"]["phase"] == "repaircollect")
            ),
            "repair_phase_adapt": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["schedule"]["phase"] == "repairadapt")
            ),
            "repair_phase_train": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["schedule"]["phase"] == "repairtrain")
            ),
            "repair_phase_stable": context["raw_goal_distance"].new_full(
                (self.num_envs,), float(context["schedule"]["phase"] == "repairstable")
            ),
            "spawn_safe_point_share": self._p4_worker_extra[
                :, p4_contract.SPAWN_SAFE_POINT_INDEX
            ],
            "spawn_full_start_share": self._p4_worker_extra[
                :, p4_contract.SPAWN_FULL_START_INDEX
            ],
            "spawn_segment_index": context["spawn_segment"].float(),
            "spawn_position_quartile": context["spawn_quartile"].float(),
            "spawn_reason4_retry_count": self._p4_worker_extra[
                :, p4_contract.SPAWN_REASON4_RETRY_COUNT_INDEX
            ],
            "spawn_reason4_exhausted_count": self._p4_worker_extra[
                :, p4_contract.SPAWN_REASON4_EXHAUSTED_COUNT_INDEX
            ],
            "spawn_reason4_fallback_applied_count": self._p4_worker_extra[
                :, p4_contract.SPAWN_REASON4_FALLBACK_APPLIED_COUNT_INDEX
            ],
            "spawn_all_position_applied_count": self._p4_worker_extra[
                :, p4_contract.SPAWN_ALL_POSITION_APPLIED_COUNT_INDEX
            ],
            "spawn_validation_failure_count": self._p4_worker_extra[
                :, p4_contract.SPAWN_VALIDATION_FAILURE_COUNT_INDEX
            ],
            "spawn_write_failure_count": self._p4_worker_extra[
                :, p4_contract.SPAWN_WRITE_FAILURE_COUNT_INDEX
            ],
            "zero_hidden_action_mae": self._zero_hidden_action_mae,
            "zero_hidden_direction_disagreement": self._zero_hidden_direction_disagreement,
            "push_epoch": self.push_epoch.float(),
            "seconds_since_push": self.seconds_since_push,
        }

    def _tick_diagnostic_values_part_4(self, context) -> dict[str, torch.Tensor]:
        return {
            "push_runtime_active": self._p4_extra[
                :, p3_contract.PUSH_RUNTIME_ACTIVE_INDEX
            ],
            "push_telemetry_valid": self._p4_extra[
                :, p3_contract.PUSH_TELEMETRY_VALID_INDEX
            ],
            "push_delta_vx": self._p4_extra[
                :, p3_contract.PUSH_DELTA_VELOCITY_SLICE.start
            ],
            "push_delta_vy": self._p4_extra[
                :, p3_contract.PUSH_DELTA_VELOCITY_SLICE.start + 1
            ],
            "push_event_count": self._push_interval_count,
            "push_lifetime_count": self._push_lifetime_count,
            "push_env_coverage": self._push_env_seen.float(),
            "push_actual_delta_vx_mean": context["interval_delta_mean"][:, 0],
            "push_actual_delta_vy_mean": context["interval_delta_mean"][:, 1],
            "push_actual_delta_vx_abs_max": self._push_interval_delta_abs_max[:, 0],
            "push_actual_delta_vy_abs_max": self._push_interval_delta_abs_max[:, 1],
            "push_recovery_pending": self._push_recovery_pending.float(),
            "push_tracking_recovery_time_s": torch.nan_to_num(
                self._push_recovery_time, nan=0.0
            ),
            "raw_goal_distance_gt10_share": (
                context["raw_goal_distance"] > 10.0
            ).float(),
            "goal_innovation_d2_p50": self._expanded_quantile(
                context["goal_diagnostics"]["goal_innovation_d2"][
                    context["measurement_due"]
                ],
                0.5,
            ),
            "goal_innovation_d2_p90": self._expanded_quantile(
                context["goal_diagnostics"]["goal_innovation_d2"][
                    context["measurement_due"]
                ],
                0.9,
            ),
            "goal_innovation_d2_p99": self._expanded_quantile(
                context["goal_diagnostics"]["goal_innovation_d2"][
                    context["measurement_due"]
                ],
                0.99,
            ),
            "goal_age_s_p50": self._expanded_quantile(
                context["goal_diagnostics"]["goal_age_s"], 0.5
            ),
            "goal_age_s_p90": self._expanded_quantile(
                context["goal_diagnostics"]["goal_age_s"], 0.9
            ),
            "motion_confined_share": self._p4_worker_extra[
                :, p4_contract.STUCK_MOTION_CONFINED_INDEX
            ],
            "wall_evidence_share": self._p4_worker_extra[
                :, p4_contract.STUCK_WALL_EVIDENCE_INDEX
            ],
            "wall_stuck_candidate_share": self._p4_worker_extra[
                :, p4_contract.STUCK_CANDIDATE_INDEX
            ],
            "wall_stuck_candidate_with_motion_intent_share": (
                context["stuck_candidate"] & context["stuck_motion_intent"]
            ).float(),
            "wall_stuck_candidate_without_motion_intent_share": (
                context["stuck_candidate"] & ~context["stuck_motion_intent"]
            ).float(),
        }

    def _tick_diagnostic_values_part_5(self, context) -> dict[str, torch.Tensor]:
        return {
            "wall_stuck_duration_s": self._p4_worker_extra[
                :, p4_contract.STUCK_DURATION_S_INDEX
            ],
            "wall_stuck_would_reset": self._p4_worker_extra[
                :, p4_contract.STUCK_WOULD_RESET_INDEX
            ],
            "wall_stuck_reset_triggered": self._p4_worker_extra[
                :, p4_contract.STUCK_RESET_TRIGGERED_INDEX
            ],
            "wall_stuck_mapping_valid": self._p4_worker_extra[
                :, p4_contract.STUCK_MAPPING_VALID_INDEX
            ],
            "reset_after_push_share": self._p4_worker_extra[
                :, p4_contract.STUCK_RESET_AFTER_PUSH_INDEX
            ],
            "wall_stuck_saved_seconds": self._p4_worker_extra[
                :, p4_contract.STUCK_SAVED_SECONDS_INDEX
            ],
            "wall_stuck_duration_p50_s": self._expanded_quantile(
                context["candidate_duration"], 0.5
            ),
            "wall_stuck_duration_p90_s": self._expanded_quantile(
                context["candidate_duration"], 0.9
            ),
            "collision_to_stuck_reset_delay_s": context["collision_delay_mean"].expand(
                self.num_envs
            ),
            "wall_stuck_term_available": self._p4_worker_extra[
                :, p4_contract.STUCK_TERM_AVAILABLE_INDEX
            ],
            "wall_stuck_term_config_valid": self._p4_worker_extra[
                :, p4_contract.STUCK_TERM_CONFIG_VALID_INDEX
            ],
            "p4_spawn_hook_installed": self._p4_worker_extra[
                :, p4_contract.SPAWN_INSTALLED_INDEX
            ],
            "wall_stuck_raw_term": self._p4_worker_extra[
                :, p4_contract.STUCK_RAW_TERM_INDEX
            ],
        }

    def _tick_diagnostic_values(self, context) -> dict[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {}
        values.update(self._tick_diagnostic_values_part_1(context))
        values.update(self._tick_diagnostic_values_part_2(context))
        values.update(self._tick_diagnostic_values_part_3(context))
        values.update(self._tick_diagnostic_values_part_4(context))
        values.update(self._tick_diagnostic_values_part_5(context))
        return values

    def _extra_tick_diagnostics(self):
        context = self._risk_response_diagnostic_context()
        context.update(
            self._goal_bucket_diagnostic_context(
                context["goal_diagnostics"], context["raw_goal_distance"]
            )
        )
        context.update(self._stuck_diagnostic_context())
        context.update(self._branch_diagnostic_context(context["raw_goal_distance"]))
        context.update(
            self._track_diagnostic_context(
                context["pending"],
                context["safe5"],
                context["raw_goal_distance"],
                context["teacher_valid"],
            )
        )
        values = self._tick_diagnostic_values(context)
        normalized = getattr(self, "_last_normalized_action", None)
        mapped = getattr(self, "_last_mapped_command", None)
        if torch.is_tensor(normalized) and normalized.shape == (self.num_envs, 3):
            for index, axis in enumerate(("vx", "vy", "wz")):
                values[f"normalized_action_{axis}"] = normalized[:, index]
        if torch.is_tensor(mapped) and mapped.shape == (self.num_envs, 3):
            for index, axis in enumerate(("vx", "vy", "wz")):
                values[f"mapped_cmd_{axis}"] = mapped[:, index]
        for label, command in (
            ("policy_target", self._last_policy_command),
            ("limited_target", self._last_limited_command),
        ):
            if torch.is_tensor(command) and command.shape == (self.num_envs, 3):
                for index, axis in enumerate(("vx", "vy", "wz")):
                    values[f"{label}_{axis}"] = command[:, index]
        for index, axis in enumerate(("vx", "vy", "wz")):
            values[f"policy_target_delta_{axis}"] = context["command_delta"][:, index]
            values[f"policy_target_reversal_{axis}"] = context["command_reversal"][
                :, index
            ].float()
        result = {
            name: torch.nan_to_num(value).detach().reshape(-1, 1).clone()
            for name, value in values.items()
            if torch.is_tensor(value) and value.numel() == self.num_envs
        }
        self._push_interval_count.zero_()
        self._push_interval_delta_sum.zero_()
        self._push_interval_delta_abs_max.zero_()
        return result

    def memory_metrics(self) -> dict[str, float]:
        result = super().memory_metrics()
        current_low = self._module_digest(
            (("vision", self.low_level_encoder), ("actor", self.low_level_actor))
        )
        result.update(
            low_digest_drift=float(current_low != self._initial_low_digest),
            low_optimizer_steps=0.0,
            high_updates=float(self.actor_gradient_steps),
            mapper_version_valid=1.0,
            p4_tbptt_sequences_per_env=2.0,
            p4_tbptt_nav_ticks=16.0,
            push_term_assembly_valid=float(
                bool(
                    torch.all(
                        self._p4_extra[:, p3_contract.PUSH_TELEMETRY_VALID_INDEX] > 0.5
                    )
                )
            ),
            push_rollout_event_count_total=float(self._push_rollout_count.sum()),
            push_lifetime_event_count_total=float(self._push_lifetime_count.sum()),
            push_env_coverage_rate=float(self._push_env_seen.float().mean()),
        )
        if self.response_buffer is not None:
            result.update(
                adapter_target_current_ratio=0.50,
                adapter_target_p35_ratio=0.25,
                adapter_target_earlier_ratio=0.25,
            )
            result["adapter_compat_rejected_records"] = float(
                sum(self.response_buffer.compatibility_rejections.values())
            )
            result["adapter_compat_migrated_legacy_parent_records"] = float(
                self.response_buffer.legacy_parent_records_migrated
            )
            result["adapter_legacy_parent_rejected_records"] = float(
                sum(self.response_buffer.legacy_parent_migration_rejections.values())
            )
            for reason in p4_contract.ADAPTER_COMPATIBILITY_REJECTION_REASONS:
                result[p4_contract.adapter_compatibility_metric_name(reason)] = float(
                    self.response_buffer.compatibility_rejections.get(reason, 0)
                )
            for index, label in enumerate(("02s", "06s", "10s")):
                result[f"adapter_push_rejected_{label}"] = float(
                    self.response_buffer.push_horizon_rejections[index]
                )
            result["adapter_push_rejected_pose"] = float(
                self.response_buffer.push_pose_rejections
            )
        return result
