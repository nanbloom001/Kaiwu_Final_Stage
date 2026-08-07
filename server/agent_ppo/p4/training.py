#!/usr/bin/env python3
"""P4 branch selection, schedules, phases, and training clocks."""

from __future__ import annotations

from agent_ppo.p4.constants import *  # noqa: F403

from agent_ppo.feature import p4_contract


class P4TrainingMixin:
    """Behavior-preserving methods extracted from AlgorithmP4NavPPO."""

    def _effective_maze_branch(self, seconds: float) -> str:
        del seconds
        configured = str(self.maze_training_branch or "actor_attack")
        if configured != "auto":
            return configured
        # The parent constructor applies the optimizer schedule before P4
        # diagnostic buffers are allocated.  Treat that initialization-only
        # call as unresolved auto; the first rollout-boundary schedule resolves
        # the branch after all diagnostics exist.
        if not hasattr(self, "_maze_diag_risk_positive_hist"):
            return "auto"
        if self.diagnostic_elapsed_seconds < p4_contract.DIAGNOSTIC_SECONDS:
            return "auto"
        if self._resolved_maze_training_branch is None:
            summary = self._maze_diagnostic_summary()
            sufficient_samples = (
                summary["wall_positive_samples"] >= 100.0
                and summary["safe_top1_samples"] >= 100.0
                and summary["scene_samples"] >= 100.0
            )
            perception_passed = (
                sufficient_samples
                and summary["teacher_coverage"] >= 0.90
                and summary["wall_auroc"] >= 0.85
                and summary["wall_miss_rate"] <= 0.15
                and summary["safe_top1_accuracy"] >= 0.70
                and summary["scene_macro_f1"] >= 0.70
                and summary["clean_live_latent_cosine"] >= 0.90
                and summary["wall_auroc"] >= summary["goal_wall_auroc"] + 0.05
                and summary["safe_top1_accuracy"]
                >= summary["goal_safe_top1_accuracy"] + 0.10
                and summary["scene_macro_f1"] >= summary["goal_scene_macro_f1"] + 0.10
                and summary["wall_auroc"] >= 0.55
                and summary["safe_top1_accuracy"] >= (1.0 / 3.0) + 0.10
                and summary["scene_macro_f1"] >= 0.30
                and summary["fault_samples"] >= 50.0
                and summary["fault_wall_auroc"] >= summary["wall_auroc"] - 0.10
                and summary["fault_safe_top1_accuracy"]
                >= summary["safe_top1_accuracy"] - 0.15
            )
            self._resolved_maze_training_branch = (
                "actor_attack" if perception_passed else "visual_recovery"
            )
            if self.logger:
                self.logger.info(
                    f"[P4Maze] auto diagnostic selected "
                    f"branch={self._resolved_maze_training_branch} "
                    f"coverage={summary['teacher_coverage']:.3f} "
                    f"wall_auroc={summary['wall_auroc']:.3f} "
                    f"wall_miss={summary['wall_miss_rate']:.3f} "
                    f"top1={summary['safe_top1_accuracy']:.3f} "
                    f"scene_f1={summary['scene_macro_f1']:.3f} "
                    f"clean_live_cos={summary['clean_live_latent_cosine']:.3f} "
                    f"goal_auc={summary['goal_wall_auroc']:.3f} "
                    f"goal_top1={summary['goal_safe_top1_accuracy']:.3f} "
                    f"goal_scene_f1={summary['goal_scene_macro_f1']:.3f} "
                    f"fault_auc={summary['fault_wall_auroc']:.3f} "
                    f"fault_top1={summary['fault_safe_top1_accuracy']:.3f} "
                    f"fault_scene_f1={summary['fault_scene_macro_f1']:.3f} "
                    f"sufficient_samples={sufficient_samples}"
                )
        return self._resolved_maze_training_branch

    def _apply_training_schedule(self, session_effective_seconds, **_kwargs):
        self.session_effective_seconds = max(0.0, float(session_effective_seconds))
        self.effective_training_seconds = self.session_effective_seconds
        self.lifetime_effective_seconds = (
            self.lifetime_base_seconds + self.session_effective_seconds
        )
        schedule = p4_contract.training_schedule(
            self.session_effective_seconds,
            branch=self._effective_maze_branch(self.session_effective_seconds),
        )
        self.cnn_unfrozen = float(schedule["navigation_multiplier"]) > 0.0
        self.entropy_coefficient = float(schedule["entropy_coefficient"])
        self.optimizer_phase = str(schedule["phase"])
        if self.actor_optimizer is not None:
            for group in self.actor_optimizer.param_groups:
                name = str(group.get("name", ""))
                if name == "navigation_safety_head":
                    base = p4_contract.SAFETY_HEAD_LR
                    multiplier = float(schedule["safety_head_multiplier"])
                elif name == "actor_stuck_head":
                    base = p4_contract.SAFETY_HEAD_LR
                    multiplier = float(
                        schedule.get(
                            "stuck_head_multiplier", schedule["safety_head_multiplier"]
                        )
                    )
                elif name.startswith("navigation_"):
                    layer = name.removeprefix("navigation_")
                    base = p4_contract.NAVIGATION_ENCODER_LRS[layer]
                    multiplier = float(schedule["navigation_multiplier"])
                else:
                    base = float(schedule.get("actor_lr", p4_contract.ACTOR_LR))
                    multiplier = float(schedule["actor_multiplier"])
                group["base_lr"] = base
                group["lr"] = base * multiplier
                trainable = multiplier > 0.0
                for parameter in group["params"]:
                    parameter.requires_grad_(trainable)
                    if not trainable:
                        parameter.grad = None
        if self.critic_optimizer is not None:
            self.critic_optimizer.param_groups[0]["lr"] = float(
                schedule.get(
                    "critic_lr",
                    p4_contract.CRITIC_LR
                    * float(schedule.get("critic_multiplier", 1.0)),
                )
            )
        if self.response_optimizer is not None:
            adapter_lr = float(
                schedule.get(
                    "adapter_lr_override",
                    p4_contract.ADAPTER_LR * float(schedule["adapter_multiplier"]),
                )
            )
            self.response_optimizer.param_groups[0]["lr"] = adapter_lr
            if self.response_scheduler is not None:
                self.response_scheduler.base_lrs = [adapter_lr]
                self.response_scheduler._last_lr = [adapter_lr]
        adapter_trainable = float(schedule["adapter_multiplier"]) > 0.0
        for parameter in self.response_adapter.parameters():
            parameter.requires_grad_(adapter_trainable)
            if not adapter_trainable:
                parameter.grad = None
        return schedule

    def maybe_unfreeze_cnn(self, effective_seconds: float) -> bool:
        del effective_seconds
        previous = self.cnn_unfrozen
        self._apply_training_schedule(self.session_effective_seconds)
        return (not previous) and self.cnn_unfrozen

    def update_training_clocks(
        self,
        session_wall_seconds: float,
        *,
        session_effective_seconds: float | None = None,
    ) -> None:
        self.session_wall_seconds = max(0.0, float(session_wall_seconds))
        supplied_effective_seconds = (
            None
            if session_effective_seconds is None
            else max(0.0, float(session_effective_seconds))
        )
        if self.maze_training_branch == "auto":
            self.diagnostic_elapsed_seconds = min(
                self.session_wall_seconds, p4_contract.DIAGNOSTIC_SECONDS
            )
            if (
                self._training_clock_origin_seconds is None
                and self.session_wall_seconds >= p4_contract.DIAGNOSTIC_SECONDS
            ):
                # The diagnostic decision is applied at this rollout boundary.
                # Start the gradient-training clock here so the diagnostic's
                # final partial rollout cannot consume the 28800-second budget.
                self._training_clock_origin_seconds = self.session_wall_seconds
            training_seconds = (
                0.0
                if self._training_clock_origin_seconds is None
                else max(
                    0.0,
                    self.session_wall_seconds - self._training_clock_origin_seconds,
                )
            )
        else:
            self.diagnostic_elapsed_seconds = 0.0
            self._training_clock_origin_seconds = 0.0
            training_seconds = (
                self.session_wall_seconds
                if supplied_effective_seconds is None
                else supplied_effective_seconds
            )
        previous = self.cnn_unfrozen
        self._apply_training_schedule(training_seconds)
        if (
            self.rollout is not None
            and self.rollout.step == 0
            and previous != self.cnn_unfrozen
        ):
            self.rollout = self.rollout.reset(store_depth=self.cnn_unfrozen)

    @property
    def current_phase(self) -> str:
        return str(
            p4_contract.training_schedule(
                self.session_effective_seconds,
                branch=self._effective_maze_branch(self.session_effective_seconds),
            )["phase"]
        )


def push_phase_config(
    session_effective_seconds: float,
) -> dict[str, float | str | bool]:
    del session_effective_seconds
    return {
        "name": "p4recovery_no_push",
        "active": False,
        "max_velocity_xy_m_s": 0.0,
        "min_interval_s": 30.0,
        "max_interval_s": 45.0,
    }


def camera_mix(session_effective_seconds: float) -> dict[str, float]:
    seconds = max(0.0, float(session_effective_seconds))
    if seconds < 7_200.0:
        return {"nominal": 0.90, "light": 0.10, "delayed": 0.0, "severe": 0.0}
    return {"nominal": 0.80, "light": 0.20, "delayed": 0.0, "severe": 0.0}


def safe_direction_weight(session_effective_seconds: float) -> float:
    """Keep the verified parent safety weight fixed during credit repair."""
    del session_effective_seconds
    return 0.012


def _instant_repair_schedule(
    seconds: float, branch: str
) -> dict[str, float | str | bool]:
    common = {
        "training_branch": branch,
        "reward_multiplier": 1.0,
        "goal_fault_multiplier": 0.0,
        "camera_aux_ratio": 0.0,
        "stuck_gradient_target_ratio": 0.0,
        "mirror_sequence_share": 0.0,
        "cruise_multiplier": 0.0,
        "teacher_gradient_hard_cap": 0.03,
        "auxiliary_gradient_hard_cap": 0.03,
        "mirror_gradient_hard_cap": 0.0,
        "navigation_multiplier": 0.0,
        "safety_head_multiplier": 0.0,
        "stuck_head_multiplier": 0.0,
        "mirror_gradient_target_ratio": 0.0,
    }
    if seconds < 300.0:
        return {
            **common,
            "phase": "repaircollect",
            "actor_multiplier": 0.0,
            "critic_multiplier": 1.0,
            "actor_lr": 0.0,
            "critic_lr": 6.0e-5,
            "teacher_gradient_target_ratio": 0.0,
            "anchor_multiplier": 0.0,
            "anchor_target_ratio": 0.0,
            "adapter_multiplier": 0.0,
            "adapter_lr_override": 0.0,
            "entropy_coefficient": 0.004,
        }
    if seconds < 1_800.0:
        return {
            **common,
            "phase": "repairadapt",
            "actor_multiplier": 1.0,
            "critic_multiplier": 1.0,
            "actor_lr": 1.0e-5,
            "critic_lr": 6.0e-5,
            "teacher_gradient_target_ratio": 0.010,
            "anchor_multiplier": 1.0,
            "anchor_target_ratio": 0.005,
            "adapter_multiplier": 1.0,
            "adapter_lr_override": 1.25e-6,
            "entropy_coefficient": 0.004,
        }
    if seconds < 5_400.0:
        return {
            **common,
            "phase": "repairtrain",
            "actor_multiplier": 1.0,
            "critic_multiplier": 1.0,
            "actor_lr": 2.0e-5,
            "critic_lr": 5.0e-5,
            "teacher_gradient_target_ratio": 0.015,
            "anchor_multiplier": 1.0,
            "anchor_target_ratio": 0.0075,
            "adapter_multiplier": 1.0,
            "adapter_lr_override": 2.5e-6,
            "entropy_coefficient": 0.004,
        }
    return {
        **common,
        "phase": "repairstable",
        "actor_multiplier": 1.0,
        "critic_multiplier": 1.0,
        "actor_lr": 7.5e-6,
        "critic_lr": 2.0e-5,
        "teacher_gradient_target_ratio": 0.010,
        "anchor_multiplier": 1.0,
        "anchor_target_ratio": 0.010,
        "adapter_multiplier": 0.0,
        "adapter_lr_override": 0.0,
        "entropy_coefficient": 0.0035,
    }


def _instant_command_schedule(
    seconds: float, branch: str
) -> dict[str, float | str | bool]:
    common = {
        "training_branch": branch,
        "reward_multiplier": 1.0,
        "goal_fault_multiplier": 0.0,
        "camera_aux_ratio": 0.0,
        "stuck_gradient_target_ratio": 0.0,
        "mirror_sequence_share": 0.0,
        "cruise_multiplier": 0.0,
        "teacher_gradient_hard_cap": 0.03,
        # Teacher and parent-anchor are the only Actor auxiliaries enabled
        # by this profile. Their largest requested sum is 2.75%, so a 3%
        # aggregate cap preserves both signals without exceeding the
        # documented auxiliary-gradient budget.
        "auxiliary_gradient_hard_cap": 0.03,
        "mirror_gradient_hard_cap": 0.0,
        "navigation_multiplier": 0.0,
        "safety_head_multiplier": 0.0,
        "stuck_head_multiplier": 0.0,
        "mirror_gradient_target_ratio": 0.0,
    }
    if seconds < 1_800.0:
        return {
            **common,
            "phase": "instantwarm",
            "actor_multiplier": 0.0,
            "critic_multiplier": 1.0,
            "actor_lr": 0.0,
            "critic_lr": 6.0e-5,
            "teacher_gradient_target_ratio": 0.0,
            "anchor_multiplier": 0.0,
            "anchor_target_ratio": 0.0,
            "adapter_multiplier": 0.0,
            "adapter_lr_override": 0.0,
            "entropy_coefficient": 0.004,
        }
    if seconds < 7_200.0:
        return {
            **common,
            "phase": "instantadapt",
            "actor_multiplier": 1.0,
            "critic_multiplier": 1.0,
            "actor_lr": 1.5e-5,
            "critic_lr": 6.0e-5,
            "teacher_gradient_target_ratio": 0.0125,
            "anchor_multiplier": 1.0,
            "anchor_target_ratio": 0.005,
            "adapter_multiplier": 1.0,
            "adapter_lr_override": 5.0e-6,
            "entropy_coefficient": 0.004,
        }
    if seconds < 10_800.0:
        return {
            **common,
            "phase": "instantcorrect",
            "actor_multiplier": 1.0,
            "critic_multiplier": 1.0,
            "actor_lr": 1.0e-5,
            "critic_lr": 4.0e-5,
            "teacher_gradient_target_ratio": 0.0175,
            "anchor_multiplier": 1.0,
            "anchor_target_ratio": 0.010,
            "adapter_multiplier": 0.0,
            "adapter_lr_override": 0.0,
            "entropy_coefficient": 0.0035,
        }
    if seconds < 12_600.0:
        return {
            **common,
            "phase": "instantstable",
            "actor_multiplier": 1.0,
            "critic_multiplier": 1.0,
            "actor_lr": 5.0e-6,
            "critic_lr": 3.0e-5,
            "teacher_gradient_target_ratio": 0.010,
            "anchor_multiplier": 1.0,
            "anchor_target_ratio": 0.0125,
            "adapter_multiplier": 0.0,
            "adapter_lr_override": 0.0,
            "entropy_coefficient": 0.0035,
        }
    return {
        **common,
        "phase": "instantfrozen",
        "actor_multiplier": 0.0,
        "critic_multiplier": 1.0,
        "actor_lr": 0.0,
        "critic_lr": 1.0e-5,
        "teacher_gradient_target_ratio": 0.0,
        "anchor_multiplier": 0.0,
        "anchor_target_ratio": 0.0,
        "adapter_multiplier": 0.0,
        "adapter_lr_override": 0.0,
        "entropy_coefficient": 0.0035,
    }


def _closed_loop_schedule(seconds: float, branch: str) -> dict[str, float | str | bool]:
    common = {
        "training_branch": branch,
        "reward_multiplier": 1.0,
        "goal_fault_multiplier": 0.0,
        "camera_aux_ratio": 0.0,
        "stuck_gradient_target_ratio": 0.0,
        "mirror_sequence_share": 0.0,
        "cruise_multiplier": 0.0,
        "teacher_gradient_hard_cap": 0.03,
        "auxiliary_gradient_hard_cap": 0.03,
        "mirror_gradient_hard_cap": 0.0,
        "navigation_multiplier": 0.0,
        "safety_head_multiplier": 0.0,
        "adapter_multiplier": 0.0,
        "stuck_head_multiplier": 0.0,
        "mirror_gradient_target_ratio": 0.0,
    }
    if seconds < 1_800.0:
        return {
            **common,
            "phase": "loopwarm",
            "actor_multiplier": 0.0,
            "actor_lr": 3.0e-5,
            "critic_lr": 6.0e-5,
            "teacher_gradient_target_ratio": 0.0,
            "entropy_coefficient": 0.004,
        }
    if seconds < 7_200.0:
        return {
            **common,
            "phase": "loopadapt",
            "actor_multiplier": 1.0,
            "actor_lr": 3.0e-5,
            "critic_lr": 6.0e-5,
            "teacher_gradient_target_ratio": (0.010 * (seconds - 1_800.0) / 5_400.0),
            "entropy_coefficient": 0.004,
        }
    if seconds < 21_600.0:
        return {
            **common,
            "phase": "looptrain",
            "actor_multiplier": 1.0,
            "actor_lr": 5.0e-5,
            "critic_lr": 5.0e-5,
            "teacher_gradient_target_ratio": 0.020,
            "entropy_coefficient": 0.003,
        }
    return {
        **common,
        "phase": "loopstable",
        "actor_multiplier": 1.0,
        "actor_lr": 2.5e-5,
        "critic_lr": 3.0e-5,
        "teacher_gradient_target_ratio": 0.010,
        "entropy_coefficient": 0.002,
    }


def _credit_repair_schedule(
    seconds: float, branch: str
) -> dict[str, float | str | bool]:
    common = {
        "training_branch": branch,
        "reward_multiplier": 1.0,
        "goal_fault_multiplier": 0.0,
        "camera_aux_ratio": 0.0,
        "stuck_gradient_target_ratio": 0.005,
        "mirror_sequence_share": 0.0,
        "cruise_multiplier": 1.0,
        "teacher_gradient_hard_cap": 0.05,
        "auxiliary_gradient_hard_cap": 0.05,
        "mirror_gradient_hard_cap": 0.0,
        "navigation_multiplier": 0.0,
        "safety_head_multiplier": 0.0,
        "adapter_multiplier": 0.0,
        "stuck_head_multiplier": 1.0,
        "mirror_gradient_target_ratio": 0.0,
        "entropy_coefficient": 0.005,
    }
    if seconds < 600.0:
        return {
            **common,
            "phase": "creditwarm",
            "actor_multiplier": 0.0,
            "critic_lr": 1.2e-4,
            "teacher_gradient_target_ratio": 0.0,
        }
    if seconds < 1_800.0:
        return {
            **common,
            "phase": "creditadapt",
            "actor_multiplier": 1.0,
            "actor_lr": 7.5e-5,
            "critic_lr": 1.2e-4,
            "teacher_gradient_target_ratio": (0.025 * (seconds - 600.0) / 1_200.0),
        }
    if seconds < 6_300.0:
        return {
            **common,
            "phase": "credittrain",
            "actor_multiplier": 1.0,
            "actor_lr": 1.0e-4,
            "critic_lr": 1.0e-4,
            "teacher_gradient_target_ratio": 0.035,
        }
    return {
        **common,
        "phase": "creditfinal",
        "actor_multiplier": 1.0,
        "actor_lr": 5.0e-5,
        "critic_lr": 6.0e-5,
        "teacher_gradient_target_ratio": 0.020,
    }


def _full_track_schedule(seconds: float, branch: str) -> dict[str, float | str | bool]:
    common = {
        "training_branch": branch,
        "reward_multiplier": 1.0,
        "goal_fault_multiplier": min(1.0, max(0.0, (seconds - 1_800.0) / 5_400.0)),
        "camera_aux_ratio": 0.01,
        "stuck_gradient_target_ratio": 0.005,
        "mirror_sequence_share": 0.10,
        "cruise_multiplier": 1.0,
        "teacher_gradient_hard_cap": 0.03,
        "auxiliary_gradient_hard_cap": 0.05,
        "mirror_gradient_hard_cap": 0.01,
    }
    if seconds < 1_800.0:
        teacher_ratio = 0.015 * seconds / 1_800.0
        return {
            **common,
            "phase": "fullwarm",
            "navigation_multiplier": 0.15,
            "actor_multiplier": 0.20,
            "critic_multiplier": 0.60,
            "safety_head_multiplier": 1.00,
            "stuck_head_multiplier": 1.00,
            "adapter_multiplier": 0.50,
            "teacher_gradient_target_ratio": teacher_ratio,
            "mirror_gradient_target_ratio": 0.005,
            "entropy_coefficient": 0.006,
        }
    if seconds < 7_200.0:
        entropy = 0.006 + (seconds - 1_800.0) / 5_400.0 * (0.005 - 0.006)
        return {
            **common,
            "phase": "fulladapt",
            "navigation_multiplier": 0.35,
            "actor_multiplier": 0.55,
            "critic_multiplier": 0.80,
            "safety_head_multiplier": 1.00,
            "stuck_head_multiplier": 1.00,
            "adapter_multiplier": 0.50,
            "teacher_gradient_target_ratio": 0.0225,
            "mirror_gradient_target_ratio": 0.005,
            "entropy_coefficient": entropy,
        }
    if seconds < 21_600.0:
        return {
            **common,
            "phase": "fulltrain",
            "navigation_multiplier": 0.30,
            "actor_multiplier": 0.45,
            "critic_multiplier": 0.60,
            "safety_head_multiplier": 0.75,
            "stuck_head_multiplier": 0.75,
            "adapter_multiplier": 0.50,
            "teacher_gradient_target_ratio": 0.020,
            "mirror_gradient_target_ratio": 0.005,
            "entropy_coefficient": 0.005,
        }
    return {
        **common,
        "phase": "fullstabilize",
        "navigation_multiplier": 0.15,
        "actor_multiplier": 0.20,
        "critic_multiplier": 0.35,
        "safety_head_multiplier": 0.50,
        "stuck_head_multiplier": 0.50,
        "adapter_multiplier": 0.25,
        "teacher_gradient_target_ratio": 0.010,
        "mirror_gradient_target_ratio": 0.005,
        "entropy_coefficient": 0.004,
    }


def training_schedule(
    session_effective_seconds: float,
    *,
    branch: str = "actor_attack",
) -> dict[str, float | str | bool]:
    seconds = max(0.0, float(session_effective_seconds))
    branch = str(branch or "actor_attack")
    if branch == "auto":
        branch = "actor_attack"
    if branch not in {
        "actor_attack",
        "visual_recovery",
        "credit_repair",
        "closed_loop_v3",
        "instant_command_r4",
        "instant_repair2h",
    }:
        branch = "actor_attack"
    if branch == "instant_repair2h":
        return _instant_repair_schedule(seconds, branch)
    if branch == "instant_command_r4":
        return _instant_command_schedule(seconds, branch)
    if branch == "closed_loop_v3":
        return _closed_loop_schedule(seconds, branch)
    if branch == "credit_repair":
        return _credit_repair_schedule(seconds, branch)
    return _full_track_schedule(seconds, branch)
