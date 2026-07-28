#!/usr/bin/env python3
"""P1.5 low-level PPO plus an independently optimized response adapter."""

from __future__ import annotations

import os
from uuid import uuid4

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.algorithm.algorithm_visual_ppo import AlgorithmVisualPPO
from agent_ppo.checkpoint_io import (
    KAIWU_TRAIN_SCHEMA_V2,
    normalize_kaiwu_train_bundle,
)
from agent_ppo.feature import p15_contract
from agent_ppo.feature.feedback_emulator import feedback_implementation_digest
from agent_ppo.model.response_adapter import response_adapter_spec


class AlgorithmP15Response(AlgorithmVisualPPO):
    """Keep PPO and response supervision physically separated."""

    STAGE_TYPE = "p15_response_adapter"

    def __init__(
        self,
        *,
        response_adapter: nn.Module,
        response_optimizer: torch.optim.Optimizer,
        response_scheduler,
        low_level_scheduler,
        response_buffer,
        response_config: dict | None = None,
        p15_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        config = dict(response_config or {})
        self.p15_config = dict(p15_config or {})
        self.response_adapter = response_adapter
        self.response_optimizer = response_optimizer
        self.response_scheduler = response_scheduler
        self.low_level_scheduler = low_level_scheduler
        self.response_buffer = response_buffer
        self.response_batch_envs = int(config.get("batch_envs", 64))
        self.response_updates_per_iteration = int(
            config.get("updates_per_iteration", 1)
        )
        self.response_max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.response_loss_weights = {
            "velocity": float(config.get("velocity_loss_weight", 1.0)),
            "pose": float(config.get("pose_loss_weight", 0.5)),
            "stuck": float(config.get("stuck_loss_weight", 0.25)),
            "nll": float(config.get("nll_loss_weight", 0.1)),
        }
        self.horizon_weights = torch.tensor(
            config.get("velocity_horizon_weights", [1.0, 0.75, 0.5]),
            device=self.device,
            dtype=torch.float32,
        )
        self.stuck_positive_ema = float(config.get("stuck_positive_ema_start", 0.1))
        self.stuck_ema_decay = float(config.get("stuck_ema_decay", 0.99))
        self.adapter_iteration = 0
        self.adapter_gradient_steps = 0
        self.adapter_skipped_nonfinite = 0
        self.low_level_gradient_steps = 0
        self.low_level_skipped_nonfinite = 0
        self.total_env_steps = 0
        self.command_event_counts = torch.zeros(5, device=self.device, dtype=torch.long)
        self.trajectory_event_counts = torch.zeros(3, device=self.device, dtype=torch.long)
        self.coverage_counts = torch.zeros(4, 4, 2, device=self.device, dtype=torch.long)
        self.terrain_level_command_counts = torch.zeros(
            8, 10, 5, device=self.device, dtype=torch.long
        )
        self._last_command_epoch = None
        self._last_transport_metrics = {}
        self.low_level_state_digest = "uninitialized"
        self.adapter_generator = torch.Generator(device=torch.device(self.device))
        self.adapter_generator.manual_seed(int(config.get("seed", 1501)))
        self._verify_optimizer_isolation()
        self._compute_low_level_digest()

    def _verify_optimizer_isolation(self) -> None:
        low_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        adapter_ids = {
            id(parameter)
            for group in self.response_optimizer.param_groups
            for parameter in group["params"]
        }
        overlap = low_ids & adapter_ids
        if overlap:
            raise ValueError(
                "P1.5 optimizer parameter sets overlap; PPO and adapter must be isolated"
            )

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=value.device, dtype=value.dtype)
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1.0)

    def observe_response_aux(self, aux: torch.Tensor, dones: torch.Tensor) -> None:
        self.response_buffer.set_low_level_version(
            self.low_level_state_digest, self.current_iteration
        )
        self.response_buffer.append(aux, dones)
        self.total_env_steps += int(aux.shape[0])
        target = aux[:, 0:3]
        executed = aux[:, 3:6]
        actual = aux[:, 12:15]
        self._last_transport_metrics = {
            "target_exec_error": float((target - executed).abs().mean().item()),
            "exec_actual_error": float((executed - actual).abs().mean().item()),
            "response_valid_rate": float((aux[:, 9] > 0.5).float().mean().item()),
            "command_phase": float(aux[:, 27].float().mean().item()),
        }
        terrain_family = aux[:, 28].round().long().clamp(0, 7)
        terrain_level = aux[:, 29].round().long().clamp(0, 9)
        frame_family = aux[:, 24].round().long().clamp(0, 4)
        flat_histogram = self.terrain_level_command_counts.view(-1)
        flat_index = (terrain_family * 10 + terrain_level) * 5 + frame_family
        flat_histogram.scatter_add_(
            0, flat_index, torch.ones_like(flat_index, dtype=torch.long)
        )
        epoch = aux[:, 26].long()
        if self._last_command_epoch is None:
            changed = torch.ones_like(epoch, dtype=torch.bool)
        else:
            changed = epoch != self._last_command_epoch
        self._last_command_epoch = epoch.clone()
        if bool(changed.any()):
            family = aux[changed, 24].long().clamp(0, 4)
            trajectory = aux[changed, 25].long().clamp(0, 2)
            self.command_event_counts.scatter_add_(
                0, family, torch.ones_like(family, dtype=torch.long)
            )
            self.trajectory_event_counts.scatter_add_(
                0, trajectory, torch.ones_like(trajectory, dtype=torch.long)
            )
            changed_target = target[changed]
            vx_bin = torch.clamp((changed_target[:, 0] / 1.0 * 4.0).long(), 0, 3)
            wz_bin = torch.clamp((changed_target[:, 2].abs() / 0.8 * 4.0).long(), 0, 3)
            sign_bin = (changed_target[:, 2] >= 0.0).long()
            flat = vx_bin * 8 + wz_bin * 2 + sign_bin
            flat_counts = self.coverage_counts.view(-1)
            flat_counts.scatter_add_(
                0, flat, torch.ones_like(flat, dtype=torch.long)
            )

    def _adapter_update(self) -> dict[str, float]:
        totals = {
            "adapter_loss": 0.0,
            "adapter_velocity_loss": 0.0,
            "adapter_pose_loss": 0.0,
            "adapter_stuck_loss": 0.0,
            "adapter_nll_loss": 0.0,
            "adapter_velocity_mae": 0.0,
            "adapter_velocity_mae_0p2s": 0.0,
            "adapter_velocity_mae_0p6s": 0.0,
            "adapter_velocity_mae_1p0s": 0.0,
            "adapter_zero_baseline_mae": 0.0,
            "adapter_copy_exec_baseline_mae": 0.0,
            "adapter_applied_updates": 0.0,
        }
        for _ in range(max(0, self.response_updates_per_iteration)):
            batch = self.response_buffer.sample(
                batch_envs=self.response_batch_envs,
                generator=self.adapter_generator,
            )
            if batch is None:
                break
            hidden = None
            if batch.burn_in_observations.shape[0] > 0:
                with torch.no_grad():
                    _, hidden = self.response_adapter(
                        batch.burn_in_observations,
                        reset_mask=batch.burn_in_reset_mask,
                    )
                hidden = hidden.detach()
            profile, _ = self.response_adapter(
                batch.observations,
                hidden=hidden,
                reset_mask=batch.reset_mask,
            )
            parts = self.response_adapter.split_profile(profile)
            velocity_error = parts["velocity"] - batch.velocity_labels
            horizon_weights = self.horizon_weights.to(
                device=velocity_error.device, dtype=velocity_error.dtype
            ).view(1, 1, 3)
            velocity_per_horizon = F.smooth_l1_loss(
                parts["velocity"], batch.velocity_labels, reduction="none"
            ).mean(dim=-1)
            weighted_mask = batch.horizon_mask.to(velocity_per_horizon.dtype) * horizon_weights
            velocity_loss = (
                velocity_per_horizon * weighted_mask
            ).sum() / weighted_mask.sum().clamp_min(1.0)
            velocity_mae = (
                velocity_error.abs().mean(dim=-1) * weighted_mask
            ).sum() / weighted_mask.sum().clamp_min(1.0)
            horizon_mae = []
            zero_horizon_mae = []
            copy_horizon_mae = []
            copy_exec = batch.observations[..., 3:6]
            for horizon in range(3):
                horizon_mask = batch.horizon_mask[..., horizon]
                horizon_mae.append(
                    self._masked_mean(
                        velocity_error[..., horizon, :].abs(), horizon_mask
                    )
                )
                zero_horizon_mae.append(
                    self._masked_mean(
                        batch.velocity_labels[..., horizon, :].abs(), horizon_mask
                    )
                )
                copy_horizon_mae.append(
                    self._masked_mean(
                        (
                            copy_exec
                            - batch.velocity_labels[..., horizon, :]
                        ).abs(),
                        horizon_mask,
                    )
                )

            pose_loss = self._masked_mean(
                F.smooth_l1_loss(
                    parts["pose_delta"], batch.pose_labels, reduction="none"
                ),
                batch.pose_mask,
            )
            pose_valid = batch.pose_mask.to(batch.stuck_labels.dtype)
            positive_rate = float(
                (batch.stuck_labels * pose_valid).sum().item()
                / pose_valid.sum().clamp_min(1.0).item()
            )
            self.stuck_positive_ema = (
                self.stuck_ema_decay * self.stuck_positive_ema
                + (1.0 - self.stuck_ema_decay) * positive_rate
            )
            pos_weight = min(
                10.0,
                max(1.0, (1.0 - self.stuck_positive_ema) / max(1.0e-4, self.stuck_positive_ema)),
            )
            stuck_loss = self._masked_mean(
                F.binary_cross_entropy_with_logits(
                    parts["stuck_logit"],
                    batch.stuck_labels,
                    pos_weight=torch.tensor(
                        pos_weight,
                        device=parts["stuck_logit"].device,
                        dtype=parts["stuck_logit"].dtype,
                    ),
                    reduction="none",
                ),
                batch.pose_mask,
            )
            # The fixed 16-D response profile predicts XYZ uncertainty only
            # for the 1.0-second velocity head. Do not reinterpret those three
            # values as one scalar uncertainty per horizon.
            log_sigma = parts["velocity_log_sigma"]
            one_second_error = velocity_error[..., 2, :]
            nll_elements = 0.5 * (
                one_second_error.square() * torch.exp(-2.0 * log_sigma)
                + 2.0 * log_sigma
            )
            nll_loss = self._masked_mean(
                nll_elements, batch.horizon_mask[..., 2]
            )
            loss = (
                self.response_loss_weights["velocity"] * velocity_loss
                + self.response_loss_weights["pose"] * pose_loss
                + self.response_loss_weights["stuck"] * stuck_loss
                + self.response_loss_weights["nll"] * nll_loss
            )
            if not bool(torch.isfinite(loss)):
                self.adapter_skipped_nonfinite += 1
                if self.logger:
                    self.logger.warning(
                        "[P15Response] nonfinite adapter minibatch skipped; training continues"
                    )
                continue
            self.response_optimizer.zero_grad()
            loss.backward()
            gradients_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in self.response_adapter.parameters()
            )
            if not gradients_finite:
                self.response_optimizer.zero_grad()
                self.adapter_skipped_nonfinite += 1
                if self.logger:
                    self.logger.warning(
                        "[P15Response] nonfinite adapter gradient skipped; training continues"
                    )
                continue
            nn.utils.clip_grad_norm_(
                self.response_adapter.parameters(), self.response_max_grad_norm
            )
            self.response_optimizer.step()
            if self.response_scheduler is not None:
                self.response_scheduler.step()
            self.adapter_gradient_steps += 1
            totals["adapter_loss"] += float(loss.item())
            totals["adapter_velocity_loss"] += float(velocity_loss.item())
            totals["adapter_pose_loss"] += float(pose_loss.item())
            totals["adapter_stuck_loss"] += float(stuck_loss.item())
            totals["adapter_nll_loss"] += float(nll_loss.item())
            totals["adapter_velocity_mae"] += float(velocity_mae.item())
            for index, suffix in enumerate(("0p2s", "0p6s", "1p0s")):
                totals[f"adapter_velocity_mae_{suffix}"] += float(
                    horizon_mae[index].item()
                )
            totals["adapter_zero_baseline_mae"] += float(
                torch.stack(zero_horizon_mae).mean().item()
            )
            totals["adapter_copy_exec_baseline_mae"] += float(
                torch.stack(copy_horizon_mae).mean().item()
            )
            totals["adapter_applied_updates"] += 1.0
        applied = max(1.0, totals["adapter_applied_updates"])
        for key in tuple(totals):
            if key != "adapter_applied_updates":
                totals[key] /= applied
        self.adapter_iteration += 1
        totals["adapter_stuck_pos_weight"] = min(
            10.0,
            max(1.0, (1.0 - self.stuck_positive_ema) / max(1.0e-4, self.stuck_positive_ema)),
        )
        totals["adapter_skipped_nonfinite"] = float(self.adapter_skipped_nonfinite)
        totals["adapter_gain_vs_zero"] = (
            totals["adapter_zero_baseline_mae"] - totals["adapter_velocity_mae"]
        )
        totals["adapter_gain_vs_copy_exec"] = (
            totals["adapter_copy_exec_baseline_mae"]
            - totals["adapter_velocity_mae"]
        )
        return totals

    def learn(self, elapsed_h: float | None = None) -> dict[str, float]:
        elapsed_h = self.anchor_session_elapsed_hours if elapsed_h is None else float(elapsed_h)
        if elapsed_h < p15_contract.P15_LOW_LEVEL_FREEZE_HOURS:
            low_metrics = super().learn(elapsed_h)
            self.low_level_gradient_steps += int(low_metrics.get("applied_updates", 0.0))
            self.low_level_skipped_nonfinite = int(
                low_metrics.get(
                    "skipped_nonfinite_updates", self.low_level_skipped_nonfinite
                )
            )
            if (
                self.low_level_scheduler is not None
                and low_metrics.get("applied_updates", 0.0) > 0
            ):
                self.low_level_scheduler.step()
            self._compute_low_level_digest()
        else:
            self.anchor_session_elapsed_hours = elapsed_h
            self.current_phase = "responsecalib"
            self.action_anchor_weight = self.command_anchor_action
            self.latent_anchor_weight_current = self.command_anchor_latent
            self._set_trainable_phase(self.current_phase)
            self.current_iteration += 1
            self.train_step += 1
            low_metrics = {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
                "action_anchor_loss": 0.0,
                "latent_anchor_loss": 0.0,
                "anchor_action_mse": 0.0,
                "hard_termination_rate": float(
                    self.storage.hard_terminations.float().mean().item()
                ),
                "applied_updates": 0.0,
            }
        # P1.5 never converts metric warnings into diagnostic-save pressure.
        self.last_diagnostic_save_requested = False
        adapter_metrics = self._adapter_update()
        buffer_state = self.response_buffer.state_dict()
        event_total = max(1, int(self.command_event_counts.sum().item()))
        trajectory_total = max(1, int(self.trajectory_event_counts.sum().item()))
        telemetry = {
            **self._last_transport_metrics,
            "response_buffer_append_calls": float(buffer_state["append_calls"]),
            "response_buffer_history_length": float(
                buffer_state["current_history_length"]
            ),
            "response_buffer_max_history_length": float(
                buffer_state["max_history_length"]
            ),
            "response_buffer_record_steps": float(buffer_state["record_steps"]),
            "response_buffer_version_resets": float(
                buffer_state["version_reset_count"]
            ),
            "command_coverage_ratio": float(
                (self.coverage_counts > 0).float().mean().item()
            ),
            "terrain_level_command_coverage_ratio": float(
                (self.terrain_level_command_counts > 0).float().mean().item()
            ),
        }
        for index, name in enumerate(p15_contract.COMMAND_FAMILIES):
            telemetry[f"command_family_{name}_ratio"] = (
                float(self.command_event_counts[index].item()) / event_total
            )
        for index, name in enumerate(p15_contract.TRAJECTORY_MODES):
            telemetry[f"trajectory_{name}_ratio"] = (
                float(self.trajectory_event_counts[index].item()) / trajectory_total
            )
        terrain_total = max(1, int(self.terrain_level_command_counts.sum().item()))
        for level in range(10):
            telemetry[f"terrain_level_{level}_ratio"] = float(
                self.terrain_level_command_counts[:, level, :].sum().item()
            ) / terrain_total
        if self.logger and self.current_iteration % 10 == 0:
            self.logger.info(
                "[P15Response] terrain_level_command_histogram="
                f"{self.terrain_level_command_counts.detach().cpu().tolist()}"
            )
            self.logger.info(
                "[P15Response] buffer "
                f"append_calls={buffer_state['append_calls']} "
                f"history={buffer_state['current_history_length']} "
                f"max_history={buffer_state['max_history_length']} "
                f"records={buffer_state['record_steps']} "
                f"version_resets={buffer_state['version_reset_count']} "
                f"valid_horizons={buffer_state['valid_horizon_counts']} "
                f"active_version={buffer_state['active_low_level_version']} "
                f"adapter_gradient_steps={self.adapter_gradient_steps}"
            )
        return {
            **low_metrics,
            **adapter_metrics,
            **telemetry,
            "low_level_frozen": float(
                elapsed_h >= p15_contract.P15_LOW_LEVEL_FREEZE_HOURS
            ),
            "low_level_gradient_steps": float(self.low_level_gradient_steps),
            "low_level_skipped_nonfinite": float(
                self.low_level_skipped_nonfinite
            ),
            "total_env_steps": float(self.total_env_steps),
        }

    def _compute_low_level_digest(self) -> str:
        digest = __import__("hashlib").sha256()
        for prefix, module in (
            ("vision", self.actor_critic.vision_encoder),
            ("actor", self.actor_critic.actor),
            ("critic", self.actor_critic.critic),
        ):
            for name, value in sorted(module.state_dict().items()):
                digest.update(f"{prefix}.{name}".encode("ascii"))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        if hasattr(self.actor_critic, "std"):
            digest.update(self.actor_critic.std.detach().cpu().contiguous().numpy().tobytes())
        self.low_level_state_digest = digest.hexdigest()
        return self.low_level_state_digest

    def save_training_bundle(self, path: str, **kwargs) -> str:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        legacy_tmp = os.path.join(directory, f".{os.path.basename(path)}.{uuid4().hex}.legacy")
        atomic_tmp = os.path.join(directory, f".{os.path.basename(path)}.{uuid4().hex}.tmp")
        try:
            super().save_training_bundle(legacy_tmp, **kwargs)
            payload = torch.load(legacy_tmp, weights_only=False, map_location="cpu")
            payload, _ = normalize_kaiwu_train_bundle(payload)
            payload["schema_version"] = KAIWU_TRAIN_SCHEMA_V2
            payload["stage_type"] = self.STAGE_TYPE
            modules = payload["modules"]
            low_level = modules["low_level"]
            low_level["locomotion_encoder"] = {
                "state_dict": modules["vision_encoder"]["state_dict"]
            }
            low_level["actor"] = {"state_dict": low_level["actor_state_dict"]}
            low_level["critic"] = {"state_dict": modules["critic"]["state_dict"]}
            modules["high_level"] = {
                "component_status": "adapter_only",
                "response_adapter": {
                    "spec": response_adapter_spec(),
                    "state_dict": self.response_adapter.state_dict(),
                },
            }
            payload["optimizers"]["response_adapter"] = self.response_optimizer.state_dict()
            payload["schedulers"] = {
                "low_level": (
                    self.low_level_scheduler.state_dict()
                    if self.low_level_scheduler is not None
                    else None
                ),
                "response_adapter": (
                    self.response_scheduler.state_dict()
                    if self.response_scheduler is not None
                    else None
                ),
            }
            low_state = dict(payload["training_state"])
            low_state.update(
                {
                    "gradient_steps": self.low_level_gradient_steps,
                    "skipped_nonfinite": self.low_level_skipped_nonfinite,
                    "env_steps": self.total_env_steps,
                }
            )
            adapter_state = {
                "iteration": self.adapter_iteration,
                "gradient_steps": self.adapter_gradient_steps,
                "skipped_nonfinite": self.adapter_skipped_nonfinite,
                "stuck_positive_ema": self.stuck_positive_ema,
                "rng_state": self.adapter_generator.get_state().cpu(),
                "buffer": self.response_buffer.checkpoint_state(),
                "command_event_counts": self.command_event_counts.cpu(),
                "trajectory_event_counts": self.trajectory_event_counts.cpu(),
                "coverage_counts": self.coverage_counts.cpu(),
                "terrain_level_command_counts": (
                    self.terrain_level_command_counts.cpu()
                ),
            }
            payload["training_states"] = {
                "global": {
                    "compound_schedule_phase": self.current_phase,
                    "session_elapsed_hours": self.anchor_session_elapsed_hours,
                    "cumulative_elapsed_hours": self.elapsed_training_hours,
                    "env_steps": self.total_env_steps,
                },
                "low_level": low_state,
                "response_adapter": adapter_state,
            }
            command_contract = p15_contract.command_contract()
            command_contract["runtime_config"] = dict(
                self.p15_config.get("command_schedule") or {}
            )
            feedback_contract = p15_contract.feedback_contract()
            feedback_contract["profile"] = dict(
                self.p15_config.get("feedback_profile")
                or p15_contract.FEEDBACK_PROFILE
            )
            feedback_contract["implementation"] = {
                "module": "agent_ppo.feature.feedback_emulator",
                "sha256": feedback_implementation_digest(),
            }
            payload["contracts"] = {
                "command": command_contract,
                "command_digest": p15_contract.stable_digest(command_contract),
                "feedback": feedback_contract,
                "feedback_digest": p15_contract.stable_digest(feedback_contract),
                "critic_transport": {
                    "wire_dim": p15_contract.PRIVILEGED_WIRE_DIM,
                    "critic_dim": p15_contract.CRITIC_OBS_DIM,
                    "response_aux_dim": p15_contract.RESPONSE_AUX_DIM,
                    "split_owner": "aisrv_before_ppo_storage",
                },
            }
            low_digest = self._compute_low_level_digest()
            payload.setdefault("lineage", {})["low_level_state_digest"] = low_digest
            payload["capabilities"].update(
                {
                    "envelope_type": "piecewise_union_v1",
                    "vx_wz_continuous": True,
                    "vy_specialized": True,
                    "high_level_component": "adapter_only",
                }
            )
            torch.save(payload, atomic_tmp)
            if not os.path.isfile(atomic_tmp) or os.path.getsize(atomic_tmp) <= 0:
                raise IOError(f"P1.5 checkpoint temporary file is empty: {atomic_tmp}")
            os.replace(atomic_tmp, path)
            return self._sha256(path)
        finally:
            for candidate in (legacy_tmp, atomic_tmp):
                try:
                    if os.path.exists(candidate):
                        os.remove(candidate)
                except OSError:
                    pass

    def load_training_bundle(self, path: str, **kwargs) -> str:
        raw = torch.load(path, weights_only=False, map_location=self.device)
        normalized, report = normalize_kaiwu_train_bundle(raw)
        load_mode = super().load_training_bundle(path, **kwargs)
        low_level_tensors = [
            *self.actor_critic.vision_encoder.state_dict().values(),
            *self.actor_critic.actor.state_dict().values(),
            *self.actor_critic.critic.state_dict().values(),
        ]
        if hasattr(self.actor_critic, "std"):
            low_level_tensors.append(self.actor_critic.std)
        if not all(bool(torch.isfinite(value).all()) for value in low_level_tensors):
            raise ValueError("P1.5 low-level checkpoint contains nonfinite weights")
        if int(raw.get("schema_version", 1)) == KAIWU_TRAIN_SCHEMA_V2:
            resume_warnings = []
            high_level = normalized.get("modules", {}).get("high_level", {})
            adapter_leaf = high_level.get("response_adapter", {})
            state_dict = adapter_leaf.get("state_dict")
            if not isinstance(state_dict, dict):
                raise KeyError(
                    "schema2 P1.5 resume requires modules.high_level.response_adapter.state_dict"
                )
            self.response_adapter.load_state_dict(state_dict, strict=True)
            if not all(
                bool(torch.isfinite(value).all())
                for value in self.response_adapter.state_dict().values()
            ):
                raise ValueError("P1.5 response adapter checkpoint contains nonfinite weights")
            optimizer_state = normalized.get("optimizers", {}).get("response_adapter")
            if not isinstance(optimizer_state, dict):
                raise KeyError("schema2 P1.5 resume requires optimizers.response_adapter")
            self.response_optimizer.load_state_dict(optimizer_state)
            scheduler_state = normalized.get("schedulers", {}).get("response_adapter")
            if self.response_scheduler is not None and isinstance(scheduler_state, dict):
                self.response_scheduler.load_state_dict(scheduler_state)
            elif self.response_scheduler is not None:
                resume_warnings.append("response_adapter scheduler state missing")
            low_scheduler_state = normalized.get("schedulers", {}).get("low_level")
            if self.low_level_scheduler is not None and isinstance(
                low_scheduler_state, dict
            ):
                self.low_level_scheduler.load_state_dict(low_scheduler_state)
            elif self.low_level_scheduler is not None:
                resume_warnings.append("low_level scheduler state missing")
            adapter_state = normalized.get("training_states", {}).get(
                "response_adapter", {}
            )
            if not isinstance(adapter_state, dict):
                adapter_state = {}
                resume_warnings.append("response_adapter training state missing")
            for required_key in (
                "iteration",
                "gradient_steps",
                "skipped_nonfinite",
                "rng_state",
                "buffer",
            ):
                if required_key not in adapter_state:
                    resume_warnings.append(
                        f"response_adapter training state missing {required_key}"
                    )
            self.adapter_iteration = int(adapter_state.get("iteration", 0))
            self.adapter_gradient_steps = int(adapter_state.get("gradient_steps", 0))
            self.adapter_skipped_nonfinite = int(
                adapter_state.get("skipped_nonfinite", 0)
            )
            self.stuck_positive_ema = float(
                adapter_state.get("stuck_positive_ema", self.stuck_positive_ema)
            )
            buffer_mode = self.response_buffer.load_checkpoint_state(
                adapter_state.get("buffer", {})
            )
            rng_state = adapter_state.get("rng_state")
            if torch.is_tensor(rng_state):
                self.adapter_generator.set_state(rng_state.cpu())
            else:
                resume_warnings.append("response_adapter RNG state missing")
            for name in (
                "command_event_counts",
                "trajectory_event_counts",
                "coverage_counts",
                "terrain_level_command_counts",
            ):
                saved = adapter_state.get(name)
                current = getattr(self, name)
                if torch.is_tensor(saved) and tuple(saved.shape) == tuple(current.shape):
                    current.copy_(saved.to(device=current.device, dtype=current.dtype))
            global_state = normalized.get("training_states", {}).get("global", {})
            self.total_env_steps = int(global_state.get("env_steps", 0))
            low_state = normalized.get("training_states", {}).get("low_level", {})
            self.low_level_gradient_steps = int(low_state.get("gradient_steps", 0))
            self.low_level_skipped_nonfinite = int(
                low_state.get(
                    "skipped_nonfinite",
                    low_state.get("skipped_nonfinite_updates", 0),
                )
            )
            if buffer_mode.startswith("buffer_reset_"):
                resume_warnings.append(f"response buffer {buffer_mode}")
            load_mode = (
                "warm_start"
                if resume_warnings
                else "weights_optim_rng_resume_with_history_reset"
            )
            if self.logger:
                for warning in resume_warnings:
                    self.logger.warning(
                        "[P15Response] checkpoint resume degraded to warm_start: "
                        f"{warning}"
                    )
        elif self.logger:
            self.logger.info(
                "[P15Response] schema1 parent migrated in memory; response adapter starts fresh"
            )
        self._compute_low_level_digest()
        if self.logger:
            self.logger.info(
                "[P15Response] checkpoint normalized "
                f"source_schema={report['source_schema']} load_mode={load_mode}"
            )
        return load_mode
