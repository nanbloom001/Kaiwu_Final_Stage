#!/usr/bin/env python3
"""Agent-side diagnostics for zero-command action stability.

The platform-owned environment does not expose body velocity, foot-slip, or
individual reward-term samples through the Agent API.  This recorder therefore
only reports quantities it can observe truthfully: commands at the policy
boundary and clipped actions passed to ``env.step``.  Missing environment-side
signals are labeled ``unavailable`` rather than inferred from reward averages.
"""

from __future__ import annotations

from typing import Any

import torch


class ZeroCommandTelemetry:
    """Keep a bounded recent window of zero-command action-delta samples."""

    def __init__(
        self,
        num_envs: int,
        device,
        *,
        capacity: int = 4096,
        command_threshold: float = 0.05,
        grace_period_s: float = 0.4,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.capacity = max(1, int(capacity))
        self.command_threshold = float(command_threshold)
        self.grace_period_s = max(0.0, float(grace_period_s))
        self.zero_elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self._previous_action = None
        self._previous_delta = None
        self._delta_samples = torch.empty(self.capacity, device=self.device)
        self._second_delta_samples = torch.empty(self.capacity, device=self.device)
        self._sample_count = 0

    def _ensure_action_state(self, actions: torch.Tensor) -> None:
        expected = (self.num_envs, actions.shape[1])
        if self._previous_action is not None and tuple(self._previous_action.shape) == expected:
            return
        self._previous_action = torch.zeros_like(actions)
        self._previous_delta = torch.zeros_like(actions)

    def reset(self, reset_mask) -> None:
        """Reset only finished environments; leave other histories untouched."""
        if reset_mask is None:
            return
        mask = torch.as_tensor(reset_mask, device=self.device).reshape(-1).bool()
        if mask.numel() != self.num_envs:
            raise ValueError("zero telemetry reset mask size does not match num_envs")
        self.zero_elapsed_s[mask] = 0.0
        if self._previous_action is not None:
            self._previous_action[mask] = 0.0
            self._previous_delta[mask] = 0.0

    def observe(
        self,
        command: torch.Tensor,
        clipped_actions: torch.Tensor,
        *,
        reset_mask=None,
        dt_s: float,
    ) -> None:
        """Record action rates for commands that survived the zero grace period."""
        command = command.to(self.device, dtype=torch.float32)
        actions = clipped_actions.to(self.device, dtype=torch.float32)
        if command.ndim != 2 or tuple(command.shape) != (self.num_envs, 3):
            raise ValueError("zero telemetry requires command [num_envs, 3]")
        if actions.ndim != 2 or actions.shape[0] != self.num_envs:
            raise ValueError("zero telemetry requires action [num_envs, action_dim]")
        self.reset(reset_mask)
        self._ensure_action_state(actions)

        zero_mask = torch.linalg.vector_norm(command, dim=1) < self.command_threshold
        self.zero_elapsed_s[zero_mask] += max(0.0, float(dt_s))
        self.zero_elapsed_s[~zero_mask] = 0.0
        active = zero_mask & (self.zero_elapsed_s >= self.grace_period_s)

        delta_vector = actions - self._previous_action
        second_delta_vector = delta_vector - self._previous_delta
        if bool(active.any()):
            delta = torch.sqrt(delta_vector.square().mean(dim=1))[active]
            second_delta = torch.sqrt(second_delta_vector.square().mean(dim=1))[active]
            self._append(delta, second_delta)

        self._previous_action.copy_(actions)
        self._previous_delta.copy_(delta_vector)

    def _append(self, delta: torch.Tensor, second_delta: torch.Tensor) -> None:
        count = int(delta.numel())
        if count <= 0:
            return
        if count > self.capacity:
            delta = delta[-self.capacity :]
            second_delta = second_delta[-self.capacity :]
            count = self.capacity
        slots = (
            torch.arange(count, device=self.device) + self._sample_count
        ) % self.capacity
        self._delta_samples[slots] = delta
        self._second_delta_samples[slots] = second_delta
        self._sample_count += count

    def _summary(self, values: torch.Tensor) -> tuple[float | None, float | None]:
        count = min(self._sample_count, self.capacity)
        if count <= 0:
            return None, None
        recent = values[:count]
        return float(recent.mean().item()), float(torch.quantile(recent, 0.95).item())

    def metrics(self) -> dict[str, Any]:
        delta_mean, delta_p95 = self._summary(self._delta_samples)
        second_mean, second_p95 = self._summary(self._second_delta_samples)
        unavailable = "unavailable:no_agent_environment_telemetry"
        return {
            "zero_telemetry_source": "workflow_clipped_action_after_worker_command_observation",
            "zero_telemetry_window_samples": min(self._sample_count, self.capacity),
            "zero_action_delta_mean": delta_mean,
            "zero_action_delta_p95": delta_p95,
            "zero_action_second_delta_mean": second_mean,
            "zero_action_second_delta_p95": second_p95,
            "zero_root_v_xy": unavailable,
            "zero_root_omega_xy": unavailable,
            "zero_foot_slide": unavailable,
            "zero_stability_penalty": unavailable,
        }
