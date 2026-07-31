#!/usr/bin/env python3
"""Worker-owned P3 recovery commands with balanced lateral and yaw signs."""

from __future__ import annotations

import torch

from agent_ppo.feature.command_schedule import CommandSchedule


P3_BUCKET_NAMES = (
    "low_forward",
    "reverse_recovery",
    "forward",
    "joint_turn",
    "lateral",
    "brake_restart",
    "zero",
)


class P3RecoveryCommandSampler(CommandSchedule):
    """A fixed-distribution command owner; no wall-clock domain expansion."""

    def __init__(self, *, num_envs, device, config=None, logger=None):
        merged = {
            "source_hold_s": [2.0, 8.0],
            "target_hold_s": [2.0, 8.0],
            "zero_hold_s": [2.0, 4.0],
            "source_vx": [0.0, 1.0],
            "source_vy": [-0.2, 0.2],
            "source_wz": [-0.3, 0.3],
        }
        merged.update(config or {})
        super().__init__(
            num_envs=num_envs, device=device, config=merged, logger=logger
        )
        self.limits = {"vx": (-0.25, 1.0), "vy": (-0.30, 0.30), "wz": (-0.90, 0.90)}
        weights = merged.get(
            "bucket_weights", [0.25, 0.15, 0.20, 0.15, 0.10, 0.10, 0.05]
        )
        self.p3_weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
        if self.p3_weights.numel() != len(P3_BUCKET_NAMES) or float(self.p3_weights.sum()) <= 0:
            raise ValueError("P3 bucket_weights must contain seven positive-sum values")
        self.p3_weights /= self.p3_weights.sum()
        self.effective_bucket_counts = torch.zeros(
            len(P3_BUCKET_NAMES), dtype=torch.long, device=self.device
        )

    @property
    def target_probability(self) -> float:
        return 1.0

    def _signed_uniform(self, low: float, high: float, count: int) -> torch.Tensor:
        magnitude = self._uniform(low, high, count)
        sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            -torch.ones_like(magnitude),
            torch.ones_like(magnitude),
        )
        return sign * magnitude

    def _sample_target_commands(self, count: int):
        if count <= 0:
            return torch.empty(0, 3, device=self.device), torch.empty(
                0, dtype=torch.long, device=self.device
            )
        buckets = torch.searchsorted(
            self.p3_weights.cumsum(0), torch.rand(count, device=self.device)
        ).long()
        commands = torch.zeros(count, 3, device=self.device)
        low = buckets == 0
        commands[low, 0] = self._uniform(0.05, 0.20, int(low.sum()))
        reverse = buckets == 1
        commands[reverse, 0] = self._uniform(-0.25, -0.05, int(reverse.sum()))
        forward = buckets == 2
        commands[forward, 0] = self._uniform(0.20, 0.85, int(forward.sum()))
        turn = buckets == 3
        commands[turn, 0] = self._uniform(0.05, 0.65, int(turn.sum()))
        commands[turn, 2] = self._signed_uniform(0.10, 0.80, int(turn.sum()))
        lateral = buckets == 4
        commands[lateral, 1] = self._signed_uniform(0.05, 0.30, int(lateral.sum()))
        brake = buckets == 5
        restart = brake & (torch.rand(count, device=self.device) < 0.5)
        commands[restart, 0] = self._uniform(0.05, 0.30, int(restart.sum()))
        return commands, buckets

    def metrics(self):
        result = super().metrics()
        result["target_bucket_counts"] = {
            name: int(self.effective_bucket_counts[index].item())
            for index, name in enumerate(P3_BUCKET_NAMES)
        }
        return result
