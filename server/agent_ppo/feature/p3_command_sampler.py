#!/usr/bin/env python3
"""Worker-owned P3 recovery commands with balanced lateral and yaw signs."""

from __future__ import annotations

import torch

from agent_ppo.feature.command_schedule import CommandSchedule
from agent_ppo.feature import p3_contract


P3_BUCKET_NAMES = (
    "straight",
    "reserved_reverse",
    "vx_vy",
    "vx_wz",
    "pure_yaw",
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
        self.limits = {"vx": (0.0, 1.0), "vy": (-0.30, 0.30), "wz": (-0.90, 0.90)}
        self.generator = torch.Generator(device=self.device).manual_seed(
            int(merged.get("seed", p3_contract.P3_COMMAND_SEED))
        )
        weights = merged.get(
            "bucket_weights", [0.25, 0.00, 0.10, 0.35, 0.08, 0.15, 0.07]
        )
        self.p3_weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
        if (
            self.p3_weights.numel() != len(P3_BUCKET_NAMES)
            or bool((self.p3_weights < 0.0).any())
            or float(self.p3_weights.sum()) <= 0
        ):
            raise ValueError("P3 bucket_weights must contain seven nonnegative values with a positive sum")
        if float(self.p3_weights[1]) != 0.0:
            raise ValueError("P3 reverse_recovery bucket is compatibility-only and must have zero weight")
        self.p3_weights /= self.p3_weights.sum()
        self.effective_bucket_counts = torch.zeros(
            len(P3_BUCKET_NAMES), dtype=torch.long, device=self.device
        )

    @property
    def target_probability(self) -> float:
        return 1.0

    def _uniform(self, low: float, high: float, count: int) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, device=self.device)
        return low + (high - low) * torch.rand(
            count, device=self.device, generator=self.generator
        )

    def _sample_hold(
        self, *, target: bool, buckets: torch.Tensor | None = None, count: int | None = None
    ) -> torch.Tensor:
        if buckets is not None:
            count = int(buckets.numel())
        elif count is None:
            count = self.num_envs
        if buckets is None:
            buckets = torch.full((count,), -1, dtype=torch.long, device=self.device)
        low, high = self.target_hold_s if target else self.source_hold_s
        values = self._uniform(low, high, count)
        zero = buckets == 6
        if bool(zero.any()):
            values[zero] = self._uniform(*self.zero_hold_s, int(zero.sum()))
        return values

    @staticmethod
    def _anchor_weight_for_targets(
        buckets: torch.Tensor, _commands: torch.Tensor
    ) -> torch.Tensor:
        weights = torch.zeros_like(buckets, dtype=torch.float32)
        weights[buckets == 0] = 0.35
        weights[buckets == 2] = 0.30
        weights[buckets == 3] = 0.25
        weights[buckets == 4] = 0.15
        weights[buckets == 5] = 0.10
        weights[buckets == 6] = 0.10
        return weights

    def _signed_uniform(self, low: float, high: float, count: int) -> torch.Tensor:
        magnitude = self._uniform(low, high, count)
        sign = torch.where(
            torch.rand(count, device=self.device, generator=self.generator) < 0.5,
            -torch.ones_like(magnitude),
            torch.ones_like(magnitude),
        )
        return sign * magnitude

    def _sample_vx(self, count: int) -> torch.Tensor:
        """Low/medium/high command bands with fixed 55/30/15 shares."""
        if count <= 0:
            return torch.empty(0, device=self.device)
        selector = torch.rand(count, device=self.device, generator=self.generator)
        result = torch.empty(count, device=self.device)
        low = selector < 0.55
        medium = (selector >= 0.55) & (selector < 0.85)
        high = selector >= 0.85
        result[low] = self._uniform(0.10, 0.35, int(low.sum()))
        result[medium] = self._uniform(0.35, 0.70, int(medium.sum()))
        result[high] = self._uniform(0.70, 1.00, int(high.sum()))
        return result

    def _sample_wz(self, count: int) -> torch.Tensor:
        """Symmetric low/medium/high yaw bands with 35/40/25 shares."""
        if count <= 0:
            return torch.empty(0, device=self.device)
        selector = torch.rand(count, device=self.device, generator=self.generator)
        result = torch.empty(count, device=self.device)
        low = selector < 0.35
        medium = (selector >= 0.35) & (selector < 0.75)
        high = selector >= 0.75
        result[low] = self._signed_uniform(0.05, 0.25, int(low.sum()))
        result[medium] = self._signed_uniform(0.25, 0.55, int(medium.sum()))
        result[high] = self._signed_uniform(0.55, 0.90, int(high.sum()))
        return result

    def _sample_target_commands(self, count: int):
        if count <= 0:
            return torch.empty(0, 3, device=self.device), torch.empty(
                0, dtype=torch.long, device=self.device
            )
        buckets = torch.searchsorted(
            self.p3_weights.cumsum(0),
            torch.rand(count, device=self.device, generator=self.generator),
        ).long()
        commands = torch.zeros(count, 3, device=self.device)
        straight = buckets == 0
        commands[straight, 0] = self._sample_vx(int(straight.sum()))
        # Bucket 1 remains reserved for schema compatibility.  Its production
        # weight is zero and a custom config cannot turn it into negative vx.
        reverse = buckets == 1
        commands[reverse, 0] = 0.0
        diagonal = buckets == 2
        commands[diagonal, 0] = self._sample_vx(int(diagonal.sum()))
        commands[diagonal, 1] = self._signed_uniform(
            0.04, 0.25, int(diagonal.sum())
        )
        turn = buckets == 3
        commands[turn, 0] = self._sample_vx(int(turn.sum()))
        commands[turn, 2] = self._sample_wz(int(turn.sum()))
        pure_yaw = buckets == 4
        commands[pure_yaw, 2] = self._sample_wz(int(pure_yaw.sum()))
        brake = buckets == 5
        restart = brake & (
            torch.rand(count, device=self.device, generator=self.generator) < 0.5
        )
        commands[restart, 0] = self._sample_vx(int(restart.sum()))
        return commands, buckets

    def metrics(self):
        result = super().metrics()
        result["target_bucket_counts"] = {
            name: int(self.effective_bucket_counts[index].item())
            for index, name in enumerate(P3_BUCKET_NAMES)
        }
        result["negative_vx_command_count"] = int(
            (self.command[:, 0] < 0.0).sum().item()
        ) if self.command is not None else 0
        return result
