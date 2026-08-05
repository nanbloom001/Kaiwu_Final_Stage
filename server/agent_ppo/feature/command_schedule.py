#!/usr/bin/env python3
"""Wall-clock command mixing for Standard visual command generalization.

This module deliberately has no Isaac Lab imports. It plans source/target
commands and keeps per-environment state; :mod:`worker_command_bridge` owns
publication inside the real Isaac environment worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


BUCKET_NAMES = (
    "zero",
    "low_forward",
    "normal_forward",
    "forward_yaw",
    "pure_yaw",
    "lateral",
)


def anchor_weights_from_commands(commands: torch.Tensor) -> torch.Tensor:
    """Derive S0 anchor weights from commands visible to the policy."""
    if commands.ndim != 2 or commands.shape[1] < 3:
        raise ValueError(f"expected command tensor [N,3+], got {tuple(commands.shape)}")
    commands = commands[:, :3]
    if not bool(torch.isfinite(commands).all()):
        raise RuntimeError("command tensor contains non-finite values")
    vx = commands[:, 0]
    weights = torch.zeros(commands.shape[0], 1, device=commands.device, dtype=torch.float32)
    weights[vx >= 0.30 - 1.0e-6] = 1.0
    low_forward = (vx > 1.0e-6) & (vx < 0.30 - 1.0e-6)
    weights[low_forward] = 0.25
    return weights


@dataclass
class CommandPlan:
    """Pending changes returned before the worker bridge publishes commands."""

    current: torch.Tensor
    pending_ids: torch.Tensor
    pending_commands: torch.Tensor
    pending_buckets: torch.Tensor
    requested_target_ids: torch.Tensor
    expired_ids: torch.Tensor
    target_probability: float
    anchor_weights: torch.Tensor


class CommandSchedule:
    """Per-environment source/target command schedule.

    The scheduler samples both source and target profiles. A planned command is
    committed only after the worker bridge writes the public command tensor and
    verifies readback. Each new training task creates a fresh scheduler.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device | str,
        config: dict[str, Any] | None = None,
        logger=None,
    ) -> None:
        config = config or {}
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.logger = logger
        self.step_dt_s = float(config.get("step_dt_s", 0.02))
        self.source_hold_s = tuple(
            float(v) for v in config.get("source_hold_s", [6.0, 10.0])
        )
        self.target_hold_s = tuple(
            float(v) for v in config.get("target_hold_s", [3.0, 6.0])
        )
        self.zero_hold_s = tuple(
            float(v) for v in config.get("zero_hold_s", [1.5, 3.0])
        )
        self.source_ranges = {
            "vx": tuple(float(v) for v in config.get("source_vx", [0.3, 1.3])),
            "vy": tuple(float(v) for v in config.get("source_vy", [-0.2, 0.2])),
            "wz": tuple(float(v) for v in config.get("source_wz", [-0.3, 0.3])),
        }
        self.limits = {
            "vx": (0.0, 1.3),
            "vy": (-0.2, 0.2),
            "wz": (-0.3, 0.3),
        }
        self.target_weights = torch.tensor(
            config.get("target_bucket_weights", [0.15, 0.20, 0.30, 0.15, 0.10, 0.10]),
            dtype=torch.float32,
            device=self.device,
        )
        if self.target_weights.numel() != len(BUCKET_NAMES):
            raise ValueError("target_bucket_weights must contain six bucket weights")
        weight_sum = float(self.target_weights.sum().item())
        if weight_sum <= 0.0:
            raise ValueError("target_bucket_weights must have a positive sum")
        self.target_weights /= weight_sum
        self._validate_config()

        self.elapsed_hours = 0.0
        self.command = None
        self.hold_remaining = torch.zeros(self.num_envs, device=self.device)
        self.is_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.bucket = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.anchor_weights = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._pending_target = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._pending_commands = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self._pending_buckets = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._pending_fallback = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.requested_samples = 0
        self.requested_target_samples = 0
        self.effective_source_samples = 0
        self.effective_target_samples = 0
        self.effective_bucket_counts = torch.zeros(
            len(BUCKET_NAMES), dtype=torch.long, device=self.device
        )
        self.command_count = 0
        self.command_sum = torch.zeros(3, device=self.device)
        self.command_min = torch.full((3,), float("inf"), device=self.device)
        self.command_max = torch.full((3,), float("-inf"), device=self.device)
        self.out_of_range_count = 0
        self._warning_no_setter = False

    def _validate_config(self) -> None:
        for name, values in (
            ("source_hold_s", self.source_hold_s),
            ("target_hold_s", self.target_hold_s),
            ("zero_hold_s", self.zero_hold_s),
        ):
            if len(values) != 2 or values[0] <= 0.0 or values[1] < values[0]:
                raise ValueError(f"{name} must be a positive [min,max] pair")
        for name, values in self.source_ranges.items():
            if len(values) != 2 or values[1] < values[0]:
                raise ValueError(f"source_{name} must be an ordered [min,max] pair")
            lower, upper = self.limits[name]
            if values[0] < lower or values[1] > upper:
                raise ValueError(
                    f"source_{name}={values} exceeds global limit {(lower, upper)}"
                )
        if self.step_dt_s <= 0.0:
            raise ValueError("step_dt_s must be positive")

    @property
    def target_probability(self) -> float:
        """Requested target probability at the current monotonic session clock."""
        minutes = self.elapsed_hours * 60.0
        if minutes < 30.0:
            return 0.0
        if minutes < 120.0:
            return (minutes - 30.0) / 90.0 * 0.5
        if minutes < 180.0:
            return 0.5 + (minutes - 120.0) / 60.0 * 0.5
        return 1.0

    def set_elapsed_hours(self, elapsed_hours: float) -> None:
        self.elapsed_hours = max(0.0, float(elapsed_hours))

    def _uniform(self, low: float, high: float, count: int) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, device=self.device)
        return low + (high - low) * torch.rand(count, device=self.device)

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
        zero = buckets == 0
        if bool(zero.any()):
            values[zero] = self._uniform(*self.zero_hold_s, int(zero.sum().item()))
        return values

    def _sample_target_commands(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        if count <= 0:
            return (
                torch.empty(0, 3, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )
        draws = torch.rand(count, device=self.device)
        buckets = torch.searchsorted(self.target_weights.cumsum(0), draws).long()
        commands = torch.zeros(count, 3, device=self.device)

        def fill(mask: torch.Tensor, vx, vy, wz) -> None:
            n = int(mask.sum().item())
            if n <= 0:
                return
            commands[mask, 0] = self._uniform(*vx, n) if vx is not None else 0.0
            commands[mask, 1] = self._uniform(*vy, n) if vy is not None else 0.0
            commands[mask, 2] = self._uniform(*wz, n) if wz is not None else 0.0

        fill(buckets == 1, (0.10, 0.30), None, None)
        fill(buckets == 2, (0.30, 0.80), None, None)
        forward_yaw = buckets == 3
        fill(forward_yaw, (0.15, 0.55), None, None)
        if bool(forward_yaw.any()):
            signs = torch.where(
                torch.rand(int(forward_yaw.sum().item()), device=self.device) < 0.5,
                -1.0,
                1.0,
            )
            commands[forward_yaw, 2] = signs * self._uniform(0.15, 0.30, int(forward_yaw.sum().item()))
        pure_yaw = buckets == 4
        if bool(pure_yaw.any()):
            signs = torch.where(
                torch.rand(int(pure_yaw.sum().item()), device=self.device) < 0.5,
                -1.0,
                1.0,
            )
            commands[pure_yaw, 2] = signs * self._uniform(0.15, 0.30, int(pure_yaw.sum().item()))
        lateral = buckets == 5
        if bool(lateral.any()):
            signs = torch.where(
                torch.rand(int(lateral.sum().item()), device=self.device) < 0.5,
                -1.0,
                1.0,
            )
            commands[lateral, 1] = signs * self._uniform(0.10, 0.20, int(lateral.sum().item()))
        return commands, buckets

    def _sample_source_commands(self, count: int) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, 3, device=self.device)
        commands = torch.empty(count, 3, device=self.device)
        commands[:, 0] = self._uniform(*self.source_ranges["vx"], count)
        commands[:, 1] = self._uniform(*self.source_ranges["vy"], count)
        commands[:, 2] = self._uniform(*self.source_ranges["wz"], count)
        return commands

    def plan(
        self,
        current_commands: torch.Tensor,
        *,
        dt_s: float | None = None,
        reset_mask=None,
        elapsed_hours: float | None = None,
    ) -> CommandPlan:
        current = current_commands.to(self.device, dtype=torch.float32)
        if current.ndim != 2 or current.shape[0] != self.num_envs or current.shape[1] < 3:
            raise ValueError(f"expected current command [num_envs,3+], got {tuple(current.shape)}")
        current = current[:, :3].clone()
        if self.command is None:
            self.command = current.clone()
            self.hold_remaining.zero_()
        if reset_mask is not None:
            reset = reset_mask.to(self.device).reshape(-1).bool()
            if reset.numel() != self.num_envs:
                raise ValueError("reset_mask size does not match num_envs")
            if bool(reset.any()):
                self.hold_remaining[reset] = 0.0
                self.is_target[reset] = False
                self.bucket[reset] = -1
                self.anchor_weights[reset] = 1.0
                self.command[reset] = current[reset]

        dt = self.step_dt_s if dt_s is None else max(0.0, float(dt_s))
        if elapsed_hours is not None:
            self.set_elapsed_hours(elapsed_hours)
        self.hold_remaining -= dt
        expired = self.hold_remaining <= 0.0
        expired_ids = torch.nonzero(expired, as_tuple=False).flatten()
        self._pending_target.zero_()
        self._pending_buckets.fill_(-1)
        self._pending_commands.zero_()
        self._pending_fallback.copy_(current)
        if expired_ids.numel() == 0:
            return CommandPlan(
                current=self.command.clone(),
                pending_ids=expired_ids,
                pending_commands=self._pending_commands.clone(),
                pending_buckets=self._pending_buckets.clone(),
                requested_target_ids=expired_ids,
                expired_ids=expired_ids,
                target_probability=self.target_probability,
                anchor_weights=self.anchor_weights.clone(),
            )

        probability = self.target_probability
        if probability >= 1.0:
            requested_target = torch.ones(
                expired_ids.numel(), dtype=torch.bool, device=self.device
            )
        elif probability <= 0.0:
            requested_target = torch.zeros(
                expired_ids.numel(), dtype=torch.bool, device=self.device
            )
        else:
            draws = torch.rand(expired_ids.numel(), device=self.device)
            requested_target = draws < probability
        requested_ids = expired_ids[requested_target]
        self.requested_samples += int(expired_ids.numel())
        self.requested_target_samples += int(requested_ids.numel())

        source_ids = expired_ids[~requested_target]
        if source_ids.numel() > 0:
            self._pending_commands[source_ids] = self._sample_source_commands(
                int(source_ids.numel())
            )

        if requested_ids.numel() > 0:
            target_commands, target_buckets = self._sample_target_commands(int(requested_ids.numel()))
            self._pending_target[requested_ids] = True
            self._pending_commands[requested_ids] = target_commands
            self._pending_buckets[requested_ids] = target_buckets

        return CommandPlan(
            current=self.command.clone(),
            pending_ids=expired_ids,
            pending_commands=self._pending_commands[expired_ids].clone(),
            pending_buckets=self._pending_buckets[expired_ids].clone(),
            requested_target_ids=requested_ids,
            expired_ids=expired_ids,
            target_probability=probability,
            anchor_weights=self.anchor_weights.clone(),
        )

    def commit(self, env_ids: torch.Tensor, *, applied: bool) -> None:
        env_ids = env_ids.to(self.device).reshape(-1).long()
        if env_ids.numel() == 0:
            return
        if applied:
            commands = self._pending_commands[env_ids]
            buckets = self._pending_buckets[env_ids]
            self.command[env_ids] = commands
            target_mask = self._pending_target[env_ids]
            self.is_target[env_ids] = target_mask
            self.bucket[env_ids] = buckets
            source_mask = ~target_mask
            if bool(source_mask.any()):
                source_ids = env_ids[source_mask]
                self.hold_remaining[source_ids] = self._sample_hold(
                    target=False, count=int(source_ids.numel())
                )
                self.anchor_weights[source_ids] = 1.0
                self.effective_source_samples += int(source_ids.numel())
            if bool(target_mask.any()):
                target_ids = env_ids[target_mask]
                target_buckets = buckets[target_mask]
                target_commands = commands[target_mask]
                self.hold_remaining[target_ids] = self._sample_hold(
                    target=True, buckets=target_buckets
                )
                self.anchor_weights[target_ids] = self._anchor_weight_for_targets(
                    target_buckets, target_commands
                )
                self.effective_target_samples += int(target_ids.numel())
                self.effective_bucket_counts.scatter_add_(
                    0, target_buckets,
                    torch.ones_like(target_buckets, dtype=torch.long),
                )
            self._record_commands(commands)
        else:
            self.is_target[env_ids] = False
            self.bucket[env_ids] = -1
            self.anchor_weights[env_ids] = 1.0
            self.hold_remaining[env_ids] = self._sample_hold(
                target=False, count=int(env_ids.numel())
            )
            self.effective_source_samples += int(env_ids.numel())
            self.command[env_ids] = self._pending_fallback[env_ids]
            self._record_commands(self.command[env_ids])

        self._pending_target[env_ids] = False
        self._pending_buckets[env_ids] = -1
        self._pending_commands[env_ids] = 0.0
        self._pending_fallback[env_ids] = 0.0

    @staticmethod
    def _anchor_weight_for_targets(
        buckets: torch.Tensor, commands: torch.Tensor
    ) -> torch.Tensor:
        """S0 anchor weights for target buckets from the execution plan."""
        weights = torch.zeros_like(buckets, dtype=torch.float32)
        weights[buckets == 1] = 0.25  # low forward
        weights[buckets == 2] = 1.0  # normal forward
        forward_yaw = buckets == 3
        weights[forward_yaw & (commands[:, 0] < 0.30)] = 0.25
        weights[forward_yaw & (commands[:, 0] >= 0.30)] = 1.0
        # zero, pure yaw and lateral have no reliable S0 supervision.
        return weights

    def _record_commands(self, commands: torch.Tensor) -> None:
        if commands.numel() == 0:
            return
        commands = commands.detach().to(self.device, dtype=torch.float32)
        self.command_count += int(commands.shape[0])
        self.command_sum += commands.sum(0)
        self.command_min = torch.minimum(self.command_min, commands.min(0).values)
        self.command_max = torch.maximum(self.command_max, commands.max(0).values)
        lower = torch.tensor(
            [self.limits["vx"][0], self.limits["vy"][0], self.limits["wz"][0]],
            device=self.device,
        )
        upper = torch.tensor(
            [self.limits["vx"][1], self.limits["vy"][1], self.limits["wz"][1]],
            device=self.device,
        )
        self.out_of_range_count += int(((commands < lower) | (commands > upper)).any(1).sum().item())

    def metrics(self) -> dict[str, Any]:
        total_effective = self.effective_source_samples + self.effective_target_samples
        mean = self.command_sum / max(1, self.command_count)
        command_min = (
            self.command_min.detach().cpu().tolist()
            if self.command_count
            else [None, None, None]
        )
        command_max = (
            self.command_max.detach().cpu().tolist()
            if self.command_count
            else [None, None, None]
        )
        return {
            "command_session_elapsed_hours": self.elapsed_hours,
            "command_target_probability": self.target_probability,
            "requested_source_probability": 1.0 - self.target_probability,
            "requested_target_probability": self.target_probability,
            "effective_source_probability": self.effective_source_samples / max(1, total_effective),
            "effective_target_probability": self.effective_target_samples / max(1, total_effective),
            "requested_command_samples": self.requested_samples,
            "requested_target_samples": self.requested_target_samples,
            "effective_target_samples": self.effective_target_samples,
            "effective_source_samples": self.effective_source_samples,
            "target_bucket_counts": {
                name: int(self.effective_bucket_counts[index].item())
                for index, name in enumerate(BUCKET_NAMES)
            },
            "command_min": command_min,
            "command_max": command_max,
            "command_mean": mean.detach().cpu().tolist(),
            "command_out_of_range_count": self.out_of_range_count,
            "anchor_weight_mean": float(self.anchor_weights.mean().item()),
        }
