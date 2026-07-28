#!/usr/bin/env python3
"""Stateful short-horizon SportMode/IMU feedback emulator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import torch

from agent_ppo.feature.p15_contract import FEEDBACK_PROFILE


def feedback_implementation_digest() -> str:
    """Hash the exact emulator implementation shipped with a checkpoint."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass
class FeedbackSample:
    measured_velocity: torch.Tensor
    velocity_valid: torch.Tensor
    velocity_age: torch.Tensor
    feedback_source: torch.Tensor
    ang_vel: torch.Tensor
    projected_gravity: torch.Tensor


class FeedbackEmulator:
    """Emulate SportMode planar velocity and IMU yaw-rate feedback.

    UWB is intentionally excluded from the 0.2/0.6/1.0-second response input.
    It remains a contract-level long-horizon source for a future stuck detector.
    """

    def __init__(self, num_envs: int, device, *, seed: int = 0, profile=None):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.profile = dict(profile or FEEDBACK_PROFILE)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed) + 941)
        self.measured = torch.zeros(self.num_envs, 3, device=self.device)
        self.age_s = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.source = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.sport_countdown = torch.zeros(self.num_envs, device=self.device)
        self.failure_remaining = torch.zeros(self.num_envs, device=self.device)
        max_delay_s = float(max(self.profile["sport"]["delay_s"]))
        self.history_length = max(2, int(math.ceil(max_delay_s / 0.02)) + 2)
        self.velocity_history = torch.zeros(
            self.history_length, self.num_envs, 3, device=self.device
        )
        self.history_index = -1
        self.sport_bias = self._normal((self.num_envs, 3)) * torch.tensor(
            self.profile["sport"]["bias_std"], device=self.device
        )
        self.gyro_bias = self._normal((self.num_envs, 3)) * torch.tensor(
            self.profile["imu"]["gyro_bias_std"], device=self.device
        )
        self.sport_timeout = self._uniform(
            *self.profile["sport"]["freshness_timeout_s"], (self.num_envs,)
        )

    def _rand(self, shape) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator, device=self.device)

    def _normal(self, shape) -> torch.Tensor:
        return torch.randn(shape, generator=self.generator, device=self.device)

    def _uniform(self, low: float, high: float, shape) -> torch.Tensor:
        return low + (high - low) * self._rand(shape)

    def reset(self, reset_mask: torch.Tensor) -> None:
        mask = reset_mask.to(self.device).reshape(-1).bool()
        if not bool(mask.any()):
            return
        self.measured[mask] = 0.0
        self.age_s[mask] = float("inf")
        self.valid[mask] = False
        self.source[mask] = 0
        self.sport_countdown[mask] = 0.0
        self.failure_remaining[mask] = 0.0
        self.velocity_history[:, mask] = 0.0
        self.sport_bias[mask] = self._normal((int(mask.sum()), 3)) * torch.tensor(
            self.profile["sport"]["bias_std"], device=self.device
        )
        self.gyro_bias[mask] = self._normal((int(mask.sum()), 3)) * torch.tensor(
            self.profile["imu"]["gyro_bias_std"], device=self.device
        )
        self.sport_timeout[mask] = self._uniform(
            *self.profile["sport"]["freshness_timeout_s"], (int(mask.sum()),)
        )

    def _delayed_velocity(
        self, env_mask: torch.Tensor, delay_range, dt_s: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = int(env_mask.sum().item())
        delay = self._uniform(float(delay_range[0]), float(delay_range[1]), (count,))
        delay_steps = torch.clamp(
            torch.round(delay / max(1.0e-6, dt_s)).long(),
            0,
            self.history_length - 1,
        )
        env_ids = torch.nonzero(env_mask, as_tuple=False).flatten()
        history_ids = (self.history_index - delay_steps) % self.history_length
        return self.velocity_history[history_ids, env_ids], delay

    def step(
        self,
        true_velocity: torch.Tensor,
        true_ang_vel: torch.Tensor,
        true_projected_gravity: torch.Tensor,
        *,
        dt_s: float,
        reset_mask: torch.Tensor | None = None,
    ) -> FeedbackSample:
        dt = max(0.0, float(dt_s))
        if reset_mask is not None:
            self.reset(reset_mask)
        true_velocity = true_velocity.to(self.device, dtype=torch.float32)
        true_ang_vel = true_ang_vel.to(self.device, dtype=torch.float32)
        true_projected_gravity = true_projected_gravity.to(
            self.device, dtype=torch.float32
        )
        self.history_index = (self.history_index + 1) % self.history_length
        self.velocity_history[self.history_index].copy_(true_velocity)
        self.age_s += dt
        self.sport_countdown -= dt
        self.failure_remaining = torch.clamp(self.failure_remaining - dt, min=0.0)

        start_failure = self._rand((self.num_envs,)) < (
            float(self.profile["complete_failure_probability"]) * dt
        )
        if bool(start_failure.any()):
            duration = self.profile["complete_failure_duration_s"]
            self.failure_remaining[start_failure] = self._uniform(
                float(duration[0]), float(duration[1]), (int(start_failure.sum()),)
            )

        sport_due = self.sport_countdown <= 0.0
        sport_dropout = self._rand((self.num_envs,)) < self._uniform(
            *self.profile["sport"]["dropout_probability"], (self.num_envs,)
        )
        sport_candidate = sport_due & ~sport_dropout & (self.failure_remaining <= 0.0)
        if bool(sport_due.any()):
            rate = self._uniform(*self.profile["sport"]["rate_hz"], (int(sport_due.sum()),))
            self.sport_countdown[sport_due] = 1.0 / rate.clamp_min(1.0)
        sport_valid = sport_candidate

        if bool(sport_valid.any()):
            delayed, delay = self._delayed_velocity(
                sport_valid, self.profile["sport"]["delay_s"], max(dt, 0.02)
            )
            noise = self._normal((int(sport_valid.sum()), 2)) * torch.tensor(
                self.profile["sport"]["noise_std"][:2], device=self.device
            )
            self.measured[sport_valid, :2] = (
                delayed[:, :2] + self.sport_bias[sport_valid, :2] + noise
            )
            self.age_s[sport_valid] = delay
            self.source[sport_valid] = 1

        self.valid = (self.source == 1) & (self.age_s <= self.sport_timeout) & (
            self.failure_remaining <= 0.0
        )
        age_clip = float(self.profile["age_clip_s"])
        normalized_age = torch.clamp(self.age_s, 0.0, age_clip) / age_clip
        normalized_age = torch.where(
            self.valid, normalized_age, torch.ones_like(normalized_age)
        )
        gyro_noise = self._normal(true_ang_vel.shape) * torch.tensor(
            self.profile["imu"]["gyro_noise_std"], device=self.device
        )
        gravity_noise = self._normal(true_projected_gravity.shape) * float(
            self.profile["imu"]["gravity_noise_std"]
        )
        noisy_ang_vel = true_ang_vel + self.gyro_bias + gyro_noise
        measured = torch.zeros_like(self.measured)
        measured[:, :2] = torch.where(
            self.valid.unsqueeze(-1), self.measured[:, :2], measured[:, :2]
        )
        measured[:, 2] = noisy_ang_vel[:, 2]
        return FeedbackSample(
            measured_velocity=measured,
            velocity_valid=self.valid.to(torch.float32).unsqueeze(-1),
            velocity_age=normalized_age.unsqueeze(-1),
            feedback_source=torch.where(
                self.valid, self.source, torch.zeros_like(self.source)
            ).to(torch.float32).unsqueeze(-1),
            ang_vel=noisy_ang_vel,
            projected_gravity=true_projected_gravity + gravity_noise,
        )
