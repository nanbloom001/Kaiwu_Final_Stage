# -*- coding: UTF-8 -*-
"""Metric-space UWB goal noise for the Track policy observation only."""

import math

import torch


class GoalNoiseAugmenter:
    """Apply per-frame jitter and per-episode bias independently per env."""

    _UNSUPPORTED_FAULT_KEYS = (
        "jump_enabled",
        "stale_enabled",
        "dropout_enabled",
    )

    def __init__(self, env, config=None):
        self.env = env
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))

        unsupported = [
            key for key in self._UNSUPPORTED_FAULT_KEYS
            if bool(self.config.get(key, False))
        ]
        if unsupported:
            raise ValueError(
                "ST7-Opt3 does not implement strong UWB faults: "
                + ", ".join(unsupported)
            )

        self.apply_probability = self._bounded(
            "apply_probability",
            default=0.75,
            lower=0.0,
            upper=1.0,
        )
        self.bearing_noise_std_rad = self._non_negative(
            "bearing_noise_std_rad",
            0.015,
        )
        self.distance_noise_std_m = self._non_negative(
            "distance_noise_std_m",
            0.03,
        )
        self.bearing_bias_std_rad = self._non_negative(
            "bearing_bias_std_rad",
            0.020,
        )
        self.bearing_bias_clip_rad = self._non_negative(
            "bearing_bias_clip_rad",
            0.060,
        )
        self.distance_bias_std_m = self._non_negative(
            "distance_bias_std_m",
            0.050,
        )
        self.distance_bias_clip_m = self._non_negative(
            "distance_bias_clip_m",
            0.150,
        )
        self.min_distance_m = self._non_negative("min_distance_m", 0.05)
        self.max_distance_m = self._non_negative("max_distance_m", 25.0)
        if self.max_distance_m < self.min_distance_m:
            raise ValueError(
                "goal_noise.max_distance_m must be >= min_distance_m, "
                f"got {self.max_distance_m} < {self.min_distance_m}."
            )

        self.num_envs = int(env.num_envs)
        self.device = env.device
        self.bearing_bias = torch.zeros(self.num_envs, device=self.device)
        self.distance_bias = torch.zeros(self.num_envs, device=self.device)
        self.noise_active = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.last_episode_length = None
        self._apply_count = 0

    def _non_negative(self, key, default):
        value = float(self.config.get(key, default))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"goal_noise.{key} must be finite and non-negative, got {value}."
            )
        return value

    def _bounded(self, key, default, lower, upper):
        value = float(self.config.get(key, default))
        if not math.isfinite(value) or value < lower or value > upper:
            raise ValueError(
                f"goal_noise.{key} must be finite and in "
                f"[{lower}, {upper}], got {value}."
            )
        return value

    def reset(self, env_ids: torch.Tensor):
        """Resample activation and persistent biases for selected envs only."""
        env_ids = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.device,
        ).reshape(-1)
        if env_ids.numel() == 0:
            return

        active = torch.rand(env_ids.numel(), device=self.device) < self.apply_probability
        self.noise_active[env_ids] = active

        bearing_bias = torch.randn(env_ids.numel(), device=self.device)
        bearing_bias = torch.clamp(
            bearing_bias * self.bearing_bias_std_rad,
            -self.bearing_bias_clip_rad,
            self.bearing_bias_clip_rad,
        )
        distance_bias = torch.randn(env_ids.numel(), device=self.device)
        distance_bias = torch.clamp(
            distance_bias * self.distance_bias_std_m,
            -self.distance_bias_clip_m,
            self.distance_bias_clip_m,
        )
        self.bearing_bias[env_ids] = bearing_bias * active.float()
        self.distance_bias[env_ids] = distance_bias * active.float()

    def _reset_new_episodes(self):
        current_length = getattr(self.env, "episode_length_buf", None)
        if current_length is None:
            if self.last_episode_length is None:
                self.reset(torch.arange(self.num_envs, device=self.device))
                self.last_episode_length = torch.zeros(
                    self.num_envs,
                    dtype=torch.long,
                    device=self.device,
                )
            return

        current_length = current_length.to(self.device).reshape(-1)
        if current_length.numel() != self.num_envs:
            raise ValueError(
                "episode_length_buf size mismatch: "
                f"expected {self.num_envs}, got {current_length.numel()}."
            )

        if self.last_episode_length is None:
            reset_mask = torch.ones_like(current_length, dtype=torch.bool)
        else:
            reset_mask = current_length < self.last_episode_length

        reset_ids = torch.nonzero(reset_mask, as_tuple=False).reshape(-1)
        if reset_ids.numel() > 0:
            self.reset(reset_ids)
        self.last_episode_length = current_length.clone()

    @staticmethod
    def _active_abs_mean(value, active_mask):
        if not active_mask.any():
            return 0.0
        return float(torch.mean(torch.abs(value[active_mask])).item())

    def _log_diagnostics(
        self,
        bearing_noise,
        distance_noise,
        active_mask,
    ):
        if self._apply_count != 1 and self._apply_count % 500 != 0:
            return

        active_ratio = float(active_mask.float().mean().item())
        bearing_noise_mean = self._active_abs_mean(bearing_noise, active_mask)
        distance_noise_mean = self._active_abs_mean(distance_noise, active_mask)
        bearing_bias_mean = self._active_abs_mean(self.bearing_bias, active_mask)
        distance_bias_mean = self._active_abs_mean(self.distance_bias, active_mask)
        print(
            "[GoalNoise] "
            f"step={self._apply_count}, "
            f"active_ratio={active_ratio:.4f}, "
            f"bearing_noise_abs_mean_rad={bearing_noise_mean:.6f}, "
            f"distance_noise_abs_mean_m={distance_noise_mean:.6f}, "
            f"bearing_bias_abs_mean_rad={bearing_bias_mean:.6f}, "
            f"distance_bias_abs_mean_m={distance_bias_mean:.6f}",
            flush=True,
        )

    def apply(self, raw_goal_xy: torch.Tensor) -> torch.Tensor:
        """Return a geometrically consistent noisy goal in robot-frame meters."""
        if not self.enabled:
            return raw_goal_xy
        if raw_goal_xy.ndim != 2 or raw_goal_xy.shape != (self.num_envs, 2):
            raise ValueError(
                "Goal noise input must have shape "
                f"[{self.num_envs}, 2], got {tuple(raw_goal_xy.shape)}."
            )

        self._reset_new_episodes()
        self._apply_count += 1

        local_x = raw_goal_xy[:, 0]
        local_y = raw_goal_xy[:, 1]
        distance = torch.sqrt(torch.square(local_x) + torch.square(local_y) + 1e-8)
        bearing = torch.atan2(local_y, local_x)

        active_mask = self.noise_active & (distance > 1e-4)
        bearing_noise = (
            torch.randn(self.num_envs, device=self.device)
            * self.bearing_noise_std_rad
            * active_mask.float()
        )
        distance_noise = (
            torch.randn(self.num_envs, device=self.device)
            * self.distance_noise_std_m
            * active_mask.float()
        )

        noisy_bearing = bearing + self.bearing_bias + bearing_noise
        noisy_distance = torch.clamp(
            distance + self.distance_bias + distance_noise,
            min=self.min_distance_m,
            max=self.max_distance_m,
        )
        noisy_x = noisy_distance * torch.cos(noisy_bearing)
        noisy_y = noisy_distance * torch.sin(noisy_bearing)
        noisy_goal_xy = torch.stack((noisy_x, noisy_y), dim=-1)
        noisy_goal_xy = torch.where(
            active_mask.unsqueeze(-1),
            noisy_goal_xy,
            raw_goal_xy,
        )
        noisy_goal_xy = torch.nan_to_num(
            noisy_goal_xy,
            nan=0.0,
            posinf=self.max_distance_m,
            neginf=-self.max_distance_m,
        )

        self._log_diagnostics(
            bearing_noise,
            distance_noise,
            active_mask,
        )
        return noisy_goal_xy
