# -*- coding: UTF-8 -*-
"""Stateful per-environment rectangular dropout for LBC depth images."""

import math

import torch


class DepthBlockDropoutAugmenter:
    """Apply independent rectangular holes that persist for several frames."""

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.frame_probability = self._bounded(
            "frame_probability", 0.0, 0.0, 1.0
        )
        self.min_blocks = self._positive_int("min_blocks", 1)
        self.max_blocks = self._positive_int("max_blocks", 1)
        self.min_width_px = self._positive_int("min_width_px", 1)
        self.max_width_px = self._positive_int("max_width_px", 1)
        self.min_height_px = self._positive_int("min_height_px", 1)
        self.max_height_px = self._positive_int("max_height_px", 1)
        self.max_total_area_ratio = self._bounded(
            "max_total_area_ratio", 0.15, 0.0, 1.0
        )
        self.min_persistence_frames = self._positive_int(
            "min_persistence_frames", 2
        )
        self.max_persistence_frames = self._positive_int(
            "max_persistence_frames", 4
        )
        self._validate_ranges()

        self.shape = None
        self.device = None
        self.top = None
        self.left = None
        self.height = None
        self.width = None
        self.ttl = None
        self.duration = None
        self.active = None
        self.last_episode_length = None

    def _positive_int(self, key, default):
        value = int(self.config.get(key, default))
        if value <= 0:
            raise ValueError(f"depth_block_dropout.{key} must be positive, got {value}.")
        return value

    def _bounded(self, key, default, lower, upper):
        value = float(self.config.get(key, default))
        if not math.isfinite(value) or value < lower or value > upper:
            raise ValueError(
                f"depth_block_dropout.{key} must be finite and in "
                f"[{lower}, {upper}], got {value}."
            )
        return value

    def _validate_ranges(self):
        pairs = (
            ("blocks", self.min_blocks, self.max_blocks),
            ("width_px", self.min_width_px, self.max_width_px),
            ("height_px", self.min_height_px, self.max_height_px),
            (
                "persistence_frames",
                self.min_persistence_frames,
                self.max_persistence_frames,
            ),
        )
        for name, lower, upper in pairs:
            if upper < lower:
                raise ValueError(
                    f"depth_block_dropout max_{name} must be >= min_{name}, "
                    f"got {upper} < {lower}."
                )

    def _ensure_state(self, depth):
        num_envs, image_height, image_width = depth.shape
        shape = (num_envs, image_height, image_width)
        if self.shape == shape and self.device == depth.device:
            return

        self.shape = shape
        self.device = depth.device
        state_shape = (num_envs, self.max_blocks)
        self.top = torch.zeros(state_shape, dtype=torch.long, device=self.device)
        self.left = torch.zeros(state_shape, dtype=torch.long, device=self.device)
        self.height = torch.zeros(state_shape, dtype=torch.long, device=self.device)
        self.width = torch.zeros(state_shape, dtype=torch.long, device=self.device)
        self.ttl = torch.zeros(state_shape, dtype=torch.long, device=self.device)
        self.duration = torch.zeros(state_shape, dtype=torch.long, device=self.device)
        self.active = torch.zeros(state_shape, dtype=torch.bool, device=self.device)
        self.last_episode_length = None

    def reset(self, env_ids):
        """Clear only the selected environments' persistent block state."""
        env_ids = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.device,
        ).reshape(-1)
        if env_ids.numel() == 0:
            return
        for value in (
            self.top,
            self.left,
            self.height,
            self.width,
            self.ttl,
            self.duration,
        ):
            value[env_ids] = 0
        self.active[env_ids] = False

    def _reset_new_episodes(self, env):
        current_length = getattr(env, "episode_length_buf", None)
        if current_length is None:
            if self.last_episode_length is None:
                self.reset(torch.arange(self.shape[0], device=self.device))
                self.last_episode_length = torch.zeros(
                    self.shape[0], dtype=torch.long, device=self.device
                )
            return

        current_length = current_length.to(self.device).reshape(-1)
        if current_length.numel() != self.shape[0]:
            raise ValueError(
                "episode_length_buf size mismatch for block dropout: "
                f"expected {self.shape[0]}, got {current_length.numel()}."
            )
        if self.last_episode_length is None:
            reset_mask = torch.ones_like(current_length, dtype=torch.bool)
        else:
            reset_mask = current_length < self.last_episode_length
        reset_ids = torch.nonzero(reset_mask, as_tuple=False).reshape(-1)
        if reset_ids.numel() > 0:
            self.reset(reset_ids)
        self.last_episode_length = current_length.clone()

    def _build_mask(self):
        num_envs, image_height, image_width = self.shape
        mask = torch.zeros(
            self.shape,
            dtype=torch.bool,
            device=self.device,
        )
        active_ids = torch.nonzero(self.active, as_tuple=False)
        for env_id, block_id in active_ids.tolist():
            top = int(self.top[env_id, block_id].item())
            left = int(self.left[env_id, block_id].item())
            height = int(self.height[env_id, block_id].item())
            width = int(self.width[env_id, block_id].item())
            bottom = min(image_height, top + height)
            right = min(image_width, left + width)
            mask[env_id, top:bottom, left:right] = True
        return mask

    def _randint(self, lower, upper):
        return int(
            torch.randint(
                lower,
                upper + 1,
                (1,),
                device=self.device,
            ).item()
        )

    def _spawn_blocks(self):
        num_envs, image_height, image_width = self.shape
        no_active_blocks = ~self.active.any(dim=1)
        spawn_envs = no_active_blocks & (
            torch.rand(num_envs, device=self.device) < self.frame_probability
        )
        max_pixels = int(self.max_total_area_ratio * image_height * image_width)

        for env_id in torch.nonzero(spawn_envs, as_tuple=False).reshape(-1).tolist():
            requested = self._randint(self.min_blocks, self.max_blocks)
            union_mask = torch.zeros(
                image_height,
                image_width,
                dtype=torch.bool,
                device=self.device,
            )
            for block_id in range(requested):
                width = min(self._randint(self.min_width_px, self.max_width_px), image_width)
                height = min(
                    self._randint(self.min_height_px, self.max_height_px),
                    image_height,
                )
                top = self._randint(0, max(0, image_height - height))
                left = self._randint(0, max(0, image_width - width))
                candidate = union_mask.clone()
                candidate[top : top + height, left : left + width] = True
                if int(candidate.sum().item()) > max_pixels:
                    continue

                duration = self._randint(
                    self.min_persistence_frames,
                    self.max_persistence_frames,
                )
                union_mask = candidate
                self.top[env_id, block_id] = top
                self.left[env_id, block_id] = left
                self.height[env_id, block_id] = height
                self.width[env_id, block_id] = width
                self.ttl[env_id, block_id] = duration
                self.duration[env_id, block_id] = duration
                self.active[env_id, block_id] = True

    def apply(self, depth, env):
        """Return depth with persistent blocks and current diagnostic metrics."""
        self._ensure_state(depth)
        if not self.enabled:
            return depth, {
                "depth_block_dropout_frame_ratio": 0.0,
                "depth_block_dropout_area_ratio": 0.0,
                "active_block_count": 0.0,
                "block_persistence_mean": 0.0,
            }

        self._reset_new_episodes(env)
        self.ttl = torch.clamp(self.ttl - 1, min=0)
        expired = self.active & (self.ttl == 0)
        self.active[expired] = False
        self.duration[expired] = 0
        self._spawn_blocks()

        mask = self._build_mask()
        per_env_area = mask.flatten(1).float().mean(dim=1)
        frame_active = mask.flatten(1).any(dim=1)
        if self.active.any():
            persistence_mean = float(self.duration[self.active].float().mean().item())
        else:
            persistence_mean = 0.0
        metrics = {
            "depth_block_dropout_frame_ratio": float(frame_active.float().mean().item()),
            "depth_block_dropout_area_ratio": float(per_env_area.mean().item()),
            "active_block_count": float(self.active.float().sum(dim=1).mean().item()),
            "block_persistence_mean": persistence_mean,
        }
        max_area_ratio = float(per_env_area.max().item())
        if max_area_ratio > self.max_total_area_ratio + 1e-7:
            raise RuntimeError(
                "Block dropout exceeded max_total_area_ratio: "
                f"{max_area_ratio:.6f} > "
                f"{self.max_total_area_ratio:.6f}."
            )
        return depth.masked_fill(mask, 0.0), metrics
