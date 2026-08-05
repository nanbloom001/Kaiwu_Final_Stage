#!/usr/bin/env python3
"""Training-only P3 depth faults and recurrent memory supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from agent_ppo.feature import p3_contract


FAULT_NAMES = ("near_only", "sparse", "block", "severe", "blackout")


class P35CameraFeatureTiming:
    """Per-environment 30 Hz capture, delayed feature selection and 50 Hz hold."""

    CONTRACT = "p35_camera_feature_timing_v1"

    def __init__(
        self,
        *,
        num_envs: int,
        feature_dim: int,
        device,
        capacity: int = 10,
        seed: int = 3361,
        control_dt_s: float = 0.02,
        capture_rate_hz: float = 30.0,
    ):
        self.num_envs = int(num_envs)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.capacity = int(capacity)
        self.control_dt_s = float(control_dt_s)
        self.capture_period_s = 1.0 / float(capture_rate_hz)
        generator_device = self.device.type if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        self.features = torch.zeros(
            self.capacity,
            self.num_envs,
            self.feature_dim,
            dtype=torch.float16,
            device=self.device,
        )
        self.timestamps = torch.full(
            (self.capacity, self.num_envs), -1.0e9, device=self.device
        )
        self.write_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.current_time_s = torch.zeros(self.num_envs, device=self.device)
        self.next_capture_s = torch.rand(
            self.num_envs, device=self.device, generator=self.generator
        ) * self.capture_period_s
        self.delay_s = torch.zeros(self.num_envs, device=self.device)
        self.shadow_delay_s = torch.zeros(self.num_envs, device=self.device)
        self.last_age_s = torch.zeros(self.num_envs, device=self.device)
        self.last_timing_changed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.capture_count = torch.zeros((), device=self.device)
        self.hold_count = torch.zeros((), device=self.device)

    def begin_rollout(self, elapsed_s: float) -> None:
        nominal, short, long = p3_contract.camera_delay_probabilities(elapsed_s)
        selector = torch.rand(
            self.num_envs, device=self.device, generator=self.generator
        )
        self.delay_s.zero_()
        short_mask = (selector >= nominal) & (selector < nominal + short)
        long_mask = selector >= nominal + short
        if bool(short_mask.any()):
            self.delay_s[short_mask] = 0.040 + 0.060 * torch.rand(
                int(short_mask.sum()), device=self.device, generator=self.generator
            )
        if long > 0.0 and bool(long_mask.any()):
            self.delay_s[long_mask] = 0.100 + 0.050 * torch.rand(
                int(long_mask.sum()), device=self.device, generator=self.generator
            )
        self.shadow_delay_s.copy_(
            0.150
            + 0.100
            * torch.rand(
                self.num_envs, device=self.device, generator=self.generator
            )
        )
        self.capture_count.zero_()
        self.hold_count.zero_()

    def reset(self, reset_mask: torch.Tensor) -> None:
        ids = reset_mask.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        self.features[:, ids] = 0.0
        self.timestamps[:, ids] = -1.0e9
        self.write_index[ids] = 0
        self.current_time_s[ids] = 0.0
        self.next_capture_s[ids] = torch.rand(
            ids.numel(), device=self.device, generator=self.generator
        ) * self.capture_period_s
        self.last_age_s[ids] = 0.0
        self.last_timing_changed[ids] = False

    def capture_mask(self, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        if reset_mask is not None:
            self.reset(reset_mask)
        capture = self.current_time_s + 1.0e-9 >= self.next_capture_s
        # Make the first observation immediately usable even when its random
        # phase lies after t=0.
        empty = (self.timestamps > -1.0e8).sum(dim=0) == 0
        return capture | empty

    def step(
        self,
        current_features: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if current_features.shape != (self.num_envs, self.feature_dim):
            raise ValueError("P3.5 camera feature shape drift")
        capture = self.capture_mask(reset_mask)
        ids = capture.nonzero(as_tuple=False).flatten()
        if ids.numel():
            slots = self.write_index[ids]
            self.features[slots, ids] = current_features[ids].detach().to(torch.float16)
            self.timestamps[slots, ids] = self.current_time_s[ids]
            self.write_index[ids] = (slots + 1) % self.capacity
            due = ids[self.next_capture_s[ids] <= self.current_time_s[ids]]
            while due.numel():
                self.next_capture_s[due] += self.capture_period_s
                due = due[
                    self.next_capture_s[due] <= self.current_time_s[due]
                ]
        target_time = self.current_time_s - self.delay_s
        valid = self.timestamps <= target_time.unsqueeze(0)
        candidates = torch.where(
            valid, self.timestamps, torch.full_like(self.timestamps, -1.0e9)
        )
        slots = candidates.argmax(dim=0)
        env_ids = torch.arange(self.num_envs, device=self.device)
        selected = self.features[slots, env_ids].to(current_features.dtype)
        selected_time = self.timestamps[slots, env_ids]
        missing = selected_time < -1.0e8
        selected[missing] = current_features[missing]
        selected_time[missing] = self.current_time_s[missing]
        self.last_age_s.copy_((self.current_time_s - selected_time).clamp_min(0.0))
        self.last_timing_changed.copy_(self.last_age_s > 1.0e-6)
        self.capture_count += capture.float().sum()
        self.hold_count += (~capture).float().sum()
        self.current_time_s.add_(self.control_dt_s)
        return selected, capture

    def diagnostics(self) -> dict[str, float]:
        ages = self.last_age_s.detach().float()
        total = self.capture_count + self.hold_count
        return {
            "camera_capture_count": float(self.capture_count),
            "camera_hold_ratio": float(self.hold_count / total.clamp_min(1.0)),
            "camera_feature_age_mean_ms": float(ages.mean() * 1000.0),
            "camera_feature_age_p95_ms": float(torch.quantile(ages, 0.95) * 1000.0),
            "camera_active_delay_p95_ms": float(
                torch.quantile(self.delay_s, 0.95) * 1000.0
            ),
            "camera_shadow_delay_p95_ms": float(
                torch.quantile(self.shadow_delay_s, 0.95) * 1000.0
            ),
        }

    def state_dict(self) -> dict:
        return {
            "contract": self.CONTRACT,
            "generator_state": self.generator.get_state().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("contract") != self.CONTRACT:
            raise ValueError("P3.5 camera timing checkpoint contract mismatch")
        self.generator.set_state(state["generator_state"].cpu())


class P3DepthFaultAugmenter:
    """Apply rollout-persistent near-field and missing-depth faults."""

    def __init__(
        self,
        *,
        num_steps: int,
        num_envs: int,
        depth_shape: tuple[int, int, int],
        device,
        seed: int = 3329,
        max_depth_m: float = 5.0,
    ):
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.depth_shape = tuple(int(value) for value in depth_shape)
        self.device = torch.device(device)
        generator_device = self.device.type if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        self.max_depth_m = float(max_depth_m)
        self.strength = 0.0
        self.near_clip_m = torch.empty(self.num_envs, device=self.device)
        self.enabled = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prefix = torch.zeros_like(self.mode)
        self.duration = torch.zeros_like(self.mode)
        self.severe_ratio = torch.zeros(self.num_envs, device=self.device)
        self.blackout_ratio = torch.zeros(self.num_envs, device=self.device)
        self.spatial_rank = torch.empty(
            self.num_envs,
            *self.depth_shape,
            dtype=torch.float16,
            device=self.device,
        )
        self.block_rects = torch.zeros(
            self.num_envs, 3, 4, dtype=torch.long, device=self.device
        )
        self.block_count = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._metric_sums: dict[str, torch.Tensor] = {}
        self._metric_steps = 0
        self._environment_initialized = False
        self.reset_environment()

    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.generator)

    def _randint(self, low: int, high: int, shape):
        return torch.randint(
            low, high, shape, device=self.device, generator=self.generator
        )

    def reset_environment(self) -> None:
        u = self._rand(self.num_envs)
        beta_sample = 1.0 - torch.pow(1.0 - u, 0.25)
        self.near_clip_m.copy_(0.10 + 0.15 * beta_sample)
        self._environment_initialized = True

    def ensure_environment_initialized(self) -> None:
        if not self._environment_initialized:
            self.reset_environment()

    def begin_rollout(self, strength: float) -> None:
        self.strength = max(0.0, min(1.0, float(strength)))
        probabilities = torch.tensor(
            [0.50, 0.20, 0.15, 0.10, 0.05], device=self.device
        )
        self.mode.copy_(
            torch.multinomial(
                probabilities,
                self.num_envs,
                replacement=True,
                generator=self.generator,
            )
        )
        self.enabled.copy_(self._rand(self.num_envs) < self.strength)
        self.prefix.copy_(self._randint(30, 71, (self.num_envs,)))
        self.duration.copy_(self._randint(20, 91, (self.num_envs,)))
        block = self.mode == 2
        severe = self.mode == 3
        blackout = self.mode == 4
        self.duration[block] = self._randint(4, 21, (int(block.sum()),))
        self.duration[severe] = self._randint(20, 76, (int(severe.sum()),))
        self.duration[blackout] = self._randint(10, 41, (int(blackout.sum()),))
        self.severe_ratio.copy_(0.60 + 0.25 * self._rand(self.num_envs))
        self.blackout_ratio.copy_(0.95 + 0.05 * self._rand(self.num_envs))
        self.spatial_rank.copy_(self._rand(*self.spatial_rank.shape))
        # Severe failures preferentially remove the center/lower stair edge,
        # while the exact missing-pixel budget remains globally bounded.
        severe_ids = severe.nonzero().flatten()
        if severe_ids.numel():
            self.spatial_rank[severe_ids, 100:145, 112:208, :] *= 0.10
        self.block_count.copy_(self._randint(1, 4, (self.num_envs,)))
        height, width, _ = self.depth_shape
        for block_id in range(3):
            block_h = self._randint(18, 31, (self.num_envs,))
            block_w = self._randint(32, 65, (self.num_envs,))
            y0 = self._randint(0, max(1, height - 30), (self.num_envs,))
            x0 = self._randint(0, max(1, width - 64), (self.num_envs,))
            self.block_rects[:, block_id, 0] = y0
            self.block_rects[:, block_id, 1] = y0 + block_h
            self.block_rects[:, block_id, 2] = x0
            self.block_rects[:, block_id, 3] = x0 + block_w
        self._metric_sums = {}
        self._metric_steps = 0

    @staticmethod
    def _hole_ratio(depth: torch.Tensor, region=None) -> torch.Tensor:
        values = depth if region is None else depth[:, region[0], region[1], :]
        invalid = (~torch.isfinite(values)) | (values <= 0.0)
        return invalid.reshape(values.shape[0], -1).float().mean(dim=-1)

    def _accumulate(self, name: str, values: torch.Tensor) -> None:
        value = values.detach().float().mean()
        self._metric_sums[name] = self._metric_sums.get(
            name, torch.zeros((), device=value.device)
        ) + value

    @staticmethod
    def _drop_to_target(
        values: torch.Tensor,
        spatial_rank: torch.Tensor,
        target_hole_ratio: torch.Tensor,
    ) -> torch.Tensor:
        pixels_per_env = values[0].numel()
        flat_values = values.reshape(values.shape[0], -1)
        flat_rank = spatial_rank.reshape(spatial_rank.shape[0], -1)
        for local_id in range(values.shape[0]):
            valid = flat_values[local_id] > 0.0
            valid_indices = valid.nonzero().flatten()
            current_invalid = pixels_per_env - int(valid_indices.numel())
            target_invalid = min(
                pixels_per_env,
                int(float(target_hole_ratio[local_id]) * pixels_per_env),
            )
            drop_count = max(0, target_invalid - current_invalid)
            if drop_count <= 0:
                continue
            scores = flat_rank[local_id, valid_indices]
            keep_count = int(valid_indices.numel()) - drop_count
            if keep_count <= 0:
                flat_values[local_id, valid_indices] = 0.0
            elif drop_count <= keep_count:
                selected = torch.topk(
                    scores, drop_count, largest=False, sorted=False
                ).indices
                flat_values[local_id, valid_indices[selected]] = 0.0
            else:
                keep = torch.topk(
                    scores, keep_count, largest=True, sorted=False
                ).indices
                drop = torch.ones_like(valid_indices, dtype=torch.bool)
                drop[keep] = False
                flat_values[local_id, valid_indices[drop]] = 0.0
        return values

    def apply(self, depth: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(depth.shape[1:]) != self.depth_shape:
            raise ValueError(
                f"P3 depth fault expected {self.depth_shape}, got {tuple(depth.shape[1:])}"
            )
        clean = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        raw_full = self._hole_ratio(clean)
        raw_center = self._hole_ratio(clean, (slice(60, 145), slice(112, 208)))
        raw_lower = self._hole_ratio(clean, (slice(100, 180), slice(None)))
        augmented = clean.clone()
        if self.strength > 0.0:
            threshold = (self.near_clip_m / self.max_depth_m).reshape(-1, 1, 1, 1)
            near = (
                self.enabled.reshape(-1, 1, 1, 1)
                & (augmented > 0.0)
                & (augmented < threshold)
            )
            augmented[near] = 0.0
            active = (
                self.enabled
                & (step >= self.prefix)
                & (step < self.prefix + self.duration)
            )
            sparse_ids = (active & (self.mode == 1)).nonzero().flatten()
            if sparse_ids.numel():
                values = augmented[sparse_ids]
                values[self.spatial_rank[sparse_ids] < 0.03] = 0.0
                augmented[sparse_ids] = values
            block_ids = (active & (self.mode == 2)).nonzero().flatten()
            for env_id in block_ids.tolist():
                for block_id in range(int(self.block_count[env_id])):
                    y0, y1, x0, x1 = self.block_rects[env_id, block_id].tolist()
                    augmented[env_id, y0:y1, x0:x1, :] = 0.0
            severe_ids = (active & (self.mode == 3)).nonzero().flatten()
            if severe_ids.numel():
                values = augmented[severe_ids]
                augmented[severe_ids] = self._drop_to_target(
                    values,
                    self.spatial_rank[severe_ids],
                    self.severe_ratio[severe_ids],
                )
            blackout_ids = (active & (self.mode == 4)).nonzero().flatten()
            if blackout_ids.numel():
                values = augmented[blackout_ids]
                augmented[blackout_ids] = self._drop_to_target(
                    values,
                    self.spatial_rank[blackout_ids],
                    self.blackout_ratio[blackout_ids],
                )

        augmented = torch.nan_to_num(augmented, nan=0.0, posinf=0.0, neginf=0.0)
        changed = (augmented != clean).reshape(self.num_envs, -1).any(dim=-1)
        self._accumulate("depth_raw_hole_full", raw_full)
        self._accumulate("depth_raw_hole_center", raw_center)
        self._accumulate("depth_raw_hole_lower", raw_lower)
        self._accumulate("depth_aug_hole_full", self._hole_ratio(augmented))
        self._accumulate(
            "depth_aug_hole_center",
            self._hole_ratio(augmented, (slice(60, 145), slice(112, 208))),
        )
        self._accumulate(
            "depth_aug_hole_lower",
            self._hole_ratio(augmented, (slice(100, 180), slice(None))),
        )
        self._accumulate("depth_fault_active_share", changed.float())
        self._metric_steps += 1
        return augmented, changed

    def diagnostics(self) -> dict[str, float]:
        divisor = max(1, self._metric_steps)
        result = {
            name: float(value / divisor) for name, value in self._metric_sums.items()
        }
        clips = self.near_clip_m.detach().float()
        result.update(
            near_clip_mean_m=float(clips.mean()),
            near_clip_p50_m=float(torch.quantile(clips, 0.50)),
            near_clip_p90_m=float(torch.quantile(clips, 0.90)),
            near_clip_p99_m=float(torch.quantile(clips, 0.99)),
            depth_fault_strength=float(self.strength),
            depth_fault_enabled_share=float(self.enabled.float().mean()),
            depth_fault_planned_duration_s=float(
                self.duration.detach().float().mean() * 0.02
            ),
        )
        histogram = torch.histc(clips, bins=10, min=0.10, max=0.25)
        histogram /= histogram.sum().clamp_min(1.0)
        for index, value in enumerate(histogram):
            result[f"near_clip_bin_{index}_share"] = float(value)
        mode_histogram = torch.bincount(
            self.mode[self.enabled], minlength=len(FAULT_NAMES)
        ).float()
        mode_histogram /= mode_histogram.sum().clamp_min(1.0)
        for index, name in enumerate(FAULT_NAMES):
            result[f"depth_fault_{name}_share"] = float(mode_histogram[index])
        for mode_index, name in ((3, "severe"), (4, "blackout")):
            selected = self.enabled & (self.mode == mode_index)
            result[f"depth_fault_{name}_event_count"] = float(selected.sum())
            result[f"depth_fault_{name}_duration_s"] = (
                float(self.duration[selected].float().mean() * 0.02)
                if bool(selected.any())
                else 0.0
            )
        result["depth_fault_recovery_telemetry_available"] = 0.0
        return result

    def state_dict(self) -> dict:
        return {
            "contract": "p3_depth_fault_v1",
            "generator_state": self.generator.get_state().cpu(),
            "max_depth_m": self.max_depth_m,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("contract") != "p3_depth_fault_v1":
            raise ValueError("P3 depth fault checkpoint contract mismatch")
        generator_state = state.get("generator_state")
        if not torch.is_tensor(generator_state):
            raise ValueError("P3 depth fault checkpoint missing RNG state")
        self.generator.set_state(generator_state.cpu())
        # Live per-environment thresholds are intentionally not checkpointed.
        # The first workflow reset after process restore samples them once.
        self._environment_initialized = False


class P3MemoryAuxiliary:
    """Train the recurrent action path to match a clean immutable teacher."""

    def __init__(self, *, num_steps, num_envs, obs_dim, device, seed=3331):
        self.device = torch.device(device)
        self.clean = torch.zeros(num_steps, num_envs, obs_dim, device=self.device)
        self.selected = torch.zeros(
            num_steps, num_envs, dtype=torch.bool, device=self.device
        )
        self.fault_selected = torch.zeros_like(self.selected)
        self.timing_selected = torch.zeros_like(self.selected)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.scale = 0.0
        self.gradient_multiplier = None
        self.gradient_ratio = 0.0
        self.gradient_cosine = 0.0

    def begin_rollout(self, scale: float) -> None:
        self.clean.zero_()
        self.selected.zero_()
        self.fault_selected.zero_()
        self.timing_selected.zero_()
        self.scale = max(0.0, min(1.0, float(scale)))
        self.gradient_multiplier = None
        self.gradient_ratio = 0.0
        self.gradient_cosine = 0.0

    def store(
        self,
        step: int,
        clean_obs: torch.Tensor,
        changed: torch.Tensor,
        timing_changed: torch.Tensor | None = None,
    ) -> None:
        self.clean[step].copy_(clean_obs)
        fault = changed.bool()
        timing = (
            torch.zeros_like(fault)
            if timing_changed is None
            else timing_changed.bool()
        )
        self.fault_selected[step].copy_(fault)
        self.timing_selected[step].copy_(timing)
        self.selected[step].copy_(fault | timing)

    def _selection_metrics(self) -> dict[str, float]:
        fault = self.fault_selected
        timing = self.timing_selected
        return {
            "memory_fault_frame_share": float(fault.float().mean()),
            "memory_timing_frame_share": float(timing.float().mean()),
            "memory_fault_only_frame_share": float((fault & ~timing).float().mean()),
            "memory_timing_only_frame_share": float((timing & ~fault).float().mean()),
            "memory_fault_timing_overlap_share": float((fault & timing).float().mean()),
            "memory_selected_frame_share": float(self.selected.float().mean()),
        }

    @staticmethod
    def _actor_with_detached_body(actor, features):
        value = features
        modules = list(actor)
        for module in modules[:-1]:
            if isinstance(module, torch.nn.Linear):
                value = F.linear(
                    value,
                    module.weight.detach(),
                    None if module.bias is None else module.bias.detach(),
                )
            else:
                value = module(value)
        return modules[-1](value)

    @staticmethod
    def _encode_anchor(encoder, compact: torch.Tensor, masks: torch.Tensor):
        previous = encoder.get_hidden_state()
        previous = (
            None
            if previous is None
            else tuple(value.detach().clone() for value in previous)
        )
        encoder.reset_hidden_state(compact.shape[1], compact.device)
        values = []
        try:
            for step in range(compact.shape[0]):
                keep = None if step == 0 else masks[step - 1].reshape(-1)
                values.append(
                    encoder.forward_from_cnn_features(
                        compact[step, :, 45:],
                        compact[step, :, :45],
                        masks=keep,
                        detach_hidden=False,
                    )
                )
            return torch.stack(values)
        finally:
            if previous is None:
                encoder.reset_hidden_state()
            else:
                encoder.set_hidden_state(previous)

    def loss(self, model, anchor_encoder, anchor_actor, fault, dones):
        zero = fault.new_zeros(())
        candidates = self.selected.any(dim=0).nonzero().flatten()
        if self.scale <= 0.0 or candidates.numel() == 0:
            return zero, {
                "memory_loss": 0.0,
                "memory_action_mae": 0.0,
                "memory_hidden_advantage": 0.0,
                "memory_latent_cosine": 0.0,
                **self._selection_metrics(),
            }
        chosen = int(
            candidates[
                int(torch.randint(candidates.numel(), (1,), generator=self.generator))
            ]
        )
        fault_seq = fault[:, chosen : chosen + 1]
        clean_seq = self.clean[:, chosen : chosen + 1]
        continuation = (~dones[:, chosen : chosen + 1]).float()
        active = self.selected[:, chosen : chosen + 1].to(fault)
        with torch.no_grad():
            clean_latent = self._encode_anchor(
                anchor_encoder, clean_seq, continuation
            )
            target = anchor_actor(
                torch.cat((clean_seq[..., :45], clean_latent), dim=-1)
            )
        fault_latent = model._encode_sequence(
            fault_seq, hidden_states=None, masks=continuation
        )
        prediction = self._actor_with_detached_body(
            model.actor, torch.cat((fault_seq[..., :45], fault_latent), dim=-1)
        )
        weights = active.unsqueeze(-1)
        denominator = (weights.sum() * prediction.shape[-1]).clamp_min(1.0)
        terms = F.smooth_l1_loss(prediction, target, reduction="none")
        loss = (terms * weights).sum() / denominator * self.scale
        with torch.no_grad():
            reset_masks = torch.zeros_like(continuation)
            reset_latent = model._encode_sequence(
                fault_seq, hidden_states=None, masks=reset_masks
            )
            reset_prediction = model.actor(
                torch.cat((fault_seq[..., :45], reset_latent), dim=-1)
            )
            full_mae = ((prediction - target).abs() * weights).sum() / denominator
            reset_mae = ((reset_prediction - target).abs() * weights).sum() / denominator
            cosine = F.cosine_similarity(fault_latent, clean_latent, dim=-1)
            cosine = (cosine * active).sum() / active.sum().clamp_min(1.0)
        return loss, {
            "memory_loss": float(loss.detach()),
            "memory_action_mae": float(full_mae),
            "memory_hidden_advantage": float(reset_mae - full_mae),
            "memory_latent_cosine": float(cosine),
            **self._selection_metrics(),
        }

    def state_dict(self) -> dict:
        return {
            "contract": "p3_memory_aux_v1",
            "generator_state": self.generator.get_state(),
            "scale": float(self.scale),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("contract") != "p3_memory_aux_v1":
            raise ValueError("P3 memory auxiliary checkpoint contract mismatch")
        generator_state = state.get("generator_state")
        if not torch.is_tensor(generator_state):
            raise ValueError("P3 memory auxiliary checkpoint missing RNG state")
        self.generator.set_state(generator_state.cpu())
