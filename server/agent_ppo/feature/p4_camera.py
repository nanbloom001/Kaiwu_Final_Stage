#!/usr/bin/env python3
"""Shared 30 Hz capture / 50 Hz delivery state for P4 low and high levels."""

from __future__ import annotations

import torch

from agent_ppo.feature import p2_contract, p4_contract


class P4SharedCameraState:
    """Apply faults once per capture and deliver one shared delayed frame.

    This class caches depth frames, never recurrent latents. The frozen low-level
    CNN may cache feature32 for an unchanged frame, while its LSTM still advances
    on every 50 Hz control tick with current proprioception.
    """

    RING_SIZE = 10

    def __init__(
        self,
        num_envs: int,
        device,
        *,
        seed: int = 0,
        max_depth_m: float = p2_contract.PREDICTIVE_COLLISION_MAX_DEPTH_M,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.max_depth_m = float(max_depth_m)
        if not (self.max_depth_m > 0.0):
            raise ValueError("P4 camera max_depth_m must be positive")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.capture_accumulator = self._uniform((self.num_envs,))
        self.near_clip = self._sample_near_clip(self.num_envs)
        self.frames: list[torch.Tensor | None] = [None] * self.RING_SIZE
        self.frame_times = torch.full(
            (self.RING_SIZE, self.num_envs), -1.0e6, device=self.device
        )
        self.frame_ids = torch.full(
            (self.RING_SIZE, self.num_envs), -1, dtype=torch.long, device=self.device
        )
        self.cursor = -1
        self.control_time_s = 0.0
        self.next_frame_id = 0
        self.delivered_frame_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.delivered_frame_time = torch.full(
            (self.num_envs,), -1.0e6, device=self.device
        )
        self.delivered = None
        self.clean_capture = None
        self.near_clip_added_hole_rate = torch.zeros(
            self.num_envs, device=self.device
        )
        self.fault_kind = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.sequence_kind = torch.zeros_like(self.fault_kind)
        self.fault_remaining_frames = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.delay_enabled = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.delay_s = torch.zeros(self.num_envs, device=self.device)
        self.pixel_fault_enabled = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.block = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device)
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def begin_rollout(
        self, session_effective_seconds: float, *, training: bool
    ) -> None:
        if not training:
            self.sequence_kind.zero_()
        else:
            mix = p4_contract.camera_mix(session_effective_seconds)
            draw = self._uniform((self.num_envs,))
            nominal = float(mix.get("nominal", 1.0))
            light = nominal + float(mix.get("light", 0.0))
            delayed = light + float(mix.get("delayed", 0.0))
            self.sequence_kind = torch.where(
                draw < nominal,
                torch.zeros_like(self.sequence_kind),
                torch.where(
                    draw < light,
                    torch.ones_like(self.sequence_kind),
                    torch.where(
                        draw < delayed,
                        torch.full_like(self.sequence_kind, 2),
                        torch.full_like(self.sequence_kind, 3),
                    ),
                ),
            )
        self.fault_kind.zero_()
        self.fault_remaining_frames.zero_()
        self.delay_enabled.zero_()
        self.pixel_fault_enabled.zero_()
        self.delay_s.zero_()

    def _uniform(self, shape) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator, device="cpu").to(self.device)

    def _randint(self, low: int, high: int, shape) -> torch.Tensor:
        return torch.randint(
            low, high, shape, generator=self.generator, device="cpu"
        ).to(self.device)

    def _sample_near_clip(self, count: int) -> torch.Tensor:
        u = self._uniform((count,))
        return 0.10 + 0.15 * (1.0 - torch.pow(1.0 - u, 0.25))

    def reset(self, mask: torch.Tensor) -> None:
        mask = mask.to(self.device).bool().reshape(-1)
        if not bool(mask.any()):
            return
        self.capture_accumulator[mask] = self._uniform((int(mask.sum()),))
        self.delivered_frame_id[mask] = -1
        self.delivered_frame_time[mask] = -1.0e6
        self.fault_kind[mask] = 0
        self.fault_remaining_frames[mask] = 0
        self.delay_enabled[mask] = False
        self.delay_s[mask] = 0.0
        self.pixel_fault_enabled[mask] = False
        for index in range(self.RING_SIZE):
            self.frame_times[index, mask] = -1.0e6
            self.frame_ids[index, mask] = -1
            if self.frames[index] is not None:
                self.frames[index][mask] = 0.0
        if self.delivered is not None:
            self.delivered[mask] = 0.0
        if self.clean_capture is not None:
            self.clean_capture[mask] = 0.0
        self.near_clip_added_hole_rate[mask] = 0.0

    def _new_faults(self, capture: torch.Tensor) -> None:
        due = capture & (self.fault_remaining_frames <= 0)
        if not bool(due.any()):
            return
        ids = due.nonzero(as_tuple=False).reshape(-1)
        kinds = self.sequence_kind[ids]
        self.fault_kind[ids] = kinds
        mode_draw = self._uniform((ids.numel(),))
        active = kinds > 0
        # Keep delay-only, pixel-fault-only and overlap as distinct sampled
        # conditions so their teacher masks have real support.
        self.delay_enabled[ids] = active & (mode_draw < 0.67)
        self.pixel_fault_enabled[ids] = active & (mode_draw >= 0.33)
        sampled_delay = torch.zeros(ids.numel(), device=self.device)
        light_delay = (kinds == 1) & self.delay_enabled[ids]
        delayed = (kinds >= 2) & self.delay_enabled[ids]
        if bool(light_delay.any()):
            sampled_delay[light_delay] = 0.04 + 0.06 * self._uniform(
                (int(light_delay.sum()),)
            )
        if bool(delayed.any()):
            sampled_delay[delayed] = 0.10 + 0.05 * self._uniform(
                (int(delayed.sum()),)
            )
        self.delay_s[ids] = sampled_delay
        duration = self._randint(4, 41, (ids.numel(),))
        severe = kinds == 3
        duration[severe] = self._randint(10, 76, (int(severe.sum()),))
        self.fault_remaining_frames[ids] = duration
        # One structured lower/central block persists with the fault event.
        height, width = p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH
        block_h = self._randint(max(4, height // 10), max(5, height // 3), (ids.numel(),))
        block_w = self._randint(max(4, width // 10), max(5, width // 2), (ids.numel(),))
        y0 = self._randint(height // 3, max(height // 3 + 1, height - int(block_h.max())), (ids.numel(),))
        x0 = self._randint(width // 4, max(width // 4 + 1, width - int(block_w.max())), (ids.numel(),))
        self.block[ids] = torch.stack((y0, x0, block_h, block_w), dim=-1)

    def _apply_capture_fault(self, clean: torch.Tensor, capture: torch.Tensor) -> torch.Tensor:
        result = clean.clone()
        ids = capture.nonzero(as_tuple=False).reshape(-1)
        if ids.numel() == 0:
            return result
        captured_values = result[ids]
        near_clip_normalized = (
            self.near_clip[ids] / self.max_depth_m
        ).reshape(-1, 1, 1, 1)
        near_invalid = (captured_values > 0.0) & (
            captured_values < near_clip_normalized
        )
        self.near_clip_added_hole_rate[ids] = near_invalid.float().mean(
            dim=(1, 2, 3)
        )
        captured_values = captured_values.masked_fill(near_invalid, 0.0)
        result[ids] = captured_values

        active_ids = ids[
            (self.fault_kind[ids] > 0) & self.pixel_fault_enabled[ids]
        ]
        if active_ids.numel() > 0:
            height, width = p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH
            kinds = self.fault_kind[active_ids]
            probability = torch.where(
                kinds == 1,
                torch.full_like(kinds, 0.03, dtype=torch.float32),
                torch.where(
                    kinds == 2,
                    torch.full_like(kinds, 0.12, dtype=torch.float32),
                    0.60 + 0.25 * self._uniform((active_ids.numel(),)),
                ),
            )
            random_mask = self._uniform(
                (active_ids.numel(), height, width)
            ) < probability.reshape(-1, 1, 1)
            y = torch.arange(height, device=self.device).reshape(1, height, 1)
            x = torch.arange(width, device=self.device).reshape(1, 1, width)
            y0, x0, block_h, block_w = self.block[active_ids].unbind(dim=-1)
            block_mask = (
                (y >= y0.reshape(-1, 1, 1))
                & (y < (y0 + block_h).reshape(-1, 1, 1))
                & (x >= x0.reshape(-1, 1, 1))
                & (x < (x0 + block_w).reshape(-1, 1, 1))
            )
            faulted = result[active_ids, :, :, 0].masked_fill(
                random_mask | block_mask, 0.0
            )
            result[active_ids, :, :, 0] = faulted
        self.fault_remaining_frames[capture] -= 1
        return result

    def process(
        self,
        clean_depth: torch.Tensor,
        *,
        reset_mask: torch.Tensor,
        session_effective_seconds: float,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean = clean_depth.to(self.device)
        if clean.shape != (
            self.num_envs,
            p2_contract.DEPTH_HEIGHT,
            p2_contract.DEPTH_WIDTH,
            1,
        ):
            raise ValueError(f"P4 camera depth shape drift: {tuple(clean.shape)}")
        self.reset(reset_mask)
        self.control_time_s += p2_contract.CONTROL_DT_S
        self.capture_accumulator += 30.0 / 50.0
        capture = self.capture_accumulator >= 1.0
        self.capture_accumulator[capture] -= 1.0
        capture |= reset_mask.to(self.device).bool().reshape(-1)
        if self.delivered is None:
            capture.fill_(True)
        if self.clean_capture is None:
            self.clean_capture = torch.zeros_like(clean)
        self.clean_capture[capture] = clean[capture]
        self._new_faults(capture)
        captured = self._apply_capture_fault(clean, capture) if training else clean.clone()
        if bool(capture.any()):
            self.cursor = (self.cursor + 1) % self.RING_SIZE
            if self.frames[self.cursor] is None:
                self.frames[self.cursor] = torch.zeros_like(clean)
            self.frames[self.cursor][capture] = captured[capture]
            self.frame_times[self.cursor, capture] = self.control_time_s
            self.frame_ids[self.cursor, capture] = self.next_frame_id
            self.next_frame_id += 1

        # Active delay is sampled once per fault event. Resampling it at 50 Hz
        # makes target_time non-monotonic and can replay older frames into both
        # recurrent policies.
        kind = self.fault_kind
        delay_s = self.delay_s
        selected_time = torch.full((self.num_envs,), -1.0e9, device=self.device)
        selected_id = torch.full_like(self.delivered_frame_id, -1)
        selected = torch.zeros_like(clean)
        target_time = self.control_time_s - delay_s
        for index in range(self.RING_SIZE):
            valid = (self.frame_ids[index] >= 0) & (self.frame_times[index] <= target_time)
            newer = valid & (self.frame_times[index] > selected_time)
            if bool(newer.any()) and self.frames[index] is not None:
                selected[newer] = self.frames[index][newer]
                selected_time[newer] = self.frame_times[index, newer]
                selected_id[newer] = self.frame_ids[index, newer]
        unavailable = selected_id < 0
        if bool(unavailable.any()):
            has_previous = unavailable & (self.delivered_frame_id >= 0)
            if bool(has_previous.any()) and self.delivered is not None:
                selected[has_previous] = self.delivered[has_previous]
                selected_time[has_previous] = self.delivered_frame_time[has_previous]
                selected_id[has_previous] = self.delivered_frame_id[has_previous]
            unavailable = selected_id < 0
            for index in range(self.RING_SIZE):
                valid = (self.frame_ids[index] >= 0) & unavailable
                newer = valid & (self.frame_times[index] > selected_time)
                if bool(newer.any()) and self.frames[index] is not None:
                    selected[newer] = self.frames[index][newer]
                    selected_time[newer] = self.frame_times[index, newer]
                    selected_id[newer] = self.frame_ids[index, newer]
        regressed = (self.delivered_frame_id >= 0) & (
            selected_id < self.delivered_frame_id
        )
        if bool(regressed.any()) and self.delivered is not None:
            selected[regressed] = self.delivered[regressed]
            selected_time[regressed] = self.delivered_frame_time[regressed]
            selected_id[regressed] = self.delivered_frame_id[regressed]
        changed = selected_id != self.delivered_frame_id
        self.delivered_frame_id.copy_(selected_id)
        self.delivered_frame_time.copy_(selected_time)
        self.delivered = selected
        hole_rate = (selected <= 0.0).float().mean(dim=(1, 2, 3))
        raw_hole_rate = (clean <= 0.0).float().mean(dim=(1, 2, 3))
        center = selected[:, 45:135, 80:240]
        lower = selected[:, 90:180]
        diagnostics = {
            "camera_capture": capture.float(),
            "camera_frame_changed": changed.float(),
            "camera_frame_id": selected_id.float(),
            "camera_age_s": (self.control_time_s - selected_time).clamp_min(0.0),
            "camera_delivered_hole_rate": hole_rate,
            # Compatibility alias for older P4 monitor captures.
            "camera_hole_rate": hole_rate,
            "camera_raw_hole_rate": raw_hole_rate,
            "camera_near_clip_added_hole_rate": (
                self.near_clip_added_hole_rate.detach().clone()
            ),
            "camera_center_hole_rate": (center <= 0.0).float().mean(dim=(1, 2, 3)),
            "camera_lower_hole_rate": (lower <= 0.0).float().mean(dim=(1, 2, 3)),
            "camera_delay_only": (self.delay_enabled & ~self.pixel_fault_enabled).float(),
            "camera_fault_only": (~self.delay_enabled & self.pixel_fault_enabled).float(),
            "camera_fault_delay_overlap": (self.delay_enabled & self.pixel_fault_enabled).float(),
            "camera_fault_kind": kind.float(),
            "camera_shadow_age_250ms": torch.clamp(delay_s + 0.10, max=0.25),
            "near_clip_m": self.near_clip.detach().clone(),
            "near_clip_normalized": (
                self.near_clip / self.max_depth_m
            ).detach().clone(),
        }
        self.last_diagnostics = diagnostics
        return selected, diagnostics

    def state_dict(self) -> dict[str, object]:
        return {
            "version": p4_contract.CAMERA_CONTRACT_VERSION,
            "generator_state": self.generator.get_state(),
            "near_clip_contract": "0.10+0.15*Beta(1,4)",
            "max_depth_m": self.max_depth_m,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("version") != p4_contract.CAMERA_CONTRACT_VERSION:
            raise ValueError("P4 camera checkpoint version mismatch")
        if abs(float(state.get("max_depth_m", self.max_depth_m)) - self.max_depth_m) > 1.0e-6:
            raise ValueError("P4 camera max-depth contract mismatch")
        generator_state = state.get("generator_state")
        if not torch.is_tensor(generator_state):
            raise ValueError("P4 camera checkpoint missing RNG state")
        # Rebuild live capture buffers without consuming the restored fault RNG.
        self.reset(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.generator.set_state(generator_state.cpu())
