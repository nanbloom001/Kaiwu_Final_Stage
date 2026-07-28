#!/usr/bin/env python3
"""P1.5 wall-clock command curriculum with independent vx/wz sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from agent_ppo.feature import p15_contract


@dataclass
class P15CommandState:
    active_target: torch.Tensor
    exec_command: torch.Tensor
    family: torch.Tensor
    trajectory_mode: torch.Tensor
    command_epoch: torch.Tensor
    phase_index: int
    original_replay: torch.Tensor
    change_mode: torch.Tensor
    emergency_stop: torch.Tensor


class P15CommandSchedule:
    """Per-environment 5 Hz target sampler and 50 Hz slew controller."""

    def __init__(self, num_envs: int, device, *, seed: int = 0, config=None):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.config = dict(config or {})
        self.dt_s = float(self.config.get("step_dt_s", p15_contract.CONTROL_DT_S))
        self.target_period_frames = int(
            self.config.get("target_period_frames", p15_contract.TARGET_PERIOD_FRAMES)
        )
        self.slew_rate = torch.tensor(
            self.config.get("slew_rate", p15_contract.SLEW_RATE),
            dtype=torch.float32,
            device=self.device,
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.active_target = torch.zeros(self.num_envs, 3, device=self.device)
        self.exec_command = torch.zeros_like(self.active_target)
        self.smooth_destination = torch.zeros_like(self.active_target)
        self.smooth_steps_left = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.hold_nav_ticks = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.family = torch.full(
            (self.num_envs,), 3, dtype=torch.long, device=self.device
        )
        self.trajectory_mode = torch.full(
            (self.num_envs,), 2, dtype=torch.long, device=self.device
        )
        self.command_epoch = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.original_replay = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.change_mode = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.emergency_stop = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.frame = 0
        self.elapsed_h = 0.0
        self._initialized = False
        self._counts = torch.zeros(
            len(p15_contract.COMMAND_FAMILIES), dtype=torch.long, device=self.device
        )
        self._active_family_counts = torch.zeros_like(self._counts)
        self._trajectory_counts = torch.zeros(
            len(p15_contract.TRAJECTORY_MODES), dtype=torch.long, device=self.device
        )
        self._change_counts = torch.zeros(
            len(p15_contract.CHANGE_MODES), dtype=torch.long, device=self.device
        )
        self._phase_counts = torch.zeros(5, dtype=torch.long, device=self.device)
        self._replay_count = 0
        self._sample_count = 0
        self._active_frame_count = 0
        self._trajectory_sample_count = 0

    def _rand(self, shape) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator, device=self.device)

    def _uniform(self, low: float, high: float, count: int) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, device=self.device)
        return low + (high - low) * self._rand((count,))

    def _categorical(self, weights, count: int) -> torch.Tensor:
        probabilities = torch.tensor(weights, device=self.device, dtype=torch.float32)
        probabilities = probabilities / probabilities.sum()
        return torch.multinomial(
            probabilities, count, replacement=True, generator=self.generator
        )

    def reset(self, reset_mask: torch.Tensor, native_command: torch.Tensor) -> None:
        mask = reset_mask.to(self.device).reshape(-1).bool()
        if mask.numel() != self.num_envs:
            raise ValueError("reset mask does not match command scheduler env count")
        if not bool(mask.any()):
            return
        native = native_command.to(self.device, dtype=torch.float32)[:, :3]
        self.active_target[mask] = native[mask]
        self.exec_command[mask] = native[mask]
        self.smooth_destination[mask] = native[mask]
        self.smooth_steps_left[mask] = 0
        self.hold_nav_ticks[mask] = 0
        self.family[mask] = 3
        self.trajectory_mode[mask] = 2
        self.change_mode[mask] = -1
        self.emergency_stop[mask] = False
        self.command_epoch[mask] += 1

    def _sample_original(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = int(ids.numel())
        commands = torch.zeros(count, 3, device=self.device)
        emergency_stop = torch.zeros(count, dtype=torch.bool, device=self.device)
        families = self._categorical(
            p15_contract.COMMAND_FAMILY_WEIGHTS, count
        )
        self._fill_by_family(
            commands,
            families,
            vx_max=1.3,
            vy_max=0.2,
            wz_max=0.3,
            original=True,
            emergency_stop=emergency_stop,
        )
        return commands, families, emergency_stop

    def _sample_expanded(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = int(ids.numel())
        limits = p15_contract.command_limits(self.elapsed_h)
        commands = torch.zeros(count, 3, device=self.device)
        emergency_stop = torch.zeros(count, dtype=torch.bool, device=self.device)
        families = self._categorical(
            p15_contract.COMMAND_FAMILY_WEIGHTS, count
        )
        self._fill_by_family(
            commands,
            families,
            vx_max=limits["vx"][1],
            vy_max=max(abs(limits["vy"][0]), abs(limits["vy"][1])),
            wz_max=max(abs(limits["wz"][0]), abs(limits["wz"][1])),
            original=False,
            emergency_stop=emergency_stop,
        )
        return commands, families, emergency_stop

    def _fill_by_family(
        self,
        commands: torch.Tensor,
        families: torch.Tensor,
        *,
        vx_max: float,
        vy_max: float,
        wz_max: float,
        original: bool,
        emergency_stop: torch.Tensor,
    ) -> None:
        def signed(low: float, high: float, count: int) -> torch.Tensor:
            magnitude = self._uniform(low, high, count)
            sign = torch.where(self._rand((count,)) < 0.5, -1.0, 1.0)
            return magnitude * sign

        joint = families == 0
        n = int(joint.sum().item())
        if n:
            # Independent stratified bins avoid a shared scale factor between vx/wz.
            vx_bin = torch.randint(0, 4, (n,), generator=self.generator, device=self.device)
            wz_bin = torch.randint(0, 4, (n,), generator=self.generator, device=self.device)
            vx_min = 0.3 if original else 0.0
            vx_edges = torch.linspace(vx_min, vx_max, 5, device=self.device)
            wz_edges = torch.linspace(0.03 if original else 0.0, wz_max, 5, device=self.device)
            commands[joint, 0] = vx_edges[vx_bin] + self._rand((n,)) * (
                vx_edges[vx_bin + 1] - vx_edges[vx_bin]
            )
            magnitude = wz_edges[wz_bin] + self._rand((n,)) * (
                wz_edges[wz_bin + 1] - wz_edges[wz_bin]
            )
            commands[joint, 2] = magnitude * torch.where(
                self._rand((n,)) < 0.5, -1.0, 1.0
            )
            if original:
                # The 34728 source box included simultaneous vy variation. Keep
                # it only in source-domain replay; expanded capability treats vy
                # as the explicit 5% lateral specialty.
                commands[joint, 1] = self._uniform(-vy_max, vy_max, n)

        straight = families == 1
        n = int(straight.sum().item())
        if n:
            low = 0.3 if original else 0.0
            commands[straight, 0] = self._uniform(low, vx_max, n)

        pure_yaw = families == 2
        n = int(pure_yaw.sum().item())
        if n:
            commands[pure_yaw, 2] = signed(0.05, wz_max, n)

        transition = families == 3
        n = int(transition.sum().item())
        if n:
            # This family is reserved for zero/start/brake events. Start uses
            # a small non-zero command; zero and brake both target an immediate
            # stop, with the previous command distinguishing their semantics.
            subtype = torch.randint(0, 3, (n,), generator=self.generator, device=self.device)
            start = subtype == 1
            brake = subtype == 2
            values = torch.zeros(n, 3, device=self.device)
            if bool(start.any()):
                start_yaw = self._rand((int(start.sum()),)) < 0.35
                start_values = torch.zeros(int(start.sum()), 3, device=self.device)
                if bool((~start_yaw).any()):
                    start_values[~start_yaw, 0] = self._uniform(
                        0.05, min(vx_max, 0.30), int((~start_yaw).sum())
                    )
                if bool(start_yaw.any()):
                    start_values[start_yaw, 2] = signed(
                        0.05, min(wz_max, 0.30), int(start_yaw.sum())
                    )
                values[start] = start_values
            commands[transition] = values
            transition_ids = torch.nonzero(transition, as_tuple=False).flatten()
            emergency_stop[transition_ids[brake]] = True

        lateral = families == 4
        n = int(lateral.sum().item())
        if n:
            commands[lateral, 1] = signed(0.05, vy_max, n)

    def _sample_targets(self, ids: torch.Tensor) -> None:
        count = int(ids.numel())
        if count == 0:
            return
        replay_probability = p15_contract.original_replay_probability(self.elapsed_h)
        replay = self._rand((count,)) < replay_probability
        targets = torch.zeros(count, 3, device=self.device)
        families = torch.zeros(count, dtype=torch.long, device=self.device)
        emergency_stop = torch.zeros(count, dtype=torch.bool, device=self.device)
        if bool(replay.any()):
            replay_targets, replay_families, replay_stop = self._sample_original(
                ids[replay]
            )
            targets[replay] = replay_targets
            families[replay] = replay_families
            emergency_stop[replay] = replay_stop
        if bool((~replay).any()):
            expanded_targets, expanded_families, expanded_stop = self._sample_expanded(
                ids[~replay]
            )
            targets[~replay] = expanded_targets
            families[~replay] = expanded_families
            emergency_stop[~replay] = expanded_stop

        # Late-stage retention keeps a small source-domain 1.0-1.3 m/s slice.
        if self.elapsed_h >= 2.0 and bool(replay.any()):
            retain = replay & (families == 1) & (self._rand((count,)) < 0.50)
            n_retain = int(retain.sum().item())
            if n_retain:
                targets[retain] = 0.0
                targets[retain, 0] = self._uniform(1.0, 1.3, n_retain)
                families[retain] = 1

        previous = self.active_target[ids].clone()
        self.emergency_stop[ids] = False
        change_modes = self._categorical(p15_contract.CHANGE_MODE_WEIGHTS, count)
        joint = families == 0
        if bool(joint.any()):
            joint_previous = previous[joint]
            joint_modes = change_modes[joint]
            joint_targets = targets[joint]
            vx_only = joint_modes == 0
            wz_only = joint_modes == 1
            if bool(vx_only.any()):
                joint_targets[vx_only, 2] = joint_previous[vx_only, 2]
            if bool(wz_only.any()):
                joint_targets[wz_only, 0] = joint_previous[wz_only, 0]
            targets[joint] = joint_targets

        modes = self._categorical(p15_contract.TRAJECTORY_MODE_WEIGHTS, count)
        # The brake subtype is the explicit emergency-stop case: it bypasses
        # slew and cannot be converted into a smooth or no-op hold event.
        modes[emergency_stop] = 1
        smooth = modes == 0
        step = modes == 1
        hold = modes == 2
        if bool(smooth.any()):
            smooth_ids = ids[smooth]
            self.smooth_destination[smooth_ids] = targets[smooth]
            self.smooth_steps_left[smooth_ids] = torch.randint(
                3, 9, (int(smooth.sum()),), generator=self.generator, device=self.device
            )
        if bool(step.any()):
            self.active_target[ids[step]] = targets[step]
            self.hold_nav_ticks[ids[step]] = torch.randint(
                8, 31, (int(step.sum()),), generator=self.generator, device=self.device
            )
        if bool(emergency_stop.any()):
            self.exec_command[ids[emergency_stop]] = 0.0
            self.emergency_stop[ids[emergency_stop]] = True
        if bool(hold.any()):
            # Stable hold deliberately preserves the current target.
            self.hold_nav_ticks[ids[hold]] = torch.randint(
                8, 31, (int(hold.sum()),), generator=self.generator, device=self.device
            )
        applied = ~hold
        applied_ids = ids[applied]
        self.family[applied_ids] = families[applied]
        self.trajectory_mode[ids] = modes
        self.original_replay[applied_ids] = replay[applied]
        self.change_mode[applied_ids] = change_modes[applied]
        step_changed = step & (targets - previous).abs().amax(dim=1).gt(1.0e-8)
        self.command_epoch[ids[step_changed]] += 1
        applied_families = families[applied]
        self._counts.scatter_add_(
            0, applied_families, torch.ones_like(applied_families)
        )
        self._trajectory_counts.scatter_add_(0, modes, torch.ones_like(modes))
        applied_joint_modes = change_modes[applied & joint]
        if applied_joint_modes.numel():
            self._change_counts.scatter_add_(
                0, applied_joint_modes, torch.ones_like(applied_joint_modes)
            )
        applied_count = int(applied.sum().item())
        self._phase_counts[p15_contract.phase_index(self.elapsed_h)] += applied_count
        self._replay_count += int(replay[applied].sum().item())
        self._sample_count += applied_count
        self._trajectory_sample_count += count

    def _nav_tick(self) -> None:
        smooth = self.smooth_steps_left > 0
        if bool(smooth.any()):
            steps = self.smooth_steps_left[smooth].to(torch.float32).unsqueeze(-1)
            delta = (self.smooth_destination[smooth] - self.active_target[smooth]) / steps
            self.active_target[smooth] += delta
            self.smooth_steps_left[smooth] -= 1
            changed = delta.abs().amax(dim=1) > 1.0e-8
            smooth_ids = torch.nonzero(smooth, as_tuple=False).flatten()
            self.command_epoch[smooth_ids[changed]] += 1

        holding = self.hold_nav_ticks > 0
        self.hold_nav_ticks[holding] -= 1
        due = (self.smooth_steps_left <= 0) & (self.hold_nav_ticks <= 0)
        ids = torch.nonzero(due, as_tuple=False).flatten()
        self._sample_targets(ids)

    def step(
        self,
        native_command: torch.Tensor,
        *,
        elapsed_h: float,
        reset_mask: torch.Tensor | None = None,
        emergency_stop: torch.Tensor | None = None,
    ) -> P15CommandState:
        native = native_command.to(self.device, dtype=torch.float32)[:, :3]
        if native.shape != self.active_target.shape:
            raise ValueError(
                f"native command shape {tuple(native.shape)} != {tuple(self.active_target.shape)}"
            )
        self.elapsed_h = max(0.0, float(elapsed_h))
        if not self._initialized:
            self.active_target.copy_(native)
            self.exec_command.copy_(native)
            self.hold_nav_ticks.zero_()
            self._initialized = True
        if reset_mask is not None:
            self.reset(reset_mask, native)
        if self.frame % self.target_period_frames == 0:
            self._nav_tick()

        max_delta = self.slew_rate * self.dt_s
        delta = torch.clamp(
            self.active_target - self.exec_command,
            min=-max_delta,
            max=max_delta,
        )
        self.exec_command += delta
        if emergency_stop is not None:
            stop = emergency_stop.to(self.device).reshape(-1).bool()
            changed = stop & (
                self.active_target.abs().amax(dim=1).gt(1.0e-8)
                | self.exec_command.abs().amax(dim=1).gt(1.0e-8)
            )
            self.active_target[stop] = 0.0
            self.exec_command[stop] = 0.0
            self.command_epoch[changed] += 1
        self._active_family_counts.scatter_add_(
            0,
            self.family,
            torch.ones_like(self.family, dtype=torch.long),
        )
        self._active_frame_count += self.num_envs
        self.frame += 1
        return self.state()

    def state(self) -> P15CommandState:
        return P15CommandState(
            active_target=self.active_target.clone(),
            exec_command=self.exec_command.clone(),
            family=self.family.clone(),
            trajectory_mode=self.trajectory_mode.clone(),
            command_epoch=self.command_epoch.clone(),
            phase_index=p15_contract.phase_index(self.elapsed_h),
            original_replay=self.original_replay.clone(),
            change_mode=self.change_mode.clone(),
            emergency_stop=self.emergency_stop.clone(),
        )

    def metrics(self) -> dict[str, object]:
        total = max(1, self._sample_count)
        trajectory_total = max(1, self._trajectory_sample_count)
        active_total = max(1, self._active_frame_count)
        change_total = max(1, int(self._change_counts.sum().item()))
        return {
            "phase_index": p15_contract.phase_index(self.elapsed_h),
            "original_replay_ratio": self._replay_count / total,
            "resampled_family_ratio": {
                name: float(self._counts[index].item()) / total
                for index, name in enumerate(p15_contract.COMMAND_FAMILIES)
            },
            "active_family_ratio": {
                name: float(self._active_family_counts[index].item()) / active_total
                for index, name in enumerate(p15_contract.COMMAND_FAMILIES)
            },
            "trajectory_ratio": {
                name: float(self._trajectory_counts[index].item()) / trajectory_total
                for index, name in enumerate(p15_contract.TRAJECTORY_MODES)
            },
            "change_ratio": {
                name: float(self._change_counts[index].item()) / change_total
                for index, name in enumerate(p15_contract.CHANGE_MODES)
            },
            "sample_count": self._sample_count,
            "active_frame_count": self._active_frame_count,
        }
