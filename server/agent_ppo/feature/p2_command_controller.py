#!/usr/bin/env python3
"""Aisrv-owned P2 target/exec command state."""

from __future__ import annotations

import torch

from agent_ppo.feature import nav_contract, p2_contract


def slew_step(
    current_cmd3: torch.Tensor,
    target_cmd3: torch.Tensor,
    slew_rate: torch.Tensor,
    slew_release_rate: torch.Tensor,
    *,
    dt_s: float = p2_contract.CONTROL_DT_S,
) -> torch.Tensor:
    """Advance one command frame while requiring a zero crossing on reversals."""
    if current_cmd3.shape != target_cmd3.shape or current_cmd3.shape[-1] != 3:
        raise ValueError("command slew expects matching [N,3] current and target tensors")
    up = torch.as_tensor(
        slew_rate, device=current_cmd3.device, dtype=current_cmd3.dtype
    ).reshape(1, 3)
    release = torch.as_tensor(
        slew_release_rate, device=current_cmd3.device, dtype=current_cmd3.dtype
    ).reshape(1, 3)
    if not torch.isfinite(up).all() or not torch.isfinite(release).all():
        raise ValueError("command slew rates must be finite")
    if bool((up < 0.0).any()) or bool((release < 0.0).any()):
        raise ValueError("command slew rates must be non-negative")
    if not torch.isfinite(torch.as_tensor(float(dt_s))) or float(dt_s) <= 0.0:
        raise ValueError("command slew dt_s must be positive and finite")
    opposite = (target_cmd3 * current_cmd3) < 0.0
    reducing = target_cmd3.abs() < current_cmd3.abs()
    rate = torch.where(opposite | reducing, release, up)
    effective_target = torch.where(opposite, torch.zeros_like(target_cmd3), target_cmd3)
    max_delta = rate * float(dt_s)
    next_cmd = current_cmd3 + torch.clamp(
        effective_target - current_cmd3, -max_delta, max_delta
    )
    crossed_zero = opposite & ((next_cmd * current_cmd3) <= 0.0)
    return torch.where(crossed_zero, torch.zeros_like(next_cmd), next_cmd)


class P2CommandController:
    _SLEW_MODE = "slew"
    _INSTANT_HOLD_10HZ_MODE = "instant_hold_10hz"
    _INSTANT_HOLD_10HZ_FRAMES = 5

    def __init__(
        self,
        num_envs: int,
        device,
        *,
        slew_rate=(0.30, 0.30, 1.00),
        slew_release_rate=(0.30, 0.60, 2.50),
        command_transition_mode: str = _SLEW_MODE,
    ):
        if command_transition_mode not in {
            self._SLEW_MODE,
            self._INSTANT_HOLD_10HZ_MODE,
        }:
            raise ValueError(
                "Unsupported P2 command transition mode: "
                f"{command_transition_mode!r}"
            )
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.command_transition_mode = command_transition_mode
        self.hold_frames = self._INSTANT_HOLD_10HZ_FRAMES
        self.slew_rate = torch.tensor(slew_rate, device=self.device).reshape(1, 3)
        self.slew_release_rate = torch.tensor(
            slew_release_rate, device=self.device
        ).reshape(1, 3)
        self.active_target = torch.zeros(self.num_envs, 3, device=self.device)
        self.exec_cmd = torch.zeros_like(self.active_target)
        self.command_epoch = 0

    def set_target(self, target_cmd3: torch.Tensor) -> None:
        if target_cmd3.shape != self.active_target.shape:
            raise ValueError(f"P2 target shape drift: {tuple(target_cmd3.shape)}")
        self.active_target.copy_(target_cmd3.to(self.device))
        if self.command_transition_mode == self._INSTANT_HOLD_10HZ_MODE:
            self.exec_cmd.copy_(self.active_target)
        self.command_epoch += 1

    def inject(self, policy_obs: torch.Tensor, critic_obs: torch.Tensor) -> None:
        p0, p1 = nav_contract.POLICY_CMD_SLICE
        c0, c1 = nav_contract.CRITIC_CMD_SLICE
        policy_obs[:, p0:p1] = self.exec_cmd.to(policy_obs)
        critic_obs[:, c0:c1] = self.exec_cmd.to(critic_obs)

    def step(self) -> None:
        if self.command_transition_mode == self._INSTANT_HOLD_10HZ_MODE:
            return
        self.exec_cmd.copy_(
            slew_step(
                self.exec_cmd,
                self.active_target,
                self.slew_rate,
                self.slew_release_rate,
            )
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.active_target[env_ids] = 0.0
        self.exec_cmd[env_ids] = 0.0

    def state_dict(self) -> dict[str, object]:
        state = {
            "command_epoch": int(self.command_epoch),
            "command_transition_mode": self.command_transition_mode,
            "live_state_persisted": False,
        }
        if self.command_transition_mode == self._INSTANT_HOLD_10HZ_MODE:
            state["hold_frames"] = int(self.hold_frames)
        else:
            state["slew_rate"] = self.slew_rate.detach().cpu().flatten().tolist()
            state["slew_release_rate"] = (
                self.slew_release_rate.detach().cpu().flatten().tolist()
            )
        return state
