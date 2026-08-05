#!/usr/bin/env python3
"""Aisrv-owned P2 target/exec command state."""

from __future__ import annotations

import torch

from agent_ppo.feature import nav_contract, p2_contract


class P2CommandController:
    def __init__(
        self,
        num_envs: int,
        device,
        *,
        slew_rate=(0.30, 0.30, 1.00),
        slew_release_rate=(0.30, 0.60, 2.50),
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
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
        self.command_epoch += 1

    def inject(self, policy_obs: torch.Tensor, critic_obs: torch.Tensor) -> None:
        p0, p1 = nav_contract.POLICY_CMD_SLICE
        c0, c1 = nav_contract.CRITIC_CMD_SLICE
        policy_obs[:, p0:p1] = self.exec_cmd.to(policy_obs)
        critic_obs[:, c0:c1] = self.exec_cmd.to(critic_obs)

    def step(self) -> None:
        target = self.active_target
        current = self.exec_cmd.clone()
        opposite = (target * current) < 0.0
        reducing = target.abs() < current.abs()
        release = opposite | reducing
        rate = torch.where(release, self.slew_release_rate, self.slew_rate)
        effective_target = torch.where(opposite, torch.zeros_like(target), target)
        max_delta = rate * p2_contract.CONTROL_DT_S
        delta = torch.clamp(effective_target - current, -max_delta, max_delta)
        self.exec_cmd.add_(delta)
        crossed_zero = opposite & ((self.exec_cmd * current) <= 0.0)
        self.exec_cmd.copy_(
            torch.where(crossed_zero, torch.zeros_like(self.exec_cmd), self.exec_cmd)
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.active_target[env_ids] = 0.0
        self.exec_cmd[env_ids] = 0.0

    def state_dict(self) -> dict[str, object]:
        return {
            "command_epoch": int(self.command_epoch),
            "slew_rate": self.slew_rate.detach().cpu().flatten().tolist(),
            "slew_release_rate": self.slew_release_rate.detach().cpu().flatten().tolist(),
            "live_state_persisted": False,
        }
