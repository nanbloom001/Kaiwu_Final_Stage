#!/usr/bin/env python3
"""CPU-backed recurrent rollout storage for P2 semi-MDP PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from agent_ppo.feature import p2_contract


def _cpu_tensor(*shape, dtype=torch.float32, pin_memory: bool = False):
    use_pin = bool(pin_memory and torch.cuda.is_available())
    return torch.empty(*shape, dtype=dtype, device="cpu", pin_memory=use_pin)


@dataclass(frozen=True)
class SequenceRef:
    start: int
    env: int


class P2RolloutStorage:
    """Stores depth on pinned CPU while recurrent PPO replays 16-step chunks."""

    def __init__(
        self,
        num_envs: int,
        *,
        num_ticks: int = p2_contract.NAV_ROLLOUT_TICKS,
        sequence_length: int = p2_contract.TBPTT_SEQUENCE_LENGTH,
        store_depth: bool,
        pin_memory: bool = True,
        actor_layers: int = 2,
        actor_hidden: int = 64,
        critic_layers: int = 2,
        critic_hidden: int = 64,
    ):
        self.num_envs = int(num_envs)
        self.num_ticks = int(num_ticks)
        self.sequence_length = int(sequence_length)
        if self.num_ticks % self.sequence_length:
            raise ValueError("P2 rollout ticks must be divisible by TBPTT length")
        self.store_depth = bool(store_depth)
        self.step = 0
        common = (self.num_ticks, self.num_envs)
        self.depth = (
            _cpu_tensor(*common, p2_contract.DEPTH_DIM, dtype=torch.float16, pin_memory=pin_memory)
            if self.store_depth
            else None
        )
        self.nav_feat = (
            None
            if self.store_depth
            else _cpu_tensor(*common, p2_contract.NAV_FEATURE_DIM)
        )
        self.nav_nonvisual = _cpu_tensor(*common, p2_contract.NAV_NONVISUAL_DIM)
        self.response_profile = _cpu_tensor(*common, p2_contract.RESPONSE_PROFILE_DIM)
        self.confidence = _cpu_tensor(*common, 1)
        self.safety_target = _cpu_tensor(*common, 3)
        self.safety_valid = _cpu_tensor(*common, 1)
        self.critic_input = _cpu_tensor(*common, p2_contract.CRITIC_INPUT_DIM)
        self.pre_tanh_action = _cpu_tensor(*common, p2_contract.ACTION_DIM)
        self.old_log_prob = _cpu_tensor(*common, 1)
        self.old_value = _cpu_tensor(*common, 1)
        self.rewards = _cpu_tensor(*common, 1)
        self.duration_frames = _cpu_tensor(*common, 1, dtype=torch.long)
        self.bootstrap_value = _cpu_tensor(*common, 1)
        self.bootstrap_mask = _cpu_tensor(*common, 1)
        self.continuation_mask = _cpu_tensor(*common, 1)
        self.reset_mask = _cpu_tensor(*common, dtype=torch.bool)
        self.actor_h = _cpu_tensor(self.num_ticks, actor_layers, self.num_envs, actor_hidden)
        self.actor_c = _cpu_tensor(self.num_ticks, actor_layers, self.num_envs, actor_hidden)
        self.critic_h = _cpu_tensor(self.num_ticks, critic_layers, self.num_envs, critic_hidden)
        self.critic_c = _cpu_tensor(self.num_ticks, critic_layers, self.num_envs, critic_hidden)
        self.advantages = _cpu_tensor(*common, 1)
        self.returns = _cpu_tensor(*common, 1)

    @property
    def full(self) -> bool:
        return self.step == self.num_ticks

    @staticmethod
    def _copy(target: torch.Tensor, value: torch.Tensor) -> None:
        target.copy_(value.detach().to(device="cpu", dtype=target.dtype))

    @staticmethod
    def own_depth_sample(value: torch.Tensor, *, pin_memory: bool = True) -> torch.Tensor:
        """Take immediate CPU FP16 ownership of a camera frame."""
        owned = _cpu_tensor(
            *value.shape,
            dtype=torch.float16,
            pin_memory=pin_memory,
        )
        owned.copy_(value.detach(), non_blocking=False)
        return owned

    def add(self, **transition) -> None:
        if self.full:
            raise RuntimeError("P2 rollout is already full")
        index = self.step
        required = {
            "nav_nonvisual", "response_profile", "confidence", "critic_input",
            "pre_tanh_action", "old_log_prob", "old_value", "reward",
            "safety_target", "safety_valid",
            "duration_frames", "bootstrap_value", "bootstrap_mask",
            "continuation_mask", "reset_mask", "actor_hidden", "critic_hidden",
        }
        missing = required.difference(transition)
        if missing:
            raise ValueError(f"P2 transition missing fields: {sorted(missing)}")
        if self.store_depth:
            depth = transition["depth"]
            if depth.numel() != self.num_envs * p2_contract.DEPTH_DIM:
                raise ValueError(
                    "P2 depth transition must contain "
                    f"{self.num_envs * p2_contract.DEPTH_DIM} values, got "
                    f"{depth.numel()}"
                )
            self._copy(
                self.depth[index],
                depth.reshape(self.num_envs, p2_contract.DEPTH_DIM),
            )
        else:
            self._copy(self.nav_feat[index], transition["nav_feat"])
        for name, source in (
            ("nav_nonvisual", "nav_nonvisual"),
            ("response_profile", "response_profile"),
            ("confidence", "confidence"),
            ("safety_target", "safety_target"),
            ("safety_valid", "safety_valid"),
            ("critic_input", "critic_input"),
            ("pre_tanh_action", "pre_tanh_action"),
            ("old_log_prob", "old_log_prob"),
            ("old_value", "old_value"),
            ("rewards", "reward"),
            ("duration_frames", "duration_frames"),
            ("bootstrap_value", "bootstrap_value"),
            ("bootstrap_mask", "bootstrap_mask"),
            ("continuation_mask", "continuation_mask"),
            ("reset_mask", "reset_mask"),
        ):
            self._copy(getattr(self, name)[index], transition[source])
        actor_h, actor_c = transition["actor_hidden"]
        critic_h, critic_c = transition["critic_hidden"]
        self._copy(self.actor_h[index], actor_h)
        self._copy(self.actor_c[index], actor_c)
        self._copy(self.critic_h[index], critic_h)
        self._copy(self.critic_c[index], critic_c)
        self.step += 1

    def compute_returns(self) -> None:
        if not self.full:
            raise RuntimeError("cannot compute P2 GAE before rollout is full")
        next_gae = torch.zeros(self.num_envs, 1)
        for step in reversed(range(self.num_ticks)):
            discount = torch.pow(
                torch.tensor(p2_contract.GAMMA_FRAME),
                self.duration_frames[step].to(torch.float32),
            )
            delta = (
                self.rewards[step]
                + self.bootstrap_mask[step]
                * discount
                * self.bootstrap_value[step]
                - self.old_value[step]
            )
            next_gae = (
                delta
                + self.continuation_mask[step]
                * discount
                * p2_contract.GAE_LAMBDA
                * next_gae
            )
            self.advantages[step] = next_gae
        self.returns.copy_(self.advantages + self.old_value)

    def sequence_refs(self, generator: torch.Generator | None = None) -> list[SequenceRef]:
        refs = [
            SequenceRef(start, env)
            for start in range(0, self.num_ticks, self.sequence_length)
            for env in range(self.num_envs)
        ]
        order = torch.randperm(len(refs), generator=generator).tolist()
        return [refs[index] for index in order]

    @staticmethod
    def microbatch_loss_scale(
        micro_valid_steps: int,
        minibatch_valid_steps: int,
    ) -> float:
        if minibatch_valid_steps <= 0 or micro_valid_steps < 0:
            raise ValueError("invalid P2 minibatch valid-step counts")
        return float(micro_valid_steps) / float(minibatch_valid_steps)

    def reset(self, *, store_depth: bool | None = None) -> "P2RolloutStorage":
        requested = self.store_depth if store_depth is None else bool(store_depth)
        if requested != self.store_depth:
            return P2RolloutStorage(
                self.num_envs,
                num_ticks=self.num_ticks,
                sequence_length=self.sequence_length,
                store_depth=requested,
            )
        self.step = 0
        return self
