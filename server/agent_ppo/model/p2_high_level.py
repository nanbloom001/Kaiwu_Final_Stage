#!/usr/bin/env python3
"""Continuous recurrent P2 navigation actor and asymmetric critic."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.feature import p2_contract
from agent_ppo.model.simple_cnn import SimpleCNN


def _apply_reset(hidden, reset_mask: torch.Tensor | None):
    if hidden is None or reset_mask is None or not bool(reset_mask.any()):
        return hidden
    keep = (~reset_mask.bool()).reshape(1, -1, 1)
    return hidden[0] * keep, hidden[1] * keep


def squashed_log_prob(
    pre_tanh: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
    """Stable log probability for a diagonal tanh-squashed Gaussian."""
    log_std = torch.clamp(log_std, p2_contract.LOG_STD_MIN, p2_contract.LOG_STD_MAX)
    inv_std = torch.exp(-log_std)
    gaussian = -0.5 * ((pre_tanh - mean) * inv_std).square()
    gaussian = gaussian - log_std - 0.5 * math.log(2.0 * math.pi)
    # log(1 - tanh(x)^2), written without subtractive cancellation.
    correction = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
    return (gaussian - correction).sum(dim=-1, keepdim=True)


class NavigationEncoder(nn.Module):
    """Independent depth-only CNN; never shares parameters with the low level."""

    def __init__(self):
        super().__init__()
        self.activation_checkpointing = False
        self.cnn = SimpleCNN(
            input_shape=(
                p2_contract.DEPTH_HEIGHT,
                p2_contract.DEPTH_WIDTH,
                p2_contract.DEPTH_CHANNELS,
            ),
            output_dim=p2_contract.NAV_FEATURE_DIM,
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(self.cnn, depth, use_reentrant=False)
        return self.cnn(depth)

    def copy_from_low_level_cnn(self, low_level_cnn: nn.Module) -> None:
        self.cnn.load_state_dict(low_level_cnn.state_dict(), strict=True)
        for own, source in zip(self.cnn.parameters(), low_level_cnn.parameters()):
            if own is source or own.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
                raise RuntimeError("NavigationEncoder illegally aliases the low-level CNN")

    def parameter_groups(self) -> dict[str, Iterable[nn.Parameter]]:
        return {
            "conv1": self.cnn.conv_layers[0].parameters(),
            "conv2": self.cnn.conv_layers[1].parameters(),
            "conv3": self.cnn.conv_layers[2].parameters(),
            "fc": self.cnn.fc.parameters(),
        }


class NavigationSafetyHead(nn.Module):
    """Training-only depth feature head for privileged safety supervision."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p2_contract.NAV_FEATURE_DIM, p2_contract.NAV_FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(p2_contract.NAV_FEATURE_DIM, 3),
        )

    def forward(self, nav_feat: torch.Tensor) -> torch.Tensor:
        return self.net(nav_feat)


class P4ActorStuckHead(nn.Module):
    """Training-only wall-stuck classifier over the recurrent Actor state."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.logit = nn.Linear(self.hidden_dim, 1)

    def forward(self, actor_features: torch.Tensor) -> torch.Tensor:
        return self.logit(actor_features)


class P2NavigationActor(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.memory = nn.LSTM(
            p2_contract.ACTOR_INPUT_DIM,
            self.hidden_dim,
            self.num_layers,
        )
        # Keep the legacy 2-D head names stable so 291713 weights and Adam
        # moments can be migrated by parameter name without reshaping.
        self.mean_head = nn.Linear(self.hidden_dim, 2)
        self.vy_mean_head = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.vy_mean_head.weight)
        with torch.no_grad():
            self.mean_head.bias.copy_(
                torch.tensor((p2_contract.INITIAL_VX_PRE_TANH, 0.0))
            )
            self.vy_mean_head.bias.zero_()
        self.log_std = nn.Parameter(
            torch.full((2,), p2_contract.INITIAL_LOG_STD)
        )
        self.vy_log_std = nn.Parameter(
            torch.full((1,), p2_contract.INITIAL_VY_LOG_STD)
        )

    def _memory_forward(self, inputs, hidden=None, reset_mask=None):
        if inputs.ndim == 2:
            hidden = _apply_reset(hidden, reset_mask)
            output, hidden = self.memory(inputs.unsqueeze(0), hidden)
            return output.squeeze(0), hidden
        if inputs.ndim != 3:
            raise ValueError("actor input must be [B,85] or [T,B,85]")
        outputs = []
        for step in range(inputs.shape[0]):
            step_reset = None if reset_mask is None else reset_mask[step]
            hidden = _apply_reset(hidden, step_reset)
            output, hidden = self.memory(inputs[step : step + 1], hidden)
            outputs.append(output)
        return torch.cat(outputs, dim=0), hidden

    def distribution_parameters(self, inputs, hidden=None, reset_mask=None):
        features, hidden = self._memory_forward(inputs, hidden, reset_mask)
        main_mean = self.mean_head(features)
        vy_mean = self.vy_mean_head(features)
        mean = torch.cat((main_mean[..., 0:1], vy_mean, main_mean[..., 1:2]), dim=-1)
        main_log_std = torch.clamp(
            self.log_std, p2_contract.LOG_STD_MIN, p2_contract.LOG_STD_MAX
        )
        vy_log_std = torch.clamp(
            self.vy_log_std, p2_contract.LOG_STD_MIN, p2_contract.LOG_STD_MAX
        )
        log_std = torch.cat(
            (main_log_std[0:1], vy_log_std, main_log_std[1:2]), dim=0
        )
        log_std = log_std.expand_as(mean)
        return mean, log_std, hidden

    def sample(
        self,
        inputs,
        hidden=None,
        reset_mask=None,
        generator=None,
        vy_generator=None,
        hard_abs_vy: float = 0.40,
    ):
        mean, log_std, hidden = self.distribution_parameters(inputs, hidden, reset_mask)
        main_noise = torch.randn(
            (*mean.shape[:-1], 2),
            device=mean.device,
            dtype=mean.dtype,
            generator=generator,
        )
        vy_noise = torch.randn(
            (*mean.shape[:-1], 1),
            device=mean.device,
            dtype=mean.dtype,
            generator=vy_generator if vy_generator is not None else generator,
        )
        noise = torch.cat((main_noise[..., 0:1], vy_noise, main_noise[..., 1:2]), dim=-1)
        pre_tanh = mean + torch.exp(log_std) * noise
        normalized = torch.tanh(pre_tanh)
        log_prob = squashed_log_prob(pre_tanh, mean, log_std)
        target = p2_contract.map_normalized_action(normalized, hard_abs_vy=hard_abs_vy)
        return target, pre_tanh, normalized, log_prob, mean, log_std, hidden

    def deterministic(
        self, inputs, hidden=None, reset_mask=None, *, hard_abs_vy: float = 0.40
    ):
        mean, log_std, hidden = self.distribution_parameters(inputs, hidden, reset_mask)
        normalized = torch.tanh(mean)
        target = p2_contract.map_normalized_action(normalized, hard_abs_vy=hard_abs_vy)
        return target, normalized, mean, log_std, hidden

    def evaluate_actions(
        self,
        inputs,
        pre_tanh,
        hidden=None,
        reset_mask=None,
        *,
        return_features: bool = False,
    ):
        features, hidden = self._memory_forward(inputs, hidden, reset_mask)
        main_mean = self.mean_head(features)
        vy_mean = self.vy_mean_head(features)
        mean = torch.cat((main_mean[..., 0:1], vy_mean, main_mean[..., 1:2]), dim=-1)
        main_log_std = torch.clamp(
            self.log_std, p2_contract.LOG_STD_MIN, p2_contract.LOG_STD_MAX
        )
        vy_log_std = torch.clamp(
            self.vy_log_std, p2_contract.LOG_STD_MIN, p2_contract.LOG_STD_MAX
        )
        log_std = torch.cat(
            (main_log_std[0:1], vy_log_std, main_log_std[1:2]), dim=0
        ).expand_as(mean)
        log_prob = squashed_log_prob(pre_tanh, mean, log_std)
        # Monte-Carlo entropy estimate at the rollout action.
        entropy = -log_prob
        if return_features:
            return log_prob, entropy, mean, log_std, hidden, features
        return log_prob, entropy, mean, log_std, hidden


class P2NavigationCritic(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.encoder = nn.Sequential(
            nn.Linear(p2_contract.CRITIC_INPUT_DIM, 128),
            nn.SiLU(),
        )
        self.memory = nn.LSTM(128, self.hidden_dim, self.num_layers)
        self.value_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, inputs, hidden=None, reset_mask=None):
        encoded = self.encoder(inputs)
        squeeze = encoded.ndim == 2
        if squeeze:
            hidden = _apply_reset(hidden, reset_mask)
            output, hidden = self.memory(encoded.unsqueeze(0), hidden)
            return self.value_head(output.squeeze(0)), hidden
        if encoded.ndim != 3:
            raise ValueError("critic input must be [B,341] or [T,B,341]")
        outputs = []
        for step in range(encoded.shape[0]):
            step_reset = None if reset_mask is None else reset_mask[step]
            hidden = _apply_reset(hidden, step_reset)
            output, hidden = self.memory(encoded[step : step + 1], hidden)
            outputs.append(output)
        return self.value_head(torch.cat(outputs, dim=0)), hidden


def navigation_encoder_spec() -> dict[str, object]:
    return {
        "input_shape": [
            p2_contract.DEPTH_HEIGHT,
            p2_contract.DEPTH_WIDTH,
            p2_contract.DEPTH_CHANNELS,
        ],
        "output_dim": p2_contract.NAV_FEATURE_DIM,
        "architecture": "simple_cnn_v1",
    }


def navigation_safety_head_spec() -> dict[str, object]:
    return {
        "input_dim": p2_contract.NAV_FEATURE_DIM,
        "hidden_dim": p2_contract.NAV_FEATURE_DIM,
        "output_dim": 3,
        "output_layout": ["left_risk", "center_risk", "right_risk"],
        "training_only": True,
    }


def p4_actor_stuck_head_spec() -> dict[str, object]:
    return {
        "input_dim": 64,
        "output_dim": 1,
        "input_source": "actor_lstm_feature",
        "training_only": True,
    }


def navigation_actor_spec() -> dict[str, object]:
    return {
        "input_dim": p2_contract.ACTOR_INPUT_DIM,
        "hidden_dim": 64,
        "num_layers": 2,
        "action_dim": p2_contract.ACTION_DIM,
        "distribution": "diagonal_tanh_squashed_gaussian",
        "head_layout": {"mean_head": ["vx", "wz"], "vy_mean_head": ["vy"]},
        "physical_output": ["vx", "vy", "wz"],
    }


def navigation_critic_spec() -> dict[str, object]:
    return {
        "input_dim": p2_contract.CRITIC_INPUT_DIM,
        "encoder_dim": 128,
        "hidden_dim": 64,
        "num_layers": 2,
        "output_dim": 1,
    }


def assemble_actor_input(
    nav_feat32: torch.Tensor,
    nav_nonvisual36: torch.Tensor,
    response_profile16: torch.Tensor,
    confidence1: torch.Tensor,
) -> torch.Tensor:
    result = torch.cat(
        (nav_feat32, nav_nonvisual36, response_profile16, confidence1), dim=-1
    )
    if result.shape[-1] != p2_contract.ACTOR_INPUT_DIM:
        raise ValueError(f"P2 actor input layout drift: {tuple(result.shape)}")
    return result


def assemble_critic_input(
    critic_obs323: torch.Tensor,
    active_target_cmd3: torch.Tensor,
    capability15: torch.Tensor,
) -> torch.Tensor:
    if capability15.ndim == 1:
        capability15 = capability15.expand(critic_obs323.shape[0], -1)
    result = torch.cat((critic_obs323, active_target_cmd3, capability15), dim=-1)
    if result.shape[-1] != p2_contract.CRITIC_INPUT_DIM:
        raise ValueError(f"P2 critic input layout drift: {tuple(result.shape)}")
    return result
