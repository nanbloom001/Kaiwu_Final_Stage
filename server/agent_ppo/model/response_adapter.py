#!/usr/bin/env python3
"""Physical-semantic command response adapter for P1.5/P2."""

from __future__ import annotations

import torch
import torch.nn as nn

from agent_ppo.feature import p15_contract


class CommandResponseAdapter(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.LayerNorm(self.input_dim + 3)
        self.input_mlp = nn.Sequential(
            nn.Linear(self.input_dim + 3, self.hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.prediction_head = nn.Linear(
            self.hidden_dim, p15_contract.RESPONSE_PROFILE_DIM
        )

    def forward(self, observation: torch.Tensor, hidden=None, reset_mask=None):
        squeeze_time = observation.ndim == 2
        if squeeze_time:
            observation = observation.unsqueeze(0)
        if observation.ndim != 3 or observation.shape[-1] != self.input_dim:
            raise ValueError(
                f"response observation must be [T,B,{self.input_dim}] or "
                f"[B,{self.input_dim}], got {tuple(observation.shape)}"
            )
        tracking_error = observation[..., 3:6] - observation[..., 6:9]
        encoded = self.input_mlp(
            self.input_norm(torch.cat((observation, tracking_error), dim=-1))
        )
        if reset_mask is None:
            recurrent, hidden_out = self.gru(encoded, hidden)
        else:
            reset_mask = reset_mask.to(observation.device).bool()
            if reset_mask.shape != observation.shape[:2]:
                raise ValueError(
                    f"response reset mask must be {tuple(observation.shape[:2])}, "
                    f"got {tuple(reset_mask.shape)}"
                )
            hidden_out = hidden
            outputs = []
            for step in range(encoded.shape[0]):
                if hidden_out is not None and bool(reset_mask[step].any()):
                    hidden_out = hidden_out.clone()
                    hidden_out[:, reset_mask[step], :] = 0.0
                recurrent_step, hidden_out = self.gru(
                    encoded[step : step + 1], hidden_out
                )
                outputs.append(recurrent_step)
            recurrent = torch.cat(outputs, dim=0)
        raw_profile = self.prediction_head(self.output_norm(recurrent))
        profile = torch.cat(
            (raw_profile[..., :13], torch.clamp(raw_profile[..., 13:16], -4.0, 1.0)),
            dim=-1,
        )
        if squeeze_time:
            profile = profile.squeeze(0)
        return profile, hidden_out

    @staticmethod
    def split_profile(profile: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "velocity": profile[..., 0:9].reshape(*profile.shape[:-1], 3, 3),
            "pose_delta": profile[..., 9:12],
            "stuck_logit": profile[..., 12:13],
            "velocity_log_sigma": torch.clamp(profile[..., 13:16], -4.0, 1.0),
        }


def response_adapter_spec() -> dict[str, object]:
    return {
        "class_name": "CommandResponseAdapter",
        "input_dim": p15_contract.RESPONSE_OBSERVATION_DIM,
        "hidden_dim": 64,
        "profile_dim": p15_contract.RESPONSE_PROFILE_DIM,
        "velocity_horizons_s": [0.2, 0.6, 1.0],
        "log_sigma_clamp": [-4.0, 1.0],
    }
