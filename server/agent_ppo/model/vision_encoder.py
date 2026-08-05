"""
Vision encoder (CNN + LSTM).
视觉编码器（CNN + LSTM）。

depth image → CNN → LSTM(吃 cnn_feat + proprio) → L2-normed latent。
深度图 → CNN → LSTM（接收 cnn_feat + proprio）→ L2 归一化 latent。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .simple_cnn import create_cnn_encoder


class VisionEncoder(nn.Module):
    """Vision encoder (CNN + LSTM).

    Architecture:
        depth image [H, W, C] → CNN → cnn_feat(32)
        [cnn_feat(32) | proprio(45)] → LSTM(2 layers, hidden=64)
                                     → Linear(64 → 32) → L2Norm → latent(32)

    LSTM 输入为 `cat(cnn_feat, proprio)`。

    Args:
        image_shape: (H, W, C)，默认 (180, 320, 1)
        proprio_dim: proprioception 维度（默认 45，拼接到 LSTM 输入）
        cnn_output_dim: CNN 输出维度（默认 32）
        rnn_hidden_dim: LSTM 隐层维度（默认 64）
        rnn_num_layers: LSTM 层数（默认 2）
        rnn_output_dim: 最终输出维度（默认 32, latent_dim）
        cnn_type: "simple" / "resnet"
        use_lstm: True 时用 LSTM；False 时用 MLP 替代
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (180, 320, 1),
        proprio_dim: int = 45,
        cnn_output_dim: int = 32,
        rnn_hidden_dim: int = 64,
        rnn_num_layers: int = 2,
        rnn_output_dim: int = 32,
        cnn_type: str = "simple",
        use_lstm: bool = True,
    ):
        super().__init__()

        self.image_shape = image_shape
        self.proprio_dim = proprio_dim
        self.cnn_output_dim = cnn_output_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.rnn_output_dim = rnn_output_dim
        self.use_lstm = use_lstm

        # CNN encoder
        self.cnn = create_cnn_encoder(
            encoder_type=cnn_type,
            input_shape=image_shape,
            output_dim=cnn_output_dim,
        )

        # LSTM input_size = cnn_output_dim + proprio_dim
        rnn_input_dim = cnn_output_dim + proprio_dim

        if use_lstm:
            self.rnn = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=rnn_hidden_dim,
                num_layers=rnn_num_layers,
                batch_first=True,
            )
            self.rnn_output_layer = nn.Linear(rnn_hidden_dim, rnn_output_dim)
            self._hidden_state = None
        else:
            self.mlp = nn.Sequential(
                nn.Linear(rnn_input_dim, rnn_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(rnn_hidden_dim, rnn_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(rnn_hidden_dim, rnn_output_dim),
            )

    def reset_hidden_state(self, batch_size=None, device=None):
        """Reset LSTM hidden state."""
        if not self.use_lstm:
            return
        if batch_size is not None and device is not None:
            self._hidden_state = (
                torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device),
                torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device),
            )
        else:
            self._hidden_state = None

    def reset_hidden_state_for_envs(self, env_ids: torch.Tensor):
        """Reset LSTM hidden state for specified env indices (typically done envs)."""
        if not self.use_lstm or self._hidden_state is None:
            return
        h, c = self._hidden_state
        h[:, env_ids, :] = 0
        c[:, env_ids, :] = 0

    def forward(
        self,
        depth_image: torch.Tensor,
        proprio: torch.Tensor,
        masks: torch.Tensor = None,
        detach_hidden: bool = True,
    ) -> torch.Tensor:
        """Forward pass: depth + proprio → CNN → LSTM → L2-normed latent.

        Args:
            depth_image: [B, H, W, C] or [B, C, H, W]  (CNN 内部会处理 layout)
            proprio:     [B, proprio_dim=45]  拼入 LSTM 输入
            masks:       [B,] bool, True=continue, False=reset LSTM hidden state for that env.

        Returns:
            latent: [B, rnn_output_dim], L2-normalized.
        """
        cnn_features = self.cnn(depth_image)  # [B, cnn_output_dim]
        return self.forward_from_cnn_features(
            cnn_features,
            proprio,
            masks=masks,
            detach_hidden=detach_hidden,
        )

    def forward_from_cnn_features(
        self,
        cnn_features: torch.Tensor,
        proprio: torch.Tensor,
        masks: torch.Tensor = None,
        detach_hidden: bool = True,
    ) -> torch.Tensor:
        """Replay the recurrent actor from frozen, precomputed CNN features."""
        if cnn_features.ndim != 2 or cnn_features.shape[-1] != self.cnn_output_dim:
            raise ValueError(
                f"CNN features must be [B,{self.cnn_output_dim}], got {tuple(cnn_features.shape)}"
            )
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"Proprio must be [B,{self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        batch_size = cnn_features.shape[0]
        device = cnn_features.device
        rnn_input = torch.cat([cnn_features, proprio], dim=-1)

        if self.use_lstm:
            # Add sequence dim [B, 1, input_dim]
            rnn_input = rnn_input.unsqueeze(1)

            # Initialize hidden state if needed
            if self._hidden_state is None:
                self._hidden_state = (
                    torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device),
                    torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device),
                )

            # Handle mask: reset hidden state for envs where masks is False
            if masks is not None:
                keep = masks.bool().reshape(1, batch_size, 1)
                self._hidden_state = (
                    self._hidden_state[0] * keep,
                    self._hidden_state[1] * keep,
                )

            rnn_output, self._hidden_state = self.rnn(rnn_input, self._hidden_state)

            # Rollout/inference detaches between environment steps. Recurrent
            # PPO sequence replay explicitly disables this so gradients flow
            # through one bounded TBPTT chunk.
            if detach_hidden:
                self._hidden_state = (
                    self._hidden_state[0].detach(),
                    self._hidden_state[1].detach(),
                )

            features = self.rnn_output_layer(rnn_output.squeeze(1))  # [B, rnn_output_dim]
        else:
            features = self.mlp(rnn_input)

        # L2-normalize：与 ActorCriticEncoder.encoder 末尾的 L2Norm 对齐，
        # teacher_actor 的输入分布假设 latent ∈ 单位球。
        features = F.normalize(features, p=2.0, dim=-1)

        return features

    def get_output_dim(self) -> int:
        return self.rnn_output_dim

    def get_hidden_state(self):
        return self._hidden_state

    def set_hidden_state(self, hidden_state: Tuple[torch.Tensor, torch.Tensor]):
        self._hidden_state = hidden_state


class DmEncoder(nn.Module):
    """Privileged-information encoder (teacher network).

    Encodes privileged height_map into a 32-dim L2-normed latent.
    Used for teacher-student distillation.

    Architecture:
        scan [input_dim] → MLP [512, 256, output_dim] → L2Norm → [output_dim]
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (512, 256),
        output_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ELU(inplace=True)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, scan: torch.Tensor) -> torch.Tensor:
        features = self.mlp(scan)
        # L2-normalize：与 ActorCriticEncoder.encoder 末尾的 L2Norm 对齐
        features = F.normalize(features, p=2.0, dim=-1)
        return features

    def get_output_dim(self) -> int:
        return self.output_dim


class CNNRNN(nn.Module):
    """Thin compatibility wrapper around VisionEncoder.

    forward() signature is (depth_image, proprio, masks)。
    """

    def __init__(
        self,
        image_size: Tuple[int, int, int] = (180, 320, 1),
        proprio_dim: int = 45,
        cnn_out_size: int = 32,
        rnn_hidden_size: int = 64,
        rnn_out_size: int = 32,
        rnn_layers: int = 2,
        no_rnn: bool = False,
    ):
        super().__init__()
        self.encoder = VisionEncoder(
            image_shape=image_size,
            proprio_dim=proprio_dim,
            cnn_output_dim=cnn_out_size,
            rnn_hidden_dim=rnn_hidden_size,
            rnn_num_layers=rnn_layers,
            rnn_output_dim=rnn_out_size,
            use_lstm=not no_rnn,
        )

    def forward(
        self,
        depth_image: torch.Tensor,
        proprio: torch.Tensor,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.encoder(depth_image, proprio, masks)

    def reset_hidden_state(self, batch_size=None, device=None):
        self.encoder.reset_hidden_state(batch_size, device)

    def reset_hidden_state_for_envs(self, env_ids: torch.Tensor):
        self.encoder.reset_hidden_state_for_envs(env_ids)

    def get_output_dim(self) -> int:
        return self.encoder.get_output_dim()
