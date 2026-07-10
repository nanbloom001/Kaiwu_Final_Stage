#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

SimpleCNN depth image encoder.
SimpleCNN 深度图编码器。

CNN encoder that compresses depth images into low-dimensional feature vectors.
CNN 编码器，用于将深度图压缩为低维特征向量。

Input : depth image [batch, height, width, channel] = [batch, 180, 320, 1]
Output: feature vector [batch, 32]
输入: 深度图 [batch, height, width, channel] = [batch, 180, 320, 1]
输出: 特征向量 [batch, 32]

Compression ratio / 压缩比: 180 × 320 × 1 = 57600 → 32 ≈ 1800:1
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class SimpleCNN(nn.Module):
    """Simple CNN depth image encoder.
    简单的 CNN 深度图编码器。

    Encodes depth images into low-dimensional feature vectors.
    将深度图编码为低维特征向量。

    Network structure / 网络结构:
        Conv2d(1, 32, 5, stride=2) → ReLU → MaxPool2d(2)
        → Conv2d(32, 64, 3, stride=2) → ReLU → MaxPool2d(2)
        → Conv2d(64, 128, 3, stride=1) → ReLU
        → Flatten → Linear(hidden, output_dim)

    Args:
        input_shape: Input image shape [height, width, channels].
                     输入图像形状 [height, width, channels]。
        output_dim: Output feature dimension (default 32).
                    输出特征维度 (默认 32)。
        hidden_channels: List of conv layer channels (default [32, 64, 128]).
                         卷积层通道数列表 (默认 [32, 64, 128])。
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (180, 320, 1),
        output_dim: int = 32,
        hidden_channels: Tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()

        self.input_shape = input_shape
        self.output_dim = output_dim

        height, width, in_channels = input_shape

        # Build conv layers
        # 构建卷积层
        self.conv_layers = nn.ModuleList()

        # Conv1: 1 → 32, kernel=5, stride=2, + MaxPool2d(2)
        # Output size / 输出尺寸: (180-5)/2+1 = 88, (320-5)/2+1 = 158
        # After MaxPool / 经过 MaxPool: 88/2 = 44, 158/2 = 79
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels[0], kernel_size=5, stride=2, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        )

        # Conv2: 32 → 64, kernel=3, stride=2, + MaxPool2d(2)
        # Output size / 输出尺寸: (44-3)/2+1 = 21, (79-3)/2+1 = 39
        # After MaxPool / 经过 MaxPool: 21/2 = 10, 39/2 = 19
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, stride=2, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        )

        # Conv3: 64 → 128, kernel=3, stride=1 (no pooling)
        # Output size / 输出尺寸: (10-3)/1+1 = 8, (19-3)/1+1 = 17
        # Conv3: 64 → 128, kernel=3, stride=1 (无池化)
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )
        )

        # Compute flattened conv output size
        # 计算卷积层输出尺寸
        self._conv_output_size = self._compute_conv_output_size(height, width)

        # Fully connected layer
        # 全连接层
        self.fc = nn.Linear(self._conv_output_size, output_dim)

    def _compute_conv_output_size(self, height: int, width: int) -> int:
        """Compute the flattened output size of the conv layers.

        计算卷积层输出的展平尺寸。
        """
        # Conv1 + MaxPool
        # 卷积层1 + 最大池化
        # stride=2
        h = (height - 5) // 2 + 1
        w = (width - 5) // 2 + 1
        # MaxPool
        # 最大池化
        h = h // 2
        w = w // 2

        # Conv2 + MaxPool
        # 卷积层2 + 最大池化
        # stride=2
        h = (h - 3) // 2 + 1
        w = (w - 3) // 2 + 1
        # MaxPool
        # 最大池化
        h = h // 2
        w = w // 2

        # Conv3 (no pool)
        # 卷积层3（无池化）
        # stride=1
        h = (h - 3) // 1 + 1
        w = (w - 3) // 1 + 1

        # Last channels × height × width
        # 最后一层通道数 × 高度 × 宽度
        return 128 * h * w

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        前向传播。

        Args:
            depth_image: Depth image tensor.
                - Shape [batch, height, width, channels] (NHWC)
                - or shape [batch, channels, height, width] (NCHW)
                深度图张量。
                - 形状 [batch, height, width, channels] (NHWC)
                - 或形状 [batch, channels, height, width] (NCHW)

        Returns:
            features: Feature vector [batch, output_dim].
                      特征向量 [batch, output_dim]。
        """
        # Ensure input is NCHW format
        # 确保输入是 NCHW 格式
        if depth_image.dim() == 4:
            if depth_image.shape[-1] == self.input_shape[2]:
                # NHWC → NCHW
                # NHWC 格式转 NCHW 格式
                depth_image = depth_image.permute(0, 3, 1, 2)
        elif depth_image.dim() == 3:
            # [batch, height, width] → [batch, 1, height, width]
            # [batch, height, width] → [batch, 1, height, width]（添加通道维）
            depth_image = depth_image.unsqueeze(1)

        # Conv layers
        # 卷积层
        x = depth_image
        for conv in self.conv_layers:
            x = conv(x)

        # Flatten
        # 展平
        x = x.reshape(x.size(0), -1)

        # Fully connected layer
        # 全连接层
        features = self.fc(x)

        return features

    def get_output_dim(self) -> int:
        """Get output dimension.

        获取输出维度。
        """
        return self.output_dim


class ResNetEncoder(nn.Module):
    """ResNet-based depth image encoder.
    基于 ResNet 的深度图编码器。

    Used for scenarios requiring stronger feature extraction (e.g. large Sim2Real gap).
    Uses the first few layers of ResNet-18 + custom head.
    用于需要更强特征提取能力的场景（如 Sim2Real gap 较大时）。
    使用 ResNet-18 前几层 + 自定义头。

    Args:
        input_shape: Input image shape [height, width, channels].
                     输入图像形状 [height, width, channels]。
        output_dim: Output feature dimension (default 32).
                    输出特征维度 (默认 32)。
        pretrained: Whether to use pretrained weights (default False).
                    是否使用预训练权重 (默认 False)。
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (180, 320, 1),
        output_dim: int = 32,
        pretrained: bool = False,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.output_dim = output_dim

        # Input channel conversion (1 → 3 if using pretrained)
        # 输入通道转换 (1 → 3 如果使用预训练)
        in_channels = input_shape[2]
        if pretrained and in_channels != 3:
            self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=1)
            in_channels = 3
        else:
            self.input_conv = None

        # Simplified ResNet backbone
        # 简化的 ResNet 骨干网络
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ResNet Block 1
        # ResNet 模块1
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)

        # ResNet Block 2
        # ResNet 模块2
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)

        # Adaptive pooling + fully connected
        # 自适应池化 + 全连接
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, output_dim)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build a ResNet layer.

        构建 ResNet 层。
        """
        layers = []

        # First block may need downsampling
        # 第一个 block 可能需要下采样
        layers.append(BasicBlock(in_channels, out_channels, stride))

        # Subsequent blocks
        # 后续 block
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        前向传播。
        """
        # Ensure input is NCHW format
        # 确保输入是 NCHW 格式
        if depth_image.dim() == 4:
            if depth_image.shape[-1] == self.input_shape[2]:
                depth_image = depth_image.permute(0, 3, 1, 2)
        elif depth_image.dim() == 3:
            depth_image = depth_image.unsqueeze(1)

        # Input channel conversion
        # 输入通道转换
        if self.input_conv is not None:
            depth_image = self.input_conv(depth_image)

        # Backbone network
        # 骨干网络
        x = self.conv1(depth_image)
        x = self.layer1(x)
        x = self.layer2(x)

        # Pooling and FC
        # 池化和全连接
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)

        return features

    def get_output_dim(self) -> int:
        return self.output_dim


class BasicBlock(nn.Module):
    """ResNet basic block.

    ResNet 基础块。
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample
        # 下采样
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def create_cnn_encoder(
    encoder_type: str = "simple",
    input_shape: Tuple[int, int, int] = (180, 320, 1),
    output_dim: int = 32,
    **kwargs,
) -> nn.Module:
    """Factory function to create a CNN encoder.
    创建 CNN 编码器工厂函数。

    Args:
        encoder_type: Encoder type ("simple" or "resnet").
                      编码器类型 ("simple" 或 "resnet")。
        input_shape: Input image shape.
                     输入图像形状。
        output_dim: Output feature dimension.
                    输出特征维度。
        **kwargs: Extra arguments passed to encoder.
                  额外参数传递给编码器。

    Returns:
        encoder: CNN encoder instance.
                 CNN 编码器实例。
    """
    if encoder_type == "simple":
        return SimpleCNN(input_shape, output_dim, **kwargs)
    elif encoder_type == "resnet":
        return ResNetEncoder(input_shape, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
