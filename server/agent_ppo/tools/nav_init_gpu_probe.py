#!/usr/bin/env python3
"""Reproduce hier-nav construction on the development-container GPU."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from agent_ppo.algorithm.algorithm_nav_dagger import AlgorithmNavDagger
from agent_ppo.feature import nav_contract
from agent_ppo.model.high_level_policy import HighLevelPolicy
from agent_ppo.model.vision_encoder import VisionEncoder


class _Logger:
    def info(self, message) -> None:
        print(message, flush=True)

    def warning(self, message) -> None:
        print(f"WARNING {message}", flush=True)


def _cuda_state(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cuda=unavailable"
    return (
        f"allocated={torch.cuda.memory_allocated(device)} "
        f"reserved={torch.cuda.memory_reserved(device)}"
    )


def _timed(label: str, device: torch.device, operation):
    print(f"[GpuProbe] {label} begin {_cuda_state(device)}", flush=True)
    started = time.monotonic()
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(
        f"[GpuProbe] {label} complete elapsed_s={time.monotonic() - started:.3f} "
        f"{_cuda_state(device)}",
        flush=True,
    )
    return result


def _low_level_actor() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(77, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 12),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(
        f"[GpuProbe] torch={torch.__version__} device={device} "
        f"cuda_available={torch.cuda.is_available()}",
        flush=True,
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    vision = _timed(
        "vision_construct_cpu",
        device,
        lambda: VisionEncoder(
            image_shape=(180, 320, 1),
            proprio_dim=45,
            cnn_output_dim=32,
            rnn_hidden_dim=64,
            rnn_num_layers=2,
            rnn_output_dim=32,
            use_lstm=True,
        ),
    )
    actor = _timed("actor_construct_cpu", device, _low_level_actor)
    high_level = _timed(
        "high_level_construct_cpu",
        device,
        lambda: HighLevelPolicy(
            input_dim=nav_contract.NAV_INPUT_DIM,
            vocab_size=nav_contract.VOCAB_SIZE,
            rnn_hidden_dim=nav_contract.NAV_LSTM_HIDDEN_SIZE,
            rnn_num_layers=nav_contract.NAV_LSTM_NUM_LAYERS,
        ),
    )

    vision = _timed("vision_to_device", device, lambda: vision.to(device))
    actor = _timed("actor_to_device", device, lambda: actor.to(device))
    high_level = _timed(
        "high_level_to_device", device, lambda: high_level.to(device)
    )

    algorithm = _timed(
        "algorithm_construct_from_cuda_modules",
        device,
        lambda: AlgorithmNavDagger(
            vision_encoder=vision,
            low_level_actor=actor,
            high_level=high_level,
            device=str(device),
            learning_rate=3e-4,
            max_grad_norm=1.0,
            proprio_dim=45,
            scan_dim=256,
            depth_shape=(180, 320, 1),
            low_level_parent_model_id="34728",
            logger=_Logger(),
            process_role="rpc_gpu_probe",
        ),
    )
    print(
        f"[GpuProbe] success optimizer_groups={len(algorithm.optimizer.param_groups)} "
        f"{_cuda_state(device)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
