#!/usr/bin/env python3
"""Validate the shipped ONNX against the 884257 PyTorch low-level modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export_loco_onnx import verify  # noqa: E402
from export_p3_standard_onnx import P3LocoExportWrapper, load_p3_weights  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    checkpoint_path = ROOT / "models/model.ckpt-highslow-884257.pkl"
    onnx_path = ROOT / "runtime/unitree_rl_lab_test/logs/loco/exported/policy.onnx"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = P3LocoExportWrapper()
    load_p3_weights(model, checkpoint)
    model.eval()
    if not verify(model, str(onnx_path), frames=args.frames):
        return 1
    print(f"PARITY PASSED: frames={args.frames} checkpoint={checkpoint_path.name} onnx={onnx_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
