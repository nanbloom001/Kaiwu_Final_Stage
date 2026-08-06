#!/usr/bin/env python3
"""Export the P3 checkpoint's standard low-level policy for fixed-mode Go2 tests.

The bundle contains both the standard locomotion policy and a track/high-level
navigation policy.  The fixed-mode runtime deliberately exports only the
low-level VisionEncoder plus Actor77Sequential and keeps the existing 8-in/8-out
LocoRunner interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Importing the existing wrapper keeps the Python graph byte-for-byte aligned
# with the C++ runner contract used by the other deployment packages.
from export_loco_onnx import (  # noqa: E402
    DEPTH_C,
    DEPTH_H,
    DEPTH_W,
    LATENT_DIM,
    NUM_ACTIONS,
    PROPRIO_DIM,
    TEACHER_ACTOR_IN,
    LocoExportWrapper,
    make_dummy,
    verify,
)


class P3LocoExportWrapper(LocoExportWrapper):
    """Keep passthrough nav inputs named ``nav_h``/``nav_c`` in ONNX.

    Newer torch exporters otherwise treat direct alias outputs as renamed
    graph inputs (``nav_h_out_orig``), which breaks the fixed C++ runner.
    """

    def forward(self, *args):
        outputs = super().forward(*args)
        nav_h, nav_c = args[5], args[6]
        return outputs[:6] + (nav_h * 1.0, nav_c * 1.0)


def _sub_state_dict(state_dict, prefix: str):
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def load_p3_weights(model: LocoExportWrapper, checkpoint: dict) -> dict:
    if checkpoint.get("format") != "kaiwu_train_v1":
        raise ValueError(f"unsupported checkpoint format: {checkpoint.get('format')!r}")
    if checkpoint.get("stage_type") != "p3_standard_joint":
        raise ValueError(f"expected p3_standard_joint, got {checkpoint.get('stage_type')!r}")

    spec = checkpoint.get("model_spec", {})
    expected_spec = {
        "proprio_dim": PROPRIO_DIM,
        "depth_height": DEPTH_H,
        "depth_width": DEPTH_W,
        "depth_channels": DEPTH_C,
        "latent_dim": LATENT_DIM,
        "action_dim": NUM_ACTIONS,
    }
    for key, expected in expected_spec.items():
        if spec.get(key) != expected:
            raise ValueError(f"model_spec.{key}={spec.get(key)!r}, expected {expected!r}")

    low_level = checkpoint.get("modules", {}).get("low_level", {})
    encoder = low_level.get("locomotion_encoder", {})
    actor = low_level.get("actor", {})
    if encoder.get("class_name") != "VisionEncoder":
        raise ValueError("low_level.locomotion_encoder is not VisionEncoder")
    if actor.get("class_name") != "Actor77Sequential":
        raise ValueError("low_level.actor is not Actor77Sequential")

    encoder_spec = encoder.get("spec", {})
    if tuple(encoder_spec.get("image_shape", ())) != (DEPTH_H, DEPTH_W, DEPTH_C):
        raise ValueError(f"unexpected encoder image_shape: {encoder_spec.get('image_shape')}")
    if encoder_spec.get("proprio_dim") != PROPRIO_DIM:
        raise ValueError("encoder proprio dimension does not match runner")
    if encoder_spec.get("rnn_output_dim") != LATENT_DIM:
        raise ValueError("encoder latent dimension does not match actor")

    actor_spec = actor.get("spec", {})
    if actor_spec.get("input_dim") != TEACHER_ACTOR_IN or actor_spec.get("output_dim") != NUM_ACTIONS:
        raise ValueError(f"unexpected actor spec: {actor_spec}")

    encoder_state = encoder.get("state_dict")
    actor_state = actor.get("state_dict")
    if not isinstance(encoder_state, dict) or not isinstance(actor_state, dict):
        raise KeyError("P3 checkpoint is missing low-level state_dict")
    model.cnn.load_state_dict(_sub_state_dict(encoder_state, "cnn."))
    model.rnn.load_state_dict(_sub_state_dict(encoder_state, "rnn."))
    model.rnn_output_layer.load_state_dict(_sub_state_dict(encoder_state, "rnn_output_layer."))
    model.teacher_actor.load_state_dict(actor_state)
    return {
        "stage_type": checkpoint["stage_type"],
        "platform_model_id": checkpoint.get("platform_model_id"),
        "phase_label": checkpoint.get("phase_label"),
        "source_sha256": None,
        "selected_modules": ["modules.low_level.locomotion_encoder", "modules.low_level.actor"],
        "ignored_modules": ["modules.high_level"],
        "model_spec": spec,
    }


def export_p3(model: LocoExportWrapper, out_path: Path, opset: int) -> None:
    """Export a single-file graph with the exact runner names and shapes."""
    model.eval()
    depth, proprio, goal, loco_h, loco_c, nav_h, nav_c = make_dummy()
    cmd_override = torch.zeros(1, 4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (depth, proprio, goal, loco_h, loco_c, nav_h, nav_c, cmd_override),
        str(out_path),
        input_names=["depth", "proprio", "goal", "loco_h", "loco_c", "nav_h", "nav_c", "cmd_override"],
        output_names=["cmd", "cmd_raw", "clearance", "joint", "loco_h_out", "loco_c_out", "nav_h_out", "nav_c_out"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
        external_data=False,
    )
    print(f"[OK] exported single-file ONNX: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint root must be dict, got {type(checkpoint).__name__}")

    model = P3LocoExportWrapper()
    manifest = load_p3_weights(model, checkpoint)
    manifest["source_sha256"] = hashlib.sha256(args.ckpt.read_bytes()).hexdigest()
    export_p3(model, args.out, opset=args.opset)
    if not args.no_verify and not verify(model, str(args.out), frames=args.frames):
        return 1

    manifest["onnx"] = str(args.out.name)
    manifest["fixed_mode"] = True
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        raise SystemExit(2)
