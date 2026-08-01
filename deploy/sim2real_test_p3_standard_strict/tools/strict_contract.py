#!/usr/bin/env python3
"""Machine-checkable safety primitives for the strict P3 low-level package."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(root: Path) -> dict:
    with (root / "artifact_contract.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_hashes(root: Path, contract: dict) -> dict[str, str]:
    checked: dict[str, str] = {}
    for key in ("checkpoint", "onnx"):
        spec = contract[key]
        path = root / spec["path"]
        if not path.is_file():
            raise ContractError(f"missing {key}: {path}")
        actual = sha256_file(path)
        if actual != spec["sha256"]:
            raise ContractError(
                f"{key} SHA256 mismatch: expected {spec['sha256']}, got {actual}"
            )
        checked[str(path.relative_to(root))] = actual
    return checked


def _onnx_dtype_name(elem_type: int) -> str:
    import onnx

    name = onnx.TensorProto.DataType.Name(elem_type).lower()
    return {"float": "float32", "double": "float64"}.get(name, name)


def validate_onnx(root: Path, contract: dict) -> dict:
    import onnx

    path = root / contract["onnx"]["path"]
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    expected = contract["onnx"]
    if model.ir_version != expected["ir_version"]:
        raise ContractError(
            f"ONNX IR mismatch: expected {expected['ir_version']}, got {model.ir_version}"
        )
    opsets = {item.domain: item.version for item in model.opset_import}
    if opsets.get("") != expected["opset"]:
        raise ContractError(
            f"ONNX opset mismatch: expected {expected['opset']}, got {opsets.get('')}"
        )

    def check_ports(actual_ports, expected_ports, kind: str) -> list[dict]:
        if len(actual_ports) != len(expected_ports):
            raise ContractError(
                f"ONNX {kind} count mismatch: expected {len(expected_ports)}, got {len(actual_ports)}"
            )
        actual_names = [port.name for port in actual_ports]
        expected_names = [port["name"] for port in expected_ports]
        if set(actual_names) != set(expected_names):
            raise ContractError(
                f"ONNX {kind} names mismatch: expected {expected_names}, got {actual_names}"
            )
        by_name = {port.name: port for port in actual_ports}
        summary = []
        for spec in expected_ports:
            tensor = by_name[spec["name"]].type.tensor_type
            dtype = _onnx_dtype_name(tensor.elem_type)
            shape = [dim.dim_value if dim.HasField("dim_value") else None for dim in tensor.shape.dim]
            if dtype != spec["dtype"]:
                raise ContractError(
                    f"ONNX {kind} {spec['name']} dtype mismatch: expected {spec['dtype']}, got {dtype}"
                )
            if len(shape) != spec["rank"] or shape != spec["shape"]:
                raise ContractError(
                    f"ONNX {kind} {spec['name']} shape mismatch: expected {spec['shape']}, got {shape}"
                )
            summary.append({"name": spec["name"], "dtype": dtype, "shape": shape})
        return summary

    inputs = check_ports(model.graph.input, expected["inputs"], "input")
    outputs = check_ports(model.graph.output, expected["outputs"], "output")
    return {"ir_version": model.ir_version, "opset": opsets[""], "inputs": inputs, "outputs": outputs}


@dataclass(frozen=True)
class ActionResult:
    model_raw_action: np.ndarray
    clipped_raw_action: np.ndarray
    requested_joint_target: np.ndarray
    slew_limited_joint_target: np.ndarray
    physical_limit_joint_target: np.ndarray
    applied_joint_target: np.ndarray
    executed_raw_action: np.ndarray
    safety_modified: np.ndarray


def apply_action_chain(
    model_raw_action: Sequence[float],
    previous_applied_target: Sequence[float],
    contract: dict,
) -> ActionResult:
    spec = contract["action"]
    raw = np.asarray(model_raw_action, dtype=np.float64)
    previous = np.asarray(previous_applied_target, dtype=np.float64)
    if raw.shape != (spec["dim"],) or previous.shape != (spec["dim"],):
        raise ContractError("action and previous target must both contain 12 values")
    if not np.isfinite(raw).all() or not np.isfinite(previous).all():
        raise ContractError("non-finite action chain input")
    clip_lo, clip_hi = spec["training_effective_raw_action_clip"]
    clipped = np.clip(raw, clip_lo, clip_hi)
    offset = np.asarray(spec["offset"], dtype=np.float64)
    requested = offset + float(spec["scale"]) * clipped
    max_step = float(spec["target_slew_rate_rad_s"]) * float(contract["control"]["step_dt"])
    slew = previous + np.clip(requested - previous, -max_step, max_step)
    limits = np.asarray(spec["physical_limits_rad"], dtype=np.float64)
    physical = np.clip(slew, limits[:, 0], limits[:, 1])
    applied = physical.copy()
    executed = (applied - offset) / float(spec["scale"])
    modified = np.abs(applied - requested) > 1e-7
    return ActionResult(raw, clipped, requested, slew, physical, applied, executed, modified)


@dataclass
class WatchdogState:
    faulted: bool = False
    reason: str = "none"
    frozen: bool = False
    takeover: str = "none"

    def evaluate(self, ages_ms: dict[str, float], contract: dict) -> "WatchdogState":
        limits = contract["watchdog_ms"]
        checks = (
            ("lowstate", "lowstate_stale", "Passive"),
            ("policy_target", "policy_target_stale", "FixStand"),
            ("command", "command_stale", "FixStand"),
        )
        for key, reason, takeover in checks:
            value = ages_ms.get(key, math.inf)
            if not math.isfinite(value) or value > limits[key]:
                self.faulted = True
                self.reason = reason
                self.frozen = True
                self.takeover = takeover
                return self
        inference = ages_ms.get("inference", 0.0)
        if not math.isfinite(inference) or inference > limits["inference_deadline"]:
            self.faulted = True
            self.reason = "inference_deadline_miss"
            self.frozen = True
            self.takeover = "FixStand"
        return self


def normalize_depth(values_m: Iterable[float]) -> np.ndarray:
    depth = np.asarray(list(values_m), dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < 5.0)
    return np.where(valid, depth / 5.0, 0.0).astype(np.float32)


def band_energy(values: Sequence[float], sample_hz: float = 50.0) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 4:
        raise ContractError("band energy requires at least four scalar samples")
    centered = data - data.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2 / data.size
    freq = np.fft.rfftfreq(data.size, d=1.0 / sample_hz)
    result = {}
    for name, lo, hi in (("5_10_hz", 5.0, 10.0), ("10_15_hz", 10.0, 15.0), ("15_25_hz", 15.0, 25.0)):
        mask = (freq >= lo) & (freq < hi if hi < sample_hz / 2 else freq <= hi)
        result[name] = float(power[mask].sum())
    delta = np.diff(data)
    signs = np.sign(delta[np.abs(delta) > 1e-9])
    result["alternating_sign_ratio"] = (
        float(np.mean(signs[1:] != signs[:-1])) if signs.size > 1 else 0.0
    )
    result["delta_rms"] = float(np.sqrt(np.mean(delta * delta)))
    return result
