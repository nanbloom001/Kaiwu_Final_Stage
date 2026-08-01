#!/usr/bin/env python3
"""Strict, non-motor preflight and ONNX ABI validator."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from strict_contract import ContractError, load_contract, sha256_file, validate_onnx, verify_hashes


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime/unitree_rl_lab_test"
CONFIG = RUNTIME / "deploy/robots/go2_loco/config/config.yaml"
DEPLOY = RUNTIME / "logs/loco/params/deploy.yaml"


def check_config(contract: dict) -> dict:
    with CONFIG.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with DEPLOY.open(encoding="utf-8") as handle:
        deploy = yaml.safe_load(handle)
    vision = config["FSM"]["VisionLoco"]
    fixed = [float(value) for value in vision["fixed_cmd"]]
    if fixed != contract["safety"]["fixed_command_default"]:
        raise ContractError(f"fixed command must default to zero, got {fixed}")
    if deploy["actions"]["JointPositionAction"]["clip"] != contract["action"]["training_effective_raw_action_clip"]:
        raise ContractError("deploy.yaml raw action clip does not match contract")
    if float(deploy["actions"]["JointPositionAction"]["scale"]) != contract["action"]["scale"]:
        raise ContractError("deploy.yaml action scale does not match contract")
    depth = vision["depth"]
    acquisition = contract["depth"]["acquisition"]
    if list(depth["preferred_profile"]) != list(acquisition["preferred_profile"]):
        raise ContractError("preferred raw depth profile does not match contract")
    if list(depth["allowed_fallback_profiles"]) != list(acquisition["allowed_fallback_profiles"]):
        raise ContractError("allowed raw depth fallback profiles do not match contract")
    if depth.get("require_allowed_profile_for_arm") is not True:
        raise ContractError("strict powered mode must require an audited raw depth profile")
    config_sha = sha256_file(CONFIG)
    deploy_sha = sha256_file(DEPLOY)
    expected = contract.get("configuration", {})
    if expected:
        if config_sha != expected["controller"]["sha256"]:
            raise ContractError("controller config SHA256 mismatch")
        if deploy_sha != expected["deploy"]["sha256"]:
            raise ContractError("deploy config SHA256 mismatch")
    return {
        "config_sha256": config_sha,
        "deploy_sha256": deploy_sha,
        "fixed_cmd": fixed,
        "depth_source": depth["source"],
        "preferred_depth_profile": depth["preferred_profile"],
        "allowed_depth_fallback_profiles": depth["allowed_fallback_profiles"],
    }


def camera_probe(required: bool) -> dict:
    tool = shutil.which("rs-enumerate-devices")
    if not tool:
        if required:
            raise ContractError("rs-enumerate-devices is unavailable")
        return {"status": "unavailable", "required": False}
    run = subprocess.run([tool, "-s"], text=True, capture_output=True, timeout=10, check=False)
    found = run.returncode == 0 and "RealSense" in (run.stdout + run.stderr)
    if required and not found:
        raise ContractError("expected Intel RealSense device was not detected")
    return {"status": "detected" if found else "not_detected", "required": required}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--require-camera", action="store_true")
    args = parser.parse_args()
    try:
        contract = load_contract(ROOT)
        if contract.get("deployable") is not False or contract.get("experimental_low_level_override") is not True:
            raise ContractError("strict package must remain experimental and non-deployable")
        hashes = verify_hashes(ROOT, contract)
        abi = validate_onnx(ROOT, contract)
        config = check_config(contract)
        if args.require_camera and config["depth_source"] != "realsense":
            raise ContractError("powered preflight requires depth.source=realsense")
        camera = camera_probe(args.require_camera)
        result = {
            "ok": True,
            "mode": "preflight",
            "motor_command_channel_created": False,
            "contract_sha256": sha256_file(ROOT / "artifact_contract.json"),
            "hashes": hashes,
            "abi": abi,
            "config": config,
            "camera": camera,
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "onnxruntime": __import__("onnxruntime").__version__,
            },
        }
    except (ContractError, OSError, ValueError, KeyError, ImportError) as exc:
        result = {"ok": False, "mode": "preflight", "motor_command_channel_created": False, "error": str(exc)}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print("STRICT PREFLIGHT PASSED")
        print(f"contract_sha256={result['contract_sha256']}")
        print(f"onnx_sha256={result['hashes'][contract['onnx']['path']]}")
        print("ABI: 8 exact float32 inputs, 8 exact float32 outputs, opset 18, IR 10")
        print("No LowCmd channel was created. No motor command was sent.")
    else:
        print(f"STRICT PREFLIGHT FAILED: {result['error']}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
