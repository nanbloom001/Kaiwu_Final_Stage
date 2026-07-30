#!/usr/bin/env python3
"""Validate non-negotiable startup and pose-continuity invariants."""

import math
import json
import sys
from pathlib import Path
from typing import List

import yaml


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"safety configuration check failed: {message}")


def finite_vector(value, size: int, label: str) -> List[float]:
    require(isinstance(value, list) and len(value) == size,
            f"{label} must contain {size} values")
    result = [float(item) for item in value]
    require(all(math.isfinite(item) for item in result),
            f"{label} contains a non-finite value")
    return result


def load_yaml(path: Path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    require(len(sys.argv) in (3, 4),
            "usage: check_safety_config.py <config.yaml> <deploy.yaml> "
            "[--suspended-parity|--harness-guard]")
    diagnostic_flag = sys.argv[3] if len(sys.argv) == 4 else ""
    require(diagnostic_flag in ("", "--suspended-parity", "--harness-guard"),
            "unsupported diagnostic safety flag")
    suspended_parity = diagnostic_flag == "--suspended-parity"
    harness_guard_mode = diagnostic_flag == "--harness-guard"
    config_path = Path(sys.argv[1])
    deploy_path = Path(sys.argv[2])
    config = load_yaml(config_path)
    deploy = load_yaml(deploy_path)

    fsm = config.get("FSM", {})
    require(fsm.get("initial_state") == "Passive",
            "FSM.initial_state must be Passive")
    enabled = fsm.get("_", {})
    require(isinstance(enabled, dict) and "Passive" in enabled,
            "Passive must be enabled")
    ids = [entry.get("id") for entry in enabled.values()]
    require(all(isinstance(state_id, int) and state_id > 0 for state_id in ids),
            "enabled state IDs must be positive integers")
    require(len(ids) == len(set(ids)), "enabled state IDs must be unique")
    global_safety = fsm.get("safety", {})
    global_tilt_deg = float(global_safety.get("max_tilt_deg", math.inf))
    require(math.isfinite(global_tilt_deg) and 10.0 <= global_tilt_deg <= 25.0,
            "FSM.safety.max_tilt_deg must be in [10, 25]")

    passive = fsm.get("Passive", {})
    fix_stand = fsm.get("FixStand", {})
    vision_loco = fsm.get("VisionLoco", {})
    require(passive.get("transitions", {}).get("FixStand") == "LT + A.on_pressed",
            "Passive must require LT+A to enter FixStand")
    require(fix_stand.get("transitions", {}).get("Passive") == "LT + B.on_pressed",
            "FixStand must retain the LT+B Passive transition")
    entry_blend_s = float(vision_loco.get("entry_blend_s", 0.0))
    require(math.isfinite(entry_blend_s) and 0.5 <= entry_blend_s <= 3.0,
            "VisionLoco.entry_blend_s must be in [0.5, 3.0]")
    ready = vision_loco.get("ready_stand", {})
    ready_gain_blend = float(ready.get("gain_blend_s", math.nan))
    ready_min_stable = float(ready.get("min_stable_s", math.nan))
    ready_max_wait = float(ready.get("max_wait_s", math.nan))
    ready_max_dq = float(ready.get("max_joint_velocity", math.nan))
    ready_max_tracking = float(ready.get("max_tracking_error_rad", math.nan))
    ready_max_support_offset = float(
        ready.get("max_support_target_offset_rad", math.nan))
    require(math.isfinite(ready_gain_blend) and 0.5 <= ready_gain_blend <= 3.0,
            "ready_stand.gain_blend_s must be in [0.5, 3.0]")
    require(math.isfinite(ready_min_stable) and 0.2 <= ready_min_stable <= 2.0,
            "ready_stand.min_stable_s must be in [0.2, 2.0]")
    require(math.isfinite(ready_max_wait) and
            ready_gain_blend + ready_min_stable <= ready_max_wait <= 10.0,
            "ready_stand.max_wait_s must cover gain blend and stability dwell")
    require(math.isfinite(ready_max_dq) and 0.0 < ready_max_dq <= 1.0,
            "ready_stand.max_joint_velocity must be in (0, 1]")
    require(math.isfinite(ready_max_tracking) and 0.0 < ready_max_tracking <= 0.5,
            "ready_stand.max_tracking_error_rad must be in (0, 0.5]")
    require(math.isfinite(ready_max_support_offset) and
            0.0 < ready_max_support_offset <= 0.3,
            "ready_stand.max_support_target_offset_rad must be in (0, 0.3]")

    keyboard = vision_loco.get("keyboard", {})
    key_step_vx = float(keyboard.get("step_vx", math.inf))
    key_max_vx = float(keyboard.get("max_vx", math.inf))
    key_max_wz = float(keyboard.get("max_wz", math.inf))
    key_max_nonzero = float(keyboard.get("max_nonzero_s", math.nan))
    require(math.isfinite(key_max_vx) and 0.0 < key_max_vx <= 0.2,
            "keyboard.max_vx must be in (0, 0.2] for ground testing")
    require(math.isfinite(key_max_wz) and 0.0 < key_max_wz <= 0.2,
            "keyboard.max_wz must be in (0, 0.2] for ground testing")
    require(math.isfinite(key_max_nonzero) and 0.0 <= key_max_nonzero <= 10.0,
            "keyboard.max_nonzero_s must be in [0, 10]")
    if suspended_parity or harness_guard_mode:
        require(vision_loco.get("command_source") == "keyboard",
                "transparent diagnostics require keyboard command source")
        require(math.isfinite(key_step_vx) and abs(key_step_vx - 0.15) <= 1.0e-9,
                "transparent diagnostics require one W step of 0.15 m/s")
        require(key_max_vx <= 0.15 and key_max_wz <= 0.10,
                "transparent diagnostic caps must not exceed [0.15, 0.10]")
        require(abs(key_max_nonzero - 2.0) <= 1.0e-9,
                "transparent diagnostics require a 2.0 second hard timeout")
    if harness_guard_mode:
        depth = vision_loco.get("depth", {})
        require(depth.get("source") == "realsense",
                "harness guard requires RealSense depth")
        require(depth.get("filters", {}).get("mode") == "none",
                "harness guard requires unfiltered depth")
        guard = vision_loco.get("harness_guard", {})
        profile_path = config_path.parent / "transparent_guard_candidate.json"
        require(profile_path.is_file(),
                "transparent guard candidate profile is missing")
        with profile_path.open(encoding="utf-8") as handle:
            profile = json.load(handle)
        require(profile.get("schema") == 1 and
                profile.get("status") == "offline_only",
                "transparent guard candidate identity is invalid")
        require(guard.get("profile") == "transparent_guard_candidate_v1",
                "harness guard profile name mismatch")
        for yaml_key, profile_key in (
            ("raw_action_abs_max", "raw_action_abs_max"),
            ("raw_action_step_max", "raw_action_step_max"),
            ("target_min_rad", "target_min_rad"),
            ("target_max_rad", "target_max_rad"),
        ):
            configured = finite_vector(guard.get(yaml_key), 12,
                                       f"harness_guard.{yaml_key}")
            expected = finite_vector(profile.get(profile_key), 12,
                                     f"profile.{profile_key}")
            require(all(abs(actual - wanted) <= 1.0e-9
                        for actual, wanted in zip(configured, expected)),
                    f"harness_guard.{yaml_key} does not match the fixed profile")
        load_guard = profile.get("provisional_load_guard", {})
        require(load_guard.get("status") == "requires_harness_validation",
                "load guard must remain marked as requiring harness validation")
        effort_limit = float(guard.get("joint_effort_abs_nm", math.nan))
        effort_frames = guard.get("effort_trip_frames")
        require(abs(effort_limit -
                    float(load_guard.get("joint_effort_abs_nm", math.nan))) <= 1.0e-9,
                "harness effort limit does not match the fixed profile")
        require(effort_frames == load_guard.get("trip_frames") == 3,
                "harness effort trip count does not match the fixed profile")

    logging = vision_loco.get("logging", {})
    require(logging.get("max_consecutive_errors") == 1,
            "one invalid inference frame must trigger the policy fault transition")
    policy_source = (Path(__file__).resolve().parents[1] /
                     "src" / "State_VisionLoco.cpp").read_text(encoding="utf-8")
    require('FSMStringMap.right.at("FixStand")' not in policy_source,
            "policy faults must never transition back to FixStand")
    require('FSMStringMap.right.at("Passive")' in policy_source,
            "VisionLoco must retain an explicit Passive fault transition")
    safety = vision_loco.get("safety", {})
    max_tilt_deg = float(safety.get("max_tilt_deg", math.inf))
    grace_s = float(safety.get("grace_s", math.nan))
    max_raw_abs = float(safety.get("max_raw_action_abs", math.inf))
    max_raw_step = float(safety.get("max_raw_action_step", math.inf))
    max_target_step = float(safety.get("max_target_step_rad", math.inf))
    zero_hold_threshold = float(
        safety.get("zero_command_hold_threshold", math.inf))
    action_step_frames = safety.get("action_step_trip_frames")
    max_tracking = float(safety.get("max_tracking_error_rad", math.inf))
    max_motion_tracking = float(
        safety.get("max_motion_tracking_error_rad", math.inf))
    tracking_frames = safety.get("tracking_error_trip_frames")
    motion_tracking_frames = safety.get("motion_tracking_error_trip_frames")
    require(math.isfinite(max_tilt_deg) and 10.0 <= max_tilt_deg <= 25.0,
            "safety.max_tilt_deg must be in [10, 25]")
    require(abs(max_tilt_deg - global_tilt_deg) <= 1.0e-6,
            "global and VisionLoco tilt limits must match")
    require(math.isfinite(grace_s) and entry_blend_s <= grace_s <= 5.0,
            "safety.grace_s must be between entry_blend_s and 5 seconds")
    require(math.isfinite(max_raw_abs) and 0.0 < max_raw_abs <= 6.0,
            "safety.max_raw_action_abs must be in (0, 6]")
    require(math.isfinite(max_raw_step) and 0.0 < max_raw_step <= 1.0,
            "safety.max_raw_action_step must be in (0, 1]")
    require(math.isfinite(max_target_step) and 0.0 < max_target_step <= 0.05,
            "safety.max_target_step_rad must be in (0, 0.05]")
    require(isinstance(action_step_frames, int) and 1 <= action_step_frames <= 5,
            "safety.action_step_trip_frames must be in [1, 5]")
    require(math.isfinite(max_tracking) and 0.0 < max_tracking <= 0.5,
            "safety.max_tracking_error_rad must be in (0, 0.5]")
    require(isinstance(tracking_frames, int) and 1 <= tracking_frames <= 25,
            "safety.tracking_error_trip_frames must be in [1, 25]")
    require(math.isfinite(max_motion_tracking) and
            max_tracking <= max_motion_tracking <= 1.0,
            "safety.max_motion_tracking_error_rad must be between the stand "
            "limit and 1.0")
    require(isinstance(motion_tracking_frames, int) and
            tracking_frames <= motion_tracking_frames <= 50,
            "safety.motion_tracking_error_trip_frames must be between the "
            "stand trip count and 50")
    require(math.isfinite(zero_hold_threshold) and
            0.0 <= zero_hold_threshold <= 0.05,
            "safety.zero_command_hold_threshold must be in [0, 0.05]")

    times = finite_vector(fix_stand.get("ts"), 3, "FSM.FixStand.ts")
    require(times[0] == 0.0 and all(b > a for a, b in zip(times, times[1:])),
            "FixStand times must start at zero and increase strictly")
    require(times[-1] >= 2.0, "FixStand transition must last at least two seconds")
    poses = fix_stand.get("qs")
    require(isinstance(poses, list) and len(poses) == len(times),
            "FSM.FixStand.qs must match ts")
    require(poses[0] == [], "FixStand first pose must be populated from live feedback")
    finite_vector(poses[1], 12, "FSM.FixStand.qs[1]")
    final_pose = finite_vector(poses[2], 12, "FSM.FixStand.qs[2]")

    joint_map = deploy.get("joint_ids_map")
    require(isinstance(joint_map, list) and sorted(joint_map) == list(range(12)),
            "deploy joint_ids_map must be a permutation of 0..11")
    policy_default = finite_vector(
        deploy.get("default_joint_pos"), 12, "deploy.default_joint_pos")
    hardware_default = [0.0] * 12
    for policy_index, hardware_index in enumerate(joint_map):
        hardware_default[hardware_index] = policy_default[policy_index]
    error = max(abs(actual - expected)
                for actual, expected in zip(final_pose, hardware_default))
    require(error <= 1.0e-6,
            f"FixStand final pose does not match mapped policy default (max error {error})")

    print("Safety configuration passed: "
          f"mode={'harness-guard' if harness_guard_mode else ('suspended-parity' if suspended_parity else 'normal')}, "
          "initial=Passive, live FixStand start, "
          f"ground caps=[{key_max_vx:.2f},{key_max_wz:.2f}], "
          f"keyboard hard nonzero={key_max_nonzero:.1f}s, "
          f"tilt={max_tilt_deg:.0f}deg, raw action <= {max_raw_abs:.1f}, "
          f"fault-guard grace={grace_s:.1f}s, entry blend={entry_blend_s:.1f}s, "
          f"ready=[blend {ready_gain_blend:.1f}s, stable {ready_min_stable:.1f}s, "
          f"support {ready_max_support_offset:.2f}rad], "
          f"target step <= {max_target_step:.3f}rad/frame, "
          f"tracking=[stand {max_tracking:.2f}/{tracking_frames}, "
          f"motion {max_motion_tracking:.2f}/{motion_tracking_frames}], "
          f"zero hold <= {zero_hold_threshold:.3f}.")


if __name__ == "__main__":
    main()
