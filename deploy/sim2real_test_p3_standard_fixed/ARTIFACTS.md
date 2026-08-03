# P3 standard fixed deployment package

This package is the first bring-up target for `model.ckpt-highslow-884257.pkl`.
It exports only the P3 standard low-level policy. The checkpoint also contains
track/high-level navigation modules; those are intentionally ignored in fixed
mode and are not part of `policy.onnx`.

## Verified artifacts

- Checkpoint: `models/model.ckpt-highslow-884257.pkl`
- Exporter: `export_p3_standard_onnx.py`
- ONNX: `runtime/unitree_rl_lab_test/logs/loco/exported/policy.onnx`
- Manifest: `runtime/unitree_rl_lab_test/logs/loco/exported/policy.manifest.json`
- Runtime source: `runtime/unitree_rl_lab_test/deploy/robots/go2_loco/`
- ONNX Runtime ABI: `runtime/unitree_rl_lab_test/deploy/thirdparty/onnxruntime-linux-aarch64-1.19.2/` (official aarch64 1.19.2 ELF files)

The ONNX graph was checked with ONNX Runtime on Windows. All eight names and
shapes match `loco_runner.h`:

```text
inputs:  depth[1,180,320,1], proprio[1,45], goal[1,4],
         loco_h[2,1,64], loco_c[2,1,64], nav_h[2,1,64], nav_c[2,1,64],
         cmd_override[1,4]
outputs: cmd[1,3], cmd_raw[1,3], clearance[1,3], joint[1,12],
         loco_h_out[2,1,64], loco_c_out[2,1,64], nav_h_out[2,1,64], nav_c_out[2,1,64]
```

The Python/ONNX eight-frame rollout check passed. Maximum joint absolute
error was `7.2e-7`.

## Fixed-mode configuration

`config.yaml` is set to:

```yaml
command_source: fixed
fixed_cmd: [0.0, 0.0, 0.0]
```

The command is deliberately conservative for first hardware bring-up. The
runner still writes the command into `proprio[6:9]`, preserves the previous
action in `proprio[33:45]`, and feeds the LSTM state frame to frame.

The runtime enters the policy from the exact mapped default joint posture,
uses the light RealSense spatial filter (hole filling and temporal filtering
disabled), and records adjacent policy action/target step sizes plus tracking
error. Policy targets are applied with a `3.0 rad/s` per-joint slew limit, so
the first recurrent output is blended in instead of being sent as a step. An
adjacent requested-target step above `0.35` rad or tracking error above `0.45`
rad is held for safety; two consecutive violations request FixStand so the
robot keeps standing instead of dropping into Passive.

Each diagnostic frame also records policy-order joint position and velocity,
motor-reported estimated effort (`tau_est`), raw policy action, requested and
actually applied targets, per-joint tracking error, nominal PD torque, and
absolute mechanical power. The analyzer remains compatible with older CSVs
that do not contain these extended columns.

## Export

```bash
python export_p3_standard_onnx.py \
  --ckpt models/model.ckpt-highslow-884257.pkl \
  --out runtime/unitree_rl_lab_test/logs/loco/exported/policy.onnx \
  --manifest runtime/unitree_rl_lab_test/logs/loco/exported/policy.manifest.json \
  --opset 18
```

## Jetson deployment

The package is installed at
`/home/unitree/Kaiwu_Final_Stage-main/deploy/sim2real_test_p3_standard_fixed`.
Set `LOCO_TEST_ROOT` if using a different path. The aarch64 controller was
built on the Jetson and its runtime libraries resolve correctly.

Validated on 2026-08-01:

- `scripts/run_loco_stage_fixed_test.sh --check` passed on `eth0`.
- Fixed command: `[0.0, 0.0, 0.0]`.
- Controller was rebuilt on Jetson after the runtime safety/filter changes.
- The two short logs that previously dropped to Passive were replayed through
  the corrected guard. Their second-frame requested steps were `0.0421` and
  `0.0339` rad, so neither is rejected by the adjacent-frame check.
- No controller process was left running and no motor command was started.

After confirming the robot is supported safely and the area is clear, run:

```bash
./scripts/run_loco_stage_fixed_test.sh
```

Motor-side smoke testing remains pending.
