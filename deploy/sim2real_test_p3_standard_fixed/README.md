# P3 standard fixed-mode Go2 deployment

Use this package for the first hardware test of checkpoint `highslow-884257`.
The model has two logical layers: a standard low-level locomotion policy and a
track/high-level navigation policy. Fixed mode uses only the standard layer;
the host-side `fixed_cmd` supplies velocity commands.

The Jetson copy is built and passes
`scripts/run_loco_stage_fixed_test.sh --check`. See `ARTIFACTS.md` for the
checkpoint schema, ONNX contract, export command, checksums, and the remaining
motor-side smoke test.

The first policy target is slew-limited from the mapped FixStand posture.
Motion guards compare adjacent policy requests and return to FixStand after
repeated violations; they do not drop an upright robot into Passive.

For a suspended depth-isolation test, temporarily replace RealSense input
without editing the persistent config:

```bash
LOCO_DEPTH_SOURCE=constant ./scripts/run_loco_stage_fixed_test.sh --network eth0
```

The wrapper restores the original RealSense configuration on normal exit or
`Ctrl+C`.
