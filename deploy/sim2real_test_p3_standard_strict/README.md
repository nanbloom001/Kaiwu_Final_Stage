# Go2 P3 Standard strict deployment

This is an experimental deployment and sim2real diagnostic package for
`p3nav8h-r1_884257-evalfix-v2`. Motor, LowState and inference guards remain
fail-closed; depth content and frame cadence are explicitly warning-only. It is
intentionally separate from the stable
`deploy/sim2real_test_loco` route and does not modify that baseline.

The frozen contract is `artifact_contract.json`. It binds the checkpoint, ONNX,
exact 8-input/8-output ABI, training-effective action clip `[-6, 6]`, action
scale/offset, policy joint order, proprio45 layout, recurrent-state semantics,
depth preprocessing, camera reference, watchdogs, and physical joint limits.

Safe offline checks:

```bash
python3 tools/preflight.py --json
python3 tools/validate_parity.py --frames 100
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Build and C++ safety tests:

```bash
cmake -S runtime/unitree_rl_lab_test/deploy/robots/go2_loco \
  -B runtime/unitree_rl_lab_test/deploy/robots/go2_loco/build-strict
cmake --build runtime/unitree_rl_lab_test/deploy/robots/go2_loco/build-strict -j2
(cd runtime/unitree_rl_lab_test/deploy/robots/go2_loco/build-strict && \
  ctest --output-on-failure)
```

Deployment stages are deliberately separate:

```bash
./scripts/run_strict.sh --sensor-only --network eth0
./scripts/run_strict.sh --shadow --network eth0
```

Live D435i depth inspection and filter tuning, without DDS or LowCmd:

```bash
bash scripts/run_depth_tuner.sh
```

The GUI shows raw and filtered depth, policy-invalid pixels, central-third
invalid fractions, and pixels recovered or lost by filtering. It exposes the
supported spatial, temporal, independent hole-filling, emitter, laser,
frame-queue, automatic-exposure/gain-limit, timestamp and error-polling controls
at runtime. Live exposure/gain/laser values come from frame metadata, and a
capture-thread stream restart applies options that take effect on the next open.
USB re-enumeration of the selected serial automatically reopens the pipeline and
rebinds camera controls without replaying stale laser or exposure requests.
See `DEPTH_TUNER.md`; tuning does not automatically
change the powered deployment configuration.

The fixed forward command is supplied from the terminal and must be finite in
`[0,1.00] m/s`, for example `--fixed-vx --vx 0.3 --arm --network eth0`.
The accepted range matches the P3 training command range; it is not evidence
that every accepted speed has been validated on hardware.

Both modes consume LowState and D435i data but never construct `LowCmd`. Powered
modes require `--arm`, a detected RealSense camera, and physical gamepad
confirmation. The default powered path has no internal startup shadow: LT+X
resets the recurrent policy and the first valid inference frame begins the
command/action/last_action closed loop. Do not run it until the staged deployment
checklist in `ARTIFACTS.md` is reviewed at the robot.

Analyze a captured CSV with:

```bash
python3 tools/analyze_strict_log.py logs/shadow_TIMESTAMP.csv --json-out report.json
```

The analyzer keeps model raw action, clipped action, requested target, slew and
physical-limit targets, applied target, executed raw feedback, measured joints,
effort, freshness, per-joint frequency bands, per-leg concentration, and
left/right asymmetry separate.

Depth pixel invalidity, a missing first frame, and stale/stopped frames are
warning-only and never request Passive or block policy publication. The first
such warning requests that the preceding projected depth ring be saved beside
the VisionLoco CSV after normal exit. Add `--record-depth` to a powered command
to request the same ring explicitly. Each frame has a 16-bit millimeter PGM
(`0=invalid`), an 8-bit validity-mask PGM (`255=valid`), and an RGB PPM preview
with invalid pixels in magenta; `manifest.csv` maps all filenames and metrics.

Every non-preflight run also writes a persistent terminal/FSM log beside the
VisionLoco CSV as `strict_controller_TIMESTAMP.log`. After a failed run, inspect
the newest log and the exact takeover chain with:

```bash
LOG_DIR=runtime/unitree_rl_lab_test/logs/loco/logs
LATEST=$(find "$LOG_DIR" -maxdepth 1 -name 'strict_controller_*.log' -printf '%T@ %p\n' \
  | sort -nr | head -n1 | cut -d' ' -f2-)
tail -n 250 "$LATEST"
rg 'FAULT_SNAPSHOT|MECHANICAL_LIMIT|DEPTH_WARNING|LOWSTATE_FAULT|POLICY_TARGET_STALE|BAD_ORIENTATION|trigger=' "$LATEST"
```

`FAULT_SNAPSHOT` records the exact reason plus policy frame, command, LowState
age/tick, projected gravity and tilt, peak joint position/velocity/torque/
temperature, battery state, recent target/tracking/PD metrics, and depth frame
age/invalid fractions. Warning and fault records are flushed immediately so the
cause survives an automatic switch to FixStand or Passive.
