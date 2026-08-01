# Artifact and validation record

## Status

- Route: experimental P3 Standard low-level override, not the stable default.
- Training reference: `/home/unitree/p3_training_reference/p3nav8h-r1_884257-evalfix-v2`.
- Checkpoint format: `kaiwu_train_v1`, schema 2, stage `p3_standard_joint`, phase `highslow`.
- `deployable=false`; `goal_dim=0`; ABI `goal[1,4]` remains an ignored placeholder.
- Hardware sensor-only strict route: passed for 104.92 s at audited 480x270@30 capture with 320x180 reprojection.
- Warning-only depth-health smoke: with `laser_power(max)=360`, the sensor-only
  process remained active for 11.52 s until the test timeout, wrote 577 control
  samples, and reached camera frame 366 even though projected invalid fractions
  remained about `0.98-1.00`. It produced `DEPTH_WARNING` only, no depth takeover,
  and never created LowCmd.
- Independent read-only D435i frame probe: passed at the device's available 640x480@30 profile.
- Shadow minimum-duration acceptance remains pending.
- Suspended powered zero-command characterization completed for 323 frames over
  6.441 s at 49.99 Hz without a rejected motion frame. A subsequent conservative
  ground attempt stopped after 10 frames when FR calf tracking error reached
  `0.451216 rad`, just above the `0.45 rad` diagnostic threshold.
- Powered ground zero-command characterization passed for 2030 frames over
  40.578 s at 50.00 Hz. Peak motor-reported/PD-estimated torque was
  `16.263/16.447 Nm`, both on FR calf; no torque-warning or rejected-motion frame
  occurred. Two `0.3 m/s` runs had no recorded internal fault. Two `0.8 m/s`
  runs ended as FL-thigh torque approached the former `18 Nm` SoftStop boundary;
  the torque SoftStop is now removed and powered retest is pending.

## Frozen artifacts

| Artifact | SHA256 |
| --- | --- |
| `models/model.ckpt-highslow-884257.pkl` | `8dc9d6028bd8850a3e59cfde2bd2ee1fe5b6768f1af47477a20c28a831edcb23` |
| `runtime/unitree_rl_lab_test/logs/loco/exported/policy.onnx` | `6ccaf7e033240720a66019885df091097efa2d6d8962a5840966ede8b9de2d87` |
| controller `config.yaml` | `154d1d47d3c11330c4384eaf6620f41e93cea390e4175f0a4c38898578058ffb` |
| policy `deploy.yaml` | `5aa6ced0023641aed7a5c801f39dc290470a6a81003b62c273dca9a33e66b6a9` |
| local aarch64 `build-strict/go2_loco_ctrl` | `5eeb65aad78226903a30477138ea0ed72c82b9ea356c87cca2d97b2320619d91` |

`artifact_contract.json` is the machine-readable source of truth. Preflight
recomputes these hashes and verifies all ONNX names, dtypes, ranks, static shapes,
opset 18, and IR version 10 before DDS or LowCmd initialization.

## Training/deployment alignment

- Selected modules: `modules.low_level.locomotion_encoder` and
  `modules.low_level.actor`; high-level/navigation/critic modules are ignored.
- Depth: `180x320x1`; VisionEncoder feature 32; LSTM input 77, hidden 64,
  two layers; latent 32 with L2 normalization; Actor77 output 12.
- Proprio: `ang_vel3 | projected_gravity3 | cmd3 | joint_pos_rel12 |
  joint_vel_rel12 | last_action12`, with scales `0.25/1/1/1/0.05/1`.
- The training workflow clamps actions to `[-6,6]` before every `env.step`.
  This effective transport rule overrides the historical generated
  `deploy.yaml` value `[-100,100]`.
- Action chain: finite check -> raw clip -> scale `0.25` + offset -> target
  slew `2.5 rad/s` -> URDF limits with `0.05 rad` margin -> applied target.
  `last_action` is the executed raw equivalent after every safety modification.
- Control period: 20 ms; Kp 25; Kd 0.5; policy-to-SDK joint map
  `[3,0,9,6,4,1,10,7,5,2,11,8]`.

## Camera boundary

Training reference extrinsics are position
`[0.339871,0.034697,0.075010] m` and quaternion WXYZ
`[0.982631,-0.007085,0.184337,-0.020153]`. These values are copied as an
uncalibrated reference, not asserted as the live mount calibration. The frozen
network input is `320x180x1`, not a RealSense capture profile. Strict acquisition
accepts only audited `424x240@30` or `480x270@30 Z16`, requires finite runtime
intrinsics matching the selected profile, and reprojects to the training pinhole
before normalization. Any other profile, missing intrinsics, geometry mismatch,
or implicit filter is rejected.

The software distance contract remains training-identical: meters in `(0,5)`
map to `(0,1)` and invalid/zero/`>=5 m` map to zero. Stair-front failures were
caused by projected holes, not the 5 m far boundary. The D435i now requests High
Density visual preset, emitter enabled, device-maximum laser power, and auto
exposure. Whole-image or central-third invalid fraction at `0.50` emits a warning;
the hard invalid boundary is `0.70` for each. Device support and applied option
values are printed at startup.

Read-only host inspection found no ROS 2 topic carrying this D435i Z16 stream.
`/frontvideostream` is `unitree_go/msg/Go2FrontVideoData` and
`/pctoimage_local` is `unitree_interfaces/msg/PcToImage`; neither replaces the
direct depth feed. The controller links librealsense `2.54.2` and opens the D435i
through `rs2::pipeline`. Installed camera firmware is `5.13.0.55`; the SDK tool
reports `5.15.1` as recommended, but no firmware update was performed.

The standalone `tools/realsense_depth_tuner.py` diagnostic opens only the D435i
and never initializes DDS, LowState, or LowCmd. Its OpenCV GUI compares raw and
filtered depth, highlights deployment-invalid pixels (`0`, non-finite, or
`>=5 m`), reports whole-image and central-third invalid fractions, and exposes
runtime-supported spatial, temporal, independent hole-filling, emitter, laser,
preset, exposure, and gain controls. Snapshots preserve raw/filtered 16-bit
depth, masks, arrays, dashboard, and exact settings. This tool does not alter the
frozen controller configuration; any selected filter profile requires a separate
reviewed configuration change and sensor-only regression.

## Local evidence

- 100-frame checkpoint-to-ONNX recurrent rollout passed. Maximum joint absolute
  error: `8.345e-07`; cmd/cmd_raw/clearance maximum error: `0`.
- Python safety tests cover artifact and ABI fault injection, depth golden
  vectors, non-finite actions, action-layer feedback, watchdog deadlines,
  concurrency, and synthetic frequency/leg diagnostics.
- C++ tests cover action layering, quaternion validity, immutable snapshot reads,
  and a ThreadSanitizer stress target.
- Sensor-only log `logs/sensor-only_1785524014.csv` contains 5247 samples over
  104.92 s at 50.000 Hz. It observed 3151 unique depth frames at 30.023 Hz,
  no backward frame numbers, no skipped forward frame numbers, and depth age
  max/p95 `33.589/31.656 ms`.
- Projected `320x180` invalid fraction mean/p95/max was
  `0.0514/0.0549/0.1070`; projected front invalid fraction was
  `0.0163/0.0186/0.0996`. LowState tick had no repeats or backwards movement.
- This is camera/LowState sensor-only evidence. Low-level ONNX shadow inference,
  powered suspension, hardware safety, and locomotion remain unverified.
- The default powered startup has `startup_shadow_enabled=false`. LT+X resets the
  recurrent policy and the first valid inference frame starts the synchronized
  command/action/last_action loop. This avoids advancing the LSTM under a walking
  command while withholding the corresponding hardware actions. An internal
  stability/time gate remains available only by explicitly setting
  `startup_shadow_enabled=true`; its gate hold is separate from the live target
  stale watchdog and remains latched after passing.
- `--suspended-test` is accepted only with `--fixed-zero --arm`. It bypasses the
  optional startup stability/time gate when that gate is explicitly enabled and
  evaluates motion-step safety after clipping, physical limiting, and
  `0.05 rad/frame` slew instead of rejecting normal pre-slew network variation.
  Tracking error, LowState freshness, inference deadline, mechanical limits,
  target watchdog, and fault takeover remain enabled. The mode is recorded in
  every VisionLoco CSV row.
- `--ground-test` is also accepted only with `--fixed-zero --arm` and is mutually
  exclusive with `--suspended-test`. It bypasses the startup gate, audits the
  post-slew applied step, and disables only the `0.45 rad` tracking-error soft
  gate. Raw-action finite/clip checks, `0.05 rad/frame` slew, URDF physical limits,
  LowState freshness, inference and policy-target deadlines, joint
  velocity/estimated torque/temperature guards, and takeover remain enabled.
  This is a short characterization mode, not the normal deployment profile.
- `--fixed-vx --vx VALUE --arm` accepts finite forward commands in
  `[0,1.00] m/s`, matching the active P3 training range. With the default startup
  configuration it begins on the first valid post-reset inference frame, audits
  only the target that remains after the `0.05 rad/frame` slew, and disables the
  tracking-error soft gate. The CSV marks this with
  `walking_test_permissive=1`. Physical joint limits, finite action checks,
  sensor/inference/target watchdogs, joint velocity, estimated torque,
  temperature, and takeover remain active. The accepted upper bound is not a
  claim that `1.00 m/s` is already hardware-validated.
- Torque above `14 Nm` on hip/thigh or `27 Nm` on calf is warning-only. The
  former torque SoftStop is disabled by setting it equal to the hard boundary.
  Torque above `22 Nm` on hip/thigh or `43 Nm` on calf remains a HardFault and
  requests immediate Passive takeover. Velocity, temperature, position, and
  data-path faults retain their existing behavior.
- Every FSM transition now prints its trigger, including configured gamepad
  transitions, LowState timeout, bad orientation, `hard_fault:<reason>`, and
  `soft_stop:<reason>`. All terminal/FSM output is also persisted to
  `logs/loco/logs/strict_controller_TIMESTAMP.log`; warning and fault records
  flush immediately. Structured `FAULT_SNAPSHOT` records include the triggering
  reason, policy frame, command, LowState freshness, projected gravity/tilt,
  peak joint position/velocity/torque/temperature, recent target/tracking/PD
  metrics, and depth freshness/invalid fractions. Dedicated records identify
  mechanical limits, LowState failure, stale policy targets, bad orientation,
  and warning-only depth-health transitions. A depth warning prints
  camera/profile/frame/age and both validity fractions without takeover. The
  latest 90 unique projected depth frames are held in RAM; after the first
  depth-health warning they are dumped on normal exit
  in three representations per frame: 16-bit millimeter PGM, 8-bit valid-pixel
  mask PGM (`255=valid`, `0=invalid`), and RGB PPM preview (invalid pixels are
  magenta), plus `manifest.csv`. Warning-triggered dumps use
  `visloco_diag_TIMESTAMP_depth_warning/`; adding `--record-depth` to a powered
  command also dumps the recent ring after a normal/manual exit under
  `visloco_diag_TIMESTAMP_depth_capture/`.

## Staged robot test order

1. `preflight`: no DDS, no LowCmd.
2. `sensor-only`: at least 60 s LowState + D435i freshness/profile logging; no LowCmd.
3. `shadow`: at least 60 s ONNX inference and safety-layer logging; no LowCmd.
4. Review the analyzer report and measured camera mount calibration.
5. Suspended conservative path: `--fixed-zero --arm`; LT+A enters FixStand,
   LT+X requests VisionLoco, resets the recurrent state, and the first valid
   inference frame begins the closed loop. LT+B returns Passive.
6. Suspended diagnostic mode, only after the conservative path is understood:
   `--fixed-zero --arm --suspended-test`. It audits the post-slew target and also
   bypasses the optional startup gate if that gate is explicitly re-enabled; it
   must never be used on the ground or with a nonzero command.
7. Initial flat-ground zero-command characterization, with clear space and an
   operator holding LT+B: `--fixed-zero --arm --ground-test`. Start with about
   one second and inspect the CSV before extending the run.
8. First nonzero command test: `--fixed-vx --vx 0.3 --arm`. Use a flat,
   high-traction, open area and review a short run before increasing duration or
   speed. `--ground-test` remains exclusive to fixed zero.
9. Later speeds may be supplied through `--vx` up to `1.00 m/s`, but must be
   increased incrementally from verified evidence rather than jumping directly
   to the accepted maximum.

Any stale/frozen LowState, invalid depth, ABI/hash mismatch, non-finite output,
inference deadline miss, mechanical limit, repeated unsafe target, or stale
policy target freezes publication and requests FixStand or Passive.
