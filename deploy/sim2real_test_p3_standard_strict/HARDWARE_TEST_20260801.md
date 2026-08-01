# Hardware static data-link test

Date: 2026-08-01, Go2 elevated, no feet contact, no motor command.

## Strict preflight

Passed with hardware USB access:

- Intel RealSense D435I detected, serial `338622073957`;
- firmware `5.13.0.55` (tool reports recommended `5.15.1`);
- checkpoint/ONNX/config/deploy hashes and exact ONNX ABI passed;
- `motor_command_channel_created=false`.

## LowState result

The strict binary successfully initialized DDS and reported:

```text
sensor-only LowState connection established
```

The strict run then stopped before the sampling loop because the camera did not
provide the required profile. This round therefore does not claim sustained
LowState freshness statistics or produce a strict sensor-only CSV.

## Camera result

The attached D435i exposes depth `640x480@30 Z16`, but not the strict training
profile `424x240@30 Z16`. The strict deployment correctly hard-failed instead of
silently resizing a different profile.

An independent read-only `pyrealsense2` probe at the available `640x480@30 Z16`
profile captured 120 frames:

- unique frame numbers: 120/120;
- repeated frames: 0; backward frames: 0;
- timestamp delta mean/min/max: `33.397 / 31.973 / 34.981 ms`;
- mean nonzero raw-depth pixel fraction: `0.862199`;
- intrinsics: `fx=387.073, fy=387.073, cx=323.185, cy=240.309`;
- depth scale: `0.00100000005 m`.

This proves the USB and camera frame stream is alive at 640x480, but it is not
evidence that the training/deployment depth geometry is aligned. Do not change
the strict contract to accept 640x480 without a deliberate retraining/export
decision and updated camera contract.

## Next gate

Keep the robot elevated. Resolve the missing `424x240@30 Z16` profile by checking
USB bandwidth/cable/hub and camera firmware, or explicitly decide on a new
training/deployment profile. Re-run strict preflight and then the 60-second
sensor-only route only after the exact profile is available.

## 2026-08-01 correction: capture profile is not the training tensor

The earlier interpretation above was incorrect: `424x240@30 Z16` is only the
preferred raw RealSense capture profile inherited from other deployment routes.
The P3 low-level training and ONNX input is `depth[1,180,320,1]`. The other
deployment routes fall back to `480x270@30` on this D435I and use the selected
profile's runtime intrinsics to reproject into the frozen training pinhole
(`fx=fy=168.61`, `cx=160`, `cy=90`).

The strict route now accepts only the audited raw profiles `424x240@30` and
`480x270@30`, still hard-fails without finite profile-matched intrinsics, and
computes validity statistics from the projected `320x180` tensor that the
low-level network actually consumes. This correction does not enable or test
the P3 high-level navigation model.

## Corrected sensor-only result

After adding a bounded first-frame startup gate, the strict sensor-only route
ran for `104.920 s` and stopped cleanly. It produced
`logs/sensor-only_1785524014.csv`; the process never created LowCmd.

- raw acquisition: `480x270@30 Z16`, no filters or hole filling;
- runtime intrinsics: `fx=241.9`, `fy=241.9`, `cx=242.0`, `cy=135.2`;
- target projection: `320x180`, `fx=fy=168.61`, `cx=160`, `cy=90`;
- policy samples: 5247 at `50.000 Hz`;
- unique camera frames: 3151 at `30.023 Hz`;
- depth frame backwards/skips: `0/0`;
- camera timestamp delta mean/p95/max: `33.302/33.314/34.723 ms`;
- depth age mean/p95/max: `16.701/31.656/33.589 ms`;
- projected invalid fraction mean/p95/max: `5.14%/5.49%/10.70%`;
- projected front invalid fraction mean/p95/max: `1.63%/1.86%/9.96%`;
- LowState tick repeats/backwards: `0/0`, mean tick step `20 ms`.

The approximately 40% repeated policy-frame ratio is expected when a 50 Hz
consumer reads the latest frame from a 30 Hz camera. This round validates the
sensor and projection path only. It does not execute the P3 high-level model,
low-level ONNX inference, LowCmd, or motor control.
