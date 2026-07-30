# ARTIFACTS - sim2real_test_378413_hardened

## Purpose

This tree is a safety-hardened comparison deployment for the historical
`model.ckpt-vision-378413.pkl` visual locomotion policy. It is intentionally
separate from both:

- `sim2real_test_loco`, which preserves the historical deployment and its old
  controller behavior; and
- `sim2real_test_p15_37953`, whose low-level policy was rejected in ground
  testing for a twisted, high-stepping gait and excessive load.

The 378413 policy has historical Go2/D435i evidence of walking with a fixed
`vx=0.15 m/s` command. A later hardened ground comparison fell forward, and a
post-failure audit proved that the hardened runtime had materially changed the
policy feedback contract. This tree is now offline/suspended-diagnostic only;
all ground use remains blocked.

## Fixed artifact identity

| Artifact | SHA256 | Status |
|---|---|---|
| `models/model.ckpt-vision-378413.pkl` | `37429c1e2c1d263844a74ecc2fdb97201ae296663f5ad8890ce173d5525a1288` | Tracked checkpoint, `format=lbc_loco` |
| `runtime/unitree_rl_lab_test/logs/loco/exported/policy.onnx` | `823614aa579b38a57d3210bfdf32a15e92002768694cb0c9882983eba6bd8e72` | External/generated Jetson artifact |
| `go2_loco_ctrl` | `bb2d2e89a6a23473f0bb6dfb529cbf71daeaf3cfbf0adaa7e3e57848c8b3d59d` | Current Jetson Release harness candidate; 4/4 CTest and fail-closed preflight passed; physical harness contact pending |

The runner refuses a checkpoint or ONNX with any other hash. `deploy.yaml`
must also declare `source_ckpt=model.ckpt-vision-378413.pkl`,
`source_ckpt_format=lbc_loco`, and `paired_onnx=policy.onnx`.

## Model and mechanical contract

- Visual policy: Actor80 (`proprio45 + latent32 + goal3 -> action12`).
- Depth: `180x320x1`; the ONNX goal input remains four-wide and the Actor uses
  the first three values.
- ONNX ABI: eight inputs and eight outputs, batch one, recurrent loco h/c.
- Control rate: 50 Hz.
- Joint map: `[3,0,9,6,4,1,10,7,5,2,11,8]`.
- Action: `target = default_joint_pos + 0.25 * raw_action`.
- Policy gains: `kp=25`, `kd=0.5` on all 12 joints.

The checkpoint, ONNX, joint map, action scale, offsets and policy gains are
unchanged from the historical deployment. Only the controller safety envelope
and qualification entry points are different.

## Hardened runtime

- Startup state is explicit `Passive`; `LT+A` enters FixStand and `LT+X` enters
  VisionLoco only after the operator starts the controller interactively.
- FixStand starts from live joint feedback and ends at the mapped policy default
  over two seconds.
- VisionLoco starts from live joint state, blends gains to `25/0.5` over
  1.5 seconds, and requires stable joints for 0.5 seconds before policy input.
- During that gain reduction, the current source estimates the pre-entry PD
  support torque and blends toward an equivalent target for `25/0.5` instead
  of setting target equal to measured position and unloading the legs. Both
  the inherited and compensated targets are bounded to `0.20 rad` from the
  measured joint position.
- Zero command holds the measured stand pose instead of running the recurrent
  gait at zero speed.
- The current source resets both recurrent state and `last_action[12]` before
  building the first non-zero-command observation, matching the historical
  motion-entry contract. The rejected binary performed this reset too late.
- Tilt above 25 degrees, non-finite inference, raw action above 6.0, persistent
  raw-action steps, excessive target steps or persistent tracking error go
  directly to Passive.
- Policy faults never return automatically to high-gain FixStand.
- RealSense shutdown stops the pipeline before joining the capture thread.
- `LT+B` remains the independent emergency transition to Passive.

Normal keyboard configuration uses a single `W=0.15 m/s` step, caps forward
speed at `0.15 m/s`, caps yaw at `0.1 rad/s`, and leaves the hard non-zero
timeout disabled. Qualification wrappers temporarily set the timeout to 2.0
seconds; key repeat cannot extend it, expiry forces the execution command
immediately to zero, and motion keys remain latched until VisionLoco is
re-entered.

## Verification gates

The existing 378413 ONNX passed the hardened `test_loco_runner` ABI/recurrent
smoke on Jetson on 2026-07-30. It echoed the command override correctly, kept
all outputs finite and stayed below the 0.05 rad/frame blended-target-step
limit. The independent Release build then passed `loco_safety_config`,
`uwb_goal_conversion`, `loco_policy_safety` and `loco_runner_onnx_abi` (4/4).
`constant pulse --check` confirmed caps `[0.15,0.10]`, hard timeout 2.0 seconds,
constant depth and no controller startup. This proves build, configuration and
runtime ABI compatibility only. A separate `realsense zero --check` detected
the D435i and passed with the same caps without starting the controller.

The physical constant-depth suspended pulse passed on 2026-07-30. Evidence is
`logs/20260730_152014_loco_keyboard_378413_hardened`: one `W` produced exactly
`vx=0.15 m/s`, the hard timeout fired after 2.019 seconds and forced the
same-frame execution command to zero, and stand hold then remained at zero for
12.4 seconds before `LT+B`. Maximum tilt was `5.19 deg`, maximum measured
absolute joint torque was `5.97 Nm`, and motion-window joint tracking error was
P95/max `0.120/0.235 rad`. Inference P95 was `7.32 ms` at 50 Hz with zero
deadline misses. The controller exited with status 0 and left no process
running.

The RealSense suspended zero-command test also passed on 2026-07-30. Evidence
is `logs/20260730_152810_loco_keyboard_378413_hardened`: the D435i ran at
`480x270@30fps`, execution remained exactly zero for 34.3 seconds, and depth
invalid mean/P95 was `15.8%/16.0%` overall and `0.3%/0.4%` in the forward
region. Maximum tilt was `5.1 deg`, maximum measured absolute joint torque was
`1.09 Nm`, joint tracking error P95/max was `0.040/0.040 rad`, and inference
P95 was `6.86 ms` at 50 Hz with zero deadline misses. RealSense shutdown was
clean, the controller exited with status 0 and no process remained. The
RealSense suspended pulse then passed in
`logs/20260730_153142_loco_keyboard_378413_hardened`: one `W` produced exactly
`vx=0.15 m/s`, the hard timeout fired after 2.021 seconds, and execution stayed
zero for another 15.4 seconds. During motion, maximum measured absolute joint
torque was `4.31 Nm`, joint tracking error P95/max was `0.112/0.165 rad`, and
the maximum target step remained exactly `0.030 rad/frame`. Overall maximum
tilt was `5.3 deg`; depth invalid P95 was `16.1%` overall and `0.4%` forward;
inference P95 was `7.77 ms` at 50 Hz with zero deadline misses. RealSense
shutdown and controller exit were clean. Both suspended gates are now complete.
The ground wrapper `--check` passed without starting the controller, so one
flat-ground `0.15 m/s`, two-second qualification was attempted.

That ground run was rejected. In
`logs/20260730_153540_loco_keyboard_378413_hardened`, the robot pitched forward
and fell about 1.34 seconds after the single `W`. The command path was correct
at `vx=0.15 m/s`, RealSense and 50 Hz inference timing were healthy, but the
real-contact policy response diverged: the left-front calf reached `16.60 Nm`
and `0.678 rad` tracking error, while the right-front calf raw action grew to
`-4.89` on the last valid frame and continued beyond the configured output
safety bound on the next inference. The controller detected the invalid or
over-limit ONNX output and transitioned directly to Passive. The recorded
VisionLoco tilt was still only `3.3 deg` because logging stopped at the policy
fault before the physical fall completed. An earlier FixStand attempt in the
same run also tripped the independent 25-degree global tilt guard. The process
exited with status 0 and no controller remained. Command, camera, ABI and timing
faults were excluded, but the initial conclusion that this isolated a model-only
contact Sim2Real gap was incorrect.

Three historical successful CSVs from the same checkpoint and ONNX show raw
action maxima of `7.138`, `7.569` and `6.872`, while `93.7%-97.5%` of normal
target steps exceeded `0.03 rad/frame`. The hardened `6.0` raw-action bound and
`0.03` target-step limiter therefore do not preserve the historical policy
distribution. The hardened robot also entered motion with its front calves
`0.24-0.28 rad` below policy defaults after the low-gain stand hold. Offline
reconstruction of the smoothstep blend and limiter matches the failed log to
within `0.003887 rad`, while their combined distortion from the policy target
reaches P95/max `0.862/1.595 rad`. The rejected ground run is consequently a
deployment-contract mismatch with possible residual model robustness weakness,
not a valid model-only qualification. All further ground use is prohibited.

The standard-library analyzer is:

```bash
python3 scripts/analyze_runtime_parity.py \
  --historical visloco_diag_1783840888.csv \
  --historical visloco_diag_1783840935.csv \
  --historical visloco_diag_1783841034.csv \
  --failed visloco_diag_1785396976.csv
```

The first-motion reset, support-preserving gain transition and exact ONNX bound
diagnostics were built on the Jetson after this audit. The Release binary
SHA256 is
`4f4c966947f8e9e11fbfe6973d3469121fb3f57a66f957f65f16de6e1dfd9df5` and
`loco_safety_config`, `uwb_goal_conversion`, `loco_policy_safety` and
`loco_runner_onnx_abi` passed 4/4.

This exact binary then passed three new fully suspended runs. In
`logs/20260730_163336_loco_keyboard_378413_hardened`, constant depth and zero
command remained stable for 147.1 seconds: support compensation was at most
`0.062 rad`, ready completion stayed at `0.089 rad/s` and `0.048 rad` tracking,
maximum tilt was `5.7 deg`, maximum torque was `1.39 Nm`, and 50 Hz inference
had zero deadline misses. In
`logs/20260730_164020_loco_keyboard_378413_hardened`, one automated constant-depth
`W` pulse produced `vx=0.15 m/s` for exactly 2.000 seconds and then remained at
zero. Motion tracking P95/max was `0.200/0.286 rad`, torque P95/max was
`5.07/7.21 Nm`, raw action P95/max was `4.82/5.15`, maximum target step was
`0.030 rad/frame`, maximum tilt was `5.74 deg`, and there were no deadline
misses. In `logs/20260730_165153_loco_keyboard_378413_hardened`, the same exact
two-second pulse passed with the D435i at `480x270@30fps`; post-pulse zero hold
lasted 38.68 seconds, motion tracking P95/max was `0.138/0.178 rad`, torque
P95/max was `3.51/4.60 Nm`, raw action P95/max was `3.53/3.64`, depth-invalid
P95 was `4.07%` overall and `2.99%` forward, maximum tilt was `5.86 deg`, and
50 Hz inference had zero deadline misses. All three exits were clean and the
final exact process check found no `go2_loco_ctrl` process.

These runs qualify the corrected takeover and suspended command path only. They
do not qualify contact behavior because the `0.030 rad/frame` limiter still
changes most historical successful motion frames.

The separately gated historical-parity diagnostic is now implemented. Its
expanded `8.0/8.0/1.0` action envelope and entry-blend bypass apply only while
the two-second non-zero command is active. Stand hold and the forced-zero return
retain the normal `6.0/1.0/0.03` envelope. The general runner rejects direct
`--suspended-parity` use, the wrapper requires the literal `SUSPENDED`
acknowledgement, and the ground wrapper remains hard-blocked. Jetson Release
build, 4/4 CTest, fail-closed invalid-config testing, config restoration and
`--check` passed with no controller process.

The constant-depth physical parity run
`logs/20260730_174509_loco_suspended_parity_378413` completed with exit status
zero. One `W` produced 98 non-zero frames and returned to zero after 1.960
seconds at the CSV sampling boundary. The sent target matched
`offset + 0.25 * raw_action` within `0.000001 rad` on every motion frame,
confirming that the motion-only blend and target-step restrictions were
transparent. Target-step P95/max was `0.310/0.455 rad/frame`, within the
historical successful range; raw-action max was `4.35`. This increased
suspended tracking-error P95/max to `0.567/0.711 rad` and torque P95/max to
`10.02/11.00 Nm`, while maximum tilt remained `6.36 deg`, 50 Hz inference
had zero deadline misses, and the forced-zero return remained limited to
`0.030 rad/frame`. The configuration was byte-restored and no controller
process remained.

An offline-only transparent guard candidate is frozen in
`config/transparent_guard_candidate.json` (SHA256
`420588a28ec6611d38d98fa5b5f7505aca10d57653bf3b305391623498a4ec75`).
Its per-joint raw-action magnitude, raw-action step and target-position bounds
come from the three hash-pinned historical successful logs with explicit
quantized margins. `scripts/replay_transparent_guard.py` (SHA256
`111be2465b3d98ea612c0fb89e4f4daa8bb790441eba31b46f4f8d4b82f14cc4`)
replayed all three historical logs and the suspended parity run with zero
output-envelope violations. The known failed ground run also had zero output
violations, proving that policy-output bounds cannot separate the failure. A
provisional `12 Nm / 3 frame` effort rule would identify that failure at
frame 28 (about 0.56 seconds, FR calf) and did not fire in the suspended parity
run, but historical successful logs have no effort columns. The profile
therefore remains `offline_only`, is not read by the controller, and cannot
authorize ground use.

The candidate is now implemented behind the separate
`run_loco_harness_guard.sh` gate. It requires RealSense, unfiltered depth,
one `W=0.15 m/s` pulse, a two-second hard timeout, the literal
`HARNESS_CLEAR` acknowledgement and a load-bearing fall-arrest harness.
During motion it passes policy targets transparently only while per-joint
raw-action magnitude, raw-action step and target-position checks pass. Reported
joint effort over `12.0 Nm` for three consecutive frames, any output-envelope
violation, target distortion or inference fault transitions to Passive. Stand
hold and forced-zero return retain the normal limiter. The general runner
rejects direct harness-mode use and ground qualification remains hard-blocked.
Jetson 4/4 CTest, RealSense `--check`, byte-exact config restoration, direct
entry rejection and tampered-profile fail-closed testing passed without
starting a controller. The current binary has not yet performed physical
contact; the earlier suspended parity evidence belongs to binary
`5e50b0a99089ac37da4c261f1ac24f6e2fd996a4c86c642231e277e2cf88c824`.

Required next work:

1. Run exactly one two-second pulse with the robot secured by a load-bearing
   fall-arrest harness and inspect the log before any repeat.
2. Do not test the candidate on unsupported ground.
3. Keep ground qualification blocked until a separately reviewed candidate and
   explicit test decision exist.

No stairs, UWB navigation, higher speed, higher gains or relaxed guards are
allowed during this comparison.

## Jetson path and commands

Install at:

```text
/home/unitree/Kaiwu-test/sim2real_test_378413_hardened
```

Read-only preflight:

```bash
cd ~/Kaiwu-test/sim2real_test_378413_hardened
./scripts/run_loco_suspended_parity.sh constant --check
```

Historical-parity physical test, only after every foot is clear:

```bash
./scripts/run_loco_suspended_parity.sh constant
```

Ground qualification is hard-blocked after the rejected run:

```bash
./scripts/run_loco_ground_qualification.sh --check
```

The wrapper exits non-zero without changing configuration or starting the
controller. Do not bypass it by invoking the general stage runner.

`--check` never starts the controller. A physical launch requires an interactive
terminal and the exact `SUSPENDED` or `GROUND_CLEAR` acknowledgement.

## Residual risk

378413 is an older coupled Actor80 policy, not the desired final hierarchical
low-level model. The parity run proves target transparency only and exposed
substantially higher suspended tracking error and torque than the hardened
envelope. It does not qualify contact behavior and does not authorize another
physical or ground test.
