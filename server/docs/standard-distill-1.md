# standard-distill-1

## Historical scope

This document describes the historical direct visual LBC experiment. It can
distill an **encoder-based bridge artifact** from privileged height scan to the
calibrated depth camera while preserving its Actor77 contract.

The original replay checkpoint
`archive/代码存档/复赛_standard/ckpt/model.ckpt-10288.pkl` has now been inspected
from its real pickle tensors. Its Actor and Critic first layers are `[512,301]`
and `[512,316]`; it contains `std`, `actor.*`, and `critic.*`, with no
`encoder.*`. Its SHA256 is
`d5999461f00c4634bdea0648e46baac9fba34e621eaa586fe26f64f68953eeed`.
It therefore **cannot** enter this LBC stage directly.

The earlier platform preload log showing `encoder.*`, `actor.*`,
`critic_encoder.*`, and `critic.*` referred to the later HJC bridge artifact
whose Actor input is 77-D. Numeric ID `10288` was reused and is not a structure
contract. New work must first complete and validate `STD-BRIDGE-R1`, then load
its `privileged_loco_teacher_v1` side artifact here.

R1 continues platform filename IDs from the parent: the completed 5000-iteration
run publishes `model.ckpt-teacher-15288.pkl`, while its payload records
`source_iteration=5000`. Phase-boundary teacher files are candidates rather than
automatic promotions; select the D1 teacher only after reviewing its quality
diagnostics and fixed-evaluation evidence.

## Runtime Contract

- Stage: `StandardDistill1Config`
- TOML: `agent_ppo/conf/train_env_conf_standard_standard_distill_1.toml`
- Algorithm: `lbc_loco`
- Environment: `Unitree-Go2-Velocity-Camera`
- Frozen teacher: validated encoder-based Standard bridge artifact
- Teacher encoder: height scan 256 -> latent 32
- Teacher Actor: proprio 45 + latent 32 = 77 -> action 12
- Student: depth 180x320 + proprio 45 -> CNN/LSTM -> latent 32
- Goal input: none

LBC accepts the new `privileged_loco_teacher_v1` package and legacy raw
`encoder.*`/`actor.*` weights. Critic-side keys and exploration standard
deviation are not part of visual distillation or deployment and are ignored.

## Checkpoint Naming

The main output is:

`model.ckpt-standard-<id>.pkl`

The save path also emits the platform alias:

`model.ckpt-<id>.pkl`

Both satisfy platform liveness discovery for `model.ckpt-*.*`. Loading extracts
the trailing numeric ID and accepts labelled names such as
`model.ckpt-locomotion-10288.pkl` without using the label to infer structure.

## Camera Pose

```toml
[camera.depth_camera]
offset_pos = [0.339871, 0.034697, 0.075010]
offset_rot = [0.982631, -0.007085, 0.184337, -0.020153]
```

The quaternion corresponds to a pitch of approximately 21.22 degrees. These
are camera mount extrinsics, not optical intrinsics.

## Curriculum And Commands

Standard terrain curriculum starts at level 3. The sampling range includes
backward, lateral, yaw, and near-stationary commands:

```toml
[commands.ranges]
lin_vel_x = [-0.15, 0.30]
lin_vel_y = [-0.10, 0.10]
ang_vel_yaw = [-0.60, 0.60]
```

`commands.limit` is deliberately identical to these ranges. This keeps the
frozen teacher inside the same command distribution throughout distillation;
the ranges still include backward, lateral, yaw, near-zero, and low-speed
samples.

## Randomization Phases

The first visual run enables friction randomization, observation noise, and
depth augmentation while external pushes remain disabled. Enable pushes only
in a later robustness continuation from a validated visual checkpoint.
