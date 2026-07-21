# standard-distill-1

## Goal

Distill the platform-selected Standard checkpoint 10288 from privileged
height scan to the calibrated depth camera while preserving its locomotion
Actor contract.

The platform preload log proves that this checkpoint is already an
`ActorCriticEncoder`: it contains `encoder.*`, `actor.*`, `critic_encoder.*`
and `critic.*`; the Actor and Critic first layers use the 77-D and 92-D final
contracts. This matches the final-round task specification. It must therefore
enter LBC directly, not the optional 301-D legacy behavior bridge.

## Runtime Contract

- Stage: `StandardDistill1Config`
- TOML: `agent_ppo/conf/train_env_conf_standard_standard_distill_1.toml`
- Algorithm: `lbc_loco`
- Environment: `Unitree-Go2-Velocity-Camera`
- Frozen teacher: platform Standard `ActorCriticEncoder` checkpoint 10288
- Teacher encoder: height scan 256 -> latent 32
- Teacher Actor: proprio 45 + latent 32 = 77 -> action 12
- Student: depth 180x320 + proprio 45 -> CNN/LSTM -> latent 32
- Goal input: none

LBC loads only `encoder.*` and `actor.*`. The teacher checkpoint's
`critic_encoder.*`, `critic.*` and exploration standard deviation are not part
of deployment and are intentionally ignored.

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
