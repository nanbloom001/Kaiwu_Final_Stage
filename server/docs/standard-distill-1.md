# standard-distill-1

## Goal

Convert the legacy flat Standard 10288 policy into the current privileged
`ActorCriticEncoder` contract. The 10288 Actor consumes all 301 policy
observations directly and has no latent encoder, so it cannot be loaded by LBC
visual distillation. This bridge first matches its 12-D raw actions.

## Runtime Contract

- Stage: `StandardDistill1Config`
- TOML: `agent_ppo/conf/train_env_conf_standard_standard_distill_1.toml`
- Algorithm: `behavior_distill`
- Output checkpoint: `model.ckpt-locomotion-<id>.pkl`
- Frozen teacher: platform-selected flat Standard checkpoint 10288
- Teacher actor input: proprio 45 + height scan 256 = 301
- Student actor input: proprio 45 + latent 32 = 77
- Goal input: none

The frozen teacher drives the environment and provides action labels. The
student encoder and Actor are optimized with raw 12-D action MSE. This stage
does not use a camera and does not train a `VisionEncoder`.

## Follow-Up Visual Stage

Only after the bridge checkpoint passes action-agreement and closed-loop
evaluation should `Config.CURRENT` be switched to
`StandardVisualDistill1Config`. Its configuration is:

`agent_ppo/conf/train_env_conf_standard_standard_visual_distill_1.toml`

That follow-up stage carries the calibrated camera pose:

```toml
[camera.depth_camera]
offset_pos = [0.339871, 0.034697, 0.075010]
offset_rot = [0.982631, -0.007085, 0.184337, -0.020153]
```

The quaternion corresponds to a pitch of approximately 21.22 degrees. These
values are camera mount extrinsics, not optical intrinsics. They intentionally
do not affect the current height-scan behavior bridge.

## Curriculum And Commands

Standard terrain curriculum starts at level 3. The initial command range spans
zero so backward, lateral, yaw, and near-stationary samples are present:

```toml
[commands.ranges]
lin_vel_x = [-0.15, 0.30]
lin_vel_y = [-0.10, 0.10]
ang_vel_yaw = [-0.60, 0.60]
```

The configured safety envelope is:

```toml
[commands.limit]
lin_vel_x = [-0.20, 0.65]
lin_vel_y = [-0.15, 0.15]
ang_vel_z = [-1.30, 1.30]
```

## Randomization Phases

Phase 1 enables friction randomization and observation noise while keeping
external pushes disabled. There is no depth augmentation in the behavior
bridge because this stage has no camera input. After selecting a stable
checkpoint, continue from it with `push_robots = true`; retain the 15-second
interval and 0.35 m/s maximum push speed for the first robustness run.
