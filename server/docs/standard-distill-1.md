# standard-distill-1

## Goal

Continue visual distillation from the hjcnew 10288 Standard teacher while
preserving the existing 77-D teacher contract, model structure, and observation
ordering. The stage expands command sampling to low-speed motion in every
direction and applies the calibrated camera mount pose.

## Runtime Contract

- Stage: `StandardDistill1Config`
- TOML: `agent_ppo/conf/train_env_conf_standard_standard_distill_1.toml`
- Checkpoint: `model.ckpt-standard-<id>.pkl`
- Parent: platform-selected hjcnew Standard teacher checkpoint 10288
- Actor input: proprio 45 + latent 32 = 77
- Goal input: none

The current sequence-aware LBC workflow is reused. The teacher encoder and
actor stay frozen; only the vision encoder and LSTM are optimized. Training
uses an 8-step sequence and a teacher/mixed/student DAgger schedule.

## Camera Pose

```toml
[camera.depth_camera]
offset_pos = [0.339871, 0.034697, 0.075010]
offset_rot = [0.982631, -0.007085, 0.184337, -0.020153]
```

The quaternion corresponds to a pitch of approximately 21.22 degrees. These
values are camera mount extrinsics, not optical intrinsics.

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

Phase 1 enables friction randomization, observation noise, and depth
augmentation while keeping external pushes disabled. After selecting a stable
checkpoint, continue from it with `push_robots = true`; retain the 15-second
interval and 0.35 m/s maximum push speed for the first robustness run.
