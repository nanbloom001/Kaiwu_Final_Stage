# STD-D2-Stair

## Purpose

Test whether the current Standard visual student's reproducible descending-
stairs failures are primarily a data-coverage problem. This is a minimal
visual-distillation continuation, not visual PPO and not a teacher-reward
change.

## Parent

Select the current accepted Standard visual student checkpoint. The stage has
`require_student_resume = true`, so selecting a teacher-only Encoder checkpoint
fails before rollout collection instead of silently starting a random visual
student.

## Only Training Change

Compared with `standard-distill-1`:

```toml
[terrain]
max_init_terrain_level = 4

[terrain.standard.pyramid_slope]
proportion = 0.05

[terrain.standard.pyramid_slope_inv]
proportion = 0.05

[terrain.standard.pyramid_stairs]
proportion = 0.30

[terrain.standard.pyramid_stairs_inv]
proportion = 0.60

[terrain.standard.maze]
proportion = 0.0
```

Network structure, latent dimension, LSTM, camera pose, losses, depth
augmentation, command distribution, randomization, and student-drive schedule
remain identical to D1.

## Checkpoint Naming

The stage inherits `model.ckpt-standard-<id>.pkl` and also emits
`model.ckpt-<id>.pkl`. Both match the platform liveness probe.
