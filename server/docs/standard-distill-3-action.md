# STD-D3A: Stair Action Alignment

## Goal

Continue from the **D2-40min visual student** and repair stair action
imitation. D3A does not change `algorithm_lbc.py`, the visual network, latent
dimension, LSTM structure, teacher model, terrain distribution, or camera
extrinsics.

## Required Parent

The platform preload must be the D2-40min Camera/LBC checkpoint. The TOML keeps
`require_student_resume = true`, so a teacher-only checkpoint or a randomly
initialized visual student is rejected before rollout collection. A successful
startup logs:

```text
resumed visual student checkpoint: <selected D2 checkpoint>
```

## Experiment Changes

- `action_loss_weight`: `0.2 -> 1.0`
- `student_drive`: `true -> false`
- `learning_rate`: `5e-4 -> 2e-4`
- commands: fixed per episode, sampled across `0.45-0.70 m/s`
- domain randomization, observation noise, and depth augmentation: disabled
- sequence length: unchanged at 8 through `bptt_steps = 8`

The D2 terrain proportions and calibrated 21.22-degree camera extrinsics remain
unchanged.

## Checkpoint Naming

The descriptive artifact is:

```text
model.ckpt-standard-distill-3-action-<id>.pkl
```

`Agent.save_model()` also writes `model.ckpt-<id>.pkl`. That alias is required
because the platform's documented probe does not reliably parse labels with
multiple hyphen-separated words.

