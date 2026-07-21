# ST7-Opt2A: Dangerous Tilt Constraint

## Baseline

- Code baseline: `/Users/cheng/Downloads/st7.zip`.
- Parent checkpoint: the original ST7 checkpoint corresponding to `logs-595320`.
- Training entry: `TrackNavConfig` (`name = "nav"`).
- Learning rate: `1.5e-5`.
- Model save interval: `20`.

Do not continue from an ST7-Opt1 checkpoint.

## Only Experiment Change

The original ST7 configuration is restored, then one existing reward is enabled:

```toml
[rewards.tilt_limit]
weight = -0.8

[rewards.tilt_limit.params]
max_roll_rad = 0.30
max_pitch_rad = 0.55
```

The reward penalizes only roll or pitch beyond the configured thresholds. No
Python reward function, model structure, terrain distribution, speed, posture,
foot, goal, or action-smoothing setting is changed relative to ST7.

## Training And Evaluation

Start two independent jobs from the same original `logs-595320` checkpoint:

- ST7-Opt2A-10min
- ST7-Opt2A-20min

For both candidates, first run the L9 video evaluation with `save_mp4=true`,
`level=[9]`, `num_envs=16`, and `seed=12345`. Record upstairs, downstairs and
maze failures plus `bad_orientation` and `base_contact` counts.
