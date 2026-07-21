# ST7-Opt3: UWB Goal Noise Robustness

## Baseline

- Code parent: ST7-Opt2B (`f8f7ec0`).
- Checkpoint parent: the exact ST7-Opt2B checkpoint evaluated as `595693`.
- Training entry remains `TrackNavConfig` (`name = "nav"`).

## Only Experiment Change

The Actor receives geometrically consistent UWB-style goal noise in metric
space. The Critic, rewards, completion checks and simulator
`env.goal_positions` continue to use clean ground truth.

Noise consists of per-frame bearing/distance jitter and independent per-episode
bearing/distance biases for 75% of environments. Jump, stale-data and dropout
faults remain disabled. Per-episode state is resampled only for reset env IDs,
detected from `episode_length_buf` rollback.

The existing three-dimensional goal encoding and clipping are retained to keep
the Opt2B checkpoint input contract unchanged.

## Startup Checks

The environment log prints `[GoalNoise]` on the first observation and every 500
policy observations. Confirm:

- `active_ratio` is close to `0.75`;
- bearing values are in radians and distance values are in meters;
- jitter and bias means are finite and non-zero;
- Critic observation code does not import or call `GoalNoiseAugmenter`.

Before training, evaluate the `595693` checkpoint once with goal noise disabled
and twice with the same Opt3 noise enabled. The evaluation TOML is platform
owned and must receive the same `[goal_noise]` block manually.
