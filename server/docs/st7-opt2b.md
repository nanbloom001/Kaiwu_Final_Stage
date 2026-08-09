# ST7-Opt2B: Dynamic Tilt Risk

## Baseline

- Code baseline: the clean ST7 source restored from the ST7 source archive (`st7.zip`).
- Parent checkpoint: the exact original ST7 checkpoint evaluated in `logs-595320`.
- Training entry: `TrackNavConfig` (`name = "nav"`).
- Do not continue from an Opt2A or Opt1A checkpoint.

## Experiment Change

Opt2B replaces the Opt2A hard tilt threshold with one continuous risk penalty:

- Roll risk grows from `0.12` to `0.30` rad.
- Pitch risk grows from `0.22` to `0.55` rad.
- Body-frame roll-rate risk grows from `0.60` to `1.80` rad/s.
- Body-frame pitch-rate risk grows from `0.80` to `2.20` rad/s.
- The normalized weighted risk is clipped to `2.0` and uses reward weight `-0.30`.

All four risk terms are non-negative and are added before the negative TOML
weight is applied. The formula therefore cannot accidentally reward dangerous
pitch or angular velocity.

## Unchanged

Commands, domain randomization, observation noise, terrain distribution,
navigation and goal settings, all other rewards, PPO parameters, model
structure, and checkpoint naming remain identical to clean ST7.

## Validation

First run a short startup job and confirm `reward_dynamic_tilt_risk` exists, is
not permanently zero, and does not dominate episode reward. Then train the
candidate from the original `logs-595320` checkpoint and evaluate L9 using 16
environments, seed 12345, and MP4 recording.
