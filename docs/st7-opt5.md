# ST7-Opt5: L9 Hard-Segment Replay

## Baseline

- Code parent: ST7-Opt3 (`9b0b3df`).
- Checkpoint parent: ST7-Opt3 30min, evaluated as `595729`.
- Do not preload an Opt4 checkpoint.
- Training entry remains `TrackNavConfig` (`name = "nav"`).

## Only Training Change

Opt5 retains the complete Opt3 reward, PPO, model, speed, friction and goal
noise configuration. It changes only the reset distribution:

- 50% full-track starts;
- 50% hard starts;
- hard starts: 45% inverse stairs, 25% inverse slope, 20% maze entry,
  10% forward stairs.

Hard starts are placed 0.5-1.2m before the selected segment using the actual
track terrain grid origins. They receive 0.05-0.08m signed lateral offset,
3-5-degree signed yaw offset, at most 0.025rad roll/pitch offset and a small
entry-speed perturbation. Full-track environments reuse the platform's exact
evaluation-start reset path. The policy observation and network do not receive
a segment label.

After the outer environment factory returns, the workflow installs the hook
directly into the live Isaac Lab EventManager and immediately applies the
wrapped `reset_base` once to all environments. It first executes the platform
reset function, preserving the existing difficulty sampling, then changes only
the root pose/velocity and track row. Policy and critic observations are
recomputed after this initial placement. Evaluation does not use this training
workflow hook.

## Diagnostics

Every abnormal termination reports a structured `[Opt5Failure]` line with:

- env id and difficulty level;
- current segment and normalized track progress;
- all fired termination terms;
- roll, pitch, roll rate and pitch rate;
- base-contact flag, goal distance and planar speed.

The dashboard reports full/hard ratios, per-start counts, per-start success
rates, and bad-orientation/base-contact/timeout counts by hard segment.
Internal metric keys are shortened to at most 20 characters to satisfy the
platform monitor schema; their Chinese panel names retain the full meaning.

## Startup Checks

1. Preload the exact checkpoint corresponding to evaluation `595729`.
2. Confirm `[Opt5HardStart] enabled` appears after environment reset.
3. Confirm `full_ratio` and `hard_ratio` are each close to 0.50.
4. Confirm all four start counters increase and their hard-only proportions
   approach `0.45/0.25/0.20/0.10`.
5. Stop immediately if the reset hook hard check fails or if robots spawn in
   the terrain.
6. Run a short platform smoke test and confirm all four hard-start types produce
   finite observations and non-constant success/failure metrics.

Local static checks cover Python compilation, TOML parsing, monitor-key length,
and an exact Opt3 configuration comparison. The local Python environment does
not contain PyTorch, so the tensor reset unit test must run in the platform
Isaac/PyTorch environment before the 60-90 minute job.

## 2026-07-21 Startup Fix

- Split four-indicator `stat` panels into two-indicator panels to satisfy the
  platform monitor validator.
- Move reset-hook installation from the observation bridge to the live
  EventManager because the former did not initialize the runtime reset state.

## Training And Evaluation

Train for 60-90 minutes and retain 20/40/60/90-minute candidates. For each
candidate run two clean evaluations with `num_envs=128` and
`level=[0,3,6,9]`. Run two L9 video evaluations for the best candidate, then
repeat the same small-UWB-noise evaluation used for Opt3.
