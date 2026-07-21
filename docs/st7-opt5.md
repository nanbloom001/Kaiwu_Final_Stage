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

The policy observation bridge installs the reset hook into `env_cfg` before
Isaac Lab constructs its EventManager. The wrapped callback first executes the
platform reset function, preserving the existing difficulty sampling, then
changes only the root pose/velocity and track row. The initial policy and
critic observation callbacks run in the Isaac worker process and invoke the
wrapped term once if construction has not already fired it. The operation is
idempotent, so either observation-group order produces a consistent pair. This
keeps installation and initialization on the same side of Kaiwu's process
boundary. Evaluation never installs this hook.

## Diagnostics

The startup log prints `[Opt5HardStart]` with all five initial reset counts.
The dashboard reports full/hard ratios and the four cumulative hard-start
counts. These values cross the worker boundary through Isaac Lab `extras`,
rather than by attempting to unwrap the Kaiwu environment proxy. Internal
metric keys are shortened to at most 20 characters to satisfy the platform
monitor schema; their Chinese panel names retain the full meaning.

The existing workflow-side segment success/failure collectors remain
best-effort because the platform does not expose the real Isaac environment
through its cross-process proxy. Do not use an empty segment outcome panel as
evidence that no failures occurred; use the platform termination logs for the
first smoke run.

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
- Install the reset hook before EventManager construction; the workflow proxy
  does not expose the underlying Isaac Lab environment for live installation.
- Initialize the first reset batch from the policy observation callback inside
  the Isaac worker; module globals cannot cross the Kaiwu environment proxy.

## Training And Evaluation

Train for 60-90 minutes and retain 20/40/60/90-minute candidates. For each
candidate run two clean evaluations with `num_envs=128` and
`level=[0,3,6,9]`. Run two L9 video evaluations for the best candidate, then
repeat the same small-UWB-noise evaluation used for Opt3.
