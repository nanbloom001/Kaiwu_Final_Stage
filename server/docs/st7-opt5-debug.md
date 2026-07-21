# ST7-Opt5-Debug: Hard-Start Reset Audit

## Purpose

This is a disposable 5-10 minute diagnostic stage. Do not use its checkpoint
as a parent. It reloads the ST7-Opt3 30min checkpoint associated with evaluation
`595729` and keeps the original Opt5 50/50 reset distribution unchanged.

The stage entry is:

```text
TrackNavOpt5DebugConfig
name = navopt5debug
```

The environment file is
`agent_ppo/conf/train_env_conf_track_navopt5debug.toml`.

## Audit Findings

The previous implementation had two correctness defects:

1. Hard-start Z used `track_ground_z + default_root_height`, so every segment
   implicitly assumed world ground Z was zero. This can spawn robots above or
   inside slopes, stairs, and elevated track sections.
2. `entry_speeds_mps` is ordered by the five-element `track_sequence`, but the
   code indexed it using the hard-start alias ID. This assigned the wrong entry
   speed to several segments, including the 1.0m/s forward-stairs value.

The reset hook itself is real: it wraps Isaac Lab's `reset_base` event before
EventManager construction and executes in the Isaac worker. The 50/50 Bernoulli
sampling and weighted hard-segment sampling were also active. The prior outcome
collector was not reliable because it lived in the learner workflow and could
not unwrap the cross-process environment proxy.

## Debug Implementation

- Spawn surface Z is obtained by a vertical Warp ray cast at each sampled XY.
- Debug sets `require_surface_query = true`; a missing mesh or ray miss stops
  the job instead of silently falling back to `track_ground_z`.
- Entry speed is indexed by the sampled target row in `track_sequence`.
- Previous-episode outcomes are captured inside the worker immediately before
  the reset callback overwrites simulator state.
- Cumulative reset ratios and outcome counters cross the process boundary via
  Isaac Lab `extras`.

Debug-only environment changes are limited to:

```toml
[domain_rand]
enable_domain_rand = false

[noise]
add_noise = false

[goal_noise]
enabled = false
```

Rewards, PPO, commands, terrain distribution, hard-start ratio, segment weights,
offsets, and entry-speed values remain identical to Opt5.

## Required Logs

The first 16 hard starts emit `[Opt5DebugSpawn]` with:

- env ID, difficulty/track column, and target segment;
- spawn X/Y, ray-cast surface Z, root Z, and clearance;
- yaw and actual entry speed.

Each cumulative 1000-reset boundary emits `[Opt5DebugSummary]` with:

- actual full/hard ratios and four hard-start proportions;
- starts and success rate by start type;
- bad orientation, base contact, and timeout by start type;
- bad orientation, base contact, and abnormal termination within 20 steps;
- terrain surface query failure count.

The dashboard also exposes these counters plus mean spawn clearance.

## Pass Criteria

1. `actual_full_track_ratio` is close to 0.50.
2. Hard-only proportions approach `0.45/0.25/0.20/0.10`.
3. Spawn samples are 0.5-1.2m before their target segment.
4. `clearance` is close to the configured initial root height (about 0.35m).
5. `surface_query_failures` remains zero.
6. Actual speeds match the track segment values, including about 1.0m/s for
   forward stairs before jitter.
7. There is no concentration of bad orientation, base contact, or termination
   in the first 20 steps.

Only after this diagnostic passes should Opt5B be implemented from the clean
ST7-Opt3 30min checkpoint. Opt5B is not part of this branch.
