# ST7-Opt5B: Conservative Hard-Start Replay

## Parent

Load the exact ST7-Opt3 30min checkpoint associated with evaluation `595729`.
Do not preload any ST7-Opt5 checkpoint or the disposable Opt5-Debug model.

The stage entry is:

```text
TrackNavOpt5BConfig
name = navopt5b
```

The historical environment file was `train_env_conf_track_navopt5b.toml`; it
has been removed from the active tree and remains available in Git history.

## Experimental Variable

Opt5B changes only training reset sampling. It keeps 75% of resets at the full
track start and assigns the remaining 25% to:

- inverse stairs: 55% of hard starts;
- inverse slope: 30% of hard starts;
- open-entry maze: 15% of hard starts.

Forward-stairs replay is disabled. Hard starts use smaller lateral, yaw, and
roll/pitch perturbations, lower entry speeds, and a 0.8-1.2m approach offset.
Spawn Z must be resolved from the terrain surface; a query failure stops the
job instead of silently using a flat-ground fallback.

## Preserved Opt3 Settings

- domain randomization and friction randomization are enabled;
- observation noise is enabled;
- Opt3 Actor-only UWB goal noise is enabled;
- jump, stale observation, and dropout simulation remain disabled;
- rewards, PPO, model structure, level mix, commands, and terrain settings are
  unchanged from the current Opt3 baseline.

## Training Gate

Train for 20 minutes first. Evaluate on 128 environments, 120 seconds, and
levels `[0, 3, 6, 9]` with hard-start replay absent from evaluation.

Continue to 40 minutes only if the 20-minute model reaches approximately:

- total score at least 63.0;
- completion at least 126/128;
- L9 completion at least 30/32.

Do not assume the latest checkpoint is the best. Stop if the 20-minute gate is
not met.
