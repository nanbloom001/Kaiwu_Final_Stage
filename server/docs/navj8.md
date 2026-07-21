# Stage3J-8: Hard-Level Replay

## Experiment

- Parent model: Stage3J-1 NoGate Rough-Stability 30min Best.
- Stage name: `navj8`.
- Config: `agent_ppo/conf/train_env_conf_track_navj8.toml`.
- Goal: reduce the train/evaluation difficulty-distribution gap by replaying more L7-L9 samples.

## Only Training Change

`terrain.level_mix.weights` changes from J1:

```text
[0.08, 0.09, 0.09, 0.10, 0.12, 0.14, 0.14, 0.08, 0.08, 0.08]
```

to J8:

```text
[0.06, 0.06, 0.07, 0.08, 0.10, 0.12, 0.14, 0.11, 0.12, 0.14]
```

- L7-L9: 24% -> 37%.
- L9: 8% -> 14%.
- The weights still sum to 1.0.

## Held Constant

- J8 inherits `TrackNavStage3J1Config`.
- Rewards, PPO, model, observations and action processing are unchanged.
- Speed remains `[0.50, 0.64]`.
- `curriculum` remains disabled.
- All seven NoGate switches remain disabled.
- `termination` remains `-7.0`.
- J2-J7 experimental rewards are absent, including `rough_energy`.

## Diagnostics

At startup, the workflow logs:

- stage and expected/loaded parent checkpoint;
- configured level mix and actual per-level environment counts;
- `rough_energy` configured/active state;
- termination weight and learning rate.

During training, monitoring accumulates per-level:

- episode count;
- success rate;
- `bad_orientation` count;
- `base_contact` count.

Terrain levels are captured before `env.step()` so auto-reset cannot assign an outcome to the next episode's level.

## Run And Evaluate

- Select Stage3J-1 30min Best as the platform pretrain.
- Train for 30 minutes.
- Preserve and evaluate checkpoints near 10, 20 and 30 minutes.
- Evaluate each candidate twice first with `seed=12345`, `num_envs=128`, `episode_length_s=120.0`, and levels `[0, 3, 6, 9]`.
