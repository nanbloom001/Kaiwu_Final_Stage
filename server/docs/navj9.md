# Stage3J-9: Fixed Learning Rate Control

## Experiment

- Parent model: Stage3J-1 30min Best.
- Stage name: `navj9`.
- Historical config: `train_env_conf_track_navj9.toml` (removed from the active tree; recover from Git history).
- Goal: determine whether a truly fixed PPO learning rate reduces L9 policy drift.

## Only Training Change

J1 declared a fixed `1e-5` range, but the PPO constructor did not receive the
stage schedule and silently used its default adaptive schedule. The configured
learning-rate bounds were also swallowed by `**kwargs`, while the adaptive
update used hard-coded bounds.

J9 makes the intended contract effective:

```text
learning_rate = 1e-5
schedule = fixed
min_learning_rate = 1e-5
max_learning_rate = 1e-5
```

The PPO interface now accepts the configured bounds explicitly. This fixes the
interface for later stages, while J9 itself does not enter the adaptive branch.

## Held Constant

- J9 inherits `TrackNavStage3J1Config` directly.
- The J9 TOML is structurally identical to J1.
- Rewards, J1 level mix, commands, NoGate switches and model structure are unchanged.
- Exploration standard-deviation behavior is unchanged.
- J7 `rough_energy` and J8 hard-level replay are not enabled.

## Runtime Checks

Startup validation requires the J1 PPO parameters, J1 level mix, fixed schedule,
and `1e-5` learning rate in both the algorithm and optimizer. Startup logs include
the expected and loaded parent, schedule, bounds, entropy coefficient, desired KL,
and level mix.

After every PPO update, training aborts if either the algorithm learning rate or
an optimizer parameter-group learning rate differs from `1e-5`. The existing
`learning_rate` monitor should therefore remain `0.00001` for the entire run.

## Run

Select Stage3J-1 30min Best as the platform pretrain. Do not let the platform
auto-select the latest J8 checkpoint.
