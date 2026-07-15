# Stage3J-5: P14 High-Pressure Completion Protection

## Experiment

- Parent model: Stage3J-1 NoGate Rough-Stability 30min best.
- Stage name: `navj5`.
- Config: `agent_ppo/conf/train_env_conf_track_navj5.toml`.
- Goal: test whether a conservative high-pressure completion bonus improves difficult successful trajectories.

## Only Training Change

- Add `difficulty_pressure_complete` with weight `8.0`.
- The reward activates only after an 80-step statistics warmup and only for completed environments whose episode energy/posture pressure exceeds the configured threshold.
- Add a required-reward startup check and a monitor panel for this reward.

## Held Constant

- J5 TOML is copied directly from J1; structured comparison differs only at `rewards.difficulty_pressure_complete`.
- NoGate switches, velocity `[0.50, 0.64]`, level mix, PPO, model structure, observation/action dimensions, and all J1 rewards are unchanged.
- Rejected J2/J3/J4 rewards are excluded.

## Package Review

- Excluded packaged `__pycache__` and `_sync_test.txt` artifacts.
- Excluded unrelated PPO learning-rate/std behavior changes to preserve a single-variable ablation.
- Kept the existing policy/critic dimension guards; the package attempted to remove them.
- Kept the correct J4 documentation instead of overwriting it with J1 comments.

## Run

- Select Stage3J-1 30min Best as the platform pretrain.
- Startup logs must show `Stage: navj5` and `train_env_conf_track_navj5.toml`.
- Confirm `reward_difficulty_pressure_complete` is registered; it is expected to remain zero during the first 80 environment steps.
