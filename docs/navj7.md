# Stage3J-7: Rough Energy

## Experiment

- Parent model: Stage3J-1 NoGate Rough-Stability 30min best.
- Stage name: `navj7`.
- Config: `agent_ppo/conf/train_env_conf_track_navj7.toml`.
- Goal: test a light energy constraint that activates only when the forward height scan indicates rough terrain.

## Only Training Change

- Add `rough_energy` with weight `-6.0e-6`.
- Use the same rough scan region and thresholds as J1: rows 4-12, columns 1-10, quantile 0.85, delta range 0.025-0.12.
- Add a startup hard check requiring `rough_energy` to be active.

## Implementation Check

- Existing `_reward_rough_energy()` computes `rough_gate * sum(abs(torque * joint_velocity))`.
- `rough_gate` comes from local `height_scanner` deltas only; it does not use terrain level or terrain names.
- `reward_process.py` is unchanged.

## Held Constant

- J7 inherits `TrackNavStage3J1Config`, not J4/J5/J6.
- Termination remains `-7.0`.
- Speed `[0.50, 0.64]`, environment count, episode length, level mix, PPO, completion rewards, seven NoGate switches, model, and observations remain identical to J1.
- J2-J5 experimental rewards are excluded.

## Run

- Select Stage3J-1 30min Best as the platform pretrain.
- Startup logs must show `Stage: navj7` and `train_env_conf_track_navj7.toml`.
