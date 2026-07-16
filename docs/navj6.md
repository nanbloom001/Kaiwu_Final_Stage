# Stage3J-6: Termination Protection

## Experiment

- Parent model: Stage3J-1 NoGate Rough-Stability 30min best.
- Stage name: `navj6`.
- Config: `agent_ppo/conf/train_env_conf_track_navj6.toml`.
- Goal: test whether a conservative increase in catastrophic termination penalty improves L9 completion.

## Only Training Change

- Change `rewards.termination.weight` from `-7.0` to `-8.0`.
- Add a startup hard check requiring the existing `termination` reward to be active.

## Held Constant

- J6 inherits `TrackNavStage3J1Config`, not J5.
- J6 TOML is copied directly from J1; structured comparison must show exactly one changed value.
- Episode length, environment count, speed `[0.50, 0.64]`, level mix, PPO, seven NoGate switches, rough-stability rewards, model, and observations remain unchanged.
- J2-J5 experimental rewards and `rough_energy` are excluded.
- `reward_process.py` and the termination formula are unchanged.

## Run

- Select Stage3J-1 30min Best as the platform pretrain.
- Startup logs must show `Stage: navj6` and `train_env_conf_track_navj6.toml`.
- Evaluate `bad_orientation`, `base_contact`, total completion, and L9 completion.
