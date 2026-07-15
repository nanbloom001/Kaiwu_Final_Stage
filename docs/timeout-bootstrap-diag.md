# Timeout Bootstrap Observation Diagnostic

## Purpose

Confirm whether `env.step()` returns the terminal pre-reset observation or an
auto-reset observation for timed-out environments. This branch records logs
only and does not change PPO rewards, storage, bootstrap, or J1 rewards.

## Parent

- Code behavior: Stage3J-1 Rough-Stability.
- Pretrained model: Stage3J-1 30min Best, selected on the platform.
- Stage: `navdiag`.
- Runtime: 5 minutes; discard every checkpoint and training result.

## Diagnostic Configuration

- `num_envs = 128`
- `episode_length_s = 5.0`
- `model_save_interval = 1000`
- All other TOML values are identical to `train_env_conf_track_navj1.toml`.

## Logged Evidence

At each timeout, compare policy observation slot `obs[:, 33:45]` (last action)
with the action just sent to `env.step()` and with zero. At most eight records
are emitted under `[TimeoutObsCheck]`.

The timeout mask is taken directly from the `truncated` tensor returned by
`env.step()`. It does not depend on an optional `infos["time_outs"]` field. A
single `[TimeoutObsCheckStart]` INFO record confirms that the diagnostic path
is active and reports the available `infos` keys.

## Interpretation

- `match_sent_action_mae << match_zero_mae` and `sent_closer_ratio` near 1:
  the returned observation likely still represents the terminal pre-reset
  state. Evaluating `V(s_{t+1})` is then a valid candidate.
- `match_zero_mae << match_sent_action_mae` and `sent_closer_ratio` near 0:
  the returned observation likely belongs to the auto-reset state. Do not use
  that observation for timeout bootstrap.
- Non-empty `terminal_keys`: inspect those fields first. A dedicated terminal
  or final observation is preferable to inference from the returned `obs`.
- Mixed or close MAE values: the last-action probe is inconclusive; keep the
  existing bootstrap and add a second state probe before changing PPO.

Do not promote this branch or use its model as a parent.
