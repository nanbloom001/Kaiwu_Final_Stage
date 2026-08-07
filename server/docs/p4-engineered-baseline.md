# P4 Engineered Baseline

## Scope

- Branch: `codex/p4-engineered-baseline`.
- Active entry: `P4NavPPOConfig` with `p4_nav_ppo` and
  `maze_instant_repair2h` from
  `agent_ppo/conf/train_env_conf_track_p4_nav_ppo.toml`.
- Active command contract: `agent_ppo.p4.contracts.command_contract`.
- Implementation baseline commit: `7b92576ac537d33d5d1e92395984ff0200b8c68b`.
- Parent package: `p4maze8h10hz_1416926.zip`.
- Parent package SHA256:
  `106908c8830f4fc7989125372f6ed397366f7ca1b4aac071128dccf71add0f6c`.
- Parent checkpoint: `ckpt/model.ckpt-mazefinal-1416926.pkl`.
- Parent checkpoint SHA256:
  `0bf54e3c18e6450492d7c1d44d69596d79beddbebb57166432220f8cb93dd2e4`.
- Frozen low-level digest:
  `8f2a214dd2a0177d8298f08711d40eb036c6eacb16d54d7cbae4f1104113e63e`.
- `maze_instant_repair2h` command digest:
  `eb1585b7539c0be454327ad05124e515e36dd7f2375806877987460e9acf331c`.
- `maze_instant_repair2h` training/reward digests:
  `ef6c6f60b176b4729f7d1a605c98cc0942f270a1dc738f775cdc4381e2f5967a` /
  `5b0f016487276afa9af8a16aa6b81b5673fff80a6c695a99ebbee8a31126ef9c`.
- `full_track` command digest:
  `648d6fac4fb0472e62a896c7417af3e3f2bd1eb1b8061b4111ef4317096d09fb`.
- `full_track` training/reward digests:
  `8d677f50ccc0c93477edc38bdad435750755ea5d17c031259fccb1fdedbf1164` /
  `67c0018ef3688030a3803db728940c960a813d88bfb83ed52c66dc76f9d139fb`.

The package and checkpoint hashes were computed from the local parent ZIP; the
low-level digest was read from the checkpoint lineage. Model IDs and labels are
selection metadata only and are not compatibility gates.

## Profile Status

| Category | Profile | Status |
| --- | --- | --- |
| Active | `maze_instant_repair2h` | Current P4 engineered baseline. |
| Active | `full_track` | Full-track active profile; select explicitly rather than retargeting the maze baseline. |
| Legacy | `maze_instant_command_r4` | Existing parent/checkpoint warm-start and eval compatibility; cannot create a new training task. |
| Legacy | `maze_closed_loop_v3` | Warm-start and eval compatibility only. |
| Legacy | `maze_credit_repair` | Warm-start and eval compatibility only. |

`instant_hold_10hz` is the active command transition mode.  The canonical
range, capability and rewrite settings come only from `p4/contracts.py`; a
model ID cannot substitute for the command-contract digest.

## Verification Layers

Run these commands from `server/`:

```bash
python3 -B -m agent_ppo.tools.verify_training --profile quick --reuse-valid
python3 -B -m agent_ppo.tools.verify_training --profile container
python3 -B -m agent_ppo.tools.verify_training --profile release --reuse-valid
```

`quick` checks the changed Python/TOML files and mapped tests. `container`
adds the container-oriented regression selection; `--nav-smoke` is opt-in and
requires a real mounted parent package. `release` compiles and parses the full
server tree and runs active tests, recording explicit quarantined historical
nodes. Successful evidence is written under ignored `.verification/` and is
reusable only when profile, Git SHA and source fingerprint match.

These checks establish local source and test evidence only. They do not prove
current development-container preload, a 128-environment GPU run, platform
lifecycle behavior, evaluation score, or real-robot safety. Record those
layers separately with the actual package SHA, command-contract digest and
runtime logs.

Run the static boundary test independently from `server/`:

```bash
python3 -B -m pytest agent_ppo/tests/test_p4_architecture_boundaries.py
```

It rejects active runtime imports from `archive` or `shared`, requires one
canonical implementation for every P4 contract while allowing only a
delegating compatibility facade, constrains profile literals to the registry,
and enforces module/function size limits with two explicit tensor-kernel
exceptions. Tests and legacy material are excluded by design.

## Evidence Status

| Layer | Status | Evidence |
| --- | --- | --- |
| Local directed tests | Passed | Each code commit was tested in an isolated worktree. |
| Local full suite | Passed | `713 passed, 5 skipped, 3 subtests`; release verifier passed. |
| Development container | Pending | Requires 1-env reset, 8-env 32-tick PPO, Adapter, save/resume and dual eval assembly. |
| Platform training/evaluation | Not run | This refactor does not claim policy-quality evidence. |
| Real robot | Not run | No Sim2Real claim is made. |

Rollback is commit-scoped: revert the newest refactor commit first; the audited
repair2h behavior snapshot remains available at `d8b0928`.

## Follow-On Branches

Create a later experiment from this baseline only after recording the parent
package SHA256, checkpoint SHA256, command-contract digest and low-level
digest. Give it a new `training_profile` and a new branch; do not silently
retarget `maze_instant_repair2h`. Any command, observation, checkpoint, ONNX
or deployment-contract change requires an atomic server/deploy/interface
update. Exact resume is allowed only when the persisted contract and digests
match; otherwise use an explicitly labelled warm start.
