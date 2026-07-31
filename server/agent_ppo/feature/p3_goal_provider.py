#!/usr/bin/env python3
"""Worker-owned P3 subgoals that do not activate platform goal termination."""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import torch
from agent_ppo.feature import p3_contract

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - platform fallback
    tomllib = None
    import toml


@lru_cache(maxsize=1)
def _goal_config():
    path = (
        Path(__file__).resolve().parent.parent
        / "conf"
        / "train_env_conf_standard_p3_standard_joint.toml"
    )
    try:
        if tomllib is not None:
            with path.open("rb") as stream:
                root = tomllib.load(stream)
        else:  # pragma: no cover
            root = toml.load(path)
    except (OSError, TypeError, ValueError):
        return {}
    section = root.get("p3_standard_joint", {})
    return section if isinstance(section, dict) else {}


def _robot(env):
    scene = getattr(env, "scene", None)
    try:
        return scene["robot"]
    except Exception:
        robot = getattr(scene, "robot", None)
        if robot is None:
            raise RuntimeError("P3 goals require the robot articulation")
        return robot


def _env_origins(env, root_xy):
    scene = getattr(env, "scene", None)
    for value in (
        getattr(scene, "env_origins", None),
        getattr(getattr(scene, "terrain", None), "env_origins", None),
    ):
        if torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == root_xy.shape[0]:
            return value[:, :2].to(root_xy)
    return root_xy.clone()


class StandardFarGoalProvider:
    def __init__(self, env, *, config=None, seed=None):
        self.env = env
        self.config = dict(_goal_config() if config is None else config)
        self.num_envs = int(env.num_envs)
        self.device = torch.device(env.device)
        configured_seed = self.config.get("subgoal_seed", self.config.get("seed", 3107))
        self.generator = torch.Generator(device=self.device).manual_seed(
            int(configured_seed if seed is None else seed)
        )
        self.goal_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.tile_origin_xy = torch.zeros_like(self.goal_xy)
        self.goal_age_s = torch.zeros(self.num_envs, device=self.device)
        self.goal_timeout_s = torch.zeros(self.num_envs, device=self.device)
        self.goal_epoch = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.subgoal_success_count = torch.zeros_like(self.goal_epoch)
        self.last_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_goal_changed = torch.zeros_like(self.last_reached)
        self.pending_resample = torch.zeros_like(self.last_reached)
        self.last_timed_out = torch.zeros_like(self.last_reached)
        self.sample_invalid = torch.zeros_like(self.last_reached)
        self._initialized = False
        self._last_step_key = None

    def _step_key(self):
        for name in ("common_step_counter", "_sim_step_counter"):
            value = getattr(self.env, name, None)
            if value is not None:
                try:
                    return (name, int(value))
                except (TypeError, ValueError):
                    pass
        lengths = getattr(self.env, "episode_length_buf", None)
        if torch.is_tensor(lengths):
            return ("episode", int(lengths.long().sum().item()))
        return None

    def _sample(self, ids, root_xy):
        if ids.numel() == 0:
            return
        distance = self.config.get(
            "subgoal_distance_m",
            [p3_contract.SUBGOAL_MIN_DISTANCE_M, p3_contract.SUBGOAL_MAX_DISTANCE_M],
        )
        goals, valid = p3_contract.sample_local_subgoals_with_validity(
            root_xy[ids],
            self.tile_origin_xy[ids],
            generator=self.generator,
            min_distance_m=float(distance[0]),
            max_distance_m=float(distance[1]),
            tile_inner_margin_m=float(
                self.config.get("tile_inner_margin_m", p3_contract.TILE_INNER_MARGIN_M)
            ),
            max_attempts=64,
        )
        self.goal_xy[ids] = goals
        self.sample_invalid[ids] = ~valid
        if bool((~valid).any()):
            invalid_ids = ids[~valid]
            # Platform-owned BaseEnv cannot be patched to force a reset. Use
            # the tile origin as a no-success recovery target until the robot
            # re-enters the configured local bound, then resume strict sampling.
            self.goal_xy[invalid_ids] = self.tile_origin_xy[invalid_ids]
        timeout_range = self.config.get("subgoal_time_limit_s", [6.0, 12.0])
        timeout = torch.empty(ids.numel(), device=self.device)
        timeout.uniform_(
            float(timeout_range[0]), float(timeout_range[1]), generator=self.generator
        )
        self.goal_timeout_s[ids] = timeout
        self.goal_age_s[ids] = 0.0
        self.goal_epoch[ids] += 1
        self.last_goal_changed[ids] = True

    def update(self, *, dt_s):
        step_key = self._step_key()
        if step_key is not None and step_key == self._last_step_key:
            return
        self._last_step_key = step_key
        robot = _robot(self.env)
        root_xy = robot.data.root_pos_w[:, :2].to(self.device)
        lengths = getattr(self.env, "episode_length_buf", None)
        reset = (
            torch.as_tensor(lengths, device=self.device).reshape(-1) == 0
            if torch.is_tensor(lengths)
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        if not self._initialized:
            reset.fill_(True)
            self._initialized = True
        self.last_goal_changed.zero_()
        self.last_reached.zero_()
        self.last_timed_out.zero_()
        pending = self.pending_resample.clone()
        self.pending_resample.zero_()
        if bool(reset.any()):
            self.tile_origin_xy[reset] = _env_origins(self.env, root_xy)[reset]
            self.subgoal_success_count[reset] = 0
            self.goal_epoch[reset] = 0
            pending[reset] = False
        local_out = p3_contract.local_out_of_bounds(
            root_xy,
            self.tile_origin_xy,
            threshold_m=float(
                self.config.get(
                    "tile_reset_local_abs_m", p3_contract.TILE_RESET_LOCAL_ABS_M
                )
            ),
        )
        pending |= self.sample_invalid & ~local_out
        if bool(pending.any()):
            self._sample(pending.nonzero(as_tuple=False).reshape(-1), root_xy)
        reached = p3_contract.subgoal_reached(
            root_xy,
            self.goal_xy,
            threshold_m=float(
                self.config.get(
                    "subgoal_success_distance_m",
                    p3_contract.SUBGOAL_SUCCESS_DISTANCE_M,
                )
            ),
        ) & ~reset & ~self.sample_invalid
        self.last_reached.copy_(reached)
        self.subgoal_success_count += reached.long()
        self.goal_age_s += float(dt_s)
        timed_out = (
            (self.goal_age_s >= self.goal_timeout_s)
            & ~reset
            & ~reached
        )
        self.last_timed_out.copy_(timed_out)
        # Keep the completed/expired goal visible for one observation so the
        # aisrv can attribute the event to the transition that caused it.
        self.pending_resample |= reached | timed_out
        self._sample(reset.nonzero(as_tuple=False).reshape(-1), root_xy)
        self.env._p3_goal_positions = torch.cat(
            (self.goal_xy, robot.data.root_pos_w[:, 2:3].detach()), dim=-1
        )
        self.env._p3_subgoal_reached = self.last_reached.clone()
        self.env._p3_subgoal_success_count = self.subgoal_success_count.clone()
        self.env._p3_goal_epoch = self.goal_epoch.clone()
        self.env._p3_goal_changed = self.last_goal_changed.clone()
        self.env._p3_subgoal_timed_out = self.last_timed_out.clone()
        self.env._p3_local_out_of_bounds = local_out | self.sample_invalid


def p3_local_out_of_bounds(env):
    provider = getattr(env, "_p3_goal_provider", None)
    if isinstance(provider, StandardFarGoalProvider) and provider._initialized:
        root_xy = _robot(env).data.root_pos_w[:, :2].to(provider.device)
        value = p3_contract.local_out_of_bounds(
            root_xy,
            provider.tile_origin_xy,
            threshold_m=float(
                provider.config.get(
                    "tile_reset_local_abs_m",
                    p3_contract.TILE_RESET_LOCAL_ABS_M,
                )
            ),
        ) | provider.sample_invalid
        return value.to(env.device).reshape(-1).bool()
    value = getattr(env, "_p3_local_out_of_bounds", None)
    if torch.is_tensor(value) and value.numel() == int(env.num_envs):
        return value.to(env.device).reshape(-1).bool()
    return torch.zeros(int(env.num_envs), dtype=torch.bool, device=env.device)


def update_p3_goal_provider(env, *, dt_s):
    provider = getattr(env, "_p3_goal_provider", None)
    if not isinstance(provider, StandardFarGoalProvider):
        provider = StandardFarGoalProvider(env)
        env._p3_goal_provider = provider
    provider.update(dt_s=dt_s)
    return provider
