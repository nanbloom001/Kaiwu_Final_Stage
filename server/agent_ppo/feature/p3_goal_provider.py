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


def _terrain_size_x(env) -> float:
    terrain = getattr(getattr(env, "scene", None), "terrain", None)
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    size = getattr(generator, "size", None)
    try:
        value = float(size[0])
    except (TypeError, ValueError, IndexError):
        value = 2.0 * p3_contract.TILE_HALF_EXTENT_M
    return value


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
        self.episode_origin_xy = torch.zeros_like(self.goal_xy)
        self.base_direction_angle = torch.zeros(self.num_envs, device=self.device)
        self.direction_angle = torch.zeros_like(self.base_direction_angle)
        self.milestone_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.replan_count = torch.zeros_like(self.milestone_index)
        self.best_radius = torch.zeros(self.num_envs, device=self.device)
        self.m3_proxy_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.goal_age_s = torch.zeros(self.num_envs, device=self.device)
        self.goal_timeout_s = torch.zeros(self.num_envs, device=self.device)
        self.goal_epoch = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.subgoal_success_count = torch.zeros_like(self.goal_epoch)
        self.last_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_goal_changed = torch.zeros_like(self.last_reached)
        self.pending_resample = torch.zeros_like(self.last_reached)
        self.pending_replan = torch.zeros_like(self.last_reached)
        self.last_timed_out = torch.zeros_like(self.last_reached)
        self.sample_invalid = torch.zeros_like(self.last_reached)
        self._initialized = False
        self._last_step_key = None
        (
            self.complete_radius_m,
            self.m3_target_radius_m,
            self.boundary_radius_m,
        ) = p3_contract.platform_completion_radii(_terrain_size_x(env))

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

    def _direction_is_valid(self, ids, angle):
        unit = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        target = self.episode_origin_xy[ids] + unit * self.m3_target_radius_m
        local = torch.abs(target - self.tile_origin_xy[ids])
        return (local <= self.boundary_radius_m - p3_contract.M3_BOUNDARY_MARGIN_M).all(
            dim=-1
        )

    def _choose_direction(self, ids, *, replan):
        if ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        chosen = self.direction_angle[ids].clone()
        valid = torch.zeros(ids.numel(), dtype=torch.bool, device=self.device)
        offsets = torch.tensor(
            p3_contract.REPLAN_ANGLE_OFFSETS_DEG,
            device=self.device,
            dtype=chosen.dtype,
        ) * (torch.pi / 180.0)
        for _ in range(64):
            pending = ~valid
            if not bool(pending.any()):
                break
            pending_ids = pending.nonzero(as_tuple=False).flatten()
            if replan:
                choice = torch.randint(
                    offsets.numel(),
                    (pending_ids.numel(),),
                    generator=self.generator,
                    device=self.device,
                )
                candidate = self.base_direction_angle[ids[pending_ids]] + offsets[choice]
            else:
                candidate = torch.empty(
                    pending_ids.numel(), device=self.device, dtype=chosen.dtype
                ).uniform_(-torch.pi, torch.pi, generator=self.generator)
            accepted = self._direction_is_valid(ids[pending_ids], candidate)
            if bool(accepted.any()):
                accepted_slots = pending_ids[accepted]
                chosen[accepted_slots] = candidate[accepted]
                valid[accepted_slots] = True
        self.direction_angle[ids] = chosen
        if not replan:
            self.base_direction_angle[ids] = chosen
        return valid

    def _sample(self, ids, *, replan=False, new_direction=False):
        if ids.numel() == 0:
            return
        if replan or new_direction:
            valid = self._choose_direction(ids, replan=bool(replan))
        else:
            valid = self._direction_is_valid(ids, self.direction_angle[ids])
        milestone = self.milestone_index[ids]
        radius = torch.full(
            (ids.numel(),), self.m3_target_radius_m, device=self.device
        )
        for index, bounds in enumerate(p3_contract.MILESTONE_RADIUS_RANGES_M):
            selected = milestone == index
            if bool(selected.any()):
                radius[selected] = torch.empty(
                    int(selected.sum()), device=self.device
                ).uniform_(float(bounds[0]), float(bounds[1]), generator=self.generator)
        unit = torch.stack(
            (torch.cos(self.direction_angle[ids]), torch.sin(self.direction_angle[ids])),
            dim=-1,
        )
        self.goal_xy[ids] = self.episode_origin_xy[ids] + unit * radius.unsqueeze(-1)
        self.sample_invalid[ids] = ~valid
        if bool((~valid).any()):
            invalid_ids = ids[~valid]
            self.goal_xy[invalid_ids] = self.episode_origin_xy[invalid_ids]
        timeout_range = self.config.get("subgoal_time_limit_s", [6.0, 12.0])
        timeout = torch.empty(ids.numel(), device=self.device)
        timeout.uniform_(
            float(timeout_range[0]), float(timeout_range[1]), generator=self.generator
        )
        self.goal_timeout_s[ids] = timeout
        self.goal_age_s[ids] = 0.0
        self.goal_epoch[ids] += 1
        if replan:
            self.replan_count[ids] += 1
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
        pending_replan = self.pending_replan.clone()
        self.pending_resample.zero_()
        self.pending_replan.zero_()
        if bool(reset.any()):
            self.tile_origin_xy[reset] = _env_origins(self.env, root_xy)[reset]
            self.episode_origin_xy[reset] = root_xy[reset]
            self.subgoal_success_count[reset] = 0
            self.goal_epoch[reset] = 0
            self.milestone_index[reset] = 0
            self.replan_count[reset] = 0
            self.best_radius[reset] = 0.0
            self.m3_proxy_latched[reset] = False
            self.sample_invalid[reset] = False
            pending[reset] = False
            pending_replan[reset] = False
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
            retry_ids = pending.nonzero(as_tuple=False).reshape(-1)
            replan_ids = retry_ids[pending_replan[retry_ids]]
            invalid_ids = retry_ids[
                ~pending_replan[retry_ids] & self.sample_invalid[retry_ids]
            ]
            advance_ids = retry_ids[
                ~pending_replan[retry_ids] & ~self.sample_invalid[retry_ids]
            ]
            self._sample(advance_ids)
            self._sample(invalid_ids, new_direction=True)
            self._sample(replan_ids, replan=True)
        radius = p3_contract.radial_distance(root_xy, self.episode_origin_xy)
        self.best_radius.copy_(torch.maximum(self.best_radius, radius))
        local_reached = p3_contract.subgoal_reached(
            root_xy,
            self.goal_xy,
            threshold_m=float(
                self.config.get(
                    "subgoal_success_distance_m",
                    p3_contract.SUBGOAL_SUCCESS_DISTANCE_M,
                )
            ),
        )
        m3_reached = (radius >= self.complete_radius_m) & (self.milestone_index >= 2)
        reached = torch.where(self.milestone_index >= 2, m3_reached, local_reached)
        reached &= ~reset & ~self.sample_invalid & ~self.m3_proxy_latched
        self.last_reached.copy_(reached)
        self.subgoal_success_count += reached.long()
        m3_new = reached & (self.milestone_index >= 2)
        self.m3_proxy_latched |= m3_new
        advance = reached & ~m3_new
        self.milestone_index[advance] += 1
        self.goal_age_s += float(dt_s)
        timed_out = (
            (self.goal_age_s >= self.goal_timeout_s)
            & ~reset
            & ~reached
            & ~self.m3_proxy_latched
        )
        self.last_timed_out.copy_(timed_out)
        # Keep the completed/expired goal visible for one observation so the
        # aisrv can attribute the event to the transition that caused it.
        self.pending_resample |= advance | timed_out
        self.pending_replan |= timed_out
        self._sample(
            reset.nonzero(as_tuple=False).reshape(-1), new_direction=True
        )
        self.env._p3_goal_positions = torch.cat(
            (self.goal_xy, robot.data.root_pos_w[:, 2:3].detach()), dim=-1
        )
        self.env._p3_subgoal_reached = self.last_reached.clone()
        self.env._p3_subgoal_success_count = self.subgoal_success_count.clone()
        self.env._p3_goal_epoch = self.goal_epoch.clone()
        self.env._p3_goal_changed = self.last_goal_changed.clone()
        self.env._p3_subgoal_timed_out = self.last_timed_out.clone()
        self.env._p3_local_out_of_bounds = local_out | self.sample_invalid
        self.env._p3_episode_origin_xy = self.episode_origin_xy.clone()
        self.env._p3_radial_distance = radius.clone()
        self.env._p3_best_radial_distance = self.best_radius.clone()
        self.env._p3_milestone_index = self.milestone_index.clone()
        self.env._p3_m3_proxy_success = self.m3_proxy_latched.clone()
        self.env._p3_goal_replan_count = self.replan_count.clone()
        self.env._p3_platform_complete_radius = float(self.complete_radius_m)
        self.env._p3_m3_target_radius = float(self.m3_target_radius_m)
        self.env._p3_platform_boundary_radius = float(self.boundary_radius_m)


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
