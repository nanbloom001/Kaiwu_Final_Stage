# -*- coding: UTF-8 -*-

import torch


class GoalObservationMixin:
    """Local goal-observation helpers for TrackNav observation processors."""

    @staticmethod
    def _wrap_to_pi(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def concatenate_terms(*terms):
        return torch.cat(terms, dim=-1)

    def _get_goal_env(self):
        seen_ids = set()
        pending = [getattr(self, "env", None)]
        fallback = None
        while pending:
            candidate = pending.pop(0)
            if candidate is None or id(candidate) in seen_ids:
                continue
            seen_ids.add(id(candidate))
            if hasattr(candidate, "goal_positions"):
                return candidate
            if fallback is None and hasattr(candidate, "scene") and hasattr(candidate, "num_envs"):
                fallback = candidate
            for attr_name in (
                "env",
                "_env",
                "unwrapped",
                "wrapped_env",
                "_wrapped_env",
                "venv",
                "isaac_env",
                "_isaac_env",
                "sim_env",
                "_sim_env",
                "task",
                "_task",
            ):
                pending.append(getattr(candidate, attr_name, None))
        return fallback if fallback is not None else getattr(self, "env", None)

    @staticmethod
    def _has_robot_state(asset):
        data = getattr(asset, "data", None)
        return data is not None and hasattr(data, "root_pos_w")

    def _get_robot_asset(self, env):
        scene = getattr(env, "scene", None)
        if scene is None:
            return None

        if hasattr(scene, "__getitem__"):
            scene_keys = []
            keys_fn = getattr(scene, "keys", None)
            if callable(keys_fn):
                try:
                    scene_keys = list(keys_fn())
                except Exception:
                    scene_keys = []
            for key in ("robot", "Robot", "go2", "Go2", "unitree_go2", "UnitreeGo2", *scene_keys):
                try:
                    asset = scene[key]
                except Exception:
                    asset = None
                if self._has_robot_state(asset):
                    return asset

        for container_name in (
            "articulations",
            "_articulations",
            "rigid_objects",
            "_rigid_objects",
            "entities",
            "_entities",
        ):
            container = getattr(scene, container_name, None)
            if container is None:
                continue
            if hasattr(container, "get"):
                for key in ("robot", "Robot", "go2", "Go2", "unitree_go2", "UnitreeGo2"):
                    asset = container.get(key)
                    if self._has_robot_state(asset):
                        return asset
            values = getattr(container, "values", None)
            if callable(values):
                for asset in values():
                    if self._has_robot_state(asset):
                        return asset

        for attr_name in ("robot", "_robot"):
            asset = getattr(env, attr_name, None)
            if self._has_robot_state(asset):
                return asset

        return None

    @staticmethod
    def _get_heading_w(asset, num_envs, device):
        data = getattr(asset, "data", None)
        if data is not None and hasattr(data, "heading_w"):
            return data.heading_w.to(device=device)
        if data is not None and hasattr(data, "root_quat_w"):
            quat = data.root_quat_w.to(device=device)
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.zeros(num_envs, device=device)

    def goal_position_in_robot_frame(self):
        env = self._get_goal_env()
        goal_positions = getattr(env, "goal_positions", None)

        device = getattr(env, "device", None)
        if device is None and goal_positions is not None:
            device = goal_positions.device
        if device is None:
            device = torch.device("cpu")

        num_envs = int(getattr(env, "num_envs", 0))
        if goal_positions is not None:
            num_envs = int(goal_positions.shape[0])

        zeros = torch.zeros(num_envs, 4, device=device)
        if goal_positions is None:
            return zeros

        asset = self._get_robot_asset(env)
        if asset is None:
            return zeros

        robot_pos = asset.data.root_pos_w[:, :2].to(device=device)
        goal_pos = goal_positions[:, :2].to(device=device)
        goal_vec_w = goal_pos - robot_pos
        heading = self._get_heading_w(asset, num_envs, device)

        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        goal_x = cos_h * goal_vec_w[:, 0] + sin_h * goal_vec_w[:, 1]
        goal_y = -sin_h * goal_vec_w[:, 0] + cos_h * goal_vec_w[:, 1]

        distance = torch.norm(goal_vec_w, dim=1)
        inv_distance = torch.reciprocal(distance.clamp_min(1.0e-6))
        dir_x = goal_x * inv_distance
        dir_y = goal_y * inv_distance

        goal_yaw = getattr(env, "goal_yaw", None)
        if goal_yaw is not None:
            yaw_error = self._wrap_to_pi(goal_yaw.to(device=device) - heading)
        else:
            yaw_error = self._wrap_to_pi(torch.atan2(goal_y, goal_x))

        # Normalize distance to [0, 1] with a 10 m reference so the network
        # input stays in the same scale as other obs dimensions (velocities,
        # angles). Values beyond 10 m saturate at 1.0 rather than blowing up.
        norm_distance = distance / 10.0

        # 3-dim goal obs: (local_x, local_y, distance).
        # Heading alignment is handled by reward shaping, not obs redundancy.
        return torch.stack((dir_x, dir_y, norm_distance), dim=1)
