# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Helpers for unwrapping Isaac Lab env objects and reading runtime physics stats."""

import torch


def get_isaac_env(env):
    seen_ids = set()
    pending = [env]
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen_ids:
            continue
        seen_ids.add(id(candidate))
        if hasattr(candidate, "command_manager") and hasattr(candidate, "scene"):
            return candidate
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
    return None


def _has_robot_state(asset) -> bool:
    data = getattr(asset, "data", None)
    return (
        data is not None
        and hasattr(data, "root_lin_vel_b")
        and hasattr(data, "root_pos_w")
        and hasattr(data, "root_ang_vel_b")
    )


def get_robot_asset_from_env(isaac_env):
    if isaac_env is None:
        return None
    scene = getattr(isaac_env, "scene", None)
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
            if _has_robot_state(asset):
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
                if _has_robot_state(asset):
                    return asset
        values = getattr(container, "values", None)
        if callable(values):
            for asset in values():
                if _has_robot_state(asset):
                    return asset

    for attr_name in ("robot", "_robot"):
        asset = getattr(isaac_env, attr_name, None)
        if _has_robot_state(asset):
            return asset
    return None


def sample_physics_stats_from_critic_obs(critic_obs):
    if critic_obs is None or not hasattr(critic_obs, "shape") or critic_obs.shape[-1] < 12:
        return {}
    actual_vx = critic_obs[:, 0]
    actual_vy = critic_obs[:, 1]
    actual_yaw = critic_obs[:, 5]
    cmd_vx = critic_obs[:, 9]
    cmd_vy = critic_obs[:, 10]
    cmd_yaw = critic_obs[:, 11]
    return {
        "obs_cmd_vel_x": cmd_vx.mean().item(),
        "obs_cmd_vel_y": cmd_vy.mean().item(),
        "obs_cmd_yaw": cmd_yaw.mean().item(),
        "obs_lin_vel_x_error": torch.abs(actual_vx - cmd_vx).mean().item(),
        "obs_lin_vel_y_error": torch.abs(actual_vy - cmd_vy).mean().item(),
        "obs_yaw_error": torch.abs(actual_yaw - cmd_yaw).mean().item(),
        "obs_actual_vel_x": actual_vx.mean().item(),
        "obs_actual_vel_y": actual_vy.mean().item(),
        "obs_actual_yaw": actual_yaw.mean().item(),
        "obs_ang_vel_xy": torch.norm(critic_obs[:, 3:5], dim=1).mean().item(),
    }


def sample_physics_stats(env, logger=None, critic_obs=None):
    try:
        isaac_env = get_isaac_env(env)
        if isaac_env is None:
            if logger is not None and not getattr(env, "_physics_stats_error_logged", False):
                env._physics_stats_error_logged = True
                logger.warning(
                    "[PhysicsStats] Failed to unwrap Isaac Lab env; "
                    "falling back to critic_obs for partial physics metrics."
                )
            return sample_physics_stats_from_critic_obs(critic_obs)

        cmd = isaac_env.command_manager.get_command("base_velocity")
        asset = get_robot_asset_from_env(isaac_env)
        if asset is None:
            if logger is not None and not getattr(env, "_physics_stats_error_logged", False):
                env._physics_stats_error_logged = True
                logger.warning(
                    "[PhysicsStats] Failed to locate robot asset in scene; "
                    "falling back to critic_obs for partial physics metrics."
                )
            return sample_physics_stats_from_critic_obs(critic_obs)

        actual_vx = asset.data.root_lin_vel_b[:, 0]
        actual_vy = asset.data.root_lin_vel_b[:, 1]
        actual_yaw = asset.data.root_ang_vel_b[:, 2]
        cmd_vx = cmd[:, 0]
        cmd_vy = cmd[:, 1]
        cmd_yaw = cmd[:, 2]
        return {
            "obs_cmd_vel_x": cmd_vx.mean().item(),
            "obs_cmd_vel_y": cmd_vy.mean().item(),
            "obs_cmd_yaw": cmd_yaw.mean().item(),
            "obs_lin_vel_x_error": torch.abs(actual_vx - cmd_vx).mean().item(),
            "obs_lin_vel_y_error": torch.abs(actual_vy - cmd_vy).mean().item(),
            "obs_yaw_error": torch.abs(actual_yaw - cmd_yaw).mean().item(),
            "obs_actual_vel_x": actual_vx.mean().item(),
            "obs_actual_vel_y": actual_vy.mean().item(),
            "obs_actual_yaw": actual_yaw.mean().item(),
            "obs_base_height": asset.data.root_pos_w[:, 2].mean().item(),
            "obs_ang_vel_xy": torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1).mean().item(),
        }
    except Exception as exc:
        if logger is not None and not getattr(env, "_physics_stats_error_logged", False):
            env._physics_stats_error_logged = True
            logger.warning(
                f"[PhysicsStats] Failed to sample physics metrics from Isaac env: {exc}; "
                "falling back to critic_obs for partial physics metrics."
            )
        return sample_physics_stats_from_critic_obs(critic_obs)


def set_env_base_velocity_command(env, command, logger=None):
    isaac_env = get_isaac_env(env)
    if isaac_env is None or not hasattr(isaac_env, "command_manager"):
        if logger is not None and not getattr(env, "_nav_command_error_logged", False):
            env._nav_command_error_logged = True
            logger.warning("[Navigation] Cannot unwrap Isaac env; only observation command will be overridden.")
        return False

    try:
        current_command = isaac_env.command_manager.get_command("base_velocity")
        if current_command.shape != command.shape:
            if logger is not None and not getattr(env, "_nav_command_error_logged", False):
                env._nav_command_error_logged = True
                logger.warning(
                    "[Navigation] base_velocity command shape mismatch: "
                    f"env={tuple(current_command.shape)}, nav={tuple(command.shape)}."
                )
            return False
        current_command.copy_(command.to(device=current_command.device, dtype=current_command.dtype))
        return True
    except Exception as exc:
        if logger is not None and not getattr(env, "_nav_command_error_logged", False):
            env._nav_command_error_logged = True
            logger.warning(f"[Navigation] Failed to override base_velocity command: {exc}")
        return False
