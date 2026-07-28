#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Resolve depth preprocessing settings without modifying platform base_env."""

from functools import lru_cache
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 platform images
    tomllib = None
    import toml


def _load_toml(path: Path) -> dict:
    if tomllib is not None:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    return toml.load(path)


@lru_cache(maxsize=8)
def load_depth_preprocess_conf(task_type: str, stage_name: str) -> dict:
    """Load a stage's depth-camera section once per worker process."""

    path = (
        Path(__file__).resolve().parent
        / f"train_env_conf_{task_type}_{stage_name}.toml"
    )
    try:
        config = _load_toml(path)
    except (OSError, TypeError, ValueError):
        return {}
    camera = config.get("camera", {})
    depth = camera.get("depth_camera", {}) if isinstance(camera, dict) else {}
    return depth if isinstance(depth, dict) else {}


def resolve_depth_preprocess_conf(env) -> dict:
    """Prefer an environment override, then resolve the active stage TOML."""

    override = getattr(env, "_depth_preprocess_conf", None)
    if isinstance(override, dict):
        return override

    try:
        from agent_ppo.conf.conf import Config

        stage = Config.CURRENT
        return load_depth_preprocess_conf(stage.task_type, stage.name)
    except (AttributeError, ImportError):
        return {}
