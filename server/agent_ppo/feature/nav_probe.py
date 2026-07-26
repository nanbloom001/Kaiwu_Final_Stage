#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""nav_probe — S0a 只读环境盘点（log-once）。

用途：在 Track+Camera+256 探测任务（S0a）里，从 worker 侧 observation
process 内一次性盘点 nav 阶段设计所依赖的环境事实：

  - 实际 num_envs（平台若擅自改写 256 → 视为 S0a 失败）
  - episode 长度（120s 是否生效）
  - env.goal_positions / env.goal_yaw 的 shape、dtype、数值样本
  - env.scene.sensors 清单与 "nav_scanner" 输出规格
  - terrain 状态（地形名分布、origins 形状）
  - CUDA 显存峰值

纪律：
  - 只读。不写任何 env 状态，不修改观测。
  - log-once：每个 worker 进程只打印一次。
  - 由激活阶段 TOML 的 ``[nav_probe] enabled = true`` 门控，默认关闭。
  - 整体包在 try/except 里，探针自身失败绝不中断训练。
"""

from functools import lru_cache
from pathlib import Path

import torch

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 platform images
    tomllib = None
    import toml

_PROBE_DONE = False


@lru_cache(maxsize=8)
def _load_probe_conf(task_type: str, stage_name: str) -> dict:
    """Load the active stage TOML's [nav_probe] table once per process."""

    path = (
        Path(__file__).resolve().parent.parent
        / "conf"
        / f"train_env_conf_{task_type}_{stage_name}.toml"
    )
    try:
        if tomllib is not None:
            with path.open("rb") as stream:
                config = tomllib.load(stream)
        else:
            config = toml.load(path)
    except (OSError, TypeError, ValueError):
        return {}
    section = config.get("nav_probe", {})
    return section if isinstance(section, dict) else {}


def _enabled() -> bool:
    try:
        from agent_ppo.conf.conf import Config

        stage = Config.CURRENT
        return bool(
            _load_probe_conf(stage.task_type, stage.name).get("enabled", False)
        )
    except Exception:
        return False


def _tensor_summary(name: str, value) -> str:
    if value is None:
        return f"{name}=ABSENT"
    if not torch.is_tensor(value):
        return f"{name}=non-tensor({type(value).__name__})"
    flat = value.reshape(-1)
    sample = flat[: min(6, flat.numel())].detach().cpu().tolist()
    finite = bool(torch.isfinite(flat).all().item()) if flat.numel() else True
    return (
        f"{name}[shape={tuple(value.shape)}, dtype={value.dtype}, "
        f"finite={finite}, sample={sample}]"
    )


def _sensor_summary(sensors, name: str) -> str:
    if name not in sensors:
        return f"{name}=ABSENT"
    sensor = sensors[name]
    parts = [f"{name}[type={type(sensor).__name__}"]
    data = getattr(sensor, "data", None)
    for attr in ("ray_hits_w", "pos_w"):
        value = getattr(data, attr, None)
        if torch.is_tensor(value):
            parts.append(f"{attr}.shape={tuple(value.shape)}")
    output = getattr(data, "output", None)
    if isinstance(output, dict):
        parts.append(
            "output_keys="
            + ",".join(
                f"{k}:{tuple(v.shape)}" if torch.is_tensor(v) else str(k)
                for k, v in output.items()
            )
        )
    pattern_cfg = getattr(getattr(sensor, "cfg", None), "pattern_cfg", None)
    if pattern_cfg is not None:
        resolution = getattr(pattern_cfg, "resolution", None)
        size = getattr(pattern_cfg, "size", None)
        parts.append(f"pattern(resolution={resolution}, size={size})")
    return " ".join(parts) + "]"


def probe_once(env) -> None:
    """Log a one-shot environment inventory; never raises."""

    global _PROBE_DONE
    if _PROBE_DONE or not _enabled():
        return
    _PROBE_DONE = True
    try:
        lines = []

        num_envs = getattr(env, "num_envs", None)
        lines.append(f"actual_num_envs={num_envs}")

        episode_steps = getattr(env, "max_episode_length", None)
        step_dt = getattr(env, "step_dt", None)
        episode_s = None
        try:
            if episode_steps is not None and step_dt is not None:
                episode_s = float(episode_steps) * float(step_dt)
        except (TypeError, ValueError):
            episode_s = None
        lines.append(
            f"episode(max_length_steps={episode_steps}, step_dt={step_dt}, "
            f"approx_seconds={episode_s})"
        )

        lines.append(_tensor_summary("goal_positions", getattr(env, "goal_positions", None)))
        lines.append(_tensor_summary("goal_yaw", getattr(env, "goal_yaw", None)))

        scene = getattr(env, "scene", None)
        sensors = getattr(scene, "sensors", {}) or {}
        lines.append(f"sensor_names={sorted(sensors.keys())}")
        lines.append(_sensor_summary(sensors, "nav_scanner"))
        lines.append(_sensor_summary(sensors, "height_scanner"))

        terrain = getattr(scene, "terrain", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is not None:
            try:
                names = sorted({str(t) for t in list(terrain_types)[:512]})
            except Exception:
                names = ["<unreadable>"]
            lines.append(f"terrain_types_sample={names[:12]}")
        origins = getattr(terrain, "terrain_origins", None)
        if torch.is_tensor(origins):
            lines.append(f"terrain_origins.shape={tuple(origins.shape)}")

        if torch.cuda.is_available():
            peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
            lines.append(f"cuda_max_memory_allocated_gib={peak_gib:.2f}")

        print("[nav_probe] " + " | ".join(str(item) for item in lines), flush=True)
    except Exception as exc:  # 探针绝不中断训练
        print(f"[nav_probe] probe failed (ignored): {exc!r}", flush=True)
