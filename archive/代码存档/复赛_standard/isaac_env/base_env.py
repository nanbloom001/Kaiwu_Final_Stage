"""
Robot 强化学习环境与评估录像相机控制入口。

<!-- @changelog -->
<table>
  <tr><th>版本</th><th>变更说明</th><th>关联</th></tr>
  <tr>
    <td>v1.2.0</td>
    <td>二次收敛评估录像机位，增加后撤余量以保证机器狗完整入镜</td>
    <td>REQ: 20260423-评估录像视角优化<br/>TECH: 04_design_tech-design.md §3.4</td>
  </tr>
  <tr>
    <td>v1.1.0</td>
    <td>收敛评估录像相机参数并加入平滑跟随视角</td>
    <td>REQ: 20260423-评估录像视角优化<br/>TECH: 04_design_tech-design.md §3.4</td>
  </tr>
</table>
<!-- /@changelog -->

@author root
"""

import argparse
import copy
import datetime
import json
import logging
import math
import os
import pathlib
import subprocess
import sys

import torch


DEFAULT_EVAL_CAMERA_EYE_OFFSET = (-1.7, 0.0, 1.5)
DEFAULT_EVAL_CAMERA_TARGET_OFFSET = (0.3, 0.0, 0.25)
DEFAULT_EVAL_CAMERA_SMOOTHING_ALPHA = 0.2


def get_default_eval_camera_profile(device: str | torch.device = "cpu") -> dict[str, torch.Tensor | float]:
    """Return the default evaluation camera profile for MP4 recording."""
    return {
        "eye_offset": torch.tensor(DEFAULT_EVAL_CAMERA_EYE_OFFSET, dtype=torch.float32, device=device),
        "target_offset": torch.tensor(DEFAULT_EVAL_CAMERA_TARGET_OFFSET, dtype=torch.float32, device=device),
        "smoothing_alpha": DEFAULT_EVAL_CAMERA_SMOOTHING_ALPHA,
    }


def get_eval_camera_profile_metrics(
    eye_offset: tuple[float, float, float] | torch.Tensor,
    target_offset: tuple[float, float, float] | torch.Tensor,
) -> dict[str, float]:
    """Summarize evaluation camera composition metrics for logs and regression tests."""
    eye_vector = torch.as_tensor(eye_offset, dtype=torch.float32).flatten()
    target_vector = torch.as_tensor(target_offset, dtype=torch.float32).flatten()
    if eye_vector.numel() != 3 or target_vector.numel() != 3:
        raise ValueError("camera profile metrics expect 3D eye/target offsets")

    view_vector = target_vector - eye_vector
    horizontal_distance = float(torch.linalg.vector_norm(view_vector[:2]).item())
    vertical_drop = float((eye_vector[2] - target_vector[2]).item())
    pitch_deg = math.degrees(math.atan2(vertical_drop, horizontal_distance))
    return {
        "rear_follow_distance": float((-eye_vector[0]).item()),
        "look_ahead_distance": float(target_vector[0].item()),
        "eye_height": float(eye_vector[2].item()),
        "target_height": float(target_vector[2].item()),
        "horizontal_distance": horizontal_distance,
        "vertical_drop": vertical_drop,
        "pitch_deg": pitch_deg,
    }


def _expand_camera_offset(offset: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Expand a single camera offset to match the batch size."""
    if offset.ndim == 1:
        offset = offset.unsqueeze(0)
    if offset.shape[0] == 1:
        offset = offset.expand(batch_size, -1)
    if offset.shape[0] != batch_size:
        raise ValueError(f"camera offset batch mismatch: expected {batch_size}, got {offset.shape[0]}")
    return offset


def _quat_apply_yaw_only(root_quat: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the yaw extracted from WXYZ quaternions."""
    if root_quat.ndim != 2 or root_quat.shape[-1] != 4:
        raise ValueError("root_quat must have shape (N, 4)")

    vectors = _expand_camera_offset(vectors, root_quat.shape[0])
    w, x, y, z = root_quat.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    vx, vy, vz = vectors.unbind(dim=-1)
    return torch.stack((cos_yaw * vx - sin_yaw * vy, sin_yaw * vx + cos_yaw * vy, vz), dim=-1)


def compute_follow_camera_view(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    eye_offset: torch.Tensor,
    target_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute world camera eye/target for the evaluation follow camera."""
    if root_pos.ndim != 2 or root_pos.shape[-1] != 3:
        raise ValueError("root_pos must have shape (N, 3)")

    eye_offset = _expand_camera_offset(eye_offset.to(root_pos.device), root_pos.shape[0])
    target_offset = _expand_camera_offset(target_offset.to(root_pos.device), root_pos.shape[0])
    cam_positions = root_pos + _quat_apply_yaw_only(root_quat, eye_offset)
    cam_targets = root_pos + _quat_apply_yaw_only(root_quat, target_offset)
    return cam_positions, cam_targets


def smooth_follow_camera_view(
    target_positions: torch.Tensor,
    target_targets: torch.Tensor,
    last_positions: torch.Tensor | None,
    last_targets: torch.Tensor | None,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply EMA smoothing to camera eye/target while preserving the first frame."""
    if (
        last_positions is None
        or last_targets is None
        or last_positions.shape != target_positions.shape
        or last_targets.shape != target_targets.shape
    ):
        return target_positions, target_targets

    alpha = float(max(0.0, min(1.0, alpha)))
    smoothed_positions = alpha * target_positions + (1.0 - alpha) * last_positions
    smoothed_targets = alpha * target_targets + (1.0 - alpha) * last_targets
    return smoothed_positions, smoothed_targets


def _get_algo_feature():
    """Dynamically import feature classes based on KAIWU_ALGORITHM env var.

    根据 KAIWU_ALGORITHM 环境变量动态加载对应算法的 feature 模块，
    避免 tools/base_env 硬编码依赖 agent_ppo。支持 agent_ppo / agent_diy。

    Returns:
        tuple: (CriticObservationProcess, PolicyObservationProcess, RewardProcess)
    """
    algo = os.environ.get("KAIWU_ALGORITHM", "ppo")
    module_name = f"agent_{algo}.feature"
    import importlib

    module = importlib.import_module(module_name)
    return module.CriticObservationProcess, module.PolicyObservationProcess, module.RewardProcess


# Logger: try KaiwuLogger first, fall back to simple print
# 日志: 优先使用 KaiwuLogger，不可用时回退到简易日志
try:
    from common_python.logging.kaiwu_logger import KaiwuLogger
    from common_python.config.config_control import CONFIG

    _HAS_KAIWU_LOGGER = True
except ImportError:
    _HAS_KAIWU_LOGGER = False


# ======================================================================
# Isaac Lab 要求: 必须在导入任何 Isaac Sim 模块之前启动 SimulationApp。
# AppLauncher 的生命周期由 Robot 实例管理。
# ======================================================================


# ------------------------------------------------------------------
# 注册 unitree_rl_lab 的任务（需要在 SimulationApp 启动之后执行）
# ------------------------------------------------------------------
def _register_tasks():
    """导入 unitree_rl_lab 任务注册模块。

    list_envs.py 可能位于不同路径:
      - 本地开发: tools/unitree_rl_lab/scripts/list_envs.py
      - 集群镜像: /workspace/unitree_rl_lab/scripts/list_envs.py
    按优先级依次查找，找到后临时加入 sys.path 并导入。
    """
    candidate_dirs = [
        str(pathlib.Path(__file__).parent / "unitree_rl_lab" / "scripts"),
        "/workspace/unitree_rl_lab/scripts",
    ]

    scripts_dir = None
    for d in candidate_dirs:
        if os.path.isfile(os.path.join(d, "list_envs.py")):
            scripts_dir = d
            break

    if scripts_dir is None:
        raise ImportError(f"list_envs.py not found in any of: {candidate_dirs}")

    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from list_envs import import_packages  # noqa: F401

    if sys.path[0] == scripts_dir:
        sys.path.pop(0)


# ======================================================================
# 简易 Logger —— 替代 KaiwuLogger（外部平台依赖，独立运行时不可用）
# ======================================================================
class _SimpleLogger:
    """兼容 KaiwuLogger 接口的简易日志。"""

    def __init__(self):
        self._logger = logging.getLogger("RobotEnv")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def setLoggerFormat(self, *args, **kwargs):
        pass

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)

    def exception(self, msg):
        self._logger.exception(msg)


def _configure_terrain_curriculum_behavior(
    *,
    tg,
    scene_terrain,
    env_cfg,
    mode: str | None,
    curriculum_enabled: bool,
    is_eval_mode: bool,
    log,
):
    """Apply terrain curriculum semantics while keeping standard monitor levels meaningful."""
    normalized_mode = (mode or "").lower()
    standard_static_grid = normalized_mode == "standard" and not curriculum_enabled and not is_eval_mode

    if standard_static_grid:
        # In standard mode, monitor difficulty labels are based on terrain_levels (rows).
        # Keep deterministic row→difficulty / col→sub-terrain generation, but disable
        # runtime promotion/demotion below. Reset will uniformly resample rows.
        tg.curriculum = True
        scene_terrain.randomize_terrain_levels_on_reset = True
        scene_terrain.terrain_level_sampling_mode = "uniform"
        log(
            "[terrain] standard curriculum=false: keep terrain_generator.curriculum=True "
            "for deterministic row→difficulty grid; randomize terrain_levels on reset"
        )
    else:
        tg.curriculum = bool(curriculum_enabled)
        scene_terrain.randomize_terrain_levels_on_reset = False
        scene_terrain.terrain_level_sampling_mode = "none"
        log(f"[terrain] Set terrain_generator.curriculum={curriculum_enabled}")

    if not curriculum_enabled and hasattr(env_cfg, "curriculum"):
        for curr_name in ("terrain_levels", "lin_vel_cmd_levels", "ang_vel_cmd_levels"):
            if hasattr(env_cfg.curriculum, curr_name):
                setattr(env_cfg.curriculum, curr_name, None)
                log(f"[terrain] Disabled curriculum.{curr_name} (curriculum=false)")


def _resolve_random_seed():
    """Read RANDOM_SEED injected by the launcher and fail fast on invalid input.
    直接读取启动脚本注入的 RANDOM_SEED；若未注入或格式非法则立即报错。"""
    env_seed = os.environ.get("RANDOM_SEED")
    if env_seed is None:
        return int(datetime.datetime.now().timestamp() * 1000)

    try:
        return int(env_seed)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"RANDOM_SEED must be an integer, got {env_seed!r}") from exc


def apply_usr_conf_to_env_cfg(env_cfg, usr_conf: dict, logger=None):
    """Apply user TOML configuration sections to Isaac Lab env_cfg.

    将用户 TOML 配置中的 terrain / commands / domain_rand / noise / init_state
    等 section 映射并应用到 Isaac Lab 的 env_cfg 对象上。

    Args:
        env_cfg: A ManagerBasedRLEnvCfg instance (e.g. RobotEnvCfg).
                 ManagerBasedRLEnvCfg 实例（如 RobotEnvCfg）。
        usr_conf: Merged user configuration dict (base + user + global).
                  合并后的用户配置字典（base + user + global）。
        logger: Optional logger for debug output.
                可选的日志对象，用于调试输出。
    """

    def _log(msg: str):
        if logger:
            logger.info(msg)

    def _coerce_override_value(current_value, override_value):
        """Coerce override value to match the shape/type of current config value."""
        if override_value is None:
            return None

        if isinstance(current_value, tuple) and isinstance(override_value, list):
            return tuple(
                _coerce_override_value(current_value[idx] if idx < len(current_value) else None, item)
                for idx, item in enumerate(override_value)
            )
        if isinstance(current_value, list) and isinstance(override_value, tuple):
            return [
                _coerce_override_value(current_value[idx] if idx < len(current_value) else None, item)
                for idx, item in enumerate(override_value)
            ]

        if isinstance(current_value, bool):
            return bool(override_value)
        if (
            isinstance(current_value, int)
            and not isinstance(current_value, bool)
            and isinstance(override_value, (int, float))
        ):
            return int(override_value)
        if isinstance(current_value, float) and isinstance(override_value, (int, float)):
            return float(override_value)

        return override_value

    def _apply_nested_overrides(target, overrides: dict, path: str):
        """Recursively apply raw overrides onto Isaac Lab config objects/dicts."""
        if not isinstance(overrides, dict):
            _log(f"[{path}] override payload must be dict, got {type(overrides).__name__}, skipping")
            return

        for key, override_value in overrides.items():
            child_path = f"{path}.{key}"

            if isinstance(target, dict):
                has_child = key in target
                current_value = target.get(key)
            else:
                has_child = hasattr(target, key)
                current_value = getattr(target, key, None)

            if not has_child:
                _log(f"[{path}] Attribute '{key}' not found, skipping")
                continue

            if (
                isinstance(override_value, dict)
                and current_value is not None
                and (isinstance(current_value, dict) or hasattr(current_value, "__dict__"))
            ):
                _apply_nested_overrides(current_value, override_value, child_path)
                continue

            coerced_value = _coerce_override_value(current_value, override_value)
            if isinstance(target, dict):
                target[key] = coerced_value
            else:
                setattr(target, key, coerced_value)
            _log(f"[{child_path}] Set value={coerced_value}")

    # ------------------------------------------------------------------
    # [terrain] → env_cfg.scene.terrain
    # ------------------------------------------------------------------
    terrain_conf = usr_conf.get("terrain", {})
    if terrain_conf and hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "terrain"):
        scene_terrain = env_cfg.scene.terrain

        # --- mode: "standard" / "track" / "plane" ---
        # "standard" or "track" → terrain_type="generator" and swap terrain_generator
        # "plane" (or legacy mesh_type="plane") → terrain_type="plane"
        mode = terrain_conf.get("mode")
        mesh_type = terrain_conf.get("mesh_type")

        if mode is not None:
            if mode == "plane":
                scene_terrain.terrain_type = "plane"
                # Disable terrain_levels curriculum — TerrainImporter has no terrain_levels for plane
                if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
                    env_cfg.curriculum.terrain_levels = None
                    _log("[terrain] Disabled curriculum.terrain_levels (plane mode)")
                _log("[terrain] Set terrain_type='plane' (mode=plane)")
            elif mode in ("standard", "track"):
                scene_terrain.terrain_type = "generator"
                # Swap terrain_generator to the matching pre-defined config
                from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
                    STANDARD_TERRAIN_CFG,
                    TRACK_TERRAIN_CFG,
                )

                if mode == "standard":
                    scene_terrain.terrain_generator = copy.deepcopy(STANDARD_TERRAIN_CFG)
                else:
                    scene_terrain.terrain_generator = copy.deepcopy(TRACK_TERRAIN_CFG)
                _log(f"[terrain] Set terrain_type='generator', mode='{mode}'")

                # Switch reset function for track mode: spawn at track start
                if mode == "track" and hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
                    from unitree_rl_lab.tasks.locomotion.mdp.events import reset_root_state_track_start

                    env_cfg.events.reset_base.func = reset_root_state_track_start
                    _log("[terrain] Switched reset_base to reset_root_state_track_start (track mode)")
            else:
                _log(f"[terrain] Unknown mode='{mode}', skipping terrain_type change")
        elif mesh_type is not None:
            # Legacy: mesh_type="plane" → terrain_type="plane", others → "generator"
            if mesh_type == "plane":
                scene_terrain.terrain_type = "plane"
                # Disable terrain_levels curriculum — TerrainImporter has no terrain_levels for plane
                if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
                    env_cfg.curriculum.terrain_levels = None
                    _log("[terrain] Disabled curriculum.terrain_levels (plane mode)")
                _log("[terrain] Set terrain_type='plane' (legacy mesh_type)")
            else:
                scene_terrain.terrain_type = "generator"
                _log(f"[terrain] Set terrain_type='generator' (legacy mesh_type={mesh_type})")

        # --- Generator-level parameters (only if terrain_generator exists) ---
        tg = scene_terrain.terrain_generator
        if tg is not None:
            # seed: evaluation transfers env_conf.seed to terrain_generator.seed
            # so the platform can fully control evaluation randomness
            # (terrain tile layout is reproducible under the platform-assigned seed).
            # Training skips this transfer; terrain_generator.seed stays None
            # and IsaacLab falls back to numpy's global RNG state, so training
            # terrain tile layout is not locked to env_conf.seed.
            # seed：评估模式将 env_conf.seed 透传给 terrain_generator.seed，
            # 让平台能完全控制评估阶段的随机性（在平台指定的 seed 下地形瓦片布局可复现）。
            # 训练模式跳过透传，terrain_generator.seed 保持 None，
            # 由 IsaacLab 回退到 numpy 全局 RNG 状态，训练阶段地形瓦片布局不被 env_conf.seed 锁定。
            is_eval_for_seed = bool(usr_conf.get("is_eval", False))
            if is_eval_for_seed:
                # Terrain seed must come from the platform-assigned seed (stable across rounds),
                # NOT from env_cfg.seed which is randomized per run in eval mode.
                # 地形 seed 必须来自平台下发的稳定 seed（每轮相同），
                # 不能回退到 env_cfg.seed —— 后者在评估模式下每轮随机。
                terrain_seed = terrain_conf.get(
                    "seed", usr_conf.get("env_conf", {}).get("seed", getattr(env_cfg, "seed", None))
                )
                if terrain_seed is not None and hasattr(tg, "seed"):
                    tg.seed = int(terrain_seed)
                    _log(f"[terrain] Eval mode: set terrain_generator.seed={tg.seed}")

                # env_cfg.seed uses RANDOM_SEED env var injected by start_eval.sh:
                #   race mode : owner_id*100 + round_index (differs per round)
                #   non-race  : millisecond timestamp (differs per run)
                # Mask to 31 bits: numpy / IsaacLab's np.random.seed() only accepts [0, 2^32-1];
                # millisecond timestamps are ~10^13 which exceeds the limit.
                # env_cfg.seed 使用 start_eval.sh 注入的 RANDOM_SEED 环境变量：
                #   比赛模式：owner_id*100 + round_index，每轮不同
                #   非比赛 ：毫秒时间戳，每次启动不同
                # 掩码截到 31 位：numpy / IsaacLab 的 np.random.seed() 只接受 [0, 2^32-1]，
                # 而毫秒时间戳约 10^13，会超限导致 ValueError。
                env_cfg.seed = _resolve_random_seed() & 0x7FFFFFFF
                _log(
                    f"[seed] Eval mode: terrain_seed={terrain_seed} (stable), "
                    f"env_cfg.seed={env_cfg.seed} (from RANDOM_SEED, differs per round)"
                )
            else:
                _log("[terrain] Train mode: skip terrain_generator.seed transfer (keep None, use global RNG state)")

            # num_rows
            num_rows = terrain_conf.get("num_rows")
            if num_rows is not None:
                tg.num_rows = int(num_rows)
                _log(f"[terrain] Set terrain_generator.num_rows={num_rows}")

            # num_cols
            num_cols = terrain_conf.get("num_cols")
            if num_cols is not None:
                tg.num_cols = int(num_cols)
                _log(f"[terrain] Set terrain_generator.num_cols={num_cols}")

            # difficulty_range
            diff_range = terrain_conf.get("difficulty_range")
            if diff_range is not None and isinstance(diff_range, (list, tuple)) and len(diff_range) == 2:
                tg.difficulty_range = (float(diff_range[0]), float(diff_range[1]))
                _log(f"[terrain] Set terrain_generator.difficulty_range={tg.difficulty_range}")

            # curriculum
            curriculum = terrain_conf.get("curriculum")
            if curriculum is not None:
                _configure_terrain_curriculum_behavior(
                    tg=tg,
                    scene_terrain=scene_terrain,
                    env_cfg=env_cfg,
                    mode=mode,
                    curriculum_enabled=bool(curriculum),
                    is_eval_mode=bool(usr_conf.get("is_eval", False)),
                    log=_log,
                )

            # --- [terrain.standard] sub-terrain parameters ---
            # Passthrough ALL user-specified attributes to sub-terrain Cfg objects.
            # 透传用户在 TOML 中指定的所有子地形属性到对应的 Cfg 对象。
            #
            # If user specifies ANY sub-terrain proportions, first zero out ALL
            # sub-terrains to avoid Isaac Lab defaults leaking through.
            # 如果用户指定了任何子地形比例，先将所有子地形清零，
            # 防止 Isaac Lab 默认值残留。
            standard_conf = terrain_conf.get("standard", {})
            if standard_conf and hasattr(tg, "sub_terrains") and tg.sub_terrains:
                # Check if any proportion is specified
                # `sub_terrains` (list form, eval-only) is NOT a per-terrain dict, skip it here
                # `sub_terrains` (列表形式，仅评估) 不是 per-terrain dict，跳过
                has_proportion = any(
                    k != "sub_terrains" and isinstance(v, dict) and "proportion" in v for k, v in standard_conf.items()
                )
                if has_proportion:
                    # Zero out all sub-terrains first
                    for sub_name, sub_cfg in tg.sub_terrains.items():
                        if hasattr(sub_cfg, "proportion"):
                            sub_cfg.proportion = 0.0
                    _log("[terrain.standard] Zeroed all sub-terrain proportions before applying user config")

                # Apply user-specified attributes (proportion + all other params)
                # 透传用户指定的所有属性（proportion + 其他参数）
                for sub_name, sub_params in standard_conf.items():
                    # Skip non-dict keys (e.g. 'sub_terrains' list form, eval-only)
                    # 跳过非 dict 字段（如评估专用的 'sub_terrains' 列表）
                    if not isinstance(sub_params, dict):
                        continue
                    if sub_name not in tg.sub_terrains:
                        _log(f"[terrain.standard] Sub-terrain '{sub_name}' not found in terrain_generator")
                        continue

                    sub_cfg = tg.sub_terrains[sub_name]
                    for param_key, param_val in sub_params.items():
                        if not hasattr(sub_cfg, param_key):
                            _log(
                                f"[terrain.standard.{sub_name}] "
                                f"Attribute '{param_key}' not found on {type(sub_cfg).__name__}, skipping"
                            )
                            continue

                        # Type coercion: TOML arrays → tuples (Isaac Lab @configclass uses tuples)
                        # 类型转换：TOML 数组 → 元组（Isaac Lab @configclass 使用 tuple）
                        original = getattr(sub_cfg, param_key)
                        if isinstance(param_val, list):
                            if isinstance(original, tuple):
                                if (
                                    original
                                    and isinstance(original[0], int)
                                    and all(isinstance(v, (int, float)) for v in param_val)
                                ):
                                    param_val = tuple(int(v) for v in param_val)
                                else:
                                    param_val = tuple(float(v) if isinstance(v, (int, float)) else v for v in param_val)
                        elif isinstance(param_val, (int, float)) and not isinstance(param_val, bool):
                            if isinstance(original, float):
                                param_val = float(param_val)
                            elif isinstance(original, int) and not isinstance(original, bool):
                                param_val = int(param_val)

                        setattr(sub_cfg, param_key, param_val)
                        _log(f"[terrain.standard.{sub_name}] Set {param_key}={param_val}")

            # --- [terrain.track] track-specific parameters ---
            # 覆盖 TrackTerrainGeneratorCfg 的 track_length、num_parallel_tracks、
            # sub_terrains_random、sub_terrains_order 等参数。
            track_conf = terrain_conf.get("track", {})
            if track_conf and hasattr(tg, "track_length"):
                # track_length → num_rows (X-axis, sub-terrain sequence)
                tl = track_conf.get("track_length")
                if tl is not None:
                    tg.track_length = int(tl)
                    tg.num_rows = int(tl)  # num_rows is overridden by track_length
                    _log(f"[terrain.track] Set track_length={tl} (num_rows={tl})")

                # num_parallel_tracks → num_cols (Y-axis, parallel tracks)
                npt = track_conf.get("num_parallel_tracks")
                if npt is not None:
                    tg.num_parallel_tracks = int(npt)
                    tg.num_cols = int(npt)  # num_cols is overridden by num_parallel_tracks
                    _log(f"[terrain.track] Set num_parallel_tracks={npt} (num_cols={npt})")

                # sub_terrains_random
                sr = track_conf.get("sub_terrains_random")
                if sr is not None:
                    tg.sub_terrains_random = bool(sr)
                    _log(f"[terrain.track] Set sub_terrains_random={sr}")

                # sub_terrains_order (explicit) takes priority over sub_terrains (shorthand)
                sto = track_conf.get("sub_terrains_order")
                if sto is not None and isinstance(sto, list):
                    tg.sub_terrains_order = list(sto)
                    _log(f"[terrain.track] Set sub_terrains_order={sto}")
                else:
                    st = track_conf.get("sub_terrains")
                    if st is not None and isinstance(st, list):
                        tg.sub_terrains_order = list(st)
                        _log(f"[terrain.track] Set sub_terrains_order={st} (from sub_terrains)")

                # track_wall_enabled / track_wall_height / track_wall_thickness
                twe = track_conf.get("track_wall_enabled")
                if twe is not None and hasattr(tg, "track_wall_enabled"):
                    tg.track_wall_enabled = bool(twe)
                    _log(f"[terrain.track] Set track_wall_enabled={twe}")

                twh = track_conf.get("track_wall_height")
                if twh is not None and hasattr(tg, "track_wall_height"):
                    tg.track_wall_height = float(twh)
                    _log(f"[terrain.track] Set track_wall_height={twh}")

                twt = track_conf.get("track_wall_thickness")
                if twt is not None and hasattr(tg, "track_wall_thickness"):
                    tg.track_wall_thickness = float(twt)
                    _log(f"[terrain.track] Set track_wall_thickness={twt}")

        # max_init_terrain_level
        max_level = terrain_conf.get("max_init_terrain_level")
        if max_level is not None:
            scene_terrain.max_init_terrain_level = int(max_level)
            _log(f"[terrain] Set max_init_terrain_level={max_level}")

        # ----------------------------------------------------------------------
        # Eval-only: deterministic level-list placement
        # 评估专用：基于 level 列表的确定性放置
        # ----------------------------------------------------------------------
        # When user provides `[terrain] level = [...]` in eval config:
        # - Standard mode: num_rows = max(level)+1, num_cols = len(sub_terrains)
        #   Sub-terrain dict is rebuilt from list (each terrain occupies one column).
        # - Track  mode: num_parallel_tracks (=num_cols) defaults to 10 and is
        #   auto-extended only when max(level)+1 is larger.
        # - Curriculum is force-disabled to keep level assignment deterministic.
        # - Stash level/sub_terrains lists onto env_cfg so the reset function can
        #   consume them later for per-env (row, col) computation.
        #
        # 当用户在评估配置里设置 `[terrain] level = [...]` 时：
        # - Standard 模式：num_rows = max(level)+1，num_cols = len(sub_terrains)；
        #   按 sub_terrains 列表顺序重建子地形 dict（每种 terrain 独占一列）
        # - Track  模式：num_parallel_tracks（=num_cols）默认 10，仅当 max(level)+1 更大时自动扩展
        # - 强制关闭 curriculum，确保 level 分配确定性
        # - 把 level / sub_terrains 列表挂到 env_cfg 上，供 reset 函数计算 (row, col)
        is_eval = bool(usr_conf.get("is_eval", False))
        level_list = terrain_conf.get("level")
        if is_eval and level_list and isinstance(level_list, list) and tg is not None:
            # Auto-normalize: cast to int, clamp to [0, 9], and sort ascending.
            # Duplicates were already rejected by the validator; clamping here
            # is defensive (e.g. when this code path is reached from tests that
            # bypass the validator) and sorting gives a predictable placement
            # order regardless of how the user wrote the list.
            # 自动规整：转 int、限幅 [0, 9]、升序。
            # 重复元素已在校验层拦截；此处 clamp 为防御性兜底（例如测试绕过校验直接调用），
            # 排序保证放置顺序可预测。
            raw_level_list = [int(v) for v in level_list]
            level_list_int = sorted(set(max(0, min(9, v)) for v in raw_level_list))
            if level_list_int != raw_level_list:
                _log(
                    f"[terrain] Normalized eval level list: {raw_level_list} → "
                    f"{level_list_int} (sort + clamp to [0, 9])"
                )
            max_level_required = max(level_list_int) + 1

            # Stash onto env_cfg for the reset function to consume
            # 挂到 env_cfg 上供 reset 函数消费
            env_cfg._eval_level_list = list(level_list_int)
            _log(f"[terrain] Eval level list: {level_list_int}")

            if mode == "standard":
                # Read sub_terrains list (eval-only deterministic order)
                # 读取 sub_terrains 列表（仅评估，确定性顺序）
                std_conf = terrain_conf.get("standard", {})
                sub_list = std_conf.get("sub_terrains")
                if isinstance(sub_list, list) and len(sub_list) > 0 and hasattr(tg, "sub_terrains"):
                    env_cfg._eval_sub_terrains = list(sub_list)

                    # Auto-compute num_rows / num_cols
                    # 自动计算 num_rows / num_cols
                    tg.num_rows = max_level_required
                    tg.num_cols = len(sub_list)
                    _log(
                        f"[terrain] eval level={level_list_int} + sub_terrains={sub_list} "
                        f"→ num_rows={tg.num_rows}, num_cols={tg.num_cols}"
                    )

                    # Rebuild sub_terrains dict in user-specified order
                    # Each terrain gets equal proportion (placement is deterministic
                    # via terrain_types, so proportion only affects column generation order).
                    # 按用户指定顺序重建 sub_terrains dict：
                    # 每种 terrain 占等比例（实际放置由 reset 函数通过 terrain_types 控制，
                    # proportion 只影响 Isaac Lab 生成时的列顺序）
                    from collections import OrderedDict

                    new_sub_terrains = OrderedDict()
                    equal_prop = 1.0 / len(sub_list)
                    for sub_name in sub_list:
                        if sub_name not in tg.sub_terrains:
                            _log(
                                f"[terrain.standard] eval sub_terrains contains '{sub_name}' "
                                f"which is not present in the default terrain_generator; skipping"
                            )
                            continue
                        sub_cfg = tg.sub_terrains[sub_name]
                        if hasattr(sub_cfg, "proportion"):
                            sub_cfg.proportion = equal_prop
                        new_sub_terrains[sub_name] = sub_cfg
                    if new_sub_terrains:
                        tg.sub_terrains = new_sub_terrains
                        _log(
                            f"[terrain.standard] Rebuilt sub_terrains in eval order: "
                            f"{list(new_sub_terrains.keys())} (each proportion={equal_prop:.4f})"
                        )

            elif mode == "track":
                # Track: ensure num_parallel_tracks (=num_cols) covers max(level)+1
                # Track：保证 num_parallel_tracks（=num_cols）≥ max(level)+1
                if hasattr(tg, "num_parallel_tracks"):
                    if tg.num_parallel_tracks < max_level_required:
                        _log(
                            f"[terrain.track] num_parallel_tracks={tg.num_parallel_tracks} "
                            f"is too small for eval level={level_list_int}; "
                            f"auto-extending to {max_level_required}"
                        )
                        tg.num_parallel_tracks = max_level_required
                        tg.num_cols = max_level_required

            # --- Keep tg.curriculum=True so Isaac Lab uses _generate_curriculum_terrains
            #     (grid gets a FIXED col→sub_terrain mapping by cumsum(proportion);
            #     with equal proportions and rebuilt OrderedDict, col k → k-th sub).
            #     If we set tg.curriculum=False, Isaac Lab would call
            #     _generate_random_terrains which independently samples each (row, col)
            #     cell, breaking our deterministic placement. We only force-disable
            #     *env_cfg.curriculum.terrain_levels* below to prevent Isaac Lab's
            #     training-time level promotion/demotion from kicking in.
            # 保留 tg.curriculum=True，让 Isaac Lab 调用 _generate_curriculum_terrains
            # （按 cumsum(proportion) 把 col 切成固定段；加上我们把 sub_terrains 重排为
            # 等比例的 OrderedDict，col k 严格对应第 k 个 sub_terrain）。
            # 如果设成 False，Isaac Lab 会调 _generate_random_terrains 对每个 (row, col)
            # 独立随机采样，会破坏我们的确定性放置。
            # 下方 env_cfg.curriculum.terrain_levels 仍然强制置 None，避免训练期
            # 的自动升降级逻辑干扰 reset 时写入的 row/col。
            if hasattr(tg, "curriculum") and not tg.curriculum:
                tg.curriculum = True
                _log(
                    "[terrain] Forced tg.curriculum=True so Isaac Lab uses curriculum "
                    "terrain generation (fixed col→sub mapping); level placement stays "
                    "deterministic via reset_root_state_eval_level_aware"
                )
            if hasattr(env_cfg, "curriculum"):
                if hasattr(env_cfg.curriculum, "terrain_levels"):
                    env_cfg.curriculum.terrain_levels = None
                    _log("[terrain] Forced curriculum.terrain_levels=None (eval level list)")
                if hasattr(env_cfg.curriculum, "lin_vel_cmd_levels"):
                    env_cfg.curriculum.lin_vel_cmd_levels = None
                if hasattr(env_cfg.curriculum, "ang_vel_cmd_levels"):
                    env_cfg.curriculum.ang_vel_cmd_levels = None

            # Hook the eval-aware reset function so per-env (row, col) gets
            # deterministically computed at every reset.
            # 挂载评估专用的 reset 函数，每次 reset 时确定性计算 (row, col)
            if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
                from unitree_rl_lab.tasks.locomotion.mdp.events import (
                    reset_root_state_eval_level_aware,
                )

                env_cfg.events.reset_base.func = reset_root_state_eval_level_aware
                _log("[terrain] Switched reset_base to reset_root_state_eval_level_aware (eval level list)")

    # ------------------------------------------------------------------
    # [sensors] → env_cfg.scene.<sensor_name> (RayCasterCfg)
    # ------------------------------------------------------------------
    sensors_conf = usr_conf.get("sensors", {})
    if sensors_conf and hasattr(env_cfg, "scene"):
        scene = env_cfg.scene
        for sensor_name, sensor_params in sensors_conf.items():
            if not isinstance(sensor_params, dict):
                continue
            if not hasattr(scene, sensor_name):
                _log(f"[sensors] Sensor '{sensor_name}' not found on scene, skipping")
                continue

            sensor_cfg = getattr(scene, sensor_name)

            # offset_pos → sensor_cfg.offset.pos
            offset_pos = sensor_params.get("offset_pos")
            if offset_pos is not None and isinstance(offset_pos, (list, tuple)) and len(offset_pos) == 3:
                sensor_cfg.offset.pos = tuple(float(v) for v in offset_pos)
                _log(f"[sensors.{sensor_name}] Set offset.pos={sensor_cfg.offset.pos}")

            # pattern_resolution → sensor_cfg.pattern_cfg.resolution
            resolution = sensor_params.get("pattern_resolution")
            if resolution is not None:
                sensor_cfg.pattern_cfg.resolution = float(resolution)
                _log(f"[sensors.{sensor_name}] Set pattern_cfg.resolution={resolution}")

            # pattern_size → sensor_cfg.pattern_cfg.size
            pattern_size = sensor_params.get("pattern_size")
            if pattern_size is not None and isinstance(pattern_size, (list, tuple)) and len(pattern_size) == 2:
                sensor_cfg.pattern_cfg.size = [float(v) for v in pattern_size]
                _log(f"[sensors.{sensor_name}] Set pattern_cfg.size={sensor_cfg.pattern_cfg.size}")

    # ------------------------------------------------------------------
    # [init_state] → env_cfg.scene.robot.init_state
    # ------------------------------------------------------------------
    init_state_conf = usr_conf.get("init_state", {})
    if init_state_conf and hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "robot"):
        robot_cfg = env_cfg.scene.robot
        if hasattr(robot_cfg, "init_state"):
            pos = init_state_conf.get("pos")
            if pos is not None and isinstance(pos, (list, tuple)) and len(pos) == 3:
                robot_cfg.init_state.pos = tuple(float(v) for v in pos)
                _log(f"[init_state] Set robot init_state.pos={robot_cfg.init_state.pos}")

    # ------------------------------------------------------------------
    # [commands] / [commands.ranges] → env_cfg.commands.base_velocity
    # ------------------------------------------------------------------
    commands_conf = usr_conf.get("commands", {})
    if commands_conf and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        base_vel_cmd = env_cfg.commands.base_velocity

        # resampling_time → resampling_time_range (supports scalar or [min, max])
        resample_time = commands_conf.get("resampling_time")
        if resample_time is not None:
            if isinstance(resample_time, (list, tuple)) and len(resample_time) == 2:
                base_vel_cmd.resampling_time_range = (float(resample_time[0]), float(resample_time[1]))
                _log(f"[commands] Set resampling_time_range={base_vel_cmd.resampling_time_range}")
            else:
                base_vel_cmd.resampling_time_range = (float(resample_time), float(resample_time))
                _log(f"[commands] Set resampling_time_range=({resample_time}, {resample_time})")

        # heading_command
        heading_cmd = commands_conf.get("heading_command")
        if heading_cmd is not None and hasattr(base_vel_cmd, "heading_command"):
            base_vel_cmd.heading_command = bool(heading_cmd)
            _log(f"[commands] Set heading_command={heading_cmd}")

        # [commands.ranges] → base_velocity.ranges
        ranges_conf = commands_conf.get("ranges", {})
        if ranges_conf and hasattr(base_vel_cmd, "ranges"):
            cmd_ranges = base_vel_cmd.ranges

            # TOML name → Isaac Lab Ranges attribute name
            range_mapping = {
                "lin_vel_x": "lin_vel_x",
                "lin_vel_y": "lin_vel_y",
                "ang_vel_yaw": "ang_vel_z",
                "heading": "heading",  # if exists
            }
            for toml_key, attr_name in range_mapping.items():
                val = ranges_conf.get(toml_key)
                if val is not None and isinstance(val, (list, tuple)) and len(val) == 2:
                    if hasattr(cmd_ranges, attr_name):
                        setattr(cmd_ranges, attr_name, (float(val[0]), float(val[1])))
                        _log(f"[commands.ranges] Set ranges.{attr_name}={val}")

            # limit_ranges (flat_* variants)
            if hasattr(base_vel_cmd, "limit_ranges"):
                limit_ranges = base_vel_cmd.limit_ranges
                limit_mapping = {
                    "flat_lin_vel_x": "lin_vel_x",
                    "flat_lin_vel_y": "lin_vel_y",
                    "flat_ang_vel_yaw": "ang_vel_z",
                    "flat_heading": "heading",
                }
                for toml_key, attr_name in limit_mapping.items():
                    val = ranges_conf.get(toml_key)
                    if val is not None and isinstance(val, (list, tuple)) and len(val) == 2:
                        if hasattr(limit_ranges, attr_name):
                            setattr(limit_ranges, attr_name, (float(val[0]), float(val[1])))
                            _log(f"[commands.ranges] Set limit_ranges.{attr_name}={val}")

        # [commands.limit] → base_velocity.limit_ranges (new structured format)
        limit_conf = commands_conf.get("limit", {})
        if limit_conf and hasattr(base_vel_cmd, "limit_ranges"):
            limit_ranges = base_vel_cmd.limit_ranges
            limit_key_mapping = {
                "lin_vel_x": "lin_vel_x",
                "lin_vel_y": "lin_vel_y",
                "ang_vel_z": "ang_vel_z",
            }
            for toml_key, attr_name in limit_key_mapping.items():
                val = limit_conf.get(toml_key)
                if val is not None and isinstance(val, (list, tuple)) and len(val) == 2:
                    if hasattr(limit_ranges, attr_name):
                        setattr(limit_ranges, attr_name, (float(val[0]), float(val[1])))
                        _log(f"[commands.limit] Set limit_ranges.{attr_name}={val}")

        # max curriculum velocities
        curriculum_mapping = {
            "max_lin_vel_x_curriculum": "max_lin_vel_x",
            "max_lin_vel_y_curriculum": "max_lin_vel_y",
            "max_ang_vel_yaw_curriculum": "max_ang_vel_z",
        }
        for toml_key, attr_name in curriculum_mapping.items():
            val = commands_conf.get(toml_key)
            if val is not None and hasattr(base_vel_cmd, attr_name):
                setattr(base_vel_cmd, attr_name, float(val))
                _log(f"[commands] Set {attr_name}={val}")

    # ------------------------------------------------------------------
    # [domain_rand] → env_cfg.events
    # ------------------------------------------------------------------
    domain_rand_conf = usr_conf.get("domain_rand", {})
    if domain_rand_conf and hasattr(env_cfg, "events"):
        events = env_cfg.events
        enable_domain_rand = domain_rand_conf.get("enable_domain_rand", True)

        # --- friction randomization → events.physics_material ---
        if hasattr(events, "physics_material"):
            randomize_friction = domain_rand_conf.get("randomize_friction", False)
            if not enable_domain_rand or not randomize_friction:
                # Disable friction randomization by using uniform range [1.0, 1.0]
                # 禁用摩擦随机化：设置为均匀范围 [1.0, 1.0]
                events.physics_material.params["static_friction_range"] = (1.0, 1.0)
                events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
                _log("[domain_rand] Disabled friction randomization")
            else:
                friction_range = domain_rand_conf.get("friction_range")
                if (
                    friction_range is not None
                    and isinstance(friction_range, (list, tuple))
                    and len(friction_range) == 2
                ):
                    fr = (float(friction_range[0]), float(friction_range[1]))
                    events.physics_material.params["static_friction_range"] = fr
                    events.physics_material.params["dynamic_friction_range"] = fr
                    _log(f"[domain_rand] Set friction_range={fr}")

            # restitution
            restitution_range = domain_rand_conf.get("restitution_range")
            if restitution_range is not None and isinstance(restitution_range, (list, tuple)):
                events.physics_material.params["restitution_range"] = (
                    float(restitution_range[0]),
                    float(restitution_range[1]),
                )
                _log(f"[domain_rand] Set restitution_range={restitution_range}")

        # --- base mass randomization → events.add_base_mass ---
        if hasattr(events, "add_base_mass"):
            randomize_base_mass = domain_rand_conf.get("randomize_base_mass", True)
            if not enable_domain_rand or not randomize_base_mass:
                events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
                _log("[domain_rand] Disabled base mass randomization")
            else:
                added_mass = domain_rand_conf.get("added_mass_range")
                if added_mass is not None and isinstance(added_mass, (list, tuple)) and len(added_mass) == 2:
                    events.add_base_mass.params["mass_distribution_params"] = (
                        float(added_mass[0]),
                        float(added_mass[1]),
                    )
                    _log(f"[domain_rand] Set added_mass_range={added_mass}")

        # --- push robots → events.push_robot ---
        if hasattr(events, "push_robot"):
            push_robots = domain_rand_conf.get("push_robots", False)
            if not enable_domain_rand or not push_robots:
                # Disable push by setting velocity to 0
                # 禁用推力：将速度设置为 0
                events.push_robot.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
                _log("[domain_rand] Disabled push_robot")
            else:
                max_push = domain_rand_conf.get("max_push_vel_xy", 0.5)
                events.push_robot.params["velocity_range"] = {
                    "x": (-float(max_push), float(max_push)),
                    "y": (-float(max_push), float(max_push)),
                }
                _log(f"[domain_rand] Set push_robot velocity_range max={max_push}")

                push_interval = domain_rand_conf.get("push_interval_s")
                min_push_interval = domain_rand_conf.get("min_push_interval_s")
                if push_interval is not None:
                    min_interval = float(min_push_interval) if min_push_interval is not None else float(push_interval)
                    events.push_robot.interval_range_s = (min_interval, float(push_interval))
                    _log(f"[domain_rand] Set push_robot interval_range_s=({min_interval}, {push_interval})")

    # ------------------------------------------------------------------
    # [noise] → env_cfg.observations (ObsTerm noise)
    # ------------------------------------------------------------------
    noise_conf = usr_conf.get("noise", {})
    if noise_conf and hasattr(env_cfg, "observations"):
        add_noise = noise_conf.get("add_noise", True)

        if hasattr(env_cfg.observations, "policy"):
            policy_obs = env_cfg.observations.policy
            if not add_noise:
                # Disable corruption (noise) on policy observations
                # 关闭策略观测上的噪声
                policy_obs.enable_corruption = False
                _log("[noise] Disabled policy observation corruption (add_noise=false)")
            else:
                policy_obs.enable_corruption = True
                _log("[noise] Enabled policy observation corruption (add_noise=true)")

                # Apply noise levels to individual observation terms
                # 将噪声级别应用到各个观测项
                noise_level = float(noise_conf.get("noise_level", 1.0))

                noise_term_mapping = {
                    "dof_pos": "joint_pos_rel",
                    "dof_vel": "joint_vel_rel",
                    "ang_vel": "base_ang_vel",
                    "gravity": "projected_gravity",
                }

                from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

                for noise_key, obs_attr in noise_term_mapping.items():
                    noise_val = noise_conf.get(noise_key)
                    if noise_val is not None and hasattr(policy_obs, obs_attr):
                        obs_term = getattr(policy_obs, obs_attr)
                        if hasattr(obs_term, "noise"):
                            scaled_noise = float(noise_val) * noise_level
                            obs_term.noise = Unoise(n_min=-scaled_noise, n_max=scaled_noise)
                            _log(f"[noise] Set policy.{obs_attr} noise=±{scaled_noise}")

    # ------------------------------------------------------------------
    # [isaaclab_override] / [env_cfg_override] → raw recursive final override
    # ------------------------------------------------------------------
    for override_key in ("isaaclab_override", "env_cfg_override"):
        override_conf = usr_conf.get(override_key, {})
        if override_conf:
            _apply_nested_overrides(env_cfg, override_conf, path=override_key)

    _log("User configuration applied to env_cfg successfully")


def parse_reward_configs(usr_conf: dict | None) -> dict:
    """Parse reward configs from TOML `[rewards.*]` sections.

    从 TOML `[rewards.*]` 部分解析奖励配置。

    TOML format:
        [rewards.term_name]
        weight = 1.5
        [rewards.term_name.params]
        std = 0.5
        command_name = "base_velocity"

    Args:
        usr_conf: User configuration dict (loaded from TOML).
                  用户配置字典（从 TOML 加载）。

    Returns:
        Dict: {term_name: {"weight": float, "params": dict}}
        If no rewards section found, returns empty dict.
        如果未找到 rewards 部分，返回空字典。
    """
    if not usr_conf:
        return {}

    rewards_section = usr_conf.get("rewards", {}) or {}
    if not rewards_section:
        return {}

    result = {}
    for term_name, term_cfg in rewards_section.items():
        if not isinstance(term_cfg, dict):
            continue
        weight = term_cfg.get("weight")
        if weight is None:
            continue
        params = dict(term_cfg.get("params", {}))
        result[term_name] = {
            "weight": float(weight),
            "params": params,
        }

    return result


def register_observation_processes(env_cfg):
    """Register custom observation processes to override Isaac Lab observation groups.

    注册自定义观测处理器以覆盖 Isaac Lab 的 observation group。

    Creates PolicyObservationProcess and CriticObservationProcess instances,
    builds ObservationBridge for each, and overrides the corresponding
    observation groups in env_cfg.
    创建 PolicyObservationProcess 和 CriticObservationProcess 实例，
    为每个构建 ObservationBridge，并覆盖 env_cfg 中对应的 observation group。

    Args:
        env_cfg: A ManagerBasedRLEnvCfg instance with observations attribute.
                 一个带有 observations 属性的 ManagerBasedRLEnvCfg 实例。
    """
    CriticObservationProcess, PolicyObservationProcess, _ = _get_algo_feature()
    for observation_process in (PolicyObservationProcess(), CriticObservationProcess()):
        observation_process.create_bridge().override_group_in_env_cfg(env_cfg)


class Robot:
    """机器人环境 —— 基于 Isaac Lab / unitree_rl_lab 的强化学习环境封装。

    对外暴露与 template.py 中 Drone 类一致的接口:
        __init__, reset, step, get_action_space, get_observation_space, close
    内部使用 gymnasium + RslRlVecEnvWrapper 管理 Isaac Lab 环境。
    """

    # ------------------------------------------------------------------
    # 可选的任务名称列表（与 unitree_rl_lab 注册的环境 ID 对应）
    # ------------------------------------------------------------------
    AVAILABLE_TASKS = [
        "Unitree-Go2-Velocity",
        "Unitree-H1-Velocity",
        "Unitree-G1-29dof-Velocity",
        "Unitree-G1-29dof-Mimic-Dance-102",
        "Unitree-G1-29dof-Mimic-Gangnanm-Style",
    ]

    def __init__(self):

        self.min_episode_length = 50
        self.max_episode_length = 500

        self.frame_no = 0

        # Logger: try KaiwuLogger first, fall back to simple print
        # 日志: 优先使用 KaiwuLogger，不可用时回退到简易日志
        self.current_pid = os.getpid()
        if _HAS_KAIWU_LOGGER:
            try:
                configure_file = f"kaiwudrl/conf/kaiwudrl/aisrv.toml"
                CONFIG.set_configure_file(configure_file)
                CONFIG.parse_aisrv_configure()
                self.logger = KaiwuLogger()
                self.logger.set_logger_format(
                    f"{CONFIG.log_dir}/{CONFIG.svr_name}/base_env_process_pid{self.current_pid}_log_"
                    f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
                )
            except Exception:
                self.logger = _SimpleLogger()
        else:
            self.logger = _SimpleLogger()

        self.logger.info(f"kaiwu_env start at pid {self.current_pid}")

        # 总控开关: 是否强制 headless（无 GUI）模式
        # 设为 False 时保留 DISPLAY 等环境变量，可通过 VNC 查看 GUI
        self.force_headless = os.environ.get("ROBOT_FORCE_HEADLESS", "1") != "0"

        if self.force_headless:
            # 设置 Headless 渲染环境变量
            os.environ.pop("DISPLAY", None)
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
            os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
            os.environ["EGL_PLATFORM"] = "surfaceless"
        else:
            self.logger.info("Headless 模式已关闭，保留 DISPLAY 环境变量以支持 VNC GUI 显示")

        # ----------------------------------------------------------
        # AppLauncher 延迟到 reset() 中启动，届时根据 is_eval 决定是否开启渲染
        # ----------------------------------------------------------
        self._app_launcher = None
        self._simulation_app = None

        # 环境实例（仅初始化一次）
        self.env = None
        # 底层 gymnasium 环境（用于 close 时同时关闭）
        self._gym_env = None

        # 监控上报相关
        self._scorer = None
        self._env_monitor = None
        self._task_type = "track"  # 默认 track，reset 时从 usr_conf["terrain"]["task"] 更新

        self.is_eval = False
        self.eval_write_json_file = False
        self.env_nums = 4096

        # Evaluation: frame capture interval
        self.capture_interval = 5

        # Evaluation: video recording
        self.enable_video = False
        self.is_need_save_mp4 = False
        self.vedio_dir = None

        # TiledCamera per-env video recording
        self._use_tiled_camera = False
        self._tiled_video_writer = None
        camera_profile = get_default_eval_camera_profile()
        self._camera_eye_offset = camera_profile["eye_offset"]
        self._camera_target_offset = camera_profile["target_offset"]
        self._camera_smoothing_alpha = float(camera_profile["smoothing_alpha"])
        self._camera_smoothed_positions: torch.Tensor | None = None
        self._camera_smoothed_targets: torch.Tensor | None = None

        # 结果文件 / 关闭流程保护
        self._result_files_written = False

        # 评估模式 per-env 单次生命跟踪：
        # 标记每个 env 是否已经完成过一次 episode（仅评估模式使用）。
        # 已完成的 env 后续步骤中 action 被归零，scorer 忽略其后续 episode。
        self._eval_env_done_mask: torch.Tensor | None = None

        # 目标点可视化 marker（仅非 headless 模式使用）
        self._goal_markers = None

    # ------------------------------------------------------------------
    # 内部: 启动 Isaac Sim SimulationApp
    # ------------------------------------------------------------------
    def _launch_sim_app(self, enable_cameras: bool = False):
        """启动 Isaac Sim SimulationApp（幂等，仅执行一次）。

        Args:
            enable_cameras: 是否启用离屏渲染（评估录视频时需要 True）。
        """
        if self._simulation_app is not None:
            return

        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description="Robot RL Environment")
        parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
        parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity", help="Name of the task.")
        AppLauncher.add_app_launcher_args(parser)
        args_cli, _ = parser.parse_known_args()

        # 根据总控开关决定是否 headless
        args_cli.headless = self.force_headless
        # 仅评估录视频时启用 cameras（offscreen rendering）
        args_cli.enable_cameras = enable_cameras

        self._app_launcher = AppLauncher(args_cli)
        self._simulation_app = self._app_launcher.app
        self.logger.info(f"SimulationApp started successfully (enable_cameras={enable_cameras})")

    def _reset_camera_follow_state(self):
        """Reset cached camera smoothing state for a new evaluation run."""
        self._camera_smoothed_positions = None
        self._camera_smoothed_targets = None

    # ------------------------------------------------------------------
    # 内部: 创建 Isaac Lab 环境
    # ------------------------------------------------------------------
    def _create_env(self, task_name: str, env_cfg, reward_configs: dict | None = None):
        """独立环境初始化 —— 仅在首次调用时真正创建。"""

        if self.env is not None:
            return

        import gymnasium as gym

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        # 视频录制目录
        if self.enable_video:
            self.vedio_dir = f"/workspace/battle/{self.game_id}/mp4"
            os.makedirs(self.vedio_dir, exist_ok=True)

        # ---- TiledCamera: 评估模式且 num_envs <= 16 时启用 per-env 跟随摄像机 ----
        self._use_tiled_camera = self.enable_video and env_cfg.scene.num_envs <= 16
        if self._use_tiled_camera:
            from isaaclab.sensors import TiledCameraCfg
            import isaaclab.sim as sim_utils_cam

            camera_metrics = get_eval_camera_profile_metrics(
                eye_offset=self._camera_eye_offset,
                target_offset=self._camera_target_offset,
            )

            # TiledCamera 会按 env 自动克隆一台相机；这里只定义相机 prim 和成像参数。
            # 实际跟随位置与朝向在 _update_camera_poses() 中按每只机器狗动态设置。
            env_cfg.scene.tiled_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/FollowCamera",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    convention="opengl",
                ),
                spawn=sim_utils_cam.PinholeCameraCfg(
                    focal_length=24.0,
                    horizontal_aperture=20.955,
                ),
                data_types=["rgb"],
                width=480,
                height=480,
                update_period=0,
            )
            self.logger.info(
                "TiledCamera 已配置（每个 env 匹配一只机器狗）: "
                f"num_envs={env_cfg.scene.num_envs}, 分辨率=480x480, "
                f"eye_offset={tuple(float(v) for v in self._camera_eye_offset.tolist())}, "
                f"target_offset={tuple(float(v) for v in self._camera_target_offset.tolist())}, "
                f"rear_follow={camera_metrics['rear_follow_distance']:.2f}m, "
                f"look_ahead={camera_metrics['look_ahead_distance']:.2f}m, "
                f"pitch={camera_metrics['pitch_deg']:.1f}deg, "
                f"smoothing_alpha={self._camera_smoothing_alpha:.2f}"
            )

        # ---- Register custom observation processes (policy / critic) ----
        # ---- 注册自定义观测处理器（policy / critic）----
        register_observation_processes(env_cfg)

        # ---- Register all rewards from TOML config (full override) ----
        # ---- 从 TOML 配置全量注册奖励（全量覆盖）----
        if reward_configs:
            _, _, RewardProcess = _get_algo_feature()
            reward_process = RewardProcess()
            bridge = reward_process.create_bridge_from_configs(reward_configs)
            bridge.register_to_env_cfg(env_cfg)

        # 创建 gymnasium 环境
        self._gym_env = gym.make(task_name, cfg=env_cfg, render_mode=None)

        # 挂载 is_eval 标志到底层 env，供 reset 函数等识别训练/评估模式
        # reset_root_state_track_start 据此决定 spawn 策略：
        #   - 训练时: 在非 maze 段中随机生成
        #   - 评估时: 固定在赛道起点生成
        self._gym_env.unwrapped._is_eval = bool(self.is_eval)

        # 设置地形块边界终止开关（仅评估模式 + 配置启用时生效）
        # 训练时默认关闭，避免正常速度跟踪被频繁截断
        enable_bounds_term = self.is_eval and self.usr_conf.get("custom_parameters", {}).get(
            "enable_terrain_bounds_termination", True
        )
        env_unwrapped = self._gym_env.unwrapped
        env_unwrapped._enable_terrain_bounds_termination = enable_bounds_term
        if enable_bounds_term:
            self.logger.info("[terrain_bounds] 地形块边界终止已启用（评估模式）")

        # TiledCamera 方案：初始化 per-env 视频写入器
        if self._use_tiled_camera:
            from tools.base_env.video_writer import TiledCameraVideoWriter

            fps = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
            # Defer terrain-name resolution to the first frame write:
            # at __init__ time Isaac Lab's `terrain.terrain_types` may still be
            # zero-filled (real assignment happens during the first env reset via
            # events like `reset_robot_track_pose`). Resolving here would produce
            # wrong mp4 prefixes.
            # 延迟到首次写帧时再解析 env 地形名：__init__ 阶段 Isaac Lab 的
            # terrain_types 可能还是初始化零值（真实分配发生在首次 env reset 时的
            # reset event 中），此时解析会导致 mp4 前缀错位（例如把 stairs_inv 的
            # env 命名成 maze）。
            self._tiled_video_writer = TiledCameraVideoWriter(
                video_dir=self.vedio_dir,
                fps=fps,
                logger=self.logger,
                env_terrain_names=None,
            )

        # RSL-RL 向量化环境封装
        self.env = RslRlVecEnvWrapper(self._gym_env)

        # 设备信息
        self.sim_device = env_cfg.sim.device if hasattr(env_cfg.sim, "device") else "cuda"
        self.frame_no = 1
        self.env_nums = env_cfg.scene.num_envs

        # 动作维度
        action_space = self.env.action_space
        self.num_actions = action_space.shape[-1] if hasattr(action_space, "shape") else 12

        # 初始化监控上报
        self._init_monitor()

    # ------------------------------------------------------------------
    # 内部: TiledCamera 位置更新 + 帧写入
    # ------------------------------------------------------------------
    def _resolve_env_terrain_names(self) -> list[str] | None:
        """解析每个 env 当前所在的子地形名称（带 level 标签），用于给 mp4 文件命名。

        返回值会被拼成 ``env_{id:03d}_{name}.mp4`` 的文件名后缀。
        命名规则按优先级：

        1. **Eval level list 模式**（``env.cfg._eval_level_list`` 存在）
           - Standard: ``L{level}_{sub_terrain}``，例如 ``L1_pyramid_slope``
           - Track:    ``L{level}_track_{sub_terrains_chain}``，例如 ``L3_track_slope-stairs-maze``
           level 来自 ``_eval_level_list``，sub 来自 ``_eval_sub_terrains``（或 track 的
           ``sub_terrains`` 序列），完全确定性，无需依赖 proportion / curriculum。

        2. **传统 Standard curriculum=True 模式**（proportion 能反推 col→terrain）
           - 返回 ``{sub_terrain}``，保持旧行为。

        3. **传统 Track 模式**
           - 返回 ``track_{sub_terrains_chain}``，保持旧行为。

        无法解析时（plane 模式、curriculum=False 且无 eval level list 等）返回 None，
        让 writer 回退到 ``env_{id:03d}.mp4`` 命名。

        Returns:
            长度等于 num_envs 的字符串列表；或 None。
        """
        try:
            env_unwrapped = self._gym_env.unwrapped
            terrain = env_unwrapped.scene.terrain
            # plane 模式没有 generator
            terrain_gen_cfg = getattr(terrain.cfg, "terrain_generator", None)
            if terrain_gen_cfg is None:
                return None

            sub_terrains = getattr(terrain_gen_cfg, "sub_terrains", None)
            if not sub_terrains:
                return None
            sub_names = list(sub_terrains.keys())

            # 每个 env 在 terrain grid 中的列索引（col）；没有则无法判断地形
            terrain_types = getattr(terrain, "terrain_types", None)
            terrain_levels = getattr(terrain, "terrain_levels", None)
            if terrain_types is None:
                return None
            col_indices = terrain_types.detach().cpu().numpy().astype(int).tolist()
            row_indices = (
                terrain_levels.detach().cpu().numpy().astype(int).tolist()
                if terrain_levels is not None
                else [0] * len(col_indices)
            )

            num_envs = env_unwrapped.scene.num_envs

            # --------------------------------------------------------------
            # Priority 1: Eval level list mode — deterministic per-env naming
            # 优先路径：eval level 列表模式，按 (row=level, col→sub) 精确命名
            # --------------------------------------------------------------
            env_cfg = getattr(env_unwrapped, "cfg", None)
            eval_level_list = getattr(env_cfg, "_eval_level_list", None) if env_cfg is not None else None
            eval_sub_terrains = getattr(env_cfg, "_eval_sub_terrains", None) if env_cfg is not None else None

            # Track 模式：同一列是多个子地形顺序拼成的赛道（按 row 变化）
            # 命名用整条赛道的子地形序列拼起来，方便识别。
            track_length = getattr(terrain_gen_cfg, "track_length", None)
            is_track_mode = track_length is not None and track_length > 0

            if eval_level_list:
                if is_track_mode:
                    # Track: row=0, col=level; use the chained sub_terrains sequence
                    # Track：row=0, col=level；取整条赛道的子地形链作为标签
                    order = getattr(terrain_gen_cfg, "sub_terrains_order", None)
                    if order:
                        sequence = [n for n in order if n in sub_terrains]
                    else:
                        sequence = list(sub_names)
                    if not sequence:
                        return None
                    while len(sequence) < track_length:
                        sequence.extend(sequence)
                    sequence = sequence[:track_length]
                    track_label = "-".join(sequence)
                    return [
                        f"L{col_indices[i] if i < len(col_indices) else 0}_track_{track_label}" for i in range(num_envs)
                    ]

                # Standard: each col maps 1-to-1 to a sub_terrain in the list
                # Standard：每个 col 和 sub_terrains 列表一一对应（base_env 已重建 OrderedDict）
                if eval_sub_terrains:
                    names = []
                    for env_id in range(num_envs):
                        level = row_indices[env_id] if env_id < len(row_indices) else 0
                        col = col_indices[env_id] if env_id < len(col_indices) else 0
                        col = max(0, min(col, len(eval_sub_terrains) - 1))
                        names.append(f"L{level}_{eval_sub_terrains[col]}")
                    return names

            # --------------------------------------------------------------
            # Priority 2: legacy Track mode (no eval level list)
            # 传统 Track 模式（无 eval level 列表）
            # --------------------------------------------------------------
            if is_track_mode:
                order = getattr(terrain_gen_cfg, "sub_terrains_order", None)
                if order:
                    sequence = [n for n in order if n in sub_terrains]
                else:
                    sequence = list(sub_names)
                if not sequence:
                    return None
                # 按 track_length 循环补齐
                while len(sequence) < track_length:
                    sequence.extend(sequence)
                sequence = sequence[:track_length]
                track_label = "-".join(sequence)
                return [f"track_{track_label}" for _ in range(num_envs)]

            # --------------------------------------------------------------
            # Priority 3: legacy Standard curriculum=True (proportion formula)
            # 传统 Standard curriculum=True 模式（proportion 反推）
            # --------------------------------------------------------------
            # Standard 模式 col → sub_terrain 映射的可用性取决于 `curriculum`：
            #   - curriculum=True  → Isaac Lab `_generate_curriculum_terrains`
            #     按 cumsum(proportion) 把 col 切成固定分段，下面的 proportion
            #     公式能**精确**还原真实布局。
            #   - curriculum=False → Isaac Lab `_generate_random_terrains`
            #     对**每个 (row, col) 独立**按 proportion 概率采样，没有任何
            #     固定的 col→type 映射可以反推。
            #
            # Standard mode col→sub_terrain mapping depends on `curriculum`:
            #   curriculum=True  → proportion formula below is exact.
            #   curriculum=False → each (row, col) is independently sampled,
            #                      no deterministic mapping exists.
            #
            # 为避免 curriculum=False 下产出错位的命名误导用户（例如 mp4 文件名说
            # 是 maze 实际却是 stairs_inv），这里显式检测 curriculum 开关：一旦
            # 为 False 且没有 eval level list 托底，立刻返回 None 让 writer 退回到
            # 无前缀命名 `env_{id:03d}.mp4`，并打 warning 提示用户修改配置。
            # When curriculum=False and no eval level list is present, we refuse
            # to guess; return None so the writer falls back to prefix-less
            # `env_{id:03d}.mp4` and emit a warning.
            curriculum_on = bool(getattr(terrain_gen_cfg, "curriculum", True))
            if not curriculum_on:
                if not getattr(self, "_curriculum_false_naming_warned", False):
                    self.logger.warning(
                        "[video] 检测到 [terrain].curriculum=false 且未配置 [terrain] level 列表，"
                        "Isaac Lab 会对每个 (row, col) 独立随机采样子地形，无法通过 proportion 反推 "
                        "col→terrain 映射。mp4 文件名将退回到 `env_{id:03d}.mp4`（不带地形前缀）"
                        "以避免错位误导。如需带地形的 mp4 命名，请配置 [terrain] level = [...] "
                        "（推荐）或把 [terrain].curriculum 改为 true。"
                    )
                    self._curriculum_false_naming_warned = True
                return None

            # Standard 模式（curriculum=True）：复现 Isaac Lab 的 col → sub_terrain 映射
            # Standard mode (curriculum=True): replicate Isaac Lab's col→sub_terrain mapping
            import numpy as np

            proportions = np.array([cfg.proportion for cfg in sub_terrains.values()], dtype=float)
            if proportions.sum() < 1e-9:
                return None
            proportions = proportions / proportions.sum()
            num_cols = int(terrain_gen_cfg.num_cols)
            cumulative = np.cumsum(proportions)
            col_to_sub_idx = [int(np.min(np.where(col / num_cols + 0.001 < cumulative)[0])) for col in range(num_cols)]

            names = []
            for env_id in range(num_envs):
                col = col_indices[env_id] if env_id < len(col_indices) else 0
                col = max(0, min(col, num_cols - 1))
                sub_idx = col_to_sub_idx[col]
                sub_idx = max(0, min(sub_idx, len(sub_names) - 1))
                names.append(sub_names[sub_idx])
            return names
        except Exception as e:
            self.logger.warning(f"[video] 解析 env 地形名失败，mp4 将使用默认命名: {e}")
            return None

    def _update_camera_poses(self):
        """根据每只机器狗的位姿更新对应 TiledCamera 的位置与朝向。

        通过 root pose 为每个 env 单独计算相机 eye/target：
        - eye: 相对机器狗左后上方的第三人称位置
        - target: 机器狗机身前上方的观察点

        两者都会随机器人平移，并跟随机器人 yaw 旋转。
        必须在 env.step() 之前调用，这样本帧渲染时摄像机已在正确位置。
        """
        if not self._use_tiled_camera:
            return
        try:
            env_unwrapped = self._gym_env.unwrapped
            tiled_cam = env_unwrapped.scene["tiled_camera"]
            robot = env_unwrapped.scene["robot"]

            # 机器狗 root pose: (num_envs, 3) / (num_envs, 4)
            root_pos = robot.data.root_pos_w
            root_quat = robot.data.root_quat_w
            target_positions, target_targets = compute_follow_camera_view(
                root_pos=root_pos,
                root_quat=root_quat,
                eye_offset=self._camera_eye_offset,
                target_offset=self._camera_target_offset,
            )
            cam_positions, cam_targets = smooth_follow_camera_view(
                target_positions=target_positions,
                target_targets=target_targets,
                last_positions=self._camera_smoothed_positions,
                last_targets=self._camera_smoothed_targets,
                alpha=self._camera_smoothing_alpha,
            )
            self._camera_smoothed_positions = cam_positions.detach().clone()
            self._camera_smoothed_targets = cam_targets.detach().clone()

            # 使用 look-at API 直接设定朝向，避免手写四元数和 convention 不一致导致镜头朝天。
            tiled_cam.set_world_poses_from_view(cam_positions, cam_targets)
        except Exception as e:
            self.logger.error(f"TiledCamera 位置更新异常: {e}")

    def _write_camera_frames(self):
        """从 TiledCamera 读取 RGB 帧并写入 per-env 视频文件。在 env.step() 之后调用。

        跳过以下 env 的帧写入：
        - **Standard 模式**：已越界的 env（scorer._out_of_bounds），避免跨块污染视频。
          * 例外：maze 类地形（名称含 "maze"）的 env 不应用越界屏蔽，因为 maze 的
            出口就位于单块边界附近，越界属正常行为；若屏蔽会导致整段 mp4 为空壳。
        - **Track 模式**：不再使用 scorer._out_of_bounds（整条赛道跨多个子块，
          机器人离开首个子块是正常行为，不应停录）。
        - 评估模式下已完成首次 episode 的 env（_eval_env_done_mask），
          避免录入 auto-reset 后的无意义站立画面。
        """
        if self._tiled_video_writer is None:
            return
        try:
            env_unwrapped = self._gym_env.unwrapped
            tiled_cam = env_unwrapped.scene["tiled_camera"]
            # output["rgb"] shape: (num_envs, H, W, 3) or (num_envs, H, W, 4), dtype uint8
            rgb_data = tiled_cam.data.output["rgb"]

            # Lazy-resolve env terrain names on the very first write:
            # by now terrain.terrain_types has been populated by the first reset,
            # so the resolved names are trustworthy for mp4 filenames.
            # 首次写帧前懒解析 env 地形名：此时 terrain_types 已被首次 reset 事件填充，
            # 得到的名字可靠，用于 mp4 文件名前缀。writer 的 _writers 也是 lazy init
            # 的（见 write_frames 的 `if self._writers is None` 分支），因此只要在
            # 首次 write_frames 调用之前补齐 _env_terrain_names 即可生效。
            if self._tiled_video_writer._env_terrain_names is None:
                resolved_names = self._resolve_env_terrain_names()
                if resolved_names is not None:
                    self._tiled_video_writer._env_terrain_names = resolved_names
                    # 同步缓存到 base_env，供 skip_mask 的 maze 豁免逻辑复用
                    self._env_terrain_names_cache = resolved_names

            # 获取越界 mask：仅 standard 模式下使用（scorer 的越界基于单块边界，
            # track 模式下该语义不适用，否则会在机器人走出第一段子地形时误停录）
            skip_mask = None
            is_track_mode = bool(getattr(self._scorer, "_is_track_mode", False)) if self._scorer is not None else False
            if not is_track_mode and self._scorer is not None and hasattr(self._scorer, "_out_of_bounds"):
                skip_mask = self._scorer._out_of_bounds.clone()
                # maze 类地形的 env 不应用越界屏蔽（出口就在边界上，否则视频会是空壳）
                try:
                    names = getattr(self, "_env_terrain_names_cache", None)
                    if names is None:
                        names = self._resolve_env_terrain_names()
                        self._env_terrain_names_cache = names
                    if names is not None:
                        import torch as _torch

                        maze_flags = _torch.tensor(
                            [("maze" in (n or "").lower()) for n in names],
                            dtype=_torch.bool,
                            device=skip_mask.device,
                        )
                        # maze env 强制不跳过（False），非 maze env 维持原 out_of_bounds
                        if maze_flags.shape[0] == skip_mask.shape[0]:
                            skip_mask = skip_mask & (~maze_flags)
                except Exception as _e:
                    self.logger.warning(f"[video] maze 地形 skip_mask 豁免失败，退回默认行为: {_e}")

            # 评估模式：已完成首次 episode 的 env 也跳过帧写入
            if self._eval_env_done_mask is not None:
                if skip_mask is not None:
                    skip_mask = skip_mask | self._eval_env_done_mask
                else:
                    skip_mask = self._eval_env_done_mask

            self._tiled_video_writer.write_frames(rgb_data, skip_mask=skip_mask)
        except Exception as e:
            self.logger.error(f"TiledCamera 帧写入异常: {e}")

    def _finalize_video_recording(self):
        """关闭视频写入器，确保 MP4 尾信息落盘。"""
        if self._tiled_video_writer is None:
            return
        try:
            self._tiled_video_writer.close()
        except Exception as e:
            self.logger.warning(f"TiledCamera 视频写入器关闭异常: {e}")
        finally:
            self._tiled_video_writer = None

    def _init_goal_markers(self):
        """创建目标到达范围的绿色圆盘可视化标记（仅非 headless 模式）。

        圆盘半径 = _goal_reached_termination 的 threshold (0.6m)，
        直观显示"到达判定范围"。该值必须与 termination threshold 以及
        nav/hier_nav TOML 中 rewards.reach_goal.threshold 保持一致，
        避免"终止-奖励死区"。
        """
        if self.force_headless or self._goal_markers is not None or self._gym_env is None:
            return

        try:
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            import isaaclab.sim as sim_utils

            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/goal_markers",
                markers={
                    "goal": sim_utils.CylinderCfg(
                        radius=0.6,  # 与 _goal_reached_termination threshold 一致 (0.6m)
                        height=0.02,  # 极薄圆盘，贴地
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0),  # 绿色
                            opacity=0.6,
                        ),
                    ),
                },
            )
            self._goal_markers = VisualizationMarkers(marker_cfg)
            self._goal_markers_logged = False
        except Exception as e:
            self._goal_markers = None
            self.logger.warning(f"目标点可视化标记初始化失败: {e}")

    def _get_goal_marker_positions(self, env_unwrapped):
        """获取需要可视化的目标点位置。

        优先显示 track 固定序列中 `maze` 子地形的真实出口；
        若不可用，再退回到 env.goal_positions。
        """
        terrain = getattr(getattr(env_unwrapped, "scene", None), "terrain", None)
        terrain_cfg = getattr(terrain, "cfg", None) if terrain is not None else None
        terrain_type = getattr(terrain_cfg, "terrain_type", None)
        if terrain_type == "generator":
            try:
                from unitree_rl_lab.terrains.terrain_exit_manager import (
                    extract_named_track_exit_positions,
                    get_terrain_generator_runtime_state,
                )

                terrain_generator_state = get_terrain_generator_runtime_state(terrain)
                maze_exit_positions = extract_named_track_exit_positions(terrain_generator_state, "maze")
                if maze_exit_positions is not None and len(maze_exit_positions) > 0:
                    if not getattr(self, "_goal_markers_logged", False):
                        self.logger.info(
                            f"[goal_markers] maze 出口已提取: {maze_exit_positions.shape[0]} 个点, "
                            f"坐标范围 x=[{maze_exit_positions[:,0].min():.1f}, {maze_exit_positions[:,0].max():.1f}], "
                            f"y=[{maze_exit_positions[:,1].min():.1f}, {maze_exit_positions[:,1].max():.1f}]"
                        )
                        self._goal_markers_logged = True
                    marker_device = (
                        env_unwrapped.goal_positions.device
                        if hasattr(env_unwrapped, "goal_positions") and env_unwrapped.goal_positions is not None
                        else getattr(env_unwrapped, "device", "cpu")
                    )
                    return torch.as_tensor(maze_exit_positions, dtype=torch.float32, device=marker_device)
            except Exception as e:
                self.logger.warning(f"[goal_markers] 提取 maze 出口失败: {e}")

        if hasattr(env_unwrapped, "goal_positions") and env_unwrapped.goal_positions is not None:
            return env_unwrapped.goal_positions
        return None

    def _visualize_goal_markers(self):
        """刷新目标点圆盘位置，直接显示 env.goal_positions（与 _goal_reached_termination 一致）。"""
        if self._goal_markers is None or self._gym_env is None:
            return

        env_unwrapped = getattr(self._gym_env, "unwrapped", None)
        if env_unwrapped is None:
            return

        if not hasattr(env_unwrapped, "goal_positions") or env_unwrapped.goal_positions is None:
            return

        vis_pos = env_unwrapped.goal_positions
        # 跳过全零（尚未初始化）
        if vis_pos.abs().max().item() < 1e-6:
            return

        try:
            # 圆盘贴地：只需抬高 0.01m 防止 z-fighting
            vis_pos = vis_pos.clone()
            vis_pos[:, 2] = 0.01
            self._goal_markers.visualize(translations=vis_pos)

            # 首次成功时记录日志
            if not getattr(self, "_goal_markers_logged", False):
                self.logger.info(
                    f"[goal_markers] 显示 env.goal_positions: {vis_pos.shape[0]} 个圆盘, "
                    f"坐标范围 x=[{vis_pos[:,0].min():.1f}, {vis_pos[:,0].max():.1f}], "
                    f"y=[{vis_pos[:,1].min():.1f}, {vis_pos[:,1].max():.1f}]"
                )
                self._goal_markers_logged = True
        except Exception as e:
            self.logger.warning(f"[goal_markers] visualize 异常: {e}")

    def _shutdown_simulation_app(self):
        """vGPU 环境下跳过 SimulationApp 关闭，直接退出进程避免 segfault。"""
        self.logger.info("SimulationApp shutdown skipped (vGPU workaround), calling os._exit(0)")
        self._simulation_app = None
        self._app_launcher = None

    # ------------------------------------------------------------------
    # 内部: 初始化监控上报
    # ------------------------------------------------------------------
    def _init_monitor(self):
        """创建 BaseScorer 和 EnvMonitor 实例（仅执行一次）。"""
        if self._env_monitor is not None:
            return

        try:
            from tools.base_env.base_scorer import BaseScorer
            from tools.base_env.monitor import EnvMonitor

            env_unwrapped = self._gym_env.unwrapped
            self._scorer = BaseScorer(
                env_unwrapped=env_unwrapped,
                max_episode_length=self.max_episode_length,
                min_episode_length=self.min_episode_length,
                task_type=self._task_type,
                is_eval=self.is_eval,
                logger=self.logger,
            )
            self._env_monitor = EnvMonitor(
                scorer=self._scorer,
                logger=self.logger,
                flush_interval_sec=60,
                task_type=self._task_type,
            )
            self.logger.info(
                f"监控上报已初始化: task_type={self._task_type}, "
                f"地形类型={self._scorer.terrain_names}, "
                f"max_episode_length={self.max_episode_length}, "
                f"min_episode_length={self.min_episode_length}"
            )
        except Exception as e:
            self.logger.warning(f"监控上报初始化失败（不影响训练）: {e}")
            self._scorer = None
            self._env_monitor = None

    # ------------------------------------------------------------------
    # Helper: extract policy / critic obs from raw observations
    # ------------------------------------------------------------------
    def _extract_obs(self, obs_raw, extras=None):
        """Extract (policy_obs, critic_obs) from raw observation data.

        RslRlVecEnvWrapper may return obs as:
        - TensorDict with 'policy' and 'critic' keys
        - Plain torch.Tensor

        Returns:
            (obs, critic_obs): two torch.Tensor
        """
        obs = None
        critic_obs = None

        # Case 1: TensorDict (Isaac Lab >= 2.3 with Dict observation space)
        if hasattr(obs_raw, "keys") and callable(obs_raw.keys):
            if "policy" in obs_raw:
                obs = obs_raw["policy"]
            if "critic" in obs_raw:
                critic_obs = obs_raw["critic"]

        # Case 2: plain Tensor
        if obs is None:
            obs = obs_raw

        # Try extras for critic obs fallback
        if critic_obs is None and isinstance(extras, dict):
            obs_dict = extras.get("observations", {})
            if isinstance(obs_dict, dict):
                critic_obs = obs_dict.get("critic", None)

        if critic_obs is None:
            critic_obs = obs

        return obs, critic_obs

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, usr_conf):
        """Reset the environment.

        Args:
            usr_conf: User configuration dict.

        Returns:
            (obs, critic_obs) — two torch.Tensor on the sim device.
            Returns None on failure.
        """
        try:
            self.game_id = usr_conf.get("game_id", "1")
            self.is_eval = usr_conf.get("is_eval", False)
            self.usr_conf = usr_conf
            self._result_files_written = False

            # ---------- Parse configuration ----------
            env_conf = usr_conf.get("env_conf", {})
            env_section = usr_conf.get("env", {})
            terrain_section = usr_conf.get("terrain", {})
            task_name = env_conf.get("task_name", "Unitree-Go2-Velocity")

            entry_point_key = "play_env_cfg_entry_point" if self.is_eval else "env_cfg_entry_point"

            num_envs = env_section.get("num_envs", env_conf.get("num_envs", 4096))
            device = env_conf.get("device", "cuda:0")
            seed = env_conf.get("seed", 0)
            use_fabric = env_conf.get("use_fabric", True)

            # Task-specific parameters
            task_key = f"task_{task_name.lower().replace('-', '_')}"
            task_conf = env_conf.get(task_key, {})

            # Episode length
            self.min_episode_length = env_conf.get("min_episode_length", task_conf.get("min_episode_length", 50))
            # max_episode_length: prefer computing from episode_length_s (seconds → steps),
            # fallback to explicit max_step / max_episode_length, then default 1000.
            # dt=0.005, decimation=4 → step_dt=0.02
            episode_length_s = env_section.get("episode_length_s")
            if episode_length_s is not None:
                step_dt = 0.02  # dt(0.005) * decimation(4)
                self.max_episode_length = int(float(episode_length_s) / step_dt)
            else:
                self.max_episode_length = env_section.get(
                    "max_step", env_conf.get("max_episode_length", task_conf.get("max_episode_length", 1000))
                )

            # Video recording — only in evaluation mode
            self.is_need_save_mp4 = bool(env_conf.get("save_mp4", False))
            self.enable_video = self.is_eval and self.is_need_save_mp4
            self._reset_camera_follow_state()

            # Task type: "standard" or "track" — affects scoring and monitor reporting
            self._task_type = terrain_section.get("mode", "track").lower()

            # ---------- Launch SimulationApp (lazy, idempotent) ----------
            self._launch_sim_app(enable_cameras=self.enable_video)

            # ---------- Load environment configuration ----------
            _register_tasks()

            import unitree_rl_lab.tasks  # noqa: F401
            from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

            env_cfg = parse_env_cfg(
                task_name,
                device=device,
                num_envs=num_envs,
                use_fabric=use_fabric,
                entry_point_key=entry_point_key,
            )
            env_cfg.seed = seed

            # Episode length: prefer episode_length_s (seconds), fallback to max_step (steps)
            episode_length_s = env_section.get("episode_length_s")
            if episode_length_s is not None:
                env_cfg.episode_length_s = float(episode_length_s)
            else:
                env_cfg.episode_length_s = self.max_episode_length * env_cfg.sim.dt * env_cfg.decimation
            reward_configs = parse_reward_configs(usr_conf)

            # ---------- Apply user TOML config to env_cfg ----------
            # 将 terrain / commands / domain_rand / noise / init_state 等配置应用到 env_cfg
            apply_usr_conf_to_env_cfg(env_cfg, usr_conf, logger=self.logger)

            # ---------- Create environment ----------
            self._create_env(task_name, env_cfg, reward_configs=reward_configs)

            # ---------- Evaluation mode setup ----------
            if self.is_eval:
                self.battle_dir = f"/workspace/battle/{self.game_id}"
                os.makedirs(self.battle_dir, exist_ok=True)
                if self.is_need_save_mp4:
                    os.makedirs(f"{self.battle_dir}/mp4", exist_ok=True)

            # Record start time
            self.start_timestamp = datetime.datetime.now(datetime.timezone.utc)

            # ---------- Get initial observations ----------
            obs_data = self.env.get_observations()

            # rsl-rl >= 2.3 returns (obs, extras)
            if isinstance(obs_data, tuple):
                obs_raw, extras = obs_data
            else:
                obs_raw = obs_data
                extras = {}

            # Extract policy/critic obs
            obs, critic_obs = self._extract_obs(obs_raw, extras)

            self.frame_no = 1
            self.env_nums = num_envs
            self.logger.info(f"Environment reset: obs.shape={obs.shape}, critic_obs.shape={critic_obs.shape}")

            # 初始化目标点可视化（此时 goal_positions 已由 observation process 按需准备）
            self._init_goal_markers()
            self._visualize_goal_markers()

            # 初始化摄像机位置（第一帧）
            self._update_camera_poses()

            return (obs, critic_obs)

        except Exception as e:
            self.logger.exception("reset error")
            return None

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, actions):
        """Step the environment.

        Args:
            actions: torch.Tensor of shape (num_envs, num_actions)

        Returns:
            (frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs))
            Returns None on failure.
        """
        try:
            self.frame_no += 1

            # ------------------------------------------------------------------
            # 评估模式 per-env 单次生命：已完成的 env 的 action 归零
            # Isaac Lab auto-reset 后机器人会被重置，归零 action 让它原地站立不动，
            # 这样不会产生有意义的新 episode 数据。
            # ------------------------------------------------------------------
            if self.is_eval and self._eval_env_done_mask is not None:
                actions = actions.clone()
                actions[self._eval_env_done_mask] = 0.0

            # 保存 step 前快照（在 env.step 之前），
            # Isaac Lab 的 env.step() 内部会：
            #   1. auto-reset → episode_length_buf 清零
            #   2. curriculum_manager.compute() → terrain_levels 更新为新 level
            # scorer 需要 reset 前的真实值，故需在 step() 前拍快照。
            pre_step_episode_lengths = None
            pre_step_terrain_levels = None
            if self._env_monitor is not None:
                try:
                    env_unwrapped = self._gym_env.unwrapped
                    pre_step_episode_lengths = env_unwrapped.episode_length_buf.clone()
                    terrain = env_unwrapped.scene.terrain
                    if hasattr(terrain, "terrain_levels"):
                        pre_step_terrain_levels = terrain.terrain_levels.clone()
                except Exception:
                    pass

            # ---------- TiledCamera: step 前更新摄像机位置 ----------
            self._update_camera_poses()

            # RslRlVecEnvWrapper.step returns (obs, rewards, dones, extras)
            obs_raw, rewards, dones, extras = self.env.step(actions)

            # ------------------------------------------------------------------
            # Track mode: actively maintain env.goal_positions as terrain infra,
            # independent of whether observation_process includes the goal term.
            # This enables evaluating standard-trained (no-goal-obs) models on
            # track terrains without scorer/termination raising on missing goals.
            #
            # Track 模式：将 env.goal_positions 作为"地形基础设施"主动维护，
            # 不再依赖 observation_process 是否包含 goal 相关 term 才触发更新。
            # 支持用 standard 训练的模型（obs 里没有 goal）在 track 地形上评估，
            # scorer / termination / reward 都能拿到有效 goal_positions。
            # ------------------------------------------------------------------
            if self._task_type == "track":
                try:
                    from tools.base_env.observation_process import (
                        ensure_goal_positions_ready,
                        update_goal_positions,
                    )

                    env_unwrapped = self._gym_env.unwrapped
                    if ensure_goal_positions_ready(env_unwrapped):
                        update_goal_positions(env_unwrapped)
                except Exception as e:
                    if not getattr(self, "_goal_maintain_warned", False):
                        self.logger.warning(f"[track] 主动维护 goal_positions 失败（仅首次告警）: {e}")
                        self._goal_maintain_warned = True

            # Extract policy obs and critic/privileged obs
            obs, privileged_obs = self._extract_obs(obs_raw, extras)

            # 刷新目标点 marker；goal_positions 已在 observation 计算过程中更新
            self._visualize_goal_markers()

            # Build terminated and truncated tensors
            terminated = dones.bool() if isinstance(dones, torch.Tensor) else torch.tensor(dones, dtype=torch.bool)

            # Extract time_outs from extras for truncated
            time_outs = None
            if isinstance(extras, dict):
                time_outs = extras.get("time_outs", None)

            if time_outs is not None:
                truncated = (
                    time_outs.bool()
                    if isinstance(time_outs, torch.Tensor)
                    else torch.tensor(time_outs, dtype=torch.bool)
                )
                # Terminated should exclude time_outs (true termination vs truncation)
                terminated = terminated & ~truncated
            else:
                truncated = torch.zeros_like(terminated, dtype=torch.bool)

            # ------------------------------------------------------------------
            # 评估模式 per-env 单次生命：跟踪每个 env 的首次完成
            # ------------------------------------------------------------------
            # dones_for_scorer: 传递给 scorer 的 dones，仅包含首次完成的 env
            dones_for_scorer = dones
            if self.is_eval:
                dones_bool = dones.bool() if isinstance(dones, torch.Tensor) else torch.tensor(dones, dtype=torch.bool)
                # 延迟初始化 mask（首次 step 时 device 已确定）
                if self._eval_env_done_mask is None:
                    self._eval_env_done_mask = torch.zeros(
                        dones_bool.shape[0], dtype=torch.bool, device=dones_bool.device
                    )

                # 只有之前未完成的 env 中新产生的 done 才算"首次完成"
                newly_done = dones_bool & ~self._eval_env_done_mask
                if newly_done.any():
                    newly_done_ids = torch.where(newly_done)[0]

                    # ---------- Layer1 调试：打印首次完成 env 的具体终止原因 ----------
                    # Debug: print which termination term fired for each newly-done env.
                    # Helps diagnose "episode ends at first step" issues (e.g. maze terrain).
                    try:
                        env_unwrapped = self._gym_env.unwrapped
                        term_mgr = getattr(env_unwrapped, "termination_manager", None)
                        if term_mgr is not None:
                            # 兼容不同 Isaac Lab 版本：active_terms 可能是 list 也可能是 property
                            active_terms = getattr(term_mgr, "active_terms", None)
                            if active_terms is None:
                                active_terms = list(term_mgr._term_names) if hasattr(term_mgr, "_term_names") else []
                            for term_name in active_terms:
                                try:
                                    term_dones = term_mgr.get_term(term_name)
                                except Exception:
                                    continue
                                if term_dones is None:
                                    continue
                                fired = term_dones[newly_done_ids]
                                if fired.any():
                                    fired_envs = newly_done_ids[fired].tolist()
                                    self.logger.warning(
                                        f"[eval term-debug] step={self.frame_no} "
                                        f"term='{term_name}' fired on envs={fired_envs}"
                                    )
                    except Exception as e:
                        self.logger.error(f"[eval term-debug] 打印终止原因异常: {e}")
                    # ---------- Layer1 调试结束 ----------

                    self._eval_env_done_mask[newly_done_ids] = True
                    self.logger.info(
                        f"[eval single-life] {len(newly_done_ids)} env(s) finished first episode "
                        f"(total done: {self._eval_env_done_mask.sum().item()}/{self._eval_env_done_mask.shape[0]})"
                    )

                # scorer 只统计首次完成的 env
                dones_for_scorer = newly_done

            # ---------- 监控上报 ----------
            if self._env_monitor is not None:
                try:
                    env_unwrapped = self._gym_env.unwrapped
                    self._env_monitor.on_step(
                        env_unwrapped,
                        dones_for_scorer,
                        pre_step_episode_lengths=pre_step_episode_lengths,
                        pre_step_terrain_levels=pre_step_terrain_levels,
                    )
                except Exception as e:
                    self.logger.error(f"监控上报 on_step 异常: {e}")

            # ---------- TiledCamera 视频帧写入 ----------
            # 写帧放在 _eval_env_done_mask 更新之后：设计原则"保留现场"——
            # 若 env 首步即终止，_eval_env_done_mask 已置 True，skip_mask 会屏蔽本帧，
            # 生成的 0 帧空壳 mp4（~258B）本身就是问题现场证据，配合 [eval term-debug]
            # 日志可定位终止原因（不做"提前写一帧以便播放"的兜底，避免掩盖问题）。
            if self._use_tiled_camera:
                self._write_camera_frames()

            # DEBUG: 每500步打印 terrain level 和速度命令范围
            if self.frame_no % 500 == 0:
                try:
                    env_unwrapped = self._gym_env.unwrapped
                    terrain = env_unwrapped.scene.terrain
                    levels = terrain.terrain_levels.float()
                    print(
                        f"[step {self.frame_no}] terrain_level: "
                        f"mean={levels.mean():.2f}, max={levels.max():.0f}, "
                        f"min={levels.min():.0f}, "
                        f"distribution={torch.bincount(terrain.terrain_levels, minlength=10).tolist()}"
                    )
                    if self._task_type == "track" and hasattr(terrain, "terrain_types"):
                        types = terrain.terrain_types.long()
                        print(
                            f"[step {self.frame_no}] terrain_type(col/difficulty): "
                            f"mean={types.float().mean():.2f}, max={types.max():.0f}, "
                            f"min={types.min():.0f}, "
                            f"distribution={torch.bincount(types, minlength=10).tolist()}"
                        )
                    cmd_term = env_unwrapped.command_manager.get_term("base_velocity")
                    ranges = cmd_term.cfg.ranges
                    print(
                        f"[step {self.frame_no}] cmd_vel_range: "
                        f"lin_x={list(ranges.lin_vel_x)}, "
                        f"lin_y={list(ranges.lin_vel_y)}, "
                        f"ang_z={list(ranges.ang_vel_z)}"
                    )
                except Exception:
                    pass

            extra_info = {
                "result_code": 0,
                "result_message": "",
            }
            if extras:
                extra_info.update(extras)

            env_reward = {
                "env_id": self.game_id,
                "frame_no": self.frame_no,
                "reward": rewards,
            }
            env_obs = {
                "env_id": self.game_id,
                "frame_no": self.frame_no,
                "observation": obs,
                "extra_info": extra_info,
                "terminated": terminated,
                "truncated": truncated,
            }

            # ---------- 终止判断逻辑 ----------
            infos = {}
            terminated_flat = terminated.flatten()
            all_terminated = bool(terminated_flat.all())
            terminated_count = int(terminated_flat.sum())
            reached_max_length = self.frame_no > self.max_episode_length

            # 评估模式：所有 env 都完成过一次 episode 即触发 all_done
            eval_all_envs_done = (
                self.is_eval and self._eval_env_done_mask is not None and bool(self._eval_env_done_mask.all())
            )

            if eval_all_envs_done or all_terminated or reached_max_length:
                if eval_all_envs_done:
                    self.logger.info(
                        f"[eval single-life] All {self._eval_env_done_mask.shape[0]} envs "
                        f"have completed their first episode at frame {self.frame_no}"
                    )
                elif self.frame_no > 500 and self.frame_no % 500 == 0:
                    self.logger.info(
                        f"[Episode done] trigger: "
                        f"{'all envs terminated' if all_terminated else 'reached max length'}, "
                        f"frame_no={self.frame_no}, max_length={self.max_episode_length}"
                    )
                # 仅评估模式生成结果文件（battle_dir 仅在 is_eval 时初始化）
                if self.is_eval and not self._result_files_written:
                    try:
                        self.make_json_and_done_file()
                    except Exception as e:
                        self.logger.error(f"[Episode done] make result files failed: {e}")
                infos["all_done"] = True

            return (self.frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs))

        except Exception as e:
            self.logger.exception("step error")
            return None

    # ------------------------------------------------------------------
    # action / observation space
    # ------------------------------------------------------------------
    def get_action_space(self):
        try:
            return self.env.action_space
        except Exception as e:
            self.logger.exception("get_action_space error")
            return None

    def get_observation_space(self):
        try:
            return self.env.observation_space
        except Exception as e:
            self.logger.exception("get_observation_space error")
            return None

    # ------------------------------------------------------------------
    # 内部: 构建 end_info 评估数据
    # ------------------------------------------------------------------
    def _build_end_info(self) -> dict:
        """从 BaseScorer 中提取评估指标，构建 end_info 字典。

        返回结构（以 standard 模式为例）:
            {
                "total_score": 65.3,
                "forward_score": 78.2,
                "energy_score": 62.1,
                "pose_score": 55.7,
                "episode_count": 4096,
                "score_detail": [
                    {
                        "terrain_type": "pyramid_slope",
                        "total_score": 70.1,
                        "forward_score": 80.5,
                        "energy_score": 65.0,
                        "pose_score": 60.2,
                        "episode_count": 1024
                    },
                    ...
                ]
                "export_score_detail": [
                    {
                        "terrain_type": "pyramid_slope_level0",
                        "total_score": 70.1,
                        "forward_score": 80.5,
                        "energy_score": 65.0,
                        "pose_score": 60.2,
                        "episode_count": 1024
                    },
                    ...
                ]
            }
        track 模式下 forward_score 替换为 time_score。
        """
        import math

        def _safe_round(val, ndigits: int = 2, default: float = 0.0) -> float:
            """安全 round：NaN/Inf 替换为 default。

            默认 ndigits=2（分数类字段统一保留 2 位小数）。
            对于 completion_coeff 等比例字段，调用处显式传 ndigits=4 以保留更高精度。
            """
            v = float(val) if not isinstance(val, float) else val
            return round(v, ndigits) if math.isfinite(v) else default

        if self._scorer is None:
            return {}

        try:
            metrics = self._scorer.get_lifetime_metrics()
        except Exception as e:
            self.logger.error(f"_build_end_info: 获取 metrics 失败: {e}")
            return {}

        if not metrics:
            return {}

        # Episode-count helper (BUGFIX: no longer double-counts timeout in standard mode).
        # episode_count 计算辅助函数（修复：standard 模式不再重复计数 timeout）。
        #
        # Background (见 base_scorer.py:22-27 的口径约定):
        #   - track  : completed / abnormal / timeout 三类互斥，三者相加 = 总 episode 数
        #   - standard: completed + abnormal = 总 episode 数；timeout 是 abnormal 的子类
        #               （abnormal ⊇ timeout），单独可观测但不参与 episode 总数计算
        #
        # 背景：scorer 里 standard 模式对一次未走穿的 done，会同时 `abnormal += 1` 和
        # `timeout += 1`（当 is_timeout=True 时）。因此顶层 / 地形组 / 难度组的 episode
        # count 若按 c+a+t 累加，会把 timeout 重复计一次，导致 episode_count 虚高。
        def _episode_count_from(key_suffix: str = "") -> int:
            c = int(metrics.get(f"completed_count{key_suffix}", 0))
            a = int(metrics.get(f"abnormal_count{key_suffix}", 0))
            t = int(metrics.get(f"timeout_count{key_suffix}", 0))
            if self._task_type == "track":
                return c + a + t
            # standard: abnormal ⊇ timeout, so don't add timeout again.
            # standard：abnormal 已包含 timeout，不再重复累加。
            return c + a

        # ---- 全局 episode 计数 ----
        total_episode_count = _episode_count_from("")
        if total_episode_count == 0:
            return {}

        # ---- 全局平均分（顶层字段）----
        completed = int(metrics.get("completed_count", 0))
        end_info: dict = {
            "total_score": _safe_round(metrics.get("total_score", 0.0)),
            "pose_score": _safe_round(metrics.get("pose_score", 0.0)),
            "energy_score": _safe_round(metrics.get("energy_score", 0.0)),
            "episode_count": total_episode_count,
            "time_score": _safe_round(metrics.get("step_score", 0.0)),
        }

        # standard 模式包含 forward_score 和 time_score，track 模式包含 time_score
        if self._task_type == "standard":
            end_info["forward_score"] = _safe_round(metrics.get("forward_score", 0.0))

        if self._task_type == "track":
            end_info["completion_coeff"] = (
                _safe_round(completed / total_episode_count) if total_episode_count > 0 else 0.0
            )

        # ---- 按地形分组的 score_detail ----
        score_detail = []
        if self._task_type == "track":
            # track 模式：按难度档（列索引）分组，每条赛道是一个整体
            for col in range(self._scorer._num_cols):
                sfx = f"track_l{col}"
                col_episode_count = _episode_count_from(f"_{sfx}")
                if col_episode_count == 0:
                    continue

                col_completed = int(metrics.get(f"completed_count_{sfx}", 0))
                detail: dict = {
                    "terrain_type": f"track_level_{col}",
                    "total_score": _safe_round(metrics.get(f"total_score_{sfx}", 0.0)),
                    "completion_coeff": (
                        _safe_round(col_completed / col_episode_count, ndigits=4) if col_episode_count > 0 else 0.0
                    ),
                    "pose_score": _safe_round(metrics.get(f"pose_score_{sfx}", 0.0)),
                    "energy_score": _safe_round(metrics.get(f"energy_score_{sfx}", 0.0)),
                    "episode_count": col_episode_count,
                    "time_score": _safe_round(metrics.get(f"step_score_{sfx}", 0.0)),
                }
                score_detail.append(detail)
        else:
            # standard 模式：按子地形类型分组
            for terrain_name in self._scorer.terrain_names:
                terrain_episode_count = _episode_count_from(f"_{terrain_name}")
                if terrain_episode_count == 0:
                    continue

                terrain_completed = int(metrics.get(f"completed_count_{terrain_name}", 0))
                detail: dict = {
                    "terrain_type": terrain_name,
                    "total_score": _safe_round(metrics.get(f"total_score_{terrain_name}", 0.0)),
                    "pose_score": _safe_round(metrics.get(f"pose_score_{terrain_name}", 0.0)),
                    "energy_score": _safe_round(metrics.get(f"energy_score_{terrain_name}", 0.0)),
                    "episode_count": terrain_episode_count,
                    "forward_score": _safe_round(metrics.get(f"forward_score_{terrain_name}", 0.0)),
                    "time_score": _safe_round(metrics.get(f"step_score_{terrain_name}", 0.0)),
                }
                score_detail.append(detail)

        end_info["score_detail"] = score_detail

        # ---- 按 (地形, 难度级别) 二维分组的 export_score_detail ----
        export_score_detail = []
        if self._task_type == "track":
            # track 模式：export_score_detail 与 score_detail 相同（按难度档分组）
            # 因为 track 整条赛道是一个整体，不存在子地形×难度级别的二维展开
            export_score_detail = list(score_detail)
        else:
            # standard 模式：按 (子地形, 难度级别) 二维分组
            for terrain_name in self._scorer.terrain_names:
                for lv in range(self._scorer.num_levels):
                    sfx = f"{terrain_name}_l{lv}"
                    lv_episode_count = _episode_count_from(f"_{sfx}")
                    if lv_episode_count == 0:
                        continue

                    lv_completed = int(metrics.get(f"completed_count_{sfx}", 0))
                    export_detail: dict = {
                        "terrain_type": f"{terrain_name}_level{lv}",
                        "total_score": _safe_round(metrics.get(f"total_score_{sfx}", 0.0)),
                        "pose_score": _safe_round(metrics.get(f"pose_score_{sfx}", 0.0)),
                        "energy_score": _safe_round(metrics.get(f"energy_score_{sfx}", 0.0)),
                        "episode_count": lv_episode_count,
                        "forward_score": _safe_round(metrics.get(f"forward_score_{sfx}", 0.0)),
                        "time_score": _safe_round(metrics.get(f"step_score_{sfx}", 0.0)),
                    }
                    export_score_detail.append(export_detail)

        end_info["export_score_detail"] = export_score_detail

        self.logger.info(
            f"_build_end_info: task_type={self._task_type}, "
            f"total_score={end_info['total_score']}, "
            f"episode_count={total_episode_count}, "
            f"terrain_groups={len(score_detail)}, "
            f"export_detail_groups={len(export_score_detail)}"
        )

        return end_info

    # ------------------------------------------------------------------
    # 生成结果文件（评估模式）
    # ------------------------------------------------------------------
    def make_json_and_done_file(self):
        """生成 json 结果文件和 done 标记文件。"""
        if self._result_files_written:
            return

        self.game_status = 1
        end_timestamp = datetime.datetime.now(datetime.timezone.utc)

        # 构建 end_info 评估数据（从 BaseScorer 中提取）
        end_info_dict = self._build_end_info()
        end_info_str = json.dumps(end_info_dict, ensure_ascii=False) if end_info_dict else "{}"

        # 组装结果数据
        result_data = {
            "name": self.game_id,
            "project_code": "robot",
            "status": self.game_status,
            "camps": [
                {
                    "camp_type": "blue",
                    "camp_code": "A",
                    "start_info": "{}",
                    "end_info": end_info_str,
                }
            ],
            "frames": "{}",
            "start_time": self.start_timestamp.isoformat(),
            "end_time": end_timestamp.isoformat(),
        }

        # 先关闭写入器，确保 mp4 文件索引完整落盘，再执行压缩。
        mp4_dir = f"{self.battle_dir}/mp4"
        if self.is_need_save_mp4 and os.path.isdir(mp4_dir):
            self._finalize_video_recording()
            mp4_files = sorted(name for name in os.listdir(mp4_dir) if name.endswith(".mp4"))
            if mp4_files:
                zip_file = os.path.join(mp4_dir, f"{self.game_id}.mp4.zip")
                subprocess.run(
                    ["zip", "-j", zip_file, *mp4_files],
                    cwd=mp4_dir,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        # 写 json 文件
        json_file = f"{self.battle_dir}/result/{self.game_id}.json"
        os.makedirs(f"{self.battle_dir}/result", exist_ok=True)
        with open(json_file, "w") as outfile:
            json.dump(result_data, outfile, indent=4)

        # 写 done 文件
        done_file = f"{self.battle_dir}/{self.game_id}.done"
        with open(done_file, "w") as done:
            done.writelines("done")

        self._result_files_written = True
        self.logger.info(f"json_file {json_file} create success, done_file {done_file} create success")

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------
    def close(self):
        """关闭环境、SimulationApp 并回收显存资源。"""

        # 关闭监控上报（最终 flush）
        if self._env_monitor is not None:
            try:
                self._env_monitor.close()
            except Exception:
                pass
            self._env_monitor = None
            self._scorer = None

        # 关闭 TiledCamera 视频写入器
        self._finalize_video_recording()
        self._reset_camera_follow_state()

        # 丢弃目标点 marker 引用
        self._goal_markers = None

        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

        if self._gym_env is not None:
            try:
                self._gym_env.close()
            except Exception:
                pass
            self._gym_env = None

        # 关闭 SimulationApp
        self._shutdown_simulation_app()

        # 释放显存
        if torch.cuda.is_available():
            for _ in range(3):
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


# ======================================================================
# __main__: standalone test
# ======================================================================
if __name__ == "__main__":
    import time as _time

    # Evaluation configuration matching cluster KAIWU_FEATURE_CONFIG env_config
    # 评估配置，与集群 KAIWU_FEATURE_CONFIG 的 env_config 对齐
    usr_conf = {
        "game_id": "test_001",
        "is_eval": True,
        "env": {
            "num_envs": 5,
            "episode_length_s": 5.0,
        },
        "env_conf": {
            "task_name": "Unitree-Go2-Velocity",
            "device": "cuda:0",
            "seed": 42,
            "use_fabric": True,
            "save_mp4": True,
        },
        "video": {
            "save_video": True,
            "video_interval": 10,
        },
        "terrain": {
            "mode": "track",
            "num_rows": 10,
            "num_cols": 20,
            "difficulty_range": [0.0, 1.0],
            "standard": {
                "goal_distance": 4.0,
                "pyramid_stairs": {"proportion": 0.5},
                "pyramid_stairs_inv": {"proportion": 0.5},
                "maze": {"proportion": 0.0},
            },
            # "track": {
            #     "track_length": 5,
            #     "sub_terrains_random": False,
            #     "sub_terrains": ["pyramid_stairs", "pyramid_slope", "maze"],
            # },
        },
        "commands": {
            "limit": {
                "lin_vel_x": [0.0, 0.8],
                "lin_vel_y": [-0.3, 0.3],
                "ang_vel_z": [-1.5, 1.5],
            },
        },
    }

    env = Robot()
    try:
        data = env.reset(usr_conf)
        if data is None:
            print("Reset failed")
        else:
            obs, critic_obs = data
            print(f"Reset obs shape: {obs.shape}, critic_obs shape: {critic_obs.shape}")

            for step_idx in range(1000):
                actions = torch.randn(5, 12, device="cuda:0")
                result = env.step(actions)
                if result is None:
                    print(f"Step {step_idx} failed")
                    break
                frame_no, obs, rewards, terminated, truncated, (infos, priv_obs) = result
                print(f"Step {step_idx}: rewards mean={rewards.mean().item():.4f}")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Test finished")
        env.close()
