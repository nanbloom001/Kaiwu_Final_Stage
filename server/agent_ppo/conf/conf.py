#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os

import toml


# Valid task types (Isaac Lab native config format)
# 有效任务类型（Isaac Lab 原生配置格式）
_VALID_TASKS = {"standard", "track"}


class StageConfig:
    """
    Base class for training stage configuration.
    训练阶段配置基类。

    Subclass this and override fields to define a new training stage.
    继承此类并覆盖字段来定义新的训练阶段。
    """

    # --- Stage identity
    # 阶段标识 ---
    name = ""
    task_type = "standard"
    algorithm = "ppo"
    ckpt_name = ""  # Checkpoint filename prefix; empty falls back to model.ckpt-{id}.pkl

    # --- Model architecture dimensions (Isaac Lab Unitree-Go2-Velocity constants)
    # These are fixed by the Isaac Lab task definition and the network structure;
    # users are not expected to change them. Do NOT move them into user TOML.
    # 模型架构维度（Isaac Lab Unitree-Go2-Velocity 常量）
    # 由 Isaac Lab 任务定义与网络结构决定，用户不应修改；也不应放进用户 TOML。
    num_actions = 12  # Go2 joint action dim / Go2 关节动作维度
    num_proprio_obs = 45  # proprioceptive obs dim / 本体感知观测维度
    num_scan = 256  # 16x16 height-scan dim / 16x16 高度扫描维度
    num_critic_observations = 316  # proprio(45) + scan(256) + privileged(15)

    # --- Observation / model architecture
    # 观测 / 模型架构 ---
    # goal_obs 维度（loco/lbc_loco 均为 0；未来若引入 nav 由子类覆盖）
    num_goal_obs = 0

    model_class = "ActorCriticEncoder"
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"

    # Encoder 参数（ActorCriticEncoder 使用；height_scan 256 → latent 32）。
    # 与 LBC 教师 encoder 结构对齐，方便 lbc_loco 阶段按 key 前缀拆分加载。
    encoder_hidden_dims = [512, 256]
    latent_dim = 32
    # critic 是否共享 actor encoder（True 时 critic 输入 = c_proprio + latent + goal）
    critic_use_encoder = False

    # --- Training hyperparameters
    # 训练超参数 ---
    lr = 3e-4
    num_learning_epochs = 5
    num_mini_batches = 4
    num_steps_per_env = 48
    min_normalized_std = [0.05, 0.02, 0.05] * 4

    # --- Saving
    # 保存 ---
    model_save_interval = 500


class CustomConfig(StageConfig):
    # TODO: you can refer to LocomotionConfig / LBCLocoConfig to design your own
    # track-terrain navigation training stage. The following items need to be
    # specified:
    # 1. stage name;
    # 2. task_type;
    # 3. whether to use hierarchical training;
    # 4. semantics and dimension of the policy action;
    # 5. obs dimension (whether to concatenate goal information);
    # 6. training hyperparameters.
    #
    # After adding a new training stage, a corresponding training config file
    # must be created in the same directory.
    # Filename convention: train_env_conf_<task_type>_<stage.name>.toml
    # Refer to train_env_conf_standard_locomotion.toml as an example.
    #
    # TODO：可参考 LocomotionConfig / LBCLocoConfig 自行设计 track 地形导航训练阶段。
    # 需要明确：
    # 1. stage 名称；
    # 2. task_type；
    # 3. 是否采用分层训练；
    # 4. policy action 的语义和维度；
    # 5. obs 维度（是否拼接 goal 信息）；
    # 6. 训练超参。
    #
    # 新增训练阶段后，需在同目录创建对应训练配置文件。
    # 文件命名规则：train_env_conf_<task_type>_<stage.name>.toml
    # 可参考 train_env_conf_standard_locomotion.toml。
    pass


class LocomotionConfig(StageConfig):
    """
    Stage: locomotion — learn stable walking on mixed terrain.
    阶段：locomotion —— 在混合地形上学习稳定行走。

    使用 ActorCriticEncoder，actor 和 encoder 同时训练。
    训练完成后 encoder 可用于 Sim-to-Real 迁移（相机替代 height_scan），
    也是 LBC 蒸馏（lbc_loco 阶段）的教师权重来源。
    """

    name = "locomotion"
    task_type = "standard"
    algorithm = "ppo"
    ckpt_name = "model.ckpt-locomotion"
    critic_use_encoder = True


class StandardRefDistillConfig(StageConfig):
    """
    Stage: standard_ref_distill — distill a mature flat standard teacher into
    the mainline ActorCriticEncoder locomotion model.
    阶段：standard_ref_distill —— 将成熟的扁平 standard 参考模型先蒸馏成
    主线 ActorCriticEncoder 形态的 locomotion ckpt。

    Why this stage exists:
      reference ckpt actor input = [proprio(45) | height_scan(256)] = 301
      mainline LBC teacher needs encoder.* + actor.* keys

    Therefore the reference ckpt cannot be used directly by lbc_loco. This
    stage makes a compatible model.ckpt-locomotion-{id}.pkl first, then the
    existing lbc_loco stage can distill depth camera vision normally.
    """

    name = "standard_ref_distill"
    task_type = "standard"
    algorithm = "behavior_distill"
    ckpt_name = "model.ckpt-locomotion"
    critic_use_encoder = True

    # Adaptive action-supervised DAgger from the frozen reference actor.
    lr = 3e-4
    max_iterations = 6000
    num_steps_per_env = 24
    max_grad_norm = 1.0
    log_interval = 10
    model_save_interval = 100

    # Flat reference standard model dimensions.
    teacher_num_obs = 301
    teacher_num_critic_obs = 316
    teacher_actor_hidden_dims = [512, 256, 128]
    teacher_critic_hidden_dims = [512, 256, 128]
    teacher_activation = "elu"


class LBCLocoConfig(StageConfig):
    """
    Stage: lbc_loco — Vision distillation for locomotion (pure supervised).
    阶段：lbc_loco —— 运控视觉蒸馏（纯监督学习，非 RL）。

    Teacher (frozen locomotion ckpt) initially drives env, then a linear DAgger
    ramp transfers control to the student (VisionEncoder CNN+LSTM). The student
    learns teacher latent and action through the frozen Actor.
    教师（冻结的 locomotion ckpt）先驱动环境，再通过线性 DAgger ramp 把控制权
    交给学生；学生同时学习 latent、cosine 和冻结 Actor 后的 action。

    Loads locomotion ckpt (model.ckpt-locomotion-{id}.pkl) as teacher and splits
    encoder.* / actor.* by key prefix.
    从 locomotion ckpt 按 key 前缀拆分 encoder.* / actor.* 加载为教师。
    """

    name = "lbc_loco"
    task_type = "standard"
    algorithm = "lbc_loco"
    ckpt_name = "model.ckpt-lbc-loco"

    # LBC-specific training hyperparameters
    # LBC 专用训练超参数
    # Default lr, fallback only; actual value overridden by toml [lbc_loco].learning_rate
    # （lbc_workflow 启动时会读 toml 并 in-place 同步到 optimizer）。
    # 调整 lr 请直接改 toml，不需要动这里。
    lr = 1e-3
    lr_min = 1e-5
    # 平台任务页控制 10 小时；此值只是高于预计 10h 进度的安全上限。
    max_iterations = 20000
    # LBC 是纯监督流式 SGD：lbc_workflow 单 iteration 内跑 num_steps_per_env
    # 个环境步，每步立即调一次 algorithm.update() → 1 step = 1 梯度更新。
    num_steps_per_env = 24
    max_grad_norm = 1.0
    # Log MSE loss every N steps
    # 每 N 步记录一次 MSE loss
    log_interval = 100
    # 以完整 outer iteration 计数；当前实测约 2.64s/iter，225 ≈ 10min。
    model_save_interval = 225

    # VisionEncoder / DmEncoder dimensions
    # 网络维度（学生 + 教师）
    proprio_dim = 45
    # height_scan dimension (16×16)
    # height_scan 维度（16×16）
    scan_dim = 256
    depth_height = 180
    depth_width = 320
    depth_channels = 1
    cnn_output_dim = 32
    lstm_hidden_size = 64
    lstm_num_layers = 2
    latent_dim = 32  # teacher / student latent 维度（必须一致）

    # Teacher MLP（DmEncoder）隐层，采用宇树 PRD V3.1 设计
    teacher_encoder_hidden_dims = (512, 256)

    # Teacher Actor 结构（与 locomotion ckpt 的 ActorCritic.actor 保持一致）
    teacher_actor_hidden_dims = [512, 256, 128]
    teacher_actor_activation = "elu"


class Config:
    """
    Unified config entry point.
    统一配置入口。

    Set ``Config.CURRENT`` to a StageConfig subclass, then read
    hyperparameters via ``Config.CURRENT.lr``, ``Config.CURRENT.num_mini_batches``, etc.

    设置 ``Config.CURRENT`` 为某个 StageConfig 子类，然后通过
    ``Config.CURRENT.lr``、``Config.CURRENT.num_mini_batches`` 等读取超参数。

    Auto stage inference (based on TOML env_conf.task_name):
        task_name = "Unitree-Go2-Velocity"        + mode=standard -> LocomotionConfig
        task_name = "Unitree-Go2-Velocity-Camera" + mode=standard -> LBCLocoConfig
    """

    # Default stage; can be overridden by TOML env_conf.task_name during eval.
    # 阶段 4（深度视觉蒸馏）：训练入口固定为 lbc_loco。
    # 阶段 2 的 standard_ref_distill 已完成（daggerfull-16288），R2 分支的默认值
    # 会让训练继续进入特权桥接，读取错误的 TOML。视觉阶段必须默认 lbc_loco，
    # 才会加载 train_env_conf_standard_lbc_loco.toml 并走 lbc_workflow。
    # eval 时仍由 _infer_stage_from_task_name 按 task_name 覆盖。
    CURRENT = LBCLocoConfig

    @staticmethod
    def load_conf(logger):
        """
        Load user configuration file based on current stage.
        根据当前阶段加载用户配置文件。

        Args:
            logger: logger instance | 日志实例

        Returns:
            tuple: (usr_conf, usr_conf_file, is_eval, stage)
        """
        from common_python.config.config_control import CONFIG
        from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

        stage = Config.CURRENT
        task_type = stage.task_type

        if task_type not in _VALID_TASKS:
            raise ValueError(
                f"Invalid task_type '{task_type}' in stage '{stage.name}'. " f"Only {_VALID_TASKS} are supported."
            )

        # Determine if it's evaluation mode
        # 判断是否为评估模式
        is_eval = False
        if hasattr(CONFIG, "run_mode"):
            is_eval = CONFIG.run_mode in [
                KaiwuDRLDefine.RUN_MODE_EVAL,
                KaiwuDRLDefine.RUN_MODE_EXAM,
            ]

        if is_eval:
            usr_conf_file = f"tools/eval/conf/eval_env_conf.toml"
        else:
            usr_conf_file = f"agent_ppo/conf/train_env_conf_{task_type}_{stage.name}.toml"

        usr_conf = _load_conf(usr_conf_file, logger)

        if usr_conf is None:
            error_msg = f"usr_conf is None, please check {usr_conf_file}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # train_test 显存不足，将 num_envs 降为 1
        # reduce num_envs to 1 for train_test due to GPU memory
        if os.environ.get("KAIWU_TRAIN_TEST"):
            usr_conf["env"]["num_envs"] = 1
            logger.info("KAIWU_TRAIN_TEST detected, set num_envs to 1")

        # Eval-time stage override: infer stage from task_name and terrain mode.
        # 评估时按 task_name + terrain mode 推断 stage，覆盖 conf.py 顶部的 CURRENT，
        # 从而同一份代码可在平台上自动识别 loco / lbc_loco 两条评估路径。
        if is_eval:
            inferred = _infer_stage_from_task_name(usr_conf, logger)
            if inferred is not None and inferred is not stage:
                logger.info(
                    f"[eval] Override Config.CURRENT: {stage.name} -> {inferred.name} "
                    f"(inferred from TOML task_name)"
                )
                Config.CURRENT = inferred
                stage = inferred
                task_type = stage.task_type

        logger.info(f"Stage: {stage.name}, task_type: {task_type}, model: {stage.model_class}")

        return usr_conf, usr_conf_file, is_eval, stage


def _infer_stage_from_task_name(usr_conf, logger):
    """Infer StageConfig subclass from TOML env_conf.task_name + terrain.mode.

    Rules:
        task_name = "Unitree-Go2-Velocity"        + mode=standard -> LocomotionConfig
        task_name = "Unitree-Go2-Velocity-Camera" + mode=standard -> LBCLocoConfig

    Returns None if task_name is missing or unrecognized (fallback to Config.CURRENT).
    """
    if not isinstance(usr_conf, dict):
        return None
    env_conf = usr_conf.get("env_conf", {})
    task_name = env_conf.get("task_name")
    if not task_name:
        logger.info("[eval] task_name not set in TOML, fallback to Config.CURRENT")
        return None

    terrain_conf = usr_conf.get("terrain", {})
    mode = str(terrain_conf.get("mode", "standard")).lower()
    if mode != "standard":
        logger.warning(
            f"[eval] Only terrain.mode='standard' is supported; "
            f"got '{mode}', fallback to Config.CURRENT"
        )
        return None

    has_camera = "Camera" in task_name
    return LBCLocoConfig if has_camera else LocomotionConfig


def _deep_merge(base, override):
    """
    Recursively merge override dict into base dict.
    递归将 override 字典合并到 base 字典中（override 优先）。

    Args:
        base: Base config dictionary | 基础配置字典
        override: Override config dictionary | 覆盖配置字典

    Returns:
        dict: Merged config dictionary
    """
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_conf(conf_file, logger):
    """
    Load config: first load base TOML, then deep-merge user TOML on top.
    加载配置：先加载 base TOML，再用用户 TOML 覆盖合并。

    Base files provide model architecture dimensions (num_actions, num_proprio_obs, etc.)
    so user configs only need business-tunable parameters.
    Base 文件提供模型架构维度参数，用户配置只需保留业务可调参数。

    Args:
        conf_file: Path to the user TOML config file | 用户配置文件路径
        logger: Logger instance | 日志实例

    Returns:
        dict: Merged config dictionary, or None on failure
    """
    if not os.path.exists(conf_file):
        logger.error(f"Config file not found: {conf_file}")
        return None

    # Determine base file by mode (eval or train)
    # 根据模式选择 base 文件（eval 或 train）
    mode = "eval" if "eval" in conf_file else "train"
    base_file = os.path.join("tools", "conf", "base", f"{mode}_env_base.toml")

    # Load base config (optional — missing base is not fatal)
    # 加载 base 配置（可选 — base 缺失不致命）
    base_config = {}
    if os.path.exists(base_file):
        try:
            with open(base_file, "r", encoding="utf-8") as f:
                base_config = toml.load(f)
            logger.info(f"Loaded base config: {base_file}")
        except Exception as e:
            logger.warning(f"Cannot load base config: {base_file}. Error: {e}")

    # Load user config
    # 加载用户配置
    try:
        with open(conf_file, "r", encoding="utf-8") as f:
            user_config = toml.load(f)
        logger.info(f"Loaded user config: {conf_file}")
    except Exception as e:
        logger.error(f"Cannot load config file: {conf_file}. Error: {e}")
        return None

    # Deep merge: base ← user (user wins)
    # 深度合并：base ← user（用户配置优先）
    if base_config:
        config = _deep_merge(base_config, user_config)
    else:
        config = user_config

    return config
