#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os

try:
    import toml
except ModuleNotFoundError:
    import tomllib

    class _TomlCompat:
        @staticmethod
        def load(source):
            if hasattr(source, "read"):
                content = source.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                return tomllib.loads(content)
            with open(source, "rb") as file:
                return tomllib.load(file)

    toml = _TomlCompat()


def _load_toml(conf_file):
    return toml.load(conf_file)


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
    clip_param = 0.2
    entropy_coef = 0.01
    desired_kl = 0.01
    init_noise_std = 1.0
    min_normalized_std = [0.05, 0.02, 0.05] * 4
    max_normalized_std = [1.2, 0.8, 1.2] * 4

    # --- Saving
    # 保存 ---
    model_save_interval = 100


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
    # Use the platform-compatible filename model.ckpt-{id}.pkl.
    ckpt_name = ""
    critic_use_encoder = True


class StairInvFineTuneConfig(LocomotionConfig):
    """Fine-tune the locomotion policy on high-level inverse stairs."""

    name = "stair_inv_finetune"
    task_type = "standard"
    lr = 1e-4
    num_learning_epochs = 3
    num_mini_batches = 4
    num_steps_per_env = 48
    model_save_interval = 100


class TrackNavConfig(LocomotionConfig):
    """Track navigation fine-tune using the existing Encoder policy."""

    name = "nav"
    task_type = "track"
    num_goal_obs = 3
    lr = 1.5e-5
    num_learning_epochs = 3
    num_mini_batches = 4
    num_steps_per_env = 48
    entropy_coef = 0.0008
    desired_kl = 0.003
    init_noise_std = 0.80
    min_normalized_std = [0.05, 0.025, 0.05] * 4
    max_normalized_std = [0.24, 0.14, 0.24] * 4
    model_save_interval = 20


class TrackNavOpt5DebugConfig(TrackNavConfig):
    """ST7-Opt5 reset diagnostics; checkpoints from this stage are disposable."""

    name = "navopt5debug"
    parent_checkpoint = "ST7-Opt3 30min (evaluation 595729)"
    model_save_interval = 1000


class TrackNavOpt5BConfig(TrackNavConfig):
    """ST7-Opt5B: conservative hard-segment replay from Opt3-30min."""

    name = "navopt5b"
    parent_checkpoint = "ST7-Opt3 30min (evaluation 595729)"


class LBCLocoConfig(StageConfig):
    """
    Stage: lbc_loco — Vision distillation for locomotion (pure supervised).
    阶段：lbc_loco —— 运控视觉蒸馏（纯监督学习，非 RL）。

    Teacher (frozen locomotion ckpt) drives env, student (VisionEncoder CNN+LSTM)
    learns to reproduce teacher's latent from depth image via MSE loss.
    教师（冻结的 locomotion ckpt）驱动环境；
    学生（VisionEncoder CNN+LSTM）通过 MSE loss 学习从深度图复现教师 latent。

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
    max_iterations = 10000
    # LBC 是纯监督流式 SGD：lbc_workflow 单 iteration 内跑 num_steps_per_env
    # 个环境步，每步立即调一次 algorithm.update() → 1 step = 1 梯度更新。
    num_steps_per_env = 24
    max_grad_norm = 1.0
    # Log MSE loss every N steps
    # 每 N 步记录一次 MSE loss
    log_interval = 100
    # Save vision encoder ckpt every N steps
    # 每 N 步保存一次 vision encoder ckpt
    model_save_interval = 1000

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


class HJCNew10288LBCLocoConfig(LBCLocoConfig):
    """Legacy visual-stage label for an encoder-based Standard teacher.

    Compatibility is determined from checkpoint keys and tensor shapes, not
    from the numeric ID. A checkpoint with encoder.* and a 77-D Actor is valid;
    a legacy 301-D flat checkpoint is not.
    """

    name = "hjcnew10288_lbc_loco"
    task_type = "standard"
    ckpt_name = "model.ckpt-hjcnew"
    num_goal_obs = 0


class StandardRefDistillConfig(LocomotionConfig):
    """Optional bridge for a legacy 301-D flat Standard teacher.

    This is not the active 10288 path. Use it only when checkpoint inspection
    proves that the selected teacher has no encoder.* keys.
    """

    name = "standard_ref_distill"
    algorithm = "behavior_distill"
    parent_checkpoint = "legacy flat Standard ActorCritic checkpoint"
    ckpt_name = "model.ckpt-locomotion"

    lr = 3e-4
    num_steps_per_env = 24
    max_iterations = 5000
    max_grad_norm = 1.0
    log_interval = 50
    model_save_interval = 500

    teacher_num_obs = 301
    teacher_num_critic_obs = 316
    teacher_actor_hidden_dims = [512, 256, 128]
    teacher_critic_hidden_dims = [512, 256, 128]
    teacher_activation = "elu"


class StandardDistill1Config(LBCLocoConfig):
    """Depth-camera LBC from the platform-selected Standard 10288 teacher.

    Platform inspection shows an ActorCriticEncoder checkpoint with
    encoder.*, actor.*, critic_encoder.* and critic.* keys. LBC consumes only
    encoder.* and actor.*; critic-side keys are intentionally ignored.
    """

    name = "standard_distill_1"
    task_type = "standard"
    num_goal_obs = 0
    parent_checkpoint = "platform Standard ActorCriticEncoder checkpoint 10288"
    ckpt_name = "model.ckpt-standard"


class StandardDistill2StairConfig(StandardDistill1Config):
    """STD-D2-Stair: stair-heavy visual continuation from Standard D1."""

    name = "standard_distill_2_stair"
    parent_checkpoint = "current accepted Standard visual student checkpoint"


class TrackLBCLocoConfig(LBCLocoConfig):
    """Depth-camera distillation for the UWB-guided TrackNav teacher."""

    name = "track_lbc_loco"
    task_type = "track"
    ckpt_name = "model.ckpt-track-lbc-loco"
    num_goal_obs = 3


class TrackLBCLocoD2Config(TrackLBCLocoConfig):
    """ST9-Opt3-D2 action-aware, closed-loop visual distillation.

    Parent checkpoint: ST9-Opt3-D1 visual student. The architecture and output
    checkpoint label remain compatible with the existing Track LBC model.
    """

    name = "track_lbc_loco_d2"
    parent_checkpoint = "ST9-Opt3-D1 visual student"


class Config:
    """
    Unified config entry point.
    统一配置入口。

    Set ``Config.CURRENT`` to a StageConfig subclass, then read
    hyperparameters via ``Config.CURRENT.lr``, ``Config.CURRENT.num_mini_batches``, etc.

    设置 ``Config.CURRENT`` 为某个 StageConfig 子类，然后通过
    ``Config.CURRENT.lr``、``Config.CURRENT.num_mini_batches`` 等读取超参数。

    Training and evaluation both use the explicitly selected ``CURRENT`` stage.
    Platform evaluation TOML describes the environment, not the checkpoint
    architecture, so it must not silently switch a Track LBC model back to the
    77-D standard LBC actor.
    """

    # Explicit stage selector for both training and evaluation.
    # 训练和评估均显式使用该阶段，避免环境名称误改模型结构。
    # STD-D2-Stair is the active platform training stage. D1 and other stages
    # remain reproducible through their independent TOML files.
    CURRENT = StandardDistill2StairConfig

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

        if is_eval:
            task_name = usr_conf.get("env_conf", {}).get("task_name", "")
            mode = usr_conf.get("terrain", {}).get("mode", "")
            logger.info(
                f"[eval] Keep explicit stage '{stage.name}' for task "
                f"'{task_name}' (terrain.mode='{mode}')"
            )

            # The platform evaluation TOML does not necessarily carry the
            # camera mount calibration. Keep the explicitly selected model
            # stage and inject its training-time camera section so evaluation
            # uses the same geometry as visual distillation.
            train_toml = (
                f"agent_ppo/conf/train_env_conf_{task_type}_{stage.name}.toml"
            )
            if os.path.exists(train_toml):
                try:
                    train_conf = _load_toml(train_toml)
                    camera_conf = train_conf.get("camera")
                    if camera_conf:
                        existing = usr_conf.get("camera")
                        usr_conf["camera"] = (
                            _deep_merge(existing, camera_conf)
                            if isinstance(existing, dict)
                            else camera_conf
                        )
                        logger.info(
                            f"[eval] Injected [camera] override from {train_toml}"
                        )
                except Exception as exc:
                    logger.warning(
                        f"[eval] Failed to inject [camera] from {train_toml}: {exc}"
                    )

        logger.info(f"Stage: {stage.name}, task_type: {task_type}, model: {stage.model_class}")
        parent_checkpoint = getattr(stage, "parent_checkpoint", None)
        if parent_checkpoint:
            logger.info(f"Parent checkpoint required: {parent_checkpoint}")

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
    has_camera = "Camera" in task_name
    if mode == "track":
        return TrackLBCLocoConfig if has_camera else TrackNavConfig
    if mode != "standard":
        logger.warning(
            f"[eval] Unsupported terrain.mode='{mode}', "
            f"fallback to Config.CURRENT"
        )
        return None

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
