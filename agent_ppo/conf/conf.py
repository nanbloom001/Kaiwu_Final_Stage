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


# ===========================================================================
# Track navigation stages
# Track 导航训练阶段（对应 nav_plan.md 第十三/二十章）
#
# 所有 nav 阶段统一使用：
#   - ActorCriticEncoder + goal_obs(3) + critic_use_encoder
#   - ckpt_name = "model.ckpt-nav"（计划3.2：所有 Track 阶段同前缀，平台始终识别为同一种模型）
#   - num_critic_observations = 319（=60 critic_proprio + 256 scan + 3 goal）
#
# 训练配置文件命名：train_env_conf_track_<name>.toml
# 模型文件统一：model.ckpt-nav-<GLOBAL_ID>.pkl（阶段信息不编码进文件名）
# ===========================================================================


class TrackNavCommonConfig(StageConfig):
    """Track 导航公共配置基类。

    所有 nav 阶段共享模型结构与基础 PPO 参数，子类只需覆盖 name / lr /
    对应阶段特有超参。ckpt_name 统一为 model.ckpt-nav。

    Policy raw obs: proprio(45) + height_scan(256) + goal(3) = 304
    Critic raw obs: critic_proprio(60) + height_scan(256) + goal(3) = 319
    Actor encoded input: proprio(45) + latent(32) + goal(3) = 80
    Critic encoded input: critic_proprio(60) + latent(32) + goal(3) = 95
    """

    task_type = "track"
    algorithm = "ppo"
    ckpt_name = "model.ckpt-nav"

    num_goal_obs = 3
    num_critic_observations = 319
    critic_use_encoder = True

    num_learning_epochs = 3
    num_mini_batches = 4
    num_steps_per_env = 48

    # 默认探索噪声范围，子类按阶段收紧
    init_noise_std = 0.80
    min_normalized_std = [0.05, 0.025, 0.05] * 4
    max_normalized_std = [0.30, 0.18, 0.30] * 4
    model_save_interval = 20


class TrackNavBaseConfig(TrackNavCommonConfig):
    """Stage: navbase — 基础导航，让机器人学会往终点走（对应复赛 xtrack1 起步）。

    预训练来源：standard locomotion ckpt（ActorCriticEncoder，actor 输入 77）。
    跨阶段迁移：actor.0.weight 前 77 列对齐（proprio+latent），goal 3 列零初始化。
    目标：completion 脱离 0。
    """

    name = "navbase"
    lr = 8.0e-5


class TrackNavSlopeConfig(TrackNavCommonConfig):
    """Stage 2.5: navslope — 坡地过渡（坡+反坡+迷宫，无楼梯）。

    从 navbase 的单段迷宫过渡到多段赛道，先只加坡和反坡（不含楼梯）。
    速度保持 navbase 水平 [0.45,0.65]，不提速。
    预训练来源：navbase 产物。
    目标：在坡地+迷宫的 3 段赛道上稳定完成，为 Stage3 加入楼梯做准备。
    """

    name = "navslope"
    lr = 5.0e-5


class TrackNavStableConfig(TrackNavCommonConfig):
    """Stage: navstable — 步态/楼梯/能耗稳定性精调（对应复赛 xtrack6-9）。

    加强足端抬脚、动作平滑、姿态、能耗、非脚接触惩罚，防止 Track 训练破坏运控。
    """

    name = "navstable"
    lr = 3.0e-5


class TrackNavStableHardConfig(TrackNavCommonConfig):
    """Stage 3A-Hard: 从已收敛 Stage3A 续训，加大中高难度地形采样。

    只改 difficulty + level_mix（Level4-6 占50%），不改速度/门控/reward。
    进入保守续训状态：lr 降到 2e-5，entropy 降到 0.002，收紧 desired_kl。
    预训练来源：Stage3A 产物（精确 ID 加载，禁止随机初始化）。
    目标：Level6 不再全为0，abnormal 下降，不破坏已有能力。
    """

    name = "navstable_hard"
    lr = 2.0e-5
    entropy_coef = 0.002
    desired_kl = 0.005
    model_save_interval = 10


class TrackNavStableAConfig(TrackNavCommonConfig):
    """Stage 3A: navstable_a — 完整赛道低速适应（对应复赛 xtrack3）。

    在完整 5 段赛道上低速 [0.45,0.65] 稳定行走，不开门控，低探索噪声。
    只用运控保持 + 基础导航两组 reward，不加足端/迷宫/近终点等复杂奖励。
    预训练来源：navbase 产物。
    目标：Level0/3 稳定完成、Level6 不全为0、单迷宫能力保持。
    """

    name = "navstable_a"
    lr = 3.0e-5
    entropy_coef = 0.004
    desired_kl = 0.005
    init_noise_std = 0.95
    min_normalized_std = [0.05, 0.025, 0.05] * 4
    max_normalized_std = [0.20, 0.12, 0.20] * 4


class TrackNavMazeConfig(TrackNavCommonConfig):
    """Stage: navmaze — 迷宫与近终点奖励精调（对应复赛 p04-06）。

    新增 maze context、wall collision/stall、corridor centering、near-goal 系列奖励。
    """

    name = "navmaze"
    lr = 1.5e-5


class TrackNavGateConfig(TrackNavCommonConfig):
    """Stage 3B: navgate — 在 Stage3A-hard 基础上开启地形速度门控。

    本阶段不提速、不修改奖励，只学习适应不同地形命令：
    - 平地/普通段：正常速度
    - 坡面：适度降速
    - 楼梯：明显降速
    - 迷宫：保持稳定导航速度
    """

    name = "navgate"
    lr = 1.5e-5
    entropy_coef = 0.0015
    desired_kl = 0.004
    min_normalized_std = [0.05, 0.025, 0.05] * 4
    max_normalized_std = [0.30, 0.18, 0.30] * 4
    model_save_interval = 10


class TrackNavScoreConfig(TrackNavCommonConfig):
    """Stage 3C: navspeed — 小幅提速，优化时间分（对应复赛 xtrack6）。

    完成率已满，本阶段只做小幅提速以提升 time_score。
    不改难度/门控/奖励，从 Stage3B-best 精确续训。
    """

    name = "navspeed"
    lr = 1.0e-5
    entropy_coef = 0.001
    desired_kl = 0.004
    min_normalized_std = [0.05, 0.025, 0.05] * 4
    max_normalized_std = [0.30, 0.18, 0.30] * 4
    model_save_interval = 10


class TrackNavX8AlignConfig(TrackNavScoreConfig):
    """Stage 3D: navx8align — xtrack8 配置对齐，提速 + 完整评分型 reward。

    适配最多128个并行环境，rollout 加长到 96 步补偿样本量。
    启用 xtrack8 的完整 reward 体系（评分型），降低完成奖励和失败惩罚极端值。
    分地形速度对齐 xtrack8，提速到 [0.55,0.80]。
    预训练来源：Stage3C-best checkpoint。
    """

    name = "navx8align"
    lr = 1.0e-5
    min_learning_rate = 5.0e-6
    max_learning_rate = 1.0e-5
    entropy_coef = 0.0015
    desired_kl = 0.004
    init_noise_std = 0.95
    num_steps_per_env = 96
    num_learning_epochs = 3
    num_mini_batches = 4
    model_save_interval = 10


class TrackNavX8D1Config(TrackNavScoreConfig):
    """Stage 3D-1: navx8d1 — 只对齐评分奖励，不提速。

    在 Stage3C 基础上只加 3 项评分 reward（pose_score/energy/command_speed_advantage），
    不提速、不改 sticky gate/level_mix/task_complete/termination。
    PPO 极保守（lr=5e-6 固定，entropy=5e-4，desired_kl=0.0025）。
    预训练来源：Stage3C-best checkpoint。
    """

    name = "navx8d1"
    lr = 5.0e-6
    min_learning_rate = 5.0e-6
    max_learning_rate = 5.0e-6
    entropy_coef = 5.0e-4
    desired_kl = 0.0025
    init_noise_std = 0.95
    num_steps_per_env = 96
    model_save_interval = 10


class TrackNavX8D2Config(TrackNavX8D1Config):
    """Stage 3D-2: navx8d2 — xtrack8 评分奖励第二步对齐。

    pose_score 提到 0.60，新增 score_guidance=0.15。
    进一步收紧 PPO（lr=3e-6 固定，entropy=3e-4，desired_kl=0.002）。
    预训练来源：Stage3D-1-best checkpoint。
    """

    name = "navx8d2"
    lr = 3.0e-6
    min_learning_rate = 3.0e-6
    max_learning_rate = 3.0e-6
    entropy_coef = 3.0e-4
    desired_kl = 0.002
    num_steps_per_env = 96
    num_learning_epochs = 3
    num_mini_batches = 4
    model_save_interval = 5


class TrackNavX8D2RConfig(TrackNavX8D1Config):
    """Stage 3D-2R: navx8d2r — 128环境迁移基线（受控实验）。

    完全保留 Stage3D-1 的全部 reward 和配置，
    只改训练规模（128env）和 PPO（lr=3e-6 固定，entropy=3e-4，desired_kl=0.002）。
    不加 score_guidance，不加任何新 reward。
    目的：隔离"环境数变化"的影响，作为后续加 reward 的对照基线。
    预训练来源：Stage3D-1-best checkpoint。
    """

    name = "navx8d2r"
    lr = 3.0e-6
    min_learning_rate = 3.0e-6
    max_learning_rate = 3.0e-6
    entropy_coef = 3.0e-4
    desired_kl = 0.002
    num_steps_per_env = 96
    num_learning_epochs = 3
    num_mini_batches = 4
    model_save_interval = 5


class TrackNavX8D4Config(TrackNavX8D1Config):
    """Stage 3D-4: navx8d4 — 128环境高难度能力保留适配。

    保持 Stage3D-1 全部 reward 和速度，只改：
    - 高难度 level_mix（L7-L9 从 24% 提到 35%，L9 从 5% 提到 12%）
    - PPO 极保守（lr=1e-6 固定，entropy=2e-4，desired_kl=0.0015，2 epochs）
    - rollout 加长到 128 步补偿 128 环境样本不足
    预训练来源：Stage3D-1-best checkpoint。
    """

    name = "navx8d4"
    lr = 1.0e-6
    min_learning_rate = 1.0e-6
    max_learning_rate = 1.0e-6
    entropy_coef = 2.0e-4
    desired_kl = 0.0015
    num_steps_per_env = 128
    num_learning_epochs = 2
    num_mini_batches = 4
    model_save_interval = 5


class TrackNavX7BridgeAConfig(TrackNavX8D1Config):
    """Stage 3E-1: navx7bridgea — 补齐 xtrack7/xtrack8 公共关节质量奖励。

    在 Stage3D-1 基础上只加 6 项关节质量 reward（joint_acc/dof_pos_limits/hip_to_default/
    joint_position_penalty/dof_vel/base_lateral_vel）。
    不提速、不改 gate/level_mix/task_complete/termination/PPO 参数。
    预训练来源：Stage3D-1-best checkpoint。
    """

    name = "navx7bridgea"
    lr = 5.0e-6
    min_learning_rate = 5.0e-6
    max_learning_rate = 5.0e-6
    entropy_coef = 5.0e-4
    desired_kl = 0.0025
    num_steps_per_env = 96
    num_learning_epochs = 3
    num_mini_batches = 4
    model_save_interval = 10


class TrackNavX7BridgeCConfig(TrackNavX7BridgeAConfig):
    """Stage3E-3：足端接触安全奖励桥接。

    Parent: Stage3E-1-late
    New rewards: feet_slide = -0.06, feet_stumble = -0.02
    Unchanged: PPO、速度、gate、level_mix 和其他奖励。
    """

    name = "navx7bridgec"
    lr = 5.0e-6
    min_learning_rate = 5.0e-6
    max_learning_rate = 5.0e-6
    entropy_coef = 5.0e-4
    desired_kl = 0.0025
    init_noise_std = 0.95
    num_steps_per_env = 96
    num_learning_epochs = 3
    num_mini_batches = 4
    model_save_interval = 5


class TrackNavX7BridgeDConfig(TrackNavX7BridgeCConfig):
    """Stage3E-4：高难度地形采样巩固。

    Parent: Stage3E-3 约1.5小时最佳 checkpoint
    Only change: L7/L8/L9 level_mix 0.12/0.07/0.05 → 0.08/0.08/0.08
    Rewards、速度、gate 和 PPO 全部继承 Stage3E-3。
    """

    name = "navx7bridged"
    lr = 5.0e-6
    min_learning_rate = 5.0e-6
    max_learning_rate = 5.0e-6
    entropy_coef = 5.0e-4
    desired_kl = 0.0025
    init_noise_std = 0.95
    num_steps_per_env = 96
    num_learning_epochs = 3
    num_mini_batches = 4
    model_save_interval = 5


class TrackNavX7BridgeEConfig(TrackNavX7BridgeDConfig):
    """Stage3E-5：加入保守权重的 feet_clearance。

    Parent: Stage3E-4 best checkpoint
    Only change: feet_clearance = 0.12
    """

    name = "navx7bridgee"


class TrackNavX7BridgeFConfig(TrackNavX7BridgeEConfig):
    """Stage3E-6：摆腿时间与节奏奖励桥接。

    Parent: Stage3E-5 best checkpoint
    New rewards: feet_air_time = 0.18, air_time_variance_penalty = -0.28
    """

    name = "navx7bridgef"


class TrackNavX7BridgeGConfig(TrackNavX7BridgeFConfig):
    """Stage3E-7：机身坐标系足端前摆奖励桥接。

    Parent: Stage3E-6 best checkpoint
    New reward: feet_swing_forward = 0.03（修正为机身航向坐标后启用）
    """

    name = "navx7bridgeg"


class TrackNavX7BridgeHConfig(TrackNavX7BridgeGConfig):
    """Stage3E-8：足端步态奖励权重对齐至xtrack8目标的约75%。

    Parent: Stage3E-7 best checkpoint
    Changed weights: feet_slide=-0.09, feet_stumble=-0.03, feet_air_time=0.26,
    air_time_variance_penalty=-0.41, feet_swing_forward=0.045
    """

    name = "navx7bridgeh"


class TrackNavX7Score1Config(TrackNavX7BridgeGConfig):
    """Stage3F-1：核心评分奖励桥接。

    Parent: Stage3E-7 best checkpoint
    Changes: score_guidance=0.25(新增), pose_score_formula=0.60,
    command_speed_advantage=0.45, track_lin_vel_xy=1.15, flat_orientation=-1.30
    足端reward使用E7权重（非E8）。
    """

    name = "navx7score1"


class TrackNavX7Nav1Config(TrackNavX7Score1Config):
    """Stage3G：导航奖励与回报尺度重平衡。

    Parent: Stage3F-1 90-min best checkpoint
    Added: forward_heading_velocity, goal_distance, reach_goal,
    wall_collision, wall_stall_penalty, wall_proximity, stuck_penalty
    Rebalanced: task_complete=220, termination=-7, undesired_contacts=-0.45
    """

    name = "navx7nav1"


class TrackNavX7Train1Config(TrackNavX7Nav1Config):
    """Stage3H：训练动力学与速度融合。

    Parent: Stage3G best checkpoint
    Changes: intermediate level_mix, intermediate terrain speeds, intermediate PPO.
    Rewards and navigation logic remain unchanged.
    """

    name = "navx7train1"
    lr = 1.0e-5
    min_learning_rate = 1.0e-5
    max_learning_rate = 1.0e-5
    entropy_coef = 1.0e-3
    desired_kl = 0.0032
    init_noise_std = 0.95
    num_steps_per_env = 64
    num_learning_epochs = 3
    num_mini_batches = 4
    model_save_interval = 5


class TrackNavNoGateConfig(TrackNavX7Train1Config):
    """Stage3I：Stage3H 无地形速度门控续训。

    所有 PPO 参数完全继承 Stage3H（Train1）。
    只在 TOML 中关闭门控开关，速度范围保持不变（隔离单变量）。
    """

    name = "navnogate"


class TrackNavNoGateSafeConfig(TrackNavNoGateConfig):
    """Stage3I-2：NoGate-Safe。

    Parent: Stage3I-1 NoGate best checkpoint.
    唯一变化：统一前进速度 [0.50,0.72] → [0.50,0.64]
    """

    name = "navx7nogate"


class TrackNavStage3J1Config(TrackNavNoGateSafeConfig):
    """Stage3J-1：NoGate Rough-Stability。

    Parent: Stage3I-2 NoGate-Safe 60min best checkpoint.
    保持 NoGate + 速度 [0.50,0.64]，仅增加保守粗糙地形稳定奖励组。
    """

    name = "navj1"


class TrackNavStage3J2Config(TrackNavStage3J1Config):
    """Stage3J-2：P04 Energy-Score Bridge。

    Parent: Stage3J-1 NoGate Rough-Stability 30min best.
    Only change: 启用低权重 energy_score_formula=0.28。
    """

    name = "navj2"


class TrackNavStage3J3Config(TrackNavStage3J1Config):
    """Stage3J-3：P05 Maze Safety。

    Parent: Stage3J-1 NoGate Rough-Stability 30min best.
    Changes: maze_anticipatory_turn=0.60, long_non_foot_contact=-8.0
    不含 energy_score_formula（J2被拒绝）。继承 J1，不继承 J2。
    """

    name = "navj3"


class TrackNavStage3J4Config(TrackNavStage3J1Config):
    """Stage3J-4：P06 Near-Goal Finish/Retreat。

    Parent: Stage3J-1 NoGate Rough-Stability 30min best.
    Changes: near_goal_finish_drive=0.70, near_goal_retreat_penalty=-0.90.
    不含 J2 energy_score_formula 或 J3 迷宫安全奖励。
    """

    name = "navj4"


class TrackNavStage3J8Config(TrackNavStage3J1Config):
    """Stage3J-8: Hard-Level Replay.

    Parent: Stage3J-1 NoGate Rough-Stability 30min best.
    Only change: increase the L7/L8/L9 training sample proportion.
    Rewards, PPO, commands, NoGate switches and model structure stay at J1.
    """

    name = "navj8"
    parent_checkpoint = "Stage3J-1 30min Best"


class TrackNavStage3J9Config(TrackNavStage3J1Config):
    """Stage3J-9: fixed-learning-rate continuation.

    Parent: Stage3J-1 30min Best.
    Only change: make the PPO learning rate truly fixed at 1e-5.
    """

    name = "navj9"
    parent_checkpoint = "Stage3J-1 30min Best"

    lr = 1.0e-5
    min_learning_rate = 1.0e-5
    max_learning_rate = 1.0e-5
    schedule = "fixed"

    model_save_interval = 5


class TrackNavX7BridgeBConfig(TrackNavX8D1Config):
    """Stage3E-2：低权重安全型关节奖励桥接。

    Parent: Stage3D-1-best
    Changes: joint_acc = -3.0e-7, dof_pos_limits = -0.15
    Unchanged: speed, gate, level_mix, completion rewards and PPO settings.
    """

    name = "navx7bridgeb"
    lr = 5.0e-6
    min_learning_rate = 5.0e-6
    max_learning_rate = 5.0e-6
    entropy_coef = 5.0e-4
    desired_kl = 0.0025
    init_noise_std = 0.95
    num_steps_per_env = 96
    num_learning_epochs = 3
    num_mini_batches = 4
    model_save_interval = 5


class TrackNavRecoveryConfig(TrackNavCommonConfig):
    """Stage: navrecovery — 极小学习率冻结微调（对应复赛 p22-r）。

    输入成熟 navscore checkpoint，lr=1e-7 只做短期恢复，不重新学习策略。
    """

    name = "navrecovery"
    lr = 1.0e-7
    min_learning_rate = 1.0e-7
    max_learning_rate = 1.0e-7
    init_noise_std = 0.95
    min_normalized_std = [0.05, 0.025, 0.05] * 4
    max_normalized_std = [0.30, 0.18, 0.30] * 4
    model_save_interval = 5


class TrackNavEvalConfig(TrackNavCommonConfig):
    """Stage: nav — 评估专用阶段。

    name="nav" 对应 train_env_conf_track_nav.toml（评估配置）。
    平台评估时按 terrain.mode="track" 自动推断到此 stage，
    只搜索 model.ckpt-nav-*.pkl 加载最新 Track 模型。
    """

    name = "nav"


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


class Config:
    """
    Unified config entry point.
    统一配置入口。

    Set ``Config.CURRENT`` to a StageConfig subclass, then read
    hyperparameters via ``Config.CURRENT.lr``, ``Config.CURRENT.num_mini_batches``, etc.

    设置 ``Config.CURRENT`` 为某个 StageConfig 子类，然后通过
    ``Config.CURRENT.lr``、``Config.CURRENT.num_mini_batches`` 等读取超参。

    Auto stage inference (based on TOML env_conf.task_name during eval):
        task_name = "Unitree-Go2-Velocity"        + mode=standard -> LocomotionConfig
        task_name = "Unitree-Go2-Velocity-Camera" + mode=standard -> LBCLocoConfig
        terrain.mode = "track" (non-Camera)                       -> TrackNavEvalConfig
    """

    # Default stage; can be overridden by TOML env_conf.task_name during eval.
    # 默认阶段；eval 时可由 TOML terrain.mode 推断覆盖。
    # 训练时设为当前段的阶段类，评估时自动推断到 TrackNavEvalConfig。
    CURRENT = TrackNavStage3J9Config

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
            inferred = _infer_stage_from_task_name(usr_conf, logger, stage)
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


def _infer_stage_from_task_name(usr_conf, logger, current_stage=None):
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

    # Track 导航评估：terrain.mode="track" 且非 Camera → TrackNavEvalConfig。
    # 只搜索 model.ckpt-nav-*.pkl 加载最新 Track 模型（计划 4.3）。
    if mode == "track" and not has_camera:
        logger.info("[eval] terrain.mode='track' -> TrackNavEvalConfig")
        return TrackNavEvalConfig

    if mode != "standard":
        logger.warning(
            f"[eval] Unsupported terrain.mode='{mode}'; fallback to Config.CURRENT"
        )
        return None

    if has_camera:
        return LBCLocoConfig

    # standard 非 Camera 任务：保留当前 stage，使 ckpt 命名与超参与实际训练的模型对齐。
    if (
        current_stage is not None
        and current_stage.task_type == "standard"
        and getattr(current_stage, "algorithm", "ppo") == "ppo"
        and current_stage.name not in {"locomotion"}
    ):
        logger.info(
            f"[eval] standard non-camera task; keep current stage '{current_stage.name}' "
            "instead of forcing locomotion"
        )
        return current_stage

    return LocomotionConfig


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
