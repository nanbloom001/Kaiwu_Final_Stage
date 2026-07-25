#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import glob
import shutil
import copy

import numpy as np
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

import torch.optim as optim

from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.feature.definition import ActData
from agent_ppo.conf.conf import Config
from agent_ppo.model.actor_critic_encoder import ActorCriticEncoder
from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO
from agent_ppo.checkpoint_io import (
    KAIWU_TRAIN_FORMAT,
    checkpoint_candidates,
    is_kaiwu_train_bundle,
    low_level_policy_state,
    validate_low_level_spec,
    validate_probe_filename,
    vision_checkpoint_candidates,
    visual_anchor_r2_checkpoint_candidates,
    visual_anchor_r2_eval_candidates,
    visual_anchor_r2_parent_candidates,
    visual_rl_checkpoint_candidates,
)
from tools.train_env_conf_validate import check_usr_conf


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device="cuda", logger=None, monitor=None):
        self.cur_model_name = "ActorCriticEncoder"
        self.device = device
        self.logger = logger
        self.monitor = monitor

        usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(self.logger)
        valid, message = check_usr_conf(usr_conf, is_eval, self.logger)
        if not valid:
            self.logger.error(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")
            raise Exception(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")

        self.is_eval = is_eval
        self.stage = stage
        # Cache usr_conf so later platform callbacks (load_model/save_model)
        # can read [env_conf].seed and stage sub-tables without re-parsing TOML.
        self.usr_conf = usr_conf
        env_conf = usr_conf["env"]
        self.num_envs = env_conf["num_envs"]

        # Model architecture dims come from StageConfig (architecture constants,
        # not user-tunable business params). Do NOT read them from TOML.
        # 模型架构维度来自 StageConfig（架构常量，非业务可调参数），不从 TOML 读。
        self.num_actions = stage.num_actions
        self.num_critic_obs = stage.num_critic_observations

        num_proprio = stage.num_proprio_obs
        num_scan = stage.num_scan
        num_goal_obs = getattr(stage, "num_goal_obs", 0)

        # Policy obs layout:
        #   [proprio(45) | height_scan(256)]  = 301 D
        # LBC 阶段观测在 LBCObservationProcess 尾部再拼接 depth，
        # 由 AlgorithmLBC._split_obs 内部切分，不影响这里的 num_obs。
        # 策略观测 = 本体感知 + 扫描 + goal_obs
        self.num_obs = num_proprio + num_scan + num_goal_obs

        # Critic obs layout: [critic_proprio | height_scan] = 316 D
        # critic 观测 = critic_proprio + height_scan
        self.num_critic_obs = stage.num_critic_observations + num_goal_obs

        # Algorithm dispatch
        # 算法分发
        self.algorithm_name = getattr(stage, "algorithm", "ppo")
        self.is_lbc = self.algorithm_name == "lbc_loco"
        self.is_behavior_distill = self.algorithm_name == "behavior_distill"
        self.is_visual_ppo = self.algorithm_name == "visual_ppo"

        if self.is_lbc:
            # LBC 阶段：创建学生 + 教师；不初始化 PPO storage
            self._init_lbc_loco(num_proprio, num_scan, env_conf, stage, usr_conf)
        elif self.is_visual_ppo:
            depth_size = (
                stage.depth_height * stage.depth_width * stage.depth_channels
            )
            self.num_obs = num_proprio + num_scan + depth_size
            self._init_visual_ppo(num_proprio, num_scan, stage, usr_conf)
        else:
            self._init_flat(num_proprio, num_scan, stage)
            if self.is_behavior_distill:
                self._init_behavior_distill(stage, usr_conf)

        self.num_steps_per_env = stage.num_steps_per_env
        self.save_interval = stage.model_save_interval

        # LBC / behavior_distill: 无 PPO storage 需要初始化
        if not (self.is_lbc or self.is_behavior_distill):
            # Initialize storage
            # 初始化存储
            self.algorithm.init_storage(
                self.num_envs,
                self.num_steps_per_env,
                actor_obs_shape=(self.num_obs,),
                critic_obs_shape=(self.num_critic_obs,),
                action_shape=(self.num_actions,),
                device=self.device,
            )
            if self.is_visual_ppo:
                self.algorithm.initialize_recurrent_states(self.num_envs)

        super().__init__(agent_type, device, logger, monitor)

    def _init_flat(self, num_proprio, num_scan, stage):
        """
        Initialize single-model (flat) architecture.
        初始化单模型（扁平）架构。

        Asymmetric AC:
          actor : proprio + actor_encoder(scan)                         →  proprio + latent
          critic_use_encoder=False: critic_proprio + scan (raw)         → 316 D
          critic_use_encoder=True : critic_proprio + critic_encoder(scan) → critic_proprio + latent
        """
        num_goal_obs = getattr(stage, "num_goal_obs", 0)
        critic_use_encoder = bool(getattr(stage, "critic_use_encoder", False))

        # Encoder parameters
        latent_dim = getattr(stage, "latent_dim", 32)
        encoder_hidden_dims = getattr(stage, "encoder_hidden_dims", [512, 256])

        # critic_obs 排布: [c_proprio | h_scan]
        critic_proprio_dim = stage.num_critic_observations - num_scan  # 60
        scan_critic_start = critic_proprio_dim                         # 60
        scan_critic_end = scan_critic_start + num_scan                 # 316

        if critic_use_encoder:
            num_critic_input = critic_proprio_dim + latent_dim + num_goal_obs
            critic_scan_slice = (scan_critic_start, scan_critic_end)
        else:
            num_critic_input = self.num_critic_obs
            critic_scan_slice = None

        self.model = ActorCriticEncoder(
            num_proprio=num_proprio,
            num_scan=num_scan,
            num_critic_input=num_critic_input,
            num_actions=self.num_actions,
            num_goal_obs=num_goal_obs,
            critic_use_encoder=critic_use_encoder,
            critic_scan_slice=critic_scan_slice,
            encoder_hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            actor_hidden_dims=stage.actor_hidden_dims,
            critic_hidden_dims=stage.critic_hidden_dims,
            activation=stage.activation,
        ).to(self.device)

        self.logger.info(f"Actor Encoder : {self.model.encoder}")
        if critic_use_encoder:
            self.logger.info(f"Critic Encoder: {self.model.critic_encoder}")
        self.logger.info(f"Actor MLP : {self.model.actor}")
        self.logger.info(f"Critic MLP: {self.model.critic}  (input={num_critic_input})")

        params = [{"params": self.model.parameters(), "name": "actor_critic"}]
        self.optimizer = optim.Adam(params, lr=stage.lr)

        self.algorithm = AlgorithmPPO(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            logger=self.logger,
            monitor=self.monitor,
            learning_rate=stage.lr,
            num_mini_batches=stage.num_mini_batches,
            num_learning_epochs=stage.num_learning_epochs,
        )

    def _init_behavior_distill(self, stage, usr_conf):
        """Replace PPO algorithm with behavior distillation around the same student model."""
        from agent_ppo.algorithm.algorithm_behavior_distill import AlgorithmBehaviorDistill

        distill_conf = usr_conf.get(stage.name, {}) if isinstance(usr_conf, dict) else {}
        teacher_ckpt = distill_conf.get("teacher_ckpt")

        learning_rate = float(distill_conf.get("learning_rate", stage.lr))
        max_grad_norm = float(distill_conf.get("max_grad_norm", stage.max_grad_norm))
        action_loss_weight = float(distill_conf.get("action_loss_weight", 1.0))
        replay_conf = distill_conf.get("replay", {})

        self.algorithm = AlgorithmBehaviorDistill(
            student=self.model,
            teacher_ckpt=teacher_ckpt,
            device=self.device,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            action_loss_weight=action_loss_weight,
            num_obs=stage.teacher_num_obs,
            num_critic_obs=stage.teacher_num_critic_obs,
            num_actions=stage.num_actions,
            teacher_actor_hidden_dims=stage.teacher_actor_hidden_dims,
            teacher_critic_hidden_dims=stage.teacher_critic_hidden_dims,
            teacher_activation=stage.teacher_activation,
            replay_capacity=int(replay_conf.get("capacity", 8192)),
            replay_batch_size=int(replay_conf.get("batch_size", 256)),
            replay_loss_ratio=float(replay_conf.get("loss_ratio", 0.25)),
            replay_add_per_step=int(replay_conf.get("add_per_step", 64)),
            logger=self.logger,
        )
        self.optimizer = self.algorithm.optimizer
        self.logger.info(
            f"[BehaviorDistill] teacher={teacher_ckpt or '<platform selected preload>'}, "
            f"student={self.model.__class__.__name__}"
        )

    def _init_lbc_loco(self, num_proprio, num_scan, env_conf, stage, usr_conf):
        """
        Initialize LBC Loco stage (vision distillation, pure supervised).
        初始化 LBC Loco 阶段（视觉蒸馏，纯监督学习）。

        - Teacher: DmEncoder + teacher_actor (frozen, loaded from locomotion ckpt)
        - Student: VisionEncoder (CNN+LSTM), trainable
        - Algorithm: AlgorithmLBC (pure supervised MSE)
        - 教师：DmEncoder + teacher_actor（冻结，从 locomotion ckpt 加载）
        - 学生：VisionEncoder (CNN+LSTM)，可训练
        - 算法：AlgorithmLBC（纯监督 MSE）

        Does not create ActorCriticEncoder / AlgorithmPPO / RolloutStorage。
        """
        import torch.nn as _nn
        from agent_ppo.model.vision_encoder import DmEncoder, VisionEncoder
        from agent_ppo.algorithm.algorithm_lbc import AlgorithmLBC

        # 1. 学生：VisionEncoder (CNN + LSTM)
        #    LSTM 输入 = cat(cnn_feat, proprio)
        self.vision_encoder = VisionEncoder(
            image_shape=(stage.depth_height, stage.depth_width, stage.depth_channels),
            proprio_dim=stage.proprio_dim,
            cnn_output_dim=stage.cnn_output_dim,
            rnn_hidden_dim=stage.lstm_hidden_size,
            rnn_num_layers=stage.lstm_num_layers,
            rnn_output_dim=stage.latent_dim,
            use_lstm=True,
        ).to(self.device)

        # 2. 教师 Encoder：DmEncoder（与 ActorCriticEncoder.encoder 结构一致，便于权重拆分）
        # 注：DmEncoder / VisionEncoder 内部强制 L2 归一化，无需也不可关闭
        # （teacher_actor 依赖单位向量输入）
        self.teacher_encoder = DmEncoder(
            input_dim=stage.scan_dim,
            hidden_dims=tuple(stage.teacher_encoder_hidden_dims),
            output_dim=stage.latent_dim,
        ).to(self.device)

        # 3. 教师 Actor：复刻 ActorCriticEncoder.actor 结构
        #    actor = Sequential(Linear(77,h0), Act, Linear(h0,h1), Act, Linear(h1,h2), Act, Linear(h2,num_actions))
        activation_map = {"elu": _nn.ELU, "relu": _nn.ReLU, "tanh": _nn.Tanh}
        Act = activation_map.get(stage.teacher_actor_activation, _nn.ELU)

        actor_input_dim = stage.proprio_dim + stage.latent_dim
        hidden = list(stage.teacher_actor_hidden_dims)
        layers = []
        prev = actor_input_dim
        for h in hidden:
            layers.append(_nn.Linear(prev, h))
            layers.append(Act())
            prev = h
        layers.append(_nn.Linear(prev, self.num_actions))
        self.teacher_actor = _nn.Sequential(*layers).to(self.device)

        self.logger.info(f"[LBC-Loco] VisionEncoder:\n{self.vision_encoder}")
        self.logger.info(f"[LBC-Loco] Teacher Encoder (DmEncoder):\n{self.teacher_encoder}")
        self.logger.info(f"[LBC-Loco] Teacher Actor:\n{self.teacher_actor}")

        # 4. AlgorithmLBC（优化器只含 vision_encoder）
        self.algorithm = AlgorithmLBC(
            vision_encoder=self.vision_encoder,
            teacher_encoder=self.teacher_encoder,
            teacher_actor=self.teacher_actor,
            device=self.device,
            learning_rate=stage.lr,
            latent_dim=stage.latent_dim,
            max_grad_norm=stage.max_grad_norm,
            proprio_dim=stage.proprio_dim,
            scan_dim=stage.scan_dim,
            depth_shape=(stage.depth_height, stage.depth_width, stage.depth_channels),
        )

        # 为与 PPO 路径下某些属性兼容，point self.model 到学生
        self.model = self.vision_encoder

    def _init_visual_ppo(self, num_proprio, num_scan, stage, usr_conf):
        """Initialize Stage-5 recurrent visual PPO from a frozen S0 preload."""
        from agent_ppo.algorithm.algorithm_visual_ppo import AlgorithmVisualPPO
        from agent_ppo.model.visual_actor_critic import VisualActorCritic

        visual_conf = usr_conf.get(stage.name, {})
        self.model = VisualActorCritic(
            num_proprio=num_proprio,
            num_scan=num_scan,
            depth_shape=(
                stage.depth_height,
                stage.depth_width,
                stage.depth_channels,
            ),
            latent_dim=stage.latent_dim,
            cnn_output_dim=stage.cnn_output_dim,
            lstm_hidden_size=stage.lstm_hidden_size,
            lstm_num_layers=stage.lstm_num_layers,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            actor_hidden_dims=stage.actor_hidden_dims,
            critic_hidden_dims=stage.critic_hidden_dims,
            activation=stage.activation,
            init_noise_std=float(visual_conf.get("init_noise_std", 0.15)),
        ).to(self.device)

        self.anchor_encoder = copy.deepcopy(self.model.vision_encoder).to(self.device)
        self.anchor_actor = copy.deepcopy(self.model.actor).to(self.device)
        for parameter in self.model.vision_encoder.cnn.parameters():
            parameter.requires_grad_(False)

        actor_parameters = [
            *self.model.actor.parameters(),
            self.model.std,
        ]
        recurrent_parameters = [
            *self.model.vision_encoder.rnn.parameters(),
            *self.model.vision_encoder.rnn_output_layer.parameters(),
        ]
        self.optimizer = optim.Adam(
            [
                {
                    "params": actor_parameters,
                    "lr": float(visual_conf.get("actor_learning_rate", stage.actor_lr)),
                    "name": "actor",
                },
                {
                    "params": recurrent_parameters,
                    "lr": float(visual_conf.get("lstm_learning_rate", stage.lstm_lr)),
                    "name": "lstm",
                },
                {
                    "params": self.model.critic.parameters(),
                    "lr": float(visual_conf.get("critic_learning_rate", stage.critic_lr)),
                    "name": "critic",
                },
            ]
        )
        self.algorithm = AlgorithmVisualPPO(
            model=self.model,
            anchor_encoder=self.anchor_encoder,
            anchor_actor=self.anchor_actor,
            optimizer=self.optimizer,
            sequence_length=int(
                visual_conf.get(
                    "tbptt_sequence_length", stage.tbptt_sequence_length
                )
            ),
            # Anchor R2 schedule (§4.1/§5.3). TOML stores minutes; divide by 60
            # to hours for the algorithm. Missing warmup LR -> None (algorithm
            # falls back to critic_lr with a warning, never silently 3e-4).
            schedule_mode=str(
                visual_conf.get("schedule_mode", "visual_anchor_anneal_v2")
            ),
            run_name=str(
                visual_conf.get("run_name", "standard-anchor-r2")
            ),
            source_parent_model_id=visual_conf.get("initial_parent_model_id"),
            anchor_schedule_hours=[
                value / 60.0
                for value in visual_conf.get("anchor_schedule_minutes", [])
            ],
            action_anchor_schedule=visual_conf.get("action_anchor_schedule"),
            latent_anchor_schedule=visual_conf.get("latent_anchor_schedule"),
            anchor_phase_labels=visual_conf.get("anchor_phase_labels"),
            anchor_phase_end_hours=[
                value / 60.0
                for value in visual_conf.get("anchor_phase_end_minutes", [])
            ],
            critic_warmup_learning_rate=(
                float(visual_conf["critic_warmup_learning_rate"])
                if "critic_warmup_learning_rate" in visual_conf
                else None
            ),
            task_end_hours=float(visual_conf.get("task_end_hours", 4.0)),
            warning_only_safety=bool(
                visual_conf.get("warning_only_safety", True)
            ),
            max_anchor_action_mse=float(
                visual_conf.get("max_anchor_action_mse", 0.05)
            ),
            max_hard_termination_delta=float(
                visual_conf.get("max_hard_termination_delta", 0.02)
            ),
            # Legacy scalar schedule kwargs kept for legacy_three_phase_v1
            # resume compatibility; Anchor R2 ignores them.
            critic_only_hours=float(visual_conf.get("critic_only_hours", 0.0)),
            actor_only_end_hours=float(
                visual_conf.get("actor_only_end_hours", 0.0)
            ),
            action_anchor_start=float(
                visual_conf.get("action_anchor_start", 0.0)
            ),
            action_anchor_mid=float(visual_conf.get("action_anchor_mid", 0.0)),
            action_anchor_end=float(visual_conf.get("action_anchor_end", 0.0)),
            latent_anchor_weight=float(
                visual_conf.get("latent_anchor_weight", 0.0)
            ),
            device=self.device,
            logger=self.logger,
            monitor=self.monitor,
            clip_param=float(visual_conf.get("clip_param", 0.2)),
            gamma=float(visual_conf.get("gamma", 0.99)),
            lam=float(visual_conf.get("lam", 0.95)),
            value_loss_coef=float(visual_conf.get("value_loss_coef", 1.0)),
            entropy_coef=float(visual_conf.get("entropy_coef", 0.01)),
            learning_rate=stage.actor_lr,
            max_grad_norm=float(
                visual_conf.get("max_grad_norm", stage.max_grad_norm)
            ),
            num_mini_batches=int(
                visual_conf.get("num_mini_batches", stage.num_mini_batches)
            ),
            num_learning_epochs=int(
                visual_conf.get("num_learning_epochs", stage.num_learning_epochs)
            ),
            desired_kl=None,
        )
        # training_elapsed_h is the Agent-side mirror of the algorithm's anchor
        # session clock (§4.3). The algorithm uses anchor_session_elapsed_hours
        # to drive phase decisions; this attribute feeds learn(elapsed_h=...).
        self.training_elapsed_h = 0.0
        self.logger.info(
            f"[VisualPPO] initialized Anchor R2: run={self.algorithm.run_name}, "
            f"schedule={self.algorithm.schedule_mode}, "
            f"CNN frozen, S0 preload required before training"
        )

    def exploit(self, list_obs_data):
        """
        Exploit learned policy for action selection in evaluation mode.
        在评估模式下利用已学习的策略进行动作选择。
        """
        (obs) = list_obs_data
        with torch.no_grad():
            if self.is_lbc:
                return self._exploit_lbc_loco(obs)
            actions = self.algorithm.actor_critic.act_inference(obs)
            return [ActData(action=actions)]

    def _exploit_lbc_loco(self, obs):
        """LBC Loco eval: 学生 VisionEncoder 闭环推理。

        obs: flat tensor [B, proprio+scan+depth] = [B, 57901]
        Returns: [ActData(action=joint_actions[B, 12])]
        """
        obs_dict = self.algorithm._split_obs(obs)

        # 1. 学生: depth + proprio → LSTM → student_latent
        student_latent = self.vision_encoder(
            depth_image=obs_dict["depth_image"],
            proprio=obs_dict["proprio"],
            masks=None,
        )  # [B, latent_dim]

        # 2. 教师 Actor: proprio + student_latent → joint_actions
        actor_input = torch.cat([obs_dict["proprio"], student_latent], dim=-1)
        joint_actions = self.teacher_actor(actor_input)  # [B, num_actions]

        return [ActData(action=joint_actions)]

    def learn(self, list_sample_data=None):
        """
        Trigger learning process using sample data.
        使用样本数据触发学习过程。

        LBC 阶段：训练在 lbc_workflow 中直接调用 algorithm.update(obs)，
                  不经 agent.learn；此处 no-op 作为安全保护。
        """
        if self.is_lbc:
            return None
        if self.is_behavior_distill:
            return None
        if self.is_visual_ppo:
            return self.algorithm.learn(self.training_elapsed_h)
        return self.algorithm.learn()

    def predict(self, list_obs_data):
        """
        Generate predictions with actor-critic network.
        使用 actor-critic 网络生成预测。

        LBC 阶段：由 lbc_workflow 直接调用 algorithm.act_teacher/update，
                  此处不使用。调用时抛出明确错误。
        """
        if self.is_lbc:
            raise RuntimeError(
                "agent.predict() is not used in LBC stage; lbc_workflow calls "
                "algorithm.act_teacher/update directly."
            )
        if self.is_behavior_distill:
            raise RuntimeError(
                "agent.predict() is not used in behavior_distill stage; "
                "behavior_distill_workflow calls algorithm.act_teacher/update directly."
            )
        (obs, critic_obs) = list_obs_data
        with torch.no_grad():
            if self.is_visual_ppo:
                self._last_rollout_hidden = self.algorithm.rollout_hidden_state()
                (
                    self._last_anchor_action,
                    self._last_anchor_latent,
                ) = self.algorithm.anchor_inference(obs)
            actions = self.algorithm.actor_critic.act(obs)
            values = self.algorithm.actor_critic.evaluate(critic_obs)
            log_probs = self.algorithm.actor_critic.get_actions_log_prob(actions)
            action_mean = self.algorithm.actor_critic.action_mean.detach()
            action_std = self.algorithm.actor_critic.action_std.detach()
            return (
                actions,
                values,
                log_probs,
                action_mean,
                action_std,
                obs.detach(),
                critic_obs.detach(),
            )

    def _current_ramp_label(self) -> str:
        """根据当前 ramp 进度和 soft-stay 状态决定 checkpoint 标签。

        让文件名本身反映训练阶段，避免 0% 教师驱动阶段就出现 visionfull：
          - soft-stay 冻结且未恢复        → visionblocked
          - ramp_probability < 0.10       → visionteacher（早期教师驱动为主）
          - 0.10 <= ramp_probability < 1.0 → visionhalf（学生逐步接管）
          - ramp_probability >= 1.0       → visionfull（纯学生闭环）
        """
        algo = self.algorithm
        if getattr(algo, "soft_stay_frozen", False):
            return "visionblocked"
        p = float(getattr(algo, "ramp_probability", 0.0))
        if p >= 1.0:
            return "visionfull"
        if p >= 0.10:
            return "visionhalf"
        return "visionteacher"

    def save_model(self, path=None, id="1"):
        """
        Save model checkpoint.
        保存 model checkpoint。

        LBC（阶段 4）：只保存一个规范的 kaiwu_train_v1 视觉训练包，文件名用
        纯字母 ramp 标签（visionteacher/visionhalf/visionfull/visionblocked），
        标签由 _current_ramp_label 按 ramp 进度动态决定。不再生成 lbc_loco
        或 locomotion 副本（那些是冻结教师或部署格式，会在产物目录造成多个
        模型文件的混淆）。

        id 由平台框架注入（调用 agent.save_model() 时不传 id）；不得用 iteration
        人工计算 id，否则平台模型池 ID / 文件名 / 任务页记录会不一致。
        """
        ckpt_name = getattr(Config.CURRENT, "ckpt_name", "") or ""
        if ckpt_name:
            model_file_path = f"{path}/{ckpt_name}-{str(id)}.pkl"
        else:
            model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        if self.is_visual_ppo:
            phase_label = self.algorithm.current_phase
            visual_rl_path = f"{path}/model.ckpt-{phase_label}-{str(id)}.pkl"
            if not validate_probe_filename(visual_rl_path):
                raise ValueError(
                    "Visual PPO checkpoint filename is not probe-compatible: "
                    f"{visual_rl_path}"
                )
            checksum = self.algorithm.save_training_bundle(
                visual_rl_path,
                platform_model_id=id,
                phase_label=phase_label,
                model_spec={
                    "proprio_dim": self.stage.proprio_dim,
                    "scan_dim": self.stage.scan_dim,
                    "depth_height": self.stage.depth_height,
                    "depth_width": self.stage.depth_width,
                    "depth_channels": self.stage.depth_channels,
                    "latent_dim": self.stage.latent_dim,
                    "action_dim": self.stage.num_actions,
                    "goal_dim": 0,
                },
            )
            # §5.6: save log prints phase, both anchors, anchor session clock,
            # trainable modules, platform id and sha256. Only one bundle is
            # written; no rlfull/locomotion/lbc_loco same-id copies.
            self.logger.info(
                f"[visual_ppo] save bundle={visual_rl_path} "
                f"(phase={phase_label}, "
                f"action_anchor={self.algorithm.action_anchor_weight:.3f}, "
                f"latent_anchor={self.algorithm.latent_anchor_weight_current:.3f}, "
                f"anchor_session_h="
                f"{self.algorithm.anchor_session_elapsed_hours:.3f}, "
                f"trainable={self.algorithm.trainable_modules_snapshot()}, "
                f"platform_id={id}, sha256={checksum})"
            )
        elif self.is_lbc:
            ramp_label = self._current_ramp_label()
            vision_file_path = f"{path}/model.ckpt-{ramp_label}-{str(id)}.pkl"
            if not validate_probe_filename(vision_file_path):
                raise ValueError(
                    f"Vision checkpoint filename not probe-compatible: {vision_file_path}"
                )
            checksum = self.algorithm.save_vision_bundle(
                vision_file_path,
                platform_model_id=id,
                ramp_label=ramp_label,
            )
            self.logger.info(
                f"[{self.algorithm_name}] save vision bundle={vision_file_path} "
                f"(ramp_p={float(getattr(self.algorithm, 'ramp_probability', 0.0)):.3f}, "
                f"label={ramp_label}, sha256={checksum})"
            )
        elif self.is_behavior_distill:
            phase_file_path = (
                f"{path}/model.ckpt-"
                f"{self.algorithm.current_phase_label}-{str(id)}.pkl"
            )
            alias_file_path = f"{path}/model.ckpt-locomotion-{str(id)}.pkl"
            for candidate in (phase_file_path, alias_file_path):
                if not validate_probe_filename(candidate):
                    raise ValueError(
                        f"Checkpoint filename is not probe-compatible: {candidate}"
                    )
            checksum = self.algorithm.save(
                phase_file_path,
                platform_model_id=id,
            )
            shutil.copyfile(phase_file_path, alias_file_path)
            alias_size = os.path.getsize(alias_file_path)
            if alias_size <= 0 or os.path.getsize(phase_file_path) != alias_size:
                raise IOError(
                    "Checkpoint alias verification failed: "
                    f"phase={phase_file_path}, alias={alias_file_path}"
                )
            self.logger.info(
                f"[{self.algorithm_name}] save phase={phase_file_path}, "
                f"alias={alias_file_path}, sha256={checksum}"
            )
        else:
            torch.save(self.model.state_dict(), model_file_path)
            self.logger.info(f"save model {model_file_path} successfully")

        # Side model: 非 lbc_loco 阶段才落 locomotion 形态的 side ckpt。
        # 阶段 4 视觉训练包已含冻结教师副本（modules.low_level），不再额外生成
        # locomotion 副本，避免产物目录出现 vision* + lbc-loco + locomotion 三个
        # 文件的混淆（locomotion 只是冻结教师，不是视觉学生）。
        if not (self.is_lbc or self.is_visual_ppo):
            self._save_side_locomotion(path, id)

    def save_vision_at_ramp_label(self, path, id, ramp_label: str):
        """在 ramp 关键比例点额外保存带标签的视觉训练包。

        ramp_label 必须是纯字母视觉标签（visionteacher/visionhalf/visionfull/
        visionblocked），满足探活正则。注意：常规定时保存已通过 _current_ramp_label
        自动按 ramp 进度命名，本方法仅用于需要显式覆盖标签的场合（如任务结束时
        强制落 visionfull）。多数情况下无需调用。
        """
        if not self.is_lbc:
            return
        vision_file_path = f"{path}/model.ckpt-{ramp_label}-{str(id)}.pkl"
        if not validate_probe_filename(vision_file_path):
            raise ValueError(
                f"Ramp-label filename not probe-compatible: {vision_file_path}"
            )
        checksum = self.algorithm.save_vision_bundle(
            vision_file_path,
            platform_model_id=id,
            ramp_label=ramp_label,
        )
        self.logger.info(
            f"[{self.algorithm_name}] save ramp-label {vision_file_path} "
            f"(sha256={checksum})"
        )

    def _save_side_locomotion(self, path, id):
        """保存 lbc_loco 教师为 locomotion 形态的 side ckpt。

        仅当 teacher_encoder / teacher_actor 都已初始化（即 lbc_loco 阶段）时触发，
        输出 model.ckpt-locomotion-{id}.pkl，key 前缀重映射为 encoder.*/actor.*。

        :param path: checkpoint 保存目录
        :param id: checkpoint 编号
        """
        if not (hasattr(self, "teacher_encoder") and hasattr(self, "teacher_actor")):
            return

        loco_path = f"{path}/model.ckpt-locomotion-{str(id)}.pkl"
        loco_state = {}
        for k, v in self.teacher_encoder.state_dict().items():
            loco_state[f"encoder.{k}"] = v
        for k, v in self.teacher_actor.state_dict().items():
            loco_state[f"actor.{k}"] = v
        torch.save(loco_state, loco_path)
        self.logger.info(
            f"save side: loco teacher (encoder.*/actor.*) {loco_path} successfully"
        )

    def load_model(self, path=None, id="1"):
        """
        Load model checkpoint.
        加载模型 checkpoint。

        Locomotion (flat)  : model.ckpt-locomotion-{id}.pkl
        LBC Loco (train)   : main ckpt (model.ckpt-lbc-loco-{id}.pkl) 存在则续训；
                             否则 fallback 到 model.ckpt-locomotion-{id}.pkl 拆分教师。
        LBC Loco (eval)    : 只加载 vision_encoder + teacher_actor（模拟真机视角）。
        """
        if self.is_visual_ppo:
            self._load_visual_ppo(path, id)
            return
        if self.is_lbc:
            self._load_lbc_loco(path, id)
            return
        if self.is_behavior_distill:
            self._load_behavior_distill_teacher(path, id)
            return
        self._load_flat(path, id)

    def _load_visual_ppo(self, path=None, id="1"):
        """Load the Anchor R2 S0 parent, an Anchor R2 resume, or a Camera eval ckpt.

        Three paths (§5.4/§5.6):
          * eval (is_eval): Camera eval candidates — Anchor R2 labels first,
            then legacy R3/Stage-4 vision/S0 visionfull. Only VisionEncoder +
            Actor are used by the deploy-shaped loader; Critic/S0 ignored.
          * training first-run: exact S0 visionfull-28401 parent (no `latest`,
            no same-id historical vis* file may preempt it).
          * training resume: Anchor R2 label candidates within the requested ID.
        ID constraint enforced in checkpoint_io.py candidate layer (§5.5 N8);
        Agent only passes the selector and prints selector/filename-id/bundle-id.
        """
        if not path:
            raise FileNotFoundError("[VisualPPO] preload path is empty")
        id_str = str(id)
        expected_spec = {
            "proprio_dim": self.stage.proprio_dim,
            "scan_dim": self.stage.scan_dim,
            "latent_dim": self.stage.latent_dim,
            "action_dim": self.stage.num_actions,
            "goal_dim": 0,
        }
        env_seed = self._visual_env_seed()

        if self.is_eval:
            candidates = visual_anchor_r2_eval_candidates(path, id)
            resolved_id = self._resolve_filename_id(candidates, path, id_str)
        elif id_str == "latest":
            # `latest` only allowed for eval/diagnostic convenience; training
            # entry rejects it. Resolve max filename ID within Anchor R2 labels.
            from agent_ppo.checkpoint_io import _latest_visual_anchor_r2_candidates
            resume_candidates = _latest_visual_anchor_r2_candidates(path)
            parent_candidates = visual_anchor_r2_parent_candidates(path, id_str)
            candidates = [*resume_candidates, *parent_candidates]
            resolved_id = self._resolve_filename_id(candidates, path, id_str)
        else:
            # Training: Anchor R2 resume first, then exact S0 parent. The
            # parent must come AFTER resume so a same-id resume wins, but the
            # parent list itself puts visionfull-28401 first among S0 files.
            resume_candidates = visual_anchor_r2_checkpoint_candidates(path, id)
            parent_candidates = visual_anchor_r2_parent_candidates(path, id)
            candidates = [*resume_candidates, *parent_candidates]
            resolved_id = id_str

        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate and os.path.isfile(candidate)
            ),
            None,
        )
        if selected is None:
            raise FileNotFoundError(
                f"[VisualPPO] no Anchor R2 / S0 checkpoint found in {path} "
                f"for selector={id_str!r}; tried={candidates}"
            )
        load_mode = self.algorithm.load_training_bundle(
            selected,
            expected_spec=expected_spec,
            env_seed=env_seed,
        )
        # Agent mirrors the algorithm's anchor session clock; elapsed_training_h
        # is only a cumulative-training record, not a phase driver (§4.3).
        self.training_elapsed_h = self.algorithm.anchor_session_elapsed_hours
        self.cur_model_name = selected
        bundle_id = self.algorithm.loaded_platform_model_id or "unknown"
        self.logger.info(
            f"[VisualPPO] run={self.algorithm.run_name} "
            f"schedule={self.algorithm.schedule_mode} "
            f"requested_selector={id_str} "
            f"resolved_filename_id={resolved_id} "
            f"bundle_platform_model_id={bundle_id} "
            f"loaded_path={selected} "
            f"loaded_sha256={self.algorithm.s0_checkpoint_sha256} "
            f"load_mode={load_mode} "
            f"resume_anchor_session_h="
            f"{self.algorithm.anchor_session_elapsed_hours:.3f} "
            f"phase={self.algorithm.current_phase} "
            f"trainable={self.algorithm.trainable_modules_snapshot()} "
            f"action_anchor={self.algorithm.action_anchor_weight:.3f} "
            f"latent_anchor={self.algorithm.latent_anchor_weight_current:.3f}"
        )

    def _visual_env_seed(self):
        """Read [env_conf].seed from the active TOML for RNG reinitialization."""
        usr_conf = getattr(self, "usr_conf", None)
        if not isinstance(usr_conf, dict):
            return 0
        env_conf = usr_conf.get("env_conf", {})
        if not isinstance(env_conf, dict):
            return 0
        try:
            return int(env_conf.get("seed", 0))
        except (TypeError, ValueError):
            return 0

    def _resolve_filename_id(self, candidates, path, id_str):
        """Print selector/filename-id/bundle-id (§5.5/§5.6) and return filename id.

        For `latest` and eval the resolved filename numeric ID is parsed from
        the first existing candidate; it is independent of the bundle's inner
        platform_model_id (logged separately by the caller).
        """
        from agent_ppo.checkpoint_io import _parse_anchor_r2_filename
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                parsed = _parse_anchor_r2_filename(candidate)
                if parsed is not None:
                    return str(parsed[1])
                # Non-Anchor-R2 legacy file: extract trailing digits.
                basename = os.path.basename(candidate)
                tail = basename.rsplit("-", 1)[-1].split(".")[0]
                if tail.isdigit():
                    return tail
                return "unknown"
        return id_str

    def _load_behavior_distill_teacher(self, path=None, id="1"):
        """Load platform-selected flat standard pretrained model as frozen teacher.

        Platform pretrained models keep their original filename, e.g.
        model.ckpt-10288.pkl. The student output of this stage is saved as
        model.ckpt-locomotion-{id}.pkl, so behavior distillation must search both
        naming schemes and only accept flat ActorCritic teacher checkpoints.
        """
        if not path:
            self.logger.info("[BehaviorDistill] load_model skipped: path is empty")
            return

        id_str = str(id)
        candidates = checkpoint_candidates(path, id_str)
        # The original 10288 teacher normally uses the unlabelled filename.
        original_teacher = f"{path}/model.ckpt-{id_str}.pkl"
        candidates = [original_teacher, *[
            candidate for candidate in candidates if candidate != original_teacher
        ]]

        # Some platform calls use id='latest'. In that case scan the provided
        # model directory and choose the first flat standard teacher we can load.
        if id_str == "latest":
            candidates.extend(
                sorted(
                    glob.glob(f"{path}/model.ckpt-*.pkl"),
                    key=os.path.getmtime,
                    reverse=True,
                )
            )

        tried = []
        for model_file_path in candidates:
            if not model_file_path or model_file_path in tried:
                continue
            tried.append(model_file_path)
            if not os.path.exists(model_file_path):
                continue

            try:
                pretrained = torch.load(model_file_path, weights_only=False, map_location=self.device)
                if is_kaiwu_train_bundle(pretrained):
                    self.algorithm.load_checkpoint_dict(
                        pretrained,
                        model_file_path,
                        load_optimizer=True,
                        load_teacher=True,
                        restore_rng=True,
                    )
                    self.cur_model_name = model_file_path
                    self.logger.info(
                        f"[BehaviorDistill] resumed common training bundle "
                        f"{model_file_path}"
                    )
                    return
                if isinstance(pretrained, dict) and "format" in pretrained:
                    self.logger.info(
                        f"[BehaviorDistill] skip non-flat ckpt {model_file_path}: "
                        f"format={pretrained.get('format')}"
                    )
                    continue
                self.algorithm.load_teacher_state_dict(pretrained, source=model_file_path)
                self.cur_model_name = model_file_path
                self.logger.info(f"[BehaviorDistill] loaded flat teacher {model_file_path}")
                return
            except Exception as exc:
                self.logger.warning(
                    f"[BehaviorDistill] failed to load teacher candidate "
                    f"{model_file_path}: {exc}"
                )

        self.logger.warning(
            f"[BehaviorDistill] no flat teacher ckpt loaded from path={path}, id={id}. "
            f"tried={tried}"
        )

    def _load_flat(self, path, id):
        """Load single-model checkpoint."""
        ckpt_name = getattr(Config.CURRENT, "ckpt_name", "") or ""
        if ckpt_name:
            model_file_path = f"{path}/{ckpt_name}-{str(id)}.pkl"
        else:
            model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        candidates = checkpoint_candidates(path, id)
        if model_file_path not in candidates:
            candidates.insert(0, model_file_path)
        model_file_path = next(
            (candidate for candidate in candidates if os.path.exists(candidate)),
            model_file_path,
        )
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"No flat checkpoint found: {model_file_path}")
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
            return

        pretrained = torch.load(
            model_file_path, weights_only=False, map_location=self.device
        )
        if is_kaiwu_train_bundle(pretrained):
            validate_low_level_spec(
                pretrained,
                expected={
                    "proprio_dim": self.stage.num_proprio_obs,
                    "scan_dim": self.stage.num_scan,
                    "latent_dim": self.stage.latent_dim,
                    "action_dim": self.stage.num_actions,
                    "goal_dim": getattr(self.stage, "num_goal_obs", 0),
                },
            )
            pretrained = low_level_policy_state(pretrained)
        current_state = self.model.state_dict()

        if self._ckpt_exact_match(pretrained, current_state):
            self.model.load_state_dict(pretrained)
            self.logger.info(f"load model {model_file_path} successfully (exact match)")
        else:
            self._load_model_partial(self.model, pretrained, model_file_path)

        self.cur_model_name = model_file_path

    def _load_lbc_loco(self, path, id):
        """LBC Loco 模型加载，按 is_eval 分发训练期 / eval 两条路径。

        阶段 4 起视觉训练包统一为 kaiwu_train_v1（modules.vision_encoder）。
        老的 lbc_loco 顶层 key 格式只作为兼容回退（部署导出制品另用 lbc_loco）。

        Eval 路径（模拟真机视角）:
            优先 vision* 标签的视觉训练包，回退 lbc-loco；
            → 仅加载 vision_encoder + teacher_actor（不加载 teacher_encoder）
            → 真机没有 height_scan，teacher_encoder 永远不会被调用

        训练期路径:
            P1 (续训): vision* 视觉训练包 → load_vision_resume(...)，恢复
               vision_encoder + 冻结教师 + optimizer + ramp_state（不恢复 LSTM hidden）
            P2 (首训): daggerfull / locomotion 父包 → load_parent_bundle(...)
               显式优先 daggerfull-16288，打印路径/SHA/model_spec 供操作者核对
            miss: FileNotFoundError
        """
        is_eval = getattr(self, "is_eval", False)

        # Eval 路径：模拟真机视角
        if is_eval:
            eval_path = self._find_vision_eval_ckpt(path, id)
            self._load_lbc_loco_for_eval(eval_path)
            self.cur_model_name = eval_path
            return

        # 训练期路径 P1: 续训视觉训练包（vision* 标签优先）
        resumed = self.algorithm.load_vision_resume(path, id)
        if resumed is not None:
            self.cur_model_name = resumed
            self.logger.info(
                f"[LBC-Loco] Resumed vision bundle {resumed} "
                f"(iter={self.algorithm.current_iteration}, "
                f"ramp_p={self.algorithm.ramp_probability:.3f})"
            )
            return

        # 训练期路径 P2: 首训，加载特权父文件（daggerfull-16288）
        try:
            parent_path = self.algorithm.load_parent_bundle(path, id)
            self.cur_model_name = parent_path
            self.logger.info(
                f"[LBC-Loco] Vision parent (frozen teacher) loaded from {parent_path}; "
                f"student randomly initialized."
            )
            return
        except FileNotFoundError:
            pass

        raise FileNotFoundError(
            f"[LBC-Loco] No ckpt found in {path}/ for id={id}: "
            f"no vision* resume bundle and no daggerfull/locomotion parent."
        )

    def _find_vision_eval_ckpt(self, path, id) -> str:
        """评估时查找视觉 checkpoint：优先 vision* 视觉包，回退 lbc-loco。"""
        candidates = vision_checkpoint_candidates(path, id)
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        # 兼容：老 lbc-loco 顶层 key 格式
        legacy = f"{path}/model.ckpt-lbc-loco-{str(id)}.pkl"
        if os.path.exists(legacy):
            return legacy
        raise FileNotFoundError(
            f"[LBC-Loco eval] No vision ckpt found in {path}/ for id={id}; "
            f"eval simulates real-robot deployment and cannot fall back to "
            f"teacher-only ckpt."
        )

    def _load_lbc_loco_for_eval(self, vision_path):
        """Eval 模式：模拟真机视角，只加载 vision_encoder + teacher_actor。

        真机部署时 teacher_encoder（吃 height_scan）不存在，故 eval 也不加载它。
        支持两种格式：
          - kaiwu_train_v1 视觉包（modules.vision_encoder / modules.low_level）
          - 老 lbc_loco 顶层 key（vision_encoder_state_dict / teacher_actor_state_dict）
        """
        ckpt = torch.load(vision_path, weights_only=False, map_location=self.device)
        fmt = ckpt.get("format")

        if fmt == KAIWU_TRAIN_FORMAT:
            # 新视觉训练包：从 modules 读取
            modules = ckpt.get("modules", {})
            vision_section = modules.get("vision_encoder", {})
            ve_state = vision_section.get("state_dict")
            if not isinstance(ve_state, dict):
                raise KeyError(
                    f"modules.vision_encoder.state_dict missing in {vision_path}"
                )
            self.vision_encoder.load_state_dict(ve_state)
            self.vision_encoder.eval()
            self.vision_encoder.reset_hidden_state(
                batch_size=self.num_envs, device=self.device
            )
            low_level = modules.get("low_level", {})
            act_state = low_level.get("actor_state_dict")
            if not isinstance(act_state, dict):
                raise KeyError(
                    f"modules.low_level.actor_state_dict missing in {vision_path}"
                )
            self.teacher_actor.load_state_dict(act_state)
            self.teacher_actor.eval()
            self.logger.info(
                f"[LBC-Loco eval] Loaded vision bundle {vision_path} "
                f"(kaiwu_train_v1; teacher_encoder NOT loaded — pure vision view, "
                f"no height_scan)"
            )
            return

        # 兼容老 lbc_loco 顶层 key 格式
        if fmt != self.algorithm_name:
            raise ValueError(
                f"Ckpt format mismatch: expected '{self.algorithm_name}' or "
                f"'{KAIWU_TRAIN_FORMAT}', got '{fmt}' at {vision_path}."
            )
        if "vision_encoder_state_dict" not in ckpt:
            raise KeyError(f"vision_encoder_state_dict missing in {vision_path}")
        self.vision_encoder.load_state_dict(ckpt["vision_encoder_state_dict"])
        self.vision_encoder.eval()
        self.vision_encoder.reset_hidden_state(batch_size=self.num_envs, device=self.device)
        if "teacher_actor_state_dict" not in ckpt:
            raise KeyError(f"teacher_actor_state_dict missing in {vision_path}")
        self.teacher_actor.load_state_dict(ckpt["teacher_actor_state_dict"])
        self.teacher_actor.eval()
        self.logger.info(
            f"[LBC-Loco eval] Loaded legacy lbc_loco {vision_path} "
            f"(teacher_encoder NOT loaded — simulates real-robot view)"
        )

    @staticmethod
    def _ckpt_exact_match(pretrained: dict, current_state: dict) -> bool:
        """判断 ckpt 是否与当前模型 state_dict 完全对齐（key 集合相同 + 所有 shape 相同）。"""
        if set(pretrained.keys()) != set(current_state.keys()):
            return False
        return all(pretrained[k].shape == current_state[k].shape for k in pretrained)

    def _load_model_partial(self, model, pretrained, model_file_path):
        """
        Partial checkpoint loading for cross-stage transfer.
        部分加载 checkpoint，用于跨阶段迁移。
        """
        current_state = model.state_dict()
        loaded_keys = []
        partial_keys = []
        skipped_keys = []

        for key in current_state:
            if key not in pretrained:
                skipped_keys.append(key)
                continue

            old_param = pretrained[key]
            new_param = current_state[key]

            if old_param.shape == new_param.shape:
                new_param.copy_(old_param)
                loaded_keys.append(key)
            else:
                with torch.no_grad():
                    new_param.zero_()
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_param.shape, new_param.shape))
                    new_param[slices] = old_param[slices]
                partial_keys.append(f"{key} {list(old_param.shape)}→{list(new_param.shape)}")

        model.load_state_dict(current_state)

        self.logger.info(
            f"Partial load model {model_file_path}: "
            f"{len(loaded_keys)} exact, {len(partial_keys)} partial, {len(skipped_keys)} skipped"
        )
        for info in partial_keys:
            self.logger.info(f"  Partial: {info}")
