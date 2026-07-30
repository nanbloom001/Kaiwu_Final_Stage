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
import hashlib
import time

import numpy as np
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

import torch.optim as optim

from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.feature.definition import ActData
from agent_ppo.feature.nav_event_log import emit_nav_event
from agent_ppo.conf.conf import Config
from agent_ppo.model.actor_critic_encoder import ActorCriticEncoder
from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO
from agent_ppo.checkpoint_io import (
    CheckpointSaveError,
    KAIWU_TRAIN_FORMAT,
    checkpoint_candidates,
    classify_locomotion_eval_high_level,
    is_kaiwu_train_bundle,
    low_level_policy_state,
    low_level_only_parent_candidates,
    validate_state_dict_finite,
    validate_low_level_spec,
    validate_probe_filename,
    validate_visual_eval_bundle_identity,
    p15_response_parent_candidates,
    p2_nav_evaluation_candidates,
    p2_nav_training_candidates,
    visual_command_parent_candidates,
    visual_eval_checkpoint_diagnostics,
    visual_eval_checkpoint_candidates,
    visual_anchor_r2_training_candidates,
    visual_latest_model_id,
)
from tools.train_env_conf_validate import check_usr_conf


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device="cuda", logger=None, monitor=None):
        lifecycle_started = time.monotonic()
        self.cur_model_name = "ActorCriticEncoder"
        self.device = device
        self._process_role = str(agent_type)
        self.logger = logger
        self.monitor = monitor
        self.worker_command_enabled = False
        self.command_step_dt_s = 0.02
        self._last_command_anchor_weights = None
        self._last_command_metrics = {}
        self._zero_command_telemetry = None
        self._last_zero_command_telemetry = {}
        self._visual_eval_diagnostic_steps = 0
        # Camera tasks may be forced by the platform onto ``lbc_loco``.  Do
        # not permit that compatibility path to infer before a named visual
        # checkpoint has completed the strict load below.
        self._eval_checkpoint_path = None
        self._eval_requested_model_id = None
        self._p2_eval_checkpoint_path = None
        self._p2_eval_requested_model_id = None
        self._lifecycle_probe_exploit_logged = False
        self._lifecycle_probe_predict_logged = False
        self._lifecycle_probe_learn_logged = False
        self._lifecycle_probe_save_logged = False

        self.logger.info(
            "[LifecycleProbe] agent_init enter "
            f"pid={os.getpid()} ppid={os.getppid()} agent_type={agent_type} device={device}"
        )

        usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(self.logger)
        self.logger.info(
            "[LifecycleProbe] agent_init config_resolved "
            f"pid={os.getpid()} stage={stage.name} is_eval={is_eval} "
            f"conf={usr_conf_file}"
        )
        valid, message = check_usr_conf(usr_conf, is_eval, self.logger)
        if not valid:
            self.logger.error(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")
            raise Exception(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")

        self.is_eval = is_eval
        self.stage = stage
        # Cache usr_conf so later platform callbacks (load_model/save_model)
        # can read [env_conf].seed and stage sub-tables without re-parsing TOML.
        self.usr_conf = usr_conf
        # Keep the actual source path with the parsed configuration. The visual
        # workflow logs this at startup so a platform-generated TOML cannot be
        # confused with the repository training template.
        self.usr_conf_file = usr_conf_file
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
        self.is_p15_response = self.algorithm_name == "p15_response"
        self.is_p2_nav = self.algorithm_name in {"p2_nav_ppo", "p2_nav_eval"}
        self.is_p2_nav_eval = self.algorithm_name == "p2_nav_eval"
        self.is_visual_ppo = self.algorithm_name in {"visual_ppo", "p15_response"}
        self.is_nav_dagger = self.algorithm_name == "nav_dagger"
        self.is_nav_eval = self.algorithm_name == "nav_eval"

        if self.is_p2_nav:
            self._init_p2_nav(stage, usr_conf)
        elif self.is_lbc:
            # LBC 阶段：创建学生 + 教师；不初始化 PPO storage
            self._init_lbc_loco(num_proprio, num_scan, env_conf, stage, usr_conf)
        elif self.is_nav_dagger:
            # hier-nav DAgger：冻结低层 + HighLevelPolicy；不初始化 PPO storage
            self._init_nav_dagger(stage, usr_conf)
        elif self.is_nav_eval:
            # hier-nav eval：只组装推理模块，永不构造训练 Algorithm
            self._init_nav_eval(stage, usr_conf)
        elif self.is_visual_ppo:
            depth_size = (
                stage.depth_height * stage.depth_width * stage.depth_channels
            )
            self.num_obs = num_proprio + num_scan + depth_size
            self._init_visual_ppo(
                num_proprio,
                num_scan,
                stage,
                usr_conf,
                p15_response=self.is_p15_response,
            )
        else:
            self._init_flat(num_proprio, num_scan, stage)
            if self.is_behavior_distill:
                self._init_behavior_distill(stage, usr_conf)

        self.num_steps_per_env = stage.num_steps_per_env
        self.save_interval = stage.model_save_interval

        # LBC / behavior_distill / nav: 无 PPO storage 需要初始化
        # （nav 分支绝不调用不存在的 AlgorithmNavDagger.init_storage）
        if not (
            self.is_lbc
            or self.is_behavior_distill
            or self.is_nav_dagger
            or self.is_nav_eval
            or self.is_p2_nav
        ):
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

        self.logger.info(
            "[LifecycleProbe] agent_init before_base_agent "
            f"pid={os.getpid()} stage={stage.name} algorithm={self.algorithm_name} "
            f"is_eval={self.is_eval} model_type={type(self.model).__name__} "
            f"model_params={sum(parameter.numel() for parameter in self.model.parameters())}"
        )
        super().__init__(agent_type, device, logger, monitor)
        self.logger.info(
            "[LifecycleProbe] agent_init complete "
            f"pid={os.getpid()} stage={stage.name} algorithm={self.algorithm_name} "
            f"elapsed_s={time.monotonic() - lifecycle_started:.3f}"
        )
        emit_nav_event(
            "agent_ready",
            role=self._process_role,
            stage=stage.name,
            algorithm=self.algorithm_name,
            num_envs=self.num_envs,
        )

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

    # ------------------------------------------------------------------
    # hier-nav（nav_dagger 训练 / nav_eval 推理）
    # ------------------------------------------------------------------

    def _init_p2_nav(self, stage, usr_conf):
        """Build P2 modules without creating any low-level PPO state."""
        if self.is_p2_nav_eval:
            self._init_p2_nav_eval(stage, usr_conf)
            return
        import torch.nn as _nn

        from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
        from agent_ppo.feature.p2_response_buffer import P2ResponseAuxBuffer
        from agent_ppo.model.p2_high_level import (
            NavigationEncoder,
            NavigationSafetyHead,
            P2NavigationActor,
            P2NavigationCritic,
        )
        from agent_ppo.model.response_adapter import CommandResponseAdapter
        from agent_ppo.model.vision_encoder import VisionEncoder

        self.num_obs = stage.num_actor_observations
        self.num_critic_obs = stage.num_critic_observations
        self.low_level_encoder = VisionEncoder(
            image_shape=(stage.depth_height, stage.depth_width, stage.depth_channels),
            proprio_dim=stage.proprio_dim,
            cnn_output_dim=stage.cnn_output_dim,
            rnn_hidden_dim=stage.lstm_hidden_size,
            rnn_num_layers=stage.lstm_num_layers,
            rnn_output_dim=stage.latent_dim,
            use_lstm=True,
        ).to(self.device)
        self.low_level_actor = self._build_nav_low_level_actor(stage)
        self.navigation_encoder = NavigationEncoder().to(self.device)
        self.navigation_safety_head = NavigationSafetyHead().to(self.device)
        self.p2_actor = P2NavigationActor().to(self.device)
        self.p2_critic = P2NavigationCritic().to(self.device)
        self.response_adapter = CommandResponseAdapter().to(self.device)
        p2_conf = usr_conf.get("p2_nav_ppo", {})
        if not isinstance(p2_conf, dict):
            p2_conf = {}
        response_conf = p2_conf.get("response_adapter", {})
        if not isinstance(response_conf, dict):
            response_conf = {}
        self.response_aux_buffer = P2ResponseAuxBuffer(
            self.num_envs,
            "cpu",
            capacity_steps=int(response_conf.get("capacity_steps", 4096)),
            sequence_length=int(response_conf.get("sequence_length", 16)),
            burn_in_steps=int(response_conf.get("burn_in_steps", 8)),
        )
        self.model = _nn.ModuleDict(
            {
                "navigation_encoder": self.navigation_encoder,
                "navigation_safety_head": self.navigation_safety_head,
                "actor": self.p2_actor,
                "critic": self.p2_critic,
                "response_adapter": self.response_adapter,
            }
        )
        self.algorithm = AlgorithmP2NavPPO(
            low_level_encoder=self.low_level_encoder,
            low_level_actor=self.low_level_actor,
            navigation_encoder=self.navigation_encoder,
            safety_head=self.navigation_safety_head,
            actor=self.p2_actor,
            critic=self.p2_critic,
            response_adapter=self.response_adapter,
            response_buffer=self.response_aux_buffer,
            num_envs=self.num_envs,
            device=self.device,
            config=p2_conf,
            logger=self.logger,
            monitor=self.monitor,
        )
        self.training_elapsed_h = 0.0
        self._p2_parent_model_id = str(p2_conf.get("parent_model_id", 291713))
        self.logger.info(
            "[P2NavPPO] dedicated high-level algorithm initialized; "
            f"parent={self._p2_parent_model_id} num_envs={self.num_envs} "
            "low_level=inference_only adapter=online_auxiliary"
        )

    def _init_p2_nav_eval(self, stage, usr_conf):
        """Build only the modules required by evaluate_full inference."""
        import torch.nn as _nn

        from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
        from agent_ppo.model.p2_high_level import NavigationEncoder, P2NavigationActor
        from agent_ppo.model.response_adapter import CommandResponseAdapter
        from agent_ppo.model.vision_encoder import VisionEncoder

        self.num_obs = stage.num_actor_observations
        self.num_critic_obs = stage.num_critic_observations
        self.low_level_encoder = VisionEncoder(
            image_shape=(stage.depth_height, stage.depth_width, stage.depth_channels),
            proprio_dim=stage.proprio_dim,
            cnn_output_dim=stage.cnn_output_dim,
            rnn_hidden_dim=stage.lstm_hidden_size,
            rnn_num_layers=stage.lstm_num_layers,
            rnn_output_dim=stage.latent_dim,
            use_lstm=True,
        ).to(self.device)
        self.low_level_actor = self._build_nav_low_level_actor(stage)
        self.navigation_encoder = NavigationEncoder().to(self.device)
        self.p2_actor = P2NavigationActor().to(self.device)
        self.p2_critic = None
        self.response_adapter = CommandResponseAdapter().to(self.device)
        self.response_aux_buffer = None
        self.model = _nn.ModuleDict(
            {
                "navigation_encoder": self.navigation_encoder,
                "actor": self.p2_actor,
                "response_adapter": self.response_adapter,
            }
        )
        p2_conf = usr_conf.get("p2_nav_ppo", {})
        if not isinstance(p2_conf, dict):
            p2_conf = {}
        self.algorithm = AlgorithmP2NavPPO(
            low_level_encoder=self.low_level_encoder,
            low_level_actor=self.low_level_actor,
            navigation_encoder=self.navigation_encoder,
            safety_head=None,
            actor=self.p2_actor,
            critic=None,
            response_adapter=self.response_adapter,
            response_buffer=None,
            num_envs=self.num_envs,
            device=self.device,
            config=p2_conf,
            logger=self.logger,
            monitor=self.monitor,
            training=False,
        )
        self.training_elapsed_h = 0.0
        self._p2_parent_model_id = str(p2_conf.get("parent_model_id", 291713))
        self.logger.info(
            "[P2NavPPO] eval-only assembly initialized; modules="
            "low_level/navigation_encoder/actor/response_adapter "
            "critic=absent optimizers=absent response_buffer=absent"
        )

    def _build_nav_low_level_actor(self, stage):
        """复刻低层 Actor77 结构（与 :287-301 教师 Actor 同形，key 对齐低层包）。"""
        import torch.nn as _nn

        activation_map = {"elu": _nn.ELU, "relu": _nn.ReLU, "tanh": _nn.Tanh}
        Act = activation_map.get(
            getattr(stage, "teacher_actor_activation", "elu"), _nn.ELU
        )
        actor_input_dim = stage.proprio_dim + stage.latent_dim
        layers = []
        prev = actor_input_dim
        for hidden in list(stage.teacher_actor_hidden_dims):
            layers.append(_nn.Linear(prev, hidden))
            layers.append(Act())
            prev = hidden
        layers.append(_nn.Linear(prev, self.num_actions))
        return _nn.Sequential(*layers).to(self.device)

    def _init_nav_common(self, stage):
        """nav 训练/评估共用：三模块组装 + 冻结双保险 + 显式观测维度。"""
        common_started = time.monotonic()

        def probe(message):
            self.logger.info(
                "[LifecycleProbe] nav_init_common "
                f"pid={os.getpid()} role={self._process_role} {message}"
            )

        probe("imports begin")
        from agent_ppo.feature import nav_contract
        from agent_ppo.model.high_level_policy import HighLevelPolicy
        from agent_ppo.model.vision_encoder import VisionEncoder
        probe("imports complete")

        # 显式布局：policy goal4 与 critic goal3 不对称，不走基类通用公式
        self.num_obs = nav_contract.POLICY_OBS_DIM        # 57905
        self.num_critic_obs = nav_contract.CRITIC_OBS_DIM  # 323

        step_started = time.monotonic()
        probe("vision_construct begin")
        vision_encoder = VisionEncoder(
            image_shape=(stage.depth_height, stage.depth_width, stage.depth_channels),
            proprio_dim=stage.proprio_dim,
            cnn_output_dim=stage.cnn_output_dim,
            rnn_hidden_dim=stage.lstm_hidden_size,
            rnn_num_layers=stage.lstm_num_layers,
            rnn_output_dim=stage.latent_dim,
            use_lstm=True,
        )
        probe(
            "vision_construct complete "
            f"elapsed_s={time.monotonic() - step_started:.3f}"
        )

        step_started = time.monotonic()
        probe("vision_to_device begin")
        self.vision_encoder = vision_encoder.to(self.device)
        probe(
            "vision_to_device complete "
            f"elapsed_s={time.monotonic() - step_started:.3f}"
        )

        step_started = time.monotonic()
        probe("low_level_construct begin")
        self.low_level_actor = self._build_nav_low_level_actor(stage)
        probe(
            "low_level_construct complete "
            f"elapsed_s={time.monotonic() - step_started:.3f}"
        )

        step_started = time.monotonic()
        probe("high_level_construct begin")
        self.high_level = HighLevelPolicy(
            input_dim=stage.nav_input_dim,
            vocab_size=stage.nav_vocab_size,
            rnn_hidden_dim=stage.nav_lstm_hidden_size,
            rnn_num_layers=stage.nav_lstm_num_layers,
        ).to(self.device)
        probe(
            "high_level_construct complete "
            f"elapsed_s={time.monotonic() - step_started:.3f}"
        )

        # 冻结双保险（.eval + requires_grad=False；optimizer 侧另有参数集合断言）
        probe("freeze begin")
        for module in (self.vision_encoder, self.low_level_actor):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)
        probe("freeze complete")

        self.model = self.high_level
        self._nav_eval_checkpoint_path = None
        self._nav_eval_requested_model_id = None
        self._nav_scheduler = None
        self._nav_frame_count = 0

        self.logger.info(f"[nav] VisionEncoder(frozen):\n{self.vision_encoder}")
        self.logger.info(f"[nav] LowLevelActor(frozen):\n{self.low_level_actor}")
        self.logger.info(f"[nav] HighLevelPolicy(trainable):\n{self.high_level}")
        probe(
            "complete "
            f"elapsed_s={time.monotonic() - common_started:.3f}"
        )

    def _init_nav_dagger(self, stage, usr_conf):
        from agent_ppo.algorithm.algorithm_nav_dagger import AlgorithmNavDagger

        nav_init_started = time.monotonic()
        self.logger.info(
            "[LifecycleProbe] nav_init_dagger enter "
            f"pid={os.getpid()} role={self._process_role} device={self.device}"
        )
        self.logger.info("[LifecycleProbe] nav_init_common begin")
        self._init_nav_common(stage)
        self.logger.info(
            "[LifecycleProbe] nav_init_common complete "
            f"elapsed_s={time.monotonic() - nav_init_started:.3f}"
        )
        nav_conf = (
            usr_conf.get(stage.name, {}) if isinstance(usr_conf, dict) else {}
        )
        self._nav_low_level_parent_id = str(
            nav_conf.get("low_level_parent_model_id", 34728)
        )
        algorithm_started = time.monotonic()
        self.logger.info(
            "[LifecycleProbe] nav_algorithm_construct begin "
            f"pid={os.getpid()} role={self._process_role} "
            f"low_level_parent={self._nav_low_level_parent_id}"
        )
        self.algorithm = AlgorithmNavDagger(
            vision_encoder=self.vision_encoder,
            low_level_actor=self.low_level_actor,
            high_level=self.high_level,
            device=self.device,
            learning_rate=stage.lr,
            max_grad_norm=stage.max_grad_norm,
            proprio_dim=stage.proprio_dim,
            scan_dim=stage.scan_dim,
            depth_shape=(stage.depth_height, stage.depth_width, stage.depth_channels),
            low_level_parent_model_id=self._nav_low_level_parent_id,
            logger=self.logger,
            process_role=self._process_role,
        )
        self.logger.info(
            "[LifecycleProbe] nav_algorithm_construct complete "
            f"pid={os.getpid()} role={self._process_role} "
            f"elapsed_s={time.monotonic() - algorithm_started:.3f}"
        )
        self.logger.info(
            "[nav] AlgorithmNavDagger ready "
            f"(low_level_parent={self._nav_low_level_parent_id}, "
            "optimizer=high_level params only)"
        )

    def _init_nav_eval(self, stage, usr_conf):
        """eval 专用装配：无 Algorithm、无 optimizer、无训练 schedule。

        与训练类分离是历史事故的堵点：显式 policy_entry 命中训练 Algorithm
        曾因缺训练 schedule 在 eval 崩溃。
        """
        self._init_nav_common(stage)
        nav_conf = (
            usr_conf.get("nav_dagger", {}) if isinstance(usr_conf, dict) else {}
        )
        self._nav_low_level_parent_id = str(
            nav_conf.get("low_level_parent_model_id", 34728)
        )
        self.algorithm = None
        self.logger.info("[nav] eval-only assembly (no training Algorithm constructed)")

    def _init_visual_ppo(
        self, num_proprio, num_scan, stage, usr_conf, *, p15_response=False
    ):
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
        algorithm_class = AlgorithmVisualPPO
        algorithm_extra = {}
        if p15_response:
            from agent_ppo.algorithm.algorithm_p15_response import AlgorithmP15Response
            from agent_ppo.feature.response_aux_buffer import ResponseAuxBuffer
            from agent_ppo.model.response_adapter import CommandResponseAdapter

            self.response_adapter = CommandResponseAdapter().to(self.device)
            response_conf = visual_conf.get("response_adapter", {})
            if not isinstance(response_conf, dict):
                response_conf = {}
            self.response_optimizer = optim.Adam(
                self.response_adapter.parameters(),
                lr=float(response_conf.get("learning_rate", 3.0e-4)),
            )
            self.response_scheduler = optim.lr_scheduler.LambdaLR(
                self.response_optimizer, lr_lambda=lambda _step: 1.0
            )
            self.low_level_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda _step: 1.0
            )
            self.response_aux_buffer = ResponseAuxBuffer(
                self.num_envs,
                self.device,
                capacity_steps=int(response_conf.get("capacity_steps", 4096)),
                sequence_length=int(response_conf.get("sequence_length", 16)),
                burn_in_steps=int(response_conf.get("burn_in_steps", 8)),
            )
            algorithm_class = AlgorithmP15Response
            algorithm_extra = {
                "response_adapter": self.response_adapter,
                "response_optimizer": self.response_optimizer,
                "response_scheduler": self.response_scheduler,
                "low_level_scheduler": self.low_level_scheduler,
                "response_buffer": self.response_aux_buffer,
                "response_config": response_conf,
                "p15_config": visual_conf,
            }

        self.algorithm = algorithm_class(
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
            source_parent_model_id=visual_conf.get(
                "transition_parent_model_id",
                visual_conf.get("initial_parent_model_id"),
            ),
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
            max_action_amplitude=float(
                visual_conf.get("max_action_amplitude", 6.0)
            ),
            command_anchor_action=float(
                visual_conf.get("command_anchor_action", 0.35)
            ),
            command_anchor_latent=float(
                visual_conf.get("command_anchor_latent", 0.10)
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
            **algorithm_extra,
        )
        command_conf = visual_conf.get("command_schedule", {})
        if not isinstance(command_conf, dict):
            command_conf = {}
        self.command_step_dt_s = float(command_conf.get("step_dt_s", 0.02))
        commands = usr_conf.get("commands", {})
        self.worker_command_enabled = bool(
            str(visual_conf.get("schedule_mode", ""))
            in {"visual_command_generalization_v1", "p15_response_adapter_v1"}
            and commands.get("worker_progressive", {}).get("enabled", False)
            and not self.is_eval
        )
        self._last_command_anchor_weights = torch.ones(
            self.num_envs, 1, device=self.device
        )
        # training_elapsed_h is the Agent-side mirror of the algorithm's anchor
        # session clock (§4.3). The algorithm uses anchor_session_elapsed_hours
        # to drive phase decisions; this attribute feeds learn(elapsed_h=...).
        self.training_elapsed_h = 0.0
        self.logger.info(
            f"[VisualPPO] initialized visual schedule: run={self.algorithm.run_name}, "
            f"schedule={self.algorithm.schedule_mode}, "
            f"CNN frozen, S0 preload required before training"
        )
        depth_conf = usr_conf.get("camera", {}).get("depth_camera", {})
        self.logger.info(
            "[VisualPPO] configured environment contract: "
            "command_sampler="
            f"{'worker_observation_bridge' if self.worker_command_enabled else 'native'}, "
            f"resampling_time={commands.get('resampling_time')}, "
            f"ranges={commands.get('ranges')}, "
            f"buckets_enabled={commands.get('buckets', {}).get('enabled')}, "
            "worker_progressive_enabled="
            f"{commands.get('worker_progressive', {}).get('enabled')}, "
            "depth_augmentation_enabled="
            f"{depth_conf.get('augmentation', {}).get('enabled')}, "
            "command_runtime_owner=environment_worker"
        )

    def split_p15_transport(self, critic_wire):
        if not self.is_p15_response:
            return critic_wire, None
        from agent_ppo.feature.response_aux_buffer import split_privileged_transport

        return split_privileged_transport(critic_wire)

    def observe_response_aux(self, aux, dones):
        if self.is_p15_response and aux is not None:
            self.algorithm.observe_response_aux(aux, dones)

    def exploit(self, list_obs_data):
        """
        Exploit learned policy for action selection in evaluation mode.
        在评估模式下利用已学习的策略进行动作选择。
        """
        obs = list_obs_data
        if not self._lifecycle_probe_exploit_logged:
            self._lifecycle_probe_exploit_logged = True
            self.logger.info(
                "[LifecycleProbe] exploit first_call "
                f"pid={os.getpid()} stage={self.stage.name} is_eval={self.is_eval} "
                f"obs_shape={getattr(obs, 'shape', None)}"
            )
        with torch.no_grad():
            if self.is_p2_nav:
                self._ensure_p2_eval_checkpoint_loaded()
                obs, critic_wire = self._p2_eval_inputs(list_obs_data)
                result, _, _ = self.algorithm.frame_begin(
                    obs, critic_wire, deterministic=True
                )
                self.algorithm.eval_frame_advance()
                return [ActData(action=result["actions"])]
            if self.is_lbc:
                return self._exploit_lbc_loco(obs)
            if self.is_nav_dagger or self.is_nav_eval:
                return self._exploit_nav(obs)
            actions = self.algorithm.actor_critic.act_inference(obs)
            if self.is_visual_ppo:
                self._log_visual_eval_runtime_diagnostics(obs, actions)
            return [ActData(action=actions)]

    def _p2_eval_inputs(self, eval_input):
        """Normalize platform single-stream or local dual-stream P2 eval input."""
        from agent_ppo.feature import nav_contract, p2_contract

        if isinstance(eval_input, (tuple, list)) and len(eval_input) == 2:
            obs, critic_wire = eval_input
            obs = torch.as_tensor(obs, device=self.device)
            critic_wire = torch.as_tensor(critic_wire, device=self.device)
            return obs, critic_wire

        if isinstance(eval_input, (tuple, list)) and len(eval_input) == 1:
            eval_input = eval_input[0]
        obs = torch.as_tensor(eval_input, device=self.device)
        if obs.ndim != 2 or obs.shape[1] != nav_contract.POLICY_OBS_DIM:
            raise ValueError(
                f"P2 eval policy obs must be [N,{nav_contract.POLICY_OBS_DIM}], "
                f"got {tuple(obs.shape)}"
            )
        aux = p2_contract.unpack_eval_response_aux(obs)
        critic_wire = torch.zeros(
            obs.shape[0],
            p2_contract.PRIVILEGED_WIRE_DIM,
            device=obs.device,
            dtype=obs.dtype,
        )
        critic_wire[
            :,
            p2_contract.CRITIC_OBS_DIM :
            p2_contract.CRITIC_OBS_DIM + p2_contract.RESPONSE_AUX_DIM,
        ] = aux
        return obs, critic_wire

    def _ensure_p2_eval_checkpoint_loaded(self) -> None:
        if not self.is_p2_nav_eval or self._p2_eval_checkpoint_path is not None:
            return
        from common_python.config.config_control import CONFIG

        model_dir = getattr(CONFIG, "eval_model_dir", None)
        model_id = getattr(CONFIG, "eval_model_id", None)
        if not model_dir or model_id in (None, ""):
            import toml

            configure_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "conf", "configure_app.toml")
            )
            app_conf = toml.load(configure_path).get("app", {})
            model_dir = model_dir or app_conf.get("eval_model_dir")
            model_id = model_id if model_id not in (None, "") else app_conf.get("eval_model_id")
        if not model_dir or model_id in (None, ""):
            raise RuntimeError(
                "P2 eval checkpoint location is unavailable; refusing random inference"
            )
        self._load_p2_nav(str(model_dir), str(model_id))
        if self._p2_eval_checkpoint_path is None:
            raise RuntimeError(
                "P2 eval checkpoint was not loaded; refusing random inference"
            )

    def _log_visual_eval_runtime_diagnostics(self, obs, actions) -> None:
        """Print bounded input/output evidence for VisualPPO evaluation.

        This is intentionally diagnostic-only.  Missing, non-finite, or
        unusual values are reported for the operator to investigate; they do
        not turn into a policy or checkpoint loading gate.
        """
        if not self.is_eval or self._visual_eval_diagnostic_steps >= 3:
            return
        self._visual_eval_diagnostic_steps += 1
        if not isinstance(obs, torch.Tensor) or not isinstance(actions, torch.Tensor):
            self.logger.warning(
                "[VisualPPO eval] runtime diagnostics unavailable: "
                "observation or action is not a tensor"
            )
            return

        def _stats(values):
            if values.numel() == 0:
                return "unavailable"
            finite = torch.isfinite(values)
            finite_rate = float(finite.float().mean().item())
            if not bool(finite.any()):
                return f"shape={tuple(values.shape)}, finite_rate={finite_rate:.4f}"
            valid = values[finite]
            return (
                f"shape={tuple(values.shape)}, finite_rate={finite_rate:.4f}, "
                f"mean={float(valid.mean().item()):.5f}, "
                f"std={float(valid.std(unbiased=False).item()):.5f}, "
                f"min={float(valid.min().item()):.5f}, "
                f"max={float(valid.max().item()):.5f}"
            )

        command = obs[:, 6:9] if obs.ndim == 2 and obs.shape[1] >= 9 else obs.new_empty(0)
        depth_start = self.stage.num_proprio_obs + self.stage.num_scan
        depth_size = (
            self.stage.depth_height
            * self.stage.depth_width
            * self.stage.depth_channels
        )
        depth_end = depth_start + depth_size
        depth = (
            obs[:, depth_start:depth_end]
            if obs.ndim == 2 and obs.shape[1] >= depth_end
            else obs.new_empty(0)
        )
        self.logger.info(
            "[VisualPPO eval] runtime diagnostics: "
            f"step={self._visual_eval_diagnostic_steps}, "
            f"depth_expected_shape=(B,{self.stage.depth_height},"
            f"{self.stage.depth_width},{self.stage.depth_channels}), "
            f"depth={_stats(depth)}, command={_stats(command)}, "
            f"action={_stats(actions)}"
        )

    def _exploit_lbc_loco(self, obs):
        """LBC Loco eval: 学生 VisionEncoder 闭环推理。

        obs: flat tensor [B, proprio+scan+depth] = [B, 57901]
        Returns: [ActData(action=joint_actions[B, 12])]
        """
        if self.is_eval and self._eval_checkpoint_path is None:
            raise RuntimeError(
                "[LBC-Loco eval] refusing inference because no checkpoint was "
                "successfully loaded"
            )

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
        if (self.is_nav_dagger or self.is_nav_eval) and not self._lifecycle_probe_learn_logged:
            self._lifecycle_probe_learn_logged = True
            self.logger.info(
                "[LifecycleProbe] nav_agent_learn first_call "
                f"pid={os.getpid()} stage={self.stage.name} "
                f"sample_data_is_none={list_sample_data is None}"
            )
        if self.is_lbc:
            return None
        if self.is_behavior_distill:
            return None
        if self.is_p2_nav:
            return None
        if self.is_nav_dagger or self.is_nav_eval:
            # nav 训练在 nav_dagger_workflow 内直接调 algorithm；此处 no-op
            # 仅用于推进平台 lifecycle（每个成功低层批量帧恰调用一次）。
            return None
        if self.is_visual_ppo:
            return self.algorithm.learn(self.training_elapsed_h)
        return self.algorithm.learn()

    def prepare_rollout_step(self, env, obs, critic_obs, reset_mask=None):
        """Derive S0 anchor weights from worker-published policy commands."""
        del env, reset_mask
        if obs.ndim != 2 or obs.shape[1] < 9:
            raise ValueError("visual policy observation has no command fields at [6:9]")
        command = obs[:, 6:9]
        if self.is_p15_response:
            from agent_ppo.feature.p15_contract import p15_anchor_weights_from_commands

            weights = p15_anchor_weights_from_commands(command)
        else:
            from agent_ppo.feature.command_schedule import anchor_weights_from_commands

            weights = anchor_weights_from_commands(command)
        metrics = {
            "command_runtime_owner": "worker_observation_bridge_v1",
            "anchor_weight_mean": float(weights.mean().item()),
            "command_observation_min": command.min(0).values.detach().cpu().tolist(),
            "command_observation_max": command.max(0).values.detach().cpu().tolist(),
            "command_observation_mean": command.mean(0).detach().cpu().tolist(),
        }
        self._last_command_anchor_weights = weights
        self._last_command_metrics = metrics
        return obs, critic_obs, weights, metrics

    def predict(self, list_obs_data):
        """
        Generate predictions with actor-critic network.
        使用 actor-critic 网络生成预测。

        LBC 阶段：由 lbc_workflow 直接调用 algorithm.act_teacher/update，
                  此处不使用。调用时抛出明确错误。
        """
        if not self._lifecycle_probe_predict_logged:
            self._lifecycle_probe_predict_logged = True
            self.logger.info(
                "[LifecycleProbe] predict first_call "
                f"pid={os.getpid()} stage={self.stage.name} algorithm={self.algorithm_name}"
            )
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
        if self.is_p2_nav:
            raise RuntimeError(
                "agent.predict() is not used in P2; p2_nav_ppo_workflow owns "
                "semi-MDP collection and recurrent PPO updates."
            )
        if self.is_nav_dagger or self.is_nav_eval:
            raise RuntimeError(
                "agent.predict() is not used in nav stages; nav_dagger_workflow "
                "calls algorithm.frame_begin/frame_end directly (eval uses exploit)."
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

    def _current_nav_label(self) -> str:
        """按 ramp 进度返回 nav 阶段标签（navbc/navdagger/navfull，纯小写）。"""
        p = float(getattr(self.algorithm, "ramp_probability", 0.0))
        if p >= 1.0:
            return "navfull"
        if p >= 0.10:
            return "navdagger"
        return "navbc"

    def _exploit_nav(self, obs):
        """nav 评估推理：手工前向链，按冻结逐帧时序（nav_contract）。

        帧序：inject 当帧 exec → 低层前向一次 → nav tick 时高层决策 →
        step_exec（新目标下一帧生效）。未加载 checkpoint 时拒绝推理评分。
        """
        from agent_ppo.feature import nav_contract
        from agent_ppo.feature.nav_scheduler import NavScheduler

        if self._nav_eval_checkpoint_path is None:
            raise RuntimeError(
                "[nav eval] checkpoint not loaded; refusing to run inference "
                "and produce scores with random parameters"
            )
        obs = obs.to(self.device)
        if obs.ndim != 2 or obs.shape[1] != nav_contract.POLICY_OBS_DIM:
            raise ValueError(
                f"[nav eval] obs dim {tuple(obs.shape)} != "
                f"[N, {nav_contract.POLICY_OBS_DIM}]"
            )
        n = obs.shape[0]
        if self._nav_scheduler is None or self._nav_scheduler.num_envs != n:
            self._nav_scheduler = NavScheduler(n, self.device)
            self.vision_encoder.reset_hidden_state(n, self.device)
            self.high_level.reset_hidden_state(n, self.device)
            self._nav_frame_count = 0
        sched = self._nav_scheduler

        obs = obs.clone()
        sched.inject(obs)

        proprio = obs[:, : nav_contract.POLICY_PROPRIO_DIM]
        goal4 = obs[:, nav_contract.GOAL4_OBS_START : nav_contract.GOAL4_OBS_END]
        depth = obs[:, nav_contract.DEPTH_OBS_START :].reshape(
            n,
            self.stage.depth_height,
            self.stage.depth_width,
            self.stage.depth_channels,
        )
        latent = self.vision_encoder(depth, proprio, masks=None)
        actions = self.low_level_actor(torch.cat((proprio, latent), dim=-1))

        if self._nav_frame_count % nav_contract.NAV_PERIOD_FRAMES == 0:
            cnn_feat_raw = self.vision_encoder.cnn(depth)
            nav_inputs = torch.cat(
                (
                    cnn_feat_raw,
                    goal4,
                    sched.exec_cmd,
                    sched.held_cmd,
                    proprio[:, 0:3],
                    proprio[:, 3:6],
                ),
                dim=-1,
            )
            logits = self.high_level(nav_inputs, dwell_mask=sched.dwell_mask())
            tokens = logits.argmax(dim=-1)
            finite = torch.isfinite(logits).all(dim=-1)
            if not bool(finite.all()):
                tokens = torch.where(
                    finite,
                    tokens,
                    torch.full_like(tokens, nav_contract.ZERO_TOKEN_INDEX),
                )
            sched.request_tokens(tokens)
        self._nav_frame_count += 1
        sched.step_exec()
        return [ActData(action=actions)]

    @staticmethod
    def _nav_checkpoint_path_category(path) -> str:
        path_text = str(path or "")
        if "/data/user_ckpt_dir" in path_text:
            return "final_platform_archive_candidate"
        if "/data/ckpt" in path_text:
            return "running_checkpoint"
        if path in (None, ""):
            return "unknown_no_platform_path"
        return "other"

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
        if (self.is_nav_dagger or self.is_nav_eval) and not self._lifecycle_probe_save_logged:
            self._lifecycle_probe_save_logged = True
            self.logger.info(
                "[LifecycleProbe] nav_save_model first_call "
                f"pid={os.getpid()} requested_id={id} path={path} "
                f"parent_loaded={getattr(self.algorithm, 'low_level_state_digest', None) is not None}"
            )
        # Local-wrapper lifecycle saves ID 0 before invoking preload_model_file().
        # A nav bundle is invalid until its frozen low-level parent has supplied
        # the lineage digest, so acknowledge only this bootstrap callback.
        if (
            self.is_nav_dagger
            and str(id) == "0"
            and not getattr(self.algorithm, "low_level_state_digest", None)
        ):
            self.logger.warning(
                "[nav_dagger] skip framework bootstrap save id=0 before parent preload; "
                "no checkpoint was written"
            )
            return
        ckpt_name = getattr(Config.CURRENT, "ckpt_name", "") or ""
        if ckpt_name:
            model_file_path = f"{path}/{ckpt_name}-{str(id)}.pkl"
        else:
            model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        if self.is_p2_nav:
            if self.is_p2_nav_eval:
                self.logger.info("[p2_nav_eval] save_model is a no-op")
                return
            if str(id) == "0" and self.algorithm.low_level_state_digest is None:
                self.logger.warning(
                    "[P2NavPPO] skip framework bootstrap save id=0 before parent preload"
                )
                return
            phase_label = self.algorithm.current_phase
            p2_path = f"{path}/model.ckpt-{phase_label}-{str(id)}.pkl"
            if not validate_probe_filename(p2_path):
                raise ValueError(f"P2 checkpoint filename not probe-compatible: {p2_path}")
            checksum = self.algorithm.save_training_bundle(
                p2_path, platform_model_id=id
            )
            file_size = os.path.getsize(p2_path)
            self.logger.info(
                f"[P2NavPPO] save bundle={p2_path} phase={phase_label} "
                f"platform_id={id} size_bytes={file_size} sha256={checksum} "
                "deployable=false"
            )
        elif self.is_visual_ppo:
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
        elif self.is_nav_dagger:
            nav_file_path = None
            try:
                nav_label = self._current_nav_label()
                nav_file_path = f"{path}/model.ckpt-{nav_label}-{str(id)}.pkl"
                if not validate_probe_filename(nav_file_path):
                    raise ValueError(
                        "Nav checkpoint filename not probe-compatible: "
                        f"{nav_file_path}"
                    )
                checksum = self.algorithm.save_nav_bundle(
                    nav_file_path,
                    platform_model_id=id,
                    phase_label=nav_label,
                )
                file_size = os.path.getsize(nav_file_path)
                if file_size <= 0:
                    raise OSError(f"checkpoint is empty: {nav_file_path}")
            except CheckpointSaveError:
                raise
            except Exception as exc:
                raise CheckpointSaveError(
                    "Nav checkpoint serialization/write/verification failed: "
                    f"path={nav_file_path or '<unresolved>'}, platform_id={id}"
                ) from exc
            saved_at = time.monotonic()
            previous_save_at = getattr(self, "_nav_last_save_wall_time", None)
            wall_since_previous = (
                float("nan")
                if previous_save_at is None
                else saved_at - float(previous_save_at)
            )
            lifecycle_callbacks = int(
                getattr(
                    self,
                    "_nav_lifecycle_attempt_callbacks",
                    getattr(self, "_nav_lifecycle_success_callbacks", 0),
                )
            )
            previous_lifecycle_callbacks = getattr(
                self, "_nav_last_save_lifecycle_callbacks", None
            )
            lifecycle_delta = (
                -1
                if previous_lifecycle_callbacks is None
                else lifecycle_callbacks - int(previous_lifecycle_callbacks)
            )
            self._nav_last_save_wall_time = saved_at
            self._nav_last_save_lifecycle_callbacks = lifecycle_callbacks
            path_category = self._nav_checkpoint_path_category(path)
            self._nav_last_save_path = nav_file_path
            self._nav_last_save_path_category = path_category
            self._nav_last_save_platform_id = str(id)
            self.logger.info(
                f"[nav_dagger] save nav bundle={nav_file_path} "
                f"(ramp_p={float(getattr(self.algorithm, 'ramp_probability', 0.0)):.3f}, "
                f"label={nav_label}, "
                f"low_level_digest={self.algorithm.low_level_state_digest}, "
                f"platform_id={id}, path_category={path_category}, "
                f"size_bytes={file_size}, sha256={checksum}, "
                f"lifecycle_callbacks={lifecycle_callbacks}, "
                f"lifecycle_delta_since_prev_save={lifecycle_delta}, "
                f"wall_s_since_prev_save={wall_since_previous:.2f})"
            )
            emit_nav_event(
                "checkpoint_saved",
                role=self._process_role,
                platform_model_id=str(id),
                path=nav_file_path,
                path_category=path_category,
                size_bytes=file_size,
                sha256=checksum,
                lifecycle_callbacks=lifecycle_callbacks,
                lifecycle_delta_since_prev_save=lifecycle_delta,
                wall_s_since_prev_save=wall_since_previous,
            )
            emit_nav_event(
                "checkpoint_saved",
                role=self._process_role,
                platform_model_id=str(id),
                path=nav_file_path,
                sha256=checksum,
            )
        elif self.is_nav_eval:
            self.logger.info("[nav_eval] save_model is a no-op in eval assembly")
            return
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
        if not (
            self.is_lbc
            or self.is_visual_ppo
            or self.is_nav_dagger
            or self.is_nav_eval
            or self.is_p2_nav
        ):
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
        self.logger.info(
            "[LifecycleProbe] load_model enter "
            f"pid={os.getpid()} stage={self.stage.name} algorithm={self.algorithm_name} "
            f"is_eval={self.is_eval} requested_id={id} path={path}"
        )
        if self.is_nav_dagger or self.is_nav_eval:
            emit_nav_event(
                "preload_selected",
                role=self._process_role,
                requested_id=str(id),
                path=path,
            )
        if self.is_nav_dagger or self.is_nav_eval:
            same_id_files = []
            if path:
                candidates = sorted(
                    set(
                        glob.glob(os.path.join(path, f"model.ckpt-*-{id}.*"))
                        + glob.glob(os.path.join(path, f"model.ckpt-{id}.*"))
                    )
                )
                for candidate in candidates:
                    try:
                        if os.path.isfile(candidate):
                            same_id_files.append(
                                {
                                    "name": os.path.basename(candidate),
                                    "size": os.path.getsize(candidate),
                                }
                            )
                    except OSError as exc:
                        self.logger.warning(
                            "[LifecycleProbe] nav_load_model inventory_entry_failed "
                            f"path={candidate} error={type(exc).__name__}:{exc}"
                        )
            self.logger.info(
                "[LifecycleProbe] nav_load_model inventory "
                f"pid={os.getpid()} requested_id={id} same_id_files={same_id_files}"
            )
        if self.is_p2_nav:
            self._load_p2_nav(path, id)
        elif self.is_visual_ppo:
            self._load_visual_ppo(path, id)
        elif self.is_nav_dagger or self.is_nav_eval:
            self._load_nav(path, id)
        elif self.is_lbc:
            self._load_lbc_loco(path, id)
        elif self.is_behavior_distill:
            self._load_behavior_distill_teacher(path, id)
        else:
            self._load_flat(path, id)
        self.logger.info(
            "[LifecycleProbe] load_model complete "
            f"pid={os.getpid()} stage={self.stage.name} requested_id={id} "
            f"selected={self.cur_model_name}"
        )
        if self.is_nav_dagger or self.is_nav_eval:
            emit_nav_event(
                "preload_complete",
                role=self._process_role,
                requested_id=str(id),
                selected=self.cur_model_name,
            )

    def _load_p2_nav(self, path=None, id="1"):
        if not path:
            raise FileNotFoundError("[P2NavPPO] preload path is empty")
        requested = str(id)
        if self.is_p2_nav_eval:
            candidates = p2_nav_evaluation_candidates(path, requested)
        else:
            candidates = p2_nav_training_candidates(
                path,
                requested,
                parent_model_id=self._p2_parent_model_id,
            )
        # Model ID/lineage differences are warning-only, but a selected file's
        # serialization and module contract are not. Candidate fallback applies
        # only when a higher-priority file is absent; never hide a corrupt exact
        # resume by silently loading another model.
        selected = next(
            (candidate for candidate in candidates if os.path.isfile(candidate)), None
        )
        if selected is None:
            raise FileNotFoundError(
                f"[P2NavPPO] no P2 resume or configured P1.5 parent checkpoint "
                f"for requested_id={requested}; "
                f"tried={candidates}"
            )
        selected_id = os.path.basename(selected).rsplit("-", 1)[-1].split(".", 1)[0]
        if requested != "latest" and selected_id != requested and self.logger:
            self.logger.warning(
                "[P2NavPPO] requested model ID did not match the selected file; "
                "continuing with structural validation instead of blocking. "
                f"requested={requested} selected_id={selected_id} selected={selected}"
            )
        if self.is_p2_nav_eval:
            load_mode = self.algorithm.load_evaluation_bundle(
                selected, platform_model_id=requested
            )
        else:
            load_mode = self.algorithm.load_bundle(
                selected, platform_model_id=requested
            )
        self.training_elapsed_h = self.algorithm.effective_training_seconds / 3600.0
        self.cur_model_name = selected
        if self.is_p2_nav_eval:
            self._p2_eval_checkpoint_path = selected
            self._p2_eval_requested_model_id = requested
        self.logger.info(
            f"[P2NavPPO] load complete mode={load_mode} requested_id={requested} "
            f"selected={selected} effective_h={self.training_elapsed_h:.3f}"
        )

    def _load_nav(self, path=None, id="1"):
        """nav 加载分派：eval 走硬停止链；训练走首载低层父 / nav resume。

        四条分支（计划 §8.2）：
          1. 首载低层父：注入 id == low_level_parent_model_id → 只认低层父候选，
             高层保持随机初始化，当场计算并缓存 low_level_state_digest。
          2. 首次保存：save_nav_bundle 内重算 digest 硬比对（algorithm 侧）。
          3. 同 schedule resume：全量恢复权重/optimizer/ramp；per-env 活状态
             不恢复（全 reset）。
          4. 换低层重训：新任务把 low_level_parent_model_id 配成新低层 ID，
             回到分支 1（高层从头重训，不做跨包权重嫁接）。
        """
        if not path:
            raise FileNotFoundError("[nav] preload path is empty")
        if self.is_eval or self.is_nav_eval:
            self._load_nav_for_eval(path, id)
            return

        id_str = str(id)
        if id_str == "latest":
            raise ValueError(
                "[nav] training resume must use an explicit numeric model id, "
                "not 'latest' (cross-ID accidents forbidden)"
            )
        if id_str == str(self._nav_low_level_parent_id):
            hit = self.algorithm.load_parent_bundle(path, id_str)
        else:
            # 非父 ID 只允许 nav resume；resume 未命中即硬失败——不得静默
            # 回退到同 ID visual/command 父包（那会在配置未改的情况下从
            # 任意低层包 bootstrap、丢弃高层进度并写出自相矛盾的 lineage；
            # 换低层重训必须显式改 low_level_parent_model_id 回首载分支）。
            hit = self.algorithm.load_nav_resume(path, id_str)
        if hit is None:
            from agent_ppo.checkpoint_io import nav_eval_checkpoint_diagnostics

            raise FileNotFoundError(
                f"[nav] no loadable checkpoint for id={id_str} under {path} "
                f"(low_level_parent_model_id={self._nav_low_level_parent_id}); "
                f"diagnostics={nav_eval_checkpoint_diagnostics(path, id_str)}"
            )
        self.cur_model_name = hit

    def _load_nav_for_eval(self, path, id):
        """nav eval 硬停止链 + 8 行自检日志；任一失败在第一次推理前 raise。"""
        import hashlib as _hashlib

        from agent_ppo.checkpoint_io import (
            high_level_parts,
            nav_eval_checkpoint_candidates,
            nav_eval_checkpoint_diagnostics,
            nav_latest_model_id,
            validate_nav_eval_bundle,
        )
        from agent_ppo.feature import nav_contract

        requested = str(id)
        id_str = requested
        if id_str == "latest":
            resolved = nav_latest_model_id(path)
            if resolved is None:
                raise FileNotFoundError(
                    "[nav-eval] no nav checkpoints for 'latest'; "
                    f"diagnostics={nav_eval_checkpoint_diagnostics(path, id)}"
                )
            id_str = str(resolved)

        candidates = nav_eval_checkpoint_candidates(path, id_str)
        if not candidates:
            raise FileNotFoundError(
                f"[nav-eval] no same-ID nav checkpoint for id={id_str}; "
                "nav eval never falls back to loco-only bundles; "
                f"diagnostics={nav_eval_checkpoint_diagnostics(path, id_str)}"
            )
        ckpt_path = candidates[0]
        bundle = torch.load(ckpt_path, weights_only=False, map_location=self.device)

        expected_low = {
            "proprio_dim": self.stage.proprio_dim,
            "scan_dim": self.stage.scan_dim,
            "latent_dim": self.stage.latent_dim,
            "action_dim": self.stage.num_actions,
            "goal_dim": 0,  # 低层 spec 恒为 0（nav goal4 只在 nav_goal_dim 上）
        }
        expected_high = nav_contract.high_level_checkpoint_contract()
        info = validate_nav_eval_bundle(
            bundle, id_str, expected_low, expected_high, logger=self.logger
        )

        modules = bundle["modules"]
        ve_state = modules["vision_encoder"]["state_dict"]
        actor_state = modules["low_level"]["actor_state_dict"]
        hl_state, _hl_meta = high_level_parts(bundle)
        # strict=True：任何 missing/unexpected keys 视同反序列化失败
        self.vision_encoder.load_state_dict(ve_state, strict=True)
        self.low_level_actor.load_state_dict(actor_state, strict=True)
        self.high_level.load_state_dict(hl_state, strict=True)
        self.vision_encoder.eval()
        self.low_level_actor.eval()
        self.high_level.eval()
        self.vision_encoder.reset_hidden_state(self.num_envs, self.device)
        self.high_level.reset_hidden_state(self.num_envs, self.device)
        self._nav_scheduler = None
        self._nav_frame_count = 0

        hasher = _hashlib.sha256()
        with open(ckpt_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        file_sha256 = hasher.hexdigest()

        self._nav_eval_checkpoint_path = ckpt_path
        self._nav_eval_requested_model_id = requested
        self.cur_model_name = ckpt_path

        lineage = info["lineage"] or {}
        self.logger.info(f"[nav-eval] requested_model_id={requested}")
        self.logger.info(f"[nav-eval] selected_ckpt={ckpt_path}")
        self.logger.info(f"[nav-eval] file_sha256={file_sha256}")
        self.logger.info(
            f"[nav-eval] platform_model_id={bundle.get('platform_model_id')}"
        )
        self.logger.info(
            "[nav-eval] lineage={"
            f"parent={lineage.get('source_parent_model_id')}, "
            f"low_level_parent={lineage.get('low_level_parent_model_id')}, "
            f"low_level_digest={lineage.get('low_level_state_digest')}"
            "}"
        )
        self.logger.info(
            f"[nav-eval] digest_check=OBSERVED high_level=PRESENT "
            f"(digest={info['low_level_state_digest']})"
        )
        self.logger.info(f"[nav-eval] policy_entry={self.algorithm_name}")
        self.logger.info(
            "[nav-eval] modules_loaded=vision_encoder,low_level_actor,high_level "
            "uses_height_scan_at_inference=false"
        )

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
            if id_str == "latest":
                resolved_numeric_id = visual_latest_model_id(path)
                if resolved_numeric_id is None:
                    raise FileNotFoundError(
                        f"[VisualPPO] latest selector found no visual checkpoint in {path}"
                    )
                resolved_id = str(resolved_numeric_id)
                candidates = visual_eval_checkpoint_candidates(path, resolved_numeric_id)
            else:
                candidates = visual_eval_checkpoint_candidates(path, id)
                resolved_id = self._resolve_filename_id(candidates, path, id_str)
        elif id_str == "latest":
            raise ValueError(
                "[VisualPPO] training preload selector 'latest' is forbidden; "
                "use the explicit platform model ID"
            )
        else:
            stage_config = self.usr_conf.get(self.stage.name, {})
            low_level_only_preload = bool(
                isinstance(stage_config, dict)
                and stage_config.get("low_level_only_preload", False)
            )
            if low_level_only_preload:
                candidates = low_level_only_parent_candidates(path, id)
            elif self.algorithm.schedule_mode == "p15_response_adapter_v1":
                candidates = p15_response_parent_candidates(path, id)
            elif self.algorithm.schedule_mode == "visual_command_generalization_v1":
                candidates = visual_command_parent_candidates(path, id)
            else:
                configured_parent_id = self.usr_conf.get(
                    Config.CURRENT.name, {}
                ).get("initial_parent_model_id")
                if configured_parent_id in (None, ""):
                    configured_parent_id = (
                        self.algorithm.source_parent_model_id or 28401
                    )
                candidates = visual_anchor_r2_training_candidates(
                    path,
                    id,
                    initial_parent_model_id=str(configured_parent_id),
                )
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
            if self.is_eval:
                diagnostic = visual_eval_checkpoint_diagnostics(path, id_str)
                self.logger.warning(
                    "[VisualPPO eval] no same-ID visual checkpoint; "
                    "cross-ID fallback remains disabled. "
                    f"requested_id={diagnostic['requested_id']}, "
                    f"same_id_expected={diagnostic['same_id_expected']}, "
                    f"same_id_existing={diagnostic['same_id_existing']}, "
                    f"other_visual_files={diagnostic['other_visual_files']}"
                )
            raise FileNotFoundError(
                f"[VisualPPO] no Anchor R2 / S0 checkpoint found in {path} "
                f"for selector={id_str!r}; tried={candidates}"
            )
        if self.is_eval:
            eval_info = self.algorithm.load_eval_bundle(
                selected, expected_spec=expected_spec, num_envs=self.num_envs
            )
            load_mode = "visual_eval"
        else:
            eval_info = None
            stage_config = self.usr_conf.get(self.stage.name, {})
            low_level_only_preload = bool(
                isinstance(stage_config, dict)
                and stage_config.get("low_level_only_preload", False)
            )
            if low_level_only_preload:
                load_mode = self.algorithm.load_low_level_only_bundle(
                    selected,
                    expected_spec=expected_spec,
                    env_seed=env_seed,
                    restore_optimizer=bool(
                        stage_config.get("low_level_only_restore_optimizer", False)
                    ),
                )
            else:
                load_mode = self.algorithm.load_training_bundle(
                    selected,
                    expected_spec=expected_spec,
                    env_seed=env_seed,
                )
        # Agent mirrors the algorithm's anchor session clock; elapsed_training_h
        # is only a cumulative-training record, not a phase driver (§4.3).
        self.training_elapsed_h = self.algorithm.anchor_session_elapsed_hours
        if self.is_p15_response and self.algorithm.resume_loaded:
            configured_offset = float(
                self.usr_conf.get(self.stage.name, {}).get(
                    "command_resume_offset_hours", 0.0
                )
            )
            if abs(configured_offset - self.training_elapsed_h) > 0.05:
                self.logger.warning(
                    "[P15Response] aisrv training clock resumed but the worker "
                    "command clock uses the TOML offset; set "
                    "p15_response.command_resume_offset_hours before an exact "
                    f"resume (saved={self.training_elapsed_h:.3f}, "
                    f"configured={configured_offset:.3f}). Training continues."
                )
        self.cur_model_name = selected
        bundle_id = self.algorithm.loaded_platform_model_id or "unknown"
        lineage_for_log = "unknown"
        if eval_info is not None:
            expected_file_id = resolved_id if str(resolved_id).isdigit() else None
            actual_bundle_id = eval_info.get("platform_model_id")
            if expected_file_id is not None and actual_bundle_id in (None, ""):
                self.logger.warning(
                    "[VisualPPO eval] checkpoint has no platform_model_id; "
                    "continuing with the operator-selected file: "
                    f"filename_id={expected_file_id}, path={selected}"
                )
            elif (
                expected_file_id is not None
                and str(actual_bundle_id) != str(expected_file_id)
            ):
                self.logger.warning(
                    "[VisualPPO eval] checkpoint platform ID differs from "
                    "the selected filename; continuing because the operator "
                    f"selected the preload: filename_id={expected_file_id}, "
                    f"bundle_id={actual_bundle_id!r}, path={selected}"
                )
            visual_conf = self.usr_conf.get("visual_policy_optimization", {})
            if not isinstance(visual_conf, dict):
                visual_conf = {}
            configured_parent = visual_conf.get("transition_parent_model_id")
            lineage_parent = (
                eval_info.get("lineage_transition_parent_platform_model_id")
                or eval_info.get("lineage_source_parent_model_id")
            )
            lineage_for_log = lineage_parent or "unknown"
            if lineage_parent in (None, ""):
                self.logger.warning(
                    "[VisualPPO eval] checkpoint has no parent lineage; "
                    "continuing with the operator-selected preload: "
                    f"configured_reference={configured_parent!r}, path={selected}"
                )
            elif (
                configured_parent not in (None, "")
                and str(lineage_parent) != str(configured_parent)
            ):
                self.logger.warning(
                    "[VisualPPO eval] checkpoint lineage differs from the "
                    "configured reference; continuing because the operator "
                    f"selected the preload: configured={configured_parent!r}, "
                    f"lineage_parent={lineage_parent!r}, path={selected}"
                )
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
        if eval_info is not None:
            self.logger.info(
                "[VisualPPO eval] student bundle loaded: "
                f"path={selected}, filename_id={resolved_id}, "
                f"bundle_id={eval_info['platform_model_id'] or 'unknown'}, "
                f"format={eval_info['format']}, schema={eval_info['schema_version']}, "
                f"schedule={eval_info['schedule_mode'] or 'unknown'}, "
                f"file_size_bytes={eval_info['file_size_bytes']}, "
                f"sha256={eval_info['sha256']}, "
                f"lineage_parent={lineage_for_log}, "
                "modules=vision_encoder+low_level_actor+action_distribution, "
                "uses_height_scan_at_inference=false"
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
            # The Arena has historically forced every Camera evaluation onto
            # the lbc_loco entry. If the requested ID is a nav bundle, promote
            # this in-memory assembly to the full nav evaluator instead of
            # dropping modules.high_level or asking the operator to create an
            # alias. The worker independently selects NavEvalConfig from the
            # Track+Camera eval TOML, so both sides retain the 57905-D contract.
            from agent_ppo.checkpoint_io import nav_eval_checkpoint_candidates

            lbc_eval_conf = self.usr_conf.get("lbc_loco", {})
            p2_eval_conf = self.usr_conf.get("p2_nav_ppo", {})
            terrain_mode = str(self.usr_conf.get("terrain", {}).get("mode", ""))
            low_level_only_eval = bool(
                isinstance(lbc_eval_conf, dict)
                and lbc_eval_conf.get("low_level_only_eval", False)
            ) or bool(
                terrain_mode == "standard"
                and isinstance(p2_eval_conf, dict)
                and p2_eval_conf.get("standard_low_level_only_eval", False)
            )
            p2_candidates = [
                candidate
                for candidate in p2_nav_evaluation_candidates(path, id)
                if candidate and os.path.isfile(candidate)
            ]
            if low_level_only_eval and p2_candidates:
                self._eval_requested_model_id = str(id)
                self._eval_checkpoint_path = None
                eval_path = p2_candidates[0]
                self._load_lbc_loco_for_eval(
                    eval_path,
                    allow_complete_hier_nav_low_level_only=True,
                )
                self.cur_model_name = eval_path
                return

            if nav_eval_checkpoint_candidates(path, id):
                self._promote_forced_lbc_eval_to_nav(path, id)
                return
            self._eval_requested_model_id = str(id)
            self._eval_checkpoint_path = None
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

    def _promote_forced_lbc_eval_to_nav(self, path, model_id) -> None:
        """Upgrade a platform-forced Camera/LBC eval to full hier-nav."""

        from agent_ppo.conf.conf import NavEvalConfig

        self.logger.warning(
            "[nav-eval] platform selected lbc_loco for a same-ID nav bundle; "
            "promoting to the full nav_eval assembly"
        )
        self.stage = NavEvalConfig
        self.algorithm_name = "nav_eval"
        self.is_lbc = False
        self.is_visual_ppo = False
        self.is_nav_dagger = False
        self.is_nav_eval = True
        # Drop the LBC-only teacher graph before allocating the nav modules;
        # keeping both assemblies resident can waste substantial Camera-eval
        # GPU memory even though the old graph is no longer callable.
        self.algorithm = None
        for stale_name in ("teacher_encoder", "teacher_actor"):
            if hasattr(self, stale_name):
                delattr(self, stale_name)
        self._init_nav_eval(NavEvalConfig, self.usr_conf)
        self._load_nav_for_eval(path, model_id)
        self.cur_model_name = self._nav_eval_checkpoint_path

    def _find_vision_eval_ckpt(self, path, id) -> str:
        """Resolve one explicit-ID Camera bundle for the platform LBC entry."""
        if not path:
            raise FileNotFoundError(
                "[LBC-Loco eval] checkpoint directory is empty; refusing "
                "uninitialized Camera inference"
            )

        requested_id = str(id)
        diagnostics = visual_eval_checkpoint_diagnostics(path, requested_id)
        candidates = visual_eval_checkpoint_candidates(path, requested_id)
        selected = candidates[0] if candidates else None
        candidate_order = [
            os.path.abspath(candidate)
            for candidate in diagnostics["same_id_expected"]
        ]
        existing_candidates = [os.path.abspath(candidate) for candidate in candidates]
        self.logger.info(
            "[LBC-Loco eval] checkpoint candidates "
            f"requested_id={requested_id}, "
            f"candidate_order={candidate_order}, "
            f"existing_candidates={existing_candidates}, "
            f"other_visual_files={diagnostics['other_visual_files']}, "
            f"selected={os.path.abspath(selected) if selected else None}"
        )
        if selected is not None:
            return selected
        raise FileNotFoundError(
            f"[LBC-Loco eval] No same-ID visual checkpoint found in {path}/ "
            f"for requested_id={requested_id}; "
            f"candidate_order={diagnostics['same_id_expected']}; "
            "refusing uninitialized Camera inference."
        )

    @staticmethod
    def _checkpoint_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _validate_lbc_eval_bundle_identity(self, ckpt, vision_path: str) -> dict:
        """Reject a Camera bundle not attributable to the requested model ID."""
        if self._eval_requested_model_id in (None, ""):
            raise RuntimeError(
                "[LBC-Loco eval] requested model ID was not set before load: "
                f"path={vision_path}"
            )
        identity = validate_visual_eval_bundle_identity(
            ckpt, self._eval_requested_model_id, logger=self.logger
        )
        identity["path"] = os.path.abspath(vision_path)
        return identity

    def _load_lbc_loco_for_eval(
        self, vision_path, *, allow_complete_hier_nav_low_level_only=False
    ):
        """Eval 模式：模拟真机视角，只加载 vision_encoder + teacher_actor。

        真机部署时 teacher_encoder（吃 height_scan）不存在，故 eval 也不加载它。
        评估只能接受带平台身份的 kaiwu_train_v1 视觉训练包；旧 lbc_loco
        导出制品没有该身份契约，因此不能被用来代替平台请求的模型。
        """
        ckpt = torch.load(vision_path, weights_only=False, map_location=self.device)
        if not isinstance(ckpt, dict):
            raise ValueError(
                "[LBC-Loco eval] checkpoint payload is not a dict: "
                f"{vision_path}"
            )
        if not is_kaiwu_train_bundle(ckpt):
            raise ValueError(
                f"[LBC-Loco eval] expected {KAIWU_TRAIN_FORMAT} checkpoint, "
                f"got format={ckpt.get('format')!r} at {vision_path}"
            )

        identity = self._validate_lbc_eval_bundle_identity(ckpt, vision_path)
        validate_low_level_spec(
            ckpt,
            {
                "proprio_dim": self.stage.num_proprio_obs,
                "scan_dim": self.stage.num_scan,
                "latent_dim": self.stage.latent_dim,
                "action_dim": self.stage.num_actions,
                "goal_dim": getattr(self.stage, "num_goal_obs", 0),
            },
        )
        modules = ckpt.get("modules", {})
        if not isinstance(modules, dict):
            raise KeyError(f"[LBC-Loco eval] modules missing in {vision_path}")
        # P1.5's ResponseAdapter is auxiliary and ignored by this deploy-shaped
        # Camera path. A real action-producing nav policy remains a hard stop.
        high_level_eval_disposition = classify_locomotion_eval_high_level(
            ckpt,
            allow_complete_hier_nav_low_level_only=(
                allow_complete_hier_nav_low_level_only
            ),
        )
        vision_section = modules.get("vision_encoder", {})
        ve_state = (
            vision_section.get("state_dict")
            if isinstance(vision_section, dict)
            else None
        )
        low_level = modules.get("low_level", {})
        act_state = (
            low_level.get("actor_state_dict")
            if isinstance(low_level, dict)
            else None
        )
        if allow_complete_hier_nav_low_level_only and isinstance(low_level, dict):
            canonical_encoder = low_level.get("locomotion_encoder")
            canonical_actor = low_level.get("actor")
            if isinstance(canonical_encoder, dict):
                ve_state = canonical_encoder.get("state_dict")
            if isinstance(canonical_actor, dict):
                act_state = canonical_actor.get("state_dict")
        if not isinstance(ve_state, dict):
            raise KeyError(
                "[LBC-Loco eval] low-level VisionEncoder state missing in "
                f"{vision_path}"
            )
        if not isinstance(act_state, dict):
            raise KeyError(
                "[LBC-Loco eval] low-level actor state missing in "
                f"{vision_path}"
            )

        # Camera evaluation must remain deploy-shaped: do not load the
        # privileged teacher encoder, critic, optimizer, or training state.
        validate_state_dict_finite(
            ve_state, "modules.vision_encoder.state_dict"
        )
        validate_state_dict_finite(
            act_state, "modules.low_level.actor_state_dict"
        )
        self.vision_encoder.load_state_dict(ve_state, strict=True)
        self.vision_encoder.eval()
        self.vision_encoder.reset_hidden_state(
            batch_size=self.num_envs, device=self.device
        )
        self.teacher_actor.load_state_dict(act_state, strict=True)
        self.teacher_actor.eval()
        checksum = self._checkpoint_sha256(vision_path)
        self._eval_checkpoint_path = identity["path"]
        self.logger.info(
            "[LBC-Loco eval] loaded visual student "
            f"selected_path={identity['path']}, sha256={checksum}, "
            f"bundle_id={identity['bundle_id']!r}, "
            f"lineage_id={identity['lineage_id']!r}, "
            f"lineage={identity['lineage']}, "
            f"high_level={high_level_eval_disposition}, "
            "modules=vision_encoder+low_level_actor, "
            "teacher_encoder_not_loaded=true, uses_height_scan_at_inference=false"
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
