#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import copy
import os

import numpy as np
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

import torch.optim as optim

from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.feature.definition import ActData
from agent_ppo.conf.conf import Config, _load_toml
from agent_ppo.model.actor_critic_encoder import ActorCriticEncoder
from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO
from tools.train_env_conf_validate import check_usr_conf


def _obs_height_grid(obs, scan_start: int = 45, scan_size: int = 256):
    if obs is None or not hasattr(obs, "shape") or obs.shape[-1] < scan_start + scan_size:
        return None
    side = int(scan_size ** 0.5)
    if side * side != scan_size:
        return None
    return obs[:, scan_start : scan_start + scan_size].view(obs.shape[0], side, side)


def _classify_pre_maze_terrain(obs, rl_nav_conf):
    grid = _obs_height_grid(
        obs,
        scan_start=int(rl_nav_conf.get("scan_start", 45)),
        scan_size=int(rl_nav_conf.get("scan_size", 256)),
    )
    if grid is None:
        return None

    row_start = max(int(rl_nav_conf.get("terrain_row_start", 3)), 0)
    row_end = min(int(rl_nav_conf.get("terrain_row_end", 13)), grid.shape[1])
    front_cols = min(int(rl_nav_conf.get("terrain_front_cols", 8)), grid.shape[2])
    if row_end <= row_start or front_cols <= 1:
        return None

    sector = grid[:, row_start:row_end, :front_cols]
    if sector.shape[1] == 0 or sector.shape[2] <= 1:
        return None

    lateral_std = sector.std(dim=1, unbiased=False).mean(dim=1)
    dx = sector[:, :, 1:] - sector[:, :, :-1]
    abs_dx = dx.abs()
    if abs_dx.numel() == 0:
        return None

    q = float(rl_nav_conf.get("terrain_step_quantile", 0.85))
    q = min(max(q, 0.0), 1.0)
    step_strength = torch.quantile(abs_dx.flatten(1), q, dim=1)
    sign_consistency = dx.mean(dim=(1, 2)).abs() / (abs_dx.mean(dim=(1, 2)) + 1e-6)
    if dx.shape[2] > 1:
        second_diff = (dx[:, :, 1:] - dx[:, :, :-1]).abs().mean(dim=(1, 2))
    else:
        second_diff = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)

    is_uniform = lateral_std < float(rl_nav_conf.get("terrain_lateral_std_threshold", 0.18))
    not_wall = sector.amin(dim=(1, 2)) > float(rl_nav_conf.get("terrain_wall_height_threshold", -1.05))
    terrain_like = is_uniform & not_wall & (
        step_strength > float(rl_nav_conf.get("terrain_slope_delta_threshold", 0.035))
    )
    stair_like = terrain_like & (
        (step_strength > float(rl_nav_conf.get("terrain_stair_delta_threshold", 0.10)))
        | (second_diff > float(rl_nav_conf.get("terrain_stair_second_diff_threshold", 0.055)))
    )
    slope_like = terrain_like & ~stair_like & (
        sign_consistency > float(rl_nav_conf.get("terrain_slope_sign_consistency_threshold", 0.55))
    )

    terrain_id = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
    terrain_id = torch.where(slope_like, torch.ones_like(terrain_id), terrain_id)
    terrain_id = torch.where(stair_like, torch.full_like(terrain_id, 2), terrain_id)
    return terrain_id


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

        if self.is_lbc:
            # LBC 阶段：创建学生 + 教师；不初始化 PPO storage
            self._init_lbc_loco(num_proprio, num_scan, env_conf, stage, usr_conf)
        else:
            self._init_flat(num_proprio, num_scan, stage)

        self.eval_command_override = None
        self.eval_phase_command_enabled = False
        self.eval_pre_maze_command = None
        self.eval_slope_command = None
        self.eval_stairs_command = None
        self.eval_maze_command = None
        self.eval_phase_maze_goal_dist_gate = 14.0
        self.eval_rl_nav_conf = {}
        if is_eval and stage.task_type == "track" and not self.is_lbc:
            train_stage_conf = self._load_train_stage_conf(stage)
            rl_nav_conf = train_stage_conf.get("rl_navigation", {}).copy()
            rl_nav_conf.update(usr_conf.get("rl_navigation", {}))
            self.eval_rl_nav_conf = rl_nav_conf.copy()
            self.eval_phase_command_enabled = bool(
                rl_nav_conf.get("phase_command_enabled", False)
            )
            self.eval_phase_maze_goal_dist_gate = float(
                rl_nav_conf.get("phase_maze_goal_dist_gate", 14.0)
            )
            pre_range = rl_nav_conf.get("pre_maze_lin_vel_x", [0.75, 1.0])
            slope_range = rl_nav_conf.get("slope_lin_vel_x", pre_range)
            stairs_range = rl_nav_conf.get("stairs_lin_vel_x", pre_range)
            maze_range = rl_nav_conf.get("maze_lin_vel_x", [0.45, 0.65])
            if len(pre_range) == 2:
                self.eval_pre_maze_command = torch.tensor(
                    [0.5 * (float(pre_range[0]) + float(pre_range[1])), 0.0, 0.0],
                    device=self.device,
                    dtype=torch.float32,
                )
            if len(slope_range) == 2:
                self.eval_slope_command = torch.tensor(
                    [0.5 * (float(slope_range[0]) + float(slope_range[1])), 0.0, 0.0],
                    device=self.device,
                    dtype=torch.float32,
                )
            if len(stairs_range) == 2:
                self.eval_stairs_command = torch.tensor(
                    [0.5 * (float(stairs_range[0]) + float(stairs_range[1])), 0.0, 0.0],
                    device=self.device,
                    dtype=torch.float32,
                )
            if len(maze_range) == 2:
                self.eval_maze_command = torch.tensor(
                    [0.5 * (float(maze_range[0]) + float(maze_range[1])), 0.0, 0.0],
                    device=self.device,
                    dtype=torch.float32,
                )
            if bool(rl_nav_conf.get("eval_command_override", True)):
                cmd = rl_nav_conf.get("eval_command", [0.55, 0.0, 0.0])
                if len(cmd) == 3:
                    self.eval_command_override = torch.tensor(
                        cmd, device=self.device, dtype=torch.float32
                    )
                    self.logger.info(
                        "[RLNavigation] Eval policy command obs override enabled: "
                        f"{cmd}"
                    )

        self.num_steps_per_env = stage.num_steps_per_env
        self.save_interval = stage.model_save_interval

        # LBC: 无 PPO storage 需要初始化
        if not self.is_lbc:
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

        super().__init__(agent_type, device, logger, monitor)

    def _load_train_stage_conf(self, stage):
        train_conf_file = (
            f"agent_ppo/conf/train_env_conf_{stage.task_type}_{stage.name}.toml"
        )
        if not os.path.exists(train_conf_file):
            return {}
        try:
            return _load_toml(train_conf_file)
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning(
                    "[RLNavigation] Failed to load train stage config "
                    f"from {train_conf_file}: {exc}"
                )
            return {}

    def _apply_eval_command_to_obs(self, obs):
        if obs is None or obs.shape[-1] < 9:
            return obs
        if (
            self.eval_phase_command_enabled
            and self.eval_pre_maze_command is not None
            and self.eval_maze_command is not None
            and obs.shape[-1] >= 304
        ):
            nav_obs = obs.clone()
            goal_dist = torch.clamp(nav_obs[:, 303], 0.0, 1.0) * 20.0
            maze_phase = goal_dist < self.eval_phase_maze_goal_dist_gate
            pre_command = self.eval_pre_maze_command.to(
                device=obs.device, dtype=obs.dtype
            ).expand(obs.shape[0], -1)
            if bool(self.eval_rl_nav_conf.get("terrain_phase_speed_enabled", False)):
                terrain_id = _classify_pre_maze_terrain(nav_obs, self.eval_rl_nav_conf)
                if terrain_id is not None:
                    if self.eval_slope_command is not None:
                        slope_command = self.eval_slope_command.to(
                            device=obs.device, dtype=obs.dtype
                        ).expand(obs.shape[0], -1)
                        pre_command = torch.where(
                            (terrain_id == 1).unsqueeze(1), slope_command, pre_command
                        )
                    if self.eval_stairs_command is not None:
                        stairs_command = self.eval_stairs_command.to(
                            device=obs.device, dtype=obs.dtype
                        ).expand(obs.shape[0], -1)
                        pre_command = torch.where(
                            (terrain_id == 2).unsqueeze(1), stairs_command, pre_command
                        )
            maze_command = self.eval_maze_command.to(
                device=obs.device, dtype=obs.dtype
            ).expand(obs.shape[0], -1)
            nav_obs[:, 6:9] = torch.where(
                maze_phase.unsqueeze(1), maze_command, pre_command
            )
            return nav_obs
        if self.eval_command_override is None:
            return obs
        nav_obs = obs.clone()
        nav_obs[:, 6:9] = self.eval_command_override.to(
            device=obs.device, dtype=obs.dtype
        )
        return nav_obs

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
            init_noise_std=getattr(stage, "init_noise_std", 1.0),
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
            clip_param=getattr(stage, "clip_param", 0.2),
            entropy_coef=getattr(stage, "entropy_coef", 0.01),
            desired_kl=getattr(stage, "desired_kl", 0.01),
            schedule=getattr(stage, "schedule", "adaptive"),
            min_learning_rate=getattr(stage, "min_learning_rate", 1e-5),
            max_learning_rate=getattr(stage, "max_learning_rate", 1e-2),
            num_mini_batches=stage.num_mini_batches,
            num_learning_epochs=stage.num_learning_epochs,
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

        actor_input_dim = (
            stage.proprio_dim
            + stage.latent_dim
            + getattr(stage, "num_goal_obs", 0)
        )
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
            goal_dim=getattr(stage, "num_goal_obs", 0),
            depth_shape=(stage.depth_height, stage.depth_width, stage.depth_channels),
        )

        # 为与 PPO 路径下某些属性兼容，point self.model 到学生
        self.model = self.vision_encoder

    def exploit(self, list_obs_data):
        """
        Exploit learned policy for action selection in evaluation mode.
        在评估模式下利用已学习的策略进行动作选择。
        """
        (obs) = list_obs_data
        with torch.no_grad():
            if self.is_lbc:
                return self._exploit_lbc_loco(obs)
            obs = self._apply_eval_command_to_obs(obs)
            actions = self.algorithm.actor_critic.act_inference(obs)
            return [ActData(action=actions)]

    def _exploit_lbc_loco(self, obs):
        """LBC Loco eval: 学生 VisionEncoder 闭环推理。

        obs: standard [proprio+scan+depth], or Track [proprio+scan+goal+depth]
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
        actor_input = self.algorithm._teacher_actor_input(obs_dict, student_latent)
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
        (obs, critic_obs) = list_obs_data
        with torch.no_grad():
            if self.is_eval:
                obs = self._apply_eval_command_to_obs(obs)
            hidden_states = None
            if getattr(self.algorithm.actor_critic, "is_recurrent", False):
                current_hidden = self.algorithm.actor_critic.get_hidden_states()
                if current_hidden is None or current_hidden[0].shape[1] != obs.shape[0]:
                    self.algorithm.actor_critic._init_hidden_states(
                        obs.shape[0], obs.device, obs.dtype
                    )
                    current_hidden = self.algorithm.actor_critic.get_hidden_states()
                hidden_states = tuple(
                    state.detach().clone() for state in current_hidden
                )

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
                hidden_states,
            )

    def save_model(self, path=None, id="1"):
        """
        Save model checkpoint.
        保存模型 checkpoint。

        Path is driven by stage.ckpt_name:
          - LocomotionConfig  -> model.ckpt-{id}.pkl
          - LBCLocoConfig     -> model.ckpt-lbc-loco-{id}.pkl
                                  plus model.ckpt-{id}.pkl platform alias
        """
        path = self._resolve_checkpoint_dir(path, create=True)
        ckpt_name = getattr(self.stage, "ckpt_name", "") or ""
        if ckpt_name:
            model_file_path = os.path.join(path, f"{ckpt_name}-{str(id)}.pkl")
        else:
            model_file_path = os.path.join(path, f"model.ckpt-{str(id)}.pkl")

        if self.is_lbc:
            # LBC saves via algorithm.save which writes the full LBC dict
            # (vision_encoder + teachers + optimizer + iter).
            self.algorithm.save(
                model_file_path,
                format_tag=self.algorithm_name,
                iteration=self.algorithm.current_iteration,
            )
            self.logger.info(f"[{self.algorithm_name}] save {model_file_path} successfully")
            platform_alias = os.path.join(path, f"model.ckpt-{str(id)}.pkl")
            if os.path.abspath(platform_alias) != os.path.abspath(model_file_path):
                self.algorithm.save(
                    platform_alias,
                    format_tag=self.algorithm_name,
                    iteration=self.algorithm.current_iteration,
                )
                self.logger.info(
                    f"[{self.algorithm_name}] save platform alias "
                    f"{platform_alias} successfully"
                )
        else:
            torch.save(self.model.state_dict(), model_file_path)
            file_size = os.path.getsize(model_file_path)
            self.logger.info(
                f"save model {model_file_path} successfully, size={file_size} bytes"
            )

        # Side model: 训练 lbc_loco 时同时落一份 locomotion 形态的 ckpt。
        # 将冻结的 teacher_encoder + teacher_actor 重组为 encoder.*/actor.* 命名，
        # 使产物目录内保留一份可独立部署 / 供下阶段拆分教师的 locomotion ckpt。
        self._save_side_locomotion(path, id)

    @staticmethod
    def _resolve_checkpoint_dir(path, create=False):
        """Resolve the framework checkpoint directory with a local fallback."""
        checkpoint_dir = path or os.environ.get("KAIWU_MODEL_CKPT_DIR")
        if not checkpoint_dir:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "ckpt")
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        if create:
            os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir

    def _save_side_locomotion(self, path, id):
        """保存 lbc_loco 教师为 locomotion 形态的 side ckpt。

        仅当 teacher_encoder / teacher_actor 都已初始化（即 lbc_loco 阶段）时触发，
        输出 model.ckpt-locomotion-{id}.pkl，key 前缀重映射为 encoder.*/actor.*。

        :param path: checkpoint 保存目录
        :param id: checkpoint 编号
        """
        if not (hasattr(self, "teacher_encoder") and hasattr(self, "teacher_actor")):
            return

        if getattr(self.stage, "task_type", "standard") == "track":
            teacher_name = "model.ckpt-track-nav"
        else:
            teacher_name = "model.ckpt-locomotion"
        loco_path = f"{path}/{teacher_name}-{str(id)}.pkl"
        loco_state = {}
        for k, v in self.teacher_encoder.state_dict().items():
            encoder_key = k.removeprefix("mlp.")
            loco_state[f"encoder.{encoder_key}"] = v
        for k, v in self.teacher_actor.state_dict().items():
            loco_state[f"actor.{k}"] = v
        torch.save(loco_state, loco_path)
        self.logger.info(
            f"save side teacher (encoder.*/actor.*) {loco_path} successfully"
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
        path = self._resolve_checkpoint_dir(path, create=False)
        if self.is_lbc:
            self._load_lbc_loco(path, id)
            return
        self._load_flat(path, id)

    def _load_flat(self, path, id):
        """Load single-model checkpoint."""
        ckpt_name = getattr(self.stage, "ckpt_name", "") or ""
        checkpoint_id = str(id)
        candidate_names = []
        if ckpt_name:
            candidate_names.append(f"{ckpt_name}-{checkpoint_id}.pkl")

        if getattr(self.stage, "task_type", "standard") == "track":
            candidate_names.extend(
                [
                    f"model.ckpt-nav-{checkpoint_id}.pkl",
                    f"model.ckpt-track-nav-{checkpoint_id}.pkl",
                    f"model.ckpt-{checkpoint_id}.pkl",
                    f"model.ckpt-locomotion-{checkpoint_id}.pkl",
                ]
            )
        else:
            candidate_names.extend(
                [
                    f"model.ckpt-{checkpoint_id}.pkl",
                    f"model.ckpt-locomotion-{checkpoint_id}.pkl",
                ]
            )

        # Preserve order while removing duplicate aliases.
        candidate_paths = []
        for name in candidate_names:
            candidate = os.path.join(path, name)
            if candidate not in candidate_paths:
                candidate_paths.append(candidate)

        model_file_path = next(
            (candidate for candidate in candidate_paths if os.path.isfile(candidate)),
            None,
        )
        if model_file_path is None:
            available = []
            if os.path.isdir(path):
                available = sorted(
                    name for name in os.listdir(path) if name.endswith(".pkl")
                )
            raise FileNotFoundError(
                "No flat checkpoint found. "
                f"Tried: {candidate_paths}. Available pkl files: {available}"
            )
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
            return

        pretrained = torch.load(model_file_path, map_location=self.device)
        current_state = self.model.state_dict()
        if not isinstance(pretrained, dict) or not any(
            key in pretrained for key in current_state
        ):
            raise ValueError(
                f"Checkpoint is not a flat {self.stage.name} policy: "
                f"{model_file_path}. Do not preload an LBC student checkpoint "
                "for PPO safety fine-tuning."
            )

        if self._ckpt_exact_match(pretrained, current_state):
            self.model.load_state_dict(pretrained)
            self.logger.info(f"load model {model_file_path} successfully (exact match)")
        else:
            self._load_model_partial(self.model, pretrained, model_file_path)

        self._enforce_action_std_bounds()
        self._install_action_anchor_policy()
        self.cur_model_name = model_file_path

    def _install_action_anchor_policy(self):
        coef = float(getattr(self.stage, "action_anchor_coef", 0.0))
        if coef <= 0.0 or not hasattr(self, "algorithm"):
            return

        reference_model = copy.deepcopy(self.model).to(self.device)
        self.algorithm.set_reference_policy(reference_model, action_anchor_coef=coef)
        if self.logger is not None:
            self.logger.info(
                f"[PPO] fixed pretrained-policy anchor enabled, coef={coef}, "
                f"ema={getattr(self.stage, 'action_anchor_ema', 0.0)}"
            )

    def _enforce_action_std_bounds(self):
        min_std_cfg = getattr(self.stage, "min_normalized_std", None)
        max_std_cfg = getattr(self.stage, "max_normalized_std", None)
        if min_std_cfg is None and max_std_cfg is None:
            return

        with torch.no_grad():
            if hasattr(self.model, "std"):
                std = torch.nan_to_num(
                    self.model.std.data,
                    nan=1.0,
                    posinf=1.0e6,
                    neginf=0.0,
                )
                if min_std_cfg is not None:
                    min_std = torch.tensor(
                        min_std_cfg, device=self.device, dtype=std.dtype
                    )
                    if min_std.shape == std.shape:
                        std = torch.maximum(std, min_std)
                if max_std_cfg is not None:
                    max_std = torch.tensor(
                        max_std_cfg, device=self.device, dtype=std.dtype
                    )
                    if max_std.shape == std.shape:
                        std = torch.minimum(std, max_std)
                self.model.std.data.copy_(std)
            elif hasattr(self.model, "log_std"):
                log_std = torch.nan_to_num(
                    self.model.log_std.data,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if min_std_cfg is not None:
                    min_std = torch.tensor(
                        min_std_cfg, device=self.device, dtype=log_std.dtype
                    )
                    if min_std.shape == log_std.shape:
                        log_std = torch.maximum(log_std, torch.log(min_std))
                if max_std_cfg is not None:
                    max_std = torch.tensor(
                        max_std_cfg, device=self.device, dtype=log_std.dtype
                    )
                    if max_std.shape == log_std.shape:
                        log_std = torch.minimum(log_std, torch.log(max_std))
                self.model.log_std.data.copy_(log_std)

            self.logger.info(
                f"[PPO] action std bounds enforced: "
                f"min={min_std_cfg}, max={max_std_cfg}"
            )

    def _load_lbc_loco(self, path, id):
        """LBC Loco 模型加载，按 is_eval 分发训练期 / eval 两条路径。

        Eval 路径（模拟真机视角）:
            必须有 main ckpt (model.ckpt-lbc-loco-{id}.pkl)
            → 仅加载 vision_encoder + teacher_actor（不加载 teacher_encoder）
            → 真机没有 height_scan，teacher_encoder 永远不会被调用

        训练期路径:
            P1 (续训): main ckpt → algorithm.load(...)，全部恢复
            P2 (首训): model.ckpt-locomotion-{id}.pkl → 按前缀拆分教师
            miss: FileNotFoundError
        """
        ckpt_name = getattr(self.stage, "ckpt_name", "model.ckpt-lbc-loco")
        main_path = f"{path}/{ckpt_name}-{str(id)}.pkl"
        alias_path = f"{path}/model.ckpt-{str(id)}.pkl"
        teacher_paths = [
            alias_path,
            f"{path}/model.ckpt-nav-{str(id)}.pkl",
            f"{path}/model.ckpt-track-nav-{str(id)}.pkl",
            f"{path}/model.ckpt-locomotion-{str(id)}.pkl",
        ]

        def _is_lbc_checkpoint(candidate):
            if not os.path.exists(candidate):
                return False
            checkpoint = torch.load(
                candidate, weights_only=False, map_location=self.device
            )
            return (
                isinstance(checkpoint, dict)
                and checkpoint.get("format") == self.algorithm_name
            )

        # Eval 路径：模拟真机视角
        is_eval = getattr(self, "is_eval", False)
        if is_eval:
            vision_path = main_path if os.path.exists(main_path) else None
            if vision_path is None and _is_lbc_checkpoint(alias_path):
                vision_path = alias_path
            if vision_path is None:
                raise FileNotFoundError(
                    f"[LBC-Loco eval] Required vision ckpt not found; tried "
                    f"{main_path} and {alias_path}. "
                    f"Eval mode simulates real-robot deployment and cannot fall back to "
                    f"teacher-only ckpt."
                )
            self._load_lbc_loco_for_eval(vision_path)
            self.cur_model_name = vision_path
            return

        # 训练期路径 P1: 续训 main ckpt
        if os.path.exists(main_path):
            self.algorithm.load(
                main_path,
                expected_format=self.algorithm_name,
                load_optimizer=True,
            )
            self.cur_model_name = main_path
            self.logger.info(
                f"[LBC-Loco] Loaded main ckpt {main_path} "
                f"(iter={self.algorithm.current_iteration})"
            )
            return

        if _is_lbc_checkpoint(alias_path):
            self.algorithm.load(
                alias_path,
                expected_format=self.algorithm_name,
                load_optimizer=True,
            )
            self.cur_model_name = alias_path
            self.logger.info(
                f"[LBC-Loco] Loaded platform-alias student ckpt {alias_path} "
                f"(iter={self.algorithm.current_iteration})"
            )
            return

        # 训练期路径 P2: locomotion ckpt
        teacher_path = next(
            (
                candidate
                for candidate in teacher_paths
                if os.path.exists(candidate) and not _is_lbc_checkpoint(candidate)
            ),
            None,
        )
        if teacher_path is not None:
            self.algorithm.load_teacher_from_locomotion_ckpt(teacher_path)
            self.cur_model_name = teacher_path
            self.logger.info(
                f"[LBC-Loco] Teacher loaded by splitting flat ckpt {teacher_path}; "
                f"student randomly initialized."
            )
            return

        raise FileNotFoundError(
            f"[LBC-Loco] No ckpt found in {path}/: "
            f"tried {main_path}/{alias_path} (resume) and "
            f"{teacher_paths} (first-train teacher)."
        )

    def _load_lbc_loco_for_eval(self, vision_path):
        """Eval 模式：模拟真机视角，只加载 vision_encoder + teacher_actor。

        真机部署时 teacher_encoder（吃 height_scan）不存在，故 eval 也不加载它。
        """
        ckpt = torch.load(vision_path, weights_only=False, map_location=self.device)
        got = ckpt.get("format")
        if got != self.algorithm_name:
            raise ValueError(
                f"Ckpt format mismatch: expected '{self.algorithm_name}', got '{got}' "
                f"at {vision_path}."
            )
        checkpoint_goal_dim = int(ckpt.get("goal_dim", 0))
        expected_goal_dim = int(getattr(self.stage, "num_goal_obs", 0))
        if checkpoint_goal_dim != expected_goal_dim:
            raise ValueError(
                f"LBC checkpoint goal_dim mismatch: expected {expected_goal_dim}, "
                f"got {checkpoint_goal_dim} at {vision_path}"
            )

        # 学生 VisionEncoder：真机推理主角
        if "vision_encoder_state_dict" not in ckpt:
            raise KeyError(f"vision_encoder_state_dict missing in {vision_path}")
        self.vision_encoder.load_state_dict(ckpt["vision_encoder_state_dict"])
        self.vision_encoder.eval()
        self.vision_encoder.reset_hidden_state(batch_size=self.num_envs, device=self.device)

        # 教师 Actor：latent → joint 的解码器，真机仍需要
        if "teacher_actor_state_dict" not in ckpt:
            raise KeyError(f"teacher_actor_state_dict missing in {vision_path}")
        self.teacher_actor.load_state_dict(ckpt["teacher_actor_state_dict"])
        self.teacher_actor.eval()

        # teacher_encoder 不加载 —— 真机无 height_scan
        self.logger.info(
            f"[LBC-Loco eval] Loaded student + teacher_actor from {vision_path} "
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
