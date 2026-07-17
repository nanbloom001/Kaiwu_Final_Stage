#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import re

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
        self.num_critic_obs = stage.num_critic_observations

        # Algorithm dispatch
        # 算法分发
        self.algorithm_name = getattr(stage, "algorithm", "ppo")
        self.is_lbc = self.algorithm_name == "lbc_loco"

        if self.is_lbc:
            # LBC 阶段：创建学生 + 教师；不初始化 PPO storage
            self._init_lbc_loco(num_proprio, num_scan, env_conf, stage, usr_conf)
        else:
            self._init_flat(num_proprio, num_scan, stage)

        self.num_steps_per_env = stage.num_steps_per_env

        # Track 评估时，平台下发的速度命令可能和训练时不一致（评估随机采样，
        # 训练恒速直线）。强制覆盖 obs 命令槽位为训练时的恒速锚点，保证评估
        # 时机器人在训练分布内行走。
        # 默认 False：Stage3B 开门控后，评估时由门控接管命令，不再恒速覆盖。
        self.eval_command_override = None
        if self.is_eval and stage.task_type == "track":
            rl_nav_conf = usr_conf.get("rl_navigation", {})
            if bool(rl_nav_conf.get("eval_command_override", False)):
                cmd = rl_nav_conf.get("eval_command", [0.45, 0.0, 0.0])
                if len(cmd) == 3:
                    self.eval_command_override = torch.tensor(
                        cmd, device=self.device, dtype=torch.float32
                    )
                    self.logger.info(
                        f"[eval] command override enabled: {cmd}"
                    )
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
        critic_proprio_dim = stage.num_critic_observations - num_scan - num_goal_obs
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
            # PPO 核心参数全部从 stage 显式传入（计划 6.3：不让 AlgorithmPPO 隐式读取）
            learning_rate=stage.lr,
            schedule=getattr(stage, "schedule", "adaptive"),
            min_learning_rate=getattr(stage, "min_learning_rate", 1e-5),
            max_learning_rate=getattr(stage, "max_learning_rate", 1e-2),
            clip_param=getattr(stage, "clip_param", 0.2),
            gamma=getattr(stage, "gamma", 0.99),
            lam=getattr(stage, "lam", 0.95),
            value_loss_coef=getattr(stage, "value_loss_coef", 1.0),
            entropy_coef=getattr(stage, "entropy_coef", 0.01),
            max_grad_norm=getattr(stage, "max_grad_norm", 1.0),
            desired_kl=getattr(stage, "desired_kl", 0.01),
            min_normalized_std=getattr(stage, "min_normalized_std", None),
            max_normalized_std=getattr(stage, "max_normalized_std", None),
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

    def exploit(self, list_obs_data):
        """
        Exploit learned policy for action selection in evaluation mode.
        在评估模式下利用已学习的策略进行动作选择。
        """
        (obs) = list_obs_data
        with torch.no_grad():
            if self.is_lbc:
                return self._exploit_lbc_loco(obs)
            # Track 评估：覆盖 obs 命令槽位 [6:9] 为训练时的恒速直线锚点
            if self.eval_command_override is not None:
                obs = obs.clone()
                obs[:, 6:9] = self.eval_command_override.expand(obs.shape[0], -1)
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

    def save_model(self, path=None, id=None):
        """
        Save model checkpoint.
        保存 model checkpoint。

        ID 分配（计划 3.3）：扫描同前缀全局最大 ID 并 +1，保证全局递增、不覆盖历史。
        若显式传入 id 且大于扫描值，则用传入值（兼容平台预分配 id）。
        """
        ckpt_name = getattr(Config.CURRENT, "ckpt_name", "") or ""

        # 计划 3.3：扫描同前缀全局最大 ID，递增分配
        scanned = self._max_global_id(path) if path else 0
        if id is not None and str(id).isdigit():
            next_id = max(scanned + 1, int(id))
        else:
            next_id = scanned + 1
        id = str(next_id)

        if ckpt_name:
            model_file_path = f"{path}/{ckpt_name}-{id}.pkl"
        else:
            model_file_path = f"{path}/model.ckpt-{id}.pkl"

        if self.is_lbc:
            # LBC saves via algorithm.save which writes the full LBC dict
            # (vision_encoder + teachers + optimizer + iter).
            self.algorithm.save(
                model_file_path,
                format_tag=self.algorithm_name,
                iteration=self.algorithm.current_iteration,
            )
            self.logger.info(f"[{self.algorithm_name}] save {model_file_path} successfully")
        else:
            self._atomic_save(self.model.state_dict(), model_file_path, id)
            self.logger.info(f"save model {model_file_path} successfully")

        self._save_standard_eval_alias(path, id, model_file_path)

        # Side model: 训练 lbc_loco 时同时落一份 locomotion 形态的 ckpt。
        # 将冻结的 teacher_encoder + teacher_actor 重组为 encoder.*/actor.* 命名，
        # 使产物目录内保留一份可独立部署 / 供下阶段拆分教师的 locomotion ckpt。
        self._save_side_locomotion(path, id)

    def _max_global_id(self, path):
        """扫描目录下所有 model.ckpt-*.* 文件的数字 ID，返回全局最大值（计划 3.3）。

        用平台同款正则提取 ID，跨所有前缀（nav/locomotion/lbc-loco）统一取最大，
        保证不同阶段之间 ID 不冲突。
        """
        if not path or not os.path.isdir(path):
            return 0
        id_re = re.compile(r"model\.ckpt-[a-z]*-*([0-9][0-9]*)\..*$")
        best = 0
        try:
            for fname in os.listdir(path):
                if not fname.startswith("model.ckpt-") or not fname.endswith(".pkl"):
                    continue
                m = id_re.match(fname)
                if m:
                    best = max(best, int(m.group(1)))
        except OSError:
            pass
        return best

    def _atomic_save(self, state_dict, final_path, id):
        """原子保存（计划 3.7）：临时文件 → 回读校验 → os.replace。

        临时文件名不以 model.ckpt- 开头，避免被探活误识别为半成品模型。
        """
        path_dir = os.path.dirname(final_path)
        tmp_path = f"{path_dir}/.tmp-save-{id}.pkl"
        torch.save(state_dict, tmp_path)
        try:
            _check = torch.load(tmp_path, map_location="cpu", weights_only=False)
            if not isinstance(_check, dict):
                raise RuntimeError("checkpoint round-trip failed: not a dict")
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        os.replace(tmp_path, final_path)

    def _save_standard_eval_alias(self, path, id, main_model_path):
        """为自定义 standard PPO 阶段额外保存默认评估兼容文件。

        自定义 standard PPO 训练阶段主 checkpoint 保持 ckpt_name 指定的命名，
        便于训练保存、预训练拷贝和探活。同时额外落一份 model.ckpt-locomotion-{id}.pkl，
        兼容平台默认 standard 评估链路按 locomotion 名称取模型的行为。
        """
        if self.is_lbc:
            return

        stage = Config.CURRENT
        if getattr(stage, "task_type", "") != "standard":
            return
        if getattr(stage, "algorithm", "") != "ppo":
            return
        if getattr(stage, "name", "") == "locomotion":
            return

        alias_path = f"{path}/model.ckpt-locomotion-{str(id)}.pkl"
        if alias_path == main_model_path:
            return

        self._atomic_save(self.model.state_dict(), alias_path, f"alias-{id}")
        self.logger.info(f"save standard eval alias {alias_path} successfully")

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
        self._atomic_save(loco_state, loco_path, f"side-{id}")
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
        if self.is_lbc:
            self._load_lbc_loco(path, id)
            return
        try:
            self._load_flat(path, id)
        except FileNotFoundError as exc:
            # 续训阶段必须有预训练模型，禁止随机初始化
            self.logger.error(
                f"[Checkpoint] required preload checkpoint missing: "
                f"path={path}, id={id}"
            )
            raise RuntimeError(
                "Stage continuation requires the specified checkpoint; "
                "random initialization is forbidden."
            ) from exc

    def _load_flat(self, path, id):
        """Load single-model checkpoint.

        加载顺序（按优先级）:
          1. 精确 id:   {ckpt_name}-{id}.pkl（平台 preload 指定 id，续训优先）
          2. 同前缀最新: 目录里 ckpt_name 前缀的最新文件（id 对不上时兜底）
          3. standard 迁移: model.ckpt-locomotion-{id}.pkl（首次从 standard 起步）
          4. 默认兜底:  model.ckpt-{id}.pkl
        """
        ckpt_name = getattr(Config.CURRENT, "ckpt_name", "") or ""

        if ckpt_name:
            exact_path = f"{path}/{ckpt_name}-{str(id)}.pkl"
            # 1. 精确 ID 优先（续训时必须加载指定 checkpoint）
            if os.path.exists(exact_path):
                model_file_path = exact_path
                self.logger.info(
                    f"[Checkpoint] exact requested checkpoint selected: {model_file_path}"
                )
            else:
                # 2. 精确 ID 不存在，寻找同前缀最新
                latest = self._find_latest_by_prefix(path, ckpt_name)
                if latest is not None:
                    model_file_path = latest
                    self.logger.warning(
                        f"[Checkpoint] requested id={id} not found, "
                        f"fallback to latest: {model_file_path}"
                    )
                else:
                    model_file_path = exact_path
        else:
            model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        if not os.path.exists(model_file_path):
            # 3/4. 兜底：standard locomotion 产物 / 默认命名
            aliases = [
                f"{path}/model.ckpt-locomotion-{str(id)}.pkl",
                f"{path}/model.ckpt-{str(id)}.pkl",
            ]
            loco_latest = self._find_latest_by_prefix(path, "model.ckpt-locomotion")
            if loco_latest is not None:
                aliases.insert(0, loco_latest)
            found = next((a for a in aliases if a != model_file_path and os.path.exists(a)), None)
            if found is None:
                raise FileNotFoundError(
                    f"No flat checkpoint found: {model_file_path}"
                )
            model_file_path = found
            self.logger.info(f"flat ckpt fallback to: {model_file_path}")
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
            return

        pretrained = torch.load(model_file_path, map_location=self.device)
        current_state = self.model.state_dict()

        if self._ckpt_exact_match(pretrained, current_state):
            self.model.load_state_dict(pretrained)
            self.logger.info(f"load model {model_file_path} successfully (exact match)")
        else:
            self._load_model_partial(self.model, pretrained, model_file_path)

        self.cur_model_name = model_file_path

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
        main_path = f"{path}/model.ckpt-lbc-loco-{str(id)}.pkl"
        loco_path = f"{path}/model.ckpt-locomotion-{str(id)}.pkl"

        # Eval 路径：模拟真机视角
        is_eval = getattr(self, "is_eval", False)
        if is_eval:
            if not os.path.exists(main_path):
                raise FileNotFoundError(
                    f"[LBC-Loco eval] Required vision ckpt not found: {main_path}. "
                    f"Eval mode simulates real-robot deployment and cannot fall back to "
                    f"teacher-only ckpt."
                )
            self._load_lbc_loco_for_eval(main_path)
            self.cur_model_name = main_path
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

        # 训练期路径 P2: locomotion ckpt
        if os.path.exists(loco_path):
            self.algorithm.load_teacher_from_locomotion_ckpt(loco_path)
            self.cur_model_name = loco_path
            self.logger.info(
                f"[LBC-Loco] Teacher loaded by splitting locomotion ckpt {loco_path}; "
                f"student randomly initialized."
            )
            return

        raise FileNotFoundError(
            f"[LBC-Loco] No ckpt found in {path}/: "
            f"tried {main_path} (resume) and {loco_path} (first-train teacher)."
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
    def _find_latest_by_prefix(path, ckpt_name):
        """在目录里按前缀找最新模型文件，ID 提取复用平台探活正则。

        与平台探活命令口径一致：
            find ... -name "model.ckpt-*.*" | sed 's/.*model.ckpt-[a-z]*-*\\([0-9]*\\)\\..*/\\1/'
        ckpt_name 形如 "model.ckpt-nav" / "model.ckpt-locomotion"。
        返回最新文件的完整路径，无匹配返回 None。
        """
        if not path or not os.path.isdir(path):
            return None
        prefix = ckpt_name.split("model.ckpt-", 1)[-1] if "model.ckpt-" in ckpt_name else ckpt_name
        id_re = re.compile(r"model\.ckpt-[a-z]*-*([0-9][0-9]*)\..*$")
        best_id, best_file = -1, None
        try:
            for fname in os.listdir(path):
                # 只匹配 .pkl 主模型，跳过临时文件/sidecar（不以 model.ckpt- 开头的不算）
                if not fname.startswith("model.ckpt-") or not fname.endswith(".pkl"):
                    continue
                # 前缀过滤：nav 前缀不应匹配到 locomotion 文件
                tag = fname[len("model.ckpt-"):]
                if "-" in tag:
                    file_prefix = tag.rsplit("-", 1)[0]
                else:
                    file_prefix = ""
                if file_prefix != prefix:
                    continue
                m = id_re.match(fname)
                if not m:
                    continue
                cid = int(m.group(1))
                if cid > best_id:
                    best_id, best_file = cid, os.path.join(path, fname)
        except OSError:
            return None
        return best_file

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
