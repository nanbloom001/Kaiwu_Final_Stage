#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

import torch.optim as optim

from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.feature.definition import ActData
from agent_ppo.conf.conf import Config
from agent_ppo.model.actor_critic import ActorCritic
from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO
from tools.train_env_conf_validate import check_usr_conf


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device="cuda", logger=None, monitor=None):
        self.cur_model_name = "ActorCritic"
        self.device = device
        self.logger = logger
        self.monitor = monitor

        usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(self.logger)
        self.is_eval = is_eval
        valid, message = check_usr_conf(usr_conf, is_eval, self.logger)
        if not valid:
            self.logger.error(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")
            raise Exception(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")

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

        # policy obs = proprio + scan + goal
        # 策略观测 = 本体感知 + 扫描 + 目标
        self.num_obs = num_proprio + num_scan + num_goal_obs
        # Command and goal-feature patching lives in the observation processors,
        # where env sensors and command-manager state are still available.

        self._init_flat(num_proprio, num_scan, num_goal_obs, stage)

        self.num_steps_per_env = stage.num_steps_per_env
        self.save_interval = stage.model_save_interval

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

    def _init_flat(self, num_proprio, num_scan, num_goal_obs, stage):
        """
        Initialize single-model (flat) architecture.
        初始化单模型（扁平）架构。
        """
        self.model = ActorCritic(
            num_obs=self.num_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            actor_hidden_dims=stage.actor_hidden_dims,
            critic_hidden_dims=stage.critic_hidden_dims,
            activation=stage.activation,
            init_noise_std=getattr(stage, "init_noise_std", 1.0),
        ).to(self.device)

        self.logger.info(f"Actor MLP: {self.model.actor}")
        self.logger.info(f"Critic MLP: {self.model.critic}")

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
            min_learning_rate=getattr(stage, "min_learning_rate", 1e-5),
            max_learning_rate=getattr(stage, "max_learning_rate", 1e-2),
            num_mini_batches=stage.num_mini_batches,
            num_learning_epochs=stage.num_learning_epochs,
        )

    def exploit(self, list_obs_data):
        """
        Exploit learned policy for action selection in evaluation mode.
        在评估模式下利用已学习的策略进行动作选择。
        """
        (obs) = list_obs_data
        with torch.no_grad():
            actions = self.algorithm.actor_critic.act_inference(obs)
            return [ActData(action=actions)]

    def learn(self, list_sample_data=None):
        """
        Trigger learning process using sample data.
        使用样本数据触发学习过程。

        Note: AlgorithmPPO.learn() doesn't take batch_data as argument anymore.
        It reads from its internal storage that was filled by workflow's run_episodes_.
        注：AlgorithmPPO.learn() 不再接受 batch_data 参数，
        而是直接读取 workflow 的 run_episodes_ 填充的内部存储。
        """
        return self.algorithm.learn()

    def predict(self, list_obs_data):
        """
        Generate predictions with actor-critic network.
        使用 actor-critic 网络生成预测。
        """
        (obs, critic_obs) = list_obs_data

        with torch.no_grad():
            hidden_states = None
            if getattr(self.algorithm.actor_critic, "is_recurrent", False):
                current_hidden = self.algorithm.actor_critic.get_hidden_states()
                if current_hidden is None or current_hidden[0].shape[1] != obs.shape[0]:
                    self.algorithm.actor_critic._init_hidden_states(obs.shape[0], obs.device, obs.dtype)
                    current_hidden = self.algorithm.actor_critic.get_hidden_states()
                hidden_states = tuple(state.detach().clone() for state in current_hidden)

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
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """
        Load model checkpoint.
        加载模型 checkpoint。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
            return

        pretrained = torch.load(model_file_path, map_location=self.device)
        current_state = self.model.state_dict()

        has_mismatch = False
        for key in pretrained:
            if key in current_state and pretrained[key].shape != current_state[key].shape:
                has_mismatch = True
                break

        if not has_mismatch:
            self.model.load_state_dict(pretrained)
            self.logger.info(f"load model {model_file_path} successfully (exact match)")
        else:
            self._load_model_partial(self.model, pretrained, model_file_path)

        self._enforce_action_std_bounds()
        self.cur_model_name = model_file_path

    def _enforce_action_std_bounds(self):
        min_std_cfg = getattr(self.stage, "min_normalized_std", None)
        max_std_cfg = getattr(self.stage, "max_normalized_std", None)
        if min_std_cfg is None and max_std_cfg is None:
            return

        def _cfg_tensor(cfg, ref_tensor):
            if cfg is None:
                return None
            candidate = torch.as_tensor(cfg, device=self.device, dtype=ref_tensor.dtype)
            return candidate if candidate.shape == ref_tensor.shape else None

        def _bound_std(std_tensor):
            current_std = std_tensor.data
            min_std = _cfg_tensor(min_std_cfg, current_std)
            max_std = _cfg_tensor(max_std_cfg, current_std)
            finite_cap = float(max_std.max().item()) if max_std is not None else 1.0e6
            bounded_std = torch.nan_to_num(current_std, nan=1.0, posinf=finite_cap, neginf=0.0)
            if min_std is not None:
                bounded_std = torch.maximum(bounded_std, min_std)
            if max_std is not None:
                bounded_std = torch.minimum(bounded_std, max_std)
            current_std.copy_(bounded_std)

        with torch.no_grad():
            if hasattr(self.model, "std"):
                _bound_std(self.model.std)
                self.logger.info(
                    f"[PPO] action std bounds enforced: min={min_std_cfg}, max={max_std_cfg}"
                )
            elif hasattr(self.model, "log_std"):
                log_std = torch.nan_to_num(
                    self.model.log_std.data,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if min_std_cfg is not None:
                    min_std = torch.tensor(min_std_cfg, device=self.device, dtype=log_std.dtype)
                    if min_std.shape == log_std.shape:
                        log_std = torch.maximum(log_std, torch.log(min_std))
                if max_std_cfg is not None:
                    max_std = torch.tensor(max_std_cfg, device=self.device, dtype=log_std.dtype)
                    if max_std.shape == log_std.shape:
                        log_std = torch.minimum(log_std, torch.log(max_std))
                self.model.log_std.data.copy_(log_std)
                self.logger.info(
                    f"[PPO] log action std bounds enforced: min={min_std_cfg}, max={max_std_cfg}"
                )

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

            if getattr(model, "is_recurrent", False) and key in {"actor.0.weight", "actor.0.bias"}:
                skipped_keys.append(f"{key} (recurrent front-end)")
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
                    if key == "actor.0.weight":
                        base_cols = self.stage.num_proprio_obs + self.stage.num_scan
                        slices = (
                            slice(0, min(old_param.shape[0], new_param.shape[0])),
                            slice(0, min(base_cols, old_param.shape[1], new_param.shape[1])),
                        )
                    elif key == "critic.0.weight":
                        base_cols = self.stage.num_critic_observations - getattr(self.stage, "num_goal_obs", 0)
                        slices = (
                            slice(0, min(old_param.shape[0], new_param.shape[0])),
                            slice(0, min(base_cols, old_param.shape[1], new_param.shape[1])),
                        )
                    new_param[slices] = old_param[slices]
                partial_keys.append(f"{key} {list(old_param.shape)}→{list(new_param.shape)}")

        model.load_state_dict(current_state)

        self.logger.info(
            f"Partial load model {model_file_path}: "
            f"{len(loaded_keys)} exact, {len(partial_keys)} partial, {len(skipped_keys)} skipped"
        )
        for info in partial_keys:
            self.logger.info(f"  Partial: {info}")
