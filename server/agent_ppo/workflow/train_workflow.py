#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import Frame
import os
import time
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import RolloutStorage
from tools.utils import load_reward_keys_from_monitor_config
import torch
from collections import deque, defaultdict


def _split_and_record_p15_transport(agent, critic_wire, dones=None):
    """Keep the 346-D worker wire out of Critic/PPO storage on every path."""
    if not getattr(agent, "is_p15_response", False):
        return critic_wire
    critic_obs, response_aux = agent.split_p15_transport(critic_wire)
    if dones is None:
        dones = torch.zeros(
            critic_obs.shape[0], dtype=torch.bool, device=critic_obs.device
        )
    agent.observe_response_aux(response_aux, dones)
    return critic_obs


def _initialize_training_state(env, agent, logger):
    """
    Initialize training state including storage, buffers, and observations.
    初始化训练状态，包括存储、缓冲区和观测。

    Returns:
        tuple: (storage, obs, critic_obs, ep_infos, rewbuffer, lenbuffer,
                cur_reward_sum, cur_episode_length, reward_keys, usr_conf)
        返回值：(storage, obs, critic_obs, ep_infos, rewbuffer, lenbuffer,
                cur_reward_sum, cur_episode_length, reward_keys, usr_conf)
    """
    usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(logger)

    # Validate configuration before proceeding
    # 在继续之前校验配置
    from tools.train_env_conf_validate import check_usr_conf

    valid, message = check_usr_conf(usr_conf, is_eval=False, logger=logger)
    if not valid:
        logger.error(message)
        raise Exception(message)

    # Set model to training mode
    # 设置模型为训练模式
    agent.algorithm.actor_critic.train()

    # Initialize buffers and statistics
    # 初始化缓冲区和统计信息
    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)
    cur_episode_length = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)

    # Use algorithm's internal storage (same object used by learn())
    # 使用算法内部的 storage（与 learn() 使用同一个对象）
    storage = agent.algorithm.storage

    # Reset environment and get initial observations
    # 重置环境并获取初始观测
    data = env.reset(usr_conf)
    if data is None:
        error_message = "reset failed, please check"
        logger.error(error_message)
        raise Exception(error_message)

    obs, critic_obs = data
    if critic_obs is None:
        critic_obs = obs
    obs = torch.clone(obs)
    critic_obs = torch.clone(critic_obs)
    critic_obs = _split_and_record_p15_transport(agent, critic_obs)
    logger.info(f"obs.shape:{obs.shape}, critic_obs.shape:{critic_obs.shape}")

    # Load reward keys from monitor config
    # 从 monitor 配置加载 reward_keys
    reward_keys = load_reward_keys_from_monitor_config()
    logger.info(f"reward_keys list is {reward_keys}")

    return (
        storage,
        obs,
        critic_obs,
        ep_infos,
        rewbuffer,
        lenbuffer,
        cur_reward_sum,
        cur_episode_length,
        reward_keys,
        usr_conf,
    )


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """
    Main training workflow.
    主训练工作流。
    """
    agent = agents[0]
    env = envs[0]
    logger.info(
        "[LifecycleProbe] train_workflow enter "
        f"pid={os.getpid()} stage={getattr(agent.stage, 'name', 'unknown')} "
        f"algorithm={getattr(agent, 'algorithm_name', 'unknown')} "
        f"flags=lbc:{getattr(agent, 'is_lbc', False)},"
        f"nav:{getattr(agent, 'is_nav_dagger', False)},"
        f"visual:{getattr(agent, 'is_visual_ppo', False)},"
        f"distill:{getattr(agent, 'is_behavior_distill', False)}"
    )

    # LBC 阶段：转发到 lbc_workflow（纯监督蒸馏，不走 PPO）
    # LBC stage: forward to lbc_workflow (pure supervised distillation)
    if getattr(agent, "is_lbc", False):
        from agent_ppo.workflow.lbc_workflow import workflow as lbc_workflow

        return lbc_workflow(envs, agents, logger=logger, monitor=monitor, *args, **kwargs)

    if getattr(agent, "is_p3_joint", False):
        from agent_ppo.workflow.p3_standard_joint_workflow import (
            workflow as p3_standard_joint_workflow,
        )

        return p3_standard_joint_workflow(
            envs, agents, logger=logger, monitor=monitor, *args, **kwargs
        )

    if getattr(agent, "is_p4_nav", False):
        from agent_ppo.workflow.p4_nav_ppo_workflow import (
            workflow as p4_nav_ppo_workflow,
        )

        return p4_nav_ppo_workflow(
            envs, agents, logger=logger, monitor=monitor, *args, **kwargs
        )

    if getattr(agent, "is_p2_nav", False):
        from agent_ppo.workflow.p2_nav_ppo_workflow import (
            workflow as p2_nav_ppo_workflow,
        )

        return p2_nav_ppo_workflow(
            envs, agents, logger=logger, monitor=monitor, *args, **kwargs
        )

    # hier-nav 高层 DAgger：转发到 nav_dagger_workflow（TBPTT 序列 BC，不走 PPO）
    # hier-nav high-level DAgger: forward to nav_dagger_workflow (TBPTT BC, no PPO)
    if getattr(agent, "is_nav_dagger", False):
        logger.info("[LifecycleProbe] train_workflow nav_import begin")
        from agent_ppo.workflow.nav_dagger_workflow import workflow as nav_dagger_workflow

        logger.info("[LifecycleProbe] train_workflow nav_import complete")
        logger.info("[LifecycleProbe] train_workflow dispatch=nav_dagger")
        return nav_dagger_workflow(envs, agents, logger=logger, monitor=monitor, *args, **kwargs)

    # Reference behavior distillation: flat standard teacher -> ActorCriticEncoder student
    # 参考模型行为蒸馏：扁平 standard teacher -> ActorCriticEncoder student
    if getattr(agent, "is_behavior_distill", False):
        from agent_ppo.workflow.behavior_distill_workflow import workflow as behavior_distill_workflow

        return behavior_distill_workflow(envs, agents, logger=logger, monitor=monitor, *args, **kwargs)

    if getattr(agent, "is_visual_ppo", False):
        from agent_ppo.workflow.visual_ppo_workflow import (
            workflow as visual_ppo_workflow,
        )

        return visual_ppo_workflow(
            envs, agents, logger=logger, monitor=monitor, *args, **kwargs
        )

    # Initialize training state
    # 初始化训练状态
    (
        storage,
        obs,
        critic_obs,
        ep_infos,
        rewbuffer,
        lenbuffer,
        cur_reward_sum,
        cur_episode_length,
        reward_keys,
        usr_conf,
    ) = _initialize_training_state(env, agent, logger)

    last_obs, last_critic_obs = torch.clone(obs), torch.clone(critic_obs)
    last_report_monitor_time = 0
    episode = 0

    # Main Training Loop
    # 主训练循环
    while True:
        logger.info(f"Episode {episode} start, usr_conf is {usr_conf}")
        start_time = time.time()

        # Phase 1: Data Collection
        # 阶段1：数据收集
        last_obs, last_critic_obs, storage_stats = run_episodes_(
            env,
            agent,
            storage,
            logger,
            last_obs,
            last_critic_obs,
            episode,
            ep_infos,
            cur_reward_sum,
            cur_episode_length,
            rewbuffer,
            lenbuffer,
        )

        episode += 1

        # Phase 2: Policy Update
        # 阶段2：策略更新
        agent.learn(list_sample_data=None)
        # Reset buffer pointer for next data collection
        # 重置 buffer 指针，为下一轮数据收集做准备
        storage.clear()
        total_cost_time = round(time.time() - start_time, 2)
        logger.info(f"Episode {episode} end, cost_time is {total_cost_time} s")

        # Phase 3: Monitoring Metrics Processing
        # 阶段3：监控指标处理
        now = time.time()
        if now - last_report_monitor_time >= 60:
            report_monitor_data(ep_infos, reward_keys, agent, monitor, episode, storage_stats)
            last_report_monitor_time = now

        ep_infos.clear()

        # Phase 4: Model Saving
        # 阶段4：模型保存
        if episode % agent.save_interval == 0:
            agent.save_model()

    env.close()


def _extract_metric_value(ep_info, key, device):
    """Extract and convert metric value to tensor.

    提取指标值并转换为 tensor。
    """
    if key not in ep_info:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    metric = ep_info[key]
    if not isinstance(metric, torch.Tensor):
        metric = torch.tensor(metric, device=device)
    return metric.float().mean()


def _aggregate_metrics(generic_metrics):
    """Aggregate metrics by computing mean values.

    通过计算均值汇总指标。
    """
    aggregated = {}
    for metric_key, values in generic_metrics.items():
        if values:
            aggregated[metric_key] = torch.stack(values).mean().item()
        else:
            aggregated[metric_key] = 0.0
    return aggregated


def _collect_episode_metrics(ep_infos, reward_keys, device):
    """Collect metrics from episode infos.

    从 episode info 中收集指标。
    """
    generic_metrics = defaultdict(list)
    for ep_info in ep_infos:
        for key in reward_keys:
            metric_value = _extract_metric_value(ep_info, key, device)
            generic_metrics[key].append(metric_value)
    return _aggregate_metrics(generic_metrics)


def report_monitor_data(ep_infos, reward_keys, agent, monitor, episode, storage_stats=None):
    """
    Report monitoring data to monitor system.
    上报监控数据到监控系统。
    """
    monitor_data = {"episode_cnt": episode}

    if storage_stats:
        monitor_data["reward_mean"] = storage_stats.get("reward_mean", 0.0)
        monitor_data["reward_std"] = storage_stats.get("reward_std", 0.0)

    if ep_infos:
        metrics = _collect_episode_metrics(ep_infos, reward_keys, agent.device)
        monitor_data.update(metrics)
        monitor_data["episode_reward"] = sum(monitor_data.get(key, 0) for key in reward_keys)

    monitor.put_data({os.getpid(): monitor_data})


def _process_env_step_result(data, episode, logger):
    """
    Process environment step result.
    处理环境交互结果。
    """
    if data is None:
        error_message = "step failed, please check"
        logger.error(error_message)
        raise Exception(error_message)

    frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs) = data

    if privileged_obs is not None:
        critic_obs = torch.clone(privileged_obs)
    else:
        critic_obs = torch.clone(obs)
    obs = torch.clone(obs)

    if obs is None:
        logger.error(f"episode {episode}, obs is None after processing!")
        raise Exception(f"episode {episode}, obs is None after processing!")

    dones = torch.logical_or(terminated, truncated)
    if not isinstance(infos, dict):
        infos = {}
    infos["hard_terminated"] = terminated
    return frame_no, obs, critic_obs, rewards, dones, infos


def _move_tensors_to_device(obs, critic_obs, rewards, dones, device):
    """Move tensors to specified device.

    将张量移动到指定设备。
    """
    return (
        obs.to(device),
        critic_obs.to(device),
        rewards.to(device),
        dones.to(device),
    )


def _update_transition_data(
    transition,
    actions,
    values,
    actions_log_prob,
    action_mean,
    action_sigma,
    obs,
    critic_obs,
    rewards,
    dones,
    infos,
    agent,
):
    """
    Update transition with step data.
    使用步骤数据更新 transition。
    """
    transition.actions = actions
    transition.values = values
    transition.actions_log_prob = actions_log_prob
    transition.action_mean = action_mean
    transition.action_sigma = action_sigma
    transition.observations = obs
    transition.critic_observations = critic_obs
    transition.rewards = rewards.clone()
    transition.dones = dones
    transition.hard_terminations = infos.get("hard_terminated", dones)
    if getattr(agent, "is_visual_ppo", False):
        transition.hidden_states = agent._last_rollout_hidden
        transition.anchor_actions = agent._last_anchor_action
        transition.anchor_latents = agent._last_anchor_latent

    # Bootstrapping on time outs
    # 处理 timeouts
    if "time_outs" in infos:
        transition.rewards += agent.algorithm.gamma * torch.squeeze(
            transition.values * infos["time_outs"].unsqueeze(1).to(agent.device), 1
        )


def _update_episode_statistics(
    dones,
    rewards,
    infos,
    cur_reward_sum,
    cur_episode_length,
    rewbuffer,
    lenbuffer,
    ep_infos,
):
    """Update episode statistics and buffers.

    更新 episode 统计和缓冲区。
    """
    if "episode" in infos:
        ep_infos.append(infos["episode"])

    cur_reward_sum += rewards
    cur_episode_length += 1

    new_ids = (dones > 0).nonzero(as_tuple=False)
    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

    cur_reward_sum[new_ids] = 0
    cur_episode_length[new_ids] = 0


def _compute_advantages_and_returns(storage, agent, critic_obs, logger):
    """
    Compute advantage function and returns.
    计算优势函数和回报。
    """
    last_critic_obs = torch.clone(critic_obs)
    last_values = agent.algorithm.actor_critic.evaluate(last_critic_obs.detach()).detach()
    storage.compute_returns(last_values, agent.algorithm.gamma, agent.algorithm.lam)

    storage_stats = {
        "reward_mean": storage.rewards.mean().item(),
        "reward_std": storage.rewards.std().item(),
    }

    return storage_stats


def run_episodes_(
    env,
    agent,
    storage,
    logger,
    last_obs,
    last_critic_obs,
    episode,
    ep_infos,
    cur_reward_sum,
    cur_episode_length,
    rewbuffer,
    lenbuffer,
):
    """
    Run episodes to collect trajectory data.
    运行 episodes 收集轨迹数据。

    Returns:
        tuple: (last_obs, last_critic_obs, storage_stats)
        返回值：(last_obs, last_critic_obs, storage_stats)
    """
    transition = RolloutStorage.Transition()
    obs, critic_obs = last_obs, last_critic_obs
    reset_mask = getattr(
        agent,
        "_rollout_reset_mask",
        torch.ones(obs.shape[0], dtype=torch.bool, device=agent.device),
    )
    zero_telemetry = None
    zero_step_dt_s = 0.02
    if getattr(agent, "is_visual_ppo", False):
        from agent_ppo.feature.zero_command_telemetry import ZeroCommandTelemetry

        usr_conf = getattr(agent, "usr_conf", {})
        if not isinstance(usr_conf, dict):
            usr_conf = {}
        stage_conf = usr_conf.get(getattr(agent.stage, "name", ""), {})
        if not isinstance(stage_conf, dict):
            stage_conf = {}
        reward_conf = stage_conf.get("rewards", {}).get(
            "zero_command_stability", {}
        )
        reward_params = reward_conf.get("params", {}) if isinstance(reward_conf, dict) else {}
        if not isinstance(reward_params, dict):
            reward_params = {}
        zero_step_dt_s = float(
            getattr(agent, "command_step_dt_s", 0.02)
        )
        existing = getattr(agent, "_zero_command_telemetry", None)
        if (
            existing is None
            or existing.num_envs != int(obs.shape[0])
            or existing.device != torch.device(agent.device)
        ):
            existing = ZeroCommandTelemetry(
                int(obs.shape[0]),
                agent.device,
                command_threshold=float(reward_params.get("command_threshold", 0.05)),
                grace_period_s=float(reward_params.get("grace_period_s", 0.4)),
            )
            agent._zero_command_telemetry = existing
        zero_telemetry = existing

    # TODO: for hierarchical training, handle the mismatch between env action and
    # PPO storage action on your own.
    # TODO：如需分层训练，自行处理 env action 与 PPO storage action 不一致的问题。

    # Policy execution loop
    # 策略执行循环
    with torch.inference_mode():
        for i in range(agent.num_steps_per_env):
            if getattr(agent, "is_visual_ppo", False):
                obs, critic_obs, anchor_weights, command_metrics = (
                    agent.prepare_rollout_step(
                        env, obs, critic_obs, reset_mask=reset_mask
                    )
                )
                agent._last_command_metrics = command_metrics
            # Predict actions
            # 预测动作
            predict_data = (obs, critic_obs)
            predict_result = agent.predict(predict_data)

            (
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                detach_obs,
                detach_critic_obs,
            ) = predict_result
            joint_actions = actions

            # Clip joint actions for env
            # 裁剪关节动作
            command_actions = torch.clip(joint_actions, -6.0, 6.0).to(agent.device)
            if i == 0:
                logger.info(f"clipped_action:{command_actions}")
            if zero_telemetry is not None:
                zero_telemetry.observe(
                    obs[:, 6:9],
                    command_actions,
                    reset_mask=reset_mask,
                    dt_s=zero_step_dt_s,
                )

            # Environment interaction
            # 环境交互
            data = env.step(command_actions)
            frame_no, obs, critic_obs, rewards, dones, infos = _process_env_step_result(data, episode, logger)

            # Move tensors to device
            # 将张量移动到设备
            obs, critic_obs, rewards, dones = _move_tensors_to_device(obs, critic_obs, rewards, dones, agent.device)
            critic_obs = _split_and_record_p15_transport(agent, critic_obs, dones)
            if getattr(agent, "is_visual_ppo", False):
                reset_mask = dones.reshape(-1).bool()
                agent._rollout_reset_mask = reset_mask.detach().clone()

            # Update episode statistics (always, regardless of decimation)
            # 更新 episode 统计（始终执行，不受降频影响）
            _update_episode_statistics(
                dones,
                rewards,
                infos,
                cur_reward_sum,
                cur_episode_length,
                rewbuffer,
                lenbuffer,
                ep_infos,
            )

            # Write transition to storage every step (flat PPO)
            # 每步写入 storage（扁平 PPO）
            _update_transition_data(
                transition,
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                detach_obs,
                detach_critic_obs,
                rewards,
                dones,
                infos,
                agent,
            )
            if getattr(agent, "is_visual_ppo", False):
                transition.anchor_weights = anchor_weights
            storage.add_transitions(transition)
            if getattr(agent, "is_visual_ppo", False):
                agent.algorithm.reset_recurrent_states(dones)
            transition.clear()

        # Compute advantages and returns
        # 计算优势函数和回报
        storage_stats = _compute_advantages_and_returns(storage, agent, critic_obs, logger)
        if zero_telemetry is not None:
            agent._last_zero_command_telemetry = zero_telemetry.metrics()
        last_obs = torch.clone(obs)

    # Note: batch generation now handled by AlgorithmPPO.learn()
    # Storage will be cleared after learning
    # 注：batch 生成已由 AlgorithmPPO.learn() 处理，
    # storage 将在训练完成后被清空。

    return last_obs, critic_obs, storage_stats
