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
from agent_ppo.feature.isaac_env_bridge import sample_physics_stats
from agent_ppo.feature.terrain_gate import worker_gate_monitor_stats
from tools.utils import load_reward_keys_from_monitor_config
import torch
from collections import deque, defaultdict


def _check_required_rewards(env, logger, usr_conf):
    """检查 toml 里声明的 reward 是否真正被环境激活。

    从 usr_conf 读 reward 名列表，从 env.reward_manager.active_terms 确认存在。
    部分阶段对核心 reward 做硬检查（缺失直接 raise），其余只 warning。
    """
    rewards_conf = usr_conf.get("rewards", {})
    if not rewards_conf:
        return

    required = set(rewards_conf.keys())

    stage_name = getattr(Config.CURRENT, "name", "")
    stage_required_rewards = {
        "navx7bridgeb": {
            "joint_acc",
            "dof_pos_limits",
        },
        "navx7bridgec": {
            "joint_acc",
            "dof_pos_limits",
            "hip_to_default",
            "joint_position_penalty",
            "dof_vel",
            "base_lateral_vel",
            "feet_slide",
            "feet_stumble",
        },
    }
    # J5 is a single-mechanism ablation: fail fast if P14 is not registered.
    stage_required_rewards["navj5"] = {"difficulty_pressure_complete"}
    # navx7bridged 继承 bridgec 的全部要求
    stage_required_rewards["navx7bridged"] = set(stage_required_rewards["navx7bridgec"])
    # navx7bridgee 继承 bridged + feet_clearance
    stage_required_rewards["navx7bridgee"] = set(stage_required_rewards["navx7bridged"]) | {"feet_clearance"}
    # navx7bridgef 继承 bridgee + feet_air_time + air_time_variance_penalty
    stage_required_rewards["navx7bridgef"] = set(stage_required_rewards["navx7bridgee"]) | {
        "feet_air_time",
        "air_time_variance_penalty",
    }
    # navx7bridgeg 继承 bridgef + feet_swing_forward
    stage_required_rewards["navx7bridgeg"] = set(stage_required_rewards["navx7bridgef"]) | {
        "feet_swing_forward",
    }
    # navx7bridgeh 继承 bridgeg（同样12项，只改权重不新增reward）
    stage_required_rewards["navx7bridgeh"] = set(stage_required_rewards["navx7bridgeg"])
    # navx7score1 继承 bridgeg + score_guidance
    stage_required_rewards["navx7score1"] = set(stage_required_rewards["navx7bridgeg"]) | {"score_guidance"}
    # navx7nav1 继承 score1 + 7项导航/墙体reward
    stage_required_rewards["navx7nav1"] = set(stage_required_rewards["navx7score1"]) | {
        "forward_heading_velocity",
        "goal_distance",
        "reach_goal",
        "wall_collision",
        "wall_stall_penalty",
        "wall_proximity",
        "stuck_penalty",
    }
    # navx7train1 继承 nav1（不新增reward，只改level_mix/速度/PPO）
    stage_required_rewards["navx7train1"] = set(stage_required_rewards["navx7nav1"])
    # navx7nogate 继承 train1（同样reward，只关门控+统一速度）
    stage_required_rewards["navx7nogate"] = set(stage_required_rewards["navx7train1"])
    hard_required = stage_required_rewards.get(stage_name, set())

    active = set()
    reward_manager = getattr(env, "reward_manager", None)

    if reward_manager is not None:
        terms = getattr(reward_manager, "active_terms", None)
        if terms is not None:
            try:
                active = set(terms)
            except TypeError:
                active = {str(term) for term in terms}

    if not active:
        logger.warning(
            f"[RewardCheck] reward_manager.active_terms unavailable; "
            f"cannot verify Stage={stage_name}, required={sorted(hard_required)}"
        )
        return

    missing_hard = hard_required - active
    if missing_hard:
        raise RuntimeError(
            f"[Stage={stage_name}] required rewards are not active: "
            f"{sorted(missing_hard)}"
        )

    missing_all = required - active
    if missing_all:
        logger.warning(
            f"[RewardCheck] other reward(s) not active: "
            f"{sorted(missing_all)}"
        )
    else:
        logger.info(
            f"[RewardCheck] all {len(required)} rewards active"
        )


def _log_terrain_level_histogram(env, logger):
    """打印初始地形难度直方图，确认 level_mix 是否真正生效。

    尝试从多个可能的字段名读取地形等级张量。
    """
    import torch as _torch

    levels_tensor = None
    field_names = [
        "terrain_levels",
        "terrain_level",
        "env_terrain_levels",
    ]
    for attr in field_names:
        val = getattr(env, attr, None)
        if val is not None and hasattr(val, "shape"):
            levels_tensor = val
            break

    if levels_tensor is None:
        # 尝试从 scene/robot 取
        scene = getattr(env, "scene", None)
        if scene is not None:
            for attr in field_names:
                val = getattr(scene, attr, None)
                if val is not None and hasattr(val, "shape"):
                    levels_tensor = val
                    break

    if levels_tensor is None:
        logger.info("[TerrainLevelHistogram] 无法获取地形等级张量，跳过直方图")
        return

    try:
        levels_long = levels_tensor.to(_torch.long).flatten()
        counts = _torch.bincount(levels_long, minlength=10)
        hist = ", ".join(f"L{i}={int(counts[i])}" for i in range(min(10, len(counts))))
        logger.info(f"[TerrainLevelHistogram] {hist}")
    except Exception as exc:
        logger.info(f"[TerrainLevelHistogram] 统计失败: {exc}")


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

    # LBC 阶段：转发到 lbc_workflow（纯监督蒸馏，不走 PPO）
    # LBC stage: forward to lbc_workflow (pure supervised distillation)
    if getattr(agent, "is_lbc", False):
        from agent_ppo.workflow.lbc_workflow import workflow as lbc_workflow

        return lbc_workflow(envs, agents, logger=logger, monitor=monitor, *args, **kwargs)

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

    # 打印初始地形难度直方图（确认 level_mix 是否真正生效）
    _log_terrain_level_histogram(env, logger)

    # 检查 Stage3E-1 需要的 reward 是否真正激活
    _check_required_rewards(env, logger, usr_conf)

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


def _compute_advantages_and_returns(storage, agent, critic_obs, logger, env=None):
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

    # 采集命令诊断和物理量监控（Stage4A）
    try:
        storage_stats.update(sample_physics_stats(env, logger=None, critic_obs=critic_obs))
    except Exception:
        pass
    try:
        gate_stats = worker_gate_monitor_stats(env)
        if gate_stats:
            storage_stats.update(gate_stats)
    except Exception:
        pass

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

    # TODO: for hierarchical training, handle the mismatch between env action and
    # PPO storage action on your own.
    # TODO：如需分层训练，自行处理 env action 与 PPO storage action 不一致的问题。

    # Policy execution loop
    # 策略执行循环
    with torch.inference_mode():
        for i in range(agent.num_steps_per_env):
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

            # Environment interaction
            # 环境交互
            data = env.step(command_actions)
            frame_no, obs, critic_obs, rewards, dones, infos = _process_env_step_result(data, episode, logger)

            # Move tensors to device
            # 将张量移动到设备
            obs, critic_obs, rewards, dones = _move_tensors_to_device(obs, critic_obs, rewards, dones, agent.device)

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
            storage.add_transitions(transition)
            transition.clear()

        # Compute advantages and returns
        # 计算优势函数和回报
        storage_stats = _compute_advantages_and_returns(storage, agent, critic_obs, logger, env)
        last_obs = torch.clone(obs)

    # Note: batch generation now handled by AlgorithmPPO.learn()
    # Storage will be cleared after learning
    # 注：batch 生成已由 AlgorithmPPO.learn() 处理，
    # storage 将在训练完成后被清空。

    return last_obs, critic_obs, storage_stats
