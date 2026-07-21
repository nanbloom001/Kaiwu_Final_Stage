#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import RolloutStorage
from agent_ppo.feature.isaac_env_bridge import (
    sample_physics_stats,
    set_env_base_velocity_command,
)
from agent_ppo.feature.phase_command_adapter import (
    apply_track_phase_command,
    reset_track_phase_command_state,
)
from agent_ppo.feature.velocity_curriculum import VelocityCurriculum
from tools.train_env_conf_validate import check_usr_conf
from tools.utils import load_reward_keys_from_monitor_config
import torch
from collections import deque, defaultdict


def _initialize_training_state(env, agent, logger):
    """
    Initialize training state including storage, buffers, and observations.
    鍒濆鍖栬缁冪姸鎬侊紝鍖呮嫭瀛樺偍銆佺紦鍐插尯鍜岃娴嬨€?
    Returns:
        tuple: (storage, obs, critic_obs, ep_infos, rewbuffer, lenbuffer,
                cur_reward_sum, cur_episode_length, reward_keys, usr_conf)
        杩斿洖鍊硷細(storage, obs, critic_obs, ep_infos, rewbuffer, lenbuffer,
                cur_reward_sum, cur_episode_length, reward_keys, usr_conf)
    """
    usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(logger)

    terrain_mode = usr_conf.get("terrain", {}).get("mode", "standard")
    if terrain_mode == "standard":
        terrain_conf = usr_conf.get("terrain", {}).get("standard", {})
        terrain_keys = (
            "pyramid_slope",
            "pyramid_slope_inv",
            "pyramid_stairs",
            "pyramid_stairs_inv",
            "maze",
        )
        terrain_total = sum(float(terrain_conf.get(key, {}).get("proportion", 0.0)) for key in terrain_keys)
        if abs(terrain_total - 1.0) > 1e-6:
            message = (
                f"Invalid standard terrain proportions: sum={terrain_total:.6f}, expected 1.0. "
                f"Please check {usr_conf_file}."
            )
            logger.error(message)
            raise ValueError(message)

    # Validate configuration before proceeding

    valid, message = check_usr_conf(usr_conf, is_eval=False, logger=logger)
    if not valid:
        logger.error(message)
        raise Exception(message)

    # Set model to training mode

    # Initialize buffers and statistics
    agent.algorithm.actor_critic.train()
    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)
    cur_episode_length = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)

    # Use algorithm's internal storage (same object used by learn())
    # 浣跨敤绠楁硶鍐呴儴鐨?storage锛堜笌 learn() 浣跨敤鍚屼竴涓璞★級
    storage = agent.algorithm.storage

    # Reset environment and get initial observations
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
    # 浠?monitor 閰嶇疆鍔犺浇 reward_keys
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
    涓昏缁冨伐浣滄祦銆?    """
    agent = agents[0]
    env = envs[0]

    # Initialize training state
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

    # Velocity curriculum: expands command ranges independently of terrain curriculum.
    # terrain difficulty spans the full [0, 1.0] band via difficulty_range in TOML,
    # with 10 curriculum rows and initial placement capped at level 0;
    # velocity stages expand independently via VelocityCurriculum.
    # 閫熷害璇剧▼锛氱嫭绔嬩簬鍦板舰璇剧▼鎵╁ぇ閫熷害鎸囦护鑼冨洿銆?    # 鍦板舰闅惧害鐢?TOML difficulty_range=[0,1.0] + 10 涓绋嬫。浣嶇嫭绔嬮檺鍒讹紝
    # 鍒濆鏀剧疆绛夌骇涓婇檺涓?0锛涢€熷害鑼冨洿鐢?VelocityCurriculum 閫愰樁鎵╁ぇ銆?
    vel_curriculum = None
    if "velocity_curriculum" in usr_conf:
        vel_curriculum = VelocityCurriculum(logger, usr_conf)

    nav_conf = usr_conf.get("navigation", {})
    if bool(nav_conf.get("enabled", False)):
        logger.warning(
            "[Navigation] Ignored navigation.enabled=true because this PPO stage "
            "uses pure-RL maze navigation. No local planner will override commands."
        )

    # Main Training Loop
    while True:
        logger.info(f"Episode {episode} start, usr_conf is {usr_conf}")
        start_time = time.time()

        # Phase 1: Data Collection
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
            usr_conf,
        )

        episode += 1

        # Phase 1.5: Velocity Curriculum Check (performance-based, before ep_infos.clear)
        # 闃舵1.5锛氶€熷害璇剧▼妫€鏌ワ紙鎬ц兘椹卞姩锛屽繀椤诲湪 ep_infos.clear() 涔嬪墠璋冪敤锛?
        vel_reset = False
        if vel_curriculum is not None:
            last_obs, last_critic_obs, vel_reset = vel_curriculum.check_and_update(
                ep_infos, usr_conf, env, last_obs, last_critic_obs, rollout_stats=storage_stats
            )
        # If env.reset was triggered by a stage change, stale accumulated rewards
        # from interrupted episodes must be discarded to prevent corrupting rewbuffer.
        if vel_reset:
            cur_reward_sum.zero_()
            cur_episode_length.zero_()

        # Phase 2: Policy Update
        # 闃舵2锛氱瓥鐣ユ洿鏂?        # framework=True lets the framework directly call back to the business layer,
        # skipping the sample data guard.
        # framework=True 璁╂鏋跺眰鐩存帴鍥炶皟涓氬姟灞傦紝璺宠繃 sample data guard
        agent.learn(list_sample_data=None)
        # Reset buffer pointer for next data collection
        # 閲嶇疆 buffer 鎸囬拡锛屼负涓嬩竴杞暟鎹敹闆嗗仛鍑嗗
        storage.clear()
        total_cost_time = round(time.time() - start_time, 2)
        logger.info(f"Episode {episode} end, cost_time is {total_cost_time} s")

        # Phase 3: Monitoring Metrics Processing
        now = time.time()
        if now - last_report_monitor_time >= 60:
            report_monitor_data(ep_infos, reward_keys, agent, monitor, episode, storage_stats,
                                vel_stage=vel_curriculum.stage if vel_curriculum is not None else 0,
                                vel_tracking_ratio=vel_curriculum.last_tracking_ratio if vel_curriculum is not None else 0.0,
                                vel_tracking_reward=vel_curriculum.last_tracking_reward if vel_curriculum is not None else 0.0,
                                lenbuffer=lenbuffer, rewbuffer=rewbuffer)
            last_report_monitor_time = now

        ep_infos.clear()

        # Phase 4: Model Saving
        if episode % agent.save_interval == 0:
            agent.save_model()

    env.close()


def _extract_metric_value(ep_info, key, device):
    """Extract and convert metric value to tensor.

    鎻愬彇鎸囨爣鍊煎苟杞崲涓?tensor銆?    """
    if key not in ep_info:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    metric = ep_info[key]
    if not isinstance(metric, torch.Tensor):
        metric = torch.tensor(metric, device=device)
    return metric.float().mean()


def _aggregate_metrics(generic_metrics):
    """Aggregate metrics by computing mean values.

    閫氳繃璁＄畻鍧囧€兼眹鎬绘寚鏍囥€?    """
    aggregated = {}
    for metric_key, values in generic_metrics.items():
        if values:
            aggregated[metric_key] = torch.stack(values).mean().item()
        else:
            aggregated[metric_key] = 0.0
    return aggregated


def _collect_episode_metrics(ep_infos, reward_keys, device):
    """Collect metrics from episode infos.

    浠?episode info 涓敹闆嗘寚鏍囥€?    """
    generic_metrics = defaultdict(list)
    for ep_info in ep_infos:
        for key in reward_keys:
            metric_value = _extract_metric_value(ep_info, key, device)
            generic_metrics[key].append(metric_value)
    return _aggregate_metrics(generic_metrics)


def report_monitor_data(ep_infos, reward_keys, agent, monitor, episode, storage_stats=None,
                        vel_stage: int = 0, vel_tracking_ratio: float = 0.0,
                        vel_tracking_reward: float = 0.0, lenbuffer=None, rewbuffer=None):
    """
    Report monitoring data to monitor system.
    涓婃姤鐩戞帶鏁版嵁鍒扮洃鎺х郴缁熴€?    """
    monitor_data = {
        "episode_cnt": episode,
        "vel_curriculum_stage": vel_stage,
        "vel_curriculum_tracking_ratio": vel_tracking_ratio,
        "vel_curriculum_tracking_reward": vel_tracking_reward,
    }

    # Merge all storage stats: reward_mean/reward_std AND physics obs_ keys.
    if storage_stats:
        monitor_data.update(storage_stats)

    # Episode health metrics: episode length and cumulative reward per episode.
    if lenbuffer:
        monitor_data["mean_episode_length"] = float(sum(lenbuffer) / len(lenbuffer))
    if rewbuffer:
        monitor_data["mean_episode_reward"] = float(sum(rewbuffer) / len(rewbuffer))

    if ep_infos:
        metrics = _collect_episode_metrics(ep_infos, reward_keys, agent.device)
        # Do not let episode-level missing keys overwrite workflow-level metrics.
        # monitor_builder exposes non-episode metrics such as vel_curriculum_* and
        # obs_*; _collect_episode_metrics returns 0 for keys absent from ep_info.
        # If blindly updated, those valid workflow/storage values become flat 0
        # on the dashboard.
        for key, value in metrics.items():
            if key not in monitor_data:
                monitor_data[key] = value
        monitor_data["episode_reward"] = sum(monitor_data.get(key, 0) for key in reward_keys)

    monitor.put_data({os.getpid(): monitor_data})


def _process_env_step_result(data, episode, logger):
    """
    Process environment step result.
    澶勭悊鐜浜や簰缁撴灉銆?    """
    if data is None:
        error_message = "step failed, please check"
        logger.error(error_message)
        raise Exception(error_message)

    if not isinstance(data, (tuple, list)):
        raise TypeError(f"Unexpected env.step return type: {type(data).__name__}")

    if len(data) == 6:
        frame_no, obs, rewards, terminated, truncated, extra = data
        if isinstance(extra, (tuple, list)):
            if len(extra) < 2:
                raise ValueError(f"Unexpected env.step extra length: {len(extra)}")
            infos, privileged_obs = extra[0], extra[1]
        elif isinstance(extra, dict):
            infos = extra
            privileged_obs = extra.get("privileged_obs", extra.get("critic_obs", None))
        else:
            raise TypeError(f"Unexpected env.step extra type: {type(extra).__name__}")
    elif len(data) >= 7:
        frame_no, obs, rewards, terminated, truncated = data[:5]
        infos_or_extra = data[5]
        if isinstance(infos_or_extra, (tuple, list)) and len(infos_or_extra) >= 2:
            infos, privileged_obs = infos_or_extra[0], infos_or_extra[1]
        else:
            infos, privileged_obs = infos_or_extra, data[6]
    else:
        raise ValueError(f"Unexpected env.step return length: {len(data)}")

    if infos is None:
        infos = {}

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

    灏嗗紶閲忕Щ鍔ㄥ埌鎸囧畾璁惧銆?    """
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
    hidden_states=None,
    timeout_bootstrap_values=None,
):
    """
    Update transition with step data.
    浣跨敤姝ラ鏁版嵁鏇存柊 transition銆?    """
    transition.actions = actions
    transition.values = values
    transition.actions_log_prob = actions_log_prob
    transition.action_mean = action_mean
    transition.action_sigma = action_sigma
    transition.observations = obs
    transition.critic_observations = critic_obs
    transition.rewards = rewards.clone()
    transition.dones = dones
    transition.hidden_states = hidden_states

    # Bootstrapping on time outs
    # 澶勭悊 timeouts
    if "time_outs" in infos:
        bootstrap_values = (
            timeout_bootstrap_values
            if timeout_bootstrap_values is not None
            else transition.values
        )
        bootstrap_values = torch.nan_to_num(
            bootstrap_values.detach(), nan=0.0, posinf=0.0, neginf=0.0
        )
        timeout_mask = infos["time_outs"].unsqueeze(1).to(
            device=agent.device, dtype=bootstrap_values.dtype
        )
        transition.rewards += agent.algorithm.gamma * torch.squeeze(
            bootstrap_values * timeout_mask, 1
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

    鏇存柊 episode 缁熻鍜岀紦鍐插尯銆?    """
    if "episode" in infos:
        ep_infos.append(infos["episode"])

    cur_reward_sum += rewards
    cur_episode_length += 1

    new_ids = (dones > 0).nonzero(as_tuple=False)
    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

    cur_reward_sum[new_ids] = 0
    cur_episode_length[new_ids] = 0


def _compute_advantages_and_returns(
    storage,
    agent,
    obs,
    critic_obs,
    logger,
    env=None,
    usr_conf=None,
):
    """
    Compute advantage function and returns.
    璁＄畻浼樺娍鍑芥暟鍜屽洖鎶ャ€?    """
    last_obs = obs
    last_critic_obs = critic_obs
    if usr_conf is not None and last_obs is not None:
        last_obs, last_critic_obs, _ = _apply_phase_command_for_rollout(
            last_obs,
            last_critic_obs,
            env,
            usr_conf,
            logger,
            update_state=False,
        )

    value_obs = last_critic_obs if last_critic_obs is not None else last_obs
    last_values = agent.algorithm.actor_critic.evaluate(value_obs.detach()).detach()
    storage.compute_returns(last_values, agent.algorithm.gamma, agent.algorithm.lam)

    storage_stats = {
        "reward_mean": storage.rewards.mean().item(),
        "reward_std": storage.rewards.std().item(),
    }

    return storage_stats


def _sample_rollout_tracking_stats(storage, usr_conf, logger=None):
    """Estimate velocity-tracking curriculum metrics from rollout critic obs.

    This avoids waiting for completed episodes.  With long stable episodes,
    ``infos["episode"]`` may be absent for many PPO updates, so an episode-only
    curriculum can remain stuck at zero even though the policy is already
    tracking commands well.

    critic_obs layout:
      [0:3] base_lin_vel, [9:12] velocity command.
    """
    critic_obs = getattr(storage, "privileged_observations", None)
    if critic_obs is None or storage.step <= 0 or critic_obs.shape[-1] < 12:
        return {}

    reward_conf = usr_conf.get("rewards", {}).get("track_lin_vel_xy", {})
    params = reward_conf.get("params", {})
    weight = abs(float(reward_conf.get("weight", 1.0)))
    std = float(params.get("std", 0.25))
    if weight <= 1e-6:
        weight = 1.0
    if std <= 1e-6:
        if logger is not None:
            logger.warning(
                "[VelocityCurriculumDebug] invalid track_lin_vel_xy std in TOML; "
                f"std={std}, falling back to 0.25"
            )
        std = 0.25

    rollout_critic_obs = critic_obs[:storage.step]
    actual_xy = rollout_critic_obs[..., 0:2]
    command_xy = rollout_critic_obs[..., 9:11]
    squared_error = torch.sum(torch.square(actual_xy - command_xy), dim=-1)
    tracking_ratio = torch.exp(-squared_error / (std * std)).mean().item()
    tracking_reward = tracking_ratio * weight

    return {
        "rollout_track_lin_vel_xy_ratio": tracking_ratio,
        "rollout_track_lin_vel_xy_reward": tracking_reward,
    }

def _compute_timeout_bootstrap_values(obs, critic_obs, env, agent, infos, logger=None, usr_conf=None):
    """Evaluate V(s_{t+1}) for timeout bootstrapping using rollout-consistent obs."""
    if "time_outs" not in infos:
        return None

    timeouts = infos["time_outs"]
    if not torch.is_tensor(timeouts):
        timeouts = torch.as_tensor(timeouts, device=agent.device)
    else:
        timeouts = timeouts.to(agent.device)
    if not timeouts.bool().any():
        return None

    value_obs = obs
    value_critic_obs = critic_obs
    if usr_conf is not None:
        value_obs, value_critic_obs, _ = _apply_phase_command_for_rollout(
            value_obs,
            value_critic_obs,
            env,
            usr_conf,
            logger,
            update_state=False,
        )

    value_input = value_critic_obs if value_critic_obs is not None else value_obs
    return agent.algorithm.actor_critic.evaluate(value_input.detach()).detach()


def _phase_command_conf(usr_conf):
    return usr_conf.get("rl_navigation", {})


def _apply_phase_command_for_rollout(obs, critic_obs, env, usr_conf, logger, update_state=True):
    return apply_track_phase_command(
        obs,
        critic_obs,
        env,
        _phase_command_conf(usr_conf),
        logger,
        update_state=update_state,
        update_env_command=update_state,
        set_env_command_fn=set_env_base_velocity_command,
    )


def _record_phase_stats(nav_metric_values, phase_stats):
    for key, value in phase_stats.items():
        nav_metric_values[key].append(value.float().mean())


def _unpack_predict_result(predict_result):
    if len(predict_result) == 8:
        (
            actions,
            values,
            actions_log_prob,
            action_mean,
            action_sigma,
            detach_obs,
            detach_critic_obs,
            hidden_states,
        ) = predict_result
    elif len(predict_result) == 7:
        (
            actions,
            values,
            actions_log_prob,
            action_mean,
            action_sigma,
            detach_obs,
            detach_critic_obs,
        ) = predict_result
        hidden_states = None
    else:
        raise ValueError(f"Unexpected agent.predict return length: {len(predict_result)}")

    return (
        actions,
        values,
        actions_log_prob,
        action_mean,
        action_sigma,
        detach_obs,
        detach_critic_obs,
        hidden_states,
    )


def _clip_policy_actions(actions, agent, step_index, logger):
    command_actions = torch.clip(actions, -6.0, 6.0).to(agent.device)
    if step_index == 0:
        logger.info(
            "clipped_action summary: "
            f"min={float(command_actions.min().item()):.4f} "
            f"max={float(command_actions.max().item()):.4f} "
            f"mean={float(command_actions.mean().item()):.4f}"
        )
    return command_actions


def _aggregate_navigation_stats(nav_metric_values):
    aggregated = {}
    for key, values in nav_metric_values.items():
        if values:
            aggregated[key] = torch.stack(values).mean().item()
    return aggregated


def _collect_rollout_monitor_stats(storage, usr_conf, logger, nav_metric_values, env, critic_obs):
    storage_stats = _sample_rollout_tracking_stats(storage, usr_conf, logger)
    storage_stats.update(_aggregate_navigation_stats(nav_metric_values))
    # sample_physics_stats catches sensor-access errors internally, so it is safe
    # to keep this monitor hook active in local and platform runs.
    storage_stats.update(sample_physics_stats(env, logger, critic_obs=critic_obs))
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
    usr_conf,
):
    """
    Run episodes to collect trajectory data.
    杩愯 episodes 鏀堕泦杞ㄨ抗鏁版嵁銆?
    Returns:
        tuple: (last_obs, last_critic_obs, storage_stats)
        杩斿洖鍊硷細(last_obs, last_critic_obs, storage_stats)
    """
    transition = RolloutStorage.Transition()
    obs, critic_obs = last_obs, last_critic_obs
    nav_metric_values = defaultdict(list)

    # TODO: for hierarchical training, handle the mismatch between env action and
    # PPO storage action on your own.
    # TODO锛氬闇€鍒嗗眰璁粌锛岃嚜琛屽鐞?env action 涓?PPO storage action 涓嶄竴鑷寸殑闂銆?
    # Policy execution loop
    # 绛栫暐鎵ц寰幆
    with torch.inference_mode():
        for i in range(agent.num_steps_per_env):
            policy_obs, policy_critic_obs, phase_stats = _apply_phase_command_for_rollout(
                obs,
                critic_obs,
                env,
                usr_conf,
                logger,
            )
            _record_phase_stats(nav_metric_values, phase_stats)

            # Predict actions
            # 棰勬祴鍔ㄤ綔
            predict_data = (policy_obs, policy_critic_obs)
            (
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                detach_obs,
                detach_critic_obs,
                hidden_states,
            ) = _unpack_predict_result(agent.predict(predict_data))
            joint_actions = actions

            # Clip joint actions for env
            # 瑁佸壀鍏宠妭鍔ㄤ綔
            command_actions = _clip_policy_actions(joint_actions, agent, i, logger)

            # Environment interaction
            # 鐜浜や簰
            data = env.step(command_actions)
            frame_no, obs, critic_obs, rewards, dones, infos = _process_env_step_result(data, episode, logger)

            # Move tensors to device
            # 灏嗗紶閲忕Щ鍔ㄥ埌璁惧
            obs, critic_obs, rewards, dones = _move_tensors_to_device(obs, critic_obs, rewards, dones, agent.device)
            timeout_bootstrap_values = _compute_timeout_bootstrap_values(
                obs,
                critic_obs,
                env,
                agent,
                infos,
                logger,
                usr_conf=usr_conf,
            )
            reset_track_phase_command_state(env, dones)

            # Update episode statistics (always, regardless of decimation)
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
                hidden_states,
                timeout_bootstrap_values=timeout_bootstrap_values,
            )
            storage.add_transitions(transition)
            transition.clear()
            if hasattr(agent.algorithm.actor_critic, "reset"):
                agent.algorithm.actor_critic.reset(dones)

        # Compute advantages and returns
        storage_stats = _compute_advantages_and_returns(
            storage,
            agent,
            obs,
            critic_obs,
            logger,
            env=env,
            usr_conf=usr_conf,
        )
        storage_stats.update(
            _collect_rollout_monitor_stats(storage, usr_conf, logger, nav_metric_values, env, critic_obs)
        )
        last_obs = torch.clone(obs)

    # Note: batch generation now handled by AlgorithmPPO.learn()
    # Storage will be cleared after learning
    # 娉細batch 鐢熸垚宸茬敱 AlgorithmPPO.learn() 澶勭悊锛?    # storage 灏嗗湪璁粌瀹屾垚鍚庤娓呯┖銆?
    return last_obs, critic_obs, storage_stats
