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


_J8_LEVELS = tuple(range(10))
_J8_LEVEL_WEIGHTS = (0.06, 0.06, 0.07, 0.08, 0.10, 0.12, 0.14, 0.11, 0.12, 0.14)
_J8_FORBIDDEN_REWARDS = {
    "rough_energy",
    "energy_score_formula",
    "maze_anticipatory_turn",
    "long_non_foot_contact",
    "near_goal_finish_drive",
    "difficulty_pressure_complete",
}
_J8_NO_GATE_SWITCHES = (
    "phase_command_enabled",
    "worker_phase_command_enabled",
    "gate_speed_advice_enabled",
    "gate_reward_metrics_enabled",
    "gate_diagnostics_enabled",
    "terrain_phase_speed_enabled",
    "raw_terrain_gate_enabled",
)
_J9_TARGET_LR = 1.0e-5
_J9_J1_LEVEL_WEIGHTS = (
    0.08,
    0.09,
    0.09,
    0.10,
    0.12,
    0.14,
    0.14,
    0.08,
    0.08,
    0.08,
)


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


def _iter_env_candidates(env):
    """Yield wrapper and unwrapped environment objects without looping forever."""
    queue = [env]
    seen = set()
    while queue and len(seen) < 12:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        for attr in ("unwrapped", "env", "_gym_env"):
            try:
                nested = getattr(candidate, attr, None)
            except Exception:
                nested = None
            if nested is not None and id(nested) not in seen:
                queue.append(nested)


def _get_terrain_levels(env):
    """Return the live per-env terrain-level tensor through known wrappers."""
    field_names = ("terrain_levels", "terrain_level", "env_terrain_levels")
    for candidate in _iter_env_candidates(env):
        for attr in field_names:
            value = getattr(candidate, attr, None)
            if value is not None and hasattr(value, "shape"):
                return value

        scene = getattr(candidate, "scene", None)
        terrain = getattr(scene, "terrain", None) if scene is not None else None
        if terrain is not None:
            for attr in field_names:
                value = getattr(terrain, attr, None)
                if value is not None and hasattr(value, "shape"):
                    return value
    return None


def _get_manager(env, manager_name):
    for candidate in _iter_env_candidates(env):
        manager = getattr(candidate, manager_name, None)
        if manager is not None:
            return manager
    return None


def _terrain_level_counts(levels_tensor):
    if levels_tensor is None:
        return None
    levels_long = levels_tensor.to(torch.long).flatten()
    return torch.bincount(levels_long, minlength=len(_J8_LEVELS))[: len(_J8_LEVELS)]


def _log_terrain_level_histogram(env, logger):
    """Log the actual initial level allocation after environment reset."""
    try:
        counts = _terrain_level_counts(_get_terrain_levels(env))
        if counts is None:
            logger.info("[TerrainLevelHistogram] unable to resolve terrain levels")
            return
        hist = ", ".join(f"L{i}={int(counts[i])}" for i in _J8_LEVELS)
        logger.info(f"[TerrainLevelHistogram] {hist}")
    except Exception as exc:
        logger.info(f"[TerrainLevelHistogram] failed: {exc}")


def _validate_navj8_contract(usr_conf):
    """Fail fast if J8 drifts beyond its single level-distribution change."""
    if getattr(Config.CURRENT, "name", "") != "navj8":
        return

    errors = []
    terrain = usr_conf.get("terrain", {})
    level_mix = terrain.get("level_mix", {})
    commands = usr_conf.get("commands", {}).get("ranges", {})
    rewards = usr_conf.get("rewards", {})
    navigation = usr_conf.get("rl_navigation", {})
    env_conf = usr_conf.get("env", {})

    if level_mix.get("enabled") is not True:
        errors.append("terrain.level_mix.enabled must be true")
    if tuple(level_mix.get("levels", ())) != _J8_LEVELS:
        errors.append("terrain.level_mix.levels must be L0-L9")
    weights = tuple(float(value) for value in level_mix.get("weights", ()))
    if weights != _J8_LEVEL_WEIGHTS:
        errors.append(f"unexpected terrain.level_mix.weights={weights}")
    if abs(sum(weights) - 1.0) > 1.0e-9:
        errors.append(f"terrain.level_mix.weights sum to {sum(weights)}")
    if level_mix.get("pool_size") != 100:
        errors.append("terrain.level_mix.pool_size must be 100")
    if terrain.get("curriculum") is not False:
        errors.append("terrain.curriculum must be false")
    expected_num_envs = 1 if os.environ.get("KAIWU_TRAIN_TEST") else 3072
    if env_conf.get("num_envs") != expected_num_envs or float(env_conf.get("episode_length_s", 0.0)) != 120.0:
        errors.append(
            f"env must remain num_envs={expected_num_envs} and episode_length_s=120.0"
        )
    if commands.get("lin_vel_x") != [0.50, 0.64]:
        errors.append("commands.ranges.lin_vel_x must remain [0.50, 0.64]")
    if (
        commands.get("lin_vel_y") != [0.0, 0.0]
        or commands.get("ang_vel_yaw") != [0.0, 0.0]
    ):
        errors.append("lateral and yaw command ranges must remain zero")
    if float(rewards.get("termination", {}).get("weight", 0.0)) != -7.0:
        errors.append("rewards.termination.weight must remain -7.0")

    forbidden_active = sorted(_J8_FORBIDDEN_REWARDS.intersection(rewards))
    if forbidden_active:
        errors.append(f"forbidden J8 rewards configured: {forbidden_active}")
    enabled_gates = [name for name in _J8_NO_GATE_SWITCHES if navigation.get(name) is not False]
    if enabled_gates:
        errors.append(f"NoGate switches must remain false: {enabled_gates}")

    if errors:
        raise RuntimeError("[Stage=navj8] config contract failed: " + "; ".join(errors))


def _read_learning_rate(agent):
    optimizer = getattr(getattr(agent, "algorithm", None), "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", None)
    if param_groups:
        return param_groups[0].get("lr")
    return getattr(Config.CURRENT, "lr", None)


def _validate_navj9_contract(usr_conf, agent):
    """Fail fast unless J9 is a fixed-LR continuation of the J1 setup."""
    if getattr(Config.CURRENT, "name", "") != "navj9":
        return

    errors = []
    stage = Config.CURRENT
    algorithm = getattr(agent, "algorithm", None)
    terrain = usr_conf.get("terrain", {})
    level_mix = terrain.get("level_mix", {})
    commands = usr_conf.get("commands", {}).get("ranges", {})
    rewards = usr_conf.get("rewards", {})
    navigation = usr_conf.get("rl_navigation", {})

    expected_stage_values = {
        "lr": _J9_TARGET_LR,
        "min_learning_rate": _J9_TARGET_LR,
        "max_learning_rate": _J9_TARGET_LR,
        "entropy_coef": 0.001,
        "desired_kl": 0.0032,
        "init_noise_std": 0.95,
        "num_steps_per_env": 64,
        "num_learning_epochs": 3,
        "num_mini_batches": 4,
    }
    for name, expected in expected_stage_values.items():
        if getattr(stage, name, None) != expected:
            errors.append(f"stage.{name} must be {expected!r}")
    if getattr(stage, "schedule", None) != "fixed":
        errors.append("stage.schedule must be 'fixed'")

    if algorithm is None:
        errors.append("agent.algorithm is unavailable")
    else:
        expected_algorithm_values = {
            "learning_rate": _J9_TARGET_LR,
            "min_learning_rate": _J9_TARGET_LR,
            "max_learning_rate": _J9_TARGET_LR,
            "entropy_coef": 0.001,
            "desired_kl": 0.0032,
            "num_learning_epochs": 3,
            "num_mini_batches": 4,
        }
        for name, expected in expected_algorithm_values.items():
            if getattr(algorithm, name, None) != expected:
                errors.append(f"algorithm.{name} must be {expected!r}")
        if getattr(algorithm, "schedule", None) != "fixed":
            errors.append("algorithm.schedule must be 'fixed'")

    if tuple(level_mix.get("levels", ())) != _J8_LEVELS:
        errors.append("terrain.level_mix.levels must remain L0-L9")
    weights = tuple(float(value) for value in level_mix.get("weights", ()))
    if weights != _J9_J1_LEVEL_WEIGHTS:
        errors.append(f"terrain.level_mix.weights must remain J1 values, got {weights}")
    if terrain.get("curriculum") is not False:
        errors.append("terrain.curriculum must remain false")
    if commands.get("lin_vel_x") != [0.50, 0.64]:
        errors.append("commands.ranges.lin_vel_x must remain [0.50, 0.64]")
    if commands.get("lin_vel_y") != [0.0, 0.0] or commands.get("ang_vel_yaw") != [0.0, 0.0]:
        errors.append("lateral and yaw command ranges must remain zero")
    if float(rewards.get("termination", {}).get("weight", 0.0)) != -7.0:
        errors.append("rewards.termination.weight must remain -7.0")

    forbidden_active = sorted(_J8_FORBIDDEN_REWARDS.intersection(rewards))
    if forbidden_active:
        errors.append(f"non-J1 experimental rewards configured: {forbidden_active}")
    enabled_gates = [name for name in _J8_NO_GATE_SWITCHES if navigation.get(name) is not False]
    if enabled_gates:
        errors.append(f"NoGate switches must remain false: {enabled_gates}")

    optimizer = getattr(algorithm, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", ())
    optimizer_lrs = [float(group.get("lr", -1.0)) for group in param_groups]
    if not optimizer_lrs or any(abs(lr - _J9_TARGET_LR) > 1.0e-12 for lr in optimizer_lrs):
        errors.append(f"optimizer learning rates must all be {_J9_TARGET_LR}, got {optimizer_lrs}")

    if errors:
        raise RuntimeError("[Stage=navj9] config contract failed: " + "; ".join(errors))


def _log_navj9_startup_summary(agent, logger, usr_conf):
    if getattr(Config.CURRENT, "name", "") != "navj9":
        return

    stage = Config.CURRENT
    algorithm = agent.algorithm
    level_mix = usr_conf["terrain"]["level_mix"]
    expected_parent = getattr(stage, "parent_checkpoint", "unspecified")
    loaded_checkpoint = getattr(agent, "cur_model_name", None) or "platform-managed/unavailable"
    logger.info(
        "[StageStartup] "
        f"stage_name=navj9, expected_parent={expected_parent!r}, "
        f"loaded_checkpoint={loaded_checkpoint!r}"
    )
    logger.info(
        "[StageStartup] "
        f"learning_rate={_read_learning_rate(agent)}, schedule={algorithm.schedule}, "
        f"min_learning_rate={algorithm.min_learning_rate}, "
        f"max_learning_rate={algorithm.max_learning_rate}, "
        f"entropy_coef={algorithm.entropy_coef}, desired_kl={algorithm.desired_kl}"
    )
    logger.info(
        "[StageStartup] "
        f"level_mix={{'levels': {level_mix['levels']}, 'weights': {level_mix['weights']}, "
        f"'pool_size': {level_mix['pool_size']}}}"
    )


def _validate_navj9_runtime_lr(agent):
    """Keep the J9 experiment fixed-LR after every PPO update."""
    if getattr(Config.CURRENT, "name", "") != "navj9":
        return

    algorithm = agent.algorithm
    optimizer_lrs = [float(group["lr"]) for group in algorithm.optimizer.param_groups]
    if (
        abs(float(algorithm.learning_rate) - _J9_TARGET_LR) > 1.0e-12
        or any(abs(lr - _J9_TARGET_LR) > 1.0e-12 for lr in optimizer_lrs)
    ):
        raise RuntimeError(
            "[Stage=navj9] fixed learning rate drifted: "
            f"algorithm={algorithm.learning_rate}, optimizer={optimizer_lrs}"
        )


def _log_navj8_startup_summary(env, agent, logger, usr_conf):
    if getattr(Config.CURRENT, "name", "") != "navj8":
        return

    level_mix = usr_conf["terrain"]["level_mix"]
    rewards = usr_conf.get("rewards", {})
    reward_manager = _get_manager(env, "reward_manager")
    active_terms = getattr(reward_manager, "active_terms", None)
    try:
        rough_energy_active = None if active_terms is None else "rough_energy" in set(active_terms)
    except TypeError:
        rough_energy_active = None
    counts = _terrain_level_counts(_get_terrain_levels(env))
    actual_counts = None if counts is None else [int(value) for value in counts]
    termination_manager = _get_manager(env, "termination_manager")
    termination_terms = getattr(termination_manager, "active_terms", None)
    try:
        termination_terms = sorted(str(term) for term in termination_terms)
    except TypeError:
        termination_terms = None

    expected_parent = getattr(Config.CURRENT, "parent_checkpoint", "unspecified")
    loaded_checkpoint = getattr(agent, "cur_model_name", None) or "platform-managed/unavailable"
    termination_weight = rewards.get("termination", {}).get("weight")
    logger.info(
        "[StageStartup] "
        f"stage_name=navj8, expected_parent={expected_parent!r}, "
        f"loaded_checkpoint={loaded_checkpoint!r}, learning_rate={_read_learning_rate(agent)}"
    )
    logger.info(
        "[StageStartup] "
        f"level_mix={{'levels': {level_mix['levels']}, 'weights': {level_mix['weights']}, "
        f"'pool_size': {level_mix['pool_size']}}}, actual_level_counts={actual_counts}"
    )
    logger.info(
        "[StageStartup] "
        f"rough_energy_configured={'rough_energy' in rewards}, "
        f"rough_energy_active={rough_energy_active}, termination_weight={termination_weight}, "
        f"tracked_termination_terms={termination_terms}"
    )


def _new_level_outcome_stats():
    return {
        "episodes": None,
        "successes": None,
        "bad_orientation": None,
        "base_contact": None,
    }


def _get_termination_mask(env, term_name, device, num_envs):
    manager = _get_manager(env, "termination_manager")
    if manager is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    try:
        values = manager.get_term(term_name)
    except Exception:
        values = None
    if values is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    mask = torch.as_tensor(values, device=device).reshape(-1).bool()
    if mask.numel() >= num_envs:
        return mask[:num_envs]
    padded = torch.zeros(num_envs, dtype=torch.bool, device=device)
    padded[: mask.numel()] = mask
    return padded


def _update_level_outcome_stats(env, pre_step_levels, dones, level_stats):
    """Accumulate J8 outcomes using levels captured before auto-reset."""
    if level_stats is None or pre_step_levels is None:
        return

    dones_flat = dones.reshape(-1).bool()
    levels = pre_step_levels.to(device=dones_flat.device, dtype=torch.long).reshape(-1)
    num_envs = min(dones_flat.numel(), levels.numel())
    if num_envs == 0:
        return

    dones_flat = dones_flat[:num_envs]
    levels = levels[:num_envs]
    valid = dones_flat & (levels >= 0) & (levels < len(_J8_LEVELS))

    masks = {
        "episodes": valid,
        "successes": valid & _get_termination_mask(env, "goal_reached", dones_flat.device, num_envs),
        "bad_orientation": valid & _get_termination_mask(env, "bad_orientation", dones_flat.device, num_envs),
        "base_contact": valid & _get_termination_mask(env, "base_contact", dones_flat.device, num_envs),
    }
    for key, mask in masks.items():
        counts = torch.bincount(levels[mask], minlength=len(_J8_LEVELS))
        if level_stats[key] is None:
            level_stats[key] = torch.zeros_like(counts)
        level_stats[key] += counts


def _level_outcome_monitor_data(level_stats):
    if level_stats is None:
        return {}

    cpu_stats = {}
    for key, counts in level_stats.items():
        if counts is None:
            cpu_stats[key] = [0] * len(_J8_LEVELS)
        else:
            cpu_stats[key] = counts.detach().cpu().tolist()

    metrics = {}
    for level in _J8_LEVELS:
        episodes = cpu_stats["episodes"][level]
        successes = cpu_stats["successes"][level]
        metrics[f"level_episode_count_l{level}"] = episodes
        metrics[f"level_success_rate_l{level}"] = successes / episodes if episodes else 0.0
        metrics[f"level_bad_orientation_l{level}"] = cpu_stats["bad_orientation"][level]
        metrics[f"level_base_contact_l{level}"] = cpu_stats["base_contact"][level]
    return metrics


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
    level_outcome_stats = (
        _new_level_outcome_stats()
        if getattr(Config.CURRENT, "name", "") == "navj8"
        else None
    )

    # J8 is a single-variable distribution experiment. Validate and log it
    # after reset, when the actual terrain allocation is available.
    _validate_navj8_contract(usr_conf)
    _validate_navj9_contract(usr_conf, agent)
    _log_terrain_level_histogram(env, logger)
    _log_navj8_startup_summary(env, agent, logger, usr_conf)
    _log_navj9_startup_summary(agent, logger, usr_conf)

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
            level_outcome_stats,
        )

        episode += 1

        # Phase 2: Policy Update
        # 阶段2：策略更新
        agent.learn(list_sample_data=None)
        _validate_navj9_runtime_lr(agent)
        # Reset buffer pointer for next data collection
        # 重置 buffer 指针，为下一轮数据收集做准备
        storage.clear()
        total_cost_time = round(time.time() - start_time, 2)
        logger.info(f"Episode {episode} end, cost_time is {total_cost_time} s")

        # Phase 3: Monitoring Metrics Processing
        # 阶段3：监控指标处理
        now = time.time()
        if now - last_report_monitor_time >= 60:
            report_monitor_data(
                ep_infos,
                reward_keys,
                agent,
                monitor,
                episode,
                storage_stats,
                level_outcome_stats,
            )
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


def report_monitor_data(
    ep_infos,
    reward_keys,
    agent,
    monitor,
    episode,
    storage_stats=None,
    level_outcome_stats=None,
):
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

    monitor_data.update(_level_outcome_monitor_data(level_outcome_stats))
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
    level_outcome_stats=None,
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
            terrain_levels = _get_terrain_levels(env)
            pre_step_levels = (
                terrain_levels.detach().clone()
                if level_outcome_stats is not None and terrain_levels is not None
                else None
            )
            data = env.step(command_actions)
            frame_no, obs, critic_obs, rewards, dones, infos = _process_env_step_result(data, episode, logger)

            # Move tensors to device
            # 将张量移动到设备
            obs, critic_obs, rewards, dones = _move_tensors_to_device(obs, critic_obs, rewards, dones, agent.device)

            # Update episode statistics (always, regardless of decimation)
            # 更新 episode 统计（始终执行，不受降频影响）
            _update_level_outcome_stats(
                env,
                pre_step_levels,
                dones,
                level_outcome_stats,
            )
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
