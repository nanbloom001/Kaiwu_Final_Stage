#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
from typing import Optional, Tuple
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import RolloutStorage
from tools.train_env_conf_validate import check_usr_conf
import torch
from collections import deque, defaultdict


# R1 uses a compact dashboard vocabulary. The raw episode keys remain tied to
# the environment's reward names; only their monitor aliases are shortened.
_R1_REWARD_KEYS = {
    "reward_track_lin_vel_xy": "rew_vx",
    "reward_track_ang_vel_z": "rew_yaw",
    "reward_flat_orientation": "rew_flat",
    "reward_undesired_contacts": "rew_contact",
    "reward_termination": "rew_term",
    "reward_energy": "rew_energy",
    "reward_joint_torques": "rew_torque",
    "reward_action_rate": "rew_action",
}

_R1_PHYSICS_KEYS = {
    "obs_cmd_vel_x": "vx_cmd",
    "obs_actual_vel_x": "vx_real",
    "obs_lin_vel_x_error": "vx_err",
    "obs_base_height": "body_h",
    "obs_ang_vel_xy": "tilt_rate",
    "terrain_level": "ter_level",
    "terrain_level_max": "ter_max",
    "terrain_level_min": "ter_min",
    "terrain_level_up": "ter_up",
    "terrain_level_down": "ter_down",
}


class VelocityStageCurriculum:
    """Performance-based velocity curriculum for command range expansion.

    Promotes to next velocity stage when mean reward_track_lin_vel_xy >
    promote_threshold for promote_count consecutive episode-batch checks.
    Demotes to previous stage when it falls below demote_threshold for
    demote_count consecutive checks. Neither counter changes when the metric
    stays in the neutral zone [demote_threshold, promote_threshold).

    It is independent of terrain.curriculum. Terrain difficulty is still
    managed by the simulator configuration, while this helper only changes
    command sampling ranges by calling env.reset(usr_conf).

    Configuration is loaded from usr_conf["velocity_curriculum"] (TOML section
    [velocity_curriculum]), so all thresholds and stage definitions live in the
    TOML file rather than being hard-coded here.

    The terrain curriculum and the velocity curriculum are intentionally
    independent. Terrain difficulty is controlled by the simulator config,
    while this helper only changes command sampling ranges.
    """

    # Default stages used only when usr_conf has no [velocity_curriculum] section.
    _DEFAULT_STAGES = [
        {"lin_vel_x": [0.0,  0.5], "lin_vel_y": [-0.3,  0.3], "ang_vel_yaw": [-1.0,  1.0]},
        {"lin_vel_x": [0.0,  0.7], "lin_vel_y": [-0.4,  0.4], "ang_vel_yaw": [-1.2,  1.2]},
        {"lin_vel_x": [0.0,  0.85], "lin_vel_y": [-0.5,  0.5], "ang_vel_yaw": [-1.4,  1.4]},
        {"lin_vel_x": [0.0,  1.0], "lin_vel_y": [-0.6,  0.6], "ang_vel_yaw": [-1.5,  1.5]},
    ]

    # ep_info key used as performance signal.  Different framework versions may
    # expose reward terms with slightly different names, so lookup is tolerant.
    _TRACKING_KEY = "reward_track_lin_vel_xy"
    _TRACKING_KEY_CANDIDATES = (
        "reward_track_lin_vel_xy",
        "track_lin_vel_xy",
        "Episode_Reward/reward_track_lin_vel_xy",
        "Episode_Reward/track_lin_vel_xy",
    )

    def __init__(self, logger, usr_conf: dict):
        """Build curriculum from usr_conf["velocity_curriculum"] (TOML section).

        Falls back to _DEFAULT_STAGES / hard-coded thresholds if the section is absent.
        """
        self.logger = logger
        vc_conf = usr_conf.get("velocity_curriculum", {})
        tracking_reward_conf = usr_conf.get("rewards", {}).get("track_lin_vel_xy", {})

        self._tracking_reward_weight = abs(float(tracking_reward_conf.get("weight", 1.0)))
        if self._tracking_reward_weight <= 1e-6:
            logger.warning(
                "[VelocityCurriculum] track_lin_vel_xy weight is non-positive; "
                "falling back to 1.0 for curriculum normalization. "
                "Please check [rewards.track_lin_vel_xy].weight in TOML."
            )
            self._tracking_reward_weight = 1.0

        self.promote_threshold: float = self._normalize_tracking_threshold(
            float(vc_conf.get("promote_threshold", 0.64)), "promote_threshold"
        )
        self.demote_threshold:  float = self._normalize_tracking_threshold(
            float(vc_conf.get("demote_threshold", 0.32)), "demote_threshold"
        )
        self.promote_count:     int   = int(vc_conf.get("promote_count", 5))
        self.demote_count:      int   = int(vc_conf.get("demote_count",  3))

        raw_stages = vc_conf.get("stages", None)
        if raw_stages:
            self.STAGES = [
                {
                    "lin_vel_x":   list(s["lin_vel_x"]),
                    "lin_vel_y":   list(s["lin_vel_y"]),
                    "ang_vel_yaw": list(s["ang_vel_yaw"]),
                }
                for s in raw_stages
            ]
        else:
            self.STAGES = self._DEFAULT_STAGES
            logger.warning(
                "[VelocityCurriculum] No [velocity_curriculum.stages] found in usr_conf; "
                "falling back to hard-coded default stages."
            )

        command_ranges = usr_conf.get("commands", {}).get("ranges", {})
        stage0 = self.STAGES[0]
        if (
            list(command_ranges.get("lin_vel_x", [])) != stage0["lin_vel_x"]
            or list(command_ranges.get("lin_vel_y", [])) != stage0["lin_vel_y"]
            or list(command_ranges.get("ang_vel_yaw", [])) != stage0["ang_vel_yaw"]
        ):
            raise ValueError(
                "Velocity curriculum Stage 0 must exactly match [commands.ranges] in TOML. "
                f"Got commands.ranges={command_ranges}, stage0={stage0}."
            )

        self._stage_idx = 0
        self._promote_streak = 0  # consecutive checks above promote_threshold
        self._demote_streak = 0   # consecutive checks below demote_threshold
        self._last_mean_tracking_reward = 0.0
        self._last_mean_tracking_ratio = 0.0
        self._tracking_key_resolved = None
        self._tracking_key_warning_logged = False
        self._debug_check_count = 0
        self._last_transition = 0
        self._promote_total = 0
        self._demote_total = 0
        logger.info(
            f"[VelocityCurriculum] Initialized: {len(self.STAGES)} stages, "
            f"tracking_weight={self._tracking_reward_weight}, "
            f"promote_threshold={self.promote_threshold}, demote_threshold={self.demote_threshold}, "
            f"promote_count={self.promote_count}, demote_count={self.demote_count}"
        )

    def _normalize_tracking_threshold(self, threshold_value: float, field_name: str) -> float:
        """Normalize legacy absolute thresholds into reward-ratio thresholds.

        Historical configs stored absolute weighted reward thresholds (e.g. 1.6).
        That makes curriculum behavior drift every time reward weight changes.
        New configs should store ratios in [0, 1], e.g. 0.55 means 55% of the
        current track_lin_vel_xy maximum reward.
        """
        if threshold_value <= 1.0:
            return threshold_value

        normalized = threshold_value / self._tracking_reward_weight
        self.logger.warning(
            f"[VelocityCurriculum] {field_name}={threshold_value} detected as legacy absolute reward; "
            f"normalized to ratio {normalized:.3f} using tracking weight {self._tracking_reward_weight:.3f}."
        )
        return normalized

    @property
    def stage(self) -> int:
        return self._stage_idx

    @property
    def last_tracking_reward(self) -> float:
        return self._last_mean_tracking_reward

    @property
    def last_tracking_ratio(self) -> float:
        return self._last_mean_tracking_ratio

    @property
    def last_transition(self) -> int:
        """Latest transition: 1=promote, -1=demote, 0=no stage change."""
        return self._last_transition

    @property
    def promote_streak(self) -> int:
        return self._promote_streak

    @property
    def demote_streak(self) -> int:
        return self._demote_streak

    @property
    def promote_total(self) -> int:
        return self._promote_total

    @property
    def demote_total(self) -> int:
        return self._demote_total

    @property
    def active_max_vx(self) -> float:
        return float(self.STAGES[self._stage_idx]["lin_vel_x"][1])

    def _resolve_tracking_metric_entry(self, ep_info):
        """Resolve the tracking metric value and the concrete key used to find it."""
        if self._tracking_key_resolved and self._tracking_key_resolved in ep_info:
            return ep_info[self._tracking_key_resolved], self._tracking_key_resolved

        for key in self._TRACKING_KEY_CANDIDATES:
            if key in ep_info:
                self._tracking_key_resolved = key
                return ep_info[key], key

        for key, value in ep_info.items():
            normalized = str(key).replace("/", "_")
            if normalized.endswith("track_lin_vel_xy"):
                self._tracking_key_resolved = key
                return value, key

        return None, None

    def _summarize_tracking_reward(self, ep_infos) -> Tuple[Optional[float], Optional[float]]:
        """Average reward_track_lin_vel_xy across completed episodes.

        Returns the weighted reward mean and the normalized reward ratio.
        """
        values = []
        for ep_info in ep_infos:
            v, key = self._resolve_tracking_metric_entry(ep_info)
            if key is None:
                continue
            values.append(v.float().mean().item() if isinstance(v, torch.Tensor) else float(v))
        if not values:
            if ep_infos and not self._tracking_key_warning_logged:
                sample_keys = list(ep_infos[0].keys())
                self.logger.warning(
                    "[VelocityCurriculum] Cannot find tracking metric for velocity curriculum. "
                    f"Tried keys={self._TRACKING_KEY_CANDIDATES}; "
                    f"sample episode keys={sample_keys[:40]}"
                )
                self._tracking_key_warning_logged = True
            return None, None

        mean_reward = sum(values) / len(values)
        mean_ratio = mean_reward / self._tracking_reward_weight
        return mean_reward, mean_ratio

    def _apply_active_velocity_stage(self, usr_conf, env, obs, critic_obs):
        """Apply the active velocity stage, reset the env, and return fresh observations."""
        cfg = self.STAGES[self._stage_idx]
        usr_conf["commands"]["ranges"]["lin_vel_x"]   = cfg["lin_vel_x"]
        usr_conf["commands"]["ranges"]["lin_vel_y"]   = cfg["lin_vel_y"]
        usr_conf["commands"]["ranges"]["ang_vel_yaw"] = cfg["ang_vel_yaw"]
        self.logger.info(
            f"[VelocityCurriculum] Stage {self._stage_idx}: "
            f"lin_vel_x={cfg['lin_vel_x']}, lin_vel_y={cfg['lin_vel_y']}, "
            f"ang_vel_yaw={cfg['ang_vel_yaw']} calling env.reset"
        )
        data = env.reset(usr_conf)
        if data is None:
            self.logger.error("[VelocityCurriculum] env.reset failed after stage change!")
            raise RuntimeError("VelocityCurriculum env.reset failed after stage change")
        new_obs, new_critic_obs = data
        if new_critic_obs is None:
            new_critic_obs = new_obs
        return torch.clone(new_obs), torch.clone(new_critic_obs), True

    def evaluate_and_update_stage(self, ep_infos, usr_conf, env, obs, critic_obs, rollout_stats=None):
        """Promote or demote the active velocity stage when tracking metrics warrant it.

        This must run before ep_infos is cleared so the current rollout still
        has access to finished-episode reward summaries.
        """
        self._debug_check_count += 1
        self._last_transition = 0
        rollout_stats = rollout_stats or {}
        mean_reward, mean_ratio = self._summarize_tracking_reward(ep_infos)
        metric_source = "episode"
        if mean_reward is None or mean_ratio is None:
            rollout_reward = rollout_stats.get("rollout_track_lin_vel_xy_reward")
            rollout_ratio = rollout_stats.get("rollout_track_lin_vel_xy_ratio")
            if rollout_reward is not None and rollout_ratio is not None:
                mean_reward = float(rollout_reward)
                mean_ratio = float(rollout_ratio)
                metric_source = "rollout_critic_obs"

        if mean_reward is None or mean_ratio is None:
            if self._debug_check_count <= 10 or self._debug_check_count % 20 == 0:
                sample_keys = list(ep_infos[0].keys())[:40] if ep_infos else []
                self.logger.warning(
                    "[VelocityCurriculumDebug] no tracking metric available; "
                    f"check={self._debug_check_count}, ep_infos={len(ep_infos)}, "
                    f"resolved_key={self._tracking_key_resolved}, sample_keys={sample_keys}, "
                    f"rollout_stats_keys={list(rollout_stats.keys())}"
                )
            return obs, critic_obs, False

        self._last_mean_tracking_reward = mean_reward
        self._last_mean_tracking_ratio = mean_ratio

        stage_changed = False
        old_stage = self._stage_idx

        if mean_ratio >= self.promote_threshold:
            self._promote_streak += 1
            self._demote_streak = 0
            if self._promote_streak >= self.promote_count and self._stage_idx < len(self.STAGES) - 1:
                self._stage_idx += 1
                self._promote_streak = 0
                self._last_transition = 1
                self._promote_total += 1
                self.logger.warning(
                    f"[VelocityCurriculum] PROMOTE stage {self._stage_idx} "
                    f"(tracking_ratio={mean_ratio:.3f} >= {self.promote_threshold:.3f}, "
                    f"tracking_reward={mean_reward:.3f}, "
                    f"for {self.promote_count} consecutive checks)"
                )
                stage_changed = True
        elif mean_ratio < self.demote_threshold:
            self._demote_streak += 1
            self._promote_streak = 0
            if self._demote_streak >= self.demote_count and self._stage_idx > 0:
                self._stage_idx -= 1
                self._demote_streak = 0
                self._last_transition = -1
                self._demote_total += 1
                self.logger.warning(
                    f"[VelocityCurriculum] DEMOTE stage {self._stage_idx} "
                    f"(tracking_ratio={mean_ratio:.3f} < {self.demote_threshold:.3f}, "
                    f"tracking_reward={mean_reward:.3f}, "
                    f"for {self.demote_count} consecutive checks)"
                )
                stage_changed = True
        else:
            # Neutral zone: slowly decay both streaks to avoid oscillation
            self._promote_streak = max(0, self._promote_streak - 1)
            self._demote_streak = max(0, self._demote_streak - 1)

        if self._debug_check_count <= 10 or self._debug_check_count % 20 == 0 or stage_changed:
            current_cfg = self.STAGES[self._stage_idx]
            self.logger.warning(
                "[VelocityCurriculumDebug] "
                f"check={self._debug_check_count}, ep_infos={len(ep_infos)}, "
                f"source={metric_source}, key={self._tracking_key_resolved}, "
                f"reward={mean_reward:.4f}, "
                f"ratio={mean_ratio:.4f}, stage={old_stage}->{self._stage_idx}, "
                f"promote={self._promote_streak}/{self.promote_count} "
                f"@{self.promote_threshold:.3f}, "
                f"demote={self._demote_streak}/{self.demote_count} "
                f"@{self.demote_threshold:.3f}, "
                f"ranges={current_cfg}"
            )

        if stage_changed:
            return self._apply_active_velocity_stage(usr_conf, env, obs, critic_obs)
        return obs, critic_obs, False


def _initialize_training_runtime_state(env, agent, logger):
    """Initialize rollout storage, episode buffers, and the first observations."""
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

    # J9 fixed-schedule startup validation
    _stage = stage
    if getattr(_stage, "schedule", "adaptive") == "fixed":
        _expected_lr = float(_stage.lr)
        _algo_lr = agent.algorithm.learning_rate
        if abs(_algo_lr - _expected_lr) > 1e-12:
            raise RuntimeError(f"J9 LR mismatch: algorithm={_algo_lr}, expected={_expected_lr}")
        for _pg in agent.algorithm.optimizer.param_groups:
            if abs(_pg["lr"] - _expected_lr) > 1e-12:
                raise RuntimeError(f"J9 LR mismatch: optimizer={_pg['lr']}, expected={_expected_lr}")
        logger.info(f"[J9] Fixed-schedule validated: lr={_expected_lr}, schedule={_stage.schedule}")

    # Initialize buffers and statistics
    agent.algorithm.actor_critic.train()
    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)
    cur_episode_length = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)

    # The workflow and the PPO learner operate on the same storage object.
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

    # Raw reward keys are mapped to compact dashboard names at report time.
    reward_keys = list(_R1_REWARD_KEYS)
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
    """Main PPO rollout/update loop used by the framework workflow loader."""
    agent = agents[0]
    env = envs[0]

    if getattr(agent, "is_lbc", False):
        from agent_ppo.workflow.lbc_workflow import workflow as lbc_workflow

        return lbc_workflow(
            envs, agents, logger=logger, monitor=monitor, *args, **kwargs
        )

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
    ) = _initialize_training_runtime_state(env, agent, logger)

    last_obs, last_critic_obs = torch.clone(obs), torch.clone(critic_obs)
    last_report_monitor_time = 0
    episode = 0

    # Velocity command expansion is separate from terrain progression so we can
    # tune speed difficulty without changing the terrain schedule.
    vel_curriculum = None
    if "velocity_curriculum" in usr_conf:
        vel_curriculum = VelocityStageCurriculum(logger, usr_conf)

    nav_controller = None
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
        last_obs, last_critic_obs, storage_stats = collect_rollout_batch(
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
            nav_controller=nav_controller,
        )

        episode += 1

        # Check velocity curriculum before ep_infos is cleared for the next rollout.
        vel_reset = False
        if vel_curriculum is not None:
            last_obs, last_critic_obs, vel_reset = vel_curriculum.evaluate_and_update_stage(
                ep_infos, usr_conf, env, last_obs, last_critic_obs, rollout_stats=storage_stats
            )
        # If env.reset was triggered by a stage change, stale accumulated rewards
        # from interrupted episodes must be discarded to prevent corrupting rewbuffer.
        if vel_reset:
            cur_reward_sum.zero_()
            cur_episode_length.zero_()
            if nav_controller is not None:
                nav_controller.reset(agent.num_envs, agent.device)

        # framework=True allows the business workflow to trigger the learner
        # directly without passing an external sample batch.
        agent.learn(list_sample_data=None)
        # Clear rollout storage after the learner has consumed the current batch.
        storage.clear()
        total_cost_time = round(time.time() - start_time, 2)
        logger.info(f"Episode {episode} end, cost_time is {total_cost_time} s")

        # Phase 3: Monitoring Metrics Processing
        now = time.time()
        if now - last_report_monitor_time >= 60:
            report_training_monitor_snapshot(
                ep_infos,
                reward_keys,
                agent,
                monitor,
                episode,
                storage_stats,
                vel_stage=vel_curriculum.stage if vel_curriculum is not None else 0,
                vel_tracking_ratio=vel_curriculum.last_tracking_ratio if vel_curriculum is not None else 0.0,
                vel_tracking_reward=vel_curriculum.last_tracking_reward if vel_curriculum is not None else 0.0,
                vel_event=vel_curriculum.last_transition if vel_curriculum is not None else 0,
                vel_up=vel_curriculum.promote_total if vel_curriculum is not None else 0,
                vel_down=vel_curriculum.demote_total if vel_curriculum is not None else 0,
                vel_p_streak=vel_curriculum.promote_streak if vel_curriculum is not None else 0,
                vel_d_streak=vel_curriculum.demote_streak if vel_curriculum is not None else 0,
                vel_max=vel_curriculum.active_max_vx if vel_curriculum is not None else 0.0,
                lenbuffer=lenbuffer,
                rewbuffer=rewbuffer,
            )
            last_report_monitor_time = now

        ep_infos.clear()

        # Phase 4: Model Saving
        if episode == 1 or episode % agent.save_interval == 0:
            logger.info(f"Saving checkpoint at episode {episode}")
            agent.save_model(id=str(episode))

    env.close()


def _extract_episode_metric_mean(ep_info, key, device):
    """Extract one episode metric and convert it into a scalar tensor mean."""
    if key not in ep_info:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    metric = ep_info[key]
    if not isinstance(metric, torch.Tensor):
        metric = torch.tensor(metric, device=device)
    return metric.float().mean()


def _average_metric_history_by_name(metric_history_by_name):
    """Reduce a metric-history dictionary into scalar means keyed by metric name."""
    aggregated_metrics = {}
    for metric_key, values in metric_history_by_name.items():
        if values:
            aggregated_metrics[metric_key] = torch.stack(values).mean().item()
        else:
            aggregated_metrics[metric_key] = 0.0
    return aggregated_metrics


def _collect_episode_metric_means(ep_infos, reward_keys, device):
    """Collect the mean value for each requested metric across finished episodes."""
    metric_history_by_name = defaultdict(list)
    for ep_info in ep_infos:
        for key in reward_keys:
            metric_value = _extract_episode_metric_mean(ep_info, key, device)
            metric_history_by_name[key].append(metric_value)
    return _average_metric_history_by_name(metric_history_by_name)


def report_training_monitor_snapshot(
    ep_infos,
    reward_keys,
    agent,
    monitor,
    episode,
    storage_stats=None,
    vel_stage: int = 0,
    vel_tracking_ratio: float = 0.0,
    vel_tracking_reward: float = 0.0,
    vel_event: int = 0,
    vel_up: int = 0,
    vel_down: int = 0,
    vel_p_streak: int = 0,
    vel_d_streak: int = 0,
    vel_max: float = 0.0,
    lenbuffer=None,
    rewbuffer=None,
):
    """Push reward, curriculum, and rollout summaries to the monitor process.

    The monitor receives a mix of per-episode metrics, rollout-level summaries,
    and live physics snapshots. Keeping the merge logic in one place avoids
    accidentally overwriting richer workflow metrics with missing episode keys.
    """
    monitor_data = {
        "ep_count": episode,
        "vel_stage": vel_stage,
        "vel_track": vel_tracking_ratio,
        "vel_rew": vel_tracking_reward,
        "vel_event": vel_event,
        "vel_up": vel_up,
        "vel_down": vel_down,
        "vel_p_streak": vel_p_streak,
        "vel_d_streak": vel_d_streak,
        "vel_max": vel_max,
    }

    # Merge all storage stats: reward_mean/reward_std AND physics obs_ keys.
    if storage_stats:
        for key, value in storage_stats.items():
            monitor_data[_R1_PHYSICS_KEYS.get(key, key)] = value

    # Episode health metrics: episode length and cumulative reward per episode.
    if lenbuffer:
        monitor_data["ep_steps"] = float(sum(lenbuffer) / len(lenbuffer))
    if rewbuffer:
        monitor_data["ep_reward"] = float(sum(rewbuffer) / len(rewbuffer))

    if ep_infos:
        metrics = _collect_episode_metric_means(ep_infos, reward_keys, agent.device)
        for key, value in metrics.items():
            alias = _R1_REWARD_KEYS.get(key)
            if alias:
                monitor_data[alias] = value

    logger = getattr(agent, "logger", None)
    if logger is not None:
        logger.warning(
            "[MonitorDebug] reporting curriculum metrics: "
            f"episode={episode}, stage={monitor_data.get('vel_stage')}, "
            f"event={monitor_data.get('vel_event')}, "
            f"tracking_ratio={monitor_data.get('vel_track')}, "
            f"max_vx={monitor_data.get('vel_max')}"
        )

    monitor.put_data({os.getpid(): monitor_data})


def _unpack_env_step_result(data, episode, logger):
    """Normalize the framework-specific env.step payload into rollout tensors."""
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


def _move_step_tensors_to_device(obs, critic_obs, rewards, dones, device):
    """Move the step outputs onto the learner device before storage writes."""
    return (
        obs.to(device),
        critic_obs.to(device),
        rewards.to(device),
        dones.to(device),
    )


def _populate_rollout_transition(
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
    """Populate one rollout transition record from the current step outputs."""
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
    # timeouts
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


def _refresh_completed_episode_buffers(
    dones,
    rewards,
    infos,
    cur_reward_sum,
    cur_episode_length,
    rewbuffer,
    lenbuffer,
    ep_infos,
):
    """Update rolling episode reward/length buffers and reset finished env slots."""
    if "episode" in infos:
        ep_infos.append(infos["episode"])

    cur_reward_sum += rewards
    cur_episode_length += 1

    new_ids = (dones > 0).nonzero(as_tuple=False)
    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

    cur_reward_sum[new_ids] = 0
    cur_episode_length[new_ids] = 0


def _finalize_rollout_returns_and_statistics(
    storage,
    agent,
    obs,
    critic_obs,
    logger,
    env=None,
    nav_controller=None,
    usr_conf=None,
):
    """Compute final values, GAE returns, and coarse rollout summary statistics."""
    last_obs = obs
    last_critic_obs = critic_obs
    if nav_controller is not None and last_obs is not None:
        last_obs, last_critic_obs, _ = _apply_navigation_controller_command(
            last_obs,
            last_critic_obs,
            env,
            nav_controller,
            logger,
            update_nav_state=False,
            update_env_command=False,
        )
    if usr_conf is not None and last_obs is not None:
        last_obs, last_critic_obs, _ = _apply_phase_conditioned_command(
            last_obs,
            last_critic_obs,
            env,
            usr_conf,
            logger,
            update_state=False,
            update_env_command=False,
        )

    value_obs = last_critic_obs if last_critic_obs is not None else last_obs
    last_values = agent.algorithm.actor_critic.evaluate(value_obs.detach()).detach()
    storage.compute_returns(last_values, agent.algorithm.gamma, agent.algorithm.lam)

    storage_stats = {
        "reward_mean": storage.rewards.mean().item(),
        "reward_std": storage.rewards.std().item(),
    }

    return storage_stats


def _estimate_rollout_tracking_metrics(storage, usr_conf, logger=None):
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


def _resolve_wrapped_isaac_env(env):
    """Try to unwrap the KaiwuDRL env wrapper to the underlying Isaac Lab env.

    Returns the first object that exposes both `command_manager` and `scene`,
    or None if no such object can be found through the wrapper chain.
    """
    # Walk common wrapper chains recursively instead of assuming one fixed depth.
    seen_ids = set()
    pending = [env]
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen_ids:
            continue
        seen_ids.add(id(candidate))
        if hasattr(candidate, "command_manager") and hasattr(candidate, "scene"):
            return candidate
        for attr_name in (
            "env",
            "_env",
            "unwrapped",
            "wrapped_env",
            "_wrapped_env",
            "venv",
            "isaac_env",
            "_isaac_env",
            "sim_env",
            "_sim_env",
            "gym_env",
            "_gym_env",
            "task",
            "_task",
        ):
            pending.append(getattr(candidate, attr_name, None))
    return None


def _has_robot_state(asset) -> bool:
    data = getattr(asset, "data", None)
    return (
        data is not None
        and hasattr(data, "root_lin_vel_b")
        and hasattr(data, "root_pos_w")
        and hasattr(data, "root_ang_vel_b")
    )


def _find_robot_asset_in_scene(isaac_env):
    """Best-effort robot asset lookup across common Isaac Lab scene layouts.

    The lookup intentionally checks several common container names so the
    monitor path survives small scene-layout differences between env wrappers.
    """
    if isaac_env is None:
        return None

    scene = getattr(isaac_env, "scene", None)
    if scene is None:
        return None

    if hasattr(scene, "__getitem__"):
        scene_keys = []
        keys_fn = getattr(scene, "keys", None)
        if callable(keys_fn):
            try:
                scene_keys = list(keys_fn())
            except Exception:
                scene_keys = []
        for key in ("robot", "Robot", "go2", "Go2", "unitree_go2", "UnitreeGo2", *scene_keys):
            try:
                asset = scene[key]
            except Exception:
                asset = None
            if _has_robot_state(asset):
                return asset

    for container_name in (
        "articulations",
        "_articulations",
        "rigid_objects",
        "_rigid_objects",
        "entities",
        "_entities",
    ):
        container = getattr(scene, container_name, None)
        if container is None:
            continue

        if hasattr(container, "get"):
            for key in ("robot", "Robot", "go2", "Go2", "unitree_go2", "UnitreeGo2"):
                asset = container.get(key)
                if _has_robot_state(asset):
                    return asset

        values = getattr(container, "values", None)
        if callable(values):
            for asset in values():
                if _has_robot_state(asset):
                    return asset

    for attr_name in ("robot", "_robot"):
        asset = getattr(isaac_env, attr_name, None)
        if _has_robot_state(asset):
            return asset

    return None


def _hard_start_monitor_metrics(env):
    """Best-effort mirror of hard-start worker metrics."""
    from agent_ppo.feature.hard_start_replay import START_NAMES

    isaac_env = _resolve_wrapped_isaac_env(env)
    state = getattr(isaac_env, "_hard_start_replay_state", None)
    if state is None:
        return {}

    starts = state["start_counts"].float()
    episodes = state["episode_counts"].float()
    successes = state["success_counts"].float()
    total = torch.clamp(starts.sum(), min=1.0)
    metrics = {
        "full_track_ratio": (starts[0] / total).item(),
        "hard_start_ratio": (starts[1:].sum() / total).item(),
        "early_bad_total": state["early_bad_counts"].sum().item(),
        "early_base_total": state["early_base_counts"].sum().item(),
        "early_term_total": state["early_term_counts"].sum().item(),
        "surface_query_fail": float(state["surface_query_failures"]),
    }
    for index, name in enumerate(START_NAMES[1:], start=1):
        metrics[f"{name}_starts"] = starts[index].item()
        metrics[f"{name}_success"] = (
            successes[index] / torch.clamp(episodes[index], min=1.0)
        ).item()
        metrics[f"bad_{name}"] = state["bad_counts"][index].item()
        metrics[f"base_{name}"] = state["base_counts"][index].item()
        metrics[f"timeout_{name}"] = state["timeout_counts"][index].item()

    clearance_count = state["spawn_clearance_count"][1:].sum().float()
    metrics["spawn_clearance"] = (
        state["spawn_clearance_sum"][1:].sum()
        / torch.clamp(clearance_count, min=1.0)
    ).item()
    return metrics


def _estimate_physics_metrics_from_critic_obs(critic_obs):
    """Fallback physics metrics derived purely from critic observation layout.

    critic_obs layout is:
      [0:3] base_lin_vel, [3:6] base_ang_vel, [9:12] velocity command.

    This path does not provide base height because height is not part of the
    documented critic observation.  It still keeps velocity and attitude panels
    alive when the wrapped Isaac env cannot be reached from the workflow.
    """
    if critic_obs is None or not hasattr(critic_obs, "shape") or critic_obs.shape[-1] < 12:
        return {}

    actual_vx = critic_obs[:, 0]
    actual_vy = critic_obs[:, 1]
    actual_yaw = critic_obs[:, 5]
    cmd_vx = critic_obs[:, 9]
    cmd_vy = critic_obs[:, 10]
    cmd_yaw = critic_obs[:, 11]

    return {
        "obs_cmd_vel_x": cmd_vx.mean().item(),
        "obs_cmd_vel_y": cmd_vy.mean().item(),
        "obs_cmd_yaw": cmd_yaw.mean().item(),
        "obs_lin_vel_x_error": torch.abs(actual_vx - cmd_vx).mean().item(),
        "obs_lin_vel_y_error": torch.abs(actual_vy - cmd_vy).mean().item(),
        "obs_yaw_error": torch.abs(actual_yaw - cmd_yaw).mean().item(),
        "obs_actual_vel_x": actual_vx.mean().item(),
        "obs_actual_vel_y": actual_vy.mean().item(),
        "obs_actual_yaw": actual_yaw.mean().item(),
        "obs_ang_vel_xy": torch.norm(critic_obs[:, 3:5], dim=1).mean().item(),
    }


def _sample_runtime_physics_metrics(env, logger=None, critic_obs=None):
    """Take a point-in-time snapshot of key physical quantities across all envs.

    Falls back to critic_obs-derived metrics when the underlying Isaac Lab
    env cannot be unwrapped from the framework wrapper chain.
    """
    try:
        isaac_env = _resolve_wrapped_isaac_env(env)
        if isaac_env is None:
            if logger is not None and not getattr(env, "_physics_stats_error_logged", False):
                env._physics_stats_error_logged = True
                logger.warning(
                    "[PhysicsStats] Failed to unwrap Isaac Lab env; "
                    "falling back to critic_obs for partial physics metrics."
                )
            return _estimate_physics_metrics_from_critic_obs(critic_obs)

        cmd = isaac_env.command_manager.get_command("base_velocity")  # (N, 3)
        asset = _find_robot_asset_in_scene(isaac_env)
        if asset is None:
            if logger is not None and not getattr(env, "_physics_stats_error_logged", False):
                env._physics_stats_error_logged = True
                logger.warning(
                    "[PhysicsStats] Failed to locate robot asset in scene; "
                    "falling back to critic_obs for partial physics metrics."
                )
            return _estimate_physics_metrics_from_critic_obs(critic_obs)

        actual_vx = asset.data.root_lin_vel_b[:, 0]
        actual_vy = asset.data.root_lin_vel_b[:, 1]
        actual_yaw = asset.data.root_ang_vel_b[:, 2]
        cmd_vx    = cmd[:, 0]
        cmd_vy    = cmd[:, 1]
        cmd_yaw   = cmd[:, 2]

        stats = {
            "obs_cmd_vel_x":       cmd_vx.mean().item(),
            "obs_cmd_vel_y":       cmd_vy.mean().item(),
            "obs_cmd_yaw":         cmd_yaw.mean().item(),
            "obs_lin_vel_x_error": torch.abs(actual_vx - cmd_vx).mean().item(),
            "obs_lin_vel_y_error": torch.abs(actual_vy - cmd_vy).mean().item(),
            "obs_yaw_error":       torch.abs(actual_yaw - cmd_yaw).mean().item(),
            "obs_actual_vel_x":    actual_vx.mean().item(),
            "obs_actual_vel_y":    actual_vy.mean().item(),
            "obs_actual_yaw":      actual_yaw.mean().item(),
            "obs_base_height":     asset.data.root_pos_w[:, 2].mean().item(),
            "obs_ang_vel_xy":      torch.norm(
                asset.data.root_ang_vel_b[:, :2], dim=1).mean().item(),
        }
        stats.update(_sample_terrain_curriculum_metrics(isaac_env, env))
        return stats
    except Exception as exc:
        if logger is not None and not getattr(env, "_physics_stats_error_logged", False):
            env._physics_stats_error_logged = True
            logger.warning(
                f"[PhysicsStats] Failed to sample physics metrics from Isaac env: {exc}; "
                "falling back to critic_obs for partial physics metrics."
            )
        return _estimate_physics_metrics_from_critic_obs(critic_obs)


def _sample_terrain_curriculum_metrics(isaac_env, state_owner=None):
    """Best-effort terrain curriculum snapshot from common Isaac Lab fields."""
    levels = _find_tensor_attr(
        isaac_env,
        (
            "terrain_levels",
            "_terrain_levels",
            "scene.terrain.terrain_levels",
            "scene.terrain._terrain_levels",
            "terrain.terrain_levels",
            "terrain._terrain_levels",
        ),
    )
    if levels is None:
        return {}

    levels = levels.detach().float().view(-1)
    if levels.numel() == 0:
        return {}

    metrics = {
        "terrain_level": levels.mean().item(),
        "terrain_level_max": levels.max().item(),
        "terrain_level_min": levels.min().item(),
    }

    owner = state_owner if state_owner is not None else isaac_env
    prev_levels = getattr(owner, "_r1_prev_terrain_levels", None)
    if prev_levels is not None and prev_levels.shape == levels.shape:
        prev_levels = prev_levels.to(device=levels.device, dtype=levels.dtype)
        metrics["terrain_level_up"] = (levels > prev_levels).float().mean().item()
        metrics["terrain_level_down"] = (levels < prev_levels).float().mean().item()
    else:
        metrics["terrain_level_up"] = 0.0
        metrics["terrain_level_down"] = 0.0

    try:
        owner._r1_prev_terrain_levels = levels.detach().clone()
    except Exception:
        pass
    return metrics


def _find_tensor_attr(root, attr_paths):
    for attr_path in attr_paths:
        value = root
        try:
            for part in attr_path.split("."):
                value = getattr(value, part)
            if torch.is_tensor(value):
                return value
        except Exception:
            continue
    return None


def _override_env_base_velocity_command(env, command, logger=None):
    """Best-effort override of Isaac Lab's sampled base_velocity command."""
    isaac_env = _resolve_wrapped_isaac_env(env)
    if isaac_env is None or not hasattr(isaac_env, "command_manager"):
        if logger is not None and not getattr(env, "_nav_command_error_logged", False):
            env._nav_command_error_logged = True
            logger.warning("[Navigation] Cannot unwrap Isaac env; only observation command will be overridden.")
        return False

    try:
        current_command = isaac_env.command_manager.get_command("base_velocity")
        if current_command.shape != command.shape:
            if logger is not None and not getattr(env, "_nav_command_error_logged", False):
                env._nav_command_error_logged = True
                logger.warning(
                    "[Navigation] base_velocity command shape mismatch: "
                    f"env={tuple(current_command.shape)}, nav={tuple(command.shape)}."
                )
            return False
        current_command.copy_(command.to(device=current_command.device, dtype=current_command.dtype))
        return True
    except Exception as exc:
        if logger is not None and not getattr(env, "_nav_command_error_logged", False):
            env._nav_command_error_logged = True
            logger.warning(f"[Navigation] Failed to override base_velocity command: {exc}")
        return False


def _compute_uniform_range_midpoint(range_values, default_value: float) -> float:
    if not isinstance(range_values, (list, tuple)) or len(range_values) != 2:
        return default_value
    return 0.5 * (float(range_values[0]) + float(range_values[1]))


def _sample_uniform_range_values(range_values, shape, device, dtype):
    low = float(range_values[0])
    high = float(range_values[1])
    if high < low:
        low, high = high, low
    if abs(high - low) <= 1e-8:
        return torch.full(shape, low, device=device, dtype=dtype)
    return low + (high - low) * torch.rand(shape, device=device, dtype=dtype)


def _infer_maze_phase_mask_from_obs(obs, usr_conf):
    """Infer the maze-phase boolean mask from goal-distance features in policy obs.

    Track navigation appends goal features at obs[301:304]:
      local goal x/y normalized by 10m, and goal distance normalized by 20m.
    The existing reward gate treats the final maze phase as goal_dist < 14m;
    using the same threshold keeps command scheduling aligned with rewards.
    """
    if obs is None or not hasattr(obs, "shape") or obs.shape[-1] < 304:
        return None

    rl_nav_conf = usr_conf.get("rl_navigation", {})
    goal_start = int(rl_nav_conf.get("goal_start", 301))
    if obs.shape[-1] < goal_start + 3:
        return None

    goal_dist_gate = float(rl_nav_conf.get("phase_maze_goal_dist_gate", 14.0))
    goal_dist = torch.clamp(obs[:, goal_start + 2], 0.0, 1.0) * 20.0
    return goal_dist < goal_dist_gate


def _reshape_height_scan_from_obs(obs, usr_conf):
    if obs is None or not hasattr(obs, "shape"):
        return None
    rl_nav_conf = usr_conf.get("rl_navigation", {})
    scan_start = int(rl_nav_conf.get("scan_start", 45))
    scan_size = int(rl_nav_conf.get("scan_size", 256))
    if obs.shape[-1] < scan_start + scan_size:
        return None
    side = int(scan_size ** 0.5)
    if side * side != scan_size:
        return None
    return obs[:, scan_start:scan_start + scan_size].view(obs.shape[0], side, side)


def _infer_pre_maze_terrain_type_from_obs(obs, usr_conf):
    """Classify non-maze terrain into flat, slope, or stairs from the height scan."""
    grid = _reshape_height_scan_from_obs(obs, usr_conf)
    if grid is None:
        return None

    rl_nav_conf = usr_conf.get("rl_navigation", {})
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


def _build_phase_command_fallback(obs, maze_phase, usr_conf, dtype):
    rl_nav_conf = usr_conf.get("rl_navigation", {})
    pre_range = rl_nav_conf.get("pre_maze_lin_vel_x", [0.75, 1.0])
    slope_range = rl_nav_conf.get("slope_lin_vel_x", pre_range)
    stairs_range = rl_nav_conf.get("stairs_lin_vel_x", pre_range)
    maze_range = rl_nav_conf.get("maze_lin_vel_x", [0.45, 0.65])
    terrain_phase_speed_enabled = bool(rl_nav_conf.get("terrain_phase_speed_enabled", False))
    pre_vx = _compute_uniform_range_midpoint(pre_range, 0.875)
    slope_vx = _compute_uniform_range_midpoint(slope_range, pre_vx)
    stairs_vx = _compute_uniform_range_midpoint(stairs_range, pre_vx)
    maze_vx = _compute_uniform_range_midpoint(maze_range, 0.55)
    command = torch.zeros(maze_phase.shape[0], 3, device=maze_phase.device, dtype=dtype)
    pre_command = torch.full_like(command[:, 0], pre_vx)
    if terrain_phase_speed_enabled and obs is not None:
        terrain_id = _infer_pre_maze_terrain_type_from_obs(obs, usr_conf)
        if terrain_id is not None:
            pre_command = torch.where(
                terrain_id == 1,
                torch.full_like(pre_command, slope_vx),
                pre_command,
            )
            pre_command = torch.where(
                terrain_id == 2,
                torch.full_like(pre_command, stairs_vx),
                pre_command,
            )
    command[:, 0] = torch.where(maze_phase, torch.full_like(command[:, 0], maze_vx), pre_command)
    return command


def _get_or_create_phase_command_state(env, num_envs, device, dtype):
    state = getattr(env, "_rl_phase_command_state", None)
    if (
        state is None
        or state["command"].shape[0] != num_envs
        or state["command"].device != device
        or state["command"].dtype != dtype
    ):
        state = {
            "command": torch.zeros(num_envs, 3, device=device, dtype=dtype),
            "timer": torch.zeros(num_envs, dtype=torch.long, device=device),
            "maze_phase": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "terrain_id": torch.zeros(num_envs, dtype=torch.long, device=device),
        }
        setattr(env, "_rl_phase_command_state", state)
    return state


def _reset_finished_phase_command_state(env, dones):
    state = getattr(env, "_rl_phase_command_state", None)
    if state is None or dones is None:
        return
    done_mask = dones.bool().view(-1)
    if done_mask.any() and state["timer"].shape[0] == done_mask.shape[0]:
        state["timer"][done_mask] = 0
        state["command"][done_mask] = 0.0
        state["maze_phase"][done_mask] = False
        if "terrain_id" in state:
            state["terrain_id"][done_mask] = 0


def _apply_phase_conditioned_command(
    obs,
    critic_obs,
    env,
    usr_conf,
    logger=None,
    update_state=True,
    update_env_command=True,
):
    """Override the locomotion command anchor before and after the maze phase.

    This is intentionally separate from the rule-based navigation controller:
    it changes only the velocity command observation/reward anchor while the
    PPO policy still controls all joints directly.
    """
    rl_nav_conf = usr_conf.get("rl_navigation", {})
    if not bool(rl_nav_conf.get("phase_command_enabled", False)):
        return obs, critic_obs, {}

    maze_phase = _infer_maze_phase_mask_from_obs(obs, usr_conf)
    if maze_phase is None:
        return obs, critic_obs, {}

    state = _get_or_create_phase_command_state(env, obs.shape[0], obs.device, obs.dtype)
    resample_steps = max(int(rl_nav_conf.get("phase_command_resample_steps", 96)), 1)
    pre_range = rl_nav_conf.get("pre_maze_lin_vel_x", [0.75, 1.0])
    slope_range = rl_nav_conf.get("slope_lin_vel_x", pre_range)
    stairs_range = rl_nav_conf.get("stairs_lin_vel_x", pre_range)
    maze_range = rl_nav_conf.get("maze_lin_vel_x", [0.45, 0.65])
    terrain_phase_speed_enabled = bool(rl_nav_conf.get("terrain_phase_speed_enabled", False))
    terrain_id = _infer_pre_maze_terrain_type_from_obs(obs, usr_conf)
    if terrain_id is None:
        terrain_id = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)

    command = state["command"]
    if update_state:
        timer = torch.clamp(state["timer"] - 1, min=0)
        phase_changed = maze_phase != state["maze_phase"]
        terrain_changed = terrain_phase_speed_enabled & (~maze_phase) & (terrain_id != state["terrain_id"])
        needs_sample = (timer <= 0) | phase_changed | terrain_changed | (command[:, 0] <= 0.0)

        if needs_sample.any():
            sampled_pre = _sample_uniform_range_values(pre_range, (obs.shape[0],), obs.device, obs.dtype)
            sampled_slope = _sample_uniform_range_values(slope_range, (obs.shape[0],), obs.device, obs.dtype)
            sampled_stairs = _sample_uniform_range_values(stairs_range, (obs.shape[0],), obs.device, obs.dtype)
            sampled_maze = _sample_uniform_range_values(maze_range, (obs.shape[0],), obs.device, obs.dtype)
            sampled_non_maze = sampled_pre
            if terrain_phase_speed_enabled:
                sampled_non_maze = torch.where(terrain_id == 1, sampled_slope, sampled_non_maze)
                sampled_non_maze = torch.where(terrain_id == 2, sampled_stairs, sampled_non_maze)
            sampled_vx = torch.where(maze_phase, sampled_maze, sampled_non_maze)
            command = command.clone()
            command[needs_sample, 0] = sampled_vx[needs_sample]
            command[needs_sample, 1] = 0.0
            command[needs_sample, 2] = 0.0
            state["command"] = command

        state["timer"] = torch.where(
            needs_sample,
            torch.full_like(timer, resample_steps),
            timer,
        )
        state["maze_phase"] = maze_phase.detach().clone()
        state["terrain_id"] = terrain_id.detach().clone()
    else:
        command = command.clone()
        phase_changed = maze_phase != state["maze_phase"]
        invalid = command[:, 0] <= 0.0
        fallback = _build_phase_command_fallback(obs, maze_phase, usr_conf, obs.dtype)
        command[phase_changed | invalid] = fallback[phase_changed | invalid]

    obs = obs.clone()
    if obs.shape[-1] >= 9:
        obs[:, 6:9] = command

    if critic_obs is not None:
        critic_obs = critic_obs.clone()
        if critic_obs.shape[-1] >= 12:
            critic_obs[:, 9:12] = command.to(device=critic_obs.device, dtype=critic_obs.dtype)

    if update_env_command:
        _override_env_base_velocity_command(env, command, logger)

    return obs, critic_obs, {
        "rl_phase_command_vx": command[:, 0].detach(),
        "rl_phase_maze_ratio": maze_phase.float().detach(),
        "rl_phase_slope_ratio": (terrain_id == 1).float().detach(),
        "rl_phase_stairs_ratio": (terrain_id == 2).float().detach(),
    }


def _apply_navigation_controller_command(
    obs,
    critic_obs,
    env,
    nav_controller,
    logger=None,
    update_nav_state=True,
    update_env_command=True,
):
    """Patch policy and critic observations with the navigation controller output."""
    if nav_controller is None:
        return obs, critic_obs, {}

    command, nav_stats = nav_controller.compute(obs, update_state=update_nav_state)
    command = command.to(device=obs.device, dtype=obs.dtype)

    obs = obs.clone()
    if obs.shape[-1] >= 9:
        obs[:, 6:9] = command

    if critic_obs is not None:
        critic_obs = critic_obs.clone()
        if critic_obs.shape[-1] >= 12:
            critic_obs[:, 9:12] = command.to(device=critic_obs.device, dtype=critic_obs.dtype)

    if update_env_command:
        _override_env_base_velocity_command(env, command, logger)
    return obs, critic_obs, nav_stats


def _evaluate_timeout_bootstrap_values(obs, critic_obs, env, nav_controller, agent, infos, logger=None, usr_conf=None):
    """Evaluate V(s_{t+1}) for timeout bootstrapping using rollout-consistent inputs."""
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
    if nav_controller is not None:
        value_obs, value_critic_obs, _ = _apply_navigation_controller_command(
            obs,
            critic_obs,
            env,
            nav_controller,
            logger,
            update_nav_state=False,
            update_env_command=False,
        )
    if usr_conf is not None:
        value_obs, value_critic_obs, _ = _apply_phase_conditioned_command(
            value_obs,
            value_critic_obs,
            env,
            usr_conf,
            logger,
            update_state=False,
            update_env_command=False,
        )

    value_input = value_critic_obs if value_critic_obs is not None else value_obs
    return agent.algorithm.actor_critic.evaluate(value_input.detach()).detach()


def _average_navigation_metric_history(nav_metric_values):
    """Reduce navigation-side metric histories into scalar means."""
    aggregated = {}
    for key, values in nav_metric_values.items():
        if values:
            aggregated[key] = torch.stack(values).mean().item()
    return aggregated


def collect_rollout_batch(
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
    nav_controller=None,
):
    """Collect one fixed-length rollout batch and update episode statistics."""
    transition = RolloutStorage.Transition()
    obs, critic_obs = last_obs, last_critic_obs
    nav_metric_values = defaultdict(list)
    hard_start_info_metrics = {}

    # TODO: for hierarchical training, handle the mismatch between env action and
    # PPO storage action on your own.
    # TODOenv action PPO storage action
    # Policy execution loop
    #
    with torch.inference_mode():
        for i in range(agent.num_steps_per_env):
            policy_obs, policy_critic_obs, nav_stats = _apply_navigation_controller_command(
                obs, critic_obs, env, nav_controller, logger
            )
            policy_obs, policy_critic_obs, phase_stats = _apply_phase_conditioned_command(
                policy_obs,
                policy_critic_obs,
                env,
                usr_conf,
                logger,
            )
            for key, value in nav_stats.items():
                nav_metric_values[key].append(value.float().mean())
            for key, value in phase_stats.items():
                nav_metric_values[key].append(value.float().mean())

            # Predict actions
            #
            predict_data = (policy_obs, policy_critic_obs)
            predict_result = agent.predict(predict_data)

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
            joint_actions = actions

            # Clip joint actions for env
            #
            command_actions = torch.clip(joint_actions, -6.0, 6.0).to(agent.device)
            if i == 0:
                logger.info(f"clipped_action:{command_actions}")

            # Advance the simulator with the sampled actions.
            data = env.step(command_actions)
            frame_no, obs, critic_obs, rewards, dones, infos = _unpack_env_step_result(data, episode, logger)
            info_metrics = (
                infos.get("hard_start_metrics", {})
                if isinstance(infos, dict)
                else {}
            )
            if isinstance(info_metrics, dict):
                for key, value in info_metrics.items():
                    if isinstance(value, (int, float)):
                        hard_start_info_metrics[key] = float(value)

            # Keep all tensors on the learner device before storing transitions.
            obs, critic_obs, rewards, dones = _move_step_tensors_to_device(
                obs,
                critic_obs,
                rewards,
                dones,
                agent.device,
            )
            timeout_bootstrap_values = _evaluate_timeout_bootstrap_values(
                obs,
                critic_obs,
                env,
                nav_controller,
                agent,
                infos,
                logger,
                usr_conf=usr_conf,
            )
            if nav_controller is not None:
                nav_controller.reset(dones=dones)
            _reset_finished_phase_command_state(env, dones)

            # Update episode statistics (always, regardless of decimation)
            _refresh_completed_episode_buffers(
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
            _populate_rollout_transition(
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
        storage_stats = _finalize_rollout_returns_and_statistics(
            storage,
            agent,
            obs,
            critic_obs,
            logger,
            env=env,
            nav_controller=nav_controller,
            usr_conf=usr_conf,
        )
        storage_stats.update(_estimate_rollout_tracking_metrics(storage, usr_conf, logger))
        storage_stats.update(_average_navigation_metric_history(nav_metric_values))
        last_obs = torch.clone(obs)

    # Batch generation is now handled by AlgorithmPPO.learn(), and storage is
    # cleared in the workflow after the optimization step finishes.
    # Append a physics snapshot (averaged across all envs).
    # Wrapped in try/except inside _sample_runtime_physics_metrics, so always safe.
    storage_stats.update(_sample_runtime_physics_metrics(env, logger, critic_obs=critic_obs))
    storage_stats.update(_hard_start_monitor_metrics(env))
    storage_stats.update(hard_start_info_metrics)

    return last_obs, critic_obs, storage_stats
