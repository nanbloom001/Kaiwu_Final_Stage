# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Velocity curriculum used by the PPO workflow."""

from typing import Optional, Tuple

import torch


class VelocityCurriculum:
    """Performance-based velocity curriculum driven by tracking reward."""

    _DEFAULT_STAGES = [
        {"lin_vel_x": [0.0, 0.5], "lin_vel_y": [-0.3, 0.3], "ang_vel_yaw": [-1.0, 1.0]},
        {"lin_vel_x": [0.0, 1.0], "lin_vel_y": [-0.5, 0.5], "ang_vel_yaw": [-1.5, 1.5]},
        {"lin_vel_x": [0.0, 1.5], "lin_vel_y": [-0.8, 0.8], "ang_vel_yaw": [-1.5, 1.5]},
        {"lin_vel_x": [-0.5, 2.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
    ]

    _TRACKING_KEY = "reward_track_lin_vel_xy"
    _TRACKING_KEY_CANDIDATES = (
        "reward_track_lin_vel_xy",
        "track_lin_vel_xy",
        "Episode_Reward/reward_track_lin_vel_xy",
        "Episode_Reward/track_lin_vel_xy",
    )

    def __init__(self, logger, usr_conf: dict):
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

        self.promote_threshold: float = self._normalize_threshold(
            float(vc_conf.get("promote_threshold", 0.64)), "promote_threshold"
        )
        self.demote_threshold: float = self._normalize_threshold(
            float(vc_conf.get("demote_threshold", 0.32)), "demote_threshold"
        )
        self.promote_count: int = int(vc_conf.get("promote_count", 5))
        self.demote_count: int = int(vc_conf.get("demote_count", 3))

        raw_stages = vc_conf.get("stages", None)
        if raw_stages:
            self.STAGES = [
                {
                    "lin_vel_x": list(s["lin_vel_x"]),
                    "lin_vel_y": list(s["lin_vel_y"]),
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
        self._promote_streak = 0
        self._demote_streak = 0
        self._last_mean_tracking_reward = 0.0
        self._last_mean_tracking_ratio = 0.0
        self._tracking_key_resolved = None
        self._tracking_key_warning_logged = False
        self._debug_check_count = 0
        logger.info(
            f"[VelocityCurriculum] Initialized: {len(self.STAGES)} stages, "
            f"tracking_weight={self._tracking_reward_weight}, "
            f"promote_threshold={self.promote_threshold}, demote_threshold={self.demote_threshold}, "
            f"promote_count={self.promote_count}, demote_count={self.demote_count}"
        )

    def _normalize_threshold(self, threshold_value: float, field_name: str) -> float:
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

    def _resolve_tracking_metric(self, ep_info):
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

    def _mean_tracking_reward(self, ep_infos) -> Tuple[Optional[float], Optional[float]]:
        values = []
        for ep_info in ep_infos:
            v, key = self._resolve_tracking_metric(ep_info)
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

    def _apply_stage(self, usr_conf, env, obs, critic_obs):
        cfg = self.STAGES[self._stage_idx]
        usr_conf["commands"]["ranges"]["lin_vel_x"] = cfg["lin_vel_x"]
        usr_conf["commands"]["ranges"]["lin_vel_y"] = cfg["lin_vel_y"]
        usr_conf["commands"]["ranges"]["ang_vel_yaw"] = cfg["ang_vel_yaw"]
        self.logger.info(
            f"[VelocityCurriculum] -> Stage {self._stage_idx}: "
            f"lin_vel_x={cfg['lin_vel_x']}, lin_vel_y={cfg['lin_vel_y']}, "
            f"ang_vel_yaw={cfg['ang_vel_yaw']} -> env.reset"
        )
        data = env.reset(usr_conf)
        if data is None:
            self.logger.error("[VelocityCurriculum] env.reset failed after stage change!")
            raise RuntimeError("VelocityCurriculum env.reset failed after stage change")
        new_obs, new_critic_obs = data
        if new_critic_obs is None:
            new_critic_obs = new_obs
        return torch.clone(new_obs), torch.clone(new_critic_obs), True

    def check_and_update(self, ep_infos, usr_conf, env, obs, critic_obs, rollout_stats=None):
        self._debug_check_count += 1
        rollout_stats = rollout_stats or {}
        mean_reward, mean_ratio = self._mean_tracking_reward(ep_infos)
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
                self.logger.warning(
                    f"[VelocityCurriculum] PROMOTE -> stage {self._stage_idx} "
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
                self.logger.warning(
                    f"[VelocityCurriculum] DEMOTE -> stage {self._stage_idx} "
                    f"(tracking_ratio={mean_ratio:.3f} < {self.demote_threshold:.3f}, "
                    f"tracking_reward={mean_reward:.3f}, "
                    f"for {self.demote_count} consecutive checks)"
                )
                stage_changed = True
        else:
            self._promote_streak = max(0, self._promote_streak - 1)
            self._demote_streak = max(0, self._demote_streak - 1)

        if self._debug_check_count <= 10 or self._debug_check_count % 20 == 0 or stage_changed:
            current_cfg = self.STAGES[self._stage_idx]
            self.logger.warning(
                "[VelocityCurriculumDebug] "
                f"check={self._debug_check_count}, ep_infos={len(ep_infos)}, "
                f"source={metric_source}, key={self._tracking_key_resolved}, "
                f"reward={mean_reward:.4f}, ratio={mean_ratio:.4f}, "
                f"stage={old_stage}->{self._stage_idx}, "
                f"promote={self._promote_streak}/{self.promote_count}@{self.promote_threshold:.3f}, "
                f"demote={self._demote_streak}/{self.demote_count}@{self.demote_threshold:.3f}, "
                f"ranges={current_cfg}"
            )

        if stage_changed:
            return self._apply_stage(usr_conf, env, obs, critic_obs)
        return obs, critic_obs, False
