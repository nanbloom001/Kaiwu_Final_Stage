# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Policy observation processor."""

from agent_ppo.conf.conf import Config
from agent_ppo.feature.goal_features import (
    build_track_goal_features,
    build_track_goal_raw,
    encode_track_goal,
)
from agent_ppo.feature.goal_noise import (
    GoalNoiseAugmenter,
    resolve_goal_noise_config,
)
from tools.base_env.observation_process import ObservationProcess


class PolicyObservationProcess(ObservationProcess):
    target_group = "policy"
    _BASE_OBS_DIM = 301

    def _goal_features(self):
        feature_dim = getattr(Config.CURRENT, "num_goal_obs", 0)
        if hasattr(self, "goal_position_in_robot_frame"):
            self.goal_position_in_robot_frame()
        if feature_dim != 3:
            return build_track_goal_features(self.env, feature_dim)

        raw_goal = build_track_goal_raw(self.env)
        if not hasattr(self, "goal_noise_augmenter"):
            self.goal_noise_augmenter = GoalNoiseAugmenter(
                env=self.env,
                config=resolve_goal_noise_config(self.env),
            )
        actor_raw_goal = self.goal_noise_augmenter.apply(raw_goal)
        return encode_track_goal(actor_raw_goal)

    def process(self):
        obs = self.default_observation()
        if obs.shape[-1] != self._BASE_OBS_DIM:
            raise ValueError(
                f"Policy observation dim mismatch: expected base {self._BASE_OBS_DIM}, got {obs.shape[-1]}."
            )

        goal_features = self._goal_features()
        if goal_features is not None:
            obs = self.concatenate_terms(obs, goal_features)
        return obs
