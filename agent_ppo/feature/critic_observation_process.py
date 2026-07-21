# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Critic observation processor."""

from agent_ppo.conf.conf import Config
from agent_ppo.feature.goal_features import build_track_goal_features
from agent_ppo.feature.hard_start_replay import (
    initialize_hard_start_replay,
    publish_hard_start_metrics,
)
from tools.base_env.observation_process import ObservationProcess


class CriticObservationProcess(ObservationProcess):
    target_group = "critic"
    _BASE_OBS_DIM = 316

    def _goal_features(self):
        feature_dim = getattr(Config.CURRENT, "num_goal_obs", 0)
        if hasattr(self, "goal_position_in_robot_frame"):
            self.goal_position_in_robot_frame()
        return build_track_goal_features(self.env, feature_dim)

    def process(self):
        # Keep the first policy/critic observation pair consistent regardless
        # of ObservationManager group evaluation order.
        initialize_hard_start_replay(self.env)
        publish_hard_start_metrics(self.env)
        obs = self.default_observation()
        if obs.shape[-1] != self._BASE_OBS_DIM:
            raise ValueError(
                f"Critic observation dim mismatch: expected base {self._BASE_OBS_DIM}, got {obs.shape[-1]}."
            )

        goal_features = self._goal_features()
        if goal_features is not None:
            obs = self.concatenate_terms(obs, goal_features)
        return obs
