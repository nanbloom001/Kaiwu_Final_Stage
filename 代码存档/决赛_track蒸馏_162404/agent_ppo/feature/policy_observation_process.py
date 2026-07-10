#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Policy observation processor.

Base policy obs is [proprio(45) | height_scan(256)] = 301.
When the active stage declares ``num_goal_obs = 3``, append the same p22-r
track goal feature layout, producing a 304-D flat teacher observation.
"""

from agent_ppo.conf.conf import Config
from agent_ppo.feature.goal_features import build_track_goal_features
from agent_ppo.feature.terrain_gate import apply_worker_gate_command
from tools.base_env.observation_process import ObservationProcess


class PolicyObservationProcess(ObservationProcess):
    target_group = "policy"
    _BASE_OBS_DIM = 301

    def _goal_features(self):
        feature_dim = getattr(Config.CURRENT, "num_goal_obs", 0)
        if feature_dim <= 0:
            return None
        return build_track_goal_features(self.env, feature_dim)

    def process(self):
        obs = self.default_observation()
        if obs.shape[-1] != self._BASE_OBS_DIM:
            raise ValueError(
                f"Policy observation dim mismatch: expected base {self._BASE_OBS_DIM}, "
                f"got {obs.shape[-1]}."
            )

        obs = apply_worker_gate_command(self.env, obs, "policy")
        goal_features = self._goal_features()
        if goal_features is not None:
            obs = self.concatenate_terms(obs, goal_features)
        return obs
