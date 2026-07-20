# -*- coding: UTF-8 -*-
"""LBC policy observation processor.

Output layout:
    [proprio(45) | height_scan(256) | optional_goal(num_goal_obs) | depth]
"""

from agent_ppo.conf.conf import Config
from agent_ppo.feature import nav_observation_utils
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


class LBCObservationProcess(ObservationProcess):
    target_group = "policy"

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
        # This one tensor is concatenated once and then shared by teacher and
        # student action paths through AlgorithmLBC._split_obs().
        shared_raw_goal = self.goal_noise_augmenter.apply(raw_goal)
        return encode_track_goal(shared_raw_goal)

    def process(self):
        obs = self.default_observation()
        goal_features = self._goal_features()
        if goal_features is not None:
            obs = self.concatenate_terms(obs, goal_features)
        depth = nav_observation_utils.depth_camera_image(self.env)
        return self.concatenate_terms(obs, depth)
