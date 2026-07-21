# -*- coding: UTF-8 -*-
"""LBC policy observation processor.

Output layout:
    [proprio(45) | height_scan(256) | optional_goal(num_goal_obs) | depth]

The optional goal slice lets a p22-r track teacher actor consume
[proprio | latent | goal3] while the student still learns the visual latent
from depth.
"""

from agent_ppo.conf.conf import Config
from agent_ppo.feature import nav_observation_utils
from agent_ppo.feature.goal_features import build_track_goal_features
from tools.base_env.observation_process import ObservationProcess


class LBCObservationProcess(ObservationProcess):
    target_group = "policy"

    def process(self):
        obs = self.default_observation()
        feature_dim = getattr(Config.CURRENT, "num_goal_obs", 0)
        goal_features = build_track_goal_features(self.env, feature_dim)
        if goal_features is not None:
            obs = self.concatenate_terms(obs, goal_features)

        depth = nav_observation_utils.depth_camera_image(self.env)
        return self.concatenate_terms(obs, depth)
