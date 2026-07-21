# -*- coding: UTF-8 -*-
"""LBC policy observation processor.

Output layout:
    [proprio(45) | height_scan(256) | optional_goal(num_goal_obs) | depth]
"""

import traceback as _tb

from agent_ppo.conf.conf import Config
from agent_ppo.feature import nav_observation_utils
from agent_ppo.feature.goal_features import build_track_goal_features
from agent_ppo.feature.terrain_gate import apply_worker_gate_command
from tools.base_env.observation_process import ObservationProcess

_GATE_ERR_COUNT = 0


def _log_gate_exc_once():
    global _GATE_ERR_COUNT
    if _GATE_ERR_COUNT >= 3:
        return
    _GATE_ERR_COUNT += 1
    try:
        import sys
        print(
            "[LBCObservationProcess] apply_worker_gate_command failed "
            "(#" + str(_GATE_ERR_COUNT) + "/3), skip patch:\n" + _tb.format_exc(),
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        pass


class LBCObservationProcess(ObservationProcess):
    target_group = "policy"

    def process(self):
        obs = self.default_observation()
        try:
            obs = apply_worker_gate_command(self.env, obs, "policy")
        except Exception:
            _log_gate_exc_once()
        feature_dim = getattr(Config.CURRENT, "num_goal_obs", 0)
        goal_features = build_track_goal_features(self.env, feature_dim)
        if goal_features is not None:
            obs = self.concatenate_terms(obs, goal_features)
        depth = nav_observation_utils.depth_camera_image(self.env)
        return self.concatenate_terms(obs, depth)
