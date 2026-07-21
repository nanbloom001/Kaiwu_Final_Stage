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
from agent_ppo.feature.goal_noise import GoalNoiseAugmenter
from agent_ppo.feature.hard_start_replay import (
    initialize_hard_start_replay,
    install_hard_start_replay_event,
    publish_hard_start_metrics,
)
from tools.base_env.observation_process import ObservationProcess


class _SilentConfigLogger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class PolicyObservationProcess(ObservationProcess):
    target_group = "policy"
    _BASE_OBS_DIM = 301

    class _BridgeProxy:
        def __init__(self, bridge, hard_start_config):
            self._bridge = bridge
            self._hard_start_config = hard_start_config

        def __getattr__(self, name):
            return getattr(self._bridge, name)

        def override_group_in_env_cfg(self, env_cfg):
            install_hard_start_replay_event(env_cfg, self._hard_start_config)
            return self._bridge.override_group_in_env_cfg(env_cfg)

    def create_bridge(self):
        """Install the training-only reset hook before gym creates the env."""
        bridge = super().create_bridge()
        usr_conf, _, is_eval, _ = Config.load_conf(_SilentConfigLogger())
        hard_start_config = {} if is_eval else usr_conf.get("hard_start_replay", {})
        return self._BridgeProxy(bridge, hard_start_config)

    def _goal_noise_config(self):
        for source in (self.env, getattr(self.env, "unwrapped", None)):
            usr_conf = getattr(source, "usr_conf", None)
            if isinstance(usr_conf, dict):
                return usr_conf.get("goal_noise", {})

        usr_conf, _, _, _ = Config.load_conf(_SilentConfigLogger())
        return usr_conf.get("goal_noise", {})

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
                config=self._goal_noise_config(),
            )
        actor_raw_goal = self.goal_noise_augmenter.apply(raw_goal)
        return encode_track_goal(actor_raw_goal)

    def process(self):
        # This code runs in the Isaac worker, where the real environment is
        # available. Initialize here rather than trying to unwrap Kaiwu's
        # cross-process proxy in the training workflow.
        initialize_hard_start_replay(self.env)
        publish_hard_start_metrics(self.env)
        obs = self.default_observation()
        if obs.shape[-1] != self._BASE_OBS_DIM:
            raise ValueError(
                f"Policy observation dim mismatch: expected base {self._BASE_OBS_DIM}, got {obs.shape[-1]}."
            )

        goal_features = self._goal_features()
        if goal_features is not None:
            obs = self.concatenate_terms(obs, goal_features)
        return obs
