#!/usr/bin/env python3
"""P4 observations: GoalBelief v2 policy goal and P3-compatible worker tail."""

from __future__ import annotations

import torch

from tools.base_env.observation_process import ObservationProcess

from agent_ppo.feature import nav_contract, nav_observation_utils, p2_contract
from agent_ppo.feature.goal_features import build_track_goal_raw
from agent_ppo.feature.p2_observation_process import P2CriticObservationProcess
from agent_ppo.feature.p2_worker_bridge import (
    install_p2_terminal_return_bridge,
    p2_response_aux,
)
from agent_ppo.feature.p4_goal_belief import GoalBeliefChainV2


class P4PolicyObservationProcess(ObservationProcess):
    target_group = "policy"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._goal_belief = None

    def process(self):
        env = self.env
        install_p2_terminal_return_bridge(env)
        obs = self.default_observation()
        if obs.shape[-1] != 301:
            raise ValueError(f"P4 policy base observation must be 301, got {obs.shape[-1]}")
        aux = p2_response_aux(env).to(device=obs.device, dtype=obs.dtype)
        is_eval = bool(getattr(env, "_is_eval", False))
        if is_eval:
            if self._goal_belief is None or self._goal_belief.num_envs != obs.shape[0]:
                self._goal_belief = GoalBeliefChainV2(obs.shape[0], obs.device, seed=4201)
            reset = aux[:, 24] > 0.5
            measured = aux[:, 6:9].clone()
            xy_valid = aux[:, 9] > 0.5
            measured[~xy_valid, :2] = 0.0
            true_xy = build_track_goal_raw(env).to(obs)
            goal4 = self._goal_belief.update(
                true_xy,
                measured,
                velocity_valid=xy_valid,
                dt_s=float(getattr(env, "step_dt", nav_contract.FRAME_DT_S)),
                reset_mask=reset,
                deterministic=True,
            )
        else:
            # Training owns the sole GoalBelief in AlgorithmP4NavPPO, where
            # session fault schedules and exact-resume RNG are available.  A
            # neutral placeholder prevents privileged raw goal leakage if the
            # algorithm-side replacement is ever skipped.
            goal4 = torch.zeros(obs.shape[0], 4, device=obs.device, dtype=obs.dtype)
        depth = nav_observation_utils.depth_camera_image(env)
        full = self.concatenate_terms(
            self.concatenate_terms(obs, goal4.to(obs.dtype)), depth.to(obs.dtype)
        )
        if full.shape[-1] != nav_contract.POLICY_OBS_DIM:
            raise ValueError(f"P4 policy observation shape drift: {tuple(full.shape)}")
        if is_eval:
            return p2_contract.pack_eval_response_aux(
                full, aux[:, : p2_contract.RESPONSE_AUX_DIM]
            )
        return full


__all__ = ["P4PolicyObservationProcess", "P2CriticObservationProcess"]
