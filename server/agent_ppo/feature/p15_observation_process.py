#!/usr/bin/env python3
"""P1.5 observation processors: policy57901 and privileged wire346."""

from __future__ import annotations

import torch

from tools.base_env.observation_process import ObservationProcess

from agent_ppo.feature import nav_observation_utils
from agent_ppo.feature.p15_worker_bridge import (
    apply_p15_worker_command,
    p15_response_aux,
)
from agent_ppo.feature import p15_contract


class P15PolicyObservationProcess(ObservationProcess):
    target_group = "policy"

    def process(self):
        apply_p15_worker_command(self.env)
        obs = self.default_observation()
        depth = nav_observation_utils.depth_camera_image(self.env)
        result = self.concatenate_terms(obs, depth)
        if result.ndim != 2 or result.shape[1] != 57901:
            raise ValueError(
                f"P1.5 policy observation must be [N,57901], got {tuple(result.shape)}"
            )
        return result


class P15CriticObservationProcess(ObservationProcess):
    target_group = "critic"

    def process(self):
        apply_p15_worker_command(self.env)
        critic = self.default_observation()
        aux = p15_response_aux(self.env).to(device=critic.device, dtype=critic.dtype)
        if critic.ndim != 2 or critic.shape[1] != p15_contract.CRITIC_OBS_DIM:
            raise ValueError(
                "P1.5 critic base observation must be [N,316], got "
                f"{tuple(critic.shape)}"
            )
        if aux.shape != (critic.shape[0], p15_contract.RESPONSE_AUX_DIM):
            raise ValueError(
                f"P1.5 response aux must be [N,30], got {tuple(aux.shape)}"
            )
        wire = torch.cat((critic, aux), dim=-1)
        if wire.shape[1] != p15_contract.PRIVILEGED_WIRE_DIM:
            raise AssertionError("P1.5 privileged transport dimension drift")
        return wire
