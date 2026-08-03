#!/usr/bin/env python3
"""P2 Track observations: nav policy57905 and critic323|worker_aux62 wire."""

from __future__ import annotations

import torch

from agent_ppo.feature.nav_observation_process import (
    NavPolicyObservationProcess,
    NavCriticObservationProcess,
)
from agent_ppo.feature.p2_worker_bridge import (
    get_p2_worker_bridge,
    install_p2_terminal_return_bridge,
    p2_response_aux,
    p3_training_extra,
    p4_training_extra,
)
from agent_ppo.feature import p2_contract, p3_contract, p4_contract


class P2PolicyObservationProcess(NavPolicyObservationProcess):
    def process(self):
        # ObservationProcess instances are created without an env while the
        # platform builds env_cfg. ObservationBridge binds the real Isaac env
        # immediately before this method runs, so this is the first reliable
        # pre-step installation point. The adapter itself is idempotent.
        install_p2_terminal_return_bridge(self.env)
        policy = super().process()
        if not bool(getattr(self.env, "_is_eval", False)):
            return policy
        aux = p2_response_aux(self.env).to(device=policy.device, dtype=policy.dtype)
        return p2_contract.pack_eval_response_aux(
            policy, aux[:, : p2_contract.RESPONSE_AUX_DIM]
        )


class P2CriticObservationProcess(NavCriticObservationProcess):
    def process(self):
        install_p2_terminal_return_bridge(self.env)
        self.env._p2_allow_scanner_gaps = True
        try:
            critic = super().process()
        finally:
            self.env._p2_allow_scanner_gaps = False
        aux = p2_response_aux(self.env).to(device=critic.device, dtype=critic.dtype)
        if critic.shape[1] != p2_contract.CRITIC_OBS_DIM:
            raise ValueError(f"P2 critic base must be [N,323], got {tuple(critic.shape)}")
        if aux.shape != (critic.shape[0], p2_contract.WORKER_AUX_DIM):
            raise ValueError(
                f"P2 worker aux must be [N,{p2_contract.WORKER_AUX_DIM}], "
                f"got {tuple(aux.shape)}"
            )
        wire = torch.cat((critic, aux), dim=-1)
        if wire.shape[1] != p2_contract.PRIVILEGED_WIRE_DIM:
            raise AssertionError("P2 privileged transport dimension drift")
        bridge = get_p2_worker_bridge(self.env)
        if bridge is not None and bridge.runtime_stage_type in {
            "p3_standard_joint",
            "p4_nav_ppo",
        }:
            extra = p3_training_extra(self.env).to(critic)
            wire = torch.cat((wire, extra), dim=-1)
            if wire.shape[1] != p3_contract.P3_PRIVILEGED_WIRE_DIM:
                raise AssertionError("P3 privileged transport dimension drift")
            if bridge.runtime_stage_type == "p4_nav_ppo":
                p4_extra = p4_training_extra(self.env).to(critic)
                wire = torch.cat((wire, p4_extra), dim=-1)
                if wire.shape[1] != p4_contract.P4_PRIVILEGED_WIRE_DIM:
                    raise AssertionError("P4 privileged transport dimension drift")
        return wire
