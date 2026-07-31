#!/usr/bin/env python3
"""P3 Standard observations with private local navigation goals."""

import torch

from agent_ppo.feature.p2_observation_process import (
    P2PolicyObservationProcess,
    P2CriticObservationProcess,
)
from agent_ppo.feature import p2_contract, p3_contract
from agent_ppo.feature.p3_goal_provider import update_p3_goal_provider


class P3PolicyObservationProcess(P2PolicyObservationProcess):
    def process(self):
        update_p3_goal_provider(self.env, dt_s=float(getattr(self.env, "step_dt", 0.02)))
        return super().process()


class P3CriticObservationProcess(P2CriticObservationProcess):
    def process(self):
        update_p3_goal_provider(self.env, dt_s=float(getattr(self.env, "step_dt", 0.02)))
        wire = super().process()
        if not torch.is_tensor(wire) or wire.shape[1] != p2_contract.PRIVILEGED_WIRE_DIM:
            raise ValueError("P3 critic transport shape drift")
        event = torch.zeros(wire.shape[0], device=wire.device, dtype=wire.dtype)
        reached = getattr(self.env, "_p3_subgoal_reached", None)
        timed_out = getattr(self.env, "_p3_subgoal_timed_out", None)
        if torch.is_tensor(reached):
            event[reached.to(wire.device).bool()] = p3_contract.SUBGOAL_EVENT_REACHED
        if torch.is_tensor(timed_out):
            event[timed_out.to(wire.device).bool()] = p3_contract.SUBGOAL_EVENT_TIMEOUT
        event_index = (
            p2_contract.CRITIC_OBS_DIM + p2_contract.CURRENT_SEGMENT_INDEX
        )
        wire = wire.clone()
        wire[:, event_index] = event
        return wire
