#!/usr/bin/env python3

import sys
import types

import torch


tools_utils = types.ModuleType("tools.utils")
tools_utils.load_reward_keys_from_monitor_config = lambda: []
sys.modules.setdefault("tools.utils", tools_utils)

from agent_ppo.workflow.train_workflow import _split_and_record_p15_transport


class _Agent:
    is_p15_response = True

    def __init__(self):
        self.records = []

    @staticmethod
    def split_p15_transport(wire):
        return wire[:, :316], wire[:, 316:]

    def observe_response_aux(self, aux, dones):
        self.records.append((aux.clone(), dones.clone()))


def test_reset_and_step_paths_split_wire_before_ppo_storage():
    agent = _Agent()
    wire = torch.randn(4, 346)
    reset_critic = _split_and_record_p15_transport(agent, wire)
    assert reset_critic.shape == (4, 316)
    assert torch.equal(agent.records[0][0], wire[:, 316:])
    assert not bool(agent.records[0][1].any())

    dones = torch.tensor([False, True, False, True])
    step_critic = _split_and_record_p15_transport(agent, wire, dones)
    assert step_critic.shape == (4, 316)
    assert torch.equal(agent.records[1][1], dones)
