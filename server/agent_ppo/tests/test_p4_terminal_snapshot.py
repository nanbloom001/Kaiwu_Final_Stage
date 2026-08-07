#!/usr/bin/env python3
"""Training-only terminal snapshot regression tests."""

import types
from types import SimpleNamespace

import torch

from agent_ppo.feature import p2_contract, p4_contract
from agent_ppo.feature.p2_worker_bridge import P2WorkerBridge


class _EventManager:
    def __init__(self, func):
        self.cfg = SimpleNamespace(func=func)

    def get_term_cfg(self, name):
        assert name == "reset_base"
        return self.cfg

    def set_term_cfg(self, name, cfg):
        assert name == "reset_base"
        self.cfg = cfg


def _bridge_for_snapshot():
    data = SimpleNamespace(
        root_lin_vel_b=torch.tensor(((0.7, -0.2, 0.0), (0.1, 0.2, 0.0))),
        root_ang_vel_b=torch.tensor(((0.1, 0.2, 0.8), (0.0, 0.0, 0.3))),
        projected_gravity_b=torch.tensor(((0.2, -0.1, -0.9), (0.0, 0.0, -1.0))),
        root_pos_w=torch.tensor(((4.0, 5.0, 0.4), (1.0, 2.0, 0.4))),
        root_quat_w=torch.tensor(((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))),
    )
    robot = SimpleNamespace(data=data)
    scene = SimpleNamespace(robot=robot, terrain=None)

    def reset_base(env, env_ids, _pose_range, _velocity_range):
        env.scene.robot.data.root_lin_vel_b[env_ids] = 0.0
        env.scene.robot.data.root_ang_vel_b[env_ids] = 0.0
        env.scene.robot.data.root_pos_w[env_ids, :2] = -9.0

    env = SimpleNamespace(
        scene=scene,
        common_step_counter=7,
        event_manager=_EventManager(reset_base),
        termination_manager=SimpleNamespace(
            active_terms=("goal_reached",),
            terminated=torch.tensor((True, False)),
            time_outs=torch.zeros(2, dtype=torch.bool),
            get_term=lambda name: torch.tensor((name == "goal_reached", False)),
        ),
    )
    bridge = object.__new__(P2WorkerBridge)
    bridge.env = env
    bridge.runtime_stage_type = "p4_nav_ppo"
    bridge.num_envs = 2
    bridge.device = torch.device("cpu")
    bridge.last_aux = torch.zeros(2, p2_contract.WORKER_AUX_DIM)
    bridge._terminal_aux_snapshot = torch.zeros_like(bridge.last_aux)
    bridge._terminal_snapshot_mask = torch.zeros(2, dtype=torch.bool)
    bridge._terminal_snapshot_reason = torch.zeros(2, dtype=torch.long)
    bridge._terminal_snapshot_hook_installed = False
    bridge._wall_stuck_precedes_success = False
    bridge.last_p4_extra = torch.zeros(2, p4_contract.P4_WORKER_EXTRA_DIM)
    bridge._terminal_p4_extra_snapshot = torch.zeros_like(bridge.last_p4_extra)
    bridge._terminal_p4_snapshot_mask = torch.zeros(2, dtype=torch.bool)
    bridge._p4_stuck_tracker = None
    contact_forces = torch.zeros(2, 6, 3)
    contact_forces[0, 4, 0] = 42.0
    bridge._gait_window = SimpleNamespace(
        valid=True,
        collision_valid=True,
        sensor=SimpleNamespace(
            data=SimpleNamespace(net_forces_w=contact_forces)
        ),
        sensor_foot_ids=torch.tensor((0, 1, 2, 3)),
        sensor_body_names=("FL", "FR", "RL", "RR", "base", "trunk"),
        disable_collision=lambda _reason: None,
    )
    bridge._terrain_metadata = types.MethodType(
        lambda self: (torch.tensor((2.0, 3.0)), torch.tensor((4.0, 5.0))),
        bridge,
    )
    bridge._goal_distance = types.MethodType(
        lambda self, robot: torch.tensor((1.25, 2.5)), bridge
    )
    return bridge


def test_reset_hook_freezes_terminal_aux_before_reset_mutates_robot_state(monkeypatch):
    bridge = _bridge_for_snapshot()
    monkeypatch.setattr(
        "agent_ppo.feature.p2_worker_bridge.build_track_goal_raw",
        lambda _env: torch.tensor(((6.0, 7.0), (8.0, 9.0))),
    )
    assert bridge._install_training_terminal_snapshot_hook() is True
    bridge.env.event_manager.cfg.func(
        bridge.env, torch.tensor((0,)), {}, {}
    )

    snapshot = bridge._terminal_aux_snapshot[0]
    torch.testing.assert_close(snapshot[12:15], torch.tensor((0.7, -0.2, 0.8)))
    torch.testing.assert_close(snapshot[15:18], torch.tensor((4.0, 5.0, 0.0)))
    assert snapshot[p2_contract.PRE_STEP_GOAL_DISTANCE_INDEX].item() == 1.25
    assert snapshot[p2_contract.BODY_COLLISION_FORCE_INDEX].item() == 42.0
    assert snapshot[p2_contract.GAIT_SENSOR_MAPPING_VALID_INDEX].item() == 1.0
    assert (
        snapshot[p2_contract.BODY_COLLISION_MAPPING_VALID_INDEX].item() == 1.0
    )
    assert snapshot[24].item() == 1.0
    assert snapshot[25].item() == 1.0
    torch.testing.assert_close(
        bridge._terminal_p4_extra_snapshot[0, :2], torch.tensor((6.0, 7.0))
    )
    assert (
        bridge._terminal_p4_extra_snapshot[
            0, p4_contract.STUCK_RAW_TERM_INDEX
        ].item()
        == 0.0
    )
    assert bridge._terminal_p4_snapshot_mask[0].item() is True
    assert bridge.env.scene.robot.data.root_pos_w[0, 0].item() == -9.0


def test_terminal_snapshot_replaces_only_captured_reset_rows():
    bridge = _bridge_for_snapshot()
    bridge._terminal_aux_snapshot[0].fill_(7.0)
    bridge._terminal_snapshot_reason[0] = 2
    bridge._terminal_snapshot_mask[0] = True
    live = torch.zeros(2, p2_contract.WORKER_AUX_DIM)
    reason = torch.zeros(2)

    merged, merged_reason = bridge._apply_training_terminal_snapshot(
        live,
        torch.tensor((True, True)),
        reason,
        step=11,
    )
    assert merged[0, 12].item() == 7.0
    assert merged[0, 24].item() == 1.0
    assert merged[0, 25].item() == 2.0
    assert merged[0, 26].item() == 11.0
    assert torch.equal(merged[1], torch.zeros_like(merged[1]))
    assert merged_reason.tolist() == [2.0, 0.0]
    assert not bridge._terminal_snapshot_mask.any()


def test_terminal_p4_snapshot_replaces_only_terminal_safe_fields():
    bridge = _bridge_for_snapshot()
    bridge._terminal_p4_extra_snapshot[
        0, p4_contract.RAW_GOAL_XY_SLICE
    ] = torch.tensor((7.0, 8.0))
    bridge._terminal_p4_extra_snapshot[
        0, p4_contract.STUCK_RAW_TERM_INDEX
    ] = 1.0
    bridge._terminal_p4_snapshot_mask[0] = True
    live = torch.full_like(bridge.last_p4_extra, 99.0)

    merged = bridge._apply_training_terminal_p4_snapshot(
        live, torch.tensor((True, True))
    )
    torch.testing.assert_close(
        merged[0, p4_contract.RAW_GOAL_XY_SLICE], torch.tensor((7.0, 8.0))
    )
    assert merged[0, p4_contract.STUCK_RAW_TERM_INDEX].item() == 1.0
    untouched = torch.ones(p4_contract.P4_WORKER_EXTRA_DIM, dtype=torch.bool)
    untouched[p4_contract.RAW_GOAL_XY_SLICE] = False
    untouched[p4_contract.STUCK_RAW_TERM_INDEX] = False
    assert torch.all(merged[0, untouched] == 99.0)
    assert torch.all(merged[1] == 99.0)
    assert not bridge._terminal_p4_snapshot_mask.any()
