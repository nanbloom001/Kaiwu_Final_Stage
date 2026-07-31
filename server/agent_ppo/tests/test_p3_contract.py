#!/usr/bin/env python3

from types import SimpleNamespace
from pathlib import Path
import tomllib

import torch
import pytest

from agent_ppo.checkpoint_io import p3_standard_joint_candidates
from agent_ppo.feature import p3_contract as p3
from agent_ppo.feature.goal_features import build_track_goal_raw
from agent_ppo.feature.p2_worker_bridge import P2WorkerBridge, _terminal_safe_root_pose
from agent_ppo.feature.p3_goal_provider import (
    StandardFarGoalProvider,
    p3_local_out_of_bounds,
)


def test_low_level_adapter_removes_only_goal4():
    obs = torch.arange(p3.POLICY_OBS_DIM, dtype=torch.float32).reshape(1, -1)
    low, goal4, depth = p3.split_policy_observation(obs)
    assert low.shape == (1, 57901)
    assert torch.equal(low[:, :301], obs[:, :301])
    assert torch.equal(low[:, 301:], obs[:, 305:])
    assert torch.equal(goal4, obs[:, 301:305])
    assert torch.equal(depth, obs[:, 305:])


def test_subgoals_stay_inside_tile_inner_boundary():
    root = torch.tensor([[2.9, 2.9], [-2.9, -2.9], [0.0, 0.0]])
    origin = torch.zeros_like(root)
    goals = p3.sample_local_subgoals(root, origin, generator=torch.Generator().manual_seed(17))
    assert bool((goals.abs() <= 3.0 + 1.0e-6).all())
    distances = torch.linalg.vector_norm(goals - root, dim=-1)
    assert bool(((distances >= 1.5) & (distances <= 2.8)).all())


def test_materialized_domain_randomization_uses_phase_values():
    config = {
        "p3_standard_joint": {
            "domain_randomization": {
                "friction_ranges": [[0.6, 1.3], [0.55, 1.35]],
                "base_added_mass_kg": [0.75, 0.85],
                "noise_levels": [0.50, 0.55],
            }
        }
    }
    realized = p3.materialize_environment_config(config, 1800.0)
    assert realized["domain_rand"]["friction_range"] == [0.55, 1.35]
    assert realized["domain_rand"]["added_mass_range"] == [-0.85, 0.85]
    assert realized["domain_rand"]["push_robots"] is False
    assert realized["p3_runtime"]["worker_command_override"] is False
    assert realized["noise"]["noise_level"] == 0.55


def test_success_contracts_are_independent():
    reached = p3.subgoal_reached(
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[0.5, 0.0], [0.7, 0.0]]),
    )
    assert reached.tolist() == [True, False]
    joint = p3.joint_episode_success(
        torch.tensor([False, True, True]),
        torch.tensor([5, 1, 2]),
        torch.tensor([False, False, False]),
    )
    assert joint.tolist() == [False, False, True]


def test_local_bounds_and_phase_boundaries():
    origin = torch.tensor([[10.0, -5.0], [10.0, -5.0]])
    root = torch.tensor([[13.2, -5.0], [13.2001, -5.0]])
    assert p3.local_out_of_bounds(root, origin).tolist() == [False, True]
    assert p3.phase_for_elapsed(0).name == "lowbase"
    assert p3.phase_for_elapsed(1800).name == "lowmild"
    assert p3.phase_for_elapsed(3600).name == "lowmedium"
    assert p3.phase_for_elapsed(5400).name == "adaptercalib"
    assert p3.phase_for_elapsed(6000).name == "highadapt"
    assert p3.phase_for_elapsed(7200).name == "highslow"


class _Scene:
    def __init__(self, robot, origins):
        self.robot = robot
        self.env_origins = origins
        self.terrain = SimpleNamespace(env_origins=origins)

    def __getitem__(self, name):
        assert name == "robot"
        return self.robot


def test_goal_provider_keeps_success_visible_for_one_observation():
    root = torch.tensor([[0.0, 0.0, 0.35]])
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        common_step_counter=0,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
    )
    env.scene = _Scene(
        SimpleNamespace(data=SimpleNamespace(root_pos_w=root)),
        torch.zeros(1, 3),
    )
    provider = StandardFarGoalProvider(env, seed=7)
    provider.update(dt_s=0.02)
    first_goal = provider.goal_xy.clone()
    first_direction = provider.direction_angle.clone()
    env.episode_length_buf.fill_(1)
    env.scene.robot.data.root_pos_w[0, :2] = first_goal[0]
    env.common_step_counter = 1
    provider.update(dt_s=0.02)
    assert bool(env._p3_subgoal_reached[0])
    assert torch.equal(provider.goal_xy, first_goal)
    env.common_step_counter = 2
    provider.update(dt_s=0.02)
    assert not torch.equal(provider.goal_xy, first_goal)
    assert torch.equal(provider.direction_angle, first_direction)
    assert provider.milestone_index.tolist() == [1]


def test_goal_timeout_replans_only_with_allowed_angular_offsets():
    root = torch.tensor([[0.0, 0.0, 0.35]])
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        common_step_counter=0,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
    )
    env.scene = _Scene(
        SimpleNamespace(data=SimpleNamespace(root_pos_w=root)),
        torch.zeros(1, 3),
    )
    provider = StandardFarGoalProvider(env, seed=11)
    provider.update(dt_s=0.02)
    base = provider.base_direction_angle.clone()
    provider.goal_timeout_s.zero_()
    env.episode_length_buf.fill_(1)
    env.common_step_counter = 1
    provider.update(dt_s=0.02)
    env.common_step_counter = 2
    provider.update(dt_s=0.02)
    offset = torch.rad2deg(
        torch.atan2(
            torch.sin(provider.direction_angle - base),
            torch.cos(provider.direction_angle - base),
        )
    ).abs()
    assert offset.item() == pytest.approx(30.0) or offset.item() == pytest.approx(60.0)


def test_invalid_subgoal_uses_non_rewarding_tile_recovery_target():
    root = torch.tensor([[10.0, 10.0, 0.35]])
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        common_step_counter=0,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
    )
    env.scene = _Scene(
        SimpleNamespace(data=SimpleNamespace(root_pos_w=root)),
        torch.zeros(1, 3),
    )
    provider = StandardFarGoalProvider(env, seed=7)
    provider.update(dt_s=0.02)
    assert bool(provider.sample_invalid[0])
    assert torch.equal(provider.goal_xy[0], provider.episode_origin_xy[0])
    assert not bool(env._p3_subgoal_reached[0])


def test_p3_out_of_bounds_diagnostic_reads_current_root_position():
    root = torch.tensor([[0.0, 0.0, 0.35]])
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        common_step_counter=0,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
    )
    env.scene = _Scene(
        SimpleNamespace(data=SimpleNamespace(root_pos_w=root)),
        torch.zeros(1, 3),
    )
    provider = StandardFarGoalProvider(env, seed=13)
    provider.update(dt_s=0.02)
    env._p3_goal_provider = provider
    env.scene.robot.data.root_pos_w[0, 0] = 3.21
    assert p3_local_out_of_bounds(env).tolist() == [True]


def test_p3_private_goal_overrides_native_standard_goal():
    root = torch.tensor([[0.0, 0.0, 0.35]])
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        goal_positions=torch.tensor([[9.0, 0.0, 0.35]]),
        _p3_goal_positions=torch.tensor([[0.0, 2.0, 0.35]]),
    )
    env.scene = _Scene(
        SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=root,
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            )
        ),
        torch.zeros(1, 3),
    )
    local_goal = build_track_goal_raw(env)
    assert torch.allclose(local_goal, torch.tensor([[0.0, 2.0]]))

    bridge = P2WorkerBridge.__new__(P2WorkerBridge)
    bridge.env = env
    bridge.num_envs = 1
    bridge.device = torch.device("cpu")
    distance = bridge._goal_distance(env.scene.robot)
    assert torch.allclose(distance, torch.tensor([2.0]))


def test_p3_candidates_prefer_exact_resume_then_p2_parent(tmp_path):
    exact = tmp_path / "model.ckpt-highslow-9.pkl"
    parent = tmp_path / "model.ckpt-safestable-648278.pkl"
    exact.write_bytes(b"x")
    parent.write_bytes(b"x")
    candidates = p3_standard_joint_candidates(
        str(tmp_path), "9", parent_model_id="648278"
    )
    assert candidates.index(str(exact)) < candidates.index(str(parent))


def test_p3_enables_shared_worker_transport(monkeypatch):
    from agent_ppo.conf.conf import Config
    from agent_ppo.feature import p2_worker_bridge

    class Stage:
        algorithm = "p3_standard_joint"

    monkeypatch.setattr(
        Config,
        "load_conf",
        classmethod(
            lambda cls, logger: (
                {
                    "env_conf": {"seed": 17},
                    "p3_standard_joint": {"run_name": "p3-test"},
                },
                "train",
                False,
                Stage(),
            )
        ),
    )
    enabled, config, seed = p2_worker_bridge._resolve_config()
    assert enabled is True
    assert config["run_name"] == "p3-test"
    assert config["_worker_stage_type"] == "p3_standard_joint"
    assert seed == 17


def test_p3_production_config_and_monitor_are_standard_specific():
    root = Path(__file__).parents[1]
    with (root / "conf/train_env_conf_standard_p3_standard_joint.toml").open(
        "rb"
    ) as stream:
        config = tomllib.load(stream)
    assert config["env"] == {
        "num_envs": 128,
        "episode_length_s": 25.0,
        "task": "standard",
    }
    assert config["terrain"]["mode"] == "standard"
    assert config["terrain"]["curriculum"] is False
    assert config["p3_standard_joint"]["run_name"] == "p3std2h30-s2r-radial"
    assert config["p3_standard_joint"]["target_effective_seconds"] == 9000
    monitor_source = (root / "conf/monitor_builder.py").read_text()
    p3_builder = monitor_source.split("def _build_p3_monitor():", 1)[1].split(
        "def build_monitor():", 1
    )[0]
    assert "p2_curriculum" not in p3_builder
    assert "p2_track_segments" not in p3_builder
    assert "p3_command_feedback" in p3_builder
    assert "p3_radial_milestones" in p3_builder
    assert "p3_frontier_clawback" not in p3_builder


def test_platform_completion_radii_keep_a_boundary_margin():
    complete, target, boundary = p3.platform_completion_radii(8.0)
    assert complete == pytest.approx(3.90)
    assert target == pytest.approx(3.93)
    assert boundary == pytest.approx(4.00)
    assert complete < target < boundary


def test_p3_terminal_transport_keeps_pre_reset_root_pose():
    live = torch.tensor([[0.0, 0.0, 0.1], [1.0, 2.0, 0.2]])
    previous = torch.tensor([[3.91, 0.0, 0.0], [0.5, 0.5, 0.1]])
    reset = torch.tensor([True, False])
    selected = _terminal_safe_root_pose(live, previous, reset, enabled=True)
    assert torch.equal(selected[0], previous[0])
    assert torch.equal(selected[1], live[1])
    assert torch.equal(
        _terminal_safe_root_pose(live, previous, reset, enabled=False), live
    )


def test_radial_new_best_cannot_be_replayed_at_the_same_radius():
    radius = torch.tensor([1.25, 1.25, 1.40])
    best = torch.tensor([1.00, 1.25, 1.35])
    reward, next_best = p3.radial_new_best(radius, best)
    assert reward.tolist() == pytest.approx([0.8, 0.0, 0.2])
    replay, replay_best = p3.radial_new_best(radius, next_best)
    assert torch.equal(replay, torch.zeros_like(replay))
    assert torch.equal(replay_best, next_best)


def test_radial_soft_brake_caps_only_outward_motion():
    command = torch.tensor([[1.0, 0.0, 0.2], [-0.4, 0.0, -0.2], [0.8, 0.0, 0.1]])
    root = torch.tensor([[3.72, 0.0], [3.88, 0.0], [3.91, 0.0]])
    origin = torch.zeros_like(root)
    guarded, hold = p3.cap_outward_body_command(
        command, root, origin, torch.zeros(3), 3.90
    )
    assert guarded[0, 0].item() == pytest.approx(0.30)
    assert guarded[1, 0].item() == pytest.approx(-0.40)
    assert torch.equal(guarded[2], torch.zeros(3))
    assert hold.tolist() == [False, False, True]
