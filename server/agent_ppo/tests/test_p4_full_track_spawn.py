"""Focused CPU coverage for P4 full-track spawn and physical stuck state."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from agent_ppo.feature.p2_worker_bridge import _p4_spawn_wire_state
from agent_ppo.feature.p4_spawn import (
    P4FullTrackSpawnController,
    P4SpawnQuotaState,
    install_p4_full_track_spawn,
)
from agent_ppo.feature.p4_stuck import MotionWallStuckTracker


SPAWN_CONFIG = {
    "enabled": True,
    "phase_boundaries": [7200, 21600],
    "full_start_share": [0.70, 0.60, 0.75],
    "random_segment_weights": [0.15, 0.15, 0.20, 0.20, 0.30],
    "safe_point_share": 0.70,
    "quartiles": 4,
    "slope_yaw_abs": 0.35,
    "maze_yaw_abs": 0.50,
    "max_reason4_same_bucket_retries": 2,
}


def test_spawn_quota_schedule_rng_roundtrip_and_five_segment_balance():
    state = P4SpawnQuotaState(128, SPAWN_CONFIG, seed=17)
    assert not bool(state.last_full.any())
    assert bool((state.last_segment == -1).all())
    assert not bool(state.last_safe.any())
    ids = torch.arange(128)
    for _ in range(80):
        assignment = state.assign(ids, elapsed_s=0.0)
        assert assignment.segment[~assignment.full_start].min().item() >= 0
        assert assignment.segment[~assignment.full_start].max().item() < 5
        assert assignment.quartile[~assignment.full_start].max().item() < 4

    full_share = state.phase_full[0].item() / state.phase_total[0].item()
    assert abs(full_share - 0.70) < 0.002
    actual_segments = state.segment_counts[0].float()
    actual_segments /= actual_segments.sum()
    assert torch.allclose(
        actual_segments,
        torch.tensor(SPAWN_CONFIG["random_segment_weights"]),
        atol=0.002,
    )

    saved = state.state_dict()
    resumed = P4SpawnQuotaState(128, SPAWN_CONFIG, seed=999)
    resumed.load_state_dict(saved)
    expected = state.assign(ids[:16], elapsed_s=8000.0)
    actual = resumed.assign(ids[:16], elapsed_s=8000.0)
    for field in expected.__dataclass_fields__:
        assert torch.equal(getattr(expected, field), getattr(actual, field))


def test_reason4_retries_same_segment_bucket_twice_then_falls_back_to_entry():
    config = dict(SPAWN_CONFIG)
    config["full_start_share"] = [0.0, 0.0, 0.0]
    state = P4SpawnQuotaState(1, config, seed=23)
    first = state.assign(torch.tensor([0]), elapsed_s=0.0)
    bucket = (first.segment.item(), first.quartile.item(), first.safe_point.item())
    for expected_retry in (1, 2):
        retry = state.assign(
            torch.tensor([0]), elapsed_s=0.0, terminal_reason=torch.tensor([4])
        )
        assert retry.reason4_retry.item()
        assert (retry.segment.item(), retry.quartile.item(), retry.safe_point.item()) == bucket
        assert state.reason4_retries[0].item() == expected_retry
    exhausted = state.assign(
        torch.tensor([0]), elapsed_s=0.0, terminal_reason=torch.tensor([4])
    )
    assert not exhausted.reason4_retry.item()
    assert exhausted.reason4_exhausted.item()
    assert exhausted.segment.item() == bucket[0]
    assert exhausted.quartile.item() == 0
    assert exhausted.safe_point.item()
    assert state.reason4_retries[0].item() == 0


def test_spawn_wire_state_composes_after_moving_cpu_quota_tensors():
    quota = SimpleNamespace(
        last_full=torch.tensor([True, False]),
        last_segment=torch.tensor([-1, 3]),
        last_quartile=torch.tensor([0, 2]),
        last_safe=torch.tensor([True, False]),
    )
    full, segment, quartile, safe = _p4_spawn_wire_state(
        quota, torch.device("cpu")
    )
    assert full.tolist() == [True, False]
    assert segment.tolist() == [0, 3]
    assert quartile.tolist() == [0, 2]
    assert safe.tolist() == [True, False]


class _Robot:
    def __init__(self, num_envs):
        self.data = SimpleNamespace(default_root_state=torch.zeros(num_envs, 13))
        self.data.default_root_state[:, 2] = 0.35
        self.pose_writes = []
        self.velocity_writes = []

    def write_root_pose_to_sim(self, value, env_ids):
        self.pose_writes.append((env_ids.clone(), value.clone()))

    def write_root_velocity_to_sim(self, value, env_ids):
        self.velocity_writes.append((env_ids.clone(), value.clone()))


class _Scene:
    def __init__(self, num_envs):
        origins = torch.zeros(5, 2, 3)
        origins[:, :, 0] = torch.arange(5).view(-1, 1) * 8.0
        origins[:, :, 1] = torch.arange(2).view(1, -1) * 8.0
        generator = SimpleNamespace(
            size=(8.0, 8.0),
            sub_terrains_order=[
                "pyramid_slope",
                "pyramid_slope_inv",
                "pyramid_stairs",
                "pyramid_stairs_inv",
                "open_entry_maze",
            ],
            terrain_spawn_positions=None,
        )
        self.terrain = SimpleNamespace(
            terrain_origins=origins,
            terrain_types=torch.arange(num_envs) % 2,
            terrain_levels=torch.zeros(num_envs, dtype=torch.long),
            env_origins=torch.zeros(num_envs, 3),
            cfg=SimpleNamespace(terrain_generator=generator),
        )
        self.env_origins = torch.zeros(num_envs, 3)
        self.robot = _Robot(num_envs)

    def __getitem__(self, name):
        if name != "robot":
            raise KeyError(name)
        return self.robot


class _EventManager:
    def __init__(self, reset_cfg):
        self.cfg = reset_cfg

    def get_term_cfg(self, name):
        assert name == "reset_base"
        return self.cfg

    def set_term_cfg(self, name, cfg):
        assert name == "reset_base"
        self.cfg = cfg


def _spawn_env(num_envs=16):
    scene = _Scene(num_envs)

    def original(env, env_ids, velocity_range, pose_range=None, asset_cfg=None):
        env.original_calls.append(
            (env_ids.clone(), bool(getattr(env, "_is_eval", False)))
        )

    reset_cfg = SimpleNamespace(
        func=original,
        params={"velocity_range": {}, "pose_range": {}},
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        scene=scene,
        event_manager=_EventManager(reset_cfg),
        termination_manager=None,
        _is_eval=False,
        original_calls=[],
        extras={},
    )
    return env


def test_runtime_reset_wrapper_fails_closed_to_segment_entry_and_reports_truth():
    env = _spawn_env()
    config = dict(SPAWN_CONFIG)
    config["full_start_share"] = [0.0, 0.0, 0.0]
    config["safe_point_share"] = 0.0
    controller = install_p4_full_track_spawn(env, config, seed=31)
    assert isinstance(controller, P4FullTrackSpawnController)
    assert controller.installed

    ids = torch.arange(env.num_envs)
    env.event_manager.cfg.func(env, ids, {}, {})
    diagnostics = controller.diagnostics()
    assert diagnostics["segment_start_count"] == env.num_envs
    assert diagnostics["all_position_requested_count"] == env.num_envs
    assert diagnostics["all_position_spawn_active"] == 0
    assert diagnostics["segment_entry_fallback_count"] == env.num_envs
    assert diagnostics["spawn_write_failure_count"] == 0
    assert len(env.scene.robot.pose_writes) == 1
    assert len(env.scene.robot.velocity_writes) == 1
    written_ids, poses = env.scene.robot.pose_writes[0]
    rows = env.scene.terrain.terrain_levels[written_ids]
    cols = env.scene.terrain.terrain_types[written_ids]
    origins = env.scene.terrain.terrain_origins[rows, cols]
    assert torch.allclose(poses[:, :2], origins[:, :2])
    assert torch.isfinite(poses).all()


def test_validated_all_position_maze_spawn_is_not_forced_back_to_entry():
    env = _spawn_env(8)
    env._agent_ppo_p4_spawn_surface_query = lambda xy: (
        torch.full((xy.shape[0],), 0.2),
        torch.ones(xy.shape[0], dtype=torch.bool),
    )
    config = dict(SPAWN_CONFIG)
    config["full_start_share"] = [0.0, 0.0, 0.0]
    config["safe_point_share"] = 0.0
    config["random_segment_weights"] = [0.0, 0.0, 0.0, 0.0, 1.0]
    controller = install_p4_full_track_spawn(env, config, seed=43)
    ids = torch.arange(env.num_envs)
    env.event_manager.cfg.func(env, ids, {}, {})

    diagnostics = controller.diagnostics()
    assert diagnostics["all_position_applied_count"] == env.num_envs
    written_ids, poses = env.scene.robot.pose_writes[0]
    rows = env.scene.terrain.terrain_levels[written_ids]
    cols = env.scene.terrain.terrain_types[written_ids]
    origins = env.scene.terrain.terrain_origins[rows, cols]
    assert torch.all(rows == 4)
    assert bool((poses[:, :2] != origins[:, :2]).any())


def test_spawn_writer_failure_is_fatal_and_counted():
    env = _spawn_env(1)
    env._agent_ppo_p4_spawn_surface_query = lambda xy: (
        torch.full((xy.shape[0],), 0.2),
        torch.ones(xy.shape[0], dtype=torch.bool),
    )
    env.scene.robot.write_root_pose_to_sim = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("writer failed")
    )
    config = dict(SPAWN_CONFIG)
    config["full_start_share"] = [0.0, 0.0, 0.0]
    controller = install_p4_full_track_spawn(env, config, seed=47)

    with pytest.raises(RuntimeError, match="could not write"):
        env.event_manager.cfg.func(env, torch.tensor([0]), {}, {})
    assert controller.diagnostics()["spawn_write_failure_count"] == 1


class _StuckTerminationManager:
    active_terms = ("nav_stuck_timeout",)

    def __init__(self, num_envs):
        self.cfg = SimpleNamespace(time_out=True, params={"max_stuck": 1})
        self.value = torch.zeros(num_envs, dtype=torch.bool)

    def get_term_cfg(self, _name):
        return self.cfg

    def set_term_cfg(self, _name, cfg):
        self.cfg = cfg

    def get_term(self, _name):
        return self.value


def _stuck_update(tracker, *, x=0.0, wall=True, speed=0.0, reset=False):
    return tracker.update(
        root_xy=torch.tensor([[x, 0.0]]),
        goal_distance=torch.tensor([1.2]),
        collision_force=torch.tensor([40.0 if wall else 0.0]),
        mapping_valid=torch.tensor([True]),
        reset=torch.tensor([reset]),
        terminal_reason=torch.tensor([4.0 if reset else 0.0]),
        seconds_since_push=torch.tensor([10.0]),
        episode_age_s=torch.tensor([10.0]),
        true_velocity3=torch.tensor([[speed, 0.0, 0.0]]),
    )


def test_sliding_spatial_diameter_triggers_without_command_intent():
    manager = _StuckTerminationManager(1)
    env = SimpleNamespace(step_dt=0.02, termination_manager=manager)
    tracker = MotionWallStuckTracker(
        env,
        num_envs=1,
        device="cpu",
        config={"mode": "active", "confirmation_s": 0.10},
    )
    for x in (0.00, 0.10, 0.20, 0.30, 0.49):
        diagnostics = _stuck_update(tracker, x=x)
    assert diagnostics[0, 0].item() == 1.0
    assert diagnostics[0, 4].item() == 1.0
    assert diagnostics[0, 12].item() == 0.0
    assert getattr(env, "_nav_motion_stuck").item() >= tracker.confirmation_steps

    moving = MotionWallStuckTracker(
        SimpleNamespace(step_dt=0.02, termination_manager=_StuckTerminationManager(1)),
        num_envs=1,
        device="cpu",
        config={"mode": "shadow", "confirmation_s": 0.10},
    )
    for x in (0.00, 0.20, 0.40, 0.60, 0.80):
        moving_diagnostics = _stuck_update(moving, x=x)
    assert moving_diagnostics[0, 0].item() == 0.0
    assert moving_diagnostics[0, 4].item() == 0.0


def test_recovery_requires_wall_absence_and_window_center_displacement():
    tracker = MotionWallStuckTracker(
        SimpleNamespace(step_dt=0.02, termination_manager=_StuckTerminationManager(1)),
        num_envs=1,
        device="cpu",
        config={"mode": "shadow", "confirmation_s": 0.10},
    )
    for _ in range(5):
        diagnostics = _stuck_update(tracker, x=0.0, wall=True)
    assert diagnostics[0, 2].item() == 1.0

    for _ in range(tracker.recovery_wall_absence_steps):
        diagnostics = _stuck_update(tracker, x=0.0, wall=False)
    assert diagnostics[0, 2].item() == 1.0

    for _ in range(tracker.confirmation_steps):
        diagnostics = _stuck_update(tracker, x=0.60, wall=False, speed=0.20)
    assert diagnostics[0, 2].item() == 0.0
    assert tracker.wall_absence_steps.item() >= tracker.recovery_wall_absence_steps
    assert torch.linalg.vector_norm(
        tracker.window_center - tracker.candidate_center, dim=-1
    ).item() >= tracker.RECOVERY_CENTER_DISPLACEMENT_M
