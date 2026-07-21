import types
import unittest

import torch

from agent_ppo.feature.hard_start_replay import (
    START_NAMES,
    _segment_rows,
    initialize_hard_start_replay,
    is_hard_start_hook_configured,
    install_hard_start_replay_event,
)


CONFIG = {
    "enabled": True,
    "full_track_ratio": 0.5,
    "hard_weights": [0.45, 0.25, 0.20, 0.10],
    "track_sequence": [
        "pyramid_slope",
        "pyramid_slope_inv",
        "pyramid_stairs",
        "pyramid_stairs_inv",
        "open_entry_maze",
    ],
    "hard_segments": [
        "pyramid_stairs_inv",
        "pyramid_slope_inv",
        "open_entry_maze",
        "pyramid_stairs",
    ],
    "approach_offset_m": [0.5, 1.2],
    "entry_speeds_mps": [0.0, 0.60, 1.00, 0.70, 0.60],
    "entry_speed_jitter_mps": 0.0,
    "track_ground_z": 0.7,
}


class _FakeRobot:
    def __init__(self, num_envs):
        self.data = types.SimpleNamespace(
            default_root_state=torch.zeros(num_envs, 13),
        )
        self.data.default_root_state[:, 2] = 0.35
        self.pose = None
        self.velocity = None

    def write_root_pose_to_sim(self, value, env_ids):
        self.pose = (env_ids.clone(), value.clone())

    def write_root_velocity_to_sim(self, value, env_ids):
        self.velocity = (env_ids.clone(), value.clone())


class _FakeScene:
    def __init__(self, num_envs):
        origins = torch.zeros(5, 10, 3)
        origins[:, :, 0] = torch.arange(5).view(5, 1) * 8.0
        origins[:, :, 1] = torch.arange(10).view(1, 10) * 10.0
        self.terrain = types.SimpleNamespace(
            terrain_origins=origins,
            terrain_types=torch.arange(num_envs) % 10,
            terrain_levels=torch.zeros(num_envs, dtype=torch.long),
            env_origins=torch.zeros(num_envs, 3),
        )
        self.env_origins = torch.zeros(num_envs, 3)
        self.robot = _FakeRobot(num_envs)

    def __getitem__(self, name):
        if name != "robot":
            raise KeyError(name)
        return self.robot


class _FakeEventManager:
    def __init__(self, reset_term):
        self.reset_term = reset_term

    def get_term_cfg(self, name):
        if name != "reset_base":
            raise ValueError(name)
        return self.reset_term


class _FakeTerminationManager:
    def __init__(self, num_envs):
        self.active_terms = ["goal_reached", "bad_orientation", "base_contact"]
        self.terminated = torch.zeros(num_envs, dtype=torch.bool)
        self.time_outs = torch.zeros(num_envs, dtype=torch.bool)
        self.terms = {
            name: torch.zeros(num_envs, dtype=torch.bool)
            for name in self.active_terms
        }

    def get_term(self, name):
        return self.terms[name]


class HardStartReplayTest(unittest.TestCase):
    def _make_env(self, num_envs=20000):
        scene = _FakeScene(num_envs)
        return types.SimpleNamespace(
            num_envs=num_envs,
            device=torch.device("cpu"),
            scene=scene,
            _is_eval=False,
        )

    def _make_cfg(self):
        def original(env, env_ids, pose_range, velocity_range, asset_cfg=None):
            return None

        reset_base = types.SimpleNamespace(
            func=original,
            params={"pose_range": {}, "velocity_range": {}},
        )
        return types.SimpleNamespace(events=types.SimpleNamespace(reset_base=reset_base))

    def test_segment_names_map_to_expected_rows(self):
        self.assertEqual(_segment_rows(CONFIG).tolist(), [3, 1, 4, 2])

    def test_reset_distribution_and_pose_writes(self):
        torch.manual_seed(7)
        env = self._make_env()
        cfg = self._make_cfg()
        self.assertTrue(install_hard_start_replay_event(cfg, CONFIG))
        self.assertTrue(is_hard_start_hook_configured())
        env.event_manager = _FakeEventManager(cfg.events.reset_base)
        self.assertTrue(initialize_hard_start_replay(env))

        state = env._hard_start_replay_state
        kinds = state["start_kind"]
        full_ratio = (kinds == 0).float().mean().item()
        self.assertAlmostEqual(full_ratio, 0.5, delta=0.02)
        hard_kinds = kinds[kinds > 0] - 1
        hard_distribution = torch.bincount(hard_kinds, minlength=4).float()
        hard_distribution /= hard_distribution.sum()
        expected = torch.tensor(CONFIG["hard_weights"])
        self.assertTrue(torch.all(torch.abs(hard_distribution - expected) < 0.025))

        expected_rows = torch.zeros_like(kinds)
        rows = _segment_rows(CONFIG)
        expected_rows[kinds > 0] = rows[kinds[kinds > 0] - 1]
        self.assertTrue(torch.equal(env.scene.terrain.terrain_levels, expected_rows))
        self.assertEqual(state["start_counts"].numel(), len(START_NAMES))
        self.assertTrue(torch.isfinite(env.scene.robot.pose[1]).all())
        self.assertTrue(torch.isfinite(env.scene.robot.velocity[1]).all())

        hard_env_ids, hard_velocity = env.scene.robot.velocity
        hard_rows = env.scene.terrain.terrain_levels[hard_env_ids]
        actual_speed = torch.linalg.norm(hard_velocity[:, :2], dim=1)
        expected_speed = torch.tensor(CONFIG["entry_speeds_mps"])[hard_rows]
        self.assertTrue(torch.allclose(actual_speed, expected_speed))

        # The fake environment has no Warp mesh, so non-debug mode uses the
        # explicit fallback. Debug mode sets require_surface_query=true and
        # must never reach this fallback silently on the platform.
        _, hard_pose = env.scene.robot.pose
        self.assertTrue(torch.allclose(hard_pose[:, 2], torch.full_like(hard_pose[:, 2], 1.05)))

    def test_eval_reset_keeps_original_behavior(self):
        env = self._make_env(num_envs=16)
        env._is_eval = True
        cfg = self._make_cfg()
        install_hard_start_replay_event(cfg, CONFIG)
        cfg.events.reset_base.func(env, torch.arange(16), {}, {})
        self.assertFalse(hasattr(env, "_hard_start_replay_state"))

    def test_worker_side_outcome_and_early_failure_counts(self):
        env = self._make_env(num_envs=32)
        cfg = self._make_cfg()
        install_hard_start_replay_event(cfg, CONFIG)
        env.event_manager = _FakeEventManager(cfg.events.reset_base)
        initialize_hard_start_replay(env)

        env.termination_manager = _FakeTerminationManager(env.num_envs)
        env.episode_length_buf = torch.full((env.num_envs,), 50, dtype=torch.long)
        env.termination_manager.terminated[:3] = True
        env.termination_manager.time_outs[3] = True
        env.termination_manager.terms["goal_reached"][0] = True
        env.termination_manager.terms["bad_orientation"][1] = True
        env.termination_manager.terms["base_contact"][2] = True
        env.episode_length_buf[:4] = torch.tensor([10, 10, 25, 5])

        cfg.events.reset_base.func(env, torch.arange(4), {}, {})
        state = env._hard_start_replay_state
        self.assertEqual(int(state["episode_counts"].sum().item()), 4)
        self.assertEqual(int(state["success_counts"].sum().item()), 1)
        self.assertEqual(int(state["bad_counts"].sum().item()), 1)
        self.assertEqual(int(state["base_counts"].sum().item()), 1)
        self.assertEqual(int(state["timeout_counts"].sum().item()), 1)
        self.assertEqual(int(state["early_bad_counts"].sum().item()), 1)
        self.assertEqual(int(state["early_base_counts"].sum().item()), 0)
        self.assertEqual(int(state["early_term_counts"].sum().item()), 1)

if __name__ == "__main__":
    unittest.main()
