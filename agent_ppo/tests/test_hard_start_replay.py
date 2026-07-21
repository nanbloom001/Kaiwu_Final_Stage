import types
import unittest

import torch

from agent_ppo.feature.hard_start_replay import (
    START_NAMES,
    _segment_rows,
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

        reset_base = types.SimpleNamespace(func=original)
        return types.SimpleNamespace(events=types.SimpleNamespace(reset_base=reset_base))

    def test_segment_names_map_to_expected_rows(self):
        self.assertEqual(_segment_rows(CONFIG).tolist(), [3, 1, 4, 2])

    def test_reset_distribution_and_pose_writes(self):
        torch.manual_seed(7)
        env = self._make_env()
        cfg = self._make_cfg()
        self.assertTrue(install_hard_start_replay_event(cfg, CONFIG))
        env_ids = torch.arange(env.num_envs)
        cfg.events.reset_base.func(env, env_ids, {}, {})

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

    def test_eval_reset_keeps_original_behavior(self):
        env = self._make_env(num_envs=16)
        env._is_eval = True
        cfg = self._make_cfg()
        install_hard_start_replay_event(cfg, CONFIG)
        cfg.events.reset_base.func(env, torch.arange(16), {}, {})
        self.assertFalse(hasattr(env, "_hard_start_replay_state"))


if __name__ == "__main__":
    unittest.main()
