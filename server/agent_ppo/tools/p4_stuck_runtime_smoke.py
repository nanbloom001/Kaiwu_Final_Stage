#!/usr/bin/env python3
"""Validate the P4 wall-stuck termination against one real Isaac environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import toml
import torch


def _config(confirmation_s: float) -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "conf"
        / "train_env_conf_track_p4_nav_ppo.toml"
    )
    config = toml.load(path)
    config["game_id"] = "p4-stuck-runtime-smoke"
    config["env"]["num_envs"] = 1
    config["env_conf"]["save_mp4"] = False
    config["terrain"]["num_cols"] = 1
    config["terrain"]["max_init_terrain_level"] = 0
    config["terrain"]["track"]["num_parallel_tracks"] = 1
    config["p4_nav_ppo"]["stuck_reset"].update(
        {
            "mode": "active",
            "confirmation_s": float(confirmation_s),
            "episode_grace_s": 0.0,
            "push_grace_s": 0.0,
        }
    )
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-s", type=float, default=0.04)
    args = parser.parse_args()

    from agent_ppo.conf.conf import Config
    from agent_ppo.feature import p2_contract
    from isaac_env.base_env import Robot

    original_load_conf = Config.load_conf

    def _load_smoke_conf(logger):
        usr_conf, conf_path, is_eval, stage = original_load_conf(logger)
        usr_conf["p4_nav_ppo"]["stuck_reset"].update(
            {
                "mode": "active",
                "confirmation_s": float(args.confirmation_s),
                "episode_grace_s": 0.0,
                "push_grace_s": 0.0,
            }
        )
        return usr_conf, conf_path, is_eval, stage

    Config.load_conf = staticmethod(_load_smoke_conf)

    env = Robot()
    try:
        reset_result = env.reset(_config(args.confirmation_s))
        if reset_result is None:
            raise RuntimeError("P4 real Isaac reset returned None")
        unwrapped = env._gym_env.unwrapped
        manager = unwrapped.termination_manager
        # The first frame also verifies the deferred manager-configuration
        # retry used when observation assembly precedes termination assembly.
        priming = env.step(torch.zeros((1, 12), device=unwrapped.device))
        if priming is None:
            raise RuntimeError("P4 real Isaac priming step returned None")
        term_cfg = manager.get_term_cfg("nav_stuck_timeout")
        max_stuck = int(term_cfg.params["max_stuck"])
        spawn_controller = getattr(
            unwrapped, "_agent_ppo_p4_full_track_spawn", None
        )
        if spawn_controller is None or not spawn_controller.installed:
            raise RuntimeError("P4 full-track spawn hook is not installed")
        maze_row = int(spawn_controller._terrain_rows()[0][4])
        spawn_controller.quota.last_full[0] = False
        spawn_controller.quota.last_segment[0] = 4
        spawn_controller.quota.last_quartile[0] = 1
        spawn_controller.quota.last_safe[0] = False
        spawn_controller.quota.reason4_retries[0] = 0
        before_episode_len = int(unwrapped.episode_length_buf[0].item())
        unwrapped._nav_motion_stuck = torch.full(
            (1,), float(max_stuck), device=unwrapped.device
        )

        step_result = env.step(torch.zeros((1, 12), device=unwrapped.device))
        if step_result is None:
            raise RuntimeError("P4 real Isaac step returned None")
        _, _, _, terminated, truncated, extra = step_result
        _, privileged = extra
        privileged = torch.as_tensor(privileged)
        worker_aux = privileged[:,
            p2_contract.CRITIC_OBS_DIM : p2_contract.PRIVILEGED_WIRE_DIM
        ]
        worker_reset = bool(worker_aux[0, 24] > 0.5)
        worker_reason = int(worker_aux[0, 25].round().item())
        spawn_diagnostics = spawn_controller.diagnostics()
        terrain_level = int(
            unwrapped.scene.terrain.terrain_levels.reshape(-1)[0].item()
        )
        payload = {
            "active_terms": list(manager.active_terms),
            "time_out": bool(term_cfg.time_out),
            "max_stuck": max_stuck,
            "step_dt": float(unwrapped.step_dt),
            "before_episode_len": before_episode_len,
            "after_episode_len": int(unwrapped.episode_length_buf[0].item()),
            "terminated": bool(torch.as_tensor(terminated).reshape(-1)[0]),
            "truncated": bool(torch.as_tensor(truncated).reshape(-1)[0]),
            "worker_reset": worker_reset,
            "worker_reason": worker_reason,
            "spawn_hook_installed": int(spawn_controller.installed),
            "spawn_raycast_status": spawn_diagnostics["raycast_status"],
            "spawn_all_position_applied_count": int(
                spawn_diagnostics["all_position_applied_count"]
            ),
            "spawn_write_failure_count": int(
                spawn_diagnostics["spawn_write_failure_count"]
            ),
            "expected_maze_row": maze_row,
            "terrain_level_after_reset": terrain_level,
        }
        print(json.dumps(payload, sort_keys=True), flush=True)
        if "nav_stuck_timeout" not in payload["active_terms"]:
            raise AssertionError("nav_stuck_timeout is not active")
        if not payload["time_out"]:
            raise AssertionError("nav_stuck_timeout is not marked as timeout")
        if payload["max_stuck"] != round(args.confirmation_s / 0.02):
            raise AssertionError("nav_stuck_timeout max_stuck readback mismatch")
        if payload["terminated"]:
            raise AssertionError("wall-stuck timeout was exposed as hard termination")
        if payload["after_episode_len"] != 0:
            raise AssertionError("wall-stuck termination did not auto-reset the env")
        if not payload["worker_reset"] or payload["worker_reason"] != 4:
            raise AssertionError("worker did not preserve wall-stuck terminal reason 4")
        if payload["spawn_all_position_applied_count"] < 1:
            raise AssertionError("reason4 reset did not apply a validated all-position spawn")
        if payload["spawn_write_failure_count"] != 0:
            raise AssertionError("reason4 spawn reported a physical writer failure")
        if payload["terrain_level_after_reset"] != payload["expected_maze_row"]:
            raise AssertionError("reason4 reset did not place the env in the requested Maze row")
        print("P4_STUCK_RUNTIME_SMOKE_PASS", flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
