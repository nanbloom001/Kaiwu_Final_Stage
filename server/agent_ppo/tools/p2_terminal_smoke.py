#!/usr/bin/env python3
"""Real-Isaac P2 goal termination bridge smoke.

Run inside the Kaiwu development container from the server root:

    /workspace/isaaclab/isaaclab.sh -p \
      agent_ppo/tools/p2_terminal_smoke.py --num-envs 1

The smoke creates the real Track+Camera environment, moves env 0 to its
current goal, performs one public BaseEnv step and verifies that the same
success reaches the P2 worker snapshot and the public terminated tensor.
It does not load or modify a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import toml
import torch


def _config(num_envs: int) -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "conf"
        / "train_env_conf_track_p2_nav_ppo.toml"
    )
    config = toml.load(path)
    config["is_eval"] = True
    config["game_id"] = "p2-terminal-smoke"
    config["env"]["num_envs"] = int(num_envs)
    config["env_conf"]["save_mp4"] = False
    config["env_conf"]["policy_entry"] = "p2_nav_eval"
    # Keep the production three-segment ordering and real Camera environment,
    # but generate only one Track column for this lifecycle smoke.  The full
    # 3x10 production terrain is covered separately by the 128-env smoke and
    # needlessly risks an IDE recycle while validating terminal propagation.
    config["terrain"]["num_cols"] = 1
    config["terrain"]["difficulty_range"] = [0.0, 0.0]
    config["terrain"]["curriculum"] = False
    config["terrain"]["max_init_terrain_level"] = 0
    config["terrain"]["track"]["num_parallel_tracks"] = 1
    return config


def _teleport_first_env_to_goal(env) -> None:
    inner = env._gym_env.unwrapped
    robot = inner.scene["robot"]
    goal = inner.goal_positions
    env_ids = torch.tensor((0,), device=robot.device, dtype=torch.long)
    pose = robot.data.root_state_w[env_ids, :7].clone()
    pose[:, :2] = goal[env_ids, :2]
    robot.write_root_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(
        torch.zeros((1, 6), device=robot.device),
        env_ids=env_ids,
    )
    inner.scene.write_data_to_sim()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 16:
        raise SystemExit("--num-envs must be in [1, 16]")

    from isaac_env.base_env import Robot

    env = Robot()
    try:
        reset_result = env.reset(_config(args.num_envs))
        if reset_result is None:
            raise RuntimeError("real Isaac environment reset failed")
        inner = env._gym_env.unwrapped
        installed = bool(
            getattr(inner, "_agent_ppo_p2_terminal_return_bridge", False)
        )
        print(
            "[P2TerminalSmoke] "
            + json.dumps(
                {
                    "event": "bridge_after_reset",
                    "installed": installed,
                    "env_type": f"{type(inner).__module__}.{type(inner).__qualname__}",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not installed:
            raise AssertionError(
                "P2 terminal bridge was not installed before the first public step"
            )

        robot = inner.scene["robot"]
        actions = torch.zeros(
            (args.num_envs, 12),
            device=robot.device,
            dtype=torch.float32,
        )
        # Platform BaseEnv materializes Track goal_positions after its first
        # public step.  Warm up that infrastructure before forcing the goal;
        # this is also the first real step through the installed bridge.
        warmup_result = env.step(actions)
        if warmup_result is None:
            raise RuntimeError("public BaseEnv warm-up step failed")
        if getattr(inner, "goal_positions", None) is None:
            raise AssertionError("Track goal_positions missing after warm-up step")
        print(
            "[P2TerminalSmoke] "
            + json.dumps(
                {
                    "event": "goal_after_warmup",
                    "available": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        _teleport_first_env_to_goal(env)
        distance_before = torch.linalg.vector_norm(
            inner.goal_positions[0, :2] - robot.data.root_pos_w[0, :2]
        ).item()
        result = env.step(actions)
        if result is None:
            raise RuntimeError("public BaseEnv step failed")
        _, _, _, terminated, truncated, _ = result
        goal_term = inner.termination_manager.get_term("goal_reached").bool()
        payload = {
            "event": "goal_step",
            "distance_before_m": float(distance_before),
            "goal_term_env0": bool(goal_term[0]),
            "public_terminated_env0": bool(terminated[0]),
            "public_truncated_env0": bool(truncated[0]),
            "single_life_done_env0": bool(env._eval_env_done_mask[0]),
        }
        print(
            "[P2TerminalSmoke] " + json.dumps(payload, sort_keys=True),
            flush=True,
        )
        if not payload["goal_term_env0"]:
            raise AssertionError("forced goal did not fire goal_reached")
        if not payload["public_terminated_env0"]:
            raise AssertionError("goal_reached did not reach public terminated")
        if payload["public_truncated_env0"]:
            raise AssertionError("goal_reached was incorrectly classified as timeout")
        if not payload["single_life_done_env0"]:
            raise AssertionError("BaseEnv single-life mask did not retain success")
        print("[P2TerminalSmoke] PASS", flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
