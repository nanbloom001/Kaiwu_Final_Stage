#!/usr/bin/env python3
"""Create and step the real one-environment P3 Standard+Camera task."""

from __future__ import annotations

import json
from pathlib import Path

import toml
import torch

from agent_ppo.feature import p2_contract, p3_contract


def _config() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "conf"
        / "train_env_conf_standard_p3_standard_joint.toml"
    )
    config = toml.load(path)
    config["game_id"] = "p3-env-smoke"
    config["env"]["num_envs"] = 1
    config["env_conf"]["save_mp4"] = False
    config["env_conf"]["policy_entry"] = "p3_standard_joint"
    config["terrain"]["num_rows"] = 1
    config["terrain"]["num_cols"] = 4
    return config


def _assert_shape(name: str, value: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise AssertionError(f"{name} shape={tuple(value.shape)} expected={expected}")


def _assert_critic_core_finite(critic_wire: torch.Tensor) -> None:
    critic_core = critic_wire[:, : p2_contract.CRITIC_OBS_DIM]
    if not bool(torch.isfinite(critic_core).all()):
        raise AssertionError("critic core contains non-finite values")


def main() -> int:
    from isaac_env.base_env import Robot

    env = Robot()
    try:
        reset_result = env.reset(_config())
        if reset_result is None:
            raise RuntimeError("P3 real Isaac reset returned None")
        obs, critic_wire = reset_result
        obs = torch.as_tensor(obs)
        critic_wire = torch.as_tensor(critic_wire)
        print(
            json.dumps(
                {
                    "event": "reset_observation",
                    "policy_shape": list(obs.shape),
                    "policy_finite_share": float(torch.isfinite(obs).float().mean()),
                    "critic_wire_shape": list(critic_wire.shape),
                    "critic_core_finite_share": float(
                        torch.isfinite(
                            critic_wire[:, : p2_contract.CRITIC_OBS_DIM]
                        ).float().mean()
                    ),
                    "worker_aux_finite_share": float(
                        torch.isfinite(
                            critic_wire[:, p2_contract.CRITIC_OBS_DIM :]
                        ).float().mean()
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _assert_shape("policy", obs, (1, p3_contract.POLICY_OBS_DIM))
        _assert_shape(
            "critic_wire", critic_wire, (1, p2_contract.PRIVILEGED_WIRE_DIM)
        )
        if not bool(torch.isfinite(obs).all()):
            raise AssertionError("policy observation contains non-finite values")
        _assert_critic_core_finite(critic_wire)
        actions = torch.zeros((1, 12), device=obs.device)
        step_payload = None
        for frame in range(3):
            result = env.step(actions)
            if result is None:
                raise RuntimeError(f"P3 real Isaac step {frame} returned None")
            _, next_obs, rewards, terminated, truncated, extra = result
            _, next_wire = extra
            next_obs = torch.as_tensor(next_obs)
            next_wire = torch.as_tensor(next_wire)
            _assert_shape("next_policy", next_obs, (1, p3_contract.POLICY_OBS_DIM))
            _assert_shape(
                "next_critic_wire",
                next_wire,
                (1, p2_contract.PRIVILEGED_WIRE_DIM),
            )
            if not bool(torch.isfinite(next_obs).all()):
                raise AssertionError("next policy observation contains non-finite values")
            _assert_critic_core_finite(next_wire)
            step_payload = {
                "frame": frame + 1,
                "reward": float(torch.as_tensor(rewards).reshape(-1)[0]),
                "terminated": bool(torch.as_tensor(terminated).reshape(-1)[0]),
                "truncated": bool(torch.as_tensor(truncated).reshape(-1)[0]),
                "worker_aux_finite": bool(
                    torch.isfinite(
                        next_wire[:, p2_contract.CRITIC_OBS_DIM :]
                    ).all()
                ),
            }
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "policy_shape": list(obs.shape),
                    "critic_wire_shape": list(critic_wire.shape),
                    "last_step": step_payload,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
