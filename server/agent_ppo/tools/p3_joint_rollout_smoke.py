#!/usr/bin/env python3
"""Run real P3 phase-boundary and joint-rollout checks in Isaac Lab."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import toml
import torch

from agent_ppo.conf.conf import Config, P3StandardJointConfig
from agent_ppo.feature import p3_contract
from agent_ppo.workflow.p3_standard_joint_workflow import (
    _collect_low_rollout,
    _reset_env,
)


PHASE_BOUNDARIES = {
    "repair": 900.0,
    "pushwarm": 4500.0,
    "pushfull": 5400.0,
    "stable": 6300.0,
}
SCENARIOS = (*PHASE_BOUNDARIES, "integrated")


class _Logger:
    def _write(self, level: str, message) -> None:
        print(f"[{level}] {message}", flush=True)

    def info(self, message) -> None:
        self._write("INFO", message)

    def warning(self, message) -> None:
        self._write("WARNING", message)

    warn = warning

    def error(self, message) -> None:
        self._write("ERROR", message)

    def debug(self, message) -> None:
        self._write("DEBUG", message)


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in module.state_dict().items()
    }


def _state_changed(module: torch.nn.Module, before: dict[str, torch.Tensor]) -> bool:
    current = module.state_dict()
    return any(
        name not in current
        or not torch.equal(current[name].detach().to(device="cpu"), value)
        for name, value in before.items()
    )


def _assert_finite_metrics(metrics: dict, names: tuple[str, ...]) -> None:
    invalid = {
        name: metrics.get(name)
        for name in names
        if not math.isfinite(float(metrics.get(name, float("nan"))))
    }
    if invalid:
        raise AssertionError(f"non-finite P3 smoke metrics: {invalid}")


def _load_config(path: Path, num_envs: int, compact_terrain: bool) -> dict:
    config = toml.load(path)
    config["game_id"] = "p3-joint-rollout-smoke"
    config["env"]["num_envs"] = num_envs
    config["env_conf"]["policy_entry"] = "p3_standard_joint"
    config["env_conf"]["save_mp4"] = False
    # Exercise the high-policy -> Adapter update boundary in a single expensive
    # Isaac rollout. Unit tests separately cover the production interval of 2.
    config["p3_standard_joint"]["response_adapter"]["high_update_interval"] = 1
    if compact_terrain:
        config["terrain"]["num_rows"] = 2
        config["terrain"]["num_cols"] = 4
        config["terrain"]["max_init_terrain_level"] = 1
    return config


def _install_config(config: dict, path: Path) -> None:
    Config.CURRENT = P3StandardJointConfig

    def _load_conf(_logger=None):
        return config, str(path), False, P3StandardJointConfig

    Config.load_conf = staticmethod(_load_conf)


def _assert_boundary_reset(agent, env, config, boundary):
    joint = agent.algorithm
    high = agent.high_level_algorithm
    joint.update_clock(boundary - 1.0)
    high.command.active_target.fill_(0.25)
    high.command.exec_cmd.fill_(0.10)
    high.pending_tick = {"test_only": True}
    changed = joint.update_clock(boundary)
    if not changed:
        raise AssertionError(f"phase did not change at boundary {boundary}")
    joint.reset_live_state()
    if high.pending_tick is not None:
        raise AssertionError("phase reset retained a pending high-level tick")
    if bool(high.command.active_target.count_nonzero()):
        raise AssertionError("phase reset retained the high-level target command")
    if bool(high.command.exec_cmd.count_nonzero()):
        raise AssertionError("phase reset retained the slew command")
    if high.actor_hidden is not None or high.critic_hidden is not None:
        raise AssertionError("phase reset retained recurrent high-level hidden state")
    if len(high.response_buffer._history_aux) != 0:
        raise AssertionError("phase reset retained unfinished Adapter history")
    return _reset_env(env, agent, config)


def _snapshot_modules(agent) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "low_cnn": _state(agent.low_level_model.vision_encoder.cnn),
        "low_lstm": _state(agent.low_level_model.vision_encoder.rnn),
        "low_actor": _state(agent.low_level_model.actor),
        "low_critic": _state(agent.low_level_model.critic),
        "nav_encoder": _state(agent.navigation_encoder),
        "high_actor": _state(agent.p2_actor),
        "high_critic": _state(agent.p2_critic),
        "adapter": _state(agent.response_adapter),
    }


def _module_changes(agent, before) -> dict[str, bool]:
    return {
        "low_cnn": _state_changed(agent.low_level_model.vision_encoder.cnn, before["low_cnn"]),
        "low_lstm": _state_changed(agent.low_level_model.vision_encoder.rnn, before["low_lstm"]),
        "low_actor": _state_changed(agent.low_level_model.actor, before["low_actor"]),
        "low_critic": _state_changed(agent.low_level_model.critic, before["low_critic"]),
        "nav_encoder": _state_changed(agent.navigation_encoder, before["nav_encoder"]),
        "high_actor": _state_changed(agent.p2_actor, before["high_actor"]),
        "high_critic": _state_changed(agent.p2_critic, before["high_critic"]),
        "adapter": _state_changed(agent.response_adapter, before["adapter"]),
    }


def _run_integrated(env, agent, config):
    """Exercise one full 128-frame low PPO + Adapter + memory update."""
    joint = agent.algorithm
    joint.update_clock(PHASE_BOUNDARIES["stairrobust"])
    obs, critic_wire = _reset_env(env, agent, config)

    low_before = _snapshot_modules(agent)
    obs, critic_wire, low_metrics = _collect_low_rollout(
        env, agent, obs, critic_wire, train_low=True
    )
    low_changes = _module_changes(agent, low_before)
    if float(low_metrics.get("applied_updates", 0.0)) <= 0.0:
        raise AssertionError(f"low PPO update was skipped: {low_metrics}")
    _assert_finite_metrics(
        low_metrics,
        ("p3_low_policy_loss", "p3_low_value_loss", "adapter_loss", "memory_loss"),
    )
    for name in ("low_lstm", "low_actor", "low_critic", "adapter"):
        if not low_changes[name]:
            raise AssertionError(f"expected low module did not change: {name}")
    for name in ("low_cnn", "nav_encoder", "high_actor", "high_critic"):
        if low_changes[name]:
            raise AssertionError(f"low phase changed a frozen module: {name}")

    if joint.high_updates != 0:
        raise AssertionError("stair-memory smoke unexpectedly advanced high_updates")
    if float(low_metrics.get("memory_loss", 0.0)) <= 0.0:
        raise AssertionError(f"memory auxiliary did not run: {low_metrics}")
    return obs, critic_wire, low_metrics, low_changes


def _round_trip(agent, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    before = {
        "phase": agent.algorithm.current_phase,
        "high_updates": agent.algorithm.high_updates,
        "low_updates": agent.algorithm.low_updates,
        "actor_steps": agent.high_level_algorithm.actor_gradient_steps,
        "critic_steps": agent.high_level_algorithm.critic_gradient_steps,
        "adapter_steps": agent.high_level_algorithm.adapter_gradient_steps,
    }
    agent.algorithm.save_training_bundle(output, platform_model_id="p3smoke")
    mode = agent.algorithm.load_checkpoint(output, platform_model_id="p3smoke")
    after = {
        "phase": agent.algorithm.current_phase,
        "high_updates": agent.algorithm.high_updates,
        "low_updates": agent.algorithm.low_updates,
        "actor_steps": agent.high_level_algorithm.actor_gradient_steps,
        "critic_steps": agent.high_level_algorithm.critic_gradient_steps,
        "adapter_steps": agent.high_level_algorithm.adapter_gradient_steps,
    }
    if not mode.startswith("p3_exact_resume:") or before != after:
        raise AssertionError(
            f"P3 exact resume mismatch: mode={mode} before={before} after={after}"
        )
    return mode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "conf"
        / "train_env_conf_standard_p3_standard_joint.toml",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        required=True,
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--full-terrain", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = _load_config(
        config_path,
        args.num_envs,
        compact_terrain=not args.full_terrain,
    )
    _install_config(config, config_path)

    from agent_ppo.agent import Agent
    from isaac_env.base_env import Robot

    logger = _Logger()
    agent = Agent(agent_type="learner", device="cuda", logger=logger, monitor=None)
    load_mode = agent.algorithm.load_checkpoint(
        checkpoint,
        platform_model_id=str(
            config["p3_standard_joint"].get("parent_model_id", "")
        ) or None,
    )
    env = Robot()
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if args.scenario == "integrated":
            obs, critic_wire, metrics, changes = _run_integrated(
                env, agent, config
            )
        else:
            boundary = PHASE_BOUNDARIES[args.scenario]
            obs, critic_wire = _assert_boundary_reset(
                agent, env, config, boundary
            )
            if agent.algorithm.current_phase != args.scenario:
                raise AssertionError(
                    f"expected {args.scenario}, got {agent.algorithm.current_phase}"
                )
        if args.scenario != "integrated":
            before = _snapshot_modules(agent)
            obs, critic_wire, metrics = _collect_low_rollout(
                env, agent, obs, critic_wire, train_low=True
            )
            changes = _module_changes(agent, before)
            if agent.algorithm.high_updates != 0:
                raise AssertionError("phase smoke unexpectedly advanced high_updates")
            for name in ("low_cnn", "nav_encoder", "high_actor", "high_critic"):
                if changes[name]:
                    raise AssertionError(f"phase smoke changed frozen module {name}")

        output = args.output
        remove_output = False
        if output is None:
            output = Path(tempfile.gettempdir()) / (
                f"p3-{args.scenario}-{os.getpid()}.pkl"
            )
            remove_output = True
        exact_mode = _round_trip(agent, output)
        memory = agent.algorithm.memory_metrics()
        result = {
            "status": "PASS",
            "scenario": args.scenario,
            "num_envs": args.num_envs,
            "load_mode": load_mode,
            "exact_resume_mode": exact_mode,
            "phase": agent.algorithm.current_phase,
            "low_updates": agent.algorithm.low_updates,
            "high_updates": agent.algorithm.high_updates,
            "actor_gradient_steps": agent.high_level_algorithm.actor_gradient_steps,
            "critic_gradient_steps": agent.high_level_algorithm.critic_gradient_steps,
            "adapter_gradient_steps": agent.high_level_algorithm.adapter_gradient_steps,
            "completed_response_records": len(agent.response_aux_buffer._records),
            "module_changes": changes,
            "metrics": {
                name: float(value)
                for name, value in metrics.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            },
            "memory": memory,
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        if remove_output:
            output.unlink(missing_ok=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        exit_code = main()
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
        exit_code = 1
    raise SystemExit(exit_code)
