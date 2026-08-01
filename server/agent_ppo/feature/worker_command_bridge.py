#!/usr/bin/env python3
"""Worker-side command scheduling for visual command generalization."""

from __future__ import annotations

import sys
import time
from typing import Any, Callable

import torch

from agent_ppo.feature.command_schedule import CommandSchedule
from agent_ppo.feature.p3_command_sampler import P3RecoveryCommandSampler


_STATE_ATTR = "_agent_ppo_worker_command_bridge"
_READBACK_TOLERANCE = 1.0e-5


class _SilentConfigLogger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


def _print(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _step_key(env) -> int | None:
    value = getattr(env, "common_step_counter", None)
    if value is None:
        value = getattr(env, "_common_step_counter", None)
    try:
        if torch.is_tensor(value):
            return int(value.item())
        if value is not None:
            return int(value)
    except Exception:
        return None
    return None


def _reset_mask(env, num_envs: int, device: torch.device) -> torch.Tensor:
    lengths = getattr(env, "episode_length_buf", None)
    if not torch.is_tensor(lengths) or lengths.numel() != num_envs:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return lengths.to(device=device).reshape(-1) == 0


def _resolve_config() -> tuple[bool, dict[str, Any], int]:
    """Load the active worker configuration from the platform-selected TOML."""
    from agent_ppo.conf.conf import Config

    usr_conf, _, is_eval, stage = Config.load_conf(_SilentConfigLogger())
    commands = usr_conf.get("commands", {})
    worker = commands.get("worker_progressive", {})
    stage_conf = usr_conf.get(stage.name, {})
    schedule_mode = str(stage_conf.get("schedule_mode", ""))
    is_p3 = getattr(stage, "algorithm", "") == "p3_standard_joint"
    enabled = (
        not bool(is_eval)
        and bool(worker.get("enabled", False))
        and schedule_mode in {
            "visual_command_generalization_v1",
            "p3_low_recovery_v1",
            "p3_stair_memory_v1",
            "p35_gaitfix_v1",
        }
    )
    schedule_conf = stage_conf.get("command_schedule", {})
    if not isinstance(schedule_conf, dict):
        schedule_conf = {}
    schedule_conf = dict(schedule_conf)
    schedule_conf["_sampler_type"] = "p3_recovery" if is_p3 else "generalization"
    return enabled, schedule_conf, max(1, int(worker.get("log_interval_steps", 500)))


class WorkerCommandBridge:
    """Own one command scheduler inside the real Isaac environment worker."""

    def __init__(
        self,
        env,
        *,
        enabled: bool,
        config: dict[str, Any] | None = None,
        log_interval_steps: int = 500,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.env = env
        self.enabled = bool(enabled)
        self.clock = clock
        self.started_at = float(clock())
        self.log_interval_steps = max(1, int(log_interval_steps))
        self.scheduler = None
        self.last_step = None
        self.last_logged_step = None
        self.last_readback_error_max = None
        self.observation_errors = {"policy": None, "critic": None}
        self.observation_steps = {"policy": None, "critic": None}
        self.failure_count = 0
        self.last_failure_reason = None
        self.status = "disabled"

        if not self.enabled:
            return
        current = self._command_tensor()
        if current is None:
            self._disable("public base_velocity command tensor unavailable")
            return
        sampler_type = str((config or {}).get("_sampler_type", "generalization"))
        sampler_class = P3RecoveryCommandSampler if sampler_type == "p3_recovery" else CommandSchedule
        self.scheduler = sampler_class(
            num_envs=int(current.shape[0]),
            device=current.device,
            config=config,
        )
        self.status = "active"
        _print(
            "[WorkerCommandBridge] status=active; owner=environment_worker; "
            f"command_tensor_shape={tuple(current.shape)}; "
            "step_source=common_step_counter; ramp_start_minutes=0"
        )

    def _command_tensor(self) -> torch.Tensor | None:
        try:
            value = self.env.command_manager.get_command("base_velocity")
        except Exception:
            return None
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] < 3:
            return None
        return value[:, :3]

    @staticmethod
    def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
        expected = expected.to(device=actual.device, dtype=actual.dtype)
        return float((actual - expected).abs().max().item())

    def _readback_error(self, expected: torch.Tensor) -> float | None:
        readback = self._command_tensor()
        if readback is None or tuple(readback.shape) != tuple(expected.shape):
            return None
        return self._max_error(readback, expected)

    def _write_and_verify(self, commands: torch.Tensor) -> float:
        current = self._command_tensor()
        if current is None or tuple(current.shape) != tuple(commands.shape):
            raise RuntimeError("base_velocity command tensor unavailable or shape changed")
        with torch.no_grad():
            current.copy_(commands.to(device=current.device, dtype=current.dtype))
        error = self._readback_error(commands)
        if error is None:
            raise RuntimeError("base_velocity command readback unavailable")
        return error

    def _restore_original(self, original: torch.Tensor) -> float:
        error = self._readback_error(original)
        if error is not None and error <= _READBACK_TOLERANCE:
            return error
        error = self._write_and_verify(original)
        if error > _READBACK_TOLERANCE:
            raise RuntimeError(f"original command restore error={error:.6g}")
        return error

    def _disable(self, reason: str) -> None:
        self.enabled = False
        self.status = "warning_disabled"
        self.failure_count += 1
        self.last_failure_reason = str(reason)
        _print(
            "[WorkerCommandBridge] status=warning_disabled; native source fallback retained; "
            f"reason={reason}"
        )

    @property
    def elapsed_hours(self) -> float:
        return max(0.0, (float(self.clock()) - self.started_at) / 3600.0)

    def apply(self) -> None:
        """Apply at most one scheduler transition for the current environment step."""
        if not self.enabled or self.scheduler is None:
            return
        step = _step_key(self.env)
        if step is None:
            self._disable("common_step_counter unavailable")
            return
        if self.last_step == step:
            return

        current = self._command_tensor()
        if current is None or current.shape[0] != self.scheduler.num_envs:
            self._disable("base_velocity command tensor unavailable or shape changed")
            return
        original = current.detach().clone()
        step_delta = 1 if self.last_step is None else max(1, step - self.last_step)
        plan = self.scheduler.plan(
            original,
            dt_s=step_delta * self.scheduler.step_dt_s,
            reset_mask=_reset_mask(self.env, self.scheduler.num_envs, current.device),
            elapsed_hours=self.elapsed_hours,
        )
        desired = self.scheduler.command.clone()
        if plan.pending_ids.numel() > 0:
            desired[plan.pending_ids] = plan.pending_commands

        try:
            error = self._write_and_verify(desired)
            if error > _READBACK_TOLERANCE:
                raise RuntimeError(f"command readback error={error:.6g}")
        except Exception as exc:
            try:
                self._restore_original(original)
            except Exception as restore_exc:
                raise RuntimeError(
                    "worker command write failed and original command could not be restored: "
                    f"write={exc}; restore={restore_exc}"
                ) from restore_exc
            self._disable(str(exc))
            return

        self.scheduler.commit(plan.pending_ids, applied=True)
        self.last_readback_error_max = error
        self.last_step = step

    def record_observation(self, group: str, obs: torch.Tensor) -> None:
        """Record that default_observation saw the already-published command."""
        if not self.enabled or self.scheduler is None or self.last_step is None:
            return
        start = 6 if group == "policy" else 9
        if obs.ndim != 2 or obs.shape[1] < start + 3:
            self.observation_errors[group] = None
        else:
            actual = obs[:, start : start + 3]
            self.observation_errors[group] = self._max_error(
                actual,
                self.scheduler.command,
            )
        self.observation_steps[group] = self.last_step
        self._maybe_log()

    def _maybe_log(self) -> None:
        step = self.last_step
        if step is None or self.last_logged_step == step:
            return
        both_seen = all(value == step for value in self.observation_steps.values())
        if not both_seen or (step != 0 and step % self.log_interval_steps != 0):
            return
        metrics = self.scheduler.metrics()
        command = self._command_tensor()
        command_shape = None if command is None else tuple(command.shape)
        _print(
            "[WorkerCommandBridge] status=active; verification=verified; "
            f"step={step}; elapsed_minutes={self.elapsed_hours * 60.0:.2f}; "
            f"command_tensor_shape={command_shape}; "
            f"requested_target_probability={metrics['requested_target_probability']:.4f}; "
            f"effective_source_samples={metrics['effective_source_samples']}; "
            f"effective_target_samples={metrics['effective_target_samples']}; "
            f"buckets={metrics['target_bucket_counts']}; min={metrics['command_min']}; "
            f"max={metrics['command_max']}; mean={metrics['command_mean']}; "
            f"out_of_range_count={metrics['command_out_of_range_count']}; "
            f"readback_error_max={self.last_readback_error_max}; "
            f"policy_error_max={self.observation_errors['policy']}; "
            f"critic_error_max={self.observation_errors['critic']}; failures={self.failure_count}"
        )
        self.last_logged_step = step


def worker_command_bridge(env) -> WorkerCommandBridge:
    bridge = getattr(env, _STATE_ATTR, None)
    if bridge is None:
        enabled, config, log_interval_steps = _resolve_config()
        bridge = WorkerCommandBridge(
            env,
            enabled=enabled,
            config=config,
            log_interval_steps=log_interval_steps,
        )
        setattr(env, _STATE_ATTR, bridge)
    return bridge


def apply_worker_command(env) -> None:
    worker_command_bridge(env).apply()


def record_worker_command_observation(env, group: str, obs: torch.Tensor) -> None:
    worker_command_bridge(env).record_observation(group, obs)


def worker_command_training_state(env) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return committed P3 bucket/anchor state without creating a bridge."""
    bridge = getattr(env, _STATE_ATTR, None)
    scheduler = getattr(bridge, "scheduler", None)
    bucket = getattr(scheduler, "bucket", None)
    anchor = getattr(scheduler, "anchor_weights", None)
    if not torch.is_tensor(bucket) or not torch.is_tensor(anchor):
        return None, None
    return bucket.detach(), anchor.detach()
