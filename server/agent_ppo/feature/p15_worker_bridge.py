#!/usr/bin/env python3
"""Worker-owned P1.5 command, feedback and response-aux transport."""

from __future__ import annotations

import sys
import time
from typing import Any

import torch

from agent_ppo.feature.feedback_emulator import FeedbackEmulator
from agent_ppo.feature.p15_command_schedule import P15CommandSchedule
from agent_ppo.feature import p15_contract


_STATE_ATTR = "_agent_ppo_p15_worker_bridge"
_READBACK_TOLERANCE = 1.0e-5


def _print(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _step_key(env) -> int:
    value = getattr(env, "common_step_counter", None)
    if value is None:
        value = getattr(env, "_common_step_counter", 0)
    if torch.is_tensor(value):
        return int(value.item())
    return int(value or 0)


def _reset_mask(env, num_envs: int, device) -> torch.Tensor:
    lengths = getattr(env, "episode_length_buf", None)
    if not torch.is_tensor(lengths) or lengths.numel() != num_envs:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return lengths.to(device=device).reshape(-1) == 0


def _yaw_from_quaternion(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw for either wxyz or xyzw Isaac-style quaternion tensors."""
    if quat.ndim != 2 or quat.shape[1] != 4:
        return torch.zeros(quat.shape[0], device=quat.device, dtype=quat.dtype)
    # Isaac Lab robot data uses wxyz. Keep this conversion local to the label path.
    w, x, y, z = quat.unbind(-1)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y.square() + z.square())
    return torch.atan2(siny, cosy)


class P15WorkerBridge:
    def __init__(self, env, *, config: dict[str, Any], seed: int = 0):
        self.env = env
        self.config = dict(config)
        current = self._command_tensor()
        if current is None:
            raise RuntimeError("P1.5 requires the public base_velocity command tensor")
        self.num_envs = int(current.shape[0])
        self.device = current.device
        self.started_at = time.monotonic()
        self.resume_offset_h = max(
            0.0, float(self.config.get("command_resume_offset_hours", 0.0))
        )
        self.last_step = None
        self.last_aux = torch.zeros(
            self.num_envs,
            p15_contract.RESPONSE_AUX_DIM,
            device=self.device,
            dtype=torch.float32,
        )
        self.scheduler = P15CommandSchedule(
            self.num_envs,
            self.device,
            seed=seed,
            config=self.config.get("command_schedule", {}),
        )
        self.feedback = FeedbackEmulator(
            self.num_envs,
            self.device,
            seed=seed,
            profile=self.config.get("feedback_profile"),
        )
        self.command_failures = 0
        self.last_failure_reason = None
        self.command_valid = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._friction_enabled = False
        self._friction_last_attempt_step = -500
        self._friction_activation_attempts = 0
        self._friction_activation_failures = 0
        self._last_log_step = -1
        self._log_interval = max(1, int(self.config.get("log_interval_steps", 500)))
        self._gait_probe = None
        _print(
            "[P15WorkerBridge] active owner=environment_worker "
            f"num_envs={self.num_envs} wire={p15_contract.PRIVILEGED_WIRE_DIM} "
            f"target_hz=5 control_hz=50 resume_offset_h={self.resume_offset_h:.3f}"
        )

    def _command_tensor(self) -> torch.Tensor | None:
        try:
            value = self.env.command_manager.get_command("base_velocity")
        except Exception:
            return None
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape[1] < 3:
            return None
        return value[:, :3]

    def _robot(self):
        scene = getattr(self.env, "scene", None)
        if scene is None:
            raise RuntimeError("P1.5 response labels require env.scene")
        try:
            return scene["robot"]
        except Exception:
            robot = getattr(scene, "robot", None)
            if robot is None:
                raise RuntimeError("P1.5 response labels require scene robot asset")
            return robot

    def _terrain_metadata(self) -> tuple[torch.Tensor, torch.Tensor]:
        family = torch.zeros(self.num_envs, device=self.device)
        level = torch.zeros(self.num_envs, device=self.device)
        terrain = getattr(getattr(self.env, "scene", None), "terrain", None)
        if terrain is None:
            return family, level
        for name, target in (("terrain_types", family), ("terrain_levels", level)):
            value = getattr(terrain, name, None)
            if value is None:
                value = getattr(self.env, name, None)
            if torch.is_tensor(value) and value.numel() == self.num_envs:
                target.copy_(value.to(self.device, dtype=torch.float32).reshape(-1))
        return family, level

    def _resolve_gait_probe(self):
        if self._gait_probe is False:
            return None
        if isinstance(self._gait_probe, dict):
            return self._gait_probe
        robot = self._robot()
        try:
            body_ids, body_names = robot.find_bodies(".*_foot")
        except Exception:
            self._gait_probe = False
            return None
        body_ids = body_ids.tolist() if torch.is_tensor(body_ids) else list(body_ids)
        body_names = list(body_names)
        preferred = ("FL", "FR", "RL", "RR")
        ordered = []
        for prefix in preferred:
            matches = [
                (index, name)
                for index, name in zip(body_ids, body_names)
                if prefix.lower() in str(name).lower()
            ]
            if not matches:
                self._gait_probe = False
                return None
            ordered.append(matches[0][0])
        sensors = getattr(getattr(self.env, "scene", None), "sensors", {})
        values = sensors.values() if hasattr(sensors, "values") else []
        contact_sensor = next(
            (
                sensor
                for sensor in values
                if hasattr(getattr(sensor, "data", None), "current_air_time")
            ),
            None,
        )
        if contact_sensor is None:
            self._gait_probe = False
            return None
        shape = (4,)
        self._gait_probe = {
            "body_ids": torch.tensor(ordered, device=self.device, dtype=torch.long),
            "sensor": contact_sensor,
            "frames": 0,
            "contact_sum": torch.zeros(shape, device=self.device),
            "contact_events": torch.zeros(shape, device=self.device),
            "air_sum": torch.zeros(shape, device=self.device),
            "air_samples": torch.zeros(shape, device=self.device),
            "air_max": torch.zeros(shape, device=self.device),
            "slip_distance": torch.zeros(shape, device=self.device),
            "swing_height_sum": torch.zeros(shape, device=self.device),
            "swing_samples": torch.zeros(shape, device=self.device),
            "previous_contact": torch.ones(
                self.num_envs, 4, device=self.device, dtype=torch.bool
            ),
        }
        return self._gait_probe

    def _update_gait_probe(self) -> dict[str, float]:
        probe = self._resolve_gait_probe()
        if probe is None:
            return {}
        ids = probe["body_ids"]
        sensor_data = probe["sensor"].data
        air = sensor_data.current_air_time[:, ids]
        contact = air <= 0.0
        events = contact & ~probe["previous_contact"]
        robot = self._robot()
        foot_velocity = robot.data.body_lin_vel_w[:, ids, :2].norm(dim=-1)
        foot_z = robot.data.body_pos_w[:, ids, 2]
        nominal_ground = robot.data.root_pos_w[:, 2:3] - 0.38
        swing_height = torch.clamp(foot_z - nominal_ground, min=0.0)
        probe["frames"] += 1
        probe["contact_sum"] += contact.float().sum(dim=0)
        probe["contact_events"] += events.float().sum(dim=0)
        probe["air_sum"] += air.sum(dim=0)
        probe["air_samples"] += (~contact).float().sum(dim=0)
        probe["air_max"] = torch.maximum(probe["air_max"], air.max(dim=0).values)
        probe["slip_distance"] += (
            foot_velocity * contact.float() * p15_contract.CONTROL_DT_S
        ).sum(dim=0)
        probe["swing_height_sum"] += (
            swing_height * (~contact).float()
        ).sum(dim=0)
        probe["swing_samples"] += (~contact).float().sum(dim=0)
        probe["previous_contact"] = contact.clone()
        denominator = max(1, probe["frames"] * self.num_envs)
        elapsed_s = max(1.0e-6, probe["frames"] * p15_contract.CONTROL_DT_S)
        metrics = {}
        for index, name in enumerate(("fl", "fr", "rl", "rr")):
            metrics[f"{name}_contact_count"] = float(
                probe["contact_events"][index].item()
            )
            metrics[f"{name}_duty_factor"] = float(
                probe["contact_sum"][index].item() / denominator
            )
            metrics[f"{name}_mean_air_time"] = float(
                probe["air_sum"][index].item()
                / max(1.0, probe["air_samples"][index].item())
            )
            metrics[f"{name}_max_air_time"] = float(
                probe["air_max"][index].item()
            )
            metrics[f"{name}_step_frequency"] = float(
                probe["contact_events"][index].item()
                / (self.num_envs * elapsed_s)
            )
            metrics[f"{name}_slip_distance"] = float(
                probe["slip_distance"][index].item() / self.num_envs
            )
            metrics[f"{name}_swing_height"] = float(
                probe["swing_height_sum"][index].item()
                / max(1.0, probe["swing_samples"][index].item())
            )
        return metrics

    @staticmethod
    def _physics_material_cfg(manager):
        getter = getattr(manager, "get_term_cfg", None)
        if callable(getter):
            try:
                return getter("physics_material")
            except Exception:
                pass
        term_cfgs = getattr(manager, "_term_cfgs", None)
        if isinstance(term_cfgs, dict):
            for cfgs in term_cfgs.values():
                candidates = cfgs if isinstance(cfgs, (list, tuple)) else [cfgs]
                for cfg in candidates:
                    params = getattr(cfg, "params", None)
                    if isinstance(params, dict) and "static_friction_range" in params:
                        return cfg
        return None

    def _try_enable_friction(self, elapsed_h: float, step: int) -> None:
        if self._friction_enabled or elapsed_h < 2.0:
            return
        if step - self._friction_last_attempt_step < 500:
            return
        self._friction_last_attempt_step = step
        self._friction_activation_attempts += 1
        manager = getattr(self.env, "event_manager", None)
        cfg = self._physics_material_cfg(manager)
        if cfg is None:
            _print(
                "[P15WorkerBridge] friction activation unavailable; event term "
                "physics_material was not found, training continues and will retry"
            )
            return
        params = getattr(cfg, "params", None)
        if not isinstance(params, dict):
            _print(
                "[P15WorkerBridge] friction activation unavailable; event params "
                "are not mutable, training continues and will retry"
            )
            return
        params["static_friction_range"] = (0.6, 1.2)
        params["dynamic_friction_range"] = (0.6, 1.2)
        setter = getattr(manager, "set_term_cfg", None)
        if callable(setter):
            try:
                setter("physics_material", cfg)
            except Exception as exc:
                _print(
                    "[P15WorkerBridge] friction term update warning; continuing "
                    f"with direct application fallback: {type(exc).__name__}: {exc}"
                )
        registered_cfg = self._physics_material_cfg(manager)
        registered_params = getattr(registered_cfg, "params", None)
        registered = isinstance(registered_params, dict) and (
            tuple(registered_params.get("static_friction_range", ())) == (0.6, 1.2)
            and tuple(registered_params.get("dynamic_friction_range", ()))
            == (0.6, 1.2)
        )
        applied = False
        apply_events = getattr(manager, "apply", None)
        if registered and callable(apply_events):
            try:
                apply_events(mode="startup", env_ids=None)
                applied = True
            except TypeError:
                try:
                    apply_events("startup", env_ids=None)
                    applied = True
                except Exception:
                    pass
            except Exception:
                pass
        if not applied:
            function = getattr(cfg, "func", None)
            if callable(function):
                try:
                    function(self.env, None, **params)
                    applied = True
                except Exception as exc:
                    _print(
                        "[P15WorkerBridge] friction direct application failed; "
                        "training continues and will retry: "
                        f"{type(exc).__name__}: {exc}"
                    )
        self._friction_enabled = applied
        if not applied:
            self._friction_activation_failures += 1
        _print(
            "[P15WorkerBridge] friction schedule crossed 2h; "
            f"registered={str(registered).lower()} "
            f"runtime_applied={str(applied).lower()} range=[0.6,1.2] "
            f"attempts={self._friction_activation_attempts} "
            f"failures={self._friction_activation_failures}"
        )

    def _write_exec(self, original: torch.Tensor, desired: torch.Tensor) -> None:
        current = self._command_tensor()
        if current is None or current.shape != desired.shape:
            raise RuntimeError("base_velocity command tensor unavailable or shape changed")
        try:
            with torch.no_grad():
                current.copy_(desired.to(device=current.device, dtype=current.dtype))
            readback = self._command_tensor()
            if readback is None:
                raise RuntimeError("base_velocity command readback unavailable")
            error = (readback - desired).abs().amax(dim=1)
            self.command_valid = error <= _READBACK_TOLERANCE
            if not bool(self.command_valid.all()):
                raise RuntimeError(
                    f"command readback max error={float(error.max().item()):.6g}"
                )
        except Exception as exc:
            self.command_failures += 1
            self.last_failure_reason = f"{type(exc).__name__}: {exc}"
            self.command_valid.zero_()
            try:
                with torch.no_grad():
                    current.copy_(original)
            except Exception:
                pass
            _print(
                "[P15WorkerBridge] command apply failed; restored native command, "
                f"adapter sample invalidated failures={self.command_failures} "
                f"reason={self.last_failure_reason}"
            )

    def _build_aux(self, state, reset: torch.Tensor) -> torch.Tensor:
        robot = self._robot()
        data = robot.data
        true_velocity = torch.stack(
            (
                data.root_lin_vel_b[:, 0],
                data.root_lin_vel_b[:, 1],
                data.root_ang_vel_b[:, 2],
            ),
            dim=-1,
        )
        true_ang_vel = data.root_ang_vel_b[:, :3]
        true_gravity = data.projected_gravity_b[:, :3]
        feedback = self.feedback.step(
            true_velocity,
            true_ang_vel,
            true_gravity,
            dt_s=p15_contract.CONTROL_DT_S,
            reset_mask=reset,
        )
        root_pos = data.root_pos_w[:, :2]
        root_quat = getattr(data, "root_quat_w", None)
        if not torch.is_tensor(root_quat):
            yaw = torch.zeros(self.num_envs, device=self.device)
        else:
            yaw = _yaw_from_quaternion(root_quat)
        true_pose = torch.cat((root_pos, yaw.unsqueeze(-1)), dim=-1)
        terrain_family, terrain_level = self._terrain_metadata()

        aux = torch.zeros_like(self.last_aux)
        aux[:, 0:3] = state.active_target
        aux[:, 3:6] = state.exec_command
        aux[:, 6:9] = feedback.measured_velocity
        aux[:, 9:10] = feedback.velocity_valid * self.command_valid.float().unsqueeze(-1)
        aux[:, 10:11] = feedback.velocity_age
        aux[:, 11:12] = feedback.feedback_source
        aux[:, 12:15] = true_velocity
        aux[:, 15:18] = true_pose
        aux[:, 18:21] = feedback.ang_vel
        aux[:, 21:24] = feedback.projected_gravity
        aux[:, 24] = state.family.to(torch.float32)
        aux[:, 25] = state.trajectory_mode.to(torch.float32)
        aux[:, 26] = state.command_epoch.to(torch.float32)
        aux[:, 27] = float(state.phase_index)
        aux[:, 28] = terrain_family
        aux[:, 29] = terrain_level
        return aux

    def step(self) -> None:
        step = _step_key(self.env)
        if self.last_step == step:
            return
        native = self._command_tensor()
        if native is None:
            raise RuntimeError("P1.5 public command tensor disappeared")
        original = native.clone()
        reset = _reset_mask(self.env, self.num_envs, self.device)
        elapsed_h = self.resume_offset_h + max(
            0.0, (time.monotonic() - self.started_at) / 3600.0
        )
        self._try_enable_friction(elapsed_h, step)
        state = self.scheduler.step(
            original,
            elapsed_h=elapsed_h,
            reset_mask=reset,
        )
        self._write_exec(original, state.exec_command)
        self.last_aux = self._build_aux(state, reset)
        gait_metrics = self._update_gait_probe()
        self.last_step = step
        if step == 0 or step - self._last_log_step >= self._log_interval:
            metrics = self.scheduler.metrics()
            _print(
                "[P15WorkerBridge] "
                f"step={step} phase={metrics['phase_index']} "
                f"replay={metrics['original_replay_ratio']:.3f} "
                f"resampled_family={metrics['resampled_family_ratio']} "
                f"active_family={metrics['active_family_ratio']} "
                f"trajectory={metrics['trajectory_ratio']} "
                f"change={metrics['change_ratio']} "
                f"command_failures={self.command_failures}"
            )
            if gait_metrics:
                _print(f"[P15GaitProbe] {gait_metrics}")
            self._last_log_step = step

    def response_aux(self) -> torch.Tensor:
        self.step()
        return self.last_aux.clone()


def _resolve_config() -> tuple[bool, dict[str, Any], int]:
    from agent_ppo.conf.conf import Config

    class _Logger:
        def info(self, _message):
            pass
        warning = info
        error = info

    usr_conf, _, is_eval, stage = Config.load_conf(_Logger())
    enabled = not bool(is_eval) and getattr(stage, "algorithm", "") == "p15_response"
    stage_conf = usr_conf.get(stage.name, {}) if isinstance(usr_conf, dict) else {}
    seed = int(usr_conf.get("env_conf", {}).get("seed", 0))
    return enabled, dict(stage_conf or {}), seed


def get_p15_worker_bridge(env) -> P15WorkerBridge | None:
    state = getattr(env, _STATE_ATTR, None)
    if state is not None:
        return state
    enabled, config, seed = _resolve_config()
    if not enabled:
        setattr(env, _STATE_ATTR, False)
        return None
    state = P15WorkerBridge(env, config=config, seed=seed)
    setattr(env, _STATE_ATTR, state)
    return state


def apply_p15_worker_command(env) -> None:
    state = get_p15_worker_bridge(env)
    if state:
        state.step()


def p15_response_aux(env) -> torch.Tensor:
    state = get_p15_worker_bridge(env)
    if not state:
        raise RuntimeError("P1.5 response aux requested while bridge is disabled")
    return state.response_aux()
