#!/usr/bin/env python3
"""Worker-owned feedback, label and curriculum transport for P2."""

from __future__ import annotations

import math
import sys
import types
from typing import Any

import torch

from agent_ppo.feature.feedback_emulator import FeedbackEmulator
from agent_ppo.feature.p2_gait import P2GaitWindowProbe
from agent_ppo.feature.p2_curriculum_probe import P2TrackCurriculumProbe
from agent_ppo.feature import p2_contract


_STATE_ATTR = "_agent_ppo_p2_worker_bridge"
_TERMINAL_RETURN_BRIDGE_ATTR = "_agent_ppo_p2_terminal_return_bridge"
_TERMINAL_RETURN_DIAGNOSTIC_COUNT_ATTR = (
    "_agent_ppo_p2_terminal_return_diagnostic_count"
)


def _print(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _step_key(env) -> int:
    value = getattr(env, "common_step_counter", None)
    if value is None:
        value = getattr(env, "_common_step_counter", 0)
    return int(value.item()) if torch.is_tensor(value) else int(value or 0)


def _reset_mask(env, num_envs: int, device) -> torch.Tensor:
    lengths = getattr(env, "episode_length_buf", None)
    if not torch.is_tensor(lengths) or lengths.numel() != num_envs:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return lengths.to(device=device).reshape(-1) == 0


def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        return torch.zeros(quaternion.shape[0], device=quaternion.device)
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )


def _track_segment_index(
    terrain,
    robot_pos_x: torch.Tensor,
    *,
    describe: bool = False,
) -> tuple[torch.Tensor, str]:
    """Resolve the live Track segment from world X without using spawn row."""
    invalid = torch.full_like(robot_pos_x, -1, dtype=torch.long)
    if terrain is None:
        return invalid, "terrain_missing"
    generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    track_length = getattr(generator_cfg, "track_length", None)
    size = getattr(generator_cfg, "size", None)
    try:
        track_length = int(track_length)
        size_x = float(size[0])
    except (IndexError, TypeError, ValueError):
        return invalid, "track_config_missing"
    if track_length <= 0 or not math.isfinite(size_x) or size_x <= 0.0:
        return invalid, "track_config_invalid"
    offset_x = -size_x * track_length * 0.5
    segment = torch.floor((robot_pos_x - offset_x) / size_x).long()
    segment = segment.clamp(0, track_length - 1)
    segment = torch.where(torch.isfinite(robot_pos_x), segment, invalid)
    if not describe:
        return segment, "resolved"
    boundaries = [offset_x + index * size_x for index in range(track_length + 1)]
    return segment, f"track_length={track_length},size_x={size_x},boundaries={boundaries}"


def _termination_reason_codes(env, reset: torch.Tensor) -> torch.Tensor:
    """Encode the worker-owned terminal cause into the existing aux30 wire."""
    result = torch.zeros(reset.shape[0], device=reset.device, dtype=torch.float32)
    if not bool(reset.any()):
        return result
    manager = getattr(env, "termination_manager", None)
    terminated = getattr(manager, "terminated", None)
    time_outs = getattr(manager, "time_outs", None)
    success = None
    active_terms = getattr(manager, "active_terms", None)
    if active_terms is None:
        active_terms = getattr(manager, "_term_names", ())
    if manager is not None and "goal_reached" in set(active_terms or ()):
        try:
            success = manager.get_term("goal_reached")
        except Exception:
            success = None
    success_mask = (
        success.to(reset.device).reshape(-1).bool()
        if torch.is_tensor(success) and success.numel() == reset.numel()
        else torch.zeros_like(reset)
    )
    timeout_mask = (
        time_outs.to(reset.device).reshape(-1).bool()
        if torch.is_tensor(time_outs) and time_outs.numel() == reset.numel()
        else torch.zeros_like(reset)
    )
    failure_mask = (
        terminated.to(reset.device).reshape(-1).bool()
        if torch.is_tensor(terminated) and terminated.numel() == reset.numel()
        else torch.zeros_like(reset)
    )
    success_mask &= reset
    timeout_mask &= reset & ~success_mask
    failure_mask &= reset & ~success_mask & ~timeout_mask
    # A reset with no retained termination term is a timeout at the public
    # wrapper boundary. This also recovers the known truncated/time_outs loss.
    unknown_reset = reset & ~success_mask & ~timeout_mask & ~failure_mask
    result[success_mask] = 1.0
    result[failure_mask] = 2.0
    result[timeout_mask | unknown_reset] = 3.0
    return result


def _merge_worker_terminal_returns(
    terminated,
    truncated,
    aux,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore terminal flags erased at the public wrapper boundary.

    The worker observation is computed after Isaac Lab has settled the current
    step and auto-reset the completed environments.  At that point aux[24:26]
    still contains the authoritative reset boundary and terminal reason.  Some
    platform wrapper builds return ``dones=False`` for those reset rows, which
    prevents the normal Track single-life scorer from seeing a success.

    This helper only ORs valid worker terminal rows into the native tensors.
    Native termination flags are never cleared or reclassified.
    """
    if (
        not torch.is_tensor(terminated)
        or not torch.is_tensor(truncated)
        or not torch.is_tensor(aux)
        or aux.ndim != 2
        or aux.shape[1] < 26
        or terminated.numel() != aux.shape[0]
        or truncated.numel() != aux.shape[0]
    ):
        return terminated, truncated

    reason = aux[:, 25].detach().round().long()
    reset = aux[:, 24].detach() > 0.5
    worker_hard = reset & ((reason == 1) | (reason == 2))
    worker_timeout = reset & (reason == 3)
    if not bool((worker_hard | worker_timeout).any()):
        return terminated, truncated

    terminated_shape = terminated.shape
    truncated_shape = truncated.shape
    worker_hard = worker_hard.to(device=terminated.device).reshape(terminated_shape)
    worker_timeout = worker_timeout.to(device=truncated.device).reshape(truncated_shape)
    return terminated.bool() | worker_hard, truncated.bool() | worker_timeout


def install_p2_terminal_return_bridge(env) -> bool:
    """Install a P2-only, model-ID-independent Gymnasium terminal adapter.

    The normal Track evaluator already scores ``terminated/truncated`` correctly.
    P2 therefore restores its worker-owned terminal snapshot at the innermost
    environment return boundary instead of changing the platform BaseEnv or
    adding a second scorer.  Installation is per environment and idempotent.
    """
    if bool(getattr(env, _TERMINAL_RETURN_BRIDGE_ATTR, False)):
        return False
    original_step = getattr(env, "step", None)
    if not callable(original_step):
        return False

    def _step_with_p2_terminal_return(inner_env, actions):
        result = original_step(actions)
        if not isinstance(result, (tuple, list)) or len(result) != 5:
            return result
        observation, reward, terminated, truncated, extras = result
        bridge = getattr(inner_env, _STATE_ATTR, None)
        aux = getattr(bridge, "last_aux", None)
        native_terminated = terminated
        native_truncated = truncated
        terminated, truncated = _merge_worker_terminal_returns(
            terminated,
            truncated,
            aux,
        )
        # Keep a bounded runtime proof that the adapter is installed on the
        # actual ManagerBasedRLEnv and that a worker terminal row reached the
        # public Gymnasium return. Diagnostics must never affect stepping.
        try:
            if (
                torch.is_tensor(aux)
                and aux.ndim == 2
                and aux.shape[1] >= 26
                and torch.is_tensor(terminated)
                and torch.is_tensor(truncated)
            ):
                reset = aux[:, 24] > 0.5
                reason = aux[:, 25].detach().round().long()
                worker_hard = reset & ((reason == 1) | (reason == 2))
                worker_timeout = reset & (reason == 3)
                if bool((worker_hard | worker_timeout).any()):
                    count = int(
                        getattr(
                            inner_env,
                            _TERMINAL_RETURN_DIAGNOSTIC_COUNT_ATTR,
                            0,
                        )
                    )
                    if count < 8:
                        native_hard = torch.as_tensor(
                            native_terminated, device=worker_hard.device
                        ).bool().reshape(-1)
                        native_timeout = torch.as_tensor(
                            native_truncated, device=worker_timeout.device
                        ).bool().reshape(-1)
                        merged_hard = terminated.bool().reshape(-1)
                        merged_timeout = truncated.bool().reshape(-1)
                        _print(
                            "[P2TerminalBridge] event "
                            f"worker_success={int((reset & (reason == 1)).sum())} "
                            f"worker_failure={int((reset & (reason == 2)).sum())} "
                            f"worker_timeout={int(worker_timeout.sum())} "
                            f"native_hard={int(native_hard.sum())} "
                            f"native_timeout={int(native_timeout.sum())} "
                            f"merged_hard={int(merged_hard.sum())} "
                            f"merged_timeout={int(merged_timeout.sum())}"
                        )
                    setattr(
                        inner_env,
                        _TERMINAL_RETURN_DIAGNOSTIC_COUNT_ATTR,
                        count + 1,
                    )
        except Exception:
            pass
        return observation, reward, terminated, truncated, extras

    env.step = types.MethodType(_step_with_p2_terminal_return, env)
    setattr(env, _TERMINAL_RETURN_BRIDGE_ATTR, True)
    _print(
        "[P2TerminalBridge] installed "
        f"env_type={type(env).__module__}.{type(env).__qualname__} "
        "model_id_gate=none"
    )
    return True


class P2WorkerBridge:
    """Expose deployment-shaped feedback without owning the P2 command."""

    def __init__(self, env, *, config: dict[str, Any], seed: int = 0):
        self.env = env
        self.config = dict(config)
        self.runtime_stage_type = str(
            self.config.pop("_worker_stage_type", "p2_nav_ppo")
        )
        self.track_diagnostics_enabled = self.runtime_stage_type in {
            "p2_nav_ppo",
            "p2_nav_eval",
        }
        robot = self._robot()
        self.num_envs = int(robot.data.root_lin_vel_b.shape[0])
        self.device = robot.data.root_lin_vel_b.device
        self.last_step = None
        self.last_aux = torch.zeros(
            self.num_envs,
            p2_contract.WORKER_AUX_DIM,
            device=self.device,
            dtype=torch.float32,
        )
        self.feedback = FeedbackEmulator(
            self.num_envs,
            self.device,
            seed=seed,
            profile=self.config.get("feedback_profile"),
        )
        self.curriculum_probe = (
            P2TrackCurriculumProbe(self.num_envs)
            if self.track_diagnostics_enabled
            else None
        )
        self._gait_window = P2GaitWindowProbe(
            env,
            robot,
            num_envs=self.num_envs,
            device=self.device,
        )
        self._collision_force_history = torch.zeros(
            p2_contract.NAV_PERIOD_FRAMES,
            self.num_envs,
            device=self.device,
        )
        self._collision_force_index = 0
        self._previous_family = torch.zeros(self.num_envs, device=self.device)
        self._previous_level = torch.zeros(self.num_envs, device=self.device)
        self._previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._previous_segment = torch.full(
            (self.num_envs,), -1.0, device=self.device
        )
        self._last_gait_log_step = -3000
        terrain = getattr(getattr(self.env, "scene", None), "terrain", None)
        initial_segment, segment_status = _track_segment_index(
            terrain, robot.data.root_pos_w[:, 0], describe=True
        )
        _print(
            "[P2WorkerBridge] active owner=environment_worker "
            f"num_envs={self.num_envs} wire={p2_contract.PRIVILEGED_WIRE_DIM} "
            "command_owner=aisrv feedback_profile=p15_shared "
            f"runtime_stage={self.runtime_stage_type} "
            f"live_segment_status={segment_status} "
            "initial_segment_histogram="
            f"{torch.bincount(initial_segment[initial_segment >= 0], minlength=3).tolist() if bool((initial_segment >= 0).any()) else []}"
        )

    def _robot(self):
        scene = getattr(self.env, "scene", None)
        if scene is None:
            raise RuntimeError("P2 response labels require env.scene")
        try:
            return scene["robot"]
        except Exception:
            robot = getattr(scene, "robot", None)
            if robot is None:
                raise RuntimeError("P2 response labels require scene robot asset")
            return robot

    def _terrain_metadata(self) -> tuple[torch.Tensor, torch.Tensor]:
        family = torch.zeros(self.num_envs, device=self.device)
        level = torch.zeros(self.num_envs, device=self.device)
        terrain = getattr(getattr(self.env, "scene", None), "terrain", None)
        if terrain is None:
            return family, level
        for name, target in (("terrain_types", family), ("terrain_levels", level)):
            value = getattr(terrain, name, None)
            if torch.is_tensor(value) and value.numel() == self.num_envs:
                target.copy_(value.to(self.device, dtype=torch.float32).reshape(-1))
        return family, level

    def _goal_distance(self, robot) -> torch.Tensor:
        goal = getattr(self.env, "_p3_goal_positions", None)
        if goal is None:
            goal = getattr(self.env, "goal_positions", None)
        if not torch.is_tensor(goal) or goal.shape[0] != self.num_envs:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.linalg.vector_norm(
            goal[:, :2].to(self.device) - robot.data.root_pos_w[:, :2], dim=-1
        )

    def _body_collision_force(self, reset: torch.Tensor) -> torch.Tensor:
        """Return the max non-foot contact force over the latest 5 Hz window."""
        if bool(reset.any()):
            self._collision_force_history[:, reset] = 0.0
        instant = torch.zeros(self.num_envs, device=self.device)
        if not self._gait_window.collision_valid:
            self._collision_force_history.zero_()
            return instant
        sensor = self._gait_window.sensor
        foot_ids = self._gait_window.sensor_foot_ids
        forces = getattr(getattr(sensor, "data", None), "net_forces_w", None)
        expected_bodies = len(self._gait_window.sensor_body_names)
        if (
            torch.is_tensor(forces)
            and forces.ndim == 3
            and forces.shape[0] == self.num_envs
            and forces.shape[1] == expected_bodies
            and forces.shape[2] == 3
        ):
            non_foot = torch.ones(forces.shape[1], dtype=torch.bool, device=self.device)
            non_foot[foot_ids] = False
            if bool(non_foot.any()):
                instant = forces[:, non_foot, :].norm(dim=-1).amax(dim=-1)
            else:
                self._gait_window.disable_collision(
                    "contact_forces_has_no_non_foot_columns"
                )
        else:
            shape = tuple(forces.shape) if torch.is_tensor(forces) else None
            self._gait_window.disable_collision(
                f"net_forces_w_shape_mismatch:actual={shape},"
                f"expected=({self.num_envs},{expected_bodies},3)"
            )
        if not self._gait_window.collision_valid:
            self._collision_force_history.zero_()
            return instant
        instant = torch.nan_to_num(instant, nan=0.0, posinf=0.0, neginf=0.0)
        instant[reset] = 0.0
        self._collision_force_history[self._collision_force_index] = instant
        self._collision_force_index = (
            self._collision_force_index + 1
        ) % p2_contract.NAV_PERIOD_FRAMES
        return self._collision_force_history.amax(dim=0)

    def step(self) -> None:
        step = _step_key(self.env)
        if self.last_step == step:
            return
        robot = self._robot()
        data = robot.data
        reset = _reset_mask(self.env, self.num_envs, self.device)
        true_velocity = torch.stack(
            (data.root_lin_vel_b[:, 0], data.root_lin_vel_b[:, 1], data.root_ang_vel_b[:, 2]),
            dim=-1,
        )
        feedback = self.feedback.step(
            true_velocity,
            data.root_ang_vel_b[:, :3],
            data.projected_gravity_b[:, :3],
            dt_s=p2_contract.CONTROL_DT_S,
            reset_mask=reset,
        )
        root_quat = getattr(data, "root_quat_w", None)
        yaw = (
            _yaw_from_wxyz(root_quat)
            if torch.is_tensor(root_quat)
            else torch.zeros(self.num_envs, device=self.device)
        )
        family, level = self._terrain_metadata()
        goal_distance = self._goal_distance(robot)
        terrain = getattr(getattr(self.env, "scene", None), "terrain", None)
        current_segment, _ = _track_segment_index(terrain, data.root_pos_w[:, 0])
        terminal_safe_segment = torch.where(
            reset,
            self._previous_segment.long(),
            current_segment,
        )
        aux = torch.zeros_like(self.last_aux)
        # [0:6] are deliberately placeholders. The aisrv command owner patches
        # them immediately after splitting the critic wire.
        aux[:, 6:9] = feedback.measured_velocity
        aux[:, 9:10] = feedback.velocity_valid
        aux[:, 10:11] = feedback.velocity_age
        aux[:, 11:12] = feedback.feedback_source
        aux[:, 12:15] = true_velocity
        aux[:, 15:18] = torch.cat((data.root_pos_w[:, :2], yaw.unsqueeze(-1)), dim=-1)
        aux[:, 18:21] = feedback.ang_vel
        aux[:, 21:24] = feedback.projected_gravity
        # Evaluation does not receive frame_end(dones). Carry the worker-known
        # reset boundary so recurrent state and the 50 Hz command clock reset.
        aux[:, 24] = reset.to(torch.float32)
        # 0=none, 1=success, 2=failure, 3=timeout. The wrapper can erase
        # truncated/time_outs, so this worker-owned field is authoritative.
        aux[:, 25] = (
            _termination_reason_codes(self.env, reset)
            if self.last_step is not None
            else torch.zeros(self.num_envs, device=self.device)
        )
        aux[:, 26] = float(step)
        aux[:, 27] = 0.0
        aux[:, 28] = family
        aux[:, 29] = level
        gait = self._gait_window.step(reset)
        aux[:, 30:55] = gait
        aux[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX] = torch.where(
            reset, self._previous_family, family
        )
        aux[:, p2_contract.PRE_STEP_TERRAIN_LEVEL_INDEX] = torch.where(
            reset, self._previous_level, level
        )
        aux[:, p2_contract.PRE_STEP_GOAL_DISTANCE_INDEX] = torch.where(
            reset, self._previous_goal_distance, goal_distance
        )
        aux[:, p2_contract.BODY_COLLISION_FORCE_INDEX] = self._body_collision_force(
            reset
        )
        aux[:, p2_contract.CURRENT_SEGMENT_INDEX] = terminal_safe_segment.float()
        aux[:, p2_contract.GAIT_SENSOR_MAPPING_VALID_INDEX] = float(
            self._gait_window.valid
        )
        aux[:, p2_contract.BODY_COLLISION_MAPPING_VALID_INDEX] = float(
            self._gait_window.collision_valid
        )
        self.last_aux = aux
        if self.curriculum_probe is not None:
            self.curriculum_probe.observe(self.env)
        gait_metrics = {}
        if bool((gait[:, 24] > 0.5).any()):
            for index, name in enumerate(("fl", "fr", "rl", "rr")):
                gait_metrics[f"{name}_duty_factor"] = float(gait[:, index].mean())
                gait_metrics[f"{name}_mean_swing_time"] = float(
                    gait[:, 4 + index].mean()
                )
                gait_metrics[f"{name}_max_air_time"] = float(
                    gait[:, 8 + index].amax()
                )
                gait_metrics[f"{name}_prolonged_air_ratio"] = float(
                    gait[:, 12 + index].mean()
                )
                gait_metrics[f"{name}_step_frequency"] = float(
                    gait[:, 16 + index].mean()
                )
                gait_metrics[f"{name}_slip_speed"] = float(
                    gait[:, 20 + index].mean()
                )
        if gait_metrics and step - self._last_gait_log_step >= 3000:
            _print(f"[P2GaitProbe] step={step} {gait_metrics}")
            self._last_gait_log_step = step
        self._previous_family.copy_(family)
        self._previous_level.copy_(level)
        self._previous_goal_distance.copy_(goal_distance)
        self._previous_segment.copy_(current_segment.float())
        self.last_step = step

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
    algorithm = getattr(stage, "algorithm", "")
    enabled = algorithm in {
        "p2_nav_ppo",
        "p2_nav_eval",
        "p3_standard_joint",
        "p3_track_eval",
    }
    enabled = enabled and (
        not is_eval
        or algorithm in {"p2_nav_eval", "p3_standard_joint", "p3_track_eval"}
    )
    config_key = (
        "p3_standard_joint"
        if algorithm in {"p3_standard_joint", "p3_track_eval"}
        else "p2_nav_ppo"
    )
    stage_conf = usr_conf.get(config_key, {}) if isinstance(usr_conf, dict) else {}
    stage_conf = dict(stage_conf or {})
    stage_conf["_worker_stage_type"] = algorithm
    seed = int(usr_conf.get("env_conf", {}).get("seed", 0))
    return enabled, stage_conf, seed


def get_p2_worker_bridge(env) -> P2WorkerBridge | None:
    state = getattr(env, _STATE_ATTR, None)
    if state is not None:
        return state if state is not False else None
    enabled, config, seed = _resolve_config()
    if not enabled:
        setattr(env, _STATE_ATTR, False)
        return None
    state = P2WorkerBridge(env, config=config, seed=seed)
    setattr(env, _STATE_ATTR, state)
    return state


def p2_response_aux(env) -> torch.Tensor:
    bridge = get_p2_worker_bridge(env)
    if bridge is None:
        raise RuntimeError("P2 response aux requested while bridge is disabled")
    return bridge.response_aux()
