#!/usr/bin/env python3
"""Worker-owned feedback, label and curriculum transport for P2."""

from __future__ import annotations

import copy
import functools
import inspect
import math
import sys
import time
import types
from typing import Any

import torch

from agent_ppo.feature.feedback_emulator import FeedbackEmulator
from agent_ppo.feature.p2_gait import P2GaitWindowProbe
from agent_ppo.feature.p2_curriculum_probe import P2TrackCurriculumProbe
from agent_ppo.feature import p2_contract, p3_contract, p4_contract
from agent_ppo.feature.goal_features import build_track_goal_raw
from agent_ppo.feature.p3_gait import validate_mirror_assembly
from agent_ppo.feature.p4_spawn import install_p4_full_track_spawn
from agent_ppo.feature.p4_stuck import MotionWallStuckTracker


_STATE_ATTR = "_agent_ppo_p2_worker_bridge"
_TERMINAL_RETURN_BRIDGE_ATTR = "_agent_ppo_p2_terminal_return_bridge"
_TERMINAL_RETURN_DIAGNOSTIC_COUNT_ATTR = (
    "_agent_ppo_p2_terminal_return_diagnostic_count"
)
_TERMINAL_SNAPSHOT_WRAPPER_ATTR = "_is_p2_training_terminal_snapshot_wrapper"


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


def _p4_spawn_wire_state(quota, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Move CPU-owned spawn quota state before composing the worker wire."""
    full = quota.last_full.to(device=device)
    raw_segment = quota.last_segment.to(device=device)
    segment = torch.where(full, torch.zeros_like(raw_segment), raw_segment)
    return (
        full,
        segment,
        quota.last_quartile.to(device=device),
        quota.last_safe.to(device=device),
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


def _termination_reason_codes(
    env,
    reset: torch.Tensor,
    wall_stuck: torch.Tensor | None = None,
) -> torch.Tensor:
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
        success.to(reset.device).reshape(-1).bool().clone()
        if torch.is_tensor(success) and success.numel() == reset.numel()
        else torch.zeros_like(reset)
    )
    timeout_mask = (
        time_outs.to(reset.device).reshape(-1).bool().clone()
        if torch.is_tensor(time_outs) and time_outs.numel() == reset.numel()
        else torch.zeros_like(reset)
    )
    failure_mask = (
        terminated.to(reset.device).reshape(-1).bool().clone()
        if torch.is_tensor(terminated) and terminated.numel() == reset.numel()
        else torch.zeros_like(reset)
    )
    wall_mask = (
        wall_stuck.to(reset.device).reshape(-1).bool().clone()
        if torch.is_tensor(wall_stuck) and wall_stuck.numel() == reset.numel()
        else torch.zeros_like(reset)
    )
    success_mask &= reset
    failure_mask &= reset & ~success_mask
    wall_mask &= reset & ~success_mask & ~failure_mask
    timeout_mask &= reset & ~success_mask & ~failure_mask & ~wall_mask
    result[success_mask] = 1.0
    result[failure_mask] = 2.0
    result[wall_mask] = 4.0
    # An unclassified wrapper reset has no trustworthy attribution.  In
    # particular, it must not become a synthetic timeout or reason 4.
    result[timeout_mask] = 3.0
    return result


def _terminal_safe_root_pose(
    live_pose: torch.Tensor,
    previous_pose: torch.Tensor,
    reset: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return live_pose
    return torch.where(reset.unsqueeze(-1), previous_pose, live_pose)


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
    worker_timeout = reset & ((reason == 3) | (reason == 4))
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
                worker_timeout = reset & ((reason == 3) | (reason == 4))
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
                            f"worker_wall_stuck={int((reset & (reason == 4)).sum())} "
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

    @property
    def _p4_enabled(self) -> bool:
        """Derive the P4 training wire from the resolved runtime stage."""
        return self.runtime_stage_type == "p4_nav_ppo"

    def __init__(self, env, *, config: dict[str, Any], seed: int = 0):
        self.env = env
        self.config = dict(config)
        live_usr_conf = getattr(env, "usr_conf", None)
        stage_type = str(self.config.get("_worker_stage_type", "p2_nav_ppo"))
        live_key = "p4_nav_ppo" if stage_type == "p4_nav_ppo" else "p3_standard_joint"
        live_stage = live_usr_conf.get(live_key) if isinstance(live_usr_conf, dict) else None
        if isinstance(live_stage, dict):
            stage_type = self.config.get("_worker_stage_type")
            self.config.update(live_stage)
            self.config["_worker_stage_type"] = stage_type
        self.runtime_stage_type = str(
            self.config.pop("_worker_stage_type", "p2_nav_ppo")
        )
        # ``track_diagnostics_enabled`` stays training-oriented (and P2 eval).
        # ``p3_track_eval`` intentionally keeps the curriculum probe off so the
        # eval worker has no dependency on training-only curriculum state.
        self.track_diagnostics_enabled = self.runtime_stage_type in {
            "p2_nav_ppo",
            "p2_nav_eval",
            "p4_nav_ppo",
            "p4_track_eval",
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
        self._terminal_aux_snapshot = torch.zeros_like(self.last_aux)
        self._terminal_snapshot_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._terminal_snapshot_reason = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._terminal_snapshot_hook_installed = False
        self.last_p3_extra = torch.zeros(
            self.num_envs,
            p3_contract.P3_WORKER_EXTRA_DIM,
            device=self.device,
            dtype=torch.float32,
        )
        self.last_p4_extra = torch.zeros(
            self.num_envs,
            p4_contract.P4_WORKER_EXTRA_DIM,
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
        if self.runtime_stage_type.startswith("p3_") or self.runtime_stage_type.startswith("p4_"):
            self._p3_mirror_assembly_valid, self._p3_mirror_assembly_checks = (
                validate_mirror_assembly(robot, self.env)
            )
        self._p35_contact_slot_ids = None
        self._p35_contact_mapping_valid = False
        self._p35_contact_previous = torch.zeros(
            self.num_envs, 14, dtype=torch.bool, device=self.device
        )
        self._p35_contact_duration = torch.zeros(
            self.num_envs, 14, device=self.device
        )
        self._p35_push_event_flag = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._p35_push_delta = torch.zeros(self.num_envs, 2, device=self.device)
        self._p35_seconds_since_push = torch.full(
            (self.num_envs,), 1.0e6, device=self.device
        )
        self._p35_push_runtime_active = False
        self._p35_push_telemetry_valid = False
        self._p35_push_event_count = 0
        self._p35_push_phase_name = None
        self._p35_started_monotonic = time.monotonic()
        push_config = self.config.get("push_schedule") or {}
        self._p35_resume_offset_s = float(push_config.get("resume_offset_s", 0.0))
        if self.runtime_stage_type in {"p3_standard_joint", "p4_nav_ppo"}:
            self._p35_install_push_wrapper()
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
        initial_quat = getattr(robot.data, "root_quat_w", None)
        initial_yaw = (
            _yaw_from_wxyz(initial_quat)
            if torch.is_tensor(initial_quat)
            else torch.zeros(self.num_envs, device=self.device)
        )
        self._previous_root_pose = torch.cat(
            (robot.data.root_pos_w[:, :2].detach(), initial_yaw.unsqueeze(-1)), dim=-1
        ).to(self.device)
        self._last_gait_log_step = -3000
        self._p4_stuck_tracker = None
        self._p4_spawn_controller = None
        if self.runtime_stage_type == "p4_nav_ppo":
            max_episode_length = getattr(self.env, "max_episode_length", None)
            episode_length_s = (
                float(max_episode_length) * p2_contract.CONTROL_DT_S
                if max_episode_length is not None
                else 120.0
            )
            self._p4_stuck_tracker = MotionWallStuckTracker(
                self.env,
                num_envs=self.num_envs,
                device=self.device,
                config=self.config.get("stuck_reset"),
                episode_length_s=episode_length_s,
            )
            _print(
                "[P4StuckResetPreflight] "
                f"requested_mode={self._p4_stuck_tracker.requested_mode} "
                f"mode={self._p4_stuck_tracker.mode} "
                f"confirmation_s={self._p4_stuck_tracker.confirmation_s:.2f} "
                f"resume_offset_s={self._p4_stuck_tracker.resume_offset_s:.2f} "
                f"dt_s={self._p4_stuck_tracker.dt_s:.4f} "
                f"dt_valid={int(self._p4_stuck_tracker.dt_valid)} "
                f"term_available={int(self._p4_stuck_tracker.term_available)} "
                f"term_config_valid={int(self._p4_stuck_tracker.term_config_valid)}"
            )
            self._p4_spawn_controller = install_p4_full_track_spawn(
                self.env,
                self.config.get("full_track_spawn"),
                seed=int(self.config.get("p4_seed", seed)) + 31,
            )
            spawn_requested = bool(
                isinstance(self.config.get("full_track_spawn"), dict)
                and self.config["full_track_spawn"].get("enabled", False)
            )
            if spawn_requested and (
                self._p4_spawn_controller is None
                or not self._p4_spawn_controller.installed
            ):
                status = (
                    self._p4_spawn_controller.last_status
                    if self._p4_spawn_controller is not None
                    else "controller_missing"
                )
                raise RuntimeError(
                    "P4 full-track training requires the reset spawn hook; "
                    f"installation failed with status={status}"
                )
            if self._p4_spawn_controller is not None:
                spawn_diagnostics = self._p4_spawn_controller.diagnostics()
                _print(
                    "[P4FullTrackSpawnPreflight] "
                    f"installed={spawn_diagnostics['installed']} "
                    f"status={spawn_diagnostics['status']} "
                    "segments=slope,slope_inv,stairs,stairs_inv,maze "
                    f"all_position_spawn_active={spawn_diagnostics['all_position_spawn_active']}"
                )
        if self.runtime_stage_type in {"p2_nav_ppo", "p4_nav_ppo"}:
            if not self._install_training_terminal_snapshot_hook():
                raise RuntimeError(
                    "P2/P4 training requires the reset-base terminal snapshot hook"
                )
        terrain = getattr(getattr(self.env, "scene", None), "terrain", None)
        initial_segment, segment_status = _track_segment_index(
            terrain, robot.data.root_pos_w[:, 0], describe=True
        )
        _print(
            "[P2WorkerBridge] active owner=environment_worker "
            f"num_envs={self.num_envs} "
            f"wire={(p4_contract.P4_PRIVILEGED_WIRE_DIM if self._p4_enabled else p2_contract.PRIVILEGED_WIRE_DIM)} "
            "command_owner=aisrv feedback_profile=p15_shared "
            f"runtime_stage={self.runtime_stage_type} "
            f"live_segment_status={segment_status} "
            "initial_segment_histogram="
            f"{torch.bincount(initial_segment[initial_segment >= 0], minlength=(len(p4_contract.FULL_TRACK_SEGMENT_LABELS) if self._p4_enabled else 3)).tolist() if bool((initial_segment >= 0).any()) else []}"
        )

    def _install_training_terminal_snapshot_hook(self) -> bool:
        manager = getattr(self.env, "event_manager", None)
        getter = getattr(manager, "get_term_cfg", None)
        setter = getattr(manager, "set_term_cfg", None)
        if not callable(getter) or not callable(setter):
            return False
        try:
            cfg = copy.deepcopy(getter("reset_base"))
            original = getattr(cfg, "func", None)
            if not callable(original):
                return False
            if bool(getattr(original, _TERMINAL_SNAPSHOT_WRAPPER_ATTR, False)):
                self._terminal_snapshot_hook_installed = True
                return True
            signature = inspect.signature(original)
            bridge = self

            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                bound = signature.bind_partial(*args, **kwargs)
                env_ids = bound.arguments.get(
                    "env_ids", args[1] if len(args) > 1 else None
                )
                bridge._capture_training_terminal_snapshot(env_ids)
                return original(*args, **kwargs)

            setattr(wrapped, _TERMINAL_SNAPSHOT_WRAPPER_ATTR, True)
            cfg.func = wrapped
            setter("reset_base", cfg)
            readback = getter("reset_base")
            self._terminal_snapshot_hook_installed = bool(
                getattr(
                    getattr(readback, "func", None),
                    _TERMINAL_SNAPSHOT_WRAPPER_ATTR,
                    False,
                )
            )
        except Exception:
            self._terminal_snapshot_hook_installed = False
        if self._terminal_snapshot_hook_installed:
            _print(
                "[P2TerminalSnapshot] installed transport=training_only "
                f"runtime_stage={self.runtime_stage_type} wire_dim_unchanged=1"
            )
        return self._terminal_snapshot_hook_installed

    def _capture_training_terminal_snapshot(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        ).reshape(-1)
        env_ids = env_ids[(env_ids >= 0) & (env_ids < self.num_envs)]
        if not env_ids.numel():
            return
        reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reset[env_ids] = True
        robot = self._robot()
        data = robot.data
        snapshot = self.last_aux.clone()
        true_velocity = torch.stack(
            (
                data.root_lin_vel_b[:, 0],
                data.root_lin_vel_b[:, 1],
                data.root_ang_vel_b[:, 2],
            ),
            dim=-1,
        )
        root_quat = getattr(data, "root_quat_w", None)
        yaw = (
            _yaw_from_wxyz(root_quat)
            if torch.is_tensor(root_quat)
            else torch.zeros(self.num_envs, device=self.device)
        )
        root_pose = torch.cat(
            (data.root_pos_w[:, :2], yaw.unsqueeze(-1)), dim=-1
        )
        family, level = self._terrain_metadata()
        goal_distance = self._goal_distance(robot)
        terrain = getattr(getattr(self.env, "scene", None), "terrain", None)
        segment, _ = _track_segment_index(terrain, data.root_pos_w[:, 0])
        snapshot[env_ids, 12:15] = true_velocity[env_ids]
        snapshot[env_ids, 15:18] = root_pose[env_ids]
        snapshot[env_ids, 18:21] = data.root_ang_vel_b[env_ids, :3]
        snapshot[env_ids, 21:24] = data.projected_gravity_b[env_ids, :3]
        snapshot[env_ids, 28] = family[env_ids]
        snapshot[env_ids, 29] = level[env_ids]
        snapshot[env_ids, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX] = family[env_ids]
        snapshot[env_ids, p2_contract.PRE_STEP_TERRAIN_LEVEL_INDEX] = level[env_ids]
        snapshot[env_ids, p2_contract.PRE_STEP_GOAL_DISTANCE_INDEX] = goal_distance[
            env_ids
        ]
        snapshot[env_ids, p2_contract.CURRENT_SEGMENT_INDEX] = segment[
            env_ids
        ].float()
        instant_collision = self._instant_body_collision_force()
        snapshot[
            env_ids, p2_contract.BODY_COLLISION_FORCE_INDEX
        ] = instant_collision[env_ids]
        snapshot[
            env_ids, p2_contract.GAIT_SENSOR_MAPPING_VALID_INDEX
        ] = float(self._gait_window.valid)
        snapshot[
            env_ids, p2_contract.BODY_COLLISION_MAPPING_VALID_INDEX
        ] = float(self._gait_window.collision_valid)
        wall_stuck = (
            self._p4_stuck_tracker.termination_mask(reset)
            if self._p4_stuck_tracker is not None
            else torch.zeros_like(reset)
        )
        reason = _termination_reason_codes(
            self.env, reset, wall_stuck=wall_stuck
        ).round().long()
        snapshot[env_ids, 24] = 1.0
        snapshot[env_ids, 25] = reason[env_ids].float()
        snapshot[env_ids, 26] = float(_step_key(self.env))
        self._terminal_aux_snapshot[env_ids] = snapshot[env_ids]
        self._terminal_snapshot_reason[env_ids] = reason[env_ids]
        self._terminal_snapshot_mask[env_ids] = True

    def _apply_training_terminal_snapshot(
        self,
        aux: torch.Tensor,
        reset: torch.Tensor,
        terminal_reason: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured = reset.bool() & self._terminal_snapshot_mask
        if not bool(captured.any()):
            return aux, terminal_reason
        aux[captured] = self._terminal_aux_snapshot[captured]
        terminal_reason = terminal_reason.clone()
        terminal_reason[captured] = self._terminal_snapshot_reason[captured].to(
            terminal_reason.dtype
        )
        aux[captured, 24] = 1.0
        aux[captured, 25] = terminal_reason[captured].float()
        aux[captured, 26] = float(step)
        self._terminal_snapshot_mask[captured] = False
        return aux, terminal_reason

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

    def _runtime_terrain_size_x(self) -> float:
        terrain = getattr(getattr(self.env, "scene", None), "terrain", None)
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        size = getattr(generator, "size", None)
        try:
            value = float(size[0])
        except (TypeError, ValueError, IndexError):
            value = 2.0 * p3_contract.TILE_HALF_EXTENT_M
        return value

    def _p35_contact_slots(self):
        if getattr(self, "_p35_contact_slot_ids", None) is not None:
            return self._p35_contact_slot_ids
        names = [
            name.lower()
            for name in getattr(self._gait_window, "sensor_body_names", ())
        ]
        slots: list[list[int]] = []

        def matches(*tokens):
            return [
                index
                for index, name in enumerate(names)
                if all(token in name for token in tokens)
            ]

        base = [index for index, name in enumerate(names) if name == "base"]
        if not base:
            base = matches("base")
        slots.append(base[:1])
        slots.append(matches("head"))
        for leg in ("fl", "fr", "rl", "rr"):
            for link in ("hip", "thigh", "calf"):
                slots.append(matches(leg, link))
        self._p35_contact_mapping_valid = (
            len(slots) == 14
            and all(len(slot) >= 1 for slot in slots)
            and all(len(slot) == 1 for index, slot in enumerate(slots) if index != 1)
        )
        self._p35_contact_slot_ids = slots
        _print(
            "[P35Baseline] contact_mapping_valid="
            f"{int(self._p35_contact_mapping_valid)} slots={slots} names={names}"
        )
        return slots

    def _p35_contact_transport(self, reset: torch.Tensor):
        force = torch.zeros(self.num_envs, 14, device=self.device)
        onset = torch.zeros_like(force)
        duration = torch.zeros_like(force)
        sensor = getattr(self._gait_window, "sensor", None)
        if not hasattr(self, "_p35_contact_previous"):
            self._p35_contact_previous = torch.zeros(
                self.num_envs, 14, dtype=torch.bool, device=self.device
            )
            self._p35_contact_duration = torch.zeros_like(force)
        values = getattr(getattr(sensor, "data", None), "net_forces_w", None)
        slots = self._p35_contact_slots()
        valid = (
            self._p35_contact_mapping_valid
            and torch.is_tensor(values)
            and values.ndim == 3
            and values.shape[:2]
            == (self.num_envs, len(self._gait_window.sensor_body_names))
            and values.shape[2] == 3
            and bool(torch.isfinite(values).all())
        )
        if not valid:
            self._p35_contact_previous.zero_()
            self._p35_contact_duration.zero_()
            return force, onset, duration, False
        magnitudes = values.norm(dim=-1)
        for index, ids in enumerate(slots):
            force[:, index] = magnitudes[:, ids].amax(dim=-1)
        thresholds = force.new_tensor(
            [10.0, 10.0, 25.0, 25.0, 35.0, 25.0, 25.0, 35.0,
             25.0, 25.0, 35.0, 25.0, 25.0, 35.0]
        )
        above = force > thresholds
        onset = (above & ~self._p35_contact_previous).to(force)
        self._p35_contact_duration = torch.where(
            above,
            self._p35_contact_duration + p2_contract.CONTROL_DT_S,
            torch.zeros_like(self._p35_contact_duration),
        )
        self._p35_contact_previous.copy_(above)
        self._p35_contact_previous[reset] = False
        self._p35_contact_duration[reset] = 0.0
        duration.copy_(self._p35_contact_duration)
        force[reset] = 0.0
        onset[reset] = 0.0
        duration[reset] = 0.0
        return force, onset, duration, True

    def _p35_push_elapsed_s(self) -> float:
        return self._p35_resume_offset_s + max(
            0.0, time.monotonic() - self._p35_started_monotonic
        )

    def _p35_install_push_wrapper(self) -> None:
        manager = getattr(self.env, "event_manager", None)
        getter = getattr(manager, "get_term_cfg", None)
        setter = getattr(manager, "set_term_cfg", None)
        if not callable(getter) or not callable(setter):
            _print("[P35PushPreflight] valid=0 reason=runtime_api_unavailable")
            return
        term_name = str((self.config.get("push_schedule") or {}).get("term_name", "push_robot"))
        try:
            cfg = getter(term_name)
        except Exception as exc:
            _print(f"[P35PushPreflight] valid=0 reason=term_missing error={type(exc).__name__}")
            return
        mode_names = getattr(manager, "active_terms", None)
        if mode_names is None:
            mode_names = getattr(manager, "_mode_term_names", {})
        interval_names = (
            list(mode_names.get("interval", ()))
            if isinstance(mode_names, dict)
            else []
        )
        original = getattr(cfg, "func", None)
        function_name = getattr(original, "__name__", "")
        params = getattr(cfg, "params", None)
        valid = (
            term_name in interval_names
            and callable(original)
            and function_name == "push_by_setting_velocity"
            and isinstance(params, dict)
        )
        if not valid:
            _print(
                "[P35PushPreflight] valid=0 "
                f"term={term_name} mode_interval={term_name in interval_names} "
                f"function={function_name}"
            )
            return
        bridge = self

        def _wrapped_push(env, env_ids, **kwargs):
            robot = bridge._robot()
            ids = env_ids
            if ids is None:
                ids = torch.arange(bridge.num_envs, device=bridge.device)
            ids = torch.as_tensor(ids, device=bridge.device, dtype=torch.long)
            before = robot.data.root_vel_w[ids, :2].detach().clone()
            original(env, env_ids, **kwargs)
            after = robot.data.root_vel_w[ids, :2].detach()
            delta = torch.nan_to_num(after - before)
            real_event = delta.abs().amax(dim=-1) > 1.0e-8
            event_ids = ids[real_event]
            if event_ids.numel() == 0:
                return
            event_delta = delta[real_event]
            bridge._p35_push_event_flag[event_ids] = True
            bridge._p35_push_delta[event_ids] = event_delta
            bridge._p35_seconds_since_push[event_ids] = 0.0
            bridge._p35_push_event_count += int(event_ids.numel())
            if bridge._p35_push_event_count <= 5 or bridge._p35_push_event_count % 50 == 0:
                _print(
                    "[P35PushEvent] "
                    f"count={bridge._p35_push_event_count} env_ids={event_ids[:8].tolist()} "
                    f"delta_mean={event_delta.mean(dim=0).tolist()}"
                )

        cfg.func = _wrapped_push
        params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        cfg.interval_range_s = (12.0, 18.0)
        try:
            setter(term_name, cfg)
            self._p35_push_telemetry_valid = True
            _print(
                "[P35PushPreflight] valid=1 term=push_robot mode=interval "
                "interval=(12,18) initial_velocity=0 wrapper=installed"
            )
        except Exception as exc:
            _print(f"[P35PushPreflight] valid=0 reason=set_failed error={type(exc).__name__}")

    def _p35_update_push_phase(self) -> None:
        if not self._p35_push_telemetry_valid:
            return
        phase = (
            p4_contract.push_phase_config(self._p35_push_elapsed_s())
            if getattr(self, "runtime_stage_type", "") == "p4_nav_ppo"
            else p3_contract.push_phase_config(self._p35_push_elapsed_s())
        )
        if phase["name"] == self._p35_push_phase_name:
            return
        manager = self.env.event_manager
        term_name = str((self.config.get("push_schedule") or {}).get("term_name", "push_robot"))
        try:
            cfg = manager.get_term_cfg(term_name)
            maximum = float(phase["max_velocity_xy_m_s"])
            cfg.params["velocity_range"] = {
                "x": (-maximum, maximum), "y": (-maximum, maximum),
                "z": (0.0, 0.0), "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            }
            cfg.interval_range_s = (
                float(phase["min_interval_s"]), float(phase["max_interval_s"])
            )
            manager.set_term_cfg(term_name, cfg)
            manager.reset(None)
            previous = self._p35_push_phase_name
            self._p35_push_phase_name = str(phase["name"])
            self._p35_push_runtime_active = bool(phase["active"])
            _print(
                "[P35PushPhase] "
                f"before={previous} after={self._p35_push_phase_name} "
                f"session_s={self._p35_push_elapsed_s():.1f} max_xy={maximum} timer_reset=1"
            )
        except Exception as exc:
            self._p35_push_telemetry_valid = False
            self._p35_push_runtime_active = False
            _print(f"[P35PushPhase] valid=0 error={type(exc).__name__}:{exc}")

    def _p3_extra(self, robot, reset: torch.Tensor) -> torch.Tensor:
        extra = torch.zeros_like(self.last_p3_extra)
        extra[:, p3_contract.RUNTIME_TERRAIN_SIZE_INDEX] = self._runtime_terrain_size_x()
        if self._gait_window.valid:
            extra[:, 1:25] = self._gait_window.last_p3_detail
        torque = getattr(robot.data, "applied_torque", None)
        velocity = getattr(robot.data, "joint_vel", None)
        if not hasattr(self, "_p3_mirror_assembly_valid"):
            self._p3_mirror_assembly_valid, self._p3_mirror_assembly_checks = (
                validate_mirror_assembly(robot, self.env)
            )
        mirror_valid = self._p3_mirror_assembly_valid
        mirror_checks = self._p3_mirror_assembly_checks
        mirror_valid = mirror_valid and bool(self._gait_window.valid)
        extra[:, p3_contract.JOINT_MAPPING_VALID_INDEX] = float(mirror_valid)
        from agent_ppo.feature.worker_command_bridge import worker_command_training_state

        command_bucket, anchor_weight = worker_command_training_state(self.env)
        if (
            torch.is_tensor(command_bucket)
            and command_bucket.numel() == self.num_envs
            and torch.is_tensor(anchor_weight)
            and anchor_weight.numel() == self.num_envs
        ):
            extra[:, p3_contract.COMMAND_BUCKET_INDEX] = command_bucket.to(
                self.device, dtype=extra.dtype
            ).reshape(-1)
            extra[:, p3_contract.COMMAND_ANCHOR_WEIGHT_INDEX] = anchor_weight.to(
                self.device, dtype=extra.dtype
            ).reshape(-1)
        else:
            extra[:, p3_contract.COMMAND_BUCKET_INDEX] = -1.0
            extra[:, p3_contract.COMMAND_ANCHOR_WEIGHT_INDEX] = 0.0
            if not getattr(self, "_p3_command_state_missing_warned", False):
                _print(
                    "[P3CommandTransport] warning=worker_command_state_missing "
                    "fallback_anchor=0"
                )
                self._p3_command_state_missing_warned = True
        previous_mirror_valid = getattr(self, "_p3_mirror_valid_last", None)
        if previous_mirror_valid is None or previous_mirror_valid != mirror_valid:
            joint_names = getattr(robot.data, "joint_names", None)
            if not joint_names:
                joint_names = getattr(robot, "joint_names", ())
            _print(
                "[P3MirrorPreflight] "
                f"enabled={mirror_valid} checks={mirror_checks} "
                f"contact_mapping={bool(self._gait_window.valid)} "
                f"joint_names={list(joint_names) if joint_names else []}"
            )
        self._p3_mirror_valid_last = mirror_valid
        if (
            torch.is_tensor(torque)
            and torque.shape == (self.num_envs, 12)
            and torch.is_tensor(velocity)
            and velocity.shape == torque.shape
        ):
            finite = torch.isfinite(torque).all(dim=-1) & torch.isfinite(velocity).all(dim=-1)
            safe_torque = torch.where(finite.unsqueeze(-1), torque, 0.0)
            safe_velocity = torch.where(finite.unsqueeze(-1), velocity, 0.0)
            extra[:, p3_contract.JOINT_TORQUE_SLICE] = safe_torque
            extra[:, p3_contract.MECHANICAL_POWER_INDEX] = (
                safe_torque * safe_velocity
            ).abs().sum(dim=-1)
        sim2real_components = getattr(self.env, "_p3_sim2real_components", None)
        component_names = (
            "sustained_torque",
            "torque_peak",
            "action_rate",
            "action_jerk",
        )
        if isinstance(sim2real_components, dict):
            values = [sim2real_components.get(name) for name in component_names]
            if all(
                torch.is_tensor(value)
                and value.numel() == self.num_envs
                for value in values
            ):
                component_values = torch.stack(
                    [value.to(self.device, dtype=extra.dtype).reshape(-1) for value in values],
                    dim=-1,
                )
                component_valid = torch.isfinite(component_values).all(dim=-1)
                torque_mapping_valid = getattr(
                    self.env, "_p3_torque_mapping_valid", None
                )
                if (
                    torch.is_tensor(torque_mapping_valid)
                    and torque_mapping_valid.numel() == self.num_envs
                ):
                    component_valid &= torque_mapping_valid.to(
                        self.device
                    ).reshape(-1).bool()
                extra[:, p3_contract.SIM2REAL_COMPONENT_SLICE] = torch.where(
                    component_valid.unsqueeze(-1), component_values, 0.0
                )
                extra[:, p3_contract.SIM2REAL_COMPONENT_VALID_INDEX] = (
                    component_valid.to(extra)
                )
        joint_acc = getattr(robot.data, "joint_acc", None)
        joint_acc_valid = (
            bool(mirror_valid)
            and torch.is_tensor(joint_acc)
            and joint_acc.shape == (self.num_envs, 12)
            and bool(torch.isfinite(joint_acc).all())
        )
        if joint_acc_valid:
            extra[:, p3_contract.JOINT_ACCELERATION_SLICE] = joint_acc
        extra[:, p3_contract.JOINT_ACCELERATION_MAPPING_VALID_INDEX] = float(
            joint_acc_valid
        )
        contact_force, contact_onset, contact_duration, contact_valid = (
            self._p35_contact_transport(reset)
        )
        extra[:, p3_contract.CONTACT_FORCE_SLICE] = contact_force
        extra[:, p3_contract.CONTACT_ONSET_SLICE] = contact_onset
        extra[:, p3_contract.CONTACT_OVER_THRESHOLD_DURATION_SLICE] = contact_duration
        extra[:, p3_contract.CONTACT_REWARD_MAPPING_VALID_INDEX] = float(contact_valid)
        push_event_flag = getattr(
            self, "_p35_push_event_flag", torch.zeros(self.num_envs, device=self.device)
        )
        push_delta = getattr(
            self, "_p35_push_delta", torch.zeros(self.num_envs, 2, device=self.device)
        )
        seconds_since_push = getattr(
            self,
            "_p35_seconds_since_push",
            torch.full((self.num_envs,), 1.0e6, device=self.device),
        )
        extra[:, p3_contract.PUSH_EVENT_FLAG_INDEX] = push_event_flag.to(extra)
        extra[:, p3_contract.PUSH_DELTA_VELOCITY_SLICE] = push_delta
        extra[:, p3_contract.SECONDS_SINCE_PUSH_INDEX] = seconds_since_push
        extra[:, p3_contract.PUSH_RUNTIME_ACTIVE_INDEX] = float(
            getattr(self, "_p35_push_runtime_active", False)
        )
        extra[:, p3_contract.PUSH_TELEMETRY_VALID_INDEX] = float(
            getattr(self, "_p35_push_telemetry_valid", False)
        )
        # Reset only episode-local measurements.  The joint-name mapping is a
        # static assembly invariant and must remain valid on reset rows;
        # clearing it would disable gait/mirror training for every rollout
        # containing any reset environment.
        extra[reset, 1 : p3_contract.JOINT_MAPPING_VALID_INDEX] = 0.0
        extra[reset, p3_contract.SIM2REAL_COMPONENT_SLICE] = 0.0
        extra[reset, p3_contract.SIM2REAL_COMPONENT_VALID_INDEX] = 0.0
        extra[reset, p3_contract.JOINT_ACCELERATION_SLICE] = 0.0
        extra[reset, p3_contract.CONTACT_FORCE_SLICE] = 0.0
        extra[reset, p3_contract.CONTACT_ONSET_SLICE] = 0.0
        extra[reset, p3_contract.CONTACT_OVER_THRESHOLD_DURATION_SLICE] = 0.0
        extra[reset, p3_contract.PUSH_EVENT_FLAG_INDEX] = 0.0
        extra[reset, p3_contract.PUSH_DELTA_VELOCITY_SLICE] = 0.0
        if hasattr(self, "_p35_push_event_flag"):
            self._p35_push_event_flag.zero_()
        if hasattr(self, "_p35_push_delta"):
            self._p35_push_delta.zero_()
        return extra

    def _goal_distance(self, robot) -> torch.Tensor:
        goal = getattr(self.env, "_p3_goal_positions", None)
        if goal is None:
            goal = getattr(self.env, "goal_positions", None)
        if not torch.is_tensor(goal) or goal.shape[0] != self.num_envs:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.linalg.vector_norm(
            goal[:, :2].to(self.device) - robot.data.root_pos_w[:, :2], dim=-1
        )

    def _instant_body_collision_force(self) -> torch.Tensor:
        """Read current non-foot contact force without advancing window state."""
        instant = torch.zeros(self.num_envs, device=self.device)
        if not self._gait_window.collision_valid:
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
            return instant
        return torch.nan_to_num(instant, nan=0.0, posinf=0.0, neginf=0.0)

    def _body_collision_force(self, reset: torch.Tensor) -> torch.Tensor:
        """Return the max non-foot contact force over the latest 5 Hz window."""
        if bool(reset.any()):
            self._collision_force_history[:, reset] = 0.0
        instant = self._instant_body_collision_force()
        if not self._gait_window.collision_valid:
            self._collision_force_history.zero_()
            return instant
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
        wall_stuck_term = (
            self._p4_stuck_tracker.termination_mask(reset)
            if self._p4_stuck_tracker is not None
            else torch.zeros_like(reset)
        )
        terminal_reason = (
            _termination_reason_codes(
                self.env,
                reset,
                wall_stuck=wall_stuck_term,
            )
            if self.last_step is not None
            else torch.zeros(self.num_envs, device=self.device)
        )
        seconds_since_push_for_stuck = self._p35_seconds_since_push.clone()
        if self.runtime_stage_type in {"p3_standard_joint", "p4_nav_ppo"}:
            self._p35_seconds_since_push.add_(p2_contract.CONTROL_DT_S)
            self._p35_seconds_since_push[reset] = 1.0e6
            self._p35_update_push_phase()
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
        root_pose = torch.cat((data.root_pos_w[:, :2], yaw.unsqueeze(-1)), dim=-1)
        root_pose = _terminal_safe_root_pose(
            root_pose,
            self._previous_root_pose,
            reset,
            enabled=self.runtime_stage_type in {"p3_standard_joint", "p4_nav_ppo"},
        )
        aux[:, 15:18] = root_pose
        aux[:, 18:21] = feedback.ang_vel
        aux[:, 21:24] = feedback.projected_gravity
        # Evaluation does not receive frame_end(dones). Carry the worker-known
        # reset boundary so recurrent state and the 50 Hz command clock reset.
        aux[:, 24] = reset.to(torch.float32)
        # 0=none, 1=success, 2=failure, 3=timeout, 4=confirmed wall-stuck.
        # The wrapper can erase
        # truncated/time_outs, so this worker-owned field is authoritative.
        aux[:, 25] = terminal_reason
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
        body_collision_force = self._body_collision_force(reset)
        aux[:, p2_contract.BODY_COLLISION_FORCE_INDEX] = body_collision_force
        aux[:, p2_contract.CURRENT_SEGMENT_INDEX] = terminal_safe_segment.float()
        aux[:, p2_contract.GAIT_SENSOR_MAPPING_VALID_INDEX] = float(
            self._gait_window.valid
        )
        aux[:, p2_contract.BODY_COLLISION_MAPPING_VALID_INDEX] = float(
            self._gait_window.collision_valid
        )
        if self.runtime_stage_type in {"p2_nav_ppo", "p4_nav_ppo"}:
            aux, terminal_reason = self._apply_training_terminal_snapshot(
                aux, reset, terminal_reason, step
            )
        self.last_aux = aux
        if self.runtime_stage_type in {"p3_standard_joint", "p4_nav_ppo"}:
            self.last_p3_extra = self._p3_extra(robot, reset)
        if self._p4_stuck_tracker is not None:
            lengths = getattr(self.env, "episode_length_buf", None)
            episode_age_s = (
                lengths.to(self.device).reshape(-1).float()
                * p2_contract.CONTROL_DT_S
                if torch.is_tensor(lengths) and lengths.numel() == self.num_envs
                else torch.zeros(self.num_envs, device=self.device)
            )
            stuck_diagnostics = self._p4_stuck_tracker.update(
                root_xy=data.root_pos_w[:, :2],
                goal_distance=goal_distance,
                collision_force=body_collision_force,
                mapping_valid=torch.full(
                    (self.num_envs,),
                    bool(self._gait_window.collision_valid),
                    dtype=torch.bool,
                    device=self.device,
                ),
                reset=reset,
                terminal_reason=terminal_reason,
                seconds_since_push=seconds_since_push_for_stuck,
                episode_age_s=episode_age_s,
                true_velocity3=true_velocity,
                yaw=yaw,
            )
            raw_goal_xy = build_track_goal_raw(self.env).to(self.device)
            if not bool(torch.isfinite(raw_goal_xy).all()):
                raise RuntimeError("P4 raw metric goal contains non-finite values")
            self.last_p4_extra.zero_()
            self.last_p4_extra[:, p4_contract.RAW_GOAL_XY_SLICE] = raw_goal_xy
            self.last_p4_extra[:, 2:15] = stuck_diagnostics
            # Preserve the raw termination-manager signal independently from
            # the exclusive reason code. Wall-stuck owns overlap so a physical
            # reset can never be counted or rewarded as success.
            self.last_p4_extra[:, p4_contract.STUCK_RAW_TERM_INDEX] = (
                wall_stuck_term.to(torch.float32)
            )
            if self._p4_spawn_controller is not None:
                self.last_p4_extra[:, p4_contract.SPAWN_INSTALLED_INDEX] = float(
                    self._p4_spawn_controller.installed
                )
                spawn = self._p4_spawn_controller.quota
                full, segment, quartile, safe = _p4_spawn_wire_state(
                    spawn, self.device
                )
                self.last_p4_extra[:, p4_contract.SPAWN_FULL_START_INDEX] = full.float()
                self.last_p4_extra[:, p4_contract.SPAWN_SEGMENT_INDEX] = segment.float()
                self.last_p4_extra[:, p4_contract.SPAWN_QUARTILE_INDEX] = quartile.float()
                self.last_p4_extra[:, p4_contract.SPAWN_SAFE_POINT_INDEX] = safe.float()
                counters = self._p4_spawn_controller.diagnostic_counts
                for index, name in (
                    (p4_contract.SPAWN_REASON4_RETRY_COUNT_INDEX, "reason4_retry_count"),
                    (p4_contract.SPAWN_REASON4_EXHAUSTED_COUNT_INDEX, "reason4_exhausted_count"),
                    (
                        p4_contract.SPAWN_REASON4_FALLBACK_APPLIED_COUNT_INDEX,
                        "reason4_fallback_applied_count",
                    ),
                    (
                        p4_contract.SPAWN_ALL_POSITION_APPLIED_COUNT_INDEX,
                        "all_position_applied_count",
                    ),
                    (
                        p4_contract.SPAWN_VALIDATION_FAILURE_COUNT_INDEX,
                        "spawn_validation_failure_count",
                    ),
                    (
                        p4_contract.SPAWN_WRITE_FAILURE_COUNT_INDEX,
                        "spawn_write_failure_count",
                    ),
                ):
                    self.last_p4_extra[:, index] = float(counters[name])
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
        self._previous_root_pose.copy_(
            torch.cat((data.root_pos_w[:, :2], yaw.unsqueeze(-1)), dim=-1)
        )
        self.last_step = step

    def response_aux(self) -> torch.Tensor:
        self.step()
        return self.last_aux.clone()

    def p3_extra(self) -> torch.Tensor:
        self.step()
        return self.last_p3_extra.clone()

    def p4_extra(self) -> torch.Tensor:
        self.step()
        return self.last_p4_extra.clone()

    def p4_spawn_state_dict(self) -> dict[str, Any] | None:
        if self._p4_spawn_controller is None:
            return None
        return self._p4_spawn_controller.state_dict()

    def load_p4_spawn_state_dict(self, state: dict[str, Any]) -> None:
        if self._p4_spawn_controller is None:
            raise RuntimeError("P4 full-track spawn is not active in this worker")
        self._p4_spawn_controller.load_state_dict(state)


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
        "p4_nav_ppo",
        "p4_track_eval",
    }
    enabled = enabled and (
        not is_eval
        or algorithm in {"p2_nav_eval", "p3_standard_joint", "p3_track_eval", "p4_track_eval"}
    )
    config_key = (
        "p4_nav_ppo"
        if algorithm in {"p4_nav_ppo", "p4_track_eval"}
        else ("p3_standard_joint" if algorithm in {"p3_standard_joint", "p3_track_eval"} else "p2_nav_ppo")
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


def p3_training_extra(env) -> torch.Tensor:
    bridge = get_p2_worker_bridge(env)
    if bridge is None or bridge.runtime_stage_type not in {"p3_standard_joint", "p4_nav_ppo"}:
        raise RuntimeError("P3/P4 training extra requested while worker bridge is disabled")
    return bridge.p3_extra()


def p4_training_extra(env) -> torch.Tensor:
    bridge = get_p2_worker_bridge(env)
    if bridge is None or bridge.runtime_stage_type != "p4_nav_ppo":
        raise RuntimeError("P4 training extra requested while worker bridge is disabled")
    return bridge.p4_extra()
