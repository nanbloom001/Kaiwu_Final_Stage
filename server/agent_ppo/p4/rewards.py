#!/usr/bin/env python3
"""P4 reward override and reward-diagnostic accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType

from agent_ppo.p4.constants import *  # noqa: F403
from agent_ppo.p4.training import safe_direction_weight

import torch

from agent_ppo.feature import p2_contract, p4_contract
from agent_ppo.p4.primitives import (
    maze_new_best_credit,
    open_straight_penalty,
    route_excess_penalty,
    segment_frontier_potential,
    sustained_wall_stuck_penalty,
    track_boundary_distance_m,
    yaw_cancellation,
    proportional_negative_cap,
)
from agent_ppo.p4.profiles import (
    PROFILE_FULL_TRACK,
    PROFILE_MAZE_CLOSED_LOOP_V3,
    PROFILE_MAZE_CREDIT_REPAIR,
    PROFILE_MAZE_INSTANT_COMMAND_R4,
    PROFILE_MAZE_INSTANT_REPAIR2H,
    PROFILE_MAZE_STABLE_DIRECTION_SMOKE,
    PROFILE_MAZE_STABLE_DIRECTION_8H,
)


@dataclass(frozen=True)
class RewardSpec:
    """Immutable selection of active and diagnostic-only reward terms."""

    enabled_components: frozenset[str]
    shadow_components: frozenset[str]
    safety_cap_components: frozenset[str]

    def enables(self, name: str) -> bool:
        return name in self.enabled_components


_BASE_REWARD_COMPONENTS = frozenset(
    {
        "frame_safety",
        "frontier_shaping",
        "success",
        "failure",
        "timeout",
        "time",
        "crawl",
        "command_rate",
        "tracking",
        "gait_symmetry",
        "body_collision",
        "predictive_collision_risk",
        "missed_safe_direction",
        "yaw_cancellation",
        "goal_safe_preference",
        "yaw_exit_response",
        "stuck_reset",
        "stuck_sustained",
        "route_excess",
        "segment_frontier",
        "open_straight",
        "soft_cruise",
    }
)
_SINGLE_SIGNAL_COMPONENTS = frozenset(
    {
        "frame_safety",
        "frontier_shaping",
        "success",
        "failure",
        "timeout",
        "time",
        "crawl",
        "command_rate",
        "tracking",
        "body_collision",
        "predictive_collision_risk",
        "stuck_reset",
        "stuck_sustained",
    }
)
_DIRECTIONAL_COMPONENTS = frozenset(
    {"missed_safe_direction", "goal_safe_preference", "yaw_exit_response"}
)
_MAZE_SHADOW_COMPONENTS = frozenset(
    {"frontier_stagnation", "route_excess", "segment_frontier", "open_straight"}
)
_SINGLE_SIGNAL_SHADOW_COMPONENTS = frozenset(
    _BASE_REWARD_COMPONENTS - _SINGLE_SIGNAL_COMPONENTS
)
_LEGACY_SAFETY_CAP_COMPONENTS = frozenset(
    {
        "predictive_collision_risk",
        "missed_safe_direction",
        "yaw_cancellation",
        "goal_safe_preference",
        "yaw_exit_response",
    }
)
_SINGLE_SIGNAL_SAFETY_CAP_COMPONENTS = frozenset(
    {"predictive_collision_risk", "yaw_cancellation"}
)
_STABLE_DIRECTION_ENABLED = _SINGLE_SIGNAL_COMPONENTS | _DIRECTIONAL_COMPONENTS
_STABLE_DIRECTION_SAFETY_CAP_COMPONENTS = frozenset(
    {
        "predictive_collision_risk",
        "missed_safe_direction",
        "goal_safe_preference",
        "yaw_exit_response",
    }
)

REWARD_SPECS = MappingProxyType(
    {
        PROFILE_FULL_TRACK: RewardSpec(
            _BASE_REWARD_COMPONENTS - {"frontier_stagnation"},
            frozenset({"frontier_stagnation"}),
            _LEGACY_SAFETY_CAP_COMPONENTS,
        ),
        PROFILE_MAZE_CREDIT_REPAIR: RewardSpec(
            _BASE_REWARD_COMPONENTS - _MAZE_SHADOW_COMPONENTS,
            _MAZE_SHADOW_COMPONENTS,
            _LEGACY_SAFETY_CAP_COMPONENTS,
        ),
        PROFILE_MAZE_CLOSED_LOOP_V3: RewardSpec(
            _SINGLE_SIGNAL_COMPONENTS,
            _SINGLE_SIGNAL_SHADOW_COMPONENTS,
            _SINGLE_SIGNAL_SAFETY_CAP_COMPONENTS,
        ),
        PROFILE_MAZE_INSTANT_COMMAND_R4: RewardSpec(
            _SINGLE_SIGNAL_COMPONENTS,
            _SINGLE_SIGNAL_SHADOW_COMPONENTS,
            _SINGLE_SIGNAL_SAFETY_CAP_COMPONENTS,
        ),
        PROFILE_MAZE_INSTANT_REPAIR2H: RewardSpec(
            _SINGLE_SIGNAL_COMPONENTS,
            _SINGLE_SIGNAL_SHADOW_COMPONENTS,
            _SINGLE_SIGNAL_SAFETY_CAP_COMPONENTS,
        ),
        PROFILE_MAZE_STABLE_DIRECTION_SMOKE: RewardSpec(
            _STABLE_DIRECTION_ENABLED,
            _BASE_REWARD_COMPONENTS - _STABLE_DIRECTION_ENABLED,
            _STABLE_DIRECTION_SAFETY_CAP_COMPONENTS,
        ),
        PROFILE_MAZE_STABLE_DIRECTION_8H: RewardSpec(
            _STABLE_DIRECTION_ENABLED,
            _BASE_REWARD_COMPONENTS - _STABLE_DIRECTION_ENABLED,
            _STABLE_DIRECTION_SAFETY_CAP_COMPONENTS,
        ),
    }
)


class P4RewardMixin:
    """Behavior-preserving methods extracted from AlgorithmP4NavPPO."""

    def _reward_rollout_context(self, components, context) -> dict[str, object]:
        reward_spec = REWARD_SPECS[self.training_profile]
        component_shadows = {
            name: value.detach().clone()
            for name, value in components.items()
            if name in reward_spec.shadow_components
        }
        for name in tuple(components):
            if not reward_spec.enables(name):
                components[name] = torch.zeros_like(components[name])
        exec_cmd = context["reward_exec_cmd"]
        source_aux = context["reward_source_aux"]
        terminal = context["terminal"].bool()
        duration_frames = torch.as_tensor(
            context.get("duration_frames", self.nav_period_frames),
            device=self.device,
            dtype=torch.float32,
        ).reshape(-1)
        continuous_time_scale = torch.clamp(
            duration_frames / float(p2_contract.NAV_PERIOD_FRAMES), 0.0, 1.0
        )
        maze_profile = self.training_profile in self.MAZE_PROFILES
        instant_profile = self.training_profile in self.INSTANT_PROFILES
        closed_loop_profile = self.training_profile == PROFILE_MAZE_CLOSED_LOOP_V3
        single_signal_profile = closed_loop_profile or instant_profile
        if maze_profile:
            components["time"] = (
                p2_contract.TIME_COST_PER_TICK
                * duration_frames
                / float(p4_contract.P4_NAV_PERIOD_FRAMES)
            )
            best_before = torch.where(
                torch.isfinite(self.best_goal_distance),
                self.best_goal_distance,
                context["start_goal_distance"].reshape(-1),
            )
            maze_credit_before = self._maze_credit_earned.clone()
            maze_credit, maze_credit_after, maze_credit_delta = (
                p4_contract.maze_new_best_credit(
                    best_before,
                    context["end_goal_distance"].reshape(-1),
                    self._maze_credit_earned,
                    weight_per_m=(
                        p4_contract.MAZE_NEW_BEST_WEIGHT_PER_M
                        if single_signal_profile
                        else p4_contract.LEGACY_MAZE_NEW_BEST_WEIGHT_PER_M
                    ),
                    episode_cap=(
                        p4_contract.MAZE_NEW_BEST_EPISODE_CAP
                        if single_signal_profile
                        else p4_contract.LEGACY_MAZE_NEW_BEST_EPISODE_CAP
                    ),
                )
            )
            terminal_clawback = torch.where(
                terminal
                & (
                    (context["reason"].reshape(-1) == 2)
                    | (context["reason"].reshape(-1) == 3)
                    | (context["reason"].reshape(-1) == 4)
                )
                & (self.training_profile == PROFILE_MAZE_INSTANT_COMMAND_R4),
                -maze_credit_after,
                torch.zeros_like(maze_credit_after),
            )
            components["frontier_shaping"] = maze_credit + terminal_clawback
            self._maze_credit_earned.copy_(maze_credit_after)
        else:
            maze_credit_before = self._maze_credit_earned
            maze_credit = torch.zeros(self.num_envs, device=self.device)
            maze_credit_after = self._maze_credit_earned
            maze_credit_delta = torch.zeros_like(maze_credit)
            terminal_clawback = torch.zeros_like(maze_credit)
        for name in ("crawl", "tracking", "gait_symmetry"):
            if name in components:
                components[name] = components[name] * continuous_time_scale
        return {
            "exec_cmd": exec_cmd,
            "source_aux": source_aux,
            "terminal": terminal,
            "duration_frames": duration_frames,
            "continuous_time_scale": continuous_time_scale,
            "maze_profile": maze_profile,
            "instant_profile": instant_profile,
            "closed_loop_profile": closed_loop_profile,
            "single_signal_profile": single_signal_profile,
            "reward_spec": reward_spec,
            "component_shadows": component_shadows,
            "reason": context["reason"],
            "maze_credit_before": maze_credit_before,
            "maze_credit": maze_credit,
            "maze_credit_after": maze_credit_after,
            "maze_credit_delta": maze_credit_delta,
            "terminal_clawback": terminal_clawback,
        }

    def _yaw_reward_context(self, reward_context) -> dict[str, object]:
        exec_cmd = reward_context["exec_cmd"]
        source_aux = reward_context["source_aux"]
        terminal = reward_context["terminal"]
        single_signal_profile = reward_context["single_signal_profile"]
        wall_stuck_terminal = reward_context["reason"].reshape(-1) == 4
        history_reset = terminal | self._goal_epoch_changed_since_tick
        if bool(history_reset.any()):
            self._yaw_exec_history[:, history_reset] = 0.0
            self._yaw_true_history[:, history_reset] = 0.0
            self._yaw_history_count[history_reset] = 0
        previous_index = (self._yaw_history_cursor - 1) % self._yaw_window_ticks
        previous_exec_wz = self._yaw_exec_history[previous_index]
        previous_true_wz = self._yaw_true_history[previous_index]
        exec_sign_flip = (previous_exec_wz * exec_cmd[:, 2] < 0.0) & (
            self._yaw_history_count > 0
        )
        true_sign_flip = (previous_true_wz * source_aux[:, 14] < 0.0) & (
            self._yaw_history_count > 0
        )
        self._yaw_exec_history[self._yaw_history_cursor] = exec_cmd[:, 2]
        self._yaw_true_history[self._yaw_history_cursor] = source_aux[:, 14]
        self._yaw_history_count = torch.where(
            terminal,
            torch.zeros_like(self._yaw_history_count),
            torch.clamp(self._yaw_history_count + 1, max=self._yaw_window_ticks),
        )
        self._yaw_history_cursor = (
            self._yaw_history_cursor + 1
        ) % self._yaw_window_ticks
        exec_cancel = p4_contract.yaw_cancellation(
            self._yaw_exec_history, dt_s=self.nav_dt_s
        )
        true_cancel = p4_contract.yaw_cancellation(
            self._yaw_true_history, dt_s=self.nav_dt_s
        )
        enough = self._yaw_history_count >= self._yaw_window_ticks
        yaw_exec_weight = (
            p4_contract.YAW_EXEC_WEIGHT
            if single_signal_profile
            else p4_contract.LEGACY_YAW_EXEC_WEIGHT
        )
        yaw_true_weight = (
            p4_contract.YAW_TRUE_WEIGHT
            if single_signal_profile
            else p4_contract.LEGACY_YAW_TRUE_WEIGHT
        )
        yaw_floor = (
            p4_contract.YAW_TOTAL_FLOOR
            if single_signal_profile
            else p4_contract.LEGACY_YAW_TOTAL_FLOOR
        )
        yaw_raw = torch.where(
            enough,
            yaw_exec_weight * exec_cancel + yaw_true_weight * true_cancel,
            torch.zeros_like(exec_cancel),
        ).clamp_min(yaw_floor)
        return {
            "wall_stuck_terminal": wall_stuck_terminal,
            "exec_sign_flip": exec_sign_flip,
            "true_sign_flip": true_sign_flip,
            "exec_cancel": exec_cancel,
            "true_cancel": true_cancel,
            "yaw_raw": yaw_raw,
        }

    def _safety_reward_context(
        self, components, context, reward_context
    ) -> dict[str, object]:
        continuous_time_scale = reward_context["continuous_time_scale"]
        single_signal_profile = reward_context["single_signal_profile"]
        reward_spec = reward_context["reward_spec"]
        terminal = reward_context["terminal"]
        yaw_raw = reward_context["yaw_raw"]
        predictive_raw = (
            components["predictive_collision_risk"]
            * p4_contract.PREDICTIVE_COLLISION_SCALE
        ).clamp_min(p4_contract.PREDICTIVE_RAW_FLOOR)
        safe3 = self.pending_tick.get(
            "safe3", torch.zeros(self.num_envs, 3, device=self.device)
        )
        safety_valid = (
            self.pending_tick.get(
                "safety_valid", torch.zeros(self.num_envs, 1, device=self.device)
            ).reshape(-1)
            > 0.5
        )
        missed_raw, missed_diagnostics = p4_contract.maze_missed_safe_direction_penalty(
            safe3,
            self._last_policy_command,
            safety_valid,
            self.session_effective_seconds,
        )
        schedule = p4_contract.training_schedule(
            self.session_effective_seconds,
            branch=self._effective_maze_branch(self.session_effective_seconds),
        )
        reward_multiplier = float(schedule["reward_multiplier"])
        predictive_raw *= reward_multiplier
        missed_raw *= reward_multiplier
        yaw_raw *= reward_multiplier
        goal_safe_raw, goal_safe_diag = p4_contract.goal_safe_direction_penalty(
            safe3,
            self._last_policy_command,
            self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE],
            safety_valid,
            self._last_goal_freshness,
            terminal,
        )
        yaw_exit_raw, yaw_exit_diag = p4_contract.yaw_exit_response_penalty(
            safe3,
            self._last_policy_command,
            self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE],
            safety_valid,
            self._last_goal_freshness,
            terminal,
        )
        goal_safe_raw *= reward_multiplier
        yaw_exit_raw *= reward_multiplier
        missed_shadow_raw = missed_raw.detach().clone()
        goal_safe_shadow_raw = goal_safe_raw.detach().clone()
        yaw_exit_shadow_raw = yaw_exit_raw.detach().clone()
        active_missed = (
            missed_raw
            if "missed_safe_direction" in reward_spec.safety_cap_components
            else torch.zeros_like(missed_raw)
        )
        active_yaw = (
            yaw_raw
            if "yaw_cancellation" in reward_spec.safety_cap_components
            else torch.zeros_like(yaw_raw)
        )
        active_goal_safe = (
            goal_safe_raw
            if "goal_safe_preference" in reward_spec.safety_cap_components
            else torch.zeros_like(goal_safe_raw)
        )
        active_yaw_exit = (
            yaw_exit_raw
            if "yaw_exit_response" in reward_spec.safety_cap_components
            else torch.zeros_like(yaw_exit_raw)
        )
        (predictive, missed, yaw, goal_safe, yaw_exit), scale = (
            p4_contract.proportional_negative_cap(
                predictive_raw,
                active_missed,
                active_yaw,
                goal_safe_raw=active_goal_safe,
                yaw_exit_raw=active_yaw_exit,
                floor=float(
                    schedule.get(
                        "safety_group_floor",
                        p4_contract.SAFETY_GROUP_FLOOR
                        if single_signal_profile
                        else p4_contract.LEGACY_SAFETY_GROUP_FLOOR,
                    )
                ),
            )
        )
        predictive *= continuous_time_scale
        missed *= continuous_time_scale
        yaw *= continuous_time_scale
        goal_safe *= continuous_time_scale
        yaw_exit *= continuous_time_scale
        components["predictive_collision_risk"] = predictive
        components["missed_safe_direction"] = (
            missed
            if reward_spec.enables("missed_safe_direction")
            else torch.zeros_like(missed)
        )
        components["yaw_cancellation"] = (
            yaw if reward_spec.enables("yaw_cancellation") else torch.zeros_like(yaw)
        )
        components["goal_safe_preference"] = (
            goal_safe
            if reward_spec.enables("goal_safe_preference")
            else torch.zeros_like(goal_safe)
        )
        components["yaw_exit_response"] = (
            yaw_exit
            if reward_spec.enables("yaw_exit_response")
            else torch.zeros_like(yaw_exit)
        )
        return {
            "predictive_raw": predictive_raw,
            "safe3": safe3,
            "safety_valid": safety_valid,
            "missed_shadow_raw": missed_shadow_raw,
            "goal_safe_shadow_raw": goal_safe_shadow_raw,
            "yaw_exit_shadow_raw": yaw_exit_shadow_raw,
            "missed_applied": missed.detach(),
            "goal_safe_applied": goal_safe.detach(),
            "yaw_exit_applied": yaw_exit.detach(),
            "safety_group_cap_hit": (scale < 0.999999).float().detach(),
            "missed_eligible": missed_diagnostics.get(
                "missed_safe_event_active",
                torch.zeros_like(missed_raw),
            ).detach(),
            "scale": scale,
            "missed_diagnostics": missed_diagnostics,
            "goal_safe_diag": goal_safe_diag,
            "yaw_exit_diag": yaw_exit_diag,
            "schedule": schedule,
            "yaw_raw": yaw_raw,
        }

    def _terminal_reward_context(
        self, components, context, reward_context
    ) -> dict[str, object]:
        source_aux = reward_context["source_aux"]
        wall_stuck_terminal = reward_context["wall_stuck_terminal"]
        single_signal_profile = reward_context["single_signal_profile"]
        instant_profile = reward_context["instant_profile"]
        components["success"] = (context["reason"].reshape(-1) == 1).float() * float(
            p4_contract.SUCCESS_IMPULSE
        )
        if single_signal_profile:
            components["failure"] = (
                context["reason"].reshape(-1) == 2
            ).float() * float(p4_contract.FAILURE_IMPULSE)
        if "timeout" in components:
            components["timeout"] = (
                context["reason"].reshape(-1) == 3
            ).float() * float(
                p4_contract.TIMEOUT_IMPULSE
                if single_signal_profile
                else p4_contract.LEGACY_TIMEOUT_IMPULSE
            )
        if single_signal_profile:
            force = torch.nan_to_num(
                source_aux[:, p2_contract.BODY_COLLISION_FORCE_INDEX],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            collision_active = components["body_collision"] < 0.0
            collision_onset = components["body_collision"] < (
                p2_contract.BODY_COLLISION_PERSISTENT - 1.0e-6
            )
            severity = torch.clamp(
                (force - p2_contract.BODY_COLLISION_SOFT_FORCE_N)
                / (
                    p2_contract.BODY_COLLISION_HARD_FORCE_N
                    - p2_contract.BODY_COLLISION_SOFT_FORCE_N
                ),
                0.0,
                1.0,
            )
            components["body_collision"] = torch.where(
                collision_onset,
                (
                    p4_contract.INSTANT_COMMAND_COLLISION_ONSET_BASE
                    if instant_profile
                    else p4_contract.P4_BODY_COLLISION_ONSET_BASE
                )
                + (
                    p4_contract.INSTANT_COMMAND_COLLISION_ONSET_SEVERITY
                    if instant_profile
                    else p4_contract.P4_BODY_COLLISION_ONSET_SEVERITY
                )
                * severity,
                torch.where(
                    collision_active,
                    torch.full_like(
                        severity,
                        (
                            p4_contract.INSTANT_COMMAND_COLLISION_PERSISTENT
                            if instant_profile
                            else p4_contract.P4_BODY_COLLISION_PERSISTENT
                        ),
                    ),
                    torch.zeros_like(severity),
                ),
            )
        stagnation_shadow = reward_context["component_shadows"].get(
            "frontier_stagnation",
            components["frontier_stagnation"].detach().clone(),
        )
        # A confirmed wall-stuck reset is a single terminal event. Recharging
        # collision/predictive/stagnation penalties on that same tick would
        # count the same failure twice and make the new terminal dominate PPO.
        for name in (
            "body_collision",
            "predictive_collision_risk",
            "missed_safe_direction",
            "yaw_cancellation",
            "yaw_exit_response",
            "goal_safe_preference",
            "frontier_stagnation",
        ):
            components[name] = torch.where(
                wall_stuck_terminal,
                torch.zeros_like(components[name]),
                components[name],
            )
        components["stuck_reset"] = wall_stuck_terminal.float() * float(
            self.stuck_reset_contract["terminal_penalty"]
        )
        return {
            "stagnation_shadow": stagnation_shadow,
        }

    def _navigation_reward_context(
        self, components, context, reward_context
    ) -> dict[str, object]:
        duration_frames = reward_context["duration_frames"]
        terminal = reward_context["terminal"]
        instant_profile = reward_context["instant_profile"]
        closed_loop_profile = reward_context["closed_loop_profile"]
        safe3 = reward_context["safe3"]
        safety_valid = reward_context["safety_valid"]
        source_aux = reward_context["source_aux"]
        true_cancel = reward_context["true_cancel"]
        continuous_time_scale = reward_context["continuous_time_scale"]
        maze_profile = reward_context["maze_profile"]
        reward_spec = reward_context["reward_spec"]
        stuck_sustained, stuck_sustained_diag = (
            p4_contract.sustained_wall_stuck_penalty(
                self._p4_worker_extra[:, p4_contract.STUCK_DURATION_S_INDEX],
                self._p4_worker_extra[:, p4_contract.STUCK_CANDIDATE_INDEX] > 0.5,
                self._p4_worker_extra[:, p4_contract.STUCK_MAPPING_VALID_INDEX] > 0.5,
                terminal,
                confirmation_s=float(self.stuck_reset_contract["confirmation_s"]),
                floor=(
                    p4_contract.INSTANT_COMMAND_STUCK_SUSTAINED_FLOOR
                    if instant_profile
                    else (
                        p4_contract.STUCK_SUSTAINED_FLOOR
                        if closed_loop_profile
                        else p4_contract.LEGACY_STUCK_SUSTAINED_FLOOR
                    )
                ),
            )
        )
        stuck_time_scale = torch.clamp(
            duration_frames / float(p4_contract.P4_NAV_PERIOD_FRAMES), 0.0, 1.0
        )
        components["stuck_sustained"] = stuck_sustained * stuck_time_scale
        path_length_m = context.get("path_length_m")
        if path_length_m is None:
            path_length_m = torch.zeros(self.num_envs, device=self.device)
        scene = p4_contract.safety_scene_diagnostics(safe3, safety_valid)
        recovery_active = (
            self._p4_worker_extra[:, p4_contract.STUCK_CANDIDATE_INDEX] > 0.5
        ) & (self._p4_worker_extra[:, p4_contract.STUCK_MAPPING_VALID_INDEX] > 0.5)
        route_excess, route_diag = p4_contract.route_excess_penalty(
            torch.as_tensor(path_length_m, device=self.device).reshape(-1),
            context["start_goal_distance"].reshape(-1),
            context["end_goal_distance"].reshape(-1),
            terminal,
            recovery_active=recovery_active,
            dead_end=scene["teacher_scene_dead_end"] > 0.5,
            goal_freshness=self._last_goal_freshness,
            contact_latch=self.previous_body_collision,
        )
        grace = self.seconds_since_push < 0.30
        route_reward = torch.where(grace, torch.zeros_like(route_excess), route_excess)
        components["route_excess"] = (
            route_reward
            if reward_spec.enables("route_excess")
            else torch.zeros_like(route_reward)
        )
        current_segment = (
            torch.full(
                (self.num_envs,),
                float(len(p4_contract.FULL_TRACK_SEGMENT_LABELS) - 1),
                device=self.device,
            )
            if maze_profile
            else torch.nan_to_num(
                source_aux[:, p2_contract.CURRENT_SEGMENT_INDEX], nan=0.0
            )
            .round()
            .clamp(0, len(p4_contract.FULL_TRACK_SEGMENT_LABELS) - 1)
        )
        initialize_segment = ~self._segment_state_initialized
        if bool(initialize_segment.any()):
            self._spawn_segment[initialize_segment] = current_segment[
                initialize_segment
            ]
            self._max_segment_reached[initialize_segment] = current_segment[
                initialize_segment
            ]
            self._segment_state_initialized[initialize_segment] = True
        segment_frontier, segment_phi_before, segment_phi_after, segment_max_after = (
            p4_contract.segment_frontier_potential(
                self._spawn_segment,
                self._max_segment_reached,
                current_segment,
                duration_frames,
                terminal,
            )
        )
        components["segment_frontier"] = (
            segment_frontier
            if reward_spec.enables("segment_frontier")
            else torch.zeros_like(segment_frontier)
        )
        self._max_segment_reached.copy_(segment_max_after)
        root_x_m = source_aux[:, 15]
        boundary_distance = p4_contract.track_boundary_distance_m(
            root_x_m, current_segment
        )
        positive_progress = torch.clamp(
            context["start_goal_distance"].reshape(-1)
            - context["end_goal_distance"].reshape(-1),
            min=0.0,
        )
        open_path_excess = torch.clamp(
            torch.as_tensor(path_length_m, device=self.device).reshape(-1)
            - positive_progress,
            min=0.0,
            max=0.20,
        )
        open_straight, open_straight_diag = p4_contract.open_straight_penalty(
            self._last_policy_command,
            source_aux[:, 12:15],
            self._p4_worker_extra[:, p4_contract.RAW_GOAL_XY_SLICE],
            safe3,
            safety_valid,
            current_segment,
            boundary_distance,
            terminal,
            junction=scene["teacher_scene_junction"] > 0.5,
            dead_end=scene["teacher_scene_dead_end"] > 0.5,
            contact_or_recovery=(self.previous_body_collision | recovery_active),
            goal_freshness=self._last_goal_freshness,
            yaw_cancellation_value=true_cancel,
            path_excess_m=open_path_excess,
        )
        open_straight_reward = open_straight * continuous_time_scale
        components["open_straight"] = (
            open_straight_reward
            if reward_spec.enables("open_straight")
            else torch.zeros_like(open_straight_reward)
        )
        return {
            "stuck_sustained_diag": stuck_sustained_diag,
            "grace": grace,
            "route_diag": route_diag,
            "segment_phi_before": segment_phi_before,
            "segment_phi_after": segment_phi_after,
            "segment_max_after": segment_max_after,
            "current_segment": current_segment,
            "open_straight_diag": open_straight_diag,
        }

    def _cruise_reward_context(self, components, reward_context) -> dict[str, object]:
        schedule = reward_context["schedule"]
        continuous_time_scale = reward_context["continuous_time_scale"]
        terminal = reward_context["terminal"]
        reward_spec = reward_context["reward_spec"]
        soft_cruise, cruise_diag = p4_contract.soft_cruise_penalty(
            self._last_policy_command,
            self.pending_tick.get(
                "safe3", torch.zeros(self.num_envs, 3, device=self.device)
            ),
            self.pending_tick.get(
                "safety_valid",
                torch.zeros(self.num_envs, 1, device=self.device),
            ).reshape(-1)
            > 0.5,
            self._last_goal_freshness,
            terminal,
            capture_active=self._near_goal_capture_diagnostics.get(
                "near_goal_capture_active",
                torch.zeros(self.num_envs, device=self.device),
            ),
        )
        soft_cruise *= (
            float(schedule.get("cruise_multiplier", 1.0)) * continuous_time_scale
        )
        components["soft_cruise"] = (
            soft_cruise
            if reward_spec.enables("soft_cruise")
            else torch.zeros_like(soft_cruise)
        )
        return {
            "cruise_diag": cruise_diag,
        }

    def _tracking_reward_context(
        self, components, context, reward_context
    ) -> dict[str, object]:
        source_aux = reward_context["source_aux"]
        exec_cmd = reward_context["exec_cmd"]
        terminal = reward_context["terminal"]
        grace = reward_context["grace"]
        capture_active = (
            self._near_goal_capture_diagnostics.get(
                "near_goal_capture_active",
                torch.zeros(self.num_envs, device=self.device),
            )
            > 0.5
        )
        if "crawl" in components:
            components["crawl"] = torch.where(
                capture_active,
                torch.zeros_like(components["crawl"]),
                components["crawl"],
            )
        components["tracking"] = torch.where(
            grace, components["tracking"] * 0.5, components["tracking"]
        )
        response_mae = (source_aux[:, 12:15] - exec_cmd).abs().mean(dim=-1)
        recovered = (
            self._push_recovery_pending
            & (self.seconds_since_push >= 0.30)
            & (response_mae <= 0.12)
            & ~terminal
        )
        self._push_recovery_time[recovered] = self.seconds_since_push[recovered]
        self._push_recovery_pending[recovered | terminal] = False
        tracking_scale = torch.tensor(
            p2_contract.COMMAND_NORMALIZATION,
            device=exec_cmd.device,
            dtype=exec_cmd.dtype,
        )
        tracking_axis_weights = torch.tensor(
            p2_contract.TRACKING_ERROR_AXIS_WEIGHTS,
            device=exec_cmd.device,
            dtype=exec_cmd.dtype,
        )
        tracking_axis_penalty = (
            self._tracking_error_weight()
            * torch.clamp(
                (exec_cmd - source_aux[:, 12:15]) / tracking_scale,
                -1.0,
                1.0,
            ).square()
            * tracking_axis_weights
        )
        tracking_axis_penalty = torch.where(
            terminal.unsqueeze(-1),
            torch.zeros_like(tracking_axis_penalty),
            tracking_axis_penalty,
        )
        tracking_axis_penalty = torch.where(
            grace.unsqueeze(-1),
            tracking_axis_penalty * 0.5,
            tracking_axis_penalty,
        )
        command_rate_axis_penalty = self.pending_tick.get(
            "command_rate_axis_penalty",
            torch.zeros(self.num_envs, 3, device=self.device),
        )
        reward_row_valid = (
            ~torch.as_tensor(
                context.get(
                    "invalid_rows",
                    torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                ),
                device=self.device,
            )
            .reshape(-1)
            .bool()
        )
        reward_row_valid &= (
            ~torch.as_tensor(
                context.get(
                    "unattributed",
                    torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                ),
                device=self.device,
            )
            .reshape(-1)
            .bool()
        )
        tracking_axis_penalty = torch.where(
            reward_row_valid.unsqueeze(-1),
            tracking_axis_penalty,
            torch.zeros_like(tracking_axis_penalty),
        )
        command_rate_axis_penalty = torch.where(
            reward_row_valid.unsqueeze(-1),
            command_rate_axis_penalty,
            torch.zeros_like(command_rate_axis_penalty),
        )
        return {
            "response_mae": response_mae,
            "tracking_axis_penalty": tracking_axis_penalty,
            "command_rate_axis_penalty": command_rate_axis_penalty,
        }

    def _set_p4_reward_diagnostics(self, reward_context, components) -> None:
        self._p4_reward_diagnostics = {
            "reward_predictive_raw": reward_context["predictive_raw"].detach(),
            "reward_missed_safe_raw": reward_context["missed_shadow_raw"],
            "reward_yaw_raw": reward_context["yaw_raw"].detach(),
            "reward_goal_safe_raw": reward_context["goal_safe_shadow_raw"],
            "reward_yaw_exit_raw": reward_context["yaw_exit_shadow_raw"],
            "reward_missed_safe_eligible": reward_context["missed_eligible"],
            "reward_goal_safe_eligible": reward_context["goal_safe_diag"][
                "goal_safe_preference_eligible"
            ].detach(),
            "reward_yaw_exit_eligible": reward_context["yaw_exit_diag"][
                "yaw_exit_response_eligible"
            ].detach(),
            "reward_missed_safe_applied": components[
                "missed_safe_direction"
            ].detach(),
            "reward_goal_safe_applied": components[
                "goal_safe_preference"
            ].detach(),
            "reward_yaw_exit_applied": components[
                "yaw_exit_response"
            ].detach(),
            "reward_safety_group_cap_hit": reward_context[
                "safety_group_cap_hit"
            ],
            "reward_continuous_time_scale": reward_context[
                "continuous_time_scale"
            ].detach(),
            "reward_safety_group_scale": reward_context["scale"].detach(),
            "reward_frontier_stagnation_shadow": reward_context["stagnation_shadow"],
            "maze_new_best_credit": reward_context["maze_credit"].detach(),
            "maze_new_best_delta_m": reward_context["maze_credit_delta"].detach(),
            "maze_new_best_episode_earned": reward_context[
                "maze_credit_after"
            ].detach(),
            "frontier_potential_before": reward_context["maze_credit_before"].detach(),
            "frontier_potential_after": reward_context["maze_credit_after"].detach(),
            "terminal_potential_clawback": reward_context["terminal_clawback"].detach(),
            "reward_command_rate_vx": reward_context["command_rate_axis_penalty"][
                :, 0
            ].detach(),
            "reward_command_rate_vy": reward_context["command_rate_axis_penalty"][
                :, 1
            ].detach(),
            "reward_command_rate_wz": reward_context["command_rate_axis_penalty"][
                :, 2
            ].detach(),
            "reward_tracking_vx": reward_context["tracking_axis_penalty"][
                :, 0
            ].detach(),
            "reward_tracking_vy": reward_context["tracking_axis_penalty"][
                :, 1
            ].detach(),
            "reward_tracking_wz": reward_context["tracking_axis_penalty"][
                :, 2
            ].detach(),
            "reward_command_rate_vy_positive": torch.where(
                self.pending_tick["target_cmd3"][:, 1] >= 0.0,
                reward_context["command_rate_axis_penalty"][:, 1],
                torch.zeros_like(reward_context["command_rate_axis_penalty"][:, 1]),
            ).detach(),
            "reward_command_rate_vy_negative": torch.where(
                self.pending_tick["target_cmd3"][:, 1] < 0.0,
                reward_context["command_rate_axis_penalty"][:, 1],
                torch.zeros_like(reward_context["command_rate_axis_penalty"][:, 1]),
            ).detach(),
            "reward_command_rate_wz_positive": torch.where(
                self.pending_tick["target_cmd3"][:, 2] >= 0.0,
                reward_context["command_rate_axis_penalty"][:, 2],
                torch.zeros_like(reward_context["command_rate_axis_penalty"][:, 2]),
            ).detach(),
            "reward_command_rate_wz_negative": torch.where(
                self.pending_tick["target_cmd3"][:, 2] < 0.0,
                reward_context["command_rate_axis_penalty"][:, 2],
                torch.zeros_like(reward_context["command_rate_axis_penalty"][:, 2]),
            ).detach(),
            "reward_tracking_vy_positive": torch.where(
                reward_context["exec_cmd"][:, 1] >= 0.0,
                reward_context["tracking_axis_penalty"][:, 1],
                torch.zeros_like(reward_context["tracking_axis_penalty"][:, 1]),
            ).detach(),
            "reward_tracking_vy_negative": torch.where(
                reward_context["exec_cmd"][:, 1] < 0.0,
                reward_context["tracking_axis_penalty"][:, 1],
                torch.zeros_like(reward_context["tracking_axis_penalty"][:, 1]),
            ).detach(),
            "reward_tracking_wz_positive": torch.where(
                reward_context["exec_cmd"][:, 2] >= 0.0,
                reward_context["tracking_axis_penalty"][:, 2],
                torch.zeros_like(reward_context["tracking_axis_penalty"][:, 2]),
            ).detach(),
            "reward_tracking_wz_negative": torch.where(
                reward_context["exec_cmd"][:, 2] < 0.0,
                reward_context["tracking_axis_penalty"][:, 2],
                torch.zeros_like(reward_context["tracking_axis_penalty"][:, 2]),
            ).detach(),
            "yaw_exec_cancellation": reward_context["exec_cancel"].detach(),
            "yaw_true_cancellation": reward_context["true_cancel"].detach(),
            "yaw_exec_sign_flip": reward_context["exec_sign_flip"].float(),
            "yaw_true_sign_flip": reward_context["true_sign_flip"].float(),
            "yaw_true_overshoot": torch.clamp(
                reward_context["source_aux"][:, 14].abs()
                - self.pending_tick["target_cmd3"][:, 2].abs(),
                min=0.0,
            ),
            "push_grace_active": reward_context["grace"].float(),
            "push_tracking_response_mae": reward_context["response_mae"],
            **{
                name: value.detach()
                for name, value in reward_context["stuck_sustained_diag"].items()
            },
            **{
                name: value.detach()
                for name, value in reward_context["goal_safe_diag"].items()
            },
            **{
                name: value.detach()
                for name, value in reward_context["yaw_exit_diag"].items()
            },
            **{
                name: value.detach()
                for name, value in reward_context["route_diag"].items()
            },
            "segment_frontier_phi_before": reward_context[
                "segment_phi_before"
            ].detach(),
            "segment_frontier_phi_after": reward_context["segment_phi_after"].detach(),
            "segment_frontier_spawn_segment": self._spawn_segment.detach().clone(),
            "segment_frontier_max_segment": reward_context[
                "segment_max_after"
            ].detach(),
            "current_segment_index": reward_context["current_segment"].detach(),
            **{
                name: value.detach()
                for name, value in reward_context["open_straight_diag"].items()
            },
            **{
                name: value.detach()
                for name, value in reward_context["missed_diagnostics"].items()
            },
            **{
                name: value.detach()
                for name, value in reward_context["cruise_diag"].items()
            },
            **self._translation_limiter_diagnostics,
            **self._near_goal_capture_diagnostics,
        }

    def _override_reward_components(self, components, **context):
        reward_context = self._reward_rollout_context(components, context)
        reward_context.update(self._yaw_reward_context(reward_context))
        reward_context.update(
            self._safety_reward_context(components, context, reward_context)
        )
        reward_context.update(
            self._terminal_reward_context(components, context, reward_context)
        )
        reward_context.update(
            self._navigation_reward_context(components, context, reward_context)
        )
        reward_context.update(self._cruise_reward_context(components, reward_context))
        reward_context.update(
            self._tracking_reward_context(components, context, reward_context)
        )
        self._set_p4_reward_diagnostics(reward_context, components)
        terminal = reward_context["terminal"]
        if bool(terminal.any()):
            self._segment_state_initialized[terminal] = False
            self._spawn_segment[terminal] = 0.0
            self._max_segment_reached[terminal] = 0.0
            self._maze_credit_earned[terminal] = 0.0
        self._goal_epoch_changed_since_tick.zero_()
        return components


def soft_cruise_penalty(
    policy_target_cmd3: torch.Tensor,
    safe3: torch.Tensor,
    teacher_valid: torch.Tensor,
    goal_freshness: torch.Tensor,
    terminal: torch.Tensor,
    capture_active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Softly discourage clear-road cruising outside the preferred vx band."""
    vx = policy_target_cmd3[..., 0]
    center_safe = torch.clamp(safe3[..., 1], 0.0, 1.0)
    best_safe = torch.clamp(safe3.max(dim=-1).values, 0.0, 1.0)
    valid = teacher_valid.to(dtype=vx.dtype).reshape(-1)
    freshness = torch.clamp(goal_freshness.to(dtype=vx.dtype).reshape(-1), 0.0, 1.0)
    clear_factor = (
        valid
        * freshness
        * center_safe
        * torch.clamp(center_safe / (best_safe + 1.0e-6), 0.0, 1.0)
    )
    low_error = torch.relu((SOFT_CRUISE_MIN_VX - vx) / SOFT_CRUISE_MIN_VX)
    high_error = torch.relu(
        (vx - SOFT_CRUISE_MAX_VX) / (P4_MAX_VX - SOFT_CRUISE_MAX_VX)
    )
    penalty = (
        SOFT_CRUISE_LOW_WEIGHT * clear_factor * low_error.square()
        + SOFT_CRUISE_HIGH_WEIGHT * high_error.square()
    )
    disabled = terminal.reshape(-1).bool()
    if capture_active is not None:
        disabled |= (
            torch.as_tensor(capture_active, device=disabled.device).reshape(-1).bool()
        )
    penalty = torch.where(disabled, torch.zeros_like(penalty), penalty)
    return penalty, {
        "soft_cruise_clear_factor": clear_factor,
        "soft_cruise_low_error": low_error,
        "soft_cruise_high_error": high_error,
    }


def goal_safe_direction_penalty(
    safe3: torch.Tensor,
    target_cmd3: torch.Tensor,
    goal_xy_m: torch.Tensor,
    teacher_valid: torch.Tensor,
    goal_freshness: torch.Tensor,
    terminal: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer the goal-facing option only among directions the teacher deems safe."""
    if safe3.ndim != 2 or safe3.shape[1] != 3:
        raise ValueError("P4 goal-safe preference expects safe3=[N,3]")
    if goal_xy_m.ndim != 2 or goal_xy_m.shape[1] != 2:
        raise ValueError("P4 goal-safe preference expects goal_xy_m=[N,2]")
    command_weights = p2_contract.command_direction_weights(target_cmd3)
    goal = torch.nan_to_num(goal_xy_m.float(), nan=0.0, posinf=0.0, neginf=0.0)
    bearing = torch.atan2(goal[:, 1], goal[:, 0]).clamp(
        min=math.radians(-80.0), max=math.radians(80.0)
    )
    centers = torch.deg2rad(
        torch.tensor(
            p2_contract.PREDICTIVE_COLLISION_SECTOR_CENTERS_DEG,
            device=goal.device,
            dtype=goal.dtype,
        )
    )
    width = math.radians(p2_contract.PREDICTIVE_COLLISION_SECTOR_WIDTH_DEG)
    goal_weights = torch.softmax(
        -0.5 * ((bearing[:, None] - centers[None, :]) / width).square(), dim=-1
    )
    score3 = torch.clamp(safe3, 0.0, 1.0) * goal_weights
    selected = (command_weights * score3).sum(dim=-1)
    best = score3.max(dim=-1).values
    gap = torch.relu(best - selected - GOAL_SAFE_PREFERENCE_MARGIN)
    top2 = torch.topk(score3, k=2, dim=-1).values
    distinct = (top2[:, 0] - top2[:, 1]) > GOAL_SAFE_PREFERENCE_MARGIN
    speed_scale = target_cmd3.new_tensor((1.0, 1.0, p2_contract.CRAWL_BODY_RADIUS_M))
    moving = torch.linalg.vector_norm(target_cmd3 * speed_scale, dim=-1) > 0.05
    goal_distance = torch.linalg.vector_norm(goal, dim=-1)
    eligible = (
        teacher_valid.reshape(-1).bool()
        & (goal_freshness.reshape(-1) >= TEACHER_GOAL_FRESHNESS_MIN)
        & moving
        & distinct
        & (safe3.max(dim=-1).values >= p2_contract.SAFE_DIRECTION_MIN_BEST_SAFE)
        & (goal_distance >= 0.60)
        & ~terminal.reshape(-1).bool()
        & (gap > 0.0)
    )
    normalized_gap = torch.clamp(gap / GOAL_SAFE_PREFERENCE_SCALE, 0.0, 1.0)
    penalty = torch.where(
        eligible,
        GOAL_SAFE_PREFERENCE_WEIGHT * normalized_gap,
        torch.zeros_like(normalized_gap),
    )
    return penalty, {
        "goal_safe_preference_eligible": eligible.float(),
        "goal_safe_preference_gap": gap,
        "goal_safe_preference_selected": selected,
        "goal_safe_preference_best": best,
    }


def yaw_exit_response_penalty(
    safe3: torch.Tensor,
    target_cmd3: torch.Tensor,
    goal_xy_m: torch.Tensor,
    teacher_valid: torch.Tensor,
    goal_freshness: torch.Tensor,
    terminal: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Require a small correctly signed yaw response at a clear side exit."""
    if safe3.ndim != 2 or safe3.shape[1] != 3:
        raise ValueError("P4 yaw-exit response expects safe3=[N,3]")
    if target_cmd3.shape != (safe3.shape[0], 3):
        raise ValueError("P4 yaw-exit response expects target_cmd3=[N,3]")
    if goal_xy_m.shape != (safe3.shape[0], 2):
        raise ValueError("P4 yaw-exit response expects goal_xy_m=[N,2]")

    values = torch.nan_to_num(
        safe3.to(target_cmd3), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    goal = torch.nan_to_num(goal_xy_m.to(target_cmd3), nan=0.0, posinf=0.0, neginf=0.0)
    bearing = torch.atan2(goal[:, 1], goal[:, 0]).clamp(
        min=math.radians(-80.0), max=math.radians(80.0)
    )
    centers = target_cmd3.new_tensor(
        p2_contract.PREDICTIVE_COLLISION_SECTOR_CENTERS_DEG
    ) * (math.pi / 180.0)
    width = math.radians(p2_contract.PREDICTIVE_COLLISION_SECTOR_WIDTH_DEG)
    goal_weights = torch.softmax(
        -0.5 * ((bearing[:, None] - centers[None, :]) / width).square(), dim=-1
    )
    score3 = values * goal_weights
    top2 = torch.topk(score3, k=2, dim=-1).values
    best_index = score3.argmax(dim=-1)
    side_exit = best_index != 1
    desired_sign = torch.where(
        best_index == 0,
        torch.ones_like(bearing),
        -torch.ones_like(bearing),
    )
    signed_wz = desired_sign * target_cmd3[:, 2]
    response_error = torch.clamp((0.10 - signed_wz) / 0.20, 0.0, 1.0)
    goal_distance = torch.linalg.vector_norm(goal, dim=-1)
    eligible = (
        teacher_valid.reshape(-1).bool()
        & (goal_freshness.reshape(-1) >= TEACHER_GOAL_FRESHNESS_MIN)
        & ~terminal.reshape(-1).bool()
        & side_exit
        & (values.max(dim=-1).values >= TEACHER_SAFE_MIN)
        & ((top2[:, 0] - top2[:, 1]) >= GOAL_SAFE_PREFERENCE_MARGIN)
        & (goal_distance >= NEAR_GOAL_CAPTURE_MAX_DISTANCE_M)
        & (target_cmd3[:, 0] >= 0.15)
        & (target_cmd3[:, 1].abs() <= 0.10)
        & (response_error > 0.0)
    )
    penalty = torch.where(
        eligible,
        YAW_EXIT_RESPONSE_RAW_FLOOR * response_error,
        torch.zeros_like(response_error),
    )
    return penalty, {
        "yaw_exit_response_eligible": eligible.float(),
        "yaw_exit_response_error": response_error,
        "yaw_exit_response_desired_sign": torch.where(
            side_exit, desired_sign, torch.zeros_like(desired_sign)
        ),
        "yaw_exit_response_signed_wz": signed_wz,
    }


def maze_missed_safe_direction_penalty(
    safe3: torch.Tensor,
    target_cmd3: torch.Tensor,
    teacher_valid: torch.Tensor,
    session_effective_seconds: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize a clear missed alternative with event-normalized severity."""
    weights = p2_contract.command_direction_weights(target_cmd3)
    selected_safe = (weights * safe3).sum(dim=-1)
    best_safe = safe3.max(dim=-1).values
    chosen_risk = torch.clamp(1.0 - selected_safe, 0.0, 1.0)
    safe_gap = torch.relu(
        best_safe - selected_safe - p2_contract.SAFE_DIRECTION_GAP_MARGIN
    )
    normalized_gap = torch.clamp(safe_gap / SAFE_DIRECTION_GAP_SCALE, 0.0, 1.0)
    speed_scale = target_cmd3.new_tensor((1.0, 1.0, p2_contract.CRAWL_BODY_RADIUS_M))
    moving = torch.linalg.vector_norm(target_cmd3 * speed_scale, dim=-1) > 0.05
    active = (
        teacher_valid.reshape(-1).bool()
        & moving
        & (best_safe >= p2_contract.SAFE_DIRECTION_MIN_BEST_SAFE)
        & (safe_gap > 0.0)
    )
    weight = safe_direction_weight(session_effective_seconds)
    severity = chosen_risk * normalized_gap
    penalty = torch.where(
        active,
        torch.clamp(-weight * severity, min=-weight, max=0.0),
        torch.zeros_like(severity),
    )
    return penalty, {
        "missed_safe_event_active": active.float(),
        "missed_safe_event_severity": severity,
        "missed_safe_weight": severity.new_full(severity.shape, weight),
    }
