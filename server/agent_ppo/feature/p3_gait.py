#!/usr/bin/env python3
"""Training-only P3 gait baselines, event rewards and mirror consistency."""

from __future__ import annotations

import hashlib
import json
import torch
import torch.nn.functional as F

from agent_ppo.feature import p2_contract, p3_contract


LEG_SWAP = torch.tensor([1, 0, 3, 2], dtype=torch.long)
# The platform exposes joints axis-major: FL/FR/RL/RR hip, then thigh, then calf.
JOINT_SWAP = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], dtype=torch.long)
JOINT_SIGN = torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)


class P35LowRewardShaper:
    """Parent-envelope P3.5 shaping with fail-closed training signals."""

    CONTRACT = "p35_low_reward_baseline_v2"
    MAX_SAMPLES = 65536

    def __init__(self, num_envs: int, device):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.samples = {
            name: []
            for name in ("joint_pos", "joint_acc_noncontact", "joint_acc_onset", "posture", "frequency")
        }
        self.sample_count = {name: 0 for name in self.samples}
        self.thresholds: dict[str, torch.Tensor] = {}
        self.finalized = False
        self.valid = False
        self.frames_since_onset = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.onset_ema = torch.zeros_like(self.frames_since_onset)

    def _append(self, name: str, values: torch.Tensor) -> None:
        remaining = self.MAX_SAMPLES - self.sample_count[name]
        if remaining <= 0 or values.numel() == 0:
            return
        selected = values.detach()[:remaining].to(self.device)
        self.samples[name].append(selected)
        self.sample_count[name] += selected.shape[0]

    def observe(
        self,
        proprio: torch.Tensor,
        aux: torch.Tensor,
        extra: torch.Tensor,
        healthy: torch.Tensor,
    ) -> None:
        if self.finalized or not bool(healthy.any()):
            return
        valid_joint = extra[:, p3_contract.JOINT_ACCELERATION_MAPPING_VALID_INDEX] > 0.5
        valid_contact = extra[:, p3_contract.CONTACT_REWARD_MAPPING_VALID_INDEX] > 0.5
        selected = healthy & valid_joint & valid_contact
        if not bool(selected.any()):
            return
        joint_acc = extra[:, p3_contract.JOINT_ACCELERATION_SLICE].abs()
        foot_onset = extra[:, p3_contract.GAIT_CONTACT_ONSET_SLICE] > 0.5
        joint_onset = foot_onset.repeat(1, 3)
        noncontact_rows = selected & ~joint_onset.any(dim=-1)
        onset_rows = selected & joint_onset.any(dim=-1)
        self._append("joint_pos", proprio[selected, 9:21].abs())
        self._append("joint_acc_noncontact", joint_acc[noncontact_rows])
        onset_acc = joint_acc.masked_fill(~joint_onset, float("nan"))
        self._append("joint_acc_onset", onset_acc[onset_rows])
        posture = torch.stack(
            (
                aux[:, 22].abs(),
                aux[:, 18].abs(),
                aux[:, 19].abs(),
                aux[:, 21].abs(),
            ),
            dim=-1,
        )
        self._append("posture", posture[selected])
        self._append("frequency", aux[selected, p2_contract.GAIT_STEP_FREQUENCY_SLICE])

    def finalize(self) -> None:
        if self.finalized:
            return
        required = ("joint_pos", "joint_acc_noncontact", "posture", "frequency")
        self.valid = all(self.sample_count[name] >= 128 for name in required)
        if self.valid:
            for name in required:
                values = torch.cat(self.samples[name], dim=0).float()
                quantile = 0.99 if name == "joint_pos" else (0.05 if name == "frequency" else 0.95)
                self.thresholds[name] = torch.quantile(values, quantile, dim=0)
            if self.sample_count["joint_acc_onset"] >= 128:
                onset = torch.cat(self.samples["joint_acc_onset"], dim=0).float()
                fallback = 1.5 * self.thresholds["joint_acc_noncontact"]
                values = []
                for joint_index in range(onset.shape[-1]):
                    joint_values = onset[:, joint_index]
                    joint_values = joint_values[torch.isfinite(joint_values)]
                    values.append(
                        torch.quantile(joint_values, 0.99)
                        if joint_values.numel() >= 32
                        else fallback[joint_index]
                    )
                self.thresholds["joint_acc_onset"] = torch.stack(values)
            else:
                self.thresholds["joint_acc_onset"] = 1.5 * self.thresholds[
                    "joint_acc_noncontact"
                ]
        self.samples = {name: [] for name in self.samples}
        self.finalized = True

    @staticmethod
    def _huber_excess(excess: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        absolute = excess.abs()
        return torch.where(
            absolute <= delta,
            0.5 * absolute.square(),
            delta * (absolute - 0.5 * delta),
        )

    def rewards(
        self,
        *,
        proprio: torch.Tensor,
        aux: torch.Tensor,
        next_aux: torch.Tensor,
        extra: torch.Tensor,
        command: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = command.new_zeros(command.shape[0])
        if not self.finalized or not self.valid:
            return zero, {
                name: zero.clone()
                for name in ("progress", "default_posture", "joint_acc", "contact", "gait", "posture")
            }
        joint_valid = extra[:, p3_contract.JOINT_ACCELERATION_MAPPING_VALID_INDEX] > 0.5
        contact_valid = extra[:, p3_contract.CONTACT_REWARD_MAPPING_VALID_INDEX] > 0.5
        finite = (
            torch.isfinite(extra).all(dim=-1)
            & torch.isfinite(aux).all(dim=-1)
            & torch.isfinite(next_aux).all(dim=-1)
            & torch.isfinite(proprio).all(dim=-1)
            & torch.isfinite(command).all(dim=-1)
        )
        valid = joint_valid & contact_valid & finite
        moving = command[:, :2].norm(dim=-1) > 0.05
        yaw = aux[:, 17]
        world_command = torch.stack(
            (
                torch.cos(yaw) * command[:, 0] - torch.sin(yaw) * command[:, 1],
                torch.sin(yaw) * command[:, 0] + torch.cos(yaw) * command[:, 1],
            ),
            dim=-1,
        )
        direction = F.normalize(world_command, dim=-1, eps=1.0e-6)
        velocity = (next_aux[:, 15:17] - aux[:, 15:17]) / p2_contract.CONTROL_DT_S
        posture_ok = aux[:, 21:23].abs().amax(dim=-1) < 0.50
        progress = 0.15 * (velocity * direction).sum(dim=-1)
        progress = torch.clamp(progress, -0.15, 0.15)
        progress = torch.where(valid & moving & posture_ok & ~dones, progress, zero)

        joint_limit = 1.10 * self.thresholds["joint_pos"].clamp_min(1.0e-4)
        joint_excess = torch.relu(proprio[:, 9:21].abs() - joint_limit) / joint_limit
        default_posture = -self._huber_excess(joint_excess).mean(dim=-1)
        default_posture *= torch.where(moving, 0.60, 1.0)
        default_posture = default_posture.clamp(-0.04, 0.0)

        joint_acc = extra[:, p3_contract.JOINT_ACCELERATION_SLICE].abs()
        foot_onset = extra[:, p3_contract.GAIT_CONTACT_ONSET_SLICE] > 0.5
        joint_onset = foot_onset.repeat(1, 3)
        noncontact_limit = 1.10 * self.thresholds["joint_acc_noncontact"].clamp_min(1.0)
        onset_limit = torch.maximum(
            self.thresholds["joint_acc_onset"], 1.5 * self.thresholds["joint_acc_noncontact"]
        )
        onset_limit = onset_limit.clone()
        onset_limit[8:12] *= 1.20
        acceleration_limit = torch.where(joint_onset, onset_limit, noncontact_limit)
        acceleration_excess = torch.relu(joint_acc - acceleration_limit) / acceleration_limit
        joint_acc_reward = -acceleration_excess.square().mean(dim=-1).clamp(max=0.08)

        force = extra[:, p3_contract.CONTACT_FORCE_SLICE]
        onset = extra[:, p3_contract.CONTACT_ONSET_SLICE] > 0.5
        duration = extra[:, p3_contract.CONTACT_OVER_THRESHOLD_DURATION_SLICE]
        contact_event = torch.zeros_like(onset)
        contact_event[:, :2] = onset[:, :2]
        contact_event[:, 2:] = (
            ((duration[:, 2:] >= 0.060) & (duration[:, 2:] < 0.080))
        )
        calf_slots = torch.tensor([4, 7, 10, 13], device=extra.device)
        contact_event[:, calf_slots] = (
            (duration[:, calf_slots] >= 0.100)
            & (duration[:, calf_slots] < 0.120)
        )
        thresholds = force.new_tensor(
            [10.0, 10.0, 25.0, 25.0, 35.0, 25.0, 25.0, 35.0,
             25.0, 25.0, 35.0, 25.0, 25.0, 35.0]
        )
        contact_excess = torch.relu(force - thresholds) / thresholds
        contact_reward = -(
            contact_excess.square() * contact_event.to(force)
        ).sum(dim=-1).clamp(max=0.06)

        foot_onset = extra[:, p3_contract.GAIT_CONTACT_ONSET_SLICE] > 0.5
        self.frames_since_onset = torch.where(
            foot_onset,
            torch.zeros_like(self.frames_since_onset),
            self.frames_since_onset + 1.0,
        )
        alpha = p2_contract.CONTROL_DT_S / 1.5
        self.onset_ema.mul_(1.0 - alpha).add_(foot_onset.to(command) * alpha)
        frequency_floor = self.thresholds["frequency"].clamp_min(0.05)
        observed_frequency = aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE]
        frequency_deficit = torch.relu(frequency_floor - observed_frequency) / frequency_floor
        prolonged = torch.relu(self.frames_since_onset * p2_contract.CONTROL_DT_S - 1.0) / 0.5
        diagonal = (self.onset_ema[:, [0, 3]] - self.onset_ema[:, [1, 2]]).abs().mean(dim=-1)
        turn_scale = (1.0 - 0.5 * (command[:, 1].abs() + command[:, 2].abs()).clamp(0.0, 1.0))
        gait_reward = -0.035 * torch.maximum(
            torch.maximum(frequency_deficit.amax(dim=-1), prolonged.amax(dim=-1).clamp(max=1.0)),
            diagonal.clamp(max=1.0),
        ) * turn_scale

        posture_values = torch.stack(
            (
                aux[:, 22].abs(),
                aux[:, 18].abs(),
                aux[:, 19].abs(),
                aux[:, 21].abs(),
            ),
            dim=-1,
        )
        posture_limit = self.thresholds["posture"].clamp_min(1.0e-3)
        posture_excess = torch.relu(posture_values - posture_limit) / posture_limit
        posture_weights = posture_excess.new_tensor((1.0, 1.0, 0.65, 0.30))
        posture_reward = -(
            self._huber_excess(posture_excess) * posture_weights
        ).sum(dim=-1).div(posture_weights.sum()).clamp(max=0.08)

        grace = extra[:, p3_contract.SECONDS_SINCE_PUSH_INDEX] <= 0.40
        grace_scale = torch.where(grace, 0.5, 1.0)
        joint_acc_reward *= grace_scale
        gait_reward *= grace_scale
        posture_reward *= grace_scale
        components = {
            "progress": progress,
            "default_posture": default_posture,
            "joint_acc": joint_acc_reward,
            "contact": contact_reward,
            "gait": gait_reward,
            "posture": posture_reward,
        }
        for name in components:
            components[name] = torch.where(valid, components[name], zero)
        self.frames_since_onset[dones] = 0.0
        self.onset_ema[dones] = 0.0
        return sum(components.values(), zero.clone()), components

    def state_dict(self) -> dict:
        return {
            "contract": self.CONTRACT,
            "finalized": self.finalized,
            "valid": self.valid,
            "sample_count": dict(self.sample_count),
            "thresholds": {name: value.detach().cpu() for name, value in self.thresholds.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("contract") != self.CONTRACT:
            raise ValueError("P3.5 reward baseline checkpoint contract mismatch")
        self.finalized = bool(state.get("finalized", False))
        self.valid = bool(state.get("valid", False))
        self.sample_count.update(state.get("sample_count") or {})
        self.thresholds = {
            name: value.to(self.device) for name, value in (state.get("thresholds") or {}).items()
        }


def validate_joint_order(names) -> bool:
    names = [str(name).lower() for name in names]
    if len(names) != 12:
        return False
    mirror_leg = {"fl": "fr", "fr": "fl", "rl": "rr", "rr": "rl"}
    parsed = []
    for name in names:
        leg = next((item for item in mirror_leg if item in name), None)
        axis = next((item for item in ("hip", "thigh", "calf") if item in name), None)
        if leg is None or axis is None:
            return False
        parsed.append((leg, axis))
    permutation = JOINT_SWAP.tolist()
    return all(
        parsed[permutation[index]] == (mirror_leg[leg], axis)
        for index, (leg, axis) in enumerate(parsed)
    )


def _mirrored_parameter_is_symmetric(value, *, permutation=JOINT_SWAP) -> bool:
    if value is None:
        return False
    try:
        tensor = torch.as_tensor(value).detach().float()
    except (TypeError, ValueError):
        return False
    if tensor.numel() == 1:
        return bool(torch.isfinite(tensor).all())
    if tensor.shape[-1] != 12 or not bool(torch.isfinite(tensor).all()):
        return False
    tensor = tensor.reshape(-1, 12)
    mirrored = tensor.index_select(-1, permutation.to(tensor.device))
    return bool(torch.allclose(tensor.abs(), mirrored.abs(), rtol=1.0e-4, atol=1.0e-6))


def _joint_position_action_term(action_manager):
    """Resolve the joint-position term without accepting an unrelated action term."""
    terms = getattr(action_manager, "_terms", None)
    if not terms:
        return None
    if hasattr(terms, "items"):
        exact = [
            term
            for name, term in terms.items()
            if str(name).lower() == "jointpositionaction"
        ]
        exact = [term for term in exact if getattr(term, "action_dim", 12) == 12]
        if len(exact) == 1:
            return exact[0]
        candidates = [
            term
            for name, term in terms.items()
            if "jointpositionaction" in str(name).lower()
            or "jointpositionaction" in type(term).__name__.lower()
        ]
    else:
        candidates = [
            term
            for term in terms
            if "jointpositionaction" in type(term).__name__.lower()
        ]
    candidates = [
        term
        for term in candidates
        if getattr(term, "action_dim", 12) == 12
    ]
    return candidates[0] if len(candidates) == 1 else None


def _joint_action_scale_matches_contract(term) -> bool:
    if term is None:
        return False
    cfg = getattr(term, "cfg", None)
    value = getattr(term, "_scale", None)
    if value is None:
        value = getattr(cfg, "scale", getattr(term, "scale", None))
    if not _mirrored_parameter_is_symmetric(value):
        return False
    try:
        scale = torch.as_tensor(value).detach().float().abs()
    except (TypeError, ValueError):
        return False
    expected = torch.full_like(scale, p3_contract.ACTION_TO_JOINT_SCALE)
    return bool(torch.allclose(scale, expected, rtol=1.0e-4, atol=1.0e-6))


def validate_mirror_assembly(robot, env=None) -> tuple[bool, dict[str, bool]]:
    """Fail-safe validation for every physical quantity used by mirror training."""
    data = getattr(robot, "data", None)
    names = getattr(data, "joint_names", None)
    if not names:
        names = getattr(robot, "joint_names", ())
    checks = {"joint_order": validate_joint_order(names)}
    aliases = {
        "stiffness": ("joint_stiffness", "default_joint_stiffness"),
        "damping": ("joint_damping", "default_joint_damping"),
        "effort_limit": ("joint_effort_limits", "soft_joint_effort_limits"),
    }
    for label, candidates in aliases.items():
        value = next(
            (getattr(data, name) for name in candidates if getattr(data, name, None) is not None),
            None,
        )
        checks[label] = _mirrored_parameter_is_symmetric(value)

    action_manager = getattr(env, "action_manager", None)
    joint_action = _joint_position_action_term(action_manager)
    checks["action_scale"] = _joint_action_scale_matches_contract(joint_action)
    return all(checks.values()), checks


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    permutation = JOINT_SWAP.to(action.device)
    signs = JOINT_SIGN.to(action)
    return action.index_select(-1, permutation) * signs


def per_leg_action_mse(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or prediction.shape[-1] != 12:
        raise ValueError("P3 per-leg action error requires matching 12-column tensors")
    error = (prediction - target).square()
    return {
        leg: error[..., indices].mean()
        for leg, indices in {
            "fl": (0, 4, 8),
            "fr": (1, 5, 9),
            "rl": (2, 6, 10),
            "rr": (3, 7, 11),
        }.items()
    }


def mirror_proprio(proprio: torch.Tensor) -> torch.Tensor:
    if proprio.shape[-1] != 45:
        raise ValueError("P3 mirror proprio must have 45 columns")
    result = proprio.clone()
    result[..., 0] = -proprio[..., 0]
    result[..., 2] = -proprio[..., 2]
    result[..., 4] = -proprio[..., 4]
    result[..., 7] = -proprio[..., 7]
    result[..., 8] = -proprio[..., 8]
    for start in (9, 21, 33):
        values = proprio[..., start : start + 12]
        result[..., start : start + 12] = mirror_action(values)
    return result


def mirror_scan_from_coordinates(scan: torch.Tensor, lateral_coordinates: torch.Tensor) -> torch.Tensor:
    if scan.shape[-1] != lateral_coordinates.numel():
        raise ValueError("scan and lateral coordinate sizes differ")
    target = -lateral_coordinates.to(scan)
    distance = (lateral_coordinates.to(scan).unsqueeze(0) - target.unsqueeze(1)).abs()
    permutation = distance.argmin(dim=1)
    return scan.index_select(-1, permutation)


class P3GaitBaseline:
    """Collect terrain/motion envelopes, then freeze event-only rewards."""

    CONTRACT_VERSION = p3_contract.GAIT_BASELINE_VERSION
    NAMES = ("slip", "impact", "margin", "stance", "frequency", "duty")
    TERRAIN_BUCKETS = ("slope", "slope_inv", "stairs", "stairs_inv")
    MOTION_BUCKETS = ("low_speed", "forward", "turn_lateral")
    MIN_SAMPLES = 128
    MAX_SAMPLES_PER_BUCKET = 65536

    def __init__(self, device):
        self.device = torch.device(device)
        self.samples = {
            (terrain, motion, name): []
            for terrain in range(len(self.TERRAIN_BUCKETS))
            for motion in range(len(self.MOTION_BUCKETS))
            for name in self.NAMES
        }
        self.sample_counts = {key: 0 for key in self.samples}
        self.values = {}
        self.valid = {}
        self.fallback_levels = {}
        self.finalized = False
        self.fallback_share = 0.0
        self.total_samples = 0

    @staticmethod
    def _terrain_bucket(aux: torch.Tensor) -> torch.Tensor:
        column = aux[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX].round().long()
        result = torch.full_like(column, -1)
        first, second, third, fourth = p3_contract.TERRAIN_COLUMN_BUCKET_BOUNDARIES
        result[(column >= 0) & (column < first)] = 0
        result[(column >= first) & (column < second)] = 1
        result[(column >= second) & (column < third)] = 2
        result[(column >= third) & (column < fourth)] = 3
        return result

    @staticmethod
    def _motion_bucket(
        aux: torch.Tensor, extra: torch.Tensor | None = None
    ) -> torch.Tensor:
        command = aux[:, 3:6]
        vx, vy, wz = command.unbind(-1)
        result = torch.full_like(vx, 1, dtype=torch.long)
        invalid = vx < -1.0e-6
        turn_lateral = (~invalid) & ((vy.abs() > 0.05) | (wz.abs() > 0.10))
        low_speed = (~invalid) & (~turn_lateral) & (vx < 0.20)
        result[turn_lateral] = 2
        result[low_speed] = 0
        result[invalid] = -1
        if torch.is_tensor(extra) and extra.shape[1] > p3_contract.COMMAND_BUCKET_INDEX:
            command_bucket = extra[:, p3_contract.COMMAND_BUCKET_INDEX].round().long()
            result[(command_bucket == 5) | (command_bucket == 6)] = -1
        return result

    def _append(self, key, value: torch.Tensor) -> None:
        remaining = self.MAX_SAMPLES_PER_BUCKET - self.sample_counts[key]
        if remaining <= 0 or value.numel() == 0:
            return
        value = value.detach().reshape(-1).float().cpu()[:remaining]
        if value.numel():
            self.samples[key].append(value)
            self.sample_counts[key] += int(value.numel())
            self.total_samples += int(value.numel())

    def observe(
        self,
        aux: torch.Tensor,
        extra: torch.Tensor,
        healthy: torch.Tensor,
        *,
        collect_continuous: bool = True,
    ) -> None:
        if self.finalized or not bool(healthy.any()):
            return
        terrain = self._terrain_bucket(aux)
        motion = self._motion_bucket(aux, extra)
        onset = extra[:, p3_contract.GAIT_CONTACT_ONSET_SLICE] > 0.5
        slip_event = extra[:, p3_contract.GAIT_COMPLETED_SLIP_EVENT_SLICE] > 0.5
        for terrain_index in range(len(self.TERRAIN_BUCKETS)):
            for motion_index in range(len(self.MOTION_BUCKETS)):
                selected = healthy & (terrain == terrain_index) & (motion == motion_index)
                if not bool(selected.any()):
                    continue
                fields = {
                    "slip": extra[selected, p3_contract.GAIT_COMPLETED_SLIP_SLICE][
                        slip_event[selected]
                    ],
                    "impact": extra[selected, p3_contract.GAIT_IMPACT_SPEED_SLICE][onset[selected]],
                    "margin": extra[selected, p3_contract.GAIT_TOUCHDOWN_Y_SLICE][onset[selected]],
                }
                if collect_continuous:
                    fields.update(
                        stance=extra[selected, p3_contract.GAIT_CONTINUOUS_STANCE_SLICE],
                        frequency=aux[selected, p2_contract.GAIT_STEP_FREQUENCY_SLICE],
                        duty=aux[selected, p2_contract.GAIT_DUTY_SLICE],
                    )
                for name, value in fields.items():
                    self._append((terrain_index, motion_index, name), value)

    def finalize(self) -> None:
        if self.finalized:
            return
        quantiles = {
            "slip": 0.95,
            "impact": 0.95,
            "margin": 0.05,
            "stance": 0.95,
            "frequency": 0.05,
            "duty": 0.95,
        }
        combined = {
            key: (torch.cat(parts) if parts else torch.empty(0))
            for key, parts in self.samples.items()
        }
        terrain_combined = {
            (terrain, name): torch.cat(
                [
                    combined[(terrain, motion, name)]
                    for motion in range(len(self.MOTION_BUCKETS))
                ]
            )
            for terrain in range(len(self.TERRAIN_BUCKETS))
            for name in self.NAMES
        }
        global_combined = {
            name: torch.cat(
                [
                    combined[(terrain, motion, name)]
                    for terrain in range(len(self.TERRAIN_BUCKETS))
                    for motion in range(len(self.MOTION_BUCKETS))
                ]
            )
            for name in self.NAMES
        }
        fallback_count = 0
        total = 0
        for terrain in range(len(self.TERRAIN_BUCKETS)):
            for motion in range(len(self.MOTION_BUCKETS)):
                for name in self.NAMES:
                    total += 1
                    exact = combined[(terrain, motion, name)]
                    terrain_values = terrain_combined[(terrain, name)]
                    global_values = global_combined[name]
                    candidates = ((exact, 0), (terrain_values, 1), (global_values, 2))
                    selected = next(
                        ((values, level) for values, level in candidates if values.numel() >= self.MIN_SAMPLES),
                        None,
                    )
                    key = (terrain, motion, name)
                    if selected is None:
                        self.values[key] = 0.0
                        self.valid[key] = False
                        self.fallback_levels[key] = 3
                        fallback_count += 1
                    else:
                        values, level = selected
                        self.values[key] = float(torch.quantile(values, quantiles[name]))
                        self.valid[key] = True
                        self.fallback_levels[key] = level
                        fallback_count += int(level > 0)
        self.fallback_share = fallback_count / max(total, 1)
        self.samples = {}
        self.finalized = True

    def fallback_level_shares(self) -> dict[str, float]:
        total = max(len(self.fallback_levels), 1)
        counts = {
            level: sum(value == level for value in self.fallback_levels.values())
            for level in range(4)
        }
        return {
            "exact": counts[0] / total,
            "terrain": counts[1] / total,
            "global": counts[2] / total,
            "disabled": counts[3] / total,
        }

    def _thresholds(
        self, aux: torch.Tensor, extra: torch.Tensor, name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        terrain = self._terrain_bucket(aux)
        motion = self._motion_bucket(aux, extra)
        threshold = torch.zeros(aux.shape[0], device=aux.device, dtype=aux.dtype)
        valid = torch.zeros(aux.shape[0], device=aux.device, dtype=torch.bool)
        confidence = torch.zeros(aux.shape[0], device=aux.device, dtype=aux.dtype)
        for terrain_index in range(len(self.TERRAIN_BUCKETS)):
            for motion_index in range(len(self.MOTION_BUCKETS)):
                selected = (terrain == terrain_index) & (motion == motion_index)
                key = (terrain_index, motion_index, name)
                if bool(selected.any()) and self.valid.get(key, False):
                    threshold[selected] = self.values[key]
                    valid[selected] = True
                    level = self.fallback_levels.get(key, 3)
                    confidence[selected] = 1.0 if level == 0 else 0.5 if level == 1 else 0.0
        return (
            threshold.unsqueeze(-1),
            valid.unsqueeze(-1),
            confidence.unsqueeze(-1),
        )

    def rewards(self, aux: torch.Tensor, extra: torch.Tensor, scale: float):
        zeros = torch.zeros(aux.shape[0], device=aux.device)
        if not self.finalized or scale <= 0.0:
            return zeros, zeros, zeros
        sensor_valid = (aux[:, p2_contract.GAIT_VALID_INDEX] > 0.5).float()
        onset = (extra[:, p3_contract.GAIT_CONTACT_ONSET_SLICE] > 0.5).float()
        slip_event = (
            extra[:, p3_contract.GAIT_COMPLETED_SLIP_EVENT_SLICE] > 0.5
        ).float()
        slip_threshold, slip_valid, slip_conf = self._thresholds(aux, extra, "slip")
        impact_threshold, impact_valid, impact_conf = self._thresholds(aux, extra, "impact")
        margin_threshold, margin_valid, margin_conf = self._thresholds(aux, extra, "margin")
        stance_threshold, stance_valid, stance_conf = self._thresholds(aux, extra, "stance")
        frequency_threshold, frequency_valid, frequency_conf = self._thresholds(aux, extra, "frequency")
        duty_threshold, duty_valid, duty_conf = self._thresholds(aux, extra, "duty")
        slip = F.relu(
            extra[:, p3_contract.GAIT_COMPLETED_SLIP_SLICE] - slip_threshold
        )
        impact = F.relu(extra[:, p3_contract.GAIT_IMPACT_SPEED_SLICE] - impact_threshold)
        contact = -(
            0.45
            * (slip.square() * slip_event * slip_valid * slip_conf).mean(-1)
            + 0.45 * (impact.square() * onset * impact_valid * impact_conf).mean(-1)
        )
        contact = contact.clamp(min=-p3_contract.GAIT_CONTACT_REWARD_CAP)
        margin = extra[:, p3_contract.GAIT_TOUCHDOWN_Y_SLICE]
        crossing = -(
            F.relu(margin_threshold - margin).square()
            * onset
            * margin_valid
            * margin_conf
        ).mean(-1)
        crossing = crossing.clamp(min=-p3_contract.GAIT_CROSS_REWARD_CAP)
        stance = (
            F.relu(extra[:, p3_contract.GAIT_CONTINUOUS_STANCE_SLICE] - stance_threshold)
            * stance_valid
            * stance_conf
        )
        freq = (
            F.relu(frequency_threshold - aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE])
            * frequency_valid
            * frequency_conf
        )
        duty = (
            F.relu(aux[:, p2_contract.GAIT_DUTY_SLICE] - duty_threshold)
            * duty_valid
            * duty_conf
        )
        starvation = -(0.4 * stance + 0.35 * freq + 0.25 * duty).amax(-1)
        starvation = starvation.clamp(min=-p3_contract.GAIT_STARVATION_REWARD_CAP)
        total = torch.stack((contact, crossing, starvation), -1)
        total = total * sensor_valid.unsqueeze(-1) * float(scale)
        summed = total.sum(-1).clamp(min=-p3_contract.GAIT_TOTAL_REWARD_CAP)
        ratio = torch.where(total.sum(-1).abs() > 1.0e-9, summed / total.sum(-1), 1.0)
        total = total * ratio.unsqueeze(-1)
        return total[:, 0], total[:, 1], total[:, 2]

    def state_dict(self):
        encoded_values = {"|".join(map(str, key)): value for key, value in self.values.items()}
        encoded_valid = {"|".join(map(str, key)): value for key, value in self.valid.items()}
        encoded_fallback = {"|".join(map(str, key)): value for key, value in self.fallback_levels.items()}
        metadata = {
            "contract_version": self.CONTRACT_VERSION,
            "finalized": self.finalized,
            "values": encoded_values,
            "valid": encoded_valid,
            "fallback_levels": encoded_fallback,
            "fallback_share": self.fallback_share,
            "total_samples": self.total_samples,
        }
        payload = dict(metadata)
        if not self.finalized:
            payload["samples"] = {
                "|".join(map(str, key)): torch.cat(parts) if parts else torch.empty(0)
                for key, parts in self.samples.items()
            }
            payload["sample_counts"] = {
                "|".join(map(str, key)): int(value)
                for key, value in self.sample_counts.items()
            }
        payload["digest"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()
        return payload

    def load_state_dict(self, state):
        if int(state.get("contract_version", 0)) != self.CONTRACT_VERSION:
            raise ValueError("incompatible P3 gait baseline contract version")
        self.finalized = bool(state.get("finalized", False))
        def decode(values, cast):
            return {
                tuple(int(item) if index < 2 else item for index, item in enumerate(key.split("|"))): cast(value)
                for key, value in dict(values or {}).items()
            }
        self.values = decode(state.get("values"), float)
        self.valid = decode(state.get("valid"), bool)
        self.fallback_levels = decode(state.get("fallback_levels"), int)
        self.fallback_share = float(state.get("fallback_share", 0.0))
        self.total_samples = int(state.get("total_samples", 0))
        if not self.finalized:
            restored = decode(state.get("samples"), lambda value: torch.as_tensor(value).float().cpu())
            counts = decode(state.get("sample_counts"), int)
            keys = (
                (terrain, motion, name)
                for terrain in range(len(self.TERRAIN_BUCKETS))
                for motion in range(len(self.MOTION_BUCKETS))
                for name in self.NAMES
            )
            self.samples = {
                key: ([restored[key]] if key in restored and restored[key].numel() else [])
                for key in keys
            }
            self.sample_counts = {
                key: int(counts.get(key, sum(item.numel() for item in parts)))
                for key, parts in self.samples.items()
            }


def gait_bucket_indices(
    aux: torch.Tensor, extra: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose the versioned baseline grouping to monitoring without duplicating it."""
    return P3GaitBaseline._terrain_bucket(aux), P3GaitBaseline._motion_bucket(aux, extra)


class P3MirrorAuxiliary:
    """Store selected compact TBPTT sequences and calculate a bounded mirror loss."""

    def __init__(self, *, num_steps, num_envs, obs_dim, sequence_length, device, seed=3197):
        self.device = torch.device(device)
        self.sequence_length = int(sequence_length)
        self.mirrored = torch.zeros(num_steps, num_envs, obs_dim, device=self.device)
        self.selected = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=self.device)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.scale = 0.0
        self.metrics = {}
        self.gradient_multiplier = None
        self.gradient_ratio = 0.0
        self.gradient_cosine = 0.0

    def begin_rollout(self, fraction: float):
        self.mirrored.zero_()
        self.selected.zero_()
        self.scale = float(fraction)
        self.gradient_multiplier = None
        self.gradient_ratio = 0.0
        self.gradient_cosine = 0.0

    def select_block(self, step: int, num_envs: int) -> torch.Tensor:
        if self.scale <= 0.0:
            return torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if step % self.sequence_length:
            return self.selected[step - 1]
        chosen = torch.rand(num_envs, generator=self.generator) < p3_contract.MIRROR_SEQUENCE_SHARE
        return chosen.to(self.device)

    def store(self, step: int, mask: torch.Tensor, mirrored_obs: torch.Tensor):
        self.selected[step] = mask
        self.mirrored[step, mask] = mirrored_obs

    @staticmethod
    def _actor_with_detached_body(actor, features):
        value = features
        modules = list(actor)
        for module in modules[:-1]:
            if isinstance(module, torch.nn.Linear):
                value = F.linear(value, module.weight.detach(), None if module.bias is None else module.bias.detach())
            else:
                value = module(value)
        return modules[-1](value)

    def loss(self, model, original: torch.Tensor, dones: torch.Tensor):
        if self.scale <= 0.0:
            return original.new_zeros(()), {"mirror_loss": 0.0, "mirror_sequence_share": 0.0}
        starts = []
        for start in range(0, original.shape[0], self.sequence_length):
            ids = self.selected[start].nonzero(as_tuple=False).flatten()
            starts.extend((start, int(env)) for env in ids.cpu())
        if not starts:
            return original.new_zeros(()), {"mirror_loss": 0.0, "mirror_sequence_share": 0.0}
        start, env = starts[int(torch.randint(len(starts), (1,), generator=self.generator))]
        end = start + self.sequence_length
        original_seq = original[start:end, env : env + 1]
        mirrored_seq = self.mirrored[start:end, env : env + 1]
        masks = (~dones[start:end, env : env + 1]).float()
        with torch.no_grad():
            original_latent = model._encode_sequence(original_seq, hidden_states=None, masks=masks)
            original_action = model.actor(torch.cat((original_seq[..., :45], original_latent), -1))
            target = mirror_action(original_action)
        mirrored_latent = model._encode_sequence(mirrored_seq, hidden_states=None, masks=masks)
        features = torch.cat((mirrored_seq[..., :45], mirrored_latent), -1)
        prediction = self._actor_with_detached_body(model.actor, features)
        loss = F.mse_loss(prediction, target) * self.scale
        leg_errors = per_leg_action_mse(prediction, target)
        metrics = {
            "mirror_loss": float(loss.detach()),
            "mirror_sequence_share": float(self.selected.float().mean()),
        }
        for leg, value in leg_errors.items():
            metrics[f"mirror_error_{leg}"] = float(value.detach())
        return loss, metrics


class P3ActionSmoothAuxiliary:
    """Bound recurrent policy-mean rate, jerk and extreme action means."""

    def __init__(self):
        self.scale = 0.0
        self.multipliers = {"smooth": None, "range": None}
        self.gradient_ratios = {"smooth": 0.0, "range": 0.0}
        self.gradient_cosines = {"smooth": 0.0, "range": 0.0}

    def begin_rollout(self, fraction: float):
        self.scale = max(0.0, min(1.0, float(fraction)))
        self.multipliers = {"smooth": None, "range": None}
        self.gradient_ratios = {"smooth": 0.0, "range": 0.0}
        self.gradient_cosines = {"smooth": 0.0, "range": 0.0}

    @staticmethod
    def action_mean_with_detached_body(actor, features: torch.Tensor) -> torch.Tensor:
        value = features
        modules = list(actor)
        for module in modules[:-1]:
            if isinstance(module, torch.nn.Linear):
                value = F.linear(
                    value,
                    module.weight.detach(),
                    None if module.bias is None else module.bias.detach(),
                )
            else:
                value = module(value)
        return modules[-1](value)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        denominator = (mask.sum() * values.shape[-1]).clamp_min(1.0)
        return (values * mask).sum() / denominator

    def loss(self, action_mean: torch.Tensor, continuation: torch.Tensor):
        zero = action_mean.new_zeros(())
        if self.scale <= 0.0 or action_mean.ndim != 3 or action_mean.shape[0] < 2:
            return zero, zero, {
                "action_smooth_scale": self.scale,
                "action_mean_rate_loss": 0.0,
                "action_mean_jerk_loss": 0.0,
                "action_mean_range_loss": 0.0,
            }

        q_mean = action_mean * p3_contract.ACTION_TO_JOINT_SCALE
        rate = q_mean[1:] - q_mean[:-1]
        rate_threshold = action_mean.new_tensor(
            [0.20] * 8 + [0.25] * 4
        ).reshape(1, 1, -1)
        rate_excess = torch.relu(rate.abs() - rate_threshold)
        rate_terms = F.smooth_l1_loss(
            rate_excess, torch.zeros_like(rate_excess), reduction="none"
        )
        rate_mask = continuation[:-1].bool()
        rate_loss = self._masked_mean(rate_terms, rate_mask)

        jerk_loss = zero
        if rate.shape[0] >= 2:
            jerk = rate[1:] - rate[:-1]
            jerk_threshold = action_mean.new_tensor(
                [0.15] * 8 + [0.20] * 4
            ).reshape(1, 1, -1)
            jerk_excess = torch.relu(jerk.abs() - jerk_threshold)
            jerk_terms = F.smooth_l1_loss(
                jerk_excess, torch.zeros_like(jerk_excess), reduction="none"
            )
            jerk_mask = continuation[:-2].bool() & continuation[1:-1].bool()
            jerk_loss = self._masked_mean(jerk_terms, jerk_mask)

        smooth_loss = (rate_loss + jerk_loss) * self.scale
        range_excess = torch.relu(
            action_mean.abs() - p3_contract.ACTION_MEAN_SOFT_LIMIT
        )
        range_loss = F.smooth_l1_loss(
            range_excess, torch.zeros_like(range_excess), reduction="mean"
        ) * self.scale
        return smooth_loss, range_loss, {
            "action_smooth_scale": self.scale,
            "action_mean_rate_loss": float(rate_loss.detach()),
            "action_mean_jerk_loss": float(jerk_loss.detach()),
            "action_mean_range_loss": float(range_loss.detach()),
        }

    @staticmethod
    @torch.no_grad()
    def diagnostics(
        action_mean: torch.Tensor,
        sampled_action: torch.Tensor,
        dones: torch.Tensor,
        *,
        control_dt_s: float = 0.02,
    ) -> dict[str, float]:
        means = action_mean.detach().float()
        sampled = sampled_action.detach().float()
        done = dones.detach().bool().squeeze(-1)
        result = {}

        def quantiles(prefix: str, values: torch.Tensor):
            flat = values.abs().reshape(-1)
            result[f"{prefix}_abs_p50"] = float(torch.quantile(flat, 0.50))
            result[f"{prefix}_abs_p95"] = float(torch.quantile(flat, 0.95))
            result[f"{prefix}_abs_max"] = float(flat.max())

        quantiles("action_mean", means)
        quantiles("action_raw", sampled)
        quantiles("action_exec", sampled.clamp(-6.0, 6.0))
        clipped = sampled.abs() > 6.0
        result["action_clip_rate"] = float(clipped.float().mean())
        for name, values in (("hip", clipped[..., :4]), ("thigh", clipped[..., 4:8]), ("calf", clipped[..., 8:12])):
            result[f"action_clip_rate_{name}"] = float(values.float().mean())

        if means.shape[0] > 1:
            continuation = ~done[:-1]
            q_mean = means * p3_contract.ACTION_TO_JOINT_SCALE
            rate = q_mean[1:] - q_mean[:-1]
            valid_rate = rate[continuation]
            if valid_rate.numel():
                quantiles("joint_target_rate", valid_rate)
            if means.shape[0] > 2:
                valid_jerk_mask = continuation[:-1] & continuation[1:]
                jerk = rate[1:] - rate[:-1]
                valid_jerk = jerk[valid_jerk_mask]
                if valid_jerk.numel():
                    quantiles("joint_target_jerk", valid_jerk)

        valid_env = ~done[:-1].any(dim=0) if means.shape[0] > 1 else torch.zeros(
            means.shape[1], dtype=torch.bool, device=means.device
        )
        result["action_spectrum_valid_env_share"] = float(valid_env.float().mean())
        if means.shape[0] >= 32 and bool(valid_env.any()):
            series = means[:, valid_env]
            window = torch.hann_window(series.shape[0], device=series.device).reshape(-1, 1, 1)
            power = torch.fft.rfft(series * window, dim=0).abs().square()
            frequency = torch.fft.rfftfreq(
                series.shape[0], d=float(control_dt_s), device=series.device
            )
            total_mask = (frequency >= 1.0) & (frequency <= 25.0)
            high_mask = (frequency >= 15.0) & (frequency <= 25.0)
            for name, slc in (("hip", slice(0, 4)), ("thigh", slice(4, 8)), ("calf", slice(8, 12))):
                total = power[total_mask, :, slc].sum().clamp_min(1.0e-12)
                high = power[high_mask, :, slc].sum()
                result[f"action_15_25hz_power_ratio_{name}"] = float(high / total)
        return result
