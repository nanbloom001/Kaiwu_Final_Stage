#!/usr/bin/env python3
"""Training-only P3 gait baselines, event rewards and mirror consistency."""

from __future__ import annotations

import hashlib
import json
import torch
import torch.nn.functional as F

from agent_ppo.feature import p2_contract, p3_contract


LEG_SWAP = torch.tensor([1, 0, 3, 2], dtype=torch.long)
JOINT_SWAP = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=torch.long)
JOINT_SIGN = torch.tensor([-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1], dtype=torch.float32)


def validate_joint_order(names) -> bool:
    names = [str(name).lower() for name in names]
    if len(names) != 12:
        return False
    expected = ("fl", "fr", "rl", "rr")
    axes = ("hip", "thigh", "calf")
    return all(
        expected[index // 3] in name and axes[index % 3] in name
        for index, name in enumerate(names)
    )


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    permutation = JOINT_SWAP.to(action.device)
    signs = JOINT_SIGN.to(action)
    return action.index_select(-1, permutation) * signs


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

    NAMES = ("slip", "impact", "margin", "stance", "frequency", "duty")
    TERRAIN_BUCKETS = ("slope", "slope_inv", "stairs", "stairs_inv")
    MOTION_BUCKETS = ("low_speed", "forward", "reverse", "turn_lateral")
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
        result[(column >= 0) & (column < 4)] = 0
        result[(column >= 4) & (column < 8)] = 1
        result[(column >= 8) & (column < 14)] = 2
        result[(column >= 14) & (column < 20)] = 3
        return result

    @staticmethod
    def _motion_bucket(aux: torch.Tensor) -> torch.Tensor:
        command = aux[:, 3:6]
        vx, vy, wz = command.unbind(-1)
        result = torch.full_like(vx, 1, dtype=torch.long)
        reverse = vx < -0.05
        turn_lateral = (~reverse) & ((vy.abs() > 0.05) | (wz.abs() > 0.10))
        low_speed = (~reverse) & (~turn_lateral) & (vx.abs() < 0.20)
        result[reverse] = 2
        result[turn_lateral] = 3
        result[low_speed] = 0
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
        motion = self._motion_bucket(aux)
        onset = extra[:, p3_contract.GAIT_CONTACT_ONSET_SLICE] > 0.5
        for terrain_index in range(len(self.TERRAIN_BUCKETS)):
            for motion_index in range(len(self.MOTION_BUCKETS)):
                selected = healthy & (terrain == terrain_index) & (motion == motion_index)
                if not bool(selected.any()):
                    continue
                fields = {
                    "impact": extra[selected, p3_contract.GAIT_IMPACT_SPEED_SLICE][onset[selected]],
                    "margin": extra[selected, p3_contract.GAIT_TOUCHDOWN_Y_SLICE][onset[selected]],
                }
                if collect_continuous:
                    fields.update(
                        slip=aux[selected, p2_contract.GAIT_SLIP_SPEED_SLICE],
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

    def _thresholds(self, aux: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        terrain = self._terrain_bucket(aux)
        motion = self._motion_bucket(aux)
        threshold = torch.zeros(aux.shape[0], device=aux.device, dtype=aux.dtype)
        valid = torch.zeros(aux.shape[0], device=aux.device, dtype=torch.bool)
        for terrain_index in range(len(self.TERRAIN_BUCKETS)):
            for motion_index in range(len(self.MOTION_BUCKETS)):
                selected = (terrain == terrain_index) & (motion == motion_index)
                key = (terrain_index, motion_index, name)
                if bool(selected.any()) and self.valid.get(key, False):
                    threshold[selected] = self.values[key]
                    valid[selected] = True
        return threshold.unsqueeze(-1), valid.unsqueeze(-1)

    def rewards(self, aux: torch.Tensor, extra: torch.Tensor, scale: float):
        zeros = torch.zeros(aux.shape[0], device=aux.device)
        if not self.finalized or scale <= 0.0:
            return zeros, zeros, zeros
        sensor_valid = (aux[:, p2_contract.GAIT_VALID_INDEX] > 0.5).float()
        onset = (extra[:, p3_contract.GAIT_CONTACT_ONSET_SLICE] > 0.5).float()
        slip_threshold, slip_valid = self._thresholds(aux, "slip")
        impact_threshold, impact_valid = self._thresholds(aux, "impact")
        margin_threshold, margin_valid = self._thresholds(aux, "margin")
        stance_threshold, stance_valid = self._thresholds(aux, "stance")
        frequency_threshold, frequency_valid = self._thresholds(aux, "frequency")
        duty_threshold, duty_valid = self._thresholds(aux, "duty")
        slip = F.relu(aux[:, p2_contract.GAIT_SLIP_SPEED_SLICE] - slip_threshold)
        impact = F.relu(extra[:, p3_contract.GAIT_IMPACT_SPEED_SLICE] - impact_threshold)
        contact = -(
            0.45 * (slip.square() * slip_valid).mean(-1)
            + 0.45 * (impact.square() * onset * impact_valid).mean(-1)
        )
        contact = contact.clamp(min=-p3_contract.GAIT_CONTACT_REWARD_CAP)
        margin = extra[:, p3_contract.GAIT_TOUCHDOWN_Y_SLICE]
        crossing = -(
            F.relu(margin_threshold - margin).square()
            * onset
            * margin_valid
        ).mean(-1)
        crossing = crossing.clamp(min=-p3_contract.GAIT_CROSS_REWARD_CAP)
        stance = F.relu(extra[:, p3_contract.GAIT_CONTINUOUS_STANCE_SLICE] - stance_threshold) * stance_valid
        freq = F.relu(frequency_threshold - aux[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE]) * frequency_valid
        duty = F.relu(aux[:, p2_contract.GAIT_DUTY_SLICE] - duty_threshold) * duty_valid
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
        return loss, {
            "mirror_loss": float(loss.detach()),
            "mirror_sequence_share": float(self.selected.float().mean()),
            "mirror_error_fl": float((prediction[..., 0:3] - target[..., 0:3]).square().mean().detach()),
            "mirror_error_fr": float((prediction[..., 3:6] - target[..., 3:6]).square().mean().detach()),
            "mirror_error_rl": float((prediction[..., 6:9] - target[..., 6:9]).square().mean().detach()),
            "mirror_error_rr": float((prediction[..., 9:12] - target[..., 9:12]).square().mean().detach()),
        }
