#!/usr/bin/env python3
"""Independent future-label buffer for response-adapter supervision."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from agent_ppo.feature import p15_contract


@dataclass
class ResponseBatch:
    burn_in_observations: torch.Tensor
    burn_in_reset_mask: torch.Tensor
    observations: torch.Tensor
    reset_mask: torch.Tensor
    velocity_labels: torch.Tensor
    pose_labels: torch.Tensor
    stuck_labels: torch.Tensor
    horizon_mask: torch.Tensor
    pose_mask: torch.Tensor
    metadata: dict[str, object]


def split_privileged_transport(wire: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(wire) or wire.ndim != 2:
        raise ValueError("privileged transport must be a rank-2 tensor")
    if wire.shape[1] != p15_contract.PRIVILEGED_WIRE_DIM:
        raise ValueError(
            f"P1.5 privileged transport must have {p15_contract.PRIVILEGED_WIRE_DIM} "
            f"columns, got {wire.shape[1]}"
        )
    return wire[:, : p15_contract.CRITIC_OBS_DIM], wire[:, p15_contract.CRITIC_OBS_DIM :]


def build_response_observation(aux: torch.Tensor, capability: torch.Tensor) -> torch.Tensor:
    if aux.ndim != 2 or aux.shape[1] != p15_contract.RESPONSE_AUX_DIM:
        raise ValueError(f"response aux must be [N,30], got {tuple(aux.shape)}")
    if capability.ndim == 1:
        capability = capability.unsqueeze(0).expand(aux.shape[0], -1)
    if capability.shape != (aux.shape[0], 15):
        raise ValueError("capability profile must be [15] or [N,15]")
    result = torch.cat(
        (
            aux[:, 0:3],
            aux[:, 3:6],
            aux[:, 6:9],
            aux[:, 9:10],
            aux[:, 10:11],
            aux[:, 18:21],
            aux[:, 21:24],
            capability.to(device=aux.device, dtype=aux.dtype),
        ),
        dim=-1,
    )
    if result.shape[1] != p15_contract.RESPONSE_OBSERVATION_DIM:
        raise AssertionError("response observation layout drift")
    return result


class ResponseAuxBuffer:
    """Build future response labels without adding aux fields to PPO storage."""

    HORIZON_STEPS = (10, 30, 50)

    def __init__(
        self,
        num_envs: int,
        device,
        *,
        capacity_steps: int = 4096,
        sequence_length: int = 16,
        burn_in_steps: int = 8,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.capacity_steps = max(64, int(capacity_steps))
        self.sequence_length = max(1, int(sequence_length))
        self.burn_in_steps = max(0, int(burn_in_steps))
        self._history_aux: deque[torch.Tensor] = deque(maxlen=51)
        self._history_done: deque[torch.Tensor] = deque(maxlen=51)
        self._records: deque[dict[str, torch.Tensor | str | int]] = deque(
            maxlen=self.capacity_steps
        )
        self.low_level_digest = "unknown"
        self.low_level_iteration = 0
        self._active_low_level_version: tuple[str, int] | None = None
        self.total_sequences = 0
        self.append_calls = 0
        self.version_reset_count = 0
        self.max_history_length = 0
        self.valid_horizon_counts = torch.zeros(3, dtype=torch.long)

    def set_low_level_version(self, digest: str, iteration: int) -> None:
        version = (str(digest), int(iteration))
        if (
            self._active_low_level_version is not None
            and version != self._active_low_level_version
        ):
            self.version_reset_count += 1
            self.clear_unfinished_history()
        self.low_level_digest, self.low_level_iteration = version
        self._active_low_level_version = version

    def clear_unfinished_history(self) -> None:
        """Drop labels that would cross a policy update or process boundary."""
        self._history_aux.clear()
        self._history_done.clear()

    @staticmethod
    def _body_pose_delta(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        dx_w = end[:, 0] - start[:, 0]
        dy_w = end[:, 1] - start[:, 1]
        yaw = start[:, 2]
        dx_b = torch.cos(yaw) * dx_w + torch.sin(yaw) * dy_w
        dy_b = -torch.sin(yaw) * dx_w + torch.cos(yaw) * dy_w
        dyaw = torch.atan2(
            torch.sin(end[:, 2] - start[:, 2]),
            torch.cos(end[:, 2] - start[:, 2]),
        )
        return torch.stack((dx_b, dy_b, dyaw), dim=-1)

    def append(self, aux: torch.Tensor, dones: torch.Tensor) -> None:
        self.append_calls += 1
        aux = aux.detach().to(self.device, dtype=torch.float32)
        done = dones.detach().to(self.device).reshape(-1).bool()
        if aux.shape != (self.num_envs, p15_contract.RESPONSE_AUX_DIM):
            raise ValueError(f"unexpected response aux shape {tuple(aux.shape)}")
        if done.shape != (self.num_envs,):
            raise ValueError(f"unexpected done shape {tuple(done.shape)}")
        self._history_aux.append(aux.clone())
        self._history_done.append(done.clone())
        self.max_history_length = max(self.max_history_length, len(self._history_aux))
        if len(self._history_aux) < 51:
            return
        history = list(self._history_aux)
        done_history = list(self._history_done)
        current = history[0]
        epoch = current[:, 26]
        horizon_mask = []
        velocity = []
        for step in self.HORIZON_STEPS:
            # ``done_history[0]`` marks that the current observation is the
            # first frame of a reset episode.  Only later boundaries invalidate
            # a future label starting at the current observation.
            no_done = ~torch.stack(done_history[1 : step + 1], dim=0).any(dim=0)
            same_target = history[step][:, 26] == epoch
            valid = no_done & same_target & (current[:, 9] > 0.5)
            horizon_mask.append(valid)
            velocity.append(history[step][:, 12:15])
        horizon_mask_tensor = torch.stack(horizon_mask, dim=-1)
        self.valid_horizon_counts += horizon_mask_tensor.sum(dim=0).detach().cpu()
        velocity_tensor = torch.stack(velocity, dim=1)
        pose_mask = horizon_mask_tensor[:, 2]
        pose_delta = self._body_pose_delta(current[:, 15:18], history[50][:, 15:18])
        commanded = torch.linalg.vector_norm(current[:, 3:6], dim=-1) > 0.10
        response_speed = torch.linalg.vector_norm(history[50][:, 12:15], dim=-1)
        displacement = torch.linalg.vector_norm(pose_delta[:, :2], dim=-1)
        stuck = commanded & (response_speed < 0.08) & (displacement < 0.08)
        self._records.append(
            {
                "aux": current.clone(),
                "velocity": velocity_tensor.clone(),
                "pose": pose_delta.clone(),
                "stuck": stuck.float().unsqueeze(-1),
                "horizon_mask": horizon_mask_tensor.clone(),
                "pose_mask": pose_mask.clone().unsqueeze(-1),
                "episode_start": done_history[0].clone(),
                "low_level_digest": self.low_level_digest,
                "low_level_iteration": self.low_level_iteration,
                "command_phase": current[:, 27].clone(),
                "terrain_family": current[:, 28].clone(),
                "terrain_level": current[:, 29].clone(),
            }
        )

    @property
    def ready(self) -> bool:
        return bool(self._candidate_starts())

    def _candidate_starts(self) -> list[int]:
        window = self.burn_in_steps + self.sequence_length
        if len(self._records) < window:
            return []
        records = list(self._records)
        candidates = []
        for start in range(len(records) - window + 1):
            selected = records[start : start + window]
            versions = {
                (str(record["low_level_digest"]), int(record["low_level_iteration"]))
                for record in selected
            }
            if len(versions) == 1:
                candidates.append(start)
        return candidates

    def sample(
        self,
        *,
        batch_envs: int,
        generator: torch.Generator | None = None,
    ) -> ResponseBatch | None:
        if not self.ready:
            return None
        candidates = self._candidate_starts()
        if not candidates:
            return None
        candidate_index = int(
            torch.randint(
                0, len(candidates), (1,), generator=generator, device=self.device
            ).item()
        )
        start = candidates[candidate_index]
        env_count = min(max(1, int(batch_envs)), self.num_envs)
        env_ids = torch.randperm(
            self.num_envs, generator=generator, device=self.device
        )[:env_count]
        window = self.burn_in_steps + self.sequence_length
        records = list(self._records)[start : start + window]
        burn_in_records = records[: self.burn_in_steps]
        train_records = records[self.burn_in_steps :]
        def _observations(selected):
            if not selected:
                return torch.empty(
                    0,
                    env_count,
                    p15_contract.RESPONSE_OBSERVATION_DIM,
                    device=self.device,
                )
            return torch.stack(
                [
                    build_response_observation(
                        record["aux"], self._capability_for_record(record)
                    )[env_ids]
                    for record in selected
                ],
                dim=0,
            )

        observations = _observations(train_records)
        burn_in_observations = _observations(burn_in_records)
        reset_mask = torch.stack(
            [record["episode_start"][env_ids] for record in train_records], dim=0
        )

        burn_in_reset_mask = (
            torch.stack(
                [record["episode_start"][env_ids] for record in burn_in_records], dim=0
            )
            if burn_in_records
            else torch.empty(0, env_count, device=self.device, dtype=torch.bool)
        )
        self.total_sequences += env_count
        metadata = {
            "low_level_digest": tuple(
                str(record["low_level_digest"]) for record in train_records
            ),
            "low_level_iteration": torch.tensor(
                [int(record["low_level_iteration"]) for record in train_records],
                device=self.device,
                dtype=torch.long,
            ),
            "command_phase": torch.stack(
                [record["command_phase"][env_ids] for record in train_records], dim=0
            ),
            "terrain_family": torch.stack(
                [record["terrain_family"][env_ids] for record in train_records], dim=0
            ),
            "terrain_level": torch.stack(
                [record["terrain_level"][env_ids] for record in train_records], dim=0
            ),
        }
        # P2 Track records may carry a live spatial segment. P1.5 and legacy
        # parent records intentionally use -1 so they cannot be mislabeled as
        # slope/stairs/maze samples.
        if any("current_segment" in record for record in train_records):
            metadata["current_segment"] = torch.stack(
                [
                    (
                        record["current_segment"]
                        if torch.is_tensor(record.get("current_segment"))
                        else torch.full(
                            (self.num_envs,),
                            -1.0,
                            device=self.device,
                        )
                    )[env_ids]
                    for record in train_records
                ],
                dim=0,
            )
        return ResponseBatch(
            burn_in_observations=burn_in_observations,
            burn_in_reset_mask=burn_in_reset_mask,
            observations=observations,
            reset_mask=reset_mask,
            velocity_labels=torch.stack(
                [record["velocity"][env_ids] for record in train_records], dim=0
            ),
            pose_labels=torch.stack(
                [record["pose"][env_ids] for record in train_records], dim=0
            ),
            stuck_labels=torch.stack(
                [record["stuck"][env_ids] for record in train_records], dim=0
            ),
            horizon_mask=torch.stack(
                [record["horizon_mask"][env_ids] for record in train_records], dim=0
            ),
            pose_mask=torch.stack(
                [record["pose_mask"][env_ids] for record in train_records], dim=0
            ),
            metadata=metadata,
        )

    def _capability_for_record(self, _record) -> torch.Tensor:
        return torch.tensor(
            p15_contract.CAPABILITY_PROFILE15,
            device=self.device,
            dtype=torch.float32,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity_steps": self.capacity_steps,
            "sequence_length": self.sequence_length,
            "burn_in_steps": self.burn_in_steps,
            "horizon_steps": self.HORIZON_STEPS,
            "record_steps": len(self._records),
            "total_response_sequences": self.total_sequences,
            "low_level_digest": self.low_level_digest,
            "low_level_iteration": self.low_level_iteration,
            "append_calls": self.append_calls,
            "current_history_length": len(self._history_aux),
            "max_history_length": self.max_history_length,
            "version_reset_count": self.version_reset_count,
            "valid_horizon_counts": self.valid_horizon_counts.tolist(),
            "active_low_level_version": (
                None
                if self._active_low_level_version is None
                else {
                    "digest_prefix": self._active_low_level_version[0][:12],
                    "iteration": self._active_low_level_version[1],
                }
            ),
        }

    @staticmethod
    def _cpu_record(record):
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in record.items()
        }

    def checkpoint_state(self) -> dict[str, object]:
        """Persist completed labels; unfinished future history never crosses jobs."""
        record_count = min(
            len(self._records), max(self.sequence_length + self.burn_in_steps, 32)
        )
        records = list(self._records)[-record_count:] if record_count else []
        return {
            **self.state_dict(),
            "resume_policy": "completed_records_only_v2",
            "records": [self._cpu_record(record) for record in records],
        }

    def load_checkpoint_state(self, state: dict[str, object]) -> str:
        if not isinstance(state, dict):
            self.clear_unfinished_history()
            self._records.clear()
            return "buffer_reset_missing"
        self._history_aux.clear()
        self._history_done.clear()
        self._records.clear()
        saved_horizons = state.get("horizon_steps", ())
        contract_compatible = (
            int(state.get("sequence_length", -1)) == self.sequence_length
            and int(state.get("burn_in_steps", -1)) == self.burn_in_steps
            and isinstance(saved_horizons, (tuple, list))
            and tuple(saved_horizons) == self.HORIZON_STEPS
        )
        saved_records = state.get("records", [])
        if not isinstance(saved_records, list):
            saved_records = []
        for record in saved_records if contract_compatible else []:
            if not isinstance(record, dict):
                continue
            aux = record.get("aux")
            episode_start = record.get("episode_start")
            if (
                not torch.is_tensor(aux)
                or tuple(aux.shape)
                != (self.num_envs, p15_contract.RESPONSE_AUX_DIM)
                or not torch.is_tensor(episode_start)
                or episode_start.numel() != self.num_envs
            ):
                continue
            restored = {
                key: (
                    value.to(self.device)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in record.items()
            }
            self._records.append(restored)
        self.total_sequences = int(state.get("total_response_sequences", 0))
        self.append_calls = int(state.get("append_calls", 0))
        self.max_history_length = int(state.get("max_history_length", 0))
        self.version_reset_count = int(state.get("version_reset_count", 0))
        saved_valid_counts = state.get("valid_horizon_counts", [0, 0, 0])
        if not isinstance(saved_valid_counts, (tuple, list)) or len(saved_valid_counts) != 3:
            saved_valid_counts = [0, 0, 0]
        self.valid_horizon_counts = torch.tensor(saved_valid_counts, dtype=torch.long)
        self.low_level_digest = str(state.get("low_level_digest", "unknown"))
        self.low_level_iteration = int(state.get("low_level_iteration", 0))
        self._active_low_level_version = (
            self.low_level_digest,
            self.low_level_iteration,
        )
        if not contract_compatible:
            return "buffer_reset_contract_mismatch"
        if saved_records and not self._records:
            return "buffer_reset_shape_mismatch"
        return "completed_records_restored_history_reset"
