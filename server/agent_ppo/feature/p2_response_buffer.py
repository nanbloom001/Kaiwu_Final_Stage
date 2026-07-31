#!/usr/bin/env python3
"""P2 response labels with tolerant command holds and parent replay."""

from __future__ import annotations

from collections import deque

import torch

from agent_ppo.feature import p15_contract, p2_contract
from agent_ppo.feature.response_aux_buffer import ResponseAuxBuffer, ResponseBatch


def split_p2_transport(wire: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(wire) or wire.ndim != 2:
        raise ValueError("P2 privileged transport must be rank-2")
    if wire.shape[1] != p2_contract.PRIVILEGED_WIRE_DIM:
        raise ValueError(
            f"P2 privileged transport must have {p2_contract.PRIVILEGED_WIRE_DIM} columns, "
            f"got {wire.shape[1]}"
        )
    return wire[:, : p2_contract.CRITIC_OBS_DIM], wire[:, p2_contract.CRITIC_OBS_DIM :]


def patch_owned_commands(
    aux: torch.Tensor,
    active_target_cmd3: torch.Tensor,
    exec_cmd3: torch.Tensor,
    command_epoch: int | torch.Tensor,
) -> torch.Tensor:
    result = aux.clone()
    result[:, 0:3] = active_target_cmd3.to(result)
    result[:, 3:6] = exec_cmd3.to(result)
    epoch = torch.as_tensor(command_epoch, device=result.device, dtype=result.dtype)
    if epoch.ndim == 0:
        result[:, 26] = epoch
    elif epoch.numel() == result.shape[0]:
        result[:, 26] = epoch.reshape(-1)
    else:
        raise ValueError(
            "command_epoch must be scalar or one value per environment, "
            f"got {tuple(epoch.shape)} for {result.shape[0]} environments"
        )
    return result


class P2ResponseAuxBuffer(ResponseAuxBuffer):
    """Keep completed Track and parent records in separate replay pools."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent_records: deque[dict] = deque(maxlen=self.capacity_steps)
        self._history_episode_start: deque[torch.Tensor] = deque(maxlen=51)
        self._history_current_segment: deque[torch.Tensor] = deque(maxlen=51)
        self._next_episode_start = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.resized_completed_records = 0

    def clear_unfinished_history(self) -> None:
        super().clear_unfinished_history()
        self._history_episode_start.clear()
        self._history_current_segment.clear()
        self._next_episode_start.fill_(True)

    def _capability_for_record(self, record) -> torch.Tensor:
        values = (
            p15_contract.CAPABILITY_PROFILE15
            if record.get("record_origin") == "parent"
            else p2_contract.RESPONSE_CAPABILITY_PROFILE15
        )
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    @staticmethod
    def _stable_target(start: torch.Tensor, window: list[torch.Tensor]) -> torch.Tensor:
        tolerance = torch.tensor(
            p2_contract.TARGET_STABILITY_TOLERANCE,
            device=start.device,
            dtype=start.dtype,
        )
        maximum_delta = torch.stack(
            [(sample[:, 0:3] - start[:, 0:3]).abs() for sample in window], dim=0
        ).amax(dim=0)
        return (maximum_delta <= tolerance).all(dim=-1)

    def append(
        self,
        aux: torch.Tensor,
        dones: torch.Tensor,
        *,
        current_segment: torch.Tensor | None = None,
    ) -> None:
        self.append_calls += 1
        aux = aux.detach().to(self.device, dtype=torch.float32)
        done = dones.detach().to(self.device).reshape(-1).bool()
        if aux.shape != (self.num_envs, p15_contract.RESPONSE_AUX_DIM):
            raise ValueError(f"unexpected P2 response aux shape {tuple(aux.shape)}")
        self._history_aux.append(aux.clone())
        self._history_done.append(done.clone())
        self._history_episode_start.append(self._next_episode_start.clone())
        segment = (
            current_segment.detach().to(self.device).reshape(-1).float()
            if torch.is_tensor(current_segment)
            and current_segment.numel() == self.num_envs
            else torch.full((self.num_envs,), -1.0, device=self.device)
        )
        self._history_current_segment.append(segment.clone())
        self._next_episode_start = done.clone()
        self.max_history_length = max(self.max_history_length, len(self._history_aux))
        if len(self._history_aux) < 51:
            return
        history = list(self._history_aux)
        done_history = list(self._history_done)
        episode_start_history = list(self._history_episode_start)
        current_segment_history = list(self._history_current_segment)
        current = history[0]
        masks, velocity = [], []
        for index, step in enumerate(self.HORIZON_STEPS):
            no_boundary = ~torch.stack(done_history[:step], dim=0).any(dim=0)
            stable = self._stable_target(current, history[1 : step + 1])
            # One nav hold is exactly 10 low-level frames. The short head uses
            # that complete hold; longer heads additionally require tolerance.
            valid = no_boundary & (current[:, 9] > 0.5)
            if index > 0:
                valid &= stable
            masks.append(valid)
            velocity.append(history[step][:, 12:15])
        horizon_mask = torch.stack(masks, dim=-1)
        self.valid_horizon_counts += horizon_mask.sum(dim=0).detach().cpu()
        pose = self._body_pose_delta(current[:, 15:18], history[50][:, 15:18])
        pose_mask = horizon_mask[:, 2]
        commanded = torch.linalg.vector_norm(current[:, 3:6], dim=-1) > 0.10
        response_speed = torch.linalg.vector_norm(history[50][:, 12:15], dim=-1)
        displacement = torch.linalg.vector_norm(pose[:, :2], dim=-1)
        stuck = commanded & (response_speed < 0.08) & (displacement < 0.08)
        self._records.append(
            {
                "aux": current.clone(),
                "velocity": torch.stack(velocity, dim=1),
                "pose": pose,
                "stuck": stuck.float().unsqueeze(-1),
                "horizon_mask": horizon_mask,
                "pose_mask": pose_mask.unsqueeze(-1),
                "episode_start": episode_start_history[0].clone(),
                "low_level_digest": self.low_level_digest,
                "low_level_iteration": self.low_level_iteration,
                "command_phase": current[:, 27].clone(),
                "terrain_family": current[:, 28].clone(),
                "terrain_level": current[:, 29].clone(),
                "current_segment": current_segment_history[0].clone(),
                "record_origin": "track",
            }
        )

    def load_parent_completed_records(self, state: dict[str, object]) -> int:
        temporary = P2ResponseAuxBuffer(
            self.num_envs,
            self.device,
            capacity_steps=self.capacity_steps,
            sequence_length=self.sequence_length,
            burn_in_steps=self.burn_in_steps,
        )
        temporary.load_checkpoint_state(state)
        self._parent_records.clear()
        for record in temporary._records:
            copied = dict(record)
            copied["record_origin"] = "parent"
            copied["current_segment"] = torch.full(
                (self.num_envs,), -1.0, device=self.device
            )
            self._parent_records.append(copied)
        return len(self._parent_records)

    @staticmethod
    def _concat_batches(track: ResponseBatch, parent: ResponseBatch) -> ResponseBatch:
        tensor_fields = (
            "burn_in_observations", "burn_in_reset_mask", "observations",
            "reset_mask", "velocity_labels", "pose_labels", "stuck_labels",
            "horizon_mask", "pose_mask",
        )
        values = {
            name: torch.cat((getattr(track, name), getattr(parent, name)), dim=1)
            for name in tensor_fields
        }
        metadata = {
            "replay_origin": "mixed_track_parent",
            "track_batch_envs": int(track.observations.shape[1]),
            "parent_batch_envs": int(parent.observations.shape[1]),
        }
        for name in (
            "command_phase",
            "terrain_family",
            "terrain_level",
            "current_segment",
        ):
            track_value = track.metadata.get(name)
            parent_value = parent.metadata.get(name)
            if torch.is_tensor(track_value) and torch.is_tensor(parent_value):
                metadata[name] = torch.cat((track_value, parent_value), dim=1)
        values["metadata"] = metadata
        return ResponseBatch(**values)

    def sample(self, *, batch_envs: int, generator=None) -> ResponseBatch | None:
        requested = max(1, int(batch_envs))
        parent_count = int(round(requested * p2_contract.PARENT_RESPONSE_REPLAY_RATIO))
        track_count = requested - parent_count
        track_records = self._records
        track_batch = super().sample(
            batch_envs=max(1, track_count), generator=generator
        )
        if not self._parent_records or parent_count <= 0:
            if track_batch is not None:
                track_batch.metadata.update(
                    {
                        "replay_origin": "track_only",
                        "track_batch_envs": int(track_batch.observations.shape[1]),
                        "parent_batch_envs": 0,
                    }
                )
            return track_batch
        self._records = self._parent_records
        try:
            parent_batch = super().sample(
                batch_envs=max(1, parent_count), generator=generator
            )
        finally:
            self._records = track_records
        if track_batch is None:
            if parent_batch is not None:
                parent_batch.metadata.update(
                    {
                        "replay_origin": "parent_only_track_unready",
                        "track_batch_envs": 0,
                        "parent_batch_envs": int(parent_batch.observations.shape[1]),
                    }
                )
            return parent_batch
        if parent_batch is None:
            track_batch.metadata.update(
                {
                    "replay_origin": "track_only_parent_unready",
                    "track_batch_envs": int(track_batch.observations.shape[1]),
                    "parent_batch_envs": 0,
                }
            )
            return track_batch
        return self._concat_batches(track_batch, parent_batch)

    def checkpoint_state(self) -> dict[str, object]:
        state = super().checkpoint_state()
        state["num_envs"] = self.num_envs
        state["parent_record_count"] = len(self._parent_records)
        state["parent_records"] = [
            self._cpu_record(record) for record in self._parent_records
        ]
        return state

    def _resize_completed_records(self, state: dict[str, object]) -> dict[str, object]:
        """Allow the documented 128 -> 96 -> 80 restart path."""
        result = dict(state)
        resized = 0
        for pool_name in ("records", "parent_records"):
            records = state.get(pool_name, [])
            if not isinstance(records, list):
                continue
            normalized = []
            for record in records:
                if not isinstance(record, dict):
                    normalized.append(record)
                    continue
                aux = record.get("aux")
                if not torch.is_tensor(aux) or aux.ndim < 1:
                    normalized.append(record)
                    continue
                saved_envs = int(aux.shape[0])
                if saved_envs < self.num_envs:
                    normalized.append(record)
                    continue
                if saved_envs == self.num_envs:
                    normalized.append(record)
                    continue
                normalized.append(
                    {
                        key: (
                            value[: self.num_envs].clone()
                            if torch.is_tensor(value)
                            and value.ndim >= 1
                            and value.shape[0] == saved_envs
                            else value
                        )
                        for key, value in record.items()
                    }
                )
                resized += 1
            result[pool_name] = normalized
        self.resized_completed_records = resized
        result["num_envs"] = self.num_envs
        return result

    def load_checkpoint_state(self, state: dict[str, object]) -> str:
        if isinstance(state, dict):
            state = self._resize_completed_records(state)
        mode = super().load_checkpoint_state(state)
        self._history_episode_start.clear()
        self._history_current_segment.clear()
        self._next_episode_start.fill_(True)
        for record in self._records:
            if "current_segment" not in record:
                record["current_segment"] = torch.full(
                    (self.num_envs,), -1.0, device=self.device
                )
        self._parent_records.clear()
        if isinstance(state, dict):
            for record in state.get("parent_records", []):
                if not isinstance(record, dict):
                    continue
                aux = record.get("aux")
                if not torch.is_tensor(aux) or tuple(aux.shape) != (
                    self.num_envs,
                    p15_contract.RESPONSE_AUX_DIM,
                ):
                    continue
                restored = {
                    key: value.to(self.device) if torch.is_tensor(value) else value
                    for key, value in record.items()
                }
                restored["current_segment"] = torch.full(
                    (self.num_envs,), -1.0, device=self.device
                )
                self._parent_records.append(restored)
        return mode
