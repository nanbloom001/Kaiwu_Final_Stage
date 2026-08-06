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
        self._history_push_epoch: deque[torch.Tensor] = deque(maxlen=51)
        self._history_seconds_since_push: deque[torch.Tensor] = deque(maxlen=51)
        self._next_episode_start = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.resized_completed_records = 0
        self.replay_policy = "track_parent_75_25"
        self.p3_replay_ratios = (0.50, 0.25, 0.25)
        self.current_record_contract: dict[str, object] | None = None
        self.compatibility_rejections: dict[str, int] = {}
        self.legacy_parent_records_migrated = 0
        self.legacy_parent_migration_rejections: dict[str, int] = {}
        self.push_horizon_rejections = torch.zeros(3, dtype=torch.long)
        self.push_pose_rejections = 0

    def set_record_contract(self, contract: dict[str, object] | None) -> None:
        self.current_record_contract = None if contract is None else dict(contract)

    def enable_p4_compatible_replay(self) -> None:
        self.replay_policy = "p4_compatible_50_25_25"

    def set_p3_replay_ratios(
        self, latest: float, recent: float, parent: float
    ) -> None:
        values = tuple(float(value) for value in (latest, recent, parent))
        if any(value < 0.0 for value in values) or abs(sum(values) - 1.0) > 1.0e-6:
            raise ValueError("P3 replay ratios must be nonnegative and sum to one")
        self.p3_replay_ratios = values

    def clear_unfinished_history(self) -> None:
        super().clear_unfinished_history()
        self._history_episode_start.clear()
        self._history_current_segment.clear()
        self._history_push_epoch.clear()
        self._history_seconds_since_push.clear()
        self._next_episode_start.fill_(True)

    def _capability_for_record(self, record) -> torch.Tensor:
        contract = record.get("record_contract")
        if isinstance(contract, dict):
            profile = contract.get("response_capability_profile15")
            if (
                isinstance(profile, (list, tuple))
                and len(profile) == len(p15_contract.CAPABILITY_PROFILE15)
            ):
                return torch.tensor(
                    profile, device=self.device, dtype=torch.float32
                )
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
        push_epoch: torch.Tensor | None = None,
        seconds_since_push: torch.Tensor | None = None,
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
        epoch = (
            push_epoch.detach().to(self.device).reshape(-1).long()
            if torch.is_tensor(push_epoch) and push_epoch.numel() == self.num_envs
            else torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        )
        since = (
            seconds_since_push.detach().to(self.device).reshape(-1).float()
            if torch.is_tensor(seconds_since_push)
            and seconds_since_push.numel() == self.num_envs
            else torch.full((self.num_envs,), 1.0e6, device=self.device)
        )
        self._history_push_epoch.append(epoch.clone())
        self._history_seconds_since_push.append(since.clone())
        self._next_episode_start = done.clone()
        self.max_history_length = max(self.max_history_length, len(self._history_aux))
        if len(self._history_aux) < 51:
            return
        history = list(self._history_aux)
        done_history = list(self._history_done)
        episode_start_history = list(self._history_episode_start)
        current_segment_history = list(self._history_current_segment)
        push_epoch_history = list(self._history_push_epoch)
        seconds_since_push_history = list(self._history_seconds_since_push)
        current = history[0]
        masks, velocity = [], []
        for index, step in enumerate(self.HORIZON_STEPS):
            no_boundary = ~torch.stack(done_history[:step], dim=0).any(dim=0)
            stable = self._stable_target(current, history[1 : step + 1])
            # One nav hold is exactly 10 low-level frames. The short head uses
            # that complete hold; longer heads additionally require tolerance.
            valid = no_boundary & (current[:, 9] > 0.5)
            horizon_s = float(step) * p2_contract.CONTROL_DT_S
            push_valid = push_epoch_history[0] == push_epoch_history[step]
            push_valid &= seconds_since_push_history[0] >= horizon_s
            self.push_horizon_rejections[index] += (
                valid & ~push_valid
            ).sum().detach().cpu()
            valid &= push_valid
            if index > 0:
                valid &= stable
            masks.append(valid)
            velocity.append(history[step][:, 12:15])
        horizon_mask = torch.stack(masks, dim=-1)
        self.valid_horizon_counts += horizon_mask.sum(dim=0).detach().cpu()
        pose = self._body_pose_delta(current[:, 15:18], history[50][:, 15:18])
        pose_mask = horizon_mask[:, 2]
        pose_push_valid = push_epoch_history[0] == push_epoch_history[50]
        pose_push_valid &= seconds_since_push_history[0] >= 1.0
        self.push_pose_rejections += int((pose_mask & ~pose_push_valid).sum())
        pose_mask &= pose_push_valid
        translation_commanded = (
            torch.linalg.vector_norm(current[:, 3:5], dim=-1) > 0.10
        )
        translation_response = torch.linalg.vector_norm(
            history[50][:, 12:14], dim=-1
        )
        translation_displacement = torch.linalg.vector_norm(pose[:, :2], dim=-1)
        translation_stuck = (
            translation_commanded
            & (translation_response < 0.08)
            & (translation_displacement < 0.08)
        )
        yaw_commanded = current[:, 5].abs() > 0.10
        yaw_stuck = (
            yaw_commanded
            & (history[50][:, 14].abs() < 0.08)
            & (pose[:, 2].abs() < 0.08)
        )
        # Keep the one-bit wire layout but make its evidence axis-specific:
        # a healthy yaw response must not hide a failed vx/vy command.
        stuck = translation_stuck | yaw_stuck
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
                "record_contract": (
                    None
                    if self.current_record_contract is None
                    else dict(self.current_record_contract)
                ),
            }
        )

    def _legacy_parent_payload_rejection(self, record: dict) -> str | None:
        expected_shapes = {
            "aux": (self.num_envs, p15_contract.RESPONSE_AUX_DIM),
            "velocity": (self.num_envs, 3, 3),
            "pose": (self.num_envs, 3),
            "stuck": (self.num_envs, 1),
            "horizon_mask": (self.num_envs, 3),
            "pose_mask": (self.num_envs, 1),
            "episode_start": (self.num_envs,),
            "command_phase": (self.num_envs,),
            "terrain_family": (self.num_envs,),
            "terrain_level": (self.num_envs,),
        }
        for name, shape in expected_shapes.items():
            value = record.get(name)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                return f"shape_{name}"
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                return f"nonfinite_{name}"
        if not isinstance(self.current_record_contract, dict):
            return "missing_current_contract"
        digest = str(record.get("low_level_digest", ""))
        if len(digest) != 64:
            return "low_level_digest"
        try:
            int(digest, 16)
            int(record.get("low_level_iteration", -1))
        except (TypeError, ValueError):
            return "low_level_version"
        return None

    def _legacy_parent_contract(
        self, lineage_tier: str, source_low_level_digest: str
    ) -> dict[str, object]:
        expected = self.current_record_contract
        if not isinstance(expected, dict):
            raise RuntimeError("legacy parent migration requires current record contract")
        if lineage_tier == "earlier_lineage":
            capability = list(p15_contract.CAPABILITY_PROFILE15)
            capability_digest = "legacy_p15_response_capability_profile15"
            action_mapper = "p15_piecewise_union_command_v1"
        else:
            capability = list(p2_contract.RESPONSE_CAPABILITY_PROFILE15)
            capability_digest = "legacy_p2_response_capability_profile15"
            action_mapper = p2_contract.command_contract()["version"]
        return {
            "version": "legacy_p3_parent_record_v1",
            "schema": expected["schema"],
            "low_level_digest": str(source_low_level_digest),
            "feedback_digest": "legacy_unversioned_structurally_validated",
            "capability_digest": capability_digest,
            "response_capability_profile15": capability,
            "action_mapper": action_mapper,
            "observation_layout": expected["observation_layout"],
            "label_layout": expected["label_layout"],
            "migration": "legacy_parent_structural_v1",
        }

    def load_parent_completed_records(self, state: dict[str, object]) -> int:
        temporary = P2ResponseAuxBuffer(
            self.num_envs,
            self.device,
            capacity_steps=self.capacity_steps,
            sequence_length=self.sequence_length,
            burn_in_steps=self.burn_in_steps,
        )
        restore_mode = temporary.load_checkpoint_state(state)
        self._parent_records.clear()
        self.legacy_parent_records_migrated = 0
        self.legacy_parent_migration_rejections.clear()
        if restore_mode.startswith("buffer_reset"):
            rejected = state.get("records", []) if isinstance(state, dict) else []
            if isinstance(rejected, list) and rejected:
                self.legacy_parent_migration_rejections[restore_mode] = len(rejected)
            return 0
        source_pools = (
            (temporary._records, "p35_parent"),
            (temporary._parent_records, "earlier_lineage"),
        )
        for records, lineage_tier in source_pools:
            for record in records:
                copied = dict(record)
                copied["record_origin"] = "parent"
                copied["record_lineage_tier"] = lineage_tier
                copied["current_segment"] = torch.full(
                    (self.num_envs,), -1.0, device=self.device
                )
                if not isinstance(copied.get("record_contract"), dict) and isinstance(
                    self.current_record_contract, dict
                ):
                    if (
                        self.current_record_contract.get("command_transition_mode")
                        == "instant_hold_10hz"
                    ):
                        reason = "legacy_missing_instant_command_provenance"
                        self.legacy_parent_migration_rejections[reason] = (
                            self.legacy_parent_migration_rejections.get(reason, 0) + 1
                        )
                        continue
                    rejection = self._legacy_parent_payload_rejection(copied)
                    if rejection is not None:
                        self.legacy_parent_migration_rejections[rejection] = (
                            self.legacy_parent_migration_rejections.get(rejection, 0) + 1
                        )
                        continue
                    copied["record_contract"] = self._legacy_parent_contract(
                        lineage_tier, str(copied["low_level_digest"])
                    )
                    self.legacy_parent_records_migrated += 1
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
        if self.replay_policy.startswith("p4_compatible"):
            return self._sample_p4_compatible(batch_envs=batch_envs, generator=generator)
        if self.replay_policy.startswith("p3_versioned"):
            return self._sample_p3_versioned(batch_envs=batch_envs, generator=generator)
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

    def _sample_pool(self, records, count, generator):
        if count <= 0 or not records:
            return None
        original = self._records
        self._records = records
        try:
            return super().sample(batch_envs=max(1, count), generator=generator)
        finally:
            self._records = original

    def _sample_p3_versioned(self, *, batch_envs: int, generator=None):
        requested = max(1, int(batch_envs))
        current = list(self._records)
        iterations = sorted({int(record.get("low_level_iteration", -1)) for record in current})
        latest_iteration = iterations[-1] if iterations else -1
        latest = deque(
            (record for record in current if int(record.get("low_level_iteration", -1)) == latest_iteration),
            maxlen=self.capacity_steps,
        )
        recent = deque(
            (record for record in current if int(record.get("low_level_iteration", -1)) < latest_iteration),
            maxlen=self.capacity_steps,
        )
        latest_ratio, recent_ratio, _ = self.p3_replay_ratios
        latest_count = int(round(requested * latest_ratio))
        recent_count = int(round(requested * recent_ratio))
        parent_count = max(0, requested - latest_count - recent_count)
        batches = [
            self._sample_pool(latest, latest_count, generator),
            self._sample_pool(recent, recent_count, generator),
            self._sample_pool(self._parent_records, parent_count, generator),
        ]
        available = [batch for batch in batches if batch is not None]
        if not available:
            return None
        result = available[0]
        for batch in available[1:]:
            result = self._concat_batches(result, batch)
        counts = [
            0 if batch is None else int(batch.observations.shape[1]) for batch in batches
        ]
        result.metadata.update(
            replay_origin="p3_latest_recent_parent",
            latest_batch_envs=counts[0],
            recent_batch_envs=counts[1],
            parent_batch_envs=counts[2],
            latest_completed_iteration=latest_iteration,
            active_version_lag=max(0, int(self.low_level_iteration) - latest_iteration),
        )
        return result

    def _contract_compatible(self, record: dict) -> bool:
        expected = self.current_record_contract
        actual = record.get("record_contract")
        if expected is None:
            return True
        if not isinstance(actual, dict):
            self.compatibility_rejections["missing_contract"] = (
                self.compatibility_rejections.get("missing_contract", 0) + 1
            )
            return False
        if actual.get("migration") == "legacy_parent_structural_v1":
            rejection = self._legacy_parent_payload_rejection(record)
            if record.get("record_origin") != "parent" or rejection is not None:
                reason = f"legacy_{rejection or 'non_parent'}"
                self.compatibility_rejections[reason] = (
                    self.compatibility_rejections.get(reason, 0) + 1
                )
                return False
            lineage_tier = str(record.get("record_lineage_tier", "p35_parent"))
            legacy_expected = self._legacy_parent_contract(
                lineage_tier, str(record.get("low_level_digest", ""))
            )
            for key in (
                "version",
                "schema",
                "low_level_digest",
                "capability_digest",
                "response_capability_profile15",
                "action_mapper",
                "observation_layout",
                "label_layout",
                "migration",
            ):
                if actual.get(key) != legacy_expected.get(key):
                    reason = f"legacy_mismatch_{key}"
                    self.compatibility_rejections[reason] = (
                        self.compatibility_rejections.get(reason, 0) + 1
                    )
                    return False
            return True
        for key in (
            "version",
            "schema",
            "low_level_digest",
            "feedback_digest",
            "capability_digest",
            "response_capability_profile15",
            "action_mapper",
            "observation_layout",
            "label_layout",
            "command_contract_digest",
            "command_transition_mode",
            "command_hold_frames",
        ):
            if actual.get(key) != expected.get(key):
                reason = f"mismatch_{key}"
                self.compatibility_rejections[reason] = (
                    self.compatibility_rejections.get(reason, 0) + 1
                )
                return False
        return True

    def _sample_p4_compatible(self, *, batch_envs: int, generator=None):
        requested = max(1, int(batch_envs))
        self.compatibility_rejections.clear()
        current = deque(
            (record for record in self._records if self._contract_compatible(record)),
            maxlen=self.capacity_steps,
        )
        parent = deque(
            (record for record in self._parent_records if self._contract_compatible(record)),
            maxlen=self.capacity_steps,
        )
        target_counts = [int(round(requested * 0.50)), int(round(requested * 0.25))]
        target_counts.append(max(0, requested - sum(target_counts)))
        p35_parent = deque(
            (
                record
                for record in parent
                if record.get("record_lineage_tier", "p35_parent") != "earlier_lineage"
            ),
            maxlen=self.capacity_steps,
        )
        earlier_lineage = deque(
            (
                record
                for record in parent
                if record.get("record_lineage_tier") == "earlier_lineage"
            ),
            maxlen=self.capacity_steps,
        )
        pools = (
            current,
            p35_parent,
            earlier_lineage,
        )
        batches = [
            self._sample_pool(pool, count, generator)
            for pool, count in zip(pools, target_counts)
        ]
        actual = [0 if batch is None else int(batch.observations.shape[1]) for batch in batches]
        missing = requested - sum(actual)
        if missing > 0:
            for pool_index, fill_pool in enumerate(pools):
                fill = self._sample_pool(fill_pool, missing, generator)
                if fill is None:
                    continue
                if batches[pool_index] is None:
                    batches[pool_index] = fill
                else:
                    batches[pool_index] = self._concat_batches(
                        batches[pool_index], fill
                    )
                actual[pool_index] += int(fill.observations.shape[1])
                break
        available = [batch for batch in batches if batch is not None]
        if not available:
            return None
        result = available[0]
        for batch in available[1:]:
            result = self._concat_batches(result, batch)
        total = max(1, sum(actual))
        result.metadata.update(
            replay_origin="p4_compatible_filtered",
            p4_current_batch_envs=actual[0],
            p35_parent_batch_envs=actual[1],
            earlier_lineage_batch_envs=actual[2],
            p4_current_actual_ratio=actual[0] / total,
            p35_parent_actual_ratio=actual[1] / total,
            earlier_lineage_actual_ratio=actual[2] / total,
            compatible_current_records=len(current),
            compatible_parent_records=len(parent),
            rejected_records=sum(self.compatibility_rejections.values()),
        )
        return result

    def checkpoint_state(self) -> dict[str, object]:
        state = super().checkpoint_state()
        state["num_envs"] = self.num_envs
        state["parent_record_count"] = len(self._parent_records)
        state["parent_records"] = [
            self._cpu_record(record) for record in self._parent_records
        ]
        state["replay_policy"] = self.replay_policy
        state["p3_replay_ratios"] = list(self.p3_replay_ratios)
        state["current_record_contract"] = self.current_record_contract
        state["compatibility_rejections"] = dict(self.compatibility_rejections)
        state["legacy_parent_records_migrated"] = self.legacy_parent_records_migrated
        state["legacy_parent_migration_rejections"] = dict(
            self.legacy_parent_migration_rejections
        )
        state["push_horizon_rejections"] = self.push_horizon_rejections.clone()
        state["push_pose_rejections"] = self.push_pose_rejections
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
        self._history_push_epoch.clear()
        self._history_seconds_since_push.clear()
        self._next_episode_start.fill_(True)
        for record in self._records:
            if "current_segment" not in record:
                record["current_segment"] = torch.full(
                    (self.num_envs,), -1.0, device=self.device
                )
        self._parent_records.clear()
        if isinstance(state, dict):
            self.replay_policy = str(state.get("replay_policy", self.replay_policy))
            ratios = state.get("p3_replay_ratios", self.p3_replay_ratios)
            self.set_p3_replay_ratios(*ratios)
            contract = state.get("current_record_contract")
            self.current_record_contract = dict(contract) if isinstance(contract, dict) else None
            rejections = state.get("compatibility_rejections", {})
            self.compatibility_rejections = {
                str(key): int(value) for key, value in rejections.items()
            } if isinstance(rejections, dict) else {}
            self.legacy_parent_records_migrated = int(
                state.get("legacy_parent_records_migrated", 0)
            )
            legacy_rejections = state.get("legacy_parent_migration_rejections", {})
            self.legacy_parent_migration_rejections = {
                str(key): int(value) for key, value in legacy_rejections.items()
            } if isinstance(legacy_rejections, dict) else {}
            push_rejections = state.get("push_horizon_rejections")
            if torch.is_tensor(push_rejections) and push_rejections.numel() == 3:
                self.push_horizon_rejections.copy_(push_rejections.reshape(3).cpu())
            self.push_pose_rejections = int(state.get("push_pose_rejections", 0))
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
