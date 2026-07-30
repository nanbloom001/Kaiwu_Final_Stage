#!/usr/bin/env python3
"""Read-only diagnostics for the generic Track terrain curriculum."""

from __future__ import annotations

import json
import time

import torch

from agent_ppo.feature import p2_contract


class P2CurriculumAccumulator:
    """Aisrv-side curriculum accounting reconstructed from response_aux30."""

    RESET_LOG_INTERVAL_S = 60.0
    RESET_LOG_SAMPLE_LIMIT = 12

    def __init__(self, *, curriculum_enabled: bool = False):
        self.curriculum_enabled = bool(curriculum_enabled)
        self.previous_rows = None
        self.previous_cols = None
        self.row_moves = torch.zeros(3, dtype=torch.long)
        self.col_moves = torch.zeros(2, dtype=torch.long)
        self.outcomes = torch.zeros(3, p2_contract.TERRAIN_NUM_COLUMNS, 3, dtype=torch.long)
        self.start_counts = torch.zeros(3, p2_contract.TERRAIN_NUM_COLUMNS, dtype=torch.long)
        self.last_rows_histogram = [0, 0, 0]
        self.last_cols_histogram = [0] * p2_contract.TERRAIN_NUM_COLUMNS
        self.last_joint_histogram = [[0] * p2_contract.TERRAIN_NUM_COLUMNS for _ in range(3)]
        self.warned_semantics = False
        self.warned_inactive = False
        self.last_reset_log_elapsed_s = 0.0
        self.pending_reset_envs = 0
        self.pending_reason_counts = torch.zeros(3, dtype=torch.long)
        self.pending_row_moves = torch.zeros(3, dtype=torch.long)
        self.pending_col_moves = torch.zeros(2, dtype=torch.long)
        self.pending_reset_samples: list[dict[str, int]] = []

    @staticmethod
    def _bounded_metadata(aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cols = aux[:, 28].detach().to(device="cpu", dtype=torch.long).clamp_(0, p2_contract.TERRAIN_NUM_COLUMNS - 1)
        rows = aux[:, 29].detach().to(device="cpu", dtype=torch.long).clamp_(0, 2)
        return rows, cols

    def _update_histograms(self, rows: torch.Tensor, cols: torch.Tensor) -> None:
        joint = torch.zeros(3, p2_contract.TERRAIN_NUM_COLUMNS, dtype=torch.long)
        joint.index_put_((rows, cols), torch.ones_like(rows), accumulate=True)
        self.last_rows_histogram = torch.bincount(rows, minlength=3).tolist()
        self.last_cols_histogram = torch.bincount(cols, minlength=p2_contract.TERRAIN_NUM_COLUMNS).tolist()
        self.last_joint_histogram = joint.tolist()

    def _clear_pending_reset_log(self) -> None:
        self.pending_reset_envs = 0
        self.pending_reason_counts.zero_()
        self.pending_row_moves.zero_()
        self.pending_col_moves.zero_()
        self.pending_reset_samples.clear()

    def _maybe_log_reset_summary(self, logger, elapsed_s: float) -> None:
        if (
            logger is None
            or self.pending_reset_envs == 0
            or elapsed_s - self.last_reset_log_elapsed_s < self.RESET_LOG_INTERVAL_S
        ):
            return
        logger.info(
            "[P2CurriculumProbe] "
            + json.dumps(
                {
                    "event": "p2_curriculum_reset_summary_aisrv",
                    "window_s": float(elapsed_s - self.last_reset_log_elapsed_s),
                    "reset_envs": self.pending_reset_envs,
                    "termination_reason_counts": {
                        "success": int(self.pending_reason_counts[0]),
                        "failure": int(self.pending_reason_counts[1]),
                        "timeout": int(self.pending_reason_counts[2]),
                    },
                    "row_moves": {
                        "demotion": int(self.pending_row_moves[0]),
                        "unchanged": int(self.pending_row_moves[1]),
                        "promotion": int(self.pending_row_moves[2]),
                    },
                    "column_moves": {
                        "unchanged": int(self.pending_col_moves[0]),
                        "changed": int(self.pending_col_moves[1]),
                    },
                    "samples": self.pending_reset_samples,
                    "row_label_interpretation": (
                        "row0=slope_inv,row1=stairs_inv,row2=maze_entry"
                    ),
                },
                ensure_ascii=True,
            )
        )
        self.last_reset_log_elapsed_s = float(elapsed_s)
        self._clear_pending_reset_log()

    def observe(self, aux: torch.Tensor, *, logger=None, elapsed_s: float = 0.0) -> None:
        if aux.ndim != 2 or aux.shape[1] != p2_contract.WORKER_AUX_DIM:
            raise ValueError(
                f"P2 curriculum aux must be [N,{p2_contract.WORKER_AUX_DIM}], "
                f"got {tuple(aux.shape)}"
            )
        rows, cols = self._bounded_metadata(aux)
        reset = (aux[:, 24] > 0.5).detach().to(device="cpu")
        reasons = aux[:, 25].detach().to(device="cpu", dtype=torch.long)
        if self.previous_rows is None:
            self.last_reset_log_elapsed_s = float(elapsed_s)
            self.start_counts.index_put_(
                (rows, cols), torch.ones_like(rows), accumulate=True
            )
        elif bool(reset.any()):
            ids = reset.nonzero(as_tuple=False).reshape(-1)
            old_rows = aux[
                ids, p2_contract.PRE_STEP_TERRAIN_LEVEL_INDEX
            ].detach().to(device="cpu", dtype=torch.long).clamp_(0, 2)
            old_cols = aux[
                ids, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX
            ].detach().to(device="cpu", dtype=torch.long).clamp_(0, p2_contract.TERRAIN_NUM_COLUMNS - 1)
            new_rows = rows[ids]
            new_cols = cols[ids]
            reset_reasons = reasons[ids]
            reset_reasons = torch.where(
                torch.isin(reset_reasons, torch.tensor((1, 2, 3))),
                reset_reasons,
                torch.full_like(reset_reasons, 3),
            )
            row_moves = torch.bincount(
                torch.sign(new_rows - old_rows) + 1, minlength=3
            )
            col_moves = torch.bincount((new_cols != old_cols).long(), minlength=2)
            self.row_moves += row_moves
            self.col_moves += col_moves
            self.pending_reset_envs += ids.numel()
            self.pending_reason_counts += torch.bincount(
                reset_reasons, minlength=4
            )[1:4]
            self.pending_row_moves += row_moves
            self.pending_col_moves += col_moves
            for index in range(ids.numel()):
                outcome = int(reset_reasons[index]) - 1
                self.outcomes[int(old_rows[index]), int(old_cols[index]), outcome] += 1
                self.start_counts[int(new_rows[index]), int(new_cols[index])] += 1
                if len(self.pending_reset_samples) < self.RESET_LOG_SAMPLE_LIMIT:
                    self.pending_reset_samples.append(
                        {
                            "env_id": int(ids[index]),
                            "old_row": int(old_rows[index]),
                            "new_row": int(new_rows[index]),
                            "old_col": int(old_cols[index]),
                            "new_col": int(new_cols[index]),
                            "termination_reason_code": int(reset_reasons[index]),
                        }
                    )
        self.previous_rows = rows.clone()
        self.previous_cols = cols.clone()
        self._update_histograms(rows, cols)
        self._maybe_log_reset_summary(logger, float(elapsed_s))
        row_changed = int(self.row_moves[0] + self.row_moves[2]) > 0
        col_changed = int(self.col_moves[1]) > 0
        if self.curriculum_enabled and elapsed_s >= 20.0 * 60.0 and logger is not None:
            if row_changed and not col_changed and not self.warned_semantics:
                logger.warning(
                    "[P2CurriculumProbe] generic curriculum appears to control "
                    "track segment, not maze difficulty"
                )
                self.warned_semantics = True
            if not row_changed and not col_changed and not self.warned_inactive:
                logger.warning(
                    "[P2CurriculumProbe] generic curriculum may be inactive for Track"
                )
                self.warned_inactive = True

    def state_dict(self) -> dict[str, object]:
        return {
            "transport": "p2_worker_aux62_v3_20cols",
            "curriculum_enabled": self.curriculum_enabled,
            "row_moves": self.row_moves.tolist(),
            "col_moves": self.col_moves.tolist(),
            "outcomes": self.outcomes.tolist(),
            "start_counts": self.start_counts.tolist(),
            "last_terrain_levels_histogram": self.last_rows_histogram,
            "last_terrain_types_histogram": self.last_cols_histogram,
            "last_row_col_joint": self.last_joint_histogram,
            "warning_semantics_emitted": self.warned_semantics,
            "warning_inactive_emitted": self.warned_inactive,
        }

    def reset_live_boundary(self) -> None:
        self.previous_rows = None
        self.previous_cols = None
        self.last_reset_log_elapsed_s = 0.0
        self._clear_pending_reset_log()

    def load_state_dict(self, state: dict[str, object]) -> None:
        if not isinstance(state, dict):
            raise ValueError("P2 curriculum diagnostics must be a mapping")
        if state.get("transport") != "p2_worker_aux62_v3_20cols":
            raise ValueError("P2 curriculum diagnostics transport mismatch")
        for name, shape in (
            ("row_moves", (3,)),
            ("col_moves", (2,)),
            ("outcomes", (3, p2_contract.TERRAIN_NUM_COLUMNS, 3)),
            ("start_counts", (3, p2_contract.TERRAIN_NUM_COLUMNS)),
        ):
            value = torch.as_tensor(state.get(name), dtype=torch.long)
            if tuple(value.shape) != shape:
                raise ValueError(f"P2 curriculum {name} shape mismatch: {tuple(value.shape)}")
            getattr(self, name).copy_(value)
        self.last_rows_histogram = list(
            state.get("last_terrain_levels_histogram", [0, 0, 0])
        )
        self.last_cols_histogram = list(
            state.get("last_terrain_types_histogram", [0] * p2_contract.TERRAIN_NUM_COLUMNS)
        )
        self.last_joint_histogram = list(
            state.get("last_row_col_joint", [[0] * p2_contract.TERRAIN_NUM_COLUMNS for _ in range(3)])
        )
        self.curriculum_enabled = bool(state.get("curriculum_enabled", False))
        self.warned_semantics = bool(state.get("warning_semantics_emitted", False))
        self.warned_inactive = bool(state.get("warning_inactive_emitted", False))
        self.reset_live_boundary()


class P2TrackCurriculumProbe:
    def __init__(self, num_envs: int, *, report_interval_s: float = 120.0):
        self.num_envs = int(num_envs)
        self.report_interval_s = float(report_interval_s)
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at
        self.previous_rows = None
        self.previous_cols = None
        self.row_moves = torch.zeros(3, dtype=torch.long)  # demote, unchanged, promote
        self.col_moves = torch.zeros(2, dtype=torch.long)  # unchanged, changed
        self.outcomes = torch.zeros(3, p2_contract.TERRAIN_NUM_COLUMNS, 3, dtype=torch.long)  # success/failure/timeout
        self.start_counts = torch.zeros(3, p2_contract.TERRAIN_NUM_COLUMNS, dtype=torch.long)
        self.warned_semantics = False
        self.warned_inactive = False
        self.initialized = False
        self.last_rows_histogram = [0, 0, 0]
        self.last_cols_histogram = [0] * p2_contract.TERRAIN_NUM_COLUMNS
        self.last_joint_histogram = [[0] * p2_contract.TERRAIN_NUM_COLUMNS for _ in range(3)]
        self.last_reset_sample = None

    @staticmethod
    def _cpu_long(value, count: int):
        if not torch.is_tensor(value) or value.numel() != count:
            return None
        return value.detach().reshape(-1).to(device="cpu", dtype=torch.long)

    @staticmethod
    def _active_terms(manager) -> tuple[str, ...]:
        if manager is None:
            return ()
        terms = getattr(manager, "active_terms", None)
        if terms is None:
            terms = getattr(manager, "_term_names", ())
        return tuple(terms or ())

    @staticmethod
    def _track_columns_initialized(env, lengths: torch.Tensor) -> bool:
        terrain = env.scene.terrain
        cfg = getattr(terrain, "cfg", None)
        generator = getattr(cfg, "terrain_generator", None)
        if not bool(getattr(generator, "curriculum", False)):
            return True

        raw_num_cols = getattr(generator, "num_cols", 1)
        num_cols = 1 if raw_num_cols is None else int(raw_num_cols)
        raw_max_init_level = getattr(cfg, "max_init_terrain_level", None)
        max_init_level = (
            num_cols - 1 if raw_max_init_level is None else int(raw_max_init_level)
        )
        if num_cols <= 1 or max_init_level >= num_cols - 1:
            return True

        marker = getattr(terrain, "_track_curriculum_col_initialized", None)
        if torch.is_tensor(marker):
            marker = marker.detach().reshape(-1).to(device="cpu", dtype=torch.bool)
            return marker.numel() == lengths.numel() and bool(marker.all())
        if isinstance(marker, (list, tuple)):
            return len(marker) == lengths.numel() and all(bool(value) for value in marker)
        if marker is not None:
            return bool(marker)

        # Older platform builds have no explicit marker. The first non-zero
        # episode length is the earliest unambiguous post-reset boundary.
        return bool((lengths > 0).any())

    @staticmethod
    def _termination_reasons(env, reset_ids: torch.Tensor) -> list[str]:
        manager = getattr(env, "termination_manager", None)
        if manager is None:
            return ["unknown"] * reset_ids.numel()
        timeouts = getattr(manager, "time_outs", None)
        terminated = getattr(manager, "terminated", None)
        active = set(P2TrackCurriculumProbe._active_terms(manager))
        goal = None
        if "goal_reached" in active:
            try:
                goal = manager.get_term("goal_reached")
            except Exception:
                goal = None
        reasons = []
        for env_id in reset_ids.tolist():
            if torch.is_tensor(goal) and bool(goal[env_id]):
                reasons.append("success")
            elif torch.is_tensor(timeouts) and bool(timeouts[env_id]):
                reasons.append("timeout")
            elif torch.is_tensor(terminated) and bool(terminated[env_id]):
                reasons.append("failure")
            else:
                reasons.append("unknown")
        return reasons

    def _initial_log(self, env, rows, cols) -> None:
        terrain = env.scene.terrain
        cfg = getattr(terrain, "cfg", None)
        generator = getattr(cfg, "terrain_generator", None)
        curriculum = getattr(env, "curriculum_manager", None)
        payload = {
            "event": "p2_curriculum_init",
            "terrain_generator_curriculum": getattr(generator, "curriculum", None),
            "curriculum_active_terms": list(self._active_terms(curriculum)),
            "track_length": getattr(generator, "track_length", None),
            "num_rows": getattr(generator, "num_rows", None),
            "num_parallel_tracks": getattr(generator, "num_parallel_tracks", None),
            "num_cols": getattr(generator, "num_cols", None),
            "sub_terrains_order": list(getattr(generator, "sub_terrains_order", ()) or ()),
            "sub_terrains_random": getattr(generator, "sub_terrains_random", None),
            "max_init_terrain_level": getattr(cfg, "max_init_terrain_level", None),
            "initial_rows": torch.bincount(rows, minlength=3).tolist(),
            "initial_cols": torch.bincount(cols, minlength=p2_contract.TERRAIN_NUM_COLUMNS).tolist(),
        }
        print("[P2CurriculumProbe] " + json.dumps(payload, ensure_ascii=True), flush=True)

    def _histograms(self, rows, cols):
        joint = torch.zeros(3, p2_contract.TERRAIN_NUM_COLUMNS, dtype=torch.long)
        for row, col in zip(rows.tolist(), cols.tolist()):
            joint[max(0, min(2, row)), max(0, min(p2_contract.TERRAIN_NUM_COLUMNS - 1, col))] += 1
        self.last_rows_histogram = torch.bincount(rows, minlength=3).tolist()
        self.last_cols_histogram = torch.bincount(cols, minlength=p2_contract.TERRAIN_NUM_COLUMNS).tolist()
        self.last_joint_histogram = joint.tolist()
        return joint

    def _summary(self, rows, cols, *, elapsed_s: float, initial: bool) -> dict:
        joint = self._histograms(rows, cols)
        return {
            "event": "p2_curriculum_summary",
            "initial": bool(initial),
            "elapsed_s": float(elapsed_s),
            "terrain_levels_histogram": self.last_rows_histogram,
            "terrain_types_histogram": self.last_cols_histogram,
            "row_col_joint": joint.tolist(),
            "row_moves": {
                "demotion": int(self.row_moves[0]),
                "unchanged": int(self.row_moves[1]),
                "promotion": int(self.row_moves[2]),
            },
            "column_moves": {
                "unchanged": int(self.col_moves[0]),
                "changed": int(self.col_moves[1]),
            },
            "outcomes_by_old_row_col": self.outcomes.tolist(),
            "starts_by_new_row_col": self.start_counts.tolist(),
            "slope_inv_starts": int(self.start_counts[0].sum()),
            "stairs_inv_starts": int(self.start_counts[1].sum()),
            "maze_entry_starts": int(self.start_counts[2].sum()),
            "last_reset_sample": self.last_reset_sample,
        }

    def observe(self, env, *, now_s: float | None = None) -> None:
        now = time.monotonic() if now_s is None else float(now_s)
        terrain = getattr(getattr(env, "scene", None), "terrain", None)
        if terrain is None:
            return
        rows = self._cpu_long(getattr(terrain, "terrain_levels", None), self.num_envs)
        cols = self._cpu_long(getattr(terrain, "terrain_types", None), self.num_envs)
        lengths = self._cpu_long(getattr(env, "episode_length_buf", None), self.num_envs)
        if rows is None or cols is None or lengths is None:
            return
        if not self.initialized:
            if not self._track_columns_initialized(env, lengths):
                return
            # Anchor reports to the first usable observation. This keeps
            # injected clocks and delayed terrain initialization consistent.
            self.started_at = now
            self.last_report_at = now
            self._initial_log(env, rows, cols)
            print(
                "[P2CurriculumProbe] "
                + json.dumps(
                    self._summary(rows, cols, elapsed_s=0.0, initial=True),
                    ensure_ascii=True,
                ),
                flush=True,
            )
            self.initialized = True
        if self.previous_rows is not None:
            reset_ids = torch.nonzero(lengths == 0, as_tuple=False).reshape(-1)
            if reset_ids.numel():
                old_rows = self.previous_rows[reset_ids]
                old_cols = self.previous_cols[reset_ids]
                new_rows = rows[reset_ids]
                new_cols = cols[reset_ids]
                reasons = self._termination_reasons(env, reset_ids)
                progress = getattr(env, "_p2_last_goal_progress", None)
                progress = (
                    progress.detach().reshape(-1).cpu()[reset_ids].tolist()
                    if torch.is_tensor(progress) and progress.numel() == self.num_envs
                    else [0.0] * reset_ids.numel()
                )
                delta = torch.sign(new_rows - old_rows) + 1
                self.row_moves += torch.bincount(delta, minlength=3)
                changed_col = (new_cols != old_cols).long()
                self.col_moves += torch.bincount(changed_col, minlength=2)
                for index, reason in enumerate(reasons):
                    row = int(torch.clamp(old_rows[index], 0, 2))
                    col = int(torch.clamp(old_cols[index], 0, p2_contract.TERRAIN_NUM_COLUMNS - 1))
                    outcome_index = {"success": 0, "failure": 1, "timeout": 2}.get(reason)
                    if outcome_index is not None:
                        self.outcomes[row, col, outcome_index] += 1
                    self.start_counts[
                        int(torch.clamp(new_rows[index], 0, 2)),
                        int(torch.clamp(new_cols[index], 0, p2_contract.TERRAIN_NUM_COLUMNS - 1)),
                    ] += 1
                sample_count = min(12, reset_ids.numel())
                self.last_reset_sample = {
                    "reset_env_count": int(reset_ids.numel()),
                    "env_ids": reset_ids[:sample_count].tolist(),
                    "old_row": old_rows[:sample_count].tolist(),
                    "new_row": new_rows[:sample_count].tolist(),
                    "old_col": old_cols[:sample_count].tolist(),
                    "new_col": new_cols[:sample_count].tolist(),
                    "termination_reason": reasons[:sample_count],
                    "last_goal_progress": progress[:sample_count],
                    "row_label_interpretation": (
                        "row0=slope_inv,row1=stairs_inv,row2=maze_entry"
                    ),
                }
        self.previous_rows = rows.clone()
        self.previous_cols = cols.clone()
        self._histograms(rows, cols)
        if now - self.last_report_at >= self.report_interval_s:
            summary = self._summary(
                rows, cols, elapsed_s=now - self.started_at, initial=False
            )
            print("[P2CurriculumProbe] " + json.dumps(summary, ensure_ascii=True), flush=True)
            self.last_report_at = now
        generator = getattr(getattr(getattr(env.scene.terrain, "cfg", None), "terrain_generator", None), "curriculum", False)
        if bool(generator) and now - self.started_at >= 20.0 * 60.0:
            row_changed = int(self.row_moves[0] + self.row_moves[2]) > 0
            col_changed = int(self.col_moves[1]) > 0
            if row_changed and not col_changed and not self.warned_semantics:
                print(
                    "[P2CurriculumProbe] WARNING generic curriculum appears to "
                    "control track segment, not maze difficulty",
                    flush=True,
                )
                self.warned_semantics = True
            if not row_changed and not col_changed and not self.warned_inactive:
                print(
                    "[P2CurriculumProbe] WARNING generic curriculum may be inactive "
                    "for Track",
                    flush=True,
                )
                self.warned_inactive = True

    def state_dict(self) -> dict[str, object]:
        return {
            "row_moves": self.row_moves.tolist(),
            "col_moves": self.col_moves.tolist(),
            "outcomes": self.outcomes.tolist(),
            "start_counts": self.start_counts.tolist(),
            "last_terrain_levels_histogram": self.last_rows_histogram,
            "last_terrain_types_histogram": self.last_cols_histogram,
            "last_row_col_joint": self.last_joint_histogram,
            "warning_semantics_emitted": self.warned_semantics,
            "warning_inactive_emitted": self.warned_inactive,
        }
