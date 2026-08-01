#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Shared checkpoint helpers for staged Kaiwu training.

The training bundle is intentionally broader than any one algorithm.  Later
stages may add ``vision_encoder`` or ``high_level`` modules without changing
the top-level schema.  Deployment artifacts remain separate.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any

import torch


KAIWU_TRAIN_FORMAT = "kaiwu_train_v1"
KAIWU_TRAIN_SCHEMA_VERSION = 1
KAIWU_TRAIN_SCHEMA_V2 = 2
SUPPORTED_KAIWU_TRAIN_SCHEMAS = (KAIWU_TRAIN_SCHEMA_VERSION, KAIWU_TRAIN_SCHEMA_V2)


class CheckpointSaveError(RuntimeError):
    """A checkpoint could not be serialized, written, or verified."""

DAGGER_PHASE_LABELS = (
    "daggerzero",
    "daggerquarter",
    "daggerhalf",
    "daggerthreequarter",
    "daggerfull",
)

# 视觉 ramp 阶段标签（阶段 4：深度视觉蒸馏）。
# 纯小写字母，必须满足探活正则 _PROBE_NAME 的 [a-z]* 段。
# 线性 ramp 不产生离散档位晋升；这些标签仅用于 ramp 关键比例点
# 和任务结束时落盘的阶段性文件名，便于追溯。
VISION_PHASE_LABELS = (
    "visionteacher",     # 早期（低学生比例）窗口的代表文件
    "visionhalf",        # ramp 到约 50% 学生比例
    "visionfull",        # ramp 到 100% 或任务结束的规范主文件
    "visionblocked",     # soft-stay 冻结后未恢复、提前结束
)

VISUAL_RL_PHASE_LABELS = (
    "rlcritic",
    "rlactor",
    "rlfull",
)

# Anchor R2 四小时四阶段标签（§5.5）。纯小写字母，满足探活正则。
# resume 候选优先级（同 ID 内）：anchorfinal -> anchoranneal -> anchoractor
# -> anchorcritic。
VISUAL_ANCHOR_R2_PHASE_LABELS = (
    "anchorcritic",
    "anchoractor",
    "anchoranneal",
    "anchorfinal",
)

# Standard command-generalization phase labels.  Digits are intentionally kept
# out of the stage segment so Arena's model probe treats these as valid files.
VISUAL_COMMAND_PHASE_LABELS = (
    "commandbase",
    "commandblend",
    "commandfull",
)

P15_RESPONSE_PHASE_LABELS = (
    "responsebase",
    "responseexpand",
    "responsefull",
    "responsecalib",
)

P2_NAV_PHASE_LABELS = (
    "navwarm",
    "navadapt",
    "vywarm",
    "vyadapt",
    "navfull",
    "safewarm",
    "safefull",
    "safestable",
)

P3_STANDARD_JOINT_PHASE_LABELS = (
    "gaitcalib",
    "lowbase",
    "lowmild",
    "lowmedium",
    "lowfull",
    "adaptercalib",
    "highadapt",
    "highslow",
    "staircalib",
    "stairwarm",
    "stairadapt",
    "stairrobust",
    "stairfinal",
    "gaitfixcalib",
    "repair",
    "pushwarm",
    "pushfull",
    "stable",
)

_PROBE_NAME = re.compile(r"^model\.ckpt-[a-z]*-*[0-9]+\.[^.]+$")


def is_kaiwu_train_bundle(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("format") == KAIWU_TRAIN_FORMAT
        and int(value.get("schema_version", 0)) in SUPPORTED_KAIWU_TRAIN_SCHEMAS
    )


def normalize_kaiwu_train_bundle(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a schema-2-compatible view while preserving legacy aliases.

    Schema 1 remains a valid low-level parent. The normalized view does not
    fabricate missing optimizer or high-level state; it only gives schema-2
    readers stable module/training-state locations.
    """
    if not is_kaiwu_train_bundle(value):
        raise ValueError(
            "Expected kaiwu_train_v1 schema 1/2 checkpoint, got "
            f"format={getattr(value, 'get', lambda *_: None)('format')!r}, "
            f"schema={getattr(value, 'get', lambda *_: None)('schema_version')!r}"
        )
    bundle = dict(value)
    original_schema = int(bundle.get("schema_version", 0))
    modules = dict(bundle.get("modules") or {})
    low_level = dict(modules.get("low_level") or {})
    vision_state = (modules.get("vision_encoder") or {}).get("state_dict")
    actor_state = low_level.get("actor_state_dict")
    critic_state = (modules.get("critic") or {}).get("state_dict")
    if isinstance(vision_state, dict):
        low_level.setdefault(
            "locomotion_encoder",
            {"class_name": "VisionEncoder", "state_dict": vision_state, "spec": {}},
        )
    if isinstance(actor_state, dict):
        low_level.setdefault(
            "actor",
            {"class_name": "Actor77Sequential", "state_dict": actor_state, "spec": {}},
        )
    if isinstance(critic_state, dict):
        low_level.setdefault(
            "critic",
            {"class_name": "VisualCritic", "state_dict": critic_state, "spec": {}},
        )
    low_level.setdefault("contract_version", "low_level_v2")
    modules["low_level"] = low_level
    bundle["modules"] = modules

    legacy_state = bundle.get("training_state")
    training_states = dict(bundle.get("training_states") or {})
    if isinstance(legacy_state, dict):
        training_states.setdefault("low_level", dict(legacy_state))
    training_states.setdefault(
        "global",
        {
            "last_active_scope": "low_level",
            "bundle_revision": 0,
            "compound_schedule_phase": None,
        },
    )
    bundle["training_states"] = training_states
    report = {
        "source_format": bundle.get("format"),
        "source_schema": original_schema,
        "normalized_schema": KAIWU_TRAIN_SCHEMA_V2,
        "recognized_modules": sorted(modules.keys()),
        "optimizer_restore_level": (
            "exact_resume"
            if isinstance((bundle.get("optimizers") or {}).get("visual_ppo"), dict)
            else "weights_only"
        ),
    }
    return bundle, report


def p15_response_checkpoint_candidates(path: str, model_id: str | int) -> list[str]:
    model_id = str(model_id)
    return [
        os.path.join(path, f"model.ckpt-{label}-{model_id}.pkl")
        for label in reversed(P15_RESPONSE_PHASE_LABELS)
    ]


def p15_response_parent_candidates(path: str, model_id: str | int) -> list[str]:
    candidates = [
        *p15_response_checkpoint_candidates(path, model_id),
        *visual_command_parent_candidates(path, model_id),
    ]
    result: list[str] = []
    for candidate in candidates:
        if candidate not in result and os.path.isfile(candidate):
            result.append(candidate)
    return result


def p2_nav_checkpoint_candidates(path: str, model_id: str | int) -> list[str]:
    """Prefer same-ID P2 resume/evaluation candidates, newest phase first."""
    model_id = str(model_id)
    return [
        os.path.join(path, f"model.ckpt-{label}-{model_id}.pkl")
        for label in reversed(P2_NAV_PHASE_LABELS)
    ]


def p2_nav_discovery_candidates(path: str) -> list[str]:
    """Discover P2 bundles without treating the platform model ID as a gate.

    The Arena-provided ID is useful selection metadata, but stale or rewritten
    IDs must not make an otherwise compatible bundle unloadable.  Phase
    priority remains deterministic; within one phase the newest file wins.
    Structural compatibility is still validated by the P2 loader.
    """
    result: list[str] = []
    for label in reversed(P2_NAV_PHASE_LABELS):
        discovered = sorted(
            glob.glob(os.path.join(path, f"model.ckpt-{label}-*.pkl")),
            key=lambda filename: (os.path.getmtime(filename), filename),
            reverse=True,
        )
        for candidate in discovered:
            if os.path.isfile(candidate) and candidate not in result:
                result.append(candidate)
    return result


def p2_nav_parent_candidates(path: str, model_id: str | int) -> list[str]:
    """Return the preferred P1.5 responsecalib parent for one lineage ID."""
    candidate = os.path.join(
        path,
        f"model.ckpt-responsecalib-{str(model_id)}.pkl",
    )
    return [candidate] if os.path.isfile(candidate) else []


def p2_nav_parent_discovery_candidates(path: str) -> list[str]:
    """Discover responsecalib parents as a warning-only identity fallback."""
    return sorted(
        (
            candidate
            for candidate in glob.glob(
                os.path.join(path, "model.ckpt-responsecalib-*.pkl")
            )
            if os.path.isfile(candidate)
        ),
        key=lambda filename: (os.path.getmtime(filename), filename),
        reverse=True,
    )


def p2_nav_evaluation_candidates(
    path: str, model_id: str | int
) -> list[str]:
    """Prefer the requested P2 ID, then discover bundles when files are absent.

    Candidate fallback is existence-only. Once a file is selected, deserialization
    and structural compatibility failures must remain hard errors.
    """
    result: list[str] = []
    for candidate in (
        *p2_nav_checkpoint_candidates(path, model_id),
        *p2_nav_discovery_candidates(path),
    ):
        if candidate not in result:
            result.append(candidate)
    return result


def p2_nav_training_candidates(
    path: str,
    model_id: str | int,
    *,
    parent_model_id: str | int,
) -> list[str]:
    """Prefer exact resume, then the configured parent, then absent-file fallbacks.

    A requested-ID mismatch is diagnostic only.  The configured parent is
    always considered, even when the platform injects another preload ID.
    Once an existing file is selected, structural failures must not silently
    downgrade exact resume or replace the configured parent.
    """
    result: list[str] = []
    for candidate in (
        *p2_nav_checkpoint_candidates(path, model_id),
        *p2_nav_checkpoint_candidates(path, parent_model_id),
        *p2_nav_parent_candidates(path, parent_model_id),
        *p2_nav_discovery_candidates(path),
        *p2_nav_parent_discovery_candidates(path),
    ):
        if candidate not in result:
            result.append(candidate)
    return result


def p3_standard_joint_candidates(
    path: str,
    model_id: str | int,
    *,
    parent_model_id: str | int,
) -> list[str]:
    """Prefer exact P3 resume, then the explicitly configured P2 parent."""
    result: list[str] = []
    for selected_id in (str(model_id), str(parent_model_id)):
        for label in reversed(P3_STANDARD_JOINT_PHASE_LABELS):
            candidate = os.path.join(
                path, f"model.ckpt-{label}-{selected_id}.pkl"
            )
            if candidate not in result:
                result.append(candidate)
    for candidate in p2_nav_checkpoint_candidates(path, parent_model_id):
        if candidate not in result:
            result.append(candidate)
    if any(os.path.isfile(candidate) for candidate in result):
        return result
    explicit = {os.path.abspath(candidate) for candidate in result}
    discovered_p3 = sorted(
        {
            candidate
            for label in P3_STANDARD_JOINT_PHASE_LABELS
            for candidate in glob.glob(os.path.join(path, f"model.ckpt-{label}-*.pkl"))
            if os.path.abspath(candidate) not in explicit
        }
    )
    discovered_p2 = [
        candidate
        for candidate in p2_nav_discovery_candidates(path)
        if os.path.abspath(candidate) not in explicit
    ]
    discovered = discovered_p3 + discovered_p2
    if len(discovered) > 1:
        raise RuntimeError(
            "P3 checkpoint discovery is ambiguous; configure the parent/model ID "
            f"explicitly. candidates={discovered}"
        )
    if discovered:
        result.append(discovered[0])
    return result


P3_EVAL_LOW_LEVEL_SPEC = {
    "locomotion_encoder": {
        "class_name": "VisionEncoder",
        "spec": {
            "image_shape": [180, 320, 1],
            "proprio_dim": 45,
            "cnn_output_dim": 32,
            "rnn_hidden_dim": 64,
            "rnn_num_layers": 2,
            "rnn_output_dim": 32,
            "use_lstm": True,
        },
    },
    "actor": {
        "class_name": "Actor77Sequential",
        "spec": {
            "input_dim": 77,
            "hidden_dims": [512, 256, 128],
            "output_dim": 12,
            "activation": "elu",
        },
    },
}


def p3_standard_joint_eval_candidates(path: str, model_id: str | int) -> list[str]:
    """Discover P3 checkpoints for the ``p3_standard_eval`` / ``p3_track_eval`` entries.

    Candidate fallback is existence-only and never crosses stage families:

      * Same-ID P3 phase files first, newest phase first
        (``highslow > highadapt > adaptercalib > lowfull > lowmedium > lowmild
        > lowbase``), exactly like the training resume priority.
      * When no same-ID P3 file exists, discover *unique* P3 phase files across
        any ID.  Multiple candidates are an explicit ambiguity error -- never a
        by-mtime pick, and never a silent fallback to P2/LBC/random weights.

    A requested-ID mismatch with the selected file is warning-only identity
    metadata.  Once a file is selected, deserialization / structural failures
    must remain hard errors in the loader.
    """
    model_id = str(model_id)
    same_id = [
        os.path.join(path, f"model.ckpt-{label}-{model_id}.pkl")
        for label in reversed(P3_STANDARD_JOINT_PHASE_LABELS)
    ]
    if any(os.path.isfile(candidate) for candidate in same_id):
        return same_id
    discovered = sorted(
        {
            candidate
            for label in P3_STANDARD_JOINT_PHASE_LABELS
            for candidate in glob.glob(os.path.join(path, f"model.ckpt-{label}-*.pkl"))
            if os.path.isfile(candidate)
        },
        key=lambda filename: (os.path.basename(filename), filename),
    )
    if len(discovered) > 1:
        raise RuntimeError(
            "P3 eval checkpoint discovery is ambiguous; configure the eval "
            f"model ID explicitly. candidates={discovered}"
        )
    return discovered


def validate_p3_eval_bundle(bundle: dict[str, Any], *, mode: str) -> dict[str, Any]:
    """Shared structural validator for P3 Standard/Track evaluation.

    ``mode`` is ``"standard"`` (low-level VisionEncoder + Actor77 only) or
    ``"track"`` (low-level + NavigationEncoder + three-axis Actor +
    ResponseAdapter).  The P3 package is the authoritative module source for
    both entries; neither may fall back to an old P2/LBC payload.

    Platform model ID, filename label and lineage are *not* structural
    compatibility inputs -- the caller reports them as warning-only identity
    metadata.  Format, stage, phase label, model_spec, module class/spec, state
    dict shape and finite values are the correctness basis.

    Returns an eval disposition dict with the selected ``phase_label`` and the
    modules that must be loaded.
    """
    if not is_kaiwu_train_bundle(bundle):
        raise ValueError(
            "P3 evaluation requires a kaiwu_train_v1 checkpoint, got "
            f"format={bundle.get('format')!r}, "
            f"schema={bundle.get('schema_version')!r}"
        )
    if bundle.get("stage_type") != "p3_standard_joint":
        raise ValueError(
            "P3 evaluation requires stage_type='p3_standard_joint', got "
            f"{bundle.get('stage_type')!r}"
        )
    phase_label = bundle.get("phase_label")
    if not isinstance(phase_label, str) or phase_label not in P3_STANDARD_JOINT_PHASE_LABELS:
        raise ValueError(
            "P3 evaluation requires a valid phase_label, got "
            f"{phase_label!r}"
        )
    validate_low_level_spec(
        bundle,
        {
            "proprio_dim": 45,
            "scan_dim": 256,
            "latent_dim": 32,
            "action_dim": 12,
            "goal_dim": 0,
        },
    )
    modules = bundle.get("modules")
    if not isinstance(modules, dict):
        raise KeyError("modules missing from P3 checkpoint")
    low = modules.get("low_level")
    if not isinstance(low, dict) or low.get("contract_version") != "low_level_v2":
        raise ValueError("P3 evaluation low-level contract mismatch")

    expected_low = P3_EVAL_LOW_LEVEL_SPEC
    loaded: list[str] = []
    for name, expected in expected_low.items():
        leaf = low.get(name)
        _validate_p3_eval_leaf(leaf, name, expected, context="P3 eval low_level")
        loaded.append(f"low_level.{name}")
    for name, state_key in (
        ("locomotion_encoder", "state_dict"),
        ("actor", "state_dict"),
    ):
        validate_state_dict_finite(low[name][state_key], f"P3 eval low_level.{name}")

    if mode == "track":
        high = modules.get("high_level")
        if (
            not isinstance(high, dict)
            or high.get("contract_version") != "high_level_continuous_v2"
            or high.get("component_status") != "complete"
        ):
            raise ValueError("P3 track evaluation requires complete high-level contract")
        from agent_ppo.model.p2_high_level import (
            navigation_actor_spec,
            navigation_encoder_spec,
        )
        from agent_ppo.model.response_adapter import response_adapter_spec

        expected_high = {
            "navigation_encoder": {
                "class_name": "NavigationEncoder",
                "spec": navigation_encoder_spec(),
            },
            "actor": {"class_name": "P2NavigationActor", "spec": navigation_actor_spec()},
            "response_adapter": {
                "class_name": "CommandResponseAdapter",
                "spec": response_adapter_spec(),
            },
        }
        for name, expected in expected_high.items():
            leaf = high.get(name)
            _validate_p3_eval_leaf(leaf, name, expected, context="P3 track eval high_level")
            loaded.append(f"high_level.{name}")
            validate_state_dict_finite(leaf["state_dict"], f"P3 track eval high_level.{name}")
    elif mode != "standard":
        raise ValueError(f"P3 eval mode must be 'standard' or 'track', got {mode!r}")

    return {
        "stage_type": "p3_standard_joint",
        "phase_label": phase_label,
        "mode": mode,
        "loaded_modules": sorted(loaded),
    }


def _validate_p3_eval_leaf(leaf: Any, name: str, expected: dict[str, Any], *, context: str) -> None:
    """Validate one P3 eval leaf (class_name / spec / state_dict presence)."""
    if not isinstance(leaf, dict):
        raise KeyError(f"{context} missing leaf {name}")
    if leaf.get("class_name") != expected["class_name"]:
        raise ValueError(
            f"{context}.{name} class mismatch: {leaf.get('class_name')!r}"
        )
    if leaf.get("spec") != expected["spec"]:
        raise ValueError(f"{context}.{name} spec mismatch: {leaf.get('spec')!r}")
    if not isinstance(leaf.get("state_dict"), dict):
        raise KeyError(f"{context}.{name} missing state_dict")


def low_level_only_parent_candidates(path: str, model_id: str | int) -> list[str]:
    """Explicit opt-in candidates for extracting only Standard low-level state."""
    result: list[str] = []
    for candidate in (
        *p15_response_checkpoint_candidates(path, model_id),
        *visual_command_checkpoint_candidates(path, model_id),
        *vision_checkpoint_candidates(path, model_id),
    ):
        if candidate not in result:
            result.append(candidate)
    return result


def classify_locomotion_eval_high_level(
    bundle: dict[str, Any], *, allow_complete_hier_nav_low_level_only: bool = False
) -> str:
    """Classify optional high-level state before low-level Camera evaluation.

    P1.5 stores its auxiliary ResponseAdapter under ``modules.high_level`` even
    though that component does not produce locomotion actions at evaluation.
    A real hier-nav policy also uses ``modules.high_level`` and must never be
    silently discarded. The adapter itself is not validated here because it is
    not loaded by this evaluation path; Standard/Track low-level reuse must not
    depend on an unused auxiliary component.
    """
    modules = bundle.get("modules")
    if not isinstance(modules, dict):
        raise KeyError("modules missing from kaiwu_train_v1 checkpoint")
    if "high_level" not in modules:
        return "absent"

    high_level = modules.get("high_level")
    if (
        allow_complete_hier_nav_low_level_only
        and bundle.get("stage_type") == "p2_nav_ppo"
        and isinstance(high_level, dict)
        and high_level.get("component_status") == "complete"
    ):
        return "complete_hier_nav_ignored_by_explicit_low_level_only_eval"
    expected_spec = {
        "class_name": "CommandResponseAdapter",
        "input_dim": 32,
        "hidden_dim": 64,
        "profile_dim": 16,
    }
    if (
        isinstance(high_level, dict)
        and high_level.get("component_status") == "adapter_only"
    ):
        if bundle.get("stage_type") != "p15_response_adapter":
            raise ValueError(
                "adapter-only high_level requires stage_type='p15_response_adapter'"
            )
        if set(high_level) != {"component_status", "response_adapter"}:
            raise ValueError(
                "adapter-only high_level contains unexpected fields; refusing to "
                "drop a possible action-producing policy"
            )
        adapter = high_level.get("response_adapter")
        spec = adapter.get("spec") if isinstance(adapter, dict) else None
        if not isinstance(spec, dict) or any(
            spec.get(key) != value for key, value in expected_spec.items()
        ):
            raise ValueError(
                "adapter-only high_level has an incompatible ResponseAdapter spec"
            )
        return "adapter_only_ignored_for_locomotion_eval"

    raise ValueError(
        "modules.high_level is not an adapter-only auxiliary component; "
        "refusing to drop a possible action-producing hier-nav policy. "
        "Evaluate hier-nav bundles via the nav_eval policy_entry"
    )


def validate_probe_filename(filename: str) -> bool:
    """Mirror the Arena probe restriction: stage labels are lowercase letters."""
    return bool(_PROBE_NAME.match(os.path.basename(filename)))


def phase_label(phase_index: int) -> str:
    try:
        return DAGGER_PHASE_LABELS[int(phase_index)]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid DAgger phase index: {phase_index}") from exc


def vision_phase_label(name: str) -> str:
    """校验并返回视觉 ramp 阶段标签。

    线性 ramp 不用整数 phase index（没有离散档位），改用显式标签名。
    仅接受 VISION_PHASE_LABELS 中的纯小写字母标签，保证探活正则可匹配。
    """
    if name not in VISION_PHASE_LABELS:
        raise ValueError(
            f"Invalid vision ramp label: {name!r}; "
            f"expected one of {VISION_PHASE_LABELS}"
        )
    return name


def checkpoint_candidates(path: str, model_id: str | int) -> list[str]:
    """Return deterministic checkpoint candidates for one platform model ID."""
    model_id = str(model_id)
    preferred = [
        os.path.join(path, f"model.ckpt-locomotion-{model_id}.pkl"),
        *[
            os.path.join(path, f"model.ckpt-{label}-{model_id}.pkl")
            for label in reversed(DAGGER_PHASE_LABELS)
        ],
        os.path.join(path, f"model.ckpt-{model_id}.pkl"),
        os.path.join(path, f"model.ckpt-lbc-loco-{model_id}.pkl"),
    ]
    discovered = sorted(
        glob.glob(os.path.join(path, f"model.ckpt-*-{model_id}.*")),
        reverse=True,
    )
    result: list[str] = []
    for candidate in [*preferred, *discovered]:
        if candidate not in result:
            result.append(candidate)
    return result


def vision_checkpoint_candidates(path: str, model_id: str | int) -> list[str]:
    """视觉蒸馏阶段的 checkpoint 候选（阶段 4）。

    与特权 DAgger 的 checkpoint_candidates 区分：
      - 优先 vision* 标签文件（规范主文件优先 visionfull，其次 visionhalf 等）；
      - 再回退 lbc-loco / 裸 ID（兼容历史视觉产物）；
      - locomotion / dagger* 不作为视觉续训候选（它们是特权教师，不是视觉学生）。

    视觉训练包 format=kaiwu_train_v1 且含 modules.vision_encoder；loader 在读取
    后会再次校验，这里只保证文件存在性和优先级。
    """
    model_id = str(model_id)
    preferred = [
        os.path.join(path, f"model.ckpt-visionfull-{model_id}.pkl"),
        os.path.join(path, f"model.ckpt-visionhalf-{model_id}.pkl"),
        os.path.join(path, f"model.ckpt-visionteacher-{model_id}.pkl"),
        os.path.join(path, f"model.ckpt-visionblocked-{model_id}.pkl"),
        os.path.join(path, f"model.ckpt-lbc-loco-{model_id}.pkl"),
        os.path.join(path, f"model.ckpt-{model_id}.pkl"),
    ]
    # Only discover visual checkpoints here.  The preload directory for the
    # first visual run normally contains both daggerfull and locomotion files.
    # Feeding either low-level parent into load_vision_bundle() raises because
    # they intentionally do not contain modules.vision_encoder, and prevents
    # the caller from reaching vision_parent_candidates().
    visual_prefixes = tuple(
        f"model.ckpt-{label}-{model_id}." for label in VISION_PHASE_LABELS
    )
    discovered = sorted(
        (
            candidate
            for candidate in glob.glob(
                os.path.join(path, f"model.ckpt-*-{model_id}.*")
            )
            if os.path.basename(candidate).startswith(visual_prefixes)
        ),
        reverse=True,
    )
    result: list[str] = []
    for candidate in [*preferred, *discovered]:
        if candidate not in result:
            result.append(candidate)
    return result


def visual_command_checkpoint_candidates(
    path: str, model_id: str | int
) -> list[str]:
    """Return command-generalization bundles for one explicit platform ID."""
    model_id = str(model_id)
    preferred = [
        os.path.join(path, f"model.ckpt-{label}-{model_id}.pkl")
        for label in reversed(VISUAL_COMMAND_PHASE_LABELS)
    ]
    result: list[str] = []
    for candidate in preferred:
        if candidate not in result:
            result.append(candidate)
    return result


def visual_command_parent_candidates(
    path: str, model_id: str | int
) -> list[str]:
    """Select a same-ID Anchor R2/command parent for transition resume."""
    candidates = [
        *visual_command_checkpoint_candidates(path, model_id),
        *visual_anchor_r2_checkpoint_candidates(path, model_id),
        *visual_anchor_r2_parent_candidates(path, model_id),
    ]
    result: list[str] = []
    for candidate in candidates:
        if candidate not in result and os.path.isfile(candidate):
            result.append(candidate)
    return result


def visual_rl_checkpoint_candidates(
    path: str, model_id: str | int
) -> list[str]:
    """Return all visual-PPO resumes for one explicit numeric model ID."""
    model_id = str(model_id)
    preferred = [
        *visual_anchor_r2_checkpoint_candidates(path, model_id),
        *[
            os.path.join(path, f"model.ckpt-{label}-{model_id}.pkl")
            for label in reversed(VISUAL_RL_PHASE_LABELS)
        ],
        *vision_checkpoint_candidates(path, model_id),
    ]
    result: list[str] = []
    for candidate in preferred:
        if candidate not in result:
            result.append(candidate)
    return result


# Anchor R2 filename: model.ckpt-<label>-<digits>.pkl
_ANCHOR_R2_FILENAME = re.compile(
    r"^model\.ckpt-(?P<label>" + "|".join(VISUAL_ANCHOR_R2_PHASE_LABELS) + r")"
    r"-(?P<id>[0-9]+)\.pkl$"
)


def _parse_anchor_r2_filename(filename: str) -> tuple[str, int] | None:
    """Return (label, numeric_id) parsed from an Anchor R2 filename, else None."""
    match = _ANCHOR_R2_FILENAME.match(os.path.basename(filename))
    if match is None:
        return None
    return match.group("label"), int(match.group("id"))


def visual_anchor_r2_checkpoint_candidates(
    path: str, model_id: str | int
) -> list[str]:
    """Anchor R2 resume candidates for an explicit numeric selector (§5.5 N8).

    ID constraint enforced at this candidate layer, not in the Agent:
      1. Explicit numeric ID: filter by filename numeric ID equality FIRST,
         then order by label priority (anchorfinal -> anchoranneal ->
         anchoractor -> anchorcritic). Cross-ID fallback never happens.
    Returns only existing files. Callers that also want the S0 parent
    (visionfull-<id>) should append vision_checkpoint_candidates separately.
    """
    try:
        requested_id = int(model_id)
    except (TypeError, ValueError):
        return []
    label_priority = {
        label: index
        for index, label in enumerate(reversed(VISUAL_ANCHOR_R2_PHASE_LABELS))
    }
    discovered: list[tuple[int, str]] = []
    for filename in glob.glob(os.path.join(path, "model.ckpt-*.pkl")):
        parsed = _parse_anchor_r2_filename(filename)
        if parsed is None:
            continue
        label, file_id = parsed
        if file_id != requested_id:
            # Cross-ID fallback is forbidden (§5.5 N8 rule 1).
            continue
        if not os.path.isfile(filename):
            continue
        discovered.append((label_priority[label], filename))
    discovered.sort(key=lambda item: item[0])
    return [filename for _, filename in discovered]


def _latest_visual_anchor_r2_candidates(path: str) -> list[str]:
    """Anchor R2 resume candidates for the ``latest`` selector (§5.5 N8 rule 2).

    Resolve the maximum filename numeric ID among Anchor R2 labels FIRST, then
    order by label priority within that single ID. A higher-priority label on a
    smaller ID is never returned before the max ID.
    """
    ids: list[tuple[int, str]] = []
    for filename in glob.glob(os.path.join(path, "model.ckpt-*.pkl")):
        parsed = _parse_anchor_r2_filename(filename)
        if parsed is None or not os.path.isfile(filename):
            continue
        ids.append((parsed[1], filename))
    if not ids:
        return []
    max_id = max(file_id for file_id, _ in ids)
    return visual_anchor_r2_checkpoint_candidates(path, max_id)


def visual_latest_model_id(path: str) -> int | None:
    """Return the largest numeric ID among all visual checkpoint labels."""
    labels = (
        *P15_RESPONSE_PHASE_LABELS,
        *VISUAL_COMMAND_PHASE_LABELS,
        *VISUAL_ANCHOR_R2_PHASE_LABELS,
        *VISUAL_RL_PHASE_LABELS,
        *VISION_PHASE_LABELS,
    )
    pattern = re.compile(
        r"^model\.ckpt-(?:"
        + "|".join(labels)
        + r")-(?P<id>[0-9]+)\.[^.]+$"
    )
    ids = []
    for filename in glob.glob(os.path.join(path, "model.ckpt-*.*")):
        match = pattern.match(os.path.basename(filename))
        if match and os.path.isfile(filename):
            ids.append(int(match.group("id")))
    return max(ids) if ids else None


def visual_anchor_r2_parent_candidates(path: str, model_id: str | int) -> list[str]:
    """First-load S0 parent candidates for Anchor R2 (§5.5).

    The exact ``visionfull-<id>`` file must come before any same-ID historical
    ``vis*`` file so the first run loads the real S0 baseline. Anchor R2 resume
    files (anchor*) are intentionally excluded here — they are resume candidates,
    not a parent.
    """
    model_id = str(model_id)
    preferred = [
        os.path.join(path, f"model.ckpt-visionfull-{model_id}.pkl"),
        os.path.join(path, f"model.ckpt-visionhalf-{model_id}.pkl"),
        os.path.join(path, f"model.ckpt-visionteacher-{model_id}.pkl"),
        os.path.join(path, f"model.ckpt-visionblocked-{model_id}.pkl"),
        os.path.join(path, f"model.ckpt-{model_id}.pkl"),
    ]
    result: list[str] = []
    for candidate in preferred:
        if candidate not in result and os.path.isfile(candidate):
            result.append(candidate)
    return result


def visual_anchor_r2_training_candidates(
    path: str,
    model_id: str | int,
    *,
    initial_parent_model_id: str | int,
) -> list[str]:
    """Select the fixed S0 first run or an explicit later Anchor R2 resume."""
    parent = visual_anchor_r2_parent_candidates(path, model_id)
    if str(model_id) == str(initial_parent_model_id):
        return parent
    return [
        *visual_anchor_r2_checkpoint_candidates(path, model_id),
        *parent,
    ]


def visual_anchor_r2_eval_candidates(path: str, model_id: str | int) -> list[str]:
    """Camera-eval candidates for Anchor R2 (§5.5).

    Tries Anchor R2 labels first (within the requested ID), then falls back to
    superseded R3 (rl*), historical Stage-4 vision labels, and finally the
    explicit S0 visionfull parent. The ID constraint (§5.5 N8 rule 1) is
    enforced on the Anchor R2 label set; legacy labels fall through to their
    own candidate functions which also scope by ID.
    """
    candidates = visual_anchor_r2_checkpoint_candidates(path, model_id)
    # Superseded R3 / historical visual RL labels (rlcritic/rlactor/rlfull).
    model_id_str = str(model_id)
    for label in reversed(VISUAL_RL_PHASE_LABELS):
        legacy = os.path.join(path, f"model.ckpt-{label}-{model_id_str}.pkl")
        if legacy not in candidates and os.path.isfile(legacy):
            candidates.append(legacy)
    # Stage-4 vision labels + S0 visionfull parent.
    for legacy in visual_anchor_r2_parent_candidates(path, model_id):
        if legacy not in candidates:
            candidates.append(legacy)
    return candidates


def visual_eval_checkpoint_candidates(path: str, model_id: str | int) -> list[str]:
    """Public camera-eval candidate API with command/Anchor ordering."""
    if str(model_id) == "latest":
        resolved_id = visual_latest_model_id(path)
        if resolved_id is None:
            return []
        model_id = resolved_id
    candidates = [
        candidate
        for candidate in p15_response_checkpoint_candidates(path, model_id)
        if os.path.isfile(candidate)
    ]
    for candidate in visual_command_checkpoint_candidates(path, model_id):
        if os.path.isfile(candidate) and candidate not in candidates:
            candidates.append(candidate)
    for candidate in visual_anchor_r2_eval_candidates(path, model_id):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def visual_eval_checkpoint_diagnostics(path: str, model_id: str | int) -> dict[str, Any]:
    """Describe an explicit-ID eval miss without relaxing the ID boundary.

    A platform-selected ID may legitimately have no compatible visual bundle.
    The loader must reject that unreadable selection, but the operator needs to
    see the same-ID filenames considered and any visual files present under
    other IDs.  This function never returns another ID as a load candidate.
    """
    requested_id = str(model_id)
    same_id_expected: list[str] = [
        *p15_response_checkpoint_candidates(path, requested_id),
        *visual_command_checkpoint_candidates(path, requested_id),
        *[
            os.path.join(path, f"model.ckpt-{label}-{requested_id}.pkl")
            for label in reversed(VISUAL_ANCHOR_R2_PHASE_LABELS)
        ],
        *[
            os.path.join(path, f"model.ckpt-{label}-{requested_id}.pkl")
            for label in reversed(VISUAL_RL_PHASE_LABELS)
        ],
        *[
            os.path.join(path, f"model.ckpt-{label}-{requested_id}.pkl")
            for label in VISION_PHASE_LABELS
        ],
        os.path.join(path, f"model.ckpt-{requested_id}.pkl"),
    ]
    labels = tuple(
        dict.fromkeys(
            (
                *P15_RESPONSE_PHASE_LABELS,
                *VISUAL_COMMAND_PHASE_LABELS,
                *VISUAL_ANCHOR_R2_PHASE_LABELS,
                *VISUAL_RL_PHASE_LABELS,
                *VISION_PHASE_LABELS,
            )
        )
    )
    visual_pattern = re.compile(
        r"^model\.ckpt-(?:"
        + "|".join(re.escape(label) for label in labels)
        + r")-(?P<id>[0-9]+)\.[^.]+$"
    )
    other_visual_files: list[str] = []
    for candidate in sorted(glob.glob(os.path.join(path, "model.ckpt-*.*"))):
        match = visual_pattern.match(os.path.basename(candidate))
        if match is not None and match.group("id") != requested_id:
            other_visual_files.append(candidate)
    return {
        "requested_id": requested_id,
        "same_id_expected": same_id_expected,
        "same_id_existing": [
            candidate for candidate in same_id_expected if os.path.isfile(candidate)
        ],
        "other_visual_files": other_visual_files,
    }


def validate_visual_eval_bundle_identity(
    bundle: dict[str, Any], model_id: str | int, logger=None
) -> dict[str, Any]:
    """Inspect camera-eval identity metadata without blocking a valid payload.

    Filename filtering is necessary but not sufficient: a manually copied or
    stale file can have a matching filename while carrying another model's
    payload. ``platform_model_id`` is the primary identity; older bundles may
    retain it under ``lineage.platform_model_id``. Identity metadata is a
    diagnostic rather than an execution contract. Candidate
    selection still uses the requested filename ID, while missing or conflicting
    metadata is surfaced as a warning. Payload format and tensor compatibility
    remain hard requirements in the caller.
    """
    if not is_kaiwu_train_bundle(bundle):
        raise ValueError(
            "Camera eval requires a kaiwu_train_v1 checkpoint, got "
            f"format={bundle.get('format')!r}, "
            f"schema={bundle.get('schema_version')!r}"
        )

    requested_id = str(model_id)
    lineage_value = bundle.get("lineage", {})
    metadata_warnings: list[str] = []
    if isinstance(lineage_value, dict):
        lineage = lineage_value
    else:
        lineage = {}
        metadata_warnings.append(
            "Camera eval checkpoint lineage metadata is not a dict; ignoring it"
        )

    bundle_id = bundle.get("platform_model_id")
    lineage_id = lineage.get("platform_model_id")
    present_ids = {
        "platform_model_id": bundle_id,
        "lineage.platform_model_id": lineage_id,
    }
    mismatches = {
        key: value
        for key, value in present_ids.items()
        if value not in (None, "") and str(value) != requested_id
    }
    warnings: list[str] = list(metadata_warnings)
    if mismatches:
        warnings.append(
            "Camera eval checkpoint identity differs from the requested model; "
            f"continuing with the operator-selected same-ID file: "
            f"requested={requested_id}, identities={present_ids}"
        )
    if all(value in (None, "") for value in present_ids.values()):
        warnings.append(
            "Camera eval checkpoint has no platform identity metadata; "
            f"continuing after structural validation: requested={requested_id}, "
            f"identities={present_ids}"
        )
    if logger is not None:
        for message in warnings:
            logger.warning(f"[CheckpointIdentity] WARNING-ONLY: {message}")

    return {
        "requested_id": requested_id,
        "bundle_id": bundle_id,
        "lineage_id": lineage_id,
        "lineage": lineage,
        "identity_warnings": warnings,
    }


def validate_low_level_spec(bundle: dict[str, Any], expected: dict[str, int]) -> None:
    if not is_kaiwu_train_bundle(bundle):
        raise ValueError(
            "Expected kaiwu_train_v1 checkpoint, got "
            f"format={bundle.get('format')!r}, schema={bundle.get('schema_version')!r}"
        )
    spec = bundle.get("model_spec")
    if not isinstance(spec, dict):
        raise KeyError("model_spec missing from kaiwu_train_v1 checkpoint")
    mismatches = {
        key: (spec.get(key), value)
        for key, value in expected.items()
        if int(spec.get(key, -1)) != int(value)
    }
    if mismatches:
        raise ValueError(f"Low-level checkpoint contract mismatch: {mismatches}")


def vision_parent_candidates(path: str, model_id: str | int) -> list[str]:
    """视觉蒸馏的特权教师父文件候选（阶段 4 首训加载）。

    P1 修正：当前 checkpoint_candidates 把 locomotion 排在 dagger* 之前，
    依赖"两文件同 SHA"的偶然排序来命中 daggerfull-16288。视觉阶段不能靠
    偶然排序，因此显式给出父文件候选顺序：

      1. 显式 daggerfull 标签（规范父文件，阶段 2 产物）
      2. locomotion 别名（兼容副本，与 daggerfull 同 payload）
      3. 任何 kaiwu_train_v1 low-level 包（兜底）

    loader 加载后会打印实际命中路径、SHA 和 model_spec，供操作者核对
    是否真的是 daggerfull-16288 血缘。
    """
    model_id = str(model_id)
    preferred = [
        os.path.join(path, f"model.ckpt-daggerfull-{model_id}.pkl"),
        os.path.join(path, f"model.ckpt-locomotion-{model_id}.pkl"),
        *[
            os.path.join(path, f"model.ckpt-{label}-{model_id}.pkl")
            for label in reversed(DAGGER_PHASE_LABELS)
            if label != "daggerfull"
        ],
        os.path.join(path, f"model.ckpt-{model_id}.pkl"),
    ]
    discovered = sorted(
        glob.glob(os.path.join(path, f"model.ckpt-*-{model_id}.*")),
        reverse=True,
    )
    result: list[str] = []
    for candidate in [*preferred, *discovered]:
        if candidate not in result:
            result.append(candidate)
    return result


def low_level_policy_state(bundle: dict[str, Any]) -> dict[str, Any]:
    modules = bundle.get("modules")
    if not isinstance(modules, dict):
        raise KeyError("modules missing from kaiwu_train_v1 checkpoint")
    low_level = modules.get("low_level")
    if not isinstance(low_level, dict):
        raise KeyError("modules.low_level missing from kaiwu_train_v1 checkpoint")
    state = low_level.get("policy_state_dict")
    if not isinstance(state, dict):
        raise KeyError("modules.low_level.policy_state_dict missing")
    return state


def low_level_teacher_parts(
    bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    modules = bundle.get("modules")
    if not isinstance(modules, dict):
        raise KeyError("modules missing from kaiwu_train_v1 checkpoint")
    low_level = modules.get("low_level")
    if not isinstance(low_level, dict):
        raise KeyError("modules.low_level missing from kaiwu_train_v1 checkpoint")
    encoder = low_level.get("encoder_state_dict")
    actor = low_level.get("actor_state_dict")
    if not isinstance(encoder, dict) or not isinstance(actor, dict):
        raise KeyError(
            "modules.low_level encoder_state_dict/actor_state_dict are required"
        )
    return encoder, actor


# =========================================================================
# hier-nav 独立 checkpoint 命名空间
#
# 纪律（计划步骤 8）：
#   - nav 标签绝不加入 visual_latest_model_id / visual_eval_checkpoint_
#     diagnostics / 任何 visual* candidates —— 否则普通 Camera/LBC 评估的
#     latest 会误选 nav 组合包。本节全部为新增符号，不改上面任何现有函数。
#   - 跨 ID fallback 禁止：显式数字 ID 只匹配同 ID 文件。
#   - nav eval 候选只认 nav 标签：没有同 ID nav 文件就返回空（宁可硬失败，
#     不回退到 loco-only 包静默评分）。
# =========================================================================

NAV_PHASE_LABELS = (
    "navbc",        # BC 预热段（ramp 低比例窗口）
    "navdagger",    # DAgger 爬坡段
    "navfull",      # ramp 100% / 任务结束的规范主文件
)


def nav_phase_label(name: str) -> str:
    if name not in NAV_PHASE_LABELS:
        raise ValueError(
            f"Invalid nav phase label: {name!r}; expected one of {NAV_PHASE_LABELS}"
        )
    return name


_NAV_FILENAME = re.compile(
    r"^model\.ckpt-(?P<label>" + "|".join(NAV_PHASE_LABELS) + r")"
    r"-(?P<id>[0-9]+)\.pkl$"
)


def _parse_nav_filename(filename: str) -> tuple[str, int] | None:
    """Return (label, numeric_id) parsed from a nav filename, else None."""
    match = _NAV_FILENAME.match(os.path.basename(filename))
    if match is None:
        return None
    return match.group("label"), int(match.group("id"))


def nav_checkpoint_candidates(path: str, model_id: str | int) -> list[str]:
    """Same-ID nav resume candidates (navfull -> navdagger -> navbc).

    Cross-ID fallback is forbidden: filename numeric ID must equal the
    requested ID. Returns only existing files.
    """
    try:
        requested_id = int(model_id)
    except (TypeError, ValueError):
        return []
    label_priority = {
        label: index for index, label in enumerate(reversed(NAV_PHASE_LABELS))
    }
    discovered: list[tuple[int, str]] = []
    for filename in glob.glob(os.path.join(path, "model.ckpt-*.pkl")):
        parsed = _parse_nav_filename(filename)
        if parsed is None:
            continue
        label, file_id = parsed
        if file_id != requested_id:
            continue
        if not os.path.isfile(filename):
            continue
        discovered.append((label_priority[label], filename))
    discovered.sort(key=lambda item: item[0])
    return [filename for _, filename in discovered]


def nav_latest_model_id(path: str) -> int | None:
    """Largest numeric ID among nav labels ONLY (never sees visual/lbc files)."""
    ids = []
    for filename in glob.glob(os.path.join(path, "model.ckpt-*.pkl")):
        parsed = _parse_nav_filename(filename)
        if parsed is not None and os.path.isfile(filename):
            ids.append(parsed[1])
    return max(ids) if ids else None


def nav_parent_candidates(path: str, model_id: str | int) -> list[str]:
    """Low-level parent candidates for nav first-load (e.g. command-34728).

    Reuses the command-generalization parent chain (commandfull ->
    commandblend -> commandbase -> anchor R2 -> vision labels), all scoped
    to the same ID. Nav labels are intentionally NOT included here — nav
    files are resume candidates, not a low-level parent.
    """
    return visual_command_parent_candidates(path, model_id)


def nav_training_candidates(
    path: str,
    model_id: str | int,
    *,
    low_level_parent_model_id: str | int,
) -> list[str]:
    """First-load from the low-level parent, or an explicit nav resume.

    When the platform-injected id equals the configured low-level parent id,
    this is the bootstrap first load: return ONLY low-level parent candidates
    (the in-memory high_level stays randomly initialized). Otherwise return
    same-ID nav resumes ONLY — falling back to a same-ID low-level parent is
    forbidden (it would silently discard high-level progress and produce a
    self-contradictory lineage; switching parents must be done explicitly by
    changing low_level_parent_model_id, which routes back to the first branch).
    """
    if str(model_id) == str(low_level_parent_model_id):
        return nav_parent_candidates(path, model_id)
    return nav_checkpoint_candidates(path, model_id)


def nav_eval_checkpoint_candidates(path: str, model_id: str | int) -> list[str]:
    """Nav-eval candidates: nav labels only, same ID only.

    ``latest`` resolves via nav_latest_model_id (nav namespace only). No
    fallback to visual/lbc labels: a nav eval that cannot find a same-ID nav
    bundle must hard-fail instead of silently scoring a loco-only bundle.
    """
    if str(model_id) == "latest":
        resolved_id = nav_latest_model_id(path)
        if resolved_id is None:
            return []
        model_id = resolved_id
    return nav_checkpoint_candidates(path, model_id)


def nav_eval_checkpoint_diagnostics(path: str, model_id: str | int) -> dict[str, Any]:
    """Actionable miss report for nav eval (mirrors the visual diagnostics)."""
    requested = str(model_id)
    resolved = requested
    if requested == "latest":
        latest = nav_latest_model_id(path)
        resolved = str(latest) if latest is not None else None
    expected = (
        [
            os.path.join(path, f"model.ckpt-{label}-{resolved}.pkl")
            for label in reversed(NAV_PHASE_LABELS)
        ]
        if resolved is not None
        else []
    )
    existing_nav = sorted(
        filename
        for filename in glob.glob(os.path.join(path, "model.ckpt-*.pkl"))
        if _parse_nav_filename(filename) is not None
    )
    return {
        "requested_model_id": requested,
        "resolved_model_id": resolved,
        "same_id_expected": expected,
        "existing_nav_files": existing_nav,
        "search_dir": path,
    }


def high_level_parts(bundle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (state_dict, meta) from modules.high_level; KeyError when absent."""
    modules = bundle.get("modules")
    if not isinstance(modules, dict):
        raise KeyError("modules missing from kaiwu_train_v1 checkpoint")
    high_level = modules.get("high_level")
    if not isinstance(high_level, dict):
        raise KeyError("modules.high_level missing from kaiwu_train_v1 checkpoint")
    state = high_level.get("state_dict")
    if not isinstance(state, dict):
        raise KeyError("modules.high_level.state_dict missing")
    meta = {key: value for key, value in high_level.items() if key != "state_dict"}
    return state, meta


def validate_high_level_spec(bundle: dict[str, Any], expected: dict[str, Any]) -> None:
    """Validate modules.high_level metadata against the nav contract.

    ``expected`` keys are compared exactly (e.g. input_layout_version, vocab,
    nav_period_frames, min_dwell_ticks). Vocab is compared as nested lists.
    """
    state, meta = high_level_parts(bundle)
    validate_state_dict_finite(state, "modules.high_level.state_dict")
    mismatches = {}
    for key, want in expected.items():
        got = meta.get(key)
        if key == "vocab":
            got_norm = [list(map(float, row)) for row in (got or [])]
            want_norm = [list(map(float, row)) for row in want]
            if got_norm != want_norm:
                mismatches[key] = (got, want)
        elif got != want:
            mismatches[key] = (got, want)
    if mismatches:
        raise ValueError(f"high_level contract mismatch: {mismatches}")


def validate_state_dict_finite(state: dict[str, Any], name: str) -> None:
    def _walk(mapping: dict[str, Any], prefix: str):
        for key, value in mapping.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                yield from _walk(value, field)
            elif torch.is_tensor(value):
                yield field, value

    for key, value in _walk(state, ""):
        if not bool(torch.isfinite(value.detach()).all()):
            raise FloatingPointError(f"non-finite tensor in {name}: {key}")


def _digest_state_dict(hasher, name: str, state: dict[str, Any]) -> None:
    validate_state_dict_finite(state, name)
    for key in sorted(state.keys()):
        value = state[key]
        hasher.update(f"{name}/{key}".encode("utf-8"))
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
            hasher.update(str(tensor.dtype).encode("utf-8"))
            hasher.update(tensor.numpy().tobytes())
        else:
            hasher.update(repr(value).encode("utf-8"))


def compute_low_level_state_digest(bundle: dict[str, Any]) -> str:
    """State-dict-level sha256 over the frozen low level.

    Covers modules.vision_encoder.state_dict + modules.low_level.actor_state_dict
    (sorted keys, shape/dtype/raw bytes). File-level hashes cannot be reused —
    every new nav save produces a different file hash even when the low level
    is bit-identical; this digest is the invariant that must not change.
    """
    import hashlib

    modules = bundle.get("modules")
    if not isinstance(modules, dict):
        raise KeyError("modules missing from kaiwu_train_v1 checkpoint")
    vision = modules.get("vision_encoder")
    if not isinstance(vision, dict) or not isinstance(vision.get("state_dict"), dict):
        raise KeyError("modules.vision_encoder.state_dict missing")
    low_level = modules.get("low_level")
    if not isinstance(low_level, dict) or not isinstance(
        low_level.get("actor_state_dict"), dict
    ):
        raise KeyError("modules.low_level.actor_state_dict missing")

    hasher = hashlib.sha256()
    _digest_state_dict(hasher, "vision_encoder", vision["state_dict"])
    _digest_state_dict(hasher, "low_level_actor", low_level["actor_state_dict"])
    return hasher.hexdigest()


def validate_nav_eval_bundle(
    bundle: dict[str, Any],
    model_id: str | int,
    expected_low_level_spec: dict[str, int],
    expected_high_level: dict[str, Any],
    logger=None,
) -> dict[str, Any]:
    """Validate executable nav structure and report identity metadata drift.

    Format, required modules, model contracts and state-dict compatibility are
    hard requirements. Platform identity, lineage and digest are provenance
    diagnostics: they warn but do not make an otherwise executable package
    unusable.
    """
    identity = validate_visual_eval_bundle_identity(bundle, model_id, logger=logger)
    validate_low_level_spec(bundle, expected_low_level_spec)
    validate_high_level_spec(bundle, expected_high_level)

    lineage = identity["lineage"] or {}
    stored_digest = lineage.get("low_level_state_digest")
    computed_digest = compute_low_level_state_digest(bundle)
    digest_warnings: list[str] = []
    if not stored_digest:
        digest_warnings.append(
            "nav eval checkpoint has no lineage.low_level_state_digest; "
            f"using computed digest={computed_digest}"
        )
    elif str(stored_digest) != computed_digest:
        digest_warnings.append(
            "nav eval low-level digest differs from lineage; continuing with "
            f"the structurally compatible loaded tensors: lineage={stored_digest} "
            f"computed={computed_digest}"
        )
    if logger is not None:
        for message in digest_warnings:
            logger.warning(f"[CheckpointIdentity] WARNING-ONLY: {message}")
    identity["identity_warnings"].extend(digest_warnings)
    identity["low_level_state_digest"] = computed_digest
    identity["high_level_present"] = True
    return identity
