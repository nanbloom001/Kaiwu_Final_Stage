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


KAIWU_TRAIN_FORMAT = "kaiwu_train_v1"
KAIWU_TRAIN_SCHEMA_VERSION = 1

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

_PROBE_NAME = re.compile(r"^model\.ckpt-[a-z]*-*[0-9]+\.[^.]+$")


def is_kaiwu_train_bundle(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("format") == KAIWU_TRAIN_FORMAT
        and int(value.get("schema_version", 0)) == KAIWU_TRAIN_SCHEMA_VERSION
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
        for candidate in visual_command_checkpoint_candidates(path, model_id)
        if os.path.isfile(candidate)
    ]
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
    bundle: dict[str, Any], model_id: str | int
) -> dict[str, Any]:
    """Require a camera-eval bundle to identify the platform-selected model.

    Filename filtering is necessary but not sufficient: a manually copied or
    stale file can have a matching filename while carrying another model's
    payload.  ``platform_model_id`` is the primary identity.  Older training
    bundles may instead retain it under ``lineage.platform_model_id``.  When
    both are present, they must each agree with the requested ID; accepting a
    contradictory bundle would make an evaluation result non-attributable.
    """
    if not is_kaiwu_train_bundle(bundle):
        raise ValueError(
            "Camera eval requires a kaiwu_train_v1 checkpoint, got "
            f"format={bundle.get('format')!r}, "
            f"schema={bundle.get('schema_version')!r}"
        )

    requested_id = str(model_id)
    lineage = bundle.get("lineage", {})
    if not isinstance(lineage, dict):
        raise ValueError("Camera eval checkpoint lineage must be a dict")

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
    if mismatches:
        raise ValueError(
            "Camera eval checkpoint identity mismatch: "
            f"requested={requested_id}, identities={present_ids}"
        )
    if all(value in (None, "") for value in present_ids.values()):
        raise ValueError(
            "Camera eval checkpoint has no platform identity: "
            f"requested={requested_id}, identities={present_ids}"
        )

    return {
        "requested_id": requested_id,
        "bundle_id": bundle_id,
        "lineage_id": lineage_id,
        "lineage": lineage,
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
    _, meta = high_level_parts(bundle)
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


def _digest_state_dict(hasher, name: str, state: dict[str, Any]) -> None:
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
) -> dict[str, Any]:
    """Nav-eval hard-stop chain (identity + spec + high_level + digest).

    Does NOT modify validate_visual_eval_bundle_identity (the lbc_loco eval
    path also calls it); composes it instead. Any failure raises before the
    first inference.
    """
    identity = validate_visual_eval_bundle_identity(bundle, model_id)
    validate_low_level_spec(bundle, expected_low_level_spec)
    validate_high_level_spec(bundle, expected_high_level)

    lineage = identity["lineage"] or {}
    stored_digest = lineage.get("low_level_state_digest")
    computed_digest = compute_low_level_state_digest(bundle)
    if not stored_digest:
        raise ValueError(
            "nav eval checkpoint has no lineage.low_level_state_digest"
        )
    if str(stored_digest) != computed_digest:
        raise ValueError(
            "nav eval low-level digest mismatch: "
            f"lineage={stored_digest} computed={computed_digest} "
            "(frozen low level was modified — refusing to evaluate)"
        )
    identity["low_level_state_digest"] = computed_digest
    identity["high_level_present"] = True
    return identity
