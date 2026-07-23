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
    discovered = sorted(
        glob.glob(os.path.join(path, f"model.ckpt-*-{model_id}.*")),
        reverse=True,
    )
    result: list[str] = []
    for candidate in [*preferred, *discovered]:
        if candidate not in result:
            result.append(candidate)
    return result


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
