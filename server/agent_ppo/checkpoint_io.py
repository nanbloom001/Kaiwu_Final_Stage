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
