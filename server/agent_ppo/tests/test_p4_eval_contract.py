"""Focused P4 Standard/Track evaluation contract checks."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from agent_ppo.checkpoint_io import (
    P3_EVAL_LOW_LEVEL_SPEC,
    P4_INSTANT_COMMAND_R4_INPUTFIX_CONTRACT,
    P4_INSTANT_COMMAND_R4_INPUTFIX_VERSION,
    p4_eval_selection_metadata,
    p4_nav_eval_candidates,
    validate_p4_eval_bundle,
)


def _leaf(spec: dict) -> dict:
    return {
        "class_name": spec["class_name"],
        "spec": spec["spec"],
        "state_dict": {"weight": torch.zeros(2)},
    }


def _p4_bundle(*, command: dict | None = None) -> dict:
    from agent_ppo.model.p2_high_level import (
        navigation_actor_spec,
        navigation_encoder_spec,
    )
    from agent_ppo.model.response_adapter import response_adapter_spec

    low = {
        "contract_version": "low_level_v2",
        "locomotion_encoder": _leaf(P3_EVAL_LOW_LEVEL_SPEC["locomotion_encoder"]),
        "actor": _leaf(P3_EVAL_LOW_LEVEL_SPEC["actor"]),
    }
    high_specs = {
        "navigation_encoder": {
            "class_name": "NavigationEncoder",
            "spec": navigation_encoder_spec(),
        },
        "actor": {
            "class_name": "P2NavigationActor",
            "spec": navigation_actor_spec(),
        },
        "response_adapter": {
            "class_name": "CommandResponseAdapter",
            "spec": response_adapter_spec(),
        },
    }
    return {
        "format": "kaiwu_train_v1",
        "schema_version": 2,
        "stage_type": "p4_nav_ppo",
        "phase_label": "repairstable",
        "model_spec": {
            "proprio_dim": 45,
            "scan_dim": 256,
            "latent_dim": 32,
            "action_dim": 12,
            "goal_dim": 0,
            "depth_height": 180,
            "depth_width": 320,
            "depth_channels": 1,
        },
        "contracts": {
            "command": copy.deepcopy(
                P4_INSTANT_COMMAND_R4_INPUTFIX_CONTRACT if command is None else command
            )
        },
        "modules": {
            "low_level": low,
            "high_level": {
                "contract_version": "high_level_continuous_v2",
                "component_status": "complete",
                **{name: _leaf(spec) for name, spec in high_specs.items()},
            },
        },
    }


def test_track_accepts_current_inputfix_contract_and_reports_digest():
    result = validate_p4_eval_bundle(_p4_bundle(), mode="track")
    assert result["phase_label"] == "repairstable"
    assert result["phase_label_known"] is True
    assert result["command_digest"]
    assert result["loaded_modules"] == [
        "high_level.actor",
        "high_level.navigation_encoder",
        "high_level.response_adapter",
        "low_level.actor",
        "low_level.locomotion_encoder",
    ]


def test_track_strictly_rejects_stale_or_slew_inputfix_contract():
    stale = copy.deepcopy(P4_INSTANT_COMMAND_R4_INPUTFIX_CONTRACT)
    stale["version"] = "p4_maze_instant_command_r4"
    with pytest.raises(ValueError, match="version"):
        validate_p4_eval_bundle(_p4_bundle(command=stale), mode="track")

    slew = copy.deepcopy(P4_INSTANT_COMMAND_R4_INPUTFIX_CONTRACT)
    slew["slew_rate"] = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="slew"):
        validate_p4_eval_bundle(_p4_bundle(command=slew), mode="track")


def test_standard_uses_low_level_only_and_ignores_high_level_command_version():
    stale = copy.deepcopy(P4_INSTANT_COMMAND_R4_INPUTFIX_CONTRACT)
    stale["version"] = "future_track_command_contract"
    result = validate_p4_eval_bundle(_p4_bundle(command=stale), mode="standard")
    assert result["loaded_modules"] == [
        "low_level.actor",
        "low_level.locomotion_encoder",
    ]
    assert result["command_digest"]


def test_requested_id_is_preferred_then_unique_mode_compatible_discovery(tmp_path: Path):
    fallback = tmp_path / "model.ckpt-repairstable-999999.pkl"
    torch.save(_p4_bundle(), fallback)

    discovered = p4_nav_eval_candidates(str(tmp_path), "1416926", mode="track")
    assert discovered[-1] == str(fallback)

    exact = tmp_path / "model.ckpt-repairtrain-1416926.pkl"
    torch.save({"not": "validated until selected"}, exact)
    preferred = p4_nav_eval_candidates(str(tmp_path), "1416926", mode="track")
    assert preferred[0] == str(exact)


def test_discovery_is_mode_sensitive_and_rejects_multiple_compatible_bundles(tmp_path: Path):
    stale = copy.deepcopy(P4_INSTANT_COMMAND_R4_INPUTFIX_CONTRACT)
    stale["version"] = "future_track_command_contract"
    standard_only = tmp_path / "model.ckpt-repairstable-999999.pkl"
    torch.save(_p4_bundle(command=stale), standard_only)

    assert p4_nav_eval_candidates(str(tmp_path), "1416926", mode="standard")[-1] == str(
        standard_only
    )
    assert str(standard_only) not in p4_nav_eval_candidates(
        str(tmp_path), "1416926", mode="track"
    )

    second = tmp_path / "model.ckpt-repairtrain-888888.pkl"
    torch.save(_p4_bundle(), second)
    with pytest.raises(RuntimeError, match="ambiguous"):
        p4_nav_eval_candidates(str(tmp_path), "1416926", mode="standard")


def test_selection_metadata_has_filename_payload_id_phase_command_digest_and_sha(tmp_path: Path):
    path = tmp_path / "model.ckpt-repairstable-999999.pkl"
    bundle = _p4_bundle()
    torch.save(bundle, path)

    metadata = p4_eval_selection_metadata(
        str(path), bundle, requested_model_id="1416926"
    )
    assert metadata["payload_id"] == "999999"
    assert metadata["phase_label"] == "repairstable"
    assert metadata["requested_model_id"] == "1416926"
    assert metadata["selected_path"] == str(path.resolve())
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["command_digest"]
    assert P4_INSTANT_COMMAND_R4_INPUTFIX_VERSION in str(
        bundle["contracts"]["command"]
    )
