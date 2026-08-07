"""Focused P4 Standard/Track evaluation contract checks."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import types

import pytest
import torch

from agent_ppo.checkpoint_io import (
    P3_EVAL_LOW_LEVEL_SPEC,
    P4_INSTANT_COMMAND_R4_INPUTFIX_VERSION,
    p4_eval_selection_metadata,
    p4_nav_eval_candidates,
    validate_p4_eval_bundle,
)
from agent_ppo.feature import p4_contract


class _Logger:
    def warning(self, *_args, **_kwargs):
        pass


def _command_contract() -> dict:
    return p4_contract.command_contract("maze_instant_repair2h")


def _leaf(spec: dict) -> dict:
    return {
        "class_name": spec["class_name"],
        "spec": spec["spec"],
        "state_dict": {"weight": torch.zeros(2)},
    }


def _p4_bundle(
    *,
    command: dict | None = None,
    training_profile: str = "maze_instant_repair2h",
) -> dict:
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
            "training": p4_contract.training_contract(
                training_profile=training_profile
            ),
            "command": copy.deepcopy(
                (
                    p4_contract.command_contract(training_profile)
                    if command is None
                    else command
                )
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
    assert result["training_profile"] == "maze_instant_repair2h"
    assert result["command_digest"]
    assert result["loaded_modules"] == [
        "high_level.actor",
        "high_level.navigation_encoder",
        "high_level.response_adapter",
        "low_level.actor",
        "low_level.locomotion_encoder",
    ]


def test_track_accepts_each_canonical_eval_profile_and_rejects_drift():
    for profile in (
        "maze_instant_repair2h",
        "full_track",
        "maze_credit_repair",
        "maze_closed_loop_v3",
        "maze_instant_command_r4",
    ):
        result = validate_p4_eval_bundle(
            _p4_bundle(training_profile=profile), mode="track"
        )
        assert result["training_profile"] == profile

    stale = _command_contract()
    stale["version"] = "p4_maze_instant_command_r4"
    with pytest.raises(ValueError, match="version"):
        validate_p4_eval_bundle(_p4_bundle(command=stale), mode="track")

    slew = _command_contract()
    slew["slew_rate"] = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="slew"):
        validate_p4_eval_bundle(_p4_bundle(command=slew), mode="track")



def test_track_requires_canonical_training_profile_metadata():
    bundle = _p4_bundle()
    del bundle["contracts"]["training"]
    with pytest.raises(ValueError, match="training_profile"):
        validate_p4_eval_bundle(bundle, mode="track")


def test_standard_uses_low_level_only_and_ignores_high_level_command_version():
    stale = _command_contract()
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
    assert next(path for path in preferred if Path(path).is_file()) == str(exact)


def test_discovery_is_mode_sensitive_and_rejects_multiple_compatible_bundles(tmp_path: Path):
    stale = _command_contract()
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
    bundle["platform_model_id"] = "payload-123"
    torch.save(bundle, path)

    metadata = p4_eval_selection_metadata(
        str(path), bundle, requested_model_id="1416926"
    )
    assert metadata["payload_id"] == "payload-123"
    assert metadata["filename_model_id"] == "999999"
    assert metadata["phase_label"] == "repairstable"
    assert metadata["training_profile"] == "maze_instant_repair2h"
    assert metadata["requested_model_id"] == "1416926"
    assert metadata["selected_path"] == str(path.resolve())
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["command_digest"]
    assert P4_INSTANT_COMMAND_R4_INPUTFIX_VERSION in str(
        bundle["contracts"]["command"]
    )


def test_track_rejects_drift_in_any_canonical_command_field():
    changed = _command_contract()
    changed["instant_physical_change_rate_per_s"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="instant_physical_change_rate_per_s"):
        validate_p4_eval_bundle(_p4_bundle(command=changed), mode="track")


@pytest.mark.parametrize("standard_eval", (True, False))
def test_p4_eval_lifecycle_never_writes_training_checkpoint(
    tmp_path: Path, standard_eval: bool, monkeypatch
):
    root_module = types.ModuleType("kaiwudrl")
    interface_module = types.ModuleType("kaiwudrl.interface")
    agent_module = types.ModuleType("kaiwudrl.interface.agent")
    agent_module.BaseAgent = object
    interface_module.agent = agent_module
    root_module.interface = interface_module
    monkeypatch.setitem(sys.modules, "kaiwudrl", root_module)
    monkeypatch.setitem(sys.modules, "kaiwudrl.interface", interface_module)
    monkeypatch.setitem(sys.modules, "kaiwudrl.interface.agent", agent_module)
    tools_module = types.ModuleType("tools")
    tools_module.__path__ = []
    validate_module = types.ModuleType("tools.train_env_conf_validate")
    validate_module.check_usr_conf = lambda *_args, **_kwargs: None
    tools_module.train_env_conf_validate = validate_module
    monkeypatch.setitem(sys.modules, "tools", tools_module)
    monkeypatch.setitem(
        sys.modules, "tools.train_env_conf_validate", validate_module
    )
    from agent_ppo.agent import Agent

    agent = object.__new__(Agent)
    agent.is_p3_eval = False
    agent.is_p4_eval = True
    agent.is_p4_standard_eval = standard_eval
    agent.is_p4_track_eval = not standard_eval
    agent.logger = _Logger()

    assert agent.save_model(str(tmp_path), id="0") is None
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(RuntimeError, match="evaluation assembly"):
        agent.save_model(str(tmp_path), id="1")
    assert list(tmp_path.iterdir()) == []
