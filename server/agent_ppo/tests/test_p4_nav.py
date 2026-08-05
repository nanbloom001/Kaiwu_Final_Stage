#!/usr/bin/env python3
"""Focused contracts for the P4 Track robustness stage."""

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest
import toml
import torch
from torch import nn

from agent_ppo.algorithm.algorithm_p4_nav_ppo import AlgorithmP4NavPPO
from agent_ppo.checkpoint_io import (
    P4_NAV_PHASE_LABELS,
    p4_nav_checkpoint_candidates,
    p4_nav_training_candidates,
    validate_p4_eval_bundle,
    validate_probe_filename,
)
from agent_ppo.feature import p2_contract, p3_contract, p4_contract
from agent_ppo.feature.p2_response_buffer import P2ResponseAuxBuffer
from agent_ppo.feature.p4_camera import P4SharedCameraState
from agent_ppo.feature.p4_goal_belief import GoalBeliefChainV2
from agent_ppo.feature.p4_stuck import MotionWallStuckTracker
from agent_ppo.model.p2_high_level import (
    NavigationEncoder,
    NavigationSafetyHead,
    P2NavigationActor,
    P2NavigationCritic,
)
from agent_ppo.model.response_adapter import CommandResponseAdapter
from agent_ppo.model.vision_encoder import VisionEncoder
from agent_ppo.workflow.p2_nav_ppo_workflow import (
    _expected_critic_wire_dim,
    _terminal_safe_p4_critic_wire,
)


def _p4_algorithm(num_envs=1):
    low_actor = nn.Sequential(
        nn.Linear(77, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12),
    )
    return AlgorithmP4NavPPO(
        low_level_encoder=VisionEncoder(),
        low_level_actor=low_actor,
        navigation_encoder=NavigationEncoder(),
        safety_head=NavigationSafetyHead(),
        actor=P2NavigationActor(),
        critic=P2NavigationCritic(),
        response_adapter=CommandResponseAdapter(),
        response_buffer=P2ResponseAuxBuffer(num_envs, "cpu"),
        num_envs=num_envs,
        device="cpu",
        config={"p4_seed": 17, "num_learning_epochs": 4},
    )


def _p4_eval_algorithm(num_envs=1):
    low_actor = nn.Sequential(
        nn.Linear(77, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12),
    )
    return AlgorithmP4NavPPO(
        low_level_encoder=VisionEncoder(),
        low_level_actor=low_actor,
        navigation_encoder=NavigationEncoder(),
        safety_head=None,
        actor=P2NavigationActor(),
        critic=None,
        response_adapter=CommandResponseAdapter(),
        response_buffer=None,
        num_envs=num_envs,
        device="cpu",
        config={"p4_seed": 17},
        training=False,
    )


def _p4_training_wire(num_envs=1):
    return torch.zeros(num_envs, p4_contract.P4_PRIVILEGED_WIRE_DIM)


def test_p4_workflow_requires_507_training_wire_without_changing_p2_eval_wire():
    assert _expected_critic_wire_dim(is_p4=True) == 507
    assert _expected_critic_wire_dim(is_p4=False) == 385


def test_p4_mapper_ranges_dynamic_cap_and_stale_goal_wait():
    normalized = torch.tensor(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    mapped = p4_contract.map_normalized_action(
        normalized, torch.tensor((0.35, 0.95)), torch.tensor((1.0, 1.0))
    )
    assert torch.allclose(mapped[0], torch.tensor((0.0, -0.30, -0.90)))
    assert torch.allclose(mapped[1], torch.tensor((0.95, 0.30, 0.90)))
    stale = p4_contract.map_normalized_action(
        torch.ones(1, 3), torch.zeros(1), torch.zeros(1)
    )
    assert torch.allclose(stale, torch.tensor(((0.0, 0.0, 0.25),)))
    legacy_input = torch.tensor(((-0.7, 0.2, 0.9), (0.4, -0.5, -0.3)))
    assert torch.equal(
        p4_contract.map_normalized_action_legacy(legacy_input),
        p2_contract.map_normalized_action(legacy_input, hard_abs_vy=0.40),
    )


def test_p4_yaw_cancellation_and_proportional_cap():
    one_direction = torch.full((5, 2), 0.5)
    alternating = torch.tensor((0.5, -0.5, 0.5, -0.5, 0.5)).reshape(5, 1)
    assert torch.all(p4_contract.yaw_cancellation(one_direction) < 1.0e-5)
    assert p4_contract.yaw_cancellation(alternating).item() > 0.5
    raw = tuple(torch.tensor((-0.03,)) for _ in range(3))
    applied, scale = p4_contract.proportional_negative_cap(*raw)
    assert sum(item.item() for item in applied) == pytest.approx(-0.05)
    assert scale.item() == pytest.approx(5.0 / 9.0)


def test_goal_belief_propagates_with_feedback_and_resets_on_epoch():
    belief = GoalBeliefChainV2(1, "cpu", seed=7)
    goal = torch.tensor(((2.0, 0.0),))
    valid = torch.ones(1, dtype=torch.bool)
    epoch = torch.zeros(1, dtype=torch.long)
    belief.update(
        goal, torch.zeros(1, 3), velocity_valid=valid, dt_s=0.2,
        goal_epoch=epoch, deterministic=True,
    )
    before = belief.estimate.clone()
    belief.measurement_clock_s.zero_()
    belief.update(
        goal, torch.tensor(((0.5, 0.0, 0.0),)), velocity_valid=valid,
        dt_s=0.1, goal_epoch=epoch, deterministic=True,
    )
    assert belief.estimate[0, 0] == pytest.approx(before[0, 0].item() - 0.05)
    belief.update(
        goal, torch.zeros(1, 3), velocity_valid=valid, dt_s=0.2,
        goal_epoch=torch.ones(1, dtype=torch.long), deterministic=True,
    )
    assert belief.has_estimate.item()
    assert belief.goal_epoch.item() == 1


def test_goal_belief_does_not_propagate_invalid_feedback_history():
    belief = GoalBeliefChainV2(1, "cpu", seed=8)
    goal = torch.tensor(((2.0, 0.0),))
    valid = torch.ones(1, dtype=torch.bool)
    belief.update(
        goal, torch.zeros(1, 3), velocity_valid=valid, dt_s=0.2,
        goal_epoch=torch.zeros(1, dtype=torch.long), deterministic=True,
    )
    estimate_before = belief.estimate.clone()
    history_before = belief.history_xy.clone()
    belief.measurement_clock_s.zero_()
    belief.update(
        goal, torch.tensor(((1.0, 0.0, 0.5),)),
        velocity_valid=torch.zeros(1, dtype=torch.bool), dt_s=0.1,
        goal_epoch=torch.zeros(1, dtype=torch.long), deterministic=True,
    )
    assert torch.linalg.vector_norm(belief.estimate).item() == pytest.approx(
        torch.linalg.vector_norm(estimate_before).item()
    )
    assert torch.linalg.vector_norm(belief.history_xy).item() == pytest.approx(
        torch.linalg.vector_norm(history_before).item()
    )
    assert belief.last_diagnostics["goal_propagated"].item() == 1.0
    assert belief.process_variance_m2.item() > 0.0


def test_goal_fault_profile_blocks_long_dropout_and_disallowed_rows():
    belief = GoalBeliefChainV2(2, "cpu", seed=9)
    belief.set_fault_scale(1.0)
    belief.dropout_remaining_s.zero_()
    allowed = torch.tensor((True, False))
    # A long dt makes a start deterministic while still exercising duration
    # selection; medium mode may only create the 0.3-1.2 s short dropout.
    belief.update(
        torch.full((2, 2), 2.0), torch.zeros(2, 3),
        velocity_valid=torch.ones(2, dtype=torch.bool), dt_s=100.0,
        deterministic=False, fault_profile="medium",
        fault_allowed_mask=allowed,
    )
    assert 0.3 <= belief.dropout_remaining_s[0].item() <= 1.2
    assert belief.dropout_remaining_s[1].item() == 0.0


def test_goal_belief_noise_only_reject_rate_stays_below_five_percent():
    belief = GoalBeliefChainV2(64, "cpu", seed=10)
    goal = torch.tensor(((15.0, 2.0),)).expand(64, -1)
    rejected = 0.0
    measured = 0.0
    for frame in range(500):
        belief.update(
            goal,
            torch.zeros(64, 3),
            velocity_valid=torch.ones(64, dtype=torch.bool),
            dt_s=0.02,
            reset_mask=(torch.ones(64, dtype=torch.bool) if frame == 0 else None),
            deterministic=False,
            fault_profile="noise_only",
        )
        diagnostics = belief.last_diagnostics
        rejected += float(diagnostics["goal_measurement_rejected"].sum())
        measured += float(
            (
                diagnostics["goal_measurement_accepted"]
                + diagnostics["goal_measurement_clipped"]
                + diagnostics["goal_measurement_rejected"]
            ).sum()
        )
    assert rejected / max(measured, 1.0) < 0.05


def test_goal_belief_requires_five_consistent_measurements_to_reacquire():
    belief = GoalBeliefChainV2(1, "cpu", seed=12)
    valid = torch.ones(1, dtype=torch.bool)
    belief.update(
        torch.tensor(((2.0, 0.0),)),
        torch.zeros(1, 3),
        velocity_valid=valid,
        dt_s=0.2,
        deterministic=True,
    )
    alternative = torch.tensor(((-2.0, 0.0),))
    for _ in range(4):
        belief.update(
            alternative,
            torch.zeros(1, 3),
            velocity_valid=valid,
            dt_s=0.2,
            deterministic=True,
        )
    assert belief.reacquire_pending.item()
    assert belief.candidate_count.item() == 4
    belief.update(
        alternative,
        torch.zeros(1, 3),
        velocity_valid=valid,
        dt_s=0.2,
        deterministic=True,
    )
    assert not belief.reacquire_pending.item()
    assert belief.last_diagnostics["goal_measurement_accepted"].item() == 1.0
    assert belief.last_diagnostics["goal_reacquisition_time_s"].item() == pytest.approx(
        0.8, abs=0.21
    )


def test_shared_camera_has_distinct_delay_fault_masks_and_bounded_age():
    camera = P4SharedCameraState(6, "cpu", seed=11)
    clean = torch.ones(
        6, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
    )
    seen = torch.zeros(3, dtype=torch.bool)
    for _ in range(120):
        camera.begin_rollout(22_000.0, training=True)
        _, diagnostics = camera.process(
            clean,
            reset_mask=torch.zeros(6, dtype=torch.bool),
            session_effective_seconds=22_000.0,
            training=True,
        )
        seen |= torch.tensor(
            (
                bool(diagnostics["camera_delay_only"].any()),
                bool(diagnostics["camera_fault_only"].any()),
                bool(diagnostics["camera_fault_delay_overlap"].any()),
            )
        )
    assert seen.all(), seen
    assert diagnostics["camera_age_s"].max().item() <= 0.20
    assert 0.10 <= camera.near_clip.min().item()
    assert camera.near_clip.max().item() <= 0.25


def test_shared_camera_delay_never_replays_an_older_frame():
    camera = P4SharedCameraState(1, "cpu", seed=13)
    clean = torch.ones(
        1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
    )
    camera.sequence_kind.fill_(2)
    camera.fault_kind.fill_(2)
    camera.fault_remaining_frames.fill_(10_000)
    camera.delay_enabled.fill_(True)
    camera.pixel_fault_enabled.fill_(False)
    camera.delay_s.fill_(0.15)
    frame_ids = []
    for _ in range(100):
        _, diagnostics = camera.process(
            clean,
            reset_mask=torch.zeros(1, dtype=torch.bool),
            session_effective_seconds=10_000.0,
            training=True,
        )
        frame_ids.append(int(diagnostics["camera_frame_id"].item()))
    assert all(
        current >= previous
        for previous, current in zip(frame_ids, frame_ids[1:])
    )


def test_shared_camera_exact_resume_does_not_consume_restored_rng():
    camera = P4SharedCameraState(4, "cpu", seed=17)
    state = camera.state_dict()
    restored = P4SharedCameraState(4, "cpu", seed=99)
    restored.load_state_dict(state)
    assert torch.equal(restored.generator.get_state(), state["generator_state"])


def test_shared_camera_near_clip_is_converted_from_meters_to_normalized_depth():
    camera = P4SharedCameraState(1, "cpu", seed=19, max_depth_m=5.0)
    camera.near_clip.fill_(0.10)
    camera.capture_accumulator.fill_(1.0)
    clean = torch.full(
        (1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1), 0.04
    )
    delivered, diagnostics = camera.process(
        clean,
        reset_mask=torch.zeros(1, dtype=torch.bool),
        session_effective_seconds=0.0,
        training=True,
    )
    assert diagnostics["near_clip_normalized"].item() == pytest.approx(0.02)
    assert bool((delivered > 0.0).all())
    assert diagnostics["camera_near_clip_added_hole_rate"].item() == 0.0
    assert diagnostics["camera_delivered_hole_rate"].item() == 0.0


def test_push_pulse_is_counted_once_for_reused_transport():
    algorithm = _p4_algorithm()
    wire = _p4_training_wire()
    wire[:, p2_contract.CRITIC_OBS_DIM + 26] = 123
    wire[:, p2_contract.PRIVILEGED_WIRE_DIM + p3_contract.PUSH_EVENT_FLAG_INDEX] = 1
    algorithm._split_transport(wire)
    algorithm._split_transport(wire)
    assert algorithm.push_epoch.item() == 1
    wire[:, p2_contract.CRITIC_OBS_DIM + 26] = 124
    algorithm._split_transport(wire)
    assert algorithm.push_epoch.item() == 2


def test_adapter_push_epoch_invalidates_all_overlapping_horizons():
    buffer = P2ResponseAuxBuffer(1, "cpu", capacity_steps=64)
    aux = torch.zeros(1, p2_contract.RESPONSE_AUX_DIM)
    aux[:, 9] = 1.0
    for step in range(51):
        epoch = torch.tensor((1 if step >= 5 else 0,))
        since = torch.tensor((max(0.0, (step - 5) * 0.02) if step >= 5 else 100.0,))
        buffer.append(
            aux,
            torch.zeros(1, dtype=torch.bool),
            push_epoch=epoch,
            seconds_since_push=since,
        )
    record = buffer._records[-1]
    assert not bool(record["horizon_mask"].any())
    assert bool((buffer.push_horizon_rejections > 0).all())


def test_adapter_record_rebuild_uses_p4_capability_profile():
    buffer = P2ResponseAuxBuffer(1, "cpu")
    contract = p4_contract.adapter_record_contract(
        low_level_digest="low", feedback_digest="feedback"
    )
    capability = buffer._capability_for_record(
        {"record_origin": "track", "record_contract": contract}
    )
    assert capability[3].item() == pytest.approx(1.0)
    assert capability[4].item() == pytest.approx(0.90)
    assert capability[7].item() == pytest.approx(0.30)
    assert contract["capability_digest"] == p4_contract.stable_digest(
        contract["response_capability_profile15"]
    )


def _legacy_parent_response_state(*, digest="a" * 64, num_envs=1):
    source = P2ResponseAuxBuffer(num_envs, "cpu", capacity_steps=64)
    source.set_low_level_version(digest, 7)
    for step in range(82):
        aux = torch.zeros(num_envs, p2_contract.RESPONSE_AUX_DIM)
        aux[:, 0] = 0.4
        aux[:, 3] = 0.35
        aux[:, 6] = 0.34
        aux[:, 9] = 1.0
        aux[:, 12] = 0.33
        aux[:, 15] = 0.01 * step
        source.append(aux, torch.zeros(num_envs, dtype=torch.bool))
    state = source.checkpoint_state()
    assert len(state["records"]) == 32
    assert all(record.get("record_contract") is None for record in state["records"])
    return state


def test_p4_migrates_structurally_valid_legacy_parent_records_and_samples_them():
    digest = "a" * 64
    buffer = P2ResponseAuxBuffer(1, "cpu", capacity_steps=64)
    buffer.set_low_level_version("b" * 64, 0)
    buffer.set_record_contract(
        p4_contract.adapter_record_contract(
            low_level_digest="b" * 64, feedback_digest="feedback"
        )
    )
    buffer.enable_p4_compatible_replay()
    restored = buffer.load_parent_completed_records(
        _legacy_parent_response_state(digest=digest)
    )
    assert restored == 32
    assert buffer.legacy_parent_records_migrated == 32
    assert not buffer.legacy_parent_migration_rejections
    assert all(
        record["record_contract"]["migration"]
        == "legacy_parent_structural_v1"
        for record in buffer._parent_records
    )
    batch = buffer.sample(batch_envs=1, generator=torch.Generator().manual_seed(3))
    assert batch is not None
    assert batch.metadata["p35_parent_batch_envs"] == 1
    assert batch.metadata["p4_current_batch_envs"] == 0
    assert bool(batch.horizon_mask[..., 1:].all())


@pytest.mark.parametrize("corruption", ("digest", "nonfinite", "shape"))
def test_p4_rejects_invalid_legacy_parent_records(corruption):
    state = _legacy_parent_response_state()
    for record in state["records"]:
        if corruption == "digest":
            record["low_level_digest"] = "other-low"
        elif corruption == "nonfinite":
            record["velocity"][0, 0, 0] = float("nan")
        else:
            record["pose"] = record["pose"][:, :2]
    buffer = P2ResponseAuxBuffer(1, "cpu", capacity_steps=64)
    buffer.set_low_level_version("b" * 64, 0)
    buffer.set_record_contract(
        p4_contract.adapter_record_contract(
            low_level_digest="b" * 64, feedback_digest="feedback"
        )
    )
    assert buffer.load_parent_completed_records(state) == 0
    assert sum(buffer.legacy_parent_migration_rejections.values()) == 32


def test_p4_missing_contract_current_records_remain_incompatible():
    buffer = P2ResponseAuxBuffer(1, "cpu")
    buffer.set_record_contract(
        p4_contract.adapter_record_contract(
            low_level_digest="low", feedback_digest="feedback"
        )
    )
    assert not buffer._contract_compatible(
        {"record_origin": "track", "record_contract": None}
    )
    assert buffer.compatibility_rejections["missing_contract"] == 1


def test_p4_maze_schedule_and_soft_cruise_are_explicit():
    warm = p4_contract.training_schedule(0.0)
    assert warm["phase"] == "mazeprobe"
    assert warm["actor_multiplier"] == pytest.approx(0.35)
    assert warm["navigation_multiplier"] == pytest.approx(0.25)
    assert warm["cruise_multiplier"] == pytest.approx(0.0)
    assert p4_contract.training_schedule(600.0)["phase"] == "mazefull"
    assert p4_contract.training_schedule(5_400.0)["phase"] == "mazefinal"
    visual = p4_contract.training_schedule(0.0, branch="visual_recovery")
    assert visual["actor_multiplier"] == pytest.approx(0.15)
    assert visual["safety_head_multiplier"] == pytest.approx(1.5)
    assert (
        p4_contract.training_contract()["safety_reward_ramp"]["version"]
        == p4_contract.SAFETY_REWARD_RAMP_VERSION
    )
    assert p4_contract.command_contract()["soft_cruise"]["preferred_vx"] == [0.60, 0.75]


def test_p4_auto_diagnostic_does_not_consume_training_clock():
    algorithm = _p4_algorithm()
    algorithm.maze_training_branch = "auto"
    algorithm.update_training_clocks(599.0)
    assert algorithm.current_phase == "mazediag"
    assert algorithm.diagnostic_elapsed_seconds == pytest.approx(599.0)
    assert algorithm.session_effective_seconds == pytest.approx(0.0)

    algorithm.update_training_clocks(600.0)
    assert algorithm.current_phase == "mazeprobe"
    assert algorithm._resolved_maze_training_branch == "visual_recovery"
    assert algorithm.session_effective_seconds == pytest.approx(0.0)

    algorithm.update_training_clocks(7_800.0)
    assert algorithm.session_effective_seconds == pytest.approx(7_200.0)


def test_p4_auto_branch_uses_accumulated_perception_metrics_not_last_tick():
    algorithm = _p4_algorithm()
    algorithm.maze_training_branch = "auto"
    algorithm.diagnostic_elapsed_seconds = p4_contract.DIAGNOSTIC_SECONDS
    algorithm._maze_diag_total.fill_(1_000.0)
    algorithm._maze_diag_valid.fill_(950.0)
    algorithm._maze_diag_wall_positive.fill_(200.0)
    algorithm._maze_diag_top1_total.fill_(200.0)
    algorithm._maze_diag_top1_correct.fill_(160.0)
    algorithm._maze_diag_risk_positive_hist[-1] = 200.0
    algorithm._maze_diag_risk_negative_hist[0] = 200.0
    algorithm._maze_diag_goal_risk_positive_hist[:] = 1.0
    algorithm._maze_diag_goal_risk_negative_hist[:] = 1.0
    algorithm._maze_diag_goal_top1_total.fill_(200.0)
    algorithm._maze_diag_goal_top1_correct.fill_(60.0)
    for index in range(5):
        algorithm._maze_diag_scene_confusion[index, index] = 50.0
        algorithm._maze_diag_goal_scene_confusion[index, 0] = 50.0
    algorithm._maze_diag_latent_cosine_sum.fill_(950.0)
    algorithm._maze_diag_latent_cosine_count.fill_(1_000.0)
    algorithm.last_tick_diagnostics = {"scanner_available": torch.zeros(1, 1)}
    assert algorithm._effective_maze_branch(0.0) == "actor_attack"


def test_p4_checkpoint_priority_prefers_new_maze_labels():
    candidates = p4_nav_checkpoint_candidates("/models", 42)
    labels = [Path(candidate).stem.split("-")[-2] for candidate in candidates]
    assert labels[:4] == ["mazefinal", "mazefull", "mazeprobe", "mazediag"]
    assert labels.index("mazefinal") < labels.index("pnavstable")


def test_p4_training_discovers_one_structurally_compatible_parent_before_p3():
    algorithm = _p4_algorithm()
    algorithm._initial_low_digest = algorithm._module_digest(
        (("vision", algorithm.low_level_encoder), ("actor", algorithm.low_level_actor))
    )
    algorithm.low_level_state_digest = algorithm._initial_low_digest
    algorithm._configure_adapter_contract()
    with tempfile.TemporaryDirectory() as directory:
        discovered = Path(directory) / "model.ckpt-mazefinal-77.pkl"
        algorithm.save_training_bundle(str(discovered), platform_model_id="77")
        candidates = p4_nav_training_candidates(
            directory, "99", parent_model_id="1013548"
        )
    assert str(discovered) in candidates
    p3_index = next(
        index
        for index, candidate in enumerate(candidates)
        if "stairfinal-1013548" in candidate
    )
    assert candidates.index(str(discovered)) < p3_index


def test_p4_configuration_and_monitor_route_are_explicit():
    root = Path(__file__).resolve().parents[1]
    config = toml.load(root / "conf/train_env_conf_track_p4_nav_ppo.toml")
    assert config["p4_nav_ppo"]["run_name"] == "p4maze2h-attack"
    assert config["p4_nav_ppo"]["maze_training_branch"] == "auto"
    assert config["env"]["num_envs"] == 128
    assert config["env"]["episode_length_s"] == 75.0
    assert config["terrain"]["track"]["track_length"] == 1
    assert config["terrain"]["track"]["sub_terrains"] == ["open_entry_maze"]
    assert config["p4_nav_ppo"]["parent_model_id"] == 1207698
    source = (root / "conf/monitor_builder.py").read_text()
    assert 'policy_entry == "p4_nav_ppo"' in source
    assert "p4_maze_perception" in source
    assert "p4_push_label_reject" in source
    assert "p4_adapter_compat" in source
    assert len(p4_contract.MONITOR_REQUIRED_METRICS) >= 100
    for metric in (
        "camera_memory_loss",
        "reward_soft_cruise",
        "head_correct_actor_wrong",
        "push_actual_delta_vx_mean",
        "adapter_compat_rejected_mismatch_response_profile15",
        "adapter_compat_migrated_legacy_parent_records",
        "reward_conservation_error",
    ):
        assert metric in p4_contract.MONITOR_REQUIRED_METRICS
    assert all(len(metric) <= 60 for metric in p4_contract.MONITOR_REQUIRED_METRICS)
    assert all(
        validate_probe_filename(f"model.ckpt-{phase}-1013600.pkl")
        for phase in P4_NAV_PHASE_LABELS
    )
    assert "mazediag" in P4_NAV_PHASE_LABELS


def test_p4_monitor_panels_obey_platform_line_limits():
    root = Path(__file__).resolve().parents[1]
    module = ast.parse((root / "conf/monitor_builder.py").read_text())
    shared_groups = ast.literal_eval(
        next(
            node.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "P2_MONITOR_GROUPS"
                for target in node.targets
            )
        )
    )
    build_p4 = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_p4_monitor"
    )
    groups_assignment = next(
        node
        for node in build_p4.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "groups"
            for target in node.targets
        )
    )
    groups = (*shared_groups, *ast.literal_eval(groups_assignment.value))
    panel_names = []
    for _, _, panels in groups:
        for _, panel_name, metrics in panels:
            panel_names.append(panel_name)
            assert len(metrics) <= 20, panel_name
            assert len(metrics) == len(set(metrics)), panel_name
    assert len(panel_names) == len(set(panel_names))


def test_p4_layered_eval_validator_requires_mapper_and_supports_both_modes():
    from agent_ppo.tests.test_p3_eval import _p3_fixture_bundle

    bundle = _p3_fixture_bundle(
        mode="track", stage_type="p4_nav_ppo", phase_label="pnavstable"
    )
    bundle["contracts"] = {
        "command": {"mapper_version": p4_contract.ACTION_MAPPER_VERSION}
    }
    standard = validate_p4_eval_bundle(bundle, mode="standard")
    track = validate_p4_eval_bundle(bundle, mode="track")
    assert standard["stage_type"] == "p4_nav_ppo"
    assert "high_level.actor" in track["loaded_modules"]
    bundle["phase_label"] = "renamed-but-structurally-compatible"
    renamed = validate_p4_eval_bundle(bundle, mode="track")
    assert renamed["phase_label"] == "renamed-but-structurally-compatible"
    assert not renamed["phase_label_known"]
    bundle["contracts"]["command"]["mapper_version"] = "legacy"
    with pytest.raises(ValueError, match="mapper"):
        validate_p4_eval_bundle(bundle, mode="track")


def test_p4_does_not_modify_platform_base_env():
    root = Path(__file__).resolve().parents[2]
    # Source-level implementation must stay outside the platform-overwritten
    # BaseEnv. In the development container ``agent_ppo`` resolves through the
    # platform-owned ``/workspace/code`` symlink while ``isaac_env`` remains
    # under the project root, so support both layouts without copying or
    # modifying the platform file.
    candidates = (
        root / "isaac_env/base_env.py",
        Path("/data/projects/legged_robot_competition_26/isaac_env/base_env.py"),
    )
    base_env = next((path for path in candidates if path.is_file()), None)
    assert base_env is not None, "platform BaseEnv source is unavailable"
    source = base_env.read_text()
    assert "GoalBeliefChainV2" not in source
    assert "p4_capability_action_mapper_v1" not in source


def test_p4_assembly_freezes_low_level_and_uses_two_tbptt16_sequences():
    algorithm = _p4_algorithm()
    assert algorithm.rollout.num_ticks == 32
    assert algorithm.rollout.sequence_length == 16
    assert all(
        not parameter.requires_grad
        for module in (algorithm.low_level_encoder, algorithm.low_level_actor)
        for parameter in module.parameters()
    )
    optimizer_ids = {
        id(parameter)
        for group in algorithm.actor_optimizer.param_groups
        for parameter in group["params"]
    }
    assert not any(
        id(parameter) in optimizer_ids
        for module in (algorithm.low_level_encoder, algorithm.low_level_actor)
        for parameter in module.parameters()
    )
    assert "camera_clean_live_action_mae" in algorithm._actor_auxiliary_metric_names()
    assert algorithm.cnn_unfrozen
    assert algorithm.rollout.store_depth
    algorithm.update_training_clocks(600.0)
    assert algorithm.cnn_unfrozen
    assert algorithm.rollout.store_depth


def test_p4_low_lstm_advances_each_tick_while_cnn_feature_is_reused():
    algorithm = _p4_algorithm()
    algorithm._camera_diagnostics = {"camera_frame_id": torch.tensor((1.0,))}
    parts = {
        "depth": torch.full((1, 180, 320, 1), 0.5),
        "proprio": torch.zeros(1, 45),
    }
    algorithm._low_level_frame(parts, None)
    hidden_before = tuple(item.clone() for item in algorithm.low_level_encoder.get_hidden_state())
    cached_before = algorithm._cached_low_cnn.clone()
    parts["proprio"][:, 0] = 0.5
    algorithm._low_level_frame(parts, None)
    hidden_after = algorithm.low_level_encoder.get_hidden_state()
    assert torch.equal(cached_before, algorithm._cached_low_cnn)
    assert any(not torch.equal(before, after) for before, after in zip(hidden_before, hidden_after))


def test_p4_one_tick_uses_57905_policy_and_507_training_wire():
    from agent_ppo.feature import nav_contract

    algorithm = _p4_algorithm()
    observation = torch.zeros(1, nav_contract.POLICY_OBS_DIM)
    observation[:, 301:305] = torch.tensor(((0.2, 0.0, 0.1, 1.0),))
    observation[:, 305:] = 0.5
    wire = _p4_training_wire()
    wire[:, nav_contract.CRITIC_GOAL3_START] = 0.2
    wire[:, p2_contract.CRITIC_OBS_DIM + 9] = 1.0
    wire[:, p2_contract.CRITIC_OBS_DIM + 24] = 1.0
    wire[:, p2_contract.CRITIC_OBS_DIM + 26] = 1.0
    wire[:, p3_contract.P3_PRIVILEGED_WIRE_DIM :][:, :2] = torch.tensor(
        ((15.0, 2.0),)
    )
    result, _, _ = algorithm.frame_begin(observation, wire, deterministic=False)
    assert result["is_tick"]
    assert result["actions"].shape == (1, 12)
    assert algorithm.pending_tick["target_cmd3"].shape == (1, 3)
    assert algorithm._p4_worker_extra[0, 0].item() == 15.0
    assert algorithm.goal_belief.last_diagnostics[
        "goal_map_distance_m"
    ].item() > 14.0


def test_p4_terminal_wire_keeps_old_raw_goal_and_stuck_diagnostics():
    live = torch.zeros(2, p4_contract.P4_PRIVILEGED_WIRE_DIM)
    live[:, p3_contract.P3_PRIVILEGED_WIRE_DIM :] = torch.tensor(
        ((99.0,) * p4_contract.P4_WORKER_EXTRA_DIM,) * 2
    )
    terminal = torch.arange(
        2 * p4_contract.P4_WORKER_EXTRA_DIM, dtype=torch.float32
    ).reshape(2, p4_contract.P4_WORKER_EXTRA_DIM)
    result = _terminal_safe_p4_critic_wire(
        live,
        terminal,
        torch.tensor((True, False)),
    )
    p4_tail = result[:, p3_contract.P3_PRIVILEGED_WIRE_DIM :]
    assert torch.equal(p4_tail[0], terminal[0])
    assert torch.equal(p4_tail[1], live[1, p3_contract.P3_PRIVILEGED_WIRE_DIM :])


def test_p4_warmup_freezes_actor_and_navigation_but_not_safety_head():
    algorithm = _p4_algorithm()
    algorithm._apply_training_schedule(0.0)
    groups = {group["name"]: group for group in algorithm.actor_optimizer.param_groups}
    for name, group in groups.items():
        assert all(parameter.requires_grad for parameter in group["params"])
    algorithm._apply_training_schedule(1_800.0)
    assert all(
        parameter.requires_grad
        for group in algorithm.actor_optimizer.param_groups
        for parameter in group["params"]
    )


def test_p4_soft_cruise_penalty_only_applies_on_clear_fresh_goal():
    algorithm = _p4_algorithm()
    policy = torch.tensor(((0.30, 0.0, 0.0), (0.30, 0.0, 0.0)))
    safe = torch.tensor(((0.2, 0.9, 0.2), (0.9, 0.2, 0.9)))
    penalty, diagnostics = p4_contract.soft_cruise_penalty(
        policy,
        safe,
        torch.ones(2, dtype=torch.bool),
        torch.ones(2),
        torch.zeros(2, dtype=torch.bool),
    )
    assert penalty[0].item() < 0.0
    assert abs(penalty[1].item()) < 1.0e-3
    assert diagnostics["soft_cruise_clear_factor"][0].item() > 0.0
    assert diagnostics["soft_cruise_clear_factor"][1].item() < 0.1
    algorithm._sample_speed_tiers(torch.ones(algorithm.num_envs, dtype=torch.bool))
    assert torch.allclose(
        algorithm.user_speed_cap,
        torch.full_like(algorithm.user_speed_cap, p4_contract.P4_MAX_VX),
    )


class _FakeTerminationManager:
    active_terms = ("nav_stuck_timeout",)

    def __init__(self, num_envs):
        self.cfg = SimpleNamespace(time_out=True, params={"max_stuck": 1})
        self.value = torch.zeros(num_envs, dtype=torch.bool)

    def get_term_cfg(self, _name):
        return self.cfg

    def set_term_cfg(self, _name, cfg):
        self.cfg = cfg

    def get_term(self, _name):
        return self.value


def test_p4_stuck_tracker_shadow_and_active_contract():
    manager = _FakeTerminationManager(1)
    env = SimpleNamespace(step_dt=0.02, termination_manager=manager)
    tracker = MotionWallStuckTracker(
        env,
        num_envs=1,
        device="cpu",
        config={"mode": "active", "confirmation_s": 0.04},
    )
    common = {
        "root_xy": torch.zeros(1, 2),
        "goal_distance": torch.ones(1),
        "collision_force": torch.full((1,), 40.0),
        "mapping_valid": torch.ones(1, dtype=torch.bool),
        "reset": torch.zeros(1, dtype=torch.bool),
        "terminal_reason": torch.zeros(1),
        "seconds_since_push": torch.full((1,), 10.0),
        "episode_age_s": torch.full((1,), 10.0),
    }
    tracker.update(**common)
    tracker.update(**common)
    diagnostics = tracker.update(**common)
    assert tracker.term_available and tracker.term_config_valid
    assert diagnostics[0, 4].item() == 1.0
    assert getattr(env, "_nav_motion_stuck").item() >= 2.0
    manager.value.fill_(True)
    assert tracker.termination_mask(torch.ones(1, dtype=torch.bool)).item()


def test_p4_stuck_tracker_retries_after_termination_manager_assembly():
    env = SimpleNamespace(step_dt=0.02, termination_manager=None)
    tracker = MotionWallStuckTracker(
        env,
        num_envs=1,
        device="cpu",
        config={"mode": "active", "confirmation_s": 0.04},
    )
    assert not tracker.term_available
    env.termination_manager = _FakeTerminationManager(1)
    tracker.update(
        root_xy=torch.zeros(1, 2),
        goal_distance=torch.ones(1),
        collision_force=torch.zeros(1),
        mapping_valid=torch.ones(1, dtype=torch.bool),
        reset=torch.zeros(1, dtype=torch.bool),
        terminal_reason=torch.zeros(1),
        seconds_since_push=torch.full((1,), 10.0),
        episode_age_s=torch.full((1,), 10.0),
    )
    assert tracker.term_available
    assert tracker.term_config_valid
    assert env.termination_manager.cfg.params["max_stuck"] == 2


def test_p4_shadow_stuck_event_and_saved_time_are_reported_once():
    env = SimpleNamespace(step_dt=0.02, termination_manager=None)
    tracker = MotionWallStuckTracker(
        env,
        num_envs=1,
        device="cpu",
        config={"mode": "shadow", "confirmation_s": 0.04},
    )
    common = {
        "root_xy": torch.zeros(1, 2),
        "goal_distance": torch.ones(1),
        "collision_force": torch.full((1,), 40.0),
        "mapping_valid": torch.ones(1, dtype=torch.bool),
        "reset": torch.zeros(1, dtype=torch.bool),
        "terminal_reason": torch.zeros(1),
        "seconds_since_push": torch.full((1,), 10.0),
        "episode_age_s": torch.full((1,), 10.0),
    }
    events = []
    saved = []
    for _ in range(8):
        diagnostics = tracker.update(**common)
        events.append(diagnostics[0, 4].item())
        saved.append(diagnostics[0, 8].item())
    assert sum(events) == 1.0
    assert sum(value > 0.0 for value in saved) == 1
    assert not tracker.termination_mask(torch.ones(1, dtype=torch.bool)).item()


def test_p4_adapter_replay_ratios_use_mutually_exclusive_pool_counts():
    latest, recent, parent, track, total = AlgorithmP4NavPPO._adapter_replay_counts(
        {
            "p4_current_batch_envs": 16,
            "p35_parent_batch_envs": 8,
            "earlier_lineage_batch_envs": 8,
            "track_batch_envs": 8,
            "parent_batch_envs": 0,
        }
    )
    assert (latest, recent, parent, track, total) == (16.0, 8.0, 8.0, 24.0, 32.0)
    assert latest / total + recent / total + parent / total == pytest.approx(1.0)


def test_p4_stuck_terminal_is_single_penalty_and_suppresses_duplicate_safety():
    algorithm = _p4_algorithm()
    algorithm.pending_tick = {"target_cmd3": torch.zeros(1, 3)}
    components = {
        "tracking": torch.full((1,), -0.1),
        "body_collision": torch.full((1,), -0.2),
        "predictive_collision_risk": torch.full((1,), -0.02),
        "missed_safe_direction": torch.full((1,), -0.01),
        "frontier_stagnation": torch.full((1,), -0.5),
    }
    result = algorithm._override_reward_components(
        components,
        reward_exec_cmd=torch.zeros(1, 3),
        reward_source_aux=torch.zeros(1, p2_contract.WORKER_AUX_DIM),
        terminal=torch.ones(1, dtype=torch.bool),
        reason=torch.full((1,), 4, dtype=torch.long),
    )
    assert result["stuck_reset"].item() == pytest.approx(-6.0)
    for name in (
        "body_collision",
        "predictive_collision_risk",
        "missed_safe_direction",
        "frontier_stagnation",
    ):
        assert result[name].item() == 0.0


def test_p4_risk_deceleration_is_measured_after_the_risk_event():
    algorithm = _p4_algorithm()
    algorithm.pending_tick = {
        "safe3": torch.tensor([[0.8, 0.2, 0.8]]),
        "safety_valid": torch.ones(1, 1),
    }
    algorithm._last_goal_freshness.fill_(1.0)
    algorithm.goal_belief.last_diagnostics = {
        "goal_innovation_d2": torch.zeros(1),
        "goal_age_s": torch.zeros(1),
    }
    algorithm._last_policy_command[0] = torch.tensor([0.6, 0.0, 0.0])
    algorithm._last_limited_command.copy_(algorithm._last_policy_command)
    first = algorithm._extra_tick_diagnostics()
    assert first["risk_decel_policy_vx"].item() == 0.0
    assert first["risk_no_deceleration"].item() == 0.0

    algorithm._last_policy_command[0, 0] = 0.4
    algorithm._last_limited_command[0, 0] = 0.4
    second = algorithm._extra_tick_diagnostics()
    assert second["risk_decel_policy_vx"].item() == 1.0
    assert second["risk_no_deceleration"].item() == 0.0


def test_p4_exact_resume_round_trip_restores_stage_rng_and_clock():
    algorithm = _p4_algorithm()
    algorithm.maze_training_branch = "auto"
    algorithm._resolved_maze_training_branch = "actor_attack"
    algorithm.diagnostic_elapsed_seconds = p4_contract.DIAGNOSTIC_SECONDS
    algorithm._initial_low_digest = algorithm._module_digest(
        (("vision", algorithm.low_level_encoder), ("actor", algorithm.low_level_actor))
    )
    algorithm.low_level_state_digest = algorithm._initial_low_digest
    algorithm._configure_adapter_contract()
    algorithm._diagnostic_nav_risk_probe.weight.data.fill_(0.125)
    algorithm._maze_diag_goal_top1_total.fill_(37.0)
    algorithm.update_training_clocks(600.0)
    algorithm.update_training_clocks(15_000.0)
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "model.ckpt-pnavfull-42.pkl")
        algorithm.save_training_bundle(path, platform_model_id="42")
        resumed = _p4_algorithm()
        mode = resumed.load_bundle(path, platform_model_id="42")
    assert mode == "p4_exact_resume_history_reset"
    assert resumed.session_effective_seconds == pytest.approx(14_400.0)
    assert resumed.session_wall_seconds == pytest.approx(15_000.0)
    assert resumed._resolved_maze_training_branch == "actor_attack"
    assert resumed.current_phase == "mazefinal"
    assert resumed._initial_low_digest == algorithm._initial_low_digest
    assert resumed.speed_generator.get_state().equal(algorithm.speed_generator.get_state())
    assert torch.equal(
        resumed._diagnostic_nav_risk_probe.weight,
        algorithm._diagnostic_nav_risk_probe.weight,
    )
    assert resumed._maze_diag_goal_top1_total.item() == pytest.approx(37.0)


def test_p4_maze_diagnostic_probe_trains_inside_finish_tick_no_grad_context():
    algorithm = _p4_algorithm(num_envs=5)
    algorithm.maze_training_branch = "auto"
    algorithm._resolved_maze_training_branch = None
    algorithm.diagnostic_elapsed_seconds = 0.0
    safe3 = torch.tensor(
        [
            [0.90, 0.20, 0.10],
            [0.10, 0.90, 0.20],
            [0.20, 0.10, 0.90],
            [0.80, 0.75, 0.10],
            [0.10, 0.15, 0.20],
        ]
    )
    valid = torch.ones(5, dtype=torch.bool)
    nav_feat = torch.randn(5, 32)
    goal4 = torch.randn(5, 4)
    before = algorithm._diagnostic_nav_risk_probe.weight.detach().clone()

    with torch.no_grad():
        algorithm._accumulate_maze_diagnostic(safe3, valid, nav_feat, goal4)

    assert not torch.equal(
        algorithm._diagnostic_nav_risk_probe.weight.detach(), before
    )
    assert all(
        parameter.grad is None
        for group in algorithm._diagnostic_probe_optimizer.param_groups
        for parameter in group["params"]
    )


def test_p4_exact_resume_rejects_stuck_reset_contract_drift():
    algorithm = _p4_algorithm()
    algorithm._initial_low_digest = algorithm._module_digest(
        (("vision", algorithm.low_level_encoder), ("actor", algorithm.low_level_actor))
    )
    algorithm.low_level_state_digest = algorithm._initial_low_digest
    algorithm._configure_adapter_contract()
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "model.ckpt-pnavwarm-42.pkl")
        algorithm.save_training_bundle(path, platform_model_id="42")
        resumed = _p4_algorithm()
        resumed.stuck_reset_contract = p4_contract.normalize_stuck_reset_contract(
            {"mode": "active", "confirmation_s": 12.0}
        )
        with pytest.raises(ValueError, match="training contract mismatch"):
            resumed.load_bundle(path, platform_model_id="42")


def test_p4_previous_contract_loads_as_maze_warm_start():
    algorithm = _p4_algorithm()
    algorithm._initial_low_digest = algorithm._module_digest(
        (("vision", algorithm.low_level_encoder), ("actor", algorithm.low_level_actor))
    )
    algorithm.low_level_state_digest = algorithm._initial_low_digest
    algorithm._configure_adapter_contract()
    algorithm.update_training_clocks(28_800.0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.ckpt-pnavstable-42.pkl"
        algorithm.save_training_bundle(str(path), platform_model_id="42")
        payload = torch.load(path, weights_only=False, map_location="cpu")
        payload["phase_label"] = "pnavstable"
        payload["contracts"]["training"]["version"] = "p4_nav_robust_v3_smooth_safety_ramp"
        payload["training_states"]["high_level"]["optimizer_phase"] = "pnavstable"
        incompatible_rng = torch.arange(16, dtype=torch.uint8)
        for key in (
            "shuffle_rng_state",
            "action_rng_state",
            "vy_action_rng_state",
            "neutral_rng_state",
        ):
            payload["training_states"]["high_level"][key] = incompatible_rng.clone()
        payload["training_states"]["response_adapter"]["rng_state"] = (
            incompatible_rng.clone()
        )
        torch.save(payload, path)
        resumed = _p4_algorithm()
        resumed.maze_training_branch = "auto"
        resumed._diagnostic_nav_risk_probe.weight.data.fill_(1.0)
        expected_rng = {
            "shuffle": resumed.ppo_generator.get_state().clone(),
            "action": resumed.action_generator.get_state().clone(),
            "vy_action": resumed.vy_action_generator.get_state().clone(),
            "neutral": resumed.neutral_generator.get_state().clone(),
            "adapter": resumed.adapter_generator.get_state().clone(),
        }
        mode = resumed.load_bundle(str(path), platform_model_id="42")
    assert mode == "p4_maze_warm_start"
    assert resumed.session_effective_seconds == 0.0
    assert resumed.session_wall_seconds == 0.0
    assert resumed._resolved_maze_training_branch is None
    assert resumed.current_phase == "mazediag"
    assert torch.count_nonzero(resumed._diagnostic_nav_risk_probe.weight) == 0
    assert resumed.ppo_generator.get_state().equal(expected_rng["shuffle"])
    assert resumed.action_generator.get_state().equal(expected_rng["action"])
    assert resumed.vy_action_generator.get_state().equal(expected_rng["vy_action"])
    assert resumed.neutral_generator.get_state().equal(expected_rng["neutral"])
    assert resumed.adapter_generator.get_state().equal(expected_rng["adapter"])
    assert resumed.optimizer_migration_report["p4_warm_start_rng"][
        "action_rng_state"
    ]["status"] == "fresh_seed"


def test_p4_runtime_stuck_reset_contract_controls_reward_and_metadata():
    algorithm = _p4_algorithm()
    algorithm.stuck_reset_contract = p4_contract.normalize_stuck_reset_contract(
        {"mode": "active", "confirmation_s": 12.0, "terminal_penalty": -4.5}
    )
    algorithm.pending_tick = {"target_cmd3": torch.zeros(1, 3)}
    components = {
        "tracking": torch.zeros(1),
        "body_collision": torch.zeros(1),
        "predictive_collision_risk": torch.zeros(1),
        "missed_safe_direction": torch.zeros(1),
        "frontier_stagnation": torch.zeros(1),
    }
    result = algorithm._override_reward_components(
        components,
        reward_exec_cmd=torch.zeros(1, 3),
        reward_source_aux=torch.zeros(1, p2_contract.WORKER_AUX_DIM),
        terminal=torch.ones(1, dtype=torch.bool),
        reason=torch.full((1,), 4, dtype=torch.long),
    )
    metadata = p4_contract.contract_metadata(algorithm.stuck_reset_contract)
    assert result["stuck_reset"].item() == pytest.approx(-4.5)
    assert metadata["training"]["stuck_reset"]["mode"] == "active"
    assert metadata["training"]["stuck_reset"]["confirmation_s"] == 12.0
    assert metadata["reward"]["new_terms"]["confirmed_wall_stuck_reset"] == -4.5


def test_p4_track_eval_loads_only_inference_modules_from_new_checkpoint():
    algorithm = _p4_algorithm()
    algorithm._initial_low_digest = algorithm._module_digest(
        (("vision", algorithm.low_level_encoder), ("actor", algorithm.low_level_actor))
    )
    algorithm.low_level_state_digest = algorithm._initial_low_digest
    algorithm.update_training_clocks(28_800.0)
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "model.ckpt-pnavstable-42.pkl")
        algorithm.save_training_bundle(path, platform_model_id="42")
        evaluation = _p4_eval_algorithm()
        mode = evaluation.load_evaluation_bundle(path, platform_model_id="42")
    assert mode == "evaluate_full_modules_only"
    assert evaluation.actor_optimizer is None
    assert evaluation.critic is None
    assert evaluation.safety_head is None
