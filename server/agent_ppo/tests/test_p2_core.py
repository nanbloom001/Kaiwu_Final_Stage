import ast
import math
from pathlib import Path
import re
from types import SimpleNamespace
import tomllib

import pytest
import torch
import torch.nn as nn

from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
from agent_ppo.checkpoint_io import (
    p2_nav_checkpoint_candidates,
    p2_nav_evaluation_candidates,
    p2_nav_training_candidates,
)
from agent_ppo.feature import nav_contract, p2_contract
from agent_ppo.feature.p2_command_controller import P2CommandController
from agent_ppo.feature.p2_curriculum_probe import (
    P2CurriculumAccumulator,
    P2TrackCurriculumProbe,
)
from agent_ppo.feature.p2_rollout import P2RolloutStorage
from agent_ppo.feature.p2_gait import P2GaitBaseline, P2GaitWindowProbe
from agent_ppo.feature.p2_response_buffer import (
    P2ResponseAuxBuffer,
    patch_owned_commands,
    split_p2_transport,
)
from agent_ppo.feature.p15_contract import CAPABILITY_PROFILE15 as P15_CAPABILITY_PROFILE15
from agent_ppo.workflow.p2_nav_ppo_workflow import (
    _curriculum_metrics,
    _final_save,
    _frame_done_masks,
    _monitor_put,
    _resolve_terminal_outcome,
    _terminal_outcome_count,
    _terminal_safe_segment,
    _terminal_safe_tensor,
    _tick_diagnostic_values,
)
from agent_ppo.model.p2_high_level import (
    NavigationEncoder,
    NavigationSafetyHead,
    P2NavigationActor,
    P2NavigationCritic,
    assemble_actor_input,
    assemble_critic_input,
    squashed_log_prob,
)
from agent_ppo.model.simple_cnn import SimpleCNN
from agent_ppo.model.response_adapter import CommandResponseAdapter
from agent_ppo.model.vision_encoder import VisionEncoder
from agent_ppo.feature.p2_worker_bridge import (
    P2WorkerBridge,
    _merge_worker_terminal_returns,
    _track_segment_index,
    _termination_reason_codes,
    install_p2_terminal_return_bridge,
)
from agent_ppo.feature import p2_observation_process
from agent_ppo.feature.p2_observation_process import P2PolicyObservationProcess
from agent_ppo.feature.nav_observation_process import NavPolicyObservationProcess


def test_p2_production_config_locks_two_hour_safe_direction_contract():
    path = Path(__file__).parents[1] / "conf" / "train_env_conf_track_p2_nav_ppo.toml"
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    assert config["env"]["num_envs"] == 128
    assert config["terrain"]["curriculum"] is False
    assert config["terrain"]["num_cols"] == 20
    assert config["terrain"]["max_init_terrain_level"] == 1
    assert config["terrain"]["track"] == {
        "track_length": 3,
        "sub_terrains": [
            "pyramid_slope_inv",
            "pyramid_stairs_inv",
            "open_entry_maze",
        ],
        "num_parallel_tracks": 20,
    }
    assert config["rewards"]["track_lin_vel_xy"] == {
        "weight": 0.0,
        "params": {"std": 0.25, "command_name": "base_velocity"},
    }
    assert config["rewards"]["track_ang_vel_z"] == {
        "weight": 0.0,
        "params": {"std": 0.25, "command_name": "base_velocity"},
    }
    assert "sub_terrains_random" not in config["terrain"]["track"]
    assert config["rewards"]["flat_orientation"] == {"weight": -0.05}
    assert config["rewards"]["energy"] == {"weight": -5.0e-6}
    assert config["rewards"]["undesired_contacts"] == {
        "weight": 0.0,
        "params": {"threshold": 1},
    }
    assert set(config["rewards"]) == {
        "flat_orientation",
        "energy",
        "undesired_contacts",
        "track_lin_vel_xy",
        "track_ang_vel_z",
    }
    assert config["commands"]["ranges"]["lin_vel_y"] == [-0.4, 0.4]
    assert config["commands"]["limit"]["lin_vel_y"] == [-0.4, 0.4]
    assert config["p2_nav_ppo"]["run_name"] == "p2nav2hsafedir"
    assert config["p2_nav_ppo"]["parent_model_id"] == 291713
    assert config["p2_nav_ppo"]["load_mode"] == (
        "p2_safe_direction_continue_warm_start"
    )
    assert config["p2_nav_ppo"]["task_end_hours"] == 2.0
    assert config["p2_nav_ppo"]["slew_release_rate"] == [0.30, 0.60, 2.50]
    assert config["p2_nav_ppo"]["first_save_minutes"] == 5.0
    assert config["p2_nav_ppo"]["save_interval_minutes"] == 10.0
    assert config["p2_nav_ppo"]["gait_baseline"] == {
        "version": "p2_gait_parent_baseline_v2",
        "source_parent_model_id": 37953,
        "source_parent_label": "p15resp8h-r1_37953-F",
        "source": "fixed_versioned_parent_envelope",
        "window_seconds": 1.5,
        "calibration_seconds": 0.0,
        "prolonged_air_seconds": 0.60,
        "penalty_cap": 0.0,
        "duty_imbalance": 0.30,
        "swing_imbalance_s": 0.20,
        "prolonged_air_ratio": 0.10,
        "step_frequency_imbalance_hz": 1.50,
    }


def test_p2_monitor_uses_grouped_live_metrics_without_legacy_track_panels():
    path = Path(__file__).parents[1] / "conf" / "monitor_builder.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "P2_MONITOR_GROUPS"
            for target in node.targets
        )
    )
    groups = ast.literal_eval(assignment.value)
    assert all("导航Reward" not in group_name for group_name, _, _ in groups)
    allowed = re.compile(r"^[A-Za-z0-9_\-*一-鿿 ]{1,20}$")
    metrics = set()
    for group_name, _group_name_en, panels in groups:
        assert allowed.fullmatch(group_name), group_name
        for panel_name, _panel_name_en, panel_metrics in panels:
            assert allowed.fullmatch(panel_name), panel_name
            assert len(panel_metrics) <= 20, (panel_name, len(panel_metrics))
            metrics.update(panel_metrics)
    for metric in (
        "rollout_reward_mean",
        "rollout_advantage_std",
        "action_std_vx",
        "actor_learning_rate",
        "reward_frontier_shaping",
        "frontier_potential_before",
        "frontier_potential_after",
        "terminal_potential_clawback",
        "reward_body_collision",
        "reward_predictive_collision_risk",
        "predictive_collision_legacy_risk",
        "predictive_collision_wallness_left",
        "predictive_collision_risk_right",
        "collision_onset_rate",
        "reward_frontier_stagnation",
        "action_std_vy",
        "vy_actor_learning_rate",
        "vy_hard_limit",
        "reward_success",
        "reward_gait_symmetry",
        "gait_frequency_excess",
        "reward_decomposed_total",
        "cmd_v0_w0_count",
        "cmd_v4_w3_tracking_mae",
        "adapter_maze_mae",
        "adapter_outer_mae",
        "success_rate",
        "episode_success_fraction",
        "episode_timeout_fraction",
        "goal_progress_m_per_s",
        "goal_progress",
        "measured_vx",
        "vx_tracking_abs_error",
        "feedback_source_sport",
        "command_core_overflow_rate",
        "tilt_xy_norm",
        "low_level_action_saturation_rate",
        "curriculum_failures",
        "max_memory_reserved_ratio",
    ):
        assert metric in metrics
    p2_builder = source.split("def _build_p2_monitor():", 1)[1].split(
        "\n\ndef build_monitor", 1
    )[0]
    assert "P2_MONITOR_GROUPS" in p2_builder
    assert "TRACK_PANEL_SPECS" not in p2_builder
    assert "completed_count_track_l" not in metrics


def test_p2_monitor_upload_uses_current_process_pid(monkeypatch):
    submitted = []

    class Monitor:
        def get_pids(self):
            raise AssertionError("P2 monitor upload must not depend on registered PIDs")

        def put_data(self, payload):
            submitted.append(payload)

    monkeypatch.setattr("agent_ppo.workflow.p2_nav_ppo_workflow.os.getpid", lambda: 4321)
    assert _monitor_put(Monitor(), {"actor_loss": 1.25}) is True
    assert submitted == [{4321: {"actor_loss": 1.25}}]


def test_p2_monitor_upload_failure_is_visible_without_stopping_training():
    warnings = []

    class Monitor:
        def put_data(self, payload):
            del payload
            raise RuntimeError("monitor unavailable")

    logger = SimpleNamespace(warning=warnings.append)
    assert _monitor_put(Monitor(), {"actor_loss": 1.25}, logger) is False
    assert len(warnings) == 1
    assert "monitor upload failed" in warnings[0]
    assert "RuntimeError: monitor unavailable" in warnings[0]


def test_p2_tick_diagnostics_separate_outcomes_feedback_and_motion_quality():
    aux = torch.zeros(2, p2_contract.WORKER_AUX_DIM)
    aux[:, 6:9] = torch.tensor(((0.80, 0.02, 0.70), (0.40, 0.10, 0.00)))
    aux[:, 9] = torch.tensor((1.0, 0.0))
    aux[:, 10] = torch.tensor((0.25, 0.75))
    aux[:, 11] = torch.tensor((1.0, 0.0))
    aux[:, 12:15] = torch.tensor(((0.75, 0.03, 0.65), (0.45, 0.12, 0.02)))
    aux[:, 21:23] = torch.tensor(((0.10, 0.20), (0.0, 0.0)))
    values, valid_values, valid = _tick_diagnostic_values(
        target=torch.tensor(((1.25, 0.0, 1.0), (0.50, 0.0, 0.0))),
        executed=torch.tensor(((1.00, 0.0, 0.80), (0.50, 0.0, 0.0))),
        response_aux=aux,
        confidence=torch.tensor(((0.7,), (0.0,))),
        actions=torch.tensor(((6.0,) + (0.0,) * 11, (0.5,) * 12)),
        start_goal=torch.tensor((4.0, 3.0)),
        end_goal=torch.tensor((0.5, 2.8)),
        done=torch.tensor((True, False)),
        hard=torch.tensor((True, False)),
        timeout=torch.tensor((False, False)),
        terminal_reason=torch.tensor((1, 0)),
        duration_frames=torch.tensor((4, 10)),
        stuck=torch.tensor((False, True)),
        feedback_age_clip_s=0.8,
    )
    assert valid.tolist() == [False, False]
    assert values["success_rate"].tolist() == [1.0, 0.0]
    assert values["failure_rate"].tolist() == [0.0, 0.0]
    assert values["early_end_rate"].tolist() == [1.0, 0.0]
    assert values["goal_progress"][0].item() == pytest.approx(3.5)
    assert values["goal_progress"][1].item() == pytest.approx(0.2)
    assert values["command_core_overflow_rate"].tolist() == [1.0, 0.0]
    assert values["low_level_action_saturation_rate"].tolist() == [1.0, 0.0]
    assert values["stuck_penalty"].tolist() == [0.0, 0.0]
    assert valid_values["feedback_age_s"][0].item() == pytest.approx(0.2)
    assert valid_values["vx_tracking_abs_error"][0].item() == pytest.approx(0.2)


def test_p2_tick_diagnostics_counts_wall_stuck_as_failure_separately_from_timeout():
    aux = torch.zeros(1, p2_contract.WORKER_AUX_DIM)
    values, _, _ = _tick_diagnostic_values(
        target=torch.zeros(1, 3),
        executed=torch.zeros(1, 3),
        response_aux=aux,
        confidence=torch.zeros(1, 1),
        actions=torch.zeros(1, 12),
        start_goal=torch.tensor((2.0,)),
        end_goal=torch.tensor((2.0,)),
        done=torch.tensor((True,)),
        hard=torch.tensor((False,)),
        timeout=torch.tensor((True,)),
        terminal_reason=torch.tensor((4,)),
        duration_frames=torch.tensor((10,)),
        stuck=torch.tensor((True,)),
        feedback_age_clip_s=0.8,
    )
    assert values["failure_rate"].item() == 1.0
    assert values["timeout_rate"].item() == 0.0
    assert values["wall_stuck_reset_rate"].item() == 1.0


def test_p2_terminal_count_does_not_double_count_wall_stuck_failures():
    assert _terminal_outcome_count(1.0, 1.0, 1.0) == 3.0


def test_p2_action_mapper_and_initial_bias_cover_hard_boundary():
    actions = torch.tensor(
        (
            (-1.0, -1.0, -1.0),
            (1.0, 1.0, 1.0),
            (p2_contract.INITIAL_VX_NORMALIZED, 0.0, 0.0),
        )
    )
    commands = p2_contract.map_normalized_action(actions)
    assert torch.allclose(commands[0], torch.tensor((0.0, -0.4, -1.0)))
    assert torch.allclose(commands[1], torch.tensor((1.25, 0.4, 1.0)))
    assert torch.allclose(commands[2], torch.tensor((0.30, 0.0, 0.0)), atol=1e-6)
    assert math.isclose(p2_contract.INITIAL_VX_PRE_TANH, -0.5763397549, rel_tol=1e-6)
    actor = P2NavigationActor()
    assert torch.all(actor.log_std >= p2_contract.LOG_STD_MIN)
    assert torch.all(actor.log_std <= p2_contract.LOG_STD_MAX)
    assert actor.vy_mean_head.weight.count_nonzero() == 0
    assert actor.vy_mean_head.bias.item() == 0.0
    assert actor.vy_log_std.item() == pytest.approx(-1.1)


def test_reward_v8_frontier_signal_prefers_new_progress_without_detour_tax():
    def shaping(best_after):
        value, _before, _after = p2_contract.frontier_potential_shaping(
            episode_start_distance=torch.tensor((10.0,)),
            best_distance_before=torch.tensor((10.0,)),
            best_distance_after=torch.tensor((best_after,)),
            duration_frames=torch.tensor((10,)),
            terminal=torch.tensor((False,)),
        )
        return value.item() + p2_contract.TIME_COST_PER_TICK

    stable = shaping(9.96)
    slow = shaping(9.99)
    no_frontier = shaping(10.0)
    assert stable > slow > no_frontier
    assert p2_contract.SUCCESS_IMPULSE > 0.0
    assert p2_contract.TIMEOUT_IMPULSE < 0.0
    assert p2_contract.FAILURE_IMPULSE < p2_contract.TIMEOUT_IMPULSE


def test_frontier_potential_is_dense_but_cancels_at_terminal():
    shaping, before, after = p2_contract.frontier_potential_shaping(
        episode_start_distance=torch.tensor((10.0,)),
        best_distance_before=torch.tensor((9.0,)),
        best_distance_after=torch.tensor((8.5,)),
        duration_frames=torch.tensor((10,)),
        terminal=torch.tensor((False,)),
    )
    assert before.item() == pytest.approx(2.0)
    assert after.item() == pytest.approx(3.0)
    assert shaping.item() == pytest.approx(p2_contract.GAMMA_NAV * 3.0 - 2.0)

    terminal_shaping, terminal_before, terminal_after = (
        p2_contract.frontier_potential_shaping(
            episode_start_distance=torch.tensor((10.0,)),
            best_distance_before=torch.tensor((8.5,)),
            best_distance_after=torch.tensor((8.0,)),
            duration_frames=torch.tensor((4,)),
            terminal=torch.tensor((True,)),
        )
    )
    assert terminal_before.item() == pytest.approx(3.0)
    assert terminal_after.item() == 0.0
    assert terminal_shaping.item() == pytest.approx(-3.0)


def test_frontier_potential_does_not_reward_repeated_or_backward_motion():
    shaping, before, after = p2_contract.frontier_potential_shaping(
        episode_start_distance=torch.tensor((10.0, 10.0)),
        best_distance_before=torch.tensor((8.0, 8.0)),
        best_distance_after=torch.tensor((8.0, 8.0)),
        duration_frames=torch.tensor((10, 10)),
        terminal=torch.tensor((False, False)),
    )
    assert torch.equal(before, after)
    assert bool((shaping <= 0.0).all())


def test_body_collision_penalty_distinguishes_onset_persistence_and_terminal():
    force = torch.tensor((30.0, 90.0, 150.0, 200.0))
    previous = torch.tensor((False, False, True, False))
    terminal = torch.tensor((False, False, False, True))
    penalty, contact = p2_contract.body_collision_penalty(
        force, previous, terminal
    )
    assert penalty.tolist() == pytest.approx((0.0, -0.14, -0.03, 0.0))
    assert contact.tolist() == [False, True, True, False]


def _directional_depth(*, sector=None, upper=0.10, middle=0.10, lower=0.10):
    depth = torch.ones(1, 180, 320, 1)
    sectors = p2_contract.PREDICTIVE_COLLISION_HORIZONTAL_SECTORS
    selected = sectors.values() if sector is None else (sectors[sector],)
    for x0, x1 in selected:
        for name, value in (("upper", upper), ("middle", middle), ("lower", lower)):
            y0, y1 = p2_contract.PREDICTIVE_COLLISION_VERTICAL_BANDS[name]
            depth[:, y0:y1, x0:x1, 0] = value
    return depth


def test_predictive_collision_v2_separates_wall_slope_stairs_and_invalid_depth():
    wall = _directional_depth()
    slope = _directional_depth(upper=0.18, middle=0.10, lower=0.02)
    stairs = _directional_depth(upper=1.0, middle=1.0, lower=0.05)
    invalid = torch.zeros_like(wall)
    depth = torch.cat((wall, slope, stairs, invalid, torch.ones_like(wall)), dim=0)
    target = torch.tensor(((1.0, 0.0, 0.0),) * 5)
    penalty, clearance, stopping, risk, legacy, wallness, sector_risk = (
        p2_contract.predictive_collision_risk_penalty(depth, target)
    )
    assert risk[0].item() > 0.75
    assert risk[1].item() < risk[0].item() * 0.25
    assert risk[2].item() < risk[0].item() * 0.10
    assert penalty[3].item() == 0.0
    assert penalty[4].item() == 0.0
    assert clearance[3].item() == pytest.approx(5.0)
    assert stopping[0].item() > 0.0
    assert legacy.shape == (5,)
    assert wallness.shape == sector_risk.shape == (5, 3)
    assert bool((penalty <= 0.0).all())
    assert bool((penalty >= p2_contract.PREDICTIVE_COLLISION_WEIGHT).all())


def test_predictive_collision_v2_selects_lateral_and_yaw_direction():
    depths = torch.cat(
        (
            _directional_depth(sector="left"),
            _directional_depth(sector="right"),
            _directional_depth(sector="left"),
            _directional_depth(sector="right"),
        ),
        dim=0,
    )
    commands = torch.tensor(
        ((0.1, 0.4, 0.0), (0.1, -0.4, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
    )
    penalty, _clearance, _stop, risk, _legacy, _wallness, sectors = (
        p2_contract.predictive_collision_risk_penalty(depths, commands)
    )
    assert bool((risk > 0.0).all())
    assert sectors[0, 0] > sectors[0, 2]
    assert sectors[1, 2] > sectors[1, 0]
    assert penalty[2].item() < 0.0
    assert penalty[3].item() < 0.0


def test_predictive_collision_v2_zero_command_is_strictly_zero():
    output = p2_contract.predictive_collision_risk_penalty(
        torch.cat((_directional_depth(), _directional_depth()), dim=0),
        torch.tensor(((0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0))),
    )
    assert output[0].tolist() == [0.0, 0.0]
    assert output[2].tolist() == [0.0, 0.0]
    assert output[3].tolist() == [0.0, 0.0]
    assert all(bool(torch.isfinite(item).all()) for item in output)


def test_worker_collision_probe_excludes_feet_and_clears_reset_history():
    bridge = P2WorkerBridge.__new__(P2WorkerBridge)
    bridge.num_envs = 2
    bridge.device = torch.device("cpu")
    forces = torch.zeros(2, 5, 3)
    forces[0, 0, 0] = 60.0
    forces[0, 1, 0] = 120.0  # foot, must be excluded
    forces[1, 4, 0] = 150.0
    bridge._gait_window = SimpleNamespace(
        sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w=forces)),
        sensor_foot_ids=torch.tensor((1, 3)),
        sensor_body_names=("trunk", "FL_foot", "hip", "FR_foot", "head"),
        collision_valid=True,
        disable_collision=lambda reason: pytest.fail(reason),
    )
    bridge._collision_force_history = torch.zeros(
        p2_contract.NAV_PERIOD_FRAMES, 2
    )
    bridge._collision_force_index = 0
    observed = bridge._body_collision_force(torch.tensor((False, False)))
    assert observed.tolist() == pytest.approx((60.0, 150.0))

    forces.zero_()
    observed = bridge._body_collision_force(torch.tensor((False, True)))
    assert observed.tolist() == pytest.approx((60.0, 0.0))


def test_worker_collision_probe_fails_safe_when_mapping_is_invalid():
    bridge = P2WorkerBridge.__new__(P2WorkerBridge)
    bridge.num_envs = 1
    bridge.device = torch.device("cpu")
    bridge._gait_window = SimpleNamespace(collision_valid=False)
    bridge._collision_force_history = torch.full(
        (p2_contract.NAV_PERIOD_FRAMES, 1), 120.0
    )
    bridge._collision_force_index = 0
    observed = bridge._body_collision_force(torch.tensor((False,)))
    assert observed.item() == 0.0
    assert not bool(bridge._collision_force_history.any())


def test_gait_probe_maps_articulation_ids_to_permuted_contact_sensor_columns():
    num_envs = 1
    sensor = SimpleNamespace(
        body_names=("trunk", "RR_foot", "FL_foot", "FR_foot", "RL_foot"),
        data=SimpleNamespace(
            current_air_time=torch.tensor(((0.0, 0.4, 0.0, 0.2, 0.3),)),
            last_air_time=torch.tensor(((0.0, 0.4, 0.0, 0.2, 0.3),)),
            net_forces_w=torch.zeros(num_envs, 5, 3),
        ),
    )
    robot_velocity = torch.zeros(num_envs, 12, 3)
    robot_velocity[:, 2, 0] = 0.7
    robot = SimpleNamespace(
        find_bodies=lambda _pattern: (
            torch.tensor((2, 5, 8, 11)),
            ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
        ),
        data=SimpleNamespace(body_lin_vel_w=robot_velocity),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "unrelated_air_sensor": SimpleNamespace(
                    body_names=("wrong",),
                    data=SimpleNamespace(current_air_time=torch.zeros(1, 1)),
                ),
                "contact_forces": sensor,
            }
        )
    )
    probe = P2GaitWindowProbe(
        env, robot, num_envs=num_envs, device=torch.device("cpu")
    )
    assert probe.valid
    assert probe.collision_valid
    assert probe.robot_foot_ids.tolist() == [2, 5, 8, 11]
    assert probe.sensor_foot_ids.tolist() == [2, 3, 4, 1]
    result = probe.step(torch.tensor((False,)))
    assert result[0, 0:4].tolist() == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert result[0, 20:24].tolist() == pytest.approx((0.7, 0.0, 0.0, 0.0))


def test_gait_probe_uses_full_root_rotation_for_touchdown_lateral_margin():
    sensor = SimpleNamespace(
        body_names=("trunk", "FL_foot", "FR_foot", "RL_foot", "RR_foot"),
        data=SimpleNamespace(
            current_air_time=torch.ones(1, 5),
            last_air_time=torch.ones(1, 5),
            net_forces_w=torch.zeros(1, 5, 3),
        ),
    )
    body_pos = torch.zeros(1, 4, 3)
    robot = SimpleNamespace(
        find_bodies=lambda _pattern: (
            torch.tensor((0, 1, 2, 3)),
            ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
        ),
        data=SimpleNamespace(
            body_lin_vel_w=torch.zeros(1, 4, 3),
            body_pos_w=body_pos,
            root_pos_w=torch.zeros(1, 3),
            root_quat_w=torch.tensor(((2**-0.5, 2**-0.5, 0.0, 0.0),)),
        ),
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}))
    probe = P2GaitWindowProbe(env, robot, num_envs=1, device="cpu")
    probe.step(torch.tensor((False,)))
    body_pos[0, 0, 2] = 0.2
    sensor.data.current_air_time.zero_()
    probe.step(torch.tensor((False,)))
    assert probe.last_p3_detail[0, 8].item() == pytest.approx(0.2)


def test_gait_probe_keeps_per_stance_slip_budget():
    sensor = SimpleNamespace(
        body_names=("trunk", "FL_foot", "FR_foot", "RL_foot", "RR_foot"),
        data=SimpleNamespace(
            current_air_time=torch.zeros(1, 5),
            last_air_time=torch.zeros(1, 5),
            net_forces_w=torch.zeros(1, 5, 3),
        ),
    )
    velocity = torch.zeros(1, 4, 3)
    velocity[0, 0, 0] = 1.0
    robot = SimpleNamespace(
        find_bodies=lambda _pattern: (torch.arange(4), ("FL_foot", "FR_foot", "RL_foot", "RR_foot")),
        data=SimpleNamespace(body_lin_vel_w=velocity),
    )
    probe = P2GaitWindowProbe(
        SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor})),
        robot,
        num_envs=1,
        device="cpu",
    )
    assert probe.step(torch.tensor((False,)))[0, 20].item() == pytest.approx(1.0)
    assert probe.step(torch.tensor((False,)))[0, 20].item() == pytest.approx(1.0)
    sensor.data.current_air_time.zero_()
    sensor.data.current_air_time[:, 1] = 1.0
    # A completed stance is exported once through the P3-only detail tail.
    result = probe.step(torch.tensor((False,)))
    assert probe.last_p3_detail[0, 16].item() == pytest.approx(0.04)
    assert probe.last_p3_detail[0, 20].item() == pytest.approx(1.0)
    result = probe.step(torch.tensor((False,)))
    assert probe.last_p3_detail[0, 16].item() == pytest.approx(0.0)
    assert probe.last_p3_detail[0, 20].item() == pytest.approx(0.0)


def test_gait_probe_excludes_reset_boundary_sample_from_window_count():
    sensor = SimpleNamespace(
        body_names=("trunk", "FL_foot", "FR_foot", "RL_foot", "RR_foot"),
        data=SimpleNamespace(
            current_air_time=torch.zeros(1, 5),
            last_air_time=torch.zeros(1, 5),
            net_forces_w=torch.zeros(1, 5, 3),
        ),
    )
    robot = SimpleNamespace(
        find_bodies=lambda _pattern: (
            torch.tensor((0, 1, 2, 3)),
            ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
        ),
        data=SimpleNamespace(body_lin_vel_w=torch.zeros(1, 4, 3)),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"contact_forces": sensor})
    )
    probe = P2GaitWindowProbe(
        env, robot, num_envs=1, device=torch.device("cpu")
    )

    reset_result = probe.step(torch.tensor((True,)))
    assert not bool(reset_result.any())
    assert probe.env_counts.item() == 0
    assert probe.contact[:, 0].sum().item() == 0.0

    first_result = probe.step(torch.tensor((False,)))
    assert probe.env_counts.item() == 1
    assert probe.contact[:, 0].sum().item() == 4.0
    assert first_result[0, 0:4].tolist() == pytest.approx((1.0, 1.0, 1.0, 1.0))


def test_gait_probe_disables_rewards_when_contact_mapping_is_incomplete():
    sensor = SimpleNamespace(
        body_names=("trunk", "FL_foot", "FR_foot", "RL_foot"),
        data=SimpleNamespace(current_air_time=torch.zeros(1, 4)),
    )
    robot = SimpleNamespace(
        find_bodies=lambda _pattern: (
            torch.tensor((2, 5, 8, 11)),
            ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
        ),
        data=SimpleNamespace(body_lin_vel_w=torch.zeros(1, 12, 3)),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"contact_forces": sensor})
    )
    probe = P2GaitWindowProbe(
        env, robot, num_envs=1, device=torch.device("cpu")
    )
    assert not probe.valid
    assert not probe.collision_valid
    assert "RR_foot" in probe.invalid_reason
    assert torch.equal(
        probe.step(torch.tensor((False,))),
        torch.zeros(1, p2_contract.GAIT_DIAGNOSTIC_DIM),
    )


def test_track_segment_index_uses_world_x_and_not_spawn_row():
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(track_length=3, size=(8.0, 8.0))
        )
    )
    segment, status = _track_segment_index(
        terrain,
        torch.tensor((-20.0, -12.0, -4.001, -4.0, 3.999, 4.0, 20.0, float("nan"))),
        describe=True,
    )
    assert segment.tolist() == [0, 0, 0, 1, 1, 2, 2, -1]
    assert "boundaries=[-12.0, -4.0, 4.0, 12.0]" in status


def test_track_segment_metric_mapping_uses_configured_terrain_semantics():
    assert p2_contract.canonical_track_segment_labels(()) == (
        "slope_inv",
        "stairs_inv",
        "maze",
    )
    maze_labels = p2_contract.canonical_track_segment_labels(
        ["open_entry_maze"]
    )
    assert maze_labels == ("maze",)
    mapped = p2_contract.track_segment_metric_indices(
        torch.tensor((0.0, -1.0, 1.0)), maze_labels
    )
    assert mapped.tolist() == [2, -1, -1]

    p2_labels = p2_contract.canonical_track_segment_labels(
        ["pyramid_slope_inv", "pyramid_stairs_inv", "open_entry_maze"]
    )
    assert p2_contract.track_segment_metric_indices(
        torch.tensor((0.0, 1.0, 2.0)), p2_labels
    ).tolist() == [0, 1, 2]


def test_terminal_safe_segment_ignores_new_episode_position_after_reset():
    observed = _terminal_safe_segment(
        live_segment=torch.tensor((0.0, 1.0, 2.0)),
        terminal_segment=torch.tensor((2.0, -1.0, 1.0)),
        transition_done=torch.tensor((True, True, False)),
    )
    assert observed.tolist() == [2.0, 1.0, 2.0]


def test_terminal_safe_tensor_keeps_old_episode_commands_and_aux():
    live = torch.tensor(((0.0, 0.0, 0.0), (0.4, 0.1, -0.2)))
    terminal = torch.tensor(((0.8, -0.2, 0.5), (9.0, 9.0, 9.0)))
    selected = _terminal_safe_tensor(
        live, terminal, torch.tensor((True, False))
    )
    assert torch.equal(selected[0], terminal[0])
    assert torch.equal(selected[1], live[1])


def test_frontier_stagnation_is_monotonic_without_recovery_bonus():
    algorithm = AlgorithmP2NavPPO.__new__(AlgorithmP2NavPPO)
    algorithm.device = torch.device("cpu")
    algorithm.num_envs = 1
    algorithm._initialize_navigation_reward_state()
    algorithm.frontier_history.fill_(5.0)

    stagnation = algorithm._frontier_rewards(
        best_before=torch.tensor((5.0,)),
        end_goal=torch.tensor((5.0,)),
        terminal=torch.tensor((False,)),
    )
    assert stagnation.item() == pytest.approx(-0.015)

    stagnation = algorithm._frontier_rewards(
        best_before=torch.tensor((5.0,)),
        end_goal=torch.tensor((4.92,)),
        terminal=torch.tensor((False,)),
    )
    assert stagnation.item() == 0.0


def test_wait_then_frontier_progress_is_worse_than_continuous_progress():
    shaping, _before, _after = p2_contract.frontier_potential_shaping(
        torch.tensor((10.0,)),
        torch.tensor((10.0,)),
        torch.tensor((9.92,)),
        torch.tensor((10,)),
        torch.tensor((False,)),
    )
    continuous = shaping + p2_contract.TIME_COST_PER_TICK
    wait_then_progress = (
        shaping
        + 16 * p2_contract.TIME_COST_PER_TICK
        + p2_contract.STAGNATION_INITIAL_PENALTY
    )
    assert continuous.item() > wait_then_progress.item()


def test_gait_baseline_is_zero_inside_envelope_and_caps_pathology():
    baseline = P2GaitBaseline()
    assert baseline.finalized
    normal = torch.zeros(1, p2_contract.WORKER_AUX_DIM)
    normal[:, p2_contract.GAIT_VALID_INDEX] = 1.0
    normal[:, p2_contract.GAIT_DUTY_SLICE] = 0.5
    normal_penalty, _ = baseline.penalty(normal)
    assert normal_penalty.item() == 0.0

    pathological = normal.clone()
    pathological[:, p2_contract.GAIT_DUTY_SLICE] = torch.tensor((1.0, 0.0, 1.0, 0.0))
    pathological[:, p2_contract.GAIT_MEAN_SWING_SLICE] = torch.tensor(
        (1.5, 0.0, 1.5, 0.0)
    )
    pathological[:, p2_contract.GAIT_PROLONGED_RATIO_SLICE] = 1.0
    pathological[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE] = torch.tensor(
        (4.0, 0.0, 4.0, 0.0)
    )
    pathological_penalty, excess = baseline.penalty(pathological)
    assert pathological_penalty.item() == 0.0
    assert excess["duty_excess"].item() > 0.0
    assert excess["prolonged_excess"].item() > 0.0


def test_gait_parent_baseline_is_fixed_and_detects_fast_slow_legs():
    baseline = P2GaitBaseline()
    thresholds_before = dict(baseline.thresholds)
    samples_before = baseline.samples
    live = torch.zeros(4, p2_contract.WORKER_AUX_DIM)
    live[:, p2_contract.GAIT_VALID_INDEX] = 1.0
    live[:, p2_contract.GAIT_DUTY_SLICE] = torch.tensor((1.0, 0.0, 1.0, 0.0))
    baseline.observe(live)
    assert baseline.thresholds == thresholds_before
    assert baseline.samples == samples_before == 0

    frequency_only = torch.zeros(1, p2_contract.WORKER_AUX_DIM)
    frequency_only[:, p2_contract.GAIT_VALID_INDEX] = 1.0
    frequency_only[:, p2_contract.GAIT_DUTY_SLICE] = 0.5
    frequency_only[:, p2_contract.GAIT_MEAN_SWING_SLICE] = 0.20
    frequency_only[:, p2_contract.GAIT_PROLONGED_RATIO_SLICE] = 0.02
    frequency_only[:, p2_contract.GAIT_STEP_FREQUENCY_SLICE] = torch.tensor(
        (4.0, 1.0, 4.0, 1.0)
    )
    penalty, excess = baseline.penalty(frequency_only)
    assert penalty.item() == 0.0
    assert excess["duty_excess"].item() == 0.0
    assert excess["swing_excess"].item() == 0.0
    assert excess["prolonged_excess"].item() == 0.0
    assert excess["frequency_excess"].item() > 0.0


@pytest.mark.parametrize(
    ("seconds", "nav", "actor", "head", "critic", "adapter", "entropy"),
    (
        (0.0, 0.15, 0.10, 1.0, 0.30, 0.50, 0.006),
        (1800.0, 0.25, 0.15, 1.0, 0.30, 0.50, 0.005),
        (5400.0, 0.15, 0.10, 0.5, 0.20, 0.50, 0.004),
        (7200.0, 0.15, 0.10, 0.5, 0.20, 0.50, 0.004),
    ),
)
def test_two_hour_safe_direction_optimizer_schedule(
    seconds, nav, actor, head, critic, adapter, entropy
):
    schedule = p2_contract.training_schedule(seconds)
    assert schedule["cnn_unfrozen"] is True
    assert schedule["navigation_multiplier"] == pytest.approx(nav)
    assert schedule["actor_trunk_multiplier"] == pytest.approx(actor)
    assert schedule["actor_multiplier"] == pytest.approx(actor)
    assert schedule["vy_actor_multiplier"] == pytest.approx(actor)
    assert schedule["safety_head_multiplier"] == pytest.approx(head)
    assert schedule["critic_multiplier"] == pytest.approx(critic)
    assert schedule["adapter_multiplier"] == pytest.approx(adapter)
    assert schedule["entropy_coefficient"] == pytest.approx(entropy)


def test_vy_action_domain_is_fully_open_from_first_rollout():
    limits = p2_contract.vy_action_limits()
    assert limits == {"trusted_abs_vy": 0.20, "hard_abs_vy": 0.40}
    contract = p2_contract.command_contract()
    assert contract["vy_action_domain"]["schedule"] == (
        "fully_open_from_first_rollout"
    )


def _safety_critic(height: torch.Tensor, *, available=1.0, front=0.0, left=0.0, right=0.0):
    critic = torch.zeros(height.shape[0], p2_contract.CRITIC_OBS_DIM)
    critic[:, 60:316] = height.reshape(height.shape[0], -1)
    critic[:, 319:323] = torch.tensor((available, front, left, right))
    return critic


def test_privileged_safety_teacher_keeps_slope_and_small_stairs_passable():
    x = torch.linspace(0.0, 0.30, 16)
    slope = x.reshape(1, 1, 16).expand(1, 16, 16)
    stairs = (torch.arange(16) // 3 * 0.10).reshape(1, 1, 16).expand(1, 16, 16)
    safe, valid, diagnostics = p2_contract.privileged_safe_directions(
        torch.cat((_safety_critic(slope), _safety_critic(stairs)), dim=0)
    )
    assert valid.tolist() == [True, True]
    assert bool((safe > 0.80).all())
    assert bool((diagnostics["terrain_passable3"] > 0.80).all())


def test_privileged_safety_teacher_uses_positive_y_as_left_and_rejects_nan_height():
    height = torch.zeros(2, 16, 16)
    height[0, 9:16] = (torch.arange(16) % 2).float()  # body +y: left sector
    height[1] = float("nan")
    safe, valid, diagnostics = p2_contract.privileged_safe_directions(
        _safety_critic(height)
    )
    assert valid.tolist() == [True, False]
    assert safe[0, 0] < 0.1
    assert safe[0, 2] > 0.9
    assert torch.equal(safe[1], torch.zeros(3))
    assert diagnostics["height_sector_valid3"][1].sum().item() == 0.0


def test_privileged_safety_teacher_marks_wall_sector_and_penalizes_missed_alternative():
    height = torch.zeros(1, 16, 16)
    critic = _safety_critic(height, front=0.95)
    safe, valid, _ = p2_contract.privileged_safe_directions(critic)
    assert safe[0, 1] < 0.1
    assert safe[0, 0] > 0.9 and safe[0, 2] > 0.9
    forward = torch.tensor(((0.5, 0.0, 0.0),))
    penalty, diagnostics = p2_contract.missed_safe_direction_penalty(
        safe, forward, valid, p2_contract.TRAINING_HOURS * 3600.0
    )
    assert penalty.item() < 0.0
    assert diagnostics["safe_gap"].item() > 0.0
    stopped, _ = p2_contract.missed_safe_direction_penalty(
        safe, torch.zeros_like(forward), valid, 7200.0
    )
    assert stopped.item() == 0.0


def test_safe_direction_penalty_is_zero_without_valid_scanner_or_alternative():
    command = torch.tensor(((0.5, 0.0, 0.0),))
    equal_safe = torch.full((1, 3), 0.5)
    invalid, _ = p2_contract.missed_safe_direction_penalty(
        equal_safe, command, torch.tensor((False,)), 7200.0
    )
    equivalent, _ = p2_contract.missed_safe_direction_penalty(
        equal_safe, command, torch.tensor((True,)), 7200.0
    )
    assert invalid.item() == 0.0
    assert equivalent.item() == 0.0
    diagnostics = p2_contract.missed_safe_direction_penalty(
        equal_safe, command, torch.tensor((True,)), 7200.0
    )[1]
    assert diagnostics["active"].item() == 0.0
    assert diagnostics["selection_eligible"].item() == 0.0
    assert diagnostics["selected_safest"].item() == 0.0


def test_p2_eval_aux_transport_reuses_scan_prefix_without_shape_drift():
    policy = torch.zeros(3, nav_contract.POLICY_OBS_DIM)
    aux = torch.arange(90, dtype=torch.float32).reshape(3, 30)
    packed = p2_contract.pack_eval_response_aux(policy, aux)
    assert packed.shape == policy.shape
    assert torch.equal(p2_contract.unpack_eval_response_aux(packed), aux)
    assert torch.equal(packed[:, :45], policy[:, :45])
    assert torch.equal(packed[:, 76:], policy[:, 76:])
    with torch.no_grad():
        missing_marker = packed.clone()
        missing_marker[:, p2_contract.EVAL_AUX_MARKER_INDEX] = 0.0
    try:
        p2_contract.unpack_eval_response_aux(missing_marker)
    except RuntimeError as exc:
        assert "marker is missing" in str(exc)
    else:
        raise AssertionError("missing P2 eval transport marker was accepted")


def test_p2_policy_process_packs_aux_only_during_eval(monkeypatch):
    base = torch.zeros(2, nav_contract.POLICY_OBS_DIM)
    aux = torch.arange(60, dtype=torch.float32).reshape(2, 30)
    monkeypatch.setattr(
        NavPolicyObservationProcess,
        "process",
        lambda _self: base.clone(),
    )
    monkeypatch.setattr(p2_observation_process, "p2_response_aux", lambda _env: aux)
    process = object.__new__(P2PolicyObservationProcess)
    process.env = SimpleNamespace(_is_eval=True)
    packed = process.process()
    assert torch.equal(p2_contract.unpack_eval_response_aux(packed), aux)
    process.env._is_eval = False
    assert torch.equal(process.process(), base)


def test_p2_policy_process_installs_terminal_bridge_after_runtime_env_binding(
    monkeypatch,
):
    class _RuntimeBoundEnv:
        def __init__(self):
            self._is_eval = False
            aux = torch.zeros(1, p2_contract.WORKER_AUX_DIM)
            aux[0, 24] = 1.0
            aux[0, 25] = 1.0
            self._agent_ppo_p2_worker_bridge = SimpleNamespace(last_aux=aux)

        def step(self, _actions):
            return (
                torch.zeros(1, 1),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {},
            )

    base = torch.zeros(1, nav_contract.POLICY_OBS_DIM)
    monkeypatch.setattr(
        NavPolicyObservationProcess,
        "process",
        lambda _self: base.clone(),
    )
    process = P2PolicyObservationProcess()
    assert process.env is None

    env = _RuntimeBoundEnv()
    process.env = env
    assert torch.equal(process.process(), base)
    assert env._agent_ppo_p2_terminal_return_bridge is True

    first_wrapped_step = env.step
    assert torch.equal(process.process(), base)
    assert env.step == first_wrapped_step

    _, _, terminated, truncated, _ = env.step(torch.zeros(1, 1))
    assert terminated.tolist() == [True]
    assert truncated.tolist() == [False]


def test_p2_resume_candidates_are_same_id_and_p2_only(tmp_path):
    names = [path.rsplit("/", 1)[-1] for path in p2_nav_checkpoint_candidates(tmp_path, 42)]
    assert names == [
        "model.ckpt-safestable-42.pkl",
        "model.ckpt-safefull-42.pkl",
        "model.ckpt-safewarm-42.pkl",
        "model.ckpt-navfull-42.pkl",
        "model.ckpt-vyadapt-42.pkl",
        "model.ckpt-vywarm-42.pkl",
        "model.ckpt-navadapt-42.pkl",
        "model.ckpt-navwarm-42.pkl",
    ]


def test_safe_phase_checkpoint_is_discovered_before_legacy_p2(tmp_path):
    (tmp_path / "model.ckpt-navfull-42.pkl").touch()
    (tmp_path / "model.ckpt-safefull-42.pkl").touch()
    candidates = p2_nav_evaluation_candidates(tmp_path, 42)
    existing = [path for path in candidates if Path(path).is_file()]
    assert Path(existing[0]).name == "model.ckpt-safefull-42.pkl"


def test_p2_training_candidates_prefer_exact_resume_over_same_id_parent(tmp_path):
    parent = tmp_path / "model.ckpt-responsecalib-37953.pkl"
    parent.touch()
    names = [
        path.rsplit("/", 1)[-1]
        for path in p2_nav_training_candidates(
            tmp_path,
            37953,
            parent_model_id=37953,
        )
    ]
    assert names[:8] == [
        "model.ckpt-safestable-37953.pkl",
        "model.ckpt-safefull-37953.pkl",
        "model.ckpt-safewarm-37953.pkl",
        "model.ckpt-navfull-37953.pkl",
        "model.ckpt-vyadapt-37953.pkl",
        "model.ckpt-vywarm-37953.pkl",
        "model.ckpt-navadapt-37953.pkl",
        "model.ckpt-navwarm-37953.pkl",
    ]
    assert names[-1] == "model.ckpt-responsecalib-37953.pkl"


def test_p2_model_id_is_preference_not_single_point_gate(tmp_path):
    parent = tmp_path / "model.ckpt-responsecalib-37953.pkl"
    parent.touch()
    training = p2_nav_training_candidates(
        tmp_path,
        88888,
        parent_model_id=37953,
    )
    assert str(parent) in training

    nav = tmp_path / "model.ckpt-navfull-42.pkl"
    nav.touch()
    evaluation = p2_nav_evaluation_candidates(tmp_path, "latest")
    assert str(nav) in evaluation


def test_squashed_log_prob_matches_change_of_variables():
    pre_tanh = torch.tensor(((0.2, -0.7), (1.1, 0.0)), dtype=torch.float64)
    mean = torch.tensor(((0.1, -0.3), (0.7, 0.1)), dtype=torch.float64)
    log_std = torch.tensor(((-0.7, -0.4), (-0.2, -0.9)), dtype=torch.float64)
    actual = squashed_log_prob(pre_tanh, mean, log_std)
    distribution = torch.distributions.Normal(mean, log_std.exp())
    expected = (
        distribution.log_prob(pre_tanh)
        - torch.log1p(-torch.tanh(pre_tanh).square())
    ).sum(-1, keepdim=True)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_adapter_confidence_decreases_with_age_sigma_and_domain_overflow():
    common = {
        "velocity_valid": torch.ones(3, 1),
        "velocity_age": torch.tensor(((0.0,), (0.2,), (0.0,))),
        "velocity_log_sigma": torch.tensor(((-4.0, -4.0, -4.0), (-4.0, -4.0, -4.0), (0.0, 0.0, 0.0))),
        "target_cmd3": torch.tensor(((0.8, 0.0, 0.5), (0.8, 0.0, 0.5), (1.25, 0.0, 1.0))),
    }
    confidence = p2_contract.adapter_confidence(**common).squeeze(-1)
    assert confidence[0] > confidence[1]
    assert confidence[0] > confidence[2]
    invalid = p2_contract.adapter_confidence(
        velocity_valid=torch.zeros(1, 1),
        velocity_age=torch.zeros(1, 1),
        velocity_log_sigma=torch.full((1, 3), -4.0),
        target_cmd3=torch.zeros(1, 3),
    )
    assert invalid.item() == 0.0


def test_tracking_penalty_masks_invalid_sport_xy_but_keeps_imu_yaw():
    exec_cmd = torch.tensor(((1.0, 0.5, 0.4), (1.0, 0.5, 0.4)))
    measured = torch.zeros_like(exec_cmd)
    error = p2_contract.normalized_tracking_error(
        exec_cmd,
        measured,
        torch.tensor(((1.0,), (0.0,))),
    )
    assert error[0] > error[1]
    assert torch.allclose(error[1], torch.tensor((0.24,)))


class _TransitionCapture:
    full = False

    def add(self, **transition):
        self.transition = transition


class _ZeroCritic:
    def __call__(self, inputs, hidden=None, reset_mask=None):
        return torch.zeros(inputs.shape[0], 1), hidden


def test_finish_tick_attributes_tracking_penalty_to_end_feedback():
    algorithm = AlgorithmP2NavPPO.__new__(AlgorithmP2NavPPO)
    algorithm.device = torch.device("cpu")
    algorithm.num_envs = 1
    algorithm.nav_period_frames = p2_contract.NAV_PERIOD_FRAMES
    algorithm.command = SimpleNamespace(
        active_target=torch.tensor([[1.0, 0.0, 0.0]]),
        exec_cmd=torch.tensor([[1.0, 0.0, 0.0]]),
        inject=lambda _obs, _critic: None,
    )
    algorithm.critic = _ZeroCritic()
    algorithm.critic_hidden = None
    algorithm.reset_since_tick = torch.zeros(1, dtype=torch.bool)
    algorithm.rollout = _TransitionCapture()
    algorithm.rollout_invalid = False
    algorithm.invalid_transition_count = 0
    algorithm.effective_training_seconds = p2_contract.SAFETY_WARM_END_SECONDS
    algorithm.gait_baseline = P2GaitBaseline()
    algorithm.gait_baseline.finalize()
    algorithm.best_goal_distance = torch.full((1,), float("inf"))
    algorithm.pending_tick = {
        "command_penalty": torch.tensor([[-0.1]]),
        "tick_penalty": torch.tensor([[-0.1]]),
        "target_cmd3": torch.tensor([[1.0, 0.0, 0.0]]),
    }
    next_wire = torch.zeros(1, p2_contract.PRIVILEGED_WIRE_DIM)
    aux_start = p2_contract.CRITIC_OBS_DIM
    next_wire[:, aux_start + 12] = 0.5
    algorithm.finish_tick(
        torch.zeros(1, nav_contract.POLICY_OBS_DIM),
        next_wire,
        frame_safety_reward=torch.zeros(1),
        start_goal_distance=torch.full((1,), 2.0),
        end_goal_distance=torch.full((1,), 2.0),
        terminal_reason=torch.zeros(1, dtype=torch.long),
        duration_frames=torch.full((1,), 10),
        hard_terminated=torch.zeros(1, dtype=torch.bool),
        timeout=torch.zeros(1, dtype=torch.bool),
    )
    # -0.1 command change, -0.02 time and -0.00064 true-velocity tracking.
    assert torch.allclose(
        algorithm.rollout.transition["reward"],
        torch.tensor([[-0.12064]]),
        atol=1.0e-7,
    )


def test_finish_tick_zeroes_unattributed_reset_reward_and_terminates_gae():
    algorithm = AlgorithmP2NavPPO.__new__(AlgorithmP2NavPPO)
    algorithm.device = torch.device("cpu")
    algorithm.num_envs = 1
    algorithm.nav_period_frames = p2_contract.NAV_PERIOD_FRAMES
    algorithm.command = SimpleNamespace(
        active_target=torch.tensor([[1.0, 0.0, 0.0]]),
        exec_cmd=torch.tensor([[1.0, 0.0, 0.0]]),
        inject=lambda _obs, _critic: None,
    )
    algorithm.critic = _ZeroCritic()
    algorithm.critic_hidden = None
    algorithm.reset_since_tick = torch.ones(1, dtype=torch.bool)
    algorithm.rollout = _TransitionCapture()
    algorithm.rollout_invalid = False
    algorithm.invalid_transition_count = 0
    algorithm.gait_baseline = P2GaitBaseline()
    algorithm.gait_baseline.finalize()
    algorithm.best_goal_distance = torch.tensor((2.0,))
    algorithm.pending_tick = {
        "command_penalty": torch.tensor([[-0.1]]),
        "tick_penalty": torch.tensor([[-0.1]]),
        "target_cmd3": torch.tensor([[1.0, 0.0, 0.0]]),
    }

    algorithm.finish_tick(
        torch.zeros(1, nav_contract.POLICY_OBS_DIM),
        torch.zeros(1, p2_contract.PRIVILEGED_WIRE_DIM),
        frame_safety_reward=torch.tensor((5.0,)),
        start_goal_distance=torch.tensor((3.0,)),
        end_goal_distance=torch.tensor((2.0,)),
        terminal_reason=torch.zeros(1, dtype=torch.long),
        duration_frames=torch.full((1,), p2_contract.NAV_PERIOD_FRAMES),
        hard_terminated=torch.zeros(1, dtype=torch.bool),
        timeout=torch.zeros(1, dtype=torch.bool),
        unattributed_boundary=torch.ones(1, dtype=torch.bool),
    )

    transition = algorithm.rollout.transition
    assert transition["reward"].item() == 0.0
    assert transition["bootstrap_mask"].item() == 0.0
    assert transition["continuation_mask"].item() == 0.0
    assert transition["valid_mask"].item() == 0.0
    assert algorithm.last_tick_diagnostics["unattributed_reset_boundary"].item() == 1.0
    assert not algorithm.rollout_invalid


def test_finish_tick_valid_mask_does_not_cross_broadcast_environment_rows():
    algorithm = AlgorithmP2NavPPO.__new__(AlgorithmP2NavPPO)
    algorithm.device = torch.device("cpu")
    algorithm.num_envs = 2
    algorithm.nav_period_frames = p2_contract.NAV_PERIOD_FRAMES
    algorithm.command = SimpleNamespace(
        active_target=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        exec_cmd=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        inject=lambda _obs, _critic: None,
    )
    algorithm.critic = _ZeroCritic()
    algorithm.critic_hidden = None
    algorithm.reset_since_tick = torch.zeros(2, dtype=torch.bool)
    algorithm.rollout = _TransitionCapture()
    algorithm.rollout_invalid = False
    algorithm.invalid_transition_count = 0
    algorithm.gait_baseline = P2GaitBaseline()
    algorithm.gait_baseline.finalize()
    algorithm.best_goal_distance = torch.full((2,), float("inf"))
    algorithm.pending_tick = {
        "command_penalty": torch.zeros(2, 1),
        "tick_penalty": torch.zeros(2, 1),
        "target_cmd3": torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        ),
    }

    algorithm.finish_tick(
        torch.zeros(2, nav_contract.POLICY_OBS_DIM),
        torch.zeros(2, p2_contract.PRIVILEGED_WIRE_DIM),
        frame_safety_reward=torch.zeros(2),
        start_goal_distance=torch.full((2,), 2.0),
        end_goal_distance=torch.full((2,), 2.0),
        terminal_reason=torch.zeros(2, dtype=torch.long),
        duration_frames=torch.full((2,), p2_contract.NAV_PERIOD_FRAMES),
        hard_terminated=torch.zeros(2, dtype=torch.bool),
        timeout=torch.zeros(2, dtype=torch.bool),
        unattributed_boundary=torch.tensor((False, True)),
    )

    valid_mask = algorithm.rollout.transition["valid_mask"]
    assert valid_mask.shape == (2, 1)
    assert torch.equal(valid_mask, torch.tensor([[1.0], [0.0]]))


def test_finish_tick_sanitizes_invalid_reward_rows_before_rollout_storage():
    algorithm = AlgorithmP2NavPPO.__new__(AlgorithmP2NavPPO)
    algorithm.device = torch.device("cpu")
    algorithm.num_envs = 1
    algorithm.nav_period_frames = p2_contract.NAV_PERIOD_FRAMES
    algorithm.command = SimpleNamespace(
        active_target=torch.tensor([[0.5, 0.0, 0.0]]),
        exec_cmd=torch.tensor([[0.5, 0.0, 0.0]]),
        inject=lambda _obs, _critic: None,
    )
    algorithm.critic = _ZeroCritic()
    algorithm.critic_hidden = None
    algorithm.reset_since_tick = torch.zeros(1, dtype=torch.bool)
    algorithm.rollout = _TransitionCapture()
    algorithm.rollout_invalid = False
    algorithm.invalid_transition_count = 0
    algorithm.gait_baseline = P2GaitBaseline()
    algorithm.gait_baseline.finalize()
    algorithm.best_goal_distance = torch.tensor((1.0,))
    algorithm.pending_tick = {
        "command_penalty": torch.zeros(1, 1),
        "tick_penalty": torch.zeros(1, 1),
        "target_cmd3": torch.tensor([[0.5, 0.0, 0.0]]),
    }
    next_wire = torch.zeros(1, p2_contract.PRIVILEGED_WIRE_DIM)
    aux_start = p2_contract.CRITIC_OBS_DIM
    next_wire[:, aux_start + p2_contract.GAIT_DUTY_SLICE.start] = float("nan")
    next_wire[
        :, aux_start + p2_contract.BODY_COLLISION_FORCE_INDEX
    ] = float("inf")

    algorithm.finish_tick(
        torch.zeros(1, nav_contract.POLICY_OBS_DIM),
        next_wire,
        frame_safety_reward=torch.full((1,), float("inf")),
        start_goal_distance=torch.tensor((2.0,)),
        end_goal_distance=torch.full((1,), float("nan")),
        terminal_reason=torch.zeros(1, dtype=torch.long),
        duration_frames=torch.zeros(1, dtype=torch.long),
        hard_terminated=torch.zeros(1, dtype=torch.bool),
        timeout=torch.zeros(1, dtype=torch.bool),
    )

    assert algorithm.rollout_invalid
    assert algorithm.invalid_transition_count == 1
    assert algorithm.rollout.transition["reward"].item() == 0.0
    assert algorithm.rollout.transition["duration_frames"].item() == 1
    assert torch.isfinite(algorithm.rollout.transition["reward"]).all()
    assert algorithm.best_goal_distance.item() == pytest.approx(1.0)
    assert all(
        torch.isfinite(value).all()
        for value in algorithm.last_tick_penalties.values()
    )


def test_terminal_tick_preserves_episode_best_until_reward_is_settled():
    algorithm = AlgorithmP2NavPPO.__new__(AlgorithmP2NavPPO)
    algorithm.device = torch.device("cpu")
    algorithm.num_envs = 1
    algorithm.nav_period_frames = p2_contract.NAV_PERIOD_FRAMES
    algorithm.command = SimpleNamespace(
        active_target=torch.tensor([[0.5, 0.0, 0.0]]),
        exec_cmd=torch.tensor([[0.5, 0.0, 0.0]]),
        command_epoch=0,
        inject=lambda _obs, _critic: None,
        reset=lambda _ids: None,
        step=lambda: None,
    )
    algorithm.response_buffer = SimpleNamespace(
        append=lambda _aux, _done, **_metadata: None
    )
    algorithm.low_level_encoder = SimpleNamespace(
        reset_hidden_state_for_envs=lambda _ids: None
    )
    algorithm.actor_hidden = None
    algorithm.critic_hidden = None
    algorithm.adapter_hidden = None
    algorithm.reset_since_tick = torch.zeros(1, dtype=torch.bool)
    algorithm.critic = _ZeroCritic()
    algorithm.rollout = _TransitionCapture()
    algorithm.rollout_invalid = False
    algorithm.invalid_transition_count = 0
    algorithm.effective_training_seconds = p2_contract.SAFETY_WARM_END_SECONDS
    algorithm.gait_baseline = P2GaitBaseline()
    algorithm.gait_baseline.finalize()
    algorithm.best_goal_distance = torch.tensor((1.0,))
    algorithm.pending_tick = {
        "command_penalty": torch.zeros(1, 1),
        "tick_penalty": torch.zeros(1, 1),
        "target_cmd3": torch.tensor([[0.5, 0.0, 0.0]]),
    }

    algorithm.frame_end(
        torch.zeros(1, p2_contract.WORKER_AUX_DIM),
        torch.ones(1, dtype=torch.bool),
    )
    assert algorithm.best_goal_distance.item() == pytest.approx(1.0)

    algorithm.finish_tick(
        torch.zeros(1, nav_contract.POLICY_OBS_DIM),
        torch.zeros(1, p2_contract.PRIVILEGED_WIRE_DIM),
        frame_safety_reward=torch.zeros(1),
        start_goal_distance=torch.tensor((2.0,)),
        end_goal_distance=torch.tensor((1.5,)),
        terminal_reason=torch.full((1,), 2, dtype=torch.long),
        duration_frames=torch.full((1,), 10),
        hard_terminated=torch.ones(1, dtype=torch.bool),
        timeout=torch.zeros(1, dtype=torch.bool),
    )
    assert algorithm.last_tick_penalties["frontier_shaping"].item() == pytest.approx(-2.0)
    assert algorithm.last_tick_diagnostics["terminal_potential_clawback"].item() == pytest.approx(-2.0)
    assert torch.isinf(algorithm.best_goal_distance).all()
    assert torch.isinf(algorithm.episode_start_goal_distance).all()


def test_p2_transport_split_and_aisrv_command_overwrite():
    wire = torch.zeros(2, p2_contract.PRIVILEGED_WIRE_DIM)
    critic, aux = split_p2_transport(wire)
    target = torch.tensor(((0.4, 0.0, 0.2), (1.1, 0.0, -0.7)))
    executed = target * 0.5
    patched = patch_owned_commands(aux, target, executed, 17)
    assert critic.shape == (2, 323)
    assert torch.equal(patched[:, 0:3], target)
    assert torch.equal(patched[:, 3:6], executed)
    assert torch.equal(patched[:, 26], torch.full((2,), 17.0))
    per_env = patch_owned_commands(aux, target, executed, torch.tensor([4, 9]))
    assert torch.equal(per_env[:, 26], torch.tensor([4.0, 9.0]))
    with pytest.raises(ValueError, match="one value per environment"):
        patch_owned_commands(aux, target, executed, torch.tensor([1, 2, 3]))


def test_p2_long_horizon_stability_checks_the_entire_target_window():
    start = torch.zeros(2, 30)
    middle = start.clone()
    end = start.clone()
    middle[0, 0] = 0.20
    end[0, 0] = 0.0
    middle[1, 0] = 0.03
    end[1, 0] = 0.04
    stable = P2ResponseAuxBuffer._stable_target(start, [middle, end])
    assert stable.tolist() == [False, True]


def test_navigation_encoder_is_a_physical_copy_not_an_alias():
    source = SimpleCNN(output_dim=32)
    target = NavigationEncoder()
    target.copy_from_low_level_cnn(source)
    for own, parent in zip(target.cnn.parameters(), source.parameters()):
        assert own is not parent
        assert own.untyped_storage().data_ptr() != parent.untyped_storage().data_ptr()
        assert torch.equal(own, parent)


def test_p2_actor_and_critic_shapes_and_three_dimensional_density():
    batch = 3
    actor = P2NavigationActor()
    critic = P2NavigationCritic()
    actor_input = assemble_actor_input(
        torch.zeros(batch, 32),
        torch.zeros(batch, 36),
        torch.zeros(batch, 16),
        torch.ones(batch, 1),
    )
    critic_input = assemble_critic_input(
        torch.zeros(batch, 323),
        torch.zeros(batch, 3),
        torch.tensor(p2_contract.NAV_CAPABILITY_PROFILE15),
    )
    target, pre_tanh, normalized, log_prob, mean, log_std, hidden = actor.sample(actor_input)
    value, critic_hidden = critic(critic_input)
    assert target.shape == (batch, 3)
    assert pre_tanh.shape == normalized.shape == mean.shape == log_std.shape == (batch, 3)
    assert log_prob.shape == value.shape == (batch, 1)
    assert hidden[0].shape == critic_hidden[0].shape == (2, batch, 64)
    assert torch.isfinite(target).all()
    assert torch.all(target[:, 1].abs() <= 0.40)


def _transition(num_envs, *, reward, value, bootstrap, duration, bootstrap_mask, continuation):
    hidden = (
        torch.zeros(2, num_envs, 64),
        torch.zeros(2, num_envs, 64),
    )
    return {
        "nav_feat": torch.zeros(num_envs, 32),
        "nav_nonvisual": torch.zeros(num_envs, 36),
        "response_profile": torch.zeros(num_envs, 16),
        "confidence": torch.ones(num_envs, 1),
        "safety_target": torch.zeros(num_envs, 3),
        "safety_valid": torch.ones(num_envs, 1),
        "critic_input": torch.zeros(num_envs, 341),
        "pre_tanh_action": torch.zeros(num_envs, 3),
        "old_log_prob": torch.zeros(num_envs, 1),
        "old_value": torch.full((num_envs, 1), value),
        "reward": torch.full((num_envs, 1), reward),
        "duration_frames": torch.full((num_envs, 1), duration),
        "bootstrap_value": torch.full((num_envs, 1), bootstrap),
        "bootstrap_mask": torch.full((num_envs, 1), bootstrap_mask),
        "continuation_mask": torch.full((num_envs, 1), continuation),
        "reset_mask": torch.zeros(num_envs, dtype=torch.bool),
        "actor_hidden": hidden,
        "critic_hidden": hidden,
    }


def test_variable_duration_gae_supports_an_explicit_terminal_bootstrap_value():
    storage = P2RolloutStorage(1, num_ticks=2, sequence_length=1, store_depth=False)
    storage.add(**_transition(1, reward=1.0, value=0.2, bootstrap=0.8, duration=5, bootstrap_mask=1.0, continuation=0.0))
    storage.add(**_transition(1, reward=2.0, value=0.4, bootstrap=9.0, duration=10, bootstrap_mask=0.0, continuation=0.0))
    storage.compute_returns()
    discount = p2_contract.GAMMA_FRAME ** 5
    expected_first = 1.0 + discount * 0.8
    expected_second = 2.0
    assert torch.allclose(storage.returns[0], torch.tensor([[expected_first]]), atol=1e-6)
    assert torch.allclose(storage.returns[1], torch.tensor([[expected_second]]), atol=1e-6)


def test_invalid_transition_is_excluded_from_gae_and_return_target():
    storage = P2RolloutStorage(1, num_ticks=2, sequence_length=1, store_depth=False)
    first = _transition(
        1,
        reward=1.0,
        value=0.2,
        bootstrap=0.8,
        duration=5,
        bootstrap_mask=1.0,
        continuation=0.0,
    )
    second = _transition(
        1,
        reward=999.0,
        value=7.0,
        bootstrap=999.0,
        duration=10,
        bootstrap_mask=1.0,
        continuation=1.0,
    )
    second["valid_mask"] = torch.zeros(1, 1)
    storage.add(**first)
    storage.add(**second)
    storage.compute_returns()

    discount = p2_contract.GAMMA_FRAME ** 5
    assert torch.allclose(
        storage.returns[0], torch.tensor([[1.0 + discount * 0.8]]), atol=1e-6
    )
    assert storage.advantages[1].item() == 0.0
    assert storage.returns[1].item() == pytest.approx(7.0)


class _ConstantCritic:
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, inputs, hidden=None, reset_mask=None):
        return torch.full((inputs.shape[0], 1), self.value), hidden


def test_timeout_never_bootstraps_from_post_reset_observation():
    algorithm = AlgorithmP2NavPPO.__new__(AlgorithmP2NavPPO)
    algorithm.device = torch.device("cpu")
    algorithm.num_envs = 1
    algorithm.nav_period_frames = p2_contract.NAV_PERIOD_FRAMES
    algorithm.command = SimpleNamespace(
        active_target=torch.tensor([[0.5, 0.0, 0.0]]),
        exec_cmd=torch.tensor([[0.2, 0.0, 0.0]]),
        inject=lambda _obs, _critic: None,
    )
    algorithm.critic = _ConstantCritic(123.0)
    algorithm.critic_hidden = None
    algorithm.reset_since_tick = torch.ones(1, dtype=torch.bool)
    algorithm.rollout = _TransitionCapture()
    algorithm.rollout_invalid = False
    algorithm.invalid_transition_count = 0
    algorithm.effective_training_seconds = p2_contract.SAFETY_WARM_END_SECONDS
    algorithm.gait_baseline = P2GaitBaseline()
    algorithm.gait_baseline.finalize()
    algorithm.best_goal_distance = torch.full((1,), float("inf"))
    algorithm.pending_tick = {
        "command_penalty": torch.tensor([[-0.1]]),
        "tick_penalty": torch.tensor([[-0.1]]),
        "target_cmd3": torch.tensor([[0.5, 0.0, 0.0]]),
    }
    wire = torch.zeros(1, p2_contract.PRIVILEGED_WIRE_DIM)
    wire[:, p2_contract.CRITIC_OBS_DIM + 24] = 1.0
    algorithm.finish_tick(
        torch.zeros(1, nav_contract.POLICY_OBS_DIM),
        wire,
        frame_safety_reward=torch.zeros(1),
        start_goal_distance=torch.ones(1),
        end_goal_distance=torch.ones(1),
        terminal_reason=torch.full((1,), 3, dtype=torch.long),
        duration_frames=torch.ones(1, dtype=torch.long),
        hard_terminated=torch.zeros(1, dtype=torch.bool),
        timeout=torch.ones(1, dtype=torch.bool),
    )
    transition = algorithm.rollout.transition
    assert transition["bootstrap_mask"].item() == 0.0
    assert transition["continuation_mask"].item() == 0.0
    assert transition["bootstrap_value"].item() == 0.0
    assert torch.allclose(transition["reward"], torch.tensor([[-22.602]]))


class _TerminationManager:
    active_terms = ("goal_reached",)

    def __init__(self, n):
        self.terminated = torch.zeros(n, dtype=torch.bool)
        self.time_outs = torch.zeros(n, dtype=torch.bool)
        self.goal = torch.zeros(n, dtype=torch.bool)

    def get_term(self, name):
        assert name == "goal_reached"
        return self.goal


def test_curriculum_probe_compares_old_and_new_row_column(capsys):
    n = 2
    terrain = SimpleNamespace(
        terrain_levels=torch.tensor((0, 0)),
        terrain_types=torch.tensor((1, 2)),
        cfg=SimpleNamespace(
            max_init_terrain_level=0,
            terrain_generator=SimpleNamespace(
                curriculum=True,
                track_length=3,
                num_rows=3,
                num_parallel_tracks=20,
                num_cols=20,
                sub_terrains_order=(
                    "pyramid_slope_inv",
                    "pyramid_stairs_inv",
                    "open_entry_maze",
                ),
            ),
        ),
    )
    terms = _TerminationManager(n)
    env = SimpleNamespace(
        scene=SimpleNamespace(terrain=terrain),
        curriculum_manager=SimpleNamespace(
            _term_names=("terrain_levels", "lin_vel_cmd_levels", "ang_vel_cmd_levels")
        ),
        termination_manager=terms,
        episode_length_buf=torch.ones(n, dtype=torch.long),
        _p2_last_goal_progress=torch.tensor((0.1, -0.2)),
    )
    probe = P2TrackCurriculumProbe(n, report_interval_s=1.0)
    probe.observe(env, now_s=0.0)
    terrain.terrain_levels[:] = torch.tensor((1, 0))
    terrain.terrain_types[:] = torch.tensor((1, 3))
    env.episode_length_buf[:] = torch.tensor((0, 0))
    terms.terminated[:] = True
    terms.goal[0] = True
    probe.observe(env, now_s=2.0)
    output = capsys.readouterr().out
    assert (
        '"curriculum_active_terms": ["terrain_levels", "lin_vel_cmd_levels", '
        '"ang_vel_cmd_levels"]' in output
    )
    assert '"old_row": [0, 0]' in output
    assert '"new_row": [1, 0]' in output
    assert '"termination_reason": ["success", "failure"]' in output
    assert probe.row_moves.tolist() == [0, 1, 1]
    assert probe.col_moves.tolist() == [1, 1]


def test_curriculum_probe_waits_for_track_column_initialization(capsys):
    n = 2
    terrain = SimpleNamespace(
        terrain_levels=torch.zeros(n, dtype=torch.long),
        terrain_types=torch.tensor((0, 1)),
        cfg=SimpleNamespace(
            max_init_terrain_level=0,
            terrain_generator=SimpleNamespace(curriculum=True, num_cols=10),
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(terrain=terrain),
        curriculum_manager=SimpleNamespace(_term_names=("terrain_levels",)),
        episode_length_buf=torch.zeros(n, dtype=torch.long),
    )
    probe = P2TrackCurriculumProbe(n)

    probe.observe(env, now_s=0.0)
    assert probe.initialized is False
    assert probe.previous_rows is None
    assert capsys.readouterr().out == ""

    terrain._track_curriculum_col_initialized = torch.ones(n, dtype=torch.bool)
    terrain.terrain_types[:] = 0
    probe.observe(env, now_s=1.0)
    output = capsys.readouterr().out
    assert probe.initialized is True
    assert output.count('"event": "p2_curriculum_init"') == 1
    assert '"initial_cols": [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' in output


def test_curriculum_probe_termination_terms_fall_back_to_private_names():
    manager = _TerminationManager(1)
    manager.active_terms = None
    manager._term_names = ("goal_reached",)
    manager.terminated[:] = True
    manager.goal[:] = True
    env = SimpleNamespace(termination_manager=manager)

    reasons = P2TrackCurriculumProbe._termination_reasons(env, torch.tensor((0,)))
    assert reasons == ["success"]


def test_curriculum_snapshot_flattens_to_monitor_metrics():
    outcomes = torch.zeros(3, p2_contract.TERRAIN_NUM_COLUMNS, 3, dtype=torch.long)
    outcomes[0, 0] = torch.tensor((1, 2, 3))
    outcomes[1, 0] = torch.tensor((4, 5, 6))
    outcomes[2, 0] = torch.tensor((7, 8, 9))
    starts = torch.zeros(3, p2_contract.TERRAIN_NUM_COLUMNS, dtype=torch.long)
    starts[0, :2] = torch.tensor((2, 3))
    starts[1, :2] = torch.tensor((4, 5))
    starts[2, :2] = torch.tensor((6, 7))
    snapshot = {
        "row_moves": [2, 3, 4],
        "col_moves": [5, 6],
        "outcomes": outcomes.tolist(),
        "start_counts": starts.tolist(),
    }
    metrics = _curriculum_metrics(snapshot)
    assert metrics["curriculum_row_promotions"] == 4.0
    assert metrics["curriculum_column_changed"] == 6.0
    assert metrics["curriculum_successes"] == 12.0
    assert metrics["curriculum_failures"] == 15.0
    assert metrics["curriculum_timeouts"] == 18.0
    assert metrics["curriculum_slope_inv_starts"] == 5.0
    assert metrics["curriculum_stairs_inv_starts"] == 9.0
    assert metrics["curriculum_maze_entry_starts"] == 13.0
    assert _curriculum_metrics({"row_moves": []}) == {}


def test_aisrv_curriculum_accumulator_uses_aux_reset_reason_and_resumes():
    accumulator = P2CurriculumAccumulator()
    first = torch.zeros(2, p2_contract.WORKER_AUX_DIM)
    first[:, 28] = torch.tensor((1.0, 2.0))
    first[:, 29] = 0.0
    accumulator.observe(first)
    reset = first.clone()
    reset[:, 24] = 1.0
    reset[:, 25] = torch.tensor((1.0, 3.0))
    reset[:, 28] = torch.tensor((1.0, 3.0))
    reset[:, 29] = torch.tensor((1.0, 0.0))
    reset[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX] = torch.tensor((1.0, 2.0))
    reset[:, p2_contract.PRE_STEP_TERRAIN_LEVEL_INDEX] = 0.0
    accumulator.observe(reset)
    state = accumulator.state_dict()
    assert state["transport"] == "p2_worker_aux62_v3_20cols"
    assert state["row_moves"] == [0, 1, 1]
    assert state["col_moves"] == [1, 1]
    assert state["outcomes"][0][1][0] == 1
    assert state["outcomes"][0][2][2] == 1
    restored = P2CurriculumAccumulator()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    # Per-environment previous row/column is intentionally not checkpointed.
    restored.observe(reset)
    assert restored.state_dict()["row_moves"] == [0, 1, 1]


class _CaptureLogger:
    def __init__(self):
        self.infos = []

    def info(self, message):
        self.infos.append(message)


def test_aisrv_curriculum_reset_logs_are_aggregated_once_per_minute():
    accumulator = P2CurriculumAccumulator()
    logger = _CaptureLogger()
    first = torch.zeros(2, p2_contract.WORKER_AUX_DIM)
    accumulator.observe(first, logger=logger, elapsed_s=0.0)

    reset = first.clone()
    reset[:, 24] = 1.0
    reset[:, 25] = torch.tensor((2.0, 3.0))
    reset[:, p2_contract.PRE_STEP_TERRAIN_TYPE_INDEX] = 0.0
    reset[:, p2_contract.PRE_STEP_TERRAIN_LEVEL_INDEX] = 0.0
    accumulator.observe(reset, logger=logger, elapsed_s=1.0)
    assert logger.infos == []

    settled = reset.clone()
    settled[:, 24] = 0.0
    settled[:, 25] = 0.0
    accumulator.observe(settled, logger=logger, elapsed_s=61.0)
    assert len(logger.infos) == 1
    assert '"event": "p2_curriculum_reset_summary_aisrv"' in logger.infos[0]
    assert '"reset_envs": 2' in logger.infos[0]
    assert '"failure": 1' in logger.infos[0]
    assert '"timeout": 1' in logger.infos[0]


def test_worker_reason_codes_do_not_classify_unknown_reset_as_timeout():
    manager = _TerminationManager(4)
    manager.terminated[:] = torch.tensor((True, True, False, False))
    manager.time_outs[:] = torch.tensor((False, False, True, False))
    manager.goal[:] = torch.tensor((True, False, False, False))
    env = SimpleNamespace(termination_manager=manager)
    reasons = _termination_reason_codes(env, torch.ones(4, dtype=torch.bool))
    assert reasons.tolist() == [1.0, 2.0, 3.0, 0.0]


def test_worker_terminal_reason_priority_preserves_success_and_hard_failure():
    manager = _TerminationManager(4)
    manager.terminated[:] = torch.tensor((False, True, False, False))
    manager.time_outs[:] = True
    manager.goal[:] = torch.tensor((True, False, False, False))
    env = SimpleNamespace(termination_manager=manager)
    reasons = _termination_reason_codes(
        env,
        torch.ones(4, dtype=torch.bool),
        wall_stuck=torch.tensor((True, True, True, False)),
    )
    assert reasons.tolist() == [1.0, 2.0, 4.0, 3.0]


def test_terminal_reason_keeps_success_and_timeout_mutually_exclusive():
    new_done = torch.tensor((True, True, True, True))
    wrapper_timeout = torch.tensor((True, True, False, True))
    raw_reason = torch.tensor((1, 0, 9, 4))
    reason, hard, timeout = _resolve_terminal_outcome(
        new_done,
        wrapper_timeout,
        raw_reason,
    )
    assert reason.tolist() == [1, 3, 0, 4]
    assert hard.tolist() == [True, False, False, False]
    assert timeout.tolist() == [False, True, False, True]
    assert not bool((hard & timeout).any())


def test_worker_reason_codes_find_goal_term_through_private_term_names():
    manager = _TerminationManager(1)
    manager.active_terms = None
    manager._term_names = ("goal_reached",)
    manager.terminated[:] = True
    manager.goal[:] = True
    env = SimpleNamespace(termination_manager=manager)
    reasons = _termination_reason_codes(env, torch.ones(1, dtype=torch.bool))
    assert reasons.tolist() == [1.0]


@pytest.mark.parametrize(
    ("reason", "expected_terminated", "expected_truncated"),
    (
        (1, [True, False, False], [False, False, False]),
        (2, [False, True, False], [False, False, False]),
        (3, [False, False, False], [False, False, True]),
    ),
)
def test_worker_terminal_snapshot_restores_wrapper_return(
    reason,
    expected_terminated,
    expected_truncated,
):
    aux = torch.zeros(3, p2_contract.WORKER_AUX_DIM)
    env_id = reason - 1
    aux[env_id, 24] = 1.0
    aux[env_id, 25] = float(reason)
    terminated, truncated = _merge_worker_terminal_returns(
        torch.zeros(3, dtype=torch.bool),
        torch.zeros(3, dtype=torch.bool),
        aux,
    )
    assert terminated.tolist() == expected_terminated
    assert truncated.tolist() == expected_truncated


def test_worker_terminal_snapshot_preserves_native_done_and_ignores_initial_reset():
    aux = torch.zeros(2, p2_contract.WORKER_AUX_DIM)
    aux[0, 24] = 1.0
    terminated, truncated = _merge_worker_terminal_returns(
        torch.tensor((False, True)),
        torch.tensor((True, False)),
        aux,
    )
    assert terminated.tolist() == [False, True]
    assert truncated.tolist() == [True, False]


def test_p2_terminal_return_bridge_is_idempotent_and_model_id_independent():
    class _FakeP2Env:
        def __init__(self):
            self.calls = 0
            aux = torch.zeros(2, p2_contract.WORKER_AUX_DIM)
            aux[0, 24] = 1.0
            aux[0, 25] = 1.0
            self._agent_ppo_p2_worker_bridge = SimpleNamespace(last_aux=aux)

        def step(self, _actions):
            self.calls += 1
            return (
                torch.zeros(2, 1),
                torch.zeros(2),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                {},
            )

    env = _FakeP2Env()
    assert install_p2_terminal_return_bridge(env) is True
    assert install_p2_terminal_return_bridge(env) is False
    _, _, terminated, truncated, _ = env.step(torch.zeros(2, 1))
    assert env.calls == 1
    assert terminated.tolist() == [True, False]
    assert truncated.tolist() == [False, False]


def _response_aux(num_envs=2):
    aux = torch.zeros(num_envs, 30)
    aux[:, 9] = 1.0
    return aux


def test_p2_response_records_keep_parent_and_track_capability_semantics():
    buffer = P2ResponseAuxBuffer(2, "cpu")
    parent = buffer._capability_for_record({"record_origin": "parent"})
    track = buffer._capability_for_record({"record_origin": "track"})
    assert torch.equal(parent, torch.tensor(P15_CAPABILITY_PROFILE15))
    assert torch.equal(track, torch.tensor(p2_contract.RESPONSE_CAPABILITY_PROFILE15))
    assert not torch.equal(track, torch.tensor(p2_contract.NAV_CAPABILITY_PROFILE15))


def test_p2_response_done_invalidates_current_future_and_resets_following_record():
    buffer = P2ResponseAuxBuffer(2, "cpu")
    for step in range(52):
        aux = _response_aux()
        aux[:, 12] = float(step)
        done = torch.zeros(2, dtype=torch.bool)
        if step == 0:
            done[0] = True
        buffer.append(aux, done)
    first, second = list(buffer._records)[:2]
    assert first["horizon_mask"][0].tolist() == [False, False, False]
    assert first["horizon_mask"][1].tolist() == [True, True, True]
    assert first["episode_start"].tolist() == [True, True]
    assert second["episode_start"].tolist() == [True, False]


def test_p2_stuck_label_separates_translation_and_yaw_response():
    buffer = P2ResponseAuxBuffer(2, "cpu")
    for step in range(51):
        aux = _response_aux()
        if step == 0:
            aux[0, 3] = 0.25
            aux[1, 5] = 0.30
        # Both environments can yaw. Env 0 still fails its commanded vx;
        # Env 1 accumulates enough yaw displacement to be a healthy turn.
        aux[:, 14] = 0.30
        aux[1, 17] = 0.003 * step
        buffer.append(aux, torch.zeros(2, dtype=torch.bool))
    record = buffer._records[0]
    assert record["stuck"].reshape(-1).tolist() == [1.0, 0.0]


def test_p2_response_buffer_is_cpu_backed_and_resume_clears_future_history():
    buffer = P2ResponseAuxBuffer(2, "cpu")
    for _ in range(51):
        buffer.append(_response_aux(), torch.zeros(2, dtype=torch.bool))
    state = buffer.checkpoint_state()
    restored = P2ResponseAuxBuffer(2, "cpu")
    assert restored.load_checkpoint_state(state) == "completed_records_restored_history_reset"
    assert restored._records[0]["aux"].device.type == "cpu"
    assert len(restored._history_aux) == 0
    assert restored._next_episode_start.all()


def test_p3_response_replay_ratios_validate_and_resume():
    buffer = P2ResponseAuxBuffer(2, "cpu")
    buffer.replay_policy = "p3_versioned_50_25_25"
    buffer.set_p3_replay_ratios(0.75, 0.15, 0.10)
    state = buffer.checkpoint_state()
    restored = P2ResponseAuxBuffer(2, "cpu")
    restored.load_checkpoint_state(state)
    assert restored.p3_replay_ratios == pytest.approx((0.75, 0.15, 0.10))
    with pytest.raises(ValueError, match="sum to one"):
        restored.set_p3_replay_ratios(0.5, 0.5, 0.5)


def test_p2_adapter_update_reports_row_domain_and_confidence_metrics():
    algorithm = _make_p2_algorithm(training=True)
    for step in range(80):
        aux = torch.zeros(1, p2_contract.RESPONSE_AUX_DIM)
        aux[:, 0] = 0.5
        aux[:, 3] = 0.4
        aux[:, 6] = 0.4
        aux[:, 9] = 1.0
        aux[:, 12] = 0.4 + 0.001 * step
        aux[:, 28] = 2.0
        aux[:, 29] = 2.0
        algorithm.response_buffer.append(
            aux,
            torch.zeros(1, dtype=torch.bool),
            current_segment=torch.tensor((2.0,)),
        )
    metrics = algorithm._adapter_update()
    assert metrics["adapter_updates"] == 1.0
    assert metrics["adapter_track_records"] == 1.0
    assert metrics["adapter_parent_records"] == 0.0
    assert metrics["adapter_maze_sample_share"] == 1.0
    assert metrics["adapter_core_sample_share"] == 1.0
    assert metrics["adapter_outer_sample_share"] == 0.0
    for name in (
        "adapter_maze_mae",
        "adapter_core_mae",
        "adapter_confidence_low_mae",
        "adapter_confidence_mid_mae",
        "adapter_confidence_high_mae",
    ):
        assert math.isfinite(metrics[name]), name


def test_adapter_group_share_ignores_invalid_future_horizons():
    group = torch.tensor(((False, True),))
    horizon_mask = torch.tensor((((True, True, True), (False, False, False)),))
    share = AlgorithmP2NavPPO._masked_group_share(group, horizon_mask)
    assert share.item() == 0.0


def test_p2_completed_records_can_shrink_for_environment_fallback():
    source = P2ResponseAuxBuffer(4, "cpu")
    for _ in range(51):
        source.append(_response_aux(4), torch.zeros(4, dtype=torch.bool))
    restored = P2ResponseAuxBuffer(2, "cpu")
    assert restored.load_checkpoint_state(source.checkpoint_state()) == (
        "completed_records_restored_history_reset"
    )
    assert restored._records[0]["aux"].shape == (2, 30)
    assert restored.resized_completed_records == 1


def test_p2_rollout_sequence_counts_support_fallback_environment_sizes():
    for num_envs in (128, 96, 80):
        storage = P2RolloutStorage(
            num_envs,
            num_ticks=32,
            sequence_length=16,
            store_depth=False,
        )
        assert len(storage.sequence_refs(torch.Generator().manual_seed(7))) == 2 * num_envs


def test_p2_depth_storage_flattens_camera_layout_without_shape_drift():
    storage = P2RolloutStorage(
        1,
        num_ticks=1,
        sequence_length=1,
        store_depth=True,
        pin_memory=False,
    )
    transition = _transition(
        1,
        reward=0.0,
        value=0.0,
        bootstrap=0.0,
        duration=10,
        bootstrap_mask=1.0,
        continuation=1.0,
    )
    transition["depth"] = torch.arange(
        p2_contract.DEPTH_DIM, dtype=torch.float32
    ).reshape(1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1)
    storage.add(**transition)
    assert storage.depth.shape == (1, 1, p2_contract.DEPTH_DIM)
    assert storage.depth[0, 0, -1] == p2_contract.DEPTH_DIM - 1


def test_timeout_info_advances_done_even_if_wrapper_truncated_is_false():
    done, timeout = _frame_done_masks(
        torch.tensor((False, True)),
        torch.tensor((False, False)),
        {"time_outs": torch.tensor((True, False))},
        "cpu",
    )
    assert done.tolist() == [True, True]
    assert timeout.tolist() == [True, False]


def test_worker_aux_recovers_timeout_when_wrapper_erases_truncated_and_infos():
    aux = torch.zeros(2, 30)
    aux[0, 24] = 1.0
    aux[0, 25] = 3.0
    aux[1, 24] = 1.0
    aux[1, 25] = 2.0
    done, timeout = _frame_done_masks(
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        {},
        "cpu",
        worker_aux=aux,
    )
    assert done.tolist() == [True, True]
    assert timeout.tolist() == [True, False]


def test_worker_reason_zero_is_an_unattributed_boundary_not_timeout():
    aux = torch.zeros(1, 30)
    aux[0, 24] = 1.0
    done, timeout = _frame_done_masks(
        torch.zeros(1, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        {"time_outs": torch.ones(1, dtype=torch.bool)},
        "cpu",
        worker_aux=aux,
    )
    reason, hard, attributed_timeout = _resolve_terminal_outcome(
        done, timeout, aux[:, 25].long()
    )
    assert done.item()
    assert not timeout.item()
    assert reason.item() == 0
    assert not hard.item()
    assert not attributed_timeout.item()


class _SaveLogger:
    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class _SaveAgent:
    def __init__(self):
        self.calls = 0
        self._p2_final_save_done = False

    def save_model(self):
        self.calls += 1


def test_p2_final_platform_save_is_no_argument_and_idempotent():
    agent = _SaveAgent()
    assert _final_save(agent, _SaveLogger(), "normal_end")
    assert not _final_save(agent, _SaveLogger(), "duplicate")
    assert agent.calls == 1


class _EmptyRollout:
    step = 0

    def __init__(self):
        self.store_depth = False

    def reset(self, *, store_depth):
        self.store_depth = bool(store_depth)
        return self


class _FakeScheduler:
    def __init__(self):
        self.base_lrs = [0.0] * 4 + [p2_contract.ACTOR_LR] * 3 + [p2_contract.SAFETY_HEAD_LR]
        self._last_lr = list(self.base_lrs)


def test_safe_direction_schedule_trains_cnn_from_first_rollout():
    algorithm = AlgorithmP2NavPPO.__new__(AlgorithmP2NavPPO)
    algorithm.effective_training_seconds = 1190.0
    algorithm.lifetime_base_seconds = 28761.0
    algorithm.cnn_unfrozen = False
    algorithm.rollout = _EmptyRollout()
    algorithm.logger = None
    algorithm.actor_optimizer = SimpleNamespace(
        param_groups=[
            {"name": "navigation_conv1", "lr": 0.0},
            {"name": "navigation_conv2", "lr": 0.0},
            {"name": "navigation_conv3", "lr": 0.0},
            {"name": "navigation_fc", "lr": 0.0},
            {"name": "actor_trunk", "lr": p2_contract.ACTOR_LR},
            {"name": "actor_main", "lr": p2_contract.ACTOR_LR},
            {"name": "actor_vy", "lr": p2_contract.ACTOR_LR},
            {"name": "navigation_safety_head", "lr": p2_contract.SAFETY_HEAD_LR},
        ]
    )
    algorithm.actor_scheduler = _FakeScheduler()
    algorithm.critic_optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])
    algorithm.response_optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])
    algorithm.critic_scheduler = None
    algorithm.response_scheduler = None
    algorithm.gait_baseline = P2GaitBaseline()
    assert algorithm.maybe_unfreeze_cnn(0.0)
    assert not algorithm.maybe_unfreeze_cnn(1200.0)
    assert algorithm.cnn_unfrozen
    assert algorithm.rollout.store_depth
    assert [
        group["lr"] for group in algorithm.actor_optimizer.param_groups[:4]
    ] == [
        0.15 * p2_contract.CNN_LAYER_LRS["conv1"],
        0.15 * p2_contract.CNN_LAYER_LRS["conv2"],
        0.15 * p2_contract.CNN_LAYER_LRS["conv3"],
        0.15 * p2_contract.CNN_LAYER_LRS["fc"],
    ]
    assert algorithm.actor_scheduler.base_lrs[:4] == [
        p2_contract.CNN_LAYER_LRS["conv1"],
        p2_contract.CNN_LAYER_LRS["conv2"],
        p2_contract.CNN_LAYER_LRS["conv3"],
        p2_contract.CNN_LAYER_LRS["fc"],
    ]


def test_cnn_unfreeze_waits_for_rollout_boundary():
    algorithm = AlgorithmP2NavPPO.__new__(AlgorithmP2NavPPO)
    algorithm.effective_training_seconds = 1190.0
    algorithm.lifetime_base_seconds = 28761.0
    algorithm.cnn_unfrozen = False
    algorithm.rollout = _EmptyRollout()
    algorithm.rollout.step = 1
    algorithm.logger = None
    algorithm.actor_optimizer = SimpleNamespace(
        param_groups=[
            {"name": "navigation_conv1", "lr": 0.0},
            {"name": "navigation_conv2", "lr": 0.0},
            {"name": "navigation_conv3", "lr": 0.0},
            {"name": "navigation_fc", "lr": 0.0},
            {"name": "actor_trunk", "lr": p2_contract.ACTOR_LR},
            {"name": "actor_main", "lr": p2_contract.ACTOR_LR},
            {"name": "actor_vy", "lr": p2_contract.ACTOR_LR},
        ]
    )
    algorithm.actor_scheduler = _FakeScheduler()
    algorithm.critic_optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])
    algorithm.response_optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])
    algorithm.critic_scheduler = None
    algorithm.response_scheduler = None
    algorithm.gait_baseline = P2GaitBaseline()
    assert not algorithm.maybe_unfreeze_cnn(1200.0)
    assert not algorithm.cnn_unfrozen
    assert all(
        group["lr"] == 0.0
        for group in algorithm.actor_optimizer.param_groups[:4]
    )
    algorithm.rollout.step = 0
    assert algorithm.maybe_unfreeze_cnn(1200.0)
    assert algorithm.cnn_unfrozen


def test_eval_command_clock_advances_one_low_level_frame_per_call():
    command = P2CommandController(1, "cpu", slew_rate=(0.30, 0.30, 1.00))
    command.set_target(torch.tensor(((1.0, 0.0, -1.0),)))
    command.step()
    assert torch.allclose(
        command.exec_cmd,
        torch.tensor(((0.006, 0.0, -0.02),)),
        atol=1.0e-7,
    )


def test_vy_wz_reversal_releases_to_zero_without_overshoot():
    command = P2CommandController(
        1,
        "cpu",
        slew_rate=(0.30, 0.30, 1.00),
        slew_release_rate=(0.30, 0.60, 2.50),
    )
    command.set_target(torch.tensor(((0.0, 0.4, 1.0),)))
    for _ in range(100):
        command.step()
    command.set_target(torch.tensor(((0.0, -0.4, -1.0),)))
    history = []
    for _ in range(100):
        before = command.exec_cmd.clone()
        command.step()
        after = command.exec_cmd.clone()
        assert not bool(((before[:, 1:] * after[:, 1:]) < 0.0).any())
        history.append(after.clone())
    stacked = torch.cat(history, dim=0)
    assert bool((stacked[:, 1] == 0.0).any())
    assert bool((stacked[:, 2] == 0.0).any())
    assert command.exec_cmd[0, 1].item() < 0.0
    assert command.exec_cmd[0, 2].item() < 0.0


def _low_actor():
    return nn.Sequential(
        nn.Linear(77, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 12),
    )


def _make_p2_algorithm(*, training=True, config=None):
    algorithm_config = {
        "ppo_seed": 7,
        "action_seed": 11,
        "neutral_seed": 13,
    }
    algorithm_config.update(config or {})
    return AlgorithmP2NavPPO(
        low_level_encoder=VisionEncoder(),
        low_level_actor=_low_actor(),
        navigation_encoder=NavigationEncoder(),
        safety_head=NavigationSafetyHead() if training else None,
        actor=P2NavigationActor(),
        critic=P2NavigationCritic() if training else None,
        response_adapter=CommandResponseAdapter(),
        response_buffer=P2ResponseAuxBuffer(1, "cpu") if training else None,
        num_envs=1,
        device="cpu",
        config=algorithm_config,
        training=training,
    )


def test_ppo_epoch_count_is_read_from_configuration():
    algorithm = _make_p2_algorithm(
        training=True, config={"num_learning_epochs": 3}
    )
    assert algorithm.num_learning_epochs == 3


def test_actor_micro_loss_casts_fp16_depth_for_cpu_smoke():
    algorithm = _make_p2_algorithm(training=True)
    hidden = (
        torch.zeros(2, 1, 64),
        torch.zeros(2, 1, 64),
    )
    loss, metrics = algorithm._actor_micro_loss(
        {
            "depth": torch.zeros(1, 1, p2_contract.DEPTH_DIM, dtype=torch.float16),
            "nav_nonvisual": torch.zeros(1, 1, p2_contract.NAV_NONVISUAL_DIM),
            "response_profile": torch.zeros(1, 1, p2_contract.RESPONSE_PROFILE_DIM),
            "confidence": torch.ones(1, 1, 1),
            "pre_tanh_action": torch.zeros(1, 1, p2_contract.ACTION_DIM),
            "actor_hidden": hidden,
            "reset_mask": torch.zeros(1, 1, dtype=torch.bool),
            "old_log_prob": torch.zeros(1, 1, 1),
            "advantages": torch.ones(1, 1, 1),
            "safety_target": torch.zeros(1, 1, 3),
            "safety_valid": torch.ones(1, 1, 1),
        }
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["safety_bce"])


def test_p2_training_monitor_metrics_report_rollout_and_optimizer_state():
    algorithm = _make_p2_algorithm(training=True)
    algorithm.rollout.rewards.fill_(2.0)
    algorithm.rollout.returns.fill_(3.0)
    algorithm.rollout.old_value.fill_(1.0)
    algorithm.rollout.advantages.fill_(2.0)
    with torch.no_grad():
        algorithm.actor.log_std.copy_(torch.log(torch.tensor((0.5, 0.25))))

    metrics = algorithm._training_monitor_metrics()
    assert metrics["rollout_reward_mean"] == 2.0
    assert metrics["rollout_reward_std"] == 0.0
    assert metrics["rollout_return_mean"] == 3.0
    assert metrics["rollout_value_mean"] == 1.0
    assert metrics["rollout_advantage_mean"] == 2.0
    assert metrics["rollout_advantage_std"] == 0.0
    assert math.isclose(metrics["action_std_vx"], 0.5, rel_tol=1.0e-6)
    assert math.isclose(metrics["action_std_wz"], 0.25, rel_tol=1.0e-6)
    assert metrics["action_std_vy"] == pytest.approx(math.exp(-1.1))
    assert metrics["actor_learning_rate"] == pytest.approx(
        p2_contract.ACTOR_LR * 0.10
    )
    assert metrics["vy_actor_learning_rate"] == pytest.approx(
        p2_contract.ACTOR_LR * 0.10
    )
    assert metrics["critic_learning_rate"] == pytest.approx(
        0.30 * p2_contract.CRITIC_LR
    )
    assert metrics["adapter_learning_rate"] == 0.5 * p2_contract.ADAPTER_LR


def test_advantage_normalization_uses_the_entire_rollout_once():
    algorithm = _make_p2_algorithm(training=True)
    algorithm.rollout.step = algorithm.rollout.num_ticks
    values = torch.arange(
        algorithm.rollout.num_ticks * algorithm.num_envs,
        dtype=torch.float32,
    ).reshape(algorithm.rollout.num_ticks, algorithm.num_envs, 1)
    algorithm.rollout.advantages.copy_(values)
    mean, std = algorithm._rollout_advantage_stats()
    assert mean == pytest.approx(float(values.mean()))
    assert std == pytest.approx(float(values.std(unbiased=False)))


def test_nonfinite_policy_output_is_sanitized_and_marks_rollout_skippable():
    algorithm = _make_p2_algorithm(training=True)

    def nonfinite_sample(
        inputs,
        hidden=None,
        reset_mask=None,
        generator=None,
        vy_generator=None,
        hard_abs_vy=0.40,
    ):
        del reset_mask, generator, vy_generator, hard_abs_vy
        batch = inputs.shape[0]
        bad3 = torch.full((batch, 3), float("nan"))
        hidden = (
            torch.ones(2, batch, 64),
            torch.ones(2, batch, 64),
        ) if hidden is None else hidden
        return bad3, bad3, bad3, bad3[:, :1], bad3, bad3, hidden

    algorithm.actor.sample = nonfinite_sample
    obs = torch.zeros(1, nav_contract.POLICY_OBS_DIM)
    wire = torch.zeros(1, p2_contract.PRIVILEGED_WIRE_DIM)
    wire[:, p2_contract.CRITIC_OBS_DIM + 9] = 1.0
    algorithm.frame_begin(obs, wire)
    assert algorithm.rollout_invalid
    assert algorithm.invalid_transition_count == 1
    assert torch.equal(algorithm.command.active_target, torch.zeros(1, 3))
    for name in (
        "depth",
        "nav_feat",
        "nav_nonvisual",
        "response_profile",
        "confidence",
        "critic_input",
        "pre_tanh_action",
        "old_log_prob",
        "old_value",
    ):
        assert torch.isfinite(algorithm.pending_tick[name]).all(), name
    for hidden_name in ("actor_hidden", "critic_hidden"):
        assert all(
            torch.isfinite(value).all()
            for value in algorithm.pending_tick[hidden_name]
        )


def test_tick_depth_owns_storage_when_environment_reuses_observation_buffer():
    algorithm = _make_p2_algorithm(training=True)
    obs = torch.zeros(1, nav_contract.POLICY_OBS_DIM)
    obs[:, nav_contract.DEPTH_OBS_START:] = 0.25
    wire = torch.zeros(1, p2_contract.PRIVILEGED_WIRE_DIM)
    wire[:, p2_contract.CRITIC_OBS_DIM + 9] = 1.0
    algorithm.frame_begin(obs, wire)
    assert algorithm.pending_tick["depth"].device.type == "cpu"
    assert algorithm.pending_tick["depth"].dtype == torch.float16
    obs[:, nav_contract.DEPTH_OBS_START:] = 0.75
    assert torch.all(algorithm.pending_tick["depth"] == 0.25)


def test_frozen_navigation_encoder_tick_stores_feature_without_depth_slot():
    algorithm = _make_p2_algorithm(training=True)
    algorithm.cnn_unfrozen = False
    algorithm.rollout = algorithm.rollout.reset(store_depth=False)
    obs = torch.zeros(1, nav_contract.POLICY_OBS_DIM)
    obs[:, nav_contract.DEPTH_OBS_START:] = 0.25
    wire = torch.zeros(1, p2_contract.PRIVILEGED_WIRE_DIM)
    wire[:, p2_contract.CRITIC_OBS_DIM + 9] = 1.0

    algorithm.frame_begin(obs, wire)

    assert algorithm.pending_tick["depth"] is None
    assert algorithm.pending_tick["nav_feat"].shape == (
        1,
        p2_contract.NAV_FEATURE_DIM,
    )
    assert torch.isfinite(algorithm.pending_tick["nav_feat"]).all()


def test_eval_runtime_constructs_no_training_only_state():
    algorithm = _make_p2_algorithm(training=False)
    assert algorithm.critic is None
    assert algorithm.response_buffer is None
    assert algorithm.rollout is None
    assert algorithm.actor_optimizer is None
    assert algorithm.critic_optimizer is None
    assert algorithm.response_optimizer is None
    assert algorithm.actor_scheduler is None
    assert algorithm.critic_scheduler is None
    assert algorithm.response_scheduler is None
    assert algorithm.safety_head is None


def test_safety_loss_updates_only_navigation_encoder_and_training_head():
    algorithm = _make_p2_algorithm(training=True)
    algorithm.actor_optimizer.zero_grad(set_to_none=True)
    depth = torch.zeros(
        1, p2_contract.DEPTH_HEIGHT, p2_contract.DEPTH_WIDTH, 1
    )
    feat = algorithm.navigation_encoder(depth)
    logits = algorithm.safety_head(feat)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits)
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in algorithm.navigation_encoder.parameters())
    assert all(parameter.grad is not None for parameter in algorithm.safety_head.parameters())
    assert all(parameter.grad is None for parameter in algorithm.actor.parameters())
    actor_ids = {
        id(parameter)
        for group in algorithm.actor_optimizer.param_groups
        for parameter in group["params"]
    }
    assert len(actor_ids) == sum(
        len(group["params"]) for group in algorithm.actor_optimizer.param_groups
    )


def test_p2_checkpoint_leaf_contract_and_action_rng_exact_resume(tmp_path):
    source = _make_p2_algorithm(training=True)
    source.low_level_payload = {"contract_version": "low_level_v2"}
    source.parent_optimizer_payload = {}
    source.parent_scheduler_payload = {}
    source.parent_training_payload = {}
    source.source_parent_model_id = "37953"
    source.parent_checkpoint_sha256 = "parent-sha"
    source.low_level_state_digest = source._module_digest(
        (("vision", source.low_level_encoder), ("actor", source.low_level_actor))
    )
    source.frame_count = 7
    path = tmp_path / "model.ckpt-safewarm-42.pkl"
    source.save_training_bundle(str(path), platform_model_id=42)
    expected_noise = torch.randn((4,), generator=source.action_generator)
    expected_vy_noise = torch.randn((4,), generator=source.vy_action_generator)

    payload = torch.load(path, weights_only=False, map_location="cpu")
    assert payload["bundle_kind"] == "hierarchical_control_v3"
    assert payload["model_spec"] == {
        "proprio_dim": 45,
        "scan_dim": 256,
        "depth_height": 180,
        "depth_width": 320,
        "depth_channels": 1,
        "latent_dim": 32,
        "action_dim": 12,
        "goal_dim": 0,
    }
    existing_eval_candidates = [
        Path(candidate)
        for candidate in p2_nav_evaluation_candidates(tmp_path, 42)
        if Path(candidate).is_file()
    ]
    assert existing_eval_candidates[0] == path
    assert payload["training_states"]["global"]["train_scope"] == (
        "high_level_and_response_adapter"
    )
    assert payload["contracts"]["training"] == p2_contract.training_contract()
    assert payload["contracts"]["training_digest"] == p2_contract.stable_digest(
        p2_contract.training_contract()
    )
    assert payload["modules"]["high_level"]["navigation_safety_head"]["spec"][
        "training_only"
    ] is True
    assert payload["modules"]["high_level"]["navigation_safety_head"][
        "training_only"
    ] is True
    for group_name in ("low_level", "high_level"):
        group = payload["modules"][group_name]
        for leaf_name, leaf in group.items():
            if leaf_name in {"contract_version", "component_status"}:
                continue
            if not isinstance(leaf, dict) or "state_dict" not in leaf:
                continue
            assert {"class_name", "spec", "state_dict"}.issubset(leaf), (
                group_name,
                leaf_name,
            )

    restored = _make_p2_algorithm(training=True)
    restored.load_mode = "p2_safe_direction_continue_warm_start"
    assert restored.load_bundle(str(path), platform_model_id=99999) == (
        "exact_resume_history_reset"
    )
    assert restored.loaded_platform_model_id == "42"
    actual_noise = torch.randn((4,), generator=restored.action_generator)
    assert torch.equal(actual_noise, expected_noise)
    actual_vy_noise = torch.randn((4,), generator=restored.vy_action_generator)
    assert torch.equal(actual_vy_noise, expected_vy_noise)
    assert restored.session_effective_seconds == 0.0
    assert restored.lifetime_effective_seconds == 0.0
    assert restored.frame_count == 10
    assert restored.frame_count % restored.nav_period_frames == 0

    inference_payload = dict(payload)
    for training_key in (
        "optimizers",
        "schedulers",
        "training_states",
        "transparent_parent_optimizers",
        "transparent_parent_schedulers",
        "transparent_parent_training_states",
    ):
        inference_payload.pop(training_key, None)
    inference_path = tmp_path / "model.ckpt-navfull-43.pkl"
    inference_payload["platform_model_id"] = "43"
    torch.save(inference_payload, inference_path)
    evaluator = _make_p2_algorithm(training=False)
    assert evaluator.load_evaluation_bundle(
        str(inference_path), platform_model_id=43
    ) == "evaluate_full_modules_only"


def test_safe_direction_warm_start_preserves_critic_and_old_optimizer_moments(tmp_path):
    source = _make_p2_algorithm(training=True)
    source.low_level_payload = {"contract_version": "low_level_v2"}
    source.parent_optimizer_payload = {}
    source.parent_scheduler_payload = {}
    source.parent_training_payload = {}
    source.low_level_state_digest = source._module_digest(
        (("vision", source.low_level_encoder), ("actor", source.low_level_actor))
    )
    source.return_statistics = {
        "count": 123,
        "mean": 4.5,
        "m2": 9.0,
        "value_normalization_enabled": False,
    }
    with torch.no_grad():
        for parameter in source.critic.parameters():
            parameter.fill_(0.125)
        for parameter in source.safety_head.parameters():
            parameter.fill_(7.0)
    for group in source.actor_optimizer.param_groups:
        if group.get("name") == "navigation_safety_head":
            continue
        for parameter in group["params"]:
            parameter.grad = torch.ones_like(parameter)
    source.actor_optimizer.step()
    source.actor_optimizer.zero_grad(set_to_none=True)
    for parameter in source.critic.parameters():
        parameter.grad = torch.ones_like(parameter)
    source.critic_optimizer.step()
    source.critic_optimizer.zero_grad(set_to_none=True)
    expected_critic = {
        name: value.detach().clone() for name, value in source.critic.state_dict().items()
    }
    path = tmp_path / "model.ckpt-safefull-100.pkl"
    source.save_training_bundle(str(path), platform_model_id=100)
    parent_payload = torch.load(path, weights_only=False, map_location="cpu")
    parent_payload["modules"]["high_level"].pop("navigation_safety_head")
    parent_payload["contracts"]["reward"]["version"] = (
        "p2_track_reward_v8_terminal_potential"
    )
    parent_payload["optimizers"]["high_level_actor"]["param_groups"] = [
        group
        for group in parent_payload["optimizers"]["high_level_actor"]["param_groups"]
        if group.get("name") != "navigation_safety_head"
    ]
    torch.save(parent_payload, path)

    restored = _make_p2_algorithm(training=True)
    restored.load_mode = "p2_safe_direction_continue_warm_start"
    assert restored.load_bundle(str(path), platform_model_id=999) == (
        "p2_safe_direction_continue_warm_start"
    )
    assert restored.return_statistics == source.return_statistics
    assert all(
        torch.equal(value, expected_critic[name])
        for name, value in restored.critic.state_dict().items()
    )
    assert all(
        not torch.allclose(parameter, torch.full_like(parameter, 7.0))
        for parameter in restored.safety_head.parameters()
    )
    safety_parameters = set(restored.safety_head.parameters())
    assert all(parameter not in restored.actor_optimizer.state for parameter in safety_parameters)
    assert any(
        parameter in restored.actor_optimizer.state
        for parameter in restored.actor.parameters()
    )
    assert restored.session_effective_seconds == 0.0


def test_two_axis_p2_migrates_main_policy_optimizer_and_clocks(tmp_path):
    class LegacyActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_std = nn.Parameter(torch.full((2,), -0.7))
            self.memory = nn.LSTM(p2_contract.ACTOR_INPUT_DIM, 64, 2)
            self.mean_head = nn.Linear(64, 2)

    source = _make_p2_algorithm(training=True)
    source.low_level_payload = {"contract_version": "low_level_v2"}
    source.parent_optimizer_payload = {}
    source.parent_scheduler_payload = {}
    source.parent_training_payload = {}
    source.source_parent_model_id = "37953"
    source.parent_checkpoint_sha256 = "parent-sha"
    source.low_level_state_digest = source._module_digest(
        (("vision", source.low_level_encoder), ("actor", source.low_level_actor))
    )
    with torch.no_grad():
        source.actor.mean_head.weight.fill_(0.123)
        source.actor.mean_head.bias.copy_(
            torch.tensor((p2_contract.INITIAL_VX_PRE_TANH, 0.25))
        )
    path = tmp_path / "model.ckpt-navfull-291713.pkl"
    source.save_training_bundle(str(path), platform_model_id=291713)
    expected_main_noise = torch.randn((4,), generator=source.action_generator)
    payload = torch.load(path, weights_only=False, map_location="cpu")

    legacy = LegacyActor()
    common = {
        name: value
        for name, value in source.actor.state_dict().items()
        if name not in {"vy_log_std", "vy_mean_head.weight", "vy_mean_head.bias"}
    }
    legacy.load_state_dict(common, strict=True)
    groups = []
    for name, parameters in source.navigation_encoder.parameter_groups().items():
        groups.append(
            {
                "params": list(parameters),
                "name": f"navigation_{name}",
                "lr": p2_contract.CNN_LAYER_LRS[name],
            }
        )
    groups.append(
        {"params": list(legacy.parameters()), "name": "actor", "lr": p2_contract.ACTOR_LR}
    )
    legacy_optimizer = torch.optim.Adam(groups)
    for group in legacy_optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.ones_like(parameter)
    legacy_optimizer.step()

    high = payload["modules"]["high_level"]
    high["contract_version"] = "high_level_continuous_v1"
    high["actor"] = {
        "class_name": "P2NavigationActor",
        "spec": {
            "input_dim": p2_contract.ACTOR_INPUT_DIM,
            "hidden_dim": 64,
            "num_layers": 2,
            "action_dim": 2,
            "distribution": "diagonal_tanh_squashed_gaussian",
            "physical_output": ["vx", "wz"],
        },
        "state_dict": legacy.state_dict(),
    }
    payload["optimizers"]["high_level_actor"] = legacy_optimizer.state_dict()
    payload["contracts"]["command"]["version"] = "p2_continuous_command_v1"
    payload["training_states"]["high_level"]["effective_training_seconds"] = 28761.0
    payload["training_states"]["high_level"].pop("lifetime_effective_seconds", None)
    torch.save(payload, path)

    restored = _make_p2_algorithm(training=True)
    restored.load_mode = "p2_command_v2_expansion_warm_start"
    assert restored.load_bundle(str(path), platform_model_id=99999) == (
        "p2_command_v2_expansion_warm_start"
    )
    assert torch.equal(restored.actor.mean_head.weight, legacy.mean_head.weight)
    assert torch.equal(restored.actor.mean_head.bias, legacy.mean_head.bias)
    assert torch.equal(restored.actor.log_std, legacy.log_std)
    assert restored.actor.vy_mean_head.weight.count_nonzero() == 0
    assert restored.actor.vy_mean_head.bias.item() == 0.0
    assert restored.actor.vy_log_std.item() == pytest.approx(-1.1)
    fixed_input = torch.linspace(-0.2, 0.2, p2_contract.ACTOR_INPUT_DIM).reshape(
        1, 1, -1
    )
    with torch.no_grad():
        legacy_features, _ = legacy.memory(fixed_input)
        legacy_mean = legacy.mean_head(legacy_features)
        migrated_mean, migrated_log_std, _ = restored.actor.distribution_parameters(
            fixed_input
        )
    assert torch.equal(migrated_mean[..., (0, 2)], legacy_mean)
    assert torch.equal(migrated_log_std[..., (0, 2)], legacy.log_std.expand_as(legacy_mean))
    assert restored.optimizer_migration_report["optimizer"]["restored"] > 0
    assert restored.optimizer_migration_report["rng"]["main_action"]["status"] == (
        "restored"
    )
    assert torch.equal(
        torch.randn((4,), generator=restored.action_generator),
        expected_main_noise,
    )
    assert restored.lifetime_base_seconds == pytest.approx(28761.0)
    assert restored.lifetime_effective_seconds == pytest.approx(28761.0)
    assert restored.session_effective_seconds == 0.0
    assert restored.return_statistics["count"] == 0


def test_command_v2_warm_start_rng_device_mismatch_is_reported_not_blocking():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    incompatible = torch.zeros(16, dtype=torch.uint8)
    report = AlgorithmP2NavPPO._restore_warm_start_rng(
        generator,
        incompatible,
        name="legacy_cuda_action_rng",
    )
    assert report["status"] == "fresh_seed"
    assert report["reason"].startswith("source_state_incompatible:")


def test_old_p2_reward_bundle_warm_starts_policy_and_rebuilds_critic(tmp_path):
    source = _make_p2_algorithm(training=True)
    source.low_level_payload = {"contract_version": "low_level_v2"}
    source.parent_optimizer_payload = {}
    source.parent_scheduler_payload = {}
    source.parent_training_payload = {}
    source.source_parent_model_id = "99393"
    source.parent_checkpoint_sha256 = "old-p2-sha"
    source.adapter_gradient_steps = 7
    source.stuck_positive_ema = 0.23
    source.adapter_batch_envs = 32
    source.adapter_oom_skips = 2
    source.low_level_state_digest = source._module_digest(
        (("vision", source.low_level_encoder), ("actor", source.low_level_actor))
    )
    with torch.no_grad():
        next(source.actor.parameters()).fill_(0.123)
        next(source.response_adapter.parameters()).fill_(0.234)
        next(source.critic.parameters()).fill_(0.456)
    path = tmp_path / "model.ckpt-navadapt-99393.pkl"
    source.save_training_bundle(str(path), platform_model_id=99393)
    expected_adapter_noise = torch.rand((4,), generator=source.adapter_generator)
    payload = torch.load(path, weights_only=False, map_location="cpu")
    payload["contracts"]["reward"]["version"] = "p2_track_reward_v1"
    payload["contracts"].pop("training", None)
    payload["contracts"].pop("training_digest", None)
    torch.save(payload, path)

    restored = _make_p2_algorithm(training=True)
    restored.load_mode = "reward_v2_warm_start"
    assert restored.load_bundle(str(path), platform_model_id=99393) == (
        "p2_reward_v2_warm_start"
    )
    assert torch.all(next(restored.actor.parameters()) == 0.123)
    assert torch.all(next(restored.response_adapter.parameters()) == 0.234)
    assert not torch.all(next(restored.critic.parameters()) == 0.456)
    assert restored.current_iteration == 0
    assert restored.effective_training_seconds == 0.0
    assert restored.return_statistics["count"] == 0
    assert restored.adapter_gradient_steps == 7
    assert restored.stuck_positive_ema == pytest.approx(0.23)
    assert restored.adapter_batch_envs == 32
    assert restored.adapter_oom_skips == 2
    assert torch.equal(
        torch.rand((4,), generator=restored.adapter_generator),
        expected_adapter_noise,
    )
    assert restored.actor_optimizer.param_groups[-1]["lr"] == pytest.approx(
        p2_contract.ACTOR_LR
    )


def test_p15_parent_auxiliary_low_modules_are_migrated_not_dropped():
    algorithm = _make_p2_algorithm(training=True)
    low = {
        "critic": {"state_dict": {"weight": torch.ones(1)}},
    }
    modules = {
        "action_distribution": {"std": torch.full((12,), 0.15)},
        "s0_anchor": {
            "vision_encoder_state_dict": algorithm.low_level_encoder.state_dict(),
            "actor_state_dict": algorithm.low_level_actor.state_dict(),
        },
    }
    algorithm._migrate_parent_low_aux_modules(modules, low)
    algorithm._validate_low_aux_modules(low, context="test low_level")
    assert low["action_distribution"]["class_name"] == (
        "DiagonalGaussianActionDistribution"
    )
    assert low["s0_anchor"]["contract_version"] == "low_level_s0_anchor_v1"
    assert low["critic"]["class_name"] == "VisualCritic"
