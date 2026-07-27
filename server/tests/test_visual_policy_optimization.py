#!/usr/bin/env python3
"""Contract tests for Standard visual command generalization.

Covers the P0 evaluation entry, worker command bridge, checkpoint selection,
and warning-only platform boundaries. Tensor tests skip locally and must run on
the platform image; local skips are never reported as tensor passes.
"""

from __future__ import annotations

import copy
import hashlib
import re
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
CONFIG = (
    SERVER
    / "agent_ppo"
    / "conf"
    / "train_env_conf_standard_visual_policy_optimization.toml"
)
REWARD_PROCESS = SERVER / "agent_ppo" / "feature" / "reward_process.py"
BASE_ENV = SERVER / "isaac_env" / "base_env.py"
EXPECTED_PLATFORM_BASE_ENV_SHA256 = (
    "75ebdaf6888e94262598a26db1586b2598cb474422e3382b6bba9e96ddbb6e67"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class CommandGeneralizationConfigTests(unittest.TestCase):
    """The active TOML matches the four-hour command-generalization plan."""

    @classmethod
    def setUpClass(cls):
        with CONFIG.open("rb") as stream:
            cls.config = tomllib.load(stream)
        cls.stage = cls.config["visual_policy_optimization"]

    def test_command_generalization_identity_and_parent(self):
        stage = self.stage
        self.assertEqual(stage["run_name"], "standard-command-r1")
        self.assertEqual(stage["schedule_mode"], "visual_command_generalization_v1")
        self.assertEqual(stage["transition_parent_model_id"], 28401)
        self.assertEqual(stage["command_anchor_action"], 0.35)
        self.assertEqual(stage["command_anchor_latent"], 0.10)
        self.assertEqual(stage["command_checkpoint_minutes"], [30.0, 120.0, 180.0, 230.0])
        self.assertTrue(stage["warning_only_safety"])

    def test_env_and_episode_match_source_domain(self):
        self.assertEqual(self.config["env"]["num_envs"], 256)
        self.assertEqual(self.config["env"]["episode_length_s"], 25)
        self.assertEqual(
            self.config["env_conf"]["policy_entry"],
            "visual_policy_optimization",
        )

    def test_terrain_matches_28401_source_domain(self):
        terrain = self.config["terrain"]
        self.assertTrue(terrain["curriculum"])
        self.assertEqual(terrain["max_init_terrain_level"], 9)
        std = terrain["standard"]
        # Source domain of visionfull-28401: 20/20/20/40/0.
        self.assertEqual(std["pyramid_slope"]["proportion"], 0.20)
        self.assertEqual(std["pyramid_slope_inv"]["proportion"], 0.20)
        self.assertEqual(std["pyramid_stairs"]["proportion"], 0.20)
        self.assertEqual(std["pyramid_stairs_inv"]["proportion"], 0.40)
        self.assertEqual(std["maze"]["proportion"], 0.0)

    def test_native_fallback_and_global_command_limits(self):
        commands = self.config["commands"]
        # The native fallback must exactly reproduce the source domain. The
        # agent-side scheduler owns short command holds after a verified write.
        self.assertEqual(commands["resampling_time"], [300.0, 300.0])
        self.assertEqual(commands["ranges"]["lin_vel_x"], [0.3, 1.3])
        self.assertEqual(commands["limit"]["lin_vel_x"], [0.0, 1.3])
        self.assertEqual(commands["ranges"]["lin_vel_y"], [-0.2, 0.2])
        self.assertEqual(commands["ranges"]["ang_vel_yaw"], [-0.3, 0.3])
        # limit.ang_vel_z and ranges.ang_vel_yaw are the platform's two legacy
        # key names for the same Z-axis yaw command (§3); keep both as-is.
        self.assertEqual(commands["limit"]["ang_vel_z"], [-0.3, 0.3])

    def test_worker_command_scheduler_is_the_only_enabled_override(self):
        commands = self.config["commands"]
        self.assertFalse(commands["buckets"]["enabled"])
        self.assertTrue(commands["worker_progressive"]["enabled"])
        self.assertEqual(commands["worker_progressive"]["log_interval_steps"], 500)

    def test_fixed_anchor_learning_rates_and_save_cadence(self):
        stage = self.stage
        self.assertEqual(stage["actor_learning_rate"], 1.0e-5)
        self.assertEqual(stage["lstm_learning_rate"], 5.0e-6)
        self.assertEqual(stage["critic_learning_rate"], 1.0e-4)
        self.assertEqual(stage["save_interval_minutes"], 10.0)
        self.assertEqual(stage["resume_first_save_minutes"], 2.0)

    def test_custom_command_scheduler_configuration(self):
        schedule = self.stage["command_schedule"]
        self.assertEqual(schedule["step_dt_s"], 0.02)
        self.assertEqual(schedule["source_hold_s"], [6.0, 10.0])
        self.assertEqual(schedule["target_hold_s"], [3.0, 6.0])
        self.assertEqual(schedule["zero_hold_s"], [1.5, 3.0])
        self.assertEqual(schedule["source_vx"], [0.3, 1.3])
        self.assertEqual(schedule["source_vy"], [-0.2, 0.2])
        self.assertEqual(schedule["source_wz"], [-0.3, 0.3])
        self.assertEqual(
            schedule["target_bucket_weights"], [0.15, 0.20, 0.30, 0.15, 0.10, 0.10]
        )

    def test_all_randomization_disabled(self):
        self.assertFalse(self.config["domain_rand"]["enable_domain_rand"])
        self.assertFalse(self.config["domain_rand"]["randomize_friction"])
        self.assertFalse(self.config["domain_rand"]["push_robots"])
        self.assertFalse(self.config["noise"]["add_noise"])
        self.assertFalse(
            self.config["camera"]["depth_camera"]["augmentation"]["enabled"]
        )

    def test_depth_preprocess_resolver_honors_active_toml(self):
        from agent_ppo.conf.depth_config import load_depth_preprocess_conf

        depth = load_depth_preprocess_conf(
            "standard", "visual_policy_optimization"
        )
        self.assertEqual(depth["max_depth"], 5.0)
        self.assertFalse(depth["augmentation"]["enabled"])

    def test_reward_surface_frozen(self):
        """The bounded zero-command term is the only reward surface addition."""
        rewards = self.config["rewards"]
        # The complete Anchor R2 reward surface (20 blocks). Any change here
        # must be intentional and reflected in the execution card.
        expected = {
            "track_lin_vel_xy", "track_ang_vel_z", "forward_velocity",
            "air_time_variance_penalty", "max_foot_air_time", "foot_symmetry",
            "trot_gait", "hip_to_default", "dof_pos_limits", "joint_acc",
            "joint_position_penalty", "feet_air_time", "feet_regulation",
            "feet_contact_forces", "feet_slide", "feet_stumble",
            "legs_distance", "x_command_hip_regular", "approach_goal",
            "reach_goal", "zero_command_stability",
        }
        self.assertEqual(set(rewards.keys()), expected)
        # Command-compatible fixes from §3.
        self.assertEqual(rewards["track_lin_vel_xy"]["weight"], 3.0)
        self.assertEqual(rewards["track_ang_vel_z"]["weight"], 1.0)
        self.assertEqual(rewards["forward_velocity"]["weight"], 0.0)
        self.assertEqual(
            rewards["joint_position_penalty"]["params"]["stand_still_scale"],
            2.0,
        )
        self.assertEqual(rewards["zero_command_stability"]["weight"], -0.05)
        self.assertEqual(
            rewards["zero_command_stability"]["params"]["command_threshold"],
            0.05,
        )

    def test_reward_process_source_unchanged(self):
        """Command-aware gates and zero stability remain explicit in source."""
        source = _read(REWARD_PROCESS)
        # Custom reward methods defined in RewardProcess subclass.
        for method in (
            "_reward_trot_gait",
            "_reward_forward_velocity",
            "_reward_joint_position_penalty",
            "_reward_foot_symmetry",
            "_reward_air_time_variance_penalty",
            "_reward_max_foot_air_time",
            "_reward_zero_command_stability",
        ):
            self.assertIn(method, source, f"{method} removed from reward_process.py")
        self.assertIn("moving = torch.linalg.vector_norm", source)
        self.assertIn("command[:, :3]", source)


class CommandGeneralizationSourceContractTests(unittest.TestCase):
    """P0 entry and agent-side command integration stay platform-independent."""

    def test_agent_ppo_does_not_import_base_env(self):
        """§5.9: no agent_ppo file imports isaac_env.base_env."""
        agent_ppo = SERVER / "agent_ppo"
        py_files = list(agent_ppy_python_files(agent_ppo))
        self.assertTrue(py_files, "expected agent_ppo python files")
        for path in py_files:
            source = _read(path)
            self.assertNotIn(
                "from isaac_env.base_env",
                source,
                f"{path.name} imports isaac_env.base_env",
            )
            self.assertNotIn(
                "import isaac_env.base_env",
                source,
                f"{path.name} imports isaac_env.base_env",
            )

    def test_base_env_matches_operator_supplied_platform_version(self):
        digest = hashlib.sha256(BASE_ENV.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_PLATFORM_BASE_ENV_SHA256)

    def test_worker_bridge_uses_only_public_command_access(self):
        source = _read(SERVER / "agent_ppo" / "feature" / "worker_command_bridge.py")
        self.assertIn('get_command("base_velocity")', source)
        self.assertIn("current.copy_", source)
        self.assertIn("common_step_counter", source)
        self.assertIn("episode_length_buf", source)
        self.assertIn("command_tensor_shape", source)
        self.assertIn("readback_error_max", source)
        self.assertIn("policy_error_max", source)
        self.assertIn("critic_error_max", source)
        self.assertNotIn("command_manager.get_term", source)
        self.assertNotIn("command_manager._command", source)

    def test_platform_eval_stage_compatibility_entry(self):
        source = _read(SERVER / "agent_ppo" / "conf" / "conf.py")
        self.assertIn(
            "def _infer_stage_for_eval(usr_conf, logger):",
            source,
        )
        self.assertIn("return _infer_stage_from_task_name(usr_conf, logger)", source)
        self.assertNotIn("Camera evaluation deliberately maps to LBCLocoConfig", source)

    def test_visual_ppo_eval_loader_cannot_fall_back_to_lbc_loader(self):
        source = _read(SERVER / "agent_ppo" / "agent.py")
        visual_loader = source.split("def _load_visual_ppo(", 1)[1].split(
            "def _visual_env_seed(", 1
        )[0]
        self.assertIn("load_eval_bundle(", visual_loader)
        self.assertNotIn("_load_lbc_loco_for_eval", visual_loader)
        self.assertIn("runtime diagnostics", source)
        self.assertIn("file_size_bytes", source)
        self.assertIn("checkpoint has no parent lineage", source)

    def test_explicit_policy_entry_precedes_camera_task_inference(self):
        if "toml" not in sys.modules:
            try:
                import toml  # noqa: F401
            except ModuleNotFoundError:
                toml_stub = types.ModuleType("toml")
                toml_stub.load = tomllib.load
                sys.modules["toml"] = toml_stub
        from agent_ppo.conf.conf import (
            LBCLocoConfig,
            StandardVisualPPOConfig,
            _get_explicit_policy_entry,
            _valid_explicit_policy_stage,
            _infer_stage_for_eval,
            _infer_stage_from_task_name,
        )

        class Logger:
            def __init__(self):
                self.warnings = []

            def info(self, message):
                pass

            def warning(self, message):
                self.warnings.append(message)

        logger = Logger()
        camera = {"env_conf": {"task_name": "Unitree-Go2-Velocity-Camera"}}
        self.assertIs(_infer_stage_from_task_name(camera, logger), LBCLocoConfig)

        explicit = {
            "env_conf": {
                "task_name": "Unitree-Go2-Velocity-Camera",
                "policy_entry": "visual_policy_optimization",
            }
        }
        self.assertIs(
            _infer_stage_from_task_name(explicit, logger),
            StandardVisualPPOConfig,
        )
        self.assertIs(
            _infer_stage_for_eval(explicit, logger),
            StandardVisualPPOConfig,
        )

        # A missing [eval] table must not discard the [env_conf] override.
        explicit_alias = {
            "env_conf": {
                "task_name": "Unitree-Go2-Velocity-Camera",
                "eval_policy": "standard_visual_ppo",
            }
        }
        self.assertIs(
            _infer_stage_from_task_name(explicit_alias, logger),
            StandardVisualPPOConfig,
        )

        top_level = {
            "env_conf": {"task_name": "Unitree-Go2-Velocity-Camera"},
            "eval": {"policy_entry": "visual_ppo"},
        }
        self.assertIs(
            _infer_stage_from_task_name(top_level, logger),
            StandardVisualPPOConfig,
        )
        self.assertEqual(
            _get_explicit_policy_entry(
                {"env_conf": {"policy_entry": "visual_ppo"}, "eval": True}
            ),
            "visual_ppo",
        )

        # A present-but-invalid high-priority key must not silently defer to a
        # valid lower-priority value from another table.
        invalid_high_priority = {
            "env_conf": {
                "task_name": "Unitree-Go2-Velocity-Camera",
                "policy_entry": 0,
            },
            "eval": {"policy_entry": "visual_policy_optimization"},
        }
        self.assertEqual(_get_explicit_policy_entry(invalid_high_priority), 0)
        self.assertIsNone(_valid_explicit_policy_stage(invalid_high_priority))
        self.assertIs(
            _infer_stage_from_task_name(invalid_high_priority, logger),
            LBCLocoConfig,
        )
        self.assertTrue(any("must be a non-empty string" in item for item in logger.warnings))

        invalid_empty = {
            "env_conf": {
                "task_name": "Unitree-Go2-Velocity-Camera",
                "policy_entry": "",
                "eval_policy": "visual_policy_optimization",
            }
        }
        self.assertIs(
            _infer_stage_from_task_name(invalid_empty, logger),
            LBCLocoConfig,
        )
        self.assertIsNone(_valid_explicit_policy_stage(invalid_empty))
        self.assertTrue(any("is empty" in item for item in logger.warnings))

    def test_training_rollout_ignores_platform_global_all_done_marker(self):
        source = _read(SERVER / "agent_ppo" / "workflow" / "train_workflow.py")
        rollout = source.split("def run_episodes_(", 1)[1]
        self.assertNotIn(
            'infos.get("all_done")',
            rollout,
            "training must not stop on platform global-frame all_done",
        )
        self.assertNotIn(
            'infos["all_done"]',
            rollout,
            "training must not stop on platform global-frame all_done",
        )

    def test_active_runtime_uses_worker_observation_bridge(self):
        agent_source = _read(SERVER / "agent_ppo" / "agent.py")
        self.assertNotIn("CommandAdapter", agent_source)
        self.assertNotIn("self.command_scheduler", agent_source)
        for name in (
            "lbc_observation_process.py",
            "policy_observation_process.py",
            "critic_observation_process.py",
        ):
            source = _read(SERVER / "agent_ppo" / "feature" / name)
            self.assertLess(
                source.index("apply_worker_command(self.env)"),
                source.index("self.default_observation()"),
            )
            self.assertIn("record_worker_command_observation", source)

    def test_visual_workflow_records_command_phase_and_runtime_contract(self):
        source = _read(
            SERVER / "agent_ppo" / "workflow" / "visual_ppo_workflow.py"
        )
        for label, code in (
            ("commandbase", 0),
            ("commandblend", 1),
            ("commandfull", 2),
        ):
            self.assertIn(f'"{label}": {code}', source)
        for field in (
            "continuous_training=",
            "iteration_semantics=completed_outer_iterations_v1",
            "outer_iteration=one_rollout_plus_one_optimizer_update",
            "inner_steps_per_outer=",
            "num_envs=",
            "config_path=",
            "terrain_proportions=",
            "depth_augmentation=",
            "command_runtime_owner=worker_observation_bridge_v1",
        ):
            self.assertIn(field, source)

    def test_visual_workflow_final_save_handles_graceful_exit_only(self):
        source = _read(
            SERVER / "agent_ppo" / "workflow" / "visual_ppo_workflow.py"
        )
        self.assertIn("def _save_final_checkpoint", source)
        self.assertIn("except (KeyboardInterrupt, SystemExit):", source)
        self.assertIn('reason="graceful_platform_exit"', source)
        self.assertIn('agent._visual_ppo_final_save_reason = "iteration_cap"', source)
        self.assertIn("_install_sigterm_checkpoint_handler", source)
        self.assertIn("signal.SIGTERM", source)
        self.assertIn("previous is not signal.SIG_DFL", source)

    def test_worker_scheduler_enabled_without_base_env_dependency(self):
        config_path = CONFIG
        with config_path.open("rb") as stream:
            import tomllib
            config = tomllib.load(stream)
        self.assertFalse(config["commands"]["buckets"]["enabled"])
        self.assertTrue(config["commands"]["worker_progressive"]["enabled"])
        self.assertIn("command_schedule", config["visual_policy_optimization"])
        # base_env.py must not be a training-feature dependency of agent_ppo
        # (the platform overwrites it). Verified by test_agent_ppo_does_not_import_base_env.

    def test_command_phase_labels_are_probe_safe(self):
        """All command labels are pure lowercase letters for the platform probe."""
        source = _read(SERVER / "agent_ppo" / "checkpoint_io.py")
        self.assertIn("VISUAL_COMMAND_PHASE_LABELS", source)
        labels = ("commandbase", "commandblend", "commandfull")
        for label in labels:
            self.assertIn(label, source)
            self.assertRegex(
                f"model.ckpt-{label}-28401.pkl",
                r"^model\.ckpt-[a-z]+-[0-9]+\.[^.]+$",
            )

    def test_command_candidate_functions_exist(self):
        """Candidate selection is explicit, ID-scoped, and eval-aware."""
        source = _read(SERVER / "agent_ppo" / "checkpoint_io.py")
        for name in (
            "visual_command_checkpoint_candidates",
            "visual_command_parent_candidates",
            "visual_latest_model_id",
            "visual_eval_checkpoint_candidates",
        ):
            self.assertIn(f"def {name}", source, f"{name} not defined")

    def test_visual_ppo_save_writes_single_bundle(self):
        """§5.6: save_model writes one phase bundle, no rlfull/locomotion copies."""
        source = _read(SERVER / "agent_ppo" / "agent.py")
        visual_block = source.split("if self.is_visual_ppo:", 1)[1].split(
            "elif self.is_lbc:", 1
        )[0]
        self.assertIn("save_training_bundle(", visual_block)
        # No same-id alias copies in the visual branch.
        self.assertNotIn("shutil.copyfile", visual_block)
        self.assertNotIn("_save_side_locomotion", visual_block)

    def test_startup_logs_worker_bridge_and_depth_contract(self):
        source = _read(SERVER / "agent_ppo" / "agent.py")
        self.assertIn("worker_observation_bridge", source)
        self.assertIn("command_runtime_owner=environment_worker", source)
        self.assertIn("depth_augmentation_enabled=", source)

    def test_load_modes_include_command_transition(self):
        source = _read(SERVER / "agent_ppo" / "algorithm" / "algorithm_visual_ppo.py")
        self.assertIn('LOAD_MODE_S0 = "s0"', source)
        self.assertIn('LOAD_MODE_ANCHOR_RESUME = "anchor_resume"', source)
        self.assertIn('LOAD_MODE_TRANSITION_RESUME = "transition_resume"', source)
        self.assertIn('LOAD_MODE_SCHEDULE_MIGRATION = "schedule_migration"', source)

    def test_checkpoint_records_task_local_worker_runtime_without_scheduler_state(self):
        source = _read(SERVER / "agent_ppo" / "algorithm" / "algorithm_visual_ppo.py")
        self.assertIn("anchor_session_elapsed_hours", source)
        self.assertIn("command_session_elapsed_hours", source)
        self.assertIn('"command_runtime_owner": "worker_observation_bridge_v1"', source)
        self.assertIn('"command_resume_policy": "new_task_restart"', source)
        self.assertNotIn("command_scheduler_state", source)
        self.assertNotIn("command_profile_mix_history", source)
        self.assertNotIn("command_hook_verification", source)


class CommandCandidateIdTests(unittest.TestCase):
    """P0: command checkpoint selection is ordered and never crosses IDs."""

    def setUp(self):
        from agent_ppo.checkpoint_io import (
            VISUAL_COMMAND_PHASE_LABELS,
            visual_command_checkpoint_candidates,
            visual_command_parent_candidates,
            visual_eval_checkpoint_diagnostics,
            visual_eval_checkpoint_candidates,
            visual_latest_model_id,
            validate_visual_eval_bundle_identity,
        )
        self.labels = VISUAL_COMMAND_PHASE_LABELS
        self.resume_fn = visual_command_checkpoint_candidates
        self.parent_fn = visual_command_parent_candidates
        self.eval_diagnostics_fn = visual_eval_checkpoint_diagnostics
        self.eval_fn = visual_eval_checkpoint_candidates
        self.latest_id_fn = visual_latest_model_id
        self.validate_identity = validate_visual_eval_bundle_identity

    def _populate(self, tmp, label, id):
        path = tmp / f"model.ckpt-{label}-{id}.pkl"
        path.write_bytes(b"")

    def test_explicit_id_never_crosses_id_boundary(self):
        """Rule 1: explicit ID filters by filename numeric ID equality first."""
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            # A higher command label on a different ID must not preempt the
            # operator-selected same-ID checkpoint.
            self._populate(tmp, "commandfull", "99999")
            self._populate(tmp, "commandbase", "28401")
            candidates = self.resume_fn(str(tmp), 28401)
            self.assertTrue(candidates)
            self.assertNotIn(str(tmp / "model.ckpt-commandfull-99999.pkl"), candidates)
            self.assertEqual(
                candidates,
                [
                    str(tmp / "model.ckpt-commandfull-28401.pkl"),
                    str(tmp / "model.ckpt-commandblend-28401.pkl"),
                    str(tmp / "model.ckpt-commandbase-28401.pkl"),
                ],
            )

    def test_resume_priority_within_same_id(self):
        """commandfull -> commandblend -> commandbase."""
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            for label in self.labels:
                self._populate(tmp, label, "28401")
            candidates = self.resume_fn(str(tmp), 28401)
            basenames = [Path(c).name for c in candidates]
            expected = [
                "model.ckpt-commandfull-28401.pkl",
                "model.ckpt-commandblend-28401.pkl",
                "model.ckpt-commandbase-28401.pkl",
            ]
            self.assertEqual(basenames, expected)

    def test_parent_prefers_same_id_command_then_anchor(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            self._populate(tmp, "anchorfinal", "28401")
            self._populate(tmp, "commandblend", "28401")
            candidates = self.parent_fn(str(tmp), 28401)
            self.assertEqual(
                Path(candidates[0]).name, "model.ckpt-commandblend-28401.pkl"
            )

    def test_eval_prefers_command_then_falls_back_to_anchor_same_id(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            self._populate(tmp, "commandbase", "28401")
            self._populate(tmp, "anchorfinal", "28401")
            self._populate(tmp, "commandfull", "99999")
            candidates = self.eval_fn(str(tmp), 28401)
            self.assertEqual(
                [Path(candidate).name for candidate in candidates[:2]],
                ["model.ckpt-commandbase-28401.pkl", "model.ckpt-anchorfinal-28401.pkl"],
            )

    def test_eval_candidate_order_spans_command_anchor_rl_and_vision(self):
        """Camera eval picks the newest supported stage within one model ID."""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            for label in ("commandbase", "anchorfinal", "rlfull", "visionfull"):
                self._populate(tmp, label, "30531")

            candidates = self.eval_fn(str(tmp), 30531)
            self.assertEqual(
                [Path(candidate).name for candidate in candidates],
                [
                    "model.ckpt-commandbase-30531.pkl",
                    "model.ckpt-anchorfinal-30531.pkl",
                    "model.ckpt-rlfull-30531.pkl",
                    "model.ckpt-visionfull-30531.pkl",
                ],
            )

    def test_eval_bundle_identity_accepts_requested_bundle_or_lineage_id(self):
        base = {
            "format": "kaiwu_train_v1",
            "schema_version": 1,
            "lineage": {},
        }
        with_bundle_id = {**base, "platform_model_id": "30531"}
        with_lineage_id = {
            **base,
            "platform_model_id": None,
            "lineage": {"platform_model_id": 30531},
        }

        self.assertEqual(
            self.validate_identity(with_bundle_id, "30531")["bundle_id"], "30531"
        )
        self.assertEqual(
            self.validate_identity(with_lineage_id, "30531")["lineage_id"], 30531
        )

    def test_eval_bundle_identity_warns_on_cross_id_missing_and_conflicting_ids(self):
        base = {
            "format": "kaiwu_train_v1",
            "schema_version": 1,
            "lineage": {},
        }
        cases = (
            {**base, "platform_model_id": "99999"},
            {**base, "platform_model_id": None},
            {
                **base,
                "platform_model_id": "30531",
                "lineage": {"platform_model_id": "99999"},
            },
        )
        for bundle in cases:
            with self.subTest(bundle=bundle):
                info = self.validate_identity(bundle, "30531")
                self.assertTrue(info["identity_warnings"])

    def test_lbc_camera_entry_requires_a_loaded_same_id_visual_bundle(self):
        """Platform-forced lbc_loco must not produce a random-policy score."""
        source = _read(SERVER / "agent_ppo" / "agent.py")
        lbc_loader = source.split("def _load_lbc_loco(", 1)[1].split(
            "def _find_vision_eval_ckpt(", 1
        )[0]
        resolver = source.split("def _find_vision_eval_ckpt(", 1)[1].split(
            "def _checkpoint_sha256(", 1
        )[0]
        identity_helper = source.split(
            "def _validate_lbc_eval_bundle_identity(", 1
        )[1].split("def _load_lbc_loco_for_eval(", 1)[0]
        eval_loader = source.split("def _load_lbc_loco_for_eval(", 1)[1].split(
            "@staticmethod\n    def _ckpt_exact_match", 1
        )[0]
        exploit_lbc = source.split("def _exploit_lbc_loco(", 1)[1].split(
            "def learn(", 1
        )[0]

        self.assertIn("self._eval_requested_model_id = str(id)", lbc_loader)
        self.assertIn("visual_eval_checkpoint_candidates", resolver)
        self.assertIn("visual_eval_checkpoint_diagnostics", resolver)
        self.assertNotIn("vision_checkpoint_candidates(path, id)", resolver)
        self.assertNotIn("model.ckpt-lbc-loco", resolver)
        self.assertIn("self._validate_lbc_eval_bundle_identity", eval_loader)
        self.assertIn("validate_visual_eval_bundle_identity", identity_helper)
        self.assertIn("validate_low_level_spec", eval_loader)
        self.assertIn("is_kaiwu_train_bundle(ckpt)", eval_loader)
        self.assertNotIn("except", eval_loader)
        self.assertIn("refusing inference because no checkpoint was", exploit_lbc)
        self.assertIn('"successfully loaded"', exploit_lbc)

    def test_eval_missing_id_reports_inventory_without_cross_id_candidate(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            self._populate(tmp, "commandfull", "99999")
            diagnostic = self.eval_diagnostics_fn(str(tmp), 28401)
            self.assertEqual(diagnostic["requested_id"], "28401")
            self.assertEqual(diagnostic["same_id_existing"], [])
            self.assertIn(
                str(tmp / "model.ckpt-commandbase-28401.pkl"),
                diagnostic["same_id_expected"],
            )
            self.assertEqual(
                diagnostic["other_visual_files"],
                [str(tmp / "model.ckpt-commandfull-99999.pkl")],
            )
            self.assertEqual(self.eval_fn(str(tmp), 28401), [])

    def test_latest_resolves_across_command_anchor_and_vision_labels(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            self._populate(tmp, "anchorfinal", "30000")
            self._populate(tmp, "visionfull", "31000")
            self._populate(tmp, "commandfull", "32000")
            self.assertEqual(self.latest_id_fn(str(tmp)), 32000)


class AnchorR2SaveDedupTests(unittest.TestCase):
    """§5.8 N4: save-trigger dedup boundary at delta <= 60.0s.

    These tests exercise the pure-Python dedup decision logic by simulating the
    four delta cases against a small state machine that mirrors the workflow's
    save-trigger resolution.
    """

    def test_dedup_boundary_table(self):
        from agent_ppo.workflow.save_schedule import (
            should_deduplicate_save,
        )

        # delta, first_save, expected_skip
        cases = [
            (0.0, False, True),       # same loop -> skip second
            (30.0, False, True),      # within window -> skip
            (60.0, False, True),      # boundary inclusive -> skip
            (60.001, False, False),   # just past -> keep (save twice)
            (0.0, True, False),       # first save of session -> keep
        ]
        for delta, first_save, expected_skip in cases:
            with self.subTest(delta=delta, first_save=first_save):
                self.assertEqual(
                    should_deduplicate_save(
                        delta,
                        first_save_this_session=first_save,
                    ),
                    expected_skip,
                    f"delta={delta}, first={first_save}",
                )


@unittest.skipIf(torch is None, "torch not installed; run on platform/CI")
class CommandScheduleTests(unittest.TestCase):
    """Pure-torch scheduler behavior is independent from Isaac Lab."""

    def _schedule(self, num_envs=8):
        from agent_ppo.feature.command_schedule import CommandSchedule

        return CommandSchedule(num_envs=num_envs, device="cpu")

    def test_target_probability_wall_clock_boundaries(self):
        schedule = self._schedule()
        cases = [
            (0.0, 0.0),
            (29.999, 0.0),
            (30.0, 0.0),
            (75.0, 0.25),
            (120.0, 0.5),
            (150.0, 0.75),
            (180.0, 1.0),
            (240.0, 1.0),
        ]
        for minutes, expected in cases:
            with self.subTest(minutes=minutes):
                schedule.set_elapsed_hours(minutes / 60.0)
                self.assertAlmostEqual(schedule.target_probability, expected)

    def test_target_bucket_ranges_and_anchor_weights(self):
        from agent_ppo.feature.command_schedule import anchor_weights_from_commands

        schedule = self._schedule(num_envs=4)
        commands, buckets = schedule._sample_target_commands(4096)
        self.assertTrue(torch.all(commands[:, 0] >= 0.0))
        self.assertTrue(torch.all(commands[:, 0] <= 1.3))
        self.assertTrue(torch.all(commands[:, 1].abs() <= 0.2))
        self.assertTrue(torch.all(commands[:, 2].abs() <= 0.3))
        self.assertEqual(set(buckets.unique().tolist()), set(range(6)))
        zero = commands[buckets == 0]
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
        pure_yaw = commands[buckets == 4]
        self.assertTrue(torch.all(pure_yaw[:, :2] == 0.0))
        self.assertTrue(torch.all(pure_yaw[:, 2].abs() >= 0.15))
        lateral = commands[buckets == 5]
        self.assertTrue(torch.all(lateral[:, 0] == 0.0))
        self.assertTrue(torch.all(lateral[:, 2] == 0.0))
        self.assertTrue(torch.all(lateral[:, 1].abs() >= 0.10))
        weights = schedule._anchor_weight_for_targets(
            torch.tensor([0, 1, 2, 3, 3, 4, 5]),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.2, 0.0, 0.2],
                    [0.4, 0.0, 0.2],
                    [0.0, 0.0, 0.2],
                    [0.0, 0.2, 0.0],
                ]
            ),
        )
        self.assertTrue(torch.equal(weights, torch.tensor([0.0, 0.25, 1.0, 0.25, 1.0, 0.0, 0.0])))
        observed_weights = anchor_weights_from_commands(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.2, 0.0, 0.2],
                    [0.4, 0.0, 0.2],
                    [0.0, 0.0, 0.2],
                    [0.0, 0.2, 0.0],
                ]
            )
        )
        self.assertTrue(
            torch.equal(
                observed_weights.flatten(),
                torch.tensor([0.0, 0.25, 1.0, 0.25, 1.0, 0.0, 0.0]),
            )
        )

    def test_subset_reset_preserves_unexpired_environments(self):
        schedule = self._schedule(num_envs=4)
        schedule.set_elapsed_hours(3.0)
        current = torch.zeros(4, 3)
        plan = schedule.plan(current)
        self.assertEqual(plan.pending_ids.numel(), 4)
        schedule.commit(plan.pending_ids, applied=True)
        before_other = schedule.hold_remaining[[1, 3]].clone()
        schedule.plan(
            schedule.command.clone(),
            reset_mask=torch.tensor([True, False, True, False]),
            dt_s=0.0,
        )
        self.assertTrue(torch.equal(schedule.hold_remaining[[1, 3]], before_other))
        self.assertFalse(hasattr(schedule, "state"))
        self.assertFalse(hasattr(schedule, "load_state"))


@unittest.skipIf(torch is None, "torch not installed; run on platform/CI")
class WorkerCommandBridgeTests(unittest.TestCase):
    """The worker owns one idempotent scheduler shared by both obs groups."""

    class _Clock:
        def __init__(self):
            self.value = 0.0

        def __call__(self):
            return self.value

    class _CommandManager:
        def __init__(self, num_envs=4, *, returns_copy=False):
            self.commands = torch.zeros(num_envs, 3)
            self.returns_copy = returns_copy

        def get_command(self, name):
            if name != "base_velocity":
                raise KeyError(name)
            return self.commands.clone() if self.returns_copy else self.commands

    class _Env:
        def __init__(self, num_envs=4, *, returns_copy=False):
            self.num_envs = num_envs
            self.device = torch.device("cpu")
            self.common_step_counter = 0
            self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
            self.command_manager = WorkerCommandBridgeTests._CommandManager(
                num_envs, returns_copy=returns_copy
            )

    def _bridge(self, env=None, *, enabled=True):
        from agent_ppo.feature.worker_command_bridge import WorkerCommandBridge

        env = env or self._Env()
        clock = self._Clock()
        bridge = WorkerCommandBridge(
            env,
            enabled=enabled,
            config={"step_dt_s": 0.02},
            log_interval_steps=500,
            clock=clock,
        )
        clock.value = 3.0 * 3600.0
        return env, bridge, clock

    @staticmethod
    def _observation(env, group):
        width, start = (45, 6) if group == "policy" else (60, 9)
        obs = torch.zeros(env.num_envs, width)
        obs[:, start : start + 3] = env.command_manager.commands
        return obs

    def test_policy_and_critic_order_samples_once_and_observe_same_command(self):
        for order in (("policy", "critic"), ("critic", "policy")):
            with self.subTest(order=order):
                env, bridge, _ = self._bridge()
                bridge.apply()
                for group in order:
                    bridge.apply()
                    bridge.record_observation(group, self._observation(env, group))
                self.assertEqual(bridge.scheduler.requested_samples, env.num_envs)
                self.assertEqual(bridge.scheduler.effective_target_samples, env.num_envs)
                self.assertEqual(bridge.last_readback_error_max, 0.0)
                self.assertEqual(bridge.observation_errors["policy"], 0.0)
                self.assertEqual(bridge.observation_errors["critic"], 0.0)
                self.assertTrue(
                    torch.equal(
                        env.command_manager.get_command("base_velocity"),
                        bridge.scheduler.command,
                    )
                )

    def test_step_idempotency_and_subset_reset(self):
        env, bridge, _ = self._bridge()
        bridge.apply()
        requested = bridge.scheduler.requested_samples
        hold = bridge.scheduler.hold_remaining.clone()
        bridge.apply()
        self.assertEqual(bridge.scheduler.requested_samples, requested)
        self.assertTrue(torch.equal(bridge.scheduler.hold_remaining, hold))

        unchanged = bridge.scheduler.command[[1, 3]].clone()
        env.common_step_counter = 1
        env.episode_length_buf[:] = 1
        env.episode_length_buf[[0, 2]] = 0
        bridge.apply()
        self.assertEqual(bridge.scheduler.requested_samples, requested + 2)
        self.assertTrue(torch.equal(bridge.scheduler.command[[1, 3]], unchanged))

    def test_all_reset_resamples_every_environment(self):
        env, bridge, _ = self._bridge()
        bridge.apply()
        requested = bridge.scheduler.requested_samples
        env.common_step_counter = 1
        env.episode_length_buf.zero_()
        bridge.apply()
        self.assertEqual(
            bridge.scheduler.requested_samples,
            requested + env.num_envs,
        )

    def test_native_overwrite_is_republished_on_next_step(self):
        env, bridge, _ = self._bridge()
        bridge.apply()
        expected = bridge.scheduler.command.clone()
        env.command_manager.commands.zero_()
        env.common_step_counter = 1
        env.episode_length_buf[:] = 1
        bridge.apply()
        self.assertTrue(torch.equal(env.command_manager.commands, expected))

    def test_returned_copy_disables_schedule_without_faking_target_samples(self):
        env, bridge, _ = self._bridge(self._Env(returns_copy=True))
        original = env.command_manager.commands.clone()
        bridge.apply()
        self.assertEqual(bridge.status, "warning_disabled")
        self.assertEqual(bridge.scheduler.effective_target_samples, 0)
        self.assertTrue(torch.equal(env.command_manager.commands, original))

    def test_write_exception_restores_native_command_and_disables(self):
        env, bridge, _ = self._bridge()
        original = env.command_manager.commands.clone()

        def fail_write(_commands):
            raise RuntimeError("write failure")

        bridge._write_and_verify = fail_write
        bridge.apply()
        self.assertEqual(bridge.status, "warning_disabled")
        self.assertTrue(torch.equal(env.command_manager.commands, original))

    def test_partial_write_is_restored_before_schedule_is_disabled(self):
        env, bridge, _ = self._bridge()
        original = env.command_manager.commands.clone()
        actual_write = bridge._write_and_verify
        attempts = 0

        def corrupt_once_then_write(commands):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                env.command_manager.commands.add_(1.0)
                raise RuntimeError("partial write")
            return actual_write(commands)

        bridge._write_and_verify = corrupt_once_then_write
        bridge.apply()
        self.assertEqual(bridge.status, "warning_disabled")
        self.assertEqual(attempts, 2)
        self.assertTrue(torch.equal(env.command_manager.commands, original))

    def test_unrestorable_partial_write_stops(self):
        env, bridge, _ = self._bridge()

        def corrupt_then_fail(_commands):
            env.command_manager.commands.add_(1.0)
            raise RuntimeError("partial write")

        bridge._write_and_verify = corrupt_then_fail
        with self.assertRaisesRegex(RuntimeError, "could not be restored"):
            bridge.apply()

    def test_disabled_bridge_never_changes_native_command(self):
        env, bridge, _ = self._bridge(enabled=False)
        env.command_manager.commands[:] = 0.5
        bridge.apply()
        self.assertTrue(torch.all(env.command_manager.commands == 0.5))


@unittest.skipIf(torch is None, "torch not installed; run on platform/CI")
class ZeroCommandTelemetryTests(unittest.TestCase):
    def test_reports_only_clipped_action_metrics_and_labels_environment_fields_unavailable(self):
        from agent_ppo.feature.zero_command_telemetry import ZeroCommandTelemetry

        telemetry = ZeroCommandTelemetry(
            num_envs=2,
            device="cpu",
            capacity=8,
            grace_period_s=0.0,
        )
        commands = torch.zeros(2, 3)
        telemetry.observe(commands, torch.zeros(2, 12), reset_mask=torch.ones(2), dt_s=0.02)
        telemetry.observe(commands, torch.ones(2, 12), dt_s=0.02)
        metrics = telemetry.metrics()
        self.assertEqual(metrics["zero_telemetry_window_samples"], 4)
        self.assertGreater(metrics["zero_action_delta_p95"], 0.0)
        self.assertTrue(metrics["zero_root_v_xy"].startswith("unavailable:"))
        self.assertTrue(metrics["zero_foot_slide"].startswith("unavailable:"))

    def test_pure_yaw_is_not_collected_as_zero_command(self):
        from agent_ppo.feature.zero_command_telemetry import ZeroCommandTelemetry

        telemetry = ZeroCommandTelemetry(
            num_envs=2,
            device="cpu",
            grace_period_s=0.0,
        )
        commands = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]])
        telemetry.observe(commands, torch.ones(2, 12), dt_s=0.02)
        self.assertEqual(telemetry.metrics()["zero_telemetry_window_samples"], 0)


@unittest.skipIf(torch is None, "torch not installed; run on platform/CI")
class CommandGeneralizationAlgorithmStateTests(unittest.TestCase):
    """Fixed anchors and phase labels for the command-generalization schedule."""

    def _build_algorithm(self):
        """Construct a minimal AlgorithmVisualPPO with a tiny VisualActorCritic."""
        from agent_ppo.algorithm.algorithm_visual_ppo import AlgorithmVisualPPO
        from agent_ppo.model.visual_actor_critic import VisualActorCritic
        import torch.optim as optim
        import torch

        torch.manual_seed(0)
        model = VisualActorCritic(
            num_proprio=45, num_scan=256, depth_shape=(8, 8, 1),
            latent_dim=32, cnn_output_dim=16, lstm_hidden_size=16,
            lstm_num_layers=1, num_critic_obs=64, num_actions=12,
            actor_hidden_dims=(16,), critic_hidden_dims=(16,),
        )
        anchor_encoder = copy.deepcopy(model.vision_encoder)
        anchor_actor = copy.deepcopy(model.actor)
        actor_parameters = [*model.actor.parameters(), model.std]
        recurrent_parameters = [
            *model.vision_encoder.rnn.parameters(),
            *model.vision_encoder.rnn_output_layer.parameters(),
        ]
        optimizer = optim.Adam([
            {"params": actor_parameters, "lr": 1e-5, "name": "actor"},
            {"params": recurrent_parameters, "lr": 5e-6, "name": "lstm"},
            {"params": model.critic.parameters(), "lr": 1e-4, "name": "critic"},
        ])
        return AlgorithmVisualPPO(
            model=model, anchor_encoder=anchor_encoder, anchor_actor=anchor_actor,
            optimizer=optimizer, sequence_length=4,
            schedule_mode="visual_command_generalization_v1", run_name="standard-command-r1",
            source_parent_model_id=28401,
            anchor_schedule_hours=None,
            action_anchor_schedule=None,
            latent_anchor_schedule=None,
            anchor_phase_labels=None,
            anchor_phase_end_hours=None,
            critic_warmup_learning_rate=3e-4, task_end_hours=4.0,
            warning_only_safety=True, max_anchor_action_mse=0.05,
            max_hard_termination_delta=0.02,
            command_anchor_action=0.35,
            command_anchor_latent=0.10,
        )

    def test_command_phase_boundaries_and_fixed_anchor_weights(self):
        algo = self._build_algorithm()
        cases = [
            (0.0, "commandbase"),
            (29.99 / 60, "commandbase"),
            (30 / 60, "commandblend"),
            (179.99 / 60, "commandblend"),
            (180 / 60, "commandfull"),
            (240 / 60, "commandfull"),
        ]
        for elapsed_h, phase in cases:
            with self.subTest(elapsed_h=elapsed_h):
                self.assertEqual(algo._phase_for_elapsed(elapsed_h), phase)
                self.assertAlmostEqual(
                    algo._anchor_weight_for_elapsed(elapsed_h, kind="action"),
                    0.35, places=3,
                )
                self.assertAlmostEqual(
                    algo._anchor_weight_for_elapsed(elapsed_h, kind="latent"),
                    0.10, places=3,
                )

    def test_optimizer_excludes_cnn_and_anchor(self):
        algo = self._build_algorithm()
        import torch
        opt_param_ids = set()
        for group in algo.optimizer.param_groups:
            for p in group["params"]:
                opt_param_ids.add(id(p))
        # CNN and S0 anchor params must be absent.
        cnn_ids = {id(p) for p in algo.actor_critic.vision_encoder.cnn.parameters()}
        anchor_ids = {id(p) for p in algo.anchor_encoder.parameters()}
        anchor_ids |= {id(p) for p in algo.anchor_actor.parameters()}
        self.assertFalse(opt_param_ids & cnn_ids)
        self.assertFalse(opt_param_ids & anchor_ids)

    def test_warning_only_safety_never_pauses(self):
        algo = self._build_algorithm()
        import torch

        algo.warning_only_safety = True
        algo.baseline_hard_termination_rate = 0.0
        algo.last_anchor_action_mse = 1.0  # exceeds 0.05 threshold
        algo.storage = SimpleNamespace(
            hard_terminations=torch.ones(2, 2, 1),
            actions=torch.full((2, 2, 12), 7.0),
        )
        algo._update_safety_state(elapsed_h=1.0)
        self.assertTrue(algo.last_diagnostic_save_requested)
        self.assertFalse(algo.actor_updates_paused)
        self.assertFalse(algo.anchor_schedule_frozen)


class CommandCheckpointTests(unittest.TestCase):
    """Command checkpoint filenames are platform probe-safe."""

    def test_command_filenames_match_probe_regex(self):
        from agent_ppo.checkpoint_io import validate_probe_filename
        for label in ("commandbase", "commandblend", "commandfull"):
            self.assertTrue(validate_probe_filename(f"model.ckpt-{label}-28401.pkl"))
            self.assertFalse(validate_probe_filename(f"model.ckpt-{label}2-28401.pkl"))


def agent_ppy_python_files(root: Path):
    """Yield .py files under root, excluding __pycache__."""
    if not root.exists():
        return []
    return [
        path for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


if __name__ == "__main__":
    unittest.main(verbosity=2)
