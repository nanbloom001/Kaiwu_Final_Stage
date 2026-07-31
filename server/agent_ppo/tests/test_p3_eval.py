#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""P3 dual-eval regression tests: ``p3_standard_eval`` / ``p3_track_eval``.

Covers the failure in BUG 599578 / p3nav8h-r1_884257: the P3 package was
routed to ``lbc_loco`` at evaluation time and the old loader did not search the
``highslow`` P3 phase labels, so no checkpoint was loaded.

The pure-logic tests (candidate discovery, shared validator, stage routing,
no-training-buffer guarantee) always run.  The tests that exercise the real
``model.ckpt-highslow-884257.pkl`` require the file at
``agent_ppo/tests/data/model.ckpt-highslow-884257.pkl`` and skip otherwise so
the suite stays green on machines without the 35 MB package.
"""

import copy
import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(__file__))
import agent_ppo.tests._nav_test_stubs  # noqa: F401

from agent_ppo.conf.conf import (  # noqa: E402
    Config,
    LBCLocoConfig,
    NavEvalConfig,
    P2NavEvalConfig,
    P2NavPPOConfig,
    P3StandardEvalConfig,
    P3StandardJointConfig,
    P3TrackEvalConfig,
    StandardVisualPPOConfig,
    _infer_stage_from_task_name,
    _valid_explicit_policy_stage,
)
from agent_ppo.checkpoint_io import (  # noqa: E402
    P3_STANDARD_JOINT_PHASE_LABELS,
    P3_EVAL_LOW_LEVEL_SPEC,
    p3_standard_joint_eval_candidates,
    validate_p3_eval_bundle,
)

REAL_CKPT = os.path.join(
    os.path.dirname(__file__), "data", "model.ckpt-highslow-884257.pkl"
)

HAS_REAL_CKPT = os.path.isfile(REAL_CKPT)


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _p3_fixture_bundle(*, mode: str, stage_type: str = "p3_standard_joint",
                       phase_label: str = "highslow",
                       corrupt_leaf: str | None = None,
                       nonfinite_leaf: str | None = None,
                       high_complete: bool = True) -> dict:
    """Build a minimal P3 kaiwu_train_v1 package matching the real structure."""
    from agent_ppo.model.p2_high_level import (
        navigation_actor_spec,
        navigation_encoder_spec,
    )
    from agent_ppo.model.response_adapter import response_adapter_spec

    def _leaf(spec, name="w"):
        state = {name: torch.zeros(4)}
        if nonfinite_leaf:
            state[name] = torch.full((4,), float("nan"))
        return {"class_name": spec["class_name"], "spec": spec["spec"], "state_dict": state}

    low_spec = P3_EVAL_LOW_LEVEL_SPEC
    low = {
        "contract_version": "low_level_v2",
        "locomotion_encoder": _leaf(low_spec["locomotion_encoder"]),
        "actor": _leaf(low_spec["actor"]),
        "critic": {"class_name": "VisualCritic", "spec": {}, "state_dict": {}},
    }
    high_spec = {
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
    high = {
        "contract_version": "high_level_continuous_v2",
        "component_status": "complete" if high_complete else "adapter_only",
        **{name: _leaf(spec) for name, spec in high_spec.items()},
        "critic": {"class_name": "P2NavigationCritic", "spec": {}, "state_dict": {}},
        "navigation_safety_head": {
            "class_name": "NavigationSafetyHead",
            "spec": {},
            "state_dict": {},
            "training_only": True,
        },
    }
    modules = {"low_level": low, "high_level": high}
    if corrupt_leaf == "locomotion_encoder":
        del low["locomotion_encoder"]
    if corrupt_leaf == "actor":
        del low["actor"]
    if corrupt_leaf == "high_actor":
        del high["actor"]
    return {
        "format": "kaiwu_train_v1",
        "schema_version": 2,
        "stage_type": stage_type,
        "phase_label": phase_label,
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
        "modules": modules,
        "capabilities": {"deployable": False},
    }


class P3EvalCandidatesTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.mkdtemp()
        self._labels = P3_STANDARD_JOINT_PHASE_LABELS

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def _touch(self, *names):
        created = []
        for name in names:
            path = os.path.join(self._tmp, name)
            with open(path, "wb") as stream:
                stream.write(b"x")
            created.append(path)
        return created[0] if created else None

    def test_same_id_priority_highslow_first(self):
        paths = [
            self._touch(f"model.ckpt-{label}-884257.pkl")
            for label in self._labels
        ]
        candidates = p3_standard_joint_eval_candidates(self._tmp, 884257)
        # All same-ID P3 phase files, newest phase first (highslow ... lowbase)
        self.assertEqual(
            [os.path.basename(path) for path in candidates],
            [f"model.ckpt-{label}-884257.pkl" for label in reversed(self._labels)],
        )

    def test_never_falls_back_to_p2_or_lbc_labels(self):
        self._touch(
            "model.ckpt-highslow-884257.pkl",
            "model.ckpt-safestable-884257.pkl",
            "model.ckpt-navfull-884257.pkl",
            "model.ckpt-lbc-loco-884257.pkl",
            "model.ckpt-responsecalib-884257.pkl",
        )
        candidates = p3_standard_joint_eval_candidates(self._tmp, 884257)
        names = [os.path.basename(path) for path in candidates]
        self.assertIn("model.ckpt-highslow-884257.pkl", names)
        self.assertFalse(any("safestable" in name or "navfull" in name for name in names))
        self.assertFalse(any("lbc-loco" in name or "responsecalib" in name for name in names))

    def test_unique_discovery_when_no_same_id(self):
        self._touch("model.ckpt-highslow-999999.pkl")
        candidates = p3_standard_joint_eval_candidates(self._tmp, 884257)
        self.assertEqual(len(candidates), 1)
        self.assertIn("highslow-999999", os.path.basename(candidates[0]))

    def test_ambiguous_discovery_raises(self):
        self._touch("model.ckpt-highslow-111111.pkl", "model.ckpt-lowfull-222222.pkl")
        with self.assertRaises(RuntimeError):
            p3_standard_joint_eval_candidates(self._tmp, 884257)

    def test_empty_when_no_p3_files(self):
        self._touch("model.ckpt-safestable-884257.pkl")
        self.assertEqual(p3_standard_joint_eval_candidates(self._tmp, 884257), [])


class P3EvalValidatorTest(unittest.TestCase):
    def test_standard_mode_loads_only_low_level(self):
        bundle = _p3_fixture_bundle(mode="standard")
        disposition = validate_p3_eval_bundle(bundle, mode="standard")
        self.assertEqual(disposition["phase_label"], "highslow")
        self.assertEqual(
            disposition["loaded_modules"],
            ["low_level.actor", "low_level.locomotion_encoder"],
        )

    def test_track_mode_requires_full_hierarchy(self):
        bundle = _p3_fixture_bundle(mode="track")
        disposition = validate_p3_eval_bundle(bundle, mode="track")
        self.assertEqual(
            disposition["loaded_modules"],
            [
                "high_level.actor",
                "high_level.navigation_encoder",
                "high_level.response_adapter",
                "low_level.actor",
                "low_level.locomotion_encoder",
            ],
        )

    def test_track_rejects_incomplete_high_level(self):
        bundle = _p3_fixture_bundle(mode="track", high_complete=False)
        with self.assertRaises(ValueError):
            validate_p3_eval_bundle(bundle, mode="track")

    def test_rejects_non_p3_stage(self):
        bundle = _p3_fixture_bundle(mode="standard", stage_type="p2_nav_ppo")
        with self.assertRaises(ValueError):
            validate_p3_eval_bundle(bundle, mode="standard")

    def test_rejects_invalid_phase_label(self):
        bundle = _p3_fixture_bundle(mode="standard", phase_label="navfull")
        with self.assertRaises(ValueError):
            validate_p3_eval_bundle(bundle, mode="standard")

    def test_rejects_missing_low_leaf(self):
        bundle = _p3_fixture_bundle(mode="standard", corrupt_leaf="locomotion_encoder")
        with self.assertRaises(KeyError):
            validate_p3_eval_bundle(bundle, mode="standard")

    def test_rejects_missing_high_actor_for_track(self):
        bundle = _p3_fixture_bundle(mode="track", corrupt_leaf="high_actor")
        with self.assertRaises(KeyError):
            validate_p3_eval_bundle(bundle, mode="track")

    def test_rejects_nonfinite_state(self):
        bundle = _p3_fixture_bundle(mode="track", nonfinite_leaf="actor")
        with self.assertRaises((ValueError, FloatingPointError)):
            validate_p3_eval_bundle(bundle, mode="track")

    def test_rejects_unknown_mode(self):
        bundle = _p3_fixture_bundle(mode="standard")
        with self.assertRaises(ValueError):
            validate_p3_eval_bundle(bundle, mode="deploy")


class P3EvalWorkerBridgeTest(unittest.TestCase):
    """P3 track eval must enable the shared P2 worker transport."""

    def _resolve(self, algorithm, *, is_eval):
        from agent_ppo.conf.conf import Config
        from agent_ppo.feature import p2_worker_bridge

        stage = type("Stage", (), {"algorithm": algorithm})()

        with mock.patch.object(
            Config,
            "load_conf",
            classmethod(
                lambda cls, logger: (
                    {
                        "env_conf": {"seed": 7},
                        "p3_standard_joint": {"run_name": "p3-track-eval"},
                    },
                    "eval",
                    is_eval,
                    stage,
                )
            ),
        ):
            return p2_worker_bridge._resolve_config()

    def test_track_eval_enables_bridge_and_reads_p3_section(self):
        enabled, config, seed = self._resolve("p3_track_eval", is_eval=True)
        self.assertTrue(enabled)
        self.assertEqual(config["run_name"], "p3-track-eval")
        self.assertEqual(config["_worker_stage_type"], "p3_track_eval")
        self.assertEqual(seed, 7)

    def test_standard_eval_does_not_enable_bridge(self):
        enabled, _config, _seed = self._resolve("p3_standard_eval", is_eval=True)
        self.assertFalse(enabled)

    def test_legacy_loco_still_disables_bridge(self):
        enabled, _config, _seed = self._resolve("lbc_loco", is_eval=True)
        self.assertFalse(enabled)


class P3EvalStageRoutingTest(unittest.TestCase):
    def test_explicit_standard_eval(self):
        usr_conf = {
            "env_conf": {
                "task_name": "Unitree-Go2-Velocity-Camera",
                "policy_entry": "p3_standard_eval",
            },
            "terrain": {"mode": "standard"},
        }
        self.assertIs(
            _valid_explicit_policy_stage(usr_conf), P3StandardEvalConfig
        )
        self.assertIs(
            _infer_stage_from_task_name(usr_conf, _Logger()), P3StandardEvalConfig
        )

    def test_explicit_track_eval(self):
        usr_conf = {
            "env_conf": {
                "task_name": "Unitree-Go2-Velocity-Camera",
                "policy_entry": "p3_track_eval",
            },
            "terrain": {"mode": "track"},
        }
        self.assertIs(
            _valid_explicit_policy_stage(usr_conf), P3TrackEvalConfig
        )
        self.assertIs(
            _infer_stage_from_task_name(usr_conf, _Logger()), P3TrackEvalConfig
        )

    def test_training_entry_remapped_to_eval_by_terrain(self):
        for mode, expected in (("standard", P3StandardEvalConfig), ("track", P3TrackEvalConfig)):
            usr_conf = {
                "env_conf": {
                    "task_name": "Unitree-Go2-Velocity-Camera",
                    "policy_entry": "p3_standard_joint",
                },
                "terrain": {"mode": mode},
            }
            self.assertIs(
                _infer_stage_from_task_name(usr_conf, _Logger()), expected
            )

    def test_track_camera_preserves_p3_lineage(self):
        usr_conf = {
            "env_conf": {"task_name": "Unitree-Go2-Velocity-Camera"},
            "terrain": {"mode": "track"},
        }
        with mock.patch.object(Config, "CURRENT", P3StandardJointConfig):
            self.assertIs(
                _infer_stage_from_task_name(usr_conf, _Logger()), P3TrackEvalConfig
            )

    def test_standard_camera_preserves_p3_lineage(self):
        usr_conf = {
            "env_conf": {"task_name": "Unitree-Go2-Velocity-Camera"},
            "terrain": {"mode": "standard"},
        }
        with mock.patch.object(Config, "CURRENT", P3StandardJointConfig):
            self.assertIs(
                _infer_stage_from_task_name(usr_conf, _Logger()), P3StandardEvalConfig
            )

    def test_legacy_lineages_unchanged(self):
        usr_conf_track = {
            "env_conf": {"task_name": "Unitree-Go2-Velocity-Camera"},
            "terrain": {"mode": "track"},
        }
        with mock.patch.object(Config, "CURRENT", P2NavPPOConfig):
            self.assertIs(
                _infer_stage_from_task_name(usr_conf_track, _Logger()), P2NavEvalConfig
            )
        with mock.patch.object(Config, "CURRENT", StandardVisualPPOConfig):
            usr_conf_std = {
                "env_conf": {"task_name": "Unitree-Go2-Velocity-Camera"},
                "terrain": {"mode": "standard"},
            }
            self.assertIs(
                _infer_stage_from_task_name(usr_conf_std, _Logger()), LBCLocoConfig
            )


@unittest.skipUnless(HAS_REAL_CKPT, "real highslow-884257 checkpoint not staged")
class P3EvalRealCheckpointTest(unittest.TestCase):
    """Real 884257 package: both eval assemblies, no training state, finite actions."""

    @staticmethod
    def _low_modules(device="cpu"):
        from agent_ppo.model.vision_encoder import VisionEncoder

        encoder = VisionEncoder(
            image_shape=(180, 320, 1),
            proprio_dim=45,
            cnn_output_dim=32,
            rnn_hidden_dim=64,
            rnn_num_layers=2,
            rnn_output_dim=32,
            use_lstm=True,
        ).to(device)
        actor = torch.nn.Sequential(
            torch.nn.Linear(77, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 12),
        ).to(device)
        return encoder, actor

    def test_standard_low_level_only_extraction_and_forward(self):
        from agent_ppo.checkpoint_io import validate_p3_eval_bundle

        raw = torch.load(REAL_CKPT, map_location="cpu", weights_only=False)
        validate_p3_eval_bundle(raw, mode="standard")
        low = raw["modules"]["low_level"]
        encoder, actor = self._low_modules()
        encoder.load_state_dict(low["locomotion_encoder"]["state_dict"], strict=True)
        actor.load_state_dict(low["actor"]["state_dict"], strict=True)
        encoder.eval()
        actor.eval()
        batch = 2
        proprio = torch.randn(batch, 45)
        depth = torch.randn(batch, 57600)
        with torch.no_grad():
            latent = encoder(depth_image=depth.reshape(batch, 180, 320, 1),
                             proprio=proprio, masks=None)
            action = actor(torch.cat((proprio, latent), dim=-1))
        self.assertEqual(tuple(action.shape), (batch, 12))
        self.assertTrue(bool(torch.isfinite(action).all()))

    def test_track_full_hierarchy_assembly_no_training_state(self):
        from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
        from agent_ppo.model.p2_high_level import NavigationEncoder, P2NavigationActor
        from agent_ppo.model.response_adapter import CommandResponseAdapter

        encoder, low_actor = self._low_modules()
        algo = AlgorithmP2NavPPO(
            low_level_encoder=encoder,
            low_level_actor=low_actor,
            navigation_encoder=NavigationEncoder(),
            safety_head=None,
            actor=P2NavigationActor(),
            critic=None,
            response_adapter=CommandResponseAdapter(),
            response_buffer=None,
            num_envs=2,
            device="cpu",
            config={},
            logger=None,
            monitor=None,
            training=False,
        )
        mode = algo.load_p3_evaluation_bundle(REAL_CKPT, platform_model_id=884257)
        self.assertEqual(mode, "evaluate_full_modules_only")
        self.assertEqual(getattr(algo, "_p3_eval_phase_label"), "highslow")
        # Eval assembly must never own training state.
        self.assertIsNone(algo.actor_optimizer)
        self.assertIsNone(algo.critic_optimizer)
        self.assertIsNone(algo.response_optimizer)
        self.assertIsNone(algo.actor_scheduler)
        self.assertIsNone(algo.critic_scheduler)
        self.assertIsNone(algo.response_scheduler)
        self.assertIsNone(algo.rollout)
        self.assertIsNone(algo.response_buffer)
        # All loaded modules frozen.
        self.assertFalse(
            any(p.requires_grad for p in algo.low_level_encoder.parameters())
        )
        self.assertFalse(
            any(p.requires_grad for p in algo.actor.parameters())
        )

    def test_track_frame_loop_finite_actions(self):
        from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
        from agent_ppo.feature import nav_contract, p2_contract
        from agent_ppo.model.p2_high_level import NavigationEncoder, P2NavigationActor
        from agent_ppo.model.response_adapter import CommandResponseAdapter

        encoder, low_actor = self._low_modules()
        algo = AlgorithmP2NavPPO(
            low_level_encoder=encoder,
            low_level_actor=low_actor,
            navigation_encoder=NavigationEncoder(),
            safety_head=None,
            actor=P2NavigationActor(),
            critic=None,
            response_adapter=CommandResponseAdapter(),
            response_buffer=None,
            num_envs=2,
            device="cpu",
            config={},
            logger=None,
            monitor=None,
            training=False,
        )
        algo.load_p3_evaluation_bundle(REAL_CKPT, platform_model_id=884257)
        batch = 2
        obs = torch.zeros(batch, nav_contract.POLICY_OBS_DIM)
        obs[:, :45] = torch.randn(batch, 45)
        obs[:, 301:305] = torch.randn(batch, 4)  # goal4
        obs[:, 305:] = torch.randn(batch, 57600)
        response_aux = torch.randn(batch, p2_contract.RESPONSE_AUX_DIM) * 0.1
        policy = p2_contract.pack_eval_response_aux(obs, response_aux)
        aux = p2_contract.unpack_eval_response_aux(policy)
        critic_wire = torch.zeros(batch, p2_contract.PRIVILEGED_WIRE_DIM)
        critic_wire[
            :,
            p2_contract.CRITIC_OBS_DIM
            : p2_contract.CRITIC_OBS_DIM + p2_contract.RESPONSE_AUX_DIM,
        ] = aux
        with torch.no_grad():
            for _ in range(12):  # cross the 5Hz tick boundary (> NAV_PERIOD_FRAMES)
                result, _, _ = algo.frame_begin(policy, critic_wire, deterministic=True)
                algo.eval_frame_advance()
                action = result["actions"]
                self.assertEqual(tuple(action.shape), (batch, 12))
                self.assertTrue(bool(torch.isfinite(action).all()))

    def test_same_checkpoint_loadable_by_both_assemblies_sequentially(self):
        from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
        from agent_ppo.model.p2_high_level import NavigationEncoder, P2NavigationActor
        from agent_ppo.model.response_adapter import CommandResponseAdapter

        raw = torch.load(REAL_CKPT, map_location="cpu", weights_only=False)
        low = raw["modules"]["low_level"]
        # Standard assembly first.
        encoder, actor = self._low_modules()
        encoder.load_state_dict(low["locomotion_encoder"]["state_dict"], strict=True)
        actor.load_state_dict(low["actor"]["state_dict"], strict=True)
        # Track assembly second, same file.
        encoder2, low_actor2 = self._low_modules()
        algo = AlgorithmP2NavPPO(
            low_level_encoder=encoder2,
            low_level_actor=low_actor2,
            navigation_encoder=NavigationEncoder(),
            safety_head=None,
            actor=P2NavigationActor(),
            critic=None,
            response_adapter=CommandResponseAdapter(),
            response_buffer=None,
            num_envs=1,
            device="cpu",
            config={},
            logger=None,
            monitor=None,
            training=False,
        )
        mode = algo.load_p3_evaluation_bundle(REAL_CKPT, platform_model_id=884257)
        self.assertEqual(mode, "evaluate_full_modules_only")
        # Standard low weights must be identical across both loads.
        self.assertTrue(
            torch.equal(
                encoder.state_dict()["cnn.fc.weight"],
                algo.low_level_encoder.state_dict()["cnn.fc.weight"],
            )
        )

    def test_corrupt_selected_checkpoint_hard_fails(self):
        from agent_ppo.algorithm.algorithm_p2_nav_ppo import AlgorithmP2NavPPO
        from agent_ppo.model.p2_high_level import NavigationEncoder, P2NavigationActor
        from agent_ppo.model.response_adapter import CommandResponseAdapter

        raw = torch.load(REAL_CKPT, map_location="cpu", weights_only=False)
        low = raw["modules"]["low_level"]
        low["actor"]["state_dict"] = copy.deepcopy(low["actor"]["state_dict"])
        for key in low["actor"]["state_dict"]:
            low["actor"]["state_dict"][key] = torch.full_like(
                low["actor"]["state_dict"][key], float("nan")
            )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            corrupt_path = os.path.join(tmp, "model.ckpt-highslow-884257.pkl")
            torch.save(raw, corrupt_path)
            encoder, low_actor = self._low_modules()
            algo = AlgorithmP2NavPPO(
                low_level_encoder=encoder,
                low_level_actor=low_actor,
                navigation_encoder=NavigationEncoder(),
                safety_head=None,
                actor=P2NavigationActor(),
                critic=None,
                response_adapter=CommandResponseAdapter(),
                response_buffer=None,
                num_envs=1,
                device="cpu",
                config={},
                logger=None,
                monitor=None,
                training=False,
            )
            with self.assertRaises((ValueError, FloatingPointError)):
                algo.load_p3_evaluation_bundle(
                    corrupt_path, platform_model_id=884257
                )


if __name__ == "__main__":
    unittest.main()
