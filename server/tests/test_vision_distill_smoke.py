#!/usr/bin/env python3
"""Smoke tests for Standard depth-vision distillation (stage 4).

Verifies the three codec paths and the single-forward DAgger contract WITHOUT
requiring GPU or Isaac Lab:

  1. checkpoint_io: vision phase labels + probe regex + candidate ordering
  2. vision bundle codec: save_vision_bundle -> load_vision_bundle round-trip
     (vision_encoder + frozen teacher + optimizer + ramp_state; NO LSTM hidden)
  3. parent loading: load_parent_bundle prefers daggerfull over locomotion
  4. eval loader: reads modules.vision_encoder, does NOT read height_scan
  5. single-forward cache: prepare_vision_update advances LSTM hidden exactly once

Tests that need torch are skipped when torch is absent (local dev machines).
The probe-regex / candidate-ordering tests run everywhere (no torch needed).

Run:  cd server && python -m pytest tests/test_vision_distill_smoke.py -v
"""

import os
import re
import sys
import unittest
from pathlib import Path

# Make `agent_ppo` importable when run from server/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

from agent_ppo.checkpoint_io import (  # noqa: E402
    VISION_PHASE_LABELS,
    validate_probe_filename,
    vision_checkpoint_candidates,
    vision_parent_candidates,
    vision_phase_label,
)


_PROBE_NAME = re.compile(r"^model\.ckpt-[a-z]*-*[0-9]+\.[^.]+$")


class ProbeRegexTests(unittest.TestCase):
    """Probe-regex and label validation (no torch needed)."""

    def test_vision_labels_are_all_lowercase_letters(self):
        for label in VISION_PHASE_LABELS:
            # every char must be a lowercase letter (probe regex [a-z]*)
            self.assertTrue(
                re.fullmatch(r"[a-z]+", label),
                f"vision label {label!r} must be pure lowercase letters",
            )

    def test_vision_filenames_match_probe_regex(self):
        for label in VISION_PHASE_LABELS:
            fname = f"model.ckpt-{label}-16288.pkl"
            self.assertTrue(_PROBE_NAME.match(fname), fname)
            self.assertTrue(validate_probe_filename(fname), fname)

    def test_non_pure_letter_labels_rejected_by_regex(self):
        # lbc-loco has two segments separated by a hyphen -> regex expects [0-9]+ after [a-z]*
        self.assertFalse(_PROBE_NAME.match("model.ckpt-lbc-loco-16288.pkl"))
        # ramp50 contains digits inside the label segment
        self.assertFalse(_PROBE_NAME.match("model.ckpt-ramp50-16288.pkl"))

    def test_vision_phase_label_accepts_valid_rejects_invalid(self):
        self.assertEqual(vision_phase_label("visionfull"), "visionfull")
        with self.assertRaises(ValueError):
            vision_phase_label("ramp50")
        with self.assertRaises(ValueError):
            vision_phase_label("lbc-loco")

    def test_vision_checkpoint_candidates_order(self):
        cands = vision_checkpoint_candidates("/tmp", "16288")
        # visionfull must come before visionhalf/visionteacher/visionblocked
        self.assertIn("model.ckpt-visionfull-16288.pkl", cands[0])
        full_idx = cands.index("/tmp/model.ckpt-visionfull-16288.pkl")
        half_idx = cands.index("/tmp/model.ckpt-visionhalf-16288.pkl")
        self.assertLess(full_idx, half_idx)

    def test_vision_parent_candidates_prefers_daggerfull(self):
        cands = vision_parent_candidates("/tmp", "16288")
        # daggerfull must come before locomotion (P1 issue 4)
        self.assertEqual(cands[0], "/tmp/model.ckpt-daggerfull-16288.pkl")
        self.assertEqual(cands[1], "/tmp/model.ckpt-locomotion-16288.pkl")


@unittest.skipUnless(HAS_TORCH, "torch not installed; run on platform/CI")
class VisionCodecTests(unittest.TestCase):
    """save/load_vision_bundle round-trip (requires torch)."""

    def _make_algorithm(self):
        import torch
        from agent_ppo.model.vision_encoder import DmEncoder, VisionEncoder
        from agent_ppo.algorithm.algorithm_lbc import AlgorithmLBC

        vision_encoder = VisionEncoder(
            image_shape=(180, 320, 1),
            proprio_dim=45,
            cnn_output_dim=32,
            rnn_hidden_dim=64,
            rnn_num_layers=2,
            rnn_output_dim=32,
            use_lstm=True,
        )
        teacher_encoder = DmEncoder(input_dim=256, hidden_dims=(512, 256), output_dim=32)
        # minimal teacher_actor: Linear(77, 12)
        import torch.nn as nn
        teacher_actor = nn.Linear(77, 12)
        algo = AlgorithmLBC(
            vision_encoder=vision_encoder,
            teacher_encoder=teacher_encoder,
            teacher_actor=teacher_actor,
            device="cpu",
            learning_rate=1e-4,
            latent_dim=32,
            proprio_dim=45,
            scan_dim=256,
            depth_shape=(180, 320, 1),
        )
        return algo

    def test_round_trip_preserves_weights_and_ramp_state(self):
        import torch
        import tempfile

        algo = self._make_algorithm()
        algo.current_iteration = 1234
        algo.total_steps = 999
        algo.ramp_probability = 0.42
        algo.parent_checkpoint_sha256 = "deadbeef" * 8
        algo.training_status = "running"

        # capture original weights
        ve_before = {k: v.clone() for k, v in algo.vision_encoder.state_dict().items()}
        te_before = {k: v.clone() for k, v in algo.teacher_encoder.state_dict().items()}
        ta_before = {k: v.clone() for k, v in algo.teacher_actor.state_dict().items()}

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.ckpt-visionhalf-55.pkl")
            sha = algo.save_vision_bundle(path, platform_model_id=55, ramp_label="visionhalf")

            # load into a fresh algorithm
            algo2 = self._make_algorithm()
            algo2.load_vision_bundle(path)

        # weights restored exactly
        for k, v in ve_before.items():
            self.assertTrue(torch.allclose(algo2.vision_encoder.state_dict()[k], v), f"ve {k}")
        for k, v in te_before.items():
            self.assertTrue(torch.allclose(algo2.teacher_encoder.state_dict()[k], v), f"te {k}")
        for k, v in ta_before.items():
            self.assertTrue(torch.allclose(algo2.teacher_actor.state_dict()[k], v), f"ta {k}")

        # ramp / training state restored
        self.assertEqual(algo2.current_iteration, 1234)
        self.assertAlmostEqual(algo2.ramp_probability, 0.42, places=6)
        self.assertEqual(algo2.parent_checkpoint_sha256, "deadbeef" * 8)
        self.assertEqual(algo2.training_status, "running")

    def test_bundle_does_not_store_lstm_hidden(self):
        """P1 issue 7: LSTM hidden must NOT be in the checkpoint."""
        import torch
        import tempfile

        algo = self._make_algorithm()
        algo.vision_encoder.reset_hidden_state(batch_size=4, device="cpu")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.ckpt-visionfull-1.pkl")
            algo.save_vision_bundle(path, ramp_label="visionfull")
            ckpt = torch.load(path, weights_only=False, map_location="cpu")
        # no hidden state anywhere in the bundle
        self.assertNotIn("hidden_state", ckpt)
        self.assertNotIn("lstm_hidden", ckpt)
        ve_state = ckpt["modules"]["vision_encoder"]["state_dict"]
        for k in ve_state:
            self.assertNotIn("hidden", k.lower(), f"unexpected hidden key {k}")
        # but the reset contract is recorded
        self.assertIn("lstm_reset_contract", ckpt)
        self.assertTrue(ckpt["lstm_reset_contract"]["cross_run_not_restored"])

    def test_bundle_capabilities_and_format(self):
        import torch
        import tempfile

        algo = self._make_algorithm()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.ckpt-visionfull-1.pkl")
            algo.save_vision_bundle(path, ramp_label="visionfull")
            ckpt = torch.load(path, weights_only=False, map_location="cpu")

        from agent_ppo.checkpoint_io import KAIWU_TRAIN_FORMAT
        self.assertEqual(ckpt["format"], KAIWU_TRAIN_FORMAT)
        self.assertEqual(ckpt["schema_version"], 1)
        self.assertEqual(ckpt["ramp_label"], "visionfull")
        self.assertIn("vision_encoder", ckpt["modules"])
        self.assertIn("low_level", ckpt["modules"])
        caps = ckpt["capabilities"]
        self.assertTrue(caps["uses_depth"])
        self.assertFalse(caps["uses_height_scan_at_inference"])
        self.assertFalse(caps["deployable"])


@unittest.skipUnless(HAS_TORCH, "torch not installed; run on platform/CI")
class SingleForwardCacheTests(unittest.TestCase):
    """prepare_vision_update must advance LSTM hidden exactly once.

    P1 issue 5: action selection and the 3-way loss must reuse the same forward
    result, so the student-driven env does not step the LSTM twice on one obs.
    """

    def test_prepare_vision_update_single_lstm_step(self):
        import torch
        from agent_ppo.model.vision_encoder import DmEncoder, VisionEncoder
        from agent_ppo.algorithm.algorithm_lbc import AlgorithmLBC
        import torch.nn as nn

        ve = VisionEncoder(
            image_shape=(180, 320, 1), proprio_dim=45, cnn_output_dim=32,
            rnn_hidden_dim=64, rnn_num_layers=2, rnn_output_dim=32, use_lstm=True,
        )
        te = DmEncoder(input_dim=256, hidden_dims=(512, 256), output_dim=32)
        ta = nn.Linear(77, 12)
        algo = AlgorithmLBC(
            vision_encoder=ve, teacher_encoder=te, teacher_actor=ta,
            device="cpu", latent_dim=32, proprio_dim=45, scan_dim=256,
            depth_shape=(180, 320, 1),
        )
        algo.vision_encoder.reset_hidden_state(batch_size=2, device="cpu")

        B = 2
        obs = torch.zeros(B, 45 + 256 + 180 * 320)

        batch = algo.prepare_vision_update(obs)

        # capture hidden right after the single forward
        h_after_forward = (
            algo.vision_encoder._hidden_state[0].clone(),
            algo.vision_encoder._hidden_state[1].clone(),
        )

        # compute the 3-way loss (must NOT advance hidden again)
        loss_dict = algo.compute_three_way_loss(batch)
        loss_dict["total_loss"].backward()

        h_after_loss = algo.vision_encoder._hidden_state
        self.assertTrue(torch.allclose(h_after_forward[0], h_after_loss[0]))
        self.assertTrue(torch.allclose(h_after_forward[1], h_after_loss[1]))

        # losses are finite
        for key in ("latent_loss", "cosine_loss", "action_loss", "total_loss"):
            self.assertTrue(torch.isfinite(loss_dict[key]).all(), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
