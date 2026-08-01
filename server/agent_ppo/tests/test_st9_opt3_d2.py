#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""ST9-Opt3-D2 distillation contracts."""

import os
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.algorithm.algorithm_lbc import AlgorithmLBC
from agent_ppo.workflow.lbc_workflow import _ramp_probability


class _Student(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent = nn.Parameter(torch.linspace(-0.3, 0.4, latent_dim))

    def forward(self, depth_image, proprio, masks=None, detach_hidden=True):
        return F.normalize(self.latent.unsqueeze(0).expand(proprio.shape[0], -1), dim=-1)


def _make_algorithm():
    latent_dim = 4
    proprio_dim = 3
    scan_dim = 5
    teacher_encoder = nn.Sequential(nn.Linear(scan_dim, latent_dim), nn.Tanh())
    teacher_actor = nn.Linear(proprio_dim + latent_dim, 12)
    return AlgorithmLBC(
        vision_encoder=_Student(latent_dim),
        teacher_encoder=teacher_encoder,
        teacher_actor=teacher_actor,
        device=torch.device("cpu"),
        learning_rate=1.0e-4,
        latent_dim=latent_dim,
        proprio_dim=proprio_dim,
        scan_dim=scan_dim,
        depth_shape=(2, 2, 1),
    )


def _obs(batch_size=6):
    return {
        "proprio": torch.randn(batch_size, 3),
        "height_scan": torch.randn(batch_size, 5),
        "depth_image": torch.randn(batch_size, 2, 2, 1),
    }


def test_d2_loss_matches_weighted_contract_and_raw_action_shape():
    algorithm = _make_algorithm()
    batch = algorithm.prepare_vision_update(_obs())
    result = algorithm.compute_three_way_loss(batch)

    expected = (
        0.5 * result["latent_loss"]
        + 0.1 * result["cosine_loss"]
        + result["action_loss"]
    )
    assert result["total_loss"].item() == pytest.approx(expected.item())
    assert batch["teacher_action"].shape == (6, 12)
    assert batch["student_action"].shape == (6, 12)

    result["total_loss"].backward()
    assert algorithm.vision_encoder.latent.grad is not None
    assert all(parameter.grad is None for parameter in algorithm.teacher_encoder.parameters())
    assert all(parameter.grad is None for parameter in algorithm.teacher_actor.parameters())


@pytest.mark.parametrize(
    ("elapsed_h", "expected"),
    [
        (0.0, 0.0),
        (0.4999, 0.0),
        (0.50, 0.0),
        (2.75, 0.50),
        (5.0, 1.0),
        (6.0, 1.0),
    ],
)
def test_dagger_schedule_boundaries(elapsed_h, expected):
    assert _ramp_probability(elapsed_h, 0.50, 5.0) == pytest.approx(expected)


def test_d2_freezes_teacher_modules_on_init():
    algorithm = _make_algorithm()
    assert all(
        not parameter.requires_grad
        for module in (algorithm.teacher_encoder, algorithm.teacher_actor)
        for parameter in module.parameters()
    )


def test_encoder_teacher_checkpoint_ignores_critic_side_keys(tmp_path):
    algorithm = _make_algorithm()
    source_encoder = nn.Sequential(nn.Linear(5, 4), nn.Tanh())
    source_actor = nn.Linear(3 + 4, 12)
    checkpoint = {
        **{
            f"encoder.{key}": value.clone()
            for key, value in source_encoder.state_dict().items()
        },
        **{
            f"actor.{key}": value.clone()
            for key, value in source_actor.state_dict().items()
        },
        "critic_encoder.0.weight": torch.randn(4, 5),
        "critic.0.weight": torch.randn(1, 9),
        "log_std": torch.zeros(12),
    }

    source = tmp_path / "model.ckpt-locomotion-10288.pkl"
    torch.save(checkpoint, source)
    algorithm.load_teacher_from_locomotion_ckpt(str(source))

    for key, value in source_encoder.state_dict().items():
        assert torch.equal(algorithm.teacher_encoder.state_dict()[key], value)
    for key, value in source_actor.state_dict().items():
        assert torch.equal(algorithm.teacher_actor.state_dict()[key], value)
    assert all(
        not parameter.requires_grad
        for module in (algorithm.teacher_encoder, algorithm.teacher_actor)
        for parameter in module.parameters()
    )


def test_dagger_schedule_is_clamped_outside_ramp():
    assert _ramp_probability(-1.0, 0.50, 5.0) == 0.0
    assert _ramp_probability(10.0, 0.50, 5.0) == 1.0
