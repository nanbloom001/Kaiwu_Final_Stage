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
from agent_ppo.workflow.lbc_workflow import _student_drive_probability


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
    goal_dim = 2
    teacher_encoder = nn.Sequential(nn.Linear(scan_dim, latent_dim), nn.Tanh())
    teacher_actor = nn.Linear(proprio_dim + latent_dim + goal_dim, 12)
    return AlgorithmLBC(
        vision_encoder=_Student(latent_dim),
        teacher_encoder=teacher_encoder,
        teacher_actor=teacher_actor,
        device=torch.device("cpu"),
        learning_rate=1.0e-4,
        latent_dim=latent_dim,
        proprio_dim=proprio_dim,
        scan_dim=scan_dim,
        goal_dim=goal_dim,
        depth_shape=(2, 2, 1),
        latent_loss_type="smooth_l1",
        action_loss_type="smooth_l1",
        latent_loss_weight=0.5,
        cosine_loss_weight=0.1,
        action_loss_weight=1.0,
    )


def _obs(batch_size=6):
    return {
        "proprio": torch.randn(batch_size, 3),
        "height_scan": torch.randn(batch_size, 5),
        "goal": torch.randn(batch_size, 2),
        "depth_image": torch.randn(batch_size, 2, 2, 1),
    }


def test_d2_loss_matches_weighted_contract_and_raw_action_shape():
    algorithm = _make_algorithm()
    result = algorithm.compute_latent_loss(_obs())

    expected = (
        0.5 * result["latent_loss"]
        + 0.1 * result["cosine_loss"]
        + result["action_loss"]
    )
    assert result["loss"].item() == pytest.approx(expected.item())
    assert result["teacher_action"].shape == (6, 12)
    assert result["student_action"].shape == (6, 12)

    result["loss"].backward()
    assert algorithm.vision_encoder.latent.grad is not None
    assert all(parameter.grad is None for parameter in algorithm.teacher_encoder.parameters())
    assert all(parameter.grad is None for parameter in algorithm.teacher_actor.parameters())


@pytest.mark.parametrize(
    ("progress", "expected"),
    [
        (0.0, 0.50),
        (0.1999, 0.50),
        (0.20, 0.75),
        (0.5999, 0.75),
        (0.60, 1.00),
        (1.0, 1.00),
    ],
)
def test_dagger_schedule_boundaries(progress, expected):
    assert _student_drive_probability(
        progress,
        [0.20, 0.40, 0.40],
        [0.50, 0.75, 1.00],
    ) == pytest.approx(expected)


def test_d2_requires_visual_student_resume():
    algorithm = _make_algorithm()
    with pytest.raises(RuntimeError, match="requires a resumed visual student"):
        algorithm.assert_student_ready()

    algorithm.student_loaded = True
    algorithm.student_source = "model.ckpt-track-lbc-loco-123.pkl"
    algorithm.assert_student_ready()


def test_encoder_teacher_checkpoint_ignores_critic_side_keys():
    algorithm = _make_algorithm()
    source_encoder = nn.Sequential(nn.Linear(5, 4), nn.Tanh())
    source_actor = nn.Linear(3 + 4 + 2, 12)
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

    algorithm.load_teacher_state_dict(
        checkpoint,
        source="model.ckpt-locomotion-10288.pkl",
    )

    assert algorithm.teacher_loaded is True
    assert algorithm.teacher_source.endswith("model.ckpt-locomotion-10288.pkl")
    for key, value in source_encoder.state_dict().items():
        assert torch.equal(algorithm.teacher_encoder.state_dict()[key], value)
    for key, value in source_actor.state_dict().items():
        assert torch.equal(algorithm.teacher_actor.state_dict()[key], value)
    assert all(
        not parameter.requires_grad
        for module in (algorithm.teacher_encoder, algorithm.teacher_actor)
        for parameter in module.parameters()
    )


def test_invalid_dagger_schedule_is_rejected():
    with pytest.raises(ValueError, match="sum to 1.0"):
        _student_drive_probability(0.0, [0.2, 0.2], [0.5, 1.0])
