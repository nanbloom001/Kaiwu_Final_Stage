#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""J9 fixed-learning-rate contract tests for AlgorithmPPO."""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

torch = pytest.importorskip("torch")
import torch.nn as nn

from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO


class _DummyStorage:
    """Empty rollout storage so AlgorithmPPO.learn() runs the post-loop guard only."""

    def mini_batch_generator(self, num_mini_batches, num_learning_epochs):
        return iter(())

    def recurrent_mini_batch_generator(self, num_mini_batches, num_learning_epochs):
        return iter(())


def _make_algo(
    schedule="fixed",
    learning_rate=1e-5,
    min_learning_rate=1e-5,
    max_learning_rate=1e-5,
):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    algo = AlgorithmPPO(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        learning_rate=learning_rate,
        schedule=schedule,
        min_learning_rate=min_learning_rate,
        max_learning_rate=max_learning_rate,
        num_mini_batches=1,
        num_learning_epochs=1,
    )
    algo.storage = _DummyStorage()
    return algo


def test_fixed_lr_preserved():
    algo = _make_algo(schedule="fixed", learning_rate=1e-5)
    result = algo.learn()
    assert len(result) == 3
    assert algo.learning_rate == pytest.approx(1e-5)
    for pg in algo.optimizer.param_groups:
        assert pg["lr"] == pytest.approx(1e-5)


def test_fixed_lr_violation_raises():
    algo = _make_algo(schedule="fixed", learning_rate=1e-5)
    algo.optimizer.param_groups[0]["lr"] = 9e-6
    with pytest.raises(RuntimeError, match="Fixed PPO learning-rate contract violated"):
        algo.learn()


def test_adaptive_bounds_use_configured_min_max():
    algo = _make_algo(
        schedule="adaptive",
        learning_rate=1e-5,
        min_learning_rate=1e-5,
        max_learning_rate=1e-2,
    )

    # High KL should decrease lr but be clamped at min_learning_rate.
    mu = torch.zeros(8, 2)
    sigma = torch.ones(8, 2)
    old_mu = mu.clone()
    old_sigma = torch.full((8, 2), 2.0)
    algo.learning_rate = 1e-5
    algo._update_learning_rate(mu, sigma, old_mu, old_sigma)
    assert algo.learning_rate == pytest.approx(1e-5)
    assert algo.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)

    # Low KL should increase lr but be clamped at max_learning_rate.
    old_sigma2 = torch.full((8, 2), 1.01)
    algo.learning_rate = 9e-3
    algo._update_learning_rate(mu, sigma, mu, old_sigma2)
    assert algo.learning_rate == pytest.approx(1e-2)
    assert algo.optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)


def test_update_learning_rate_noop_for_fixed():
    algo = _make_algo(schedule="fixed", learning_rate=1e-5)
    mu = torch.zeros(8, 2)
    sigma = torch.ones(8, 2)
    old_sigma = torch.full((8, 2), 2.0)
    algo._update_learning_rate(mu, sigma, mu, old_sigma)
    assert algo.learning_rate == pytest.approx(1e-5)
    assert algo.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_validate_fixed_lr_raises_on_algorithm_lr_drift():
    algo = _make_algo(schedule="fixed", learning_rate=1e-5)
    algo.learning_rate = 9e-6
    with pytest.raises(RuntimeError, match="algorithm lr="):
        algo._validate_fixed_lr()


def test_validate_fixed_lr_raises_on_optimizer_lr_drift():
    algo = _make_algo(schedule="fixed", learning_rate=1e-5)
    algo.optimizer.param_groups[0]["lr"] = 9e-6
    with pytest.raises(RuntimeError, match="optimizer lr="):
        algo._validate_fixed_lr()


def test_validate_fixed_lr_passes_when_consistent():
    algo = _make_algo(schedule="fixed", learning_rate=1e-5)
    algo._validate_fixed_lr()  # should not raise
    assert algo.learning_rate == pytest.approx(1e-5)
    for pg in algo.optimizer.param_groups:
        assert pg["lr"] == pytest.approx(1e-5)
