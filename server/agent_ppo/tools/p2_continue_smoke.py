#!/usr/bin/env python3
"""Exercise a real P2 warm start through update, save, and exact resume."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import agent_ppo.tests._nav_test_stubs  # noqa: F401
from agent_ppo.feature import p15_contract, p2_contract
from agent_ppo.model.p2_high_level import assemble_actor_input
from agent_ppo.tests.test_p2_core import _make_p2_algorithm


def _same_state(module: torch.nn.Module, saved: dict[str, torch.Tensor]) -> bool:
    current = module.state_dict()
    return set(current) == set(saved) and all(
        torch.equal(current[name].cpu(), value.cpu())
        for name, value in saved.items()
    )


def _different_state(module: torch.nn.Module, saved: dict[str, torch.Tensor]) -> bool:
    current = module.state_dict()
    return set(current) != set(saved) or any(
        not torch.equal(current[name].cpu(), value.cpu())
        for name, value in saved.items()
        if name in current
    )


def _same_legacy_actor_state(
    module: torch.nn.Module, saved: dict[str, torch.Tensor]
) -> bool:
    current = module.state_dict()
    return set(saved).issubset(current) and all(
        torch.equal(current[name].cpu(), value.cpu())
        for name, value in saved.items()
    )


def _seed_response_records(algorithm) -> None:
    aux = torch.zeros(1, p15_contract.RESPONSE_AUX_DIM)
    aux[:, 0] = 0.5
    aux[:, 3] = 0.5
    aux[:, 9] = 1.0
    aux[:, 12] = 0.3
    for frame in range(64):
        sample = aux.clone()
        sample[:, 15] = 0.006 * frame
        algorithm.response_buffer.append(
            sample,
            torch.zeros(1, dtype=torch.bool),
            current_segment=torch.full((1,), 2.0),
        )


def _fill_rollout(algorithm) -> None:
    reset = torch.zeros(1, dtype=torch.bool)
    actor_hidden = (
        torch.zeros(2, 1, 64),
        torch.zeros(2, 1, 64),
    )
    critic_hidden = (
        torch.zeros(2, 1, 64),
        torch.zeros(2, 1, 64),
    )
    nav_feat = torch.zeros(1, p2_contract.NAV_FEATURE_DIM)
    depth = torch.zeros(1, p2_contract.DEPTH_DIM, dtype=torch.float16)
    nav_nonvisual = torch.zeros(1, p2_contract.NAV_NONVISUAL_DIM)
    response_profile = torch.zeros(1, p2_contract.RESPONSE_PROFILE_DIM)
    confidence = torch.ones(1, 1)
    safety_target = torch.tensor([[0.2, 0.8, 0.4]], dtype=torch.float32)
    safety_valid = torch.ones(1, 1)
    critic_input = torch.zeros(1, p2_contract.CRITIC_INPUT_DIM)
    pre_tanh = torch.zeros(1, p2_contract.ACTION_DIM)
    actor_input = assemble_actor_input(
        nav_feat,
        nav_nonvisual,
        response_profile,
        confidence,
    )
    with torch.no_grad():
        old_log_prob, _, _, _, _ = algorithm.actor.evaluate_actions(
            actor_input,
            pre_tanh,
            actor_hidden,
            reset,
        )
        old_value, _ = algorithm.critic(
            critic_input,
            critic_hidden,
            reset,
        )
    for tick in range(p2_contract.NAV_ROLLOUT_TICKS):
        algorithm.rollout.add(
            depth=depth,
            nav_feat=nav_feat,
            nav_nonvisual=nav_nonvisual,
            response_profile=response_profile,
            confidence=confidence,
            safety_target=safety_target,
            safety_valid=safety_valid,
            critic_input=critic_input,
            pre_tanh_action=pre_tanh,
            old_log_prob=old_log_prob,
            old_value=old_value,
            reward=torch.full((1, 1), 0.05 + 0.002 * tick),
            duration_frames=torch.full(
                (1, 1), p2_contract.NAV_PERIOD_FRAMES, dtype=torch.long
            ),
            bootstrap_value=old_value,
            bootstrap_mask=torch.ones(1, 1),
            continuation_mask=torch.ones(1, 1),
            reset_mask=reset,
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    high = raw["modules"]["high_level"]

    algorithm = _make_p2_algorithm(training=True)
    algorithm.load_mode = "p2_command_v2_expansion_warm_start"
    mode = algorithm.load_bundle(
        str(checkpoint),
        platform_model_id=raw.get("platform_model_id", "unknown"),
    )
    checks = {
        "navigation_encoder_preserved": _same_state(
            algorithm.navigation_encoder,
            high["navigation_encoder"]["state_dict"],
        ),
        "legacy_actor_preserved": _same_legacy_actor_state(
            algorithm.actor,
            high["actor"]["state_dict"],
        ),
        "vy_head_zero_initialized": bool(
            algorithm.actor.vy_mean_head.weight.count_nonzero() == 0
            and algorithm.actor.vy_mean_head.bias.item() == 0.0
            and math.isclose(
                algorithm.actor.vy_log_std.item(),
                p2_contract.INITIAL_VY_LOG_STD,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
        ),
        "adapter_preserved": _same_state(
            algorithm.response_adapter,
            high["response_adapter"]["state_dict"],
        ),
        "critic_rebuilt": _different_state(
            algorithm.critic,
            high["critic"]["state_dict"],
        ),
        "vy_domain_fully_open": (
            algorithm.current_vy_trusted_limit == 0.20
            and algorithm.current_vy_hard_limit == 0.40
        ),
    }
    if mode != "p2_command_v2_expansion_warm_start" or not all(checks.values()):
        raise AssertionError({"mode": mode, **checks})

    _seed_response_records(algorithm)
    _fill_rollout(algorithm)
    metrics = algorithm.update()
    if not (
        metrics["updates"] > 0
        and algorithm.actor_gradient_steps > 0
        and algorithm.critic_gradient_steps > 0
        and metrics.get("adapter_updates", 0.0) > 0
        and math.isfinite(metrics.get("safety_bce", float("nan")))
    ):
        raise AssertionError(
            {
                "metrics": metrics,
                "actor_gradient_steps": algorithm.actor_gradient_steps,
                "critic_gradient_steps": algorithm.critic_gradient_steps,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    algorithm.save_training_bundle(str(output), platform_model_id=291714)
    restored = _make_p2_algorithm(training=True)
    resume_mode = restored.load_bundle(str(output), platform_model_id=291714)
    if resume_mode != "exact_resume_history_reset":
        raise AssertionError(f"unexpected resume mode: {resume_mode}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "warm_start_mode": mode,
                "exact_resume_mode": resume_mode,
                **checks,
                "ppo_updates": metrics["updates"],
                "adapter_updates": metrics.get("adapter_updates", 0.0),
                "actor_gradient_steps": algorithm.actor_gradient_steps,
                "critic_gradient_steps": algorithm.critic_gradient_steps,
                "adapter_gradient_steps": algorithm.adapter_gradient_steps,
                "lifetime_base_seconds": algorithm.lifetime_base_seconds,
                "optimizer_migration": algorithm.optimizer_migration_report,
                "saved_checkpoint": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
