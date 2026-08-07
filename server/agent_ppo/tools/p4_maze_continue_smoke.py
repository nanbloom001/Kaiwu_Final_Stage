#!/usr/bin/env python3
"""Exercise a real P4 maze warm start through update, save, and resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import torch
from torch import nn

import agent_ppo.tests._nav_test_stubs  # noqa: F401
from agent_ppo.algorithm.algorithm_p4_nav_ppo import AlgorithmP4NavPPO
from agent_ppo.feature import p15_contract, p2_contract, p4_contract
from agent_ppo.feature.p2_response_buffer import P2ResponseAuxBuffer
from agent_ppo.model.p2_high_level import (
    NavigationEncoder,
    NavigationSafetyHead,
    P2NavigationActor,
    P2NavigationCritic,
    assemble_actor_input,
)
from agent_ppo.model.response_adapter import CommandResponseAdapter
from agent_ppo.model.vision_encoder import VisionEncoder
from agent_ppo.p4.profiles import (
    PROFILE_FULL_TRACK,
    PROFILE_MAZE_INSTANT_COMMAND_R4,
    PROFILE_MAZE_INSTANT_REPAIR2H,
)


def _algorithm(num_envs: int, device: str, config: dict) -> AlgorithmP4NavPPO:
    low_actor = nn.Sequential(
        nn.Linear(77, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 12),
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
        device=device,
        config=config,
    )


def _seed_response_records(algorithm: AlgorithmP4NavPPO) -> None:
    num_envs = algorithm.num_envs
    aux = torch.zeros(num_envs, p15_contract.RESPONSE_AUX_DIM)
    aux[:, 0] = 0.5
    aux[:, 3] = 0.5
    aux[:, 9] = 1.0
    aux[:, 12] = 0.3
    # The adapter sampler needs 51 future-label frames before it can emit a
    # record, then 8 burn-in + 16 training records for one recurrent batch.
    for frame in range(80):
        sample = aux.clone()
        sample[:, 15] = 0.006 * frame
        algorithm.response_buffer.append(
            sample,
            torch.zeros(num_envs, dtype=torch.bool),
            current_segment=torch.full((num_envs,), 2.0),
        )
    if not algorithm.response_buffer.ready:
        raise AssertionError(
            "response smoke fixture did not produce a complete adapter sequence"
        )


def _fill_rollout(algorithm: AlgorithmP4NavPPO) -> None:
    num_envs = algorithm.num_envs
    device = algorithm.device
    reset = torch.zeros(num_envs, dtype=torch.bool, device=device)
    actor_hidden = (
        torch.zeros(2, num_envs, 64, device=device),
        torch.zeros(2, num_envs, 64, device=device),
    )
    critic_hidden = (
        torch.zeros(2, num_envs, 64, device=device),
        torch.zeros(2, num_envs, 64, device=device),
    )
    nav_feat = torch.zeros(num_envs, p2_contract.NAV_FEATURE_DIM, device=device)
    depth = torch.zeros(
        num_envs,
        p2_contract.DEPTH_DIM,
        dtype=torch.float16,
        device=device,
    )
    nav_nonvisual = torch.zeros(
        num_envs, p2_contract.NAV_NONVISUAL_DIM, device=device
    )
    response_profile = torch.zeros(
        num_envs, p2_contract.RESPONSE_PROFILE_DIM, device=device
    )
    confidence = torch.ones(num_envs, 1, device=device)
    safety_target = torch.tensor(
        [0.2, 0.8, 0.4], dtype=torch.float32, device=device
    ).repeat(num_envs, 1)
    safety_valid = torch.ones(num_envs, 1, device=device)
    critic_input = torch.zeros(
        num_envs, p2_contract.CRITIC_INPUT_DIM, device=device
    )
    pre_tanh = torch.zeros(num_envs, p2_contract.ACTION_DIM, device=device)
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
    for tick in range(algorithm.nav_rollout_ticks):
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
            reward=torch.full(
                (num_envs, 1), 0.05 + 0.002 * tick, device=device
            ),
            duration_frames=torch.full(
                (num_envs, 1),
                algorithm.nav_period_frames,
                dtype=torch.long,
                device=device,
            ),
            bootstrap_value=old_value,
            bootstrap_mask=torch.ones(num_envs, 1, device=device),
            continuation_mask=torch.ones(num_envs, 1, device=device),
            reset_mask=reset,
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
        )


def _module_digest(algorithm: AlgorithmP4NavPPO) -> str:
    return algorithm._module_digest(
        (("vision", algorithm.low_level_encoder), ("actor", algorithm.low_level_actor))
    )


def _exercise_fp16_mirror_auxiliary(
    algorithm: AlgorithmP4NavPPO,
) -> None:
    """Deterministically exercise the rollout FP16 mirror-CNN boundary."""
    timesteps = 16
    batch_size = 1
    device = algorithm.device
    actor_input = torch.randn(
        timesteps,
        batch_size,
        p2_contract.ACTOR_INPUT_DIM,
        device=device,
    )
    hidden = (
        torch.zeros(2, batch_size, 64, device=device),
        torch.zeros(2, batch_size, 64, device=device),
    )
    reset_mask = torch.cat(
        (
            torch.ones(1, batch_size, dtype=torch.bool, device=device),
            torch.zeros(
                timesteps - 1,
                batch_size,
                dtype=torch.bool,
                device=device,
            ),
        ),
        dim=0,
    )
    pre_tanh = torch.zeros(
        timesteps,
        batch_size,
        p2_contract.ACTION_DIM,
        device=device,
    )
    _, _, mean, _, _, features = algorithm.actor.evaluate_actions(
        actor_input,
        pre_tanh,
        hidden,
        reset_mask,
        return_features=True,
    )
    nav_nonvisual = torch.zeros(
        timesteps,
        batch_size,
        p2_contract.NAV_NONVISUAL_DIM,
        device=device,
    )
    response_profile = torch.zeros(
        timesteps,
        batch_size,
        p2_contract.RESPONSE_PROFILE_DIM,
        device=device,
    )
    confidence = torch.ones(timesteps, batch_size, 1, device=device)
    batch = {
        "nav_nonvisual": nav_nonvisual,
        "response_profile": response_profile,
        "confidence": confidence,
        "pre_tanh_action": pre_tanh,
        "reset_mask": reset_mask,
        "actor_hidden": hidden,
        "camera_aux_mask": torch.zeros(
            timesteps, batch_size, 3, device=device
        ),
        "clean_action_mean": torch.zeros(
            timesteps, batch_size, 3, device=device
        ),
        "teacher_safe3": torch.zeros(
            timesteps, batch_size, 3, device=device
        ),
        "teacher_goal_xy": torch.zeros(
            timesteps, batch_size, 2, device=device
        ),
        "teacher_predictive_risk": torch.zeros(
            timesteps, batch_size, 1, device=device
        ),
        "teacher_mask": torch.zeros(
            timesteps, batch_size, 1, device=device
        ),
        "teacher_goal_mask": torch.zeros(
            timesteps, batch_size, 1, device=device
        ),
        "teacher_weight": torch.ones(
            timesteps, batch_size, 1, device=device
        ),
        "stuck_label": torch.zeros(
            timesteps, batch_size, 1, device=device
        ),
        "stuck_mask": torch.zeros(
            timesteps, batch_size, 1, device=device
        ),
        "mirror_eligible": torch.ones(
            timesteps, batch_size, 1, device=device
        ),
        "mirror_batch": {
            "depth": torch.zeros(
                timesteps,
                batch_size,
                p2_contract.DEPTH_HEIGHT,
                p2_contract.DEPTH_WIDTH,
                1,
                dtype=torch.float16,
                device=device,
            ),
            "nav_nonvisual": nav_nonvisual,
            "response_profile": response_profile,
            "confidence": confidence,
            "pre_tanh_action": pre_tanh,
            "reset_mask": reset_mask,
        },
    }
    saved_coefficients = dict(algorithm._auxiliary_coefficients)
    algorithm._auxiliary_coefficients = {"mirror": 1.0}
    algorithm.actor_optimizer.zero_grad(set_to_none=True)
    loss, metrics = algorithm._actor_auxiliary_loss(
        normalized_mean=torch.tanh(mean),
        actor_features=features,
        nav_feat=torch.zeros(
            timesteps,
            batch_size,
            p2_contract.NAV_FEATURE_DIM,
            device=device,
        ),
        batch=batch,
        ppo_actor_loss=mean.square().mean(),
    )
    loss.backward()
    if not bool(torch.isfinite(loss)):
        raise AssertionError("P4 FP16 mirror auxiliary produced non-finite loss")
    if float(metrics.get("mirror_aux_sequence_share", 0.0)) <= 0.0:
        raise AssertionError("P4 FP16 mirror auxiliary was not exercised")
    if any(
        parameter.grad is not None
        for parameter in algorithm.navigation_encoder.parameters()
    ):
        raise AssertionError("P4 mirror auxiliary updated NavigationEncoder")
    if not any(
        parameter.grad is not None for parameter in algorithm.actor.parameters()
    ):
        raise AssertionError("P4 mirror auxiliary did not update Actor")
    algorithm.actor_optimizer.zero_grad(set_to_none=True)
    algorithm._auxiliary_coefficients = saved_coefficients


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Minimal contract smoke uses 8 envs; scale tests are separate and opt-in.",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with Path(args.config).open("rb") as stream:
        full_config = tomllib.load(stream)
    config = dict(full_config["p4_nav_ppo"])
    config["track_segment_labels"] = list(
        p2_contract.canonical_track_segment_labels(
            full_config["terrain"]["track"]["sub_terrains"]
        )
    )
    training_profile = str(config.get("training_profile", PROFILE_FULL_TRACK))
    instant_r4 = training_profile == PROFILE_MAZE_INSTANT_COMMAND_R4
    instant_repair = training_profile == PROFILE_MAZE_INSTANT_REPAIR2H
    config["maze_training_branch"] = (
        "instant_repair2h"
        if instant_repair
        else ("instant_command_r4" if instant_r4 else "auto")
    )

    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    platform_model_id = raw.get("platform_model_id", 1256446)

    device = torch.device(args.device)
    cuda_smoke = device.type == "cuda"
    cuda_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    ) if cuda_smoke else None
    if cuda_smoke:
        # Isaac's prebundled torch rejects peak-stat operations before the
        # primary CUDA context exists, even when ``is_available()`` is true.
        torch.empty(0, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_index)
    algorithm = _algorithm(args.num_envs, args.device, config)
    mode = algorithm.load_bundle(
        str(checkpoint), platform_model_id=platform_model_id
    )
    expected_warm_start = (
        "p4_maze_instant_repair2h_warm_start"
        if instant_repair
        else (
            "p4_maze_instant_r4_warm_start"
            if instant_r4
            else "p4_full_track_warm_start"
        )
    )
    if mode != expected_warm_start:
        raise AssertionError(f"unexpected warm-start mode: {mode}")
    if training_profile == PROFILE_FULL_TRACK:
        _exercise_fp16_mirror_auxiliary(algorithm)
    low_digest_before = _module_digest(algorithm)

    algorithm._resolved_maze_training_branch = (
        "instant_repair2h"
        if instant_repair
        else ("instant_command_r4" if instant_r4 else "actor_attack")
    )
    algorithm.update_training_clocks(
        1_800.0
        if (instant_r4 or instant_repair)
        else p4_contract.DIAGNOSTIC_SECONDS
    )
    valid_phases = (
        {"repairtrain"}
        if instant_repair
        else (
            {"instantadapt"}
            if instant_r4
            else {"fullwarm", "fulladapt", "fulltrain", "fullstabilize"}
        )
    )
    if algorithm.current_phase not in valid_phases:
        raise AssertionError("P4 smoke failed to enter the full-track schedule")
    _seed_response_records(algorithm)
    _fill_rollout(algorithm)
    metrics = algorithm.update()
    low_digest_after = _module_digest(algorithm)
    if low_digest_before != low_digest_after:
        raise AssertionError("frozen low-level digest drifted during P4 update")
    if not (
        metrics.get("updates", 0.0) > 0.0
        and metrics.get("adapter_updates", 0.0) > 0.0
        and algorithm.actor_gradient_steps > 0
        and algorithm.critic_gradient_steps > 0
        and algorithm.adapter_gradient_steps > 0
    ):
        raise AssertionError({"metrics": metrics})

    output.parent.mkdir(parents=True, exist_ok=True)
    algorithm.save_training_bundle(str(output), platform_model_id=1207699)
    restored = _algorithm(args.num_envs, args.device, config)
    resume_mode = restored.load_bundle(str(output), platform_model_id=1207699)
    if resume_mode != "p4_exact_resume_history_reset":
        raise AssertionError(f"unexpected exact-resume mode: {resume_mode}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "num_envs": args.num_envs,
                "warm_start_mode": mode,
                "exact_resume_mode": resume_mode,
                "phase": algorithm.current_phase,
                "training_branch": algorithm._resolved_maze_training_branch,
                "ppo_updates": metrics["updates"],
                "adapter_updates": metrics["adapter_updates"],
                "actor_gradient_steps": algorithm.actor_gradient_steps,
                "critic_gradient_steps": algorithm.critic_gradient_steps,
                "adapter_gradient_steps": algorithm.adapter_gradient_steps,
                "low_digest_unchanged": low_digest_before == low_digest_after,
                "fp16_mirror_auxiliary": (
                    "not_applicable_disabled_by_r4_contract"
                    if instant_r4
                    else "PASS"
                ),
                "memory_allocated": (
                    torch.cuda.memory_allocated(cuda_index) if cuda_smoke else 0
                ),
                "memory_reserved": (
                    torch.cuda.memory_reserved(cuda_index) if cuda_smoke else 0
                ),
                "max_memory_allocated": (
                    torch.cuda.max_memory_allocated(cuda_index) if cuda_smoke else 0
                ),
                "max_memory_reserved": (
                    torch.cuda.max_memory_reserved(cuda_index) if cuda_smoke else 0
                ),
                "pinned_depth_bytes": 0
                if algorithm.rollout.depth is None
                else algorithm.rollout.depth.numel()
                * algorithm.rollout.depth.element_size(),
                "checkpoint": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
