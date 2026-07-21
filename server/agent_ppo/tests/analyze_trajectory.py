#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Offline analyzer for trajectory + death metadata embedded in model.ckpt.

用法:
  python3 agent_ppo/tests/analyze_trajectory.py --ckpt <path>/model.ckpt-12345.pkl

  - 存完整轨迹 (trajectory_metadata) + 死亡原因 + 死亡位置 + 死亡地形

读取:
  ckpt = {
    "model_state_dict": <weights>,
    "schema_version": 3,
    "storage_format": "sparse_episode_metrics_with_trajectory",
    "reward_state_metadata": <drone-style scalar aggregates>,
    "trajectory_metadata": [
      {
        "env_idx": int,
        "done_reason": "goal_reached" | "timeout" | "fallen_base" | "fallen_orientation" | "abnormal",
        "done_terminator": str,
        "done_position_world": [x, y, z] | None,
        "done_yaw": float | None,
        "done_terrain_type": str,
        "done_terrain_level": int,
        "done_track_segment": int,
        "done_goal_dist": float,
        "done_goal_pos": [x, y, z] | None,
        "episode_length": int,
        "trajectory_xy": [[x0, y0], [x1, y1], ...],
        "trajectory_yaw": [...],
        "trajectory_term_rewards": {term_name: [per-step floats]},
        "terms": {term_name: episode_sum},
      },
      ...
    ]
  }

输出:
  - 死亡原因分布
  - 死亡地形 × 死亡原因交叉表
  - 按地形计算平均 episode_length
  - 成功 episode 的轨迹长度 + 距出口距离
  - (可选) matplotlib 散点图 + 轨迹图
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


def load_ckpt(ckpt_path: Path) -> dict:
    """Load checkpoint, handling both v2 (drone) and v3 (ours) schemas."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        # Legacy: pure state_dict
        return {"model_state_dict": ckpt, "trajectory_metadata": [], "reward_state_metadata": {}}
    return ckpt


def print_header(ckpt_path: Path, ckpt: dict) -> None:
    print("=" * 72)
    print(f"  TRAJECTORY ANALYSIS: {ckpt_path.name}")
    print("=" * 72)
    schema = ckpt.get("schema_version", "N/A")
    storage = ckpt.get("storage_format", "N/A")
    print(f"  schema_version: {schema}")
    print(f"  storage_format: {storage}")

    meta = ckpt.get("reward_state_metadata", {}) or {}
    print(f"  Scalar episodes (drone-style): {meta.get('completed_env_count', 'N/A')}")

    traj = ckpt.get("trajectory_metadata", []) or []
    print(f"  Episodes with trajectory: {len(traj)}")

    if not traj:
        print("\n  No trajectory data found in this checkpoint.")
    print()


def analyze_death_reasons(traj: list[dict]) -> None:
    print("=" * 72)
    print("  Death Reason Distribution")
    print("=" * 72)
    if not traj:
        return
    reasons = Counter(ep["done_reason"] for ep in traj)
    total = len(traj)
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {reason:<30s} {count:5d} ({pct:5.1f}%)")
    print()


def analyze_terrain_deaths(traj: list[dict]) -> None:
    print("=" * 72)
    print("  Death by Terrain (top 20)")
    print("=" * 72)
    if not traj:
        return
    terrain_deaths = Counter(
        (ep.get("done_terrain_type", "unknown"), ep.get("done_reason", "unknown"))
        for ep in traj
    )
    for (terrain, reason), count in sorted(terrain_deaths.items(), key=lambda x: -x[1])[:20]:
        print(f"  {terrain:<25s} {reason:<25s} {count:4d}")
    print()


def analyze_terrain_lengths(traj: list[dict]) -> None:
    print("=" * 72)
    print("  Mean Trajectory Length by Terrain")
    print("=" * 72)
    if not traj:
        return
    terrain_lens: dict[str, list[int]] = defaultdict(list)
    for ep in traj:
        t = ep.get("done_terrain_type", "unknown")
        terrain_lens[t].append(len(ep.get("trajectory_xy", [])))
    for t, lens in sorted(terrain_lens.items()):
        avg = sum(lens) / max(len(lens), 1)
        print(f"  {t:<25s} {avg:6.0f} steps (n={len(lens)})")
    print()


def analyze_goal_reached(traj: list[dict]) -> None:
    print("=" * 72)
    print("  Goal-Reached Episodes")
    print("=" * 72)
    if not traj:
        return
    reached = [ep for ep in traj if ep.get("done_reason") == "goal_reached"]
    print(f"  Total reached: {len(reached)} / {len(traj)}")
    if not reached:
        return
    reach_lens = [len(ep.get("trajectory_xy", [])) for ep in reached]
    reach_dists = [ep.get("done_goal_dist", -1.0) for ep in reached if ep.get("done_goal_dist", -1) >= 0]
    print(f"  Mean trajectory length: {sum(reach_lens) / max(len(reach_lens), 1):.0f} steps")
    print(f"  Mean final goal dist:   {sum(reach_dists) / max(len(reach_dists), 1):.2f}m")
    # Sample first 5
    for ep in reached[:5]:
        pos = ep.get("done_position_world", "N/A")
        gp = ep.get("done_goal_pos", "N/A")
        print(f"    ep_len={ep.get('episode_length'):4d}  "
              f"final_pos={pos}  goal={gp}  dist={ep.get('done_goal_dist', -1):.2f}m")
    print()


def analyze_term_rewards(traj: list[dict]) -> None:
    print("=" * 72)
    print("  Per-Term Reward Statistics (from trajectory_term_rewards)")
    print("=" * 72)
    if not traj:
        return
    term_sums: dict[str, list[float]] = defaultdict(list)
    for ep in traj:
        for term, vals in (ep.get("trajectory_term_rewards") or {}).items():
            if vals:
                term_sums[term].append(sum(vals))
    if not term_sums:
        print("  (no per-term reward data)")
        return
    print(f"  {'Term':<35} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    for term, vals in sorted(term_sums.items(), key=lambda x: -abs(sum(x[1]) / max(len(x[1]), 1)))[:20]:
        import math
        m = sum(vals) / max(len(vals), 1)
        var = sum((v - m) ** 2 for v in vals) / max(len(vals), 1)
        sd = math.sqrt(var)
        print(f"  {term:<35} {m:>10.4f} {sd:>10.4f} {min(vals):>10.4f} {max(vals):>10.4f}")
    print()


def plot_results(traj: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skip plot generation")
        return

    if not traj:
        return

    # 1. 死亡点散点图 (按 reason 着色)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    for reason in sorted(set(ep.get("done_reason", "unknown") for ep in traj)):
        pts = [
            (ep["done_position_world"][0], ep["done_position_world"][1])
            for ep in traj
            if ep.get("done_reason") == reason
            and ep.get("done_position_world")
        ]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, label=reason, s=8, alpha=0.6)
    ax.set_title("Death positions on track")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 成功 episode 的轨迹
    ax = axes[1]
    reached = [ep for ep in traj if ep.get("done_reason") == "goal_reached"]
    for i, ep in enumerate(reached[:50]):
        xs = [p[0] for p in ep.get("trajectory_xy", [])]
        ys = [p[1] for p in ep.get("trajectory_xy", [])]
        if xs and ys:
            ax.plot(xs, ys, alpha=0.3, linewidth=0.5)
    ax.set_title(f"Successful trajectories (n={min(50, len(reached))})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = out_dir / f"trajectory_analysis.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"  Saved plot: {out_png}")
    plt.close()

    # 3. (可选) 死亡地形分布柱状图
    fig, ax = plt.subplots(figsize=(12, 5))
    terrain_deaths: dict[str, Counter] = defaultdict(Counter)
    for ep in traj:
        t = ep.get("done_terrain_type", "unknown")
        r = ep.get("done_reason", "unknown")
        terrain_deaths[t][r] += 1
    terrains = sorted(terrain_deaths.keys())
    reasons = sorted({r for c in terrain_deaths.values() for r in c.keys()})
    bottom = [0] * len(terrains)
    for reason in reasons:
        counts = [terrain_deaths[t].get(reason, 0) for t in terrains]
        ax.bar(terrains, counts, label=reason, bottom=bottom)
        bottom = [b + c for b, c in zip(bottom, counts)]
    ax.set_title("Death reasons stacked by terrain")
    ax.set_ylabel("Death count")
    ax.set_xticklabels(terrains, rotation=20, ha="right")
    ax.legend()
    plt.tight_layout()
    out_png2 = out_dir / "deaths_by_terrain.png"
    plt.savefig(out_png2, dpi=120, bbox_inches="tight")
    print(f"  Saved plot: {out_png2}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Offline trajectory + death-reason analysis for quadruped ckpt"
    )
    parser.add_argument(
        "--ckpt", required=True, help="path to model.ckpt-*.pkl (v3 schema)"
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="directory to save plots (default: ckpt's parent dir)",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="skip matplotlib plot generation"
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: file not found: {ckpt_path}")
        sys.exit(1)

    ckpt = load_ckpt(ckpt_path)
    print_header(ckpt_path, ckpt)

    traj = ckpt.get("trajectory_metadata", []) or []
    if not traj:
        sys.exit(0)

    analyze_death_reasons(traj)
    analyze_terrain_deaths(traj)
    analyze_terrain_lengths(traj)
    analyze_goal_reached(traj)
    analyze_term_rewards(traj)

    if not args.no_plot:
        out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_results(traj, out_dir)


if __name__ == "__main__":
    main()
