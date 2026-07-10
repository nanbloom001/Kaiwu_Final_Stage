#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""定量分析训练日志中的奖励贡献与动作分布。

用法:
  python agent_ppo/tests/analyze_rewards.py [--bridge-dir /tmp/xxx]
  python agent_ppo/tests/analyze_rewards.py --jsonl-file reward_terms_12345.jsonl

  - 每 episode 结束时奖励项贡献 → JSONL 桥接
  - 离线读 JSONL → 按 reward term 分组统计 mean/std/min/max
  - 按死亡原因分组 → 统计 goal_reached / timeout / fallen 比例
  - 动作分布分析 → 均值/标准差/饱和率

对于四足机器人, 重点采集:
  1. Action — joint cmd / nav cmd 的 distribution
  2. 死亡/终止原因 — fallen / timeout / goal_reached
  3. 各 reward term 的 episode 累计值
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_bridge_rows(paths: list[str]) -> list[dict]:
    """Load all JSONL rows from one or more bridge files."""
    rows = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            print(f"[WARN] File not found: {pp}")
            continue
        with pp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def analyze_death_reasons(rows: list[dict]) -> dict:
    """统计 episode 终止原因分布."""
    reason_counts = defaultdict(int)
    goal_dists_no_goal = []
    episode_lengths = {"goal_reached": [], "timeout": [], "fallen": [], "abnormal": []}

    for row in rows:
        reason = row.get("done_reason", "unknown")
        reason_counts[reason] += 1
        length = row.get("episode_length", 0)
        goal_dist = row.get("goal_dist", -1)

        if reason == "goal_reached":
            episode_lengths["goal_reached"].append(length)
        elif reason == "timeout":
            episode_lengths["timeout"].append(length)
            if goal_dist > 0:
                goal_dists_no_goal.append(goal_dist)
        elif reason == "fallen":
            episode_lengths["fallen"].append(length)
            if goal_dist > 0:
                goal_dists_no_goal.append(goal_dist)
        else:
            episode_lengths["abnormal"].append(length)
            if goal_dist > 0:
                goal_dists_no_goal.append(goal_dist)

    total = len(rows)
    result = {
        "total_episodes": total,
        "goal_rate": reason_counts["goal_reached"] / total if total else 0,
        "timeout_rate": reason_counts["timeout"] / total if total else 0,
        "fallen_rate": reason_counts["fallen"] / total if total else 0,
        "abnormal_rate": reason_counts.get("abnormal", 0) / total if total else 0,
    }

    if goal_dists_no_goal:
        result["mean_goal_dist_no_goal"] = float(np.mean(goal_dists_no_goal))
    for key, lengths in episode_lengths.items():
        if lengths:
            result[f"mean_ep_len_{key}"] = float(np.mean(lengths))

    return dict(result)


def analyze_reward_terms(rows: list[dict]) -> dict:
    """统计每个 reward term 的贡献 (per-episode 累计)."""
    term_values = defaultdict(list)

    for row in rows:
        terms = row.get("terms", {})
        for name, value in terms.items():
            try:
                term_values[name].append(float(value))
            except (TypeError, ValueError):
                pass

    result = {}
    for name, values in sorted(term_values.items()):
        arr = np.array(values)
        result[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(arr),
        }

    return result


def analyze_actions(rows: list[dict]) -> dict:
    """统计 episode 内 action 的均值和标准差."""
    action_means = []
    action_stds = []

    for row in rows:
        mean = row.get("actions_mean")
        std = row.get("actions_std")
        if mean is not None and isinstance(mean, list):
            action_means.append(mean)
        if std is not None and isinstance(std, list):
            action_stds.append(std)

    result: dict = {"action_episodes": len(action_means)}

    if action_means:
        arr = np.array(action_means)
        result["actions_dim"] = arr.shape[1] if arr.ndim == 2 else 0
        if arr.ndim == 2:
            result["actions_grand_mean"] = arr.mean(axis=0).tolist()
            result["actions_grand_std"] = arr.std(axis=0).tolist()

    if action_stds:
        arr = np.array(action_stds)
        if arr.ndim == 2:
            result["actions_in_ep_std_mean"] = arr.mean(axis=0).tolist()

    return result


def print_table(title: str, data: dict, key_width: int = 40, val_width: int = 12):
    """Print a formatted table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<{key_width}} {'Value':>{val_width}}")
    print(f"  {'-' * key_width} {'-' * val_width}")
    for k, v in data.items():
        if isinstance(v, dict):
            continue
        if isinstance(v, float):
            print(f"  {k:<{key_width}} {v:>{val_width}.4f}")
        else:
            print(f"  {k:<{key_width}} {str(v):>{val_width}}")


def print_reward_table(term_stats: dict):
    """Print reward term breakdown table."""
    print(f"\n{'=' * 70}")
    print(f"  Per-Term Reward Breakdown (per-episode accumulated)")
    print(f"{'=' * 70}")
    header = f"  {'Term':<35} {'Mean':>10} {'Std':>10} {'Min':>10}"
    print(header)
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10}")
    for name, stats in term_stats.items():
        print(f"  {name:<35} {stats['mean']:>10.5f} {stats['std']:>10.5f} {stats['min']:>10.5f}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantitative reward analysis for quadruped navigation training"
    )
    parser.add_argument(
        "--bridge-dir",
        type=str,
        default="/tmp/kaiwu_quadruped_reward_terms",
        help="Directory containing reward_terms_*.jsonl files",
    )
    parser.add_argument(
        "--jsonl-file",
        type=str,
        nargs="*",
        help="Specific JSONL file(s) to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save report to file (default: print to stdout)",
    )
    args = parser.parse_args()

    # Collect files
    if args.jsonl_file:
        paths = args.jsonl_file
    else:
        bridge_dir = Path(args.bridge_dir)
        if bridge_dir.exists():
            paths = sorted(bridge_dir.glob("reward_terms_*.jsonl"))
            paths = [str(p) for p in paths]
        else:
            print(f"[ERROR] Bridge dir not found: {bridge_dir}")
            print(f"  First run training to generate bridge files, or use --jsonl-file")
            sys.exit(1)

    if not paths:
        print(f"[ERROR] No JSONL files found")
        sys.exit(1)

    print(f"[INFO] Loading {len(paths)} bridge file(s):")
    for p in paths:
        print(f"  {p}")

    rows = load_bridge_rows(paths)
    if not rows:
        print("[ERROR] No valid rows found in bridge files")
        sys.exit(1)

    print(f"[INFO] Loaded {len(rows)} episode records")

    # Analysis 1: Death Reasons
    death_stats = analyze_death_reasons(rows)
    print_table("Episode Outcome Distribution", death_stats)

    # Analysis 2: Reward Terms
    term_stats = analyze_reward_terms(rows)
    if term_stats:
        print_reward_table(term_stats)

    # Analysis 3: Actions
    action_stats = analyze_actions(rows)
    if action_stats:
        print_table("Action Distribution", action_stats)

    # Summary insights
    total = death_stats.get("total_episodes", 0)
    if total > 0:
        print(f"\n{'=' * 70}")
        print(f"  INSIGHTS")
        print(f"{'=' * 70}")
        goal_rate = death_stats.get("goal_rate", 0) * 100
        timeout_rate = death_stats.get("timeout_rate", 0) * 100
        fallen_rate = death_stats.get("fallen_rate", 0) * 100

        if goal_rate < 30:
            print(f"  [LOW] Goal rate only {goal_rate:.1f}% — policy struggles to reach exit")
        else:
            print(f"  [OK]  Goal rate {goal_rate:.1f}%")

        if fallen_rate > 20:
            print(f"  [HIGH] Fallen rate {fallen_rate:.1f}% — locomotion fails on some terrains")
        elif fallen_rate > 5:
            print(f"  [WARN] Fallen rate {fallen_rate:.1f}% — check locomotion stability")

        if timeout_rate > 50:
            print(f"  [LOW] Timeout rate {timeout_rate:.1f}% — policy is too slow or stuck")

        # Reward analysis insights
        for name, stats in term_stats.items():
            if "wall_collision" in name.lower() and stats["mean"] < -3:
                print(f"  [HIGH WALL] '{name}' mean={stats['mean']:.3f} — too much wall contact")

    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"Total Episodes: {total}\n")
            f.write(f"Goal Rate: {death_stats.get('goal_rate', 0)*100:.1f}%\n")
            f.write(f"Timeout Rate: {death_stats.get('timeout_rate', 0)*100:.1f}%\n")
            f.write(f"Fallen Rate: {death_stats.get('fallen_rate', 0)*100:.1f}%\n")
            f.write(f"\n--- Reward Terms ---\n")
            for name, stats in term_stats.items():
                f.write(f"{name}: mean={stats['mean']:.5f} std={stats['std']:.5f}\n")
        print(f"\n[INFO] Report saved to {output_path}")


if __name__ == "__main__":
    main()
