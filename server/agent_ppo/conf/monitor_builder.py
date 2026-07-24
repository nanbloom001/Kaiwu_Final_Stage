#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    #
    # Note: this builder only keeps metrics that are unique to algorithm training
    # (loss-series metrics, episode_reward, track traversal progress).
    # Other reward_* metrics (velocity tracking, posture, gait, navigation rewards, etc.)
    # are rendered by the project-side tools/conf/monitor_default.yaml and
    # tools/conf/monitor_default_track.yaml, and are no longer redefined here,
    # to avoid duplicated panels with the same name in the final merged dashboard.
    #
    # 注意：本 builder 只保留算法训练独有的指标（loss 类、episode_reward、赛道穿越进度）。
    # 其余 reward_* 指标（速度跟踪、姿态、步态、导航奖励等）由项目侧
    # tools/conf/monitor_default.yaml 与 tools/conf/monitor_default_track.yaml 负责展示，
    # 这里不再重复定义，避免最终合并后的监控面板出现同名指标重复绘制。

    Returns:
        dict: monitor configuration dictionary
        返回值：监控配置字典
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("四足机器人导航")
        # ==============================================================
        # Group 1: Algorithm training loss metrics (unique to this builder, not covered by yaml)
        # Group 1: 算法训练损失指标（本 builder 独有，yaml 未覆盖）
        # ==============================================================
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .end_group()
        # ==============================================================
        # Group 2: Reward metrics (examples, players can add more reward panels as needed)
        # Group 2: Reward 指标（示例，选手可按需补充更多 reward 面板）
        # ==============================================================
        .add_group(group_name="奖励指标", group_name_en="reward")
        .add_panel(name="线速度跟踪奖励", name_en="reward_track_lin_vel_xy", type="line")
            .add_metric(metrics_name="reward_track_lin_vel_xy",
                        expr="avg(reward_track_lin_vel_xy{})")
            .end_panel()
        .end_group()
        .build()
    )
    return config_dict
