#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""nav_contract — 高低层接口契约 v1（权威常量源，训练/部署共用）。

本文件是 hier-nav 阶段全部接口常数的唯一权威来源：词表、48 维输入布局、
观测偏移、时序、slew/驻留、UWB 测量链成对常数、TBPTT 长度，以及供 C++
部署镜像比对的 golden vectors 生成器。任何其他文件（TOML、workflow、
NavScheduler、导出脚本、测试）与本文件不一致时，以本文件为准并修正对方。

== cnn_feat32_raw 定义（冻结语义）==

高层 48 维输入中的视觉 32 维定义为 ``vision_encoder.cnn(depth)`` 的
**原始 CNN 特征**（`model/vision_encoder.py:130`，无 L2 归一化、不经 LSTM）。

禁止使用完整 ``vision_encoder(depth, proprio)`` 的 latent32 输出：低层 LSTM
按 50Hz 逐帧推进隐状态，而部署侧高层按 5Hz 独立运行——若高层分接经过
LSTM 的 latent，训练（50Hz 推进的 LSTM 状态）与部署（高层图内无法复现
同一状态轨迹）的输入语义必然漂移。原始 CNN 特征是纯前馈、单帧自洽的，
5Hz 下训练与部署逐位一致。

== 冻结逐帧时序（唯一合法时序）==

frame t（50Hz）：
  1. exec_cmd[t] 写入低层观测副本 obs[:,6:9] / critic_obs[:,9:12]
  2. 低层 VisionEncoder+Actor 前向一次 → joint_action[t]（LSTM 恰推进一帧）
  3. 若 t 为 nav tick：depth[t] → 冻结 CNN → cnn_feat32_raw，拼
     goal4 / exec_cmd[t] / 更新前的 held_cmd / ang_vel3 / proj_grav3 → 高层 → token
  4. 更新 held token / held_cmd；新的 slew 目标自 frame t+1 起生效（一帧确定性延迟）
  5. env.step(joint_action[t])；done 环境：低层 hidden 与 NavScheduler 状态全清零
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# =========================================================================
# 词表（10 项，顺序冻结；逐项落在 command-34728 训练指令域内）
# 顺序即 categorical head 的类别编号，随权重序列化进 checkpoint，禁止重排。
# =========================================================================

VOCAB: tuple[tuple[float, float, float], ...] = (
    (0.00, 0.0, 0.00),   # 0 zero（急停/驻停；阶跃到零是低层最常见训练形态）
    (0.20, 0.0, 0.00),   # 1 forward_slow   （low_forward 桶 [0.10,0.30)）
    (0.45, 0.0, 0.00),   # 2 forward_mid    （normal_forward 桶 [0.30,0.80)）
    (0.70, 0.0, 0.00),   # 3 forward_fast   （normal_forward 桶上锚；0.8 是从未采到的开区间边界）
    (0.35, 0.0, 0.25),   # 4 forward_left   （forward_yaw 桶）
    (0.35, 0.0, -0.25),  # 5 forward_right
    (0.20, 0.0, 0.25),   # 6 creep_left     （forward_yaw 桶下沿）
    (0.20, 0.0, -0.25),  # 7 creep_right
    (0.00, 0.0, 0.25),   # 8 spin_left      （pure_yaw 桶）
    (0.00, 0.0, -0.25),  # 9 spin_right
)
# 横移 (0, ±0.15, 0) 两项暂缓：lateral 桶采样权重仅 0.10、步态质量未经 N0
# 单独验收；放行后追加到词表尾部（不得插入中间，保持既有编号稳定）。

VOCAB_SIZE = len(VOCAB)
ZERO_TOKEN_INDEX = 0
TOKEN_NAMES: tuple[str, ...] = (
    "zero", "forward_slow", "forward_mid", "forward_fast",
    "forward_left", "forward_right", "creep_left", "creep_right",
    "spin_left", "spin_right",
)

# command-34728 训练目标桶（feature/command_schedule.py:218-244 逐点核实）。
# 词表校验用；(lo, hi) 均为含端点近似（0.80 为开区间上界，用 0.79 表示）。
_COMMAND_34728_BUCKETS = (
    # (vx_lo, vx_hi, |vy|_lo, |vy|_hi, |wz|_lo, |wz|_hi)
    (0.00, 0.00, 0.00, 0.00, 0.00, 0.00),   # zero
    (0.10, 0.30, 0.00, 0.00, 0.00, 0.00),   # low_forward
    (0.30, 0.79, 0.00, 0.00, 0.00, 0.00),   # normal_forward
    (0.15, 0.55, 0.00, 0.00, 0.15, 0.30),   # forward_yaw
    (0.00, 0.00, 0.00, 0.00, 0.15, 0.30),   # pure_yaw
    (0.00, 0.00, 0.10, 0.20, 0.00, 0.00),   # lateral（词表暂未使用）
)


def validate_vocab_in_domain() -> None:
    """断言每个词表项都落在 command-34728 的某个训练目标桶内。"""

    for index, (vx, vy, wz) in enumerate(VOCAB):
        in_domain = False
        for vx_lo, vx_hi, vy_lo, vy_hi, wz_lo, wz_hi in _COMMAND_34728_BUCKETS:
            if (
                vx_lo - 1e-9 <= vx <= vx_hi + 1e-9
                and vy_lo - 1e-9 <= abs(vy) <= vy_hi + 1e-9
                and wz_lo - 1e-9 <= abs(wz) <= wz_hi + 1e-9
            ):
                in_domain = True
                break
        if not in_domain:
            raise ValueError(
                f"vocab token {index} {TOKEN_NAMES[index]}={VOCAB[index]} "
                "is outside the command-34728 trained command domain"
            )


# =========================================================================
# 高层 48 维输入布局（顺序冻结；input_layout_version 随权重序列化）
# =========================================================================

INPUT_LAYOUT_VERSION = 2  # v2 = cnn_feat32_raw 语义（v1 的 LSTM latent 分接已废弃）

CNN_FEAT_DIM = 32
NAV_INPUT_DIM = 48
NAV_LSTM_HIDDEN_SIZE = 64
NAV_LSTM_NUM_LAYERS = 2

# 48 维内部切片（起, 止）
CNN_FEAT_SLICE = (0, 32)      # cnn_feat32_raw = vision_encoder.cnn(depth)
GOAL4_SLICE = (32, 36)        # [local_x/10, local_y/10, dist/20, freshness]
EXEC_CMD_SLICE = (36, 39)     # slew 后真正执行、写入 proprio[6:9] 的当帧值
HELD_CMD_SLICE = (39, 42)     # 当前词表目标（token 查表值）
ANG_VEL_SLICE = (42, 45)      # 低层 proprio[0:3]（含其既有标度）
PROJ_GRAV_SLICE = (45, 48)    # 低层 proprio[3:6]

# =========================================================================
# 观测布局（NavPolicy/NavCriticObservationProcess 与所有消费方共同遵守）
# =========================================================================

POLICY_PROPRIO_DIM = 45
SCAN_DIM = 256
GOAL4_OBS_START = 301         # policy obs 中 goal4 的起点
GOAL4_OBS_END = 305
DEPTH_OBS_START = 305         # depth 起点（GOAL4 之后）
DEPTH_DIM = 57600             # 180 * 320
POLICY_OBS_DIM = 57905        # 45 + 256 + 4 + 57600

CRITIC_PROPRIO_DIM = 60
CRITIC_GOAL3_START = 316      # critic obs 中真值 goal3 的起点
CRITIC_NAV_PRIV_START = 319   # [available, front_score, left_score, right_score]
CRITIC_NAV_PRIV_DIM = 4
CRITIC_NAV_PRIV_SLICE = (CRITIC_NAV_PRIV_START, CRITIC_NAV_PRIV_START + CRITIC_NAV_PRIV_DIM)
CRITIC_OBS_DIM = 323          # 60 + 256 + goal3(3) + nav_scanner privilege(4)

POLICY_CMD_SLICE = (6, 9)     # 低层 policy 观测的指令位
CRITIC_CMD_SLICE = (9, 12)    # critic 60 维 proprio 布局的指令位
POLICY_ANG_VEL_SLICE = (0, 3)
POLICY_PROJ_GRAV_SLICE = (3, 6)
CRITIC_LIN_VEL_SLICE = (0, 3)         # 特权真值线速度（仅 Oracle/监控）
CRITIC_SCAN_SLICE = (60, 316)         # 特权 height_scan（仅 Oracle/监控）

# =========================================================================
# 时序常数
# =========================================================================

CONTROL_FREQ_HZ = 50
NAV_PERIOD_FRAMES = 10        # 高层每 10 个低层帧决策一次 → 5Hz
NAV_TICK_HZ = CONTROL_FREQ_HZ / NAV_PERIOD_FRAMES
MIN_DWELL_TICKS = 10          # 最小驻留 10 tick = 2.0s（首训参数，非永久接口）
TBPTT_T = 16                  # 一个 TBPTT 段 = 16 个 nav tick = 160 低层帧

# =========================================================================
# 指令交接：训练域白名单 clamp + slew（训练/部署逐帧镜像的成对常数）
# =========================================================================

CMD_CLAMP_MIN = (0.0, -0.2, -0.3)
CMD_CLAMP_MAX = (0.7, 0.2, 0.3)

# 每秒变化率（单位 m/s^2 与 rad/s^2 等效表述）：加速保守、减速快。
SLEW_RATE_UP = (0.4, 0.4, 0.5)      # |cmd| 增大方向
SLEW_RATE_DOWN = (1.2, 1.2, 1.2)    # |cmd| 减小方向（含过零）
# zero token 例外：立即覆盖（阶跃到零是低层最常见训练形态，安全且用于急停）。
ZERO_TOKEN_BYPASSES_SLEW = True

FRAME_DT_S = 1.0 / CONTROL_FREQ_HZ

# =========================================================================
# UWB 模拟测量链成对常数（训练侧生成 == 部署侧解读；TODO 项须在部署联调时
# 与 deploy config 逐项配对校准，配对前使用保守值）
# =========================================================================

UWB_RATE_RANGE_HZ = (3.0, 8.0)        # 每 episode 采样一次等效 UWB 率
UWB_BEARING_NOISE_STD_RAD = 0.015     # 沿用历史 goal_noise 已验证量级
UWB_DISTANCE_NOISE_STD_M = 0.03
UWB_BEARING_BIAS_STD_RAD = 0.020
UWB_BEARING_BIAS_CLIP_RAD = 0.060
UWB_DISTANCE_BIAS_STD_M = 0.050
UWB_DISTANCE_BIAS_CLIP_M = 0.150
UWB_HEADING_NOISE_STD_RAD = 0.010     # body 系变换用带噪 heading（模拟 IMU yaw 漂移）
UWB_FILTER_TAU_S = 0.15               # 低通滤波时间常数（TODO：与 deploy 配对）
UWB_STALE_TIMEOUT_S = 0.5             # freshness 开始衰减（TODO：与 deploy 配对）
UWB_HOLD_TIMEOUT_S = 2.0              # freshness 衰减到 0（TODO：与 deploy 配对）
UWB_DROPOUT_RATE_PER_S = 0.05         # 丢帧段泊松到达率（每秒）
UWB_DROPOUT_DURATION_S = (0.3, 2.0)   # 丢帧段时长均匀采样
GOAL_XY_SCALE_M = 10.0                # encode: local_xy / 10, clamp ±1
GOAL_DIST_SCALE_M = 20.0              # encode: dist / 20, clamp [0,1]


def high_level_checkpoint_contract() -> dict:
    """Return every inference-affecting high-level checkpoint invariant."""

    return {
        "input_layout_version": INPUT_LAYOUT_VERSION,
        "nav_input_dim": NAV_INPUT_DIM,
        "policy_obs_dim": POLICY_OBS_DIM,
        "critic_obs_dim": CRITIC_OBS_DIM,
        "vocab": [list(v) for v in VOCAB],
        "token_names": list(TOKEN_NAMES),
        "vocab_size": len(VOCAB),
        "nav_lstm_hidden_size": NAV_LSTM_HIDDEN_SIZE,
        "nav_lstm_num_layers": NAV_LSTM_NUM_LAYERS,
        "nav_period_frames": NAV_PERIOD_FRAMES,
        "min_dwell_ticks": MIN_DWELL_TICKS,
        "control_freq_hz": CONTROL_FREQ_HZ,
        "frame_dt_s": FRAME_DT_S,
        "slew_rate_up": list(SLEW_RATE_UP),
        "slew_rate_down": list(SLEW_RATE_DOWN),
        "cmd_clamp_min": list(CMD_CLAMP_MIN),
        "cmd_clamp_max": list(CMD_CLAMP_MAX),
        "zero_token_bypasses_slew": ZERO_TOKEN_BYPASSES_SLEW,
        "cnn_feature_semantics": "vision_encoder.cnn(depth)_raw_v1",
        "input_slices": {
            "cnn": list(CNN_FEAT_SLICE),
            "goal4": list(GOAL4_SLICE),
            "exec_cmd": list(EXEC_CMD_SLICE),
            "held_cmd": list(HELD_CMD_SLICE),
            "ang_vel": list(ANG_VEL_SLICE),
            "projected_gravity": list(PROJ_GRAV_SLICE),
        },
        "uwb_measurement_contract": {
            "rate_range_hz": list(UWB_RATE_RANGE_HZ),
            "bearing_noise_std_rad": UWB_BEARING_NOISE_STD_RAD,
            "distance_noise_std_m": UWB_DISTANCE_NOISE_STD_M,
            "bearing_bias_std_rad": UWB_BEARING_BIAS_STD_RAD,
            "bearing_bias_clip_rad": UWB_BEARING_BIAS_CLIP_RAD,
            "distance_bias_std_m": UWB_DISTANCE_BIAS_STD_M,
            "distance_bias_clip_m": UWB_DISTANCE_BIAS_CLIP_M,
            "heading_noise_std_rad": UWB_HEADING_NOISE_STD_RAD,
            "filter_tau_s": UWB_FILTER_TAU_S,
            "stale_timeout_s": UWB_STALE_TIMEOUT_S,
            "hold_timeout_s": UWB_HOLD_TIMEOUT_S,
            "dropout_rate_per_s": UWB_DROPOUT_RATE_PER_S,
            "dropout_duration_s": list(UWB_DROPOUT_DURATION_S),
            "goal_xy_scale_m": GOAL_XY_SCALE_M,
            "goal_dist_scale_m": GOAL_DIST_SCALE_M,
        },
    }

# =========================================================================
# Golden vectors —— NavScheduler / C++ 部署镜像的逐帧行为基准
# =========================================================================


def _reference_slew_step(exec_cmd, target, is_zero_token):
    """纯 Python 参考实现（独立于 NavScheduler，供交叉验证，勿改成调用它）。"""

    if ZERO_TOKEN_BYPASSES_SLEW and is_zero_token:
        return [0.0, 0.0, 0.0]
    result = []
    for axis in range(3):
        current = exec_cmd[axis]
        goal = target[axis]
        delta = goal - current
        if delta == 0.0:
            result.append(current)
            continue
        # 判定加速（|cmd| 增大且同号）还是减速（|cmd| 减小或过零）
        moving_away_from_zero = abs(goal) > abs(current) and (
            current == 0.0 or (current > 0) == (goal > 0)
        )
        rate = SLEW_RATE_UP[axis] if moving_away_from_zero else SLEW_RATE_DOWN[axis]
        step = rate * FRAME_DT_S
        if abs(delta) <= step:
            result.append(goal)
        else:
            result.append(current + math.copysign(step, delta))
    return result


def _clamp_cmd(cmd):
    return [
        min(max(cmd[axis], CMD_CLAMP_MIN[axis]), CMD_CLAMP_MAX[axis])
        for axis in range(3)
    ]


def generate_golden_vectors(num_frames: int = 400) -> dict:
    """确定性 token 脚本 → 逐帧期望 exec_cmd 轨迹（含驻留与一帧延迟语义）。

    脚本固定、无随机数（部署侧 C++ 测试回放同一脚本比对逐帧输出）。
    返回 dict：{"token_script": [(tick, token)...], "frames": [{...}]}
    """

    # tick 序号 → 请求 token（请求会被最小驻留掩码约束；zero 例外立即生效）
    token_script = {
        0: 3,    # forward_fast
        4: 8,    # 驻留未满，请求被压制，held 保持 3
        10: 4,   # forward_left（驻留已满）
        20: 0,   # zero（急停例外，立即生效且旁路 slew）
        22: 2,   # forward_mid（zero 之后 dwell 重新计数：22-20=2 < 10 → 压制）
        30: 2,   # forward_mid（驻留满，生效）
        36: 9,   # spin_right（驻留未满 → 压制）
    }

    frames = []
    exec_cmd = [0.0, 0.0, 0.0]
    held_token = ZERO_TOKEN_INDEX
    dwell = MIN_DWELL_TICKS  # 起始视为驻留已满
    pending_target = list(VOCAB[held_token])
    pending_is_zero = held_token == ZERO_TOKEN_INDEX

    for frame in range(num_frames):
        # 步骤 1/2：当帧 exec_cmd 生效（写观测、驱动低层）
        frames.append(
            {
                "frame": frame,
                "exec_cmd": [round(v, 6) for v in exec_cmd],
                "held_token": held_token,
            }
        )
        # 步骤 3/4：nav tick 决策，新目标 frame t+1 起生效
        if frame % NAV_PERIOD_FRAMES == 0:
            tick = frame // NAV_PERIOD_FRAMES
            requested = token_script.get(tick)
            if requested is not None and requested != held_token:
                allowed = dwell >= MIN_DWELL_TICKS or requested == ZERO_TOKEN_INDEX
                if allowed:
                    held_token = requested
                    dwell = 0
            dwell += 1
            pending_target = _clamp_cmd(list(VOCAB[held_token]))
            pending_is_zero = held_token == ZERO_TOKEN_INDEX
        # 步骤 5 之后、下一帧步骤 1 之前：slew 演化
        exec_cmd = _reference_slew_step(exec_cmd, pending_target, pending_is_zero)

    return {
        "contract": {
            "input_layout_version": INPUT_LAYOUT_VERSION,
            "vocab": [list(v) for v in VOCAB],
            "nav_period_frames": NAV_PERIOD_FRAMES,
            "min_dwell_ticks": MIN_DWELL_TICKS,
            "slew_rate_up": list(SLEW_RATE_UP),
            "slew_rate_down": list(SLEW_RATE_DOWN),
            "zero_token_bypasses_slew": ZERO_TOKEN_BYPASSES_SLEW,
            "cmd_clamp_min": list(CMD_CLAMP_MIN),
            "cmd_clamp_max": list(CMD_CLAMP_MAX),
        },
        "token_script": sorted(token_script.items()),
        "frames": frames,
    }


def write_golden_vectors(path: str | Path, num_frames: int = 400) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(generate_golden_vectors(num_frames), stream, indent=1)
    return path
