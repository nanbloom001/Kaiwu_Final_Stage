# P3 Standard 低层 Sim2Real 与径向导航 2.5 小时训练方案

> 日期：2026-07-31
> 状态：核心代码已实现并完成本地验证，待开发容器和平台验证
> 建议任务名：`p3std2h30-s2r-radial`
> 父包：`p3nav8h-r1_884257`
> 有效训练时间：`9000s`，其中低层/Adapter `5400s`，高层/Adapter `3600s`

> 实施边界：径向 M1/M2/M3、软制动、Adapter 更新闭环、平台支持的
> friction/base-mass/noise、奖励与面板口径已在 `codex/p3-sim2real-radial-nav` 落地。
> 平台覆盖 `base_env.py`，因此 push、action delay/gain、PD、COM 和额外相机随机化未纳入
> 本轮运行合同；新的 gait-baseline 奖励等待评估视频确定父模型基线后再实现，现阶段保留轻量
> 既有步态约束与完整诊断。不得把这些未实现项描述为已生效。

## 1. 本轮目标

1. 继续降低低层能耗、力矩峰值、action rate 和 jerk，缩小 Sim2Real gap。
2. 改善四足接触占空比、对角腿步频、长期悬空和 slip 对称性，但不针对某条腿单独塑形。
3. 强化姿态恢复和 push 抗扰能力，不妨碍坡面、台阶必要的 pitch 调整。
4. 修复 P3 随机局部目标与 Standard 平台 `3.9m` 走穿条件不一致的问题。
5. 修复 Adapter 早期未真正更新、Standard success 内部计数为零、reward 聚合口径不一致和空面板。
6. 所有表现指标只用于分析，不设自动早停或分数门禁。

## 2. 平台 Standard 完成阈值与 M3 设计

已有官方规则说明：Standard 从地形块中心附近出生，相对出生点的二维欧氏距离达到“半块长度减 `0.1m`”即走穿。默认地形块为 `8m`，对应 `3.9m`。

官方规则已确认完成阈值；本地平台源码镜像中的 Unitree RL Lab 终止项同时确认：默认 `8m` 地形超过 `|x|>4m` 或 `|y|>4m` 会作为 timeout 处理。因此：

- `4.2m` 不能作为安全的 M3 目标，它已经超过默认地形边界。
- 代理完成条件必须与平台阈值使用同一公式，不能再硬编码另一个半径。
- 控制目标半径与完成判定半径要分开。

实施时从运行时 terrain generator 读取 size：

```text
half_extent = terrain_size_x / 2
platform_complete_radius = half_extent - 0.10
platform_boundary_radius = half_extent

默认8m地形：
platform_complete_radius = 3.90m
platform_boundary_radius = 4.00m
```

M3 的控制和判定：

```text
M3 virtual target radius = min(complete_radius + 0.03, boundary_radius - 0.03)
M3 proxy success        = radial_distance >= complete_radius

默认对应：
virtual target = 3.93m
proxy success  = 3.90m
```

为避免 5Hz 高层命令在一个 0.2s tick 内越过 `4.0m`，增加终点前软刹车：

```text
r < complete_radius - 0.20m：保持正常命令
complete_radius - 0.20m <= r < complete_radius - 0.04m：向外线速度上限0.30m/s
complete_radius - 0.04m <= r < complete_radius：向外线速度上限0.12m/s
r >= complete_radius：target/exec立即归零，不等待普通slew
```

`proxy success` 使用 step 后、auto-reset 前的 terminal-safe root position，每个 episode 只触发一次。触发后保持零命令并停止 push，等待平台 scorer 结算。

证据边界如下：

- [`项目简介.md`](../规则说明/开发指南/项目简介.md) 已明确 Standard 使用“出生点二维欧氏距离达到半块长度减 `0.1m`”作为走穿阈值。
- 本地镜像 `velocity_env_cfg.py::_terrain_bounds_termination()` 已确认越过半块 X/Y 边界使用严格 `>`，并被标记为 timeout。
- 当前镜像清单只覆盖 `/data/projects/legged_robot_competition_26/isaac_env`，没有收录 `tools/base_env/base_scorer.py`。因此 scorer 内部究竟写成 `>=3.9m` 还是 `>3.9m`，不能伪装成源码已确认。

实现不依赖精确踩中浮点边界：控制目标取 `3.93m`，内部 proxy 阈值仍使用平台公式 `3.90m`，并通过 `3.89/3.90/3.91m` 边界 fixture 与平台窗口结果校准。若平台实际采用严格 `>`，`3.93m` 的 `0.03m` 余量仍能触发完成；同时距离 `4.0m` timeout 边界保留 `0.07m` 制动余量。

## 3. 径向里程碑

每次 reset 保存真实 `episode_origin_xy`，不使用介质块理论中心替代实际出生点。

```text
M1：相对出生点半径1.3-1.6m，距目标<=0.5m
M2：相对出生点半径2.5-2.9m，距目标<=0.5m
M3：平台公式的complete_radius，以径向距离判定
```

目标方向保持连续。局部目标超时时保持当前半径层级，只允许从原方向左右 `30°/60°` 重规划，不能重新采样到后方。重规划不清空历史最大径向距离。

## 4. 训练时间表

| 有效时间 | 训练模块 | 主要目标 |
|---|---|---|
| 0-30m | 低层 + Adapter | 能耗、力矩、平滑、基础对称性 |
| 30-60m | 低层 + Adapter | 强 DR、push、左右转向与横移可控性 |
| 60-90m | 低层 + Adapter | 强 DR 下收敛，避免标称能力退化 |
| 90-100m | 高层 Critic + Adapter | 新径向奖励校准，Actor/CNN冻结 |
| 100-120m | 高层 Actor/Critic/CNN + Adapter | 径向里程碑 warm-up |
| 120-150m | 高层完整训练 + Adapter | Standard 走穿闭环 |

低层 CNN 始终冻结。保留低层 Critic 和兼容 optimizer moments；高层奖励合同变化较大，重建高层 Critic 和 return/value statistics，保留高层 Actor、LSTM 和 NavigationEncoder。

## 5. 低层 tracking、Sim2Real、步态与姿态奖励

```text
track_lin_vel_xy：weight 3.0 -> 2.5，std 0.25 -> 0.22
track_ang_vel_z：weight 1.0 -> 1.25，std保持0.25

energy：               -1e-5 -> -1.2e-5
joint_acc：            -1e-6 -> -1.25e-6
sustained torque：      0.05 -> 0.06
torque peak：           0.02，保持
action rate：           0.03 -> 0.04
action jerk：           0.01 -> 0.015
sim2real frame cap：    0.12，保持
```

关闭重复且强的旧步态塑形：

```text
air_time_variance_penalty = 0
max_foot_air_time = 0
foot_symmetry = 0
trot_gait = 0
```

保留 `feet_air_time`、`foot_contact_participation`、`max_continuous_air_time_penalty` 和轻量 `contact_duty_factor_symmetry`。新增 1.5 秒窗口的 `gait_baseline_excess`，分别使用 straight/turn/lateral 父模型 P99：

```text
gait_penalty = -(
    0.30*duty_excess
  + 0.30*diagonal_frequency_excess
  + 0.25*prolonged_air_excess
  + 0.15*slip_excess
)
```

正常父模型范围内严格为零，加权后单帧下限为 `-0.05`。不对某条腿单独加权，不规定固定 trot 相位。

姿态项：

```text
posture_stability = roll_like + 0.35*pitch_like + 0.15*(wx^2 + wy^2)
flat_orientation weight = -0.05
p3_posture_stability weight = -0.25
姿态项加权单帧下限 = -0.06
```

不增加固定世界高度的 base-height 奖励，避免误罚坡面和台阶动作。

## 6. 域随机化与 push

本轮不低于上一轮终点强度：

| 时间 | friction | added mass | noise | push |
|---|---|---:|---:|---:|
| 0-30m | `[0.60,1.30]` | `+-0.75kg` | `0.50` | `0.30m/s` |
| 30-150m | `[0.55,1.35]` | `+-0.85kg` | `0.55` | `0.35m/s` |

新增高优先级执行与传感随机化：

```text
action delay：0/1/2帧 = 30%/40%/30%
global action gain：[0.92,1.08]
每腿action gain残差：+-3%
joint position bias：+-0.008rad
joint velocity scale：[0.97,1.03]
IMU gyro bias：+-0.015rad/s
gravity小角偏置：+-1deg
```

PD `[0.90,1.10]` 和 COM `xy+-0.015m/z+-0.010m` 只在运行时 readback 验证生效后启用。不允许只写 TOML 就宣称随机化生效。

Push 使用随机方向、30%-35% 环境、10-14 秒间隔和 reset 后 1.5 秒 grace。当前实现将 `push_robots` 强制设为 `false`，必须建立真正的 P3 worker 消费者或使用已验证的平台 event term，并以 push 前后 root velocity 变化作为生效证据。

高层阶段新增轻量视觉随机化：

```text
相机pitch/yaw误差：+-1.5deg
相机位置误差：+-0.01m
depth scale：[0.98,1.02]
depth dropout：1%-3%
depth量化：5-10mm
depth delay：0-1帧
```

## 7. 高层奖励

```text
径向new-best          +4.0 * clamp(delta_radius, 0, 0.20)
局部目标进度           正向1.5x，负向0.4x
M1/M2成功             各+1.5，每级一次
M3/platform proxy成功 +15.0，每episode一次
局部目标超时           -0.5
平台episode超时        -4.0
摔倒/硬失败             -8.0
时间成本                -0.02 / 5Hz tick
```

不增加绝对距离、heading、速度幅值、横移幅值或转向幅值奖励。径向进度只奖励历史最大半径的新增部分，防止往返刷分。

## 8. Adapter 修复与更新

修复 defer 导致低层和校准阶段 `adapter_updates=0` 的问题。每次低层 optimizer step 后：

1. 重新计算低层 digest 并增加 version。
2. 清除未完成 future history。
3. 保留 completed records。
4. 使用独立 optimizer 执行 Adapter update。

Batch 比例：

```text
50% 最新低层版本
25% 近期P3版本
25% 父模型completed records
```

0-60 分钟每轮 1 次更新，60-90 分钟每轮 2 次；高层阶段每 2 个高层 rollout 更新 1 次。继续使用 reset、target-stability 和长 horizon mask。

## 9. 统一修复项

1. `p3_standard_success_count` 改用平台公式的 M3 proxy，不再依赖 Standard 不会产生的 `goal_reached` reason code。
2. 平台 `completed_count` 仍为权威结果；增加 proxy/platform 一致率和边界差值。
3. `frontier_clawback` 从 rollout 原始总和改为每有效环境、每 tick 加权均值。
4. terminal 首次 done 立即冻结命令、速度、径向半径、目标和奖励 aux，reset 后数据不得进入旧 transition。
5. 完成数分为“60 秒窗口值”和“lifetime 累计值”。
6. Reward 分解使用真实加权均值，分解和必须等于 PPO storage reward。
7. 正负 `vy/wz` 分开统计，禁止正负平均相互抵消。
8. M3 成功后、平台 reset 前的环境命令和 push 保持零值。

## 10. 面板优化

删除未启用的 P2 Track safety/reward 面板和永远为空的图。每个 line 面板不超过 20 个指标，高频数据在 GPU 聚合，每 60 秒 flush。

| 面板 | 指标 |
|---|---|
| 低层 Tracking | vx/vy/wz MAE、正负 wz MAE、overshoot、settling time |
| Sim2Real 代价 | sustained、peak、action rate、jerk、joint acc、energy |
| 力矩 | Hip/Thigh/Calf P50/P95/max、持续超限率 |
| 步态 | 四腿 duty/frequency/max-air/slip、左右/对角 excess |
| 姿态 | roll/pitch RMS、wx/wy RMS、bad orientation、fall |
| DR 实际值 | friction、mass、gain、delay、bias、noise、push |
| Push 恢复 | push delta-v、0.5/1/2s 姿态与 tracking 恢复、fall 率 |
| Adapter 健康 | attempts/applied/skipped、版本比例、MAE/NLL/coverage |
| 径向目标 | milestone、当前/最大半径、目标残差、重规划次数 |
| Standard 结果 | M1/M2、M3 proxy、平台完成/失败/超时、一致率 |
| 高层命令 | target/exec/true 三轴、正负占比、跟踪误差 |
| Reward 分解 | 所有加权均值、positive/negative、storage total |
| 性能 | samples/s、rollout/update/Adapter 时间、GPU 和 storage |

特别修复数值异常大的面板：

```text
旧：p3_frontier_clawback = 整个rollout中所有timeout env原始求和
新：reward_frontier_settlement_mean
   = sum(weighted_clawback) / valid_timeout_env_ticks
```

原始累计值只作隐藏诊断，不与 reward mean 绘制在同一纵轴。

`reward_mean` 拆成：

```text
low_reward_mean
high_reward_mean
platform_episode_reward_mean
```

没有数据的来源显式标记 unavailable，不再生成空曲线。Energy 面板同时展示 reward 加权贡献、机械功率 P50/P95 和平台 energy score。

## 11. Checkpoint 与验证

- 5 分钟首存，之后每 10 分钟保存。
- 30/60/90/100/120/150 分钟阶段边界额外保存。
- 保存低/高层、Adapter、独立 optimizer/scheduler/RNG、低层 version/digest、DR 实际值、M3 合同和面板累计状态。
- 模型 ID、请求 ID 和 lineage 只用于选择与追溯，不形成单点硬门禁。
- bundle 保持 `deployable=false`。

实施后的最小验证顺序：

1. 单测 M1/M2/M3、径向 new-best、软刹车、同 episode 不重复发放 M3。
2. 使用官方规则和本地平台镜像锁定 `3.90m` 完成公式与 `4.00m` timeout 边界；若容器可读取 scorer，再补充核验其比较符号和调用顺序，但不得以模型 ID 或 scorer 文件不可读形成训练单点阻断。
3. 验证 proxy/platform 在 `3.89/3.90/3.91/3.93/3.99/4.01m` fixture 上的一致性，特别检查 `3.93m` 能完成且不会被归为 timeout。
4. 验证 Adapter 在低层、校准和高层阶段都产生真实参数更新。
5. 验证 friction/mass/noise/push/action delay/gain 的实际 readback。
6. 验证步态映射有效，父模型基线范围内 penalty 为零。
7. 验证 reward 分解和与 PPO storage reward 一致。
8. 运行定向 pytest、Python 编译、TOML 解析和 `git diff --check`。
9. 开发容器运行 1-env smoke 和一次 128-env 低层 PPO/高层 PPO/Adapter/save-resume 联合 smoke。

正式训练不设表现门禁或自动早停。只有非有限状态、step/reset 失败、checkpoint 结构不兼容、DR/push 声明与运行时完全不一致或不可恢复 OOM 可以阻止训练。
