# P4 Maze 八小时即时命令 R4

## 目标与父模型

- 任务：`p4maze8h-instant-r4-inputfix`
- 父模型：`p4maze8h10hz_1416926-mazefinal`
- 有效训练：`28800s`；平台墙钟 `8.25h`
- 目标：消除高层 target 到低层 exec 之间未被 Actor 观察的控制延迟，降低大半径转弯、碰撞后持续
  前顶、错误方向高速行进和后段策略漂移。

本轮不加入五段地形和低层联合训练。地形固定为单段 `open_entry_maze`，20 个静态难度列，课程
关闭，128 环境，120 秒 episode。低层、NavigationEncoder、SafetyHead 和 StuckHead 全程冻结。

## 命令合同

```text
normalized action
-> finite/numerical check
-> hard map vx=[0,1.0], vy=[-0.30,0.30], wz=[-0.90,0.90]
-> policy_target_cmd3 == limited_target3 == exec_cmd3 at the 10Hz boundary
-> hold unchanged for five 50Hz low-level frames
```

取消以下机制：

- 三轴 slew；
- 反向先归零；
- 两次反向确认；
- 运行时 translation limiter；
- near-goal command rewrite；
- recovery command override。

这不是无限大 slew 的近似，而是独立版本的 `instant_hold_10hz`。动作分布、PPO log-prob、storage
action 和实际 exec 使用同一个 target。命令硬范围和 command-rate penalty 保留。旧 slew 包只能
作为 warm start；R4 exact resume 必须携带完全相同的 command digest。

Actor capability 同样始终表达真实硬映射范围：`vx` 上限为 `1.0m/s`。但 capability15 是父 Actor85
未经归一化直接读取的 observation；父模型训练时末六维为
`up=[0.30,0.30,1.00]`、`release=[0.30,0.60,2.50]`。instant 模式必须保留这组输入兼容值，不能把
10Hz 单周期完整轴范围 `[10,6,18]` 直接写入旧 Actor。后者只作为 controller 的真实物理变化率写入
command contract 和监控，不进入策略输入，也不构成新的 slew。Goal freshness 和预测风险保留在
各自的观测/诊断通道，不得通过一个与实际 mapper 不一致的隐式 speed cap 改变策略行为。

父 Actor anchor 只在父 target 由上一条 exec 按旧 slew 在一个 10Hz tick 内可达、且三轴没有符号
反转时启用。不可达的父 target 是旧控制器的未来目标，不是瞬时执行策略应当模仿的当前物理命令。

## 学习时程

| 有效时间 | Actor | Critic | Teacher/Anchor | Adapter |
|---|---:|---:|---:|---:|
| 0-30m | 冻结 | `6e-5` | 关闭 | 冻结 |
| 30m-2h | `1.5e-5` | `6e-5` | `1.25% / 0.5%` | `5e-6` |
| 2-3h | `1e-5` | `4e-5` | `1.75% / 1.0%` | 冻结 |
| 3-3.5h | `5e-6` | `3e-5` | `1.0% / 1.25%` | 冻结 |
| 3.5-8h | 冻结 | `1e-5` | 关闭 | 冻结 |

上一轮不是到 4h50 才退化，而是约 3h50 已经出现同步劣化。因此 3h 开始收敛、3h30 固定冻结
Actor。八小时是完整任务墙钟，不表示策略权重必须连续更新八小时；后半段用于 Critic 校准、稳定
统计和固定 checkpoint 对比，不能按更晚时间自动判定模型更好。

## 奖励与 reset

- success/failure/timeout/reason4：`+200/-60/-40/-75`。
- new-best：`1.0/m`，episode 上限 `+6`。failure、timeout、reason4 精确 clawback；success 保留。
- collision onset：`-0.16-0.24*severity`；persistent：`-0.06/tick`。
- stuck sustained：0.8 秒后开始，2 秒达到 `-0.04/tick`。
- predictive depth collision 保留；旧 missed-safe/Goal-safe/yaw-exit/yaw-cancellation 仅 shadow，避免
  多个 privileged reward 与五方向 Actor teacher 重复竞争。

reason4 前 30 分钟 shadow/12 秒，30 分钟后 active/12 秒，2 小时后 active/10 秒。它只接受平台
`nav_stuck_timeout` 的实际 term readback；未分类 reset 不能合成为 timeout 或 reason4。旋转 20 度
只有在墙接触 EMA 同时下降至少 30% 时才算脱离；foot-jam 只记录 shadow。

## Checkpoint 与选择

阶段标签：

```text
instantwarm -> instantadapt -> instantcorrect -> instantstable -> instantfrozen
```

checkpoint 保存命令合同、父 Actor anchor leaf/source SHA/digest、训练时钟、optimizer/RNG、奖励和
stuck 合同。保存节奏为首个约 5 分钟、之后每 10 分钟。训练后固定比较父包与 2h、3h、3.5h 和
最终包；`instantfrozen` 只表示策略已冻结，不代表它优于更早 checkpoint。

## 验收

- `policy_limited_command_mae == 0`、`policy_exec_command_mae == 0`；
- 正负 `vy/wz` reversal 在一个高层 tick 内建立；
- collision、wall-stuck、timeout 同时下降，完成率不低于父模型；
- Actor anchor 低风险误差保持有界，冻结低层 digest 不变；
- reason4 不计完成，failure/timeout/reason4 episode 的 new-best 净贡献为零；
- 3h50 之后 Actor optimizer step 与 Adam state 不再变化。

本地测试、开发容器 smoke、平台启动和固定条件评估必须分层记录。代码测试通过不能替代 Maze
完成率、碰撞率或真机证据。
