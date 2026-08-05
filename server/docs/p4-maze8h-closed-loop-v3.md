# P4 Maze 八小时闭环强化 v3

## 决策与目标

- 分支：`codex/p4-maze8h-closed-loop-r3`。
- 任务：`p4maze8h-closedloop-r3`。
- 父包：`p4maze8h10hz_1416926-mazefinal`。
- 有效训练 `28800s`，平台墙钟 `8.25h`；单段 `open_entry_maze`、20 列、课程关闭、
  `120s` episode、128 env、10Hz 高层、50Hz 低层、32 tick rollout、TBPTT16、4 PPO epochs。

本轮只回答一个问题：在不改变网络、观测、动作和部署接口的前提下，是否能让既有高层策略更早
选择安全且朝向目标的出口，并降低碰撞、卡墙和超时。五段泛化、低层联合训练、额外 Camera/Goal
故障、Push 和新记忆结构均不进入本轮，避免再次混淆归因。

## 低漂移边界

冻结低层、NavigationEncoder、SafetyHead、ResponseAdapter 和 StuckHead，只训练高层 Actor 的
LSTM/action head 与 Critic。父 Actor/Critic 权重和 Critic return statistics保留；由于 reward、
slew 和训练信号合同变化，Actor/Critic Adam moments 在 warm start 时重置。模型 ID只参与候选排序，
文件、模块 spec、shape、有限值、合同和冻结低层 digest 才决定是否可加载。

本轮不允许任何未进入策略闭环的运行时动作改写：

- near-goal capture 全程 shadow；
- 平移 limiter 全程 shadow，只报告反事实 `alpha_raw`，实际 `limited_target == policy_target`；
- yaw-cancellation、missed-safe、goal-safe 和 yaw-exit 只保留诊断，PPO reward 为零；
- stale Goal cap 保留，因为 Goal freshness 已在 Actor 输入中显式可见；
- 50Hz slew 是确定性执行动力学，target/exec/true 全链路进入监控。

## 控制与教师

动作范围不变：`vx=[0,1.0]`、`vy=[-0.30,0.30]`、`wz=[-0.90,0.90]`，不开放负 `vx`。
slew 使用：

```text
increase = [0.60, 0.60, 2.00]
release  = [1.20, 1.20, 4.00]
```

五方向 training-only 教师由 `nav_scanner + height_scan` 产生
`far-left/left/center/right/far-right`。只有 scanner/mapping 有效、非 terminal/reset/grace、
最优安全度 `>=0.65` 且与次优差值足够时才参与。Goal 只在与最安全值相差不超过 `0.10` 的安全
出口间 tie-break；教师不覆盖动作，只对 Actor mean 施加 `direction/speed/yaw=0.55/0.10/0.35`
的容差损失。速度项仅在 predictive risk `>=0.90` 或确认卡滞时生效；辅助梯度硬上限为 PPO Actor
梯度的 `3%`。

## 奖励

PPO allowlist 固定为：frame safety、不可重复 new-best、success/failure/timeout、time、crawl、
command-rate、tracking、body collision、部署可得 predictive collision、持续卡滞和 reason4。
其余历史项即使仍计算诊断也强制写零。

```text
new-best       +1.0 / m，episode 上限 +6，不 clawback
success        +200
hard failure   -60
timeout        -40
reason4        -75
time           -0.02 / 10Hz tick
collision      onset -0.12 - 0.18*severity；persistent -0.05/tick
stuck          0.8s 后渐进，2s 后 -0.03/tick
```

frame safety 只有非正项。即使领取全部 `+6` new-best，timeout、failure 和 reason4 episode 的理论
上界仍严格为负；reason4 比 timeout 更差，不能通过卡墙 reset 刷取局部进展。

## 两套时间轴

学习率与教师梯度按 learner 的有效训练秒数，在 rollout 边界切换：

| 有效时间 | Actor | Critic | 教师梯度 | Entropy |
|---|---:|---:|---:|---:|
| 0-30m | 冻结 | 6e-5 | shadow | 0.004 |
| 30m-2h | 3e-5 | 6e-5 | 0 -> 1% | 0.004 |
| 2-6h | 5e-5 | 5e-5 | 2% | 0.003 |
| 6-8h | 2.5e-5 | 3e-5 | 1% | 0.002 |

worker 无法在不修改平台 `BaseEnv` 的情况下实时读取 learner effective clock，因此 reason4 安全课程
使用单独保存的 session wall clock：0-30m shadow/12s，30m-2h active/12s，2-8h active/10s。
checkpoint 恢复 wall offset 后继续该阶段，不把 worker monotonic 与 learner effective clock混成
伪 exact。阶段切换时清空 tracker 窗口，避免旧候选跨阶段触发。

reason4 仅在空间受限、非足端墙接触、真实低运动、Goal 距离和 grace 均成立时触发；不依赖平台
原生命令或不可验证的高层运动意图。它属于 failure/truncated，bootstrap/continuation mask 均为零，
不得计入完成。接触映射或 termination term 不可验证时 fail closed。

## Checkpoint 与验收

新合同为 `p4_maze_closed_loop_v3`，标签优先级为
`loopstable > looptrain > loopadapt > loopwarm >` 历史 P4 标签。旧 P4 只能结构 warm start；只有
command/reward/training/stuck/camera 合同完整一致的新包才能 exact resume。Standard eval 只抽取低层，
Track eval 加载完整层级；training-only 教师和诊断头均不进入部署执行。

启动前必须通过本地合同/奖励/梯度/恢复测试，以及开发容器 1-env reset 和 8-env 一次 32-tick
PPO/save/resume。平台 smoke 只证明装配、reset、首轮 PPO 和保存可运行，不证明能力改善。

固定条件评估比较父包与 30m/2h/6h/8h checkpoint，分别报告完成率、timeout、collision、reason4、
卡滞时长、teacher 正确但 Actor 选错、正确 yaw 建立、`vy` 替代 yaw、正负 `vy/wz` 执行误差和
policy/exec/true 链。若 teacher 指标改善但完成率不升，结论应指向视觉表征或空间记忆缺口，不继续
叠加奖励。
