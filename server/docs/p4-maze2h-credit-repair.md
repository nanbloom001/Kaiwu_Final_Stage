# P4 Maze 两小时信用分配修复说明

## 目标与边界

本轮从 `p4maze8h10hz_1416926` 的 `mazefinal` checkpoint warm start，任务名为
`p4maze2h-credit-r2`，平台任务 ID 为 `236585`。代码内累计 `7200s` 有效训练，平台墙钟
`2h15min` 只为 rollout、checkpoint 和正常退出预留余量。

本轮只回答一个问题：在保留父模型 Maze 感知和低层能力的前提下，修复高层策略的信用分配和
墙前控制后，是否能恢复被五段混训削弱的 Maze 完成能力。它不是新的五段训练，也不宣称已经改善
固定种子评估、真实机器人或 Sim2Real 表现。

保持不变的公开合同：

- 高层 Actor 输入 85 维、三轴动作和 10Hz 决策频率不变；
- 低层 50Hz、低层 observation/action 和部署接口不变；
- Track eval wire、Standard 低层评估入口和 checkpoint leaf 结构不变；
- 不修改平台覆盖的 `server/isaac_env/base_env.py`；
- 模型 ID 只用于父包候选选择，模块 spec、shape 和有限值决定兼容性。

## 环境配置

| 项目 | 配置 |
|---|---|
| 环境数 | 128 |
| 地形 | 单段 `open_entry_maze` |
| 难度列 | 20 个静态列 |
| Curriculum | 关闭 |
| Episode timeout | 75 秒 |
| 高层 rollout | 32 tick |
| TBPTT | 16 |
| PPO | 4 epochs，4 minibatches |

本轮关闭 Push、额外 Camera/Goal fault、五段随机出生、segment frontier、route-excess 和
open-straight。卡墙 reset 全程保持 `shadow`，只记录候选，不产生 reason 4 terminal。

## 模块训练职责

- 冻结低层、NavigationEncoder、SafetyHead 和 ResponseAdapter；训练期间持续核对低层 digest。
- 保留 Actor/LSTM 权重，重置 Actor Adam moments，避免继承五段目标下的优化惯性。
- 重建 Critic、Critic optimizer 和 return/value statistics。
- 新建 training-only StuckHead，eval/export 不装配该模块。

| 有效训练时间 | Actor/LSTM | Critic | StuckHead | Teacher |
|---|---:|---:|---:|---:|
| 0-10 分钟 | 冻结 | `1.2e-4` | 校准 | 关闭 |
| 10-30 分钟 | `7.5e-5` | `1.2e-4` | 训练 | 0 -> 2.5% |
| 30-105 分钟 | `1.0e-4` | `1.0e-4` | 训练 | 3.5%，硬上限 5% |
| 105-120 分钟 | `5.0e-5` | `6.0e-5` | 训练 | 2.0% |

## 奖励与控制修复

1. 删除 terminal frontier clawback。过去正确逼近终点后失败，会把已经获得的进展统一回扣，难以
   区分“正确绕行后失败”和“完全没有推进”。
2. 改为 episode 内不可重复的历史最短距离 credit：`2.0/m`，累计上限 `+12`，terminal 不回扣。
3. 保持 success `+200`、failure/timeout `-25`、时间成本 `-0.02/tick`。即使拿满进展 credit 后
   timeout，episode 回报仍为负，不能通过接近终点但不完成刷分。
4. 卡滞候选从 0.8 秒开始轻罚 `-0.005/tick`，2 秒达到 `-0.02/tick`；不提供可刷取的脱困正奖励。
5. 平移 limiter 改为紧急限制：risk `<=0.75` 不缩放，risk=1 时仍保留 60% `(vx,vy)`，`wz`
   不缩放；风险解除后每个 10Hz tick 最多恢复 0.20。
6. 保留父模型 predictive collision、missed-safe、yaw response 和安全方向内 Goal 偏好，不在本轮
   同时扩大安全奖励权重。

## 首轮 PPO 阻断修复

失败任务 `p4maze2h-creditfix`（ID `236575`）在 `creditwarm` 首轮 PPO update 抛出：

```text
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

该阶段 Actor/CNN 冻结，部分 minibatch 又没有有效 stuck 标签，组合 Actor loss 因此是有限常量。
公共 PPO 循环此前仍无条件调用 `backward()`。

现在只有 loss 具有 gradient graph 时才 backward；一个 minibatch 至少产生过一次真实 Actor 梯度
才执行 optimizer step 和增加 Actor gradient-step。无梯度批次不记为 non-finite，Critic 继续独立
更新。

## 验证证据

- 本地 P4 定向回归：`131 passed`；Python 编译、TOML 和 `git diff --check` 通过。
- 开发容器：recovery auxiliary `16 passed`，完整 P4 回归 `131 passed`。
- 平台替代任务 `236585` 已越过原崩溃点并运行到 `iter=4`、`effective_min=1.5`、
  `phase=creditwarm`；`lifecycle_fail=0`、`low_digest_drift=0`、`low_optimizer_steps=0`。

这些证据只证明装配、首轮 PPO 和冻结边界正常。最终策略效果仍需使用相同固定种子比较父模型与
30/60/120 分钟 checkpoint，重点报告 Maze completion、timeout、collision、卡滞持续时间、
teacher 正确但 Actor 选错率和 yaw 响应。

## 回滚

- 策略回滚：重新使用未修改的 `p4maze8h10hz_1416926` 父包。
- 训练合同回滚：旧 P4 包始终按 warm start 加载，只有 `p4_maze_credit_repair_v1` 可 exact resume。
- 启动修复回滚：恢复公共 PPO loop 的无条件 backward；该操作会重新引入空辅助批崩溃，不建议执行。
