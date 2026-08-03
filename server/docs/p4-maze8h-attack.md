# P4 Maze 感知诊断与进攻式八小时强化

## 任务定位

- 分支：`codex/p4-maze8h-attack`
- 任务名：`p4maze8h-10hzroute`
- 父模型：`p4nav2h_1256446-F`（平台模型 ID `1256446`）
- 父 checkpoint SHA256：
  `8aae0892664f2f263949f7e5f9f3b53ecd2936b44e80f3783091579bcf3d3461`
- 地形：单段 `open_entry_maze`，20 个静态难度列，课程关闭
- 时钟：先执行 600 秒只读诊断，再累计完整 `28800s` 梯度训练；平台任务配置
  `29700s`（8 小时 15 分钟），额外 300 秒用于 rollout、保存和退出，工作流达到目标后自行结束

网络和部署接口不变：低层 observation 57901、Actor77、动作 12；完整 policy observation
57905，高层 Actor 输入 85。低层全程冻结，NavigationEncoder、高层 Actor/Critic、SafetyHead
和 ResponseAdapter 按诊断结果训练。

Track eval 使用精简装配，不加载 training-only SafetyHead。公共相机教师路径在 Head 缺失时保持
三方向风险诊断为零，不能重新创建随机 Head，也不能影响 Actor、NavigationEncoder 或 Adapter。

## 目标和归因

本轮首先区分四类失败：视觉未识别墙/路口、SafetyHead 已识别但 Actor 选错、风险出现后
Actor 未减速、局部安全选择正确但仍因记忆或全局路线绕圈。前 600 秒以 detached
`nav_feat32` 和 `goal4` 训练只读线性探针，并在同一 clean depth 上构造 10% 轻故障 shadow；
shadow 不进入 Actor observation、PPO reward 或 rollout storage。

诊断通过时采用 `actor_attack`，否则采用 `visual_recovery`。两条分支共享 8 小时阶段边界：

```text
0-1h       mazeprobe
1-5h       mazeattack
5-7h       mazehard
7-8h       mazefinal
```

## 奖励和终止

- 动作映射保持 `vx=[0,1.0]`、`vy=+-0.30`、`wz=+-0.90`。
- 高层频率由 5Hz 提升为 10Hz（每 5 个低层帧决策一次），rollout 仍为 32 tick、TBPTT16；
  单个 rollout 的物理时长从 6.4 秒缩短到 3.2 秒。连续 tick 奖励按
  `duration_frames/10` 归一化，保持与原 5Hz 合同相同的每秒尺度；success/failure/reset 等事件
  impulse、按米计算的 route excess 和每次策略决策的 command-rate 不做该缩放。
- 成功 impulse 为 `+200`。增加三项负奖励：卡墙候选从 1 秒后由 `-0.02` 逐渐加深到
  `-0.10`，安全方向内的目标偏好最多 `-0.04`，未转化为目标进展的额外路程按
  `-0.05/m` 处罚且单 tick 最多计算 0.20m。route excess 在 push 后 0.30 秒 grace 内为零。
  目标偏好使用 terminal-safe 的仿真米制真值；带噪 GoalBelief 只进入 Actor 和速度保护，
  不得污染 PPO 奖励。
- 软巡航只产生负奖励：0-30 分钟关闭，30-60 分钟线性启用，此后在前方清晰且 Goal 新鲜时
  轻罚 `policy_target_vx<0.60` 或 `>0.75`；风险升高时允许主动减速或停止。
- missed-safe 权重在 0-30 分钟升至 0.015，2 小时升至 0.040，保持到 6 小时后收敛到 0.030。
- predictive collision 相对父合同放大 1.25 倍，单 tick 下限为 -0.03；安全组总下限 -0.06。
- `frontier_stagnation` 只保留 shadow，不写入 PPO reward。
- active wall-stuck 在 7 秒确认后使用 reason 4 和 `-15` terminal impulse，并回收未结算的
  frontier potential；它不计作成功或普通 timeout，且不 bootstrap。

监控分别报告 actual/shadow stagnation、missed-safe eligible event、Head-correct/Actor-wrong 条件率、
risk-to-deceleration 条件率，以及 wall-stuck terminal episode return。目标是确认 reset episode 的
整体回报为负，而不是仅观察单个 terminal impulse。

## 地形和指标语义

worker wire 中的 physical segment index 仍保留。指标层根据 TOML 的 `sub_terrains` 动态映射到
稳定的 `slope_inv/stairs_inv/maze` 三类；单段 `open_entry_maze` 的 physical index 0 必须进入
Maze bucket，不能再误记为坡面。未使用的两个 bucket 保持零，避免破坏旧面板 schema。

## Checkpoint

- 新合同：`p4_maze_attack10hz_v1`
- 标签优先级：`mazefinal > mazehard > mazeattack > mazefull > mazeprobe > mazediag > legacy P4`
- exact resume 同时核验 training、reward、command mapper、stuck-reset 和 camera 合同。
- P4 启动要求运行时 segment 语义严格为单段 `maze`，实际三轴 slew 必须与完整 command contract
  一致；exact resume 比较完整 command contract，不只比较 mapper 版本。
- 保存诊断轻故障 RNG、探针累计统计和已选择分支；live hidden、pending rollout 和未完成
  Adapter history 不保存，resume 后统一 reset。
- 模型 ID和标签只用于候选定位与 lineage，不作为结构正确时的单点硬门禁。

## 最小验证

从仓库根目录执行：

```bash
PYTHONPATH=server /Users/nanbloom001/miniconda3/envs/PY311test/bin/python -m pytest -q \
  server/agent_ppo/tests/test_p4_nav.py \
  server/agent_ppo/tests/test_p2_core.py \
  server/agent_ppo/tests/test_nav_stage_and_metrics.py
```

开发容器继续使用最小真实父包联合 smoke，不重复 128-env 压测；本轮没有改变网络、storage 或
相机分辨率。平台短 smoke 必须确认 10Hz period、32-tick rollout、reward conservation、diagnostic
fault、Maze segment 归因、reason 4 回报和四阶段保存标签，再创建 8 小时 15 分钟墙钟任务，确保
只读诊断后仍完整训练 8 小时。
