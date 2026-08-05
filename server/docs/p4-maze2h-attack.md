# P4 Maze 感知诊断与软巡航两小时强化

## 任务定位

- 分支：`codex/p4-maze2h-attack`
- 任务名：`p4maze2h-attack`
- 父模型：`p4nav8h-r3` 的 `pnavstable-1207698`
- 父 checkpoint SHA256：
  `781022129ac17e564830c34570213a63f111b44d29b9d1bd247c3481c7b9ea55`
- 地形：单段 `open_entry_maze`，20 个静态难度列，课程关闭
- 正式梯度训练：诊断完成后累计完整 `7200s`

本阶段不改变部署接口。低层 observation 仍为 57901、Actor77、动作 12；完整 policy observation
仍为 57905，高层 Actor 输入仍为 85，三轴物理动作范围不变。低层网络全程冻结。

## 核心目标

1. 区分“视觉没有识别墙或路口”和“视觉已识别但 Actor 没有采取安全动作”。
2. 在不硬改动作映射的前提下，使开阔路段优先使用 `0.60-0.75m/s`，迷宫风险升高时仍可主动减速。
3. 降低撞墙、确认卡滞、S 型摆头和高层看到风险后不减速的比例。
4. 若局部安全选向正确但仍反复绕圈，将结果归因为空间记忆或全局路线能力，不继续叠加避障惩罚。

## 十分钟只读诊断

`maze_training_branch=auto` 时，前 600 秒只训练四个 training-only 线性探针，不执行 PPO 或 Adapter
更新。探针输入为 detached `nav_feat32` 或 `goal4`，标签来自训练期特权 `nav_scanner + height_scan`。
累计 teacher coverage、墙面 AUROC/漏检率、安全方向 top-1、场景 macro-F1、clean/live latent
cosine，并与 `goal4` 和随机基线比较。

诊断满足感知阈值时选择 `actor_attack`；样本不足或视觉指标不满足时保守选择
`visual_recovery`。诊断 wall clock 与正式训练 clock 分离，因此 600 秒诊断不会占用 7200 秒正式训练。

平台 `finish_tick()` 在 `torch.no_grad()` 中运行。线性探针的 forward/loss/backward/step 使用局部
`torch.enable_grad()`，输入保持 detached；NavigationEncoder、Actor 和 rollout 收集不会因此建图。

## 奖励与终止

- 保留 P4 的进度、成功、失败、超时、predictive collision、missed-safe、body collision 和
  yaw-cancellation 合同。
- 新增软巡航负奖励。前方 clear、Goal 新鲜且中心方向接近最安全方向时，仅处罚
  `policy_target_vx < 0.60` 或 `> 0.75`；区间内为零，不产生速度正奖励。
- 前方堵塞、侧向明显更安全或 Goal 失效时，低速惩罚自动衰减，允许停止、横移或转向。
- episode 为 75 秒；确认墙面卡滞使用 `active + 7s`，reason 4 的独立惩罚为 `-10`，不 bootstrap，
  不计作完成或普通 timeout，并回收尚未结算的 frontier potential。

## Checkpoint 合同

- 新合同：`p4_maze_soft_cruise_v2_training_clock`
- 标签优先级：`mazefinal > mazefull > mazeprobe > mazediag > legacy P4`
- 诊断阶段按 wall clock 在约 5 分钟保存 `mazediag`；正式阶段仍按 training clock 保存。
- 旧 P4 合同只能 warm start，新 session 重置 rollout、hidden、诊断状态和训练时钟。
- warm start 不强制恢复其他设备后端的 RNG 格式，而使用当前运行时确定性新种子；同合同 exact
  resume 仍严格恢复全部 RNG、阶段、探针和优化器状态。

模型 ID和标签只负责候选定位和 lineage 记录，不是单点硬门禁。stage、模块 leaf、spec、shape、
有限值与合同兼容性仍是加载依据。

## 验证证据

截至 2026-08-03：

- 本地 P4 定向测试：`47 passed`。
- 开发容器 P4 定向测试：`47 passed`。
- 真实父包 8-env CUDA 最小联合 smoke：32 个高层 tick、4 PPO epoch、16 次 PPO update、一次
  Adapter update、checkpoint 保存及 exact resume 全部通过。
- 冻结低层 digest 未漂移。
- 联合 smoke CUDA 峰值约为 `231,538,176` allocated、`287,309,824` reserved；pinned depth
  约 `29,491,200` bytes。
- 按当前测试政策，不重复运行 128-env 规模测试。只有 storage shape、相机分辨率或 Isaac 环境装配
  变化时才单独执行规模或 Isaac smoke。

平台首次任务曾在诊断探针 backward 处触发外层 `no_grad` 错误；代码和容器测试已修复，但仍需新的
平台 smoke 才能标记为平台已验证。当前证据不能替代完整两小时训练或固定种子评估。

## 最小测试流程

从 `server/` 目录执行：

```bash
python -m pytest -q agent_ppo/tests/test_p4_nav.py
```

开发容器真实父包联合 smoke 使用：

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  agent_ppo/tools/p4_maze_continue_smoke.py \
  --checkpoint <pnavstable-checkpoint> \
  --output <temporary-smoke-checkpoint> \
  --config agent_ppo/conf/train_env_conf_track_p4_nav_ppo.toml
```

脚本默认 8 env。生成的父包副本、解包目录和 smoke checkpoint 属于测试制品，测试结束后必须精确清理。

## 回滚

若平台复测仍在诊断阶段失败，停止任务并回到 `pnavstable-1207698`，不要使用失败任务产生的
checkpoint。若仅需关闭 Maze 强化，可恢复上一版 P4 TOML 和 checkpoint 标签；不要回滚 P4 已验证的
相机单位、GoalBelief、terminal snapshot、Adapter replay 与低层冻结修复。
