# P4 Maze 三小时卡滞恢复与转向稳定

## 任务合同

- 分支：`codex/p4-maze3h-recovery`
- 任务：`p4maze3h-recovery-r2`
- 父包：`p4maze8h10hz_1416926`
- 父 checkpoint SHA256：
  `0bf54e3c18e6450492d7c1d44d69596d79beddbebb57166432220f8cb93dd2e4`
- 训练：128 env，单段 `open_entry_maze`，20 个静态列，课程关闭，`10800s`
- 平台墙钟：`3.15h`，达到有效训练目标后由工作流保存并结束

网络、输入和部署接口保持不变：低层 57901/Actor77/action12，高层 Actor85；低层全程冻结，
NavigationEncoder、高层 Actor/Critic、SafetyHead、training-only StuckHead 和 ResponseAdapter
按三小时课程训练。高层 10Hz、低层 50Hz，rollout 32 tick，TBPTT16。

## 行为修复

- 三轴 slew 为 `vx 0.30/0.30`、`vy 0.40/0.80`、`wz 1.50/3.00`，反向先回零。
- predictive depth risk 同比例缩放 policy `(vx,vy)`，不压缩 `wz`；风险解除的 alpha 每 tick
  最多恢复 0.15。
- Goal 新鲜且距离 `0.65-1.20m`、policy 正朝向目标时启用平移软捕获；不修改方向、不提前终止、
  不在 0.6m 成功半径外归零。
- rollout-time safe3 教师只处罚超出方向/速度/yaw 容差的 Actor mean，`nav_feat32` 在该路径 detach。
  Camera clean/live 辅助继续更新 NavigationEncoder。
- 从 rollout 内任意真实 reset 起点构造初始 recurrent hidden 为零、序列内不再跨 reset 的完整
  TBPTT16；按全部 PPO sequence-update 的 10% 调度左右镜像。合格 episode 数、实际 microbatch
  share 和 rollout 调度 share 分别上报；training-only StuckHead 使用 Actor recurrent feature，
  不进入评估。
- SafetyHead 对墙面/卡滞 hard-positive 使用 2x 权重，并报告 Actor StuckHead PR-AUC、P/R/F1。

## 奖励与终止

predictive collision、missed-safe、yaw cancellation、yaw-exit 和 goal-safe preference 共用原 5Hz
等效 `-0.06` 预算；正常 10Hz tick 再按 duration 缩放。success 保持 `+200`；reason 4 卡墙
terminal 保持 `-15`，同 tick 不重复 collision/safety/stagnation。route-excess 在 recovery、dead-end、
Goal stale 和 contact latch 时暂停，近终点 capture 时屏蔽 soft-cruise 低速项和 crawl。

worker 物理 reset 不修改平台 `BaseEnv`。当前平台没有经过验证的 aisrv 到 worker 高层运动意图
transport，且原生 `base_velocity` 不是 P4 实际 exec command，不能代用。因此活动配置为 `shadow`：
worker 只报告世界位移、低真值速度、墙面证据、候选时长以及运动意图 hook 可用性；没有显式
`_agent_ppo_p4_motion_intent` 时 active reset 必须 fail-closed。训练 wire 为
`509 = P3 493 + raw goal2 + stuck diagnostics13 + raw wall term1`。训练侧 Actor/StuckHead 标签仍使用 aisrv 的
policy/exec 意图；平台 smoke 应确认 `p4_spawn_hook_installed=1`，并按 active/shadow 合同核对
reset count。出生 hook 或 active termination hook 无法装配时属于正确性失败，不能静默继续训练。

恢复事件口径固定为：候选首次出现为 entry；候选在 episode 未结束时消失且 true XY 速度恢复或
目标距离继续缩短，才算 recovery success；墙面证据暂失但没有运动证据记为 unverified exit，并
保持恢复状态。候选状态下 terminal 单独计数，平台完成单独作为成功终止，reset 不计恢复或完成。
面板提供当前窗口、60 秒成功恢复数和 lifetime，exact resume 恢复这些 lifetime 统计。

## Checkpoint

新合同为 `p4_maze_recovery_v1`。旧 `1416926` 包只能 continuation warm start：保留旧 8 个
Actor/CNN/SafetyHead optimizer 组的兼容 Adam moments，新 `actor_stuck_head` 使用空 state；session
时钟、rollout、hidden 和 live limiter 重置。同合同 exact resume 恢复 StuckHead leaf、mirror RNG、
auxiliary calibration、optimizer/return statistics 和 Adapter completed records。

本地验证、容器验证和平台验证的实际证据以 Bug 台账 `BUG-20260804-003` 为准。
