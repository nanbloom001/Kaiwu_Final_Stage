# server CHANGELOG

本文件记录 `server/` 主线的活动变更。小调参不再创建永久分支，改为：Changelog + 独立 TOML + 实验编号/父 checkpoint/配置哈希/评估结果。

## [未发布]

- 仓库从扁平结构迁移至 `server/` 目录（2026-07-21）。源基线 = `codex/st7-opt3`（`9b0b3df`）。
- **[J9 接口修复移植]** 源 `codex/stage3j9-fixed-learning-rate`（`68222ad`）。修复 PPO 构造器静默忽略 `schedule`/学习率上下界的 latent bug：`AlgorithmPPO` 现显式接收 `min_learning_rate`/`max_learning_rate`（默认 `1e-5`/`1e-2` 保自适应原行为），`_update_learning_rate` 改用配置界而非硬编码，新增 `_validate_fixed_lr()` 在 `learn()` 入口与出口双查（算法 LR + 所有 optimizer 组）。**策略决策**：仅移植通用接口修复，`TrackNavConfig` 保持 Opt3 自适应 `1.5e-5`（未提升 J9 的固定学习率实验进基线）；J9 固定学习率实验以 `train_env_conf_track_navj9.toml` 作记录，需要时设 `schedule="fixed"` 即生效。
- **[Opt4 移植]** 源 `codex/st7-opt4-angular-rate`（`5f8764d`）。仅调 `dynamic_tilt_risk` 角速度阈值（roll-rate `0.60/1.80->0.50/1.60`、pitch-rate `0.80/2.20->0.65/1.90`）与风险组合权重（`0.25->0.35` roll_rate²、`0.20->0.30` pitch_rate²）。保留 Opt3 goal noise + Opt2B 结构。
- 新增 `docs/st7-opt4.md`（从源分支移植实验笔记）。

## 基线

- **ST7-Opt3**（`9b0b3df`，2026-07-20）：UWB goal noise。Actor 接收几何一致的 UWB 式 goal 噪声（75% 环境的 bearing/distance jitter + per-episode bias），Critic/奖励/完成检查继续用干净真值。父 = ST7-Opt2B（`595693`）。
- 含 ST7-Opt2B `dynamic_tilt_risk`（连续倾斜风险惩罚）、行为蒸馏/LBC 机制（`codex/st7-opt2a` 并入）。

## 待补（后续阶段，未执行）

- 阶段 3：`git subtree add --prefix=deploy` 导入部署树 + 四套 `ARTIFACTS.md`。
- 阶段 4：纳入未跟踪资料（378413/分析报告/旧部署包）+ 凭证/license 扫描。
- 阶段 5：PR 合入 main（merge commit，非 squash/rebase）+ main 保护 + 验证（训练周期 + Jetson 部署）。
