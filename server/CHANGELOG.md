# server CHANGELOG

本文件记录 `server/` 主线的活动变更。小调参不再创建永久分支，改为：Changelog + 独立 TOML + 实验编号/父 checkpoint/配置哈希/评估结果。

## [未发布]

- 仓库从扁平结构迁移至 `server/` 目录（2026-07-21）。源基线 = `codex/st7-opt3`（`9b0b3df`）。

## 基线

- **ST7-Opt3**（`9b0b3df`，2026-07-20）：UWB goal noise。Actor 接收几何一致的 UWB 式 goal 噪声（75% 环境的 bearing/distance jitter + per-episode bias），Critic/奖励/完成检查继续用干净真值。父 = ST7-Opt2B（`595693`）。
- 含 ST7-Opt2B `dynamic_tilt_risk`（连续倾斜风险惩罚）、行为蒸馏/LBC 机制（`codex/st7-opt2a` 并入）。

<!-- 待补（阶段 2）：
- [阶段2] 重新实现 J9 通用学习率接口修复（显式传 schedule/min/max_learning_rate，默认配置保持原行为，增加固定/自适应边界测试）
- [阶段2] 移植 Opt4 角速度提前约束（源提交 5f8764d；只调角速度阈值与权重组合，保留 Opt3 goal noise + Opt2B dynamic_tilt_risk）
-->
