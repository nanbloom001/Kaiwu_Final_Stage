# server CHANGELOG

本文件记录 `server/` 主线的活动变更。小调参不再创建永久分支，改为：Changelog + 独立 TOML + 实验编号/父 checkpoint/配置哈希/评估结果。

## [未发布]

- **[standard-distill-1]** 新增基于 hjcnew 10288 Standard 教师的 77-D
  视觉蒸馏阶段，并设为当前入口。训练命令覆盖低速前后、横移和双向转动，
  开启 Standard 地形课程、摩擦随机化、观测噪声和深度增强；第一阶段关闭
  外部 push，后续可从最佳 checkpoint 单独开启。相机安装外参更新为
  `offset_pos=[0.339871,0.034697,0.075010]`、
  `offset_rot=[0.982631,-0.007085,0.184337,-0.020153]`，评估时从当前阶段
  TOML 注入相同外参。复用现有动作蒸馏、DAgger 和序列训练代码，网络与
  checkpoint 接口不变。
- **[checkpoint 兼容修复]** LBC 预加载按平台探活规则识别同一 ID 的带标签
  文件名与任意扩展名，例如 `model.ckpt-hjcnew-10288.pkl`。若候选是结构兼容
  的 Camera/LBC checkpoint，则恢复学生继续蒸馏；否则按平面教师拆分。
  `goal_dim` 和三组 state dict 必须与当前阶段完全一致，不兼容模型仍会拒绝。
- **[遗留路线收敛]** 将 `codex/st7-opt5-debug` 的 Opt5/Opt5-debug/Opt5B
  困难段出生模块、独立 TOML、监控、契约测试和实验文档迁入 `server/`；
  `Config.CURRENT` 保持 ST9-Opt3-D2，不启用 hard-start。补入 ST9-Opt3-D1
  的原始配置与实验文档；D1 的相机、goal-noise、深度增强和蒸馏基础已由
  D2 实现吸收。源提交继续由 annotated tags 保存，远程活动分支统一收敛到
  `main`。
- **[ST9-Opt3-D2]** 从 ST9-Opt3-D1 视觉学生继续训练：迁移 D1 的相机观测、UWB goal 小噪声与持续块状深度空洞链路；蒸馏目标改为 `0.5*latent SmoothL1 + 0.1*cosine + 1.0*raw-action SmoothL1`，教师网络保持冻结。DAgger 改为按环境独立采样的 50%/75%/100% 三阶段学生驱动，TBPTT 序列长度改为 16，高难度完整赛道占比提高，并减轻深度空洞增强。新增 D1 学生 checkpoint 强制续训检查、损失量级日志、监控面板和契约测试。网络、观测顺序、checkpoint 命名与部署接口不变。父 checkpoint = ST9-Opt3-D1 视觉学生；训练与评估结果待平台补录。
- 仓库从扁平结构迁移至 `server/` 目录（2026-07-21）。源基线 = `codex/st7-opt3`（`9b0b3df`）。
- **[J9 接口修复移植]** 源 `codex/stage3j9-fixed-learning-rate`（`68222ad`）。修复 PPO 构造器静默忽略 `schedule`/学习率上下界的 latent bug：`AlgorithmPPO` 现显式接收 `min_learning_rate`/`max_learning_rate`（默认 `1e-5`/`1e-2` 保自适应原行为），`_update_learning_rate` 改用配置界而非硬编码，新增 `_validate_fixed_lr()` 在 `learn()` 入口与出口双查（算法 LR + 所有 optimizer 组）。**策略决策**：仅移植通用接口修复，`TrackNavConfig` 保持 Opt3 自适应 `1.5e-5`（未提升 J9 的固定学习率实验进基线）；J9 固定学习率实验以 `train_env_conf_track_navj9.toml` 作记录，需要时设 `schedule="fixed"` 即生效。
- **[Opt4 移植]** 源 `codex/st7-opt4-angular-rate`（`5f8764d`）。仅调 `dynamic_tilt_risk` 角速度阈值（roll-rate `0.60/1.80->0.50/1.60`、pitch-rate `0.80/2.20->0.65/1.90`）与风险组合权重（`0.25->0.35` roll_rate²、`0.20->0.30` pitch_rate²）。保留 Opt3 goal noise + Opt2B 结构。
- 新增 `docs/st7-opt4.md`（从源分支移植实验笔记）。

## 基线

- **ST7-Opt3**（`9b0b3df`，2026-07-20）：UWB goal noise。Actor 接收几何一致的 UWB 式 goal 噪声（75% 环境的 bearing/distance jitter + per-episode bias），Critic/奖励/完成检查继续用干净真值。父 = ST7-Opt2B（`595693`）。
- 含 ST7-Opt2B `dynamic_tilt_risk`（连续倾斜风险惩罚）、行为蒸馏/LBC 机制（`codex/st7-opt2a` 并入）。

## 仓库迁移状态

- 阶段 3 已完成：`deploy/` 通过保留第二父的 subtree merge 导入，四套目录均有 `ARTIFACTS.md`。
- 阶段 4 已完成：378413、分析报告和旧部署包分别纳入 `archive/` 与 `shared/`。
- 阶段 5：通过 PR merge commit 合入 `main`，随后按分支登记执行带远程 SHA 复核的分支收敛。
- 训练长跑与 Jetson 真机验证属于后续模型发布验收，不作为本次仓库布局合并的阻断条件。
