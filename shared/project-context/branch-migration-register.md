# 分支迁移登记

> 本文件记录 Kaiwu_Final_Stage 仓库迁移的**源 SHA 锚点**与**分支分类**，作为无需重写历史的回滚锚点。
> 迁移不会删除、重命名或重写任何现有远程分支。"归档"仅表示在实验登记中标记状态，不代表删除 Git 引用。

## 迁移元信息

| 项 | 值 |
|---|---|
| 迁移日期 | 2026-07-21 |
| 迁移分支 | `codex/repository-layout-migration` |
| 起点（server 基线） | `origin/codex/st7-opt3` = `9b0b3df4172f7e0e1ca17b367e44ada23b9a891b` |
| 原 `main` HEAD | `042a1f883801d1d859106fcab07c8ffae2ced9d5` |
| 迁移 worktree | `/Users/nanbloom001/codespace/fwwb-migration` |
| 原工作区 | `/Users/nanbloom001/codespace/fwwb-Final`（保持在 `main`，未跟踪文件未动） |
| 迁移方式 | `git mv` 整体移动 + 新增文档；不重写历史、不 force-push、不删旧分支 |

## 源 SHA 锚点（回滚参考）

### 远程分支 tip

| 分支 | tip SHA | 说明 |
|---|---|---|
| `origin/main` | `042a1f8` | 初始决赛代码 |
| `origin/cyy` | `d372dce` | Stage3J-4 near-goal（= tag `stage3j4-neargoal-rejected`） |
| `origin/timeout-bootstrap-diag` | `13a1b19` | timeout 观测诊断 |
| `origin/codex/stage3j5-p14` | `b589ed1` | J-5 P14 高压完成奖励 |
| `origin/codex/stage3j6-termination` | `b2891e4` | J-6 termination 权重 |
| `origin/codex/stage3j7-rough-energy` | `1acc731` | J-7 rough energy |
| `origin/codex/stage3j8-hard-level-replay` | `81ca14e` | J-8 困难等级重放 |
| `origin/codex/stage3j9-fixed-learning-rate` | `68222ad` | J-9 固定学习率 |
| `origin/codex/st7-opt1a-speed-wall-stall` | `84d7d8b` | ST7 Opt1A 速度/墙停 |
| `origin/codex/st7-opt2a-tilt-limit` | `09d88e7` | ST7 Opt2A 恢复基线+倾斜阈值+蒸馏并入 |
| `origin/codex/st7-opt2b-dynamic-stability` | `f8f7ec0` | ST7 Opt2B 动态倾斜风险 |
| `origin/codex/st7-opt3` | `9b0b3df` | **ST7 Opt3 UWB goal noise（统一 server 基线）** |
| `origin/codex/st7-opt4-angular-rate` | `5f8764d` | ST7 Opt4 角速度保护 |
| `origin/codex/st7-opt5-hard-start-replay` | `b81cbf4` | ST7 Opt5 hard-segment replay |
| `origin/codex/st9-opt3-d1` | `5ea7ba9` | ST9 Opt3 D1 视觉蒸馏 |
| `origin/deploy/jetson-sim2real` | `b297b3f` | Jetson Sim2Real 四套部署树（独立历史根） |

### 标签

| 标签 | SHA | 说明 |
|---|---|---|
| `stage3i2-nogate-safe-60m` | `d777c1f` | Stage3I-2 NoGateSafe 60min 最佳（冻结） |
| `stage3j4-neargoal-rejected` | `d372dce` | Stage3J-4 near-goal（已否决） |

## 分支迁移分类

| 分支 | 改动规模 | 实验状态 | 分类 | 新分支/文档去向 |
|---|---|---|---|---|
| `origin/main` | - | 初始 | 基线 | 不直接改，只通过迁移 PR 合入 |
| `origin/cyy` | 中（Stage3H/I-2/J-4） | J-4 已否决 | 历史基线 | 保留分支+标签，登记 |
| `origin/timeout-bootstrap-diag` | 小 | 禁止晋升模型 | 归档 | 提取结论到实验登记 |
| `origin/codex/stage3j5-p14` | 小（单 reward） | 已被 ST7 路线取代 | 归档 | 配置+结论登记 |
| `origin/codex/stage3j6-termination` | 小（1 数值） | 已被 ST7 路线取代 | 归档 | 配置+结论登记 |
| `origin/codex/stage3j7-rough-energy` | 小（单 reward） | 已被 ST7 路线取代 | 归档 | 配置+结论登记 |
| `origin/codex/stage3j8-hard-level-replay` | 中（workflow +330） | 被 ST7 基线恢复取代 | 归档 | 保留旧分支，登记 |
| `origin/codex/stage3j9-fixed-learning-rate` | 中（接口修复） | 通用修复有效 | 主线移植 | J9 LR 接口修复重新移植到 `server/`（阶段 2）；TOML 仅作记录 |
| `origin/codex/st7-opt1a-speed-wall-stall` | 小（4 行） | 被 Opt2A 恢复覆盖 | 归档 | 参数实验登记 |
| `origin/codex/st7-opt2a-tilt-limit` | 🔴 大（基线重写+蒸馏并入） | Opt3 的历史基础 | 历史基础 | 保留，不单独重复合并 |
| `origin/codex/st7-opt2b-dynamic-stability` | 小（单 reward） | 已含在 Opt3 血缘 | 历史基础 | 保留 |
| `origin/codex/st7-opt3` | 中（goal_noise 模块） | **统一 server 基线** | **主线** | = `server/` 基线 |
| `origin/codex/st7-opt4-angular-rate` | 小（阈值+系数） | 移植到活动主线 | 主线移植 | 移植到 `server/`，写入 CHANGELOG（阶段 2，源 `5f8764d`） |
| `origin/codex/st7-opt5-hard-start-replay` | 中（模块+测试） | 独立实验路线 | 独立路线 | 从迁移后 Opt3 节点分叉 `codex/server-exp-st7-opt5`（阶段 2） |
| `origin/codex/st9-opt3-d1` | 🔴 大（视觉蒸馏 stage） | 独立视觉蒸馏路线 | 独立路线 | 从迁移后 Opt3 节点分叉 `codex/server-exp-st9-opt3-d1`（阶段 2） |
| `origin/deploy/jetson-sim2real` | -（独立根） | 部署树 | subtree 导入 | `git subtree add --prefix=deploy`（阶段 3，保留第二父） |

## 回滚说明

- 原工作区始终在 `main`（`042a1f8`），未跟踪文件未动；迁移在独立 worktree 完成。
- 迁移分支未推送前：直接 `git worktree remove` 即放弃，不影响 `main` 或旧实验分支。
- 合并后：只用 `git revert`；普通迁移提交按逆序 revert，deploy subtree 合并用 `git revert -m 1`。不使用 reset/rebase/历史改写/force-push。
- 旧分支与源 SHA 持续保留，可随时从本登记恢复原树。

## 待办（后续阶段，未在本轮执行）

- **阶段 2**：在 Opt3 基线上重新实现 J9 LR 接口修复 + 移植 Opt4；从迁移后 Opt3 节点分叉 `codex/server-exp-st7-opt5` 与 `codex/server-exp-st9-opt3-d1`。
- **阶段 3**：`git subtree add --prefix=deploy` 导入 `origin/deploy/jetson-sim2real`；为四套目录生成 `ARTIFACTS.md`。
- **阶段 4**：纳入本地未跟踪资料（`track蒸馏3_378413` -> `archive/`、新增分析报告 -> `shared/分析记录/`、`unitree_isaaclab_deploy` -> `archive/`）；提交前凭证/license 扫描。
- **阶段 5**：完善文档 + PR；PR 合并前冻结新服务器实验提交。
