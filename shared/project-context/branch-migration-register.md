# 分支迁移登记

> 本文件记录 Kaiwu_Final_Stage 仓库迁移的**源 SHA 锚点**与**分支分类**，作为无需重写历史的回滚锚点。
>
> **裁剪策略（用户确认，偏离原 Plan 的"全保留"）**：保留重大改动节点作记录/回退/分叉点（tag 或活动分支），将散落的历史记录分支合并/删除。原 Plan 的"不删除任何分支"已被用户改为有选择裁剪。

## 迁移元信息

| 项 | 值 |
|---|---|
| 迁移日期 | 2026-07-21 |
| 迁移分支 | `codex/repository-layout-migration` |
| 起点（server 基线） | `origin/codex/st7-opt3` = `9b0b3df4172f7e0e1ca17b367e44ada23b9a891b` |
| 原 `main` HEAD | `042a1f883801d1d859106fcab07c8ffae2ced9d5` |
| 迁移 worktree | `/Users/nanbloom001/codespace/fwwb-migration` |
| 原工作区 | `/Users/nanbloom001/codespace/fwwb-Final`（保持在 `main`，未跟踪文件未动） |
| 迁移方式 | `git mv` 整体移动 + J9/Opt4 移植 + 新增文档；不重写历史、不 force-push |
| 阶段 2 状态 | ✅ J9 接口修复 + Opt4 已移植并合并（merge commit `97ac193`） |

## 源 SHA 锚点（回滚参考）

### 远程分支 tip（18 个，含 st7-opt5-debug）

| 分支 | tip SHA | 说明 |
|---|---|---|
| `origin/main` | `042a1f8` | 初始决赛代码 |
| `origin/cyy` | `d372dce` | Stage3J-4 near-goal（= tag `stage3j4-neargoal-rejected`） |
| `origin/timeout-bootstrap-diag` | `13a1b19` | timeout 观测诊断 |
| `origin/codex/stage3j5-p14` | `b589ed1` | J-5 P14 高压完成奖励 |
| `origin/codex/stage3j6-termination` | `b2891e4` | J-6 termination 权重 |
| `origin/codex/stage3j7-rough-energy` | `1acc731` | J-7 rough energy |
| `origin/codex/stage3j8-hard-level-replay` | `81ca14e` | J-8 困难等级重放 |
| `origin/codex/stage3j9-fixed-learning-rate` | `68222ad` | J-9 固定学习率（接口已移植） |
| `origin/codex/st7-opt1a-speed-wall-stall` | `84d7d8b` | ST7 Opt1A 速度/墙停 |
| `origin/codex/st7-opt2a-tilt-limit` | `09d88e7` | ST7 Opt2A 恢复基线+蒸馏并入 |
| `origin/codex/st7-opt2b-dynamic-stability` | `f8f7ec0` | ST7 Opt2B 动态倾斜风险 |
| `origin/codex/st7-opt3` | `9b0b3df` | **ST7 Opt3 UWB goal noise（统一 server 基线）** |
| `origin/codex/st7-opt4-angular-rate` | `5f8764d` | ST7 Opt4 角速度保护（已移植） |
| `origin/codex/st7-opt5-hard-start-replay` | `b81cbf4` | ST7 Opt5 hard-segment replay（活动路线） |
| `origin/codex/st7-opt5-debug` | `9570780` | ST7 Opt5-debug 硬启动诊断（活动子路线，+1048/-119） |
| `origin/codex/st9-opt3-d1` | `5ea7ba9` | ST9 Opt3 D1 视觉蒸馏（活动路线） |
| `origin/deploy/jetson-sim2real` | `b297b3f` | Jetson Sim2Real 四套部署树（独立历史根） |

### 标签（7 个，已推送 origin）

| 标签 | SHA | 说明 |
|---|---|---|
| `stage3i2-nogate-safe-60m` | `d777c1f` | Stage3I-2 NoGateSafe 60min 最佳（冻结） |
| `stage3j4-neargoal-rejected` | `d372dce` | Stage3J-4 near-goal（已否决） |
| `st7-opt2a-baseline-rewrite` | `09d88e7` | Opt2A 基线重写+蒸馏并入（重大节点） |
| `st7-opt3-server-baseline` | `9b0b3df` | Opt3 统一 server 基线（重大节点） |
| `st7-opt5-hard-start-replay` | `b81cbf4` | Opt5 hard-start replay（重大节点/活动路线） |
| `st7-opt5-debug-hard-start-diagnostics` | `9570780` | Opt5-debug 硬启动诊断（重大节点/活动子路线） |
| `st9-opt3-d1-vision-distill` | `5ea7ba9` | ST9-D1 视觉蒸馏（重大节点/活动路线） |

## 分支迁移分类

| 分支 | 改动规模 | 实验状态 | 分类 | 处理 |
|---|---|---|---|---|
| `origin/main` | - | 初始 | 基线 | 通过迁移 PR 合入（merge commit，非 squash） |
| `origin/cyy` | 中 | J-4 已否决 | 历史基线 | 已 tag（stage3j4）；tip 在主干，可删分支 |
| `origin/timeout-bootstrap-diag` | 小 | 禁止晋升 | 归档 | 删前加 `archived/` tag |
| `origin/codex/stage3j5-p14` | 小 | 被 ST7 取代 | 归档 | 删前加 `archived/` tag |
| `origin/codex/stage3j6-termination` | 小 | 被 ST7 取代 | 归档 | 删前加 `archived/` tag |
| `origin/codex/stage3j7-rough-energy` | 小 | 被 ST7 取代 | 归档 | 删前加 `archived/` tag |
| `origin/codex/stage3j8-hard-level-replay` | 中 | 被 ST7 取代 | 归档 | tip 在主干，可删分支 |
| `origin/codex/stage3j9-fixed-learning-rate` | 中 | 通用修复有效 | ✅ 已移植 | 接口修复已移植到 server/（仅接口，TrackNav 保自适应）；tip 在主干，可删分支 |
| `origin/codex/st7-opt1a-speed-wall-stall` | 小 | 被 Opt2A 覆盖 | 归档 | tip 在主干，可删分支 |
| `origin/codex/st7-opt2a-tilt-limit` | 🔴 大 | Opt3 历史基础 | 历史节点 | 已 tag；tip 在主干，可删分支 |
| `origin/codex/st7-opt2b-dynamic-stability` | 小 | 在 Opt3 血缘 | 历史节点 | tip 在主干，可删分支 |
| `origin/codex/st7-opt3` | 中 | **统一基线** | **主线** | 已 tag；= server/ 基线；tip 在主干，可删分支 |
| `origin/codex/st7-opt4-angular-rate` | 小 | 已移植 | ✅ 已移植 | 角速度阈值已移植到 server/；删前加 `archived/` tag |
| `origin/codex/st7-opt5-hard-start-replay` | 中 | 独立路线 | 活动路线 | 已 tag；**保留分支**（或迁 `codex/server-exp-st7-opt5`） |
| `origin/codex/st7-opt5-debug` | 🔴 大 | 独立子路线 | 活动路线 | 已 tag；**保留分支** |
| `origin/codex/st9-opt3-d1` | 🔴 大 | 独立路线 | 活动路线 | 已 tag；**保留分支**（或迁 `codex/server-exp-st9-opt3-d1`） |
| `origin/deploy/jetson-sim2real` | - | 部署树 | subtree 导入 | `git subtree add --prefix=deploy`（保留第二父）；导入后删分支 |

## 安全删除判定（基于 ancestry，已 git merge-base 核实）

- **tip 在主干上（迁移分支保护，删分支零损失）**：cyy、stage3j8、stage3j9、st7-opt1a、st7-opt2a、st7-opt2b、st7-opt3 -- 这些提交是 `codex/st7-opt5` 或迁移分支的祖先，删分支不丢提交。
- **独立侧支（删前必须先 tag，否则 gc 丢失）**：stage3j5/j6/j7、timeout-diag、st7-opt4（已移植）、deploy -- 已加 tag 或待加 `archived/` tag 后删。
- **必须保留分支**：st7-opt5、st7-opt5-debug、st9-opt3-d1（活动实验路线，其 tip 提交不在迁移分支保护范围内）。

## 回滚说明

- 原工作区始终在 `main`（`042a1f8`），未跟踪文件未动。
- 迁移分支未推送前：`git worktree remove` 即放弃。
- 合并后：只用 `git revert`；普通提交逆序 revert，deploy subtree 用 `git revert -m 1`。不用 reset/rebase/force-push。
- 旧分支与 tag 持续保留，可随时恢复原树。

## 待办（后续阶段，未在本轮执行）

- **阶段 3**：`git subtree add --prefix=deploy` 导入 `origin/deploy/jetson-sim2real`（非 squash，验证第二父 = `b297b3f`）；为四套目录生成 `ARTIFACTS.md`；ST7/standard 缺失 checkpoint/ONNX 需人工提供。
- **阶段 4**：纳入未跟踪资料（`track蒸馏3_378413` -> `archive/`、新增分析报告 -> `shared/分析记录/`、`unitree_isaaclab_deploy` -> `archive/`）；提交前凭证/license/大文件/LFS pointer 扫描（gate）。
- **阶段 5**：推送迁移分支 -> reviewed PR（**merge commit，非 squash/rebase**）-> 启用 main 保护 -> 验证（一个训练周期 + 一次 Jetson 部署）-> 最后才删散落分支（按上表"安全删除判定"）。
