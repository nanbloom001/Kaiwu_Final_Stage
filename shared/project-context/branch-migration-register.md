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
| 迁移内容状态 | ✅ 四层重构、J9 接口修复、Opt4 移植、deploy subtree、资料归档均已完成 |

## 源 SHA 锚点（回滚参考）

### 迁移收尾前的源分支 tip（17 个，不含迁移分支）

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
| `origin/codex/st7-opt5-debug` | `c603ed7` | 活动路线；含 Opt5、`9570780` diagnostics 与 Opt5B conservative replay |
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
| `origin/codex/st7-opt5-hard-start-replay` | 中 | Opt5 稳定节点 | 历史节点 | 已 tag，且为保留的 Opt5-debug/Opt5B 分支祖先；删分支 |
| `origin/codex/st7-opt5-debug` | 🔴 大 | Opt5 diagnostics + Opt5B | **活动路线** | **保留分支**；为 `c603ed7` 增加 Opt5B tag |
| `origin/codex/st9-opt3-d1` | 🔴 大 | 独立视觉蒸馏路线 | **活动路线** | 已 tag；**保留分支** |
| `origin/deploy/jetson-sim2real` | - | 部署树 | subtree 导入 | `git subtree add --prefix=deploy`（保留第二父）；导入后删分支 |

## 安全删除判定（基于 ancestry，已 git merge-base 核实）

- **tip 在合并后 `main` 上**：迁移分支、cyy、stage3j8、stage3j9、st7-opt1a、st7-opt2a、st7-opt2b、st7-opt3、deploy；验证 ancestry 后删分支。
- **独立侧支**：stage3j5/j6/j7、timeout-diag、st7-opt4；先建立明确的 `archived/` annotated tag，再删分支。J5/J6/J7/timeout 是被淘汰或纯诊断实验，不合入活动代码。
- **Opt5 稳定节点**：st7-opt5-hard-start-replay 已有 tag，且是保留的 st7-opt5-debug 分支祖先；删分支但保留节点。
- **最终保留远程分支**：`main`、`codex/st7-opt5-debug`、`codex/st9-opt3-d1`。

## 回滚说明

- 原工作区始终在 `main`（`042a1f8`），未跟踪文件未动。
- 迁移分支未推送前：`git worktree remove` 即放弃。
- 合并后：只用 `git revert`；普通提交逆序 revert，deploy subtree 用 `git revert -m 1`。不用 reset/rebase/force-push。
- 已删除分支均可从 `main`、保留实验分支或 annotated tag 恢复。

## 阶段状态与收尾

- **阶段 3 已完成**：deploy subtree 第二父为 `b297b3f`，四套 `ARTIFACTS.md` 已生成。
- **阶段 4 已完成**：378413、分析报告、旧部署包已按 `archive/`/`shared/` 边界纳入。
- **阶段 5 收尾**：PR 以 merge commit 合入 `main`；每次远程写操作前重新读取 heads/tags；创建保护 tag；启用 `main` 保护；最后将远程 heads 收敛为 3 个。
- 训练长跑和 Jetson 真机验证属于后续模型发布验收，不阻断仓库布局合并。
