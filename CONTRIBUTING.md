# Kaiwu_Final_Stage 仓库协作与版本管理规范

本规范面向第一次参与 Git 协作的成员和 AI Agent。它说明什么时候更新 `main`、如何安全同步、怎样提交实验，以及训练端和部署端同时修改时如何避免覆盖彼此的工作。

AI Agent 必须先阅读 [`AGENTS.md`](./AGENTS.md)。仓库布局和活动入口见 [`README.md`](./README.md)。

## 1. 先理解四个 Git 概念

- **分支（branch）**：可继续产生新提交的工作路线。普通修复使用短期分支；仍在发展的、不兼容的大型实验才保留长期分支。
- **提交（commit）**：一次可审查、可回退的修改快照。一个提交只表达一个意图。
- **Pull Request（PR）**：把分支中的提交送入 `main` 的审查和验证入口。
- **标签（Tag）**：固定指向某个历史提交的只读里程碑。Tag 不用于继续开发，适合保存关键基线、最佳模型或删除分支前的独立节点。

## 2. main 的含义与更新频率

`main` 采用**事件驱动更新**，不按每日或每周集中更新。一个 PR 完成与其风险相匹配的验证后，应及时合入；没有完成验证的半成品不得为了“同步进度”进入 `main`。

`main` 必须始终代表最新稳定版本：

- 训练代码、部署代码和共享接口契约彼此一致。
- 默认部署路线固定为 `deploy/sim2real_test_loco`。
- 文档 PR 可以不执行真机验证，但必须通过文档、链接和差异检查，且不得改变运行行为。
- `st7`、`st9`、`standard` 等非默认实验目录允许暂缺制品，但其 `ARTIFACTS.md` 必须明确标记“不可直接部署”，不能冒充稳定入口。

“可部署”不是指目录存在，而是同时满足：

1. 源码与配置完整。
2. checkpoint、ONNX、运行库和二进制已经入库，或在对应 `ARTIFACTS.md` 中记录了可取得位置、状态与 SHA256。
3. 对应启动脚本的 `--check` 或 preflight 通过。
4. 最近一次部署验证的环境、命令、结果和制品版本有记录。

因此，默认入口的指定不等于当前制品已经自动齐全；部署前必须以 [`deploy/sim2real_test_loco/ARTIFACTS.md`](./deploy/sim2real_test_loco/ARTIFACTS.md) 和最新验证记录为准。

## 3. 仓库目录责任

| 目录 | 责任 | 协作要求 |
|---|---|---|
| `server/` | 独立训练项目 | 以该目录为运行根；变更记录写入 [`server/CHANGELOG.md`](./server/CHANGELOG.md) |
| `deploy/` | 独立真机部署项目 | 每条部署路线维护自己的 `ARTIFACTS.md`，不得运行时依赖 `server/` |
| `shared/` | 规则、分析、接口与协作文档 | 只作共享资料，不成为两端运行时依赖 |
| `archive/` | 历史快照和旧资料 | 不得被活动训练或部署代码引用 |

跨端数据定义以 [`shared/interfaces/server-deploy-contract.md`](./shared/interfaces/server-deploy-contract.md) 为准。

## 4. 开始工作：先同步，再创建分支

正常的新任务必须在**干净工作区**从最新 `main` 建立分支：

```bash
git status --short --branch
git fetch --all --prune
git switch main
git pull --ff-only origin main
git switch -c <成员或工具>/<范围>-<主题>
```

例如：

```bash
git switch -c codex/deploy-camera-watchdog
```

这里 `pull --ff-only` 只允许本地 `main` 快进到远程 `main`，不会自动制造合并提交。不要先在 `main` 修改，再临时补建分支；也不要在存在未提交修改时盲目 `pull`。如果开始时已经有用户修改，应先记录并避开，必要时请用户决定如何处理。

如果用户明确要求继续一个已有功能分支，可以留在该分支，但仍须先检查：

```bash
git status --short --branch
git fetch --all --prune
git log --oneline HEAD..origin/main
git log --oneline origin/main..HEAD
```

## 5. “先 pull 再 commit”的准确做法

“先 pull 再 commit”真正要解决的是：不要基于过时主线提交并覆盖别人。正确流程分为两个时间点。

### 修改前

先把干净的本地 `main` 快进到最新 `origin/main`，再创建分支。不要对带有未提交修改的工作区直接执行 `pull`。

### 修改完成、准备提交前

先确认主线是否在你工作期间前进：

```bash
git fetch --all --prune
git log --oneline HEAD..origin/main
```

- 没有输出：`origin/main` 没有当前分支尚未包含的新提交，可以继续验证和提交。
- 有输出：主线已经更新。先把当前工作做成一个**本地安全提交**，再合并主线：

```bash
git add <本任务文件>
git commit -m "<本任务提交信息>"
git merge origin/main
```

然后解决冲突、检查最终差异并重新运行全部相关测试。不要用 stash 隐藏来源不明的用户文件，也不要为了获得线性历史而 rebase 已推送或共享的分支。

如果远程同名功能分支也被其他人更新，只能在干净工作区执行：

```bash
git pull --ff-only origin <功能分支>
```

若 `--ff-only` 拒绝，说明本地和远程已经分叉。此时停止，核对双方提交并人工协调；禁止用 force-push 覆盖另一方。

## 6. server 与 deploy 并发修改

两个人可以分别开发训练端和部署端，但是否能拆成两个 PR 取决于接口有没有变化。

### 可以使用独立 PR

- 训练内部实现变化，不改变导出和部署契约。
- 部署内部日志、诊断或安全修复，不改变模型输入输出。
- 纯文档更新。

两个 PR 不能假设同时合入。第一个 PR 合入后，第二个合并者必须重新同步 `main`、解决冲突并重跑验证。

### 必须使用一个原子 PR

以下任一内容发生变化时，必须在同一个 PR 中同时更新 `server/`、`deploy/` 和 [`shared/interfaces/server-deploy-contract.md`](./shared/interfaces/server-deploy-contract.md)：

- 网络输入维度或网络结构；
- 观测字段及其顺序；
- 深度、proprio、command 或 goal 的归一化和语义；
- checkpoint 格式或选择；
- ONNX 输入、输出、shape 或状态回喂；
- `deploy.yaml` 中与上述契约有关的配置。

不允许先合训练端、稍后再补部署端，因为这会让中间状态的 `main` 不可部署。

每个 PR 都要声明影响范围：`server`、`deploy`、`shared contract`、`artifact`。普通配置调整、单点修复和文档使用短期分支，合并后删除；只有网络架构、训练算法、观测契约或互不兼容的部署路线才允许长期实验分支。

## 7. 提交规范

提交消息使用以下格式：

```text
feat(server): add ...
fix(deploy): correct ...
docs: document ...
exp(st7): evaluate ...
archive: preserve ...
```

每个提交只表达一个意图。只显式暂存任务文件：

```bash
git add path/to/file1 path/to/file2
git diff --cached --name-only
git diff --cached --check
```

禁止使用不经检查的 `git add .`，尤其当根目录存在旧资料、训练产物或未跟踪目录时。

## 8. 推送前检查

每次推送前，AI Agent 必须重新阅读本节。按顺序执行：

```bash
git status --short --branch
git diff --cached --name-only
git diff --cached --check
git fetch --all --prune
git log --oneline HEAD..origin/main
```

然后确认：

1. 当前不在 `main`。
2. 暂存区只包含本任务文件，任务外未跟踪文件仍保持原状。
3. 已运行与改动风险匹配的测试，并记录结果。
4. `origin/main` 若有新提交，已按第 5 节先安全提交、合并主线、解决冲突并重测。
5. 远程同名分支没有未知提交或分叉。
6. 接口改动已原子覆盖两端与共享契约。
7. 制品状态和 SHA256 真实可核查，没有把 Git LFS pointer 当成可用制品。
8. 差异中没有凭据、Token、Cookie、私钥、缓存、日志或任务外文件。

满足后只推送当前分支：

```bash
git push -u origin <当前分支>
```

禁止 force-push。

## 9. PR 内容与合并方式

PR 描述必须记录：

- 改动目的和影响目录；
- 父提交与父 checkpoint（不涉及模型时写“不涉及”）；
- 配置、观测、网络或部署契约变化；
- 测试、preflight 和真机部署证据，未执行的项目也要明确写出；
- checkpoint、ONNX、二进制、运行库等制品的状态与 SHA256；
- 回滚方法。

普通小改动默认使用 **squash merge**，合入后删除短期分支。需要保留实验血缘、subtree 或仓库迁移历史时，才使用 **merge commit**。PR 合并后，其他仍在开发的分支必须重新同步 `main` 并验证。

### AI Agent 合并授权与合并前检查

仓库所有者明确授权 AI Agent：完成本节全部检查后，可以直接合并 PR 进入 `main`，无需为合并动作再次请求人工确认，也不要求额外 reviewer approval。此授权不等于允许直接 push `main`，所有改动仍必须通过 PR 和分支保护。

合并前必须：

1. 重新阅读 [`AGENTS.md`](./AGENTS.md) 和本节。
2. 执行 `git fetch --all --prune`，直接核对最新 `origin/main`、远程功能分支和 PR head。
3. 确认功能分支已经包含最新 `origin/main`；如果主线刚刚更新，先合并主线、解决冲突并重跑验证。
4. 确认 PR head 正是刚刚审核和测试的 commit，没有未知的新提交或任务外文件。
5. 确认 PR 状态为 `MERGEABLE/CLEAN`，没有失败检查、requested changes 或未解决的审查意见。
6. 确认 PR 描述已经如实记录改动范围、验证证据、制品状态和回滚方法。
7. 训练、部署或接口改动必须完成对应验证；缺少规则要求的长跑、preflight、真机或制品证据时不得以文档检查代替。
8. 根据改动类型确认 squash merge 或 merge commit，并记录待合并的 `CANDIDATE_SHA`。

普通短期分支使用带 head 锁定的 squash merge：

```bash
CANDIDATE_SHA=$(git rev-parse origin/<功能分支>)

gh pr merge <PR编号> \
  --repo nanbloom001/Kaiwu_Final_Stage \
  --squash \
  --match-head-commit "$CANDIDATE_SHA" \
  --delete-branch
```

只有需要保留实验血缘、subtree 或迁移历史时，才将 `--squash` 改为 `--merge`。禁止使用 rebase merge。

`--match-head-commit` 是强制安全锁：如果检查完成后 PR head 又被更新，GitHub 必须拒绝本次合并。AI Agent 不得去掉该参数绕过拒绝，而应重新 fetch、审核和测试。

合并后立即执行：

```bash
git fetch --all --prune
gh pr view <PR编号> \
  --repo nanbloom001/Kaiwu_Final_Stage \
  --json state,mergedAt,mergeCommit
git log -1 --oneline origin/main
```

确认 PR 为 `MERGED`、远程 `main` 已更新，并检查短期远程分支是否按计划删除。由于 squash merge 会生成新的提交，不能用“原功能分支 tip 必须成为 `main` 祖先”判断 squash 是否成功，应以 PR 的 `mergeCommit` 和最终文件树为准。

出现以下任一情况必须停止合并并向用户报告：

- PR head、远程功能分支或 `origin/main` 在检查后发生变化；
- PR 冲突、不可合并或 merge state 不是 `CLEAN`；
- 测试失败、检查未完成或验证证据不足；
- 有 requested changes、未解决审查意见或未知提交；
- server 与 deploy 接口只更新了一端；
- checkpoint、ONNX、运行库、二进制或 SHA256 状态不明；
- 需要保留历史但合并方式尚未确认。

## 10. 实验、Changelog 与制品

- 小调参写入独立 TOML、[`server/CHANGELOG.md`](./server/CHANGELOG.md) 和实验文档，不为每个参数建立永久分支。
- 每个实验记录父模型、父 checkpoint、Git commit、配置哈希、seed、训练时长、指标和结论。
- `ARTIFACTS.md` 记录 checkpoint、ONNX、二进制和运行库的来源、当前状态、放置位置与 SHA256。
- Git LFS pointer 只是下载指针，不是可执行或可加载的真实制品；文件未完整取得时必须标记缺失。
- 禁止提交缓存、运行日志、`.DS_Store`、`.nfs*`、`__pycache__` 和任务范围外文件。

## 11. Tag 与分支收敛

关键基线、最佳 checkpoint 和删除分支前的独立历史节点使用 annotated tag：

```bash
git tag -a st7-example-v1 <commit-sha> -m "ST7 example baseline"
git push origin st7-example-v1
```

Tag 命名使用：

- `st7-*`
- `st9-*`
- `deploy-*-v*`
- `archived/*`

只推送经过确认的具体 Tag，不使用未经审查的 `git push --tags`，也不移动已发布 Tag。

删除分支前，必须确认分支 tip 满足至少一项：

1. 已经是 `main` 的祖先；
2. 已被仍保留的活动分支包含；
3. 已由明确的 annotated tag 保护。

## 12. 回滚

- 功能分支尚未合入：修复分支或关闭 PR，不影响 `main`。
- PR 已合入 `main`：新建回滚 PR，使用 `git revert` 生成反向提交。
- merge commit 回滚时需确认主父，例如 `git revert -m 1 <merge-sha>`。
- 不得对共享历史使用 reset、rebase 或 force-push。
- 已删除分支需要恢复时，从受保护的 commit 或 Tag 创建新分支。

## 13. 三个常见协作场景

### 单人完成一个小修复

同步最新 `main` → 创建短期分支 → 修改和测试 → 推送前检查 → 提交和推送 → 创建 PR → squash merge → 删除分支。

### server PR 先于 deploy PR 合入

server PR 合入后，deploy 分支不能直接按旧基线合入。deploy 开发者先 fetch，把最新 `origin/main` merge 到自己的安全提交上，再重新检查部署测试和接口契约，然后更新 PR。

### 修改期间 main 更新

不要对脏工作区直接 pull。先把任务文件做成本地安全提交，再 `git merge origin/main`，解决冲突并重测；确认远程同名分支未分叉后再推送。
