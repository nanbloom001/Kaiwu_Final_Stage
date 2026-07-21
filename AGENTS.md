# AI Agent 仓库操作规则

本文件适用于仓库根目录及其全部子目录。任何 AI Agent 在读取、分析、修改、提交或推送本仓库前，都必须遵守以下规则。

## 开始任务前必须阅读

按顺序阅读并执行：

1. 本文件。
2. [`CONTRIBUTING.md`](./CONTRIBUTING.md)。这是详细协作流程的唯一规范来源。
3. 根目录 [`README.md`](./README.md)，确认目录边界和当前活动入口。
4. 涉及训练端时，再阅读 [`server/README.md`](./server/README.md) 与 [`server/CHANGELOG.md`](./server/CHANGELOG.md)。
5. 涉及部署端时，再阅读目标部署目录的 `ARTIFACTS.md`；默认稳定入口对应 [`deploy/sim2real_test_loco/ARTIFACTS.md`](./deploy/sim2real_test_loco/ARTIFACTS.md)。
6. 涉及训练与部署接口时，再阅读 [`shared/interfaces/server-deploy-contract.md`](./shared/interfaces/server-deploy-contract.md)。

开始任何修改前，首先执行：

```bash
git status --short --branch
```

记录已有的未提交、已暂存和未跟踪文件。它们默认属于用户，不得修改、暂存、移动或删除，除非任务明确将其纳入范围。

## 分支与修改边界

- 不得直接在 `main` 上开发、提交或推送。新任务应从最新的 `origin/main` 创建短期功能分支；继续用户明确指定的已有分支时，先核对该分支、远程同名分支和 `origin/main`。
- 不得为了获得“干净工作区”而清理任务范围外的文件。
- `server/` 与 `deploy/` 必须保持各自独立运行，不得新增彼此之间或指向 `shared/`、`archive/` 的运行时依赖。
- 网络输入、观测顺序、归一化、goal、checkpoint、ONNX I/O 或 `deploy.yaml` 的契约变化，必须在同一个 PR 中原子更新 `server/`、`deploy/` 和接口契约，不能让 `main` 处于两端不兼容状态。
- 默认可部署路线是 `deploy/sim2real_test_loco`。不得把其他实验目录描述成默认稳定入口；其真实可用性以各自 `ARTIFACTS.md` 为准。

## 提交与推送禁令

- 禁止 force-push、重写共享历史、移动或覆盖已发布 Tag。
- 禁止使用 `git reset --hard`、对共享分支 rebase，或未经审查执行 `git push --tags`。
- 禁止提交凭据、Token、Cookie、私钥，以及缓存、日志、`.DS_Store`、`.nfs*`、`__pycache__` 或任务外文件。
- 只显式暂存本任务文件；提交前必须检查暂存清单。
- 回滚 `main` 使用 `git revert`，不得 reset、rebase 或 force-push。

## 每次推送前必须执行

重新阅读 [`CONTRIBUTING.md` 的“推送前检查”](./CONTRIBUTING.md#推送前检查)，逐项执行其中的命令、同步判断和验证要求。发现远程分支发生意外变化、同名分支发生分叉、制品状态不明或接口无法原子更新时，应停止推送并向用户说明。

## AI Agent 合并授权

仓库所有者明确授权 AI Agent：PR 完成 [`CONTRIBUTING.md` 的“AI Agent 合并授权与合并前检查”](./CONTRIBUTING.md#ai-agent-合并授权与合并前检查) 后，可以直接合并进 `main`，无需再次请求人工确认，也不要求额外 reviewer approval。

该授权只适用于已经审核的准确 PR head，不授权跳过分支保护、验证、制品检查或接口一致性检查。合并时必须使用 `--match-head-commit` 锁定已审核 SHA。PR head 或 `origin/main` 在检查后变化、PR 不为 `MERGEABLE/CLEAN`、存在未解决审查意见或失败检查、所需训练/部署证据缺失、变更范围不明时，必须停止合并并向用户报告。
