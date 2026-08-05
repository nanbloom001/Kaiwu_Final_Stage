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
7. 任务包含 Bug 诊断、回归修复或线上故障处理时，再阅读
   [`shared/分析记录/Bug修复台账.md`](./shared/分析记录/Bug修复台账.md)，先检索相同症状、
   平台边界和既有回归测试，再开始修改。
8. 任务涉及腾讯开悟容器中的 Isaac Lab、Unitree RL Lab、Unitree ROS、地形生成、
   传感器、课程或平台运行框架时，先阅读
   [`shared/container_source_mirror/README.md`](./shared/container_source_mirror/README.md)，
   再检查本地镜像清单并读取相关平台源码。

开始任何修改前，首先执行：

```bash
git status --short --branch
```

记录已有的未提交、已暂存和未跟踪文件。它们默认属于用户，不得修改、暂存、移动或删除，除非任务明确将其纳入范围。

## 平台托管源码本地镜像

本地平台源码镜像固定位于
`shared/arena_frontend_monitor/runtime/container_source_mirror/`，目录本身由 Git 忽略。
涉及平台托管实现的分析必须先读取其中的 `mirror_manifest.json`，确认 `generated_at`、
`remote_roots`、文件 SHA256 和目标文件是否存在，再读取镜像中的相关源码。禁止只依据仓库内
训练代码推断平台地形、传感器、课程、重置或运行框架行为。

镜像是本机只读查询缓存，不是实时容器状态，也不是 `server/` 或 `deploy/` 的运行时依赖。
若镜像缺失、目标文件未收录、清单来自旧容器，或结论依赖当前容器运行时对象、张量 shape、
动态配置和补丁状态，必须通过腾讯开悟 RPC 在线复核，并在结论中区分“镜像证据”与
“当前容器证据”。刷新镜像使用
[`shared/container_source_mirror/pull_container_source_mirror.py`](./shared/container_source_mirror/pull_container_source_mirror.py)；
禁止提交生成出的镜像文件、Token、Cookie、`.env`、日志、模型、二进制或第三方资产。

## Bug 修复强制记录

凡是修改代码、配置、同步工具、训练/评估入口或部署逻辑来修复 Bug，必须在同一次任务和
同一个 PR 中更新 [`shared/分析记录/Bug修复台账.md`](./shared/分析记录/Bug修复台账.md)。
`server/CHANGELOG.md` 只说明“改了什么”，不能替代 Bug 台账中的根因和验证记录。

开始修复前：

1. 在台账中按报错原文、日志关键词、模块名、checkpoint 标签和平台行为检索历史案例；
2. 复用已经验证的修复边界与回归测试，禁止把历史失败方案当作新方案重新引入；
3. 记录当前分支、父模型/制品、实际配置、复现证据和工作区状态；
4. 无法稳定复现时，必须把结论标为“假设”或“待验证”，不得编造确定根因。

完成修复后，台账条目至少包含：

- 日期、Bug ID、状态和影响范围；
- 用户可见症状及关键日志原文；
- 根因与排除过的错误方向；
- 修改文件、核心修复和为什么这样修；
- 本地测试、容器同步、平台 smoke、评估/真机验证分别做到哪一步；
- 回归测试或长期防线；
- 关联 commit/PR、任务名、模型 ID、checkpoint、SHA256；不适用或未知必须明确写出；
- 遗留风险、回滚方式和再次遇到时的最短检查路径。

状态必须如实使用“调查中”“代码已修复待验证”“本地已验证”“平台已验证”“评估已验证”
或“已回滚”。只有实际取得对应层级证据后才能升级状态。日志消失、测试被跳过、文件上传
成功或 HTTP 200 都不能单独证明平台行为已经修复。若后续发现旧结论错误，不得静默改写
历史；应在原条目下追加带日期的更正记录。

纯文档错字、格式调整和没有行为影响的重命名不需要登记。尚未修复的观察可以登记为“调查中”，
但不得写成已修复。台账中禁止记录 Token、Cookie、私钥、完整凭据或其他秘密。

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
