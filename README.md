# Kaiwu_Final_Stage

腾讯开悟"四足机器人自主导航 Sim2Real"决赛仓库。本仓库采用四层目录结构，将**训练运行时**、**真机部署**、**共享文档**与**历史归档**分离，避免活动训练/部署入口互相引用。

## 目录结构

| 目录 | 用途 | 运行时依赖 |
|---|---|---|
| [`server/`](./server/) | 训练运行时代码（Opt3 基线 + J9 接口修复 + Opt4 角速度保护）。腾讯开悟上传/同步以 `server/` 为项目根 | 自身目录运行 |
| [`deploy/`](./deploy/) | Jetson Sim2Real 真机部署树（四套并列：loco/st7/st9/standard） | 不导入 server 的 Python/配置 |
| [`shared/`](./shared/) | 规则说明、分析记录、接口契约、协作文档 | 禁止成为运行时依赖 |
| [`archive/`](./archive/) | 历史代码快照、旧部署包 | 不能被活动训练/部署引用 |

## 资料放置与本地旧副本

仓库根目录只保留上表四个正式内容入口。布局迁移前留下的以下本地目录已由根 [`.gitignore`](./.gitignore) 精确忽略，不属于当前活动目录：

| 根目录本地副本 | 正式归档或维护位置 |
|---|---|
| `unitree_isaaclab_deploy/` | [`archive/unitree_isaaclab_deploy/`](./archive/unitree_isaaclab_deploy/) |
| `代码存档/` | [`archive/代码存档/`](./archive/代码存档/) |
| `分析记录/` | [`shared/分析记录/`](./shared/分析记录/) |

以后新增代码、归档或报告必须写入正式位置，不能继续写入这些被忽略的根目录副本。被忽略的本地副本可能含未归档的大型模型、运行库或临时文件；如需将其中内容提升为正式资料，应先核对来源和差异，再显式迁入对应目录，并同步更新 `ARTIFACTS.md` 或资料索引。不要使用 `git add -f` 绕过该边界。

## 活动基线

- **训练基线**：`server/` 以 `codex/st7-opt3`（`9b0b3df`）为起点，已移植 J9 通用学习率接口修复和 Opt4 角速度保护；TrackNav 仍使用 Opt3 自适应学习率。
- **部署入口**：`deploy/`（独立历史根，经 subtree 导入；四套并列不去重）

## 快速开始

1. 训练：进入 `server/`，以 `server/` 为工作目录运行（腾讯开悟容器内执行 `server/train_test.py`）。
2. 真机部署：默认稳定入口为 `deploy/sim2real_test_loco`，部署前必须核对其 [`ARTIFACTS.md`](./deploy/sim2real_test_loco/ARTIFACTS.md)；其他实验路线的可用性以各自 `ARTIFACTS.md` 为准。
3. 接口契约：见 [`shared/interfaces/server-deploy-contract.md`](./shared/interfaces/server-deploy-contract.md)。
4. 分支迁移与源 SHA 锚点：见 [`shared/project-context/branch-migration-register.md`](./shared/project-context/branch-migration-register.md)。
5. 协作与版本管理：所有贡献者阅读 [`CONTRIBUTING.md`](./CONTRIBUTING.md)；AI Agent 还必须先阅读 [`AGENTS.md`](./AGENTS.md)。

## 协作与版本管理

`main` 采用“验证通过即合并”的事件驱动更新方式，始终代表最新稳定版本，不用于存放半成品。任何开发都应从最新 `main` 创建功能分支，并通过 PR 合入；训练与部署接口变化必须在一个 PR 中原子更新两端及共享契约。完整的同步、提交、实验、Tag、制品和回滚规则见 [`CONTRIBUTING.md`](./CONTRIBUTING.md)。

## 仓库布局背景

本仓库于 2026-07-21 从扁平结构迁移为四层结构，迁移在独立 worktree 分支 `codex/repository-layout-migration`（起点 `codex/st7-opt3`）完成，全程不 rebase、不 force-push。迁移合入后，已被主线或 annotated tag 保护的散落分支按确认后的裁剪策略删除。详见 [`shared/project-context/repository-layout.md`](./shared/project-context/repository-layout.md) 与 [`shared/project-context/branch-migration-register.md`](./shared/project-context/branch-migration-register.md)。
