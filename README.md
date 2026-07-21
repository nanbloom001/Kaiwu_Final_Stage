# Kaiwu_Final_Stage

腾讯开悟"四足机器人自主导航 Sim2Real"决赛仓库。本仓库采用四层目录结构，将**训练运行时**、**真机部署**、**共享文档**与**历史归档**分离，避免活动训练/部署入口互相引用。

## 目录结构

| 目录 | 用途 | 运行时依赖 |
|---|---|---|
| [`server/`](./server/) | 训练运行时代码（活动基线 = ST7-Opt3）。腾讯开悟上传/同步以 `server/` 为项目根 | 自身目录运行 |
| [`deploy/`](./deploy/) | Jetson Sim2Real 真机部署树（四套并列：loco/st7/st9/standard） | 不导入 server 的 Python/配置 |
| [`shared/`](./shared/) | 规则说明、分析记录、接口契约、协作文档 | 禁止成为运行时依赖 |
| [`archive/`](./archive/) | 历史代码快照、旧部署包 | 不能被活动训练/部署引用 |

## 活动基线

- **训练基线**：`server/` = `codex/st7-opt3`（`9b0b3df`，ST7 Opt3 UWB goal noise）
- **部署入口**：`deploy/`（独立历史根，经 subtree 导入；四套并列不去重）

## 快速开始

1. 训练：进入 `server/`，以 `server/` 为工作目录运行（腾讯开悟容器内执行 `server/train_test.py`）。
2. 真机部署：见 `deploy/<tree>/ARTIFACTS.md`（需人工补齐 ST7/standard 缺失的 checkpoint/ONNX 制品）。
3. 接口契约：见 [`shared/interfaces/server-deploy-contract.md`](./shared/interfaces/server-deploy-contract.md)。
4. 分支迁移与源 SHA 锚点：见 [`shared/project-context/branch-migration-register.md`](./shared/project-context/branch-migration-register.md)。

## 仓库布局背景

本仓库于 2026-07-21 从扁平结构迁移为四层结构，迁移在独立 worktree 分支 `codex/repository-layout-migration`（起点 `codex/st7-opt3`）完成，未重写任何历史、未删除任何旧分支。详见 [`shared/project-context/repository-layout.md`](./shared/project-context/repository-layout.md) 与 [`shared/project-context/branch-migration-register.md`](./shared/project-context/branch-migration-register.md)。
