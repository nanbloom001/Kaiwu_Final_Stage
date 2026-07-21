# shared/

规则、分析记录、接口契约与协作文档。**禁止成为运行时依赖**--`server/` 与 `deploy/` 不得 import 或路径引用本目录。

## 内容

| 路径 | 说明 |
|---|---|
| `规则说明/` | 腾讯开悟官方文档镜像（开发指南 + 强化学习框架 + 资源配图） |
| `分析记录/` | 团队分析报告（迁移审计、Sim2Real 部署问题追踪等，按日期归档） |
| `interfaces/` | 跨目录接口契约 |
| `project-context/` | 仓库布局与分支迁移登记 |

## 接口契约

- [`interfaces/server-deploy-contract.md`](./interfaces/server-deploy-contract.md)：固化 server↔deploy 的 checkpoint 来源/SHA、Actor 输入维度、depth 尺寸/归一化、proprio/command/goal 字段顺序、ONNX I/O、`deploy.yaml` 对应关系。

## 项目背景

- [`project-context/repository-layout.md`](./project-context/repository-layout.md)：四层目录结构说明。
- [`project-context/branch-migration-register.md`](./project-context/branch-migration-register.md)：旧分支源 SHA 锚点 + 迁移分类登记（回滚锚点）。
- [`分析记录/版本训练演进与改动规模详解.md`](./分析记录/版本训练演进与改动规模详解.md)：复赛、决赛、蒸馏/Sim2Real 与导航训练分支的版本演进记录。
