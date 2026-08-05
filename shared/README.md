# shared/

规则、分析记录、接口契约、协作文档与人工运维工具。**禁止成为运行时依赖**--`server/` 与 `deploy/` 不得 import 或路径引用本目录。

## 内容

| 路径 | 说明 |
|---|---|
| `规则说明/` | 腾讯开悟官方文档镜像（开发指南 + 强化学习框架 + 资源配图） |
| `分析记录/` | 团队分析报告（迁移审计、Sim2Real 部署问题追踪等，按日期归档） |
| `interfaces/` | 跨目录接口契约 |
| `project-context/` | 仓库布局与分支迁移登记 |
| `container_source_mirror/` | 平台托管源码本地镜像的刷新工具与读取规则；生成数据不入库 |
| `arena_frontend_monitor/` | 腾讯竞技平台网页端训练日志与监控曲线采集工具 |

## 接口契约

- [`interfaces/server-deploy-contract.md`](./interfaces/server-deploy-contract.md)：区分训练恢复、特权教师和可部署 checkpoint，并固化 Standard Actor77 与 Track Actor80 的边界；发布候选仍需补齐 depth、ONNX、动作和 `deploy.yaml` 的逐字段契约。

## 项目背景

- [`../AGENTS.md`](../AGENTS.md)：AI Agent 开始任务和推送前必须遵守的仓库入口规则。
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md)：面向成员与 AI Agent 的完整协作、同步、提交、PR、Tag、制品和回滚规范。
- [`分析记录/Bug修复台账.md`](./分析记录/Bug修复台账.md)：跨阶段 Bug 的症状、根因、修复、验证层级与复发检查入口；每次行为修复必须追加记录。
- [`project-context/repository-layout.md`](./project-context/repository-layout.md)：四层目录结构说明。
- [`project-context/branch-migration-register.md`](./project-context/branch-migration-register.md)：旧分支源 SHA 锚点 + 迁移分类登记（回滚锚点）。
- [`分析记录/版本训练演进与改动规模详解.md`](./分析记录/版本训练演进与改动规模详解.md)：复赛、决赛、蒸馏/Sim2Real 与导航训练分支的版本演进记录。

## 当前 P2 监控优化计划

- [`分析记录/2026-07-30_P2监控面板精简与方向速度统计优化方案.md`](./分析记录/2026-07-30_P2监控面板精简与方向速度统计优化方案.md)：
  保留 5×4 核心联合分桶，补齐 `vy/wz` 正负方向统计，精简重复面板，并通过
  GPU 窗口聚合和单次 CPU 快照降低监控同步开销。

## 当前 Standard 深度训练计划

- [`分析记录/2026-07-22_Standard深度模型五阶段实施计划.md`](./分析记录/2026-07-22_Standard深度模型五阶段实施计划.md)：冻结从复赛 Standard 10288 到可部署深度视觉 Standard 的五阶段路线、质量诊断与防漂移规则。
- [`分析记录/2026-07-22_10288到latent32结构蒸馏执行计划.md`](./分析记录/2026-07-22_10288到latent32结构蒸馏执行计划.md)：细化第一轮 301→77 结构迁移的纯 BC、分比例 DAgger、学生闭环、checkpoint 和失败回滚流程。
- [`分析记录/2026-07-23_STD-DAGGER-R2执行卡.md`](./分析记录/2026-07-23_STD-DAGGER-R2执行卡.md)：已完成的三小时特权网络结构蒸馏、平台结果、污染保护和 checkpoint 记录。
- [`分析记录/2026-07-23_Standard深度视觉蒸馏10小时长训计划.md`](./分析记录/2026-07-23_Standard深度视觉蒸馏10小时长训计划.md)：已完成的十小时视觉蒸馏方案、实际平台结果和阶段 4 遗留诊断。
- [`分析记录/2026-07-24_Standard视觉学生独立化与发布冻结计划.md`](./分析记录/2026-07-24_Standard视觉学生独立化与发布冻结计划.md)：阶段 5 历史初稿；当前训练调度已由双四小时与 Anchor R2 实施计划取代。
- [`分析记录/2026-07-25_Standard视觉学生双四小时训练计划.md`](./分析记录/2026-07-25_Standard视觉学生双四小时训练计划.md)：将视觉学生独立退火与 command 泛化拆为两个独立四小时任务。
- [`分析记录/2026-07-25_StandardAnchorR2四小时实施计划.md`](./分析记录/2026-07-25_StandardAnchorR2四小时实施计划.md)：当前 `standard-anchor-r2` 的代码、配置、清理、短启动与验收契约。
- [`分析记录/2026-07-24_main归档与Standard视觉主线合并记录.md`](./分析记录/2026-07-24_main归档与Standard视觉主线合并记录.md)：旧 main 的归档、R2/视觉主线合并顺序，以及可复用事实与失败路线的边界。
- [`分析记录/2026-07-23_Standard桥接蒸馏昨夜至今迭代复盘.md`](./分析记录/2026-07-23_Standard桥接蒸馏昨夜至今迭代复盘.md)：记录 R1、HJC minimal、环境/同步/lifecycle 修复到 R2 的完整时间线、踩坑和下一轮检查清单。
- [`分析记录/2026-07-23_STD-BRIDGE-R1执行卡.md`](./分析记录/2026-07-23_STD-BRIDGE-R1执行卡.md)：历史 R1 设计稿，已由 R2 取代。
