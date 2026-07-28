# main 归档与 Standard 视觉主线合并记录

> 日期：2026-07-24
> 当前稳定 main：`4f3d923fa223dbb8c0907c41f20c7352008152f0`
> 后续功能分支：`codex/visual-policy-optimization`

## 1. 归档与合并结果

合并前的 `main@ba175ca0e58fcd9c3d3b1a42fb84a353619dd891` 已使用 annotated
Tag 保护：

```text
archived/main-pre-standard-vision-20260724
```

随后按依赖顺序合入：

1. PR #17：Standard DAgger R2，合并后 main 为 `2e060ad`；
2. PR #18：Standard 深度视觉蒸馏 10 小时版本，最终 main 为 `4f3d923`。

两个原始开发节点另有 annotated Tag：

```text
archived/standard-dagger-r2-source-20260724 -> 6b43450
archived/depth-vision-source-20260724       -> f47fdc6
```

临时 release 分支已在远程删除。旧的
`codex/standard-bridge-r1` 与 `codex/standard-bridge-r1-minimal` 暂时保留，
仅用于追溯平台启动、同步与缺文件问题，不作为新训练分支的父节点。

## 2. 旧 main 中仍有正向参考价值的内容

- Track 配置和 Goal/UWB 边界：用于未来高层导航接口审计，不作为当前 Standard
  低层训练参数；
- deploy 四套目录与 `ARTIFACTS.md`：用于维持训练包和部署制品的严格边界；
- 本地同步、网页监控与平台评估配置手册：用于复现实验和排查平台问题；
- 通用 PPO、奖励桥接、环境监控与测试组织方式：可以复用接口和诊断机制，但
  每个训练超参数仍需重新论证。

## 3. 只保留为踩坑证据的内容

- D4/D5 的配置、checkpoint、低锚定/衰减调度和训练结论；
- 旧 `format="visual_ppo"` 训练包；
- 把视觉 PPO 当成无状态 MLP、打乱 LSTM 时间序列的 minibatch；
- 未完成 S0 对照就扩大命令域、改奖励或直接长训；
- forced promotion、弱保护和仅凭 loss 判断楼梯能力。

这些内容不得作为下一阶段的正向初始化、超参数或实现来源。D1–D3 已被阶段 4
重建路线取代，D4/D5 已判定失败；现存历史 TOML 和测试不作为当前 `conf.py`
的活动入口。实验结论保留在 Changelog/分析文档，源码原文由上述归档 Tag
完整保留。

## 4. 新分支边界

`codex/visual-policy-optimization` 从合并后的 `main@4f3d923` 创建，只实现：

- 从 `visionfull-28401` 初始化 Standard 视觉低层；
- recurrent PPO、TBPTT、冻结 S0 锚定和命令包络训练；
- `kaiwu_train_v1` 的 `standard_visual_ppo` 恢复包；
- 平台 smoke、S0/S1 同集比较所需的日志与保存契约。

本分支不实现 Goal/UWB、Track 高层、迷宫、部署 ONNX 或真机控制。未经平台
tensor smoke 和 S0/S1 评估，不得合并回 `main`。
