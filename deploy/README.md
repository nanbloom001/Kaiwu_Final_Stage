# deploy/

Jetson Sim2Real 真机部署树（四套并列，subtree 导入自 `origin/deploy/jetson-sim2real`，独立历史根作为第二父保留）。

> **部署完整性标准**：本轮采用"源码自洽 + 外部制品清单"。Jetson 二进制、ONNX Runtime、部分 ONNX 不直接提交进 Git。每套目录的 `ARTIFACTS.md` 记录所需制品、当前状态与 SHA256。

## 四套部署树

| 目录 | 目标模型 | checkpoint | 状态 |
|---|---|---|---|
| [`sim2real_test_loco/`](./sim2real_test_loco/) | lbc_loco vision student (Actor80, 楼梯测试) | `model.ckpt-vision-378413.pkl` ✅已入库 | 源码+ckpt 在，ONNX/二进制待生成/构建 |
| [`sim2real_test_st9/`](./sim2real_test_st9/) | lbc_loco vision student (UWB goal-driven 避障) | `model.ckpt-vision-378413.pkl` ✅已入库 | 同上（config 与 loco 不同） |
| [`sim2real_test_st7/`](./sim2real_test_st7/) | ST7-Opt3-D1 蒸馏学生 (Actor80, goal_noise) | ❌缺失 `model.ckpt-track-lbc-loco-28608.pkl` | **源码自洽，不可直接启动**（缺 ckpt/ONNX/二进制） |
| [`sim2real_test_standard/`](./sim2real_test_standard/) | standard locomotion (Actor77, 无 goal) | ❌缺失 `model.ckpt-hjcnew-20288.pkl` | **源码自洽，不可直接启动**（缺 ckpt/ONNX/二进制） |

## 约定

- **不去重、不抽公共代码**：四套目录保持相互并列结构，Git 对相同 blob（如两份 378413 checkpoint）自动复用对象。
- **ST7/standard 缺失制品**：需人工提供实际 Jetson 路径或制品哈希；未补齐前只能源码静态验收。
- **`LOCO_TEST_ROOT`**：每套目录的 scripts 通过该环境变量覆盖部署根目录（默认 `/home/unitree/Kaiwu-test/<tree>/`）。
- **接口契约**：见 [`../shared/interfaces/server-deploy-contract.md`](../shared/interfaces/server-deploy-contract.md)（待人工填充）。
- 部署分支历史作为本仓库 subtree 合并的第二父保留（合并提交 `6aed926`，第二父 = `b297b3f`）。
