# ARTIFACTS - sim2real_test_standard

## 概述
Go2 lbc_loco 阶段常规部署树，对应 standard locomotion 模型（77 维 teacher_actor，无 goal）。
`config.yaml` 默认 `command_source: uwb`，UWB max_vx=0.70 m/s。
本树曾成功在 Jetson 上运行（`logs/` 下有 7 份实机测试记录，时间戳 2026-07-15 ~ 2026-07-16），但迁移工作树中**缺失所有模型制品与运行时二进制**。

## 所需制品清单
| 制品 | 类型 | 当前状态 | SHA256 | 说明 |
|---|---|---|---|---|
| `model.ckpt-hjcnew-20288.pkl` | checkpoint | ❌待提供 | (待人工提供) | lbc_loco 格式，含 `vision_encoder_state_dict` + `teacher_actor_state_dict`。teacher_actor 77 维（无 goal 分支）。来源：`logs/loco/params/deploy.yaml:37-42`。训练日志 log-591723。 |
| `policy.onnx` | ONNX | ❌待提供/待生成 | `97fa1af6c7a282c34043951d0e2dfd5ebf323d605f8e6fc88686c34188b367d1`（历史实机值，见 `logs/20260715_195928_*/manifest.txt:10`） | 由 `export_loco_onnx.py` 生成。历史值来自 2026-07-15 实机测试。当前 `logs/loco/exported/` 仅含 .gitkeep。 |
| `go2_loco_ctrl` | C++ binary | ❌待构建 | `fc25ae9634fdf81e036fcf441d24120e5967fddb8c067ff074b8ef4c18509b56`（历史实机值，见 `manifest.txt:9`） | 源码在 `runtime/unitree_rl_lab_test/deploy/robots/go2_loco/`。Jetson aarch64 上 cmake 构建。当前 `build/` 目录不存在。 |
| `onnxruntime-linux-aarch64-1.19.2` | runtime dep | ❌待部署 | (待人工提供) | 预期路径 `deploy/thirdparty/onnxruntime-linux-aarch64-1.19.2/`。当前不存在。 |
| RealSense SDK | runtime dep | (Jetson 系统级) | - | 非本树制品。 |

## ONNX I/O 契约（export_loco_onnx.py:274-280）
输入(8): `depth[1,180,320,1]` `proprio[1,45]` `goal[1,4]` `loco_h[2,1,64]` `loco_c[2,1,64]` `nav_h[2,1,64]` `nav_c[2,1,64]` `cmd_override[1,4]`
输出(8): `cmd[1,3]` `cmd_raw[1,3]` `clearance[1,3]` `joint[1,12]` `loco_h_out` `loco_c_out` `nav_h_out` `nav_c_out`

**关键语义**：teacher_actor 吃 77 维输入（`cat[proprio(45), loco_latent(32)]`，**不含 goal**）。与 st7 的 80 维区别显著。其余 I/O 端口布局与 st7 一致（8 入 8 出）。

## 生成命令
```bash
python export_loco_onnx.py \
    --ckpt model.ckpt-hjcnew-20288.pkl \
    --out  logs/loco/exported/policy.onnx
```
⚠️ teacher_actor 维度为 77（与 st7 的 80 不通用），必须用 standard 专属 ckpt。

## Jetson 安装位置
默认：`/home/unitree/Kaiwu-test/sim2real_test_standard`（`scripts/run_loco_stage_test.sh:12`）。
- binary：`runtime/unitree_rl_lab_test/deploy/robots/go2_loco/build/go2_loco_ctrl`
- model：`runtime/unitree_rl_lab_test/logs/loco/exported/policy.onnx`
- config：`runtime/unitree_rl_lab_test/deploy/robots/go2_loco/config/config.yaml`
- ONNX Runtime lib：`runtime/unitree_rl_lab_test/deploy/thirdparty/onnxruntime-linux-aarch64-1.19.2/lib`

## LOCO_TEST_ROOT 使用方法
同 st7：环境变量 `$LOCO_TEST_ROOT` 默认 `/home/unitree/Kaiwu-test/sim2real_test_standard`，配置到目标目录即可。

## ⚠️ 缺失制品说明
- **checkpoint 缺失**：`model.ckpt-hjcnew-20288.pkl` 不在本树。需人工从训练日志 log-591723 找回，或提供等效产物。
- **ONNX 缺失**：`logs/loco/exported/policy.onnx` 不存在。历史实机 SHA256 为 `97fa1af6...`（`manifest.txt` 可查），但文件未入版本管理。需从 checkpoint 重新生成或从 Jetson 备份恢复。
- **C++ binary 缺失**：`build/go2_loco_ctrl` 不存在。历史实机 SHA256 为 `fc25ae96...`，但源码需在 Jetson 上重新构建。
- **ONNX Runtime 缺失**：`deploy/thirdparty/` 被 gitignore，需人工部署 aarch64 版本。
- **当前状态**：源码自洽。有 7 份实机运行日志（`logs/20260715_*/`）可做离线分析验证（`analyze_sim2real_logs.py` 可处理这些 CSV），但**不可直接重复启动实机测试**。需补齐制品后按 `run_loco_stage_test.sh` 流程执行。该树曾是首个在 Jetson 上成功运行的 loco 部署树（2026-07-15 固定速度测试），二进制和 ONNX 的历史 SHA256 值可作为重新部署的参照。
