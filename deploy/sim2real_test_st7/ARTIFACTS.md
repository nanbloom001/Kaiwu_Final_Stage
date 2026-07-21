# ARTIFACTS - sim2real_test_st7

## 概述
Go2 lbc_loco 阶段（VisionLoco）独立部署树，对应 ST7-Opt3-D1 学生蒸馏策略（goal_noise 开启）。
`config.yaml` 默认 `command_source: fixed`，固定 vx=0.62 m/s。
本树源码自洽，但**缺失所有模型制品与运行时二进制**，不可直接启动。

## 所需制品清单
| 制品 | 类型 | 当前状态 | SHA256 | 说明 |
|---|---|---|---|---|
| `model.ckpt-track-lbc-loco-28608.pkl` | checkpoint | ❌待提供 | (待人工提供) | lbc_loco 格式，含 `vision_encoder_state_dict` + `teacher_actor_state_dict`。训练日志 log-591723-18327676。对应 ST9-Opt3-D1 student 蒸馏 step 28608，teacher_actor 80 维（含 goal）。来源：`logs/loco/params/deploy.yaml:37-44` |
| `policy.onnx` | ONNX | ❌待提供/待生成 | `4dd6c5f6ff3d4a0a150531b85bde62373d3fbd562067dbe291f752d0c86f61a3`（deploy.yaml 记载的历史值） | 由 `export_loco_onnx.py` 从上方 ckpt 生成。应放 `logs/loco/exported/policy.onnx`。当前目录不存在。 |
| `go2_loco_ctrl` | C++ binary | ❌待构建 | (待构建后记录) | 源码在 `runtime/unitree_rl_lab_test/deploy/robots/go2_loco/`。需在 Jetson aarch64 上 cmake 构建，依赖 ONNX Runtime + librealsense2 + Unitree SDK。当前 `build/` 目录不存在（gitignored）。 |
| `onnxruntime-linux-aarch64-1.19.2` | runtime dep | ❌待部署 | (待人工提供) | ONNX Runtime 1.19.2 for aarch64，预期路径：`deploy/thirdparty/onnxruntime-linux-aarch64-1.19.2/`。当前 `deploy/thirdparty/` 不存在（gitignored）。 |
| RealSense SDK 2.54.2 | runtime dep | (Jetson 系统级) | - | 系统已装（见 config.yaml depth.filters 配置），非本树制品。 |

## ONNX I/O 契约（export_loco_onnx.py:274-280）
输入(8): `depth[1,180,320,1]` `proprio[1,45]` `goal[1,4]` `loco_h[2,1,64]` `loco_c[2,1,64]` `nav_h[2,1,64]` `nav_c[2,1,64]` `cmd_override[1,4]`
输出(8): `cmd[1,3]` `cmd_raw[1,3]` `clearance[1,3]` `joint[1,12]` `loco_h_out` `loco_c_out` `nav_h_out` `nav_c_out`

**关键语义**：teacher_actor 吃 80 维输入（`cat[proprio(45), loco_latent(32), goal[:, :3]]`）。goal 参与计算（与 standard 不同）。loco LSTM 状态由调用方逐帧回喂；nav LSTM 状态原样透传。joint 为原始动作（C++ runner 做 scale/offset/clip）。

## 生成命令
```bash
python export_loco_onnx.py \
    --ckpt model.ckpt-track-lbc-loco-28608.pkl \
    --out  logs/loco/exported/policy.onnx
```
⚠️ 需要先拿到 `.pkl` checkpoint。

## Jetson 安装位置
默认：`/home/unitree/Kaiwu-test/sim2real_test_st7`（由 `LOCO_TEST_ROOT` 环境变量覆盖，见 `scripts/run_loco_stage_test.sh:12`）。
- binary：`runtime/unitree_rl_lab_test/deploy/robots/go2_loco/build/go2_loco_ctrl`
- model：`runtime/unitree_rl_lab_test/logs/loco/exported/policy.onnx`
- config：`runtime/unitree_rl_lab_test/deploy/robots/go2_loco/config/config.yaml`
- ONNX Runtime lib：`runtime/unitree_rl_lab_test/deploy/thirdparty/onnxruntime-linux-aarch64-1.19.2/lib`

## LOCO_TEST_ROOT 使用方法
所有脚本读取 `$LOCO_TEST_ROOT` 环境变量（默认 `/home/unitree/Kaiwu-test/sim2real_test_st7`）。设置到期望落点即可切换部署根目录，无需修改脚本。

## ⚠️ 缺失制品说明
- **checkpoint 缺失**：`model.ckpt-track-lbc-loco-28608.pkl` 未在本树。需人工找到训练产物（训练日志 log-591723-18327676）或提供等效产物，提供实际 Jetson 路径或制品哈希。
- **ONNX 缺失**：`logs/loco/exported/policy.onnx` 不存在。可由上述 checkpoint 调用 `export_loco_onnx.py` 生成，或直接提供 Jetson 上已验证的 ONNX 文件。
- **C++ binary 缺失**：`build/go2_loco_ctrl` 不存在。需在 Jetson 上完成 cmake 构建（依赖 aarch64 版 ONNX Runtime 1.19.2、librealsense2、Unitree SDK）。
- **ONNX Runtime 缺失**：`deploy/thirdparty/` 被 gitignore，需人工下载 aarch64 版本部署。
- **当前状态**：源码自洽且可做静态验收，但**不可直接启动 Go2 实机测试**。需补齐以上制品后，按手册顺序：构建 binary -> 部署 ONNX Runtime -> 放置 ONNX -> 运行 `run_loco_stage_test.sh --check` -> 正式 run。
