# ARTIFACTS - sim2real_test_loco

## 概述
Go2 lbc_loco 学生策略部署树（楼梯测试 + 固定命令模式）。目标 checkpoint `model.ckpt-vision-378413.pkl`，即 Sim2Real vision student（Actor80），已在 Go2/Jetson 上以固定 cmd [0.15,0,0] + D435i depth 验证可行走。

## 所需制品清单
| 制品 | 类型 | 当前状态 | SHA256 | 说明 |
|---|---|---|---|---|
| model.ckpt-vision-378413.pkl | checkpoint | ✅已入库 | 37429c1e2c1d263844a74ecc2fdb97201ae296663f5ad8890ce173d5525a1288 | Sim2Real vision 学生 Actor80，lbc_loco 训练 ckpt |
| policy.onnx | ONNX | ❌待生成 | - | 需从 ckpt 经 export_loco_onnx.py 导出；当前 `exported/` 仅空目录 |
| go2_loco_ctrl | C++ binary | ❌待构建 | - | 二进制未构建/未提交（`build/` 被 .gitignore）；需在 Jetson 上 cmake 编译 |
| ONNX Runtime | runtime dep | ❌待部署 | - | `thirdparty/onnxruntime-linux-aarch64-1.19.2` 未入库（被 .gitignore） |
| deploy.yaml | config | ✅已入库 | 同源校验中 | `logs/loco/params/deploy.yaml`，与 ckpt 同源（训练日志 log-591723-18327676） |
| config.yaml | config | ✅已入库 | 同源校验中 | FSM.VisionLoco 配置；command_source=uwb，fixed_cmd=[0.7,0,0]（楼梯测试） |

## ONNX I/O 契约
8 入 8 出（与 vision-nav 图同端口契约，loco 用不到的 nav 端口占位透传）：
- **输入**: `depth[1,180,320,1]` `proprio[1,45]` `goal[1,4]` `loco_h[2,1,64]` `loco_c[2,1,64]` `nav_h[2,1,64]` `nav_c[2,1,64]` `cmd_override[1,4]`
- **输出**: `cmd[1,3]` `cmd_raw[1,3]` `clearance[1,3]`（恒 0）`joint[1,12]`（原始动作，未 scale/offset）`loco_h_out[2,1,64]` `loco_c_out[2,1,64]` `nav_h_out[2,1,64]` `nav_c_out[2,1,64]`（原样透传）
- **LSTM 语义**: loco LSTM 状态 (h,c) 须由 C++ runner 逐帧回喂，episode 起始重置为 0；nav LSTM 状态原样透传（此图不更新）
- **proprio[6:9] 覆写**: 推理前必须由 C++ 覆写为上一帧 cmd / 固定注入 cmd（图外契约①）
- **opset=17**, `dynamic_axes=None`（固定 batch=1）

## 生成命令
```bash
python export_loco_onnx.py \
  --ckpt models/model.ckpt-vision-378413.pkl \
  --out  runtime/unitree_rl_lab_test/logs/loco/exported/policy.onnx
```
默认 opset=17，含 Python↔ONNX 数值对齐 verify（8 帧 rollout，tol=1e-4）。

## Jetson 安装位置
`/home/unitree/Kaiwu-test/sim2real_test_loco/`（由 `scripts/run_loco_stage_test.sh` 默认）。安装内容：
- 整个 tree rsync 至该目录
- `runtime/unitree_rl_lab_test/` 包含 C++ 源码 + deploy 树
- 二进制落点: `deploy/robots/go2_loco/build/go2_loco_ctrl`
- ONNX 模型: `logs/loco/exported/policy.onnx`
- 诊断 CSV: `logs/loco/logs/visloco_diag_*.csv`
- ONNX Runtime 依赖: `deploy/thirdparty/onnxruntime-linux-aarch64-1.19.2/lib`（LD_LIBRARY_PATH 注入）

## LOCO_TEST_ROOT 使用方法
脚本通过 `LOCO_TEST_ROOT` 环境变量覆盖部署根目录。默认值即上述 Jetson 落点。核心 runner `run_loco_stage_test.sh` 根据 `$TEST_ROOT` 推导 `$LAB_ROOT`、binary、config、model、log 路径。通过 `run_loco_stage_mode.sh` 可切换 `command_source`（fixed/uwb/nav）。
