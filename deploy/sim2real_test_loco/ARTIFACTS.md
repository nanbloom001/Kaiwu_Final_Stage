# ARTIFACTS - sim2real_test_loco

## 概述
Go2 lbc_loco 学生策略部署树（楼梯测试 + 固定命令模式）。目标 checkpoint
`model.ckpt-vision-378413.pkl`，即 Sim2Real vision student（Actor80）。历史
真机证据是在 Go2/Jetson 上以固定 cmd `[0.15,0,0]` + D435i depth 验证可行走；
这不是当前配置默认值。仓库当前 `config.yaml` 使用 `command_source=uwb`，并把
`fixed_cmd=[0.7,0,0]` 作为切换到 fixed 模式时的回退值，二者不得混写为同一次
已验证工况。

## 与当前 Standard 训练包的边界

`behavior_distill_v2`、`privileged_loco_teacher_v1` 和
`kaiwu_train_v1`（包括 `daggerfull-16288`、`visionfull-28401`、P1.5 `response*`、
P2 `navwarm`/`navadapt`/`navfull` 以及 `standard_visual_ppo` 的
`rlcritic`/`rlactor`/`rlfull` 文件）都是训练/恢复制品，
`capabilities.deployable=false`。它们可能包含 `height_scan256`、optimizer、
冻结教师或其他真机不可提供的状态，当前 `export_loco_onnx.py` 不接受这些格式。

P2 的 `modules.high_level.component_status="complete"` 只表示训练包包含完整动作型高层，
不表示已有部署 runtime。其输入还要求 `goal4`、`nav_nonvisual36`、
`response_profile16`、`adapter_confidence1` 与两组 recurrent state；当前 command-v2 高层输出
三轴 `[vx,vy,wz]`，旧二维 evaluator/exporter 不得静默补 `vy=0`。在独立完成导出、
接口审查和真机验证前，部署端必须拒绝直接加载或通过改名伪装。

默认部署路线仍只接受经过单独导出审查的 `format="lbc_loco"` 视觉策略候选；
不得通过改名把上述训练包伪装成可部署 checkpoint。本次 R2 合入不改变 ONNX
输入、关节顺序、动作缩放、控制频率或 Jetson 运行代码。

P4 训练 checkpoint 不是本部署树的可部署候选：尚未完成与此 Actor80 `goal[1,4]`
输入兼容的导出审查、ONNX 数值校验和 Jetson/真机验证前，必须保持
`capabilities.deployable=false`，不得作为本路线的 `policy.onnx` 或通过改名加载。

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
- **UWB goal 编码**: 当前默认部署入口和现有 Actor80 继续使用
  `ActorGoalEncoding::LegacyActor80`，即 `[clamp(x/10),clamp(y/10),min(d/20,1),0]`。
  `DirectionPreservingV2` 只为未来与 P4 v2 合同匹配的可部署制品显式启用；`d>10m` 时其前两维
  使用单位方位 `[x/d,y/d]`。当前 `p4full8h-r2` bundle 标记为 non-deployable，不能让训练期合同
  静默改变稳定低层运行时。UWB 新鲜度衰减与保持超时不由编码版本改变。
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
