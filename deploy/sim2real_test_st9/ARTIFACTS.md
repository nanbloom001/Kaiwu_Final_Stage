# ARTIFACTS - sim2real_test_st9

## 概述
Go2 lbc_loco 学生策略部署树（st9 训练分布对齐 + UWB goal-driven 避障测试）。目标 checkpoint `model.ckpt-vision-378413.pkl` 与 loco 树相同，但使用不同的 UWB 命令参数和深度后处理滤波（A/B/C 对照已选出 light_spatial 配置）。

## 所需制品清单
| 制品 | 类型 | 当前状态 | SHA256 | 说明 |
|---|---|---|---|---|
| model.ckpt-vision-378413.pkl | checkpoint | ✅已入库 | 37429c1e2c1d263844a74ecc2fdb97201ae296663f5ad8890ce173d5525a1288 | 同源 Sim2Real vision 学生 |
| policy.onnx | ONNX | ❌待生成 | - | `exported/` 目录不存在；需从 ckpt 导出 |
| go2_loco_ctrl | C++ binary | ❌待构建 | - | 未构建/未提交 |
| ONNX Runtime | runtime dep | ❌待部署 | - | thirdparty 未入库 |
| deploy.yaml | config | ✅已入库 | 同源 | 与 loco 树完全一致（同 ckpt） |
| config.yaml | config | ✅已入库 | 同源 | **与 loco 树不同**：`fixed_cmd=[0.30,0,0]`（训练分布内），depth.filters.mode=light_spatial，cmd_vx_mode=goal_08，goal_driven=false，yaw_kp=0.80 |
| depth_viz ROS 节点 | ROS2 工具 | ✅已入库 | - | `ros/depth_viz/` 下 D435i PointCloud2 可视化节点，非部署必需 |

## ONNX I/O 契约
与 loco 树 **完全一致**（同一份 `export_loco_onnx.py`）：
- **输入**(8): `depth[1,180,320,1]` `proprio[1,45]` `goal[1,4]` `loco_h[2,1,64]` `loco_c[2,1,64]` `nav_h[2,1,64]` `nav_c[2,1,64]` `cmd_override[1,4]`
- **输出**(8): `cmd[1,3]` `cmd_raw[1,3]` `clearance[1,3]`（恒 0）`joint[1,12]` `loco_h_out[2,1,64]` `loco_c_out[2,1,64]` `nav_h_out[2,1,64]` `nav_c_out[2,1,64]`
- **LSTM 语义**: loco LSTM (h,c) 逐帧回喂 + 重置；nav 状态透传
- **proprio[6:9] 覆写**: C++ runner 负责（s9 部署用 `_uwb_command` 注入，与训练分布对齐）
- **opset=17**, 固定 batch=1

## 生成命令
与 loco 树相同：
```bash
python export_loco_onnx.py \
  --ckpt models/model.ckpt-vision-378413.pkl \
  --out  runtime/unitree_rl_lab_test/logs/loco/exported/policy.onnx
```

## Jetson 安装位置
`/home/unitree/Kaiwu-test/sim2real_test_st9/`（与 loco 树平级，互不干扰）。安装结构同 loco 树。特有内容：
- `scripts/run_depth_filter_abx.sh`：A/B/C 深度滤波对照测试脚本
- `ros/depth_viz/`：ROS2 D435i 可视化（非部署必需，仅供调试）

## LOCO_TEST_ROOT 使用方法
默认值 `/home/unitree/Kaiwu-test/sim2real_test_st9`。与 loco 树唯一差异是默认路径。st9 特有脚本 `run_depth_filter_abx.sh` 也通过 `LOCO_TEST_ROOT` 定位 config 和 runner。注意：st9 的 `config.yaml` 中 `command_source` 默认 `uwb`，且 `cmd_vx_mode=goal_08`（C++ 端关闭转向，匹配训练分布：proprio[6:9]=[0.8,0,0] 恒定）。
