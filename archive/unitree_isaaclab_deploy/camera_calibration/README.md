# D435i 手眼标定工具

在 Go2 Jetson Nano 上标定 RealSense D435i 相对于 Go2 前置摄像头的位姿，生成含相机的 URDF。摄像机物理位置改变后必须重新走标定流程。

## 坐标系约定

```
Optical frame (相机标定输出):    z-forward, x-right, y-down
URDF/Robot frame (URDF 使用):   x-forward, y-left, z-up

转换关系:
  urdf_x = optical_z
  urdf_y = -optical_x
  urdf_z = -optical_y

Optical frame 旋转到 URDF frame: rpy = [-pi/2, 0, -pi/2]
```

标定结果 (`calibration_result.json`) 保存为 **optical frame** 原始数据。
URDF 生成时自动转换到 **URDF frame**。

## 环境要求

标定在 Go2 的 **Jetson Nano** (192.168.123.18) 上执行：

- Python 3.8+
- OpenCV (`cv2`)
- NumPy, SciPy
- pyrealsense2
- unitree_sdk2py (Go2 前置摄像头通过 SDK 获取。 https://github.com/unitreerobotics/unitree_sdk2_python.git) 

```bash
pip3 install pyrealsense2
pip3 install -e /home/unitree/unitree_sdk2_python
```

## 标定原理

通过同时在 Go2 前置摄像头和 D435i RGB 中检测同一块棋盘格标定板：

1. 两台相机各自通过 PnP 求解标定板在各自相机坐标系（optical frame）中的 pose
2. 计算 D435i RGB 相对于 Go2 相机的变换: `T_rel = T_board_go2 @ inv(T_board_d435i)`
3. 多组测量取四元数平均，提高鲁棒性
4. 生成 URDF 时将结果从 optical frame 转换到 URDF frame
5. D435i depth optical frame 使用固件出厂标定的 color-to-depth extrinsics

D435i 最终挂载到 `front_camera` link 下。

## 使用方法

```bash
ssh unitree@192.168.123.18  # 密码: 123
cd ~/camera_calibration
```

### Step 1: 采集图像

```bash
python3 calibrate_d435i.py capture --num_captures 15 --network eth0
```

- 将标定板放在两台相机都能看到的位置
- 终端显示 `Go2:OK  D435i:OK` 后按 Enter 采集
- 每次采集后移动标定板位置/角度，建议采集 15+ 组
- 采集时自动保存 D435i 的 depth-to-color extrinsics

### Step 2: 运行标定

```bash
python3 calibrate_d435i.py calibrate
```

输出 D435i RGB 相对于 Go2 前置摄像头的位姿 (optical frame)。

### Step 3: (可选) 单独获取 depth extrinsics

```bash
python3 calibrate_d435i.py query_extrinsics
```

从 D435i 固件读取 color-to-depth 出厂标定数据，保存到 `d435i_extrinsics.json`。

### Step 4: 生成 URDF

```bash
python3 calibrate_d435i.py generate_urdf --base_urdf ~/camera_calibration/go2_description.urdf
```

输出 `~/calibration_data/go2_with_d435i.urdf`，包含：
- `d435i_link`: RGB 摄像头位置（含 visual/collision/inertial）
- `d435i_depth_optical_frame`: 深度传感器 optical frame

### Step 5: 提取内参

```bash
python3 get_base_pos_d435i.py
```

将输出的xyz, rpy参数，填入训练代码对应的train_env_conf_xxx.toml中


# 附录

## D435i 物理参数

来源：[IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros) `_d435.urdf.xacro`

| 参数 | 值 | 说明 |
|------|-----|------|
| 尺寸 (WxHxD) | 90 x 25 x 25.05 mm | 铝合金外壳 |
| 质量 | 72g | |
| depth_to_color | ~15mm | depth 在 RGB 左侧约15mm |

Depth optical frame 的 z 轴朝前（和相机观察方向一致），这是 optical frame 的标准约定。

## 标定板参数

```python
CHESSBOARD_SIZE = (9, 6)   # 内角点数 (列, 行)
SQUARE_SIZE_M = 0.025      # 方格边长 25mm
```


## 输出文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 标定结果 | `~/calibration_data/calibration_result.json` | RGB optical frame 原始数据 |
| Depth extrinsics | `~/calibration_data/d435i_extrinsics.json` | D435i 固件 color-to-depth 数据 |
| URDF | `~/calibration_data/go2_with_d435i.urdf` | URDF frame，d435i 挂载到 front_camera |
| D435i 内参 | `~/calibration_data/K_d435i.npy` | D435i RGB 内参矩阵 |
| 采集图像 | `~/calibration_data/go2_front/` `~/calibration_data/d435i_rgb/` | 标定用图像对 |

## 标定结果参考 (2026-03-04)

**RGB 标定** (d435i_link 相对于 front_camera):
```
URDF frame:
  xyz: [0.0277, 0.0323, 0.0359]   # 前28mm, 左32mm, 上36mm
  rpy: [-1.70, 21.05, 0.62] deg   # pitch约21度下倾
```

**Depth frame** (d435i_depth_optical_frame 相对于 d435i_link):
```
URDF frame:
  xyz: [-0.0003, -0.0149, 0]      # 右侧约15mm (从用户视角是左侧)
  rpy: [-1.57, -0.01, -1.58]      # 含固件~0.3度校正
```

D435i RGB 摄像头物理位置偏右约 32mm（相对于 Go2 前置摄像头），这是 D435i 安装位置导致的偏移。


##  D435i 内参 → PinholeCameraCfg 映射

###  FOV 计算

D435i Depth 模组参数 (来源: Intel 官方 datasheet):

| 参数 | 值 |
|------|-----|
| HFOV | ~87 deg |
| VFOV | ~58 deg |
| 原生分辨率 | 848 x 480 / 640 x 480 |
| 有效范围 | 0.1 ~ 5.0 m (室内实测) |

Isaac Sim `PinholeCameraCfg` 通过 `focal_length` 和 `horizontal_aperture` 控制 FOV：

```
HFOV = 2 * atan(horizontal_aperture / (2 * focal_length))
```

反推：给定 HFOV = 87 deg：

```python
import math
hfov_rad = math.radians(87)
focal_length = 1.88  # mm (自由选择)
horizontal_aperture = 2 * focal_length * math.tan(hfov_rad / 2)
# = 2 * 1.88 * 0.9489 = 3.568 mm
```

VFOV 由分辨率纵横比自动确定：

```
VFOV = 2 * atan(height/width * horizontal_aperture / (2 * focal_length))
     = 2 * atan(180/320 * 3.568 / (2 * 1.88))
     ≈ 55.6 deg  (接近 D435i 实际 58 deg，误差因降采样比例引起)
```

### 分辨率选择

| 分辨率 | 像素数 | 用途 |
|--------|-------|------|
| 848 x 480 | 407,040 | D435i 原生，GPU 显存不足以批量渲染 |
| 424 x 240 | 101,760 | 2x 降采样，仍然偏大 |
| **320 x 180** | **57,600** | 训练推荐，CNN 输入合理大小 |
| 160 x 90 | 14,400 | 超低分辨率，丢失太多细节 |

320x180 是训练效率和视觉信息的平衡点。64 envs 时 GPU 显存占用约 2-3 GB。

### clipping_range

```python
clipping_range=(0.1, 5.0)  # meters
```

- `0.1m`: D435i 最近有效深度 (更近的返回 0/NaN)
- `5.0m`: 室内实测可靠距离上限 (D435i 标称 10m，但 >5m 噪声剧增)

超出范围的像素由 `depth_clipping_behavior="max"` 处理，在 observation 函数中转为 0 (模拟真实 D435i 的 hole)。

## 坐标系对照

```
                    Isaac Sim (USD)         URDF            D435i Optical
Forward:              +X                    +X                +Z
Left:                 +Y                    +Y                -X
Up:                   +Z                    +Z                -Y

TiledCamera 默认朝向: -Z (USD convention)
通过 OffsetCfg.rot 旋转到实际朝向 (前下方约 21 deg)
```

`convention="world"` 让 Isaac Sim 在 URDF/world frame 中解释 offset，
然后自动处理 USD 内部的坐标变换。不需要手动做 optical frame 转换。

##  热重载支持

**Isaac Sim 不支持 sensor 配置的热重载。** 修改以下参数后必须重启仿真：

| 参数 | 需要重启 | 说明 |
|------|---------|------|
| `prim_path` | 是 | Prim 在场景加载时创建 |
| `offset` (pos/rot) | 是 | Prim transform 在初始化时设置 |
| `spawn` (focal_length 等) | 是 | Camera prim 属性在 spawn 时写入 |
| `width`, `height` | 是 | 渲染 buffer 在初始化时分配 |
| `data_types` | 是 | 数据管线在初始化时构建 |
| `update_period` | **否** | 可在 `__post_init__` 中动态设置 |
| `debug_vis` | **否** | 可在 play 时切换 |

## 常见问题

### Q: 深度图全黑/全白？
- 全黑：相机朝向错误（可能指向天空或地面外）。检查 `rot` quaternion 格式是否为 wxyz。
- 全白：所有像素超出 `clipping_range`。检查 `pos` 是否合理（相机是否在地下或太高）。
- 验证：`--num_envs 1 --enable_cameras` 启动后在 GUI 中查看 Camera 视图。

### Q: 深度图分辨率和实际 D435i 不一致？
- 训练故意降采样到 320x180，在 `depth_camera_image` observation 函数中展平为 57600-dim 向量。
- CNN 编码器会进一步压缩。低分辨率减少 GPU 显存和渲染开销。
- 如需更改，修改 `TiledCameraCfg` 的 `width`/`height`，但要同步调整 CNN 输入维度。

### Q: 仿真中 D435i 的噪声如何模拟？
- Isaac Sim 渲染的深度图是理想值（无噪声）。
- Sim2real augmentation 在 `depth_camera_image` observation 函数中实现：
  - `pixel_dropout_prob=0.1`: 随机 10% 像素归零 (模拟 D435i 的 hole)
  - `gaussian_noise_scale=0.02`: 深度成比例高斯噪声 (远处更嘈杂)
  - `roll_range_deg=10.0`: 随机图像旋转 (模拟行走时相机抖动)
  - `pitch_shift_pix=10`: 随机垂直平移 (模拟俯仰振荡)

### Q: 能否直接用含 D435i 的 URDF 加载？
- 理论上可以。但 Isaac Sim 从 URDF 导入的 camera link 只是刚体，不会自动创建渲染传感器。
- 仍需手动配置 `TiledCameraCfg`。将 `prim_path` 指向 URDF 导入的 `d435i_link` prim 并不能省去内参配置。
- 当前方案（基础 URDF + 独立 sensor）更清晰，避免 URDF 导入的不确定性。
