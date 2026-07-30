#!/usr/bin/env bash
# =============================================================================
# launch_d435i.sh — 启动 D435i ROS 节点, 发布 /depth_camera/depth/image_raw 等 topic
#
# 启动后这些 topic 会发布:
#   /depth_camera/depth/image_rect_raw   (深度图, 480x270, Z16 编码 → sensor_msgs/Image)
#   /depth_camera/depth/color/camera_info  (内参)
#   /depth_camera/color/image_raw         (彩色图, 对齐参考用)
#   /depth_camera/depth/points            ★ PointCloud2 (XYZ, RViz 直接显示)
#
# 参数和 st9 部署完全对齐 (480x270@30, depth_scale=1mm)
# =============================================================================
set -uo pipefail

# ROS 环境
source /opt/ros/foxy/setup.bash

# 检查包是否装了
if ! ros2 pkg list 2>/dev/null | grep -q "^realsense2_camera$"; then
    echo "ERROR: realsense2_camera 包没装" >&2
    echo "请先跑: sudo apt install -y ros-foxy-realsense2-camera ros-foxy-realsense2-camera-msgs" >&2
    exit 1
fi

# 检查相机
if ! rs-enumerate-devices --compact 2>&1 | grep -q "D435I"; then
    echo "ERROR: 没检测到 D435I 相机" >&2
    exit 1
fi

echo "[启动] D435I ROS 节点..."
echo "[topic] /depth_camera/depth/points (PointCloud2, RViz 显示这个)"
echo ""

# 480x270@30, 与 st9 部署 _pick_profile 选的 profile 完全一致
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=depth_camera \
    depth_module.width:=480 \
    depth_module.height:=270 \
    depth_module.fps:=30 \
    depth_module.enable_auto_exposure:=true \
    pointcloud.enable:=true \
    pointcloud.order_z:=false \
    pointcloud.allow_negative_z:=false \
    align_depth.enable:=true \
    enable_color:=true \
    enable_depth:=true \
    rgb_camera.width:=640 \
    rgb_camera.height:=480 \
    rgb_camera.fps:=30
