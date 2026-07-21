#!/usr/bin/env bash
# =============================================================================
# show_pointcloud.sh — 一键启动 D435i Python 节点 + RViz2 显示点云
#
# 为什么用 Python 节点而不是 ros-foxy-realsense2_camera:
#   装 ROS 版会拉 ros-foxy-librealsense2, 与系统 librealsense2 2.54.2 冲突,
#   会破坏 st9 部署在用的 SDK. 改用 pyrealsense2 直接发 PointCloud2, 零冲突.
#
# 发布的 topic:
#   /depth_camera/depth/points       sensor_msgs/PointCloud2 (XYZRGB)
#   /depth_camera/depth/image_raw    sensor_msgs/Image (mono16)
#   /depth_camera/color/image_raw    sensor_msgs/Image (rgb8)
#   /depth_camera/camera_info        sensor_msgs/CameraInfo
#
# 必须在 Jetson 本机终端跑 (DISPLAY=:1), 不能纯 SSH.
# =============================================================================
set -uo pipefail

# 必须有 DISPLAY
if [[ -z "${DISPLAY:-}" ]]; then
    echo "ERROR: DISPLAY 没设, 这是 GUI 程序" >&2
    echo "请在 Jetson 本机终端跑 (GNOME 桌面 Terminal), 不是 SSH" >&2
    echo "如果用 SSH 加 X 转发: ssh -X unitree@192.168.123.18 (会卡)" >&2
    exit 1
fi

source /opt/ros/foxy/setup.bash

VIZ_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 启动 Python 节点 (后台)
echo "[1/2] 启动 D435i Python 节点 (pyrealsense2 → ROS2 PointCloud2)..."
python3 "${VIZ_ROOT}/d435i_pointcloud_node.py" &
NODE_PID=$!

# 退出时清理
cleanup() {
    echo ""
    echo "[cleanup] 关闭节点和 RViz..."
    kill ${NODE_PID} 2>/dev/null
    wait ${NODE_PID} 2>/dev/null
    echo "[done]"
}
trap cleanup EXIT INT TERM

# 等 topic 就绪
echo "[2/2] 等待 ROS topic 就绪..."
sleep 4
TOPICS=$(ros2 topic list 2>/dev/null | grep -c "depth_camera")
if [[ ${TOPICS} -lt 2 ]]; then
    echo "WARN: depth_camera topic 还没就绪, 继续启动 RViz..."
    echo "  (在 RViz 里手动加 Display → PointCloud2 → topic /depth_camera/depth/points)"
else
    echo "✓ topic 就绪 (${TOPICS} 个 depth_camera/* topic)"
fi

echo ""
echo "=================================================================="
echo "  启动 RViz2 (加载 ${VIZ_ROOT}/rviz/depth_viz.rviz)"
echo "  鼠标拖动旋转 / 滚轮缩放 / 关窗口退出"
echo "=================================================================="
echo ""

rviz2 -d "${VIZ_ROOT}/rviz/depth_viz.rviz"
