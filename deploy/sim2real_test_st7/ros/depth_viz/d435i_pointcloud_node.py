#!/usr/bin/env python3
"""
d435i_pointcloud_node.py — 用 pyrealsense2 直接读 D435i, 发布 ROS2 PointCloud2

为什么不直接用 ros-foxy-realsense2_camera:
  - 装 ROS 版会要求装 ros-foxy-librealsense2, 与系统已装的 librealsense2 2.54.2 冲突
  - 冲突会破坏 st9 部署在用的 SDK
  - 用 pyrealsense2 (和部署同一套 SDK) 直接发布 PointCloud2, 零冲突, 零依赖

发布的 topic:
  /depth_camera/depth/points   sensor_msgs/PointCloud2  (XYZ only, 按强度/距离着色)
  /depth_camera/depth/image_raw  sensor_msgs/Image      (深度图, RViz 用 ColorFu)
  /depth_camera/camera_info     sensor_msgs/CameraInfo  (内参)

参数和 st9 部署完全对齐: 480x270@30, depth_scale=1mm, 内参 per-unit.
"""
import struct
import sys
import time

import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import Header


# PointField dtype 常量
PFTYPE_FLOAT32 = 7
PFTYPE_UINT16 = 2
PFTYPE_UINT8 = 1


def pick_profile(dev, want=(424, 240, 30)):
    """复刻 DepthSource.h::_pick_profile 的选 profile 逻辑."""
    profs = []
    for s in dev.query_sensors():
        for p in s.get_stream_profiles():
            if p.stream_type() == rs.stream.depth and p.format() == rs.format.z16:
                vp = p.as_video_stream_profile()
                profs.append((vp.width(), vp.height(), p.fps(), p))
    if not profs:
        sys.exit("[ERR] 没找到 z16 stream")
    # 优先精确, 否则最接近 want
    for w, h, f, p in profs:
        if (w, h, f) == want:
            return p
    return min(profs, key=lambda x: abs(x[0]-want[0]) + abs(x[1]-want[1]) + abs(x[2]-30)*10)[3]


class D435iPublisher(Node):
    def __init__(self):
        super().__init__('d435i_publisher')

        # ---- 启动 RealSense ----
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) == 0:
            self.get_logger().error("没找到 RealSense 设备")
            sys.exit(1)
        dev = devs[0]

        depth_profile = pick_profile(dev)
        w = depth_profile.as_video_stream_profile().width()
        h = depth_profile.as_video_stream_profile().height()
        fps = depth_profile.fps()

        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        # 不开 color stream — D435I 同时开 depth+color 时偶发 bad_alloc (USB 带宽/内存),
        # 而点云可视化只需要 depth. 如果需要看彩色图, 单独跑 rs-viewer.
        profile = self.pipe.start(cfg)
        self.get_logger().info(f"RealSense 启动: depth {w}x{h}@{fps}")

        # 内参 + scale
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.intrin = depth_stream.get_intrinsics()
        dev = profile.get_device()
        self.depth_scale = dev.first_depth_sensor().get_depth_scale()
        self.get_logger().info(
            f"内参 fx={self.intrin.fx:.1f} fy={self.intrin.fy:.1f} "
            f"cx={self.intrin.ppx:.1f} cy={self.intrin.ppy:.1f} "
            f"depth_scale={self.depth_scale*1000:.2f}mm"
        )
        self.get_logger().info(
            f"FOV={np.degrees(2*np.arctan(self.intrin.width/(2*self.intrin.fx))):.1f}x"
            f"{np.degrees(2*np.arctan(self.intrin.height/(2*self.intrin.fy))):.1f}°"
        )

        # 预计算反投影网格 (每像素的归一化射线方向)
        xs = np.arange(w)
        ys = np.arange(h)
        gx, gy = np.meshgrid(xs, ys)
        self.ray_x = (gx - self.intrin.ppx) / self.intrin.fx  # [H, W]
        self.ray_y = (gy - self.intrin.ppy) / self.intrin.fy
        self.W = w
        self.H = h

        # ---- ROS2 发布器 ----
        # QoS: 用 Best Effort + Keep Last 5 (相机数据常用)
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos = QoSProfile(depth=5,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.pc_pub = self.create_publisher(PointCloud2, '/depth_camera/depth/points', qos)
        self.img_pub = self.create_publisher(Image, '/depth_camera/depth/image_raw', qos)
        self.info_pub = self.create_publisher(CameraInfo, '/depth_camera/camera_info', qos)

        # 发布 CameraInfo 一次
        self._publish_camera_info()

        # 定时器: 30 Hz 抓帧 + 发布
        self.frame_count = 0
        self.timer = self.create_timer(1.0 / 30.0, self.on_tick)
        self.get_logger().info("✓ 节点就绪, 30Hz 发布 /depth_camera/depth/points")
        self.get_logger().info("✓ RViz 里加 PointCloud2, topic 选 /depth_camera/depth/points")

    def _publish_camera_info(self):
        msg = CameraInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'depth_camera_depth_optical_frame'
        msg.height = self.intrin.height
        msg.width = self.intrin.width
        msg.distortion_model = 'plumb_bob'
        msg.d = list(self.intrin.coeffs[:5])
        msg.k = [self.intrin.fx, 0.0, self.intrin.ppx,
                 0.0, self.intrin.fy, self.intrin.ppy,
                 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [self.intrin.fx, 0.0, self.intrin.ppx, 0.0,
                 0.0, self.intrin.fy, self.intrin.ppy, 0.0,
                 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(msg)

    def on_tick(self):
        try:
            frames = self.pipe.wait_for_frames(timeout_ms=200)
        except Exception as e:
            self.get_logger().warn(f"wait_for_frames: {e}")
            return

        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return

        # depth_scale (m/raw uint16)
        depth_m = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            self.H, self.W).astype(np.float32) * self.depth_scale

        ts = self.get_clock().now().to_msg()
        frame_id = 'depth_camera_depth_optical_frame'

        # ---- 发布 PointCloud2 (XYZRGB, 颜色按距离 jet 着色) ----
        pc_msg = self._build_pointcloud2(depth_m, ts, frame_id)
        self.pc_pub.publish(pc_msg)

        # ---- 发布 depth image (raw uint16) ----
        img_msg = Image()
        img_msg.header.stamp = ts
        img_msg.header.frame_id = frame_id
        img_msg.height = self.H
        img_msg.width = self.W
        img_msg.encoding = 'mono16'
        img_msg.step = self.W * 2
        img_msg.data = bytes(depth_frame.get_data())
        self.img_pub.publish(img_msg)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            n_valid = (depth_m > 0).sum()
            pct = 100 * n_valid / depth_m.size
            self.get_logger().info(
                f"frame={self.frame_count} 有效点={n_valid} ({pct:.1f}%) "
                f"min={depth_m[depth_m>0].min():.2f}m max={depth_m[depth_m>0].max():.2f}m"
            )

    def _build_pointcloud2(self, depth_m, ts, frame_id):
        """深度 → XYZRGB PointCloud2 (跳过无效像素)."""
        valid = (depth_m > 0) & (depth_m < 5.0) & np.isfinite(depth_m)
        ys, xs = np.where(valid)
        ds = depth_m[ys, xs]

        # 反投影到 3D (相机坐标系: X 右, Y 下, Z 前)
        X = ((xs - self.intrin.ppx) / self.intrin.fx) * ds
        Y = ((ys - self.intrin.ppy) / self.intrin.fy) * ds
        Z = ds

        # 颜色: 按距离 jet 着色 (0m 红, 5m 蓝)
        rgb = self._jet_rgb(ds / 5.0)

        # 拼成结构数组 (x, y, z, rgb float-packed)
        points = np.zeros(len(X), dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('rgb', np.uint32),
        ])
        points['x'] = X
        points['y'] = Y
        points['z'] = Z
        # 把 RGB 编码成单个 float32 (RViz 约定: little-endian 0x00RRGGBB 当 float 解)
        rgb_packed = (rgb[:, 0].astype(np.uint32) << 16) | \
                     (rgb[:, 1].astype(np.uint32) << 8) | \
                     rgb[:, 2].astype(np.uint32)
        points['rgb'] = rgb_packed

        msg = PointCloud2()
        msg.header.stamp = ts
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PFTYPE_FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PFTYPE_FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PFTYPE_FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PFTYPE_FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16  # 4 floats
        msg.row_step = 16 * len(points)
        msg.is_dense = True
        msg.data = points.tobytes()
        return msg

    @staticmethod
    def _jet_rgb(t):
        """t in [0,1] → RGB uint8 (jet colormap), t<0→red, t>1→blue."""
        t = np.clip(t, 0, 1)
        r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
        return np.stack([r, g, b], axis=-1).astype(np.uint8) * 255

    def destroy_node(self):
        try:
            self.pipe.stop()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = D435iPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
