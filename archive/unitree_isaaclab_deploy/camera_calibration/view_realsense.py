import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time


def main():
    save_dir = "realsense_capture"
    os.makedirs(save_dir, exist_ok=True)

    # 创建 RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用彩色流：640x480, 30 FPS
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 启用深度流：640x480, 30 FPS
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        # 启动相机
        print("Starting RealSense camera...")
        profile = pipeline.start(config)

        # 获取深度比例，单位通常是 meter
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth scale:", depth_scale)

        # 等待几帧，让自动曝光稳定
        print("Warming up camera...")
        for _ in range(30):
            pipeline.wait_for_frames()

        # 捕获一帧
        print("Capturing one frame...")
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Failed to get color or depth frame.")
            return

        # 转为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 保存彩色图像
        color_path = os.path.join(save_dir, "color.png")
        cv2.imwrite(color_path, color_image)

        # 保存原始深度图，16-bit PNG，单位需要乘 depth_scale 才是米
        depth_raw_path = os.path.join(save_dir, "depth_raw.png")
        cv2.imwrite(depth_raw_path, depth_image)

        # 保存可视化深度图
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        depth_vis_path = os.path.join(save_dir, "depth_vis.png")
        cv2.imwrite(depth_vis_path, depth_colormap)

        print("Saved files:")
        print("  RGB image:        ", color_path)
        print("  Raw depth image:  ", depth_raw_path)
        print("  Depth visualization:", depth_vis_path)

        # 打印中心点深度
        h, w = depth_image.shape
        center_x = w // 2
        center_y = h // 2
        center_depth_raw = depth_image[center_y, center_x]
        center_depth_meter = center_depth_raw * depth_scale

        print(f"Center depth raw value: {center_depth_raw}")
        print(f"Center depth in meters: {center_depth_meter:.4f} m")

    except RuntimeError as e:
        print("RuntimeError:", e)
        print("Please check whether the RealSense camera is connected and not occupied.")

    finally:
        pipeline.stop()
        print("Camera stopped.")


if __name__ == "__main__":
    main()