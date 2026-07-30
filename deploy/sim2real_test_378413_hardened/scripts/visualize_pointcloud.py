#!/usr/bin/env python3
"""
点云可视化 (matplotlib 3D) — 不依赖 ROS, 直接读 D435i 实时显示。

两种模式:
  --mode raw       原始点云 (真机内参反投影)
  --mode sim       模型实际看到的点云 (重投影到 sim 针孔 320x180, 与 CNN 输入一致)
                   ★ 这个模式让你看到"模型视角"的点云, 验证 spatial 滤波效果

颜色: 按深度值着色 (红=近, 蓝=远)

操作:
  鼠标拖动 = 旋转视角
  鼠标滚轮 = 缩放
  按 q 或关窗口 = 退出
  每 0.5 秒自动刷新一帧

用法:
  cd ~/Kaiwu-test/sim2real_test_378413_hardened
  python3 scripts/visualize_pointcloud.py --mode sim
  python3 scripts/visualize_pointcloud.py --mode raw
  python3 scripts/visualize_pointcloud.py --mode sim --spatial   # 同时开 spatial 滤波对比
"""
import argparse
import sys
import time

import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ── sim 内参 (与 DepthSource.h::SIM_FX/FY/CX/CY 完全一致) ──
SIM_FX, SIM_FY = 168.61, 168.61
SIM_CX, SIM_CY = 160.0, 90.0
SIM_W, SIM_H   = 320, 180


def start_camera():
    """启动 D435i, 用 _pick_profile 同款逻辑选 480x270@30."""
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        sys.exit("[ERR] 没找到 RealSense 设备")

    profiles = []
    for s in devs[0].query_sensors():
        for p in s.get_stream_profiles():
            if p.stream_type() == rs.stream.depth and p.format() == rs.format.z16:
                vp = p.as_video_stream_profile()
                profiles.append((vp.width(), vp.height(), p.fps(), p))

    if not profiles:
        sys.exit("[ERR] 没有可用的 z16 stream")

    # 优先 480x270@30, 否则最接近 424x240@30
    want = (424, 240, 30)
    best = min(profiles, key=lambda x: abs(x[0]-want[0]) + abs(x[1]-want[1]) + abs(x[2]-30)*10)
    w, h, fps, profile = best
    print(f"[camera] 用 profile {w}x{h}@{fps}")

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    pipe.start(cfg)
    return pipe, w, h


def get_intrinsics(pipe):
    """取真机内参 (rs.intrinsics)."""
    active = pipe.get_active_profile()
    vs = active.get_stream(rs.stream.depth).as_video_stream_profile()
    return vs.get_intrinsics()


def deproject_raw(depth_m, intrin, max_m=5.0, step=2):
    """原始点云: 用真机内参反投影, 降采样 (step)."""
    h, w = depth_m.shape
    pts = []
    cols = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            d = depth_m[y, x]
            if d <= 0 or d >= max_m or not np.isfinite(d):
                continue
            # rs2_deproject_pixel_to_point 返回 [X, Y, Z] (相机坐标系, 米)
            coord = rs.rs2_deproject_pixel_to_point(intrin, [x, y], d)
            pts.append(coord)
            cols.append(d)
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.array(pts), np.array(cols)


def reproject_to_sim(depth_m, real_intrin, max_m=5.0):
    """复刻 DepthSource.h::reproject_normalize_into 的重投影 (但输出米, 不归一化).

    对每个 sim 像素, 用 sim 针孔模型算射线, 投影到真机像素采样深度值.
    """
    H, W = SIM_H, SIM_W
    out_depth = np.zeros((H, W), dtype=np.float32)
    real_W = real_intrin.width
    real_H = real_intrin.height

    for v in range(H):
        yn = (v - SIM_CY) / SIM_FY
        for u in range(W):
            xn = (u - SIM_CX) / SIM_FX
            ur = int(round(real_intrin.fx * xn + real_intrin.ppx))
            vr = int(round(real_intrin.fy * yn + real_intrin.ppy))
            if 0 <= ur < real_W and 0 <= vr < real_H:
                d = depth_m[vr, ur]
                if np.isfinite(d) and 0 < d < max_m:
                    out_depth[v, u] = d
    return out_depth


def deproject_sim(depth_sim_m, max_m=5.0, step=2):
    """sim 点云: 用 sim 内参反投影 (与 CNN 看到的完全一致)."""
    pts = []
    cols = []
    for y in range(0, SIM_H, step):
        for x in range(0, SIM_W, step):
            d = depth_sim_m[y, x]
            if d <= 0 or d >= max_m or not np.isfinite(d):
                continue
            xn = (x - SIM_CX) / SIM_FX
            yn = (y - SIM_CY) / SIM_FY
            X = xn * d
            Y = yn * d
            pts.append([X, Y, d])
            cols.append(d)
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.array(pts), np.array(cols)


def main():
    ap = argparse.ArgumentParser(description="D435i 点云可视化 (matplotlib 3D)")
    ap.add_argument("--mode", choices=["raw", "sim"], default="sim",
                    help="raw=原始相机视角, sim=模型视角 (重投影到 320x180, 默认)")
    ap.add_argument("--spatial", action="store_true",
                    help="对 raw 模式开启 spatial 滤波, 对比看滤波前后效果")
    ap.add_argument("--max-depth", type=float, default=5.0)
    ap.add_argument("--step", type=int, default=2, help="点云降采样步长 (越小越密)")
    ap.add_argument("--every", type=float, default=0.5, help="刷新间隔秒")
    args = ap.parse_args()

    pipe, cw, ch = start_camera()
    intrin = get_intrinsics(pipe)
    print(f"[intrinsics] fx={intrin.fx:.1f} fy={intrin.fy:.1f} "
          f"cx={intrin.ppx:.1f} cy={intrin.ppy:.1f}  {intrin.width}x{intrin.height}")
    print(f"[intrinsics] FOV={np.degrees(2*np.arctan(intrin.width/(2*intrin.fx))):.1f}x"
          f"{np.degrees(2*np.arctan(intrin.height/(2*intrin.fy))):.1f}°")
    print(f"[mode] {args.mode} (spatial={'ON' if args.spatial else 'OFF'})")
    print(f"[hint] 鼠标拖动旋转, 滚轮缩放, 按 q 退出")
    print("-" * 60)

    spatial = None
    if args.spatial:
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        spatial.set_option(rs.option.holes_fill, 0)

    # 初始化 figure
    plt.ion()
    fig = plt.figure(figsize=(11, 8))
    title_mode = f"{args.mode}" + (" + spatial" if args.spatial and args.mode == "raw" else "")
    fig.suptitle(f"D435i 点云  [{title_mode}]  (红=近, 蓝=远, 单位 m)", fontsize=13)
    ax = fig.add_subplot(111, projection='3d')
    sc = None

    last_update = 0
    frame = 0
    try:
        while True:
            now = time.time()
            if now - last_update < args.every:
                # 不刷新点云, 但让 GUI 响应事件
                plt.pause(0.05)
                continue

            frames = pipe.wait_for_frames(1000)
            depth = frames.get_depth_frame()
            if not depth:
                continue
            if spatial is not None and args.mode == "raw":
                depth = spatial.process(depth)

            w, h = depth.get_width(), depth.get_height()
            raw_arr = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(h, w).astype(np.float32)
            depth_m = raw_arr * 0.001  # depth_scale (D435i 默认 1mm)

            if args.mode == "raw":
                pts, cols = deproject_raw(depth_m, intrin, args.max_depth, args.step)
            else:
                depth_sim_m = reproject_to_sim(depth_m, intrin, args.max_depth)
                pts, cols = deproject_sim(depth_sim_m, args.max_depth, args.step)

            n_pts = len(pts)
            if n_pts == 0:
                print(f"\r[f{frame:4d}] 无有效点云", end="", flush=True)
                plt.pause(0.05)
                continue

            # 重画
            if sc is not None:
                sc.remove()
            # 翻转轴: 相机坐标系 X 右, Y 下, Z 前 → 显示 X 右, Y 前, Z 上 (直观)
            X = pts[:, 0]
            Y = pts[:, 2]   # 深度
            Z = -pts[:, 1]  # 翻转上下

            sc = ax.scatter(X, Y, Z, c=cols, cmap='jet', s=2, alpha=0.7,
                            vmin=0, vmax=args.max_depth)

            ax.set_xlim(-args.max_depth*0.6, args.max_depth*0.6)
            ax.set_ylim(0, args.max_depth)
            ax.set_zlim(-1.5, 1.5)
            ax.set_xlabel('X (左右, m)')
            ax.set_ylabel('Z (深度, m)')
            ax.set_zlabel('Y (上下, m, 翻转)')
            ax.view_init(elev=20, azim=-90)

            # 颜色条只在第一帧画
            if frame == 0:
                cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
                cb.set_label('深度 (m, 红近蓝远)')

            # 统计
            near = (cols < 1.0).sum()
            mid  = ((cols >= 1.0) & (cols < 2.0)).sum()
            far  = (cols >= 2.0).sum()
            inval_pct = 100 * (1 - n_pts / ((SIM_H*SIM_W/args.step**2 if args.mode=='sim'
                                              else h*w/args.step**2)))

            print(f"\r[f{frame:4d}] {n_pts:5d} 点 | "
                  f"<1m:{near}  1-2m:{mid}  >2m:{far}  | "
                  f"min={cols.min():.2f}m max={cols.max():.2f}m | "
                  f"无效≈{inval_pct:.0f}%",
                  end="", flush=True)

            fig.canvas.draw_idle()
            plt.pause(0.001)
            last_update = now
            frame += 1

    except KeyboardInterrupt:
        print("\n[退出]")
    finally:
        try: pipe.stop()
        except: pass
        plt.close('all')


if __name__ == "__main__":
    main()
