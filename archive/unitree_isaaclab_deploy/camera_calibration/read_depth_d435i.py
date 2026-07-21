#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
D435i 实时深度读取 + 核验脚本（sim2real 第一步）。

作用：在相机所在机器上读 D435i 深度流，逐帧打印统计，并产出模型要的
      320x180 归一化帧（与训练/sim 评测的 depth 完全一致的处理）。

为什么先做这个：sim2real 的输入端契约是「前向 D435i 深度 → clamp(d,0,5)/5 →
越界/无效→0 → reshape(180,320,1)」。先确认真机这一帧拿得到、数值合理（min≈0、
max≈1、mean 每帧有波动 = 活体传感器、无 NaN），整条推理链才有意义。
sim 探针实测：shape (N,180,320,1)、min≈0/max≈1、mean 0.196~0.213。真机应对齐到同一量级。

依赖：pyrealsense2（pip install pyrealsense2；Jetson 上一般随 librealsense 装好）。
      numpy。存伪彩 png / --show 需要 opencv-python。

用法：
    # 只打印统计（headless 安全），跑 200 帧
    python3 read_depth_d435i.py --frames 200

    # 存一帧给推理脚本干跑用：raw 米制 .npy + 模型用 320x180 归一化 .npy + 伪彩 png
    python3 read_depth_d435i.py --save-one ./depth_sample

    # 有显示器时实时看伪彩深度
    python3 read_depth_d435i.py --show
"""
from __future__ import annotations
import argparse
import threading
import time
import numpy as np

# ---- 模型侧深度契约（与 tools/base_env/observation_process.py 一致）----
MODEL_H, MODEL_W = 180, 320
MAX_DEPTH = 5.0          # clamp 上限，归一化 = clamp(d,0,5)/5
CLIP_NEAR = 0.1          # sim clipping_range 近端（仅供参考；归一化用 0~5）


# ============ 几何遮挡检测（与 vision_nav_infer_test.py 同口径）============
def region_occlusion(depth2d, near_thresh=0.30, row_lo=0.10, row_hi=0.55):
    """把深度切成 左/中/右 三竖条，只取与障碍同高的水平带(排除近地面/天花板)。
    每区返回：block(近+空洞占比,0..1)、near(有效近距占比)、hole(无效=0占比)、min_m(最近有效米)。
    near_thresh 是归一化阈值：0.30 ≈ 1.5m 以内算近(=MAX_DEPTH*0.30)。"""
    H, W = depth2d.shape
    r0, r1 = int(H * row_lo), int(H * row_hi)
    band = depth2d[r0:r1]
    w3 = W // 3
    regs = {"L": band[:, :w3], "F": band[:, w3:2 * w3], "R": band[:, 2 * w3:]}
    occ = {}
    for n, reg in regs.items():
        blocked = reg < near_thresh                 # 近(含无效0)=挡
        near_valid = (reg > 0.0) & (reg < near_thresh)
        hole = (reg == 0.0)
        valid = reg[reg > 0.0]
        occ[n] = {
            "block": float(blocked.mean()),
            "near":  float(near_valid.mean()),
            "hole":  float(hole.mean()),
            "min_m": float(valid.min() * MAX_DEPTH) if valid.size else float("nan"),
        }
    return occ


def occlusion_verdict(occ, block_ratio=0.15):
    """某区 block 占比 > block_ratio 即判该区被挡。"""
    return {n: occ[n]["block"] > block_ratio for n in ("L", "F", "R")}


# ============ 灰度 + L/F/R 遮挡标注渲染 ============
def render_overlay(model_in, occ, verdict, roi_lo, roi_hi, scale=3):
    """model_in:(180,320,1)∈[0,1] → 放大的灰度 BGR 图，叠加 ROI 带、三分竖线、
    每区 block%/min_m，挡住的区用红框高亮。需要 cv2。"""
    import cv2
    d = model_in.reshape(MODEL_H, MODEL_W)
    gray = (np.clip(d, 0, 1) * 255).astype(np.uint8)      # 0=近(黑) 1=远(白)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (MODEL_W * scale, MODEL_H * scale),
                     interpolation=cv2.INTER_NEAREST)
    H, W = img.shape[:2]
    w3 = W // 3
    r0, r1 = int(H * roi_lo), int(H * roi_hi)

    # ROI 水平带（黄色虚线框）
    cv2.rectangle(img, (0, r0), (W - 1, r1), (0, 255, 255), 1)
    # 三分竖线（青色）
    for x in (w3, 2 * w3):
        cv2.line(img, (x, 0), (x, H), (255, 255, 0), 1)

    names = ["L", "F", "R"]
    for i, n in enumerate(names):
        x0, x1 = i * w3, (i + 1) * w3
        blocked = verdict[n]
        color = (0, 0, 255) if blocked else (0, 200, 0)   # 挡=红 通=绿
        # 挡住的区在 ROI 带内画红框高亮
        if blocked:
            cv2.rectangle(img, (x0 + 2, r0 + 2), (x1 - 2, r1 - 2), (0, 0, 255), 2)
        mm = occ[n]["min_m"]
        mm_s = f"{mm:.2f}m" if mm == mm else "--"        # nan 检查
        cv2.putText(img, f"{n} {'BLK' if blocked else 'OK'}",
                    (x0 + 6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img, f"blk {occ[n]['block']*100:4.0f}%",
                    (x0 + 6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, f"min {mm_s}",
                    (x0 + 6, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


# ============ MJPEG 服务（headless 浏览器看图）============
class _MJPEGState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
    def update(self, jpeg_bytes):
        with self.lock:
            self.jpeg = jpeg_bytes
    def get(self):
        with self.lock:
            return self.jpeg


def start_mjpeg_server(port, state):
    """后台线程起一个 MJPEG HTTP 服务，访问 http://<ip>:<port>/ 看实时流。"""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = state.get()
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    print(f"[mjpeg] 浏览器打开 http://<本机IP>:{port}/ 看实时灰度+遮挡标注")
    return srv


def list_devices(rs):
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        print("[err] 没找到任何 RealSense 设备。")
        print("      → 确认 D435i USB 直连到**本机**（不是插在 Go2 .161 主控板上）。")
        print("      → 用 `rs-enumerate-devices` 单独验证；权限不足时检查 udev 规则。")
        return False
    print(f"[ok] 找到 {len(devs)} 台 RealSense 设备：")
    for d in devs:
        name = d.get_info(rs.camera_info.name)
        sn = d.get_info(rs.camera_info.serial_number)
        fw = d.get_info(rs.camera_info.firmware_version)
        usb = d.get_info(rs.camera_info.usb_type_descriptor) if d.supports(rs.camera_info.usb_type_descriptor) else "?"
        print(f"     - {name}  SN={sn}  FW={fw}  USB={usb}")
    return True


def query_depth_profiles(rs):
    """返回这台相机支持的所有 (w, h, fps) 深度 z16 模式列表。"""
    ctx = rs.context()
    devs = ctx.query_devices()
    profiles = []
    for s in devs[0].query_sensors():
        for p in s.get_stream_profiles():
            if p.stream_type() == rs.stream.depth and p.format() == rs.format.z16:
                vp = p.as_video_stream_profile()
                profiles.append((vp.width(), vp.height(), p.fps()))
    return sorted(set(profiles))


def pick_depth_profile(rs, want_w, want_h, want_fps):
    """从支持列表里挑一个深度模式：优先精确命中；否则挑分辨率不太大、fps 合理的。"""
    profs = query_depth_profiles(rs)
    if not profs:
        return None, []
    if (want_w, want_h, want_fps) in profs:
        return (want_w, want_h, want_fps), profs
    # 退而求其次：宽<=848，fps 优先 30→15→6，分辨率取最接近 320x180 偏大的
    def score(p):
        w, h, f = p
        fps_rank = {30: 0, 15: 1, 6: 2}.get(f, 3)
        return (0 if w <= 848 else 1, fps_rank, abs(w - want_w) + abs(h - want_h))
    return sorted(profs, key=score)[0], profs


def normalize_for_model(depth_m: np.ndarray) -> np.ndarray:
    """米制深度 → 模型输入 (180,320,1) ∈[0,1]。
    clamp(d,0,5)/5；无效(<=0/nan)与越界(>=5)像素 → 0.0（与训练一致）。"""
    d = depth_m.astype(np.float32).copy()
    invalid = ~np.isfinite(d) | (d <= 0.0) | (d >= MAX_DEPTH)
    d = np.clip(d, 0.0, MAX_DEPTH) / MAX_DEPTH
    d[invalid] = 0.0
    if d.shape != (MODEL_H, MODEL_W):
        try:
            import cv2
            d = cv2.resize(d, (MODEL_W, MODEL_H), interpolation=cv2.INTER_NEAREST)
        except ImportError:
            # 无 cv2 时用最近邻手工降采样
            ys = (np.linspace(0, d.shape[0] - 1, MODEL_H)).astype(int)
            xs = (np.linspace(0, d.shape[1] - 1, MODEL_W)).astype(int)
            d = d[ys][:, xs]
    return d.reshape(MODEL_H, MODEL_W, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=424, help="深度流原生宽（会 resize 到 320）")
    ap.add_argument("--height", type=int, default=240, help="深度流原生高（会 resize 到 180）")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=200, help="跑多少帧后停；0=无限")
    ap.add_argument("--show", action="store_true", help="cv2 窗口实时看伪彩深度（需显示器）")
    ap.add_argument("--save-one", default=None, metavar="PREFIX",
                    help="存一帧：PREFIX_raw_m.npy + PREFIX_model_320x180.npy + PREFIX_color.png")
    ap.add_argument("--print-every", type=int, default=10, help="每 N 帧打印一次统计")
    ap.add_argument("--list-profiles", action="store_true", help="只列出相机支持的深度模式后退出")
    # ---- 灰度 MJPEG + L/F/R 遮挡标注 ----
    ap.add_argument("--mjpeg-port", type=int, default=0,
                    help=">0 时起 MJPEG 服务，浏览器看实时灰度+遮挡标注(headless 可用)")
    ap.add_argument("--scale", type=int, default=3, help="MJPEG 放大倍数(320x180 * scale)")
    ap.add_argument("--near-thresh", type=float, default=0.30,
                    help="归一化近距阈值，<它算近/挡(0.30≈1.5m, 0.18≈0.9m)")
    ap.add_argument("--block-ratio", type=float, default=0.15,
                    help="某区近点占比 > 它即判该区被挡")
    ap.add_argument("--roi-rows", default="0.10,0.55",
                    help="ROI 水平带行范围(比例) lo,hi，排除近地面/天花板")
    args = ap.parse_args()
    roi_lo, roi_hi = [float(x) for x in args.roi_rows.split(",")]

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[err] 未安装 pyrealsense2。`pip install pyrealsense2`（Jetson 上若用源码 "
              "编译的 librealsense，pyrealsense2 应已在 PYTHONPATH 里）。")
        return

    if not list_devices(rs):
        return

    # 列出 / 自动挑选这台相机真正支持的深度模式（避免 "Couldn't resolve requests"）
    chosen, profs = pick_depth_profile(rs, args.width, args.height, args.fps)
    print(f"[profiles] 支持的深度 z16 模式（w x h @ fps）：")
    for w, h, f in profs:
        mark = "  <= 选用" if (w, h, f) == chosen else ""
        print(f"     {w}x{h}@{f}{mark}")
    if args.list_profiles:
        return
    if chosen is None:
        print("[err] 该相机没有可用的深度 z16 模式。")
        return
    cw, ch, cfps = chosen

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, cw, ch, rs.format.z16, cfps)
    print(f"[start] depth {cw}x{ch}@{cfps}fps ...")
    try:
        profile = pipe.start(cfg)
    except RuntimeError as e:
        print(f"[err] 启动失败：{e}")
        print("      → 试试换 USB3 口/线；或用 --list-profiles 看支持模式后用 --width/--height/--fps 手动指定。")
        return
    sensor = profile.get_device().first_depth_sensor()
    depth_scale = sensor.get_depth_scale()   # z16 单位 → 米
    print(f"[info] depth_scale = {depth_scale}  (z16 * scale = 米)")

    # MJPEG 服务（可选）
    mjpeg_state = None
    if args.mjpeg_port > 0:
        try:
            import cv2  # noqa: F401  渲染/编码都需要
            mjpeg_state = _MJPEGState()
            start_mjpeg_server(args.mjpeg_port, mjpeg_state)
        except ImportError:
            print("[warn] --mjpeg-port 需要 opencv-python；已关闭 MJPEG。")
            args.mjpeg_port = 0

    saved = False
    step = 0
    t_last = time.time()
    try:
        while True:
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            z16 = np.asanyarray(depth_frame.get_data())          # (H,W) uint16
            depth_m = z16.astype(np.float32) * depth_scale       # 米
            model_in = normalize_for_model(depth_m)              # (180,320,1) ∈[0,1]

            # 几何遮挡 + 灰度标注推流
            if mjpeg_state is not None:
                import cv2
                occ = region_occlusion(model_in.reshape(MODEL_H, MODEL_W),
                                       args.near_thresh, roi_lo, roi_hi)
                verdict = occlusion_verdict(occ, args.block_ratio)
                vis = render_overlay(model_in, occ, verdict, roi_lo, roi_hi, args.scale)
                ok, buf = cv2.imencode(".jpg", vis,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    mjpeg_state.update(buf.tobytes())

            # 统计（基于模型输入帧，和 sim 探针口径一致）
            mflat = model_in.reshape(-1)
            valid_ratio = float(np.mean((depth_m > 0) & (depth_m < MAX_DEPTH)))
            has_nan = bool(np.isnan(mflat).any())

            if args.print_every and step % args.print_every == 0:
                now = time.time()
                fps = args.print_every / (now - t_last) if step > 0 else 0.0
                t_last = now
                print(f"[{step:5d}] ~{fps:4.1f}fps | raw {z16.shape} 米[{depth_m[depth_m>0].min() if (depth_m>0).any() else 0:.2f},"
                      f"{depth_m.max():.2f}] | model(180,320) min={mflat.min():.3f} "
                      f"max={mflat.max():.3f} mean={mflat.mean():.3f} | 有效像素={valid_ratio*100:.1f}% | NaN={has_nan}")

            if args.save_one and not saved:
                np.save(f"{args.save_one}_raw_m.npy", depth_m)
                np.save(f"{args.save_one}_model_320x180.npy", model_in.astype(np.float32))
                try:
                    import cv2
                    vis = (np.clip(depth_m, 0, MAX_DEPTH) / MAX_DEPTH * 255).astype(np.uint8)
                    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    cv2.imwrite(f"{args.save_one}_color.png", vis)
                    print(f"[save] 已存 {args.save_one}_raw_m.npy / _model_320x180.npy / _color.png")
                except ImportError:
                    print(f"[save] 已存 {args.save_one}_raw_m.npy / _model_320x180.npy（无 cv2，跳过 png）")
                saved = True

            if args.show:
                try:
                    import cv2
                    vis = (np.clip(depth_m, 0, MAX_DEPTH) / MAX_DEPTH * 255).astype(np.uint8)
                    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    cv2.imshow("D435i depth (JET, 0-5m)", vis)
                    # cv2.imshow("D435i depth (gary, 0-5m)", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except ImportError:
                    print("[warn] --show 需要 opencv-python；跳过显示。")
                    args.show = False

            step += 1
            if args.frames and step >= args.frames:
                break
    except KeyboardInterrupt:
        print("\n[stop] 用户中断。")
    finally:
        pipe.stop()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        print(f"[done] 共 {step} 帧。")


if __name__ == "__main__":
    main()
