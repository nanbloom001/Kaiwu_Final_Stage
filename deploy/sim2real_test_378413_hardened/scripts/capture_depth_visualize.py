#!/usr/bin/env python3
"""
D435i 深度图抓帧 + 可视化对比脚本

用途:
  在真机上抓 30 帧 D435i 深度图, 对比 [原始 vs hole_filling vs spatial_filter]
  的效果, 判断真机 depth 的空洞形态 + 是否能用 RealSense 滤波改善。

输出:
  ~/Kaiwu-test/sim2real_test_378413_hardened/logs/depth_capture/depth_compare_*.png
  每张图含 3 列: [原始] [hole_filling] [spatial+hole_filling]

用法:
  1. 把机器狗放在有障碍物的场景 (面对墙/箱子 0.5-1m)
  2. 运行: python3 scripts/capture_depth_visualize.py
  3. 等 5 秒抓完 30 帧
  4. 看输出的 PNG 图, 或发给我分析

按键:
  Ctrl+C 提前停止
"""
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 未安装: pip3 install pyrealsense2")

try:
    import cv2
except ImportError:
    sys.exit("opencv-python 未安装: pip3 install opencv-python")


def depth_to_color(depth_m, max_m=5.0):
    """深度图归一化 + colormap.
    修正: jet(0)=蓝, jet(255)=红. 我们要 近=红 远=蓝, 所以"近物值大".
    实际真机大部分深度 0.5-1m, 直接 /5 会让它们落在 25-50 (深蓝), 看不清.
    改为: 反转 + 拉伸. 近物(<=1m) → 高值(红), 远处 → 低值(蓝), 无效 → 0(深蓝).
    """
    d = depth_m.copy().astype(np.float32)
    # 无效像素 (<=0 或 >=5 或 NaN) 单独标记
    invalid = (d <= 0) | (d >= max_m) | ~np.isfinite(d)
    # 有效深度做拉伸: 0m→255, 2m→0, 让 0.5-1m 落在中高值区 (黄/绿)
    d_norm = np.clip(d / 2.0, 0, 1)  # 0~2m 映射到 0~1
    d_norm = (1.0 - d_norm) * 255   # 反转: 近=255(红), 2m+=0(蓝)
    d_norm[invalid] = 0  # 无效 = 深蓝 (和远处一样, 但实际从统计能区分)
    d_8u = d_norm.astype(np.uint8)
    return cv2.applyColorMap(d_8u, cv2.COLORMAP_JET)


def stats(depth_m):
    """统计 depth 质量指标"""
    total = depth_m.size
    invalid = np.sum((depth_m <= 0) | (depth_m >= 5.0) | ~np.isfinite(depth_m))
    valid = total - invalid
    if valid > 0:
        valid_vals = depth_m[(depth_m > 0) & (depth_m < 5.0)]
        mean_d = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0
        # front box (训练的中下方中央 96x72): 图像中央偏下
        h, w = depth_m.shape
        front = depth_m[int(h*0.4):int(h*0.8), int(w*0.35):int(w*0.65)]
        front_invalid = np.sum((front <= 0) | (front >= 5.0))
        front_valid = front[(front > 0) & (front < 5.0)]
        front_min = float(np.min(front_valid)) if len(front_valid) > 0 else -1
        front_mean = float(np.mean(front_valid)) if len(front_valid) > 0 else -1
    else:
        mean_d = 0; front_min = -1; front_mean = -1; front_invalid = 0
    return {
        "invalid_pct": 100 * invalid / total,
        "mean_depth_m": mean_d,
        "front_min_m": front_min,
        "front_mean_m": front_mean,
        "front_invalid_pct": 100 * front_invalid / front.size,
    }


def main():
    out_dir = Path.home() / "Kaiwu-test/sim2real_test_378413_hardened/logs/depth_capture"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[init] 启动 D435i...")
    pipe = rs.pipeline()
    cfg = rs.config()

    # D435i 实际支持的 depth profile (从 pyrealsense2 查询):
    #   640x480@30, 640x360@30, 480x270@30, 256x144@90, 848x480@6/8/10, ...
    # 部署代码请求 424x240 但 _pick_profile 会 fallback 到 480x270@30.
    # 这里直接用 480x270@30 (和部署实际使用的最接近).
    # 先列出支持的 profile, 自动选 30fps + 接近 424x240 的.
    ctx = rs.context()
    dev = ctx.query_devices()[0]
    target_w, target_h, target_fps = 424, 240, 30
    best_profile = None
    best_score = float('inf')
    print(f"[init] 搜索最接近 {target_w}x{target_h}@{target_fps} 的 profile...")
    for sensor in dev.query_sensors():
        for p in sensor.get_stream_profiles():
            if p.stream_type() == rs.stream.depth and p.format() == rs.format.z16:
                vp = p.as_video_stream_profile()
                w, h, f = vp.width(), vp.height(), p.fps()
                if f == 30:
                    score = abs(w - target_w) + abs(h - target_h)
                    if score < best_score:
                        best_score = score
                        best_profile = (w, h, f)
                    print(f"  candidate: {w}x{h}@{f} (score={score})")

    if best_profile is None:
        sys.exit("[ERR] D435i 没有支持 30fps 的 depth profile")
    cw, ch, cfps = best_profile
    print(f"[init] 使用 profile: {cw}x{ch}@{cfps}")
    cfg.enable_stream(rs.stream.depth, cw, ch, rs.format.z16, cfps)
    profile = pipe.start(cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"[init] depth_scale = {depth_scale:.6f} (m/raw_unit)")
    print(f"[init] 实际采集分辨率: {cw}x{ch}, 模型输入 320x180, 会 resize")

    # 准备滤波器
    # 1. hole_filling (填补空洞, 最相关)
    hf = rs.hole_filling_filter()
    # 2. spatial (边缘保留空间滤波)
    sp = rs.spatial_filter()
    sp.set_option(rs.option.filter_magnitude, 2)
    sp.set_option(rs.option.holes_fill, 2)
    # 3. temporal (时序, 需要 2 帧; 先不深用)
    tp = rs.temporal_filter()
    tp.set_option(rs.option.filter_smooth_alpha, 0.4)

    print("[init] 预热 1 秒...")
    for _ in range(30):
        pipe.wait_for_frames()
    print("[init] 就绪. 现在把机器狗对着障碍物 (墙/箱子), 按 Enter 开始抓帧...")
    input()

    n_frames = 30
    print(f"\n[capture] 抓 {n_frames} 帧 (每 0.1s 一帧)...")

    all_stats_raw = []
    all_stats_hf = []
    all_stats_sp = []
    frames_saved = 0

    for i in range(n_frames):
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # 原始 depth (米)
        raw = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

        # hole_filling
        hf_frame = hf.process(depth_frame)
        hf_arr = np.asanyarray(hf_frame.get_data()).astype(np.float32) * depth_scale

        # spatial + hole_filling
        sp_frame = sp.process(depth_frame)
        sp_frame = hf.process(sp_frame)
        sp_arr = np.asanyarray(sp_frame.get_data()).astype(np.float32) * depth_scale

        # 统计
        s_raw = stats(raw)
        s_hf = stats(hf_arr)
        s_sp = stats(sp_arr)
        all_stats_raw.append(s_raw)
        all_stats_hf.append(s_hf)
        all_stats_sp.append(s_sp)

        # 可视化 (resize 到 320x180 匹配模型输入)
        h, w = 180, 320
        raw_v = cv2.resize(depth_to_color(raw), (w, h), interpolation=cv2.INTER_NEAREST)
        hf_v = cv2.resize(depth_to_color(hf_arr), (w, h), interpolation=cv2.INTER_NEAREST)
        sp_v = cv2.resize(depth_to_color(sp_arr), (w, h), interpolation=cv2.INTER_NEAREST)

        # 拼接 + 标注
        def label(img, text, inval_pct):
            img2 = img.copy()
            cv2.putText(img2, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(img2, f"inval:{inval_pct:.0f}%", (5, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            return img2

        combined = np.hstack([
            label(raw_v, "RAW (no filter)", s_raw['invalid_pct']),
            label(hf_v, "hole_filling", s_hf['invalid_pct']),
            label(sp_v, "spatial+hole_fill", s_sp['invalid_pct']),
        ])

        cv2.putText(combined, f"frame {i+1}/{n_frames}", (5, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 每 5 帧保存一张
        if i % 5 == 0 or i == n_frames - 1:
            ts = datetime.now().strftime("%H%M%S")
            path = out_dir / f"depth_compare_f{i:02d}_{ts}.png"
            cv2.imwrite(str(path), combined)
            frames_saved += 1

        # 实时打印
        print(f"  [{i+1:2d}/{n_frames}] raw_inval={s_raw['invalid_pct']:5.1f}%  "
              f"hf_inval={s_hf['invalid_pct']:5.1f}%  "
              f"sp_inval={s_sp['invalid_pct']:5.1f}%  "
              f"raw_front_min={s_raw['front_min_m']:.2f}m  "
              f"hf_front_min={s_hf['front_min_m']:.2f}m",
              flush=True)

        time.sleep(0.1)

    pipe.stop()
    print(f"\n[done] 保存 {frames_saved} 张对比图到: {out_dir}")

    # 汇总统计
    print("\n" + "=" * 72)
    print(" 汇总统计 (30 帧平均)")
    print("=" * 72)

    def avg(stats_list, key):
        return float(np.mean([s[key] for s in stats_list]))

    print(f"\n{'指标':<25} {'原始':<12} {'hole_fill':<12} {'spatial+hf':<12}")
    print("-" * 60)
    for key, label in [
        ("invalid_pct", "全图无效像素 %"),
        ("front_invalid_pct", "前方区域无效 %"),
        ("front_min_m", "前方最近距离 (m)"),
        ("front_mean_m", "前方平均距离 (m)"),
    ]:
        r = avg(all_stats_raw, key)
        h = avg(all_stats_hf, key)
        s = avg(all_stats_sp, key)
        fmt = f"{r:<12.2f} {h:<12.2f} {s:<12.2f}" if "pct" not in key else f"{r:<12.1f} {h:<12.1f} {s:<12.1f}"
        print(f"  {label:<23} {fmt}")

    print("\n" + "=" * 72)
    print(" 判定")
    print("=" * 72)
    raw_inv = avg(all_stats_raw, "invalid_pct")
    hf_inv = avg(all_stats_hf, "invalid_pct")
    if hf_inv < raw_inv - 3:
        print(f"✅ hole_filling 显著减少无效像素 ({raw_inv:.1f}% → {hf_inv:.1f}%)")
        print("   → 建议在部署代码 DepthSource.h 里启用 hole_filling_filter")
    elif raw_inv > 15:
        print(f"⚠️ 原始 depth 无效像素偏高 ({raw_inv:.1f}%), hole_filling 效果不明显")
        print("   → 可能是物理遮挡/反光, 滤波解决不了, 需要调整相机角度")
    else:
        print(f"✓ 原始 depth 质量可接受 ({raw_inv:.1f}%), 滤波改善有限")

    # 块状 vs 散点: 看无效像素帧间方差
    raw_inv_ts = [s["invalid_pct"] for s in all_stats_raw]
    raw_inv_std = float(np.std(raw_inv_ts))
    print(f"\n无效像素帧间波动: std={raw_inv_std:.1f}%")
    if raw_inv_std > 5:
        print("⚠️ 波动大 → 时序不稳定, 部分帧突然失效 (块状失效特征)")
    else:
        print("✓ 波动小 → 稳定的散点噪声为主")

    print(f"\n对比图: {out_dir}")
    print("把这些 PNG 发给分析人员 (或直接看图判断空洞形态)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] 用户停止")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
