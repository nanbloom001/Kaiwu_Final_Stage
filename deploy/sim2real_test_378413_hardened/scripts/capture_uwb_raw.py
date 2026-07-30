#!/usr/bin/env python3
"""
UWB 原始数据抓包脚本 —— 静态录制(狗不动,人拿遥控器走动)

功能:
  1. 订阅 rt/uwbstate, 录制全部字段到 CSV(带时间戳)
  2. 实时终端显示关键字段 + 跳变告警
  3. 录制结束后自动统计: 跳变次数、分布、时序特征
  4. 不发送任何运动指令, 不启动控制器

用法:
  cd ~/Kaiwu-test/sim2real_test_loco
  python3 scripts/capture_uwb_raw.py --duration 60 --network eth0

  # 录60秒, 期间拿着UWB遥控器在狗周围走动
  # 录完自动生成统计 + 保存到 logs/uwb_capture/

按键:
  Ctrl+C 提前停止(也会保存已录数据)
"""
import argparse
import csv
import math
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# UWB 字段列表(全部)
UWB_FIELDS = [
    "orientation_est",  # beta: 水平方位角(弧度)
    "pitch_est",        # 俯仰角(弧度)
    "distance_est",     # 三维空间距离(米)
    "yaw_est",          # 标签朝向偏航角
    "tag_roll", "tag_pitch", "tag_yaw",    # 遥控器自身姿态
    "base_roll", "base_pitch", "base_yaw", # 机器狗姿态
    "error_state",      # 0=正常
    "enabled_from_app", # 1=UWB已启用
    "channel",
]

CSV_HEADER = ["timestamp", "t_relative", "valid"] + UWB_FIELDS + [
    "planar_distance",   # = distance * cos(pitch)
    "local_x",           # = planar * cos(beta)
    "local_y",           # = planar * sin(beta)
    "beta_deg",          # 方位角(度,方便人看)
]


def main():
    ap = argparse.ArgumentParser(description="UWB 原始数据抓包")
    ap.add_argument("--duration", type=float, default=60, help="录制时长(秒), 0=无限")
    ap.add_argument("--network", default="eth0", help="网卡")
    ap.add_argument("--output-dir", default=None, help="输出目录(默认 logs/uwb_capture)")
    ap.add_argument("--jump-threshold", type=float, default=30.0,
                    help="方位角跳变告警阈值(度), 默认30°")
    args = ap.parse_args()

    # 输出目录
    out_dir = Path(args.output_dir) if args.output_dir else \
              Path.home() / "Kaiwu-test/sim2real_test_loco/logs/uwb_capture"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"uwb_raw_{ts}.csv"

    # 初始化 DDS
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import UwbState_
    except ImportError as e:
        sys.exit(f"SDK导入失败: {e}")

    print(f"[初始化] 网卡={args.network}")
    ChannelFactoryInitialize(0, args.network)
    sub = ChannelSubscriber("rt/uwbstate", UwbState_)

    # 录制状态(回调函数和主循环共享)
    rows = []
    t_start = time.time()
    last_beta_deg = None
    jump_count = 0
    max_jump = 0.0
    frame_count = 0
    valid_count = 0
    recording = True
    latest_msg = {"data": None, "time": 0.0}
    msg_lock = threading.Lock()

    def on_msg(msg):
        """UWB 回调: 收到一帧就处理+存储"""
        nonlocal last_beta_deg, jump_count, max_jump, frame_count, valid_count
        if not recording:
            return

        now = time.time()
        t_rel = now - t_start
        frame_count += 1

        valid = (msg.error_state == 0 and msg.enabled_from_app == 1
                 and msg.distance_est >= 0.0 and math.isfinite(msg.orientation_est))
        if valid:
            valid_count += 1

        beta = msg.orientation_est
        pitch = msg.pitch_est
        dist = msg.distance_est
        planar = dist * math.cos(pitch) if valid else 0.0
        lx = planar * math.cos(beta) if valid else 0.0
        ly = planar * math.sin(beta) if valid else 0.0
        beta_deg = math.degrees(beta) if valid else 0.0

        # 跳变检测
        jump_flag = ""
        if valid and last_beta_deg is not None:
            jump = abs(beta_deg - last_beta_deg)
            if jump > 180:
                jump = 360 - jump
            if jump > args.jump_threshold:
                jump_count += 1
                jump_flag = f" JUMP{jump:.0f}"
                if jump > max_jump:
                    max_jump = jump
        if valid:
            last_beta_deg = beta_deg

        row = {
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),
            "t_relative": f"{t_rel:.4f}",
            "valid": int(valid),
            "orientation_est": f"{beta:.6f}",
            "pitch_est": f"{pitch:.6f}",
            "distance_est": f"{dist:.4f}",
            "yaw_est": f"{msg.yaw_est:.6f}",
            "tag_roll": f"{msg.tag_roll:.6f}",
            "tag_pitch": f"{msg.tag_pitch:.6f}",
            "tag_yaw": f"{msg.tag_yaw:.6f}",
            "base_roll": f"{msg.base_roll:.6f}",
            "base_pitch": f"{msg.base_pitch:.6f}",
            "base_yaw": f"{msg.base_yaw:.6f}",
            "error_state": int(msg.error_state),
            "enabled_from_app": int(msg.enabled_from_app),
            "channel": int(msg.channel),
            "planar_distance": f"{planar:.4f}",
            "local_x": f"{lx:.4f}",
            "local_y": f"{ly:.4f}",
            "beta_deg": f"{beta_deg:.2f}",
        }
        rows.append(row)
        with msg_lock:
            latest_msg["data"] = row
            latest_msg["time"] = now
            latest_msg["jump"] = jump_flag

    # 用回调方式初始化订阅
    sub.Init(on_msg, 10)

    def signal_handler(sig, frame):
        nonlocal recording
        recording = False
    signal.signal(signal.SIGINT, signal_handler)

    duration_str = f"{args.duration:.0f}秒" if args.duration > 0 else "无限(Ctrl+C停止)"
    print(f"[录制] 时长={duration_str}  输出={csv_path}")
    print(f"[录制] 跳变告警阈值={args.jump_threshold}°")
    print(f"[录制] 现在拿着UWB遥控器在狗周围走动...")
    print("=" * 72)

    # 主循环: 显示最新帧 + 计时
    while recording:
        with msg_lock:
            row = latest_msg.get("data")
            jump_flag = latest_msg.get("jump", "")
        if row:
            t_rel = float(row["t_relative"])
            status = "OK" if int(row["valid"]) else "BAD"
            print(f"\r[{t_rel:6.1f}s] {status} b={float(row['beta_deg']):+7.1f}d "
                  f"d={float(row['distance_est']):5.2f}m "
                  f"plan={float(row['planar_distance']):5.2f}m "
                  f"(x={float(row['local_x']):+.2f},y={float(row['local_y']):+.2f}) "
                  f"err={row['error_state']}{jump_flag}     ", end="", flush=True)
        else:
            t_rel = time.time() - t_start
            print(f"\r[{t_rel:6.1f}s] 等待UWB数据...", end="", flush=True)

        if args.duration > 0 and (time.time() - t_start) >= args.duration:
            break
        time.sleep(0.05)

    # 保存CSV
    print("\n" + "=" * 72)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    # 统计
    print(f"\n[保存] {csv_path}  ({len(rows)} 帧)")
    elapsed = time.time() - t_start
    hz = valid_count / elapsed if elapsed > 0 else 0
    print(f"\n{'='*72}")
    print(f" 录制统计")
    print(f"{'='*72}")
    print(f"  总时长:     {elapsed:.1f} 秒")
    print(f"  总帧数:     {frame_count}")
    print(f"  有效帧:     {valid_count} ({valid_count/max(frame_count,1)*100:.1f}%)")
    print(f"  频率:       {hz:.1f} Hz")
    print(f"  跳变次数:   {jump_count} (>±{args.jump_threshold}°)")
    print(f"  最大跳变:   {max_jump:.0f}°")

    # 距离/beta 分布
    if valid_count > 0:
        valid_rows = [r for r in rows if int(r["valid"]) == 1]
        dists = [float(r["distance_est"]) for r in valid_rows]
        betas = [float(r["beta_deg"]) for r in valid_rows]
        planars = [float(r["planar_distance"]) for r in valid_rows]
        print(f"\n  距离分布:    {min(dists):.2f}m ~ {max(dists):.2f}m  均值={sum(dists)/len(dists):.2f}m")
        print(f"  方位角分布:  {min(betas):+.1f}° ~ {max(betas):+.1f}°")

        # 距离分档
        bins = [(0,2,"近"),(2,4,"中"),(4,6,"中远"),(6,99,"远")]
        print(f"\n  距离分档有效帧:")
        for lo, hi, label in bins:
            cnt = sum(1 for d in dists if lo <= d < hi)
            print(f"    {label}({lo}-{hi}m): {cnt}帧 ({cnt/len(dists)*100:.0f}%)")

    print(f"\n[下一步] 用脚本分析: python3 scripts/capture_uwb_raw.py --analyze {csv_path}")


if __name__ == "__main__":
    main()
