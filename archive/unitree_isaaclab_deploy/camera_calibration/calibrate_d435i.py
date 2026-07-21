#!/usr/bin/env python3
"""
D435i hand-eye calibration for Go2.

Coordinate frames:
- Optical frame: z-forward, x-right, y-down (camera convention)
- URDF/Robot frame: x-forward, y-left, z-up (ROS convention)

The calibration output (calibration_result.json) stores raw optical frame results.
URDF generation applies the coordinate transform from optical to robot frame.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient
except ImportError:
    VideoClient = None

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_M = 0.025
DATA_DIR = Path(os.path.expanduser("~/calibration_data"))
RESULT_FILE = DATA_DIR / "calibration_result.json"
EXTRINSICS_FILE = DATA_DIR / "d435i_extrinsics.json"

# Transform: optical (z-fwd, x-right, y-down) -> URDF (x-fwd, y-left, z-up)
T_OPT_TO_URDF = np.array([
    [0, 0, 1],    # urdf_x = opt_z
    [-1, 0, 0],   # urdf_y = -opt_x
    [0, -1, 0]    # urdf_z = -opt_y
], dtype=np.float64)


class Go2Camera:
    def __init__(self, iface="eth0"):
        if VideoClient is None:
            raise RuntimeError("unitree_sdk2py not installed")
        ChannelFactoryInitialize(0, iface)
        self.client = VideoClient()
        self.client.SetTimeout(3.0)
        self.client.Init()

    def read(self):
        code, data = self.client.GetImageSample()
        if code != 0:
            return False, None
        img = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
        return (True, img) if img is not None else (False, None)


def detect_chessboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)
    if ok:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ok, corners


def make_objpoints():
    pts = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    pts[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    return pts * SQUARE_SIZE_M


def save_d435i_extrinsics(profile):
    """Save depth-to-color extrinsics from D435i pipeline profile."""
    depth_s = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_s = profile.get_stream(rs.stream.color).as_video_stream_profile()
    ext = color_s.get_extrinsics_to(depth_s)
    data = {
        "color_to_depth_translation": list(ext.translation),
        "color_to_depth_rotation": [list(ext.rotation[i*3:(i+1)*3]) for i in range(3)]
    }
    with open(str(EXTRINSICS_FILE), 'w') as f:
        json.dump(data, f, indent=2)
    t = ext.translation
    print(f"D435i extrinsics saved: color->depth [{t[0]*1000:.2f}, {t[1]*1000:.2f}, {t[2]*1000:.2f}] mm")


def cmd_query_extrinsics(args):
    """Query and save D435i depth-to-color extrinsics (standalone)."""
    if rs is None:
        sys.exit("pyrealsense2 not installed")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    time.sleep(1)
    save_d435i_extrinsics(profile)
    pipe.stop()


def cmd_capture(args):
    if rs is None:
        sys.exit("pyrealsense2 not installed")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    go2_dir, d435i_dir = DATA_DIR / "go2_front", DATA_DIR / "d435i_rgb"
    go2_dir.mkdir(exist_ok=True)
    d435i_dir.mkdir(exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    time.sleep(2)

    save_d435i_extrinsics(profile)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K_d435i = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    np.save(str(DATA_DIR / "K_d435i.npy"), K_d435i)
    print(f"D435i intrinsics saved: fx={intr.fx:.1f}")

    go2 = Go2Camera(args.network)
    ret, _ = go2.read()
    if not ret:
        pipe.stop()
        sys.exit("Go2 camera connection failed")

    count = 0
    try:
        while count < args.num_captures:
            frames = pipe.wait_for_frames()
            d435i_frame = np.asanyarray(frames.get_color_frame().get_data())
            ret, go2_frame = go2.read()
            if not ret:
                continue

            ok_g, _ = detect_chessboard(go2_frame)
            ok_d, _ = detect_chessboard(d435i_frame)
            status = f"[{count}/{args.num_captures}] Go2:{'OK' if ok_g else 'NO'} D435i:{'OK' if ok_d else 'NO'}"
            print(f"{status} (Enter to capture, q to quit): ", end="", flush=True)

            cmd = input().strip().lower()
            if cmd == 'q':
                break
            if ok_g and ok_d:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(go2_dir / f"go2_{count:03d}_{ts}.png"), go2_frame)
                cv2.imwrite(str(d435i_dir / f"d435i_{count:03d}_{ts}.png"), d435i_frame)
                count += 1
                print(f"  Captured pair {count}")
    finally:
        pipe.stop()
    print(f"Done: {count} image pairs saved")


def cmd_calibrate(args):
    print("=== D435i Calibration ===\n")

    go2_imgs = sorted((DATA_DIR / "go2_front").glob("*.png"))
    d435i_imgs = sorted((DATA_DIR / "d435i_rgb").glob("*.png"))
    if len(go2_imgs) != len(d435i_imgs) or not go2_imgs:
        sys.exit("No matching image pairs found")

    objp = make_objpoints()

    # Calibrate Go2 intrinsics from captured images
    obj_pts, corners_list = [], []
    img_size = None
    for p in go2_imgs:
        img = cv2.imread(str(p))
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
        ok, c = detect_chessboard(img)
        if ok:
            obj_pts.append(objp)
            corners_list.append(c)

    _, K_go2, dist_go2, _, _ = cv2.calibrateCamera(obj_pts, corners_list, img_size, None, None)
    print(f"Go2 intrinsics: fx={K_go2[0,0]:.1f}, fy={K_go2[1,1]:.1f}")

    K_d435i = np.load(str(DATA_DIR / "K_d435i.npy")) if (DATA_DIR / "K_d435i.npy").exists() else np.eye(3) * 615
    dist_d435i = np.zeros(5)

    # Compute relative poses
    estimates = []
    for gp, dp in zip(go2_imgs, d435i_imgs):
        ok_g, c_g = detect_chessboard(cv2.imread(str(gp)))
        ok_d, c_d = detect_chessboard(cv2.imread(str(dp)))
        if not (ok_g and ok_d):
            print(f"  SKIP: {gp.name} (detection failed)")
            continue

        _, rv_g, tv_g = cv2.solvePnP(objp, c_g, K_go2, dist_go2)
        _, rv_d, tv_d = cv2.solvePnP(objp, c_d, K_d435i, dist_d435i)

        Rg, _ = cv2.Rodrigues(rv_g)
        Rd, _ = cv2.Rodrigues(rv_d)

        Tg = np.eye(4)
        Tg[:3, :3], Tg[:3, 3] = Rg, tv_g.flatten()
        Td = np.eye(4)
        Td[:3, :3], Td[:3, 3] = Rd, tv_d.flatten()

        # D435i pose relative to Go2 camera (both in optical frame)
        T_rel = Tg @ np.linalg.inv(Td)
        estimates.append(T_rel)
        print(f"  OK: {gp.name}")

    if len(estimates) < 3:
        sys.exit(f"Need at least 3 valid pairs, got {len(estimates)}")

    # Average translation
    trans = np.mean([T[:3, 3] for T in estimates], axis=0)

    # Average rotation via quaternion averaging
    quats = np.array([R.from_matrix(T[:3, :3]).as_quat() for T in estimates])
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]
    avg_q = np.mean(quats, axis=0)
    avg_q /= np.linalg.norm(avg_q)
    rot = R.from_quat(avg_q).as_matrix()
    rpy = R.from_matrix(rot).as_euler('xyz')

    # Save in optical frame (raw calibration result)
    result = {
        "xyz": trans.tolist(),
        "rpy": rpy.tolist(),
        "rpy_degrees": np.degrees(rpy).tolist(),
        "parent_link": "front_camera",
        "frame": "optical",
        "num_pairs": len(estimates)
    }

    with open(str(RESULT_FILE), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Result (optical frame: z-fwd, x-right, y-down) ===")
    print(f"xyz: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]")
    print(f"rpy: [{np.degrees(rpy[0]):.2f}, {np.degrees(rpy[1]):.2f}, {np.degrees(rpy[2]):.2f}] deg")
    print(f"Saved to: {RESULT_FILE}")


def optical_to_urdf(xyz_opt, rpy_opt):
    """Transform from optical frame to URDF/robot frame."""
    # Position: urdf_xyz = T @ optical_xyz
    xyz_urdf = T_OPT_TO_URDF @ xyz_opt

    # Rotation: R_urdf = T @ R_opt @ T^T
    R_opt = R.from_euler('xyz', rpy_opt).as_matrix()
    R_urdf = T_OPT_TO_URDF @ R_opt @ T_OPT_TO_URDF.T
    rpy_urdf = R.from_matrix(R_urdf).as_euler('xyz')

    return xyz_urdf, rpy_urdf


def cmd_generate_urdf(args):
    import xml.etree.ElementTree as ET

    with open(str(RESULT_FILE)) as f:
        calib = json.load(f)

    xyz_opt = np.array(calib["xyz"])
    rpy_opt = np.array(calib["rpy"])
    parent = calib.get("parent_link", "front_camera")

    # Transform to URDF frame
    xyz, rpy = optical_to_urdf(xyz_opt, rpy_opt)

    # Load depth extrinsics (color->depth offset in optical frame)
    if EXTRINSICS_FILE.exists():
        with open(str(EXTRINSICS_FILE)) as f:
            ext = json.load(f)
        c2d_opt = np.array(ext["color_to_depth_translation"])
        c2d_rot = np.array(ext["color_to_depth_rotation"])
    else:
        print("WARNING: No extrinsics file, using D435i defaults")
        c2d_opt = np.array([-0.015, 0.0, 0.0])  # Intel spec: depth_to_color=15mm
        c2d_rot = np.eye(3)

    # Depth position relative to RGB (d435i_link)
    # D435i layout (front view): [left IR/depth] ... [RGB], depth LEFT of RGB
    # User facing Go2: user's left = Go2's right = urdf y-
    depth_xyz_urdf = T_OPT_TO_URDF @ c2d_opt
    depth_xyz_urdf[1] = -depth_xyz_urdf[1]  # Flip: depth on Go2's RIGHT (user's left)

    # Depth optical frame rotation (with factory calibration)
    R_c2d = np.array(c2d_rot)
    R_for_joint = T_OPT_TO_URDF @ R_c2d.T
    depth_rpy = R.from_matrix(R_for_joint).as_euler('xyz')

    tree = ET.parse(args.base_urdf)
    root = tree.getroot()

    # Remove existing d435i elements
    for tag, name in [("link", "d435i_link"), ("joint", "d435i_joint"),
                      ("link", "d435i_depth_optical_frame"), ("joint", "d435i_depth_optical_joint")]:
        for e in root.findall(f"{tag}[@name='{name}']"):
            root.remove(e)

    # D435i physical dimensions (from Intel realsense-ros _d435.urdf.xacro)
    # d435i_link origin = RGB camera position
    # Box center offset: RGB is ~2.5mm from box center (depth_to_color=15mm, depth_py=17.5mm)
    cam_depth = 0.02505   # 25.05mm (x in URDF)
    cam_width = 0.090     # 90mm (y in URDF)
    cam_height = 0.025    # 25mm (z in URDF)
    box_offset_y = 0.0025 # Box center ~2.5mm to Go2's left of RGB

    # D435i link with visual/collision/inertial (Intel official values)
    link = ET.SubElement(root, "link", name="d435i_link")
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=f"0 {box_offset_y:.4f} 0", rpy="0 0 0")
    ET.SubElement(ET.SubElement(vis, "geometry"), "box", size=f"{cam_depth} {cam_width} {cam_height}")
    mat = ET.SubElement(vis, "material", name="aluminum")
    ET.SubElement(mat, "color", rgba="0.5 0.5 0.5 1")
    
    col = ET.SubElement(link, "collision")
    ET.SubElement(col, "origin", xyz=f"0 {box_offset_y:.4f} 0", rpy="0 0 0")
    ET.SubElement(ET.SubElement(col, "geometry"), "box", size=f"{cam_depth} {cam_width} {cam_height}")
    
    iner = ET.SubElement(link, "inertial")
    ET.SubElement(iner, "mass", value="0.072")
    ET.SubElement(iner, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(iner, "inertia", ixx="0.003881243", ixy="0", ixz="0",
                  iyy="0.000498940", iyz="0", izz="0.003879257")

    # D435i joint (RGB position relative to front_camera)
    jnt = ET.SubElement(root, "joint", name="d435i_joint", type="fixed")
    ET.SubElement(jnt, "parent", link=parent)
    ET.SubElement(jnt, "child", link="d435i_link")
    ET.SubElement(jnt, "origin",
                  xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
                  rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")

    # Depth optical frame (z-forward, x-right, y-down)
    # rpy=[-pi/2, 0, -pi/2] converts URDF frame to optical frame
    ET.SubElement(root, "link", name="d435i_depth_optical_frame")
    ojnt = ET.SubElement(root, "joint", name="d435i_depth_optical_joint", type="fixed")
    ET.SubElement(ojnt, "parent", link="d435i_link")
    ET.SubElement(ojnt, "child", link="d435i_depth_optical_frame")
    ET.SubElement(ojnt, "origin",
                  xyz=f"{depth_xyz_urdf[0]:.6f} {depth_xyz_urdf[1]:.6f} {depth_xyz_urdf[2]:.6f}",
                  rpy=f"{depth_rpy[0]:.6f} {depth_rpy[1]:.6f} {depth_rpy[2]:.6f}")

    out = args.output or str(DATA_DIR / "go2_with_d435i.urdf")
    tree.write(out, encoding="utf-8", xml_declaration=True)

    print(f"=== URDF Generated ===")
    print(f"File: {out}")
    print(f"Parent: {parent}")
    print(f"D435i (URDF): xyz=[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")
    print(f"              rpy=[{np.degrees(rpy[0]):.2f}, {np.degrees(rpy[1]):.2f}, {np.degrees(rpy[2]):.2f}] deg")
    print(f"Depth frame:  xyz=[{depth_xyz_urdf[0]:.4f}, {depth_xyz_urdf[1]:.4f}, {depth_xyz_urdf[2]:.4f}]")


def main():
    p = argparse.ArgumentParser(description="D435i hand-eye calibration for Go2")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("capture", help="Capture calibration images")
    c.add_argument("--num_captures", type=int, default=15, help="Number of image pairs")
    c.add_argument("--network", default="eth0", help="Network interface for Go2")

    sub.add_parser("calibrate", help="Run calibration on captured images")

    sub.add_parser("query_extrinsics", help="Query and save D435i depth extrinsics")

    u = sub.add_parser("generate_urdf", help="Generate URDF with calibrated D435i")
    u.add_argument("--base_urdf", required=True, help="Base Go2 URDF file")
    u.add_argument("--output", help="Output URDF path")

    args = p.parse_args()
    cmds = {"capture": cmd_capture, "calibrate": cmd_calibrate,
            "query_extrinsics": cmd_query_extrinsics, "generate_urdf": cmd_generate_urdf}
    cmds.get(args.cmd, lambda _: p.print_help())(args)


if __name__ == "__main__":
    main()
