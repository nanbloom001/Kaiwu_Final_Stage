#!/usr/bin/env python3
"""
Hand-eye calibration for D435i relative to Go2 base.

Solves the AX=XB problem to find the transformation from D435i to Go2 base,
using the Go2's front camera as the reference.

Usage:
    python hand_eye_calibration.py --data_dir ./calibration_data --go2_urdf /path/to/go2.urdf --output calibration_result.json
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_go2_front_camera_transform(urdf_path: str) -> np.ndarray:
    """
    Parse Go2 URDF to extract front_camera_joint transformation.
    
    Args:
        urdf_path: Path to Go2 URDF file
        
    Returns:
        4x4 transformation matrix from front_camera to base
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Find front_camera_joint
    for joint in root.findall('joint'):
        if joint.get('name') == 'front_camera_joint':
            origin = joint.find('origin')
            if origin is not None:
                xyz_str = origin.get('xyz', '0 0 0')
                rpy_str = origin.get('rpy', '0 0 0')
                
                xyz = np.array([float(x) for x in xyz_str.split()])
                rpy = np.array([float(x) for x in rpy_str.split()])
                
                # Build transformation matrix
                rot = R.from_euler('xyz', rpy)
                T = np.eye(4)
                T[:3, :3] = rot.as_matrix()
                T[:3, 3] = xyz
                
                return T
    
    raise ValueError(f"front_camera_joint not found in URDF: {urdf_path}")


class HandEyeCalibrator:
    """Hand-eye calibration solver for D435i to Go2 base transformation."""
    
    # Default Go2 front camera intrinsics (should be obtained from Unitree SDK)
    DEFAULT_GO2_INTRINSICS = {
        "fx": 400.0,
        "fy": 400.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480
    }
    
    # D435i default intrinsics (should be obtained from RealSense SDK)
    DEFAULT_D435I_INTRINSICS = {
        "fx": 615.0,
        "fy": 615.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480
    }
    
    def __init__(self, 
                 chessboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 0.025,
                 go2_urdf_path: Optional[str] = None):
        """
        Args:
            chessboard_size: Inner corners (cols, rows)
            square_size: Size of each square in meters
            go2_urdf_path: Path to Go2 URDF file for reading front_camera_joint transform
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Generate 3D object points
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 
                                    0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Camera matrices
        self.K_go2 = self._intrinsics_to_matrix(self.DEFAULT_GO2_INTRINSICS)
        self.K_d435i = self._intrinsics_to_matrix(self.DEFAULT_D435I_INTRINSICS)
        
        # Go2 front camera to base transformation (from URDF)
        if go2_urdf_path:
            self.T_go2_base = parse_go2_front_camera_transform(go2_urdf_path)
            xyz = self.T_go2_base[:3, 3]
            print(f"Loaded Go2 front_camera transform from URDF:")
            print(f"  xyz: [{xyz[0]:.5f}, {xyz[1]:.5f}, {xyz[2]:.5f}]")
        else:
            # Fallback: use values from official Go2 URDF
            # front_camera_joint: xyz="0.32715 -0.00003 0.04297" rpy="0 0 0"
            self.T_go2_base = np.array([
                [1, 0, 0, 0.32715],
                [0, 1, 0, -0.00003],
                [0, 0, 1, 0.04297],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            print("WARNING: No URDF provided, using default Go2 front_camera values")
        
    def _intrinsics_to_matrix(self, intrinsics: dict) -> np.ndarray:
        """Convert intrinsics dict to camera matrix."""
        return np.array([
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def set_go2_intrinsics(self, intrinsics: dict):
        """Set Go2 camera intrinsics from Unitree SDK."""
        self.K_go2 = self._intrinsics_to_matrix(intrinsics)
        
    def set_d435i_intrinsics(self, intrinsics: dict):
        """Set D435i intrinsics from RealSense SDK."""
        self.K_d435i = self._intrinsics_to_matrix(intrinsics)
        
    def set_go2_to_base_transform(self, T: np.ndarray):
        """Set the Go2 front camera to base transformation from URDF."""
        self.T_go2_base = T
        
    def detect_chessboard(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Detect chessboard corners in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags)
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
        return ret, corners
    
    def estimate_pose(self, corners: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate board pose from detected corners."""
        dist_coeffs = np.zeros(5)  # Assume no distortion for simplicity
        
        ret, rvec, tvec = cv2.solvePnP(self.objp, corners, K, dist_coeffs)
        
        if not ret:
            raise RuntimeError("Failed to solve PnP")
        
        return rvec.flatten(), tvec.flatten()
    
    def rvec_tvec_to_matrix(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Convert rotation vector and translation to 4x4 transformation matrix."""
        R_mat, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = tvec
        return T
    
    def process_image_pair(self, go2_image: np.ndarray, d435i_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a pair of images to get board poses in both camera frames."""
        # Detect in Go2 image
        ret_go2, corners_go2 = self.detect_chessboard(go2_image)
        if not ret_go2:
            raise ValueError("Chessboard not found in Go2 image")
        
        # Detect in D435i image
        ret_d435i, corners_d435i = self.detect_chessboard(d435i_image)
        if not ret_d435i:
            raise ValueError("Chessboard not found in D435i image")
        
        # Estimate poses
        rvec_go2, tvec_go2 = self.estimate_pose(corners_go2, self.K_go2)
        rvec_d435i, tvec_d435i = self.estimate_pose(corners_d435i, self.K_d435i)
        
        T_board_go2 = self.rvec_tvec_to_matrix(rvec_go2, tvec_go2)
        T_board_d435i = self.rvec_tvec_to_matrix(rvec_d435i, tvec_d435i)
        
        return T_board_go2, T_board_d435i
    
    def solve_hand_eye(self, 
                       T_board_go2_list: List[np.ndarray],
                       T_board_d435i_list: List[np.ndarray]) -> np.ndarray:
        """
        Solve the hand-eye calibration problem.
        
        Given:
            - T_go2_base: Go2 camera to base (known from URDF)
            - T_board_go2[i]: Board to Go2 camera (computed from detection)
            - T_board_d435i[i]: Board to D435i camera (computed from detection)
        
        Find:
            - T_d435i_base: D435i to base
        
        Relationship:
            T_board_base = T_go2_base @ T_board_go2
            T_board_base = T_d435i_base @ T_board_d435i
            
        Therefore:
            T_go2_base @ T_board_go2 = T_d435i_base @ T_board_d435i
            T_d435i_base = T_go2_base @ T_board_go2 @ inv(T_board_d435i)
        """
        # Compute T_d435i_base for each pair
        T_estimates = []
        
        for T_board_go2, T_board_d435i in zip(T_board_go2_list, T_board_d435i_list):
            # T_board_base = T_go2_base @ T_board_go2
            T_board_base = self.T_go2_base @ T_board_go2
            
            # T_d435i_base = T_board_base @ inv(T_board_d435i)
            T_d435i_base = T_board_base @ np.linalg.inv(T_board_d435i)
            T_estimates.append(T_d435i_base)
        
        # Average the estimates
        T_d435i_base = self._average_transforms(T_estimates)
        
        return T_d435i_base
    
    def _average_transforms(self, transforms: List[np.ndarray]) -> np.ndarray:
        """Average multiple transformation matrices."""
        # Average translations
        translations = np.array([T[:3, 3] for T in transforms])
        avg_translation = np.mean(translations, axis=0)
        
        # Average rotations using quaternions
        rotations = [R.from_matrix(T[:3, :3]) for T in transforms]
        quats = np.array([r.as_quat() for r in rotations])
        
        # Simple quaternion averaging (works well for close rotations)
        avg_quat = np.mean(quats, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)  # Normalize
        
        avg_rotation = R.from_quat(avg_quat).as_matrix()
        
        # Construct result
        T_avg = np.eye(4)
        T_avg[:3, :3] = avg_rotation
        T_avg[:3, 3] = avg_translation
        
        return T_avg
    
    def compute_reprojection_error(self,
                                   T_d435i_base: np.ndarray,
                                   T_board_go2_list: List[np.ndarray],
                                   T_board_d435i_list: List[np.ndarray]) -> float:
        """Compute average reprojection error to validate calibration."""
        errors = []
        
        for T_board_go2, T_board_d435i in zip(T_board_go2_list, T_board_d435i_list):
            # Compute board in base frame via Go2
            T_board_base_go2 = self.T_go2_base @ T_board_go2
            
            # Compute board in base frame via D435i
            T_board_base_d435i = T_d435i_base @ T_board_d435i
            
            # Compute position error
            pos_error = np.linalg.norm(T_board_base_go2[:3, 3] - T_board_base_d435i[:3, 3])
            errors.append(pos_error)
        
        return np.mean(errors)
    
    def calibrate_from_directory(self, data_dir: str) -> dict:
        """Run calibration from a directory of captured images."""
        data_path = Path(data_dir)
        go2_dir = data_path / "go2_front"
        d435i_dir = data_path / "d435i_rgb"
        
        if not go2_dir.exists() or not d435i_dir.exists():
            raise FileNotFoundError(f"Expected directories: {go2_dir}, {d435i_dir}")
        
        # Match image pairs by index
        go2_images = sorted(go2_dir.glob("*.png"))
        d435i_images = sorted(d435i_dir.glob("*.png"))
        
        if len(go2_images) != len(d435i_images):
            raise ValueError("Mismatched number of images")
        
        print(f"Processing {len(go2_images)} image pairs...")
        
        T_board_go2_list = []
        T_board_d435i_list = []
        valid_pairs = 0
        
        for go2_path, d435i_path in zip(go2_images, d435i_images):
            try:
                go2_img = cv2.imread(str(go2_path))
                d435i_img = cv2.imread(str(d435i_path))
                
                T_board_go2, T_board_d435i = self.process_image_pair(go2_img, d435i_img)
                T_board_go2_list.append(T_board_go2)
                T_board_d435i_list.append(T_board_d435i)
                valid_pairs += 1
                print(f"  Processed: {go2_path.name}")
                
            except Exception as e:
                print(f"  Skipped {go2_path.name}: {e}")
        
        if valid_pairs < 3:
            raise RuntimeError(f"Need at least 3 valid pairs, got {valid_pairs}")
        
        print(f"\nSolving hand-eye calibration with {valid_pairs} pairs...")
        T_d435i_base = self.solve_hand_eye(T_board_go2_list, T_board_d435i_list)
        
        # Validate
        error = self.compute_reprojection_error(T_d435i_base, T_board_go2_list, T_board_d435i_list)
        print(f"Average position error: {error*1000:.2f} mm")
        
        # Extract pose parameters
        rotation = R.from_matrix(T_d435i_base[:3, :3])
        rpy = rotation.as_euler('xyz', degrees=False)
        xyz = T_d435i_base[:3, 3]
        
        result = {
            "transform_matrix": T_d435i_base.tolist(),
            "xyz": xyz.tolist(),
            "rpy": rpy.tolist(),
            "rpy_degrees": np.degrees(rpy).tolist(),
            "position_error_mm": error * 1000,
            "num_valid_pairs": valid_pairs
        }
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Hand-eye calibration for D435i")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing calibration images")
    parser.add_argument("--output", type=str, default="calibration_result.json",
                       help="Output JSON file for calibration result")
    parser.add_argument("--go2_urdf", type=str, 
                       default="/home/unitreeedu/tencent_2026/unitree_ros/robots/go2_description/urdf/go2_description.urdf",
                       help="Path to Go2 URDF file (for reading front_camera_joint transform)")
    parser.add_argument("--chessboard", type=str, default="9x6",
                       help="Chessboard size (inner corners), e.g., 9x6")
    parser.add_argument("--square_size", type=float, default=0.025,
                       help="Chessboard square size in meters")
    args = parser.parse_args()
    
    # Parse chessboard size
    cols, rows = map(int, args.chessboard.split('x'))
    
    calibrator = HandEyeCalibrator(
        chessboard_size=(cols, rows),
        square_size=args.square_size,
        go2_urdf_path=args.go2_urdf
    )
    
    result = calibrator.calibrate_from_directory(args.data_dir)
    
    # Save result
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nCalibration result saved to: {args.output}")
    print(f"\nD435i position (xyz): {result['xyz']}")
    print(f"D435i orientation (rpy deg): {result['rpy_degrees']}")


if __name__ == "__main__":
    main()
