#!/usr/bin/env python3
"""
Synchronous dual-camera image capture for D435i calibration.

This script captures synchronized images from:
1. Go2's built-in front camera (via Unitree_sdk2)
2. RealSense D435i RGB camera

Usage:
    python capture_images.py --output_dir ./calibration_data --num_captures 20
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Optional imports - will be checked at runtime
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available")

try:
    from unitree_sdk2py.core.channel import ChannelFactory
    from unitree_sdk2py.go2.video.video_client import VideoClient
    UNITREE_SDK_AVAILABLE = True
except ImportError:
    UNITREE_SDK_AVAILABLE = False
    print("Warning: unitree_sdk2py not available")


class DualCameraCapture:
    """Synchronized capture from Go2 front camera and D435i."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.go2_dir = self.output_dir / "go2_front"
        self.d435i_dir = self.output_dir / "d435i_rgb"
        self.go2_dir.mkdir(exist_ok=True)
        self.d435i_dir.mkdir(exist_ok=True)
        
        self.go2_client = None
        self.rs_pipeline = None
        self.capture_count = 0
        
    def init_go2_camera(self, network_interface: str = "eth0"):
        """Initialize Go2 front camera via Unitree SDK."""
        if not UNITREE_SDK_AVAILABLE:
            raise RuntimeError("unitree_sdk2py is not installed")
        
        ChannelFactory.Instance().Init(0, network_interface)
        self.go2_client = VideoClient()
        self.go2_client.Init()
        print("Go2 front camera initialized")
        
    def init_d435i(self):
        """Initialize RealSense D435i pipeline."""
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not installed")
        
        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure RGB stream (640x480 for calibration)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.rs_pipeline.start(config)
        # Allow auto-exposure to stabilize
        time.sleep(2.0)
        print("D435i initialized")
        
    def capture_go2_frame(self) -> np.ndarray:
        """Capture a single frame from Go2 front camera."""
        if self.go2_client is None:
            raise RuntimeError("Go2 camera not initialized")
        
        frame_data = self.go2_client.GetImageSample()
        # Convert to numpy array (format depends on SDK version)
        # This may need adjustment based on actual SDK output format
        image = np.frombuffer(frame_data.data, dtype=np.uint8)
        image = image.reshape((frame_data.height, frame_data.width, 3))
        return image
    
    def capture_d435i_frame(self) -> np.ndarray:
        """Capture a single frame from D435i RGB camera."""
        if self.rs_pipeline is None:
            raise RuntimeError("D435i not initialized")
        
        frames = self.rs_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            raise RuntimeError("Failed to get D435i color frame")
        
        return np.asanyarray(color_frame.get_data())
    
    def capture_synchronized(self) -> tuple:
        """Capture frames from both cameras as close together as possible."""
        # Capture D435i first (usually faster)
        d435i_frame = self.capture_d435i_frame()
        go2_frame = self.capture_go2_frame()
        
        return go2_frame, d435i_frame
    
    def save_capture(self, go2_frame: np.ndarray, d435i_frame: np.ndarray):
        """Save captured frames with matching timestamps."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        go2_path = self.go2_dir / f"go2_{self.capture_count:04d}_{timestamp}.png"
        d435i_path = self.d435i_dir / f"d435i_{self.capture_count:04d}_{timestamp}.png"
        
        cv2.imwrite(str(go2_path), go2_frame)
        cv2.imwrite(str(d435i_path), d435i_frame)
        
        self.capture_count += 1
        print(f"Saved capture {self.capture_count}: {go2_path.name}, {d435i_path.name}")
        
    def preview_and_capture(self, num_captures: int = 20):
        """Interactive capture with preview windows."""
        print(f"\nStarting capture session. Target: {num_captures} captures")
        print("Press 'c' to capture, 'q' to quit\n")
        
        while self.capture_count < num_captures:
            try:
                go2_frame, d435i_frame = self.capture_synchronized()
                
                # Display preview
                preview_go2 = cv2.resize(go2_frame, (640, 480))
                preview_d435i = cv2.resize(d435i_frame, (640, 480))
                
                combined = np.hstack([preview_go2, preview_d435i])
                cv2.putText(combined, f"Captures: {self.capture_count}/{num_captures}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined, "Go2 Front", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(combined, "D435i RGB", (650, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                cv2.imshow("Dual Camera Preview - Press 'c' to capture", combined)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    self.save_capture(go2_frame, d435i_frame)
                elif key == ord('q'):
                    print("Capture session ended by user")
                    break
                    
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        print(f"\nCapture complete. Total captures: {self.capture_count}")
        
    def cleanup(self):
        """Release camera resources."""
        if self.rs_pipeline:
            self.rs_pipeline.stop()
        print("Cameras released")


def main():
    parser = argparse.ArgumentParser(description="Dual camera capture for calibration")
    parser.add_argument("--output_dir", type=str, default="./calibration_data",
                       help="Output directory for captured images")
    parser.add_argument("--num_captures", type=int, default=20,
                       help="Number of image pairs to capture")
    parser.add_argument("--network", type=str, default="eth0",
                       help="Network interface for Go2 connection")
    args = parser.parse_args()
    
    capture = DualCameraCapture(args.output_dir)
    
    try:
        print("Initializing cameras...")
        capture.init_go2_camera(args.network)
        capture.init_d435i()
        
        capture.preview_and_capture(args.num_captures)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        capture.cleanup()


if __name__ == "__main__":
    main()
