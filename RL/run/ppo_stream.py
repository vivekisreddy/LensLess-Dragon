#!/usr/bin/env python3
"""
PPO inference from Intel RealSense depth stream.
Extracted from run_live_stream.py as a reusable class.
You control when to get frames and when to predict.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from stable_baselines3 import PPO


class PPOStream:
    """PPO model with RealSense camera. You control when to get frames and predict."""
    
    def __init__(self, model_path="models/cnn_policy_model_run_depth_cube_small_cnn_noise_depth9.zip",
                 width=224, height=168, num_frames=10, channels_first=True,
                 max_depth_m=3.0, vx_scale=0.5, vy_scale=1.0, vz_scale=0.2,
                 show_window=False):
        """
        Initialize PPO model and RealSense camera.
        
        Args:
            model_path: Path to PPO model file
            width: Target width for depth frames (default: 224)
            height: Target height for depth frames (default: 168)
            num_frames: Number of frames for temporal stacking (default: 10)
            channels_first: Whether to use channels_first format (default: True)
            max_depth_m: Maximum depth in meters for clipping (default: 3.0)
            vx_scale: Scaling factor for vx (default: 0.5)
            vy_scale: Scaling factor for vy (default: 1.0)
            vz_scale: Scaling factor for vz (default: 0.2)
            show_window: Whether to show visualization window (default: False)
        """
        # Load model
        print("🔁 Loading PPO model...")
        self.model = PPO.load(model_path)
        print(f"✅ Model loaded: {model_path}")
        
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        print("🎥 RealSense stream started")
        
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"📏 Depth scale: {self.depth_scale:.6f} meters/unit")
        
        # Store parameters
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.channels_first = channels_first
        self.max_depth_m = max_depth_m
        self.vx_scale = vx_scale
        self.vy_scale = vy_scale
        self.vz_scale = vz_scale
        self.show_window = show_window
        
        # Frame buffer
        self.frame_buffer = []
        
        # For visualization
        self.last_depth_resized = None
        self.last_action = None
        
        # Arrow visualization configs
        self.arrow_scale_px_per_unit = 80.0
        self.arrow_max_len_px = 60
        self.arrow_thickness = 2
        self.arrow_tip = 0.3
    
    def get_frame(self):
        """
        Get one depth frame, process it, and add to buffer.
        Call this whenever you want a new frame.
        
        Returns:
            bool: True if frame was added, False if no frame available
        """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return False
        
        # Convert to meters
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = np.clip(depth_raw * self.depth_scale, 0.0, self.max_depth_m)
        
        # Resize and normalize
        depth_resized = cv2.resize(depth_m, (self.width, self.height))
        depth_norm = (depth_resized / self.max_depth_m).astype(np.float32)
        
        # Save for visualization
        if self.show_window:
            self.last_depth_resized = depth_resized
        
        # Add to buffer
        self.frame_buffer.append(depth_norm)
        if len(self.frame_buffer) > self.num_frames:
            del self.frame_buffer[0]
        
        return True
    
    def predict(self):
        """
        Predict action from current frame buffer.
        Call this after you have enough frames (10 frames).
        
        Returns:
            tuple: (vx, vy, vz, yaw) or None if buffer not full
        """
        if len(self.frame_buffer) < self.num_frames:
            return None
        
        # Stack frames
        obs = np.stack(self.frame_buffer, axis=0)[None, ...]
        if not self.channels_first:
            obs = np.transpose(obs, (0, 2, 3, 1))
        
        # Predict
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.array(action).flatten()
        
        # Unpack
        vx = vy = vz = yaw = 0.0
        if len(action) >= 4:
            vx, vy, vz, yaw = action[:4]
        elif len(action) == 3:
            vx, vy, vz = action
        elif len(action) == 2:
            vx, vz = action
        
        # Apply scaling
        vx = self.vx_scale * vx
        vy = self.vy_scale * vy
        vz = self.vz_scale * vz
        
        # Save for visualization
        if self.show_window:
            self.last_action = (vx, vy, vz, yaw)
            self._visualize()
        
        return (vx, vy, vz, yaw)
    
    def _clamp_arrow(self, dx_px: float, dy_px: float, max_len: float):
        """Clamp arrow length to avoid overshooting the screen."""
        length = np.hypot(dx_px, dy_px)
        if length > max_len and length > 1e-6:
            scale = max_len / length
            dx_px *= scale
            dy_px *= scale
        return int(round(dx_px)), int(round(dy_px))
    
    def _visualize(self):
        """Visualize depth map with action arrows."""
        if not self.show_window or self.last_depth_resized is None:
            return
        
        # Convert depth to colormap
        depth_vis = cv2.convertScaleAbs(self.last_depth_resized, alpha=255.0 / self.max_depth_m)
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        h, w = depth_colormap.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Crosshair
        cv2.drawMarker(depth_colormap, (cx, cy), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
        
        # Draw arrow if action available
        if self.last_action is not None:
            vx, vy, vz, yaw = self.last_action
            if vx is not None and vz is not None:
                dx = self.arrow_scale_px_per_unit * float(vx)
                dy = -self.arrow_scale_px_per_unit * float(vz)  # +vz = up
                dx, dy = self._clamp_arrow(dx, dy, self.arrow_max_len_px)
                
                end_x = cx + dx
                end_y = cy + dy
                
                # Draw arrow representing combined X/Z motion
                cv2.arrowedLine(depth_colormap, (cx, cy), (end_x, end_y),
                                color=(0, 255, 255), thickness=self.arrow_thickness, 
                                tipLength=self.arrow_tip)
        
        cv2.imshow("Depth (224x168, 3 m clip + X/Z arrows)", depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.show_window = False  # Stop visualization on 'q'
    
    def stop(self):
        """Stop the RealSense pipeline."""
        self.pipeline.stop()
        if self.show_window:
            cv2.destroyAllWindows()
        print("🛑 RealSense stream stopped")


if __name__ == "__main__":
    # For testing
    stream = PPOStream()
    try:
        while True:
            stream.get_frame()
            action = stream.predict()
            if action is not None:
                vx, vy, vz, yaw = action
                print(f"→ vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}, yaw={yaw:.3f}", end="\r")
    except KeyboardInterrupt:
        stream.stop()

