#!/usr/bin/env python3
"""
Live PPO inference from Intel RealSense depth stream.
Visualizes PPO actions with arrows: X (left/right) and Z (up/down).
Prints full action (vx, vy, vz, yaw) to the terminal each frame.
Depth values are clipped to 3 meters and normalized to [0, 1].
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
from stable_baselines3 import PPO

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "models/cnn_policy_model_run_depth_cube_small_cnn_noise_depth9.zip"
WIDTH, HEIGHT = 224, 168
NUM_FRAMES = 10
CHANNELS_FIRST = True
SHOW_WINDOW = True
MAX_DEPTH_M = 3.0

# Arrow visualization configs
ARROW_SCALE_PX_PER_UNIT = 80.0
ARROW_MAX_LEN_PX = 60
ARROW_THICKNESS = 2
ARROW_TIP = 0.3

# ============================================================
# Helpers
# ============================================================
def clamp_arrow(dx_px: float, dy_px: float, max_len: float):
    """Clamp arrow length to avoid overshooting the screen."""
    length = np.hypot(dx_px, dy_px)
    if length > max_len and length > 1e-6:
        scale = max_len / length
        dx_px *= scale
        dy_px *= scale
    return int(round(dx_px)), int(round(dy_px))

# ============================================================
# Initialize model and RealSense
# ============================================================
print("🔁 Loading PPO model...")
model = PPO.load(MODEL_PATH)
print("✅ Model loaded:", MODEL_PATH)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
print("🎥 RealSense stream started")

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"📏 Depth scale: {depth_scale:.6f} meters/unit")

frame_buffer = []
last_action = None

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert to meters
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = np.clip(depth_raw * depth_scale, 0.0, MAX_DEPTH_M)

        # Resize and normalize
        depth_resized = cv2.resize(depth_m, (WIDTH, HEIGHT),  interpolation=cv2.INTER_AREA)
        depth_norm = (depth_resized / MAX_DEPTH_M).astype(np.float32)

        # Maintain temporal stack
        frame_buffer.append(depth_norm)
        if len(frame_buffer) > NUM_FRAMES:
            del frame_buffer[0]

        vx = vy = vz = yaw = None
        if len(frame_buffer) == NUM_FRAMES:
            obs = np.stack(frame_buffer, axis=0)[None, ...]
            if not CHANNELS_FIRST:
                obs = np.transpose(obs, (0, 2, 3, 1))

            action, _ = model.predict(obs, deterministic=True)
            action = np.array(action).flatten()
            last_action = action.copy()

            # Unpack (adjust indices if your PPO uses different order)
            if len(action) >= 4:
                vx, vy, vz, yaw = action[:4]
            elif len(action) == 3:
                vx, vy, vz = action
            elif len(action) == 2:
                vx, vz = action

            vx = 0.5 * vx
            vy = 1.0 * vy
            vz = 0.2 * vz  # Scale down

            # Print all available values
            print(f"→ vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}, yaw={yaw:.3f}", end="\r")

        # Visualization
        if SHOW_WINDOW:
            depth_vis = cv2.convertScaleAbs(depth_resized, alpha=255.0 / MAX_DEPTH_M)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            h, w = depth_colormap.shape[:2]
            cx, cy = w // 2, h // 2

            # Crosshair
            cv2.drawMarker(depth_colormap, (cx, cy), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

            # Draw single combined X/Z arrow
            if last_action is not None and (vx is not None and vz is not None):
                dx = ARROW_SCALE_PX_PER_UNIT * float(vx)
                dy = -ARROW_SCALE_PX_PER_UNIT * float(vz)  # +vz = up
                dx, dy = clamp_arrow(dx, dy, ARROW_MAX_LEN_PX)

                end_x = cx + dx
                end_y = cy + dy

                # Draw one arrow representing combined X/Z motion
                cv2.arrowedLine(depth_colormap, (cx, cy), (end_x, end_y),
                                color=(0, 255, 255), thickness=ARROW_THICKNESS, tipLength=ARROW_TIP)

            cv2.imshow("Depth (224x168, 3 m clip + X/Z arrows)", depth_colormap)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        time.sleep(0.03)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\n🛑 Stream stopped")
