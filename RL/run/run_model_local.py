#!/usr/bin/env python3
"""
Run PPO Policy on 10 Depth Frames (Stacked Input, No Blender)
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from stable_baselines3 import PPO
import time

# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL_PATH = "models/cnn_policy_model_run_depth_cube_small_cnn_noise_depth9.zip"
DEFAULT_DEPTH_DIR = "depth_images/"
NUM_FRAMES = 10
CHANNELS_FIRST = True  # SB3 CNN policies expect (C,H,W)


# ============================================================
# Core functions
# ============================================================

def load_depth_image(path):
    """Load a single depth image and normalize to float32 [0,1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("Could not read {}".format(path))
    return img.astype(np.float32) / 255.0


def load_depth_stack(depth_dir, num_frames=NUM_FRAMES):
    """
    Load the most recent N depth frames from a directory.
    Returns stacked tensor of shape (1, num_frames, H, W).
    """
    depth_dir = Path(depth_dir)
    if not depth_dir.exists():
        raise FileNotFoundError("Depth directory not found: {}".format(depth_dir))

    # List depth image files
    files = sorted(
        [p for p in depth_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".npy"]],
        key=lambda x: x.stat().st_mtime
    )

    if len(files) < num_frames:
        raise ValueError("Not enough frames in {} (found {}, need {})".format(
            depth_dir, len(files), num_frames))

    # Pick last N files
    selected = files[-num_frames:]

    # Load all frames
    frames = []
    for f in selected:
        if f.suffix.lower() == ".npy":
            d = np.load(str(f)).astype(np.float32)
        else:
            d = load_depth_image(f)
        if d.ndim == 3:
            d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        frames.append(d)

    # Stack -> (num_frames, H, W)
    dstack = np.stack(frames, axis=0)

    # Add batch dimension -> (1, num_frames, H, W)
    if not CHANNELS_FIRST:
        dstack = np.transpose(dstack, (1, 2, 0))  # (H, W, num_frames)
        dstack = dstack[None, ...]
    else:
        dstack = dstack[None, ...]  # already (1, num_frames, H, W)

    return dstack.astype(np.float32)


def run_policy(model_path, depth_stack):
    """Load PPO model and run it on stacked frames."""
    print("🔁 Loading model from:", model_path)
    model = PPO.load(str(model_path))

    print("🧠 Running policy on stacked frames...")
    action, _ = model.predict(depth_stack, deterministic=True)
    return np.array(action).flatten()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run PPO on stacked depth frames.")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to PPO model (.zip)")
    parser.add_argument("--depth_dir", "-d", type=str, default=None, help="Path to directory with depth frames")
    args = parser.parse_args()

    model_path = Path(args.model or DEFAULT_MODEL_PATH)
    depth_dir = Path(args.depth_dir or DEFAULT_DEPTH_DIR)

    if not model_path.exists():
        print("❌ Model not found:", model_path)
        sys.exit(1)
    if not depth_dir.exists():
        print("❌ Depth folder not found:", depth_dir)
        sys.exit(1)
    counter = 0
    total_inference_time = 0
    while counter < 20:
        start = time.time()    
        depth_stack = load_depth_stack(depth_dir)
        start_action = time.time()
        action = run_policy(model_path, depth_stack)
        inference_time = time.time() - start
        total_inference_time += inference_time
        counter += 1
    
    print('Average computation time ', total_inference_time / (counter+1))
    if action.size == 3:
        vx, vy, vz = action
        print("\n✅ PPO action (vx, vy, vz): %.4f, %.4f, %.4f" % (vx, vy, vz))
    else:
        print("\n✅ PPO action:", action)


if __name__ == "__main__":
    main()
