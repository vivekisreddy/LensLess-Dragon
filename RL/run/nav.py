#!/usr/bin/env python3
# nav.py — PPO model → Pixhawk via MavlinkController queue
# BODY frame (FRD). Velocity-only mask. +0.1s post-send delay.

import os, sys, time, threading, queue, argparse
import numpy as np
from pymavlink import mavutil

# ===== PPO stream module =====
from ppo_stream import PPOStream

# ===== Pixhawk parameters =====
FLIGHT_MODES = {
    "STABILIZE": 0,
    "ALT_HOLD": 2,
    "GUIDED": 4,
    "LOITER": 5,
    "RTL": 6,
    "LAND": 9
}

# Masks: ignore POS, ACC, YAW, YAW_RATE; use VELOCITIES only
MAVLINK_SET_POS_TYPE_MASK_POS_IGNORE = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
)
MAVLINK_SET_POS_TYPE_MASK_VEL_USE = 0  # (we'll NOT set VX/VY/VZ ignore bits)
MAVLINK_SET_POS_TYPE_MASK_ACC_IGNORE = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
)
MAVLINK_SET_POS_TYPE_MASK_YAW_IGNORE = mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
MAVLINK_SET_POS_TYPE_MASK_YAW_RATE_IGNORE = mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE

TYPE_MASK_VEL_ONLY = (
    MAVLINK_SET_POS_TYPE_MASK_POS_IGNORE |
    MAVLINK_SET_POS_TYPE_MASK_ACC_IGNORE |
    MAVLINK_SET_POS_TYPE_MASK_YAW_IGNORE |
    MAVLINK_SET_POS_TYPE_MASK_YAW_RATE_IGNORE |
    MAVLINK_SET_POS_TYPE_MASK_VEL_USE  # keep velocity bits enabled
)

# Global lock (as referenced in your code)
perception_lock = threading.Lock()


class MavlinkController:
    def __init__(self, connection_string='/dev/ttyACM0', connection_baudrate=115200):
        self.connection_string = connection_string
        self.connection_baudrate = connection_baudrate
        self.conn = None
        self.command_queue = queue.Queue()
        self.running = True
        self.mavlink_thread = None

    def initialize(self):
        os.environ["MAVLINK20"] = "1"
        sys.path.append("/usr/local/lib/")
        self.setup_mavlink_connection()
        self.start_command_thread()

    def setup_mavlink_connection(self):
        self.conn = mavutil.mavlink_connection(
            device=str(self.connection_string),
            autoreconnect=True,
            source_system=1,
            source_component=93,
            baud=self.connection_baudrate,
            force_connected=True,
        )
        print("[mav] Waiting for heartbeat…")
        self.conn.wait_heartbeat()
        print(f"[mav] Connected to {self.connection_string} @ {self.connection_baudrate}")

    def start_command_thread(self):
        self.mavlink_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.mavlink_thread.start()

    def process_commands(self):
        while self.running:
            try:
                command = self.command_queue.get(timeout=1)
                if command is None:
                    continue
                vx, vy, vz, yaw, duration = command

                with perception_lock:
                    self.send_guided_message(vx, vy, vz, yaw)
                    print(f"[mav] SENT: vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} yaw={yaw:.2f} "
                          f"for {duration:.2f}s")
                    # Hold this command for duration…
                    time.sleep(max(duration, 0.0))
                    # …and add the requested 0.1 s delay AFTER the command is passed
                    time.sleep(0.1)

            except queue.Empty:
                pass

    def set_mode_guided(self):
        self.conn.mav.set_mode_send(
            self.conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            FLIGHT_MODES["GUIDED"]
        )

    def arm(self, arm: bool = True):
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1 if arm else 0, 0, 0, 0, 0, 0, 0
        )
        self.conn.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)

    def send_guided_message(self, vx=0, vy=0, vz=0, yaw=0):
        """
        BODY_OFFSET_NED: x forward, y right, z down; yaw is body-relative.
        We IGNORE yaw & yaw_rate via mask; only velocities are used.
        """
        self.conn.mav.set_position_target_local_ned_send(
            int(round(time.time() * 1e3)) % 2**32,    # time_boot_ms (approx)
            self.conn.target_system,
            0,  # target component
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            TYPE_MASK_VEL_ONLY,
            0, 0, 0,        # x,y,z (ignored)
            vx, vy, vz,     # velocities (used)
            0, 0, 0,        # accelerations (ignored)
            0, 0            # yaw, yaw_rate (ignored per mask)
        )

    def enqueue_command(self, vx, vy, vz, yaw, duration=0.1):
        self.command_queue.put((vx, vy, vz, yaw, duration))

    def stop(self):
        self.running = False
        self.command_queue.put(None)
        if self.mavlink_thread:
            self.mavlink_thread.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="/dev/ttyACM0", help="e.g., /dev/ttyACM0 or udp:127.0.0.1:14550")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--rate_hz", type=float, default=10.0, help="command rate")
    ap.add_argument("--model", default="models/cnn_policy_model_run_depth_cube_small_cnn_noise_depth9.zip",
                    help="Path to PPO model file")
    ap.add_argument("--show_viz", action="store_true", help="Show visualization window")
    args = ap.parse_args()

    # Init MAVLink
    mc = MavlinkController(connection_string=args.conn, connection_baudrate=args.baud)
    mc.initialize()
    mc.set_mode_guided()
    time.sleep(0.5)
    mc.arm(True)
    time.sleep(0.5)

    # Initialize PPO stream
    ppo_stream = PPOStream(
        model_path=args.model,
        width=224,
        height=168,
        num_frames=10,
        channels_first=True,
        max_depth_m=3.0,
        vx_scale=0.5,
        vy_scale=1.0,
        vz_scale=0.2,
        show_window=args.show_viz
    )

    period = 1.0 / max(args.rate_hz, 1e-6)
    print(f"[nav] Loop @ {args.rate_hz:.1f} Hz. Ctrl+C to stop.")
    try:
        while True:
            t0 = time.time()
            
            # Get new frame and add to buffer
            ppo_stream.get_frame()
            
            # Predict action from buffer
            action = ppo_stream.predict()
            if action is not None:
                vx, vy, vz, yaw = action
                print(f"[nav] PPO: vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}, yaw={yaw:.3f}", end="\r")
            else:
                # Buffer not full yet, use zero velocity
                vx, vy, vz, yaw = 0.0, 0.0, 0.0, 0.0

            # Enqueue with duration equal to the period
            mc.enqueue_command(vx, vy, vz, yaw=yaw, duration=period)

            # pace this producer loop roughly at rate_hz
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[nav] Stopping… sending zero velocity & shutting down thread.")
        for _ in range(5):
            mc.enqueue_command(0.0, 0.0, 0.0, yaw=0.0, duration=0.05)
        time.sleep(1.0)
        ppo_stream.stop()
        mc.stop()


if __name__ == "__main__":
    main()
