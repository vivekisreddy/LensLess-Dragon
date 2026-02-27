import dv_processing as dv
import cv2
import numpy as np

# ---- USER SETTINGS ----
input_file = "dvSave18feb-2026_02_18_15_25_43.aedat4"
accumulation_ms = 10  # frame time window in milliseconds
# ------------------------

recording = dv.io.MonoCameraRecording(input_file)
resolution = recording.getEventResolution()

frame_id = 0
start_time = None
window_us = accumulation_ms * 1000  # microseconds

while recording.isRunning():
    events = recording.getNextEventBatch()
    if events is None:
        continue

    if start_time is None:
        start_time = events.getLowestTime()

    # Create a fresh accumulator for each window
    accumulator = dv.Accumulator(resolution)
    accumulator.setDecayFunction(dv.Accumulator.Decay.NONE)
    accumulator.accept(events)

    current_time = events.getHighestTime()

    if current_time - start_time >= window_us:
        frame = accumulator.generateFrame()
        gray_image = frame.image  # current grayscale image

        # Convert polarity to RGB:
        # ON events -> red, OFF events -> blue
        rgb_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
        on_mask = gray_image > 0
        off_mask = gray_image < 0
        rgb_image[on_mask] = [0, 0, 255]    # Red for ON
        rgb_image[off_mask] = [255, 0, 0]   # Blue for OFF

        cv2.imwrite(f"frame_{frame_id:05d}.png", rgb_image)

        frame_id += 1
        start_time = current_time  # start next window
