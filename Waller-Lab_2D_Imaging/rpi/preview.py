from picamera2 import Picamera2
import cv2
import os
import datetime
import time
import numpy as np

picam2 = Picamera2()

# Use YUV format (no RGB processing)
config = picam2.create_still_configuration(
    main={"format": "YUV420", "size": (3280, 2464)}
)

picam2.configure(config)
picam2.start()
time.sleep(1)

# Lock controls
picam2.set_controls({
    "AwbEnable": False,
    "AeEnable": False,
    "ExposureTime": 20000,
    "AnalogueGain": 1.0
})

save_folder = "captures"
os.makedirs(save_folder, exist_ok=True)

print("Grayscale full sensor preview. Press ENTER to capture. q to quit.")

try:
    while True:
        frame = picam2.capture_array()

        # Extract ONLY luminance (Y channel)
        height = 2464
        width = 3280
        y_plane = frame[:height, :width]

        # Resize for display only
        display = cv2.resize(y_plane, (1280, 960))
        cv2.imshow("Full Sensor Preview (Grayscale)", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == 13:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = os.path.join(save_folder, f"2nd_capture_{timestamp}.png")
            cv2.imwrite(filename, y_plane)
            print("Saved:", filename)

finally:
    picam2.stop()
    cv2.destroyAllWindows()