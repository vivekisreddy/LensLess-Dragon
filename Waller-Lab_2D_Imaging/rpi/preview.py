from picamera2 import Picamera2
import cv2
import os
import datetime
import time

picam2 = Picamera2()

# FULL SENSOR MODE
config = picam2.create_still_configuration(
    main={"format": "RGB888", "size": (3280, 2464)}
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

print("Full sensor preview. Press ENTER to capture. q to quit.")

try:
    while True:
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize for display ONLY (so OpenCV window fits screen)
        display = cv2.resize(frame_bgr, (1280, 960))
        cv2.imshow("Full Sensor Preview (Scaled Display)", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == 13:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = os.path.join(save_folder, f"psf_{timestamp}.png")
            cv2.imwrite(filename, frame_bgr)
            print("Saved:", filename)

finally:
    picam2.stop()
    cv2.destroyAllWindows()

