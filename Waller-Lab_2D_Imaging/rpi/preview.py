import cv2
import time

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Press ENTER to capture image")
print("Press ESC to exit")

img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Preview", frame)

    key = cv2.waitKey(1) & 0xFF

    # ESC to exit
    if key == 27:
        break

    # ENTER to capture
    if key == 13 or key == 10:
        filename = f"capture_{img_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image saved: {filename}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()