import cv2
import os

def video_to_images_by_time(video_path, output_folder, interval_ms):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # hedre is the frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    print(f"Video FPS: {fps}, Total frames: {total_frames}, Duration: {duration_sec:.2f}s")

    interval_sec = interval_ms / 1000.0
    
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    t = 0.0
    while t <= duration_sec:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        image_path = os.path.join(output_folder, f"frame_{count+1}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"SAVED YAY: {image_path}")
        count += 1
        t += interval_sec  
    
    cap.release()
    print(f"\nDONE! Extracted {count} images.")

video_path = r"C:\Users\Janu\Desktop\wpi\academics\senior\mqp\video to images\test1.MOV"
output_folder = r"C:\Users\Janu\Desktop\wpi\academics\senior\mqp\test image"

video_to_images_by_time(video_path, output_folder, interval_ms=50)