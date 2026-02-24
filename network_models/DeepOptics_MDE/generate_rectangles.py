import numpy as np
import cv2
import os

def generate_rectangles(num_samples=1000, img_size=512, num_depth_bins=12, min_depth=0.1, max_depth=1.0):
    """
    Generate synthetic Rectangles dataset for depth estimation.
    - num_samples: Number of images to generate.
    - img_size: Image resolution (e.g., 512 for 512x512).
    - num_depth_bins: Number of depth bins (paper uses 12).
    - min_depth, max_depth: Depth range in meters (inverse depth spacing).
    Saves RGB images (white rectangles on black) and depth maps.
    """
    # Create output directory
    os.makedirs('data/rectangles', exist_ok=True)
    
    # Depth bins: Linear in inverse depth (1/z) as per paper
    depth_bins = np.linspace(1/max_depth, 1/min_depth, num_depth_bins)  # e.g., 1/1 to 1/0.1
    depth_bins = 1 / depth_bins  # Convert to depth (z)
    
    for i in range(num_samples):
        # Initialize black RGB image
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        # Initialize depth map with random depth from bins
        depth_value = np.random.choice(depth_bins)
        depth_map = np.ones((img_size, img_size)) * depth_value
        
        # Add random white rectangle
        max_size = img_size // 2
        w, h = np.random.randint(10, max_size, 2)  # Random width/height
        x, y = np.random.randint(0, img_size - w), np.random.randint(0, img_size - h)  # Random top-left
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), -1)  # White filled rectangle
        
        # Save RGB image and depth map
        cv2.imwrite(f'data/rectangles/rgb_{i}.png', img)
        np.save(f'data/rectangles/depth_{i}.npy', depth_map)
    
    print(f"Generated {num_samples} samples in data/rectangles/")

if __name__ == "__main__":
    generate_rectangles(num_samples=1000, img_size=512, num_depth_bins=12)