import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generate_print_ready_pattern(
        grid_size=40,         # number of random blocks
        block_size_pix=35,    # block size in pixels (will be resized)
        physical_mm=60,       # final size in mm
        dpi=600,              # print DPI
        filename="calibration_pattern_60mm.png"
    ):
    """
    Generates a printable chunky black/white random pattern
    guaranteed to be 60mm x 60mm at the given DPI.
    """

    # Convert mm → pixels
    physical_inches = physical_mm / 25.4
    final_pixels = int(physical_inches * dpi)

    # Build initial random block matrix
    block_matrix = np.random.randint(0, 2, (grid_size, grid_size), dtype=np.uint8)
    img = np.kron(block_matrix, np.ones((block_size_pix, block_size_pix), dtype=np.uint8)) * 255

    # Resize exactly to target physical size
    img_resized = Image.fromarray(img).resize((final_pixels, final_pixels), resample=Image.NEAREST)

    # Save with DPI metadata so it prints at EXACT size
    img_resized.save(filename, dpi=(dpi, dpi))
    print(f"Saved {filename} with physical size {physical_mm}mm x {physical_mm}mm at {dpi} DPI")

    # Show preview on screen
    plt.figure(figsize=(6, 6))
    plt.imshow(img_resized, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()


# Example usage
generate_print_ready_pattern(
    grid_size=70,
    block_size_pix=40,
    physical_mm=60,
    dpi=600,
    filename="calibration_pattern_60mm4.png"
)
