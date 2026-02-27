# synthetic_test_generator.py
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

OUT_DIR = "synthetic"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- basic FFT conv -------------
def conv_fft_same(I: np.ndarray, p: np.ndarray) -> np.ndarray:
    H, W = I.shape
    h, w = p.shape
    padH, padW = H + h - 1, W + w - 1

    FI = np.fft.rfft2(I, s=(padH, padW))
    Fp = np.fft.rfft2(p, s=(padH, padW))
    Y  = np.fft.irfft2(FI * Fp, s=(padH, padW))

    sy = (h - 1) // 2
    sx = (w - 1) // 2
    return Y[sy:sy+H, sx:sx+W]

# ------------- synthetic chart -------------

def make_synthetic_chart(H=512, W=768, square_size=20):
    """
    Create: left 3x12 checkerboard, center noise, right 3x12 checkerboard.
    """
    img = np.ones((H, W), dtype=np.float32) * 0.8  # light gray background

    # vertical center
    rows = 12
    cols = 3
    board_h = rows * square_size
    board_w = cols * square_size

    cy = H // 2
    y0 = cy - board_h // 2

    # left board
    x0_left = 20
    for r in range(rows):
        for c in range(cols):
            y = y0 + r * square_size
            x = x0_left + c * square_size
            if (r + c) % 2 == 0:
                img[y:y+square_size, x:x+square_size] = 0.2  # dark square
            else:
                img[y:y+square_size, x:x+square_size] = 0.9  # bright square

    # right board
    x0_right = W - board_w - 20
    for r in range(rows):
        for c in range(cols):
            y = y0 + r * square_size
            x = x0_right + c * square_size
            if (r + c) % 2 == 0:
                img[y:y+square_size, x:x+square_size] = 0.2
            else:
                img[y:y+square_size, x:x+square_size] = 0.9

    # center noise block
    noise_h = board_h
    noise_w = W - (x0_left + board_w + 20) - (W - x0_right) + 20
    y_noise = y0
    x_noise = x0_left + board_w + 20

    noise = 0.5 + 0.2 * np.random.randn(noise_h, noise_w).astype(np.float32)
    noise = np.clip(noise, 0.0, 1.0)
    img[y_noise:y_noise+noise_h, x_noise:x_noise+noise_w] = noise

    return img

def make_gaussian_psf(size=21, sigma=3.0):
    h = w = size
    yy, xx = np.mgrid[:h, :w]
    cy, cx = (h-1)/2.0, (w-1)/2.0
    psf = np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*sigma*sigma)).astype(np.float32)
    psf = np.maximum(psf, 0.0)
    psf /= psf.sum()
    return psf

def main():
    H, W = 512, 768
    square_size = 20

    sharp = make_synthetic_chart(H, W, square_size=square_size)
    psf_true = make_gaussian_psf(size=21, sigma=3.0)
    blur = conv_fft_same(sharp, psf_true)

    # rescale to [0,255] uint8
    def to_u8(x):
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0 + 0.5).astype(np.uint8)

    sharp_u8 = to_u8(sharp)
    blur_u8  = to_u8(blur)

    cv2.imwrite(os.path.join(OUT_DIR, "0_sharp.png"), sharp_u8)
    cv2.imwrite(os.path.join(OUT_DIR, "0_blur.png"), blur_u8)

    np.save(os.path.join(OUT_DIR, "0_psf_true.npy"), psf_true)

    # visualize true PSF
    plt.figure(figsize=(5,5))
    plt.imshow(psf_true, cmap="gray", interpolation="bicubic")
    plt.title("psf_true")
    plt.axis("off")
    plt.savefig(os.path.join(OUT_DIR, "0_psf_true.png"),
                dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print("Synthetic data written to ./synthetic:")
    print("  0_sharp.png, 0_blur.png, 0_psf_true.npy, 0_psf_true.png")

if __name__ == "__main__":
    main()
