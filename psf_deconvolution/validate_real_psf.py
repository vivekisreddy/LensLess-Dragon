import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import cm

# ---------- FFT "same" convolution ----------
def conv_fft_same(I: np.ndarray, p: np.ndarray) -> np.ndarray:
    H, W = I.shape
    h, w = p.shape
    padH, padW = H + h - 1, W + w - 1

    FI = np.fft.rfft2(I, s=(padH, padW))
    Fp = np.fft.rfft2(p, s=(padH, padW))
    FY = FI * Fp
    Y  = np.fft.irfft2(FY, s=(padH, padW))

    sy = (h - 1) // 2
    sx = (w - 1) // 2
    return Y[sy:sy+H, sx:sx+W]


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.min()
    rng = np.ptp(x)
    if rng < 1e-8:
        return np.zeros_like(x)
    return x / rng


def validate_one_label(sharp_dir, blur_dir, psf_dir, out_dir, lbl):
    # Correct file paths
    sharp_path = os.path.join(sharp_dir, f"{lbl}_sharp.png")
    blur_path  = os.path.join(blur_dir,  f"{lbl}_blur.png")
    psf_path   = os.path.join(psf_dir,   f"{lbl}_psf.npy")

    # Check existence
    missing = []
    if not os.path.exists(sharp_path): missing.append("sharp")
    if not os.path.exists(blur_path):  missing.append("blur")
    if not os.path.exists(psf_path):   missing.append("psf")

    if missing:
        print(f"[{lbl}] missing: {missing} — skipping.")
        return None

    # Load images
    I = imageio.imread(sharp_path).astype(np.float64)
    J = imageio.imread(blur_path).astype(np.float64)
    if I.ndim == 3: I = I[..., 0]
    if J.ndim == 3: J = J[..., 0]
    p = np.load(psf_path).astype(np.float64)

    # Crop sharp/blur to match if needed
    if I.shape != J.shape:
        H = min(I.shape[0], J.shape[0])
        W = min(I.shape[1], J.shape[1])
        I = I[:H, :W]
        J = J[:H, :W]
        print(f"[{lbl}] Cropped to {I.shape}")

    # Reblur using the PSF
    J_hat = conv_fft_same(I, p)

    # Brightness scaling
    alpha = (J.sum() + 1e-8) / (J_hat.sum() + 1e-8)
    J_hat *= alpha

    # Compute metrics
    resid = J_hat - J
    mse = float(np.mean(resid ** 2))
    max_val = float(np.max(J))
    psnr = 10 * np.log10((max_val ** 2) / mse) if mse > 1e-12 else float("inf")

    print(f"[{lbl}] MSE={mse:.4e}, PSNR={psnr:.2f} dB")

    # Normalize for visualization
    I_n    = normalize01(I)
    J_n    = normalize01(J)
    Jhat_n = normalize01(J_hat)
    diff_n = normalize01(np.abs(J_hat - J))
    psf_n  = normalize01(p)

    # Color-graded PSF (jet colormap)
    psf_color = cm.jet(psf_n)  # returns RGBA heatmap

    # ---- Visualization panel (6 images) ----
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    titles = ["Sharp I", "Blur J", "Reblur I⊗p",
              "|J − I⊗p|", "PSF (gray)", "PSF (color)"]
    images = [I_n, J_n, Jhat_n, diff_n, psf_n, psf_color]

    for ax, im, t in zip(axes, images, titles):
        if im.ndim == 3:
            ax.imshow(im)
        else:
            ax.imshow(im, cmap="gray")
        ax.set_title(t, fontsize=9)
        ax.axis("off")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lbl}_validate.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close(fig)

    return mse, psnr


def main():
    SHARP_DIR = "outputs/aligned_pairs"
    BLUR_DIR  = "outputs/aligned_pairs"
    PSF_DIR   = "outputs/psfs"
    OUT_DIR   = "results/psf_validation"

    labels = [str(i) for i in range(-10, 11)]

    print("Validating...")
    for lbl in labels:
        validate_one_label(SHARP_DIR, BLUR_DIR, PSF_DIR, OUT_DIR, lbl)

    print("Done.")


if __name__ == "__main__":
    main()
