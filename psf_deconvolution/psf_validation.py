import os
import glob
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
PATCH_DIR = "outputs/psf_patches"   # where *_sharp_patch.png, *_blur_patch.png, *_psf.npy live
OUT_DIR   = "outputs/psf_validation"

os.makedirs(OUT_DIR, exist_ok=True)

# If you want to hard-code labels, uncomment and edit:
# LABELS = [str(i) for i in range(-10, 11)]
# Otherwise we auto-detect any *_psf.npy in PATCH_DIR:
LABELS = None


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def conv_fft_same(I: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    I ⊗ p  (2D convolution, 'same' size as I) via FFT.
    I: (H, W)
    p: (h, w) PSF
    """
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


def wiener_deconv(J: np.ndarray, p: np.ndarray, gamma: float = 1e-3) -> np.ndarray:
    """
    Very simple Wiener deconvolution assuming circular convolution.
    J: blurred image, (H, W)
    p: PSF, (h, w)
    gamma: regularization (larger -> more smoothing, less ringing)
    """
    H, W = J.shape
    # PSF to full image size (centered)
    Fp = np.fft.fft2(p, s=(H, W))
    FJ = np.fft.fft2(J)

    denom = (np.abs(Fp) ** 2) + gamma
    F_est = np.conj(Fp) / denom * FJ
    I_rec = np.fft.ifft2(F_est).real
    return I_rec


def normalize01(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] for visualization."""
    x = x.astype(np.float64)
    mn = x.min()
    mx = x.max()
    rng = float(np.ptp(x))  # np.ptp for NumPy 2.x
    if rng < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (rng + 1e-8)


def load_gray(path: str) -> np.ndarray:
    """Load grayscale (uint8) and convert to float32."""
    img = imageio.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse)


# ---------------------------------------------------------
# Per-label validation visualization
# ---------------------------------------------------------

def validate_one_label(label: str):
    sharp_path = os.path.join(PATCH_DIR, f"{label}_sharp_patch.png")
    blur_path  = os.path.join(PATCH_DIR, f"{label}_blur_patch.png")
    psf_path   = os.path.join(PATCH_DIR, f"{label}_psf.npy")

    if not (os.path.exists(sharp_path) and os.path.exists(blur_path) and os.path.exists(psf_path)):
        print(f"[{label}] Missing files, skipping.")
        return None

    I = load_gray(sharp_path)
    J = load_gray(blur_path)
    p = np.load(psf_path).astype(np.float64)

    # Normalize patches to [0,1] for fair comparison
    I_n = normalize01(I)
    J_n = normalize01(J)

    # Forward simulation
    J_sim = conv_fft_same(I_n, p)
    # Optional brightness scaling: match mean intensity of J
    scale = J_n.mean() / (J_sim.mean() + 1e-8)
    J_sim_scaled = np.clip(J_sim * scale, 0.0, 1.0)

    # Residual
    resid = J_n - J_sim_scaled
    mse = float(np.mean(resid ** 2))
    psnr = psnr_from_mse(mse, max_val=1.0)

    print(f"[{label}] Forward MSE={mse:.6e}, PSNR={psnr:.2f} dB, PSF sum={p.sum():.6f}")

    # Simple deconvolution
    I_rec = wiener_deconv(J_n, p, gamma=1e-3)
    I_rec_n = normalize01(I_rec)

    # -------------------------------------------------
    # Plot diagnostics
    # -------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(f"PSF validation — label {label}", fontsize=16)

    # Row 1: sharp / blur / simulated blur
    ax = axes[0, 0]
    ax.imshow(I_n, cmap="gray", vmin=0, vmax=1)
    ax.set_title("sharp patch")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(J_n, cmap="gray", vmin=0, vmax=1)
    ax.set_title("blur patch")
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(J_sim_scaled, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"sharp ⊗ PSF\n(MSE={mse:.3e}, PSNR={psnr:.1f} dB)")
    ax.axis("off")

    # Row 2: error / PSF / deblur
    ax = axes[1, 0]
    err_vis = normalize01(np.abs(resid))
    ax.imshow(err_vis, cmap="inferno", vmin=0, vmax=1)
    ax.set_title("|blur − sim| (normalized)")
    ax.axis("off")

    ax = axes[1, 1]
    ax.imshow(p, cmap="gray", interpolation="bicubic")
    ax.set_title(f"PSF {label}")
    ax.axis("off")

    ax = axes[1, 2]
    ax.imshow(I_rec_n, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Wiener deconv(blur, PSF)")
    ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(OUT_DIR, f"{label}_validation.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    return mse, psnr


# ---------------------------------------------------------
# Grid of all PSFs
# ---------------------------------------------------------

def save_psf_grid(labels):
    psfs = []
    used_labels = []
    for lbl in labels:
        path = os.path.join(PATCH_DIR, f"{lbl}_psf.npy")
        if os.path.exists(path):
            p = np.load(path)
            psfs.append(p)
            used_labels.append(lbl)

    if not psfs:
        print("No PSFs found, skipping PSF grid.")
        return

    # normalize each PSF just for display
    psfs_disp = [normalize01(p) for p in psfs]

    # make a grid roughly square
    n = len(psfs_disp)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx, (lbl, p_img) in enumerate(zip(used_labels, psfs_disp)):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.imshow(p_img, cmap="gray", interpolation="bicubic")
        ax.set_title(f"label {lbl}")
        ax.axis("off")

    # turn off any unused axes
    for idx in range(len(psfs_disp), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "psf_grid.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved PSF grid to {out_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    global LABELS
    if LABELS is None:
        # auto-detect labels from *_psf.npy files
        labels = []
        for path in glob.glob(os.path.join(PATCH_DIR, "*_psf.npy")):
            base = os.path.basename(path)
            lbl = base.replace("_psf.npy", "")
            labels.append(lbl)
        labels = sorted(labels, key=lambda x: float(x))  # works for numeric string labels
        LABELS = labels

    if not LABELS:
        print("No labels/PSFs found. Check PATCH_DIR.")
        return

    print("Validating labels:", LABELS)

    all_mse = []
    all_psnr = []
    for lbl in LABELS:
        res = validate_one_label(lbl)
        if res is not None:
            mse, psnr = res
            all_mse.append(mse)
            all_psnr.append(psnr)

    if all_mse:
        print("\n=== Summary over all labels ===")
        print(f"  MSE:  mean={np.mean(all_mse):.4e}, min={np.min(all_mse):.4e}, max={np.max(all_mse):.4e}")
        print(f"  PSNR: mean={np.mean(all_psnr):.2f} dB, min={np.min(all_psnr):.2f}, max={np.max(all_psnr):.2f}")

    save_psf_grid(LABELS)


if __name__ == "__main__":
    main()
