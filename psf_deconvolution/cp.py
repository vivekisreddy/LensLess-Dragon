import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ============================================================
#  FFT "same" convolution + adjoint
# ============================================================

def conv_fft_same(I: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    I ⊗ p  (2D conv, 'same' size as I) via FFT.
    I: HxW
    p: hxw (PSF)
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


def conv_adj_fft_to_psf(I: np.ndarray, resid: np.ndarray, psf_shape) -> np.ndarray:
    """
    Adjoint of conv_fft_same with respect to p.
    Given I (sharp) and resid = (I⊗p - sJ), returns gradient wrt p.
    """
    H, W = I.shape
    h, w = psf_shape
    padH, padW = H + h - 1, W + w - 1

    FI = np.fft.rfft2(I,     s=(padH, padW))
    FR = np.fft.rfft2(resid, s=(padH, padW))
    K  = np.fft.irfft2(np.conj(FI) * FR, s=(padH, padW))

    # crop so that adjoint matches PSF support
    sy = H - 1 - (h - 1) // 2
    sx = W - 1 - (w - 1) // 2
    return K[sy:sy+h, sx:sx+w]


# ============================================================
#  TV dual divergence
# ============================================================

def divergence(qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    """
    Compute divergence of dual field q=(qx,qy).
    """
    div = np.zeros_like(qx)
    # y-direction
    div[:-1, :] += qy[:-1, :]
    div[1:,  :] -= qy[:-1, :]
    # x-direction
    div[:, :-1] += qx[:, :-1]
    div[:, 1: ] -= qx[:, :-1]
    return div


# ============================================================
#  Automatic checkerboard patch finder (left side)
# ============================================================

def extract_checkerboard_patch(
    I: np.ndarray,
    J: np.ndarray,
    patch_h: int = 256,
    patch_w: int = 256,
    left_frac: float = 0.5,
) :
    """
    Automatically pick a 256x256 patch on the *left* checkerboard by
    looking for a high-variance region in |I-J|.

    Returns:
        I_patch, J_patch, (x0, y0)
    """
    assert I.shape == J.shape, "Sharp and blur must be same shape"
    H, W = I.shape

    # if patch bigger than image, just center-crop
    if (patch_h > H) or (patch_w > W):
        print("  [warn] patch larger than image; using center crop")
        cy, cx = H // 2, W // 2
        y0 = max(0, cy - patch_h // 2)
        x0 = max(0, cx - patch_w // 2)
        y0 = min(y0, H - patch_h)
        x0 = min(x0, W - patch_w)
        return I[y0:y0+patch_h, x0:x0+patch_w], \
               J[y0:y0+patch_h, x0:x0+patch_w], \
               (x0, y0)

    # difference image highlights edges (checkerboard)
    D = np.abs(I.astype(np.float64) - J.astype(np.float64))
    D = D - D.min()
    D /= (D.max() + 1e-8)

    # restrict search to left part of image
    W_left = int(W * left_frac)
    if W_left < patch_w:
        W_left = patch_w

    # candidate top-left positions
    y_max = H - patch_h
    x_max = W_left - patch_w
    if x_max <= 0 or y_max <= 0:
        # fallback: center crop in left half
        cy = H // 2
        cx = W_left // 2
        y0 = max(0, cy - patch_h // 2)
        x0 = max(0, cx - patch_w // 2)
        y0 = min(y0, H - patch_h)
        x0 = min(x0, W - patch_w)
        return I[y0:y0+patch_h, x0:x0+patch_w], \
               J[y0:y0+patch_h, x0:x0+patch_w], \
               (x0, y0)

    ys = np.arange(0, y_max + 1)
    xs = np.arange(0, x_max + 1)

    # integral image for fast rectangle sums
    S = D.cumsum(axis=0).cumsum(axis=1)
    # pad to (H+1, W+1)
    S = np.pad(S, ((1,0),(1,0)), mode='constant', constant_values=0)

    # compute sum over all possible patch positions (vectorized)
    # sum[y,x] = sum_{i=y}^{y+ph-1} sum_{j=x}^{x+pw-1} D[i,j]
    Y0 = ys[:, None]
    X0 = xs[None, :]

    Y1 = Y0 + patch_h
    X1 = X0 + patch_w

    sum_map = S[Y1, X1] - S[Y0, X1] - S[Y1, X0] + S[Y0, X0]  # (len(ys), len(xs))

    # pick location with largest energy (most edges/texture)
    idx = np.argmax(sum_map)
    iy, ix = np.unravel_index(idx, sum_map.shape)
    y0 = int(ys[iy])
    x0 = int(xs[ix])

    I_patch = I[y0:y0+patch_h, x0:x0+patch_w]
    J_patch = J[y0:y0+patch_h, x0:x0+patch_w]

    return I_patch, J_patch, (x0, y0)


def save_overlay_triplet(I, J, x0, y0, ph, pw, out_path, title="patch"):
    """
    Debug visualization: sharp, blur, |sharp-blur| with red rectangle.
    """
    D = np.abs(I.astype(np.float64) - J.astype(np.float64))
    D -= D.min()
    D /= (D.max() + 1e-8)

    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    for ax, img, name in zip(axes,
                             [I, J, D],
                             ["sharp", "blur", "|sharp-blur|"]):
        ax.imshow(img, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
        rect = patches.Rectangle(
            (x0, y0),
            pw, ph,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ============================================================
#  Chambolle–Pock for PSF
# ============================================================

def chambolle_pock_psf(I, J, p_init,
                        lam=1e-5,   # small TV (we want structure)
                        mu=10.0,    # sum-to-one penalty
                        tau=0.05,
                        sigma=0.05,
                        theta=1.0,
                        max_iter=300):
    """
    Estimate PSF p from sharp patch I and blurred patch J using Chambolle-Pock.
    """
    I = I.astype(np.float64)
    J = J.astype(np.float64)

    # normalize each patch to [0,1]
    I = I - I.min()
    J = J - J.min()
    I /= (I.max() + 1e-8)
    J /= (J.max() + 1e-8)

    p = p_init.copy().astype(np.float64)
    p_bar = p.copy()
    h, w = p.shape

    # dual variables for TV
    qx = np.zeros_like(p)
    qy = np.zeros_like(p)

    # brightness scaling between I and J
    s = I.sum() / (J.sum() + 1e-8)

    for it in range(max_iter):
        # --- dual update (TV of p_bar) ---
        grad_x = np.zeros_like(p_bar)
        grad_y = np.zeros_like(p_bar)
        grad_x[:, :-1] = p_bar[:, 1:] - p_bar[:, :-1]
        grad_y[:-1, :] = p_bar[1:, :] - p_bar[:-1, :]

        qx += sigma * grad_x
        qy += sigma * grad_y
        norm = np.maximum(1.0, np.sqrt(qx*qx + qy*qy))
        qx /= norm
        qy /= norm

        # --- primal update (PSF) ---
        I_conv_p = conv_fft_same(I, p)
        resid = I_conv_p - s * J

        g_data = 2.0 * conv_adj_fft_to_psf(I, resid, (h, w))
        div_q  = divergence(qx, qy)
        g_sum  = 2.0 * mu * (p.sum() - 1.0)

        p_new = p - tau * (g_data + g_sum - lam * div_q)

        # nonnegativity + renormalize
        p_new = np.maximum(p_new, 0.0)
        ps = p_new.sum()
        if ps > 0:
            p_new /= ps

        # extrapolation
        p_bar = p_new + theta * (p_new - p)
        p = p_new

        if (it + 1) % 50 == 0:
            resid_dbg = conv_fft_same(I, p) - s * J
            data_term = (resid_dbg * resid_dbg).mean()
            print(f"    iter {it+1:4d} | data_term={data_term:.6e}, psf_max={p.max():.4e}")

    return p.astype(np.float32)


# ============================================================
#  Main driver
# ============================================================

def main():
    IN_DIR  = "outputs/aligned_pairs"   # rectified full images
    OUT_DIR = "outputs/psf_patches_auto"
    os.makedirs(OUT_DIR, exist_ok=True)

    PATCH_H = 256
    PATCH_W = 256
    PSF_SIZE = 21  # 21x21 PSF, as you requested

    labels = [str(i) for i in range(-10, 11)]

    for lbl in labels:
        sharp_path = os.path.join(IN_DIR, f"{lbl}_sharp.png")
        blur_path  = os.path.join(IN_DIR, f"{lbl}_blur.png")

        if not (os.path.exists(sharp_path) and os.path.exists(blur_path)):
            print(f"[{lbl}] missing images, skipping.")
            continue

        # load
        I_full = imageio.imread(sharp_path).astype(np.float32)
        J_full = imageio.imread(blur_path).astype(np.float32)

        # force grayscale if RGB
        if I_full.ndim == 3:
            I_full = I_full[..., 0]
        if J_full.ndim == 3:
            J_full = J_full[..., 0]

        print(f"[{lbl}] full shape = {I_full.shape}")

        # ----- extract best checkerboard patch on left -----
        I_patch, J_patch, (x0, y0) = extract_checkerboard_patch(
            I_full, J_full,
            patch_h=PATCH_H,
            patch_w=PATCH_W,
            left_frac=0.5
        )
        print(f"[{lbl}] patch @ (x0={x0}, y0={y0}), shape={I_patch.shape}")

        # debug overlay
        overlay_path = os.path.join(OUT_DIR, f"{lbl}_overlay.png")
        save_overlay_triplet(I_full, J_full, x0, y0, PATCH_H, PATCH_W,
                             overlay_path, title=f"label {lbl}")

        # normalize patches for saving
        I_norm = (I_patch - I_patch.min()) / (np.ptp(I_patch) + 1e-6)
        J_norm = (J_patch - J_patch.min()) / (np.ptp(J_patch) + 1e-6)
        imageio.imwrite(os.path.join(OUT_DIR, f"{lbl}_sharp_patch.png"),
                        (I_norm * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(OUT_DIR, f"{lbl}_blur_patch.png"),
                        (J_norm * 255).astype(np.uint8))

        # ----- initial PSF: small centered Gaussian -----
        h = w = PSF_SIZE
        yy, xx = np.mgrid[:h, :w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        sigma_psf = 0.25 * min(h, w)
        p_init = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2.0 * sigma_psf**2))
        p_init = np.maximum(p_init, 0.0)
        p_init /= (p_init.sum() + 1e-8)

        # ----- run Chambolle–Pock -----
        print(f"[{lbl}] running CP PSF estimation...")
        p_est = chambolle_pock_psf(
            I_patch, J_patch, p_init,
            lam=1e-5, mu=10.0,
            tau=0.05, sigma=0.05,
            max_iter=300
        )

        print(f"[{lbl}] PSF sum={p_est.sum():.6f}, max={p_est.max():.4e}")

        # save PSF
        np.save(os.path.join(OUT_DIR, f"{lbl}_psf.npy"), p_est)

        plt.figure(figsize=(5,5))
        plt.imshow(p_est, cmap="gray", interpolation="bicubic")
        plt.title(f"PSF {lbl}")
        plt.axis("off")
        plt.savefig(os.path.join(OUT_DIR, f"{lbl}_psf.png"),
                    dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
