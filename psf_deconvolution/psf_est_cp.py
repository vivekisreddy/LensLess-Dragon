 #!/usr/bin/env python3
import os
import numpy as np
import imageio.v2 as imageio
from scipy.sparse.linalg import LinearOperator, cg
import matplotlib.pyplot as plt
from itertools import product

# --- embed/crop helpers ---
def embed_kernel_to_full(p_small, H, W):
    kh, kw = p_small.shape
    full = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    sy, sx = cy - kh // 2, cx - kw // 2
    full[sy:sy+kh, sx:sx+kw] = p_small
    return full, (sy, sx)

def crop_center_from_full(full, kh, kw):
    H, W = full.shape
    cy, cx = H // 2, W // 2
    sy, sx = cy - kh // 2, cx - kw // 2
    return full[sy:sy+kh, sx:sx+kw]

def fft2c(x): return np.fft.fft2(x)
def ifft2c(X): return np.fft.ifft2(X).real

# --- data operators ---
class DataOperators:
    def __init__(self, I):
        self.I = I.astype(np.float32)
        self.H, self.W = I.shape
        self.FI = fft2c(self.I)
        self.absFI2 = np.abs(self.FI) ** 2

    def conv_I_p_full(self, p_full):
        return ifft2c(self.FI * fft2c(p_full))

    def BT_mult_image(self, img):
        return ifft2c(np.conj(self.FI) * fft2c(img))

    def BTB_mult_full(self, p_full):
        return ifft2c(self.absFI2 * fft2c(p_full))

# --- Chambolle-Pock PSF estimator ---
def chambolle_pock_psf(I, J, kernel_size=(15,15),
                        lam=1e-2, mu=1e2,
                        tau=None, sigma=None, theta=1.0,
                        n_iters=300, cg_tol=1e-6, cg_maxiter=60,
                        verbose=True):
    H, W = I.shape
    kh, kw = kernel_size
    D = DataOperators(I)
    s = (I.sum()) / (J.sum() + 1e-12)

    p = np.ones((kh, kw), dtype=np.float32) / (kh*kw)
    yx = np.zeros_like(p)
    yy = np.zeros_like(p)

    L_K = 3.0
    if sigma is None: sigma = 1.0
    if tau is None: tau = 0.9 / (sigma * (L_K ** 2))

    if verbose:
        print(f"CP params: sigma={sigma:.4g}, tau={tau:.4g}, lambda={lam}, mu={mu}")

    sJ = (s * J).astype(np.float32)
    BT_sJ_full = D.BT_mult_image(sJ)
    BT_sJ_small = crop_center_from_full(BT_sJ_full, kh, kw)

    def K_of(p_small):
        gx = np.zeros_like(p_small)
        gy = np.zeros_like(p_small)
        gx[:, :-1] = p_small[:, 1:] - p_small[:, :-1]
        gy[:-1, :] = p_small[1:, :] - p_small[:-1, :]
        return gx, gy

    def Kstar_of(gx, gy):
        out = np.zeros_like(gx)
        out[:, :-1] -= gx[:, :-1]; out[:, 1:] += gx[:, :-1]
        out[:-1, :] -= gy[:-1, :]; out[1:, :] += gy[:-1, :]
        return out

    def apply_BTB_to_small(p_small):
        p_full, _ = embed_kernel_to_full(p_small, H, W)
        res_full = D.BTB_mult_full(p_full)
        return crop_center_from_full(res_full, kh, kw)

    one_small = np.ones((kh, kw), dtype=np.float32)
    def A_mul(p_vec, one_small=one_small):
        p_small = p_vec.reshape((kh, kw))
        BTB_part = apply_BTB_to_small(p_small)
        ssum = p_small.sum()
        return ((1.0 / tau) * p_small + (2.0 / lam) * BTB_part + (2.0 * mu / lam) * (ssum * one_small)).ravel()

    def proxG_tilde(tilde_u):
        b = (1.0 / tau) * tilde_u + (2.0 / lam) * BT_sJ_small
        size = kh*kw
        linop = LinearOperator((size, size), matvec=lambda v: A_mul(v), dtype=np.float64)
        x0 = tilde_u.ravel().astype(np.float64)
        x, info = cg(linop, b.ravel().astype(np.float64), x0=x0, rtol=cg_tol, maxiter=cg_maxiter)
        u_opt = x.reshape((kh, kw)).astype(np.float32)
        u_opt[u_opt < 0] = 0
        ssum = u_opt.sum()
        if ssum > 0: u_opt /= ssum
        else: u_opt[:] = 1.0/(kh*kw)
        return u_opt

    p_bar = p.copy()
    for it in range(1, n_iters+1):
        gx_bar, gy_bar = K_of(p_bar)
        yx += sigma * gx_bar; yy += sigma * gy_bar
        mag = np.sqrt(yx**2 + yy**2); denom = np.maximum(1.0, mag)
        yx /= denom; yy /= denom

        Kt_y = Kstar_of(yx, yy)
        x_tilde = p + tau * Kt_y
        p_new = proxG_tilde(x_tilde)
        p_bar = p_new + theta * (p_new - p)
        p = p_new

    return p

# --- tiled + color PSF runner ---
def run_all_tiles(IN_DIR="outputs/aligned_pairs", OUT_DIR="outputs/psfs",
                  kernel_size=(15,15), lam=1e-3, mu=1e3, cp_iters=200,
                  tile_size=(128,128)):
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = [str(i) for i in range(-10, 11)]
    
    for lbl in labels:
        sharp_path = os.path.join(IN_DIR, f"{lbl}_sharp.png")
        blur_path  = os.path.join(IN_DIR, f"{lbl}_blur.png")
        if not (os.path.exists(sharp_path) and os.path.exists(blur_path)):
            print(f"[skip] {lbl} missing"); continue

        I = imageio.imread(sharp_path).astype(np.float32)/255.0
        J = imageio.imread(blur_path).astype(np.float32)/255.0

        if I.ndim == 2: I = I[..., None]; J = J[..., None]
        H, W, C = I.shape

        # Tiles along HxW
        th, tw = tile_size
        n_tiles_h = H // th
        n_tiles_w = W // tw

        for c in range(C):
            psf_tiles = []
            for i, j in product(range(n_tiles_h), range(n_tiles_w)):
                I_tile = I[i*th:(i+1)*th, j*tw:(j+1)*tw, c]
                J_tile = J[i*th:(i+1)*th, j*tw:(j+1)*tw, c]
                p_hat = chambolle_pock_psf(I_tile, J_tile,
                                           kernel_size=kernel_size,
                                           lam=lam, mu=mu,
                                           n_iters=cp_iters,
                                           verbose=False)
                psf_tiles.append(p_hat)
                # save each tile
                tile_out = os.path.join(OUT_DIR, f"{lbl}_c{c}_tile{i}_{j}_psf.npy")
                np.save(tile_out, p_hat)
            # grid visualization of tiles
            fig, axes = plt.subplots(n_tiles_h, n_tiles_w, figsize=(n_tiles_w*2,n_tiles_h*2))
            for idx, ax in enumerate(axes.flat):
                ax.imshow(psf_tiles[idx], cmap='viridis', interpolation='bicubic')
                ax.axis('off')
            plt.suptitle(f"Label {lbl} - Channel {c}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{lbl}_channel{c}_psf_grid.png"), dpi=200)
            plt.close()
            print(f"[{lbl}][c={c}] PSF tiles saved & grid visualized.")

if __name__ == "__main__":
    run_all_tiles(IN_DIR="outputs/aligned_pairs",
                  OUT_DIR="outputs/psfs",
                  kernel_size=(15,15),
                  lam=1e-3, mu=1e3,
                  cp_iters=200,
                  tile_size=(128,128))
