#!/usr/bin/env python3
"""
Chambolle-Pock PSF estimator (faithful to Heide et al. 'High-Quality Computational Imaging'):
- solves: p_opt = argmin_p || I * p - s J ||_2^2 + lambda * ||grad p||_1 + mu * (sum(p)-1)^2
- implements Chambolle-Pock with K = gradient, F = l1, G = data + sum constraint.
Author: adapted for Vivek (explanatory implementation).
"""

import os
import numpy as np
import imageio.v2 as imageio
from scipy.sparse.linalg import LinearOperator, cg
import matplotlib.pyplot as plt

# ---------- Helpers: FFT convolution / embedding ----------
def embed_kernel_to_full(p_small, H, W):
    """Embed small kernel centered into HxW full array."""
    kh, kw = p_small.shape
    full = np.zeros((H, W), dtype=np.float32)
    cy = H // 2
    cx = W // 2
    sy = cy - kh // 2
    sx = cx - kw // 2
    full[sy:sy+kh, sx:sx+kw] = p_small
    return full, (sy, sx)

def crop_center_from_full(full, kh, kw):
    H, W = full.shape
    cy = H // 2
    cx = W // 2
    sy = cy - kh // 2
    sx = cx - kw // 2
    return full[sy:sy+kh, sx:sx+kw]

def fft2c(x):
    return np.fft.fft2(x)

def ifft2c(X):
    return np.fft.ifft2(X).real

# ---------- Operators for PSF estimation ----------
class DataOperators:
    """Precompute FFTs and provide multiplies B, B^T, B^T B using image-sized FFTs.
       Here B(p_full) = conv(I, p_full) (full-size); p_full is same size as I (H,W).
    """
    def __init__(self, I):
        # I: sharp reference image in [0,1], shape (H,W)
        self.I = I.astype(np.float32)
        self.H, self.W = self.I.shape
        self.FI = fft2c(self.I)
        self.absFI2 = np.abs(self.FI) ** 2

    def conv_I_p_full(self, p_full):
        """I * p_full, both HxW, computed by FFT"""
        return ifft2c(self.FI * fft2c(p_full))

    def BT_mult_image(self, img):  # B^T img (full)
        # adjoint of convolution by I is correlation with I => conj(FI) * F(img)
        return ifft2c(np.conj(self.FI) * fft2c(img))

    def BTB_mult_full(self, p_full):
        # B^T B p_full == ifft( |FI|^2 * F(p_full) )
        return ifft2c(self.absFI2 * fft2c(p_full))

# ---------- Chambolle-Pock PSF estimator ----------
def chambolle_pock_psf(I, J, kernel_size=(15,15),
                        lam=1e-2, mu=1e2,
                        tau=None, sigma=None, theta=1.0,
                        n_iters=300, cg_tol=1e-6, cg_maxiter=60,
                        verbose=True):
    """
    I, J: sharp and coded (blur) image patches, floats in [0,1], same shape HxW
    kernel_size: (kh, kw) small PSF support
    lam: TV weight (lambda)
    mu: energy conservation weight
    tau, sigma: CP steps (if None, computed heuristically)
    Returns: p_small (kh x kw)
    """

    H, W = I.shape
    kh, kw = kernel_size

    # precompute data operator
    D = DataOperators(I)

    # exposure normalization scalar s
    s = (I.sum()) / (J.sum() + 1e-12)

    # initialize primal: small kernel p (positive, normalized)
    p = np.ones((kh, kw), dtype=np.float32)


    # initialize dual: gradient of p (2 x kh x kw)
    yx = np.zeros_like(p)
    yy = np.zeros_like(p)

    # Steps: choose sigma,tau based on Lipschitz of K (here grad operator, norm <= sqrt(8) approx)
    # For gradient K, ||K||^2 <= 8 for discrete gradients on small arrays; use conservative L=3.0
    L_K = 3.0
    if sigma is None: sigma = 1.0
    if tau is None: tau = 0.9 / (sigma * (L_K ** 2))

    if verbose:
        print(f"CP params: sigma={sigma:.4g}, tau={tau:.4g}, theta={theta}, lambda={lam}, mu={mu}")

    # Precompute RHS term in data solve that does not change: B^T (s * J) (full -> crop to small)
    sJ = (s * J).astype(np.float32)
    BT_sJ_full = D.BT_mult_image(sJ)  # HxW full
    BT_sJ_small = crop_center_from_full(BT_sJ_full, kh, kw)

    # helper: K (gradient) map small p -> (gx, gy)
    def K_of(p_small):
        gx = np.zeros_like(p_small)
        gy = np.zeros_like(p_small)
        gx[:, :-1] = p_small[:, 1:] - p_small[:, :-1]
        gy[:-1, :] = p_small[1:, :] - p_small[:-1, :]
        return gx, gy

    # helper: K* (divergence) map dual (gx,gy) -> small array
    def Kstar_of(gx, gy):
        out = np.zeros_like(gx)
        # div_x
        out[:, :-1] -= gx[:, :-1]
        out[:, 1:] += gx[:, :-1]
        # div_y
        out[:-1, :] -= gy[:-1, :]
        out[1:, :] += gy[:-1, :]
        return out

    # Helper: apply BTB to small kernel (returns small kernel)
    def apply_BTB_to_small(p_small):
        # embed to full
        p_full, _ = embed_kernel_to_full(p_small, H, W)
        res_full = D.BTB_mult_full(p_full)  # HxW
        res_small = crop_center_from_full(res_full, kh, kw)
        return res_small

    # Linear operator A for proxG in small-kernel space:
    # A(p) = (1/tau) p + (2/lam) * BTB(p) + (2*mu/lam) * sum(p) * ones
    def A_mul(p_vec, one_small):
        p_small = p_vec.reshape((kh, kw))
        BTB_part = apply_BTB_to_small(p_small)
        ssum = p_small.sum()
        out = (1.0 / tau) * p_small + (2.0 / lam) * BTB_part + (2.0 * mu / lam) * (ssum * one_small)
        return out.ravel()

    # proxG(˜u) solves A u = b where
    # b = (1/tau) * ˜u + (2/lam) * BT^T(sJ) cropped
    one_small = np.ones((kh, kw), dtype=np.float32)
    def proxG_tilde(tilde_u):
        # tilde_u : small (kh,kw)
        b = (1.0 / tau) * tilde_u + (2.0 / lam) * BT_sJ_small

        # define LinearOperator for CG
        size = kh * kw
        linop = LinearOperator((size, size), matvec=lambda v: A_mul(v, one_small), dtype=np.float64)

        # initial guess: tilde_u flattened
        x0 = tilde_u.ravel().astype(np.float64)

        # Solve with CG
        x, info = cg(
            linop,
            b.ravel().astype(np.float64),
            x0=x0,
            rtol=cg_tol,         # <-- correct name
            maxiter=cg_maxiter
        )

        if info != 0:
            # CG did not fully converge; still return current iterate as float
            if verbose:
                print(f"  CG info={info} (nonzero) — continuing with best effort")
        u_opt = x.reshape((kh, kw)).astype(np.float32)

        # enforce non-neg & small renorm (prox should handle constraints; we clamp then renormalize)
        u_opt[u_opt < 0.0] = 0.0
        ssum = u_opt.sum()
        if ssum > 0:
            u_opt /= ssum
        else:
            # fallback: small uniform
            u_opt = np.ones_like(u_opt); u_opt /= u_opt.sum()

        return u_opt

    # --- Main CP loop ---
    p_bar = p.copy()
    for it in range(1, n_iters + 1):
        # dual update: y = prox_{sigma F*}( y + sigma * K p_bar )
        gx_bar, gy_bar = K_of(p_bar)
        yx += sigma * gx_bar
        yy += sigma * gy_bar
        # projection onto l2 ball of radius 1 (per-pixel)
        mag = np.sqrt(yx**2 + yy**2)
        denom = np.maximum(1.0, mag)
        yx /= denom
        yy /= denom

        # primal step: x_hat = p + tau * K^T y
        Kt_y = Kstar_of(yx, yy)
        x_tilde = p + tau * Kt_y

        # proxG: solve linear system for data + sum constraint
        p_new = proxG_tilde(x_tilde)

        # extrapolate
        p_bar = p_new + theta * (p_new - p)
        p = p_new

        # diagnostics
        if verbose and (it % max(1, n_iters // 10) == 0 or it <= 5):
            # compute energy terms
            p_full, _ = embed_kernel_to_full(p, H, W)
            Iconvp = D.conv_I_p_full(p_full)
            data_res = (Iconvp - s * J).ravel()
            data_term = (data_res**2).sum()
            gx, gy = K_of(p)
            tv_term = lam * np.sqrt(gx*gx + gy*gy + 1e-12).sum()
            energy = data_term + tv_term + mu * (p.sum() - 1.0)**2
            print(f"Iter {it:4d}/{n_iters}  energy={energy:.6e}  data={data_term:.6e}  tv={tv_term:.6e}  sum_err={p.sum()-1.0:.3e}")

    return p

# ---------- Runner: loop over labels, save outputs ----------
def run_all(IN_DIR="outputs/aligned_pairs", OUT_DIR="outputs/psfs",
            kernel_size=(15,15), lam=1e-2, mu=1e2, cp_iters=300):
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = [str(i) for i in range(-10, 11)]
    for lbl in labels:
        sharp_path = os.path.join(IN_DIR, f"{lbl}_sharp.png")
        blur_path  = os.path.join(IN_DIR, f"{lbl}_blur.png")
        if not (os.path.exists(sharp_path) and os.path.exists(blur_path)):
            print(f"[skip] {lbl} missing")
            continue
        I = imageio.imread(sharp_path).astype(np.float32) / 255.0
        J = imageio.imread(blur_path).astype(np.float32) / 255.0
        if I.ndim == 3:
            I = I[..., 0]
        if J.ndim == 3:
            J = J[..., 0]
        print(f"\nProcessing label {lbl}  shape={I.shape}")

        p_hat = chambolle_pock_psf(I, J, kernel_size=kernel_size,
                                   lam=lam, mu=mu,
                                   n_iters=cp_iters, verbose=True)

        # Save kernel
        np.save(os.path.join(OUT_DIR, f"{lbl}_psf.npy"), p_hat)
        # Save image visualization (enhanced)
        fig_path = os.path.join(OUT_DIR, f"{lbl}_psf.png")
        plt.figure(figsize=(3.5,3.5))
        plt.imshow(p_hat, cmap='viridis', interpolation='bicubic')
        plt.axis('off')
        plt.title(f"PSF {lbl}")
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
        plt.close()
        print(f"[{lbl}] saved PSF (sum={p_hat.sum():.6f}) -> {fig_path}")

if __name__ == "__main__":
    # example run (tweak parameters to taste)
    run_all(IN_DIR="outputs/aligned_pairs", OUT_DIR="outputs/psfs",
            kernel_size=(15,15), lam=1e-3, mu=1e3, cp_iters=200)
