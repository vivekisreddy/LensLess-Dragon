#!/usr/bin/env python3
"""
Image reconstruction using precomputed PSFs (from Heide et al. method):
- Loads sharp and blurred image pairs
- Loads PSFs estimated previously (per depth/label)
- Performs FFT-based deconvolution to restore blurred image
- Optionally re-blurs the reconstructed image with the estimated PSF
Author: Vivek (Python implementation)
"""

import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ---------- FFT helpers ----------
def fft2c(x):
    return np.fft.fft2(x)

def ifft2c(X):
    return np.fft.ifft2(X).real

def pad_to_shape(img, target_shape):
    """Pad image to target shape (H,W) with zeros"""
    H, W = target_shape
    h, w = img.shape
    pad_y = H - h
    pad_x = W - w
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    return np.pad(img, ((pad_top, pad_bottom),(pad_left, pad_right)), mode='constant')

def crop_center(full, shape):
    """Crop center of image to shape"""
    H, W = full.shape
    h, w = shape
    cy, cx = H // 2, W // 2
    sy, sx = cy - h // 2, cx - w // 2
    return full[sy:sy+h, sx:sx+w]

# ---------- Wiener / regularized deconvolution ----------
def wiener_deconv(blurred, psf, eps=1e-3):
    """
    FFT-based Wiener deconvolution:
    X = conj(PSF) / (|PSF|^2 + eps) * FFT(blurred)
    """
    H, W = blurred.shape
    kh, kw = psf.shape

    # pad PSF to image size
    psf_full = pad_to_shape(psf, (H, W))

    # FFTs
    B = fft2c(blurred)
    P = fft2c(np.fft.ifftshift(psf_full))  # center PSF

    # Wiener deconvolution
    P_conj = np.conj(P)
    deconv = ifft2c((P_conj / (np.abs(P)**2 + eps)) * B)
    return deconv

def blur_with_psf(image, psf):
    """Convolve image with PSF using FFT"""
    H, W = image.shape
    kh, kw = psf.shape
    psf_full = pad_to_shape(psf, (H, W))
    P = fft2c(np.fft.ifftshift(psf_full))
    I = fft2c(image)
    blurred = ifft2c(P * I)
    return blurred

# ---------- Runner ----------
def reconstruct_images(IN_DIR="outputs/aligned_pairs",
                       PSF_DIR="outputs/psfs",
                       OUT_DIR="outputs/reconstructed",
                       eps_wiener=1e-3):
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = [str(i) for i in range(-10, 11)]
    for lbl in labels:
        sharp_path = os.path.join(IN_DIR, f"{lbl}_sharp.png")
        blur_path  = os.path.join(IN_DIR, f"{lbl}_blur.png")
        psf_path   = os.path.join(PSF_DIR, f"{lbl}_psf.npy")
        if not (os.path.exists(sharp_path) and os.path.exists(blur_path) and os.path.exists(psf_path)):
            print(f"[skip] {lbl} missing input/PSF")
            continue

        # Load images
        I = imageio.imread(sharp_path).astype(np.float32)/255.0
        J = imageio.imread(blur_path).astype(np.float32)/255.0
        if I.ndim == 3: I = I[..., 0]
        if J.ndim == 3: J = J[..., 0]

        # Load PSF
        psf = np.load(psf_path)

        # Reconstruct image
        I_rec = wiener_deconv(J, psf, eps=eps_wiener)
        I_rec = np.clip(I_rec, 0, 1)

        # Re-blur reconstructed image
        I_blur_recon = blur_with_psf(I_rec, psf)
        I_blur_recon = np.clip(I_blur_recon, 0, 1)

        # Save reconstructed image
        rec_path = os.path.join(OUT_DIR, f"{lbl}_reconstructed.png")
        imageio.imwrite(rec_path, (I_rec*255).astype(np.uint8))
        blur_path_recon = os.path.join(OUT_DIR, f"{lbl}_reblur.png")
        imageio.imwrite(blur_path_recon, (I_blur_recon*255).astype(np.uint8))

        print(f"[{lbl}] reconstructed -> {rec_path}, reblur -> {blur_path_recon}")

if __name__ == "__main__":
    reconstruct_images()
