import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# ===============================================================
#                 CONFIGURATION (EDIT THESE)
# ===============================================================
SHARP_PATH = "images/not_calibration/0.75m(rand).jpg"         # sharp scene (no PM)
BLUR_PATH  = "images/not_calibration/0.75m(rand-pm).jpg"      # blurred scene (with PM)
PSF_PATH   = "outputs/psfs/0_psf.npy"                  # YOU CHOOSE DEPTH
OUT_DIR    = "lab_scene_test"
PATCH_H    = 256
PATCH_W    = 256
# ===============================================================


def normalize01(x):
    x = x.astype(np.float64)
    x -= x.min()
    ptp = np.ptp(x)
    return np.zeros_like(x) if ptp < 1e-12 else x / ptp


def conv_fft_same(I, p):
    """Same-mode convolution using FFT."""
    H, W = I.shape
    h, w = p.shape

    padH, padW = H + h - 1, W + w - 1
    FI = np.fft.rfft2(I, s=(padH, padW))
    Fp = np.fft.rfft2(p, s=(padH, padW))
    out = np.fft.irfft2(FI * Fp, s=(padH, padW))

    sy = (h - 1) // 2
    sx = (w - 1) // 2
    return out[sy:sy+H, sx:sx+W]


# ----------------------------------------------------------------
#               (1) LOAD IMAGES + PSF
# ----------------------------------------------------------------
print("Loading scene...")

I = imageio.imread(SHARP_PATH).astype(np.float64)
J = imageio.imread(BLUR_PATH).astype(np.float64)
p = np.load(PSF_PATH).astype(np.float64)

if I.ndim == 3: I = I[..., 0]
if J.ndim == 3: J = J[..., 0]

# crop equal size
H = min(I.shape[0], J.shape[0])
W = min(I.shape[1], J.shape[1])
I = I[:H, :W]
J = J[:H, :W]

# crop central patch for testing
cy, cx = H//2, W//2
I_patch = I[cy-PATCH_H//2:cy+PATCH_H//2,
            cx-PATCH_W//2:cx+PATCH_W//2]

J_patch = J[cy-PATCH_H//2:cy+PATCH_H//2,
            cx-PATCH_W//2:cx+PATCH_W//2]


# ----------------------------------------------------------------
#               (2) REBLUR USING YOUR ESTIMATED PSF
# ----------------------------------------------------------------
print("Reblurring using estimated PSF...")
J_reblur = conv_fft_same(I_patch, p)

# brightness scaling
alpha = (J_patch.sum() + 1e-8) / (J_reblur.sum() + 1e-8)
J_reblur *= alpha


# ----------------------------------------------------------------
#               (3) DECONVOLUTION (Richardson–Lucy)
# ----------------------------------------------------------------
def richardson_lucy(image, psf, iterations=20):
    image = image.astype(np.float64)
    psf = psf.astype(np.float64)
    psf = psf / psf.sum()

    estimate = np.full_like(image, 0.5)
    psf_mirror = psf[::-1, ::-1]

    for _ in range(iterations):
        conv_est = conv_fft_same(estimate, psf)
        conv_est = np.maximum(conv_est, 1e-8)
        ratio = image / conv_est
        estimate *= conv_fft_same(ratio, psf_mirror)

    return estimate


print("Running deconvolution...")
J_deconv = richardson_lucy(J_patch, p, iterations=30)


# ----------------------------------------------------------------
#               (4) SAVE OUTPUTS
# ----------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# ---- safe save helper ----
def save_img(path, img):
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    imageio.imwrite(path, img)

save_img(os.path.join(OUT_DIR, "sharp_patch.png"), normalize01(I_patch))
save_img(os.path.join(OUT_DIR, "blur_patch.png"), normalize01(J_patch))
save_img(os.path.join(OUT_DIR, "reblur.png"), normalize01(J_reblur))
save_img(os.path.join(OUT_DIR, "deconv.png"), normalize01(J_deconv))



# comparison panel
fig, axs = plt.subplots(1, 4, figsize=(14, 4))
imgs = [
    (I_patch,  "Sharp Patch"),
    (J_patch,  "Blur Patch (PM)"),
    (J_reblur, "Reblur I⊗p"),
    (J_deconv, "Deconvolution")
]


for ax, (im, title) in zip(axs, imgs):
    ax.imshow(normalize01(im), cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "panel.png"), dpi=200)
plt.close()

print("Saved results in:", OUT_DIR)
