# batch_rectify_pairs_ecc.py
# Align each depth pair (-10..10) using ECC (intensity-based) and crop to
# the valid overlap region. No checkerboard detection needed.
#
# Inputs:
#   images/wo-phasemask/  -10.png ... 0.png ... 10.png   (sharp)
#   images/phasemask/     -10.png ... 0.png ... 10.png   (blur)
#
# Outputs:
#   outputs/aligned_pairs/<label>_sharp.png
#   outputs/aligned_pairs/<label>_blur.png
#   outputs/aligned_pairs/<label>_diff.png   (visual check)

import os, glob
import numpy as np
import cv2
from PIL import Image

# ---------------- Config ----------------
SHARP_DIR = "images/phasecam2_images/wo-phasemask"
BLUR_DIR  = "images/phasecam2_images/phasemask"
OUT_DIR   = "outputs/aligned_pairs"

LABELS    = [str(i) for i in range(-10, 11)]
IMG_EXTS  = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

ECC_ITERS = 300
ECC_EPS   = 1e-6

# ------------- Helpers -------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def find_image_path(folder, label):
    for ext in IMG_EXTS:
        p = os.path.join(folder, f"{label}{ext}")
        if os.path.isfile(p):
            return p
    hits = []
    for ext in IMG_EXTS:
        hits += glob.glob(os.path.join(folder, f"{label}*{ext}"))
    return hits[0] if hits else None

def load_u8_gray(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.asarray(Image.open(path).convert("L"))
    return img

def to_f01(u8):
    return (u8.astype(np.float32) / 255.0).clip(0,1)

def save_img01(arr, path):
    x = np.clip(arr, 0.0, 1.0)
    Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(path)

def ecc_align_homography(I01, J01, iters=ECC_ITERS, eps=ECC_EPS):
    """
    Align J to I using ECC with homography; returns (J_aligned, mask).
    I01, J01 in [0,1], float32, same initial size.
    """
    H, W = I01.shape
    warp_mode = cv2.MOTION_HOMOGRAPHY
    WARP = np.eye(3, dtype=np.float32)

    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)

    Iu8 = (I01 * 255).astype(np.uint8)
    Ju8 = (J01 * 255).astype(np.uint8)

    try:
        _, WARP = cv2.findTransformECC(Iu8, Ju8, WARP, warp_mode, crit, None, 5)
    except cv2.error:
        # Fallback: affine motion (a bit weaker but more robust sometimes)
        warp_mode = cv2.MOTION_AFFINE
        WARP2 = np.eye(2,3, dtype=np.float32)
        _, WARP2 = cv2.findTransformECC(Iu8, Ju8, WARP2, warp_mode, crit, None, 5)
        Jw_u8 = cv2.warpAffine(Ju8, WARP2, (W, H), flags=cv2.INTER_LINEAR)
        Jw = to_f01(Jw_u8)
        # valid mask = all pixels inside result (affine keeps full)
        mask = np.ones_like(I01, dtype=np.uint8)
        return Jw, mask

    # Homography warp
    Jw_u8 = cv2.warpPerspective(Ju8, WARP, (W, H), flags=cv2.INTER_LINEAR)
    Jw = to_f01(Jw_u8)

    # build a mask of valid pixels (where homography mapped inside image)
    mask = cv2.warpPerspective(np.ones_like(Ju8, dtype=np.uint8),
                               WARP, (W, H),
                               flags=cv2.INTER_NEAREST)
    return Jw, mask

def crop_to_valid_overlap(I01, J01, mask):
    """
    Crop I and J to the bounding box of mask>0 (valid overlap region).
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        # no overlap, just center-crop to common min size as worst case
        Hmin = min(I01.shape[0], J01.shape[0])
        Wmin = min(I01.shape[1], J01.shape[1])
        syI = (I01.shape[0]-Hmin)//2; sxI = (I01.shape[1]-Wmin)//2
        syJ = (J01.shape[0]-Hmin)//2; sxJ = (J01.shape[1]-Wmin)//2
        I_c = I01[syI:syI+Hmin, sxI:sxI+Wmin]
        J_c = J01[syJ:syJ+Hmin, sxJ:sxJ+Wmin]
        return I_c, J_c

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    I_c = I01[y0:y1, x0:x1]
    J_c = J01[y0:y1, x0:x1]
    return I_c, J_c

# --------------- Main ---------------

def main():
    ensure_dir(OUT_DIR)
    print("ECC alignment, homography -> overlap crop")

    for lbl in LABELS:
        sharp_path = find_image_path(SHARP_DIR, lbl)
        blur_path  = find_image_path(BLUR_DIR,  lbl)

        if sharp_path is None or blur_path is None:
            print(f"[skip] missing pair for label {lbl}")
            continue

        Iu8 = load_u8_gray(sharp_path)
        Ju8 = load_u8_gray(blur_path)

        # First, resize J to I size if completely different resolutions
        if Iu8.shape != Ju8.shape:
            Ju8 = cv2.resize(Ju8, (Iu8.shape[1], Iu8.shape[0]), interpolation=cv2.INTER_LINEAR)

        I01 = to_f01(Iu8)
        J01 = to_f01(Ju8)

        try:
            Jw, mask = ecc_align_homography(I01, J01)
            I_c, J_c = crop_to_valid_overlap(I01, Jw, mask)
            print(f"[{lbl}] aligned & cropped -> shape={I_c.shape}")
        except Exception as e:
            print(f"[warn] ECC failed for label {lbl}: {e}")
            # last-resort: center crop to common size
            Hmin = min(I01.shape[0], J01.shape[0])
            Wmin = min(I01.shape[1], J01.shape[1])
            syI = (I01.shape[0]-Hmin)//2; sxI = (I01.shape[1]-Wmin)//2
            syJ = (J01.shape[0]-Hmin)//2; sxJ = (J01.shape[1]-Wmin)//2
            I_c = I01[syI:syI+Hmin, sxI:sxI+Wmin]
            J_c = J01[syJ:syJ+Hmin, sxJ:sxJ+Wmin]
            print(f"[{lbl}] fallback center-crop -> shape={I_c.shape}")

        # save aligned sharp/blur
        save_img01(I_c, os.path.join(OUT_DIR, f"{lbl}_sharp.png"))
        save_img01(J_c, os.path.join(OUT_DIR, f"{lbl}_blur.png"))

        # diff for sanity check
        diff = np.abs(I_c - J_c)
        if diff.max() > 0:
            diff = diff / diff.max()
        save_img01(diff, os.path.join(OUT_DIR, f"{lbl}_diff.png"))

if __name__ == "__main__":
    main()
