"""
simulator.py

Simulator to study how aperture radius R and focal length f affect
depth-dependence of PSFs for a pupil-plane phase-mask imaging system.

1. The PSF changes with defocus (z).

2. The rate and structure of that change (w.r.t. z) determine how much depth information the PSF carries.

3. The Fisher information metric computed here estimates how detectable a small change in depth is based on PSF changes.

Outputs:
 - PSF cubes for sampled depths
 - PSF L2-difference and approximate Fisher/CRLB metrics
 - Heatmaps / plots for metric as functions of R and f

Requires: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import os

# -----------------------------
#  User-editable configuration
# -----------------------------
# Sensor (IMX477)
pixel_pitch = 1.55e-6        # m
sensor_pixels = (4056, 3040) # not actively used for PSF grid mapping (optional)
sensor_width = 6.287e-3      # m
sensor_height = 4.712e-3     # m

# Simulation grid (pupil sampling)
PUPIL_RES = 256              # grid size for pupil plane (use 256 or 512)
PAD_FACTOR = 2               # zero-pad factor for FFT to get sharper PSF sampling

# Wavelengths (PhaseCam3D choice)
wavelengths = np.array([610e-9, 530e-9, 470e-9])  # meters

# Default lens focal lengths to sweep (meters) - include 10-16 mm
f_list = np.array([8e-3, 10e-3, 12e-3, 16e-3])  # you can change or expand

# Aperture radii to sweep (meters) - example: 0.5 mm to 2.5 mm
R_list = np.linspace(0.5e-3, 3.0e-3, 8)  # 8 values from 0.5 mm to 3.0 mm

# Depths and focus
# Option A: fix sensor image distance i (m) and compute z0 = object distance in focus
sensor_image_dist = None  # if None, we'll compute i for given z0 from lens formula
# Option B: set the in-focus object distance z0 directly (m)
z0 = 1.0  # in-focus object distance (meters) - change as needed

# Depths to evaluate (meters). We evaluate relative to z0.
z_offsets = np.linspace(-0.5, 0.5, 21)  # offsets from z0; e.g., -0.5..+0.5 meters
z_vals = z0 + z_offsets

# Mask (height map) — start with zero mask (no phase), you can load your own .npy height map
use_mask_file = False
mask_file = "heightmap.npy"  # must match grid PUPIL_RES x PUPIL_RES
delta_n = 0.5  # refractive index difference (n_mask - n_air) — set around 0.5~0.6 for common resins

# Numerical stability
EPS = 1e-12

# Output directory
out_dir = "psf_sweep_outputs"
os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# Utility functions
# -----------------------------
def thin_lens_object_distance(f, i):
    """Given focal length f and image distance i (sensor-lens), return object distance z0."""
    # 1/i + 1/z0 = 1/f  =>  1/z0 = 1/f - 1/i
    denom = 1.0 / f - 1.0 / i
    if abs(denom) < 1e-12:
        return np.inf
    return 1.0 / denom

def thin_lens_image_distance(f, z0):
    """Given focal length f and object distance z0, return image distance i (sensor-lens)."""
    denom = 1.0 / f - 1.0 / z0
    if abs(denom) < 1e-12:
        return np.inf
    return 1.0 / denom

def make_pupil_grid(R_phys, res=PUPIL_RES):
    """
    Make a square grid sampling the pupil physical coordinates from -R..+R
    Returns: X, Y (meters), R_norm (rho/R), aperture_mask (1 inside)
    """
    x = np.linspace(-R_phys, R_phys, res)
    y = np.linspace(-R_phys, R_phys, res)
    X, Y = np.meshgrid(x, y)
    R_sq = X**2 + Y**2
    aperture_mask = (R_sq <= R_phys**2).astype(np.float64)
    rho_norm = np.sqrt(R_sq) / (R_phys + EPS)
    return X, Y, rho_norm, aperture_mask

def compute_defocus_phase(rho_norm, R_phys, z, z0, wavelength):
    """
    Phase term for defocus: phi_DF = k * Wm * r^2  (PhaseCam3D)
    where Wm = R^2/2 * (1/z - 1/z0), and r = rho_norm * R_phys
    """
    k = 2.0 * np.pi / wavelength
    Wm = (R_phys**2 / 2.0) * (1.0 / z - 1.0 / z0)
    r_phys = rho_norm * R_phys
    phi_df = k * Wm * (r_phys**2)
    return phi_df

def compute_mask_phase_from_height(h, wavelength, delta_n):
    """Given a height map h (meters) produce phase phi_M = k * delta_n * h"""
    k = 2.0 * np.pi / wavelength
    return k * delta_n * h

def compute_psf_from_pupil(pupil, pad_factor=PAD_FACTOR):
    """
    Compute PSF = |FFT{pupil}|^2 with zero padding to increase sampling.
    Returns normalized PSF (2D array)
    """
    res = pupil.shape[0]
    pad_res = int(res * pad_factor)
    # zero-pad pupil to pad_res x pad_res
    pad = np.zeros((pad_res, pad_res), dtype=np.complex128)
    start = (pad_res - res)//2
    pad[start:start+res, start:start+res] = pupil
    U = fftshift(fft2(pad))
    I = np.abs(U)**2
    I = I / (I.sum() + EPS)
    # crop center to original resolution for compactness (optional)
    # Here return full pad size for higher sampling
    return I

# -----------------------------
# Load or build mask height map
# -----------------------------
def load_or_make_mask(res=PUPIL_RES, use_file=False, fname="heightmap.npy"):
    if use_file:
        h = np.load(fname)
        if h.shape != (res, res):
            raise ValueError("Heightmap shape mismatch. Expected (%d,%d)"%(res,res))
        return h.astype(np.float64)
    # default: zero mask
    return np.zeros((res, res), dtype=np.float64)

# -----------------------------
# Metrics
# -----------------------------
def psf_l2_difference(psf_ref, psf_other):
    """Simple L2 norm between two PSFs."""
    return np.linalg.norm(psf_ref.ravel() - psf_other.ravel())

def approximate_fisher_crlb(psfs, z_vals):
    """
    Approximate Fisher information for the depth parameter using discrete PSFs:
    For each depth index i, approximate dPSF/dz via central finite differences.
    Fisher info I(z) ~ sum_pixels ( (dPSF/dz)^2 / (PSF + eps) )
    We return mean info across z (excluding boundaries) and average CRLB = 1 / I
    """
    n = len(z_vals)
    infos = []
    for i in range(1, n-1):
        dz = z_vals[i+1] - z_vals[i-1]
        dpsf_dz = (psfs[i+1] - psfs[i-1]) / dz
        psf = psfs[i]
        info = np.sum((dpsf_dz**2) / (psf + EPS))
        infos.append(info)
    infos = np.array(infos)
    mean_info = np.mean(infos)
    # CRLB ~ 1 / info
    mean_crlb = 1.0 / (mean_info + EPS)
    return mean_info, mean_crlb

# -----------------------------
# Main sweep routine
# -----------------------------
def sweep_R_f(f_list, R_list, z_vals, wavelengths, z0, mask_h):
    """
    For each (f, R) compute PSFs for all z in z_vals (averaged over wavelengths),
    compute metrics (PSF L2 diffs to z0, mean Fisher info / CRLB).
    Returns dictionaries or arrays for later plotting.
    """
    # storage
    metric_l2 = np.zeros((len(f_list), len(R_list)))
    metric_crlb = np.zeros((len(f_list), len(R_list)))
    metric_info = np.zeros((len(f_list), len(R_list)))

    # For reference, pick a small delta_z to compute L2 between z0 and z0+delta
    # But we'll also compute L2 between z0 and each other plane and average.
    for fi, f in enumerate(f_list):
        for ri, R in enumerate(R_list):
            # Build pupil grid for this R
            X, Y, rho_norm, aperture_mask = make_pupil_grid(R_phys=R, res=PUPIL_RES)
            psfs_stack = []
            for z in z_vals:
                psf_sum = None
                # combine over wavelengths
                for wl in wavelengths:
                    # defocus phase
                    phi_df = compute_defocus_phase(rho_norm, R, z, z0, wl)
                    # mask phase
                    phi_mask = compute_mask_phase_from_height(mask_h, wl, delta_n)
                    pupil = aperture_mask * np.exp(1j * (phi_df + phi_mask))
                    psf = compute_psf_from_pupil(pupil, pad_factor=PAD_FACTOR)
                    if psf_sum is None:
                        psf_sum = psf
                    else:
                        psf_sum += psf
                psf_avg = psf_sum / len(wavelengths)
                # optionally crop center region if you want to compare same-sized images later
                psfs_stack.append(psf_avg)
            psfs_stack = np.array(psfs_stack)  # shape (n_z, pad_res, pad_res)
            # compute ref psf (closest to z0)
            idx_ref = np.argmin(np.abs(z_vals - z0))
            psf_ref = psfs_stack[idx_ref]
            # compute average L2 over all z vs ref
            l2vals = [psf_l2_difference(psf_ref, psfs_stack[i]) for i in range(len(z_vals))]
            mean_l2 = np.mean(l2vals)
            metric_l2[fi, ri] = mean_l2
            # compute Fisher approx
            info, crlb = approximate_fisher_crlb(psfs_stack, z_vals)
            metric_info[fi, ri] = info
            metric_crlb[fi, ri] = crlb
            print(f"f={f*1e3:.1f}mm, R={R*1e3:.2f}mm -> meanL2={mean_l2:.3e}, info={info:.3e}, crlb={crlb:.3e}")
    return metric_l2, metric_info, metric_crlb

# -----------------------------
# Run sweep
# -----------------------------
if __name__ == "__main__":
    mask_h = load_or_make_mask(res=PUPIL_RES, use_file=use_mask_file, fname=mask_file)
    L2, INF, CRLB = sweep_R_f(f_list, R_list, z_vals, wavelengths, z0, mask_h)

    # -------------------------
    # Plotting results
    # -------------------------
    # Axis extents
    Fs = f_list * 1e3  # mm
    Rs = R_list * 1e3  # mm

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(L2, origin='lower', extent=[Rs[0], Rs[-1], Fs[0], Fs[-1]], aspect='auto')
    plt.colorbar()
    plt.xlabel("Aperture radius R (mm)")
    plt.ylabel("Focal length f (mm)")
    plt.title("Mean PSF L2 diff vs (f, R)")

    plt.subplot(1,3,2)
    plt.imshow(INF, origin='lower', extent=[Rs[0], Rs[-1], Fs[0], Fs[-1]], aspect='auto')
    plt.colorbar()
    plt.xlabel("Aperture radius R (mm)")
    plt.title("Mean Fisher info vs (f, R)")

    plt.subplot(1,3,3)
    plt.imshow(CRLB, origin='lower', extent=[Rs[0], Rs[-1], Fs[0], Fs[-1]], aspect='auto')
    plt.colorbar()
    plt.xlabel("Aperture radius R (mm)")
    plt.title("Mean CRLB (lower is better)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sweep_heatmaps.png"), dpi=200)
    print("Saved sweep heatmaps to", os.path.join(out_dir, "sweep_heatmaps.png"))

    # Save numpy arrays for offline analysis
    np.save(os.path.join(out_dir, "metric_l2.npy"), L2)
    np.save(os.path.join(out_dir, "metric_info.npy"), INF)
    np.save(os.path.join(out_dir, "metric_crlb.npy"), CRLB)
    print("Saved metric arrays to", out_dir)

    # For a selected (f,R) show PSF slices (example: mid values)
    idx_f = len(f_list)//2
    idx_r = len(R_list)//2
    f_sel = f_list[idx_f]
    R_sel = R_list[idx_r]
    print(f"Selected f={f_sel*1e3:.1f}mm, R={R_sel*1e3:.2f}mm for PSF visualization")

    # compute PSF stack for visualization (recompute small set)
    X, Y, rho_norm, aperture_mask = make_pupil_grid(R_sel, res=PUPIL_RES)
    psfs_vis = []
    for z in z_vals:
        psf_sum = 0
        for wl in wavelengths:
            phi_df = compute_defocus_phase(rho_norm, R_sel, z, z0, wl)
            phi_mask = compute_mask_phase_from_height(mask_h, wl, delta_n)
            pupil = aperture_mask * np.exp(1j * (phi_df + phi_mask))
            psf = compute_psf_from_pupil(pupil, pad_factor=PAD_FACTOR)
            psf_sum += psf
        psf_avg = psf_sum / len(wavelengths)
        psfs_vis.append(psf_avg)
    psfs_vis = np.array(psfs_vis)

    # pick 3 depths to show: near, focus, far
    idx_focus = np.argmin(np.abs(z_vals - z0))
    idx_near = max(0, idx_focus - 4)
    idx_far = min(len(z_vals)-1, idx_focus + 4)

    plt.figure(figsize=(10,3))
    for i, idx in enumerate([idx_near, idx_focus, idx_far]):
        plt.subplot(1,3,i+1)
        plt.imshow(np.log(psfs_vis[idx] + 1e-14), cmap='hot')
        plt.title(f"PSF at z={z_vals[idx]:.2f} m")
        plt.axis('off')
    plt.suptitle(f"PSFs for f={f_sel*1e3:.1f}mm, R={R_sel*1e3:.2f}mm")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psf_example.png"), dpi=200)
    print("Saved PSF examples to", os.path.join(out_dir, "psf_example.png"))

    print("Done.")

"""
Simulates PSFs across several depths 

Computes:

1. L2 difference between in-focus and defocused PSFs (simple discriminability metric).
2. Approx. Fisher information (quantifies how much information PSF contains about depth).
3. CRLB (Cramér–Rao lower bound; lower is better — implies better depth precision).
4. Generates heatmaps showing:
    Mean L2 vs (f, R)
    Fisher information vs (f, R)
    CRLB vs (f, R)
→ You can visually see where your configuration performs best.
"""