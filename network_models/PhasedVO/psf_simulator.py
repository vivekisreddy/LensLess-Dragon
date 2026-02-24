# psf_simulator.py
"""
Depth-dependent PSF simulator (PhaseCam3D-style optics) — pure NumPy/SciPy.

Features:
- Physical defocus phase (PhaseCam3D math)
- Optional phase mask: either raw heightmap (meters) or Zernike coefficients + basis (.mat)
- Multi-wavelength PSFs (R/G/B)
- Pupil sampling, zero-padding for FFT, PSF normalization
- Metrics: mean L2 difference to ref depth, approximate Fisher info & CRLB
- Export PSF stack as numpy for CodedVO (depth, H, W, 3)
"""

import numpy as np
from scipy.fft import fft2, fftshift
from scipy.io import loadmat
from scipy.ndimage import zoom
import os

# -----------------------------
# Utility / Helper functions
# -----------------------------
def make_pupil_grid(R_phys, N):
    """
    Create a square grid [-R, R] x [-R, R] sampled N x N, plus circular aperture mask.
    Returns: X, Y, Rho, aperture_mask
    Units: meters
    """
    x = np.linspace(-R_phys, R_phys, N)
    X, Y = np.meshgrid(x, x)
    Rho = np.sqrt(X**2 + Y**2)
    aperture = (Rho <= R_phys).astype(np.float64)
    return X, Y, Rho, aperture

def load_heightmap_npy(fname, N):
    """
    Load a heightmap in meters and resample to NxN (bicubic).
    """
    h = np.load(fname)
    if h.shape != (N, N):
        # resample
        zoom_f = (N / h.shape[0], N / h.shape[1])
        h = zoom(h, zoom_f, order=3)
    return h.astype(np.float64)

def load_zernike_heightmap(matfile, coeffs, N):
    """
    Load zernike basis from .mat (expects 'u2' variable shaped (N*N, n_modes) or similar).
    'coeffs' should be vector length n_modes. Returns NxN heightmap (meters).
    If matfile doesn't have u2, raise.
    NOTE: PhaseCam3D's u2 expects a specific normalization; user should match units.
    """
    m = loadmat(matfile)
    if 'u2' not in m:
        raise ValueError("MAT file has no 'u2' key for zernike basis")
    u2 = m['u2']  # shape (N*N, n_modes)
    # multiply and reshape
    g = np.matmul(u2, coeffs.reshape(-1, 1)).squeeze()  # length N*N
    # default offset used in PhaseCam3D: add reference wavelength or clamp; keep raw here
    h = g.reshape(int(np.sqrt(g.size)), int(np.sqrt(g.size)))
    # ensure shape NxN requested
    if h.shape != (N, N):
        # resample
        zoom_f = (N / h.shape[0], N / h.shape[1])
        h = zoom(h, zoom_f, order=3)
    return h.astype(np.float64)

def compute_defocus_phase_cam(Rho, R_phys, z, z0, wavelength):
    """
    PhaseCam3D-style defocus:
      Wm = R^2 / 2 * (1/z - 1/z0)
      phi_df = k * Wm * r^2
    where r = Rho (meters).
    Returns phi_df in radians (N x N).
    """
    k = 2.0 * np.pi / wavelength
    Wm = (R_phys**2 / 2.0) * (1.0 / z - 1.0 / z0)
    phi_df = k * Wm * (Rho**2)
    return phi_df

def compute_mask_phase(h, wavelength, delta_n=0.5):
    """
    phi_mask = k * delta_n * h(x,y)
    h in meters, delta_n mask refractive index difference (n_mask - n_air)
    """
    k = 2.0 * np.pi / wavelength
    return k * delta_n * h

def pupil_to_psf(pupil_complex, pad_factor=2):
    """
    Compute PSF as |FFT{pupil}|^2 with zero padding.
    pupil_complex: N x N complex array
    pad_factor: integer
    Returns: PSF as padN x padN float normalized to sum=1
    """
    N = pupil_complex.shape[0]
    padN = int(N * pad_factor)
    pad = np.zeros((padN, padN), dtype=np.complex128)
    start = (padN - N) // 2
    pad[start:start+N, start:start+N] = pupil_complex
    U = fftshift(fft2(pad))
    I = np.abs(U)**2
    s = I.sum()
    if s <= 0:
        return I.astype(np.float32)
    return (I / s).astype(np.float32)

# -----------------------------
# Metrics
# -----------------------------
def psf_l2(psf_ref, psf_other):
    return np.linalg.norm((psf_ref - psf_other).ravel())

def approximate_fisher_crlb(psf_stack, z_vals):
    """
    Approximate Fisher information for depth using finite differences.
    psf_stack: (n_z, H, W) for *one* color channel
    z_vals: depth list (n_z)
    Returns: mean_info, mean_crlb, infos_array
    """
    n = len(z_vals)
    infos = []
    for i in range(1, n-1):
        dz = z_vals[i+1] - z_vals[i-1]
        dpsf_dz = (psf_stack[i+1] - psf_stack[i-1]) / dz
        psf = psf_stack[i]
        # avoid divide by zero by adding eps
        eps = 1e-12
        info = np.sum((dpsf_dz**2) / (psf + eps))
        infos.append(info)
    infos = np.array(infos)
    mean_info = np.mean(infos) if infos.size > 0 else 0.0
    mean_crlb = 1.0 / (mean_info + 1e-12)
    return mean_info, mean_crlb, infos

# -----------------------------
# Top-level PSF generator
# -----------------------------
def generate_psf_stack(
    wavelengths,
    z_vals,
    R_phys,
    N_pupil=256,
    pad_factor=2,
    z0=1.0,
    heightmap=None,
    zernike_matfile=None,
    zernike_coeffs=None,
    delta_n=0.5,
    aperture_mask_override=None,
):
    """
    Generate PSF stack across depths and wavelengths.
    Returns: psf_stack shaped (n_z, padN, padN, 3) [float32]
    - wavelengths: array-like of length 3 [R,G,B] in meters
    - z_vals: array-like depths in meters (object distances)
    - R_phys: aperture radius in meters
    - N_pupil: resolution across pupil (square)
    - pad_factor: FFT padding factor
    - z0: in-focus object distance (meters)
    - heightmap: optional NxN array in meters (if provided)
    - zernike_matfile + zernike_coeffs: alternative to heightmap
    - delta_n: refractive index difference (mask material - air)
    - aperture_mask_override: optional NxN mask (1 inside, 0 outside)
    """
    # Prepare pupil grid
    X, Y, Rho, aperture = make_pupil_grid(R_phys, N_pupil)
    if aperture_mask_override is not None:
        aperture = aperture * (aperture_mask_override.astype(np.float64))

    # Build heightmap if needed
    if zernike_matfile is not None and zernike_coeffs is not None:
        # user provided zernike basis .mat and coeffs
        h = load_zernike_heightmap(zernike_matfile, np.asarray(zernike_coeffs), N_pupil)
    elif heightmap is not None:
        # load/resample if necessary
        h = heightmap
        if h.shape != (N_pupil, N_pupil):
            from scipy.ndimage import zoom
            h = zoom(h, (N_pupil/h.shape[0], N_pupil/h.shape[1]), order=3)
    else:
        h = np.zeros((N_pupil, N_pupil), dtype=np.float64)

    # ensure mask zero outside aperture
    h = h * aperture

    n_depths = len(z_vals)
    padN = int(N_pupil * pad_factor)
    psf_stack = np.zeros((n_depths, padN, padN, 3), dtype=np.float32)

    for iz, z in enumerate(z_vals):
        # sum psf across wavelengths into RGB channels (store per color separately)
        psf_per_color = np.zeros((padN, padN, 3), dtype=np.float64)
        for iw, wl in enumerate(wavelengths):
            # defocus + mask phase
            phi_df = compute_defocus_phase_cam(Rho, R_phys, z, z0, wl)
            phi_m = compute_mask_phase(h, wl, delta_n)
            pupil = aperture * np.exp(1j * (phi_df + phi_m))
            psf = pupil_to_psf(pupil, pad_factor=pad_factor)  # padN x padN
            psf_per_color[..., iw] = psf
        # assign to stack (R,G,B channels)
        # psf_per_color currently already normalized per wl; keep as-is
        psf_stack[iz] = psf_per_color.astype(np.float32)

    # Normalize each channel per depth to sum=1 (safety)
    for iz in range(n_depths):
        for c in range(3):
            s = psf_stack[iz, :, :, c].sum()
            if s > 0:
                psf_stack[iz, :, :, c] /= s

    return psf_stack  # (n_z, padN, padN, 3)

# -----------------------------
# Helper: approximate image-plane sampling
# -----------------------------
def approx_image_plane_sampling(lambda_ref, f, R_phys):
    """
    approx delta_x_img ~ lambda * f / (2 * R_phys)
    This gives the spacing of image-plane samples corresponding to pupil sampling.
    Compare to sensor pixel pitch to detect undersampling.
    """
    return lambda_ref * f / (2.0 * R_phys)

# -----------------------------
# Convenience: sweep R and f and compute metrics
# -----------------------------
def sweep_and_metric(
    wavelengths,
    z_vals,
    f_list,
    R_list,
    N_pupil=256,
    pad_factor=2,
    z0=1.0,
    heightmap=None,
    delta_n=0.5,
    pixel_pitch=1.55e-6,
):
    """
    Sweeps focal lengths (f_list) and aperture radii (R_list).
    For each pair compute PSF stack, compute two metrics:
      - mean L2 distance between PSFs at all z and the reference at z0
      - mean Fisher info (averaged over depths and channels)
    Returns dict with metric arrays (shape len(f_list) x len(R_list)).
    """
    L2_metrics = np.zeros((len(f_list), len(R_list)))
    Info_metrics = np.zeros_like(L2_metrics)
    CRLB_metrics = np.zeros_like(L2_metrics)

    for i, f in enumerate(f_list):
        for j, R in enumerate(R_list):
            psfs = generate_psf_stack(
                wavelengths=wavelengths,
                z_vals=z_vals,
                R_phys=R,
                N_pupil=N_pupil,
                pad_factor=pad_factor,
                z0=z0,
                heightmap=heightmap,
                delta_n=delta_n,
            )  # (n_z, padN, padN, 3)
            padN = psfs.shape[1]
            idx_ref = np.argmin(np.abs(np.array(z_vals) - z0))
            # compute mean L2 across colors and depths
            l2_list = []
            info_list = []
            for c in range(3):
                psf_channel_stack = psfs[:, :, :, c]
                ref = psf_channel_stack[idx_ref]
                for k in range(len(z_vals)):
                    l2_list.append(psf_l2(ref, psf_channel_stack[k]))
                # fisher
                mean_info, mean_crlb, infos = approximate_fisher_crlb(psf_channel_stack, z_vals)
                info_list.append(mean_info)
            mean_l2 = np.mean(l2_list)
            mean_info = np.mean(info_list)
            mean_crlb = 1.0 / (mean_info + 1e-12)
            L2_metrics[i, j] = mean_l2
            Info_metrics[i, j] = mean_info
            CRLB_metrics[i, j] = mean_crlb
            print(f"[Sweep] f={f*1e3:.1f}mm R={R*1e3:.2f}mm L2={mean_l2:.3e} Info={mean_info:.3e} CRLB={mean_crlb:.3e}")
    return {
        'f_list': np.array(f_list),
        'R_list': np.array(R_list),
        'L2': L2_metrics,
        'Info': Info_metrics,
        'CRLB': CRLB_metrics
    }

# -----------------------------
# Example script usage (if run)
# -----------------------------
if __name__ == "__main__":
    # IMX477 / ArduCAM HQ defaults
    pixel_pitch = 1.55e-6
    sensor_w, sensor_h = 6.287e-3, 4.712e-3
    wavelengths = np.array([610e-9, 530e-9, 470e-9])  # R,G,B
    # focal lengths to inspect (m)
    f_list = [8e-3, 10e-3, 12e-3, 16e-3]
    # aperture radii to inspect (m)
    R_list = np.linspace(0.5e-3, 3.0e-3, 6)  # 0.5mm .. 3.0mm
    # depths (z) and reference focus z0
    z0 = 1.0
    z_vals = np.linspace(0.4, 2.0, 21)
    # choose pupil sampling
    N_pupil = 256
    pad_factor = 2

    # No mask initially (flat). If you have heightmap set it here:
    heightmap = None
    # OR: if you have heightmap file:
    # heightmap = load_heightmap_npy("heightmap.npy", N_pupil)

    # Quick single PSF stack generation example for one (R,f)
    R_test = 1.25e-3
    psf_stack = generate_psf_stack(
        wavelengths=wavelengths,
        z_vals=z_vals,
        R_phys=R_test,
        N_pupil=N_pupil,
        pad_factor=pad_factor,
        z0=z0,
        heightmap=heightmap,
        delta_n=0.5
    )
    print("PSF stack shape (depth, H, W, 3):", psf_stack.shape)

    # Save example PSF stack (compatible with CodedVO: move last axis to channel)
    outname = "psfs_example.npy"
    np.save(outname, psf_stack)
    print("Saved example PSF stack to", outname)

    # Run sweep (this will be slower)
    metrics = sweep_and_metric(
        wavelengths=wavelengths,
        z_vals=z_vals,
        f_list=f_list,
        R_list=R_list,
        N_pupil=N_pupil,
        pad_factor=pad_factor,
        z0=z0,
        heightmap=heightmap,
        delta_n=0.5,
        pixel_pitch=pixel_pitch
    )
    # Save metrics
    np.savez("psf_sweep_metrics.npz", **metrics)
    print("Saved sweep metrics to psf_sweep_metrics.npz")
