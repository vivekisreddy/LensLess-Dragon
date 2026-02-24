import numpy as np
from math import factorial
import scipy.ndimage as ndi


# Key imaging parameters
DOF_range = (-1.5, 1.5)    # in micrometers
signal_photons = 2000      # detected photons per emitter
background_photons = 28    # background photons per pixel
pixel_size = 0.1           # 0.1 µm per pixel (100 nm)
wavelength = 0.67          # 0.67 µm emission wavelength
NA = 1.4                   # numerical aperture of the objective
n_medium = 1.52            # refractive index (e.g., oil immersion)



# Pupil grid (256x256, normalized radius)
N = 256
coords = (np.arange(N) - N/2) / (N/2)  # range -1 to 1
xg, yg = np.meshgrid(coords, coords)
r = np.sqrt(xg**2 + yg**2)
theta = np.arctan2(yg, xg)
aperture = (r <= 1.0).astype(float)  # circular pupil aperture mask

# Function to compute Zernike radial polynomials https://en.wikipedia.org/wiki/Zernike_polynomials
def zernike_radial(n, m, r):
    R = np.zeros_like(r)
    m = abs(m)
    if (n - m) % 2 != 0:
        return R  # zero if parity condition not met
    for k in range((n - m)//2 + 1):
        c = ((-1)**k * factorial(n - k) /
             (factorial(k) * factorial((n+m)//2 - k) * factorial((n-m)//2 - k)))
        R += c * r**(n - 2*k)
    return R

# Define Zernike polynomial Z_n^m (unnormalized) 
def zernike(n, m, r, theta):
    Rad = zernike_radial(n, m, r)
    if m >= 0:
        return Rad * np.cos(m * theta)
    else:
        return Rad * np.sin(-m * theta)

# Generate first 55 Zernike modes (excluding piston and tilt)
zernike_modes = []
mode_count = 0
for n in range(0, 10):  # n=0 to 9 gives 55 modes including piston
    for m in range(-n, n+1, 2):
        if n == 0 and m == 0:
            continue  # skip piston
        if n == 1:
            continue  # skip tilt modes n=1 (m = ±1)
        if (n - abs(m)) % 2 == 0:
            Znm = zernike(n, m, r, theta)
            zernike_modes.append(Znm)
            mode_count += 1
            if mode_count == 55:
                break
    if mode_count == 55:
        break

print(f"Generated {len(zernike_modes)} Zernike modes.")
# >> Generated 55 Zernike modes.


# Precompute constant pupil amplitude (within aperture) and coordinate scaling
ap_area = np.sum(aperture)  # area (pixels) of the aperture

# Defocus phase function
def defocus_phase(z):
    # Phase delay for defocus z at pupil radius r
    return (2*np.pi/wavelength) * n_medium * z * (np.sqrt(np.maximum(0, 1 - (NA/n_medium * r)**2)) - 1)

# Function to simulate PSF intensity for given mask coefficients and defocus z
def simulate_psf_intensity(coefs, z):
    # Compute total pupil phase = mask phase + defocus phase
    phase_mask = np.zeros_like(r)
    for a, Z in zip(coefs, zernike_modes):
        phase_mask += a * Z
    total_phase = phase_mask + defocus_phase(z)
    # Pupil field with phase (amplitude = 1 inside aperture)
    pupil_field = aperture * np.exp(1j * total_phase)
    # Fourier transform to image plane
    pupil_centered = np.fft.ifftshift(pupil_field)       # shift pupil so center is at (0,0) for FFT
    field = np.fft.fft2(pupil_centered) / N              # normalized FFT
    intensity = np.fft.fftshift(np.abs(field)**2)        # intensity, shifted so PSF center is in middle
    return intensity

# Example: PSF for a flat pupil (no phase mask) at focus and slight defocus
coefs_flat = np.zeros(len(zernike_modes))
psf_focus = simulate_psf_intensity(coefs_flat, z=0.0)
psf_defocus = simulate_psf_intensity(coefs_flat, z=0.5)  # 0.5 µm defocus
print(f"PSF sum at focus = {psf_focus.sum():.2f}, PSF sum at 0.5µm defocus = {psf_defocus.sum():.2f}")
# >> PSF sum at focus = 51431.00, PSF sum at 0.5µm defocus = 51431.00


# Finite difference step sizes (in micrometers)
dx = 0.02
dz = 0.02

def compute_crlb(coefs):
    # Arrays to accumulate CRLB as function of z
    zs = np.arange(DOF_range[0], DOF_range[1] + 1e-9, 0.25)  # z positions at 250 nm steps
    crlb_x = []
    crlb_y = []
    crlb_z = []
    for z in zs:
        # Expected signal intensity at this z (scale to photon counts):
        I = simulate_psf_intensity(coefs, z)
        # Scale so that total signal = signal_photons
        I_signal = I * (signal_photons / ap_area)
        mu = I_signal + background_photons  # include background in mean
        
        # Partial derivatives via shifting (using spline interpolation of order 1)
        # dI/dx: shift image +/- dx in x-direction
        shift_px = dx / pixel_size  # shift in pixels
        I_x_plus  = ndi.shift(I_signal, shift=(0,  shift_px), order=1, mode='constant', cval=0.0)
        I_x_minus = ndi.shift(I_signal, shift=(0, -shift_px), order=1, mode='constant', cval=0.0)
        dmu_dx = (I_x_plus - I_x_minus) / (2 * dx)
        # dI/dy: shift image +/- dx in y-direction
        I_y_plus  = ndi.shift(I_signal, shift=( shift_px, 0), order=1, mode='constant', cval=0.0)
        I_y_minus = ndi.shift(I_signal, shift=(-shift_px, 0), order=1, mode='constant', cval=0.0)
        dmu_dy = (I_y_plus - I_y_minus) / (2 * dx)
        # dI/dz: simulate at z+dz and z-dz
        I_z_plus  = simulate_psf_intensity(coefs, z + dz) * (signal_photons / ap_area)
        I_z_minus = simulate_psf_intensity(coefs, z - dz) * (signal_photons / ap_area)
        dmu_dz = (I_z_plus - I_z_minus) / (2 * dz)
        
        # Compute Fisher Information matrix elements by summing over pixels
        inv_mu = 1.0 / mu  # elementwise inverse
        Ixx = np.sum(inv_mu * dmu_dx * dmu_dx)
        Iyy = np.sum(inv_mu * dmu_dy * dmu_dy)
        Izz = np.sum(inv_mu * dmu_dz * dmu_dz)
        Ixy = np.sum(inv_mu * dmu_dx * dmu_dy)
        Ixz = np.sum(inv_mu * dmu_dx * dmu_dz)
        Iyz = np.sum(inv_mu * dmu_dy * dmu_dz)
        FIM = np.array([[Ixx, Ixy, Ixz],
                        [Ixy, Iyy, Iyz],
                        [Ixz, Iyz, Izz]])
        # Invert Fisher matrix to get CRLB
        try:
            cov = np.linalg.inv(FIM)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(FIM)  # handle singular matrix by pseudo-inverse
        crlb = np.diag(cov)  # variance lower bounds for x,y,z
        crlb_x.append(crlb[0]); crlb_y.append(crlb[1]); crlb_z.append(crlb[2])
    return zs, np.array(crlb_x), np.array(crlb_y), np.array(crlb_z)

# Test CRLB for the unaberrated PSF (flat mask)
zs, crlb_x_flat, crlb_y_flat, crlb_z_flat = compute_crlb(coefs_flat)
print(f"CRLB_z at focus (z=0) for flat PSF: {crlb_z_flat[zs==0][0]:.2e}")
# >> CRLB_z at focus (z=0) for flat PSF:  inf (effectively, since derivative ~0 at z=0)


# Define the objective function for optimization: sum of sqrt(CRLB) over x,y,z and all z-samples
def objective(coefs):
    zs, crlb_x, crlb_y, crlb_z = compute_crlb(coefs)
    return np.sum(np.sqrt(crlb_x) + np.sqrt(crlb_y) + np.sqrt(crlb_z))


# Optimize over first 12 Zernike modes (for speed)
param_count = 12
initial_coefs = (np.random.rand(param_count) - 0.5) * 1.0  # small random init

# Pad the coefficient vector to full length (55) for evaluation
def pad_coefs(p):
    full = np.zeros(len(zernike_modes))
    full[:len(p)] = p
    return full

# Objective for the subset of modes
def objective_subset(p):
    return objective(pad_coefs(p))

# Run a local optimization (e.g., Nelder-Mead or Powell) 
from scipy.optimize import minimize
result = minimize(objective_subset, initial_coefs, method='Nelder-Mead', options={'maxiter':50})
opt_coefs_12 = pad_coefs(result.x)
print("Optimized cost value:", result.fun)


import matplotlib.pyplot as plt

def plot_phase_mask(phase, aperture=None, title="Pupil Phase (wrapped in [-π, π])"):
    """
    Display a 2D phase mask (in radians) wrapped to [-π, π].
    phase:    2D numpy array of phase in radians
    aperture: optional 2D mask (1 inside pupil, 0 outside). Outside set to NaN for display.
    """
    wrapped = np.angle(np.exp(1j * phase))  # wrap to [-π, π]
    if aperture is not None:
        wrapped = wrapped.copy()
        wrapped[aperture == 0] = np.nan

    # Show in normalized pupil coordinates [-1, 1]
    h, w = phase.shape
    extent = (-1, 1, -1, 1)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(wrapped, extent=extent, origin="lower", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Normalized pupil $x_0$")
    plt.ylabel("Normalized pupil $y_0$")
    cbar = plt.colorbar(im)
    cbar.set_label("Phase (rad)")
    plt.tight_layout()
    plt.show()

    