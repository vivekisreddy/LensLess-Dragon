import numpy as np
import scipy.ndimage as nd
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# ---------- helpers ----------
def fresnel_transfer_function(N, dx, wavelength, z):
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(1j * 2*np.pi * z / wavelength * np.sqrt(1 - (wavelength*FX)**2 - (wavelength*FY)**2))
    return H

def fresnel_prop(u, H):
    U = fft2(u)
    return ifft2(U * H)

# ---------- parameters ----------
Nx = 256
aperture_size = 1.5e-3          # 1.5 mm
dx = aperture_size / Nx
wavelengths = [530e-9]         # list of lambdas (start with green)
d = 0.8e-3                     # mask-to-sensor distance (adjust)
n = 1.5                        # mask material index
dn = n - 1.0

# pupil aperture
x = (np.arange(Nx) - Nx//2) * dx
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)
R_ap = aperture_size/2
A = (R <= R_ap).astype(np.float32)

# ---------- target PSF (example: small Gaussian)
def gaussian_psf(N, sigma_pixels):
    xx = np.arange(-N//2, N//2)
    Xg, Yg = np.meshgrid(xx, xx)
    g = np.exp(-(Xg**2 + Yg**2) / (2*sigma_pixels**2))
    g /= g.sum()
    return fftshift(g)

target_psf = gaussian_psf(Nx, sigma_pixels=6)

# ---------- Near-field Phase Retrieval (NfPR) ----------
def nfrp_design(target_psf, wavelength, d, iterations=200):
    k = 2*np.pi / wavelength
    # init sensor field: amplitude = sqrt(target_psf), random phase
    amp = np.sqrt(target_psf)
    phase = np.random.uniform(0, 2*np.pi, (Nx, Nx))
    Mp = amp * np.exp(1j*phase)
    H_forward = fresnel_transfer_function(Nx, dx, wavelength, d)
    H_backward = np.conj(H_forward)  # approx inverse

    for it in range(iterations):
        # back-propagate to mask plane
        Mphi = fresnel_prop(Mp, H_backward)
        # constrain amplitude to 1 (pure phase)
        phi = np.angle(Mphi)
        Mphi_new = np.exp(1j * phi)
        # forward propagate to sensor
        Mp_new = fresnel_prop(Mphi_new, H_forward)
        # enforce sensor amplitude = target amplitude
        Mp = amp * np.exp(1j * np.angle(Mp_new))

    # final mask phase map (wrapped 0..2pi)
    mask_phase = np.mod(phi, 2*np.pi)
    # convert to height map
    h = (wavelength / (2*np.pi * dn)) * mask_phase
    return mask_phase, h

mask_phase, height_map = nfrp_design(target_psf, wavelengths[0], d, iterations=200)

# ---------- simulate PSF from generated mask ----------
def pupil_from_height(h, wavelength, dn, A):
    k = 2*np.pi / wavelength
    phiM = k * dn * h
    P = A * np.exp(1j * phiM)
    return P

P = pupil_from_height(height_map, wavelengths[0], dn, A)
U = fft2(P)
PSF_sim = np.abs(U)**2
PSF_sim /= PSF_sim.max()

# ---------- convolve with scene layer (demo) ----------
# example scene L (grayscale); convolved image:
# image = scipy.signal.fftconvolve(L, PSF_sim, mode='same')  # or use FFT multiplication

