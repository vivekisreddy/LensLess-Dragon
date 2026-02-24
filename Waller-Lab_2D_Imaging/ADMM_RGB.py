import numpy as np
import numpy.fft as fft
from PIL import Image
import matplotlib.pyplot as plt
import yaml

# ---------------------------
# Original loadData, but allow RGB
# ---------------------------
def loadData(show_im=True):
    psf = np.array(Image.open(psfname), dtype='float32')
    data = np.array(Image.open(imgname), dtype='float32')

    # If RGBA, drop alpha
    if psf.ndim == 3 and psf.shape[2] == 4:
        psf = psf[..., :3]
    if data.ndim == 3 and data.shape[2] == 4:
        data = data[..., :3]

    # Background subtraction (per-channel if RGB)
    if psf.ndim == 2:
        bg = np.mean(psf[5:15, 5:15])
    else:
        bg = np.mean(psf[5:15, 5:15, :], axis=(0, 1))  # (3,)

    psf = psf - bg
    data = data - bg

    # Downsample by factor f (must be 1/(2^k)) -- original approach, but RGB-safe
    def resize(img, factor):
        num = int(-np.log2(factor))
        for _ in range(num):
            img = 0.25 * (
                img[::2, ::2, ...] + img[1::2, ::2, ...] +
                img[::2, 1::2, ...] + img[1::2, 1::2, ...]
            )
        return img

    psf = resize(psf, f)
    data = resize(data, f)

    # Normalize power (per-channel if RGB) -- original intent, RGB-safe
    def normalize(img):
        if img.ndim == 2:
            denom = np.linalg.norm(img.ravel()) + 1e-12
            return img / denom
        else:
            out = img.copy()
            for c in range(out.shape[2]):
                denom = np.linalg.norm(out[..., c].ravel()) + 1e-12
                out[..., c] /= denom
            return out

    psf = normalize(psf)
    data = normalize(data)

    if show_im:
        plt.figure()
        if psf.ndim == 2:
            plt.imshow(psf, cmap="gray")
        else:
            p = psf - psf.min()
            p = p / (p.max() + 1e-12)
            plt.imshow(p)
        plt.title("PSF")

        plt.figure()
        if data.ndim == 2:
            plt.imshow(data, cmap="gray")
        else:
            d = data - data.min()
            d = d / (d.max() + 1e-12)
            plt.imshow(d)
        plt.title("Raw data")
        plt.show()

    return psf, data


# ---------------------------
# Original ADMM code (unchanged)
# ---------------------------
def U_update(eta, image_est, tau):
    return SoftThresh(Psi(image_est) + eta/mu2, tau/mu2)

def SoftThresh(x, tau):
    return np.sign(x)*np.maximum(0, np.abs(x) - tau)

def Psi(v):
    return np.stack((np.roll(v,1,axis=0) - v, np.roll(v, 1, axis=1) - v), axis=2)

def X_update(xi, image_est, H_fft, sensor_reading, X_divmat):
    return X_divmat * (xi + mu1*M(image_est, H_fft) + CT(sensor_reading))

def M(vk, H_fft):
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(vk))*H_fft)))

def C(Mmat):
    top = (full_size[0] - sensor_size[0])//2
    bottom = (full_size[0] + sensor_size[0])//2
    left = (full_size[1] - sensor_size[1])//2
    right = (full_size[1] + sensor_size[1])//2
    return Mmat[top:bottom,left:right]

def CT(b):
    v_pad = (full_size[0] - sensor_size[0])//2
    h_pad = (full_size[1] - sensor_size[1])//2
    return np.pad(b, ((v_pad, v_pad), (h_pad, h_pad)), 'constant', constant_values=(0,0))

def precompute_X_divmat():
    return 1./(CT(np.ones(sensor_size)) + mu1)

def W_update(rho, image_est):
    return np.maximum(rho/mu3 + image_est, 0)

def r_calc(w, rho, u, eta, x, xi, H_fft):
    return (mu3*w - rho)+PsiT(mu2*u - eta) + MT(mu1*x - xi, H_fft)

def V_update(w, rho, u, eta, x, xi, H_fft, R_divmat):
    freq_space_result = R_divmat*fft.fft2( fft.ifftshift(r_calc(w, rho, u, eta, x, xi, H_fft)) )
    return np.real(fft.fftshift(fft.ifft2(freq_space_result)))

def PsiT(U):
    diff1 = np.roll(U[...,0],-1,axis=0) - U[...,0]
    diff2 = np.roll(U[...,1],-1,axis=1) - U[...,1]
    return diff1 + diff2

def MT(x, H_fft):
    x_zeroed = fft.ifftshift(x)
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(x_zeroed) * np.conj(H_fft))))

def precompute_PsiTPsi():
    PsiTPsi = np.zeros(full_size)
    PsiTPsi[0,0] = 4
    PsiTPsi[0,1] = PsiTPsi[1,0] = PsiTPsi[0,-1] = PsiTPsi[-1,0] = -1
    return fft.fft2(PsiTPsi)

def precompute_R_divmat(H_fft, PsiTPsi):
    MTM_component = mu1*(np.abs(np.conj(H_fft)*H_fft))
    PsiTPsi_component = mu2*np.abs(PsiTPsi)
    id_component = mu3
    return 1./(MTM_component + PsiTPsi_component + id_component)

def xi_update(xi, V, H_fft, X):
    return xi + mu1*(M(V,H_fft) - X)

def eta_update(eta, V, U):
    return eta + mu2*(Psi(V) - U)

def rho_update(rho, V, W):
    return rho + mu3*(V - W)

def init_Matrices(H_fft):
    X = np.zeros(full_size)
    U = np.zeros((full_size[0], full_size[1], 2))
    V = np.zeros(full_size)
    W = np.zeros(full_size)

    xi = np.zeros_like(M(V,H_fft))
    eta = np.zeros_like(Psi(V))
    rho = np.zeros_like(W)
    return X,U,V,W,xi,eta,rho

def precompute_H_fft(psf):
    return fft.fft2(fft.ifftshift(CT(psf)))

def ADMM_Step(X,U,V,W,xi,eta,rho, precomputed):
    H_fft, data, X_divmat, R_divmat = precomputed
    U = U_update(eta, V, tau)
    X = X_update(xi, V, H_fft, data, X_divmat)
    V = V_update(W, rho, U, eta, X, xi, H_fft, R_divmat)
    W = W_update(rho, V)
    xi = xi_update(xi, V, H_fft, X)
    eta = eta_update(eta, V, U)
    rho = rho_update(rho, V, W)
    return X,U,V,W,xi,eta,rho


# ---------------------------
# Original runADMM (grayscale) unchanged
# ---------------------------
def runADMM(psf2d, data2d):
    # STRICT: no cropping/padding; must match sizes
    if psf2d.shape != data2d.shape:
        raise ValueError(f"PSF shape {psf2d.shape} must equal data shape {data2d.shape} (no cropping).")

    H_fft = precompute_H_fft(psf2d)
    X,U,V,W,xi,eta,rho = init_Matrices(H_fft)
    X_divmat = precompute_X_divmat()
    PsiTPsi = precompute_PsiTPsi()
    R_divmat = precompute_R_divmat(H_fft, PsiTPsi)

    image = None
    for i in range(iters):
        X,U,V,W,xi,eta,rho = ADMM_Step(X,U,V,W,xi,eta,rho, [H_fft, data2d, X_divmat, R_divmat])
        if i % disp_pic == 0:
            print(i)
            image = C(V)
            image[image < 0] = 0
            plt.figure()
            plt.imshow(image, cmap='gray')
            plt.title(f'Reconstruction after iteration {i}')
            plt.axis("off")
            plt.show()

    if image is None:
        image = C(V)
        image[image < 0] = 0
    return image


# ---------------------------
# RGB wrapper (only added feature)
# ---------------------------
def runADMM_rgb(psf, data):
    # grayscale measurement
    if data.ndim == 2:
        psf2d = psf if psf.ndim == 2 else psf[..., 0]
        return runADMM(psf2d, data)

    # RGB measurement: choose PSF per channel or reuse grayscale PSF
    if psf.ndim == 2:
        psf_rgb = np.stack([psf, psf, psf], axis=2)
    else:
        psf_rgb = psf

    if psf_rgb.shape[:2] != data.shape[:2]:
        raise ValueError(
            f"PSF spatial size {psf_rgb.shape[:2]} must equal data spatial size {data.shape[:2]} (no cropping)."
        )

    recon = []
    for c in range(3):
        print(f"\n=== Reconstructing channel {c} (0=R,1=G,2=B) ===")
        recon_c = runADMM(psf_rgb[..., c], data[..., c])
        recon.append(recon_c)

    return np.stack(recon, axis=2)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    params = yaml.safe_load(open("admm_config.yml"))
    for k, v in params.items():
        exec(k + "=v")

    psf, data = loadData(True)

    # sensor_size / full_size are defined by the measurement size (like original code)
    if data.ndim == 2:
        sensor_size = np.array(data.shape)
    else:
        sensor_size = np.array(data[..., 0].shape)
    full_size = 2 * sensor_size

    final_im = runADMM_rgb(psf, data)

    plt.figure()
    if final_im.ndim == 2:
        plt.imshow(final_im, cmap="gray")
    else:
        im = final_im - final_im.min()
        im = im / (im.max() + 1e-12)
        plt.imshow(im)
    plt.title(f"Final reconstructed image after {iters} iterations")
    plt.axis("off")
    plt.show()

    saveim = input('Save final image? (y/n) ')
    if saveim.lower() == 'y':
        filename = input('Name of file (no extension): ')
        if final_im.ndim == 2:
            plt.imsave(filename + ".png", final_im, cmap="gray")
        else:
            im = final_im - final_im.min()
            im = im / (im.max() + 1e-12)
            plt.imsave(filename + ".png", im)