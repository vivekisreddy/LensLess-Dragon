# Phase Mask Design for Lensless Depth Estimation

This repository contains code and notes for designing **custom diffractive phase masks** for lensless imaging and depth estimation on lightweight drones.  
The approach builds on **Near-field Phase Retrieval (NfPR)** methods from *PhlatCam3D* and related works, adapting them for fabrication using two-photon polymerization (2PP).

---

## 📖 Overview

A **phase mask** modulates the wavefront of incoming light before it reaches the sensor.  
By engineering the mask’s **phase profile**, we can create depth-dependent Point Spread Functions (PSFs) that encode scene depth into captured images.  
A neural network or deconvolution algorithm can then recover a **depth map** for navigation.

This workflow allows:
- Lensless, lightweight cameras for drones
- Depth cues via chromatic aberration or engineered PSFs
- Fabrication-ready mask profiles

---

## 🧮 Core Math

1. **Fresnel Propagation**  
   Forward propagation from mask → sensor uses the Fresnel diffraction model:
   \[
   U(x,y; z) = \mathcal{F}^{-1} \{ \mathcal{F}\{ U(x,y) \} H(f_x, f_y) \}
   \]  
   where \( H(f_x, f_y) \) is the Fresnel transfer function.

2. **Near-field Phase Retrieval (NfPR)**  
   Iterative algorithm inspired by Gerchberg–Saxton, but using Fresnel propagation:  
   - Backpropagate sensor field → mask plane  
   - Enforce unit amplitude at mask plane  
   - Forward propagate → sensor plane  
   - Enforce target PSF amplitude  

3. **Phase → Height Map Conversion**  
   The fabricated mask must have a height profile \( h(x,y) \) derived from phase:  
   \[
   h(x,y) = \frac{\lambda}{2\pi (n - 1)} \, \phi(x,y)
   \]  
   where \( n \) is the refractive index of the mask material.

---

## ⚙️ Parameters

- **Footprint (X × Y):** 1.5 mm × 1.5 mm  
- **Maximum Height (Z):** 2.1 µm  
- **Step Size (layer height):** 0.2 µm (~10 layers)  
- **Mask-to-Sensor Distance (d):** 0.5–1.0 mm (adjust for drone packaging)  
- **Wavelengths:** 470 nm, 530 nm, 610 nm (RGB) or single λ = 532 nm  
- **Material Index:** n ≈ 1.5 (photoresist polymer)  

---

## 📂 Repository Structure

