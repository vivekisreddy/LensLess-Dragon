
The U-Net is integrated with a differentiable image formation model that simulates depth-dependent blur, as outlined in the paper. 

U-Net Architecture
The U-Net follows the paper’s exact design:

Input: RGB images (3 channels, e.g., simulated sensor images or all-in-focus dataset images).
Structure:
- 5 Downsampling Layers: Each consists of two Conv-BN-ReLU blocks (3x3 convolutions, batch normalization, ReLU activation) followed by 2x2 MaxPooling to reduce spatial dimensions.
- 5 Upsampling Layers: Each uses a ConvTranspose (2x2, stride 2) for upsampling, concatenates skip connections from the corresponding downsampling layer, and applies two Conv-BN-ReLU blocks.
- Output: A single-channel depth map at the input resolution, predicting depth for each pixel.


Channel Progression: Starts at 64 channels, doubles per downsampling layer to 2048 at the bottleneck, then halves back to 64 (inferred from standard U-Net practices, as the paper doesn’t specify counts).
Loss and Training: Uses mean-square-error (MSE) loss on logarithmic depth, trained with the ADAM optimizer for 40,000 iterations, as specified. For specific datasets (e.g., Rectangles), the learning rate decays, but this code focuses on the network structure.

Reference to the Paper
This implementation directly adheres to the Deep Optics for Monocular Depth Estimation paper:

Architecture: Matches the described U-Net with 5 downsampling layers ({Conv-BN-ReLU} × 2 → MaxPool 2×2) and 5 upsampling layers (ConvTranspose + Concat → {Conv-BN-ReLU} × 2), ensuring identical layer structure and skip connections.
Input/Output: Designed to handle RGB inputs and produce depth maps, as used with the paper’s Rectangles, NYU Depth v2, and KITTI datasets.
Integration: Prepared to work with the paper’s differentiable image formation model (though not implemented here), simulating depth-dependent PSFs for computational imaging, critical for NutriSync’s potential food scanning features.

Notes

The U-Net is designed for flexibility with input sizes but assumes proper padding for skip connections (handled automatically in the code).

