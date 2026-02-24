import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper block for double convolution (Conv -> BN -> ReLU) x 2, as per paper
class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2 - Used in both down and up paths."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3 Conv, padding to maintain size
            nn.BatchNorm2d(out_channels),  # Batch Normalization
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Second 3x3 Conv
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Downsampling block: MaxPool -> DoubleConv, as per paper
class Down(nn.Module):
    """Downsampling layer: MaxPool2x2 -> DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2x2 MaxPooling
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Upsampling block: ConvTranspose -> Concat with skip -> DoubleConv, as per paper
class Up(nn.Module):
    """Upsampling layer: ConvTranspose -> Concat skip connection -> DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # Transpose Conv for upsampling
        self.conv = DoubleConv(in_channels, out_channels)  # DoubleConv after concat (in_channels for concat)

    def forward(self, x1, x2):
        # x1 is from previous up layer, x2 is skip connection from down path
        x1 = self.up(x1)
        # Concatenate along channel dimension (assumes same spatial size; in practice, add padding if needed)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Full U-Net model as described: 5 down layers, 5 up layers with skips
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        """U-Net with 5 downsampling and 5 upsampling layers.
        - n_channels: Input channels (e.g., 3 for RGB sensor images).
        - n_classes: Output channels (e.g., 1 for depth map).
        Channel progression: 64 -> 128 -> 256 -> 512 -> 1024 -> 2048 (bottleneck), then reverse.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Input convolution (initial DoubleConv)
        self.inc = DoubleConv(n_channels, 64)

        # 5 Downsampling layers
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048)  # 5th downsampling to bottleneck

        # 5 Upsampling layers with skip connections
        self.up1 = Up(2048, 1024)  # Upsample from bottleneck
        self.up2 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up4 = Up(256, 128)
        self.up5 = Up(128, 64)  # Final up to match input resolution

        # Output convolution (1x1 Conv to map to n_classes)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Downsampling path with feature maps saved for skips
        x1 = self.inc(x)  # Level 1
        x2 = self.down1(x1)  # Level 2
        x3 = self.down2(x2)  # Level 3
        x4 = self.down3(x3)  # Level 4
        x5 = self.down4(x4)  # Level 5
        x6 = self.down5(x5)  # Bottleneck (Level 6)

        # Upsampling path with skip connections
        x = self.up1(x6, x5)  # Up from bottleneck + skip Level 5
        x = self.up2(x, x4)   # + skip Level 4
        x = self.up3(x, x3)   # + skip Level 3
        x = self.up4(x, x2)   # + skip Level 2
        x = self.up5(x, x1)   # + skip Level 1

        # Final output: Depth map
        logits = self.outc(x)
        return logits

# Example usage (for testing the network)
if __name__ == '__main__':
    model = UNet(n_channels=3, n_classes=1)  # RGB input, single-channel depth output
    input_tensor = torch.randn(1, 3, 512, 512)  # Batch size 1, RGB, example 512x512 image
    output = model(input_tensor)
    print("Model output shape:", output.shape)  # Expected: (1, 1, 512, 512) - same resolution as input