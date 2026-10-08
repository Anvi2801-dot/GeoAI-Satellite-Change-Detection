import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SiameseUNet(nn.Module):
    """
    Siamese UNet Architecture for Bitemporal Satellite Change Detection.
    Processes Image T1 and Image T2 through a weight-sharing encoder,
    computes feature differences, and decodes to predict pixel-level change.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(SiameseUNet, self).__init__()

        # Shared Encoder Feature Extractor
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # UNet Decoder with Skip Connection Concatenations
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)

        # Final Classifier Head
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward_single(self, x):
        """Pass single image through shared encoder layers."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4

    def forward(self, img1, img2):
        # 1. Feature Extraction via Shared Encoder
        x1_t1, x2_t1, x3_t1, x4_t1 = self.forward_single(img1)
        x1_t2, x2_t2, x3_t2, x4_t2 = self.forward_single(img2)

        # 2. Compute Absolute Feature Differences at each depth
        diff4 = torch.abs(x4_t1 - x4_t2)
        diff3 = torch.abs(x3_t1 - x3_t2)
        diff2 = torch.abs(x2_t1 - x2_t2)
        diff1 = torch.abs(x1_t1 - x1_t2)

        # 3. Decoder Stream with Skip Connections
        x = self.up1(diff4)
        x = torch.cat([x, diff3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, diff2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, diff1], dim=1)
        x = self.conv_up3(x)

        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    # Test model initialization and Apple Silicon MPS / CPU forward pass
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Testing Siamese UNet on device: {device}")

    model = SiameseUNet(in_channels=3, out_channels=1).to(device)

    # Dummy satellite image inputs (Batch=2, Channels=3, Height=256, Width=256)
    img_t1 = torch.randn(2, 3, 256, 256).to(device)
    img_t2 = torch.randn(2, 3, 256, 256).to(device)

    output = model(img_t1, img_t2)
    print(f"[SUCCESS] Network Forward Pass Successful!")
    print(f"[INFO] Input Tensor Shape: {img_t1.shape}")
    print(f"[INFO] Output Mask Shape: {output.shape}")