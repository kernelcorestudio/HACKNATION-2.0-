"""
Deep Residual Attention Super-Resolution Network (SRM-Net) with Monte-Carlo Dropout.
Designed for multi-spectral remote sensing imagery (Sentinel-2 10m -> 2.5m).
Includes test-time MC-Dropout for epistemic uncertainty quantification.
"""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class ChannelAttention(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)

    class ResidualBlockWithDropout(nn.Module):
        def __init__(self, channels, dropout_rate=0.1):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.ca = ChannelAttention(channels)

        def forward(self, x, force_dropout=False):
            residual = x
            out = self.conv1(x)
            out = self.relu(out)
            if force_dropout:
                out = F.dropout2d(out, p=self.dropout.p, training=True)
            else:
                out = self.dropout(out)
            out = self.conv2(out)
            out = self.ca(out)
            out = out + residual
            return out

    class SRMNet(nn.Module):
        """
        Deep Residual Channel Attention Network for Sentinel-2 Super Resolution.
        Upscales 4x (e.g. 10m -> 2.5m).
        Supports Monte Carlo Dropout for epistemic uncertainty mapping.
        """
        def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=8, scale_factor=4, dropout_rate=0.15):
            super().__init__()
            self.scale_factor = scale_factor
            self.dropout_rate = dropout_rate

            # Shallow feature extraction
            self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

            # Deep residual trunk
            self.blocks = nn.ModuleList([
                ResidualBlockWithDropout(num_features, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ])
            self.trunk_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

            # Upsampling (PixelShuffle 4x)
            self.upconv1 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.upconv2 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
            self.pixel_shuffle2 = nn.PixelShuffle(2)

            # Reconstruction tail
            self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
            # Initialize tail with scaled Kaiming normal for high-frequency detail synthesis
            nn.init.kaiming_normal_(self.tail.weight, a=0.1, nonlinearity='leaky_relu')
            self.tail.weight.data *= 0.35
            nn.init.zeros_(self.tail.bias)

        def forward(self, x, force_dropout=False):
            # Bicubic baseline skip connection
            bicubic = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

            feat = self.head(x)
            res = feat
            for block in self.blocks:
                res = block(res, force_dropout=force_dropout)
            res = self.trunk_conv(res)
            feat = feat + res

            feat = self.pixel_shuffle1(F.relu(self.upconv1(feat)))
            feat = self.pixel_shuffle2(F.relu(self.upconv2(feat)))

            residual_detail = self.tail(feat)
            sr = bicubic + residual_detail

            # Radiometric Spectral Preservation Constraint:
            # Strictly preserves the Sentinel-2 true color distribution (B04 Red, B03 Green, B02 Blue)
            # preventing channel imbalances and eliminating magenta/pink color casting
            color_shift = sr.mean(dim=(2, 3), keepdim=True) - bicubic.mean(dim=(2, 3), keepdim=True)
            sr = sr - color_shift
            return torch.clamp(sr, 0.0, 1.0)
else:
    class SRMNet:
        pass
