"""
Real-ESRGAN (RRDBNet Generator) — Basic PyTorch Implementation & Demo.
Perceptual Super-Resolution baseline (Wang et al., ICCV 2021).

Features:
- Residual Dense Block (RDB) with 5 densely connected convolutions
- Residual-in-Residual Dense Block (RRDB) with 3 nested RDBs and residual scale 0.2
- Dual 2x Nearest-Neighbor Interpolation + Convolution Upsampling (4x total)
- High-Frequency Feature Refinement
- Demonstration of Forward Pass & Optimization
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Component: Residual Dense Block (RDB)
# ============================================================================
class ResidualDenseBlock(nn.Module):
    """5 convolutional layers with dense concatenation."""
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


# ============================================================================
# 2. Component: Residual-in-Residual Dense Block (RRDB)
# ============================================================================
class RRDB(nn.Module):
    """Contains 3 nested Residual Dense Blocks."""
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


# ============================================================================
# 3. Full RRDBNet Architecture
# ============================================================================
class RRDBNet(nn.Module):
    """
    Residual-in-Residual Dense Block Network (RRDBNet).
    Official Generator for ESRGAN and Real-ESRGAN.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_feat: int = 64,
        num_blocks: int = 3,  # Scaled to 3 RRDBs for fast, responsive demonstration
        num_grow_ch: int = 32,
        scale_factor: int = 4
    ):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        
        # Deep dense trunk
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_blocks)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 4x Nearest-neighbor + Convolution upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # High-resolution refinement and final exit
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # 4x upsampling via dual 2x nearest-neighbor interpolation + conv
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return torch.clamp(out, 0.0, 1.0)


# ============================================================================
# 4. Execution Demo (Forward Pass & Mini-Training Loop)
# ============================================================================
def main():
    print("=" * 65)
    print("[DEMO] Real-ESRGAN (RRDBNet Generator) Demonstration")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Compute Device: {device}")

    # Instantiate Model
    model = RRDBNet(in_channels=3, out_channels=3, num_feat=64, num_blocks=3, num_grow_ch=32, scale_factor=4).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters: {total_params:,} (~{total_params / 1e6:.2f} M)")

    # Synthetic Low-Resolution Input Tile (Batch=1, C=3, H=32, W=32)
    torch.manual_seed(42)
    dummy_input = torch.rand(1, 3, 32, 32, device=device)
    print(f"\n[1] Input LR Tensor Shape           : {dummy_input.shape}")

    # 1. Run Forward Inference Pass
    start_time = time.time()
    with torch.no_grad():
        sr_output = model(dummy_input)
    latency_ms = (time.time() - start_time) * 1000

    print(f"[2] Output Super-Resolved SR Shape  : {sr_output.shape} (4x Upscaled)")
    print(f"[3] Inference Latency               : {latency_ms:.2f} ms")
    print(f"[4] Value Range Check               : Min={sr_output.min():.4f}, Max={sr_output.max():.4f}")

    # 2. Run Single-Step Mini Training (Optimization Demo)
    print("\n[5] Simulating Mini Training Step (L1 + Gradient Step)...")
    dummy_target = torch.rand(1, 3, 128, 128, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    model.train()
    optimizer.zero_grad()
    prediction = model(dummy_input)
    loss = criterion(prediction, dummy_target)
    loss.backward()
    optimizer.step()

    print(f"    Reconstruction Loss = {loss.item():.5f}")
    print("    Gradient backpropagation & weights update: SUCCESSFUL")
    print("\n[SUCCESS] Real-ESRGAN Code.py executed completely without errors!")
    print("=" * 65)


if __name__ == "__main__":
    main()
