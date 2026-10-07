"""
CARN (Cascading Residual Network) — Basic PyTorch Implementation & Demo.
ESA EvoLand & WorldStrat baseline for Sentinel-2 10m -> 2.5m (4x) Super-Resolution.

Features:
- Local and Global Cascading Connections
- Local Residual Blocks (LRB)
- 1x1 Convolutional Bottleneck Feature Compression
- Dual Sub-Pixel Upsampling (PixelShuffle 4x)
- Charbonnier Robust Loss Optimization
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Component: Local Residual Block (LRB)
# ============================================================================
class LocalResidualBlock(nn.Module):
    """Standard residual block inside cascading units."""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + res


# ============================================================================
# 2. Component: Cascading Block (CB)
# ============================================================================
class CascadingBlock(nn.Module):
    """Cascading Block aggregating 3 Local Residual Blocks via local skips."""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.b1 = LocalResidualBlock(channels)
        self.b2 = LocalResidualBlock(channels)
        self.b3 = LocalResidualBlock(channels)

        # 1x1 Compression Convolutions for local cascades
        self.c1 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.c2 = nn.Conv2d(channels * 3, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.b1(x)
        b2 = self.b2(b1)
        # Local cascade fusion
        c1 = self.c1(torch.cat([b1, b2], dim=1))
        b3 = self.b3(c1)
        out = self.c2(torch.cat([b1, b2, b3], dim=1))
        return out + x


# ============================================================================
# 3. Full CARN Architecture
# ============================================================================
class CARN(nn.Module):
    """
    Cascading Residual Network (CARN).
    3 Cascading Blocks + Global Cascading Skip + PixelShuffle 4x.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        scale_factor: int = 4
    ):
        super().__init__()
        self.scale_factor = scale_factor

        # Entry feature extractor
        self.entry = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # 3 Cascading Blocks
        self.cb1 = CascadingBlock(num_features)
        self.cb2 = CascadingBlock(num_features)
        self.cb3 = CascadingBlock(num_features)

        # Global Cascading Compression (Concatenates F0, F1, F2, F3)
        self.global_compress = nn.Conv2d(num_features * 4, num_features, kernel_size=1)

        # 4x Sub-Pixel Upsampling (Cascaded 2x PixelShuffle)
        self.upconv1 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.upconv2 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)

        # Reconstruction exit
        self.exit = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.entry(x)
        f1 = self.cb1(f0)
        f2 = self.cb2(f1)
        f3 = self.cb3(f2)

        # Global cascading representation
        f_global = self.global_compress(torch.cat([f0, f1, f2, f3], dim=1))
        feat = f_global + f0

        # Sub-pixel upscaling (2x -> 4x)
        feat = self.pixel_shuffle1(F.relu(self.upconv1(feat)))
        feat = self.pixel_shuffle2(F.relu(self.upconv2(feat)))

        out = self.exit(feat)
        return torch.clamp(out, 0.0, 1.0)


# ============================================================================
# 4. Charbonnier Robust Loss Function
# ============================================================================
class CharbonnierLoss(nn.Module):
    """L1-like robust penalty used during WorldStrat pretraining."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)


# ============================================================================
# 5. Execution Demo (Forward Pass & Mini-Training Loop)
# ============================================================================
def main():
    print("=" * 65)
    print("[DEMO] CARN (Cascading Residual Network) Demonstration")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Compute Device: {device}")

    # Instantiate Model
    model = CARN(in_channels=3, out_channels=3, num_features=64, scale_factor=4).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters: {total_params:,} (~{total_params / 1e6:.2f} M)")

    # Create Synthetic Low-Resolution Satellite Tile (Batch=1, C=3, H=32, W=32)
    # Simulates Sentinel-2 10m optical reflectance [0.0, 1.0]
    torch.manual_seed(42)
    dummy_input = torch.rand(1, 3, 32, 32, device=device)
    print(f"\n[1] Input Sentinel-2 LR Tensor Shape : {dummy_input.shape} (10m GSD)")

    # 1. Run Forward Inference Pass
    start_time = time.time()
    with torch.no_grad():
        sr_output = model(dummy_input)
    latency_ms = (time.time() - start_time) * 1000

    print(f"[2] Output Super-Resolved SR Shape  : {sr_output.shape} (2.5m GSD - 4x)")
    print(f"[3] Inference Latency               : {latency_ms:.2f} ms")
    print(f"[4] Value Range Check               : Min={sr_output.min():.4f}, Max={sr_output.max():.4f}")

    # 2. Run Single-Step Mini Training (Optimization Demo with Charbonnier Loss)
    print("\n[5] Simulating Mini Training Step (Charbonnier Loss & Backprop)...")
    dummy_target = torch.rand(1, 3, 128, 128, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = CharbonnierLoss(eps=1e-3)

    model.train()
    optimizer.zero_grad()
    prediction = model(dummy_input)
    loss = criterion(prediction, dummy_target)
    loss.backward()
    optimizer.step()

    print(f"    Charbonnier Reconstruction Loss = {loss.item():.5f}")
    print("    Gradient backpropagation & weights update: SUCCESSFUL")
    print("\n[SUCCESS] CARN Code.py executed completely without errors!")
    print("=" * 65)


if __name__ == "__main__":
    main()
