"""
SRM-Net (Residual Channel Attention Network + MC-Dropout) — Basic PyTorch Implementation.
Upscales Sentinel-2 optical imagery 4x with Active Epistemic Uncertainty Estimation.

Features:
- Squeeze-and-Excitation Channel Attention (SE-CA)
- Native 2D Spatial Bernoulli Dropout for Monte-Carlo Sampling
- Cascaded Sub-Pixel Upsampling (PixelShuffle 4x)
- Additive Bicubic Anchor Recombination
- Bayesian Epistemic Uncertainty Heatmap Calculation (T=8 stochastic passes)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Component: Squeeze-and-Excitation Channel Attention (SE-CA)
# ============================================================================
class ChannelAttention(nn.Module):
    """Dynamically recalibrates channel-wise feature responses."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ============================================================================
# 2. Component: Residual Block with Active Spatial Dropout
# ============================================================================
class ResidualBlockWithDropout(nn.Module):
    """Residual Block featuring 2D Spatial Dropout and Channel Attention."""
    def __init__(self, channels: int = 64, dropout_rate: float = 0.20):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ca = ChannelAttention(channels, reduction=16)

    def forward(self, x: torch.Tensor, force_dropout: bool = False) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        
        # Keep dropout stochastically active during test-time Bayesian inference
        if force_dropout:
            out = F.dropout2d(out, p=self.dropout.p, training=True)
        else:
            out = self.dropout(out)
            
        out = self.conv2(out)
        out = self.ca(out)
        return out + residual


# ============================================================================
# 3. SRM-Net Full Architecture
# ============================================================================
class SRMNet(nn.Module):
    """
    Super-Resolution Mapping Network (SRM-Net).
    4x spatial resolution magnifier with Monte-Carlo uncertainty estimation.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 6,  # 6 blocks for lightweight efficient demonstration
        scale_factor: int = 4,
        dropout_rate: float = 0.20
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_rate = dropout_rate

        # Shallow feature extraction head
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Deep residual trunk with embedded spatial dropout
        self.blocks = nn.ModuleList([
            ResidualBlockWithDropout(channels=num_features, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])
        self.trunk_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Sub-pixel upsamplers (PixelShuffle 2x * 2x = 4x)
        self.upconv1 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.upconv2 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)

        # High-frequency reconstruction tail
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, force_dropout: bool = False) -> torch.Tensor:
        # Bicubic baseline anchor
        bicubic = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

        # Feature processing
        feat = self.head(x)
        res = feat
        for block in self.blocks:
            res = block(res, force_dropout=force_dropout)
        res = self.trunk_conv(res)
        feat = feat + res

        # 4x Spatial Expansion
        feat = self.pixel_shuffle1(F.relu(self.upconv1(feat)))
        feat = self.pixel_shuffle2(F.relu(self.upconv2(feat)))

        detail = self.tail(feat)
        sr = bicubic + detail
        return torch.clamp(sr, 0.0, 1.0)


# ============================================================================
# 4. Execution Demo (Forward Pass, MC-Dropout Ensemble, & Mini-Training)
# ============================================================================
def main():
    print("=" * 65)
    print("[DEMO] SRM-Net (Residual Attention + MC-Dropout) Demonstration")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Compute Device: {device}")

    # Instantiate Model
    model = SRMNet(in_channels=3, out_channels=3, num_features=64, num_blocks=6, scale_factor=4).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters: {total_params:,} (~{total_params / 1e6:.2f} M)")

    # Synthetic Low-Resolution Sentinel-2 Tile (Batch=1, C=3, H=32, W=32)
    torch.manual_seed(42)
    dummy_input = torch.rand(1, 3, 32, 32, device=device)
    print(f"\n[1] Input Sentinel-2 LR Tensor Shape : {dummy_input.shape} (10m GSD)")

    # 1. Deterministic Single Forward Pass
    start_time = time.time()
    with torch.no_grad():
        sr_output = model(dummy_input, force_dropout=False)
    single_pass_latency_ms = (time.time() - start_time) * 1000

    print(f"[2] Output Super-Resolved SR Shape  : {sr_output.shape} (2.5m GSD - 4x)")
    print(f"[3] Single-Pass Latency             : {single_pass_latency_ms:.2f} ms")

    # 2. Bayesian Monte-Carlo Epistemic Uncertainty Estimation (T=8 passes)
    print("\n[4] Executing 8-Pass Monte-Carlo Uncertainty Sampling...")
    T = 8
    mc_predictions = []
    start_mc = time.time()
    with torch.no_grad():
        for t in range(T):
            # force_dropout=True activates stochastic spatial dropout at test time
            y_t = model(dummy_input, force_dropout=True)
            mc_predictions.append(y_t)
    mc_latency_ms = (time.time() - start_mc) * 1000

    # Stack predictions: (T, 1, 3, 128, 128)
    mc_tensor = torch.stack(mc_predictions, dim=0)
    mean_sr = torch.mean(mc_tensor, dim=0)
    variance_map = torch.var(mc_tensor, dim=0).mean(dim=1, keepdim=True)  # (1, 1, 128, 128)

    print(f"    Total Time for {T} Stochastic Passes : {mc_latency_ms:.2f} ms (~{mc_latency_ms/1000:.2f} s)")
    print(f"    Mean Super-Resolved Tensor Shape    : {mean_sr.shape}")
    print(f"    Variance Map (Uncertainty Heatmap)  : {variance_map.shape}")
    print(f"    Epistemic Variance Range            : Min={variance_map.min().item():.2e}, Max={variance_map.max().item():.2e}")

    # 3. Mini Training Step (Loss & Backprop)
    print("\n[5] Simulating Mini Training Step (Optimization Demo)...")
    dummy_target = torch.rand(1, 3, 128, 128, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    model.train()
    optimizer.zero_grad()
    prediction = model(dummy_input, force_dropout=True)
    loss = criterion(prediction, dummy_target)
    loss.backward()
    optimizer.step()

    print(f"    L1 Reconstruction Loss = {loss.item():.5f}")
    print("    Gradient backpropagation & weights update: SUCCESSFUL")
    print("\n[SUCCESS] SRM-Net Code.py executed completely without errors!")
    print("=" * 65)


if __name__ == "__main__":
    main()
