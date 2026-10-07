"""
SwinIR (Shifted Window Transformer) — Basic PyTorch Implementation & Demo.
Vision Transformer baseline for Sentinel-2 10m -> 2.5m (4x) Super-Resolution.

Features:
- Window Multi-Head Self-Attention (W-MSA)
- Shifted Window Multi-Head Self-Attention (SW-MSA)
- Residual Swin Transformer Block (RSTB)
- Sub-Pixel PixelShuffle 4x Upsampling
- Demonstrates forward pass and mini training optimization
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Swin Transformer Layer (W-MSA & SW-MSA)
# ============================================================================
class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Layer supporting regular or shifted window attention.
    """
    def __init__(self, dim: int = 60, num_heads: int = 6, window_size: int = 8, shift_size: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # Cyclic shift if shift_size > 0 (SW-MSA)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition: (B, H, W, C) -> (num_windows*B, window_size*window_size, C)
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            shifted_x = F.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = shifted_x.shape

        x_windows = shifted_x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        N, L, _ = windows.shape

        # Multi-Head Attention
        qkv = self.qkv(windows).reshape(N, L, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(N, L, self.dim)
        out = self.proj(out)

        # Window unpartition
        out = out.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :]

        # Reverse cyclic shift
        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # FFN with residual connections
        x = shortcut + out
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# 2. Residual Swin Transformer Block (RSTB)
# ============================================================================
class RSTB(nn.Module):
    """Residual Swin Transformer Block containing multiple alternating STLs."""
    def __init__(self, dim: int = 60, num_heads: int = 6, window_size: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0),
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        # Convert BCHW to BHWC for transformer layers
        x = x.permute(0, 2, 3, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 3, 1, 2)
        return res + self.conv(x)


# ============================================================================
# 3. Full SwinIR Architecture
# ============================================================================
class SwinIR(nn.Module):
    """
    SwinIR Architecture for 4x Super-Resolution.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 60,  # Scaled to 60 for fast CPU demonstration
        num_blocks: int = 3,
        num_heads: int = 6,
        window_size: int = 8,
        scale_factor: int = 4
    ):
        super().__init__()
        self.scale_factor = scale_factor

        # Shallow feature extraction
        self.shallow = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # Deep feature extraction trunk
        self.rstb_blocks = nn.ModuleList([
            RSTB(dim=embed_dim, num_heads=num_heads, window_size=window_size)
            for _ in range(num_blocks)
        ])
        self.trunk_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # 4x PixelShuffle upsampling
        self.upconv1 = nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.upconv2 = nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)

        # Reconstruction exit
        self.exit = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_shallow = self.shallow(x)
        feat = f_shallow
        for rstb in self.rstb_blocks:
            feat = rstb(feat)
        feat = f_shallow + self.trunk_conv(feat)

        # 4x Upscaling
        feat = self.pixel_shuffle1(F.leaky_relu(self.upconv1(feat), 0.1))
        feat = self.pixel_shuffle2(F.leaky_relu(self.upconv2(feat), 0.1))

        out = self.exit(feat)
        return torch.clamp(out, 0.0, 1.0)


# ============================================================================
# 4. Execution Demo (Forward Pass & Mini-Training Loop)
# ============================================================================
def main():
    print("=" * 65)
    print("[DEMO] SwinIR (Shifted Window Transformer) Demonstration")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Compute Device: {device}")

    # Instantiate Model
    model = SwinIR(in_channels=3, out_channels=3, embed_dim=60, num_blocks=3, scale_factor=4).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters: {total_params:,} (~{total_params / 1e6:.2f} M)")

    # Create Synthetic Low-Resolution Satellite Tile (Batch=1, C=3, H=32, W=32)
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

    # 2. Run Single-Step Mini Training (Optimization Demo)
    print("\n[5] Simulating Mini Training Step (L1 Loss & Backprop)...")
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
    print("\n[SUCCESS] SwinIR Code.py executed completely without errors!")
    print("=" * 65)


if __name__ == "__main__":
    main()
