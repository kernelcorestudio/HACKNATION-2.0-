"""
Hybrid Attention Transformer (HAT) — Basic PyTorch Implementation & Demo.
Upscales multi-spectral satellite imagery 4x (e.g., Sentinel-2 10m -> 2.5m GSD).

Features:
- Window-based Multi-Head Self-Attention (W-MSA)
- Channel Attention Block (CAB)
- Cascaded Sub-Pixel Upsampling (PixelShuffle 4x)
- Additive Bicubic Anchor Recombination
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Component: Channel Attention Block (CAB)
# ============================================================================
class ChannelAttentionBlock(nn.Module):
    """Models global inter-band spectral relationships."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


# ============================================================================
# 2. Component: Window-based Multi-Head Self-Attention (W-MSA)
# ============================================================================
class WindowMultiHeadAttention(nn.Module):
    """Local window attention for capturing sharp structural edge boundaries."""
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4, dropout_rate: float = 0.15):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Pad to multiple of window_size if necessary
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        _, _, Hp, Wp = x.shape

        # Window partition: (B, C, Hp, Wp) -> (num_windows * B, window_size * window_size, C)
        x_perm = x.permute(0, 2, 3, 1)  # (B, Hp, Wp, C)
        x_win = x_perm.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        windows = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        N, L, _ = windows.shape

        # Compute Q, K, V
        qkv = self.qkv(windows).reshape(N, L, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(N, L, self.dim)
        out = self.proj(out)

        # Unpartition windows back to image tensor
        out = out.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C).permute(0, 3, 1, 2)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        return out


# ============================================================================
# 3. Hybrid Attention Block (HAB)
# ============================================================================
class HybridAttentionBlock(nn.Module):
    """Combines W-MSA (Spatial Attention) and CAB (Spectral Channel Attention)."""
    def __init__(self, dim: int = 64, window_size: int = 8, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.w_msa = WindowMultiHeadAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.BatchNorm2d(dim)
        self.cab = ChannelAttentionBlock(dim, reduction=16)
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial Attention branch
        res1 = x
        x_norm = self.norm1(x)
        x = res1 + self.w_msa(x_norm)

        # Spectral Channel Attention branch
        res2 = x
        x = self.cab(self.norm2(x))
        x = self.conv_ffn(x)
        return x + res2


# ============================================================================
# 4. Full HAT Super-Resolution Network
# ============================================================================
class HAT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 4,  # Scaled to 4 blocks for fast demonstration
        scale_factor: int = 4
    ):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Shallow feature extraction
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Deep Hybrid Attention Trunk
        self.blocks = nn.ModuleList([
            HybridAttentionBlock(dim=num_features, window_size=8, num_heads=4)
            for _ in range(num_blocks)
        ])
        self.trunk_fusion = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # 4x Upsampling Module (Cascaded 2x PixelShuffle)
        self.upconv1 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.upconv2 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)

        # Synthesis Tail
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bicubic anchor skip connection
        bicubic_base = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

        # Deep feature extraction
        f_shallow = self.head(x)
        res = f_shallow
        for block in self.blocks:
            res = block(res)
        res = self.trunk_fusion(res)
        feat = f_shallow + res

        # Sub-pixel upscaling (2x -> 4x)
        feat = self.pixel_shuffle1(F.leaky_relu(self.upconv1(feat), 0.1))
        feat = self.pixel_shuffle2(F.leaky_relu(self.upconv2(feat), 0.1))

        # Synthesize high-frequency details
        detail = self.tail(feat)

        # Final additive recombination
        sr = bicubic_base + detail
        return torch.clamp(sr, 0.0, 1.0)


# ============================================================================
# 5. Execution Demo (Forward Pass & Mini-Training Loop)
# ============================================================================
def main():
    print("=" * 65)
    print("[DEMO] HAT (Hybrid Attention Transformer) Demonstration")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Compute Device: {device}")

    # Instantiate Model
    model = HAT(in_channels=3, out_channels=3, num_features=64, num_blocks=4, scale_factor=4).to(device)
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

    # 2. Run Single-Step Mini Training (Optimization Demo)
    print("\n[5] Simulating Mini Training Step (Loss & Backprop)...")
    # Ground truth HR simulation (1, 3, 128, 128)
    dummy_target = torch.rand(1, 3, 128, 128, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.L1Loss()

    model.train()
    optimizer.zero_grad()
    prediction = model(dummy_input)
    loss = criterion(prediction, dummy_target)
    loss.backward()
    optimizer.step()

    print(f"    L1 Reconstruction Loss = {loss.item():.5f}")
    print("    Gradient backpropagation & weights update: SUCCESSFUL")
    print("\n[SUCCESS] HAT Code.py executed completely without errors!")
    print("=" * 65)


if __name__ == "__main__":
    main()
