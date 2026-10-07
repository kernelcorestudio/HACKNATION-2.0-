"""
LDSR-S2 (Latent Diffusion Super-Resolution) — Basic PyTorch Implementation & Demo.
ESA OpenSR multi-spectral Latent Diffusion baseline (Sentinel-2 10m -> 2.5m, 4x GSD).

Features:
- Native 4-Band Multi-Spectral Support (B02 Blue, B03 Green, B04 Red, B08 NIR)
- Spatial Autoencoder Compression (Encoder & Decoder)
- Conditional Denoising UNet with Timestep Embeddings
- Deterministic DDIM Reverse Sampling Loop
- Demonstrates iterative noise removal and training optimization step
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. First-Stage Spatial Autoencoder (Encoder & Decoder)
# ============================================================================
class SpatialEncoder(nn.Module):
    """Compresses 4-band input into latent space."""
    def __init__(self, in_channels: int = 4, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_dim, kernel_size=3, stride=2, padding=1),     # H/4
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialDecoder(nn.Module):
    """Reconstructs super-resolved 4-band imagery from latent space."""
    def __init__(self, latent_dim: int = 16, out_channels: int = 4):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(latent_dim, 32, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.exit = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.up1(z))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        x = F.relu(self.up4(x))
        return torch.clamp(self.exit(x), 0.0, 1.0)


# ============================================================================
# 2. Conditional Denoising UNet
# ============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    """Encodes scalar timestep t into embedding vector."""
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DenoisingUNet(nn.Module):
    """Predicts added Gaussian noise given latent z_t, timestep t, and condition y."""
    def __init__(self, latent_dim: int = 16, cond_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        # Latent + Conditioning input
        self.entry = nn.Conv2d(latent_dim + cond_dim, hidden_dim, kernel_size=3, padding=1)
        self.res1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.res2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.exit = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, padding=1)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Resize condition to match latent spatial dimensions
        cond_resized = F.interpolate(cond, size=z_t.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([z_t, cond_resized], dim=1)
        h = F.relu(self.entry(x))

        # Add time embedding
        t_emb = self.time_mlp(t)[:, :, None, None]
        h = h + t_emb

        h = F.relu(self.res1(h))
        h = F.relu(self.res2(h))
        return self.exit(h)


# ============================================================================
# 3. Full LDSR-S2 Model & Reverse DDIM Sampling Pipeline
# ============================================================================
class LDSRS2(nn.Module):
    """
    Latent Diffusion Super-Resolution Model for Sentinel-2.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SpatialEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.unet = DenoisingUNet(latent_dim=latent_dim, cond_dim=in_channels, hidden_dim=64)
        self.decoder = SpatialDecoder(latent_dim=latent_dim, out_channels=out_channels)

    def sample_ddim(self, cond_lr: torch.Tensor, num_steps: int = 5) -> torch.Tensor:
        """
        Runs reverse DDIM diffusion trajectory over discrete timesteps.
        """
        B, C, H, W = cond_lr.shape
        # Latent dimensions (H/4, W/4)
        latent_h, latent_w = H // 4, W // 4

        # Start from pure standard Gaussian noise
        z_t = torch.randn(B, self.latent_dim, latent_h, latent_w, device=cond_lr.device)

        # Reverse timesteps (e.g. 5, 4, 3, 2, 1)
        timesteps = torch.linspace(100, 1, num_steps, device=cond_lr.device).long()

        for step in timesteps:
            t_batch = torch.full((B,), step.item(), device=cond_lr.device, dtype=torch.float32)
            # Predict noise epsilon
            noise_pred = self.unet(z_t, t_batch, cond_lr)
            # DDIM reverse step update (simplified deterministic step)
            alpha_step = 1.0 - (step.item() / 100.0) * 0.1
            z_t = z_t - (1.0 - alpha_step) * noise_pred

        # Decode denoised latent into 4x super-resolved 4-band pixels
        return self.decoder(z_t)


# ============================================================================
# 4. Execution Demo (Sampling & Mini-Training Loop)
# ============================================================================
def main():
    print("=" * 65)
    print("[DEMO] LDSR-S2 (ESA OpenSR Latent Diffusion) Demonstration")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Compute Device: {device}")

    # Instantiate Model
    model = LDSRS2(in_channels=4, out_channels=4, latent_dim=16).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters: {total_params:,} (~{total_params / 1e6:.2f} M)")

    # Create Synthetic 4-Band Sentinel-2 LR Tile (B=1, C=4, H=32, W=32)
    # Bands: [Blue, Green, Red, Near-Infrared]
    torch.manual_seed(42)
    dummy_input = torch.rand(1, 4, 32, 32, device=device)
    print(f"\n[1] Input Sentinel-2 LR Tensor Shape : {dummy_input.shape} (4 Bands: RGB+NIR)")

    # 1. Run Iterative DDIM Reverse Sampling Pass (5 steps demo)
    start_time = time.time()
    with torch.no_grad():
        sr_output = model.sample_ddim(dummy_input, num_steps=5)
    latency_ms = (time.time() - start_time) * 1000

    print(f"[2] Output Super-Resolved SR Shape  : {sr_output.shape} (4x Upscaled: 128x128)")
    print(f"[3] Reverse DDIM Sampling Latency   : {latency_ms:.2f} ms (5 steps)")
    print(f"[4] Value Range Check               : Min={sr_output.min():.4f}, Max={sr_output.max():.4f}")

    # 2. Run Denoising Score Matching Training Step
    print("\n[5] Simulating Diffusion Score-Matching Step (Noise Prediction & Backprop)...")
    # Sample random latent, random timestep, and inject true noise
    z_0 = model.encoder(dummy_input)
    t = torch.randint(1, 100, (1,), device=device).float()
    true_noise = torch.randn_like(z_0)
    z_t = z_0 + true_noise * 0.5  # Noisy latent

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    optimizer.zero_grad()

    # Predict noise
    predicted_noise = model.unet(z_t, t, dummy_input)
    loss = F.mse_loss(predicted_noise, true_noise)
    loss.backward()
    optimizer.step()

    print(f"    Diffusion Denoising MSE Loss = {loss.item():.5f}")
    print("    Score-matching gradient backpropagation: SUCCESSFUL")
    print("\n[SUCCESS] LDSR-S2 Code.py executed completely without errors!")
    print("=" * 65)


if __name__ == "__main__":
    main()
