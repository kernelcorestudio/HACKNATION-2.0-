"""
Comparison of Restoration Passes on Varanasi AOI:
1. Raw HAT (baseline, unsharpened)
2. SwinIR real_sr PSNR (non-GAN, preferred)
3. Real-ESRGAN with spyndex NDWI water mask (alternative)
4. Real-ESRGAN unmasked (demonstrating the ripple problem)
Evaluates sharpness on urban infrastructure and smoothness on river water.
"""
import sys
import io
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image
import spyndex

from backend.models.sr_engine import SREngine
from backend.models.realesrgan_sharpener import RealESRGANSharpener
from backend.models.architectures.network_swinir import SwinIR

OUT_DIR = ROOT_DIR / "test_outputs" / "water_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\ankus\.gemini\antigravity-ide\brain\0d5e03d1-8366-4dc0-baa9-d84dd8d6a918")

print("=" * 80)
print("VARANASI AOI: COMPARING SWINIR PSNR VS REAL-ESRGAN + NDWI WATER MASK")
print("=" * 80)

# 1. Load Varanasi Sentinel-2 Tile (all bands)
# Find latest saved tile for varanasi_river
tile_files = sorted((ROOT_DIR / "saved_tiles").glob("varanasi_river_*.npy"))
assert len(tile_files) > 0, "No saved Varanasi tiles found!"
tile_path = tile_files[-1]
print(f"Loading Varanasi Sentinel-2 tile: {tile_path.name}")
tile_data = np.load(tile_path) # (128, 128, 10)

# Extract RGB display bands (B04, B03, B02) and NIR (B08)
# In Copernicus client: 0=B04, 1=B03, 2=B02, 3=B08
b04 = tile_data[:, :, 0] # Red
b03 = tile_data[:, :, 1] # Green
b02 = tile_data[:, :, 2] # Blue
b08 = tile_data[:, :, 3] # NIR

lr_rgb = np.stack([b04, b03, b02], axis=-1)
# Normalized display reflectance
lr_display = np.clip(lr_rgb * 2.5, 0.0, 1.0)
Image.fromarray((lr_display * 255).astype(np.uint8)).save(OUT_DIR / "1_varanasi_lr_input.png")

# 2. Compute NDWI with spyndex (McFeeters: (G - N) / (G + N))
print("\n[Step 1] Computing NDWI water mask using spyndex...")
ndwi_lr = spyndex.computeIndex(["NDWI"], params={"G": b03, "N": b08}) # shape (128, 128)
print(f"NDWI stats: min={ndwi_lr.min():.4f}, max={ndwi_lr.max():.4f}, mean={ndwi_lr.mean():.4f}")

# Threshold for water: NDWI > -0.02 (captures turbid river channel accurately)
water_mask_lr = (ndwi_lr > -0.02).astype(np.float32)
# Upsample water mask to 512x512 with smooth feathering
water_mask_sr = cv2.resize(water_mask_lr, (512, 512), interpolation=cv2.INTER_LINEAR)
water_mask_feathered = gaussian_filter(water_mask_sr, sigma=3.0)
water_mask_feathered = np.clip(water_mask_feathered, 0.0, 1.0)
Image.fromarray((water_mask_feathered * 255).astype(np.uint8)).save(OUT_DIR / "2_varanasi_ndwi_water_mask.png")
print(f"Water coverage: {np.sum(water_mask_feathered > 0.5) / water_mask_feathered.size * 100:.2f}% of tile")

# 3. Primary SR Inference: HAT 4x
print("\n[Step 2] Running HAT 4x Super-Resolution...")
sr_engine = SREngine(scale_factor=4, model_name="hat")
t0 = time.time()
hat_sr = sr_engine.predict(lr_rgb, use_tiling=False)
t_hat = time.time() - t0
print(f"HAT 4x complete: shape={hat_sr.shape}, time={t_hat:.2f}s")
hat_display = np.clip(hat_sr * 2.5, 0.0, 1.0)
Image.fromarray((hat_display * 255).astype(np.uint8)).save(OUT_DIR / "3_varanasi_raw_hat.png")

# 4. Standard Real-ESRGAN Unmasked (Demonstrates the ripple artifact on water)
print("\n[Step 3] Running Real-ESRGAN (Unmasked)...")
realesrgan = RealESRGANSharpener()
t0 = time.time()
sr_realesrgan_unmasked = realesrgan.enhance_preview(hat_display, outscale=1)
t_realesrgan = time.time() - t0
print(f"Real-ESRGAN unmasked complete: time={t_realesrgan:.2f}s")
Image.fromarray((sr_realesrgan_unmasked * 255).astype(np.uint8)).save(OUT_DIR / "4_varanasi_realesrgan_unmasked.png")

# 5. Real-ESRGAN with spyndex NDWI Water Mask (Alternative Option)
print("\n[Step 4] Applying Real-ESRGAN with spyndex NDWI Water Masking...")
# Water pixels use raw HAT, land pixels use Real-ESRGAN
w_3ch = np.repeat(water_mask_feathered[:, :, np.newaxis], 3, axis=2)
sr_realesrgan_masked = (1.0 - w_3ch) * sr_realesrgan_unmasked + w_3ch * hat_display
sr_realesrgan_masked = np.clip(sr_realesrgan_masked, 0.0, 1.0)
Image.fromarray((sr_realesrgan_masked * 255).astype(np.uint8)).save(OUT_DIR / "5_varanasi_realesrgan_ndwi_masked.png")
print("Real-ESRGAN + NDWI water mask blend complete.")

# 6. SwinIR Non-GAN PSNR Restoration (Preferred Option)
print("\n[Step 5] Running SwinIR Non-GAN real_sr PSNR Restoration...")
swinir = SwinIR(
    upscale=4, in_chans=3, img_size=64, window_size=8,
    img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
    upsampler='nearest+conv', resi_connection='1conv'
)
ckpt_path = ROOT_DIR / "weights" / "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
key = "params_ema" if "params_ema" in ckpt else "params"
swinir.load_state_dict(ckpt[key] if key in ckpt else ckpt, strict=True)
swinir.eval()

# Input to SwinIR: (1, 3, 128, 128) display RGB
inp_tensor = torch.from_numpy(np.transpose(lr_display, (2, 0, 1))).float().unsqueeze(0)
t0 = time.time()
with torch.no_grad():
    out_swinir = swinir(inp_tensor)
t_swinir = time.time() - t0
swinir_arr = out_swinir.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy()
print(f"SwinIR PSNR complete: shape={swinir_arr.shape}, time={t_swinir:.2f}s")
Image.fromarray((swinir_arr * 255).astype(np.uint8)).save(OUT_DIR / "6_varanasi_swinir_psnr.png")

# Also copy outputs to artifact directory
if ARTIFACT_DIR.exists():
    for f in OUT_DIR.glob("*.png"):
        Image.open(f).save(ARTIFACT_DIR / f.name)

# 7. Zoom Patches for Visual Comparison:
# (a) Urban / Railway / Buildings: center region around [220:340, 200:320]
# (b) Water Corridor: river curve at top-left [10:130, 20:140]
crop_urban = (slice(220, 340), slice(200, 320))
crop_river = (slice(10, 130), slice(30, 150))

def save_crop(img, name, crop):
    c = img[crop[0], crop[1]]
    c_zoom = cv2.resize((c * 255).astype(np.uint8), (240, 240), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(c_zoom).save(OUT_DIR / name)
    if ARTIFACT_DIR.exists():
        Image.fromarray(c_zoom).save(ARTIFACT_DIR / name)

save_crop(hat_display, "crop_urban_hat.png", crop_urban)
save_crop(sr_realesrgan_unmasked, "crop_urban_realesrgan.png", crop_urban)
save_crop(sr_realesrgan_masked, "crop_urban_realesrgan_masked.png", crop_urban)
save_crop(swinir_arr, "crop_urban_swinir.png", crop_urban)

save_crop(hat_display, "crop_river_hat.png", crop_river)
save_crop(sr_realesrgan_unmasked, "crop_river_realesrgan.png", crop_river)
save_crop(sr_realesrgan_masked, "crop_river_realesrgan_masked.png", crop_river)
save_crop(swinir_arr, "crop_river_swinir.png", crop_river)

# 8. Quantitative Sharpness & Noise Metrics:
def laplacian_var(img, crop=None):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if crop:
        gray = gray[crop[0], crop[1]]
    return cv2.Laplacian(gray, cv2.CV_64F).var()

print("\n" + "=" * 80)
print("QUANTITATIVE EVALUATION: URBAN SHARPNESS VS WATER SMOOTHNESS")
print("=" * 80)
print(f"{'Method':<35} | {'Urban Sharpness (LapVar)':<25} | {'Water Roughness (LapVar)':<25}")
print("-" * 90)

methods = [
    ("1. Raw HAT (Baseline)", hat_display),
    ("2. Real-ESRGAN (Unmasked)", sr_realesrgan_unmasked),
    ("3. Real-ESRGAN + NDWI Mask (Alt)", sr_realesrgan_masked),
    ("4. SwinIR PSNR (Preferred)", swinir_arr),
]

for name, img in methods:
    u_var = laplacian_var(img, crop_urban)
    w_var = laplacian_var(img, crop_river)
    print(f"{name:<35} | {u_var:<25.2f} | {w_var:<25.2f}")

print("=" * 80)
print("TEST SCRIPT COMPLETE!")
print("=" * 80)
