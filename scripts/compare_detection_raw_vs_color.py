import numpy as np
import cv2
import io
from pathlib import Path
from PIL import Image
from backend.models.sr_engine import SREngine
from backend.infrastructure.detector import InfrastructureDetector
from backend.spectral.spectral_indices import apply_reinhard_color_transfer
from backend.validation.reference_manager import get_or_fetch_esri_basemap

aoi_id = "haldwani"
bbox = [79.48, 29.18, 79.54, 29.24]

tile_path = Path("cache/tiles/ac528e8fba430ac8e6314319522d86f2.npy")
lr_raw = np.load(tile_path)
lr_rgb = lr_raw[..., :3].astype(np.float32)
if float(np.nanmax(lr_rgb)) > 2.0:
    lr_rgb /= 10000.0
lr_rgb = np.clip(lr_rgb, 0.0, 1.0)

sr_engine = SREngine(scale_factor=4, model_name="hat")
raw_sr = sr_engine.predict(lr_rgb, use_tiling=True)
raw_sr = np.clip(raw_sr, 0.0, 1.0)

esri = get_or_fetch_esri_basemap(bbox=bbox, aoi_id=aoi_id, target_shape=(raw_sr.shape[0], raw_sr.shape[1]))
p2, p98 = np.percentile(raw_sr, (2, 98))
stretched = np.clip((raw_sr - p2) / (p98 - p2), 0.0, 1.0) if p98 > p2 else raw_sr
disp_base = np.clip(stretched ** (1.0 / 1.8), 0.0, 1.0)
color_sr = apply_reinhard_color_transfer(disp_base, esri)

detector = InfrastructureDetector()

res_raw = detector.detect(sr_image=raw_sr, bbox=bbox, lr_raw=lr_raw, aoi_id=aoi_id)
res_col = detector.detect(sr_image=color_sr, bbox=bbox, lr_raw=lr_raw, aoi_id=aoi_id)

print("=== INFRASTRUCTURE DETECTION COMPARISON (HALDWANI) ===")
print("1. RAW SR OUTPUT (Physical Surface Reflectance):")
print("   Buildings:", res_raw["summary"]["buildings_count"])
print("   Road Length:", res_raw["summary"]["roads_length_km"], "km (", res_raw["summary"]["road_segments_count"], "segments)")
print("   Avg Confidence:", res_raw["summary"]["avg_detection_confidence"], "%")

print("\n2. COLOR-TRANSFERRED SR OUTPUT (Reinhard LAB Display):")
print("   Buildings:", res_col["summary"]["buildings_count"])
print("   Road Length:", res_col["summary"]["roads_length_km"], "km (", res_col["summary"]["road_segments_count"], "segments)")
print("   Avg Confidence:", res_col["summary"]["avg_detection_confidence"], "%")

# Generate visual comparison of the detections
import base64
raw_overlay_bytes = base64.b64decode(res_raw["overlay_base64"].split(",")[1])
col_overlay_bytes = base64.b64decode(res_col["overlay_base64"].split(",")[1])

raw_overlay = Image.open(io.BytesIO(raw_overlay_bytes))
col_overlay = Image.open(io.BytesIO(col_overlay_bytes))

# Composite on top of their respective backgrounds
raw_bg = Image.fromarray((np.clip(raw_sr, 0.0, 1.0) * 255.0).astype(np.uint8))
col_bg = Image.fromarray((np.clip(color_sr, 0.0, 1.0) * 255.0).astype(np.uint8))

comp_raw = Image.alpha_composite(raw_bg.convert("RGBA"), raw_overlay)
comp_col = Image.alpha_composite(col_bg.convert("RGBA"), col_overlay)

w, h = 512, 512
header_h = 36
collage = Image.new("RGB", (w * 2, h + header_h), (20, 24, 30))
collage.paste(comp_raw.convert("RGB"), (0, header_h))
collage.paste(comp_col.convert("RGB"), (w, header_h))

from PIL import ImageDraw
draw = ImageDraw.Draw(collage)
draw.text((w // 2, 18), f"Detection on RAW SR: {res_raw['summary']['buildings_count']} bldgs, {res_raw['summary']['roads_length_km']} km roads", fill=(240, 180, 60), anchor="mm")
draw.text((w + w // 2, 18), f"Detection on COLOR-TRANSFERRED: {res_col['summary']['buildings_count']} bldgs, {res_col['summary']['roads_length_km']} km roads", fill=(100, 220, 255), anchor="mm")

dest_artifact = Path("C:/Users/ankus/.gemini/antigravity-ide/brain/e7e2c205-c69b-4c22-a56b-cb3b9fd4da4f/infrastructure_detection_comparison.png")
collage.save("cache/infrastructure_detection_comparison.png")
collage.save(dest_artifact)
print(f"\nSaved visual comparison collage to {dest_artifact}!")

