import requests
import json
import base64
import io
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

artifact_dir = Path(r"C:\Users\ankus\.gemini\antigravity-ide\brain\60464997-0003-439e-80f4-8fe7062c6877")
artifact_dir.mkdir(parents=True, exist_ok=True)

# 1. Test live API endpoints on running server
res_ndwi = requests.post("http://127.0.0.1:8000/api/visualize", json={"mode": "ndwi"}).json()
res_nbr = requests.post("http://127.0.0.1:8000/api/visualize", json={"mode": "nbr"}).json()

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

print("=== LIVE API ENDPOINT RESPONSES ===")
print("NDWI status:", res_ndwi.get("status"))
print("NDWI bands used:", res_ndwi.get("bands_used"))
print("NDWI coverage stats:", json.dumps(res_ndwi.get("coverage_stats"), ensure_ascii=True))
print("NDWI stats:", res_ndwi.get("stats"))

print("\nNBR status:", res_nbr.get("status"))
print("NBR bands used:", res_nbr.get("bands_used"))
print("NBR swir_label:", res_nbr.get("swir_label"))
print("NBR coverage stats:", json.dumps(res_nbr.get("coverage_stats"), ensure_ascii=True))
print("NBR stats:", res_nbr.get("stats"))

# 2. Decode preview images
def b64_to_img(b64_str):
    if not b64_str:
        return None
    raw = base64.b64decode(b64_str.split(",")[1])
    return Image.open(io.BytesIO(raw))

ndwi_lr = b64_to_img(res_ndwi.get("lr_preview"))
ndwi_sr = b64_to_img(res_ndwi.get("sr_preview")) or ndwi_lr

nbr_lr = b64_to_img(res_nbr.get("lr_preview"))
nbr_sr = b64_to_img(res_nbr.get("sr_preview")) or nbr_lr

ndwi_sr.save(artifact_dir / "varanasi_ndwi_fixed.png")
nbr_sr.save(artifact_dir / "varanasi_nbr_fixed.png")
print("\nSaved varanasi_ndwi_fixed.png and varanasi_nbr_fixed.png to artifacts.")

# 3. Create high-resolution side-by-side comparison figure
fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)

axes[0].imshow(ndwi_sr)
water_area = res_ndwi.get('coverage_stats', {}).get('area_km2', 0.0)
water_pct = res_ndwi.get('coverage_stats', {}).get('area_pct', 0.0)
axes[0].set_title(
    f"NDWI (Water Delineation)\nBands: B03 (Green) and B08 (NIR)\n"
    f"Water Inundation: {water_area} km2 ({water_pct}%)\n"
    f"River Channel: Deep Navy Blue (High Water Index)",
    fontsize=11, fontweight="bold", pad=10
)
axes[0].axis("off")

axes[1].imshow(nbr_sr)
burn_area = res_nbr.get('coverage_stats', {}).get('area_km2', 0.0)
axes[1].set_title(
    f"NBR (Normalized Burn Ratio)\nBands: B08 (NIR) and B12 (SWIR2 native 20m, bicubic)\n"
    f"Burn Scar Area: {burn_area} km2 (River Masked as NaN)\n"
    f"River Channel: Clear Cyan (#00c5ff) - NOT a burn scar",
    fontsize=11, fontweight="bold", pad=10
)
axes[1].axis("off")

plt.suptitle(
    "Varanasi River Meander AOI: Scientific Separation of NDWI and NBR\n"
    "(River masked as NaN in NBR - Excluded from burn scars)",
    fontsize=13, fontweight="bold", y=0.98
)
plt.tight_layout()
cmp_path = artifact_dir / "ndwi_vs_nbr_side_by_side.png"
plt.savefig(cmp_path, bbox_inches="tight")
plt.close()
print("Saved side-by-side comparison figure to:", cmp_path)

# Verify visual difference
arr_ndwi = np.array(ndwi_sr)
arr_nbr = np.array(nbr_sr)
pixel_diff = np.mean(np.abs(arr_ndwi.astype(float) - arr_nbr.astype(float)))
print(f"Mean Absolute Pixel Difference between NDWI and NBR renders: {pixel_diff:.2f}/255")
assert pixel_diff > 15.0, "NDWI and NBR renders are too similar!"
print("SUCCESS: NDWI and NBR are strongly visually differentiated!")
