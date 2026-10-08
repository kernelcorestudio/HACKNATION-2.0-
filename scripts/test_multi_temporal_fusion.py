"""
Demonstrates and measures the real quality gain of Multi-Temporal Sentinel-2 Fusion
(2-3 temporally close acquisitions) vs Single-Image Super Resolution on the Punjab AOI.
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import requests
import tifffile
import io
import os
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

from backend.models.sr_engine import SREngine
from backend.validation.metrics import evaluate_all

client_id = os.environ.get("SH_CLIENT_ID")
client_secret = os.environ.get("SH_CLIENT_SECRET")
token_url = os.environ.get("SH_TOKEN_URL", "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token")
process_url = os.environ.get("SH_BASE_URL", "https://sh.dataspace.copernicus.eu") + "/api/v1/process"

print("[Multi-Temporal] Authenticating with Copernicus CDSE...")
token_resp = requests.post(token_url, data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret})
access_token = token_resp.json()["access_token"]

evalscript = """//VERSION=3
function setup() {
  return { input: ["B04", "B03", "B02"], output: { bands: 3, sampleType: "FLOAT32" } };
}
function evaluatePixel(sample) { return [sample.B04, sample.B03, sample.B02]; }
"""

# Fetch 3 temporally distinct acquisitions over Punjab (March, April, May 2024)
time_ranges = [
    ("2024-03-01T00:00:00Z", "2024-03-20T23:59:59Z"),
    ("2024-04-01T00:00:00Z", "2024-04-20T23:59:59Z"),
    ("2024-05-01T00:00:00Z", "2024-05-20T23:59:59Z"),
]

headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json", "Accept": "image/tiff"}
bbox = [75.30, 30.55, 75.36, 30.60]

acquisitions = []
for i, tr in enumerate(time_ranges):
    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": tr[0], "to": tr[1]}, "maxCloudCoverage": 20}}]
        },
        "output": {"width": 128, "height": 128, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript
    }
    r = requests.post(process_url, json=payload, headers=headers, timeout=30)
    if r.status_code == 200:
        arr = np.clip(tifffile.imread(io.BytesIO(r.content)).astype(np.float32), 0.0, 1.0)
        acquisitions.append(arr)
        print(f"   Fetched Acquisition {i+1} ({tr[0][:10]}): shape {arr.shape}, range [{arr.min():.3f}, {arr.max():.3f}]")

if len(acquisitions) >= 2:
    # 1. Single Image baseline (Acquisition 1)
    single_lr = acquisitions[0]

    # 2. Multi-temporal Fusion: Temporal median + SNR enhancement
    stacked = np.stack(acquisitions, axis=0) # (3, H, W, 3)
    fused_lr = np.median(stacked, axis=0) # Removes atmospheric transients & noise

    # Load independent high-res ground truth reference
    ref_path = BASE_DIR / "data" / "reference_hr" / "punjab_agri_spot_1.5m.npy"
    ref_hr = np.load(ref_path)

    # Initialize SR Engine
    engine = SREngine(scale_factor=4)

    print("\n--- Running Inference Comparison ---")
    sr_single = engine.predict(single_lr)
    sr_fused = engine.predict(fused_lr)

    metrics_single = evaluate_all(sr_single, ref_hr, scale=4.0)
    metrics_fused = evaluate_all(sr_fused, ref_hr, scale=4.0)

    print("\n================ QUALITY COMPARISON ================")
    print(f"Single-Image SR:   PSNR = {metrics_single['psnr_db']} dB | SSIM = {metrics_single['ssim']} | ERGAS = {metrics_single['ergas']}")
    print(f"Multi-Temporal SR: PSNR = {metrics_fused['psnr_db']} dB | SSIM = {metrics_fused['ssim']} | ERGAS = {metrics_fused['ergas']}")
    psnr_gain = metrics_fused['psnr_db'] - metrics_single['psnr_db']
    ssim_gain = metrics_fused['ssim'] - metrics_single['ssim']
    ergas_reduction = metrics_single['ergas'] - metrics_fused['ergas']
    print(f"-> Quality Gain:   +{psnr_gain:.2f} dB PSNR | +{ssim_gain:.4f} SSIM | -{ergas_reduction:.2f} ERGAS error reduction")
    print("====================================================")

    # Save multi-temporal cached file
    np.save(BASE_DIR / "cache" / "tiles" / "punjab_agri_multitemporal_fused.npy", fused_lr)
