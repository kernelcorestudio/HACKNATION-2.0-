"""
Fetches real Sentinel-2 L2A tiles for the demo AOIs directly from
Copernicus Data Space Ecosystem (CDSE) using OAuth2 Process API.
"""
import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv
import tifffile
import numpy as np
import io

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

CACHE_DIR = BASE_DIR / "cache" / "tiles"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

client_id = os.environ.get("SH_CLIENT_ID")
client_secret = os.environ.get("SH_CLIENT_SECRET")
token_url = os.environ.get("SH_TOKEN_URL", "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token")
process_url = os.environ.get("SH_BASE_URL", "https://sh.dataspace.copernicus.eu") + "/api/v1/process"

print("[CDSE] Authenticating with Copernicus OAuth2...")
token_resp = requests.post(
    token_url,
    data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    },
    timeout=15
)
if token_resp.status_code != 200:
    print(f"[ERROR] Failed to get token: {token_resp.status_code} - {token_resp.text}")
    exit(1)

access_token = token_resp.json()["access_token"]
print("[CDSE] Access token acquired successfully!")

evalscript = """//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    output: { bands: 10, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [
    sample.B04, sample.B03, sample.B02, sample.B08,
    sample.B05, sample.B06, sample.B07, sample.B8A,
    sample.B11, sample.B12
  ];
}
"""

DEMO_AOIS = {
    "punjab_agri": {
        "title": "Punjab Agricultural Parcels (India)",
        "bbox": [75.30, 30.55, 75.36, 30.60],
        "timeRange": {"from": "2024-03-01T00:00:00Z", "to": "2024-05-30T23:59:59Z"},
        "maxCloud": 20
    },
    "delhi_ncr": {
        "title": "Delhi NCR Urban Infrastructure",
        "bbox": [77.15, 28.55, 77.22, 28.61],
        "timeRange": {"from": "2024-03-01T00:00:00Z", "to": "2024-05-30T23:59:59Z"},
        "maxCloud": 20
    },
    "varanasi_river": {
        "title": "Varanasi River Meander & Floodplain",
        "bbox": [82.95, 25.30, 83.00, 25.34],
        "timeRange": {"from": "2024-02-01T00:00:00Z", "to": "2024-05-30T23:59:59Z"},
        "maxCloud": 15
    }
}

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
    "Accept": "image/tiff"
}

for aoi_id, info in DEMO_AOIS.items():
    print(f"\n[CDSE] Fetching live Sentinel-2 L2A tile for '{aoi_id}' ({info['title']})...")
    payload = {
        "input": {
            "bounds": {
                "bbox": info["bbox"],
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": info["timeRange"],
                    "maxCloudCoverage": info["maxCloud"]
                }
            }]
        },
        "output": {
            "width": 128,
            "height": 128,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff"}
            }]
        },
        "evalscript": evalscript
    }

    resp = requests.post(process_url, json=payload, headers=headers, timeout=40)
    if resp.status_code == 200:
        raw_arr = tifffile.imread(io.BytesIO(resp.content))
        # Ensure float32 normalized [0, 1]
        arr = np.clip(raw_arr.astype(np.float32), 0.0, 1.0)
        # Boost contrast for visible display if low reflectance
        print(f"   [SUCCESS] Live Tile received! Shape: {arr.shape}, Range: [{arr.min():.3f}, {arr.max():.3f}]")

        # Save to cache
        cache_file = CACHE_DIR / f"{aoi_id}_real.npy"
        meta_file = CACHE_DIR / f"{aoi_id}_real.json"
        np.save(cache_file, arr)

        meta = {
            "aoi_id": aoi_id,
            "title": info["title"],
            "bbox": info["bbox"],
            "source": "Copernicus Data Space Ecosystem (CDSE) Sentinel-2 L2A",
            "bands": ["B04 (Red)", "B03 (Green)", "B02 (Blue)", "B08 (NIR)"],
            "resolution": "10m GSD",
            "dimensions": f"{arr.shape[1]}x{arr.shape[0]}",
            "time_range": info["timeRange"]
        }
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"   Saved to cache: {cache_file.name}")
    else:
        print(f"   [FAILED] Status {resp.status_code}: {resp.text[:300]}")

print("\n[COMPLETE] Real Copernicus tile fetch finished!")
