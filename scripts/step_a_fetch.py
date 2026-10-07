import sys
import base64
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

print("=== STEP A: Sentinel-2 Fetch Only ===")
aoi_id = "punjab_agri"
bbox = [75.30, 30.55, 75.36, 30.60]

response = client.post("/api/fetch-tile", json={
    "bbox": bbox,
    "aoi_id": aoi_id,
    "max_cloud": 15
})

assert response.status_code == 200, f"Fetch failed: {response.text}"
data = response.json()

b64_str = data["lr_preview"].split(",")[1]
img_bytes = base64.b64decode(b64_str)
img = Image.open(io.BytesIO(img_bytes))

out_dir = Path(__file__).resolve().parent.parent / "test_outputs"
out_dir.mkdir(exist_ok=True)
img_path = out_dir / "step_a_sentinel2_raw_input.png"
img.save(img_path)

arr = np.array(img)
metadata = data.get("metadata", {})
print(f"Status: SUCCESS ({response.status_code})")
print(f"AOI ID: {aoi_id}")
print(f"Bounding Box: {bbox}")
print(f"Dimensions: {metadata.get('dimensions')}")
print(f"Source: {metadata.get('source')}")
print(f"Input Resolution: {metadata.get('input_resolution')}")
print(f"Image Array Shape: {arr.shape}")
print(f"RGB Pixel Range: min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}, std={arr.std():.2f}")
print(f"Saved Image Path: {img_path}")
