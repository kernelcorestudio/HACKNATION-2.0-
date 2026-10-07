"""
Downloads ESA LDSR-S2 pretrained checkpoint from Hugging Face.
URL: https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/opensr-ldsrs2_v1_0_0.ckpt
"""
import time
from pathlib import Path
import requests

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_FILE = WEIGHTS_DIR / "opensr-ldsrs2_v1_0_0.ckpt"
URL = "https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/opensr-ldsrs2_v1_0_0.ckpt"

def download():
    if TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 1_000_000_000:
        print(f"[LDSR-S2] Checkpoint already exists: {TARGET_FILE} ({TARGET_FILE.stat().st_size / (1024*1024):.1f} MB)")
        return

    temp_file = WEIGHTS_DIR / "opensr-ldsrs2_v1_0_0.ckpt.tmp"
    existing_bytes = temp_file.stat().st_size if temp_file.exists() else 0

    headers = {}
    mode = "ab" if existing_bytes > 0 else "wb"
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"[LDSR-S2] Resuming download from byte {existing_bytes} ({existing_bytes / (1024*1024):.1f} MB)...")
    else:
        print(f"[LDSR-S2] Initiating download from Hugging Face: {URL}...")

    t0 = time.time()
    resp = requests.get(URL, headers=headers, stream=True, timeout=30)
    total_size = int(resp.headers.get("content-length", 0)) + existing_bytes
    print(f"[LDSR-S2] Target total size: {total_size / (1024*1024):.1f} MB")

    downloaded = existing_bytes
    last_log = time.time()
    last_bytes = downloaded

    with open(temp_file, mode) as f:
        for chunk in resp.iter_content(chunk_size=512 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_log >= 5.0:
                speed = (downloaded - last_bytes) / (now - last_log) / (1024 * 1024)
                pct = (downloaded / total_size) * 100.0 if total_size > 0 else 0
                print(f"[LDSR-S2] Progress: {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) - Speed: {speed:.2f} MB/s")
                last_log = now
                last_bytes = downloaded

    temp_file.rename(TARGET_FILE)
    dur = time.time() - t0
    print(f"[LDSR-S2] Download complete in {dur:.1f}s: {TARGET_FILE} ({TARGET_FILE.stat().st_size / (1024*1024):.1f} MB)")

if __name__ == "__main__":
    download()
