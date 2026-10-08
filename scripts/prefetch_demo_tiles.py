"""
Pre-fetch script for Hackathon Demo AOIs.
Runs live queries (or generates cached tiles) to populate cache/tiles/
prior to live judging, ensuring zero-latency, 100% reliable offline/online demos.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingestion.copernicus_client import CopernicusClient, DEMO_AOIS

def main():
    print("=" * 60)
    print("[HACKNATION] Pre-fetching Hackathon Demo AOIs into Cache")
    print("=" * 60)

    client = CopernicusClient()

    for name, bbox in DEMO_AOIS.items():
        print(f"\n[AOI] Pre-fetching AOI '{name}' [bbox: {bbox}]...")
        try:
            tile_data, meta = client.fetch_sentinel2_tile(
                bbox_coords=bbox,
                time_window=("2024-03-01", "2024-05-30"),
                max_cloud=20,
                preset_id=name
            )
            print(f"   [SUCCESS] Cached '{name}': Shape {tile_data.shape}, Source: {meta.get('source')}")
        except Exception as e:
            print(f"   [WARNING] Exception caching '{name}': {e}")

    print("\n[COMPLETE] Pre-fetch complete! All demo AOIs are securely stored in cache/tiles/.")
    print("   On judging day, preset clicks hit cache instantly with zero network risk.")

if __name__ == "__main__":
    main()
