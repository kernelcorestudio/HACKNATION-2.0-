import numpy as np
import pytest
from pathlib import Path
from backend.validation.reference_manager import get_or_fetch_esri_basemap, PREPARED_HR_REFERENCES
from backend.spectral.spectral_indices import convert_bands_to_display_rgb, apply_reinhard_color_transfer

def test_reinhard_transfer_basic():
    # Source image: yellowish/olive warm bias
    src = np.full((100, 100, 3), [0.7, 0.6, 0.3], dtype=np.float32)
    # Target image: deep green natural basemap
    tgt = np.full((100, 100, 3), [0.3, 0.45, 0.25], dtype=np.float32)
    
    result = apply_reinhard_color_transfer(src, tgt)
    assert result.shape == (100, 100, 3)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    # Mean should be pulled toward target
    assert np.mean(result[..., 1]) > np.mean(result[..., 0])  # Green exceeds red

def test_all_4_aois_esri_basemap():
    test_aois = [
        ('haldwani', [79.48, 29.18, 79.54, 29.24]),
        ('punjab_agri', [75.30, 30.55, 75.36, 30.60]),
        ('delhi_ncr', [77.15, 28.55, 77.22, 28.61]),
        ('varanasi_river', [82.95, 25.30, 83.00, 25.34])
    ]
    for aoi_id, bbox in test_aois:
        esri = get_or_fetch_esri_basemap(bbox=bbox, aoi_id=aoi_id, target_shape=(256, 256))
        assert esri is not None, f"Failed to get Esri basemap for {aoi_id}"
        assert esri.shape == (256, 256, 3)
        assert esri.min() >= 0.0 and esri.max() <= 1.0
        print(f"Verified Esri basemap for {aoi_id}: mean={esri.mean(axis=(0,1))*255}")

if __name__ == "__main__":
    test_reinhard_transfer_basic()
    test_all_4_aois_esri_basemap()
    print("All basic unit checks passed!")
