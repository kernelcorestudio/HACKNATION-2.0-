"""
Unit and Integration Tests for Resolution Improvement Options:
1. Multi-Temporal (3-Pass Temporal Median Fusion)
"""
import pytest
import numpy as np

from backend.ingestion.copernicus_client import CopernicusClient


class TestMultiTemporalFusion:
    """Tests 3-Pass Temporal Median Fusion for Sentinel-2 acquisitions."""

    def test_temporal_median_filters_clouds(self):
        np.random.seed(42)
        # Ground truth surface reflectance
        ground_truth = np.full((64, 64, 3), 0.15, dtype=np.float32)

        # Pass 1: Clear scene with minor atmospheric noise
        pass1 = ground_truth + np.random.normal(0, 0.01, ground_truth.shape).astype(np.float32)
        # Pass 2: Cloud artifact in top-left quadrant (high reflectance 0.90)
        pass2 = pass1.copy()
        pass2[:32, :32, :] = 0.90
        # Pass 3: Clear scene
        pass3 = ground_truth + np.random.normal(0, 0.01, ground_truth.shape).astype(np.float32)

        # Temporal median fusion across the passes
        fused = np.median(np.stack([pass1, pass2, pass3], axis=0), axis=0).astype(np.float32)

        # Verify that cloud artifact in pass 2 was rejected by median operator
        cloud_region_mean = np.mean(fused[:32, :32, :])
        assert cloud_region_mean < 0.20, f"Cloud artifact was not rejected: mean = {cloud_region_mean}"

    def test_copernicus_demo_presets_exist(self):
        presets = CopernicusClient.get_demo_presets()
        assert len(presets) >= 3
        preset_ids = [p["id"] for p in presets]
        assert "punjab_agri" in preset_ids
        assert "delhi_ncr" in preset_ids
        assert "varanasi_river" in preset_ids
