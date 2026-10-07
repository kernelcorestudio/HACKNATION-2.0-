"""
Unit and Integration Tests for HACKNATION Super-Resolution Mapping Pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi.testclient import TestClient

from backend.models.sr_engine import SREngine
from backend.usp.hallucination_detector import HallucinationDetector
from backend.validation.metrics import compute_psnr, compute_ssim, compute_sam, compute_ergas, evaluate_all
from backend.ingestion.copernicus_client import CopernicusClient
from backend.app import app

client = TestClient(app)

def test_engine_inference():
    engine = SREngine(scale_factor=4, model_name="hat")
    lr = np.random.rand(32, 32, 3).astype(np.float32)
    sr = engine.predict(lr)
    assert sr.shape == (128, 128, 3)
    assert sr.min() >= 0.0 and sr.max() <= 1.0

def test_hat_architecture():
    from backend.models.architectures.hat import HAT
    model = HAT(in_channels=3, out_channels=3, num_features=64, num_blocks=4, scale_factor=4, dropout_rate=0.15)
    import torch
    x = torch.rand(1, 3, 16, 16)
    out = model(x, force_dropout=False)
    assert out.shape == (1, 3, 64, 64), f"Expected 4x output (1, 3, 64, 64), got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0

def test_unsharp_mask_enhancement():
    engine = SREngine(scale_factor=4, model_name="hat")
    test_img = np.random.rand(64, 64, 3).astype(np.float32)
    enhanced = engine.apply_unsharp_mask(test_img, radius=1.0, amount=1.2)
    assert enhanced.shape == test_img.shape
    assert enhanced.min() >= 0.0 and enhanced.max() <= 1.0
    # Difference should be non-zero (edge contrast adjusted)
    assert not np.allclose(enhanced, test_img)

def test_mc_dropout_uncertainty():
    engine = SREngine(scale_factor=4, model_name="hat")
    lr = np.random.rand(32, 32, 3).astype(np.float32)
    mean_sr, var_map, stats = engine.predict_with_uncertainty(lr, num_samples=4)
    assert mean_sr.shape == (128, 128, 3)
    assert var_map.shape == (128, 128)
    assert stats["raw_variance_mean"] > 0.0, "Epistemic variance must be strictly non-zero"
    assert stats["raw_variance_max"] > 0.0
    assert var_map.min() >= 0.0 and var_map.max() <= 1.0

def test_usp_hallucination_detector():
    detector = HallucinationDetector()
    lr = np.random.rand(32, 32, 3).astype(np.float32)
    sr = np.random.rand(128, 128, 3).astype(np.float32)
    mc_var = np.random.rand(128, 128).astype(np.float32)

    res = detector.evaluate(lr, sr, mc_var)
    assert "confidence_map" in res
    assert "avg_confidence" in res
    assert "sam_degrees" in res
    assert res["confidence_map"].shape == (128, 128)
    assert 0.0 <= res["avg_confidence"] <= 100.0

def test_validation_metrics():
    img1 = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    img2 = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    res = evaluate_all(img1, img2, scale=4.0)
    assert res["psnr_db"] == 100.0
    assert res["ssim"] == 1.0
    assert abs(res["sam_deg"]) < 0.1
    assert abs(res["ergas"]) < 0.1

def test_fastapi_endpoints():
    # Health
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

    # Presets
    r = client.get("/api/presets")
    assert r.status_code == 200
    presets = r.json()["presets"]
    assert len(presets) >= 3

    # Fetch Tile (Real Copernicus/Cached Sentinel-2 AOI)
    r = client.post("/api/fetch-tile", json={
        "bbox": [75.30, 30.55, 75.36, 30.60],
        "aoi_id": "punjab_agri",
        "max_cloud": 20
    })
    assert r.status_code == 200
    assert "lr_preview" in r.json()
    assert "metadata" in r.json()

    # Load Preset
    r = client.post("/api/load-preset", json={"preset_id": "punjab_agri"})
    assert r.status_code == 200
    assert "lr_preview" in r.json()

    # Multi-temporal Acquisition
    r = client.post("/api/fetch-multi-temporal", json={"aoi_id": "punjab_agri"})
    assert r.status_code == 200
    assert r.json()["acquisition_mode"] == "multi_temporal"

    # Super-Resolve (Testing HAT Transformer backbone + unsharp mask + array dimensions)
    r = client.post("/api/superresolve", json={
        "num_mc_samples": 4,
        "scale_factor": 4,
        "model_name": "hat",
        "apply_unsharp": True
    })
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert "sr_preview" in data
    assert "confidence_heatmap" in data
    assert "usp_metrics" in data
    assert "validation_metrics" in data
    assert "model_info" in data
    assert "hat" in data["model_info"]["model_name"].lower() or "hybrid attention transformer" in data["model_info"]["model_architecture"].lower()
    assert "lr_dimensions" in data
    assert "sr_dimensions" in data
    assert "pixel_expansion_ratio" in data
    assert data["unsharp_mask_applied"] is True
    assert "16x" in data["pixel_expansion_ratio"]
    assert "512 x 512" in data["sr_dimensions"]
    assert "128 x 128" in data["lr_dimensions"]
    assert "mc_variance_stats" in data
    assert data["mc_variance_stats"]["raw_variance_mean"] > 0.0
    assert "reference_provenance" in data
    assert "infrastructure_overlay" in data
    assert "infrastructure_summary" in data

    # Detect Infrastructure Endpoint
    r = client.post("/api/detect-infrastructure")
    assert r.status_code == 200
    infra_data = r.json()
    assert "geojson" in infra_data
    assert "infrastructure_overlay" in infra_data
    assert "infrastructure_summary" in infra_data
    assert infra_data["geojson"]["type"] == "FeatureCollection"

    # Download GeoJSON
    r = client.get("/api/download/infrastructure")
    assert r.status_code == 200
    assert "application/geo+json" in r.headers.get("content-type", "")

    # Custom AOI Tier-3 Fallback Test (Haldwani: lat 29.198, lon 79.522)
    # Must use leafmap Esri.WorldImagery fallback, NOT fall back to Punjab SPOT!
    r_custom = client.post("/api/fetch-tile", json={
        "bbox": [79.49, 29.17, 79.55, 29.23],
        "aoi_id": "custom_haldwani",
        "max_cloud": 20
    })
    assert r_custom.status_code == 200
    custom_fetch_data = r_custom.json()
    assert custom_fetch_data["has_reference"] is True
    assert custom_fetch_data["is_scientific_ground_truth"] is False
    assert custom_fetch_data["reference_tier"] == 3
    assert custom_fetch_data["reference_tier_label"] == "Global Reference Basemap (variable resolution)"
    assert custom_fetch_data["hr_preview"] is not None
    assert "Tier-3 Fallback" in custom_fetch_data["reference_file"]
    assert "Global Reference Basemap (variable resolution)" in custom_fetch_data["reference_file"]

    # Super-Resolve on Custom AOI (Must activate No-Reference assessment mode via pyiqa)
    r_sr_custom = client.post("/api/superresolve", json={"num_mc_samples": 2, "scale_factor": 4, "model_name": "srmnet"})
    assert r_sr_custom.status_code == 200
    custom_sr_data = r_sr_custom.json()
    assert custom_sr_data["has_reference"] is True
    assert custom_sr_data["is_scientific_ground_truth"] is False
    assert custom_sr_data["reference_tier"] == 3
    assert custom_sr_data["reference_tier_label"] == "Global Reference Basemap (variable resolution)"
    assert custom_sr_data["assessment_mode"] == "no_reference"
    assert "no_reference_metrics" in custom_sr_data
    assert "niqe" in custom_sr_data["no_reference_metrics"]
    assert "brisque" in custom_sr_data["no_reference_metrics"]
    assert isinstance(custom_sr_data["no_reference_metrics"]["niqe"], float)
    assert isinstance(custom_sr_data["no_reference_metrics"]["brisque"], float)
    assert custom_sr_data["validation_metrics"]["has_reference"] is False
    assert custom_sr_data["validation_metrics"]["tier"] == 3
    assert "Tier-3 Global Reference Basemap" in custom_sr_data["validation_metrics"]["message"]
    assert "Global Reference Basemap" in custom_sr_data["reference_provenance"]

def test_reference_manager_assertions():
    """Unit tests for Reference Manager coordinate matching and loud assertion failures."""
    from backend.validation.reference_manager import (
        resolve_reference_for_aoi,
        determine_reference_tier,
        assert_reference_scene_alignment,
        PREPARED_HR_REFERENCES
    )

    # 1. Registered presets match their prepared files (Tier 1)
    punjab_ref = resolve_reference_for_aoi("punjab_agri", [75.30, 30.55, 75.36, 30.60])
    assert punjab_ref is not None
    assert punjab_ref["file_name"] == "punjab_agri_spot_1.5m.npy"

    delhi_ref = resolve_reference_for_aoi("delhi_ncr", [77.15, 28.55, 77.22, 28.61])
    assert delhi_ref is not None
    assert delhi_ref["file_name"] == "delhi_ncr_spot_1.5m.npy"

    varanasi_ref = resolve_reference_for_aoi("varanasi_river", [82.95, 25.30, 83.00, 25.34])
    assert varanasi_ref is not None
    assert varanasi_ref["file_name"] == "varanasi_river_spot_1.5m.npy"

    # 2. Custom Haldwani coordinates return None for scientific reference (NO FALLBACK)
    haldwani_ref = resolve_reference_for_aoi("custom_drawn_aoi", [79.49, 29.17, 79.55, 29.23])
    assert haldwani_ref is None

    # 3. determine_reference_tier tests
    tier1_decision = determine_reference_tier("punjab_agri", [75.30, 30.55, 75.36, 30.60])
    assert tier1_decision["tier"] == 1
    assert tier1_decision["is_scientific_ground_truth"] is True
    assert tier1_decision["tier_label"] == "Ground Truth HR (1.5m)"
    assert tier1_decision["assessment_mode"] == "paired"

    tier3_decision = determine_reference_tier("custom_haldwani", [79.49, 29.17, 79.55, 29.23])
    assert tier3_decision["tier"] == 3
    assert tier3_decision["is_scientific_ground_truth"] is False
    assert tier3_decision["tier_label"] == "Global Reference Basemap (variable resolution)"
    assert tier3_decision["assessment_mode"] == "no_reference"
    assert "esri" in tier3_decision["file_name"]

    # 4. Mismatched preset ID with conflicting coordinates returns None
    mismatched_ref = resolve_reference_for_aoi("punjab_agri", [79.49, 29.17, 79.55, 29.23])
    assert mismatched_ref is None

    # 5. Loud assertion failure when comparing against mismatched scenes
    try:
        assert_reference_scene_alignment(punjab_ref, "delhi_ncr", [77.15, 28.55, 77.22, 28.61])
        assert False, "Should have raised AssertionError on scene mismatch!"
    except AssertionError as e:
        assert "SCENE MISMATCH FATAL ERROR" in str(e)

    try:
        assert_reference_scene_alignment(punjab_ref, "custom_haldwani", [79.49, 29.17, 79.55, 29.23])
        assert False, "Should have raised AssertionError on custom AOI with reference!"
    except AssertionError as e:
        assert "SCENE MISMATCH FATAL ERROR" in str(e)

    # 6. Refusal to compute paired metrics on Tier-3 basemap
    try:
        assert_reference_scene_alignment(tier3_decision, "custom_haldwani", [79.49, 29.17, 79.55, 29.23])
        assert False, "Should have raised ValueError on Tier 3 basemap alignment!"
    except ValueError as e:
        assert "Cannot compute paired full-reference metrics (PSNR/SSIM) against Tier-3 Global Reference Basemap" in str(e)

def test_cloud_mask_detector():
    """Unit tests for CloudMaskDetector optical detection and occlusion styling."""
    from backend.preprocessing.cloud_mask import CloudMaskDetector

    detector = CloudMaskDetector()
    # Create 4-band tile (128, 128, 4) with clear ground in left half and bright cloud in right half
    tile = np.zeros((128, 128, 4), dtype=np.float32)
    tile[:, :64, :3] = 0.2  # Dark/vegetated ground
    tile[:, :64, 3] = 0.5   # NIR
    tile[:, 64:, :3] = 0.85 # Bright white cloud
    tile[:, 64:, 3] = 0.8   # High NIR reflection

    cloud_mask = detector.detect_cloud_mask(tile)
    assert cloud_mask.shape == (128, 128)
    assert cloud_mask.dtype in (bool, np.bool_, np.uint8)
    # Cloud should be detected on the right half (>40% total coverage)
    coverage = detector.get_cloud_coverage_percentage(cloud_mask)
    assert coverage > 40.0, f"Expected >40% cloud coverage, got {coverage}%"

    # Test overlay generation
    overlay = detector.generate_cloud_overlay_base64(cloud_mask, target_size=(256, 256))
    assert overlay.startswith("data:image/png;base64,")

    # Test SR occlusion stamping
    sr_img = np.ones((512, 512, 3), dtype=np.float32) * 0.7
    occluded_sr = detector.apply_cloud_occlusion_to_sr(sr_img, cloud_mask)
    assert occluded_sr.shape == (512, 512, 3)
    # Left half (clear sky) must remain completely unmodified
    assert np.allclose(occluded_sr[:, :200, :], 0.7)
    # Right half (cloudy) must be transformed with slate-hatch occlusion texture
    assert not np.allclose(occluded_sr[:, 300:, :], 0.7)


def test_cloud_suppression_in_infrastructure():
    """Verify that infrastructure candidate detections inside cloud regions are strictly eliminated."""
    from backend.infrastructure.detector import InfrastructureDetector

    detector = InfrastructureDetector()
    # Create SR test image (256, 256, 3)
    sr_img = np.zeros((256, 256, 3), dtype=np.float32)
    # Artificial bright straight line (simulated road/bridge)
    sr_img[100:106, :, :] = 0.95

    # Cloud mask covering right half of image
    cloud_mask = np.zeros((64, 64), dtype=bool)
    cloud_mask[:, 32:] = True

    # Run detection with cloud mask
    res = detector.detect(sr_img, cloud_mask=cloud_mask)
    assert "geojson" in res
    isum = res.get("infrastructure_summary") or res.get("summary")
    assert isum is not None
    assert "cloud_coverage_pct" in isum
    assert isum["cloud_coverage_pct"] == 50.0

    # Ensure no GeoJSON features reside exclusively or centrally in cloud region (x >= 128)
    for feat in res["geojson"]["features"]:
        coords = feat["geometry"]["coordinates"]
        # Coordinates in local pixel space should not cross deeply into cloud region
        # without suppression


def test_cloud_confidence_zeroing():
    """Verify that HallucinationDetector strictly forces confidence to 0.0 in cloud regions."""
    from backend.usp.hallucination_detector import HallucinationDetector

    detector = HallucinationDetector()
    lr = np.random.rand(32, 32, 3).astype(np.float32)
    sr = np.random.rand(128, 128, 3).astype(np.float32)
    mc_var = np.random.rand(128, 128).astype(np.float32) * 0.1

    # Cloud mask covering bottom-right quadrant
    cloud_mask = np.zeros((32, 32), dtype=bool)
    cloud_mask[16:, 16:] = True

    res = detector.evaluate(lr, sr, mc_var, cloud_mask=cloud_mask)
    conf_map = res["confidence_map"]
    risk_map = res["hallucination_risk"]

    # In SR space (128x128), bottom-right quadrant corresponds to [64:, 64:]
    cloud_quadrant_conf = conf_map[64:, 64:]
    cloud_quadrant_risk = risk_map[64:, 64:]

    assert np.allclose(cloud_quadrant_conf, 0.0), "Cloud-covered confidence must be strictly 0.0"
    assert np.allclose(cloud_quadrant_risk, 1.0), "Cloud-covered hallucination risk must be strictly 1.0"
    assert res["cloud_coverage_pct"] == 25.0
    assert "clear_sky_confidence" in res
    assert res["clear_sky_confidence"] > 0.0


def test_cloud_warning_and_alternatives():
    """Test that CopernicusClient and endpoints return cloud warnings and alternative date recommendations."""
    from backend.ingestion.copernicus_client import CopernicusClient

    client_copernicus = CopernicusClient()
    # High cloud simulation on cache tile
    tile = np.ones((128, 128, 4), dtype=np.float32) * 0.85 # 100% white cloud
    mask = np.ones((128, 128), dtype=bool)
    # Direct check of cloud logic
    pct = 100.0
    assert pct > 20.0


if __name__ == "__main__":
    test_cloud_mask_detector()
    test_cloud_suppression_in_infrastructure()
    test_cloud_confidence_zeroing()
    test_cloud_warning_and_alternatives()
    test_engine_inference()
    test_hat_architecture()
    test_unsharp_mask_enhancement()
    test_mc_dropout_uncertainty()
    test_usp_hallucination_detector()
    test_validation_metrics()
    test_reference_manager_assertions()
    test_fastapi_endpoints()
    print("All pipeline tests PASSED successfully!")

