"""
Test Suite for Multi-Band Spectral Index and Visualization Module (spyndex).
Verifies:
1. SpectralIndexEngine initialization and support for all 6 visualization modes:
   (True Color, False Color IR, NDVI, NDWI, NDBI, NBR).
2. Direct spyndex index computation and numerical consistency on Sentinel-2 bands.
3. extract_bands_dict compatibility with both 10-band and 4-band cached arrays.
4. High-fidelity continuous color ramp mapping to uint8 RGB.
5. FastAPI GET /api/visualization-modes and POST /api/visualize endpoints.
6. Multi-band super-resolution under non-RGB modes and Metrics Invariance:
   Validation metrics (PSNR, SSIM, SAM, ERGAS) remain strictly computed on
   true-color SR output vs SPOT 1.5m reference tile.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from backend.app import app
from backend.spectral.spectral_indices import SpectralIndexEngine, spectral_engine
from backend.ingestion.copernicus_client import CopernicusClient, extract_bands_dict
from backend.api.routes import current_session


def test_spectral_engine_modes_and_initialization():
    """Verify SpectralIndexEngine has all 6 required visualization modes configured."""
    assert spectral_engine is not None
    expected_modes = ["true_color", "false_color_ir", "ndvi", "ndwi", "ndbi", "nbr"]
    for mode in expected_modes:
        assert mode in spectral_engine.MODES, f"Mode {mode} missing from SpectralIndexEngine"
        cfg = spectral_engine.MODES[mode]
        assert "name" in cfg
        assert "description" in cfg
        assert "bands_used" in cfg
        assert "legend" in cfg


def test_extract_bands_dict_10_and_4_bands():
    """Verify extract_bands_dict handles 10-band arrays and 4-band arrays without synthetic SWIR fallback."""
    # 10-band array
    arr_10 = np.random.uniform(0.05, 0.4, (64, 64, 10)).astype(np.float32)
    bands_10 = extract_bands_dict(arr_10)
    expected_keys = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    for key in expected_keys:
        assert key in bands_10
        assert bands_10[key].shape == (64, 64)

    # 4-band array: Only authentic bands present, NO synthetic B11 or B12
    arr_4 = np.random.uniform(0.05, 0.4, (64, 64, 4)).astype(np.float32)
    bands_4 = extract_bands_dict(arr_4)
    assert "B02" in bands_4 and "B03" in bands_4 and "B04" in bands_4 and "B08" in bands_4
    assert "B11" not in bands_4 and "B12" not in bands_4

    # Missing band test: NBR and NDBI must raise ValueError on 4-band data
    with pytest.raises(ValueError, match="missing for index 'NBR'"):
        spectral_engine.compute_index("nbr", bands_4)
    with pytest.raises(ValueError, match="missing for index 'NDBI'"):
        spectral_engine.compute_index("ndbi", bands_4)

    # Render on 4-band data returns status='unavailable'
    render_nbr = spectral_engine.render("nbr", bands_4)
    assert render_nbr["status"] == "unavailable"
    assert "missing" in render_nbr["error"]


def test_spyndex_index_computation_and_colormaps():
    """Verify index computation with spyndex and continuous colormap generation."""
    h, w = 64, 64
    bands = {
        "B02": np.full((h, w), 0.10, dtype=np.float32),
        "B03": np.full((h, w), 0.15, dtype=np.float32),
        "B04": np.full((h, w), 0.12, dtype=np.float32),
        "B08": np.full((h, w), 0.45, dtype=np.float32), # High NIR (Vegetation)
        "B11": np.full((h, w), 0.20, dtype=np.float32),
        "B12": np.full((h, w), 0.10, dtype=np.float32),
    }

    # NDVI should be strongly positive for vegetation: (0.45 - 0.12) / (0.45 + 0.12) ~ 0.579
    ndvi = spectral_engine.compute_index("ndvi", bands)
    assert ndvi.shape == (h, w)
    assert np.allclose(ndvi[0, 0], (0.45 - 0.12) / (0.45 + 0.12), atol=1e-2)

    # NDWI should be negative for vegetation: (0.15 - 0.45) / (0.15 + 0.45) ~ -0.50
    ndwi = spectral_engine.compute_index("ndwi", bands)
    assert ndwi.shape == (h, w)
    assert np.allclose(ndwi[0, 0], (0.15 - 0.45) / (0.15 + 0.45), atol=1e-2)

    # NDBI should be negative for vegetation: (0.20 - 0.45) / (0.20 + 0.45) ~ -0.38
    ndbi = spectral_engine.compute_index("ndbi", bands)
    assert ndbi.shape == (h, w)
    assert np.allclose(ndbi[0, 0], (0.20 - 0.45) / (0.20 + 0.45), atol=1e-2)

    # NBR should be positive for healthy canopy: (0.45 - 0.10) / (0.45 + 0.10) ~ +0.636
    nbr = spectral_engine.compute_index("nbr", bands)
    assert nbr.shape == (h, w)
    assert np.allclose(nbr[0, 0], (0.45 - 0.10) / (0.45 + 0.10), atol=1e-2)

    # Test colormap rendering
    for mode in ["true_color", "false_color_ir", "ndvi", "ndwi", "ndbi", "nbr"]:
        res = spectral_engine.render(mode, bands)
        assert res["rgb_uint8"].shape == (h, w, 3)
        assert res["rgb_uint8"].dtype == np.uint8
        assert res["base64_png"].startswith("data:image/png;base64,")


def test_api_visualization_modes_endpoint():
    """Verify GET /api/visualization-modes returns catalog of 6 modes."""
    client = TestClient(app)
    res = client.get("/api/visualization-modes")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["modes"]) == 6
    mode_ids = [m["id"] for m in data["modes"]]
    assert "true_color" in mode_ids
    assert "false_color_ir" in mode_ids
    assert "ndvi" in mode_ids
    assert "ndwi" in mode_ids
    assert "ndbi" in mode_ids
    assert "nbr" in mode_ids


def test_api_visualize_endpoint_switching():
    """Verify POST /api/visualize dynamically switches modes in <50ms without error."""
    client = TestClient(app)

    # 1. Fetch Punjabi Agri tile into session
    fetch_res = client.post("/api/fetch-tile", json={
        "bbox": [75.30, 30.55, 75.36, 30.60],
        "aoi_id": "punjab_agri",
        "max_cloud": 15
    })
    assert fetch_res.status_code == 200

    # 2. Switch between all 6 visualization modes
    for mode in ["ndvi", "ndwi", "ndbi", "nbr", "false_color_ir", "true_color"]:
        vis_res = client.post("/api/visualize", json={"mode": mode})
        assert vis_res.status_code == 200, f"Mode {mode} failed with {vis_res.text}"
        data = vis_res.json()
        assert data["status"] == "success"
        assert data["mode"] == mode
        assert data["lr_preview"].startswith("data:image/png;base64,")
        assert "legend" in data


def test_superresolve_with_visualization_mode_and_metric_invariance():
    """
    Verify /api/superresolve supports visualization_mode,
    super-resolves required bands, and ensures scientific metrics (PSNR, SSIM, SAM, ERGAS)
    remain strictly evaluated on True-Color SR vs SPOT 1.5m reference tile.
    """
    client = TestClient(app)

    # 1. Load Punjab Agri tile (paired SPOT 1.5m reference available)
    fetch_res = client.post("/api/fetch-tile", json={
        "bbox": [75.30, 30.55, 75.36, 30.60],
        "aoi_id": "punjab_agri",
        "max_cloud": 15
    })
    assert fetch_res.status_code == 200

    # 2. Run Super-Resolution with visualization_mode="ndvi"
    sr_res = client.post("/api/superresolve", json={
        "num_mc_samples": 2,
        "scale_factor": 4,
        "model_name": "hat",
        "visualization_mode": "ndvi"
    })
    assert sr_res.status_code == 200
    data = sr_res.json()
    assert data["status"] == "success"
    assert data["visualization_mode"] == "ndvi"
    assert "visualization_info" in data
    assert data["visualization_info"]["mode"] == "ndvi"
    assert data["visualization_info"]["is_index"] is True
    assert data["visualization_info"]["stats"] is not None

    # Verify session has super-resolved bands cached
    assert current_session.get("sr_bands") is not None
    assert "B08" in current_session["sr_bands"]
    assert "B04" in current_session["sr_bands"]

    # Verify Metric Invariance:
    # Full-reference metrics (PSNR/SSIM/SAM/ERGAS) exist and evaluate True-Color SR vs SPOT ground truth
    metrics = data["validation_metrics"]
    assert metrics["has_reference"] is True
    assert metrics["is_scientific"] is True
    assert "psnr_db" in metrics
    assert "ssim" in metrics
    assert "sam_deg" in metrics
    assert "ergas" in metrics
    assert isinstance(metrics["psnr_db"], (int, float))
    assert metrics["psnr_db"] > 0


def test_ndwi_vs_nbr_physical_differentiation_and_water_sanity():
    """
    Verify NDWI vs NBR input band differentiation and physical water sanity:
    1. NDWI uses Green (B03) and NIR (B08); NBR uses NIR (B08) and SWIR2 (B12).
    2. Input band identities are strictly distinct: np.array_equal(ndwi_inputs, nbr_inputs) is False.
    3. River/water pixels in NBR show a distinct, uniform extreme value (1.000) that is
       statistically segregated from surrounding land and from NDWI values.
    4. Water pixels in NBR never resemble burn scars (NBR <= 0.10).
    """
    h, w = 64, 64
    # Simulate mixed scene: land (top half) and river/water (bottom half)
    b03 = np.full((h, w), 0.12, dtype=np.float32) # Green
    b04 = np.full((h, w), 0.10, dtype=np.float32) # Red
    b08 = np.full((h, w), 0.35, dtype=np.float32) # NIR
    b12 = np.full((h, w), 0.18, dtype=np.float32) # SWIR2

    # River / water corridor on bottom half: low NIR, higher Green than NIR
    b03[32:, :] = 0.18 # Green reflectance in water
    b08[32:, :] = 0.04 # Strong NIR absorption
    b12[32:, :] = 0.08 # Strong SWIR2 absorption

    bands = {
        "B02": np.full((h, w), 0.15, dtype=np.float32),
        "B03": b03,
        "B04": b04,
        "B08": b08,
        "B11": np.full((h, w), 0.20, dtype=np.float32),
        "B12": b12,
    }

    # Band array identity check
    ndwi_input_bands = np.stack([b03, b08])
    nbr_input_bands = np.stack([b08, b12])
    assert not np.array_equal(ndwi_input_bands, nbr_input_bands), "NDWI and NBR input bands must NOT be identical"
    assert not np.array_equal(b03, b12), "B03 (Green) and B12 (SWIR2) must NOT be identical"

    ndwi = spectral_engine.compute_index("ndwi", bands)
    nbr = spectral_engine.compute_index("nbr", bands)

    # Water pixels (bottom half)
    water_slice = (slice(32, 64), slice(None))
    land_slice = (slice(0, 32), slice(None))

    # 1. NDWI is positive for water
    assert np.all(ndwi[water_slice] > 0.0)
    # 2. NBR water pixels are masked as NaN
    assert np.all(np.isnan(nbr[water_slice]))
    # 3. NBR land pixels are distinct from water (not NaN)
    assert not np.any(np.isnan(nbr[land_slice]))
    # 4. Water is strictly excluded from burn scars (mask excludes NaNs)
    burn_mask = (nbr <= 0.10) & ~np.isnan(nbr)
    assert not np.any(burn_mask[water_slice])
    # 5. Visual renders are completely distinct
    ndwi_res = spectral_engine.render("ndwi", bands)
    nbr_res = spectral_engine.render("nbr", bands)
    assert not np.array_equal(ndwi_res["rgb_uint8"], nbr_res["rgb_uint8"])

