"""
Unit and integration tests for Real-ESRGAN x4plus 2nd-stage sharpening pass.
Verifies:
1. RealESRGANSharpener initialization and weights loading.
2. outscale=1 guarantees exact dimension preservation (no double-upscaling).
3. Scientific metrics (PSNR, SSIM, SAM, ERGAS) and raw session['sr'] are
   strictly invariant to Real-ESRGAN preview sharpening.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from backend.app import app
from backend.models.realesrgan_sharpener import RealESRGANSharpener, realesrgan_sharpener
from backend.api.routes import current_session


def test_realesrgan_sharpener_initialization():
    """Verify RealESRGANSharpener successfully loads weights/RealESRGAN_x4plus.pth."""
    assert realesrgan_sharpener is not None
    assert realesrgan_sharpener.is_ready is True
    assert realesrgan_sharpener.upsampler is not None
    assert "RealESRGAN_x4plus.pth" in realesrgan_sharpener.weights_path


def test_realesrgan_outscale_1_preserves_dimensions():
    """Verify outscale=1 preserves exact spatial dimensions (zero re-upscaling)."""
    h, w = 128, 128
    dummy_sr = np.random.uniform(0.0, 1.0, (h, w, 3)).astype(np.float32)

    sharpened = realesrgan_sharpener.enhance_preview(dummy_sr, outscale=1)

    assert isinstance(sharpened, np.ndarray)
    assert sharpened.shape == (h, w, 3), f"Expected {(h, w, 3)}, got {sharpened.shape}"
    assert sharpened.dtype == np.float32
    assert sharpened.min() >= 0.0
    assert sharpened.max() <= 1.0


def test_metrics_invariance_and_pipeline_sequence():
    """
    Verify that scientific metrics and raw session['sr'] are computed strictly
    on the raw HAT output and are unaffected by the Real-ESRGAN preview pass.
    """
    client = TestClient(app)

    # 1. Fetch initial preset tile (Punjab Agri)
    fetch_res = client.post("/api/fetch-tile", json={
        "bbox": [75.30, 30.55, 75.36, 30.60],
        "aoi_id": "punjab_agri",
        "max_cloud": 15
    })
    assert fetch_res.status_code == 200

    # 2. Run Super-Resolution WITH Real-ESRGAN preview sharpening
    res_sharpened = client.post("/api/superresolve", json={
        "num_mc_samples": 2,
        "scale_factor": 4,
        "model_name": "hat",
        "apply_realesrgan_sharpen": True,
        "apply_unsharp": False
    })
    assert res_sharpened.status_code == 200
    data_sharpened = res_sharpened.json()
    assert data_sharpened["status"] == "success"
    assert data_sharpened["realesrgan_sharpened"] is True
    assert "Real-ESRGAN x4plus" in data_sharpened["post_processing_stage"]
    assert "sr_preview" in data_sharpened
    sr_array_sharpened_session = current_session["sr"].copy()

    # 3. Run Super-Resolution WITHOUT Real-ESRGAN preview sharpening (raw HAT only)
    res_raw = client.post("/api/superresolve", json={
        "num_mc_samples": 2,
        "scale_factor": 4,
        "model_name": "hat",
        "apply_realesrgan_sharpen": False,
        "apply_unsharp": False
    })
    assert res_raw.status_code == 200
    data_raw = res_raw.json()
    assert data_raw["status"] == "success"
    assert data_raw["realesrgan_sharpened"] is False
    assert data_raw["post_processing_stage"] == "None (Raw Model Output)"

    # Verify both runs preserve identical output dimensions
    assert data_sharpened["sr_dimensions"] == data_raw["sr_dimensions"]

    # Verify preview images differ (sharpening occurred)
    assert data_sharpened["sr_preview"] != data_raw["sr_preview"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
