"""
Tests for ESA OpenSR LDSR-S2 Latent Diffusion Super-Resolution Architecture.
"""
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from backend.app import app
from backend.models.architectures.ldsr_s2 import LDSRS2
from backend.models.sr_engine import SREngine

client = TestClient(app)

def test_ldsr_architecture_initialization():
    """Verifies that LDSRS2 architecture initializes with official config."""
    model = LDSRS2(scale_factor=4, sampling_steps=5, device="cpu")
    assert model is not None
    assert model.scale_factor == 4
    assert model.sampling_steps == 5
    assert hasattr(model, "model")

def test_ldsr_forward_dimensions():
    """Verifies that LDSRS2 forward pass preserves 4x spatial scaling and valid radiometric bounds."""
    model = LDSRS2(scale_factor=4, sampling_steps=2, device="cpu")
    # 3-channel input (RGB: B04, B03, B02) - min size 64x64 required for reflection padding
    x3 = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    out3 = model(x3)
    assert out3.shape == (1, 3, 256, 256)
    assert out3.min().item() >= 0.0
    assert out3.max().item() <= 1.0

    # 4-channel input (RGB-NIR: B04, B03, B02, B08)
    x4 = torch.rand(1, 4, 64, 64, dtype=torch.float32)
    out4 = model(x4)
    assert out4.shape == (1, 4, 256, 256)
    assert out4.min().item() >= 0.0
    assert out4.max().item() <= 1.0

def test_sr_engine_switch_to_ldsr():
    """Verifies that SREngine dynamically switches to LDSR-S2."""
    engine = SREngine(scale_factor=4, model_name="hat")
    assert engine.model_name == "hat"

    engine.switch_model(model_name="ldsr_s2", scale_factor=4)
    assert engine.model_name == "ldsr_s2"
    assert engine.checkpoint_meta.get("model_name") == "LDSR-S2"
    assert "Latent Diffusion" in engine.checkpoint_meta.get("training_dataset", "")

def test_api_superresolve_ldsr_s2():
    """Verifies end-to-end API inference and metric computation with LDSR-S2."""
    # 1. Fetch initial preset tile
    r_fetch = client.post("/api/fetch-tile", json={
        "bbox": [82.98, 25.28, 83.05, 25.35],
        "aoi_id": "varanasi_river",
        "max_cloud": 15
    })
    assert r_fetch.status_code == 200

    # 2. Run Super-Resolution using ldsr_s2
    r_sr = client.post("/api/superresolve", json={
        "num_mc_samples": 1,
        "scale_factor": 4,
        "model_name": "ldsr_s2",
        "apply_realesrgan_sharpen": False,
        "apply_unsharp": False
    })
    assert r_sr.status_code == 200
    data = r_sr.json()
    assert data["status"] == "success"
    assert "512 x 512" in data["sr_dimensions"]
    assert "sr_preview" in data
    assert "confidence_heatmap" in data
    assert "usp_metrics" in data
    assert data["usp_metrics"]["avg_confidence"] > 0.0
    assert data["model_info"]["model_architecture"] in ["LDSRS2", "LDSR_S2"]
