"""
Unit tests for NETRA Blockchain Provenance Module.
Validates:
1. Fresh tile registration (v1, previousHash = 0x0)
2. Reprocessing version chaining (v2, previousHash = v1.imageHash)
3. Third reprocessing version chaining (v3, previousHash = v2.imageHash)
4. Verification endpoint with authentic hash (MATCH, verified=True)
5. Verification endpoint with tampered/altered hash (TAMPER_DETECTED, verified=False)
6. Chronological history chain traversal
7. Sub-11cm coordinate scaling (* 1e6)
"""
import time
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from backend.blockchain.provenance_manager import (
    ProvenanceManager,
    scale_coordinate,
    unscale_coordinate,
    compute_sha256
)

client = TestClient(app)


def test_coordinate_scaling():
    """Verify sub-11cm integer scaling precision (* 1e6)."""
    lat = 31.147123
    lon = 75.341256

    scaled_lat = scale_coordinate(lat)
    scaled_lon = scale_coordinate(lon)

    assert scaled_lat == 31147123
    assert scaled_lon == 75341256

    # Round trip unscale
    unscaled_lat = unscale_coordinate(scaled_lat)
    unscaled_lon = unscale_coordinate(scaled_lon)

    assert pytest.approx(unscaled_lat, abs=1e-5) == lat
    assert pytest.approx(unscaled_lon, abs=1e-5) == lon


def test_sha256_computation():
    """Verify deterministic SHA-256 hash generation."""
    arr = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    h1 = compute_sha256(arr)
    h2 = compute_sha256(arr)

    assert h1 == h2
    assert h1.startswith("0x")
    assert len(h1) == 66  # 0x + 64 hex characters (bytes32)

    # Altered array produces different hash
    arr_modified = arr.copy()
    arr_modified[0, 0, 0] = 0.501
    h3 = compute_sha256(arr_modified)
    assert h1 != h3


def test_blockchain_status_endpoint():
    """Verify /api/blockchain/status endpoint."""
    response = client.get("/api/blockchain/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chain_id"] == 80002
    assert "Polygon Amoy" in data["network"]


def test_blockchain_version_chaining():
    """
    Test version chain logic:
    v1: previousHash = 0x0
    v2: previousHash = v1.imageHash
    v3: previousHash = v2.imageHash
    """
    pm = ProvenanceManager.get_instance()
    test_tile_id = f"test_tile_{time.time_ns()}"
    bbox = [75.30, 30.55, 75.36, 30.60]

    # Version 1
    tile_data_v1 = np.ones((32, 32, 3), dtype=np.float32) * 0.1
    rec_v1 = pm.register_tile(test_tile_id, tile_data_v1, bbox, model_name="HAT", scale_factor=4)

    assert rec_v1["versionNumber"] == 1
    assert rec_v1["previousHash"] == "0x" + "00" * 32
    assert rec_v1["imageHash"].startswith("0x")

    # Version 2 (Reprocess with different parameters/model)
    tile_data_v2 = np.ones((32, 32, 3), dtype=np.float32) * 0.2
    rec_v2 = pm.register_tile(test_tile_id, tile_data_v2, bbox, model_name="SRM-NET", scale_factor=4)

    assert rec_v2["versionNumber"] == 2
    assert rec_v2["previousHash"] == rec_v1["imageHash"]
    assert rec_v2["imageHash"] != rec_v1["imageHash"]

    # Version 3 (Third pass)
    tile_data_v3 = np.ones((32, 32, 3), dtype=np.float32) * 0.3
    rec_v3 = pm.register_tile(test_tile_id, tile_data_v3, bbox, model_name="CARN", scale_factor=4)

    assert rec_v3["versionNumber"] == 3
    assert rec_v3["previousHash"] == rec_v2["imageHash"]

    # Verify chronological history endpoint
    res_history = client.get(f"/api/blockchain/history/{test_tile_id}")
    assert res_history.status_code == 200
    h_data = res_history.json()
    assert h_data["total_versions"] == 3
    assert len(h_data["history"]) == 3

    assert h_data["history"][0]["version"] == 1
    assert h_data["history"][0]["is_first_version"] is True
    assert h_data["history"][1]["previous_hash"] == h_data["history"][0]["image_hash"]
    assert h_data["history"][2]["previous_hash"] == h_data["history"][1]["image_hash"]


def test_blockchain_verification_tamper_detection():
    """
    Test on-chain verification endpoint:
    - Valid hash returns verified=True, MATCH
    - Altered hash returns verified=False, TAMPER_DETECTED
    """
    pm = ProvenanceManager.get_instance()
    test_tile_id = f"verify_tile_{time.time_ns()}"
    bbox = [75.30, 30.55, 75.36, 30.60]

    tile_data = np.ones((32, 32, 3), dtype=np.float32) * 0.42
    rec = pm.register_tile(test_tile_id, tile_data, bbox)
    correct_hash = rec["imageHash"]

    # Test authentic verification
    res_match = client.get(f"/api/blockchain/verify/{test_tile_id}?hash={correct_hash}")
    assert res_match.status_code == 200
    data_match = res_match.json()["result"]
    assert data_match["verified"] is True
    assert data_match["status"] == "MATCH"
    assert data_match["version_number"] == 1

    # Test tampered hash verification
    tampered_hash = "0x" + "ff" * 32
    res_tamper = client.get(f"/api/blockchain/verify/{test_tile_id}?hash={tampered_hash}")
    assert res_tamper.status_code == 200
    data_tamper = res_tamper.json()["result"]
    assert data_tamper["verified"] is False
    assert data_tamper["status"] == "TAMPER_DETECTED"
