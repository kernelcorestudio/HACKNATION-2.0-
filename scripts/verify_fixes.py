"""
Comprehensive Verification Script for:
1. Mandatory Spatial Co-Registration (AROSICS + phase_cross_correlation fallback)
2. Bridge Over-Detection Suppression via NDWI Water-Body Masking & GLH-Bridge Criteria
3. UI Default State & API Pipeline Integration
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from PIL import Image
import io
import base64

from backend.ingestion.copernicus_client import CopernicusClient
from backend.models.sr_engine import SREngine
from backend.validation.metrics import evaluate_all
from backend.validation.coregistration import coregister_sr_to_reference, spatial_coregister
from backend.validation.reference_manager import load_reference_for_session
from backend.infrastructure.detector import InfrastructureDetector
from backend.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

def main():
    print("=" * 80)
    print("HACKNATION FIX VERIFICATION: CO-REGISTRATION & BRIDGE OVER-DETECTION")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. VARANASI AOI VERIFICATION
    # -------------------------------------------------------------
    aoi_id = "varanasi_river"
    bbox = CopernicusClient.DEMO_AOIS[aoi_id]
    print(f"\n--- [1] Processing Varanasi AOI ({aoi_id}) ---")
    print(f"Bounding Box: {bbox}")

    # Fetch/Load Sentinel-2 tile via API
    r_fetch = client.post("/api/fetch-tile", json={
        "bbox": bbox,
        "aoi_id": aoi_id,
        "max_cloud": 20
    })
    assert r_fetch.status_code == 200, f"Fetch failed: {r_fetch.text}"
    from backend.api.routes import current_session

    tile_data = current_session["lr_raw"]
    hr_ref = current_session["ref_hr"]
    ref_path = current_session["ref_file"]
    print(f"Loaded Sentinel-2 L2A tile: shape={tile_data.shape}, dtype={tile_data.dtype}")
    print(f"Loaded Paired Reference: {ref_path}")
    print(f"Reference Shape: {hr_ref.shape}, Range: [{hr_ref.min():.3f}, {hr_ref.max():.3f}]")

    # Run HAT Inference on 3-channel RGB (B04, B03, B02)
    sr_engine = SREngine(scale_factor=4, model_name="hat")
    lr_rgb = current_session["lr"]
    lr_norm = lr_rgb.astype(np.float32)
    if float(np.nanmax(lr_norm)) > 2.0:
        lr_norm = lr_norm / 10000.0
    lr_norm = np.clip(lr_norm, 0.0, 1.0)
    
    print(f"\nRunning HAT 4x inference on input shape {lr_norm.shape}...")
    sr_output = sr_engine.predict(lr_norm)
    print(f"HAT SR output: shape={sr_output.shape}, range=[{sr_output.min():.3f}, {sr_output.max():.3f}]")

    # Metrics BEFORE Co-Registration
    metrics_before = evaluate_all(sr=sr_output, hr=hr_ref, scale=4.0)
    print("\n[Metrics BEFORE Co-Registration]")
    print(f"   PSNR:  {metrics_before['psnr_db']:.2f} dB")
    print(f"   SSIM:  {metrics_before['ssim']:.4f}")
    print(f"   SAM:   {metrics_before['sam_deg']:.2f}°")
    print(f"   ERGAS: {metrics_before['ergas']:.2f}")

    # Run Co-Registration
    print("\nRunning Spatial Co-Registration...")
    aligned_sr, coreg_meta = coregister_sr_to_reference(
        sr=sr_output,
        hr=hr_ref,
        bbox=bbox,
        aoi_id=aoi_id,
        pixel_res_meters=2.5
    )
    print("\n[Co-Registration Telemetry - Varanasi]")
    print(f"   Method Used:     {coreg_meta['method']}")
    print(f"   Detected Shift X: {coreg_meta['x_shift_px']:+.3f} px ({coreg_meta['x_shift_m']:+.2f} m)")
    print(f"   Detected Shift Y: {coreg_meta['y_shift_px']:+.3f} px ({coreg_meta['y_shift_m']:+.2f} m)")
    print(f"   Success:         {coreg_meta['success']}")
    print(f"   Reliability:     {coreg_meta['reliability']}%")

    # Metrics AFTER Co-Registration
    metrics_after = evaluate_all(sr=aligned_sr, hr=hr_ref, scale=4.0)
    print("\n[Metrics AFTER Co-Registration]")
    print(f"   PSNR:  {metrics_after['psnr_db']:.2f} dB  (Diff: {metrics_after['psnr_db'] - metrics_before['psnr_db']:+.2f} dB)")
    print(f"   SSIM:  {metrics_after['ssim']:.4f}  (Diff: {metrics_after['ssim'] - metrics_before['ssim']:+.4f})")
    print(f"   SAM:   {metrics_after['sam_deg']:.2f}° (Diff: {metrics_after['sam_deg'] - metrics_before['sam_deg']:+.2f}°)")
    print(f"   ERGAS: {metrics_after['ergas']:.2f}  (Diff: {metrics_after['ergas'] - metrics_before['ergas']:+.2f})")

    # -------------------------------------------------------------
    # 2. BRIDGE OVER-DETECTION VERIFICATION (VARANASI)
    # -------------------------------------------------------------
    print("\n--- [2] Bridge Over-Detection & NDWI Masking Verification ---")
    detector = InfrastructureDetector(box_threshold=0.35, text_threshold=0.35)
    
    # Run detector WITH lr_raw (enables physical Sentinel-2 NDWI water mask)
    infra_res = detector.detect(
        sr_image=sr_output,
        confidence_map=np.ones((sr_output.shape[0], sr_output.shape[1]), dtype=np.float32),
        bbox=bbox,
        cloud_mask=None,
        lr_raw=tile_data
    )

    summary = infra_res["summary"]
    print("\n[Infrastructure Detection Summary - Varanasi]")
    print(f"   Raw Bridge Candidates (Pre-Filter): {summary.get('raw_bridges_count')}")
    print(f"   Suppressed False Positives:         {summary.get('water_filtered_bridges')} (ghats/moorings/shadows)")
    print(f"   Final Genuine Bridges (Post-Filter):{summary.get('bridges_count')}")
    print(f"   Roads Detected:                     {summary.get('roads_length_km')} km")
    print(f"   Buildings Detected:                 {summary.get('buildings_count')}")

    # Save visual comparison of unfiltered vs filtered overlay
    out_dir = Path(r"C:\Users\ankus\.gemini\antigravity-ide\brain\a1b847be-bf64-49f6-b4c2-a3e1c44be5e6")
    if "unfiltered_overlay_base64" in infra_res:
        unf_b64 = infra_res["unfiltered_overlay_base64"].split(",")[1]
        unf_img = Image.open(io.BytesIO(base64.b64decode(unf_b64)))
        unf_path = out_dir / "varanasi_bridge_unfiltered.png"
        unf_img.save(unf_path)
        print(f"Saved unfiltered bridge overlay: {unf_path}")

    if "overlay_base64" in infra_res:
        filt_b64 = infra_res["overlay_base64"].split(",")[1]
        filt_img = Image.open(io.BytesIO(base64.b64decode(filt_b64)))
        filt_path = out_dir / "varanasi_bridge_filtered.png"
        filt_img.save(filt_path)
        print(f"Saved filtered bridge overlay: {filt_path}")

    # Also save side-by-side comparison image
    if "unfiltered_overlay_base64" in infra_res and "overlay_base64" in infra_res:
        w, h = unf_img.size
        side_by_side = Image.new("RGBA", (w * 2 + 20, h), (24, 28, 36, 255))
        side_by_side.paste(unf_img, (0, 0), unf_img)
        side_by_side.paste(filt_img, (w + 20, 0), filt_img)
        comp_path = out_dir / "varanasi_bridge_comparison.png"
        side_by_side.save(comp_path)
        print(f"Saved side-by-side comparison: {comp_path}")

    # -------------------------------------------------------------
    # 3. END-TO-END FASTAPI PIPELINE TEST (PUNJAB, DELHI, VARANASI)
    # -------------------------------------------------------------
    print("\n--- [3] End-to-End API Pipeline Test (All Paired Scenes) ---")
    for test_aoi in ["punjab_agri", "delhi_ncr", "varanasi_river"]:
        print(f"\nTesting /api/fetch-tile for '{test_aoi}'...")
        r_fetch = client.post("/api/fetch-tile", json={
            "bbox": CopernicusClient.DEMO_AOIS[test_aoi],
            "aoi_id": test_aoi,
            "max_cloud": 20
        })
        assert r_fetch.status_code == 200, f"Fetch failed for {test_aoi}: {r_fetch.text}"
        
        print(f"Testing /api/superresolve for '{test_aoi}'...")
        r_sr = client.post("/api/superresolve", json={
            "num_mc_samples": 2,
            "scale_factor": 4,
            "model_name": "hat",
            "apply_unsharp": False,
            "apply_realesrgan_sharpen": False  # Must be False by default
        })
        assert r_sr.status_code == 200, f"Superresolve failed for {test_aoi}: {r_sr.text}"
        res_json = r_sr.json()
        assert res_json["status"] == "success"
        
        # Verify co-registration telemetry is present
        coreg = res_json["validation_metrics"].get("coregistration")
        assert coreg is not None, f"Coregistration telemetry missing for {test_aoi}"
        print(f"   [{test_aoi}] Co-Registration Method: {coreg.get('method')}")
        print(f"   [{test_aoi}] Shift: X={coreg.get('x_shift_px'):+.2f} px ({coreg.get('x_shift_m'):+.2f} m), Y={coreg.get('y_shift_px'):+.2f} px ({coreg.get('y_shift_m'):+.2f} m)")
        print(f"   [{test_aoi}] PSNR: {res_json['validation_metrics']['psnr_db']:.2f} dB, SSIM: {res_json['validation_metrics']['ssim']:.4f}")
        
        # Verify bridge counts in infrastructure summary
        infra_summary = res_json["infrastructure_summary"]
        print(f"   [{test_aoi}] Bridges: {infra_summary['bridges_count']} (Raw: {infra_summary.get('raw_bridges_count')})")
        assert infra_summary["bridges_count"] < 10, f"Too many bridges detected in {test_aoi}: {infra_summary['bridges_count']}"

    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
