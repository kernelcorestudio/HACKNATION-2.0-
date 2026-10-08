"""
Verification Script for Punjab AOI Co-Registration, Caching, and Metric Tracing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.app import app

def main():
    print("=" * 80)
    print("STEP 1: TRACING FUNCTION CALL CHAIN")
    print("=" * 80)
    print("1. Frontend: selectPreset('punjab_agri') in frontend/app.js (line 664)")
    print("2. Frontend: fetchTileForBbox([75.30, 30.55, 75.36, 30.60], 'punjab_agri') (line 688)")
    print("3. Backend:  POST /api/fetch-tile -> fetch_tile() in backend/api/routes.py (line 199)")
    print("4. Backend:  load_reference_for_session() in backend/validation/reference_manager.py (line 330)")
    print("             Loads: data/reference_hr/punjab_agri_spot_1.5m.npy into current_session['ref_hr']")
    print("5. Frontend: runSuperResolution() in frontend/app.js (line 782)")
    print("6. Backend:  POST /api/superresolve -> superresolve() in backend/api/routes.py (line 512)")
    print("7. Backend:  sr_engine.predict_with_uncertainty() in backend/models/sr_engine.py (line 130)")
    print("             PyTorch HAT forward pass produces mean_sr (512x512x3)")
    print("8. Backend:  coregister_sr_to_reference() in backend/validation/coregistration.py (line 280)")
    print("9. Backend:  [AROSICS] cr = COREG(im_ref, im_tgt, ...) in backend/validation/coregistration.py (line 146)")
    print("10. Backend: [AROSICS] cr.calculate_spatial_shifts() in backend/validation/coregistration.py (line 153)")
    print("11. Backend: [AROSICS] cr.correct_shifts() -> writes aligned GeoTIFF (line 163)")
    print("12. Backend: evaluate_all(sr=aligned_sr, hr=ref_hr) in backend/validation/metrics.py (line 90)")
    print("13. Backend: compute_psnr() & compute_ssim() in backend/validation/metrics.py (lines 23, 33)")
    print("14. Backend: JSON response returned with validation_metrics, freshness timestamp, & telemetry")
    print("15. Frontend: Updates DOM elements (#metric-psnr, #metric-ssim, #coreg-shift-status, #metric-computed-at)")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("STEP 2: RUNNING PUNJAB AOI (FRESH INFERENCE & COREGISTRATION)")
    print("=" * 80)

    client = TestClient(app)

    # 1. Fetch Punjab tile
    r_fetch = client.post("/api/fetch-tile", json={
        "bbox": [75.30, 30.55, 75.36, 30.60],
        "aoi_id": "punjab_agri",
        "max_cloud": 15
    })
    assert r_fetch.status_code == 200, f"Fetch failed: {r_fetch.text}"
    fetch_data = r_fetch.json()
    print(f"[Fetch Tile] AOI: 'punjab_agri' | Reference File: {fetch_data.get('reference_file')}")
    print(f"[Fetch Tile] Tier: {fetch_data.get('reference_tier')} | Scientific Ground Truth: {fetch_data.get('has_reference')}")

    # 2. Run Super-Resolution
    r_sr = client.post("/api/superresolve", json={
        "num_mc_samples": 1,
        "scale_factor": 4,
        "model_name": "hat",
        "apply_realesrgan_sharpen": False,
        "apply_unsharp": False
    })
    assert r_sr.status_code == 200, f"SR failed: {r_sr.text}"
    sr_data = r_sr.json()

    metrics = sr_data["validation_metrics"]
    coreg = metrics["coregistration"]
    raw = metrics["raw_unaligned"]
    delta = metrics["delta_from_raw"]

    print("\n" + "=" * 80)
    print("STEP 3: LITERAL METRICS & TELEMETRY RESULT")
    print("=" * 80)
    print(f"Timestamp:              {metrics.get('computed_at')}")
    print(f"Execution ID:           {metrics.get('execution_id')}")
    print(f"AROSICS Method:         {coreg.get('method')}")
    print(f"Detected Shift (Pixels): X = {coreg.get('x_shift_px'):+.4f} px | Y = {coreg.get('y_shift_px'):+.4f} px")
    print(f"Detected Shift (Meters): X = {coreg.get('x_shift_m'):+.2f} m  | Y = {coreg.get('y_shift_m'):+.2f} m")
    print(f"AROSICS Reliability:    {coreg.get('reliability')}%")
    print("-" * 80)
    print(f"Raw (Unaligned) PSNR:   {raw.get('psnr_db'):.2f} dB")
    print(f"Coregistered PSNR:      {metrics.get('psnr_db'):.2f} dB  (Delta: {delta.get('psnr_diff_db'):+.2f} dB)")
    print(f"Raw (Unaligned) SSIM:   {raw.get('ssim'):.4f}")
    print(f"Coregistered SSIM:      {metrics.get('ssim'):.4f}  (Delta: {delta.get('ssim_diff'):+.4f} -> +{(delta.get('ssim_diff') / raw.get('ssim')) * 100:.1f}%)")
    print(f"Raw (Unaligned) SAM:    {raw.get('sam_deg'):.2f} deg")
    print(f"Coregistered SAM:       {metrics.get('sam_deg'):.2f} deg  (Delta: {delta.get('sam_diff_deg'):+.2f} deg)")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("STEP 4: RADIOMETRIC DYNAMIC RANGE INVESTIGATION (WHY PSNR IS ~9.58 dB)")
    print("=" * 80)
    from backend.api.routes import current_session
    import numpy as np
    from backend.validation.metrics import compute_psnr, compute_ssim, compute_sam
    sr = current_session["sr"]
    ref = current_session["ref_hr"]
    print(f"SR (Sentinel-2 BOA):  Range = [{sr.min():.4f}, {sr.max():.4f}] | Channel Means: R={sr[:,:,0].mean():.4f}, G={sr[:,:,1].mean():.4f}, B={sr[:,:,2].mean():.4f}")
    print(f"Ref (SPOT Display):   Range = [{ref.min():.4f}, {ref.max():.4f}] | Channel Means: R={ref[:,:,0].mean():.4f}, G={ref[:,:,1].mean():.4f}, B={ref[:,:,2].mean():.4f}")
    diff_mean = ref.mean() - sr.mean()
    print(f"Radiometric Offset:   Delta_mu = {diff_mean:.4f}")
    print(f"Mathematical Max PSNR: 10 * log10(1 / ({diff_mean:.4f})^2) = {10 * np.log10(1 / (diff_mean**2)):.2f} dB")
    
    # Scale matching test (matching mean and standard deviation)
    sr_scaled = np.zeros_like(sr)
    for c in range(3):
        sr_scaled[:, :, c] = (sr[:, :, c] - sr[:, :, c].mean()) / (sr[:, :, c].std() + 1e-6) * ref[:, :, c].std() + ref[:, :, c].mean()
    sr_scaled = np.clip(sr_scaled, 0.0, 1.0)
    print(f"With Radiometric Normalization: PSNR = {compute_psnr(sr_scaled, ref):.2f} dB, SSIM = {compute_ssim(sr_scaled, ref):.4f}, SAM = {compute_sam(sr_scaled, ref):.2f} deg")
    print("=" * 80)

if __name__ == "__main__":
    main()
