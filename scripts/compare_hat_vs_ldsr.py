"""
Side-by-Side Comparison Script: HAT vs LDSR-S2 vs Ground Truth Reference.
Evaluates:
1. Full 3-Way Composite (HAT vs LDSR-S2 vs 1.5m Reference)
2. Detail Zoom Crop (High-Frequency Edge / Texture comparison)
3. Confidence & Uncertainty Maps (Hallucination Risk Assessment)
4. Scientific Metrics (PSNR, SSIM, SAM, ERGAS, Avg Confidence)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.cm as cm

from backend.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
ARTIFACT_DIR = Path(r"C:\Users\ankus\.gemini\antigravity-ide\brain\a1b847be-bf64-49f6-b4c2-a3e1c44be5e6")

def main():
    print("=" * 80)
    print("COMPARATIVE BENCHMARK: HAT vs. LDSR-S2 vs. REFERENCE")
    print("=" * 80)

    # 1. Fetch Varanasi AOI (Tier 1 Ground Truth)
    aoi_id = "varanasi_river"
    print(f"\n[1] Fetching active scene: {aoi_id}...")
    r_fetch = client.post("/api/fetch-tile", json={
        "bbox": [82.98, 25.28, 83.05, 25.35],
        "aoi_id": aoi_id,
        "max_cloud": 15
    })
    assert r_fetch.status_code == 200, f"Fetch failed: {r_fetch.text}"
    from backend.api.routes import current_session

    ref_hr = current_session["ref_hr"]
    print(f"Ground truth reference loaded: shape={ref_hr.shape}, range=[{ref_hr.min():.3f}, {ref_hr.max():.3f}]")

    # 2. Run HAT Inference (num_mc_samples=2)
    print("\n[2] Running HAT (Hybrid Attention Transformer) Inference...")
    r_hat = client.post("/api/superresolve", json={
        "num_mc_samples": 2,
        "scale_factor": 4,
        "model_name": "hat",
        "apply_realesrgan_sharpen": False,
        "apply_unsharp": False
    })
    assert r_hat.status_code == 200, f"HAT failed: {r_hat.text}"
    hat_data = r_hat.json()
    sr_hat = current_session["sr"].copy()
    conf_hat = current_session["confidence_map"].copy()

    hat_metrics = hat_data.get("validation_metrics", {})
    hat_confidence = hat_data["usp_metrics"]["avg_confidence"]
    hat_max_unc = hat_data["mc_variance_stats"]["raw_variance_max"]
    print(f"HAT Metrics: PSNR={hat_metrics.get('psnr_db', 0):.2f} dB, SSIM={hat_metrics.get('ssim', 0):.4f}, SAM={hat_metrics.get('sam_deg', 0):.2f}°, Confidence={hat_confidence:.2f}%")

    # 3. Run LDSR-S2 Inference (num_mc_samples=2)
    print("\n[3] Running LDSR-S2 (ESA Latent Diffusion) Inference...")
    r_ldsr = client.post("/api/superresolve", json={
        "num_mc_samples": 2,
        "scale_factor": 4,
        "model_name": "ldsr_s2",
        "apply_realesrgan_sharpen": False,
        "apply_unsharp": False
    })
    assert r_ldsr.status_code == 200, f"LDSR-S2 failed: {r_ldsr.text}"
    ldsr_data = r_ldsr.json()
    sr_ldsr = current_session["sr"].copy()
    conf_ldsr = current_session["confidence_map"].copy()

    ldsr_metrics = ldsr_data.get("validation_metrics", {})
    ldsr_confidence = ldsr_data["usp_metrics"]["avg_confidence"]
    ldsr_max_unc = ldsr_data["mc_variance_stats"]["raw_variance_max"]
    print(f"LDSR-S2 Metrics: PSNR={ldsr_metrics.get('psnr_db', 0):.2f} dB, SSIM={ldsr_metrics.get('ssim', 0):.4f}, SAM={ldsr_metrics.get('sam_deg', 0):.2f}°, Confidence={ldsr_confidence:.2f}%")

    # 4. Generate Visual Side-by-Side Composites
    print("\n[4] Generating Side-by-Side Visual Artifacts...")

    # Convert to uint8 RGB
    ref_u8 = (np.clip(ref_hr, 0.0, 1.0) * 255.0).astype(np.uint8)
    hat_u8 = (np.clip(sr_hat, 0.0, 1.0) * 255.0).astype(np.uint8)
    ldsr_u8 = (np.clip(sr_ldsr, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Resize reference if dimensions differ slightly
    if ref_u8.shape[:2] != hat_u8.shape[:2]:
        ref_u8 = cv2.resize(ref_u8, (hat_u8.shape[1], hat_u8.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    H, W, _ = hat_u8.shape

    # Composite 1: 3-Way Overview (HAT vs LDSR-S2 vs Ground Truth)
    header_h = 44
    canvas_w = W * 3
    canvas_h = H + header_h
    comp_full = Image.new("RGB", (canvas_w, canvas_h), (17, 24, 39))
    draw = ImageDraw.Draw(comp_full)

    comp_full.paste(Image.fromarray(hat_u8), (0, header_h))
    comp_full.paste(Image.fromarray(ldsr_u8), (W, header_h))
    comp_full.paste(Image.fromarray(ref_u8), (W * 2, header_h))

    # Headers with quantitative telemetry
    draw.text((12, 14), f"HAT (Transformer) - Conf: {hat_confidence:.1f}% | PSNR: {hat_metrics.get('psnr_db', 0):.2f} dB", fill=(56, 189, 248))
    draw.text((W + 12, 14), f"LDSR-S2 (Diffusion) - Conf: {ldsr_confidence:.1f}% | PSNR: {ldsr_metrics.get('psnr_db', 0):.2f} dB", fill=(245, 158, 11))
    draw.text((W * 2 + 12, 14), f"Ground Truth Reference (SPOT 1.5m)", fill=(52, 211, 153))

    out_full_path = ARTIFACT_DIR / "ldsr_vs_hat_comparison.png"
    comp_full.save(str(out_full_path))
    print(f"Saved: {out_full_path}")

    # Composite 2: Detail Zoom Crop (2x digital zoom on central 160x160 region)
    cx, cy = W // 2, H // 2
    cw = 80
    crop_hat = cv2.resize(hat_u8[cy-cw:cy+cw, cx-cw:cx+cw], (384, 384), interpolation=cv2.INTER_NEAREST)
    crop_ldsr = cv2.resize(ldsr_u8[cy-cw:cy+cw, cx-cw:cx+cw], (384, 384), interpolation=cv2.INTER_NEAREST)
    crop_ref = cv2.resize(ref_u8[cy-cw:cy+cw, cx-cw:cx+cw], (384, 384), interpolation=cv2.INTER_NEAREST)

    comp_crop = Image.new("RGB", (384 * 3, 384 + header_h), (17, 24, 39))
    draw_crop = ImageDraw.Draw(comp_crop)
    comp_crop.paste(Image.fromarray(crop_hat), (0, header_h))
    comp_crop.paste(Image.fromarray(crop_ldsr), (384, header_h))
    comp_crop.paste(Image.fromarray(crop_ref), (384 * 2, header_h))

    draw_crop.text((12, 14), "HAT 4x Detail Zoom (Structural Boundaries)", fill=(56, 189, 248))
    draw_crop.text((384 + 12, 14), "LDSR-S2 4x Detail Zoom (Diffusion Texture)", fill=(245, 158, 11))
    draw_crop.text((384 * 2 + 12, 14), "Ground Truth Reference (SPOT 1.5m)", fill=(52, 211, 153))

    out_crop_path = ARTIFACT_DIR / "ldsr_vs_hat_zoom.png"
    comp_crop.save(str(out_crop_path))
    print(f"Saved: {out_crop_path}")

    # Composite 3: Uncertainty / Confidence Hotspots
    conf_hat_colored = (cm.turbo(np.clip(conf_hat, 0.0, 1.0))[:, :, :3] * 255.0).astype(np.uint8)
    conf_ldsr_colored = (cm.turbo(np.clip(conf_ldsr, 0.0, 1.0))[:, :, :3] * 255.0).astype(np.uint8)

    comp_unc = Image.new("RGB", (W * 2, H + header_h), (17, 24, 39))
    draw_unc = ImageDraw.Draw(comp_unc)
    comp_unc.paste(Image.fromarray(conf_hat_colored), (0, header_h))
    comp_unc.paste(Image.fromarray(conf_ldsr_colored), (W, header_h))

    draw_unc.text((12, 14), f"HAT Confidence Heatmap (Avg: {hat_confidence:.1f}%)", fill=(56, 189, 248))
    draw_unc.text((W + 12, 14), f"LDSR-S2 Confidence Heatmap (Avg: {ldsr_confidence:.1f}%) - Hotspots", fill=(245, 158, 11))

    out_unc_path = ARTIFACT_DIR / "ldsr_vs_hat_uncertainty.png"
    comp_unc.save(str(out_unc_path))
    print(f"Saved: {out_unc_path}")

    # 5. Print Comparison Summary
    print("\n" + "=" * 88)
    print("QUANTITATIVE BENCHMARK SUMMARY")
    print("=" * 88)
    print(f"{'Metric':<30} | {'HAT (Transformer)':<22} | {'LDSR-S2 (Diffusion)':<22} | {'Advantage'}")
    print("-" * 88)
    print(f"{'Average Confidence Score':<30} | {hat_confidence:.2f}%{'':<15} | {ldsr_confidence:.2f}%{'':<15} | {'HAT (+Higher Trust)' if hat_confidence > ldsr_confidence else 'LDSR-S2'}")
    print(f"{'Peak Uncertainty (MC Var)':<30} | {hat_max_unc:.4f}{'':<16} | {ldsr_max_unc:.4f}{'':<16} | {'HAT (Lower Risk)' if hat_max_unc < ldsr_max_unc else 'LDSR-S2'}")
    print(f"{'PSNR vs SPOT 1.5m':<30} | {hat_metrics.get('psnr_db', 0):.2f} dB{'':<15} | {ldsr_metrics.get('psnr_db', 0):.2f} dB{'':<15} | {'HAT' if hat_metrics.get('psnr_db', 0) > ldsr_metrics.get('psnr_db', 0) else 'LDSR-S2'}")
    print(f"{'SSIM (Structural Similarity)':<30} | {hat_metrics.get('ssim', 0):.4f}{'':<16} | {ldsr_metrics.get('ssim', 0):.4f}{'':<16} | {'HAT' if hat_metrics.get('ssim', 0) > ldsr_metrics.get('ssim', 0) else 'LDSR-S2'}")
    print(f"{'SAM (Spectral Angle Mapper)':<30} | {hat_metrics.get('sam_deg', 0):.2f}°{'':<16} | {ldsr_metrics.get('sam_deg', 0):.2f}°{'':<16} | {'HAT (Closer Radiometry)' if hat_metrics.get('sam_deg', 0) < ldsr_metrics.get('sam_deg', 0) else 'LDSR-S2'}")
    print(f"{'Inference Architecture':<30} | {'Regression + W-MSA':<22} | {'Latent Diffusion (DDIM)':<22} | {'Textural Sharpness: LDSR'}")
    print("=" * 88)

if __name__ == "__main__":
    main()
