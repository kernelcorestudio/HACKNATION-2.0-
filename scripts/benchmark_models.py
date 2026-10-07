"""
Comprehensive model benchmarking script for HACKNATION Super-Resolution Mapping.
Evaluates HAT, SRM-Net, and CARN against paired Sentinel-2 <-> SPOT 6/7 1.5m ground truth.
Calculates PSNR, SSIM, SAM, ERGAS, NIQE, BRISQUE, inference latency, parameter counts, and variance.
"""
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from backend.models.sr_engine import SREngine
from backend.validation.metrics import evaluate_all
from backend.validation.reference_manager import PREPARED_HR_REFERENCES
from backend.usp.hallucination_detector import HallucinationDetector

def benchmark():
    print("=" * 80)
    print("HACKNATION MODEL BENCHMARKING SUITE (Sentinel-2 10m -> 2.5m)")
    print("=" * 80)

    models_to_test = ["hat", "srmnet"]
    datasets = ["punjab_agri", "delhi_ncr", "varanasi_river"]

    # Try pyiqa if available
    has_pyiqa = False
    try:
        import pyiqa
        niqe_metric = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)
        brisque_metric = pyiqa.create_metric('brisque', device=torch.device('cpu'), as_loss=False)
        has_pyiqa = True
    except Exception:
        has_pyiqa = False

    hallu_detector = HallucinationDetector()

    results = {}

    for model_name in models_to_test:
        engine = SREngine(scale_factor=4, model_name=model_name)
        param_count = sum(p.numel() for p in engine.model.parameters()) / 1e6
        results[model_name] = {"params_m": round(param_count, 2), "scenes": {}}

        print(f"\nEvaluating Architecture: {model_name.upper()} ({param_count:.2f}M Parameters, Device: {engine.device.upper()})")
        print("-" * 80)

        for aoi_id in datasets:
            ref_info = PREPARED_HR_REFERENCES[aoi_id]
            tile_path = Path(f"cache/tiles/{aoi_id}_real.npy")
            if not tile_path.exists():
                continue
            
            lr_tile = np.load(tile_path)[:, :, :3]
            hr_ref = np.load(ref_info["relative_path"])

            # Warmup
            _ = engine.predict(lr_tile[:32, :32, :])

            # Measure Latency (Deterministic)
            t0 = time.perf_counter()
            sr_det = engine.predict(lr_tile)
            det_latency = (time.perf_counter() - t0) * 1000

            # Measure Latency (MC-Dropout 8 passes)
            t0 = time.perf_counter()
            sr_mc, mc_var, var_stats = engine.predict_with_uncertainty(lr_tile, num_samples=8)
            mc_latency = (time.perf_counter() - t0) * 1000

            # Evaluate Metrics
            val_metrics = evaluate_all(sr_det, hr_ref, scale=4.0)
            usp_metrics = hallu_detector.evaluate(lr_tile, sr_det, mc_var)

            niqe_val = None
            brisque_val = None
            if has_pyiqa:
                try:
                    sr_tensor = torch.from_numpy(sr_det).permute(2, 0, 1).unsqueeze(0).float()
                    niqe_val = round(float(niqe_metric(sr_tensor).item()), 2)
                    brisque_val = round(float(brisque_metric(sr_tensor).item()), 2)
                except Exception:
                    pass

            scene_res = {
                "psnr_db": val_metrics["psnr_db"],
                "ssim": val_metrics["ssim"],
                "sam_deg": val_metrics["sam_deg"],
                "ergas": val_metrics["ergas"],
                "niqe": niqe_val,
                "brisque": brisque_val,
                "det_latency_ms": round(det_latency, 1),
                "mc_latency_ms": round(mc_latency, 1),
                "avg_confidence": usp_metrics["avg_confidence"],
                "mean_variance": var_stats["raw_variance_mean"]
            }
            results[model_name]["scenes"][aoi_id] = scene_res

            print(f"[{aoi_id.upper()}] PSNR: {scene_res['psnr_db']} dB | SSIM: {scene_res['ssim']} | SAM: {scene_res['sam_deg']}° | ERGAS: {scene_res['ergas']} | Conf: {scene_res['avg_confidence']}% | Latency (Det/MC-8): {scene_res['det_latency_ms']}/{scene_res['mc_latency_ms']} ms")

    return results

if __name__ == "__main__":
    benchmark()
