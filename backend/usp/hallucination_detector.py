"""
USP: Hallucination-Aware Uncertainty & Confidence Detector.
Combines Monte-Carlo Dropout epistemic variance with ESA opensr-test
spectral consistency & low-frequency cycle consistency checks.
"""
import numpy as np
from scipy.ndimage import zoom, gaussian_filter

class HallucinationDetector:
    """
    Evaluates trust and hallucination risk on Super-Resolved satellite imagery.
    Produces:
      1. Epistemic Uncertainty Map (model uncertainty via MC-Dropout variance)
      2. Spectral Consistency Map (deviation from Sentinel-2 observed spectral angles)
      3. Cycle Consistency Error (low-frequency downsampled SR vs original LR)
      4. Fused Confidence Map (0.0 = high hallucination risk, 1.0 = verified observed feature)
    """
    def __init__(self, mc_weight: float = 0.4, spectral_weight: float = 0.3, consistency_weight: float = 0.3):
        self.mc_weight = mc_weight
        self.spectral_weight = spectral_weight
        self.consistency_weight = consistency_weight

    def compute_spectral_angle(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Spectral Angle Mapper (SAM) per pixel between two multi-channel images (H, W, C).
        Returns angle in radians (0 = identical spectral signature).
        """
        dot = np.sum(img1 * img2, axis=2)
        norm1 = np.linalg.norm(img1, axis=2)
        norm2 = np.linalg.norm(img2, axis=2)
        denom = norm1 * norm2 + 1e-7
        cos_angle = np.clip(dot / denom, -1.0, 1.0)
        return np.arccos(cos_angle)

    def evaluate(
        self,
        lr_image: np.ndarray,
        sr_image: np.ndarray,
        mc_variance: np.ndarray = None,
        cloud_mask: np.ndarray = None
    ) -> dict:
        """
        lr_image: original Sentinel-2 LR tile (H, W, C)
        sr_image: super-resolved output (H*scale, W*scale, C)
        mc_variance: per-pixel variance map from MC-Dropout passes (H*scale, W*scale)
        cloud_mask: binary cloud mask (H, W) or (H_sr, W_sr) where 1=cloud, 0=clear
        """
        H_sr, W_sr, C = sr_image.shape
        H_lr, W_lr, _ = lr_image.shape
        scale = H_sr / H_lr

        # Handle and upscale cloud mask if provided
        if cloud_mask is not None:
            import cv2
            if cloud_mask.shape != (H_sr, W_sr):
                cloud_mask_sr = cv2.resize(cloud_mask.astype(np.uint8), (W_sr, H_sr), interpolation=cv2.INTER_NEAREST)
            else:
                cloud_mask_sr = cloud_mask.astype(np.uint8)
            cloud_pts = (cloud_mask_sr > 0)
            cloud_coverage_pct = round(float(np.mean(cloud_pts)) * 100.0, 2)
        else:
            cloud_pts = np.zeros((H_sr, W_sr), dtype=bool)
            cloud_mask_sr = np.zeros((H_sr, W_sr), dtype=np.uint8)
            cloud_coverage_pct = 0.0

        # 1. Cycle-consistency check (opensr-test principle: downsampled SR must reconstruct LR)
        sr_downsampled = zoom(sr_image, (1.0 / scale, 1.0 / scale, 1), order=1)
        # Ensure identical shape
        sr_downsampled = sr_downsampled[:H_lr, :W_lr, :]
        cycle_residual_lr = np.mean(np.abs(sr_downsampled - lr_image), axis=2)
        cycle_error_sr = zoom(cycle_residual_lr, (scale, scale), order=1)[:H_sr, :W_sr]
        # Normalize cycle error [0, 1]
        cycle_error_norm = np.clip(cycle_error_sr / 0.15, 0.0, 1.0)

        # 2. Spectral Consistency Check (SAM check against bicubic interpolated LR)
        lr_upscaled = zoom(lr_image, (scale, scale, 1), order=1)[:H_sr, :W_sr, :]
        sam_map = self.compute_spectral_angle(sr_image, lr_upscaled)
        # High spectral angle (> 0.2 rad) suggests unnatural spectral distortion
        sam_error_norm = np.clip(sam_map / 0.20, 0.0, 1.0)

        # 3. Epistemic Model Uncertainty (MC-Dropout variance)
        if mc_variance is not None:
            mc_norm = np.clip(mc_variance, 0.0, 1.0)
        else:
            mc_norm = np.zeros((H_sr, W_sr), dtype=np.float32)

        # 4. Total Hallucination Risk Index (0 = safe, 1 = severe hallucination)
        hallucination_risk = (
            self.mc_weight * mc_norm +
            self.spectral_weight * sam_error_norm +
            self.consistency_weight * cycle_error_norm
        )
        hallucination_risk = np.clip(hallucination_risk, 0.0, 1.0)

        # 5. Fused Confidence Map (1 = fully observed / trustworthy, 0 = synthetic / hallucinated)
        confidence_map = 1.0 - hallucination_risk

        # 6. Strict Cloud Occlusion Enforcement
        # For cloud-covered pixels, force confidence to strictly 0.0% (Zero ground truth data available)
        confidence_map[cloud_pts] = 0.0
        hallucination_risk[cloud_pts] = 1.0

        # Mean summary statistics
        clear_pts = ~cloud_pts
        if np.any(clear_pts):
            clear_sky_confidence = float(np.mean(confidence_map[clear_pts]) * 100.0)
        else:
            clear_sky_confidence = 0.0

        total_avg_confidence = float(np.mean(confidence_map) * 100.0)
        # Report clear sky confidence as the benchmark confidence, but note cloud coverage
        display_confidence = clear_sky_confidence if cloud_coverage_pct < 99.0 else 0.0

        avg_sam_deg = float(np.degrees(np.mean(sam_map[clear_pts]))) if np.any(clear_pts) else float(np.degrees(np.mean(sam_map)))
        mean_uncertainty = float(np.mean(mc_norm[clear_pts])) if np.any(clear_pts) else float(np.mean(mc_norm))

        print(f"[HallucinationDetector] Evaluated trust: Clear-Sky Confidence={display_confidence:.2f}%, Cloud Occlusion={cloud_coverage_pct}% ({np.sum(cloud_pts)} px set to 0.0% confidence)")

        return {
            "confidence_map": confidence_map,
            "hallucination_risk": hallucination_risk,
            "mc_variance": mc_norm,
            "sam_degrees": round(avg_sam_deg, 2),
            "avg_confidence": round(display_confidence, 2),
            "total_scene_confidence": round(total_avg_confidence, 2),
            "clear_sky_confidence": round(clear_sky_confidence, 2),
            "cloud_coverage_pct": cloud_coverage_pct,
            "cloud_mask_sr": cloud_mask_sr,
            "mean_uncertainty": round(mean_uncertainty, 4),
            "cycle_consistency_mae": round(float(np.mean(cycle_residual_lr)), 4)
        }
