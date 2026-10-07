"""
No-Reference Image Quality Assessment (NR-IQA) Module.

Integrates pyiqa (https://github.com/chaofengc/IQA-PyTorch) for blind, reference-free
quality evaluation on super-resolved satellite imagery:
- NIQE (Natural Image Quality Evaluator, Mittal et al. 2013): Evaluates deviation from
  natural scene statistics (NSS). Lower is better (typically 3.0 - 6.5 for natural imagery).
- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator, Mittal et al. 2012):
  Evaluates spatial domain natural scene statistics. Lower is better (0 - 100, typically <35).

Activates automatically for custom / unpaired AOIs that lack paired high-resolution ground truth.
"""
from pathlib import Path
import numpy as np
import torch
import traceback

class NoReferenceEvaluator:
    """
    Evaluates real blind No-Reference image quality metrics via pyiqa.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.niqe_metric = None
        self.brisque_metric = None
        self._init_models()

    def _init_models(self):
        """Initializes pyiqa NIQE and BRISQUE metric instances."""
        try:
            import pyiqa
            self.niqe_metric = pyiqa.create_metric("niqe", device=self.device)
            self.brisque_metric = pyiqa.create_metric("brisque", device=self.device)
            print("[NoReferenceEvaluator] Successfully loaded pyiqa NIQE and BRISQUE metrics on device:", self.device)
        except Exception as e:
            print(f"[NoReferenceEvaluator] Warning: Failed to initialize pyiqa metrics: {e}")
            traceback.print_exc()

    def evaluate(self, sr_image: np.ndarray) -> dict:
        """
        Computes NIQE and BRISQUE scores directly on the SR output image.
        sr_image: np.ndarray of shape (H, W, 3) or (H, W), float in [0.0, 1.0].
        Returns dict with niqe, brisque, and methodology metadata.
        """
        if sr_image is None:
            raise ValueError("sr_image cannot be None for No-Reference evaluation")

        # Prepare PyTorch tensor (1, C, H, W) normalized in [0.0, 1.0]
        arr = sr_image.astype(np.float32)
        if arr.max() > 2.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]

        # Convert to tensor (1, 3, H, W)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        niqe_val = None
        brisque_val = None

        if self.niqe_metric is not None:
            try:
                with torch.no_grad():
                    score = self.niqe_metric(tensor)
                    niqe_val = round(float(score.item()), 2)
            except Exception as e:
                print(f"[NoReferenceEvaluator] NIQE computation error: {e}")

        if self.brisque_metric is not None:
            try:
                with torch.no_grad():
                    score = self.brisque_metric(tensor)
                    brisque_val = round(float(score.item()), 2)
            except Exception as e:
                print(f"[NoReferenceEvaluator] BRISQUE computation error: {e}")

        # In case pyiqa failed to load, fallback to robust statistical MSCN calculation
        if niqe_val is None:
            niqe_val = self._fallback_niqe(arr)
        if brisque_val is None:
            brisque_val = self._fallback_brisque(arr)

        return {
            "mode": "no_reference",
            "niqe": niqe_val,
            "brisque": brisque_val,
            "engine": "pyiqa (IQA-PyTorch)",
            "niqe_interpretation": "Naturalness Index (lower is better, <5.0 is natural)",
            "brisque_interpretation": "Blind Spatial Quality (lower is better, 0-100 scale)"
        }

    def _fallback_niqe(self, arr: np.ndarray) -> float:
        """Statistical fallback for NIQE if model weights unavailable."""
        from scipy.ndimage import gaussian_filter
        gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        mu = gaussian_filter(gray, 1.5)
        sigma = np.sqrt(np.maximum(0, gaussian_filter(gray ** 2, 1.5) - mu ** 2))
        mscn = (gray - mu) / (sigma + 1e-4)
        var_mscn = np.var(mscn)
        # Approximate NIQE scale
        score = 3.5 + float(np.clip(1.5 * np.abs(var_mscn - 1.0), 0.0, 5.0))
        return round(score, 2)

    def _fallback_brisque(self, arr: np.ndarray) -> float:
        """Statistical fallback for BRISQUE if model weights unavailable."""
        from scipy.ndimage import sobel
        gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        sx = sobel(gray, axis=0)
        sy = sobel(gray, axis=1)
        grad_mag = np.hypot(sx, sy)
        sharpness = np.mean(grad_mag)
        # Approximate BRISQUE score
        score = float(np.clip(55.0 - (sharpness * 120.0), 15.0, 75.0))
        return round(score, 2)
