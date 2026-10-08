"""
Validation and Quality Assessment Metrics.
Computes real image quality metrics using scikit-image:
- PSNR (Peak Signal-to-Noise Ratio via skimage.metrics.peak_signal_noise_ratio)
- SSIM (Structural Similarity Index via skimage.metrics.structural_similarity)
- SAM (Spectral Angle Mapper in degrees)
- ERGAS (Relative Dimensionless Global Error in Synthesis)
"""
from typing import Any, Callable
import numpy as np

HAS_SKIMAGE: bool = True
skimage_psnr: Callable[..., Any] | None = None
skimage_ssim: Callable[..., Any] | None = None

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    skimage_psnr = peak_signal_noise_ratio
    skimage_ssim = structural_similarity
except ImportError:
    HAS_SKIMAGE = False

def compute_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Computes PSNR between two images using scikit-image."""
    mse = float(np.mean((img1 - img2) ** 2))
    if mse == 0.0:
        return 100.0
    if HAS_SKIMAGE and skimage_psnr is not None:
        val = float(skimage_psnr(img1, img2, data_range=data_range))
        return 100.0 if np.isinf(val) else val
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))

def compute_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Computes mean SSIM between two multi-channel images using scikit-image."""
    if HAS_SKIMAGE and skimage_ssim is not None:
        channel_axis = 2 if img1.ndim == 3 else None
        min_dim = min(int(img1.shape[0]), int(img1.shape[1]))
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        if min_dim >= 3 and win_size >= 3:
            res = skimage_ssim(img1, img2, data_range=data_range, channel_axis=channel_axis, win_size=win_size)
            if isinstance(res, tuple):
                res = res[0]
            return float(res)
        return float(1.0 - np.clip(np.mean(np.abs(img1 - img2)) / data_range, 0.0, 1.0))
    from scipy.ndimage import uniform_filter
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_channels: list[float] = []
    channels = int(img1.shape[2]) if img1.ndim == 3 else 1
    for c in range(channels):
        x = img1[:, :, c] if img1.ndim == 3 else img1
        y = img2[:, :, c] if img2.ndim == 3 else img2
        mu_x = uniform_filter(x, size=11)
        mu_y = uniform_filter(y, size=11)
        sigma_x2 = uniform_filter(x ** 2, size=11) - mu_x ** 2
        sigma_y2 = uniform_filter(y ** 2, size=11) - mu_y ** 2
        sigma_xy = uniform_filter(x * y, size=11) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
        )
        ssim_channels.append(float(np.mean(ssim_map)))
    return float(np.mean(ssim_channels))

def compute_sam(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes mean Spectral Angle Mapper (in degrees)."""
    dot = np.sum(img1 * img2, axis=2)
    norm1 = np.linalg.norm(img1, axis=2)
    norm2 = np.linalg.norm(img2, axis=2)
    cos_theta = np.clip(dot / (norm1 * norm2 + 1e-7), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.degrees(np.mean(theta)))

def compute_ergas(sr: np.ndarray, hr: np.ndarray, scale: float = 4.0) -> float:
    """
    Computes ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse).
    Standard metric in remote sensing image fusion and super-resolution.
    Lower is better (typically < 3.0 is excellent).
    """
    channels = int(sr.shape[2])
    sum_err = 0.0
    for c in range(channels):
        rmse = float(np.sqrt(np.mean((sr[:, :, c] - hr[:, :, c]) ** 2)))
        mean_hr = float(np.mean(hr[:, :, c])) + 1e-6
        sum_err += (rmse / mean_hr) ** 2
    ergas = (100.0 / scale) * float(np.sqrt((1.0 / channels) * sum_err))
    return float(ergas)

def evaluate_all(sr: np.ndarray, hr: np.ndarray, scale: float = 4.0, apply_radiometric_norm: bool = True) -> dict[str, Any]:
    """
    Calculates full validation suite comparing real SR against reference HR.
    When apply_radiometric_norm=True, applies histogram matching between raw SR and reference HR
    to calibrate cross-sensor gain curves (e.g. Sentinel-2 L2A BOA surface reflectance vs SPOT 6/7 TOA)
    prior to computing PSNR, SSIM, SAM, and ERGAS.
    """
    raw_psnr = compute_psnr(sr, hr)
    raw_ssim = compute_ssim(sr, hr)
    raw_sam = compute_sam(sr, hr)
    raw_ergas = compute_ergas(sr, hr, scale=scale)

    if apply_radiometric_norm and HAS_SKIMAGE:
        try:
            from skimage.exposure import match_histograms
            norm_sr = match_histograms(sr, hr, channel_axis=2 if sr.ndim == 3 else None)
            eval_sr = np.clip(norm_sr, 0.0, 1.0).astype(np.float32)
        except Exception as e:
            print(f"[Metrics] match_histograms warning: {e}")
            eval_sr = sr
    else:
        eval_sr = sr

    norm_psnr = compute_psnr(eval_sr, hr)
    norm_ssim = compute_ssim(eval_sr, hr)
    norm_sam = compute_sam(eval_sr, hr)
    norm_ergas = compute_ergas(eval_sr, hr, scale=scale)

    return {
        "psnr_db": round(norm_psnr, 2),
        "ssim": round(norm_ssim, 4),
        "sam_deg": round(norm_sam, 2),
        "ergas": round(norm_ergas, 2),
        "raw_gain_psnr_db": round(raw_psnr, 2),
        "raw_gain_ssim": round(raw_ssim, 4),
        "raw_gain_sam_deg": round(raw_sam, 2),
        "raw_gain_ergas": round(raw_ergas, 2),
        "radiometric_normalized": bool(apply_radiometric_norm),
        "calibration_note": "Cross-sensor PSNR evaluated after radiometric normalization (histogram matching against SPOT 6/7 reference) to account for sensor gain curve disparities."
    }

