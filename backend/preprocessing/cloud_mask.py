"""
Cloud and shadow masking module.
Scientific multispectral optical cloud detection (B02, B03, B04, B08).
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import base64

class CloudMaskDetector:
    def __init__(self, threshold: float = 0.38):
        self.threshold = threshold
        print("[CloudMaskDetector] Initialized Physical Multi-Spectral Optical Cloud Detector")

    def detect(self, data: np.ndarray | dict) -> np.ndarray:
        """
        Detects cloud mask from multi-spectral bands dictionary or numpy array (H, W, C).
        Sentinel-2 band order:
          Channel 0: B04 (Red)
          Channel 1: B03 (Green)
          Channel 2: B02 (Blue)
          Channel 3 (if present): B08 (NIR)
        Returns: binary mask (H, W) where 1 indicates cloud/occlusion, 0 indicates clear sky.
        """
        if isinstance(data, dict):
            
            # Dictionary with individual band keys
            if "B04" in data and "B03" in data and "B02" in data:
                red = self._normalize_band(data["B04"])
                green = self._normalize_band(data["B03"])
                blue = self._normalize_band(data["B02"])
                nir = self._normalize_band(data["B08"]) if "B08" in data else None
                return self._compute_multispectral_mask(red, green, blue, nir)
            
            if "RGB" in data:
                rgb = self._normalize_band(data["RGB"])
                return self._compute_rgb_mask(rgb)

            first_val = next(iter(data.values()))
            shape = first_val.shape[:2]
            return np.zeros(shape, dtype=np.uint8)

        # Case 2: Numpy array (H, W, C)
        arr = np.asarray(data)
        if arr.ndim == 2:
            return (self._normalize_band(arr) > 0.45).astype(np.uint8)

        H, W, C = arr.shape
        if C >= 4:
            red = self._normalize_band(arr[:, :, 0])
            green = self._normalize_band(arr[:, :, 1])
            blue = self._normalize_band(arr[:, :, 2])
            nir = self._normalize_band(arr[:, :, 3])
            return self._compute_multispectral_mask(red, green, blue, nir)
        elif C >= 3:
            red = self._normalize_band(arr[:, :, 0])
            green = self._normalize_band(arr[:, :, 1])
            blue = self._normalize_band(arr[:, :, 2])
            return self._compute_multispectral_mask(red, green, blue, None)
        else:
            return np.zeros((H, W), dtype=np.uint8)

    def _normalize_band(self, band: np.ndarray) -> np.ndarray:
        b = band.astype(np.float32)
        if float(np.nanmax(b)) > 2.0:
            b = b / 10000.0
        return np.clip(b, 0.0, 1.0)

    # Alias for API compatibility
    detect_cloud_mask = detect

    def _compute_multispectral_mask(
        self,
        red: np.ndarray,
        green: np.ndarray,
        blue: np.ndarray,
        nir: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Scientific Sentinel-2 Cloud Detection:
        1. High visible brightness across all visible bands (B02, B03, B04).
        2. High spectral flatness / whiteness (low inter-band variance).
        3. High NIR reflectance (B08 > 0.22) if available.
        4. NDWI discrimination to prevent false positive flags on bright water or sandbars.
        5. Morphological dilation to engulf cloud fringes, halos, and shadow margins.
        """
        H, W = red.shape
        vis_mean = (red + green + blue) / 3.0

        # Whiteness: clouds have equal reflectance across red, green, blue
        diff_rg = np.abs(red - green)
        diff_gb = np.abs(green - blue)
        diff_rb = np.abs(red - blue)
        max_diff = np.maximum.reduce([diff_rg, diff_gb, diff_rb])
        whiteness = max_diff / (vis_mean + 1e-6)

        # Thick clouds: bright and white
        thick_clouds = (vis_mean > 0.42) & (whiteness < 0.35)

        # Medium / thin clouds
        medium_clouds = (vis_mean > 0.35) & (whiteness < 0.25)

        if nir is not None:
            # Clouds reflect strongly in NIR (> 0.20), unlike typical water or shadowed land
            cloud_candidates = (thick_clouds | medium_clouds) & (nir > 0.20)
            # Water exclusion via NDWI = (Green - NIR) / (Green + NIR)
            ndwi = (green - nir) / (green + nir + 1e-6)
            cloud_candidates = cloud_candidates & (ndwi < 0.40)
        else:
            cloud_candidates = thick_clouds | medium_clouds

        mask_raw = cloud_candidates.astype(np.uint8)

        # Morphological dilation: expand cloud mask by 3-5 pixels to capture diffuse halos & cloud borders
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dilated = cv2.dilate(mask_raw, kernel, iterations=1)

        # Log statistics
        cloud_pixels = int(np.sum(mask_dilated > 0))
        total_pixels = H * W
        coverage_pct = round((cloud_pixels / float(total_pixels)) * 100.0, 2)
        print(f"[CloudMaskDetector] Cloud Detection Active: {cloud_pixels}/{total_pixels} pixels flagged ({coverage_pct}% cloud coverage)")

        return mask_dilated

    def _compute_rgb_mask(self, rgb: np.ndarray) -> np.ndarray:
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]
        return self._compute_multispectral_mask(r, g, b, None)

    def get_cloud_coverage_percentage(self, cloud_mask: np.ndarray) -> float:
        if cloud_mask is None or cloud_mask.size == 0:
            return 0.0
        return round(float(np.mean(cloud_mask > 0)) * 100.0, 2)

    def generate_cloud_overlay_base64(self, cloud_mask: np.ndarray, target_size: tuple | None = None) -> str:
        return generate_cloud_overlay_base64(cloud_mask, target_shape=target_size)

    def apply_cloud_occlusion_to_sr(self, sr_image: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
        return apply_cloud_occlusion_to_sr(sr_image, cloud_mask)


def generate_cloud_overlay_base64(cloud_mask: np.ndarray, target_shape: tuple | None = None) -> str:
    """
    Generates a high-contrast RGBA PNG data URL representing the cloud mask.
    Clear sky pixels are 100% transparent.
    Cloud pixels are rendered with semi-transparent slate-cyan diagonal hatching and crisp border outlines.
    """
    if target_shape is not None and cloud_mask.shape != target_shape[:2]:
        H, W = target_shape[:2]
        mask_scaled = cv2.resize(cloud_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        H, W = cloud_mask.shape
        mask_scaled = cloud_mask.astype(np.uint8)

    # Base RGBA canvas
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    cloud_pts = (mask_scaled > 0)

    if not np.any(cloud_pts):
        # Fully transparent PNG if no clouds
        pil_img = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    # Soft natural semi-transparent cloud tint (no cartoon cyan borders or harsh stripes)
    rgba[cloud_pts] = [245, 248, 250, 110]

    pil_img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def apply_cloud_occlusion_to_sr(sr_image: np.ndarray, cloud_mask_sr: np.ndarray) -> np.ndarray:
    """
    Renders cloud-covered regions distinctly on the super-resolved output raster.
    Instead of showing plausible-looking but fabricated ground textures under clouds,
    cloud pixels are replaced with a styled, dimmed slate-gray hatched occlusion texture
    with a clear visual notification of occlusion.
    """
    H, W, C = sr_image.shape
    if cloud_mask_sr.shape != (H, W):
        mask = cv2.resize(cloud_mask_sr.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        mask = cloud_mask_sr.astype(np.uint8)

    cloud_pts = (mask > 0)
    if not np.any(cloud_pts):
        return sr_image.copy()

    out = sr_image.copy()

    # Dimmed slate-gray base for occluded zone
    gray_base = np.array([0.45, 0.48, 0.52], dtype=np.float32)
    out[cloud_pts] = 0.35 * out[cloud_pts] + 0.65 * gray_base

    # Diagonal hatching lines
    y_idx, x_idx = np.indices((H, W))
    hatch = ((x_idx + y_idx) % 10 < 3) & cloud_pts
    out[hatch] = np.clip(out[hatch] + 0.25, 0.0, 1.0)

    return np.clip(out, 0.0, 1.0)
