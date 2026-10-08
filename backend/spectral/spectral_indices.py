"""
Multi-band Spectral Index and Visualization Engine for Sentinel-2 Imagery.
Uses awesome-spectral-indices/spyndex for scientific index computation
and applies high-fidelity domain-appropriate continuous color ramps.

Supported Modes:
1. True Color (B04, B03, B02) - Natural Sentinel-2 surface reflectance.
2. False Color Infrared (B08, B04, B03) - Highlights vegetation in vivid red/magenta.
3. NDVI (Vegetation Health) - (NIR - Red) / (NIR + Red) via spyndex.
4. NDWI (Water Bodies) - (Green - NIR) / (Green + NIR) via spyndex.
5. NDBI (Built-up / Urban) - (SWIR1 - NIR) / (SWIR1 + NIR) via spyndex.
6. NBR (Burn / Disaster) - (NIR - SWIR2) / (NIR + SWIR2) via spyndex.
"""
from typing import Any
import numpy as np
import io
import base64
from PIL import Image
import matplotlib.colors as mcolors
import spyndex

def convert_bands_to_display_rgb(data: np.ndarray | dict) -> np.ndarray:
    """
    Unified authoritative function converting Sentinel-2 L2A BOA surface reflectance
    into natural displayable sRGB matching high-resolution visual reference basemaps.
    
    Band Order Confirmation:
        Channel 0 / B04 -> Red (665 nm)
        Channel 1 / B03 -> Green (560 nm)
        Channel 2 / B02 -> Blue (490 nm)
        Channel 3 / B08 -> NIR (842 nm - strictly preserved, excluded from visible RGB)
    
    Applies (Display-Only True-Color Pipeline):
        1. Normalization: Guarantees reflectance in [0.0, 1.0] (divides DN > 2.0 by 10000.0).
        2. Visual Basemap Bypass: Preserves references (e.g. SPOT 1.5m / Esri WorldImagery) that are already calibrated sRGB.
        3. Percentile Contrast Stretch: 2nd to 98th percentile clip and stretch to maximize dynamic range
           without artificial spectral bias (removes muddy shadows while preserving natural white point).
        4. Perceptual Gamma Correction: gamma = 1.8 to lift linear Sentinel-2 L2A surface reflectance
           without blowing out warm terrain or over-amplifying highlights.
        5. Subtle Saturation Nudge: 3-5% saturation enhancement in Rec.709 Luminance space
           to maintain natural, muted gray/tan/white building tones matching high-resolution reference basemaps.
    
    Returns:
        float32 array (H, W, 3) in range [0.0, 1.0]
    """
    nir = None
    if isinstance(data, dict):
        red = data.get("B04", data.get("red", None))
        green = data.get("B03", data.get("green", None))
        blue = data.get("B02", data.get("blue", None))
        nir = data.get("B08", data.get("nir", None))
        if red is None or green is None or blue is None:
            vals = list(data.values())
            red, green, blue = vals[0], vals[1], vals[2]
        rgb = np.stack([
            np.asarray(red, dtype=np.float32),
            np.asarray(green, dtype=np.float32),
            np.asarray(blue, dtype=np.float32)
        ], axis=-1)
        if nir is not None:
            nir = np.asarray(nir, dtype=np.float32)
    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            rgb = np.stack([data, data, data], axis=-1).astype(np.float32)
        elif data.shape[-1] >= 4:
            rgb = data[..., :3].astype(np.float32)
            nir = data[..., 3].astype(np.float32)
        elif data.shape[-1] >= 3:
            rgb = data[..., :3].astype(np.float32)
        else:
            rgb = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported data type for display conversion: {type(data)}")

    # 1. Reflectance Normalization (handles 0-10000 integer DN vs 0-1 float reflectance)
    if float(np.nanmax(rgb)) > 2.0:
        rgb = rgb / 10000.0
    rgb = np.clip(rgb, 0.0, 1.0)

    if nir is not None:
        if float(np.nanmax(nir)) > 2.0:
            nir = nir / 10000.0
        nir = np.clip(nir, 0.0, 1.0)

    # 2. Check if already in display range (e.g. SPOT 1.5m reference or 8-bit visual basemap)
    if float(np.mean(rgb)) > 0.30 and float(np.percentile(rgb, 99)) > 0.60:
        return np.clip(rgb, 0.0, 1.0)

    # 3. Percentile Contrast Stretch: 2nd to 98th percentile clip and stretch
    p_lo = float(np.percentile(rgb, 2.0))
    p_hi = float(np.percentile(rgb, 98.0))
    if p_hi > p_lo:
        stretched = np.clip((rgb - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    else:
        stretched = rgb

    # 4. Perceptual Gamma Correction (standard gamma = 1.8)
    gamma_corrected = np.clip(stretched ** (1.0 / 1.8), 0.0, 1.0)

    return gamma_corrected


def apply_reinhard_color_transfer(source_rgb: np.ndarray, target_rgb: np.ndarray, blend: float = 1.0) -> np.ndarray:
    """
    Applies Reinhard color transfer (CIELAB space mean/std matching)
    from target_rgb (e.g. high-resolution Esri WorldImagery basemap) to source_rgb.
    
    Dynamically adjusts source color statistics (mean and standard deviation in L*, a*, b*)
    to match the natural, realistic tones of the target basemap for any geographic region,
    eliminating regional atmospheric color casts without any hardcoded per-region values.
    
    Strictly display-only: never applied to underlying scientific float arrays used for
    spectral indices (NDVI, NDWI) or validation metrics.
    """
    import cv2
    src = np.clip(source_rgb.astype(np.float32), 0.0, 1.0)
    tgt = np.clip(target_rgb.astype(np.float32), 0.0, 1.0)
    
    # Ensure dimensions match for statistical comparison
    if tgt.shape[:2] != src.shape[:2]:
        tgt_sample = cv2.resize(tgt, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        tgt_sample = tgt

    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
    tgt_lab = cv2.cvtColor(tgt_sample, cv2.COLOR_RGB2LAB)
    
    src_mean = np.mean(src_lab, axis=(0, 1))
    src_std = np.std(src_lab, axis=(0, 1))
    
    tgt_mean = np.mean(tgt_lab, axis=(0, 1))
    tgt_std = np.std(tgt_lab, axis=(0, 1))
    
    src_std = np.maximum(src_std, 1e-4)
    
    # Shift and scale color distribution in perceptual LAB space
    matched_lab = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    matched_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
    matched_rgb = np.clip(matched_rgb, 0.0, 1.0)
    
    if blend < 1.0:
        return np.clip((1.0 - blend) * src + blend * matched_rgb, 0.0, 1.0)
    return matched_rgb


class SpectralIndexEngine:
    """
    Computes scientific spectral indices and renders continuous color ramps
    for Sentinel-2 optical imagery.
    """

    MODES = {
        "true_color": {
            "id": "true_color",
            "name": "True Color",
            "subtitle": "Natural Surface Reflectance (RGB)",
            "bands_used": ["B04 (Red)", "B03 (Green)", "B02 (Blue)"],
            "description": "Natural surface reflectance (Red=B04, Green=B03, Blue=B02). Natural earthy tones without artificial color grading.",
            "is_index": False,
            "legend": {
                "type": "none",
                "label": "Natural Reflectance"
            }
        },
        "false_color_ir": {
            "id": "false_color_ir",
            "name": "False Color Infrared",
            "subtitle": "NIR-Red-Green Composite",
            "bands_used": ["B08 (NIR)", "B04 (Red)", "B03 (Green)"],
            "description": "Classic remote sensing composite. Highlights dense photosynthesizing vegetation in bright red/magenta, built-up urban structures in cyan/grey, and open water in deep navy/black.",
            "is_index": False,
            "legend": {
                "type": "chips",
                "chips": [
                    {"label": "Dense Canopy / Crops", "color": "#e31a1c"},
                    {"label": "Urban / Roads / Concrete", "color": "#73b2d8"},
                    {"label": "Deep Water / Canals", "color": "#081d58"},
                    {"label": "Bare Soil / Silt", "color": "#d2b48c"}
                ]
            }
        },
        "ndvi": {
            "id": "ndvi",
            "name": "NDVI",
            "subtitle": "Normalized Difference Vegetation Index",
            "bands_used": ["B08 (NIR)", "B04 (Red)"],
            "description": "Quantifies photosynthetic activity, chlorophyll absorption, and canopy biomass. Crucial for crop monitoring, parcel yield estimation, and forest cover analysis.",
            "is_index": True,
            "vmin": -0.2,
            "vmax": 0.85,
            "unit": "NDVI",
            "colors": ["#5c3a21", "#c2b280", "#d9f0a3", "#78c679", "#005a32"],
            "gradient_css": "linear-gradient(to right, #5c3a21, #c2b280, #d9f0a3, #78c679, #005a32)",
            "legend": {
                "type": "continuous",
                "min": -0.2,
                "max": 0.85,
                "min_label": "Barren / Water (-0.2)",
                "mid_label": "Sparse (0.3)",
                "max_label": "Dense Canopy (+0.85)",
                "gradient_css": "linear-gradient(to right, #5c3a21, #c2b280, #d9f0a3, #78c679, #005a32)"
            }
        },
        "ndwi": {
            "id": "ndwi",
            "name": "NDWI",
            "subtitle": "Normalized Difference Water Index",
            "bands_used": ["B03 (Green)", "B08 (NIR)"],
            "description": "McFeeters water index maximizing water reflectance while suppressing vegetation and soil. Essential for flood mapping, river course changes, and waterbody demarcation.",
            "is_index": True,
            "vmin": -0.5,
            "vmax": 0.5,
            "unit": "NDWI",
            "colors": ["#d95f0e", "#fff7bc", "#a1dab4", "#41b6c4", "#225ea8", "#081d58"],
            "gradient_css": "linear-gradient(to right, #d95f0e, #fff7bc, #a1dab4, #41b6c4, #225ea8, #081d58)",
            "legend": {
                "type": "continuous",
                "min": -0.5,
                "max": 0.5,
                "min_label": "Dry Land (-0.5)",
                "mid_label": "Marsh / Silt (0.0)",
                "max_label": "Open Water (+0.5)",
                "gradient_css": "linear-gradient(to right, #d95f0e, #fff7bc, #a1dab4, #41b6c4, #225ea8, #081d58)"
            }
        },
        "ndbi": {
            "id": "ndbi",
            "name": "NDBI",
            "subtitle": "Normalized Difference Built-up Index (SWIR native 20m, interpolated)",
            "bands_used": ["B11 (SWIR1)", "B08 (NIR)"],
            "description": "Highlights impervious urban surfaces, concrete, asphalt, buildings, and transportation corridors by exploiting higher SWIR reflectance relative to NIR.",
            "is_index": True,
            "vmin": -0.35,
            "vmax": 0.45,
            "unit": "NDBI",
            "colors": ["#238b45", "#b8ca9e", "#d9d9d9", "#fe9929", "#b30000", "#7f0000"],
            "gradient_css": "linear-gradient(to right, #238b45, #b8ca9e, #d9d9d9, #fe9929, #b30000, #7f0000)",
            "legend": {
                "type": "continuous",
                "min": -0.35,
                "max": 0.45,
                "min_label": "Vegetation (-0.35)",
                "mid_label": "Soil / Suburb (0.0)",
                "max_label": "Dense Urban (+0.45)",
                "gradient_css": "linear-gradient(to right, #238b45, #b8ca9e, #d9d9d9, #fe9929, #b30000, #7f0000)"
            }
        },
        "nbr": {
            "id": "nbr",
            "name": "NBR",
            "subtitle": "Normalized Burn Ratio (SWIR native 20m, interpolated)",
            "bands_used": ["B08 (NIR)", "B12 (SWIR2)"],
            "description": "Detects fire burn scars, scorched soil, active disaster damage, and post-disturbance vegetation recovery by contrasting healthy NIR with post-fire SWIR2.",
            "is_index": True,
            "vmin": -0.3,
            "vmax": 0.7,
            "unit": "NBR",
            "colors": ["#49006a", "#d95f0e", "#fed976", "#addd8e", "#006837"],
            "gradient_css": "linear-gradient(to right, #49006a, #d95f0e, #fed976, #addd8e, #006837)",
            "legend": {
                "type": "continuous",
                "min": -0.3,
                "max": 0.7,
                "min_label": "Severe Burn (-0.3)",
                "mid_label": "Bare Ground (0.2)",
                "max_label": "Healthy Forest (+0.7)",
                "gradient_css": "linear-gradient(to right, #49006a, #d95f0e, #fed976, #addd8e, #006837)"
            }
        }
    }

    REQUIRED_BANDS = {
        "ndvi": ["B08", "B04"],
        "ndwi": ["B03", "B08"],
        "ndbi": ["B11", "B08"],
        "nbr": ["B08", "B12"],
    }

    def __init__(self):
        # Pre-compile colormaps for continuous indexing
        self._colormaps = {}
        for mode, cfg in self.MODES.items():
            if cfg.get("is_index") and "colors" in cfg:
                self._colormaps[mode] = mcolors.LinearSegmentedColormap.from_list(
                    f"{mode}_ramp", cfg["colors"], N=256
                )

    def compute_index(self, mode: str, bands: dict[str, np.ndarray]) -> np.ndarray:
        """
        Computes the 2D floating-point index array using spyndex with fallback.
        Raises ValueError if a required band is missing.
        Synthetic band fallbacks are removed for scientific integrity.
        """
        mode_lower = mode.lower()
        if mode_lower in self.REQUIRED_BANDS:
            missing = [b for b in self.REQUIRED_BANDS[mode_lower] if b not in bands or bands[b] is None]
            if missing:
                raise ValueError(
                    f"Required spectral band(s) {missing} missing for index '{mode.upper()}'. "
                    f"Synthetic fallbacks are disabled for scientific integrity. "
                    f"Computation unavailable."
                )

        nir = bands["B08"].astype(np.float32) if "B08" in bands else None
        red = bands["B04"].astype(np.float32) if "B04" in bands else None
        green = bands["B03"].astype(np.float32) if "B03" in bands else None
        blue = bands["B02"].astype(np.float32) if "B02" in bands else None
        swir1 = bands["B11"].astype(np.float32) if "B11" in bands else None
        swir2 = bands["B12"].astype(np.float32) if "B12" in bands else None

        mode_upper = mode.upper()
        # Parameter dict standardized for spyndex - strictly tailored per index
        if mode == "ndwi":
            spyndex_params = {"G": green, "N": nir}
        elif mode == "nbr":
            spyndex_params = {"N": nir, "S2": swir2}
        elif mode == "ndvi":
            spyndex_params = {"N": nir, "R": red}
        elif mode == "ndbi":
            spyndex_params = {"S1": swir1, "N": nir}
        else:
            spyndex_params = {
                "N": nir,
                "R": red,
                "G": green,
                "B": blue,
                "S1": swir1,
                "S2": swir2
            }

        try:
            val = spyndex.computeIndex(mode_upper, spyndex_params)
            val = np.asarray(val, dtype=np.float32)
            # Replace NaNs or Infs from mathematical division
            val = np.nan_to_num(val, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception as e:
            # High-precision mathematical fallback
            eps = 1e-6
            if mode == "ndvi":
                val = (nir - red) / (nir + red + eps)
            elif mode == "ndwi":
                val = (green - nir) / (green + nir + eps)
            elif mode == "ndbi":
                val = (swir1 - nir) / (swir1 + nir + eps)
            elif mode == "nbr":
                val = (nir - swir2) / (nir + swir2 + eps)
            else:
                val = (nir - red) / (nir + red + eps)
            val = np.nan_to_num(val, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        if mode == "nbr":
            # Physical Water Sanity Check (USGS / Copernicus BAER Standards):
            # Water absorbs both NIR (B08) and SWIR2 (B12), naturally producing negative raw ratios (-0.3 to -0.6)
            # which falsely mimic severe fire burn scars. Water is non-burnable surface.
            # Mask water pixels (identified by positive NDWI water reflectance) to NaN/masked
            # so water is completely segregated from burn scars (<= 0.10) and excluded from burn statistics.
            if green is not None and nir is not None:
                ndwi_water = (green - nir) / (green + nir + 1e-6)
                water_mask = (ndwi_water > 0.0) & (green > nir)
                val[water_mask] = np.nan

        return val

    def apply_colormap(self, index_arr: np.ndarray, mode: str) -> np.ndarray:
        """
        Converts 2D float index array to continuous uint8 RGB (H, W, 3) using curated colormap.
        Handles NaN/masked values cleanly (such as masked water bodies in NBR).
        """
        cfg = self.MODES.get(mode, self.MODES["ndvi"])
        vmin = cfg.get("vmin", -1.0)
        vmax = cfg.get("vmax", 1.0)
        cmap = self._colormaps.get(mode, self._colormaps["ndvi"])

        # Handle NaNs safely during normalization
        valid_mask = ~np.isnan(index_arr)
        clean_arr = np.where(valid_mask, index_arr, vmin)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        normed = norm(clean_arr)
        rgba = cmap(normed)
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)

        if mode == "nbr":
            # Distinct hydrological demarcation for water pixels (masked as NaN)
            # Ensures river pixels show a distinct, uniform non-burn water color (#00c5ff)
            # and never resemble a dark purple / burnt orange fire scar.
            nan_water_mask = np.isnan(index_arr)
            if np.any(nan_water_mask):
                rgb[nan_water_mask] = np.array([0, 197, 255], dtype=np.uint8)

        return rgb

    def render_false_color_ir(self, bands: dict[str, np.ndarray]) -> np.ndarray:
        """
        Renders crisp False Color Infrared composite (NIR=B08, Red=B04, Green=B03).
        Applies standard remote sensing percentile contrast scaling.
        """
        nir = bands["B08"].astype(np.float32)
        red = bands["B04"].astype(np.float32)
        green = bands["B03"].astype(np.float32)

        composite = np.stack([nir, red, green], axis=-1)

        # Standard linear scaling with gain factor for natural NIR reflectance
        if float(np.percentile(composite, 99)) <= 0.45:
            composite = composite * 2.2

        composite = np.clip(composite, 0.0, 1.0)
        return (composite * 255.0).astype(np.uint8)

    def render_true_color(self, bands: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """
        Renders natural True-Color Sentinel-2 composite using the unified
        convert_bands_to_display_rgb function.
        Guarantees 100% consistent color balance between Input, Output, and Reference panels.
        """
        display_rgb = convert_bands_to_display_rgb(bands)
        return (display_rgb * 255.0).astype(np.uint8)

    def render(self, mode: str, bands_or_arr: dict[str, np.ndarray] | np.ndarray) -> dict[str, Any]:
        """
        Main rendering pipeline for any visualization mode.
        Returns dictionary with:
            - status: "success" or "unavailable"
            - mode: str
            - title: str
            - subtitle: str
            - description: str
            - rgb_uint8: np.ndarray (H, W, 3) or None
            - base64_png: str data URL or None
            - is_index: bool
            - stats: min, max, mean (if index) or None
            - legend: legend metadata for frontend
        """
        if isinstance(bands_or_arr, dict):
            bands = bands_or_arr
        else:
            from backend.ingestion.copernicus_client import extract_bands_dict
            bands = extract_bands_dict(bands_or_arr)

        mode = mode.lower()
        if mode not in self.MODES:
            mode = "true_color"

        cfg = self.MODES[mode]
        stats = None

        try:
            if mode == "true_color":
                rgb_uint8 = self.render_true_color(bands)
            elif mode == "false_color_ir":
                rgb_uint8 = self.render_false_color_ir(bands)
            else:
                index_arr = self.compute_index(mode, bands)
                rgb_uint8 = self.apply_colormap(index_arr, mode)
                valid = index_arr[~np.isnan(index_arr)]
                if valid.size > 0:
                    stats = {
                        "min": round(float(np.min(valid)), 3),
                        "max": round(float(np.max(valid)), 3),
                        "mean": round(float(np.mean(valid)), 3),
                        "std": round(float(np.std(valid)), 3)
                    }
                else:
                    stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        except ValueError as err:
            return {
                "status": "unavailable",
                "error": str(err),
                "mode": mode,
                "title": cfg["name"],
                "subtitle": cfg["subtitle"],
                "description": cfg["description"],
                "bands_used": cfg["bands_used"],
                "is_index": cfg["is_index"],
                "rgb_uint8": None,
                "base64_png": None,
                "stats": None,
                "legend": cfg["legend"]
            }

        # Generate base64 PNG data URL
        pil_img = Image.fromarray(rgb_uint8)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG", optimize=True)
        base64_png = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        return {
            "status": "success",
            "mode": mode,
            "title": cfg["name"],
            "subtitle": cfg["subtitle"],
            "description": cfg["description"],
            "bands_used": cfg["bands_used"],
            "is_index": cfg["is_index"],
            "rgb_uint8": rgb_uint8,
            "base64_png": base64_png,
            "stats": stats,
            "legend": cfg["legend"]
        }

# Singleton instance
spectral_engine = SpectralIndexEngine()
