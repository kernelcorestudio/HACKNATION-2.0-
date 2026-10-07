"""
Ground Truth High-Resolution Reference Manager.

Strictly manages independent high-resolution ground truth references across tiered tiers:
- Tier 1: Paired Scientific Ground Truth (e.g., WorldStrat SPOT 6/7 1.5m Optical).
  Strict coordinate and preset matching with loud assertion failure on scene mismatches.
- Tier 2: Open Regional / Orthophoto (reserved for regional aerial surveys).
- Tier 3: Global Reference Basemap (variable resolution) via leafmap (Esri.WorldImagery).
  Fallback when no scientific reference exists for an AOI. Caches GeoTIFF to disk.
  Never mislabeled as 'Ground Truth HR (1.5m)'. Evaluated with No-Reference Quality Assessment.
"""
from typing import TypedDict
from pathlib import Path
import numpy as np
import math
import hashlib
import time
from PIL import Image as PILImage

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class ReferenceEntry(TypedDict):
    aoi_id: str
    aoi_name: str
    file_name: str
    relative_path: str
    bbox: list[float]
    center: list[float]
    sensor: str
    source: str

class ResolvedReference(TypedDict, total=False):
    aoi_id: str
    aoi_name: str
    file_name: str
    relative_path: str
    bbox: list[float]
    center: list[float]
    sensor: str
    source: str
    full_path: str
    has_reference: bool
    is_scientific_ground_truth: bool
    tier: int
    tier_label: str

class TierInfo(TypedDict, total=False):
    tier: int
    tier_id: str
    tier_label: str
    tier_name: str
    is_scientific_ground_truth: bool
    assessment_mode: str
    source: str
    sensor: str
    file_name: str
    relative_path: str
    full_path: str
    bbox: list[float] | None
    has_reference: bool

# Explicit registry of AOIs with genuine, paired High-Resolution references on disk (Tier 1)
PREPARED_HR_REFERENCES: dict[str, ReferenceEntry] = {
    "punjab_agri": {
        "aoi_id": "punjab_agri",
        "aoi_name": "Punjab Agricultural Parcels",
        "file_name": "punjab_agri_spot_1.5m.npy",
        "relative_path": "data/reference_hr/punjab_agri_spot_1.5m.npy",
        "bbox": [75.30, 30.55, 75.36, 30.60],  # [min_lon, min_lat, max_lon, max_lat]
        "center": [75.33, 30.575],             # [lon, lat]
        "sensor": "SPOT 6/7 (1.5m Optical)",
        "source": "WorldStrat (Paired Sentinel-2 L2A <-> SPOT 1.5m)"
    },
    "delhi_ncr": {
        "aoi_id": "delhi_ncr",
        "aoi_name": "Delhi NCR Urban Infrastructure",
        "file_name": "delhi_ncr_spot_1.5m.npy",
        "relative_path": "data/reference_hr/delhi_ncr_spot_1.5m.npy",
        "bbox": [77.15, 28.55, 77.22, 28.61],
        "center": [77.185, 28.58],
        "sensor": "SPOT 6/7 (1.5m Optical)",
        "source": "WorldStrat (Paired Sentinel-2 L2A <-> SPOT 1.5m)"
    },
    "varanasi_river": {
        "aoi_id": "varanasi_river",
        "aoi_name": "Varanasi River Corridor",
        "file_name": "varanasi_river_spot_1.5m.npy",
        "relative_path": "data/reference_hr/varanasi_river_spot_1.5m.npy",
        "bbox": [82.95, 25.30, 83.00, 25.34],
        "center": [82.975, 25.32],
        "sensor": "SPOT 6/7 (1.5m Optical)",
        "source": "WorldStrat (Paired Sentinel-2 L2A <-> SPOT 1.5m)"
    }
}

def calculate_bbox_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Calculates Intersection-over-Union (IoU) between two bounding boxes [min_lon, min_lat, max_lon, max_lat]."""
    inter_min_lon = max(bbox1[0], bbox2[0])
    inter_min_lat = max(bbox1[1], bbox2[1])
    inter_max_lon = min(bbox1[2], bbox2[2])
    inter_max_lat = min(bbox1[3], bbox2[3])

    if inter_min_lon >= inter_max_lon or inter_min_lat >= inter_max_lat:
        return 0.0

    inter_area = (inter_max_lon - inter_min_lon) * (inter_max_lat - inter_min_lat)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def resolve_reference_for_aoi(aoi_id: str | None = None, bbox: list[float] | None = None) -> ResolvedReference | None:
    """
    Resolves genuine high-resolution ground truth reference (Tier 1: SPOT 1.5m) for an AOI.
    Strictly verifies coordinates and spatial consistency.
    Returns None if no paired scientific reference exists. NEVER falls back to Punjab or any other scene.
    """
    # 1. Direct lookup by registered AOI preset ID
    if aoi_id and aoi_id in PREPARED_HR_REFERENCES:
        ref_candidate = PREPARED_HR_REFERENCES[aoi_id]
        
        # Verify coordinates if bbox is provided
        if bbox and len(bbox) == 4:
            c_lon = (bbox[0] + bbox[2]) / 2.0
            c_lat = (bbox[1] + bbox[3]) / 2.0
            ref_lon, ref_lat = ref_candidate["center"]
            dist_deg = math.sqrt((c_lon - ref_lon) ** 2 + (c_lat - ref_lat) ** 2)
            
            # If distance exceeds 0.25 degrees (~28 km), coordinates contradict the requested AOI preset!
            if dist_deg > 0.25:
                print(
                    f"[ReferenceManager] WARNING: aoi_id '{aoi_id}' specified but coordinates [{c_lon:.3f}, {c_lat:.3f}] "
                    + f"are {dist_deg:.2f} deg away from reference center [{ref_lon:.3f}, {ref_lat:.3f}]. "
                    + "Refusing mismatched reference assignment."
                )
                return None

        # Verify reference file exists on local disk
        full_path = BASE_DIR / ref_candidate["relative_path"]
        if full_path.exists():
            resolved: ResolvedReference = {
                "aoi_id": ref_candidate["aoi_id"],
                "aoi_name": ref_candidate["aoi_name"],
                "file_name": ref_candidate["file_name"],
                "relative_path": ref_candidate["relative_path"],
                "bbox": ref_candidate["bbox"],
                "center": ref_candidate["center"],
                "sensor": ref_candidate["sensor"],
                "source": ref_candidate["source"],
                "full_path": str(full_path),
                "has_reference": True,
                "is_scientific_ground_truth": True,
                "tier": 1,
                "tier_label": "Ground Truth HR (1.5m)"
            }
            return resolved
        else:
            print(f"[ReferenceManager] WARNING: Registered reference file not found on disk: {full_path}")
            return None

    # 2. Coordinate-based matching for custom/drawn AOIs against known Tier-1 scenes
    if bbox and len(bbox) == 4:
        c_lon = (bbox[0] + bbox[2]) / 2.0
        c_lat = (bbox[1] + bbox[3]) / 2.0

        for candidate_id, candidate in PREPARED_HR_REFERENCES.items():
            ref_lon, ref_lat = candidate["center"]
            dist_deg = math.sqrt((c_lon - ref_lon) ** 2 + (c_lat - ref_lat) ** 2)
            iou = calculate_bbox_iou(bbox, candidate["bbox"])

            # High spatial overlap with prepared reference scene
            if dist_deg < 0.05 or iou > 0.4:
                full_path = BASE_DIR / candidate["relative_path"]
                if full_path.exists():
                    print(f"[ReferenceManager] Spatial match: Bbox [{c_lon:.3f}, {c_lat:.3f}] matched to reference '{candidate_id}' (dist={dist_deg:.3f} deg, IoU={iou:.2f})")
                    spatial_match: ResolvedReference = {
                        "aoi_id": candidate["aoi_id"],
                        "aoi_name": candidate["aoi_name"],
                        "file_name": candidate["file_name"],
                        "relative_path": candidate["relative_path"],
                        "bbox": candidate["bbox"],
                        "center": candidate["center"],
                        "sensor": candidate["sensor"],
                        "source": candidate["source"],
                        "full_path": str(full_path),
                        "has_reference": True,
                        "is_scientific_ground_truth": True,
                        "tier": 1,
                        "tier_label": "Ground Truth HR (1.5m)"
                    }
                    return spatial_match

    # 3. No match found -> Return None
    return None

def determine_reference_tier(aoi_id: str | None = None, bbox: list[float] | None = None) -> TierInfo:
    """
    Decides which reference tier to use for an AOI:
    - Tier 1 (Paired Scientific Ground Truth): Genuine high-resolution paired satellite
      imagery (e.g., SPOT 6/7 1.5m optical via WorldStrat). Strictly verified against
      AOI coordinates with high spatial overlap. Evaluated with full-reference metrics (PSNR, SSIM, SAM, ERGAS).
    - Tier 2 (Regional / Aerial Orthophoto): Reserved for regional/national orthophoto archives.
    - Tier 3 (Global Reference Basemap - Fallback): Automatically triggered when no
      scientific reference exists for the active AOI (e.g. custom drawn AOIs, Haldwani).
      Fetches Esri WorldImagery via leafmap.map_tiles_to_geotiff(zoom=16) and caches to disk.
      Explicitly labeled as 'Global Reference Basemap (variable resolution)' and NEVER
      mislabeled as 'Ground Truth HR (1.5m)'. Evaluated under No-Reference mode (pyiqa NIQE/BRISQUE).
    """
    # Check Tier 1: Paired scientific ground truth reference on disk
    scientific_ref = resolve_reference_for_aoi(aoi_id=aoi_id, bbox=bbox)
    if scientific_ref is not None:
        tier1_info: TierInfo = {
            "tier": 1,
            "tier_id": "tier_1_scientific",
            "tier_label": "Ground Truth HR (1.5m)",
            "tier_name": "Tier-1: Paired Scientific Ground Truth",
            "is_scientific_ground_truth": True,
            "assessment_mode": "paired",
            "source": str(scientific_ref.get("source", "WorldStrat (Paired Sentinel-2 L2A <-> SPOT 1.5m)")),
            "sensor": str(scientific_ref.get("sensor", "SPOT 6/7 (1.5m Optical)")),
            "file_name": str(scientific_ref.get("file_name", "")),
            "relative_path": str(scientific_ref.get("relative_path", "")),
            "full_path": str(scientific_ref.get("full_path", "")),
            "bbox": scientific_ref.get("bbox", bbox),
            "has_reference": True
        }
        return tier1_info

    # Tier 3 fallback: Global Reference Basemap (Esri.WorldImagery via leafmap)
    cache_dir = BASE_DIR / "cache" / "reference_basemaps"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if bbox and len(bbox) == 4:
        bbox_clean = [round(float(c), 4) for c in bbox]
        bbox_hash = hashlib.md5(f"{bbox_clean}".encode("utf-8")).hexdigest()[:8]
    else:
        bbox_hash = "default"

    safe_aoi = "".join(c if c.isalnum() else "_" for c in (aoi_id or "custom"))
    geotiff_name = f"{safe_aoi}_{bbox_hash}_esri_zoom16.tif"
    geotiff_path = cache_dir / geotiff_name

    tier3_info: TierInfo = {
        "tier": 3,
        "tier_id": "tier_3_basemap",
        "tier_label": "Global Reference Basemap (variable resolution)",
        "tier_name": "Tier-3: Global Reference Basemap (Fallback)",
        "is_scientific_ground_truth": False,
        "assessment_mode": "no_reference",
        "source": "Esri.WorldImagery (via leafmap tile-fetch)",
        "sensor": "Global Basemap Imagery (variable resolution)",
        "file_name": geotiff_name,
        "relative_path": f"cache/reference_basemaps/{geotiff_name}",
        "full_path": str(geotiff_path),
        "bbox": bbox,
        "has_reference": True
    }
    return tier3_info

def fetch_tier3_leafmap_basemap(bbox: list[float], output_path: Path) -> np.ndarray | None:
    """
    Fetches Tier-3 fallback reference GeoTIFF using leafmap.map_tiles_to_geotiff.
    Caches the resulting GeoTIFF to disk per AOI so repeated requests do not re-download.
    Returns normalized float32 RGB array [0.0, 1.0].
    """
    try:
        import leafmap  # type: ignore[import-untyped, import-not-found]
        map_tiles_fn = getattr(leafmap, "map_tiles_to_geotiff", None)
        if map_tiles_fn is None:
            print("[Tier-3 Basemap] leafmap.map_tiles_to_geotiff not found.")
            return None
    except ImportError:
        print("[Tier-3 Basemap] leafmap package not installed.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 1024:
        print(f"[Tier-3 Basemap] Cache hit: loading existing GeoTIFF from {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"[Tier-3 Basemap] Cache miss: fetching Esri.WorldImagery via leafmap.map_tiles_to_geotiff for bbox={bbox} (zoom=16)...")
        t0 = time.time()
        try:
            map_tiles_fn(
                output=str(output_path),
                bbox=bbox,
                zoom=16,
                source="Esri.WorldImagery",
                quiet=True
            )
            print(f"[Tier-3 Basemap] Successfully downloaded and cached GeoTIFF in {time.time() - t0:.2f}s ({output_path.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            print(f"[Tier-3 Basemap] ERROR fetching basemap via leafmap: {e}")
            return None

    try:
        pil_img = PILImage.open(output_path).convert("RGB")
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        # Automatically clean up custom drawn AOI files so they never litter the project directory
        if "custom" in output_path.name.lower() and output_path.exists():
            try:
                output_path.unlink(missing_ok=True)
            except OSError as _e:
                print(f"[RefMgr] WARN: could not remove custom AOI temp file {output_path}: {_e}")
        return arr
    except Exception as e:
        print(f"[Tier-3 Basemap] ERROR reading cached GeoTIFF {output_path}: {e}")
        return None

def load_reference_for_session(
    aoi_id: str | None,
    bbox: list[float] | None,
    target_shape: tuple[int, int, int] | None = None
) -> tuple[np.ndarray | None, str | None, TierInfo | None]:
    """
    Loads reference imagery according to the determined reference tier:
    - Tier 1: Loads paired scientific SPOT 1.5m reference array from .npy file.
    - Tier 3: Fetches/loads cached Tier-3 Esri.WorldImagery GeoTIFF via leafmap.
    
    Returns: (ref_array, ref_provenance_str, ref_metadata_dict)
    If no reference can be obtained, returns (None, None, None).
    """
    tier_info = determine_reference_tier(aoi_id=aoi_id, bbox=bbox)

    if tier_info.get("tier") == 1:
        # Load Tier-1 Scientific SPOT Reference
        full_path_str = str(tier_info.get("full_path", ""))
        file_path = Path(full_path_str)
        if not file_path.exists():
            print(f"[Validation Reference] Tier-1 file missing: {file_path}")
            return None, None, None

        ref_raw = np.load(file_path)
        ref_hr: np.ndarray = np.asarray(ref_raw, dtype=np.float32)
        if float(np.max(ref_hr)) > 1.0:
            ref_hr = ref_hr / 255.0

        if target_shape and ref_hr.shape != target_shape:
            ref_clipped = np.clip(ref_hr, 0.0, 1.0)
            ref_uint8 = np.asarray(ref_clipped * 255.0, dtype=np.uint8)
            target_w = int(target_shape[1])
            target_h = int(target_shape[0])
            pil_ref = PILImage.fromarray(ref_uint8).resize((target_w, target_h), PILImage.Resampling.LANCZOS)
            ref_hr = np.asarray(pil_ref, dtype=np.float32) / 255.0

        provenance_str = f"[Tier-1 Scientific] {tier_info.get('relative_path', '')} ({tier_info.get('sensor', '')} - {tier_info.get('source', '')})"
        print(f"[Validation Reference] Genuinely independent Tier-1 HR reference loaded from: {provenance_str}")
        print(f"[Validation Reference] Shape: {ref_hr.shape}, Reflectance Range: [{float(np.min(ref_hr)):.4f}, {float(np.max(ref_hr)):.4f}]")
        return ref_hr, provenance_str, tier_info

    elif tier_info.get("tier") == 3:
        # Load Tier-3 Fallback Global Reference Basemap via leafmap
        if not bbox or len(bbox) != 4:
            print("[Validation Reference] Cannot fetch Tier-3 basemap: no valid bbox provided.")
            return None, None, None

        full_path_str = str(tier_info.get("full_path", ""))
        ref_hr = fetch_tier3_leafmap_basemap(bbox=bbox, output_path=Path(full_path_str))
        if ref_hr is None:
            return None, None, None

        if target_shape and ref_hr.shape != target_shape:
            ref_clipped = np.clip(ref_hr, 0.0, 1.0)
            ref_uint8 = np.asarray(ref_clipped * 255.0, dtype=np.uint8)
            target_w = int(target_shape[1])
            target_h = int(target_shape[0])
            pil_ref = PILImage.fromarray(ref_uint8).resize((target_w, target_h), PILImage.Resampling.LANCZOS)
            ref_hr = np.asarray(pil_ref, dtype=np.float32) / 255.0

        provenance_str = f"[Tier-3 Fallback] Global Reference Basemap (variable resolution) — {tier_info.get('source', '')} (cached: {tier_info.get('file_name', '')})"
        print(f"[Validation Reference] Tier-3 Global Reference Basemap loaded: {provenance_str}")
        print(f"[Validation Reference] Shape: {ref_hr.shape}, Reflectance Range: [{float(np.min(ref_hr)):.4f}, {float(np.max(ref_hr)):.4f}]")
        return ref_hr, provenance_str, tier_info

    return None, None, None

def assert_reference_scene_alignment(ref_info: TierInfo | ResolvedReference | None, active_aoi_id: str | None, active_bbox: list[float] | None) -> None:
    """
    Asserts that the loaded reference corresponds strictly to the active scene before computing metrics.
    Fails loudly with AssertionError if there is any mismatch between the reference and the active scene.
    """
    if ref_info is None:
        raise ValueError(
            f"Cannot compute validation metrics: No ground truth reference is available for AOI '{active_aoi_id}'. "
            + "Validation metrics (PSNR/SSIM/ERGAS) require a paired high-resolution reference tile."
        )

    # Tier-3 basemap check: Cannot compute paired full-reference metrics against uncalibrated basemap
    if not ref_info.get("is_scientific_ground_truth", False) or ref_info.get("tier") == 3:
        raise ValueError(
            "Cannot compute paired full-reference metrics (PSNR/SSIM) against Tier-3 Global Reference Basemap. "
            + "Tier-3 basemap is an uncalibrated reference with variable resolution. "
            + "Must evaluate using No-Reference Quality Assessment (pyiqa NIQE/BRISQUE)."
        )

    expected_ref = resolve_reference_for_aoi(aoi_id=active_aoi_id, bbox=active_bbox)
    if expected_ref is None:
        raise AssertionError(
            f"SCENE MISMATCH FATAL ERROR: Active AOI is '{active_aoi_id}' (bbox={active_bbox}), "
            + f"which has NO valid ground truth reference, but a reference '{ref_info.get('file_name')}' was present! "
            + "Refusing to compare mismatched scenes."
        )

    if expected_ref.get("file_name") != ref_info.get("file_name"):
        raise AssertionError(
            f"SCENE MISMATCH FATAL ERROR: Active scene '{active_aoi_id}' expects reference '{expected_ref.get('file_name')}', "
            + f"but loaded reference is '{ref_info.get('file_name')}'! "
            + "Refusing to compare mismatched scenes."
        )

    print(f"[Validation Reference] Scene Alignment Verified: Reference '{ref_info.get('file_name')}' matches active AOI '{active_aoi_id}'")


def get_or_fetch_esri_basemap(
    bbox: list[float] | None,
    aoi_id: str | None = None,
    target_shape: tuple[int, int] | None = None
) -> np.ndarray | None:
    """
    Fetches or loads cached Esri.WorldImagery basemap for the exact given AOI and bounding box.
    Used for dynamic per-AOI Reinhard color matching.
    Returns RGB float32 array in [0.0, 1.0] resized to target_shape if provided.
    """
    import cv2
    cache_dir = BASE_DIR / "cache" / "reference_basemaps"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve bbox if not directly provided
    resolved_bbox = bbox
    if not resolved_bbox and aoi_id and aoi_id in PREPARED_HR_REFERENCES:
        resolved_bbox = PREPARED_HR_REFERENCES[aoi_id]["bbox"]

    # 2. Known cached files check for instantaneous loading
    candidate_file = None
    if aoi_id == "punjab_agri":
        for fname in ["real_punjab_esri.tif", "punjab_agri_90aadc2b_esri_zoom16.tif"]:
            if (cache_dir / fname).exists():
                candidate_file = cache_dir / fname
                break
    elif aoi_id == "delhi_ncr":
        if (cache_dir / "delhi_ncr_esri_zoom16.tif").exists():
            candidate_file = cache_dir / "delhi_ncr_esri_zoom16.tif"
    elif aoi_id == "varanasi_river":
        if (cache_dir / "varanasi_river_esri_zoom16.tif").exists():
            candidate_file = cache_dir / "varanasi_river_esri_zoom16.tif"
    elif aoi_id and "haldwani" in aoi_id.lower():
        if (cache_dir / "custom_haldwani_b745bce6_esri_zoom16.tif").exists():
            candidate_file = cache_dir / "custom_haldwani_b745bce6_esri_zoom16.tif"
        elif (BASE_DIR / "cache" / "exact_haldwani_esri_basemap.png").exists():
            candidate_file = BASE_DIR / "cache" / "exact_haldwani_esri_basemap.png"

    # 3. Hash-based check
    if candidate_file is None:
        if resolved_bbox and len(resolved_bbox) == 4:
            bbox_clean = [round(float(c), 4) for c in resolved_bbox]
            bbox_hash = hashlib.md5(f"{bbox_clean}".encode("utf-8")).hexdigest()[:8]
        else:
            bbox_hash = "default"
        safe_aoi = "".join(c if c.isalnum() else "_" for c in (aoi_id or "custom"))
        geotiff_name = f"{safe_aoi}_{bbox_hash}_esri_zoom16.tif"
        geotiff_path = cache_dir / geotiff_name
        if geotiff_path.exists():
            candidate_file = geotiff_path
        elif resolved_bbox and len(resolved_bbox) == 4:
            # Fallback to fetching live via leafmap
            arr = fetch_tier3_leafmap_basemap(bbox=resolved_bbox, output_path=geotiff_path)
            if arr is not None:
                candidate_file = geotiff_path

    if candidate_file is None or not candidate_file.exists():
        return None

    try:
        pil_img = PILImage.open(candidate_file).convert("RGB")
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0

        # Center crop to square if aspect ratio is non-square and target is square
        if target_shape and target_shape[0] == target_shape[1]:
            h, w = arr.shape[:2]
            if h != w:
                dim = min(h, w)
                sy = (h - dim) // 2
                sx = (w - dim) // 2
                arr = arr[sy:sy+dim, sx:sx+dim]

        if target_shape and arr.shape[:2] != target_shape[:2]:
            interp = cv2.INTER_AREA if (arr.shape[0] > target_shape[0]) else cv2.INTER_LINEAR
            arr = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=interp)
        return np.clip(arr, 0.0, 1.0)
    except Exception as e:
        print(f"[ReferenceManager] Error loading Esri basemap from {candidate_file}: {e}")
        return None

