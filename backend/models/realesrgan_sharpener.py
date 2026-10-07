"""
Real-ESRGAN Dedicated Second-Stage Sharpening Module.
Applies pretrained RRDBNet perceptual edge and texture sharpening
strictly as a post-processing pass on the 4x HAT output (outscale=1).
Leaves raw HAT output unperturbed for all scientific metrics (PSNR/SSIM/SAM).
"""
import os
import sys
import types
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from backend.config import BASE_DIR

# Compatibility bridge for basicsr on modern torchvision versions
try:
    import torchvision.transforms.functional as F_t
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        m = types.ModuleType("torchvision.transforms.functional_tensor")
        m.rgb_to_grayscale = F_t.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = m
except Exception as e:
    print(f"[RealESRGANSharpener] Torchvision compat warning: {e}")


class RealESRGANSharpener:
    """
    Dedicated 2nd-stage sharpening processor using RealESRGAN_x4plus (RRDBNet).
    Runs with outscale=1 and tile=256 to ensure zero dimension expansion beyond 4x,
    avoiding memory overflow and distortion while recovering rich micro-textures.
    """

    def __init__(self, weights_path: Optional[str] = None, tile: int = 512):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tile = tile
        self.upsampler = None
        self.weights_path = None
        self.is_ready = False

        self._init_upsampler(weights_path)

    def _init_upsampler(self, weights_path: Optional[str] = None):
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            default_dir = BASE_DIR / "weights"
            default_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = Path(weights_path) if weights_path else default_dir / "RealESRGAN_x4plus.pth"

            if not candidate_path.exists():
                # Check site-packages cache or fallback download
                fallback_cached = Path(sys.prefix) / "weights" / "RealESRGAN_x4plus.pth"
                if fallback_cached.exists():
                    candidate_path = fallback_cached
                else:
                    print(f"[RealESRGANSharpener] Weights not found locally at {candidate_path}. RealESRGANer will download on init.")
                    candidate_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

            self.weights_path = str(candidate_path)

            # Standard RRDBNet architecture for RealESRGAN_x4plus (16.7M params)
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )

            # Use half precision (FP16) on GPU for high throughput; FP32 on CPU
            use_half = (self.device == "cuda")

            # Use tile=self.tile with generous tile_pad=32 for boundary continuity
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=self.weights_path,
                model=model,
                tile=self.tile,
                tile_pad=32,
                pre_pad=0,
                half=use_half
            )
            self.is_ready = True
            print(f"[RealESRGANSharpener] Initialized successfully on {self.device.upper()} (half={use_half}, tile={self.tile})")
            print(f"[RealESRGANSharpener] Checkpoint: {self.weights_path}")
        except Exception as e:
            print(f"[RealESRGANSharpener] Initialization warning: {e}. Fallback post-processing will be used.")
            self.upsampler = None
            self.is_ready = False

    def enhance_preview(self, rgb_image: np.ndarray, outscale: int = 1) -> np.ndarray:
        """
        Enhances perceptual sharpness and micro-textures on 4x HAT output.
        Args:
            rgb_image: (H, W, 3) image array (float32 [0, 1] or uint8 [0, 255]).
            outscale: Output scaling factor relative to input. Strictly 1 to prevent double-upscaling.
        Returns:
            Sharpened (H, W, 3) image array in float32 [0.0, 1.0].
        """
        if not self.is_ready or self.upsampler is None:
            print("[RealESRGANSharpener] Model not ready. Returning unsharpened input.")
            return np.clip(rgb_image, 0.0, 1.0).astype(np.float32)

        try:
            import cv2
            h_orig, w_orig = rgb_image.shape[:2]

            # Convert to uint8 [0, 255] for RealESRGANer
            if rgb_image.dtype in (np.float32, np.float64):
                img_uint8 = np.clip(rgb_image * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                img_uint8 = np.clip(rgb_image, 0, 255).astype(np.uint8)

            # Ensure strictly 3-channel (H, W, 3) format
            if img_uint8.ndim == 2:
                img_uint8 = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
            elif img_uint8.ndim == 3 and img_uint8.shape[2] > 3:
                img_uint8 = img_uint8[:, :, :3]

            # RealESRGANer internally operates on BGR format
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

            # Run enhancement with outscale=1
            sharpened_bgr, _ = self.upsampler.enhance(img_bgr, outscale=outscale)

            # Convert BGR back to RGB
            sharpened_uint8 = cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)

            # Verify dimensions
            if sharpened_uint8.shape[:2] != (h_orig, w_orig):
                print(f"[RealESRGANSharpener] Notice: Resizing {sharpened_uint8.shape[:2]} -> {(h_orig, w_orig)}")
                sharpened_uint8 = cv2.resize(sharpened_uint8, (w_orig, h_orig), interpolation=cv2.INTER_AREA)

            # Convert back to float32 [0.0, 1.0]
            sharpened_float = sharpened_uint8.astype(np.float32) / 255.0
            print(f"[RealESRGANSharpener] Applied 2nd-stage sharpening (outscale={outscale}, shape={sharpened_float.shape})")
            return np.clip(sharpened_float, 0.0, 1.0)

        except Exception as e:
            print(f"[RealESRGANSharpener] Error during enhancement: {e}. Falling back to unsharpened input.")
            return np.clip(rgb_image, 0.0, 1.0).astype(np.float32)


# Global singleton instance for high-throughput reuse
realesrgan_sharpener = RealESRGANSharpener()
