"""
Tiling and Patch Reconstruction Module.
Handles slicing large satellite scenes into model-sized overlapping patches (e.g. 128x128)
and reconstructing the high-resolution output with cosine-weighted blending to eliminate seams.
"""
import numpy as np

class Tiler:
    def __init__(self, patch_size: int = 128, stride: int = 96, scale_factor: int = 4):
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = scale_factor

    def extract_patches(self, image: np.ndarray):
        """
        image shape: (H, W, C)
        Returns: list of (patch, y, x, h_patch, w_patch)
        """
        H, W, C = image.shape
        patches = []
        y = 0
        while y < H:
            x = 0
            while x < W:
                y_end = min(y + self.patch_size, H)
                x_end = min(x + self.patch_size, W)
                y_start = max(0, y_end - self.patch_size)
                x_start = max(0, x_end - self.patch_size)

                patch = image[y_start:y_end, x_start:x_end, :]
                patches.append((patch, y_start, x_start, y_end - y_start, x_end - x_start))

                if x_end == W:
                    break
                x += self.stride
            if y_end == H:
                break
            y += self.stride
        return patches

    def reconstruct(self, sr_patches, target_shape):
        """
        sr_patches: list of (sr_patch, y_lr, x_lr, h_lr, w_lr)
        target_shape: (H_sr, W_sr, C)
        Reconstructs with blending weights to eliminate patch boundary artifacts.
        """
        H_sr, W_sr, C = target_shape
        output = np.zeros((H_sr, W_sr, C), dtype=np.float32)
        weight_map = np.zeros((H_sr, W_sr, 1), dtype=np.float32)

        s = self.scale_factor

        for patch, y_lr, x_lr, h_lr, w_lr in sr_patches:
            y_start = y_lr * s
            x_start = x_lr * s
            h_sr = h_lr * s
            w_sr = w_lr * s
            y_end = y_start + h_sr
            x_end = x_start + w_sr

            # 2D Hanning/Cosine window for smooth overlap blending
            wy = np.sin(np.linspace(0, np.pi, h_sr)) ** 2
            wx = np.sin(np.linspace(0, np.pi, w_sr)) ** 2
            window = np.outer(wy, wx)[:, :, np.newaxis]
            window = np.maximum(window, 1e-4)

            output[y_start:y_end, x_start:x_end, :] += patch * window
            weight_map[y_start:y_end, x_start:x_end, :] += window

        weight_map = np.maximum(weight_map, 1e-5)
        reconstructed = output / weight_map
        return np.clip(reconstructed, 0, 1)
