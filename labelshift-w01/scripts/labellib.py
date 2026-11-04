"""Small shared helpers for the label-shift prep scripts."""
import os

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(1 << 34))
import numpy as np  # noqa: E402


def read_label_tif(path):
    """2D uint8 array from an LZW-tiled label TIFF. tifffile needs imagecodecs for LZW, which the
    fix2277 venv does not carry, so fall back to OpenCV (already a villa dependency)."""
    try:
        import tifffile
        return np.ascontiguousarray(tifffile.imread(path))
    except ValueError as exc:
        if "imagecodecs" not in str(exc):
            raise
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cv2 could not read {path}")
    if img.ndim == 3:
        img = img[..., 0]
    return np.ascontiguousarray(img)
