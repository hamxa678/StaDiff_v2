"""
Image reading, resizing and intensity normalisation, shared by every script so
that training, evaluation and inference see images processed the same way.

Why a separate preparation step (scripts/prepare_data.py)?
Your clean images are 1600x1200 grayscale while the anomalous ones are 256x256
RGB. If the two classes reach the model through different resizing, the model
can separate them by sharpness or compression alone, without looking at any
defect. prepare_data.py pushes EVERY image through the same function below and
reports a sharpness statistic per class so you can check the gap is gone.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

RESIZE_METHODS = ("area", "bicubic", "bilinear_noaa")


def read_gray(path) -> np.ndarray:
    """Any image file -> float32 (h, w) in [0, 255]."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def _bilinear_noaa(a: np.ndarray, H: int, W: int) -> np.ndarray:
    """Plain bilinear sampling without anti-aliasing (what cv2.resize INTER_LINEAR does).
    Sharper than 'area' when shrinking; try it if your anomalous images were made with OpenCV."""
    h, w = a.shape
    ys = np.clip((np.arange(H) + 0.5) * h / H - 0.5, 0, h - 1)
    xs = np.clip((np.arange(W) + 0.5) * w / W - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def resize_gray(a: np.ndarray, size_hw, method: str = "area") -> np.ndarray:
    """float32 (h, w) -> float32 (H, W). The full frame is resized (no cropping), like the
    256x256 anomalous images, so the fractional ROI boxes stay valid."""
    H, W = size_hw
    if a.shape == (H, W):
        return a
    if method == "bilinear_noaa":
        return _bilinear_noaa(a, H, W)
    resample = {"area": Image.BOX, "bicubic": Image.BICUBIC}[method]
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(im.resize((W, H), resample), dtype=np.float32)


def normalize(a: np.ndarray, norm: str = "percentile") -> np.ndarray:
    """[0, 255] -> [-1, 1].

    "percentile": stretch the image's 1st..99th percentile to [-1, 1]. Your data mixes a bright
                  and a dark camera setup; this removes exposure as a nuisance variable.
    "none":       fixed linear scaling.
    """
    if norm == "percentile":
        lo, hi = np.percentile(a, [1, 99])
        return np.clip((a - lo) / max(hi - lo, 1e-3) * 2 - 1, -1, 1).astype(np.float32)
    if norm == "none":
        return (a / 127.5 - 1.0).astype(np.float32)
    raise ValueError(f"unknown norm: {norm}")


def load_image(path, size_hw, method="area", norm="percentile") -> np.ndarray:
    """File -> model-ready float32 (H, W) in [-1, 1]."""
    return normalize(resize_gray(read_gray(path), size_hw, method), norm)


def sharpness(a: np.ndarray) -> float:
    """Mean absolute Laplacian of a [0, 255] image: a simple blur/sharpness indicator."""
    a = a.astype(np.float32)
    lap = 4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float(np.abs(lap).mean())


def guess_setup(a: np.ndarray, threshold: float = 115.0) -> str:
    """Your samples come from a bright and a dark camera setup; mean intensity separates them.
    Used only to report metrics per setup."""
    return "bright" if float(a.mean()) > threshold else "dark"
