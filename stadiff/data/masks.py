"""
ROI masks.

[StaDiff-V3 ≠ DDPM] A standard DDPM has no notion of a region of interest: every
pixel gets the same noise level. StaDiff-V3 splits the image into
    ROI        (M = 1): the three rods together with their welds
    background (M = 0): everything else, used as context
and noises the two regions to different levels.

The mask is ONE FIXED TEMPLATE for every image. It is never segmented from the
image itself: if a rod is bent or shifted, it moves out of its column, and the
model (which only learned straight, evenly spaced rods with one centred weld)
reconstructs the normal rod there. That disagreement is the anomaly signal.
A per-image mask would follow the defect and hide it.

Boxes are fractions of width/height, so they work at any image size.
They were measured on the clean samples (both the bright and the dark camera
setup). Check them on your data with scripts/make_mask.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# ROI = rod + weld: each rod slot over its full visible height.  (x0, y0, x1, y1)
ROD_BOXES = [
    (0.160, 0.02, 0.310, 0.98),
    (0.430, 0.02, 0.570, 0.98),
    (0.697, 0.02, 0.847, 0.98),
]

# Weld beads only (union of both camera setups). Kept for comparison experiments.
WELD_BOXES = [
    (0.165, 0.33, 0.305, 0.88),
    (0.435, 0.33, 0.565, 0.88),
    (0.702, 0.33, 0.842, 0.88),
]


def _box_to_slice(box, H, W):
    x0, y0, x1, y1 = box
    return (slice(int(round(y0 * H)), int(round(y1 * H))),
            slice(int(round(x0 * W)), int(round(x1 * W))))


def box_mask(size_hw, boxes=ROD_BOXES) -> np.ndarray:
    """Union of all boxes -> float32 (H, W) in {0, 1}."""
    H, W = size_hw
    M = np.zeros((H, W), np.float32)
    for b in boxes:
        M[_box_to_slice(b, H, W)] = 1.0
    return M


def per_rod_masks(size_hw, boxes=ROD_BOXES) -> np.ndarray:
    """One mask per rod -> float32 (n_rods, H, W). Used for per-rod anomaly scores."""
    H, W = size_hw
    out = np.zeros((len(boxes), H, W), np.float32)
    for i, b in enumerate(boxes):
        out[i][_box_to_slice(b, H, W)] = 1.0
    return out


def load_mask_png(path, size_hw) -> np.ndarray:
    H, W = size_hw
    m = Image.open(path).convert("L").resize((W, H), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.float32)


def rod_masks_from_union(M: np.ndarray) -> np.ndarray:
    """Split a hand-drawn union mask into connected column groups (left to right)."""
    cols = M.max(axis=0) > 0
    d = np.diff(np.concatenate([[0], cols.astype(int), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = np.zeros((len(starts), *M.shape), np.float32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        out[i][:, s:e] = M[:, s:e]
    return out


def get_masks(data_cfg) -> tuple[np.ndarray, np.ndarray]:
    """-> (M union (H, W), per-rod masks (n, H, W)) according to the config."""
    size = tuple(data_cfg["image_size"])
    src = data_cfg["mask_source"]
    if src == "rods":
        return box_mask(size, ROD_BOXES), per_rod_masks(size, ROD_BOXES)
    if src == "welds":
        return box_mask(size, WELD_BOXES), per_rod_masks(size, WELD_BOXES)
    if src == "file":
        if not data_cfg.get("mask_path"):
            raise ValueError("data.mask_source is 'file' but data.mask_path is empty")
        M = load_mask_png(data_cfg["mask_path"], size)
        return M, rod_masks_from_union(M)
    raise ValueError(f"unknown data.mask_source: {src}")


def overlay(image01: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Grayscale image in [0, 1] + mask -> RGB uint8 with the ROI tinted and outlined in red."""
    g = (np.clip(image01, 0, 1) * 255).astype(np.float32)
    rgb = np.stack([g] * 3, -1)
    inside = M > 0.5
    rgb[inside] = 0.8 * rgb[inside] + 0.2 * np.array([255, 60, 60])
    edge = np.zeros_like(inside)
    edge[1:-1, 1:-1] = inside[1:-1, 1:-1] & ~(inside[:-2, 1:-1] & inside[2:, 1:-1]
                                              & inside[1:-1, :-2] & inside[1:-1, 2:])
    rgb[edge] = [255, 40, 40]
    return rgb.astype(np.uint8)


def save_mask_png(M: np.ndarray, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((M * 255).astype(np.uint8)).save(path)
