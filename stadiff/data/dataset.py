"""
Dataset handling.

Expected layout (after scripts/prepare_data.py, same structure as your DATASET):
    <root>/Clean/Train/*      normal images      -> train (+ a small validation split)
    <root>/Clean/Test/*       normal images      -> test, label 0
    <root>/Anomalous/*        anomalous images   -> test, label 1

Training uses ONLY clean images: the model learns what a normal weld looks like.
The validation split (held out from Clean/Train) is used for the validation loss
and to calibrate the anomaly threshold without ever looking at anomalous images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .files import load_manifest, split_files
from .masks import get_masks
from .preprocess import load_image

class WeldDataset(Dataset):
    """Each item: dict(x0 (1,H,W) in [-1,1], mask (1,H,W), label int, path str, group str)."""

    def __init__(self, items, root, mask: np.ndarray, size_hw, method="area", norm="percentile",
                 augment=False):
        self.items = items
        self.root = Path(root)
        self.mask = torch.from_numpy(mask.astype(np.float32))[None]
        self.size, self.method, self.norm, self.augment = tuple(size_hw), method, norm, augment
        self.manifest = load_manifest(root)

    def __len__(self):
        return len(self.items)

    def group_of(self, path: Path) -> str:
        rel = path.relative_to(self.root).as_posix()
        return self.manifest.get(rel, {}).get("setup", "unknown")

    def __getitem__(self, i):
        path, label = self.items[i]
        x = load_image(path, self.size, self.method, self.norm)
        if self.augment:
            # Photometric jitter only. No flips / shifts / rotations: the ROI mask is a fixed
            # template, and a moved rod is exactly what counts as an anomaly.
            # torch RNG (re-seeded per DataLoader worker), not numpy (identical across workers)
            gain = 0.9 + 0.2 * torch.rand(()).item()
            bias = -0.05 + 0.1 * torch.rand(()).item()
            x = np.clip(x * gain + bias, -1, 1)
        return {
            "x0": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None],
            "mask": self.mask,
            "label": int(label),
            "path": str(path),
            "group": self.group_of(path),
        }


def build_datasets(cfg):
    """-> (train_ds, val_ds, test_ds, M (H,W) numpy, rod_masks (n,H,W) numpy)."""
    d = cfg.data
    M, rods = get_masks(d)
    files = split_files(d.root, d.val_fraction, d.split_seed)
    kw = dict(root=d.root, mask=M, size_hw=d.image_size, method=d.prepare_method, norm=d.norm)
    train = WeldDataset(files["train"], augment=d.augment, **kw)
    val = WeldDataset(files["val"], augment=False, **kw)
    test = WeldDataset(files["test"], augment=False, **kw)
    return train, val, test, M, rods


def make_loader(ds, batch_size, shuffle, num_workers=4, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)
