"""
Locating the images of the dataset (no torch needed, so preparation scripts stay light).

Expected layout:
    <root>/Clean/Train/*      normal images      -> train (+ a small validation split)
    <root>/Clean/Test/*       normal images      -> test, label 0
    <root>/Anomalous/*        anomalous images   -> test, label 1
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"missing folder: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    return files


def find_dir(parent: Path, name: str) -> Path:
    """Case-insensitive sub-folder lookup (Clean / clean / CLEAN)."""
    for p in parent.iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    raise FileNotFoundError(f"no '{name}' folder in {parent}")


def split_files(root, val_fraction=0.05, seed=0) -> dict[str, list[tuple[Path, int]]]:
    root = Path(root)
    clean, anom = find_dir(root, "Clean"), find_dir(root, "Anomalous")
    train = list_images(find_dir(clean, "Train"))
    rng = random.Random(seed)
    shuffled = train[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(train) * val_fraction)) if val_fraction > 0 else 0
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return {
        "train": [(p, 0) for p in train],
        "val": [(p, 0) for p in val],
        "test": [(p, 0) for p in list_images(find_dir(clean, "Test"))]
                + [(p, 1) for p in list_images(anom)],
    }


def load_manifest(root) -> dict[str, dict]:
    """manifest.csv written by prepare_data.py: rel_path -> row (setup, original size, ...)."""
    f = Path(root) / "manifest.csv"
    if not f.exists():
        return {}
    with f.open() as fh:
        return {row["rel_path"]: row for row in csv.DictReader(fh)}
