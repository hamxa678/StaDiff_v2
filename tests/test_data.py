"""Data, masks and preprocessing (no GPU needed)."""
import numpy as np

from stadiff.data.files import split_files
from stadiff.data.masks import ROD_BOXES, box_mask, get_masks, per_rod_masks, rod_masks_from_union
from stadiff.data.preprocess import load_image, normalize, resize_gray


def test_rod_masks_are_disjoint_and_cover_union():
    M = box_mask((256, 256), ROD_BOXES)
    rods = per_rod_masks((256, 256), ROD_BOXES)
    assert rods.shape == (3, 256, 256)
    assert (rods.sum(0) <= 1).all(), "rod boxes overlap"
    assert np.array_equal(rods.max(0), M)
    assert 0.2 < M.mean() < 0.6


def test_union_split_recovers_three_rods():
    M = box_mask((128, 128))
    assert rod_masks_from_union(M).shape[0] == 3


def test_get_masks_from_config(tiny_cfg):
    M, rods = get_masks(tiny_cfg.data)
    assert M.shape == (64, 64) and rods.shape == (3, 64, 64)


def test_resize_and_normalize_ranges():
    a = np.random.default_rng(0).uniform(0, 255, (120, 160)).astype(np.float32)
    for m in ("area", "bicubic", "bilinear_noaa"):
        r = resize_gray(a, (64, 64), m)
        assert r.shape == (64, 64)
    x = normalize(a, "percentile")
    assert x.min() >= -1 and x.max() <= 1


def test_split_files(fake_dataset):
    s = split_files(fake_dataset, val_fraction=0.25, seed=0)
    assert len(s["train"]) == 9 and len(s["val"]) == 3
    assert not set(s["train"]) & set(s["val"])
    assert sum(l for _, l in s["test"]) == 3 and len(s["test"]) == 6
    x = load_image(s["test"][0][0], (64, 64))
    assert x.shape == (64, 64)
