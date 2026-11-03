"""
Step 2 — build the ROI mask and check it visually.

Writes  <out>/roi_mask.png  and  <out>/mask_check.png  (the mask outline drawn on clean and
anomalous images from both camera setups). Open mask_check.png: each red column must contain
one rod with its weld in every CLEAN image. If not, edit ROD_BOXES in stadiff/data/masks.py,
or draw your own mask (white = ROI) and use  --mask-source file --mask-path my_mask.png

    python scripts/make_mask.py --data DATASET_prepared --mask-source rods --out runs/mask
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stadiff.cli import add_common_flags, add_config_flags, config_from_args  # noqa: E402
from stadiff.data.files import split_files                     # noqa: E402
from stadiff.data.masks import get_masks, overlay, save_mask_png  # noqa: E402
from stadiff.data.preprocess import load_image                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/mask", help="folder for roi_mask.png and mask_check.png")
    ap.add_argument("--n", type=int, default=8, help="images per class in the check sheet")
    add_common_flags(ap)
    add_config_flags(ap, ["data.root", "data.image_size", "data.mask_source", "data.roi_box", "data.mask_path",
                          "data.prepare_method", "data.norm"])
    a = ap.parse_args()

    cfg = config_from_args(a)
    size = tuple(cfg.data.image_size)
    M, rods = get_masks(cfg.data)
    out = Path(a.out)
    save_mask_png(M, out / "roi_mask.png")

    files = split_files(cfg.data.root, 0.0)
    clean = [p for p, _ in files["train"]]
    anom = [p for p, l in files["test"] if l == 1]
    pick = lambda lst: [lst[i] for i in np.linspace(0, len(lst) - 1, min(a.n, len(lst))).round().astype(int)]
    tiles = []
    for p in pick(clean) + pick(anom):
        x = load_image(p, size, cfg.data.prepare_method, cfg.data.norm)
        tiles.append(overlay((x + 1) / 2, M))
    cols = 4
    H, W = size
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (H + 4), cols * (W + 4), 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * (H + 4):r * (H + 4) + H, c * (W + 4):c * (W + 4) + W] = t
    Image.fromarray(sheet).save(out / "mask_check.png")
    print(f"ROI covers {M.mean():.1%} of the image, {len(rods)} rods")
    print(f"first {len(pick(clean))} tiles: clean, then anomalous")
    print(f"saved {out / 'roi_mask.png'} and {out / 'mask_check.png'}")


if __name__ == "__main__":
    main()
