"""
Step 1 — harmonise the dataset.

Every image (clean and anomalous) is converted to grayscale and resized to data.image_size by
the SAME function, then written to a new folder with the same Clean/Train, Clean/Test,
Anomalous structure. A manifest.csv records each image's original size, its camera setup
(bright/dark) and a sharpness value, and a summary is printed so you can check that clean and
anomalous images now look alike apart from the defects.

    python scripts/prepare_data.py --src /path/to/DATASET --dst DATASET_prepared --image-size 256 256 --method area
    python scripts/prepare_data.py --src DATASET --dst DATASET_prepared_noaa --method bilinear_noaa --format jpg

If the sharpness of clean and anomalous images still differs clearly, try another --method
(and/or --format jpg) and pick the one that makes the two classes most similar.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stadiff.cli import add_common_flags, add_config_flags, config_from_args  # noqa: E402
from stadiff.data.files import find_dir, list_images                      # noqa: E402
from stadiff.data.preprocess import (RESIZE_METHODS, guess_setup, read_gray,  # noqa: E402
                                     resize_gray, sharpness)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="original DATASET folder")
    ap.add_argument("--dst", default=None, help="output folder (default: DATASET_prepared)")
    ap.add_argument("--method", choices=RESIZE_METHODS, default=None,
                    help="resize method (default: area). Use the same value as --prepare-method when training.")
    ap.add_argument("--format", choices=["png", "jpg"], default="png", help="output file format")
    ap.add_argument("--jpeg-quality", type=int, default=95, help="quality when --format jpg")
    add_common_flags(ap)
    add_config_flags(ap, ["data.image_size"])
    a = ap.parse_args()

    cfg = config_from_args(a)
    size = tuple(cfg.data.image_size)
    method = a.method or cfg.data.prepare_method
    src = Path(a.src)
    dst = Path(a.dst or cfg.data.root)
    if src.resolve() == dst.resolve():
        sys.exit("--dst must differ from --src (originals are never overwritten)")

    clean = find_dir(src, "Clean")
    groups = [("Clean/Train", find_dir(clean, "Train"), 0),
              ("Clean/Test", find_dir(clean, "Test"), 0),
              ("Anomalous", find_dir(src, "Anomalous"), 1)]

    rows = []
    stats = defaultdict(list)
    for rel_dir, folder, label in groups:
        files = list_images(folder)
        (dst / rel_dir).mkdir(parents=True, exist_ok=True)
        for f in files:
            with Image.open(f) as im:
                orig_w, orig_h, mode = im.width, im.height, im.mode
            a_full = read_gray(f)
            a_small = resize_gray(a_full, size, method)
            out = dst / rel_dir / (f.stem + "." + a.format)
            img = Image.fromarray(np.clip(np.round(a_small), 0, 255).astype(np.uint8))
            if a.format == "jpg":
                img.save(out, quality=a.jpeg_quality)
            else:
                img.save(out)
            setup = guess_setup(a_small)
            sharp = sharpness(np.asarray(Image.open(out).convert("L"), dtype=np.float32))
            rows.append({"rel_path": f"{rel_dir}/{out.name}", "label": label, "source": str(f),
                         "orig_w": orig_w, "orig_h": orig_h, "orig_mode": mode,
                         "setup": setup, "sharpness": round(sharp, 3)})
            stats[(rel_dir, setup)].append(sharp)
        print(f"{rel_dir:12s}: {len(files)} images")

    with (dst / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nresize method: {method}, size {size[0]}x{size[1]}, format {a.format}")
    print(f"{'subset':12s} {'setup':7s} {'n':>5s} {'sharpness mean ± sd':>22s}")
    for (rel_dir, setup), v in sorted(stats.items()):
        print(f"{rel_dir:12s} {setup:7s} {len(v):5d} {np.mean(v):12.2f} ± {np.std(v):.2f}")
    sizes = defaultdict(set)
    for r in rows:
        sizes[r["rel_path"].split("/")[0]].add((r["orig_w"], r["orig_h"], r["orig_mode"]))
    print("\noriginal formats:", {k: sorted(v) for k, v in sizes.items()})
    print("\nCheck: clean and anomalous sharpness should be close within the same setup. A large gap"
          "\nmeans the model could tell the classes apart by image quality instead of defects.")
    print(f"\nwrote {len(rows)} images and manifest.csv to {dst}")


if __name__ == "__main__":
    main()
