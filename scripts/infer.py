"""
Step 5 — check new images with a trained model.

    python scripts/infer.py --checkpoint runs/g800_400/ckpt_best.pt --images new_photos/ \
        --metrics runs/g800_400/eval_ckpt_best_troi800_tbg400/metrics.json
    python scripts/infer.py --checkpoint runs/g800_400/ckpt_best.pt --images one.jpg --threshold 0.12 --t-roi 600

Images can be raw camera images (any size): they are prepared exactly like the training data
(grayscale, resized with data.prepare_method to data.image_size, normalised). The decision
threshold is read from the metrics.json written by evaluate.py, or given with --threshold.
With --metrics, the evaluation settings stored there (t_roi, map, sampler, ...) are reused.
Writes results.csv and one visualisation per image (input | reconstruction | anomaly map).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stadiff.data.files import IMG_EXTS                                    # noqa: E402
from stadiff.data.masks import get_masks                                   # noqa: E402
from stadiff.data.preprocess import load_image                             # noqa: E402
from stadiff.diffusion import RegionalDiffusion                            # noqa: E402
from stadiff.evaluation.runner import reconstruct_and_score                # noqa: E402
from stadiff.utils import heat, pick_device, save_triplets, to_uint8       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EVAL_FLAGS, load_model_from_checkpoint                # noqa: E402
from stadiff.cli import add_config_flags, flag_values                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True, help="an image file or a folder of images")
    ap.add_argument("--metrics", default=None, help="metrics.json from evaluate.py (for the threshold)")
    ap.add_argument("--threshold", type=float, default=None, help="decision threshold (overrides --metrics)")
    ap.add_argument("--out", default="runs/infer")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="any other config value")
    add_config_flags(ap, EVAL_FLAGS)
    a = ap.parse_args()

    device = pick_device(a.device)
    model, cfg, _ = load_model_from_checkpoint(a.checkpoint, a, device)
    diffusion = RegionalDiffusion(cfg.diffusion, device)
    thr = a.threshold
    if a.metrics:
        # Reuse the evaluation settings that produced the threshold (unless overridden by a flag),
        # otherwise scores and threshold would not be comparable.
        m = json.loads(Path(a.metrics).read_text())
        given = flag_values(a)
        for k, v in m["eval_config"].items():
            if f"eval.{k}" not in given:
                cfg.eval[k] = v
        for k in ("t_roi_max", "t_bg"):
            if f"eval.{k}" not in given:
                cfg.eval[k] = m[k]
        if thr is None:
            thr = m["threshold"]
        print(f"using evaluation settings and threshold {thr:.5f} from {a.metrics}")
    if thr is None:
        print("no threshold given: reporting scores only")

    src = Path(a.images)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    M, rods = get_masks(cfg.data)
    M_t = torch.as_tensor(M, device=device)[None, None]
    rods_t = torch.as_tensor(rods, device=device)
    out = Path(a.out)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg.eval.seed))

    rows = []
    bs = max(1, int(cfg.eval.batch_size))
    e = cfg.eval
    calls = (e.steps_stage1 + e.steps_stage2) if e.sampler == "ddim" else None
    print(f"{len(files)} images, batches of {bs}, sampler {e.sampler}"
          + (f" ({calls} network passes per batch" if calls else " (one network pass per timestep")
          + f" x {e.n_samples} sample(s)), half precision {bool(e.amp) and device.type == 'cuda'}")
    t_start = time.time()
    for i in range(0, len(files), bs):
        chunk = files[i:i + bs]
        x0 = torch.stack([torch.from_numpy(load_image(f, cfg.data.image_size, cfg.data.prepare_method,
                                                      cfg.data.norm))[None] for f in chunk]).to(device)
        x_hat, amap, img, rod = reconstruct_and_score(model, diffusion, x0, M_t, rods_t, cfg)
        for j, f in enumerate(chunk):
            score = float(img[j])
            verdict = "" if thr is None else ("ANOMALOUS" if score >= thr else "normal")
            a_np = amap[j, 0].float().cpu().numpy()
            vmax = max(thr * 2, 1e-6) if thr else float(np.percentile(a_np, 99.5))
            save_triplets([(to_uint8(x0[j]), to_uint8(x_hat[j]), heat(a_np, vmax), M)], out / "maps" / f"{f.stem}.png")
            rods_list = [float(v) for v in rod[j]]
            worst = int(np.argmax(rods_list)) + 1
            rows.append([str(f), f"{score:.6f}", verdict, worst] + [f"{v:.6f}" for v in rods_list])
            print(f"{f.name:40s} score {score:.5f} {verdict:10s} (highest rod: {worst})")
        done = i + len(chunk)
        el = time.time() - t_start
        print(f"  [{done}/{len(files)}]  {el / done:.2f} s per image, about {el / done * (len(files) - done):.0f} s left",
              flush=True)

    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "score", "verdict", "worst_rod"] + [f"rod{k + 1}" for k in range(len(rods))])
        w.writerows(rows)
    print(f"\nresults in {out}")


if __name__ == "__main__":
    main()
