"""
Step 4 — evaluate a trained model on the test set (Clean/Test + Anomalous).

    python scripts/evaluate.py --checkpoint runs/g800_400/ckpt_best.pt
    python scripts/evaluate.py --checkpoint runs/g800_400/ckpt_best.pt --t-roi 600 --n-samples 3 --map l1+ssim
    python scripts/evaluate.py --help     # every available flag

What happens:
  1. The threshold is calibrated WITHOUT anomalies: every clean validation image is
     reconstructed and scored; threshold = eval.threshold_percentile of those scores.
  2. Every test image is reconstructed with the two-stage sampler and scored.
  3. Metrics: AUROC and average precision (threshold-free), precision / recall / F1 at the
     calibrated threshold (the honest operating point), and the best F1 over all thresholds
     (an optimistic upper bound because it peeks at test labels). Also per camera setup.
Outputs (in --out): metrics.json, scores.csv, vis_clean.png, vis_anomalous.png
Settings come from the checkpoint's training config; the flags change the evaluation ones.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stadiff.cli import add_config_flags, config_from_args                     # noqa: E402
from stadiff.config import to_dict                                             # noqa: E402
from stadiff.data.dataset import WeldDataset, build_datasets, make_loader     # noqa: E402
from stadiff.diffusion import RegionalDiffusion                                # noqa: E402
from stadiff.evaluation.metrics import summarize                               # noqa: E402
from stadiff.evaluation.runner import resolve_levels, score_loader, threshold_from_clean  # noqa: E402
from stadiff.models import build_model                                         # noqa: E402
from stadiff.utils import heat, pick_device, save_triplets, torch_load, write_json  # noqa: E402


EVAL_FLAGS = ["eval", "data.root", "data.mask_source", "data.mask_path"]


def load_model_from_checkpoint(path, args, device):
    """Model + config. The config is the TRAINING config stored in the checkpoint, updated with the
    evaluation flags given on the command line."""
    ck = torch_load(path, device)
    cfg = config_from_args(args, base=ck["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["ema"]["model"])      # EMA weights give smoother reconstructions
    model.eval()
    return model, cfg, ck.get("iter")


def save_vis(vis, label, path, M, per_page=8):
    items = [v for v in vis if v[3] == label][:per_page]
    if not items:
        return
    vmax = float(np.percentile(np.concatenate([v[2].ravel() for v in vis]), 99.5))
    save_triplets([(x, r, heat(a, vmax), M) for x, r, a, _, _ in items], path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="ckpt_best.pt / ckpt_last.pt / ckpt_<iter>.pt")
    ap.add_argument("--out", default=None, help="results folder (default: next to the checkpoint)")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="any other config value")
    add_config_flags(ap, EVAL_FLAGS)
    a = ap.parse_args()

    device = pick_device(a.device)
    model, cfg, it = load_model_from_checkpoint(a.checkpoint, a, device)
    diffusion = RegionalDiffusion(cfg.diffusion, device)
    t_roi_max, t_bg = resolve_levels(cfg)
    diffusion.check_inference(t_roi_max, t_bg)
    out = Path(a.out) if a.out else Path(a.checkpoint).parent / f"eval_{Path(a.checkpoint).stem}_troi{t_roi_max}_tbg{t_bg}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint iteration {it}; t_roi_max={t_roi_max}, t_bg={t_bg}, noise={cfg.diffusion.noise_type}, "
          f"sampler={cfg.eval.sampler}; device {device}")

    _, val_ds, test_ds, M, rods = build_datasets(cfg)
    e = cfg.eval
    if len(val_ds) == 0:
        print("WARNING: no validation images (data.val_fraction=0); calibrating on up to 50 training images")
        train_items = build_datasets(cfg)[0].items[:50]
        val_ds = WeldDataset(train_items, cfg.data.root, M, cfg.data.image_size, cfg.data.prepare_method,
                             cfg.data.norm, augment=False)

    # 1. threshold from clean validation images only
    val_rec, _ = score_loader(model, diffusion, make_loader(val_ds, e.batch_size, False, 2), M, rods, cfg,
                              device, desc="calibration (clean val)")
    thr = threshold_from_clean([r["score"] for r in val_rec], e.threshold_percentile)

    # 2. test set
    test_rec, vis = score_loader(model, diffusion, make_loader(test_ds, e.batch_size, False, 2), M, rods,
                                 cfg, device, max_vis=e.save_vis, desc="test")
    labels = np.array([r["label"] for r in test_rec])
    scores = np.array([r["score"] for r in test_rec])

    # 3. metrics, overall and per camera setup
    metrics = {"checkpoint": str(a.checkpoint), "iteration": it, "t_roi_max": t_roi_max, "t_bg": t_bg,
               "eval_config": to_dict(cfg.eval), "noise_type": cfg.diffusion.noise_type,
               "threshold": thr, "n_val_clean": len(val_rec),
               "overall": summarize(labels, scores, thr), "per_group": {}}
    groups = defaultdict(list)
    for i, r in enumerate(test_rec):
        groups[r["group"]].append(i)
    for g, idx in groups.items():
        idx = np.array(idx)
        metrics["per_group"][g] = summarize(labels[idx], scores[idx], thr)
    write_json(metrics, out / "metrics.json")

    with (out / "scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        n_rods = len(test_rec[0]["rod_scores"])
        w.writerow(["path", "label", "group", "score", "predicted_anomalous"] + [f"rod{k + 1}" for k in range(n_rods)])
        for r in test_rec:
            w.writerow([r["path"], r["label"], r["group"], f"{r['score']:.6f}", int(r["score"] >= thr)]
                       + [f"{v:.6f}" for v in r["rod_scores"]])

    save_vis(vis, 0, out / "vis_clean.png", M)
    save_vis(vis, 1, out / "vis_anomalous.png", M)

    o = metrics["overall"]
    c = o["at_calibrated_threshold"]
    print(f"\nAUROC {o['auroc']:.4f}   AP {o['average_precision']:.4f}")
    print(f"at threshold {thr:.5f} ({e.threshold_percentile}th pct of clean val): "
          f"precision {c['precision']:.3f} recall {c['recall']:.3f} F1 {c['f1']:.3f} "
          f"(TP {c['tp']} FP {c['fp']} TN {c['tn']} FN {c['fn']})")
    print(f"best F1 over thresholds (optimistic): {o['best_f1_oracle']['f1']:.3f}")
    for g, m in metrics["per_group"].items():
        print(f"  setup {g:8s}: n={m['n']:4d} anomalous={m['n_anomalous']:4d} AUROC {m['auroc']:.4f}")
    print(f"\nresults in {out}")


if __name__ == "__main__":
    main()
