"""Reconstruct every image of a loader and score it. Shared by train.py (previews), evaluate.py
and infer.py."""
from __future__ import annotations

import numpy as np
import torch

from ..diffusion.sampler import reconstruct
from .anomaly import anomaly_map, score_batch


def resolve_levels(cfg):
    """Inference noise levels: eval.* if set, else the training values."""
    t_roi_max = cfg.eval.t_roi_max if cfg.eval.t_roi_max is not None else cfg.diffusion.t_roi_max
    t_bg = cfg.eval.t_bg if cfg.eval.t_bg is not None else cfg.diffusion.t_bg
    return int(t_roi_max), int(t_bg)


@torch.no_grad()
def reconstruct_and_score(model, diffusion, x0, M_t, rods_t, cfg):
    """x0 (B,1,H,W) on device -> (x_hat, anomaly map, image scores (B,), rod scores (B, n))."""
    e = cfg.eval
    t_roi_max, t_bg = resolve_levels(cfg)
    Mb = M_t.expand(x0.shape[0], -1, -1, -1)
    recons = [reconstruct(model, diffusion, x0, Mb, t_roi_max, t_bg, e.sampler,
                          e.steps_stage1, e.steps_stage2) for _ in range(max(1, int(e.n_samples)))]
    x_hat = torch.stack(recons).mean(0)       # averaging several reconstructions reduces noise
    amap = anomaly_map(x0, x_hat, e.map, e.smooth_sigma)
    img, rods = score_batch(amap, M_t, rods_t, e.reduce, e.topk_frac, e.image_score)
    return x_hat, amap, img, rods


@torch.no_grad()
def score_loader(model, diffusion, loader, M, rod_masks, cfg, device, max_vis=0, desc="scoring"):
    """-> (records list of dicts, vis list of (x0 uint8, x_hat uint8, amap float32, label, path))."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda it, **k: it
    model.eval()
    torch.manual_seed(int(cfg.eval.seed))          # same forward noise every run -> comparable
    M_t = torch.as_tensor(M, dtype=torch.float32, device=device)[None, None]
    rods_t = torch.as_tensor(rod_masks, dtype=torch.float32, device=device)
    records, vis = [], []
    per_label_vis = {0: 0, 1: 0}
    for batch in tqdm(loader, desc=desc):
        x0 = batch["x0"].to(device)
        x_hat, amap, img, rods = reconstruct_and_score(model, diffusion, x0, M_t, rods_t, cfg)
        for i in range(x0.shape[0]):
            label = int(batch["label"][i])
            records.append({"path": batch["path"][i], "label": label, "group": batch["group"][i],
                            "score": float(img[i]), "rod_scores": [float(v) for v in rods[i]]})
            if per_label_vis[label] < max_vis // 2:
                per_label_vis[label] += 1
                to8 = lambda t: ((t.squeeze().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
                vis.append((to8(x0[i]), to8(x_hat[i]), amap[i].squeeze().float().cpu().numpy(),
                            label, batch["path"][i]))
    return records, vis


def threshold_from_clean(scores, percentile: float) -> float:
    """Unsupervised threshold: the given percentile of clean validation scores."""
    return float(np.percentile(np.asarray(scores, dtype=np.float64), percentile))
