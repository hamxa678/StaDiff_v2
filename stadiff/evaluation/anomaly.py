"""
Anomaly maps and scores.

[StaDiff-V3 ≠ DDPM] (relative to the generative model) the score is restricted to the ROI and
can be computed per rod: an image is anomalous if ANY rod is (default image_score="max_rod").
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    r = int(math.ceil(3 * sigma))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    C = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def ssim_map(a: torch.Tensor, b: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """Per-pixel SSIM of two images in [0, 1]. (B,1,H,W) -> (B,1,H,W)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = gaussian_blur(a, sigma), gaussian_blur(b, sigma)
    var_a = gaussian_blur(a * a, sigma) - mu_a ** 2
    var_b = gaussian_blur(b * b, sigma) - mu_b ** 2
    cov = gaussian_blur(a * b, sigma) - mu_a * mu_b
    return ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2))


def anomaly_map(x0: torch.Tensor, x_hat: torch.Tensor, kind="l1", smooth_sigma=2.0) -> torch.Tensor:
    """x0, x_hat in [-1, 1] -> anomaly map (B,1,H,W), larger = more anomalous."""
    a, b = (x0 + 1) / 2, (x_hat + 1) / 2
    if kind == "l1":
        m = (a - b).abs()
    elif kind == "ssim":
        m = (1 - ssim_map(a, b)) / 2
    elif kind == "l1+ssim":
        m = 0.5 * (a - b).abs() + 0.5 * (1 - ssim_map(a, b)) / 2
    else:
        raise ValueError(f"unknown anomaly map kind: {kind}")
    return gaussian_blur(m, smooth_sigma)


def region_score(amap: torch.Tensor, region: torch.Tensor, reduce="topk", topk_frac=0.02) -> torch.Tensor:
    """amap (B,1,H,W), region (1 or B,1,H,W) in {0,1} -> (B,)"""
    region = region.expand_as(amap) > 0.5
    out = []
    for m, r in zip(amap, region):
        v = m[r]
        if v.numel() == 0:
            out.append(torch.zeros((), device=amap.device))
        elif reduce == "mean":
            out.append(v.mean())
        elif reduce == "max":
            out.append(v.max())
        elif reduce == "topk":
            k = max(1, int(round(topk_frac * v.numel())))
            out.append(v.topk(k).values.mean())
        else:
            raise ValueError(f"unknown reduce: {reduce}")
    return torch.stack(out)


def score_batch(amap, M, rod_masks, reduce="topk", topk_frac=0.02, image_score="max_rod"):
    """-> (image scores (B,), per-rod scores (B, n_rods))."""
    rods = torch.stack([region_score(amap, rm[None, None], reduce, topk_frac) for rm in rod_masks], 1)
    if image_score == "max_rod":
        img = rods.max(dim=1).values
    elif image_score == "roi":
        img = region_score(amap, M, reduce, topk_frac)
    else:
        raise ValueError(f"unknown image_score: {image_score}")
    return img, rods
