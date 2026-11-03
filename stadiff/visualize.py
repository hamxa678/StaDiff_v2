"""
Image helpers for figures with captions (numpy + Pillow only, no torch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size=13):
    try:
        return ImageFont.load_default(size=size)       # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def gray(a: np.ndarray, lo=-1.0, hi=1.0) -> np.ndarray:
    """(H, W) float in [lo, hi] -> RGB uint8."""
    v = np.clip((np.asarray(a, np.float32) - lo) / (hi - lo), 0, 1)
    g = (v * 255).round().astype(np.uint8)
    return np.stack([g] * 3, -1)


def gray_auto(a: np.ndarray) -> np.ndarray:
    """Noisy image -> RGB, contrast-stretched to its own 1st..99th percentile (display only).
    Noisy states have values far outside [-1, 1]; clipping them would show pure black/white."""
    lo, hi = np.percentile(np.asarray(a, np.float32), [1, 99])
    return gray(a, float(lo), float(max(hi, lo + 1e-6)))


def heat_rgb(a: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Non-negative map -> black-red-yellow-white."""
    a = np.asarray(a, np.float32)
    vmax = float(np.percentile(a, 99.5)) if vmax is None else vmax
    v = np.clip(a / max(vmax, 1e-8), 0, 1)
    rgb = np.stack([np.clip(3 * v, 0, 1), np.clip(3 * v - 1, 0, 1), np.clip(3 * v - 2, 0, 1)], -1)
    return (rgb * 255).astype(np.uint8)


def signed_rgb(a: np.ndarray, vmax: float = 3.0) -> np.ndarray:
    """Signed map (e.g. noise) -> blue (negative) / white (0) / red (positive)."""
    v = np.clip(np.asarray(a, np.float32) / vmax, -1, 1)
    r = np.where(v > 0, 1.0, 1.0 + v)
    b = np.where(v < 0, 1.0, 1.0 - v)
    g = 1.0 - np.abs(v)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def outline(rgb: np.ndarray, M: np.ndarray, color=(255, 60, 60)) -> np.ndarray:
    rgb = rgb.copy()
    inside = M > 0.5
    edge = np.zeros_like(inside)
    edge[1:-1, 1:-1] = inside[1:-1, 1:-1] & ~(inside[:-2, 1:-1] & inside[2:, 1:-1]
                                              & inside[1:-1, :-2] & inside[1:-1, 2:])
    rgb[edge] = color
    return rgb


def labelled_grid(rows, captions, row_titles=None, title=None, pad=4, cap_h=30) -> np.ndarray:
    """rows: list of rows, each a list of HxWx3 uint8 tiles (same size).
    captions: same shape as rows, text drawn above each tile (use \\n for 2 lines).
    row_titles: optional text drawn on a band above each row. title: optional figure title."""
    H, W = rows[0][0].shape[:2]
    ncol = max(len(r) for r in rows)
    font, bold = _font(12), _font(15)
    band = 20 if row_titles else 0
    top = 26 if title else 0
    row_h = band + cap_h + H + pad
    canvas = Image.new("RGB", (ncol * (W + pad) + pad, top + len(rows) * row_h + pad), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((pad, 5), title, fill=(0, 0, 0), font=bold)
    for r, (tiles, caps) in enumerate(zip(rows, captions)):
        y0 = top + r * row_h
        if row_titles:
            d.text((pad, y0 + 3), row_titles[r], fill=(20, 60, 160), font=bold)
        for c, (tile, cap) in enumerate(zip(tiles, caps)):
            x0 = pad + c * (W + pad)
            d.multiline_text((x0, y0 + band + 2), cap, fill=(0, 0, 0), font=font, spacing=1)
            canvas.paste(Image.fromarray(tile), (x0, y0 + band + cap_h))
    return np.asarray(canvas)


def save_png(arr: np.ndarray, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def plot_curves(log_csv, out_png) -> bool:
    """Training curves from log.csv (needs matplotlib; returns False if unavailable)."""
    try:
        import csv

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    with open(log_csv) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return False

    def col(name):
        xs, ys = [], []
        for r in rows:
            if r.get(name) not in (None, "", "nan"):
                xs.append(int(r["iter"]))
                ys.append(float(r[name]))
        return xs, ys

    panels = [
        ("Loss", ["loss", "loss_stage1", "loss_stage2", "val_loss"], True),
        ("Stage 1 loss by ROI noise level (low -> high t)", [k for k in rows[0] if k.startswith("s1_t")], True),
        ("Stage 2 loss by noise level (low -> high t)", [k for k in rows[0] if k.startswith("s2_t")], True),
        ("Stage 2 noise error: ROI vs background pixels", ["s2_err_roi", "s2_err_bg"], True),
        ("Gradient norm / learning rate", ["grad_norm"], True),
        ("Quick eval: mean anomaly score", ["qe_score_clean", "qe_score_anom"], False),
        ("Quick eval: AUROC on a small subset", ["qe_auroc"], False),
        ("Quick eval: clean reconstruction error (MSE)", ["qe_mse_roi", "qe_mse_bg"], True),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(14, 15))
    for ax, (ttl, cols, logy) in zip(axes.ravel(), panels):
        drawn = False
        for c in cols:
            xs, ys = col(c)
            if xs:
                ax.plot(xs, ys, marker="o" if len(xs) < 30 else None, ms=3, label=c)
                drawn = True
        if ttl.startswith("Gradient"):
            xs, ys = col("lr")
            if xs:
                ax2 = ax.twinx()
                ax2.plot(xs, ys, color="gray", ls="--", label="lr")
                ax2.set_ylabel("lr")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("iteration")
        if drawn:
            if logy and all(v > 0 for c in cols for v in col(c)[1]):
                ax.set_yscale("log")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    return True
