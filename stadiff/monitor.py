"""
Training monitor: everything you can look at while the model trains.

  forward_process_figure  how one image is corrupted at every noise intensity, exactly as the
                          model sees it in Stage 1 and Stage 2 training, plus the test-time start
  training_batch_figure   for real training samples: clean image, noisy input, true noise,
                          predicted noise and the prediction error
  reverse_process_figure  the two-stage reconstruction step by step (clean and anomalous images)
  quick_eval              a small detection test during training (scores, AUROC, reconstruction error)
  BucketStats             loss split by noise level, so you see which levels are learned
  RunLogger               log.csv + TensorBoard (scalars and images)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .diffusion.sampler import reconstruct
from .evaluation.anomaly import anomaly_map, score_batch
from .evaluation.metrics import auroc
from .visualize import gray, gray_auto, heat_rgb, labelled_grid, outline, signed_rgb


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def signal_share(diff, t: int) -> float:
    """sqrt(alpha_bar_t): how much of the clean image is left at noise level t."""
    return float(diff.abar[int(t)].sqrt().item())


def _one(v, device):
    return torch.full((1,), int(v), device=device, dtype=torch.long)


# ---------------------------------------------------------------- forward process
@torch.no_grad()
def forward_process_figure(diff, x0, M, t_roi_max, t_bg, n_levels=5, seed=0):
    """x0, M: (1,1,H,W). One fixed noise field shown at several intensities."""
    dev = x0.device
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev.type == "cuda" else []):
        torch.manual_seed(seed)
        eps = diff.noise(x0)
    xn, Mn, en = _np(x0)[0, 0], _np(M)[0, 0], _np(eps)[0, 0]

    def xt(t_r, t_b):
        Tm = diff.timestep_map(_one(t_r, dev), _one(t_b, dev), M)
        return _np(diff.q_sample(x0, Tm, eps))[0, 0]

    pct = lambda t: f"{signal_share(diff, t):.0%}"
    s1 = np.unique(np.linspace(t_bg + 1, t_roi_max, n_levels).round().astype(int))
    s2 = np.unique(np.linspace(1, t_bg, n_levels).round().astype(int))[::-1]
    Tmap = _np(diff.timestep_map(_one(t_roi_max, dev), _one(t_bg, dev), M))[0, 0]

    row1 = [outline(gray(xn), Mn), gray(Mn, 0, 1), signed_rgb(en)] + \
           [outline(gray_auto(xt(t, t_bg)), Mn) for t in s1]
    cap1 = ["clean input x0", "ROI mask M (white)", f"noise field ({diff.noise.noise_type})"] + \
           [f"ROI t={t}, signal {pct(t)}\nbg t={t_bg}, signal {pct(t_bg)}" for t in s1]
    row2 = [outline(gray(xn), Mn), heat_rgb(Tmap / diff.T, 1.0), outline(gray_auto(xt(t_roi_max, t_bg)), Mn)] + \
           [gray_auto(xt(t, t)) for t in s2]
    cap2 = ["clean input x0", f"timestep map T(i,j)\nROI {t_roi_max} / bg {t_bg}",
            f"TEST-TIME START\nROI {t_roi_max} / bg {t_bg}"] + \
           [f"whole image t={t}\nsignal {pct(t)}" for t in s2]
    return labelled_grid(
        [row1, row2], [cap1, cap2],
        row_titles=["Stage 1 training inputs: ROI noised to t, background fixed at t_bg (red = ROI outline)",
                    "Test-time start, then Stage 2 training inputs: the whole image at one level t"],
        title=f"FORWARD PROCESS  ({diff.noise.noise_type} noise; 'signal' = share of the clean image left, "
              f"sqrt(alpha_bar); noisy tiles contrast-stretched for display)")


# ---------------------------------------------------------------- what the network sees and predicts
@torch.no_grad()
def training_batch_figure(x0, x_t, eps, eps_hat, M, stage, t_roi, t_bg, lam, it, n=4):
    rows, caps, titles = [], [], []
    for i in range(min(n, x0.shape[0])):
        m = _np(M[i, 0])
        err = _np((eps[i, 0] - eps_hat[i, 0]) ** 2)
        e_roi = float(err[m > 0.5].mean())
        e_bg = float(err[m < 0.5].mean()) if (m < 0.5).any() else float("nan")
        s = int(stage[i])
        rows.append([outline(gray(_np(x0[i, 0])), m), outline(gray_auto(_np(x_t[i, 0])), m),
                     signed_rgb(_np(eps[i, 0])), signed_rgb(_np(eps_hat[i, 0])), heat_rgb(err, 2.0)])
        caps.append(["clean x0", "network input x_t", "true noise eps", "predicted noise eps_hat",
                     f"squared error\nROI {e_roi:.3f} | bg {e_bg:.3f}"])
        rule = "loss on ROI pixels only" if s == 1 else f"loss on all pixels, ROI weight {1 + lam:g}"
        titles.append(f"sample {i + 1}: stage {s}, t_roi={int(t_roi[i])}, t_bg={int(t_bg[i])}  ({rule})")
    return labelled_grid(rows, caps, row_titles=titles,
                         title=f"TRAINING SAMPLES at iteration {it}: what the network gets and what it predicts")


# ---------------------------------------------------------------- reverse process
@torch.no_grad()
def reverse_process_figure(model, diff, items, M, rods, cfg, t_roi_max, t_bg, it, seed=0):
    """items: list of (name, x0 (1,1,H,W)). Returns (figure, list of (name, score, per-rod scores))."""
    e = cfg.eval
    rows, caps, titles, results = [], [], [], []
    Mn = _np(M)[0, 0]
    for name, x0 in items:
        states = []
        cb = lambda stage, t, x: states.append((stage, int(t), _np(x)[0, 0]))
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if x0.device.type == "cuda" else []):
            torch.manual_seed(seed)
            x_hat = reconstruct(model, diff, x0, M, t_roi_max, t_bg, e.sampler, e.steps_stage1,
                                e.steps_stage2, callback=cb)
        amap = anomaly_map(x0, x_hat, e.map, e.smooth_sigma)
        img, per_rod = score_batch(amap, M, rods, e.reduce, e.topk_frac, e.image_score)
        s1 = [s for s in states if s[0] == 1]
        s2 = [s for s in states if s[0] == 2]
        pick = lambda lst, k: [lst[int(j)] for j in np.linspace(0, len(lst) - 1, k + 2).round()[1:-1]] if lst else []
        snaps = [states[0]] + pick(s1[:-1], 3) + s1[-1:] + pick(s2[:-1], 2)
        tiles = [outline(gray(_np(x0)[0, 0]), Mn)]
        cap = ["input"]
        for st, t, x in snaps:
            tiles.append(outline(gray_auto(x), Mn) if st < 2 else gray_auto(x))
            if st == 0:
                cap.append(f"start: noised\nROI {t_roi_max} / bg {t_bg}")
            elif st == 1:
                cap.append(f"Stage 1 (ROI only)\nROI t={t}" + ("  HANDOFF" if t == t_bg else ""))
            else:
                cap.append(f"Stage 2 (whole)\nt={t}")
        a = _np(amap)[0, 0]
        tiles += [gray(_np(x_hat)[0, 0]), heat_rgb(a)]
        cap += ["reconstruction x_hat", "anomaly map |x0 - x_hat|"]
        rods_l = [float(v) for v in per_rod[0]]
        results.append((name, float(img[0]), rods_l))
        rows.append(tiles)
        caps.append(cap)
        titles.append(f"{name}:  score {float(img[0]):.4f}   per rod " +
                      " / ".join(f"{v:.4f}" for v in rods_l) + f"   (highest: rod {int(np.argmax(rods_l)) + 1})")
    fig = labelled_grid(rows, caps, row_titles=titles,
                        title=f"REVERSE PROCESS at iteration {it}  ({e.sampler}, {e.steps_stage1}+{e.steps_stage2} steps; "
                              f"background frozen during Stage 1)")
    return fig, results


# ---------------------------------------------------------------- small detection test
@torch.no_grad()
def quick_eval(model, diff, clean, anom, M, rods, cfg):
    """clean / anom: (N,1,H,W) tensors on the device. Returns a dict of metrics."""
    from .evaluation.runner import reconstruct_and_score
    bs = int(cfg.eval.batch_size)
    out = {}
    scores, labels = [], []
    se_roi, se_bg, n_img = 0.0, 0.0, 0
    for label, data in ((0, clean), (1, anom)):
        if data is None or data.shape[0] == 0:
            continue
        for i in range(0, data.shape[0], bs):
            xb = data[i:i + bs]
            x_hat, _, img, _ = reconstruct_and_score(model, diff, xb, M, rods, cfg)
            scores += img.tolist()
            labels += [label] * xb.shape[0]
            if label == 0:
                se = (xb - x_hat) ** 2
                se_roi += float((se * M).sum() / M.sum())
                se_bg += float((se * (1 - M)).sum() / (1 - M).sum().clamp_min(1))
                n_img += xb.shape[0]
    scores, labels = np.array(scores), np.array(labels)
    out["qe_score_clean"] = float(scores[labels == 0].mean()) if (labels == 0).any() else float("nan")
    out["qe_score_anom"] = float(scores[labels == 1].mean()) if (labels == 1).any() else float("nan")
    out["qe_auroc"] = auroc(labels, scores)
    if n_img:
        out["qe_mse_roi"] = se_roi / n_img
        out["qe_mse_bg"] = se_bg / n_img
        out["qe_psnr_clean"] = float(10 * np.log10(4.0 / max((se_roi + se_bg) / (2 * n_img), 1e-12)))
    return out


# ---------------------------------------------------------------- loss by noise level
class BucketStats:
    """Average loss per noise-level bucket, separately for stage 1 (by t_roi) and stage 2 (by t)."""

    def __init__(self, t_roi_lo, t_roi_max, t_bg_max, n=4):
        self.e1 = np.linspace(t_roi_lo, t_roi_max + 1, n + 1).round().astype(int)
        self.e2 = np.linspace(1, t_bg_max + 1, n + 1).round().astype(int)
        self.names = ([f"s1_t{a}-{b - 1}" for a, b in zip(self.e1[:-1], self.e1[1:])] +
                      [f"s2_t{a}-{b - 1}" for a, b in zip(self.e2[:-1], self.e2[1:])])
        self.reset()

    def reset(self):
        self.sum = {k: 0.0 for k in self.names}
        self.cnt = {k: 0 for k in self.names}

    def add(self, per_sample, stage, t_roi):
        per, st, tr = _np(per_sample), _np(stage).astype(int), _np(t_roi).astype(int)
        for p, s, t in zip(per, st, tr):
            edges, prefix = (self.e1, "s1") if s == 1 else (self.e2, "s2")
            j = int(np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2))
            k = f"{prefix}_t{edges[j]}-{edges[j + 1] - 1}"
            self.sum[k] += float(p)
            self.cnt[k] += 1

    def values(self):
        return {k: (self.sum[k] / self.cnt[k] if self.cnt[k] else "") for k in self.names}


# ---------------------------------------------------------------- logging
class RunLogger:
    """log.csv (one row per log interval) and, if available, TensorBoard."""

    def __init__(self, out_dir, columns, resume=False, tensorboard=True):
        self.columns = ["iter"] + list(columns)
        self.path = Path(out_dir) / "log.csv"
        append = False
        if resume and self.path.exists():
            with self.path.open() as fh:
                header = next(csv.reader(fh), [])
            if header == self.columns:
                append = True
            else:
                self.path.replace(self.path.with_name("log_previous.csv"))
        self.fh = self.path.open("a" if append else "w", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=self.columns, extrasaction="ignore")
        if not append:
            self.w.writeheader()
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(Path(out_dir) / "tensorboard"))
            except Exception:            # tensorboard not installed
                self.tb = None

    def log(self, it, values: dict):
        row = {"iter": it, **values}
        self.w.writerow({k: row.get(k, "") for k in self.columns})
        self.fh.flush()
        if self.tb:
            for k, v in values.items():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    self.tb.add_scalar(k.replace("_", "/", 1) if k.startswith(("s1_", "s2_", "qe_")) else k, v, it)

    def image(self, tag, arr, it):
        if self.tb:
            self.tb.add_image(tag, arr, it, dataformats="HWC")

    def close(self):
        self.fh.close()
        if self.tb:
            self.tb.close()
