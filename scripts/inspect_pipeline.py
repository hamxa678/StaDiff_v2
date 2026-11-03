"""
Check that a trained model behaves exactly as designed, one image at a time.

    python scripts/inspect_pipeline.py --checkpoint runs/simplex_800_400/ckpt_best.pt
    python scripts/inspect_pipeline.py --checkpoint runs/simplex_800_400/ckpt_best.pt --images a.png b.png --shift 14

For each image it runs the two-stage reconstruction, records every intermediate state and checks:
  [1] noise LEVELS: the measured share of the clean image left in the ROI and in the background
      matches sqrt(alpha_bar) at t_roi and t_bg
  [2] noise TYPE: neighbouring noise values are correlated for simplex, uncorrelated for gaussian
  [3] Stage 1 changes the ROI ...
  [4] ... and leaves every background pixel exactly unchanged (frozen at t_bg)
  [5] Stage 1 ends at t_bg (handoff), Stage 2 ends at t = 0
  [6] Stage 2 changes the background too (whole-image denoising)
  [7] the anomaly score and which rod scores highest
  [8] distance to the input image, per region, at every step: in Stage 1 the ROI gets closer while
      the background distance stays exactly the same; in Stage 2 both get closer
By default it uses 2 clean test images, 2 anomalous images, and a synthetic anomaly: a clean image
whose middle rod is shifted sideways by --shift pixels (a spacing defect you know the answer to).

Writes
  inspect.png     one row per image: input | regional noising | Stage 1 steps | handoff | Stage 2 steps |
                  reconstruction | anomaly map
  trajectory.png  per image: distance of the ROI and of the background to the input at every step,
                  with the noise level of each region (the numerical proof of "ROI first, then all")
  trajectory.csv  the same numbers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EVAL_FLAGS, load_model_from_checkpoint                    # noqa: E402
from stadiff.cli import add_config_flags                                       # noqa: E402
from stadiff.data.files import split_files                                     # noqa: E402
from stadiff.data.masks import get_masks                                       # noqa: E402
from stadiff.data.preprocess import load_image                                 # noqa: E402
from stadiff.diffusion import RegionalDiffusion, reconstruct                   # noqa: E402
from stadiff.evaluation.anomaly import anomaly_map, score_batch                # noqa: E402
from stadiff.evaluation.runner import resolve_levels                           # noqa: E402
from stadiff.utils import pick_device                                          # noqa: E402
from stadiff.visualize import gray, gray_auto, heat_rgb, labelled_grid, outline, save_png  # noqa: E402

OK, BAD = "PASS", "FAIL"


def shift_middle_rod(x: np.ndarray, rods: np.ndarray, shift: int) -> np.ndarray:
    """Synthetic spacing defect: move the content of rod 2's column sideways by `shift` pixels."""
    cols = np.flatnonzero(rods[1].max(0) > 0)
    x0, x1 = max(cols[0] - abs(shift), 0), min(cols[-1] + abs(shift) + 1, x.shape[1])
    y = x.copy()
    y[:, x0:x1] = np.roll(x[:, x0:x1], shift, axis=1)
    return y


def lag1_corr(r: np.ndarray, m: np.ndarray) -> float:
    """Correlation between horizontally neighbouring residual values inside region m."""
    a, b = r[:, :-1], r[:, 1:]
    k = (m[:, :-1] > 0) & (m[:, 1:] > 0)
    if k.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[k], b[k])[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", nargs="*", default=None, help="images to inspect (default: picked from the dataset)")
    ap.add_argument("--shift", type=int, default=14, help="pixels for the synthetic shifted-rod image (0 = none)")
    ap.add_argument("--out", default=None, help="output folder (default: next to the checkpoint)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    add_config_flags(ap, EVAL_FLAGS)
    a = ap.parse_args()

    device = pick_device(a.device)
    model, cfg, it = load_model_from_checkpoint(a.checkpoint, a, device)
    diff = RegionalDiffusion(cfg.diffusion, device)
    t_roi, t_bg = resolve_levels(cfg)
    e = cfg.eval
    size = tuple(cfg.data.image_size)
    M, rods = get_masks(cfg.data)
    M_t = torch.as_tensor(M, device=device)[None, None]
    rods_t = torch.as_tensor(rods, device=device)
    out = Path(a.out) if a.out else Path(a.checkpoint).parent / "inspect"
    out.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint iteration {it}")
    print(f"noise type {cfg.diffusion.noise_type} | t_roi {t_roi} | t_bg {t_bg} | sampler {e.sampler} "
          f"| ROI = {cfg.data.mask_source} ({M.mean():.0%} of the image, {len(rods)} rods)")
    exp_roi, exp_bg = diff.abar[t_roi].sqrt().item(), diff.abar[t_bg].sqrt().item()
    print(f"expected share of the clean image left after noising: ROI {exp_roi:.3f}, background {exp_bg:.3f}\n")

    items = []
    if a.images:
        items = [(Path(p).name, load_image(p, size, cfg.data.prepare_method, cfg.data.norm)) for p in a.images]
    else:
        files = split_files(cfg.data.root, 0.0)
        clean = [p for p, l in files["test"] if l == 0][:2]
        anom = [p for p, l in files["test"] if l == 1][:2]
        for p in clean + anom:
            items.append((("clean " if p in clean else "anomalous ") + p.name,
                          load_image(p, size, cfg.data.prepare_method, cfg.data.norm)))
        if a.shift and clean:
            x = load_image(clean[0], size, cfg.data.prepare_method, cfg.data.norm)
            items.append((f"SYNTHETIC: rod 2 shifted {a.shift}px", shift_middle_rod(x, rods, a.shift)))

    rows, caps, titles, trajs, all_ok = [], [], [], [], True
    torch.manual_seed(int(e.seed))
    for name, x_np in items:
        states = []
        cb = lambda stage, t, x: states.append((stage, int(t), x[0, 0].detach().float().cpu().numpy().copy()))
        x0 = torch.from_numpy(x_np)[None, None].to(device)
        x_hat = reconstruct(model, diff, x0, M_t, t_roi, t_bg, e.sampler, e.steps_stage1, e.steps_stage2, cb)
        amap = anomaly_map(x0, x_hat, e.map, e.smooth_sigma)
        img, per_rod = score_batch(amap, M_t, rods_t, e.reduce, e.topk_frac, e.image_score)

        xT = states[0][2]
        s1 = [s for s in states if s[0] == 1]
        s2 = [s for s in states if s[0] == 2]
        roi, bg = M > 0.5, M < 0.5
        checks = []

        # [1] noise levels: regression slope of x_T on x0 per region = sqrt(alpha_bar)
        def slope(region):
            xa, xb = x_np[region], xT[region]
            return float(np.cov(xa, xb)[0, 1] / max(np.var(xa), 1e-8))
        m_roi, m_bg = slope(roi), slope(bg)
        tol = 0.08 if cfg.diffusion.noise_type == "gaussian" else 0.2    # simplex estimates are noisier
        checks.append((abs(m_roi - exp_roi) < tol and abs(m_bg - exp_bg) < tol,
                       f"[1] noise levels  ROI {m_roi:.3f} (expected {exp_roi:.3f}), "
                       f"background {m_bg:.3f} (expected {exp_bg:.3f})"))
        # [2] noise type from the spatial correlation of the noise
        resid = xT - np.where(roi, m_roi, m_bg) * x_np
        c = lag1_corr(resid, M)
        detected = "simplex" if c > 0.3 else "gaussian"
        checks.append((detected == cfg.diffusion.noise_type,
                       f"[2] noise type    neighbour correlation {c:.2f} -> looks {detected}"))
        # [3] / [4] Stage 1
        last1 = s1[-1][2] if s1 else xT
        roi_change = float(np.abs(last1 - xT)[roi].mean())
        bg_change = max(float(np.abs(s[2] - xT)[bg].max()) for s in s1) if s1 else 0.0
        checks.append((roi_change > 1e-3, f"[3] Stage 1 ROI   mean change {roi_change:.4f}"))
        checks.append((bg_change == 0.0, f"[4] Stage 1 bg    max change {bg_change:.2e} (must be exactly 0)"))
        # [5] handoff and end
        end1 = s1[-1][1] if s1 else None
        end2 = s2[-1][1] if s2 else None
        checks.append((end1 == t_bg and end2 == 0, f"[5] schedule      Stage 1 ends at t={end1}, Stage 2 ends at t={end2}"
                                                   f" ({len(s1)} + {len(s2)} steps)"))
        # [6] Stage 2 touches the background
        bg2 = float(np.abs(s2[-1][2] - last1)[bg].mean()) if s2 else 0.0
        checks.append((bg2 > 1e-3, f"[6] Stage 2 bg    mean change {bg2:.4f}"))
        # [8] distance to the input per region along the whole trajectory
        rmse = lambda x, r: float(np.sqrt(((x - x_np)[r] ** 2).mean()))
        traj = [(st, t, rmse(x, roi), rmse(x, bg), (t if st < 2 else t), (t_bg if st < 2 else t))
                for st, t, x in states]
        trajs.append((name, traj))
        d0, d1, d2 = traj[0], (s1 and traj[len(s1)]) or traj[0], traj[-1]
        ok8 = (d1[2] < d0[2] and d1[3] == d0[3] and d2[3] < d1[3] and d2[2] <= d1[2] + 1e-6)
        checks.append((ok8, f"[8] distance to input (RMSE)  ROI {d0[2]:.3f} -> {d1[2]:.3f} (handoff) -> {d2[2]:.3f}"
                            f"  |  background {d0[3]:.3f} -> {d1[3]:.3f} -> {d2[3]:.3f}"))

        print(f"--- {name}")
        for ok, msg in checks:
            all_ok &= ok
            print(f"  {OK if ok else BAD}  {msg}")
        rs = per_rod[0].tolist()
        print(f"  [7] score {img.item():.5f}   per rod: " + "  ".join(f"rod{i + 1} {v:.5f}" for i, v in enumerate(rs))
              + f"   highest: rod {int(np.argmax(rs)) + 1}")

        pick = lambda lst, k: [lst[int(j)] for j in np.linspace(0, len(lst) - 1, k + 2).round()[1:-1]] if lst else []
        snaps = [states[0]] + pick(s1[:-1], 3) + s1[-1:] + pick(s2[:-1], 3)
        tiles, cap = [outline(gray(x_np), M)], ["input"]
        for st, t, x in snaps:
            tiles.append(outline(gray_auto(x), M) if st < 2 else gray_auto(x))
            cap.append(f"start: ROI {t_roi} / bg {t_bg}" if st == 0 else
                       (f"Stage 1  ROI t={t}" + ("\nHANDOFF (all at t_bg)" if t == t_bg else "\nbackground frozen")
                        if st == 1 else f"Stage 2  t={t}\nwhole image"))
        amap_np = amap[0, 0].cpu().numpy()
        tiles += [gray(x_hat[0, 0].float().cpu().numpy()), heat_rgb(amap_np)]
        cap += ["reconstruction", "anomaly map"]
        rows.append(tiles)
        caps.append(cap)
        titles.append(f"{name}:  score {img.item():.4f}, highest rod {int(np.argmax(rs)) + 1}")

    fig = labelled_grid(rows, caps, row_titles=titles,
                        title=f"REVERSE PROCESS  t_roi {t_roi}, t_bg {t_bg}, {cfg.diffusion.noise_type} noise, "
                              f"{e.sampler} (noisy tiles contrast-stretched; red = ROI)")
    save_png(fig, out / "inspect.png")

    with (out / "trajectory.csv").open("w") as fh:
        fh.write("image,step,stage,t_roi,t_bg,rmse_roi_to_input,rmse_bg_to_input\n")
        for name, traj in trajs:
            for k, (st, t, dr, db, tr, tb) in enumerate(traj):
                fh.write(f"\"{name}\",{k},{st},{tr},{tb},{dr:.6f},{db:.6f}\n")
    plotted = plot_trajectories(trajs, out / "trajectory.png", t_roi, t_bg)

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED - see above'}")
    print(f"saved {out / 'inspect.png'}  (every stage of the reconstruction, per image)")
    if plotted:
        print(f"saved {out / 'trajectory.png'}  (distance to the input per region, step by step)")
    print(f"saved {out / 'trajectory.csv'}")


def plot_trajectories(trajs, path, t_roi, t_bg) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    n = len(trajs)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), squeeze=False)
    for ax, (name, traj) in zip(axes[:, 0], trajs):
        k = np.arange(len(traj))
        st = np.array([s[0] for s in traj])
        ax.plot(k, [s[2] for s in traj], color="tab:red", lw=2, label="ROI: distance to input")
        ax.plot(k, [s[3] for s in traj], color="tab:blue", lw=2, label="background: distance to input")
        s1_end = int(np.flatnonzero(st == 1).max()) if (st == 1).any() else 0
        ax.axvspan(0, s1_end, color="tab:red", alpha=0.07)
        ax.axvline(s1_end, color="k", ls=":", lw=1)
        ax.text(s1_end / 2, ax.get_ylim()[1] * 0.95, "Stage 1: ROI only", ha="center", va="top", fontsize=9)
        ax.text((s1_end + len(traj) - 1) / 2, ax.get_ylim()[1] * 0.95, "Stage 2: whole image",
                ha="center", va="top", fontsize=9)
        ax.set_ylabel("RMSE to input")
        ax2 = ax.twinx()
        ax2.plot(k, [s[4] for s in traj], color="tab:red", ls="--", lw=1, label="ROI noise level t")
        ax2.plot(k, [s[5] for s in traj], color="tab:blue", ls="--", lw=1, label="background noise level t")
        ax2.set_ylabel("noise level t")
        ax2.set_ylim(0, t_roi * 1.05)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel(f"reverse step (handoff at t_bg = {t_bg}, dotted line)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return True


if __name__ == "__main__":
    main()
