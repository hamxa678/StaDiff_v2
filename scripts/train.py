"""
Step 3 — train StaDiff-V3 on clean images.

    python scripts/train.py --data DATASET_prepared --noise gaussian --t-roi 800 --t-bg 400 --out runs/g800_400
    python scripts/train.py --data DATASET_prepared --noise simplex --t-roi 700 --t-bg 300 --iters 40000 --out runs/s700_300
    python scripts/train.py --out runs/g800_400 --resume runs/g800_400/ckpt_last.pt
    python scripts/train.py --help        # every available flag

Each iteration (see stadiff/diffusion/schedule.py):
  1. take a batch of clean images x0 and the fixed ROI mask M
  2. draw for every image a stage (1 or 2) and its timesteps (t_roi, t_bg)
  3. build the regional noisy image x_t and the target noise eps
  4. predict eps with the dual-stream U-Net and apply the stage-dependent loss

What you get in --out while it trains
  vis/00_forward_process.png       the forward corruption of one image at every noise intensity (at start)
  vis/train_samples_<it>.png       real training samples: input, true noise, predicted noise, error
  vis/reverse_<it>.png             step-by-step two-stage reconstruction of clean and anomalous images
  curves.png                       all training curves (updated regularly)
  log.csv                          every number printed on the console
  tensorboard/                     the same as live dashboards:  tensorboard --logdir <out>/tensorboard
  ckpt_best.pt / ckpt_last.pt / ckpt_<it>.pt, config.yaml, roi_mask.png
Anomalous images are used ONLY for these figures and the quick test, never for training or for
choosing the best checkpoint (that uses the clean validation loss).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stadiff.cli import add_common_flags, add_config_flags, config_from_args  # noqa: E402
from stadiff.config import save_config, to_dict                               # noqa: E402
from stadiff.data.dataset import build_datasets, make_loader                  # noqa: E402
from stadiff.data.masks import save_mask_png                                  # noqa: E402
from stadiff.diffusion import RegionalDiffusion                               # noqa: E402
from stadiff.evaluation.runner import resolve_levels                          # noqa: E402
from stadiff.models import build_model, count_parameters                      # noqa: E402
from stadiff.monitor import (BucketStats, RunLogger, forward_process_figure,  # noqa: E402
                             quick_eval, reverse_process_figure, signal_share, training_batch_figure)
from stadiff.utils import EMA, pick_device, save_checkpoint, seed_everything, torch_load  # noqa: E402
from stadiff.visualize import plot_curves, save_png                           # noqa: E402


def infinite(loader):
    while True:
        for b in loader:
            yield b


def fmt_time(sec: float) -> str:
    sec = int(max(sec, 0))
    h, m = divmod(sec // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {sec % 60:02d}s"


@torch.no_grad()
def validation_loss(model, diffusion, loader, device, max_batches=50):
    """Average loss on held-out clean images with a fixed random state (comparable over time)."""
    model.eval()
    total, n = 0.0, 0
    cuda_devs = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devs):
        torch.manual_seed(1234)
        for i, b in enumerate(loader):
            if i >= max_batches:
                break
            x0, M = b["x0"].to(device), b["mask"].to(device)
            x_t, Tmap, t_roi, t_bg, stage, eps = diffusion.training_batch(x0, M)
            loss, _ = diffusion.loss(eps, model(x_t, M, Tmap, t_roi, t_bg, stage).float(), M, stage)
            total += loss.item() * x0.shape[0]
            n += x0.shape[0]
    model.train()
    return total / max(n, 1)


def stack_images(ds, idx, device):
    if not len(idx):
        return None
    return torch.stack([ds[i]["x0"] for i in idx]).to(device)


def banner(cfg, diffusion, model, n_train, n_val, n_anom, M, rods, device, amp_txt):
    d, m, t = cfg.diffusion, cfg.model, cfg.train
    line = "=" * 92
    parts = {"ROI encoder": model.roi_enc, "context encoder": model.ctx_enc, "encoder exchange": model.fusions,
             "condition embedding": model.cond}
    pc = {k: (count_parameters(v) if v is not None else 0) for k, v in parts.items()}
    total = count_parameters(model)
    pc["bottleneck + decoder"] = total - sum(pc.values())
    noise_txt = (f"simplex (scale {d.simplex_scale:g}px, {d.simplex_octaves} octaves, persistence {d.simplex_persistence:g})"
                 if d.noise_type == "simplex" else "gaussian")
    bg_txt = (f"t_bg random in [{diffusion.t_bg_min}, {diffusion.t_bg_max}]" if d.random_bg else f"t_bg = {d.t_bg}")
    shares = "  ".join(f"t={k}:{signal_share(diffusion, k):.2f}" for k in
                       sorted({0, 100, 200, d.t_bg, 600, d.t_roi_max, diffusion.T}) if k <= diffusion.T)
    epochs = t.iters * t.batch_size / max(n_train, 1)
    print(line)
    print(" StaDiff-V3 training")
    print(line)
    print(f" data        {n_train} clean train | {n_val} clean val | {n_anom} anomalous (figures + quick test only)")
    print(f" images      {cfg.data.image_size[0]}x{cfg.data.image_size[1]}, resize {cfg.data.prepare_method}, "
          f"norm {cfg.data.norm}, augment {cfg.data.augment}")
    print(f" ROI mask    {cfg.data.mask_source}: {M.mean():.0%} of the image, {len(rods)} rods")
    print(f" noise       {noise_txt}, same type in ROI and background")
    print(f" schedule    {d.beta_schedule}, T = {diffusion.T}")
    print(f" stage 1     ROI t_roi in [{diffusion.t_bg_min + 1}, {d.t_roi_max}], background {bg_txt}, "
          f"loss on ROI only      (p = {d.p_stage1:g})")
    print(f" stage 2     whole image t in [1, {diffusion.t_bg_max}], loss on all pixels, ROI weight {1 + d.lam:g}"
          f"   (p = {1 - d.p_stage1:g})")
    print(f" signal left sqrt(alpha_bar):  {shares}")
    print(f" model       {total / 1e6:.1f} M parameters: " +
          ", ".join(f"{k} {v / 1e6:.1f}M" for k, v in pc.items()))
    print(f"             fusion per level {list(m.fusion)}, context stream {m.use_context_stream}, "
          f"bottleneck cross-attn {m.bottleneck_cross_attn}")
    print(f" training    {t.iters} iterations x batch {t.batch_size} = {epochs:.0f} epochs, lr {t.lr:g} "
          f"(warm-up {t.warmup_iters}), EMA {t.ema_decay}, grad clip {t.grad_clip}")
    print(f" device      {device}{amp_txt}")
    print(f" every       log {t.log_every} | val loss {t.val_every} | samples figure {t.batch_vis_every} | "
          f"reverse figure {t.preview_every} | quick test {t.quick_eval_every or 'off'} | ckpt {t.ckpt_every}")
    print(f" output      {t.out_dir}   (vis/, curves.png, log.csv, tensorboard/, checkpoints)")
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | mps | cpu")
    add_common_flags(ap)
    add_config_flags(ap, ["data", "diffusion", "model", "train"])
    a = ap.parse_args()

    cfg = config_from_args(a)
    tc = cfg.train
    out = Path(tc.out_dir)
    (out / "vis").mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    seed_everything(tc.seed)
    device = pick_device(a.device)

    train_ds, val_ds, test_ds, M, rods = build_datasets(cfg)
    save_mask_png(M, out / "roi_mask.png")
    if len(train_ds) < tc.batch_size:
        sys.exit("fewer training images than train.batch_size")
    train_loader = make_loader(train_ds, tc.batch_size, True, tc.num_workers, drop_last=True)
    val_loader = make_loader(val_ds, tc.batch_size, False, tc.num_workers) if len(val_ds) else None

    model = build_model(cfg).to(device)
    diffusion = RegionalDiffusion(cfg.diffusion, device)
    ema = EMA(model, tc.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / max(1, tc.warmup_iters)))

    use_amp = bool(tc.amp) and device.type == "cuda"
    # bfloat16 only on GPUs that support it natively (compute capability >= 8: A100, L4, RTX 30xx/40xx).
    # torch.cuda.is_bf16_supported() also returns True on older GPUs such as the T4, where bf16 is
    # only emulated and very slow; those use float16 + gradient scaling instead.
    native_bf16 = use_amp and torch.cuda.get_device_capability(device)[0] >= 8
    amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
    scaler_enabled = use_amp and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):          # older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    amp_txt = (f" ({torch.cuda.get_device_name(device)}, mixed precision {str(amp_dtype).replace('torch.', '')})"
               if use_amp else (f" ({torch.cuda.get_device_name(device)}, full precision)" if device.type == "cuda" else ""))

    # images used only for looking at the model (never for training)
    anom_idx = [i for i, (_, l) in enumerate(test_ds.items) if l == 1]
    mon_ds = val_ds if len(val_ds) else train_ds
    n_q = int(tc.quick_eval_n)
    qe_clean = stack_images(mon_ds, list(range(min(n_q, len(mon_ds)))), device)
    qe_anom = stack_images(test_ds, anom_idx[:n_q], device)
    vis_items = [(f"clean {Path(mon_ds.items[i][0]).name}", qe_clean[i:i + 1]) for i in range(min(2, qe_clean.shape[0]))]
    if qe_anom is not None:
        vis_items += [(f"ANOMALOUS {Path(test_ds.items[anom_idx[i]][0]).name}", qe_anom[i:i + 1])
                      for i in range(min(2, qe_anom.shape[0]))]
    M_t = torch.as_tensor(M, device=device)[None, None]
    rods_t = torch.as_tensor(rods, device=device)
    t_roi_eval, t_bg_eval = resolve_levels(cfg)

    banner(cfg, diffusion, model, len(train_ds), len(val_ds), len(anom_idx), M, rods, device, amp_txt)

    start_it, best_val = 0, math.inf
    if tc.resume:
        ck = torch_load(tc.resume, device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_it, best_val = ck["iter"], ck.get("best_val", math.inf)
        print(f"resumed from {tc.resume} at iteration {start_it}")

    buckets = BucketStats(diffusion.t_bg_min + 1, diffusion.t_roi_max, diffusion.t_bg_max)
    qe_cols = ["qe_score_clean", "qe_score_anom", "qe_auroc", "qe_mse_roi", "qe_mse_bg", "qe_psnr_clean"]
    columns = (["loss", "loss_stage1", "loss_stage2", "val_loss", "s2_err_roi", "s2_err_bg"] + buckets.names
               + ["grad_norm", "lr", "sec_per_iter", "data_sec", "gpu_mem_gb"] + qe_cols)
    logger = RunLogger(out, columns, resume=bool(tc.resume), tensorboard=bool(tc.tensorboard))

    # forward process figure: what the corruption looks like, before any training
    fig = forward_process_figure(diffusion, vis_items[0][1], M_t, diffusion.t_roi_max, diffusion.t_bg)
    save_png(fig, out / "vis" / "00_forward_process.png")
    logger.image("forward_process", fig, start_it)
    print(f"saved {out / 'vis' / '00_forward_process.png'}  (forward corruption at every intensity)\n", flush=True)

    def checkpoint(it, name, full=True):
        """full: everything needed to resume. Otherwise only the EMA weights (smaller files),
        which is all evaluate.py / infer.py need."""
        state = dict(ema=ema.state_dict(), iter=it, best_val=best_val, config=to_dict(cfg))
        if full:
            state.update(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict())
        save_checkpoint(out / name, **state)

    def new_acc():
        return {"loss": 0.0, "n": 0, "s1": 0.0, "n1": 0, "s2": 0.0, "n2": 0, "er": 0.0, "eb": 0.0,
                "ne": 0, "grad": 0.0, "data": 0.0}

    model.train()
    batches = infinite(train_loader)
    acc = new_acc()
    t0 = time.time()
    run_start, it_start = time.time(), start_it
    for it in range(start_it + 1, tc.iters + 1):
        td = time.time()
        b = next(batches)
        x0, M_b = b["x0"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
        acc["data"] += time.time() - td
        x_t, Tmap, t_roi, t_bg, stage, eps = diffusion.training_batch(x0, M_b)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            eps_hat = model(x_t, M_b, Tmap, t_roi, t_bg, stage)
        eps_hat = eps_hat.float()
        loss, per = diffusion.loss(eps, eps_hat, M_b, stage)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model)

        # ---- statistics
        with torch.no_grad():
            acc["loss"] += loss.item()
            acc["n"] += 1
            acc["grad"] += float(gnorm)
            s1 = stage == 1
            acc["s1"] += per[s1].sum().item()
            acc["n1"] += int(s1.sum())
            acc["s2"] += per[~s1].sum().item()
            acc["n2"] += int((~s1).sum())
            if (~s1).any():                      # stage 2: plain noise error on ROI vs background pixels
                se = (eps - eps_hat)[~s1] ** 2
                Ms = M_b[~s1]
                acc["er"] += float((se * Ms).sum() / Ms.sum().clamp_min(1))
                acc["eb"] += float((se * (1 - Ms)).sum() / (1 - Ms).sum().clamp_min(1))
                acc["ne"] += 1
            buckets.add(per, stage, t_roi)

        if tc.batch_vis_every and (it % tc.batch_vis_every == 0 or it == start_it + 1):
            fig = training_batch_figure(x0, x_t, eps, eps_hat, M_b, stage, t_roi, t_bg, cfg.diffusion.lam, it)
            save_png(fig, out / "vis" / f"train_samples_{it:07d}.png")
            logger.image("train_samples", fig, it)

        row = {}
        if val_loader is not None and it % tc.val_every == 0:
            row["val_loss"] = validation_loss(ema.model, diffusion, val_loader, device)
            improved = row["val_loss"] < best_val
            if improved:
                best_val = row["val_loss"]
                checkpoint(it, "ckpt_best.pt", full=False)
            print(f"  validation loss {row['val_loss']:.4f}  (best {best_val:.4f})"
                  + ("  -> saved ckpt_best.pt" if improved else ""), flush=True)

        if tc.quick_eval_every and it % tc.quick_eval_every == 0:
            tq = time.time()
            ema.model.eval()
            qe = quick_eval(ema.model, diffusion, qe_clean, qe_anom, M_t, rods_t, cfg)
            row.update(qe)
            n_a = 0 if qe_anom is None else qe_anom.shape[0]
            print(f"  quick test ({qe_clean.shape[0]} clean + {n_a} anomalous, t_roi {t_roi_eval}, t_bg {t_bg_eval}): "
                  f"mean score clean {qe['qe_score_clean']:.4f} / anomalous {qe['qe_score_anom']:.4f} | "
                  f"AUROC {qe['qe_auroc']:.3f} | clean reconstruction MSE ROI {qe.get('qe_mse_roi', float('nan')):.4f} "
                  f"bg {qe.get('qe_mse_bg', float('nan')):.4f}, PSNR {qe.get('qe_psnr_clean', float('nan')):.1f} dB"
                  f"  [{fmt_time(time.time() - tq)}]", flush=True)

        if it % tc.log_every == 0 or row:
            n = max(acc["n"], 1)
            spi = (time.time() - t0) / n
            done = it - it_start
            eta = (time.time() - run_start) / max(done, 1) * (tc.iters - it)
            mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
            row.update({
                "loss": acc["loss"] / n, "loss_stage1": acc["s1"] / max(acc["n1"], 1) if acc["n1"] else "",
                "loss_stage2": acc["s2"] / max(acc["n2"], 1) if acc["n2"] else "",
                "s2_err_roi": acc["er"] / acc["ne"] if acc["ne"] else "",
                "s2_err_bg": acc["eb"] / acc["ne"] if acc["ne"] else "",
                **buckets.values(), "grad_norm": acc["grad"] / n, "lr": sched.get_last_lr()[0],
                "sec_per_iter": round(spi, 4), "data_sec": round(acc["data"] / n, 4), "gpu_mem_gb": round(mem, 2),
            })
            logger.log(it, row)
            f = lambda k: f"{row[k]:.4f}" if row.get(k, "") != "" else "  -   "
            print(f"it {it:>7d}/{tc.iters} | loss {row['loss']:.4f} | stage1 {f('loss_stage1')} | stage2 {f('loss_stage2')}"
                  f" (noise err ROI {f('s2_err_roi')}, bg {f('s2_err_bg')}) | grad {row['grad_norm']:.2f} | "
                  f"lr {row['lr']:.1e} | {spi:.3f}s/it (data {row['data_sec']:.3f}) | mem {mem:.1f}GB | ETA {fmt_time(eta)}",
                  flush=True)
            bv = buckets.values()
            s1b = "  ".join(f"{k[4:]} {v:.3f}" for k, v in bv.items() if k.startswith("s1_") and v != "")
            s2b = "  ".join(f"{k[4:]} {v:.3f}" for k, v in bv.items() if k.startswith("s2_") and v != "")
            print(f"           stage1 loss by ROI level: {s1b}\n           stage2 loss by level:     {s2b}", flush=True)
            acc = new_acc()
            buckets.reset()
            t0 = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        if it % tc.ckpt_every == 0 or it == tc.iters:
            checkpoint(it, "ckpt_last.pt")
            checkpoint(it, f"ckpt_{it:07d}.pt", full=False)
            print(f"  saved checkpoints ckpt_last.pt, ckpt_{it:07d}.pt", flush=True)

        if it % tc.preview_every == 0 or it == tc.iters:
            ema.model.eval()
            fig, res = reverse_process_figure(ema.model, diffusion, vis_items, M_t, rods_t, cfg,
                                              t_roi_eval, t_bg_eval, it)
            save_png(fig, out / "vis" / f"reverse_{it:07d}.png")
            logger.image("reverse_process", fig, it)
            print(f"  saved vis/reverse_{it:07d}.png: " + " | ".join(f"{n.split()[0]} score {s:.4f}" for n, s, _ in res),
                  flush=True)
            if plot_curves(logger.path, out / "curves.png"):
                print(f"  updated curves.png", flush=True)
        model.train()

    logger.close()
    if val_loader is None:          # no validation split: "best" = last
        checkpoint(tc.iters, "ckpt_best.pt", full=False)
    plot_curves(out / "log.csv", out / "curves.png")
    print(f"done. everything is in {out}")


if __name__ == "__main__":
    main()
