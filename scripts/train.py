"""
Step 3 — train StaDiff-V3 on clean images.

    python scripts/train.py --data DATASET_prepared --noise gaussian --t-roi 800 --t-bg 400 --out runs/g800_400
    python scripts/train.py --data DATASET_prepared --noise simplex --t-roi 700 --t-bg 300 --iters 80000 --out runs/s700_300
    python scripts/train.py --out runs/g800_400 --resume runs/g800_400/ckpt_last.pt
    python scripts/train.py --help        # every available flag

Each iteration (see stadiff/diffusion/schedule.py):
  1. take a batch of clean images x0 and the fixed ROI mask M
  2. draw for every image a stage (1 or 2) and its timesteps (t_roi, t_bg)
  3. build the regional noisy image x_t and the target noise eps
  4. predict eps with the dual-stream U-Net and apply the stage-dependent loss
Outputs in train.out_dir: config.yaml, roi_mask.png, log.csv, ckpt_last.pt, ckpt_best.pt,
ckpt_<iter>.pt, previews/preview_<iter>.png (input | reconstruction | anomaly map).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stadiff.cli import add_common_flags, add_config_flags, config_from_args  # noqa: E402
from stadiff.config import save_config, to_dict                               # noqa: E402
from stadiff.data.dataset import build_datasets, make_loader                  # noqa: E402
from stadiff.data.masks import save_mask_png                                  # noqa: E402
from stadiff.diffusion import RegionalDiffusion                               # noqa: E402
from stadiff.evaluation.runner import reconstruct_and_score                   # noqa: E402
from stadiff.models import build_model, count_parameters                      # noqa: E402
from stadiff.utils import (EMA, heat, pick_device, save_checkpoint, save_triplets,  # noqa: E402
                           seed_everything, to_uint8, torch_load)


def infinite(loader):
    while True:
        for b in loader:
            yield b


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


@torch.no_grad()
def save_preview(model, diffusion, batch, M, rods, cfg, device, path):
    model.eval()
    x0 = batch["x0"][:4].to(device)
    M_t = torch.as_tensor(M, device=device)[None, None]
    rods_t = torch.as_tensor(rods, device=device)
    x_hat, amap, img, _ = reconstruct_and_score(model, diffusion, x0, M_t, rods_t, cfg)
    vmax = float(amap.flatten().quantile(0.995)) if amap.numel() < 16_000_000 else float(amap.max())
    rows = [(to_uint8(x0[i]), to_uint8(x_hat[i]), heat(amap[i, 0].cpu().numpy(), vmax), M)
            for i in range(x0.shape[0])]
    save_triplets(rows, path)
    model.train()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | mps | cpu")
    add_common_flags(ap)
    add_config_flags(ap, ["data", "diffusion", "model", "train"])
    a = ap.parse_args()

    cfg = config_from_args(a)
    tc = cfg.train
    out = Path(tc.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    save_config(cfg, out / "config.yaml")
    seed_everything(tc.seed)
    device = pick_device(a.device)

    train_ds, val_ds, _, M, rods = build_datasets(cfg)
    save_mask_png(M, out / "roi_mask.png")
    print(f"train {len(train_ds)} clean images, val {len(val_ds)}; device {device}")
    if len(train_ds) < tc.batch_size:
        sys.exit("fewer training images than train.batch_size")
    train_loader = make_loader(train_ds, tc.batch_size, True, tc.num_workers, drop_last=True)
    val_loader = make_loader(val_ds, tc.batch_size, False, tc.num_workers) if len(val_ds) else None
    preview_batch = next(iter(make_loader(val_ds if len(val_ds) else train_ds, 4, False, 0)))

    model = build_model(cfg).to(device)
    print(f"model parameters: {count_parameters(model) / 1e6:.1f} M")
    diffusion = RegionalDiffusion(cfg.diffusion, device)
    ema = EMA(model, tc.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / max(1, tc.warmup_iters)))

    use_amp = bool(tc.amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    start_it, best_val = 0, math.inf
    if tc.resume:
        ck = torch_load(tc.resume, device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_it, best_val = ck["iter"], ck.get("best_val", math.inf)
        print(f"resumed from {tc.resume} at iteration {start_it}")

    log_path = out / "log.csv"
    new_log = not log_path.exists() or not tc.resume
    log_f = log_path.open("w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["iter", "loss", "loss_stage1", "loss_stage2", "val_loss", "lr", "sec_per_iter"])

    def checkpoint(it, name, full=True):
        """full: everything needed to resume. Otherwise only the EMA weights (smaller files),
        which is all evaluate.py / infer.py need."""
        state = dict(ema=ema.state_dict(), iter=it, best_val=best_val, config=to_dict(cfg))
        if full:
            state.update(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict())
        save_checkpoint(out / name, **state)

    model.train()
    batches = infinite(train_loader)
    acc = {"loss": 0.0, "s1": 0.0, "n1": 0, "s2": 0.0, "n2": 0, "n": 0}
    t0 = time.time()
    for it in range(start_it + 1, tc.iters + 1):
        b = next(batches)
        x0, M_b = b["x0"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
        x_t, Tmap, t_roi, t_bg, stage, eps = diffusion.training_batch(x0, M_b)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            eps_hat = model(x_t, M_b, Tmap, t_roi, t_bg, stage)
        loss, per = diffusion.loss(eps, eps_hat.float(), M_b, stage)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model)

        acc["loss"] += loss.item()
        acc["n"] += 1
        s1 = stage == 1
        acc["s1"] += per[s1].sum().item()
        acc["n1"] += int(s1.sum())
        acc["s2"] += per[~s1].sum().item()
        acc["n2"] += int((~s1).sum())

        val = ""
        if val_loader is not None and it % tc.val_every == 0:
            val = validation_loss(ema.model, diffusion, val_loader, device)
            if val < best_val:
                best_val = val
                checkpoint(it, "ckpt_best.pt", full=False)
        if it % tc.log_every == 0 or val != "":
            spi = (time.time() - t0) / max(acc["n"], 1)
            row = [it, acc["loss"] / max(acc["n"], 1), acc["s1"] / max(acc["n1"], 1),
                   acc["s2"] / max(acc["n2"], 1), val, sched.get_last_lr()[0], round(spi, 4)]
            log.writerow(row)
            log_f.flush()
            vtxt = f"  val {val:.4f}" if val != "" else ""
            print(f"it {it:7d}  loss {row[1]:.4f}  (stage1 {row[2]:.4f}, stage2 {row[3]:.4f}){vtxt}"
                  f"  {spi:.3f}s/it", flush=True)
            acc = {k: 0 for k in acc}
            t0 = time.time()
        if it % tc.ckpt_every == 0 or it == tc.iters:
            checkpoint(it, "ckpt_last.pt")
            checkpoint(it, f"ckpt_{it:07d}.pt", full=False)
        if it % tc.preview_every == 0 or it == tc.iters:
            save_preview(ema.model, diffusion, preview_batch, M, rods, cfg, device,
                         out / "previews" / f"preview_{it:07d}.png")
    log_f.close()
    if val_loader is None:          # no validation split: "best" = last
        checkpoint(tc.iters, "ckpt_best.pt", full=False)
    print(f"done. checkpoints in {out}")


if __name__ == "__main__":
    main()
