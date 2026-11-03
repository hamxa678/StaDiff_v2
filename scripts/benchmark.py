"""
Measure inference speed on the machine that will run in production (e.g. a CPU-only PC).

    python scripts/benchmark.py --device cpu
    python scripts/benchmark.py --device cpu --threads 8 --presets default small tiny
    python scripts/benchmark.py --device cpu --checkpoint runs/simplex_500_200/ckpt_best.pt

It times ONE network pass for each model size (weights do not matter for speed, so no training is
needed for the presets) and then estimates the time per part for different numbers of
reconstruction steps:  time per part ~ preprocessing + (steps_stage1 + steps_stage2) x one pass.
Use the result to decide which model size and step count can meet your cycle time.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stadiff.config import load_config                    # noqa: E402
from stadiff.data.preprocess import normalize, resize_gray  # noqa: E402
from stadiff.models import build_model, count_parameters  # noqa: E402
from stadiff.utils import torch_load                       # noqa: E402

PRESETS = {
    "default": [],
    "small": ["model.roi_channels=[32, 64, 128, 256]", "model.ctx_channels=[16, 32, 64, 128]",
              "model.dec_channels=[32, 32, 64, 128]", "model.roi_blocks=1", "model.dec_blocks=1",
              "model.emb_dim=128"],
    "tiny": ["model.roi_channels=[16, 32, 64, 128]", "model.ctx_channels=[8, 16, 32, 64]",
             "model.dec_channels=[16, 16, 32, 64]", "model.roi_blocks=1", "model.dec_blocks=1",
             "model.emb_dim=64", "model.attn_heads=2"],
}
STEP_SETTINGS = [(1, 1), (2, 2), (5, 5), (10, 10), (25, 25), (50, 50)]


def time_pass(model, H, W, device, repeat, dtype=None):
    x = torch.randn(1, 1, H, W, device=device)
    M = torch.zeros_like(x)
    M[..., W // 4: W // 2] = 1
    Tm = M * 500 + (1 - M) * 200
    tr, tb, st = (torch.tensor([500], device=device), torch.tensor([200], device=device),
                  torch.tensor([1], device=device))
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if dtype == "fp16" else torch.no_grad()
    with torch.no_grad(), ctx:
        for _ in range(3):                    # warm-up
            model(x, M, Tm, tr, tb, st)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            model(x, M, Tm, tr, tb, st)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: PyTorch's choice)")
    ap.add_argument("--presets", nargs="*", default=["default", "small", "tiny"], choices=list(PRESETS))
    ap.add_argument("--checkpoint", default=None, help="also time this trained model")
    ap.add_argument("--image-size", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--budget", type=float, default=1.0, help="seconds available per part (default 1.0)")
    a = ap.parse_args()

    device = torch.device(a.device)
    if a.threads:
        torch.set_num_threads(a.threads)
    H, W = a.image_size
    print(f"device {device}" + (f", {torch.get_num_threads()} CPU threads, {os.cpu_count()} cores visible"
                                if device.type == "cpu" else f", {torch.cuda.get_device_name(device)}"))

    # preprocessing cost (resize a camera-sized frame + normalise)
    frame = np.random.default_rng(0).uniform(0, 255, (1200, 1600)).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(10):
        normalize(resize_gray(frame, (H, W), "area"))
    pre = (time.perf_counter() - t0) / 10
    print(f"preprocessing one 1600x1200 frame: {pre * 1000:.1f} ms\n")

    models = []
    for p in a.presets:
        cfg = load_config(None, [f"data.image_size=[{H}, {W}]"] + PRESETS[p])
        models.append((p, build_model(cfg)))
    if a.checkpoint:
        ck = torch_load(a.checkpoint, "cpu")
        cfg = load_config(None, base=ck["config"])
        m = build_model(cfg)
        m.load_state_dict(ck["ema"]["model"])
        models.append((f"checkpoint ({Path(a.checkpoint).name})", m))

    header = "  ".join(f"{s1}+{s2}".rjust(8) for s1, s2 in STEP_SETTINGS)
    print(f"{'model':28s} {'params':>8s} {'1 pass':>9s}   seconds per part for steps stage1+stage2:")
    print(f"{'':28s} {'':>8s} {'':>9s}   {header}")
    for name, m in models:
        m = m.to(device).eval()
        t = time_pass(m, H, W, device, a.repeat, "fp16" if device.type == "cuda" else None)
        cells = []
        for s1, s2 in STEP_SETTINGS:
            total = pre + (s1 + s2) * t
            cells.append((f"{total:.2f}" + ("*" if total <= a.budget else " ")).rjust(8))
        print(f"{name:28s} {count_parameters(m) / 1e6:7.1f}M {t * 1000:7.0f}ms   " + "  ".join(cells))
    print(f"\n* = fits the budget of {a.budget:g} s per part (frame selection not included)")
    print("Few-step settings only help if detection quality holds: check with evaluate.py "
          "--steps-stage1 N --steps-stage2 N on a GPU first.")


if __name__ == "__main__":
    main()
