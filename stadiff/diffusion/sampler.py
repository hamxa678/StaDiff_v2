"""
Two-stage reconstruction of a test image.

[StaDiff-V3 ≠ DDPM] A DDPM anomaly detector (e.g. AnoDDPM) noises the whole image to ONE level
t and denoises the whole image back to 0. StaDiff-V3 instead:

  0. Forward-noises the input regionally:  ROI -> t_roi_max (heavy, erases defects),
     background -> t_bg (light, keeps the scene as context).
  1. Stage 1 (s = 1): denoises ONLY the ROI from t_roi_max down to t_bg, the background
     stays frozen at t_bg:      x_{t-1} = M * x_hat_{t-1} + (1 - M) * x_t
     When stage 1 ends, ROI and background are both at t_bg: a uniformly noised image.
  2. Stage 2 (s = 2): ordinary whole-image denoising from t_bg down to 0, same network.

The model never saw anomalies, so it rebuilds normal straight rods with one centred weld;
|input - reconstruction| inside the ROI is the anomaly map.
"""
from __future__ import annotations

import numpy as np
import torch


def _schedule(t_start: int, t_end: int, n_steps: int | None) -> list[tuple[int, int]]:
    """Pairs (t, t_next) from t_start down to t_end. n_steps=None -> every integer step."""
    span = t_start - t_end
    if span <= 0:
        return []
    if n_steps is None or n_steps >= span:
        ts = list(range(t_start, t_end - 1, -1))
    else:
        ts = np.unique(np.round(np.linspace(t_start, t_end, n_steps + 1)).astype(int))[::-1].tolist()
    return list(zip(ts[:-1], ts[1:]))


@torch.no_grad()
def reconstruct(model, diffusion, x0, M, t_roi_max: int, t_bg: int, sampler: str = "ddim",
                steps_stage1: int | None = 50, steps_stage2: int | None = 50):
    """x0, M: (B, 1, H, W). Returns the reconstruction x_hat (B, 1, H, W) in [-1, 1].

    sampler="ddpm": ancestral sampling over every step (t_roi_max + ... network calls, slow).
    sampler="ddim": deterministic DDIM with steps_stage1 + steps_stage2 network calls.
    """
    diffusion.check_inference(t_roi_max, t_bg)
    B, dev = x0.shape[0], x0.device
    full = lambda v: torch.full((B,), int(v), device=dev, dtype=torch.long)

    # 0. regional forward noising (one noise field, two levels)
    T0 = diffusion.timestep_map(full(t_roi_max), full(t_bg), M)
    x = diffusion.q_sample(x0, T0, diffusion.noise(x0))

    use_ddpm = sampler == "ddpm"
    if sampler not in ("ddpm", "ddim"):
        raise ValueError(f"unknown sampler: {sampler}")

    # 1. Stage 1: ROI only, t_roi_max -> t_bg, background frozen at t_bg
    for t, t_next in _schedule(t_roi_max, t_bg, None if use_ddpm else steps_stage1):
        Tm = diffusion.timestep_map(full(t), full(t_bg), M)
        eps_hat = model(x, M, Tm, full(t), full(t_bg), full(1)).float()
        if use_ddpm:
            x = diffusion.ddpm_step(x, Tm, eps_hat, update_mask=M)
        else:
            Tn = diffusion.timestep_map(full(t_next), full(t_bg), M)
            x = diffusion.ddim_step(x, Tm, Tn, eps_hat, update_mask=M)

    # 2. Stage 2: whole image, t_bg -> 0
    ones = torch.ones_like(M)
    for t, t_next in _schedule(t_bg, 0, None if use_ddpm else steps_stage2):
        Tm = torch.full_like(M, float(t))
        eps_hat = model(x, M, Tm, full(t), full(t), full(2)).float()
        if use_ddpm:
            x = diffusion.ddpm_step(x, Tm, eps_hat, update_mask=ones)
        else:
            x = diffusion.ddim_step(x, Tm, torch.full_like(M, float(t_next)), eps_hat, update_mask=ones)
    return x.clamp(-1, 1)
