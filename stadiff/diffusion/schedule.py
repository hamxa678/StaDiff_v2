"""
Regional diffusion process: forward noising, training targets, losses, reverse steps.

Notation: index t = 0 is the clean image, index t = k is the image after k noising steps.
All coefficient tables are padded so that table[t] belongs to step t.

[StaDiff-V3 ≠ DDPM] — summary of what this file changes
  1. Per-pixel timesteps. In DDPM every pixel shares one t. Here each pixel has its own
     t(i, j): t_roi inside the ROI mask and t_bg outside (the "timestep map" T(i, j)).
     Every coefficient (alpha_bar, beta, posterior terms) is looked up per pixel.
  2. Two training modes ("stages") sampled in every batch:
        stage 1: background fixed at t_bg, ROI at t_roi ~ U{t_bg+1, ..., t_roi_max}
        stage 2: whole image at one t ~ U{1, ..., t_bg}
     These cover exactly the states the two-stage sampler visits.
  3. Two losses: stage 1 is computed on ROI pixels only (background predictions are thrown
     away by the sampler); stage 2 is whole-image with extra weight lambda on the ROI.
  4. Masked reverse step: a step can update only some pixels (the ROI in stage 1) and leave
     the rest unchanged (the background, kept at t_bg as context).
  5. t_roi_max and t_bg are user settings, not constants. With random_bg=True the model is
     trained over a range of t_bg, so t_bg can be changed at test time without retraining.
"""
from __future__ import annotations

import math

import torch

from .noise import NoiseSampler


def make_betas(T, schedule="linear", beta_start=1e-4, beta_end=2e-2) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    if schedule == "cosine":   # Nichol & Dhariwal 2021
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        abar = f / f[0]
        return (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    raise ValueError(f"unknown beta_schedule: {schedule}")


class RegionalDiffusion:
    def __init__(self, dcfg, device="cpu"):
        self.cfg = dcfg
        self.T = int(dcfg["T_total"])
        self.t_roi_max = int(dcfg["t_roi_max"])
        self.t_bg = int(dcfg["t_bg"])
        if dcfg.get("random_bg"):
            self.t_bg_min = int(dcfg["t_bg_min"] if dcfg["t_bg_min"] is not None else self.t_bg)
            self.t_bg_max = int(dcfg["t_bg_max"] if dcfg["t_bg_max"] is not None else self.t_bg)
        else:
            self.t_bg_min = self.t_bg_max = self.t_bg
        self.p_stage1 = float(dcfg["p_stage1"])
        self.lam = float(dcfg["lam"])
        self._check_training_ranges()

        betas = make_betas(self.T, dcfg["beta_schedule"], dcfg["beta_start"], dcfg["beta_end"])
        betas = torch.cat([torch.zeros(1, dtype=torch.float64), betas])   # betas[0] = 0 (clean)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)                                   # abar[0] = 1
        abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
        denom = (1.0 - abar).clamp_min(1e-12)
        tables = {
            "betas": betas,
            "alphas": alphas,
            "abar": abar,
            "post_var": betas * (1.0 - abar_prev) / denom,
            "post_c0": betas * abar_prev.sqrt() / denom,            # weight of x0_hat in the mean
            "post_ct": (1.0 - abar_prev) * alphas.sqrt() / denom,   # weight of x_t in the mean
        }
        self.tables = {k: v.float() for k, v in tables.items()}
        self.to(device)
        self.noise = NoiseSampler.from_cfg(dcfg)

    def to(self, device):
        self.device = torch.device(device)
        self.tables = {k: v.to(self.device) for k, v in self.tables.items()}
        return self

    def __getattr__(self, k):
        tables = self.__dict__.get("tables", {})
        if k in tables:
            return tables[k]
        raise AttributeError(k)

    def _check_training_ranges(self):
        if not (1 <= self.t_bg_min <= self.t_bg <= self.t_bg_max):
            raise ValueError(f"need 1 <= t_bg_min <= t_bg <= t_bg_max, got "
                             f"{self.t_bg_min}, {self.t_bg}, {self.t_bg_max}")
        if not (self.t_bg_max < self.t_roi_max <= self.T):
            raise ValueError(f"need t_bg_max < t_roi_max <= T_total, got "
                             f"{self.t_bg_max}, {self.t_roi_max}, {self.T}")
        if not 0.0 <= self.p_stage1 <= 1.0:
            raise ValueError("p_stage1 must be in [0, 1]")

    def check_inference(self, t_roi_max: int, t_bg: int):
        """The model only knows the noise levels it was trained on."""
        if t_roi_max > self.t_roi_max:
            raise ValueError(f"t_roi_max={t_roi_max} exceeds the trained maximum {self.t_roi_max}")
        if not self.t_bg_min <= t_bg <= self.t_bg_max:
            raise ValueError(f"t_bg={t_bg} outside the trained range [{self.t_bg_min}, {self.t_bg_max}]."
                             " Train with diffusion.random_bg=true to vary it.")
        if not t_bg < t_roi_max:
            raise ValueError(f"need t_bg < t_roi_max, got {t_bg}, {t_roi_max}")

    # ------------------------------------------------------------------ training
    def sample_timesteps(self, B: int):
        """[StaDiff-V3 ≠ DDPM] DDPM draws one t ~ U{1..T}. Here each sample is a stage-1 or
        stage-2 sample with its own (t_roi, t_bg)."""
        dev = self.device
        t_bg = torch.randint(self.t_bg_min, self.t_bg_max + 1, (B,), device=dev)
        stage = torch.where(torch.rand(B, device=dev) < self.p_stage1,
                            torch.ones(B, dtype=torch.long, device=dev),
                            torch.full((B,), 2, dtype=torch.long, device=dev))
        # stage 1: t_roi ~ U{t_bg+1, ..., t_roi_max}
        span1 = self.t_roi_max - t_bg
        t_roi_s1 = t_bg + 1 + torch.minimum((torch.rand(B, device=dev) * span1).long(), span1 - 1)
        # stage 2: t ~ U{1, ..., t_bg} for both regions
        t_s2 = 1 + torch.minimum((torch.rand(B, device=dev) * t_bg).long(), t_bg - 1)
        t_roi = torch.where(stage == 1, t_roi_s1, t_s2)
        t_bgv = torch.where(stage == 1, t_bg, t_s2)
        return t_roi, t_bgv, stage

    @staticmethod
    def timestep_map(t_roi, t_bg, M):
        """[StaDiff-V3 ≠ DDPM] T(i, j) = t_roi inside the mask, t_bg outside. M: (B,1,H,W) in {0,1}."""
        return M * t_roi.view(-1, 1, 1, 1).float() + (1 - M) * t_bg.view(-1, 1, 1, 1).float()

    def q_sample(self, x0, Tmap, eps):
        """Forward process with a per-pixel noise level (DDPM uses one abar per image)."""
        ab = self.abar[Tmap.long()]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    def training_batch(self, x0, M):
        t_roi, t_bg, stage = self.sample_timesteps(x0.shape[0])
        Tmap = self.timestep_map(t_roi, t_bg, M)
        eps = self.noise(x0)            # ONE noise field over the whole image
        x_t = self.q_sample(x0, Tmap, eps)
        return x_t, Tmap, t_roi, t_bg, stage, eps

    def loss(self, eps, eps_hat, M, stage):
        """[StaDiff-V3 ≠ DDPM] DDPM: plain mean squared error over all pixels.
        Stage 1: MSE over ROI pixels only.  Stage 2: (1 + lambda * M)-weighted MSE over the image."""
        se = (eps - eps_hat) ** 2
        l1 = (M * se).flatten(1).sum(1) / M.flatten(1).sum(1).clamp_min(1.0)
        l2 = ((1 + self.lam * M) * se).flatten(1).mean(1)
        per_sample = torch.where(stage == 1, l1, l2)
        return per_sample.mean(), per_sample

    # ------------------------------------------------------------------ reverse steps
    def predict_x0(self, x_t, Tmap, eps_hat):
        ab = self.abar[Tmap.long()]
        return (x_t - (1 - ab).sqrt() * eps_hat) / ab.sqrt()

    def ddpm_step(self, x_t, Tmap, eps_hat, update_mask, clip=True):
        """One ancestral step t -> t-1, done per pixel with that pixel's own t.
        [StaDiff-V3 ≠ DDPM] only pixels with update_mask = 1 change; the rest keep their value."""
        t = Tmap.long()
        x0 = self.predict_x0(x_t, Tmap, eps_hat)
        if clip:
            x0 = x0.clamp(-1, 1)
        mean = self.post_c0[t] * x0 + self.post_ct[t] * x_t
        noise = self.noise(x_t)          # same noise type as training (Gaussian or simplex)
        x_prev = mean + (t > 1).float() * self.post_var[t].sqrt() * noise
        return update_mask * x_prev + (1 - update_mask) * x_t

    def ddim_step(self, x_t, Tmap, Tmap_next, eps_hat, update_mask, clip=True):
        """Deterministic DDIM jump t -> t_next (eta = 0), per pixel; lets you sample with far
        fewer network calls. Same masking rule as ddpm_step."""
        ab = self.abar[Tmap.long()]
        ab_next = self.abar[Tmap_next.long()]
        x0 = self.predict_x0(x_t, Tmap, eps_hat)
        if clip:
            x0 = x0.clamp(-1, 1)
            eps_hat = (x_t - ab.sqrt() * x0) / (1 - ab).sqrt().clamp_min(1e-8)
        x_next = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps_hat
        return update_mask * x_next + (1 - update_mask) * x_t
