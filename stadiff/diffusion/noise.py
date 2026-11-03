"""
Noise generators: Gaussian or Simplex.

[StaDiff-V3 ≠ DDPM] A standard DDPM always uses i.i.d. Gaussian noise. StaDiff-V3 can also use
Simplex noise (as in AnoDDPM): smooth, spatially correlated noise that erases larger structures
(a bent rod, a missing weld) at a given noise level better than per-pixel Gaussian noise.

Rules that keep the maths consistent:
  * One noise TYPE at a time for the whole image (both ROI and background).
  * One noise FIELD drawn over the whole image; only the noise LEVEL differs between regions,
    so there is no seam at the mask boundary.
  * The same type is used for the forward corruption, the training target eps, and the random
    term of every reverse step. A model trained with one type cannot denoise the other.
  * Simplex fields are normalised to zero mean and unit variance per image, so the usual
    q(x_t | x_0) = sqrt(abar) x_0 + sqrt(1 - abar) eps still holds marginally.
"""
from __future__ import annotations

import math

import torch

_F2 = 0.5 * (math.sqrt(3.0) - 1.0)
_G2 = (3.0 - math.sqrt(3.0)) / 6.0
_GX = [1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0]
_GY = [1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1]


def simplex2d(x, y, perm, gx, gy):
    """Classic 2D simplex noise (Gustavson), vectorised in torch. Values in about [-1, 1].
    x, y: float tensors of equal shape. perm: long tensor (512,). gx, gy: float tensors (12,)."""
    s = (x + y) * _F2
    i = torch.floor(x + s)
    j = torch.floor(y + s)
    t = (i + j) * _G2
    x0 = x - (i - t)
    y0 = y - (j - t)
    upper = x0 > y0
    i1 = upper.long()
    j1 = 1 - i1
    x1 = x0 - i1 + _G2
    y1 = y0 - j1 + _G2
    x2 = x0 - 1.0 + 2.0 * _G2
    y2 = y0 - 1.0 + 2.0 * _G2

    ii = i.long() & 255
    jj = j.long() & 255
    g0 = perm[ii + perm[jj]] % 12
    g1 = perm[ii + i1 + perm[jj + j1]] % 12
    g2 = perm[ii + 1 + perm[jj + 1]] % 12

    def corner(g, cx, cy):
        tt = (0.5 - cx * cx - cy * cy).clamp_min(0.0)
        return tt ** 4 * (gx[g] * cx + gy[g] * cy)

    return 70.0 * (corner(g0, x0, y0) + corner(g1, x1, y1) + corner(g2, x2, y2))


class NoiseSampler:
    """noise(like) -> tensor shaped like `like`, zero mean, unit variance."""

    def __init__(self, noise_type="gaussian", simplex_scale=32.0, simplex_octaves=6,
                 simplex_persistence=0.8):
        if noise_type not in ("gaussian", "simplex"):
            raise ValueError(f"noise_type must be 'gaussian' or 'simplex', got {noise_type!r}")
        if simplex_octaves < 1 or simplex_scale <= 0 or not (0 < simplex_persistence <= 1):
            raise ValueError("need simplex_octaves >= 1, simplex_scale > 0, 0 < persistence <= 1")
        self.noise_type = noise_type
        self.scale = float(simplex_scale)
        self.octaves = int(simplex_octaves)
        self.persistence = float(simplex_persistence)

    @classmethod
    def from_cfg(cls, d):
        return cls(d["noise_type"], d["simplex_scale"], d["simplex_octaves"], d["simplex_persistence"])

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "gaussian":
            return torch.randn_like(like)
        B, C, H, W = like.shape
        return self.simplex_field(B * C, H, W, like.device).view(B, C, H, W).to(like.dtype)

    def simplex_field(self, N, H, W, device="cpu") -> torch.Tensor:
        gx = torch.tensor(_GX, dtype=torch.float32, device=device)
        gy = torch.tensor(_GY, dtype=torch.float32, device=device)
        ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
        total = torch.zeros(N, H, W, device=device)
        amp = 1.0
        for o in range(self.octaves):
            f = (2 ** o) / self.scale
            off = torch.rand(N, 2, 1, 1, device=device) * 256.0     # new random field per image
            p = torch.randperm(256, device=device)
            perm = torch.cat([p, p])
            total = total + amp * simplex2d(xs * f + off[:, 0], ys * f + off[:, 1], perm, gx, gy)
            amp *= self.persistence
        total = total - total.mean(dim=(1, 2), keepdim=True)
        return total / total.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
