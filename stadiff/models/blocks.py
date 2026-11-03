"""
Building blocks of the dual-stream U-Net.

Standard diffusion-U-Net parts (same idea as in DDPM / improved DDPM):
    SinusoidalEmbedding, ResBlock (with FiLM conditioning), Downsample, Upsample, SelfAttention2D
StaDiff-V3 additions, marked [StaDiff-V3 ≠ DDPM]:
    ConditionEmbedding, GatedFusion, CrossAttention2D
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(ch: int) -> nn.GroupNorm:
    g = 32
    while ch % g:
        g //= 2
    return nn.GroupNorm(g, ch)


def zero_module(m: nn.Module) -> nn.Module:
    """Zero-initialise a layer so a residual branch starts as the identity."""
    for p in m.parameters():
        p.detach().zero_()
    return m


class SinusoidalEmbedding(nn.Module):
    """Transformer-style embedding of a scalar timestep (as in DDPM)."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim, self.max_period = dim, max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) *
                          torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionEmbedding(nn.Module):
    """[StaDiff-V3 ≠ DDPM] DDPM conditions on ONE timestep t. In stage 1 the image carries two
    noise levels at once, so the network is told both of them plus which stage it is in:
        e = MLP([ E(t_roi), E(t_bg), E_stage(s) ])
    In stage 2, t_roi = t_bg = t."""

    def __init__(self, emb_dim: int, base_dim: int = 128, n_stages: int = 2):
        super().__init__()
        self.t_embed = SinusoidalEmbedding(base_dim)
        self.stage_embed = nn.Embedding(n_stages + 1, base_dim)   # index 1 = stage 1, 2 = stage 2
        self.mlp = nn.Sequential(nn.Linear(3 * base_dim, emb_dim), nn.SiLU(),
                                 nn.Linear(emb_dim, emb_dim))

    def forward(self, t_roi, t_bg, stage):
        e = torch.cat([self.t_embed(t_roi), self.t_embed(t_bg), self.stage_embed(stage.long())], -1)
        return self.mlp(e)


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv x2 with the conditioning injected as scale/shift (FiLM)."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = group_norm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


def _heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, heads, H*W, C/heads)"""
    B, C, H, W = x.shape
    return x.reshape(B, n_heads, C // n_heads, H * W).transpose(-1, -2)


class SelfAttention2D(nn.Module):
    """Multi-head self-attention over spatial positions (used in the bottleneck, as in DDPM)."""

    def __init__(self, ch: int, n_heads: int = 4):
        super().__init__()
        assert ch % n_heads == 0, "channels must be divisible by attn_heads"
        self.n_heads = n_heads
        self.norm = group_norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = zero_module(nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        o = F.scaled_dot_product_attention(_heads(q, self.n_heads), _heads(k, self.n_heads),
                                           _heads(v, self.n_heads))
        o = o.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(o)


class CrossAttention2D(nn.Module):
    """[StaDiff-V3 ≠ DDPM] Cross-attention between the two encoder streams.
    Queries come from the ROI features, keys/values from the context features, so every ROI
    position can look anywhere in the background (e.g. where the neighbouring rods are) to decide
    what a normal rod should look like there. Only the ROI stream is updated; the context stream
    is left unchanged. Zero-initialised output -> starts as identity."""

    def __init__(self, q_ch: int, kv_ch: int, n_heads: int = 4):
        super().__init__()
        assert q_ch % n_heads == 0, "ROI channels must be divisible by attn_heads"
        self.n_heads = n_heads
        self.norm_q = group_norm(q_ch)
        self.norm_kv = group_norm(kv_ch)
        self.q = nn.Conv2d(q_ch, q_ch, 1)
        self.kv = nn.Conv2d(kv_ch, 2 * q_ch, 1)
        self.proj = zero_module(nn.Conv2d(q_ch, q_ch, 1))

    def forward(self, roi, ctx):
        B, C, H, W = roi.shape
        q = _heads(self.q(self.norm_q(roi)), self.n_heads)
        k, v = self.kv(self.norm_kv(ctx)).chunk(2, dim=1)
        o = F.scaled_dot_product_attention(q, _heads(k, self.n_heads), _heads(v, self.n_heads))
        o = o.transpose(-1, -2).reshape(B, C, H, W)
        return roi + self.proj(o)


class GatedFusion(nn.Module):
    """[StaDiff-V3 ≠ DDPM] Local context injection at high resolution (where attention would be
    too expensive). The context features are projected to the ROI width and added through a
    learned per-pixel, per-channel gate:
        roi_out = roi + sigmoid(gate([roi, P(ctx)])) * P(ctx)
    P is zero-initialised, so at the start of training the two streams are independent and the
    network learns how much context to let in."""

    def __init__(self, roi_ch: int, ctx_ch: int):
        super().__init__()
        self.proj = zero_module(nn.Conv2d(ctx_ch, roi_ch, 1))
        self.gate = nn.Conv2d(2 * roi_ch, roi_ch, 3, padding=1)

    def forward(self, roi, ctx):
        c = self.proj(ctx)
        g = torch.sigmoid(self.gate(torch.cat([roi, c], dim=1)))
        return roi + g * c
