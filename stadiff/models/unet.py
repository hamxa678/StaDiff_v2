"""
StaDiff-V3 dual-stream context-aware U-Net (noise predictor eps_theta).

        x_t ⊙ M, M, T/T_total ──► ROI encoder (heavy) ──┐ skips (red)
                                     ▲  GatedFusion (H/2, H/4)        │
                                     ▲  CrossAttention (H/8, H/16)    ├─► shared decoder ─► eps_hat (whole image)
        x_t ⊙ (1-M), 1-M, T/T_total ► context encoder (light) ┘ skips (blue)

[StaDiff-V3 ≠ DDPM] — what differs from the standard DDPM U-Net
  1. Inputs. DDPM: the noisy image x_t only. Here: two masked copies of x_t, each with the
     mask and the per-pixel noise-level map T(i, j) as extra channels (3 channels per stream).
  2. Two encoders. DDPM has one. Here a heavy ROI encoder looks at the heavily noised rods
     and a light context encoder at the lightly noised background.
  3. Information exchange between encoders: gated fusion at high resolution, cross-attention
     (ROI queries, context keys/values) at low resolution and in the bottleneck.
  4. Two sets of skip connections (ROI skips and context skips) concatenated in ONE shared
     decoder. DDPM has one encoder -> one set of skips.
  5. Conditioning on (t_roi, t_bg, stage) instead of a single t (see ConditionEmbedding).
  The output is the same as DDPM: a noise prediction for every pixel of the image.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (ConditionEmbedding, CrossAttention2D, Downsample, GatedFusion, ResBlock,
                     SelfAttention2D, Upsample, group_norm)


class StreamEncoder(nn.Module):
    """One encoder stream: stem conv at full resolution, then per level: downsample + ResBlocks.
    Level i works at resolution H / 2**(i+1). The fusion between the two streams happens in
    StaDiffUNet.forward, between levels, so this class only holds the layers."""

    def __init__(self, in_ch, channels, n_blocks, emb_dim, dropout):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, channels[0], 3, padding=1)
        self.downs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.downs.append(Downsample(prev))
            self.blocks.append(nn.ModuleList(
                [ResBlock(prev if j == 0 else ch, ch, emb_dim, dropout) for j in range(n_blocks)]))
            prev = ch

    def level(self, i, h, emb):
        h = self.downs[i](h)
        for blk in self.blocks[i]:
            h = blk(h, emb)
        return h


class StaDiffUNet(nn.Module):
    def __init__(self, mcfg, T_total: int):
        super().__init__()
        self.T_total = float(T_total)
        rc, cc, dc = list(mcfg["roi_channels"]), list(mcfg["ctx_channels"]), list(mcfg["dec_channels"])
        L = len(rc)
        self.L = L
        self.use_ctx = bool(mcfg["use_context_stream"])
        emb, drop, heads = mcfg["emb_dim"], mcfg["dropout"], mcfg["attn_heads"]
        if not self.use_ctx:
            cc = [0] * L

        self.cond = ConditionEmbedding(emb)

        # ---- two encoders  [StaDiff-V3 ≠ DDPM: DDPM has a single encoder]
        self.roi_enc = StreamEncoder(3, rc, mcfg["roi_blocks"], emb, drop)   # heavy
        self.ctx_enc = StreamEncoder(3, cc, mcfg["ctx_blocks"], emb, drop) if self.use_ctx else None

        # ---- encoder-to-encoder exchange per level  [StaDiff-V3 ≠ DDPM]
        self.fusion_kinds = [mcfg["fusion"][i] if self.use_ctx else "none" for i in range(L)]
        self.fusions = nn.ModuleList()
        for i, kind in enumerate(self.fusion_kinds):
            if kind == "gated":
                self.fusions.append(GatedFusion(rc[i], cc[i]))
            elif kind == "cross":
                self.fusions.append(CrossAttention2D(rc[i], cc[i], heads))
            else:
                self.fusions.append(nn.Identity())     # ablation: no exchange at this level

        # ---- bottleneck (H/16): merge streams, self-attention, cross-attention to context
        bc = rc[-1]
        self.bot_in = nn.Conv2d(rc[-1] + cc[-1], bc, 1)        # concatenation + projection
        self.bot_res1 = ResBlock(bc, bc, emb, drop)
        self.bot_self = SelfAttention2D(bc, heads)
        self.bot_cross = (CrossAttention2D(bc, cc[-1], heads)
                          if self.use_ctx and mcfg["bottleneck_cross_attn"] else None)
        self.bot_res2 = ResBlock(bc, bc, emb, drop)

        # ---- ONE shared decoder fed by BOTH skip sets  [StaDiff-V3 ≠ DDPM]
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        prev = bc
        for i in reversed(range(L)):
            in_ch = prev + rc[i] + cc[i]          # concat(h, ROI skip, context skip)
            self.dec_blocks.append(nn.ModuleList(
                [ResBlock(in_ch if j == 0 else dc[i], dc[i], emb, drop) for j in range(mcfg["dec_blocks"])]))
            self.ups.append(Upsample(dc[i]))
            prev = dc[i]
        # full-resolution head also sees both stems (keeps fine weld texture)
        self.final = ResBlock(prev + rc[0] + cc[0], dc[0], emb, drop)
        self.out_norm = group_norm(dc[0])
        self.out = nn.Conv2d(dc[0], 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t, M, Tmap, t_roi, t_bg, stage):
        """x_t, M, Tmap: (B,1,H,W). t_roi, t_bg, stage: (B,). Returns eps_hat (B,1,H,W)."""
        emb = self.cond(t_roi, t_bg, stage)
        tm = Tmap / self.T_total

        # [StaDiff-V3 ≠ DDPM] the ROI stream sees only the ROI, the context stream only the background
        r = self.roi_enc.stem(torch.cat([x_t * M, M, tm], dim=1))
        c = self.ctx_enc.stem(torch.cat([x_t * (1 - M), 1 - M, tm], dim=1)) if self.use_ctx else None
        stems = (r, c)

        roi_skips, ctx_skips = [], []
        for i in range(self.L):
            r = self.roi_enc.level(i, r, emb)
            if self.use_ctx:
                c = self.ctx_enc.level(i, c, emb)
                if self.fusion_kinds[i] != "none":
                    r = self.fusions[i](r, c)          # context -> ROI (context is not modified)
            roi_skips.append(r)                         # red skip connections
            ctx_skips.append(c)                         # blue skip connections

        h = self.bot_in(torch.cat([r, c], dim=1) if self.use_ctx else r)
        h = self.bot_res1(h, emb)
        h = self.bot_self(h)
        if self.bot_cross is not None:
            h = self.bot_cross(h, c)
        h = self.bot_res2(h, emb)

        for k, i in enumerate(reversed(range(self.L))):
            parts = [h, roi_skips[i]] + ([ctx_skips[i]] if self.use_ctx else [])
            h = torch.cat(parts, dim=1)
            for blk in self.dec_blocks[k]:
                h = blk(h, emb)
            h = self.ups[k](h)

        parts = [h, stems[0]] + ([stems[1]] if self.use_ctx else [])
        h = self.final(torch.cat(parts, dim=1), emb)
        return self.out(F.silu(self.out_norm(h)))


def build_model(cfg) -> StaDiffUNet:
    return StaDiffUNet(cfg.model, cfg.diffusion.T_total)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
