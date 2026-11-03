"""Network shapes, gradients, ablation switches, the full sampler and one training step."""
import pytest
import torch

from stadiff.config import load_config
from stadiff.diffusion import RegionalDiffusion, reconstruct
from stadiff.evaluation.anomaly import anomaly_map, score_batch
from stadiff.models import build_model
from tests.conftest import TINY


def _inputs(B=2, H=64):
    x = torch.randn(B, 1, H, H)
    M = torch.zeros(B, 1, H, H)
    M[..., 10:20] = 1
    Tm = M * 25 + (1 - M) * 12
    return x, M, Tm, torch.full((B,), 25), torch.full((B,), 12), torch.ones(B, dtype=torch.long)


def test_forward_shape_and_zero_init(tiny_cfg):
    model = build_model(tiny_cfg)
    out = model(*_inputs())
    assert out.shape == (2, 1, 64, 64)
    assert torch.count_nonzero(out) == 0          # zero-initialised output layer


def test_backward_reaches_both_streams(tiny_cfg):
    model = build_model(tiny_cfg)
    d = RegionalDiffusion(tiny_cfg.diffusion)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    x0 = torch.rand(2, 1, 64, 64) * 2 - 1
    M = _inputs()[1]
    for _ in range(3):                             # a few steps so zero-init layers open up
        x_t, Tm, tr, tb, s, eps = d.training_batch(x0, M)
        loss, _ = d.loss(eps, model(x_t, M, Tm, tr, tb, s), M, s)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    assert model.roi_enc.stem.weight.grad.abs().sum() > 0
    assert model.ctx_enc.stem.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("extra", [["model.use_context_stream=false"],
                                   ["model.fusion=[none, none, none, none]"],
                                   ["model.fusion=[cross, cross, gated, gated]"]])
def test_ablation_variants(extra):
    cfg = load_config(None, TINY + extra)
    assert build_model(cfg)(*_inputs()).shape == (2, 1, 64, 64)


@pytest.mark.parametrize("sampler", ["ddim", "ddpm"])
@pytest.mark.parametrize("noise", ["gaussian", "simplex"])
def test_two_stage_reconstruction_and_scores(tiny_cfg, sampler, noise):
    tiny_cfg.diffusion.noise_type = noise
    tiny_cfg.diffusion.simplex_scale = 8.0
    model = build_model(tiny_cfg).eval()
    d = RegionalDiffusion(tiny_cfg.diffusion)
    x0 = torch.rand(2, 1, 64, 64) * 2 - 1
    M = _inputs()[1]
    x_hat = reconstruct(model, d, x0, M, 30, 12, sampler, 4, 4)
    assert x_hat.shape == x0.shape and torch.isfinite(x_hat).all()
    amap = anomaly_map(x0, x_hat, "l1+ssim", 1.0)
    rods = torch.zeros(3, 64, 64)
    rods[0, :, 10:13], rods[1, :, 13:16], rods[2, :, 16:20] = 1, 1, 1
    img, per_rod = score_batch(amap, M[:1], rods)
    assert img.shape == (2,) and per_rod.shape == (2, 3)
