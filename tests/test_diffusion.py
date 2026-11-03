"""Regional diffusion: timestep sampling, forward process, losses, masked reverse steps."""
import pytest
import torch

from stadiff.config import load_config
from stadiff.diffusion import NoiseSampler, RegionalDiffusion


@pytest.mark.parametrize("noise", ["gaussian", "simplex"])
def test_noise_is_standardised(noise):
    n = NoiseSampler(noise, simplex_scale=8)(torch.zeros(4, 1, 64, 64))
    assert n.shape == (4, 1, 64, 64)
    assert abs(n.mean().item()) < 0.1 and abs(n.std().item() - 1) < 0.1


def test_stage_timestep_ranges():
    d = RegionalDiffusion(load_config().diffusion)
    t_roi, t_bg, stage = d.sample_timesteps(20000)
    s1, s2 = stage == 1, stage == 2
    assert s1.any() and s2.any()
    assert (t_bg[s1] == 400).all() and (t_roi[s1] > 400).all() and (t_roi[s1] <= 800).all()
    assert t_roi[s1].min() == 401 and t_roi[s1].max() == 800        # whole stage-1 range covered
    assert (t_roi[s2] == t_bg[s2]).all() and t_roi[s2].min() == 1 and t_roi[s2].max() == 400


def test_random_bg_range():
    cfg = load_config(None, ["diffusion.random_bg=true", "diffusion.t_bg_min=200", "diffusion.t_bg_max=500",
                             "diffusion.t_roi_max=900"])
    d = RegionalDiffusion(cfg.diffusion)
    t_roi, t_bg, stage = d.sample_timesteps(20000)
    s1 = stage == 1
    assert t_bg[s1].min() >= 200 and t_bg[s1].max() <= 500 and (t_roi[s1] > t_bg[s1]).all()
    d.check_inference(700, 450)
    with pytest.raises(ValueError):
        d.check_inference(950, 400)       # above trained maximum


def test_invalid_levels_rejected():
    with pytest.raises(ValueError):
        RegionalDiffusion(load_config(None, ["diffusion.t_bg=800"]).diffusion)
    d = RegionalDiffusion(load_config().diffusion)
    with pytest.raises(ValueError):
        d.check_inference(800, 350)       # fixed-t_bg model asked for another t_bg


def test_forward_and_losses():
    d = RegionalDiffusion(load_config().diffusion)
    x0 = torch.rand(2, 1, 16, 16) * 2 - 1
    M = torch.zeros(2, 1, 16, 16)
    M[..., 4:8] = 1
    T0 = torch.zeros_like(M)
    assert torch.allclose(d.q_sample(x0, T0, torch.randn_like(x0)), x0)   # t = 0 is the clean image
    eps = torch.randn_like(x0)
    eps_hat = eps.clone()
    eps_hat[..., :, :4] += 5.0           # error only in the background
    stage = torch.tensor([1, 2])
    _, per = d.loss(eps, eps_hat, M, stage)
    assert per[0] == 0                   # stage 1 ignores background errors
    assert per[1] > 0                    # stage 2 does not


def test_masked_steps_leave_background_untouched():
    d = RegionalDiffusion(load_config().diffusion)
    x = torch.randn(2, 1, 16, 16)
    M = torch.zeros_like(x)
    M[..., 4:8] = 1
    Tm = M * 600 + (1 - M) * 400
    eps_hat = torch.randn_like(x)
    for y in (d.ddpm_step(x, Tm, eps_hat, M), d.ddim_step(x, Tm, M * 500 + (1 - M) * 400, eps_hat, M)):
        assert torch.equal(y[M == 0], x[M == 0])
        assert not torch.equal(y[M == 1], x[M == 1])
