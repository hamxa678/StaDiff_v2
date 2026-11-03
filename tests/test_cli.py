"""Named command-line flags map onto the right configuration values."""
import argparse

from stadiff.cli import add_common_flags, add_config_flags, config_from_args
from stadiff.config import to_dict


def test_train_flags():
    ap = add_config_flags(add_common_flags(argparse.ArgumentParser()), ["data", "diffusion", "model", "train"])
    a = ap.parse_args("--data D --noise simplex --t-roi 700 --t-bg 300 --batch-size 4 --lr 2e-4 "
                      "--fusion gated gated cross none --random-bg true --t-bg-min 200 --t-bg-max 400 "
                      "--out runs/x --set model.dropout=0.0".split())
    c = config_from_args(a)
    assert c.data.root == "D" and c.diffusion.noise_type == "simplex"
    assert (c.diffusion.t_roi_max, c.diffusion.t_bg) == (700, 300)
    assert c.train.batch_size == 4 and abs(c.train.lr - 2e-4) < 1e-12 and c.train.out_dir == "runs/x"
    assert c.model.fusion == ["gated", "gated", "cross", "none"] and c.model.dropout == 0.0
    assert c.diffusion.random_bg is True and c.diffusion.t_bg_min == 200


def test_eval_flags_on_top_of_checkpoint_config():
    base = to_dict(config_from_args(argparse.Namespace(config=None, set=[])))
    ap = add_config_flags(argparse.ArgumentParser(), ["eval", "data.root"])
    a = ap.parse_args("--t-roi 600 --sampler ddpm --n-samples 3 --map l1+ssim".split())
    a.set = []
    c = config_from_args(a, base=base)
    assert c.eval.t_roi_max == 600 and c.eval.sampler == "ddpm" and c.eval.n_samples == 3
    assert c.diffusion.t_roi_max == 800          # training value untouched
