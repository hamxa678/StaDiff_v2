"""
Named command-line flags for every configuration value.

Each script calls add_config_flags(parser, keys) to get flags such as
    --noise simplex  --t-roi 800  --t-bg 400  --batch-size 8  --random-bg true
and config_from_args(args) to build the final configuration.

Precedence (later wins):  built-in defaults  <  --config file  <  named flags  <  --set key=value
"""
from __future__ import annotations

import argparse

from .config import DEFAULTS, apply_overrides, load_config, to_cfg, to_dict, validate

# Short, friendly names in addition to the automatic ones (--t-roi-max etc.)
ALIASES = {
    "data.root": ["--data"],
    "diffusion.noise_type": ["--noise"],
    "diffusion.t_roi_max": ["--t-roi"],
    "diffusion.t_bg": ["--t-bg"],
    "train.out_dir": ["--out"],
    "eval.t_roi_max": ["--t-roi"],
    "eval.t_bg": ["--t-bg"],
}

CHOICES = {
    "data.prepare_method": ["area", "bicubic", "bilinear_noaa"],
    "data.norm": ["percentile", "none"],
    "data.mask_source": ["rods", "welds", "file"],
    "diffusion.beta_schedule": ["linear", "cosine"],
    "diffusion.noise_type": ["gaussian", "simplex"],
    "eval.sampler": ["ddim", "ddpm"],
    "eval.map": ["l1", "ssim", "l1+ssim"],
    "eval.reduce": ["mean", "max", "topk"],
    "eval.image_score": ["max_rod", "roi"],
}

# type for keys whose default is None
NONE_TYPES = {
    "data.mask_path": str, "diffusion.t_bg_min": int, "diffusion.t_bg_max": int,
    "train.resume": str, "eval.t_roi_max": int, "eval.t_bg": int,
}

HELP = {
    "data.root": "prepared dataset folder (output of prepare_data.py)",
    "data.image_size": "H W of the network input, both divisible by 16",
    "data.prepare_method": "resize method for raw images",
    "data.norm": "intensity normalisation per image",
    "data.mask_source": "ROI: rods = rod+weld columns, welds = bead boxes, file = your PNG",
    "data.mask_path": "mask PNG (white = ROI) when --mask-source file",
    "data.val_fraction": "share of Clean/Train held out for validation and threshold calibration",
    "data.split_seed": "seed of the train/validation split",
    "data.augment": "photometric augmentation during training",
    "diffusion.T_total": "length of the noise schedule",
    "diffusion.beta_schedule": "noise schedule",
    "diffusion.beta_start": "first beta (linear schedule)",
    "diffusion.beta_end": "last beta (linear schedule)",
    "diffusion.t_roi_max": "ROI noise level = start of stage 1",
    "diffusion.t_bg": "background noise level = end of stage 1 / start of stage 2",
    "diffusion.random_bg": "train over a range of t_bg so it can be changed at test time",
    "diffusion.t_bg_min": "lowest t_bg when --random-bg true",
    "diffusion.t_bg_max": "highest t_bg when --random-bg true",
    "diffusion.p_stage1": "probability that a training sample is a stage-1 sample",
    "diffusion.lam": "extra weight on ROI pixels in the stage-2 loss",
    "diffusion.noise_type": "noise used in both regions (one model per type)",
    "diffusion.simplex_scale": "coarsest simplex blob size in pixels",
    "diffusion.simplex_octaves": "number of simplex octaves",
    "diffusion.simplex_persistence": "amplitude ratio between simplex octaves",
    "model.roi_channels": "ROI encoder channels per level (H/2 .. H/16)",
    "model.ctx_channels": "context encoder channels per level",
    "model.dec_channels": "decoder channels per level",
    "model.roi_blocks": "ResBlocks per ROI-encoder level",
    "model.ctx_blocks": "ResBlocks per context-encoder level",
    "model.dec_blocks": "ResBlocks per decoder level",
    "model.emb_dim": "conditioning embedding size",
    "model.attn_heads": "attention heads",
    "model.dropout": "dropout in ResBlocks",
    "model.fusion": "exchange per level: gated | cross | none",
    "model.use_context_stream": "false = single-stream ablation",
    "model.bottleneck_cross_attn": "cross-attention to the context in the bottleneck",
    "train.out_dir": "folder for checkpoints, logs and previews",
    "train.batch_size": "images per training step",
    "train.lr": "learning rate (AdamW)",
    "train.weight_decay": "AdamW weight decay",
    "train.iters": "number of training iterations",
    "train.warmup_iters": "linear learning-rate warm-up",
    "train.ema_decay": "EMA decay of the weights used for sampling",
    "train.grad_clip": "gradient-norm clipping",
    "train.amp": "mixed precision on CUDA",
    "train.num_workers": "data-loading processes",
    "train.log_every": "print/log every N iterations",
    "train.val_every": "validation loss every N iterations",
    "train.ckpt_every": "checkpoint every N iterations",
    "train.preview_every": "save the step-by-step reverse-process figure every N iterations",
    "train.batch_vis_every": "save a figure of training samples (input, true/predicted noise) every N iterations",
    "train.quick_eval_every": "small detection test (scores, AUROC, reconstruction error) every N iterations; 0 = off",
    "train.quick_eval_n": "number of clean and of anomalous images in the quick test",
    "train.tensorboard": "also log to TensorBoard (out/tensorboard)",
    "train.seed": "random seed",
    "train.resume": "checkpoint (ckpt_last.pt) to continue training from",
    "eval.t_roi_max": "ROI noise level at test time (<= trained value; default: trained value)",
    "eval.t_bg": "background level at test time (needs a --random-bg model to differ)",
    "eval.sampler": "ddim = fast deterministic, ddpm = every step",
    "eval.steps_stage1": "DDIM network calls in stage 1",
    "eval.steps_stage2": "DDIM network calls in stage 2",
    "eval.n_samples": "reconstructions averaged per image",
    "eval.map": "anomaly map type",
    "eval.smooth_sigma": "Gaussian blur of the anomaly map (pixels)",
    "eval.reduce": "how pixels in a region become a score",
    "eval.topk_frac": "fraction of pixels averaged for --reduce topk",
    "eval.image_score": "max_rod = worst rod decides, roi = whole ROI",
    "eval.threshold_percentile": "threshold = this percentile of clean validation scores",
    "eval.batch_size": "images per evaluation batch",
    "eval.save_vis": "number of images in the visualisation sheets",
    "eval.seed": "random seed of the test-time noise",
}


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y", "on"):
        return True
    if v.lower() in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")


def _expand(keys):
    out = []
    for k in keys:
        if "." in k:
            out.append(k)
        else:
            out += [f"{k}.{kk}" for kk in DEFAULTS[k]]
    return out


def add_config_flags(parser: argparse.ArgumentParser, keys, aliases: bool = True):
    """Add one flag per config key. keys: section names ("train") or full keys ("data.root")."""
    groups = {}
    for full in _expand(keys):
        sec, key = full.split(".")
        default = DEFAULTS[sec][key]
        g = groups.setdefault(sec, parser.add_argument_group(f"{sec} settings"))
        flags = ["--" + key.lower().replace("_", "-")]
        if aliases:
            flags += [a for a in ALIASES.get(full, []) if a not in flags]
        kw = dict(dest=full, default=argparse.SUPPRESS)
        shown = default if not isinstance(default, list) else " ".join(map(str, default))
        kw["help"] = f"{HELP.get(full, '')} (default: {shown})"
        if isinstance(default, bool):
            kw.update(type=str2bool, metavar="true|false")
        elif isinstance(default, list):
            kw.update(nargs="+", type=type(default[0]))
            if full == "model.fusion":
                kw["choices"] = ["gated", "cross", "none"]
        elif default is None:
            kw.update(type=NONE_TYPES.get(full, str))
        else:
            kw.update(type=type(default))
        if full in CHOICES:
            kw["choices"] = CHOICES[full]
        if "metavar" not in kw and "choices" not in kw:
            kw["metavar"] = key.upper()
        g.add_argument(*flags, **kw)
    return parser


def add_common_flags(parser: argparse.ArgumentParser, config_default=None):
    parser.add_argument("--config", default=config_default, help="YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="any other config value, e.g. --set model.dropout=0.0")
    return parser


def flag_values(args) -> dict:
    return {k: v for k, v in vars(args).items() if "." in k}


def set_values(cfg: dict, values: dict) -> dict:
    for full, v in values.items():
        sec, key = full.split(".")
        cfg[sec][key] = v
    return cfg


def config_from_args(args, base: dict | None = None):
    """defaults (or base) < --config < named flags < --set"""
    cfg = to_dict(load_config(getattr(args, "config", None), (), base=base))
    set_values(cfg, flag_values(args))
    apply_overrides(cfg, getattr(args, "set", []))
    validate(cfg)
    return to_cfg(cfg)
