"""
Configuration: one nested dictionary, loaded from a YAML (or JSON) file and
optionally overridden from the command line with  --set section.key=value.

Every key has a default in DEFAULTS below, so a config file only needs the
values you want to change. Values are reachable as attributes: cfg.train.lr
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS: dict = {
    "data": {
        "root": "DATASET_prepared",     # output of scripts/prepare_data.py
        "image_size": [256, 256],       # [H, W]; must be divisible by 16
        "prepare_method": "area",       # resize used by prepare_data.py and infer.py
        "norm": "percentile",           # per-image 1-99 percentile stretch, or "none"
        "mask_source": "rods",          # "rods" (rod + weld columns), "welds" (bead boxes), "file"
        "mask_path": None,              # PNG mask when mask_source == "file"
        "val_fraction": 0.05,           # share of Clean/Train held out (val loss + threshold calibration)
        "split_seed": 0,
        "augment": True,                # photometric only (geometry must stay aligned with the mask)
    },
    "diffusion": {
        "T_total": 1000,
        "beta_schedule": "linear",      # "linear" or "cosine"
        "beta_start": 1.0e-4,
        "beta_end": 2.0e-2,
        "t_roi_max": 800,               # ROI noise level at the start of Stage 1 (user-selectable)
        "t_bg": 400,                    # background noise level = Stage 1 end = Stage 2 start
        "random_bg": False,             # True: train over t_bg in [t_bg_min, t_bg_max]
        "t_bg_min": None,
        "t_bg_max": None,
        "p_stage1": 0.5,                # probability of a Stage 1 sample in a training batch
        "lam": 4.0,                     # ROI weight in the Stage 2 loss
        "noise_type": "gaussian",       # "gaussian" or "simplex" (one type for both regions)
        "simplex_scale": 32.0,          # coarsest simplex feature size in pixels
        "simplex_octaves": 6,
        "simplex_persistence": 0.8,
    },
    "model": {
        "roi_channels": [64, 128, 256, 512],   # heavy ROI encoder, at H/2, H/4, H/8, H/16
        "ctx_channels": [32, 64, 128, 256],    # light context encoder, same resolutions
        "dec_channels": [64, 64, 128, 256],    # decoder output channels per level
        "roi_blocks": 2,
        "ctx_blocks": 1,
        "dec_blocks": 2,
        "emb_dim": 256,
        "attn_heads": 4,
        "dropout": 0.1,
        "fusion": ["gated", "gated", "cross", "cross"],   # per level: gated | cross | none
        "use_context_stream": True,     # False = single-stream ablation
        "bottleneck_cross_attn": True,
    },
    "train": {
        "out_dir": "runs/stadiff_v3",
        "batch_size": 8,
        "lr": 1.0e-4,
        "weight_decay": 0.0,
        "iters": 100000,
        "warmup_iters": 1000,
        "ema_decay": 0.999,
        "grad_clip": 1.0,
        "amp": True,                    # mixed precision on CUDA
        "num_workers": 4,
        "log_every": 100,
        "val_every": 2000,
        "ckpt_every": 5000,
        "preview_every": 5000,          # reverse-process figure (step-by-step reconstruction)
        "batch_vis_every": 1000,        # figure of training samples: input, true / predicted noise
        "quick_eval_every": 5000,       # small detection test during training (0 = off)
        "quick_eval_n": 8,              # clean val + anomalous images used by the quick test
        "tensorboard": True,
        "seed": 0,
        "resume": None,
    },
    "eval": {
        "t_roi_max": None,              # None = value used in training
        "t_bg": None,
        "sampler": "ddim",              # "ddim" (fast, deterministic) or "ddpm" (every step)
        "steps_stage1": 50,
        "steps_stage2": 50,
        "n_samples": 1,                 # reconstructions averaged per image
        "map": "l1",                    # "l1", "ssim" or "l1+ssim"
        "smooth_sigma": 2.0,            # Gaussian blur of the anomaly map (pixels)
        "reduce": "topk",               # pixel -> score inside a region: mean | max | topk
        "topk_frac": 0.02,              # fraction of pixels averaged for "topk"
        "image_score": "max_rod",       # "max_rod" (worst rod decides) or "roi" (whole ROI)
        "threshold_percentile": 99.0,   # threshold = this percentile of clean validation scores
        "batch_size": 8,
        "save_vis": 40,
        "seed": 0,
    },
}


class Cfg(dict):
    """dict with attribute access (cfg.train.lr)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def to_cfg(d):
    if isinstance(d, dict):
        return Cfg({k: to_cfg(v) for k, v in d.items()})
    if isinstance(d, list):
        return [to_cfg(v) for v in d]
    return d


def to_dict(c):
    if isinstance(c, dict):
        return {k: to_dict(v) for k, v in c.items()}
    if isinstance(c, list):
        return [to_dict(v) for v in c]
    return c


def _merge(base: dict, upd: dict, path=""):
    for k, v in upd.items():
        if k not in base:
            raise KeyError(f"unknown config key: {path}{k}")
        if isinstance(base[k], dict) and isinstance(v, dict):
            _merge(base[k], v, f"{path}{k}.")
        else:
            base[k] = v


def _parse_value(s: str):
    try:
        import yaml
        v = yaml.safe_load(s)
    except ImportError:
        try:
            v = json.loads(s)
        except json.JSONDecodeError:
            v = {"true": True, "false": False, "null": None}.get(s.lower(), s)
    if isinstance(v, str):          # YAML reads "2e-4" as text; accept it as a number
        try:
            return float(v) if any(ch in v for ch in ".eE") else int(v)
        except ValueError:
            pass
    return v


def _read_file(path: Path) -> dict:
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml   # pip install pyyaml
        return yaml.safe_load(text) or {}
    return json.loads(text)


def apply_overrides(cfg: dict, overrides) -> dict:
    for o in overrides or []:
        if "=" not in o:
            raise ValueError(f"override must look like section.key=value, got {o!r}")
        key, val = o.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node:
                raise KeyError(f"unknown config section: {p}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key: {key}")
        node[parts[-1]] = _parse_value(val)
    return cfg


def load_config(path=None, overrides=(), base: dict | None = None) -> Cfg:
    cfg = copy.deepcopy(base if base is not None else DEFAULTS)
    if path:
        _merge(cfg, _read_file(Path(path)))
    apply_overrides(cfg, overrides)
    validate(cfg)
    return to_cfg(cfg)


def validate(cfg: dict):
    H, W = cfg["data"]["image_size"]
    if H % 16 or W % 16:
        raise ValueError("data.image_size must be divisible by 16 (four downsamplings)")
    m = cfg["model"]
    n = len(m["roi_channels"])
    for k in ("ctx_channels", "dec_channels", "fusion"):
        if len(m[k]) != n:
            raise ValueError(f"model.{k} must have {n} entries, like model.roi_channels")
    for f in m["fusion"]:
        if f not in ("gated", "cross", "none"):
            raise ValueError(f"model.fusion entries must be gated|cross|none, got {f}")


def save_config(cfg, path):
    d = to_dict(cfg)
    path = Path(path)
    try:
        import yaml
        path.with_suffix(".yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    except ImportError:
        path.with_suffix(".json").write_text(json.dumps(d, indent=2))
