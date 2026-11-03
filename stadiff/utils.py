"""Small helpers: seeding, device choice, EMA, checkpoints, image grids."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")     # Apple-silicon Mac: works, but much slower than a CUDA GPU
    return torch.device("cpu")


class EMA:
    """Exponential moving average of the weights; the EMA copy is used for sampling."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))     # warm-up
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)

    def state_dict(self):
        return {"model": self.model.state_dict(), "n": self.n, "decay": self.decay}

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd["model"])
        self.n, self.decay = sd["n"], sd["decay"]


def torch_load(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:          # torch < 1.13 has no weights_only argument
        return torch.load(path, map_location=device)


def save_checkpoint(path, **state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(1,H,W) or (H,W) tensor in [-1,1] -> uint8 (H,W)."""
    x = x.detach().float().cpu().squeeze()
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().numpy()


def heat(amap: np.ndarray, vmax: float) -> np.ndarray:
    """Anomaly map -> RGB heat image (black-red-yellow-white), fixed scale vmax."""
    v = np.clip(amap / max(vmax, 1e-8), 0, 1)
    r = np.clip(3 * v, 0, 1)
    g = np.clip(3 * v - 1, 0, 1)
    b = np.clip(3 * v - 2, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def save_triplets(rows, path, gap=4):
    """rows: list of (input uint8 HxW, recon uint8 HxW, heat RGB HxWx3, mask HxW or None)."""
    tiles = []
    for x, r, h, m in rows:
        xi = np.stack([x] * 3, -1)
        if m is not None:            # outline the ROI on the input
            edge = np.zeros_like(m, bool)
            inside = m > 0.5
            edge[1:-1, 1:-1] = inside[1:-1, 1:-1] & ~(inside[:-2, 1:-1] & inside[2:, 1:-1]
                                                      & inside[1:-1, :-2] & inside[1:-1, 2:])
            xi[edge] = [255, 60, 60]
        ri = np.stack([r] * 3, -1)
        pad = np.full((x.shape[0], gap, 3), 255, np.uint8)
        tiles.append(np.concatenate([xi, pad, ri, pad, h], 1))
    sep = np.full((gap, tiles[0].shape[1], 3), 255, np.uint8)
    grid = np.concatenate(sum([[t, sep] for t in tiles], [])[:-1], 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, default=conv))
