import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stadiff.config import load_config  # noqa: E402

TINY = [
    "data.image_size=[64, 64]",
    "diffusion.T_total=40", "diffusion.t_roi_max=30", "diffusion.t_bg=12",
    "model.roi_channels=[16, 32, 32, 64]", "model.ctx_channels=[8, 16, 16, 32]",
    "model.dec_channels=[16, 16, 32, 32]", "model.roi_blocks=1", "model.ctx_blocks=1",
    "model.dec_blocks=1", "model.emb_dim=32", "model.attn_heads=2", "model.dropout=0.0",
    "eval.steps_stage1=4", "eval.steps_stage2=4", "eval.batch_size=2",
]


@pytest.fixture
def tiny_cfg():
    return load_config(None, TINY)


@pytest.fixture
def fake_dataset(tmp_path):
    """Synthetic DATASET with the real folder layout: three bright rods, anomalies = shifted rod."""
    rng = np.random.default_rng(0)

    def img(shift=0):
        a = rng.normal(60, 10, (120, 160))
        for k, (x0, x1) in enumerate([(30, 46), (73, 88), (115, 131)]):
            s = shift if k == 1 else 0
            a[:, x0 + s:x1 + s] = rng.normal(180, 20, (120, x1 - x0))
        return Image.fromarray(a.clip(0, 255).astype(np.uint8))

    for rel, n, shift in [("Clean/Train", 12, 0), ("Clean/Test", 3, 0), ("Anomalous", 3, 12)]:
        d = tmp_path / "DATASET" / rel
        d.mkdir(parents=True)
        for i in range(n):
            img(shift).save(d / f"im{i:02d}.jpg")
    return tmp_path / "DATASET"
