"""End to end on a synthetic dataset: prepare_data -> train (a few iterations) -> evaluate -> infer.
Slowest test (about a minute on CPU). Skip with:  pytest -m "not slow" """
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import TINY

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{args[0]} failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


@pytest.mark.slow
def test_full_pipeline(fake_dataset, tmp_path):
    prepared = tmp_path / "prepared"
    run("scripts/prepare_data.py", "--src", str(fake_dataset), "--dst", str(prepared), "--image-size", "64", "64")
    run_dir = tmp_path / "run"
    run("scripts/train.py", "--device", "cpu", "--data", str(prepared), "--out", str(run_dir),
        "--noise", "gaussian", "--t-roi", "30", "--t-bg", "12", "--iters", "6", "--batch-size", "2",
        "--val-fraction", "0.25", "--num-workers", "0", "--log-every", "2", "--val-every", "3",
        "--ckpt-every", "3", "--preview-every", "6", "--amp", "false", "--set", *TINY)
    assert (run_dir / "ckpt_last.pt").exists() and (run_dir / "ckpt_best.pt").exists()
    eval_dir = tmp_path / "eval"
    out = run("scripts/evaluate.py", "--checkpoint", str(run_dir / "ckpt_best.pt"), "--device", "cpu",
              "--out", str(eval_dir), "--save-vis", "4", "--t-roi", "25", "--map", "l1+ssim")
    assert "AUROC" in out and (eval_dir / "metrics.json").exists()
    run("scripts/infer.py", "--checkpoint", str(run_dir / "ckpt_best.pt"), "--device", "cpu",
        "--images", str(fake_dataset / "Anomalous"), "--metrics", str(eval_dir / "metrics.json"),
        "--out", str(tmp_path / "infer"))
    assert (tmp_path / "infer" / "results.csv").exists()
