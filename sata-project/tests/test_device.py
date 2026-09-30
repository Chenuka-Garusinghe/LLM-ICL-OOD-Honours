from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.sata import SATA
from src.selection import sata_select
from src.utils.device import device_name, resolve_device, resolve_dtype

HAS_CUDA = torch.cuda.is_available()
HAS_MPS = torch.backends.mps.is_available()


def test_auto_prefers_cuda_then_mps_then_cpu():
    expected = "cuda" if HAS_CUDA else "mps" if HAS_MPS else "cpu"
    assert resolve_device("auto").type == expected


def test_explicit_cpu_is_honoured():
    assert resolve_device("cpu").type == "cpu"


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        resolve_device("tpu")


@pytest.mark.skipif(HAS_CUDA, reason="only meaningful without a CUDA GPU")
def test_unavailable_cuda_raises():
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_cpu_dtype_is_float32():
    assert resolve_dtype(torch.device("cpu")) == torch.float32


@pytest.mark.skipif(not HAS_MPS, reason="needs Apple Metal")
def test_mps_keeps_bfloat16():
    assert resolve_dtype(torch.device("mps")) == torch.bfloat16


def test_device_name_is_nonempty():
    assert device_name(resolve_device("auto"))


def _toy_pool(n: int = 64, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(10)]
    pool = pd.DataFrame(rng.normal(size=(n, 10)), columns=cols)
    pool["label"] = (pool["f0"] > 0).astype(int)
    return pool, pool.iloc[0], cols


def test_sata_select_runs_on_resolved_device():
    torch.manual_seed(0)
    model = SATA(n_features=10).to(resolve_device("auto"))
    pool, query, cols = _toy_pool()
    ids = sata_select.select(model, pool, query, cols, "label", k=8)
    assert len(set(ids)) == 8
    assert pool.loc[ids, "label"].sum() == 4


@pytest.mark.skipif(not (HAS_CUDA or HAS_MPS), reason="needs a GPU to compare against CPU")
def test_sata_scores_agree_across_devices():
    torch.manual_seed(0)
    cpu_model = SATA(n_features=10).eval()
    gpu_model = SATA(n_features=10).eval()
    gpu_model.load_state_dict(cpu_model.state_dict())
    gpu_model.to(resolve_device("auto"))

    x = torch.randn(1, 64, 10)
    y = torch.randint(0, 2, (1, 64))
    q = torch.randn(1, 10)
    with torch.no_grad():
        cpu_scores = cpu_model(x, y, q)
        gpu_scores = gpu_model(x.to(gpu_model.score_head.weight.device), y.to(gpu_model.score_head.weight.device),
                               q.to(gpu_model.score_head.weight.device)).cpu()
    assert torch.allclose(cpu_scores, gpu_scores, atol=1e-5)
