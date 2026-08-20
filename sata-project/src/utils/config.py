"""Config loading. Single source of truth for hyperparameters, paths, and seeds.

Every notebook and module should load config via `load_config()` rather than
hardcoding values, so a single edit to configs/default.yaml propagates everywhere.
"""

from __future__ import annotations

import os

# torch and xgboost each link/bundle their own libomp.dylib on macOS; loading
# both in one process reliably segfaults unless OpenMP is forced single-threaded
# before either is imported. This module is always the first src import in
# every notebook's setup cell, so it's the one place that's guaranteed to run
# before torch/xgboost do.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _to_namespace(obj: Any) -> Any:
    """Recursively convert nested dicts into SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> SimpleNamespace:
    """Load the YAML config and return a dot-accessible namespace.

    Example:
        config = load_config()
        config.sata.d_model
        config.seed_accuracy
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


def resolve_path(relative: str, config: SimpleNamespace | None = None) -> Path:
    """Resolve a path relative to the project root (not the notebook's cwd)."""
    return PROJECT_ROOT / relative


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class RunContext:
    """Bundles the (dataset, model, method, seed) identity of one experimental run.

    Used to tag rows written to the results parquet consistently across notebooks.
    """

    arm: str          # "real" or "synthetic"
    dataset: str
    model: str
    method: str
    seed: int
