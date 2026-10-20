"""Label-balanced top-k selection, shared by src/selection/sata_select.py
(inference-time selection) and src/models/sata_train.py (proxy validation).

An uncontrolled top-k lets a selector's demo-label balance itself become the
signal the LLM reacts to (Min et al. -- label-space statistics dominate
input-label mapping at this scale), and let SATA's proxy validation reward a
degenerate all-one-class selection via a since-removed single-class fallback
(REDESIGN_RATIONALE.md §4.2 Fact C, §4.3, §5.3).
"""

from __future__ import annotations

import numpy as np


def balanced_top_k(scores: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k//2 highest-scoring demos per label class (binary
    labels assumed), filling any remainder from whichever positions are left
    -- same stratify-then-fill pattern as src/selection/label_diversity.py.
    """
    order = np.argsort(-scores)
    per_class = k // 2
    selected: list[int] = []
    for cls in (0, 1):
        cls_positions = order[labels[order] == cls]
        selected.extend(cls_positions[:per_class].tolist())
    if len(selected) < k:
        leftover = [int(i) for i in order if int(i) not in selected]
        selected.extend(leftover[: k - len(selected)])
    return np.array(selected[:k])


def balanced_sample(scores: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k//2 positions per label class drawn without replacement with
    probability proportional to `scores` (non-negative, e.g. SATA's softmax),
    filling any remainder the same way from the positions left.

    SATA's selection rule from P5 (hiccups/21): its training targets are a
    distribution over rows, and drawing from the learned distribution keeps
    the target sets' composition (half the rows agreeing with the shortcut),
    which top-k of a learned approximation does not.
    """
    scores = np.clip(np.asarray(scores, dtype=float), 0, None)

    def draw(positions: np.ndarray, n: int) -> list[int]:
        n = min(n, len(positions))
        if n <= 0:
            return []
        w = scores[positions]
        p = w / w.sum() if w.sum() > 0 and np.count_nonzero(w) >= n else None
        return [int(i) for i in rng.choice(positions, size=n, replace=False, p=p)]

    selected: list[int] = []
    for cls in (0, 1):
        selected.extend(draw(np.flatnonzero(labels == cls), k // 2))
    if len(selected) < k:
        selected.extend(draw(np.setdiff1d(np.arange(len(scores)), selected), k - len(selected)))
    return np.array(selected[:k])
