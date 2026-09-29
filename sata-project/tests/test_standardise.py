"""Verifies unified train/inference standardisation (REDESIGN_RATIONALE.md
§4.5/§5.3): the same physical row must get the same z-scored value
regardless of whether it's later pre-filtered, since stats are fit on the
full pool before any filtering -- not on whatever subset `pre_filtered_idx`
happens to select (v1's bug: `sata_alone` saw the full-64-row pool's stats,
`best_protocol_sata` saw only the pre-filtered 32's)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.standardise import standardise


def test_standardise_matches_hand_computed_zscore():
    pool = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    query = np.array([[2.0, 20.0]])
    pool_z, query_z = standardise(pool, query)
    mean = pool.mean(axis=0)
    std = pool.std(axis=0) + 1e-8
    assert np.allclose(pool_z, (pool - mean) / std)
    assert np.allclose(query_z, (query - mean) / std)


def test_select_uses_full_pool_stats_not_prefiltered_subset_stats():
    """A model that scores each demo by how close its (standardised)
    feature_0 is to zero -- peaks at whichever raw value equals the mean
    used for standardisation. The pre-filtered subset here is deliberately
    a high-value slice of the pool, so its own mean is very different from
    the full pool's mean: this distinguishes "stats fit on the full pool"
    (correct) from "stats fit on the subset" (v1's bug) by which row
    actually gets selected.
    """
    torch = pytest.importorskip("torch")
    from src.selection.sata_select import select

    class _PeakAtZeroModel(torch.nn.Module):
        def eval(self):
            return self

        def __call__(self, demo_features, demo_labels, query_features):
            with torch.no_grad():
                return -(demo_features[:, :, 0] ** 2)

    feature_cols = ["feature_0", "feature_1"]
    # Full pool: feature_0 spans roughly [-3, 3], mean ~0.
    full_values = np.linspace(-3, 3, 20)
    pool = pd.DataFrame({"feature_0": full_values, "feature_1": np.zeros(20)})
    pool["label"] = ([0, 1] * 10)
    query = pool.iloc[0]

    # Pre-filtered subset: only the high end (feature_0 in [1, 3]) -- its
    # own mean is ~2, very different from the full pool's ~0.
    pre_filtered_idx = pool.index[full_values >= 1]
    assert len(pre_filtered_idx) >= 4

    model = _PeakAtZeroModel()
    selected = select(model, pool, query, feature_cols, "label", k=2, pre_filtered_idx=pre_filtered_idx)

    # Under correct (full-pool) standardisation, the peak is at raw
    # feature_0 == full-pool mean (~0) -- within the subset (all >= 1), the
    # row closest to 0 is the smallest value in the subset. Under the v1 bug
    # (subset-fit standardisation), the peak would instead be at the
    # subset's own mean (~2), selecting a different row.
    subset_values = pool.loc[pre_filtered_idx, "feature_0"]
    closest_to_full_pool_mean = (subset_values - 0.0).abs().idxmin()
    closest_to_subset_mean = (subset_values - subset_values.mean()).abs().idxmin()
    assert closest_to_full_pool_mean != closest_to_subset_mean, "test setup must distinguish the two hypotheses"
    assert closest_to_full_pool_mean in selected
    assert closest_to_subset_mean not in selected
