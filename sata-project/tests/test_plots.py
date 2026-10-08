"""The notebook charts (src/evaluation/plots.py) render headless on small tidy frames."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation import plots  # noqa: E402


def _tidy():
    rng = np.random.default_rng(0)
    rows = [{"naming": n, "strategy": s, "env": e, "auroc": rng.uniform(0.4, 0.7)}
            for n in plots.NAMINGS for s in ("random", "label_diversity") for e in plots.ENVS]
    return pd.DataFrame(rows)


def test_palette_is_the_validated_three():
    assert plots.SERIES == ("#2a78d6", "#eb6834", "#1baf7a")


def test_facet_dots_with_intervals_and_hollow_markers():
    plots.use_style()
    d = _tidy().assign(lo=lambda x: x.auroc - 0.05, hi=lambda x: x.auroc + 0.05,
                       sig=lambda x: x.auroc > 0.55)
    fig = plots.facet_dots(d, facet="naming", y="strategy", x="auroc", hue="env", facet_order=plots.NAMINGS,
                           hue_order=plots.ENVS, lo="lo", hi="hi", filled="sig", ref=0.5, ref_label="chance",
                           hollow_note=True, ncols=2, title="t")
    assert len([a for a in fig.axes if a.get_visible()]) == 3
    plt.close(fig)


def test_forest_keeps_the_reference_in_view():
    d = pd.DataFrame({"row": ["a", "b"], "est": [0.12, 0.15], "lo": [0.08, 0.10], "hi": [0.17, 0.2],
                      "sig": [True, False], "note": ["p 0.01", "p 0.2"]})
    fig, ax = plt.subplots()
    plots.forest(ax, d, "row", "est", "lo", "hi", emphasis="sig", note="note")
    left, right = ax.get_xlim()
    assert left < 0 < right
    plt.close(fig)


def test_slopes_heatmap_density_and_strip():
    fig, axes = plt.subplots(2, 2)
    wide = pd.DataFrame({"abstract": [0.5, 0.6], "flipped": [0.4, 0.45], "g": ["x", "y"]})
    handles = plots.slopes(axes[0, 0], wide, ["abstract", "flipped"], group=wide["g"], ref=0.5, ref_label="chance")
    assert len(handles) == 2
    mat = pd.DataFrame([[0.1, -0.05], [0.0, 0.08]], index=["s1", "s2"],
                       columns=pd.MultiIndex.from_tuples([("abstract", "id"), ("flipped", "id")]))
    plots.diverging_heatmap(axes[0, 1], mat, annotate_above=0.05)
    x = np.random.default_rng(1).normal(size=500)
    plots.density_identity(axes[1, 0], x, x + 0.1)
    plots.strip(axes[1, 1], pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10, "v": np.arange(20.0)}), "g", "v", ref=0)
    plots.legend_below(fig, ["a", "b"], plots.SERIES[:2])
    plt.close(fig)
