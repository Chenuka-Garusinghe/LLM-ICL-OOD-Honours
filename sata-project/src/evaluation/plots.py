"""Charts for the analysis notebooks: one style, one palette, a few forms.

Every chart sits directly below the table it explains, so the table is its
exact-values view.

Colours follow the data's job:
- **Series (identity).** At most three per chart, from a validated categorical
  palette: blue, orange and aqua. These pass the colour-vision checks for any
  pair (worst simulated CVD ΔE 9.2).
- **Signed differences.** A blue-grey-red diverging scale, positive in blue.
- **Densities.** One blue ramp, light to dark.
- **Emphasis.** The mark that carries the finding is blue, and context is grey.

Marks are thin, gridlines are solid hairlines, filled dots carry a ring in the
surface colour, and reference lines (chance, no difference) are labelled
rather than dashed.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
SERIES = (BLUE, ORANGE, AQUA)          # categorical slots 1-3, in this order
CONTEXT = "#c3c2b7"                    # de-emphasised marks
DIVERGING = LinearSegmentedColormap.from_list("red_grey_blue", [RED, "#f0efec", BLUE])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

ENVS = ("id", "covariate", "spurious_reversal")
NAMINGS = ("abstract", "aligned", "flipped")
ARMS = ("none", "spurious", "spurious_noise")
STRATEGIES = ("zero_shot", "random", "label_diversity", "feature_range", "rule_diversity",
              "counter_spurious", "similarity", "counter_prior", "counter_prior_matched")

LABELS = {
    "id": "ID", "covariate": "covariate", "spurious_reversal": "spurious reversal",
    "covariate_scale": "covariate (variance)",
    "none": "full demonstrations", "spurious": "no shortcut", "spurious_noise": "no shortcut or noise",
    "zero_shot": "zero-shot", "load_bearing": "rule features", "spurious_feature": "shortcut",
}


def label(value) -> str:
    """Display label of a category (the code name when there is no nicer one)."""
    return LABELS.get(value, str(value))


def use_style() -> None:
    """Apply the shared style to every later figure."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        # A family list, so glyphs missing from the first font (→, σ) fall back to the next.
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 9.5, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "figure.titlesize": 11.5, "figure.titleweight": "bold",
    })


# ---------------------------------------------------------------- building blocks
def _dot(ax, x, y, colour, filled=True, size=7.0, zorder=3):
    """Filled dots carry a surface-coloured ring; hollow dots mark 'interval includes the reference'."""
    ax.plot(x, y, linestyle="none", marker="o", markersize=size,
            markerfacecolor=colour if filled else SURFACE,
            markeredgecolor=SURFACE if filled else colour,
            markeredgewidth=1.4 if filled else 1.8, zorder=zorder)


def reference(ax, value: float, text: str | None = None, axis: str = "x", where: str = "top") -> None:
    """A labelled reference line (chance, no difference): a solid hairline, never dashed.
    `where` places the label: top or bottom of a vertical line, left or right of a horizontal one."""
    if axis == "x":
        ax.axvline(value, color=MUTED, linewidth=1.0, zorder=1)
        if text:
            y, va = (1.0, "bottom") if where == "top" else (0.0, "bottom")
            ax.text(value, y, f" {text}", transform=ax.get_xaxis_transform(), color=MUTED,
                    fontsize=7.5, ha="left", va=va)
    else:
        ax.axhline(value, color=MUTED, linewidth=1.0, zorder=1)
        if text:
            x, ha = (0.01, "left") if where == "left" else (0.99, "right")
            ax.text(x, value, f" {text} ", transform=ax.get_yaxis_transform(), color=MUTED,
                    fontsize=7.5, ha=ha, va="bottom")


def _rows(ax, rows: Sequence) -> dict:
    """Categorical y axis, first row at the top; vertical hairline grid."""
    rows = list(rows)
    pos = dict(zip(rows, np.arange(len(rows))[::-1]))
    ax.set_yticks(list(pos.values()), [label(r) for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    return pos


def legend_handles(names: Sequence, colours: Sequence, hollow_note: bool = False) -> list:
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=c,
                      markeredgecolor=SURFACE, label=label(n)) for n, c in zip(names, colours)]
    if hollow_note:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=SURFACE,
                              markeredgecolor=MUTED, markeredgewidth=1.8, label="interval includes 0"))
    return handles


def legend_below(fig, handles_or_names, colours=None, hollow_note: bool = False, ncol: int | None = None) -> None:
    """The figure's legend, centred below the plots so it never covers data. Takes handles,
    or category names with their colours."""
    if colours is not None:
        handles_or_names = legend_handles(handles_or_names, colours, hollow_note)
    _figure_legend(fig, list(handles_or_names), ncol)


def _figure_legend(fig, handles, ncol=None) -> None:
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=ncol or len(handles), handletextpad=0.3,
                   columnspacing=1.6)


def dots(ax, df: pd.DataFrame, y: str, x: str, hue: str | None = None, y_order=None, hue_order=None,
         colours=None, lo: str | None = None, hi: str | None = None, filled: str | None = None,
         connect: bool = False, spread: float = 0.24, xlabel: str | None = None,
         ref: float | None = None, ref_label: str | None = None) -> dict:
    """Dot plot: one row per `y` category and one dot per `hue` series.

    Optional `lo`/`hi` columns draw intervals. A boolean `filled` column draws
    hollow dots where it is False. `connect` joins a row's dots with a thin
    grey line (a dumbbell). Series are offset vertically within a row, so
    equal values never hide each other.
    """
    y_order = list(y_order if y_order is not None else pd.unique(df[y]))
    pos = _rows(ax, y_order)
    hues = [None] if hue is None else list(hue_order if hue_order is not None else pd.unique(df[hue]))
    colours = list(colours if colours is not None else SERIES)
    n = len(hues)
    offsets = np.zeros(n) if (n == 1 or connect) else np.linspace(spread / 2, -spread / 2, n)
    if connect and hue is not None:
        wide = df.pivot_table(index=y, columns=hue, values=x)
        for row in wide.index.intersection(y_order):
            vals = wide.loc[row].dropna()
            if len(vals) >= 2:
                ax.hlines(pos[row], vals.min(), vals.max(), color=CONTEXT, linewidth=2, zorder=2)
    for i, h in enumerate(hues):
        d = df if h is None else df[df[hue] == h]
        d = d[d[y].isin(pos)]
        if d.empty:
            continue
        yy = np.array([pos[v] for v in d[y]], dtype=float) + offsets[i]
        c = colours[i % len(colours)]
        if lo and hi:
            ax.hlines(yy, d[lo].to_numpy(float), d[hi].to_numpy(float), color=c, linewidth=1.8, zorder=2)
        if filled:
            f = d[filled].to_numpy(bool)
            _dot(ax, d[x].to_numpy(float)[f], yy[f], c, filled=True)
            _dot(ax, d[x].to_numpy(float)[~f], yy[~f], c, filled=False)
        else:
            _dot(ax, d[x].to_numpy(float), yy, c)
    if ref is not None:
        reference(ax, ref, ref_label)
    if xlabel:
        ax.set_xlabel(xlabel)
    return pos


def facet_dots(df: pd.DataFrame, facet: str, y: str, x: str, hue: str | None = None, facet_order=None,
               y_order=None, hue_order=None, colours=None, title: str | None = None, xlabel: str | None = None,
               ref: float | None = None, ref_label: str | None = None, lo=None, hi=None, filled=None,
               connect=False, panel_width: float = 3.9, row_height: float = 0.34, sharex: bool = True,
               hollow_note: bool = False, ncols: int | None = None):
    """Small multiples of `dots`, one panel per `facet` value, sharing the row axis."""
    facets = list(facet_order if facet_order is not None else pd.unique(df[facet]))
    y_order = list(y_order if y_order is not None else pd.unique(df[y]))
    hues = [] if hue is None else list(hue_order if hue_order is not None else pd.unique(df[hue]))
    ncols = ncols or len(facets)
    nrows = int(np.ceil(len(facets) / ncols))
    height = 0.9 + row_height * len(y_order) * (1 + 0.18 * max(len(hues) - 1, 0))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_width * ncols + 1.4, height * nrows + 0.5),
                             sharex=sharex, sharey=True, layout="constrained", squeeze=False)
    for ax, f in zip(axes.flat, facets):
        dots(ax, df[df[facet] == f], y, x, hue, y_order, hues or None, colours, lo, hi, filled, connect,
             xlabel=xlabel, ref=ref, ref_label=ref_label)
        ax.set_title(label(f))
    for ax in list(axes.flat)[len(facets):]:
        ax.set_visible(False)
    for ax in axes.flat:                     # every panel keeps its own tick labels
        ax.tick_params(labelbottom=True)
    if len(hues) >= 2 or hollow_note:
        _figure_legend(fig, legend_handles(hues, list(colours or SERIES), hollow_note))
    if title:
        fig.suptitle(title, x=0.01, ha="left")
    return fig


def forest(ax, df: pd.DataFrame, row: str, est: str, lo: str, hi: str, emphasis: str | None = None,
           note: str | None = None, ref: float = 0.0, ref_label: str = "no difference", xlabel: str | None = None):
    """Estimates with 95% intervals, one row each; `emphasis` rows in blue, the rest grey.
    `note` prints a short text (for example an adjusted p-value) just right of each interval."""
    pos = _rows(ax, df[row])
    for _, r in df.iterrows():
        c = BLUE if (emphasis is None or bool(r[emphasis])) else CONTEXT
        ax.hlines(pos[r[row]], r[lo], r[hi], color=c, linewidth=2, zorder=2)
        _dot(ax, r[est], pos[r[row]], c, size=7.5)
        if note:
            ax.text(r[hi], pos[r[row]], f"  {r[note]}", color=INK_2, fontsize=7.5, va="center", ha="left")
    left, right = ax.get_xlim()
    left, right = min(left, ref), max(right, ref)          # the reference line stays in view
    pad = 0.05 * (right - left)
    ax.set_xlim(left - pad, right + pad + (0.35 * (right - left) if note else 0))   # room for the notes
    reference(ax, ref, ref_label)
    if xlabel:
        ax.set_xlabel(xlabel)
    return pos


def slopes(ax, wide: pd.DataFrame, columns: Sequence, group: pd.Series | None = None, group_order=None,
           colours=None, ref: float | None = None, ref_label: str | None = None, ylabel: str | None = None):
    """One thin line per row of `wide` across `columns` (for example one task across namings), and a
    thick line for each group's mean. Without groups the mean is blue and the rows grey."""
    x = np.arange(len(columns))
    groups = [None] if group is None else list(group_order if group_order is not None else pd.unique(group))
    colours = list(colours if colours is not None else SERIES)
    handles = []
    for i, g in enumerate(groups):
        rows = wide if g is None else wide[group == g]
        c = BLUE if g is None else colours[i]
        thin, alpha = (CONTEXT, 0.6) if g is None else (c, 0.3)
        for _, r in rows.iterrows():
            ax.plot(x, r[list(columns)].to_numpy(float), color=thin, linewidth=1.0, alpha=alpha, zorder=2,
                    marker="o", markersize=3.5, markeredgewidth=0)
        mean = rows[list(columns)].mean().to_numpy(float)
        ax.plot(x, mean, color=c, linewidth=2.6, zorder=4)
        _dot(ax, x, mean, c, size=8, zorder=5)
        handles.append(Line2D([], [], color=c, linewidth=2.6, marker="o", markersize=7, markerfacecolor=c,
                              markeredgecolor=SURFACE, label="mean" if g is None else f"{label(g)} (mean)"))
    ax.set_xticks(x, [label(c) for c in columns])
    ax.set_xlim(-0.3, len(columns) - 0.7)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    if ref is not None:
        reference(ax, ref, ref_label, axis="y")
    if ylabel:
        ax.set_ylabel(ylabel)
    return handles


def diverging_heatmap(ax, mat: pd.DataFrame, vmax: float | None = None, annotate_above: float = 0.05,
                      cbar_label: str = "", fmt: str = "{:+.2f}"):
    """Signed differences on a red-grey-blue scale centred on 0. Only cells with
    |value| >= `annotate_above` are labelled; the table above holds every value."""
    values = mat.to_numpy(dtype=float)
    vmax = vmax or float(np.nanmax(np.abs(values)))
    im = ax.imshow(values, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    cols = [" · ".join(label(p) for p in c) if isinstance(c, tuple) else label(c) for c in mat.columns]
    ax.set_xticks(np.arange(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(mat.index)), [label(r) for r in mat.index])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)      # the surface gap between cells
    ax.tick_params(which="minor", length=0)
    for (i, j), v in np.ndenumerate(values):
        if np.isfinite(v) and abs(v) >= annotate_above:
            r, g, b = to_rgb(DIVERGING((v + vmax) / (2 * vmax)))
            dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5, color="white" if dark else INK)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8)
    cb.set_label(cbar_label, color=INK_2)
    return im


def density_identity(ax, x, y, gridsize: int = 70, xlabel: str = "", ylabel: str = ""):
    """Agreement of two measurements of the same thing: a log-density hexbin with the y = x line."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=SEQUENTIAL, mincnt=1, bins="log", linewidths=0)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, color=INK_2, linewidth=1.0, zorder=3)
    ax.text(lim[1], lim[1], " y = x", color=INK_2, fontsize=8, ha="left", va="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    cb = ax.figure.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8)
    cb.set_label("queries (log scale)", color=INK_2)
    return hb


def identity_scatter(ax, x, y, xlabel: str = "", ylabel: str = "", size: float = 14):
    """Paired values of the same quantity, with the y = x line."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ax.scatter(x, y, s=size, color=BLUE, alpha=0.45, linewidths=0, zorder=2)
    lim = [min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y))]
    ax.plot(lim, lim, color=INK_2, linewidth=1.0, zorder=3)
    ax.text(lim[1], lim[1], " y = x", color=INK_2, fontsize=8, ha="left", va="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_axisbelow(True)


def strip(ax, df: pd.DataFrame, group: str, value: str, order=None, ref: float | None = None,
          ref_label: str | None = None, xlabel: str | None = None, n_bootstrap: int = 2000, seed: int = 0):
    """Every unit as a grey dot, and the group mean with a bootstrap 95% interval in blue."""
    order = list(order if order is not None else pd.unique(df[group]))
    pos = _rows(ax, order)
    rng = np.random.default_rng(seed)
    for g in order:
        v = df.loc[df[group] == g, value].dropna().to_numpy(float)
        if len(v) == 0:
            continue
        ax.scatter(v, pos[g] + rng.uniform(-0.2, 0.2, len(v)), s=11, color=CONTEXT, alpha=0.8, linewidths=0,
                   zorder=2)
        boots = rng.choice(v, size=(n_bootstrap, len(v)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ax.hlines(pos[g], lo, hi, color=BLUE, linewidth=2.6, zorder=3)
        _dot(ax, v.mean(), pos[g], BLUE, size=8.5, zorder=4)
    if ref is not None:
        reference(ax, ref, ref_label)
    if xlabel:
        ax.set_xlabel(xlabel)
    return [Line2D([], [], linestyle="none", marker="o", markersize=5, markerfacecolor=CONTEXT,
                   markeredgewidth=0, label="one unit"),
            Line2D([], [], color=BLUE, linewidth=2.6, marker="o", markersize=7, markerfacecolor=BLUE,
                   markeredgecolor=SURFACE, label="mean, 95% interval")]
