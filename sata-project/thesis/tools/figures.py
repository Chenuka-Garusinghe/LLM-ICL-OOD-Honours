"""Thesis figures, redrawn from the result files (run from sata-project/: python thesis/tools/figures.py).

Design rules, after Healy (2019), Data Visualization: A Practical Introduction:
- encode the quantity of interest as position on a common scale (dot plots, dot-and-interval
  plots, lines), never as area, angle or colour alone (ch. 1.4-1.5);
- show the uncertainty of every estimate (ch. 6.3), and order categories by value or by a
  substantive grouping, not alphabetically;
- few channels per chart: one colour per meaning, grey for context and one hue for emphasis
  (ch. 1.3, 8.1-8.2); a colour-blind-safe qualitative palette (Okabe-Ito, the palette Healy
  recommends from Chang 2013), a ColorBrewer sequential palette for ordered values;
- label directly instead of using legends where it fits, and keep labels substantive;
- one typeface (the thesis's Times-like serif), thin marks, light gridlines, vector output.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path  # noqa: E402  (first src import)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.data.suites import load_suite  # noqa: E402
from src.evaluation.analysis import behavioural_importance, cell_metrics  # noqa: E402
from src.evaluation.mechanisms import layer_auroc, load_lens  # noqa: E402

CFG = load_config()
RES = resolve_path(CFG.paths.results)
H = RES / "h200"
OUT = Path(__file__).resolve().parents[1] / "figures"
WIDTH = 6.27                                   # the thesis text width, inches

# ---------------------------------------------------------------- style
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
         "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442"}
INK, INK2, GREY, LIGHT = "#1a1a1a", "#555555", "#9a9a9a", "#d4d4d4"
HIGHLIGHT = OKABE["blue"]
ENV_COL = {"id": OKABE["blue"], "covariate": OKABE["orange"], "spurious_reversal": OKABE["green"]}
ENV_LAB = {"id": "ID", "covariate": "covariate", "spurious_reversal": "spurious reversal"}
NAMINGS = ("abstract", "aligned", "flipped")
BLUES = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]   # ColorBrewer Blues
STRAT_LAB = {"zero_shot": "zero-shot", "random": "random", "label_diversity": "label_diversity",
             "feature_range": "feature_range", "rule_diversity": "rule_diversity",
             "counter_spurious": "counter_spurious", "similarity": "similarity",
             "counter_prior": "counter_prior", "counter_prior_matched": "counter_prior_matched",
             "sata": "SATA", "sata_mi": "SATA-MI", "sata_names": "SATA-names", "sata_agnostic": "SATA (no query)"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def ref_line(ax, x, text=None, axis="x", where="top"):
    (ax.axvline if axis == "x" else ax.axhline)(x, color=GREY, linewidth=0.8, zorder=1)
    if text:
        if axis == "x":
            ax.text(x, 0.995 if where == "top" else 0.005, f" {text}", transform=ax.get_xaxis_transform(),
                    color=INK2, fontsize=7, va="top" if where == "top" else "bottom", ha="left")
        else:
            ax.text(1.0, x, f"{text} ", transform=ax.get_yaxis_transform(), color=INK2, fontsize=7,
                    va="bottom", ha="right")


def boot_ci(per_task: np.ndarray, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(per_task, float)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), *np.quantile(b, [0.025, 0.975])


def coefficient_plot(ax, df, labels, groups=None, xlabel="", note=True):
    """Dot-and-interval rows (Healy 6.3): significant filled in the highlight colour, the rest hollow grey;
    Holm p printed at the right; groups separated by a gap and a bold heading."""
    rows, y, yt = [], 0, []
    for gname, ids in (groups or [(None, list(df.id))]):
        if gname:
            yt.append((y, gname, True)); y += 1
        for i in ids:
            rows.append((i, y)); yt.append((y, labels[i], False)); y += 1
        y += 0.4
    pos = dict(rows)
    for _, r in df[df.id.isin(pos)].iterrows():
        sig = bool(r.get("significant", False))
        c = HIGHLIGHT if sig else GREY
        ax.hlines(pos[r.id], r.ci_low, r.ci_high, color=c, linewidth=1.4, zorder=2)
        ax.plot(r.estimate, pos[r.id], "o", ms=4.5, mfc=c if sig else "white", mec=c, mew=1.2, zorder=3)
        if note:
            ax.text(1.005, pos[r.id], f"{r.p_holm:.2f}" if r.p_holm >= 0.01 else f"{r.p_holm:.3f}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=INK2)
    ax.set_yticks([p for p, _, _ in yt], [t for _, t, _ in yt])
    for lab, (_, _, bold) in zip(ax.get_yticklabels(), yt):
        if bold:
            lab.set_fontweight("bold"); lab.set_color(INK)
    ax.set_ylim(y - 0.6, -0.8)
    ax.grid(axis="y", visible=False)
    ref_line(ax, 0)
    ax.set_xlabel(xlabel)
    if note:
        ax.text(1.005, -0.8, "Holm $p$", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7.5, color=INK2)


def dot_rows(ax, df, rows, x, hue, hue_col, hue_lab=None, connect=True, ms=4.2):
    """Cleveland dot plot: one row per category, one dot per hue, joined by a thin line (gestalt connection)."""
    pos = {r: i for i, r in enumerate(rows)}
    for r in rows:
        g = df[df["_row"] == r]
        if connect and len(g) > 1:
            ax.hlines(pos[r], g[x].min(), g[x].max(), color=LIGHT, linewidth=1.6, zorder=1)
        for _, q in g.iterrows():
            ax.plot(q[x], pos[r], "o", ms=ms, color=hue_col[q[hue]], mec="white", mew=0.6, zorder=3)
    ax.set_yticks(range(len(rows)), [STRAT_LAB.get(r, r) for r in rows])
    ax.set_ylim(len(rows) - 0.5, -1.0)
    ax.grid(axis="y", visible=False)


def direct_key(fig, items, y=1.0, x0=0.08):
    handles = [Line2D([], [], marker="o", ls="", color=c, mec="white", ms=5, label=l) for l, c in items]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x0, y), ncol=len(items), frameon=False,
               handletextpad=0.2, columnspacing=1.2)


# ---------------------------------------------------------------- data
def p4_cells():
    grid = pd.concat([pd.read_parquet(f) for f in (H / "p4" / "grid.parquet", H / "p4" / "grid_cpm.parquet")],
                     ignore_index=True)
    sigma = {t["task_id"]: t["spurious_direction"] for t in load_suite("eval", CFG)["tasks"]}
    cells = cell_metrics(grid)
    cells["sigma"] = cells.task_id.map(sigma)
    return grid, cells[cells.label_mode == "gold"]


# ---------------------------------------------------------------- figures
def fig_p4_contrasts():
    c = pd.read_csv(H / "p4" / "contrasts.csv")
    lab = {"C1a": "C1a  ID $-$ covariate", "C1b": "C1b  ID $-$ spurious reversal",
           "C1c": "C1c  aligned $-$ flipped names", "C2a": "C2a  counter_spurious, reversal",
           "C2b": "C2b  similarity, covariate", "C2c": "C2c  feature_range, covariate",
           "C2d": "C2d  counter_prior, flipped names", "C2e": "C2e  C2a under reversal $-$ under covariate",
           "C3a": "C3a  demo-following index", "C3b": r"C3b  $\rho(\pi_{true}, \pi_{behav})$"}
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 4.6), gridspec_kw={"height_ratios": [11.6, 2.4, 2.4], "hspace": 0.75})
    coefficient_plot(axes[0], c[c.statistic == "auroc"], lab,
                     groups=[("RQ1: cost of each shift (random demonstrations)", ["C1a", "C1b", "C1c"]),
                             ("RQ2: strategy $-$ label_diversity", ["C2a", "C2b", "C2c", "C2d", "C2e"])],
                     xlabel="difference in AUROC")
    coefficient_plot(axes[1], c[c.id == "C3a"], lab, groups=[("RQ3: abstract $-$ flipped names", ["C3a"])],
                     xlabel="difference in demo-following index")
    coefficient_plot(axes[2], c[c.id == "C3b"], lab, groups=[("RQ3: abstract $-$ flipped names", ["C3b"])],
                     xlabel=r"difference in Spearman $\rho$")
    save(fig, "fig_p4_contrasts")


def fig_strategies_by_naming(cells):
    m = cells.groupby(["naming", "strategy", "env", "task_id"]).auroc.mean().groupby(["naming", "strategy", "env"]).mean().reset_index()
    order = (m[m.naming == "aligned"].groupby("strategy").auroc.mean().sort_values(ascending=False).index.tolist())
    m["_row"] = m.strategy
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.5), sharey=True, sharex=True, gridspec_kw={"wspace": 0.08})
    for ax, n in zip(axes, NAMINGS):
        dot_rows(ax, m[m.naming == n], order, "auroc", "env", ENV_COL)
        ref_line(ax, 0.5, "chance" if n == "abstract" else None)
        ax.set_title(f"{n} names", loc="left", fontsize=9)
        ax.set_xlim(0.38, 0.75)
    axes[1].set_xlabel("AUROC, mean over 24 tasks")
    direct_key(fig, [(ENV_LAB[e], ENV_COL[e]) for e in ENV_COL], y=0.95, x0=0.25)
    save(fig, "fig_strategies_by_naming")


def fig_reversal_direction(cells):
    sr = cells[cells.env.isin(["id", "spurious_reversal"])]
    per = (sr.groupby(["strategy", "naming", "sigma", "env", "task_id"]).auroc.mean()
             .groupby(["strategy", "naming", "sigma", "env"]).mean().unstack("env"))
    per["gap"] = per["id"] - per["spurious_reversal"]
    g = per.gap.reset_index()
    order = g[g.sigma == 1].groupby("strategy").gap.mean().sort_values(ascending=False).index.tolist()
    col = {1: OKABE["vermillion"], -1: OKABE["sky"]}
    g["_row"] = g.strategy
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.4), sharey=True, sharex=True, gridspec_kw={"wspace": 0.08})
    for ax, n in zip(axes, NAMINGS):
        dot_rows(ax, g[g.naming == n], order, "gap", "sigma", col)
        ref_line(ax, 0, "no cost" if n == "abstract" else None)
        ax.set_title(f"{n} names", loc="left")
    axes[1].set_xlabel("AUROC, ID $-$ spurious reversal (positive = reversal hurts)")
    direct_key(fig, [("prior agrees with the shortcut ($\\sigma_t=+1$)", col[1]),
                     ("prior opposes it ($\\sigma_t=-1$)", col[-1])], y=0.95, x0=0.15)
    save(fig, "fig_reversal_direction")


def fig_pi_behav():
    bi = behavioural_importance(pd.read_parquet(H / "p4" / "rq3.parquet"))
    roles = {"spurious": "shortcut", "load_bearing": "rule features (3)", "noise": "noise feature",
             "distractor": "distractors (5)"}
    m = bi.groupby(["feature_role", "naming"]).margin_drop.mean().reset_index()
    col = dict(zip(NAMINGS, [GREY, OKABE["blue"], OKABE["vermillion"]]))
    m["_row"] = m.feature_role
    fig, ax = plt.subplots(figsize=(WIDTH * 0.75, 1.9))
    dot_rows(ax, m, list(roles), "margin_drop", "naming", col)
    ax.set_yticks(range(len(roles)), list(roles.values()))
    ref_line(ax, 0, "no reliance")
    ax.set_xlabel("drop in the correct-label margin when the feature is replaced (logits)")
    direct_key(fig, [(f"{n} names", col[n]) for n in NAMINGS], y=0.97, x0=0.2)
    save(fig, "fig_pi_behav")


def fig_sata_proxy():
    gate = pd.read_parquet(RES / "p5" / "gate_s3a.parquet")
    tr = pd.read_parquet(RES / "p5" / "proxy_trained.parquet")
    keep = ["sata", "sata_designed", "sata_agnostic", "oracle_matched", "random", "label_diversity",
            "feature_range", "rule_diversity", "counter_spurious", "similarity"]
    d = pd.concat([gate, tr])
    d = d[d.method.isin(keep) & d.env.isin(ENV_COL)]
    names = {"sata": "SATA (trained)", "sata_designed": "SATA, planned targets", "sata_agnostic": "SATA, no query",
             "oracle_matched": "matched targets (oracle)"}
    m = d.groupby(["method", "env", "task_id"]).accuracy.mean().groupby(["method", "env"]).mean().reset_index()
    worst = m.groupby("method").accuracy.min().sort_values(ascending=False)
    order = worst.index.tolist()
    m["_row"] = m.method
    fig, ax = plt.subplots(figsize=(WIDTH * 0.78, 3.0))
    dot_rows(ax, m, order, "accuracy", "env", ENV_COL)
    ax.set_yticks(range(len(order)), [names.get(o, o) for o in order])
    for lab, o in zip(ax.get_yticklabels(), order):
        if o == "sata":
            lab.set_fontweight("bold"); lab.set_color(INK)
    ref_line(ax, 0.5, "chance")
    ax.set_xlabel("stand-in learner's accuracy (gradient-boosted trees fitted on the 8 chosen rows)")
    direct_key(fig, [(ENV_LAB[e], ENV_COL[e]) for e in ENV_COL], y=0.97, x0=0.25)
    save(fig, "fig_sata_proxy")


def fig_rq4():
    r = pd.read_parquet(H / "p5" / "rq4_contrasts.parquet")
    k = pd.read_parquet(H / "p5" / "topk_contrasts.parquet")
    d = pd.concat([r, k[k.id.str.startswith("K")]], ignore_index=True)
    lab = {"R1": "R1  SATA $-$ label_diversity, spurious reversal", "R2": "R2  SATA $-$ label_diversity, covariate",
           "R3": "R3  SATA $-$ label_diversity, ID", "R4": "R4  SATA $-$ similarity, mean of 3 environments",
           "R5": "R5  SATA $-$ SATA without the query, mean of 3", "R6": "R6  SATA $-$ label_diversity, flipped names, ID",
           "K1": "K1  abstract names", "K2": "K2  aligned names", "K3": "K3  flipped names"}
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    coefficient_plot(ax, d, lab, groups=[("Pre-registered RQ4 family (Holm over 6)", [f"R{i}" for i in range(1, 7)]),
                                         ("Top-$k$ $-$ drawn SATA, mean of 3 environments (Holm over 3)", ["K1", "K2", "K3"])],
                     xlabel="difference in AUROC (positive favours the first)")
    save(fig, "fig_rq4")


def fig_mi(cells_grid):
    m = pd.read_parquet(H / "p5" / "mi_contrasts.parquet")
    lab = {"R7": "R7  SATA-MI $-$ SATA, ID", "R8": "R8  SATA-MI $-$ SATA-names, ID",
           "R9": "R9  SATA-MI $-$ counter_prior, ID", "R10": "R10  SATA-MI $-$ SATA, mean of 3"}
    g = pd.concat([cells_grid, pd.read_parquet(H / "p5" / "grid_sata.parquet"), pd.read_parquet(H / "p5" / "grid_mi.parquet")],
                  ignore_index=True)
    c = cell_metrics(g)
    c = c[(c.label_mode == "gold") & (c.naming == "flipped") & (c.strategy != "counter_prior_matched")]
    per = c.groupby(["strategy", "task_id"]).auroc.mean().reset_index()
    est = pd.DataFrame([(s, *boot_ci(x.auroc.to_numpy())) for s, x in per.groupby("strategy")],
                       columns=["strategy", "mean", "lo", "hi"]).sort_values("mean", ascending=False)
    zs = est[est.strategy == "zero_shot"]["mean"].iloc[0]
    est = est[~est.strategy.isin(["zero_shot", "sata_agnostic", "sata_designed"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.7), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.95})
    coefficient_plot(a1, m.assign(significant=False), lab, groups=[("Flipped names (Holm over 4)", list(lab))],
                     xlabel="difference in AUROC")
    for i, (_, r) in enumerate(est.iterrows()):
        c_ = HIGHLIGHT if r.strategy == "sata_mi" else GREY
        a2.hlines(i, r.lo, r.hi, color=c_, linewidth=1.4)
        a2.plot(r["mean"], i, "o", ms=4.5, color=c_, mec="white", mew=0.6, zorder=3)
    a2.set_yticks(range(len(est)), [STRAT_LAB.get(s, s) for s in est.strategy])
    for lab_, s in zip(a2.get_yticklabels(), est.strategy):
        if s == "sata_mi":
            lab_.set_fontweight("bold"); lab_.set_color(INK)
    a2.set_ylim(len(est) - 0.5, -1.2)
    a2.grid(axis="y", visible=False)
    ref_line(a2, 0.5, "chance")
    a2.axvline(zs, color=OKABE["orange"], linewidth=0.9, zorder=1)
    a2.text(zs, -0.75, "zero-shot ", color=OKABE["orange"], fontsize=7, ha="right", va="center")
    a2.set_xlabel("AUROC under flipped names\n(mean of 3 environments, 95% CI over tasks)")
    a2.set_title("Every strategy, flipped names", loc="left", fontweight="bold", fontsize=8.5)
    save(fig, "fig_mi")


def _lens():
    A = load_lens(sorted(glob.glob(str(H / "p6" / "A" / "lens.*.parquet"))))
    return A, layer_auroc(A)


def fig_lens(la):
    curves = la.groupby(["naming", "strategy", "layer"]).auroc.mean().reset_index()
    hi = {"random": OKABE["blue"], "zero_shot": OKABE["orange"]}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, n in zip(axes, NAMINGS):
        for s in ["label_diversity", "counter_prior", "sata", "sata_mi", "random", "zero_shot"]:
            c = curves[(curves.naming == n) & (curves.strategy == s)]
            ax.plot(c.layer, c.auroc, color=hi.get(s, LIGHT), linewidth=1.6 if s in hi else 1.0, zorder=3 if s in hi else 2)
        if n == "aligned":
            ax.text(1, 0.675, "random demonstrations", color=hi["random"], fontsize=7.5)
            ax.text(1, 0.655, "zero-shot", color=hi["zero_shot"], fontsize=7.5)
        ref_line(ax, 0.5, axis="y")
        ax.axvline(23, color=GREY, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
        ax.set_title(f"{n} names", loc="left")
        ax.set_xlim(0, 28); ax.set_xticks([0, 7, 14, 21, 28])
    axes[0].text(23, 0.405, "layer 23 ", ha="right", fontsize=7, color=INK2)
    axes[0].set_ylabel("logit-lens AUROC")
    axes[1].set_xlabel("layer (grey lines: label_diversity, counter_prior, SATA, SATA-MI)")
    axes[0].set_ylim(0.40, 0.70)
    save(fig, "fig_lens")


def fig_lens_gap(la):
    from src.evaluation.mechanisms import critical_layer
    fig, ax = plt.subplots(figsize=(WIDTH * 0.8, 2.5))
    spec = [("zero_shot", OKABE["orange"]), ("sata", GREY), ("sata_mi", OKABE["green"]), ("random", OKABE["blue"])]
    for s, col in spec:
        t = critical_layer(la[la.strategy == s])["table"]
        if s == "random":
            ax.fill_between(t.layer, t.ci_low, t.ci_high, color=col, alpha=0.15, linewidth=0)
        ax.plot(t.layer, t.gap, color=col, linewidth=1.6)
        ax.text(28.4, t.gap.iloc[-1], STRAT_LAB[s], color=col, fontsize=7, va="center")
    ref_line(ax, 0, axis="y")
    ax.axhline(0.1, color=GREY, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.text(0.3, 0.103, "threshold 0.1", fontsize=7, color=INK2, va="bottom")
    ax.set_xlim(0, 28); ax.set_xticks([0, 7, 14, 21, 23, 28])
    ax.set_xlabel("layer"); ax.set_ylabel("AUROC, aligned $-$ flipped")
    save(fig, "fig_lens_gap")


def fig_probes():
    pc = pd.read_parquet(H / "p6" / "probe_curves.parquet")
    col = {"prior prediction": OKABE["orange"], "rule label": OKABE["blue"]}
    lab = {"prior prediction": "Qwen's zero-shot answer", "rule label": "rule label"}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, n in zip(axes, NAMINGS):
        for t, c in col.items():
            d = pc[(pc.naming == n) & (pc.target == t)]
            ax.plot(d.layer, d.auroc, color=c, linewidth=1.6, marker="o", ms=2.5)
        if n == "aligned":
            ax.text(1, 0.76, lab["prior prediction"], color=col["prior prediction"], fontsize=7.5)
            ax.text(1, 0.42, lab["rule label"], color=col["rule label"], fontsize=7.5)
        ref_line(ax, 0.5, axis="y")
        ax.set_title(f"{n} names", loc="left"); ax.set_xticks([0, 7, 14, 21, 28])
    axes[0].set_ylabel("probe AUROC (held-out tasks)")
    axes[1].set_xlabel("layer")
    save(fig, "fig_probes")


def fig_steering(A):
    B = load_lens(sorted(glob.glob(str(H / "p6" / "B" / "lens.*.parquet"))))
    odd = A.task_id.map(lambda t: int(t[-4:]) % 2 == 1)
    base = A[(A.strategy == "random") & A.naming.isin(["aligned", "flipped"]) & odd].assign(condition="none")
    lb = layer_auroc(pd.concat([B, base[B.columns]]), by=("condition", "naming"))
    fin = lb[lb.layer == lb.layer.max()].pivot_table(index=["naming", "task_id"], columns="condition", values="auroc")
    delta = fin.sub(fin["none"], axis=0).drop(columns="none")
    sh = (B.merge(base[["naming", "task_id", "row_id", "margin"]], on=["naming", "task_id", "row_id"], suffixes=("", "_0"))
          .assign(s=lambda d: d.margin - d.margin_0).groupby(["naming", "condition", "task_id"]).s.mean())
    alphas = [1, 2, 4, 8, 16, 32]
    layers = [8, 12, 16, 20, 24]
    acol = dict(zip(alphas, BLUES))
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.0), sharex=True, sharey="row",
                             gridspec_kw={"hspace": 0.25, "wspace": 0.08})
    for j, n in enumerate(["aligned", "flipped"]):
        for row, data in enumerate([delta.loc[n], sh.loc[n].unstack("condition")]):
            ax = axes[row, j]
            for li, L in enumerate(layers):
                conds = [(f"v_L{L}_a{a}", acol[a], "o") for a in alphas] + [(f"rand_L{L}_a{a}", GREY, "D") for a in (4, 32)]
                for k, (cnd, c, mk) in enumerate(conds):
                    x = li + (k - (len(conds) - 1) / 2) * 0.095
                    mu, lo, hi = boot_ci(data[cnd].dropna().to_numpy())
                    ax.vlines(x, lo, hi, color=c, linewidth=1.0)
                    ax.plot(x, mu, mk, ms=3.2 if mk == "o" else 2.8, color=c, mfc=c if mk == "o" else "white", mec=c, mew=0.8)
            ref_line(ax, 0, axis="y")
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(f"{n} names", loc="left")
        axes[1, j].set_xticks(range(len(layers)), [f"layer {L}" for L in layers])
    axes[0, 0].set_ylabel("$\\Delta$ final AUROC")
    axes[1, 0].set_ylabel("$\\Delta$ margin (logits)")
    handles = [Line2D([], [], marker="o", ls="", color=acol[a], ms=4, label=f"$\\alpha={a}$") for a in alphas]
    handles += [Line2D([], [], marker="D", ls="", color=GREY, mfc="white", ms=3.5, label="random direction ($\\alpha=4, 32$)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=7, frameon=False,
               handletextpad=0.1, columnspacing=0.9)
    save(fig, "fig_steering")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    grid, cells = p4_cells()
    fig_p4_contrasts()
    fig_strategies_by_naming(cells)
    fig_reversal_direction(cells)
    fig_pi_behav()
    fig_sata_proxy()
    fig_rq4()
    fig_mi(grid[grid.label_mode == "gold"])
    A, la = _lens()
    fig_lens(la)
    fig_lens_gap(la)
    fig_probes()
    fig_steering(A)
