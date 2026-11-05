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
    lab = {"C1a": "covariate shift (ID $-$ covariate)", "C1b": "shortcut reversal (ID $-$ reversal)",
           "C1c": "flipped names (aligned $-$ flipped)", "C2a": "counter_spurious, under reversal",
           "C2b": "similarity, under covariate", "C2c": "feature_range, under covariate",
           "C2d": "counter_prior, under flipped names", "C2e": "counter_spurious's gain: reversal $-$ covariate",
           "C3a": "demo-following index", "C3b": r"feature reliance $\rho(\pi_{true}, \pi_{behav})$"}
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
    cells = cells[cells.strategy != "counter_prior_matched"]      # exploratory variant, not in the strategy table
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


def fig_rq2_interaction(grid):
    """The one RQ2 contrast that survives Holm (C2e): counter_spurious - label_diversity per shift, abstract names."""
    from src.evaluation.analysis import grid_blocks, prepare
    from src.evaluation.bootstrap import hierarchical_contrast
    df = prepare(grid)
    envs = [("covariate", "covariate shift"), ("spurious_reversal", "shortcut reversal")]
    res = [hierarchical_contrast(grid_blocks(df, {"kind": "strategy", "a": "counter_spurious", "b": "label_diversity",
                                                   "naming": "abstract", "env": e}, "auroc"), "auroc",
                                 n_bootstrap=2000, seed=0) for e, _ in envs]
    per_task = np.array([r["per_task"] for r in res], dtype=float)          # 2 x tasks, same task order
    fig, ax = plt.subplots(figsize=(WIDTH * 0.55, 2.4))
    for t in per_task.T:
        ax.plot([0, 1], t, color=LIGHT, linewidth=0.7, zorder=1)
    est = [r["estimate"] for r in res]
    ax.plot([0, 1], est, color=HIGHLIGHT, linewidth=1.6, zorder=3)
    for x, r in enumerate(res):
        ax.plot([x, x], [r["ci_low"], r["ci_high"]], color=HIGHLIGHT, linewidth=1.6, zorder=3)
        ax.plot(x, r["estimate"], "o", color=HIGHLIGHT, mec="white", ms=6, zorder=4)
        ax.text(x + (-0.07 if x == 0 else 0.07), r["estimate"], f"{r['estimate']:+.3f}", color=INK,
                fontsize=8, va="center", ha="right" if x == 0 else "left")
    ref_line(ax, 0, axis="y")
    ax.text(-0.43, 0.006, "no difference", color=INK2, fontsize=7, va="bottom", ha="left")
    ax.set_xlim(-0.45, 1.45); ax.set_xticks([0, 1], [lab for _, lab in envs])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("AUROC, counter_spurious $-$\nlabel_diversity (abstract names)")
    save(fig, "fig_rq2_interaction")


def _shift_size_panel(ax, groups, band, xmax, xticks, xlabel, title):
    """One jittered dot per value, a row per feature group, a grey band over the designed range."""
    rng = np.random.default_rng(0)
    for i, x in enumerate(groups.values()):
        ax.plot(x, i + rng.uniform(-0.12, 0.12, len(x)), "o", ms=3, color=INK2, alpha=0.7, mec="none")
    ax.axvspan(*band, color="#f2f2f2", zorder=0)
    ax.set_yticks(range(len(groups)), list(groups)); ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_xlim(0, xmax); ax.set_xticks(xticks); ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left")


def _shift_cost_panel(fig, ax, probe, xlabel, yardstick=False):
    """ID minus shifted AUROC per strategy and naming, with 95% intervals (exploratory rows of covariate_probes.csv)."""
    e = pd.read_csv(H / "p4" / "covariate_probes.csv")
    e = e[e.id == probe]
    strategies = ["random", "label_diversity", "counter_spurious", "rule_diversity", "similarity"]
    col = dict(zip(NAMINGS, [GREY, OKABE["blue"], OKABE["vermillion"]]))
    off = dict(zip(NAMINGS, [-0.22, 0.0, 0.22]))
    for i, s in enumerate(strategies):
        for n in NAMINGS:
            q = e[(e.strategy == s) & (e.naming == n)].iloc[0]
            ax.plot([q.ci_low, q.ci_high], [i + off[n]] * 2, color=col[n], linewidth=1.2, zorder=2)
            ax.plot(q.estimate, i + off[n], "o", ms=3.8, color=col[n], mec="white", mew=0.5, zorder=3)
    ref_line(ax, 0, "no cost")
    if yardstick:
        ax.axvline(0.204, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
        ax.text(0.204, 0.995, " cost of flipped\n names (0.20)", transform=ax.get_xaxis_transform(), fontsize=7,
                color=INK2, va="top", ha="left")
    ax.set_yticks(range(len(strategies)), [STRAT_LAB[s] for s in strategies])
    ax.set_ylim(len(strategies) - 0.5, -0.7); ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel); ax.set_title("What it costs Qwen", loc="left")
    handles = [Line2D([], [], marker="o", color=col[n], ms=4, linewidth=1.2, label=f"{n} names") for n in NAMINGS]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.47, 0.98), ncol=3, frameon=False,
               fontsize=7.5, handlelength=1.4, columnspacing=1.0)


def fig_covariate_strength():
    """The covariate shift is not small, yet costs no strategy much: shift sizes from the task manifest (left),
    ID - covariate AUROC for every strategy and naming from the exploratory E1 rows (right)."""
    tasks = load_suite("eval", CFG)["tasks"]
    labels = {"towards_1": "rule feature pushed\ntowards label 1", "towards_0": "rule feature pushed\ntowards label 0",
              "distractor": "distractor"}
    groups = {lab: [abs(d) for t in tasks for r, d in zip(t["covariate_shift"]["shift_roles"],
                                                          t["covariate_shift"]["shift_delta"]) if r == role]
              for role, lab in labels.items()}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.4), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.75})
    _shift_size_panel(a1, groups, (1, 2), 2.3, [0, 1, 2], "shift (standard deviations)",
                      "How far the shift moves the inputs")
    _shift_cost_panel(fig, a2, "E1", "AUROC, ID $-$ covariate (positive = shift hurts)", yardstick=True)
    save(fig, "fig_covariate_strength")


def fig_covariate_dose(cells):
    """Dose-response: per task, how far the covariate shift moved the two rule features (mean |delta|, SD)
    against the AUROC drop (ID - covariate), averaged over the five strategies, three namings and seeds."""
    from scipy.stats import spearmanr
    size = {t["task_id"]: np.mean([abs(d) for r, d in zip(t["covariate_shift"]["shift_roles"],
                                                          t["covariate_shift"]["shift_delta"]) if r != "distractor"])
            for t in load_suite("eval", CFG)["tasks"]}
    strategies = ["random", "label_diversity", "counter_spurious", "rule_diversity", "similarity"]
    c = cells[cells.env.isin(["id", "covariate"]) & cells.strategy.isin(strategies)]
    per = c.groupby(["task_id", "env"]).auroc.mean().unstack("env")
    drop, x = per["id"] - per["covariate"], per.index.map(size).to_numpy()
    fig, ax = plt.subplots(figsize=(WIDTH * 0.6, 2.4))
    ax.axhline(0, color=GREY, linewidth=0.8, zorder=1)
    ax.text(2.04, 0.006, "no drop", fontsize=7, color=INK2, va="bottom", ha="right")
    ax.plot(x, drop, "o", ms=4.5, color=OKABE["blue"], mec="white", mew=0.5, zorder=3)
    ax.text(0.02, 0.97, f"Spearman $\\rho$ = {spearmanr(x, drop)[0]:.2f}  (24 tasks)", transform=ax.transAxes,
            fontsize=7.5, color=INK2, va="top")
    ax.set_xlim(1.0, 2.05); ax.set_ylim(-0.25, 0.25)
    ax.set_xlabel("how far the shift moves the rule features (standard deviations)")
    ax.set_ylabel("AUROC drop under the shift\n(ID $-$ covariate)")
    ax.set_title("Bigger shifts do not cost more", loc="left")
    save(fig, "fig_covariate_dose")


def fig_variance_shift():
    """Exploratory variance shift (covariate_scale): every feature but the shortcut stretched by its factor (left),
    ID - covariate_scale AUROC per strategy and naming from the E2 rows (right)."""
    tasks = load_suite("eval", CFG)["tasks"]
    groups = {"rule features (3)": [], "distractors (5)": [], "noise feature": []}
    for t in tasks:
        lb = set(t["load_bearing_generator_idx"])
        for j, s in zip(t["covariate_scale"]["scale_features"], t["covariate_scale"]["scale"]):
            key = "rule features (3)" if j in lb else "noise feature" if j == 9 else "distractors (5)"
            groups[key].append(s)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.4), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.75})
    _shift_size_panel(a1, groups, (1.5, 3), 3.3, [1, 2, 3], "spread after the shift ($\\times$ original)",
                      "How much the shift stretches the inputs")
    a1.text(0.05, 2.45, "shortcut: not stretched", fontsize=7, color=INK2, va="center")
    a1.set_ylim(2.7, -0.5)
    _shift_cost_panel(fig, a2, "E2", "AUROC, ID $-$ variance shift (positive = shift hurts)")
    save(fig, "fig_variance_shift")


def fig_reversal_direction(cells):
    cells = cells[cells.strategy != "counter_prior_matched"]      # exploratory variant, not in the strategy table
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


def fig_base_sata():
    """Exploratory: every strategy run on Qwen2.5-7B (base), by naming (panels) and environment (dot colour),
    as mean AUROC over 24 tasks. Rows sorted by each strategy's weakest environment under abstract names."""
    from src.evaluation.analysis import cell_metrics
    grid = pd.concat([pd.read_parquet(H / "p4" / "grid_base.parquet"),
                      pd.read_parquet(H / "p5" / "grid_sata_base.parquet")], ignore_index=True)
    c = cell_metrics(grid)
    c = c[(c.label_mode == "gold") & c.strategy.isin(["sata", "counter_spurious", "random", "counter_prior", "zero_shot"])]
    m = c.groupby(["naming", "strategy", "env", "task_id"]).auroc.mean().groupby(["naming", "strategy", "env"]).mean().reset_index()
    order = m[m.naming == "abstract"].groupby("strategy").auroc.min().sort_values(ascending=False).index.tolist()
    m["_row"] = m.strategy
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.8, 2.3), sharey=True, sharex=True, gridspec_kw={"wspace": 0.08})
    for ax, n in zip(axes, ["abstract", "flipped"]):
        dot_rows(ax, m[m.naming == n], order, "auroc", "env", ENV_COL)
        ref_line(ax, 0.5, "chance" if n == "abstract" else None)
        ax.set_title(f"{n} names", loc="left", fontsize=9)
        ax.set_xlim(0.40, 0.68)
    for lab, o in zip(axes[0].get_yticklabels(), order):
        if o == "sata":
            lab.set_fontweight("bold"); lab.set_color(INK)
    fig.supxlabel("AUROC on Qwen2.5-7B (base), mean over 24 tasks", fontsize=8, y=-0.04)
    direct_key(fig, [(ENV_LAB[e], ENV_COL[e]) for e in ENV_COL], y=0.95, x0=0.22)
    save(fig, "fig_base_sata")


def fig_rq4():
    r = pd.read_parquet(H / "p5" / "rq4_contrasts.parquet")
    k = pd.read_parquet(H / "p5" / "topk_contrasts.parquet")
    d = pd.concat([r, k[k.id.str.startswith("K")]], ignore_index=True)
    lab = {"R1": "SATA $-$ label_diversity, spurious reversal", "R2": "SATA $-$ label_diversity, covariate",
           "R3": "SATA $-$ label_diversity, ID", "R4": "SATA $-$ similarity, mean of 3 environments",
           "R5": "SATA $-$ SATA without the query, mean of 3", "R6": "SATA $-$ label_diversity, flipped names, ID",
           "K1": "abstract names", "K2": "aligned names", "K3": "flipped names"}
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    coefficient_plot(ax, d, lab, groups=[("Planned RQ4 family (Holm over 6)", [f"R{i}" for i in range(1, 7)]),
                                         ("Top-$k$ $-$ drawn SATA, mean of 3 environments (Holm over 3)", ["K1", "K2", "K3"])],
                     xlabel="difference in AUROC (positive favours the first)")
    save(fig, "fig_rq4")


def fig_mi(cells_grid):
    m = pd.read_parquet(H / "p5" / "mi_contrasts.parquet")
    lab = {"R7": "SATA-MI $-$ SATA, ID", "R8": "SATA-MI $-$ SATA-names, ID",
           "R9": "SATA-MI $-$ counter_prior, ID", "R10": "SATA-MI $-$ SATA, mean of 3"}
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


def fig_lens_strategies(la):
    """fig_lens with every strategy in its own colour and no SATA-MI (Okabe-Ito subset, worst CVD dE 13)."""
    curves = la.groupby(["naming", "strategy", "layer"]).auroc.mean().reset_index()
    spec = [("zero_shot", OKABE["orange"]), ("random", OKABE["blue"]), ("label_diversity", OKABE["sky"]),
            ("counter_prior", OKABE["vermillion"]), ("sata", INK)]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, n in zip(axes, NAMINGS):
        for s, col in spec:
            c = curves[(curves.naming == n) & (curves.strategy == s)]
            ax.plot(c.layer, c.auroc, color=col, linewidth=1.3)
        ref_line(ax, 0.5, axis="y")
        ax.axvline(23, color=GREY, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
        ax.set_title(f"{n} names", loc="left")
        ax.set_xlim(0, 28); ax.set_xticks([0, 7, 14, 21, 28])
    axes[0].text(23, 0.405, "layer 23 ", ha="right", fontsize=7, color=INK2)
    axes[0].set_ylabel("logit-lens AUROC")
    axes[1].set_xlabel("layer")
    axes[0].set_ylim(0.40, 0.70)
    handles = [Line2D([], [], color=col, linewidth=1.6, label=STRAT_LAB[s]) for s, col in spec]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=len(spec), frameon=False,
               handlelength=1.6, columnspacing=1.2)
    save(fig, "fig_lens_strategies")


def fig_lens_gap(la):
    from src.evaluation.mechanisms import critical_layer
    fig, ax = plt.subplots(figsize=(WIDTH * 0.8, 2.5))
    spec = [("zero_shot", OKABE["orange"]), ("sata", GREY), ("random", OKABE["blue"])]
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
    fig_rq2_interaction(pd.read_parquet(H / "p4" / "grid.parquet"))
    fig_covariate_strength()
    fig_covariate_dose(cells)
    fig_variance_shift()
    fig_reversal_direction(cells)
    fig_pi_behav()
    fig_sata_proxy()
    fig_base_sata()
    fig_rq4()
    fig_mi(grid[grid.label_mode == "gold"])
    A, la = _lens()
    fig_lens(la)
    fig_lens_strategies(la)
    fig_lens_gap(la)
    fig_probes()
    fig_steering(A)
