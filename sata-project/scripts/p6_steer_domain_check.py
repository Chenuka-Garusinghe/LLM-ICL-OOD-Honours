"""Does the P6 steering vector depend on the domain it was fitted on?

The original vector was fitted on the even-indexed evaluation tasks, which are all loan tasks,
and tested on the odd-indexed ones, which are all medical. Refit it on the medical tasks from the
same stage-A residuals and compare directions layer by layer, against a same-domain baseline
(two disjoint halves of one domain) at matched size.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.mechanisms import load_lens, load_residuals, steering_vectors, with_prior  # noqa: E402

A = ROOT / "results/v3/synthetic/h200/p6/A"
LAYERS = [8, 12, 16, 20, 24]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def idx(t):
    return int(t.rsplit("_", 1)[1])


df = with_prior(load_lens(sorted(A.glob("lens.*.parquet"))))
rnd = df[(df.strategy == "random") & df.resid_file.notna()].reset_index(drop=True)
resid = load_residuals(rnd, A)
print(f"random-demonstration rows with residuals: {len(rnd)}, tasks {rnd.task_id.nunique()}, residuals {resid.shape}")


def fit(mask):
    m = np.asarray(mask)
    return steering_vectors(rnd[m].reset_index(drop=True), resid[m])


def label_halves(mask):
    """The two per-label differences whose average is v (to show they point opposite ways)."""
    sub, r = rnd[np.asarray(mask)].reset_index(drop=True), resid[np.asarray(mask)]
    conflict = (sub.y_clean != sub.prior).to_numpy() & sub.prior.notna().to_numpy()
    fd_all = (sub.pred == sub.y_clean).to_numpy()
    out = []
    for c in (0, 1):
        lab = (sub.y_clean == c).to_numpy() & conflict
        out.append(r[lab & fd_all].astype(np.float64).mean(0) - r[lab & ~fd_all].astype(np.float64).mean(0))
    return out


loan = rnd.task_id.map(lambda t: idx(t) % 2 == 0)
med = ~loan
assert set(rnd[loan].task_id.map(idx) % 2) == {0}

v_loan, v_med = fit(loan), fit(med)
saved = np.load(ROOT / "models/p6_steer.npz")["v"]
print(f"refit on loan tasks reproduces the saved vector: max |diff| = {np.abs(v_loan['v'] - saved).max():.2e}")
print("conflict-query counts  loan:", v_loan["counts"], " medical:", v_med["counts"])

# Same-domain baseline at matched size: within each domain, two disjoint halves of 6 tasks,
# each with 3 linear and 3 tree tasks (the suite lists linear tasks first: indices 0-11).
def halves(domain_parity):
    ids = sorted({t for t in rnd.task_id if idx(t) % 2 == domain_parity}, key=idx)
    lin, tree = [t for t in ids if idx(t) < 12], [t for t in ids if idx(t) >= 12]
    return set(lin[0::2] + tree[0::2]), set(lin[1::2] + tree[1::2])

(la, lb), (ma, mb) = halves(0), halves(1)
half = {k: fit(rnd.task_id.isin(s)) for k, s in [("la", la), ("lb", lb), ("ma", ma), ("mb", mb)]}

d_loan, d_med = label_halves(loan), label_halves(med)
rng = np.random.default_rng(0)
rand_cos = [cos(rng.normal(size=resid.shape[2]), rng.normal(size=resid.shape[2])) for _ in range(2000)]

rows = []
for l in LAYERS:
    vl, vm = v_loan["v"][l], v_med["v"][l]
    within = np.mean([cos(half["la"]["v"][l], half["lb"]["v"][l]), cos(half["ma"]["v"][l], half["mb"]["v"][l])])
    across = np.mean([cos(half[a]["v"][l], half[b]["v"][l]) for a in ("la", "lb") for b in ("ma", "mb")])
    rows.append({
        "layer": l,
        "cos(loan 12, medical 12)": cos(vl, vm),
        "same domain, 6 vs 6": within,
        "across domains, 6 vs 6": across,
        "norm medical / loan": np.linalg.norm(vm) / np.linalg.norm(vl),
        "label halves cos, loan": cos(d_loan[0][l], d_loan[1][l]),
        "label halves cos, medical": cos(d_med[0][l], d_med[1][l]),
    })
res = pd.DataFrame(rows).set_index("layer")
pd.set_option("display.width", 200)
print(res.round(3).to_string())
print(f"two random directions in {resid.shape[2]} dimensions: |cos| 95th percentile = {np.quantile(np.abs(rand_cos), 0.95):.3f}")
res.to_csv(A.parent / "steer_domain_check.csv")
