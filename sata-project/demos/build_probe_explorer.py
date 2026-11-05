"""Build demos/probe_data.js for demos/probe_explorer.html (the interactive linear-probe explainer).

The thesis probes (src/evaluation/mechanisms.py, probe_curves) read Qwen2.5-7B-Instruct's last-position
residual state h_l (3,584 numbers) at every second layer, for the random-demonstration runs of P6 stage A
(24 evaluation tasks x 60 queries, per naming), and predict either the rule label (y_clean) or Qwen's own
zero-shot answer (`prior`, median-centred; with_prior). A browser cannot hold 3 x 15 x 1,440 x 3,584
numbers comfortably, so each (naming, layer) matrix is standardised and reduced to its top K principal
components, stored as int8 with one scale per component. A probe on 32 components reproduces the
full-dimension curves closely (checked 4 Nov 2026).

Usage: .venv/bin/python demos/build_probe_explorer.py   (from the sata-project root; a few seconds)
"""

from __future__ import annotations

import base64
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src.evaluation.mechanisms import load_lens, load_residuals, with_prior  # noqa: E402

P6 = ROOT / "results" / "v3" / "synthetic" / "h200" / "p6"
K = 32
LAYERS = list(range(0, 29, 2))
NAMINGS = ["abstract", "aligned", "flipped"]


def main() -> None:
    A = with_prior(load_lens(sorted(glob.glob(str(P6 / "A" / "lens.*.parquet")))))
    rnd = A[(A.strategy == "random") & A.resid_file.notna()].reset_index(drop=True)
    res = load_residuals(rnd, P6 / "A")                       # (rows, 29, 3584) float16
    tasks = sorted(rnd.task_id.unique())
    out = {"K": K, "layers": LAYERS, "namings": NAMINGS, "d_model": int(res.shape[2]),
           "tasks": tasks, "data": {}}
    for n in NAMINGS:
        sel = (rnd.naming == n).to_numpy()
        g = rnd[sel]
        block = {"task": [tasks.index(t) for t in g.task_id],
                 "env": [["id", "covariate", "spurious_reversal"].index(e) for e in g.env],
                 "rule": g.y_clean.astype(int).tolist(), "prior": g.prior.astype(int).tolist(),
                 "layers": {}}
        for L in LAYERS:
            X = res[sel, L].astype(np.float32)
            if float(X.std(0).max()) < 1e-4:
                # Layer 0 at the last position is the embedding of the prompt's final token: the same
                # for every query, so there is nothing to compress or to probe.
                q, scale, var = np.zeros((len(X), K), np.int8), np.ones(K), np.zeros(K)
            else:
                X = (X - X.mean(0)) / (X.std(0) + 1e-6)
                pca = PCA(K, random_state=0).fit(X)
                Z = pca.transform(X)
                scale = np.abs(Z).max(0) / 127.0
                q = np.clip(np.round(Z / scale), -127, 127).astype(np.int8)
                var = pca.explained_variance_ratio_
            block["layers"][str(L)] = {"scale": [float(f"{s:.6g}") for s in scale],
                                       "var": [round(float(v), 4) for v in var],
                                       "constant": bool(var.sum() == 0),
                                       "z": base64.b64encode(q.tobytes()).decode()}
        out["data"][n] = block
    pc = pd.read_parquet(P6 / "probe_curves.parquet")
    out["thesis"] = {f"{r.naming}|{r.target}|{int(r.layer)}": round(float(r.auroc), 4) for r in pc.itertuples()}
    path = ROOT / "demos" / "probe_data.js"
    path.write_text("window.PROBE_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB): {sel.sum()} queries per naming, "
          f"{len(LAYERS)} layers, {K} components")


if __name__ == "__main__":
    main()
