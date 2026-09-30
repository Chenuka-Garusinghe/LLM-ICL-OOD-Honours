#!/usr/bin/env python3
"""Gate G3 (docs/research_plan.md, P3): do the names carry priors?

Pre-registered: the prior surrogate's out-of-fold R^2 >= 0.5 in every domain,
and zero-shot AUROC(aligned) - AUROC(flipped) >= 0.15. On Qwen2.5-7B-Instruct
the R^2 criterion failed (0.20 loan, 0.28 medical) while the separation
passed, and the user decided (30 September 2026, hiccups/14) to keep the
naming factor with counter_prior and the prior-data conflict computed from
the model's measured zero-shot answers on the pool rows (PoolPrior) instead
of the surrogate. This script records both criteria and, once the pool
priors exist, the conflict check: C_t below 1/2 under aligned names and
above 1/2 under flipped names.

  python scripts/p3_naming_gate.py --models qwen2.5-7b-instruct

Inputs: p3/zero_shot_pilot_provisional.parquet (the pilot zero-shot run with
the names that became final), p3/screen_<model>.json (R^2) and, if present,
p3/pool_prior_<model>.parquet. Output: p3/naming_gate.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.gates import naming_gate, zero_shot_by_naming  # noqa: E402


def pool_conflicts(path: Path) -> pd.DataFrame:
    """C_t per (task, naming) from the measured, median-centred zero-shot margins."""
    t = pd.read_parquet(path)
    rows = []
    for (task, naming), g in t.groupby(["task_id", "naming"]):
        m = (g["logprob_1"] - g["logprob_0"]).to_numpy()
        pred = (m - np.median(m)) > 0
        rows.append({"task_id": task, "naming": naming,
                     "conflict": float(np.mean(pred != (g["label"].to_numpy() == 1)))})
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="qwen2.5-7b-instruct")
    parser.add_argument("--results", default="p3/zero_shot_pilot_provisional")
    args = parser.parse_args()

    res_dir = resolve_path(config.paths.results)
    results = pd.read_parquet(res_dir / f"{args.results}.parquet")
    report = {"results": args.results, "models": {}}
    for model in args.models.split(","):
        with open(res_dir / "p3" / f"screen_{model}.json") as f:
            fits = json.load(f)["fits"]
        res = results[results["model"] == model]
        gate = naming_gate(res, {d: fits[d]["r2_cv"] for d in fits if d != "abstract"})
        gate["abstract_r2"] = fits["abstract"]["r2_cv"]
        gate["zero_shot_by_naming"] = zero_shot_by_naming(res).drop(columns="model").to_dict(orient="records")
        by_domain = {}
        for dom, g in res.groupby("domain"):
            by_domain[dom] = zero_shot_by_naming(g).set_index("naming")["auroc"].to_dict()
        gate["zero_shot_auroc_by_domain"] = by_domain
        gate["decision"] = ("G3 fails as pre-registered (R^2). By the user's decision of 30 September 2026 the "
                            "naming factor is kept: counter_prior and C_t use the measured pool priors, so the "
                            "surrogate is descriptive only; the separation criterion passes.")
        pp = res_dir / "p3" / f"pool_prior_{model}.parquet"
        if pp.exists():
            c = pool_conflicts(pp)
            w = c.pivot_table(index="task_id", columns="naming", values="conflict")
            gate["conflict_check"] = {
                "n_tasks": len(w), "mean": w.mean().to_dict(),
                "aligned_below_half": int((w["aligned"] < 0.5).sum()) if "aligned" in w else None,
                "flipped_above_half": int((w["flipped"] > 0.5).sum()) if "flipped" in w else None,
                "abstract_above_half": int((w["abstract"] > 0.5).sum()) if "abstract" in w else None,
            }
            c.to_csv(res_dir / "p3" / f"conflicts_{model}.csv", index=False)
        report["models"][model] = gate
        print(f"{model}: R^2 {gate['r2_by_domain']} (pre-registered criterion {'met' if gate['passes_r2'] else 'not met'}); "
              f"zero-shot AUROC aligned - flipped {gate['delta_auroc']:+.3f} [{gate['ci_low']:+.3f}, {gate['ci_high']:+.3f}], "
              f"{gate['tasks_positive']}/{gate['n_tasks']} tasks; accuracy {gate['delta_accuracy']:+.3f}")
        if "conflict_check" in gate:
            print(f"  C_t: {gate['conflict_check']}")
    with open(res_dir / "p3" / "naming_gate.json", "w") as f:
        json.dump(report, f, indent=1, default=float)


if __name__ == "__main__":
    main()
