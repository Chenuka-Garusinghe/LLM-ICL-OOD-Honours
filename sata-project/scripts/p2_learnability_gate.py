#!/usr/bin/env python3
"""P2 learnability gate from the pilot grid (docs/research_plan.md, P2).

Reads results/v3/synthetic/p2_learnability.parquet (written by
scripts/run_synth_grid.py --run p2_learnability ...) and, for each model and
k, compares AUROC with gold against shuffled demonstration labels. Go:
difference >= 0.05 with a hierarchical-bootstrap 95% interval above 0; the
smallest passing k is used from then on. Writes
results/v3/synthetic/p2/learnability_gate.json and .csv.

Usage: python scripts/p2_learnability_gate.py [--results PATH] [--bootstrap 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.gates import LEARNABILITY_THRESHOLD, learnability_gate, smallest_passing_k  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(resolve_path(config.paths.results) / "p2_learnability.parquet"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    results = pd.read_parquet(args.results)
    gate = learnability_gate(results, n_bootstrap=args.bootstrap)
    chosen = smallest_passing_k(gate)
    out_dir = resolve_path(config.paths.results) / "p2"
    out_dir.mkdir(parents=True, exist_ok=True)
    gate.to_csv(out_dir / "learnability_gate.csv", index=False)
    summary = {
        "threshold": LEARNABILITY_THRESHOLD, "n_bootstrap": args.bootstrap,
        "smallest_passing_k": chosen, "gate": gate.to_dict(orient="records"),
    }
    (out_dir / "learnability_gate.json").write_text(json.dumps(summary, indent=2, default=float))
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(gate.to_string(index=False))
    print(f"\nsmallest passing k: {chosen}")
    print(f"-> {out_dir / 'learnability_gate.json'}")


if __name__ == "__main__":
    main()
