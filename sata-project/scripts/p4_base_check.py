#!/usr/bin/env python3
"""P2 checks for Qwen2.5-7B base at k = 8 (run by scripts/run_p4.sh, stage `base`).

Reads results/v3/synthetic/p4/base_p2.parquet (the pilot learnability grid:
zero-shot and label_diversity with gold and shuffled labels, ID queries) and
writes p4/base_p2_gate.json with the learnability gate and the median label
mass. Exits with status 1 when the label mass is below 0.9, so the chain can
skip the reduced grid: a model that does not answer with the label tokens
gives no valid scores (docs/research_plan.md, P2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402

import pandas as pd  # noqa: E402

from src.evaluation.gates import LABEL_MASS_THRESHOLD, label_mass, learnability_gate  # noqa: E402


def main() -> int:
    config = load_config()
    res = resolve_path(config.paths.results) / "p4"
    df = pd.read_parquet(res / "base_p2.parquet")
    gate = learnability_gate(df)
    record = {"gate": gate.to_dict(orient="records"),
              "label_mass_all": float(label_mass(df).median()),
              "rule": f"reduced grid runs if the median label mass is at least {LABEL_MASS_THRESHOLD}"}
    valid = bool(gate["valid"].all()) and record["label_mass_all"] >= LABEL_MASS_THRESHOLD
    record["valid"] = valid
    (res / "base_p2_gate.json").write_text(json.dumps(record, indent=1, default=float))
    print(gate.to_string())
    print("valid answer format:", valid)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
