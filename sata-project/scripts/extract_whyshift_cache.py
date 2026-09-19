"""ACS PUMS data extraction for the WHYSHIFT arm (Notebook 01).

Two tasks, each a real state-level population shift with a measurable DISDE
decomposition (Liu et al., "WHYSHIFT", NeurIPS 2023) into a covariate (P(X))
component and a concept (P(Y|X)) component -- unlike TableShift, where the
"concept shift" metric turns out to be covariate shift under another name
(see ../TABLESHIFT_AUDIT.md). The decomposition itself is computed later, in
Notebook 01 (via src/data/whyshift_loader.screen_disde_shifts), against
several candidate target states per task; this script's only job is to fetch
and cache the raw per-state data.

Tasks (feature lists and preprocessing filters match the standard folktables
`ACSIncome` / `ACSPublicCoverage` problem definitions exactly, so results are
comparable to the wider ACS-fairness/shift literature):

  whyshift_income   -- ACSIncome: predict PINCP > $50,000.
                       Source: California (CA). Candidate targets: MS, WV,
                       AR, AL, WA, NY, CO, FL.
  whyshift_pubcov   -- ACSPublicCoverage: predict PUBCOV == 1 (has public
                       health insurance), restricted to low-income
                       non-Medicare-eligible adults.
                       Source: Texas (TX). Candidate targets: MA, VT, MN,
                       CT, FL, AZ, GA, NV.

`ST` (state) is dropped from whyshift_pubcov's usable features (it is part
of the official ACSPublicCoverage feature list, but here it is the split
variable itself -- keeping it would let a downstream classifier trivially
read off the domain). `ST` is not in ACSIncome's feature list at all.

Unlike TableShift, `folktables` and `whyshift` install cleanly into this
project's own environment (no numpy/ray/xport pin conflicts) -- this script
runs directly, no separate isolated environment needed.

Codebook: rather than hand-maintaining a value-mapping table, this pulls the
actual Census PUMS data dictionary for the survey year (`ACSDataSource.
get_definitions()`) and derives each categorical feature's code -> label
mapping via folktables' own `generate_categories()`. This is the real
Census text (e.g. RAC1P's 9 categories, SEX's "Male"/"Female"), not
TableShift's factorised-and-relabelled codes, which had visible artefacts
(RAC1P collapsed to 2 values, SEX's second category rendered as the string
"0" -- see data/tableshift_raw_cache/acsincome/codebook.json).

Usage:
    python scripts/extract_whyshift_cache.py                  # both tasks
    python scripts/extract_whyshift_cache.py whyshift_income   # one task

Output, per task, under data/whyshift_raw_cache/{task_name}/:
    train.parquet              -- source state, 80% (random split, seeded)
    test_id.parquet            -- source state, remaining 20%
    test_ood_{STATE}.parquet   -- one file per candidate target state
    codebook.json              -- {column: {name_extended, description, kind, values}}

Raw per-state Census CSVs are cached separately by folktables itself under
data/folktables_raw/ (~a few hundred MB across all states used here); safe
to re-run this script, nothing is re-downloaded once cached.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage, generate_categories
from folktables.acs import adult_filter, public_coverage_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLKTABLES_RAW_DIR = PROJECT_ROOT / "data" / "folktables_raw"
OUT_ROOT = PROJECT_ROOT / "data" / "whyshift_raw_cache"

ACS_YEAR = "2018"
ACS_HORIZON = "1-Year"
TRAIN_FRACTION = 0.8

TASKS: dict[str, dict] = {
    "whyshift_income": {
        "problem": ACSIncome,
        "preprocess": adult_filter,
        "source_state": "CA",
        "target_states": ["MS", "WV", "AR", "AL", "WA", "NY", "CO", "FL"],
        "drop_features": [],
    },
    "whyshift_pubcov": {
        "problem": ACSPublicCoverage,
        "preprocess": public_coverage_filter,
        "source_state": "TX",
        "target_states": ["MA", "VT", "MN", "CT", "FL", "AZ", "GA", "NV"],
        "drop_features": ["ST"],
    },
}


def _load_state(
    data_source: ACSDataSource, problem, preprocess, state: str, feature_cols: list[str]
) -> pd.DataFrame:
    """Fetch one state's ACS PUMS person data and apply the task's own filter
    + target transform. Returns raw feature values (small integer codes for
    categorical columns, natural units for numeric ones, NaN preserved for
    missing) plus an int `label` column -- no z-scoring, no one-hot
    expansion, no NaN-to-sentinel substitution (folktables'
    `BasicProblem.df_to_pandas` does `nan_to_num(x, -1)`; we bypass it here
    so missing values stay NaN for src/data/tableshift_loader.impute_missing
    to handle downstream, exactly as the TableShift cache already does).

    `preprocess` (e.g. `folktables.acs.adult_filter`) is passed explicitly
    rather than via `problem.preprocess` -- `BasicProblem` only stores it as
    the private `_preprocess`, with no public accessor.
    """
    raw = data_source.get_data(states=[state], download=True)
    filtered = preprocess(raw)
    label = problem.target_transform(filtered[problem.target]).astype(int)

    out = filtered[feature_cols].copy()
    out["label"] = label.to_numpy()
    return out.reset_index(drop=True)


def _build_codebook(data_source: ACSDataSource, feature_cols: list[str]) -> dict:
    """Real Census PUMS value mappings for the given features, via
    folktables' own data dictionary + generate_categories() -- see module
    docstring for why this is preferred over hand-maintained mappings.
    """
    definitions = data_source.get_definitions(download=True)
    categories = generate_categories(features=feature_cols, definition_df=definitions)

    codebook: dict = {}
    for feat in feature_cols:
        if feat in categories:
            values = {
                int(code): str(label)
                for code, label in categories[feat].items()
                if not (isinstance(code, float) and np.isnan(code))
            }
            codebook[feat] = {
                "name_extended": feat,
                "description": feat,
                "kind": "categorical",
                "values": values,
            }
        else:
            codebook[feat] = {
                "name_extended": feat,
                "description": feat,
                "kind": "numeric",
                "values": None,
            }
    return codebook


def extract_task(task_name: str, cfg: dict, seed: int = 42) -> None:
    problem = cfg["problem"]
    preprocess = cfg["preprocess"]
    source_state = cfg["source_state"]
    target_states = cfg["target_states"]
    drop_features = set(cfg.get("drop_features", []))
    feature_cols = [f for f in problem.features if f not in drop_features]

    data_source = ACSDataSource(
        survey_year=ACS_YEAR, horizon=ACS_HORIZON, survey="person", root_dir=str(FOLKTABLES_RAW_DIR)
    )

    print(f"[{task_name}] fetching source state {source_state}...")
    source_df = _load_state(data_source, problem, preprocess, source_state, feature_cols)

    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(len(source_df))
    split_point = int(len(source_df) * TRAIN_FRACTION)
    train_df = source_df.iloc[shuffled_idx[:split_point]].reset_index(drop=True)
    test_id_df = source_df.iloc[shuffled_idx[split_point:]].reset_index(drop=True)

    out_dir = OUT_ROOT / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    test_id_df.to_parquet(out_dir / "test_id.parquet", index=False)
    print(f"[{task_name}] {source_state}: train={len(train_df)} test_id={len(test_id_df)}")

    for target_state in target_states:
        print(f"[{task_name}] fetching target state {target_state}...")
        target_df = _load_state(data_source, problem, preprocess, target_state, feature_cols)
        target_df.to_parquet(out_dir / f"test_ood_{target_state}.parquet", index=False)
        print(f"[{task_name}] {target_state}: test_ood={len(target_df)}")

    codebook = _build_codebook(data_source, feature_cols)
    with open(out_dir / "codebook.json", "w") as f:
        json.dump(codebook, f, indent=2)
    print(f"[{task_name}] wrote codebook.json ({len(codebook)} features) to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "tasks", nargs="*", default=list(TASKS.keys()), help="Subset of tasks to extract (default: all)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the source state's train/test_id split")
    args = parser.parse_args()

    for task_name in args.tasks:
        if task_name not in TASKS:
            raise SystemExit(f"Unknown task {task_name!r}; choices are {list(TASKS)}")
        extract_task(task_name, TASKS[task_name], seed=args.seed)


if __name__ == "__main__":
    main()
