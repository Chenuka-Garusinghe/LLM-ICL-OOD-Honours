"""P6 helpers run between the probe stages (docs/research_plan.md, P6 implementation).

  # steering vectors from stage A's random-demonstration residuals, on the training tasks
  python scripts/p6_probes.py vectors --probe-dir results/v3/synthetic/p6/A --out models/p6_steer.npz

  # the P6 gate: |m_L - m| <= 1e-3 on every probe row
  python scripts/p6_probes.py gate --probe-dir results/v3/synthetic/p6/A

Training tasks are the even-indexed evaluation tasks (eval_0000, eval_0002, ...); steering is
measured on the odd-indexed ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.evaluation.mechanisms import load_lens, load_residuals, steering_vectors, with_prior  # noqa: E402


def train_task(task_id: str) -> bool:
    return int(task_id.rsplit("_", 1)[1]) % 2 == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["vectors", "gate"])
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--out", default="models/p6_steer.npz")
    args = parser.parse_args()
    probe_dir = Path(args.probe_dir)
    df = load_lens(sorted(probe_dir.glob("lens.*.parquet")))

    if args.command == "gate":
        lens_last = np.stack(df["lens"].to_numpy())[:, -1]
        err = float(np.abs(lens_last - df["margin"].to_numpy()).max())
        report = {"rows": len(df), "max_abs_error": err, "passed": err <= 1e-3,
                  "fallbacks": int(df["fallback"].sum())}
        (probe_dir / "gate_p6.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report))
        return

    df = with_prior(df)
    train = df[(df.strategy == "random") & df.resid_file.notna() & df.task_id.map(train_task)].reset_index(drop=True)
    resid = load_residuals(train, probe_dir)
    out = steering_vectors(train, resid)
    np.savez(args.out, v=out["v"], rand=out["rand"])
    summary = {"rows": len(train), "tasks": sorted(train.task_id.unique()), "counts": out["counts"],
               "norms": [round(float(x), 3) for x in out["norms"]]}
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("rows", "counts")}))


if __name__ == "__main__":
    main()
