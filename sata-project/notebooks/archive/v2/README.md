# v2 notebooks (archived)

These are the synthetic-arm notebooks from the v2 design, kept for reference only.
They do not run against the current code: they need the removed MLX runner and the
old merged-pool datasets.

The v2 design had defects that invalidate its results. `docs/research_plan.md` (bug table)
and `docs/generator_spec.pdf` (Part "Known issues in the v2 implementation") list them.
No v2 results were saved.

| Archived notebook | Replaced by |
|---|---|
| `02_synthetic_baselines.ipynb` | `notebooks/03_grid_run.ipynb` (driver over `scripts/run_synth_grid.py`) |
| `03_synthetic_analysis.ipynb` | `notebooks/05_analysis.ipynb` (P4) |

`01_synthetic_data.ipynb` was updated in place for the per-task design; its v2 version is in git history.
