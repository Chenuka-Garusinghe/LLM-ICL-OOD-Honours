# v2 redesign notebooks (09–14)

Notebooks 01–08 are the **v1** design. Everything under `results/` from that
design is void (`REDESIGN_RATIONALE.md`); nothing in 09–14 derives from it.

These six cover the v2 redesign end to end. They are numbered to continue the
existing sequence, use the same `sata-project` kernel, and follow the same
structure (config boilerplate cell, markdown rationale between every step).

| notebook | what it does | compute |
|---|---|---|
| `09_shift_estimation` | Fits the deployment-legal shift context (domain discriminator, gated BBSE prior, per-feature drift) and reports the shift taxonomy | CPU, seconds |
| `10_protocol_screen_cpu` | Defines the `mechanism × composition` factorial replacing v1's flat protocol list; screens the grid with order-invariant surrogate readers | CPU, seconds |
| `11_corruption_gate` | Gate S0c at 8B: does the model use the demonstration labels? Finds the demonstration-induced threshold collapse | loads results |
| `12_verbaliser_diagnostic` | Four verbalisers incl. a swapped pair; establishes token anchoring, and re-checks the gate on a calibrated metric | loads results |
| `13_scale_comparison` | Adds the ID split and a 70B arm. **No ID→OOD gap**; calibration is sign-inverted at 70B | loads results |
| `14_feature_channel` | The 0%-vs-100%-corruption contrast. **Query-conditional selection has a real feature-channel gain at 8B that inverts at 70B** | loads results |

## Running them

Run in order — 09 and 10 write nothing the others need (all six read from
`results/v2/`), but the rationale is cumulative and 14 refers back to 11–13.

```bash
cd sata-project
jupyter lab notebooks/
```

The result parquets are already in `results/v2/` (gitignored, so they are local
only). If that directory is empty, the collection commands are in the markdown at
the top of notebooks 11, 13 and 14 — each needs a GPU and takes 1–3 hours.

Notebooks 09 and 10 depend on `_screen_cache/*.pkl`, regenerable on CPU:

```bash
PYTHONPATH=. python scripts/prep_shift_context.py --cache data/tableshift_raw_cache --out _screen_cache
PYTHONPATH=. python scripts/screen_protocols.py --cache-dir _screen_cache --out results/v2
```

## Two schema gotchas

- The GPU runner records the query split in **`environment`** (`"id"` / `"ood"`).
  The CPU screen uses **`query_split`**. Code does not move between them
  unchanged.
- `prediction` is contextually calibrated; `prediction_raw` is not. **Use
  `prediction` at 8B and `prediction_raw` at 70B** — notebook 13 §4 shows the
  calibration is sign-inverted at 70B and collapses thresholded accuracy there.
  AUROC on `logprob_1 - logprob_0` is unaffected either way, which is why it is
  the primary statistic throughout.

## Statistical caveat that applies to all of them

The replicated unit is the **demonstration draw** (one seed = one pool draw, one
demo selection, one query sample), so seeds — not queries — are the unit of
replication for any claim about demonstration design. A Wilcoxon signed-rank test
cannot return *p* < 0.05 at 5 seeds at all (floor 0.0625); notebooks 11–13 use 5
seeds and are therefore descriptive. Notebook 14 uses 8 (floor 0.0078) and is the
only one making a confirmatory claim, restricted to a pre-registered 6-test
family. See `14_feature_channel` §3 for the arithmetic.
