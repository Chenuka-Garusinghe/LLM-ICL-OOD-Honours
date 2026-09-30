# 04 — The v2 code audit: fourteen defects before a single v2 result

**When:** 26 September 2026, before the v3 plan was approved · **Status:** resolved in P1 (row 10, sample size, is handled in P4)

## What happened

An audit of the v2 code on branch `synthetic-experiments` found 14 defects before any v2 results were saved. The ones that most affect the story:

- **Cross-task pools.** NB01 merged 5 tasks with different rules into one 1,280-row pool, so demonstrations came from other rules, and `head(30)` evaluated task 0 only.
- **One fixed seed in every `select()` call.** Each strategy had one demo set for all queries.
- **Leakage.** The counter-spurious proxy searched using OOD test rows and labels.
- **Double BOS token on Llama** (`[128000, 128000, …]`).
- **A visible shortcut.** The spurious feature was always in column 9, and its standard deviation (about 1.44) was wider than every other column's (1.0).
- **Mismatched definitions.** The regime code used all 3–5 causal features while the tree rule used 3.
- **An asymmetric similarity strategy.** MiniLM embedded labelled pool rows but unlabelled queries.
- **An inefficient runner.** `MLXRunner` computed logits at every position and had no prefix cache.
- **Too small a sample.** n = 30 per cell gave an interval half-width of 0.179.

## Why it matters

Several of these would have produced confident but wrong results, for example cross-task demonstrations or leaked test labels.

## What we did

P1 rebuilt the pipeline around one dataset per task:
- per-task pools and a manifest;
- a column-role permutation;
- z-scoring with pool statistics;
- ground-truth tags;
- magnitude-matched counter-spurious sets;
- feature-space kNN;
- `HFRunner` with prefix caching.

Gate G1 turns each invariant into a test (74 tests at G1, 78 now).

## Evidence

- `docs/generator_spec.pdf`, Part "Known issues in the v2 implementation" (rows 1–16, with resolutions).
- `tests/test_synth_pipeline.py`.

## How to tell it

> Before running v2, we audited it and found fourteen defects, several of which would have produced plausible-looking but wrong numbers (demonstrations drawn from other tasks, test labels leaking into selection). The v3 design turns each of these into a tested invariant, so the gate for the implementation phase is a test suite rather than a judgement call.

- **Figure idea:** none needed. A compact table of defect → invariant → test is enough.
