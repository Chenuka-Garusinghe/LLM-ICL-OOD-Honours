# 01 — The first full run measured the instrument, not the model

**When:** the v1 run, analysed in `REDESIGN_RATIONALE.md` (branch `main`, added 14 September 2026) · **Status:** resolved by the v2 redesign

## What happened

The first complete run of the SATA experiment covered:
- four TableShift datasets and a synthetic arm;
- Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct;
- seven demonstration conditions.

Its headline results were:
- **RQ1:** little of the expected OOD degradation;
- **RQ2:** no protocol × shift interaction;
- **RQ3:** too noisy to support a claim;
- **RQ4:** SATA was the worst method, reaching 0.045 accuracy under spurious reversal.

## How we found it

Each result was traced to specific lines of code. There were five defects:

1. **No chat template, and an uninformative prompt.** The instruct models received raw strings. They predicted class 1 for 96–99% of queries, and every method comparison became a comparison of how well each demo set happened to counteract that bias.
2. **SATA learned the shortcut it was meant to fight.** Three things combined:
   - its training target leaked the query's label;
   - the spurious feature was a near-copy of the label;
   - validation used the ID environment only, where the shortcut looks excellent.

   Under spurious reversal, SATA inferred the wrong label and chose demos that carried it, which gave accuracy of about 1 − strength.
3. **Method comparisons were confounded.** Only one selector balanced the demo labels; the retrieval-based methods put their best demo first (farthest from the query); and the synthetic grid used a single seed.
4. **Generator defects:**
   - an impossible cell (sparse_interaction × missing_feature);
   - a covariate shift that also moved the label rate;
   - an extrapolation environment that standardisation erased;
   - demo pools drawn from the shifted distribution.
5. **SATA train/inference mismatches** in how inputs were standardised.

## Why it matters

None of the v1 numbers could answer the research questions: they measured the pipeline's defects.

## What we did

The v2 redesign has three stages, each with a gate that must pass before the next expensive run:
- **S1, the instrument:** chat templates, contextual calibration, seeded demo order and 5 seeds;
- **S2, the testbed:** a rejection-sampled covariate shift, a continuous spurious feature, and ID-pool provenance;
- **S3, SATA's supervision:** new targets and an oracle check of those targets before any training.

## Evidence

- `REDESIGN_RATIONALE.md` on `main` (full diagnosis with quoted code).
- v1 results under `results/` on `main`.

## How to tell it

> The first complete run produced clear-looking results that turned out to describe the pipeline rather than the models: a 96–99% class-1 bias from a missing chat template, and a selector that learned the very shortcut it was built to avoid. Tracing each result to its cause changed how the project worked: every later phase starts with a gate that checks the instrument before any expensive run.

- **Figure idea:** a v1 → v2 table (defect → symptom → fix → gate).
- **Examiner question:** "How do you know v3's results aren't artefacts too?"
  Point to the gates (G1, P2) and to hiccups 07–10, where the same habit caught problems before the main grid.
