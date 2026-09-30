# 06 — P1: bugs the tests caught

**When:** 28 September 2026 (P1, gate G1 passed the same day) · **Status:** resolved

## What happened, and what we did

1. **The feature-coverage strategy ported from `main` only ever used one feature.**
   - It assigned rows to strata feature by feature, so only the top feature ever formed strata.
   - Fix: joint quantile cells over the top 3 features.
2. **A ported generator test failed about one run in six.**
   - It sampled tasks from the unseeded global random state, so it sometimes failed on sampling noise.
   - Fix: a seeded fixture.
3. **The prompt snapshot was built from the wrong task.**
   - The snapshot test built "evaluation task 0" with `sample_eval_tasks(1)`.
   - Because a task's family depends on the suite size (the first half of a suite is linear), a one-task suite made task 0 a tree task. The real suite's task 0 is linear.
   - Fix: build the snapshot from the configured suite, and regenerate the snapshots.
4. **Similarity is expensive.** It builds one prompt per query, so it costs about 5× as much per unit (about 22 s against about 4 s in the smoke run). P4 must budget for it.

The kill-and-resume check passed: a run stopped after 6 of 26 units was rerun and saved each unit exactly once.

## Why it matters

The first bug would have made one strategy a mislabelled copy of another. The third would have frozen and quoted a prompt that no evaluation task actually produces.

## Evidence

- `tests/test_synth_pipeline.py`, `tests/test_hf_runner.py`, `tests/snapshots/`.
- Spec, Known issues rows 15–16.

## How to tell it

A sentence in the methods chapter is enough:

> The implementation phase ended with a test suite covering every design invariant (78 tests), which caught, among others, a ported selector that used only one of its three features.
