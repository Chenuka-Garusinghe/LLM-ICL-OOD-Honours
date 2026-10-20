# 20 — Should demonstrations show the shortcut at all?

**When:** 5 October 2026, spec review · **Status:** decided by the user the same day; run on 11 October 2026 (stage E on a Runpod H200, 45 minutes; notebook 03.1, section 8)

## What happened

While working through demonstration selection, the user argued that demonstrations should not include the shortcut feature, or the noise feature. Demonstrations are examples meant to point the model to the real, causal signal, so showing a feature that only correlates with the label works against them.

This came straight after hiccup 19: feature_range spread its demonstrations over the shortcut in every task.

## The discussion

- **The case for keeping it (the P4 design).**
  - In real data, nobody knows which feature is spurious. A real demonstration is a whole past record, with its causal, irrelevant and spurious fields together.
  - Spurious reversal changes only the shortcut, so it can only test a model that sees the shortcut.
  - A selector told which feature is spurious is an oracle.
- **The user's case.**
  - An example offered to a model should show what matters.
  - The interesting question is how much of the shortcut problem disappears when the demonstrations never show the shortcut while the query still does, as a new case would.
- **Both are kept.** The P4 design is unchanged, and this is a separate exploratory experiment.

## What we did

The user specified the design:
- **Two parts:** demonstrations without the shortcut feature, and demonstrations without both the shortcut and the noise feature.
- **The query still shows both.**
- **Ground truth, not estimation:** take the hidden features from the task record made before the columns are shuffled.

Implementation (`src/data/demo_mask.py`, `GridRunner(demo_mask=...)`, stage E of `scripts/run_p4_pod.sh`):

**Three arms of the same grid.**
- `none`: full demonstrations, rerun as the baseline.
- `spurious`: Part 1.
- `spurious_noise`: Part 2.

**Ground truth.** The generator always puts the shortcut in its column 8 and the noise feature in column 9. Each task's manifest records the permutation that displays them. So the hidden features are looked up as f_{π_t(8)} and f_{π_t(9)} and checked against the stored roles.

**Selection never reads a hidden feature.**
- feature_range ranks, and similarity measures distance, on the visible features only.
- counter_prior uses Qwen's zero-shot prior re-measured on the masked pool rows.
- counter_spurious selects on the shortcut, so it is refused under a mask. It runs in the full arm as the "neutralise instead of remove" reference.

**Count-free system message in all three arms:** "Each example lists numeric measurements …". Masked demonstrations show 9 or 8 measurements and the query 10, so the old "10 measurements" would be false.

**Checks.**
- 14 new tests in `tests/test_demo_mask.py`; 122 pass in total.
- A one-task smoke run on Qwen (Mac) rendered all three arms through the chat template with label mass 1.000 and no prefix fallbacks.
- Masked priors and counter_prior ran end to end.

**Literature.**
- Colored MNIST's grayscale oracle (Arjovsky et al. 2019) is the classic model trained without the spurious feature. Here only the demonstrations lose it.
- Nastl & Hardt (NeurIPS 2024) found that causal-feature predictors do not generalise better on tabular shifts, a caution against expecting removal to help.

## Already visible before the run (notebook 03.1, section 3)

- **feature_range with the shortcut hidden.** It covers the shortcut in 24 of 24 tasks with full demonstrations, and in none when the shortcut is hidden.
  - Its chosen rows' shortcut agreement returns from 0.62 to 0.80 (seed 0, abstract), close to label_diversity's 0.81.
  - With only the shortcut hidden, it spreads over the noise feature in 2 tasks, because the mutual-information estimate is noisy. Part 2 removes that too.
- **Same rows elsewhere.** random, label_diversity and rule_diversity pick the same rows in every arm (216 of 216 sets).

## Why it matters

- **It separates two routes to shortcut use:**
  - learning it from the demonstrations;
  - reading it from the query through Qwen's "bigger → 1" prior (hiccup 09).
  The second shows up as an ID − reversal gap with opposite signs for the two spurious directions.
- **It gives a reference for selection:** how well Qwen does when the demonstrations carry no shortcut signal at all, against counter_spurious, which keeps the feature but makes it uninformative.

## Evidence

- `src/data/demo_mask.py`, `tests/test_demo_mask.py`, `demo_mask_contrasts` in `src/evaluation/analysis.py`.
- `notebooks/03.1_demos_without_spurious_feats.ipynb`.
- Spec section "Demonstrations without the shortcut", and the research plan's P4 exploratory list.
- Results (after the run): `results/v3/synthetic/h200/demo_mask/`.

## How to tell it

> The examples shown to a language model are usually whole records, so they carry whatever spurious correlations the data holds. We asked what happens if they don't. Three versions of every prompt were scored: full demonstrations; demonstrations with the planted shortcut removed; and demonstrations with the shortcut and a pure-noise feature removed. The query kept every feature, as a new case would. The features to hide were looked up from the generator's ground truth rather than estimated, and no selection strategy could read them.

- **Figure idea:** ID and spurious-reversal AUROC for full, masked and counter-spurious demonstrations, side by side, with the gap split by spurious direction.
- **Examiner question:** "Isn't hiding the shortcut cheating?"
  - Yes, deliberately. It is an oracle condition, like a grayscale Colored MNIST model, and it shows how much is at stake for a selector that has to find the shortcut on its own.


## Result (11 October 2026)

**Qwen learns the shortcut from its demonstrations, but removing it does not teach the rule, and the query still carries it.**
- **M1, no shortcut − full** (mean over 6 strategies × 3 namings):
  - ID −0.048 (interval below 0 in 10 of 18);
  - covariate −0.040 (5 of 18);
  - spurious reversal +0.049 (10 of 18).
- **The ID − reversal gap shrinks (M4)** in all 18 cells, significantly in 14, by 0.07–0.19. feature_range is the exception, since it already under-represented the shortcut (hiccups/19).
- **Removing the noise feature adds nothing (M3 ≈ 0).**
- **Removing and neutralising (counter_spurious) do equally well (M5):** no interval excludes 0.
- **Accuracy falls under abstract names:** label_diversity's ID AUROC drops from 0.589 to 0.528, because the shortcut was most of what Qwen used.
- **The gap left over follows the shortcut's direction:** +0.04 to +0.11 for σ = +1 and −0.06 to −0.15 for σ = −1. Qwen's "bigger → 1" prior still reads the shortcut in the query.

**How to tell it (addition).**
> Hiding the shortcut from the demonstrations closed part of the gap between in-distribution and reversed-shortcut queries, confirming that the model learns the shortcut from what it is shown. It did not make the model learn the rule, and it could not stop the model reading the shortcut in the query itself through its numeric prior. Neutralising the shortcut in the demonstrations, which needs no knowledge of which feature is spurious once a selector can find it, did as well as removing it.
