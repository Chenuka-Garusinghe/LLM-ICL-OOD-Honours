# 16 — Counter-prior demonstrations also made the shortcut look more reliable

**When:** 1 October 2026, during P4 on the Runpod H200 · **Status:** decided the same day by the user: keep counter_prior as pre-registered and add an f8-matched variant (exploratory)

## What happened

counter_prior shows demonstrations whose label contradicts the model's measured zero-shot answer and follows the rule (hiccups/14). The early H200 results (seed 0, abstract names) showed:
- it beat label_diversity on ID queries: AUROC 0.646 against 0.599;
- but it lost under spurious reversal: 0.511 against 0.543.

That is the signature of a strategy that strengthens the shortcut.

## How we found it

The share of demonstrations whose spurious feature (f8, read in the task's direction σ_t) agrees with the label:

| | label_diversity | counter_prior, all tasks | counter_prior, active tasks only |
| --- | --- | --- | --- |
| abstract, σ_t = −1 | 0.84 | 0.89 | 0.91 |
| aligned, σ_t = −1 | 0.84 | 0.89 | 0.92 |
| aligned, σ_t = +1 | 0.84 | 0.80 | 0.73 |
| flipped, σ_t = −1 | 0.82 | 0.87 | 0.86 |
| flipped, σ_t = +1 | 0.84 | 0.84 | 0.84 |

**The mechanism.**
- Qwen's zero-shot answer mostly follows "bigger values mean 1", including for f8.
- Where σ_t = −1, a large f8 goes with label 0. So the rows on which that numeric prior is wrong are mostly rows where the shortcut is right.
- Selecting prior-contradicting rows therefore over-selects shortcut-agreeing rows, which makes the shortcut look more reliable in the prompt.

**Also seen.** counter_prior was active in fewer tasks than designed: 15 of 24 under flipped names and 8 of 24 under aligned names. The mean prior–data conflict was 0.44 aligned against 0.54 flipped. The measured prior tracks the names only weakly (hiccups/14).

## Why it matters

- **RQ2's counter_prior contrast (C2d) should measure evidence against the prior,** not a change in how reliable the shortcut looks.
- **Under flipped names, C2d's own condition, the confound is small** (0.84–0.86 against 0.82–0.84). Under abstract names it is large and pushes ID up and spurious reversal down.

## What we did

The user chose option 1 of 3. The others were to keep counter_prior and only report the confound, or to replace it as the confirmatory strategy.
- **counter_prior stays as pre-registered** (C2d unchanged).
- **New exploratory strategy, `counter_prior_matched`** (`select_counter_prior_matched` in `src/selection/protocols.py`).
  - It takes label_diversity's set for the same seed as a reference, and keeps its number of rows in each (label, f8 agrees) cell. So label counts and the shortcut's apparent reliability match label_diversity's exactly, set by set.
  - Within each cell it prefers clean prior-contradicting rows, then other clean rows, then label-noise rows.
  - Where counter_prior is inactive, it is label_diversity's set.
- **It ran on the H200 the same day** for Qwen-Instruct (3 namings, 216 units) and Qwen-base (abstract and flipped), about 5 minutes. A test checks that its (label, agreement) counts equal label_diversity's.

## Evidence

- `results/v3/synthetic/h200/p4/grid*.parquet` (`selection_meta` holds `counter_prior_active` and the conflict).
- `grid_cpm.parquet` and `grid_base_cpm.parquet` (the matched variant).
- The composition table: notebook 05, section "What the demonstration sets contain".

## How to tell it

> Choosing demonstrations that contradict the model's prior sounds like a clean intervention, but the prior is not only about the feature names. Qwen's zero-shot answers rise with every value, including the spurious feature's. In tasks where that feature's direction is negative, the rows that contradict the prior are disproportionately rows that obey the shortcut, so counter-prior prompts quietly advertised the shortcut. We therefore added a matched version that holds the shortcut's reliability at the baseline level, separating evidence against the prior from evidence for the shortcut.

- **Figure idea:** f8 agreement in the demonstration sets by strategy and σ_t, next to the ID-minus-reversal gap.
- **Examiner question:** "Is counter_prior's effect real or a shortcut artefact?"
  - Compare `counter_prior` with `counter_prior_matched`. The matched version differs from label_diversity only in which prior-contradicting rows fill the same (label, shortcut) slots.
