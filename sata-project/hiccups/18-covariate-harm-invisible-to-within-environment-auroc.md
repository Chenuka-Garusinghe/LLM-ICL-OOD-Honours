# 18 — Covariate shift hurts the threshold, not the ranking our test measures

**When:** 1 October 2026, P4 analysis; raised by the user ("theoretically a covariate shift should hurt") · **Status:** resolved 1 October: all three options done (the user's decision); the covariate design question goes to the design review

## What happened

The pre-registered covariate contrast C1a (ID minus covariate, random demonstrations, abstract names) came out at AUROC +0.024 [−0.044, 0.093], Holm p 0.95. Taken at face value, covariate shift does not hurt Qwen. The user asked whether shortcut learning could explain this.

## How we found it

**1. Shortcut protection (partly true).**
- The covariate environment leaves f8 untouched. f8 is regenerated from the label in every environment.
- Scoring queries by the shortcut alone gives AUROC 0.927 on ID and 0.901 on covariate queries, but 0.108 under spurious reversal (linear tasks; tree is similar).
- Under abstract names Qwen relies mainly on f8 (RQ3): replacing f8 costs 0.141 of correct-label margin, the three rule features together 0.009. A shortcut-reliant model has little to lose when only rule features move.

**2. The design and the metric (the larger reason).**
- Every covariate query gets the same shift vector, and the rule is unchanged. Within one environment, a uniform shift cannot reorder the queries of any scorer that is roughly linear in the features. The true rule's AUROC is 0.884 on ID and 0.891 on covariate queries.
- Qwen's scores do move. On covariate queries the mean margin rises by 0.5–1.0 logits, and the share predicted "1" by 3–9 points.
- AUROC within an environment ignores a uniform rise. Ranking ID and covariate queries together exposes it (label_diversity, abstract names):

| Comparison | AUROC |
| --- | --- |
| ID positives against ID negatives | 0.593 |
| Covariate positives against covariate negatives | 0.652 |
| Covariate positives against ID negatives | 0.688 |
| ID positives against covariate negatives | **0.512** |

Shifted negatives now look like positives. Under a fixed decision threshold, this is the harm.

**3. Two checks that point to the metric rather than the shortcut.**
- Counter-spurious demonstrations, which make f8 uninformative, show no within-environment covariate penalty either (−0.02 to −0.01).
- Across tasks, how much Qwen relies on the rule features against f8 does not predict its covariate gap (Spearman 0.06, p 0.79).

## Why it matters

- **What C1a can claim.** Its null means "the ranking survives a uniform shift of 1–2 SD", not "covariate shift is harmless".
- **Where the harm actually sits.** It is a shift in the score's threshold (calibration), which the confirmatory AUROC does not test. The calibrated accuracy reported alongside moved only a little (0.00–0.03), because contextual calibration with a content-free query does not track the shift.
- **What the covariate design does not test.** It does not move the shortcut, so it cannot show shortcut-driven covariate failures.

## Options for the design review

1. **Measure the threshold effect.** Report cross-environment AUROC (ID positives against covariate negatives) and the score shift, or accuracy at an ID-fitted threshold, alongside C1a. This needs no new runs.
2. **Make the covariate shift non-uniform.** For example, draw covariate queries from a different region or covariance of the rule features, so the shift can reorder queries and test extrapolation. This needs new data and a rerun of the covariate rows.
3. **Add a covariate variant that also moves the shortcut's distribution,** to test shortcut protection directly.

The pre-registered C1a result stands as reported, and the interpretation above goes with it.

## Evidence

- `results/v3/synthetic/h200/p4/grid.parquet` and the scorer checks on `data/synthetic/v3/eval/` (true-rule, shortcut-only and rule-feature AUROC per environment).
- RQ3 reliance: notebook 06, section 1.

## How to tell it

> Our covariate shift moved two rule features in opposite label directions by the same amount for every query, keeping the class balance and the rule fixed. Qwen's scores rose by up to a logit on those queries, but because every query rose together, its ranking within the shifted set, which AUROC measures, was unchanged. The damage appeared only when shifted and unshifted queries were compared: shifted negatives scored like positives. A pre-registered null therefore said something precise and narrow — rankings survive a uniform shift — rather than that covariate shift is harmless.

## Decision and results so far (1 October)

The user asked for all three options.

**Option 1, done** (`cross_environment_auroc` in `src/evaluation/analysis.py`; notebook 05, section 10).
- With demonstrations, covariate queries score **+0.44 to +1.21 logits** higher than ID queries for the same prompt, across every strategy and naming. Zero-shot shows no such shift (−0.23 to +0.06), so the shift comes from what Qwen takes from the demonstrations.
- Random demonstrations, abstract names: +0.49 logits [0.03, 0.97 over tasks]. ID positives against covariate negatives score 0.539 [0.478, 0.606].
- Under abstract names, the cross-environment AUROC falls to 0.46–0.54 across strategies, while within-environment AUROC is 0.55–0.64.

**Options 2 and 3, implemented.**
- **New splits, appended to every task's file:** `covariate_scale` and `id_f8neutral`, `covariate_f8neutral` and `covariate_scale_f8neutral`.
  - Every existing row is unchanged, checked on all 36 files.
  - `covariate_scale`: rule-feature SD 2.3 against 1.0, values up to about 6; label rate 0.5; shortcut agreement 0.84.
  - f8-neutral copies: shortcut agreement 0.50; every other value, the label and the query id are those of the source query.
- **Analysis and tests:** `exploratory_contrasts` (E1–E3c) in `analysis.py`; tests in `test_synth_pipeline.py` and `test_naming.py` (108 pass).
- **Pod stage:** `scripts/run_p4_pod.sh` stage D.

**Options 2 and 3, results.**
- **How they ran.** A second H200 Pod, on 1 October, because the first Pod's host had no free GPU. It scored the 4 new query sets on every existing demonstration set: 1,656 Qwen-Instruct units and 624 Qwen-base units, in 23 min.
  - The pool priors were copied from the first run, so all demonstration sets are identical (checked: 100% of units).
  - Results: `results/v3/synthetic/h200/p4/grid_probe.parquet` and `covariate_probes.csv`.
- **Statistics.** AUROC differences with hierarchical-bootstrap 95% intervals, over 24 tasks and 3 seeds. Qwen-Instruct, by naming:

| Probe | Abstract | Aligned | Flipped |
| --- | --- | --- | --- |
| **E1** uniform shift (ID − covariate) | −0.06 to +0.02, all intervals include 0 | +0.02 to +0.05, all include 0 | −0.03 to +0.02, all include 0 |
| **E2** variance shift (ID − covariate_scale) | −0.03 to +0.07, include 0 | −0.02 to +0.01, include 0 | **+0.08 to +0.13**; above 0 for random 0.091 [0.01, 0.17], label_diversity 0.104 [0.02, 0.18], rule_diversity 0.128 [0.05, 0.21] |
| **E3a** what the shortcut adds on ID (ID − id_f8neutral) | random 0.062 [0.02, 0.12], label_diversity 0.048 [0.02, 0.08], rule_diversity 0.081 [0.04, 0.13]; counter_spurious 0.013 | similar or smaller | similar or smaller |
| **E3b** shortcut protection, uniform shift | about 0 for every strategy (−0.03 to +0.01) | about 0 | about 0 |
| **E3c** shortcut protection, variance shift | slightly negative (−0.02 to −0.05; random −0.045 [−0.09, −0.00]) | slightly negative | slightly negative |

- **Qwen-base** looks the same: E3a for random is 0.084 [0.03, 0.14] (abstract), E3b is about 0, and E2 is small (0.04–0.05, intervals include 0).

**Answers.**
- **Does the shortcut hide covariate harm?** No. Qwen does use the shortcut: making it uninformative costs 0.05–0.08 ID AUROC, but nothing when counter-spurious demonstrations never taught it. Removing it does not open a covariate gap (E3b about 0).
  - The variance shift stretches every other feature but not f8, so it dilutes the shortcut's relative weight. That is why the gap is slightly smaller without it (E3c < 0).
- **Why C1a is null.** The uniform shift moves all covariate scores together (+0.44 to +1.21 logits) without reordering them, and within-environment AUROC cannot see that.
- **When covariate shift does hurt the ranking.** When it is non-uniform and the model reads the rule features wrongly: under flipped names, stretching them amplifies the wrong direction (−0.09 to −0.13 AUROC). With abstract or aligned names, the variance shift does no harm.
