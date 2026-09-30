# 13 — The covariate shift mostly moves features the rule ignores

**When:** 30 September 2026, found while walking through spec §11.2 · **Status:** resolved: option 1 was implemented the same day (the user's decision)

## What happened

The covariate environment adds a fixed shift vector to 3 features drawn at random from features 0–7:
- δ_j ~ U(1, 2) × Rademacher, a 1–2 standard deviation offset up or down;
- the offset is added to every covariate row, and the rule is unchanged.

To keep covariate shift separate from label shift, a candidate shift is accepted only if it moves the label-1 rate by at most 0.03. The rate is estimated on a 2,048-row probe, with up to 50 draws. This rejection step was added in v2, because v1's covariate shift had moved the label rate to 0.388.

## How we found it

Explaining why the shift is "added to 3 features" led to checking which features they are. In the evaluation suite, **21 of 24 tasks shift only distractors**. The other 3 shift two load-bearing features whose effects on the label cancel. Example: `eval_0012` shifts f8, f3 and f1 (all distractors), while its rule features f4, f7 and f6 keep their ID distribution.

The rejection step causes this. 300 random proposals per evaluation task:

| Load-bearing features among the 3 shifted | Share of proposals | Median change in label-1 rate | Accepted (≤ 0.03) |
| --- | --- | --- | --- |
| 0 | 18% | 0.000 | 100% |
| 1 | 53% | 0.233 | 0% |
| 2 | 27% | 0.351 | 22% |
| 3 | 2% | 0.390 | 1% |

A shift that avoids the rule's features cannot change any label, so it always passes. A shift of exactly one load-bearing feature never passes.

## Why it matters

- **What most tasks test.** A model that relies on the right features should be unaffected by a distractor-only shift. For most tasks, the covariate environment therefore tests sensitivity to irrelevant features, not the rule-relevant covariate shift of the ICL literature (e.g. Zhang, Frei & Bartlett 2024), in which queries move to regions the demonstrations cover sparsely.
- **RQ2.** Strategies meant to help with shifted queries, such as feature coverage, have nothing to fix. The 3 remaining tasks mean something different, so pooling the 24 mixes two kinds of shift.

## Options

1. **(Recommended) Shift two load-bearing features in opposite label directions, plus one distractor.** Push one towards label 1 and the other towards label 0, by the same amount in label terms.
   - Tree: the label-1 rate is ½(p₁ + p₂), where p₁ and p₂ are the vote-1 probabilities of the two shifted features. Equal and opposite pushes give p₁ + p₂ = 1, so the rate is exactly 0.5.
   - Linear: c_iδ_i = −c_jδ_j keeps c·x unchanged, so the rate is unchanged exactly.
   - No rejection loop is needed, and every covariate query lands in a new region of the rule's input space.
   - Only the covariate rows change. P2 used ID queries only, so its results stand.
2. **Keep the current design.** Describe the environment as a distractor shift for most tasks, and add "load-bearing features shifted" as a covariate.
3. **Drop the label-rate constraint.** Not recommended: it mixes covariate and label shift, the v1 problem.

## Evidence

- `task_manifest.json`, `covariate_shift` (features, δ, probe rates, draws).
- The `covariate` branch of `generate_environment` in `src/data/generator.py`.
- Spec §11.2.

## How to tell it

> A filter meant to keep one shift type pure can quietly select a different shift. Requiring the covariate shift to preserve the class balance made the generator accept, almost exclusively, shifts of features the labelling rule ignores. Checking which features were actually shifted revealed this before the main grid.

## Decision and result (30 September)

The user chose option 1.
- **Code.** `_label_neutral_shift` in `src/data/generator.py` moves two load-bearing features, one towards each label, by pushes of +m and −m on the logit (tree: equal m ~ U(1, 2)), and one distractor by U(1, 2) × Rademacher.
  - For linear tasks, m ~ U(max|c|, 2 min|c|), so both |δ| lie in [1, 2].
  - The v2 rejection sampler is kept only for the families that are not evaluated.
- **Manifest.** It records each shifted feature's role and the push. A probe check (the same 20,000 rows before and after) gives exactly 0 for every linear task and at most 0.009 for tree tasks.
- **Tests.** `test_covariate_shift_moves_two_rule_features` and `test_covariate_shift_keeps_the_label_rate`: 82 tests pass.
- **What changed in the data.** Only the covariate rows changed. The pool, ID and spurious-reversal rows and all their query ids are identical (checked file by file), so the P2 results stand.
- **Example.** `eval_0012` now shifts f7 +1.75 (towards 1), f6 −1.75 (towards 0) and the distractor f0 +1.86.
- **Documentation.** Spec §11.2 was rewritten (new lemma, and a "Why not rejection sampling" paragraph), plus Known issues row 20 and a bug-hunting note. NB01 was re-run, and the smoke runs were redone.
