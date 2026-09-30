# 12 — The v1 generator gate shaped the rule families

**When:** 6 September 2026, v1, notebook `04_task_generator` (commit `91338ba` on branch `main`) · **Status:** resolved. The rules carried into v3, where they were proved and gained new purposes.

## What happened

Before any LLM was run, the generator had to pass a validation gate. For 50 random tasks, a simple classifier was trained on 64 demonstrations, and each task had to pass three criteria:

1. **Learnable:** the ERM classifier (logistic regression on all 10 features) scores above 80% in distribution.
2. **The shortcut is taken:** the same classifier falls below 60% under spurious reversal.
3. **The mechanism really changes:** an oracle (XGBoost on the causal features only) loses at least 15 points under the `mechanism` shift.

At least 80% of tasks had to pass. The first run passed 12% (6 of 50).

## How we found it

The failing tasks were diagnosed family by family, and then by the Boolean function each tree task used.

- **Update 1 (12% → 34%).** Two problems were fixed:
  - The tree rule was a flat 2-feature AND, so easy that no learner needed the shortcut. It became a depth-3 lookup over 3 features (8 leaves).
  - Opposite leaves (L and 7 − L) were forced to opposite labels. The `mechanism` shift flips every causal sign, which moves each row to its opposite leaf. With random tables, opposite leaves often shared a label, so the shift changed little and criterion 3 failed.
  - Separately, the validator changed from XGBoost to logistic regression, the model class that the shortcut-learning theory (Nagarajan et al. 2021) is written for.
- **Update 2 (34% → 90%, 45 of 50).** The remaining tree failures fell into two groups:
  - **Dictators** (the label is one feature's sign). These are linearly separable, so the classifier never needs the shortcut: criterion 2 passed about 43% of the time.
  - **Parity** (XOR of the three sign bits). It cannot be learned from 64 examples even by the oracle (in-distribution accuracy about 0.57), so criterion 3 passed about 50% of the time.
  - The remaining "signed majority" tables passed both criteria about 100% of the time, so the sampler rejected dictators and parity.
  - The same update replaced thresholded linear labels (sigmoid(logit) > 0.5, which is just sign(logit), so the coefficient scale had no effect) with Bernoulli labels. This made the causal signal noisy enough for the shortcut to compete, the Colored MNIST logic.
- **Final gate result:** linear 81%, tree 93%, threshold 95%.

## Why it matters

These rules define what a tree task is. Without them, tree tasks would either never tempt a learner into the shortcut (dictators) or be unlearnable (parity), and the `mechanism` shift would often do nothing.

## What we did in v3

- **A theorem.** The v1 notebook had already noted that the surviving tables are exactly the 8 signed majorities, making tree structurally close to the old threshold family. The v3 spec (§10.2) proves it: 16 complement-antisymmetric tables, minus 6 dictators and 2 parities, leaves 8.
- **No rejection sampling.** v3 builds the table directly from three random direction signs (`signed_majority_leaf_labels`).
- **New purposes for the same rules.**
  - Every load-bearing feature now has a direction, which the aligned and flipped namings need. Parity has none, and dictators make two features decoys.
  - Complement antisymmetry balances the classes.
- **Mechanism no longer evaluated.** v3 treats `mechanism` as a placebo for in-context learning, so it appears only in SATA training.
- **Threshold family dropped from evaluation.** Its directions were all positive (Known issues row 9).

## Evidence

- `notebooks/04_task_generator.ipynb` on `main`, commit `91338ba`: the "Validation gate" section and its Updates 1–2.
- The docstrings of `_sample_tree_leaf_labels` and `_is_degenerate_leaf_labels` in `src/data/generator.py`.
- Spec §10.2 (theorem and corollary) and Appendix (the 8 tables).

## How to tell it

> The tree family's constraints were not chosen a priori. An early validation gate required that a simple learner could learn each task, would take the planted shortcut, and would be hurt by a genuine change of rule. Random decision trees failed: trees that depend on a single feature never tempted the learner into the shortcut, and parity trees could not be learned at all. Excluding them, and pairing every leaf with its sign-flipped opposite, left exactly the eight majority-vote rules, which we later proved is the whole admissible set.

- **Figure idea:** the 16 complement-antisymmetric tables, coloured by which gate criterion each group failed.
- **Examiner question:** "Why majority votes and not general trees?"
  General trees include rules with no feature directions (parity) or with decoy features (dictators). The first cannot be named, and the second never tests shortcut use.
