# 19 — feature_range always spreads its demonstrations over the shortcut feature

**When:** 5 October 2026, spec review; raised by the user ("should the demos we select not be shortcut or noise feats … ideally") · **Status:** open, for the design review: keep as pre-registered, or add a rule-feature coverage variant. The demo-mask experiment (hiccups/20) already runs a feature_range that cannot see the shortcut

## What happened

feature_range is the coverage strategy aimed at covariate shift.
- It takes the 3 pool features with the highest mutual information with the label and cuts each into low, middle and high bins (27 cells).
- Within each label, it takes one row per cell.
- The spec said these 3 features "often" include the shortcut feature.

## How we found it

**1. The shortcut is among the 3 in every task.**
- In all 24 evaluation tasks, the 3 features include the shortcut. The choice is the same across seeds and namings.
- The other two are rule features in 23 tasks. The noise feature is never chosen.
- The reason: with s between 0.80 and 0.90, the shortcut alone scores AUROC Φ(√2·d) ≈ 0.93 (spec §12). That is more predictive than any single rule feature, so mutual information ranks it near the top. The pool cannot tell a spurious feature from a causal one.

**2. Side effect: covering the shortcut's bins weakens it in the prompt.**
- Within each label, the cells where the shortcut disagrees with the label are rare in the pool. Taking one row per cell over-samples them.
- Shortcut agreement in the demonstration sets, abstract names:

| Strategy | Share of demonstrations where the shortcut agrees with the label |
| --- | --- |
| rule_diversity | 0.866 |
| random | 0.844 |
| label_diversity | 0.837 |
| **feature_range** | **0.661** |
| counter_spurious | 0.500 |

**3. feature_range is the weakest strategy.** AUROC, abstract names:

| Strategy | ID | Covariate | Spurious reversal |
| --- | --- | --- | --- |
| label_diversity | 0.593 | 0.652 | 0.520 |
| feature_range | 0.552 | 0.573 | 0.495 |
| counter_spurious | 0.561 | 0.575 | 0.555 |

C2c (feature_range − label_diversity under covariate shift) is −0.078 [−0.141, −0.018].

## Why it matters

- **What C2c was meant to test:** covering the feature space helps under covariate shift.
- **What feature_range actually does:**
  - it covers the three most label-related features, always including the shortcut;
  - as a side effect, it shows the shortcut agreeing in only two-thirds of its demonstrations.
- **What the negative C2c can't tell apart:** a failure of coverage, and the shortcut entering the coverage.
- **This is still a realistic result.** A practitioner using a pool-only coverage heuristic doesn't know which feature is spurious, so feature_range would fall into exactly this trap. It should be reported as that, not as a test of coverage of the rule.

## Options for the design review

1. **Keep feature_range as pre-registered.** Describe C2c as coverage over the most label-related features, which in every task include the shortcut.
2. **Add an exploratory variant that covers the 3 rule features.** Like rule_diversity, it would use the ground truth, so it would separate coverage from the shortcut. It needs one strategy rerun, at about the scale of stage C (minutes on an H200).

## Evidence

- `results/v3/synthetic/h200/p4/grid.parquet`: `selection_meta.features` for feature_range, and `demo_ids`.
- `data/synthetic/v3/eval/tasks/*.parquet`: pool `agree_obs`.
- Spec §12 (the shortcut-only AUROC) and the feature_coverage mechanism.

## How to tell it

> A coverage heuristic that spreads demonstrations across the most label-related features sounds neutral. In every one of our 24 tasks, however, the most label-related features included the planted shortcut: a shortcut that is right 80–90% of the time is more predictive than any single rule feature. The heuristic spent a third of its coverage on the shortcut. As a side effect, its demonstrations showed the shortcut agreeing with the label in only two-thirds of rows. Without knowing which feature is spurious, a selector cannot avoid this, because the pool does not contain that information.

- **Figure idea:** for each task, the 3 features feature_range covers, coloured by role (rule, shortcut, distractor, noise).
- **Examiner question:** "Is feature_range's poor result about coverage or about the shortcut?"
  - Under the pre-registered design the two cannot be separated. Option 2 would separate them.
