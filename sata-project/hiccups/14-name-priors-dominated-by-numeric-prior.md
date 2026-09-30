# 14 — Qwen's name priors are mostly "bigger value means 1"

**When:** 30 September 2026, P3 · **Status:** decided the same day by the user: keep the naming factor, with counter_prior using measured priors

## What happened

P3 checks that Qwen2.5-7B-Instruct holds real-world beliefs about the feature names before the named conditions are run.
- **Method.** Qwen answered zero-shot on 2,000 random profiles: values N(0, 1) in z-units, with lexicon names in random columns. A ridge regression of its margin on the named values gave each name a slope (the prior surrogate).
- **Pre-registered screening rules.** A pair (p, q) is admitted when b_p > 0 > b_q, both intervals exclude 0, and the weaker slope is at least half the stronger. G3 then needs an out-of-fold R² ≥ 0.5 and a zero-shot AUROC(aligned) − AUROC(flipped) ≥ 0.15.

The screening admitted 1 of 16 pairs, and the surrogate failed the R² criterion.

## How we found it

**1. A numeric prior dominates.**
- **Abstract names.** Every column f0–f9 has a positive slope (0.22 to 0.89): a bigger value pushes towards label 1 whatever the column is called.
- **Named columns.** The same offset appears. The mean weak-name slope is +0.29 in loan and +0.48 in medical.

**2. Names that should lower P(label 1) mostly just push less.**
Slopes on the 800-profile screening run:

| Domain | Pair (raises / lowers P(1)) | b_p | b_q | Contrast b_p − b_q, 95% interval |
| --- | --- | --- | --- | --- |
| loan | net worth / collection accounts | +2.17 | −0.30 | 2.47 [1.50, 3.43] |
| loan | credit score / missed payments | +1.50 | −0.79 | 2.29 [1.51, 3.11] |
| loan | annual income / debt-to-income ratio | +0.87 | +0.67 | 0.20 [−0.65, 1.12] |
| medical | systolic blood pressure / weekly exercise hours | +1.57 | −0.05 | 1.62 [0.92, 2.32] |
| medical | LDL / HDL cholesterol | +1.00 | +0.35 | 0.65 [−0.04, 1.41] |
| medical | fasting blood glucose / daily step count | +0.67 | +0.74 | −0.07 [−0.66, 0.48] |

- **Medical.** Protective names barely register. Daily step count and lung function score push towards high risk.
- **Loan.** Only "missed payments" reverses clearly.

**3. The surrogate cannot fit the named margins.**
- **Out-of-fold R².** 0.51 abstract, 0.20 loan, 0.28 medical.
- **Richer models do not rescue it.** Gradient boosting over names, values and positions reaches only 0.29 (loan) and 0.44 (medical). With names, Qwen's zero-shot answer depends on the row in ways no simple summary captures.

**4. The names still shift the prior on real tasks.**
The check used the 12 pilot tasks, ID queries, and the 5 pairs per domain with the largest contrast.

| Zero-shot AUROC | Abstract | Aligned | Flipped | Aligned − flipped |
| --- | --- | --- | --- | --- |
| All pilot tasks | 0.499 | 0.594 | 0.430 | 0.164 [0.053, 0.287], 10 of 12 tasks |
| Loan | 0.508 | 0.688 | 0.410 | 0.278 |
| Medical | 0.490 | 0.500 | 0.450 | 0.050 |

## Why it matters

- **The naming factor carries three results:** the concept-shift arm of RQ1, the counter_prior contrast of RQ2 and the naming contrasts of RQ3, and later SATA-MI.
- **The design assumed symmetric priors (b_p ≈ −b_q).** It also assumed that flipped names reverse the prior for every load-bearing feature. For Qwen they do not.
  - The conflict comes mostly from positive names placed on negative-direction features ("credit score" on a column where higher means denied).
  - It is strong in loan and weak in medical.
- **counter_prior needs the model's prior for every pool row.** A surrogate with R² 0.2 would get many rows wrong.

## What we did

The user chose to keep the naming factor (option 1 of 3). The others were to screen new medical names first, or to drop naming, which was the plan's pre-registered fallback.
- **Final lexicon.** In each domain, the 5 pairs with the largest measured contrast and the 7 weak names with the smallest |b| (`configs/lexicons.yaml`, `final`). These are exactly the names used in the pilot check.
- **Measured priors instead of the surrogate.** counter_prior and the prior–data conflict C_t now use Qwen's own zero-shot answers on every pool row, with the margin centred on the pool median (`PoolPrior`, `scripts/p3_pool_priors.py`).
  - This is exact for the rows it is used on, so the R² criterion no longer applies.
  - The surrogate is kept as a descriptive table of name priors.
- **G3 as recorded.** R² criterion not met; separation passes (0.164, CI above 0). The C_t check follows once the pool priors are measured. Recorded in `results/v3/synthetic/p3/naming_gate.json`.
- **Metric fixed before the check ran.** Zero-shot predictions are uncalibrated, so the G3 separation uses AUROC rather than raw accuracy, which would mix in the label bias. Raw accuracy moved by 0.113.

## Result

- The named grid runs with this lexicon in P4.
- Medical is reported as a weak-prior domain, and domain × naming becomes a descriptive result.

## Evidence

- `results/v3/synthetic/p3/screen_qwen2.5-7b-instruct.json` (fits, slopes, intervals, screening verdicts).
- `results/v3/synthetic/p3/profiles_screen_qwen2.5-7b-instruct.parquet` (the 2,000 profiles and margins).
- `results/v3/synthetic/p3/zero_shot_pilot_provisional.parquet` and `naming_gate.json`.
- `configs/lexicons.yaml` (candidates, final names, the original screening rules).

## How to tell it

> Before asking whether demonstrations can override a model's priors, we measured the priors. Qwen2.5-7B-Instruct's zero-shot answers were driven mainly by magnitude: under any name, a larger value made label 1 more likely. Names adjusted this only asymmetrically. Positive names such as "credit score" raised the slope strongly, while names such as "debt-to-income ratio", or "daily step count" for medical risk, did not reverse it. A linear summary of the prior explained only a quarter of the named answers. Renaming the three rule features from aligned to flipped still moved the model's zero-shot AUROC from 0.69 to 0.41 in the loan domain (0.50 to 0.45 in medical), so the prior conflict is real but domain-dependent. We therefore measured the prior directly, from the model's own answer on each candidate demonstration, instead of trusting a surrogate.

- **Figure idea:** slope of each name, pair members side by side, against the abstract baseline (the numeric prior), by domain.
- **Examiner question:** "If your names do not reverse the prior, is 'flipped' really a concept shift?"
  - For loan tasks it clearly is: flipped names drive zero-shot AUROC below 0.5.
  - For medical tasks the conflict is weak, and we report it that way.
  - The mechanism is asymmetric: conflict arises mostly where a positive-sounding name sits on a feature that lowers the label.
