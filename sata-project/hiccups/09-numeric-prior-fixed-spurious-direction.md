# 09 — A hidden numeric prior lined up with the planted shortcut

**When:** 28 September 2026, the first P2 run (Qwen half) · **Status:** resolved: the spurious direction is randomised per task (the user's decision), and P2 was rerun

## What happened

With abstract feature names (f0–f9) and no demonstrations, Qwen2.5-7B-Instruct scored the pilot tasks' ID queries at AUROC 0.633. That is as high as with 8 gold demonstrations, and it should not happen with anonymous features.

## How we found it

Qwen's zero-shot score follows a simple rule of thumb: bigger feature values mean label 1.
- Its zero-shot margin had a Spearman correlation of 0.70 with the plain sum of the 10 features.
- 82% of the per-task least-squares slopes of the margin on the features were positive.

In every task, the spurious feature f8 pointed the same way: f8 = +(2y − 1)d + ε. The load-bearing directions were random.
So the prior backed the shortcut in every task:

| Scorer on the ID queries | AUROC |
| --- | --- |
| Sum of all 10 features | 0.628 |
| Sum without f8 | 0.511 |
| Qwen, zero-shot | 0.633 |

Qwen's zero-shot signal came from f8, through the prior.

## Why it matters

- **The P2 gate.** Gold labels added almost nothing over zero-shot (0.66 against 0.63). Much of the gold − shuffled gap was shuffled labels breaking the prior's f8 signal, rather than gold labels teaching the rule.
- **P4.** Under spurious reversal, every condition, zero-shot included, would have dropped because of this prior. That drop would have been confused with shortcut use learned from the demonstrations.

## What we did

Three options were offered: randomise the direction; keep it and add a covariate; or let P2 finish first. The user chose to randomise.

- **The change.** Each task has a spurious direction σ_t ∈ {+1, −1}, and f8 = σ_t((2y − 1)d + ε).
  - σ_t is +1 in half of each (family, domain) cell and −1 in the other half.
  - It is drawn from a separate random stream, so every other draw is unchanged.
- **Only f8's sign changes.** σ_t multiplies the whole column, so a −1 task differs from before only in the sign of f8. The other columns, the labels, |f8|, the agreement flags and the counter-spurious sets are identical. This is tested, and checked against the archived data file by file.
- **Archive.** The first run and its data were kept in `archive/fixed_f8/` under both `results/v3/synthetic/` and `data/synthetic/v3/`.
- **Documentation.** The spec, plan and NB01 were updated, and the tests grew to 78.

## Result (P2 rerun, same tasks)

Because only f8's sign changed, the two runs form a natural experiment.

| Qwen, zero-shot AUROC | first run (all σ = +1) | rerun |
| --- | --- | --- |
| tasks with σ_t = +1 (prior backs the shortcut) | 0.642 | 0.642 |
| tasks with σ_t = −1 (prior opposes it) | 0.625 | **0.357** |
| all 12 tasks | 0.633 | 0.499 |

- **The gate still passes for Qwen at k = 8:** Δ = 0.098, 95% CI [0.021, 0.188], with gold beating shuffled in 9 of 12 tasks. It also passes at k = 16 (Δ = 0.159) and fails at k = 32.
- **Gold labels beat shuffled labels in both halves.** Where the prior opposes the shortcut, gold demonstrations mostly cancel the prior: AUROC rises from 0.36 to about 0.53.
- **The results are reproducible.** The 6 unchanged tasks gave bit-identical scores in both runs (2,280 rows, p1 difference 0.0).

## Evidence

- Archived first run: `results/v3/synthetic/archive/fixed_f8/`, with a README.
- Rerun: `results/v3/synthetic/p2_learnability.parquet` and `p2/learnability_gate.json`.
- Spec §Spurious feature construction ("Why the direction is random") and Known issues row 18.
- Tests `test_spurious_direction_is_balanced` and `test_spurious_direction_reflects_only_f8`.

## How to tell it

This is one of the strongest stories in the thesis.

> Abstract feature names are not prior-free. Qwen's zero-shot predictions follow a simple numeric heuristic, "bigger values mean 1", which in our first design coincided with the planted shortcut in every task. Flipping only the sign of the spurious column in half of the tasks moved zero-shot AUROC on those tasks from 0.63 to 0.36 while leaving everything else identical. The final design balances the shortcut's direction, so the prior helps in half of the tasks and hurts in the other half, and the two halves can be compared.

- **Figure idea (a strong slide):** paired bars of zero-shot AUROC for the σ_t = −1 tasks, before and after the flip, next to the unchanged σ_t = +1 tasks.
- **Examiner question:** "Could other, unmeasured priors confound the results?"
  This one was found because the gate was checked against zero-shot. The design now balances the direction of every planted signal, and load-bearing directions are random. P3 measures priors explicitly.
