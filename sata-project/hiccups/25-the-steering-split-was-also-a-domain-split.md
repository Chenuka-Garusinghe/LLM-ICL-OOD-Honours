# 25 — The steering split was also a domain split

**When:** 22 October 2026, during the thesis read-through, on the Mac (no GPU) · **Status:** checked. The domain split does not explain the steering null. To be disclosed in chapter 6.

## What happened

P6 fitted the steering vector on the even-indexed evaluation tasks and tested it on the odd-indexed ones. The evaluation suite alternates domains, so the split was also a domain split. Every task the vector was built on is a loan task (`eval_0000`, `0002`, …), and every task it was tested on is a medical task. Rule family (6 linear, 6 tree) and shortcut direction (6 × +1, 6 × −1) are balanced on both sides. The thesis said only "even-indexed" and "odd-indexed".

A critic could say that the vector failed because a direction learned with loan names does not carry over to medical names.

## How we found it

While explaining the steering setup, I listed each task's family, domain and σ_t from `data/synthetic/v3/eval/task_manifest.json`.

## What we checked

`scripts/p6_steer_domain_check.py` refits the vector on the 12 medical tasks from the same stage-A residuals. Results are in `results/v3/synthetic/h200/p6/steer_domain_check.csv`. The refit on the loan tasks reproduces the saved `models/p6_steer.npz` exactly.

All entries below are cosines between steering directions. Two random directions give |cos| < 0.032 (95th percentile). The fair baseline is the matched-size pair: halves of 6 tasks, each with 3 linear and 3 tree.

| Layer | Loan 12 vs medical 12 | Same domain, 6 vs 6 | Across domains, 6 vs 6 | Label halves, loan | Label halves, medical |
|---|---|---|---|---|---|
| 8 | 0.64 | 0.31 | 0.38 | +0.95 | +0.68 |
| 12 | 0.55 | 0.29 | 0.32 | +0.93 | +0.70 |
| 16 | 0.44 | 0.24 | 0.26 | +0.83 | +0.06 |
| 20 | 0.01 | 0.18 | 0.02 | −0.82 | −0.84 |
| 24 | 0.17 | 0.12 | 0.08 | −0.96 | −0.84 |

- **Layers 8–16:** the vector transfers across domains as well as it does between halves of one domain. The domain does not matter here.
- **Layers 20–24,** where the large margin shifts happened: the vector barely replicates even within a domain (0.12–0.18), and across domains it is close to zero. The reason is the last two columns. The two per-label differences point in nearly opposite directions (cos −0.82 to −0.96), so their average is a small remainder of two large, cancelling vectors, and that remainder is mostly task-specific noise.
- At layers 8–16 the two label halves point the same way, so nothing cancels there. Before the answer forms (the logit lens is at chance until about layer 20), what separates "follows the demonstrations" from "follows the prior" does not depend on the label.

## What it means

The domain split is not why steering failed. At the layers where steering moved the margins, the vector was not a stable direction at all, because averaging over labels cancels it. This measures directly what hiccup 24 inferred from the margins, and it confirms that a label-specific or learned direction is the right next test.

## For the thesis

- Chapter 6, method: say the fitting tasks were the loan tasks and the test tasks the medical ones.
- Chapter 6, results: replace the margin-based explanation of the cancellation with the direct measurement (label halves at cos −0.82 to −0.96 at layers 20 and 24).
- Chapter 6, limitations: one line on the domain split and this check.
