# 26 — Flipped names are not concept shift, and SATA's robustness has a price

**When:** 5 November 2026, final supervisor review before submission · **Status:** applied in the Overleaf files (diff: `thesis/overleaf-edits-2026-11-05.diff`).

## What happened

The supervisor's final review raised two content points, alongside proofreading items:

1. The thesis called the flipped-name condition "concept shift". Renaming features does not change the rule that generates the labels, so $P(y \mid x)$ in the data never changes. The thesis had defended the term "from the model's point of view" in a critique, but the supervisor asked for a precise description.
2. The trade-off between SATA's robustness and its predictive performance was stated only in passing ("flattens the environments").

## How we found it

Both points came from the supervisor's written review. The numbers for the trade-off were recomputed from the result files: `p5/gate_s3a.parquet` and `p5/proxy_trained.parquet` for the stand-in learner; `p4` cells plus `p5/grid_sata.parquet` for Qwen-Instruct; `p4/grid_base.parquet` plus `p5/grid_sata_base.parquet` for the base model.

For each strategy, we compared its **weakest environment** (robustness) with its **mean over the three environments** (predictive performance):

| Learner | SATA weakest | SATA mean | Comparator weakest | Comparator mean |
|---|---|---|---|---|
| Stand-in learner (accuracy) | 0.652 | 0.658 | counter_spurious 0.555 | similarity 0.609 (highest hand-designed) |
| Qwen-Instruct, abstract (AUROC) | 0.552 | 0.554 | label_diversity 0.520 | label_diversity 0.588 |
| Qwen-Instruct, aligned | 0.624 | 0.628 | label_diversity 0.642 | label_diversity 0.669 |
| Qwen base, abstract (exploratory) | 0.589 | 0.601 | random 0.504 | counter_prior 0.591 |

## Why it matters

- Point 1 matters because an examiner could read "concept shift" as a claim about the data.
- Point 2 changes how SATA's null should be read. The trade-off is free on the stand-in learner and on the base model: there, SATA is best on both counts. Only on the instruction-tuned model does SATA trade mean performance for a flatter profile. Under aligned names, it is worse on both counts.

## What we did

- **The rename (the supervisor's request, wording mine).** "Concept shift" became **semantic-prior conflict** wherever it described flipped names. Method §3.3 now has one paragraph explaining how it differs from conventional concept shift. The discussion critique "Naming is not concept shift" was dropped.
- **RQ3 wording.** RQ3 now asks *where* in the model the prior wins, not *why*, because the logit lens and probes locate the effect but do not explain the decision.
- **The trade-off.** A new figure, `fig_sata_tradeoff.pdf` (`figures.py::fig_sata_tradeoff`), plots the weakest environment against the mean for the three learners. The Discussion gained a paragraph on SATA's robustness and its cost.
- **One main takeaway, stated the same way** in the abstract, introduction, discussion and conclusion: *the prior carried by the feature names influences the model's predictions more than the choice of demonstrations does.* In numbers, flipping names costs 0.20 AUROC; the strategies' typical difference from label_diversity is 0.025, the median |difference| over 54 cells.

## Result

The thesis compiles with no errors or undefined references.

Proofreading fixes from the same review:

- the duplicated References entry in the contents (amsbook already lists starred chapters, so `heading=bibintoc` listed it twice);
- a missing `\newtheorem{lemma}`;
- the repository link;
- the acknowledgements (a draft for the user to personalise);
- `fig_reversal_direction.pdf` regenerated without the undocumented `counter_prior_matched` row.
- (6 Nov) The §7.3 covariate critique ("the shift is too weak") now rests only on planned evidence: the design's 1–2 SD shift, the planned covariate contrast (+0.024 [−0.044, 0.093]) and the detected cost of flipped names (0.20). `fig_covariate_strength` was dropped: 14 of its 15 intervals, and the shift-size correlation, were exploratory.

## How to tell it

*Draft paragraph:* "We first called the flipped-name condition concept shift, because from the model's side the mapping from features to label reverses. But the data-generating rule never changes, so we now call it semantic-prior conflict: what we test is whether demonstrations can teach an unchanged rule that the model's prior contradicts."

*Likely examiner question:* "Why not test real concept shift?" If the queries followed a different rule from the demonstrations, the demonstrations would be wrong, and no selection could help.
