# 22 — SATA wins on the proxy, ties on Qwen-Instruct, and helps Qwen-base

**When:** 11 October 2026, P5 stage F on a Runpod H200 · **Status:** result. RQ4 confirmatory family null; base model exploratory positive.

## What happened

Trained SATA-v2 (matched targets, drawn selection; hiccups/21) beat every hand-designed strategy on the XGBoost proxy:
- worst-environment accuracy 0.652, against 0.555 for counter_spurious;
- no weak environment: ID 0.669, covariate 0.654, spurious reversal 0.652.

On Qwen2.5-7B-Instruct the advantage did not carry over.

## How we found it

**RQ4 family** (AUROC, one-sided hierarchical bootstrap, Holm over 6; 24 evaluation tasks; frozen 10 October, before any SATA result):

| Contrast | Estimate [95% CI] | Holm p | Tasks in direction |
| --- | --- | --- | --- |
| R1 sata − label_diversity, spurious reversal | +0.032 [−0.024, 0.089] | 0.79 | 14/24 |
| R2 sata − label_diversity, covariate | −0.096 [−0.154, −0.041] | 1.00 | 5/24 |
| R3 sata − label_diversity, ID | −0.040 [−0.102, 0.023] | 1.00 | 8/24 |
| R4 sata − similarity, mean of 3 environments | −0.020 [−0.064, 0.026] | 1.00 | 11/24 |
| R5 sata − sata_agnostic, mean of 3 environments | −0.025 [−0.063, 0.010] | 1.00 | 10/24 |
| R6 sata − label_diversity, flipped, ID | −0.014 [−0.077, 0.048] | 1.00 | 13/24 |

**SATA flattens rather than raises.** Under abstract names its ID, covariate and reversal AUROCs are 0.553, 0.556 and 0.552, like counter_spurious (0.561, 0.575, 0.555). Its worst environment ties counter_spurious (0.552 against 0.555) and beats label_diversity (0.520).

**Qwen-base is different** (exploratory, reduced grid, paired bootstrap, no Holm):
- SATA has the best worst environment under both namings: abstract 0.589 (counter_spurious 0.550, random 0.504) and flipped 0.551 (counter_spurious 0.471, random 0.461).
- Against random, spurious reversal +0.085 (abstract) and +0.094 (flipped), both intervals above 0, with no ID cost (−0.027, +0.008).
- Under flipped names it also beats counter_spurious on ID (+0.077) and covariate (+0.079).

**The proxy still ranks Qwen's strategies moderately well:** Spearman over strategy × environment on the 8 validation tasks is 0.68 under abstract names and 0.46 under named ones.

## Why it matters

- **This answers RQ4 for the instruct model: no.** A selector trained on ground-truth structure does not beat simple balanced random demonstrations on Qwen2.5-7B-Instruct. It reproduces what neutralising the shortcut does: gains under spurious reversal, losses on ID and covariate.
- **Notebook 03.1 explains why.**
  - The proxy rewards sets from which a learner can recover the rule from 8 rows.
  - Qwen-Instruct under abstract names mostly reads the shortcut and its "bigger → 1" prior. With the shortcut neutralised it falls back towards chance, so better rule evidence has little to work on.
- **The base model, which leans less on priors (P4: −0.065 against −0.119 from flipped names), can use it.** That fits Wei et al.'s graded picture: following in-context evidence over priors depends on how strong the priors are, not only on the demonstrations.

## What we did

- Ran the frozen RQ4 family as written (notebook 05, section 11), with the base model as an exploratory section 11b.
- 5b (SATA-MI) is the plan's next step: SATA reads Qwen's measured prior for every row. The instruct result makes it more relevant, since the prior is the bottleneck.
  - Its gate needs proxy–LLM Spearman ≥ 0.5; abstract names pass (0.68), named ones fall just short (0.46).
  - Decide with the user before running it.

## Evidence

- `results/v3/synthetic/h200/p5/grid_sata.parquet`, `grid_sata_base.parquet`, `llmval.parquet`, `rq4_contrasts.parquet`, `base_sata_contrasts.parquet`.
- `results/v3/synthetic/p5/proxy_trained.parquet`, `train_*.json`; notebook 04 (proxy, training, LLM validation); notebook 05, section 11.

## How to tell it

> We trained a small transformer, about 0.8 million parameters, to choose demonstrations, using synthetic tasks where the best choices are known. Judged by a simple classifier fitted on its choices, it beat every hand-designed strategy in every shift. Given to a 7-billion-parameter instruction-tuned model, its demonstrations did no better than balanced random ones. The trade-off was the same as when the shortcut is removed: better under the reversed shortcut, worse elsewhere. The base version of the same model, which leans less on its prior habits, did benefit, with the best worst-case accuracy of any strategy. The demonstrations carried the rule. The instruction-tuned model mostly did not use it.

- **Figure idea:** proxy worst-environment accuracy (x) against Qwen worst-environment AUROC (y), one dot per strategy, separate panels for instruct and base.
- **Examiner question:** "Is SATA's failure on the instruct model a failure of SATA?"
  - The proxy says the sets are good, and the base model says they are usable.
  - The instruct model's own reliance on the shortcut and its prior (RQ3, notebook 03.1) is the simpler explanation.
