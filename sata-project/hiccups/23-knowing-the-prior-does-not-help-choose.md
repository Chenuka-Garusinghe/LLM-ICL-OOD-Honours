# 23 — Telling SATA what Qwen believes helps on the proxy, not on Qwen

**When:** 11 October 2026, P5 stage G (5b) on a Runpod H200 · **Status:** result. Gate 5b failed; the 5b contrast family R7–R10 is null.

## What happened

5a showed that SATA picks good rule evidence, but Qwen-Instruct mostly reads its "bigger → 1" prior instead (hiccups/22). 5b gave SATA the prior itself. For every pool row and the query, it read Qwen's measured zero-shot margin under the task's naming (`sata_mi`). The control `sata_names` read only the names' lexicon directions.

On the proxy, SATA-MI did what it was built to do. On Qwen it changed nothing.

## How we found it

**Gate S3a-MI** (reader proxy, 60 validation tasks): the MI targets reach a worst environment of 0.619, against 0.586 for counter_spurious. Passed.
- The planned counter-prior bonus made no difference: λ = 0, 0.5 and 1 score 0.619, 0.616 and 0.617. λ = 0 was kept, so SATA-MI's targets are the 5a matched targets.

**Training** (30 epochs, the user's budget cap): SATA-MI reached 0.620 at epoch 25, the targets' own level. SATA-names reached 0.590. The whole gap is under spurious reversal.

**Gate 5b** (LLM validation, 8 held-out tasks): failed on both halves.
- Proxy–Qwen Spearman over strategy × environment: 0.60 abstract, 0.48 aligned, 0.47 flipped (bar 0.5). This is in-sample, because the reader was fitted on these tasks.
- SATA-MI ≥ both controls only under abstract names. Under aligned names it is below SATA-names (mean AUROC 0.607 against 0.637). Under flipped names it is also below SATA-names (0.504 against 0.526).

**R7–R10** (24 evaluation tasks, flipped names, Holm over 4; frozen before any 5b result):

| Contrast | Estimate [95% CI] | Holm p | Tasks in direction |
| --- | --- | --- | --- |
| R7 sata_mi − sata, ID | +0.005 [−0.050, 0.062] | 1.00 | 11/24 |
| R8 sata_mi − sata_names, ID | −0.010 [−0.066, 0.047] | 1.00 | 10/24 |
| R9 sata_mi − counter_prior, ID | −0.032 [−0.099, 0.036] | 1.00 | 11/24 |
| R10 sata_mi − sata, mean of 3 environments | +0.001 [−0.031, 0.032] | 1.00 | 12/24 |

Under flipped names every strategy is close to chance: means 0.44–0.52, zero-shot 0.454.

## Why it matters

- **The bottleneck is not the selector's knowledge of the prior.** With 8 numeric demonstrations, Qwen-Instruct under flipped names barely moves from its prior, whatever it is shown. A selector can only choose evidence, and knowing what the model believes does not help when the model does not update on evidence.
- **It closes the SATA line for the instruct model.** 5a (structure) and 5b (structure plus prior) are both null. The positive SATA result stays on Qwen-base, whose priors pull less (hiccups/22).
- **It supports the "turn the prior" idea the user parked** (research_plan.md): working with the prior, for example by choosing demonstrations whose labels contradict it more strongly, would need much larger effects than any selection at k = 8 produced here.

## What we did

- Ran the plan as written: stage G in one H200 session, 1.85 h, about $9.80. Training was capped at 30 epochs instead of 50 for budget, which is the user's decision. SATA-MI's best was at epoch 25, but its spurious-reversal score was still edging up at the cap.
- Reported R7–R10 as pre-registered (notebook 05, section 12), with gate 5b's failure stated first (notebook 04, section 6).

## Evidence

- `results/v3/synthetic/p5/gate_s3a_mi.json`, `train_sata_mi.json`, `train_sata_names.json`, `gate_5b.json`, `llmval_mi_proxy_vs_qwen.parquet`; the Qwen priors for the new suites in `p5/mi/`.
- `results/v3/synthetic/h200/p5/llmval_mi.parquet`, `grid_mi.parquet`, `mi_contrasts.parquet`, `logs5b/`.
- `models/sata_mi.pt`, `models/sata_names.pt`.

## How to tell it

> The trained selector chose good evidence, but the instruction-tuned model mostly followed its own prior habit that bigger numbers mean "1". So we told the selector what that habit looked like, row by row, measured from the model itself. On our stand-in for the model, this helped: it chose sets that work against the habit and matched the best choices we could compute. On the real model it made no difference. When the variable names pointed the wrong way, every strategy, including the new one, stayed near a coin flip. Knowing what a model believes doesn't help you choose its examples if eight examples can't change its mind.

- **Figure idea:** flipped-name AUROC per strategy (notebook 05, section 12), every dot packed around 0.5.
- **Examiner question:** "Was 30 epochs enough?"
  - Probably, for the proxy: SATA-MI already matched its own targets (0.620 against 0.619), though its spurious-reversal score was still edging up at the cap.
  - The proxy itself failed to track Qwen (gate 5b), which is the binding problem.
