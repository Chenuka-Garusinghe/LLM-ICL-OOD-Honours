# Hiccups: how the design got here

This folder records what went wrong, how it was noticed and what was decided, so that the thesis and the presentation can tell the story of the project, not just its final design.
Each file is one hiccup and uses the same headings:

- **What happened**: the symptom.
- **How we found it**: the evidence, with numbers.
- **Why it matters**: which result it would have distorted.
- **What we did**: the fix or decision, and who made it.
- **Result**: what the fix changed, measured.
- **Evidence**: files, archives and commits to cite or re-analyse.
- **How to tell it**: a draft paragraph, a figure idea and a likely examiner question.

The numbers come from the evidence files listed in each entry.
The "How to tell it" paragraphs are drafts in your voice to adapt, not final text.

## The story in four acts

1. **The first run measured the instrument, not the model** ([12](12-v1-generator-gate-shaped-the-rules.md), [01](01-v1-run-results-were-artefacts.md)).
   Even before it, a generator gate had shaped the rule families: the tree family's constraints and the linear family's Bernoulli labels came from making a simple learner take the planted shortcut.
   v1's headline results were artefacts of five code defects. Lesson: gate every expensive run on a check that the instrument works.
2. **The question needed a different testbed and a different scale** ([02](02-tableshift-cannot-isolate-shift-types.md), [03](03-compute-lost-and-scale-pivot.md)).
   TableShift cannot isolate shift types, so the experiments became synthetic. The GPU cluster was lost and 72B was too slow on a laptop, so the thesis moved to 7–8B models and to the question of whether demonstrations can override a small model's priors.
3. **Rebuilding cleanly** ([04](04-v2-code-audit.md), [05](05-citations-rejected-papers.md), [06](06-p1-implementation-bugs.md)).
   An audit of the v2 code, a per-task v3 design with tests for every invariant, proper attribution of borrowed ideas, and the bugs those tests caught.
4. **Validating the instrument on real models (P2)** ([07](07-bf16-output-rounding.md), [08](08-cache-agreement-rule-near-ties.md), [09](09-numeric-prior-fixed-spurious-direction.md), [10](10-llama-answer-format.md), [11](11-throughput-and-compute-budget.md)).
   - Numerical precision of the scores.
   - What "cached equals full" should mean.
   - A hidden numeric prior that lined up with the planted shortcut.
   - A model that was not answering in the expected format.
   - The real compute cost.
   - A filter on the covariate shift that quietly selected shifts of irrelevant features ([13](13-covariate-shift-mostly-moves-distractors.md)).
   - The name priors the naming factor relies on turned out to be mostly a numeric prior, so the prior is now measured directly ([14](14-name-priors-dominated-by-numeric-prior.md)).
   - Contextual calibration left whole prompts predicting one class, which would have hidden shift effects in balanced accuracy, so the tests use AUROC ([15](15-calibration-collapses-random-demos.md)).
   - Counter-prior demonstrations also advertised the shortcut, because the model's prior includes the spurious feature, so a matched variant holds the shortcut fixed ([16](16-counter-prior-reinforces-the-shortcut.md)).
   - P4 moved to a cloud H200: eight copies of the model shared one GPU, so all of P4 ran on one device in 42 minutes and replicated the Mac ([17](17-p4-moved-to-a-cloud-h200.md)).
   - The covariate null turned out to be about the metric: a uniform shift moves Qwen's scores together, which within-environment AUROC cannot see, and the shortcut stays valid under that shift ([18](18-covariate-harm-invisible-to-within-environment-auroc.md)).

## Timeline

| Date (2026) | Phase | Hiccup | Outcome | File |
| --- | --- | --- | --- | --- |
| 6 Sep | v1 | The generator gate failed (12% of tasks): random trees were too easy (dictators), unlearnable (parity) or unaffected by the mechanism shift | opposite leaves get opposite labels; dictators and parity rejected; Bernoulli linear labels (gate 90%) | [12](12-v1-generator-gate-shaped-the-rules.md) |
| by 14 Sep | v1 | First full run's results were artefacts (96–99% class-1 bias; SATA 0.045 under spurious reversal) | v2 redesign with gates | [01](01-v1-run-results-were-artefacts.md) |
| 18 Sep | v2 | TableShift names three shift types but isolates none | synthetic-first; no TableShift-fitted corrections | [02](02-tableshift-cannot-isolate-shift-types.md) |
| 19–27 Sep | v2 | GPU cluster lost; 72B at about 44 s/query; MLX kept as a second backend | 7–8B models; PyTorch only | [03](03-compute-lost-and-scale-pivot.md) |
| 26 Sep | v3 plan | v2 code audit: 14 defects | per-task v3 design (P1) | [04](04-v2-code-audit.md) |
| 27 Sep | spec | Borrowed ideas lacked citations; two cited papers were rejected at review | 119 verified references; rejected papers removed | [05](05-citations-rejected-papers.md) |
| 28 Sep | P1 | Ported selector bug, flaky test, snapshot built from the wrong task | fixed; gate G1 passed | [06](06-p1-implementation-bugs.md) |
| 28 Sep | P2 | bf16 output layer rounded the scores | fp32 label logits; normaliser fixed | [07](07-bf16-output-rounding.md) |
| 28 Sep | P2 | Llama missed the 99% cache-agreement rule, on near-ties only | AUROC rule adopted (mean AUROC within 0.005); Qwen passes (0.0025) | [08](08-cache-agreement-rule-near-ties.md) |
| 28 Sep | P2 | Qwen's "bigger values mean 1" prior backed the fixed-direction shortcut | spurious direction randomised per task; P2 rerun | [09](09-numeric-prior-fixed-spurious-direction.md) |
| 28 Sep | P2 | Llama puts about 2% of its probability on the answer tokens: it starts explaining ("To infer the rule…") | label-mass validity rule (≥ 0.9); Llama paused (the user's decision; a `Label: ` prefill fixes it if resumed) | [10](10-llama-answer-format.md) |
| 28 Sep | P2 | Measured PyTorch cost differs from the MLX-based budget | units cost 1.3× more; total about 65–71 GPU-hours (similarity seeds still to be fixed) | [11](11-throughput-and-compute-budget.md) |
| 30 Sep | spec review | The label-rate filter on the covariate shift accepts almost only shifts of distractors (21 of 24 tasks) | two load-bearing features shifted in opposite label directions plus a distractor; label rate unchanged by construction | [13](13-covariate-shift-mostly-moves-distractors.md) |
| 30 Sep | P3 | Qwen's name priors are mostly "bigger value means 1": 1 of 16 pairs passes the symmetry rule; surrogate R² 0.20/0.28 (G3 needs 0.5); aligned − flipped zero-shot AUROC still 0.164 (loan 0.28, medical 0.05) | naming kept (the user's decision); pairs chosen by measured contrast; counter_prior uses measured zero-shot priors on the pool rows | [14](14-name-priors-dominated-by-numeric-prior.md) |
| 30 Sep | P4 | Calibration with a content-free query left 36% of plain-random cells predicting one class (balanced accuracy pinned at 0.5; shift gaps forced to 0) | AUROC made the confirmatory statistic (the user's decision); calibrated BA reported alongside | [15](15-calibration-collapses-random-demos.md) |
| 1 Oct | P4 (H200) | counter_prior over-selected shortcut-agreeing rows where the shortcut opposes Qwen's "bigger = 1" prior (91% against 84%) | kept as pre-registered; exploratory `counter_prior_matched` added and run (the user's decision) | [16](16-counter-prior-reinforces-the-shortcut.md) |
| 1 Oct | P4 | The Mac chain was projected at about 30 h and stopped partway when its session ended | all of P4 rerun on one Runpod H200 with 8 MPS workers: 42 min, about $3.30; replicates the Mac (mean AUROC within 0.001) | [17](17-p4-moved-to-a-cloud-h200.md) |
| 1 Oct | P4 analysis | Covariate shift showed no harm (C1a +0.024); the user asked whether shortcut learning explains it | not the shortcut (removing it opens no covariate gap) but the metric: the uniform shift raises scores by 0.44–1.21 logits without reordering; a variance shift (new probe) hurts only under flipped names (−0.09 to −0.13); all three follow-ups done (the user's decision) | [18](18-covariate-harm-invisible-to-within-environment-auroc.md) |
| 5 Oct | Spec review | feature_range spread over the shortcut feature in all 24 tasks (the spec said "often"), so its demonstrations show the shortcut agreeing in 66% of rows against 84% for label_diversity | open: keep as pre-registered, or add a rule-feature coverage variant | [19](19-feature-range-always-covers-the-shortcut.md) |
| 5 Oct | Spec review | The user argued that demonstrations should not show the shortcut or the noise feature, since they are meant to point to the causal signal | exploratory experiment (the user's design): three arms (full, no shortcut, no shortcut or noise) with queries kept complete; hidden features from the ground truth; selection blind to them; built and tested, run pending (notebook 03.1) | [20](20-demonstrations-without-the-shortcut.md) |

## Keeping it up to date

Add a numbered file for each new hiccup, using the headings above, and a row to the timeline.
When an open item is decided, update its status line and the timeline row rather than starting a new file.
