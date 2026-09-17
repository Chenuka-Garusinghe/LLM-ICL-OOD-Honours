# Demonstration protocols for tabular OOD tasks — specification and pre-GPU screen

Scope: the real arm (TableShift). Defines the protocol design space, replaces the
v1 protocol list with a factorial, and reports a CPU-only screen that filters the
grid before any GPU time is spent. All v1 results under `results/` are treated as
void (see `REDESIGN_RATIONALE.md`); nothing here derives from them.

New code: `src/selection/shift_estimation.py`, `src/selection/protocols.py`,
`scripts/prep_shift_context.py`, `scripts/screen_protocols.py`. No existing
module was modified.

---

## 1. The design problem, restated

A demonstration protocol is a deterministic map

    (demo pool, query, k, seed, shift context) -> ordered prompt

The v1 suite treats "protocol" as a single categorical choice with six levels.
That conflates four independent axes:

| axis | what it controls | v1 status |
|---|---|---|
| **selection** | which k rows | the only axis varied |
| **composition** | the demo set's label counts | uncontrolled, then pinned to 50/50 |
| **ordering** | demo position in the prompt | fixed by `src/selection/ordering.py` (correct) |
| **serialisation** | row → text, number format, schema preamble | fixed |

This document formalises **selection × composition** and leaves ordering and
serialisation as the controlled nuisances they already are. Composition is
promoted from nuisance to factor for a measured reason, given in §3.

---

## 2. Deployment-legal shift estimation

The protocols must not be told which shift they face — a deployed system is not.
What it *can* see is unlabelled target-domain inputs. Three estimands are built
from (labelled source pool, unlabelled target inputs) and nothing else:

- **s(x), w(x)** — domain-discriminator score and density ratio p_T(x)/p_S(x).
  Addresses covariate shift and extrapolation.
- **π_T** — target label prior via BBSE (Lipton et al. 2018). Addresses prior shift.
- **per-feature drift** — SMD (numeric) / total-variation (categorical). Ranks
  shortcut candidates and supplies the feature ordering for coverage protocols.

Fitted per dataset by `scripts/prep_shift_context.py`; see
[`shift_context_summary.csv`](shift_context_summary.csv).

| dataset | domain AUC | p(x) shift detectable | π_S | π̂_T used | BBSE raw | π̂_T reliable | π_T true |
|---|---|---|---|---|---|---|---|
| `acsincome` | 0.915 | yes | 0.323 | 0.385 | 0.385 | yes | 0.399 |
| `acspubcov` | 0.808 | yes | 0.224 | 0.224 | **1.000** | **no** | 0.637 |
| `brfss_diabetes` | 0.669 | yes | 0.125 | 0.208 | 0.208 | yes | 0.174 |
| `anes` | 0.557 | **no** | 0.696 | 0.663 | 0.663 | yes | 0.617 |

The two diagnostics dissociate, which gives an empirical shift taxonomy of the
benchmark rather than an assumed one:

- `acsincome` — covariate-shift dominant, prior mild and well estimated.
- `anes` — **no detectable p(x) shift** at all (AUC 0.557); the movement is in
  the prior / p(y|x). Any protocol conditioning on w(x) is conditioning on noise
  here, which is why `DomainShift.detectable` exists and why the shift-aware
  mechanisms degrade to uniform on this dataset.
- `acspubcov` — the largest prior shift (0.224 → 0.637) is the one that is **not
  identifiable** from unlabelled inputs. BBSE's solution pins to the simplex
  boundary at 1.000, the detectable signature of its p(x|y)-invariance assumption
  failing. The estimator reports `prior_reliable=False` and falls back to π_S
  rather than shipping a confidently wrong number.

That last row is a finding, not a bug: on the dataset where prior correction
would matter most, the correction cannot be estimated without target labels.

### 2.1 The RQ1 premise needs a stronger gate than "there is a gap"

An ID-trained HistGradientBoosting model on all 12 features:

| dataset | acc ID | acc OOD | AUROC ID | AUROC OOD | majority-class OOD |
|---|---|---|---|---|---|
| `acsincome` | 0.815 | 0.803 | 0.884 | 0.893 | 0.602 |
| `brfss_diabetes` | 0.876 | 0.832 | 0.810 | 0.808 | **0.826** |
| `acspubcov` | 0.797 | 0.608 | 0.769 | 0.796 | **0.637** |
| `anes` | 0.808 | 0.762 | 0.844 | 0.833 | 0.617 |

Two things follow.

**AUROC is flat across every shift; accuracy is not.** Discriminative structure
transfers almost perfectly (`acsincome` even improves, 0.884 → 0.893). What breaks
is the decision threshold, and it breaks in proportion to the prior movement. On
this benchmark, OOD failure is a *calibration* failure, not a representation
failure.

**On two of four datasets, a fully supervised model is at or below the
majority-class baseline OOD** — `brfss_diabetes` 0.832 vs 0.826, `acspubcov`
0.608 vs 0.637. Accuracy is therefore close to uninformative on those two, and
NB00's data gate should add `acc_ood > majority_ood` to its criteria; as written
(`gap >= 0.02`) `acspubcov` passes the gate while being unusable on accuracy.

**Recommendation.** Make balanced accuracy (or AUROC) the headline metric and
report raw accuracy beside it. This is not cosmetic: §3 shows the two metrics
select different protocol winners.

---

## 3. The protocol grid

`src/selection/protocols.py` defines protocols as (mechanism × composition).

**Mechanisms** — score, weight, or stratify the pool:

| mechanism | rule | targets |
|---|---|---|
| `random` | uniform | baseline |
| `similarity` | k nearest by Gower distance | local decision boundary |
| `feature_coverage` | marginal quantile-bin coverage over highest-drift features | feature-range / extrapolation |
| `rule_coverage` | depth-3 decision-tree leaf coverage | rule diversity |
| `counter_spurious` | rank by number of shortcut features contradicted | shortcut breaking |
| `importance_weighted` | sample ∝ w(x) | covariate shift |
| `shift_axis_coverage` | strata over s(x) quantiles | span the shift direction |

**Compositions** — the demo set's label counts at fixed k:

| composition | rule |
|---|---|
| `free` | mechanism decides (≈ source prior for the unstratified pool) |
| `balanced` | k/2 per class — v1's `balanced_topk` behaviour |
| `target_prior` | round(k·π̂_T) positives, clipped to keep both classes present |

v1's suite is recovered as cells: `label_diversity` = `random × balanced`,
`similarity` = `similarity × free`, and so on (`protocols.V1_EQUIVALENTS`). The
factorial makes the confound estimable instead of baked in.

### 3.1 Defects found in the v1 selectors, and what changed

1. **`feature_range` did not deliver feature-range coverage.** It binned the top
   3 features *jointly* (up to 27 cells) and sampled 8 distinct joint cells,
   which gives no guarantee that any single feature's bins are all represented —
   the property that matters when the target sits in an unseen range. Replaced by
   stratification on (feature, bin) pairs, ordered by drift rather than mutual
   information: the ranges worth spanning are the ones that move.

2. **`counter_spurious` cannot run on the real arm as written.** It requires a
   `shift_col` naming the domain variable; the extracted parquet cache does not
   contain it (TableShift consumes the splitter when forming the splits).
   Replaced by ranking on |corr(feature, label)| × drift(feature). It also
   thresholded on the *pool's* median, so a row's counter-shortcut status moved
   with the sampling seed; thresholds now come from the labelled ID reference.

3. **`similarity` used sentence-embedding cosine on serialised rows.** For numeric
   rows, MiniLM similarity tracks surface token overlap of formatted numbers, and
   is worst behaved exactly where OOD values fall outside the demonstrated range.
   Replaced by Gower distance on feature values. Keep the text-embedding variant
   as a deliberate ablation, not the default.

4. **Return-type inconsistency.** `similarity_select.select` returned positional
   offsets into `pool_texts` while every other selector returned pool index
   labels. All mechanisms now return index labels.

5. **Four of six v1 protocols ignore `query` entirely** (it is in the signature,
   unused). That is a legitimate design choice but it means those arms are
   stratified samplings of the pool, not retrieval — one demo set per (pool, seed)
   rather than per query. §4 shows this is the single largest distinction in the
   results, so it should be stated as a factor (`query_conditional`) rather than
   left implicit.

6. **`build_demo_pool` returns a label-stratified (50/50) pool.** With composition
   now a factor, a pre-balanced pool pre-empts it. The screen draws an
   unstratified pool that preserves π_S; `build_demo_pool` should gain a
   `stratify` flag before the GPU runs.

---

## 4. Pre-GPU screen

`scripts/screen_protocols.py`. For every cell, build the demo set the protocol
would put in the prompt and score it with three **order-invariant surrogate
readers**, each standing in for a documented ICL behaviour:

- **`prior_only`** — predict Bernoulli(demo positive fraction); the
  majority-label / label-copy channel (Zhao et al. 2021; the "label space, not
  input–label mapping" reading of Min et al. 2022). Reads no query.
- **`nn1`** — label of the nearest demo. Retrieval without rule learning.
- **`logreg`** — L2 logistic regression fit on the k=8 demo rows. Task *learning*;
  the optimistic end of what 8 rows can support.

4 datasets × 20 cells × 5 seeds × 250 queries × {ID, OOD}, k=8, `id_pool`
provenance (demos from source, queries from target). Full cell table:
[`protocol_screen_cells.csv`](protocol_screen_cells.csv); ranking:
[`protocol_ranking_logreg.csv`](protocol_ranking_logreg.csv).

**This is a necessary-condition filter, not a substitute for the GPU runs.** It
answers "is the information in the eight rows at all", and it is blind by
construction to ordering, serialisation, and prompt format — the axes that need
the LLM. A protocol whose demo sets carry no more OOD information than random-k
does not deserve GPU hours; passing does not imply an LLM will exploit it.

### 4.1 Results

**Mechanism matters; the winner is query-conditionality.** OOD balanced accuracy
averaged over compositions and datasets (`logreg`):

    similarity 0.586 > feature_coverage 0.552 > importance_weighted 0.549
      > random 0.541 > shift_axis_coverage 0.540 > rule_coverage 0.533
      >> counter_spurious 0.442

Best cell per dataset against the random-k baseline:

| dataset | best cell | balanced acc | Δ vs random-k |
|---|---|---|---|
| `acsincome` | `similarity × free` | 0.701 | +0.149 |
| `anes` | `similarity × free` | 0.663 | +0.149 |
| `brfss_diabetes` | `feature_coverage × balanced` | 0.683 | +0.178 |
| `acspubcov` | `similarity × balanced` | 0.543 | +0.040 |

**Counter-spurious selection is below chance on all four datasets** (0.407–0.495;
worst cell −0.146 vs random-k). Over-sampling shortcut-contradicting rows makes
the demo set unrepresentative of the target region, and the surrogate generalises
worse than from a random draw. This has a direct consequence for Stage 3a of the
redesign, which gives `is_counter_spurious` a **+1.5** weight in the SATA target:
on the real arm that term rewards demo sets that measurably carry *less*
OOD-usable signal. Gate S3a should be run with and without the term before
training, and the term should probably be scoped to the synthetic arm where
`is_counter_spurious` is ground truth rather than an estimate.

**The composition effect is almost entirely a metric artefact.** Averaged over
mechanisms and datasets (`logreg`, OOD):

| composition | raw accuracy | balanced accuracy |
|---|---|---|
| `balanced` (50/50) | 0.546 | 0.550 |
| `free` (source prior) | 0.617 | 0.540 |
| `target_prior` | 0.582 | 0.529 |

On raw accuracy, composition looks like the dominant factor and `free` looks like
the clear winner — because on imbalanced datasets a demo set carrying the source
prior makes the reader predict the majority class, which scores well without
using the query. On balanced accuracy the ordering nearly vanishes (0.550 /
0.540 / 0.529). **Composition moves where the threshold sits; mechanism moves how
well the demo set separates the classes.** They act on different quantities, and
which one appears to matter is decided by the metric you report. The v1 design
reports raw accuracy — which is why a protocol-vs-shift interaction could look
absent even if mechanism effects were present.

Consistent with this, the ID→OOD gap in balanced accuracy is ≈0.002 for every
composition, against raw-accuracy gaps of 0.09–0.14.

**Per-dataset headroom for demonstration content** (`logreg` − `prior_only`,
balanced accuracy):

| dataset | headroom |
|---|---|
| `acsincome` | +0.076 |
| `anes` | +0.049 |
| `brfss_diabetes` | +0.034 |
| `acspubcov` | **+0.002** |

On `acspubcov` a reader that learns a mapping does no better than one that copies
the demo label distribution. No selection protocol can help there, because the
information is not in the eight rows to begin with. Treat `acspubcov` as a
negative control, not a test case.

**Graceful degradation works.** On `anes` (domain AUC 0.557) `importance_weighted`
and `shift_axis_coverage` correctly fell back to uniform and are numerically
identical to `random`, flagged `degraded=True` (45 cells each).

### 4.2 Statistical note on seed count

With 5 paired seeds, the two-sided Wilcoxon signed-rank floor is p = 0.0625. The
gains above report p = 0.062, i.e. all 5 seeds moved the same way — the strongest
result obtainable at n=5, and still not nominally significant at 0.05. If the
thesis needs inferential claims about protocol differences, `seed_accuracy` needs
more than 5 entries, or the test must be over queries with seed as a random
effect. This applies to the GPU runs as configured.

---

## 5. How this bears on the redesign, and on the six papers

The literature partitions cleanly across the two arms, which is worth stating
explicitly in the thesis because it makes the arms test different claims.

**Real arm = can ICL override priors under shift? (scale-gated.)** Features carry
real-world semantics (income, BMI, turnout), so the model has priors about the
mapping. Wei et al. (2023, *Larger language models do in-context learning
differently*) find prior-override emerges with scale; Krishna Kumar (2025),
*Semantic Anchors in In-Context Learning: Why Small LLMs Cannot Flip Their
Labels* (arXiv 2511.21038, 2025-11-26), reports across eight classification
tasks and eight open-source LLMs of 1–12B parameters that with inverted
demonstrations models "cannot learn coherent anti-semantic classifiers", with
semantic override rates **exactly zero** in that few-shot 1–12B setting, and
concludes that ICL mainly adjusts how inputs project onto stable pre-trained
semantic directions rather than remapping label meanings — so overriding label
semantics at these scales "requires interventions beyond ICL". Both
primary models (Llama-3.1-8B, Qwen2.5-7B) sit inside that band. So any
demonstration protocol whose mechanism requires contradicting the model's prior —
`counter_spurious` above all — is predicted to fail at 7–8B *for reasons
independent of the protocol's quality*. That is a pre-registrable prediction, and
it makes Gate S1's escalation ladder (8B → 32B → 70B) a **designed factor**
rather than a rescue path. Recommend running the smallest sufficient grid at each
scale instead of the full grid at 8B.

**Symbol tuning is a training intervention, not a prompt one.** The commit
"updating LABEL_TOKEN … to reduce semantic prior dependence" is the right move
for the label channel, but Wei et al.'s symbol-tuning result comes from
finetuning; neutral label tokens in the prompt do not buy the same robustness.
Note also that neutralising label tokens leaves the *feature* side semantic —
"AGEP: 67, WKHP: 50" still evokes priors. To isolate input–label mapping learning
on the real arm you would have to anonymise feature names too, which is worth one
ablation cell rather than a redesign.

**Min et al. 2022 vs Yoo et al. 2022 should be measured, not chosen.** Whether
input–label correspondence is used at all determines whether *any* selection
protocol can work, and the answer is task- and scale-dependent. Recommend adding
a **label-corruption ablation as a gate before the protocol grid**: run
`random × balanced` at 0% / 50% / 100% demo-label corruption. If accuracy is flat
in corruption level, the model is doing task recognition and the protocol grid has
zero ceiling on that dataset/model — a decisive, cheap result that either
justifies the GPU spend or redirects it. This is the missing gate in the current
staging: S1 checks the model can do the task, S2 checks the data are sound, but
nothing checks that demonstrations are being *used*.

**LLMTabBench** (Grushina, Kuvshinova, Kostromina, Temirkhanov, Mitrovic &
Simakov, arXiv 2605.24417, 2026-05-23) reports three things that bear directly on
this grid: LLMs can be highly competitive zero-shot on tabular classification,
sometimes outperforming models given few-shot examples; additional examples "may
conflict with prior knowledge, thereby degrading performance"; and there is a
complexity threshold beyond which LLM performance declines and few-shot examples
become less useful.

Three consequences. Zero-shot with a task description must be a first-class arm
in the headline grid, not just a pilot diagnostic, and the headline contrast
should be *protocol vs zero-shot*, not *protocol vs random-k* — if zero-shot wins
on a dataset, demonstration design is not the lever there. Second, because
examples can *actively* conflict with priors, a protocol can plausibly score
below zero-shot; the grid needs to permit and report negative effects rather than
only ranking protocols against each other. Third, the complexity threshold means
k and feature count interact with whether demonstrations help at all, which is an
argument for keeping the k=8/k=16 sweep and the 12-feature budget explicit in the
reporting rather than treating them as fixed constants.

**Calibration and composition are competing interventions on the same
quantity.** Stage 1e adds contextual calibration (Zhao et al. 2021), which
estimates and removes the prompt's label prior. Composition *is* the prompt's
label prior. Stacking them silently means the calibration step absorbs the
composition effect. They must be crossed — {raw, calibrated} × {balanced,
target_prior} — or the composition factor is unmeasurable. Given §2, note that
on `acspubcov` the target prior is not estimable anyway, so calibration is the
only available correction there.

### 5.1 Concrete changes recommended to the staged plan

1. Add **Gate S0c — demonstration-utility gate**: label-corruption ablation per
   dataset × model. No protocol grid on a dataset/model that fails it.
2. Add `acc_ood > majority_ood` to NB00's real-arm checks; `acspubcov` currently
   passes the gap criterion while being unusable on accuracy.
3. Headline metric → balanced accuracy or AUROC, with raw accuracy reported
   alongside. Re-do the RQ2 heatmap in both; expect different structure.
4. Promote model scale to a designed factor; run the reduced grid at 8B and the
   headline cells at 70B.
5. Cross calibration with composition rather than stacking.
6. Run Gate S3a with and without the `is_counter_spurious` target term; consider
   scoping that term to the synthetic arm.
7. Add a `stratify=False` option to `build_demo_pool` and use it wherever
   composition is a factor.
8. Increase `seed_accuracy` beyond 5, or move inference to queries-as-replicates.
9. Screen-informed grid reduction for the GPU runs: `similarity` and
   `feature_coverage` are the only mechanisms that clear random-k on the screen;
   `counter_spurious` is below chance on all four. Carry `similarity`,
   `feature_coverage`, `importance_weighted`, `random` × {`balanced`,
   `target_prior`, `free`} and keep `counter_spurious` as a single documented
   negative control rather than a full row of the grid.

---

## 6. Limitations

- The surrogate readers are order-invariant, so the screen says nothing about
  ordering, serialisation, or prompt format.
- `logreg` on 8 rows with fixed C=0.5 is one point in a hyperparameter space; it
  is held constant across cells so the comparison is between demo sets, but it is
  not a bound on what an LLM could extract.
- Feature selection is the top 12 by mutual information (Notebook 01's choice);
  protocol rankings may shift with a different feature budget.
- 60k-row subsamples per split, k=8, `id_pool` provenance only. k=16 and
  `matched_pool` are not screened.
- Shift estimates use the *full* target input sample. A deployed system with only
  a small unlabelled target batch would have noisier w(x) and π̂_T; the screen
  does not quantify that sensitivity.
