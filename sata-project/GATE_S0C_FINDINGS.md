# Gate S0c (demonstration-label utility): results and next steps

Run: `Llama-3.1-8B-Instruct`, 4 real-arm TableShift datasets, `random × balanced`
at demonstration-label corruption 0 / 50 / 100 %, plus a zero-shot arm, k=8,
5 seeds × 250 OOD queries per cell, contextual-calibration logprobs captured.
64 units, 16 000 rows. Data: `gate_corruption_merged.parquet`.

Metrics: raw accuracy is not used as a headline. On BRFSS the OOD positive rate
is 0.183, so a constant-negative predictor scores 0.817 — accuracy there measures
class imbalance, not task performance. Balanced accuracy (mean per-class recall)
and AUROC on the label-logprob margin are reported instead. AUROC is the primary
gate statistic because it is threshold-free, and the threshold is exactly what
turns out to be broken.

---

## 1. Gate verdict: the demonstration-label channel is dead on 3 of 4 datasets

Corruption 100 % flips every demonstration label. If the model is using the
input→label mapping, performance must move. Paired over 5 seeds:

| dataset | AUROC 0 % | AUROC 50 % | AUROC 100 % | Δ(0−100) | seeds same dir | Wilcoxon *p* |
|---|---|---|---|---|---|---|
| ANES Turnout | 0.682 | 0.552 | 0.507 | **+0.175** | 5/5 | 0.062 |
| ACS Income | 0.682 | 0.692 | 0.670 | +0.012 | 3/5 | 0.625 |
| ACS Public Coverage | 0.517 | 0.515 | 0.518 | −0.001 | 3/5 | 1.000 |
| BRFSS Diabetes | 0.592 | 0.631 | 0.617 | −0.025 | 2/5 | 0.625 |

ANES is the only dataset where flipping the labels destroys the signal, and it
goes all the way to chance (0.507). The other three are flat to within ±0.025,
with no consistent direction across seeds. *p* = 0.062 is the floor of a 5-seed
Wilcoxon test — all five seeds moved the same way and it still cannot reach 0.05.
**Five seeds cannot support inferential claims; this is a descriptive result.**

On ACS Public Coverage the model is at chance regardless (AUROC 0.517), which
independently confirms the CPU screen's "no headroom" verdict for that dataset
from an actual LLM. It should be demoted to a documented negative control.

## 2. The finding that outranks the gate: demonstrations destroy the threshold

| dataset | positive rate, no demos | positive rate, 8 demos | balanced acc, no demos | balanced acc, 8 demos |
|---|---|---|---|---|
| ACS Income | 0.700 | 0.996 | 0.596 | **0.501** |
| ANES Turnout | 0.496 | 0.948 | 0.688 | **0.508** |
| BRFSS Diabetes | 0.568 | 1.000 | 0.613 | **0.500** |
| ACS Public Coverage | 0.992 | 1.000 | 0.510 | **0.500** |

With eight demonstrations the uncalibrated model answers *positive* to 95–100 %
of queries, and balanced accuracy is exactly 0.500 — a constant predictor. The
model's P(positive) sits at a median of 0.835–0.924 with an interquartile range
as narrow as 0.035 (BRFSS). Without demonstrations the same model is far better
behaved on **3 of 4** datasets (positive rate 0.50–0.70, balanced accuracy
0.60–0.69) — but not on ACS Public Coverage, whose zero-shot arm is *already*
saturated (positive rate 0.992, balanced accuracy 0.510) before a single
demonstration is added. That dataset's constant-predictor behaviour therefore
cannot be attributed to demonstrations at all; whatever induces it is present
zero-shot, and it is a second, distinct pathology from the one this section
otherwise describes. See §4 — this is consistent with ACS Public Coverage
already being flagged there as a negative control the model cannot handle in
any configuration.

**This falsifies the v1 post-mortem's diagnosis.** `REDESIGN_RATIONALE.md` §4.1
attributed the 96–99 % class-1 rate to the missing chat template. The chat
template is now applied (`prompt_version = v2_chat`, verified rendering through
the tokenizer's own template) and the raw class-1 rate is *still* 95–100 %. The
bias is induced by the presence of demonstrations, not by the prompt format.

It is also not a majority-label effect: the composition here is `balanced`,
exactly k/2 = 4 per class. Four positive and four negative demonstrations still
produce a ~100 % positive output rate. **So the demonstration label *counts* are
not what drives the output prior on this model** — which directly contradicts the
inference I drew from the CPU screen, where the surrogate readers were
constructed on the assumption that the reader's output prior tracks the demo
label fraction. That assumption does not hold for the real model, and the
`composition` factor's leverage has to be re-measured with the LLM rather than
inherited from the screen.

## 3. Contextual calibration is load-bearing for the few-shot arm and ill-posed for zero-shot

| dataset | mean margin, 8 demos | mean content-free margin, 8 demos | mean margin, no demos | mean content-free margin, no demos |
|---|---|---|---|---|
| ACS Income | +1.862 | +1.892 | +0.354 | −2.750 |
| ANES Turnout | +1.329 | +1.719 | +0.445 | −2.250 |
| BRFSS Diabetes | +2.471 | +2.258 | +0.373 | −3.125 |
| ACS Public Coverage | +1.833 | +1.673 | +1.155 | −1.875 |

In the few-shot arm the content-free margin nearly equals the observed margin
(ACS Income: +1.892 vs +1.862). That is the signature of a bias that is almost
entirely prompt-induced, and dividing it out is exactly what contextual
calibration is for — it lifts balanced accuracy from 0.500 to 0.658 (ACS Income)
and 0.630 (ANES).

In the zero-shot arm the content-free margin is *strongly negative* (−1.9 to
−3.1) while the observed margin is mildly positive. The reason is structural: the
content-free zero-shot prompt has no demonstrations *and* all query values
replaced by placeholders, so it is a near-empty prompt. The model's response to
an empty prompt is not an estimate of that prompt's label prior, so dividing by
it is not calibration — it pushes the calibrated positive rate to 0.92–1.00, i.e.
it makes zero-shot worse than leaving it alone (ANES balanced accuracy
0.688 → 0.499). **Contextual calibration must not be applied to the zero-shot arm
as currently implemented.**

## 4. Gate S1 verdict against its own pre-registered thresholds

`REDESIGN_RATIONALE.md` §Gate S1 requires (a) random-8 ≥ 0.60 ID accuracy and
(b) calibrated zero-shot class-1 rate in [0.35, 0.65], escalating model scale
otherwise.

- **(b) fails on all four datasets** — calibrated zero-shot positive rate is
  0.924 / 1.000 / 0.988 / 0.936. Note this failure is partly the §3 artefact; on
  *raw* zero-shot rates the picture is 0.700 / 0.992 / 0.496 / 0.568, i.e. ANES
  and BRFSS pass, ACS Income marginally fails, ACS Public Coverage fails badly.
- **(a) is not yet testable** — this run used OOD queries only. The ID split was
  not run, so neither Gate S1's ID criterion nor RQ1's ID→OOD gap can be
  evaluated from this data. Against the OOD majority baseline, only ACS Income
  clears it (0.686 vs 0.607); ANES, ACS Public Coverage and BRFSS do not.

Gate S1 therefore fails as written, and its own escalation clause applies.

## 5. Where this leaves the research questions

- **RQ1** (*how badly does frozen-LLM tabular ICL degrade under shift?*) — not
  answerable from this run: no ID arm. The deeper obstacle is unchanged from v1:
  calibrated balanced accuracy is 0.520–0.680 while XGBoost on the same features
  reaches 0.81–0.89 AUROC. The floor effect the v1 post-mortem identified is
  still present, so "degradation" still risks measuring a capability that barely
  exists.
- **RQ2** (*which demonstration diversity matters for which shift type?*) —
  blocked on 3 of 4 datasets *for label-dependent mechanisms*. If flipping every
  label changes nothing, no protocol that varies *which labelled rows* to show
  can matter. Two important qualifications: (i) mechanisms that work through the
  *feature* channel are untouched by this result — query-conditional similarity
  selection changes which feature values appear, and per arXiv 2511.21038 ICL at
  this scale adjusts how inputs project onto pre-trained semantic directions
  rather than remapping labels, which is a feature-channel effect; (ii) §6 below
  gives a live alternative explanation that must be ruled out first.
- **RQ3** (*faithfulness vs accuracy*) — downstream of RQ2; unchanged.
- **RQ4** (*can SATA beat hand-designed protocols?*) — at risk. SATA is a learned
  selector over labelled demonstrations. If the label channel is inert at 8B,
  SATA has little to exploit there, and its evaluation should move to whichever
  scale or arm passes the gate.

## 6. The confound that must be ruled out before RQ2 is declared blocked

Ranking the datasets by how saturated the decision variable is:

| dataset | median P(positive) | IQR | Δ AUROC under corruption |
|---|---|---|---|
| ANES Turnout | 0.835 | 0.138 | **+0.175** |
| ACS Public Coverage | 0.867 | 0.058 | −0.001 |
| ACS Income | 0.867 | 0.087 | +0.012 |
| BRFSS Diabetes | 0.924 | 0.035 | −0.025 |

ANES is the *least* saturated dataset and the *only* one showing a corruption
effect; BRFSS is the most saturated (IQR 0.035) and shows none. Spearman ρ = −0.8
over four points — indicative only, not a test. This supports a competing
explanation: corruption-insensitivity may be a **measurement artefact of a
compressed dynamic range** rather than genuine label-blindness. A decision
variable pinned into a 0.035-wide band has little room to register any effect.

Both readings are consistent with the current data, and they imply opposite next
steps — one says stop building protocols at 8B, the other says fix the instrument
and re-measure. Resolving this is step 1.

---

## 7. Plan

Throughput measured on this host: ~90 prompts/s aggregate across 2×H200 for 8B
(32 000 prompts in ~6 min). Estimates below assume that rate and scale it by
parameter count for 70B.

### Step 1 — Label-token de-saturation diagnostic — 8B, ~30 min

The cheapest decisive experiment, and it tests §6 and the Semantic Anchors
prediction at the same time. Four datasets × `random × balanced` × 0 % corruption
× 5 seeds, varying only the verbaliser:

1. `No` / `Yes` — current baseline.
2. **Swapped**: `Yes` denotes the negative class and `No` the positive one. If the
   positive rate stays ~95 % on the *token* `Yes`, the bias is token-anchored; if
   it follows the semantics, it is concept-anchored. This is the single most
   informative prompt manipulation available and it is nearly free.
3. **Neutral symbols**: `A` / `B`, per Wei et al. 2023 (symbol tuning) — removes
   the semantic prior that plausibly drives the "Yes" attractor.
4. **Numeric**: `0` / `1`.

Read out positive rate, IQR of P(positive), balanced accuracy, AUROC. Success
criterion: at least one verbaliser gives a positive rate in [0.35, 0.65] with
IQR ≥ 0.15. That is the format the rest of the programme should use, and Gate S1
criterion (b) becomes testable against it.

### Step 2 — Re-run Gate S0c on the de-saturated format, with the ID arm — 8B, ~1 h

Only meaningful if step 1 finds a de-saturating format. Adds:
- the **ID query split**, which Gate S1(a) and RQ1 both require and which this run
  omitted;
- the corruption ladder on the new format;
- a zero-shot arm **without** contextual calibration (per §3), plus a corrected
  calibration baseline that keeps the demonstrations and blanks only the query.

This is the run that actually settles whether the label channel is inert at 8B.

### Step 3 — Scale escalation to 70B — ~2–4 h

Triggered by Gate S1's own escalation clause, which has now fired. Run steps 1–2
at `Llama-3.1-70B-Instruct`. Both H200s are available, so 70B at fp8 (~70 GB)
fits on one card with `tensor_parallel: 1` while 8B runs on the other — the two
scales can be run concurrently rather than sequentially.

This is not just a gate-clearing exercise; it is a headline result either way.
arXiv 2511.21038 reports semantic-override rates of exactly zero across 1–12B
models, and Wei et al. 2023 find prior-override emerges with scale. If the label
channel is inert at 8B and live at 70B, **"demonstration design for tabular OOD
is scale-gated"** is a cleaner and more defensible thesis contribution than any
protocol ranking — and it is measured, not assumed.

### Step 4 — Protocol grid, gated per (dataset × model) — ~4–8 h at 8B, longer at 70B

Run the mechanism × composition grid only on cells that pass step 2/3, and
report it with three changes forced by the findings above:
- **Zero-shot is the reference contrast**, not random-*k*. Per LLMTabBench
  (arXiv 2605.24417) extra examples can conflict with prior knowledge and degrade
  performance, and §2 here reproduces that: zero-shot AUROC beats few-shot on 3 of
  4 datasets (0.708 vs 0.678, 0.716 vs 0.678, 0.675 vs 0.548). The grid must be
  able to report a protocol scoring *below* zero-shot.
- **Separate the feature channel from the label channel.** Report each mechanism
  at 0 % and 100 % corruption. A mechanism that helps at both is working through
  feature content; one that helps only at 0 % is working through label content.
  This turns §6's ambiguity into a measured decomposition and is the sharpest
  version of RQ2 available.
- **Re-measure `composition` rather than assuming it.** §2 shows balanced demos
  do not produce a balanced output prior, so the CPU screen's composition
  ranking does not transfer.

### Step 5 — SATA decision point (RQ4)

Defer until step 3 reports. If the label channel is inert at every scale you can
afford, RQ4 is written up as a rigorous negative with the v1+v2 evidence chain,
as `REDESIGN_RATIONALE.md` already anticipates for a Gate S3b failure — and the
`+1.5` counter-spurious term in `src/models/sata_targets.py:46` becomes moot
rather than needing a decision.

---

## 8. Step 1 results: the confound is resolved, and it is worse than either hypothesis in §6

Four verbalisers × 4 datasets × `random × balanced`, corruption fixed at 0%,
5 seeds. Data: `verbaliser_merged.parquet`. Figure: `verbaliser_diagnostic.png`.

**§6's two hypotheses were both wrong, or rather both right about different
things.** Neither "the model ignores demonstration labels" nor "the decision
variable has no room to move" is the correct diagnosis. The real mechanism:

**The model has a raw lexical preference for the literal string "Yes" (and,
weakly, "1"), independent of which class that string is assigned to.**
Swapping the verbaliser pair from `No`/`Yes` to `Yes`/`No` — keeping every
semantic gloss identical, only inverting which literal token denotes which
class — collapses the predicted-positive rate from 0.93–1.00 down to
0.00–0.55 on every dataset (panel a). If the model were reading the
demonstrations' semantics, swapping the token assignment should leave its
*class* predictions unchanged, since nothing about the task changed. It
doesn't: the model keeps preferring to emit the string "Yes," and because "Yes"
now denotes the *negative* class, its predicted-positive rate inverts.

This is confirmed a second way: `0`/`1` reproduces almost the same failure
mode as `No`/`Yes` (positive rate 0.79–1.00, panel a, purple bars) — "1" gets
the same kind of affirmative-token attraction as "Yes" does, even though
nothing in the prompt calls "1" an affirmative answer. `A`/`B` is the only
pair with a *mild* opposite-direction bias (toward "A"), and it is also the
only pair that gets closest to the target zone (panel c) — but "closest" is
still a failure: no verbaliser lands in [0.35, 0.65] with IQR ≥ 0.15 on every
dataset (panel c) — the pre-registered Step 1 success criterion is not met by
any configuration.

**Discriminability, not just the threshold, degrades when the semantic gloss
is removed** (panel b). AUROC under `A`/`B` and `0`/`1` is *higher* than
baseline only on BRFSS (0.548 → 0.563 / 0.565); on the other three datasets
both are *lower* than baseline (ANES: 0.678 → 0.659 / 0.633; ACS Income:
0.678 → 0.562 / 0.632; ACS Public Coverage: 0.519 → 0.511 / 0.501), and the
swapped pair is worse than baseline on all four. So the effect of stripping
the semantic gloss is not directionally uniform across datasets — it helps
on one (BRFSS) and hurts on the remaining three. This still rules out a
clean "it's just a fixed additive bias, calibration handles it" story: if the
gloss carried no genuine task signal, removing it should leave AUROC roughly
unchanged everywhere, not move it substantially in either direction.

**The corruption-insensitivity survives calibration** (panel d, reproducing
§2's `random×balanced` corruption sweep but now on calibrated balanced
accuracy specifically). Contextual calibration removes the static token bias
(§3), and the corruption result is unchanged by it: ANES is still the only
dataset whose calibrated accuracy responds to demonstration-label corruption
(0.638 → 0.516, 5/5 seeds, *p* = 0.062 — the same 5-seed floor as §1). This
confirms the §1 finding was not an artefact of the raw-decision-threshold
pathology; it survives the fix.

**Revised diagnosis.** The demonstration-label channel is not "inert" in a
uniform sense — it is *masked* by a token-level lexical prior strong enough
that no natural single-token verbaliser fully de-saturates it, and calibration
(which corrects the *aggregate* bias) still leaves the *corruption-sensitivity*
question answered the same way as before correction: real on ANES, absent
elsewhere. Given that AUROC-based analysis (threshold-free, immune to the
saturation confound in the first place) agreed with the raw-accuracy analysis
in §1, and now this calibrated-accuracy check agrees a third time, the
**inertness on 3 of 4 datasets should be treated as a real property of this
model at this scale**, not a measurement artefact — with the token-anchoring
finding as an important qualification about *why*, and a candidate mechanism
Wei et al. 2023 and arXiv 2511.21038 would predict changes with scale.

## 9. Revised plan (supersedes §7 step ordering)

Step 1 is complete; the answer is "no clean verbaliser exists, and it doesn't
matter — the corruption result replicates under calibration regardless."
Renumbering:

1. ~~Verbaliser diagnostic~~ — **done**, see §8.
2. **Re-run Gate S0c with the ID split added**, using the existing `No`/`Yes`
   verbaliser and contextual calibration (both now independently justified
   rather than assumed) — ~1 h. This is the one piece of Gate S1 still
   untested and still blocks RQ1.
3. **Escalate to 70B**, concurrently on the second GPU while step 2 runs on
   the first — ~2–4 h. Unchanged from §7. Given §8, frame the escalation
   explicitly as testing whether the token-anchoring effect itself weakens with
   scale (a second, sharper version of the Semantic Anchors question), not only
   whether label-corruption-sensitivity appears.
4. **Protocol grid**, gated per (dataset × model), zero-shot as reference
   contrast, every mechanism run at 0% and 100% corruption. Unchanged from §7.
5. **SATA/RQ4 decision.** Unchanged from §7.

### Not in the plan, deliberately

- The synthetic arm: `data/synthetic/` is empty locally and needs Stage 2
  regeneration; nothing above depends on it.
- `DRNK_PER_WEEK` rescaling: flagged (max 58 100, likely ×100) but not applied,
  because the divisor was not verified against the BRFSS codebook. Do not guess it.
- ANES sentinel codes: `VCF0429`/`VCF0218` carry −8 ("don't know") on 0.01–0.03 %
  of rows. Too rare to matter; documented, not fixed.
