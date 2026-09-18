# Can TableShift support shift-type claims? An audit

Prompted by the concern that TableShift is being used improperly because it does
not "have other shift types in support". **The concern is substantially correct**,
though for a sharper reason than the summary that prompted it gives, and it
affects the project's *motivating* claims rather than its *empirical* ones.

Source: Gardner, Popović & Schmidt, "Benchmarking Distribution Shift in Tabular
Data with TableShift", **NeurIPS 2023** (arXiv:2312.07577v2), §2 and Appendix E.2.

## 1. Two corrections to the summary that prompted this

- It cites the paper as **NeurIPS 2026**. It is NeurIPS 2023 (Datasets &
  Benchmarks track).
- It lists **two** shift types (feature/covariate and label). The paper's §2
  names **three**: covariate shift (`p(x)` changes), label shift (`p(y)`
  changes), and **concept shift** (`p(y|x)` changes).

So "TableShift only has two shift types" is not right as stated. The real problem
is different, and worse.

## 2. The real problem: TableShift names three shift types but isolates none

Three facts from the paper, in its own terms:

1. **Shifts are acknowledged mixtures.** §2: real-world distribution shifts are
   "composed of an unknown mixture of all three forms of shift", and the paper
   adds a footnote conceding this is "a slight abuse of the terminology".
2. **Disentangling them is explicitly out of scope.** §2: "disentangling the
   effects of these forms of shift is not a focus of the current work."
3. **The splits are defined by domain variables, not shift types.** Each task's
   OOD split is cut on a real-world variable — geography, year, demographic
   attribute — chosen because it *produces* a shift gap, not because it produces
   a shift of a particular type.

**And the decisive detail.** TableShift's own "concept shift" metric `Δ_y|x`
(Appendix E.2) is the **Frechet Dataset Distance between the layer activations of
a trained classifier** on the ID and OOD splits. It is computed from feature
vectors only — `y` never enters it. It is therefore not a measure of `p(y|x)`
change at all; it is covariate shift measured in representation space. The paper's
own Figure 7 confirms this empirically: across the 15 tasks,

| pair | reported ρ |
|---|---|
| `Δ_x` (covariate) vs `Δ_y|x` ("concept") | **+0.99** |
| `Δ_x` vs `Δ_y` (label) | −0.20 |
| `Δ_y|x` vs `Δ_y` | −0.20 |

A correlation of 0.99 means the benchmark has **one** covariate-shift measure
reported twice under two names, plus a genuinely independent label-shift measure.
There is no independent quantification of concept shift anywhere in TableShift.

Consistently, Figure 8 reports that **only label shift predicts the shift gap**
(ρ = 0.73), and Figure 1 that ID accuracy together with `Δ_y` explains
R² = 0.993 of OOD accuracy. TableShift is, empirically, a **label-shift**
benchmark with covariate shift along for the ride.

### Verdict

Any claim of the form *"protocol X helps under covariate shift but hurts under
concept shift"* **cannot be tested on TableShift.** The benchmark does not label,
isolate, or independently measure the contrast. With n=4 datasets it would have no
power to support such a claim even if it did.

## 3. What this does and does not invalidate

### Not affected — every empirical result

All results in `GATE_S0C_FINDINGS.md` are reported **per dataset** and were never
aggregated or compared by shift type. The feature-channel result (§11), the
label-channel inertness (§10.4), the absent ID→OOD gap (§10.3), the scale-dependent
calibration inversion (§10.1) — none of these is a shift-type claim. They stand.

### Affected — the motivating rationale

`PROTOCOL_SPEC.md` motivates the mechanism set with reasoning of the form
"importance-weighted selection helps under covariate shift but can actively hurt
under concept shift". That is a **hypothesis this benchmark cannot evaluate**. It
should be reframed as such, and tested — if at all — on the synthetic arm
(`04_task_generator`, `06_synthetic_evaluation`), where shift type is a
controllable generative parameter. This is the constructive answer: the project
already has the right instrument for shift-type claims, and it is not TableShift.

### Affected — the shift taxonomy labels

Notebook 09 labels datasets "covariate-shift dominant", "prior/concept shift with
no covariate signal", etc. Those are **our own estimates**, not benchmark ground
truth, and they are computed on a **top-12 mutual-information feature subset**
rather than the full feature space. They should be presented as such. Our
independent measurements only partly track TableShift's:

| dataset | TS `Δ_x` | our domain AUC | TS `Δ_y` | our prior shift | TS `Δ_y\|x` | our BBSE identifiable | supervised acc drop |
|---|---|---|---|---|---|---|---|
| `acsincome` | 30.603 | 0.915 | 0.006 | 0.076 | 1.395 | yes | 0.011 |
| `anes` | 13.602 | 0.557 | 0.002 | 0.079 | 2.229 | yes | 0.046 |
| `brfss_diabetes` | 12.280 | 0.668 | 0.033 | 0.049 | 0.104 | yes | 0.044 |
| `acspubcov` | 5.793 | 0.808 | 0.170 | 0.414 | 4.058 | **no** | 0.189 |

Covariate rankings disagree (TableShift puts `acspubcov` lowest on `Δ_x`; our
discriminator puts `anes` lowest) — expected, since the metrics differ and ours
runs on 12 features. **Label shift agrees strongly**: `acspubcov` is the extreme
on `Δ_y`, on our prior-shift estimate, and on the supervised accuracy drop, and it
is the only dataset where our BBSE identifiability gate fires.

That gate failure is worth noting as a methodological point in our favour: it
detects non-invariance of `p(x|y)`, which is *actual* evidence about `p(y|x)`
structure — something TableShift's mislabelled `Δ_y|x` does not provide.

## 4. Two things this makes stronger

**The absent ID→OOD gap is a stronger result than reported.** TableShift's task
selection criteria (§3.1) include: *"we explicitly select datasets where strong
hyperparameter-tuned tabular baselines display a statistically significant shift
gap (Δ_Acc ≠ 0)"*. The datasets were **chosen to guarantee a measurable gap for
supervised tabular models**. Finding that LLM ICL shows no such gap on
benchmark-by-construction-gapped data is a sharper finding than finding it on
arbitrary data — and our own supervised baseline reproduces the intended gap
(`acspubcov` 0.797→0.608), confirming the split works as designed and that the
absence is specific to ICL.

**The calibration diagnosis converges with the benchmark's own headline.**
TableShift finds label shift is what drives the shift gap; we find ICL's OOD
failure is a threshold/calibration failure rather than a representation failure
(AUROC flat, balanced accuracy collapsing). Label shift breaking a decision
threshold is the same phenomenon seen from two directions.

## 5. One genuine protocol departure to disclose

TableShift states its assumed setting explicitly (§2): *"we assume that no
information about the target `D^test` is available"*. Our
`src/selection/shift_estimation.py` uses **unlabelled target inputs** to fit the
domain discriminator and the BBSE prior. That is a standard and defensible
transductive / unsupervised-domain-adaptation setting, and it is deliberately
label-free — but it is **not** TableShift's assumed protocol, and results using it
are not directly comparable to the benchmark's published numbers. This should be
stated plainly in the methods rather than left implicit.

## 6. Recommendation

1. Keep TableShift, and describe it accurately: a **label-shift-dominant** tabular
   benchmark with mixed, type-unlabelled real-world shifts, selected for
   supervised shift gaps.
2. Drop shift-type contrasts from the real-arm claims. Move them to the synthetic
   arm, where shift type is controllable, or drop them.
3. Relabel notebook 09's taxonomy as estimates on a 12-feature subset.
4. Disclose the unlabelled-target-input departure in the methods.
5. Report the task-selection criterion when presenting the no-gap result — it
   strengthens it.
