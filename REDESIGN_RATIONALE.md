# Why the v1 experiment failed, and why the v2 redesign is justified

This is a full account of the first complete run of the SATA experiment: what we were trying to show, how we built it, what the results were, why the results came out the way they did (traced to specific lines of code), what the literature says about each failure, and exactly what we plan to change. Every claim is linked to a source — a section of the lit review, a results file, or a code block quoted verbatim from the repository.

**Resources referenced throughout:**

- Lit review: [`Lit-review.pdf`](Lit-review.pdf) — citation numbers like [27] refer to its reference list
- Experiment spec: [`SATA_experiment_spec.md`](SATA_experiment_spec.md)
- Results: [`sata-project/results/`](sata-project/results/) (parquets), [`sata-project/figures/`](sata-project/figures/) (PDFs), [`sata-project/tables/`](sata-project/tables/)
- Code: [`sata-project/src/`](sata-project/src/), notebooks in [`sata-project/notebooks/`](sata-project/notebooks/)
- Config actually used: [`sata-project/configs/default.yaml`](sata-project/configs/default.yaml)

---

## 1. What we are targeting and why

The central hypothesis (lit review §3, p. 11–12) is that **out-of-distribution generalisation in tabular in-context learning improves when the demonstrations span a richer coverage of decision boundaries, feature interactions, and subpopulation structure** — and that a small meta-learned adapter (SATA) which selects and reweights demonstrations *conditioned on the query* can push both accuracy and decision faithfulness beyond what hand-designed diversity protocols achieve.

The reason this hypothesis is theoretically sound comes from four converging strands in the literature review:

1. **Shortcut learning explains OOD failure, and it is architecture-agnostic.** Geirhos et al. [9] establish that models fail under shift because learning systems extract whatever structure in the data is cheapest to fit, not the structure that is causally right. Nagarajan et al. [10] show the mechanism formally for ERM: geometric and statistical skews cause spurious features to be absorbed into the learned solution *even when a fully invariant solution exists*. The failure comes from the training signal, not the architecture — which means an intervention on the signal (here: the demonstrations in context) is a legitimate lever.

2. **ICL is a learned behaviour governed by the content and structure of the context.** Xie et al. [14] frame ICL as implicit Bayesian inference: the model treats the prompt as evidence about which latent concept generated the demonstrations. Chan et al. [15] show ICL only emerges when the data has particular distributional structure. If context structure governs what the model infers, then *context design* is a control knob.

3. **A diversity threshold governs OOD generalisation in ICL.** Goddard et al. [17] demonstrate a phase transition: below a critical pretraining-task diversity, ICL produces specialised solutions that fail out of task distribution; above it, the same architecture generalises. Our project asks whether an analogous diversity lever exists **at inference time**, through the choice of demonstrations, without touching the frozen model.

4. **ICL is itself vulnerable to in-context spurious correlations.** Harutyunyan et al. [28] and Chen et al. [30] show in-context learners pick up spurious correlations *present in the demonstration set*. This cuts both ways: it is the threat the protocols defend against, and it is the mechanism SATA is supposed to exploit deliberately (select demos that break the shortcut).

This yields the four research questions (lit review §3):

- **RQ1** — how badly does frozen-LLM tabular ICL degrade under controlled distribution shift?
- **RQ2** — which kinds of demonstration diversity matter for which shift types?
- **RQ3** — do configurations that improve OOD accuracy also improve global decision faithfulness, or do the two dissociate?
- **RQ4** — can a small query-conditioned meta-learned adapter (SATA) induce a qualitative improvement over hand-designed protocols?

---

## 2. What we set up and how we approached it

The design ([`SATA_experiment_spec.md`](SATA_experiment_spec.md)) has two parallel arms.

**Real arm** (notebooks [`01`](sata-project/notebooks/01_tableshift_setup.ipynb)–[`03`](sata-project/notebooks/03_faithfulness_real.ipynb)): four TableShift [6] datasets (ACS Income, ACS Public Coverage, BRFSS Diabetes, ANES), a fixed 256-row demo pool per dataset, 500 ID + 500 OOD test rows, two frozen LLMs (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct) served by vLLM, and seven demonstration-selection conditions: zero-shot, random-k, similarity-k, and the four diversity protocols (label, feature-range, rule, counter-spurious). Faithfulness is measured as Spearman ρ between the model's self-reported feature ranking (π_self) and a behavioural ranking obtained by hot-deck leave-one-out ablation (π_behav) — the STaDS-style global faithfulness measure [8].

**Synthetic arm** (notebooks [`04`](sata-project/notebooks/04_task_generator.ipynb)–[`06`](sata-project/notebooks/06_synthetic_evaluation.ipynb)): a generator of tabular tasks with 10 features, of which 3–5 are causal under one of four rule families (linear, threshold, tree, sparse_interaction), feature 8 is a planted spurious correlate of the label, and feature 9 is pure noise. Each task is instantiated under six environments: `id`, `covariate`, `spurious_reversal`, `extrapolation`, `missing_feature`, `mechanism`. Because the ground truth is known, we can (a) supervise SATA's demonstration scoring, and (b) measure whether the LLM relies on the *correct* features (π_true vs π_behav).

SATA itself ([`src/models/sata.py`](sata-project/src/models/sata.py)) is a 4-layer transformer encoder (d_model=128) that takes the demo pool (features + labels) and the query (features only), attends over all of them jointly, and outputs a softmax relevance score per demo. It is trained with a KL loss against target distributions computed from generator ground truth, then used at inference to pick the top-k demos for the prompt.

The evaluation prompt is assembled by [`src/inference/prompts.py`](sata-project/src/inference/prompts.py):

```python
SYSTEM_TEMPLATE = (
    "You are a classifier. Given the features of an individual, predict {task_description}.\n"
    "Respond with exactly one word: {label_0} or {label_1}."
)

def build_classification_prompt(task_description, label_tokens, demo_lines, query_line, label_meanings=None):
    ...
    body = "\n".join(demo_lines)
    return f"{system}\n\n{body}\n\n{query_line}"
```

and each row is serialised by [`src/data/serialisation.py`](sata-project/src/data/serialisation.py) into lines like `feature_0: -0.43; feature_1: 1.27; ... -> 1`. Inference ([`src/inference/llm_runner.py`](sata-project/src/inference/llm_runner.py)) is greedy single-token decoding with the logits masked to the two label-token ids, and confidence computed by a constrained two-way softmax:

```python
def get_confidence(logprobs_dict, label_tokens):
    lp0 = logprobs_dict.get(label_tokens[0], -100)
    lp1 = logprobs_dict.get(label_tokens[1], -100)
    p0 = np.exp(lp0) / (np.exp(lp0) + np.exp(lp1))
    ...
```

---

## 3. What the results were

Everything below is reproducible from the parquets in [`sata-project/results/`](sata-project/results/).

### RQ1: the OOD degradation we assumed largely did not appear

From [`real_arm_baselines_summary.parquet`](sata-project/results/real_arm_baselines_summary.parquet) (figure: [`rq1_ood_degradation.pdf`](sata-project/figures/rq1_ood_degradation.pdf)): ID−OOD accuracy gaps for the few-shot methods are small and **mostly negative** — OOD accuracy is as high as or higher than ID accuracy in the majority of (dataset × model × method) cells. The only large positive gap is zero-shot Llama on ACS Public Coverage (0.794 ID vs 0.402 OOD), a condition that is unstable in general. Absolute accuracy sits at 0.50–0.73 everywhere, with some cells *below chance* (Llama random-8 on ACS Public Coverage ID: 0.337).

The interpretation matters: TableShift's published shift gaps [6] were measured on **trained** models (XGBoost and friends) that first reach high ID accuracy and then lose it under shift. A frozen 7–8B LLM given 8 serialised rows never acquires much task competence in the first place, so there is nothing for the shift to take away. RQ1's honest answer in v1 is "the premise doesn't manifest at this competence level" — a floor effect, not a robustness result.

### RQ2: no protocol × shift interaction, and the wrong protocol won everything

From [`rq2_grid.parquet`](sata-project/results/rq2_grid.parquet) (figure: [`rq2_protocol_shift_heatmap.pdf`](sata-project/figures/rq2_protocol_shift_heatmap.pdf)), accuracy by method × shift type:

| method (Qwen2.5-7B) | covariate | extrap | id | mechanism | missing | spur_rev |
|---|---|---|---|---|---|---|
| label_diversity | **0.618** | **0.650** | **0.680** | **0.654** | **0.633** | **0.591** |
| rule_diversity | 0.579 | 0.647 | 0.664 | 0.588 | 0.592 | 0.563 |
| random | 0.495 | 0.620 | 0.635 | 0.555 | 0.585 | 0.547 |
| counter_spurious | 0.454 | 0.638 | 0.630 | 0.579 | 0.538 | 0.564 |
| best_protocol_sata | 0.296 | 0.333 | 0.393 | 0.360 | 0.346 | **0.065** |
| sata_alone | 0.294 | 0.479 | 0.575 | 0.515 | 0.352 | **0.045** |

The pre-registered success condition for RQ2 was a *crossover*: feature-range should win covariate shift, counter-spurious should win spurious reversal. Instead **label_diversity wins every column on both models**, and counter-spurious does not win the one environment it was purpose-built for. Note also the overall level: the best cell in the entire grid is 0.680 on a binary task, while XGBoost trained on the *same 64-demo pools* exceeds 0.80 (notebook 04 gate). Most methods sit at ~0.5 — chance — in the `id` column for Llama.

One number explains the label_diversity dominance. From [`synthetic_evaluation.parquet`](sata-project/results/synthetic_evaluation.parquet), the zero-shot rows:

```
zero-shot, per environment:   P(label = 1)      P(prediction = 1)
covariate                        0.388               0.964
extrapolation                    0.506               0.993
id                               0.492               0.990
mechanism                        0.449               0.988
missing_feature                  0.468               0.962
spurious_reversal                0.492               0.994
```

With no demonstrations, both models predict class "1" **96–99% of the time regardless of input**. The single most valuable thing a demo set can do in this regime is repair the label prior — and label_diversity is the only selector in the codebase that enforces class balance ([`src/selection/label_diversity.py`](sata-project/src/selection/label_diversity.py)). Its victory is a prior-correction effect, not evidence about shift-targeted coverage.

### RQ3: too noisy to support a claim

From [`faithfulness_real_rho_summary.parquet`](sata-project/results/faithfulness_real_rho_summary.parquet) (figure: [`rq3_accuracy_vs_faithfulness.pdf`](sata-project/figures/rq3_accuracy_vs_faithfulness.pdf)): ρ(π_self, π_behav) ranges from −0.19 to 0.41 with bootstrap CIs straddling zero in nearly every cell, and the Δaccuracy-vs-Δρ scatter shows no relationship. Technically this satisfies RQ3's success criterion ("configurations exist where accuracy improves but ρ does not"), but a Spearman correlation over ~10–12 ranked features has enormous sampling variance, so the defensible claim is only the descriptive one: faithfulness is low everywhere and uncorrelated with accuracy gains — consistent with STaDS [8], but not a sharp result.

### RQ4: SATA is the worst method in the grid, and catastrophically so under spurious reversal

The four headline facts, with sources:

| # | Result | Value | Source |
|---|---|---|---|
| 1 | Gate 2 proxy accuracy: SATA vs best protocol | **0.961** vs 0.724 (label_diversity) | [`sata_gate2_summary.parquet`](sata-project/results/sata_gate2_summary.parquet) |
| 2 | SATA-based methods on spurious_reversal (with LLM) | **0.045–0.099** | [`rq2_grid.parquet`](sata-project/results/rq2_grid.parquet) |
| 3 | Query-agnostic ablation on the same cell | 0.494–0.558 | [`tables/sata_ablations.csv`](sata-project/tables/sata_ablations.csv) |
| 4 | Best training checkpoint | **epoch 0**; val-proxy falls 0.958 → 0.924 while KL loss falls 0.298 → 0.184 | [`sata_training_log.parquet`](sata-project/results/sata_training_log.parquet) |

Fact 2 deserves unpacking, because it is not "SATA failed", it is "SATA anti-succeeded". In the `spurious_reversal` environment for Qwen with `sata_alone`, the raw predictions look like this (from `synthetic_evaluation.parquet`):

```
predictions:  {'1': 6414, '0': 6386}    # balanced
labels:       {'0': 6498, '1': 6302}    # balanced
accuracy:     0.063
```

Balanced predictions against balanced labels agreeing only 6.3% of the time means the model *disagrees with the truth 94% of the time* — it is tracking something almost perfectly, and that something is anti-correlated with the label. The only variable in the environment with that property is the spurious feature 8, whose agreement with the label under reversal is 1 − 0.96…0.995 ≈ 0.005–0.04. The accuracy of SATA-based methods lands almost exactly on that number. The LLM, fed SATA's selections, is reading feature 8 and nothing else.

Fact 3 inverts the design hypothesis outright: removing SATA's ability to see the query (the query-agnostic ablation) *improves* it from 0.045 to 0.558 on Qwen. Per-query conditioning — the core differentiator SATA was built around — is precisely the thing doing the damage.

There is also one cell that is statistically impossible for any genuine predictor: label_diversity on held-out-family × missing_feature scores **0.000 accuracy over 1600 predictions** ([`rq4_comparison.parquet`](sata-project/results/rq4_comparison.parquet)). Zero out of 1600 on a binary task cannot be model behaviour; it means the environment itself is degenerate (§4.4 shows it is: every label in that cell is 0, and the "1"-biased model predicts 1 every time).

These facts — an anti-prediction pattern, an impossible cell, and a proxy gate that disagreed with the downstream result by 0.9 accuracy points — pointed at implementation defects rather than at the hypothesis. The code confirms it.

---

## 4. The diagnosis: five defects, each traced to code

### 4.1 The instruct models never received their chat template, and the prompt gave them nothing to work with

**What the code does.** [`src/inference/llm_runner.py`](sata-project/src/inference/llm_runner.py) sends raw prompt strings straight into vLLM:

```python
sampling_params = SamplingParams(**sp_kwargs)      # logprobs=20, max_tokens=1, temperature=0
outputs = self.llm.generate(prompts, sampling_params)
```

`apply_chat_template` appears **nowhere in the codebase** (grep-confirmed across `src/` and `notebooks/`). Both configured models are `-Instruct` checkpoints: they were post-trained to operate inside a chat structure (`<|start_header_id|>system<|end_header_id|>…` for Llama, ChatML for Qwen), and we ran them in raw-completion mode, a format they were never optimised for. On top of that, the synthetic arm's system prompt is semantically empty — notebook 06 passes `task_description = "the label of a synthetic binary classification task"`, so the model is told, in effect, "predict the label of a task" with rows of anonymous numbers and no statement that a *rule over the features* exists and should be induced from the examples.

**What it caused.** The 96–99% class-1 prior shown in §3 (RQ2 table above). Every accuracy number in both arms is contaminated by this bias, and every method comparison is dominated by whether that method's demo set happens to counteract it.

**What theory says.** Min et al. [27] — already central to the lit review (§2.6) — show that the label space, the input distribution, and the **format** of demonstrations drive ICL far more than the input–label mapping itself. Zhao et al. (2021, *Calibrate Before Use*; not yet in the lit review — needs adding) document exactly this failure: LLM classifiers exhibit majority-label and format biases large enough to swamp the signal, and they are correctable by estimating the model's output on a content-free input and dividing it out. We measured a textbook case of that bias and did nothing to control it.

### 4.2 SATA learned the exact shortcut it was designed to fight

This is the central finding of v1, and it takes three code facts together.

**Fact A — the target function leaks the query's true label.** [`src/models/sata_targets.py`](sata-project/src/models/sata_targets.py), the complete scoring loop:

```python
for i, demo in enumerate(demo_metadata):
    score = 0.0

    # Same regime bonus
    if demo["regime"] == query_metadata["regime"]:
        score += 2.0

    # Counter-spurious bonus
    if demo["is_counter_spurious"]:
        score += 1.5

    # Correct label bonus (mild)
    # Not too strong — we want structural alignment, not just label matching
    if demo["label"] == query_metadata["label"]:
        score += 0.5

    # Penalty for spurious-only demos
    if demo["spurious_consistent"] and demo["regime"] != query_metadata["regime"]:
        score += 0.1  # near-zero but not exactly zero for numerical stability

    scores[i] = score

scores = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature))
```

Two problems live in this function. First, `query_metadata["label"]` is the query's **ground-truth label** — information SATA cannot have at inference time. Whenever a target uses hidden information, a model trained on it is incentivised to *reconstruct* that hidden information from whatever inputs it does have. Second, the comment says "Penalty for spurious-only demos" but `score += 0.1` is a **bonus** — there is no negative term anywhere in the function, so nothing in the supervision ever pushes SATA *away* from spurious-consistent demos.

**Fact B — the query's label is trivially reconstructable, because the spurious feature is a near-photocopy of it.** [`src/data/generator.py`](sata-project/src/data/generator.py), the spurious-feature construction:

```python
# Spurious feature: agrees with label w.p. spurious_strength.
agree = rng.random(n_samples) < spurious_strength
X[:, self.spurious_idx] = np.where(agree, y_clean, 1 - y_clean).astype(float) \
                          + rng.normal(scale=0.1, size=n_samples)
```

Feature 8 is literally the 0/1 label plus σ=0.1 Gaussian noise. And [`configs/default.yaml`](sata-project/configs/default.yaml) line 66 sets:

```yaml
spurious_strength_range: [0.96, 0.995]
```

— silently recalibrated up from the spec's intended `[0.70, 0.95]` ([`SATA_experiment_spec.md`](SATA_experiment_spec.md), Week-0 decisions). So SATA's cheapest path to reducing its KL loss was never "learn regime structure"; it was **"read the query's label off feature 8, then upweight demos whose given label matches"** — remember SATA receives every demo's label as an explicit input ([`src/models/sata.py`](sata-project/src/models/sata.py): `demo_emb = feature_embed(demo_features) + label_embed(demo_labels)`).

**Fact C — the validation gate rewarded the degenerate policy instead of catching it.** [`src/models/sata_train.py`](sata-project/src/models/sata_train.py): `evaluate_sata_proxy` validates on the `id` environment **only**:

```python
(demo_X, demo_y, demo_meta), (query_X, query_y, query_meta) = _task_batch(
    task, "id", n_demos=config.max_demos, n_queries=32
)
```

and the per-query scorer `_fit_predict_one` contains a fallback that converts a label-homogeneous selection into a free prediction:

```python
if len(np.unique(demo_y_topk)) < 2:
    # XGBoost needs >=2 classes to fit; an early/untrained model's top-k
    # can easily be single-class by chance. Fall back to that class.
    pred = demo_y_topk[0]
```

If SATA selects 8 demos that all carry the (inferred) query label, the proxy doesn't even fit a classifier — it *directly predicts that label* and scores it correct. The label-copy policy therefore achieves proxy accuracy equal to how well feature 8 predicts the label: 0.96–0.995. Gate 2's observed value was **0.9614** ([`sata_gate2_summary.parquet`](sata-project/results/sata_gate2_summary.parquet)) — the shortcut's signature, sitting exactly inside the spurious-strength range.

**The causal chain to the 0.045 accuracy.** At LLM evaluation time under `spurious_reversal`, the generator flips the agreement (`spurious_strength = 1.0 - spurious_strength`), so feature 8 now *anti*-predicts the label at 96–99.5%. SATA, keying on feature 8, confidently infers the **wrong** label for nearly every query and selects demos carrying that wrong label. The LLM — whose behaviour is dominated by in-context label statistics (§4.1, [27]) — copies the majority demo label. Result: accuracy ≈ 1 − spurious_strength ≈ 0.005–0.04, observed 0.045–0.099. The query-agnostic ablation cannot see the query, cannot infer its label, cannot execute the shortcut — which is exactly why it scores 10× higher on this cell. Even the training log corroborates it: the supervision (KL loss) and the goal (proxy accuracy) actively *disagree* from epoch 0 onward — loss falls monotonically while val-proxy falls with it ([`sata_training_log.parquet`](sata-project/results/sata_training_log.parquet)) — which is what training against a mis-specified target looks like.

**What theory says.** This is Geirhos et al.'s [9] shortcut-learning account applied to our own auxiliary model: we handed a learner a feature that made the target cheap to fit without learning the intended structure, and it took the deal — exactly as the ERM analyses in [10] predict. It is also a manufactured instance of [28]/[30]: an in-context learner (the frozen LLM) inheriting a spurious correlation *that our own selection procedure planted in its context*. As a side observation, the base-rate structure made the intended signal nearly unlearnable anyway: `is_counter_spurious` fires on only 0.5–4% of demos in five environments (at strength 0.96–0.995 almost every demo is spurious-consistent) and on 96–99.5% in the sixth — a term that is almost always silent, or almost always constant, carries next to no training gradient either way.

### 4.3 The comparison between selection methods was confounded by demo label balance

**What the code does.** Only one of the eight selectors controls the class composition of the selected demos — [`src/selection/label_diversity.py`](sata-project/src/selection/label_diversity.py) (k/2 per class by construction). Every other method, including SATA's greedy top-k —

```python
top_k_local = np.argsort(-scores)[:k]
return list(candidates.index[top_k_local])
```

([`src/selection/sata_select.py`](sata-project/src/selection/sata_select.py)) — lets the label composition fall out uncontrolled. Given §4.1 (the model's output prior tracks the in-context label distribution) and §4.2 (SATA actively skews toward single-label selections), the methods differ not only in *which* rows they pick but in the *label statistics* they present — and the latter, per [27], is the stronger driver. RQ2's method comparison was therefore never a clean comparison of coverage strategies.

Two further asymmetries compound it:

- **Demo ordering.** Notebook 06 renders demos in selector-return order. For SATA and similarity that is *descending* score — the best demo is placed **first**, i.e. farthest from the query, where recency-sensitive attention values it least (Lu et al. 2022, *Fantastically Ordered Prompts*; also not yet in the lit review). Random and the protocols return effectively shuffled order. The ordering nuisance thus hits precisely the retrieval-based methods.
- **Single seed.** Notebook 06 runs the entire synthetic grid with `seed = config.seed_accuracy[0]` despite the config declaring five seeds — so every synthetic number is a single draw with no variance estimate.

### 4.4 Generator defects corrupted specific cells and confounded a shift axis

All in [`src/data/generator.py`](sata-project/src/data/generator.py).

**(a) The impossible 0.000 cell: `sparse_interaction` × `missing_feature`.** The rule family is:

```python
if self.rule_family == "sparse_interaction":
    i, j = self.causal_features[0], self.causal_features[min(1, len(self.causal_features) - 1)]
    return ((X[:, i] * X[:, j]) > self.threshold).astype(int)
```

`self.threshold` is **never assigned by `_sample_tasks`** — it keeps its default of 0.0 for every task. The `missing_feature` environment does:

```python
elif env_type == "missing_feature":
    drop_idx = self.causal_features[0]
    X[:, drop_idx] = 0.0
```

`drop_idx` is the product's first factor, so the product is identically 0, `0 > 0.0` is `False`, and **`y_clean ≡ 0` for every row** — the only 1-labels come from the 2% label-noise flips. label_diversity then feeds the model balanced demos, the "1"-biased model (§4.1) predicts 1, and the cell scores exactly 0.000 on 1600 predictions. The reason this survived until the final results: the notebook-04 validation gate sampled tasks from `train_tasks`, which **excludes the held-out family** — `sparse_interaction` was never validated at all.

**(b) The covariate environment leaks label shift.** The shift is applied *before* the rule is evaluated:

```python
elif env_type == "covariate":
    shift_feats = rng.choice(self.n_features, size=min(3, self.n_features), replace=False)
    X[:, shift_feats] += rng.uniform(1.0, 2.0, size=len(shift_feats)) * rng.choice([-1, 1], size=len(shift_feats))
```

Adding ±1–2σ offsets to causal features moves the label distribution: the covariate environment's label-1 rate is **0.388** versus ~0.49 elsewhere (§3). Under the shift taxonomy the entire lit review is structured around (§2.1; [1] Moreno-Torres, [3] Storkey), covariate shift means P(x) moves while P(y|x) is preserved — but P(y) moving this much means a "1"-biased predictor fails this environment for a reason that has nothing to do with input-space coverage, which is what the feature-range protocol was supposed to be tested on. Additionally `shift_feats` is drawn from all **10** indices, but features 8 and 9 are overwritten after the shift (spurious and noise construction come later in the function) — so roughly 30% of the shift budget lands on features where it is a silent no-op.

**(c) The extrapolation environment tests nothing.** It is one global scalar:

```python
elif env_type == "extrapolation":
    X *= rng.uniform(2.0, 3.0)
```

For `tree` (labels from sign patterns: `X > 0`) and `sparse_interaction` (product vs threshold 0.0) a positive global scale **cannot change a single label**. And at SATA-inference time the pool-statistics standardisation in `sata_select.standardise` removes a global scale factor entirely. The environment is label-inert for half the rule families and invisible to the selector.

**(d) The demo pools assume oracle access to the shifted distribution.** Both in training —

```python
def _task_batch(task, env_type, n_demos, n_queries, seed=None):
    X, y, metadata = task.generate_environment(env_type, n_samples=n_demos + n_queries, seed=seed)
    demo_X, demo_y, demo_meta = X[:n_demos], y[:n_demos], metadata[:n_demos]
    query_X, query_y, query_meta = X[n_demos:], y[n_demos:], metadata[n_demos:]
```

— and in the saved evaluation data (notebook 04's `save_task_environments`, consumed by notebook 06 as `pool = env_df[env_df['split']=='demo']` inside the per-environment loop), the demonstrations are drawn from the **same shifted environment as the test queries**. Under spurious_reversal, the demos themselves carry the reversed correlation; under extrapolation, the demos are also scaled. This quietly assumes the practitioner has labelled data from the deployment distribution — the one thing that, if you had it, would make "shift-aware selection" unnecessary. The realistic setting — demo pool from the training (ID) distribution, queries from the shifted one — is never tested anywhere in v1.

### 4.5 Train/inference mismatches around SATA's inputs

Training feeds SATA **raw** generator features (`_task_batch` does no standardisation), while inference z-scores them:

```python
def standardise(pool_features, query_features):
    mean = pool_features.mean(axis=0)
    std = pool_features.std(axis=0) + 1e-8
    return (pool_features - mean) / std, (query_features - mean) / std
```

— and the statistics are computed over **the candidate set**, which is the full 64-row pool for `sata_alone` but only the 32 pre-filtered rows for `best_protocol_sata`, so the same physical row arrives at SATA with different values depending on the condition. The mismatch is worst on exactly the feature SATA keys on: feature 8 is {0,1}±0.1 raw (mean ≈ 0.5, std ≈ 0.5) but ~±1 after z-scoring. Finally, small formatting inconsistencies exist at the prompt level — `_format_number` renders `0.5` and `0.50` differently depending on the value (`round(f, ndigits)` drops trailing zeros), a tokenisation nuisance between demos and queries.

---

## 5. The changes we are making, and the justification for each

The governing principle: **every change corrects an identified bug or confound.** No change is a free parameter to tune until the answer comes out positive. Each fix names the defect it corrects and the literature that grounds it, and the plan pre-registers gates so that a failure after the fixes is attributable to the hypothesis rather than the harness.

### 5.1 Fix the measurement instrument (Stage 1)

**Chat templates.** New `src/inference/chat.py` wrapping the tokenizer's own template:

```python
prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": system}, {"role": "user", "content": user}],
    tokenize=False, add_generation_prompt=True,
)
```

The label becomes the assistant's first generated token; the existing `allowed_token_ids` constraint and `get_confidence` carry over unchanged. *Corrects 4.1. Justification: the models' post-training contract; format sensitivity per [27].*

**Semantic task framing.** The synthetic system prompt will state what the task actually is — "each example lists 10 numeric measurements and its category; the category is determined by an unknown rule over some of the measurements; infer the rule from the labelled examples and classify the final one" — while keeping abstract feature names (`feature_0`…) so no real-world priors are injected into the synthetic arm. *Corrects 4.1. Justification: under the Bayesian view of ICL [14], the prompt is the evidence from which the model infers the latent task; v1's prompt did not identify the task as rule induction at all.*

**Contextual calibration, reported alongside raw.** For each prompt, a paired content-free prompt (same demos, query values replaced with "N/A") estimates the model's prior; predictions are debiased by dividing it out (Zhao et al. 2021). Both raw and calibrated results are reported — this is a correction for a *measured* 96–99% prior bias, not a silent change. *Corrects 4.1.*

**Uniform seeded random demo order for every method**, with the order seed recorded per row. *Corrects the ordering asymmetry in 4.3 (Lu et al. 2022). Ordering becomes a controlled nuisance identical across methods, because the hypothesis is about selection, not ordering.*

**Fixed-width number formatting** (`f"{f:.2f}"`), and **all 5 seeds** on the synthetic arm. *Corrects 4.5 and the single-seed problem in 4.3.*

**Gate S1 (pilot, before any large run):** with the fixes above, random-8 must reach ≥ 0.60 ID accuracy on a 20-task pilot (chance is 0.5; XGBoost on the same pools exceeds 0.8), and calibrated zero-shot must have a class-1 rate in [0.35, 0.65]. If the 7–8B models still cannot clear this, the base model escalates (Qwen2.5-32B → 70B-class), because RQ1–RQ4 are unanswerable in a model with no ID competence — you cannot measure robustness of a capability that does not exist. Only models passing the gate proceed.

### 5.2 Fix the testbed (Stage 2)

**Sample `sparse_interaction` thresholds** (from a target base rate U(0.35, 0.65) via a quantile probe of the product distribution) and **replace the missing feature with an independent marginal redraw instead of zeroing it**:

```python
elif env_type == "missing_feature":
    drop_idx = self.causal_features[0]
    X[:, drop_idx] = rng.normal(size=n_samples)   # broken measurement, not a constant
```

*Corrects 4.4(a) — eliminates the constant-label cell; "the feature is uninformative" is the intended semantics of the shift, "the feature is pinned to a constant that annihilates the rule" is not.*

**Covariate shift restricted to features 0–7, with base-rate preservation by rejection sampling** (draw the shift vector, probe the resulting label rate on 2048 synthetic rows, accept only if within ±0.03 of the ID rate). *Corrects 4.4(b). Rejection keeps the environment a pure P(x) intervention, which is what the shift taxonomy ([1], [3]; lit review §2.1) requires "covariate shift" to mean — RQ2's whole premise is that shift types are distinguishable.*

**Extrapolation as per-feature range extension** — sample 2–3 causal features from the truncated region |x| ∈ [2, 4] (essentially unseen in a 64-row ID pool), leave the rest alone. *Corrects 4.4(c): queries are now genuinely outside demonstrated support, the intervention is not removed by standardisation, and it is label-relevant for every rule family.*

**Spurious feature re-parameterised as a continuous correlate, at the spec's intended strength:**

```python
d = scipy.stats.norm.ppf(spurious_strength)          # sign-agreement = strength exactly
X[:, self.spurious_idx] = (2 * y_clean - 1) * d + rng.normal(size=n_samples)
```

with `spurious_strength_range: [0.80, 0.90]`. *Corrects the enabling condition of 4.2 (Fact B). At 0.96–0.995, feature 8 was a copyable label column — no learner should be expected to resist it, and per [9] the interesting regime for studying shortcut reliance is a shortcut that is tempting but breakable, more reliable than the causal signal on the training distribution but not a photocopy of the target.*

**Pool provenance becomes an explicit experimental dimension, with ID-pool as the headline.** Demos drawn from the `id` environment, queries from the shifted one — the deployment-realistic setting and the only one in which "shift-aware selection" is a meaningful capability. The v1 matched-pool setting is retained as a labelled secondary condition. *Corrects 4.4(d). No generator change needed — the saved task parquets already contain all splits.*

**Gate S2:** the generator re-validation now includes the held-out family and all six environments, with an explicit no-degenerate-cells check (every environment's label-1 rate within [0.2, 0.8] per task). *This closes the exact blind spot that let 4.4(a) through.*

### 5.3 Fix SATA's supervision (Stage 3)

**New target function — purely structural, no label term, a real penalty:**

```python
score = ( 2.0 * (demo["regime"] == query_metadata["regime"])
        + 1.5 * demo["is_counter_spurious"]
        - 1.5 * (demo["spurious_consistent"] and demo["regime"] != query_metadata["regime"]) )
```

*Corrects 4.2 (Fact A). The label-match term is removed because it conditions the supervision on information SATA cannot legitimately have; the +0.1 "penalty" becomes an actual negative term. Everything remaining is computable from generator structure. A supervision signal must not be cheaper to fit through a shortcut than through the intended structure [9,10] — v1's was.*

**Label-balanced top-k selection** (top k/2 per demo-label class, in both `sata_select` and the proxy). *Corrects 4.3 for SATA specifically: demo label statistics move the LLM's output prior [27], so balance isolates the content effect — and it makes the proxy's single-class fallback unreachable.*

**Unified standardisation** — one shared function, applied identically in training and inference, with statistics always computed on the full pool before any pre-filtering. *Corrects 4.5.*

**Training provenance matches evaluation** — demos from `id`, queries from the shifted environment (with a 25% matched-pool mix so the secondary condition is not out-of-training-distribution). *Corrects 4.4(d) on the training side: v1's matched pools made "shift-awareness" undefined during training, since pool and queries always co-shifted.*

**Proxy validation across all six environments with the checkpoint chosen on worst-environment accuracy, and the single-class fallback removed.** *Corrects 4.2 (Fact C). Worst-case validation is the operational meaning of "shift-aware"; v1 validated on `id` only, where the shortcut policy looks excellent.*

**Gate S3a — target-oracle sanity check, run before any training:** select top-k directly by the target function (no learned model) and evaluate with the all-environment proxy. If the supervision signal itself cannot beat the best protocol, nothing trained on it can, and the targets go back to the drawing board before any GPU time is spent. *This gate did not exist in v1; had it existed, the label-leak would have been caught in minutes rather than after the full run.*

**Gate S3b (Gate 2 redefined):** trained SATA must beat the best protocol on worst-environment proxy accuracy AND beat its own query-agnostic ablation. If it fails, RQ4 is written up as a rigorous negative with the full v1+v2 evidence chain — and the rest of the evaluation still runs, because RQ1–RQ3 do not depend on SATA.

### 5.4 Re-run and report both versions (Stage 4)

Both arms re-run with the fixed instrument (the chat-template and calibration fixes apply equally to the real arm, and may materially change RQ1). v1 results are preserved under a git tag and reported alongside v2. The v1 failure analysis is itself a contribution: a selector that passes its proxy gate at 0.96 and then scores 0.045 with the LLM is a clean demonstration of the gap between *demonstrations that are good training data for a learner* and *demonstrations that are good prompt content for an LLM* — the same distinction the lit review draws between TabPFN-style purpose-trained tabular ICL [18, 19] and general-purpose LLM ICL (§2.4), observed here in a single experiment.

---

## 6. What outcome confirms, and what outcome refutes

**Confirming outcome.** After the fixes: (RQ2) at least one diversity protocol beats random on its targeted shift type with a genuine crossover pattern in the protocol × shift grid; (RQ4) SATA with structural supervision beats the best protocol on worst-environment accuracy and beats its query-agnostic ablation; (RQ3) faithfulness ρ moves with the interventions rather than sitting at noise level.

**Refuting outcome — and why it would still be a strong thesis.** If, with a valid instrument (chat template, calibration, controlled ordering and label balance), a base model that demonstrably can learn the tasks (Gate S1), non-degenerate environments (Gate S2), and leak-free supervision that is verifiably worth learning (Gate S3a), demonstration selection *still* adds nothing beyond label-balance correction and SATA still fails to beat query-agnostic selection — then the honest conclusion is that demonstration **content** does not steer frozen-LLM feature reliance at this scale; only demonstration **label statistics** do. That is a direct, evidence-backed answer to the question the lit review poses at the end of §2.6 ("demonstration design has not targeted faithful OOD generalisation"), it is consistent with Min et al. [27] and extends their claim into the shift setting, and the v1-vs-v2 chain documents exactly why each alternative explanation was eliminated.

Either way, after v2 the experiment measures what it claims to measure. That is the entire purpose of the redesign.
