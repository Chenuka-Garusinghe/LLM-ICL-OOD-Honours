# Research plan: Can demo selection help 7–8B models follow the demos over their priors?

_Branch `synthetic-experiments` · approved 26 September 2026, updated the same day to PyTorch-first and on 27 September to PyTorch only · companion to `generator_spec.pdf`_

Repo root `LLM-ICL-OOD-Honours/`, project `sata-project/`. Synthetic data only.

## Context

**Why the change**

- Running 72B through AirLLM on the M3 Max takes about 40s per query, which is too slow. The thesis therefore moves to 7–8B models.
- Wei et al. (2023) found that following in-context evidence over semantic priors emerges with scale. The reframed question is whether demo selection can help 7–8B models do it, and in particular a **SATA selector trained with model-informed signals** (the model's zero-shot priors and internal readouts).
- The four RQs and SATA stay.
- **Backend: PyTorch only.** Everything runs in PyTorch through Hugging Face `transformers` on whichever GPU is available, auto-detected: Apple Metal (MPS) on the Mac, NVIDIA CUDA elsewhere, CPU as a last resort.
  - SATA is already PyTorch, as are the interpretability tooling and the code released with the papers in the reading list.
  - It uses the official checkpoints and runs unchanged on CUDA if GPU access returns.
  - AirLLM is dropped; it was only needed for 72B.
  - MLX is dropped too, for consistency. With one backend, every result comes from the same tokenisation, caching and scoring code, and there is only one runner to test.

**What the literature search found**

- As far as the search reached, no prior work uses a model's internal signals to select demos that overcome priors, on tabular data, or under shift.
- The nearest work is Kato et al. (2025), which is correlational only and never builds a selector.
- The diagnose-then-steer template already exists, so steering appears only as a causal check here.
- 7–8B models _partly_ follow flipped evidence (about half the time on sentiment tasks), so the framing is a **graded tug-of-war between prior and demos**, not a missing ability.

**The review found bugs that invalidate the current NB02 design.** No results were ever saved, so nothing needs re-running.

| #   | Bug (verified)                                                                                                                                                                                                                                                                                                                                      | Fix                                                                                                                      | Phase |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----- |
| 1   | NB01 merges 5 tasks with different rules into one 1,280-row pool. NB02 draws demos from all of them, and `head(30)` evaluates task 0 only                                                                                                                                                                                                           | Per-task pools; queries stratified across tasks                                                                          | P1    |
| 2   | Fixed `SEED` in every `select()` call gives one demo set per strategy for all queries. ID data is also byte-identical across the 3 datasets, so ID results are triplicated                                                                                                                                                                          | Fixed demo set per (task, strategy, seed); 3 seeds                                                                       | P1    |
| 3   | `counter_spurious` searches for its proxy using `test_ood` rows and labels (leak). Its intended ground-truth version over-samples counter-spurious rows, which reverses f8 in the demos and mirrors the spurious-reversal test. `rule_diversity` fits a tree on the mixed pool. The ground-truth `regime` and `is_counter_spurious` are never saved | Save metadata; use ground truth; make counter-spurious decorrelated                                                      | P1    |
| 4   | The prompt renders "…features of an individual, predict Classify the input…". The spec quotes a different prompt                                                                                                                                                                                                                                    | One clean template; the spec quotes real `render()` output                                                               | P1    |
| 5   | Llama prompts start with two beginning-of-sequence tokens (`[128000, 128000, …]`), because `MLXRunner` re-encodes an already-templated string                                                                                                                                                                                                       | The new runner tokenises templated prompts without adding special tokens                                                 | P1    |
| 6   | f8's standard deviation is about 1.44 against 1.0 for other features, so it stands out                                                                                                                                                                                                                                                              | Z-score every feature with ID-pool statistics                                                                            | P1    |
| 7   | `task_meta.json` lacks coefficients, thresholds and leaf labels; result rows lack task_id, seed and demo_ids                                                                                                                                                                                                                                        | Full task manifest; results schema                                                                                       | P1    |
| 8   | `_regime` uses all causal features, but `tree` uses only the first 3                                                                                                                                                                                                                                                                                | Use load-bearing features only (evaluation tasks use exactly 3)                                                          | P1    |
| 9   | `threshold`: every direction is positive, which confounds aligned vs flipped naming. `sparse_interaction` is not monotone                                                                                                                                                                                                                           | Evaluate on `linear` + `tree`. After the degeneracy filter, `tree` is always a signed majority of 3 signs, i.e. monotone | P1    |
| 10  | n=30 per cell gives a 95% CI of about ±0.18                                                                                                                                                                                                                                                                                                         | 24 tasks × 20 queries × 3 seeds                                                                                          | P4    |

**Intended outcome:** a correct, resumable pipeline; a prior-conflict (naming) factor; SATA ported with model-informed variants; a mechanistic diagnosis; and an updated `docs/generator_spec.pdf`.

## Confirmed decisions

- Timeline is more than 10 weeks.
- Models: **Qwen2.5-7B-Instruct** (primary), **Qwen2.5-7B base**, **Llama-3.1-8B-Instruct** (paused on 28 September 2026 by the user's decision: with the current prompt it explains its reasoning instead of answering, see P2; Qwen only for now).
- **Backend:** PyTorch + `transformers`, with the GPU auto-detected (NVIDIA CUDA, then Apple Metal/MPS, then CPU) and bf16 wherever the device supports it. AirLLM and MLX are both dropped (MLX on 27 September, for consistency).
- **RQ4 unchanged** (SATA vs hand-designed strategies); model-informed signals are used _inside_ SATA.
- **Concept shift is redefined through naming**: the task's feature→label mapping contradicts the model's pretrained mapping.
- The generator's `mechanism` environment stays for SATA training and is documented as an uninformative placebo for ICL.

## Research design

**Factors**

| Factor                  | Levels                                                                                                                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Naming (prior conflict) | abstract (neutral names) · aligned (semantic names agree with the rule) · flipped (the same names contradict it)                                                                               |
| Environment             | ID · covariate (P(X) moves) · spurious reversal (f8 flips). Concept shift = flipped vs aligned                                                                                                 |
| Selection               | zero-shot · random · label-balanced random · feature coverage · regime coverage · counter-spurious · feature-space kNN · counter-prior heuristic · SATA-v2 · SATA-MI (· SATA-MI-mech if gated) |
| Task                    | 24 per model: 12 linear + 12 tree, 3 causal features each. Domain (loan approval / medical risk) varies between tasks. The spurious direction is +1 in half of each (family, domain) cell, −1 in the other half |
| Seed                    | 3 demo seeds per (task, strategy)                                                                                                                                                              |

**RQ mapping**

| RQ  | Question on 7–8B models                                                          | Main evidence                                                                                               |
| --- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| RQ1 | Does ICL degrade under covariate, spurious and concept (flipped-name) shift?     | ID vs OOD, zero-shot and random, calibrated                                                                 |
| RQ2 | Do different strategies win under different shift types?                         | Strategy × shift interaction (e.g. counter-spurious on spurious, counter-prior on concept)                  |
| RQ3 | Does the model rely on the right features, and why does the prior win?           | Ground-truth importance vs ablation-based vs self-reported importance, by naming; logit-lens critical layer |
| RQ4 | Does SATA beat the hand-designed strategies, and do model-informed signals help? | SATA-v2 vs strategies; SATA-MI vs SATA-v2, with naming-aware controls                                       |

**Metrics and statistics**

- Primary: AUROC (from p1) and contextually calibrated balanced accuracy (Zhao et al. 2021; `calibration.py` from main).
- Secondary: raw accuracy, macro-F1, and **prior-agreement rate** (the share of predictions matching the model's zero-shot prediction).
- f8 reliance is read in the task's spurious direction, and reported for prior-aligned and prior-opposed tasks separately as well as pooled (the model's "bigger values mean 1" prior backs the shortcut in one half and opposes it in the other; found in P2).
- Hierarchical bootstrap over tasks (`bootstrap.py`: `bootstrap_ci`, `paired_bootstrap_diff`).
- Paired designs: the same queries across strategies; the same demo sets across environments and namings.
- Per-task sign tests, and Holm correction over a contrast list written down before P4.

**Novelty claim you can defend**

> First to train a demonstration selector on the model's measured semantic priors (a zero-shot prior surrogate) and on datamodel utilities (optionally read at the P6 critical layer), tested on whether it helps 7–8B models follow in-context evidence over conflicting priors under controlled covariate, concept and spurious shift.

Don't claim "first to train a selector from model feedback" or "first to use model internals for selection":

- EPR, UDR and CEIL already train selectors from the language model's own scores (`generator_spec.pdf`, Part SATA).
- Chang & Jia (ACL 2023) fit datamodels to in-context examples on the same correct-class margin that 5c uses.
- Kato et al. and DiSP partly use model internals or learned judges.

## Reading list (★ = read before starting that phase)

**P1–P2 (instrument and pilot)**

- ★ Wei et al. 2023, arXiv:2303.03846
- ★ Min et al. 2022, _Rethinking the Role of Demonstrations_, arXiv:2202.12837. Small models barely use the labels; this motivates the P2 gate.
- ★ Zhao et al. 2021, _Calibrate Before Use_, arXiv:2102.09690
- Pan et al. 2023 on task recognition vs task learning, arXiv:2305.09731
- Kossen et al. ICLR 2024, arXiv:2307.12375
- Lu et al. 2022, _Fantastically Ordered Prompts_

**P3 (priors and naming)**

- ★ Nafar et al. 2024, named vs anonymised tabular features, arXiv:2409.04318
- Krishna Kumar 2025, "Semantic Anchors", arXiv:2511.21038. Cite only its ~50% partial-following numbers; its "override rate = 0" is zero by construction.

**P5 (SATA)**

- ★ Kato et al. 2025, arXiv:2502.14380 (nearest scoop)
- ★ Xu & Zhang 2024, misconfidence-based selection, arXiv:2401.06301 (closest to counter-prior)
- ★ Nguyen & Wong 2023, influence-based selection, arXiv:2302.11042
- Ilyas et al. 2022, _Datamodels_, arXiv:2202.00622 (the method for 5c)
- ★ Chang & Jia, ACL 2023, _Data Curation Alone Can Stabilize In-context Learning_ (datamodels on in-context examples; the closest precedent for 5c)
- Rubin et al., NAACL 2022 (EPR); Li et al., ACL 2023 (UDR); Ye et al., ICML 2023 (CEIL): selectors trained from LM feedback, the precedents for SATA
- Peng et al. ICCD, arXiv:2502.13738 (optional baseline)

**P6 (mechanisms)**

- ★ Halawi et al. ICLR 2024, _Overthinking the Truth_, arXiv:2307.09476 (critical layer, logit lens)
- ★ Minder et al. ICLR 2025, arXiv:2411.07404
- Hendel et al. 2023, arXiv:2310.15916; Todd et al. 2024, arXiv:2310.15213

**RQ1 background**

- Garg et al. 2022, arXiv:2208.01066
- Zhang, Frei & Bartlett, arXiv:2306.09927 (ICL under covariate shift)

## Phases (each ends with a go/no-go gate)

### P0 — Housekeeping (½ day)

- **Save the AirLLM work first.** `src/inference/airllm_runner.py` has never been committed. This plan doesn't use it; this step just keeps the work. Run `git stash -u`, switch to `synthetic-exp-airllm`, run `git stash pop`, commit the runner and the NB02 edits there, then switch back. NB01's diff is outputs only.
- **Port from main** (the files are self-contained):
  ```
  git checkout main -- src/models/{sata,sata_targets,sata_train,standardise,xgb_proxy}.py \
    src/selection/{protocols,balanced_topk,sata_select}.py src/inference/calibration.py \
    src/evaluation/{bootstrap,faithfulness,faithfulness_correctness}.py src/utils/ tests/ \
    scripts/gate_s3a_oracle_check.py configs/default.yaml
  ```
- **Trim `protocols.py`:** remove `importance_weighted`, `shift_axis_coverage` and `target_prior`, which need real-data shift estimation.
- **`configs/default.yaml`:** synthetic values (families `[linear, tree]`, 3 causal features, spurious strength [0.80, 0.90], pool size shared by SATA and evaluation).
- **`requirements.txt`:** remove vllm, folktables, whyshift and graphviz; keep torch and transformers; add python-dotenv.
- **Download the official checkpoints** (about 15–16 GB each): `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-7B` and `meta-llama/Llama-3.1-8B-Instruct`. The Llama repo is gated, so request access on Hugging Face first; the download then uses the `HF_TOKEN` in `.env`.
- **`.gitignore`:** add the LaTeX build files (`docs/*.aux|log|out|toc` are currently tracked).
- **GPU auto-detection:** new `src/utils/device.py`.
  - `resolve_device`: CUDA, then MPS, then CPU, or a forced choice.
  - `resolve_dtype`: bf16, but fp16 on pre-Ampere CUDA and fp32 on CPU.
  - `device_name`: the hardware name, recorded with results.
  - Used by SATA training and SATA selection (which previously assumed CPU tensors) and the old similarity selector; the P1 runner will use it too.
  - `configs/default.yaml` sets `inference.device` and `sata.device` to `auto`; `PYTORCH_ENABLE_MPS_FALLBACK` is set in `src/utils/config.py`; tests in `tests/test_device.py`.
  - Device choice doesn't change the design, but kernels differ slightly across devices, so every result row records device and dtype and each comparison runs on one device.
- _Optional, your call:_ delete the 72B cache (about 290 GB including the split copy) and the MLX copies of the 7B/8B models (about 31 GB), which the PyTorch pipeline doesn't use. Re-downloading takes hours, so only delete what you won't need.
- **Gate:** `pytest tests/` passes.
- **Status: done 26 September 2026** (27 tests pass; checkpoints downloaded; Qwen verified on MPS in bf16).

### P1 — Correctness (1 week)

- **`src/data/generator.py`:**
  - `to_meta()` / `from_meta()` for the full parameters;
  - `load_bearing_features()`, with `_regime` computed over them;
  - `is_counter_spurious` measured against the observed label;
  - an evaluation-task sampler (3 causal features, linear |coef| ~ U(1, 2) with random sign).
- **`src/data/synthetic_bridge.py`:**
  - one long-format parquet per task (split, env, features, label, y_clean, regime, is_counter_spurious);
  - z-scores using ID-pool statistics;
  - a per-task random **column-role permutation**, so the spurious and noise features aren't always f8/f9 (this removes a positional shortcut for SATA);
  - `task_manifest.json`;
  - ID data written once.
- **`src/inference/prompts.py`:**
  - one synthetic template reusing the `SYNTHETIC_TASK_DESCRIPTION` rule-induction wording, plus "values are standardised (0 = average)" and "Respond with 0 or 1, where 1 = {meaning}";
  - an optional domain sentence; label tokens stay `0`/`1` in every condition;
  - a raw-completion variant for the base model that ends with `"-> "`, because `" 0"` tokenises as `[220, 15]` (confirm at token level in the G1 prompt test).
- **New `src/inference/hf_runner.py` (`HFRunner`).** It keeps the old `MLXRunner` interface (`chat_formatter()`, `batch_predict()`) and reuses `PredictionResult` and `get_confidence` from `llm_runner.py`. It:
  - loads `AutoModelForCausalLM` on `resolve_device(config.inference.device)` with `resolve_dtype(...)`, in eval mode under `torch.inference_mode()`;
  - loads `.env` (python-dotenv) first: the gated Llama repo needs `HF_TOKEN` even when cached, because transformers probes it for an optional tokenizer file and gets a 401 without the token;
  - tokenises templated prompts with `add_special_tokens=False`, so Llama gets exactly one BOS;
  - applies the output layer at the last position only, and computes the two label logits in fp32 from the final hidden state (changed in P2: the bf16 output layer put every margin on a 1/16 grid);
  - provides `score(prefix, suffixes)`: runs the demo prefix once, then scores each query suffix against the cached `past_key_values` (a `DynamicCache`, cropped back to the prefix length after each suffix). It checks that the prefix tokens really are a prefix of the full prompt, and scores a suffix that fails the check with a full forward pass;
  - provides `generate_text()` (greedy `model.generate`) for the feature-ranking prompts;
  - has a `use_chat_template` flag for the base model;
  - uses `sdpa` attention by default (`eager` only if P6 needs attention weights);
  - sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, so an unsupported op falls back to CPU instead of crashing.
- **`src/inference/mlx_runner.py`:** remove it once `HFRunner` passes its tests. PyTorch is the only backend. Its only caller is the old `02_synthetic_baselines` notebook, which `03_grid_run` replaces.
- **`src/selection/protocols.py`:** add synthetic mechanisms:
  - `gt_regime` (strata);
  - `gt_counter_spurious`, **magnitude-matched**: per class, pair each counter-spurious row with the spurious-consistent row of nearest |f8|. Simply balancing the label × agreement cells still leaves corr(f8, y) ≈ 0.33 at s = 0.85, because agreeing rows have larger |f8|; matching brings it to 0.00 (derivation in `generator_spec.pdf` §19.1);
  - `feature_knn` (Euclidean distance on z-scores, replacing MiniLM, whose embeddings change with the names).
  - Structured strategies use `composition="balanced"`; plain random stays `"free"`. Keep the old names through `V1_EQUIVALENTS`.
  - Demo sets are fixed per (task, strategy, seed) using `np.random.SeedSequence([base, task, strategy, seed])` and reused across environments and namings. Query-conditional strategies select per query.
- **New `scripts/run_synth_grid.py`:**
  - resumable units (model, naming, task, strategy, seed) written with `results_schema.append_results`;
  - result rows carry model, device, dtype, naming, domain, task_id, family, env, strategy, seed, query_id, demo_ids, prompt_hash, p0/p1, calibrated p1, prediction, label;
  - calibration uses a content-free query (`calibration.content_free_features`) on the same demo prefix.
- **New `tests/test_synth_pipeline.py` and `tests/test_hf_runner.py`.** The runner tests use the tiny, already-cached `hf-internal-testing/tiny-random-gpt2`, so they run in seconds without loading a 7B model.
- **Spec v3, part 1** (see below).
- **Gate G1:**
  - pools hold only the query's task, and no test rows reach any selector;
  - structured demo sets are label-balanced, and counter-spurious sets have pooled |corr(f8, y)| ≤ 0.1;
  - one BOS token for Llama;
  - on the tiny test model, prefix-cached scoring matches a full forward pass (label log-probs within 1e-4);
  - per-task label rate is in [0.35, 0.65], and f8 agreement is within ±0.03 of the configured strength;
  - the prompt snapshot test passes.
- **Status: done 28 September 2026.** G1 passes (74 tests).
  - **Code:**
    - `src/data/generator.py`: `sample_eval_tasks`, `load_bearing_features`, `directions`, `to_meta`/`from_meta`, observed-label agreement;
    - `src/data/synthetic_bridge.py` and `src/data/suites.py`: per-task parquets and manifest; `eval` = 24 tasks, `pilot` = 12;
    - `src/inference/prompts.py` and `src/inference/hf_runner.py`;
    - `gt_regime`, `gt_counter_spurious` and `feature_knn` in `protocols.py`, plus `demo_seed`;
    - `src/experiments/synth_grid.py` with `scripts/run_synth_grid.py` and `scripts/make_synth_data.py`;
    - tests in `test_synth_pipeline.py`, `test_hf_runner.py` and `test_gates.py`.
  - **Smoke run:** on Qwen2.5-7B-Instruct it takes about 4 s per unit of 13 scored queries. The kill-and-resume check passed. Flipped naming waits for P3.
  - **Refinements made while implementing** (in `generator_spec.pdf`):
    - feature coverage uses joint cells (the version ported from main only ever used the top feature);
    - both ground-truth strategies exclude label-noise rows;
    - regime coverage uses only regimes where the row's class is the majority.
  - **Removed:** `mlx_runner.py`. The v2 NB02 and NB03 are in `notebooks/archive/v2/`.
  - **Similarity is expensive:** it builds one prompt per query, so it costs about 5× as much per unit. P4 should budget for it, or use one seed, since kNN is almost deterministic.

### P2 — Pilot (3 days, about 1 GPU-hour)

- **Throughput:** time Qwen2.5-7B-Instruct with PyTorch on MPS, for a full 1k-token prompt and for a cached query suffix, and update the compute budget with the measured figures.
  - If the measured budget no longer fits the timeline, apply the cut order (end of P7) rather than adding a second backend.
- Check that prefix caching does not change the results: the mean per-task AUROC from the cache must be within 0.005 of full forward passes (adopted 28 September in place of "at least 99% prediction agreement", which fails on near-ties unrelated to the cache). Prediction agreement is still reported.
- Check that each model answers with the label tokens: its median P("0") + P("1") on the pilot must be at least 0.9, or its gate result does not count (added 28 September).
- **Learnability gate** (abstract names, Qwen-Instruct and Llama): zero-shot vs gold labels vs shuffled labels, k ∈ {8, 16, 32}, 12 tasks × 3 seeds.
  - **Go:** AUROC(gold) − AUROC(shuffled) ≥ 0.05, with a CI above 0. Use the smallest k that passes.
  - **No-go:** retry with 6 visible features. If it still fails, abstract names become the floor condition and the named conditions carry RQ2–4. That outcome is still consistent with Wei et al.
- **Status: done for Qwen2.5-7B-Instruct (28 September 2026); Llama paused.** The history is in `hiccups/` 07–11.
  - **First run** (12:52–14:10; archived in `results/v3/synthetic/archive/fixed_f8/`, data in `data/synthetic/v3/archive/fixed_f8/`):
    - Benchmarks at k = 8 (sustained): Qwen prefill 361 tokens/s and 0.275 s per cached suffix; Llama 291 tokens/s and 0.368 s.
    - Cached vs full: Qwen agreed on 99.2% of predictions, Llama on 97.6%. All 6 Llama flips had |margin| ≤ 0.096, the largest cached-vs-full difference; Llama's median |margin| is 0.21 logits against Qwen's 5.2, so most of its queries are near-ties. Proposed replacement for the 99% rule (awaiting a decision): the mean per-task AUROC from the cache within 0.005 of full passes (a tenth of the gate threshold), with prediction agreement reported alongside. The benchmark now saves per-query scores for this.
    - Qwen learnability: passed at k = 8 (Δ = 0.108, 95% CI [0.056, 0.163], 12/12 tasks) and k = 16, failed at k = 32 (CI includes 0). The run was stopped after 32 Llama units.
    - **Finding:** Qwen's zero-shot margin rises with every feature value (Spearman 0.70 with the sum of the features). With f8's direction fixed, that prior backed the shortcut in every task: the sum of features scores ID queries at AUROC 0.628 (0.511 without f8), and Qwen's zero-shot AUROC was 0.633. Gold demonstrations added little over zero-shot (0.66 against 0.63).
  - **Fix:** a per-task spurious direction σ_t ∈ {+1, −1}, balanced within each (family, domain) cell and drawn from a separate stream. Only f8's sign changes (other columns, labels, |f8|, agreement flags and counter-spurious sets are identical; tested). Suites regenerated; 78 tests pass; NB01 re-run.
  - **Rerun** (14:12–16:13, same tasks with σ_t):
    - **Qwen gate passes at k = 8:** Δ = 0.098, 95% CI [0.021, 0.188], 9/12 tasks. It also passes at k = 16 (Δ = 0.159, [0.074, 0.238]) and fails at k = 32. Label mass 0.996. **k = 8** (`configs/default.yaml`).
    - **Zero-shot AUROC:** 0.499 overall; 0.642 where σ_t = +1 (prior backs the shortcut) and 0.357 where σ_t = −1. Those σ_t = −1 tasks scored 0.625 in the first run, and flipping f8's sign was the only change. The 6 unchanged tasks gave bit-identical scores in both runs.
    - **Qwen cache check passes:** mean per-task AUROC 0.5867 cached against 0.5842 full (difference 0.0025); prediction agreement 99.6%.
    - **Throughput:** Qwen prefill 450 tokens/s and 0.294 s per cached suffix; Llama 326 tokens/s and 0.351 s.
    - **Llama fails at every k** (Δ = 0.026–0.045, every interval includes 0), but the result is **not valid**. Its label mass is 0.009–0.11 with demonstrations and 0.000 zero-shot. Its most likely first token is "To", as in "To infer the rule, I will analyze the given examples…". Its scores barely vary between queries (sd 0.05), so its cache check fails as well (AUROC difference 0.013).
    - **Llama fix, if it is resumed:** starting the assistant's reply with `Label: ` raised its label mass to about 0.99 on pilot prompts. The user paused Llama instead (Qwen only for now).
    - Smoke runs regenerated with the final data and runner; the kill-and-resume check passed again (26 units, no duplicates).

### P3 — Priors and naming (1 week)

- **New `configs/lexicons.yaml`:** 2 domains.
  - Causal features get antonym pairs (e.g. income ↔ debt ratio).
  - **Every** non-causal column gets a plausible name with a weak prior, so names don't reveal which features are causal.
- **New `src/data/naming.py`:**
  - builds the codebook per (task, naming), feeding `serialise_row` through `name_extended`;
  - directions come from ground truth (coefficient sign or majority sign);
  - aligned and flipped pairs are matched on measured prior strength;
  - the aligned and flipped system prompts must be byte-identical.
- **New `src/inference/priors.py`:**
  - zero-shot margins on about 400 random profiles per (model, domain), then ridge regression gives a per-name slope (the **prior surrogate**), reported with held-out R²;
  - a random-label demo control.
- **Add a `counter_prior` mechanism:** it scores demos that contradict the prior _and_ follow the true rule, and applies only when the pool shows prior–data conflict.
- **Gate G3:** surrogate held-out R² ≥ 0.5, and zero-shot aligned minus flipped accuracy ≥ 0.15. Replace names that fail. If the gate can't be met, drop the naming factor and document it.

### P4 — Main grid + RQ3 behavioural (1.5 weeks, overnight runs)

- Full grid on Qwen-Instruct.
- Reduced grid on Qwen-base (and Llama, if resumed with a fixed answer format): abstract + flipped namings; zero-shot, random, counter-spurious, counter-prior and SATA variants.
- **RQ3:**
  - π_true from `true_importance_scores`, π_behav from `hot_deck_impute_feature`, π_self from `build_feature_ranking_prompt` + `parse_feature_ranking`, all by naming;
  - directional sensitivity: nudge each feature ±δ and check whether the margin follows the data direction or the name direction.

### P5 — SATA (2 weeks; CPU training overlaps P4)

- **5a SATA-v2** (the RQ4 baseline, as designed):
  - train on the evaluation task distribution with the same pool size;
  - device auto-detected (`sata.device: auto`); benchmark it once against `cpu`, since SATA's batches are tiny and GPU launch overhead may outweigh the gain;
  - environments id, covariate, spurious_reversal and mechanism (matched-pool, as on main);
  - `gate_s3a_oracle_check.py` first: the targets must beat the strategies on the proxy;
  - XGBoost proxy for early stopping;
  - final **LLM validation on 8 held-out validation tasks**, never the evaluation tasks.
  - Disclose that the rule and counter-spurious strategies get ground-truth tags at inference, while SATA doesn't.
- **5b SATA-MI:**
  - add per-row inputs (x, slope⊙x, prior logit) from the P3 surrogate, including for the query;
  - add a conditional counter-prior target term;
  - controls: **SATA-names** (lexicon directions ±1 instead of LLM-measured slopes) and the counter-prior heuristic;
  - validation reader: σ(α·prior(x) + w·x) in a new `src/models/prior_proxy.py`.
  - **Gate:** proxy–LLM Spearman ≥ 0.5, and SATA-MI ≥ both controls on validation.
- **5c SATA-MI-mech** (headline novelty, gated):
  - targets are **datamodel utilities** measured with the LLM: about 20 tasks × 2 namings × 128 balanced subsets from a 64-row pool; the mean correct-label margin over a label-balanced query set is regressed on subset membership (new `src/models/datamodels.py`);
  - label-symmetric, so no query-label leak;
  - optionally measured at the P6 critical layer.
  - **Gate:** held-out R² ≥ 0.2, and top-k by utility beats random under the LLM.
- **Files:** `src/models/sata.py` (configurable input width), `sata_targets.py`, `sata_train.py`, `src/selection/sata_select.py`.

### P6 — Mechanistic diagnosis (1 week)

- **New `src/inference/probes.py`** (PyTorch, on `HFRunner`'s model):
  - per-layer last-token residuals from `output_hidden_states=True`;
  - forward hooks on `model.model.layers[i]` wherever a layer's output must be changed (steering);
  - logit lens through `model.model.norm` + the label rows of `lm_head` (both models are untied);
  - store per-layer margins for all probe prompts and residuals for a subset;
  - per-layer linear probes;
  - optional steering vector, as a causal check only;
  - TransformerLens is optional; plain hooks are enough.
- **Analyses:**
  - the critical layer where aligned and flipped margin trajectories diverge;
  - whether counter-prior or SATA-MI demos move the crossover earlier;
  - steering as an upper bound.
- **Gate:** the final-layer logit-lens margin equals the runner margin (within 1e-3).

### P7 — Analysis and write-up (remaining weeks)

- **Design history for the write-up and the presentation:** `hiccups/`, one entry per problem, pivot and decision (what happened, evidence, decision, a draft paragraph), with a timeline in `hiccups/README.md`. Entries are added as issues arise.

**Cut order if time runs short:** P6 → 5c → reduced grids → 5b. The minimum thesis is P0–P2, P4 and 5a, which still answers all four RQs.

## Notebook layout (experiments run from `scripts/`; notebooks drive and analyse)

| New notebook                 | Replaces                         | Purpose                         |
| ---------------------------- | -------------------------------- | ------------------------------- |
| `01_synthetic_data`          | (updated, P1 ✓)                  | Per-task generation             |
| `02_pilot_and_priors`        | new (P2 ✓; P3 part later)        | Pilot and prior screening       |
| `03_grid_run`                | `02_synthetic_baselines` (P1 ✓)  | Driver over `run_synth_grid.py` |
| `04_sata_train`              | port of main's `05_sata_train`   | SATA training                   |
| `05_analysis`                | `03_synthetic_analysis`          | RQ1, RQ2, RQ4                   |
| `06_faithfulness_and_probes` | new                              | RQ3                             |

Results go to `results/v3/synthetic/`.

## `docs/generator_spec.tex` → recompiled PDF (v3)

Build with `latexmk -pdf generator_spec.tex` in `docs/`.

**Part 1, in P1:**

- Title and date → v3.
- §Config table → new values; families linear + tree; 3 causal features.
- §Tree → the signed-majority characterisation (all 8 admissible leaf tables).
- §Environments → evaluation uses id / covariate / spurious_reversal; `mechanism` documented as a placebo (why: symmetric features); concept shift is defined via naming.
- §Dataset construction → per-task pools, standardisation, column-role permutation, manifest, shared ID data.
- New §Naming protocol → abstract / aligned / flipped, lexicons, ground-truth directions, identical prompts.
- §Demo selection → mechanism × composition, ground-truth regime and counter-spurious strata, feature kNN, seeding and reuse.
- §Serialisation / §Prompt → the new template quoted from real `render()` output; the base-model `"-> "` format.
- §Inference → `HFRunner` (PyTorch on MPS, bf16) replaces `MLXRunner`; double-BOS fix, prefix caching, calibration.
- §Evaluation → AUROC, calibrated balanced accuracy, hierarchical bootstrap, Holm.
- §Orchestration → the grid script and result schema.
- Bug-hunting notes → mark fixed items and add the new ones (f8 spread, `" 0"` tokenisation, positional shortcut).

**Part 2, in P3/P5/P6:** new §Prior measurement, a new Part "SATA" (v2 / MI / MI-mech: inputs, targets, leakage controls, validation), and a new Part "Mechanistic probes".

## Compute budget (measured in P2)

Measured on 28 September 2026: M3 Max, PyTorch on MPS, bf16, batch size 1, one model in memory at a time.
A P4 unit is one demonstration set scored on 60 queries (20 per environment) plus the content-free query.

| Unit (P4 size)                                   | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
| ------------------------------------------------ | ------------------- | --------------------- |
| Query-agnostic strategy, k = 8 / 16 / 32          | 24 / 26 / 32 s      | 24 / 25 / 31 s        |
| Zero-shot                                         | about 23 s          | about 24 s            |
| Similarity (one prompt per query), k = 8          | about 155 s         | about 185 s (est.)    |

Sources: `results/v3/synthetic/p2/p4_unit_timing.parquet`, the smoke runs, and `unit_seconds` in `p2_learnability.parquet`.
The provisional figures (from the old MLX pipeline) assumed about 18.5 s per k = 8 unit, so units cost about 1.3× more than budgeted.

**Qwen-Instruct grid at k = 8** (3 namings × 24 tasks):
- 6 query-agnostic strategies × 3 seeds: 8.7 h;
- zero-shot: 0.5 h;
- similarity: 9.3 h at 3 seeds, or 3.1 h at 1 seed (kNN is almost deterministic).

That is 12–18 GPU-hours, against the provisional 10. The other items are rescaled by 1.3×.

| Item                        | Provisional | Rescaled (P2)                          |
| --------------------------- | ----------- | -------------------------------------- |
| Pilot + prior screening     | 2.5         | 7 (P2 alone used about 4.5 by 28 Sep)  |
| Qwen-Instruct grid          | 10          | 12–18 (similarity at 1 or 3 seeds)     |
| SATA conditions in the grid | 6           | 8                                      |
| Reduced grids               | 9 (2 models) | 6 (Qwen base; 12 if Llama resumes)    |
| RQ3                         | 3           | 4                                      |
| SATA LLM validation         | 5           | 6.5                                    |
| 5c datamodels               | 9           | 12                                     |
| P6                          | 3           | 4                                      |
| **Total**                   | **≈ 48**    | **≈ 60–66 (7–8 overnight runs)**       |

The largest lever is the number of similarity seeds, to be fixed before P4. If the total no longer fits the timeline, apply the cut order (P7). Without prefix caching the total would be well over 150. Run under `caffeinate` with resumable units.

## Verification

- `pytest tests/` (ported tests plus `test_synth_pipeline.py`, `test_hf_runner.py`, `test_naming.py`, `test_probes.py`) passes at every phase.
- Smoke run: `python scripts/run_synth_grid.py --run smoke --models qwen2.5-7b-instruct --tasks 2 --queries 4 --strategies random,counter_spurious` (add `--namings abstract,flipped` once P3 exists).
  - Check the parquet schema.
  - Kill it mid-run and rerun: there should be no duplicate units.
- Gates G1, P2, G3, 5a–5c and P6 above are the go/no-go checks.
- Spec: `latexmk` builds cleanly, and the quoted prompt equals `ChatFormatter.render()` for one real row.
