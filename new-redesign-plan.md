╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ SATA Experiment Redesign — Give the Hypothesis a Valid Test │
│ │
│ Context │
│ │
│ The first full run of the SATA experiment produced negative results everywhere: all methods near chance on the synthetic arm, SATA variants the worst methods (5–10% accuracy on spurious-reversal — near-perfect anti-correlation), no protocol × shift interaction, no real-arm OOD degradation. Code exploration traced these to concrete implementation defects, not to the hypothesis itself. Each defect below is a bug or confound whose correction is scientifically justified — this is a redesign to make the test valid, not to force a positive result. The hypothesis may still fail after the fixes; either way the thesis gets a defensible answer. │
│ │
│ Root causes found (confirmed in code) │
│ │
│ A. Inference layer — why every method is near chance (both arms) │
│ 1. src/inference/llm*runner.py sends raw completion strings; the -Instruct models never get their chat template (apply_chat_template absent from the codebase). │
│ 2. src/inference/prompts.py synthetic prompts are semantically empty (feature_0: -0.43; … -> 1); zero-shot models predict class "1" 96–99% of the time; no calibration anywhere. │
│ 3. Demo order in prompt = selector return order (SATA/similarity: best demo first, i.e. furthest from the query). │
│ 4. Synthetic arm ran on 1 seed despite 5 in config; inconsistent number formatting. │
│ │
│ B. SATA target/training — why SATA actively inverts │
│ 1. src/models/sata_targets.py gives +0.5 when demo label == the query's true label, and the spurious feature 8 is literally the label ±N(0, 0.1) at 96–99.5% agreement. SATA learned: read the query's label off feature 8, select label-matching demos. Gate-2 proxy 0.9614 ≈ spurious strength. Under spurious-reversal feature 8 anti-predicts, so SATA selects wrong-label demos and the LLM copies the majority label → 5–10% accuracy. Mechanism fully explains the worst result in the data. │
│ 2. The "spurious-only penalty" is a +0.1 bonus; no negative terms exist. The counter-spurious term is nearly dead in 5 of 6 environments (0.5–4% base rate) and nearly constant in the 6th (96–99.5%). │
│ 3. evaluate_sata_proxy validates on the id environment only, and a single-class fallback (top-k all one label → predict that label) directly rewards the degenerate label-copy policy. │
│ 4. Standardisation mismatch: training feeds raw features; inference z-scores over the candidate set (differently for sata_alone vs best_protocol_sata). │
│ 5. The kept checkpoint is epoch 0; val-proxy falls monotonically as KL loss improves — the target and the goal disagree. │
│ │
│ C. Generator — degenerate/confounded environments │
│ 1. sparse_interaction.threshold is never sampled (stays 0.0) → missing_feature zeroes the product's first factor → labels ~98% constant. The held-out family was excluded from the notebook-04 gate, so this went unnoticed. │
│ 2. Covariate shift perturbs 3 of all 10 features (wasted on 8/9, which get overwritten) before rule evaluation → shifts the label base rate to 0.388 (label shift confound). │
│ 3. Extrapolation is one global scalar ×U(2,3): label-preserving for tree/sparse_interaction and erased by inference-time standardisation. │
│ 4. Demo pools are always drawn from the same shifted environment as the queries (oracle assumption); the deployment-realistic setting — pool from ID, queries from OOD — is never tested. │
│ │
│ Approach │
│ │
│ Five stages with go/no-go gates: (0) CPU-only diagnostics on existing results to document the failure mechanisms for the thesis, plus a new NB00 data-diagnostics notebook that makes every consumed dataset visually diagnosable with paired automated checks (v1's defects survived precisely because no notebook ever plotted the data — NB01 and NB04 contain zero plots); (1) inference-layer fixes + pilot gate on whether 7–8B models can learn the tasks at all (escalate model size if not); (2) generator fixes + re-gate, with Gate S2 now consuming NB00's check functions; (3) SATA target/training/selection fixes + retrain; (4) full re-run of both arms and refresh of figures/tables. Old results are kept under results/v1/ for the thesis narrative (before/after is itself a contribution). │
│ │
│ Versioning (do first) │
│ │
│ - git tag v1-results on current commit — v1 is the "before" evidence for the thesis. │
│ - New configs/v2.yaml (copy of default.yaml) routing outputs to data/synthetic_v2/, results/v2/, models/v2/, figures/v2/, tables/v2/; new keys: prompting: {chat_template, order_policy: shuffle, calibration}, generator.spurious_strength_range: [0.80, 0.90], pool_provenance: [id_pool, matched_pool]. │
│ - src/utils/config.py::load_config: honor env var SATA_CONFIG (mirrors existing SATA_MODEL_FILTER pattern), default configs/v2.yaml. │
│ - src/utils/results_schema.py: add columns pool_provenance, order_seed, logprob_0_cf, logprob_1_cf, prediction_raw, prompt_version; keep v1 schema importable for side-by-side loading in NB08. │
│ │
│ Stage 0 — CPU diagnostics on v1 results (thesis evidence, no GPU) │
│ │
│ New scripts/diagnose_v1.py (needs HPC or rsync of data/synthetic/tasks*_ — local data/synthetic/ is empty). Outputs results/v1_diagnostics/_.parquet: │
│ 1. Demo-set label composition per method × env (join saved demo*ids to task parquets): fraction of demos matching query's true label; fraction of single-class demo sets. Prediction: SATA ≈ spurious strength, near-single-class. │
│ 2. Spurious-agreement rate: P(prediction == feature-8-implied label) vs P(prediction == label) per method × env — should show SATA methods tracking feature 8 everywhere, explaining the 5–10% reversal accuracy. │
│ 3. SATA selection composition in spurious_reversal (spurious_consistent fraction, demo-label vs implied-label agreement). │
│ 4. Zero-shot class-1 prediction rate per model × dataset/env (both arms). │
│ 5. Demo-position vs distance-to-query correlation (documents v1's inverted ordering). │
│ 6. Full-SATA vs query-agnostic selection Jaccard + training-log curves (documents epoch-0 checkpoint / target-proxy disagreement). │
│ No gate — confirmatory evidence for the NB08 "v1 failure analysis" section. │
│ Stage 0b — NB00 data-diagnostics notebook (new; visual + automated gate) │
│ Motivation: every v1 data defect (constant-label cell, covariate label-shift leak, photocopy spurious feature, dead counter-spurious signal, inert extrapolation) was visible in the data itself, but NB01 and NB04 contazero plots, so nothing was ever seen. NB00 makes the datasets visually diagnosable and pairs eeadable pass/fail check. It runs on v1 data now (expected result: FAIL, flagging exactly the │post-mortem defects — that's the acceptance test and thesis evidence) and re-runs on v2 data as the visual half of Gate S2. │
│ Files: - New src/evaluation/data_diagnostics.py (~500 lines): CheckResult dataclass (name, scope, pashreshold, detail), pure check functions (DataFrames in, CheckResults out — no I/O/plotting so Stage │ 2's Gate S2 imports them directly), shared stats helpers (smd, tv_distance, ks_stat, feature_shift_table respecting codebook kind), rehydrate_task_from_meta (rebuilds SyntheticTask from the \_meta.json sidecar to call \_apply_rule for the extrapolation-inertness check), write_gate_report. │
│ - New notebooks/00_data_diagnostics.ipynb — named 00*, no renumbering of NB01–08 (HPC PBS/deploy scripts and model-suffixed variants reference current filenames; 00 already encodes "runs first"). - configs/default.yaml: add paths.tableshift_raw_cache and a data_gate: thresholds block (so vwithout code edits). NB00 setup reads SATA_CONFIG env var → fully version-agnostic (audits v1 │data/synthetic or v2 data/synthetic_v2 by config switch alone). │
│ Check catalogue (each check = one plot + one gate row; thresholds in config): - Real arm (runs LOCALLY off data/tableshift_raw_cache/ — data/real being HPC-only doesn't blo[0.2, 0.8] per split; ID-vs-OOD per-feature shift (SMD numeric / TV categorical, categoricals │rendered via codebook value labels, never as numerics) with max-shift ≥ 0.10; XGBoost ID→OOD gap (learnable above majority, no below-chance cells, gap ≥ 0.02 — directly tests the RQ1 premise on a trained model); missingness ≤ 50% per selected feature (NaN + na_values codes); demo-pool representativeness (stically with build_demo_pool(256, seed=42), pool-vs-train max |SMD| ≤ 0.25); MI bar chart (re-call │ mutual_info_classif inline — select_top_features discards the scores). - Synthetic arm (runs on HPC; audit set = all test + all heldout + 50 seeded train tasks, load: per-env label rate ∈ [0.2, 0.8] per task (catches the 0.000 cell); |rate_env − rate_id| ≤ 0.10 │for covariate/extrapolation (catches the label-shift leak); configured spurious strength ≤ 0.95 + empirical agreement within ±0.03 of configured (catches the photocopy — deliberately FAILS on v1); counter-spurious rate ∈ [0.05, 0.5] in non-reversal envs (catches the dead signal); extrapolation label-flip fractiole (catches inertness); fresh ERM/oracle learnability profile per env × rule family, independent of │ NB04's gate, cross-checked against gate_validation.parquet; demo-vs-query drift per env + env-pool-vs-id-pool provenance drift. │
│ - Shared: prompt token budget vs max_model_len (try/except-guarded transformers import; SKIPPE │
│ │
│ Notebook shape (~22 cells): setup with SATA_CONFIG indirection → availability probe (REAL_AVAIt arm → all its checks recorded SKIPPED with reason, never an exception) → Section A real arm (~7 │figure cells) → Section B synthetic arm (~7 figure cells) → Section C token budget → Section D gate report: results/data_gate_report.json (machine contract: overall_pass ignoring SKIPPED, n_skipped exposed so HPC runs │
│ can demand zero skips), tables/data_gate_report.csv, styled PASS/FAIL table + banner cell (pris to figures/data_diagnostics/\*.pdf following NB08 conventions (whitegrid, vector PDF, viridis │annotated heatmaps). │
│ │
│ Pipeline contract: NB01–08 must not run (and Stage 2 must not freeze a generator) unless the latest data_gate_report.json for the target data version has overall_pass: true (with zero skips on HPC). Gate S2's │
│ programmatic half = the same check functions imported from data_diagnostics.py. │
│ │
│ Effort: ~2.5–3 days (module → real-arm cells testable locally → synthetic cells validated on H │
│ │
│ Stage 1 — Inference-layer fixes + pilot gate │
│ │
│ - 1a. Chat template — new src/inference/chat.py: ChatFormatter wrapping tokenizer.apply_chat_tion_prompt=True, tokenize=False); label = assistant's first token; existing allowed_token_ids │constraint and get_confidence in src/inference/llm_runner.py carry over unchanged (runner already takes strings — no vLLM changes). │
│ - 1b. Prompt content — edit src/inference/prompts.py: new build_chat_messages() with semantic for the synthetic arm ("unknown rule over numeric measurements; infer from labelled examples") — │deliberately NOT real-world feature names (would inject priors). Keep v1 builder for comparability (prompt_version column). │
│ - 1c. Formatting — src/data/serialisation.py::\_format_number: fixed-width f"{f:.2f}" (tokenisa │
│ - 1d. Demo order — new src/selection/ordering.py: primary policy = seeded random shuffle for ALL methods (corrects v1's best-demo-first anti-recency bug asymmetrically affecting SATA/similarity; makes ordering a uniform │
│ controlled nuisance). similarity_ascending as pilot-only ablation. Store order_seed. │
│ - 1e. Contextual calibration — new src/inference/calibration.py (Zhao et al. content-free "N/A" query paired in the same batch; ~2× calls, prefix caching makes it cheap). Report raw AND calibrated — it's a correction │
│ for the measured 96–99% class-1 prior bias, not a silent change. │
│ - 1f. Multi-seed — NB06 loops all 5 seed_accuracy seeds. │
│ - Gate S1 (pilot) — scripts/pilot_stage1.py: ~20 existing v1 test tasks, id env, {zero-shot, r}, both models. Pass: random-8 ID accuracy ≥ 0.60 (chance 0.5, XGBoost-64 ≈ 0.8) and calibrated │zero-shot class-1 rate ∈ [0.35, 0.65]. On fail: escalate to Qwen2.5-32B (TP2) → Llama-3.3-70B/Qwen2.5-72B (TP4); last resort reduce n_features 10→6. Only passing models proceed to Stage 4. │
│ │
│ Stage 2 — Generator fixes (src/data/generator.py) + re-gate │
│ │
│ - 2a. Sample sparse_interaction.threshold from a target base rate U(0.35, 0.65) via quantile of a 10k-sample product probe (v1 left it 0.0 — never assigned). │
│ - 2b. missing_feature: replace dropped causal feature with an independent marginal redraw (unich degenerates the product rule to constant label). │
│ - 2c. Covariate shift: restrict shifted features to 0–7; preserve label base rate by rejection-sampling the shift vector (accept if probe base-rate delta ≤ 0.03, ≤50 tries) — keeps it a pure P(x) intervention. │
│ - 2d. Extrapolation: per-feature range extension (sample 2–3 causal features from |x| ∈ [2, 4]ar — genuinely outside demonstrated support, not erased by standardisation, label-relevant for all │families. If the gate still shows no degradation profile, explicitly demote to a labelled nuisance-control column in the RQ2 grid. │
│ - 2e. Spurious feature: continuous construction X8 = (2y−1)·d + N(0,1) with d = Φ⁻¹(strength) th exactly; strength 0.80–0.90 (spec intent: more reliable than the causal signal, but breakable, │not a label photocopy). │
│ - 2f. Pool provenance: id_pool (demos from ID, queries from shifted env) is the new headline —setting and the only one where "shift-aware selection" is meaningful; matched_pool (v1 behaviour) │retained as secondary. No generator change needed — task parquets already store all splits; NB06 gains a pool_provenance loop. │
│ - Gate S2 — regenerate suites into data/synthetic_v2/, re-run NB04 gate on a stratified sampleion and all 6 envs (v1 excluded the held-out family — how 2a/2b slipped through). Pass: ≥80% show │the degradation profile, no env with label-1 rate outside [0.2, 0.8], and NB00 re-run against data/synthetic_v2 reports overall_pass: true with zero skipped checks (same check functions, imported from │
│ src/evaluation/data_diagnostics.py). Freeze only on pass. │
│ │
│ Stage 3 — SATA fixes + retrain │
│ │
│ - 3a. Targets (src/models/sata_targets.py): score = 2.0·same_regime + 1.5·is_counter_spurious ∧ ¬same_regime). Label-match term removed (it was the leak channel); purely structural │supervision. Temperature sweep {0.5, 1, 2}. │
│ - 3b. Standardisation: single shared standardise used by both sata_train.\_task_batch and sata_tats computed on the full pool BEFORE pre-filtering (fixes sata_alone vs best_protocol_sata │inconsistency). │
│ - 3c. Training provenance: \_task_batch(..., pool_env="id") — demos from ID, queries from the shed-pool batches so matched evaluation isn't OOD for SATA. │
│ - 3d. Balanced top-k: sata_select and proxy take top-k/2 per label class (confound correction — demo label balance shifts the LLM's output prior; also kills the proxy's single-class exploit). │
│ - 3e. Proxy validation: all 6 envs with id_pool provenance; checkpoint/stop on worst-env accurss fallback; optional feature-8-dropped proxy variant as a shortcut diagnostic. │
│ - Gate S3a (before any training): select top-k directly by the target function (no model) and run the proxy — oracle targets must beat the best protocol on worst-env accuracy, else redesign targets first. (This gate would have caught the v1 label leak immediately.) │
│ - 3f. Retrain SATA + query-agnostic on v2 tasks (existing train_sata machinery, checkpoints to models/v2/). Gate S3b: trained SATA ≥ best protocol on worst-env proxy AND > query-agnostic. On fail: RQ4 is a documented │
│ negative with the full evidence chain; Stage 4 proceeds regardless (RQ1–RQ3 don't depend on it │
│ │
│ Stage 4 — Full re-run │
│ │
│ 1. NB06: v2 config, 5 seeds, pool-provenance loop, chat prompts, ordering, calibration, new scd: 10 conditions × 5 seeds × k=8 × id_pool × 6 envs × 32 queries × 100 test tasks ≈ 0.9M │prompts/model (×2 calibration). Secondary at 1 seed: full 200 tasks, matched_pool, k=16, held-out family. ~25–50 h per 8B model on one GPU with prefix caching; shard via existing SATA_MODEL_FILTER/SATA_RUN_SHARD + │
│ scripts/merge_sharded_results.py. Any Gate-S1-escalated model runs the headline grid only. │
│ 2. NB02/03 (real arm) re-run with the fixed inference layer (~24–48 h across 2 GPUs) — chat template + calibration may change RQ1 materially too. │
│ 3. NB07/08: point at results/v2/; add v1-vs-v2 comparison figure, Stage-0 diagnostics section,pool-provenance dimension in the RQ2 heatmap. │
│ │
│ Gate summary │
│ │
│ ┌─────────────────┬──────────────────────────┬───────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────── │─┐ │
│ │ Gate │ After │ Criterio │ On fail │ │ │
│ ├─────────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────── │─┤ │
│ │ S0 data gate │ Stage 0b, re-run each │ data_gate_report.json overall_pass (zero skips │ fix data before any GPU spend; on v1 data a FAIL flagging the five known defects │ │ │
│ │ (NB00) │ data regen │ │ is the expected acceptance result │ │ │
│ ├─────────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────── │─┤ │
│ │ S1 pilot │ Stage 1 │ random-8 ID ≥ 0.60; calibrated zero-shot class- │ escalate 32B → 70B → reduce n_features │ │ │
│ ├─────────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────── │─┤ │
│ │ S2 generator │ Stage 2 │ ≥80% of stratified sample (incl. sparse_interaconstant-label │ tune generator params, re-gate │ │ │
│ │ │ │ cells; NB00 passes on v2 data │ │ │ │
│ ├─────────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────── │─┤ │
│ │ S3a target │ before training │ oracle-target selection ≥ best protocol (worst- │ redesign targets before training │ │ │
│ │ oracle │ │ │ │ │ │
│ ├─────────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────── │─┤ │
│ │ S3b Gate 2 │ Stage 3 │ SATA ≥ best protocol AND > query-agnostic (wors │ RQ4 = rigorous negative; Stage 4 still runs │ │ │
│ └─────────────────┴──────────────────────────┴───────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────── │─┘ │
│ │
│ Critical files │
│ │
│ - New src/evaluation/data_diagnostics.py + notebooks/00_data_diagnostics.ipynb — Stage 0b (check functions + visual audit; reuses src/data/tableshift_loader.py loaders, SyntheticTask.\_apply_rule rehydration, NB08 │
│ plotting conventions) │
│ - src/data/generator.py — Stage 2 (threshold sampling, missing-feature redraw, covariate rejection-sampling, extrapolation, spurious reparameterisation) │
│ - src/models/sata_targets.py, src/models/sata_train.py — Stage 3 (targets, standardisation, po) │
│ - src/inference/prompts.py + new src/inference/chat.py, src/inference/calibration.py — Stage 1 │
│ - src/selection/sata_select.py (balanced top-k, standardise-then-filter) + new src/selection/o │
│ - src/data/serialisation.py (number formatting) │
│ - notebooks/04, 05, 06 re-runs; configs/v2.yaml; src/utils/config.py; src/utils/results_schema │
│ - New: scripts/diagnose_v1.py, scripts/pilot_stage1.py │
│ │
│ Verification │
│ │
│ - Stage 0 outputs must confirm the predicted mechanisms (SATA demo sets ≈ single-class tracking feature 8; zero-shot class-1 rate 96–99%) — if they don't, revisit the diagnosis before changing code. │
│ - NB00 acceptance test: run against v1 data on HPC — the gate must FAIL flagging exactly the ft-label cell, covariate label-rate drift, spurious strength > 0.95, dead counter-spurious signal, │inert extrapolation). Locally (real arm only, off data/tableshift_raw_cache), synthetic checks must record as SKIPPED with no exceptions; SATA_CONFIG=configs/v2.yaml must switch the audited paths with zero cell edits. │
│ - Each stage has its gate (S1, S2, S3a, S3b) with pre-registered pass criteria and documented m in order; never spend the next stage's compute before the gate passes. │
│ - Unit-level checks: generator asserts no env has label-1 rate outside [0.2, 0.8] per task; prompt snapshot test (one rendered chat prompt per model committed as a fixture); sata_select test that selected sets are │
│ label-balanced and invariant to pre-filter standardisation. │
│ - Final: NB08 regenerates all figures from results/v2/; v1 figures preserved via the v1-results tag. │
│ │
│ Timeline │
│ │
│ Stage 0 (~1 day, CPU) + Stage 0b NB00 (~2.5–3 days, real-arm half testable locally) → Stage 1 code (1–2 days) + Gate S1 (hours GPU) → Stage 2 (1–2 days + gate, NB00 re-run on v2 data) → Stage 3 (2–3 days incl. retrain) → Stage 4 (days of GPU, sharded). Stages 0b, 1 and 2 code can proceed in parallel; Stage 3 nee
