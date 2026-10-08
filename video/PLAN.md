# Plan: Manim explainer video of how the experiment is run (`video/`)

## Context

The user wants a silent explainer video (no voiceover or transcript yet), built with Manim, showing how the experiment is conducted. It should cover every section of `docs/generator_spec` Parts I–VI, from the Overview to Evaluation and statistics, then the P4 headline results. SATA, the probes and the compute orchestration are left out.

The user's choices:
- **Depth:** visuals plus key equations.
- **Extras:** RQ3 faithfulness, statistics and P4 results.
- **Style:** dark background.
- **Coverage:** every section of Parts I–VI, about 10 minutes.

A voiceover may come later. So the video is built from short sections that can be rendered separately, with holds that are easy to retime. Chapters follow the spec's parts, so any chapter can be shown or cut in a talk.

Every example on screen is a real P4 artefact: task `eval_0000` (loan domain, linear rule), its prompt, scores, demonstration picks and priors, the real bootstrap and the contrast results.

## Deliverables

All files go in `/Users/chenuka/Documents/USYD/LLM-ICL-OOD-Honours/video/`, inside the parent git repo. Nothing is committed unless the user asks.

| File | Purpose |
|---|---|
| `PLAN.md` | This plan, written first (the user's plan-file-first rule) |
| `experiment_explainer.py` | The video, with one function per section. Each section also has its own Scene class (`S00Title` … `S21Closing`) for fast iteration. `FullVideo` runs them all with `self.next_section(key)`, so `--save_sections` also writes per-section clips for a later voiceover or for slides. Render commands go in the module docstring. |
| `extract_data.py` | Run once with `sata-project/.venv/bin/python` (it has pandas, pyarrow and scipy). It reads the P4 outputs, writes `explainer_data.json`, and asserts the invariants listed below. The video environment needs no pandas. |
| `explainer_data.json` | The real values shown on screen (small), plus the citation strings |
| `pyproject.toml`, `uv.lock`, `.python-version` | Python 3.12 and `manim==0.21.0`, the latest release. It needs Python ≥3.11 and writes video through PyAV, so no ffmpeg is needed. |
| `.gitignore` | `media/`, `.venv/`, `__pycache__/`, `frames/` |

## Setup (one-time)

1. **Install pkg-config.** `brew install pkgconf`. pycairo 1.29.1 ships only an sdist for macOS, and building it needs pkg-config to find Homebrew cairo. Already present: cairo 1.18.4, pango 1.58.0, and MacTeX's `latex` and `dvisvgm`.
2. **Create the uv project.** In `video/`, run `uv init --bare`, `uv python pin 3.12` and `uv add "manim==0.21.0"`, then `uv run manim checkhealth`.
3. **Check the fonts.** Menlo, Avenir Next and Helvetica Neue must appear in `manimpango.list_fonts()`.

## Real data (`extract_data.py` → `explainer_data.json`)

**Sources**
- `sata-project/data/synthetic/v3/eval/` (manifest and task parquets).
- `results/v3/synthetic/h200/p4/`: `grid.parquet`, `grid.s0.names.json`, `rq3*.parquet`, `contrasts.csv`, `covariate_probes.csv`. Use this copy, not the stale `results/v3/synthetic/p4/`.
- `results/v3/synthetic/h200/p3/` (pool priors).
- `results/v3/synthetic/p3/` (screen and naming gate).
- `docs/references.bib` (author–year strings for on-screen credits).

**Task `eval_0000`**
- **Roles:** displayed rule columns f3 (+), f9 (+), f7 (−). The spurious feature is shown as f4 and the noise feature as f8.
- **Parameters:** s = 0.880, σₜ = +1. Also extract the column permutation, the covariate shift δ, and the names under each naming.
- **Pool scatter:** all 256 rows, as rule score cᵀx_C against σₜ·f8 on the generator scale, plus labels, |f8| and agreement flags.
- **Measured prior:** the zero-shot margin on each pool row under each naming, and the conflict Cₜ.

**Demonstrations**
- Seed-0 picks of every strategy for query 14.
- counter_prior's picks are taken under flipped names. For `eval_0000` it is inactive under abstract and aligned names (conflict 0.449).
- The counter_spurious |f8|-matched pairs.

**Prompt and scores**
- The label_diversity seed-0 prompt: the system message and 8 demonstration lines, verbatim.
- Query 14: its log-probabilities and content-free log-probabilities. p₁ = 0.776 becomes 0.164 after calibration; the true label is 0.
- The cell's 20 ID margins and labels (AUROC 0.56).

**Other examples**
- `eval_0012`'s tree leaf table.
- The surrogate slopes (abstract slopes all positive, 0.22–0.89) and the G3 numbers.

**C1c bootstrap**
- The per-task values and the 2,000 bootstrap draws.
- The draws are recomputed with the same loop and seed as `hierarchical_contrast` (reusing `_block_value` and `COLUMN_STATISTICS` from `src/evaluation/bootstrap.py`). The input comes from `grid_blocks` in `src/evaluation/analysis.py`.
- They must reproduce 0.204 [0.131, 0.280].

**RQ3**
- From `analysis.py`: `behavioural_importance`, `directional_sensitivity`, `dfi`, `correctness_rho` and `self_report_rho`.
- The role × naming reliance table: abstract spurious 0.141 and rule features 0.009 per feature; aligned rule features 0.241; flipped −0.047.
- One hot-deck example and one nudge example (query 2).
- One self-report ranking against the behavioural ranking.

**Asserts**
- The cell's AUROC is 0.56.
- `holm()` reproduces the Holm p-values in the CSV.
- The histogram sums to 2,000.
- C1c is above 0 in 22 of 24 tasks.
- The pool has 256 rows.
- The label_diversity picks equal the prompt's rows.
- The counter_spurious set has agreement 0.5.

## Script design

**Palette** (Okabe–Ito, on background `#0E1117`). Each role colour is always paired with a text badge.

| Role | Colour |
|---|---|
| Label 1 | `#56B4E9` |
| Label 0 | `#E69F00` |
| Load-bearing | `#009E73` |
| Distractor | `#6E7781` |
| Spurious | `#CC79A7` |
| Noise | `#C9D1D9`, dashed |
| Accent | `#F0E442` |

**Text**
- `Text` (Pango, Avenir Next or Helvetica Neue) for prose; Menlo for prompt lines.
- `MathTex` for one to three equations per section. The default template loads amsmath and amssymb; use `\mathbf{1}`.
- Australian spelling throughout.

**Helpers**
- **Layout:** `fit()` keeps every object inside x ∈ [−6.6, 6.6] and y ∈ [−3.4, 3.4]. A header shows the part, the spec section number and the title, e.g. "II · §12 The spurious feature". Captions are at most 14 words, and borrowed ideas get a small grey credit line (author and year from `references.bib`).
- **Chapter map:** the spec's pipeline diagram (§3) reappears at the start of each part, with the active box highlighted.
- **Visual builders:** `prompt_line()` with role underlays; `gauss_pair()`, driven by ValueTrackers for d and σₜ through a single `always_redraw`, with `DecimalNumber` readouts; dot clouds as one VGroup; hand-made bars, ROC staircase, histogram, forest plot and DAG.
- **Cleanup:** `clear()` removes updaters before fading.

**Pacing.** `beat(caption, *anims)` plays the caption with its animation, then holds for the word count × `PACE`. That one constant retimes the whole video for a voiceover.

**Determinism.** A NumPy `default_rng` per section. Raise `config.max_files_cached`, and render the final cut with `--disable_caching`.

**Layout check.** With `LAYOUT_CHECK=1`, each hold warns about text that is off the frame or overlapping.

## Storyboard (≈10:20; durations in seconds)

### Part I — Overview (≈50 s)
| # | Section | s | On screen |
|---|---|---|---|
| 00 | Title | 6 | Working title (a constant): "Choosing demonstrations under distribution shift"; "How the experiment is run"; author line; feature-box motif |
| 01 | §2 Research questions | 26 | ICL as a contest between prior and demonstrations: following the context over priors emerges with scale, and 7–8B models follow contradicting demonstrations only partially. RQ1–RQ3 with H1–H3 in one line each. Chip: "RQ4: a learned selector (SATA), P5" |
| 02 | §3 Pipeline overview | 18 | The spec's diagram, drawn box by box: generator → per-task data → selection (mechanism × composition) → naming and prompt → runner → metrics and statistics, with the measured-priors side box. SATA and the probes are greyed out as later phases |

### Part II — Data generation (≈165 s)
| # | Section | s | On screen |
|---|---|---|---|
| 03 | §7–9 Configuration, roles, task sampling | 25 | Ten feature boxes by role: 3 load-bearing (from f0–f7, directions ±), 5 distractors, f8 spurious, f9 noise. 24 evaluation tasks (12 linear, 12 tree; loan/medical) and 12 pilot tasks (P2 only), each with its own seed stream. Column shuffle with eval_0000's permutation, so the spurious column lands on f4; P(position) = 1/10 means there is no positional shortcut |
| 04 | §10 Rule families | 25 | **Linear:** p = σ(cᵀx_C), y ~ Bernoulli(p), \|c_j\| ~ U(1, 2); Bayes accuracy 0.82 ≈ s, so the shortcut competes. **Tree:** signed majority, depth 3, 8 admissible tables (eval_0012's leaves); Bayes accuracy 0.98 > s, so the shortcut is dominated. Directions s_j are well defined. Excluded: threshold (every s_j = +1 would confound naming) and sparse interaction (no direction) |
| 05 | §11 Environment generation | 40 | The five-step skeleton: draw X ~ 𝒩(0, I) → perturb → rule → f8 and f9 → label noise. **id.** **Covariate:** the real eval_0000 cloud slides along constant-score lines; cᵀδ = 0, and for trees P(y = 1) stays ½, so the label rate is unchanged. **Spurious reversal:** s → 1 − s (as in Colored MNIST). **mechanism:** the rule is inverted, so with ID demonstrations only f8 keeps its relation — a placebo, which is why concept shift is done with names. extrapolation and missing_feature are in the generator but not evaluated |
| 06 | §12 The spurious feature | 35 | 𝒩(±d, 1) with shaded agreement; X₈ = σₜ(ỹd + ε), d = Φ⁻¹(s); P(sign σₜX₈ = ỹ) = s, with a sweep of s from 0.80 to 0.90. **Why σₜ is random:** Qwen's numeric prior. Summing the features gives AUROC 0.628, or 0.511 without f8, against Qwen's zero-shot 0.633; 12 tasks each way. **Scale:** Var = 1 + d², so the SD is 1.31–1.63; hence z-scoring with pool statistics, with agreement read on the generator scale |
| 07 | §13–15 Label noise, datasets, causal graph | 40 | Label noise η = 0.02, applied after f8 is drawn from y_clean, plus the agreement flags. The split diagram: id 356 rows → pool 256 + test 100; 100 rows each for covariate and reversal → standardise with pool statistics (so covariate shift shows as an offset in z-units) → permute → one parquet per task. 20 queries per environment (10 + 10). No cross-task mixing. The causal DAG (X_C → y_clean → f8, y_clean → y; f8 is anticausal) |

### Part III — Presentation (≈60 s)
| # | Section | s | On screen |
|---|---|---|---|
| 08 | §16 Naming protocol | 35 | Lexicon: antonym pairs (p_m, q_m) plus weak names. Two domains: loan (1 = approved) and medical (1 = high risk). Definitions of abstract, aligned and flipped. A real eval_0000 row's names morph abstract → aligned → flipped; only f3, f9 and f7 change. **Identification proposition:** the aligned and flipped prompts differ only in those three names. **Concept shift through naming:** the mechanism environment under aligned names equals the flipped condition |
| 09 | §17–18 Serialisation and prompts | 25 | Format `name: value; … -> label` (z-scores, 2 dp, fixed column order). The abstract and loan system messages. The real chat-template prompt. The base model gets a raw completion ending `-> ` with a trailing space. "0" and "1" are single tokens (15, 16) |

### Part IV — Demonstration selection (≈90 s)
| # | Section | s | On screen |
|---|---|---|---|
| 10 | §19–20 Selection engine and strategies | 40 | Mechanism × composition: free or balanced (balanced fixes the label counts, per the proposition). The real pool scatter with k = 8; for each strategy, its real picks are ringed with a one-line definition: zero-shot, random, label_diversity, feature_range, rule_diversity, counter_spurious, similarity, counter_prior (flipped names). counter_prior_matched is marked exploratory |
| 11 | §20.1 Counter-spurious must match \|f8\| | 25 | Balancing the 4 (label, agrees) cells naively gives agreement ½ but leaves Cov(f8, 2y − 1) > 0, because agreeing rows lie further from 0 (E\|f8\| 1.31 against 0.52; corr 0.33 at s = 0.85). Shown on the real pool's \|f8\| histograms. Matching each disagreeing row to the nearest-\|f8\| agreeing row gives Cov = 0 (measured corr 0.005, agreement 0.500) |
| 12 | §21–22 Seeding, reuse, pairing, leakage | 25 | Seed streams (task, strategy, seed, purpose), never including the model, naming or environment. One demonstration set is reused for every query, environment and naming, so contrasts are paired within a prompt. Order is shuffled per set. The shuffled-label control. Demonstrations always come from the ID pool. **No test leakage:** selectors, standardisation and priors see only the pool |

### Part V — Model inference (≈97 s)
| # | Section | s | On screen |
|---|---|---|---|
| 13 | §23 Models and the answer format | 18 | Qwen2.5-7B-Instruct (primary), Qwen2.5-7B base (reduced grid), Llama-3.1-8B-Instruct (paused). PyTorch, bf16, batch 1. Label mass P("0") + P("1"): Qwen 0.996; Llama 0.009–0.11, because it starts "To infer the rule…" |
| 14 | §24, §26 Scoring and calibration | 35 | The real prompt → Qwen → log P("0") and log P("1"), computed in fp32. m = ℓ₁ − ℓ₀ and p₁ = σ(m) = 0.78, so ŷ = 1, but the true label is 0. A content-free "N/A" query gives (c₀, c₁); then p̃₁ = (p₁/c₁)/(p₁/c₁ + p₀/c₀), equivalently m̃ = m − log(c₁/c₀), so p̃₁ = 0.16 ✓. Every query shifts equally, so ranks within a set are unchanged |
| 15 | §25 Prefix caching | 14 | The prefix is run once; each query suffix runs against the cached keys and values, then the cache is cropped. This is exact under causal attention when tok(P‖S) = tok(P)‖tok(S). Check: AUROC moved 0.0025, under the 0.005 rule |
| 16 | §27 Measuring priors | 30 | **Surrogate:** random profiles, ridge slope per name. It found a numeric prior — every abstract slope is positive — and names weaken it rather than reverse it. **G3:** R² 0.20 and 0.28 < 0.5 fails; separation 0.164 passes. Hence the measured prior: the zero-shot margin per pool row, median-centred (the eval_0000 pool coloured by it). **Conflict** Cₜ(N) = 1 − agreement; counter_prior is active when Cₜ > ½ |

### Part VI — Evaluation and statistics (≈105 s)
| # | Section | s | On screen |
|---|---|---|---|
| 17 | §28 Metrics | 30 | The metric table: AUROC (primary), calibrated BA (reported alongside), BA, macro-F1, prior agreement, shift gap Δ = ID − OOD, DFI. The real cell: a 10×10 grid of pairs gives 56/100, so AUROC = P(score⁺ > score⁻) = 0.56, then the ROC staircase. Why AUROC: calibration put every query on one side in 36% of random cells |
| 18 | §29 Design and inference | 40 | Units: task, query, seed. The grid: 24 × 3 namings × 3 environments × 20 queries × strategies × 3 seeds. Why more tasks: Var = σ_T²/T + σ_W²/(Tn). Hierarchical bootstrap with the real C1c draws (tasks, then 10 + 10 queries), the histogram and [0.131, 0.280]. Sign test (22/24). Holm over the 10 frozen, one-sided contrasts |
| 19 | §30 Faithfulness and reliance (RQ3) | 35 | π_true; π_behav from hot-deck replacement (a real query's margin drop); π_self from the ranking prompt. Correctness ρ(π_true, π_behav) and consistency ρ(π_self, π_behav). Nudges g_j = m(x + 0.5e_j) − m(x − 0.5e_j) and DFI = Pr[sign g_j = s_j], with a flipped-name example that follows the name. 381 suffixes per unit |

### Results and closing (≈56 s)
| # | Section | s | On screen |
|---|---|---|---|
| 20 | P4 headline results | 40 | Forest plot of the 10 contrasts with intervals and Holm p. Highlighted: C1c +0.204, C2e +0.111, C3a +0.123. C1b +0.115 (p 0.063) "just misses"; C2c's interval is below 0. Reliance by naming: abstract → shortcut 0.141; aligned → rule 0.241; flipped → −0.047. Self-reports are unfaithful. Covariate follow-up (exploratory): scores rise 0.44–1.21 logits without reordering; the variance shift hurts only under flipped names (−0.09 to −0.13); removing f8 hides no covariate harm |
| 21 | Closing and sources | 16 | Pipeline recap; "Next: SATA (P5), probes (P6)"; a sources card with the key borrowed ideas |

## Accuracy notes, from checking the data

- **Seeds.** Say "3 seeds per randomised strategy": similarity and zero-shot ran with seed 0 only.
- **Naming the shortcut.** "f8" is the generator's name for it. After the shuffle, call it "the spurious column (shown as f4 here)".
- **Rule-feature reliance.** 0.009 is the mean drop per rule feature, not the three together; notebook 06 averages over feature rows.
- **Bootstrap.** Within each task, positives and negatives are resampled separately.
- **System message.** It is identical under aligned and flipped names; the abstract one differs.
- **Exploratory items.** counter_prior_matched and the covariate probes are labelled exploratory, outside the frozen family.

## Side fix (records)

`hiccups/18-…md:14` and `docs/research_plan.md:342` say the three rule features cost 0.009 "together". Change both to "0.009 per rule feature on average".

## Verification

1. **Data.** `extract_data.py` runs and every assert passes.
2. **Per-section renders.** Each section renders at `-ql` without errors, and `LAYOUT_CHECK=1` reports no off-frame or overlapping text.
3. **Contact sheets.** A scratchpad tool (PyAV) grabs a frame every 2 s from each `-ql` clip. I review them as images for overlaps, legibility and colour roles.
4. **Full cut.** `FullVideo -qm --save_sections` comes to 9.5–11 min in total, and each section matches its standalone scene.
5. **Facts.** Every number and definition on screen is checked against the spec (Parts I–VI) and `contrasts.csv`. Spelling is Australian, and nothing from SATA, the probes or the H200 setup appears.
6. **Final render.** `uv run manim -qh --save_sections --disable_caching experiment_explainer.py FullVideo` produces `media/videos/experiment_explainer/1080p60/FullVideo.mp4` and the per-section clips. Send the file to the user.

## Outcome (2 October 2026)

**What was rendered**
- `media/videos/experiment_explainer/1080p60/FullVideo.mp4`: 1920×1080 at 60 fps, 9.8 minutes, about 51 MB.
- One clip per section in `.../1080p60/sections/`, with `FullVideo.json` listing their names and durations.
- 22 sections, 370 animations.
- Each section in the full video matches its standalone render to within a frame.

**Checks**
- **Data:** `extract_data.py` passes every assert. It reproduces C1c's bootstrap interval exactly, 0.204 [0.131, 0.280], and the chat template matches the frozen prompt snapshot.
- **Layout:** `LAYOUT_CHECK=1` reports nothing for the full video.
- **Facts:** every on-screen fact was checked against the spec. Four statements were corrected during the build:
  - mechanism is a placebo;
  - the naming equivalence holds up to a coefficient rescale;
  - the switch to AUROC came early in P4;
  - the order is shuffled per set.

**Records corrected along the way**
- **Covariate score shift.** It is +0.32 to +1.21 logits, not +0.44 to +1.21: the lower bound had ignored flipped names. Fixed in hiccups/18, the hiccups README, the spec (PDF rebuilt) and the research plan.
- **Rule-feature reliance.** The 0.009 drop is per rule feature, on average, not for the three together.

**To re-render**, see the commands in the docstring of `experiment_explainer.py`. If the P4 outputs change, rerun `extract_data.py` first.

## Narration (2 October 2026, at the user's request)

**Voice.** Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0) runs locally with the British voice `bf_emma`, Kokoro's highest-graded British voice, at speed 1.05.
- `voice.py` synthesises each line to WAV and caches it by its text, voice and speed.
- `uv run python voice.py` reports the duration of every section; the narration totals about 10.7 minutes.
- `SAY_AS` respells words for the voice only: Qwen becomes "Kwen", MNIST becomes "em-nist".

**Script.** `narration.md` has 109 lines, one per narrated step: every captioned step, each chapter card, and each strategy in section 10. It is written in Australian English, for the ear.

**Wiring.** Each section gets a `Narrator`.
- A narrated step starts its line through Manim's `add_sound`.
- The next line, or the end of the section, waits for the current line to finish, so the voice sets the pace.
- Each line is also written as a subtitle, an `.srt` beside the video.
- If a section's line count and narrated steps differ, the render stops with an error. A `--dry_run` checks all sections in a few minutes.

**Commands.**
- Narration is the default.
- `NARRATE=0` renders the silent version.
- `VOICE=bm_george` or `VOICE_SPEED=1.1` change the voice or its speed.

**Dependencies added.**
- `kokoro==0.9.4`, `soundfile`, `transformers>=4.45`. The resolver otherwise picked an old transformers whose tokenizer fails to build.
- spaCy's `en-core-web-sm`, which Kokoro's English G2P needs; pinned so `uv sync` keeps it.

**Rendered (2 October 2026).**
- `1080p60/FullVideo.mp4`: 12.1 minutes, AAC stereo, 70 MB.
- `FullVideo.srt`: 109 subtitles.
- `sections_narrated/`: the 22 clips with their narration, made by `add_section_audio.py` from the full video's audio, because Manim writes section clips without audio.
- `FullVideo_silent.mp4`: the silent cut, kept alongside.
- **Sync** was checked by grabbing frames at subtitle times; each matched its visual.
