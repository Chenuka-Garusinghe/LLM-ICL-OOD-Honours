# 15 — Contextual calibration often leaves a whole cell on one side of the threshold

**When:** 30 September 2026, early in P4 · **Status:** decided the same day by the user: AUROC becomes the confirmatory statistic

## What happened

The P4 contrast family was frozen before any P4 result existed, with calibrated balanced accuracy (BA) as the statistic for RQ1 and RQ2. Calibration follows Zhao et al. (2021): the model also scores a content-free query (every value "N/A") with the same demonstrations. Each query's margin is shifted by that query's margin before thresholding at 0.5.

A sanity check on the first 64 units (11 tasks, seed 0, abstract names) showed that for plain random demonstrations the calibrated share predicted "1" was 0.727 in every environment. That is 8/11: in 8 tasks every query was predicted 1, whatever its label or environment.

## How we found it

Per cell (one demonstration set, 20 queries):

| Strategy | Cells predicting one class after calibration | Content-free margin minus median query margin | Calibrated BA | AUROC |
| --- | --- | --- | --- | --- |
| random (free label counts) | 36% | −2.97 | 0.497 | 0.575 (ID) |
| label_diversity (balanced) | 0% | −2.10 | 0.529 | 0.551 (ID) |
| counter_spurious | 9% | −1.66 | 0.524 | 0.563 (ID) |

- The content-free query sits 2 to 3 logits below the typical query, so calibration over-corrects towards label 1.
- With random demonstrations the label counts are unbalanced (for example 7 of 8 labelled 1), and the spread of margins within a cell (IQR about 3.4) often does not reach the threshold.
- In those cells BA is exactly 0.5 in every environment, so the ID-minus-OOD gap is forced to 0. AUROC has no threshold and still moves: 0.575 on ID against 0.401 under spurious reversal for random demonstrations.

## Why it matters

- **RQ1 is tested with random demonstrations.** With calibrated BA, the shift gaps of about a third of the cells would be 0 by construction, biasing the RQ1 tests towards "no degradation".
- **It is a finding in itself.** A content-free input is a poor stand-in for numeric rows, for this model at least. Later calibration methods estimate the bias from the test inputs themselves (e.g. batch calibration, Zhou et al. 2024).

## What we did

The user chose AUROC as the confirmatory statistic for C1 and C2 (option 1 of 3). The others were to keep BA, or to use label_diversity as the RQ1 baseline.
- **Why AUROC.** It was already one of the plan's primary metrics and the metric of the P2 and G3 gates.
- **How it is computed.** Within each cell, from the calibrated p1:
  - query-agnostic strategies: calibration shifts every query equally, so this equals the raw AUROC;
  - similarity: each query has its own prompt, so calibration removes each prompt's bias.
- **BA is still reported alongside**, not tested.
- **Disclosure.** The change was made after seeing descriptive results from 11 tasks at seed 0 under abstract names, and before any contrast was computed on complete data. The runs themselves are unchanged.

## Evidence

- `results/v3/synthetic/p4/grid.parquet` (first 64 units).
- The diagnostic above: per cell, the share predicted 1, the content-free margin minus the median query margin, and one-class cells.
- `src/evaluation/analysis.py` (module docstring, `run_contrasts`).

## How to tell it

> We calibrated every prompt with a content-free query, as is standard. For Qwen2.5-7B-Instruct on numeric rows, the content-free query sat two to three logits below a typical query. With plain random demonstrations, a third of the prompts then assigned the same label to every query, which pins balanced accuracy at chance in every environment, and with it any measured shift. We therefore test with AUROC, which compares queries within a prompt and needs no threshold, and report calibrated accuracy alongside.

- **Figure idea:** per cell, the query margins against the content-free margin (a strip plot), random against balanced strategies.
- **Examiner question:** "Did you change the metric because it gave the answer you wanted?"
  - The change was prompted by a measurement failure: a fixed cut-off producing constant predictions. It was decided before any complete contrast was computed.
  - AUROC was already a primary metric in the plan and the gate metric throughout, and BA is still reported for every contrast.
