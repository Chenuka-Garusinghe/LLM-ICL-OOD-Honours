# 21 — SATA's targets preferred label-noise rows and taught the shortcut backwards

**When:** 10 October 2026, P5 gate S3a, before any SATA training · **Status:** decided by the user the same day. The main SATA-v2 learns the neutral matched targets (gate S3a: 0.640 against 0.555), and the planned targets with the label-noise fix are kept as the `sata_designed` ablation. After the first training runs, SATA draws its demonstrations in proportion to its scores instead of taking the top k (my decision). On 12 October it was checked on Qwen-Instruct: top-k is no better (see Result).

## What happened

SATA learns from target scores computed from the generator's ground truth (spec, Part SATA). For each candidate row and query:
- **+2** for a row in the query's regime;
- **+1.5** for a row whose shortcut disagrees with its label ("counter-spurious");
- **−1.5** for a row whose shortcut agrees and sits outside the query's regime.

Gate S3a checks the targets *before* training. Take the top 4 rows of each label by target score, with no model, and fit an XGBoost classifier on those 8 rows. Its accuracy on the query, in the worst of ID, covariate and spurious reversal, must reach the best hand-designed strategy's. Two problems surfaced, one inside the other.

## How we found it

**1. Label-noise rows scored the maximum.** A label-noise row's shortcut is generated from its *true* label, so against its observed label it disagrees. It therefore counted as counter-spurious, and inside the query's regime it scored 3.5, the highest score possible. The targets as designed picked label-noise rows in 9.0% of their demonstrations, against 1.9% for random selection (100 validation tasks).

**2. The fix still taught the shortcut backwards.** With the bonuses restricted to clean rows, the targets still prefer rows that break the shortcut. So in the chosen sets the shortcut agrees with the label in only 25% of rows, against about 84% in the pool. On tree tasks the figure is **7%**. Within the query's leaf all clean rows share one label, so the "counter-spurious first" rule fills that class entirely with rows that break the shortcut. The demonstrations then show the shortcut perfectly reversed.

**3. A first probe hid problem 2.** On 30 tasks without the feature shuffle, the fixed targets looked strong (worst environment 0.628). With only 8 rows, XGBoost often finds several features that separate them equally well and takes the first one. Unshuffled, the rule features come before the shortcut, so the reversed shortcut was rarely used. With the evaluation's per-task shuffle (the column-role permutation), it often is, and ID accuracy fell from 0.650 to 0.556. The gate uses the shuffle, as evaluation does.

**Gate S3a, 100 validation tasks** (`results/v3/synthetic/p5/gate_s3a.json`):

| Method | ID | Covariate | Spurious reversal | Worst | Shortcut agrees | Label-noise rows |
| --- | --- | --- | --- | --- | --- | --- |
| Targets, label-noise fix | 0.556 | 0.585 | 0.671 | **0.556** | 0.25 | 0.000 |
| counter_spurious | 0.581 | 0.555 | 0.595 | 0.555 | 0.50 | 0.000 |
| feature_range | 0.578 | 0.575 | 0.507 | 0.507 | 0.66 | 0.072 |
| Targets as first designed | 0.476 | 0.522 | 0.657 | 0.476 | 0.24 | 0.090 |
| random | 0.550 | 0.533 | 0.468 | 0.468 | 0.84 | 0.019 |
| similarity | 0.684 | 0.680 | 0.462 | 0.462 | 0.76 | 0.036 |
| label_diversity | 0.661 | 0.651 | 0.431 | 0.431 | 0.84 | 0.020 |
| rule_diversity | 0.680 | 0.684 | 0.368 | 0.368 | 0.85 | 0.000 |

The fixed targets pass, by 0.0003, which is a tie with counter_spurious. As first designed they fail by 0.08.

## Why it matters

- **This is the same flaw P1 fixed in the hand-designed counter_spurious strategy** (`hiccups/` 04, spec "counter-spurious selection must match on |f8|"). Over-sampling rows that break the shortcut does not remove the shortcut, it reverses it. That helps under spurious reversal and hurts on ID.
- **A selector trained on these targets can at best tie the best hand-designed strategy on the proxy,** and learning is never perfect. For a model that reads the shortcut from its demonstrations, as Qwen does under abstract names (RQ3: replacing the shortcut costs 0.141 of margin), the ID cost could be larger.

## What we did

1. **Label-noise rows earn no bonus** (`src/models/sata_targets.py`). This is a bug fix, applied.
2. **SATA is training with the planned targets** plus that fix, as the user asked ("train sata as planned … and see the results"). Both variants run on Apple Metal; checkpoints go to `models/sata_v2*.pt`.
3. **A neutral target was tested on the same data** (scratchpad prototypes; not in `src/` until decided).
   - The **matched set** mirrors counter_spurious with the regime preference. Within each label it takes 2 clean rows that break the shortcut, query's regime first, and pairs each with the clean agreeing row nearest in |shortcut|, query's regime first.
   - A row's target is its chance of being in that set, computed exactly in 0.17 ms per query. SATA's architecture, loss and top-k selection are unchanged.

| Targets (oracle, 100 tasks) | Worst environment | Tree tasks | Linear tasks | Shortcut agrees |
| --- | --- | --- | --- | --- |
| Matched sets, drawn directly | 0.648 | 0.762 | 0.529 | 0.50 |
| **Top 4 per label by matched pick rate** | **0.640** | 0.742 | 0.538 | 0.60 |
| Planned targets, label-noise fix | 0.556 | 0.556 | 0.532 | 0.25 |
| Counter bonus only outside the query's regime | 0.568 | | | 0.61 |
| No shortcut terms at all | 0.482 | | | 0.85 |

   - The pick-rate target beats every hand-designed strategy by 0.085. Linear tasks stay hard for every method (about 0.53–0.56); the gain is on tree tasks.
   - Earlier I worried that per-row scores could not represent a half-and-half set. That turned out to be wrong: the agreeing partners chosen are concentrated, so their pick rates are high too.

## The decision

The user chose the matched target for the main SATA and kept the planned run as an ablation ("Matched + keep planned").
- `sata` and `sata_agnostic` learn the matched targets (`matched_target_scores`).
- `sata_designed` learns the planned structural targets with the label-noise fix. It resumed its run from epoch 2.
- The planned blank-query run was stopped, because it is in no contrast.
- The RQ4 contrasts (R1–R6) were written before any SATA result, so choosing which targets `sata` uses still came before the results.

**Gate S3a with the main targets:** 0.640 against counter_spurious's 0.555, a clear pass (`gate_s3a.json`, `oracle_matched`).

## The first training runs: learning the targets brought the reversal back

All three models trained on Apple Metal with the plan's top-k selection. As the training loss fell, the proxy fell too:

| Model | Best worst-environment proxy | Epoch 10 |
| --- | --- | --- |
| `sata` (matched targets) | 0.539 (epoch 1) | 0.475 |
| `sata_agnostic` (matched, blank query) | 0.428 (epoch 0) | 0.288 (ID 0.35, spurious reversal 0.73; stopped early) |
| `sata_designed` (planned targets) | 0.514 (epoch 2) | 0.357 |

**Why.** Picked directly, the exact matched pick rates keep each shortcut-breaking row's agreeing partner ranked as high as the breaking row itself. A learned approximation cannot tell *which* agreeing row is the partner, so it spreads the score over many agreeing rows. Top-k then takes the breaking rows first, and the sets show the shortcut reversed again (27% agreement with the latest `sata` weights). This was the worry raised before the gate, and it bites for the learned model even though it did not for the exact targets.

**Test** (50 validation tasks, 25 linear and 25 tree, latest `sata` weights):

| Selection | ID | Covariate | Spurious reversal | Worst | Shortcut agrees |
| --- | --- | --- | --- | --- | --- |
| Top 4 per label by score | 0.478 | 0.507 | 0.657 | 0.478 | 0.27 |
| **4 per label drawn in proportion to the scores** | 0.600 | 0.638 | 0.558 | **0.558** | 0.54 |

**Decision (mine, 10 October, flagged for the user's review).**
- SATA's scores are a distribution over rows (the KL loss trains them as one), so its demonstrations are drawn from that distribution, k/2 per label.
- Its sets use 3 seeds, like label_diversity's.
- The decision is reversible: every epoch logs the proxy under both rules, and the best checkpoint under each is saved (`models/sata_v2*_topk.pt` for top-k).
- All three models restarted from scratch with this validation; the first runs are archived in `models/archive_topk_runs/` and `results/v3/synthetic/p5/archive_topk_runs/`.

## Result

- **Proxy (notebook 04):** drawn SATA-v2 reached a worst environment of 0.652 at epoch 39, against 0.555 for counter_spurious. The RQ4 family on Qwen-Instruct was null (`hiccups/22`).
- **Checked on Qwen, 12 October** (notebook 05, section 13; exploratory). This answers the question "is the null down to drawing?". The top-k checkpoint (`models/sata_v2_topk.pt`) was run with top-k selection, 1 seed, on Qwen-Instruct under all three namings. That took 4 minutes of inference on one H200 (`scripts/run_topk_pod.sh`, about $1.20 with setup).
  - **Top-k is no better than drawing.** Top-k minus drawn, mean of the three environments, 24 evaluation tasks:
    - abstract −0.005 [−0.051, 0.042];
    - aligned −0.029 [−0.067, 0.007];
    - flipped −0.028 [−0.070, 0.016].
    - Every Holm p is 1.00.
  - **R1–R6 repeated with top-k SATA are still null** (every Holm p = 1.00): R1k +0.003, R2k −0.122 [−0.210, −0.044], R3k +0.002, R4k −0.024, R5k −0.030, R6k −0.017.
  - Under abstract names top-k gains on ID (0.595 against 0.553) but loses under covariate and spurious reversal. Its worst environment is 0.523, against 0.552 for drawn SATA.
  - On the 8 LLM-validation tasks top-k is lower under every naming (mean 0.462 / 0.543 / 0.432 against 0.549 / 0.606 / 0.493). That matches the proxy result above.
- **So drawing stands, and the RQ4 null is not an artefact of it.**

## Evidence

- `results/v3/synthetic/p5/gate_s3a.json`, `gate_s3a.parquet` (per task and environment), `gate_s3a.log`.
- `scripts/gate_s3a_oracle_check.py`, `src/models/sata_gate.py` (the proxy comparison).
- `src/models/sata_targets.py`, `tests/test_sata_targets.py::test_label_noise_rows_earn_no_bonus`.
- Top-k check: `results/v3/synthetic/h200/p5/llmval_topk.parquet`, `grid_topk.parquet`, `topk_contrasts.parquet`, `logs_topk/`; `scripts/run_topk_pod.sh`.

## How to tell it

> SATA is trained to imitate target scores computed from ground truth, so the targets themselves were tested first: choose the demonstrations the targets prefer, with no model, and see whether a simple classifier fitted on them generalises across shifts. As first designed, the targets rewarded rows whose shortcut disagreed with their label. That was meant to discourage reliance on the shortcut, but it rewarded label-noise rows most of all, since their shortcut follows the true label. Once those rows were excluded, the targets still reversed the shortcut rather than removing it. On tree tasks 93% of the chosen rows broke the shortcut, which is exactly what the shifted test environment does. A target built like our magnitude-matched counter-spurious strategy instead, half the rows agreeing and half disagreeing, kept the shortcut uninformative and improved the worst-environment proxy from 0.56 to 0.64.

- **Figure idea:** shortcut agreement in the chosen sets (x) against worst-environment proxy accuracy (y), one dot per target version and strategy, with the neutral line at 0.5.
- **Examiner question:** "Why not let SATA learn whatever reduces worst-environment error, rather than hand-built targets?"
  - That is the learned-utility route (5c's datamodels, or the RL selector the user raised; deferred).
  - The structural targets are the cheap baseline. This hiccup shows how much their exact form matters.
