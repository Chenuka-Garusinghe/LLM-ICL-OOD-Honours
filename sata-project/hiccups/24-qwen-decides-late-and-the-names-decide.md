# 24 — Inside Qwen: the answer forms late, the names decide it, and steering only shifts the bias

**When:** 12 October 2026, P6 on a Runpod H200 · **Status:** result. The gate passed. P6 was taken off hold by the user before the write-up. The design choices were mine (flagged in `docs/research_plan.md`, P6 implementation).

## What happened

P4 and P5 showed in behaviour that Qwen2.5-7B-Instruct follows its "bigger → 1" prior and the feature names more than 8 demonstrations, and that no selector changes this. P6 looked inside the model to see where that happens.

At every layer, the last-position state was read out through the final norm and the two label rows of the unembedding (the logit lens). This was done for six strategies under three namings. Probes and a steering vector were fitted on the residuals.

## How we found it

**Gate P6:** the final-layer logit-lens margin equals the runner margin within 8.6e-6 on all 25,920 rows (bar 1e-3).

**The same prompts on a fresh pod.** 407 of 432 units are bit-identical to the stored P4/5a/5b results. The other 25 differ on almost every row, by up to 3.4 logits, with identical prompts.
- They are spread over particular workers of the earlier runs, mostly zero-shot units, so this is run-to-run GPU numerics.
- Cell AUROC differs by 0.001 on average, and strategy × naming means by at most 0.002.

**Logit lens** (24 tasks, seed 0; AUROC of mₗ against the rule label, within task × environment cells).
- For the first 20 of 28 layers every strategy and naming is at chance (about 0.45–0.55).
- The answer forms in the last 6 layers:
  - aligned names: 0.63–0.67 with demonstrations, 0.59 zero-shot;
  - flipped names: 0.45–0.52 with demonstrations, 0.45 zero-shot.
- **Critical layer** (aligned − flipped > 0.1 with a task-bootstrap interval above 0): 23 for random, label_diversity, counter_prior and SATA-MI; 24 for zero-shot; 25 for SATA.
- Zero-shot already shows a final gap of 0.141, so the split comes from the names. Demonstrations widen it a little: random 0.192.
- The prior-aware strategies only narrow the final gap (counter_prior 0.174, SATA-MI 0.119, SATA 0.104), the flattening seen in their behaviour. None moves the critical layer earlier.

**Linear probes** (random demonstrations, 4 folds by task).
- Qwen's zero-shot prediction is decodable from layer 2 on: AUROC 0.80–0.87 with abstract names, 0.65–0.74 with named features.
- The rule label stays at 0.43–0.59, a little above chance only at the top layers under named features (aligned 0.59 at layer 28).

**Steering** (12 held-out tasks; vectors fitted on the other 12).
- vₗ is the mean residual of predictions that follow the demonstrations minus that of predictions that follow the prior, on queries where the two disagree. It was fitted on 465 against 654 queries and averaged over the rule labels.
- It is only 1–2% of the residual norm, so after the planned α = 1, 2, 4 I added α = 8, 16, 32 in the same session (my decision; about $0.50).
- **AUROC changes by at most ±0.008 for α ≤ 16.** At α = 32 a few cells move by 0.02–0.04 in both directions, as the random direction of the same norm does.
- **Yet the margins move a lot:** −9.4 logits on average at layer 24, α = 32. Steering shifts every query together.
- **Why:** following the demonstrations means opposite things for the two labels.
  - On rule-label-0 conflict queries it means a lower margin (0.56 against 6.08).
  - On rule-label-1 queries it means a higher one (5.25 against 0.41).
  - The two halves cancel in the average, leaving a bias with no rule in it.

## Why it matters

- **It explains the RQ4 nulls mechanistically.** The demonstrated rule barely reaches the residual stream in a form that transfers across tasks, while the prior is there from layer 2. A selector changes which evidence is shown, but the computation that would use it hardly exists at k = 8.
- **It rules out the cheapest fix.** A single "follow the context" direction (Minder et al. 2025 find one in Llama-3.1-8B) does not exist here in a usable form, because following the demonstrations is label-specific.
- **It answers the plan's selection question** ("do counter_prior or SATA-MI demonstrations move the divergence point?"): no.

## What we did

- Built `src/inference/probes.py` (`ProbeRunner`, `ProbeGridRunner`), `src/evaluation/mechanisms.py` and `scripts/p6_probes.py`, with tests in `tests/test_probes.py`.
- `run_synth_grid.py` gained `--probe-dir`, `--keep-residuals`, `--steer`, `--condition` and `--task-ids`.
- Smoke-tested the gate on the Mac (4.5e-6), then ran `scripts/run_p6_pod.sh` on a fresh H200.
  - The previous pod's host had no free GPU, so the user approved a new pod and terminating the old one.
  - Stage A took 7.5 min, the vectors seconds, and the two steering passes 5.5 min each. Including setup and copying 875 MB of residuals back, the pod was up 28.5 min, about $2.50.
- The critical layer uses AUROC, not the spec's accuracy of 1[mₗ > 0], because Qwen's class bias would decide the accuracy (hiccups/15). Centred accuracy is reported alongside and agrees (random 23).

## Result

Notebook 06, section 5: logit-lens curves, the critical layer, probes and steering, each with a chart and a "Reading the chart" caption, plus a plain-English summary (5b).

## Evidence

- `results/v3/synthetic/h200/p6/A/` (`grid.parquet`, `lens.*.parquet`, `resid/`, `gate_p6.json`), `B/` (40 steering conditions), `logs/`, `probe_curves.parquet`.
- `models/p6_steer.npz`, `models/p6_steer.json` (counts and norms).
- `scripts/run_p6_pod.sh`, `scripts/p6_probes.py`.

## How to tell it

> We looked inside the model to see where it makes up its mind. For most of its depth, its read-out of the answer is no better than a coin flip. The decision forms only in the last few layers, and that is exactly where the feature names start to matter: with misleading names the answer turns the wrong way, with or without examples. The model's own guess can be read from its internal state almost immediately, but the rule shown in the examples barely appears. None of our selection strategies changed this. When we tried to push the model directly toward "follow the examples", its answers moved a lot but no more correctly, because following the examples means opposite things for the two labels.

- **Figure idea:** the logit-lens curves (notebook 06, section 5), flat for 20 layers and then splitting by naming.
- **Examiner question:** "Is the logit lens reliable for Qwen?"
  - The final layer matches the runner exactly (the gate), but earlier layers can be biased (Belrose et al. 2023).
  - The probes avoid that bias and tell the same story: the prior is early, the rule is weak.
