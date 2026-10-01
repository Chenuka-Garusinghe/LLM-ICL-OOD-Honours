# 17 — P4 moved from the Mac to a cloud H200, and ran on one device in 42 minutes

**When:** 1 October 2026 · **Status:** done; the Pod is stopped (its disk is kept until the user decides)

## What happened

The P4 chain on the Mac (Apple Metal, 30 September) was projected at about 30 hours. It had finished the abstract-name grid when the session that ran it ended, partway through the named grid.
- **The user asked:** move the compute to Runpod and get the results within about 40 minutes.
- **The setup:** the Runpod plugin and MCP server, signed in with OAuth, so no API key was stored. A Pod was created only after the user confirmed the GPU and price.

## How we found the right setup

**One copy of the model cannot use a big GPU.** Scoring one query at a time, each forward pass is mostly CPU-side launch overhead, so a faster GPU barely shortens it. The whole P4 workload is about 255,000 passes, which is 1.5–2.5 hours on any single GPU with one process.
- **The fix was parallel workers on one GPU.** One H200 (141 GB) held 8 copies of Qwen, each a worker on its own shard of units (`--shard i/N`, writing separate files, merged afterwards).

**Measured throughput:**

| Setup | Forward passes per second, all workers |
| --- | --- |
| 8 processes sharing the GPU by time-slicing | about 84 |
| 8 processes as NVIDIA MPS clients (concurrent kernels) | about 160 |
| The Mac, for comparison | about 3.4 |

**Timeline** (UTC, Pod created at 00:46:55):
1. **Setup, 5 minutes.** Pinned venv (torch 2.14.0+cu130, which needs a CUDA 13 host), both models downloaded to RAM disk, 94 tests passed.
2. **Stage A, 6 minutes.** Pool priors for both models, the cache-agreement benchmark and Qwen-base's P2 checks.
3. **Stage B, 27 minutes.** The full Qwen-Instruct grid, RQ3 on 3 seeds and Qwen-base's reduced grid.
4. **Stage C, 3 minutes.** counter_prior_matched (hiccups/16).
5. **Total:** 42 minutes, $3.30 of GPU time at $4.59/hr.

## Why it matters

- **One device for all of P4.** Every P4 comparison is now within one device, as the plan requires. The abstract-name rerun shares its prompts with the Mac run, so it doubles as a cross-device replication.
- **Instrument checks on the new device:**
  - **Cache rule:** passes on the H200. Mean per-task AUROC is 0.5783 cached against 0.5825 full; the difference of 0.0042 is under the 0.005 rule (the Mac's was 0.0025).
  - **Qwen-base:** answers in the expected format (label mass 1.0) and passes the learnability gate at k = 8: Δ 0.103 [0.023, 0.186].
- **Mac against H200, on 718 units both ran:**
  - prompts and demonstration rows identical (equal prompt hashes);
  - margin correlation 0.996, median |difference| 0.22 logits;
  - mean cell AUROC 0.5711 against 0.5704;
  - raw predictions agree on 98% of queries, but calibrated predictions on only 87%. Hardware noise flips predictions near the calibration threshold. This is a further reason why AUROC, not calibrated accuracy, is the test statistic (hiccups/15).

## Evidence

- `results/v3/synthetic/h200/` (all P4 results, logs in `p4/logs/`, the H200 benchmark in `p2/`).
- `scripts/run_p4_pod.sh` (the stages as run), `scripts/merge_shards.py`, `src/utils/shard.py`.
- Notebook 05, section 8 (the cross-device check).

## How to tell it

> Scoring one query at a time leaves a modern GPU mostly idle, so we ran eight copies of the 7B model side by side on one H200, sharing it through NVIDIA's Multi-Process Service. All of P4 — about 255,000 forward passes — finished in 42 minutes for about four US dollars, on one device. Re-running units already scored on a laptop GPU reproduced them: the same prompts, margins correlated at 0.996 and mean AUROC within 0.001. Thresholded, calibrated predictions agreed on only 87% of queries, which is one more reason the analysis rests on AUROC.

- **Examiner question:** "Do your results depend on the hardware?"
  - The ranking-based results do not: the mean AUROC differs by 0.0007 between devices.
  - Calibrated accuracy does move, which is why it is reported but not tested.
