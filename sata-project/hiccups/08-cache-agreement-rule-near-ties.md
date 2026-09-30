# 08 — What should "cached equals full" mean?

**When:** 28 September 2026, P2 · **Status:** resolved: the user adopted the AUROC rule on 28 September, and Qwen passes it

## What happened

The plan's P2 check says cached and full forward passes must agree on at least 99% of predictions.
After the fp32 fix ([07](07-bf16-output-rounding.md)), Qwen passed but Llama did not.

## How we found it

| | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
| --- | --- | --- |
| Prediction agreement, first run / rerun | 99.2% / 99.6% | 97.6% / 98.0% |
| Largest \|margin\| at a flipped prediction (rerun) | 0.017 | 0.022 |
| Median cached − full margin difference (rerun) | 0.042 | 0.010 |
| Median \|margin\| (rerun) | 4.72 | 0.22 |
| Share of \|margin\| below 0.25 (rerun) | 2% | 58% |

A prediction can only flip when the margin is smaller than the numerical difference between the two passes.
So the 99% rule mostly measures how many of a model's margins sit near zero, not whether the cache is correct.
The cache itself is exact:
- by construction: the prefix-caching proposition in the spec;
- in the tests: it matches full passes within 10⁻⁴ on the tiny model, and within 5 × 10⁻⁷ in fp32 on CPU;
- no suffix ever needed the full-pass fallback.

## The proposed rule

Proposed before the AUROC comparison was run:

> The mean per-task AUROC from the cache must be within 0.005 of the full-pass value, a tenth of the learnability gate's threshold. Prediction agreement is still reported.

Results on the rerun:
- **Qwen:** 0.5867 against 0.5842, a difference of 0.0025, so it **passes**. It also passes the old rule.
- **Llama:** 0.465 against 0.478, a difference of 0.013, so it **fails**. The per-task AUROC moves by 0.043 on average.

Llama fails because its margins barely vary between queries: the within-task sd is 0.053, against a median rounding difference of 0.010. Neither path gives Llama a reliable ranking in the current format, which points to [10](10-llama-answer-format.md).

## Decision

The user adopted the AUROC rule on 28 September.
- It is implemented as `cache_agreement(...)["passes"]` (`CACHE_AUROC_TOLERANCE = 0.005` in `src/evaluation/gates.py`).
- The benchmark JSONs record `passes_agreement` under the new rule and name it in `agreement_rule`.
- Qwen passes (0.0025). Llama fails, but Llama is paused anyway because of its answer format ([10](10-llama-answer-format.md)).

## Why it matters

- The grid uses the cached path, so the thesis must show that caching does not change any reported metric.
- A rule that fails for reasons unrelated to the cache is a poor gate.

## Evidence

- `results/v3/synthetic/p2/benchmark_<model>.json`.
- `benchmark_<model>_agreement.parquet` (per-query cached and full scores).
- `src/evaluation/gates.py::cache_agreement`, with a test in `tests/test_gates.py`.
- First-run benchmarks: `results/v3/synthetic/archive/fixed_f8/p2/`.

## How to tell it

> All grid results use prefix caching, which is exact in exact arithmetic. In bf16, the cached and full passes differ slightly, so predictions on near-ties can flip; a prediction-agreement rule therefore depends on how confident the model is. We instead require that caching changes the mean AUROC by less than a tenth of the smallest effect the thesis tests. For Qwen the difference is 0.0025.

- **Figure idea:** scatter of the full-pass margin against the cached − full difference, with flips highlighted, per model (NB02).
