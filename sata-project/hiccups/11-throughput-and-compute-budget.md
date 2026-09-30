# 11 — The real compute cost, and the machine itself

**When:** 26–28 September 2026 · **Status:** measured, and the budget was rescaled on 28 September. The number of similarity seeds is still to be fixed before P4.

## What happened

The provisional compute budget of about 48 GPU-hours came from timings of the old MLX pipeline:
- prefill about 500 tokens/s;
- about 2 s per full prompt at k = 8;
- about 0.27 s per cached query.

P2 measured the PyTorch-on-MPS runner:

| | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
| --- | --- | --- |
| Prefill (tokens/s), benchmark runs | 361–633 (depends on how hot the machine is) | 290–326 |
| Per cached query (s) | 0.275–0.294 | 0.351–0.368 |
| P2 grid unit (20 queries + content-free), k = 8 / 16 / 32 | 8.3 / 10.3 / 14.4 s | about 14 s on average |

The first benchmark (633 tokens/s) ran on a cool machine; the sustained rates are lower.

## Infrastructure hiccups

- **Docker competed for memory.** Docker Desktop's virtual machine held about 24 GB. With a 15 GB model loaded as well, the Mac swapped (27 GB of swap) and unit times grew up to 20×. The user quit Docker; before long runs, memory is now checked first.
- **Long runs outlive the tool's 10-minute limit.** They run detached (`nohup` under `caffeinate`). zsh runs them niced, which does not matter for GPU-bound work.

## Result: P4-sized units

A P4 unit is 60 queries (3 environments × 20) plus the content-free query, measured on 28 September.

| | k = 8 | k = 16 | k = 32 |
| --- | --- | --- | --- |
| Qwen, query-agnostic strategy | 24.1 s | 25.6 s | 32.2 s |
| Llama, query-agnostic strategy | 23.6 s | 25.2 s | 30.5 s |

- **Other strategies:** a zero-shot unit takes about 23 s. A similarity unit takes about 155 s, because it builds one prompt per query (2.58 s per query in the smoke run).
- **Against the provisional budget:** that assumed about 18.5 s per unit, so units cost about 1.3× more.
- **The Qwen grid at k = 8** needs 12–18 GPU-hours. The range depends on whether similarity uses 1 seed or 3.
- **The total** rises from about 48 to about 65–71 GPU-hours, or 8–9 overnight runs (`docs/research_plan.md`, Compute budget).
- **Cut rather than add.** If the total no longer fits the timeline, the plan's cut order applies, rather than a second backend.

## Evidence

- `results/v3/synthetic/p2/benchmark_<model>.json`.
- `unit_seconds` in `p2_learnability.parquet`.

## How to tell it

One paragraph in the methods chapter, "Compute":
- the machine: M3 Max, 36 GB;
- one model in memory at a time;
- bf16, batch size 1;
- prefix caching and measured per-unit costs;
- the total GPU-hours actually used.
