# 03 — The GPU cluster was lost, 72B was too slow, and one backend won

**When:** 19–27 September 2026 · **Status:** resolved

## What happened

1. **The cluster was lost.** The Apollo cluster (CUDA, H200 GPUs) became unavailable on 19 September 2026, and university and NCI GPU access proved unreliable. Every run since then happens on one Apple M3 Max laptop with 36 GB of unified memory.
2. **72B was too slow.** Running Qwen2.5-72B on the laptop through AirLLM worked, after an attention-bias and rope_theta patch, but took about 44 s per query, roughly 13 hours per grid.
3. **Two backends.** The plan first kept MLX as a speed fallback next to PyTorch.

## Why it matters

- **Scale:** at 13 hours per grid, the planned experiments would not fit an Honours timeline.
- **Two backends:** results from two implementations of tokenisation, caching and scoring would not be directly comparable.

## What we did

- **24 September, the user's decision:** stay with 7–8B models (Qwen2.5-7B-Instruct as the primary, Qwen2.5-7B base, Llama-3.1-8B-Instruct). The questions were reframed around whether demonstration selection can help *small* models override their priors, which Wei et al. (2023) describe as emerging with scale.
- **27 September, the user's decision:** PyTorch only (Hugging Face transformers on MPS, bf16), dropping the MLX fallback. One runner is tested once and is portable to CUDA. If PyTorch is slow, the plan's cut order applies, rather than adding a second backend.
- **Kept for reference:** the AirLLM runner, on branch `synthetic-exp-airllm`.

## Evidence

- `docs/research_plan.md` (models, compute budget, cut order).
- `src/inference/hf_runner.py` (the single runner).
- Branch `synthetic-exp-airllm`.

## How to tell it

> Losing GPU access forced a choice between a few very slow runs of a large model and complete experiments on 7–8B models. We chose the latter and turned the constraint into the question: previous work suggests that overriding semantic priors in context emerges with scale, so we ask whether demonstration selection can do for small models what scale does for large ones. To keep every result comparable, all inference runs through one PyTorch implementation.

- **Figure idea:** a compute table (72B: about 44 s/query; 7B: about 0.3 s per cached query).
- **Examiner question:** "Would the results hold at 70B?"
  That is out of scope by design. The claims are about 7–8B models, and the scale-dependence literature is the reason they are interesting.
