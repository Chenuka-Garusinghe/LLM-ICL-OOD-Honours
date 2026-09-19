#!/usr/bin/env python3
"""Gate S1 -- pilot check for Stage 1's inference-layer fixes.

Backend-agnostic: runs on Apple Silicon via MLX (branch:
local-testing-w-mlx-models) or on an NVIDIA GPU via vLLM (branch:
appollo-cluster-v, H200) through the same MLXRunner/VLLMWorkerRunner
`chat_formatter()`/`batch_predict()` interface -- see
src/inference/mlx_runner.py and src/inference/llm_runner.py. `--backend
auto` (default) picks vLLM when `torch.cuda.is_available()`, else MLX, so
the same invocation works unmodified on either branch/machine.

Tests whether the chat template + honest task framing fixes
(REDESIGN_RATIONALE.md §5.1) actually cure the 96-99% zero-shot class-1
bias measured in v1, and whether random-8 clears a real-competence floor.

Pass criteria (hard gate): random-8 ID accuracy >= 0.60 (chance is 0.5;
XGBoost on the same pools reaches ~0.8).

Zero-shot class-1 rate is reported as a DIAGNOSTIC, not a hard gate.  With
semantically grounded "No"/"Yes" labels, zero-shot is dominated by the
model's pretraining prior (Wei et al. 2023) -- 70B instruct models show a
near-total "No" prior (class-1 rate ~ 0.0) that vanishes once demos are
provided (random-8 accuracy well above chance).  The prior confirms the
labels are semantically active; gating on it would conflate "the model has
a directional prior" with "the model can't do the task."

Zero-shot is intentionally reported RAW, not calibrated: contextual
calibration (Zhao et al. 2021) works by dividing out what the model outputs
on a content-free version of the SAME demos with the query blanked -- it
isolates the demos' + format's bias, independent of the query's specific
content. With no demos at all, "content-free zero-shot" collapses to a
fixed, task-independent placeholder input (verified empirically: identical
p0/p1 across different tasks, down to the exact logprob), which measures
something closer to "how does the model react to a garbled input" than a
meaningful prior -- dividing by it doesn't correct anything, it just adds a
constant, occasionally-opposite-signed bias (observed: pushed Llama's
zero-shot class-1 rate from a raw 0.812 to a "calibrated" 1.000, i.e. made
it worse). Calibration remains legitimate for random-8/label_diversity-8,
where real demos anchor the content-free query -- just not for zero-shot.

On fail: escalate to a larger model if one is available, or treat as a
signal that Stage 1's prompt fixes need more work before Stage 4.

Usage: python3 scripts/pilot_stage1.py [--n-tasks 8] [--n-queries 4] [--config configs/v2.yaml] [--backend auto|mlx|vllm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generator import generate_val_test_tasks  # noqa: E402
from src.data.serialisation import serialise_row  # noqa: E402
from src.inference.prompts import SYNTHETIC_TASK_DESCRIPTION, build_chat_messages  # noqa: E402
from src.selection.ordering import shuffle_order  # noqa: E402
from src.utils.config import load_config  # noqa: E402

LABEL_TOKENS = ("No", "Yes")
FEATURE_COLS_N = 10


def _detect_backend() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "vllm"
    except ImportError:
        pass
    return "mlx"


def _make_runner(backend: str, model_path: str, vllm_config):
    if backend == "mlx":
        from src.inference.mlx_runner import MLXRunner

        return MLXRunner(model_path)

    from src.inference.llm_runner import VLLMWorkerRunner

    return VLLMWorkerRunner(
        model_path,
        tensor_parallel=vllm_config.tensor_parallel,
        gpu_memory_utilisation=vllm_config.gpu_memory_utilisation,
        max_model_len=vllm_config.max_model_len,
        quantization=getattr(vllm_config, "quantization", None),
    )


def _row_features(X_row: np.ndarray) -> dict:
    return {f"feature_{i}": X_row[i] for i in range(FEATURE_COLS_N)}


def _build_prompt(formatter, demo_rows, demo_labels, query_features, order_seed) -> str:
    demo_ids = list(range(len(demo_rows)))
    ordered = shuffle_order(demo_ids, seed=order_seed)
    demo_lines = [serialise_row(_row_features(demo_rows[i]), label=LABEL_TOKENS[int(demo_labels[i])]) for i in ordered]
    query_line = serialise_row(query_features)
    messages = build_chat_messages(SYNTHETIC_TASK_DESCRIPTION, LABEL_TOKENS, demo_lines, query_line)
    return formatter.render(system=messages[0]["content"], user=messages[1]["content"])


def run_pilot(model_name: str, model_path: str, tasks: list, n_queries: int, pool_size: int, k: int, backend: str, vllm_config) -> dict:
    print(f"\n=== {model_name} ({model_path}) [{backend}] ===")
    runner = _make_runner(backend, model_path, vllm_config)
    try:
        formatter = runner.chat_formatter()

        zero_shot_raw_predictions = []
        random8_correct = []

        for t_idx, task in enumerate(tasks):
            X, y, meta = task.generate_environment("id", n_samples=pool_size + n_queries, seed=t_idx)
            pool_X, pool_y = X[:pool_size], y[:pool_size]
            query_X, query_y = X[pool_size:], y[pool_size:]

            for q_idx in range(n_queries):
                query_features = _row_features(query_X[q_idx])

                # --- zero-shot (no demos), raw only -- see module docstring for
                # why calibration doesn't apply here.
                prompt = _build_prompt(formatter, np.empty((0, FEATURE_COLS_N)), np.empty((0,), dtype=int), query_features, order_seed=0)
                [pred] = runner.batch_predict([prompt], LABEL_TOKENS)
                zero_shot_raw_predictions.append(pred.prediction)

                # --- random-8, ID accuracy ---
                demo_ids = list(np.random.default_rng(q_idx).choice(pool_size, size=k, replace=False))
                prompt = _build_prompt(formatter, pool_X[demo_ids], pool_y[demo_ids], query_features, order_seed=q_idx)
                [pred] = runner.batch_predict([prompt], LABEL_TOKENS)
                random8_correct.append(int(pred.prediction == LABEL_TOKENS[int(query_y[q_idx])]))
    finally:
        # vLLM (unlike MLX) holds GPU memory that must be explicitly released
        # via VLLMWorkerRunner's subprocess teardown before the next model in
        # the loop can load -- see llm_runner.py::VLLMRunner.shutdown's
        # docstring for why in-process teardown alone isn't reliable.
        if hasattr(runner, "shutdown"):
            runner.shutdown()

    random8_accuracy = float(np.mean(random8_correct))
    raw_class1_rate = float(np.mean([p == "Yes" for p in zero_shot_raw_predictions]))

    print(f"  zero-shot class-1 rate (raw): {raw_class1_rate:.3f}")
    print(f"  random-8 ID accuracy: {random8_accuracy:.3f}")

    return {
        "model": model_name,
        "raw_class1_rate": raw_class1_rate,
        "random8_accuracy": random8_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=8)
    parser.add_argument("--n-queries", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--backend", choices=["auto", "mlx", "vllm"], default="auto")
    parser.add_argument(
        "--size", choices=["8b", "70b"], default="8b",
        help="8b: config.base_llms/config.mlx.models (bf16). "
             "70b: config.base_llms_70b via vLLM with FP8 (single-GPU H200 memory budget "
             "-- see configs/v2.yaml's vllm_70b block). MLX backend has no 70b path.",
    )
    args = parser.parse_args()

    import os

    os.environ["SATA_CONFIG"] = args.config
    config = load_config()

    backend = _detect_backend() if args.backend == "auto" else args.backend
    print(f"Backend: {backend}, size: {args.size}")

    if args.size == "70b":
        if backend != "vllm":
            raise SystemExit("--size 70b requires --backend vllm (or auto on a CUDA machine) -- no MLX 70b path")
        model_list = config.base_llms_70b
        vllm_config = config.vllm_70b
    else:
        model_list = config.mlx.models if backend == "mlx" else config.base_llms
        vllm_config = config.vllm

    _, test_tasks = generate_val_test_tasks(config.generator)
    tasks = test_tasks[: args.n_tasks]

    results = []
    for model_cfg in model_list:
        result = run_pilot(model_cfg.name, model_cfg.path, tasks, args.n_queries, args.pool_size, args.k, backend, vllm_config)
        results.append(result)

    print("\n=== Gate S1 summary ===")
    overall_pass = True
    for r in results:
        random8_pass = r["random8_accuracy"] >= 0.60
        class1_balanced = 0.35 <= r["raw_class1_rate"] <= 0.65
        overall_pass &= random8_pass
        print(
            f"{r['model']}: random-8={r['random8_accuracy']:.3f} ({'PASS' if random8_pass else 'FAIL'} >= 0.60), "
            f"raw zero-shot class-1 rate={r['raw_class1_rate']:.3f} "
            f"({'balanced' if class1_balanced else 'biased -- semantic prior, see docstring'} "
            f"[diagnostic, not gated])"
        )
    print(f"\nGate S1: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
