#!/usr/bin/env python3
"""P2 answer-format check: does the model answer with the label tokens?

Scoring compares the log-probabilities of "0" and "1" at the answer position
(generator_spec.pdf, scoring). That comparison is only meaningful if the model
puts most of its next-token probability on those two tokens. For a few pilot
prompts (zero-shot and k=8, label-balanced gold demonstrations) this prints
the label-token mass, the 10 most likely next tokens and a short greedy
continuation.

Writes results/v3/synthetic/p2/answer_format_<model>.json.

Usage: python scripts/p2_answer_format.py [--model llama-3.1-8b-instruct] [--task pilot_0000] [--queries 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402  (first: sets thread/MPS env vars)
import torch  # noqa: E402

from src.data.suites import load_suite, suite_dir  # noqa: E402
from src.experiments.synth_grid import GridRunner, Unit  # noqa: E402
from src.inference.hf_runner import runner_from_config  # noqa: E402


@torch.inference_mode()
def next_token_report(runner, prompt: str, top: int = 10) -> dict:
    out = runner._forward(runner.encode(prompt))
    probs = torch.softmax(runner.model.get_output_embeddings()(out.last_hidden_state[0, -1]).float(), dim=-1)
    p, idx = probs.topk(top)
    i0, i1 = runner.label_ids()
    return {
        "label_mass": float(probs[i0] + probs[i1]), "p0": float(probs[i0]), "p1": float(probs[i1]),
        "top_tokens": [[runner.tokenizer.decode([int(i)]), round(float(q), 4)] for q, i in zip(p, idx)],
    }


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="llama-3.1-8b-instruct")
    parser.add_argument("--task", default="pilot_0000")
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()

    models = {m.name: m for m in config.models}
    runner = runner_from_config(models[args.model], config.inference)
    manifest = load_suite("pilot", config)
    grid = GridRunner(runner, args.model, suite_dir("pilot", config), manifest,
                      base_seed=config.selection.base_seed, envs=["id"])
    report = {"model": args.model, "task": args.task, "prompts": []}
    for strategy, k in (("zero_shot", 0), ("label_diversity", 8)):
        group = grid.prompt_groups(Unit(args.model, "abstract", args.task, strategy, "gold", k, 0))[0]
        for prompt in group["fulls"][: args.queries]:
            entry = {"strategy": strategy, "k": k, **next_token_report(runner, prompt),
                     "greedy": runner.generate_text([prompt], max_new_tokens=args.max_new_tokens)[0],
                     "prompt_tail": prompt[-160:]}
            report["prompts"].append(entry)
            print(json.dumps(entry, ensure_ascii=False))
    out = resolve_path(config.paths.results) / "p2" / f"answer_format_{args.model}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
