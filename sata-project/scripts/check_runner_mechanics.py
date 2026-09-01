"""Fast CPU check of HFRunner's plumbing, using a tiny random-init model.

Complements scripts/check_runner_parity.py: that one needs a GPU box with a
working vLLM and checks whether the *numbers* match the old implementation;
this one runs anywhere in seconds and checks the *mechanics* -- left padding,
position ids, label-token resolution, batching. Model quality is irrelevant
here (a random-init model predicts noise); what's under test is the plumbing.

Usage (from the project root):
    python scripts/check_runner_mechanics.py

Exits non-zero on failure. Worth re-running after any edit to
src/inference/llm_runner.py::HFRunner.

Why this exists: the padding/position-id bug it guards against is silent. Left
padding fixes attention via attention_mask, but a plain forward pass still
derives position ids from the raw sequence index, so a prompt batched behind N
pad tokens gets encoded at positions N, N+1, ... instead of 0, 1, ... . Nothing
raises; the logprobs are just quietly wrong, and differ depending on which
other prompts happened to share the batch. The solo-vs-batched invariant below
catches exactly that -- it failed with a delta of 6.4e-02 before position ids
were derived from the mask, and passes at ~5e-07 (float32 noise) after.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.inference.llm_runner import HFRunner  # noqa: E402

# Tiny random-init model: a few MB, CPU-only, no HF auth needed.
MODEL = "hf-internal-testing/tiny-random-gpt2"

# Deliberately unequal lengths -- that is what forces padding, and padding is
# what the invariant below is testing.
PROMPTS = [
    "A: 1; B: 2 -> 1\nA: 3; B: 4 ->",
    "A: 1 ->",
    "A: 1; B: 2; C: 3; D: 4; E: 5; F: 6 -> 0\nA: 9; B: 9; C: 9; D: 9; E: 9; F: 9 ->",
    "A: 7; B: 8 ->",
]

# Tolerance for "same computation, different batch shape". Pure float32
# reduction-order noise is ~1e-7; the padding bug this guards against showed up
# at ~1e-2, so there are five orders of magnitude between pass and fail.
PADDING_ATOL = 1e-3


def main() -> int:
    failures: list[str] = []

    print(f"loading {MODEL} (CPU)")
    runner = HFRunner(MODEL, max_model_len=512, batch_size=4)

    # Tokeniser must be configured for decoder-only batching.
    if runner.tokenizer.padding_side != "left":
        failures.append(f"padding_side is {runner.tokenizer.padding_side!r}, expected 'left'")
    if runner.tokenizer.truncation_side != "left":
        failures.append(f"truncation_side is {runner.tokenizer.truncation_side!r}, expected 'left'")
    if runner.tokenizer.pad_token_id is None:
        failures.append("pad_token_id is None; batching cannot pad")
    print(f"  padding_side={runner.tokenizer.padding_side} "
          f"truncation_side={runner.tokenizer.truncation_side} "
          f"pad_token={runner.tokenizer.pad_token!r}")

    # Label ids resolve with a leading space, because serialise_row ends the
    # query line with "->" and writes demos as "... -> 1.0".
    for labels in [("0", "1"), ("0.0", "1.0")]:
        ids = runner._resolve_label_token_ids(labels)
        decoded = [runner.tokenizer.decode([i]) for i in ids]
        print(f"  labels {labels} -> ids {ids} -> {decoded}")
        if not all(d.startswith(" ") for d in decoded):
            failures.append(f"labels {labels} resolved to {decoded}, expected leading spaces")
        if ids[0] == ids[1]:
            failures.append(f"labels {labels} resolved to identical ids {ids}")

    # Indistinguishable labels must raise loudly, not silently mis-score.
    try:
        runner._resolve_label_token_ids(("cat", "cat"))
        failures.append("identical label strings did not raise")
    except ValueError:
        print("  identical labels correctly raise ValueError")

    results = runner.batch_predict(PROMPTS, ("0", "1"))
    if len(results) != len(PROMPTS):
        failures.append(f"got {len(results)} results for {len(PROMPTS)} prompts")
    for i, p in enumerate(results):
        if p.prediction not in ("0", "1"):
            failures.append(f"[{i}] prediction {p.prediction!r} not a label token")
        if not 0.0 <= p.confidence <= 1.0:
            failures.append(f"[{i}] confidence {p.confidence} outside [0, 1]")
        if abs((p.p0 + p.p1) - 1.0) > 1e-6:
            failures.append(f"[{i}] p0+p1 = {p.p0 + p.p1}, expected 1.0")
        # Label ids are always indexable, so the -100 sentinel should never appear.
        if p.logprob_0 <= -50 or p.logprob_1 <= -50:
            failures.append(f"[{i}] logprob floor hit ({p.logprob_0}, {p.logprob_1})")

    # THE key invariant: a prompt scored alone must match the same prompt scored
    # inside a mixed-length batch. See the module docstring.
    solo = runner.batch_predict([PROMPTS[1]], ("0", "1"))[0]
    delta = abs(solo.logprob_0 - results[1].logprob_0)
    print(f"  padding invariant: solo={solo.logprob_0:.7f} "
          f"batched={results[1].logprob_0:.7f} delta={delta:.2e}")
    if delta > PADDING_ATOL:
        failures.append(
            f"solo vs batched logprob differ by {delta:.2e} (> {PADDING_ATOL}) -- "
            "left padding and/or position ids are wrong"
        )

    generated = runner.generate_text([PROMPTS[0]], max_tokens=5)
    if not isinstance(generated, list) or len(generated) != 1:
        failures.append(f"generate_text returned {generated!r}, expected a 1-element list")
    print(f"  generate_text -> {generated!r} (content meaningless for a random-init model)")

    if failures:
        print(f"\nFAIL ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: HFRunner mechanics are correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
